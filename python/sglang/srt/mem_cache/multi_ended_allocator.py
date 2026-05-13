"""
MultiEndedAllocator: one per sub-pool over a shared byte buffer.

Invariants between allocator calls (enforced by eager compaction):
  * This allocator's allocated slots form a CONTIGUOUS range in its own slot-
    index space: grow-up -> slots [min_slot_index, watermark); grow-down ->
    slots (watermark, max_slots - 1]. Slots [0, min_slot_index) are reserved
    across every pool.
  * The peer's allocated slots likewise form a contiguous range.
  * The union of each allocator's consumed BYTE range is disjoint.
  * No hole bookkeeping: `free` eagerly relocates boundary slots into freed
    holes via `SharedMemoryPool`-backed `move_kv_cache` / `copy_from` calls,
    keeping the allocated range hole-free.

Concurrency: the scheduler runs alloc/free serially; no mutex is taken here.
"""

from __future__ import annotations

import logging
import os
import weakref
from typing import Dict, List, Optional, Tuple

import torch

from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.relocation_log import Relocator
from sglang.srt.mem_cache.shared_memory_pool import SharedMemoryPool
from sglang.srt.mem_cache.swa_memory_pool import SWATokenToKVPoolAllocator

logger = logging.getLogger(__name__)


def _flatten_inverse_history_pairs(
    history_slice: List[Tuple[torch.Tensor, torch.Tensor]],
) -> List[Tuple[int, int]]:
    """Materialize a slice of an allocator's `_inverse_history` into a flat
    list of `(src, dst)` Python-int pairs.

    Each entry of `_inverse_history` is one batched relocation
    ``(src_tensor, dst_tensor)`` produced by a single
    ``MultiEndedAllocator._apply_relocations`` call. A composite
    allocator (SWA or Mamba) snapshots ``len(inner._inverse_history)``
    before invoking inner ``free``, then reads the suffix to learn which
    pairs were just emitted (used by
    ``SharedSWATokenToKVPoolAllocator._apply_new_relocations_to_mapping``
    to update ``full_to_swa_index_mapping``).
    """
    out: List[Tuple[int, int]] = []
    for src_tensor, dst_tensor in history_slice:
        # Each tensor is 1-D; lengths match.
        out.extend(
            zip(src_tensor.tolist(), dst_tensor.tolist())
        )
    return out


class MultiEndedAllocator(BaseTokenToKVPoolAllocator):
    """Allocator for one sub-pool over a `SharedMemoryPool`."""

    def __init__(
        self,
        *,
        kvcache,
        shared_buffer: SharedMemoryPool,
        sub_pool_name: str,
        relocation_log: Relocator,
        device: str,
        need_sort: bool = False,
    ):
        spec = shared_buffer.spec(sub_pool_name)
        max_slots = shared_buffer.max_slots(sub_pool_name)
        # `dtype` on BaseTokenToKVPoolAllocator is informational only. MHA specs
        # have `store_dtype`; Mamba specs have `conv_dtype` and `temporal_dtype`.
        spec_dtype = getattr(
            spec, "store_dtype", getattr(spec, "conv_dtype", None)
        )
        super().__init__(
            size=max_slots,
            page_size=1,
            dtype=spec_dtype,
            device=device,
            kvcache=kvcache,
            need_sort=need_sort,
        )
        self.shared_buffer = shared_buffer
        self.sub_pool_name = sub_pool_name
        self.spec = spec
        self.max_slots = max_slots
        self.grow_direction = spec.grow_direction
        self.entry_bytes = spec.entry_bytes()
        self.relocation_log = relocation_log
        # Slots [0, min_slot_index) are RESERVED on every pool to guarantee
        # real data starts at bytes ≥ entry_max.
        # Slot 0 is the padding dummy sink; slots 1..min_slot_index-1 are
        # simply unused.
        self.min_slot_index = shared_buffer.min_slot_index(sub_pool_name)

        self._peer: Optional["MultiEndedAllocator"] = None

        # Inverse history of relocations: each entry is one
        # (src_tensor, dst_tensor) batch produced by `_apply_relocations`.
        # Two consumers:
        #   * spec-decode rollback (replays inverse moves);
        #   * composite-allocator post-free hooks that need to know which
        #     pairs were just emitted (e.g.,
        #     `SharedSWATokenToKVPoolAllocator.
        #     _apply_new_relocations_to_mapping` updating
        #     `full_to_swa_index_mapping`).
        # The composite is responsible for calling
        # `clear_inverse_history` after consuming, so the list remains
        # bounded across long runs.
        self._inverse_history: List[Tuple[torch.Tensor, torch.Tensor]] = []

        self.clear()

        logger.info(
            "[shared-pool] MultiEndedAllocator(%r) ready: grow=%s, "
            "max_slots=%d, min_slot_index=%d, entry_bytes=%d, "
            "initial_watermark=%d, allocatable=%d",
            self.sub_pool_name,
            self.grow_direction,
            self.max_slots,
            self.min_slot_index,
            self.entry_bytes,
            self.watermark,
            self.max_slots - self.min_slot_index,
        )

    # -- peer binding --

    def bind_peer(self, peer: "MultiEndedAllocator") -> None:
        self._peer = peer

    @property
    def peer(self) -> Optional["MultiEndedAllocator"]:
        return self._peer

    # -- state --

    def clear(self) -> None:
        # Slots [0, min_slot_index) are reserved. Grow-up allocator starts
        # handing out slots at min_slot_index; grow-down stops at
        # min_slot_index (never hands out lower indices).
        if self.grow_direction == "up":
            self.watermark = self.min_slot_index
        else:
            # Next-to-allocate = max_slots - 1; allocator walks down to
            # min_slot_index (inclusive).
            self.watermark = self.max_slots - 1
        self.is_not_in_free_group = True
        self.free_group: List[torch.Tensor] = []

    def backup_state(self):
        # Capture the watermark and the current `_inverse_history` length.
        # SGLang's spec-decode pattern only allocates inside a backup
        # window (no free), so under correct usage `_inverse_history`
        # does not grow during the window — `restore_state` then just
        # restores the watermark and there are no inverse moves to
        # replay.
        return (self.watermark, len(self._inverse_history))

    def restore_state(self, state):
        watermark, n_inverse = state
        self.watermark = watermark
        # If the history grew during the backup window, a free() ran
        # there. Eager compaction overwrites the freed (`dst`) slot's
        # bytes with the boundary slot's data, so a simple dst->src
        # copy cannot recover the original `dst` bytes — the rollback
        # is fundamentally lossy. Warn loudly so the unsupported
        # scheduling pattern is visible.
        new_entries = self._inverse_history[n_inverse:]
        if new_entries:
            logger.warning(
                "MultiEndedAllocator.restore_state: %d relocation(s) "
                "were recorded inside a backup window (sub_pool=%s). "
                "Eager compaction is not fully reversible — the data "
                "at the freed (dst) slots was overwritten and cannot "
                "be restored by a dst->src copy. This indicates a "
                "free() inside a backup window, which SGLang's spec "
                "decoding path does not currently produce.",
                len(new_entries),
                self.sub_pool_name,
            )
        # Truncate so the return reflects only entries from this window.
        del self._inverse_history[n_inverse:]
        return new_entries

    def clear_inverse_history(self) -> None:
        """Drop accumulated inverse-history entries.

        Called by composite allocators after consuming the relevant
        suffix in their post-free hooks (e.g., SWA mapping update).
        Bounded growth is essential because each entry holds tensor
        references that the GC cannot reclaim while in the list.
        """
        self._inverse_history.clear()

    # -- size reporting --

    def allocated_count(self) -> int:
        if self.grow_direction == "up":
            return max(0, self.watermark - self.min_slot_index)
        else:
            return max(0, self.max_slots - 1 - self.watermark)

    def is_slot_allocated(self, slot: int) -> bool:
        """Return True iff `slot` is currently in this sub-pool's allocated
        range. Used by callers (e.g., MambaRadixCache.cache_finished_req)
        to validate a Req-held slot id before laundering it into a tree
        node — a stale id (sentinel `-1`, or a positive id whose watermark
        has since moved past) would otherwise survive into the tree and
        crash on a later evict.

        For `grow_direction='up'`:  allocated == [min_slot_index, watermark)
        For `grow_direction='down'`: allocated == (watermark, max_slots)
        """
        if slot < self.min_slot_index or slot >= self.max_slots:
            return False
        if self.grow_direction == "up":
            return slot < self.watermark
        else:
            return slot > self.watermark

    def allocator_state_str(self) -> str:
        """One-line state summary used in diagnostic warnings."""
        return (
            f"sub_pool={self.sub_pool_name!r}, "
            f"grow_direction={self.grow_direction}, "
            f"min_slot_index={self.min_slot_index}, "
            f"max_slots={self.max_slots}, "
            f"watermark={self.watermark}, "
            f"allocated_count={self.allocated_count()}"
        )

    def _byte_high_frontier(self) -> int:
        """Byte offset one past the highest byte this allocator's REAL data has
        claimed.

        All sub-pool anchors are 0 under the `min_slot_index` scheme (see
        design doc §5.6), so byte offsets are simply `slot_index * entry_bytes`.

        - Grow-up: end of last allocated slot, i.e. `watermark * entry_bytes`.
          When no slots are allocated this equals `min_slot_index * entry_bytes`
          (the low-address floor of the pool's real-data region).
        - Grow-down: end of this sub-pool's view, i.e. `max_slots * entry_bytes`.
        """
        if self.grow_direction == "up":
            return self.watermark * self.entry_bytes
        return self.max_slots * self.entry_bytes

    def _byte_low_frontier(self) -> int:
        """Byte offset of the lowest byte this allocator's REAL data can claim.

        - Grow-up: the real-data floor — `min_slot_index * entry_bytes`. The
          reserved slots `[0, min_slot_index)` below this are NOT part of this
          allocator's used byte range (their bytes are overlapped with peer
          pools' slot-0 dummy writes).
        - Grow-down: lowest byte currently occupied by an allocated slot,
          i.e. `(watermark + 1) * entry_bytes`.
        """
        if self.grow_direction == "up":
            return self.min_slot_index * self.entry_bytes
        return (self.watermark + 1) * self.entry_bytes

    def available_size(self) -> int:
        """Slots this allocator can still allocate before colliding with peer."""
        if self.grow_direction == "up":
            my_high = self._byte_high_frontier()
            peer_low = (
                self._peer._byte_low_frontier()
                if self._peer is not None
                else self.shared_buffer.total_bytes
            )
            gap_bytes = max(0, peer_low - my_high)
        else:
            my_low = self._byte_low_frontier()
            peer_high = (
                self._peer._byte_high_frontier()
                if self._peer is not None
                else 0
            )
            gap_bytes = max(0, my_low - peer_high)
        slots_by_bytes = gap_bytes // self.entry_bytes
        # Clamp by this sub-pool's own slot-index headroom. Allocatable slots
        # are those in [min_slot_index, max_slots), minus already-allocated.
        slots_by_index_space = (
            self.max_slots - self.min_slot_index - self.allocated_count()
        )
        return min(slots_by_bytes, slots_by_index_space)

    # -- alloc --

    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        if need_size <= 0:
            return torch.empty(0, dtype=torch.int64, device=self.device)
        if need_size > self.available_size():
            return None
        if self.grow_direction == "up":
            start = self.watermark
            select = torch.arange(
                start,
                start + need_size,
                dtype=torch.int64,
                device=self.device,
            )
            self.watermark += need_size
        else:
            end = self.watermark  # last slot in this batch (highest id)
            start = end - need_size + 1
            if start < self.min_slot_index:
                return None
            select = torch.arange(
                start, end + 1, dtype=torch.int64, device=self.device
            )
            self.watermark -= need_size
        return select

    # -- free with eager compaction --

    def free(self, free_index: torch.Tensor) -> None:
        if free_index is None or free_index.numel() == 0:
            return
        if not self.is_not_in_free_group:
            self.free_group.append(free_index)
            return
        self._free_eager(free_index)

    def _free_eager(self, free_index: torch.Tensor) -> None:
        # Work on CPU indices for clarity. Slot counts per `free` call are
        # typically small (<1000), so CPU-side sorting is cheap.
        free_cpu = free_index.detach().to(torch.int64).cpu().tolist()
        if not free_cpu:
            return
        # No `resolve_to_current` chain-chasing needed: under the
        # immediate-apply design, by the time `free()` returns, every
        # bound holder (`req_to_token`, `TreeNode.value`, aux/py_attr)
        # has already been rewritten to the post-compaction id. A caller
        # holding a slot id via a *bound* path (e.g., `node.value[i]`)
        # automatically reads the current id on next access; only callers
        # capturing pre-compaction Python clones of slot ids need to
        # avoid that pattern. If a stale-slot assertion fires below, the
        # diagnostic dump points at the offending caller — fix the
        # caller, do not reintroduce chain-chasing.
        # Assert all indices are within our allocated range. If this fires,
        # we dump rich diagnostic context (binder state for each stale slot,
        # call-site identification, recent watermark history) before raising
        # so we can root-cause the source. The previous defensive-filter
        # implementation was reverted because it masked the underlying bug.
        alloc_low, alloc_high = self._allocated_range()
        stale = [idx for idx in free_cpu if not (alloc_low <= idx < alloc_high)]
        if stale:
            self._raise_stale_slot_assertion(
                stale_slots=stale,
                free_cpu=free_cpu,
                alloc_low=alloc_low,
                alloc_high=alloc_high,
            )
        # Defensive binder cleanup on free.
        #
        # The eager-compaction free path overwrites the freed slot's bytes
        # with the boundary slot's data, so any reverse-mapping entry still
        # pointing at the freed slot is stale. Owners are supposed to unbind
        # before freeing (tree-side via `_unbind_*_value`; req-side via the
        # row-aware unbinds; mamba-side via `transfer_mamba_to_radix` /
        # `free_mamba_cache`). `req_position` in particular is not unbound
        # by `_free_dedup_slots`, so a freed slot can keep a `(col, {row})`
        # entry that the next allocation of the slot would collide with
        # (`bind_req_position` multi-col assert). `clear_slot` pops every
        # binder entry for the freed slot so that can't happen.
        backtrack = self.relocation_log.backtracks.get(self.sub_pool_name)
        if backtrack is not None:
            for s in free_cpu:
                if alloc_low <= s < alloc_high:
                    backtrack.clear_slot(s)

        # Unique + sort descending for grow-up (walk from boundary inward),
        # ascending for grow-down.
        uniq = sorted(set(free_cpu), reverse=(self.grow_direction == "up"))

        # Compacting pointer `b` starts at the boundary and moves inward.
        src_list: List[int] = []
        dst_list: List[int] = []
        if self.grow_direction == "up":
            b = self.watermark - 1  # last allocated slot
            for s in uniq:  # descending
                if s == b:
                    # Boundary slot freed; just shrink.
                    b -= 1
                    self.watermark -= 1
                elif s < b:
                    # Relocate slot `b` into `s`.
                    src_list.append(b)
                    dst_list.append(s)
                    b -= 1
                    self.watermark -= 1
                else:
                    # s > b shouldn't happen given contiguous invariant and sort
                    raise AssertionError(
                        f"free: slot {s} > boundary {b}; corrupt allocator state"
                    )
        else:
            b = self.watermark + 1  # last (lowest-idx) allocated slot in grow-down
            for s in uniq:  # ascending
                if s == b:
                    b += 1
                    self.watermark += 1
                elif s > b:
                    src_list.append(b)
                    dst_list.append(s)
                    b += 1
                    self.watermark += 1
                else:
                    raise AssertionError(
                        f"free: slot {s} < boundary {b}; corrupt allocator state"
                    )

        if src_list:
            self._apply_relocations(src_list, dst_list)

    def _apply_relocations(
        self, src_list: List[int], dst_list: List[int]
    ) -> None:
        """Perform the actual KV-buffer row copies and immediately rewrite
        every reverse-mapping reference holder. After this returns,
        `req_to_token`, `TreeNode.value`/`mamba_value`, aux tensor cells,
        and `Req.mamba_pool_idx` Python attrs all point at the new slot
        ids. No deferred flush is required.
        """
        src_tensor = torch.tensor(src_list, dtype=torch.int64, device=self.device)
        dst_tensor = torch.tensor(dst_list, dtype=torch.int64, device=self.device)
        # Delegate the data copy to the sub-pool (MHA: move_kv_cache; Mamba: copy_from).
        move_fn = getattr(self._kvcache, "move_kv_cache", None)
        if move_fn is not None:
            move_fn(dst_tensor, src_tensor)
        else:
            copy_from = getattr(self._kvcache, "copy_from", None)
            assert copy_from is not None, (
                f"sub-pool {self.sub_pool_name!r} supports neither "
                f"move_kv_cache nor copy_from"
            )
            copy_from(src_tensor, dst_tensor)

        # Immediate reverse-mapping rewrite. On return every live
        # reference holder of `src_list[i]` has been switched to
        # `dst_list[i]` (TreeNode.value, req_to_token cells via row
        # tracking, aux tensor cells, py_attr scalars). No deferred
        # flush is required.
        self.relocation_log.apply(self.sub_pool_name, src_tensor, dst_tensor)
        # Inverse-history for two consumers:
        #   1. Spec-decode rollback (restore_state replays inverse moves).
        #   2. Composite-allocator post-free hooks (e.g.,
        #      `SharedSWATokenToKVPoolAllocator.
        #      _apply_new_relocations_to_mapping` reads the suffix to
        #      update `full_to_swa_index_mapping`).
        # The composite is responsible for clearing the slice it
        # consumed via `clear_inverse_history` so the list does not
        # grow unboundedly across long runs.
        self._inverse_history.append((src_tensor, dst_tensor))

    def _allocated_range(self) -> tuple:
        """Return [low, high) slot range currently allocated."""
        if self.grow_direction == "up":
            return (self.min_slot_index, self.watermark)
        else:
            return (self.watermark + 1, self.max_slots)

    # -- diagnostic instrumentation --

    def _raise_stale_slot_assertion(
        self,
        *,
        stale_slots: List[int],
        free_cpu: List[int],
        alloc_low: int,
        alloc_high: int,
    ) -> None:
        """Build a rich diagnostic report explaining why each stale slot
        ended up in the free batch despite being outside the allocated range,
        then raise AssertionError. Goal: root-cause the bug in one log dump
        rather than iterate through more eval runs.

        For each stale slot we report:
          * Where the SlotBacktrack thinks the slot is referenced
            (tree_node / req_position / aux / py_attr).
          * Whether the referenced tensor cell ACTUALLY contains the slot
            (cross-check: stale id vs. binder's claim).
          * Recent inverse-history entries (last few `(src, dst)` pairs
            this allocator emitted) so the chain that landed at the
            stale slot is visible.

        We also dump:
          * The current free batch shape / first/last 32 slots / count over
            and under the watermark.
          * A backtrace from inspect to identify the call site.
          * Recent watermark history (if tracked).
        """
        import inspect

        bt = self.relocation_log.backtracks.get(self.sub_pool_name)
        # Surface the most recent inverse-history pairs (this allocator
        # only). Bounded to the last 32 (src, dst) pairs to keep the dump
        # readable.
        recent_pairs = _flatten_inverse_history_pairs(self._inverse_history)[-32:]
        forward_map: Dict[int, int] = {}
        reverse_map: Dict[int, List[int]] = {}
        for src, dst in recent_pairs:
            forward_map[src] = dst
            reverse_map.setdefault(dst, []).append(src)

        # Identify the call site so we know which radix-cache path is
        # passing the stale slot — this is the single most useful piece of
        # information for narrowing down the root cause.
        try:
            stack = inspect.stack()
            # Skip: _raise_stale_slot_assertion, _free_eager, free,
            # the composite allocator's free wrapper. Show the next 6 frames.
            relevant = [
                f"{frame.filename.split('/')[-1]}:{frame.lineno} in {frame.function}"
                for frame in stack[1:12]
            ]
        except Exception:
            relevant = ["<inspect failed>"]

        # Global bind-hygiene counters (overwrite-aborted tree binds, dup
        # aux / py_attr binds) — usually zero; printed in the dump for
        # cross-correlation.
        if bt is not None:
            bind_overwrite_count = getattr(bt, "bind_tree_overwrite_count", 0)
            bind_aux_dup_count = getattr(bt, "bind_aux_dup_count", 0)
            bind_py_attr_dup_count = getattr(bt, "bind_py_attr_dup_count", 0)
        else:
            bind_overwrite_count = 0
            bind_aux_dup_count = 0
            bind_py_attr_dup_count = 0

        per_slot_reports: List[str] = []
        for s in stale_slots[:16]:
            parts = [f"  slot {s}:"]

            if bt is not None:
                # Tree node binding. Cross-check: does the bound tensor
                # cell still hold `s`?
                tn = bt.tree_node.get(s)
                if tn is not None:
                    target = getattr(tn.node, tn.attr, None)
                    cell_value: Optional[int] = None
                    if isinstance(target, torch.Tensor) and tn.position < target.numel():
                        try:
                            cell_value = int(
                                target.flatten()[tn.position].item()
                            )
                        except Exception:
                            cell_value = None
                    parts.append(
                        f"    tree_node: bound to "
                        f"node=id={id(tn.node):x}, attr={tn.attr!r}, pos={tn.position}; "
                        f"cell_value={cell_value} "
                        f"({'matches stale id' if cell_value == s else 'DIFFERENT from stale id'})"
                    )
                else:
                    parts.append("    tree_node: <no binding>")

                # req_to_token position binding.
                rp = bt.req_position.get(s)
                if rp is not None:
                    parts.append(
                        f"    req_position: col={rp.col}, "
                        f"rows={sorted(rp.rows)}"
                    )
                else:
                    parts.append("    req_position: <no binding>")

                # aux entries.
                ax = bt.aux.get(s)
                if ax:
                    summary = []
                    for ref in ax[:4]:
                        try:
                            cv = int(ref.tensor.flatten()[ref.index].item())
                        except Exception:
                            cv = None
                        summary.append(
                            f"(tensor=id={id(ref.tensor):x}, "
                            f"shape={tuple(ref.tensor.shape)}, "
                            f"index={ref.index}, cell_value={cv})"
                        )
                    parts.append(
                        f"    aux ({len(ax)} ref(s)): "
                        + ", ".join(summary)
                        + (", ..." if len(ax) > 4 else "")
                    )
                else:
                    parts.append("    aux: <no bindings>")

                # py_attr entries.
                pa = bt.py_attr.get(s)
                if pa:
                    summary = []
                    for obj, attr in pa[:4]:
                        cur = getattr(obj, attr, None)
                        cur_val = (
                            int(cur.item()) if hasattr(cur, "item") else cur
                        )
                        summary.append(
                            f"(obj=id={id(obj):x}, attr={attr!r}, current={cur_val})"
                        )
                    parts.append(
                        f"    py_attr ({len(pa)} ref(s)): "
                        + ", ".join(summary)
                        + (", ..." if len(pa) > 4 else "")
                    )
                else:
                    parts.append("    py_attr: <no bindings>")

            # Pending relocation chains involving this slot.
            if s in forward_map:
                # Walk forward: s → s' → s'' → ...
                chain = [s]
                cur = s
                seen = set()
                while cur in forward_map and cur not in seen:
                    seen.add(cur)
                    cur = forward_map[cur]
                    chain.append(cur)
                parts.append(
                    f"    forward chain (s → s' → ...): {' → '.join(map(str, chain))}"
                )
            if s in reverse_map:
                parts.append(
                    f"    reverse (slots that relocated INTO {s}): {reverse_map[s]}"
                )

            # Per-slot lifecycle history (most recent N events).
            if bt is not None and getattr(bt, "history_enabled", False):
                hist = bt.get_history(s)
                if hist:
                    parts.append(
                        f"    lifecycle history ({len(hist)} most recent events):"
                    )
                    for ev in hist:
                        parts.append(f"      {ev}")
                else:
                    parts.append("    lifecycle history: <empty>")

            per_slot_reports.append("\n".join(parts))

        # Truncate the free batch dump to keep log size sane.
        free_first = free_cpu[:32]
        free_last = free_cpu[-32:] if len(free_cpu) > 32 else []
        n_above = sum(1 for x in free_cpu if x >= alloc_high)
        n_below = sum(1 for x in free_cpu if x < alloc_low)

        msg = (
            f"\n"
            f"=== MultiEndedAllocator.free({self.sub_pool_name}) STALE-SLOT ASSERTION ===\n"
            f"Stale slot count: {len(stale_slots)} of {len(free_cpu)} in free batch\n"
            f"  - {n_above} above watermark={alloc_high} (most likely: boundary-freed elsewhere or freed twice)\n"
            f"  - {n_below} below min_slot_index={alloc_low} (reserved zone — invalid id)\n"
            f"\n"
            f"Allocator state: grow_direction={self.grow_direction}, "
            f"min_slot_index={self.min_slot_index}, max_slots={self.max_slots}, "
            f"watermark={self.watermark}, allocated_range=[{alloc_low}, {alloc_high})\n"
            f"\n"
            f"Binder overwrite-detector counters (cumulative since startup):\n"
            f"  bind_tree_node     OVERWRITE: {bind_overwrite_count} "
            f"(same slot bound to TWO tree nodes)\n"
            f"  bind_aux           DUPLICATE: {bind_aux_dup_count} "
            f"(same (slot, tensor, index) bound twice — missing unbind on caller)\n"
            f"  bind_py_attr       DUPLICATE: {bind_py_attr_dup_count} "
            f"(same (slot, obj, attr) bound twice — missing unbind on caller)\n"
            f"  Note: bind_req_position multi-col is an *assert* under the "
            f"immediate-apply refactor — process aborts on first occurrence "
            f"rather than counting.\n"
            f"\n"
            f"Recent relocations (last {len(recent_pairs)} (src, dst) pairs "
            f"emitted by this allocator):\n"
            f"  {recent_pairs}\n"
            f"\n"
            f"Free batch ({len(free_cpu)} slots):\n"
            f"  first 32: {free_first}\n"
            f"  last 32:  {free_last if free_last else '(<= 32 total)'}\n"
            f"\n"
            f"Per-stale-slot binder state (first {min(16, len(stale_slots))}):\n"
            + "\n".join(per_slot_reports)
            + "\n"
            f"\n"
            f"Call stack (innermost first, topmost {len(relevant)} frames):\n"
            "  " + "\n  ".join(relevant)
            + "\n"
            f"=== END STALE-SLOT ASSERTION ==="
        )
        # Also emit as a single logger.error so the dump is fully captured
        # even if the traceback truncates. Then raise.
        logger.error(msg)
        raise AssertionError(msg)

    # -- free-group semantics (deferred-free during a batch) --

    def free_group_begin(self) -> None:
        self.is_not_in_free_group = False
        self.free_group = []

    def free_group_end(self) -> None:
        self.is_not_in_free_group = True
        if self.free_group:
            merged = torch.cat(self.free_group)
            self._free_eager(merged)
            self.free_group = []


class SharedSWATokenToKVPoolAllocator(SWATokenToKVPoolAllocator):
    """
    Composite allocator over a shared-buffer SWA KV pool.

    Mirrors `SWATokenToKVPoolAllocator`'s public interface but wraps two
    `MultiEndedAllocator` instances (one per sub-pool) that coordinate over a
    single `SharedMemoryPool`. Maintains the `full_to_swa_index_mapping`
    tensor and keeps it in sync with alloc/free on both sides.

    Inherits from `SWATokenToKVPoolAllocator` purely for the typing/contract
    relationship — `isinstance(allocator, SWATokenToKVPoolAllocator)` is
    asserted in SWARadixCache, schedule_batch, chunk_cache, and elsewhere.
    We do NOT call `SWATokenToKVPoolAllocator.__init__`: it allocates two
    static-partition sub-pools, which is exactly what shared-pool replaces.
    Instead the grand-parent base init is called directly, and the parent's
    expected `_size_full` / `_size_swa` attributes are set by us so the
    inherited `size_full` / `size_swa` / `size` properties work.
    """

    # The parent declares `size` as a `@property` without a setter, but
    # `BaseTokenToKVPoolAllocator.__init__` does `self.size = size`. Override
    # the property here with a no-op setter so the base init's assignment
    # doesn't raise; reading still returns `min(_size_full, _size_swa)` as
    # the parent intends.
    @property
    def size(self) -> int:
        return min(self._size_full, self._size_swa)

    @size.setter
    def size(self, value) -> None:
        # No-op: `size` is computed from `_size_full` / `_size_swa`. Base
        # class init writes here; we ignore.
        pass

    def __init__(
        self,
        *,
        shared_buffer: SharedMemoryPool,
        kvcache,  # SharedSWAKVPool
        relocation_log: Relocator,
        device: str,
        full_max_total_num_tokens: int,
        swa_max_total_num_tokens: int,
        need_sort: bool = False,
    ):
        # Set _size_full / _size_swa BEFORE the base init so anything that
        # reads `self.size` / `self.size_full` / `self.size_swa` during base
        # init sees a sane value.
        self._size_full = full_max_total_num_tokens
        self._size_swa = swa_max_total_num_tokens
        self._full_max_total_num_tokens = full_max_total_num_tokens
        self._swa_max_total_num_tokens = swa_max_total_num_tokens

        # Skip SWATokenToKVPoolAllocator.__init__ — call the grand-parent
        # base init directly. The base's `self.size = size` call is
        # absorbed by our no-op size setter above.
        BaseTokenToKVPoolAllocator.__init__(
            self,
            size=full_max_total_num_tokens,
            page_size=1,
            dtype=shared_buffer.mha_spec("full").store_dtype,
            device=device,
            kvcache=kvcache,
            need_sort=need_sort,
        )
        self.shared_buffer = shared_buffer
        self._kvcache = kvcache
        self.relocation_log = relocation_log

        self.full_attn_allocator = MultiEndedAllocator(
            kvcache=kvcache.full_kv_pool,
            shared_buffer=shared_buffer,
            sub_pool_name="full",
            relocation_log=relocation_log,
            device=device,
            need_sort=need_sort,
        )
        self.swa_attn_allocator = MultiEndedAllocator(
            kvcache=kvcache.swa_kv_pool,
            shared_buffer=shared_buffer,
            sub_pool_name="swa",
            relocation_log=relocation_log,
            device=device,
            need_sort=need_sort,
        )
        self.full_attn_allocator.bind_peer(self.swa_attn_allocator)
        self.swa_attn_allocator.bind_peer(self.full_attn_allocator)

        # full-slot-id -> swa-slot-id mapping. Sized to cover the max possible
        # full-slot-id (plus a trailing -1 sentinel for loc=-1).
        mapping_size = shared_buffer.max_slots("full") + 1
        self.full_to_swa_index_mapping = torch.cat(
            [
                torch.zeros(
                    mapping_size - 1, dtype=torch.int64, device=device
                ),
                torch.tensor([-1], dtype=torch.int64, device=device),
            ]
        )
        # Register with the kvcache composite (kept weakref to avoid cycles).
        self._kvcache.register_mapping(weakref.proxy(self.full_to_swa_index_mapping))

        self.is_not_in_free_group = True
        self.free_group: List[torch.Tensor] = []

        # See SharedMambaTokenToKVPoolAllocator for why these need empty
        # tensors instead of the base class's None.
        self.free_pages = torch.empty(0, dtype=torch.int64, device=device)
        self.release_pages = torch.empty(0, dtype=torch.int64, device=device)

        logger.info(
            "[shared-pool] SharedSWATokenToKVPoolAllocator ready: "
            "full max_slots=%d (min_slot_index=%d), "
            "swa max_slots=%d (min_slot_index=%d), "
            "available_size=%d slot pairs",
            self.full_attn_allocator.max_slots,
            self.full_attn_allocator.min_slot_index,
            self.swa_attn_allocator.max_slots,
            self.swa_attn_allocator.min_slot_index,
            self.available_size(),
        )

    # -- size reporting --

    def available_size(self) -> int:
        """Max N such that `alloc(N)` succeeds.

        `alloc(N)` consumes N slots on *both* sub-pools, so N slots cost
        `N * (entry_full + entry_swa)` bytes out of the shared byte gap. A
        naive `min(avail_full, avail_swa)` would use only ONE entry size and
        overshoots when both pools alloc from the same byte pool — in the
        worst case the pre-check passes but the sequential alloc crashes
        mid-way because the peer's byte frontier shifted.

        Clamp also by each sub-pool's own slot-index headroom."""
        fa = self.full_attn_allocator
        sa = self.swa_attn_allocator
        entry_sum = fa.entry_bytes + sa.entry_bytes
        # Byte gap between the two allocators' frontiers.
        full_high = fa._byte_high_frontier()
        swa_low = sa._byte_low_frontier()
        gap_bytes = max(0, swa_low - full_high)
        slots_by_bytes = gap_bytes // entry_sum
        # Slot-index-space headroom on each side. Allocatable slots are in
        # [min_slot_index, max_slots); subtract already-allocated.
        full_room = fa.max_slots - fa.min_slot_index - fa.allocated_count()
        swa_room = sa.max_slots - sa.min_slot_index - sa.allocated_count()
        return min(slots_by_bytes, full_room, swa_room)

    # Slot-conservation views of per-side capacity. The leak check at
    # `scheduler_runtime_checker_mixin._check_hybrid_memory` computes
    #   num_used = full_tokens_per_layer - (full_available_size + full_evictable)
    # and expects `num_used == full_protected + session_held` at idle. For
    # this to balance, `full_available_size` must equal
    # `full_max_total_num_tokens - allocated_count`. Returning the byte-
    # coordinated (dynamic, peer-aware) value here would generate spurious
    # leak detections whenever the swa side is using bytes from the shared
    # buffer that originally counted toward the full side's static budget.
    def full_available_size(self) -> int:
        return (
            self._full_max_total_num_tokens
            - self.full_attn_allocator.allocated_count()
        )

    def swa_available_size(self) -> int:
        return (
            self._swa_max_total_num_tokens
            - self.swa_attn_allocator.allocated_count()
        )

    # Byte-coordinated views, used by the scheduler's alloc planner
    # (`common.alloc_for_extend`, `schedule_policy` checks). These reflect
    # the actual current capacity given the peer's byte usage and may be
    # smaller than `full_available_size()` / `swa_available_size()`. On the
    # non-shared `SWATokenToKVPoolAllocator` these methods alias the static
    # views (no peer coupling); the split only matters here.
    def schedulable_full_available_size(self) -> int:
        return self.full_attn_allocator.available_size()

    def schedulable_swa_available_size(self) -> int:
        return self.swa_attn_allocator.available_size()

    # `size_full` / `size_swa` are inherited from `SWATokenToKVPoolAllocator`
    # — they read `_size_full` / `_size_swa`, which we set to the static
    # `full_max_total_num_tokens` / `swa_max_total_num_tokens` in __init__.
    # We deliberately do NOT report `max_slots - 1` here: under shared pool
    # `max_slots("full") = full_max + swa_max`, which would over-promise to
    # any caller treating these as static budgets.

    def debug_print(self) -> str:
        return (
            f"#full-available={self.full_attn_allocator.available_size()}, "
            f"#swa-available={self.swa_attn_allocator.available_size()}"
        )

    def get_kvcache(self):
        return self._kvcache

    def translate_loc_from_full_to_swa(self, kv_indices: torch.Tensor):
        return self._kvcache.translate_loc_from_full_to_swa(kv_indices)

    # -- alloc --

    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        # Joint byte-budget check: both sub-pools consume from the same byte
        # gap, so one alloc of N slot-pairs costs
        #   N * (entry_bytes_full + entry_bytes_swa).
        # This is strictly required when the two sub-pools have different
        # `entry_bytes` (e.g. full_layer_num != swa_layer_num, which is
        # almost always the case for SWA hybrid models), but it is also
        # required in the equal-entry case because the gap is *shared*.
        if need_size > self.available_size():
            return None
        full_select = self.full_attn_allocator.alloc(need_size)
        if full_select is None:
            return None
        swa_select = self.swa_attn_allocator.alloc(need_size)
        if swa_select is None:
            # Roll back the full-side alloc so composite state stays coherent.
            self.full_attn_allocator.watermark -= need_size
            return None
        # Record 1:1 mapping full -> swa.
        self.full_to_swa_index_mapping[full_select] = swa_select
        return full_select

    def is_slot_allocated(self, slot: int) -> bool:
        """Whether `slot` is inside the FULL sub-pool's allocated range. The
        token-slot surface is the full-attn side; SWA radix cache passes
        full-pool slot ids when validating before `free()`."""
        return self.full_attn_allocator.is_slot_allocated(slot)

    def allocator_state_str(self) -> str:
        return self.full_attn_allocator.allocator_state_str()

    # -- free --

    def free(self, free_index: torch.Tensor) -> None:
        if free_index is None or free_index.numel() == 0:
            return
        if not self.is_not_in_free_group:
            self.free_group.append(free_index)
            return
        self._free_both(free_index)

    def _apply_new_relocations_to_mapping(
        self,
        n_full_before: Optional[int],
        n_swa_before: int,
    ) -> None:
        """Apply relocation log entries appended *during this call* to
        `full_to_swa_index_mapping` immediately, instead of deferring to
        `flush_relocations`.

        WHY EAGER: a later `_free_both` / `free_swa` in the same batch may
        clear `mapping[src]` (when src happens to be in that call's
        `free_index`) before flush runs. Once `mapping[src]` is zero, flush's
        `mapping[dst] = mapping[src]` step writes 0 instead of the swa pair
        that the relocation moved into `dst`, and the swa slot becomes
        orphaned (the cause of the SWA-leaked=33 we saw under skewed bench
        load on gpt-oss-20b). Applying eagerly captures `mapping[src]` at
        record time, while it is still correct.

        Full side: `mapping[dst] = mapping[src]; mapping[src] = 0`. Done in
        recorded order so chained relocations (A→B then B→C) resolve
        left-to-right correctly.

        Swa side: relocations move swa SLOT IDS, so any cell of `mapping`
        whose VALUE equals a relocated `src` must become `dst`. Done with
        a single GPU `lookup_swa[mapping]` so chaining is implicit (build
        lookup_swa in order — later entries overwrite earlier ones — and
        any prior entry already bears its final dst).
        """
        # Diagnostic: assert mapping consistency after each step. A consistent
        # mapping satisfies: every non-zero `mapping[i]` is a SWA slot id in
        # the swa allocator's allocated range, AND every allocated SWA slot
        # is referenced by exactly one full slot's mapping entry. We only
        # check the cheaper subset (no orphan-swa-slot scan, that's O(swa_max))
        # at debug-mode boundaries. Enable with
        # `SGLANG_SHARED_POOL_MAPPING_ASSERT=1`.
        import os
        do_assert = os.environ.get("SGLANG_SHARED_POOL_MAPPING_ASSERT") == "1"

        if n_full_before is not None:
            new_full = _flatten_inverse_history_pairs(
                self.full_attn_allocator._inverse_history[n_full_before:]
            )
            for src, dst in new_full:
                if do_assert:
                    # Pre-step invariant: src should be in the post-relocation
                    # free zone (above watermark), dst should be allocated.
                    src_pair_before = int(self.full_to_swa_index_mapping[src].item())
                    dst_pair_before = int(self.full_to_swa_index_mapping[dst].item())
                    if dst_pair_before != 0:
                        logger.warning(
                            "[_apply_new_relocations_to_mapping] full step "
                            "(src=%d, dst=%d) ABOUT TO OVERWRITE non-zero "
                            "mapping[dst]=%d with mapping[src]=%d. This "
                            "leaks the SWA pair previously at dst.",
                            src, dst, dst_pair_before, src_pair_before,
                        )
                self.full_to_swa_index_mapping[dst] = (
                    self.full_to_swa_index_mapping[src]
                )
                self.full_to_swa_index_mapping[src] = 0

        new_swa = _flatten_inverse_history_pairs(
            self.swa_attn_allocator._inverse_history[n_swa_before:]
        )
        if new_swa:
            mapping = self.full_to_swa_index_mapping[:-1]  # skip -1 sentinel
            swa_max = self.swa_attn_allocator.max_slots
            # Build the chain-resolved lookup table. Within ONE _free_eager
            # call, eager compaction can emit pending entries (b1, s1), (b2, s2)
            # where the relocation chain (e.g. b1 → s1, then later b2 → s1
            # if s1 ends up as a boundary in a follow-on iteration) means a
            # mapping cell originally holding b1 should land on the FINAL
            # destination, not the intermediate one. A single-pass
            # `mapping.copy_(lookup_swa[mapping])` only does ONE hop — it
            # leaves cells holding intermediate values stale. Surface as
            # "swa step left mapping cell holding stale src=X after remap"
            # in eval_results_18 with the MAPPING_ASSERT instrumentation.
            forward = {src: dst for src, dst in new_swa}
            chain_resolved = {}
            for src in forward:
                cur = forward[src]
                visited = {src, cur}
                while cur in forward and forward[cur] not in visited:
                    cur = forward[cur]
                    visited.add(cur)
                chain_resolved[src] = cur
            lookup_swa = torch.arange(
                swa_max, dtype=mapping.dtype, device=self.device
            )
            for src, terminal_dst in chain_resolved.items():
                lookup_swa[src] = terminal_dst
            if do_assert:
                # Pre-remap snapshot: which full slots had which swa pairs.
                pre_mapping_cpu = self.full_to_swa_index_mapping.detach().cpu().clone()
            mapping.copy_(lookup_swa[mapping])
            if do_assert:
                post_mapping_cpu = self.full_to_swa_index_mapping.detach().cpu().clone()
                # Verify: for each (src, dst) in new_swa, every cell that was
                # `src` is now `dst`. (Built into lookup_swa, but we double-
                # check.) Additionally check that the count of non-zero pairs
                # is preserved (no swa pair was silently dropped).
                pre_nonzero = int((pre_mapping_cpu != 0).sum().item())
                post_nonzero = int((post_mapping_cpu != 0).sum().item())
                if pre_nonzero != post_nonzero:
                    logger.warning(
                        "[_apply_new_relocations_to_mapping] swa step changed "
                        "non-zero pair count from %d -> %d after %d swa "
                        "relocations. Pairs may have been lost.",
                        pre_nonzero, post_nonzero, len(new_swa),
                    )
                # And: no remaining cell holds a value in {src for src,_ in new_swa}
                # (every src should have been replaced by dst).
                src_values = {src for src, _ in new_swa}
                if src_values:
                    for v in post_mapping_cpu.flatten().tolist():
                        if v in src_values:
                            logger.warning(
                                "[_apply_new_relocations_to_mapping] swa step "
                                "left mapping cell holding stale src=%d after "
                                "remap; expected to be replaced by its dst.",
                                v,
                            )
                            break

    def _free_both(self, free_index: torch.Tensor) -> None:
        # Under immediate-apply, by the time the radix cache calls this
        # method the slot ids in `free_index` were sourced from a *bound*
        # holder (TreeNode.value, req_to_token, aux), so the binder has
        # already kept them current — no chain-chasing through a
        # relocation log is required (the pending-log was eliminated in
        # the immediate-apply refactor). Callers passing pre-compaction
        # Python clones must avoid that pattern.
        # Read swa pair indices BEFORE any compaction touches the mapping.
        swa_indices = self.full_to_swa_index_mapping[free_index]
        swa_indices = swa_indices[swa_indices > 0]

        # Snapshot inner-allocator inverse-history lengths so the
        # post-free hook can read just the suffix produced by this call.
        n_full_before = len(self.full_attn_allocator._inverse_history)
        n_swa_before = len(self.swa_attn_allocator._inverse_history)

        self.full_attn_allocator.free(free_index)
        self.swa_attn_allocator.free(swa_indices)

        # Clear mapping at the freed positions FIRST, then apply this
        # call's new relocations on top. Order matters: a slot may
        # appear both as a freed position (cleared here) and as the dst
        # of a relocation (filled below from mapping[src]).
        self.full_to_swa_index_mapping[free_index] = 0
        self._apply_new_relocations_to_mapping(n_full_before, n_swa_before)

        # Drop inverse-history entries we just consumed so they don't
        # accumulate across long runs. Spec-decode rollback is unaffected
        # because the spec pattern doesn't free inside a backup window.
        self.full_attn_allocator.clear_inverse_history()
        self.swa_attn_allocator.clear_inverse_history()

    def free_swa(self, free_index: torch.Tensor) -> None:
        """Free only the SWA-side slots corresponding to the given full
        slot ids. The full side is left untouched.

        Mirrors `SWATokenToKVPoolAllocator.free_swa`. Called by
        `SWARadixCache._evict_swa_only` when a tree node has aged past
        the sliding-window horizon — its swa state is no longer
        reachable but its full state still is, so the swa-side budget
        gets reclaimed without disturbing full bookkeeping.
        """
        if free_index is None or free_index.numel() == 0:
            return
        swa_indices = self.full_to_swa_index_mapping[free_index]
        swa_indices = swa_indices[swa_indices > 0]

        # Snapshot inverse-history lengths so the post-free hook reads
        # only the suffix produced by this call.
        n_swa_before = len(self.swa_attn_allocator._inverse_history)

        if swa_indices.numel() > 0:
            self.swa_attn_allocator.free(swa_indices)

        # Mark mapping entries as "no swa" (-> 0) for the freed full
        # slots. The full slot ids themselves remain valid.
        self.full_to_swa_index_mapping[free_index] = 0
        # Apply any swa-side relocations triggered by the swa.free above.
        # Full side untouched, so n_full_before=None.
        self._apply_new_relocations_to_mapping(None, n_swa_before)

        # Drop the swa-side history we just consumed.
        self.swa_attn_allocator.clear_inverse_history()

    def free_group_begin(self) -> None:
        self.is_not_in_free_group = False
        self.free_group = []

    def free_group_end(self) -> None:
        self.is_not_in_free_group = True
        if self.free_group:
            merged = torch.cat(self.free_group)
            self._free_both(merged)
            self.free_group = []

    # -- state mgmt --

    def backup_state(self):
        return [
            self.full_attn_allocator.backup_state(),
            self.swa_attn_allocator.backup_state(),
        ]

    def restore_state(self, state):
        # Note: returning per-sub-pool rollback lists (caller may need them
        # to reverse KV-buffer moves in speculative rollback).
        assert len(state) == 2
        full_rollback = self.full_attn_allocator.restore_state(state[0])
        swa_rollback = self.swa_attn_allocator.restore_state(state[1])
        return full_rollback + swa_rollback

    def clear(self) -> None:
        self.full_attn_allocator.clear()
        self.swa_attn_allocator.clear()
        # Reset mapping except the trailing -1 sentinel.
        self.full_to_swa_index_mapping[:-1].fill_(0)
        self.is_not_in_free_group = True
        self.free_group = []


class SharedMambaTokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):
    """
    Composite allocator for the MHA (full-attn) + Mamba hybrid pair over a
    `SharedMemoryPool`.

    Public allocator surface (the slot allocator used by the scheduler's
    token-slot path) delegates to the full-attn side — `alloc(N)` allocates
    MHA token slots. The Mamba sub-pool's `alloc(1)`-per-request is driven
    by `SharedHybridReqToTokenPool.alloc`, which calls
    `mamba_pool.alloc(1)`; SharedMambaPool then delegates to
    `mamba_allocator` (attached at construction).
    """

    def __init__(
        self,
        *,
        shared_buffer: SharedMemoryPool,
        kvcache,  # HybridLinearKVPool
        relocation_log: Relocator,
        device: str,
        need_sort: bool = False,
    ):
        full_max = shared_buffer.max_slots("full")
        super().__init__(
            size=full_max - 1,
            page_size=1,
            dtype=shared_buffer.mha_spec("full").store_dtype,
            device=device,
            kvcache=kvcache,
            need_sort=need_sort,
        )
        self.shared_buffer = shared_buffer
        self._kvcache = kvcache
        self.relocation_log = relocation_log

        self.full_attn_allocator = MultiEndedAllocator(
            kvcache=kvcache.full_kv_pool,
            shared_buffer=shared_buffer,
            sub_pool_name="full",
            relocation_log=relocation_log,
            device=device,
            need_sort=need_sort,
        )
        self.mamba_allocator = MultiEndedAllocator(
            kvcache=kvcache.mamba_pool,
            shared_buffer=shared_buffer,
            sub_pool_name="mamba",
            relocation_log=relocation_log,
            device=device,
            need_sort=need_sort,
        )
        self.full_attn_allocator.bind_peer(self.mamba_allocator)
        self.mamba_allocator.bind_peer(self.full_attn_allocator)

        # Wire SharedMambaPool to delegate alloc/free to the mamba_allocator.
        kvcache.mamba_pool.attach_allocator(self.mamba_allocator)

        self.is_not_in_free_group = True
        self.free_group: List[torch.Tensor] = []

        # The base class init left these as None (it expected the caller to
        # populate them with index tensors). Our allocator uses watermark
        # math, not free-lists. Provide empty tensors so the scheduler's leak
        # checker (`free_pages.tolist()`) doesn't NPE if it ever runs.
        self.free_pages = torch.empty(0, dtype=torch.int64, device=device)
        self.release_pages = torch.empty(0, dtype=torch.int64, device=device)

        logger.info(
            "[shared-pool] SharedMambaTokenToKVPoolAllocator ready: "
            "full max_slots=%d (min_slot_index=%d), "
            "mamba max_slots=%d (min_slot_index=%d), "
            "full_available=%d, mamba_available=%d",
            self.full_attn_allocator.max_slots,
            self.full_attn_allocator.min_slot_index,
            self.mamba_allocator.max_slots,
            self.mamba_allocator.min_slot_index,
            self.full_attn_allocator.available_size(),
            self.mamba_allocator.available_size(),
        )

    # -- size: dynamic --
    # The leak checker in scheduler_runtime_checker_mixin computes
    #   num_used = size - available - evictable
    # If `size` is a static `max_slots - 1`, the slot-0-safety reservation
    # plus the byte-leftover at the buffer's end show up as bogus "num_used"
    # at startup, triggering false-positive leak detection. Override `size`
    # so it always equals available + allocated (a self-consistent count of
    # this allocator's current capacity), which makes num_used == allocated
    # - evictable — the value the leak check actually wants.
    @property
    def size(self) -> int:
        return (
            self.full_attn_allocator.available_size()
            + self.full_attn_allocator.allocated_count()
        )

    @size.setter
    def size(self, value) -> None:
        # Base.__init__ sets self.size = size; ignore — we compute dynamically.
        pass

    # -- token-slot surface: MHA side only --

    def available_size(self) -> int:
        return self.full_attn_allocator.available_size()

    def full_available_size(self) -> int:
        return self.full_attn_allocator.available_size()

    def mamba_available_size(self) -> int:
        return self.mamba_allocator.available_size()

    @property
    def size_full(self) -> int:
        return self.full_attn_allocator.max_slots - 1

    @property
    def size_mamba(self) -> int:
        return self.mamba_allocator.max_slots - 1

    def debug_print(self) -> str:
        return (
            f"#full-available={self.full_attn_allocator.available_size()}, "
            f"#mamba-available={self.mamba_allocator.available_size()}"
        )

    def get_kvcache(self):
        return self._kvcache

    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        return self.full_attn_allocator.alloc(need_size)

    def is_slot_allocated(self, slot: int) -> bool:
        """Whether `slot` is currently inside the FULL sub-pool's allocated
        range. The token-slot surface here is the full-attn side, so callers
        validating a full-pool slot id (e.g. `_free_dedup_slots`) route here."""
        return self.full_attn_allocator.is_slot_allocated(slot)

    def allocator_state_str(self) -> str:
        return self.full_attn_allocator.allocator_state_str()

    def free(self, free_index: torch.Tensor) -> None:
        if free_index is None or free_index.numel() == 0:
            return
        if not self.is_not_in_free_group:
            self.free_group.append(free_index)
            return
        self.full_attn_allocator.free(free_index)
        # Mamba composite has no auxiliary mapping to maintain — the
        # full/mamba aux + py_attr bindings are already updated by the
        # immediate apply inside `_apply_relocations`. Drop the
        # inverse-history we just produced so it doesn't accumulate.
        self.full_attn_allocator.clear_inverse_history()
        self.mamba_allocator.clear_inverse_history()

    def free_group_begin(self) -> None:
        self.is_not_in_free_group = False
        self.free_group = []

    def free_group_end(self) -> None:
        self.is_not_in_free_group = True
        if self.free_group:
            merged = torch.cat(self.free_group)
            self.full_attn_allocator.free(merged)
            self.free_group = []
            self.full_attn_allocator.clear_inverse_history()
            self.mamba_allocator.clear_inverse_history()

    def backup_state(self):
        return [
            self.full_attn_allocator.backup_state(),
            self.mamba_allocator.backup_state(),
        ]

    def restore_state(self, state):
        assert len(state) == 2
        full_rollback = self.full_attn_allocator.restore_state(state[0])
        mamba_rollback = self.mamba_allocator.restore_state(state[1])
        return full_rollback + mamba_rollback

    def clear(self) -> None:
        self.full_attn_allocator.clear()
        self.mamba_allocator.clear()
        self.is_not_in_free_group = True
        self.free_group = []
