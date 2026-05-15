"""MultiEndedAllocator: one allocator per sub-pool over a `SharedMemoryPool`.

See `shared_memory_pool_design.md` (same directory). Key points for v2:

* Each `MultiEndedAllocator` owns: a physical watermark (grows up from
  `min_slot_index`, or down from `max_slots-1`), per-sub-pool
  `virtual_to_physical` / `physical_to_virtual` tables, and — iff it is the
  *id-owner* of its virtual-id granularity — the `free_virtual_ids` free-list.
* `alloc(N)` (id-owner only): pop N virtual ids, take N physical slots, bind.
  `alloc_with_virtual(virtual_ids)` (physical-holding non-owner): take physical
  slots for caller-supplied virtual ids, bind. `free(virtual_ids)`: translate to
  this pool's physical ids, eager-compact (boundary slot copied into freed holes,
  remap the two tables), and (if id-owner) recycle the virtual ids.
* Eager compaction touches **only** `virtual_to_physical` / `physical_to_virtual`
  (O(num_relocations) scalar ops) — no reference rewriting, no binder.

Concurrency: the scheduler runs alloc/free serially; no mutex is taken here.
"""

from __future__ import annotations

import inspect
import logging
from typing import List, Optional, Tuple

import torch

from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.shared_memory_pool import SharedMemoryPool
from sglang.srt.mem_cache.swa_memory_pool import SWATokenToKVPoolAllocator

logger = logging.getLogger(__name__)


class MultiEndedAllocator(BaseTokenToKVPoolAllocator):
    """Allocator for one sub-pool over a `SharedMemoryPool`."""

    def __init__(
        self,
        *,
        kvcache,
        shared_buffer: SharedMemoryPool,
        sub_pool_name: str,
        device: str,
        is_id_owner: bool,
        need_sort: bool = False,
        forward_stream: Optional[torch.cuda.Stream] = None,
    ):
        spec = shared_buffer.spec(sub_pool_name)
        max_slots = shared_buffer.max_slots(sub_pool_name)
        # `dtype` on the base is informational; MHA specs have `store_dtype`,
        # Mamba specs have `conv_dtype`.
        spec_dtype = getattr(spec, "store_dtype", getattr(spec, "conv_dtype", None))
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
        self.min_slot_index = shared_buffer.min_slot_index(sub_pool_name)
        self.is_id_owner = is_id_owner
        # Optional handle for the model forward stream. In overlap mode the
        # scheduler runs `pop_and_process` (which triggers `free` →
        # `_compact_pending`'s `move_kv_cache`) on `schedule_stream` while
        # the in-flight forward batch is still reading v2p / reading+writing
        # K/V slots on `forward_stream`. We use this handle to drop a one-way
        # `schedule_stream.wait_stream(forward_stream)` barrier at the top of
        # `free` so the v2p writes and the move kernel serialize after the
        # forward's reads/writes complete. The reverse direction — the next
        # forward seeing the allocator's writes — is already handled by the
        # existing `forward_stream.wait_stream(schedule_stream)` barrier at
        # the top of `run_batch`. In normal schedule the barrier is a near-
        # no-op because sampling's CPU sync has already drained forward_stream
        # before pop_and_process runs.
        self.forward_stream = forward_stream

        # The virtual-id space for this granularity has the same size as the
        # physical slot space here (both = total_bytes // entry_bytes).
        self.num_virtual_ids = max_slots

        # v -> p (slot id, page_size=1); -1 if unmapped; trailing entry [-1] -> -1.
        self.virtual_to_physical = torch.full(
            (self.num_virtual_ids + 1,), -1, dtype=torch.int64, device=device
        )
        # p -> v; -1 if physical slot free; trailing entry [-1] -> -1.
        self.physical_to_virtual = torch.full(
            (self.max_slots + 1,), -1, dtype=torch.int64, device=device
        )

        self._peer: Optional["MultiEndedAllocator"] = None

        # Inverse history of relocations (Stage-5 spec rollback). Each entry is
        # one batch (src_phys_tensor, dst_phys_tensor, v_moved_tensor). The
        # composite calls `clear_inverse_history` after each `free` so it stays
        # bounded.
        self._inverse_history: List[
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] = []

        self.clear()

        logger.info(
            "[shared-pool] MultiEndedAllocator(%r) ready: grow=%s, max_slots=%d, "
            "min_slot_index=%d, entry_bytes=%d, is_id_owner=%s, initial_watermark=%d, "
            "allocatable=%d",
            self.sub_pool_name,
            self.grow_direction,
            self.max_slots,
            self.min_slot_index,
            self.entry_bytes,
            self.is_id_owner,
            self.watermark_physical,
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
        if self.grow_direction == "up":
            self.watermark_physical = self.min_slot_index
        else:
            self.watermark_physical = self.max_slots - 1
        self.virtual_to_physical.fill_(-1)
        self.virtual_to_physical[0] = 0  # virtual id 0 <-> physical 0 (padding sink)
        self.virtual_to_physical[-1] = -1  # trailing sentinel
        self.physical_to_virtual.fill_(-1)
        self.physical_to_virtual[0] = 0
        self.physical_to_virtual[-1] = -1
        if self.is_id_owner:
            # Virtual id 0 and 1..min-1 are reserved (mirroring the physical
            # reservation); the trailing sentinel row is not in the free list.
            self.free_virtual_ids = torch.arange(
                self.min_slot_index, self.max_slots, dtype=torch.int64, device=self.device
            )
        else:
            self.free_virtual_ids = None
        self.is_not_in_free_group = True
        self.free_group: List[torch.Tensor] = []
        self._inverse_history.clear()

    def backup_state(self):
        # SGLang's spec-decode pattern allocates only inside a backup window
        # (no free), so under correct usage `_inverse_history` doesn't grow.
        return (
            self.watermark_physical,
            (len(self.free_virtual_ids) if self.is_id_owner else None),
            len(self._inverse_history),
        )

    def restore_state(self, state):
        watermark, n_free_virtual, n_inverse = state
        self.watermark_physical = watermark
        if self.is_id_owner and n_free_virtual is not None:
            # In-window allocs sliced `free_virtual_ids` from the front; we can't
            # un-pop a slice without re-tracking. Simplest correct restore: the
            # ids consumed in-window are still bound (their v2p entries point at
            # physical slots now below the restored watermark — harmless, they get
            # overwritten on next alloc). We only need to restore the *count* of
            # free virtual ids by re-deriving the free list from the watermark +
            # the bound set. Cheapest: rebuild from scratch is O(max_slots); since
            # spec is asserted off in Stage 1 this path is not exercised, so we
            # take the simple route: re-derive on the next `alloc` is not enough,
            # so do a full rebuild here.
            #
            # NOTE: this is intentionally conservative; Stage 5 will revisit if
            # the spec hot path needs O(1) rollback.
            pass  # placeholder; spec asserted off in Stage 1.
        new_entries = self._inverse_history[n_inverse:]
        if new_entries:
            logger.warning(
                "MultiEndedAllocator.restore_state: %d relocation(s) recorded inside "
                "a backup window (sub_pool=%s). Eager compaction is not fully "
                "reversible; SGLang's spec path should not produce a free() inside a "
                "backup window.",
                len(new_entries),
                self.sub_pool_name,
            )
        del self._inverse_history[n_inverse:]
        return new_entries

    def clear_inverse_history(self) -> None:
        self._inverse_history.clear()

    # -- size reporting --

    def allocated_count(self) -> int:
        if self.grow_direction == "up":
            return max(0, self.watermark_physical - self.min_slot_index)
        return max(0, self.max_slots - 1 - self.watermark_physical)

    def is_slot_allocated(self, slot: int) -> bool:
        """Whether VIRTUAL id `slot` is currently in use."""
        if slot < 0 or slot >= self.num_virtual_ids:
            return False
        return int(self.virtual_to_physical[slot].item()) != -1

    def allocator_state_str(self) -> str:
        return (
            f"sub_pool={self.sub_pool_name!r}, grow_direction={self.grow_direction}, "
            f"is_id_owner={self.is_id_owner}, min_slot_index={self.min_slot_index}, "
            f"max_slots={self.max_slots}, watermark_physical={self.watermark_physical}, "
            f"allocated_count={self.allocated_count()}"
        )

    def _byte_high_frontier(self) -> int:
        if self.grow_direction == "up":
            return self.watermark_physical * self.entry_bytes
        return self.max_slots * self.entry_bytes

    def _byte_low_frontier(self) -> int:
        if self.grow_direction == "up":
            return self.min_slot_index * self.entry_bytes
        return (self.watermark_physical + 1) * self.entry_bytes

    def available_size(self) -> int:
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
            peer_high = self._peer._byte_high_frontier() if self._peer is not None else 0
            gap_bytes = max(0, my_low - peer_high)
        slots_by_bytes = gap_bytes // self.entry_bytes
        slots_by_index_space = self.max_slots - self.min_slot_index - self.allocated_count()
        return min(slots_by_bytes, slots_by_index_space)

    # -- physical-slot primitives --

    def take_physical(self, need_size: int) -> Optional[torch.Tensor]:
        """Advance the physical watermark by `need_size`; return the new physical
        slot ids. Returns None if the grow-down watermark would underflow
        `min_slot_index` (the byte-frontier check in `available_size` should have
        caught this earlier; this is a defensive backstop)."""
        if need_size <= 0:
            return torch.empty(0, dtype=torch.int64, device=self.device)
        if self.grow_direction == "up":
            start = self.watermark_physical
            phys = torch.arange(start, start + need_size, dtype=torch.int64, device=self.device)
            self.watermark_physical += need_size
            return phys
        else:
            end = self.watermark_physical
            start = end - need_size + 1
            if start < self.min_slot_index:
                return None
            phys = torch.arange(start, end + 1, dtype=torch.int64, device=self.device)
            self.watermark_physical -= need_size
            return phys

    def bind(self, virtual_ids: torch.Tensor, physical_ids: torch.Tensor) -> None:
        self.virtual_to_physical[virtual_ids] = physical_ids
        self.physical_to_virtual[physical_ids] = virtual_ids

    # -- alloc --

    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        """Allocate `need_size` virtual ids (id-owner only). Returns the virtual
        ids, or None if there is not enough byte room / virtual headroom.

        Stream model (minimal):
          - All allocator GPU ops run on the scheduler thread's current stream
            (== `schedule_stream` inside the scheduler's `StreamContext`).
            This matches the upstream non-shared allocator behavior, so there
            are zero cross-stream interactions between cat/slice/v2p writes
            and `write_cache_indices` (which is also on schedule_stream).
          - For correctness in overlap mode (where the model forward runs on
            `forward_stream` and may still be in flight when `free` is
            called), we issue one `current_stream.wait_stream(forward_stream)`
            barrier at the very top of `free` — see `free` for the rationale.
            `alloc` doesn't need that barrier: its writes to v2p / p2v are
            picked up by the forward via the existing
            `forward_stream.wait_stream(schedule_stream)` at the top of
            `run_batch`.
        """
        assert self.is_id_owner, (
            f"MultiEndedAllocator({self.sub_pool_name!r}).alloc called on a "
            "non-id-owner allocator; use alloc_with_virtual instead"
        )
        if need_size <= 0:
            return torch.empty(0, dtype=torch.int64, device=self.device)
        if need_size > self.available_size():
            return None
        v = self.free_virtual_ids[:need_size]
        self.free_virtual_ids = self.free_virtual_ids[need_size:]
        phys = self.take_physical(need_size)
        if phys is None:
            # Undo the virtual pop.
            self.free_virtual_ids = torch.cat([v, self.free_virtual_ids])
            return None
        self.bind(v, phys)
        return v

    def alloc_with_virtual(self, virtual_ids: torch.Tensor) -> None:
        """Take physical slots for caller-supplied virtual ids (physical-holding
        non-owner). Used by the SWA `swa` sub-allocator in Stage 2; unused in
        Stage 1 (both Mamba sub-pools are id-owners)."""
        if virtual_ids.numel() == 0:
            return
        phys = self.take_physical(int(virtual_ids.numel()))
        assert phys is not None, (
            f"MultiEndedAllocator({self.sub_pool_name!r}).alloc_with_virtual: out of "
            "physical room (the composite's byte-budget check should have caught this)"
        )
        self.bind(virtual_ids, phys)

    # -- free with eager compaction --

    def free(self, free_index: torch.Tensor) -> None:
        """Free virtual ids; un-map v2p / p2v; (if id-owner) recycle ids;
        trigger eager compaction.

        Stream model: all GPU ops run on the scheduler thread's current
        stream (schedule_stream). The ONE cross-stream concern is overlap
        mode, where this is called by `pop_and_process` while the in-flight
        forward batch is still reading v2p on `forward_stream` and reading/
        writing K/V slots that `_compact_pending`'s `move_kv_cache` will
        relocate. The single `current.wait_stream(forward_stream)` barrier
        below makes schedule_stream wait for the in-flight forward kernels
        before any v2p write or move — eliminating that race without any
        stream switching (which previously introduced data-corruption bugs).
        """
        if free_index is None or free_index.numel() == 0:
            return
        if not self.is_not_in_free_group:
            self.free_group.append(free_index)
            return
        # Overlap-mode barrier (single, at the start). In normal mode this is
        # a near-no-op because sampling's CPU sync has already drained
        # forward_stream. In overlap mode it serializes free+compaction with
        # the in-flight forward, which is what we need for correctness.
        if self.forward_stream is not None:
            torch.cuda.current_stream().wait_stream(self.forward_stream)
        free_v = torch.unique(free_index.detach().to(torch.int64))
        freed_p = self.virtual_to_physical[free_v]
        if bool((freed_p < 0).any().item()):
            self._raise_stale_slot_assertion(free_v=free_v, freed_p=freed_p)
        # Un-map; (if id-owner) recycle the virtual ids.
        self.virtual_to_physical[free_v] = -1
        if self.is_id_owner:
            self.free_virtual_ids = torch.cat([self.free_virtual_ids, free_v])
        self._compact_pending(freed_p)

    def _compact_pending(self, freed_physical_ids: torch.Tensor) -> None:
        """Eager compaction over the freed PHYSICAL slots: move the survivors that
        fall in the *vacated band* (the K slots adjacent to the watermark, where
        K = #freed) into the *holes in the kept band*, then advance the watermark
        and remap the two tables. The `src` set (⊆ vacated band) and the `dst` set
        (⊆ kept band) are disjoint by construction, so the batched data copy is
        order-independent. Separable so a future flag can defer it (lazy mode);
        in Stage 1 `free` calls it inline (eager).

        All GPU ops here run on the scheduler thread's current stream
        (schedule_stream). The wait_stream barrier in the caller (`free`)
        already serialized us with any in-flight forward kernels."""
        freed_set = set(int(x) for x in freed_physical_ids.tolist())
        if not freed_set:
            return
        K = len(freed_set)
        if self.grow_direction == "up":
            # allocated == [min_slot_index, old_wm); after the free == [min_slot_index, new_wm)
            old_wm = self.watermark_physical
            new_wm = old_wm - K
            assert new_wm >= self.min_slot_index, (
                f"_compact_pending({self.sub_pool_name!r}): freeing {K} would push the "
                f"watermark below min_slot_index ({new_wm} < {self.min_slot_index})"
            )
            assert all(self.min_slot_index <= h < old_wm for h in freed_set), (
                f"_compact_pending({self.sub_pool_name!r}): freed physical slots "
                f"{sorted(freed_set)} not all within allocated range "
                f"[{self.min_slot_index}, {old_wm})"
            )
            # vacated band = [new_wm, old_wm); kept band = [min_slot_index, new_wm)
            src_list = [s for s in range(new_wm, old_wm) if s not in freed_set]
            dst_list = sorted(h for h in freed_set if h < new_wm)
            self.watermark_physical = new_wm
            vacated_lo, vacated_hi = new_wm, old_wm
        else:
            # allocated == (old_wm, max_slots); after the free == (new_wm, max_slots)
            old_wm = self.watermark_physical
            new_wm = old_wm + K
            assert new_wm <= self.max_slots - 1, (
                f"_compact_pending({self.sub_pool_name!r}): freeing {K} would push the "
                f"watermark above max_slots ({new_wm} > {self.max_slots - 1})"
            )
            assert all(old_wm < h < self.max_slots for h in freed_set), (
                f"_compact_pending({self.sub_pool_name!r}): freed physical slots "
                f"{sorted(freed_set)} not all within allocated range "
                f"({old_wm}, {self.max_slots})"
            )
            # vacated band = (old_wm, new_wm] = [old_wm+1, new_wm+1); kept band = (new_wm, max_slots)
            src_list = [s for s in range(old_wm + 1, new_wm + 1) if s not in freed_set]
            dst_list = sorted(h for h in freed_set if h > new_wm)
            self.watermark_physical = new_wm
            vacated_lo, vacated_hi = old_wm + 1, new_wm + 1

        assert len(src_list) == len(dst_list), (
            f"_compact_pending({self.sub_pool_name!r}): {len(src_list)} survivors vs "
            f"{len(dst_list)} holes — corrupt allocator state"
        )

        if src_list:
            src_t = torch.tensor(src_list, dtype=torch.int64, device=self.device)
            dst_t = torch.tensor(dst_list, dtype=torch.int64, device=self.device)
            v_moved = self.physical_to_virtual[src_t].clone()  # read before clearing
            # Data copy. MHA (full) -> SharedMHATokenToKVPool.move_kv_cache(dst, src);
            # Mamba -> SharedMambaPool._copy_from_physical(src, dst) (un-translated —
            # the public copy_from translates virtual ids, which we must NOT do here).
            move_fn = getattr(self._kvcache, "move_kv_cache", None)
            if move_fn is not None:
                move_fn(dst_t, src_t)
            else:
                copy_phys = getattr(self._kvcache, "_copy_from_physical", None)
                assert copy_phys is not None, (
                    f"sub-pool {self.sub_pool_name!r} supports neither move_kv_cache "
                    "nor _copy_from_physical"
                )
                copy_phys(src_t, dst_t)
            # Clear the whole vacated band, then re-bind the relocated dst slots
            # (dst_t ⊆ kept band, disjoint from the vacated band).
            self.physical_to_virtual[vacated_lo:vacated_hi] = -1
            self.virtual_to_physical[v_moved] = dst_t
            self.physical_to_virtual[dst_t] = v_moved
            self._inverse_history.append((src_t, dst_t, v_moved))
        else:
            self.physical_to_virtual[vacated_lo:vacated_hi] = -1

    def _raise_stale_slot_assertion(self, *, free_v, freed_p) -> None:
        bad = free_v[freed_p < 0].tolist()
        frames = inspect.stack()[1:9]
        callers = " <- ".join(f"{f.filename.split('/')[-1]}:{f.lineno}" for f in frames)
        raise AssertionError(
            f"MultiEndedAllocator({self.sub_pool_name!r}).free: virtual id(s) {bad} have "
            f"virtual_to_physical == -1 (double-free or never-allocated). "
            f"State: {self.allocator_state_str()}. free_index unique={free_v.tolist()}. "
            f"recent _inverse_history (last 3): "
            f"{[(s.tolist(), d.tolist()) for s, d, _ in self._inverse_history[-3:]]}. "
            f"Caller: {callers}."
        )

    # -- free-group --

    def free_group_begin(self) -> None:
        self.is_not_in_free_group = False
        self.free_group = []

    def free_group_end(self) -> None:
        self.is_not_in_free_group = True
        if self.free_group:
            merged = torch.cat(self.free_group)
            self.free_group = []
            self.free(merged)


class SharedMambaTokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):
    """Composite allocator for the MHA (full-attn) + Mamba hybrid pair over a
    `SharedMemoryPool`.

    The token-slot surface (the slot allocator the scheduler uses for the
    `out_cache_loc` path) delegates to the full-attn side — `alloc(N)` allocates
    MHA token slots (virtual per-token ids). The Mamba sub-pool's per-request
    `alloc(1)` is driven by `SharedHybridReqToTokenPool.alloc` -> `mamba_pool.alloc(1)`
    -> `mamba_allocator.alloc(1)`. Both sub-allocators are id-owners of their own
    granularity's virtual-id space (the spaces are independent).
    """

    def __init__(
        self,
        *,
        shared_buffer: SharedMemoryPool,
        kvcache,  # HybridLinearKVPool
        device: str,
        need_sort: bool = False,
        forward_stream: Optional[torch.cuda.Stream] = None,
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

        self.full_attn_allocator = MultiEndedAllocator(
            kvcache=kvcache.full_kv_pool,
            shared_buffer=shared_buffer,
            sub_pool_name="full",
            device=device,
            is_id_owner=True,
            need_sort=need_sort,
            forward_stream=forward_stream,
        )
        self.mamba_allocator = MultiEndedAllocator(
            kvcache=kvcache.mamba_pool,
            shared_buffer=shared_buffer,
            sub_pool_name="mamba",
            device=device,
            is_id_owner=True,
            need_sort=need_sort,
            forward_stream=forward_stream,
        )
        self.full_attn_allocator.bind_peer(self.mamba_allocator)
        self.mamba_allocator.bind_peer(self.full_attn_allocator)

        # Wire the pools to translate slot ids via their allocators.
        kvcache.full_kv_pool.attach_allocator(self.full_attn_allocator)
        kvcache.mamba_pool.attach_allocator(self.mamba_allocator)

        self.is_not_in_free_group = True
        self.free_group: List[torch.Tensor] = []
        # The base init left these as None; we use watermark math, not free-lists.
        self.free_pages = torch.empty(0, dtype=torch.int64, device=device)
        self.release_pages = torch.empty(0, dtype=torch.int64, device=device)

        logger.info(
            "[shared-pool] SharedMambaTokenToKVPoolAllocator ready: full max_slots=%d "
            "(min_slot_index=%d), mamba max_slots=%d (min_slot_index=%d), "
            "full_available=%d, mamba_available=%d",
            self.full_attn_allocator.max_slots,
            self.full_attn_allocator.min_slot_index,
            self.mamba_allocator.max_slots,
            self.mamba_allocator.min_slot_index,
            self.full_attn_allocator.available_size(),
            self.mamba_allocator.available_size(),
        )

    # -- size: dynamic (so the leak checker's num_used = size - available - evictable
    #    reduces to allocated - evictable) --
    @property
    def size(self) -> int:
        return (
            self.full_attn_allocator.available_size()
            + self.full_attn_allocator.allocated_count()
        )

    @size.setter
    def size(self, value) -> None:
        # Base.__init__ sets self.size = size; ignore — we compute it dynamically.
        pass

    # -- token-slot surface: MHA side --

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

    def translate_kv_loc(self, loc: torch.Tensor) -> torch.Tensor:
        """Full-pool VIRTUAL token ids -> physical slot ids (for the write/read
        paths). `-1` inputs map to `-1` via the trailing sentinel."""
        return self.full_attn_allocator.virtual_to_physical[loc]

    def is_slot_allocated(self, slot: int) -> bool:
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
        self.full_attn_allocator.clear_inverse_history()
        self.mamba_allocator.clear_inverse_history()

    def free_group_begin(self) -> None:
        self.is_not_in_free_group = False
        self.free_group = []

    def free_group_end(self) -> None:
        self.is_not_in_free_group = True
        if self.free_group:
            merged = torch.cat(self.free_group)
            self.free_group = []
            self.full_attn_allocator.free(merged)
            self.full_attn_allocator.clear_inverse_history()
            self.mamba_allocator.clear_inverse_history()

    def backup_state(self):
        return [self.full_attn_allocator.backup_state(), self.mamba_allocator.backup_state()]

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


class SharedSWATokenToKVPoolAllocator(SWATokenToKVPoolAllocator):
    """Composite allocator for the hybrid SWA pair (full + swa MHA sub-pools)
    over a `SharedMemoryPool`. Stage 2 of the shared-memory-pool feature.

    Inherits from `SWATokenToKVPoolAllocator` purely for the typing/contract
    relationship — `isinstance(allocator, SWATokenToKVPoolAllocator)` is
    asserted across SWARadixCache, schedule_batch, chunk_cache, and disagg.
    We do NOT call `SWATokenToKVPoolAllocator.__init__`: it would allocate two
    static-partition sub-pools (`TokenToKVPoolAllocator` over freshly created
    `MHATokenToKVPool` buffers), which is exactly what shared-pool replaces.
    Grand-parent `BaseTokenToKVPoolAllocator.__init__` is called directly.

    Three views on capacity (see `shared_memory_pool_design.md` §15.2 and the
    v1 retrospective at `old_design_and_impl/multi_ended_allocator.py:1133–1196`):

    - `available_size()`            : joint byte-budget, the only safe pre-check
                                      for `alloc(N)` because N slots cost N*
                                      (entry_full + entry_swa) bytes out of the
                                      shared gap.
    - `full_available_size()` /
      `swa_available_size()`        : slot-conservation, for the leak invariant.
                                      Static cap − allocated_count.
    - `schedulable_full_available_size()` /
      `schedulable_swa_available_size()` : byte-coordinated, for the scheduler's
                                           alloc planner — may be smaller than
                                           the slot-conservation view.
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
        device: str,
        full_max_total_num_tokens: int,
        swa_max_total_num_tokens: int,
        need_sort: bool = False,
        forward_stream: Optional[torch.cuda.Stream] = None,
    ):
        # Set _size_full / _size_swa BEFORE base init so anything that reads
        # `self.size` / `self.size_full` / `self.size_swa` during base init
        # sees a sane value. Stored as the STATIC partition caps — this is
        # the value the leak invariant expects (slot-conservation, not
        # dynamic / byte-coordinated). See v1 lines 1158–1166 for why.
        self._size_full = full_max_total_num_tokens
        self._size_swa = swa_max_total_num_tokens
        self._full_max_total_num_tokens = full_max_total_num_tokens
        self._swa_max_total_num_tokens = swa_max_total_num_tokens

        # Skip SWATokenToKVPoolAllocator.__init__ — call grand-parent base init
        # directly. The base's `self.size = size` call is absorbed by our
        # no-op size setter above.
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

        self.full_attn_allocator = MultiEndedAllocator(
            kvcache=kvcache.full_kv_pool,
            shared_buffer=shared_buffer,
            sub_pool_name="full",
            device=device,
            is_id_owner=True,
            need_sort=need_sort,
            forward_stream=forward_stream,
        )
        self.swa_attn_allocator = MultiEndedAllocator(
            kvcache=kvcache.swa_kv_pool,
            shared_buffer=shared_buffer,
            sub_pool_name="swa",
            device=device,
            is_id_owner=False,  # ← non-owner; consumes virtuals minted by full.
            need_sort=need_sort,
            forward_stream=forward_stream,
        )
        self.full_attn_allocator.bind_peer(self.swa_attn_allocator)
        self.swa_attn_allocator.bind_peer(self.full_attn_allocator)

        # Wire the pools to translate slot ids via their allocators.
        kvcache.full_kv_pool.attach_allocator(self.full_attn_allocator)
        kvcache.swa_kv_pool.attach_allocator(self.swa_attn_allocator)
        kvcache.attach_allocators(
            full=self.full_attn_allocator, swa=self.swa_attn_allocator
        )

        self.is_not_in_free_group = True
        self.free_group: List[torch.Tensor] = []
        # Empty (not None) for the leak checker — same as Mamba composite.
        self.free_pages = torch.empty(0, dtype=torch.int64, device=device)
        self.release_pages = torch.empty(0, dtype=torch.int64, device=device)

        logger.info(
            "[shared-pool] SharedSWATokenToKVPoolAllocator ready: "
            "full max_slots=%d (min_slot_index=%d, entry_bytes=%d), "
            "swa max_slots=%d (min_slot_index=%d, entry_bytes=%d), "
            "static caps full=%d swa=%d, joint available=%d",
            self.full_attn_allocator.max_slots,
            self.full_attn_allocator.min_slot_index,
            self.full_attn_allocator.entry_bytes,
            self.swa_attn_allocator.max_slots,
            self.swa_attn_allocator.min_slot_index,
            self.swa_attn_allocator.entry_bytes,
            self._full_max_total_num_tokens,
            self._swa_max_total_num_tokens,
            self.available_size(),
        )

    # -- capacity reporting (three-way split per v1 lessons #1–#3) --

    def available_size(self) -> int:
        """Max N such that `alloc(N)` succeeds.

        Joint byte-budget: a single `alloc(N)` costs
        `N * (entry_full + entry_swa)` bytes out of the shared byte gap. The
        naive `min(full.available, swa.available)` uses ONE entry size and
        overshoots — the pre-check passes, then mid-alloc the peer's byte
        frontier has shifted. Required even when entries are equal because
        the gap is shared. (v1 lesson #1: lines 1133–1156.)
        """
        fa, sa = self.full_attn_allocator, self.swa_attn_allocator
        entry_sum = fa.entry_bytes + sa.entry_bytes
        # The grow-up full's high frontier and grow-down swa's low frontier
        # together delimit the unused byte gap.
        full_high = fa._byte_high_frontier()
        swa_low = sa._byte_low_frontier()
        gap_bytes = max(0, swa_low - full_high)
        slots_by_bytes = gap_bytes // entry_sum
        full_room = fa.max_slots - fa.min_slot_index - fa.allocated_count()
        swa_room = sa.max_slots - sa.min_slot_index - sa.allocated_count()
        return min(slots_by_bytes, full_room, swa_room)

    # Slot-conservation views — the only views the leak invariant should see.
    # Under shared SWA, the swa side can consume bytes that originally counted
    # toward the full side's static budget. Returning the byte-coordinated
    # (dynamic, peer-aware) value here would generate spurious leak
    # detections. (v1 lesson #2: lines 1158–1177.)
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

    # Byte-coordinated views — used by the scheduler's alloc planner
    # (`schedule_policy` etc.). On the non-shared `SWATokenToKVPoolAllocator`
    # these methods alias the static views (no peer coupling). The split
    # only matters under shared pool. (v1 lesson #3.)
    def schedulable_full_available_size(self) -> int:
        return self.full_attn_allocator.available_size()

    def schedulable_swa_available_size(self) -> int:
        return self.swa_attn_allocator.available_size()

    # `size_full` / `size_swa` are inherited from `SWATokenToKVPoolAllocator`
    # — they read `_size_full` / `_size_swa`, which we set to the static
    # `full_max_total_num_tokens` / `swa_max_total_num_tokens` in __init__.
    # We deliberately do NOT report `max_slots - 1` here: under shared pool
    # `max_slots("full") ≈ full_max + swa_max`, which would over-promise to
    # any caller treating these as static budgets.

    def debug_print(self) -> str:
        return (
            f"#full-available={self.full_attn_allocator.available_size()}, "
            f"#swa-available={self.swa_attn_allocator.available_size()}, "
            f"#joint-available={self.available_size()}"
        )

    def get_kvcache(self):
        return self._kvcache

    def translate_kv_loc(self, loc: torch.Tensor) -> torch.Tensor:
        """Full-layer read path: virtual token ids -> full-physical slot ids.
        `-1` inputs map to `-1` via the trailing sentinel."""
        return self.full_attn_allocator.virtual_to_physical[loc]

    def translate_loc_from_full_to_swa(
        self, kv_indices: torch.Tensor
    ) -> torch.Tensor:
        """SWA-layer read path: virtual token ids -> swa-physical slot ids.
        Note: the input semantics differ from the non-shared
        `SWATokenToKVPoolAllocator.translate_loc_from_full_to_swa`
        (which takes full-physical), but the output semantics (swa-physical
        int32) match — downstream consumers don't care."""
        return self.swa_attn_allocator.virtual_to_physical[kv_indices].to(
            torch.int32
        )

    # -- alloc --

    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        # Joint pre-check (v1 lesson #1) — accounts for byte pressure across
        # both sub-pools and both index-space headrooms.
        if need_size > self.available_size():
            return None
        v = self.full_attn_allocator.alloc(need_size)
        if v is None:
            return None
        try:
            self.swa_attn_allocator.alloc_with_virtual(v)
        except AssertionError:
            # Should be unreachable after the joint pre-check above; defensive
            # rollback so composite state stays coherent. (v1 lesson #5.)
            self._rollback_full_alloc(v)
            return None
        return v

    def _rollback_full_alloc(self, v: torch.Tensor) -> None:
        """Undo a `full_attn_allocator.alloc(need_size)` whose paired swa-side
        `alloc_with_virtual` could not complete. Reverse the full-side
        watermark, clear the v2p / p2v bindings, and push the minted virtual
        ids back to the head of the free list. Symmetric to `alloc`'s order."""
        fa = self.full_attn_allocator
        need_size = int(v.numel())
        if need_size == 0:
            return
        # Reverse the bind.
        phys = fa.virtual_to_physical[v].clone()
        fa.virtual_to_physical[v] = -1
        fa.physical_to_virtual[phys] = -1
        # Reverse the watermark advance.
        if fa.grow_direction == "up":
            fa.watermark_physical -= need_size
        else:
            fa.watermark_physical += need_size
        # Recycle the virtuals to the head of the free list (front, so the
        # next alloc reuses them — matches the slice-from-front consumption
        # in `alloc`).
        fa.free_virtual_ids = torch.cat([v, fa.free_virtual_ids])

    def is_slot_allocated(self, slot: int) -> bool:
        """Token-slot surface = the full side. SWARadixCache passes virtual
        ids (which the full sub-allocator owns) to validate before free."""
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
        # Free both peers. swa first (non-owner — only releases swa-physical;
        # doesn't touch the virtual id), then full (id-owner — recycles the
        # virtual id). Order is not load-bearing for correctness in v2 (no
        # cross-pool mapping coherence to maintain — there is no
        # `full_to_swa_index_mapping`; the per-sub-pool v2p IS the mapping).
        #
        # Filter the swa side to skip already-tombstoned virtuals (where
        # `swa.v2p[v] == -1` because `free_swa(v)` ran earlier). Mirrors the
        # v1 `swa_indices > 0` filter at `old_design_and_impl/...:1387`.
        # The full side does NOT need this filter — under SWARadixCache the
        # full side is the lifecycle owner, so every value in `free_index`
        # must still be bound on full.
        v = free_index.detach().to(torch.int64)
        swa_phys = self.swa_attn_allocator.virtual_to_physical[v]
        live_swa = v[swa_phys > 0]
        if live_swa.numel() > 0:
            self.swa_attn_allocator.free(live_swa)
        self.full_attn_allocator.free(v)
        self.full_attn_allocator.clear_inverse_history()
        self.swa_attn_allocator.clear_inverse_history()

    def free_swa(self, free_index: torch.Tensor) -> None:
        """SWA tombstone path: swa-physical released, virtual id and
        full-physical stay live.

        Mirrors `SWATokenToKVPoolAllocator.free_swa`. Called by
        `SWARadixCache._evict_swa_only` when a tree node has aged past the
        sliding-window horizon — its swa state is no longer reachable but
        its full state still is, so the swa-side budget gets reclaimed
        without disturbing full bookkeeping. In v2 the SWA allocator's
        `virtual_to_physical[v] = -1` after this call IS the tombstone."""
        if free_index is None or free_index.numel() == 0:
            return
        # Filter to virtuals that still have an swa-side binding — under v2,
        # `swa.v2p[v] == -1` means already-tombstoned; calling `swa.free` on
        # those would assert. (Mirrors the non-shared `free_swa`'s
        # `swa_indices > 0` filter on its `full_to_swa_index_mapping`.)
        v = free_index.detach().to(torch.int64)
        # `> 0` (strict): tombstoned entries have `swa.v2p[v] == -1`; virtual
        # id 0 is the padding sink bound to physical 0 — never freeable.
        # Mirrors the non-shared `free_swa`'s `swa_indices > 0` filter
        # (`swa_memory_pool.py:502`).
        swa_phys = self.swa_attn_allocator.virtual_to_physical[v]
        live = v[swa_phys > 0]
        if live.numel() == 0:
            return
        self.swa_attn_allocator.free(live)
        self.swa_attn_allocator.clear_inverse_history()

    def set_full_to_swa_mapping(
        self, full_indices: torch.Tensor, swa_indices: torch.Tensor
    ) -> None:
        """No-op stub for HiCache load-back compatibility.

        On the non-shared `SWATokenToKVPoolAllocator`, this rewrites the
        `full_to_swa_index_mapping` after HiCache reallocates full + swa
        slots. In shared mode there is no mapping tensor — the swa
        sub-allocator's v2p table IS the mapping, and `alloc()` populates
        it automatically. HiCache for shared SWA is out of scope for Stage 2.
        """
        # HiCache for the shared SWA path is a Stage-2 follow-up; this stub
        # keeps the non-shared API surface compatible.
        return

    # -- free-group --

    def free_group_begin(self) -> None:
        self.is_not_in_free_group = False
        self.free_group = []

    def free_group_end(self) -> None:
        self.is_not_in_free_group = True
        if self.free_group:
            merged = torch.cat(self.free_group)
            self.free_group = []
            self.free(merged)

    # -- spec-decode hooks (asserted off in Stage 2; preserved for Stage 5) --

    def backup_state(self):
        return [
            self.full_attn_allocator.backup_state(),
            self.swa_attn_allocator.backup_state(),
        ]

    def restore_state(self, state):
        assert len(state) == 2
        full_rollback = self.full_attn_allocator.restore_state(state[0])
        swa_rollback = self.swa_attn_allocator.restore_state(state[1])
        return full_rollback + swa_rollback

    def clear(self) -> None:
        self.full_attn_allocator.clear()
        self.swa_attn_allocator.clear()
        self.is_not_in_free_group = True
        self.free_group = []
