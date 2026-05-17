"""MultiEndedAllocator: one allocator per sub-pool over a `SharedMemoryPool`.

See `shared_memory_pool_design.md` (same directory). Key points for v2:

* Each `MultiEndedAllocator` owns: a physical watermark (grows up from
  `min_page_index`, or down from `num_virtual_pages-1`), per-sub-pool
  `virtual_to_physical` / `physical_to_virtual` tables (sized by PAGES),
  and — iff it is the *id-owner* of its virtual-id granularity — the
  `free_virtual_ids` free-list (also page-granular).
* `alloc(N)` (id-owner only, N must be page-aligned): pop N/page_size virtual
  pages, take N/page_size physical pages, bind, return token IDs.
  `alloc_with_virtual(virtual_pages)` (physical-holding non-owner): take
  physical pages for caller-supplied virtual page ids, bind.
  `alloc_extend` / `alloc_decode`: call the upstream `alloc_extend_kernel` /
  `alloc_decode_kernel` ONCE in virtual space using `free_virtual_ids` as the
  free-page pointer; emit virtual token ids that respect the tail-page-reuse
  contract. `free(virtual_token_ids)`: recover page ids via
  `unique(// page_size)`, un-map, eager-compact whole pages, (if id-owner)
  recycle the virtual page ids.
* Eager compaction touches **only** `virtual_to_physical` /
  `physical_to_virtual` page tables (O(num_relocations_pages) scalar ops) —
  no reference rewriting, no binder. Compaction's `move_kv_cache` call
  expands page ids to token ids before invoking the token-granular move.
* **Token IDs vs Page IDs on the surface**: every public method takes/returns
  TOKEN-granular tensors (matching `PagedTokenToKVPoolAllocator`'s contract).
  Only the internal `free_virtual_ids` list and the v2p/p2v tables are
  page-granular.
* For `page_size == 1` (Stages 1/2), behavior is byte-identical: a "page" is
  a single slot, and all the page math collapses to slot math.

Concurrency: the scheduler runs alloc/free serially; no mutex is taken here.
"""

from __future__ import annotations

import inspect
import logging
from typing import List, Optional, Tuple

import torch

from sglang.srt.mem_cache.allocator import (
    BaseTokenToKVPoolAllocator,
    alloc_decode_kernel,
    alloc_extend_kernel,
)
from sglang.srt.mem_cache.shared_memory_pool import SharedMemoryPool
from sglang.srt.mem_cache.swa_memory_pool import SWATokenToKVPoolAllocator
from sglang.srt.utils.common import get_num_new_pages, next_power_of_2

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
        page_size: int = 1,
        need_sort: bool = False,
        forward_stream: Optional[torch.cuda.Stream] = None,
    ):
        spec = shared_buffer.spec(sub_pool_name)
        max_slots = shared_buffer.max_slots(sub_pool_name)
        # `dtype` on the base allocator is informational. Each `SubPoolSpec`
        # subclass implements `get_dtype()` to return its representative
        # storage dtype (MHA: `store_dtype`; Mamba: `conv_dtype` — the
        # dominant state buffer's dtype). See `SubPoolSpec.get_dtype`.
        super().__init__(
            size=max_slots,
            page_size=page_size,
            dtype=spec.get_dtype(),
            device=device,
            kvcache=kvcache,
            need_sort=need_sort,
        )
        self.shared_buffer = shared_buffer
        self.sub_pool_name = sub_pool_name
        self.spec = spec
        self.max_slots = max_slots
        self.grow_direction = spec.grow_direction
        # Per-token (per-slot) entry bytes — unchanged by paging.
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

        # --- Page-aware bookkeeping (Stage 3) ---
        #
        # When `page_size == 1`, num_virtual_pages == max_slots and
        # min_page_index == min_slot_index, so all the page math collapses
        # back to Stage 1/2 slot math (behavior byte-identical).
        #
        # When `page_size > 1`:
        # - `num_virtual_pages = max_slots // page_size` (truncate).
        # - `min_page_index = ceil(min_slot_index / page_size)` — the
        #   smallest page id that is fully outside the dummy-write reserved
        #   byte zone `[0, entry_max)`. The "Dummy-write safety proof" in
        #   the Stage-3 plan shows this preserves the Stage-1 invariant:
        #       min_page_index * entry_bytes_per_page
        #       ≥ min_slot_index * entry_bytes
        #       ≥ entry_max.
        # - `entry_bytes_per_page = entry_bytes * page_size` — used by the
        #   joint byte-budget check on the SWA composite.
        self.page_size = page_size
        self.num_virtual_pages = max_slots // page_size
        self.min_page_index = (
            self.min_slot_index + page_size - 1
        ) // page_size  # ceil
        self.entry_bytes_per_page = self.entry_bytes * page_size

        # v -> p, sized by PAGES (not slots). Page id 0 ↔ page 0 is the
        # dummy-padding anchor; trailing `[-1]` row is the `-1` sentinel.
        self.virtual_to_physical = torch.full(
            (self.num_virtual_pages + 1,),
            -1,
            dtype=torch.int64,
            device=device,
        )
        # p -> v, also sized by PAGES.
        self.physical_to_virtual = torch.full(
            (self.num_virtual_pages + 1,),
            -1,
            dtype=torch.int64,
            device=device,
        )
        # Back-compat alias: `num_virtual_ids` was the Stage-1/2 name and is
        # still consulted by `is_slot_allocated` etc. Under Stage 3 it
        # represents the COUNT OF VIRTUAL PAGES (matches table sizing).
        self.num_virtual_ids = self.num_virtual_pages

        self._peer: Optional["MultiEndedAllocator"] = None

        # Inverse history of relocations (Stage-5 spec rollback). Each entry is
        # one batch (src_phys_tensor, dst_phys_tensor, v_moved_tensor) — all
        # at PAGE granularity. The composite calls `clear_inverse_history`
        # after each `free` so it stays bounded.
        self._inverse_history: List[
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] = []

        self.clear()

        logger.info(
            "[shared-pool] MultiEndedAllocator(%r) ready: grow=%s, max_slots=%d, "
            "min_slot_index=%d, page_size=%d, num_virtual_pages=%d, min_page_index=%d, "
            "entry_bytes=%d, entry_bytes_per_page=%d, is_id_owner=%s, "
            "initial_watermark_page=%d, allocatable_pages=%d",
            self.sub_pool_name,
            self.grow_direction,
            self.max_slots,
            self.min_slot_index,
            self.page_size,
            self.num_virtual_pages,
            self.min_page_index,
            self.entry_bytes,
            self.entry_bytes_per_page,
            self.is_id_owner,
            self.watermark_physical,
            self.num_virtual_pages - self.min_page_index,
        )

    # -- peer binding --

    def bind_peer(self, peer: "MultiEndedAllocator") -> None:
        self._peer = peer

    @property
    def peer(self) -> Optional["MultiEndedAllocator"]:
        return self._peer

    # -- state --

    def clear(self) -> None:
        """Reset to initial state.

        Watermark and free-list are at PAGE granularity. Page 0 is the
        padding anchor (`virtual_to_physical[0] = 0` ↔ token 0 = dummy sink).
        Page ids in `[0, min_page_index)` are reserved (see "Dummy-write
        safety proof" in the Stage-3 plan).
        """
        if self.grow_direction == "up":
            self.watermark_physical = self.min_page_index
        else:
            self.watermark_physical = self.num_virtual_pages - 1
        self.virtual_to_physical.fill_(-1)
        # Virtual PAGE 0 ↔ physical PAGE 0 (padding sink page). Within page 0,
        # only token 0 is the dummy-write target; tokens 1..page_size-1 in
        # page 0 are reserved but never written to (allocator never emits
        # them since min_page_index ≥ 1).
        self.virtual_to_physical[0] = 0
        self.virtual_to_physical[-1] = -1  # trailing sentinel
        self.physical_to_virtual.fill_(-1)
        self.physical_to_virtual[0] = 0
        self.physical_to_virtual[-1] = -1
        if self.is_id_owner:
            # Virtual pages 0..min_page_index-1 are reserved; trailing
            # sentinel row is not in the free list. For page_size == 1,
            # this collapses to `arange(min_slot_index, max_slots)`.
            self.free_virtual_ids = torch.arange(
                self.min_page_index,
                self.num_virtual_pages,
                dtype=torch.int64,
                device=self.device,
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

    def _allocated_pages(self) -> int:
        """Internal: number of allocated PAGES (page-granular math).

        Used by `available_size()` and the SWA composite to compute the
        index-space headroom in PAGE units. Callers that need TOKEN units
        must use `allocated_count()`.
        """
        if self.grow_direction == "up":
            return max(0, self.watermark_physical - self.min_page_index)
        return max(0, self.num_virtual_pages - 1 - self.watermark_physical)

    def allocated_count(self) -> int:
        """Public: number of allocated TOKENS.

        Matches the upstream convention that all external-facing capacity
        methods report tokens — cf. ``BaseTokenToKVPoolAllocator.available_size``
        which returns ``len(free_pages) * page_size``. For ``page_size == 1``
        this is identical to ``_allocated_pages()`` (Stage 1/2 behavior).

        The leak invariant in the scheduler runtime checker
        (``_check_swa_pool`` / ``_check_full_pool``) is:
        ``total_TOKENS == available_TOKENS + allocated_TOKENS + ...``.
        Returning pages here (as Stage 3 did before this fix) caused the
        "pool memory leak detected" crash in eval_results_15.
        """
        return self._allocated_pages() * self.page_size

    def is_slot_allocated(self, slot: int) -> bool:
        """Whether the PAGE containing this token-level virtual id is in use.

        The `slot` argument is a TOKEN-granular virtual id (matching the
        Stage-1/2 API). We recover the virtual page and look it up.
        """
        # Recover the virtual page from the token-granular virtual id. For
        # page_size == 1, virt_page == slot (no change in behavior).
        virt_page = slot // self.page_size
        if virt_page < 0 or virt_page >= self.num_virtual_pages:
            return False
        return int(self.virtual_to_physical[virt_page].item()) != -1

    def allocator_state_str(self) -> str:
        return (
            f"sub_pool={self.sub_pool_name!r}, grow_direction={self.grow_direction}, "
            f"is_id_owner={self.is_id_owner}, page_size={self.page_size}, "
            f"min_page_index={self.min_page_index}, "
            f"num_virtual_pages={self.num_virtual_pages}, "
            f"watermark_physical={self.watermark_physical}, "
            f"allocated_pages={self._allocated_pages()}"
        )

    def _byte_high_frontier(self) -> int:
        """Byte just past this side's last-allocated page (grow-up) /
        the buffer's top (grow-down)."""
        if self.grow_direction == "up":
            return self.watermark_physical * self.entry_bytes_per_page
        return self.num_virtual_pages * self.entry_bytes_per_page

    def _byte_low_frontier(self) -> int:
        """Byte that begins this side's allocatable range (grow-up) /
        first byte just below this side's lowest live page (grow-down)."""
        if self.grow_direction == "up":
            return self.min_page_index * self.entry_bytes_per_page
        return (self.watermark_physical + 1) * self.entry_bytes_per_page

    def available_size(self) -> int:
        """Tokens (NOT pages) allocatable on this side.

        Matches `BaseTokenToKVPoolAllocator.available_size()`'s
        `len(free_pages) * page_size` contract. Used by the scheduler's
        `available_size() >= num_tokens` checks (`schedule_batch.py:2157`,
        etc.).
        """
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
        pages_by_bytes = gap_bytes // self.entry_bytes_per_page
        # `_allocated_pages()` (page-granular) — the index-space headroom
        # is a page-count, NOT a token-count. (We multiply by page_size at
        # the return statement, the single external token boundary.)
        pages_by_index_space = (
            self.num_virtual_pages - self.min_page_index - self._allocated_pages()
        )
        pages = min(pages_by_bytes, pages_by_index_space)
        return pages * self.page_size

    # -- physical-slot / physical-page primitives --

    def take_physical(self, need_size: int) -> Optional[torch.Tensor]:
        """Advance the physical watermark by ``need_size`` TOKENS; return the
        newly-allocated physical PAGE ids.

        ``need_size`` must be a multiple of ``page_size`` (asserted). Returns
        ``None`` if the watermark would over/underflow the page-index
        headroom (defensive backstop — the byte-frontier check in
        ``available_size`` should normally have caught this earlier, but the
        joint byte-budget in the SWA composite makes a stale-state edge case
        possible, and Stage 1 already has the symmetric guard on grow-down).
        """
        if need_size <= 0:
            return torch.empty(0, dtype=torch.int64, device=self.device)
        assert need_size % self.page_size == 0, (
            f"take_physical: need_size={need_size} must be a multiple of "
            f"page_size={self.page_size}"
        )
        num_pages = need_size // self.page_size
        if self.grow_direction == "up":
            start = self.watermark_physical
            end_exclusive = start + num_pages
            # Defensive overflow check — Stage 3 added the symmetric guard
            # for grow-up (Stage 1 only had it for grow-down).
            if end_exclusive > self.num_virtual_pages:
                return None
            phys_pages = torch.arange(
                start, end_exclusive, dtype=torch.int64, device=self.device
            )
            self.watermark_physical = end_exclusive
            return phys_pages
        else:
            end = self.watermark_physical
            start = end - num_pages + 1
            if start < self.min_page_index:
                return None
            phys_pages = torch.arange(
                start, end + 1, dtype=torch.int64, device=self.device
            )
            self.watermark_physical -= num_pages
            return phys_pages

    def take_physical_pages(self, num_pages: int) -> Optional[torch.Tensor]:
        """Page-granular wrapper around ``take_physical``. Used by the SWA
        composite's ``alloc_extend`` to bind the non-owner side."""
        return self.take_physical(num_pages * self.page_size)

    def bind(
        self, virtual_ids: torch.Tensor, physical_ids: torch.Tensor
    ) -> None:
        """Bind page-granular virtual ids to page-granular physical ids.

        For page_size == 1, virtual_ids and physical_ids are slot ids
        (behavior matches Stage 1/2 exactly). For page_size > 1, both are
        PAGE ids — the v2p / p2v tables are page-granular.
        """
        self.virtual_to_physical[virtual_ids] = physical_ids
        self.physical_to_virtual[physical_ids] = virtual_ids

    def bind_pages(
        self, virtual_pages: torch.Tensor, physical_pages: torch.Tensor
    ) -> None:
        """Explicit page-granular binder. Alias of ``bind`` — kept distinct
        for readability at the SWA composite call site."""
        self.bind(virtual_pages, physical_pages)

    # -- translate (virtual TOKEN ids -> physical TOKEN ids) --

    def translate_kv_loc(
        self,
        virt_tokens: torch.Tensor,
        *,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Translate token-granular virtual ids to token-granular physical ids.

        Stage 3.5: the ``out=`` parameter writes results in-place into a
        caller-owned buffer. Required under cuda-graph capture for
        buffer-stability — the captured graph records the gather/multiply/add
        sequence against a fixed ``data_ptr``, replay re-runs against the
        same buffer.

        Args:
            virt_tokens: int64[N] virtual token ids (page-structured).
            out: optional int64[N] output tensor. When ``None`` (default),
                returns a fresh tensor — byte-identical to Stage 1/2/3
                behavior. When provided, writes in-place and returns ``out``.

        Returns:
            int64[N] physical token ids. If ``out`` was given, returns ``out``.
        """
        if out is not None:
            assert out.dtype == torch.int64, (
                f"translate_kv_loc: out= dtype must be int64 (matches v2p), "
                f"got {out.dtype}"
            )
            assert out.shape == virt_tokens.shape, (
                f"translate_kv_loc: out= shape {tuple(out.shape)} must match "
                f"virt_tokens shape {tuple(virt_tokens.shape)}"
            )
        # Tombstone-safety clamp: tombstoned `v2p` entries (-1) must not reach
        # `k_buffer[-1]` when read by the captured graph. Under cuda-graph
        # capture, this method is called eagerly each replay-prep to populate
        # capture-stable buffers (`out_cache_loc_full_physical`,
        # `cuda_graph_kv_indices`); the captured kernels then index k/v
        # buffers with those values. A captured `k_buffer[-1]` is an illegal
        # memory access and crashes `graph.replay()`.
        #
        # Negative outputs can arise from:
        #   - padded-tail positions in the captured buffer whose stale virtual
        #     ids point at pages that have since been tombstoned by free /
        #     `_compact_pending`;
        #   - the zero-clear sentinel positions in `bs != raw_bs` replays
        #     (these are 0 -> v2p[0] = 0 -> 0; clamp is a no-op for those).
        # Clamping to 0 routes any tombstoned read/write to physical slot 0,
        # which is reserved padding-sink space by Stage 1's `min_slot_index`
        # invariant: bytes `[0, entry_max)` across all sub-pools hold no real
        # data (see §S3 dummy-write safety proof). Cost: one elementwise op
        # per call; safe.
        if self.page_size == 1:
            if out is not None:
                # CRITICAL: `torch.index_select(src, dim, index, out=out)`
                # does NOT support aliasing between `index` and `out`. The
                # canonical caller from `triton_backend.py` is
                # `self._translate_kv_loc(kv_indices, out=kv_indices)`, where
                # `virt_tokens` and `out` are the SAME buffer (in-place
                # translate). Route through a transient gather + in-place
                # `copy_` to satisfy index_select's no-aliasing contract.
                # The transient `tmp` is fresh per call but caching-allocator-
                # cached under cuda-graph capture; the observable mutation is
                # `out.copy_(tmp)` into the stable buffer.
                tmp = torch.index_select(
                    self.virtual_to_physical, 0, virt_tokens
                )
                tmp = torch.clamp_min(tmp, 0)
                out.copy_(tmp)
                return out
            result = torch.index_select(
                self.virtual_to_physical, 0, virt_tokens
            )
            return torch.clamp_min(result, 0)
        # page_size > 1: page math.
        # Note: `virt_pages` and `offsets` are fresh tensors (results of
        # `// page_size` and `% page_size`), so they cannot alias `out`. The
        # `index_select(out=out)` below is therefore safe even when `out`
        # aliases `virt_tokens`.
        virt_pages = virt_tokens // self.page_size  # fresh int64[N]
        offsets = virt_tokens % self.page_size  # fresh int64[N]
        if out is not None:
            torch.index_select(
                self.virtual_to_physical, 0, virt_pages, out=out
            )
            out.mul_(self.page_size)
            out.add_(offsets)
            # Tombstoned page: -1 * ps + offset is in [-ps, -1]; clamp to 0.
            out.clamp_(min=0)
            return out
        phys_pages = self.virtual_to_physical[virt_pages]
        result = phys_pages * self.page_size + offsets
        return torch.clamp_min(result, 0)

    # -- alloc --

    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        """Allocate `need_size` virtual TOKEN ids (id-owner only). Returns
        the virtual token ids (token-granular, page-structured), or None if
        there is not enough byte room / page headroom.

        Contract (matches upstream ``PagedTokenToKVPoolAllocator.alloc``
        at ``allocator.py:386–407``): ``need_size`` MUST be a multiple of
        ``page_size``. Token ids are emitted as
        ``(virtual_pages[:, None] * page_size + arange(page_size)).reshape(-1)``.

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
        assert need_size % self.page_size == 0, (
            f"MultiEndedAllocator({self.sub_pool_name!r}).alloc: need_size="
            f"{need_size} must be a multiple of page_size={self.page_size}"
        )
        if need_size > self.available_size():
            return None
        num_pages = need_size // self.page_size
        v_pages = self.free_virtual_ids[:num_pages]
        self.free_virtual_ids = self.free_virtual_ids[num_pages:]
        phys_pages = self.take_physical_pages(num_pages)
        if phys_pages is None:
            # Undo the virtual pop.
            self.free_virtual_ids = torch.cat([v_pages, self.free_virtual_ids])
            return None
        self.bind_pages(v_pages, phys_pages)
        if self.page_size == 1:
            # Avoid the extra reshape — v_pages already IS the token id list.
            return v_pages
        # Expand page ids to token ids: (P, 1) * S + (S,) → (P, S) → (P*S,).
        return (
            v_pages[:, None] * self.page_size
            + torch.arange(self.page_size, device=self.device)
        ).reshape(-1)

    def alloc_with_virtual(self, virtual_pages: torch.Tensor) -> None:
        """Take physical PAGES for caller-supplied virtual PAGE ids
        (physical-holding non-owner). Used by the SWA `swa` sub-allocator
        from Stage 2 onward.

        Note: under Stage 3 the input is **virtual page ids** (not token ids),
        matching the composite's ``alloc_extend`` design where the kernel
        produces virtual token ids and the composite snapshots the
        corresponding virtual page ids before consuming them from the
        id-owner's free-list.
        """
        if virtual_pages.numel() == 0:
            return
        phys_pages = self.take_physical_pages(int(virtual_pages.numel()))
        assert phys_pages is not None, (
            f"MultiEndedAllocator({self.sub_pool_name!r}).alloc_with_virtual: out of "
            "physical room (the composite's byte-budget check should have caught this)"
        )
        self.bind_pages(virtual_pages, phys_pages)

    # -- paged alloc surface (Stage 3) --

    def alloc_extend(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        extend_num_tokens: int,
        num_new_pages: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        """Allocate ``extend_num_tokens`` new tokens across ``bs`` requests,
        preserving the tail-page-reuse contract.

        Mirrors ``PagedTokenToKVPoolAllocator.alloc_extend``
        (``allocator.py:409–457``) but operates in **virtual space**: the
        kernel's ``free_page_ptr`` is this allocator's ``free_virtual_ids``
        (virtual pages, not physical), so the emitted ``out_indices`` are
        **virtual token ids** that respect ``(last_loc + 1) % page_size ==
        prefix_lens % page_size`` in virtual space.

        Each consumed virtual page is also bound to a physical page on THIS
        sub-allocator (via ``take_physical_pages`` + ``bind_pages``) so that
        downstream ``translate_kv_loc(virt_token)`` resolves to a valid
        physical token id. Without this binding, ``v2p[virt_page]`` would
        stay ``-1`` and translation would produce negative token ids that
        crash the Triton attention kernel with a CUDA OOB.

        Peers (e.g., the swa side of the SWA composite) call
        ``alloc_with_virtual(new_virtual_pages)`` to bind their own physical
        pages to the same virtual pages (the SWA composite handles this).
        """
        assert self.is_id_owner, (
            f"alloc_extend on a non-id-owner allocator ({self.sub_pool_name!r})"
        )
        if num_new_pages is None:
            num_new_pages = get_num_new_pages(
                seq_lens=seq_lens_cpu,
                page_size=self.page_size,
                prefix_lens=prefix_lens_cpu,
            )
        if num_new_pages > len(self.free_virtual_ids):
            return None
        bs = len(prefix_lens)
        if self.need_sort and extend_num_tokens // self.page_size + bs + 1 > len(
            self.free_virtual_ids
        ):
            self.merge_and_sort_free()

        # Snapshot the virtual pages the kernel is about to consume, so we
        # can bind them to physical pages on THIS sub-allocator afterward.
        # (eval_results_14 regression: without this, v2p stayed -1 and
        # `translate_kv_loc` returned negative token ids → CUDA OOB.)
        if num_new_pages > 0:
            new_virtual_pages = self.free_virtual_ids[:num_new_pages].clone()
        else:
            new_virtual_pages = None

        out_indices = torch.empty(
            (extend_num_tokens,), dtype=torch.int64, device=self.device
        )
        # Pass `free_virtual_ids` (virtual pages) as `free_page_ptr` — the
        # kernel just does `page_id * page_size + offset` math and doesn't
        # care whether page ids are virtual or physical.
        alloc_extend_kernel[(bs,)](
            prefix_lens,
            seq_lens,
            last_loc,
            self.free_virtual_ids,
            out_indices,
            next_power_of_2(bs),
            self.page_size,
        )

        # Bind the consumed virtual pages to fresh physical pages on this
        # sub-allocator. Advances the watermark + sets v2p / p2v. The peer
        # (swa side, if any) does its own binding via `alloc_with_virtual`.
        if new_virtual_pages is not None:
            phys_pages = self.take_physical_pages(num_new_pages)
            if phys_pages is None:
                # Defensive — the pre-check should have prevented this. Return
                # None so the composite can decide whether to roll back.
                return None
            self.bind_pages(new_virtual_pages, phys_pages)

        # Consume the new virtual pages from the free-list.
        self.free_virtual_ids = self.free_virtual_ids[num_new_pages:]
        return out_indices  # virtual token ids

    def alloc_decode(
        self,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Allocate one new token per request (decode step), preserving the
        tail-page-reuse contract.

        Mirrors ``PagedTokenToKVPoolAllocator.alloc_decode``
        (``allocator.py:459–496``) in virtual space. Each consumed virtual
        page is bound to a physical page on THIS sub-allocator afterward
        (same correctness requirement as ``alloc_extend``: without binding,
        v2p stays -1 and downstream translation produces negative token ids
        → CUDA OOB).
        """
        assert self.is_id_owner, (
            f"alloc_decode on a non-id-owner allocator ({self.sub_pool_name!r})"
        )
        bs = len(seq_lens)
        # Compute num_new_pages BEFORE the kernel so we can snapshot the
        # exact slice of `free_virtual_ids` the kernel will consume.
        # `get_num_new_pages` is CPU-only and matches the kernel's count.
        num_new_pages = get_num_new_pages(
            seq_lens=seq_lens_cpu, page_size=self.page_size, decode=True
        )
        if num_new_pages > len(self.free_virtual_ids):
            return None
        if self.need_sort and bs > len(self.free_virtual_ids):
            self.merge_and_sort_free()

        # Snapshot the virtual pages the kernel will consume (if any).
        # Most decode steps reuse the prefix's tail page → num_new_pages == 0.
        if num_new_pages > 0:
            new_virtual_pages = self.free_virtual_ids[:num_new_pages].clone()
        else:
            new_virtual_pages = None

        out_indices = torch.empty(
            (bs,), dtype=torch.int64, device=self.device
        )
        alloc_decode_kernel[(bs,)](
            seq_lens,
            last_loc,
            self.free_virtual_ids,
            out_indices,
            next_power_of_2(bs),
            self.page_size,
        )

        # Bind the consumed virtual pages to fresh physical pages on this
        # sub-allocator. Advances the watermark + sets v2p / p2v.
        if new_virtual_pages is not None:
            phys_pages = self.take_physical_pages(num_new_pages)
            if phys_pages is None:
                return None
            self.bind_pages(new_virtual_pages, phys_pages)

        self.free_virtual_ids = self.free_virtual_ids[num_new_pages:]
        return out_indices  # virtual token ids

    # -- free with eager compaction --

    def free(self, free_index: torch.Tensor) -> None:
        """Free virtual TOKEN ids; recover virtual PAGE ids via
        ``unique(// page_size)``; un-map v2p / p2v at page granularity;
        (if id-owner) recycle the virtual page ids; trigger eager compaction.

        Mirrors ``PagedTokenToKVPoolAllocator.free``
        (``allocator.py:498–512``) — ``free_index`` is token-granular and
        does NOT need to be page-aligned; the caller-side invariant is that
        a token in a page is freed iff the page is no longer referenced.

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
        # Recover virtual PAGE ids from token-granular `free_index`. For
        # page_size == 1, `// page_size` is identity, so this collapses to
        # Stage 1/2 behavior byte-identically.
        free_v_pages = torch.unique(
            free_index.detach().to(torch.int64) // self.page_size
        )
        freed_p_pages = self.virtual_to_physical[free_v_pages]
        if bool((freed_p_pages < 0).any().item()):
            self._raise_stale_slot_assertion(
                free_v=free_v_pages, freed_p=freed_p_pages
            )
        # Un-map; (if id-owner) recycle the virtual page ids.
        self.virtual_to_physical[free_v_pages] = -1
        if self.is_id_owner:
            self.free_virtual_ids = torch.cat(
                [self.free_virtual_ids, free_v_pages]
            )
        self._compact_pending(freed_p_pages)

    def _compact_pending(self, freed_physical_pages: torch.Tensor) -> None:
        """Eager compaction over the freed PHYSICAL pages: move the survivor
        pages that fall in the *vacated band* (the K pages adjacent to the
        watermark, where K = #freed pages) into the *holes in the kept band*,
        then advance the watermark and remap the two tables. The `src` set
        (⊆ vacated band) and the `dst` set (⊆ kept band) are disjoint by
        construction, so the batched data copy is order-independent.

        `move_kv_cache` is called with TOKEN-granular indices (per-page
        tokens flattened) since `move_kv_cache_native` already operates at
        token granularity. For page_size == 1, src_pages/dst_pages ARE the
        token ids (no expansion).

        Separable so a future flag can defer it (lazy mode); in Stages 1/2/3
        `free` calls it inline (eager).

        All GPU ops here run on the scheduler thread's current stream
        (schedule_stream). The wait_stream barrier in the caller (`free`)
        already serialized us with any in-flight forward kernels."""
        freed_set = set(int(x) for x in freed_physical_pages.tolist())
        if not freed_set:
            return
        K = len(freed_set)
        if self.grow_direction == "up":
            # allocated == [min_page_index, old_wm); after the free == [min_page_index, new_wm)
            old_wm = self.watermark_physical
            new_wm = old_wm - K
            assert new_wm >= self.min_page_index, (
                f"_compact_pending({self.sub_pool_name!r}): freeing {K} pages "
                f"would push the watermark below min_page_index "
                f"({new_wm} < {self.min_page_index})"
            )
            assert all(self.min_page_index <= h < old_wm for h in freed_set), (
                f"_compact_pending({self.sub_pool_name!r}): freed physical pages "
                f"{sorted(freed_set)} not all within allocated range "
                f"[{self.min_page_index}, {old_wm})"
            )
            # vacated band = [new_wm, old_wm); kept band = [min_page_index, new_wm)
            src_list = [s for s in range(new_wm, old_wm) if s not in freed_set]
            dst_list = sorted(h for h in freed_set if h < new_wm)
            self.watermark_physical = new_wm
            vacated_lo, vacated_hi = new_wm, old_wm
        else:
            # allocated == (old_wm, num_virtual_pages); after the free == (new_wm, num_virtual_pages)
            old_wm = self.watermark_physical
            new_wm = old_wm + K
            assert new_wm <= self.num_virtual_pages - 1, (
                f"_compact_pending({self.sub_pool_name!r}): freeing {K} pages "
                f"would push the watermark above num_virtual_pages "
                f"({new_wm} > {self.num_virtual_pages - 1})"
            )
            assert all(old_wm < h < self.num_virtual_pages for h in freed_set), (
                f"_compact_pending({self.sub_pool_name!r}): freed physical pages "
                f"{sorted(freed_set)} not all within allocated range "
                f"({old_wm}, {self.num_virtual_pages})"
            )
            # vacated band = (old_wm, new_wm] = [old_wm+1, new_wm+1); kept band = (new_wm, num_virtual_pages)
            src_list = [s for s in range(old_wm + 1, new_wm + 1) if s not in freed_set]
            dst_list = sorted(h for h in freed_set if h > new_wm)
            self.watermark_physical = new_wm
            vacated_lo, vacated_hi = old_wm + 1, new_wm + 1

        assert len(src_list) == len(dst_list), (
            f"_compact_pending({self.sub_pool_name!r}): {len(src_list)} survivors vs "
            f"{len(dst_list)} holes — corrupt allocator state"
        )

        if src_list:
            src_pages = torch.tensor(
                src_list, dtype=torch.int64, device=self.device
            )
            dst_pages = torch.tensor(
                dst_list, dtype=torch.int64, device=self.device
            )
            v_moved = self.physical_to_virtual[src_pages].clone()  # read before clearing

            # Expand page ids to token ids for the move kernel (which is
            # token-granular, see memory_pool.py:2204 `move_kv_cache_native`).
            # For page_size == 1, src_pages/dst_pages == src_t/dst_t.
            if self.page_size == 1:
                src_t, dst_t = src_pages, dst_pages
            else:
                offsets = torch.arange(
                    self.page_size, dtype=torch.int64, device=self.device
                )
                src_t = (
                    src_pages[:, None] * self.page_size + offsets
                ).reshape(-1)
                dst_t = (
                    dst_pages[:, None] * self.page_size + offsets
                ).reshape(-1)

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
            # Clear the whole vacated band, then re-bind the relocated dst pages
            # (dst_pages ⊆ kept band, disjoint from the vacated band). All
            # remapping is at PAGE granularity (the tables are page-granular).
            self.physical_to_virtual[vacated_lo:vacated_hi] = -1
            self.virtual_to_physical[v_moved] = dst_pages
            self.physical_to_virtual[dst_pages] = v_moved
            self._inverse_history.append((src_pages, dst_pages, v_moved))
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
        page_size: int = 1,
        need_sort: bool = False,
        forward_stream: Optional[torch.cuda.Stream] = None,
    ):
        full_max = shared_buffer.max_slots("full")
        super().__init__(
            size=full_max - 1,
            page_size=page_size,
            dtype=shared_buffer.mha_spec("full").store_dtype,
            device=device,
            kvcache=kvcache,
            need_sort=need_sort,
        )
        self.shared_buffer = shared_buffer
        self._kvcache = kvcache
        self.page_size = page_size

        # FULL sub-allocator is page-aware (Stage 3). MAMBA sub-allocator
        # stays page_size=1 because the Mamba state is per-request (one slot
        # per req), orthogonal to the per-token paging on the full side.
        self.full_attn_allocator = MultiEndedAllocator(
            kvcache=kvcache.full_kv_pool,
            shared_buffer=shared_buffer,
            sub_pool_name="full",
            device=device,
            is_id_owner=True,
            page_size=page_size,
            need_sort=need_sort,
            forward_stream=forward_stream,
        )
        self.mamba_allocator = MultiEndedAllocator(
            kvcache=kvcache.mamba_pool,
            shared_buffer=shared_buffer,
            sub_pool_name="mamba",
            device=device,
            is_id_owner=True,
            page_size=1,  # Mamba state stays slot-granular (1-per-req)
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
            "[shared-pool] SharedMambaTokenToKVPoolAllocator ready: "
            "full max_slots=%d (min_slot_index=%d, page_size=%d, "
            "num_virtual_pages=%d), mamba max_slots=%d (min_slot_index=%d), "
            "full_available=%d, mamba_available=%d",
            self.full_attn_allocator.max_slots,
            self.full_attn_allocator.min_slot_index,
            self.full_attn_allocator.page_size,
            self.full_attn_allocator.num_virtual_pages,
            self.mamba_allocator.max_slots,
            self.mamba_allocator.min_slot_index,
            self.full_attn_allocator.available_size(),
            self.mamba_allocator.available_size(),
        )

    # -- size: dynamic (so the leak checker's num_used = size - available - evictable
    #    reduces to allocated - evictable) --
    @property
    def size(self) -> int:
        # Both terms are in TOKENS (post-Stage-3 unit-correction):
        #   - `available_size()` already converts pages → tokens on the boundary.
        #   - `allocated_count()` now returns tokens (was pages in an earlier
        #     Stage 3 revision; that mismatch caused the eval_results_15 leak).
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

    def alloc_extend(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        extend_num_tokens: int,
        num_new_pages: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        """Paged extend allocation (Stage 3). Mamba state is per-request and
        does NOT advance on per-token alloc, so the composite only forwards
        to the full sub-allocator."""
        return self.full_attn_allocator.alloc_extend(
            prefix_lens,
            prefix_lens_cpu,
            seq_lens,
            seq_lens_cpu,
            last_loc,
            extend_num_tokens,
            num_new_pages=num_new_pages,
        )

    def alloc_decode(
        self,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Paged decode allocation (Stage 3). Same dispatch logic as
        ``alloc_extend`` — the mamba side stays untouched per-decode."""
        return self.full_attn_allocator.alloc_decode(
            seq_lens, seq_lens_cpu, last_loc
        )

    def translate_kv_loc(
        self,
        loc: torch.Tensor,
        *,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Full-pool virtual TOKEN ids -> physical TOKEN ids (for the
        write/read paths). Delegates to the full-side sub-allocator's
        ``translate_kv_loc`` (page-math is in the base class).

        Stage 3.5: supports ``out=`` for cuda-graph buffer stability —
        passes through to the base-class implementation.

        `-1` inputs map to `-1` via the trailing sentinel (page=1) or via
        the page math `(-1 // ps == -1)`, `v2p[-1] == -1`, `(-1)*ps + offset
        ≤ 0` — Triton's `select_index` semantics still treat this as padding.
        """
        return self.full_attn_allocator.translate_kv_loc(loc, out=out)

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
        page_size: int = 1,
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
        self.page_size = page_size

        # Skip SWATokenToKVPoolAllocator.__init__ — call grand-parent base init
        # directly. The base's `self.size = size` call is absorbed by our
        # no-op size setter above.
        BaseTokenToKVPoolAllocator.__init__(
            self,
            size=full_max_total_num_tokens,
            page_size=page_size,
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
            page_size=page_size,
            need_sort=need_sort,
            forward_stream=forward_stream,
        )
        self.swa_attn_allocator = MultiEndedAllocator(
            kvcache=kvcache.swa_kv_pool,
            shared_buffer=shared_buffer,
            sub_pool_name="swa",
            device=device,
            is_id_owner=False,  # ← non-owner; consumes virtuals minted by full.
            page_size=page_size,
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
        """Tokens available for `alloc(N)` / `alloc_extend(N)`.

        Joint byte-budget at PAGE granularity (Stage 3): one composite
        alloc of one page-pair costs
        ``(entry_full + entry_swa) * page_size`` bytes out of the shared
        gap, equivalently ``(entry_full_per_page + entry_swa_per_page)``.
        Returns TOKENS (matches the BaseTokenToKVPoolAllocator contract
        `available_size() == len(free_pages) * page_size`).

        For page_size == 1: collapses to Stage 2 behavior — entry_sum_per_page
        == entry_bytes_full + entry_bytes_swa, return value is the number of
        slot-pairs (== tokens). (v1 lesson #1, design doc §16.2.3.)
        """
        fa, sa = self.full_attn_allocator, self.swa_attn_allocator
        # Per-page entry cost: both sides consume one page-entry each.
        entry_sum_per_page = fa.entry_bytes_per_page + sa.entry_bytes_per_page
        # The grow-up full's high frontier and grow-down swa's low frontier
        # together delimit the unused byte gap.
        full_high = fa._byte_high_frontier()
        swa_low = sa._byte_low_frontier()
        gap_bytes = max(0, swa_low - full_high)
        pages_by_bytes = gap_bytes // entry_sum_per_page
        # `_allocated_pages()` (page-granular) — index-space headroom is in
        # PAGE units. The page_size multiplication happens at the single
        # external boundary on the return statement below.
        full_room_pages = (
            fa.num_virtual_pages - fa.min_page_index - fa._allocated_pages()
        )
        swa_room_pages = (
            sa.num_virtual_pages - sa.min_page_index - sa._allocated_pages()
        )
        return min(pages_by_bytes, full_room_pages, swa_room_pages) * self.page_size

    # Slot-conservation views — the only views the leak invariant should see.
    # Under shared SWA, the swa side can consume bytes that originally counted
    # toward the full side's static budget. Returning the byte-coordinated
    # (dynamic, peer-aware) value here would generate spurious leak
    # detections. (v1 lesson #2: lines 1158–1177.)
    #
    # `allocated_count()` returns TOKENS (matching upstream convention), so
    # `cap_TOKENS - allocated_count()` is in TOKENS — the unit the leak
    # invariant expects. The Stage-3 eval_results_15 crash was caused by an
    # earlier revision that returned pages here.
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

    def translate_kv_loc(
        self,
        loc: torch.Tensor,
        *,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Full-layer read path: virtual TOKEN ids -> full-physical TOKEN ids.
        Delegates to the full-side sub-allocator's ``translate_kv_loc``
        (page-math is in the base class).

        Stage 3.5: supports ``out=`` for cuda-graph buffer stability.
        """
        return self.full_attn_allocator.translate_kv_loc(loc, out=out)

    def translate_loc_from_full_to_swa(
        self,
        kv_indices: torch.Tensor,
        *,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """SWA-layer read path: virtual TOKEN ids -> swa-physical TOKEN ids.

        For page_size == 1: direct v2p_swa lookup. For page_size > 1: page
        math, identical to ``translate_kv_loc`` but against the swa side's
        v2p table. Output is int32 to match the non-shared API contract.

        Stage 3.5: supports ``out=`` for cuda-graph buffer stability. The
        ``out=`` buffer MUST be int32 and the same shape as ``kv_indices``.

        Note: the input semantics differ from the non-shared
        `SWATokenToKVPoolAllocator.translate_loc_from_full_to_swa`
        (which takes full-physical), but the output semantics (swa-physical
        int32) match — downstream consumers don't care.
        """
        if out is not None:
            assert out.dtype == torch.int32, (
                f"translate_loc_from_full_to_swa: out= dtype must be int32 "
                f"(matches SWA Triton kernel contract), got {out.dtype}"
            )
            assert out.shape == kv_indices.shape, (
                f"translate_loc_from_full_to_swa: out= shape "
                f"{tuple(out.shape)} must match kv_indices shape "
                f"{tuple(kv_indices.shape)}"
            )
        # Tombstone-safety clamp (mirrors the full-side clamp in
        # `MultiEndedAllocator.translate_kv_loc`): v2p_swa entries can be
        # tombstoned to -1 by `_compact_pending` / `free` / `free_swa`. The
        # captured SWA attention kernel reads `swa_k_buffer[result[i]]` at
        # replay; `swa_k_buffer[-1]` is illegal memory access. Negative
        # outputs are routed to physical slot 0 (the reserved padding sink
        # under Stage 1's `min_slot_index` invariant — bytes
        # `[0, entry_max)` across all sub-pools hold no real data; see §S3
        # dummy-write safety proof). For page_size > 1 a tombstoned page
        # produces values in `[-ps, -1]` via `swa_phys * ps + offsets`; the
        # clamp covers that range too.
        if self.swa_attn_allocator.page_size == 1:
            if out is not None:
                # Two-step: gather into a transient int64 then cast into out.
                # The intermediate `tmp` is fresh per call but caching-
                # allocator-cached; the observable mutation is `out.copy_`.
                tmp = torch.index_select(
                    self.swa_attn_allocator.virtual_to_physical, 0, kv_indices
                )
                tmp = torch.clamp_min(tmp, 0)
                out.copy_(tmp.to(torch.int32))
                return out
            result = self.swa_attn_allocator.virtual_to_physical[kv_indices]
            result = torch.clamp_min(result, 0)
            return result.to(torch.int32)
        ps = self.swa_attn_allocator.page_size
        virt_pages = kv_indices // ps
        offsets = kv_indices % ps
        swa_phys_pages = self.swa_attn_allocator.virtual_to_physical[virt_pages]
        result = (swa_phys_pages * ps + offsets).to(torch.int32)
        result = torch.clamp_min(result, 0)
        if out is not None:
            out.copy_(result)
            return out
        return result

    # -- alloc --

    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        # Joint pre-check (v1 lesson #1) — accounts for byte pressure across
        # both sub-pools and both index-space headrooms.
        if need_size > self.available_size():
            return None
        # Snapshot the virtual PAGES the full-side alloc is about to consume,
        # so we can bind them on the swa side too. For page_size == 1, this
        # is just `free_virtual_ids[:need_size]` (token == page). For page>1,
        # it's `free_virtual_ids[:need_size // page_size]` (page-granular).
        fa = self.full_attn_allocator
        num_pages = need_size // self.page_size
        new_virtual_pages = fa.free_virtual_ids[:num_pages].clone()

        v_tokens = fa.alloc(need_size)
        if v_tokens is None:
            return None
        try:
            self.swa_attn_allocator.alloc_with_virtual(new_virtual_pages)
        except AssertionError:
            # Should be unreachable after the joint pre-check above; defensive
            # rollback so composite state stays coherent. (v1 lesson #5.)
            self._rollback_full_alloc(new_virtual_pages, v_tokens)
            return None
        return v_tokens

    def alloc_extend(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        extend_num_tokens: int,
    ) -> Optional[torch.Tensor]:
        """Paged extend allocation (Stage 3).

        Runs ``alloc_extend_kernel`` ONCE in virtual space (the kernel
        doesn't care whether its `free_page_ptr` is virtual or physical —
        it does `page_id * page_size + offset` math identically). Output
        is virtual TOKEN ids preserving the tail-page-reuse invariant in
        virtual space. The composite then snapshots the new virtual PAGES
        consumed by the kernel and binds them on the swa sub-allocator via
        `alloc_with_virtual`.

        Returns virtual TOKEN ids that respect:
        - the page-boundary tail-page-reuse contract
          `(last_loc + 1) % page_size == prefix_lens % page_size`
        - the cross-sub-pool identity (same virtual page id maps to
          full-physical-page on full side and swa-physical-page on swa side).
        """
        num_new_pages = get_num_new_pages(
            seq_lens=seq_lens_cpu,
            page_size=self.page_size,
            prefix_lens=prefix_lens_cpu,
        )
        # Joint pre-check at page granularity (matches v1 lesson #1).
        # `available_size()` returns TOKENS; divide by page_size for pages.
        if num_new_pages > (self.available_size() // self.page_size):
            return None

        # Snapshot the virtual PAGES that the full-side kernel call is
        # about to consume — `free_virtual_ids[:num_new_pages]` is the
        # slice the kernel reads via `free_page_ptr`. Clone so the swa
        # side has its own view even after the slice is sliced off.
        fa = self.full_attn_allocator
        new_virtual_pages = fa.free_virtual_ids[:num_new_pages].clone()

        # Run the kernel ONCE in virtual space.
        out_indices = fa.alloc_extend(
            prefix_lens,
            prefix_lens_cpu,
            seq_lens,
            seq_lens_cpu,
            last_loc,
            extend_num_tokens,
            num_new_pages=num_new_pages,
        )
        if out_indices is None:
            return None

        # Bind the new virtual pages on the swa side.
        try:
            self.swa_attn_allocator.alloc_with_virtual(new_virtual_pages)
        except AssertionError:
            # Defensive rollback (the joint pre-check should have prevented
            # this). Reverse the full-side state and return None.
            self._rollback_full_alloc(new_virtual_pages, out_indices)
            return None

        return out_indices  # virtual TOKEN ids

    def alloc_decode(
        self,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Paged decode allocation (Stage 3). One new token per request;
        a page is consumed iff the decode wraps to a new page.

        Same one-kernel-in-virtual-space discipline as ``alloc_extend``.
        """
        num_new_pages = get_num_new_pages(
            seq_lens=seq_lens_cpu, page_size=self.page_size, decode=True
        )
        # Joint pre-check at page granularity. (Even when num_new_pages == 0,
        # we still need to call the kernel to fill out_indices — but the
        # available-size check is automatic since 0 ≤ anything.)
        if num_new_pages > (self.available_size() // self.page_size):
            return None

        fa = self.full_attn_allocator
        new_virtual_pages = fa.free_virtual_ids[:num_new_pages].clone()

        out_indices = fa.alloc_decode(seq_lens, seq_lens_cpu, last_loc)
        if out_indices is None:
            return None

        if new_virtual_pages.numel() > 0:
            try:
                self.swa_attn_allocator.alloc_with_virtual(new_virtual_pages)
            except AssertionError:
                self._rollback_full_alloc(new_virtual_pages, out_indices)
                return None

        return out_indices  # virtual TOKEN ids

    def _rollback_full_alloc(
        self, v_pages: torch.Tensor, v_tokens: torch.Tensor
    ) -> None:
        """Undo a `full_attn_allocator.alloc(need_size)` whose paired swa-side
        `alloc_with_virtual` could not complete. Reverse the full-side
        watermark, clear the v2p / p2v bindings, and push the minted virtual
        PAGE ids back to the head of the free list. Symmetric to `alloc`'s
        order."""
        fa = self.full_attn_allocator
        num_pages = int(v_pages.numel())
        if num_pages == 0:
            return
        # Reverse the bind (at PAGE granularity — tables are page-granular).
        phys_pages = fa.virtual_to_physical[v_pages].clone()
        fa.virtual_to_physical[v_pages] = -1
        fa.physical_to_virtual[phys_pages] = -1
        # Reverse the watermark advance.
        if fa.grow_direction == "up":
            fa.watermark_physical -= num_pages
        else:
            fa.watermark_physical += num_pages
        # Recycle the virtual PAGES to the head of the free list (front, so
        # the next alloc reuses them — matches the slice-from-front
        # consumption in `alloc`).
        fa.free_virtual_ids = torch.cat([v_pages, fa.free_virtual_ids])

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
        # `swa.v2p_page[v_page] == -1` because `free_swa(...)` ran earlier).
        # Mirrors the v1 `swa_indices > 0` filter at
        # `old_design_and_impl/...:1387`. The full side does NOT need this
        # filter — under SWARadixCache the full side is the lifecycle owner,
        # so every value in `free_index` must still be bound on full.
        #
        # The filter operates at PAGE granularity (recovering v_pages via
        # `// page_size`) and emits TOKEN-granular `live_swa_tokens` so
        # `swa.free` can apply its own `unique(// page_size)` internally.
        v = free_index.detach().to(torch.int64)
        v_pages = v // self.page_size
        swa_v2p_pages = self.swa_attn_allocator.virtual_to_physical[v_pages]
        # `> 0` (strict): -1 = tombstoned, 0 = padding-sink page — both
        # skipped. Mirrors the non-shared `swa_indices > 0`.
        live_token_mask = swa_v2p_pages > 0
        live_tokens = v[live_token_mask]
        if live_tokens.numel() > 0:
            self.swa_attn_allocator.free(live_tokens)
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
        `virtual_to_physical_page[v_page] = -1` after this call IS the
        tombstone.

        Page-aware filter: recover `v_pages = v // page_size`, look up
        `swa.v2p_page[v_pages]`, keep token IDs whose page is still bound
        on the swa side. Token-granular output goes to ``swa.free`` which
        applies its own `unique(// page_size)` internally.
        """
        if free_index is None or free_index.numel() == 0:
            return
        # Filter to tokens whose virtual PAGE still has an swa-side binding —
        # under v2, `swa.v2p_page[v_page] == -1` means already-tombstoned;
        # calling `swa.free` on those would assert.
        v = free_index.detach().to(torch.int64)
        v_pages = v // self.page_size
        # `> 0` (strict): tombstoned entries have v2p_page[...] == -1; virtual
        # page 0 is the padding-sink page bound to physical page 0 — never
        # freeable. Mirrors the non-shared `free_swa`'s `swa_indices > 0`
        # filter (`swa_memory_pool.py:502`).
        swa_v2p_pages = self.swa_attn_allocator.virtual_to_physical[v_pages]
        live = v[swa_v2p_pages > 0]
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
