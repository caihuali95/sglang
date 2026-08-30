# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Turns the KV ids stored in `req_to_token` into ids attention kernels can use.

WHY THIS EXISTS. A KV slot can be named in more than one **id space**:

  * **virtual** - what `req_to_token` stores. Stable: it keeps naming the same
    logical slot even after the pool moves data around.
  * **physical** - where that slot sits in the pool right now.
  * **kernel-facing** - what a kernel can index the per-layer K/V tensors with.
    Same as physical for a plain pool; under the unified pool's dense views it
    is the physical page scaled by the per-page block count.

On a plain pool all three coincide and nothing here does any work. Under the
unified memory pool they differ, so somebody must convert - and if every
backend converts for itself, each one has to know the pool's internals and
each is a place to get it wrong. This module is the one place that converts.

WHAT BACKENDS GET. A `KVIndexTable`, which answers one question: *what do I
gather from, and which row is mine?*

    ids[row_ids[b], pos]        <- the gather every backend already does

    plain pool : ids = req_to_token, row_ids = req_pool_indices
                 (literally those objects - no copy, no kernel, no change)
    unified    : ids = a freshly built array of kernel-facing ids,
                 row_ids = arange(batch_size)

Backends call their own copy a *page table* (fa3) or a *block table*
(trtllm); this module calls what it hands them the **index table**.

WHY ONE TABLE IS ENOUGH FOR EVERYONE. Converting only ever rewrites the page
number and keeps the offset inside the page. So a page-granular table serves
both kinds of consumer: a block-table backend uses its rows as-is, and a
backend that wants a flat per-token list rebuilds one with

    token_id = entry * entry_page_size + pos % entry_page_size

WRITES, IN TWO PHASES. The full-side write loc is rebound to kernel-facing
ids at ForwardBatch construction - the earliest consumer can snapshot it
right after. The sliding-window write loc is derived at the same moment as
read table, into the same index table.
"""

from __future__ import annotations

import functools
import weakref
from typing import Optional, Tuple

import msgspec
import torch

from sglang.kernels.ops.kvcache.kv_read_table import build_kv_read_table
from sglang.srt.mem_cache.multi_ended_allocator import (
    UnifiedMambaTokenToKVPoolAllocator,
    UnifiedSWATokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.mem_cache.unified_memory_pool import UnifiedDraftKVPool


def _same_loc(loc: torch.Tensor) -> torch.Tensor:
    """Single id space: window layers write the SAME slots as full layers, so
    the sliding-window write loc IS the full one (the fused draft region)."""
    return loc


class KVIndexTable(msgspec.Struct, frozen=True):
    """Collection of what one batch gathers from."""

    ids: torch.Tensor  # 2-D array of KV ids to gather from
    row_ids: torch.Tensor  # which row belongs to batch lane b
    row_stride: int  # stride between rows of `ids`, in elements
    entry_page_size: int  # what one entry covers: 1 = a token, N = a page of N
    is_translated: bool  # entries are already kernel-facing ids
    sliding_window_ids: Optional[torch.Tensor]  # SWA models: the parallel swa array
    # SWA model's per batch sliding-window WRITE loc.
    sliding_window_write_loc: Optional[torch.Tensor] = None

    def sliding_window_read_ids(self) -> torch.Tensor:
        """Which array a sliding-window gather reads: the parallel swa array
        when translated, else the full-attention array, which the caller maps
        through the pool's own full->swa map. A fused draft region keeps no
        separate swa id space, so its window reads come from the one array
        too."""
        if not self.is_translated:
            return self.ids
        return (
            self.sliding_window_ids if self.sliding_window_ids is not None else self.ids
        )

    @classmethod
    def passthrough(
        cls, *, req_to_token: torch.Tensor, req_pool_indices: torch.Tensor
    ) -> KVIndexTable:
        """The raw (req_to_token, req_pool_indices) table — for a caller on a
        plain pool with no translator in reach (the aiter updaters)."""
        return cls(
            ids=req_to_token,
            row_ids=req_pool_indices,
            row_stride=req_to_token.stride(0),
            entry_page_size=1,
            is_translated=False,
            sliding_window_ids=None,
        )


class KVIndexTranslator:
    """Built once per ModelRunner."""

    def __init__(
        self,
        *,
        req_to_token: torch.Tensor,
        token_to_kv_pool_allocator,
        token_to_kv_pool,
        page_size: int,
        device: str,
    ):
        self.req_to_token = req_to_token
        self.page_size = page_size
        self.device = device

        is_unified_target = (
            isinstance(
                token_to_kv_pool_allocator,
                (UnifiedMambaTokenToKVPoolAllocator, UnifiedSWATokenToKVPoolAllocator),
            )
            and token_to_kv_pool_allocator.get_kvcache() is token_to_kv_pool
        )
        # Fused draft KV: the draft runner reads/writes the DRAFT region fused
        # into the target's pages — same v2p table, its own dense stride. The
        # host_allocator identity replaces the target's get_kvcache() identity
        # (the draft pool is deliberately NOT the allocator's kvcache).
        # Private-pool drafts (DSPARK / DFLASH) match NEITHER branch and keep
        # the strict passthrough — their virtual-indexed buffers must never be
        # translated (see the class docstring).
        is_fused_draft = (
            isinstance(token_to_kv_pool, UnifiedDraftKVPool)
            and token_to_kv_pool.host_allocator is token_to_kv_pool_allocator
        )
        self.is_translating = is_unified_target or is_fused_draft
        if is_fused_draft:
            alloc = token_to_kv_pool_allocator
            draft_mult = token_to_kv_pool.draft_kernel_page_multiplier
            self._full_v2p_table = alloc.full_v2p_page_table
            self._full_page_multiplier = draft_mult
            self._translate_full = functools.partial(
                alloc.full_attn_allocator.translate_kv_loc_dense,
                multiplier=draft_mult,
            )
            # The draft family is dense-only: window layers read and write
            # the SAME fused slots as full layers, so there is no separate swa
            # id space — the sliding-window write loc aliases the dense loc at
            # the per-batch build instead.
            self._full_p2v_table = None
            self._swa_v2p_table = None
            self._swa_page_multiplier = 1
            self._swa_write_loc_from_full = _same_loc
        elif self.is_translating:
            alloc = token_to_kv_pool_allocator
            self._full_v2p_table = alloc.full_v2p_page_table
            self._full_p2v_table = alloc.full_p2v_page_table
            self._full_page_multiplier = alloc.kernel_page_multiplier
            self._translate_full = alloc.translate_kv_loc_dense
            if isinstance(alloc, UnifiedSWATokenToKVPoolAllocator):
                self._swa_v2p_table = alloc.swa_v2p_page_table
                self._swa_page_multiplier = alloc.swa_kernel_page_multiplier
                self._swa_write_loc_from_full = self._swa_write_loc_from_dense
            else:
                self._swa_v2p_table = None
                self._swa_page_multiplier = 1
                self._swa_write_loc_from_full = None
        else:
            self._full_v2p_table = None
            self._full_p2v_table = None
            self._full_page_multiplier = 1
            self._translate_full = None
            self._swa_v2p_table = None
            self._swa_page_multiplier = 1
            self._swa_write_loc_from_full = (
                token_to_kv_pool.translate_loc_from_full_to_swa
                if isinstance(token_to_kv_pool, SWAKVPool)
                else None
            )

        self._rows: Optional[torch.Tensor] = (
            torch.arange(req_to_token.shape[0], dtype=torch.int64, device=device)
            if self.is_translating
            else None
        )
        self._capture_full_ids: Optional[torch.Tensor] = None
        self._capture_swa_ids: Optional[torch.Tensor] = None
        self._index_table_memo: Optional[Tuple[weakref.ref, KVIndexTable]] = None

    def full_flat_translate_args(self) -> Optional[Tuple[torch.Tensor, int]]:
        """``(v2p_page_table, kernel_page_multiplier)`` for a kernel that
        translates flat full-side ids itself, or ``None`` when this runner
        does not translate (pass-through and static pools)."""
        if not self.is_translating:
            return None
        return (self._full_v2p_table, self._full_page_multiplier)

    # -- capture-stable buffers ------------------------------------------------

    def ensure_capture_buffers(self, *, max_bs: int, max_context_len: int) -> None:
        """Idempotent. Zero-filled ``(max_bs, ceil(ctx/ps))`` int32 per kind:
        entry 0 is the reserved padding slot in every id space, so a captured
        graph replaying before any refresh reads padding, not garbage."""
        if not self.is_translating or self._capture_full_ids is not None:
            return
        max_pages = -(-max_context_len // self.page_size)
        self._capture_full_ids = torch.zeros(
            (max_bs, max_pages), dtype=torch.int32, device=self.device
        )
        if self._swa_v2p_table is not None:
            self._capture_swa_ids = torch.zeros(
                (max_bs, max_pages), dtype=torch.int32, device=self.device
            )

    # -- per-batch view --------------------------------------------------------

    def build_index_table(
        self,
        *,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        max_pages: Optional[int] = None,
        captured: bool = False,
        out_cache_loc: Optional[torch.Tensor] = None,
        seq_len_delta: int = 0,
    ) -> KVIndexTable:
        """The one per-batch entry point.

        Non-unified: the raw ``(req_to_token, req_pool_indices)`` passthrough,
        no tensor ops and no copies. Unified: a fresh table of width
        ``max_pages`` (eager), or a live-prefix refresh of the capture buffers
        (``captured=True``), returned WHOLE so its pointer is capture-bakeable.

        ``out_cache_loc`` is the batch's already-kernel-facing write loc; the
        index table returns the matching ``sliding_window_write_loc``.

        ``seq_len_delta`` widens every row's live prefix — the whole-sequence
        verify contract (draft KV read back from the pool). An eager caller's
        ``max_pages`` must already cover the delta.
        """
        if not self.is_translating:
            return KVIndexTable(
                ids=self.req_to_token,
                row_ids=req_pool_indices,
                row_stride=self.req_to_token.stride(0),
                entry_page_size=1,
                is_translated=False,
                sliding_window_ids=None,
                sliding_window_write_loc=self._translated_swa_write_loc(out_cache_loc),
            )

        bs = int(req_pool_indices.numel())
        if captured:
            assert self._capture_full_ids is not None, (
                "KVIndexTranslator.build_index_table(captured=True) before "
                "ensure_capture_buffers()"
            )
            out_full = self._capture_full_ids
            out_swa = self._capture_swa_ids
            width = out_full.shape[1] if max_pages is None else max_pages
        else:
            assert max_pages is not None, (
                "KVIndexTranslator.build_index_table: eager path needs max_pages "
                "(from the batch's seq_lens_cpu max)"
            )
            width = max_pages
            out_full = torch.zeros((bs, width), dtype=torch.int32, device=self.device)
            out_swa = (
                torch.zeros((bs, width), dtype=torch.int32, device=self.device)
                if self._swa_v2p_table is not None
                else None
            )

        build_kv_read_table(
            req_to_token=self.req_to_token,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            v2p=self._full_v2p_table,
            multiplier=self._full_page_multiplier,
            page_size=self.page_size,
            max_pages=width,
            out=out_full,
            seq_len_delta=seq_len_delta,
        )
        if out_swa is not None:
            build_kv_read_table(
                req_to_token=self.req_to_token,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                v2p=self._swa_v2p_table,
                multiplier=self._swa_page_multiplier,
                page_size=self.page_size,
                max_pages=width,
                out=out_swa,
                seq_len_delta=seq_len_delta,
            )
        return KVIndexTable(
            ids=out_full,
            row_ids=self._rows[:bs],
            row_stride=out_full.stride(0),
            entry_page_size=self.page_size,
            is_translated=True,
            sliding_window_ids=out_swa,
            sliding_window_write_loc=self._translated_swa_write_loc(out_cache_loc),
        )

    def fill_read_table(
        self,
        *,
        out: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_len_delta: int = 0,
    ) -> torch.Tensor:
        """Fill a backend-owned padded 2-D block table's live prefix with
        full-attention entries (trtllm_mla / flashmla consume such tables
        directly — their rows ARE the index table's rows).

        Prefix-only: columns past each row's live pages keep the backend's own
        fill (-1 sentinel or stale-but-unread values).

        Unified-only: callers dispatch on ``self.is_translating`` and keep
        their static builder otherwise.
        """
        assert (
            self.is_translating
        ), "KVIndexTranslator.fill_read_table on a pool that needs no translation"
        max_pages = min(
            out.shape[1],
            -(-self.req_to_token.shape[1] // self.page_size),
        )
        return build_kv_read_table(
            req_to_token=self.req_to_token,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            v2p=self._full_v2p_table,
            multiplier=self._full_page_multiplier,
            page_size=self.page_size,
            max_pages=max_pages,
            out=out,
            seq_len_delta=seq_len_delta,
        )

    def index_table_for_batch(self, forward_batch) -> KVIndexTable:
        """Eager per-batch view, memoized in one slot keyed by batch identity
        so multi-consumer metadata builds share a build. The next batch
        replaces the slot; consumers only read during their own build. The
        captured path does not memoize — it refreshes its buffers per
        replay."""
        memo = self._index_table_memo
        if memo is not None and memo[0]() is forward_batch:
            return memo[1]
        max_pages = None
        if self.is_translating:
            slc = forward_batch.seq_lens_cpu
            if slc is not None and slc.numel() > 0:
                max_seq = int(slc.max())
            else:
                # gpu_only batches carry no CPU mirror; bound by the table.
                max_seq = self.req_to_token.shape[1]
            max_pages = max(-(-max_seq // self.page_size), 1)
        view = self.build_index_table(
            req_pool_indices=forward_batch.req_pool_indices,
            seq_lens=forward_batch.seq_lens,
            max_pages=max_pages,
            out_cache_loc=forward_batch.out_cache_loc,
        )
        self._index_table_memo = (weakref.ref(forward_batch), view)
        return view

    def assert_backends_carry_translator(self, backends) -> None:
        """Boot guard: under the unified pool every backend a forward can reach
        must carry THIS translator."""
        if not self.is_translating:
            return
        for backend in backends:
            if backend is None:
                continue
            assert backend.kv_index_translator is self, (
                f"{type(backend).__name__} does not carry the runner's "
                "KVIndexTranslator. A backend (or wrapper) reachable under "
                "--enable-unified-memory must forward `kv_index_translator`, or "
                "read-index producers silently skip the virtual->kernel-facing "
                "translation."
            )

    # -- write loc (phase 1; phase 2 lives in build_index_table) ----------------

    def rebind_write_loc(self, forward_batch) -> None:
        """Phase 1 of the WRITE contract: translate the batch's write loc to
        FULL-side kernel-facing ids exactly once, at ForwardBatch
        construction. No-op on non-unified pools.

        REBIND, never mutate: the translate returns a FRESH tensor, so the
        ScheduleBatch's aliased tensor stays VIRTUAL for the radix / accept /
        in-flight machinery that reads it.
        """
        self._index_table_memo = None
        if not self.is_translating or forward_batch.out_cache_loc is None:
            return
        forward_batch.out_cache_loc = self._translate_full(forward_batch.out_cache_loc)

    def _translated_swa_write_loc(
        self, out_cache_loc: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """Phase 2: this batch's sliding-window write loc, or None when there
        is no loc this forward or the pool has no sliding-window id space."""
        if out_cache_loc is None or self._swa_write_loc_from_full is None:
            return None
        return self._swa_write_loc_from_full(out_cache_loc)

    def _swa_write_loc_from_dense(self, dense_loc: torch.Tensor) -> torch.Tensor:
        """Sliding-window write loc, derived pointwise from FULL-side
        kernel-facing values (phase 2 of the write contract).
        """
        full_stride = self.page_size * self._full_page_multiplier
        offset = dense_loc % full_stride  # == virtual_token % page_size
        virt_page = self._full_p2v_table[dense_loc // full_stride]
        swa_stride = self.page_size * self._swa_page_multiplier
        return (self._swa_v2p_table[virt_page] * swa_stride + offset).clamp_(min=0)

    # -- token-level translate surface (the mixin / local-attn consumers) ------

    def translate_full_attn_ids(
        self, kv_indices: torch.Tensor, *, out: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Virtual token ids -> kernel-facing full-attention ids (the identity
        when no translation is needed, so callers never branch)."""
        if not self.is_translating:
            assert out is None, "passthrough translate takes no out="
            return kv_indices
        return self._translate_full(kv_indices, out=out)
