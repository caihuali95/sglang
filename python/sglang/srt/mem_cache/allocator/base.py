"""
Copyright 2025 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import KVCache


class BaseTokenToKVPoolAllocator(abc.ABC):
    @abc.abstractmethod
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        device: str,
        kvcache: KVCache,
        need_sort: bool,
    ):
        self.size = size
        self.page_size = page_size
        self.dtype = dtype
        self.device = device
        self._kvcache = kvcache
        self.need_sort = need_sort

        self.free_pages = None
        self.release_pages = None
        self.is_not_in_free_group = True
        self.free_group = []

    @property
    def size_full(self):
        return self.size

    def debug_print(self) -> str:
        return ""

    def available_size(self):
        return (len(self.free_pages) + len(self.release_pages)) * self.page_size

    def get_kvcache(self):
        return self._kvcache

    def restore_state(self, state):
        self.free_pages, self.release_pages = state

    def backup_state(self):
        return (self.free_pages, self.release_pages)

    def free_group_begin(self):
        self.is_not_in_free_group = False
        self.free_group = []

    def free_group_end(self):
        self.is_not_in_free_group = True
        if self.free_group:
            self.free(torch.cat(self.free_group))

    def on_forward_launched(
        self,
        forward_stream: torch.Stream,
        out_cache_loc: Optional[torch.Tensor],
    ) -> None:
        """Lifecycle hook: a forward was just launched asynchronously.

        Called by the scheduler once per launched forward, on the scheduling
        thread, immediately after ALL of the forward's kernels have been
        enqueued on ``forward_stream`` (for a speculative step that includes
        every draft/verify kernel) and BEFORE anything else — frees of the
        previous batch, new allocations — can run. ``out_cache_loc`` is the
        forward's write-location tensor as the scheduler allocated it.

        Default: no-op. An allocator that relocates or reuses pages while a
        forward may still be reading or writing them (the unified memory
        pool's lazy compaction) overrides this to record a completion event
        and register the forward's write-set. The call-time guarantee above
        is what makes that registration race-free: the allocator learns about
        the in-flight forward before any flush can consider its pages.
        """

    def on_idle_proceed(self) -> None:
        """Lifecycle hook: the scheduler has no work and is about to idle.

        Called by the scheduler from ``on_idle`` (only when fully idle, so no
        forward is in flight). Default: no-op. An allocator that defers work
        while requests are running overrides this to catch up now — the
        unified memory pool compacts its lazy holes here, when doing so is
        free of any concurrent forward.
        """

    def make_admission_budget(self, adder) -> Optional[object]:
        """Admission-budget hook: return a specialized admission budget for
        the given ``PrefillAdder``, or ``None`` for the scheduler's default
        token budget.

        The scheduler constructs exactly one budget per prefill pass and
        routes every admission gate/charge through it, so an allocator whose
        capacity is not token-denominated (the unified memory pool: one byte
        buffer shared by full-KV / mamba-state / SWA-KV / spec-state band)
        can supply byte-denominated admission without any allocator-specific
        logic in the scheduler. Default: ``None`` (token budget).
        """
        return None

    def check_decode_capacity(
        self, tree_cache, num_tokens: int, num_reqs: int
    ) -> bool:
        """Decode-capacity gate: can the pool back ``num_tokens`` more tokens
        (page-aligned, summed over ``num_reqs`` running requests) for the next
        decode step? Evicts from ``tree_cache`` as needed first.

        Called by ``ScheduleBatch.check_decode_mem`` (and therefore by the
        retract loop) once per decode scheduling step. Default: the standard
        token gate. An allocator with additional per-step reservations (the
        unified pool's spec-state band) or non-token capacity overrides this.
        """
        from sglang.srt.mem_cache.common import evict_from_tree_cache

        evict_from_tree_cache(tree_cache, num_tokens)
        return self.available_size() >= num_tokens

    def evict_to_free_tokens(self, tree_cache, num_tokens: int) -> bool:
        """Eviction hook: make room for ``num_tokens`` tokens, returning True
        when this allocator OWNED the eviction sizing (the caller must then
        skip the standard sizing). Default: False (standard sizing applies).

        An allocator whose sub-pools share one byte budget (the unified pool:
        per-side views alias the same gap bytes) overrides this — per-side
        token shortfalls cannot size a joint eviction correctly there.
        """
        return False

    def verify_byte_accounting(self) -> list:
        """Idle-time byte-accounting self-check: return a list of problem
        strings (empty == healthy). Default: no byte accounting to verify
        (the standard pools are covered by the token-identity leak check).

        Called unconditionally by the scheduler's idle invariant checker
        AFTER ``on_idle_proceed()``. An allocator managing a shared byte
        buffer overrides this with its buffer-conservation invariants
        (frontier ordering, per-side page conservation, band state) — the
        structure the token identity cannot see.
        """
        return []

    def merge_and_sort_free(self):
        if len(self.release_pages) > 0:
            self.free_pages = torch.cat((self.free_pages, self.release_pages))
            self.free_pages, _ = torch.sort(self.free_pages)
            self.release_pages = torch.empty(
                (0,), dtype=self.release_pages.dtype, device=self.device
            )

    def get_cpu_copy(self, indices, mamba_indices=None):
        # FIXME: reuse the get_cpu_copy after paged allocator is implemented
        raise NotImplementedError()

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        # FIXME: reuse the load_cpu_copy after paged allocator is implemented
        raise NotImplementedError()

    def alloc_extend(self, *args, **kwargs):
        raise NotImplementedError("alloc_extend is only for paged allocator")

    def alloc_decode(self, *args, **kwargs):
        raise NotImplementedError("alloc_decode is only for paged allocator")

    @abc.abstractmethod
    def clear(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def alloc(self, need_size: int):
        raise NotImplementedError()

    @abc.abstractmethod
    def free(self, free_index: torch.Tensor):
        raise NotImplementedError()
