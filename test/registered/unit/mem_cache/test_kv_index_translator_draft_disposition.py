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
"""KVIndexTranslator's three id-space dispositions with a fused draft region.

1. TARGET runner (allocator's own kvcache): enabled at the HOST multiplier.
2. FUSED-DRAFT runner (`UnifiedDraftKVPool` bound to this allocator): enabled
   at the DRAFT multiplier — same v2p table, draft-dense stride, no SWA rail.
3. PRIVATE-POOL draft (DSPARK/DFLASH shape: target's allocator, own
   virtual-indexed buffer): strict passthrough — translating such a runner
   would address a slot-count buffer with dense ids (OOB both directions),
   so the no-op is load-bearing, not a default.

    python -m pytest test/registered/unit/mem_cache/test_kv_index_source_draft_disposition.py -v
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.kv_index_translator import KVIndexTranslator
from sglang.srt.mem_cache.multi_ended_allocator import (
    UnifiedSWATokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.unified_memory_pool import (
    DenseDraftRegion,
    MHASubPoolSpec,
    UnifiedDraftKVPool,
    UnifiedKVPool,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_DEV = "cpu"
_PS = 2


class _FakeKVCache:
    def __init__(self, max_slots):
        self.buf = torch.full((max_slots,), -1, dtype=torch.int64)
        self.allocator = None

    def attach_allocator(self, allocator):
        self.allocator = allocator

    def move_kv_cache(self, dst_loc, src_loc):
        self.buf[dst_loc] = self.buf[src_loc].clone()


class _FakeUnifiedSWAKVPool:
    def __init__(self, shared_pool):
        self.full_kv_pool = _FakeKVCache(shared_pool.max_slots("full"))
        self.swa_kv_pool = _FakeKVCache(shared_pool.max_slots("swa"))
        self.full_to_swa_index_mapping = None

    def attach_allocators(self, *, full_allocator, swa_allocator):
        self._full_allocator = full_allocator
        self._swa_allocator = swa_allocator


def _build(n_full=32, n_swa=16):
    full_spec = MHASubPoolSpec(
        name="full",
        layer_num=2,
        head_num=2,
        head_dim=4,
        store_dtype=torch.bfloat16,
        grow_direction="down",
        draft_region=DenseDraftRegion(
            layer_num=1, head_num=1, head_dim=3, store_dtype=torch.bfloat16
        ),
    )
    swa_spec = MHASubPoolSpec(
        name="swa",
        layer_num=1,
        head_num=2,
        head_dim=4,
        store_dtype=torch.bfloat16,
        grow_direction="up",
    )
    total = n_full * full_spec.entry_bytes() + n_swa * swa_spec.entry_bytes()
    pool = UnifiedKVPool(
        total_bytes=total,
        sub_pool_specs=[full_spec, swa_spec],
        device=_DEV,
        enable_memory_saver=False,
        page_size=_PS,
    )
    kvcache = _FakeUnifiedSWAKVPool(pool)
    allocator = UnifiedSWATokenToKVPoolAllocator(
        unified_buffer=pool,
        kvcache=kvcache,
        device=_DEV,
        full_max_total_num_tokens=n_full,
        swa_max_total_num_tokens=n_swa,
        page_size=_PS,
        need_sort=False,
        forward_stream=None,
    )
    draft_pool = UnifiedDraftKVPool(
        unified_buffer=pool,
        host_sub_pool_name="full",
        host_allocator=allocator,
        page_size=_PS,
    )
    return pool, allocator, kvcache, draft_pool


def _source(allocator, pool_obj):
    return KVIndexTranslator(
        req_to_token=torch.zeros((4, 16), dtype=torch.int32),
        token_to_kv_pool_allocator=allocator,
        token_to_kv_pool=pool_obj,
        page_size=_PS,
        device=_DEV,
    )


class TestKVIndexSourceDraftDisposition(unittest.TestCase):
    def test_target_runner_uses_the_host_multiplier(self):
        _, allocator, kvcache, _ = _build()
        src = _source(allocator, kvcache)
        self.assertTrue(src.is_translating)
        self.assertEqual(src._full_page_multiplier, allocator.kernel_page_multiplier)

    def test_fused_draft_runner_uses_the_draft_multiplier(self):
        pool, allocator, _, draft_pool = _build()
        spec = pool.mha_spec("full")
        src = _source(allocator, draft_pool)
        self.assertTrue(src.is_translating)
        self.assertEqual(src._full_page_multiplier, spec.draft_kernel_page_multiplier())
        self.assertIsNone(src._swa_v2p_table)  # draft family: no swa id space

        # rebind_write_loc translates to DRAFT-dense ids: same v2p pages,
        # draft stride.
        v = allocator.alloc(2 * _PS)
        self.assertIsNotNone(v)
        fb = SimpleNamespace(out_cache_loc=v, swa_out_cache_loc=None)
        src.rebind_write_loc(fb)
        fa = allocator.full_attn_allocator
        phys_pages = fa.virtual_to_physical[v // _PS]
        expected = torch.clamp_min(
            phys_pages * (_PS * spec.draft_kernel_page_multiplier()) + v % _PS, 0
        )
        torch.testing.assert_close(fb.out_cache_loc, expected, rtol=0, atol=0)
        self.assertIsNone(fb.swa_out_cache_loc)

    def test_private_pool_draft_stays_a_strict_passthrough(self):
        _, allocator, _, _ = _build()
        private_draft_pool = _FakeKVCache(64)  # own buffer, not the allocator's
        src = _source(allocator, private_draft_pool)
        self.assertFalse(src.is_translating)
        v = torch.arange(2 * _PS, dtype=torch.int64)
        fb = SimpleNamespace(out_cache_loc=v, swa_out_cache_loc=None)
        src.rebind_write_loc(fb)
        self.assertIs(fb.out_cache_loc, v)  # untouched, not even a copy

    def test_foreign_allocator_draft_pool_is_not_enabled(self):
        # A UnifiedDraftKVPool bound to a DIFFERENT allocator must not enable
        # against this one (identity, not type, decides).
        pool, allocator, kvcache, _ = _build()
        _, other_alloc, _, other_draft = _build()
        src = _source(allocator, other_draft)
        self.assertFalse(src.is_translating)


if __name__ == "__main__":
    unittest.main()
