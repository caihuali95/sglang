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
"""`UnifiedDraftKVPool` binding : the draft worker's KV pool is a
pure view object over the DRAFT region of the target's fused full sub-pool.

Covers: buffers alias the unified draft views exactly (same storage, same
strides); multi-layer `layer_offset` sub-ranges slice the region correctly and
don't overlap; the pool advertises the draft geometry (head dims/layer_num)
not the host's; relocation and contiguous-buffer description fail loud
(host-driven move / PD excluded); constructing over a non-fused sub-pool
fails the has-draft-region assert.

CPU-only: view binding + geometry checks; the (inherited, stride-generic)
Triton write path is exercised by the host-pool GPU tests.
"""

import unittest

import torch

from sglang.srt.mem_cache.unified_memory_pool import (
    MHARegionGeometry,
    MHASubPoolSpec,
    UnifiedDraftKVPool,
    UnifiedKVPool,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _make_fused(*, draft_layers=4, page_size=2, want_pages=4):
    region = MHARegionGeometry(
        layer_num=draft_layers,
        head_num=1,
        head_dim=8,
        v_head_dim=4,
        store_dtype=torch.bfloat16,
    )
    full_spec = MHASubPoolSpec(
        name="full",
        grow_direction="down",
        layer_num=2,
        head_num=2,
        head_dim=16,
        store_dtype=torch.bfloat16,
        draft_region=region,
    )
    peer_spec = MHASubPoolSpec(
        name="peer",
        grow_direction="up",
        layer_num=1,
        head_num=1,
        head_dim=4,
        store_dtype=torch.bfloat16,
    )
    unified = UnifiedKVPool(
        total_bytes=(want_pages + 2) * page_size * full_spec.entry_bytes(),
        sub_pool_specs=[full_spec, peer_spec],
        device="cpu",
        enable_memory_saver=False,
        page_size=page_size,
    )
    return unified, region


class TestUnifiedDraftPool(CustomTestCase):
    def test_buffers_alias_draft_views(self):
        unified, region = _make_fused()
        pool = UnifiedDraftKVPool(
            unified_buffer=unified,
            sub_pool_name="full",
            page_size=2,
            enable_alt_stream=False,
        )
        draft_k, draft_v = unified.draft_views_for("full")
        self.assertEqual(pool.layer_num, region.layer_num)
        for i in range(region.layer_num):
            self.assertEqual(pool.k_buffer[i].data_ptr(), draft_k[i].data_ptr())
            self.assertEqual(pool.k_buffer[i].stride(), draft_k[i].stride())
            self.assertEqual(pool.v_buffer[i].data_ptr(), draft_v[i].data_ptr())
        # Draft geometry, not the host's.
        self.assertEqual(pool.head_num, region.head_num)
        self.assertEqual(pool.head_dim, region.head_dim)
        self.assertEqual(pool.start_layer, 0)

    def test_layer_offset_subranges(self):
        # Multi-layer EAGLE: 4-layer region split across 2 per-step pools.
        unified, region = _make_fused(draft_layers=4)
        draft_k, _ = unified.draft_views_for("full")
        pools = [
            UnifiedDraftKVPool(
                unified_buffer=unified,
                sub_pool_name="full",
                page_size=2,
                layer_offset=step * 2,
                layer_num=2,
                enable_alt_stream=False,
            )
            for step in range(2)
        ]
        self.assertEqual(pools[0].k_buffer[0].data_ptr(), draft_k[0].data_ptr())
        self.assertEqual(pools[0].k_buffer[1].data_ptr(), draft_k[1].data_ptr())
        self.assertEqual(pools[1].k_buffer[0].data_ptr(), draft_k[2].data_ptr())
        self.assertEqual(pools[1].k_buffer[1].data_ptr(), draft_k[3].data_ptr())
        # Out-of-range sub-range rejected.
        with self.assertRaises(AssertionError):
            UnifiedDraftKVPool(
                unified_buffer=unified,
                sub_pool_name="full",
                page_size=2,
                layer_offset=3,
                layer_num=2,
                enable_alt_stream=False,
            )

    def test_move_and_contiguous_fail_loud(self):
        unified, _ = _make_fused()
        pool = UnifiedDraftKVPool(
            unified_buffer=unified,
            sub_pool_name="full",
            page_size=2,
            enable_alt_stream=False,
        )
        with self.assertRaises(NotImplementedError):
            pool.move_kv_cache(torch.tensor([2]), torch.tensor([3]))
        with self.assertRaises(NotImplementedError):
            pool.get_contiguous_buf_infos()

    def test_non_fused_sub_pool_rejected(self):
        unified, _ = _make_fused()
        with self.assertRaises(AssertionError):
            UnifiedDraftKVPool(
                unified_buffer=unified,
                sub_pool_name="peer",  # no draft region
                page_size=2,
                enable_alt_stream=False,
            )


if __name__ == "__main__":
    unittest.main()
