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
"""Tombstone clamp on the POOL-level SWA translate.

`UnifiedSWAKVPool.translate_loc_from_full_to_swa` had no tombstone clamp
while the composite allocator's translate did (documented there as required:
a tombstoned v2p_swa entry yields a NEGATIVE token id, which a captured graph
reads out of bounds — `swa_k_buffer[-1]` under replay). Both implementations
must route tombstones to the reserved padding sink (slot 0).

    python -m pytest test/registered/unit/mem_cache/test_unified_swa_translate_clamp.py -v
"""

import unittest

import torch

from sglang.srt.mem_cache.unified_memory_pool import init_unified_swa_pools
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

_DEV = "cpu"


def _swa_bundle(page_size=1):
    return init_unified_swa_pools(
        device=_DEV,
        kv_cache_dtype=torch.float16,
        head_num=2,
        head_dim=8,
        v_head_dim=8,
        swa_head_num=2,
        swa_head_dim=8,
        swa_v_head_dim=8,
        page_size=page_size,
        start_layer=0,
        end_layer=4,
        swa_attention_layer_ids=[1, 3],
        full_attention_layer_ids=[0, 2],
        full_max_total_num_tokens=64,
        swa_max_total_num_tokens=32,
        enable_memory_saver=False,
        need_sort=False,
    )


class TestPoolLevelSwaTranslateClamp(unittest.TestCase):
    """Driven through the real factory bundle (CPU), on both page shapes."""

    def _drive(self, page_size):
        bundle = _swa_bundle(page_size=page_size)
        allocator = bundle.token_to_kv_pool_allocator
        kvcache = bundle.token_to_kv_pool
        n = 2 * page_size if page_size > 1 else 8
        v = allocator.alloc(n)
        self.assertIsNotNone(v)
        half = n // 2
        allocator.free_swa(v[:half])  # slid out of the window -> tombstones
        got = kvcache.translate_loc_from_full_to_swa(v)
        self.assertEqual(got.dtype, torch.int32)
        # Tombstoned ids land on the padding sink, never negative; a captured
        # graph gathers through this result, so a -1 here is an OOB read at
        # replay time, not an error message.
        self.assertTrue(bool((got[:half] == 0).all()), got.tolist())
        self.assertTrue(bool((got[half:] > 0).all()), got.tolist())

    def test_tombstoned_entry_lands_on_the_sink_page_size_1(self):
        self._drive(page_size=1)

    def test_tombstoned_entry_lands_on_the_sink_paged(self):
        """The page-math path: a tombstoned page stays negative through
        `(-1) * ps + offset` for every in-page offset, so clamping the final
        result is exact — pinned here so a refactor to per-page clamping
        keeps the same observable."""
        self._drive(page_size=4)


if __name__ == "__main__":
    unittest.main()
