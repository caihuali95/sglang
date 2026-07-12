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
"""`store_cache_4d_kernel`'s row contract, enforced at the unified pool.

The kernel flattens (head_num, head_dim) into one ROW_DIM addressed by a linear
offset and is handed only `stride(0)`, so a source whose row elements are not
adjacent in memory is read with the wrong element spacing -- the store lands the
WRONG BYTES at the RIGHT slot. `store_cache_4d()` asserts the contract, but the
unified pool calls the kernel directly (the wrapper cannot merge its 4-D
layer-major view at page_size > 1), so it must normalize the source itself.

The same hazard applies to `loc`, which the kernel reads as `tl.load(loc_ptr + i)`
-- raw pointer math, strides ignored. That one was live: a strided per-step draft
loc made the store consume the step-INTERLEAVED window, silently misplacing draft
KV (see test_per_step_out_cache_loc.py). No location-based check can see either
failure, so the normalization is the guard.

CPU-only: pure layout arithmetic.
"""

import unittest

import torch

from sglang.srt.mem_cache.unified_memory_pool import _as_row_contiguous
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestStoreRowContract(CustomTestCase):
    def test_row_compact_sources_pass_through_uncopied(self):
        # Plain contiguous, and a row-compact SLICE of a wider tensor (what a fused
        # QKV projection hands over): stride(0) is wide, but rows are adjacent --
        # legal, and must not be copied (the kernel takes stride(0) as an argument).
        plain = torch.zeros(4, 2, 8)
        qkv = torch.zeros(4, 3 * 2 * 8).view(4, 3, 2, 8)[:, 1]  # the K slice
        self.assertEqual(qkv.stride(), (48, 8, 1))
        for t in (plain, qkv):
            self.assertIs(_as_row_contiguous(t), t)

    def test_non_row_contiguous_source_is_normalized(self):
        # head-major -> head_dim-major transpose: rows are NOT adjacent, so the
        # kernel would read every element with the wrong spacing.
        t = torch.arange(4 * 2 * 8, dtype=torch.float32).reshape(4, 8, 2).transpose(
            1, 2
        )
        self.assertNotEqual(t.stride(-2), t.shape[-1])
        out = _as_row_contiguous(t)
        self.assertEqual(out.stride(-1), 1)
        self.assertEqual(out.stride(-2), out.shape[-1])
        self.assertTrue(torch.equal(out, t))  # same values, compact layout


if __name__ == "__main__":
    unittest.main()
