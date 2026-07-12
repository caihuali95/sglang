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
"""The KV readback check (`SGLANG_DEBUG_FUSED_KV_READBACK`) must actually FIRE.

The write-loc checks verify WHERE a store is aimed and are blind to HOW FAR it
reaches: a store with a wrong row stride passes them while spilling out of its
region of the fused slot -- right slot, wrong bytes. This check exists to see
that, so it has to be proven to catch it, or a clean run means nothing.

Pins the two failure modes, using the pool's own peer-view machinery so the test
exercises the same code the GPU run does:
  (a) the store does not land the bytes it was given (readback mismatch),
  (b) the store lands its own bytes correctly but CLOBBERS the peer region --
      the spill the readback alone cannot see.
plus the must-NOT-fire case: a correct, in-region store.

CPU-only: the checks are pure tensor comparisons over the strided views.
"""

import unittest

import torch

from sglang.srt.mem_cache.unified_memory_pool import (
    MHARegionGeometry,
    MHASubPoolSpec,
    UnifiedKVPool,
    UnifiedMHATokenToKVPool,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

PAGE_SIZE = 2


def _make_fused_pool():
    region = MHARegionGeometry(
        layer_num=1,
        head_num=1,
        head_dim=4,
        v_head_dim=4,
        store_dtype=torch.bfloat16,
    )
    full_spec = MHASubPoolSpec(
        name="full",
        grow_direction="down",
        layer_num=1,
        head_num=1,
        head_dim=4,
        v_head_dim=4,
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
        total_bytes=8 * PAGE_SIZE * full_spec.entry_bytes(),
        sub_pool_specs=[full_spec, peer_spec],
        device="cpu",
        enable_memory_saver=False,
        page_size=PAGE_SIZE,
    )
    pool = UnifiedMHATokenToKVPool(
        unified_buffer=unified,
        sub_pool_name="full",
        page_size=PAGE_SIZE,
        enable_alt_stream=False,
    )
    return unified, pool


class TestKVReadbackCheck(CustomTestCase):
    def setUp(self):
        self.unified, self.pool = _make_fused_pool()
        base = max(int(self.unified.min_slot_index("full")), PAGE_SIZE)
        self.loc = torch.tensor([base, base + PAGE_SIZE + 1], dtype=torch.int64)
        n = self.loc.numel()
        self.k = torch.full((n, 1, 4), 1.5, dtype=torch.bfloat16)
        self.v = torch.full((n, 1, 4), -2.5, dtype=torch.bfloat16)
        self.k_view = self.pool.k_buffer[0]
        self.v_view = self.pool.v_buffer[0]

    def _store(self):
        """Write correctly, the way the real store kernel is supposed to."""
        page, tok = self.loc // PAGE_SIZE, self.loc % PAGE_SIZE
        self.k_view[page, tok] = self.k
        self.v_view[page, tok] = self.v

    def _check(self, peer_before):
        self.pool._debug_readback(
            0, self.loc, self.k, self.v, self.k_view, self.v_view, PAGE_SIZE, peer_before
        )

    def test_correct_store_passes(self):
        peer_before = self.pool._debug_peer_snapshot(self.loc, PAGE_SIZE)
        self._store()
        self._check(peer_before)  # must not raise

    def test_store_that_lands_wrong_bytes_fires(self):
        # (a) the store did not put down what it was handed.
        peer_before = self.pool._debug_peer_snapshot(self.loc, PAGE_SIZE)
        self._store()
        page, tok = self.loc[0] // PAGE_SIZE, self.loc[0] % PAGE_SIZE
        self.k_view[page, tok, 0, 2] = 99.0  # one element off
        with self.assertRaises(AssertionError) as cm:
            self._check(peer_before)
        msg = str(cm.exception)
        self.assertIn("did NOT land the bytes it was given", msg)
        self.assertIn(str(int(self.loc[0])), msg)  # names the physical slot

    def test_store_that_clobbers_the_peer_region_fires(self):
        # (b) THE case the readback alone is blind to: every byte the store meant
        # to write landed correctly, but its extent overran into the draft region.
        peer_before = self.pool._debug_peer_snapshot(self.loc, PAGE_SIZE)
        self._store()  # own region: perfect
        draft_k, _ = self.unified.draft_views_for("full")
        page, tok = self.loc[1] // PAGE_SIZE, self.loc[1] % PAGE_SIZE
        draft_k[0][page, tok] = 7.0  # the spill
        with self.assertRaises(AssertionError) as cm:
            self._check(peer_before)
        msg = str(cm.exception)
        self.assertIn("CLOBBERED the draft region", msg)
        self.assertIn("EXTENT overruns", msg)
        self.assertIn(str(int(self.loc[1])), msg)

    def test_untouched_peer_does_not_fire(self):
        # The peer legitimately holds data from earlier writes; only a CHANGE
        # during this store is a spill.
        draft_k, draft_v = self.unified.draft_views_for("full")
        for t in draft_k + draft_v:
            t.fill_(3.0)
        peer_before = self.pool._debug_peer_snapshot(self.loc, PAGE_SIZE)
        self._store()
        self._check(peer_before)  # must not raise

    def test_duplicate_and_sink_locs_are_excluded(self):
        # Two tokens writing one slot legitimately leave a value matching neither
        # source row; the sink slot is written by every padded row. Judging either
        # would be a false positive.
        base = max(int(self.unified.min_slot_index("full")), PAGE_SIZE)
        self.loc = torch.tensor([base, base, 0], dtype=torch.int64)
        n = self.loc.numel()
        self.k = torch.arange(n * 4, dtype=torch.bfloat16).reshape(n, 1, 4)
        self.v = torch.arange(n * 4, dtype=torch.bfloat16).reshape(n, 1, 4)
        peer_before = self.pool._debug_peer_snapshot(self.loc, PAGE_SIZE)
        self._store()  # last writer wins at `base`; slot 0 is the sink
        self._check(peer_before)  # must not raise



class TestStoreRowContract(CustomTestCase):
    """`store_cache_4d_kernel` flattens (head_num, head_dim) into one ROW_DIM and
    is handed only `stride(0)`. A source whose row elements are not adjacent is
    therefore read with the wrong element spacing -- the store lands the wrong
    bytes at the RIGHT slot, which no location-based check can see.

    `store_cache_4d()` asserts this contract, but the unified pool calls the
    kernel directly (the wrapper cannot merge its 4-D layer-major view at
    page_size > 1), so the contract went unchecked. Pin the normalization.
    """

    def test_row_compact_sources_pass_through_uncopied(self):
        from sglang.srt.mem_cache.unified_memory_pool import _as_row_contiguous

        # Plain contiguous, and a row-compact SLICE of a wider tensor (what a fused
        # QKV projection hands over): stride(0) is wide, but rows are adjacent --
        # legal, and must not be copied.
        plain = torch.zeros(4, 2, 8)
        qkv = torch.zeros(4, 3 * 2 * 8).view(4, 3, 2, 8)[:, 1]  # the K slice
        self.assertEqual(qkv.stride(), (48, 8, 1))
        for t in (plain, qkv):
            self.assertIs(_as_row_contiguous(t), t)

    def test_non_row_contiguous_source_is_normalized(self):
        from sglang.srt.mem_cache.unified_memory_pool import _as_row_contiguous

        # head-major -> head_dim-major transpose: rows are NOT adjacent. The kernel
        # would read every element with the wrong spacing.
        t = torch.arange(4 * 2 * 8, dtype=torch.float32).reshape(4, 8, 2).transpose(1, 2)
        self.assertNotEqual(t.stride(-2), t.shape[-1])
        out = _as_row_contiguous(t)
        self.assertEqual(out.stride(-1), 1)
        self.assertEqual(out.stride(-2), out.shape[-1])
        self.assertTrue(torch.equal(out, t))  # same values, compact layout

    def test_store_through_a_transposed_source_round_trips(self):
        # End to end through the pool: a non-row-contiguous source must still land
        # byte-exact. Before the normalization this silently stored garbage.
        unified, pool = _make_fused_pool()
        base = max(int(unified.min_slot_index("full")), PAGE_SIZE)
        loc = torch.tensor([base, base + PAGE_SIZE + 1], dtype=torch.int64)
        src = torch.arange(2 * 4 * 1, dtype=torch.bfloat16).reshape(2, 4, 1)
        k = src.transpose(1, 2)  # (2, 1, 4), stride (4, 1, 1) -- rows NOT adjacent
        self.assertNotEqual(k.stride(-2), k.shape[-1])
        v = k.clone()

        from sglang.srt.mem_cache.unified_memory_pool import _as_row_contiguous

        kc, vc = _as_row_contiguous(k), _as_row_contiguous(v)
        page, tok = loc // PAGE_SIZE, loc % PAGE_SIZE
        pool.k_buffer[0][page, tok] = kc
        pool.v_buffer[0][page, tok] = vc
        pool._debug_readback(
            0, loc, kc, vc, pool.k_buffer[0], pool.v_buffer[0], PAGE_SIZE, None
        )
        self.assertTrue(torch.equal(pool.k_buffer[0][page, tok], k))

if __name__ == "__main__":
    unittest.main()
