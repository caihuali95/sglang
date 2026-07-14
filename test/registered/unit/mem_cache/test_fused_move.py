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
"""Fused-slot relocation: `UnifiedMHATokenToKVPool.move_kv_cache`
must move the WHOLE `[host KV | draft KV]` envelope in one call — compaction,
inverse-history rollback, and `move_accept_kv` all route through this method,
so a host-only move would silently strand the draft bytes at the old physical
slots (wrong-slot draft reads after any relocation).

Covers: both regions relocate (ps=1 and ps>1 page-split paths); a non-fused
pool is byte-identical to before (host-only lists); the fused pool's
`get_contiguous_buf_infos` fails loud (PD transfer cannot consume strided
fused rows) while the non-fused pool's stays functional.

CPU-only: advanced-indexing moves over strided views.
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


def _make_host_pool(*, page_size, want_pages=4, fused=True):
    draft_region = (
        MHARegionGeometry(
            layer_num=2,
            head_num=1,
            head_dim=4,
            v_head_dim=4,
            store_dtype=torch.bfloat16,
        )
        if fused
        else None
    )
    full_spec = MHASubPoolSpec(
        name="full",
        grow_direction="down",
        layer_num=2,
        head_num=2,
        head_dim=8,
        v_head_dim=8,
        store_dtype=torch.bfloat16,
        draft_region=draft_region,
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
    pool = UnifiedMHATokenToKVPool(
        unified_buffer=unified,
        sub_pool_name="full",
        page_size=page_size,
        enable_alt_stream=False,
    )
    return unified, pool


def _stamp(views, loc, page_size, value):
    """Write `value` into every (layer, head, elem) of token slot `loc`."""
    page, tok = loc // page_size, loc % page_size
    for view in views:
        view[page, tok] = value


def _read(views, loc, page_size):
    page, tok = loc // page_size, loc % page_size
    return [view[page, tok].clone() for view in views]


class TestFusedMove(CustomTestCase):
    def _move_carries_both_regions(self, page_size):
        unified, pool = _make_host_pool(page_size=page_size)
        host_k, host_v = unified.mha_views_for("full")
        draft_k, draft_v = unified.draft_views_for("full")

        src = max(int(unified.min_slot_index("full")), page_size)  # a real slot
        tgt = src + page_size + 1  # different page AND different slot-in-page

        _stamp(host_k + host_v, src, page_size, 1.25)
        _stamp(draft_k + draft_v, src, page_size, -3.5)
        _stamp(host_k + host_v, tgt, page_size, 0.0)
        _stamp(draft_k + draft_v, tgt, page_size, 0.0)

        # device="cpu" is REQUIRED, not decorative: this is a CPU pool, but on a
        # CUDA box an ambient default-device context sends a bare torch.tensor()
        # to cuda:0, and indexing a CPU buffer with a CUDA index raises.
        pool.move_kv_cache(
            torch.tensor([tgt], dtype=torch.int64, device="cpu"),
            torch.tensor([src], dtype=torch.int64, device="cpu"),
        )

        for t in _read(host_k + host_v, tgt, page_size):
            self.assertTrue(torch.all(t == 1.25), "host bytes did not move")
        for t in _read(draft_k + draft_v, tgt, page_size):
            self.assertTrue(torch.all(t == -3.5), "draft bytes did not move")

    def test_move_carries_both_regions_ps1(self):
        self._move_carries_both_regions(page_size=1)

    def test_move_carries_both_regions_ps4(self):
        self._move_carries_both_regions(page_size=4)

    def test_non_fused_move_unchanged(self):
        unified, pool = _make_host_pool(page_size=1, fused=False)
        self.assertEqual(pool._draft_k_views, [])
        host_k, host_v = unified.mha_views_for("full")
        src, tgt = int(unified.min_slot_index("full")), int(
            unified.min_slot_index("full")
        ) + 1
        _stamp(host_k + host_v, src, 1, 2.5)
        pool.move_kv_cache(
            torch.tensor([tgt], dtype=torch.int64, device="cpu"),
            torch.tensor([src], dtype=torch.int64, device="cpu"),
        )
        for t in _read(host_k + host_v, tgt, 1):
            self.assertTrue(torch.all(t == 2.5))

    def test_fused_contiguous_buf_infos_fails_loud(self):
        _, fused_pool = _make_host_pool(page_size=1, fused=True)
        with self.assertRaises(NotImplementedError):
            fused_pool.get_contiguous_buf_infos()

    def test_non_fused_contiguous_buf_infos_still_works(self):
        _, pool = _make_host_pool(page_size=1, fused=False)
        ptrs, lens, item_lens = pool.get_contiguous_buf_infos()
        self.assertEqual(len(ptrs), 2 * pool.layer_num)


if __name__ == "__main__":
    unittest.main()
