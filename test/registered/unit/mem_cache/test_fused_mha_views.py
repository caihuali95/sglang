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
"""Layout correctness of fused draft-KV slots (fused draft KV): ``MHASubPoolSpec.draft_region`` + ``UnifiedKVPool`` draft views.

A fused slot is one contiguous envelope ``[host KV | pad | draft KV]``; within
a page block the layout is ``[host L0_K·ps | host L0_V·ps | ... | pad·ps |
draft D0_K·ps | draft D0_V·ps | ...]`` and BOTH regions share
``stride[0] = ps * entry_bytes() / itemsize``. These tests prove:

  - draft-view writes land at independently computed raw-byte offsets
    (round-trip against a reference formula, not against the views);
  - host and draft regions never alias (within a slot, across layers,
    across pages) — filling one leaves the other intact;
  - host and draft views share the fused per-page stride;
  - asymmetric draft geometry (EAGLE3/DFLASH-like head dims != host's) and
    ps in {1, 64, 256};
  - dtype alignment: the draft region base aligns up to the draft itemsize;
  - gate-off identity: ``draft_region=None`` reproduces the pre-fusion
    entry_bytes/strides/offsets exactly (byte-identical layout).

CPU-only: pure ``as_strided`` view math over a uint8 buffer.
"""

import unittest

import torch

from sglang.srt.mem_cache.unified_memory_pool import (
    MHARegionGeometry,
    MHASubPoolSpec,
    UnifiedKVPool,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _make_pool(*, page_size, want_pages, host_kwargs=None, draft_region=None):
    """Two-end-pool ``UnifiedKVPool``: the fused "full" pool under test
    (grow-down) + a minimal plain MHA peer (grow-up)."""
    full_spec = MHASubPoolSpec(
        name="full",
        grow_direction="down",
        layer_num=2,
        head_num=2,
        head_dim=8,
        v_head_dim=8,
        store_dtype=torch.bfloat16,
        draft_region=draft_region,
        **(host_kwargs or {}),
    )
    peer_spec = MHASubPoolSpec(
        name="peer",
        grow_direction="up",
        layer_num=1,
        head_num=1,
        head_dim=4,
        store_dtype=torch.bfloat16,
    )
    total_bytes = (want_pages + 2) * page_size * full_spec.entry_bytes()
    pool = UnifiedKVPool(
        total_bytes=total_bytes,
        sub_pool_specs=[full_spec, peer_spec],
        device="cpu",
        enable_memory_saver=False,
        page_size=page_size,
    )
    return pool, full_spec


def _draft_elem_byte_offset(
    spec, *, page_size, layer, page, slot, head, elem, is_v=False
):
    """Independent reference: raw-byte offset of one draft-region element."""
    region = spec.draft_region
    itemsize = region.store_dtype.itemsize
    dk_row = region.k_row_bytes()
    dv_row = region.v_row_bytes()
    off = page * page_size * spec.entry_bytes()  # page block base
    off += page_size * spec.draft_region_offset_bytes()  # draft region base
    off += layer * page_size * (dk_row + dv_row)  # layer block
    if is_v:
        off += page_size * dk_row  # V follows layer's K block
        row = dv_row
        dim = region.v_head_dim
    else:
        row = dk_row
        dim = region.head_dim
    off += slot * row  # slot within the K (or V) block
    off += (head * dim + elem) * itemsize
    return off


class TestFusedMhaViews(CustomTestCase):
    def _round_trip(self, page_size):
        region = MHARegionGeometry(
            layer_num=3,
            head_num=1,
            head_dim=16,
            v_head_dim=16,
            store_dtype=torch.bfloat16,
        )
        pool, spec = _make_pool(
            page_size=page_size, want_pages=3, draft_region=region
        )
        self.assertTrue(pool.has_draft_region("full"))
        k_views, v_views = pool.draft_views_for("full")
        self.assertEqual(len(k_views), region.layer_num)

        probes = [
            # (layer, page, slot, head, elem, is_v, sentinel)
            (0, 0, 0, 0, 0, False, 1.5),
            (1, 1, page_size - 1, 0, 7, False, -2.0),
            (2, 2, page_size // 2, 0, 15, True, 3.25),
        ]
        raw = pool._raw
        for layer, page, slot, head, elem, is_v, val in probes:
            views = v_views if is_v else k_views
            views[layer][page, slot, head, elem] = val
            off = _draft_elem_byte_offset(
                spec,
                page_size=page_size,
                layer=layer,
                page=page,
                slot=slot,
                head=head,
                elem=elem,
                is_v=is_v,
            )
            got = raw[off : off + 2].view(torch.bfloat16).item()
            self.assertEqual(got, val, msg=f"probe {(layer, page, slot, is_v)}")
            # And the reverse direction: write raw, read view.
            raw[off : off + 2].view(torch.bfloat16)[0] = val * 2
            self.assertEqual(views[layer][page, slot, head, elem].item(), val * 2)

    def test_round_trip_ps1(self):
        self._round_trip(page_size=1)

    def test_round_trip_ps64(self):
        self._round_trip(page_size=64)

    def test_round_trip_ps256(self):
        self._round_trip(page_size=256)

    def test_host_and_draft_do_not_alias(self):
        region = MHARegionGeometry(
            layer_num=2,
            head_num=3,  # asymmetric: differs from host head_num=2
            head_dim=4,
            v_head_dim=2,
            store_dtype=torch.bfloat16,
        )
        pool, spec = _make_pool(page_size=4, want_pages=4, draft_region=region)
        host_k, host_v = pool.mha_views_for("full")
        draft_k, draft_v = pool.draft_views_for("full")

        for t in draft_k + draft_v:
            t.fill_(2.0)
        for t in host_k + host_v:
            t.fill_(1.0)
        # Host fill after draft fill must leave every draft element intact.
        for t in draft_k + draft_v:
            self.assertTrue(torch.all(t == 2.0), "host writes leaked into draft")
        for t in host_k + host_v:
            self.assertTrue(torch.all(t == 1.0))
        # Draft layers must not alias each other either.
        draft_k[0].fill_(5.0)
        self.assertTrue(torch.all(draft_k[1] == 2.0))
        self.assertTrue(torch.all(draft_v[0] == 2.0))

    def test_shared_page_stride(self):
        region = MHARegionGeometry(
            layer_num=1,
            head_num=2,
            head_dim=8,
            v_head_dim=8,
            store_dtype=torch.bfloat16,
        )
        pool, spec = _make_pool(page_size=16, want_pages=2, draft_region=region)
        host_k, _ = pool.mha_views_for("full")
        draft_k, _ = pool.draft_views_for("full")
        expected = 16 * spec.entry_bytes() // torch.bfloat16.itemsize
        self.assertEqual(host_k[0].stride(0), expected)
        self.assertEqual(draft_k[0].stride(0), expected)

    def test_draft_offset_aligns_to_draft_itemsize(self):
        # Host entry = 1 layer * 1 head * (1 K + 2 V) elems * 2 B = 6 bytes;
        # an fp32 draft region must start at 8, not 6.
        region = MHARegionGeometry(
            layer_num=1,
            head_num=1,
            head_dim=2,
            v_head_dim=2,
            store_dtype=torch.float32,
        )
        spec = MHASubPoolSpec(
            name="full",
            grow_direction="down",
            layer_num=1,
            head_num=1,
            head_dim=1,
            v_head_dim=2,
            store_dtype=torch.bfloat16,
            draft_region=region,
        )
        self.assertEqual(spec.host_entry_bytes(), 6)
        self.assertEqual(spec.draft_region_offset_bytes(), 8)
        self.assertEqual(spec.entry_bytes(), 8 + region.entry_bytes())

    def test_gate_off_layout_is_byte_identical(self):
        # draft_region=None must reproduce the pre-fusion layout exactly.
        pool, spec = _make_pool(page_size=4, want_pages=4, draft_region=None)
        self.assertFalse(pool.has_draft_region("full"))
        self.assertEqual(spec.entry_bytes(), spec.host_entry_bytes())
        k_views, v_views = pool.mha_views_for("full")
        itemsize = torch.bfloat16.itemsize
        k_row = spec.k_row_bytes()
        v_row = spec.v_row_bytes()
        # Pre-fusion formulas, transcribed: stride[0] = ps*entry/itemsize,
        # layer L's K base = L*ps*(k_row+v_row), V follows K.
        self.assertEqual(
            k_views[0].stride(0), 4 * spec.entry_bytes() // itemsize
        )
        for layer in range(spec.layer_num):
            k_base = layer * 4 * (k_row + v_row)
            self.assertEqual(k_views[layer].storage_offset(), k_base // itemsize)
            self.assertEqual(
                v_views[layer].storage_offset(),
                (k_base + 4 * k_row) // itemsize,
            )


if __name__ == "__main__":
    unittest.main()
