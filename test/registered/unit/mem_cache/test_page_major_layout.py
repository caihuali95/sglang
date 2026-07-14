"""CPU correctness tests for the page-major layer-major envelope layout.

Covers the standalone view builders (no allocator / shared pool):

  - ``build_page_major_mha_views``: 4-D K/V views with correct addressing at
    page_size 1 (token-granularity envelope) and > 1 (layer-major within a page),
    and no aliasing across layers / slots.
  - ``build_page_major_mamba_views``: conv / temporal state views.
  - ``move_kv_cache_native`` 4-D branch: relocating token rows preserves data.

Runs on CPU — pure-torch advanced indexing, no Triton.

    python -m pytest test/registered/unit/mem_cache/test_page_major_layout.py -v
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

import unittest

import torch

from sglang.srt.mem_cache.layout.page_major import (
    build_page_major_mamba_views,
    build_page_major_mha_views,
    build_spec_state_views,
    mamba_entry_bytes,
    mha_entry_bytes,
    spec_state_entry_bytes,
)
from sglang.srt.mem_cache.memory_pool import move_kv_cache_native
from cpu_pool_case import CpuPoolTestCase

_DEV = "cpu"
_DT = torch.float32


def _make_mha_views(layer_num, head_num, head_dim, v_head_dim, page_size, num_pages):
    entry = mha_entry_bytes(
        layer_num=layer_num,
        head_num=head_num,
        head_dim=head_dim,
        v_head_dim=v_head_dim,
        itemsize=_DT.itemsize,
    )
    raw = torch.zeros(num_pages * page_size * entry, dtype=torch.uint8, device=_DEV)
    k, v = build_page_major_mha_views(
        raw,
        layer_num=layer_num,
        head_num=head_num,
        head_dim=head_dim,
        v_head_dim=v_head_dim,
        store_dtype=_DT,
        page_size=page_size,
        num_pages=num_pages,
    )
    return raw, k, v


class TestPageMajorMHAViews(CpuPoolTestCase):
    def test_view_shapes(self):
        _, k, v = _make_mha_views(3, 2, 4, 4, page_size=2, num_pages=4)
        self.assertEqual(len(k), 3)
        for t in k:
            self.assertEqual(tuple(t.shape), (4, 2, 2, 4))
        for t in v:
            self.assertEqual(tuple(t.shape), (4, 2, 2, 4))

    def test_no_aliasing_ps1(self):
        # Every (layer, slot) cell must be independently addressable.
        layer_num, slots = 3, 5
        _, k, v = _make_mha_views(layer_num, 2, 4, 4, page_size=1, num_pages=slots)
        for L in range(layer_num):
            for s in range(slots):
                k[L][s, 0] = float(100 + L * 10 + s)
                v[L][s, 0] = float(200 + L * 10 + s)
        for L in range(layer_num):
            for s in range(slots):
                self.assertTrue(torch.all(k[L][s, 0] == float(100 + L * 10 + s)))
                self.assertTrue(torch.all(v[L][s, 0] == float(200 + L * 10 + s)))

    def test_page_slot_addressing_ps_gt1(self):
        # token id t -> page t // ps, slot t % ps; no aliasing across tokens.
        ps, pages = 2, 4
        total = ps * pages
        _, k, _ = _make_mha_views(2, 1, 2, 2, page_size=ps, num_pages=pages)
        for L in range(2):
            for t in range(total):
                k[L][t // ps, t % ps, 0] = float(1000 + L * 100 + t)
        for L in range(2):
            for t in range(total):
                self.assertEqual(
                    float(k[L][t // ps, t % ps, 0, 0].item()), 1000 + L * 100 + t
                )

    def test_asymmetric_v_head_dim(self):
        _, k, v = _make_mha_views(2, 2, 6, 4, page_size=1, num_pages=3)
        self.assertEqual(tuple(k[0].shape), (3, 1, 2, 6))
        self.assertEqual(tuple(v[0].shape), (3, 1, 2, 4))


class TestPageMajorMove(CpuPoolTestCase):
    def test_move_ps1(self):
        slots = 6
        _, k, v = _make_mha_views(2, 1, 4, 4, page_size=1, num_pages=slots)
        for L in range(2):
            for s in range(slots):
                k[L][s, 0] = float(s + 1)
                v[L][s, 0] = float(-(s + 1))
        tgt = torch.tensor([0, 1], dtype=torch.int64)
        src = torch.tensor([4, 5], dtype=torch.int64)
        move_kv_cache_native(k, v, tgt, src, page_size=1)
        for L in range(2):
            self.assertTrue(torch.all(k[L][0, 0] == 5.0))
            self.assertTrue(torch.all(k[L][1, 0] == 6.0))
            self.assertTrue(torch.all(v[L][0, 0] == -5.0))

    def test_move_ps_gt1(self):
        ps, pages = 2, 4
        total = ps * pages
        _, k, v = _make_mha_views(1, 1, 2, 2, page_size=ps, num_pages=pages)
        for t in range(total):
            k[0][t // ps, t % ps, 0] = float(t + 1)
        tgt = torch.tensor([0, 3], dtype=torch.int64)  # page0 slot0, page1 slot1
        src = torch.tensor([6, 7], dtype=torch.int64)  # page3 slot0, page3 slot1
        move_kv_cache_native(k, v, tgt, src, page_size=ps)
        self.assertEqual(float(k[0][0, 0, 0, 0].item()), 7.0)
        self.assertEqual(float(k[0][1, 1, 0, 0].item()), 8.0)


class TestMambaEnvelopeViews(CpuPoolTestCase):
    def test_conv_temporal_shapes_no_alias(self):
        layers, slots = 2, 4
        conv_shapes = [(2, 3)]
        temp_shape = (2, 2)
        conv_dt, temp_dt = torch.bfloat16, torch.float32
        entry = mamba_entry_bytes(
            layer_num=layers,
            conv_state_shapes=conv_shapes,
            conv_dtype=conv_dt,
            temporal_state_shape=temp_shape,
            temporal_dtype=temp_dt,
        )
        raw = torch.zeros(slots * entry, dtype=torch.uint8, device=_DEV)
        conv_views, temporal = build_page_major_mamba_views(
            raw,
            layer_num=layers,
            conv_state_shapes=conv_shapes,
            conv_dtype=conv_dt,
            temporal_state_shape=temp_shape,
            temporal_dtype=temp_dt,
            max_slots=slots,
        )
        self.assertEqual(tuple(conv_views[0].shape), (layers, slots, 2, 3))
        self.assertEqual(tuple(temporal.shape), (layers, slots, 2, 2))
        for L in range(layers):
            for s in range(slots):
                temporal[L, s] = float(s + L * 10 + 1)
        for L in range(layers):
            for s in range(slots):
                self.assertTrue(torch.all(temporal[L, s] == float(s + L * 10 + 1)))


class TestSpecStateEnvelopeViews(CpuPoolTestCase):
    _LAYERS, _SLOTS, _STEPS = 2, 4, 3
    _CONV_SHAPES = [(2, 3)]
    _SSM_SHAPE = (2, 2, 2)
    _CONV_DT, _SSM_DT = torch.bfloat16, torch.float32

    def _make_views(self):
        entry = spec_state_entry_bytes(
            layer_num=self._LAYERS,
            num_draft_tokens=self._STEPS,
            conv_window_shapes=self._CONV_SHAPES,
            conv_dtype=self._CONV_DT,
            ssm_state_shape=self._SSM_SHAPE,
            ssm_dtype=self._SSM_DT,
        )
        raw = torch.zeros(self._SLOTS * entry, dtype=torch.uint8, device=_DEV)
        ssm, conv_views = build_spec_state_views(
            raw,
            layer_num=self._LAYERS,
            num_draft_tokens=self._STEPS,
            conv_window_shapes=self._CONV_SHAPES,
            conv_dtype=self._CONV_DT,
            ssm_state_shape=self._SSM_SHAPE,
            ssm_dtype=self._SSM_DT,
            max_slots=self._SLOTS,
        )
        return raw, entry, ssm, conv_views

    def test_shapes(self):
        _, _, ssm, conv_views = self._make_views()
        self.assertEqual(
            tuple(ssm.shape),
            (self._LAYERS, self._SLOTS, self._STEPS) + self._SSM_SHAPE,
        )
        self.assertEqual(
            tuple(conv_views[0].shape),
            (self._LAYERS, self._SLOTS, self._STEPS) + self._CONV_SHAPES[0],
        )

    def test_no_aliasing_across_layer_slot_step(self):
        _, _, ssm, conv_views = self._make_views()
        conv = conv_views[0]
        for L in range(self._LAYERS):
            for s in range(self._SLOTS):
                for d in range(self._STEPS):
                    ssm[L, s, d] = float(1000 + L * 100 + s * 10 + d)
                    conv[L, s, d] = float(-(1000 + L * 100 + s * 10 + d))
        for L in range(self._LAYERS):
            for s in range(self._SLOTS):
                for d in range(self._STEPS):
                    self.assertTrue(
                        torch.all(ssm[L, s, d] == float(1000 + L * 100 + s * 10 + d))
                    )
                    self.assertTrue(
                        torch.all(conv[L, s, d] == float(-(1000 + L * 100 + s * 10 + d)))
                    )

    def test_slot_writes_stay_within_entry(self):
        # Writing everything of slot k touches exactly bytes [k*entry, (k+1)*entry).
        raw, entry, ssm, conv_views = self._make_views()
        k = 2
        ssm[:, k] = 1.0
        for conv in conv_views:
            conv[:, k] = 1.0
        nz = raw.view(torch.uint8) != 0
        self.assertTrue(torch.any(nz[k * entry : (k + 1) * entry]))
        self.assertFalse(torch.any(nz[: k * entry]))
        self.assertFalse(torch.any(nz[(k + 1) * entry :]))

    def test_views_tile_raw_exactly(self):
        # Filling every cell of every view touches every byte of raw. Fill
        # values are chosen so EVERY byte of the element is nonzero (0x3f..).
        raw, _, ssm, conv_views = self._make_views()
        ssm_fill = (
            torch.tensor([0x3F3F3F3F], dtype=torch.int32).view(torch.float32).item()
        )
        conv_fill = (
            torch.tensor([0x3F3F], dtype=torch.int16).view(torch.bfloat16).item()
        )
        ssm.fill_(ssm_fill)
        for conv in conv_views:
            conv.fill_(conv_fill)
        self.assertTrue(torch.all(raw.view(torch.uint8) != 0))

    def test_band_slice_matches_reference_zeros_layout(self):
        # A contiguous band slice [band_start : band_start+bs] must behave like
        # rows [0:bs] of a reference torch.zeros dense layout (the read-back
        # slice trick used by the commit scatter).
        _, _, ssm, _ = self._make_views()
        ref = torch.zeros(
            (self._LAYERS, self._SLOTS, self._STEPS) + self._SSM_SHAPE,
            dtype=self._SSM_DT,
            device=_DEV,
        )
        torch.manual_seed(0)
        band_start, bs = 1, 2
        data = torch.randn(
            (self._LAYERS, bs, self._STEPS) + self._SSM_SHAPE, dtype=self._SSM_DT
        )
        ssm[:, band_start : band_start + bs] = data
        ref[:, 0:bs] = data
        self.assertTrue(
            torch.equal(ssm[:, band_start : band_start + bs], ref[:, 0:bs])
        )


if __name__ == "__main__":
    unittest.main()
