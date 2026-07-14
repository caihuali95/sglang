"""Unified-memory-pool gate for the mamba state-scatter kernels.

Lives in its own file rather than appended to the upstream kernel test: the
capability under test (envelope-strided dst, band-slice src) is ours, and the
upstream test file is not ours to edit.
"""

import unittest

import torch

class TestEnvelopeStridedScatter(unittest.TestCase):
    """Unified-memory-pool gate: the scatter kernels must accept an
    envelope-strided dst (the unified conv/temporal views) and a BAND-SLICE src
    (`intermediate_view[:, band_start:band_start+bs]` of the spec-state
    envelope views) and produce results byte-identical to the dense
    out-of-buffer path. Guards the trailing-dims contiguity relaxation and the
    read-back slice trick."""

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_band_slice_src_envelope_dst_byte_parity(self):
        from sglang.srt.layers.attention.mamba.mamba_state_scatter_triton import (
            fused_conv_window_scatter_with_mask,
            fused_mamba_state_scatter_with_mask,
        )
        from sglang.srt.mem_cache.layout.page_major import (
            build_page_major_mamba_views,
            build_spec_state_views,
            mamba_entry_bytes,
            spec_state_entry_bytes,
        )

        torch.manual_seed(0)
        L, cache_slots, spec_slots, D = 3, 10, 8, 4
        ssm_shape, conv_shape = (2, 3, 4), (6, 3)
        dt = torch.float32

        # Envelope-strided persistent state (dst).
        m_entry = mamba_entry_bytes(
            layer_num=L,
            conv_state_shapes=[conv_shape],
            conv_dtype=dt,
            temporal_state_shape=ssm_shape,
            temporal_dtype=dt,
        )
        raw_m = torch.zeros(cache_slots * m_entry, dtype=torch.uint8, device="cuda")
        conv_views, temporal_view = build_page_major_mamba_views(
            raw_m,
            layer_num=L,
            conv_state_shapes=[conv_shape],
            conv_dtype=dt,
            temporal_state_shape=ssm_shape,
            temporal_dtype=dt,
            max_slots=cache_slots,
        )
        self.assertFalse(temporal_view.is_contiguous())  # the point of the test

        # Envelope-strided spec intermediates (src) + the band slice.
        s_entry = spec_state_entry_bytes(
            layer_num=L,
            num_draft_tokens=D,
            conv_window_shapes=[conv_shape],
            conv_dtype=dt,
            ssm_state_shape=ssm_shape,
            ssm_dtype=dt,
        )
        raw_s = torch.zeros(spec_slots * s_entry, dtype=torch.uint8, device="cuda")
        ssm_view, conv_win_views = build_spec_state_views(
            raw_s,
            layer_num=L,
            num_draft_tokens=D,
            conv_window_shapes=[conv_shape],
            conv_dtype=dt,
            ssm_state_shape=ssm_shape,
            ssm_dtype=dt,
            max_slots=spec_slots,
        )
        band_start, bs = 2, 3
        ssm_src = ssm_view[:, band_start : band_start + bs]
        conv_src = conv_win_views[0][:, band_start : band_start + bs]
        ssm_src.copy_(torch.randn(ssm_src.shape, device="cuda", dtype=dt))
        conv_src.copy_(torch.randn(conv_src.shape, device="cuda", dtype=dt))

        dst_indices = torch.tensor([5, 7, 1], device="cuda", dtype=torch.int32)
        steps = torch.tensor([2, -1, 0], device="cuda", dtype=torch.int32)

        # Dense out-of-buffer reference.
        ref_ssm = torch.zeros(L, cache_slots, *ssm_shape, device="cuda", dtype=dt)
        ref_conv = torch.zeros(L, cache_slots, *conv_shape, device="cuda", dtype=dt)
        for r in range(bs):
            step = int(steps[r].item())
            if step < 0:
                continue
            d = int(dst_indices[r].item())
            ref_ssm[:, d] = ssm_src[:, r, step]
            ref_conv[:, d] = conv_src[:, r, step]

        fused_mamba_state_scatter_with_mask(temporal_view, ssm_src, dst_indices, steps)
        fused_conv_window_scatter_with_mask(conv_views[0], conv_src, dst_indices, steps)

        torch.cuda.synchronize()
        self.assertTrue(torch.equal(temporal_view, ref_ssm))
        self.assertTrue(torch.equal(conv_views[0], ref_conv))

    def test_trailing_contiguity_guard(self):
        # CPU-checkable: the relaxed guard accepts leading-strided tensors and
        # still rejects trailing-strided ones.
        from sglang.srt.layers.attention.mamba.mamba_state_scatter_triton import (
            _trailing_dims_contiguous,
        )

        base = torch.zeros(4, 10, 2, 3)
        lead_strided = base.as_strided((4, 5, 2, 3), (60, 12, 3, 1))
        self.assertTrue(_trailing_dims_contiguous(lead_strided, 2))
        trail_strided = base[:, :, :, ::2]
        self.assertFalse(_trailing_dims_contiguous(trail_strided, 2))


if __name__ == "__main__":
    unittest.main()
