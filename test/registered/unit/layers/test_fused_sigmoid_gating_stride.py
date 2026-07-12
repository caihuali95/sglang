from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=7, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=7, suite="stage-b-test-1-gpu-small-amd-mi35x")

import unittest

import torch

try:
    from sglang.srt.layers.attention.fla.fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_update,
    )

    _IMPORT_ERROR = None
except Exception as e:  # pragma: no cover
    fused_sigmoid_gating_delta_rule_update = None
    _IMPORT_ERROR = e


def _envelope_view(shape, slot_stride_elems, dtype, device, fill=None):
    """Build a page-major-envelope-like strided view: slot dim strided by
    ``slot_stride_elems`` (>= per-slot elements), inner dims contiguous —
    the layout `build_page_major_mamba_views` / `build_spec_state_views`
    produce (slot stride = entry_bytes // itemsize, layer-interleaved)."""
    slots = shape[0]
    inner = 1
    for d in shape[1:]:
        inner *= d
    assert slot_stride_elems >= inner
    raw = torch.zeros(slots * slot_stride_elems, dtype=dtype, device=device)
    inner_strides = []
    s = 1
    for d in reversed(shape[1:]):
        inner_strides.append(s)
        s *= d
    view = torch.as_strided(
        raw, size=shape, stride=(slot_stride_elems, *reversed(inner_strides))
    )
    if fill is not None:
        view.copy_(fill)
    return view, raw


@unittest.skipIf(
    not torch.cuda.is_available(), "fused_sigmoid_gating kernels require CUDA"
)
class TestFusedSigmoidGatingEnvelopeStride(unittest.TestCase):
    """Envelope-stride regression: the kernel
    hard-coded the h0 slot stride (`HV*K*V`) and the intermediate-cache
    slot/step strides (`cache_steps*HV*K*V` / `HV*K*V`), silently reading and
    writing the WRONG slot for the page-major envelope's strided pool views —
    the GDN spec target-verify state corruption (GSM8K 0.90 -> 0.00). The fix
    derives the strides from the tensors; contiguous callers are bit-identical.
    These tests run the SAME inputs through a contiguous pool and an
    envelope-strided pool and require byte-equal outputs and states."""

    def setUp(self):
        if fused_sigmoid_gating_delta_rule_update is None:  # pragma: no cover
            self.skipTest(f"import failed: {_IMPORT_ERROR}")
        torch.manual_seed(1234)
        self.device = "cuda"

    def _make_inputs(self, B, T, H, HV, K, V, dtype):
        # Layouts mirror gdn_backend's target_verify/decode call: q/k [1, B*T, H, K],
        # v [1, B*T, HV, V], a/b [B*T, HV], varlen cu_seqlens.
        BT = B * T
        q = torch.randn(1, BT, H, K, device=self.device, dtype=dtype)
        k = torch.randn(1, BT, H, K, device=self.device, dtype=dtype)
        v = torch.randn(1, BT, HV, V, device=self.device, dtype=dtype)
        a = torch.randn(BT, HV, device=self.device, dtype=dtype)
        b = torch.randn(BT, HV, device=self.device, dtype=dtype)
        A_log = torch.randn(HV, device=self.device, dtype=torch.float32)
        dt_bias = torch.randn(HV, device=self.device, dtype=torch.float32)
        cu = torch.arange(0, BT + 1, T, device=self.device, dtype=torch.int32)
        return q, k, v, a, b, A_log, dt_bias, cu

    def _run(self, ssm, cache, q, k, v, a, b, A_log, dt_bias, cu, indices, *,
             disable_state_update, cache_indices=None, cache_steps=None):
        return fused_sigmoid_gating_delta_rule_update(
            A_log=A_log,
            dt_bias=dt_bias,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            q=q,
            k=k,
            v=v,
            b=b,
            a=a,
            initial_state_source=ssm,
            initial_state_indices=indices,
            cu_seqlens=cu,
            use_qk_l2norm_in_kernel=True,
            is_kda=False,
            disable_state_update=disable_state_update,
            intermediate_states_buffer=cache,
            intermediate_state_indices=cache_indices,
            cache_steps=cache_steps,
        )

    def test_decode_mode_envelope_vs_contiguous_byte_parity(self):
        """Decode (T=1, state write-back ON): h0 read + write-back stride."""
        B, T, H, HV, K, V = 5, 1, 4, 8, 64, 64
        slots = 9
        dtype = torch.bfloat16
        inputs = self._make_inputs(B, T, H, HV, K, V, dtype)
        ssm_ref = torch.randn(
            slots, HV, V, K, device=self.device, dtype=torch.float32
        )
        # Non-identity slot mapping so a wrong-slot read/write breaks parity.
        indices = torch.tensor([7, 2, 5, 0, 8], device=self.device, dtype=torch.int32)

        ssm_c = ssm_ref.clone()  # contiguous: slot stride == HV*V*K
        o_c = self._run(ssm_c, None, *inputs, indices, disable_state_update=False)

        # Envelope: slot stride ~3.7x the per-slot size, layer-interleaved-like.
        ssm_e, _ = _envelope_view(
            (slots, HV, V, K), HV * V * K * 3 + 4096, torch.float32,
            self.device, fill=ssm_ref,
        )
        o_e = self._run(ssm_e, None, *inputs, indices, disable_state_update=False)

        torch.testing.assert_close(o_c, o_e, rtol=0, atol=0)
        torch.testing.assert_close(ssm_c, ssm_e.contiguous(), rtol=0, atol=0)

    def test_target_verify_envelope_vs_contiguous_byte_parity(self):
        """Verify (T=D, write-back OFF, intermediate cache ON): h0 read stride
        + intermediate slot/step write strides."""
        B, D, H, HV, K, V = 4, 4, 4, 8, 64, 64
        slots, band_slots = 9, 6
        dtype = torch.bfloat16
        inputs = self._make_inputs(B, D, H, HV, K, V, dtype)
        ssm_ref = torch.randn(
            slots, HV, V, K, device=self.device, dtype=torch.float32
        )
        indices = torch.tensor([6, 1, 4, 8], device=self.device, dtype=torch.int32)
        cache_indices = torch.tensor(
            [3, 0, 5, 2], device=self.device, dtype=torch.int32
        )

        ssm_c = ssm_ref.clone()
        cache_c = torch.zeros(
            band_slots, D, HV, V, K, device=self.device, dtype=torch.float32
        )
        o_c = self._run(
            ssm_c, cache_c, *inputs, indices,
            disable_state_update=True, cache_indices=cache_indices, cache_steps=D,
        )

        ssm_e, _ = _envelope_view(
            (slots, HV, V, K), HV * V * K * 3 + 4096, torch.float32,
            self.device, fill=ssm_ref,
        )
        cache_e, _ = _envelope_view(
            (band_slots, D, HV, V, K), D * HV * V * K * 2 + 8192, torch.float32,
            self.device,
        )
        o_e = self._run(
            ssm_e, cache_e, *inputs, indices,
            disable_state_update=True, cache_indices=cache_indices, cache_steps=D,
        )

        torch.testing.assert_close(o_c, o_e, rtol=0, atol=0)
        # Band contents byte-equal (the commit scatter reads these).
        torch.testing.assert_close(cache_c, cache_e.contiguous(), rtol=0, atol=0)
        # Verify mode must NOT write back the persistent state.
        torch.testing.assert_close(ssm_e.contiguous(), ssm_ref, rtol=0, atol=0)

    def test_large_stride_int64_no_overflow(self):
        """Overflow guard: idx * slot_stride past 2**31 must not wrap.
        Production-scale envelope slot stride (~2.6e7 elems) x slot 88 ≈ 2.3e9
        elements > int32 max; the un-cast int32 load would fault or mis-copy."""
        B, T, H, HV, K, V = 1, 1, 2, 4, 64, 64
        dtype = torch.bfloat16
        inputs = self._make_inputs(B, T, H, HV, K, V, dtype)
        inner = HV * V * K  # 2**16 elems
        slot_stride = 26_000_000  # ~99 MiB/slot at fp32 — production envelope scale
        slots = 89
        indices = torch.tensor([slots - 1], device=self.device, dtype=torch.int32)
        need_gib = slots * slot_stride * 4 / (1 << 30)
        free, _ = torch.cuda.mem_get_info()
        if free < (need_gib + 1) * (1 << 30):  # pragma: no cover
            self.skipTest(f"needs ~{need_gib:.1f} GiB free GPU memory")

        ssm_ref = torch.randn(slots, HV, V, K, device=self.device, dtype=torch.float32)
        ssm_c = ssm_ref.clone()
        o_c = self._run(ssm_c, None, *inputs, indices, disable_state_update=False)
        ssm_e, _ = _envelope_view(
            (slots, HV, V, K), slot_stride, torch.float32, self.device, fill=ssm_ref
        )
        o_e = self._run(ssm_e, None, *inputs, indices, disable_state_update=False)
        torch.testing.assert_close(o_c, o_e, rtol=0, atol=0)
        torch.testing.assert_close(
            ssm_c[slots - 1], ssm_e[slots - 1].contiguous(), rtol=0, atol=0
        )


if __name__ == "__main__":
    unittest.main()
