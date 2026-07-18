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
"""Envelope-stride regression for ``chunk_gated_delta_rule_fwd_h`` — the
chunk-scan recurrent-state read/write kernel shared by the GDN (`chunk.py`) and
KDA (`kda.py`) EXTEND/prefill paths.

The kernel addressed the per-slot recurrent state as ``index * (H*V*K)`` — the
DENSE contiguous stride. That is correct only for a contiguous
``[num_slots, H, V, K]`` state; under the unified memory pool the state is an
ENVELOPE-strided pool view whose per-slot stride is the full mamba entry
(``entry_bytes // itemsize`` >> ``H*V*K``). The kernel therefore read the prior
state from — and scattered the updated state to — the WRONG slot, silently
corrupting the KDA recurrent state on prefill: the prefill output stayed correct
(the scan runs over in-batch tokens) but every decode step read a stale/empty
state and collapsed (Kimi-Linear GSM8K 0.90 → 0.00 under the unified pool). This
is the same bug class already fixed for the sibling
``fused_sigmoid_gating_delta_rule_update`` (see test_fused_sigmoid_gating_stride).

The fix derives the per-slot stride from ``initial_state.stride(0)`` (and widens
the index to int64). For a contiguous state ``stride(0) == H*V*K`` so GDN /
baseline are BYTE-IDENTICAL; only the envelope-strided case changes. These tests
run identical inputs through a contiguous pool and an envelope-strided pool and
require byte-equal outputs AND written-back states.

    python -m pytest test/registered/unit/layers/test_chunk_delta_h_stride.py -v
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=7, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=7, suite="stage-b-test-1-gpu-small-amd-mi35x")

try:
    from sglang.srt.layers.attention.fla.chunk_delta_h import (
        chunk_gated_delta_rule_fwd_h,
    )

    _IMPORT_ERROR = None
except Exception as e:  # pragma: no cover
    chunk_gated_delta_rule_fwd_h = None
    _IMPORT_ERROR = e


def _envelope_view(shape, slot_stride_elems, dtype, device, fill=None):
    """A page-major-envelope-like strided view: dim-0 (slot) strided by
    ``slot_stride_elems`` (>= per-slot element count), inner dims contiguous —
    the layout ``build_page_major_mamba_views`` produces (per-slot stride =
    entry_bytes // itemsize, much larger than the per-slot payload)."""
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
    not torch.cuda.is_available(), "chunk_gated_delta_rule_fwd_h requires CUDA"
)
class TestChunkDeltaHEnvelopeStride(unittest.TestCase):
    def setUp(self):
        if chunk_gated_delta_rule_fwd_h is None:  # pragma: no cover
            self.skipTest(f"import failed: {_IMPORT_ERROR}")
        torch.manual_seed(1234)
        self.device = "cuda"

    def _make_inputs(self, B, T, H, Hg, K, V, dtype):
        # chunk_gated_delta_rule_fwd_h reads: k [B,T,Hg,K], u [B,T,H,V],
        # w [B,T,H,K], g [B,T,H] (per-head gate). Non-varlen (cu_seqlens=None):
        # one sequence of length T per batch row, indexed by initial_state_indices.
        k = torch.randn(B, T, Hg, K, device=self.device, dtype=dtype)
        w = torch.randn(B, T, H, K, device=self.device, dtype=dtype)
        u = torch.randn(B, T, H, V, device=self.device, dtype=dtype)
        g = torch.randn(B, T, H, device=self.device, dtype=torch.float32)
        return k, w, u, g

    def _run(self, ssm, k, w, u, g, indices):
        return chunk_gated_delta_rule_fwd_h(
            k=k,
            w=w,
            u=u,
            g=g,
            initial_state=ssm,
            initial_state_indices=indices,
            cu_seqlens=None,
        )

    def test_extend_envelope_vs_contiguous_byte_parity(self):
        """Prefill/extend (h0 read + final-state write-back): contiguous and
        envelope-strided states must give byte-identical hidden states AND
        byte-identical written-back recurrent state."""
        B, T, H, Hg, K, V = 5, 64, 8, 4, 64, 64
        slots = 9
        dtype = torch.bfloat16
        k, w, u, g = self._make_inputs(B, T, H, Hg, K, V, dtype)
        ssm_ref = torch.randn(slots, H, V, K, device=self.device, dtype=torch.float32)
        # Non-identity slot mapping so a wrong-slot read/write breaks parity.
        indices = torch.tensor(
            [7, 2, 5, 0, 8], device=self.device, dtype=torch.int32
        )

        ssm_c = ssm_ref.clone()  # contiguous: stride(0) == H*V*K
        h_c, _ = self._run(ssm_c, k, w, u, g, indices)

        # Envelope: per-slot stride ~3.7x the per-slot payload (layer-interleaved
        # scale), inner dims contiguous — like the unified pool's temporal view.
        ssm_e, _ = _envelope_view(
            (slots, H, V, K),
            H * V * K * 3 + 4096,
            torch.float32,
            self.device,
            fill=ssm_ref,
        )
        h_e, _ = self._run(ssm_e, k, w, u, g, indices)

        torch.testing.assert_close(h_c, h_e, rtol=0, atol=0)
        # Final recurrent state written back in place at the referenced slots.
        torch.testing.assert_close(ssm_c, ssm_e.contiguous(), rtol=0, atol=0)

    def test_multi_chunk_envelope_vs_contiguous(self):
        """T spanning multiple chunks (state read once, written once): same
        parity requirement across the full scan."""
        B, T, H, Hg, K, V = 3, 192, 8, 8, 64, 64  # 3 chunks of 64
        dtype = torch.bfloat16
        k, w, u, g = self._make_inputs(B, T, H, Hg, K, V, dtype)
        slots = 7
        ssm_ref = torch.randn(slots, H, V, K, device=self.device, dtype=torch.float32)
        indices = torch.tensor([6, 1, 4], device=self.device, dtype=torch.int32)

        ssm_c = ssm_ref.clone()
        h_c, _ = self._run(ssm_c, k, w, u, g, indices)
        ssm_e, _ = _envelope_view(
            (slots, H, V, K),
            H * V * K * 2 + 8192,
            torch.float32,
            self.device,
            fill=ssm_ref,
        )
        h_e, _ = self._run(ssm_e, k, w, u, g, indices)

        torch.testing.assert_close(h_c, h_e, rtol=0, atol=0)
        torch.testing.assert_close(ssm_c, ssm_e.contiguous(), rtol=0, atol=0)

    def test_large_stride_int64_no_overflow(self):
        """Production-scale envelope slot stride: index * slot_stride past 2**31
        must not wrap. slot 88 x ~26M elems ≈ 2.3e9 > int32 max — the pre-fix
        int32 index load would fault or mis-address."""
        B, T, H, Hg, K, V = 1, 64, 4, 4, 64, 64
        dtype = torch.bfloat16
        k, w, u, g = self._make_inputs(B, T, H, Hg, K, V, dtype)
        inner = H * V * K
        slot_stride = 26_000_000  # ~99 MiB/slot at fp32 — production envelope scale
        slots = 89
        indices = torch.tensor([slots - 1], device=self.device, dtype=torch.int32)

        ssm_ref = torch.randn(1, H, V, K, device=self.device, dtype=torch.float32)
        # Contiguous reference at a small slot count with identity index 0.
        ssm_c = ssm_ref.clone()
        h_c, _ = self._run(ssm_c, k, w, u, g, torch.zeros(1, device=self.device, dtype=torch.int32))

        ssm_e, _ = _envelope_view(
            (slots, H, V, K), slot_stride, torch.float32, self.device
        )
        ssm_e[slots - 1].copy_(ssm_ref[0])
        h_e, _ = self._run(ssm_e, k, w, u, g, indices)

        torch.testing.assert_close(h_c, h_e, rtol=0, atol=0)
        torch.testing.assert_close(
            ssm_c[0], ssm_e[slots - 1].contiguous(), rtol=0, atol=0
        )


if __name__ == "__main__":
    unittest.main()
