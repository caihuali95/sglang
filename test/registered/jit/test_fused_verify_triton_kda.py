"""Tests for the fused sigmoid gating delta rule verify kernel with the KDA
(per-K decay vector) gate — the KDA target_verify path.

Compares `fused_sigmoid_gating_delta_rule_update(is_kda=True, ...)` in verify
mode (disable_state_update + per-step intermediate capture + optional tree
ancestors) against the reference two-step implementation used by
test_kda_kernels.py:
    1. torch-side gate: g = -exp(A_log) * softplus(a + dt_bias)  (per-K),
       beta = sigmoid(b)
    2. o = fused_recurrent_kda(q, k, v, g, beta, ...)

This doubles as the first production-grade coverage of the `is_kda=True`
fused-recurrent path itself: on CUDA the decode path always takes
`fused_recurrent_kda_packed_decode`, so before the KDA verify branch the
gate-generic kernel's KDA flavor ran nowhere in production.
"""

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

try:
    from sglang.srt.layers.attention.fla.fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_update,
    )
    from sglang.srt.layers.attention.fla.kda import fused_recurrent_kda

    KERNELS_AVAILABLE = True
except ImportError:
    KERNELS_AVAILABLE = False

register_cuda_ci(est_time=8, suite="base-b-kernel-unit-1-gpu-large")
register_cuda_ci(est_time=120, suite="nightly-kernel-1-gpu", nightly=True)


def _make_tensors(N, T, H, K, V, device="cuda", seed=2026, num_slots=None):
    """Inputs for KDA target_verify: N requests x T draft tokens.

    KDA layouts (mirroring test_kda_kernels.py setUp): per-K decay `a` flat
    [1, N*T, H*K], `dt_bias` flat [H*K], A_log [1, 1, H, 1] fp32, raw beta
    [1, N*T, H] (kernel applies sigmoid in-kernel).
    """
    torch.manual_seed(seed)
    num_slots = num_slots or (3 * N + 2)
    A_log = torch.randn(1, 1, H, 1, dtype=torch.float32, device=device)
    dt_bias = torch.randn(H * K, dtype=torch.bfloat16, device=device)
    a = torch.randn(1, N * T, H * K, dtype=torch.bfloat16, device=device)
    b = torch.randn(1, N * T, H, dtype=torch.bfloat16, device=device)
    q = torch.randn(1, N * T, H, K, dtype=torch.bfloat16, device=device)
    k = torch.randn(1, N * T, H, K, dtype=torch.bfloat16, device=device)
    v = torch.randn(1, N * T, H, V, dtype=torch.bfloat16, device=device)
    # Non-identity, spread-out persistent slots (test_kda_kernels idiom).
    cache_indices = torch.randperm(num_slots, device=device)[:N].to(torch.int32)
    ssm_states = torch.randn(
        num_slots, H, K, V, dtype=torch.float32, device=device
    )
    cu_seqlens = torch.arange(0, N * T + 1, T, dtype=torch.int32, device=device)
    return A_log, dt_bias, a, b, q, k, v, ssm_states, cache_indices, cu_seqlens


def _reference_chain(A_log, dt_bias, q, k, v, a, b, initial_state, cu_seqlens):
    """Torch-gate + fused_recurrent_kda reference (from test_kda_kernels.py).

    `initial_state` is a GATHERED clone [N, H, K, V]; returns (out, last_state)
    where last_state is per-row [N, H, K, V].
    """
    H = q.shape[-2]
    K = q.shape[-1]
    beta = b.float().sigmoid()
    raw_g = a.float() + dt_bias.float()
    g = -torch.exp(A_log.float().view(1, 1, H, 1)) * torch.nn.functional.softplus(
        raw_g.view(1, -1, H, K)
    )
    out, last_state = fused_recurrent_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_seqlens,
    )
    return out, last_state


def _run_verify(
    A_log,
    dt_bias,
    q,
    k,
    v,
    a,
    b,
    ssm_states,
    cache_indices,
    cu_seqlens,
    intermediate_states_buffer,
    intermediate_state_indices,
    cache_steps,
    retrieve_parent_token=None,
):
    """The exact call TritonKDAKernel.target_verify makes."""
    return fused_sigmoid_gating_delta_rule_update(
        A_log=A_log,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        a=a,
        b=b,
        initial_state_source=ssm_states,
        initial_state_indices=cache_indices,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=True,
        softplus_beta=1.0,
        softplus_threshold=20.0,
        is_kda=True,
        disable_state_update=True,
        intermediate_states_buffer=intermediate_states_buffer,
        intermediate_state_indices=intermediate_state_indices,
        cache_steps=cache_steps,
        retrieve_parent_token=retrieve_parent_token,
    )


def _make_intermediate(N, T, H, K, V, device, padded=False, seed=7):
    """Intermediate buffer [slots, T, H, K, V] + shuffled slot indices.

    padded=True allocates through a wider last dim and slices — an
    envelope-strided (non-contiguous) view like the unified pool's band slice,
    guarding the stride/i64 addressing class.
    """
    torch.manual_seed(seed)
    slots = N + 3
    if padded:
        backing = torch.zeros(
            slots, T, H, K, V + 64, dtype=torch.float32, device=device
        )
        buf = backing[..., :V]
    else:
        buf = torch.zeros(slots, T, H, K, V, dtype=torch.float32, device=device)
    indices = torch.randperm(slots, device=device)[:N].to(torch.int32)
    return buf, indices


@pytest.mark.skipif(not KERNELS_AVAILABLE, reason="Kernels not available")
@pytest.mark.parametrize("N", [1, 4, 16])
@pytest.mark.parametrize("T", [1, 4, 12])
@pytest.mark.parametrize("H", [16, 32])  # tp2 / tp1 shard shapes
def test_kda_verify_output_precision_and_state_freeze(N, T, H):
    """Chain verify: output matches the KDA reference, and the persistent
    ssm_states are byte-identical afterwards (disable_state_update)."""
    K = V = 128
    A_log, dt_bias, a, b, q, k, v, ssm, idx, cu = _make_tensors(N, T, H, K, V)
    inter, inter_idx = _make_intermediate(N, T, H, K, V, ssm.device)

    ssm_before = ssm.clone()
    out_fused = _run_verify(
        A_log, dt_bias, q, k, v, a, b, ssm, idx, cu, inter, inter_idx, T
    )
    # Persistent state NEVER advances during verify.
    torch.testing.assert_close(ssm, ssm_before, rtol=0, atol=0)

    initial = ssm_before[idx.long()].clone()
    out_ref, _ = _reference_chain(A_log, dt_bias, q, k, v, a, b, initial, cu)
    torch.testing.assert_close(
        out_fused.view_as(out_ref), out_ref, rtol=1e-2, atol=1e-2
    )


@pytest.mark.skipif(not KERNELS_AVAILABLE, reason="Kernels not available")
@pytest.mark.parametrize("padded", [False, True])
def test_kda_verify_intermediate_states_per_step(padded):
    """intermediate[slot, t] holds the post-state of draft step t: compare
    against the reference run truncated to the first t+1 tokens per row.
    padded=True runs the same check through an envelope-strided buffer view."""
    N, T, H, K, V = 4, 6, 16, 128, 128
    A_log, dt_bias, a, b, q, k, v, ssm, idx, cu = _make_tensors(N, T, H, K, V)
    inter, inter_idx = _make_intermediate(N, T, H, K, V, ssm.device, padded=padded)

    _run_verify(A_log, dt_bias, q, k, v, a, b, ssm, idx, cu, inter, inter_idx, T)

    initial_full = ssm[idx.long()].clone()
    for t in range(T):
        L = t + 1

        def trunc(x, feat_shape):
            return (
                x.view(1, N, T, *feat_shape)[:, :, :L]
                .reshape(1, N * L, *feat_shape)
                .contiguous()
            )

        cu_t = torch.arange(0, N * L + 1, L, dtype=torch.int32, device=ssm.device)
        _, last_state = _reference_chain(
            A_log,
            dt_bias,
            trunc(q, (H, K)),
            trunc(k, (H, K)),
            trunc(v, (H, V)),
            trunc(a, (H * K,)),
            trunc(b, (H,)),
            initial_full.clone(),
            cu_t,
        )
        got = inter[inter_idx.long(), t]
        torch.testing.assert_close(got, last_state, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not KERNELS_AVAILABLE, reason="Kernels not available")
def test_kda_verify_tree_two_branches():
    """Tree verify (topk > 1 shape): a two-branch tree must reproduce, per
    branch, exactly what the chain kernel computes on that branch's token
    path. Token 0 is the root (parent -1); tokens 1,2 chain off it; tokens
    3,4 form a second branch also rooted at token 0."""
    N, H, K, V = 2, 16, 128, 128
    T = 5
    parents = [-1, 0, 1, 0, 3]
    branch_a = [0, 1, 2]
    branch_b = [0, 3, 4]

    A_log, dt_bias, a, b, q, k, v, ssm, idx, cu = _make_tensors(N, T, H, K, V)
    inter, inter_idx = _make_intermediate(N, T, H, K, V, ssm.device)
    retrieve_parent_token = torch.tensor(
        [parents] * N, dtype=torch.int64, device=ssm.device
    )

    out_fused = _run_verify(
        A_log,
        dt_bias,
        q,
        k,
        v,
        a,
        b,
        ssm,
        idx,
        cu,
        inter,
        inter_idx,
        T,
        retrieve_parent_token=retrieve_parent_token,
    )
    out_fused = out_fused.view(1, N * T, H, V)

    def gather_tokens(x, branch, feat_shape):
        xs = x.view(1, N, T, *feat_shape)[:, :, branch]
        return xs.reshape(1, N * len(branch), *feat_shape).contiguous()

    for branch in (branch_a, branch_b):
        L = len(branch)
        cu_b = torch.arange(0, N * L + 1, L, dtype=torch.int32, device=ssm.device)
        initial = ssm[idx.long()].clone()
        out_ref, _ = _reference_chain(
            A_log,
            dt_bias,
            gather_tokens(q, branch, (H, K)),
            gather_tokens(k, branch, (H, K)),
            gather_tokens(v, branch, (H, V)),
            gather_tokens(a, branch, (H * K,)),
            gather_tokens(b, branch, (H,)),
            initial,
            cu_b,
        )
        got = gather_tokens(out_fused, branch, (H, V))
        torch.testing.assert_close(got, out_ref, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
