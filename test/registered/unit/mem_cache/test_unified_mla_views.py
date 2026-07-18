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
"""Round-trip correctness of the unified-pool MLA machinery: the 4-D
page-major latent views (``build_page_major_mla_views``), the
``MLASubPoolSpec`` byte accounting, and ``UnifiedMLATokenToKVPool``'s
page-aware write/read/move overrides.

An MLA sub-pool stores ONE latent region per layer
(``kv_cache_dim = kv_lora_rank + qk_rope_head_dim``), no K/V pair. The views
are page-major: token ``t`` addresses page ``t // ps``, slot ``t % ps`` —
non-affine in ``t`` at page_size > 1. The layout is page-size-general by
design; today's only MLA-hybrid (Kimi Linear) runs at ps == 1, so the ps > 1
sweeps here are what keep the two-level addressing honest ahead of any
scheduler unlock.

NB on aliasing: every sub-pool's views span the WHOLE unified buffer from
anchor 0 (byte-range disjointness is an ALLOCATOR property, not a view
property). Cross-sub-pool tests therefore write to slot sets whose implied
byte ranges are disjoint — exactly the invariant the allocator maintains at
runtime — and assert that disjointness from the spec byte math first.

These tests prove:
  - every (layer, page, slot) element round-trips through the strided views
    against the raw byte buffer at the hand-computed offset, ps in {1, 4, 64};
  - layer views do NOT alias each other (intra-envelope overlap);
  - byte-disjoint slot sets of a co-resident Kimi-shaped mamba sub-pool and
    the MLA sub-pool round-trip independently;
  - ``MLASubPoolSpec.entry_bytes()`` equals the shared ``mla_entry_bytes``
    helper the configurator prices with (drift guard);
  - the pool's page-aware ``move_kv_cache`` relocates whole latent rows
    (two-level torch indexing — CPU-checkable);
  - a ``KVWriteLoc`` with no physical loc RAISES (untranslated-backend
    tripwire), and the PD/HiCache surfaces raise NotImplementedError;
  - [CUDA] the page-aware store/scatter/gather kernels are byte-exact against
    a dense ``MLATokenToKVPool`` oracle at ps in {1, 4, 64}.

    python -m pytest test/registered/unit/mem_cache/test_unified_mla_views.py -v
"""

import unittest

import torch

from sglang.srt.mem_cache.layout.page_major import (
    build_page_major_mla_views,
    mla_entry_bytes,
)
from sglang.srt.mem_cache.memory_pool import KVWriteLoc
from sglang.srt.mem_cache.unified_memory_pool import (
    MambaSubPoolSpec,
    MLASubPoolSpec,
    UnifiedKVPool,
    UnifiedMLATokenToKVPool,
)
from sglang.test.ci.ci_register import register_cuda_ci

_HAS_CUDA = torch.cuda.is_available()
_DEV = "cuda" if _HAS_CUDA else "cpu"

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")

# Kimi-Linear-shaped MLA dims: lora 512 + rope 64 = 576-dim latent, bf16.
LORA, ROPE = 512, 64
KV_DIM = LORA + ROPE
DTYPE = torch.bfloat16
MLA_LAYERS = 3


def _mla_spec(layer_num=MLA_LAYERS):
    return MLASubPoolSpec(
        name="full",
        layer_num=layer_num,
        kv_lora_rank=LORA,
        qk_rope_head_dim=ROPE,
        store_dtype=DTYPE,
        grow_direction="down",
    )


def _kimi_mamba_spec(layer_num=4):
    # Kimi-shaped KDA state: single fused conv, transposed (kernel-1, dim)
    # orientation; square temporal state per head.
    return MambaSubPoolSpec(
        name="mamba",
        layer_num=layer_num,
        grow_direction="up",
        conv_state_shapes=((3, 96),),
        conv_dtype=torch.bfloat16,
        temporal_state_shape=(4, 32, 32),
        temporal_dtype=torch.float32,
    )


def _make_pool(*, want_mla_slots=32, page_size=1, mla_layers=MLA_LAYERS):
    """2-sub-pool UnifiedKVPool: Kimi-shaped mamba (grow-up) + MLA (grow-down),
    sized for >= ``want_mla_slots`` MLA slots and a handful of mamba slots."""
    mla = _mla_spec(mla_layers)
    mamba = _kimi_mamba_spec()
    total = want_mla_slots * mla.entry_bytes() + 8 * mamba.entry_bytes()
    total = ((total + 7) // 8) * 8  # cover bf16/fp32 .view() alignment
    pool = UnifiedKVPool(
        total_bytes=total,
        sub_pool_specs=[mamba, mla],
        device=_DEV,
        enable_memory_saver=False,
        page_size=page_size,
    )
    return pool, mla, mamba


class _FakeLayer:
    def __init__(self, layer_id):
        self.layer_id = layer_id


class TestMLAViews(unittest.TestCase):
    def test_roundtrip_all_pages_slots_layers(self):
        """A sentinel written through the view must land at the hand-computed
        page-major byte offset in the raw buffer (and vice versa)."""
        for ps in (1, 4, 64):
            num_pages = 4
            entry = MLA_LAYERS * KV_DIM * DTYPE.itemsize
            raw = torch.zeros(
                num_pages * ps * entry, dtype=torch.uint8, device=_DEV
            )
            views = build_page_major_mla_views(
                raw,
                layer_num=MLA_LAYERS,
                kv_lora_rank=LORA,
                qk_rope_head_dim=ROPE,
                store_dtype=DTYPE,
                page_size=ps,
                num_pages=num_pages,
            )
            self.assertEqual(len(views), MLA_LAYERS)
            for L, view in enumerate(views):
                self.assertEqual(tuple(view.shape), (num_pages, ps, 1, KV_DIM))
                for pg in (0, num_pages - 1):
                    for sl in (0, ps - 1):
                        sentinel = torch.arange(
                            KV_DIM, dtype=DTYPE, device=_DEV
                        ) + (L * 1000 + pg * 100 + sl)
                        view[pg, sl, 0] = sentinel
                        # Page-major: page block, then layer L's ps-row block,
                        # then the in-page slot row.
                        off = (
                            pg * ps * entry
                            + L * ps * KV_DIM * DTYPE.itemsize
                            + sl * KV_DIM * DTYPE.itemsize
                        )
                        got = raw[off : off + KV_DIM * DTYPE.itemsize].view(DTYPE)
                        self.assertTrue(
                            torch.equal(got, sentinel),
                            f"ps={ps} L={L} pg={pg} sl={sl}: view write did not "
                            f"land at the expected byte offset {off}",
                        )
                        view[pg, sl, 0] = 0  # keep later probes clean

    def test_no_cross_layer_aliasing(self):
        """Zero the buffer; fill ONE layer's view; every other layer's view
        must read all-zero (per-page layer regions must not overlap)."""
        pool, _, _ = _make_pool(want_mla_slots=32, page_size=4)
        views = pool.mla_views_for("full")
        for target in range(len(views)):
            pool._raw.zero_()
            views[target].fill_(7.0)
            for other in range(len(views)):
                if other == target:
                    self.assertTrue(
                        bool((views[other] == 7.0).all().item()),
                        f"write to MLA layer {target} did not fully land",
                    )
                else:
                    self.assertTrue(
                        bool((views[other] == 0).all().item()),
                        f"writing MLA layer {target} CORRUPTED layer {other} "
                        f"(page envelope layer regions overlap)",
                    )

    def test_coresident_mamba_byte_disjoint_roundtrip(self):
        """Write mamba slots at the LOW byte end (grow-up) and MLA pages at the
        HIGH byte end (grow-down) — byte ranges proven disjoint from the spec
        math, mirroring the allocator's runtime invariant — and check both
        round-trip untouched."""
        ps = 4
        pool, mla, mamba = _make_pool(want_mla_slots=64, page_size=ps)
        views = pool.mla_views_for("full")
        conv_views, temporal_view = pool.mamba_views_for("mamba")
        num_pages = views[0].shape[0]
        mamba_slots = [1, 2, 3]
        mla_page = num_pages - 1

        # Precondition: the two byte ranges are disjoint.
        mamba_hi = (max(mamba_slots) + 1) * mamba.entry_bytes()
        mla_lo = mla_page * ps * mla.entry_bytes()
        self.assertLess(
            mamba_hi,
            mla_lo,
            "test setup broken: chosen mamba slots and MLA page overlap in bytes",
        )

        pool._raw.zero_()
        for s in mamba_slots:
            conv_views[0][:, s].fill_(float(s))
            temporal_view[:, s].fill_(float(10 + s))
        for L, v in enumerate(views):
            v[mla_page] = (
                torch.arange(v[mla_page].numel(), dtype=DTYPE, device=_DEV).view(
                    v[mla_page].shape
                )
                + L
            )

        for s in mamba_slots:
            self.assertTrue(bool((conv_views[0][:, s] == float(s)).all().item()))
            self.assertTrue(
                bool((temporal_view[:, s] == float(10 + s)).all().item())
            )
        for L, v in enumerate(views):
            expect = (
                torch.arange(v[mla_page].numel(), dtype=DTYPE, device=_DEV).view(
                    v[mla_page].shape
                )
                + L
            )
            self.assertTrue(
                torch.equal(v[mla_page], expect),
                f"co-resident mamba writes corrupted MLA layer {L}",
            )

    def test_entry_bytes_single_sourced(self):
        spec = _mla_spec(layer_num=7)  # Kimi Linear's MLA layer count
        self.assertEqual(
            spec.entry_bytes(),
            mla_entry_bytes(
                layer_num=7,
                kv_lora_rank=LORA,
                qk_rope_head_dim=ROPE,
                itemsize=DTYPE.itemsize,
            ),
        )
        self.assertEqual(spec.entry_bytes(), 7 * KV_DIM * DTYPE.itemsize)
        self.assertEqual(spec.kv_cache_dim(), KV_DIM)

    def test_pool_move_kv_cache_page_aware(self):
        """``move_kv_cache`` must relocate whole latent rows with two-level
        indexing — pure torch, CPU-checkable, cross-page src/tgt."""
        for ps in (1, 4):
            pool, _, _ = _make_pool(want_mla_slots=64, page_size=ps)
            mla_pool = UnifiedMLATokenToKVPool(
                unified_buffer=pool, sub_pool_name="full", page_size=ps
            )
            src, tgt = 5 * ps + (ps - 1), 2 * ps  # cross-page move
            for L, kv in enumerate(mla_pool.kv_buffer):
                kv[src // ps, src % ps, 0] = torch.arange(
                    KV_DIM, dtype=DTYPE, device=_DEV
                ) + (L + 1) * 10
            mla_pool.move_kv_cache(
                torch.tensor([tgt], device=_DEV), torch.tensor([src], device=_DEV)
            )
            for L, kv in enumerate(mla_pool.kv_buffer):
                expect = torch.arange(KV_DIM, dtype=DTYPE, device=_DEV) + (
                    L + 1
                ) * 10
                self.assertTrue(
                    torch.equal(kv[tgt // ps, tgt % ps, 0], expect),
                    f"ps={ps} layer {L}: page-aware move mismatch",
                )

    def test_kvwriteloc_without_full_loc_raises(self):
        pool, _, _ = _make_pool(want_mla_slots=32, page_size=1)
        mla_pool = UnifiedMLATokenToKVPool(
            unified_buffer=pool, sub_pool_name="full", page_size=1
        )
        loc = torch.tensor([3], device=_DEV)
        wl = KVWriteLoc(loc=loc)  # no full_loc — untranslated backend
        with self.assertRaisesRegex(RuntimeError, "did not translate"):
            mla_pool.set_kv_buffer(
                _FakeLayer(0),
                wl,
                torch.zeros(1, 1, KV_DIM, dtype=DTYPE, device=_DEV),
                None,
            )

    def test_excluded_surfaces(self):
        pool, _, _ = _make_pool()
        mla_pool = UnifiedMLATokenToKVPool(
            unified_buffer=pool, sub_pool_name="full", page_size=1
        )
        # SCALAR contract: HybridLinearKVPool.__init__ divides this by 2^30.
        self.assertEqual(mla_pool.get_kv_size_bytes(), 0)
        with self.assertRaises(NotImplementedError):
            mla_pool.get_contiguous_buf_infos()
        with self.assertRaises(NotImplementedError):
            mla_pool.get_cpu_copy(torch.tensor([1]))


@unittest.skipUnless(_HAS_CUDA, "page-aware Triton kernels need CUDA")
class TestMLAKernelsVsDenseOracle(unittest.TestCase):
    """Byte-exactness of the page-aware store/scatter/gather against a dense
    ``MLATokenToKVPool`` oracle across page sizes."""

    def _oracle(self, n_slots):
        from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool

        return MLATokenToKVPool(
            size=n_slots - 1,
            page_size=1,
            dtype=DTYPE,
            kv_lora_rank=LORA,
            qk_rope_head_dim=ROPE,
            layer_num=MLA_LAYERS,
            device=_DEV,
            enable_memory_saver=False,
        )

    def test_store_scatter_gather_vs_oracle(self):
        torch.manual_seed(7)
        layer = _FakeLayer(1)
        for ps in (1, 4, 64):
            n_slots = 8 * ps
            pool, _, _ = _make_pool(want_mla_slots=n_slots, page_size=ps)
            mla_pool = UnifiedMLATokenToKVPool(
                unified_buffer=pool, sub_pool_name="full", page_size=ps
            )
            oracle = self._oracle(n_slots)
            n = 7
            loc = torch.randperm(n_slots - 1, device=_DEV)[:n] + 1

            # Backend-path store (KVWriteLoc carrying the physical loc).
            latent = torch.randn(n, 1, KV_DIM, dtype=DTYPE, device=_DEV)
            mla_pool.set_kv_buffer(
                layer, KVWriteLoc(loc=loc, full_loc=loc), latent, None
            )
            oracle.set_kv_buffer(layer, loc, latent, None)
            for t in loc.tolist():
                self.assertTrue(
                    torch.equal(
                        mla_pool.kv_buffer[1][t // ps, t % ps, 0],
                        oracle.kv_buffer[1][t, 0],
                    ),
                    f"ps={ps} loc={t}: page-aware store != dense oracle",
                )

            # Model-side scatter (nope|rope halves) ...
            nope = torch.randn(n, 1, LORA, dtype=DTYPE, device=_DEV)
            rope = torch.randn(n, 1, ROPE, dtype=DTYPE, device=_DEV)
            mla_pool.set_mla_kv_buffer(layer, loc, nope, rope)
            oracle.set_mla_kv_buffer(layer, loc, nope, rope)
            for t in loc.tolist():
                self.assertTrue(
                    torch.equal(
                        mla_pool.kv_buffer[1][t // ps, t % ps, 0],
                        oracle.kv_buffer[1][t, 0],
                    ),
                    f"ps={ps} loc={t}: page-aware scatter != dense oracle",
                )

            # ... and the gather twin round-trips it.
            got_nope, got_rope = mla_pool.get_mla_kv_buffer(layer, loc)
            self.assertTrue(torch.equal(got_nope, nope))
            self.assertTrue(torch.equal(got_rope, rope))


if __name__ == "__main__":
    unittest.main()
