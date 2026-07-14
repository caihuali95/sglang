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
"""``UnifiedKVPool`` N-sub-pool sweep — geometry + golden-layout regression.

Guards the 2 -> N generalization of the constructor:

  - GOLDEN LAYOUT: spec-OFF 2-pool configs ([full(down), mamba(up)] and
    [full(down), swa(up)]) produce byte-identical geometry (total_bytes,
    max_slots, min_slot_index, anchors) to the pre-N-pool formulae.
  - 3-pool chain: a ``SpecStateSubPoolSpec`` float middle is accepted, its
    views are built, the canonical chain order is [up, floats..., down]
    regardless of input order, and the slot-0 sink (`entry_max` over ALL
    pools) grows when the spec entry dominates.
  - Validation: two grow-up pools / unknown directions / missing end pools
    are rejected.

Runs on CPU — pure construction + strided-view checks, no kernels.

    python -m pytest test/registered/unit/mem_cache/test_unified_npool_sweep.py -v
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

import unittest

import torch

from sglang.srt.mem_cache.unified_memory_pool import (
    MambaSubPoolSpec,
    MHASubPoolSpec,
    SpecStateSubPoolSpec,
    UnifiedKVPool,
)
from cpu_pool_case import CpuPoolTestCase

_DEV = "cpu"


def _mha_spec(name, direction, layer_num=2, head_num=2, head_dim=4):
    return MHASubPoolSpec(
        name=name,
        layer_num=layer_num,
        grow_direction=direction,
        head_num=head_num,
        head_dim=head_dim,
        store_dtype=torch.float16,
    )


def _mamba_spec(name, direction, layer_num=2):
    return MambaSubPoolSpec(
        name=name,
        layer_num=layer_num,
        grow_direction=direction,
        conv_state_shapes=((2, 3),),
        conv_dtype=torch.bfloat16,
        temporal_state_shape=(2, 2),
        temporal_dtype=torch.float32,
    )


def _spec_state_spec(name="spec_state", layer_num=2, num_draft_tokens=3):
    return SpecStateSubPoolSpec(
        name=name,
        layer_num=layer_num,
        num_draft_tokens=num_draft_tokens,
        conv_window_shapes=((2, 3),),
        conv_dtype=torch.bfloat16,
        ssm_state_shape=(2, 2),
        ssm_dtype=torch.float32,
    )


class TestGoldenTwoPoolLayout(CpuPoolTestCase):
    """Spec-OFF 2-pool geometry must match the pre-N-pool formulae exactly."""

    def _check_golden(self, specs, total_bytes):
        pool = UnifiedKVPool(
            total_bytes=total_bytes,
            sub_pool_specs=specs,
            device=_DEV,
            enable_memory_saver=False,
        )
        entry_max = max(s.entry_bytes() for s in specs)
        for s in specs:
            eb = s.entry_bytes()
            self.assertEqual(pool.max_slots(s.name), total_bytes // eb)
            self.assertEqual(
                pool.min_slot_index(s.name), (entry_max + eb - 1) // eb
            )
            self.assertEqual(pool.anchor_bytes(s.name), 0)
        self.assertEqual(pool.total_bytes, total_bytes)
        return pool

    def test_full_mamba(self):
        full = _mha_spec("full", "down")
        mamba = _mamba_spec("mamba", "up")
        pool = self._check_golden([full, mamba], total_bytes=1 << 16)
        # Canonical chain: up end first, down end last.
        self.assertEqual(
            [s.name for s in pool.sub_pool_specs], ["mamba", "full"]
        )

    def test_full_swa(self):
        full = _mha_spec("full", "down")
        swa = _mha_spec("swa", "up", head_dim=2)
        pool = self._check_golden([full, swa], total_bytes=1 << 16)
        self.assertEqual([s.name for s in pool.sub_pool_specs], ["swa", "full"])


class TestThreePoolChain(CpuPoolTestCase):
    def test_chain_order_and_views(self):
        full = _mha_spec("full", "down")
        mamba = _mamba_spec("mamba", "up")
        spec = _spec_state_spec()
        # Input order deliberately scrambled; chain must canonicalize.
        pool = UnifiedKVPool(
            total_bytes=1 << 16,
            sub_pool_specs=[full, spec, mamba],
            device=_DEV,
            enable_memory_saver=False,
        )
        self.assertEqual(
            [s.name for s in pool.sub_pool_specs], ["mamba", "spec_state", "full"]
        )
        ssm, conv_views = pool.spec_state_views_for("spec_state")
        max_slots = pool.max_slots("spec_state")
        self.assertEqual(
            tuple(ssm.shape), (2, max_slots, 3, 2, 2)
        )  # (L, slots, D, *ssm_shape)
        self.assertEqual(tuple(conv_views[0].shape), (2, max_slots, 3, 2, 3))
        self.assertIs(pool.spec_state_spec("spec_state"), spec)

    def test_entry_max_sink_covers_spec_entry(self):
        # The spec entry (D x mamba-ish entry) dominates -> every pool's
        # min_slot_index must be computed against it.
        full = _mha_spec("full", "down")
        mamba = _mamba_spec("mamba", "up")
        spec = _spec_state_spec(num_draft_tokens=8)
        pool = UnifiedKVPool(
            total_bytes=1 << 16,
            sub_pool_specs=[full, mamba, spec],
            device=_DEV,
            enable_memory_saver=False,
        )
        entry_max = max(s.entry_bytes() for s in (full, mamba, spec))
        self.assertEqual(entry_max, spec.entry_bytes())
        for s in (full, mamba, spec):
            eb = s.entry_bytes()
            self.assertEqual(
                pool.min_slot_index(s.name), (entry_max + eb - 1) // eb
            )

    def test_spec_views_isolated_from_neighbors_at_same_bytes(self):
        # All views alias the SAME raw buffer by design; this checks the spec
        # views' addressing is self-consistent (slot k writes stay in slot k's
        # entry) exactly like the layout-builder unit test, but through the
        # pool wrapper.
        pool = UnifiedKVPool(
            total_bytes=1 << 14,
            sub_pool_specs=[
                _mha_spec("full", "down"),
                _mamba_spec("mamba", "up"),
                _spec_state_spec(),
            ],
            device=_DEV,
            enable_memory_saver=False,
        )
        ssm, conv_views = pool.spec_state_views_for("spec_state")
        entry = pool.spec_state_spec("spec_state").entry_bytes()
        k = 1
        ssm[:, k] = 1.0
        for conv in conv_views:
            conv[:, k] = 1.0
        nz = pool._raw != 0
        self.assertTrue(torch.any(nz[k * entry : (k + 1) * entry]))
        self.assertFalse(torch.any(nz[: k * entry]))
        self.assertFalse(torch.any(nz[(k + 1) * entry :]))


class TestUnifiedMambaPoolSpecViews(CpuPoolTestCase):
    """Step-6 gate: with spec decoding on, UnifiedMambaPool's intermediate_*
    are strided views into the SAME `_raw` byte buffer (the spec_state band
    sub-pool), not private torch.zeros."""

    def _make(self, D=3, spec_state_size=4):
        from sglang.srt.mem_cache.unified_memory_pool import UnifiedMambaPool

        full = _mha_spec("full", "down")
        mamba = _mamba_spec("mamba", "up")
        spec = SpecStateSubPoolSpec(
            name="spec_state",
            layer_num=mamba.layer_num,
            num_draft_tokens=D,
            conv_window_shapes=mamba.conv_state_shapes,
            conv_dtype=mamba.conv_dtype,
            ssm_state_shape=mamba.temporal_state_shape,
            ssm_dtype=mamba.temporal_dtype,
        )
        base_bytes = 1 << 14
        total = base_bytes + (spec_state_size + 1) * spec.entry_bytes()
        pool = UnifiedKVPool(
            total_bytes=total,
            sub_pool_specs=[full, mamba, spec],
            device=_DEV,
            enable_memory_saver=False,
        )
        mamba_pool = UnifiedMambaPool(
            unified_buffer=pool,
            sub_pool_name="mamba",
            spec_state_size=spec_state_size,
            mamba_layer_ids=list(range(mamba.layer_num)),
            speculative_num_draft_tokens=D,
            spec_state_sub_pool="spec_state",
        )
        return pool, mamba_pool, spec, D

    def test_intermediates_are_buffer_views(self):
        pool, mamba_pool, spec, D = self._make()
        inter_ssm = mamba_pool.mamba_cache.intermediate_ssm
        inter_conv = mamba_pool.mamba_cache.intermediate_conv_window[0]
        # Same storage as the unified byte buffer — no private allocation.
        self.assertEqual(
            inter_ssm.untyped_storage().data_ptr(),
            pool._raw.untyped_storage().data_ptr(),
        )
        self.assertEqual(
            inter_conv.untyped_storage().data_ptr(),
            pool._raw.untyped_storage().data_ptr(),
        )
        # Consumer-visible ranks: [L, slots, D, *state_shape].
        L = spec.layer_num
        self.assertEqual(inter_ssm.shape[:1], (L,))
        self.assertEqual(inter_ssm.shape[2], D)
        self.assertEqual(tuple(inter_ssm.shape[3:]), tuple(spec.ssm_state_shape))
        self.assertEqual(tuple(inter_conv.shape[3:]), tuple(spec.conv_window_shapes[0]))
        # Outer dim spans the whole buffer (>= spec_state_size + 1).
        self.assertGreaterEqual(inter_ssm.shape[1], 4 + 1)

    def test_spec_missing_sub_pool_is_loud(self):
        from sglang.srt.mem_cache.unified_memory_pool import UnifiedMambaPool

        full = _mha_spec("full", "down")
        mamba = _mamba_spec("mamba", "up")
        pool = UnifiedKVPool(
            total_bytes=1 << 14,
            sub_pool_specs=[full, mamba],
            device=_DEV,
            enable_memory_saver=False,
        )
        with self.assertRaises(AssertionError):
            UnifiedMambaPool(
                unified_buffer=pool,
                sub_pool_name="mamba",
                spec_state_size=4,
                mamba_layer_ids=list(range(mamba.layer_num)),
                speculative_num_draft_tokens=3,
                spec_state_sub_pool=None,
            )

    def test_spec_off_matches_two_pool_footprint(self):
        # With spec off, no third pool and no extra bytes — the exact old path.
        from sglang.srt.mem_cache.unified_memory_pool import UnifiedMambaPool

        full = _mha_spec("full", "down")
        mamba = _mamba_spec("mamba", "up")
        pool = UnifiedKVPool(
            total_bytes=1 << 14,
            sub_pool_specs=[full, mamba],
            device=_DEV,
            enable_memory_saver=False,
        )
        mamba_pool = UnifiedMambaPool(
            unified_buffer=pool,
            sub_pool_name="mamba",
            spec_state_size=4,
            mamba_layer_ids=list(range(mamba.layer_num)),
            speculative_num_draft_tokens=None,
            spec_state_sub_pool=None,
        )
        self.assertFalse(hasattr(mamba_pool.mamba_cache, "intermediate_ssm"))


class TestSweepValidation(CpuPoolTestCase):
    def test_rejects_two_up_pools(self):
        with self.assertRaises(AssertionError):
            UnifiedKVPool(
                total_bytes=1 << 14,
                sub_pool_specs=[_mha_spec("a", "up"), _mha_spec("b", "up")],
                device=_DEV,
                enable_memory_saver=False,
            )

    def test_rejects_float_only_middles_without_ends(self):
        with self.assertRaises(AssertionError):
            UnifiedKVPool(
                total_bytes=1 << 14,
                sub_pool_specs=[_spec_state_spec("s1"), _spec_state_spec("s2")],
                device=_DEV,
                enable_memory_saver=False,
            )

    def test_rejects_duplicate_names(self):
        with self.assertRaises(AssertionError):
            UnifiedKVPool(
                total_bytes=1 << 14,
                sub_pool_specs=[_mha_spec("x", "down"), _mamba_spec("x", "up")],
                device=_DEV,
                enable_memory_saver=False,
            )


if __name__ == "__main__":
    unittest.main()
