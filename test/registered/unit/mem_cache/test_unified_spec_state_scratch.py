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
"""`SpecStateScratchSpec` byte layout + the `UnifiedKVPool` scratch region.

Target-verify on mamba-family models snapshots per-draft-token intermediate
state (SSM + conv windows). The verify kernels index those buffers
POSITIONALLY by batch lane (`build_verify_intermediate_state_indices`), so
the scratch is a boot-sized STATIC region inside `_raw` — carved below every
sub-pool's first allocatable slot — not a dynamically placed sub-pool.

Pinned here:
  - the region's byte math (256-aligned chunks, non-overlapping, exact
    total) — the single source the boot deduction, the pool reservation, and
    the view construction all read;
  - view identity: `build_views` yields contiguous tensors of exactly the
    shapes/dtypes of the raw `torch.zeros` allocations they replace
    (`MambaPool.SpeculativeState` consumers see no difference), confined to
    their own chunks;
  - the carve invariants: the region lives strictly below every sub-pool's
    first allocatable slot inside `_raw`, and a pool built WITHOUT a region
    is byte-identical to before the carve existed.

    python -m pytest test/registered/unit/mem_cache/test_unified_spec_state_scratch.py -v
"""

import unittest

import torch

from sglang.srt.mem_cache.unified_memory_pool import (
    _SCRATCH_ALIGN,
    MambaSubPoolSpec,
    MHASubPoolSpec,
    SpecStateScratchSpec,
    UnifiedKVPool,
    _reserved_floor_bytes,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_DEV = "cpu"


def _mamba_spec(layer_num=2):
    # Falcon-like dtype mix: bf16 conv, fp32 temporal — the alignment-hazard
    # combination the mamba views already guard.
    return MambaSubPoolSpec(
        name="mamba",
        layer_num=layer_num,
        grow_direction="down",
        conv_state_shapes=((6, 3), (4, 5)),
        conv_dtype=torch.bfloat16,
        temporal_state_shape=(2, 4, 4),
        temporal_dtype=torch.float32,
    )


def _scratch(spec_state_size=5, draft_tokens=3, layer_num=2):
    return SpecStateScratchSpec.from_mamba_spec(
        _mamba_spec(layer_num),
        spec_state_size=spec_state_size,
        draft_tokens=draft_tokens,
    )


def _pool(scratch_region_bytes=0, total_bytes=1 << 20):
    full_spec = MHASubPoolSpec(
        name="full",
        layer_num=1,
        head_num=1,
        head_dim=8,
        store_dtype=torch.bfloat16,
        grow_direction="up",
    )
    return UnifiedKVPool(
        total_bytes=total_bytes,
        sub_pool_specs=[full_spec, _mamba_spec()],
        device=_DEV,
        enable_memory_saver=False,
        scratch_region_bytes=scratch_region_bytes,
    )


class TestSpecStateScratchSpec(unittest.TestCase):
    def test_byte_math(self):
        """Chunks are 256-aligned, non-overlapping, and sum to total_bytes —
        the derived property every reader of the single source relies on."""
        s = _scratch()
        offsets = s._chunk_offsets()
        sizes = [s._ssm_bytes()] + [
            s._conv_bytes(i) for i in range(len(s.conv_state_shapes))
        ]
        self.assertEqual(len(offsets), len(sizes))
        for off in offsets:
            self.assertEqual(off % _SCRATCH_ALIGN, 0)
        for (off, size), next_off in zip(
            zip(offsets, sizes), offsets[1:] + [s.total_bytes()]
        ):
            self.assertLessEqual(off + size, next_off)
        self.assertEqual(s.total_bytes() % _SCRATCH_ALIGN, 0)

    def test_from_mamba_spec_is_deterministic(self):
        """Two builds from the same inputs are EQUAL — the foundation of the
        rebuild-and-assert drift tripwire in UnifiedMambaPool."""
        self.assertEqual(_scratch(), _scratch())
        self.assertNotEqual(_scratch(), _scratch(draft_tokens=4))

    def test_views_match_the_raw_allocations_they_replace(self):
        """`build_views` must be indistinguishable from the `torch.zeros`
        allocations consumers were built against: same shapes, same dtypes,
        contiguous."""
        s = _scratch(spec_state_size=5, draft_tokens=3, layer_num=2)
        region = torch.zeros(s.total_bytes(), dtype=torch.uint8, device=_DEV)
        ssm, convs = s.build_views(region)
        self.assertEqual(tuple(ssm.shape), (2, 6, 3, 2, 4, 4))
        self.assertEqual(ssm.dtype, torch.float32)
        self.assertTrue(ssm.is_contiguous())
        self.assertEqual(len(convs), 2)
        self.assertEqual(tuple(convs[0].shape), (2, 6, 3, 6, 3))
        self.assertEqual(tuple(convs[1].shape), (2, 6, 3, 4, 5))
        for c in convs:
            self.assertEqual(c.dtype, torch.bfloat16)
            self.assertTrue(c.is_contiguous())

    def test_views_are_confined_to_their_chunks(self):
        """Writing one view touches only its own chunk: filling every view
        with ones leaves exactly the alignment-padding bytes zero, and no two
        views alias."""
        s = _scratch()
        region = torch.zeros(s.total_bytes(), dtype=torch.uint8, device=_DEV)
        ssm, convs = s.build_views(region)
        ssm.fill_(1.0)
        ssm_nonzero = int((region != 0).sum())
        self.assertEqual(ssm_nonzero, s._ssm_bytes() // 2)  # fp32 1.0 = 2 nonzero B
        for c in convs:
            c.fill_(torch.nan)  # bf16 nan: both bytes nonzero
        touched = int((region != 0).sum())
        self.assertEqual(
            touched,
            s._ssm_bytes() // 2
            + sum(s._conv_bytes(i) for i in range(len(s.conv_state_shapes))),
        )
        # Round-trip isolation: zero the ssm view; conv bytes stay intact.
        ssm.zero_()
        self.assertEqual(
            int((region != 0).sum()),
            sum(s._conv_bytes(i) for i in range(len(s.conv_state_shapes))),
        )

    def test_build_views_rejects_wrong_region_size(self):
        s = _scratch()
        with self.assertRaises(AssertionError):
            s.build_views(torch.zeros(s.total_bytes() - 1, dtype=torch.uint8))


class TestUnifiedKVPoolScratchRegion(unittest.TestCase):
    def test_region_sits_below_every_first_allocatable_slot(self):
        """The carve raises every sub-pool's min_slot_index past the region:
        no allocator-reachable byte overlaps it."""
        n = _scratch().total_bytes()
        pool = _pool(scratch_region_bytes=n)
        region = pool.scratch_region()
        self.assertEqual(region.numel(), n)
        self.assertEqual(region.dtype, torch.uint8)
        # Same storage as _raw (a view, not a copy).
        self.assertEqual(
            region.untyped_storage().data_ptr(), pool._raw.untyped_storage().data_ptr()
        )
        start = pool._scratch_region_start
        self.assertEqual(start % _SCRATCH_ALIGN, 0)
        sink = _reserved_floor_bytes(pool.sub_pool_specs, 1)
        self.assertGreaterEqual(start, sink)
        for spec in pool.sub_pool_specs:
            first_byte = pool.min_slot_index(spec.name) * spec.entry_bytes()
            self.assertGreaterEqual(
                first_byte,
                start + n,
                f"sub-pool {spec.name!r} first allocatable byte {first_byte} "
                f"overlaps the scratch region ending at {start + n}",
            )

    def test_no_region_is_byte_identical_to_before(self):
        """scratch_region_bytes=0 keeps every min_slot_index exactly as the
        pre-carve pool computed it, and the accessor refuses."""
        pool = _pool(scratch_region_bytes=0)
        sink = _reserved_floor_bytes(pool.sub_pool_specs, 1)
        for spec in pool.sub_pool_specs:
            self.assertEqual(
                pool.min_slot_index(spec.name),
                -(-sink // spec.entry_bytes()),
            )
        with self.assertRaises(AssertionError):
            pool.scratch_region()


if __name__ == "__main__":
    unittest.main()
