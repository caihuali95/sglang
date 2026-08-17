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
"""Fused virtual->physical translate (`translate_v2p`).

The per-step translate used to be a six-op tensor chain (`//`, `%`,
index_select, `mul_`, `add_`, `clamp_`); it is now one kernel. These tests pin
the fused result against the ORIGINAL formulas, so the fusion can never drift:

    plain: max(v2p[t // ps] * ps + t % ps, 0)
    dense: max(v2p[t // ps] * (ps * mult) + t % ps, 0)

with the two no-location id classes both landing on the padding sink 0:
a TOMBSTONED page (v2p == -1) and a PADDING id (t < 0). Also pins the
``out is virt_tokens`` aliasing the old ``index_select(out=)`` path could not
express (it needed a transient gather + copy).

    python -m pytest test/registered/unit/mem_cache/test_translate_v2p_fusion.py -v
"""

import unittest

import torch

from sglang.kernels.ops.memory.virtual_slot import translate_v2p
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

_DEV = "cpu"


def _reference(virt, v2p, *, page_size, page_stride):
    """Pre-fusion formula, evaluated elementwise in plain Python."""
    out = []
    for t in virt.tolist():
        if t < 0:  # padding id -> sink
            out.append(0)
            continue
        phys = int(v2p[t // page_size])
        out.append(max(phys * page_stride + (t % page_size), 0))
    return torch.tensor(out, dtype=torch.int64)


class TestTranslateV2PFusion(unittest.TestCase):
    def _v2p(self, num_pages, tombstone_pages=()):
        # page p -> physical page (num_pages - p), i.e. a non-identity mapping
        # so a dropped gather cannot pass by accident.
        v2p = torch.tensor(
            [num_pages - p for p in range(num_pages)] + [-1],  # trailing sentinel
            dtype=torch.int64,
            device=_DEV,
        )
        for p in tombstone_pages:
            v2p[p] = -1
        return v2p

    def test_matches_reference_across_page_sizes_and_strides(self):
        for page_size in (1, 4, 256):
            for multiplier in (1, 3):
                with self.subTest(page_size=page_size, multiplier=multiplier):
                    num_pages = 32
                    v2p = self._v2p(num_pages, tombstone_pages=(2, 7))
                    stride = page_size * multiplier
                    virt = torch.tensor(
                        [
                            0,
                            1,
                            page_size,
                            page_size + 1,
                            5 * page_size + (page_size - 1),
                            2 * page_size,
                            7 * page_size + 1,
                            31 * page_size,
                        ],
                        dtype=torch.int64,
                        device=_DEV,
                    )
                    got = translate_v2p(
                        virt, v2p, page_size=page_size, page_stride=stride
                    )
                    want = _reference(
                        virt, v2p, page_size=page_size, page_stride=stride
                    )
                    self.assertTrue(
                        torch.equal(got, want),
                        f"got {got.tolist()} want {want.tolist()}",
                    )

    def test_tombstoned_page_and_padding_id_land_on_the_sink(self):
        page_size = 8
        v2p = self._v2p(16, tombstone_pages=(3,))
        virt = torch.tensor(
            [3 * page_size, 3 * page_size + 7, -1, 4 * page_size],
            dtype=torch.int64,
            device=_DEV,
        )
        got = translate_v2p(virt, v2p, page_size=page_size, page_stride=page_size)
        self.assertEqual(got[0].item(), 0)  # tombstoned page
        self.assertEqual(got[1].item(), 0)  # tombstoned page, nonzero offset
        self.assertEqual(got[2].item(), 0)  # padding id
        self.assertGreater(got[3].item(), 0)  # live page unaffected

    def test_out_may_alias_the_input(self):
        for page_size in (1, 16):
            with self.subTest(page_size=page_size):
                v2p = self._v2p(16)
                virt = torch.tensor(
                    [0, page_size, 3 * page_size + (page_size - 1)],
                    dtype=torch.int64,
                    device=_DEV,
                )
                want = _reference(virt, v2p, page_size=page_size, page_stride=page_size)
                buf = virt.clone()
                ret = translate_v2p(
                    buf, v2p, page_size=page_size, page_stride=page_size, out=buf
                )
                self.assertIs(ret, buf)
                self.assertTrue(torch.equal(buf, want))

    def test_out_is_written_in_place_and_returned(self):
        page_size = 4
        v2p = self._v2p(16)
        virt = torch.tensor([0, 4, 9], dtype=torch.int64, device=_DEV)
        out = torch.full((3,), -999, dtype=torch.int64, device=_DEV)
        ret = translate_v2p(
            virt, v2p, page_size=page_size, page_stride=page_size, out=out
        )
        self.assertIs(ret, out)
        self.assertTrue(
            torch.equal(
                out, _reference(virt, v2p, page_size=page_size, page_stride=page_size)
            )
        )

    def test_strided_input_is_handled_not_misread(self):
        """A strided view must not be read flatly (the kernel indexes both
        sides flatly, so a raw-pointer read of a view is a wrong-slot bug)."""
        page_size = 4
        v2p = self._v2p(16)
        dense = torch.tensor([0, 99, 4, 99, 9, 99], dtype=torch.int64, device=_DEV)
        strided = dense[::2]  # [0, 4, 9] — non-contiguous
        self.assertFalse(strided.is_contiguous())
        got = translate_v2p(strided, v2p, page_size=page_size, page_stride=page_size)
        want = _reference(strided, v2p, page_size=page_size, page_stride=page_size)
        self.assertTrue(
            torch.equal(got, want), f"got {got.tolist()} want {want.tolist()}"
        )

    def test_strided_out_is_rejected(self):
        page_size = 4
        v2p = self._v2p(16)
        virt = torch.tensor([0, 4, 9], dtype=torch.int64, device=_DEV)
        strided_out = torch.zeros(6, dtype=torch.int64, device=_DEV)[::2]
        with self.assertRaises(AssertionError):
            translate_v2p(
                virt, v2p, page_size=page_size, page_stride=page_size, out=strided_out
            )

    def test_empty_input(self):
        v2p = self._v2p(8)
        virt = torch.empty((0,), dtype=torch.int64, device=_DEV)
        got = translate_v2p(virt, v2p, page_size=4, page_stride=4)
        self.assertEqual(got.numel(), 0)
        self.assertEqual(got.dtype, torch.int64)


class TestInt32SwaOutput(unittest.TestCase):
    """The SWA read path takes int32 ids, so the same kernel must narrow on
    store. Page ids are pool-bounded, so the post-clamp value always fits."""

    def _v2p(self, num_pages):
        return torch.tensor(
            [num_pages - p for p in range(num_pages)] + [-1],
            dtype=torch.int64,
            device=_DEV,
        )

    def test_int32_out_dtype_matches_int64_values(self):
        for page_size in (1, 8):
            with self.subTest(page_size=page_size):
                v2p = self._v2p(16)
                virt = torch.tensor(
                    [0, page_size, 5 * page_size + (page_size - 1)],
                    dtype=torch.int64,
                    device=_DEV,
                )
                wide = translate_v2p(
                    virt, v2p, page_size=page_size, page_stride=page_size
                )
                narrow = translate_v2p(
                    virt,
                    v2p,
                    page_size=page_size,
                    page_stride=page_size,
                    out_dtype=torch.int32,
                )
                self.assertEqual(narrow.dtype, torch.int32)
                self.assertTrue(torch.equal(narrow.to(torch.int64), wide))

    def test_int32_out_buffer_is_written_in_place(self):
        page_size = 4
        v2p = self._v2p(16)
        virt = torch.tensor([0, 4, 9], dtype=torch.int64, device=_DEV)
        out = torch.full((3,), -999, dtype=torch.int32, device=_DEV)
        ret = translate_v2p(
            virt,
            v2p,
            page_size=page_size,
            page_stride=page_size,
            out=out,
        )
        self.assertIs(ret, out)
        self.assertEqual(out.dtype, torch.int32)
        self.assertTrue(
            torch.equal(
                out.to(torch.int64),
                _reference(virt, v2p, page_size=page_size, page_stride=page_size),
            )
        )


class TestPoolLevelSwaTranslate(unittest.TestCase):
    """Regressions on `UnifiedSWAKVPool.translate_loc_from_full_to_swa`, driven
    through the real factory bundle (CPU)."""

    def _bundle(self):
        from test_unified_swa_translate_clamp import _swa_bundle

        return _swa_bundle()

    def test_out_int64_written_in_place_without_narrowing_round_trip(self):
        """The cuda-graph window refill passes its INT64 capture-stable buffer
        as out= — the result must land there in the buffer's dtype, in place
        (the old path narrowed to int32 only to be widened again by the
        assignment)."""
        bundle = self._bundle()
        allocator = bundle.token_to_kv_pool_allocator
        kvcache = bundle.token_to_kv_pool
        v = allocator.alloc(4)
        self.assertIsNotNone(v)
        want = kvcache.translate_loc_from_full_to_swa(v).to(torch.int64)
        buf = v.clone()  # int64, aliasing-style in-place like the refill
        ret = kvcache.translate_loc_from_full_to_swa(buf, out=buf)
        self.assertIs(ret, buf)
        self.assertEqual(buf.dtype, torch.int64)
        self.assertTrue(torch.equal(buf, want))


class TestAllocatorTranslateSurface(unittest.TestCase):
    """The allocator methods must keep their published contracts on top of the
    fused kernel: dense == plain when the multiplier is 1, out= honoured."""

    def _full_band(self):
        from test_multi_ended_allocator import (
            TestPagedMultiEndedAllocator as _PagedFixture,
        )

        inst = _PagedFixture(
            [m for m in dir(_PagedFixture) if m.startswith("test_")][0]
        )
        _pool, full, _swa, _fkv, _skv = inst._build()
        return full

    def test_dense_equals_plain_when_multiplier_is_one(self):
        full = self._full_band()
        self.assertEqual(full.kernel_page_multiplier, 1)
        v = full.alloc(full.page_size * 2)
        self.assertIsNotNone(v)
        self.assertTrue(
            torch.equal(full.translate_kv_loc(v), full.translate_kv_loc_dense(v))
        )

    def test_translate_out_matches_returned(self):
        full = self._full_band()
        v = full.alloc(full.page_size * 2)
        self.assertIsNotNone(v)
        expect = full.translate_kv_loc(v)
        buf = torch.empty_like(expect)
        got = full.translate_kv_loc(v, out=buf)
        self.assertIs(got, buf)
        self.assertTrue(torch.equal(buf, expect))


if __name__ == "__main__":
    unittest.main()
