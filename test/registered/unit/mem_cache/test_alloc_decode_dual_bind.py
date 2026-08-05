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
"""Fused composite decode alloc (`alloc_decode_dual_bind`).

The SWA/tri composite's `alloc_decode` is one kernel: token slots + dual
virtual->physical bind, fed by per-side `_PhysReservation`s planned in Python.
These tests pin it against the LEGACY composite recipe (snapshot clone +
band bind + `alloc_with_virtual`), which still exists piecewise:

  - token slots (`out_indices`) must be IDENTICAL in every scenario;
  - v2p/p2v tables must be identical whenever the fresh-page pairing is
    deterministic (pure-extend / pure-drain sides); the mixed drain+extend
    case is pairing-free by the `bind` order-agnostic contract and is checked
    structurally (same virtual set bound, same physical set consumed, same
    span/watermark/hole/live state, byte accounting clean);
  - `_reserve_phys_pages` + `_materialize_reservation` must reproduce the
    historical `take_physical` tensors byte-for-byte (covered implicitly by
    the legacy twin running on the refactored primitives).

    python -m pytest test/registered/unit/mem_cache/test_alloc_decode_dual_bind.py -v
"""

import unittest

import torch

from sglang.srt.mem_cache import multi_ended_allocator as mea_mod
from sglang.srt.utils.common import get_num_new_pages
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

_DEV = "cpu"
_PS = 4


def _build_tri_paged(n_full=256, n_swa=256, lazy_compaction=False):
    # Function-scope import: the fixture is a TestCase subclass, and a
    # module-scope binding would make pytest collect its tests AGAIN here.
    from test_unified_tri_pool import TestTriPagedFreeGroup

    inst = TestTriPagedFreeGroup(
        [m for m in dir(TestTriPagedFreeGroup) if m.startswith("test_")][0]
    )
    _pool, allocator = inst._build_paged(
        page_size=_PS, n_full=n_full, n_swa=n_swa
    )
    for band in (allocator.full_attn_allocator, allocator.swa_attn_allocator):
        band.lazy_compaction = lazy_compaction and not isinstance(
            band, mea_mod.FloatMultiEndedAllocator
        )
    allocator.lazy_compaction = lazy_compaction
    return allocator


def _legacy_alloc_decode(allocator, seq_lens, seq_lens_cpu, last_loc):
    """The pre-fusion composite recipe from the still-real pieces.

    The band's Triton `alloc_decode_kernel` has no CPU shim, so ONLY the
    token-slot math is reproduced in torch (from the upstream kernel's
    semantics); every state effect — full-side `_alloc_bind_fast_or_slow`,
    free-list slice, swa `alloc_with_virtual` — is the production code.
    """
    ps = allocator.page_size
    num_new_pages = get_num_new_pages(
        seq_lens=seq_lens_cpu, page_size=ps, decode=True
    )
    need_tokens = num_new_pages * ps
    if need_tokens > allocator.available_size():
        if not mea_mod._relieve_for_alloc(allocator, need_tokens):
            return None
    fa = allocator.full_attn_allocator
    new_virtual_pages = fa.free_virtual_ids[:num_new_pages].clone()

    # upstream alloc_decode_kernel, in torch
    pre = seq_lens - 1
    need = (seq_lens + ps - 1) // ps - (pre + ps - 1) // ps
    ordinal = torch.cumsum(need, 0) - need
    out_indices = last_loc + 1
    wraps = need > 0
    if int(wraps.sum()) > 0:
        out_indices = torch.where(
            wraps, fa.free_virtual_ids[ordinal.clamp(min=0)] * ps, out_indices
        )

    if num_new_pages > 0:
        phys = fa._alloc_bind_fast_or_slow(new_virtual_pages, num_new_pages)
        assert phys is not None
    fa.free_virtual_ids = fa.free_virtual_ids[num_new_pages:]
    if new_virtual_pages.numel() > 0:
        allocator.swa_attn_allocator.alloc_with_virtual(new_virtual_pages)
    return out_indices


class _DecodeSim:
    """Drive per-request decode steps against one allocator."""

    def __init__(self, allocator, seq_lens):
        self.allocator = allocator
        self.seq_lens = list(seq_lens)
        # Seed: allocate page-aligned prefixes and set per-req last_loc.
        self.last_loc = []
        for n in self.seq_lens:
            pages = -(-n // _PS)
            v = allocator.alloc(pages * _PS)
            assert v is not None
            self.last_loc.append(int(v[n - 1]))

    def step(self, fused: bool):
        self.seq_lens = [n + 1 for n in self.seq_lens]
        seq_lens = torch.tensor(self.seq_lens, dtype=torch.int64, device=_DEV)
        last_loc = torch.tensor(self.last_loc, dtype=torch.int64, device=_DEV)
        if fused:
            out = self.allocator.alloc_decode(seq_lens, seq_lens, last_loc)
        else:
            out = _legacy_alloc_decode(self.allocator, seq_lens, seq_lens, last_loc)
        assert out is not None
        self.last_loc = [int(x) for x in out]
        return out


def _assert_tables_equal(test, a, b):
    for name in ("full_attn_allocator", "swa_attn_allocator"):
        ba, bb = getattr(a, name), getattr(b, name)
        test.assertTrue(
            torch.equal(ba.virtual_to_physical, bb.virtual_to_physical),
            msg=f"{name} v2p diverged",
        )
        test.assertTrue(
            torch.equal(ba.physical_to_virtual, bb.physical_to_virtual),
            msg=f"{name} p2v diverged",
        )


def _assert_structurally_equal(test, a, b):
    """Pairing-free equality: same bound virtual/physical sets + same state."""
    for name in ("full_attn_allocator", "swa_attn_allocator"):
        ba, bb = getattr(a, name), getattr(b, name)
        va = (ba.virtual_to_physical >= 0).nonzero().flatten()
        vb = (bb.virtual_to_physical >= 0).nonzero().flatten()
        test.assertTrue(torch.equal(va, vb), msg=f"{name} bound virtual sets differ")
        pa = set(ba.virtual_to_physical[va].tolist())
        pb = set(bb.virtual_to_physical[vb].tolist())
        test.assertEqual(pa, pb, msg=f"{name} consumed physical sets differ")
        # p2v is the exact inverse of v2p
        test.assertTrue(
            torch.equal(ba.physical_to_virtual[ba.virtual_to_physical[va]], va),
            msg=f"{name} p2v not inverse of v2p",
        )
    test.assertEqual(a.available_size(), b.available_size())
    test.assertEqual(a.verify_byte_accounting(), [])
    test.assertEqual(b.verify_byte_accounting(), [])


class TestFusedAllocDecodeEquivalence(unittest.TestCase):
    def _run_twin(self, steps, seq_lens, prepare=None, lazy=False, exact=True):
        fused = _build_tri_paged(lazy_compaction=lazy)
        legacy = _build_tri_paged(lazy_compaction=lazy)
        sim_f = _DecodeSim(fused, seq_lens)
        sim_l = _DecodeSim(legacy, seq_lens)
        if prepare is not None:
            prepare(fused, sim_f)
            prepare(legacy, sim_l)
        for _ in range(steps):
            out_f = sim_f.step(fused=True)
            out_l = sim_l.step(fused=False)
            self.assertTrue(
                torch.equal(out_f, out_l),
                msg=f"out_indices diverged: {out_f.tolist()} vs {out_l.tolist()}",
            )
            if exact:
                _assert_tables_equal(self, fused, legacy)
            _assert_structurally_equal(self, fused, legacy)
        return fused, legacy

    def test_pure_extend_wraps_match_exactly(self):
        # Staggered lengths: reqs wrap on different steps (incl. multi-wrap steps).
        self._run_twin(steps=2 * _PS, seq_lens=[1, 2, 3, 4, 4])

    def test_no_wrap_step_allocates_nothing(self):
        fused = _build_tri_paged()
        sim = _DecodeSim(fused, [1, 1])
        fa = fused.full_attn_allocator
        epoch_before = fa._chain_capacity_epoch()
        avail_before = fused.available_size()
        out = sim.step(fused=True)  # lens 2,2: no page boundary at ps=4
        self.assertEqual(out.tolist(), [x for x in sim.last_loc])
        self.assertEqual(fused.available_size(), avail_before)
        self.assertEqual(fa._chain_capacity_epoch(), epoch_before)

    def test_swa_hole_recycling_matches_exactly(self):
        # Free a middle swa slice -> interior float holes (the float defers
        # boundary absorption to flush, so every freed page is a hole here);
        # the next wraps must recycle them in place (swa pure-drain) while
        # full pure-extends.
        def prepare(allocator, sim):
            sa = allocator.swa_attn_allocator
            v = torch.arange(2 * _PS, 3 * _PS, dtype=torch.int64, device=_DEV)
            allocator.free_swa(v)
            assert sa._hole_pages() > 0

        self._run_twin(steps=_PS + 1, seq_lens=[4, 4, 4], prepare=prepare)

    def test_lazy_full_holes_drain_matches_exactly(self):
        # Lazy mode: free one request on both sides -> full-side lazy holes.
        # STAGGERED lengths so each step wraps at most one page: the hole-drain
        # step is a PURE drain (n <= holes), later wraps pure-extend — both
        # pairing-deterministic, so tables must match exactly. (All-aligned
        # lengths would go mixed drain+extend on the full side, the
        # pairing-free case covered by the structural test below.)
        def prepare(allocator, sim):
            fa = allocator.full_attn_allocator
            v = torch.arange(2 * _PS, 3 * _PS, dtype=torch.int64, device=_DEV)
            allocator.free(v)
            assert len(fa._free_phys_pages) > 0

        self._run_twin(
            steps=2 * _PS, seq_lens=[4, 5, 6], prepare=prepare, lazy=True
        )

    def test_mixed_drain_extend_structurally_equal(self):
        # One full-side hole but TWO wraps in one step -> mixed drain+extend.
        # Pairing may legitimately differ (bind is order-agnostic): structural
        # equality only.
        def prepare(allocator, sim):
            v = torch.arange(0, _PS, dtype=torch.int64, device=_DEV)
            allocator.free(v)

        self._run_twin(
            steps=_PS,
            seq_lens=[4, 4],  # both wrap on the first step together
            prepare=prepare,
            lazy=True,
            exact=False,
        )

    def test_shortfall_returns_none_with_state_untouched(self):
        fused = _build_tri_paged(n_full=8, n_swa=8)
        v = fused.alloc(fused.available_size())
        self.assertIsNotNone(v)
        n = int(v.numel())
        seq_lens = torch.tensor([n + 1], dtype=torch.int64, device=_DEV)
        last_loc = torch.tensor([int(v[-1])], dtype=torch.int64, device=_DEV)
        fa = fused.full_attn_allocator
        epoch_before = fa._chain_capacity_epoch()
        out = fused.alloc_decode(seq_lens, seq_lens, last_loc)
        self.assertIsNone(out)
        self.assertEqual(fa._chain_capacity_epoch(), epoch_before)
        self.assertEqual(fused.verify_byte_accounting(), [])


class TestDualBindLauncherReference(unittest.TestCase):
    """Direct pinning of the CPU reference's per-side phys formula."""

    def test_hole_drain_reads_front_first(self):
        """phys(i) = holes[i] for i < nd, else start + (i - nd) — holes drain
        from the slice FRONT on every band type (matching `take_physical`)."""
        from sglang.kernels.ops.memory.virtual_slot import alloc_decode_dual_bind

        v2p_a = torch.full((17,), -1, dtype=torch.int64)
        p2v_a = torch.full((17,), -1, dtype=torch.int64)
        v2p_b = torch.full((17,), -1, dtype=torch.int64)
        p2v_b = torch.full((17,), -1, dtype=torch.int64)
        holes_a = torch.tensor([11, 12, 13], dtype=torch.int64)
        # two reqs, both wrapping (seq 5 = 4+1 -> new page at ps=4)
        seq_lens = torch.tensor([5, 5], dtype=torch.int64)
        last_loc = torch.tensor([3, 7], dtype=torch.int64)
        free_pages = torch.tensor([6, 7, 8], dtype=torch.int64)
        out = alloc_decode_dual_bind(
            seq_lens=seq_lens,
            last_loc=last_loc,
            free_virtual_pages=free_pages,
            num_new_pages=2,
            page_size=4,
            v2p_a=v2p_a,
            p2v_a=p2v_a,
            holes_a=holes_a,
            nd_a=2,
            start_a=0,
            v2p_b=v2p_b,
            p2v_b=p2v_b,
            holes_b=None,
            nd_b=0,
            start_b=9,
        )
        self.assertEqual(out.tolist(), [6 * 4, 7 * 4])
        self.assertEqual(v2p_a[6].item(), 11)
        self.assertEqual(v2p_a[7].item(), 12)
        self.assertEqual(p2v_a[11].item(), 6)
        self.assertEqual(p2v_a[12].item(), 7)
        self.assertEqual(v2p_b[6].item(), 9)
        self.assertEqual(v2p_b[7].item(), 10)
        self.assertEqual(p2v_b[9].item(), 6)
        self.assertEqual(p2v_b[10].item(), 7)

    def test_more_pages_than_programs_fails_loud(self):
        """One program per request binds one page; a page beyond the grid would
        stay unbound (v2p=-1) and route to the slot-0 sink silently."""
        from sglang.kernels.ops.memory.virtual_slot import alloc_decode_dual_bind

        v2p = torch.full((17,), -1, dtype=torch.int64)
        p2v = torch.full((17,), -1, dtype=torch.int64)
        with self.assertRaises(AssertionError):
            alloc_decode_dual_bind(
                seq_lens=torch.tensor([5], dtype=torch.int64),  # bs == 1
                last_loc=torch.tensor([3], dtype=torch.int64),
                free_virtual_pages=torch.tensor([6, 7], dtype=torch.int64),
                num_new_pages=2,  # > bs
                page_size=4,
                v2p_a=v2p,
                p2v_a=p2v,
                holes_a=None,
                nd_a=0,
                start_a=0,
                v2p_b=v2p,
                p2v_b=p2v,
                holes_b=None,
                nd_b=0,
                start_b=0,
            )


if __name__ == "__main__":
    unittest.main()
