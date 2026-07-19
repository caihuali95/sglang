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
"""Unit tests for the shared-KV-pool v2 core: UnifiedKVPool views and
MultiEndedAllocator (virtual<->physical slot ids + eager compaction).

CPU-only — no GPU / Triton needed (the allocator's data-copy delegates to a
fake kvcache here; the UnifiedKVPool view math is pure torch).

    python -m pytest test/registered/unit/mem_cache/test_multi_ended_allocator.py -v
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

import random
import unittest

import torch

from sglang.srt.mem_cache.multi_ended_allocator import (
    FloatMultiEndedAllocator,
    MultiEndedAllocator,
    SpecStateBandAllocator,
    UnifiedSWATokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.unified_memory_pool import (
    MambaSubPoolSpec,
    MHASubPoolSpec,
    SpecStateSubPoolSpec,
    UnifiedKVPool,
)
from cpu_pool_case import CpuPoolTestCase

_DEV = "cpu"


def _make_mha_spec(name, grow, layer_num=2, head_num=2, head_dim=4):
    return MHASubPoolSpec(
        name=name,
        layer_num=layer_num,
        head_num=head_num,
        head_dim=head_dim,
        store_dtype=torch.float16,
        grow_direction=grow,
    )


def _make_mamba_spec(name, grow, layer_num=2):
    return MambaSubPoolSpec(
        name=name,
        layer_num=layer_num,
        conv_state_shapes=((4, 3),),
        conv_dtype=torch.float32,
        temporal_state_shape=(2, 2, 2),
        temporal_dtype=torch.float32,
        grow_direction=grow,
    )


class _FakeKVCache:
    """Tracks, per *physical* slot, the virtual id whose data lives there.
    `move_kv_cache(dst, src)` copies the marker — so after compaction we can
    check that the data followed the relocation.
    """

    def __init__(self, max_slots: int):
        # buf[p] == virtual id currently stored at physical slot p (-1 if free).
        self.buf = torch.full((max_slots,), -1, dtype=torch.int64)

    def move_kv_cache(self, dst_loc: torch.Tensor, src_loc: torch.Tensor):
        self.buf[dst_loc] = self.buf[src_loc].clone()


class TestUnifiedKVPoolViews(CpuPoolTestCase):
    def test_min_slot_index_and_disjoint_bytes(self):
        full = _make_mha_spec("full", "up", layer_num=4)
        mamba = _make_mamba_spec("mamba", "down", layer_num=2)
        entry_max = max(full.entry_bytes(), mamba.entry_bytes())
        total = full.entry_bytes() * 64 + mamba.entry_bytes() * 16
        pool = UnifiedKVPool(
            total_bytes=total,
            sub_pool_specs=[full, mamba],
            device=_DEV,
            enable_memory_saver=False,
        )
        for s in (full, mamba):
            min_idx = pool.min_slot_index(s.name)
            # real data of every pool begins at bytes >= entry_max
            self.assertGreaterEqual(min_idx * s.entry_bytes(), entry_max)
            self.assertGreater(pool.max_slots(s.name), min_idx)

    def test_mha_view_roundtrip(self):
        full = _make_mha_spec("full", "up", layer_num=3, head_num=2, head_dim=4)
        swa = _make_mha_spec("swa", "down", layer_num=2, head_num=2, head_dim=4)
        total = full.entry_bytes() * 32 + swa.entry_bytes() * 32
        pool = UnifiedKVPool(
            total_bytes=total,
            sub_pool_specs=[full, swa],
            device=_DEV,
            enable_memory_saver=False,
        )
        k_full, v_full = pool.mha_views_for("full")
        k_swa, v_swa = pool.mha_views_for("swa")
        self.assertEqual(len(k_full), 3)
        self.assertEqual(len(k_swa), 2)
        # Write distinct patterns into a couple of slots/layers of "full" and
        # confirm they read back, and that "swa" was not disturbed. "full" and
        # "swa" share the buffer from byte 0; "full" grows up (low slots) and
        # "swa" grows down, so the allocator places "swa" at the high slots.
        # Mirror that here — a low "swa" slot would byte-overlap "full" slot 5
        # (a configuration the byte-frontier coordination never produces).
        swa_slot = pool.max_slots("swa") - 1
        for lyr in range(3):
            k_full[lyr][5] = float(lyr + 1)
            v_full[lyr][5] = float(-(lyr + 1))
        for lyr in range(2):
            k_swa[lyr][swa_slot] = 99.0
        for lyr in range(3):
            self.assertTrue(torch.all(k_full[lyr][5] == float(lyr + 1)))
            self.assertTrue(torch.all(v_full[lyr][5] == float(-(lyr + 1))))
        for lyr in range(2):
            self.assertTrue(torch.all(k_swa[lyr][swa_slot] == 99.0))
        # "full" slot 5 layer-0 K must not alias "full" slot 6 layer-0 K
        self.assertFalse(torch.all(k_full[0][6] == float(1)))

    def test_mamba_view_shapes(self):
        full = _make_mha_spec("full", "up", layer_num=2)
        mamba = _make_mamba_spec("mamba", "down", layer_num=3)
        total = full.entry_bytes() * 16 + mamba.entry_bytes() * 8
        pool = UnifiedKVPool(
            total_bytes=total,
            sub_pool_specs=[full, mamba],
            device=_DEV,
            enable_memory_saver=False,
        )
        conv_views, temporal_view = pool.mamba_views_for("mamba")
        max_slots = pool.max_slots("mamba")
        self.assertEqual(len(conv_views), 1)
        self.assertEqual(tuple(conv_views[0].shape), (3, max_slots, 4, 3))
        self.assertEqual(tuple(temporal_view.shape), (3, max_slots, 2, 2, 2))
        # roundtrip a write at (layer=1, slot=4)
        conv_views[0][1, 4] = 3.5
        temporal_view[2, 6] = -1.25
        self.assertTrue(torch.all(conv_views[0][1, 4] == 3.5))
        self.assertTrue(torch.all(temporal_view[2, 6] == -1.25))


class TestMultiEndedAllocator(CpuPoolTestCase):
    def _build_pair(self, n_full_slots=64, n_mamba_slots=16):
        full = _make_mha_spec("full", "up", layer_num=2)
        mamba = _make_mamba_spec("mamba", "down", layer_num=2)
        total = full.entry_bytes() * n_full_slots + mamba.entry_bytes() * n_mamba_slots
        pool = UnifiedKVPool(
            total_bytes=total,
            sub_pool_specs=[full, mamba],
            device=_DEV,
            enable_memory_saver=False,
        )
        full_kv = _FakeKVCache(pool.max_slots("full"))
        mamba_kv = _FakeKVCache(pool.max_slots("mamba"))
        full_alloc = MultiEndedAllocator(
            kvcache=full_kv,
            unified_buffer=pool,
            sub_pool_name="full",
            device=_DEV,
            is_id_owner=True,
        )
        mamba_alloc = MultiEndedAllocator(
            kvcache=mamba_kv,
            unified_buffer=pool,
            sub_pool_name="mamba",
            device=_DEV,
            is_id_owner=True,
        )
        full_alloc.bind_peer(mamba_alloc)
        mamba_alloc.bind_peer(full_alloc)
        return pool, full_alloc, mamba_alloc, full_kv, mamba_kv

    def _check_invariants(self, alloc: MultiEndedAllocator, kv: _FakeKVCache):
        v2p = alloc.virtual_to_physical
        p2v = alloc.physical_to_virtual
        # live virtual ids = those with v2p != -1, excluding the reserved id 0.
        live_v = [
            v for v in range(1, alloc.num_virtual_ids) if int(v2p[v].item()) != -1
        ]
        # mutual-inverse on the live set
        for v in live_v:
            p = int(v2p[v].item())
            self.assertEqual(int(p2v[p].item()), v, f"p2v[{p}] != {v}")
            # data followed any relocations
            self.assertEqual(int(kv.buf[p].item()), v, f"kv.buf[{p}] != {v}")
        # allocated physical range is hole-free + matches live count
        if alloc.grow_direction == "up":
            alloc_lo, alloc_hi = alloc.min_slot_index, alloc.watermark_physical
        else:
            alloc_lo, alloc_hi = alloc.watermark_physical + 1, alloc.max_slots
        self.assertEqual(alloc_hi - alloc_lo, len(live_v))
        for p in range(alloc_lo, alloc_hi):
            self.assertNotEqual(int(p2v[p].item()), -1, f"hole at physical {p}")
        # free virtual ids ∪ live = [min_slot_index, max_slots)
        free_set = set(int(x) for x in alloc.free_virtual_ids.tolist())
        self.assertEqual(
            free_set | set(live_v),
            set(range(alloc.min_slot_index, alloc.max_slots)),
        )
        self.assertEqual(free_set & set(live_v), set())

    def _alloc(self, alloc: MultiEndedAllocator, kv: _FakeKVCache, n: int):
        avail = alloc.available_size()
        v = alloc.alloc(n)
        if n > avail:
            self.assertIsNone(v)
            return None
        self.assertIsNotNone(v)
        self.assertEqual(int(v.numel()), n)
        # stamp the data marker at each new physical slot
        p = alloc.virtual_to_physical[v]
        kv.buf[p] = v
        return v

    def _free(self, alloc: MultiEndedAllocator, kv: _FakeKVCache, v: torch.Tensor):
        p = alloc.virtual_to_physical[v]
        kv.buf[p] = -1  # the freed virtual id's data is gone
        alloc.free(v)

    def test_basic_alloc_free_compaction(self):
        _, full_alloc, mamba_alloc, full_kv, mamba_kv = self._build_pair()
        # alloc three batches on the full side
        a = self._alloc(full_alloc, full_kv, 3)
        b = self._alloc(full_alloc, full_kv, 5)
        c = self._alloc(full_alloc, full_kv, 2)
        self._check_invariants(full_alloc, full_kv)
        # free the middle batch -> forces eager compaction (boundary slots move in)
        self._free(full_alloc, full_kv, b)
        self._check_invariants(full_alloc, full_kv)
        # `a` and `c` virtual ids unchanged; their physical slots may have moved.
        for v in a.tolist() + c.tolist():
            self.assertNotEqual(int(full_alloc.virtual_to_physical[v].item()), -1)
        # free the boundary batch (no relocation needed)
        self._free(full_alloc, full_kv, c)
        self._check_invariants(full_alloc, full_kv)
        self._free(full_alloc, full_kv, a)
        self._check_invariants(full_alloc, full_kv)
        self.assertEqual(full_alloc.allocated_count(), 0)

    def test_grow_down_side(self):
        _, full_alloc, mamba_alloc, full_kv, mamba_kv = self._build_pair()
        a = self._alloc(mamba_alloc, mamba_kv, 2)
        b = self._alloc(mamba_alloc, mamba_kv, 3)
        c = self._alloc(mamba_alloc, mamba_kv, 1)
        self._check_invariants(mamba_alloc, mamba_kv)
        self._free(mamba_alloc, mamba_kv, b)  # interior -> compaction
        self._check_invariants(mamba_alloc, mamba_kv)
        self._free(mamba_alloc, mamba_kv, a)
        self._free(mamba_alloc, mamba_kv, c)
        self._check_invariants(mamba_alloc, mamba_kv)
        self.assertEqual(mamba_alloc.allocated_count(), 0)

    def test_byte_frontier_coordination(self):
        # full has 8 slots' worth of bytes; mamba's entry is larger, so a few
        # mamba allocs should shrink full's available_size below its slot headroom.
        _, full_alloc, mamba_alloc, full_kv, mamba_kv = self._build_pair(
            n_full_slots=8, n_mamba_slots=8
        )
        full_avail0 = full_alloc.available_size()
        self._alloc(mamba_alloc, mamba_kv, 3)
        self.assertLess(full_alloc.available_size(), full_avail0)
        # over-alloc the full side -> None
        self.assertIsNone(full_alloc.alloc(full_alloc.available_size() + 1))

    def test_randomized(self):
        rng = random.Random(0xC0FFEE)
        _, full_alloc, mamba_alloc, full_kv, mamba_kv = self._build_pair(
            n_full_slots=48, n_mamba_slots=24
        )
        live_full = []  # list of virtual-id tensors still allocated
        live_mamba = []
        for _ in range(400):
            side = rng.random() < 0.6  # 60% full
            alloc, kv, live = (
                (full_alloc, full_kv, live_full)
                if side
                else (mamba_alloc, mamba_kv, live_mamba)
            )
            if rng.random() < 0.55 or not live:
                n = rng.randint(1, 5)
                v = self._alloc(alloc, kv, n)
                if v is not None:
                    live.append(v)
            else:
                idx = rng.randrange(len(live))
                v = live.pop(idx)
                self._free(alloc, kv, v)
            self._check_invariants(full_alloc, full_kv)
            self._check_invariants(mamba_alloc, mamba_kv)
        # drain
        for live, alloc, kv in (
            (live_full, full_alloc, full_kv),
            (live_mamba, mamba_alloc, mamba_kv),
        ):
            for v in live:
                self._free(alloc, kv, v)
            self._check_invariants(alloc, kv)
            self.assertEqual(alloc.allocated_count(), 0)

    def _build_lazy_full(self, n_full_slots=64, n_mamba_slots=16, move_cap=2):
        """A LAZY-compaction 'full' (grow-up) allocator + its peer, with a
        small per-call move cap so flushes are partial (the retract-pressure
        regime where the ghost bug appears)."""
        full = _make_mha_spec("full", "up", layer_num=2)
        mamba = _make_mamba_spec("mamba", "down", layer_num=2)
        total = full.entry_bytes() * n_full_slots + mamba.entry_bytes() * n_mamba_slots
        pool = UnifiedKVPool(
            total_bytes=total,
            sub_pool_specs=[full, mamba],
            device=_DEV,
            enable_memory_saver=False,
        )
        full_kv = _FakeKVCache(pool.max_slots("full"))
        mamba_kv = _FakeKVCache(pool.max_slots("mamba"))
        full_alloc = MultiEndedAllocator(
            kvcache=full_kv,
            unified_buffer=pool,
            sub_pool_name="full",
            device=_DEV,
            is_id_owner=True,
            lazy_compaction=True,
        )
        mamba_alloc = MultiEndedAllocator(
            kvcache=mamba_kv,
            unified_buffer=pool,
            sub_pool_name="mamba",
            device=_DEV,
            is_id_owner=True,
            lazy_compaction=True,
        )
        full_alloc.bind_peer(mamba_alloc)
        mamba_alloc.bind_peer(full_alloc)
        full_alloc._lazy_max_moves_per_call = move_cap
        return pool, full_alloc, full_kv

    def test_lazy_retract_churn_no_ghost(self):
        """Regression: heavy free/flush churn in LAZY mode must never
        leave a 'ghost' page (p2v<0 not registered in _free_phys_pages or
        _pending_reuse). This reproduces the retract×lazy-compaction pattern.
        The fail-fast ghost invariant is enabled so any ghost trips at the
        CREATING op (free_lazy / flush_exit), not at a later survivor walk.

        CPU scope: the GPU write-race *urgent* path (unfired CUDA events ->
        _pending_reuse) needs a real device; this covers the no-event
        move/absorb/free/released_fired bookkeeping.
        """
        rng = random.Random(0x5373)
        _, alloc, kv = self._build_lazy_full(n_full_slots=64, move_cap=2)
        live = []
        # Bias toward near-capacity occupancy (watermark high, holes pile
        # up) then churn free/flush — the conditions that surface the bug.
        for _ in range(3000):
            avail = alloc.available_size()
            if (rng.random() < 0.55 and avail > 0) or not live:
                n = rng.randint(1, min(5, max(1, avail)))
                v = self._alloc(alloc, kv, n)
                if v is not None:
                    live.append(v)
            else:
                v = live.pop(rng.randrange(len(live)))
                self._free(alloc, kv, v)  # often non-boundary -> a hole
            if rng.random() < 0.5:
                alloc.flush_opportunistic()  # partial flush (move_cap)
        # Drain to quiescence; must end empty AND ghost-free.
        for v in live:
            self._free(alloc, kv, v)
        for _ in range(64):
            alloc.flush_opportunistic()
        self.assertEqual(alloc.allocated_count(), 0)

    def test_double_free_raises(self):
        _, full_alloc, mamba_alloc, full_kv, mamba_kv = self._build_pair()
        v = self._alloc(full_alloc, full_kv, 3)
        self._free(full_alloc, full_kv, v)
        with self.assertRaises(AssertionError):
            full_alloc.free(v)

    # -- `out=` parameter regression tests --

    def test_translate_kv_loc_with_out_writes_inplace(self):
        """REGRESSION: `translate_kv_loc(virt, out=buf)` must
        modify `buf` in place AND preserve `buf.data_ptr()` — the buffer-
        stability invariant for cuda-graph capture."""
        _, full_alloc, _, full_kv, _ = self._build_pair()
        v = self._alloc(full_alloc, full_kv, 5)
        buf = torch.empty(v.shape, dtype=torch.int64, device=_DEV)
        ptr_before = buf.data_ptr()
        ret = full_alloc.translate_kv_loc(v, out=buf)
        self.assertIs(ret, buf, "must return the `out=` buffer, not a fresh tensor")
        self.assertEqual(
            buf.data_ptr(), ptr_before, "out= buffer's data_ptr must be stable"
        )
        # Result matches v2p directly (page_size == 1 here)
        expected = full_alloc.virtual_to_physical[v]
        self.assertTrue(bool((buf == expected).all().item()))

    def test_translate_kv_loc_without_out_returns_fresh_tensor(self):
        """REGRESSION: without `out=`, behavior returns a fresh tensor."""
        _, full_alloc, _, full_kv, _ = self._build_pair()
        v = self._alloc(full_alloc, full_kv, 5)
        ret = full_alloc.translate_kv_loc(v)
        # Fresh tensor: different storage from v2p table
        self.assertNotEqual(ret.data_ptr(), full_alloc.virtual_to_physical.data_ptr())
        expected = full_alloc.virtual_to_physical[v]
        self.assertTrue(bool((ret == expected).all().item()))

    def test_translate_kv_loc_out_matches_no_out(self):
        """REGRESSION: result of translate_kv_loc(v, out=buf) byte-equals
        translate_kv_loc(v)."""
        _, full_alloc, _, full_kv, _ = self._build_pair()
        v = self._alloc(full_alloc, full_kv, 5)
        buf = torch.empty(v.shape, dtype=torch.int64, device=_DEV)
        with_out = full_alloc.translate_kv_loc(v, out=buf)
        no_out = full_alloc.translate_kv_loc(v)
        self.assertTrue(bool((with_out == no_out).all().item()))

    def test_translate_kv_loc_dtype_assertion(self):
        """REGRESSION: wrong-dtype `out=` (int32 instead of int64) raises
        AssertionError. Guards against the copy/paste hazard where someone
        might allocate the full-physical buffer with the SWA int32 pattern."""
        _, full_alloc, _, full_kv, _ = self._build_pair()
        v = self._alloc(full_alloc, full_kv, 5)
        wrong_dtype = torch.empty(v.shape, dtype=torch.int32, device=_DEV)
        with self.assertRaises(AssertionError):
            full_alloc.translate_kv_loc(v, out=wrong_dtype)

    def test_translate_kv_loc_shape_assertion(self):
        """REGRESSION: mismatched `out=` shape raises AssertionError."""
        _, full_alloc, _, full_kv, _ = self._build_pair()
        v = self._alloc(full_alloc, full_kv, 5)
        wrong_shape = torch.empty((v.numel() + 1,), dtype=torch.int64, device=_DEV)
        with self.assertRaises(AssertionError):
            full_alloc.translate_kv_loc(v, out=wrong_shape)

    # REGRESSION: `translate_kv_loc(buf, out=buf)` — same
    # tensor for input and output — is the canonical in-place form used by
    # the cuda-graph capture/replay paths in `triton_backend.py`:
    #
    #     self._translate_kv_loc(kv_indices, out=kv_indices)
    #
    # A naive implementation routes this through
    # `torch.index_select(v2p, 0, virt_tokens, out=out)`, which crashes with
    #     "unsupported operation: some elements of the input tensor and the
    #      written-to tensor refer to a single memory location"
    # because index_select does NOT support aliasing between `index` and
    # `out`. Fix: gather into a transient buffer then `out.copy_(tmp)`.
    def test_translate_kv_loc_with_out_aliasing_input(self):
        """REGRESSION: in-place form `translate_kv_loc(buf, out=buf)` must
        succeed and produce identical results to the no-out form."""
        _, full_alloc, _, full_kv, _ = self._build_pair()
        v_orig = self._alloc(full_alloc, full_kv, 5).clone()
        # Save the expected output (no-out form) before mutating `buf`.
        expected = full_alloc.translate_kv_loc(v_orig)
        # Now exercise the aliasing form: buf serves as BOTH input and out.
        buf = v_orig.clone()
        ptr_before = buf.data_ptr()
        ret = full_alloc.translate_kv_loc(buf, out=buf)
        self.assertIs(ret, buf)
        self.assertEqual(
            buf.data_ptr(),
            ptr_before,
            "out= buffer's data_ptr must be stable (cuda-graph invariant)",
        )
        self.assertTrue(
            bool((buf == expected).all().item()),
            "in-place result must equal no-out result",
        )

    # Tombstone-safety clamp regression.
    #
    # The captured cuda-graph paths (full-layer set_kv_buffer elif,
    # init_forward_metadata_*_cuda_graph kv_indices translate, init_new
    # precompute) all eventually call `translate_kv_loc` against `v2p_full`.
    # Padded / stale-tail entries in the cuda-graph input buffers can carry
    # virtual ids whose v2p entries got tombstoned (-1) by free/compaction
    # between replays. Without a clamp, the captured `k_buffer[result[i]]`
    # would index at -1 (illegal memory access). The clamp routes those to
    # physical slot 0 (the reserved padding sink under the `min_slot_index`
    # invariant — bytes [0, entry_max) hold no real data).
    # These tests lock in the clamp contract so a future refactor can't
    # quietly remove it and re-introduce the crash.
    def test_translate_kv_loc_clamps_tombstoned_v2p(self):
        """`translate_kv_loc` must clamp `v2p[v] == -1` entries to 0 (the
        padding sink). Required for cuda-graph capture safety."""
        _, full_alloc, _, full_kv, _ = self._build_pair()
        v = self._alloc(full_alloc, full_kv, 5)
        # Inject a tombstone at one of the live virtual id positions WITHOUT
        # going through `free` (which would also touch p2v / compaction).
        # This emulates the steady-state where a captured-graph input
        # buffer's padded/stale entries reference virtual ids that have
        # since been tombstoned by free/compaction.
        v_tombstoned = int(v[2].item())
        full_alloc.virtual_to_physical[v_tombstoned] = -1
        # No-out form: result must clamp.
        out = full_alloc.translate_kv_loc(v)
        self.assertTrue(
            bool((out >= 0).all().item()),
            f"translate_kv_loc must clamp tombstoned entries to >=0, got {out.tolist()}",
        )
        self.assertEqual(
            int(out[2].item()),
            0,
            "tombstoned virtual id must map to slot 0 (padding sink)",
        )

    def test_translate_kv_loc_with_out_clamps_tombstoned_v2p(self):
        """`translate_kv_loc(..., out=buf)` (the captured-graph path) must
        clamp tombstoned entries in-place."""
        _, full_alloc, _, full_kv, _ = self._build_pair()
        v = self._alloc(full_alloc, full_kv, 5).clone()
        full_alloc.virtual_to_physical[int(v[1].item())] = -1
        buf = torch.empty_like(v)
        ret = full_alloc.translate_kv_loc(v, out=buf)
        self.assertIs(ret, buf)
        self.assertTrue(
            bool((buf >= 0).all().item()),
            "out= path must clamp tombstoned entries",
        )
        self.assertEqual(int(buf[1].item()), 0)


# ---------------------------------------------------------------------------
# Shared SWA composite — unit tests
# ---------------------------------------------------------------------------


class _FakeUnifiedSWAKVPool:
    """Minimal stand-in for `UnifiedSWAKVPool` that the composite allocator
    needs. Exposes the two sub-pool views (each a `_FakeKVCache` with an
    `attach_allocator` no-op) and an `attach_allocators` setter.

    CPU-only — avoids constructing a real `UnifiedMHATokenToKVPool` (which
    instantiates `MHATokenToKVPool` and is heavier than these tests need).
    """

    class _SubKV(_FakeKVCache):
        def __init__(self, max_slots):
            super().__init__(max_slots)
            self.allocator = None

        def attach_allocator(self, allocator):
            self.allocator = allocator

    def __init__(self, shared_pool: UnifiedKVPool):
        self.full_kv_pool = self._SubKV(shared_pool.max_slots("full"))
        self.swa_kv_pool = self._SubKV(shared_pool.max_slots("swa"))
        self._full_allocator = None
        self._swa_allocator = None

    def attach_allocators(self, *, full_allocator, swa_allocator):
        self._full_allocator = full_allocator
        self._swa_allocator = swa_allocator


class TestUnifiedSWATokenToKVPoolAllocator(CpuPoolTestCase):
    """Tests for the SWA composite — joint byte-budget, slot-conservation
    leak invariant, tombstone semantics for `free_swa`, divergent compaction
    of the two sub-pools, and the alloc-rollback path.

    These tests cover the core invariants: joint byte-budget,
    slot-conservation, the `schedulable_*` split, and watermark
    rollback."""

    def _build(
        self,
        n_full_slots=32,
        n_swa_slots=16,
        full_layer_num=4,
        swa_layer_num=2,
        head_num=2,
        head_dim=4,
    ):
        full_spec = MHASubPoolSpec(
            name="full",
            layer_num=full_layer_num,
            head_num=head_num,
            head_dim=head_dim,
            store_dtype=torch.float16,
            grow_direction="up",
        )
        swa_spec = MHASubPoolSpec(
            name="swa",
            layer_num=swa_layer_num,
            head_num=head_num,
            head_dim=head_dim,
            store_dtype=torch.float16,
            grow_direction="down",
        )
        total = (
            n_full_slots * full_spec.entry_bytes()
            + n_swa_slots * swa_spec.entry_bytes()
        )
        pool = UnifiedKVPool(
            total_bytes=total,
            sub_pool_specs=[full_spec, swa_spec],
            device=_DEV,
            enable_memory_saver=False,
        )
        kvcache = _FakeUnifiedSWAKVPool(pool)
        allocator = UnifiedSWATokenToKVPoolAllocator(
            unified_buffer=pool,
            kvcache=kvcache,
            device=_DEV,
            full_max_total_num_tokens=n_full_slots,
            swa_max_total_num_tokens=n_swa_slots,
            need_sort=False,
            forward_stream=None,
        )
        return pool, allocator, kvcache

    def _alloc(self, allocator, kvcache, n):
        """Allocate N virtual ids; stamp the data marker on both sub-pools."""
        v = allocator.alloc(n)
        if v is None:
            return None
        full_phys = allocator.full_attn_allocator.virtual_to_physical[v]
        swa_phys = allocator.swa_attn_allocator.virtual_to_physical[v]
        kvcache.full_kv_pool.buf[full_phys] = v
        kvcache.swa_kv_pool.buf[swa_phys] = v
        return v

    def _free(self, allocator, kvcache, v):
        """Erase markers on both sub-pools (mirror compaction's no-data-at
        -freed-slot invariant), then call the composite's free."""
        full_phys = allocator.full_attn_allocator.virtual_to_physical[v]
        swa_phys = allocator.swa_attn_allocator.virtual_to_physical[v]
        # erase only the LIVE swa entries (`free_swa` may have already
        # tombstoned some of `v`).
        valid_swa = swa_phys[swa_phys >= 0]
        kvcache.full_kv_pool.buf[full_phys] = -1
        kvcache.swa_kv_pool.buf[valid_swa] = -1
        allocator.free(v)

    def _check_sub_pool_invariants(self, sub, kv):
        """Per-sub-pool: v2p ∘ p2v identity on the live set, hole-free
        allocated band, data followed relocations."""
        v2p = sub.virtual_to_physical
        p2v = sub.physical_to_virtual
        live_v = [v for v in range(1, sub.num_virtual_ids) if int(v2p[v].item()) != -1]
        for v in live_v:
            p = int(v2p[v].item())
            self.assertEqual(int(p2v[p].item()), v)
            # data marker followed any relocation
            self.assertEqual(int(kv.buf[p].item()), v)
        if sub.grow_direction == "up":
            lo, hi = sub.min_slot_index, sub.watermark_physical
        else:
            lo, hi = sub.watermark_physical + 1, sub.max_slots
        self.assertEqual(hi - lo, len(live_v))
        for p in range(lo, hi):
            self.assertNotEqual(int(p2v[p].item()), -1)

    # 1. Both peers hold a physical slot per virtual after composite alloc.
    def test_swa_alloc_both_peers_hold(self):
        _, allocator, _ = self._build()
        v = allocator.alloc(3)
        self.assertIsNotNone(v)
        self.assertEqual(int(v.numel()), 3)
        full_v2p = allocator.full_attn_allocator.virtual_to_physical
        swa_v2p = allocator.swa_attn_allocator.virtual_to_physical
        for vi in v.tolist():
            self.assertGreaterEqual(int(full_v2p[vi].item()), 0)
            self.assertGreaterEqual(int(swa_v2p[vi].item()), 0)
        # Full sub-pool is id-owner -> the minted ids are out of free_virtual_ids.
        free_full = set(
            int(x) for x in allocator.full_attn_allocator.free_virtual_ids.tolist()
        )
        self.assertTrue(set(v.tolist()).isdisjoint(free_full))
        # Swa sub-pool is non-owner -> free_virtual_ids is None.
        self.assertIsNone(allocator.swa_attn_allocator.free_virtual_ids)

    # 2. Composite `free` releases both sub-pools' v2p; the virtual goes back
    # to the full id-owner's free list.
    def test_swa_free_releases_both(self):
        _, allocator, kvcache = self._build()
        v = self._alloc(allocator, kvcache, 3)
        self._free(allocator, kvcache, v)
        for vi in v.tolist():
            self.assertEqual(
                int(allocator.full_attn_allocator.virtual_to_physical[vi].item()), -1
            )
            self.assertEqual(
                int(allocator.swa_attn_allocator.virtual_to_physical[vi].item()), -1
            )
        free_full = set(
            int(x) for x in allocator.full_attn_allocator.free_virtual_ids.tolist()
        )
        self.assertTrue(set(v.tolist()).issubset(free_full))

    # 3. `free_swa` tombstones swa side only; virtual + full-physical stay live.
    def test_swa_free_swa_keeps_virtual_alive(self):
        _, allocator, kvcache = self._build()
        v = self._alloc(allocator, kvcache, 3)
        # Tombstone the middle one. Erase its swa marker first (compaction
        # will run inside `free_swa`).
        target = v[1:2]
        target_swa = allocator.swa_attn_allocator.virtual_to_physical[target]
        kvcache.swa_kv_pool.buf[target_swa] = -1
        allocator.free_swa(target)
        tgt = int(target.item())
        # full side still bound:
        self.assertGreaterEqual(
            int(allocator.full_attn_allocator.virtual_to_physical[tgt].item()), 0
        )
        # swa side tombstoned:
        self.assertEqual(
            int(allocator.swa_attn_allocator.virtual_to_physical[tgt].item()), -1
        )
        # NOT recycled to the id-owner's free list yet:
        free_full = set(
            int(x) for x in allocator.full_attn_allocator.free_virtual_ids.tolist()
        )
        self.assertNotIn(tgt, free_full)
        # composite `free` of the same virtual still works (filters out
        # already-tombstoned on the swa side).
        full_phys = int(allocator.full_attn_allocator.virtual_to_physical[tgt].item())
        kvcache.full_kv_pool.buf[full_phys] = -1
        allocator.free(target)
        # now in free list:
        free_full = set(
            int(x) for x in allocator.full_attn_allocator.free_virtual_ids.tolist()
        )
        self.assertIn(tgt, free_full)

    # 4. Compaction diverges between the two sub-pools (each runs its own).
    def test_swa_compaction_diverges_physical_layout(self):
        _, allocator, kvcache = self._build()
        a = self._alloc(allocator, kvcache, 1)
        b = self._alloc(allocator, kvcache, 1)
        c = self._alloc(allocator, kvcache, 1)
        # Snapshot swa-side physical for c BEFORE we free_swa(b).
        c_swa_before = int(allocator.swa_attn_allocator.virtual_to_physical[c].item())
        c_full_before = int(allocator.full_attn_allocator.virtual_to_physical[c].item())
        # Tombstone b on swa only.
        b_swa = allocator.swa_attn_allocator.virtual_to_physical[b]
        kvcache.swa_kv_pool.buf[b_swa] = -1
        allocator.free_swa(b)
        # c's full-physical UNCHANGED (full side did not compact):
        self.assertEqual(
            int(allocator.full_attn_allocator.virtual_to_physical[c].item()),
            c_full_before,
        )
        # c's swa-physical MUST have moved (b was interior to swa's
        # allocated band on grow-down: a then b then c means b is between
        # them; freeing b triggers compaction relocating c into b's slot).
        c_swa_after = int(allocator.swa_attn_allocator.virtual_to_physical[c].item())
        self.assertNotEqual(c_swa_after, c_swa_before)
        # Per-sub-pool invariants still hold.
        self._check_sub_pool_invariants(
            allocator.full_attn_allocator, kvcache.full_kv_pool
        )
        self._check_sub_pool_invariants(
            allocator.swa_attn_allocator, kvcache.swa_kv_pool
        )

    # 5. Byte-frontier coordination — peer-aware available_size shrinks as
    # the peer grows.
    def test_swa_byte_frontier_coordination(self):
        _, allocator, kvcache = self._build(n_full_slots=8, n_swa_slots=8)
        avail0 = allocator.available_size()
        # Allocate enough that the joint budget visibly tightens.
        self._alloc(allocator, kvcache, 3)
        self.assertLess(allocator.available_size(), avail0)
        # Joint budget enforcement: over-alloc returns None.
        self.assertIsNone(allocator.alloc(allocator.available_size() + 1))

    # 6. Randomized stress — invariants under mixed alloc / free / free_swa.
    def test_swa_randomized_alloc_free_freeswa(self):
        rng = random.Random(0xBADBEE)
        _, allocator, kvcache = self._build(
            n_full_slots=48, n_swa_slots=24, full_layer_num=3, swa_layer_num=3
        )
        live = []  # list of (virtual-id tensor)
        # Independent conservation ledger. The dynamic leak totals are
        # `size_x = schedulable_x + allocated_count()` BY CONSTRUCTION, so
        # asserting that identity against the allocator's own views would be
        # a tautology — the checker's invariant genuinely reduces to
        # `allocated == live tokens the RADIX layer believes it holds`, and
        # this ledger plays the radix layer's role: it must match
        # `allocated_count()` on both sides after every op, under holes,
        # compaction, borrowing, and one-sided free_swa.
        full_live_tokens = 0
        swa_live_tokens = 0
        for _ in range(400):
            r = rng.random()
            if r < 0.5 or not live:  # alloc
                n = rng.randint(1, 4)
                v = self._alloc(allocator, kvcache, n)
                if v is not None:
                    live.append(("live", v))
                    full_live_tokens += int(v.numel())
                    swa_live_tokens += int(v.numel())
            elif r < 0.8:  # composite free
                idx = rng.randrange(len(live))
                kind, v = live.pop(idx)
                self._free(allocator, kvcache, v)
                full_live_tokens -= int(v.numel())
                if kind == "live":  # swa side already released for tombstones
                    swa_live_tokens -= int(v.numel())
            else:  # free_swa on some entries
                idx = rng.randrange(len(live))
                kind, v = live[idx]
                if kind != "live":
                    continue
                # Tombstone all of v on swa only.
                swa_phys = allocator.swa_attn_allocator.virtual_to_physical[v]
                kvcache.swa_kv_pool.buf[swa_phys] = -1
                allocator.free_swa(v)
                live[idx] = ("swa_tomb", v)
                swa_live_tokens -= int(v.numel())
            # Invariants after every op.
            self._check_sub_pool_invariants(
                allocator.full_attn_allocator, kvcache.full_kv_pool
            )
            self._check_sub_pool_invariants(
                allocator.swa_attn_allocator, kvcache.swa_kv_pool
            )
            # Conservation vs the INDEPENDENT ledger (see above).
            self.assertEqual(
                allocator.full_attn_allocator.allocated_count(), full_live_tokens
            )
            self.assertEqual(
                allocator.swa_attn_allocator.allocated_count(), swa_live_tokens
            )
        # Drain.
        for _, v in live:
            self._free(allocator, kvcache, v)
        self.assertEqual(allocator.full_attn_allocator.allocated_count(), 0)
        self.assertEqual(allocator.swa_attn_allocator.allocated_count(), 0)

    # 6b. Joint-byte-budget admission soundness (mechanical byte-conservation
    # check).
    def test_swa_joint_byte_budget_admission_soundness(self):
        """Admitting under the JOINT BYTE budget — each request charged
        `T x (e_full + e_swa)` bytes against `schedulable_free_bytes()` —
        never promises more than the composite can actually allocate; the
        naive dual per-side budget provably over-admits, because both
        per-side views credit the SAME shared gap. This is the soundness job
        the deleted `min(conserve, schedulable)` caps (and later the
        full-token cross-charge) used to do — now held by the single byte
        ledger, with no per-request ceil rounding."""
        REQ_TOKENS = 4
        pool, allocator, kvcache = self._build(
            n_full_slots=64, n_swa_slots=64, full_layer_num=4, swa_layer_num=2
        )
        # The composite allocates every token on BOTH sides: T = S = n.
        naive_k = min(
            allocator.full_available_size() // REQ_TOKENS,
            allocator.swa_available_size() // REQ_TOKENS,
        )
        charge_bytes = allocator.alloc_tokens_to_bytes(REQ_TOKENS)
        # e_swa < e_full here (2 vs 4 layers) — both sides must be priced,
        # and the swa side must be a real, non-trivial surcharge.
        self.assertGreater(
            charge_bytes,
            REQ_TOKENS * allocator.full_attn_allocator.entry_bytes,
        )
        sound_k = allocator.schedulable_free_bytes() // charge_bytes
        self.assertLess(sound_k, naive_k, "naive dual budget must over-promise")

        # Every cross-charge-admitted request must be allocatable.
        live = []
        for i in range(sound_k):
            v = self._alloc(allocator, kvcache, REQ_TOKENS)
            self.assertIsNotNone(
                v, f"request {i}/{sound_k} admitted by the cross-charged "
                f"budget failed to allocate — the charge under-prices"
            )
            live.append(v)
        # ... and the naive count must NOT be (the shared gap was promised
        # twice): keep allocating up to naive_k and require a shortfall.
        over_admitted_ok = 0
        for _ in range(naive_k - sound_k):
            v = self._alloc(allocator, kvcache, REQ_TOKENS)
            if v is None:
                break
            live.append(v)
            over_admitted_ok += 1
        self.assertLess(
            over_admitted_ok,
            naive_k - sound_k,
            "the naive per-side budgets were fully allocatable — then the "
            "cross-charge is stricter than physics and over-reserves",
        )
        for v in live:
            self._free(allocator, kvcache, v)

    # 6c. Eviction sizing must use the JOINT view.
    def test_swa_evict_sizing_uses_joint_view(self):
        """`evict_from_tree_cache` sized SWA eviction from the per-side views;
        under the unified composite both views alias the SAME gap, so a state
        with each side individually >= need but joint < need evicted NOTHING
        and the caller's alloc died on the fail-loud shortfall. The unified
        branch must size (and re-check) against the joint `available_size()`."""
        from sglang.srt.mem_cache.common import evict_from_tree_cache

        _, allocator, kvcache = self._build(
            n_full_slots=64, n_swa_slots=64, full_layer_num=3, swa_layer_num=3
        )

        # Fill most of the pool, keep a handle on evictable "radix" entries.
        held = []
        while True:
            v = self._alloc(allocator, kvcache, 4)
            if v is None or allocator.available_size() < 16:
                if v is not None:
                    held.append(v)
                break
            held.append(v)

        outer = self

        class _FakeTree:
            """Minimal radix stand-in: evict(params) frees held entries."""

            def __init__(self, alloc):
                self.token_to_kv_pool_allocator = alloc
                self.evict_calls = []

            def is_chunk_cache(self):
                return False

            def full_evictable_size(self):
                return sum(int(v.numel()) for v in held)

            def swa_evictable_size(self):
                return sum(int(v.numel()) for v in held)

            def evict(self, params):
                self.evict_calls.append(params)
                freed = 0
                while held and freed < params.num_tokens:
                    v = held.pop()
                    freed += int(v.numel())
                    outer._free(allocator, kvcache, v)

        tree = _FakeTree(allocator)
        need = allocator.available_size() + 8  # joint shortfall by construction
        # Per-side views may each exceed `need` (they price the gap at their
        # own entry size); the joint view must drive the eviction anyway.
        evict_from_tree_cache(tree, need)
        self.assertGreater(len(tree.evict_calls), 0, "joint shortfall not seen")
        self.assertGreaterEqual(
            allocator.available_size(),
            need,
            "eviction under-sized: the subsequent alloc would hit the "
            "fail-loud extend shortfall",
        )
        self.assertIsNotNone(self._alloc(allocator, kvcache, need))

    # 7. Joint byte-budget pre-check.
    def test_swa_joint_byte_budget_pre_check(self):
        # Pick sizes where the byte gap, not slot-index headroom, is the bind.
        full_spec = MHASubPoolSpec(
            name="full",
            layer_num=2,
            head_num=2,
            head_dim=4,
            store_dtype=torch.float16,
            grow_direction="up",
        )
        swa_spec = MHASubPoolSpec(
            name="swa",
            layer_num=2,
            head_num=2,
            head_dim=4,
            store_dtype=torch.float16,
            grow_direction="down",
        )
        n_full, n_swa = 10, 10
        total = n_full * full_spec.entry_bytes() + n_swa * swa_spec.entry_bytes()
        pool = UnifiedKVPool(
            total_bytes=total,
            sub_pool_specs=[full_spec, swa_spec],
            device=_DEV,
            enable_memory_saver=False,
        )
        kvcache = _FakeUnifiedSWAKVPool(pool)
        allocator = UnifiedSWATokenToKVPoolAllocator(
            unified_buffer=pool,
            kvcache=kvcache,
            device=_DEV,
            full_max_total_num_tokens=n_full,
            swa_max_total_num_tokens=n_swa,
            need_sort=False,
            forward_stream=None,
        )
        fa = allocator.full_attn_allocator
        sa = allocator.swa_attn_allocator
        # Compute the "naive min" against the joint budget — at idle, the
        # joint budget is strictly less than min(full.available, swa.available)
        # because the joint uses (entry_full + entry_swa) per slot.
        naive = min(fa.available_size(), sa.available_size())
        joint = allocator.available_size()
        # The joint must be no greater than naive (typically strictly less).
        self.assertLessEqual(joint, naive)
        # And it must equal `gap_bytes // (entry_full + entry_swa)` clamped
        # by slot-room.
        gap = sa._byte_low_frontier() - fa._byte_high_frontier()
        expected = min(
            gap // (fa.entry_bytes + sa.entry_bytes),
            fa.max_slots - fa.min_slot_index - fa.allocated_count(),
            sa.max_slots - sa.min_slot_index - sa.allocated_count(),
        )
        self.assertEqual(joint, expected)

    # 8. Watermark rollback on partial alloc failure.
    def test_swa_alloc_swa_failure_is_fail_loud(self):
        """The SWA composite runs a tight JOINT pre-check before allocating, so
        a swa-side ``alloc_with_virtual`` failure after the full-side alloc can
        only mean an internal-state inconsistency. By design (``UnifiedSWA.alloc``:
        "assert rather than silently rollback") that surfaces as a loud error,
        NOT a silent ``None`` / rollback — masking it would hide the bug. The
        real ``alloc_with_virtual`` self-asserts on shortfall, so the production
        path is fail-loud too; here we force the failure to prove it propagates.
        """
        _, allocator, kvcache = self._build()
        sa = allocator.swa_attn_allocator
        original = sa.alloc_with_virtual

        def _bomb(virtual_ids):
            raise AssertionError("synthetic alloc_with_virtual failure")

        sa.alloc_with_virtual = _bomb
        try:
            with self.assertRaises(AssertionError):
                allocator.alloc(3)
        finally:
            sa.alloc_with_virtual = original

    # -- `out=` parameter regression tests for the SWA composite --

    def test_swa_translate_kv_loc_with_out_writes_inplace(self):
        """REGRESSION: composite delegates to base-class
        translate_kv_loc with `out=` passthrough. Result lands in `buf`."""
        _, allocator, _ = self._build()
        v = allocator.alloc(4)
        self.assertIsNotNone(v)
        buf = torch.empty(v.shape, dtype=torch.int64, device=_DEV)
        ptr_before = buf.data_ptr()
        ret = allocator.translate_kv_loc(v, out=buf)
        self.assertIs(ret, buf)
        self.assertEqual(buf.data_ptr(), ptr_before)
        expected = allocator.translate_kv_loc(v)
        self.assertTrue(bool((buf == expected).all().item()))

    def test_swa_translate_loc_from_full_to_swa_with_out_writes_inplace(self):
        """REGRESSION: `translate_loc_from_full_to_swa(v, out=buf)`
        must modify `buf` in place AND preserve `buf.data_ptr()`. `out=`
        buffer MUST be int32 (matches SWA Triton kernel contract)."""
        _, allocator, _ = self._build()
        v = allocator.alloc(4)
        self.assertIsNotNone(v)
        buf = torch.empty(v.shape, dtype=torch.int32, device=_DEV)
        ptr_before = buf.data_ptr()
        ret = allocator.translate_loc_from_full_to_swa(v, out=buf)
        self.assertIs(ret, buf)
        self.assertEqual(buf.data_ptr(), ptr_before)
        # Byte-identical to the no-out form:
        no_out = allocator.translate_loc_from_full_to_swa(v)
        self.assertEqual(no_out.dtype, torch.int32)
        self.assertTrue(bool((buf == no_out).all().item()))

    def test_swa_translate_loc_from_full_to_swa_dtype_assertion(self):
        """REGRESSION: wrong-dtype `out=` (int64 instead of int32)
        raises AssertionError. Guards against accidentally reusing the int64
        full-physical buffer pattern for the SWA precompute."""
        _, allocator, _ = self._build()
        v = allocator.alloc(4)
        self.assertIsNotNone(v)
        wrong_dtype = torch.empty(v.shape, dtype=torch.int64, device=_DEV)
        with self.assertRaises(AssertionError):
            allocator.translate_loc_from_full_to_swa(v, out=wrong_dtype)

    # Tombstone-safety clamp for SWA — mirrors the full-side test
    # in `TestMultiEndedAllocator`. The captured SWA attention kernel reads
    # `swa_k_buffer[result[i]]` at replay; without the clamp, a tombstoned
    # `v2p_swa[v] == -1` would index at `swa_k_buffer[-1]` (illegal access).
    def test_swa_translate_loc_from_full_to_swa_clamps_tombstoned(self):
        _, allocator, _ = self._build()
        v = allocator.alloc(4)
        self.assertIsNotNone(v)
        # Inject a tombstone on the swa side at one of the live virtual ids.
        v_tomb = int(v[1].item())
        allocator.swa_attn_allocator.virtual_to_physical[v_tomb] = -1
        # No-out form: result must be int32 AND every entry >= 0.
        out = allocator.translate_loc_from_full_to_swa(v)
        self.assertEqual(out.dtype, torch.int32)
        self.assertTrue(
            bool((out >= 0).all().item()),
            "translate_loc_from_full_to_swa must clamp tombstoned to >=0",
        )
        self.assertEqual(int(out[1].item()), 0)
        # out= form (int32 buffer) must also clamp.
        buf = torch.empty(v.shape, dtype=torch.int32, device=_DEV)
        ret = allocator.translate_loc_from_full_to_swa(v, out=buf)
        self.assertIs(ret, buf)
        self.assertTrue(bool((buf >= 0).all().item()))
        self.assertEqual(int(buf[1].item()), 0)


# ---------------------------------------------------------------------------
# page_size > 1 — paged unit tests
# ---------------------------------------------------------------------------


class TestPagedMultiEndedAllocator(CpuPoolTestCase):
    """Per-sub-pool paged tests for `MultiEndedAllocator(page_size=...)`.

    All tests use ``page_size = 8`` against a buffer sized for ~16 pages per
    sub-pool. Invariants are page-granular: free-list, v2p/p2v tables, and
    compaction operate on pages. The external API (alloc → token ids, free
    takes token ids) is byte-identical to the page_size == 1 case.
    """

    PAGE_SIZE = 8

    def _build(self, n_full_pages=16, n_swa_pages=8, full_layer_num=2, swa_layer_num=2):
        full_spec = MHASubPoolSpec(
            name="full",
            layer_num=full_layer_num,
            head_num=2,
            head_dim=4,
            store_dtype=torch.float16,
            grow_direction="up",
        )
        swa_spec = MHASubPoolSpec(
            name="swa",
            layer_num=swa_layer_num,
            head_num=2,
            head_dim=4,
            store_dtype=torch.float16,
            grow_direction="down",
        )
        # entry_bytes_per_page = layer_num * (k_row + v_row) * page_size
        # We size the buffer to fit `n_full_pages` full-pages + `n_swa_pages`
        # swa-pages (token-equivalent: n_*_pages * page_size).
        total = (
            n_full_pages * self.PAGE_SIZE * full_spec.entry_bytes()
            + n_swa_pages * self.PAGE_SIZE * swa_spec.entry_bytes()
        )
        pool = UnifiedKVPool(
            total_bytes=total,
            sub_pool_specs=[full_spec, swa_spec],
            device=_DEV,
            enable_memory_saver=False,
        )
        full_kv = _FakeKVCache(pool.max_slots("full"))
        swa_kv = _FakeKVCache(pool.max_slots("swa"))
        full_alloc = MultiEndedAllocator(
            kvcache=full_kv,
            unified_buffer=pool,
            sub_pool_name="full",
            device=_DEV,
            is_id_owner=True,
            page_size=self.PAGE_SIZE,
        )
        swa_alloc = MultiEndedAllocator(
            kvcache=swa_kv,
            unified_buffer=pool,
            sub_pool_name="swa",
            device=_DEV,
            is_id_owner=True,
            page_size=self.PAGE_SIZE,
        )
        full_alloc.bind_peer(swa_alloc)
        swa_alloc.bind_peer(full_alloc)
        return pool, full_alloc, swa_alloc, full_kv, swa_kv

    def _stamp_tokens(
        self, alloc: MultiEndedAllocator, kv: _FakeKVCache, v_tokens: torch.Tensor
    ):
        """Mark `kv.buf[phys_token] = some_unique_id` for every returned
        token. Uses the alloc's v2p_page table to compute physical tokens."""
        if v_tokens.numel() == 0:
            return
        ps = alloc.page_size
        virt_pages = v_tokens // ps
        offsets = v_tokens % ps
        phys_pages = alloc.virtual_to_physical[virt_pages]
        phys_tokens = phys_pages * ps + offsets
        kv.buf[phys_tokens] = v_tokens

    def _check_invariants(
        self, alloc: MultiEndedAllocator, kv: _FakeKVCache, stamped_tokens: dict
    ):
        v2p = alloc.virtual_to_physical
        p2v = alloc.physical_to_virtual
        ps = alloc.page_size
        # Live virtual pages (excluding the reserved padding page 0).
        live_v_pages = [
            v for v in range(1, alloc.num_pages) if int(v2p[v].item()) != -1
        ]
        # Mutual inverse on the live page set.
        for v_page in live_v_pages:
            p_page = int(v2p[v_page].item())
            self.assertEqual(
                int(p2v[p_page].item()),
                v_page,
                f"p2v[{p_page}] != {v_page}",
            )
        # Allocated physical-page range is hole-free + matches live count.
        if alloc.grow_direction == "up":
            alloc_lo, alloc_hi = alloc.min_page_index, alloc.watermark_physical
        else:
            alloc_lo, alloc_hi = (
                alloc.watermark_physical + 1,
                alloc.num_pages,
            )
        self.assertEqual(alloc_hi - alloc_lo, len(live_v_pages))
        for p_page in range(alloc_lo, alloc_hi):
            self.assertNotEqual(
                int(p2v[p_page].item()), -1, f"hole at physical page {p_page}"
            )
        # Free virtual page ids ∪ live = [min_page_index, num_pages).
        free_set = set(int(x) for x in alloc.free_virtual_ids.tolist())
        self.assertEqual(
            free_set | set(live_v_pages),
            set(range(alloc.min_page_index, alloc.num_pages)),
        )
        self.assertEqual(free_set & set(live_v_pages), set())
        # For every token we stamped, verify data followed any relocations.
        for v_tok, mark in stamped_tokens.items():
            v_page = v_tok // ps
            offset = v_tok % ps
            p_page_t = int(v2p[v_page].item())
            if p_page_t == -1:
                continue  # was freed; don't check
            phys_tok = p_page_t * ps + offset
            self.assertEqual(
                int(kv.buf[phys_tok].item()),
                mark,
                f"data drift: stamped {mark} at virtual token {v_tok} "
                f"(page {v_page}+offset {offset}) — found {int(kv.buf[phys_tok].item())}",
            )

    # 1. alloc(N) returns N TOKEN ids that are page-aligned.
    def test_paged_alloc_token_aligned(self):
        _, full_alloc, swa_alloc, full_kv, swa_kv = self._build()
        v = full_alloc.alloc(16)  # 2 pages × 8 tokens
        self.assertIsNotNone(v)
        self.assertEqual(int(v.numel()), 16)
        # The output must consist of exactly 2 contiguous page-ranges.
        v_pages = sorted(set((v // self.PAGE_SIZE).tolist()))
        self.assertEqual(len(v_pages), 2)
        for p in v_pages:
            page_tokens = sorted(int(t) for t in v if t // self.PAGE_SIZE == p)
            self.assertEqual(
                page_tokens,
                [p * self.PAGE_SIZE + i for i in range(self.PAGE_SIZE)],
                "Page contents should be contiguous token ids",
            )

    # 2. alloc(N) requires N % page_size == 0.
    def test_paged_alloc_non_aligned_raises(self):
        _, full_alloc, _, _, _ = self._build()
        with self.assertRaises(AssertionError):
            full_alloc.alloc(5)  # not a multiple of 8

    # 3. v2p / p2v tables are sized by PAGES.
    def test_paged_v2p_sized_by_pages(self):
        pool, full_alloc, _, _, _ = self._build(n_full_pages=10)
        # +1 for the trailing -1 sentinel row.
        self.assertEqual(
            int(full_alloc.virtual_to_physical.numel()),
            full_alloc.num_pages + 1,
        )
        self.assertEqual(
            int(full_alloc.physical_to_virtual.numel()),
            full_alloc.num_pages + 1,
        )
        # `num_pages` should be > 1 to be a meaningful test.
        self.assertGreater(full_alloc.num_pages, 1)

    # 4. Compaction relocates a whole page at once (data follows).
    def test_paged_compaction_relocates_whole_pages(self):
        _, full_alloc, _, full_kv, _ = self._build()
        stamped = {}
        # Alloc 3 pages worth of tokens.
        a = full_alloc.alloc(self.PAGE_SIZE)  # tokens of page X
        b = full_alloc.alloc(self.PAGE_SIZE)  # tokens of page Y (middle)
        c = full_alloc.alloc(self.PAGE_SIZE)  # tokens of page Z

        # Stamp each token with a UNIQUE marker. (alloc returns unique virtuals,
        # but we want each token to be distinguishable from its in-page
        # siblings, so we use the virtual-token value itself.)
        for v in (a, b, c):
            self._stamp_tokens(full_alloc, full_kv, v)
            for t in v.tolist():
                stamped[t] = t

        # Free the MIDDLE page (token ids of `b`). This forces a compaction
        # where page `c` (boundary, grow-up) relocates into page `b`'s slot.
        full_alloc.free(b)
        for t in b.tolist():
            stamped.pop(t, None)
        # Erase markers for the freed page in the fake kv buf so the
        # invariant check doesn't see stale data.
        # (The compaction kernel moved the survivor's data; we don't manually
        # touch full_kv.buf for the freed page — the test below verifies that
        # `c`'s data followed the relocation.)
        self._check_invariants(full_alloc, full_kv, stamped)
        # `a` and `c` pages must still be live.
        for t in a.tolist():
            v_page = t // self.PAGE_SIZE
            self.assertNotEqual(int(full_alloc.virtual_to_physical[v_page].item()), -1)
        for t in c.tolist():
            v_page = t // self.PAGE_SIZE
            self.assertNotEqual(int(full_alloc.virtual_to_physical[v_page].item()), -1)

    # 5. free() recovers pages via unique(// page_size) — matches upstream.
    def test_paged_free_unique_by_page(self):
        _, full_alloc, _, full_kv, _ = self._build()
        a = full_alloc.alloc(self.PAGE_SIZE * 2)  # 2 pages = 2*PS tokens
        allocated_count_before = full_alloc.allocated_count()
        # `allocated_count()` returns TOKENS.
        self.assertEqual(allocated_count_before, 2 * self.PAGE_SIZE)
        # Internal page count.
        self.assertEqual(full_alloc._allocated_pages(), 2)
        # Free a SUBSET of tokens — but covering all tokens of both pages.
        # (Matches the upstream contract: caller passes coherent ranges.)
        full_alloc.free(a)
        self.assertEqual(full_alloc.allocated_count(), 0)
        self.assertEqual(full_alloc._allocated_pages(), 0)

    # 6. take_physical overflow check (grow-up direction).
    def test_paged_take_physical_overflow_check(self):
        _, full_alloc, _, _, _ = self._build(n_full_pages=4)
        # Try to take more pages than the buffer can hold; should return None.
        # First, fill normally up to the available_size, then over-alloc by 1.
        avail = full_alloc.available_size()
        n_pages = avail // self.PAGE_SIZE
        result = full_alloc.take_physical(n_pages * self.PAGE_SIZE)
        self.assertIsNotNone(result)
        # Now one more page would overflow.
        overflow = full_alloc.take_physical(self.PAGE_SIZE)
        self.assertIsNone(overflow, "Overflow should return None, not crash")

    # 7. SWA composite joint byte-budget in page units.
    def test_paged_swa_joint_byte_budget(self):
        from sglang.srt.mem_cache.multi_ended_allocator import (
            UnifiedSWATokenToKVPoolAllocator,
        )

        full_spec = MHASubPoolSpec(
            name="full",
            layer_num=2,
            head_num=2,
            head_dim=4,
            store_dtype=torch.float16,
            grow_direction="up",
        )
        swa_spec = MHASubPoolSpec(
            name="swa",
            layer_num=2,
            head_num=2,
            head_dim=4,
            store_dtype=torch.float16,
            grow_direction="down",
        )
        n_full_pages, n_swa_pages = 8, 8
        total = (
            n_full_pages * self.PAGE_SIZE * full_spec.entry_bytes()
            + n_swa_pages * self.PAGE_SIZE * swa_spec.entry_bytes()
        )
        pool = UnifiedKVPool(
            total_bytes=total,
            sub_pool_specs=[full_spec, swa_spec],
            device=_DEV,
            enable_memory_saver=False,
        )
        kvcache = _FakeUnifiedSWAKVPool(pool)
        allocator = UnifiedSWATokenToKVPoolAllocator(
            unified_buffer=pool,
            kvcache=kvcache,
            device=_DEV,
            full_max_total_num_tokens=n_full_pages * self.PAGE_SIZE,
            swa_max_total_num_tokens=n_swa_pages * self.PAGE_SIZE,
            page_size=self.PAGE_SIZE,
            need_sort=False,
            forward_stream=None,
        )
        # available_size() returns TOKENS. The joint byte-budget at page
        # granularity uses `entry_sum_per_page = entry_full_per_page +
        # entry_swa_per_page`. Pre-check:
        fa = allocator.full_attn_allocator
        sa = allocator.swa_attn_allocator
        entry_sum_pp = fa.entry_bytes_per_page + sa.entry_bytes_per_page
        gap = sa._byte_low_frontier() - fa._byte_high_frontier()
        expected_pages_by_bytes = gap // entry_sum_pp
        expected = (
            min(
                expected_pages_by_bytes,
                fa.num_pages - fa.min_page_index,
                sa.num_pages - sa.min_page_index,
            )
            * self.PAGE_SIZE
        )
        self.assertEqual(allocator.available_size(), expected)
        # And it's strictly less than min(fa.available_size, sa.available_size)
        # (since the joint cost is heavier than either single-side cost).
        self.assertLessEqual(
            allocator.available_size(),
            min(fa.available_size(), sa.available_size()),
        )

    # 9. REGRESSION: alloc_extend must bind v2p / p2v on
    # this allocator. Without binding, `virtual_to_physical[virt_page]`
    # stays -1 and `translate_kv_loc(virt_token)` returns negative token
    # ids → CUDA OOB in the Triton attention kernel.
    def test_paged_alloc_extend_binds_v2p_p2v(self):
        from sglang.srt.mem_cache import multi_ended_allocator as mea_mod

        _, full_alloc, _, _, _ = self._build()
        PS = self.PAGE_SIZE
        free_before = full_alloc.free_virtual_ids.clone()
        watermark_before = full_alloc.watermark_physical
        allocated_count_before = full_alloc.allocated_count()

        # Stub the kernel — we only need to verify the BINDING contract.
        # (Driving the real Triton kernel needs a GPU; the contract we're
        # checking is that the v2p/p2v tables get updated regardless of
        # what the kernel writes into out_indices.)
        original_kernel = mea_mod.alloc_extend_kernel

        class _NoOpKernelGrid:
            def __getitem__(self, _grid):
                return self

            def __call__(self, *a, **kw):
                pass

        mea_mod.alloc_extend_kernel = _NoOpKernelGrid()
        try:
            # bs=1, prefix=0, seq=2 pages worth, so num_new_pages=2.
            prefix_lens = torch.tensor([0], dtype=torch.int64, device=_DEV)
            prefix_lens_cpu = torch.tensor([0], dtype=torch.int64)
            seq_lens = torch.tensor([2 * PS], dtype=torch.int64, device=_DEV)
            seq_lens_cpu = torch.tensor([2 * PS], dtype=torch.int64)
            last_loc = torch.tensor([-1], dtype=torch.int64, device=_DEV)

            out = full_alloc.alloc_extend(
                prefix_lens,
                prefix_lens_cpu,
                seq_lens,
                seq_lens_cpu,
                last_loc,
                2 * PS,
                num_new_pages=2,
            )
        finally:
            mea_mod.alloc_extend_kernel = original_kernel

        self.assertIsNotNone(out)
        # The two virtual pages consumed from the front of free_virtual_ids
        # must now be BOUND in v2p_page (not -1).
        consumed_pages = free_before[:2]
        v2p_values = full_alloc.virtual_to_physical[consumed_pages]
        for v_page, p_page in zip(consumed_pages.tolist(), v2p_values.tolist()):
            self.assertNotEqual(
                p_page,
                -1,
                f"REGRESSION: virtual page {v_page} not bound after "
                f"alloc_extend (translate_kv_loc would return negative)",
            )
        # And p2v_page must round-trip.
        for v_page, p_page in zip(consumed_pages.tolist(), v2p_values.tolist()):
            self.assertEqual(int(full_alloc.physical_to_virtual[p_page].item()), v_page)
        # Watermark must have advanced by 2 pages.
        # `allocated_count()` returns TOKENS, so it
        # advances by 2 * PAGE_SIZE; `_allocated_pages()` is the page count.
        self.assertEqual(
            full_alloc.allocated_count(),
            allocated_count_before + 2 * PS,
        )
        self.assertEqual(
            full_alloc._allocated_pages(),
            (allocated_count_before // PS) + 2,
        )
        if full_alloc.grow_direction == "up":
            self.assertEqual(full_alloc.watermark_physical, watermark_before + 2)
        else:
            self.assertEqual(full_alloc.watermark_physical, watermark_before - 2)
        # Free-list must have shrunk by 2.
        self.assertEqual(
            int(full_alloc.free_virtual_ids.numel()),
            int(free_before.numel()) - 2,
        )

    # 10. REGRESSION: alloc_decode must bind v2p / p2v on
    # this allocator when num_new_pages > 0. Most decode steps reuse the
    # prefix's tail page (num_new_pages == 0), but the page-wrapping case
    # must update tables.
    def test_paged_alloc_decode_binds_v2p_p2v_on_page_wrap(self):
        from sglang.srt.mem_cache import multi_ended_allocator as mea_mod

        _, full_alloc, _, _, _ = self._build()
        PS = self.PAGE_SIZE
        # Pre-allocate ~1 page so an arbitrary `seq_len % page_size == 1`
        # decode step triggers a new-page consumption.
        v = full_alloc.alloc(PS)
        self.assertIsNotNone(v)
        free_before = full_alloc.free_virtual_ids.clone()
        watermark_before = full_alloc.watermark_physical
        allocated_count_before = full_alloc.allocated_count()

        # Build a decode that wraps to a new page: seq_len % page_size == 1
        # (one req that just stepped past a page boundary). The kernel will
        # consume 1 new page from `free_virtual_ids[0]`.
        seq_lens = torch.tensor([PS + 1], dtype=torch.int64, device=_DEV)
        seq_lens_cpu = torch.tensor([PS + 1], dtype=torch.int64)
        last_loc = torch.tensor(
            # last token of page-N at offset page_size-1.
            [int(v[-1].item())],
            dtype=torch.int64,
            device=_DEV,
        )

        original_kernel = mea_mod.alloc_decode_kernel

        class _NoOpKernelGrid:
            def __getitem__(self, _grid):
                return self

            def __call__(self, *a, **kw):
                pass

        mea_mod.alloc_decode_kernel = _NoOpKernelGrid()
        try:
            out = full_alloc.alloc_decode(seq_lens, seq_lens_cpu, last_loc)
        finally:
            mea_mod.alloc_decode_kernel = original_kernel

        self.assertIsNotNone(out)
        # 1 virtual page consumed from the head of free_virtual_ids.
        consumed_page = int(free_before[0].item())
        # v2p_page must now map to a valid physical page (not -1).
        p_page = int(full_alloc.virtual_to_physical[consumed_page].item())
        self.assertNotEqual(
            p_page,
            -1,
            f"REGRESSION: virtual page {consumed_page} not bound after "
            f"alloc_decode (translate_kv_loc would return negative)",
        )
        # p2v round-trip.
        self.assertEqual(
            int(full_alloc.physical_to_virtual[p_page].item()), consumed_page
        )
        # Watermark must have advanced by 1 page.
        # `allocated_count()` returns TOKENS (advance by PAGE_SIZE);
        # `_allocated_pages()` is the page count.
        self.assertEqual(
            full_alloc.allocated_count(),
            allocated_count_before + PS,
        )
        self.assertEqual(
            full_alloc._allocated_pages(),
            (allocated_count_before // PS) + 1,
        )
        if full_alloc.grow_direction == "up":
            self.assertEqual(full_alloc.watermark_physical, watermark_before + 1)
        else:
            self.assertEqual(full_alloc.watermark_physical, watermark_before - 1)
        # Free-list must have shrunk by 1.
        self.assertEqual(
            int(full_alloc.free_virtual_ids.numel()),
            int(free_before.numel()) - 1,
        )

    # 11. REGRESSION: alloc_decode with num_new_pages == 0
    # (the common case — the decode token reuses the prefix's tail page)
    # must NOT advance the watermark and NOT touch v2p / p2v.
    def test_paged_alloc_decode_no_op_when_no_new_page(self):
        from sglang.srt.mem_cache import multi_ended_allocator as mea_mod

        _, full_alloc, _, _, _ = self._build()
        PS = self.PAGE_SIZE
        # Pre-allocate 2 pages worth. We'll simulate a decode where seq_len
        # advances WITHIN the existing tail page (no new page consumed).
        v = full_alloc.alloc(PS)
        free_before = full_alloc.free_virtual_ids.clone()
        watermark_before = full_alloc.watermark_physical
        allocated_count_before = full_alloc.allocated_count()

        # seq_len = PS - 1 (just inside the prefix page), pre-prefix-len = PS - 2.
        # `(seq_lens % page_size == 1)` is FALSE here, so num_new_pages == 0.
        seq_lens = torch.tensor([PS - 1], dtype=torch.int64, device=_DEV)
        seq_lens_cpu = torch.tensor([PS - 1], dtype=torch.int64)
        last_loc = torch.tensor(
            [int(v[PS - 2].item())],
            dtype=torch.int64,
            device=_DEV,
        )

        original_kernel = mea_mod.alloc_decode_kernel

        class _NoOpKernelGrid:
            def __getitem__(self, _grid):
                return self

            def __call__(self, *a, **kw):
                pass

        mea_mod.alloc_decode_kernel = _NoOpKernelGrid()
        try:
            out = full_alloc.alloc_decode(seq_lens, seq_lens_cpu, last_loc)
        finally:
            mea_mod.alloc_decode_kernel = original_kernel

        self.assertIsNotNone(out)
        # Nothing should have moved — no new page consumed.
        self.assertEqual(full_alloc.watermark_physical, watermark_before)
        self.assertEqual(full_alloc.allocated_count(), allocated_count_before)
        self.assertEqual(
            int(full_alloc.free_virtual_ids.numel()),
            int(free_before.numel()),
        )

    # 12. translate_kv_loc preserves token-level identity end-to-end.
    def test_paged_translate_kv_loc_token_round_trip(self):
        _, full_alloc, _, _, _ = self._build()
        v = full_alloc.alloc(self.PAGE_SIZE * 2)
        # Build the composite-style translation manually: virt_page * ps + offset.
        ps = self.PAGE_SIZE
        virt_pages = v // ps
        offsets = v % ps
        phys_pages = full_alloc.virtual_to_physical[virt_pages]
        phys_tokens = phys_pages * ps + offsets
        # `phys_tokens` should be a coherent set of two contiguous PAGES.
        phys_pages_unique = sorted(set(phys_pages.tolist()))
        self.assertEqual(len(phys_pages_unique), 2)
        # Within each page the tokens go through offsets 0..7 in order.
        for p in phys_pages_unique:
            page_phys = sorted(
                int(t)
                for i, t in enumerate(phys_tokens.tolist())
                if int(phys_pages[i].item()) == p
            )
            self.assertEqual(
                page_phys,
                [p * ps + i for i in range(ps)],
            )

    # REGRESSION: `translate_kv_loc(virt, out=buf)` must work
    # under page_size > 1 — the page-math branch writes via
    # `index_select(out=out)` + in-place `mul_` / `add_` and must match the
    # no-`out=` form byte-for-byte. Tests the actual page-math path of the
    # base-class implementation.
    def test_paged_translate_kv_loc_with_out(self):
        _, full_alloc, _, _, _ = self._build()
        ps = self.PAGE_SIZE
        v = full_alloc.alloc(2 * ps)
        self.assertIsNotNone(v)
        # Compare with-out vs no-out.
        buf = torch.empty(v.shape, dtype=torch.int64, device=_DEV)
        ptr_before = buf.data_ptr()
        with_out = full_alloc.translate_kv_loc(v, out=buf)
        no_out = full_alloc.translate_kv_loc(v)
        self.assertIs(with_out, buf, "must return the out= buffer")
        self.assertEqual(
            buf.data_ptr(), ptr_before, "out= buffer's data_ptr must be stable"
        )
        # Page-math correctness: result equals virt_page * ps + offset
        # against the real v2p table.
        virt_pages = v // ps
        offsets = v % ps
        phys_pages = full_alloc.virtual_to_physical[virt_pages]
        expected = phys_pages * ps + offsets
        self.assertTrue(bool((buf == expected).all().item()))
        self.assertTrue(bool((with_out == no_out).all().item()))

    # REGRESSION: the in-place aliasing form
    # `translate_kv_loc(buf, out=buf)` must work at page_size > 1 too. The
    # page-math branch computes `virt_pages` and `offsets` BEFORE writing
    # into `out`, so those fresh tensors capture the pre-mutation values of
    # virt_tokens. The final result must equal `phys_page * ps + offset`
    # against the original input.
    def test_paged_translate_kv_loc_with_out_aliasing_input(self):
        _, full_alloc, _, _, _ = self._build()
        ps = self.PAGE_SIZE
        v_orig = full_alloc.alloc(2 * ps)
        self.assertIsNotNone(v_orig)
        # Expected (no-out form) computed BEFORE mutating buf.
        expected = full_alloc.translate_kv_loc(v_orig)
        # In-place form: buf serves as both input and out.
        buf = v_orig.clone()
        ptr_before = buf.data_ptr()
        ret = full_alloc.translate_kv_loc(buf, out=buf)
        self.assertIs(ret, buf)
        self.assertEqual(buf.data_ptr(), ptr_before)
        self.assertTrue(
            bool((buf == expected).all().item()),
            "page>1 in-place result must equal no-out result",
        )

    # REGRESSION: the stale-tail scenario.
    #
    # A naive cuda-graph replay path in `triton_backend.py` would call
    # `_translate_kv_loc(kv_indices, out=kv_indices)` on the WHOLE
    # pre-allocated buffer (`self.cuda_graph_kv_indices`), even though only
    # the `kv_indptr[-1]`-length prefix was freshly written by
    # `create_flashinfer_kv_indices_triton`. Stale tail data, when fed
    # through `v2p` repeatedly across replays, eventually produced negative
    # values (via `v2p[unbound] = -1 → -1 * page_size + offset` ∈ [-ps,-1]),
    # and the NEXT translation's `// page_size` produced `-1`, which CUDA's
    # `index_select` rejects with a scatter-gather OOB device-side assert.
    #
    # The fix in `triton_backend.py` slices the translate to the valid
    # prefix `kv_indices[:kv_indptr[-1]]`. This test confirms slicing is
    # transparent to the translate — the in-place result on a contiguous
    # slice of a larger buffer matches the standalone-tensor result.
    def test_paged_translate_kv_loc_on_buffer_slice(self):
        _, full_alloc, _, _, _ = self._build()
        ps = self.PAGE_SIZE
        v = full_alloc.alloc(2 * ps)
        self.assertIsNotNone(v)
        # Simulate the cuda-graph buffer pattern: a large pre-allocated
        # buffer where only a prefix is freshly written.
        N = v.numel()
        big_buf = torch.zeros((N * 4,), dtype=torch.int64, device=_DEV)
        big_buf[:N] = v
        # Translate the valid prefix slice in-place (this is what the
        # post-fix triton_backend call does).
        slice_view = big_buf[:N]
        ptr_before = slice_view.data_ptr()
        ret = full_alloc.translate_kv_loc(slice_view, out=slice_view)
        self.assertIs(ret, slice_view)
        self.assertEqual(
            slice_view.data_ptr(),
            ptr_before,
            "slice in-place write must preserve data_ptr",
        )
        # The slice's translation must match a standalone translate of v.
        expected = full_alloc.translate_kv_loc(v)
        self.assertTrue(
            bool((slice_view == expected).all().item()),
            "slice in-place translate must equal standalone translate",
        )
        # And the tail [N:] must remain UNTOUCHED — zeros, not corrupted
        # by the translate.
        tail = big_buf[N:]
        self.assertTrue(
            bool((tail == 0).all().item()),
            "translating a slice must NOT touch the buffer tail; if this "
            "test fails, the translate is reading/writing past the slice "
            "bound — the same regression that caused a scatter-gather OOB "
            "after several replays.",
        )

    # REGRESSION: tombstone-safety clamp at page_size > 1.
    #
    # For ps > 1, `v2p_page[vpage] == -1` produces `-1 * ps + offset` for
    # output tokens — a negative value in `[-ps, -1]`. Without clamping,
    # the captured `k_buffer[result[i]]` is an illegal access. The clamp
    # must produce `>= 0` for every output token.
    def test_paged_translate_kv_loc_clamps_tombstoned_v2p(self):
        _, full_alloc, _, _, _ = self._build()
        ps = self.PAGE_SIZE
        v = full_alloc.alloc(2 * ps)
        self.assertIsNotNone(v)
        # Tombstone one page (any v2p_page entry) -> all ps tokens in that
        # page should clamp to 0 in the translate output.
        tomb_page = int((v[0] // ps).item())
        full_alloc.virtual_to_physical[tomb_page] = -1
        # No-out form.
        out = full_alloc.translate_kv_loc(v)
        self.assertTrue(
            bool((out >= 0).all().item()),
            f"paged translate_kv_loc must clamp tombstoned to >=0; got {out.tolist()}",
        )
        # The first `ps` tokens belong to the tombstoned page → all 0.
        self.assertTrue(
            bool((out[:ps] == 0).all().item()),
            "all tokens in a tombstoned page must map to slot 0 (padding sink)",
        )
        # The second page is still bound; its outputs must be > 0.
        self.assertTrue(
            bool((out[ps:] > 0).all().item()),
            "non-tombstoned pages must still translate to live physical slots",
        )

    def test_paged_translate_kv_loc_with_out_clamps_tombstoned_v2p(self):
        _, full_alloc, _, _, _ = self._build()
        ps = self.PAGE_SIZE
        v = full_alloc.alloc(2 * ps).clone()
        tomb_page = int((v[0] // ps).item())
        full_alloc.virtual_to_physical[tomb_page] = -1
        buf = torch.empty_like(v)
        ret = full_alloc.translate_kv_loc(v, out=buf)
        self.assertIs(ret, buf)
        self.assertTrue(
            bool((buf >= 0).all().item()),
            "paged out= path must clamp tombstoned entries",
        )
        self.assertTrue(bool((buf[:ps] == 0).all().item()))

    # 13. REGRESSION: `allocated_count()` MUST return
    # TOKENS, not pages — matching upstream's convention that all external
    # capacity methods report tokens. At page_size > 1, returning pages
    # here breaks the leak invariant
    # (`available + evictable + ... == total`, with all terms in tokens).
    def test_paged_allocated_count_returns_tokens(self):
        _, full_alloc, _, _, _ = self._build()
        PS = self.PAGE_SIZE
        # Idle → allocated_count == 0.
        self.assertEqual(full_alloc.allocated_count(), 0)
        # Alloc 2 pages = 2 * PS tokens.
        v = full_alloc.alloc(2 * PS)
        self.assertIsNotNone(v)
        # allocated_count() must report TOKENS (= 2 * PS), not pages (= 2).
        self.assertEqual(
            full_alloc.allocated_count(),
            2 * PS,
            "REGRESSION: allocated_count() must return TOKENS at page_size > 1",
        )
        # _allocated_pages() is the page-granular internal helper.
        self.assertEqual(full_alloc._allocated_pages(), 2)

    # 14. REGRESSION: the leak-invariant terms used by the
    # scheduler runtime checker must all be in TOKENS. Specifically the
    # dynamic self-canceling identity `size_x == full/swa_available_size() +
    # allocated_count()` must hold in token units for the SWA composite (a
    # page-count drop anywhere in the chain shows up as a spurious
    # "pool memory leak detected" crash).
    def test_paged_swa_full_available_size_in_tokens(self):
        from sglang.srt.mem_cache.multi_ended_allocator import (
            UnifiedSWATokenToKVPoolAllocator,
        )

        full_spec = MHASubPoolSpec(
            name="full",
            layer_num=2,
            head_num=2,
            head_dim=4,
            store_dtype=torch.float16,
            grow_direction="up",
        )
        swa_spec = MHASubPoolSpec(
            name="swa",
            layer_num=2,
            head_num=2,
            head_dim=4,
            store_dtype=torch.float16,
            grow_direction="down",
        )
        PS = self.PAGE_SIZE
        n_full_pages, n_swa_pages = 16, 16
        total = (
            n_full_pages * PS * full_spec.entry_bytes()
            + n_swa_pages * PS * swa_spec.entry_bytes()
        )
        pool = UnifiedKVPool(
            total_bytes=total,
            sub_pool_specs=[full_spec, swa_spec],
            device=_DEV,
            enable_memory_saver=False,
        )
        kvcache = _FakeUnifiedSWAKVPool(pool)
        full_max = n_full_pages * PS
        swa_max = n_swa_pages * PS
        allocator = UnifiedSWATokenToKVPoolAllocator(
            unified_buffer=pool,
            kvcache=kvcache,
            device=_DEV,
            full_max_total_num_tokens=full_max,
            swa_max_total_num_tokens=swa_max,
            page_size=PS,
            need_sort=False,
            forward_stream=None,
        )
        # Idle: the dynamic totals equal the available views (nothing
        # allocated), and everything is in TOKENS. The self-canceling
        # identity `size_x == available_x + allocated_x` must hold at every
        # step — the checker's leak equation collapses onto it.
        self.assertEqual(allocator.full_attn_allocator.allocated_count(), 0)
        self.assertEqual(allocator.size_full, allocator.full_available_size())
        self.assertEqual(allocator.size_swa, allocator.swa_available_size())
        idle_size_full = allocator.size_full

        # Alloc 2 pages = 2*PS tokens.
        v = allocator.alloc(2 * PS)
        self.assertIsNotNone(v)

        # allocated_count must rise by 2*PS TOKENS, not by 2 (pages), and the
        # identity must keep balancing in token units.
        self.assertEqual(
            allocator.full_attn_allocator.allocated_count(),
            2 * PS,
            "REGRESSION: allocated_count must be token-granular. A "
            "'pool memory leak detected' crash is caused by a page-count "
            "value here.",
        )
        self.assertEqual(allocator.swa_attn_allocator.allocated_count(), 2 * PS)
        self.assertEqual(
            allocator.size_full,
            allocator.full_available_size()
            + allocator.full_attn_allocator.allocated_count(),
        )
        self.assertEqual(
            allocator.size_swa,
            allocator.swa_available_size()
            + allocator.swa_attn_allocator.allocated_count(),
        )
        # Both regions of the fresh allocation came out of the shared buffer,
        # so the dynamic full total shrinks by the SWA side's byte share
        # (2*PS swa tokens priced in full-token units) — never grows.
        self.assertLessEqual(allocator.size_full, idle_size_full)

    # 15. REGRESSION: UnifiedMambaTokenToKVPoolAllocator.size
    # must be TOTAL TOKENS (available + allocated, both in tokens). At
    # page_size > 1, the earlier `available + allocated_pages` formula gave
    # `tokens + pages` which silently broke the chunk-cache Mamba log lines
    # (`#full token`, `full token usage`) and would have crashed Mamba+radix
    # if radix weren't auto-downgraded to page=1.
    def test_paged_mamba_size_in_tokens(self):
        from sglang.srt.mem_cache.multi_ended_allocator import (
            UnifiedMambaTokenToKVPoolAllocator,
        )

        # Build a minimal Mamba composite: one MHA spec for full + one
        # Mamba spec. The mamba sub-allocator always uses page_size=1, but
        # the full sub-allocator uses self.PAGE_SIZE.
        PS = self.PAGE_SIZE
        full_spec = MHASubPoolSpec(
            name="full",
            layer_num=2,
            head_num=2,
            head_dim=4,
            store_dtype=torch.float16,
            grow_direction="up",
        )
        mamba_spec = MambaSubPoolSpec(
            name="mamba",
            layer_num=2,
            conv_state_shapes=((4, 3),),
            conv_dtype=torch.float32,
            temporal_state_shape=(2, 2, 2),
            temporal_dtype=torch.float32,
            grow_direction="down",
        )
        n_full_pages, n_mamba_slots = 16, 8
        total = (
            n_full_pages * PS * full_spec.entry_bytes()
            + n_mamba_slots * mamba_spec.entry_bytes()
        )
        pool = UnifiedKVPool(
            total_bytes=total,
            sub_pool_specs=[full_spec, mamba_spec],
            device=_DEV,
            enable_memory_saver=False,
        )
        # Build a fake HybridLinearKVPool-like object with two sub-pool kv
        # caches. We only need `.full_kv_pool` and `.mamba_pool` with
        # `attach_allocator` / `move_kv_cache` stubs.
        full_kv = _FakeKVCache(pool.max_slots("full"))
        full_kv.attach_allocator = lambda allocator: None
        mamba_kv = _FakeKVCache(pool.max_slots("mamba"))
        mamba_kv.attach_allocator = lambda allocator: None
        # _copy_from_physical for the mamba sub-pool (kept un-translated).
        mamba_kv._copy_from_physical = lambda src, dst: None

        class _FakeHybridLinearKVPool:
            full_kv_pool = full_kv
            mamba_pool = mamba_kv

        allocator = UnifiedMambaTokenToKVPoolAllocator(
            unified_buffer=pool,
            kvcache=_FakeHybridLinearKVPool(),
            device=_DEV,
            page_size=PS,
            need_sort=False,
            forward_stream=None,
        )

        # Idle: size == full_available_size() (entirely in tokens).
        full_avail_before = allocator.full_attn_allocator.available_size()
        self.assertEqual(allocator.size, full_avail_before)
        # available_size == size (no allocations yet).
        self.assertEqual(allocator.available_size(), allocator.size)

        # Alloc 2 pages = 2*PS tokens on full side.
        v = allocator.alloc(2 * PS)
        self.assertIsNotNone(v)

        # size should be CONSERVED in tokens: (available + allocated_tokens)
        # stays at the initial total. (For the Mamba composite, `.size` is
        # dynamic — it shrinks as the peer consumes bytes — but at this
        # point the peer is idle so we should see `size == full_avail_before`.)
        self.assertEqual(
            allocator.full_attn_allocator.available_size()
            + allocator.full_attn_allocator.allocated_count(),
            full_avail_before,
            "REGRESSION: full.available_size() + full.allocated_count() must "
            "be conserved at TOKEN granularity (was `tokens + pages` in the "
            "buggy revision).",
        )
        # And .size matches this conserved sum.
        self.assertEqual(allocator.size, full_avail_before)

    # 16. REGRESSION: the page-math helper used by
    # `UnifiedSWAKVPool.translate_loc_from_full_to_swa`,
    # `UnifiedSWAKVPool.get_cpu_copy`, and `load_cpu_copy` must do
    # `virt_pages = loc // page_size; offsets = loc % page_size;
    # phys_tokens = v2p_page[virt_pages] * page_size + offsets`.
    #
    # A naive implementation does `v2p[loc]` directly —
    # indexing a page-granular table with token-granular ids, producing
    # wrong physical token ids and (when used as Triton kernel inputs)
    # OOB reads (the same bug class as the alloc_extend/alloc_decode
    # binding regressions above).
    #
    # We can't easily construct a real UnifiedSWAKVPool in the CPU test shim
    # (it inherits SWAKVPool which builds MHATokenToKVPool sub-pools), so
    # we exercise the static helper `_virt_tokens_to_phys_tokens` directly.
    # The instance methods in production wrap this helper, so the same
    # math is covered.
    def test_paged_pool_translate_helper_returns_physical_tokens(self):
        from sglang.srt.mem_cache.multi_ended_allocator import (
            UnifiedSWATokenToKVPoolAllocator,
        )
        from sglang.srt.mem_cache.unified_memory_pool import UnifiedSWAKVPool

        full_spec = MHASubPoolSpec(
            name="full",
            layer_num=2,
            head_num=2,
            head_dim=4,
            store_dtype=torch.float16,
            grow_direction="up",
        )
        swa_spec = MHASubPoolSpec(
            name="swa",
            layer_num=2,
            head_num=2,
            head_dim=4,
            store_dtype=torch.float16,
            grow_direction="down",
        )
        PS = self.PAGE_SIZE
        n_pages = 8
        total = (
            n_pages * PS * full_spec.entry_bytes()
            + n_pages * PS * swa_spec.entry_bytes()
        )
        pool = UnifiedKVPool(
            total_bytes=total,
            sub_pool_specs=[full_spec, swa_spec],
            device=_DEV,
            enable_memory_saver=False,
        )
        kvcache = _FakeUnifiedSWAKVPool(pool)
        allocator = UnifiedSWATokenToKVPoolAllocator(
            unified_buffer=pool,
            kvcache=kvcache,
            device=_DEV,
            full_max_total_num_tokens=n_pages * PS,
            swa_max_total_num_tokens=n_pages * PS,
            page_size=PS,
            need_sort=False,
            forward_stream=None,
        )

        # Alloc 2 pages worth of tokens — the swa allocator's v2p_page table
        # now has bindings for the consumed virtual pages.
        v_tokens = allocator.alloc(2 * PS)
        self.assertIsNotNone(v_tokens)

        # The static helper does the page math: same as the instance methods.
        swa_phys = UnifiedSWAKVPool._virt_tokens_to_phys_tokens(
            v_tokens, allocator.swa_attn_allocator
        )

        # Output must:
        #   1. Be non-negative for every input (none unbound at this point).
        #   2. Be distinct (one-to-one mapping).
        #   3. Match `swa_phys_page * page_size + offset` reconstructed directly.
        self.assertTrue(
            bool((swa_phys >= 0).all().item()),
            "REGRESSION: _virt_tokens_to_phys_tokens returned negative "
            "physical token ids (page-math fix likely reverted).",
        )
        self.assertEqual(
            int(torch.unique(swa_phys).numel()),
            int(swa_phys.numel()),
            "Physical token ids must be unique (one-to-one mapping).",
        )
        virt_pages_in = v_tokens // PS
        offsets_in = v_tokens % PS
        swa_phys_pages_direct = allocator.swa_attn_allocator.virtual_to_physical[
            virt_pages_in
        ]
        expected = swa_phys_pages_direct * PS + offsets_in
        self.assertTrue(
            bool((swa_phys == expected).all().item()),
            "REGRESSION: _virt_tokens_to_phys_tokens output must equal "
            "v2p_page[virt_pages] * page_size + offsets.",
        )

        # And the composite allocator's translate method must produce the
        # same token-granular result (same page math).
        composite_out = allocator.translate_loc_from_full_to_swa(v_tokens)
        self.assertTrue(
            bool((swa_phys.long() == composite_out.long()).all().item()),
            "REGRESSION: the UnifiedSWAKVPool helper and the composite "
            "allocator's translate_loc_from_full_to_swa must agree.",
        )


class TestLazyCompaction(CpuPoolTestCase):
    """Lazy compaction invariants and lazy-vs-eager
    equivalence harness. CPU-only (no GPU events; the conservative Phase A
    `_flush` uses `wait_stream(forward_stream)` only when `forward_stream is
    not None`, so passing `forward_stream=None` keeps it a no-op).
    """

    def _make_full(self, *, lazy: bool, n_full_slots=64, n_mamba_slots=16):
        full = _make_mha_spec("full", "up", layer_num=2)
        mamba = _make_mamba_spec("mamba", "down", layer_num=2)
        total = full.entry_bytes() * n_full_slots + mamba.entry_bytes() * n_mamba_slots
        pool = UnifiedKVPool(
            total_bytes=total,
            sub_pool_specs=[full, mamba],
            device=_DEV,
            enable_memory_saver=False,
        )
        full_kv = _FakeKVCache(pool.max_slots("full"))
        mamba_kv = _FakeKVCache(pool.max_slots("mamba"))
        full_alloc = MultiEndedAllocator(
            kvcache=full_kv,
            unified_buffer=pool,
            sub_pool_name="full",
            device=_DEV,
            is_id_owner=True,
            lazy_compaction=lazy,
        )
        mamba_alloc = MultiEndedAllocator(
            kvcache=mamba_kv,
            unified_buffer=pool,
            sub_pool_name="mamba",
            device=_DEV,
            is_id_owner=True,
            lazy_compaction=lazy,
        )
        full_alloc.bind_peer(mamba_alloc)
        mamba_alloc.bind_peer(full_alloc)
        return pool, full_alloc, full_kv

    def _stamp_kv(self, kv: _FakeKVCache, alloc: MultiEndedAllocator, tokens) -> None:
        """Write a marker into KV[phys] for each freshly-alloced virtual
        token id, so we can later check the data followed any relocation.
        """
        for v in tokens.tolist():
            p = int(alloc.virtual_to_physical[v].item())
            kv.buf[p] = int(v)

    def test_lazy_state_initialized(self):
        """Lazy allocator initializes the new state cleanly."""
        _pool, fa, _kv = self._make_full(lazy=True)
        self.assertTrue(fa.lazy_compaction)
        self.assertEqual(len(fa._free_phys_pages), 0)
        self.assertEqual(fa._pending_reuse, {})
        self.assertEqual(fa.live_page_count, 0)
        # Watermark + free virtual list start equivalent to eager.
        self.assertEqual(fa.watermark_physical, fa.min_page_index)

    def test_lazy_alloc_increments_live_page_count(self):
        _pool, fa, _kv = self._make_full(lazy=True)
        tokens = fa.alloc(8)
        self.assertIsNotNone(tokens)
        self.assertEqual(int(tokens.numel()), 8)
        self.assertEqual(fa.live_page_count, 8)
        self.assertEqual(len(fa._free_phys_pages), 0)

    def test_lazy_free_boundary_shortcut(self):
        """Boundary absorption is DEFERRED to `_flush` (the hot
        path `_free_lazy` does only a `torch.cat`, no watermark mutation).
        After `_flush`, the freed boundary page is absorbed into the
        watermark.
        """
        _pool, fa, _kv = self._make_full(lazy=True)
        a = fa.alloc(3)  # virtual tokens
        before_wm = fa.watermark_physical
        # Free the last-alloced virtual id (its physical IS the boundary).
        last = a[-1:].clone()
        fa.free(last)
        # Watermark is NOT shrunk inline; freed page is in the
        # free list.
        self.assertEqual(fa.watermark_physical, before_wm)
        self.assertEqual(len(fa._free_phys_pages), 1)
        # `live_page_count` is decremented at free time (CPU-side metadata).
        self.assertEqual(fa.live_page_count, 2)
        # `_flush` runs the complete boundary absorb → watermark shrinks
        # by 1, free list emptied.
        fa._flush(urgent=True)
        self.assertEqual(fa.watermark_physical, before_wm - 1)
        self.assertEqual(len(fa._free_phys_pages), 0)
        self.assertEqual(fa.live_page_count, 2)

    def test_lazy_free_non_boundary_pushes_hole(self):
        """Freeing a non-boundary page enters _free_phys_pages, watermark
        stays put.
        """
        _pool, fa, _kv = self._make_full(lazy=True)
        a = fa.alloc(5)
        wm_before = fa.watermark_physical
        # Free a middle id (NOT the topmost), boundary-shortcut should
        # NOT fire.
        mid = a[2:3].clone()
        fa.free(mid)
        self.assertEqual(fa.watermark_physical, wm_before)
        self.assertEqual(len(fa._free_phys_pages), 1)
        self.assertEqual(fa.live_page_count, 4)

    def test_lazy_free_inward_walk(self):
        """The inward walk (multiple contiguous holes absorbed
        into the watermark in one pass) is DEFERRED to `_flush`. After
        flush, the watermark shrinks past all contiguous-from-boundary
        holes, regardless of the order they were freed.
        """
        _pool, fa, _kv = self._make_full(lazy=True)
        a = fa.alloc(5)
        wm_before = fa.watermark_physical
        # Free a middle slot first → hole.
        fa.free(a[2:3].clone())
        # Free the topmost ids — in eager mode these would absorb inline,
        # but in lazy mode they cat onto the free list.
        fa.free(a[4:5].clone())
        fa.free(a[3:4].clone())
        # 3 entries in the free list now; watermark still at the
        # pre-free position.
        self.assertEqual(fa.watermark_physical, wm_before)
        self.assertEqual(len(fa._free_phys_pages), 3)
        # `_flush` runs the complete CPU-side boundary absorb. All 3
        # contiguous holes near the boundary get absorbed in one pass.
        fa._flush(urgent=True)
        self.assertEqual(fa.watermark_physical, wm_before - 3)
        self.assertEqual(len(fa._free_phys_pages), 0)
        self.assertEqual(fa.live_page_count, 2)

    def test_lazy_take_physical_drains_holes_first(self):
        """alloc reuses holes before extending the watermark."""
        _pool, fa, _kv = self._make_full(lazy=True)
        a = fa.alloc(5)
        # Free two non-boundary virtuals to populate _free_phys_pages.
        fa.free(a[1:2].clone())
        fa.free(a[3:4].clone())
        n_holes_before = len(fa._free_phys_pages)
        self.assertEqual(n_holes_before, 2)
        wm_before = fa.watermark_physical
        # Allocate 2 more — both should come from holes, watermark unchanged.
        a2 = fa.alloc(2)
        self.assertEqual(fa.watermark_physical, wm_before)
        self.assertEqual(len(fa._free_phys_pages), 0)
        # Allocate 1 more — must extend (no holes left).
        a3 = fa.alloc(1)
        self.assertEqual(fa.watermark_physical, wm_before + 1)
        # All three new alloc batches are non-None.
        self.assertIsNotNone(a2)
        self.assertIsNotNone(a3)

    def test_lazy_available_size_includes_holes(self):
        """available_size counts drainable holes + extension capacity."""
        _pool, fa, _kv = self._make_full(lazy=True)
        avail_initial = fa.available_size()
        a = fa.alloc(5)
        # 3 non-boundary frees → 3 holes; watermark unchanged.
        fa.free(a[0:1].clone())
        fa.free(a[1:2].clone())
        fa.free(a[2:3].clone())
        self.assertEqual(len(fa._free_phys_pages), 3)
        avail_after = fa.available_size()
        # holes (3) + remaining extension capacity == original capacity
        # adjusted by the 2 still-live tokens at the top.
        # Concretely: avail_after = avail_initial - 2 (live).
        self.assertEqual(avail_after, avail_initial - 2)

    def test_lazy_flush_compacts_holes_into_gap(self):
        """_flush(urgent=True) moves a survivor into a hole and shrinks the
        watermark, freeing bytes back into the shared gap.
        """
        _pool, fa, _kv = self._make_full(lazy=True)
        a = fa.alloc(5)
        # Stamp KV so we can assert the data followed the relocation.
        self._stamp_kv(_kv, fa, a)
        # Free a low-index hole; keep the topmost live.
        fa.free(a[1:2].clone())
        self.assertEqual(len(fa._free_phys_pages), 1)
        wm_before = fa.watermark_physical
        n_moves = fa._flush(urgent=True)
        # At least one move should have happened (topmost survivor → hole).
        self.assertGreaterEqual(n_moves, 1)
        # Watermark shrunk; hole list is now empty.
        self.assertLess(fa.watermark_physical, wm_before)
        self.assertEqual(len(fa._free_phys_pages), 0)
        # live_page_count invariant under compaction.
        self.assertEqual(fa.live_page_count, 4)

    def test_lazy_v2p_p2v_identity_after_flush(self):
        """After a flush, v2p ∘ p2v == identity on the live set."""
        _pool, fa, _kv = self._make_full(lazy=True)
        a = fa.alloc(8)
        # Free a scattered set of virtuals (not at boundary).
        fa.free(a[1:2].clone())
        fa.free(a[3:4].clone())
        fa.free(a[5:6].clone())
        fa._flush(urgent=True)
        # For every still-live virtual token, v2p ∘ p2v == identity.
        for v in a.tolist():
            p = int(fa.virtual_to_physical[v].item())
            if p == -1:
                continue  # freed
            self.assertEqual(int(fa.physical_to_virtual[p].item()), v)

    def _replay_sequence(self, ops, lazy: bool):
        """Run a given alloc/free op trace under eager OR lazy mode and
        return the final (live virtual set, alloc-time KV stamps)."""
        _pool, fa, kv = self._make_full(lazy=lazy)
        live = set()  # set of virtual ids
        kv_stamps = {}  # v -> stamp (the data we wrote at alloc time)
        next_stamp = 100
        for kind, n in ops:
            if kind == "alloc":
                tokens = fa.alloc(n)
                if tokens is None:
                    continue
                for v in tokens.tolist():
                    p = int(fa.virtual_to_physical[v].item())
                    kv.buf[p] = next_stamp
                    kv_stamps[v] = next_stamp
                    live.add(v)
                    next_stamp += 1
            elif kind == "free":
                if not live:
                    continue
                # Take up to n from live, deterministically by id.
                victims = sorted(live)[:n]
                live.difference_update(victims)
                fa.free(torch.tensor(victims, dtype=torch.int64))
        # Force final compaction on lazy so the comparison is at quiescence.
        if lazy:
            fa._flush(urgent=True)
        # Read back the data for each live id.
        live_data = {}
        for v in live:
            p = int(fa.virtual_to_physical[v].item())
            live_data[v] = int(kv.buf[p].item())
        return live, live_data, kv_stamps

    def test_lazy_vs_eager_equivalence(self):
        """Same random alloc/free sequence under lazy and eager modes must
        yield identical live virtual sets AND identical KV reads (the data
        followed any relocation).
        """
        rng = random.Random(42)
        ops = []
        for _ in range(200):
            if rng.random() < 0.6:
                ops.append(("alloc", rng.randint(1, 6)))
            else:
                ops.append(("free", rng.randint(1, 4)))
        eager_live, eager_data, eager_stamps = self._replay_sequence(ops, lazy=False)
        lazy_live, lazy_data, lazy_stamps = self._replay_sequence(ops, lazy=True)
        self.assertEqual(eager_live, lazy_live, "live virtual set diverged")
        self.assertEqual(eager_stamps, lazy_stamps, "alloc-time stamps diverged")
        # For every live id, the data we read back must match what we wrote.
        for v in eager_live:
            self.assertEqual(
                eager_data[v], eager_stamps[v], f"eager: KV[v={v}] != stamp"
            )
            self.assertEqual(lazy_data[v], lazy_stamps[v], f"lazy: KV[v={v}] != stamp")

    def test_lazy_hole_set_directional_pop(self):
        """The _HoleSet pops smallest-first for grow-up; alloc must drain
        the deepest hole first (the greedy clustering rule keeps near-
        boundary holes available for cheap absorption by compaction).
        """
        _pool, fa, _kv = self._make_full(lazy=True)
        a = fa.alloc(6)
        # Free middle and lower middles so the holes are NOT at boundary.
        fa.free(a[1:2].clone())  # frees physical at index v2p[a[1]]
        fa.free(a[3:4].clone())
        # Capture which physical pages are now in the hole set.
        # `_free_phys_pages` is a torch.Tensor; `.tolist()` returns
        # Python ints so `sorted` produces ints (not 0-dim tensors).
        holes_before = sorted(fa._free_phys_pages.tolist())
        self.assertEqual(len(holes_before), 2)
        # Alloc 1 — should drain a hole (grow-up).
        # With sort-after-merge OFF (default), the drain order
        # is FIFO over the free-list tensor — NOT "smallest first".
        # We only assert that the bound physical is ONE OF the holes.
        a2 = fa.alloc(1)
        bound_phys = int(fa.virtual_to_physical[int(a2.item())].item())
        self.assertIn(bound_phys, holes_before)

    def test_lazy_non_urgent_stops_at_write_set_blocker(self):
        """Write-race case: when the topmost survivor IS in an
        in-flight batch's write-set, non-urgent `_flush` STOPS the
        boundary walk (skipping past would shuffle holes without
        shrinking the watermark — wasted work)."""
        _pool, fa, _kv = self._make_full(lazy=True)
        a = fa.alloc(5)

        class _FakeEvent:
            def __init__(self):
                self.fired = False

            def query(self):
                return self.fired

        ev = _FakeEvent()
        fa.set_latest_forward_done_event(ev)
        # Free a non-boundary slot to create a compactable hole.
        fa.free(a[1:2].clone())
        self.assertEqual(len(fa._free_phys_pages), 1)
        # Register an in-flight forward whose write-set INCLUDES the
        # topmost survivor. We pass the virtual `out_cache_loc` TENSOR
        # (not a materialized physical set) — `_flush` translates it
        # lazily on the scheduler thread when classifying survivors.
        topmost_phys = int(fa.virtual_to_physical[int(a[-1].item())].item())
        oclv = a[-1:].clone()  # virtual id that translates to topmost_phys
        fa.set_inflight_forward(ev, oclv)
        # Non-urgent flush → case A blocker at the top → STOP.
        n_moves = fa._flush(urgent=False)
        self.assertEqual(
            n_moves,
            0,
            "non-urgent flush must STOP when the topmost survivor is in "
            "an in-flight write-set",
        )
        # State untouched: hole still present.
        self.assertEqual(len(fa._free_phys_pages), 1)
        self.assertEqual(len(fa._pending_reuse), 0)
        # Fire the event (forward done) so the write-set entry prunes; a
        # subsequent flush proceeds and releases src directly (event fired
        # → no pending reuse entry needed).
        ev.fired = True
        n_moves2 = fa._flush(urgent=False)
        self.assertGreaterEqual(n_moves2, 1)
        self.assertEqual(len(fa._pending_reuse), 0)

    def test_lazy_non_urgent_read_race_uses_pending_reuse(self):
        """Read-race case (read race, no write race): when the
        topmost survivor is NOT in any in-flight write-set, non-urgent
        `_flush` compacts immediately (read+read on KV[src] is safe) and
        pushes `(src, latest_event)` to `_pending_reuse` so a future
        alloc can't write KV[src] while iter N+1's read is still pending.
        """
        _pool, fa, _kv = self._make_full(lazy=True)
        a = fa.alloc(5)

        class _FakeEvent:
            def __init__(self):
                self.fired = False

            def query(self):
                return self.fired

        ev = _FakeEvent()
        fa.set_latest_forward_done_event(ev)
        # Free a non-boundary slot → 1 hole.
        fa.free(a[1:2].clone())
        # Empty in-flight write-set for the in-flight forward → topmost
        # survivor is NOT in the write-set → compact proceeds, src goes
        # to pending. Pass `None` (or an empty tensor) for
        # `out_cache_loc_virtual` to signal "no write race on this pool"
        # — this is the same path Mamba uses (forward writes mamba state
        # via its own kernels, not via `out_cache_loc`).
        fa.set_inflight_forward(ev, None)
        n_moves = fa._flush(urgent=False)
        self.assertGreaterEqual(n_moves, 1)
        # `_pending_reuse` has ONE entry per BATCH (keyed by
        # event), not per src. So len(_pending_reuse) == 1 here. The
        # total pages held is tracked in `_pending_reuse_pages_cpu`.
        self.assertEqual(len(fa._pending_reuse), 1)
        self.assertEqual(len(fa._pending_reuse_pages_cpu), n_moves)
        # Fire the event and drain — srcs return to availability.
        ev.fired = True
        fa._drain_pending_reuse(urgent=False)
        self.assertEqual(len(fa._pending_reuse), 0)
        self.assertEqual(len(fa._pending_reuse_pages_cpu), 0)

    @unittest.skipUnless(
        torch.cuda.is_available(), "urgent drain calls stream.wait_event"
    )
    def test_lazy_pending_reuse_urgent_wait(self):
        """An urgent drain must release pages whose event has NOT fired.

        The non-urgent path only releases entries whose event has already
        fired; the urgent path instead makes the current stream wait on the
        event and releases the pages anyway. This pins that second path.

        The event must be a REAL `torch.cuda.Event`, because the drain hands it
        to `stream.wait_event`, which rejects a duck-typed stand-in. Subclass it
        so `query()` reports not-yet-fired while the object stays a genuine
        event: recording it and then overriding `query()` is the only way to be
        both a valid `wait_event` argument and deterministically unfired.
        """
        _pool, fa, _kv = self._make_full(lazy=True)
        a = fa.alloc(4)

        class _NeverQueriesReady(torch.cuda.Event):
            def query(self):
                return False

        # Simulate a prior compaction whose event hasn't fired: one batch entry
        # in _pending_reuse, keyed by the event, valued (cpu_list, gpu_tensor).
        # The parallel CPU set must be updated too.
        p = int(fa.virtual_to_physical[int(a[2].item())].item())
        # Clear v2p/p2v so post-drain reuse is safe.
        fa.virtual_to_physical[int(a[2].item())] = -1
        fa.physical_to_virtual[p] = -1

        ev = _NeverQueriesReady()
        ev.record()
        fa._pending_reuse[ev] = ([p], torch.tensor([p], dtype=torch.int64, device=fa.device))
        fa._pending_reuse_pages_cpu.add(p)

        # Urgent: releases p despite query() == False. Any exception here is a
        # real failure -- do NOT swallow it into a skip.
        fa._drain_pending_reuse(urgent=True)

        self.assertEqual(len(fa._pending_reuse), 0)
        self.assertEqual(len(fa._pending_reuse_pages_cpu), 0)

    def test_lazy_flush_opportunistic_hook(self):
        """The public flush_opportunistic method runs the non-urgent path
        and is safe to call when no holes exist."""
        _pool, fa, _kv = self._make_full(lazy=True)
        # No holes → returns 0 moves, no-op.
        self.assertEqual(fa.flush_opportunistic(), 0)
        # Create a hole then call flush_opportunistic; latest_event=None
        # means src releases immediately.
        a = fa.alloc(3)
        fa.free(a[0:1].clone())
        moves = fa.flush_opportunistic()
        self.assertGreaterEqual(moves, 1)


class TestO3FusedAllocBind(CpuPoolTestCase):
    """Fused take_physical_pages + bind_pages.

    GPU-only tests (Triton kernel requires CUDA). Exercise the helper
    `_alloc_bind_fast_or_slow` directly: fast-path correctness, slow-
    path fallback when holes exist (Invariant B), overflow handling,
    eager vs lazy modes, grow-up vs grow-down, and page_size > 1.
    """

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("O3 fused alloc-bind kernel requires CUDA")

    def _make_full(
        self,
        *,
        lazy: bool = True,
        n_full_slots: int = 64,
        n_mamba_slots: int = 16,
        page_size: int = 1,
    ):
        full = _make_mha_spec("full", "up", layer_num=2)
        mamba = _make_mamba_spec("mamba", "down", layer_num=2)
        total = full.entry_bytes() * n_full_slots + mamba.entry_bytes() * n_mamba_slots
        pool = UnifiedKVPool(
            total_bytes=total,
            sub_pool_specs=[full, mamba],
            device="cuda",
            enable_memory_saver=False,
        )
        full_kv = _FakeKVCache(pool.max_slots("full"))
        mamba_kv = _FakeKVCache(pool.max_slots("mamba"))
        fa = MultiEndedAllocator(
            kvcache=full_kv,
            unified_buffer=pool,
            sub_pool_name="full",
            device="cuda",
            is_id_owner=True,
            page_size=page_size,
            lazy_compaction=lazy,
        )
        ma = MultiEndedAllocator(
            kvcache=mamba_kv,
            unified_buffer=pool,
            sub_pool_name="mamba",
            device="cuda",
            is_id_owner=True,
            page_size=1,  # mamba is per-request, always page=1
            lazy_compaction=lazy,
        )
        fa.bind_peer(ma)
        ma.bind_peer(fa)
        return pool, fa, full_kv

    def test_helper_exists_and_returns_tensor(self):
        """The helper `_alloc_bind_fast_or_slow` is wired and returns a
        tensor on success."""
        _pool, fa, _kv = self._make_full(lazy=True)
        v_pages = torch.tensor([10, 11, 12], dtype=torch.int64, device="cuda")
        phys = fa._alloc_bind_fast_or_slow(v_pages, 3)
        self.assertIsNotNone(phys)
        self.assertEqual(phys.shape, (3,))
        self.assertEqual(phys.dtype, torch.int64)
        self.assertEqual(phys.device.type, "cuda")

    def test_fast_path_when_no_holes(self):
        """When `_free_phys_pages` is empty, the fast path fires.
        Verifies: watermark advanced, v2p and p2v scattered correctly,
        return tensor matches the kernel's arange."""
        _pool, fa, _kv = self._make_full(lazy=True)
        # Sanity: empty holeset.
        self.assertEqual(len(fa._free_phys_pages), 0)
        wm_before = fa.watermark_physical
        # Pick virtual page ids the kernel will bind.
        v_pages = torch.tensor([20, 21, 22, 23], dtype=torch.int64, device="cuda")
        phys = fa._alloc_bind_fast_or_slow(v_pages, 4)
        # Watermark advanced by N.
        self.assertEqual(fa.watermark_physical, wm_before + 4)
        # Returned phys ids match the grow-up arange [wm_before, wm_before+4).
        expected_phys = torch.arange(
            wm_before, wm_before + 4, dtype=torch.int64, device="cuda"
        )
        self.assertTrue(torch.equal(phys, expected_phys))
        # v2p table: each virtual → its physical.
        for v, p in zip(v_pages.tolist(), expected_phys.tolist()):
            self.assertEqual(int(fa.virtual_to_physical[v].item()), p)
        # p2v table: each physical → its virtual.
        for v, p in zip(v_pages.tolist(), expected_phys.tolist()):
            self.assertEqual(int(fa.physical_to_virtual[p].item()), v)
        # live_page_count updated.
        self.assertEqual(fa.live_page_count, 4)

    def test_slow_path_when_holes_exist(self):
        """Invariant B (greedy hole reuse): when a hole exists, alloc
        drains it BEFORE extending the watermark. The fast path MUST
        NOT fire."""
        _pool, fa, _kv = self._make_full(lazy=True)
        # Build a hole by alloc-then-free-non-boundary.
        a = fa.alloc(3)
        fa.free(a[0:1].clone())  # frees a non-boundary slot → enters holeset
        self.assertEqual(len(fa._free_phys_pages), 1)
        # `_free_phys_pages` is a torch.Tensor; read the single
        # hole position via `.tolist()` (`._alive` no longer exists).
        hole_pos = int(fa._free_phys_pages.tolist()[0])
        wm_before = fa.watermark_physical
        # Alloc 1 page via the helper. Slow path should drain the hole.
        v_pages = torch.tensor([42], dtype=torch.int64, device="cuda")
        phys = fa._alloc_bind_fast_or_slow(v_pages, 1)
        # Hole drained, NOT a watermark extension.
        self.assertEqual(int(phys[0].item()), hole_pos)
        self.assertEqual(fa.watermark_physical, wm_before)
        self.assertEqual(len(fa._free_phys_pages), 0)
        # v2p/p2v updated.
        self.assertEqual(int(fa.virtual_to_physical[42].item()), hole_pos)
        self.assertEqual(int(fa.physical_to_virtual[hole_pos].item()), 42)

    def test_fast_path_in_eager_mode(self):
        """Eager mode (no lazy compaction) ALWAYS uses the fast path —
        no holes ever accumulate."""
        _pool, fa, _kv = self._make_full(lazy=False)
        self.assertFalse(fa.lazy_compaction)
        wm_before = fa.watermark_physical
        v_pages = torch.tensor([30, 31, 32], dtype=torch.int64, device="cuda")
        phys = fa._alloc_bind_fast_or_slow(v_pages, 3)
        self.assertEqual(fa.watermark_physical, wm_before + 3)
        expected_phys = torch.arange(
            wm_before, wm_before + 3, dtype=torch.int64, device="cuda"
        )
        self.assertTrue(torch.equal(phys, expected_phys))

    def test_index_space_overflow_returns_none(self):
        """When the requested allocation would overflow `num_pages`,
        the helper returns None and leaves the allocator unchanged."""
        _pool, fa, _kv = self._make_full(lazy=True, n_full_slots=8, n_mamba_slots=2)
        # Try to alloc more pages than exist.
        N = fa.num_pages + 100
        wm_before = fa.watermark_physical
        # Note: we need v_pages of size N; but only its NUMEL matters for
        # the helper. We pass a dummy tensor of the right shape.
        v_pages = torch.zeros(N, dtype=torch.int64, device="cuda")
        phys = fa._alloc_bind_fast_or_slow(v_pages, N)
        self.assertIsNone(phys)
        # Allocator state unchanged.
        self.assertEqual(fa.watermark_physical, wm_before)
        self.assertEqual(fa.live_page_count, 0)

    def test_empty_alloc_returns_empty_tensor(self):
        """N=0 returns an empty tensor (no kernel launch, no state change)."""
        _pool, fa, _kv = self._make_full(lazy=True)
        wm_before = fa.watermark_physical
        v_pages = torch.empty(0, dtype=torch.int64, device="cuda")
        phys = fa._alloc_bind_fast_or_slow(v_pages, 0)
        self.assertIsNotNone(phys)
        self.assertEqual(phys.numel(), 0)
        self.assertEqual(fa.watermark_physical, wm_before)

    def test_fast_path_equivalent_to_slow_path(self):
        """For the same input on an empty-holeset allocator, the fast
        path produces byte-identical v2p / p2v / return-tensor to the
        slow path (which is the unfused take_physical_pages + bind
        sequence). Verifies the kernel correctness against the
        reference implementation."""
        # Two identical allocators; one takes fast path, one takes slow.
        _pool_a, fa_a, _kv_a = self._make_full(lazy=True)
        _pool_b, fa_b, _kv_b = self._make_full(lazy=True)
        v_pages = torch.tensor([50, 51, 52, 53, 54], dtype=torch.int64, device="cuda")
        # Fast path on fa_a.
        phys_a = fa_a._alloc_bind_fast_or_slow(v_pages, 5)
        # Slow path on fa_b: directly call take_physical_pages + bind
        # (the unfused reference implementation).
        phys_b = fa_b.take_physical_pages(5)
        fa_b.bind(v_pages, phys_b)
        # take_physical_pages already advances live_page_count (matching the
        # fused fast path), so no manual bump here.
        # Identical return tensors.
        self.assertTrue(torch.equal(phys_a, phys_b))
        # Identical v2p / p2v after the operation.
        self.assertTrue(torch.equal(fa_a.virtual_to_physical, fa_b.virtual_to_physical))
        self.assertTrue(torch.equal(fa_a.physical_to_virtual, fa_b.physical_to_virtual))
        # Identical watermark + live_page_count.
        self.assertEqual(fa_a.watermark_physical, fa_b.watermark_physical)
        self.assertEqual(fa_a.live_page_count, fa_b.live_page_count)

    def test_eager_mode_live_page_count_not_updated(self):
        """Match `_take_physical_eager` semantics: in eager mode,
        `live_page_count` is NOT maintained (the leak-checker uses
        `allocated_count()` based on the watermark span). The helper's
        fast path must respect this — updating it would break the
        invariant that eager-mode `live_page_count == 0` always."""
        _pool, fa, _kv = self._make_full(lazy=False)
        self.assertFalse(fa.lazy_compaction)
        self.assertEqual(fa.live_page_count, 0)
        v_pages = torch.tensor([10, 11, 12], dtype=torch.int64, device="cuda")
        fa._alloc_bind_fast_or_slow(v_pages, 3)
        # live_page_count UNCHANGED (eager mode invariant).
        self.assertEqual(fa.live_page_count, 0)

    def test_lazy_mode_live_page_count_updated_on_fast_path(self):
        """Lazy mode: the fast path advances `live_page_count` by N
        (matches `take_physical`'s lazy-path bookkeeping)."""
        _pool, fa, _kv = self._make_full(lazy=True)
        self.assertEqual(fa.live_page_count, 0)
        v_pages = torch.tensor([20, 21, 22], dtype=torch.int64, device="cuda")
        fa._alloc_bind_fast_or_slow(v_pages, 3)
        self.assertEqual(fa.live_page_count, 3)
        # Another fast-path call accumulates.
        v_pages2 = torch.tensor([23, 24], dtype=torch.int64, device="cuda")
        fa._alloc_bind_fast_or_slow(v_pages2, 2)
        self.assertEqual(fa.live_page_count, 5)

    def test_lazy_mode_live_page_count_updated_on_slow_path(self):
        """Lazy mode + holes exist: the slow path advances
        `live_page_count` via the existing `take_physical_pages` call
        (which updates it internally). End state must match the fast
        path's accumulation."""
        _pool, fa, _kv = self._make_full(lazy=True)
        a = fa.alloc(3)
        # alloc(3) used the fast path (no holes at the time).
        self.assertEqual(fa.live_page_count, 3)
        # Free one non-boundary → creates a hole; subsequent alloc takes
        # the slow path.
        fa.free(a[0:1].clone())
        self.assertEqual(fa.live_page_count, 2)
        self.assertEqual(len(fa._free_phys_pages), 1)
        # Now alloc(1) takes the slow path (holes exist). Should still
        # update live_page_count back to 3.
        b = fa.alloc(1)
        self.assertIsNotNone(b)
        self.assertEqual(fa.live_page_count, 3)
        # Verify slow path actually fired: hole drained, watermark unchanged.
        self.assertEqual(len(fa._free_phys_pages), 0)

    def test_page_size_gt_1(self):
        """Helper works at page_size > 1: virtual ids and table indices
        are page-granular. Verifies kernel scatters one v2p entry per
        PAGE (not per token)."""
        _pool, fa, _kv = self._make_full(
            lazy=True, n_full_slots=64, n_mamba_slots=16, page_size=4
        )
        self.assertEqual(fa.page_size, 4)
        v_pages = torch.tensor([3, 4, 5], dtype=torch.int64, device="cuda")
        wm_before = fa.watermark_physical
        phys = fa._alloc_bind_fast_or_slow(v_pages, 3)
        self.assertIsNotNone(phys)
        self.assertEqual(phys.shape, (3,))
        # Watermark advances by N PAGES (not N tokens).
        self.assertEqual(fa.watermark_physical, wm_before + 3)
        # v2p table updated at page granularity.
        for v, p in zip(v_pages.tolist(), phys.tolist()):
            self.assertEqual(int(fa.virtual_to_physical[v].item()), p)
            self.assertEqual(int(fa.physical_to_virtual[p].item()), v)

    def test_grow_down_fast_path(self):
        """The mamba sub-pool is grow-down. Verifies fast-path arithmetic
        in the descending direction."""
        _pool, _fa, _kv = self._make_full(lazy=True)
        # Build a grow-down allocator standalone for the test.
        from sglang.srt.mem_cache.unified_memory_pool import (
            UnifiedKVPool,
        )

        full = _make_mha_spec("full", "up", layer_num=2)
        swa = _make_mha_spec("swa", "down", layer_num=2)  # grow-down
        total = (full.entry_bytes() + swa.entry_bytes()) * 32
        pool = UnifiedKVPool(
            total_bytes=total,
            sub_pool_specs=[full, swa],
            device="cuda",
            enable_memory_saver=False,
        )
        full_kv = _FakeKVCache(pool.max_slots("full"))
        swa_kv = _FakeKVCache(pool.max_slots("swa"))
        fa = MultiEndedAllocator(
            kvcache=full_kv,
            unified_buffer=pool,
            sub_pool_name="full",
            device="cuda",
            is_id_owner=True,
            lazy_compaction=True,
        )
        sa = MultiEndedAllocator(
            kvcache=swa_kv,
            unified_buffer=pool,
            sub_pool_name="swa",
            device="cuda",
            is_id_owner=False,  # non-owner, grow-down
            lazy_compaction=True,
        )
        fa.bind_peer(sa)
        sa.bind_peer(fa)
        # Grow-down: watermark starts at num_pages - 1, decreases.
        self.assertEqual(sa.grow_direction, "down")
        wm_before = sa.watermark_physical
        v_pages = torch.tensor([5, 6, 7], dtype=torch.int64, device="cuda")
        phys = sa._alloc_bind_fast_or_slow(v_pages, 3)
        # Grow-down: kernel emits ASCENDING (matches `_take_physical_eager`'s
        # `torch.arange(wm-N+1, wm+1)` output). For wm=wm_before, N=3:
        # range is [wm_before-2, wm_before-1, wm_before].
        expected = torch.tensor(
            [wm_before - 2, wm_before - 1, wm_before],
            dtype=torch.int64,
            device="cuda",
        )
        self.assertTrue(torch.equal(phys, expected))
        # Watermark decreased by N.
        self.assertEqual(sa.watermark_physical, wm_before - 3)
        # v2p / p2v consistent with the ascending mapping:
        # v_pages[0] → wm_before - 2 (lowest of the new range)
        # v_pages[2] → wm_before (highest of the new range)
        for v, p in zip(v_pages.tolist(), expected.tolist()):
            self.assertEqual(int(sa.virtual_to_physical[v].item()), p)
            self.assertEqual(int(sa.physical_to_virtual[p].item()), v)


def _make_spec_state_spec(name="spec_state", layer_num=2, num_draft_tokens=3):
    return SpecStateSubPoolSpec(
        name=name,
        layer_num=layer_num,
        num_draft_tokens=num_draft_tokens,
        conv_window_shapes=((2, 3),),
        conv_dtype=torch.bfloat16,
        ssm_state_shape=(2, 2),
        ssm_dtype=torch.float32,
    )


def _make_3pool_chain(
    *, total_bytes=6336, float_cls=SpecStateBandAllocator, float_kwargs=None
):
    """[mamba(up) | float spec-state | full(down)] chain over one buffer.
    Returns (pool, mamba_alloc, band_alloc, full_alloc, fake_kv)."""
    full = _make_mha_spec("full", "down", layer_num=2, head_num=2, head_dim=4)
    mamba = _make_mamba_spec2("mamba", "up")
    spec = _make_spec_state_spec()
    pool = UnifiedKVPool(
        total_bytes=total_bytes,
        sub_pool_specs=[full, spec, mamba],
        device=_DEV,
        enable_memory_saver=False,
    )
    fake = _FakeKVCache(max(pool.max_slots(s.name) for s in (full, mamba, spec)))
    mamba_alloc = MultiEndedAllocator(
        kvcache=fake,
        unified_buffer=pool,
        sub_pool_name="mamba",
        device=_DEV,
        is_id_owner=True,
    )
    full_alloc = MultiEndedAllocator(
        kvcache=fake,
        unified_buffer=pool,
        sub_pool_name="full",
        device=_DEV,
        is_id_owner=True,
    )
    if float_cls is SpecStateBandAllocator:
        band_alloc = SpecStateBandAllocator(
            kvcache=fake,
            unified_buffer=pool,
            sub_pool_name="spec_state",
            device=_DEV,
        )
    else:
        band_alloc = float_cls(
            kvcache=fake,
            unified_buffer=pool,
            sub_pool_name="spec_state",
            device=_DEV,
            **(float_kwargs or {}),
        )
    # End-to-end lazy/flush peers + explicit chain through the float middle.
    mamba_alloc.bind_high_peer(band_alloc)
    full_alloc.bind_low_peer(band_alloc)
    mamba_alloc.bind_peer(full_alloc)
    full_alloc.bind_peer(mamba_alloc)
    band_alloc.bind_low_peer(mamba_alloc)
    band_alloc.bind_high_peer(full_alloc)
    return pool, mamba_alloc, band_alloc, full_alloc, fake


def _make_mamba_spec2(name, grow, layer_num=2):
    return MambaSubPoolSpec(
        name=name,
        layer_num=layer_num,
        conv_state_shapes=((2, 3),),
        conv_dtype=torch.bfloat16,
        temporal_state_shape=(2, 2),
        temporal_dtype=torch.float32,
        grow_direction=grow,
    )


class TestChainFrontierBackCompat(CpuPoolTestCase):
    """N=2 gap math must be byte-identical through the chain helpers."""

    def test_two_pool_available_matches_manual_gap(self):
        full = _make_mha_spec("full", "down", layer_num=2, head_num=2, head_dim=4)
        mamba = _make_mamba_spec2("mamba", "up")
        total = 4096
        pool = UnifiedKVPool(
            total_bytes=total,
            sub_pool_specs=[full, mamba],
            device=_DEV,
            enable_memory_saver=False,
        )
        fake = _FakeKVCache(pool.max_slots("full"))
        fa = MultiEndedAllocator(
            kvcache=fake,
            unified_buffer=pool,
            sub_pool_name="full",
            device=_DEV,
            is_id_owner=True,
        )
        ma = MultiEndedAllocator(
            kvcache=fake,
            unified_buffer=pool,
            sub_pool_name="mamba",
            device=_DEV,
            is_id_owner=True,
        )
        fa.bind_peer(ma)
        ma.bind_peer(fa)
        self.assertIsNotNone(ma.alloc(5))
        self.assertIsNotNone(fa.alloc(11))
        # Old single-peer formula, computed by hand:
        for a, peer in ((ma, fa), (fa, ma)):
            if a.grow_direction == "up":
                gap = peer._byte_low_frontier() - a._byte_high_frontier()
            else:
                gap = a._byte_low_frontier() - peer._byte_high_frontier()
            pages_by_bytes = max(0, gap) // a.entry_bytes_per_page
            index_room = a.num_pages - a.min_page_index - a._allocated_pages()
            self.assertEqual(a.available_size(), min(pages_by_bytes, index_room))


class TestSpecStateBandAllocator(CpuPoolTestCase):
    def test_empty_band_is_transparent(self):
        pool, ma, band, fa, _ = _make_3pool_chain()
        self.assertTrue(band._is_frontier_transparent())
        # Ends must see straight through to each other (same as a 2-pool config).
        self.assertEqual(
            ma._chain_low_frontier_above_bytes(), fa._byte_low_frontier()
        )
        self.assertEqual(
            fa._chain_high_frontier_below_bytes(), ma._byte_high_frontier()
        )

    def test_place_band_binds_rows_and_blocks_ends(self):
        pool, ma, band, fa, _ = _make_3pool_chain()
        avail_full_before = fa.available_size()
        self.assertTrue(band.place_band(4))
        self.assertEqual(band.allocated_count(), 4)
        self.assertGreaterEqual(band.band_start_page, band.min_page_index)
        # v2p: row r -> band_start + r; standard translate API.
        rows = torch.arange(4, dtype=torch.int64)
        phys = band.translate_kv_loc(rows)
        expected = band.band_start_page + rows
        self.assertTrue(torch.equal(phys, expected))
        # Placed band is opaque: the full end's gap now stops at the band.
        self.assertEqual(
            fa._chain_high_frontier_below_bytes(), band._byte_high_frontier()
        )
        self.assertLess(fa.available_size(), avail_full_before)

    def test_default_placement_hugs_low_side(self):
        pool, ma, band, fa, _ = _make_3pool_chain()
        self.assertTrue(band.place_band(4))
        lo, _hi = band._region_bounds_pages()
        self.assertEqual(band.band_start_page, lo)
        self.assertTrue(band.place_band(4, prefer_side="high"))
        lo2, hi2 = band._region_bounds_pages()
        self.assertEqual(band.band_start_page, hi2 - 4)

    def test_unplaced_rows_translate_to_sink(self):
        pool, ma, band, fa, _ = _make_3pool_chain()
        self.assertTrue(band.place_band(2))
        rows = torch.arange(5, dtype=torch.int64)
        phys = band.translate_kv_loc(rows)
        self.assertTrue(torch.all(phys[2:] == 0))  # tombstone -> sink clamp

    def test_relocation_across_placements(self):
        pool, ma, band, fa, _ = _make_3pool_chain()
        # Park high so the low end can grow underneath.
        self.assertTrue(band.place_band(4, prefer_side="high"))
        start_a = band.band_start_page
        # Grow the low end so the region's lo bound moves up.
        self.assertIsNotNone(ma.alloc(24))
        self.assertTrue(band.place_band(3))  # default: hug the (new) low bound
        start_b = band.band_start_page
        self.assertNotEqual(start_a, start_b)
        lo, _hi = band._region_bounds_pages()
        self.assertEqual(start_b, lo)
        rows = torch.arange(3, dtype=torch.int64)
        self.assertTrue(torch.equal(band.translate_kv_loc(rows), start_b + rows))
        # Rows beyond the new bs are tombstoned even though bs shrank.
        self.assertEqual(int(band.translate_kv_loc(torch.tensor([3])).item()), 0)

    def test_place_band_failure_leaves_state_unchanged(self):
        pool, ma, band, fa, _ = _make_3pool_chain()
        self.assertTrue(band.place_band(4, prefer_side="high"))
        start = band.band_start_page
        self.assertIsNotNone(ma.alloc(24))  # shrink the region
        lo, hi = band._region_bounds_pages()
        too_big = (hi - lo) + 1
        capacity = band.num_pages - band.min_page_index
        self.assertLessEqual(too_big, capacity)  # guard must not be vacuous
        self.assertFalse(band.place_band(too_big))
        self.assertEqual(band.band_start_page, start)
        self.assertEqual(band.allocated_count(), 4)
        rows = torch.arange(4, dtype=torch.int64)
        self.assertTrue(torch.equal(band.translate_kv_loc(rows), start + rows))

    def test_pin_blocks_placement(self):
        pool, ma, band, fa, _ = _make_3pool_chain()
        self.assertTrue(band.place_band(2))
        band.pin_band()
        with self.assertRaises(AssertionError):
            band.place_band(3)
        band.unpin_band()
        self.assertTrue(band.place_band(3))

    def test_shrink_to_zero_restores_transparency(self):
        pool, ma, band, fa, _ = _make_3pool_chain()
        avail_before = fa.available_size()
        self.assertTrue(band.place_band(6))
        self.assertLess(fa.available_size(), avail_before)
        self.assertTrue(band.place_band(0))
        self.assertTrue(band._is_frontier_transparent())
        self.assertEqual(fa.available_size(), avail_before)

    def test_band_rejects_per_slot_api(self):
        pool, ma, band, fa, _ = _make_3pool_chain()
        for call in (
            lambda: band.alloc(1),
            lambda: band.free(torch.tensor([1])),
            lambda: band.take_physical(1),
            lambda: band.alloc_with_virtual(torch.tensor([1])),
        ):
            with self.assertRaises(AssertionError):
                call()


class TestFloatMultiEndedAllocator(CpuPoolTestCase):
    """General per-slot float middle pool: smart-direction alloc +
    tighter-side compaction. Unit-validated; production use arrives with the
    first cache-class middle pool."""

    def _make(self):
        return _make_3pool_chain(
            float_cls=FloatMultiEndedAllocator,
            float_kwargs=dict(is_id_owner=True, page_size=1),
        )

    def test_alloc_from_empty_repositions_to_midpoint(self):
        pool, ma, fl, fa, _ = self._make()
        lo, hi = fl._region_bounds_pages()
        v = fl.alloc(4)
        self.assertIsNotNone(v)
        expected_mid = lo + (hi - lo - 4) // 2
        self.assertEqual(fl.low_wm_page, expected_mid)
        self.assertEqual(fl.high_wm_page, expected_mid + 4)

    def test_smart_direction_extends_roomier_side(self):
        pool, ma, fl, fa, _ = self._make()
        self.assertIsNotNone(fl.alloc(4))
        # Shrink the high gap by growing the full (down) end a lot.
        self.assertIsNotNone(fa.alloc(40))
        lo, hi = fl._region_bounds_pages()
        gap_low = fl.low_wm_page - lo
        gap_high = hi - fl.high_wm_page
        self.assertGreater(gap_low, gap_high)
        low_before = fl.low_wm_page
        self.assertIsNotNone(fl.alloc(3))
        self.assertEqual(fl.low_wm_page, low_before - 3)  # extended low

    def test_v2p_p2v_mutual_inverse_and_hole_free(self):
        pool, ma, fl, fa, _ = self._make()
        vs = fl.alloc(6)
        self.assertIsNotNone(vs)
        self._check_invariants(fl)

    def _check_invariants(self, fl):
        # Every page in the live band is bound, and v2p/p2v are inverses.
        for p in range(fl.low_wm_page, fl.high_wm_page):
            v = int(fl.physical_to_virtual[p].item())
            self.assertNotEqual(v, -1)
            self.assertEqual(int(fl.virtual_to_physical[v].item()), p)

    def test_free_compacts_tighter_side_and_moves_data(self):
        pool, ma, fl, fa, fake = self._make()
        vs = fl.alloc(7)
        self.assertIsNotNone(vs)
        # Mark each physical slot with its virtual id.
        phys = fl.virtual_to_physical[vs]
        fake.buf[phys] = vs
        # Make the high side the tighter one.
        self.assertIsNotNone(fa.alloc(40))
        lo, hi = fl._region_bounds_pages()
        self.assertLessEqual(hi - fl.high_wm_page, fl.low_wm_page - lo)
        high_before = fl.high_wm_page
        free_vs = vs[torch.tensor([1, 3, 5])]
        fl.free(free_vs)
        # Tighter (high) side retreated by K=3.
        self.assertEqual(fl.high_wm_page, high_before - 3)
        self._check_invariants(fl)
        # Surviving virtual ids still find their data at their new physical.
        keep = [int(v) for v in vs.tolist() if v not in set(free_vs.tolist())]
        for v in keep:
            p = int(fl.virtual_to_physical[v].item())
            self.assertEqual(int(fake.buf[p].item()), v)

    def test_backup_restore_roundtrip(self):
        pool, ma, fl, fa, _ = self._make()
        self.assertIsNotNone(fl.alloc(4))
        snap = fl.backup_state()
        low0, high0 = fl.low_wm_page, fl.high_wm_page
        self.assertIsNotNone(fl.alloc(2))
        fl.restore_state(snap)
        self.assertEqual((fl.low_wm_page, fl.high_wm_page), (low0, high0))

    def test_available_size_is_max_gap(self):
        pool, ma, fl, fa, _ = self._make()
        self.assertIsNotNone(fl.alloc(4))
        lo, hi = fl._region_bounds_pages()
        gap_low = fl.low_wm_page - lo
        gap_high = hi - fl.high_wm_page
        index_room = fl.num_pages - fl.min_page_index - fl._allocated_pages()
        self.assertEqual(fl.available_size(), min(max(gap_low, gap_high), index_room))


class _FakeHybridLinearKVPool:
    """Duck-typed HybridLinearKVPool: full/mamba pure stores with a physical
    move marker buffer on the full side."""

    def __init__(self, full_slots: int):
        self.full_kv_pool = _FakeKVCache(full_slots)
        self.mamba_pool = _FakeKVCache(full_slots)


class TestUnifiedMambaCompositeWithBand(CpuPoolTestCase):
    def _make(self, spec_on=True, total_bytes=6336):
        from sglang.srt.mem_cache.multi_ended_allocator import (
            UnifiedMambaTokenToKVPoolAllocator,
        )

        full = _make_mha_spec("full", "down", layer_num=2, head_num=2, head_dim=4)
        mamba = _make_mamba_spec2("mamba", "up")
        specs = [full, mamba]
        if spec_on:
            specs.append(_make_spec_state_spec())
        pool = UnifiedKVPool(
            total_bytes=total_bytes,
            sub_pool_specs=specs,
            device=_DEV,
            enable_memory_saver=False,
        )
        kvcache = _FakeHybridLinearKVPool(pool.max_slots("full"))
        comp = UnifiedMambaTokenToKVPoolAllocator(
            unified_buffer=pool,
            kvcache=kvcache,
            device=_DEV,
            spec_state_sub_pool="spec_state" if spec_on else None,
        )
        return pool, comp, kvcache

    def test_band_created_and_wired(self):
        _, comp, _ = self._make()
        band = comp.spec_state_allocator
        self.assertIsNotNone(band)
        self.assertEqual(len(comp.sub_allocators), 3)
        self.assertIs(comp.sub_allocators[1], band)
        # Chain: band's low peer is the up end, high peer the down end.
        self.assertEqual(band.low_peer.grow_direction, "up")
        self.assertEqual(band.high_peer.grow_direction, "down")
        # End-pool lazy peers still point end-to-end.
        self.assertIs(comp.full_attn_allocator.peer, comp.mamba_allocator)

    def test_spec_off_has_no_band(self):
        _, comp, _ = self._make(spec_on=False)
        self.assertIsNone(comp.spec_state_allocator)
        self.assertEqual(len(comp.sub_allocators), 2)

    def test_place_pin_unpin_via_composite(self):
        _, comp, _ = self._make()
        self.assertTrue(comp.place_spec_state_band(4))
        comp.pin_spec_state_band()
        with self.assertRaises(AssertionError):
            comp.place_spec_state_band(2)
        comp.unpin_spec_state_band()
        self.assertTrue(comp.place_spec_state_band(2))

    def test_alloc_shortfall_reclaims_unpinned_band(self):
        _, comp, _ = self._make()
        # Park the band flush against the full end, eating most of its gap.
        lo, hi = comp.spec_state_allocator._region_bounds_pages()
        big = hi - lo - 2
        self.assertTrue(comp.place_spec_state_band(big, prefer_side="high"))
        want = comp.full_attn_allocator.available_size() + 8
        got = comp.alloc(want)
        # The reclaim hook must have parked the band and satisfied the alloc.
        self.assertIsNotNone(got)
        self.assertEqual(comp.spec_state_allocator.allocated_count(), 0)

    def test_pinned_band_blocks_reclaim(self):
        _, comp, _ = self._make()
        lo, hi = comp.spec_state_allocator._region_bounds_pages()
        big = hi - lo - 2
        self.assertTrue(comp.place_spec_state_band(big, prefer_side="high"))
        comp.pin_spec_state_band()
        want = comp.full_attn_allocator.available_size() + 8
        self.assertIsNone(comp.alloc(want))
        self.assertEqual(comp.spec_state_allocator.allocated_count(), big)

    def test_backup_restore_clear_are_n_way(self):
        _, comp, _ = self._make()
        self.assertTrue(comp.place_spec_state_band(3))
        snap = comp.backup_state()
        self.assertEqual(len(snap), 3)
        band = comp.spec_state_allocator
        low0, high0 = band.low_wm_page, band.high_wm_page
        self.assertTrue(comp.place_spec_state_band(5, prefer_side="high"))
        comp.restore_state(snap)
        self.assertEqual((band.low_wm_page, band.high_wm_page), (low0, high0))
        comp.clear()
        self.assertEqual(band.allocated_count(), 0)
        self.assertEqual(band.num_placed_slots, 0)

    def test_move_accept_kv_moves_full_physical(self):
        _, comp, kvcache = self._make()
        vs = comp.alloc(6)
        self.assertIsNotNone(vs)
        phys = comp.full_attn_allocator.translate_kv_loc(vs)
        kvcache.full_kv_pool.buf[phys] = vs  # mark each physical with its vid
        tgt, src = vs[:2], vs[4:6]
        comp.move_accept_kv(tgt, src)
        tgt_phys = comp.full_attn_allocator.translate_kv_loc(tgt)
        src_v = src
        self.assertTrue(
            torch.equal(kvcache.full_kv_pool.buf[tgt_phys], src_v)
        )

    def test_band_entry_charged_separately_from_mamba_slot(self):
        # The two admission charges are deliberately NOT folded together.
        #
        # A mamba slot is owed only by a request that takes a FRESH state; a
        # request whose state comes from the radix cache owes none. A band slot
        # is owed by EVERY running request, because every verify-row occupies
        # one regardless of where its mamba state came from.
        #
        # Folding the band into the mamba charge therefore under-reserves the
        # band exactly when the radix cache is hitting, and the next decode
        # step's band placement fails. Keep them separate; the admission path
        # adds them (see PrefillAdder).
        _, comp_spec, _ = self._make(spec_on=True)
        _, comp_off, _ = self._make(spec_on=False)
        full_entry = comp_off.full_attn_allocator.entry_bytes
        mamba_entry = comp_off.mamba_allocator.entry_bytes_per_page
        spec_entry = comp_spec.spec_state_allocator.entry_bytes_per_page

        def state_band_bytes(comp, fresh_slots):
            cost = comp.new_request_prefill_cost_bytes(
                extend_tokens_paged=0,
                max_new_tokens=0,
                page_size=0,
                fresh_mamba_slots=fresh_slots,
            )
            return cost.immediate_bytes  # extend/page are 0 → state + band only

        # A radix-reused state (fresh_slots=0) is charged the BAND alone —
        # and only when spec is on.
        self.assertEqual(state_band_bytes(comp_spec, 0), spec_entry)
        self.assertEqual(state_band_bytes(comp_off, 0), 0)

        # A fresh state adds exactly the mamba slot bytes ON TOP of the band —
        # the band charge never disappears into the mamba charge.
        self.assertEqual(
            state_band_bytes(comp_spec, 1), spec_entry + mamba_entry
        )
        self.assertEqual(state_band_bytes(comp_off, 1), mamba_entry)
        # Sanity: the byte charges are exact — no full-token ceil rounding.
        self.assertNotEqual(mamba_entry % full_entry, 0)

    def test_flush_opportunistic_does_not_park_band(self):
        # Flushes must not churn a live band: they can run every scheduler
        # loop, and a continuously-decoding band would park->re-place per
        # step. Parking happens ONLY on alloc shortfall
        # (_reclaim_spec_state_band, wired into every full-KV and mamba-slot
        # alloc path); there is no idle-time parking.
        _, comp, _ = self._make()
        self.assertTrue(comp.place_spec_state_band(4))
        comp.flush_opportunistic()
        self.assertEqual(comp.spec_state_allocator.num_placed_slots, 4)
        self.assertTrue(comp._reclaim_spec_state_band())
        self.assertEqual(comp.spec_state_allocator.num_placed_slots, 0)
        # A pinned band survives even the shortfall reclaim: a verify batch is
        # in flight, so its scratch must not be pulled out from under it.
        self.assertTrue(comp.place_spec_state_band(3))
        comp.pin_spec_state_band()
        self.assertFalse(comp._reclaim_spec_state_band())
        self.assertEqual(comp.spec_state_allocator.num_placed_slots, 3)


class TestBandPlacementSemantics(CpuPoolTestCase):
    def test_identity_placement_is_stable(self):
        pool, ma, band, fa, _ = _make_3pool_chain()
        self.assertTrue(band.place_band(4))
        start = band.band_start_page
        v2p_before = band.virtual_to_physical.clone()
        self.assertTrue(band.place_band(4))  # identical -> no-op fast path
        self.assertEqual(band.band_start_page, start)
        self.assertTrue(torch.equal(band.virtual_to_physical, v2p_before))

    def test_shrink_tombstones_only_the_tail(self):
        pool, ma, band, fa, _ = _make_3pool_chain()
        self.assertTrue(band.place_band(5))
        self.assertTrue(band.place_band(3))
        rows = torch.arange(5, dtype=torch.int64)
        phys = band.translate_kv_loc(rows)
        expected = band.band_start_page + torch.arange(3, dtype=torch.int64)
        self.assertTrue(torch.equal(phys[:3], expected))
        self.assertTrue(torch.all(phys[3:] == 0))  # tombstone -> sink clamp

    def test_recover_stale_pin_allows_next_placement(self):
        # Review finding #3: a pin leaked by a crash between placement and
        # commit must not wedge every subsequent decode step.
        pool, ma, band, fa, _ = _make_3pool_chain()
        self.assertTrue(band.place_band(2))
        band.pin_band()  # simulate a crash: commit's unpin never ran
        with self.assertRaises(AssertionError):
            band.place_band(3)
        band.recover_stale_pin()  # the sanctioned placement-point recovery
        self.assertTrue(band.place_band(3))

    def test_pinned_band_blocks_peer_hole_credit(self):
        # Review finding #4: schedulable credit must not see through a pinned
        # band to the far peer's drainable holes.
        pool, ma, band, fa, fake = _make_3pool_chain()
        # Rebuild the ends as LAZY allocators so hole credit is in play.
        fa_lazy = MultiEndedAllocator(
            kvcache=fake,
            unified_buffer=pool,
            sub_pool_name="full",
            device=_DEV,
            is_id_owner=True,
            lazy_compaction=True,
        )
        ma_lazy = MultiEndedAllocator(
            kvcache=fake,
            unified_buffer=pool,
            sub_pool_name="mamba",
            device=_DEV,
            is_id_owner=True,
            lazy_compaction=True,
        )
        band2 = SpecStateBandAllocator(
            kvcache=fake,
            unified_buffer=pool,
            sub_pool_name="spec_state",
            device=_DEV,
        )
        ma_lazy.bind_high_peer(band2)
        fa_lazy.bind_low_peer(band2)
        ma_lazy.bind_peer(fa_lazy)
        fa_lazy.bind_peer(ma_lazy)
        band2.bind_low_peer(ma_lazy)
        band2.bind_high_peer(fa_lazy)
        # Give the mamba peer some drainable holes.
        ma_lazy._free_phys_pages = torch.tensor([5, 6, 7], dtype=torch.int64)
        credit_no_band = fa_lazy._peer_drainable_hole_bytes()
        self.assertGreater(credit_no_band, 0)
        # A placed UNPINNED band: credit includes peer holes + band bytes
        # (both realizable: reclaim + urgent flush).
        self.assertTrue(band2.place_band(4))
        credit_unpinned = fa_lazy._peer_drainable_hole_bytes()
        self.assertEqual(
            credit_unpinned,
            credit_no_band + 4 * band2.entry_bytes_per_page,
        )
        # PINNED band: nothing beyond it is reachable.
        band2.pin_band()
        self.assertEqual(fa_lazy._peer_drainable_hole_bytes(), 0)


class TestByteAdmissionSurface(CpuPoolTestCase):
    """The composites' byte-denominated admission surface (`schedulable_free_
    bytes`, `new_request_prefill_cost_bytes`, `new_token_decode_cost_bytes`,
    `alloc_tokens_to_bytes`, `shortfall_to_evict_params`).

    Equivalence contract vs the token-equivalent converters (both APIs
    coexist until the byte admission flip): every byte cost equals the
    UN-ceiled numerator of its converter — i.e.
    `byte_cost <= token_equiv * e_full` with the difference strictly below
    one full token's bytes per ceil() the converter applied. These tests pin
    both the byte values (hand math) and that contract, so deleting the
    converters later cannot silently change admission pricing."""

    # -- mamba composite -----------------------------------------------------

    def _make_mamba_comp(self, spec_on=True, total_bytes=6336):
        from sglang.srt.mem_cache.multi_ended_allocator import (
            UnifiedMambaTokenToKVPoolAllocator,
        )

        full = _make_mha_spec("full", "down", layer_num=2, head_num=2, head_dim=4)
        mamba = _make_mamba_spec2("mamba", "up")
        specs = [full, mamba]
        if spec_on:
            specs.append(_make_spec_state_spec())
        pool = UnifiedKVPool(
            total_bytes=total_bytes,
            sub_pool_specs=specs,
            device=_DEV,
            enable_memory_saver=False,
        )
        kvcache = _FakeHybridLinearKVPool(pool.max_slots("full"))
        comp = UnifiedMambaTokenToKVPoolAllocator(
            unified_buffer=pool,
            kvcache=kvcache,
            device=_DEV,
            spec_state_sub_pool="spec_state" if spec_on else None,
        )
        return comp

    def test_mamba_prefill_cost_hand_math_and_equivalence(self):
        for spec_on in (True, False):
            comp = self._make_mamba_comp(spec_on=spec_on)
            e_f = comp.full_attn_allocator.entry_bytes
            e_m = comp.mamba_allocator.entry_bytes_per_page
            e_b = (
                comp.spec_state_allocator.entry_bytes_per_page if spec_on else 0
            )
            for fresh_slots in (0, 1, 2):
                cost = comp.new_request_prefill_cost_bytes(
                    extend_tokens_paged=96,
                    max_new_tokens=64,
                    page_size=1,
                    fresh_mamba_slots=fresh_slots,
                )
                expect_imm = (96 + 1) * e_f + fresh_slots * e_m + e_b
                self.assertEqual(cost.immediate_bytes, expect_imm)
                self.assertEqual(cost.lifetime_bytes, expect_imm + 64 * e_f)
                self.assertEqual(cost.fresh_mamba_slots, fresh_slots)

                # Equivalence vs the RETIRED token converters (formulas
                # inlined): the byte state+band cost is the un-ceiled
                # numerator of ceil(entry / e_f) per resource.
                state_band_bytes = fresh_slots * e_m + e_b
                token_equiv = fresh_slots * -(-e_m // e_f) + (
                    -(-e_b // e_f) if spec_on else 0
                )
                self.assertLessEqual(state_band_bytes, token_equiv * e_f)
                # Ceil slack: < e_f per applied ceil (one per fresh slot
                # bundle + one for the band).
                n_ceils = (1 if fresh_slots else 0) + (1 if e_b else 0)
                self.assertLess(
                    token_equiv * e_f - state_band_bytes,
                    (n_ceils + 1) * e_f,
                )

    def test_mamba_decode_cost_equivalence(self):
        comp = self._make_mamba_comp(spec_on=True)
        e_f = comp.full_attn_allocator.entry_bytes
        e_b = comp.spec_state_allocator.entry_bytes_per_page
        from sglang.srt.mem_cache.multi_ended_allocator import (
            SPEC_BAND_ALIGNMENT_MARGIN_SLOTS,
        )

        for bs, new_tokens in ((1, 1), (4, 4), (7, 16)):
            got = comp.new_token_decode_cost_bytes(num_reqs=bs, new_tokens=new_tokens)
            self.assertEqual(
                got,
                new_tokens * e_f
                + (bs + SPEC_BAND_ALIGNMENT_MARGIN_SLOTS) * e_b,
            )
            # vs the RETIRED converter (formula inlined): un-ceiled
            # numerator contract.
            band_tokens = (bs + SPEC_BAND_ALIGNMENT_MARGIN_SLOTS) * -(
                -e_b // e_f
            )
            self.assertLessEqual(got, (new_tokens + band_tokens) * e_f)

        comp_off = self._make_mamba_comp(spec_on=False)
        self.assertEqual(
            comp_off.new_token_decode_cost_bytes(num_reqs=5, new_tokens=3),
            3 * comp_off.full_attn_allocator.entry_bytes,
        )

    def test_mamba_schedulable_free_bytes_band_states(self):
        comp = self._make_mamba_comp(spec_on=True)
        up, down = comp.sub_allocators[0], comp.sub_allocators[-1]
        gap = down._byte_low_frontier() - up._byte_high_frontier()
        # Clean state: free == whole inter-end gap (no holes, no band).
        self.assertEqual(comp.schedulable_free_bytes(), gap)

        # Band placed + UNPINNED: reclaimable, so free is unchanged.
        self.assertTrue(comp.place_spec_state_band(2))
        self.assertEqual(comp.schedulable_free_bytes(), gap)

        # PINNED: the band's live bytes leave the schedulable pool.
        comp.pin_spec_state_band()
        band = comp.spec_state_allocator
        band_live = band._allocated_pages() * band.entry_bytes_per_page
        self.assertGreater(band_live, 0)
        self.assertEqual(comp.schedulable_free_bytes(), gap - band_live)
        comp.unpin_spec_state_band()
        self.assertEqual(comp.schedulable_free_bytes(), gap)

    def test_mamba_alloc_tokens_and_evict_params(self):
        comp = self._make_mamba_comp(spec_on=False)
        e_f = comp.full_attn_allocator.entry_bytes
        self.assertEqual(comp.alloc_tokens_to_bytes(17), 17 * e_f)
        p = comp.shortfall_to_evict_params(10 * e_f + 1)
        self.assertEqual(p.num_tokens, 11)  # ceil

    # -- SWA composite -------------------------------------------------------

    def _make_swa_comp(self, n_full_slots=32, n_swa_slots=16):
        from sglang.srt.mem_cache.multi_ended_allocator import (
            UnifiedSWATokenToKVPoolAllocator,
        )

        full_spec = MHASubPoolSpec(
            name="full",
            layer_num=4,
            head_num=2,
            head_dim=4,
            store_dtype=torch.float16,
            grow_direction="up",
        )
        swa_spec = MHASubPoolSpec(
            name="swa",
            layer_num=2,
            head_num=2,
            head_dim=4,
            store_dtype=torch.float16,
            grow_direction="down",
        )
        total = (
            n_full_slots * full_spec.entry_bytes()
            + n_swa_slots * swa_spec.entry_bytes()
        )
        pool = UnifiedKVPool(
            total_bytes=total,
            sub_pool_specs=[full_spec, swa_spec],
            device=_DEV,
            enable_memory_saver=False,
        )
        kvcache = _FakeUnifiedSWAKVPool(pool)
        return UnifiedSWATokenToKVPoolAllocator(
            unified_buffer=pool,
            kvcache=kvcache,
            device=_DEV,
            full_max_total_num_tokens=n_full_slots,
            swa_max_total_num_tokens=n_swa_slots,
            need_sort=False,
            forward_stream=None,
        )

    def test_swa_prefill_cost_hand_math_and_equivalence(self):
        comp = self._make_swa_comp()
        e_f = comp.full_attn_allocator.entry_bytes
        e_s = comp.swa_attn_allocator.entry_bytes
        cost = comp.new_request_prefill_cost_bytes(
            extend_tokens_paged=48,
            max_new_tokens=32,
            page_size=1,
            swa_budget_tokens=70,
        )
        self.assertEqual(cost.immediate_bytes, (48 + 1) * e_f + 70 * e_s)
        self.assertEqual(cost.lifetime_bytes, cost.immediate_bytes + 32 * e_f)
        self.assertEqual(cost.fresh_mamba_slots, 0)
        # Equivalence vs the RETIRED cross-charge (formula inlined): swa
        # bytes are the un-ceiled numerator of ceil(tokens * e_s / e_f).
        cross_charge = -(-(70 * e_s) // e_f)
        self.assertLessEqual(70 * e_s, cross_charge * e_f)
        self.assertLess(cross_charge * e_f - 70 * e_s, e_f)

    def test_swa_decode_and_alloc_costs_price_both_sides(self):
        comp = self._make_swa_comp()
        e_f = comp.full_attn_allocator.entry_bytes
        e_s = comp.swa_attn_allocator.entry_bytes
        self.assertEqual(
            comp.new_token_decode_cost_bytes(num_reqs=9, new_tokens=5),
            5 * (e_f + e_s),
        )
        self.assertEqual(comp.alloc_tokens_to_bytes(5), 5 * (e_f + e_s))
        p = comp.shortfall_to_evict_params(3 * (e_f + e_s) + 1)
        self.assertEqual(p.num_tokens, 4)
        self.assertEqual(p.swa_num_tokens, 4)

    # -- ByteAdmissionBudget end-to-end (real composite + adder stub) --------

    class _AdderStub:
        """Minimal PrefillAdder surface the budget consumes."""

        def __init__(self, tree_cache, page_size=1, new_token_ratio=1.0,
                     num_mixed=0, window=8):
            self.tree_cache = tree_cache
            self.page_size = page_size
            self.new_token_ratio = new_token_ratio
            self.num_mixed_decode_tokens = num_mixed
            self._window = window

        def ceil_paged_tokens(self, tokens):
            return -(-tokens // self.page_size) * self.page_size

        def _swa_budget_for_req(self, extend, swa_host_hit_length=0):
            return max(extend, self._window) + self.page_size

    class _TreeStub:
        def __init__(self, full_evict=0, swa_evict=0, mamba_evict=0,
                     supports_mamba=False, window=8):
            self._f, self._s, self._m = full_evict, swa_evict, mamba_evict
            self._sm = supports_mamba
            self.sliding_window_size = window

        def full_evictable_size(self):
            return self._f

        def swa_evictable_size(self):
            return self._s

        def mamba_evictable_size(self):
            return self._m

        def supports_mamba(self):
            return self._sm

        def is_chunk_cache(self):
            return False

    class _ReqStub:
        def __init__(self, max_new, out_len=0, mamba_idx=None):
            class _SP:
                pass

            self.sampling_params = _SP()
            self.sampling_params.max_new_tokens = max_new
            self.output_ids = [0] * out_len
            self.mamba_pool_idx = mamba_idx

    def test_byte_budget_end_to_end_mamba(self):
        comp = self._make_mamba_comp(spec_on=True)
        e_f = comp.full_attn_allocator.entry_bytes
        e_m = comp.mamba_allocator.entry_bytes_per_page
        e_b = comp.spec_state_allocator.entry_bytes_per_page
        from sglang.srt.mem_cache.multi_ended_allocator import (
            SPEC_BAND_ALIGNMENT_MARGIN_SLOTS,
        )

        tree = self._TreeStub(full_evict=3, mamba_evict=2, supports_mamba=True)
        adder = self._AdderStub(tree)
        budget = comp.make_admission_budget(adder)

        # Construction: band alignment margin charged to both offsets;
        # rem_mamba_slots = schedulable + mamba-evictable.
        margin = SPEC_BAND_ALIGNMENT_MARGIN_SLOTS * e_b
        self.assertEqual(budget.lifetime_offset_bytes, margin)
        self.assertEqual(budget.immediate_offset_bytes, margin)
        self.assertEqual(
            budget.rem_mamba_slots,
            comp.mamba_allocator.schedulable_available_size() + 2,
        )

        free = comp.schedulable_free_bytes()
        evict = 3 * e_f
        self.assertEqual(budget.lifetime_remaining(), free + evict - margin)

        # Running charge: est tokens x e_f + band per req.
        budget.charge_running([self._ReqStub(10, out_len=4)])
        self.assertEqual(
            budget.lifetime_offset_bytes, margin + 6 * e_f + e_b
        )

        # Admit a fresh-state request: gate, then charge tracks hand math.
        req = self._ReqStub(5, mamba_idx=None)
        cost = budget.prefill_cost(req, 4, 5)
        self.assertEqual(
            cost.lifetime_bytes, (4 + 1) * e_f + 2 * e_m + e_b + 5 * e_f
        )
        self.assertTrue(budget.fits(cost))
        slots_before = budget.rem_mamba_slots
        imm_before = budget.immediate_offset_bytes
        budget.charge(cost, 4, 5)
        self.assertEqual(budget.rem_mamba_slots, slots_before - 2)
        self.assertEqual(
            budget.immediate_offset_bytes,
            imm_before + (4 + 1) * e_f + 2 * e_m + e_b,
        )

        # Radix-reused state: no slot decrement, band still charged.
        req2 = self._ReqStub(5, mamba_idx=7)
        cost2 = budget.prefill_cost(req2, 4, 5)
        self.assertEqual(cost2.fresh_mamba_slots, 0)
        budget.charge(cost2, 4, 5)
        self.assertEqual(budget.rem_mamba_slots, slots_before - 2)

        # Chunk cap closed form: (rem - fixed) // e_f, capped by chunk budget.
        rem = budget.lifetime_remaining()
        fixed = 1 * e_f + 2 * e_m + e_b  # page + fresh state + band
        self.assertEqual(
            budget.chunk_memory_cap(10_000, req), min(10_000, (rem - fixed) // e_f)
        )

        # Exhaustion: drive the ledger past the availability.
        budget.lifetime_offset_bytes += rem + 1
        self.assertTrue(budget.exhausted())

    def test_byte_budget_end_to_end_swa(self):
        comp = self._make_swa_comp()
        e_f = comp.full_attn_allocator.entry_bytes
        e_s = comp.swa_attn_allocator.entry_bytes

        tree = self._TreeStub(full_evict=4, swa_evict=6, window=8)
        adder = self._AdderStub(tree, window=8)
        budget = comp.make_admission_budget(adder)

        self.assertIsNone(budget.rem_mamba_slots)
        free = comp.schedulable_free_bytes()
        evict = 4 * e_f + 6 * e_s  # BOTH sides' evictable bytes credited
        self.assertEqual(budget.lifetime_remaining(), free + evict)

        # Prefill cost prices both sides (window-floored swa budget).
        req = self._ReqStub(3)
        cost = budget.prefill_cost(req, 4, 3)
        swa_budget = max(4, 8) + 1  # window floor + page
        self.assertEqual(
            cost.lifetime_bytes, (4 + 1) * e_f + swa_budget * e_s + 3 * e_f
        )
        budget.charge(cost, 4, 3)
        self.assertEqual(
            budget.lifetime_offset_bytes,
            (4 + 1) * e_f + swa_budget * e_s + 3 * e_f,
        )

        # Chunk cap: regime A (v >= W) when roomy — v = rem//(e_f+e_s) - page.
        rem = budget.lifetime_remaining()
        v_a = rem // (e_f + e_s) - 1
        if v_a >= 8:
            self.assertEqual(budget.chunk_memory_cap(10_000, req), v_a)
        # Regime B (v < W): starve the ledger until below the window boundary.
        budget.lifetime_offset_bytes += rem - (8 + 1) * (e_f + e_s) // 2
        rem_b = budget.lifetime_remaining()
        v_a2 = rem_b // (e_f + e_s) - 1
        if v_a2 < 8:
            expect = (rem_b - (8 + 1) * e_s) // e_f - 1
            self.assertEqual(budget.chunk_memory_cap(10_000, req), expect)

    # -- byte-accounting self-check (idle leak check) ------------------------

    def test_verify_byte_accounting_clean_and_band_states(self):
        comp = self._make_mamba_comp(spec_on=True)
        self.assertEqual(comp.verify_byte_accounting(), [])
        # Placed band inside the gap, unpinned: still healthy.
        self.assertTrue(comp.place_spec_state_band(2))
        self.assertEqual(comp.verify_byte_accounting(), [])
        # Pinned at idle is NOT flagged — a stale pin is a self-healing
        # condition (recover_stale_pin), not a byte leak; flagging it under
        # strict-raise would crash on a recoverable state.
        comp.pin_spec_state_band()
        self.assertEqual(comp.verify_byte_accounting(), [])
        comp.unpin_spec_state_band()
        self.assertEqual(comp.verify_byte_accounting(), [])
        # Spec-off composite: no band terms at all.
        comp_off = self._make_mamba_comp(spec_on=False)
        self.assertEqual(comp_off.verify_byte_accounting(), [])

    def test_verify_byte_accounting_tolerates_exterior_pending(self):
        # An idle following an urgent absorb can leave compaction-src pages in
        # `_pending_reuse` ABOVE the watermark (outside the span). The check
        # must NOT flag that (the old span==live+holes+pending equation did).
        comp = self._make_mamba_comp(spec_on=False)
        up = comp.sub_allocators[0]
        if up.lazy_compaction:
            # Simulate an un-reaped exterior pending page: live+holes unchanged,
            # pending non-empty. The bound 0<=live<=span still holds.
            up._pending_reuse_pages_cpu.add(10_000)
            self.assertEqual(comp.verify_byte_accounting(), [])
            up._pending_reuse_pages_cpu.discard(10_000)

    def test_verify_byte_accounting_trips_on_frontier_corruption(self):
        comp = self._make_mamba_comp(spec_on=False)
        up = comp.sub_allocators[0]
        # Corrupt the up end's watermark past the down end's low frontier.
        up.watermark_physical = up.num_pages + 10
        problems = comp.verify_byte_accounting()
        self.assertTrue(problems, "corrupted frontier not detected")
        self.assertTrue(
            any("frontier" in p or "collision" in p for p in problems), problems
        )

    def test_verify_byte_accounting_swa_clean(self):
        comp = self._make_swa_comp()
        self.assertEqual(comp.verify_byte_accounting(), [])

    def test_swa_schedulable_free_bytes_and_now_view_rename(self):
        comp = self._make_swa_comp()
        fa, sa = comp.full_attn_allocator, comp.swa_attn_allocator
        gap = sa._byte_low_frontier() - fa._byte_high_frontier()
        self.assertEqual(comp.schedulable_free_bytes(), gap)
        # The public NOW view still delegates to the private 3-phase formula
        # (facade flips to schedulable-derived only with the byte admission
        # flip).
        self.assertEqual(
            comp.available_size(), comp._joint_realizable_now_tokens()
        )
        # Soundness-lemma spot check (eager, clean): NOW tokens == what the
        # byte sum affords at the joint per-token cost, up to index caps.
        e_f = fa.entry_bytes_per_page
        e_s = sa.entry_bytes_per_page
        R_f = fa.num_pages - fa.min_page_index - fa._allocated_pages()
        R_s = sa.num_pages - sa.min_page_index - sa._allocated_pages()
        expect = min(gap // (e_f + e_s), R_f, R_s) * comp.page_size
        self.assertEqual(comp.available_size(), expect)


if __name__ == "__main__":
    unittest.main()
