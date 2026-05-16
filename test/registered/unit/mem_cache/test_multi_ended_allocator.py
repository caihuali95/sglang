"""Unit tests for the shared-memory-pool v2 core: SharedMemoryPool views and
MultiEndedAllocator (virtual<->physical slot ids + eager compaction).

CPU-only — no GPU / Triton needed (the allocator's data-copy delegates to a
fake kvcache here; the SharedMemoryPool view math is pure torch).

    python -m pytest test/registered/unit/mem_cache/test_multi_ended_allocator.py -v
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="stage-a-test-cpu")

import random
import unittest

import torch

from sglang.srt.mem_cache.multi_ended_allocator import (
    MultiEndedAllocator,
    SharedSWATokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.shared_memory_pool import (
    MambaSubPoolSpec,
    MHASubPoolSpec,
    SharedMemoryPool,
)

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


class TestSharedMemoryPoolViews(unittest.TestCase):
    def test_min_slot_index_and_disjoint_bytes(self):
        full = _make_mha_spec("full", "up", layer_num=4)
        mamba = _make_mamba_spec("mamba", "down", layer_num=2)
        entry_max = max(full.entry_bytes(), mamba.entry_bytes())
        total = full.entry_bytes() * 64 + mamba.entry_bytes() * 16
        pool = SharedMemoryPool(
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
        pool = SharedMemoryPool(
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
        # confirm they read back, and that "swa" was not disturbed.
        for lyr in range(3):
            k_full[lyr][5] = float(lyr + 1)
            v_full[lyr][5] = float(-(lyr + 1))
        for lyr in range(2):
            k_swa[lyr][7] = 99.0
        for lyr in range(3):
            self.assertTrue(torch.all(k_full[lyr][5] == float(lyr + 1)))
            self.assertTrue(torch.all(v_full[lyr][5] == float(-(lyr + 1))))
        for lyr in range(2):
            self.assertTrue(torch.all(k_swa[lyr][7] == 99.0))
        # "full" slot 5 layer-0 K must not alias "full" slot 6 layer-0 K
        self.assertFalse(torch.all(k_full[0][6] == float(1)))

    def test_mamba_view_shapes(self):
        full = _make_mha_spec("full", "up", layer_num=2)
        mamba = _make_mamba_spec("mamba", "down", layer_num=3)
        total = full.entry_bytes() * 16 + mamba.entry_bytes() * 8
        pool = SharedMemoryPool(
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


class TestMultiEndedAllocator(unittest.TestCase):
    def _build_pair(self, n_full_slots=64, n_mamba_slots=16):
        full = _make_mha_spec("full", "up", layer_num=2)
        mamba = _make_mamba_spec("mamba", "down", layer_num=2)
        total = full.entry_bytes() * n_full_slots + mamba.entry_bytes() * n_mamba_slots
        pool = SharedMemoryPool(
            total_bytes=total,
            sub_pool_specs=[full, mamba],
            device=_DEV,
            enable_memory_saver=False,
        )
        full_kv = _FakeKVCache(pool.max_slots("full"))
        mamba_kv = _FakeKVCache(pool.max_slots("mamba"))
        full_alloc = MultiEndedAllocator(
            kvcache=full_kv, shared_buffer=pool, sub_pool_name="full",
            device=_DEV, is_id_owner=True,
        )
        mamba_alloc = MultiEndedAllocator(
            kvcache=mamba_kv, shared_buffer=pool, sub_pool_name="mamba",
            device=_DEV, is_id_owner=True,
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
                (full_alloc, full_kv, live_full) if side else (mamba_alloc, mamba_kv, live_mamba)
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

    def test_double_free_raises(self):
        _, full_alloc, mamba_alloc, full_kv, mamba_kv = self._build_pair()
        v = self._alloc(full_alloc, full_kv, 3)
        self._free(full_alloc, full_kv, v)
        with self.assertRaises(AssertionError):
            full_alloc.free(v)


# ---------------------------------------------------------------------------
# Shared SWA composite — Stage 2 unit tests
# ---------------------------------------------------------------------------


class _FakeSharedSWAKVPool:
    """Minimal stand-in for `SharedSWAKVPool` that the composite allocator
    needs. Exposes the two sub-pool views (each a `_FakeKVCache` with an
    `attach_allocator` no-op) and an `attach_allocators` setter.

    CPU-only — avoids constructing a real `SharedMHATokenToKVPool` (which
    instantiates `MHATokenToKVPool` and is heavier than these tests need).
    """

    class _SubKV(_FakeKVCache):
        def __init__(self, max_slots):
            super().__init__(max_slots)
            self.allocator = None

        def attach_allocator(self, allocator):
            self.allocator = allocator

    def __init__(self, shared_pool: SharedMemoryPool):
        self.full_kv_pool = self._SubKV(shared_pool.max_slots("full"))
        self.swa_kv_pool = self._SubKV(shared_pool.max_slots("swa"))
        self._full_allocator = None
        self._swa_allocator = None

    def attach_allocators(self, *, full, swa):
        self._full_allocator = full
        self._swa_allocator = swa


class TestSharedSWATokenToKVPoolAllocator(unittest.TestCase):
    """Tests for the SWA composite — joint byte-budget, slot-conservation
    leak invariant, tombstone semantics for `free_swa`, divergent compaction
    of the two sub-pools, and the alloc-rollback path.

    These tests exercise the v1 lessons captured in
    `shared_memory_pool_design.md` and the Stage-2 plan: lessons #1 (joint
    byte-budget), #2 (slot-conservation), #3 (`schedulable_*` split), #5
    (watermark rollback)."""

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
        pool = SharedMemoryPool(
            total_bytes=total,
            sub_pool_specs=[full_spec, swa_spec],
            device=_DEV,
            enable_memory_saver=False,
        )
        kvcache = _FakeSharedSWAKVPool(pool)
        allocator = SharedSWATokenToKVPoolAllocator(
            shared_buffer=pool,
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
        live_v = [
            v for v in range(1, sub.num_virtual_ids) if int(v2p[v].item()) != -1
        ]
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
        free_full = set(int(x) for x in allocator.full_attn_allocator.free_virtual_ids.tolist())
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
        free_full = set(int(x) for x in allocator.full_attn_allocator.free_virtual_ids.tolist())
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
        free_full = set(int(x) for x in allocator.full_attn_allocator.free_virtual_ids.tolist())
        self.assertNotIn(tgt, free_full)
        # composite `free` of the same virtual still works (filters out
        # already-tombstoned on the swa side).
        full_phys = int(allocator.full_attn_allocator.virtual_to_physical[tgt].item())
        kvcache.full_kv_pool.buf[full_phys] = -1
        allocator.free(target)
        # now in free list:
        free_full = set(int(x) for x in allocator.full_attn_allocator.free_virtual_ids.tolist())
        self.assertIn(tgt, free_full)

    # 4. Compaction diverges between the two sub-pools (each runs its own).
    def test_swa_compaction_diverges_physical_layout(self):
        _, allocator, kvcache = self._build()
        a = self._alloc(allocator, kvcache, 1)
        b = self._alloc(allocator, kvcache, 1)
        c = self._alloc(allocator, kvcache, 1)
        # Snapshot swa-side physical for c BEFORE we free_swa(b).
        c_swa_before = int(
            allocator.swa_attn_allocator.virtual_to_physical[c].item()
        )
        c_full_before = int(
            allocator.full_attn_allocator.virtual_to_physical[c].item()
        )
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
        for _ in range(400):
            r = rng.random()
            if r < 0.5 or not live:  # alloc
                n = rng.randint(1, 4)
                v = self._alloc(allocator, kvcache, n)
                if v is not None:
                    live.append(("live", v))
            elif r < 0.8:  # composite free
                idx = rng.randrange(len(live))
                kind, v = live.pop(idx)
                self._free(allocator, kvcache, v)
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
            # Invariants after every op.
            self._check_sub_pool_invariants(
                allocator.full_attn_allocator, kvcache.full_kv_pool
            )
            self._check_sub_pool_invariants(
                allocator.swa_attn_allocator, kvcache.swa_kv_pool
            )
            # Slot-conservation invariant balances at all times.
            self.assertEqual(
                allocator.full_available_size(),
                allocator._full_max_total_num_tokens
                - allocator.full_attn_allocator.allocated_count(),
            )
            self.assertEqual(
                allocator.swa_available_size(),
                allocator._swa_max_total_num_tokens
                - allocator.swa_attn_allocator.allocated_count(),
            )
        # Drain.
        for _, v in live:
            self._free(allocator, kvcache, v)
        self.assertEqual(allocator.full_attn_allocator.allocated_count(), 0)
        self.assertEqual(allocator.swa_attn_allocator.allocated_count(), 0)

    # 7. Joint byte-budget pre-check (v1 lesson #1).
    def test_swa_joint_byte_budget_pre_check(self):
        # Pick sizes where the byte gap, not slot-index headroom, is the bind.
        full_spec = MHASubPoolSpec(
            name="full", layer_num=2, head_num=2, head_dim=4,
            store_dtype=torch.float16, grow_direction="up",
        )
        swa_spec = MHASubPoolSpec(
            name="swa", layer_num=2, head_num=2, head_dim=4,
            store_dtype=torch.float16, grow_direction="down",
        )
        n_full, n_swa = 10, 10
        total = n_full * full_spec.entry_bytes() + n_swa * swa_spec.entry_bytes()
        pool = SharedMemoryPool(
            total_bytes=total,
            sub_pool_specs=[full_spec, swa_spec],
            device=_DEV,
            enable_memory_saver=False,
        )
        kvcache = _FakeSharedSWAKVPool(pool)
        allocator = SharedSWATokenToKVPoolAllocator(
            shared_buffer=pool, kvcache=kvcache, device=_DEV,
            full_max_total_num_tokens=n_full, swa_max_total_num_tokens=n_swa,
            need_sort=False, forward_stream=None,
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

    # 8. Watermark rollback on partial alloc failure (v1 lesson #5).
    def test_swa_alloc_rollback_on_partial_failure(self):
        _, allocator, kvcache = self._build()
        fa = allocator.full_attn_allocator
        sa = allocator.swa_attn_allocator
        wm_before = fa.watermark_physical
        free_count_before = int(fa.free_virtual_ids.numel())
        # Monkeypatch swa.alloc_with_virtual to fail. The composite must
        # catch the AssertionError and roll back full's allocation.
        original = sa.alloc_with_virtual

        def _bomb(virtual_ids):
            raise AssertionError("synthetic alloc_with_virtual failure")

        sa.alloc_with_virtual = _bomb
        try:
            v = allocator.alloc(3)
            self.assertIsNone(v)
        finally:
            sa.alloc_with_virtual = original
        # Full side state must be restored exactly:
        self.assertEqual(fa.watermark_physical, wm_before)
        self.assertEqual(int(fa.free_virtual_ids.numel()), free_count_before)
        # And the now-unbound slot range must be clean:
        for p in range(fa.watermark_physical, fa.max_slots):
            self.assertEqual(int(fa.physical_to_virtual[p].item()), -1)


# ---------------------------------------------------------------------------
# Stage 3: page_size > 1 — paged unit tests
# ---------------------------------------------------------------------------


class TestPagedMultiEndedAllocator(unittest.TestCase):
    """Per-sub-pool paged tests for `MultiEndedAllocator(page_size=...)`.

    All tests use ``page_size = 8`` against a buffer sized for ~16 pages per
    sub-pool. Invariants are page-granular: free-list, v2p/p2v tables, and
    compaction operate on pages. The external API (alloc → token ids, free
    takes token ids) is byte-identical to Stage 1/2.
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
        pool = SharedMemoryPool(
            total_bytes=total,
            sub_pool_specs=[full_spec, swa_spec],
            device=_DEV,
            enable_memory_saver=False,
        )
        full_kv = _FakeKVCache(pool.max_slots("full"))
        swa_kv = _FakeKVCache(pool.max_slots("swa"))
        full_alloc = MultiEndedAllocator(
            kvcache=full_kv,
            shared_buffer=pool,
            sub_pool_name="full",
            device=_DEV,
            is_id_owner=True,
            page_size=self.PAGE_SIZE,
        )
        swa_alloc = MultiEndedAllocator(
            kvcache=swa_kv,
            shared_buffer=pool,
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
            v
            for v in range(1, alloc.num_virtual_pages)
            if int(v2p[v].item()) != -1
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
                alloc.num_virtual_pages,
            )
        self.assertEqual(alloc_hi - alloc_lo, len(live_v_pages))
        for p_page in range(alloc_lo, alloc_hi):
            self.assertNotEqual(
                int(p2v[p_page].item()), -1, f"hole at physical page {p_page}"
            )
        # Free virtual page ids ∪ live = [min_page_index, num_virtual_pages).
        free_set = set(int(x) for x in alloc.free_virtual_ids.tolist())
        self.assertEqual(
            free_set | set(live_v_pages),
            set(range(alloc.min_page_index, alloc.num_virtual_pages)),
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
            page_tokens = sorted(
                int(t) for t in v if t // self.PAGE_SIZE == p
            )
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
            full_alloc.num_virtual_pages + 1,
        )
        self.assertEqual(
            int(full_alloc.physical_to_virtual.numel()),
            full_alloc.num_virtual_pages + 1,
        )
        # `num_virtual_pages` should be > 1 to be a meaningful test.
        self.assertGreater(full_alloc.num_virtual_pages, 1)

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
            self.assertNotEqual(
                int(full_alloc.virtual_to_physical[v_page].item()), -1
            )
        for t in c.tolist():
            v_page = t // self.PAGE_SIZE
            self.assertNotEqual(
                int(full_alloc.virtual_to_physical[v_page].item()), -1
            )

    # 5. free() recovers pages via unique(// page_size) — matches upstream.
    def test_paged_free_unique_by_page(self):
        _, full_alloc, _, full_kv, _ = self._build()
        a = full_alloc.alloc(self.PAGE_SIZE * 2)  # 2 pages = 2*PS tokens
        allocated_count_before = full_alloc.allocated_count()
        # `allocated_count()` returns TOKENS (post-Stage-3-unit-fix).
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
            SharedSWATokenToKVPoolAllocator,
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
        pool = SharedMemoryPool(
            total_bytes=total,
            sub_pool_specs=[full_spec, swa_spec],
            device=_DEV,
            enable_memory_saver=False,
        )
        kvcache = _FakeSharedSWAKVPool(pool)
        allocator = SharedSWATokenToKVPoolAllocator(
            shared_buffer=pool,
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
                fa.num_virtual_pages - fa.min_page_index,
                sa.num_virtual_pages - sa.min_page_index,
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

    # 9. REGRESSION (eval_results_14): alloc_extend must bind v2p / p2v on
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
        for v_page, p_page in zip(
            consumed_pages.tolist(), v2p_values.tolist()
        ):
            self.assertNotEqual(
                p_page,
                -1,
                f"REGRESSION: virtual page {v_page} not bound after "
                f"alloc_extend (translate_kv_loc would return negative)",
            )
        # And p2v_page must round-trip.
        for v_page, p_page in zip(
            consumed_pages.tolist(), v2p_values.tolist()
        ):
            self.assertEqual(
                int(full_alloc.physical_to_virtual[p_page].item()), v_page
            )
        # Watermark must have advanced by 2 pages.
        # `allocated_count()` returns TOKENS (post-Stage-3-unit-fix), so it
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
            self.assertEqual(
                full_alloc.watermark_physical, watermark_before + 2
            )
        else:
            self.assertEqual(
                full_alloc.watermark_physical, watermark_before - 2
            )
        # Free-list must have shrunk by 2.
        self.assertEqual(
            int(full_alloc.free_virtual_ids.numel()),
            int(free_before.numel()) - 2,
        )

    # 10. REGRESSION (eval_results_14): alloc_decode must bind v2p / p2v on
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
            self.assertEqual(
                full_alloc.watermark_physical, watermark_before + 1
            )
        else:
            self.assertEqual(
                full_alloc.watermark_physical, watermark_before - 1
            )
        # Free-list must have shrunk by 1.
        self.assertEqual(
            int(full_alloc.free_virtual_ids.numel()),
            int(free_before.numel()) - 1,
        )

    # 11. REGRESSION (eval_results_14): alloc_decode with num_new_pages == 0
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
        self.assertEqual(
            full_alloc.watermark_physical, watermark_before
        )
        self.assertEqual(
            full_alloc.allocated_count(), allocated_count_before
        )
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

    # 13. REGRESSION (eval_results_15): `allocated_count()` MUST return
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

    # 14. REGRESSION (eval_results_15): the leak-invariant terms used by the
    # scheduler runtime checker must all be in TOKENS. Specifically
    # `full_available_size() + allocated_tokens == static_cap` must hold for
    # the SWA composite.
    def test_paged_swa_full_available_size_in_tokens(self):
        from sglang.srt.mem_cache.multi_ended_allocator import (
            SharedSWATokenToKVPoolAllocator,
        )

        full_spec = MHASubPoolSpec(
            name="full", layer_num=2, head_num=2, head_dim=4,
            store_dtype=torch.float16, grow_direction="up",
        )
        swa_spec = MHASubPoolSpec(
            name="swa", layer_num=2, head_num=2, head_dim=4,
            store_dtype=torch.float16, grow_direction="down",
        )
        PS = self.PAGE_SIZE
        n_full_pages, n_swa_pages = 16, 16
        total = (
            n_full_pages * PS * full_spec.entry_bytes()
            + n_swa_pages * PS * swa_spec.entry_bytes()
        )
        pool = SharedMemoryPool(
            total_bytes=total,
            sub_pool_specs=[full_spec, swa_spec],
            device=_DEV,
            enable_memory_saver=False,
        )
        kvcache = _FakeSharedSWAKVPool(pool)
        full_max = n_full_pages * PS
        swa_max = n_swa_pages * PS
        allocator = SharedSWATokenToKVPoolAllocator(
            shared_buffer=pool, kvcache=kvcache, device=_DEV,
            full_max_total_num_tokens=full_max,
            swa_max_total_num_tokens=swa_max,
            page_size=PS, need_sort=False, forward_stream=None,
        )
        # Idle: full_available_size == cap (in tokens).
        self.assertEqual(allocator.full_available_size(), full_max)
        self.assertEqual(allocator.swa_available_size(), swa_max)

        # Alloc 2 pages = 2*PS tokens.
        v = allocator.alloc(2 * PS)
        self.assertIsNotNone(v)

        # full_available_size must drop by 2*PS TOKENS, not by 2 (pages).
        self.assertEqual(
            allocator.full_available_size(),
            full_max - 2 * PS,
            "REGRESSION: full_available_size() must drop by token-count, "
            "not page-count. The eval_results_15 'pool memory leak detected' "
            "crash was caused by a page-count drop here.",
        )
        self.assertEqual(
            allocator.swa_available_size(),
            swa_max - 2 * PS,
        )

        # First-principles leak invariant: at this point, allocated tokens
        # are all "live" (no eviction yet). So:
        #   total = full_available_size() + allocated_tokens
        # where allocated_tokens = full_max - full_available_size().
        allocated_tokens = full_max - allocator.full_available_size()
        self.assertEqual(allocated_tokens, 2 * PS)
        self.assertEqual(
            allocated_tokens
            + allocator.full_available_size(),
            full_max,
        )

    # 15. REGRESSION (eval_results_15): SharedMambaTokenToKVPoolAllocator.size
    # must be TOTAL TOKENS (available + allocated, both in tokens). At
    # page_size > 1, the earlier `available + allocated_pages` formula gave
    # `tokens + pages` which silently broke the chunk-cache Mamba log lines
    # (`#full token`, `full token usage`) and would have crashed Mamba+radix
    # if radix weren't auto-downgraded to page=1.
    def test_paged_mamba_size_in_tokens(self):
        from sglang.srt.mem_cache.multi_ended_allocator import (
            SharedMambaTokenToKVPoolAllocator,
        )

        # Build a minimal Mamba composite: one MHA spec for full + one
        # Mamba spec. The mamba sub-allocator always uses page_size=1, but
        # the full sub-allocator uses self.PAGE_SIZE.
        PS = self.PAGE_SIZE
        full_spec = MHASubPoolSpec(
            name="full", layer_num=2, head_num=2, head_dim=4,
            store_dtype=torch.float16, grow_direction="up",
        )
        mamba_spec = MambaSubPoolSpec(
            name="mamba", layer_num=2,
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
        pool = SharedMemoryPool(
            total_bytes=total, sub_pool_specs=[full_spec, mamba_spec],
            device=_DEV, enable_memory_saver=False,
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

        allocator = SharedMambaTokenToKVPoolAllocator(
            shared_buffer=pool, kvcache=_FakeHybridLinearKVPool(),
            device=_DEV, page_size=PS, need_sort=False, forward_stream=None,
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

    # 16. REGRESSION (audit follow-up): the page-math helper used by
    # `SharedSWAKVPool.translate_loc_from_full_to_swa`,
    # `SharedSWAKVPool.get_cpu_copy`, and `load_cpu_copy` must do
    # `virt_pages = loc // page_size; offsets = loc % page_size;
    # phys_tokens = v2p_page[virt_pages] * page_size + offsets`.
    #
    # Before this fix, those three methods did `v2p[loc]` directly —
    # indexing a page-granular table with token-granular ids, producing
    # wrong physical token ids and (when used as Triton kernel inputs)
    # OOB reads. Same bug class as eval_results_14.
    #
    # We can't easily construct a real SharedSWAKVPool in the CPU test shim
    # (it inherits SWAKVPool which builds MHATokenToKVPool sub-pools), so
    # we exercise the static helper `_virt_tokens_to_phys_tokens` directly.
    # The instance methods in production wrap this helper, so the same
    # math is covered.
    def test_paged_pool_translate_helper_returns_physical_tokens(self):
        from sglang.srt.mem_cache.shared_memory_pool import SharedSWAKVPool
        from sglang.srt.mem_cache.multi_ended_allocator import (
            SharedSWATokenToKVPoolAllocator,
        )

        full_spec = MHASubPoolSpec(
            name="full", layer_num=2, head_num=2, head_dim=4,
            store_dtype=torch.float16, grow_direction="up",
        )
        swa_spec = MHASubPoolSpec(
            name="swa", layer_num=2, head_num=2, head_dim=4,
            store_dtype=torch.float16, grow_direction="down",
        )
        PS = self.PAGE_SIZE
        n_pages = 8
        total = (
            n_pages * PS * full_spec.entry_bytes()
            + n_pages * PS * swa_spec.entry_bytes()
        )
        pool = SharedMemoryPool(
            total_bytes=total, sub_pool_specs=[full_spec, swa_spec],
            device=_DEV, enable_memory_saver=False,
        )
        kvcache = _FakeSharedSWAKVPool(pool)
        allocator = SharedSWATokenToKVPoolAllocator(
            shared_buffer=pool, kvcache=kvcache, device=_DEV,
            full_max_total_num_tokens=n_pages * PS,
            swa_max_total_num_tokens=n_pages * PS,
            page_size=PS, need_sort=False, forward_stream=None,
        )

        # Alloc 2 pages worth of tokens — the swa allocator's v2p_page table
        # now has bindings for the consumed virtual pages.
        v_tokens = allocator.alloc(2 * PS)
        self.assertIsNotNone(v_tokens)

        # The static helper does the page math: same as the instance methods.
        swa_phys = SharedSWAKVPool._virt_tokens_to_phys_tokens(
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
        swa_phys_pages_direct = (
            allocator.swa_attn_allocator.virtual_to_physical[virt_pages_in]
        )
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
            "REGRESSION: the SharedSWAKVPool helper and the composite "
            "allocator's translate_loc_from_full_to_swa must agree.",
        )


if __name__ == "__main__":
    unittest.main()
