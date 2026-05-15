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


if __name__ == "__main__":
    unittest.main()
