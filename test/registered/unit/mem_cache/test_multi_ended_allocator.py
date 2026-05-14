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

from sglang.srt.mem_cache.multi_ended_allocator import MultiEndedAllocator
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


if __name__ == "__main__":
    unittest.main()
