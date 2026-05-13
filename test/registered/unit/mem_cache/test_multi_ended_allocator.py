"""Unit tests for MultiEndedAllocator + SharedMemoryPool (CPU-only, no model load)."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="stage-a-test-cpu")

import unittest

import torch

from sglang.srt.mem_cache.multi_ended_allocator import MultiEndedAllocator
from sglang.srt.mem_cache.relocation_log import (
    Relocator,
    SlotBacktrack,
    SlotBacktrackBinder,
    null_backtrack_binder,
)
from sglang.srt.mem_cache.shared_memory_pool import (
    MambaSubPoolSpec,
    MHASubPoolSpec,
    SharedMemoryPool,
    SharedMHATokenToKVPool,
)


class _FakeKVCache:
    """Minimal kvcache stand-in that records relocation invocations."""

    def __init__(self, k_buffer, v_buffer):
        self.k_buffer = k_buffer
        self.v_buffer = v_buffer
        self.move_calls = []

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        # Track and do the native copy.
        self.move_calls.append((tgt_loc.cpu().tolist(), src_loc.cpu().tolist()))
        for k, v in zip(self.k_buffer, self.v_buffer):
            k[tgt_loc.long()] = k[src_loc.long()]
            v[tgt_loc.long()] = v[src_loc.long()]


def _build(
    total_bytes=4096,
    head_num=2,
    head_dim=4,
    layer_num=2,
    device="cpu",
):
    specs = [
        MHASubPoolSpec(
            name="full",
            layer_num=layer_num,
            head_num=head_num,
            head_dim=head_dim,
            store_dtype=torch.float16,
            grow_direction="up",
        ),
        MHASubPoolSpec(
            name="swa",
            layer_num=layer_num,
            head_num=head_num,
            head_dim=head_dim,
            store_dtype=torch.float16,
            grow_direction="down",
        ),
    ]
    shared = SharedMemoryPool(
        total_bytes=total_bytes,
        sub_pool_specs=specs,
        device=device,
        enable_memory_saver=False,
    )

    backtracks = {
        "full": SlotBacktrack(size_total=shared.max_slots("full")),
        "swa": SlotBacktrack(size_total=shared.max_slots("swa")),
    }
    reloc = Relocator(backtracks=backtracks, device=device)

    k_full, v_full = shared.mha_views_for("full")
    k_swa, v_swa = shared.mha_views_for("swa")

    kv_full = _FakeKVCache(k_full, v_full)
    kv_swa = _FakeKVCache(k_swa, v_swa)

    alloc_full = MultiEndedAllocator(
        kvcache=kv_full,
        shared_buffer=shared,
        sub_pool_name="full",
        relocation_log=reloc,
        device=device,
    )
    alloc_swa = MultiEndedAllocator(
        kvcache=kv_swa,
        shared_buffer=shared,
        sub_pool_name="swa",
        relocation_log=reloc,
        device=device,
    )
    alloc_full.bind_peer(alloc_swa)
    alloc_swa.bind_peer(alloc_full)
    return shared, reloc, alloc_full, alloc_swa, kv_full, kv_swa


class TestSharedMemoryPoolBasics(unittest.TestCase):
    def test_construction_and_view_shapes(self):
        shared, *_ = _build()
        k_full, v_full = shared.mha_views_for("full")
        self.assertEqual(len(k_full), 2)  # layer_num
        self.assertEqual(k_full[0].shape[1:], (2, 4))  # (head_num, head_dim)

    def test_views_alias_raw_bytes(self):
        """Writing via sub-pool A's view should be visible at the same byte
        offset regardless of which sub-pool reads (they share physical memory)."""
        shared, *_ = _build()
        k_full, _ = shared.mha_views_for("full")
        # Write a known value at full-pool slot 1.
        k_full[0][1].fill_(42.0)
        # Read raw bytes: full-pool slot 1 is at byte entry_bytes (= 2*layer_num*row_bytes
        # = 2*2*2*4*2 = 64 bytes).
        spec = shared.spec("full")
        entry = spec.entry_bytes()
        raw = shared._raw
        # Layer 0 K row starts at byte 0 within the slot entry.
        start = 1 * entry + 0
        # 2 * 4 * 2 = 16 bytes of data
        raw_bytes = raw[start : start + spec.k_row_bytes()]
        # Interpret as float16.
        as_fp16 = raw_bytes.view(torch.float16)
        self.assertTrue((as_fp16 == 42.0).all().item())


class TestMultiEndedAllocator(unittest.TestCase):
    def test_alloc_grow_up(self):
        _, _, alloc_full, _, _, _ = _build()
        out = alloc_full.alloc(3)
        self.assertIsNotNone(out)
        self.assertEqual(out.tolist(), [1, 2, 3])
        self.assertEqual(alloc_full.watermark, 4)

    def test_alloc_grow_down(self):
        _, _, _, alloc_swa, _, _ = _build()
        max_swa = alloc_swa.max_slots
        out = alloc_swa.alloc(3)
        self.assertIsNotNone(out)
        # Grow-down returns the three highest slot ids in ascending order.
        self.assertEqual(out.tolist(), [max_swa - 3, max_swa - 2, max_swa - 1])
        self.assertEqual(alloc_swa.watermark, max_swa - 4)

    def test_middle_gap_rejection(self):
        """When allocators meet in the middle, the next alloc fails."""
        _, _, alloc_full, alloc_swa, _, _ = _build(total_bytes=4096)
        # Each slot costs 2 (K+V) * 2 (layers) * 2 (head_num) * 4 (head_dim) * 2
        # (float16 bytes) = 64 bytes. Max slots = 64.
        self.assertEqual(alloc_full.max_slots, 64)
        alloc_full.alloc(30)  # slots [1..30]; pool A high byte = 31*64 = 1984
        alloc_swa.alloc(30)  # slots [34..63]; pool B low byte = 34*64 = 2176
        # Byte gap = 2176 - 1984 = 192. Slots = 192/64 = 3.
        self.assertEqual(alloc_full.available_size(), 3)
        self.assertIsNone(alloc_full.alloc(4))
        out = alloc_full.alloc(3)
        self.assertIsNotNone(out)
        self.assertEqual(alloc_full.available_size(), 0)
        self.assertIsNone(alloc_swa.alloc(1))

    def test_free_eager_compaction_boundary(self):
        """Freeing the boundary slot shrinks watermark without a move."""
        _, _, alloc_full, _, kv_full, _ = _build()
        alloc_full.alloc(5)  # slots [1,2,3,4,5]; watermark=6
        alloc_full.free(torch.tensor([5], dtype=torch.int64))
        self.assertEqual(alloc_full.watermark, 5)
        self.assertEqual(kv_full.move_calls, [])
        # Boundary-only free: no relocation produced.
        self.assertEqual(len(alloc_full._inverse_history), 0)

    def test_free_eager_compaction_interior(self):
        """Freeing an interior slot triggers a relocation of the boundary."""
        _, _, alloc_full, _, kv_full, _ = _build()
        alloc_full.alloc(5)  # slots [1..5]; watermark=6
        alloc_full.free(torch.tensor([2], dtype=torch.int64))
        # boundary (slot 5) moved into slot 2; watermark shrinks to 5.
        self.assertEqual(alloc_full.watermark, 5)
        self.assertEqual(kv_full.move_calls, [([2], [5])])
        # The relocation is recorded on the inverse-history.
        self.assertEqual(len(alloc_full._inverse_history), 1)
        src_t, dst_t = alloc_full._inverse_history[-1]
        self.assertEqual(src_t.tolist(), [5])
        self.assertEqual(dst_t.tolist(), [2])

    def test_free_batch_compaction(self):
        """Multiple interior frees compact in a single batched move."""
        _, reloc, alloc_full, _, kv_full, _ = _build()
        alloc_full.alloc(6)  # slots [1..6]; watermark=7
        # Free slots 2 and 4 in the same call.
        alloc_full.free(torch.tensor([2, 4], dtype=torch.int64))
        # Descending order of frees: [4, 2]. b starts at 6.
        # Step s=4: 4 != 6, move 6->4, b=5, watermark=6
        # Step s=2: 2 != 5, move 5->2, b=4, watermark=5
        self.assertEqual(alloc_full.watermark, 5)
        self.assertEqual(len(kv_full.move_calls), 1)
        dst, src = kv_full.move_calls[0]
        self.assertEqual(dst, [4, 2])
        self.assertEqual(src, [6, 5])

    def test_available_size_reflects_peer_growth(self):
        """When the peer grows, this allocator's available_size shrinks."""
        _, _, alloc_full, alloc_swa, _, _ = _build()
        before = alloc_full.available_size()
        alloc_swa.alloc(5)
        after = alloc_full.available_size()
        self.assertEqual(before - after, 5)

    def test_backup_restore_state_alloc_only(self):
        """SGLang's actual spec-decoding pattern: only alloc() occurs inside
        the backup window. restore_state is a pure watermark shrink; the
        inverse-history slice for the window is empty."""
        _, _, alloc_full, _, _, _ = _build()
        alloc_full.alloc(2)  # pre-existing state, watermark=3
        snap = alloc_full.backup_state()
        # Spec window: only alloc.
        alloc_full.alloc(5)
        self.assertEqual(alloc_full.watermark, 8)
        # Rollback: watermark shrinks, no relocations to replay.
        rollback_entries = alloc_full.restore_state(snap)
        self.assertEqual(alloc_full.watermark, 3)
        self.assertEqual(rollback_entries, [])

    def test_backup_restore_state_free_inside_window_warns(self):
        """Theoretical fallback: if a free() occurs inside a backup window,
        restore_state returns a non-empty inverse-history list AND logs a
        warning. Eager compaction overwrites the ``dst`` slot's data, so a
        simple dst->src copy cannot fully restore the pre-checkpoint
        state. SGLang's current scheduling doesn't produce this pattern;
        the test documents the detection path."""
        _, _, alloc_full, _, _, _ = _build()
        alloc_full.alloc(5)  # watermark=6
        snap = alloc_full.backup_state()
        alloc_full.free(torch.tensor([2], dtype=torch.int64))  # 5 -> 2
        self.assertEqual(alloc_full.watermark, 5)

        with self.assertLogs(
            "sglang.srt.mem_cache.multi_ended_allocator", level="WARNING"
        ):
            rollback_entries = alloc_full.restore_state(snap)

        self.assertEqual(alloc_full.watermark, 6)
        # One (src_tensor, dst_tensor) pair was emitted by the free.
        self.assertEqual(len(rollback_entries), 1)
        src_t, dst_t = rollback_entries[0]
        self.assertEqual(src_t.tolist(), [5])
        self.assertEqual(dst_t.tolist(), [2])
        # Caveat documented in the design doc §6.5: the original bytes
        # at slot 2 (before the free) are overwritten and unrecoverable.


class TestMambaSharedMemoryPool(unittest.TestCase):
    """Part B: heterogeneous (MHA + Mamba) SharedMemoryPool."""

    def _make_mamba_spec(self, num_layers=2, grow="down"):
        return MambaSubPoolSpec(
            name="mamba",
            layer_num=num_layers,
            conv_state_shapes=((4,),),  # one conv tensor, inner shape (4,)
            conv_dtype=torch.float16,
            temporal_state_shape=(2, 4),
            temporal_dtype=torch.float16,
            grow_direction=grow,
        )

    def _make_mha_spec(self, grow="up"):
        return MHASubPoolSpec(
            name="full",
            layer_num=2,
            head_num=2,
            head_dim=4,
            store_dtype=torch.float16,
            grow_direction=grow,
        )

    def test_mamba_spec_entry_bytes(self):
        spec = self._make_mamba_spec()
        # conv entry = num_layers * conv_row_bytes = 2 * 4*2 = 16
        # temporal entry = num_layers * temporal_row_bytes = 2 * 2*4*2 = 32
        # total = 48
        self.assertEqual(spec.entry_bytes(), 48)

    def test_shared_memory_pool_heterogeneous(self):
        mha = self._make_mha_spec()
        mamba = self._make_mamba_spec()
        # MHA entry = 2 (K+V) * 2 layers * 2 head_num * 4 head_dim * 2 bytes = 64
        # Mamba entry = 48
        shared = SharedMemoryPool(
            total_bytes=4096,
            sub_pool_specs=[mha, mamba],
            device="cpu",
            enable_memory_saver=False,
        )
        self.assertEqual(shared.max_slots("full"), 4096 // 64)
        self.assertEqual(shared.max_slots("mamba"), 4096 // 48)

    def test_mamba_views_shapes(self):
        mha = self._make_mha_spec()
        mamba = self._make_mamba_spec(num_layers=3)
        shared = SharedMemoryPool(
            total_bytes=8192,
            sub_pool_specs=[mha, mamba],
            device="cpu",
            enable_memory_saver=False,
        )
        conv_views, temporal_view = shared.mamba_views_for("mamba")
        self.assertEqual(len(conv_views), 1)
        self.assertEqual(
            conv_views[0].shape,
            (3, shared.max_slots("mamba"), 4),
        )
        self.assertEqual(
            temporal_view.shape,
            (3, shared.max_slots("mamba"), 2, 4),
        )

    def test_mamba_view_aliases_raw_buffer(self):
        """Writing via the mamba view should be visible at the expected byte
        offset in the raw buffer."""
        mha = self._make_mha_spec()
        mamba = self._make_mamba_spec(num_layers=2)
        shared = SharedMemoryPool(
            total_bytes=4096,
            sub_pool_specs=[mha, mamba],
            device="cpu",
            enable_memory_saver=False,
        )
        conv_views, temporal_view = shared.mamba_views_for("mamba")
        # Write to temporal view: layer=0, slot=1, full inner shape (2, 4).
        temporal_view[0, 1].fill_(7.0)
        # Raw offset: anchor + 0 * (2*conv_row) + entry_bytes + 0 * temporal_row
        # Actually slot 1 of mamba view is at anchor + 1 * entry_bytes.
        # Inside a slot's entry: bytes = [all conv[0] rows for N layers] + [all temporal rows for N layers].
        # temporal rows for layers come after all conv layers. Layer 0 temporal at
        # anchor + slot*entry + num_layers*conv_row_bytes.
        anchor = shared.anchor_bytes("mamba")
        slot = 1
        spec = shared.spec("mamba")
        # Assert a sanity check: bytes match.
        temporal_bytes_start = (
            anchor
            + slot * spec.entry_bytes()
            + spec.layer_num * spec.conv_row_bytes(0)
        )
        raw_slice = shared._raw[
            temporal_bytes_start : temporal_bytes_start + spec.temporal_row_bytes()
        ]
        as_fp16 = raw_slice.view(torch.float16)
        self.assertTrue((as_fp16 == 7.0).all().item())


class TestSlotBacktrackPyAttr(unittest.TestCase):
    """Part A extended: py_attr backtracking for Req.mamba_pool_idx-style refs."""

    def test_bind_unbind_py_attr(self):
        from types import SimpleNamespace

        bt = SlotBacktrack(size_total=64)
        obj = SimpleNamespace(mamba_pool_idx=None)
        bt.bind_py_attr(5, obj, "mamba_pool_idx")
        self.assertIn(5, bt.py_attr)
        bt.unbind_py_attr(5, obj, "mamba_pool_idx")
        self.assertNotIn(5, bt.py_attr)

class TestSharedSWAAllocatorHeterogeneousEntryBytes(unittest.TestCase):
    """
    Regression coverage: SWA sub-pools are heterogeneous — `full_layer_nums`
    and `swa_layer_nums` almost always differ, so `entry_bytes_full` and
    `entry_bytes_swa` differ. The composite allocator MUST coordinate byte
    consumption across both sub-pools. A naive `min(avail_full, avail_swa)`
    pre-check overshoots and crashes mid-alloc.
    """

    def _build_composite(
        self,
        *,
        total_bytes: int,
        full_layer_num: int,
        swa_layer_num: int,
        head_num: int = 2,
        head_dim: int = 4,
    ):
        # Shared memory pool with heterogeneous entry bytes.
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
        shared = SharedMemoryPool(
            total_bytes=total_bytes,
            sub_pool_specs=[full_spec, swa_spec],
            device="cpu",
            enable_memory_saver=False,
        )

        # Stand-in for SharedSWAKVPool exposing the two sub-pools.
        from sglang.srt.mem_cache.shared_memory_pool import (
            SharedMHATokenToKVPool,
        )

        class _FakeSWAKVPool:
            def __init__(self, full_pool, swa_pool):
                self.full_kv_pool = full_pool
                self.swa_kv_pool = swa_pool
                self.full_to_swa_index_mapping = None

            def register_mapping(self, m):
                self.full_to_swa_index_mapping = m

            def translate_loc_from_full_to_swa(self, idx):
                return self.full_to_swa_index_mapping[idx].to(torch.int32)

        full_pool = SharedMHATokenToKVPool(
            shared_buffer=shared, sub_pool_name="full",
            dtype=torch.float16, enable_memory_saver=False,
        )
        swa_pool = SharedMHATokenToKVPool(
            shared_buffer=shared, sub_pool_name="swa",
            dtype=torch.float16, enable_memory_saver=False,
        )
        kv = _FakeSWAKVPool(full_pool, swa_pool)

        full_bt = SlotBacktrack(size_total=shared.max_slots("full"))
        swa_bt = SlotBacktrack(size_total=shared.max_slots("swa"))
        reloc = Relocator(
            backtracks={"full": full_bt, "swa": swa_bt}, device="cpu",
        )

        from sglang.srt.mem_cache.multi_ended_allocator import (
            SharedSWATokenToKVPoolAllocator,
        )

        composite = SharedSWATokenToKVPoolAllocator(
            shared_buffer=shared,
            kvcache=kv,
            relocation_log=reloc,
            device="cpu",
            need_sort=False,
        )
        return shared, kv, composite

    def test_heterogeneous_entry_sizes_accepted(self):
        """Different layer counts yield different entry_bytes and different
        max_slots per sub-pool — construction should succeed."""
        shared, _, comp = self._build_composite(
            total_bytes=4096, full_layer_num=2, swa_layer_num=6,
        )
        # full entry = 2*2*2*4*2 = 64 bytes  -> max_slots_full = 4096//64 = 64
        # swa  entry = 2*6*2*4*2 = 192 bytes -> max_slots_swa  = 4096//192 = 21
        self.assertEqual(shared.max_slots("full"), 64)
        self.assertEqual(shared.max_slots("swa"), 21)
        self.assertEqual(comp.full_attn_allocator.max_slots, 64)
        self.assertEqual(comp.swa_attn_allocator.max_slots, 21)

    def test_min_slot_index_computed_from_entry_max(self):
        """Each sub-pool reserves `min_slot_index = ceil(entry_max/entry_i)`
        slots so real data always starts above every pool's slot-0 dummy-
        write range. See design doc §5.6."""
        shared, _, comp = self._build_composite(
            total_bytes=4096, full_layer_num=2, swa_layer_num=6,
        )
        # entry_full=64, entry_swa=192. entry_max=192.
        # min_full = ceil(192/64) = 3; min_swa = ceil(192/192) = 1.
        self.assertEqual(shared.min_slot_index("full"), 3)
        self.assertEqual(shared.min_slot_index("swa"), 1)
        self.assertEqual(comp.full_attn_allocator.min_slot_index, 3)
        self.assertEqual(comp.swa_attn_allocator.min_slot_index, 1)

    def test_grow_up_alloc_returns_slot_at_min_slot_index(self):
        """First allocatable slot on the grow-up pool is min_slot_index,
        not 1 — slots 1..min-1 are reserved alongside slot 0."""
        shared, _, comp = self._build_composite(
            total_bytes=4096, full_layer_num=2, swa_layer_num=6,
        )
        out = comp.full_attn_allocator.alloc(1)
        self.assertIsNotNone(out)
        self.assertEqual(out.tolist(), [3])  # min_slot_index_full

    def test_pool_L_real_data_outside_pool_H_slot0(self):
        """Pool L's first real slot starts at bytes ≥ entry_max, so any
        pool's slot-0 dummy write cannot overlap pool L's real data."""
        shared, kv, comp = self._build_composite(
            total_bytes=4096, full_layer_num=2, swa_layer_num=6,
        )
        entry_max = max(
            shared.spec("full").entry_bytes(),
            shared.spec("swa").entry_bytes(),
        )
        first_real_slot_full = comp.full_attn_allocator.min_slot_index
        first_real_byte_full = (
            first_real_slot_full * shared.spec("full").entry_bytes()
        )
        # Slot-0 dummy writes on either pool cover [0, entry_max). Pool L's
        # real data starts at first_real_byte_full ≥ entry_max.
        self.assertGreaterEqual(first_real_byte_full, entry_max)

    def test_slot0_dummy_write_does_not_clobber_pool_L_real_data(self):
        """Regression test: simulate a slot-0 dummy write on pool H (the
        peer with the larger entry) and verify that pool L's first real
        slot's bytes are preserved. This is the entire point of the
        min_slot_index scheme."""
        shared, kv, comp = self._build_composite(
            total_bytes=4096, full_layer_num=2, swa_layer_num=6,
        )
        # Write a sentinel into pool L's first real slot.
        sentinel = 0x4D  # arbitrary non-zero byte
        first_real_full = comp.full_attn_allocator.min_slot_index
        # Per-layer K and V tensors come from shared_buffer.mha_views_for.
        k_full, v_full = shared.mha_views_for("full")
        k_full[0][first_real_full].fill_(sentinel)
        v_full[0][first_real_full].fill_(sentinel)
        # Simulate the slot-0 dummy write on pool H (kernel writes zeros).
        k_swa, v_swa = shared.mha_views_for("swa")
        for l in range(shared.mha_spec("swa").layer_num):
            k_swa[l][0].fill_(0)
            v_swa[l][0].fill_(0)
        # Pool L's first real slot must still be the sentinel.
        self.assertTrue((k_full[0][first_real_full] == sentinel).all().item())
        self.assertTrue((v_full[0][first_real_full] == sentinel).all().item())

    def test_available_size_uses_sum_of_entry_bytes(self):
        """Composite `available_size()` returns the max N such that
        N*(entry_full + entry_swa) fits in the middle gap, after each pool
        reserves its own `min_slot_index * entry_bytes` worth of low-index
        slots."""
        shared, _, comp = self._build_composite(
            total_bytes=4096, full_layer_num=2, swa_layer_num=6,
        )
        # min_full=3 -> pool L's real-data floor = 3*64 = 192.
        # min_swa=1 -> pool H's initial low frontier = 21*192 = 4032.
        # Gap bytes = 4032 - 192 = 3840. Per pair = 256. avail = 3840 // 256 = 15.
        self.assertEqual(comp.available_size(), 15)

    def test_alloc_succeeds_up_to_combined_budget(self):
        shared, _, comp = self._build_composite(
            total_bytes=4096, full_layer_num=2, swa_layer_num=6,
        )
        # 15 pair-allocations fit; 16 would exceed the byte budget.
        out = comp.alloc(15)
        self.assertIsNotNone(out)
        self.assertEqual(out.numel(), 15)
        self.assertEqual(comp.available_size(), 0)
        self.assertIsNone(comp.alloc(1))

    def test_alloc_at_joint_boundary_does_not_crash(self):
        """The historical `min(avail_full, avail_swa)` pre-check could pass
        while the sequential alloc crashed mid-way. Fix rejects safely."""
        shared, _, comp = self._build_composite(
            total_bytes=256, full_layer_num=2, swa_layer_num=2,
        )
        # Equal entries: entry_max=entry=64, min_slot_index=1 for both
        # (identical to v1 semantics). Reserved = 2*64 = 128 bytes.
        # Gap = 128; pair cost = 128; avail = 1.
        self.assertEqual(comp.available_size(), 1)
        out = comp.alloc(1)
        self.assertIsNotNone(out)
        self.assertEqual(comp.available_size(), 0)
        self.assertIsNone(comp.alloc(1))

    def test_composite_alloc_leaves_state_unchanged_on_reject(self):
        """A rejected composite.alloc must not leave full/swa watermarks
        partially advanced."""
        shared, _, comp = self._build_composite(
            total_bytes=256, full_layer_num=2, swa_layer_num=2,
        )
        comp.alloc(1)  # consume the only available pair
        full_wm_before = comp.full_attn_allocator.watermark
        swa_wm_before = comp.swa_attn_allocator.watermark
        self.assertIsNone(comp.alloc(1))
        self.assertEqual(comp.full_attn_allocator.watermark, full_wm_before)
        self.assertEqual(comp.swa_attn_allocator.watermark, swa_wm_before)


class TestSetReqMambaPoolIdxHelper(unittest.TestCase):
    """BasePrefixCache._set_req_mamba_pool_idx routes assignments through the
    Mamba SlotBacktrackBinder. `Req.mamba_pool_idx` stays a plain attribute
    on Req (no property / class-level hook)."""

    def test_helper_binds_and_unbinds(self):
        from types import SimpleNamespace

        bt = SlotBacktrack(size_total=64)
        binder = SlotBacktrackBinder(bt)

        # Minimal stand-in for a radix cache: BasePrefixCache.*_req_mamba_pool_idx
        # reads `self.req_to_token_pool.mamba_pool.slot_backtrack_binder`.
        req = SimpleNamespace(mamba_pool_idx=None)
        pool = SimpleNamespace(slot_backtrack_binder=binder)
        rtp = SimpleNamespace(mamba_pool=pool)

        from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache

        dummy_cache = BasePrefixCache.__new__(BasePrefixCache)
        dummy_cache.req_to_token_pool = rtp

        # Assign a slot -> bind.
        dummy_cache._set_req_mamba_pool_idx(req, torch.tensor(7))
        self.assertEqual(req.mamba_pool_idx.item(), 7)
        self.assertIn(7, bt.py_attr)

        # Reassign -> unbind old, bind new.
        dummy_cache._set_req_mamba_pool_idx(req, torch.tensor(9))
        self.assertEqual(req.mamba_pool_idx.item(), 9)
        self.assertNotIn(7, bt.py_attr)
        self.assertIn(9, bt.py_attr)

        # Assign None -> unbind only.
        dummy_cache._set_req_mamba_pool_idx(req, None)
        self.assertIsNone(req.mamba_pool_idx)
        self.assertNotIn(9, bt.py_attr)


class TestRelocatorApplyImmediate(unittest.TestCase):
    """Verify the immediate-apply path: every reference holder is rewritten
    inside `_apply_relocations` (i.e., during `free()`) — no need to call
    `flush()` to observe the rewrite. Row-tracking, multi-col assert, and
    dst-poisoning are exercised end-to-end."""

    def test_free_immediately_updates_tree_node(self):
        from types import SimpleNamespace

        shared, reloc, alloc_full, _, _, _ = _build()
        alloc_full.alloc(7)  # slots [1..7]
        node = SimpleNamespace(value=torch.tensor([1, 5, 7], dtype=torch.int32))
        backtrack = reloc.backtracks["full"]
        backtrack.bind_tree_node(slot=7, node=node, position=2)

        alloc_full.free(torch.tensor([3], dtype=torch.int64))

        # Tree node was rewritten inline by apply(); no flush needed.
        self.assertEqual(node.value[2].item(), 3)

    def test_free_immediately_updates_req_to_token(self):
        from types import SimpleNamespace

        shared, reloc, alloc_full, _, _, _ = _build()
        alloc_full.alloc(5)
        backtrack = reloc.backtracks["full"]
        req_to_token = torch.zeros(2, 4, dtype=torch.int32)
        req_to_token[0, 0] = 5
        req_to_token[1, 0] = 5
        backtrack.bind_req_position(slot=5, row=0, col=0)
        backtrack.bind_req_position(slot=5, row=1, col=0)
        reloc._req_to_token_pool_ref = SimpleNamespace(
            req_to_token=req_to_token
        )

        alloc_full.free(torch.tensor([2], dtype=torch.int64))

        # Both rows rewritten inline. No column scan.
        self.assertEqual(req_to_token[0, 0].item(), 2)
        self.assertEqual(req_to_token[1, 0].item(), 2)

    def test_req_position_tracks_multiple_rows(self):
        backtrack = SlotBacktrack(size_total=128)
        backtrack.bind_req_position(slot=42, row=0, col=3)
        backtrack.bind_req_position(slot=42, row=2, col=3)
        backtrack.bind_req_position(slot=42, row=5, col=3)
        ref = backtrack.req_position[42]
        self.assertEqual(ref.col, 3)
        self.assertEqual(ref.rows, {0, 2, 5})

        backtrack.unbind_req_position(slot=42, row=2)
        self.assertEqual(backtrack.req_position[42].rows, {0, 5})

        # Drop all rows -> entry removed from dict.
        backtrack.unbind_req_position(slot=42, row=0)
        backtrack.unbind_req_position(slot=42, row=5)
        self.assertNotIn(42, backtrack.req_position)

    def test_bind_req_position_multi_col_asserts(self):
        backtrack = SlotBacktrack(size_total=128)
        backtrack.bind_req_position(slot=10, row=0, col=4)
        with self.assertRaises(AssertionError):
            backtrack.bind_req_position(slot=10, row=1, col=5)

    def test_apply_skips_stale_rows(self):
        """Rows in the binder's set whose req_to_token cell no longer
        holds `src` are skipped — the verify mask filters them out so we
        don't corrupt a different req that took over the row."""
        from types import SimpleNamespace

        shared, reloc, alloc_full, _, _, _ = _build()
        alloc_full.alloc(5)
        backtrack = reloc.backtracks["full"]
        req_to_token = torch.zeros(4, 4, dtype=torch.int32)
        # Row 0 still holds slot 5 at col 1; row 3 was reassigned and the
        # cell holds something else (999), but the binder still has row=3
        # in its set (e.g., the row owner exited a non-cache-aware path).
        req_to_token[0, 1] = 5
        req_to_token[3, 1] = 999
        backtrack.bind_req_position(slot=5, row=0, col=1)
        backtrack.bind_req_position(slot=5, row=3, col=1)
        reloc._req_to_token_pool_ref = SimpleNamespace(
            req_to_token=req_to_token
        )

        alloc_full.free(torch.tensor([2], dtype=torch.int64))

        # Row 0 rewritten; row 3 left untouched (verify mask filtered).
        self.assertEqual(req_to_token[0, 1].item(), 2)
        self.assertEqual(req_to_token[3, 1].item(), 999)

    def test_dst_not_empty_warns_and_poisons_tree_node(self):
        """If the catch-all is bypassed and dst still has a tree_node
        binding, apply() warns and poisons the prior holder with -1."""
        from types import SimpleNamespace

        shared, reloc, alloc_full, _, _, _ = _build()
        alloc_full.alloc(7)  # slots [1..7]
        backtrack = reloc.backtracks["full"]
        # Bind boundary slot 7 to a node (it's the source of the move).
        node_src = SimpleNamespace(
            value=torch.tensor([7], dtype=torch.int32)
        )
        backtrack.bind_tree_node(slot=7, node=node_src, position=0)

        # Bypass the catch-all: directly stuff a binding at slot 3 (which
        # will be the dst of the upcoming relocation).
        node_stale = SimpleNamespace(
            value=torch.tensor([3], dtype=torch.int32)
        )
        backtrack.tree_node[3] = backtrack.tree_node.get(7).__class__(
            node=node_stale, position=0, attr="value"
        )

        alloc_full.free(torch.tensor([3], dtype=torch.int64))

        # Prior holder (node_stale) was poisoned to -1.
        self.assertEqual(node_stale.value[0].item(), -1)
        # New holder (node_src) was rewritten 7 -> 3.
        self.assertEqual(node_src.value[0].item(), 3)
        # Binder at slot 3 now holds the transferred entry from slot 7.
        self.assertIn(3, backtrack.tree_node)
        self.assertIs(backtrack.tree_node[3].node, node_src)


if __name__ == "__main__":
    unittest.main()
