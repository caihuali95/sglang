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
"""`ByteAdmissionBudget` on the unified composites, driven through the REAL
`PrefillAdder` (the hook path a scheduler pass takes).

Pinned here:
  - both composites hand the adder a byte budget (the hook dispatch);
  - byte-cost hand-math: the sliding-window charge priced at the swa entry
    cost, the page overhead and decode estimate at the full entry cost, a
    fresh mamba state at its own slot bytes (NOT the retired ceil-rounded
    full-token conversion);
  - gate/charge round trip against a hand-computed ledger;
  - the `swa_binding` verdict (the chunk-shrink escape-hatch trigger);
  - the fresh-mamba-slot gate inside the budget;
  - THE CAPACITY WIN: with the swa side allocated to its static
    `swa_full_tokens_ratio` split, the byte budget still admits (bytes
    remain in the shared gap) and the allocator REALIZES the admission —
    the conserve wall the token path enforced is gone, and the per-side
    conservation views stay exact (negative by the borrowed amount).

    python -m pytest test/registered/unit/mem_cache/test_byte_admission_budget.py -v
"""

import unittest
from unittest.mock import MagicMock

import torch

from sglang.srt.managers.schedule_policy import PrefillAdder
from sglang.srt.mem_cache.multi_ended_allocator import (
    ByteAdmissionBudget,
    UnifiedMambaTokenToKVPoolAllocator,
    UnifiedSWATokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.unified_memory_pool import (
    MambaSubPoolSpec,
    MHASubPoolSpec,
    UnifiedKVPool,
)
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_DEV = "cpu"
_WINDOW = 8


class _FakeKVCache:
    def __init__(self, max_slots):
        self.buf = torch.full((max_slots,), -1, dtype=torch.int64)

    def move_kv_cache(self, dst_loc, src_loc):
        self.buf[dst_loc] = self.buf[src_loc].clone()


class _FakeSWAKVPool:
    def __init__(self, pool):
        self.full_kv_pool = _FakeKVCache(pool.max_slots("full"))
        self.swa_kv_pool = _FakeKVCache(pool.max_slots("swa"))
        self.full_to_swa_index_mapping = None

    def attach_allocators(self, *, full_allocator, swa_allocator):
        pass


class _FakeMambaKVPool:
    def __init__(self, pool):
        self.full_kv_pool = _FakeKVCache(pool.max_slots("full"))
        self.mamba_pool = _FakeKVCache(pool.max_slots("mamba"))


def _swa_composite(n_full=32, n_swa=8):
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
        layer_num=1,
        head_num=2,
        head_dim=4,
        store_dtype=torch.float16,
        grow_direction="down",
    )
    total = n_full * full_spec.entry_bytes() + n_swa * swa_spec.entry_bytes()
    pool = UnifiedKVPool(
        total_bytes=total,
        sub_pool_specs=[full_spec, swa_spec],
        device=_DEV,
        enable_memory_saver=False,
    )
    allocator = UnifiedSWATokenToKVPoolAllocator(
        unified_buffer=pool,
        kvcache=_FakeSWAKVPool(pool),
        device=_DEV,
        full_max_total_num_tokens=n_full,
        swa_max_total_num_tokens=n_swa,
        need_sort=False,
        forward_stream=None,
    )
    return pool, allocator


def _mamba_composite(n_full=32, n_mamba=8):
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
        grow_direction="down",
        conv_state_shapes=((4, 3),),
        conv_dtype=torch.float32,
        temporal_state_shape=(2, 2, 2),
        temporal_dtype=torch.float32,
    )
    total = n_full * full_spec.entry_bytes() + n_mamba * mamba_spec.entry_bytes()
    pool = UnifiedKVPool(
        total_bytes=total,
        sub_pool_specs=[full_spec, mamba_spec],
        device=_DEV,
        enable_memory_saver=False,
    )
    allocator = UnifiedMambaTokenToKVPoolAllocator(
        unified_buffer=pool,
        kvcache=_FakeMambaKVPool(pool),
        device=_DEV,
    )
    return pool, allocator


class TestByteAdmissionBudget(unittest.TestCase):
    def setUp(self):
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

    def _tree_cache(self, *, mamba=False):
        tree = MagicMock()
        tree.full_evictable_size.return_value = 0
        tree.swa_evictable_size.return_value = 0
        tree.evictable_size.return_value = 0
        tree.mamba_evictable_size.return_value = 0
        tree.supports_mamba.return_value = mamba
        tree.sliding_window_size = _WINDOW
        tree.disable = False
        return tree

    def _adder(self, allocator, tree_cache):
        return PrefillAdder(
            page_size=1,
            tree_cache=tree_cache,
            token_to_kv_pool_allocator=allocator,
            running_batch=None,
            new_token_ratio=1.0,
            rem_input_tokens=10**6,
            rem_chunk_tokens=None,
        )

    def test_composites_hand_out_the_byte_budget(self):
        _, swa_alloc = _swa_composite()
        _, mamba_alloc = _mamba_composite()
        adder_swa = self._adder(swa_alloc, self._tree_cache())
        adder_mamba = self._adder(mamba_alloc, self._tree_cache(mamba=True))
        self.assertIsInstance(adder_swa.budget, ByteAdmissionBudget)
        self.assertIsInstance(adder_mamba.budget, ByteAdmissionBudget)
        self.assertTrue(adder_swa.budget.is_swa)
        self.assertIsNone(adder_swa.budget.rem_mamba_slots)
        self.assertIsNotNone(adder_mamba.budget.rem_mamba_slots)

    def test_swa_prefill_cost_hand_math(self):
        """extend=4, max_new=3, window=8, page=1: the sliding-window charge is
        max(4-8,0) + min(4+3,8) + 1 = 8 tokens at the swa entry cost; the full
        side charges (extend + page) immediately and max_new over the
        lifetime."""
        _, allocator = _swa_composite()
        budget = self._adder(allocator, self._tree_cache()).budget
        e_f = allocator.full_attn_allocator.entry_bytes
        e_s = allocator.swa_attn_allocator.entry_bytes
        cost = budget.prefill_cost(
            cand_extend_input_len=4,
            swa_extend_input_len=4,
            max_new_tokens=3,
        )
        self.assertEqual(cost.swa_bytes, 8 * e_s)
        self.assertEqual(cost.immediate_bytes, (4 + 1) * e_f + 8 * e_s)
        self.assertEqual(cost.lifetime_bytes, (4 + 1) * e_f + 8 * e_s + 3 * e_f)

    def test_mamba_prefill_cost_prices_the_slot_in_bytes(self):
        """A fresh mamba state charges its own slot bytes — not the retired
        ceil-rounded full-token-equivalent conversion."""
        _, allocator = _mamba_composite()
        budget = self._adder(allocator, self._tree_cache(mamba=True)).budget
        e_f = allocator.full_attn_allocator.entry_bytes
        slot_bytes = allocator.mamba_allocator.entry_bytes_per_page
        with_state = budget.prefill_cost(
            cand_extend_input_len=4,
            swa_extend_input_len=4,
            max_new_tokens=3,
            new_mamba_state=True,
        )
        without = budget.prefill_cost(
            cand_extend_input_len=4,
            swa_extend_input_len=4,
            max_new_tokens=3,
            new_mamba_state=False,
        )
        self.assertEqual(
            with_state.immediate_bytes - without.immediate_bytes, slot_bytes
        )
        self.assertEqual(without.immediate_bytes, (4 + 1) * e_f)

    def test_gate_charge_round_trip(self):
        """charge_admitted moves lifetime_remaining by exactly the cost's
        lifetime bytes (same lengths at gate and charge)."""
        _, allocator = _swa_composite()
        budget = self._adder(allocator, self._tree_cache()).budget
        cost = budget.prefill_cost(
            cand_extend_input_len=4,
            swa_extend_input_len=4,
            max_new_tokens=3,
        )
        before = budget.lifetime_remaining()
        self.assertTrue(budget.fits(cost).ok)
        budget.charge_admitted(
            extend_input_len=4, max_new_tokens=3, new_mamba_state=False
        )
        self.assertEqual(before - budget.lifetime_remaining(), cost.lifetime_bytes)

    def test_swa_binding_verdict(self):
        """A candidate that fails ONLY on its sliding-window share reports
        swa_binding — the chunk-shrink escape-hatch trigger."""
        from sglang.srt.mem_cache.multi_ended_allocator import BytePrefillCost

        _, allocator = _swa_composite()
        budget = self._adder(allocator, self._tree_cache()).budget
        remaining = budget.lifetime_remaining()
        swa_share = remaining  # the swa share alone overflows
        cost = BytePrefillCost(
            immediate_bytes=swa_share + 16,
            lifetime_bytes=swa_share + 16,
            swa_bytes=swa_share,
        )
        verdict = budget.fits(cost)
        self.assertFalse(verdict.ok)
        self.assertTrue(verdict.swa_binding)
        # ...and a candidate whose FULL share overflows does not.
        cost2 = BytePrefillCost(
            immediate_bytes=remaining + 16,
            lifetime_bytes=remaining + 16,
            swa_bytes=0,
        )
        verdict2 = budget.fits(cost2)
        self.assertFalse(verdict2.ok)
        self.assertFalse(verdict2.swa_binding)

    def test_fresh_mamba_slot_gate(self):
        """Slots exhausted -> exhausted() even while bytes remain: the gate
        guards a distinct resource (full-evictable bytes cannot produce a
        mamba slot at alloc time)."""
        _, allocator = _mamba_composite()
        budget = self._adder(allocator, self._tree_cache(mamba=True)).budget
        self.assertFalse(budget.exhausted())
        budget.rem_mamba_slots = 0
        self.assertTrue(budget.exhausted())

    def test_static_split_no_longer_walls_admission(self):
        """THE capacity win. Allocate the composite to the swa side's static
        split (n_swa): the old token path's swa gate hit its conserve wall
        (swa_available_size <= 0), but bytes remain in the shared gap — the
        byte budget admits, and the allocator REALIZES the admission. The
        per-side conservation view goes negative by the borrowed amount,
        keeping the leak identity exact."""
        n_swa = 8
        _, allocator = _swa_composite(n_full=64, n_swa=n_swa)
        v = allocator.alloc(n_swa)
        self.assertIsNotNone(v)
        # The old wall: the static split is exhausted...
        self.assertLessEqual(allocator.swa_available_size(), 0)
        # ...but the joint byte view still has room,
        budget = self._adder(allocator, self._tree_cache()).budget
        cost = budget.prefill_cost(
            cand_extend_input_len=2,
            swa_extend_input_len=2,
            max_new_tokens=1,
        )
        self.assertTrue(budget.fits(cost).ok)
        # ...and the admission is REALIZABLE, not just optimistic.
        borrowed = allocator.alloc(2)
        self.assertIsNotNone(borrowed)
        self.assertEqual(allocator.swa_available_size(), -2)


if __name__ == "__main__":
    unittest.main()


class TestPerSideViewsArePureConserve(unittest.TestCase):
    """The per-side availability views must NOT consult the byte-coordinated
    value: the idle leak identity sums the conservation number, and folding a
    tighter schedulable view back in (the old `min`) breaks the identity
    whenever bytes are tighter than slots."""

    def setUp(self):
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

    def test_tighter_schedulable_view_does_not_leak_in(self):
        from unittest.mock import patch

        _, allocator = _swa_composite(n_full=32, n_swa=8)
        conserve_full = allocator._conserve_full_available_size()
        conserve_swa = allocator._conserve_swa_available_size()
        with patch.object(
            allocator, "schedulable_full_available_size", return_value=1
        ), patch.object(allocator, "schedulable_swa_available_size", return_value=1):
            self.assertEqual(allocator.full_available_size(), conserve_full)
            self.assertEqual(allocator.swa_available_size(), conserve_swa)
