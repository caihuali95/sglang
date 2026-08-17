import types
import unittest

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_batch import ScheduleBatch  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_INTERVAL = 8


class _RecordingAllocator:
    """Mamba slot allocator stub that records the SHAPE of each draw.

    Under the unified pool every draw binds virtual->physical with a kernel
    launch, so `alloc(k)` for k reqs must cost one call, not k.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.draws = []  # requested sizes, in order
        self._next = 100

    def alloc(self, need_size: int):
        self.draws.append(need_size)
        if need_size > self.capacity:
            return None
        self.capacity -= need_size
        slots = torch.arange(self._next, self._next + need_size, dtype=torch.int64)
        self._next += need_size
        return slots


def _make_req(next_track_idx: int = 0, pending: bool = False):
    # buf[idx] == -1 means "slot free"; a non -1 "other" slot means the
    # previous forward's post-processing has not freed it yet.
    buf = torch.tensor([-1, -1], dtype=torch.int64)
    if pending:
        buf[1 - next_track_idx] = 55
    return types.SimpleNamespace(
        mamba_ping_pong_track_buffer=buf,
        mamba_next_track_idx=next_track_idx,
        req_pool_idx=0,
    )


def _make_batch(reqs, seq_lens):
    batch = ScheduleBatch(reqs=reqs)
    batch.device = "cpu"
    batch.seq_lens_cpu = torch.tensor(seq_lens, dtype=torch.int64)
    return batch


def _make_pool(allocator):
    def set_slot(req, idx, value):
        req.mamba_ping_pong_track_buffer[idx] = value

    return types.SimpleNamespace(
        mamba_allocator=allocator,
        set_mamba_ping_pong_slot=set_slot,
    )


class TestMambaLazyPreallocBatching(unittest.TestCase):
    def test_boundary_reqs_draw_in_one_batched_alloc(self):
        reqs = [_make_req(), _make_req(), _make_req()]
        # reqs 0 and 2 sit on a track boundary; req 1 does not.
        batch = _make_batch(reqs, [_INTERVAL, _INTERVAL + 1, 2 * _INTERVAL])
        allocator = _RecordingAllocator(capacity=8)
        batch.req_to_token_pool = _make_pool(allocator)

        batch.mamba_lazy_prealloc_at_boundary(_INTERVAL)

        self.assertEqual(allocator.draws, [2], "one batched draw for 2 boundary reqs")
        self.assertEqual(reqs[0].mamba_next_track_idx, 1)
        self.assertEqual(reqs[2].mamba_next_track_idx, 1)
        self.assertEqual(reqs[1].mamba_next_track_idx, 0)  # untouched
        # Distinct slots, each written to the req's "other" ping-pong index.
        self.assertEqual(reqs[0].mamba_ping_pong_track_buffer[1].item(), 100)
        self.assertEqual(reqs[2].mamba_ping_pong_track_buffer[1].item(), 101)
        self.assertEqual(reqs[1].mamba_ping_pong_track_buffer.tolist(), [-1, -1])

    def test_no_boundary_reqs_draws_nothing(self):
        reqs = [_make_req(), _make_req()]
        batch = _make_batch(reqs, [1, 2])
        allocator = _RecordingAllocator(capacity=8)
        batch.req_to_token_pool = _make_pool(allocator)

        batch.mamba_lazy_prealloc_at_boundary(_INTERVAL)

        self.assertEqual(allocator.draws, [])

    def test_req_with_unfreed_pending_slot_is_skipped(self):
        # Overlap: the previous forward's post-processing has not run yet.
        reqs = [_make_req(pending=True), _make_req()]
        batch = _make_batch(reqs, [_INTERVAL, _INTERVAL])
        allocator = _RecordingAllocator(capacity=8)
        batch.req_to_token_pool = _make_pool(allocator)

        batch.mamba_lazy_prealloc_at_boundary(_INTERVAL)

        self.assertEqual(allocator.draws, [1], "only the eligible req draws")
        self.assertEqual(reqs[0].mamba_ping_pong_track_buffer[1].item(), 55)
        self.assertEqual(reqs[0].mamba_next_track_idx, 0)  # untouched
        self.assertEqual(reqs[1].mamba_next_track_idx, 1)

    def test_tight_capacity_degrades_to_per_request_draws(self):
        """The batch does not fit: fall back so a partial set still tracks,
        exactly as drawing one at a time would."""
        reqs = [_make_req(), _make_req(), _make_req()]
        batch = _make_batch(reqs, [_INTERVAL] * 3)
        allocator = _RecordingAllocator(capacity=2)  # 3 requested, only 2 fit
        batch.req_to_token_pool = _make_pool(allocator)

        batch.mamba_lazy_prealloc_at_boundary(_INTERVAL)

        self.assertEqual(allocator.draws, [3, 1, 1, 1], "batch draw, then per-req")
        # First two tracked, third degraded in place (no slot, index unchanged).
        self.assertEqual(reqs[0].mamba_next_track_idx, 1)
        self.assertEqual(reqs[1].mamba_next_track_idx, 1)
        self.assertEqual(reqs[2].mamba_next_track_idx, 0)
        self.assertEqual(reqs[2].mamba_ping_pong_track_buffer.tolist(), [-1, -1])

    def test_fault_injection_allocates_nothing(self):
        from sglang.srt.environ import envs

        reqs = [_make_req()]
        batch = _make_batch(reqs, [_INTERVAL])
        allocator = _RecordingAllocator(capacity=8)
        batch.req_to_token_pool = _make_pool(allocator)

        with envs.SGLANG_TEST_MAMBA_LAZY_ALLOC_FAIL.override(True):
            batch.mamba_lazy_prealloc_at_boundary(_INTERVAL)

        self.assertEqual(allocator.draws, [])
        self.assertEqual(reqs[0].mamba_next_track_idx, 0)
        self.assertEqual(reqs[0].mamba_ping_pong_track_buffer.tolist(), [-1, -1])


if __name__ == "__main__":
    unittest.main()
