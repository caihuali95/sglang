"""Routing tests for the Stage 3.5 full-attention `set_full_loc` fast path.

Background: `ForwardBatch.init_new` precomputes the full-physical write
location ONCE per batch (`out_cache_loc_full_physical =
allocator.translate_kv_loc(out_cache_loc)`) and `model_runner` pins it via
`pool.set_full_loc(...)`. The intent is that per-layer `set_kv_buffer`
calls reuse this precompute instead of each re-running the
`virtual_to_physical[loc]` gather + clamp.

The composite write paths must therefore forward the pinned `full_loc`
(full-physical) to the inner `SharedMHATokenToKVPool`, so its data-ptr
fast path fires and skips the per-call v2p gather. A prior bug forwarded
the *virtual* `loc` instead (data-ptr never matched the pinned physical
tensor), so the gather fired on every full-attention layer every forward.

These tests pin the routing for BOTH composites:
  - `SharedSWAKVPool`     (hybrid-SWA models, e.g. gpt-oss-20b)
  - `HybridLinearKVPool`  (hybrid-Mamba models, e.g. Falcon-H1)

They are pure dispatch tests: the inner sub-pools are recording stubs, so
no GPU / real buffers are needed (CPU CI).

    python -m pytest test/registered/unit/mem_cache/test_full_loc_fast_path.py -v
"""

import types
import unittest

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


class _RecordingPool:
    """Stub sub-pool that records the `loc` passed to `set_kv_buffer`."""

    def __init__(self):
        self.calls = []

    def set_kv_buffer(self, layer, loc, cache_k, cache_v, *args, **kwargs):
        self.calls.append(loc)

    # SharedSWAKVPool also calls set_loc on the inner pool via set_swa_loc;
    # not exercised here but present for completeness.
    def set_loc(self, loc):
        pass


class TestSharedSWAFullLocRouting(unittest.TestCase):
    """`SharedSWAKVPool.set_kv_buffer` must forward the pinned full-physical
    `full_loc` (not the virtual `loc`) on full-attention layers."""

    def _make_bare_pool(self):
        from sglang.srt.mem_cache.shared_memory_pool import SharedSWAKVPool

        # Bypass the heavy __init__; set only the attributes set_kv_buffer reads.
        pool = object.__new__(SharedSWAKVPool)
        pool.full_kv_pool = _RecordingPool()
        pool.swa_kv_pool = _RecordingPool()
        # layer 0 -> full attention; layer 1 -> SWA. (pool_layer_id, is_swa)
        pool.layers_mapping = {0: (0, False), 1: (0, True)}
        pool.swa_loc = None
        pool.full_loc = None
        return pool

    def test_full_layer_forwards_full_loc_when_pinned(self):
        pool = self._make_bare_pool()
        virtual_loc = torch.tensor([10, 11, 12], dtype=torch.int64)
        full_phys = torch.tensor([3, 4, 5], dtype=torch.int64)  # translated
        pool.full_loc = full_phys

        layer = types.SimpleNamespace(layer_id=0)  # full layer
        ck = torch.zeros(3, 4, 8)
        cv = torch.zeros(3, 4, 8)
        pool.set_kv_buffer(layer, virtual_loc, ck, cv)

        self.assertEqual(len(pool.full_kv_pool.calls), 1)
        forwarded = pool.full_kv_pool.calls[0]
        # Must forward the pinned full-physical tensor (same object), NOT the
        # virtual loc — otherwise the inner data-ptr fast path can't fire.
        self.assertIs(forwarded, full_phys)
        self.assertIsNot(forwarded, virtual_loc)

    def test_full_layer_falls_back_to_loc_when_not_pinned(self):
        pool = self._make_bare_pool()
        virtual_loc = torch.tensor([10, 11, 12], dtype=torch.int64)
        pool.full_loc = None  # no precompute pinned (slice-safe fallback)

        layer = types.SimpleNamespace(layer_id=0)
        pool.set_kv_buffer(layer, virtual_loc, torch.zeros(3, 4, 8), torch.zeros(3, 4, 8))

        self.assertEqual(len(pool.full_kv_pool.calls), 1)
        # Fallback: forward the (virtual) loc for per-call translate.
        self.assertIs(pool.full_kv_pool.calls[0], virtual_loc)

    def test_swa_layer_forwards_swa_loc_when_pinned(self):
        # Sanity: the SWA branch (already correct) still forwards swa_loc.
        pool = self._make_bare_pool()
        virtual_loc = torch.tensor([10, 11, 12], dtype=torch.int64)
        swa_phys = torch.tensor([1, 2, 0], dtype=torch.int64)
        pool.swa_loc = swa_phys

        layer = types.SimpleNamespace(layer_id=1)  # SWA layer
        pool.set_kv_buffer(layer, virtual_loc, torch.zeros(3, 4, 8), torch.zeros(3, 4, 8))

        self.assertEqual(len(pool.swa_kv_pool.calls), 1)
        self.assertIs(pool.swa_kv_pool.calls[0], swa_phys)
        # Full pool untouched for an SWA layer.
        self.assertEqual(len(pool.full_kv_pool.calls), 0)


class TestHybridLinearFullLocRouting(unittest.TestCase):
    """`HybridLinearKVPool.set_kv_buffer` (non-MLA) must forward the pinned
    full-physical `full_loc` on full-attention layers."""

    def _make_bare_pool(self):
        from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool

        pool = object.__new__(HybridLinearKVPool)
        pool.full_kv_pool = _RecordingPool()
        pool.use_mla = False
        pool.full_attention_layer_id_mapping = {0: 0}
        pool.full_loc = None
        return pool

    def test_forwards_full_loc_when_pinned(self):
        pool = self._make_bare_pool()
        virtual_loc = torch.tensor([7, 8, 9], dtype=torch.int64)
        full_phys = torch.tensor([2, 3, 4], dtype=torch.int64)
        pool.full_loc = full_phys

        layer = types.SimpleNamespace(layer_id=0)
        pool.set_kv_buffer(layer, virtual_loc, torch.zeros(3, 4, 8), torch.zeros(3, 4, 8))

        self.assertEqual(len(pool.full_kv_pool.calls), 1)
        forwarded = pool.full_kv_pool.calls[0]
        self.assertIs(forwarded, full_phys)
        self.assertIsNot(forwarded, virtual_loc)

    def test_falls_back_to_loc_when_not_pinned(self):
        pool = self._make_bare_pool()
        virtual_loc = torch.tensor([7, 8, 9], dtype=torch.int64)
        pool.full_loc = None

        layer = types.SimpleNamespace(layer_id=0)
        pool.set_kv_buffer(layer, virtual_loc, torch.zeros(3, 4, 8), torch.zeros(3, 4, 8))

        self.assertEqual(len(pool.full_kv_pool.calls), 1)
        self.assertIs(pool.full_kv_pool.calls[0], virtual_loc)


if __name__ == "__main__":
    unittest.main()
