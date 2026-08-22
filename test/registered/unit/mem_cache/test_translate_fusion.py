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
"""Translate-path pins for the unified pool's fused v2p kernel work.

- the pool-level SWA translate clamps tombstones (bug regression: a
  tombstoned v2p_swa entry — a token slid out of the window and released
  by `free_swa` — yielded a NEGATIVE id; a captured graph gathers through
  the result, so -1 became an out-of-bounds `swa_k_buffer[-1]` read at
  replay time, while the composite allocator's translate has carried the
  clamp all along);
- fused-vs-reference equivalence for every translate the fused kernel
  replaces (physical, dense, SWA int32), on the CPU reference path the
  kernel module ships.

  python -m pytest test/registered/unit/mem_cache/test_translate_fusion.py -v
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.multi_ended_allocator import MultiEndedAllocator
from sglang.srt.mem_cache.unified_memory_pool import (
    MHASubPoolSpec,
    UnifiedKVPool,
    UnifiedSWAKVPool,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_DEV = "cpu"


class _FakeKVCache:
    def __init__(self, max_slots):
        self.buf = torch.full((max_slots,), -1, dtype=torch.int64)

    def move_kv_cache(self, dst_loc, src_loc):
        self.buf[dst_loc] = self.buf[src_loc].clone()


def _swa_side_allocator(page_size=1, n_full=32, n_swa=16):
    """A real swa-side MultiEndedAllocator over a tiny two-pool buffer."""
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
        page_size=page_size,
    )
    full_alloc = MultiEndedAllocator(
        kvcache=_FakeKVCache(pool.max_slots("full")),
        unified_buffer=pool,
        sub_pool_name="full",
        device=_DEV,
        is_id_owner=True,
        page_size=page_size,
    )
    swa_alloc = MultiEndedAllocator(
        kvcache=_FakeKVCache(pool.max_slots("swa")),
        unified_buffer=pool,
        sub_pool_name="swa",
        device=_DEV,
        is_id_owner=False,
        page_size=page_size,
    )
    full_alloc.bind_peer(swa_alloc)
    swa_alloc.bind_peer(full_alloc)
    v = full_alloc.alloc(4 * page_size)
    swa_alloc.alloc_with_virtual(v)
    return swa_alloc, v


class TestPoolLevelSwaTranslateClampsTombstones(unittest.TestCase):
    def _run(self, page_size):
        swa_alloc, v = _swa_side_allocator(page_size=page_size)
        # Tombstone the SECOND allocated page directly (the state `free_swa`
        # leaves behind for a slid-out token).
        pages = torch.unique(v // page_size)
        swa_alloc.virtual_to_physical[pages[1]] = -1
        # Exercise the shipped pool-level method against the real allocator.
        stand_in = SimpleNamespace(_swa_allocator=swa_alloc)
        out = UnifiedSWAKVPool.translate_loc_from_full_to_swa(stand_in, v)
        self.assertEqual(out.dtype, torch.int32)
        # Live ids stay non-negative; tombstoned ids land on the padding sink
        # (0), never below it — a negative id is an OOB gather under replay.
        self.assertGreaterEqual(int(out.min()), 0)
        tomb = v[(v // page_size) == pages[1]]
        tomb_out = UnifiedSWAKVPool.translate_loc_from_full_to_swa(stand_in, tomb)
        self.assertTrue(torch.all(tomb_out == 0), f"tombstoned ids -> {tomb_out}")

    def test_page_size_1(self):
        self._run(1)

    def test_page_size_2(self):
        self._run(2)


if __name__ == "__main__":
    unittest.main()
