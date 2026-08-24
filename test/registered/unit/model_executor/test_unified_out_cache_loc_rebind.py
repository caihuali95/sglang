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
"""ForwardBatch construction wires the unified write-loc rebind.

The rebind SEMANTICS (fresh tensor, swa-from-virtual order, resolver, pad
handover) are pinned in test_kv_index_source.py over the real composite.
This file pins the WIRING — the two ForwardBatch-side call sites that make
the contract hold for every batch:

  1. `init_new` calls `kv_index_source.rebind_write_loc` (a construction
     path that skipped it would ship VIRTUAL write ids to the kernels, a
     silent wrong-slot store under the unified pool);
  2. the two in-flight transforms that REPLACE the rebound tensor hand the
     new tensor to `note_write_loc_replaced` — `_pad_inputs_to_size` (after
     the out_cache_loc pad) and the eager runner's `load_batch` (which
     rebuilds every eager batch into the input registry's static buffers).
     A dropped handover strands the source on a retired tensor and the next
     resolve refuses a legitimate write loc;

plus an end-to-end run of the REAL `_pad_inputs_to_size` against a live
source: padded lanes resolve to the slot-0 sink and post-pad slices (the
TBO-child shape) still resolve.

    python -m pytest test/registered/unit/model_executor/test_unified_out_cache_loc_rebind.py -v
"""

import ast
import inspect
import textwrap
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.kv_index_source import KVIndexSource
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_DEV = "cpu"


def _make_fb(out_cache_loc, **kw):
    """Minimal ForwardBatch with only the required core fields."""
    n = 0 if out_cache_loc is None else out_cache_loc.shape[0]
    defaults = dict(
        forward_mode=ForwardMode.DECODE,
        batch_size=max(n, 1),
        input_ids=torch.zeros(max(n, 1), dtype=torch.int64),
        req_pool_indices=torch.zeros(max(n, 1), dtype=torch.int64),
        seq_lens=torch.ones(max(n, 1), dtype=torch.int64),
        out_cache_loc=out_cache_loc,
        seq_lens_sum=max(n, 1),
    )
    defaults.update(kw)
    return ForwardBatch(**defaults)


def _armed_source(v2p, swa_map):
    """A KVIndexSource hand-armed with fake translates: this file pins the
    ForwardBatch-side wiring, not the composite's formulas (those are pinned
    in test_kv_index_source.py over the real allocator)."""
    src = KVIndexSource(
        req_to_token=torch.zeros((1, 4), dtype=torch.int64),
        token_to_kv_pool_allocator=SimpleNamespace(),
        token_to_kv_pool=SimpleNamespace(),
        page_size=1,
        device=_DEV,
    )
    src.enabled = True
    src._translate_full = lambda t, out=None: v2p[t.to(torch.int64)]
    src._translate_swa = lambda t: swa_map[t.to(torch.int64)]
    return src


def _call_names(func) -> list:
    """Dotted call targets appearing in `func`'s body, e.g.
    'model_runner.kv_index_source.rebind_write_loc'."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            parts = []
            cur = node.func
            while isinstance(cur, ast.Attribute):
                parts.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name):
                parts.append(cur.id)
            names.append(".".join(reversed(parts)))
    return names


class TestForwardBatchWiring(CustomTestCase):
    """Critical-path bookkeeping: the construction-time call sites."""

    def test_init_new_calls_the_rebind(self):
        self.assertIn(
            "model_runner.kv_index_source.rebind_write_loc",
            _call_names(ForwardBatch.init_new.__func__),
            "init_new must rebind the write loc through the source; a batch "
            "built without it ships virtual ids to the kernels",
        )

    def test_pad_notifies_the_source_after_padding_the_loc(self):
        src = textwrap.dedent(inspect.getsource(ForwardBatch._pad_inputs_to_size))
        pad_pos = src.index("self.out_cache_loc = self._pad_tensor_to_size")
        note_pos = src.index("note_write_loc_replaced")
        self.assertGreater(
            note_pos,
            pad_pos,
            "_pad_inputs_to_size must hand the PADDED tensor to "
            "note_write_loc_replaced (after the out_cache_loc pad), or the "
            "source keeps resolving against the retired tensor",
        )

    def test_eager_load_batch_hands_the_rebuilt_loc_to_the_source(self):
        """The eager input registry rebuilds EVERY eager batch into its
        static buffers before metadata init; without the handover the
        resolver refuses the first eager extend on a hybrid-SWA unified
        server (the registry copy is a different allocation from the
        rebound rail)."""
        from sglang.srt.model_executor.runner.eager_runner import EagerRunner

        src = textwrap.dedent(inspect.getsource(EagerRunner.load_batch))
        extract_pos = src.index("extract_buffer")
        note_pos = src.index("note_write_loc_replaced")
        self.assertGreater(
            note_pos,
            extract_pos,
            "EagerRunner.load_batch must hand the registry-rebuilt "
            "out_cache_loc to note_write_loc_replaced after extract_buffer",
        )


class TestPadInputsHandsRailToSource(CustomTestCase):
    def _fake_runner_for_pad(self, src):
        return SimpleNamespace(
            attn_backend=SimpleNamespace(get_cuda_graph_seq_len_fill_value=lambda: 0),
            kv_index_source=src,
        )

    def test_pad_then_resolve_covers_sink_and_slices(self):
        n, padded = 3, 6
        v2p = torch.arange(64, dtype=torch.int64) * 3
        swa_map = torch.arange(64, dtype=torch.int64) * 5
        src = _armed_source(v2p, swa_map)
        virt = torch.tensor([11, 12, 13], dtype=torch.int64)
        fb = _make_fb(virt.clone())
        fb.positions = torch.arange(n, dtype=torch.int64)
        fb.lora_ids = [None] * fb.batch_size
        src.rebind_write_loc(fb)
        self.assertTrue(torch.equal(fb.out_cache_loc, v2p[virt]))

        fb._pad_inputs_to_size(self._fake_runner_for_pad(src), padded, fb.batch_size)

        self.assertEqual(fb.out_cache_loc.shape[0], padded)
        # Padded tail lanes go to slot 0 — the reserved dummy-write sink.
        self.assertTrue(bool((fb.out_cache_loc[n:] == 0).all()))
        rail = src.resolve_swa_write_loc(fb.out_cache_loc)
        self.assertTrue(torch.equal(rail[:n], swa_map[virt]))
        self.assertTrue(bool((rail[n:] == 0).all()))
        self.assertEqual(rail.dtype, torch.int64)
        # The TBO-child shape: a slice of the PADDED tensor still resolves.
        sub = src.resolve_swa_write_loc(fb.out_cache_loc[1:5])
        self.assertTrue(torch.equal(sub, rail[1:5]))

    def test_empty_loc_rebinds_to_empty(self):
        src = _armed_source(
            torch.arange(8, dtype=torch.int64), torch.arange(8, dtype=torch.int64)
        )
        fb = _make_fb(torch.empty(0, dtype=torch.int64))
        src.rebind_write_loc(fb)
        self.assertEqual(fb.out_cache_loc.numel(), 0)
        self.assertEqual(src.resolve_swa_write_loc(fb.out_cache_loc).numel(), 0)


if __name__ == "__main__":
    unittest.main()
