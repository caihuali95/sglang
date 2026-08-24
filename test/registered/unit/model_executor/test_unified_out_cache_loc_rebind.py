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
"""ForwardBatch construction wires the unified write-loc rebind (phase 1).

The write contract has two phases: `init_new` rebinds the FULL side once
(phase 1), and the sliding-window side derives at the per-batch build,
pointwise from the dense values (phase 2 — semantics pinned in
test_kv_index_translator.py over the real composite). Phase 2 needs no
ForwardBatch-side wiring at all: pads, slices, and buffer copies preserve
the values it derives from. So this file pins the ONE call site the
contract hangs on:

  `init_new` calls `kv_index_translator.rebind_write_loc` — a construction
  path that skipped it would ship VIRTUAL write ids to the kernels, a
  silent wrong-slot store under the unified pool;

plus an end-to-end run of the REAL `_pad_inputs_to_size` against a live
translator: pad lanes are zeros, and zeros derive to the slot-0 sink —
the property that lets the pad need no handover.

    python -m pytest test/registered/unit/model_executor/test_unified_out_cache_loc_rebind.py -v
"""

import ast
import inspect
import textwrap
import unittest
from types import SimpleNamespace

import torch

import sglang.srt as _sglang_srt
from sglang.srt.mem_cache.kv_index_translator import KVIndexTranslator

_SRC_ROOT = _sglang_srt.__path__[0]
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
    """A KVIndexTranslator hand-armed with fake translates: this file pins the
    ForwardBatch-side wiring, not the composite's formulas (those are pinned
    in test_kv_index_translator.py over the real allocator)."""
    src = KVIndexTranslator(
        req_to_token=torch.zeros((1, 4), dtype=torch.int64),
        token_to_kv_pool_allocator=SimpleNamespace(),
        token_to_kv_pool=SimpleNamespace(),
        page_size=1,
        device=_DEV,
    )
    src.is_translating = True
    src._translate_full = lambda t, out=None: v2p[t.to(torch.int64)]
    # Phase 2 derives from DENSE values through p2v + the swa v2p; arm the
    # inverse of the fake v2p (ps=1, both multipliers 1: dense == physical,
    # and the expected swa loc for virtual t is swa_map[t]).
    p2v = torch.zeros(int(v2p.max()) + 1, dtype=torch.int64)
    p2v[v2p] = torch.arange(v2p.numel(), dtype=torch.int64)
    src._full_p2v_table = p2v
    src._swa_v2p_table = swa_map
    src._full_page_multiplier = 1
    src._swa_page_multiplier = 1
    return src


def _call_names(func) -> list:
    """Dotted call targets appearing in `func`'s body, e.g.
    'model_runner.kv_index_translator.rebind_write_loc'."""
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
            "model_runner.kv_index_translator.rebind_write_loc",
            _call_names(ForwardBatch.init_new.__func__),
            "init_new must rebind the write loc through the source; a batch "
            "built without it ships virtual ids to the kernels",
        )


class TestPadComposesWithDerivation(CustomTestCase):
    def _fake_runner_for_pad(self, src):
        return SimpleNamespace(
            attn_backend=SimpleNamespace(get_cuda_graph_seq_len_fill_value=lambda: 0),
            kv_index_translator=src,
        )

    def test_pad_lanes_derive_to_sink_and_slices_stay_pointwise(self):
        """The REAL `_pad_inputs_to_size` composes with phase 2: pad lanes are
        zeros, zeros derive to the slot-0 sink, and any slice of the padded
        tensor (the TBO-child shape) derives pointwise — no handover call
        exists for the pad to make."""
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
        loc = src._swa_write_loc_from_dense(fb.out_cache_loc)
        self.assertTrue(torch.equal(loc[:n], swa_map[virt]))
        self.assertTrue(bool((loc[n:] == 0).all()))
        self.assertEqual(loc.dtype, torch.int64)
        # The TBO-child shape: a slice of the PADDED tensor derives pointwise.
        sub = src._swa_write_loc_from_dense(fb.out_cache_loc[1:5])
        self.assertTrue(torch.equal(sub, loc[1:5]))

    def test_empty_loc_rebinds_to_empty(self):
        src = _armed_source(
            torch.arange(8, dtype=torch.int64), torch.arange(8, dtype=torch.int64)
        )
        fb = _make_fb(torch.empty(0, dtype=torch.int64))
        src.rebind_write_loc(fb)
        self.assertEqual(fb.out_cache_loc.numel(), 0)
        self.assertEqual(src._swa_write_loc_from_dense(fb.out_cache_loc).numel(), 0)


class TestReadRailTranslatesAtProduction(CustomTestCase):
    """The model-door READ indices (req_to_token-derived, VIRTUAL under the
    unified pool) are translated at their PRODUCTION site — the cache then
    holds the kernel-facing result and the pool door never translates."""

    def _fb_for_one_shot(self):
        fb = _make_fb(torch.tensor([1, 2], dtype=torch.int64))
        fb.batch_size = 2
        fb.seq_lens = torch.tensor([2, 3], dtype=torch.int64)
        fb.seq_lens_cpu = torch.tensor([2, 3], dtype=torch.int32)
        fb.req_pool_indices = torch.tensor([0, 1], dtype=torch.int64)
        return fb

    def test_one_shot_indices_translated_once_and_cached(self):
        from unittest.mock import patch

        from sglang.srt.model_executor import forward_batch_deepseek_mha_mixin as mix

        calls = []
        sentinel = torch.arange(5, dtype=torch.int64) + 5000

        def translate(t):
            calls.append(t)
            return sentinel

        fb = self._fb_for_one_shot()
        fake_pool = SimpleNamespace(
            req_to_token=torch.zeros((4, 16), dtype=torch.int32)
        )
        fake_backend = SimpleNamespace(
            kv_index_translator=SimpleNamespace(translate_full_attn_ids=translate)
        )
        with (
            patch.object(mix, "get_req_to_token_pool", return_value=fake_pool),
            patch.object(mix, "get_attn_backend", return_value=fake_backend),
            patch.object(mix, "create_flashinfer_kv_indices_triton"),
        ):
            r1 = fb.fetch_mha_one_shot_kv_indices()
            r2 = fb.fetch_mha_one_shot_kv_indices()

        self.assertIs(r1, sentinel)  # production site translated
        self.assertIs(r2, sentinel)  # cache holds the TRANSLATED result
        self.assertEqual(len(calls), 1)  # translated exactly once
        self.assertEqual(calls[0].dtype, torch.int32)  # raw producer output

    def test_one_shot_indices_noop_on_unmigrated_backend(self):
        from unittest.mock import patch

        from sglang.srt.model_executor import forward_batch_deepseek_mha_mixin as mix

        fb = self._fb_for_one_shot()
        fake_pool = SimpleNamespace(
            req_to_token=torch.zeros((4, 16), dtype=torch.int32)
        )
        # A backend that never set the attribute inherits the base-class None.
        fake_backend = SimpleNamespace(kv_index_translator=None)
        with (
            patch.object(mix, "get_req_to_token_pool", return_value=fake_pool),
            patch.object(mix, "get_attn_backend", return_value=fake_backend),
            patch.object(mix, "create_flashinfer_kv_indices_triton"),
        ):
            r = fb.fetch_mha_one_shot_kv_indices()
        # The raw int32 producer output passes through untouched.
        self.assertEqual(r.dtype, torch.int32)

    def test_get_mla_kv_buffer_door_passes_loc_untranslated(self):
        from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool

        recorded = {}

        class _RecordingLeafPool:
            def get_mla_kv_buffer(self, layer, loc, dst_dtype):
                recorded["loc"] = loc
                return None, None

            def get_kv_size_bytes(self):
                return 0

        pool = HybridLinearKVPool(
            size=16,
            dtype=torch.float16,
            page_size=1,
            head_num=1,
            head_dim=8,
            full_attention_layer_ids=[0],
            device=_DEV,
            mamba_pool=SimpleNamespace(get_size_per_token=lambda: 0),
            enable_memory_saver=False,
            use_mla=True,
            start_layer=0,
            full_kv_pool=_RecordingLeafPool(),
        )
        loc = torch.tensor([9, 10], dtype=torch.int64)
        pool.get_mla_kv_buffer(SimpleNamespace(layer_id=0), loc, torch.float16)
        self.assertIs(recorded["loc"], loc)


class TestWrapperBackendsCarryTheSource(CustomTestCase):
    """BUG REGRESSION. A wrapper backend that does not forward
    `kv_index_translator` makes `get_attn_backend().kv_index_translator` None, and the
    MHA read-index producers then SKIP the virtual->kernel-facing translate:
    the prefix KV is read from wrong slots and the model emits garbage with no
    crash (observed as GSM8K collapsing from ~0.9 to ~0.1 on an MLA model with
    prefix caching on, only for backends that use the one-shot/chunked-prefix
    path). Both shapes are pinned: the wrappers forward, and the boot guard
    refuses a backend that does not."""

    def _enabled_source(self):
        src = KVIndexTranslator(
            req_to_token=torch.zeros((1, 4), dtype=torch.int64),
            token_to_kv_pool_allocator=SimpleNamespace(),
            token_to_kv_pool=SimpleNamespace(),
            page_size=1,
            device=_DEV,
        )
        src.is_translating = True
        return src

    def test_hybrid_linear_wrapper_forwards_the_source(self):
        from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
            HybridLinearAttnBackend,
        )

        src = self._enabled_source()
        full = SimpleNamespace(
            token_to_kv_pool=SimpleNamespace(),
            req_to_token_pool=SimpleNamespace(),
            kv_index_translator=src,
            max_context_len=16,
            needs_cpu_seq_lens=False,
            extend_dummy_seqs_capped_by_req_pool=False,
        )
        linear = SimpleNamespace(
            needs_cpu_seq_lens=False,
            extend_dummy_seqs_capped_by_req_pool=False,
        )
        wrapper = HybridLinearAttnBackend.__new__(HybridLinearAttnBackend)
        HybridLinearAttnBackend.__init__(
            wrapper,
            full_attn_backend=full,
            linear_attn_backend=linear,
            full_attn_layers=[0],
        )
        self.assertIs(wrapper.kv_index_translator, src)

    def test_every_wrapper_backend_forwards_the_source(self):
        """Completeness: a wrapper forwards `token_to_kv_pool` from its child
        precisely because a forward reaches the WRAPPER, not the child — the
        same is true of the id-space choke point, so every such class must
        forward both. Adding a wrapper that forwards the pool but not the
        source reintroduces the silent-skip bug."""
        import pathlib
        import re

        root = pathlib.Path(_SRC_ROOT) / "layers" / "attention"
        offenders = []
        for path in sorted(root.glob("*.py")):
            text = path.read_text()
            # Wrapper shape: takes token_to_kv_pool off ANOTHER BACKEND (a leaf
            # takes it off model_runner, and an unmigrated leaf legitimately
            # has no source). Count them: one forward per wrapper class.
            wrappers = len(
                re.findall(
                    r"self\.token_to_kv_pool = \w*backend\.token_to_kv_pool", text
                )
            )
            if not wrappers:
                continue
            forwards = len(re.findall(r"self\.kv_index_translator = ", text))
            if forwards < wrappers:
                offenders.append(f"{path.name} ({forwards}/{wrappers})")
        self.assertEqual(
            offenders,
            [],
            "wrapper backends forward the pool but not kv_index_translator "
            f"(forwards/wrappers): {offenders}",
        )

    def test_boot_guard_refuses_a_backend_without_the_source(self):
        src = self._enabled_source()
        # The failure shape: a wrapper that forwards the pools but not the
        # source, so it inherits the base-class None.
        from sglang.srt.layers.attention.base_attn_backend import AttentionBackend

        class _ForgetfulWrapper(AttentionBackend):
            pass

        with self.assertRaises(AssertionError):
            src.assert_backends_carry_translator([_ForgetfulWrapper()])
        # And the healthy shape passes.
        carrier = _ForgetfulWrapper()
        carrier.kv_index_translator = src
        src.assert_backends_carry_translator([carrier, None])

    def test_boot_guard_is_inert_on_a_passthrough_source(self):
        """Non-unified servers must not be forced to carry the attribute."""
        from sglang.srt.layers.attention.base_attn_backend import AttentionBackend

        class _Plain(AttentionBackend):
            pass

        src = KVIndexTranslator(
            req_to_token=torch.zeros((1, 4), dtype=torch.int64),
            token_to_kv_pool_allocator=SimpleNamespace(),
            token_to_kv_pool=SimpleNamespace(),
            page_size=1,
            device=_DEV,
        )
        self.assertFalse(src.is_translating)
        src.assert_backends_carry_translator([_Plain()])


if __name__ == "__main__":
    unittest.main()
