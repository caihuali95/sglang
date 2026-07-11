"""Unit tests for `resolve_draft_kv_geometry` (speculative/draft_utils.py) --
CPU only, pure config arithmetic.

Stage 4.1 (fused draft KV, design doc §33.4 Design B) attaches the draft
model's per-slot KV byte region to the host sub-pool's fused entry. These
tests pin the per-case geometry resolution:
  1. MTP/NEXTN self-draft (no draft path): target geometry, nnpl layers;
     nnpl absent/0 -> None,
  2. EAGLE separate checkpoint: DRAFT-config geometry (EAGLE3-like head
     dims differ from the target's),
  3. DFLASH: layer count from the parsed dflash draft config,
  4. no-draft-KV algorithms (NONE, NGRAM) -> None,
  5. multi-layer EAGLE: layer sum = num_steps x per-step count (single-layer
     MTP arch override), and
  6. entry_bytes math incl. asymmetric v_head_dim and dtype itemsize.
"""

import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.speculative.draft_utils import resolve_draft_kv_geometry
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _model_config(
    *,
    arch="LlamaForCausalLM",
    num_nextn_predict_layers=None,
    num_hidden_layers=32,
    num_attention_layers=0,
    head_dim=128,
    v_head_dim=None,
    num_kv_heads=8,
    hf_config_extra=None,
):
    """Duck-typed stand-in for ModelConfig: only the fields the resolver reads."""
    hf_config = SimpleNamespace(
        architectures=[arch], **(hf_config_extra or {})
    )
    return SimpleNamespace(
        hf_config=hf_config,
        num_nextn_predict_layers=num_nextn_predict_layers,
        num_hidden_layers=num_hidden_layers,
        num_attention_layers=num_attention_layers,
        head_dim=head_dim,
        v_head_dim=v_head_dim,
        get_num_kv_heads=lambda attn_tp_size, _n=num_kv_heads: max(
            1, _n // attn_tp_size
        ),
    )


def _server_args(*, enable_multi_layer_eagle=False, speculative_num_steps=None):
    return SimpleNamespace(
        enable_multi_layer_eagle=enable_multi_layer_eagle,
        speculative_num_steps=speculative_num_steps,
    )


def _resolve(
    *,
    spec_algorithm,
    target_model_config,
    draft_model_config,
    server_args=None,
    kv_cache_dtype=torch.bfloat16,
    attn_tp_size=1,
    is_self_draft=False,
):
    return resolve_draft_kv_geometry(
        server_args=server_args or _server_args(),
        spec_algorithm=spec_algorithm,
        target_model_config=target_model_config,
        draft_model_config=draft_model_config,
        kv_cache_dtype=kv_cache_dtype,
        attn_tp_size=attn_tp_size,
        is_self_draft=is_self_draft,
    )


class TestResolveDraftKvGeometry(CustomTestCase):
    def test_mtp_self_draft_uses_target_geometry(self):
        target = _model_config(
            num_nextn_predict_layers=1, head_dim=64, v_head_dim=32, num_kv_heads=4
        )
        geometry = _resolve(
            spec_algorithm=SpeculativeAlgorithm.EAGLE,
            target_model_config=target,
            draft_model_config=None,
        )
        self.assertIsNotNone(geometry)
        self.assertEqual(geometry.layer_num, 1)
        self.assertEqual(geometry.head_num, 4)
        self.assertEqual(geometry.head_dim, 64)
        self.assertEqual(geometry.v_head_dim, 32)
        self.assertEqual(geometry.store_dtype, torch.bfloat16)

    def test_mtp_self_draft_without_mtp_layers_is_none(self):
        for nnpl in (None, 0):
            target = _model_config(num_nextn_predict_layers=nnpl)
            self.assertIsNone(
                _resolve(
                    spec_algorithm=SpeculativeAlgorithm.EAGLE,
                    target_model_config=target,
                    draft_model_config=None,
                ),
                msg=f"nnpl={nnpl}",
            )

    def test_self_draft_reads_the_mtp_swapped_draft_config(self):
        """eval_270 regression: targets outside the auto-draft-path list (Qwen3.5,
        Qwen3-Next) leave speculative_draft_model_path=None, so the caller re-reads
        the TARGET checkpoint with is_draft_model=True — which is what swaps in the
        MTP arch and sets num_nextn_predict_layers. Reading the raw TARGET config
        instead sees no MTP head and silently skips fusion."""
        # Target config as-loaded: no MTP head advertised.
        target = _model_config(arch="Qwen3_5ForConditionalGeneration", head_dim=128)
        # Same checkpoint re-read with is_draft_model=True (arch swapped, nnpl set).
        draft = _model_config(
            arch="Qwen3_5ForCausalLMMTP",
            num_nextn_predict_layers=1,
            head_dim=128,
            num_kv_heads=8,
        )
        geometry = _resolve(
            spec_algorithm=SpeculativeAlgorithm.EAGLE,
            target_model_config=target,
            draft_model_config=draft,
            is_self_draft=True,
        )
        self.assertIsNotNone(geometry, "self-draft MTP geometry must resolve")
        self.assertEqual(geometry.layer_num, 1)
        self.assertEqual(geometry.head_num, 8)

    def test_mtp_draft_never_sizes_at_the_full_model(self):
        """eval_272 regression: ModelConfig surfaces num_nextn_predict_layers from
        hf_TEXT_config, but _config_draft_model sets it on hf_config — for nested
        configs (Qwen3.5) those differ, so nnpl read as None and the layer count
        fell through to num_hidden_layers: the draft region was sized at the WHOLE
        32-layer target (5x cell inflation) instead of the 1-layer MTP head.
        ModelConfig now falls back to hf_config; this pins the resolver's guard so
        an MTP draft with an indeterminable head size refuses to fuse rather than
        fusing the full model."""
        mtp_no_nnpl = _model_config(
            arch="Qwen3_5ForCausalLMMTP",
            num_nextn_predict_layers=None,  # nnpl failed to surface
            num_hidden_layers=32,  # the FULL target's depth
        )
        self.assertIsNone(
            _resolve(
                spec_algorithm=SpeculativeAlgorithm.EAGLE,
                target_model_config=_model_config(),
                draft_model_config=mtp_no_nnpl,
                is_self_draft=True,
            ),
            "an MTP draft must never size its region at the full model",
        )
        # With nnpl surfaced (the ModelConfig fix), it fuses at 1 layer.
        mtp = _model_config(
            arch="Qwen3_5ForCausalLMMTP",
            num_nextn_predict_layers=1,
            num_hidden_layers=32,
        )
        geometry = _resolve(
            spec_algorithm=SpeculativeAlgorithm.EAGLE,
            target_model_config=_model_config(),
            draft_model_config=mtp,
            is_self_draft=True,
        )
        self.assertEqual(geometry.layer_num, 1)

    def test_self_draft_without_mtp_head_does_not_fuse_full_model(self):
        """Safety guard: if the is_draft_model re-read did NOT swap in an MTP arch,
        the target has no MTP head and cfg.num_hidden_layers is the FULL model's —
        fusing that would size the draft region at the whole target."""
        no_mtp = _model_config(arch="LlamaForCausalLM", num_hidden_layers=32)
        self.assertIsNone(
            _resolve(
                spec_algorithm=SpeculativeAlgorithm.EAGLE,
                target_model_config=no_mtp,
                draft_model_config=no_mtp,
                is_self_draft=True,
            )
        )

    def test_eagle_checkpoint_uses_draft_geometry(self):
        # EAGLE3-like: the draft checkpoint's head dims differ from the target's.
        target = _model_config(head_dim=128, num_kv_heads=8)
        draft = _model_config(
            num_hidden_layers=1, head_dim=96, v_head_dim=96, num_kv_heads=2
        )
        geometry = _resolve(
            spec_algorithm=SpeculativeAlgorithm.EAGLE3,
            target_model_config=target,
            draft_model_config=draft,
        )
        self.assertEqual(geometry.layer_num, 1)
        self.assertEqual(geometry.head_num, 2)
        self.assertEqual(geometry.head_dim, 96)
        self.assertEqual(geometry.v_head_dim, 96)

    def test_eagle_checkpoint_nnpl_wins_over_hidden_layers(self):
        draft = _model_config(num_nextn_predict_layers=2, num_hidden_layers=40)
        geometry = _resolve(
            spec_algorithm=SpeculativeAlgorithm.EAGLE,
            target_model_config=_model_config(),
            draft_model_config=draft,
        )
        self.assertEqual(geometry.layer_num, 2)

    def test_dflash_layer_count_from_draft_config(self):
        # Stub the dflash_utils module: the resolver's DFLASH branch only
        # calls parse_dflash_draft_config(...).require_num_layers(); the real
        # parser is dflash_utils' own tested code (and its import chain pulls
        # triton, unavailable on CPU-only hosts).
        draft = _model_config(num_hidden_layers=5, head_dim=64, num_kv_heads=8)
        fake_dflash_utils = SimpleNamespace(
            parse_dflash_draft_config=lambda *, draft_hf_config: SimpleNamespace(
                require_num_layers=lambda: 5
            )
        )
        with mock.patch.dict(
            sys.modules, {"sglang.srt.speculative.dflash_utils": fake_dflash_utils}
        ):
            geometry = _resolve(
                spec_algorithm=SpeculativeAlgorithm.DFLASH,
                target_model_config=_model_config(),
                draft_model_config=draft,
            )
        self.assertEqual(geometry.layer_num, 5)
        self.assertEqual(geometry.head_dim, 64)

    def test_no_draft_kv_algorithms_return_none(self):
        target = _model_config(num_nextn_predict_layers=1)
        for algo in (SpeculativeAlgorithm.NONE, SpeculativeAlgorithm.NGRAM):
            self.assertIsNone(
                _resolve(
                    spec_algorithm=algo,
                    target_model_config=target,
                    draft_model_config=None,
                ),
                msg=str(algo),
            )

    def test_multi_layer_eagle_sums_per_step_layers(self):
        # MiMoV2MTP: per-step layer count forced to 1; 3 steps -> 3 layers.
        draft = _model_config(arch="MiMoV2MTP", num_nextn_predict_layers=3)
        geometry = _resolve(
            spec_algorithm=SpeculativeAlgorithm.EAGLE,
            target_model_config=_model_config(),
            draft_model_config=draft,
            server_args=_server_args(
                enable_multi_layer_eagle=True, speculative_num_steps=3
            ),
        )
        self.assertEqual(geometry.layer_num, 3)

    def test_multi_layer_eagle_without_steps_is_none(self):
        draft = _model_config(arch="MiMoV2MTP")
        self.assertIsNone(
            _resolve(
                spec_algorithm=SpeculativeAlgorithm.EAGLE,
                target_model_config=_model_config(),
                draft_model_config=draft,
                server_args=_server_args(
                    enable_multi_layer_eagle=True, speculative_num_steps=None
                ),
            )
        )

    def test_entry_bytes_math(self):
        # 2 layers x 4 heads x (64 K + 32 V) x bf16(2B) per slot,
        # and TP sharding halves head_num.
        draft = _model_config(
            num_nextn_predict_layers=2, head_dim=64, v_head_dim=32, num_kv_heads=4
        )
        geometry = _resolve(
            spec_algorithm=SpeculativeAlgorithm.EAGLE,
            target_model_config=_model_config(),
            draft_model_config=draft,
            attn_tp_size=2,
        )
        self.assertEqual(geometry.head_num, 2)
        self.assertEqual(geometry.k_row_bytes(), 2 * 64 * 2)
        self.assertEqual(geometry.v_row_bytes(), 2 * 32 * 2)
        self.assertEqual(geometry.entry_bytes(), 2 * (256 + 128))


if __name__ == "__main__":
    unittest.main()
