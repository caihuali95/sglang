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
):
    return resolve_draft_kv_geometry(
        server_args=server_args or _server_args(),
        spec_algorithm=spec_algorithm,
        target_model_config=target_model_config,
        draft_model_config=draft_model_config,
        kv_cache_dtype=kv_cache_dtype,
        attn_tp_size=attn_tp_size,
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
