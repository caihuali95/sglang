import logging
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.server_args import ServerArgs, get_global_server_args
from sglang.srt.utils.common import is_blackwell, is_hip, is_musa, is_npu

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.mem_cache.unified_memory_pool import MHARegionGeometry
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

logger = logging.getLogger(__name__)

# Draft archs whose per-runner layer count is forced to 1 regardless of config
# (mirrors the ModelRunner layer-count override; these are the multi-layer
# EAGLE families where each per-step runner loads ONE MTP module).
_SINGLE_LAYER_MTP_ARCHS = ("MiMoV2MTP", "Step3p5MTP")


def _draft_layer_count(model_config: "ModelConfig") -> Optional[int]:
    """KV layer count of one draft runner, mirroring the ModelRunner rules:
    the single-layer MTP arch override first, then `num_nextn_predict_layers`,
    then the dense-draft hidden/attention layer count.

    Returns None for an MTP/NextN draft whose head size cannot be determined —
    falling back to `num_hidden_layers` there would size the draft region at the
    WHOLE target (32 layers instead of 1, a 5x cell inflation). A
    dense draft checkpoint (DFLASH / EAGLE3) legitimately uses its own layer
    count, so only the MTP archs get the guard.
    """
    arch = model_config.hf_config.architectures[0]
    if arch in _SINGLE_LAYER_MTP_ARCHS:
        return 1
    nnpl = model_config.num_nextn_predict_layers
    if nnpl is not None and int(nnpl) > 0:
        return int(nnpl)
    if _is_mtp_draft_arch(model_config):
        return None
    return int(
        max(model_config.num_hidden_layers, model_config.num_attention_layers)
    )


def _is_mtp_draft_arch(model_config: "ModelConfig") -> bool:
    """True when a config re-read with `is_draft_model=True` actually became an
    MTP/NextN draft (`ModelConfig._config_draft_model` swapped the arch). Guards
    the self-draft path: a target with NO MTP head keeps its original arch, and
    its layer count is the FULL model's — fusing that would size the draft region
    at the whole target."""
    arch = model_config.hf_config.architectures[0]
    if arch.endswith("MTP") or arch.endswith("NextN"):
        return True
    nnpl = model_config.num_nextn_predict_layers
    return nnpl is not None and int(nnpl) > 0


def resolve_draft_kv_geometry(
    *,
    server_args: ServerArgs,
    spec_algorithm: "SpeculativeAlgorithm",
    target_model_config: "ModelConfig",
    draft_model_config: Optional["ModelConfig"],
    kv_cache_dtype: torch.dtype,
    attn_tp_size: int,
    is_self_draft: bool = False,
) -> Optional["MHARegionGeometry"]:
    """Resolve the draft model's per-slot KV geometry for the fused draft-KV layout.

    Returns the geometry of the draft byte region that rides inside the host
    sub-pool's fused slot entry, or None when there is no draft KV to fuse
    (spec off, NGRAM, self-draft without MTP layers). Pure function of the
    configs — config loading is the caller's job (ModelRunner helper) so each
    case's geometry is unit-testable.

    `is_self_draft` = the run has no `speculative_draft_model_path`, so the
    caller built `draft_model_config` from the TARGET checkpoint with
    `is_draft_model=True`. That is what materializes the MTP head (the arch
    swap + `num_nextn_predict_layers` in `ModelConfig._config_draft_model`).
    If the swap did NOT happen the target has no MTP head, and its layer count
    is the FULL model's — fusing that would size the draft region at the whole
    target, so return None instead.
    """
    from sglang.srt.mem_cache.unified_memory_pool import MHARegionGeometry

    if spec_algorithm.is_none() or not spec_algorithm.has_draft_kv():
        return None

    cfg = draft_model_config if draft_model_config is not None else target_model_config

    if is_self_draft and not _is_mtp_draft_arch(cfg):
        return None

    if server_args.enable_multi_layer_eagle:
        # One ModelRunner per draft step, each loading its own MTP module.
        # All step-runners share one config, so per-step geometric identity
        # holds by construction; the fused draft region concatenates their
        # layers (step i occupies layer sub-range i, keyed by draft_model_idx).
        num_steps = int(server_args.speculative_num_steps or 0)
        per_step = _draft_layer_count(cfg)
        if num_steps <= 0 or per_step is None:
            return None
        layer_num = num_steps * per_step
    elif spec_algorithm.is_dflash():
        from sglang.srt.speculative.dflash_utils import parse_dflash_draft_config

        dflash_config = parse_dflash_draft_config(draft_hf_config=cfg.hf_config)
        layer_num = int(dflash_config.require_num_layers())
    elif draft_model_config is not None:
        layer_num = _draft_layer_count(draft_model_config)
        if layer_num is None:
            return None
    else:
        # MTP/NEXTN self-draft: no MTP layers -> no draft KV to fuse.
        nnpl = target_model_config.num_nextn_predict_layers
        if nnpl is None or int(nnpl) <= 0:
            return None
        layer_num = int(nnpl)

    head_dim = int(cfg.head_dim)
    v_head_dim = int(cfg.v_head_dim) if cfg.v_head_dim is not None else head_dim
    geometry = MHARegionGeometry(
        layer_num=layer_num,
        head_num=int(cfg.get_num_kv_heads(attn_tp_size)),
        head_dim=head_dim,
        v_head_dim=v_head_dim,
        store_dtype=kv_cache_dtype,
    )
    geometry.validate()
    return geometry


class DraftBackendFactory:
    def __init__(
        self,
        server_args: ServerArgs,
        draft_model_runner,
        topk: int,
        speculative_num_steps: int,
    ):
        self.server_args = server_args
        self.draft_model_runner = draft_model_runner
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        self.draft_attn_backend = server_args.speculative_draft_attention_backend

    def _create_backend(
        self, backend_name: str, backend_map: dict, error_template: str
    ):
        backend_type = (
            self.draft_attn_backend
            if self.draft_attn_backend
            else getattr(self.server_args, backend_name)
        )
        if backend_type is None:
            backend_type = self.server_args.attention_backend

        if backend_type not in backend_map:
            raise ValueError(error_template.format(backend_type=backend_type))

        return backend_map[backend_type]()

    def create_decode_backend(self):
        # No multi-step draft backend for steps=0 (nospec) or steps=1.
        if self.speculative_num_steps <= 1:
            return None

        backend_map = {
            "flashinfer": self._create_flashinfer_decode_backend,
            "triton": self._create_triton_decode_backend,
            "aiter": self._create_aiter_decode_backend,
            "fa3": self._create_fa3_decode_backend,
            "hybrid_linear_attn": (
                self._create_fa3_decode_backend
                if not is_blackwell()
                else self._create_triton_decode_backend
            ),
            "flashmla": self._create_flashmla_decode_backend,
            "trtllm_mha": self._create_trtllm_mha_decode_backend,
            "trtllm_mla": self._create_trtllm_mla_decode_backend,
            "cutedsl_mla": self._create_cutedsl_mla_decode_backend,
            "tokenspeed_mla": self._create_tokenspeed_mla_decode_backend,
            "dsa": self._create_dsa_decode_backend,
            "nsa": self._create_dsa_decode_backend,  # Deprecated alias for "dsa"
            "ascend": self._create_ascend_decode_backend,
            "fa4": self._create_fa4_decode_backend,
            "dsv4": self._create_dsv4_decode_backend,
        }

        return self._create_backend(
            "decode_attention_backend",
            backend_map,
            "EAGLE is not supported in decode attention backend {backend_type}",
        )

    def create_draft_extend_backend(self):
        backend_map = {
            "flashinfer": self._create_flashinfer_prefill_backend,
            "triton": self._create_triton_prefill_backend,
            "aiter": self._create_aiter_prefill_backend,
            "fa3": self._create_fa3_prefill_backend,
            "hybrid_linear_attn": (
                self._create_fa3_prefill_backend
                if not is_blackwell()
                else self._create_triton_prefill_backend
            ),
            "flashmla": self._create_flashmla_prefill_backend,
            "trtllm_mha": self._create_trtllm_mha_prefill_backend,
            "trtllm_mla": self._create_trtllm_mla_prefill_backend,
            # cute-dsl MLA only supports decode; draft-extend falls back to trtllm-gen.
            "cutedsl_mla": self._create_trtllm_mla_prefill_backend,
            "tokenspeed_mla": self._create_tokenspeed_mla_prefill_backend,
            "dsa": self._create_dsa_prefill_backend,
            "nsa": self._create_dsa_prefill_backend,  # Deprecated alias for "dsa"
            "ascend": self._create_ascend_prefill_backend,
            "fa4": self._create_fa4_prefill_backend,
            "dsv4": self._create_dsv4_prefill_backend,
        }
        backend_name = (
            "decode_attention_backend"
            if self.server_args.speculative_attention_mode == "decode"
            else "prefill_attention_backend"
        )
        return self._create_backend(
            backend_name,
            backend_map,
            "EAGLE is not supported in attention backend {backend_type}",
        )

    def _create_dsa_decode_backend(self):
        from sglang.srt.layers.attention.dsa_backend import (
            DeepseekSparseAttnMultiStepBackend,
        )

        return DeepseekSparseAttnMultiStepBackend(
            self.draft_model_runner, self.topk, self.speculative_num_steps
        )

    def _create_dsa_prefill_backend(self):
        from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend

        return DeepseekSparseAttnBackend(self.draft_model_runner, skip_prefill=False)

    def _create_flashinfer_decode_backend(self):
        if not get_global_server_args().use_mla_backend:
            from sglang.srt.layers.attention.flashinfer_backend import (
                FlashInferMultiStepDraftBackend,
            )

            return FlashInferMultiStepDraftBackend(
                self.draft_model_runner, self.topk, self.speculative_num_steps
            )
        else:
            from sglang.srt.layers.attention.flashinfer_mla_backend import (
                FlashInferMLAMultiStepDraftBackend,
            )

            return FlashInferMLAMultiStepDraftBackend(
                self.draft_model_runner, self.topk, self.speculative_num_steps
            )

    def _create_triton_decode_backend(self):
        from sglang.srt.layers.attention.triton_backend import (
            TritonMultiStepDraftBackend,
        )

        return TritonMultiStepDraftBackend(
            self.draft_model_runner, self.topk, self.speculative_num_steps
        )

    def _create_aiter_decode_backend(self):
        from sglang.srt.layers.attention.aiter_backend import AiterMultiStepDraftBackend

        return AiterMultiStepDraftBackend(
            self.draft_model_runner, self.topk, self.speculative_num_steps
        )

    def _create_fa_decode_backend(self, fa_impl_ver: int = 3):
        if not is_musa():
            from sglang.srt.layers.attention.flashattention_backend import (
                FlashAttentionMultiStepBackend,
            )
        else:
            from sglang.srt.hardware_backend.musa.attention.flashattention_backend import (
                MusaFlashAttentionMultiStepBackend as FlashAttentionMultiStepBackend,
            )

        return FlashAttentionMultiStepBackend(
            self.draft_model_runner,
            self.topk,
            self.speculative_num_steps,
            fa_impl_ver=fa_impl_ver,
        )

    def _create_fa3_decode_backend(self):
        return self._create_fa_decode_backend(fa_impl_ver=3)

    def _create_fa4_decode_backend(self):
        return self._create_fa_decode_backend(fa_impl_ver=4)

    def _create_flashmla_decode_backend(self):
        from sglang.srt.layers.attention.flashmla_backend import (
            FlashMLAMultiStepDraftBackend,
        )

        return FlashMLAMultiStepDraftBackend(
            self.draft_model_runner, self.topk, self.speculative_num_steps
        )

    def _create_trtllm_mha_decode_backend(self):
        from sglang.srt.layers.attention.trtllm_mha_backend import (
            TRTLLMHAAttnMultiStepDraftBackend,
        )

        return TRTLLMHAAttnMultiStepDraftBackend(
            self.draft_model_runner, self.topk, self.speculative_num_steps
        )

    def _create_trtllm_mla_decode_backend(self, backend: str = "trtllm-gen"):
        if not get_global_server_args().use_mla_backend:
            raise ValueError(
                "trtllm_mla backend requires MLA model (use_mla_backend=True)."
            )

        from sglang.srt.layers.attention.trtllm_mla_backend import (
            TRTLLMMLAMultiStepDraftBackend,
        )

        return TRTLLMMLAMultiStepDraftBackend(
            self.draft_model_runner,
            self.topk,
            self.speculative_num_steps,
            backend=backend,
        )

    def _create_cutedsl_mla_decode_backend(self):
        return self._create_trtllm_mla_decode_backend(backend="cute-dsl")

    def _create_tokenspeed_mla_decode_backend(self):
        if not get_global_server_args().use_mla_backend:
            raise ValueError(
                "tokenspeed_mla backend requires MLA model (use_mla_backend=True)."
            )

        from sglang.srt.layers.attention.tokenspeed_mla_backend import (
            TokenspeedMLAMultiStepDraftBackend,
        )

        return TokenspeedMLAMultiStepDraftBackend(
            self.draft_model_runner, self.topk, self.speculative_num_steps
        )

    def _create_ascend_decode_backend(self):
        from sglang.srt.hardware_backend.npu.attention.ascend_backend import (
            AscendAttnMultiStepDraftBackend,
        )

        return AscendAttnMultiStepDraftBackend(
            self.draft_model_runner, self.topk, self.speculative_num_steps
        )

    def _create_dsv4_decode_backend(self):
        # On NPU the "dsv4" backend resolves to the Ascend V4 subclass; its
        # draft path reuses the Ascend multi-step draft backend.
        if is_npu():
            return self._create_ascend_decode_backend()
        elif is_hip():
            from sglang.srt.layers.attention.deepseek_v4_backend_hip_radix import (
                DeepseekV4MultiStepBackend,
            )
        else:
            from sglang.srt.layers.attention.deepseek_v4_backend import (
                DeepseekV4MultiStepBackend,
            )

        return DeepseekV4MultiStepBackend(
            self.draft_model_runner, self.topk, self.speculative_num_steps
        )

    def _create_flashinfer_prefill_backend(self):
        if not get_global_server_args().use_mla_backend:
            from sglang.srt.layers.attention.flashinfer_backend import (
                FlashInferAttnBackend,
            )

            return FlashInferAttnBackend(self.draft_model_runner, skip_prefill=False)
        else:
            from sglang.srt.layers.attention.flashinfer_mla_backend import (
                FlashInferMLAAttnBackend,
            )

            return FlashInferMLAAttnBackend(self.draft_model_runner, skip_prefill=False)

    def _create_triton_prefill_backend(self):
        from sglang.srt.layers.attention.triton_backend import TritonAttnBackend

        return TritonAttnBackend(self.draft_model_runner, skip_prefill=False)

    def _create_aiter_prefill_backend(self):
        from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend

        return AiterAttnBackend(self.draft_model_runner, skip_prefill=False)

    def _create_fa_prefill_backend(self, fa_impl_ver: int = 3):
        if not is_musa():
            from sglang.srt.layers.attention.flashattention_backend import (
                FlashAttentionBackend,
            )
        else:
            from sglang.srt.hardware_backend.musa.attention.flashattention_backend import (
                MusaFlashAttentionBackend as FlashAttentionBackend,
            )
        return FlashAttentionBackend(
            self.draft_model_runner, skip_prefill=False, fa_impl_ver=fa_impl_ver
        )

    def _create_fa3_prefill_backend(self):
        return self._create_fa_prefill_backend(fa_impl_ver=3)

    def _create_fa4_prefill_backend(self):
        return self._create_fa_prefill_backend(fa_impl_ver=4)

    def _create_trtllm_mha_prefill_backend(self):
        from sglang.srt.layers.attention.trtllm_mha_backend import TRTLLMHAAttnBackend

        return TRTLLMHAAttnBackend(self.draft_model_runner, skip_prefill=False)

    def _create_trtllm_mla_prefill_backend(self):
        if not get_global_server_args().use_mla_backend:
            raise ValueError(
                "trtllm_mla backend requires MLA model (use_mla_backend=True)."
            )

        from sglang.srt.layers.attention.trtllm_mla_backend import TRTLLMMLABackend

        return TRTLLMMLABackend(self.draft_model_runner, skip_prefill=False)

    def _create_tokenspeed_mla_prefill_backend(self):
        if not get_global_server_args().use_mla_backend:
            raise ValueError(
                "tokenspeed_mla backend requires MLA model (use_mla_backend=True)."
            )

        from sglang.srt.layers.attention.tokenspeed_mla_backend import (
            TokenspeedMLABackend,
        )

        return TokenspeedMLABackend(self.draft_model_runner, skip_prefill=False)

    def _create_ascend_prefill_backend(self):
        from sglang.srt.hardware_backend.npu.attention.ascend_backend import (
            AscendAttnBackend,
        )

        return AscendAttnBackend(self.draft_model_runner)

    def _create_flashmla_prefill_backend(self):
        logger.warning(
            "flashmla prefill backend is not yet supported for draft extend."
        )
        return None

    def _create_dsv4_prefill_backend(self):
        # On NPU the "dsv4" backend resolves to the Ascend V4 subclass; its
        # draft-extend path reuses the Ascend prefill draft backend.
        if is_npu():
            return self._create_ascend_prefill_backend()
        elif is_hip():
            from sglang.srt.layers.attention.deepseek_v4_backend_hip_radix import (
                DeepseekV4HipRadixBackend,
            )

            return DeepseekV4HipRadixBackend(
                self.draft_model_runner, skip_prefill=False
            )
        from sglang.srt.layers.attention.deepseek_v4_backend import (
            DeepseekV4AttnBackend,
        )

        return DeepseekV4AttnBackend(self.draft_model_runner, skip_prefill=False)
