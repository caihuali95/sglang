from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

import torch
import triton

from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.distributed.parallel_state import get_dcp_group
from sglang.srt.environ import envs
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.triton_ops.kv_indices import (
    create_flashinfer_kv_indices_triton,
)
from sglang.srt.layers.attention.triton_ops.metadata import get_num_kv_splits_triton
from sglang.srt.layers.attention.utils import (
    cp_lse_ag_out_rs,
    create_triton_kv_indices_for_dcp_triton,
    get_dcp_lens,
)
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.mem_cache.memory_pool import KVWriteLoc
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.model_executor.cuda_graph_config import cuda_graph_fully_disabled
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.runtime_context import get_parallel
from sglang.srt.speculative.spec_utils import (
    draft_kv_indices_buffer_width,
    draft_kv_indices_used_len,
    generate_draft_decode_kv_indices,
)
from sglang.srt.utils import (
    get_bool_env_var,
    get_device_core_count,
    get_int_env_var,
    is_cuda,
    is_gfx95_supported,
    is_gfx942_supported,
    is_xpu,
    next_power_of_2,
)

_is_cuda = is_cuda()
_is_gfx942 = is_gfx942_supported()
_is_xpu = is_xpu()

if _is_cuda:
    from sgl_kernel.utils import is_arch_support_pdl

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.spec_info import SpecInput


_MLA_DECODE_MIN_BLOCK_KV = 32


def _mla_decode_kv_splits_cap(
    base_max_kv_splits: int, sm_count: int, max_context_len: int
) -> int:
    if sm_count <= 0:
        return base_max_kv_splits
    sm_cap = next_power_of_2(sm_count)
    ctx_cap = next_power_of_2(triton.cdiv(max_context_len, _MLA_DECODE_MIN_BLOCK_KV))
    return max(base_max_kv_splits, min(sm_cap, ctx_cap))


def logit_capping_mod(logit_capping_method, logit_cap):
    # positive logit_cap -> tanh cap
    if logit_capping_method == "tanh":
        return logit_cap
    else:
        raise ValueError()


@dataclass
class ForwardMetadata:
    attn_logits: torch.Tensor
    attn_lse: torch.Tensor
    max_extend_len: int
    num_kv_splits: torch.Tensor
    kv_indptr: torch.Tensor
    kv_indices: torch.Tensor
    qo_indptr: torch.Tensor
    custom_mask: torch.Tensor
    mask_indptr: torch.Tensor
    # Sliding window
    window_kv_indptr: torch.Tensor
    window_kv_indices: torch.Tensor
    window_num_kv_splits: torch.Tensor
    window_kv_offsets: torch.Tensor
    # Separate attn_logits for SWA layers when v_head_dim differs
    swa_attn_logits: Optional[torch.Tensor] = None
    # full->SWA translated out_cache_loc (SWA KV-store write target)
    swa_out_cache_loc: Optional[torch.Tensor] = None
    # PHYSICAL full-attn write target for the unified pool (eager: translated tensor;
    # cuda-graph: capture-stable buffer view). None for non-unified pools.
    out_cache_loc_full_physical: Optional[torch.Tensor] = None


class TritonAttnBackend(AttentionBackend):
    # CUDA-graph replay rebuilds metadata from preallocated kv_indptr/kv_indices
    # buffers; it never reads seq_lens_cpu / seq_lens_sum.
    needs_cpu_seq_lens: bool = False

    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        kv_indptr_buf: Optional[torch.Tensor] = None,
    ):
        # Lazy import to avoid the initialization of cuda context
        from sglang.srt.layers.attention.triton_ops.decode_attention import (
            decode_attention_fwd,
        )
        from sglang.srt.layers.attention.triton_ops.extend_attention import (
            build_unified_kv_indices,
            extend_attention_fwd,
            extend_attention_fwd_unified,
        )
        from sglang.srt.layers.attention.triton_ops.verify_splitkv import (
            verify_splitkv_fwd,
        )

        super().__init__()

        self.decode_attention_fwd = torch.compiler.disable(decode_attention_fwd)
        self.extend_attention_fwd = torch.compiler.disable(extend_attention_fwd)
        self.extend_attention_fwd_unified = torch.compiler.disable(
            extend_attention_fwd_unified
        )
        self.build_unified_kv_indices = torch.compiler.disable(build_unified_kv_indices)
        # Split-KV EAGLE-verify kernel; enabled below once topk is known (valid only at topk == 1).
        self.verify_splitkv_fwd = torch.compiler.disable(verify_splitkv_fwd)

        # Parse args
        self.skip_prefill = skip_prefill
        max_bs = model_runner.req_to_token_pool.size
        self.sliding_window_size = model_runner.sliding_window_size
        self.req_to_token_pool = model_runner.req_to_token_pool
        self.token_to_kv_pool = model_runner.token_to_kv_pool
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.token_to_kv_pool_allocator = model_runner.token_to_kv_pool_allocator
        self.use_sliding_window_kv_pool = isinstance(self.token_to_kv_pool, SWAKVPool)
        # Lets the Triton wrappers specialize on PAGE_SIZE; page_size=1 is
        # byte-identical to the slot-based envelope.
        self.page_size = getattr(model_runner, "page_size", 1) or 1
        # Unified pool v2p hook (None = no-op): req_to_token holds VIRTUAL ids but
        # kernels need PHYSICAL. Applied eagerly so the captured graph has no
        # translate. Every pool bound to a unified allocator needs it — target
        # pools and the fused draft view pool alike, since both ride in the
        # target's relocatable slots. Non-unified allocators have no
        # translate_kv_loc, so the probe degenerates to None. Draft mamba
        # states keep their own translate (_translate_mamba_indices).
        self._translate_kv_loc = getattr(
            self.token_to_kv_pool_allocator, "translate_kv_loc", None
        )
        if self._translate_kv_loc is not None:
            # Unified pool: the CG replay-prep translate needs the exact filled
            # counts (kv_indptr[bs] / the window sum), sourced from
            # seq_lens_sum / seq_lens_cpu. Triton normally opts OUT of the
            # FutureMap's CPU-mirror publication (class attr False), which
            # under sync-free spec-v2 overlap would leave both unset and push
            # EVERY replay onto the blocking `.item()` fallback. Opting in
            # routes the mirror through the FutureMap's overlapped pinned-D2H
            # publication (event-gated private stream) instead — the designed
            # low-cost path (see overlap_utils.resolve_seq_lens_cpu).
            self.needs_cpu_seq_lens = True
        self.num_draft_tokens = model_runner.server_args.speculative_num_draft_tokens
        self.speculative_num_steps = model_runner.server_args.speculative_num_steps
        self.topk = model_runner.server_args.speculative_eagle_topk or 0
        # Split-KV verify is bit-equivalent only for a pure-causal chain (topk==1)
        # and is gfx95-only; else fall back to extend_attention_fwd.
        self.use_verify_splitkv = (
            is_gfx95_supported()
            and envs.SGLANG_ENABLE_SPLITKV_VERIFY.get()
            and self.topk == 1
        )
        self.use_mla = model_runner.model_config.attention_arch == AttentionArch.MLA
        self.dcp_size = getattr(model_runner, "dcp_size", 1)
        self.dcp_rank = getattr(model_runner, "dcp_rank", 0)
        self.num_head = (
            model_runner.model_config.num_attention_heads // get_parallel().attn_tp_size
        ) * self.dcp_size
        self.num_kv_head = model_runner.model_config.get_num_kv_heads(
            get_parallel().attn_tp_size
        )
        # The decode kernel's "// Lv" stride trick requires attn_logits.shape[-1]
        # to exactly match the layer's v_head_dim, so hybrid SWA models with
        # differing SWA/full v_head_dim need a second buffer for SWA layers.
        full_v_head_dim = model_runner.model_config.v_head_dim
        swa_v_head_dim = model_runner.model_config.swa_v_head_dim
        if self.sliding_window_size is not None and swa_v_head_dim != full_v_head_dim:
            self.v_head_dim = full_v_head_dim
            self.swa_v_head_dim = swa_v_head_dim
        elif (
            model_runner.hybrid_gdn_config is not None
            or model_runner.kimi_linear_config is not None
            or model_runner.linear_attn_model_spec is not None
        ):
            # For hybrid linear models, layer_id = 0 may not be full attention
            self.v_head_dim = model_runner.token_to_kv_pool.get_v_head_dim()
            self.swa_v_head_dim = None
        else:
            self.v_head_dim = model_runner.token_to_kv_pool.get_value_buffer(0).shape[
                -1
            ]
            self.swa_v_head_dim = None
        self.max_context_len = model_runner.model_config.context_len
        self.device = model_runner.device
        self.device_core_count = get_device_core_count(model_runner.gpu_id)
        self.static_kv_splits = get_bool_env_var(
            "SGLANG_TRITON_DECODE_ATTN_STATIC_KV_SPLITS", "false"
        )
        self.max_kv_splits = model_runner.server_args.triton_attention_num_kv_splits
        if self.use_mla and not _is_xpu:
            self.max_kv_splits = _mla_decode_kv_splits_cap(
                self.max_kv_splits,
                self.device_core_count,
                self.max_context_len,
            )
            if _is_gfx942:
                # gfx942's 304 CUs round up to 512 splits, doubling the persistent
                # fp32 attn_logits buffer to ~4 GiB on Kimi-K2.6 and faulting in
                # ROCm graph replay; pin to 256 to match validated gfx950 behavior.
                self.max_kv_splits = min(self.max_kv_splits, 256)
        if _is_cuda:
            self.use_pdl = is_arch_support_pdl()
        else:
            self.use_pdl = False

        self.allow_bidirectional_attention_in_extend = (
            cuda_graph_fully_disabled()
            and model_runner.server_args.chunked_prefill_size == -1
        )

        self.enable_deterministic = (
            model_runner.server_args.enable_deterministic_inference
        )

        if self.enable_deterministic:
            # Fixed split tile size for batch invariance
            self.split_tile_size = get_int_env_var(
                "SGLANG_TRITON_DECODE_SPLIT_TILE_SIZE", 256
            )
            self.static_kv_splits = False
        else:
            self.split_tile_size = (
                model_runner.server_args.triton_attention_split_tile_size
            )

        if self.split_tile_size is not None:
            self.max_kv_splits = (
                self.max_context_len + self.split_tile_size - 1
            ) // self.split_tile_size

        assert not (
            model_runner.sliding_window_size is not None
            and model_runner.model_config.is_encoder_decoder
        ), "Sliding window and cross attention are not supported together"

        # TODO(Jianan Ji): verify behavior when kv_indptr_buf is provided and sliding window is enabled
        if kv_indptr_buf is None:
            self.kv_indptr = torch.zeros(
                (max_bs + 1,), dtype=torch.int32, device=model_runner.device
            )
        else:
            self.kv_indptr = kv_indptr_buf

        # Sliding window may need a second buffer for interleaved attention types
        self.window_kv_indptr = None
        if self.sliding_window_size is not None and self.sliding_window_size > 0:
            if kv_indptr_buf is None:
                self.window_kv_indptr = torch.zeros(
                    (max_bs + 1,), dtype=torch.int32, device=model_runner.device
                )
            else:
                self.window_kv_indptr = torch.zeros_like(kv_indptr_buf)

        if not self.skip_prefill:
            self.qo_indptr = torch.zeros(
                (max_bs + 1,), dtype=torch.int64, device=model_runner.device
            )

            self.mask_indptr = torch.zeros(
                (max_bs + 1,), dtype=torch.int64, device=model_runner.device
            )

        self.forward_metadata: ForwardMetadata = None

        self.cuda_graph_custom_mask = None

    def get_num_kv_splits(
        self,
        num_kv_splits: torch.Tensor,
        seq_lens: torch.Tensor,
    ):
        num_token, num_seq = num_kv_splits.shape[0], seq_lens.shape[0]
        # NOTE(alcanderian): Considering speculative_decodeing,
        # num_kv_splits.shape[0] will be topk * real_num_token.
        # And the real_num_token is num_seq in decoding phase.
        num_group = num_token // num_seq

        assert (
            num_group * num_seq == num_token
        ), f"num_seq({num_seq}), num_token({num_token}), something goes wrong!"

        if (
            self.static_kv_splits or self.device_core_count <= 0
        ) and not self.enable_deterministic:
            num_kv_splits.fill_(self.max_kv_splits)
            return

        if self.split_tile_size is not None and self.enable_deterministic:
            if num_group > 1:
                expanded_seq_lens = seq_lens.repeat_interleave(num_group)
            else:
                expanded_seq_lens = seq_lens

            num_kv_splits[:] = (
                expanded_seq_lens + self.split_tile_size - 1
            ) // self.split_tile_size
            return

        if num_seq < 256:
            SCHEDULE_SEQ = 256
        else:
            SCHEDULE_SEQ = triton.next_power_of_2(num_seq)

        get_num_kv_splits_triton[(1,)](
            num_kv_splits,
            seq_lens,
            num_seq,
            num_group,
            self.num_head,
            self.num_kv_head,
            self.max_kv_splits,
            self.device_core_count,
            MAX_NUM_SEQ=SCHEDULE_SEQ,
        )

    def _dcp_lens(self, lens: torch.Tensor, start: Optional[torch.Tensor] = None):
        return get_dcp_lens(lens, self.dcp_size, self.dcp_rank, start)

    def _dcp_kv_indices(
        self,
        req_pool_indices: torch.Tensor,
        lens: torch.Tensor,
        kv_indptr: torch.Tensor,
        kv_indices: Optional[torch.Tensor] = None,
        kv_start_idx: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Build per-DCP-rank sharded KV indptr/indices. eager passes kv_indices=None
        # (fresh tensor); cuda-graph passes an address-stable buffer to fill in place.
        dcp_lens = self._dcp_lens(lens, kv_start_idx)
        kv_indptr[1 : len(req_pool_indices) + 1] = torch.cumsum(dcp_lens, dim=0)
        kv_indptr = kv_indptr[: len(req_pool_indices) + 1]
        if kv_indices is None:
            kv_indices = torch.empty(
                int(dcp_lens.sum().item()), dtype=torch.int64, device=self.device
            )
        create_triton_kv_indices_for_dcp_triton[(len(req_pool_indices),)](
            self.req_to_token,
            req_pool_indices,
            dcp_lens,
            kv_indptr,
            kv_start_idx,
            kv_indices,
            self.req_to_token.stride(0),
            self.dcp_size,
            self.dcp_rank,
        )
        return kv_indptr, kv_indices, dcp_lens

    def _fill_kv_indptr_and_indices(
        self,
        bs: int,
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        kv_indices: torch.Tensor,
    ) -> torch.Tensor:
        kv_indptr = self.kv_indptr[: bs + 1]
        kv_indptr[1:] = torch.cumsum(seq_lens, dim=0)
        create_flashinfer_kv_indices_triton[(bs,)](
            self.req_to_token,
            req_pool_indices,
            seq_lens,
            kv_indptr,
            None,
            kv_indices,
            self.req_to_token.stride(0),
        )
        return kv_indptr

    def _translate_kv_indices_ragged(
        self,
        kv_indices: torch.Tensor,
        kv_indptr: torch.Tensor,
        bs: int,
        filled_len: Optional[int],
    ) -> torch.Tensor:
        """Virtual->physical translate of a ragged kv_indices buffer (unified
        pool read path); no-op when translation is off (baseline / draft).

        `kv_indices` is frequently OVER-ALLOCATED: when the CPU token total is
        unavailable (gpu_only decode, and every spec TARGET_VERIFY batch, whose
        `seq_lens_sum` is None), the caller sizes it to `bs * max_context_len`
        with `torch.empty`, leaving the tail past the ragged frontier
        UNINITIALIZED. The attention kernel reads only `[0, kv_indptr[bs])` via
        the indptr, so that tail is harmless to it -- but `index_select` touches
        every element, and an uninitialized tail id >= the v2p length trips a
        device-side assert ("scatter gather kernel index out of bounds"). So
        translate ONLY the `[0, filled_len)` prefix that
        `_fill_kv_indptr_and_indices` actually wrote, and return it (downstream
        reads are indptr-bounded, so a shorter buffer is equivalent).

        `filled_len` is the exact ragged length when the caller already knows it
        without a sync (`forward_batch.seq_lens_sum` / a CPU prefix-sum); None
        falls back to reading `kv_indptr[bs]` (one D2H sync -- only reached on
        mirror-less batches, and strictly better than the crash it replaces).
        Translating the real prefix keeps the check honest: a genuinely
        out-of-range id inside the written region still asserts.
        """
        if self._translate_kv_loc is None:
            return kv_indices
        if filled_len is None:
            filled_len = int(kv_indptr[bs].item())
        return self._translate_kv_loc(kv_indices[:filled_len])

    def _update_decode_kv_buffers(
        self,
        bs: int,
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
    ):
        """Fill KV (and SWA) cuda-graph buffers for decode/idle mode.

        Returns ``(kv_indptr, window_kv_indptr, window_kv_lens, num_kv_splits_lens)``
        where ``window_kv_lens`` is ``None`` when sliding-window is disabled and
        ``num_kv_splits_lens`` is the per-request length used to size kv splits
        (per-DCP-rank length clamped to >=1 when DCP is enabled, full seq_lens
        otherwise).
        """
        seq_lens = seq_lens[:bs]
        req_pool_indices = req_pool_indices[:bs]
        if self.dcp_size > 1:
            # DCP: per-rank sharded; write into the same cuda-graph buffers
            # _build_cuda_graph_forward_metadata reads back.
            _, _, dcp_seq_lens = self._dcp_kv_indices(
                req_pool_indices,
                seq_lens,
                self.kv_indptr,
                self.cuda_graph_kv_indices,
                None,
            )
            kv_indptr = self.kv_indptr[: bs + 1]
            num_kv_splits_lens = dcp_seq_lens.clamp_min(1)
        else:
            kv_indptr = self._fill_kv_indptr_and_indices(
                bs, seq_lens, req_pool_indices, self.cuda_graph_kv_indices
            )
            # Unified pool: VIRTUAL ids written here are translated to PHYSICAL in
            # init_forward_metadata_out_graph (replay-prep) so the captured graph
            # carries zero translate nodes.
            num_kv_splits_lens = seq_lens
        window_kv_indptr = self.window_kv_indptr
        window_kv_lens = None
        if self.sliding_window_size is not None and self.sliding_window_size > 0:
            # Unified pool: leave the window VIRTUAL too (translated alongside the
            # full kv_indices later); baseline SWA keeps the eager window translate.
            window_kv_indptr, _, window_kv_lens, _ = update_sliding_window_buffer(
                self.window_kv_indptr,
                self.req_to_token,
                self.sliding_window_size,
                seq_lens,
                req_pool_indices,
                bs,
                token_to_kv_pool=self.token_to_kv_pool,
                window_kv_indices=self.cuda_graph_window_kv_indices,
                skip_full_to_swa_translation=(self._translate_kv_loc is not None),
            )
        return kv_indptr, window_kv_indptr, window_kv_lens, num_kv_splits_lens

    def _update_target_verify_buffers(
        self,
        bs: int,
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        spec_info,
    ):
        """Fill all cuda-graph buffers for target_verify mode."""
        qo_indptr = self.qo_indptr[: bs + 1]
        qo_indptr[: bs + 1] = torch.arange(
            0,
            (1 + bs) * self.num_draft_tokens,
            step=self.num_draft_tokens,
            dtype=torch.int32,
            device=self.device,
        )
        kv_indptr = self._fill_kv_indptr_and_indices(
            bs, seq_lens, req_pool_indices, self.cuda_graph_kv_indices
        )
        window_kv_indptr = self.window_kv_indptr
        window_kv_indices = None
        window_num_kv_splits = None
        window_kv_offsets = None
        if self.sliding_window_size is not None and self.sliding_window_size > 0:
            window_kv_indices = self.cuda_graph_window_kv_indices
            window_num_kv_splits = self.cuda_graph_window_num_kv_splits
            window_kv_offsets = self.cuda_graph_window_kv_offsets
            window_kv_indptr, window_kv_indices, _, window_kv_offsets[:bs] = (
                update_sliding_window_buffer(
                    self.window_kv_indptr,
                    self.req_to_token,
                    self.sliding_window_size,
                    seq_lens[:bs],
                    req_pool_indices,
                    bs,
                    token_to_kv_pool=self.token_to_kv_pool,
                    window_kv_indices=window_kv_indices,
                    # Unified pool: leave the window VIRTUAL here — the deferred
                    # cuda-graph translate (_translate_cuda_graph_shared_pool_locs)
                    # applies full->swa exactly once. Without this the window
                    # would be translated TWICE (here + deferred), reading wrong
                    # slots. Mirrors _update_decode_kv_buffers.
                    skip_full_to_swa_translation=(self._translate_kv_loc is not None),
                )
            )
        custom_mask = self.cuda_graph_custom_mask
        if (
            spec_info is not None
            and getattr(spec_info, "custom_mask", None) is not None
        ):
            # Clamp to the static buffer: a worst-case-sized source mask
            # (draft_token_num*(max_context_len + draft_token_num) per row,
            # built when the draft loop has neither seq_lens_sum nor a
            # preallocated mask buffer) exceeds this buffer by exactly
            # bs*draft_token_num**2. The excess tail is never indexed —
            # mask_indptr[-1] = sum(draft*(seq+draft)) over real rows stays
            # within max_num_tokens*max_context_len for any admissible batch —
            # so truncating is lossless while the unclamped copy hard-crashes
            # replay prep with a size-mismatch RuntimeError.
            _mask_numel = min(spec_info.custom_mask.shape[0], custom_mask.shape[0])
            custom_mask[:_mask_numel] = spec_info.custom_mask[:_mask_numel]
        else:
            custom_mask = None
        seq_mask_len = self.num_draft_tokens * (seq_lens + self.num_draft_tokens)
        mask_indptr = self.mask_indptr[: bs + 1]
        mask_indptr[1 : bs + 1] = torch.cumsum(seq_mask_len, dim=0)
        return (
            qo_indptr,
            kv_indptr,
            custom_mask,
            mask_indptr,
            window_kv_indptr,
            window_kv_indices,
            window_num_kv_splits,
            window_kv_offsets,
        )

    def _update_draft_extend_buffers(
        self,
        bs: int,
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
    ):
        """Fill QO + KV cuda-graph buffers for draft_extend mode."""
        seq_lens = seq_lens[:bs]
        # V2 draft-extend fills num_draft_tokens per req; num_steps+1 only equals
        # that when topk == 1.
        num_tokens_per_bs = (
            self.num_draft_tokens
            if forward_mode.is_draft_extend_v2()
            else self.speculative_num_steps + 1
        )
        qo_indptr = self.qo_indptr[: bs + 1]
        qo_indptr[: bs + 1] = torch.arange(
            0,
            bs * num_tokens_per_bs + 1,
            step=num_tokens_per_bs,
            dtype=torch.int32,
            device=self.device,
        )
        # DRAFT_EXTEND_V2: kv_indptr/kv_indices cover only the prefix (extend K/V go
        # separately). Capture warmup lacks extend_seq_lens_tensor -> fall back to
        # zeros; clamp at 0 so padded rows (seq_lens==fill 1) don't go negative.
        if (
            spec_info is not None
            and getattr(spec_info, "extend_seq_lens_tensor", None) is not None
        ):
            extend_seq_lens = spec_info.extend_seq_lens_tensor[:bs].to(torch.int32)
        else:
            extend_seq_lens = torch.zeros(bs, dtype=torch.int32, device=seq_lens.device)
        kv_lens = torch.clamp(seq_lens - extend_seq_lens, min=0).to(torch.int32)
        kv_indptr = self._fill_kv_indptr_and_indices(
            bs, kv_lens, req_pool_indices, self.cuda_graph_kv_indices
        )
        return qo_indptr, kv_indptr, num_tokens_per_bs

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ):
        bs = forward_batch.batch_size
        req_pool_indices = forward_batch.req_pool_indices
        seq_lens = forward_batch.seq_lens
        forward_mode = forward_batch.forward_mode
        spec_info = forward_batch.spec_info

        if in_capture:
            assert forward_batch.encoder_lens is None, "Not supported"
            # Multi-step spec decode: kv buffers come from spec_info, not the
            # cuda-graph pool, so replay is not involved.
            if forward_mode.is_decode_or_idle() and spec_info is not None:
                self.forward_metadata = ForwardMetadata(
                    attn_logits=self.cuda_graph_attn_logits,
                    attn_lse=self.cuda_graph_attn_lse,
                    max_extend_len=None,
                    num_kv_splits=self.cuda_graph_num_kv_splits,
                    kv_indptr=spec_info.kv_indptr,
                    kv_indices=spec_info.kv_indices,
                    qo_indptr=None,
                    custom_mask=None,
                    mask_indptr=None,
                    window_kv_indptr=self.window_kv_indptr,
                    window_kv_indices=None,
                    window_num_kv_splits=None,
                    window_kv_offsets=None,
                    swa_attn_logits=self.cuda_graph_swa_attn_logits,
                )
                return

            self._apply_cuda_graph_metadata(
                bs=bs,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                forward_mode=forward_mode,
                spec_info=spec_info,
            )
            out_cache_loc_full_physical = self._translate_cuda_graph_shared_pool_locs(
                forward_batch, bs
            )
            swa_out_cache_loc = self._fill_cuda_graph_swa_out_cache_loc(
                forward_batch, in_capture=True
            )
            self.forward_metadata = self._build_cuda_graph_forward_metadata(
                bs,
                forward_mode,
                spec_info,
                swa_out_cache_loc,
                out_cache_loc_full_physical,
            )
        else:
            self._apply_cuda_graph_metadata(
                bs=bs,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                forward_mode=forward_mode,
                spec_info=spec_info,
            )
            # Physical-loc contract tripwire (replay only — capture batches
            # are runner-built with zero-filled static buffers and carry no
            # flag): a live ForwardBatch that skipped the rebind would feed
            # virtual ids into the captured store as if physical.
            if (
                self._translate_kv_loc is not None
                and forward_batch.out_cache_loc is not None
            ):
                assert getattr(forward_batch, "out_cache_loc_is_physical", False), (
                    "unified pool: forward_batch.out_cache_loc is not physical "
                    "— the ForwardBatch was built without "
                    "apply_unified_kv_loc_rebind"
                )
            # Metadata view is reused from capture; just refill the buffers.
            self._translate_cuda_graph_shared_pool_locs(forward_batch, bs)
            self._fill_cuda_graph_swa_out_cache_loc(forward_batch)

    def _fill_cuda_graph_swa_out_cache_loc(
        self, forward_batch: ForwardBatch, in_capture: bool = False
    ) -> Optional[torch.Tensor]:
        """Refill the SWA write-target buffer, returning the [:n] view (None for
        non-SWA / multi-step draft) so the captured store reads fresh slots on
        replay."""
        if not self.use_sliding_window_kv_pool:
            return None
        out_cache_loc = forward_batch.out_cache_loc
        if (
            out_cache_loc is None
            or out_cache_loc.shape[0] > self.cuda_graph_swa_out_cache_loc.shape[0]
        ):
            return None
        n = out_cache_loc.shape[0]
        self.cuda_graph_swa_out_cache_loc[n:].zero_()
        if self._translate_kv_loc is not None:
            # Unified pool: the swa-physical rail is prepared at ForwardBatch
            # construction (apply_unified_kv_loc_rebind) — out_cache_loc here
            # is already FULL-physical, so running the virtual->swa map on it
            # would produce garbage. Capture-time batches are runner-built
            # from zero-filled static buffers and carry no rail; zero-fill is
            # exactly what the old translate-of-zeros produced (slot 0 is the
            # reserved sink in both id spaces).
            if in_capture:
                self.cuda_graph_swa_out_cache_loc[:n].zero_()
            else:
                swa_loc = getattr(forward_batch, "swa_out_cache_loc", None)
                assert swa_loc is not None, (
                    "unified hybrid-SWA replay: forward_batch.swa_out_cache_loc "
                    "missing — the ForwardBatch was built without "
                    "apply_unified_kv_loc_rebind"
                )
                self.cuda_graph_swa_out_cache_loc[:n].copy_(swa_loc)
        else:
            # Non-unified SWA pool: out_cache_loc is physical-full already;
            # the static full->swa mapping applies (upstream-verbatim).
            self.cuda_graph_swa_out_cache_loc[:n].copy_(
                self.token_to_kv_pool.translate_loc_from_full_to_swa(out_cache_loc)
            )
        return self.cuda_graph_swa_out_cache_loc[:n]

    def _translate_window_kv_indices(
        self, window_kv_indices: torch.Tensor
    ) -> torch.Tensor:
        """Sliding-window READ ids (virtual) -> physical.

        Two pool families need different maps, and the discriminator is the
        POOL, not the model: `sliding_window_size` merely says the MODEL has
        window layers.
          * Hybrid-SWA pools keep the window in a SEPARATE swa sub-pool with
            its own id space -> `translate_loc_from_full_to_swa`.
          * Single-pool models whose window layers read the SAME slots as the
            full layers -> the ordinary v2p translate. This is the fused draft
            view pool: a DFLASH draft has sliding-window layers but
            its KV lives entirely in the target's fused full-pool slots, so its
            window ids are plain full-pool ids. Before the translate flip the
            draft skipped translation altogether, which is why this path was
            never exercised: it raised AttributeError under cuda-graph capture,
            and silently read VIRTUAL ids in eager mode.
        """
        swa_translate = getattr(
            self.token_to_kv_pool, "translate_loc_from_full_to_swa", None
        )
        if swa_translate is not None:
            out = swa_translate(window_kv_indices)
        elif self._translate_kv_loc is not None:
            out = self._translate_kv_loc(window_kv_indices)
        else:
            return window_kv_indices  # non-unified: ids are already physical
        # Preserve the caller's index dtype: `translate_loc_from_full_to_swa`
        # returns INT32, and callers that REBIND (rather than assign in place
        # into a static int64 buffer) would otherwise silently downcast their
        # int64 index tensor. See `_translate_window_kv_indices_single_pool`.
        if out.dtype != window_kv_indices.dtype:
            out = out.to(window_kv_indices.dtype)
        return out

    def _translate_window_kv_indices_single_pool(
        self, window_kv_indices: torch.Tensor
    ) -> torch.Tensor:
        """Eager-path window translate for SINGLE-pool models only.

        Pools that own a separate SWA sub-pool are translated IN PLACE inside
        `update_sliding_window_buffer` (unchanged, pre-Stage-4.1 behavior) — do
        NOT re-route them through the returning helper: that translate yields
        int32, and rebinding the caller's int64 index tensor to it silently
        downcast the eager indices, corrupting the window read (the cuda-graph
        path was spared only because it assigns in place into a static int64
        buffer, which preserves the dtype).

        Only the fused draft view pool needs this: it has no SWA sub-pool, so
        `update_sliding_window_buffer`'s hasattr guard leaves its window ids
        VIRTUAL, and they need the ordinary v2p translate.
        """
        if hasattr(self.token_to_kv_pool, "translate_loc_from_full_to_swa"):
            return window_kv_indices  # already translated in place by the util
        return self._translate_window_kv_indices(window_kv_indices)

    def _translate_cuda_graph_shared_pool_locs(
        self, forward_batch: ForwardBatch, bs: int
    ) -> Optional[torch.Tensor]:
        """Unified pool: eager refill of the cuda-graph read+write LOC buffers,
        run BEFORE graph.replay() so the captured graph carries zero translate
        nodes. No-op for non-unified pools.

        READ buffers (full kv_indices, SWA window) are v2p-translated IN PLACE
        (req_to_token-derived indices are VIRTUAL — the read rail's translate
        lives here, reading the live post-compaction v2p). The full-attn WRITE
        loc is already PHYSICAL (rebound at ForwardBatch construction by
        apply_unified_kv_loc_rebind) and is COPIED into the backend-owned
        out_cache_loc_full_physical buffer, RETURNED as the [:n] view. Eager
        .item() bounds are fine here (out-of-graph), so no in-graph translate
        variant is needed.
        """
        if self._translate_kv_loc is None:
            return None
        # Full-attention read path. kv_indptr[bs] == sum(padded seq_lens) ==
        # forward_batch.seq_lens_sum, a CPU int the replay batch already carries
        # (the replay wrapper adds the padded rows' fill contribution, keeping
        # it equal to kv_indptr[bs]); read it instead of a per-step D2H
        # `.item()` sync.
        #
        # The `.item()` fallbacks below CANNOT become asserts: gpu_only batches
        # legitimately leave seq_lens_sum None AND seq_lens_cpu unset
        # (forward_batch_info keeps them sync-free, see #26738). Worse, the
        # replay wrapper hands out `buffers.seq_lens_cpu[:bs]` unconditionally
        # while the buffer is only REFRESHED when the batch carries a CPU
        # mirror — so under gpu_only the tensor is non-None but STALE. Use
        # `seq_lens_sum is not None` as the single freshness signal for BOTH
        # CPU-derived counts (the two fields are set/unset together).
        # Steady state never falls back: with the unified pool the backend
        # opts into needs_cpu_seq_lens (see __init__), so the FutureMap
        # publishes fresh mirrors even under sync-free spec-v2 overlap; the
        # fallback covers only mirror-less direct constructions (bench-style
        # gpu_only batches).
        cpu_mirrors_fresh = forward_batch.seq_lens_sum is not None
        n_kv = (
            forward_batch.seq_lens_sum
            if cpu_mirrors_fresh
            else int(self.kv_indptr[bs].item())
        )
        if n_kv > 0:
            self.cuda_graph_kv_indices[:n_kv] = self._translate_kv_loc(
                self.cuda_graph_kv_indices[:n_kv]
            )
        # SWA window read path (hybrid-SWA unified pools only). window_kv_indptr[bs]
        # == sum(min(seq_len, window)); compute it from the CPU seq_lens mirror
        # when fresh, else fall back to the authoritative device counter.
        if self.sliding_window_size is not None and self.sliding_window_size > 0:
            if cpu_mirrors_fresh and forward_batch.seq_lens_cpu is not None:
                n_win = int(
                    forward_batch.seq_lens_cpu[:bs]
                    .clamp(max=self.sliding_window_size)
                    .sum()
                )
            else:
                n_win = int(self.window_kv_indptr[bs].item())
            if n_win > 0:
                self.cuda_graph_window_kv_indices[:n_win] = (
                    self._translate_window_kv_indices(
                        self.cuda_graph_window_kv_indices[:n_win]
                    )
                )
        # Full-attention write path: out_cache_loc is ALREADY PHYSICAL (rebound
        # at ForwardBatch construction by apply_unified_kv_loc_rebind; at
        # capture the runner-built batch holds zeros, and copy-of-zeros ==
        # translate-of-zeros because slot 0 is the reserved sink in both id
        # spaces). Copy it into the capture-stable buffer and RETURN the [:n]
        # view.
        out_cache_loc = forward_batch.out_cache_loc
        n = out_cache_loc.shape[0]
        # Zero the padded tail first: a smaller replay batch leaves [n:] holding
        # stale ids that the captured store would write; send them to slot 0 (sink).
        self.cuda_graph_out_cache_loc_full_physical[n:].zero_()
        self.cuda_graph_out_cache_loc_full_physical[:n].copy_(out_cache_loc)
        return self.cuda_graph_out_cache_loc_full_physical[:n]

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Init auxiliary variables for triton attention backend."""

        bs = forward_batch.batch_size
        window_kv_indptr = self.window_kv_indptr
        window_kv_indices = None
        window_num_kv_splits = None
        window_kv_offsets = None
        swa_attn_logits = None
        spec_info = forward_batch.spec_info

        if forward_batch.forward_mode.is_decode_or_idle():
            if spec_info is None or spec_info.kv_indptr is None:
                # kv_indptr is None for draft-extend's idle batch; build from seq_lens.
                if self.dcp_size > 1:
                    # DCP: per-rank sharded KV indices, else each rank reads the
                    # whole KV instead of its owner shard.
                    kv_indptr, kv_indices, _ = self._dcp_kv_indices(
                        forward_batch.req_pool_indices,
                        forward_batch.seq_lens,
                        self.kv_indptr,
                    )
                else:
                    # gpu_only: seq_lens_sum may be None; over-allocate is safe
                    # (ragged write + prefix-only translate).
                    real_total = forward_batch.seq_lens_sum
                    seq_lens_sum = (
                        real_total
                        if real_total is not None
                        else bs * self.max_context_len
                    )
                    kv_indices = torch.empty(
                        seq_lens_sum, dtype=torch.int64, device=self.device
                    )
                    kv_indptr = self._fill_kv_indptr_and_indices(
                        bs,
                        forward_batch.seq_lens,
                        forward_batch.req_pool_indices,
                        kv_indices,
                    )
                    kv_indices = self._translate_kv_indices_ragged(
                        kv_indices, kv_indptr, bs, real_total
                    )
                if (
                    self.sliding_window_size is not None
                    and self.sliding_window_size > 0
                ):
                    window_kv_indptr, window_kv_indices, window_kv_lens, _ = (
                        update_sliding_window_buffer(
                            self.window_kv_indptr,
                            self.req_to_token,
                            self.sliding_window_size,
                            forward_batch.seq_lens,
                            forward_batch.req_pool_indices,
                            bs,
                            self.device,
                            self.token_to_kv_pool,
                        )
                    )
                    # Pools WITH an swa sub-pool were translated in place above
                    # (unchanged). Only the single-pool fused draft needs v2p.
                    window_kv_indices = self._translate_window_kv_indices_single_pool(
                        window_kv_indices
                    )
                    window_num_kv_splits = torch.empty(
                        (bs,), dtype=torch.int32, device=self.device
                    )
                    self.get_num_kv_splits(window_num_kv_splits, window_kv_lens)
            else:
                kv_indptr, kv_indices = spec_info.kv_indptr, spec_info.kv_indices
                bs = kv_indptr.shape[0] - 1

            attn_logits = torch.empty(
                (bs, self.num_head, self.max_kv_splits, self.v_head_dim),
                dtype=torch.float32,
                device=self.device,
            )
            if self.swa_v_head_dim is not None:
                swa_attn_logits = torch.empty(
                    (bs, self.num_head, self.max_kv_splits, self.swa_v_head_dim),
                    dtype=torch.float32,
                    device=self.device,
                )
            else:
                swa_attn_logits = None
            attn_lse = torch.empty(
                (bs, self.num_head, self.max_kv_splits),
                dtype=torch.float32,
                device=self.device,
            )
            num_kv_splits = torch.empty((bs,), dtype=torch.int32, device=self.device)
            self.get_num_kv_splits(
                num_kv_splits,
                (
                    self._dcp_lens(forward_batch.seq_lens).clamp_min(1)
                    if self.dcp_size > 1
                    else forward_batch.seq_lens
                ),
            )

            qo_indptr = None
            custom_mask = None
            mask_indptr = None
            max_extend_len = None
        elif forward_batch.forward_mode.is_target_verify():
            bs = len(forward_batch.req_pool_indices)
            qo_indptr = torch.arange(
                0,
                (1 + bs) * self.num_draft_tokens,
                step=self.num_draft_tokens,
                dtype=torch.int32,
                device=self.device,
            )
            # gpu_only: seq_lens_sum is None for spec TARGET_VERIFY batches
            # (prepare_for_verify builds them without it) -> over-allocate +
            # prefix-only translate.
            real_total = forward_batch.seq_lens_sum
            seq_lens_sum = (
                real_total
                if real_total is not None
                else bs * self.max_context_len
            )
            if self._translate_kv_loc is not None and real_total is not None:
                # TARGET_VERIFY's seq_lens_sum may be an UPPER BOUND, not the
                # exact ragged total: DFLASH publishes host seq_lens as
                # prefix + one verify block while the GPU seq_lens (which
                # drive the ragged fill) stay at the committed prefix
                # (dflash_worker_v2 "Verify host bound"). The translate below
                # covers [0, real_total), so a [filled, real_total)
                # uninitialized tail would device-assert in index_select.
                # Zero-init: virtual id 0 is always in-range (sink slot) and
                # the attention read is kv_indptr-bounded regardless.
                kv_indices = torch.zeros(
                    seq_lens_sum, dtype=torch.int64, device=self.device
                )
            else:
                kv_indices = torch.empty(
                    seq_lens_sum, dtype=torch.int64, device=self.device
                )
            kv_indptr = self._fill_kv_indptr_and_indices(
                bs,
                forward_batch.seq_lens,
                forward_batch.req_pool_indices,
                kv_indices,
            )
            # Unified pool read path: req_to_token rows hold VIRTUAL ids;
            # translate to physical for the verify attention read (mirrors the
            # decode/extend branches — this branch was the missing one). Only the
            # ragged prefix is translated: the over-allocated tail is
            # uninitialized and would trip an index_select device assert.
            kv_indices = self._translate_kv_indices_ragged(
                kv_indices, kv_indptr, bs, real_total
            )

            if self.sliding_window_size is not None and self.sliding_window_size > 0:
                # window_kv_offsets gives the start position in custom mask
                (
                    window_kv_indptr,
                    window_kv_indices,
                    window_kv_lens,
                    window_kv_offsets,
                ) = update_sliding_window_buffer(
                    self.window_kv_indptr,
                    self.req_to_token,
                    self.sliding_window_size,
                    forward_batch.seq_lens,
                    forward_batch.req_pool_indices,
                    bs,
                    self.device,
                    self.token_to_kv_pool,
                )
                window_kv_indices = self._translate_window_kv_indices_single_pool(
                    window_kv_indices
                )

            custom_mask = spec_info.custom_mask
            seq_mask_len = self.num_draft_tokens * (
                forward_batch.seq_lens + self.num_draft_tokens
            )
            mask_indptr = self.mask_indptr
            mask_indptr[1 : bs + 1] = torch.cumsum(seq_mask_len[:bs], dim=0)
            mask_indptr = mask_indptr[: bs + 1]
            max_extend_len = self.num_draft_tokens
            num_kv_splits = None
            attn_logits = None
            attn_lse = None

        else:
            if self.dcp_size > 1:
                kv_indptr, kv_indices, _ = self._dcp_kv_indices(
                    forward_batch.req_pool_indices,
                    forward_batch.extend_prefix_lens,
                    self.kv_indptr,
                )
            else:
                # gpu_only leaves _cpu unset; over-allocate is safe (ragged write
                # + prefix-only translate).
                if forward_batch.extend_prefix_lens_cpu is not None:
                    real_total = sum(forward_batch.extend_prefix_lens_cpu)
                else:
                    real_total = None
                kv_indices_len = (
                    real_total
                    if real_total is not None
                    else bs * self.max_context_len
                )
                kv_indices = torch.empty(
                    kv_indices_len,
                    dtype=torch.int64,
                    device=self.device,
                )
                kv_indptr = self._fill_kv_indptr_and_indices(
                    bs,
                    forward_batch.extend_prefix_lens,
                    forward_batch.req_pool_indices,
                    kv_indices,
                )
                kv_indices = self._translate_kv_indices_ragged(
                    kv_indices, kv_indptr, bs, real_total
                )
            if self.sliding_window_size is not None and self.sliding_window_size > 0:
                (
                    window_kv_indptr,
                    window_kv_indices,
                    window_kv_lens,
                    window_kv_offsets,
                ) = update_sliding_window_buffer(
                    self.window_kv_indptr,
                    self.req_to_token,
                    self.sliding_window_size,
                    forward_batch.extend_prefix_lens,
                    forward_batch.req_pool_indices,
                    bs,
                    self.device,
                    self.token_to_kv_pool,
                )
                window_kv_indices = self._translate_window_kv_indices_single_pool(
                    window_kv_indices
                )

            qo_indptr = self.qo_indptr
            qo_indptr[1 : bs + 1] = torch.cumsum(forward_batch.extend_seq_lens, dim=0)
            qo_indptr = qo_indptr[: bs + 1]
            custom_mask = None
            mask_indptr = None
            attn_logits = None
            attn_lse = None
            # Defensive GPU-max fallback when extend_seq_lens_cpu is absent.
            if forward_batch.extend_seq_lens_cpu is not None:
                max_extend_len = max(forward_batch.extend_seq_lens_cpu)
            else:
                max_extend_len = int(forward_batch.extend_seq_lens.max())
            num_kv_splits = None

        swa_out_cache_loc = None
        if self.use_sliding_window_kv_pool and forward_batch.out_cache_loc is not None:
            if self._translate_kv_loc is not None:
                # Unified pool: the swa-physical rail was computed from the
                # VIRTUAL loc at ForwardBatch construction
                # (apply_unified_kv_loc_rebind) — out_cache_loc here is
                # already FULL-physical, so the virtual->swa map must not run
                # on it.
                swa_out_cache_loc = forward_batch.swa_out_cache_loc
                assert swa_out_cache_loc is not None, (
                    "unified hybrid-SWA: forward_batch.swa_out_cache_loc "
                    "missing — the ForwardBatch was built without "
                    "apply_unified_kv_loc_rebind"
                )
            else:
                # Non-unified SWA pool: out_cache_loc is physical-full; the
                # static full->swa mapping applies (upstream-verbatim).
                swa_out_cache_loc = self.token_to_kv_pool.translate_loc_from_full_to_swa(
                    forward_batch.out_cache_loc
                )

        # Unified pool full-attention WRITE loc: ALREADY PHYSICAL (rebound at
        # ForwardBatch construction by apply_unified_kv_loc_rebind), carried in
        # the metadata (-> KVWriteLoc.full_loc). None for non-unified pools.
        out_cache_loc_full_physical = None
        if (
            self._translate_kv_loc is not None
            and forward_batch.out_cache_loc is not None
        ):
            # Physical-loc contract tripwire: a hand-built ForwardBatch that
            # skipped the rebind would write virtual ids as if physical.
            assert getattr(forward_batch, "out_cache_loc_is_physical", False), (
                "unified pool: forward_batch.out_cache_loc is not physical — "
                "the ForwardBatch was built without apply_unified_kv_loc_rebind"
            )
            out_cache_loc_full_physical = forward_batch.out_cache_loc

        self.forward_metadata = ForwardMetadata(
            attn_logits,
            attn_lse,
            max_extend_len,
            num_kv_splits,
            kv_indptr,
            kv_indices,
            qo_indptr,
            custom_mask,
            mask_indptr,
            window_kv_indptr,
            window_kv_indices,
            window_num_kv_splits,
            window_kv_offsets,
            swa_attn_logits=swa_attn_logits,
            swa_out_cache_loc=swa_out_cache_loc,
            out_cache_loc_full_physical=out_cache_loc_full_physical,
        )

    def init_cuda_graph_state(
        self,
        max_bs: int,
        max_num_tokens: int,
        kv_indices_buf: Optional[torch.Tensor] = None,
        cuda_graph_num_kv_splits_buf: Optional[torch.Tensor] = None,
    ):
        self.cuda_graph_attn_logits = torch.zeros(
            (max_num_tokens, self.num_head, self.max_kv_splits, self.v_head_dim),
            dtype=torch.float32,
            device=self.device,
        )
        if self.swa_v_head_dim is not None:
            self.cuda_graph_swa_attn_logits = torch.zeros(
                (
                    max_num_tokens,
                    self.num_head,
                    self.max_kv_splits,
                    self.swa_v_head_dim,
                ),
                dtype=torch.float32,
                device=self.device,
            )
        else:
            self.cuda_graph_swa_attn_logits = None
        self.cuda_graph_attn_lse = torch.zeros(
            (max_num_tokens, self.num_head, self.max_kv_splits),
            dtype=torch.float32,
            device=self.device,
        )

        if cuda_graph_num_kv_splits_buf is None:
            self.cuda_graph_num_kv_splits = torch.full(
                (max_num_tokens,),
                self.max_kv_splits,
                dtype=torch.int32,
                device=self.device,
            )
        else:
            self.cuda_graph_num_kv_splits = cuda_graph_num_kv_splits_buf

        if kv_indices_buf is None:
            self.cuda_graph_kv_indices = torch.zeros(
                (max_num_tokens * self.max_context_len),
                dtype=torch.int64,
                device=self.device,
            )
        else:
            self.cuda_graph_kv_indices = kv_indices_buf

        if not self.skip_prefill:
            self.cuda_graph_custom_mask = torch.zeros(
                (max_num_tokens * self.max_context_len),
                dtype=torch.uint8,
                device=self.device,
            )

        if self.sliding_window_size is not None and self.sliding_window_size > 0:
            if kv_indices_buf is None:
                self.cuda_graph_window_kv_indices = torch.zeros(
                    (max_num_tokens * self.sliding_window_size),
                    dtype=torch.int64,
                    device=self.device,
                )
            else:
                self.cuda_graph_window_kv_indices = torch.zeros_like(kv_indices_buf)

            self.cuda_graph_window_num_kv_splits = torch.full(
                (max_num_tokens,),
                self.max_kv_splits,
                dtype=torch.int32,
                device=self.device,
            )

            self.cuda_graph_window_kv_offsets = torch.zeros(
                (max_bs,),
                dtype=torch.int32,
                device=self.device,
            )

        if self.use_sliding_window_kv_pool:
            # SWA write-target buffer; refilled at replay from out_cache_loc.
            self.cuda_graph_swa_out_cache_loc = torch.zeros(
                (max_num_tokens,),
                dtype=torch.int64,
                device=self.device,
            )

        if self._translate_kv_loc is not None:
            # Unified pool full-attention write-target buffer, refilled at replay
            # (-> KVWriteLoc.full_loc). Capture-stable, mirrors cuda_graph_swa_out_cache_loc.
            self.cuda_graph_out_cache_loc_full_physical = torch.zeros(
                (max_num_tokens,),
                dtype=torch.int64,
                device=self.device,
            )

    def _build_cuda_graph_forward_metadata(
        self,
        bs: int,
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
        swa_out_cache_loc: Optional[torch.Tensor] = None,
        out_cache_loc_full_physical: Optional[torch.Tensor] = None,
    ) -> ForwardMetadata:
        """Construct ForwardMetadata from the current cuda-graph buffer state.

        Called by capture after the buffer-update helpers have already run
        (either via replay or directly).  All fields reference the same
        self.cuda_graph_* tensors that the captured graph kernels will
        read — the Python object is rebuilt each capture, but the underlying
        GPU memory addresses are stable. ``swa_out_cache_loc`` is the
        pre-allocated SWA write-target buffer view (or None for non-SWA).
        """
        swa = self.sliding_window_size is not None and self.sliding_window_size > 0
        if forward_mode.is_decode_or_idle():
            return ForwardMetadata(
                attn_logits=self.cuda_graph_attn_logits,
                attn_lse=self.cuda_graph_attn_lse,
                max_extend_len=None,
                num_kv_splits=self.cuda_graph_num_kv_splits,
                kv_indptr=self.kv_indptr[: bs + 1],
                kv_indices=self.cuda_graph_kv_indices,
                qo_indptr=None,
                custom_mask=None,
                mask_indptr=None,
                window_kv_indptr=self.window_kv_indptr[: bs + 1] if swa else None,
                window_kv_indices=self.cuda_graph_window_kv_indices if swa else None,
                window_num_kv_splits=(
                    self.cuda_graph_window_num_kv_splits if swa else None
                ),
                window_kv_offsets=None,
                swa_attn_logits=self.cuda_graph_swa_attn_logits,
                swa_out_cache_loc=swa_out_cache_loc,
                out_cache_loc_full_physical=out_cache_loc_full_physical,
            )
        elif forward_mode.is_target_verify():
            custom_mask = (
                self.cuda_graph_custom_mask
                if spec_info is not None
                and getattr(spec_info, "custom_mask", None) is not None
                else None
            )
            return ForwardMetadata(
                attn_logits=None,
                attn_lse=None,
                max_extend_len=self.num_draft_tokens,
                num_kv_splits=None,
                kv_indptr=self.kv_indptr[: bs + 1],
                kv_indices=self.cuda_graph_kv_indices,
                qo_indptr=self.qo_indptr[: bs + 1],
                custom_mask=custom_mask,
                mask_indptr=self.mask_indptr[: bs + 1],
                window_kv_indptr=self.window_kv_indptr[: bs + 1] if swa else None,
                window_kv_indices=self.cuda_graph_window_kv_indices if swa else None,
                window_num_kv_splits=(
                    self.cuda_graph_window_num_kv_splits if swa else None
                ),
                window_kv_offsets=self.cuda_graph_window_kv_offsets if swa else None,
                swa_out_cache_loc=swa_out_cache_loc,
                out_cache_loc_full_physical=out_cache_loc_full_physical,
            )
        elif forward_mode.is_draft_extend_v2():
            return ForwardMetadata(
                attn_logits=None,
                attn_lse=None,
                # Must match the per-req query count (num_tokens_per_bs) used to
                # build qo_indptr above, else the extend kernel grid is too small
                # for topk > 1 (num_draft_tokens > num_steps+1) and drops query
                # blocks.
                max_extend_len=(
                    self.num_draft_tokens
                    if forward_mode.is_draft_extend_v2()
                    else self.speculative_num_steps + 1
                ),
                num_kv_splits=None,
                kv_indptr=self.kv_indptr[: bs + 1],
                kv_indices=self.cuda_graph_kv_indices,
                qo_indptr=self.qo_indptr[: bs + 1],
                custom_mask=None,
                mask_indptr=None,
                window_kv_indptr=self.window_kv_indptr,
                window_kv_indices=None,
                window_num_kv_splits=None,
                window_kv_offsets=None,
                swa_out_cache_loc=swa_out_cache_loc,
                out_cache_loc_full_physical=out_cache_loc_full_physical,
            )
        else:
            raise ValueError(f"Invalid forward mode: {forward_mode=} for CUDA Graph.")

    def _apply_cuda_graph_metadata(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
    ):
        """Shared capture+replay body for the cuda-graph init path.

        Public entry: :py:meth:`init_forward_metadata_out_graph`.
        """
        # NOTE: encoder_lens expected to be zeros or None
        if forward_mode.is_decode_or_idle():
            assert spec_info is None, "Multi-step cuda graph init is not done here."
            _, _, window_kv_lens, num_kv_splits_lens = self._update_decode_kv_buffers(
                bs, seq_lens, req_pool_indices
            )
            self.get_num_kv_splits(
                self.cuda_graph_num_kv_splits[:bs], num_kv_splits_lens[:bs]
            )
            if window_kv_lens is not None:
                self.get_num_kv_splits(
                    self.cuda_graph_window_num_kv_splits[:bs], window_kv_lens[:bs]
                )
        elif forward_mode.is_target_verify():
            bs = len(req_pool_indices)
            self._update_target_verify_buffers(
                bs, seq_lens, req_pool_indices, spec_info
            )
        elif forward_mode.is_draft_extend_v2():
            self._update_draft_extend_buffers(
                bs, seq_lens, req_pool_indices, forward_mode, spec_info
            )
        else:
            raise ValueError(
                f"Invalid forward mode: {forward_mode=} for CUDA Graph replay."
            )

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    def get_verify_buffers_to_fill_after_draft(self):
        """
        Return buffers for verify attention kernels that needs to be filled after draft.

        Typically, these are tree mask and position buffers.
        """
        return [self.cuda_graph_custom_mask, None]

    def update_verify_buffers_to_fill_after_draft(
        self, spec_info: SpecInput, cuda_graph_bs: Optional[int]
    ):
        pass

    def _set_kv_buffer(
        self,
        forward_batch: ForwardBatch,
        layer: RadixAttention,
        loc_info,
        k: torch.Tensor,
        v: torch.Tensor,
        k_scale=None,
        v_scale=None,
    ) -> None:
        # DCP writes to the local physical shard (loc = out_cache_loc //
        # dcp_size) through the masked path so each rank only stores the tokens
        # it owns. Non-DCP keeps the original write loc and plain set_kv_buffer.
        if self.dcp_size > 1:
            loc = forward_batch.out_cache_loc // self.dcp_size
            if (
                forward_batch.positions is not None
                and forward_batch.positions.numel() == loc.numel()
            ):
                dcp_kv_mask = forward_batch.positions % self.dcp_size == self.dcp_rank
            else:
                dcp_kv_mask = forward_batch.dcp_kv_mask
            kwargs = {"dcp_kv_mask": dcp_kv_mask}
        else:
            loc = loc_info
            kwargs = {}
        if k_scale is None and v_scale is None:
            self.token_to_kv_pool.set_kv_buffer(layer, loc, k, v, **kwargs)
        else:
            self.token_to_kv_pool.set_kv_buffer(
                layer, loc, k, v, k_scale, v_scale, **kwargs
            )

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        sinks=None,
    ):
        # TODO: reuse the buffer across layers
        attn_out = getattr(forward_batch, "_attn_output", None)
        if attn_out is not None:
            o = attn_out
        elif layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        if k is None and v is None:
            pool = self.token_to_kv_pool
            cache_loc = forward_batch.out_cache_loc
            if isinstance(pool, SWAKVPool) and pool.layers_mapping[layer.layer_id][1]:
                cache_loc = pool.translate_loc_from_full_to_swa(cache_loc)
            elif self._translate_kv_loc is not None:
                # Unified pool: `out_cache_loc` is VIRTUAL and `get_kv_buffer`
                # returns a page-major (4-D) view — a bare `k_buffer[cache_loc]`
                # here would index the PAGE axis with an untranslated id, a
                # silent-corruption read. No unified-pool model shares KV across
                # layers (k/v are always recomputed, so this branch is dead for
                # them); fail loud rather than corrupt. A future KV-sharing
                # unified model must add a translated, page-aware gather here.
                raise RuntimeError(
                    "unified memory pool: cross-layer KV-sharing read "
                    "(k is None) is unsupported — get_kv_buffer returns a "
                    "page-major (4-D) view that a flat cache_loc cannot index "
                    "(even the physical out_cache_loc needs a page-aware "
                    "gather). No current unified model reaches this path."
                )
            k_buffer, v_buffer = pool.get_kv_buffer(layer.layer_id)
            k = k_buffer[cache_loc]
            v = v_buffer[cache_loc]
        elif k is None or v is None:
            raise ValueError("Both k and v should be None or not None")
        else:
            # Save KV cache first (must do this before unified kernel)
            if save_kv_cache:
                loc_info = KVWriteLoc(
                    forward_batch.out_cache_loc,
                    self.forward_metadata.swa_out_cache_loc,
                    full_loc=self.forward_metadata.out_cache_loc_full_physical,
                )
                if layer.k_scale is None:
                    self._set_kv_buffer(forward_batch, layer, loc_info, k, v)
                elif self.use_mla:
                    # For MLA, scale K manually before storing since MLATokenToKVPool
                    # doesn't accept scale parameters. Clone to protect k from mutation
                    # since it's used later in the attention kernel.
                    k_scaled = k.clone().div_(layer.k_scale)
                    self.token_to_kv_pool.set_kv_buffer(
                        layer,
                        loc_info,
                        k_scaled,
                        v,
                    )
                else:
                    self._set_kv_buffer(
                        forward_batch,
                        layer,
                        loc_info,
                        k.clone(),  # cloned to protect k,v from in-place mutation in set_kv_buffer
                        v.clone(),
                        layer.k_scale,
                        layer.v_scale,
                    )

        logits_soft_cap = logit_capping_mod(layer.logit_capping_method, layer.logit_cap)

        causal = True
        if (
            layer.is_cross_attention
            or layer.attn_type == AttentionType.ENCODER_ONLY
            or (
                layer.attn_type == AttentionType.DECODER_BIDIRECTIONAL
                and self.allow_bidirectional_attention_in_extend
            )
        ):
            causal = False

        if self.dcp_size > 1:
            return self._forward_extend_dcp(
                q, k, v, layer, forward_batch, causal, logits_soft_cap, sinks
            )

        # Deterministic mode: use unified 1-stage kernel
        if self.enable_deterministic:
            return self._forward_extend_unified(
                q, o, layer, forward_batch, causal, logits_soft_cap, sinks
            )

        # Normal mode: use original 2-stage kernel
        if layer.sliding_window_size is not None and layer.sliding_window_size > -1:
            sliding_window_size = (
                layer.sliding_window_size
            )  # Needed for sliding window mask
            kv_indptr = self.forward_metadata.window_kv_indptr
            kv_indices = self.forward_metadata.window_kv_indices
            window_kv_offsets = self.forward_metadata.window_kv_offsets
        else:
            sliding_window_size = -1
            kv_indptr = self.forward_metadata.kv_indptr
            kv_indices = self.forward_metadata.kv_indices
            window_kv_offsets = None

        if layer.k_scale is not None and layer.v_scale is not None:
            k_descale = layer.k_scale_float
            v_descale = layer.v_scale_float
        else:
            k_descale = 1.0
            v_descale = 1.0

        # Split-KV EAGLE-verify fast path (ROCm/Triton). On target-verify
        # (topk=1 causal chain), run the bandwidth-efficient split-KV kernel
        # instead of the serial-prefix extend kernel. verify_splitkv_fwd()
        # returns True if it ran (o written), or False for any case it cannot
        # serve bit-equivalently (its can_handle() gates on non-causal / sinks /
        # sliding-window / ragged / topk>1), so we fall through to
        # extend_attention_fwd below. Correctness is never at risk.
        if (
            self.use_verify_splitkv
            and forward_batch.forward_mode.is_target_verify()
            and self.verify_splitkv_fwd(
                q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
                k.contiguous(),
                v.contiguous(),
                o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
                self.token_to_kv_pool.get_key_buffer(layer.layer_id),
                self.token_to_kv_pool.get_value_buffer(layer.layer_id),
                self.forward_metadata.qo_indptr,
                kv_indptr,
                kv_indices,
                self.forward_metadata.custom_mask,
                causal,
                self.forward_metadata.mask_indptr,
                self.forward_metadata.max_extend_len,
                k_descale,
                v_descale,
                layer.scaling,
                logit_cap=logits_soft_cap,
                sliding_window_size=sliding_window_size,
                sinks=sinks,
                window_kv_offsets=window_kv_offsets,
                xai_temperature_len=layer.xai_temperature_len,
                max_bs=self.req_to_token_pool.size,
            )
        ):
            return o

        self.extend_attention_fwd(
            q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
            k.contiguous(),
            v.contiguous(),
            o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
            self.token_to_kv_pool.get_key_buffer(layer.layer_id),
            self.token_to_kv_pool.get_value_buffer(layer.layer_id),
            self.forward_metadata.qo_indptr,
            kv_indptr,
            kv_indices,
            self.forward_metadata.custom_mask,
            causal,
            self.forward_metadata.mask_indptr,
            self.forward_metadata.max_extend_len,
            k_descale,
            v_descale,
            layer.scaling,
            logit_cap=logits_soft_cap,
            sliding_window_size=sliding_window_size,
            sinks=sinks,
            window_kv_offsets=window_kv_offsets,
            xai_temperature_len=layer.xai_temperature_len,
            page_size=self.page_size,
        )
        return o

    def _forward_extend_dcp(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        causal: bool,
        logits_soft_cap: float,
        sinks: Optional[torch.Tensor],
    ):
        if sinks is not None:
            raise NotImplementedError("DCP Triton extend does not support sinks")
        if self.forward_metadata.custom_mask is not None:
            raise NotImplementedError("DCP Triton extend does not support custom masks")
        if layer.sliding_window_size is not None and layer.sliding_window_size > -1:
            raise NotImplementedError(
                "DCP Triton extend does not support sliding window"
            )

        group = get_dcp_group()
        q_local = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim).contiguous()
        total_tokens, local_heads, _ = q_local.shape

        kv_indptr = self.forward_metadata.kv_indptr
        kv_indices = self.forward_metadata.kv_indices
        max_extend_len = self.forward_metadata.max_extend_len

        if layer.k_scale is not None and layer.v_scale is not None:
            k_descale = layer.k_scale_float
            v_descale = layer.v_scale_float
        else:
            k_descale = 1.0
            v_descale = 1.0

        k_buffer = self.token_to_kv_pool.get_key_buffer(layer.layer_id)
        v_buffer = self.token_to_kv_pool.get_value_buffer(layer.layer_id)

        current_out = torch.zeros(
            (total_tokens, local_heads, layer.v_head_dim),
            device=q.device,
            dtype=torch.float32,
        )
        current_lse = torch.full(
            (total_tokens, local_heads),
            -float("inf"),
            device=q.device,
            dtype=torch.float32,
        )

        # Current chunk K/V is still local before masked cache write, so it can
        # use the original extend kernel's current-token stage directly.
        if k.numel() > 0:
            empty_kv_indptr = torch.zeros_like(kv_indptr)
            self.extend_attention_fwd(
                q_local,
                k.contiguous(),
                v.contiguous(),
                current_out,
                k_buffer,
                v_buffer,
                self.forward_metadata.qo_indptr,
                empty_kv_indptr,
                kv_indices[:0],
                None,
                causal,
                None,
                max_extend_len,
                1.0,
                1.0,
                sm_scale=layer.scaling,
                logit_cap=logits_soft_cap,
                xai_temperature_len=layer.xai_temperature_len,
                lse_extend=current_lse,
                skip_prefix=True,
            )

        if kv_indices.numel() == 0:
            return current_out.reshape(-1, layer.tp_q_head_num * layer.v_head_dim).to(
                q.dtype
            )

        # Prefix KV is sharded across DCP ranks, so compute each rank's
        # partial attention with all gathered query heads and merge by LSE.
        q_all = group.all_gather(q_local, dim=1).contiguous()
        total_heads = q_all.shape[1]
        prefix_out = torch.zeros(
            (total_tokens, total_heads, layer.v_head_dim),
            device=q.device,
            dtype=torch.float32,
        )
        prefix_lse = torch.full(
            (total_tokens, total_heads),
            -float("inf"),
            device=q.device,
            dtype=torch.float32,
        )
        empty_k = k[:0].contiguous()
        empty_v = v[:0].contiguous()
        self.extend_attention_fwd(
            q_all,
            empty_k,
            empty_v,
            prefix_out,
            k_buffer,
            v_buffer,
            self.forward_metadata.qo_indptr,
            kv_indptr,
            kv_indices,
            None,
            False,
            None,
            max_extend_len,
            k_descale,
            v_descale,
            sm_scale=layer.scaling,
            logit_cap=logits_soft_cap,
            xai_temperature_len=layer.xai_temperature_len,
            lse_extend=prefix_lse,
            skip_extend=True,
        )

        prefix_out, prefix_lse = cp_lse_ag_out_rs(
            prefix_out, prefix_lse, group, return_lse=True
        )
        final_lse = torch.logaddexp(prefix_lse, current_lse)
        prefix_scale = torch.exp(prefix_lse - final_lse).unsqueeze(-1)
        current_scale = torch.exp(current_lse - final_lse).unsqueeze(-1)
        prefix_scale = torch.nan_to_num(prefix_scale, nan=0.0, posinf=0.0, neginf=0.0)
        current_scale = torch.nan_to_num(current_scale, nan=0.0, posinf=0.0, neginf=0.0)
        out = prefix_out * prefix_scale + current_out * current_scale
        return out.reshape(-1, layer.tp_q_head_num * layer.v_head_dim).to(q.dtype)

    def _forward_extend_unified(
        self,
        q: torch.Tensor,
        o: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        causal: bool,
        logits_soft_cap: float,
        sinks: Optional[torch.Tensor],
    ):
        """
        Unified 1-stage extend attention for deterministic inference.
        Both prefix and extend KV are accessed through unified kv_indices.
        """
        bs = forward_batch.batch_size

        # Determine sliding window settings
        if layer.sliding_window_size is not None and layer.sliding_window_size > -1:
            sliding_window_size = layer.sliding_window_size
            # Note: for unified kernel, we use full kv_indptr (not window)
            prefix_kv_indptr = self.forward_metadata.window_kv_indptr
            prefix_kv_indices = self.forward_metadata.window_kv_indices
            # Compute window start positions (absolute position of first key in window)
            # window_start_pos = seq_len - window_len
            window_kv_lens = prefix_kv_indptr[1 : bs + 1] - prefix_kv_indptr[:bs]
            # Handle TARGET_VERIFY mode where extend_prefix_lens might not be set
            if forward_batch.extend_prefix_lens is not None:
                window_start_pos = (
                    forward_batch.extend_prefix_lens[:bs] - window_kv_lens
                )
            else:
                # Infer from spec_info: prefix_len = seq_len - draft_token_num
                if forward_batch.spec_info is not None and hasattr(
                    forward_batch.spec_info, "draft_token_num"
                ):
                    extend_prefix_lens = (
                        forward_batch.seq_lens[:bs]
                        - forward_batch.spec_info.draft_token_num
                    )
                    window_start_pos = extend_prefix_lens - window_kv_lens
                else:
                    window_start_pos = None
        else:
            sliding_window_size = -1
            prefix_kv_indptr = self.forward_metadata.kv_indptr
            prefix_kv_indices = self.forward_metadata.kv_indices
            window_start_pos = None

        extend_kv_indices = forward_batch.out_cache_loc
        pool = self.token_to_kv_pool
        if (
            layer.sliding_window_size is not None
            and layer.sliding_window_size > -1
            and isinstance(pool, SWAKVPool)
            and pool.layers_mapping[layer.layer_id][1]
        ):
            extend_kv_indices = pool.translate_loc_from_full_to_swa(extend_kv_indices)

        # Handle cases where extend_seq_lens or extend_start_loc might not be set
        # In speculative decoding, we can infer these from spec_info or compute them
        if forward_batch.extend_seq_lens is None:
            # TARGET_VERIFY mode: infer extend_seq_lens from spec_info
            if forward_batch.spec_info is not None and hasattr(
                forward_batch.spec_info, "draft_token_num"
            ):
                draft_token_num = forward_batch.spec_info.draft_token_num
                extend_seq_lens = torch.full(
                    (bs,), draft_token_num, dtype=torch.int32, device=self.device
                )
            else:
                raise RuntimeError(
                    "extend_seq_lens is None but cannot infer from spec_info. "
                    "This should not happen in TARGET_VERIFY mode."
                )
        else:
            extend_seq_lens = forward_batch.extend_seq_lens

        # Check extend_start_loc separately - it might be None even when extend_seq_lens is set
        if forward_batch.extend_start_loc is None:
            # Compute extend_start_loc from extend_seq_lens
            # extend_start_loc[i] = sum(extend_seq_lens[0:i])
            extend_start_loc = torch.cat(
                [
                    torch.zeros(1, dtype=torch.int32, device=self.device),
                    torch.cumsum(extend_seq_lens[:-1], dim=0),
                ]
            )
        else:
            extend_start_loc = forward_batch.extend_start_loc

        unified_kv_indptr, unified_kv_indices, prefix_lens = (
            self.build_unified_kv_indices(
                prefix_kv_indptr,
                prefix_kv_indices,
                extend_start_loc,
                extend_seq_lens,
                extend_kv_indices,
                bs,
            )
        )

        # Convert prefix_lens to int32 for the kernel
        prefix_lens = prefix_lens.to(torch.int32)

        if layer.k_scale is not None and layer.v_scale is not None:
            k_descale = layer.k_scale_float
            v_descale = layer.v_scale_float
        else:
            k_descale = 1.0
            v_descale = 1.0

        # Call unified kernel
        self.extend_attention_fwd_unified(
            q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
            o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
            self.token_to_kv_pool.get_key_buffer(layer.layer_id),
            self.token_to_kv_pool.get_value_buffer(layer.layer_id),
            k_descale,
            v_descale,
            self.forward_metadata.qo_indptr,
            unified_kv_indptr,
            unified_kv_indices,
            prefix_lens,
            self.forward_metadata.max_extend_len,
            custom_mask=self.forward_metadata.custom_mask,
            mask_indptr=self.forward_metadata.mask_indptr,
            sm_scale=layer.scaling,
            logit_cap=logits_soft_cap,
            is_causal=causal,
            sliding_window_size=sliding_window_size,
            sinks=sinks,
            window_start_pos=window_start_pos,
            xai_temperature_len=layer.xai_temperature_len,
            page_size=self.page_size,
        )

        return o

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        sinks=None,
    ):
        # During torch.compile, there is a bug in rotary_emb that causes the
        # output value to have a 3D tensor shape. This reshapes the output correctly.
        q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)

        # TODO: reuse the buffer across layers
        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        logits_soft_cap = logit_capping_mod(layer.logit_capping_method, layer.logit_cap)

        if save_kv_cache:
            if self.use_mla:
                if layer.k_scale is not None:
                    # MLATokenToKVPool doesn't accept scale parameters; k is unused
                    # after this point in decode, so scale in place.
                    k.div_(layer.k_scale)
                # Carry the pre-translated physical loc like the non-MLA branch
                # below (and the MLA extend path): under the unified memory
                # pool `out_cache_loc` is VIRTUAL, and a bare loc would make
                # the pool write to wrong slots silently.
                self.token_to_kv_pool.set_kv_buffer(
                    layer,
                    KVWriteLoc(
                        forward_batch.out_cache_loc,
                        full_loc=self.forward_metadata.out_cache_loc_full_physical,
                    ),
                    k,
                    v,
                )
            else:
                self._set_kv_buffer(
                    forward_batch,
                    layer,
                    KVWriteLoc(
                        forward_batch.out_cache_loc,
                        self.forward_metadata.swa_out_cache_loc,
                        full_loc=self.forward_metadata.out_cache_loc_full_physical,
                    ),
                    k,
                    v,
                    layer.k_scale,
                    layer.v_scale,
                )

        if layer.sliding_window_size is not None and layer.sliding_window_size > -1:
            kv_indptr = self.forward_metadata.window_kv_indptr
            kv_indices = self.forward_metadata.window_kv_indices
        else:
            kv_indptr = self.forward_metadata.kv_indptr
            kv_indices = self.forward_metadata.kv_indices

        if layer.k_scale is not None and layer.v_scale is not None:
            k_descale = layer.k_scale_float
            v_descale = layer.v_scale_float
        else:
            k_descale = 1.0
            v_descale = 1.0

        # Select the correctly-sized attn_logits buffer for this layer.
        # The triton kernel's // Lv stride trick requires attn_logits.shape[-1]
        # to exactly match the layer's v_head_dim.
        attn_logits = self.forward_metadata.attn_logits
        if (
            self.forward_metadata.swa_attn_logits is not None
            and layer.v_head_dim == self.swa_v_head_dim
        ):
            attn_logits = self.forward_metadata.swa_attn_logits

        if self.dcp_size > 1:
            group = get_dcp_group()
            with use_symmetric_memory(group):
                q_for_decode = q.view(
                    -1, layer.tp_q_head_num, layer.qk_head_dim
                ).contiguous()
            q_for_decode = group.all_gather(q_for_decode, dim=1).contiguous()
            o_for_decode = torch.empty(
                (q_for_decode.shape[0], q_for_decode.shape[1], layer.v_head_dim),
                dtype=torch.float32,
                device=q.device,
            )
            self.forward_metadata.attn_lse.fill_(-float("inf"))
            self.decode_attention_fwd(
                q_for_decode,
                self.token_to_kv_pool.get_key_buffer(layer.layer_id),
                self.token_to_kv_pool.get_value_buffer(layer.layer_id),
                o_for_decode,
                kv_indptr,
                kv_indices,
                attn_logits,
                self.forward_metadata.attn_lse,
                self.forward_metadata.num_kv_splits,
                self.max_kv_splits,
                layer.scaling,
                k_descale,
                v_descale,
                logit_cap=logits_soft_cap,
                sinks=sinks,
                xai_temperature_len=layer.xai_temperature_len,
            )
            local_lse = torch.logsumexp(
                self.forward_metadata.attn_lse[
                    : q_for_decode.shape[0], : q_for_decode.shape[1], :
                ],
                dim=-1,
            )
            o = cp_lse_ag_out_rs(o_for_decode, local_lse, group)
            return o.reshape(-1, layer.tp_q_head_num * layer.v_head_dim).to(q.dtype)

        self.decode_attention_fwd(
            q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
            self.token_to_kv_pool.get_key_buffer(layer.layer_id),
            self.token_to_kv_pool.get_value_buffer(layer.layer_id),
            o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
            kv_indptr,
            kv_indices,
            attn_logits,
            self.forward_metadata.attn_lse,
            self.forward_metadata.num_kv_splits,
            self.max_kv_splits,
            layer.scaling,
            k_descale,
            v_descale,
            logit_cap=logits_soft_cap,
            sinks=sinks,
            xai_temperature_len=layer.xai_temperature_len,
            has_mla=self.use_mla,
            use_pdl=self.use_pdl,
            page_size=self.page_size,
        )
        return o


class TritonMultiStepDraftBackend:
    """
    Wrap multiple triton attention backends as one for multiple consecutive
    draft decoding steps.
    """

    needs_cpu_seq_lens: bool = False

    def __init__(
        self,
        model_runner: ModelRunner,
        topk: int,
        speculative_num_steps: int,
    ):
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        max_bs = model_runner.req_to_token_pool.size * self.topk
        self.kv_indptr = torch.zeros(
            (
                self.speculative_num_steps,
                max_bs + 1,
            ),
            dtype=torch.int32,
            device=model_runner.device,
        )
        self.attn_backends: List[TritonAttnBackend] = []
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends.append(
                TritonAttnBackend(
                    model_runner,
                    skip_prefill=True,
                    kv_indptr_buf=self.kv_indptr[i],
                )
            )
        self.max_context_len = self.attn_backends[0].max_context_len
        self.num_head = (
            model_runner.model_config.num_attention_heads // get_parallel().attn_tp_size
        )
        self.device = model_runner.device
        # Cached variables for generate_draft_decode_kv_indices
        self.req_to_token_pool = model_runner.req_to_token_pool
        self.pool_len = model_runner.req_to_token_pool.req_to_token.shape[1]
        self.page_size = model_runner.server_args.page_size
        # Fused draft KV: the READ translate is fused INTO
        # generate_draft_decode_kv_indices (the v2p table rides into the
        # kernel), so all steps' kv_indices emerge PHYSICAL from the single
        # metadata-build/replay-prep launch. The v2p mapping is fixed across
        # the whole spec step (all alloc/free happens scheduler-side before
        # the forward), so one pass covers every step. Non-unified allocators
        # have no v2p table, so this degenerates to None.
        full_mea = getattr(
            model_runner.token_to_kv_pool_allocator, "full_attn_allocator", None
        )
        self._kv_v2p_table = (
            full_mea.virtual_to_physical
            if full_mea is not None and hasattr(full_mea, "virtual_to_physical")
            else None
        )
        self.cuda_graph_out_cache_loc_phys: Optional[torch.Tensor] = None

    def _per_step_out_cache_loc_phys(
        self, forward_batch: ForwardBatch
    ) -> Optional[torch.Tensor]:
        """View the pre-allocated multi-step out_cache_loc per step:
        `(num_steps, bs*topk)` PHYSICAL write locs. The buffer is ALREADY
        physical (rebound once at ForwardBatch construction by
        apply_unified_kv_loc_rebind — same timing as the translate this method
        used to do itself: once, before the step loop, no alloc in between).
        Each step's slice goes into that step's sub-backend metadata — the
        whole buffer must never reach KVWriteLoc (its __post_init__ silently
        truncates a mismatched full_loc to step-0's locs: silent wrong-slot
        writes)."""
        if self._kv_v2p_table is None or forward_batch.out_cache_loc is None:
            return None
        from sglang.srt.speculative.eagle_utils import per_step_draft_out_cache_loc

        assert getattr(forward_batch, "out_cache_loc_is_physical", False), (
            "unified pool: draft multi-step out_cache_loc is not physical — "
            "the ForwardBatch was built without apply_unified_kv_loc_rebind"
        )
        return per_step_draft_out_cache_loc(
            forward_batch.out_cache_loc,
            forward_batch.batch_size,
            self.topk,
            self.speculative_num_steps,
        )

    def common_template(
        self,
        forward_batch: ForwardBatch,
        kv_indices_buffer: Optional[torch.Tensor],
        call_fn: int,
    ):
        if kv_indices_buffer is None:
            kv_indices_buffer = self.cuda_graph_kv_indices

        num_seqs = forward_batch.batch_size
        bs = self.topk * num_seqs
        seq_lens_sum = forward_batch.seq_lens_sum
        if seq_lens_sum is None:
            # seq_lens_sum here only slice-clamps a preallocated kv_indices buffer;
            # over-estimate is safe. Use a static UB to skip the per-iter .sum().item() D2H.
            seq_lens_sum = num_seqs * self.max_context_len

        generate_draft_decode_kv_indices[
            (self.speculative_num_steps, num_seqs, self.topk)
        ](
            forward_batch.req_pool_indices,
            self.req_to_token_pool.req_to_token,
            forward_batch.seq_lens,
            kv_indices_buffer,
            self.kv_indptr,
            forward_batch.positions,
            self.pool_len,
            kv_indices_buffer.shape[1],
            self.kv_indptr.shape[1],
            next_power_of_2(num_seqs),
            next_power_of_2(self.speculative_num_steps),
            next_power_of_2(bs),
            self.page_size,
            v2p_ptr=self._kv_v2p_table,
            HAS_V2P=self._kv_v2p_table is not None,
        )

        if call_fn is None:
            return

        for i in range(self.speculative_num_steps - 1):
            forward_batch.spec_info.kv_indptr = self.kv_indptr[i, : bs + 1]
            forward_batch.spec_info.kv_indices = kv_indices_buffer[i][
                : draft_kv_indices_used_len(seq_lens_sum, self.topk, bs, i + 1)
            ]
            call_fn(i, forward_batch)

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        kv_indices_width = draft_kv_indices_buffer_width(
            forward_batch.batch_size, self.topk, self.max_context_len
        )
        # Zeroed, not empty: `generate_draft_decode_kv_indices` writes only the
        # entries a step actually uses, and the per-step slice below is sized from
        # `seq_lens_sum` -- an UPPER BOUND when the exact sum isn't on hand. Any
        # slack is therefore handed to attention as KV slot ids. Uninitialized
        # there means reading arbitrary slots (another request's KV, or out of
        # range); zeroed means the slot-0 sink, which is what every other
        # padding path in the unified pool already targets. The cuda-graph buffer
        # is zeroed for exactly this reason -- the eager one was not, which is why
        # only the eager path was ever affected.
        kv_indices = torch.zeros(
            (self.speculative_num_steps, kv_indices_width),
            dtype=torch.int64,
            device=self.device,
        )
        # Fused draft KV: per-step PHYSICAL write locs (see helper docstring).
        out_cache_loc_phys_steps = self._per_step_out_cache_loc_phys(forward_batch)

        def call_fn(i, forward_batch):
            forward_batch.spec_info.kv_indptr = (
                forward_batch.spec_info.kv_indptr.clone()
            )
            forward_batch.spec_info.kv_indices = (
                forward_batch.spec_info.kv_indices.clone()
            )
            self.attn_backends[i].init_forward_metadata(forward_batch)
            if out_cache_loc_phys_steps is not None:
                # Overwrite the sub-backend's own full-buffer translate
                # (built from the not-yet-sliced forward_batch.out_cache_loc)
                # with step i's slice; draft_forward later slices
                # forward_batch.out_cache_loc identically.
                self.attn_backends[i].forward_metadata.out_cache_loc_full_physical = (
                    out_cache_loc_phys_steps[i]
                )

        self.common_template(forward_batch, kv_indices, call_fn)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        kv_indices_width = draft_kv_indices_buffer_width(
            max_bs, self.topk, self.max_context_len
        )
        self.cuda_graph_kv_indices = torch.zeros(
            (self.speculative_num_steps, kv_indices_width),
            dtype=torch.int64,
            device=self.device,
        )
        self.cuda_graph_num_kv_splits = torch.full(
            (max_num_tokens,),
            self.attn_backends[0].max_kv_splits,
            dtype=torch.int32,
            device=self.device,
        )

        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_cuda_graph_state(
                max_bs,
                max_num_tokens,
                kv_indices_buf=self.cuda_graph_kv_indices[i],
                cuda_graph_num_kv_splits_buf=self.cuda_graph_num_kv_splits,
            )

        if self._kv_v2p_table is not None:
            # Fused draft KV: capture-stable per-step PHYSICAL write-loc rows.
            # The decode+spec capture branch builds its ForwardMetadata without
            # out_cache_loc_full_physical (there is no such buffer on the
            # plain multi-step path), so the captured KV save would fall back
            # to the VIRTUAL out_cache_loc; each step's captured metadata
            # instead references its row here, refilled at every replay-prep.
            self.cuda_graph_out_cache_loc_phys = torch.zeros(
                (self.speculative_num_steps, max_bs * self.topk),
                dtype=torch.int64,
                device=self.device,
            )

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ):
        from sglang.srt.model_executor.forward_batch_info import build_inner_fb_view

        if in_capture:
            inner_fb = build_inner_fb_view(
                forward_batch,
                bs=forward_batch.batch_size,
                forward_mode=ForwardMode.DECODE,
            )
            capture_num_token = forward_batch.batch_size * self.topk

            def call_fn(i, _forward_batch):
                self.attn_backends[i].init_forward_metadata_out_graph(
                    inner_fb, in_capture=True
                )
                if self.cuda_graph_out_cache_loc_phys is not None:
                    # Reference the capture-stable row; replay-prep refills it.
                    self.attn_backends[i].forward_metadata.out_cache_loc_full_physical = self.cuda_graph_out_cache_loc_phys[
                        i, :capture_num_token
                    ]

            self.common_template(forward_batch, None, call_fn)
        else:
            bs = forward_batch.batch_size
            self.common_template(forward_batch, None, None)

            # NOTE: Multi-step's attention backends use the slice of
            # - kv_indptr buffer (cuda graph and non-cuda graph)
            # - kv_indices buffer (cuda graph only)
            # So we don't need to assign the KV indices inside the attention backend.

            # Fused draft KV: refill the captured per-step write-loc rows.
            # Derive the live token count from the RAW out_cache_loc (the
            # runner hands the outer, unpadded fb here); zero the stale tail
            # first — a smaller replay batch leaves [n:] holding previous
            # replays' physical ids that the captured store would write; send
            # them to the slot-0 sink (same rule as
            # _translate_cuda_graph_shared_pool_locs).
            if (
                self.cuda_graph_out_cache_loc_phys is not None
                and forward_batch.out_cache_loc is not None
            ):
                from sglang.srt.speculative.eagle_utils import (
                    per_step_draft_out_cache_loc,
                )

                live_bs = forward_batch.out_cache_loc.shape[0] // (
                    self.topk * self.speculative_num_steps
                )
                # out_cache_loc is ALREADY PHYSICAL (rebound at ForwardBatch
                # construction); slice it per step directly.
                assert getattr(forward_batch, "out_cache_loc_is_physical", False), (
                    "unified pool: draft multi-step replay out_cache_loc is "
                    "not physical — the ForwardBatch was built without "
                    "apply_unified_kv_loc_rebind"
                )
                phys_steps = per_step_draft_out_cache_loc(
                    forward_batch.out_cache_loc,
                    live_bs,
                    self.topk,
                    self.speculative_num_steps,
                )
                num_token = live_bs * self.topk
                self.cuda_graph_out_cache_loc_phys[:, num_token:].zero_()
                self.cuda_graph_out_cache_loc_phys[:, :num_token].copy_(phys_steps)

            # Compute num_kv_splits only once
            num_token = bs * self.topk
            self.attn_backends[-1].get_num_kv_splits(
                self.attn_backends[-1].cuda_graph_num_kv_splits[:num_token],
                forward_batch.seq_lens[:bs],
            )

    def init_forward_metadata_in_graph(self, forward_batch: ForwardBatch) -> None:
        for attn_backend in self.attn_backends:
            attn_backend.init_forward_metadata_in_graph(forward_batch)


def update_sliding_window_buffer(
    window_kv_indptr,
    req_to_token,
    sliding_window_size,
    seq_lens,
    req_pool_indices,
    bs,
    device=None,
    token_to_kv_pool=None,
    window_kv_indices=None,
    skip_full_to_swa_translation=False,
):
    """Fill window KV buffers for sliding-window attention.

    Pass ``window_kv_indices`` to write into a pre-allocated buffer (CUDA-graph
    path); omit it (or pass ``None``) to allocate a fresh tensor (eager path,
    requires ``device``).

    ``skip_full_to_swa_translation=True`` leaves ``window_kv_indices`` as VIRTUAL
    full-token ids (no eager full->swa translate). The unified-memory-pool cuda-graph
    builder passes this so the window translate is deferred to
    ``TritonAttnBackend._translate_cuda_graph_shared_pool_locs`` (run in
    ``init_forward_metadata_out_graph``, BEFORE ``graph.replay()``), which reads
    the live v2p and rewrites the static window buffer to swa-physical in place;
    baseline SWA leaves it False (eager).
    """
    window_kv_lens = torch.minimum(
        seq_lens,
        torch.tensor(sliding_window_size),
    )
    window_kv_indptr[1 : bs + 1] = torch.cumsum(window_kv_lens, dim=0)
    window_kv_indptr = window_kv_indptr[: bs + 1]
    if window_kv_indices is None:
        window_kv_indices = torch.empty(
            window_kv_indptr[-1], dtype=torch.int64, device=device
        )
    window_kv_start_idx = seq_lens - window_kv_lens
    create_flashinfer_kv_indices_triton[(bs,)](
        req_to_token,
        req_pool_indices,
        window_kv_lens,
        window_kv_indptr,
        window_kv_start_idx,
        window_kv_indices,
        req_to_token.stride(0),
    )
    if not skip_full_to_swa_translation and hasattr(
        token_to_kv_pool, "translate_loc_from_full_to_swa"
    ):
        kv_last_index = window_kv_indptr[-1]
        window_kv_indices[:kv_last_index] = (
            token_to_kv_pool.translate_loc_from_full_to_swa(
                window_kv_indices[:kv_last_index]
            )
        )
    return window_kv_indptr, window_kv_indices, window_kv_lens, window_kv_start_idx
