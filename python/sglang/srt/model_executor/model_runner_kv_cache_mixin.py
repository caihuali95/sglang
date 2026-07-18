from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Optional

import msgspec
import torch

from sglang.srt.configs.model_config import (
    get_dsa_index_head_dim,
    get_minimax_sparse_attention_config,
    get_minimax_sparse_disable_value_layer_ids,
    get_minimax_sparse_layer_ids,
    is_deepseek_dsa,
    is_deepseek_v4,
    is_minimax_sparse,
)
from sglang.srt.distributed.parallel_state import (
    get_world_group,
)
from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.mem_cache.allocator import (
    PagedTokenToKVPoolAllocator,
    TokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.allocator.hisparse import (
    DeepSeekV4HiSparseTokenToKVPoolAllocator,
    HiSparseTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.allocator.swa import (
    PureSWATokenToKVPoolAllocator,
    SWATokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.common import get_req_to_token_extra_context_len
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.mem_cache.hisparse_memory_pool import HiSparseDSATokenToKVPool
from sglang.srt.mem_cache.memory_pool import (
    DSATokenToKVPool,
    HybridLinearKVPool,
    HybridReqToTokenPool,
    MHATokenToKVPool,
    MHATokenToKVPoolFP4,
    MiniMaxSparseKVPool,
    MLATokenToKVPool,
    MLATokenToKVPoolFP4,
    NoOpMHATokenToKVPool,
    PageMajorMHATokenToKVPool,
    ReqToTokenPool,
)
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.platforms import current_platform
from sglang.srt.utils.common import (
    get_available_gpu_memory,
    is_float4_e2m1fn_x2,
    is_hip,
    is_npu,
)

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.model_executor.pool_configurator import MemoryPoolConfig


def _should_enable_lazy_compaction() -> bool:
    """Lazy compaction default — ON unless
    `SGLANG_DISABLE_LAZY_COMPACTION=1` (escape hatch for A/B / rollback).
    Centralized here so both unified-memory-pool factory call sites stay in sync.
    """
    return not envs.SGLANG_DISABLE_LAZY_COMPACTION.get()


# the ratio of mamba cache pool size to max_running_requests
MAMBA_CACHE_SIZE_MAX_RUNNING_REQUESTS_RATIO = 3
MAMBA_CACHE_V2_ADDITIONAL_RATIO_OVERLAP = 2
MAMBA_CACHE_V2_ADDITIONAL_RATIO_OVERLAP_LAZY = 1
MAMBA_CACHE_V2_ADDITIONAL_RATIO_NO_OVERLAP = 1

# Unified-pool joint admission (_handle_max_mamba_cache_unified, spec ON or
# OFF): the per-request token floor charged by the joint hard-floor solve. It
# bounds how far short-context concurrency can squeeze the token pool at boot;
# requests with longer contexts draw on the shared gap and, under pressure,
# the existing retract machinery.
UNIFIED_MIN_TOKENS_PER_REQ = 2048

# Unified-pool admission ceiling used when the SPECULATIVE hook defaulted
# --max-running-requests to 48 (a static-partition tuning, not a user
# decision; under the unified pool the joint byte solve is the real bound).
# Matches the default decode CUDA-graph max_bs; batches beyond the captured
# sizes fall back to eager, and runtime pressure lands on the admission
# charges + check_decode_mem probe->retract machinery.
UNIFIED_MAX_RUNNING_REQUESTS_CEILING = 256

# Spec-OFF analogue: without spec, --max-running-requests has no default at
# all (None), so the solve needs a finite `requested` seed. Matches
# _resolve_max_num_reqs's own `estimated` hard cap — deliberately LARGER than
# the spec ceiling so the byte solve, not this constant, is the binding bound
# (256 here would SHRINK spec-off admission vs the pre-solve sizing on large
# GPUs). _resolve_max_num_reqs still mins with estimated/user/capacity//2.
UNIFIED_SPEC_OFF_REQ_CEILING = 4096


class UnifiedSpecAdmission(msgspec.Struct):
    """Result of `solve_unified_spec_admission` (pure arithmetic; unit-tested
    in test/registered/unit/model_executor/test_unified_spec_admission.py)."""

    max_num_reqs: int
    hard_mamba_slots: int
    headroom_slots: int
    mamba_slots: int  # hard + radix headroom (possibly explicit-capped)
    band_bytes: int  # spec-state band at max_num_reqs
    deducted_bytes: int  # profile deduction: band + mamba bytes
    token_budget_bytes: int  # rest - deducted (converted by the fused cell)


def solve_unified_spec_admission(
    *,
    rest_bytes: int,
    requested: int,
    mamba_bytes_per_req: int,
    num_draft_tokens: int,
    hard_slots: int,
    mamba_ratio: int,
    cell_size: int,
    margin_slots: int,
    explicit_mamba_slots: Optional[int] = None,
) -> UnifiedSpecAdmission:
    """Joint hard-floor admission solve for the unified pool, spec ON or OFF
    (`num_draft_tokens=0` degenerates every band term to exactly zero).

    All charges are in raw unified-buffer bytes. The draft's KV needs no
    separate accounting: it is fused into the full-KV slot entry, so
    `cell_size` (the fused entry, bytes per token across all layers) already
    carries the draft's per-token cost, and the mamba/band bytes have no
    draft-pool backing to reserve — there is no separate draft pool.
    """
    assert cell_size > 0

    token_floor_bytes = UNIFIED_MIN_TOKENS_PER_REQ * cell_size
    band_const_bytes = (1 + margin_slots) * num_draft_tokens * mamba_bytes_per_req
    per_req_bytes = (
        hard_slots + num_draft_tokens
    ) * mamba_bytes_per_req + token_floor_bytes
    n = (rest_bytes - band_const_bytes) // per_req_bytes
    n = min(requested, n)
    if explicit_mamba_slots is not None:
        # An explicit --max-mamba-cache-size caps both the slot grant and
        # the hard-floor admission it can back.
        n = min(n, explicit_mamba_slots // hard_slots)
    if n <= 0:
        return UnifiedSpecAdmission(
            max_num_reqs=int(n),
            hard_mamba_slots=0,
            headroom_slots=0,
            mamba_slots=0,
            band_bytes=0,
            deducted_bytes=0,
            token_budget_bytes=0,
        )

    band_bytes = (n + 1 + margin_slots) * num_draft_tokens * mamba_bytes_per_req
    hard_mamba_slots = n * hard_slots
    # Radix-cache headroom: leftover bytes beyond the band + hard slots +
    # the token floor, capped at the base ratio's allowance. The grant is
    # itemization, not a fence — at runtime the frontiers share bytes
    # freely. Accounting invariant vs the factory: the factory re-adds the
    # band + mamba bytes into total_bytes; this solve deducts exactly the
    # same bytes (accounting must not drift on either side).
    leftover = (
        rest_bytes
        - band_bytes
        - hard_mamba_slots * mamba_bytes_per_req
        - n * token_floor_bytes
    )
    headroom_slots = min(
        max(0, leftover) // mamba_bytes_per_req,
        n * mamba_ratio - hard_mamba_slots,
    )
    mamba_slots = int(hard_mamba_slots + headroom_slots)
    if explicit_mamba_slots is not None:
        mamba_slots = min(mamba_slots, explicit_mamba_slots)

    deducted_bytes = band_bytes + mamba_slots * mamba_bytes_per_req
    return UnifiedSpecAdmission(
        max_num_reqs=int(n),
        hard_mamba_slots=int(hard_mamba_slots),
        headroom_slots=int(headroom_slots),
        mamba_slots=mamba_slots,
        band_bytes=band_bytes,
        deducted_bytes=deducted_bytes,
        token_budget_bytes=rest_bytes - deducted_bytes,
    )


logger = logging.getLogger(__name__)


def check_unified_mamba_family_admissible(config) -> None:
    """Positive model-family allow-list for the unified-pool mamba path.

    The unified pool stores linear-attention state in envelope-strided views
    addressed by physical slot ids, so every state-touching kernel of an
    admitted family must honor explicit slot strides (the full view's
    ``.stride(0)``) instead of assuming dense ``[batch, ...]`` state tensors,
    and the family's pool/backend wiring must be validated end to end.
    Families are admitted one by one after that audit; anything else fails
    loudly here instead of corrupting state silently at runtime.

    ``config`` is the RESOLVED mambaish config (``ModelRunner.mambaish_config``
    — already unwrapped from any VL wrapper). Notable EXCLUSIONS, kept out on
    purpose:
      - JetNemotron / JetVLM: their decode state-update kernel addresses the
        initial state densely (no slot stride) — silent wrong-slot state under
        envelope-strided views.
      - BailingMoeV2_5 (hybrid lightning): mambaish AND MLA-capable, but its
        lightning-attention backend + overlap path is entirely unvetted here.
    """
    from sglang.srt.configs import (
        FalconH1Config,
        GraniteMoeHybridConfig,
        KimiLinearConfig,
        Lfm2Config,
        Lfm2MoeConfig,
        Lfm2VlConfig,
        NemotronHConfig,
        Qwen3_5Config,
        Qwen3NextConfig,
        ZayaConfig,
    )

    allowed = (
        # GDN via the Qwen-family configs specifically (NOT the whole
        # hybrid_gdn union — that also admits JetNemotron/JetVLM). Subclasses
        # (e.g. the Qwen3.5-MoE-derived VL configs) are covered by isinstance.
        Qwen3NextConfig,
        Qwen3_5Config,
        # Mamba2 family — shares the stride-aware mamba2 kernel set.
        FalconH1Config,
        NemotronHConfig,
        Lfm2Config,
        Lfm2MoeConfig,
        Lfm2VlConfig,
        ZayaConfig,
        GraniteMoeHybridConfig,
        # KDA + MLA hybrid — KDA kernels are stride-aware; the MLA full side
        # runs the unified MLA sub-pool.
        KimiLinearConfig,
    )
    if not isinstance(config, allowed):
        raise ValueError(
            "--enable-unified-memory: model family "
            f"{type(config).__name__} has not been validated against the "
            "unified memory pool's state-layout and kernel-stride contracts. "
            "Drop --enable-unified-memory for this model — plain "
            "--enable-page-major-kv-layout still works."
        )


def _get_dsv4_compress_state_dtypes() -> tuple[torch.dtype, torch.dtype]:
    dtype_name = envs.SGLANG_DSV4_COMPRESS_STATE_DTYPE.get().strip().lower()
    if dtype_name in ("float32", "fp32"):
        return torch.float32, torch.float32
    if dtype_name in ("bfloat16", "bf16"):
        if envs.SGLANG_OPT_USE_ONLINE_COMPRESS.get():
            raise ValueError(
                "SGLANG_DSV4_COMPRESS_STATE_DTYPE=bf16 is not supported when "
                "SGLANG_OPT_USE_ONLINE_COMPRESS=1; online c128 state must stay float32."
            )
        return torch.bfloat16, torch.bfloat16
    raise ValueError(
        "Unsupported SGLANG_DSV4_COMPRESS_STATE_DTYPE="
        f"{dtype_name!r}. Expected one of: float32, fp32, bfloat16, bf16."
    )


_is_npu = is_npu()
_is_hip = is_hip()


class ModelRunnerKVCacheMixin:
    def _profile_available_bytes(self: ModelRunner, pre_model_load_memory: int) -> int:
        # KV pool budget = currently-free GPU memory minus the non-static runtime
        # slack (pre_model_load_memory * (1 - mem_fraction_static)). Whatever is
        # already resident (model weights, etc.) is thus charged against it.
        available_gpu_memory = get_available_gpu_memory(
            self.device,
            self.gpu_id,
            distributed=get_world_group().world_size > 1,
            cpu_group=get_world_group().cpu_group,
        )

        rest_memory = available_gpu_memory - pre_model_load_memory * (
            1 - self.mem_fraction_static
        )
        if self.mambaish_config is not None:
            rest_memory = self.handle_max_mamba_cache(rest_memory)

        # Loaded weights (target + draft) can exceed the static budget
        if rest_memory <= 0:
            minimum_mem_fraction_static = (
                1 - available_gpu_memory / pre_model_load_memory
            )
            suggested_mem_fraction_static = (
                math.ceil(minimum_mem_fraction_static * 1000) / 1000
            )
            raise ValueError(
                f"Loaded weights leave no GPU memory for the KV cache under "
                f"--mem-fraction-static={self.mem_fraction_static}. "
                f"Raise --mem-fraction-static above "
                f"{suggested_mem_fraction_static:.3f} "
                f"(minimum viable = 1 - available/pre = "
                f"{minimum_mem_fraction_static:.4f}). If using speculative "
                f"decoding, draft weights are now counted."
            )

        return int(rest_memory * (1 << 30))  # return in bytes

    def handle_max_mamba_cache(self: ModelRunner, total_rest_memory):
        config = self.mambaish_config
        server_args = self.server_args
        assert config is not None

        # Consumed by _init_unified_mamba_pools (spec-state band row count) and
        # _resolve_max_num_reqs (admission bypass). None on every non-unified /
        # spec-off path.
        self.mamba_spec_state_size: Optional[int] = None
        self._unified_max_num_reqs: Optional[int] = None

        # Unified pool (target worker), spec ON or OFF: admission and the
        # reservations are solved JOINTLY against the shared buffer's byte
        # budget instead of the static-split worst case below — see
        # _handle_max_mamba_cache_unified. At spec-OFF the solve degenerates
        # to hard mamba slots + the token floor (D=0, no band), replacing the
        # mamba_full_memory_ratio byte split AND the max_mamba_cache_size //
        # mamba_ratio admission clamp that produced boot-frozen worst-case
        # caps ("N=11 at 3% pool usage"). Draft workers never reach here
        # (they inherit the target's MemoryPoolConfig).
        if (
            server_args.enable_unified_memory
            and not self.is_draft_worker
            and server_args.disaggregation_mode == "null"
        ):
            return self._handle_max_mamba_cache_unified(total_rest_memory)

        # -------------------------------------------------------------------
        # Non-unified path: UPSTREAM-VERBATIM sizing, deliberately un-fixed.
        #
        # TODO(upstream bugfix PR): the spec-decode intermediate reservation
        # below is sized by the RAW requested max_running_requests (the spec
        # hook forces 48 when unset), but the mamba clamp in
        # _resolve_max_num_reqs (max_mamba_cache_size // ratio) makes most of
        # those rows unreachable — the reservation defeats itself. Concretely
        # (Falcon-H1-7B @ mfs0.45, D=4): 24.9 GiB reserved to support 48
        # concurrent verifies, after which only 1-2 requests are admittable
        # (radix branch), and the disable-radix branch's unchecked
        # demand-derived slot count goes budget-negative -> "no GPU memory for
        # KV cache" at boot. The reservation and the cap must be solved
        # JOINTLY (the closed form of their fixed point). Kept verbatim so
        # this branch's diff vs upstream stays empty; the fix is slated as a
        # standalone upstream PR. The unified path above does not have this
        # problem.
        # -------------------------------------------------------------------
        # reserve the memory for the intermediate mamba states used for spec dec
        if not self.spec_algorithm.is_none():
            assert server_args.speculative_num_draft_tokens is not None
            assert server_args.max_running_requests is not None

            max_running_requests = server_args.max_running_requests // (
                self.dp_size if server_args.enable_dp_attention else 1
            )
            mamba_state_intermediate_size = (
                config.mamba2_cache_params.mamba_cache_per_req
                * max_running_requests
                * server_args.speculative_num_draft_tokens
            )
            total_rest_memory = total_rest_memory - (
                mamba_state_intermediate_size / (1 << 30)
            )

        if server_args.max_mamba_cache_size is not None:
            # Use explicitly set max_mamba_cache_size
            server_args.max_mamba_cache_size = server_args.max_mamba_cache_size // (
                server_args.dp_size if server_args.enable_dp_attention else 1
            )
        elif (
            server_args.disable_radix_cache
            and server_args.max_running_requests is not None
        ):
            # Use explicitly set max_running_requests when radix cache is disabled
            server_args.max_mamba_cache_size = server_args.max_running_requests // (
                server_args.dp_size if server_args.enable_dp_attention else 1
            )
        else:
            # Use ratio-based calculation to auto-fit available memory
            assert config.mamba2_cache_params.mamba_cache_per_req > 0

            # allocate the memory based on the ratio between mamba state memory vs. full kv cache memory
            # solve the equations:
            # 1. mamba_state_memory + full_kv_cache_memory == total_rest_memory
            # 2. mamba_state_memory / full_kv_cache_memory == server_args.mamba_full_memory_ratio
            mamba_state_memory_raw = (
                total_rest_memory
                * server_args.mamba_full_memory_ratio
                / (1 + server_args.mamba_full_memory_ratio)
            )
            # calculate the max_mamba_cache_size based on the given total mamba memory
            server_args.max_mamba_cache_size = int(
                (mamba_state_memory_raw * (1 << 30))
                // config.mamba2_cache_params.mamba_cache_per_req
            )

        mamba_state_memory = (
            server_args.max_mamba_cache_size
            * config.mamba2_cache_params.mamba_cache_per_req
            / (1 << 30)
        )
        return total_rest_memory - mamba_state_memory

    def _handle_max_mamba_cache_unified(self: ModelRunner, total_rest_memory):
        """Unified-pool (target) sizing, spec ON or OFF: solve admission and
        the reservations JOINTLY against the shared buffer's byte budget.

        The static-partition model (upstream path above) charges every request
        its worst case FOREVER — `ratio` mamba slots + D intermediate rows —
        then splits bytes by `mamba_full_memory_ratio`. At Falcon-7B/mfs0.45/
        D=4 that admits 11 requests against a 313k-token pool the workload
        cannot fill (3% peak usage with 189 requests queued). Under the
        unified pool all partitions share one byte buffer with floating
        frontiers, so boot-time admission only needs each request's
        IRREDUCIBLE floor:

            floor mamba slots (active + extra_buffer track + radix
            checkpoint allowance = the full mamba_ratio)           x per_req
          + D spec-band rows (worst case: all N verify together)   x per_req
          + UNIFIED_MIN_TOKENS_PER_REQ tokens                      x cell_size

        The radix-checkpoint allowance is INSIDE the floor: it is
        consumed by every running chunked request as a LOCKED tree checkpoint,
        so leftover-funded "headroom" starves under an uncapped byte solve and
        trips the fail-loud `_alloc_mamba_slot` assert). Runtime pressure
        lands on the existing machinery: per-request band/mamba charges in
        admission (schedule_policy), check_decode_mem's probe->place->retract,
        and radix eviction.

        DRAFT KV: fused into the full-KV slot entry (`cell_size` is the fused
        entry from the draft's own geometry), so the draft worker allocates no
        pool of its own and nothing beyond the cell needs reserving.

        ADMISSION CEILING: when --max-running-requests was NOT set by the
        user, the speculative hook's default of 48 (a static-partition tuning)
        is lifted to UNIFIED_MAX_RUNNING_REQUESTS_CEILING and the byte
        solve becomes the binding constraint; the solved value is written back
        to server_args so the draft worker and downstream consumers see it.
        Spec-OFF there is no hook default at all (None) — the solve seeds
        `requested` with UNIFIED_SPEC_OFF_REQ_CEILING and nothing is written
        back (_resolve_max_num_reqs re-clamps as usual).

        Sets `self._unified_max_num_reqs` (consumed by _resolve_max_num_reqs,
        which then skips the static ratio clamp), `self.mamba_spec_state_size`
        (band rows = N), and `server_args.max_mamba_cache_size` (hard slots +
        headroom grant); returns the token-pool budget in GiB, mirroring the
        upstream path's contract.
        """
        config = self.mambaish_config
        server_args = self.server_args
        assert config is not None
        # Spec-OFF is a first-class caller: speculative_num_draft_tokens and
        # max_running_requests may both be None (no spec hook ran to default
        # them) — handled below, not asserted away.
        assert config.mamba2_cache_params.mamba_cache_per_req > 0

        from sglang.srt.mem_cache.multi_ended_allocator import (
            SPEC_BAND_ALIGNMENT_MARGIN_SLOTS,
        )
        from sglang.srt.model_executor.pool_configurator import (
            create_memory_pool_configurator,
        )

        dp = self.dp_size if server_args.enable_dp_attention else 1
        # `requested` seeds the solve's admission cap. Three cases:
        #   - spec-OFF with no --max-running-requests: no default exists at
        #     all (None) — seed with the LARGE spec-off ceiling so the byte
        #     solve is the binding bound (_resolve_max_num_reqs re-clamps
        #     with estimated/user/capacity afterwards; nothing written back).
        #   - the speculative hook DEFAULTED it to 48: a static-partition
        #     tuning, not a user decision — lift to the unified ceiling and
        #     let the byte solve bound admission (retract is the backstop).
        #   - explicit --max-running-requests: binds.
        requested_defaulted = getattr(
            server_args, "_max_running_requests_spec_defaulted", False
        )
        # 0 when spec is OFF: every band term in the solve degenerates to 0.
        D = server_args.speculative_num_draft_tokens or 0
        if server_args.max_running_requests is None:
            # Normally spec-OFF (the spec hook always defaults the cap). If a
            # spec path ever skips the hook, seed the SPEC ceiling instead —
            # 4096 verify-band rows would be charged per admitted request.
            if D == 0:
                requested = UNIFIED_SPEC_OFF_REQ_CEILING // dp
                requested_note = " = spec-off ceiling, byte solve binds"
            else:
                requested = UNIFIED_MAX_RUNNING_REQUESTS_CEILING // dp
                requested_note = " = unified ceiling (cap unset)"
            requested_defaulted = False
        elif requested_defaulted:
            requested = UNIFIED_MAX_RUNNING_REQUESTS_CEILING // dp
            requested_note = " = unified ceiling, hook default lifted"
        else:
            requested = server_args.max_running_requests // dp
            requested_note = ""
        per_req = config.mamba2_cache_params.mamba_cache_per_req
        ratio = self._calculate_mamba_ratio()
        # The per-request slot FLOOR is the full ratio: 1 active + the
        # extra_buffer ping-pong track slots + the base ratio's radix
        # allowance. The radix slots are NOT optional headroom: a running
        # chunked request checkpoints its mamba state into the tree
        # (`cache_unfinished_req -> _alloc_mamba_slot`, a fail-loud assert)
        # and that checkpoint stays LOCKED while the request runs — charging
        # only active+track lets an uncapped byte solve admit huge N with a
        # handful of leftover-funded radix slots and die on that assert
        # under load. `mamba_ratio` is 1 under disable_radix_cache, so the
        # floor degenerates correctly. The solve's headroom term then caps at
        # zero (floor == ratio) — the guarantee is exactly upstream's
        # `max_mamba_cache_size // ratio` allowance, byte-solved.
        hard_slots = ratio

        # Per-token full-KV bytes from the SAME configurator class that sizes
        # the token pool right after this returns. Under the unified pool the
        # draft's KV is fused into the full-KV slot entry, so `_cell_size` IS
        # the fused entry (target + draft bytes per token) — one cell prices
        # everything; no separate draft pool exists to reserve for.
        configurator = create_memory_pool_configurator(self)
        cell_size = configurator._cell_size

        rest_bytes = int(total_rest_memory * (1 << 30))
        explicit_mamba_slots = (
            server_args.max_mamba_cache_size // dp
            if server_args.max_mamba_cache_size is not None
            else None
        )
        admission = solve_unified_spec_admission(
            rest_bytes=rest_bytes,
            requested=requested,
            mamba_bytes_per_req=per_req,
            num_draft_tokens=D,
            hard_slots=hard_slots,
            mamba_ratio=ratio,
            cell_size=cell_size,
            margin_slots=SPEC_BAND_ALIGNMENT_MARGIN_SLOTS,
            explicit_mamba_slots=explicit_mamba_slots,
        )
        n = admission.max_num_reqs
        if n <= 0:
            per_req_floor_bytes = (
                hard_slots + D
            ) * per_req + UNIFIED_MIN_TOKENS_PER_REQ * cell_size
            band_note = f" + {D} band rows" if D else ""
            spec_tip = " (2) reduce --speculative-num-draft-tokens, or" if D else ""
            raise RuntimeError(
                f"Not enough GPU memory for unified-pool admission: "
                f"joint hard-floor solve admits {n} requests "
                f"(rest={total_rest_memory:.2f} GiB, per-request floor="
                f"{per_req_floor_bytes / (1 << 30):.2f} GiB: {hard_slots} mamba "
                f"slots{band_note} x {per_req / (1 << 20):.1f} MiB "
                f"+ {UNIFIED_MIN_TOKENS_PER_REQ} tokens x {cell_size} B). "
                f"Try: (1) increase --mem-fraction-static,{spec_tip} "
                f"({3 if D else 2}) use GPUs with more memory."
            )

        server_args.max_mamba_cache_size = admission.mamba_slots
        # Band rows exist only under spec; None keeps the factory on its
        # no-band path (mirrors the upstream spec-off contract).
        self.mamba_spec_state_size = int(n) if D > 0 else None
        self._unified_max_num_reqs = int(n)
        if requested_defaulted:
            # Write the solved cap back so every downstream consumer — the
            # draft worker (via memory_pool_config), _resolve_max_num_reqs,
            # scheduler — sees one consistent value. Explicit user values are
            # never overwritten (_resolve_max_num_reqs min's + warns instead).
            server_args.max_running_requests = int(n * dp)

        draft_note = (
            f"draft KV fused into cell ({cell_size} B/token)"
            if self.draft_kv_geometry is not None
            else f"no draft KV ({cell_size} B/token)"
        )
        logger.info(
            "[unified-memory-pool] joint admission solve: max_num_reqs=%d "
            "(requested %d%s), floor slots/req=%d (active+track+radix ckpt), "
            "D=%d, mamba slots=%d (floor %d + headroom %d), band %.2f GiB, "
            "%s, token floor %d tok/req, token-pool budget %.2f GiB.",
            n,
            requested,
            requested_note,
            hard_slots,
            D,
            admission.mamba_slots,
            admission.hard_mamba_slots,
            admission.headroom_slots,
            admission.band_bytes / (1 << 30),
            draft_note,
            UNIFIED_MIN_TOKENS_PER_REQ,
            admission.token_budget_bytes / (1 << 30),
        )
        return total_rest_memory - admission.deducted_bytes / (1 << 30)

    def calculate_mla_kv_cache_dim(self: ModelRunner) -> int:
        is_dsa_model = is_deepseek_dsa(self.model_config.hf_config)
        kv_cache_dtype = self.kv_cache_dtype
        kv_lora_rank = self.model_config.kv_lora_rank
        qk_rope_head_dim = self.model_config.qk_rope_head_dim
        kv_cache_dim = kv_lora_rank + qk_rope_head_dim  # default mla kv cache dim

        # For non-DSA models, MLA kv cache dim is simply kv_lora_rank + qk_rope_head_dim
        if not is_dsa_model:
            return kv_cache_dim

        # TRTLLM backend does not override kv_cache_dim for MLA kv cache
        # Assuming dsa prefill and decode backends are the same when using trtllm MLA backend,
        # since it is not compatible for trtllm and other mla attn backend due to the different
        # kv cache layout.
        if (
            self.server_args.dsa_prefill_backend == "trtllm"
            or self.server_args.dsa_decode_backend == "trtllm"
        ):
            return kv_cache_dim

        # On HIP, TileLang and AITER DSA kernels consume the raw MLA KV layout:
        # nope(512 fp8) + rope(64 fp8), without extra per-block scales.
        if _is_hip and (
            self.server_args.dsa_prefill_backend in ("tilelang", "aiter")
            or self.server_args.dsa_decode_backend in ("tilelang", "aiter")
        ):
            return kv_cache_dim

        quant_block_size = DSATokenToKVPool.quant_block_size
        rope_storage_dtype = DSATokenToKVPool.rope_storage_dtype
        # Calculate override_kv_cache_dim for FP8 storage in backends that use scaled KV layout
        # (excluding TRTLLM and HIP raw-layout kernels).
        # kv_lora_rank + scale storage (kv_lora_rank // quant_block_size * 4 bytes) + rope dimension storage
        # Note: rope dimension is stored in original dtype (bf16), not quantized to fp8
        if kv_cache_dtype == torch.float8_e4m3fn:
            assert (
                kv_lora_rank % quant_block_size == 0
            ), f"kv_lora_rank {kv_lora_rank} must be multiple of quant_block_size {quant_block_size}"

            return (
                kv_lora_rank
                + kv_lora_rank // quant_block_size * 4
                + qk_rope_head_dim * rope_storage_dtype.itemsize
            )

        return kv_cache_dim

    def _calculate_mamba_ratio(self: ModelRunner) -> int:
        if self.server_args.disable_radix_cache:
            return 1

        additional_ratio = 0
        if self.server_args.enable_mamba_extra_buffer():
            # ping-pong buffer size is 2 when overlap schedule is on, 1 otherwise.
            # Lazy mode saves 1 slot (2 → 1) for overlap; non-overlap already uses 1.
            if not self.server_args.disable_overlap_schedule:
                if self.server_args.enable_mamba_extra_buffer_lazy():
                    additional_ratio = MAMBA_CACHE_V2_ADDITIONAL_RATIO_OVERLAP_LAZY
                else:
                    additional_ratio = MAMBA_CACHE_V2_ADDITIONAL_RATIO_OVERLAP
            else:
                assert (
                    not self.server_args.enable_mamba_extra_buffer_lazy()
                ), "Lazy extra buffer requires overlap schedule (--disable-overlap-schedule is incompatible)"
                additional_ratio = MAMBA_CACHE_V2_ADDITIONAL_RATIO_NO_OVERLAP

        return MAMBA_CACHE_SIZE_MAX_RUNNING_REQUESTS_RATIO + additional_ratio

    def _validate_prefill_only_disable_kv_cache_pool_family(
        self: ModelRunner,
        is_dsa_model: bool,
        is_dsv4_model: bool,
        current_platform,
    ):
        if not self.server_args.prefill_only_disable_kv_cache or self.is_draft_worker:
            return

        unsupported_pool_family = None
        if is_dsv4_model:
            unsupported_pool_family = "DeepSeekV4TokenToKVPool"
        elif current_platform.is_out_of_tree() and not self.mambaish_config:
            unsupported_pool_family = "out-of-tree platform KV pool"
        elif (
            self.server_args.attention_backend == "ascend" and not self.mambaish_config
        ):
            unsupported_pool_family = "NPU/Ascend KV pool"
        elif self.use_mla_backend and is_dsa_model:
            unsupported_pool_family = "DSA/MLA KV pool"
        elif self.use_mla_backend and not self.mambaish_config:
            unsupported_pool_family = "MLA KV pool"
        elif self.is_hybrid_swa:
            unsupported_pool_family = "SWA KV pool"
        elif self.mambaish_config:
            unsupported_pool_family = "hybrid linear/Mamba KV pool"
        elif is_float4_e2m1fn_x2(self.kv_cache_dtype):
            unsupported_pool_family = "FP4 MHA KV pool"

        if unsupported_pool_family is not None:
            raise RuntimeError(
                "--prefill-only-disable-kv-cache is not supported for "
                f"{unsupported_pool_family}. Supported configurations today: plain MHA "
                "models on CUDA with the FA (fa3/fa4) prefill backend, --is-embedding, "
                "--chunked-prefill-size=-1, --disable-radix-cache, no context-parallel "
                "attention, no HiSparse, and --kv-cache-dtype != fp4_e2m1."
            )

    def _warn_inert_partition_ratios(self: ModelRunner) -> None:
        """The unified pool sizes ONE byte budget and splits it dynamically at
        runtime — the static-partition ratios govern nothing on this path.
        Warn (once, at boot) when the user set a non-default value so a tuned
        launch script doesn't silently expect the old split. Parse-time range
        validation and the non-unified path keep using them unchanged."""
        from sglang.srt.server_args import ServerArgs

        sa = self.server_args
        for flag, field in (
            ("--mamba-full-memory-ratio", "mamba_full_memory_ratio"),
            ("--swa-full-tokens-ratio", "swa_full_tokens_ratio"),
        ):
            if field == "swa_full_tokens_ratio" and getattr(sa, field) == 1.0:
                # Step3p5 + hierarchical cache internally resets this to 1.0
                # (server_args __post_init__) — not a user decision; and 1.0
                # is a no-op split anyway. Don't warn.
                continue
            if getattr(sa, field) != getattr(ServerArgs, field):
                logger.warning(
                    "[unified-memory-pool] %s=%s is IGNORED under "
                    "--enable-unified-memory: the pool is one byte budget with "
                    "a fully dynamic runtime split (admission is solved "
                    "jointly in bytes).",
                    flag,
                    getattr(sa, field),
                )

    def _init_unified_mamba_pools(self: ModelRunner, max_num_reqs: int):
        """Build the shared-KV-pool stack for a hybrid-Mamba model:
        one byte buffer split between the full-attn MHA KV pool and the
        per-request Mamba state pool, with virtual slot ids above the
        allocator."""
        from sglang.srt.mem_cache.unified_memory_pool import init_unified_mamba_pools

        config = self.mambaish_config
        assert config is not None
        # Positive family allow-list — fails loud for unvetted linear-attention
        # families before any pool is built.
        check_unified_mamba_family_admissible(config)
        # The full sub-pool is page-aware (via `MultiEndedAllocator(page_size=...)`);
        # the mamba sub-pool stays page=1.
        assert self.page_size >= 1, f"page_size must be >= 1, got {self.page_size}"
        # Mirror the non-shared path's extra_max_context_len computation.
        extra_max_context_len = 4
        if self.server_args.speculative_num_draft_tokens is not None:
            extra_max_context_len += self.server_args.speculative_num_draft_tokens

        mamba_layer_ids = [
            i
            for i in config.mamba2_cache_params.layers
            if self.start_layer <= i < self.end_layer
        ]
        full_attention_layer_ids = [
            i
            for i in config.full_attention_layer_ids
            if self.start_layer <= i < self.end_layer
        ]

        bundle = init_unified_mamba_pools(
            device=self.device,
            kv_cache_dtype=self.kv_cache_dtype,
            head_num=self.model_config.get_num_kv_heads(get_attention_tp_size()),
            head_dim=self.model_config.head_dim,
            page_size=self.page_size,
            start_layer=self.start_layer,
            end_layer=self.end_layer,
            is_draft_worker=self.is_draft_worker,
            use_mla_backend=self.use_mla_backend,
            mamba_layer_ids=mamba_layer_ids,
            full_attention_layer_ids=full_attention_layer_ids,
            mamba2_cache_params=config.mamba2_cache_params,
            model_context_len=self.model_config.context_len,
            extra_max_context_len=extra_max_context_len,
            max_total_num_tokens=self.max_total_num_tokens,
            max_mamba_cache_size=self.server_args.max_mamba_cache_size,
            max_num_reqs=max_num_reqs,
            enable_memory_saver=self.server_args.enable_memory_saver,
            enable_mamba_extra_buffer=self.server_args.enable_mamba_extra_buffer(),
            speculative_num_draft_tokens=self.server_args.speculative_num_draft_tokens,
            disable_overlap_schedule=self.server_args.disable_overlap_schedule,
            need_sort=self.server_args.disaggregation_mode in ("decode", "prefill"),
            # Overlap mode: the allocator's `free` drops a wait_stream(forward_stream)
            # barrier so eager compaction serializes after the in-flight forward's
            # v2p/KV reads. Near-no-op in normal mode.
            forward_stream=self.forward_stream,
            # Lazy compaction: default ON, env-var escape hatch for rollback / A/B.
            lazy_compaction=_should_enable_lazy_compaction(),
            # Spec-state band slot count: the SAME capped-request count
            # handle_max_mamba_cache reserved intermediate bytes for (falls
            # back to max_num_reqs inside the factory when None). Direct
            # access: handle_max_mamba_cache always runs before pool init for
            # mambaish configs.
            spec_state_size=self.mamba_spec_state_size,
            # Fused draft KV: attach the draft region to the full
            # sub-pool so draft KV rides inside the fused slot entries
            # (non-None <=> fusion enabled; gated in maybe_init_draft_kv_geometry).
            draft_kv_geometry=self.draft_kv_geometry,
            # MLA-hybrid (e.g. Kimi Linear): the full sub-pool stores the MLA
            # latent — same dim source as the baseline composite's MLA pool.
            kv_lora_rank=(
                self.model_config.kv_lora_rank if self.use_mla_backend else None
            ),
            qk_rope_head_dim=(
                self.model_config.qk_rope_head_dim if self.use_mla_backend else None
            ),
        )
        self.req_to_token_pool = bundle.req_to_token_pool
        self.token_to_kv_pool = bundle.token_to_kv_pool
        self.token_to_kv_pool_allocator = bundle.token_to_kv_pool_allocator
        # Keep a reference so the shared byte buffer is not GC'd.
        self._unified_memory_pool = bundle.unified_memory_pool

    def _init_unified_swa_pools(self: ModelRunner, max_num_reqs: int):
        """Build the unified-pool stack for a hybrid-SWA model (Triton): one byte
        buffer split between the full-attention and SWA KV pools."""
        from sglang.srt.mem_cache.unified_memory_pool import init_unified_swa_pools

        assert self.is_hybrid_swa, "_init_unified_swa_pools called on a non-SWA model"
        # Both sub-pools are page-aware; the SWA composite runs alloc_extend_kernel
        # once in virtual space and binds the new pages on both sub-allocators.
        assert self.page_size >= 1, f"page_size must be >= 1, got {self.page_size}"
        assert (
            not self.use_mla_backend
        ), "unified memory pool does not support MLA-SWA hybrid yet"
        # Mirror the non-shared path's extra_max_context_len computation.
        extra_max_context_len = 4
        if self.server_args.speculative_num_draft_tokens is not None:
            extra_max_context_len += self.server_args.speculative_num_draft_tokens
        self.req_to_token_pool = ReqToTokenPool(
            size=max_num_reqs,
            max_context_len=self.model_config.context_len + extra_max_context_len,
            device=self.device,
            enable_memory_saver=self.server_args.enable_memory_saver,
        )

        head_num = self.model_config.get_num_kv_heads(get_attention_tp_size())
        head_dim = self.model_config.head_dim
        if self.is_hybrid_swa_compress:
            # Asymmetric head dims between full and SWA (NPU compress path):
            # pull SWA-specific dims from the hf text config.
            v_head_dim = self.model_config.hf_text_config.v_head_dim
            swa_head_num = max(
                1,
                self.model_config.hf_text_config.swa_num_key_value_heads
                // get_attention_tp_size(),
            )
            swa_head_dim = self.model_config.hf_text_config.swa_head_dim
            swa_v_head_dim = self.model_config.hf_text_config.swa_v_head_dim
        else:
            v_head_dim = head_dim
            swa_head_num = head_num
            swa_head_dim = head_dim
            swa_v_head_dim = head_dim

        # Filter layer ids to this worker's [start_layer, end_layer) range.
        swa_attention_layer_ids = [
            i
            for i in self.model_config.swa_attention_layer_ids
            if self.start_layer <= i < self.end_layer
        ]
        full_attention_layer_ids = [
            i
            for i in self.model_config.full_attention_layer_ids
            if self.start_layer <= i < self.end_layer
        ]

        bundle = init_unified_swa_pools(
            device=self.device,
            kv_cache_dtype=self.kv_cache_dtype,
            head_num=head_num,
            head_dim=head_dim,
            v_head_dim=v_head_dim,
            swa_head_num=swa_head_num,
            swa_head_dim=swa_head_dim,
            swa_v_head_dim=swa_v_head_dim,
            page_size=self.page_size,
            start_layer=self.start_layer,
            end_layer=self.end_layer,
            swa_attention_layer_ids=swa_attention_layer_ids,
            full_attention_layer_ids=full_attention_layer_ids,
            full_max_total_num_tokens=self.full_max_total_num_tokens,
            swa_max_total_num_tokens=self.swa_max_total_num_tokens,
            enable_memory_saver=self.server_args.enable_memory_saver,
            need_sort=self.server_args.disaggregation_mode in ("decode", "prefill"),
            # Overlap mode: same wait_stream(forward_stream) rationale as
            # `_init_unified_mamba_pools`.
            forward_stream=self.forward_stream,
            # Lazy compaction: default ON, with env var escape hatch for rollback / A/B.
            lazy_compaction=_should_enable_lazy_compaction(),
            # Fused draft KV: the dense draft fuses into the FULL
            # sub-pool (a DSV4-style draft would attach to "swa" instead;
            # non-None <=> fusion enabled, gated in maybe_init_draft_kv_geometry).
            draft_kv_geometry=self.draft_kv_geometry,
            # Ratio-free byte budget: allocate the buffer from the
            # profiled bytes directly; the token counts are labels. None falls
            # back to count-summing (SWAChunkCapPoolConfigurator path).
            total_bytes=getattr(self, "unified_swa_total_bytes", None),
        )
        self.token_to_kv_pool = bundle.token_to_kv_pool
        self.token_to_kv_pool_allocator = bundle.token_to_kv_pool_allocator
        # Keep a reference so the shared byte buffer is not GC'd.
        self._unified_memory_pool = bundle.unified_memory_pool

    def _init_unified_draft_pool(self: ModelRunner) -> bool:
        """DRAFT worker under a fused-draft-KV target: bind
        `token_to_kv_pool` to the DRAFT-region views of the target's unified
        full sub-pool instead of allocating a private, virtual-index-sized
        pool. Returns True when the fused binding was made; the caller must
        then skip all pool construction (the shared allocator/req_to_token
        came in through the spec-worker handoff).
        """
        if not self.is_draft_worker:
            return False
        unified_buffer = getattr(
            self.token_to_kv_pool_allocator, "unified_buffer", None
        )
        if unified_buffer is None:
            # Baseline (non-unified) target: the draft builds its own private
            # pool below, indexed by the physical ids the shared allocator
            # mints. Nothing unified-specific applies.
            return False
        if not unified_buffer.has_draft_region("full"):
            # A unified target with a draft worker but no draft region cannot
            # happen by construction: resolve_draft_kv_geometry either returns
            # a geometry (the factory then attaches the region) or the target
            # raises at init. Falling through would build a private pool
            # indexed as if the shared allocator's VIRTUAL ids were physical —
            # silent wrong-slot KV. Fail here instead.
            raise RuntimeError(
                "unified-memory draft worker found no draft region on the "
                "target's full sub-pool. The target must attach one whenever "
                "the speculative algorithm carries draft KV (fused draft KV "
                "has no fallback path)."
            )

        from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool
        from sglang.srt.mem_cache.unified_memory_pool import UnifiedDraftKVPool

        # The target built the region from resolve_draft_kv_geometry; validate
        # this runner's actual geometry against it (a mismatch would silently
        # corrupt neighboring slots).
        region = unified_buffer.mha_spec("full").draft_region
        head_num = self.model_config.get_num_kv_heads(get_attention_tp_size())
        head_dim = self.model_config.head_dim
        v_head_dim = self.model_config.v_head_dim or head_dim
        for name, got, expected in (
            ("head_num", head_num, region.head_num),
            ("head_dim", head_dim, region.head_dim),
            ("v_head_dim", v_head_dim, region.v_head_dim),
        ):
            assert got == expected, (
                f"fused draft region {name} mismatch: draft runner has {got}, "
                f"target reserved {expected} (resolve_draft_kv_geometry drift?)"
            )
        from sglang.srt.mem_cache.unified_memory_pool import _store_dtype_for

        assert _store_dtype_for(self.kv_cache_dtype) == region.store_dtype, (
            f"fused draft region store_dtype mismatch: draft runner stores "
            f"{_store_dtype_for(self.kv_cache_dtype)}, target reserved "
            f"{region.store_dtype} (4.1 requires draft dtype == target dtype)"
        )
        layer_offset = (self.draft_model_idx or 0) * self.num_effective_layers
        assert layer_offset + self.num_effective_layers <= region.layer_num, (
            f"draft layers [{layer_offset}, "
            f"{layer_offset + self.num_effective_layers}) exceed the fused "
            f"region's {region.layer_num} layers"
        )

        draft_pool = UnifiedDraftKVPool(
            unified_buffer=unified_buffer,
            sub_pool_name="full",
            page_size=self.page_size,
            layer_offset=layer_offset,
            layer_num=self.num_effective_layers,
            enable_alt_stream=not self.server_args.enable_pdmux,
        )
        if self.mambaish_config is not None:
            # MTP self-draft on hybrid Mamba: keep the HybridLinearKVPool
            # shell (mamba states stay in the shared mamba pool, translated).
            self.token_to_kv_pool = HybridLinearKVPool(
                page_size=self.page_size,
                size=self.max_total_num_tokens,
                dtype=self.kv_cache_dtype,
                head_num=head_num,
                head_dim=head_dim,
                full_attention_layer_ids=[0],
                device=self.device,
                mamba_pool=self.req_to_token_pool.mamba_pool,
                enable_memory_saver=self.server_args.enable_memory_saver,
                use_mla=self.use_mla_backend,
                start_layer=self.start_layer,
                full_kv_pool=draft_pool,
            )
        else:
            self.token_to_kv_pool = draft_pool
        logger.info(
            "[unified-memory-pool] draft worker: fused draft KV bound to the "
            "target's 'full' sub-pool (layers [%d, %d) of the %d-layer draft "
            "region); no private draft pool allocated.",
            layer_offset,
            layer_offset + self.num_effective_layers,
            region.layer_num,
        )
        return True

    def _init_pools(self: ModelRunner):
        """Initialize the memory pools."""
        max_num_reqs = self.max_running_requests

        # Draft worker under a unified-pool target: the draft's KV is fused
        # into the target's full-KV slot entry, so the draft worker binds
        # views of the target's unified sub-pool instead of allocating a pool
        # of its own — every construction path below is then irrelevant.
        if self._init_unified_draft_pool():
            return

        # Unified-pool fast path: build req_to_token + token_to_kv pool +
        # allocator, then return. Gated to the target worker (req_to_token_pool
        # is None; PD disaggregation is already rejected at argument parsing).
        # Supports hybrid Mamba and hybrid SWA (not DSV4); everything else fails
        # loud.
        if (
            self.server_args.enable_unified_memory
            and self.req_to_token_pool is None
        ):
            self._warn_inert_partition_ratios()
            if self.mambaish_config is not None:
                self._init_unified_mamba_pools(max_num_reqs)
                return
            if self.is_hybrid_swa and not is_deepseek_v4(self.model_config.hf_config):
                self._init_unified_swa_pools(max_num_reqs)
                return
            raise ValueError(
                "--enable-unified-memory requires a hybrid model "
                "(full-attention + Mamba-family state — MHA- or MLA-full — "
                "or full + sliding-window KV); the current model "
                f"({self.model_config.hf_config.architectures}) is not one "
                "(dense, non-hybrid MLA, and DeepSeek-V4 models are "
                "unsupported, and --disable-hybrid-swa-memory disables the "
                "SWA pairing). Drop --enable-unified-memory for this model — "
                "plain --enable-page-major-kv-layout still works."
            )

        # Initialize req_to_token_pool
        if self.req_to_token_pool is None:
            max_spec_draft_tokens = self.server_args.max_speculative_num_draft_tokens
            extra_max_context_len = get_req_to_token_extra_context_len(self.server_args)

            if self.server_args.disaggregation_mode == "decode":
                from sglang.srt.disaggregation.decode import (
                    DecodeReqToTokenPool,
                    HybridMambaDecodeReqToTokenPool,
                )

                # Extra slots for pre-allocated requests
                pre_alloc_size = self.server_args.disaggregation_decode_extra_slots
                if config := self.mambaish_config:
                    self.req_to_token_pool = HybridMambaDecodeReqToTokenPool(
                        size=max_num_reqs,
                        max_context_len=self.model_config.context_len
                        + extra_max_context_len,
                        device=self.device,
                        enable_memory_saver=self.server_args.enable_memory_saver,
                        cache_params=config.mamba2_cache_params,
                        mamba_layer_ids=(
                            [
                                i
                                for i in config.mamba2_cache_params.layers
                                if self.start_layer <= i < self.end_layer
                            ]
                        ),
                        speculative_num_draft_tokens=max_spec_draft_tokens,
                        speculative_eagle_topk=self.server_args.speculative_eagle_topk,
                        enable_mamba_extra_buffer=self.server_args.enable_mamba_extra_buffer(),
                        pre_alloc_size=pre_alloc_size,
                        enable_overlap_schedule=not self.server_args.disable_overlap_schedule,
                        mamba_size=self.server_args.max_mamba_cache_size,
                        start_layer=self.start_layer,
                    )
                else:
                    self.req_to_token_pool = DecodeReqToTokenPool(
                        size=max_num_reqs,
                        max_context_len=self.model_config.context_len
                        + extra_max_context_len,
                        device=self.device,
                        enable_memory_saver=self.server_args.enable_memory_saver,
                        pre_alloc_size=pre_alloc_size,
                    )
            elif config := self.mambaish_config:
                self.req_to_token_pool = HybridReqToTokenPool(
                    size=max_num_reqs,
                    mamba_size=self.server_args.max_mamba_cache_size,
                    mamba_spec_state_size=max_num_reqs,
                    max_context_len=self.model_config.context_len
                    + extra_max_context_len,
                    device=self.device,
                    enable_memory_saver=self.server_args.enable_memory_saver,
                    cache_params=config.mamba2_cache_params,
                    mamba_layer_ids=(
                        [
                            i
                            for i in config.mamba2_cache_params.layers
                            if self.start_layer <= i < self.end_layer
                        ]
                    ),
                    enable_mamba_extra_buffer=self.server_args.enable_mamba_extra_buffer(),
                    enable_mamba_extra_buffer_lazy=self.server_args.enable_mamba_extra_buffer_lazy(),
                    speculative_num_draft_tokens=max_spec_draft_tokens,
                    speculative_eagle_topk=self.server_args.speculative_eagle_topk,
                    enable_overlap_schedule=not self.server_args.disable_overlap_schedule,
                    start_layer=self.start_layer,
                    enable_linear_replayssm=self.server_args.enable_linear_replayssm,
                    linear_replayssm_cache_len=self.server_args.linear_replayssm_cache_len,
                    mamba_envelope_layout=self.server_args.enable_page_major_kv_layout,
                )
            else:
                # DSV4 on NPU needs an extended ReqToTokenPool holding per-req
                # swa/c4/c128/c{4,128}_state tables; others stay on the stock one.
                req_to_token_pool_cls = ReqToTokenPool
                if _is_npu and is_deepseek_v4(self.model_config.hf_config):
                    from sglang.srt.hardware_backend.npu.dsv4.dsv4_req_to_token_pool import (
                        DSV4NPUReqToTokenPool,
                    )

                    req_to_token_pool_cls = DSV4NPUReqToTokenPool

                self.req_to_token_pool = req_to_token_pool_cls(
                    size=max_num_reqs,
                    max_context_len=self.model_config.context_len
                    + extra_max_context_len,
                    device=self.device,
                    enable_memory_saver=self.server_args.enable_memory_saver,
                )
        else:
            # Draft worker shares req_to_token_pool with the target worker.
            assert self.is_draft_worker

        # Initialize token_to_kv_pool
        is_dsa_model = is_deepseek_dsa(self.model_config.hf_config)
        is_dsv4_model = is_deepseek_v4(self.model_config.hf_config)

        self._validate_prefill_only_disable_kv_cache_pool_family(
            is_dsa_model, is_dsv4_model, current_platform
        )

        # Page-granularity envelope layout for the MHA-shaped (full / SWA) pools,
        # selected by swapping in the PageMajorMHATokenToKVPool subclass. The
        # default keeps upstream's per-layer layout. The Mamba state pool is routed
        # separately via `mamba_envelope_layout` on the req-to-token pool above.
        enable_page_major = self.server_args.enable_page_major_kv_layout
        mha_pool_class = (
            PageMajorMHATokenToKVPool if enable_page_major else MHATokenToKVPool
        )

        if is_dsv4_model:
            swa_page_size = self.page_size
            if not _is_npu:
                assert swa_page_size == 256, "In paged swa mode, page_size must be 256."

            if self.is_draft_worker:
                from sglang.srt.models.deepseek_v4_nextn import (
                    COMPRESS_RATIO_NEXTN_LAYER,
                )

                compression_ratios = [
                    COMPRESS_RATIO_NEXTN_LAYER
                ] * self.num_effective_layers
            else:
                compression_ratios = self.model_config.compress_ratios

            # NPU + DSV4 → paged-state subclass: the fused compressor kernel
            # needs cache_mode=1 (paged); Atlas A3 rejects cache_mode=2 (ring),
            # so the CUDA ring-buffer state path can't be shared. CUDA keeps
            # DeepSeekV4TokenToKVPool unchanged; NPU recomputes state sizes below.
            if _is_npu:
                from sglang.srt.hardware_backend.npu.dsv4.dsv4_memory_pool import (
                    DSV4NPUTokenToKVPool,
                    npu_state_pool_size,
                )

                pool_cls = DSV4NPUTokenToKVPool
                # Recompute state pool sizes for the NPU paged formula (CUDA's
                # ring sizes are dropped here). Tail-only allocation keeps the
                # per-req-budget formula sufficient at any prefill length: long
                # prompts allocate only ``tail+128`` (c4) / ``tail`` (c128)
                # slots (tail = seq_len % 128), and decode is drained by
                # sliding eviction in ``ScheduleBatch._evict_swa``.
                c4_state_pool_size = npu_state_pool_size(
                    ratio=4,
                    page_size=self.page_size,
                    max_num_reqs=self.max_running_requests,
                )
                c128_state_pool_size = npu_state_pool_size(
                    ratio=128,
                    page_size=self.page_size,
                    max_num_reqs=self.max_running_requests,
                )
            else:
                pool_cls = DeepSeekV4TokenToKVPool
                c4_state_pool_size = self.c4_state_pool_size
                c128_state_pool_size = self.c128_state_pool_size

            self.token_to_kv_pool = pool_cls(
                max_num_reqs=self.max_running_requests,
                # SWA ring is indexed by req_pool_idx; PD decode inflates req_to_token
                # past max_running_requests (pre-alloc), so size to the real capacity.
                num_req_slots=self.req_to_token_pool.req_to_token.shape[0],
                swa_size=self.swa_max_total_num_tokens,
                c4_size=self.c4_max_total_num_tokens,
                c128_size=self.c128_max_total_num_tokens,
                c4_state_pool_size=c4_state_pool_size,
                c128_state_pool_size=c128_state_pool_size,
                page_size=self.page_size,
                swa_page_size=swa_page_size,
                sliding_window=self.model_config.window_size,
                dtype=self.kv_cache_dtype,
                c4_state_dtype=self.c4_state_dtype,
                c128_state_dtype=self.c128_state_dtype,
                qk_nope_head_dim=self.model_config.qk_nope_head_dim,
                qk_rope_head_dim=self.model_config.qk_rope_head_dim,
                indexer_head_dim=self.model_config.index_head_dim,
                layer_num=self.num_effective_layers,
                device=self.device,
                enable_memory_saver=self.server_args.enable_memory_saver,
                compression_ratios=compression_ratios,
                start_layer=self.start_layer,
                end_layer=self.end_layer,
                enable_hisparse=self.enable_hisparse,
                online_mtp_max_draft_tokens=(
                    self.server_args.max_speculative_num_draft_tokens or 0
                ),
            )
        elif current_platform.is_out_of_tree() and not self.mambaish_config:
            if self.use_mla_backend and is_dsa_model:
                PoolCls = current_platform.get_dsa_kv_pool_cls()
                self.token_to_kv_pool = PoolCls(
                    self.max_total_num_tokens,
                    page_size=self.page_size,
                    dtype=self.kv_cache_dtype,
                    kv_lora_rank=self.model_config.kv_lora_rank,
                    qk_rope_head_dim=self.model_config.qk_rope_head_dim,
                    layer_num=self.num_effective_layers,
                    device=self.device,
                    kv_cache_dim=self.calculate_mla_kv_cache_dim(),
                    enable_memory_saver=self.server_args.enable_memory_saver,
                    start_layer=self.start_layer,
                    end_layer=self.end_layer,
                    index_head_dim=get_dsa_index_head_dim(self.model_config.hf_config),
                )
            elif self.use_mla_backend:
                PoolCls = current_platform.get_mla_kv_pool_cls()
                self.token_to_kv_pool = PoolCls(
                    self.max_total_num_tokens,
                    page_size=self.page_size,
                    dtype=self.kv_cache_dtype,
                    kv_lora_rank=self.model_config.kv_lora_rank,
                    qk_rope_head_dim=self.model_config.qk_rope_head_dim,
                    index_head_dim=(
                        self.model_config.index_head_dim if is_dsa_model else None
                    ),
                    layer_num=self.num_effective_layers,
                    device=self.device,
                    enable_memory_saver=self.server_args.enable_memory_saver,
                    start_layer=self.start_layer,
                    end_layer=self.end_layer,
                )
            else:
                PoolCls = current_platform.get_mha_kv_pool_cls()
                self.token_to_kv_pool = PoolCls(
                    self.max_total_num_tokens,
                    page_size=self.page_size,
                    dtype=self.kv_cache_dtype,
                    head_num=self.model_config.get_num_kv_heads(
                        get_attention_tp_size()
                    ),
                    head_dim=self.model_config.head_dim,
                    layer_num=self.num_effective_layers,
                    device=self.device,
                    enable_memory_saver=self.server_args.enable_memory_saver,
                    start_layer=self.start_layer,
                    end_layer=self.end_layer,
                )
        elif (
            self.server_args.attention_backend == "ascend" and not self.mambaish_config
        ):
            if self.is_hybrid_swa:
                from sglang.srt.hardware_backend.npu.memory_pool_npu import (
                    NPUMHATokenToKVPool,
                )

                kwargs = {}
                if self.is_hybrid_swa_compress:
                    kwargs = {
                        "swa_head_num": max(
                            1,
                            self.model_config.hf_text_config.swa_num_key_value_heads
                            // get_attention_tp_size(),
                        ),
                        "swa_head_dim": self.model_config.swa_head_dim,
                        "swa_v_head_dim": self.model_config.swa_v_head_dim,
                        "v_head_dim": self.model_config.v_head_dim,
                    }
                self.token_to_kv_pool = SWAKVPool(
                    size=self.full_max_total_num_tokens,
                    size_swa=self.swa_max_total_num_tokens,
                    page_size=self.page_size,
                    dtype=self.kv_cache_dtype,
                    head_num=self.model_config.get_num_kv_heads(
                        get_attention_tp_size()
                    ),
                    head_dim=self.model_config.head_dim,
                    swa_attention_layer_ids=self.model_config.swa_attention_layer_ids,
                    full_attention_layer_ids=self.model_config.full_attention_layer_ids,
                    device=self.device,
                    token_to_kv_pool_class=NPUMHATokenToKVPool,
                    **kwargs,
                )
            elif self.use_mla_backend:
                from sglang.srt.hardware_backend.npu.memory_pool_npu import (
                    NPUMLATokenToKVPool,
                )

                self.token_to_kv_pool = NPUMLATokenToKVPool(
                    self.max_total_num_tokens,
                    page_size=self.page_size,
                    dtype=self.kv_cache_dtype,
                    kv_lora_rank=self.model_config.kv_lora_rank,
                    qk_rope_head_dim=self.model_config.qk_rope_head_dim,
                    index_head_dim=(
                        self.model_config.index_head_dim if is_dsa_model else None
                    ),
                    layer_num=self.num_effective_layers,
                    device=self.device,
                    enable_memory_saver=self.server_args.enable_memory_saver,
                    start_layer=self.start_layer,
                    end_layer=self.end_layer,
                )
            else:
                from sglang.srt.hardware_backend.npu.memory_pool_npu import (
                    NPUMHATokenToKVPool,
                )

                self.token_to_kv_pool = NPUMHATokenToKVPool(
                    self.max_total_num_tokens,
                    page_size=self.page_size,
                    dtype=self.kv_cache_dtype,
                    head_num=self.model_config.get_num_kv_heads(
                        get_attention_tp_size()
                    ),
                    head_dim=self.model_config.head_dim,
                    layer_num=self.num_effective_layers,
                    device=self.device,
                    enable_memory_saver=self.server_args.enable_memory_saver,
                    start_layer=self.start_layer,
                    end_layer=self.end_layer,
                )
        elif self.use_mla_backend and is_dsa_model:
            PoolCls = (
                HiSparseDSATokenToKVPool if self.enable_hisparse else DSATokenToKVPool
            )
            pool_kwargs = {}
            if self.enable_hisparse:
                from sglang.srt.mem_cache.sparsity import parse_hisparse_config

                pool_kwargs["host_to_device_ratio"] = parse_hisparse_config(
                    self.server_args
                ).host_to_device_ratio
            self.token_to_kv_pool = PoolCls(
                self.max_total_num_tokens,
                page_size=self.page_size,
                dtype=self.kv_cache_dtype,
                kv_lora_rank=self.model_config.kv_lora_rank,
                qk_rope_head_dim=self.model_config.qk_rope_head_dim,
                layer_num=self.num_effective_layers,
                device=self.device,
                kv_cache_dim=self.calculate_mla_kv_cache_dim(),
                enable_memory_saver=self.server_args.enable_memory_saver,
                start_layer=self.start_layer,
                end_layer=self.end_layer,
                index_head_dim=get_dsa_index_head_dim(self.model_config.hf_config),
                **pool_kwargs,
            )
        elif self.use_mla_backend and not self.mambaish_config:
            assert not is_dsa_model
            if is_float4_e2m1fn_x2(self.kv_cache_dtype):
                self.token_to_kv_pool = MLATokenToKVPoolFP4(
                    self.max_total_num_tokens,
                    page_size=self.page_size,
                    dtype=self.kv_cache_dtype,
                    kv_lora_rank=self.model_config.kv_lora_rank,
                    qk_rope_head_dim=self.model_config.qk_rope_head_dim,
                    layer_num=self.num_effective_layers,
                    device=self.device,
                    enable_memory_saver=self.server_args.enable_memory_saver,
                    start_layer=self.start_layer,
                    end_layer=self.end_layer,
                )
            else:
                self.token_to_kv_pool = MLATokenToKVPool(
                    self.max_total_num_tokens,
                    page_size=self.page_size,
                    dtype=self.kv_cache_dtype,
                    kv_lora_rank=self.model_config.kv_lora_rank,
                    qk_rope_head_dim=self.model_config.qk_rope_head_dim,
                    layer_num=self.num_effective_layers,
                    device=self.device,
                    enable_memory_saver=self.server_args.enable_memory_saver,
                    start_layer=self.start_layer,
                    end_layer=self.end_layer,
                )
        else:
            if self.is_hybrid_swa:
                kwargs = {}
                if self.is_hybrid_swa_compress:
                    kwargs = {
                        "swa_head_num": max(
                            1,
                            self.model_config.hf_text_config.swa_num_key_value_heads
                            // get_attention_tp_size(),
                        ),
                        "swa_head_dim": self.model_config.swa_head_dim,
                        "swa_v_head_dim": self.model_config.swa_v_head_dim,
                        "v_head_dim": self.model_config.v_head_dim,
                    }
                self.token_to_kv_pool = SWAKVPool(
                    size=self.full_max_total_num_tokens,
                    size_swa=self.swa_max_total_num_tokens,
                    page_size=self.page_size,
                    dtype=self.kv_cache_dtype,
                    head_num=self.model_config.get_num_kv_heads(
                        get_attention_tp_size()
                    ),
                    head_dim=self.model_config.head_dim,
                    swa_attention_layer_ids=self.model_config.swa_attention_layer_ids,
                    full_attention_layer_ids=self.model_config.full_attention_layer_ids,
                    device=self.device,
                    enable_kv_cache_copy=(
                        self.server_args.speculative_algorithm is not None
                    ),
                    token_to_kv_pool_class=mha_pool_class,
                    **kwargs,
                )
            elif is_minimax_sparse(self.model_config.hf_config):
                _hf_config = self.model_config.hf_config
                sparse_cfg = get_minimax_sparse_attention_config(_hf_config)
                dense_layer_ids, sparse_layer_ids = get_minimax_sparse_layer_ids(
                    sparse_cfg
                )
                disable_value_sparse_layer_ids = (
                    get_minimax_sparse_disable_value_layer_ids(sparse_cfg)
                )
                self.token_to_kv_pool = MiniMaxSparseKVPool(
                    size=self.max_total_num_tokens,
                    page_size=self.page_size,
                    dtype=self.kv_cache_dtype,
                    index_dtype=self.dtype,
                    head_num=self.model_config.get_num_kv_heads(
                        get_attention_tp_size()
                    ),
                    head_dim=self.model_config.head_dim,
                    idx_head_dim=sparse_cfg["sparse_index_dim"],
                    dense_layer_ids=dense_layer_ids,
                    sparse_layer_ids=sparse_layer_ids,
                    disable_value_sparse_layer_ids=disable_value_sparse_layer_ids,
                    device=self.device,
                    enable_memory_saver=self.server_args.enable_memory_saver,
                    start_layer=self.start_layer,
                    end_layer=self.end_layer,
                )
            elif config := self.mambaish_config:
                extra_args = {}
                if self.use_mla_backend:
                    extra_args = {
                        "kv_lora_rank": self.model_config.kv_lora_rank,
                        "qk_rope_head_dim": self.model_config.qk_rope_head_dim,
                    }
                self.token_to_kv_pool = HybridLinearKVPool(
                    page_size=self.page_size,
                    size=self.max_total_num_tokens,
                    dtype=self.kv_cache_dtype,
                    head_num=self.model_config.get_num_kv_heads(
                        get_attention_tp_size()
                    ),
                    head_dim=self.model_config.head_dim,
                    # if draft worker, we only need 1 attention layer's kv pool
                    full_attention_layer_ids=(
                        [0]
                        if self.is_draft_worker
                        else [
                            i
                            for i in config.full_attention_layer_ids
                            if self.start_layer <= i < self.end_layer
                        ]
                    ),
                    device=self.device,
                    mamba_pool=self.req_to_token_pool.mamba_pool,
                    enable_memory_saver=self.server_args.enable_memory_saver,
                    enable_kv_cache_copy=(
                        self.server_args.speculative_algorithm is not None
                    ),
                    use_mla=self.use_mla_backend,
                    start_layer=self.start_layer,
                    full_kv_pool_class=mha_pool_class,
                    **extra_args,
                )
            else:
                if is_float4_e2m1fn_x2(self.kv_cache_dtype):
                    assert (
                        not enable_page_major
                    ), "page-major KV layout is not supported with fp4 KV cache"
                    self.token_to_kv_pool = MHATokenToKVPoolFP4(
                        self.max_total_num_tokens,
                        page_size=self.page_size,
                        dtype=self.kv_cache_dtype,
                        head_num=self.model_config.get_num_kv_heads(
                            get_attention_tp_size()
                        ),
                        head_dim=self.model_config.head_dim,
                        v_head_dim=self.model_config.v_head_dim,
                        layer_num=self.num_effective_layers,
                        device=self.device,
                        enable_memory_saver=self.server_args.enable_memory_saver,
                        start_layer=self.start_layer,
                        end_layer=self.end_layer,
                        enable_alt_stream=not self.server_args.enable_pdmux,
                        enable_kv_cache_copy=(
                            self.server_args.speculative_algorithm is not None
                        ),
                    )
                else:
                    pool_cls = (
                        NoOpMHATokenToKVPool
                        if self.server_args.prefill_only_disable_kv_cache
                        else mha_pool_class
                    )
                    self.token_to_kv_pool = pool_cls(
                        self.max_total_num_tokens,
                        page_size=self.page_size,
                        dtype=self.kv_cache_dtype,
                        head_num=self.model_config.get_num_kv_heads(
                            get_attention_tp_size()
                        ),
                        head_dim=self.model_config.head_dim,
                        v_head_dim=self.model_config.v_head_dim,
                        layer_num=self.num_effective_layers,
                        device=self.device,
                        enable_memory_saver=self.server_args.enable_memory_saver,
                        start_layer=self.start_layer,
                        end_layer=self.end_layer,
                        enable_alt_stream=not self.server_args.enable_pdmux,
                        enable_kv_cache_copy=(
                            self.server_args.speculative_algorithm is not None
                        ),
                    )

        # Initialize token_to_kv_pool_allocator
        need_sort = self.server_args.disaggregation_mode in ("decode", "prefill")
        if self.token_to_kv_pool_allocator is None:
            if current_platform.is_out_of_tree():
                AllocatorCls = current_platform.get_paged_allocator_cls()
                self.token_to_kv_pool_allocator = AllocatorCls(
                    self.max_total_num_tokens,
                    page_size=self.page_size,
                    dtype=self.kv_cache_dtype,
                    device=self.device,
                    kvcache=self.token_to_kv_pool,
                    need_sort=need_sort,
                )
            elif _is_npu and (
                self.server_args.attention_backend == "ascend"
                or is_dsv4_model
                or self.hybrid_gdn_config is not None
            ):
                if self.is_hybrid_swa:
                    # DSV4 on NPU: SWA allocator subclass that also drives the
                    # c4/c128 allocators, producing a DSV4OutCacheLoc per alloc.
                    if is_dsv4_model:
                        from sglang.srt.hardware_backend.npu.dsv4.dsv4_allocator import (
                            DSV4NPUTokenToKVPoolAllocator,
                        )

                        swa_allocator_cls = DSV4NPUTokenToKVPoolAllocator
                    else:
                        swa_allocator_cls = SWATokenToKVPoolAllocator
                    self.token_to_kv_pool_allocator = swa_allocator_cls(
                        self.full_max_total_num_tokens,
                        self.swa_max_total_num_tokens,
                        page_size=self.page_size,
                        dtype=self.kv_cache_dtype,
                        device=self.device,
                        kvcache=self.token_to_kv_pool,
                        need_sort=need_sort,
                    )
                else:
                    from sglang.srt.hardware_backend.npu.allocator_npu import (
                        NPUPagedTokenToKVPoolAllocator,
                    )

                    self.token_to_kv_pool_allocator = NPUPagedTokenToKVPoolAllocator(
                        self.max_total_num_tokens,
                        page_size=self.page_size,
                        dtype=self.kv_cache_dtype,
                        device=self.device,
                        kvcache=self.token_to_kv_pool,
                        need_sort=need_sort,
                    )
            else:
                if self.is_hybrid_swa and self.full_max_total_num_tokens == 0:
                    self.token_to_kv_pool_allocator = PureSWATokenToKVPoolAllocator(
                        self.swa_max_total_num_tokens,
                        page_size=self.page_size,
                        dtype=self.kv_cache_dtype,
                        device=self.device,
                        kvcache=self.token_to_kv_pool,
                        need_sort=need_sort,
                    )
                elif self.is_hybrid_swa:
                    self.token_to_kv_pool_allocator = SWATokenToKVPoolAllocator(
                        self.full_max_total_num_tokens,
                        self.swa_max_total_num_tokens,
                        page_size=self.page_size,
                        dtype=self.kv_cache_dtype,
                        device=self.device,
                        kvcache=self.token_to_kv_pool,
                        need_sort=need_sort,
                    )
                else:
                    if self.enable_hisparse:
                        from sglang.srt.mem_cache.sparsity import (
                            parse_hisparse_config,
                        )

                        hisparse_cfg = parse_hisparse_config(self.server_args)
                        self.token_to_kv_pool_allocator = (
                            HiSparseTokenToKVPoolAllocator(
                                self.max_total_num_tokens,
                                page_size=self.page_size,
                                dtype=self.kv_cache_dtype,
                                device=self.device,
                                kvcache=self.token_to_kv_pool,
                                need_sort=need_sort,
                                host_to_device_ratio=hisparse_cfg.host_to_device_ratio,
                            )
                        )
                    elif self.page_size == 1 and self.dcp_size == 1:
                        self.token_to_kv_pool_allocator = TokenToKVPoolAllocator(
                            self.max_total_num_tokens,
                            dtype=self.kv_cache_dtype,
                            device=self.device,
                            kvcache=self.token_to_kv_pool,
                            need_sort=need_sort,
                        )
                    else:
                        self.token_to_kv_pool_allocator = PagedTokenToKVPoolAllocator(
                            self.max_total_num_tokens * self.dcp_size,
                            page_size=self.page_size * self.dcp_size,
                            dtype=self.kv_cache_dtype,
                            device=self.device,
                            kvcache=self.token_to_kv_pool,
                            need_sort=need_sort,
                        )

            if self.enable_hisparse and is_dsv4_model:
                assert self.is_hybrid_swa, "DeepSeek V4 HiSparse requires SWA mode."
                self.token_to_kv_pool_allocator = (
                    DeepSeekV4HiSparseTokenToKVPoolAllocator(
                        self.token_to_kv_pool_allocator
                    )
                )

            # DSV4-NPU: wire allocator back-ref into req_to_token_pool so its
            # free(req) can release c4/c128 pool pages alongside the slot.
            if hasattr(self.req_to_token_pool, "register_dsv4_allocator"):
                self.req_to_token_pool.register_dsv4_allocator(
                    self.token_to_kv_pool_allocator
                )

        else:
            assert self.is_draft_worker
            if self.is_hybrid_swa:
                swa_allocator = getattr(
                    self.token_to_kv_pool_allocator,
                    "logical_attn_allocator",
                    self.token_to_kv_pool_allocator,
                )
                assert swa_allocator.__class__ == SWATokenToKVPoolAllocator
                self.token_to_kv_pool.full_to_swa_index_mapping = (
                    swa_allocator.full_to_swa_index_mapping
                )

        # Defensive check: the explicit validation above should reject known
        # unsupported pool families before allocation. Keep this guard here so
        # future pool-selection refactors fail at boot instead of on first use.
        if (
            self.server_args.prefill_only_disable_kv_cache
            and not self.is_draft_worker
            and not isinstance(self.token_to_kv_pool, NoOpMHATokenToKVPool)
        ):
            raise RuntimeError(
                "--prefill-only-disable-kv-cache expected NoOpMHATokenToKVPool but the "
                f"runtime pool is {type(self.token_to_kv_pool).__name__}. This pool "
                "family is not yet supported by --prefill-only-disable-kv-cache. "
                "Supported configurations today: plain MHA models on CUDA with the FA "
                "(fa3/fa4) prefill backend, --is-embedding, --chunked-prefill-size=-1, "
                "--disable-radix-cache, no context-parallel attention, no HiSparse, "
                "and --kv-cache-dtype != fp4_e2m1."
            )

    def _apply_token_constraints(self: ModelRunner, token_capacity: int) -> int:
        """Apply external constraints to token capacity: user cap, PP sync.

        Page alignment is handled by the configurator, not here.
        If constraints change the value, the configurator re-runs and re-aligns.
        """
        user_limit = self.server_args.max_total_tokens

        # Apply user-specified upper bound
        if user_limit is not None:
            if user_limit > token_capacity:
                logging.warning(
                    f"max_total_tokens={user_limit} is larger than the profiled value "
                    f"{token_capacity}. Use the profiled value instead."
                )
            token_capacity = min(token_capacity, user_limit)

        # Sync across PP ranks (each may have different layer counts)
        if self.pp_size > 1:
            tensor = torch.tensor(token_capacity, dtype=torch.int64)
            torch.distributed.all_reduce(
                tensor,
                op=torch.distributed.ReduceOp.MIN,
                group=get_world_group().cpu_group,
            )
            token_capacity = tensor.item()

        return token_capacity

    def _resolve_max_num_reqs(self: ModelRunner, token_capacity: int) -> int:
        """Compute max concurrent requests (per dp worker) from the finalized
        token capacity."""
        # Estimate pool size (used as upper bound when user specifies max_running_requests)
        estimated = int(token_capacity / self.model_config.context_len * 512)
        estimated = max(min(estimated, 4096), 2048)

        max_num_reqs = self.server_args.max_running_requests
        if max_num_reqs is not None:
            requested_per_worker = max_num_reqs // self.dp_size
            max_num_reqs = min(requested_per_worker, token_capacity // 2)
        else:
            requested_per_worker = None
            max_num_reqs = min(estimated, token_capacity // 2)

        if self.mambaish_config is not None:
            unified_n = getattr(self, "_unified_max_num_reqs", None)
            if unified_n is not None:
                # Unified + spec: admission was solved jointly against the
                # shared buffer's byte budget (_handle_max_mamba_cache_unified,
                # hard-floor per-request charge; radix headroom is evictable).
                # The static ratio clamp below would re-impose the worst-case
                # partition model the unified pool replaces.
                max_num_reqs = min(max_num_reqs, unified_n)
            else:
                ratio = self._calculate_mamba_ratio()
                max_num_reqs = min(
                    max_num_reqs, self.server_args.max_mamba_cache_size // ratio
                )

                if max_num_reqs <= 0:
                    raise RuntimeError(
                        f"Hybrid (mamba/linear-attention) state cache is too small to serve "
                        f"any requests. max_mamba_cache_size={self.server_args.max_mamba_cache_size}, "
                        f"mamba_ratio={ratio}, resulting max_num_reqs={max_num_reqs}. "
                        f"Try: (1) reduce --max-running-requests, "
                        f"(2) increase --mem-fraction-static, or "
                        f"(3) use GPUs with more memory."
                    )
        if requested_per_worker is not None and max_num_reqs < requested_per_worker:
            logger.warning(
                "max_running_requests was reduced from the requested %d to %d "
                "(per dp worker) due to the available KV cache capacity.",
                requested_per_worker,
                max_num_reqs,
            )
        return max_num_reqs

    def _apply_memory_pool_config(self: ModelRunner, config: MemoryPoolConfig):
        """Apply a resolved MemoryPoolConfig and initialize pools."""
        self.max_total_num_tokens = config.max_total_num_tokens
        self.max_running_requests = config.max_running_requests
        if self.is_hybrid_swa:
            self.full_max_total_num_tokens = config.full_max_total_num_tokens
            self.swa_max_total_num_tokens = config.swa_max_total_num_tokens
            # Unified path: the buffer's DIRECT byte budget (the token counts
            # above are feasibility/label values, not a partition). None on
            # non-unified and on configurators that still size by counts.
            self.unified_swa_total_bytes = getattr(
                config, "unified_total_bytes", None
            )

        # DSV4 compressed-attention pool sizes. Draft worker reuses target's
        # full/swa sizes but does NOT own c4/c128/state pools (those live on
        # the target rank only); zero them out regardless of what config holds.
        if self.is_draft_worker:
            self.c4_max_total_num_tokens = 0
            self.c128_max_total_num_tokens = 0
            self.c4_state_pool_size = 0
            self.c128_state_pool_size = 0
        else:
            self.c4_max_total_num_tokens = config.c4_max_total_num_tokens
            self.c128_max_total_num_tokens = config.c128_max_total_num_tokens
            self.c4_state_pool_size = config.c4_state_pool_size
            self.c128_state_pool_size = config.c128_state_pool_size

        # Draft worker does not own the compression-state pools, but keep the
        # dtype attributes initialized so _init_pools can share one code path.
        if is_deepseek_v4(self.model_config.hf_config):
            self.c4_state_dtype, self.c128_state_dtype = (
                _get_dsv4_compress_state_dtypes()
            )

        self._init_pools()

    def _resolve_memory_pool_config(
        self: ModelRunner, pre_model_load_memory: int
    ) -> MemoryPoolConfig:
        """Profile GPU memory and resolve all pool parameters into a config."""
        from sglang.srt.model_executor.pool_configurator import (
            create_memory_pool_configurator,
        )

        available_bytes = self._profile_available_bytes(pre_model_load_memory)
        page_size = self.server_args.page_size

        configurator = create_memory_pool_configurator(self)
        config = configurator.calculate_pool_sizes(available_bytes, page_size)

        # Apply external constraints (user cap, page alignment, PP sync)
        constrained = self._apply_token_constraints(config.max_total_num_tokens)
        if constrained != config.max_total_num_tokens:
            config = configurator.calculate_pool_sizes_from_max_tokens(
                constrained, page_size
            )

        config.max_running_requests = self._resolve_max_num_reqs(
            config.max_total_num_tokens
        )
        config.mem_fraction_static = self.server_args.mem_fraction_static
        return config

    def init_memory_pool(self: ModelRunner, pre_model_load_memory: int):
        if not self.spec_algorithm.is_none() and self.is_draft_worker:
            assert (
                self.memory_pool_config is not None
            ), "Draft worker requires memory_pool_config"
        else:
            self.memory_pool_config = self._resolve_memory_pool_config(
                pre_model_load_memory
            )

        self._apply_memory_pool_config(self.memory_pool_config)

        logger.info(
            f"Memory pool end. "
            f"avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
        )
