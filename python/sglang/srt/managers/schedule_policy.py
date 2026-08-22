from __future__ import annotations

import logging
from array import array

from sglang.srt.environ import envs
from sglang.srt.managers.prefill_delayer import PrefillDelayerSinglePassExecutor
from sglang.srt.runtime_context import get_disagg
from sglang.srt.utils import get_bool_env_var, is_hip

_ROUTING_KEY_POLICY_DEBUG_LOG = get_bool_env_var("SGLANG_ROUTING_KEY_POLICY_DEBUG_LOG")
logger = logging.getLogger(__name__)

# Copyright 2023-2024 SGLang Team
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
"""Request scheduler policy"""

import os
import random
from collections import Counter, defaultdict
from contextlib import contextmanager
from enum import Enum, auto
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Union

import msgspec
import torch

from sglang.srt.dllm.config import DllmConfig
from sglang.srt.layers.attention.dsa.utils import is_dsa_prefill_cp_in_seq_split
from sglang.srt.layers.utils.cp_utils import is_prefill_context_parallel_enabled
from sglang.srt.managers.schedule_batch import (
    Req,
    ScheduleBatch,
    split_cached_prefix_by_tier,
)
from sglang.srt.mem_cache.allocator.base import FitVerdict
from sglang.srt.mem_cache.allocator.hisparse import (
    DeepSeekV4HiSparseTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.allocator.swa import (
    PureSWATokenToKVPoolAllocator,
    SWATokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    InitLoadBackParams,
    InsertParams,
    MatchPrefixParams,
    zero_match_result,
)
from sglang.srt.mem_cache.multi_ended_allocator import (
    UnifiedMambaTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey, TreeNode
from sglang.srt.server_args import ServerArgs

if TYPE_CHECKING:
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator

# Clip the estimation of max_new_tokens for the request whose max_new_tokens is very large.
# This can prevent the server from being too conservative.
# Note that this only clips the estimation in the scheduler but does not change the stop
# condition. The request can still generate tokens until it hits the unclipped max_new_tokens.
CLIP_MAX_NEW_TOKENS = int(
    os.environ.get("SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION", "4096")
)

# Threshold for in-batch prefix cache.
# If a request has a matched prefix length (against existing cache) less than this value,
# the scheduler runs the in-batch prefix caching check for this request.
# If we set it to -1, it means we disable in-batch prefix caching.
IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD = int(
    os.environ.get("IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD", "32")
)

# Threshold for in-batch prefix cache.
# If a request has a matched prefix length (within the waiting queue) larger than this value,
# the scheduler deprioritizes this request
IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD = int(
    os.environ.get("IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD", "32")
)


IGNORE_EOS_RESERVE_TOKENS = 1
# AMD/HIP-only: the prefill tile-budget admission control is part of the AMD
# compact extend-attention work and is gated on HIP (see _check_prefill_tile_budget),
# so non-AMD vendors keep the exact legacy scheduler behavior.
_IS_HIP = is_hip()
PREFILL_TILE_BUDGET = envs.SGLANG_PREFILL_TILE_BUDGET.get()
PREFILL_TILE_BUDGET_MODE = envs.SGLANG_PREFILL_TILE_BUDGET_MODE.get().strip().lower()
if PREFILL_TILE_BUDGET_MODE not in {"legacy", "compact"}:
    logger.warning(
        "Unsupported SGLANG_PREFILL_TILE_BUDGET_MODE=%s. Falling back to compact.",
        PREFILL_TILE_BUDGET_MODE,
    )
    PREFILL_TILE_BUDGET_MODE = "compact"


def _ceil_div(value: int, divisor: int) -> int:
    return -(-value // divisor)


def estimate_prefill_extend_tile_metrics(
    extend_lens: List[int], block_m: int
) -> Dict[str, Union[int, float, List[int], None]]:
    """Estimate extend-attention query tiles per head for a prefill batch."""
    normalized_lens = [max(0, int(length)) for length in extend_lens]
    q_tiles = [
        _ceil_div(length, block_m) if length > 0 else 0 for length in normalized_lens
    ]
    legacy_tiles = len(q_tiles) * max(q_tiles) if q_tiles else 0
    compact_tiles = sum(q_tiles)
    saved_tiles = legacy_tiles - compact_tiles
    saved_ratio = saved_tiles / legacy_tiles if legacy_tiles else None
    return {
        "block_m": int(block_m),
        "request_count": len(normalized_lens),
        "extend_lens": normalized_lens,
        "q_tiles_per_request": q_tiles,
        "max_extend_len": max(normalized_lens) if normalized_lens else 0,
        "sum_extend_len": sum(normalized_lens),
        "legacy_q_tiles_per_head": legacy_tiles,
        "compact_q_tiles_per_head": compact_tiles,
        "saved_q_tiles_per_head": saved_tiles,
        "saved_q_tile_ratio": saved_ratio,
    }


def match_prefix_for_req(
    tree_cache: BasePrefixCache,
    req: Req,
    token_ids: Optional[array[int]] = None,
    *,
    cow_mamba: bool = False,
    include_req: bool = False,
):
    if token_ids is None:
        token_ids = req.origin_input_ids + req.output_ids

    # unified_kv SWA lives in a per-request ring that's not content-stable and is
    # never stored in the radix tree, so a reused prefix carries stale SWA. Cap
    # the match by the trailing sliding window so it gets re-prefilled, rewriting
    # this request's SWA ring. No-op for other layouts.
    reprefill_tail = tree_cache.swa_reprefill_tail_tokens()
    key_limit = max(0, len(token_ids) - reprefill_tail) if reprefill_tail else None

    match_result = tree_cache.match_prefix(
        MatchPrefixParams(
            key=RadixKey(
                token_ids=token_ids,
                extra_key=req.extra_key,
                limit=key_limit,
                cache_salt=req.cache_salt,
            ),
            cow_mamba=cow_mamba,
            req=req if include_req else None,
        )
    )
    if envs.SGLANG_RADIX_FORCE_MISS.get():
        match_result = zero_match_result(
            tree_cache, match_result, extra_key=req.extra_key
        )
    (
        req.prefix_indices,
        req.last_node,
        req.last_host_node,
        req.best_match_node,
        req.host_hit_length,
        req.swa_host_hit_length,
        req.mamba_host_hit_length,
    ) = (
        match_result.device_indices,
        match_result.last_device_node,
        match_result.last_host_node,
        match_result.best_match_node,
        match_result.host_hit_length,
        match_result.swa_host_hit_length,
        match_result.mamba_host_hit_length,
    )
    max_len = req._compute_max_prefix_len(len(token_ids))
    req.num_matched_prefix_tokens = min(
        len(req.prefix_indices) + req.host_hit_length, max_len
    )
    if match_result.mamba_branching_seqlen is not None:
        req.mamba_branching_seqlen = match_result.mamba_branching_seqlen
    if match_result.cache_protected_len is not None:
        req.cache_protected_len = match_result.cache_protected_len
    return match_result


class CacheAwarePolicy(Enum):
    """Scheduling policies that are aware of the tree cache."""

    LPM = "lpm"  # longest prefix match
    DFS_WEIGHT = "dfs-weight"  # depth-first search weighting


class CacheAgnosticPolicy(Enum):
    """Scheduling policies that are not aware of the tree cache."""

    FCFS = "fcfs"  # first come first serve
    LOF = "lof"  # longest output first
    RANDOM = "random"
    ROUTING_KEY = "routing-key"  # prioritize by routing key frequency in running batch


class SchedulePolicy:
    Policy = Union[CacheAwarePolicy, CacheAgnosticPolicy]

    def __init__(
        self,
        policy: str,
        tree_cache: BasePrefixCache,
        enable_hierarchical_cache: bool,
        enable_priority_scheduling: bool,
        schedule_low_priority_values_first: bool,
    ):
        self.policy = self._validate_and_adjust_policy(policy, tree_cache)
        self.tree_cache = tree_cache
        self.enable_hierarchical_cache = enable_hierarchical_cache
        self.enable_priority_scheduling = enable_priority_scheduling
        self.schedule_low_priority_values_first = schedule_low_priority_values_first
        self.priority_sign = 1 if schedule_low_priority_values_first else -1

        # It is used to find the matching prefix for in-batch prefix caching.
        self.waiting_queue_radix_tree = RadixCache.create_simulated()

    def calc_priority(
        self, waiting_queue: List[Req], running_batch: Optional[ScheduleBatch] = None
    ) -> None:
        policy = self._determine_active_policy(waiting_queue)

        # Populate req.num_matched_prefix_tokens at schedule time. Cache-aware policies
        # set it in _compute_prefix_matches; do the same full match for
        # cache-agnostic policies when the radix supports it, so the load
        # snapshot has it. Skip on decode (never prefills).
        if (
            not isinstance(policy, CacheAwarePolicy)
            and self.tree_cache.supports_fast_match_prefix()
            and get_disagg().disaggregation_mode != "decode"
        ):
            for r in waiting_queue:
                match_prefix_for_req(self.tree_cache, r, include_req=True)

        if self.policy == CacheAgnosticPolicy.FCFS:
            if self.enable_priority_scheduling:
                SchedulePolicy._sort_by_priority_and_fcfs(
                    waiting_queue, self.priority_sign
                )
            return

        if isinstance(policy, CacheAwarePolicy):
            temporary_deprioritized = self._compute_prefix_matches(
                waiting_queue, policy
            )
            if policy == CacheAwarePolicy.LPM:
                SchedulePolicy._sort_by_longest_prefix(
                    waiting_queue, temporary_deprioritized
                )
            elif policy == CacheAwarePolicy.DFS_WEIGHT:
                SchedulePolicy._sort_by_dfs_weight(waiting_queue, self.tree_cache)
            else:
                raise ValueError(f"Unknown CacheAware Policy: {policy=}")
        else:
            if policy == CacheAgnosticPolicy.FCFS:
                pass
            elif policy == CacheAgnosticPolicy.LOF:
                SchedulePolicy._sort_by_longest_output(
                    waiting_queue,
                    self.enable_priority_scheduling,
                    self.priority_sign,
                )
            elif policy == CacheAgnosticPolicy.RANDOM:
                SchedulePolicy._sort_randomly(waiting_queue)
            elif policy == CacheAgnosticPolicy.ROUTING_KEY:
                if running_batch is not None:
                    SchedulePolicy._sort_by_routing_key(waiting_queue, running_batch)
            else:
                raise ValueError(f"Unknown CacheAgnostic Policy: {policy=}")

    def _determine_active_policy(self, waiting_queue: List[Req]) -> Policy:
        if self.policy == CacheAwarePolicy.LPM and len(waiting_queue) > 128:
            # Turn off the expensive prefix matching and sorting when the #queue is large.
            return CacheAgnosticPolicy.FCFS
        return self.policy

    def _validate_and_adjust_policy(
        self, policy: str, tree_cache: BasePrefixCache
    ) -> Policy:
        """
        Validates the policy and adjusts it if necessary based on tree cache settings.
        """
        try:
            policy_enum = CacheAwarePolicy(policy)
            if getattr(tree_cache, "disable", True):
                # If tree_cache is disabled, using CacheAgnosticPolicy policy
                return CacheAgnosticPolicy.FCFS
            return policy_enum
        except ValueError:
            try:
                return CacheAgnosticPolicy(policy)
            except ValueError:
                raise ValueError(f"Unknown schedule_policy: {policy=}")

    def _compute_prefix_matches(
        self, waiting_queue: List[Req], policy: CacheAwarePolicy
    ) -> Set[int]:
        """
        Computes and caches the matching prefixes for requests in the waiting queue,
            and handles in-batch prefix caching logic.
        """
        temporary_deprioritized: Set[int] = set()
        self.waiting_queue_radix_tree.reset()

        for r in waiting_queue:
            prefix_ids = r.origin_input_ids + r.output_ids
            extra_key = r.extra_key
            cache_salt = r.cache_salt
            match_result = match_prefix_for_req(
                self.tree_cache, r, prefix_ids, include_req=True
            )

            # NOTE(sang): This logic is for in-batch prefix caching;
            # If there are more than 1 request that have small matching prefix from
            # existing cache, but all those requests share the same prefix, we prefer
            # to schedule only one of them so that we can increase the cache hit rate.
            # We prefer to set IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD > 0 because too small
            # threshold means we cannot use in-batch prefix caching for short prefixes.
            # It is kind of common when the engine is long running (e.g., imagine the prefix "the").
            if len(r.prefix_indices) <= IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD:
                match_result = self.waiting_queue_radix_tree.match_prefix(
                    MatchPrefixParams(
                        key=RadixKey(
                            token_ids=prefix_ids,
                            extra_key=extra_key,
                            cache_salt=cache_salt,
                        )
                    )
                )
                if envs.SGLANG_RADIX_FORCE_MISS.get():
                    match_result = zero_match_result(
                        self.waiting_queue_radix_tree, match_result, extra_key=extra_key
                    )
                in_batch_matching_prefixes = match_result.device_indices
                if (
                    len(in_batch_matching_prefixes)
                    >= IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD
                ):
                    temporary_deprioritized.add(r.rid)
                else:
                    # Insert with a dummy key
                    self.waiting_queue_radix_tree.insert(
                        InsertParams(
                            key=RadixKey(
                                token_ids=prefix_ids,
                                extra_key=extra_key,
                                cache_salt=cache_salt,
                            ),
                            value=torch.empty(len(prefix_ids), dtype=torch.bool),
                        )
                    )
        return temporary_deprioritized

    @staticmethod
    def _sort_by_longest_prefix(
        waiting_queue: List[Req], temporary_deprioritized: Set[int]
    ) -> None:
        """Sorts the waiting queue based on the longest prefix match."""
        waiting_queue.sort(
            key=lambda r: (
                -r.num_matched_prefix_tokens
                if r.rid not in temporary_deprioritized
                else float("inf")
            )
        )

    @staticmethod
    def _sort_by_dfs_weight(
        waiting_queue: List[Req], tree_cache: BasePrefixCache
    ) -> None:
        """Sorts the waiting queue based on a depth-first search weighting."""
        last_node_to_reqs = defaultdict(list)
        for req in waiting_queue:
            last_node = tree_cache.resolve_node_handle(req.last_node)
            last_node_to_reqs[last_node].append(req)

        node_to_weight = defaultdict(int)
        for node in last_node_to_reqs:
            node_to_weight[node] = len(last_node_to_reqs[node])
        SchedulePolicy._calc_weight(tree_cache.root_node, node_to_weight)

        waiting_queue.clear()
        SchedulePolicy._get_dfs_priority(
            tree_cache.root_node,
            node_to_weight,
            last_node_to_reqs,
            waiting_queue,
        )

    @staticmethod
    def _sort_by_longest_output(
        waiting_queue: List[Req],
        enable_priority_scheduling: bool,
        priority_sign: int,
    ) -> None:
        """Sorts the waiting queue based on the longest output (max_new_tokens). If using priority scheduling, sort by priority first."""
        if enable_priority_scheduling:
            waiting_queue.sort(
                key=lambda x: (
                    x.priority * priority_sign,
                    -x.sampling_params.max_new_tokens,
                )
            )
        else:
            waiting_queue.sort(key=lambda x: -x.sampling_params.max_new_tokens)

    @staticmethod
    def _sort_randomly(waiting_queue: List[Req]) -> None:
        """Shuffles the waiting queue randomly."""
        random.shuffle(waiting_queue)

    @staticmethod
    def _sort_by_priority_and_fcfs(
        waiting_queue: List[Req], priority_sign: int
    ) -> None:
        """Sorts the waiting queue based on the request priority then received titmestamp."""
        waiting_queue.sort(
            key=lambda x: (
                x.priority * priority_sign,
                x.time_stats.wait_queue_entry_time,
            )
        )

    @staticmethod
    def _sort_by_routing_key(
        waiting_queue: List[Req], running_batch: ScheduleBatch
    ) -> None:
        """Sorts waiting queue by routing key frequency in running batch."""
        routing_key_counts = Counter(
            r.routing_key for r in running_batch.reqs if r.routing_key
        )

        if _ROUTING_KEY_POLICY_DEBUG_LOG:
            waiting_keys_before = [r.routing_key for r in waiting_queue]
            logger.info(
                f"routing_key_counts={dict(routing_key_counts)}, "
                f"waiting_keys_before={waiting_keys_before}"
            )

        if not routing_key_counts:
            return

        def sort_key(req: Req):
            key = req.routing_key
            if key and key in routing_key_counts:
                count = routing_key_counts[key]
                return (0, -count, key)
            else:
                return (1, 0, key or "")

        waiting_queue.sort(key=sort_key)

        if _ROUTING_KEY_POLICY_DEBUG_LOG:
            waiting_keys_after = [r.routing_key for r in waiting_queue]
            logger.info(f"waiting_keys_after={waiting_keys_after}")

    @staticmethod
    def _calc_weight(cur_node: TreeNode, node_to_weight: Dict[TreeNode, int]) -> None:
        for child in cur_node.children.values():
            SchedulePolicy._calc_weight(child, node_to_weight)
            node_to_weight[cur_node] += node_to_weight[child]

    @staticmethod
    def _get_dfs_priority(
        cur_node: TreeNode,
        node_to_priority: Dict[TreeNode, int],
        last_node_to_reqs: Dict[TreeNode, List[Req]],
        q: List,
    ) -> None:
        children = [child for child in cur_node.children.values()]
        children.sort(key=lambda x: -node_to_priority[x])
        for child in children:
            SchedulePolicy._get_dfs_priority(
                child, node_to_priority, last_node_to_reqs, q
            )
        q.extend(last_node_to_reqs[cur_node])


class AddReqResult(Enum):
    CONTINUE = auto()  # Continue to add requests
    NO_TOKEN = auto()  # No token left
    OTHER = auto()  # Other reasons to stop adding requests


class TokenPrefillCost(msgspec.Struct, frozen=True):
    """Gate-time admission cost of one candidate, token-denominated (the
    default budget's currency). Quantities mirror the pre-extraction gates:
    `total_tokens` is `add_one_req`'s lifetime gate quantity (extend +
    estimated decode + page overhead + the mamba-slot token-equivalent),
    `paged_input_tokens` is `add_one_req_ignore_eos`'s immediate quantity,
    `swa_needed` the sliding-window budget."""

    total_tokens: int
    paged_input_tokens: int
    swa_needed: int


class TokenAdmissionBudget:
    """Default admission budget: the scheduler's token-denominated memory
    gates and offset bookkeeping, extracted from the `PrefillAdder` main flow.

    One instance per prefill pass, built by
    `BaseTokenToKVPoolAllocator.make_admission_budget`; the adder routes every
    MEMORY admission decision through the budget protocol (`prefill_cost` /
    `fits` / `fits_immediate` / `charge_admitted` / `exhausted` /
    `chunk_admission_cap` / `swa_chunk_cap` / running-batch and preemption
    charges), so an allocator can substitute a different accounting without
    any allocator-specific logic in the adder. Non-memory knobs (input /
    chunk / dllm budgets, tile caps, delayer) stay on the adder.

    This default reads the adder's token views (`rem_total_tokens` /
    `cur_rem_tokens` / `rem_swa_tokens` — upstream logic that stays on the
    adder) and mutates the adder's offsets, reproducing the pre-extraction
    behavior bit-for-bit — including the shared-Mamba-composite
    full-token-equivalent cross-charge (`mamba_slot_full_token_cost`) and the
    separate fresh-mamba-slot gate.
    """

    def __init__(self, adder: PrefillAdder):
        self.adder = adder
        # Unified-pool joint budget: a new mamba state consumes shared-gap
        # bytes that `rem_total_tokens` (full KV) otherwise counts as free, so
        # reserve the gap per new mamba slot or admission over-commits. Gate on
        # the ALLOCATOR being the unified Mamba composite, NOT on
        # `is_hybrid_ssm_cache` (False for `ChunkCache`, which would skip the
        # reservation on the chunk-cache path): the gap coupling is a property
        # of the byte buffer.
        self._mamba_slot_cost = 0
        allocator = adder.token_to_kv_pool_allocator
        if isinstance(allocator, UnifiedMambaTokenToKVPoolAllocator):
            self._mamba_slot_cost = allocator.mamba_slot_full_token_cost()
        # `mamba_gap_reserve` is charged to `rem_total_tokens`, which INCLUDES
        # `full_evictable_size()` — but `alloc_req_slots` can only recover
        # MAMBA-recoverable bytes for a mamba slot (shared gap + peer holes +
        # mamba-evictable radix), NOT full-evictable. Gate new mamba slots on
        # that mamba-recoverable budget separately or an over-admit hits the
        # fail-loud `RuntimeError`. `None` outside the unified Mamba pool.
        self.rem_mamba_slots = None
        if self._mamba_slot_cost:
            self.rem_mamba_slots = (
                allocator.mamba_allocator.schedulable_available_size()
            )
            if adder.is_hybrid_ssm_cache:
                self.rem_mamba_slots += adder.tree_cache.mamba_evictable_size()

    # -- running-batch and preemption charges --

    def _running_request_total_token_offset(self, req: Req) -> int:
        return (
            min(
                (req.sampling_params.max_new_tokens - len(req.output_ids)),
                CLIP_MAX_NEW_TOKENS,
            )
            * self.adder.new_token_ratio
        )

    def charge_running(self, reqs: List[Req]) -> None:
        """Charge every running request's estimated remaining decode to the
        lifetime budget."""
        self.adder.rem_total_token_offset += sum(
            self._running_request_total_token_offset(r) for r in reqs
        )

    def preemption_demand(self, req: Req, extend_len: int) -> int:
        """Tokens preemption must free for `req` to fit (<= 0 == fits)."""
        return (
            extend_len
            + min(req.sampling_params.max_new_tokens, CLIP_MAX_NEW_TOKENS)
            - self.adder.rem_total_tokens
        )

    def running_request_credit(self, req: Req) -> int:
        """Budget credit released by preempting one running request."""
        return self._running_request_total_token_offset(req)

    def uncharge_running(self, req: Req) -> None:
        """Release a preempted running request's per-pass charge."""
        self.adder.rem_total_token_offset -= self._running_request_total_token_offset(
            req
        )

    # -- per-candidate cost / gates / charge --

    def new_mamba_state(self, req: Req) -> bool:
        """Whether admitting `req` allocates a fresh mamba state slot
        (`mamba_pool_idx is None`, mirroring `HybridReqToTokenPool.alloc`);
        False keeps baseline / SWA / non-Mamba unchanged."""
        return bool(self._mamba_slot_cost) and req.mamba_pool_idx is None

    def _mamba_gap_reserve(self, new_mamba_state: bool) -> int:
        """Shared-gap reservation (full-token-equivalents) for a request's new
        mamba state; 0 keeps baseline / SWA / non-Mamba unchanged.
        Conservative by design (`mamba_slot_full_token_cost` rounds UP). Does
        NOT reserve radix COW headroom or locked-but-evictable bytes — that
        residual is backstopped by the fail-loud RuntimeError in
        `alloc_req_slots`."""
        return self._mamba_slot_cost if new_mamba_state else 0

    def prefill_cost(
        self,
        *,
        cand_extend_input_len: int,
        swa_extend_input_len: int,
        max_new_tokens: int,
        swa_host_hit_length: int = 0,
        new_mamba_state: bool = False,
    ) -> TokenPrefillCost:
        """Gate-time cost of one candidate. `swa_extend_input_len` is the
        length the sliding-window budget is computed from (`add_one_req`
        passes the host-hit-adjusted length; the ignore-eos path the raw
        candidate length)."""
        adder = self.adder
        swa_needed = 0
        if adder.is_hybrid_swa:
            swa_needed = adder._swa_budget_for_req(
                swa_extend_input_len,
                max_new_tokens,
                swa_host_hit_length=swa_host_hit_length,
            )
        gap = self._mamba_gap_reserve(new_mamba_state)
        return TokenPrefillCost(
            total_tokens=cand_extend_input_len + max_new_tokens + adder.page_size + gap,
            paged_input_tokens=adder.ceil_paged_tokens(cand_extend_input_len) + gap,
            swa_needed=swa_needed,
        )

    def fits(self, cost: TokenPrefillCost) -> FitVerdict:
        """`add_one_req`'s memory gate (lifetime budget + swa side)."""
        adder = self.adder
        if cost.total_tokens >= adder.rem_total_tokens:
            return FitVerdict(ok=False)
        if adder.is_hybrid_swa and cost.swa_needed >= adder.rem_swa_tokens:
            return FitVerdict(ok=False, swa_binding=True)
        return FitVerdict(ok=True)

    def fits_immediate(self, cost: TokenPrefillCost) -> FitVerdict:
        """`add_one_req_ignore_eos`'s memory gate (immediate + lifetime
        floor)."""
        adder = self.adder
        if cost.paged_input_tokens > min(adder.cur_rem_tokens, adder.rem_total_tokens):
            return FitVerdict(ok=False)
        if adder.is_hybrid_swa and cost.swa_needed > adder.rem_swa_tokens:
            return FitVerdict(ok=False, swa_binding=True)
        return FitVerdict(ok=True)

    def charge_admitted(
        self,
        *,
        extend_input_len: int,
        max_new_tokens: int,
        new_mamba_state: bool,
    ) -> None:
        """Charge an admitted candidate. `extend_input_len` is the ADMITTED
        (possibly chunk-truncated) length, page-ceiled by the caller; the
        swa-side budget is recomputed from it, mirroring pre-extraction
        behavior. The new mamba slot is charged to BOTH full budgets (the
        slot is allocated immediately and held for the request lifetime) and
        consumes one mamba-recoverable slot, gated separately."""
        adder = self.adder
        gap = self._mamba_gap_reserve(new_mamba_state)
        # alloc_extend reserves an extra page_size per request to make sure
        # the budget doesn't over-commit.
        page_overhead = adder.page_size
        adder.rem_total_token_offset += (
            extend_input_len + max_new_tokens + page_overhead + gap
        )
        adder.cur_rem_token_offset += extend_input_len + page_overhead + gap
        if gap and self.rem_mamba_slots is not None:
            self.rem_mamba_slots -= 1
        if adder.is_hybrid_swa:
            adder.rem_swa_token_offset += adder._swa_budget_for_req(
                extend_input_len, max_new_tokens
            )

    def exhausted(self) -> bool:
        """`budget_state`'s no-token condition."""
        adder = self.adder
        no_token = adder.rem_total_tokens <= 0 or adder.cur_rem_tokens <= 0
        if not no_token and adder.is_hybrid_swa:
            no_token = adder.rem_swa_tokens <= 0
        # Gate new mamba slots separately: rem_total_tokens' full_evictable
        # can't cover a mamba slot, which needs mamba-recoverable bytes.
        if not no_token and self.rem_mamba_slots is not None:
            no_token = self.rem_mamba_slots <= 0
        return no_token

    # -- chunked-prefill length derivation --

    def chunk_admission_cap(self, chunk_tokens: int) -> int:
        """Memory-bound cap on a chunked-prefill admission this pass (the one
        gate that DERIVES the admitted length from the budget)."""
        adder = self.adder
        v = min(chunk_tokens, int(adder.rem_total_tokens))
        if adder.is_hybrid_swa:
            # alloc_extend needs extend_num_tokens + page_size per request,
            # so reserve one page here to avoid OOM.
            v = min(v, int(adder.rem_swa_tokens) - adder.page_size)
        return v

    def swa_chunk_cap(self, max_new_tokens: int, swa_host_hit_length: int = 0) -> int:
        """Largest page-aligned extend chunk the SWA side can admit right now,
        keeping a sliding window of headroom; 0 if not even one page fits.

        Escape hatch for a request whose budget can never pass the admission
        gate (extend near/above the pool size, or a large load-back charge):
        without shrinking its chunk it would be rejected forever (head-of-line
        livelock). Shrinking is sound because past a chunk boundary only the
        sliding window stays locked — the rest turns evictable — so each
        pass's transient footprint fits the pool. `extend_input_len=0` in the
        reservation: this solves for the extend chunk itself, so the reserved
        headroom is the post-chunk decode window only."""
        adder = self.adder
        cap = int(adder.rem_swa_tokens) - adder._swa_reserved_tokens(
            0, max_new_tokens, swa_host_hit_length
        )
        if cap <= 0:
            return 0
        return cap // adder.page_size * adder.page_size

    # -- token-denominated floors for scheduler heuristics --

    def solvency_cur_rem(self) -> int:
        """This-pass remaining budget in tokens — the ignore-eos solvency
        heuristic's input."""
        return self.adder.cur_rem_tokens

    def total_tokens_floor(self) -> int:
        """Lifetime remaining in tokens — dllm chunk sizing and metrics."""
        return int(self.adder.rem_total_tokens)


class PrefillAdder:
    def __init__(
        self,
        page_size: int,
        tree_cache: BasePrefixCache,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        running_batch: ScheduleBatch,
        new_token_ratio: float,
        rem_input_tokens: int,
        rem_chunk_tokens: Optional[int],
        num_mixed_decode_tokens: int = 0,
        priority_scheduling_preemption_threshold: int = 0,
        max_prefill_bs: int = 0,
        max_running_requests: Optional[int] = None,
        prefill_max_requests: Optional[int] = None,
        prefill_delayer_single_pass: Optional[PrefillDelayerSinglePassExecutor] = None,
        dllm_config: Optional[DllmConfig] = None,
        waiting_queue_len: int = 0,
        prefill_tile_block_m: int = 64,
    ):
        self.page_size = page_size
        self.prefill_tile_block_m = prefill_tile_block_m
        self.tree_cache = tree_cache
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.running_batch = running_batch
        self.new_token_ratio = new_token_ratio
        self.rem_input_tokens = rem_input_tokens - num_mixed_decode_tokens
        self.rem_chunk_tokens = rem_chunk_tokens
        self.dllm_config = dllm_config

        if self.dllm_config is not None:
            self._init_dllm_meta(dllm_config)

        if self.rem_chunk_tokens is not None:
            self.rem_chunk_tokens -= num_mixed_decode_tokens
        # Kept for the budgets: the byte budget charges the mixed-decode
        # tokens allocated this pass in its own currency.
        self.num_mixed_decode_tokens = num_mixed_decode_tokens
        self.rem_total_token_offset = num_mixed_decode_tokens
        self.cur_rem_token_offset = num_mixed_decode_tokens

        self.req_states = None
        self.can_run_list = []
        self.preempt_list = []
        self.new_chunked_req = None
        self.log_hit_tokens = 0
        self.reprocessed_log_hit_tokens = 0
        self.log_device_hit_tokens = 0
        self.log_host_hit_tokens = 0
        self.log_storage_hit_tokens = 0
        # TODO(lsyin): report the real input tokens excluding page alignment
        self.log_input_tokens = 0
        self.reprocessed_log_input_tokens = 0

        # DeepSeek V4 HiSparse wraps an SWATokenToKVPoolAllocator internally and
        # exposes the full SWA allocator interface.
        self.is_hybrid_swa = isinstance(
            self.token_to_kv_pool_allocator,
            (SWATokenToKVPoolAllocator, DeepSeekV4HiSparseTokenToKVPoolAllocator),
        )
        self.is_all_swa = isinstance(
            self.token_to_kv_pool_allocator, PureSWATokenToKVPoolAllocator
        )
        self.is_hybrid_ssm_cache = self.tree_cache.supports_mamba()

        self.rem_swa_token_offset = 0

        # Every MEMORY admission decision routes through the budget the
        # allocator hands out (family flags above must be set first — the
        # default token budget reads them). Unified composites substitute a
        # byte-denominated budget here; the scheduler side stays feature-free.
        self.budget = self.token_to_kv_pool_allocator.make_admission_budget(adder=self)

        if running_batch is not None:
            # Estimate the offset in the remaining token space
            self.budget.charge_running(running_batch.reqs)

        self.priority_scheduling_preemption_threshold = (
            priority_scheduling_preemption_threshold
        )
        self.dsa_prefill_cp_in_seq_split = is_dsa_prefill_cp_in_seq_split()
        self.max_running_requests = max_running_requests
        self.prefill_context_parallel_enabled = is_prefill_context_parallel_enabled()
        self.prefill_max_requests = prefill_max_requests
        self.prefill_delayer_single_pass = prefill_delayer_single_pass
        self.max_prefill_bs = max_prefill_bs
        # Snapshot of scheduler waiting_queue length at the start of this
        # prefill pass. Used by PrefillDelayer's queue-based trigger.
        self.waiting_queue_len = waiting_queue_len

    def _admitted_extend_lens(self) -> List[int]:
        return [int(getattr(req, "extend_input_len", 0)) for req in self.can_run_list]

    def _tile_admission_metric_key(self) -> str:
        return f"{PREFILL_TILE_BUDGET_MODE}_q_tiles_per_head"

    def _candidate_tile_metrics(self, candidate_extend_len: int) -> Dict[str, object]:
        return estimate_prefill_extend_tile_metrics(
            [*self._admitted_extend_lens(), int(candidate_extend_len)],
            block_m=self.prefill_tile_block_m,
        )

    def _check_prefill_tile_budget(
        self, candidate_extend_len: int
    ) -> Optional[AddReqResult]:
        # AMD-only: leave non-AMD scheduler admission unchanged even if the env
        # budget is set.
        if not _IS_HIP or PREFILL_TILE_BUDGET <= 0:
            return None

        if not self.can_run_list:
            return None

        metrics = self._candidate_tile_metrics(candidate_extend_len)
        candidate_metric = int(metrics.get(self._tile_admission_metric_key()) or 0)
        if candidate_metric <= PREFILL_TILE_BUDGET:
            return None

        return AddReqResult.OTHER

    def _init_dllm_meta(self, dllm_config: DllmConfig):
        self.dllm_block_size = dllm_config.block_size
        max_running_reqs = dllm_config.max_running_requests

        self.rem_dllm_tokens = max_running_reqs * self.dllm_block_size

    @property
    def rem_total_tokens(self):
        if self.is_all_swa:
            available_and_evictable = (
                self.token_to_kv_pool_allocator.swa_available_size()
                + self.tree_cache.swa_evictable_size()
            )
        elif self.is_hybrid_swa:
            available_and_evictable = (
                self.token_to_kv_pool_allocator.full_available_size()
                + self.tree_cache.full_evictable_size()
            )
        elif self.is_hybrid_ssm_cache:
            available_and_evictable = (
                self.token_to_kv_pool_allocator.available_size()
                + self.tree_cache.full_evictable_size()
            )
        else:
            available_and_evictable = (
                self.token_to_kv_pool_allocator.available_size()
                + self.tree_cache.evictable_size()
            )
        return available_and_evictable - self.rem_total_token_offset

    @property
    def rem_swa_tokens(self):
        return (
            self.token_to_kv_pool_allocator.swa_available_size()
            + self.tree_cache.swa_evictable_size()
            - self.rem_swa_token_offset
        )

    @property
    def cur_rem_tokens(self):
        if self.is_all_swa:
            available_and_evictable = (
                self.token_to_kv_pool_allocator.swa_available_size()
                + self.tree_cache.swa_evictable_size()
            )
        elif self.is_hybrid_swa:
            available_and_evictable = (
                self.token_to_kv_pool_allocator.full_available_size()
                + self.tree_cache.full_evictable_size()
            )
        elif self.is_hybrid_ssm_cache:
            available_and_evictable = (
                self.token_to_kv_pool_allocator.available_size()
                + self.tree_cache.full_evictable_size()
            )
        else:
            available_and_evictable = (
                self.token_to_kv_pool_allocator.available_size()
                + self.tree_cache.evictable_size()
            )

        return available_and_evictable - self.cur_rem_token_offset

    def _swa_budget_for_req(
        self, extend_input_len: int, max_new_tokens: int, swa_host_hit_length: int = 0
    ) -> int:
        """SWA pool budget per request. Only valid when is_hybrid_swa is True.

        With chunked prefill + overlap scheduler, the peak SWA occupancy is:
          chunk N (running, not yet in tree) + sliding window (locked in tree)
          + chunk N+1 (new allocation)
        Since chunk N and locked tokens are already excluded from
        swa_available + swa_evictable, the budget only needs to cover the
        chunk N+1 allocation plus decode headroom:

          budget = max(alloc - window, 0) + min(extend + max_new_tokens, window) + page

        where alloc = min(extend, rem_chunk); the min() cap keeps the two terms
        from double-counting extend, so budget <= extend + max_new_tokens + page.
        """
        if self.rem_chunk_tokens is not None:
            alloc = min(extend_input_len, self.rem_chunk_tokens)
        else:
            alloc = extend_input_len
        window = self.tree_cache.sliding_window_size
        return max(alloc - window, 0) + self._swa_reserved_tokens(
            extend_input_len, max_new_tokens, swa_host_hit_length
        )

    def _swa_reserved_tokens(
        self, extend_input_len: int, max_new_tokens: int, swa_host_hit_length: int = 0
    ) -> int:
        """SWA slots a request adds to its own sliding window + page slack + the
        load-back charge. Shared floor of _swa_budget_for_req and _swa_chunk_cap.

        The headroom is min(extend + decode, window), not a constant window: a
        request contributes only extend + decode fresh tokens to its window and
        a cached SWA prefix funds the rest. Charging a full window double-counted
        a short cached-prefix resume and livelocked admission at a ~2-window
        pool; keeping extend in the min() holds the reservation >= the prefill
        allocation so an admitted request cannot OOM."""
        window = self.tree_cache.sliding_window_size
        headroom = min(extend_input_len + max_new_tokens, window)
        reserved = headroom + self.page_size
        if swa_host_hit_length > 0:
            reserved += self.ceil_paged_tokens(swa_host_hit_length)
        return reserved

    def _swa_new_tokens(self, req: Req) -> int:
        """Tokens a request may still decode, for SWA headroom sizing. Mirrors
        the remaining-then-clip order of add_one_req's max_new: clip-then-subtract
        would zero out a request that has already generated >= CLIP tokens but
        still has a long decode ahead, under-reserving its window."""
        return min(
            max(req.sampling_params.max_new_tokens - len(req.output_ids), 0),
            CLIP_MAX_NEW_TOKENS,
        )

    def _swa_req_never_fits(
        self, extend_input_len: int, max_new_tokens: int, swa_host_hit_length: int = 0
    ) -> bool:
        """True when a request's SWA budget exceeds the *entire* SWA pool, so it
        can never be admitted whole no matter how far the pool drains.

        This is the head-of-line livelock the _swa_chunk_cap escape hatch exists
        for; the hatch must fire only in this case. A request that merely
        exceeds *current* rem_swa (transient pressure) would fit once running
        decodes free their windows, so it must wait — admitting it into the
        decode headroom collapses the SWA evictable cushion and forces running
        requests to retract (observed as a severe retraction/re-prefill storm on
        hybrid-SWA models at high concurrency)."""
        capacity = self.token_to_kv_pool_allocator.size_swa
        return (
            self._swa_budget_for_req(
                extend_input_len, max_new_tokens, swa_host_hit_length
            )
            >= capacity
        )

    def ceil_paged_tokens(self, tokens: int) -> int:
        return -(-tokens // self.page_size) * self.page_size

    def budget_state(self):
        if self.budget.exhausted():
            return AddReqResult.NO_TOKEN

        if self.rem_input_tokens <= 0:
            return AddReqResult.OTHER

        if self.dllm_config is not None:
            if self.rem_dllm_tokens <= 0:
                return AddReqResult.OTHER
        else:
            if self.rem_chunk_tokens is not None and self.rem_chunk_tokens <= 0:
                return AddReqResult.OTHER

        return AddReqResult.CONTINUE

    def _update_prefill_budget(
        self,
        prefix_len: int,
        extend_input_len: int,
        max_new_tokens: int,
        retracted_stain: bool,
        new_mamba_state: bool = False,
        host_hit_len: int = 0,
        storage_hit_len: int = 0,
    ):
        # TODO(lsyin): check this workaround logic, which only ensures the prefill will not out of memory, and may be too conservative
        extend_input_len = self.ceil_paged_tokens(extend_input_len)

        self.budget.charge_admitted(
            extend_input_len=extend_input_len,
            max_new_tokens=max_new_tokens,
            new_mamba_state=new_mamba_state,
        )
        self.rem_input_tokens -= extend_input_len

        if self.dllm_config is not None:
            self.rem_dllm_tokens -= extend_input_len
        elif self.rem_chunk_tokens is not None:
            self.rem_chunk_tokens -= extend_input_len

        # reprocessed_log_* is a subset of log_*; metrics_reporter subtracts it
        # when computing the first-attempt prefix cache hit rate.
        self.log_hit_tokens += prefix_len
        self.log_input_tokens += extend_input_len
        if retracted_stain:
            self.reprocessed_log_hit_tokens += prefix_len
            self.reprocessed_log_input_tokens += extend_input_len
        elif prefix_len > 0:
            device_hit, host_hit, storage_hit = split_cached_prefix_by_tier(
                prefix_len=prefix_len,
                host_hit_len=host_hit_len,
                storage_hit_len=storage_hit_len,
            )
            self.log_device_hit_tokens += device_hit
            self.log_host_hit_tokens += host_hit
            self.log_storage_hit_tokens += storage_hit

    def _get_dllm_remain_tokens(self) -> int:
        _rem_tokens = min(
            self.rem_dllm_tokens,
            self.dllm_block_size,
            self.budget.total_tokens_floor(),
        )
        if _rem_tokens <= 0:
            _rem_tokens = self.rem_dllm_tokens

        return _rem_tokens

    def _add_dllm_req(self, req: Req, prefix_len: int):
        # FIXME: consider the case when rem_dllm_tokens < dllm_block_size,
        # the diffusion unmask process may have some problems
        # Make sure at least one page is available
        trunc_len = (
            min(self.rem_dllm_tokens, self.dllm_block_size)
            // self.page_size
            * self.page_size
        )

        req.set_extend_range(prefix_len, prefix_len + trunc_len)

        self.can_run_list.append(req)

        self._update_prefill_budget(
            prefix_len,
            trunc_len,
            0,
            req.retracted_stain,
            new_mamba_state=self.budget.new_mamba_state(req),
            host_hit_len=req.host_hit_length,
            storage_hit_len=req.storage_hit_length,
        )

    def _req_inc_lock_ref(self, req: Req):
        result = self.tree_cache.inc_lock_ref(req.last_node)
        if self.is_hybrid_swa:
            req.swa_uuid_for_lock = result.swa_uuid_for_lock
        # match locks this node's components, so clear any stale skip set
        # carried from a previous scheduling of this req.
        req.skip_lock_node_ids = {}

    def add_dllm_staging_req(self, req: Req):
        assert self.dllm_config is not None
        _rem_tokens = self._get_dllm_remain_tokens()

        if _rem_tokens <= 0:
            return AddReqResult.NO_TOKEN

        # Truncate input length to available tokens and update request metadata
        cand_extend_input_len = len(req.full_untruncated_fill_ids) - len(
            req.prefix_indices
        )
        if req.dllm_incomplete_ids and cand_extend_input_len > _rem_tokens:
            return AddReqResult.NO_TOKEN
        truncated = cand_extend_input_len > _rem_tokens
        new_len = min(cand_extend_input_len, _rem_tokens)
        req.set_extend_range(len(req.prefix_indices), len(req.prefix_indices) + new_len)
        self.can_run_list.append(req)

        # Update budget: reserve max_new_tokens only if not truncated
        max_new_tokens = (
            min(req.sampling_params.max_new_tokens, CLIP_MAX_NEW_TOKENS)
            if not truncated
            else 0
        )
        self._update_prefill_budget(
            0,
            req.extend_range.length,
            max_new_tokens,
            req.retracted_stain,
            new_mamba_state=self.budget.new_mamba_state(req),
        )

        # Return based on remaining token availability
        return (
            AddReqResult.NO_TOKEN
            if self._get_dllm_remain_tokens() <= 0
            else AddReqResult.CONTINUE
        )

    def add_chunked_req(self, req: Req):
        if self.dllm_config is not None:
            _rem_tokens = self._get_dllm_remain_tokens()
        else:
            _rem_tokens = self.budget.chunk_admission_cap(self.rem_chunk_tokens)
            # The chunked_req must be added to the list; otherwise, it will cause a memory leak.
            # Therefore, in certain cases where _rem_tokens <= 0, it should be replaced with rem_chunk_tokens.
            if _rem_tokens <= 0:
                if self.is_hybrid_swa:
                    return req
                _rem_tokens = self.rem_chunk_tokens

        # A mid-chunk rank prefills this pass regardless of the delayer
        # verdict, so report prefillable=True and ignore the result.
        if self.prefill_delayer_single_pass is not None:
            self.prefill_delayer_single_pass.negotiate_should_allow_prefill(
                local_prefillable=True,
                running_batch=self.running_batch.batch_size(),
                max_prefill_bs=self.max_prefill_bs,
                max_running_requests=self.max_running_requests,
                waiting_queue_len=self.waiting_queue_len,
            )

        cand_extend_input_len = len(req.full_untruncated_fill_ids) - len(
            req.prefix_indices
        )
        truncated = cand_extend_input_len > _rem_tokens
        new_len = min(cand_extend_input_len, _rem_tokens)
        req.set_extend_range(len(req.prefix_indices), len(req.prefix_indices) + new_len)
        self.can_run_list.append(req)
        self._update_prefill_budget(
            0,
            req.extend_range.length,
            (
                min(req.sampling_params.max_new_tokens, CLIP_MAX_NEW_TOKENS)
                if not truncated
                else 0
            ),
            req.retracted_stain,
            new_mamba_state=self.budget.new_mamba_state(req),
        )

        # Return if chunked prefill not finished
        return req if truncated else None

    @contextmanager
    def _lock_node(self, last_node: TreeNode):
        dec_lock_params = None
        try:
            result = self.tree_cache.inc_lock_ref(last_node)
            if self.tree_cache.is_tree_cache():
                # init_load_back may revive SWA/Mamba tombstones while this
                # temporary admission lock is held. Release must mirror the
                # exact nodes skipped at acquire time.
                dec_lock_params = result.to_dec_params()
            yield None
        finally:
            if dec_lock_params is not None:
                self.tree_cache.dec_lock_ref(last_node, dec_lock_params)
            else:
                self.tree_cache.dec_lock_ref(last_node)

    def add_one_req_ignore_eos(self, req: Req):
        cand_extend_input_len = len(req.full_untruncated_fill_ids) - len(
            req.prefix_indices
        )
        cost = self.budget.prefill_cost(
            cand_extend_input_len=cand_extend_input_len,
            swa_extend_input_len=cand_extend_input_len,
            max_new_tokens=self._swa_new_tokens(req) if self.is_hybrid_swa else 0,
            new_mamba_state=self.budget.new_mamba_state(req),
        )
        if not self.budget.fits_immediate(cost).ok:
            return AddReqResult.NO_TOKEN

        def add_req_state(r, insert_sort=False):
            new_token_ratio = (
                1.0 if r.sampling_params.ignore_eos else self.new_token_ratio
            )
            tokens_left = r.sampling_params.max_new_tokens * new_token_ratio - len(
                r.output_ids
            )
            tokens_occupied = len(r.origin_input_ids) + len(r.output_ids)

            if tokens_left <= 0:
                return

            if not insert_sort:
                self.req_states.append((tokens_left, tokens_occupied))
            else:
                i = 0
                for i in range(len(self.req_states)):
                    if tokens_left <= self.req_states[i][0]:
                        break
                self.req_states.insert(i, (tokens_left, tokens_occupied))

        if self.req_states is None:
            self.req_states = []
            add_req_state(req)
            if self.running_batch is not None:
                for r in self.running_batch.reqs:
                    add_req_state(r)
            for r in self.can_run_list:
                add_req_state(r)
            self.req_states.sort(key=lambda x: x[0])
        else:
            add_req_state(req, insert_sort=True)

        if not self.is_hybrid_swa:
            # Skip this logic for swa. The SWA has different memory management, and
            # this mechanism is underestimating the memory usage.
            cur_rem_tokens = self.budget.solvency_cur_rem() - self.ceil_paged_tokens(
                cand_extend_input_len
            )
            tokens_freed = 0
            for i, (tokens_left, tokens_occupied) in enumerate(self.req_states):
                # tokens_left gives a reservative calculation as the last token is not stored
                bs = len(self.req_states) - i
                min_free_tokens = cur_rem_tokens + tokens_freed - tokens_left * bs
                # reserve tokens for corner cases
                if min_free_tokens <= IGNORE_EOS_RESERVE_TOKENS * bs:
                    return AddReqResult.NO_TOKEN
                tokens_freed += tokens_occupied

        if (self.prefill_delayer_single_pass is not None) and (
            not self.prefill_delayer_single_pass.negotiate_should_allow_prefill(
                local_prefillable=True,
                running_batch=self.running_batch.batch_size(),
                max_prefill_bs=self.max_prefill_bs,
                max_running_requests=self.max_running_requests,
                waiting_queue_len=self.waiting_queue_len,
            )
        ):
            return AddReqResult.OTHER

        if self.dllm_config is not None:
            if self.rem_dllm_tokens <= 0:
                return AddReqResult.OTHER

            if (
                tile_stop := self._check_prefill_tile_budget(cand_extend_input_len)
            ) is not None:
                return tile_stop

            self._add_dllm_req(req, 0)
        elif (
            self.rem_chunk_tokens is None  # chunked prefill is disabled
            or cand_extend_input_len <= self.rem_chunk_tokens  # it is the last chunk
        ):
            if (
                tile_stop := self._check_prefill_tile_budget(cand_extend_input_len)
            ) is not None:
                return tile_stop

            # Non-chunked prefill — the whole sequence is committed this iter.
            req.set_extend_range(
                len(req.prefix_indices), len(req.full_untruncated_fill_ids)
            )
            self.can_run_list.append(req)
            self._update_prefill_budget(
                0,
                req.extend_range.length,
                min(req.sampling_params.max_new_tokens, CLIP_MAX_NEW_TOKENS),
                req.retracted_stain,
                new_mamba_state=self.budget.new_mamba_state(req),
            )
        else:
            if self.rem_chunk_tokens <= 0:
                return AddReqResult.OTHER

            # Chunked prefill
            trunc_len = self.rem_chunk_tokens

            if (tile_stop := self._check_prefill_tile_budget(trunc_len)) is not None:
                return tile_stop

            assert len(req.prefix_indices) == 0
            req.set_extend_range(
                len(req.prefix_indices), len(req.prefix_indices) + trunc_len
            )
            self.can_run_list.append(req)
            self.new_chunked_req = req
            self._update_prefill_budget(
                0,
                trunc_len,
                0,
                req.retracted_stain,
                new_mamba_state=self.budget.new_mamba_state(req),
            )

        return self.budget_state()

    def add_one_req(
        self, req: Req, has_chunked_req: bool, truncation_align_size: Optional[int]
    ):
        # TODO support cp with multiple requests
        # Enabling context parallelism currently presents precision issues;
        # therefore, the prefill-batch setting is temporarily set to 1.
        if (self.dsa_prefill_cp_in_seq_split) and len(self.can_run_list) >= 1:
            return AddReqResult.OTHER

        if (x := self.prefill_max_requests) is not None and len(self.can_run_list) >= x:
            return AddReqResult.OTHER

        if req.sampling_params.ignore_eos and getattr(self.tree_cache, "disable", True):
            return self.add_one_req_ignore_eos(req)

        # Reserve page_size for page-alignment overhead: the paged allocator may
        # consume one extra page per request (see alloc_extend), which
        # _update_prefill_budget also deducts.
        max_new = min(
            max(req.sampling_params.max_new_tokens - len(req.output_ids), 0),
            CLIP_MAX_NEW_TOKENS,
        )
        cand_extend_input_len = len(req.full_untruncated_fill_ids) - len(
            req.prefix_indices
        )
        # adjusting the input_tokens based on host_hit_length and page_size
        # (host-hit prefix is loaded back, not re-prefilled, so the SWA peak is
        # driven only by the freshly-prefilled tail; the loaded window is
        # charged separately via swa_host_hit_length).
        real_input_tokens = cand_extend_input_len - req.host_hit_length
        real_input_tokens = self.ceil_paged_tokens(real_input_tokens)
        prefix_len = len(req.prefix_indices)

        cost = self.budget.prefill_cost(
            cand_extend_input_len=cand_extend_input_len,
            swa_extend_input_len=real_input_tokens,
            max_new_tokens=max_new,
            swa_host_hit_length=(req.swa_host_hit_length if self.is_hybrid_swa else 0),
            new_mamba_state=self.budget.new_mamba_state(req),
        )
        verdict = self.budget.fits(cost)
        chunk_tokens_limit = self.rem_chunk_tokens
        if not verdict.ok:
            if not verdict.swa_binding:
                return AddReqResult.NO_TOKEN
            # SWA charge is the binding constraint: the chunk-shrink escape
            # hatch applies only to a request that can NEVER fit whole.
            if not self._swa_req_never_fits(
                real_input_tokens,
                self._swa_new_tokens(req),
                req.swa_host_hit_length,
            ):
                return AddReqResult.NO_TOKEN
            swa_cap = self.budget.swa_chunk_cap(
                self._swa_new_tokens(req), req.swa_host_hit_length
            )
            if self.rem_chunk_tokens is None or swa_cap <= 0:
                return AddReqResult.NO_TOKEN
            chunk_tokens_limit = min(self.rem_chunk_tokens, swa_cap)

        if (
            self.rem_chunk_tokens is None
            and len(self.can_run_list) != 0
            and real_input_tokens >= self.rem_input_tokens
        ):
            # If without chunked prefill:
            # - if the can_run_list is not empty, we satisfy the constraint of (max_prefill_tokens)
            # - if the can_run_list is empty, always accept the first prefill request
            return AddReqResult.OTHER

        with self._lock_node(req.last_node):
            # The budget's availability may decrease after the lock acquisition
            verdict = self.budget.fits(cost)
            if not verdict.ok:
                if not verdict.swa_binding:
                    return AddReqResult.NO_TOKEN
                if not self._swa_req_never_fits(
                    real_input_tokens,
                    self._swa_new_tokens(req),
                    req.swa_host_hit_length,
                ):
                    return AddReqResult.NO_TOKEN
                swa_cap = self.budget.swa_chunk_cap(
                    self._swa_new_tokens(req), req.swa_host_hit_length
                )
                if self.rem_chunk_tokens is None or swa_cap <= 0:
                    return AddReqResult.NO_TOKEN
                chunk_tokens_limit = min(self.rem_chunk_tokens, swa_cap)

            # Negotiate only after every KV-budget gate (a NO_TOKEN rank must
            # report not-prefillable via finalize()) and before init_load_back
            # (a delay verdict must not start KV load-back).
            if (self.prefill_delayer_single_pass is not None) and (
                not self.prefill_delayer_single_pass.negotiate_should_allow_prefill(
                    local_prefillable=True,
                    running_batch=self.running_batch.batch_size(),
                    max_prefill_bs=self.max_prefill_bs,
                    max_running_requests=self.max_running_requests,
                    waiting_queue_len=self.waiting_queue_len,
                )
            ):
                return AddReqResult.OTHER

            if req.needs_host_load_back():
                new_indices, req.last_node = self.tree_cache.init_load_back(
                    InitLoadBackParams(
                        best_match_node=req.best_match_node,
                        host_hit_length=req.host_hit_length,
                        req=req,
                    )
                )
                req.prefix_indices = torch.cat([req.prefix_indices, new_indices])
                prefix_len = len(req.prefix_indices)
                req.cache_protected_len = prefix_len

            input_tokens = self.ceil_paged_tokens(
                len(req.full_untruncated_fill_ids) - len(req.prefix_indices)
            )

            if (
                self.rem_chunk_tokens is None
                and len(self.can_run_list) != 0
                and input_tokens >= self.rem_input_tokens
            ):
                # If without chunked prefill:
                # - if the can_run_list is not empty, we satisfy the constraint of (max_prefill_tokens)
                # - if the can_run_list is empty, always accept the first prefill request
                return AddReqResult.OTHER

            if self.dllm_config is not None:
                if self.rem_dllm_tokens <= 0:
                    return AddReqResult.OTHER

                assert (
                    truncation_align_size is None
                ), "truncation_align_size is not supported for dllm prefill"

                if (
                    tile_stop := self._check_prefill_tile_budget(input_tokens)
                ) is not None:
                    return tile_stop

                self._add_dllm_req(req, prefix_len)
                self._req_inc_lock_ref(req)
            elif chunk_tokens_limit is None or input_tokens <= chunk_tokens_limit:
                if (
                    tile_stop := self._check_prefill_tile_budget(input_tokens)
                ) is not None:
                    return tile_stop

                # Non-chunked prefill — the whole sequence is committed this iter.
                req.set_extend_range(
                    len(req.prefix_indices), len(req.full_untruncated_fill_ids)
                )
                self.can_run_list.append(req)

                self._req_inc_lock_ref(req)
                self._update_prefill_budget(
                    prefix_len,
                    input_tokens,
                    min(
                        req.sampling_params.max_new_tokens,
                        CLIP_MAX_NEW_TOKENS,
                    ),
                    req.retracted_stain,
                    new_mamba_state=self.budget.new_mamba_state(req),
                    host_hit_len=req.host_hit_length,
                    storage_hit_len=req.storage_hit_length,
                )
            else:
                # Make sure at least one page is available
                trunc_len = chunk_tokens_limit // self.page_size * self.page_size

                if trunc_len <= 0:
                    return AddReqResult.OTHER

                # When truncation align size is set, we want to assert that the prefill prefix length is multiple of truncation align size
                # A typical use case is when deterministic inference is enabled with flashinfer attention backend,
                # we need the prefill prefix length to be multiple of attention split size
                if truncation_align_size is not None:
                    if trunc_len < truncation_align_size:
                        return AddReqResult.OTHER
                    else:
                        trunc_len = truncation_align_size * (
                            trunc_len // truncation_align_size
                        )

                now_input_len = trunc_len + len(req.prefix_indices)
                now_input_len = now_input_len // self.page_size * self.page_size
                trunc_len = now_input_len - len(req.prefix_indices)

                if trunc_len <= 0:
                    return AddReqResult.OTHER

                if (
                    tile_stop := self._check_prefill_tile_budget(trunc_len)
                ) is not None:
                    return tile_stop

                # Chunked prefill
                req.set_extend_range(
                    len(req.prefix_indices), len(req.prefix_indices) + trunc_len
                )

                self.can_run_list.append(req)
                self.new_chunked_req = req

                self._req_inc_lock_ref(req)
                self._update_prefill_budget(
                    prefix_len,
                    trunc_len,
                    0,
                    req.retracted_stain,
                    new_mamba_state=self.budget.new_mamba_state(req),
                    host_hit_len=req.host_hit_length,
                    storage_hit_len=req.storage_hit_length,
                )

        return self.budget_state()

    def preempt_to_schedule(self, req: Req, server_args: ServerArgs) -> bool:
        """
        Preempt running requests to serve the new request if the priority threshold is met and token count sum is verified.
        Returns True if preemption was committed, and the new request can be scheduled.
        """
        # Iterate running requests to find preemptible requests
        priority_sign = 1 if server_args.schedule_low_priority_values_first else -1

        # NOTE: A request finishes in two phases:
        #   1) update_finish_state + release_kv_cache  (in process_batch_result)
        #   2) filter out of batch                (in get_next_batch_to_run / update_running_batch)
        # Preemption runs between these two phases (inside get_new_batch_prefill),
        # so running_batch may still contain requests whose KV cache is already freed.
        # We must skip them here to avoid a double-free on release_req.
        valid_running_reqs = (
            r
            for r in self.running_batch.reqs
            if r not in self.preempt_list and not r.finished()
        )

        sorted_valid_running_reqs = sorted(
            valid_running_reqs,
            key=lambda x: (
                x.priority * (-priority_sign),
                -x.time_stats.wait_queue_entry_time,
            ),
        )

        preemptible_reqs = []
        min_tokens_to_remove = self.budget.preemption_demand(
            req,
            len(req.full_untruncated_fill_ids) - len(req.prefix_indices),
        )
        for running_req in sorted_valid_running_reqs:
            # Priority difference needs to meet the threshold to be preemptible.
            priority_diff = (req.priority - running_req.priority) * (-priority_sign)

            if priority_diff > self.priority_scheduling_preemption_threshold:
                preemptible_reqs.append(running_req)
                min_tokens_to_remove -= self.budget.running_request_credit(running_req)
                if min_tokens_to_remove <= 0:
                    break
            else:
                break

        # Check max token count limit can be met
        if len(preemptible_reqs) == 0 or min_tokens_to_remove > 0:
            return False

        # Preempt running requests. Release allocated resources for immediate usage.
        preemptible_reqs = set(preemptible_reqs)
        keep_indices = []
        release_counter = 0
        for i, running_req in enumerate(self.running_batch.reqs):
            if running_req in preemptible_reqs:
                self.budget.uncharge_running(running_req)
                release_counter += 1
                self.running_batch.release_req(
                    i, len(self.running_batch.reqs) - release_counter, server_args
                )
            else:
                keep_indices.append(i)
        self.running_batch.filter_batch(keep_indices=keep_indices)
        self.preempt_list.extend(preemptible_reqs)
        return True
