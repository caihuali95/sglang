from __future__ import annotations

import dataclasses
import logging
import time
from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    NamedTuple,
    Optional,
    Protocol,
    Tuple,
    runtime_checkable,
)

import torch

from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.observability.metrics_collector import RadixCacheMetricsCollector

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.radix_cache import RadixKey


@runtime_checkable
class PrefixCacheTrait(Protocol):
    req_to_token_pool: ReqToTokenPool
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator
    page_size: int
    disable: bool


@dataclasses.dataclass
class MatchPrefixParams:
    """Unified parameters for match_prefix across different cache types"""

    key: RadixKey

    # Mamba specific
    cow_mamba: bool = False
    req: Optional[Req] = None


@dataclasses.dataclass
class InsertParams:
    """Unified parameters for insert across different cache types"""

    key: RadixKey
    value: Optional[torch.Tensor] = None

    # Mamba specific
    mamba_value: Optional[torch.Tensor] = None

    # SWA specific
    prev_prefix_len: int = 0
    swa_evicted_seqlen: int = 0

    # General
    chunked: bool = False
    priority: int = 0


@dataclasses.dataclass
class InsertResult:
    """Result of an insert operation"""

    prefix_len: int
    mamba_exist: bool = False


@dataclasses.dataclass
class EvictParams:
    """Unified parameters for evict across different cache types"""

    num_tokens: int
    swa_num_tokens: int = 0
    mamba_num: int = 0


@dataclasses.dataclass
class EvictResult:
    """Result of an evict operation"""

    num_tokens_evicted: int = 0
    swa_num_tokens_evicted: int = 0
    mamba_num_evicted: int = 0


@dataclasses.dataclass
class IncLockRefResult:
    """Result of an inc_lock_ref operation."""

    delta: Optional[int] = None
    swa_uuid_for_lock: Optional[int] = None


@dataclasses.dataclass
class DecLockRefParams:
    """Parameters for dec_lock_ref operation."""

    swa_uuid_for_lock: Optional[int] = None


@dataclasses.dataclass
class DecLockRefResult:
    """Result of an dec_lock_ref operation."""

    delta: Optional[int] = None


@dataclasses.dataclass
class InitLoadBackParams:
    """Unified parameters for init_load_back across different cache types"""

    last_host_node: Any
    host_hit_length: int
    mem_quota: Optional[int] = None
    req: Optional[Req] = None


class MatchResult(NamedTuple):
    """Result of a prefix match operation.

    Attributes:
        device_indices  :   Indices of the KV cache on the device matched by common prefix.
        last_device_node:   The last TreeNode on the device that was matched.
        last_host_node  :   The last TreeNode on the host that was matched.
                            Note that if HiCache is not enabled,
                            this **must** be the same as `last_device_node`.
        host_hit_length :   Length of the host cache hit. For pure-KV caches this is the
                            number of evicted KV tokens on CPU. For hybrid Mamba models this
                            is max(kv_host_tokens, 1-if-mamba-on-host) so that a mamba-only
                            host hit still triggers load-back without adding a separate field.
                            0 if HiCache is not enabled.
        mamba_branching_seqlen: The mamba radix cache branching point, which is the longest
                                page-aligned position that could've been cache hit if there
                                exists a mamba state.
    """

    device_indices: torch.Tensor
    last_device_node: Any
    last_host_node: Any
    host_hit_length: int = 0
    mamba_branching_seqlen: Optional[int] = None
    cache_protected_len: Optional[int] = None


class BasePrefixCache(ABC, PrefixCacheTrait):
    """Cache can be indexed by either rid or key."""

    metrics_collector: Optional[RadixCacheMetricsCollector] = (
        None  # metrics collector for the cache
    )

    def init_metrics_collector(self):
        from sglang.srt.server_args import get_global_server_args

        server_args = get_global_server_args()
        labels = {"cache_type": self.__class__.__name__}
        if server_args.extra_metric_labels:
            labels.update(server_args.extra_metric_labels)
        self.metrics_collector = RadixCacheMetricsCollector(labels=labels)

    def update_eviction_metrics(self, num_evicted: int, start_time: float):
        if self.metrics_collector is not None and num_evicted > 0:
            self.metrics_collector.observe_eviction_duration(
                time.perf_counter() - start_time
            )
            self.metrics_collector.increment_eviction_num_tokens(num_evicted)

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        pass

    @abstractmethod
    def cache_finished_req(self, req: Req, is_insert: bool = True, **kwargs):
        pass

    @abstractmethod
    def cache_unfinished_req(self, req: Req, **kwargs):
        pass

    @abstractmethod
    def evict(self, params: EvictParams) -> EvictResult:
        pass

    @abstractmethod
    def inc_lock_ref(self, node: Any) -> IncLockRefResult:
        pass

    @abstractmethod
    def dec_lock_ref(
        self, node: Any, params: Optional[DecLockRefParams] = None
    ) -> DecLockRefResult:
        pass

    def evictable_size(self):
        return 0

    # -- SlotBacktrack helpers (no-op unless shared memory pool is on) --

    def _slot_backtrack_binder(self):
        """Resolve the SlotBacktrackBinder for the full-MHA sub-pool, or return
        the process-wide null binder. Called on every `node.value` update."""
        from sglang.srt.mem_cache.relocation_log import null_backtrack_binder

        allocator = getattr(self, "token_to_kv_pool_allocator", None)
        if allocator is None:
            return null_backtrack_binder()
        get_kv = getattr(allocator, "get_kvcache", None)
        pool = get_kv() if get_kv is not None else None
        if pool is None:
            return null_backtrack_binder()
        # For composite SWA pool, the radix tree's slot ids are FULL-pool ids.
        full_pool = getattr(pool, "full_kv_pool", None)
        if full_pool is not None:
            pool = full_pool
        return getattr(pool, "slot_backtrack_binder", null_backtrack_binder())

    def _is_tree_node_alive(self, node) -> bool:
        """Return True if `node` is still reachable as a child of its parent
        in the radix tree. Used by the dedup-free path in `_insert_helper`
        to distinguish a real `binder.tree_node` entry (the bound tree node
        is alive — skip free, the tree owns the slot) from a stale entry
        (the bound node was detached via `_delete_leaf` /
        `_delete_tombstone_leaf` but is still Python-alive because the
        binder's strong reference kept it from being GC'd — clean up and
        allow the free).

        Cheap (one dict lookup), used per-slot in the dedup-free path.
        """
        root = getattr(self, "root_node", None)
        get_child_key_fn = getattr(self, "get_child_key_fn", None)
        if root is None or get_child_key_fn is None:
            return False
        if node is root:
            return True
        parent = getattr(node, "parent", None)
        if parent is None:
            return False
        try:
            key = get_child_key_fn(node.key)
        except Exception:
            return False
        children = getattr(parent, "children", None)
        if children is None:
            return False
        return children.get(key) is node

    def _free_dedup_slots(self, value_segment, tree_value_segment) -> None:
        """Used by `_insert_helper`'s dedup-free path. Free slots from
        `value_segment` that are SAFE to free, i.e.:
          (a) differ from `tree_value_segment` at the matching position
              (so we don't free a slot that's just a cache-hit reuse of
              the walker's currently-matching tree node), AND
          (b) are not currently bound to any LIVE tree node's value tensor
              (so we don't corrupt another tree node that legitimately
              owns the slot — this is the bug eval_results_17 surfaced
              via the diagnostic warning at multi_ended_allocator.py:1260
              + bind_tree_node OVERWRITE cum=114K).

        Stale binder entries (binding to a detached tree node that's
        still Python-alive only because the binder retains a strong ref)
        are cleaned up and the slot is then freed normally — that avoids
        the eval_13 leak regression where the safer-than-this `safe_free`
        helper skipped legitimate frees because it couldn't distinguish
        real from stale bindings.
        """
        if not isinstance(value_segment, torch.Tensor) or value_segment.numel() == 0:
            return
        diff_mask = value_segment != tree_value_segment
        if not bool(diff_mask.any()):
            return
        candidates = value_segment[diff_mask]
        binder = self._slot_backtrack_binder()
        if binder.is_null():
            # Non-shared-pool path: no binder, just free.
            self.token_to_kv_pool_allocator.free(candidates)
            return
        cand_cpu = candidates.detach().cpu().tolist()
        backtrack = getattr(binder, "_backtrack", None)
        is_slot_allocated = getattr(
            self.token_to_kv_pool_allocator, "is_slot_allocated", None
        )
        safe: list = []
        for s in cand_cpu:
            # Guard: never hand a slot that is already outside the
            # allocator's allocated range to `free()` — that trips the
            # stale-slot assertion in `MultiEndedAllocator._free_eager`
            # and aborts the request. Such a slot got here because the
            # `value` tensor segment we diffed still carried a pre-free
            # id; the slot's bytes were already reclaimed (or never ours).
            # Skip it, but log (capped) so a regression can be traced.
            if is_slot_allocated is not None and not is_slot_allocated(s):
                cap = 32
                if not hasattr(self, "_free_dedup_stale_warn_count"):
                    self._free_dedup_stale_warn_count = 0
                if self._free_dedup_stale_warn_count < cap:
                    self._free_dedup_stale_warn_count += 1
                    tn = (
                        backtrack.tree_node.get(s)
                        if backtrack is not None
                        else None
                    )
                    state = ""
                    state_fn = getattr(
                        self.token_to_kv_pool_allocator,
                        "allocator_state_str",
                        None,
                    )
                    if state_fn is not None:
                        try:
                            state = state_fn()
                        except Exception:
                            state = ""
                    import inspect as _inspect

                    frames = _inspect.stack()[1:9]
                    callers = " <- ".join(
                        f"{f.filename.split('/')[-1]}:{f.lineno}"
                        for f in frames
                    )
                    logger.warning(
                        "_free_dedup_slots: slot %d is OUTSIDE the "
                        "allocator's allocated range — skipping its free "
                        "(would crash _free_eager's stale-slot assertion). "
                        "binder tree_node entry: %s. allocator state: %s. "
                        "Caller: %s. cum_stale=%d/%d.",
                        s,
                        (
                            f"node_id=0x{id(tn.node):x}, attr={tn.attr!r}, "
                            f"pos={tn.position}"
                        )
                        if tn is not None
                        else "(none)",
                        state or "(unavailable)",
                        callers,
                        self._free_dedup_stale_warn_count,
                        cap,
                    )
                # Intentionally do NOT touch the binder here: if a live
                # node's `value` cell holds this out-of-range id, popping the
                # binder entry wouldn't fix the cell. Leave it for the
                # `_safe_unbind_tree_node` diagnostics to surface later.
                continue
            tn = backtrack.tree_node.get(s) if backtrack is not None else None
            if tn is None:
                safe.append(s)
                continue
            if self._is_tree_node_alive(tn.node):
                # Real binding: a live tree node legitimately owns this
                # slot. Don't free; the tree will release it on eviction.
                continue
            # Stale binding: the bound tree node was detached but the
            # binder kept it alive. Clean up and allow the free.
            backtrack.tree_node.pop(s, None)
            safe.append(s)
        if safe:
            self.token_to_kv_pool_allocator.free(
                torch.tensor(safe, dtype=candidates.dtype, device=candidates.device)
            )

    def _set_node_value(self, node, new_value) -> None:
        """Route a `node.value = new_value` assignment through the
        SlotBacktrackBinder so that relocations can target this node directly.

        For eviction (`new_value is None` or an empty list), call
        :py:meth:`_unbind_all_node_value` instead.

        Match-aware: routes the OLD-value unbinds through
        :py:meth:`_safe_unbind_tree_node` and runs an overlap pre-flight
        on `new_value` (warns if any incoming slot is already bound to a
        different node — see :py:meth:`_check_value_overlap`)."""
        binder = self._slot_backtrack_binder()
        if not binder.is_null():
            # Pre-flight: are any of `new_value`'s slots already bound to
            # a DIFFERENT tree node? If so, the upstream allocator handed
            # the same slot to two reqs, OR `new_value`'s source tensor
            # was stale. Logs a capped warning with the offending slots
            # and the caller stack so the upstream double-allocation can
            # be localized (eval_results_41 OVERWRITE diagnosis).
            self._check_value_overlap(
                binder, new_value, node, attr="value", op="_set_node_value"
            )
            old = getattr(node, "value", None)
            if isinstance(old, torch.Tensor) and old.numel() > 0:
                for pos, s in enumerate(old.detach().cpu().tolist()):
                    self._safe_unbind_tree_node(
                        binder=binder,
                        slot=int(s),
                        node=node,
                        position=int(pos),
                        attr="value",
                        op="_set_node_value",
                    )
        node.value = new_value
        if not binder.is_null():
            if isinstance(new_value, torch.Tensor) and new_value.numel() > 0:
                for pos, s in enumerate(new_value.detach().cpu().tolist()):
                    binder.bind_tree_node(s, node, pos)

    def _unbind_all_node_value(self, node):
        """Drop backtrack entries for every slot in `node.value`,
        preserving the current value. Used before eviction
        (`node.value = None`).

        Match-aware: only pops the binder's entry at `slot` when it
        actually points at this `node` (eval_results_40 root cause —
        unconditional pop silently corrupted other nodes' bindings when
        called twice across an intervening apply-rebind). See
        :py:meth:`_safe_unbind_tree_node`.

        Returns the LIST of slot ids actually popped from the binder
        (= the slots `node` legitimately owned, per the binder).
        Callers should pass ONLY these slots to
        `token_to_kv_pool_allocator.free` — passing the full
        `node.value` tensor would free slots that may be owned by
        OTHER tree nodes, advancing the allocator's watermark past
        slots still bound in the binder (the eval_results_42
        multi-col cascade).

        When the binder is null (non-shared pool), returns the entire
        `node.value` slot list — no divergence is possible without a
        binder.
        """
        binder = self._slot_backtrack_binder()
        old = getattr(node, "value", None)
        if not isinstance(old, torch.Tensor) or old.numel() == 0:
            return []
        slot_ids = old.detach().cpu().tolist()
        if binder.is_null():
            # Non-shared path: no binder, free everything.
            return [int(s) for s in slot_ids]
        unbound: list = []
        for pos, s in enumerate(slot_ids):
            popped = self._safe_unbind_tree_node(
                binder=binder,
                slot=int(s),
                node=node,
                position=int(pos),
                attr="value",
                op="_unbind_all_node_value",
            )
            if popped:
                unbound.extend(popped)
        return unbound

    # --- match-aware unbind + free helpers (eval_results_42 fix) ---------
    #
    # These wrap the common "unbind then free" pattern used by all
    # eviction code paths. The fix: only free slots that `node` actually
    # OWNS in the binder (= returned by `_unbind_*_value`). Previously
    # callers passed the full `node.value` tensor to the allocator's
    # free — which freed slots that may be bound to OTHER tree nodes
    # (when `node.value` was stale), advancing the allocator's watermark
    # past those slots and producing the eval_results_42 multi-col
    # cascade once the allocator re-allocated them.

    def _unbind_and_free_node_value(self, node) -> int:
        """Match-aware unbind + free of `node.value` (full-pool slots).
        Returns the number of slots actually freed (= the slots the
        binder agreed `node` owned).

        Slots in `node.value` that the binder didn't agree on (e.g.,
        bound to a different node — "wrong_node_at_slot") are NOT
        freed; the rightful owner will free them on its own eviction.
        See :py:meth:`_safe_unbind_tree_node`."""
        unbound = self._unbind_all_node_value(node)
        if unbound and isinstance(node.value, torch.Tensor):
            self.token_to_kv_pool_allocator.free(
                torch.tensor(
                    unbound,
                    dtype=node.value.dtype,
                    device=node.value.device,
                )
            )
        return len(unbound)

    def _unbind_and_free_node_mamba_value(self, node) -> int:
        """Match-aware unbind + free of `node.mamba_value` (mamba-pool
        slots). Mirrors :py:meth:`_unbind_and_free_node_value` for the
        mamba pool. Returns the number of slots actually freed.

        Used by Mamba-aware caches only — for non-mamba caches this is
        a 1-element no-op since `node.mamba_value` is empty."""
        unbound = self._unbind_node_mamba_value(node)
        if unbound and isinstance(node.mamba_value, torch.Tensor):
            rtp = getattr(self, "req_to_token_pool", None)
            mamba_pool = getattr(rtp, "mamba_pool", None) if rtp else None
            if mamba_pool is not None:
                mamba_pool.free(
                    torch.tensor(
                        unbound,
                        dtype=node.mamba_value.dtype,
                        device=node.mamba_value.device,
                    )
                )
        return len(unbound)

    def _bind_node_value(self, node) -> None:
        """Bind every slot in `node.value` to (node, position). Assumes the
        backtrack has no existing entries for these slots (e.g., post-split).

        Includes an overlap pre-flight (see :py:meth:`_check_value_overlap`)
        — under a correct caller this assumption holds and the check is
        silent; if the binder *does* have a different node at any of
        these slots, the overlap warning fires AND
        :py:meth:`SlotBacktrack.bind_tree_node`'s OVERWRITE-abort path
        will then preserve the prior entry."""
        binder = self._slot_backtrack_binder()
        if binder.is_null():
            return
        val = getattr(node, "value", None)
        if isinstance(val, torch.Tensor) and val.numel() > 0:
            self._check_value_overlap(
                binder, val, node, attr="value", op="_bind_node_value"
            )
            for pos, s in enumerate(val.detach().cpu().tolist()):
                binder.bind_tree_node(s, node, pos)

    # -- Mamba-side TreeNode.mamba_value bindings.
    # `MambaRadixCache.TreeNode.mamba_value` is a 1-element tensor cloned from
    # `req.mamba_pool_idx` at cache-insert time. Without binding, eager
    # compaction relocations leave the clone holding a stale slot id; later
    # `evict_mamba` then frees a slot outside the watermark and the
    # MultiEndedAllocator assertion fires (observed in eval runs on
    # 2026-05-03). Hooking the assignment here keeps the clone in sync via
    # the relocation log's flush, identical to how `_set_node_value` keeps
    # `node.value` in sync for full/SWA slots.

    def _mamba_slot_backtrack_binder(self):
        """Resolve the SlotBacktrackBinder for the MAMBA sub-pool, or return
        the process-wide null binder. The mamba binder lives on the
        SharedMambaPool inner pool of the SharedHybridReqToTokenPool."""
        from sglang.srt.mem_cache.relocation_log import null_backtrack_binder

        rtp = getattr(self, "req_to_token_pool", None)
        mamba_pool = getattr(rtp, "mamba_pool", None) if rtp else None
        return getattr(
            mamba_pool, "slot_backtrack_binder", null_backtrack_binder()
        )

    def _set_node_mamba_value(self, node, new_value) -> None:
        """Route a `node.mamba_value = new_value` assignment through the
        MAMBA SlotBacktrackBinder so relocations of those slot ids reach
        this tree node's clone tensor.

        For tombstone (`new_value is None`), the bind step is skipped — call
        :py:meth:`_unbind_node_mamba_value` first if you need explicit unbind
        of the prior tensor.

        Match-aware: routes OLD-value unbinds through
        :py:meth:`_safe_unbind_tree_node` and runs an overlap pre-flight
        on `new_value` (see :py:meth:`_check_value_overlap`)."""
        binder = self._mamba_slot_backtrack_binder()
        if not binder.is_null():
            self._check_value_overlap(
                binder, new_value, node,
                attr="mamba_value", op="_set_node_mamba_value",
            )
            old = getattr(node, "mamba_value", None)
            if isinstance(old, torch.Tensor) and old.numel() > 0:
                for pos, s in enumerate(old.detach().cpu().tolist()):
                    self._safe_unbind_tree_node(
                        binder=binder,
                        slot=int(s),
                        node=node,
                        position=int(pos),
                        attr="mamba_value",
                        op="_set_node_mamba_value",
                    )
        node.mamba_value = new_value
        if not binder.is_null():
            if isinstance(new_value, torch.Tensor) and new_value.numel() > 0:
                for pos, s in enumerate(new_value.detach().cpu().tolist()):
                    binder.bind_tree_node(s, node, pos, attr="mamba_value")

    def _unbind_node_mamba_value(self, node):
        """Drop backtrack entries for every slot in `node.mamba_value`,
        preserving the current value. Used before freeing the slot via the
        allocator (so the binder isn't left referencing a freed slot id)
        and before tombstoning.

        Match-aware (eval_results_40 root-cause fix): only pops the
        binder's entry at `slot` when it actually points at this `node`.
        Previously this method unconditionally popped `tree_node[slot]`
        regardless of whose binding lived there — which silently
        corrupted other nodes' bindings when called twice across an
        intervening eager-compaction `apply` (the redundant unbind in
        `_tombstone_internal_node` after `mamba_pool.free` did exactly
        that). See :py:meth:`_safe_unbind_tree_node`.

        Returns the LIST of slot ids actually popped from the binder
        (eval_results_42 fix). Callers should pass ONLY these slots to
        the allocator's `free` to keep allocator/binder consistent.
        Returns the full `node.mamba_value` list when the binder is
        null (non-shared path).
        """
        binder = self._mamba_slot_backtrack_binder()
        old = getattr(node, "mamba_value", None)
        if not isinstance(old, torch.Tensor) or old.numel() == 0:
            return []
        slot_ids = old.detach().cpu().tolist()
        if binder.is_null():
            return [int(s) for s in slot_ids]
        unbound: list = []
        for pos, s in enumerate(slot_ids):
            popped = self._safe_unbind_tree_node(
                binder=binder,
                slot=int(s),
                node=node,
                position=int(pos),
                attr="mamba_value",
                op="_unbind_node_mamba_value",
            )
            if popped:
                unbound.extend(popped)
        return unbound

    # Cap on divergence warnings to bound the O(|tree_node|) scan cost
    # (see `_safe_unbind_tree_node`). The diagnostic counter is shared
    # across all base-prefix-cache subclasses via type(self)/getattr().
    _UNBIND_DIVERGENCE_WARN_CAP: int = 32

    def _safe_unbind_tree_node(
        self,
        binder,
        *,
        slot: int,
        node,
        position: int,
        attr: str,
        op: str,
    ):
        """Match-aware tree_node unbind. Returns the list of slot ids
        whose binder entries were actually popped (= slots that `node`
        legitimately owned, per the binder).

        Replaces a naive ``binder.unbind_tree_node(slot)`` (which
        unconditionally pops whatever entry is at `slot`) with one that:

          * pops the binder's entry at `slot` ONLY if it actually points
            at `(node, position, attr)` — the "ok" case, by far the most
            common. Returns ``[slot]``.
          * if the binder has `node` bound at a DIFFERENT slot (or slots)
            matching `(position, attr)` — the "divergent" case — pops
            those instead, then logs a capped diagnostic warning. This
            catches the case where `node.<attr>` got out of sync with
            the binder's view (e.g., a stale tensor read). Returns
            ``payload`` (the actually-bound slots).
          * if the slot's entry belongs to a DIFFERENT node — the
            "wrong_node_at_slot" case — leaves the binder alone
            (skipping the pop) and logs a capped warning. This is the
            eval_results_40 root-cause: a redundant unbind call with
            a stale `node.<attr>` value would otherwise silently steal
            the other node's binding and propagate the corruption.
            Returns ``[]``.
          * if there's no entry at all — the "unbound" case — no-op,
            no warning (legitimate; e.g., the node was already unbound
            earlier). Returns ``[]``.

        The return value is used by callers (`_evict_leaf_node`,
        `_iteratively_delete_tombstone_leaf`) to call
        `token_to_kv_pool_allocator.free` with ONLY the slots that
        `node` actually owned — not the full `node.value` tensor. This
        is the eval_results_42 fix: preventing the allocator's
        watermark from advancing past slots that are still bound to
        OTHER tree nodes (which previously surfaced as
        `bind_req_position multi-col` after the allocator double-handed
        a slot to a new req).

        Cap-gated diagnostic writes to bound the O(|tree_node|) scan
        cost in degraded states; the underlying pop / skip behavior
        always runs, even after the cap.
        """
        if binder.is_null():
            # Non-shared path: no binder, so caller should free
            # everything in node.value (no divergence possible).
            return [slot]
        status, payload = binder.diagnose_tree_node_for_unbind(
            slot, node, position, attr
        )

        # Action — happens always, regardless of cap.
        unbound: list = []
        if status == "ok":
            binder.unbind_tree_node(slot)
            return [slot]
        if status == "unbound":
            # Legitimate no-op — no warning. Don't free (we don't own
            # this slot; either it was already freed, or never bound).
            return []
        if status == "divergent":
            # Pop the actually-bound slot(s). Must do this even after
            # the warning cap is reached, otherwise node's binder entry
            # leaks. These are the slots node actually owns.
            for actual_slot in payload:
                binder.unbind_tree_node(int(actual_slot))
                unbound.append(int(actual_slot))
        elif status == "wrong_node_at_slot":
            # Leave the binder alone — this is the eval_results_40
            # corruption-prevention. Don't pop another node's binding.
            # Return [] — the caller must not free this slot either
            # (eval_results_42: freeing slots we don't own pushes the
            # allocator's watermark past slots still bound to other
            # nodes, leading to multi-col when re-allocated).
            pass

        # Diagnostic warning (cap-gated).
        cls = type(self)
        cap = getattr(cls, "_UNBIND_DIVERGENCE_WARN_CAP", 32)
        if not hasattr(cls, "_unbind_divergence_count"):
            cls._unbind_divergence_count = 0
        cls._unbind_divergence_count += 1
        if cls._unbind_divergence_count > cap:
            return unbound  # past cap, suppress the warning text

        # Capture the unbind caller stack.
        callers = "unknown"
        try:
            import inspect
            frames = inspect.stack()[2:9]
            callers = " <- ".join(
                f"{f.filename.split('/')[-1]}:{f.lineno}" for f in frames
            )
        except Exception:
            pass

        if status == "divergent":
            divergent_slots = payload
            logger.warning(
                "%s DIVERGENCE for node 0x%x: "
                "node.%s[%d]=%d (caller expected this slot), "
                "but binder has node bound at slot(s) %s. "
                "Popped the actually-bound slot(s) instead; the "
                "stale tensor cell is left alone (cleaned up by node "
                "teardown / `node.<attr> = None`). Will free the "
                "actually-bound slot(s) instead of the stale id. "
                "cum=%d/%d. UNBIND caller: %s",
                op, id(node), attr, position, slot,
                divergent_slots,
                cls._unbind_divergence_count, cap,
                callers,
            )
            return unbound

        if status == "wrong_node_at_slot":
            ref = payload
            logger.warning(
                "%s WRONG-NODE-AT-SLOT for slot %d: caller expected "
                "node 0x%x at attr=%r position=%d, but binder has "
                "node 0x%x at attr=%r position=%d. SKIPPED the unbind "
                "to preserve the other node's binding (this is the "
                "eval_results_40 corruption-prevention path; the "
                "redundant `_unbind_node_mamba_value` after "
                "`mamba_pool.free`'s eager-compaction rebind would "
                "otherwise silently steal the other node's entry). "
                "Will NOT free this slot (eval_results_42 fix — "
                "preserves allocator/binder invariant). cum=%d/%d.\n"
                "  UNBIND caller: %s",
                op, slot, id(node), attr, position,
                id(getattr(ref, "node", None)),
                getattr(ref, "attr", "?"),
                getattr(ref, "position", "?"),
                cls._unbind_divergence_count, cap,
                callers,
            )
            return unbound

        return unbound

    # Cap on VALUE-OVERLAP warnings (see `_check_value_overlap`).
    _VALUE_OVERLAP_WARN_CAP: int = 32
    # Cap on VALUE-FREED-SLOT warnings (see `_check_value_overlap`). This
    # is the high-value tripwire — a tree node about to be bound to a slot
    # the allocator has already freed — so give it a generous budget.
    _VALUE_FREED_SLOT_WARN_CAP: int = 256

    def _value_allocator_for_attr(self, attr: str):
        """Return the allocator that owns the slot id space for `attr`'s
        `node.<attr>` tensor, or None if it doesn't expose
        `is_slot_allocated` (non-shared pools). `attr='value'` → full-pool
        kv allocator; `attr='mamba_value'` → the mamba sub-pool."""
        if attr == "mamba_value":
            pool = getattr(self.req_to_token_pool, "mamba_pool", None)
            if pool is not None and hasattr(pool, "is_slot_allocated"):
                return pool
            return None
        alloc = getattr(self, "token_to_kv_pool_allocator", None)
        if alloc is not None and hasattr(alloc, "is_slot_allocated"):
            return alloc
        return None

    def _check_value_overlap(
        self,
        binder,
        value_tensor,
        node,
        *,
        attr: str,
        op: str,
    ) -> None:
        """Pre-flight diagnostic for `_set_node_value` /
        `_set_node_mamba_value` / `_bind_node_value`: scan `value_tensor`
        for slot ids that are ALREADY bound (in the binder) to a
        DIFFERENT tree node at the same `(attr, position)`. Each hit is
        a 1:1 invariant violation — the kv allocator handed the same
        physical slot to two reqs, OR `value_tensor`'s source data was
        stale (e.g., a `req_to_token` row whose cells weren't kept in
        sync by the apply step's `req_position` rewrites).

        Logs a single capped warning per call with up to 8 sample
        overlaps and the caller stack — that points directly at the
        upstream double-allocation source. The actual binds proceed
        unchanged; `bind_tree_node`'s OVERWRITE-abort logic (see
        :py:meth:`SlotBacktrack.bind_tree_node`) preserves the prior
        entries so the cascade can be traced from this warning.

        Cap-gated by ``_VALUE_OVERLAP_WARN_CAP`` (32). After the cap,
        the scan is skipped entirely to avoid the ``O(|value_tensor|)``
        dict-lookup cost per call in a degraded run.
        """
        if binder.is_null():
            return
        if not isinstance(value_tensor, torch.Tensor) or value_tensor.numel() == 0:
            return
        cls = type(self)
        cap = getattr(cls, "_VALUE_OVERLAP_WARN_CAP", 32)
        freed_cap = getattr(cls, "_VALUE_FREED_SLOT_WARN_CAP", 256)
        if not hasattr(cls, "_value_overlap_count"):
            cls._value_overlap_count = 0
        if not hasattr(cls, "_value_freed_count"):
            cls._value_freed_count = 0
        # Skip the O(|value_tensor|) scan only when BOTH tripwires are spent.
        if (
            cls._value_overlap_count >= cap
            and cls._value_freed_count >= freed_cap
        ):
            return

        # Access the underlying SlotBacktrack to read `tree_node` dict.
        bt = getattr(binder, "_backtrack", None)
        if bt is None or not hasattr(bt, "tree_node"):
            return  # null binder shouldn't have gotten here, defensive

        value_alloc = self._value_allocator_for_attr(attr)

        overlaps = []  # List[(pos, slot, prior_node_id, prior_attr, prior_pos)]
        freed = []  # List[(pos, slot)] — slots NOT in the allocator's range
        for pos, s in enumerate(value_tensor.detach().cpu().tolist()):
            slot = int(s)
            if slot <= 0:
                continue  # slot 0 = padding, never bound; negative = sentinel
            if value_alloc is not None and not value_alloc.is_slot_allocated(slot):
                freed.append((pos, slot))
                # A freed slot can also appear "bound to a different node"
                # (the prior owner) — but the freed diagnosis subsumes that,
                # so don't double-report.
                continue
            prior = bt.tree_node.get(slot)
            if prior is None:
                continue
            if (
                prior.node is not node
                or prior.attr != attr
                or prior.position != pos
            ):
                overlaps.append((
                    pos, slot, id(prior.node), prior.attr, prior.position,
                ))

        if not overlaps and not freed:
            return

        # Capture the caller stack (shared by both warnings).
        callers = "unknown"
        try:
            import inspect
            frames = inspect.stack()[2:9]
            callers = " <- ".join(
                f"{f.filename.split('/')[-1]}:{f.lineno}" for f in frames
            )
        except Exception:
            pass

        if freed and cls._value_freed_count < freed_cap:
            cls._value_freed_count += 1
            sample = freed[:8]
            more = len(freed) - len(sample)
            suffix = "" if more <= 0 else f", ...(+{more} more freed slots)"
            sample_str = "; ".join(
                f"pos={p} slot={s}" for (p, s) in sample
            )
            state = ""
            state_fn = getattr(value_alloc, "allocator_state_str", None)
            if state_fn is not None:
                try:
                    state = state_fn()
                except Exception:
                    state = ""
            logger.warning(
                "%s VALUE-FREED-SLOT for node 0x%x: about to bind %d "
                "slot(s) in node.%s that the allocator has ALREADY FREED "
                "(not in its allocated range). The source tensor (a "
                "req_to_token row or a tree node's value) carried stale "
                "slot ids — this is the laundering point that puts freed "
                "slots into the radix tree and crashes later on "
                "alloc()/free(). cum=%d/%d.\n"
                "  Freed slots (first %d of %d): %s%s\n"
                "  Allocator: %s\n"
                "  Caller: %s",
                op, id(node), len(freed), attr,
                cls._value_freed_count, freed_cap,
                len(sample), len(freed), sample_str, suffix,
                state or "(unavailable)",
                callers,
            )

        if overlaps and cls._value_overlap_count < cap:
            cls._value_overlap_count += 1
            sample = overlaps[:8]
            more = len(overlaps) - len(sample)
            suffix = "" if more <= 0 else f", ...(+{more} more overlapping slots)"
            sample_str = "; ".join(
                f"pos={p} slot={s} (already at node 0x{n:x} attr={a!r} pos={pp})"
                for (p, s, n, a, pp) in sample
            )
            logger.warning(
                "%s VALUE-OVERLAP for node 0x%x: about to bind %d slots in "
                "node.%s, but %d slot(s) are ALREADY bound to a DIFFERENT "
                "tree node. The kv allocator handed the same physical slot "
                "to two reqs, OR the source value tensor was stale. "
                "Subsequent `bind_tree_node` calls will ABORT (not replace) "
                "for the overlapping slots — see "
                "`SlotBacktrack.bind_tree_node` OVERWRITE-abort. cum=%d/%d.\n"
                "  Overlapping slots (first %d): %s%s\n"
                "  Caller: %s",
                op, id(node), value_tensor.numel(), attr, len(overlaps),
                cls._value_overlap_count, cap,
                len(sample), sample_str, suffix,
                callers,
            )

    # -- Req.prefix_indices binding.
    # `req.prefix_indices` is a clone of FULL-pool slot ids captured at
    # radix-cache lookup / cache-finish time. Without binding, eager-compaction
    # relocations applied via the relocation log update `req_to_token` and
    # `TreeNode.value` (both bound) but NOT the prefix_indices clone — so the
    # next prefill that reads `req.prefix_indices` (via
    # `common.alloc_for_extend` line 351) writes stale slot ids back into
    # `req_to_token`, and a later free of those cells fires the
    # MultiEndedAllocator out-of-range assertion (observed across all 8
    # `shared` cells in the 2026-05-03 second eval run).
    #
    # We use cell-level `bind_aux` per slot; on flush the binder writes
    # `prefix_indices[pos] = dst`, keeping the clone in sync.

    def _set_req_prefix_indices(self, req, new_value) -> None:
        """Route a `req.prefix_indices = new_value` assignment through the
        full-pool SlotBacktrackBinder so eager-compaction relocations of
        those slot ids reach this Req's clone tensor.

        For empty / None new_value: just clears any prior bindings."""
        binder = self._slot_backtrack_binder()
        if not binder.is_null():
            old = getattr(req, "prefix_indices", None)
            if isinstance(old, torch.Tensor) and old.numel() > 0:
                for pos, s in enumerate(old.detach().cpu().tolist()):
                    binder.unbind_aux(int(s), old, pos)
        req.prefix_indices = new_value
        if not binder.is_null():
            if isinstance(new_value, torch.Tensor) and new_value.numel() > 0:
                for pos, s in enumerate(new_value.detach().cpu().tolist()):
                    binder.bind_aux(int(s), new_value, pos)

    # -- Local-`value`-tensor aux binding for `_insert_helper`.
    # The radix cache's `_insert_helper` keeps a local Python `value`
    # tensor (the req's KV slot ids for the keys being inserted). Inside
    # the loop it (a) calls `_free_dedup_slots(value[start:prefix_len],
    # node.value[start:prefix_len])` — which can trigger eager
    # compaction relocating a boundary slot that happens to live inside
    # `value` — and (b) calls `_set_node_value(new_node, value.clone())`
    # using the (possibly post-relocation) suffix.
    #
    # Without aux binding, the local `value` tensor is not tracked by
    # `Relocator.apply`, so any in-place rewrite of relocated slot ids
    # never reaches it. The result is stale ids leaking into the radix
    # tree (manifests as `bind_tree_node OVERWRITE` on the next insert
    # that splits the same prefix) or into a later `_free_dedup_slots`
    # call (manifests as a STALE-SLOT assertion when the freed id is
    # already above the watermark).
    #
    # The fix: temporarily bind every cell of `value` as an `aux` ref
    # for the duration of `_insert_helper`. `apply` then updates the
    # cells in place exactly the way it updates `req.prefix_indices`.
    # On exit we unbind by re-reading the (now post-relocation) cell
    # contents — `apply` already moved any binder entries that were at
    # pre-relocation slot ids onto the post-relocation ids, so reading
    # the current cells gives us the correct slot ids to unbind.

    def _bind_value_as_aux(self, value) -> None:
        """Bind every cell of the 1-D slot-id tensor `value` as an aux
        backtrack reference. No-op when the binder is null (non-shared
        pool) or when `value` is not a non-empty tensor.

        Pair with :py:meth:`_unbind_value_as_aux` in a try/finally so
        the bindings are dropped even if `_insert_helper` raises.
        """
        binder = self._slot_backtrack_binder()
        if binder.is_null():
            return
        if not isinstance(value, torch.Tensor) or value.numel() == 0:
            return
        for pos, s in enumerate(value.detach().cpu().tolist()):
            binder.bind_aux(int(s), value, pos)

    def _unbind_value_as_aux(self, value) -> None:
        """Drop the aux bindings registered by
        :py:meth:`_bind_value_as_aux`.

        Reads the CURRENT cell contents — `Relocator.apply` may have
        rewritten cells in place during the bind window — so we unbind
        the slot ids that the binder actually holds right now.
        """
        binder = self._slot_backtrack_binder()
        if binder.is_null():
            return
        if not isinstance(value, torch.Tensor) or value.numel() == 0:
            return
        for pos, s in enumerate(value.detach().cpu().tolist()):
            binder.unbind_aux(int(s), value, pos)

    # -- Mamba-side helper: bind Req.mamba_pool_idx when set by radix cache
    # operations (cache-hit fork_from, etc.). Only applicable to the Mamba
    # hybrid path; a null binder short-circuits for non-shared pools.

    def _set_req_mamba_pool_idx(self, req, new_value) -> None:
        """Assign `req.mamba_pool_idx = new_value` and route the change into
        the Mamba SlotBacktrackBinder. No-op for non-shared paths."""
        from sglang.srt.mem_cache.relocation_log import null_backtrack_binder

        rtp = getattr(self, "req_to_token_pool", None)
        mamba_pool = getattr(rtp, "mamba_pool", None) if rtp else None
        binder = getattr(
            mamba_pool, "slot_backtrack_binder", null_backtrack_binder()
        )
        old = req.mamba_pool_idx
        if not binder.is_null() and old is not None:
            try:
                old_slot = int(old.item()) if hasattr(old, "item") else int(old)
                binder.unbind_py_attr(old_slot, req, "mamba_pool_idx")
            except Exception:
                pass
        req.mamba_pool_idx = new_value
        if not binder.is_null() and new_value is not None:
            try:
                new_slot = (
                    int(new_value.item())
                    if hasattr(new_value, "item")
                    else int(new_value)
                )
                binder.bind_py_attr(new_slot, req, "mamba_pool_idx")
            except Exception:
                pass

    def full_evictable_size(self):
        return 0

    def swa_evictable_size(self):
        return 0

    def protected_size(self):
        return 0

    def full_protected_size(self):
        return 0

    def swa_protected_size(self):
        return 0

    def total_size(self):
        raise NotImplementedError()

    def pretty_print(self):
        raise NotImplementedError()

    def init_load_back(
        self,
        params: InitLoadBackParams,
    ) -> Tuple[torch.Tensor, Any]:
        """
        Preparing KV cache loading from host to device.
        """
        raise NotImplementedError()

    def ready_to_load_host_cache(self) -> Any:
        """
        Notify the cache controller to start the KV cache loading
        """
        raise NotImplementedError()

    def flush_write_through_acks(self) -> None:
        """Release lock_ref on radix-tree nodes whose write-through has completed.

        Lightweight operation that only processes finished write acks.
        No-op for caches without hierarchical write-through support.
        """
        pass

    def check_hicache_events(self) -> Any:
        """
        Check HiCache related activities to update radix tree and synchronize across TP workers if needed
        """
        raise NotImplementedError()

    def take_events(self):
        return []

    def supports_swa(self) -> bool:
        return False

    def supports_mamba(self) -> bool:
        return False

    def is_chunk_cache(self) -> bool:
        return False

    def is_tree_cache(self) -> bool:
        return not self.is_chunk_cache()

    def available_and_evictable_str(self) -> str:
        available_size = self.token_to_kv_pool_allocator.available_size()
        evictable_size = self.evictable_size()
        return f"Available tokens: {available_size + evictable_size} ({available_size=} + {evictable_size=})\n"
