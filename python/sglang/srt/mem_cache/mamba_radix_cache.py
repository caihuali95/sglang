from __future__ import annotations

"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
The radix tree data structure for managing the hybrid (full and Mamba) KV cache.
"""

import heapq
from collections import defaultdict
from functools import lru_cache, partial
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch
from numpy import float64

from sglang.srt.distributed import get_tensor_model_parallel_rank
from sglang.srt.layers.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE
from sglang.srt.mem_cache.allocator import (
    PagedTokenToKVPoolAllocator,
    TokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    DecLockRefParams,
    DecLockRefResult,
    EvictParams,
    EvictResult,
    IncLockRefResult,
    InsertParams,
    InsertResult,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
from sglang.srt.mem_cache.radix_cache import (
    RadixKey,
    _key_match_page_size1,
    _key_match_paged,
    get_child_key,
)
from sglang.srt.server_args import get_global_server_args

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams

import logging

logger = logging.getLogger(__name__)


class TreeNode:

    counter = 0
    last_access_time_counter_float = float64(1.0)

    def __init__(self, id: Optional[int] = None):
        self.children = defaultdict(TreeNode)
        self.parent: TreeNode = None
        self.key: RadixKey = None
        self.value: Optional[torch.Tensor] = None
        self.mamba_value: Optional[torch.Tensor] = None
        self.mamba_host_value: Optional[torch.Tensor] = None
        # invariant: for any node, if mamba_lock_ref is locked, full_lock_ref must be locked;
        # if full_lock_ref is locked, mamba_lock_ref doesn't need to be locked. So,
        # full_lock_ref is always >= mamba_lock_ref.
        # for full_lock, once it is locked, its parent must be locked as well
        # for mamba_lock, it only need lock node itself
        self.full_lock_ref = 0
        self.mamba_lock_ref = 0
        # last access time is only used for sanity check. LRU is maintained by the lru list.
        self.last_access_time = get_last_access_time()

        self.hit_count = 0
        self.host_ref_counter = 0
        self.host_mamba_ref_counter = 0
        # store the host indices of KV cache
        self.host_value = None
        # store hash values of each pages
        self.hash_value: Optional[List[str]] = None

        # for lru list, invariant:
        # 1. prev has greater last_access_time
        # 2. next has smaller last_access_time
        self.prev = None
        self.next = None
        self.mamba_prev = None
        self.mamba_next = None
        self.host_mamba_prev = None
        self.host_mamba_next = None

        self.id = TreeNode.counter if id is None else id
        TreeNode.counter += 1

    @property
    def evicted(self):
        return self.value is None

    @property
    def mamba_evicted(self):
        return self.mamba_value is None

    @property
    def backuped(self):
        return self.host_value is not None

    @property
    def mamba_backuped(self):
        return self.mamba_host_value is not None

    def protect_host(self):
        """Protect the host KV value from eviction."""
        self.host_ref_counter += 1

    def release_host(self):
        """Release the host KV value, allowing it to be evicted."""
        if self.host_ref_counter > 0:
            self.host_ref_counter -= 1
        else:
            raise RuntimeError("Host reference counter is already zero.")

    def protect_host_mamba(self):
        """Protect the host mamba value from eviction."""
        self.host_mamba_ref_counter += 1

    def release_host_mamba(self):
        """Release the host mamba value, allowing it to be evicted."""
        if self.host_mamba_ref_counter > 0:
            self.host_mamba_ref_counter -= 1
        else:
            raise RuntimeError("Host mamba reference counter is already zero.")

    def get_last_hash_value(self) -> Optional[str]:
        """Returns the hash value of the last page in this node."""
        if self.hash_value is None or len(self.hash_value) == 0:
            return None
        return self.hash_value[-1]

    @lru_cache(maxsize=1)
    def get_prefix_hash_values(self, node: "TreeNode") -> List[str]:
        if node is None or node.hash_value is None:
            return []
        return node.get_prefix_hash_values(node.parent) + node.hash_value

    def __lt__(self, other: "TreeNode"):
        return self.last_access_time < other.last_access_time


def get_last_access_time() -> float64:
    ret = TreeNode.last_access_time_counter_float
    TreeNode.last_access_time_counter_float += 1.0
    return ret


class LRUList:
    def __init__(self, mamba: bool = False):
        self.mamba = mamba
        if self.mamba:
            self.prv = "mamba_prev"
            self.nxt = "mamba_next"
            self.lock_ref = "mamba_lock_ref"
        else:
            self.prv = "prev"
            self.nxt = "next"
            self.lock_ref = "full_lock_ref"
        # Initialize dummy head and tail nodes
        self.head = TreeNode()  # Most recently used side
        self.tail = TreeNode()  # Least recently used side
        setattr(self.head, self.nxt, self.tail)  # self.head.next = self.tail
        setattr(self.tail, self.prv, self.head)  # self.tail.prev = self.head
        self.cache = {}

    def _add_node(self, node):
        """Helper to add node right after head (most recently used)"""
        self._add_node_after(self.head, node)

    def _add_node_after(self, old_node, new_node):
        """Helper to add node right after old_node"""
        setattr(new_node, self.prv, old_node)  # new_node.prev = old_node
        setattr(
            new_node, self.nxt, getattr(old_node, self.nxt)
        )  # new_node.next = old_node.next
        setattr(
            getattr(old_node, self.nxt), self.prv, new_node
        )  # old_node.next.prev = new_node
        setattr(old_node, self.nxt, new_node)  # old_node.next = new_node

    def _remove_node(self, node):
        """Helper to remove node from linked list"""
        setattr(
            getattr(node, self.prv), self.nxt, getattr(node, self.nxt)
        )  # node.prev.next = node.next
        setattr(
            getattr(node, self.nxt), self.prv, getattr(node, self.prv)
        )  # node.next.prev = node.prev

    def _get_lru(self) -> Optional[TreeNode]:
        """
        Get the least recently used node
        """
        if len(self.cache) == 0:
            return None
        return getattr(self.tail, self.prv)

    def reset_node_mru(self, node):
        """
        Move a (existing) node to most recently used position
        """
        assert node.id in self.cache, f"Resetting node {node.id=} not in lru list"
        assert (
            not self.mamba or node.mamba_value is not None
        ), f"Resetting mamba tombstone node in mamba lru list: {node.id=}"
        self._remove_node(node)
        self._add_node(node)

    def reset_node_and_parents_mru(self, node, root_node):
        """
        Move an (existing) node and its parents to most recently used position. Child node is
        more recently used than parent node.
        """
        prev_node = self.head
        while node != root_node:
            if not self.mamba or node.mamba_value is not None:
                assert (
                    node.id in self.cache
                ), f"Resetting node {node.id=} not in lru list when resetting node and parents mru"
                self._remove_node(node)
                self._add_node_after(prev_node, node)
                prev_node = node
            node = node.parent

    def insert_mru(self, node):
        """
        Insert a (new) node as most recently used
        """
        assert (
            not self.mamba or node.mamba_value is not None
        ), f"Inserting mamba tombstone node in mamba lru list: {node.id=}"
        assert (
            node.id not in self.cache
        ), f"Inserting node {node.id=} already in lru list, existing node: {self.cache[node.id].id=}"
        self.cache[node.id] = node
        self._add_node(node)

    def remove_node(self, node: TreeNode):
        """
        Remove node from lru list
        """
        assert node.id in self.cache, f"Removing node {node.id=} not in lru list"
        assert (
            not self.mamba or node.mamba_value is not None
        ), f"Removing mamba tombstone node from mamba lru list: {node.id=}"
        del self.cache[node.id]
        self._remove_node(node)

    def get_lru_no_lock(self) -> Optional[TreeNode]:
        """
        Get the least recently used node that is not locked
        """
        return self.get_prev_no_lock(self.tail, check_id=False)

    def get_leaf_lru_no_lock(self) -> Optional[TreeNode]:
        """
        Get the least recently used leaf node that is not locked
        """
        return self.get_prev_leaf_no_lock(self.tail, check_id=False)

    def get_prev_no_lock(
        self, node: TreeNode, check_id: bool = True
    ) -> Optional[TreeNode]:
        """
        Get the previous (i.e. more recently used) node that is not locked
        """
        if check_id:
            assert (
                node.id in self.cache
            ), f"Getting prev of node {node.id=} not in lru list"
        x = getattr(node, self.prv)  # x = node.prev
        while getattr(x, self.lock_ref) > 0:
            x = getattr(x, self.prv)  # x = x.prev
        # if x is the head, it means there is no node in the lru list without lock
        if x == self.head:
            return None
        return x

    def get_prev_leaf_no_lock(self, node: TreeNode, check_id: bool = True):
        """
        Get the previous (i.e. more recently used) leaf node that is not locked
        """
        if check_id:
            assert (
                node.id in self.cache
            ), f"Getting prev of node {node.id=} not in lru list"
        x = getattr(node, self.prv)  # x = node.prev
        while getattr(x, self.lock_ref) > 0 or len(x.children) > 0:
            x = getattr(x, self.prv)  # x = x.prev
        # if x is the head, it means there is no leaf node in the lru list without lock
        if x == self.head:
            return None
        return x

    def in_list(self, node: Optional[TreeNode]):
        """
        Check if the node is in the lru list
        """
        if not node:
            return False
        return node.id in self.cache

    def pretty_print(self, tree_cache: Optional["MambaRadixCache"] = None):
        """
        Pretty print the lru list
        """
        msg = f"{self.mamba=} LRU list: "
        x_lru = self._get_lru()
        while x_lru is not None and x_lru.id in self.cache:
            msg += f"[{x_lru.id}] {x_lru.last_access_time:f} -> "
            x_lru = getattr(x_lru, self.prv)
        print(msg)

        if not tree_cache:
            return
        msg = f"{self.mamba=} Nodes (sorted by last_access_time): "
        if self.mamba:
            nodes = tree_cache._collect_nontombstone_nodes()
        else:
            nodes = tree_cache._collect_all_nodes()
        heapq.heapify(nodes)
        while len(nodes):
            x = heapq.heappop(nodes)
            msg += f"[{x.id}] {x.last_access_time:f} -> "
        print(msg)

    # Note: this is expensive, only use for debug
    def sanity_check_evictable_size(self):
        """
        Check the evictable size (i.e. the size of the nodes that are not locked)
        """
        node = self.get_lru_no_lock()
        evictable_size = 0
        while self.in_list(node):
            evictable_size += (
                len(node.value) if not self.mamba else len(node.mamba_value)
            )
            node = self.get_prev_no_lock(node)
        return evictable_size

    # Note: this is expensive, only use for debug or idle check
    def sanity_check(self, tree_cache: "MambaRadixCache"):
        """
        Check if the lru list is valid by rebuilding the lru list from the tree, heapifying it, and
        checking if the lru list is valid.
        """
        try:
            if self.mamba:
                nodes = tree_cache._collect_nontombstone_nodes()
            else:
                nodes = tree_cache._collect_all_nodes()
            total_nodes = len(nodes)
            total_lru = len(self.cache)
            # heapify based on last_access_time
            heapq.heapify(nodes)
            # the root node is not in the lru list
            assert len(nodes) == (
                total_lru + (0 if self.mamba else 1)
            ), f"len(nodes): {len(nodes)}, total_lru: {total_lru}"

            x_lru = self._get_lru()
            while len(nodes):
                x = heapq.heappop(nodes)
                if x == tree_cache.root_node:
                    # root node is not in the lru list
                    continue
                assert (
                    x_lru is not None and x_lru.id in self.cache
                ), f"Incorrect LRU list, x_lru is None or not in cache: {x_lru=}, {x.id=}"

                assert (
                    x == x_lru
                ), f"Incorrect LRU list, {self.mamba=}, x: {x.id=} != x_lru: {x_lru.id=}, {x.last_access_time=}, {x_lru.last_access_time=}"
                assert (
                    x_lru.full_lock_ref == 0
                ), f"x_lru should not be locked when idle, {x_lru.full_lock_ref=}, {x_lru.id=}"
                assert (
                    x_lru.mamba_lock_ref == 0
                ), f"x_lru should not be locked when idle, {x_lru.mamba_lock_ref=}, {x_lru.id=}"
                x_lru = getattr(x, self.prv)

            if self.mamba:
                evictable_size = tree_cache.mamba_evictable_size()
                lru_list_evictable_size = self.sanity_check_evictable_size()
            else:
                evictable_size = tree_cache.full_evictable_size()
                lru_list_evictable_size = self.sanity_check_evictable_size()

            assert (
                evictable_size == lru_list_evictable_size
            ), f"{self.mamba=}, total nodes: {total_nodes}, total lru: {total_lru}, evictable size: {evictable_size} != lru list evictable size: {lru_list_evictable_size}"
        except Exception as e:
            if get_tensor_model_parallel_rank() == 0:
                msg = f"Mamba Radix tree sanity check failed, ping @yizhang2077: {e}"
                logger.error(msg)
                tree_cache.pretty_print()
                tree_cache.full_lru_list.pretty_print(tree_cache)
                tree_cache.mamba_lru_list.pretty_print(tree_cache)
                raise Exception(msg)


class MambaRadixCache(BasePrefixCache):
    def __init__(self, params: CacheInitParams):
        # Accept the shared-memory-pool variant too (lazy import to avoid a
        # circular dependency with multi_ended_allocator -> shared_memory_pool
        # -> memory_pool -> mamba_radix_cache).
        from sglang.srt.mem_cache.multi_ended_allocator import (
            SharedMambaTokenToKVPoolAllocator,
        )
        assert isinstance(
            params.token_to_kv_pool_allocator,
            (
                TokenToKVPoolAllocator,
                PagedTokenToKVPoolAllocator,
                SharedMambaTokenToKVPoolAllocator,
            ),
        ), (
            f"MambaRadixCache: unsupported allocator type "
            f"{type(params.token_to_kv_pool_allocator).__name__}"
        )
        self.req_to_token_pool: HybridReqToTokenPool = params.req_to_token_pool
        self.token_to_kv_pool_allocator = params.token_to_kv_pool_allocator

        self.page_size = params.page_size
        self.disable = params.disable
        self.enable_mamba_extra_buffer = params.enable_mamba_extra_buffer

        if not self.enable_mamba_extra_buffer:
            assert (
                self.page_size == 1
            ), f"Page size must be 1 for MambaRadixCache v1, got {self.page_size}"

        if self.token_to_kv_pool_allocator:
            self.device = self.token_to_kv_pool_allocator.device
        else:
            self.device = torch.device("cpu")

        if params.enable_metrics:
            self.init_metrics_collector()

        if self.page_size == 1:
            self.key_match_fn = _key_match_page_size1
            self.get_child_key_fn = get_child_key
        else:
            self.key_match_fn = partial(_key_match_paged, page_size=self.page_size)
            self.get_child_key_fn = partial(get_child_key, page_size=self.page_size)
        self.reset()

    ##### Public API #####

    def supports_mamba(self) -> bool:
        return True

    def reset(self) -> None:
        self.root_node = TreeNode()
        self.root_node.key = RadixKey([], None)
        self.root_node.value = []
        self.root_node.full_lock_ref = 1
        self.root_node.mamba_lock_ref = 1
        self.full_evictable_size_ = 0
        self.mamba_evictable_size_ = 0
        self.full_protected_size_ = 0
        self.mamba_protected_size_ = 0
        # LRU lists are used to maintain the order of eviction of the nodes in the tree
        self.full_lru_list = LRUList(mamba=False)
        self.mamba_lru_list = LRUList(mamba=True)

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        """Find the matching prefix from the radix tree.
        Args:
            params: MatchPrefixParams containing key and optional Mamba-specific parameters.
        Returns:
            A tuple of a tensor of matching prefix token IDs and
            the last node that contains the prefix values. Note that
            this API can modify the internal state of the Radix tree.
            The last node create a new child if the prefix is shorter
            than the last node's value.
        """
        key = self._match_pre_processor(params)
        if key is None:
            return MatchResult(
                device_indices=torch.empty(
                    (0,),
                    dtype=torch.int64,
                    device=self.device,
                ),
                last_device_node=self.root_node,
                last_host_node=self.root_node,
            )

        value, last_node, best_value_len = self._match_prefix_helper(key)
        return self._match_post_processor(params, value, last_node, best_value_len)

    def insert(self, params: InsertParams) -> InsertResult:
        if self.disable:
            return InsertResult(prefix_len=0, mamba_exist=False)

        key = params.key
        value = params.value
        mamba_value = params.mamba_value
        prev_prefix_len = params.prev_prefix_len

        if value is None:
            value = torch.tensor([x for x in key.token_ids], dtype=torch.int64)

        # Bind the local `value` tensor as aux so any eager-compaction
        # relocation triggered inside `_insert_helper` (via
        # `_free_dedup_slots`) updates `value`'s cells in place. See
        # :py:meth:`BasePrefixCache._bind_value_as_aux` for rationale.
        self._bind_value_as_aux(value)
        try:
            prefix_len, mamba_exist = self._insert_helper(
                self.root_node, key, value, mamba_value, params.chunked, prev_prefix_len
            )
        finally:
            self._unbind_value_as_aux(value)
        return InsertResult(prefix_len=prefix_len, mamba_exist=mamba_exist)

    # -- defensive validation backstop ------------------------------------
    #
    # eval_results_36 demonstrated that even after the
    # `transfer_mamba_to_radix` fix sealed the dominant
    # `cache_finished_req(mamba_exist=False)` ownership leak, a residual
    # cascade still surfaces as `mamba_pool.free([stale_id])` from
    # `_evict_leaf_node`. We have not yet localized the upstream write
    # that produces the stale `req.mamba_pool_idx`; the strengthened
    # weakref orphan tripwire (relocation_log.py) is part 1 of finding
    # it. This validation backstop is part 2: refuse to launder a stale
    # mamba slot id into the radix tree, breaking the chain at the
    # point where it would otherwise become silent corruption.
    #
    # On detection: log a capped diagnostic warning with allocator state
    # and caller chain (so the next eval round has the upstream context),
    # then abort the cache for this Req — free its kv slots locally and
    # skip the insert.

    def _is_mamba_pool_idx_valid(self, req) -> bool:
        """Returns True iff `req.mamba_pool_idx` names a slot that is
        currently in the mamba allocator's allocated range. Returns False
        for None, sentinel `-1`, structurally out-of-range ids, and ids
        that fell into the reserved-zone after a watermark advance. The
        caller decides what to do with a False (typically: log + abort)."""
        mid = getattr(req, "mamba_pool_idx", None)
        if mid is None:
            return False
        try:
            slot = int(mid.item()) if hasattr(mid, "item") else int(mid)
        except Exception:
            return False
        pool = self.req_to_token_pool.mamba_pool
        if hasattr(pool, "is_slot_allocated"):
            return pool.is_slot_allocated(slot)
        # Non-shared pool: best we can do is a positivity check.
        return slot > 0

    @staticmethod
    def _stale_mamba_warn_cap() -> int:
        # Cap on the `STALE req.mamba_pool_idx` / `STALE kv-indices` backstop
        # warnings (the aborts themselves are uncapped — only the logging is).
        return 64

    def _warn_stale_mamba_pool_idx(self, req, op: str) -> None:
        """One-shot capped warning for a detected stale `req.mamba_pool_idx`.
        Includes the offending slot id, allocator state, identifying req
        fields, and the call stack — everything we need to localize the
        upstream write that produced the stale value."""
        if not hasattr(self, "_stale_mamba_warn_count"):
            self._stale_mamba_warn_count = 0
        self._stale_mamba_warn_count += 1
        cap = self._stale_mamba_warn_cap()
        if cap == 0 or self._stale_mamba_warn_count > cap:
            return

        mid = getattr(req, "mamba_pool_idx", None)
        try:
            slot_repr = (
                "None"
                if mid is None
                else f"{int(mid.item())}"
                if hasattr(mid, "item")
                else f"{int(mid)}"
            )
        except Exception:
            slot_repr = repr(mid)

        pool = self.req_to_token_pool.mamba_pool
        state_str = (
            pool.allocator_state_str()
            if hasattr(pool, "allocator_state_str")
            else "<unknown>"
        )

        import inspect
        frames = inspect.stack()[2:10]
        callers = " <- ".join(
            f"{f.filename.split('/')[-1]}:{f.lineno}" for f in frames
        )

        logger.warning(
            "MambaRadixCache.%s: STALE req.mamba_pool_idx detected — "
            "slot=%s is not in the mamba allocator's allocated range. "
            "Aborting cache for this req to avoid laundering a stale id "
            "into the radix tree (would crash later via mamba_pool.free). "
            "Allocator: %s. req fields: req_pool_idx=%s, rid=%s, "
            "obj_id=0x%x, retraction_count=%s. cum_stale=%d/%d. Caller: %s",
            op,
            slot_repr,
            state_str,
            getattr(req, "req_pool_idx", "?"),
            getattr(req, "rid", "?"),
            id(req),
            getattr(req, "retraction_count", "?"),
            self._stale_mamba_warn_count,
            cap,
            callers,
        )

    def _abort_cache_finished_for_stale_mamba(
        self, req, kv_indices: torch.Tensor
    ) -> None:
        """Defensive abort path for `cache_finished_req` when
        `req.mamba_pool_idx` is stale. Frees the FULL-pool kv slots
        locally (otherwise leaked) and clears any binder bookkeeping
        that may still tie the Req to the stale mamba slot. Does NOT
        call `mamba_pool.free` — the slot is stale and the call would
        crash on the watermark assertion."""
        # Free full-pool kv slots that would have been inserted.
        if kv_indices.numel() > 0:
            self.token_to_kv_pool_allocator.free(kv_indices)
        # Best-effort: drop binder entries that might still tie the Req
        # to a (possibly stale) mamba slot. `transfer_mamba_to_radix`
        # tolerates stale ids — its unbind_py_attr / unbind_aux are
        # identity-keyed by `(slot, obj, attr)` / `(slot, tensor, idx)`
        # and silently no-op when no entry matches. After the call,
        # `req.mamba_pool_idx` is None.
        if getattr(req, "mamba_pool_idx", None) is not None and hasattr(
            self.req_to_token_pool, "transfer_mamba_to_radix"
        ):
            self.req_to_token_pool.transfer_mamba_to_radix(req)
        else:
            req.mamba_pool_idx = None

    # -- full-pool kv-indices validation backstop --------------------------
    #
    # If a request's `req_to_token` row ever holds a FULL-pool slot id the
    # allocator has already freed, `cache_*_req` would launder it into the
    # radix tree via `_insert_helper` → `_set_node_value`, where it becomes
    # silent corruption that surfaces later as a stale-slot / multi-col
    # assertion. This backstop refuses the launder: if any slot in the
    # kv-indices about to be cached is outside the allocator's allocated
    # range, log a capped diagnostic and skip/abort the cache for this Req.

    def _are_kv_indices_valid(self, kv_indices: torch.Tensor) -> bool:
        """True iff every positive slot id in `kv_indices` is currently in
        the full-pool allocator's allocated range. Non-positive ids
        (0 = padding, negative = sentinel) count as INVALID. Returns True
        when the allocator can't answer (non-shared pool)."""
        if not isinstance(kv_indices, torch.Tensor) or kv_indices.numel() == 0:
            return True
        is_alloc = getattr(
            self.token_to_kv_pool_allocator, "is_slot_allocated", None
        )
        if is_alloc is None:
            return True
        for s in kv_indices.detach().cpu().tolist():
            si = int(s)
            if si <= 0:
                return False
            if not is_alloc(si):
                return False
        return True

    def _first_invalid_kv_indices(self, kv_indices: torch.Tensor):
        """Return up to 8 (pos, slot) pairs that fail `_are_kv_indices_valid`,
        plus the total count, for diagnostics."""
        is_alloc = getattr(
            self.token_to_kv_pool_allocator, "is_slot_allocated", None
        )
        bad = []
        total = 0
        for pos, s in enumerate(kv_indices.detach().cpu().tolist()):
            si = int(s)
            if si <= 0 or (is_alloc is not None and not is_alloc(si)):
                total += 1
                if len(bad) < 8:
                    bad.append((pos, si))
        return bad, total

    def _warn_stale_kv_indices(
        self, req, op: str, kv_indices: torch.Tensor
    ) -> None:
        """Capped warning for kv-indices carrying freed full-pool slot ids.
        Shares `_stale_mamba_warn_cap()` with the other backstop warnings."""
        if not hasattr(self, "_stale_kv_warn_count"):
            self._stale_kv_warn_count = 0
        self._stale_kv_warn_count += 1
        cap = self._stale_mamba_warn_cap()
        if cap == 0 or self._stale_kv_warn_count > cap:
            return
        bad, total = self._first_invalid_kv_indices(kv_indices)
        bad_str = "; ".join(f"pos={p} slot={s}" for (p, s) in bad)
        more = total - len(bad)
        if more > 0:
            bad_str += f", ...(+{more} more)"
        alloc = self.token_to_kv_pool_allocator
        state_str = (
            alloc.allocator_state_str()
            if hasattr(alloc, "allocator_state_str")
            else "<unknown>"
        )
        import inspect
        frames = inspect.stack()[2:10]
        callers = " <- ".join(
            f"{f.filename.split('/')[-1]}:{f.lineno}" for f in frames
        )
        logger.warning(
            "MambaRadixCache.%s: STALE kv-indices detected — %d of %d "
            "slot id(s) in the req_to_token row are NOT in the full-pool "
            "allocator's allocated range (already freed). Aborting the "
            "cache for this req to avoid laundering freed slots into the "
            "radix tree. Bad slots: %s. Allocator: %s. req fields: "
            "req_pool_idx=%s, rid=%s, obj_id=0x%x, cache_protected_len=%s, "
            "retraction_count=%s. cum_stale_kv=%d/%d. Caller: %s",
            op,
            total,
            kv_indices.numel(),
            bad_str,
            state_str,
            getattr(req, "req_pool_idx", "?"),
            getattr(req, "rid", "?"),
            id(req),
            getattr(req, "cache_protected_len", "?"),
            getattr(req, "retraction_count", "?"),
            self._stale_kv_warn_count,
            cap,
            callers,
        )

    def _abort_cache_finished_for_stale_kv(
        self, req, kv_indices: torch.Tensor
    ) -> None:
        """Abort `cache_finished_req` when kv-indices carry freed ids.
        Frees only the suffix beyond `cache_protected_len` (the req's own
        fresh decode slots), and only the ones still in the allocated
        range — the protected-prefix part belongs to the tree and the
        out-of-range ids are already accounted for as free. Then releases
        the mamba state. Does NOT insert anything into the tree."""
        protected = int(getattr(req, "cache_protected_len", 0) or 0)
        if isinstance(kv_indices, torch.Tensor) and kv_indices.numel() > protected:
            suffix = kv_indices[protected:]
            is_alloc = getattr(
                self.token_to_kv_pool_allocator, "is_slot_allocated", None
            )
            if is_alloc is None:
                self.token_to_kv_pool_allocator.free(suffix)
            else:
                safe = [
                    int(s) for s in suffix.detach().cpu().tolist()
                    if int(s) > 0 and is_alloc(int(s))
                ]
                if safe:
                    self.token_to_kv_pool_allocator.free(
                        torch.tensor(
                            safe, dtype=kv_indices.dtype, device=kv_indices.device
                        )
                    )
        # Release the mamba state (the req is finishing; it isn't going
        # into the tree on this aborted path).
        self.req_to_token_pool.free_mamba_cache(req)

    def cache_finished_req(self, req: Req, is_insert: bool = True) -> None:
        """Cache request when it finishes."""
        kv_committed_len = req.pop_committed_kv_cache()
        if self.disable:
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, :kv_committed_len
            ]
            self.token_to_kv_pool_allocator.free(kv_indices)
            self.req_to_token_pool.free_mamba_cache(req)
            return
        # Defer eager compaction triggered inside this call (the `insert`'s
        # `_free_dedup_slots`, and the `free(kv_indices[...])` paths) until
        # every bound holder is in place — so relocations land on bound
        # holders, not on the in-flight `value` clone. Only the OUTERMOST
        # opener manages the group (the decode result path already wraps
        # `release_kv_cache` in one; nesting `free_group_begin()` would
        # reset `free_group = []` and lose the outer batch).
        alloc = self.token_to_kv_pool_allocator
        own_free_group = bool(getattr(alloc, "is_not_in_free_group", True))
        if own_free_group:
            alloc.free_group_begin()
        try:
            self._cache_finished_req_inner(req, is_insert, kv_committed_len)
        finally:
            if own_free_group:
                alloc.free_group_end()

    def _cache_finished_req_inner(
        self, req: Req, is_insert: bool, kv_committed_len: int
    ) -> None:
        token_ids = (req.origin_input_ids + req.output_ids)[:kv_committed_len]
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :kv_committed_len
        ]

        if is_insert:
            cache_len = (
                req.mamba_last_track_seqlen
                if self.enable_mamba_extra_buffer
                else len(token_ids)
            )
            if cache_len is None:
                cache_len = 0
            if cache_len != len(token_ids):
                cache_end_idx = max(cache_len, req.cache_protected_len)
                self.token_to_kv_pool_allocator.free(kv_indices[cache_end_idx:])
                token_ids = token_ids[:cache_len]
                kv_indices = kv_indices[:cache_len]

            if self.page_size != 1:
                page_aligned_len = len(kv_indices) // self.page_size * self.page_size
                page_aligned_kv_indices = kv_indices[:page_aligned_len].to(
                    dtype=torch.int64, copy=True
                )
            else:
                page_aligned_len = len(kv_indices)
                page_aligned_kv_indices = kv_indices.to(dtype=torch.int64, copy=True)

            assert (
                cache_len == page_aligned_len
            ), f"It is required {cache_len=}, {page_aligned_len=}, {kv_committed_len=}, {len(req.origin_input_ids)=}, {len(req.output_ids)=} ping @yizhang2077 if you see this"

            # Radix Cache takes one ref in memory pool
            # insert the token_ids and kv_indices into the radix tree
            if self.enable_mamba_extra_buffer:
                mamba_ping_pong_track_buffer_to_keep = (
                    self.req_to_token_pool.get_mamba_ping_pong_other_idx(
                        req.mamba_next_track_idx
                    )
                )
                mamba_value = (
                    req.mamba_ping_pong_track_buffer[
                        mamba_ping_pong_track_buffer_to_keep
                    ]
                    .unsqueeze(-1)
                    .clone()
                )
            else:
                # Defensive validation backstop (eval_results_36): refuse
                # to launder a stale `req.mamba_pool_idx` into the tree.
                # See `_warn_stale_mamba_pool_idx` for diagnostic output.
                if not self._is_mamba_pool_idx_valid(req):
                    self._warn_stale_mamba_pool_idx(req, "cache_finished_req")
                    self._abort_cache_finished_for_stale_mamba(
                        req, page_aligned_kv_indices
                    )
                    self.dec_lock_ref(req.last_node)
                    return
                mamba_value = req.mamba_pool_idx.unsqueeze(-1).clone()
                mamba_ping_pong_track_buffer_to_keep = None

            # Defensive validation backstop (eval_results_44): refuse to
            # launder already-freed full-pool slot ids into the tree.
            if not self._are_kv_indices_valid(page_aligned_kv_indices):
                self._warn_stale_kv_indices(
                    req, "cache_finished_req", page_aligned_kv_indices
                )
                self._abort_cache_finished_for_stale_kv(
                    req, page_aligned_kv_indices
                )
                self.dec_lock_ref(req.last_node)
                return

            result = self.insert(
                InsertParams(
                    key=RadixKey(token_ids[:page_aligned_len], req.extra_key),
                    value=page_aligned_kv_indices,
                    mamba_value=mamba_value,
                    prev_prefix_len=req.cache_protected_len,
                )
            )
            mamba_exist = result.mamba_exist
        else:
            self.token_to_kv_pool_allocator.free(kv_indices[req.cache_protected_len :])
            mamba_exist = True

        if mamba_exist:
            mamba_ping_pong_track_buffer_to_keep = None

        free_mamba_cache = True if self.enable_mamba_extra_buffer else mamba_exist

        if free_mamba_cache:
            self.req_to_token_pool.free_mamba_cache(
                req,
                mamba_ping_pong_track_buffer_to_keep=mamba_ping_pong_track_buffer_to_keep,
            )
        elif is_insert and req.mamba_pool_idx is not None:
            # mamba_exist=False on the insert path: `_set_node_mamba_value`
            # just bound `req.mamba_pool_idx`'s slot to a new (or revived
            # tombstone) tree node. Slot ownership has transferred to the
            # tree, but the alloc-time binder entries (py_attr on
            # `req.mamba_pool_idx`, aux on `mapping[req_pool_idx]`) still
            # point at the slot. If left, a future tree-evict frees the
            # slot and the catch-all classifies those bookkeeping entries
            # as a forgotten live reference, poisoning the holders with -1
            # — which then laundries back into a fresh `node.mamba_value`
            # via line 599 and crashes on a later `mamba_pool.free([-1])`
            # (eval_results_1 cascade). Hand off ownership cleanly: drop
            # the req's binder entries and clear its dangling handle, but
            # keep the slot allocated for the tree.
            self.req_to_token_pool.transfer_mamba_to_radix(req)

        self.dec_lock_ref(req.last_node)

    def cache_unfinished_req(self, req: Req, chunked=False) -> None:
        """Cache request when it is unfinished."""

        def _skip_cache_unfinished_req(req: Req) -> None:
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, : len(req.fill_ids)
            ]
            # `kv_indices` is a live view of `req_to_token` (always current).
            # The int64 snapshot is taken on the same line that hands it to
            # `_set_req_prefix_indices`, which immediately binds every cell via
            # `bind_aux` — no unbound window for a relocation to slip through.
            # `req.prefix_indices` will be used in `PrefillAdder::add_chunked_req` later.
            self._set_req_prefix_indices(
                req, kv_indices.to(dtype=torch.int64, copy=True)
            )
            return

        token_ids = req.fill_ids
        cache_len = (
            req.mamba_last_track_seqlen
            if self.enable_mamba_extra_buffer
            else len(token_ids)
        )
        if self.disable or cache_len is None:
            return _skip_cache_unfinished_req(req)

        # `kv_indices` is a live VIEW of `req_to_token` — it tracks
        # eager-compaction relocations automatically because `req_to_token`
        # is a bound container. We deliberately do NOT materialize the int64
        # copy here: `mamba_pool.fork_from` / `self.evict(...)` below can free
        # full-pool slots and relocate this request's near-boundary slots,
        # and an int64 copy made now would NOT be updated by `Relocator.apply`
        # until `insert()` binds it as aux — exactly the laundering bug
        # eval_results_44/45 surfaced (`VALUE-FREED-SLOT`). Take the copy LATE,
        # after `fork_from`/`evict`, when `req_to_token` has absorbed any
        # relocations.
        kv_indices_orig = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]
        kv_indices = kv_indices_orig[:cache_len]
        if self.page_size != 1:
            page_aligned_len = len(kv_indices) // self.page_size * self.page_size
        else:
            page_aligned_len = len(kv_indices)

        assert page_aligned_len == len(
            kv_indices
        ), f"page_aligned_len != len(kv_indices), {page_aligned_len=}, {len(kv_indices)=}, {cache_len=}, {self.page_size=}, {FLA_CHUNK_SIZE=}"

        page_aligned_token_ids = token_ids[:page_aligned_len]

        if self.enable_mamba_extra_buffer:
            # copy from the ping pong track buffer
            mamba_ping_pong_track_buffer_to_keep = (
                self.req_to_token_pool.get_mamba_ping_pong_other_idx(
                    req.mamba_next_track_idx
                )
            )
            mamba_value = (
                req.mamba_ping_pong_track_buffer[mamba_ping_pong_track_buffer_to_keep]
                .unsqueeze(-1)
                .clone()
            )
        else:
            # Defensive validation backstop (eval_results_36): if
            # `req.mamba_pool_idx` is stale, the parallel mapping cell
            # `mapping[req.req_pool_idx]` is most likely stale too. Reading
            # it into `mamba_value` and `fork_from`-ing produces a fresh
            # forked slot via a corrupted CUDA copy_from(stale_row) — the
            # forked slot is structurally valid but carries garbage state.
            # That isn't a crash on its own, but it pollutes the radix tree
            # with semantically-wrong cached states. Skip the cache in
            # this case (and log diagnostic so the upstream write site can
            # be traced — same warning helper / cap as cache_finished_req).
            if not self._is_mamba_pool_idx_valid(req):
                self._warn_stale_mamba_pool_idx(req, "cache_unfinished_req")
                return _skip_cache_unfinished_req(req)
            mamba_value = self.req_to_token_pool.get_mamba_indices(
                req.req_pool_idx
            ).unsqueeze(-1)
        # radix tree mamba value is forked from req space
        mamba_value_forked = self.req_to_token_pool.mamba_pool.fork_from(mamba_value)

        # if alloc mamba cache failed, do evict and alloc again
        if mamba_value_forked is None:
            self.evict(EvictParams(num_tokens=0, mamba_num=1))
            mamba_value_forked = self.req_to_token_pool.mamba_pool.fork_from(
                mamba_value
            )
            assert mamba_value_forked is not None, "Can not alloc mamba cache"

        # NOW take the int64 snapshot — after `fork_from`/`evict`, `kv_indices`
        # (a view of `req_to_token`) reflects any relocations they triggered,
        # so the snapshot is fresh. `insert()` will bind it as aux for the
        # `_insert_helper` walk; the `free_group` wrapper below defers the
        # walk's own `_free_dedup_slots` compaction until every bound holder
        # (new node value, rewritten `req_to_token`, `req.prefix_indices`) is
        # in place — so the snapshot can never go stale while it's live.
        page_aligned_kv_indices = kv_indices[:page_aligned_len].to(
            dtype=torch.int64, copy=True
        )
        # Defensive validation backstop (eval_results_44/45): even with the
        # late snapshot, refuse to launder an already-freed full-pool slot id
        # into the tree. Should essentially never fire now.
        if not self._are_kv_indices_valid(page_aligned_kv_indices):
            self._warn_stale_kv_indices(
                req, "cache_unfinished_req", page_aligned_kv_indices
            )
            # Don't leak the mamba slot we just forked.
            self.req_to_token_pool.mamba_pool.free(mamba_value_forked)
            return _skip_cache_unfinished_req(req)

        # Defer eager compaction triggered inside `insert` (via
        # `_free_dedup_slots`) until after the new tree node is bound and
        # `req_to_token` / `req.prefix_indices` are rewritten. Only the
        # OUTERMOST opener manages the group — the decode result path already
        # wraps `release_kv_cache` in one, and nesting `free_group_begin()`
        # would reset `free_group = []` and lose the outer batch.
        alloc = self.token_to_kv_pool_allocator
        own_free_group = bool(getattr(alloc, "is_not_in_free_group", True))
        if own_free_group:
            alloc.free_group_begin()
        try:
            result = self.insert(
                InsertParams(
                    key=RadixKey(page_aligned_token_ids, req.extra_key),
                    value=page_aligned_kv_indices,
                    mamba_value=mamba_value_forked,
                    prev_prefix_len=req.cache_protected_len,
                )
            )
            new_prefix_len, mamba_exist = result.prefix_len, result.mamba_exist
            # there is a mamba cache in radix cache, release it
            if mamba_exist:
                self.req_to_token_pool.mamba_pool.free(mamba_value_forked)

            # The prefix indices could be updated, reuse it
            match_result = self.match_prefix(
                MatchPrefixParams(key=RadixKey(page_aligned_token_ids, req.extra_key))
            )
            new_indices, new_last_node = (
                match_result.device_indices,
                match_result.last_device_node,
            )

            if not mamba_exist:
                assert torch.equal(new_last_node.mamba_value, mamba_value_forked)

            assert (
                req.cache_protected_len <= len(new_indices) + self.page_size - 1
            ), f"{req.cache_protected_len=}, {len(new_indices)=}, {len(page_aligned_token_ids)=}, {mamba_exist=}"
            assert new_prefix_len <= len(
                new_indices
            ), f"{new_prefix_len=}, {len(new_indices)=}"

            self.req_to_token_pool.write(
                (req.req_pool_idx, slice(req.cache_protected_len, len(new_indices))),
                new_indices[req.cache_protected_len :],
            )

            self.dec_lock_ref(req.last_node)
            self.inc_lock_ref(new_last_node)

            # `req.prefix_indices` will be used in `PrefillAdder::add_chunked_req` later
            # NOTE: this is needed for both page_size == 1 and page_size > 1
            self._set_req_prefix_indices(
                req, torch.cat([new_indices, kv_indices_orig[len(new_indices) :]])
            )
            req.cache_protected_len = len(new_indices)
            req.mamba_last_track_seqlen = None
            req.last_node = new_last_node
        finally:
            if own_free_group:
                alloc.free_group_end()

    def pretty_print(self) -> None:
        self._print_helper(self.root_node, 0)
        total_size, total_mamba_size = self._total_size_helper()
        print(f"#full_tokens: {total_size}, #mamba_num: {total_mamba_size}")

    def total_size(self) -> Tuple[int, int]:
        return self._total_size_helper()

    def _evict_leaf_node(
        self, x: TreeNode, is_evict_mamba: bool
    ) -> Tuple[int, int, TreeNode, TreeNode]:
        assert (
            x.full_lock_ref == 0 and x.mamba_lock_ref == 0
        ), f"evict leaf node invalid with {x.id=} {x.full_lock_ref=} {x.mamba_lock_ref=}"

        assert x.mamba_value is not None, f"leaf node mamba value is not None, {x.id=}"
        # 1. a leaf node, free full tokens and mamba
        # Match-aware unbind + free: pass only the slots that the binder
        # agrees `x` owns to the allocator's `free`. Slots in `x.value`
        # that are bound to OTHER tree nodes (stale-tensor situation)
        # are NOT freed — that prevents the allocator from advancing its
        # watermark past slots still bound to other nodes (eval_results_42
        # multi-col cascade root cause).
        full_num_evicted = self._unbind_and_free_node_value(x)
        mamba_num_evicted = self._unbind_and_free_node_mamba_value(x)

        # 2. get the next node, update the lru lists
        if is_evict_mamba:
            x_next = self.mamba_lru_list.get_prev_no_lock(x)
        else:
            x_next = self.full_lru_list.get_prev_leaf_no_lock(x)
        self.full_lru_list.remove_node(x)
        self.mamba_lru_list.remove_node(x)

        # 3. delete the leaf node
        self._delete_leaf(x)

        # 4. Iteratively delete tombstone leaves to maintain invariant that leaf nodes are not tombstone
        x, leaf_full_num_evicted = self._iteratively_delete_tombstone_leaf(x)
        full_num_evicted += leaf_full_num_evicted
        return full_num_evicted, mamba_num_evicted, x, x_next

    def evict(self, params: EvictParams) -> EvictResult:
        if self.disable:
            return EvictResult()

        full_num_evicted = 0
        mamba_num_evicted = 0

        if params.num_tokens > 0:
            full_num_evicted = self.evict_full(params.num_tokens)
        if params.mamba_num > 0:
            mamba_num_evicted = self.evict_mamba(params.mamba_num)

        return EvictResult(
            num_tokens_evicted=full_num_evicted, mamba_num_evicted=mamba_num_evicted
        )

    def evict_mamba(self, mamba_num: int) -> int:
        """Evict mamba states. Returns the number of mamba states evicted."""
        if self.disable or mamba_num <= 0:
            return 0
        # get the least recently used node that is not locked, doesn't have to be a leaf
        x = self.mamba_lru_list.get_lru_no_lock()
        mamba_num_evicted = 0
        # evict lru leaf nodes until mamba_num_tokens is reached
        while mamba_num_evicted < mamba_num and (self.mamba_lru_list.in_list(x)):
            assert x.mamba_value is not None, f"node has no mamba value, {x.id=}"
            assert (
                len(x.mamba_value) == 1
            ), f"node has abnormal mamba length, {x.id=}, {len(x.mamba_value)=}"
            assert x != self.root_node, f"root node is not evictable, {x.id=}"
            assert x.mamba_lock_ref == 0, f"node is in use by mamba kv indices, {x.id=}"

            if len(x.children) > 0:
                # 1. an internal node, free mamba tokens.
                # Match-aware unbind + free (eval_results_42): only
                # frees slots actually owned by x per the binder.
                mamba_num_evicted += self._unbind_and_free_node_mamba_value(x)

                # 2. get the next node, update the lru lists
                x_next = self.mamba_lru_list.get_prev_no_lock(x)
                self.mamba_lru_list.remove_node(x)

                # 3. tombstone the node
                self._tombstone_internal_node(x)
            else:
                _, mamba_evicted_delta, _, x_next = self._evict_leaf_node(x, True)
                mamba_num_evicted += mamba_evicted_delta

            x = x_next

        return mamba_num_evicted

    def evict_full(self, full_num_tokens: int) -> int:
        """Evict full KV cache. Returns the number of tokens evicted."""
        if self.disable or full_num_tokens <= 0:
            return 0

        full_num_evicted = 0
        # get the least recently used leaf node that is not locked
        x = self.full_lru_list.get_leaf_lru_no_lock()

        while full_num_evicted < full_num_tokens and self.full_lru_list.in_list(x):
            assert (
                x != self.root_node
            ), f"root node should not exist in full lru list, {x.id=}"
            full_num_evicted_delta, _, x, x_next = self._evict_leaf_node(x, False)
            full_num_evicted += full_num_evicted_delta

            # if parent has no more children, it is a leaf. It is possible that this node is lru, so
            # we need to get the first leaf node in the lru list
            if len(x.parent.children) == 0:
                x_next = self.full_lru_list.get_leaf_lru_no_lock()

            x = x_next

        return full_num_evicted

    def inc_lock_ref(self, node: TreeNode) -> IncLockRefResult:
        """
        Increment the lock reference count for the node.
        It locks the full_lock_ref for nodes between the [last node, root), exclusive.
        It locks the mamba_lock_ref for current node if its mamba_value exists.
        """
        if self.disable:
            return IncLockRefResult()

        # protect mamba value in current node if it exists
        if node.mamba_value is not None:
            if node.mamba_lock_ref == 0:
                self.mamba_evictable_size_ -= len(node.mamba_value)
                self.mamba_protected_size_ += len(node.mamba_value)
            node.mamba_lock_ref += 1

        while node != self.root_node:
            # lock full from node to root
            assert (
                node.full_lock_ref >= 0
            ), f"inc_lock_ref on node with {node.full_lock_ref=}, {node.id=}"
            if node.full_lock_ref == 0:
                self.full_evictable_size_ -= len(node.value)
                self.full_protected_size_ += len(node.value)
            node.full_lock_ref += 1
            node = node.parent
        return IncLockRefResult()

    def dec_lock_ref(
        self, node: TreeNode, params: Optional[DecLockRefParams] = None
    ) -> DecLockRefResult:
        """
        Decrement the lock reference count for the node.
        It unlocks the full_lock_ref for nodes between the [last node, root), exclusive.
        It unlocks the mamba_lock_ref for current node if its mamba_value exists.
        """
        if self.disable:
            return DecLockRefResult()

        if node.mamba_value is not None:
            assert (
                node.mamba_lock_ref > 0
            ), f"dec_lock_ref on node with {node.mamba_lock_ref=}, {node.id=}"
            if node.mamba_lock_ref == 1:
                self.mamba_evictable_size_ += len(node.mamba_value)
                self.mamba_protected_size_ -= len(node.mamba_value)
            node.mamba_lock_ref -= 1

        while node != self.root_node:
            assert (
                node.full_lock_ref > 0
            ), f"dec_lock_ref on node with {node.full_lock_ref=}, {node.id=}"
            if node.full_lock_ref == 1:
                self.full_evictable_size_ += len(node.value)
                self.full_protected_size_ -= len(node.value)
            node.full_lock_ref -= 1
            node = node.parent

        return DecLockRefResult()

    def sanity_check(self):
        if self.disable:
            return
        self.full_lru_list.sanity_check(self)
        self.mamba_lru_list.sanity_check(self)

    def evictable_size(self) -> Tuple[int, int]:
        # Note: use full_evictable_size() and mamba_evictable_size() instead.
        raise NotImplementedError

    def full_evictable_size(self) -> int:
        return self.full_evictable_size_

    def mamba_evictable_size(self) -> int:
        return self.mamba_evictable_size_

    def protected_size(self) -> Tuple[int, int]:
        # Note: use full_protected_size() and mamba_protected_size() instead.
        raise NotImplementedError

    def full_protected_size(self) -> int:
        # protected size refers to the size of the full cache that is locked
        return self.full_protected_size_

    def mamba_protected_size(self) -> int:
        # protected size refers to the size of the mamba cache that is locked
        return self.mamba_protected_size_

    def all_values_flatten(self) -> torch.Tensor:
        values = []

        def _dfs_helper(node: TreeNode):
            for _, child in node.children.items():
                values.append(child.value)
                _dfs_helper(child)

        _dfs_helper(self.root_node)
        return torch.cat(values) if len(values) > 0 else torch.tensor([])

    def all_mamba_values_flatten(self) -> torch.Tensor:
        values = []

        def _dfs_helper(node: TreeNode):
            if node.mamba_value is not None:
                values.append(node.mamba_value)
            for _, child in node.children.items():
                _dfs_helper(child)

        _dfs_helper(self.root_node)
        return torch.cat(values) if len(values) > 0 else torch.tensor([])

    def available_and_evictable_str(self) -> str:
        full_available_size = self.token_to_kv_pool_allocator.available_size()
        full_evictable_size = self.full_evictable_size()
        return (
            f"Available full tokens: {full_available_size + full_evictable_size} ({full_available_size=} + {full_evictable_size=})\n"
            f"Full LRU list evictable size: {self.full_lru_list.sanity_check_evictable_size()}\n"
        )

    ##### Internal Helper Functions #####

    def _match_prefix_helper(
        self, key: RadixKey
    ) -> Tuple[List[torch.Tensor], TreeNode, int]:
        """
        Mamba prefix matching helper. It factors in the sliding window size such that
        the matched node is guaranteed to either 1. connected to root without mamba tombstone,
        or 2. the number of matching tokens from the matched node to the last mamba tombstone
        node is greater than or equal to the sliding window size.
        """
        node = self.root_node
        child_key = self.get_child_key_fn(key)

        value: List[torch.Tensor] = []
        best_value_len = 0
        best_last_node = node
        while len(key) > 0 and child_key in node.children.keys():
            child = node.children[child_key]
            # update best_value_len and best_last_node if needed
            if node.mamba_value is not None:
                best_value_len = len(value)
                best_last_node = node

            prefix_len = self.key_match_fn(child.key, key)
            if prefix_len < len(child.key):
                new_node = self._split_node(child.key, child, prefix_len)
                value.append(new_node.value)
                node = new_node
                break
            else:
                value.append(child.value)
                node = child
                key = key[prefix_len:]

                if len(key):
                    child_key = self.get_child_key_fn(key)
        # handle best_value_len and best_last_node, for the case that last node is fully matched
        if node.mamba_value is not None:
            best_value_len = len(value)
            best_last_node = node

        return value, best_last_node, best_value_len

    def _match_pre_processor(self, params: MatchPrefixParams) -> Optional[RadixKey]:
        """Preprocess the key before matching."""
        key = params.key

        if self.disable or len(key) == 0:
            return None

        return key

    def _match_post_processor(
        self,
        params: MatchPrefixParams,
        value: List[torch.Tensor],
        last_node: TreeNode,
        best_value_len: int,
    ) -> MatchResult:
        """Post-process the matched result."""
        cow_mamba = params.cow_mamba
        req = params.req

        # update time for matched nodes, and make nodes closer to root to be least recently used
        # this allows mamba to evict nodes closer to root first
        node_update = last_node
        self.full_lru_list.reset_node_and_parents_mru(node_update, self.root_node)
        self.mamba_lru_list.reset_node_and_parents_mru(node_update, self.root_node)

        # This last_access_time is for sanity check, can be deleted after validation in production
        cur_time = get_last_access_time()
        while node_update:
            node_update.last_access_time = cur_time
            cur_time -= (
                0.00001  # assuming less than 100000 nodes in a branch of the tree
            )
            node_update = node_update.parent

        # Calculate the branching point. It is defined as the last aligned position that
        # does not have a mamba value.
        if len(value) > best_value_len:
            mamba_cache_chunk_size = get_global_server_args().mamba_cache_chunk_size
            mamba_cache_chunk_aligned_seqlen = (
                sum(len(v) for v in value) // mamba_cache_chunk_size
            ) * mamba_cache_chunk_size
            mamba_branching_seqlen = (
                mamba_cache_chunk_aligned_seqlen
                if mamba_cache_chunk_aligned_seqlen > 0
                else None
            )
        else:
            mamba_branching_seqlen = None

        # Copy mamba state to req local space if cow is true
        if cow_mamba and last_node.mamba_value is not None:
            # for reqs without mamba cache
            if req.mamba_pool_idx is None:
                dst_index = self.req_to_token_pool.mamba_pool.alloc(1)
                # try to alloc again, protect last_node from eviction
                if dst_index is None:
                    self.inc_lock_ref(last_node)
                    self.evict(EvictParams(num_tokens=0, mamba_num=1))
                    dst_index = self.req_to_token_pool.mamba_pool.alloc(1)
                    self.dec_lock_ref(last_node)
                    assert dst_index is not None, "Can not alloc mamba cache"
                src_index = last_node.mamba_value
                self.req_to_token_pool.mamba_pool.copy_from(src_index, dst_index)
                self._set_req_mamba_pool_idx(req, dst_index[0])
            else:
                src_index = last_node.mamba_value
                dst_index = req.mamba_pool_idx.unsqueeze(0)
                self.req_to_token_pool.mamba_pool.copy_from(src_index, dst_index)

        value = value[:best_value_len]
        if value:
            value = torch.cat(value)
        else:
            value = torch.empty((0,), dtype=torch.int64, device=self.device)

        return MatchResult(
            device_indices=value,
            last_device_node=last_node,
            last_host_node=last_node,
            mamba_branching_seqlen=mamba_branching_seqlen,
        )

    def _split_node(self, key: RadixKey, child: TreeNode, split_len: int) -> TreeNode:
        # new_node -> child
        new_node = TreeNode()
        new_node.children = {self.get_child_key_fn(key[split_len:]): child}
        new_node.parent = child.parent
        # mamba cache can not be split — new_node carries no mamba state.
        # No prior binding to unbind here (new_node is freshly constructed).
        new_node.mamba_value = None
        new_node.full_lock_ref = child.full_lock_ref
        new_node.mamba_lock_ref = 0
        new_node.key = child.key[:split_len]
        self._unbind_all_node_value(child)
        new_node.value = child.value[:split_len].clone()

        # child time should be later than parent's time for mamba tombstone
        child.last_access_time = get_last_access_time()

        self.full_lru_list.remove_node(child)
        if child.mamba_value is not None:
            self.mamba_lru_list.remove_node(child)
        child.parent = new_node
        child.key = child.key[split_len:]
        child.value = child.value[split_len:].clone()
        self._bind_node_value(new_node)
        self._bind_node_value(child)
        new_node.parent.children[self.get_child_key_fn(key)] = new_node

        # insert the new node and child into the lru lists, insert
        # parent first so that parent is after child in the lru list
        self.full_lru_list.insert_mru(new_node)
        self.full_lru_list.insert_mru(child)
        if child.mamba_value is not None:
            self.mamba_lru_list.insert_mru(child)
        return new_node

    def _insert_helper(
        self,
        node: TreeNode,
        key: RadixKey,
        value,
        mamba_value,
        chunked: bool = False,
        prev_prefix_len: int = 0,
    ) -> Tuple[int, bool]:
        # Update the last access time from root to leaf, so that
        # mamba will tombstone the node closer to root first
        assert mamba_value is not None, "Mamba value should not be None here."
        node.last_access_time = get_last_access_time()
        if node != self.root_node:
            self.full_lru_list.reset_node_mru(node)
            if node.mamba_value is not None:
                self.mamba_lru_list.reset_node_mru(node)
        if len(key) == 0:
            return 0, True

        child_key = self.get_child_key_fn(key)

        total_prefix_length = 0
        while len(key) > 0 and child_key in node.children.keys():
            node = node.children[child_key]
            node.last_access_time = get_last_access_time()
            self.full_lru_list.reset_node_mru(node)
            if node.mamba_value is not None:
                self.mamba_lru_list.reset_node_mru(node)
            prefix_len = self.key_match_fn(node.key, key)

            if prev_prefix_len < total_prefix_length + prefix_len:
                start = max(0, prev_prefix_len - total_prefix_length)
                # Use `_free_dedup_slots`: skip slots that match `node.value`
                # (cache-hit reuse of the walker's current node) AND skip
                # slots bound to another live tree node (freeing those would
                # leak the OTHER tree's slot — the silent-drift bug surfaced
                # in eval_results_17 with the smoking-gun warning at
                # multi_ended_allocator.py:1260, attr='value', cum=114K).
                # Stale binder entries (bound node detached but
                # Python-alive) are cleaned up and the slot is freed.
                self._free_dedup_slots(
                    value[start:prefix_len], node.value[start:prefix_len]
                )

            total_prefix_length += prefix_len
            key = key[prefix_len:]
            value = value[prefix_len:]

            if prefix_len < len(node.key):
                new_node = self._split_node(node.key, node, prefix_len)
                node = new_node

            if len(key):
                child_key = self.get_child_key_fn(key)

        mamba_value_exist = False
        if len(key):
            new_node = TreeNode()
            new_node.parent = node
            new_node.key = key
            self._set_node_value(new_node, value.clone())
            self._set_node_mamba_value(new_node, mamba_value)
            self.full_lru_list.insert_mru(new_node)
            self.mamba_lru_list.insert_mru(new_node)
            node.children[child_key] = new_node
            self.full_evictable_size_ += len(value)
            self.mamba_evictable_size_ += len(mamba_value)
        elif node.mamba_value is None:  # add for mamba tombstone
            self._set_node_mamba_value(node, mamba_value)
            self.full_lru_list.reset_node_mru(node)
            self.mamba_lru_list.insert_mru(node)
            self.mamba_evictable_size_ += len(mamba_value)
            node.last_access_time = get_last_access_time()
        else:  # mamba value already exists
            mamba_value_exist = True
            self.full_lru_list.reset_node_mru(node)
            self.mamba_lru_list.reset_node_mru(node)
            node.last_access_time = get_last_access_time()

        return total_prefix_length, mamba_value_exist

    def _iteratively_delete_tombstone_leaf(
        self, node: TreeNode
    ) -> Tuple[TreeNode, int]:
        full_num_evicted = 0
        while node.parent.mamba_value is None and len(node.parent.children) == 0:
            # root node is not evictable
            if node.parent == self.root_node:
                break
            # if locked, means node is in use, skip
            if node.parent.full_lock_ref > 0:
                break
            assert (
                node.parent.mamba_lock_ref == 0
            ), f"tombstone mamba_lock_ref should always be 0, {node.parent.full_lock_ref=}, {node.parent.mamba_lock_ref=}, {node.parent.id=}"
            # delete tombstone node evicts full tokens.
            # Match-aware unbind + free (eval_results_42): only frees
            # slots that the binder agrees `node.parent` owns; stale
            # entries in `node.parent.value` are left to their rightful
            # owner's eviction. Without this filter, the allocator's
            # watermark advances past slots still bound in the binder
            # to OTHER tree nodes, producing the multi-col cascade.
            full_num_evicted += self._unbind_and_free_node_value(node.parent)
            self.full_lru_list.remove_node(node.parent)
            self._delete_tombstone_leaf(node.parent)
            node = node.parent

        return node, full_num_evicted

    def _delete_leaf(self, node: TreeNode) -> None:
        assert (
            node.mamba_value is not None
        ), f"Invariant violated: leaf node is a tombstone, {node.id=}"
        assert len(node.children) == 0, f"leaf node has children, {node.id=}"
        key = self.get_child_key_fn(node.key)
        v = node.parent.children.pop(key, None)
        assert v == node, f"parent does not have child key, {key}"

        self.full_evictable_size_ -= len(node.key)
        self.mamba_evictable_size_ -= len(node.mamba_value)

    def _tombstone_internal_node(self, node: TreeNode) -> None:
        assert len(node.children) != 0, f"Cannot tombstone a leaf node, {node.id=}"
        self.mamba_evictable_size_ -= len(node.mamba_value)
        # NOTE: the binder unbind has ALREADY been done by the caller —
        # `evict_mamba` INTERNAL branch (mamba_radix_cache.py:989) calls
        # `_unbind_node_mamba_value(x)` BEFORE `mamba_pool.free(...)`.
        # Calling `_unbind_node_mamba_value(node)` here would be:
        #   (a) redundant (the binder entry was already popped), and
        #   (b) PREVIOUSLY, harmful: between the caller's unbind and
        #       this point, `mamba_pool.free`'s eager-compaction
        #       (`_apply_one` step 6) may have rebound the slot id in
        #       `node.mamba_value` to a DIFFERENT tree node. A naive
        #       second unbind would silently pop that other node's
        #       binding (eval_results_40 cascade). The match-aware
        #       `_safe_unbind_tree_node` now skips this case, so even
        #       if a future caller path forgets to unbind first, this
        #       function would no-op cleanly. But removing the call
        #       outright avoids the wasted scan and keeps responsibility
        #       with the caller.
        node.mamba_value = None

    def _delete_tombstone_leaf(self, node: TreeNode) -> None:
        assert (
            node.mamba_value is None
        ), f"Deleting a unexpected non-tombstone leaf node, {node.id=}"
        assert len(node.children) == 0, f"leaf node has children, {node.id=}"
        key = self.get_child_key_fn(node.key)
        v = node.parent.children.pop(key, None)
        assert v == node, f"parent does not have child key, {key}"

        self.full_evictable_size_ -= len(node.key)

    def _collect_nontombstone_nodes(self) -> List[TreeNode]:
        ret_list = []
        stack = [self.root_node]

        while stack:
            cur_node = stack.pop()
            if cur_node.mamba_value is not None:
                ret_list.append(cur_node)
            stack.extend(cur_node.children.values())

        return ret_list

    def _collect_all_nodes(self) -> List[TreeNode]:
        ret_list = []
        stack = [self.root_node]
        while stack:
            cur_node = stack.pop()
            ret_list.append(cur_node)
            stack.extend(cur_node.children.values())
        return ret_list

    def _print_helper(self, node: TreeNode, indent: int) -> None:
        """Prints the radix tree in a human-readable format."""
        stack = [(node, indent)]
        while stack:
            current_node, current_indent = stack.pop()
            print(
                " " * current_indent,
                f"[{current_node.id}]",
                len(current_node.key),
                f"fr={current_node.full_lock_ref}",
                f"mr={current_node.mamba_lock_ref}",
                f"fll={self.full_lru_list.in_list(current_node)}",
                f"mll={self.mamba_lru_list.in_list(current_node)}",
                f"mv={current_node.mamba_value}",
            )
            for key, child in current_node.children.items():
                stack.append((child, current_indent + 2))

                assert key == self.get_child_key_fn(
                    child.key
                ), f"{key=}, {self.get_child_key_fn(child.key)=}"

    def _total_size_helper(self) -> Tuple[int, int]:
        total_size = 0
        total_mamba_size = 0
        stack = [self.root_node]
        while stack:
            current_node = stack.pop()
            total_size += len(current_node.value)
            if current_node.mamba_value is not None:
                total_mamba_size += len(current_node.mamba_value)
            for child in current_node.children.values():
                if child.evicted:
                    continue
                stack.append(child)
        return total_size, total_mamba_size
