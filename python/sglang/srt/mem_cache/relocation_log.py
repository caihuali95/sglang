"""
Relocator + SlotBacktrack + SlotBacktrackBinder: bookkeeping for slot
relocations produced by the MultiEndedAllocator's eager compaction.

When an allocator's ``free`` compacts a boundary slot ``b`` into a freed
hole ``s``, every reference to ``b`` (which now names the same physical
data that lives at ``s``) must be rewritten to ``s`` so that subsequent
reads — be they for the next forward, the next ``free`` chunk, or any
caller path that captured the slot id — see the relocated address.

Synchronous design:

  * ``SlotBacktrack`` is the reverse index from a slot id to its
    references: at most one TreeNode, a ``_ReqPosRef`` carrying both the
    ``req_to_token`` column AND the set of rows that wrote to that
    column, any number of aux tensor cells (e.g., the SWA mapping), and
    any CPU-side ``py_attr`` bindings. Updated on every slot-id *write*
    in the rest of the codebase; consulted at relocation time.

  * ``Relocator.apply(sub_pool, src, dst)`` rewrites every reference
    holder of ``src`` to point at ``dst`` immediately, before returning
    to the caller. There is no deferred ``_pending`` log, no end-of-tick
    flush, no chain-chasing.

  * On ``free``, ``SlotBacktrack.clear_slot(s)`` pops any stale binder
    entries for the freed slot ``s`` (defensive cleanup — the owners are
    supposed to have unbound first, but a missed unbind would otherwise
    leave a stale entry that the next allocation of ``s`` collides with).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import torch

logger = logging.getLogger(__name__)


@dataclass
class _TreeNodeRef:
    node: object  # TreeNode; typed as object to avoid a circular import
    position: int  # index into node.<attr> where this slot id lives
    attr: str = "value"  # attribute on `node` that holds the slot tensor.
    # Defaults to "value" (the canonical full-pool / SWA-pool radix tree value).
    # `MambaRadixCache.TreeNode.mamba_value` is bound with attr="mamba_value" so
    # eager-compaction relocations of mamba slots correctly update the cloned
    # tensor that the radix tree carries (otherwise that clone goes stale and a
    # later evict_mamba frees a slot id outside the watermark — see
    # docker/shared-pool-test eval logs from 2026-05-03).


@dataclass
class _AuxRef:
    tensor: torch.Tensor
    index: int  # 1-D index into tensor


@dataclass
class _ReqPosRef:
    """Reverse-index entry for a slot id present in `req_to_token`.

    `col` is the unique column the slot occupies (prefix-sharing invariant:
    if multiple reqs share a prefix containing this slot, they all hold it
    at the same column).

    `rows` is the set of `req_pool_idx` rows whose `req_to_token[row, col]`
    cell currently holds this slot. Tracked so relocations can rewrite
    only those rows (verified against current cell contents) instead of
    scanning the entire column.
    """

    col: int
    rows: Set[int] = field(default_factory=set)


class SlotBacktrackBinder:
    """Thin facade that routes slot-id writes into a SlotBacktrack.

    Every slot-id-writing call site in the codebase talks to a binder (not a
    SlotBacktrack directly). Static-partition pools expose a NullBacktrackBinder
    whose methods are no-ops, so the hook sites can unconditionally call
    `binder.bind_*` without isinstance checks.
    """

    def __init__(self, backtrack: "SlotBacktrack"):
        self._backtrack = backtrack

    def is_null(self) -> bool:
        return False

    def bind_tree_node(
        self, slot: int, node: object, position: int, attr: str = "value"
    ) -> None:
        if slot <= 0:  # slot 0 is reserved padding across all sub-pools
            return
        self._backtrack.bind_tree_node(int(slot), node, int(position), attr)

    def unbind_tree_node(self, slot: int) -> None:
        if slot <= 0:
            return
        self._backtrack.unbind_tree_node(int(slot))

    def diagnose_tree_node_for_unbind(
        self, slot: int, node: object, position: int, attr: str
    ):
        """Pre-flight diagnostic before `unbind_tree_node(slot)`. See
        :py:meth:`SlotBacktrack.diagnose_tree_node_for_unbind` for the
        full contract."""
        if slot <= 0:
            return ("unbound", None)
        return self._backtrack.diagnose_tree_node_for_unbind(
            int(slot), node, int(position), attr
        )

    def bind_req_position(self, slot: int, row: int, col: int) -> None:
        if slot <= 0:  # slot 0 is reserved padding across all sub-pools
            return
        self._backtrack.bind_req_position(int(slot), int(row), int(col))

    def unbind_req_position(self, slot: int, row: int) -> None:
        if slot <= 0:
            return
        self._backtrack.unbind_req_position(int(slot), int(row))

    def clear_slot_req_position(self, slot: int) -> None:
        """Catch-all clear of req_position for a slot (any rows)."""
        if slot <= 0:
            return
        self._backtrack.clear_slot_req_position(int(slot))

    def bind_aux(
        self, slot: int, tensor: torch.Tensor, index: int
    ) -> None:
        if slot <= 0:
            return
        self._backtrack.bind_aux(int(slot), tensor, int(index))

    def unbind_aux(
        self, slot: int, tensor: torch.Tensor, index: int
    ) -> None:
        if slot <= 0:
            return
        self._backtrack.unbind_aux(int(slot), tensor, int(index))

    def bind_py_attr(self, slot: int, obj, attr: str) -> None:
        if slot <= 0:
            return
        self._backtrack.bind_py_attr(int(slot), obj, attr)

    def unbind_py_attr(self, slot: int, obj, attr: str) -> None:
        if slot <= 0:
            return
        self._backtrack.unbind_py_attr(int(slot), obj, attr)


class NullBacktrackBinder(SlotBacktrackBinder):
    """No-op binder for pools running in static-partition mode."""

    def __init__(self):  # type: ignore[override]
        self._backtrack = None

    def is_null(self) -> bool:
        return True

    def bind_tree_node(self, slot, node, position, attr="value"):
        return

    def unbind_tree_node(self, slot):
        return

    def diagnose_tree_node_for_unbind(self, slot, node, position, attr):
        return ("unbound", None)

    def bind_req_position(self, slot, row, col):
        return

    def unbind_req_position(self, slot, row):
        return

    def clear_slot_req_position(self, slot):
        return

    def bind_aux(self, slot, tensor, index):
        return

    def unbind_aux(self, slot, tensor, index):
        return

    def bind_py_attr(self, slot, obj, attr):
        return

    def unbind_py_attr(self, slot, obj, attr):
        return


_NULL_BINDER_SINGLETON = NullBacktrackBinder()


def null_backtrack_binder() -> SlotBacktrackBinder:
    """Process-wide null binder singleton. Cheap to reference from every KVCache."""
    return _NULL_BINDER_SINGLETON


@dataclass
class SlotBacktrack:
    """
    Per-sub-pool reverse index from slot id to its references.

    A slot id may be referenced by:
      * at most one TreeNode (radix cache) at some position inside its value;
      * at most one column position `pos` in `req_to_token` (the prefix-sharing
        invariant — if multiple running requests share a common prefix that
        contains this slot, they all use the same `pos`);
      * any number of aux tensor cells (mapping tensors).

    Updated by the code sites that write slot ids; consumed by Relocator.apply.
    """

    size_total: int
    tree_node: Dict[int, _TreeNodeRef] = field(default_factory=dict)
    req_position: Dict[int, _ReqPosRef] = field(default_factory=dict)
    aux: Dict[int, List[_AuxRef]] = field(default_factory=dict)
    # (object, attribute_name) pairs for CPU-side scalar slot-id attrs
    # (e.g., Req.mamba_pool_idx). On flush we do `setattr(obj, attr, new_value)`
    # skipping any wrapping property setter via direct dict assignment.
    py_attr: Dict[int, List[tuple]] = field(default_factory=dict)
    # -- DEBUG: bind-overwrite detection.
    # Per-slot lifecycle ring buffer. Each event is a tuple
    #   (op, source_func, slot, detail)
    # where op ∈ {"bind_tree_node", "unbind_tree_node",
    #             "transfer_overwrite_dst", "release"}. We capture the
    # most recent N events PER SLOT so the stale-slot assertion can dump
    # the full history that led to the staleness.
    # Set `RELOCATION_LOG_DEBUG_HISTORY = 1` env var to enable; off by
    # default to keep the hot path overhead at zero.
    history: Dict[int, List[tuple]] = field(default_factory=dict)
    history_enabled: bool = False
    history_limit_per_slot: int = 16
    # Diagnostic counters retained from the deferred-flush era. The
    # `bind_tree_overwrite_count` still fires in `bind_tree_node` when a
    # 1:1 invariant violation is observed (same slot bound to two tree
    # nodes). The `bind_aux_dup_count` / `bind_py_attr_dup_count`
    # counters likewise still fire on duplicate bind hygiene errors.
    # The `bind_req_position` overwrite case is now an *assert* (the
    # data structure makes it unrepresentable), not a counter.
    bind_tree_overwrite_count: int = 0
    bind_aux_dup_count: int = 0
    bind_py_attr_dup_count: int = 0
    # Multi-owner counter: distinct from `bind_py_attr_dup_count` (which
    # only counts re-bind of the *same* `(obj, attr)` pair). This
    # increments when `bind_py_attr` is asked to add a *different* obj's
    # binding while the slot already has live entries from one or more
    # other objs — i.e., the single-owner invariant ("at any given time,
    # exactly one Req's `mamba_pool_idx` references slot S") was violated.
    # Empirically this is the leak that pollutes `req_index_to_mamba_index_mapping`
    # cells across multiple Reqs (eval_results_36 / 37 lifecycle dumps:
    # 4+ distinct obj_ids accumulating at one mamba slot without
    # intervening unbinds). Each increment emits a capped warning with
    # the prior obj's bind caller AND the new bind's caller, so the
    # leaking allocation path can be localized.
    bind_py_attr_multi_owner_count: int = 0

    # -- tree node --
    def bind_tree_node(
        self, slot: int, node: object, position: int, attr: str = "value"
    ) -> None:
        prior = self.tree_node.get(slot)
        if prior is not None and (
            prior.node is not node
            or prior.position != position
            or prior.attr != attr
        ):
            # OVERWRITE: existing tree_node binding for this slot would
            # be replaced. This is a **1:1 invariant violation** — two
            # tree nodes' value tensors both contain `slot`. The prior
            # node's binding was never unbound (either its
            # `_unbind_all_node_value` skipped this slot under a stale
            # value tensor, or the kv allocator handed `slot` out to two
            # Reqs simultaneously).
            #
            # Previously this branch silently replaced the prior entry
            # and only logged. That just propagated the corruption — the
            # OLD node's binding was lost without trace, and the next
            # apply / eviction of the old node hit the cascading
            # symptoms (eval_results_41 full-pool failures).
            #
            # New behavior: **abort the bind**. Preserve the prior entry,
            # log the warning, record the abort in lifecycle. The caller
            # (the new bind site) will not have its binding tracked — a
            # downstream symptom will appear at the new node, not at the
            # old. That symptom plus this warning's caller stack pinpoint
            # the upstream double-allocation source.
            self.bind_tree_overwrite_count += 1
            if self.bind_tree_overwrite_count <= 32:
                import inspect
                frames = inspect.stack()[2:10]
                callers = " <- ".join(
                    f"{f.filename.split('/')[-1]}:{f.lineno}" for f in frames
                )
                logger.warning(
                    "SlotBacktrack.bind_tree_node OVERWRITE ABORTED for slot %d: "
                    "prior=(node_id=%x, attr=%r, pos=%d), "
                    "new=(node_id=%x, attr=%r, pos=%d). "
                    "Abort preserves the prior entry (preventing silent "
                    "loss of the old node's binding); the new bind is "
                    "DROPPED. cum_overwrites=%d. Caller: %s",
                    slot,
                    id(prior.node),
                    prior.attr,
                    prior.position,
                    id(node),
                    attr,
                    position,
                    self.bind_tree_overwrite_count,
                    callers,
                )
            self._record_history(
                slot,
                ("bind_tree_node_overwrite_aborted", id(node), position, attr,
                 id(prior.node), prior.position, prior.attr),
            )
            # Abort: do NOT replace `self.tree_node[slot]`. Return early.
            return
        self._record_history(
            slot, ("bind_tree_node", id(node), position, attr)
        )
        self.tree_node[slot] = _TreeNodeRef(
            node=node, position=position, attr=attr
        )

    def unbind_tree_node(self, slot: int) -> None:
        if slot in self.tree_node:
            self._record_history(slot, ("unbind_tree_node",))
        self.tree_node.pop(slot, None)

    def diagnose_tree_node_for_unbind(
        self, slot: int, node: object, position: int, attr: str
    ):
        """Pre-flight diagnostic for an upcoming `unbind_tree_node(slot)`.

        Returns a (status, payload) tuple describing whether the binder
        state matches what the caller expects (i.e. that `slot/position/attr`
        is currently bound to `node`). Used by `_unbind_node_*_value`
        helpers to detect binder/tensor divergence — the case where the
        caller reads `node.<attr>[position] == slot` but the binder thinks
        `node` is bound at some OTHER slot (or nowhere). Such a divergence
        means `node.<attr>` was mutated in-place / replaced without going
        through `bind/unbind_tree_node`.

        Status values:
          - ``"ok"``: binder has tree_node(slot, node, position, attr) as
            expected. Unbind will work normally. Payload: None.
          - ``"divergent"``: binder has no entry at `slot` for this
            (node, attr, position), but DOES have `node` bound at one or
            more OTHER slots at the same (attr, position). Payload:
            ``List[int]`` of those other slots — the smoking gun for
            tracing where the in-place write to node.<attr> came from.
          - ``"wrong_node_at_slot"``: binder has an entry at `slot` but
            for a DIFFERENT node / position / attr. Payload: the
            `_TreeNodeRef` actually at `slot`.
          - ``"unbound"``: binder has no entry at `slot` AND `node` is
            not bound anywhere matching (attr, position). Payload: None.
            Legitimate (e.g. node was already explicitly unbound), no
            warning needed.

        Cost: O(|tree_node|) when a divergence/unbound case is hit (we
        scan the dict). The "ok" case is O(1). Callers should cap-gate
        the warnings to bound scan overhead under stress.
        """
        ref = self.tree_node.get(slot)
        if (
            ref is not None
            and ref.node is node
            and ref.position == position
            and ref.attr == attr
        ):
            return ("ok", None)
        divergent: List[int] = []
        for s, r in self.tree_node.items():
            if r.node is node and r.attr == attr and r.position == position:
                divergent.append(s)
        if divergent:
            return ("divergent", divergent)
        if ref is not None:
            return ("wrong_node_at_slot", ref)
        return ("unbound", None)

    def clear_slot(self, slot: int) -> bool:
        """Drop EVERY binding for ``slot`` — tree_node, req_position, aux,
        py_attr. Returns True iff any binding was actually present.

        This is the catch-all "slot is going back to the free list, nothing
        can possibly hold a live reference to its bytes anymore" hook. It is
        called from :py:meth:`MultiEndedAllocator._free_eager` for every
        slot in the free batch, regardless of whether the caller path did
        its own unbind. Goal: eliminate the class of bugs where one of the
        binding kinds is left dangling on a freed slot — most prominently
        ``req_position`` (which no production caller ever unbinds), but
        also ``aux`` / ``py_attr`` on paths like ``Req.reset_for_retract``
        that bypass the proper helpers.

        Records a single ``("clear_on_free", (kinds,))`` event in the
        per-slot history (when history is enabled) so the lifecycle dump
        shows the moment the slot was zeroed out.
        """
        cleared: List[str] = []
        if slot in self.tree_node:
            cleared.append("tree_node")
        if slot in self.req_position:
            cleared.append("req_position")
        ax = self.aux.get(slot)
        if ax:
            cleared.append(f"aux(n={len(ax)})")
        pa = self.py_attr.get(slot)
        if pa:
            cleared.append(f"py_attr(n={len(pa)})")
        if cleared:
            self._record_history(slot, ("clear_on_free", tuple(cleared)))
        self.tree_node.pop(slot, None)
        self.req_position.pop(slot, None)
        self.aux.pop(slot, None)
        self.py_attr.pop(slot, None)
        return bool(cleared)

    def _record_history(self, slot: int, event: tuple) -> None:
        if not self.history_enabled:
            return
        buf = self.history.setdefault(slot, [])
        buf.append(event)
        if len(buf) > self.history_limit_per_slot:
            del buf[: len(buf) - self.history_limit_per_slot]

    def get_history(self, slot: int) -> List[tuple]:
        return list(self.history.get(slot, []))

    # -- req_to_token position --
    def bind_req_position(self, slot: int, row: int, col: int) -> None:
        ref = self.req_position.get(slot)
        if ref is None:
            self.req_position[slot] = _ReqPosRef(col=col, rows={row})
            self._record_history(slot, ("bind_req_position", row, col))
            return
        # Prefix-sharing invariant: a given slot id occupies exactly ONE
        # column across every req that holds it. If we ever observe a
        # different col here it means the catch-all unbind missed a
        # free path — assert loudly with full context.
        assert ref.col == col, (
            f"SlotBacktrack.bind_req_position multi-col for slot {slot}: "
            f"prior col={ref.col} (rows={sorted(ref.rows)}), "
            f"new (row={row}, col={col}). The catch-all unbind did not "
            f"clear the slot's prior binding before reuse — this is a "
            f"correctness bug, not a tolerable race. Investigate the "
            f"caller path that freed the slot without going through the "
            f"row-aware unbind."
        )
        ref.rows.add(row)
        self._record_history(slot, ("bind_req_position", row, col))

    def unbind_req_position(self, slot: int, row: int) -> None:
        ref = self.req_position.get(slot)
        if ref is None:
            return
        ref.rows.discard(row)
        self._record_history(slot, ("unbind_req_position", row))
        if not ref.rows:
            self.req_position.pop(slot, None)

    def clear_slot_req_position(self, slot: int) -> None:
        """Drop the entire `_ReqPosRef` for a slot regardless of rows.

        Used by the free-time catch-all in
        :py:meth:`MultiEndedAllocator._free_eager` so a slot returning to
        the free pool carries no `req_position` baggage. Compared with
        ``unbind_req_position`` (per-row), this is the all-rows variant.
        """
        if slot in self.req_position:
            self._record_history(slot, ("clear_slot_req_position",))
        self.req_position.pop(slot, None)

    # -- aux (mapping tensors) --
    def bind_aux(self, slot: int, tensor: torch.Tensor, index: int) -> None:
        refs = self.aux.setdefault(slot, [])
        # Detect duplicate bind for the same (tensor, index) — would
        # indicate a missing unbind on the caller's side.
        for r in refs:
            if r.tensor is tensor and r.index == index:
                self.bind_aux_dup_count += 1
                if self.bind_aux_dup_count <= 32:
                    logger.warning(
                        "SlotBacktrack.bind_aux DUPLICATE for slot %d, "
                        "tensor_id=%x, index=%d (cum=%d).",
                        slot,
                        id(tensor),
                        index,
                        self.bind_aux_dup_count,
                    )
                self._record_history(
                    slot,
                    ("bind_aux_duplicate", id(tensor), index, len(refs)),
                )
                refs.append(_AuxRef(tensor=tensor, index=index))
                return
        self._record_history(
            slot, ("bind_aux", id(tensor), index, len(refs))
        )
        refs.append(_AuxRef(tensor=tensor, index=index))

    def unbind_aux(self, slot: int, tensor: torch.Tensor, index: int) -> None:
        refs = self.aux.get(slot)
        if not refs:
            return
        for i, r in enumerate(refs):
            if r.tensor is tensor and r.index == index:
                self._record_history(
                    slot, ("unbind_aux", id(tensor), index)
                )
                refs.pop(i)
                break
        if not refs:
            self.aux.pop(slot, None)

    # -- py_attr bindings for non-tensor Python attributes holding slot ids --

    def bind_py_attr(self, slot: int, obj, attr: str) -> None:
        refs = self.py_attr.setdefault(slot, [])
        for o, a in refs:
            if o is obj and a == attr:
                self.bind_py_attr_dup_count += 1
                if self.bind_py_attr_dup_count <= 32:
                    logger.warning(
                        "SlotBacktrack.bind_py_attr DUPLICATE for slot %d, "
                        "obj_id=%x, attr=%r (cum=%d).",
                        slot,
                        id(obj),
                        attr,
                        self.bind_py_attr_dup_count,
                    )
                self._record_history(
                    slot,
                    ("bind_py_attr_duplicate", id(obj), attr, len(refs)),
                )
                refs.append((obj, attr))
                return
        # Single-owner invariant check: by design exactly one Req's
        # `(obj, attr)` should hold a given mamba slot at any time. If
        # `refs` is non-empty here (and we already checked for exact
        # duplicate above), it means a *different* obj is being added
        # alongside an existing binding — the prior owner was never
        # unbound. Empirically this is what produces "4+ distinct
        # obj_ids accumulating at one slot" in the lifecycle dumps from
        # eval_results_36 / 37. Surface the leak loudly with both the
        # prior bind's caller and the new bind's caller so the leaking
        # allocation path can be localized.
        if refs:
            self.bind_py_attr_multi_owner_count += 1
            cap = 32
            if self.bind_py_attr_multi_owner_count <= cap:
                prior_caller = "unknown (bind_py_attr does not record caller history)"
                # Capture the new bind's caller stack now.
                new_caller = "unknown"
                try:
                    import inspect
                    frames = inspect.stack()[2:9]
                    new_caller = " <- ".join(
                        f"{f.filename.split('/')[-1]}:{f.lineno}" for f in frames
                    )
                except Exception:
                    pass
                prior_summary = ", ".join(
                    f"(obj_id=0x{id(o):x}, attr={a!r})" for o, a in refs[:4]
                )
                if len(refs) > 4:
                    prior_summary += f", ...(+{len(refs)-4} more)"
                logger.warning(
                    "SlotBacktrack.bind_py_attr MULTI-OWNER for slot %d: "
                    "slot already had live entries [%s], adding new "
                    "(obj_id=0x%x, attr=%r). Single-owner invariant "
                    "violated — the prior owner(s) were never unbound. "
                    "cum=%d/%d.\n"
                    "  NEW bind caller: %s\n"
                    "  PRIOR bind caller (best-effort): %s",
                    slot, prior_summary, id(obj), attr,
                    self.bind_py_attr_multi_owner_count, cap,
                    new_caller, prior_caller,
                )
            self._record_history(
                slot,
                ("bind_py_attr_multi_owner", id(obj), attr, len(refs)),
            )
        self._record_history(
            slot, ("bind_py_attr", id(obj), attr, len(refs))
        )
        refs.append((obj, attr))

    def unbind_py_attr(self, slot: int, obj, attr: str) -> None:
        refs = self.py_attr.get(slot)
        if not refs:
            return
        for i, (o, a) in enumerate(refs):
            if o is obj and a == attr:
                self._record_history(slot, ("unbind_py_attr", id(obj), attr))
                refs.pop(i)
                break
        if not refs:
            self.py_attr.pop(slot, None)

    def clear(self) -> None:
        self.tree_node.clear()
        self.req_position.clear()
        self.aux.clear()
        self.py_attr.clear()


class Relocator:
    """
    Synchronous reference-update engine for shared-memory-pool slot
    relocations.

    Owns one :class:`SlotBacktrack` per sub-pool plus a back-reference to
    the ``ReqToTokenPool`` so that, when the
    :class:`MultiEndedAllocator`'s eager compaction relocates a boundary
    slot ``src`` into a freed slot ``dst``, every live reference to
    ``src`` is immediately rewritten to ``dst`` before the call returns.

    There is no deferred ``_pending`` log, no end-of-tick flush, no
    chain-chasing. Callers wanting to know which ``(src, dst)`` pairs an
    allocator emitted in a particular batch can read the
    per-allocator ``_inverse_history`` suffix instead.

    Caller:
      * :meth:`MultiEndedAllocator._apply_relocations` invokes
        :meth:`Relocator.apply` immediately after
        ``move_kv_cache`` / ``copy_from``.
    """

    def __init__(
        self,
        backtracks: Dict[str, SlotBacktrack],
        device: str,
    ):
        self.backtracks = backtracks
        self.device = device
        # Set by the shared-pool wiring code (`init_shared_swa_pools` /
        # `init_shared_mamba_pools`) so `apply()` can reach
        # `req_to_token` and rewrite `(row, col)` cells inline.
        self._req_to_token_pool_ref: Optional[object] = None
        # Enable per-slot lifecycle history on every backtrack — used by
        # MultiEndedAllocator's stale-slot assertion to dump where each
        # stale slot has been bound / unbound / overwritten over its
        # lifetime. The cost is O(1) per bind/unbind (a single append to
        # a 16-deep ring buffer) and the history is bounded per-slot, so
        # it's safe to leave on permanently.
        for bt in backtracks.values():
            bt.history_enabled = True

    def apply(
        self,
        sub_pool: str,
        src: torch.Tensor,
        dst: torch.Tensor,
    ) -> int:
        """Immediate, synchronous reference rewrite for `(src[i], dst[i])`
        pairs. For each pair, rewrite every live holder of `src[i]` to point
        at `dst[i]` and move the binder entries `src -> dst`.

        `dst` was just freed by the same `_free_eager` call (so
        `SlotBacktrack.clear_slot(dst)` already dropped its binder entries);
        step 0 below pops any leftover anyway so the step-6 re-bind cannot
        silently overwrite.

        Returns the number of (src, dst) pairs applied.
        """
        assert sub_pool in self.backtracks, (
            f"unknown sub_pool {sub_pool!r}; registered: "
            f"{list(self.backtracks)}"
        )
        if src.numel() == 0:
            return 0
        src_list = src.detach().cpu().tolist()
        dst_list = dst.detach().cpu().tolist()
        assert len(src_list) == len(dst_list)

        bt = self.backtracks[sub_pool]
        rt_obj = self._req_to_token_pool_ref
        if callable(rt_obj):
            rt_obj = rt_obj()
        req_to_token = (
            getattr(rt_obj, "req_to_token", None) if rt_obj is not None else None
        )
        applied = 0
        for s, d in zip(src_list, dst_list):
            self._apply_one(bt, int(s), int(d), req_to_token)
            applied += 1
        return applied

    def _apply_one(
        self,
        bt: SlotBacktrack,
        src: int,
        dst: int,
        req_to_token: Optional[torch.Tensor],
    ) -> None:
        # ---- Step 0: pre-flight — `dst` was just freed (clear_slot ran),
        # so its binder entries should be empty. Pop any leftover so the
        # step-6 re-bind below cannot silently overwrite.
        bt.tree_node.pop(dst, None)
        bt.req_position.pop(dst, None)
        bt.aux.pop(dst, None)
        bt.py_attr.pop(dst, None)

        # ---- Step 1: pop src bindings.
        tn = bt.tree_node.pop(src, None)
        rp = bt.req_position.pop(src, None)
        ax = bt.aux.pop(src, None)
        pa = bt.py_attr.pop(src, None)

        # ---- Step 2: tree_node — write dst into the bound holder.
        if tn is not None:
            target = getattr(tn.node, tn.attr, None)
            if target is not None:
                target[tn.position] = dst

        # ---- Step 3: req_position — rewrite ONLY the tracked rows,
        #              verifying each cell still holds src.
        if rp is not None and req_to_token is not None and rp.rows:
            rows_t = torch.tensor(
                sorted(rp.rows),
                dtype=torch.int64,
                device=req_to_token.device,
            )
            vals = req_to_token[rows_t, rp.col]
            mask = (vals == src)
            if mask.any():
                req_to_token[rows_t[mask], rp.col] = dst

        # ---- Step 4: aux — scalar-write dst at each (tensor, index).
        if ax is not None:
            for r in ax:
                r.tensor[r.index] = dst

        # ---- Step 5: py_attr — replace the obj attr with a fresh scalar.
        if pa is not None:
            new_t = torch.tensor(dst, dtype=torch.int32, device=self.device)
            for obj, attr in pa:
                priv = f"_{attr}"
                if hasattr(obj, priv):
                    setattr(obj, priv, new_t)
                else:
                    setattr(obj, attr, new_t)

        # ---- Step 6: re-bind reverse-index entries at dst (move src->dst).
        # Step 0 already popped any prior dst entries, so these assignments
        # cannot silently overwrite — by construction the dict is empty at dst.
        if tn is not None:
            bt.tree_node[dst] = tn
        if rp is not None:
            bt.req_position[dst] = rp
        if ax is not None:
            bt.aux[dst] = ax
        if pa is not None:
            bt.py_attr[dst] = pa

