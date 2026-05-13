"""
Shared memory pool backing two (or more in future) sub-pools with one physical
byte buffer.

- ONE raw byte buffer.
- Each MHA sub-pool gets its own per-layer K/V views that are strided views
  into the shared buffer. One sub-pool sits at the low-byte end (grow-up slot
  indices); the other sits at the high-byte end (grow-down slot indices).
- Byte-level sharing: allocators coordinate so each pool's byte range is
  disjoint from its peer's byte range at all times.

Supports MHA-shaped sub-pools and Mamba-shaped sub-pools.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import reduce
from operator import mul
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.mem_cache.memory_pool import (
    HybridReqToTokenPool,
    MambaPool,
    MHATokenToKVPool,
    move_kv_cache_native,
)
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter


def _prod(iterable) -> int:
    return reduce(mul, iterable, 1)


def wrap_req_to_token_pool_write(req_to_token_pool, binder) -> None:
    """Wrap `req_to_token_pool.write(indices, values)` to feed
    `(slot, pos)` pairs into the given SlotBacktrackBinder. Leaves the
    underlying `ReqToTokenPool` class untouched; only the instance's bound
    method is replaced.

    Call once during shared-pool wiring; idempotent assignment is safe.
    """

    original_write = req_to_token_pool.write

    def _extract_and_bind(indices, values):
        try:
            if isinstance(indices, tuple) and len(indices) == 2:
                row, col = indices
                # `row` may be (a) a Python int / 0-D tensor for single-row
                # writes (prefill non-Triton path uses this), or (b) a 1-D
                # tensor of `req_pool_indices` for multi-row writes (the
                # DECODE path at common.py:509 passes
                # `(req_pool_indices_tensor, locs_tensor)`). Detect and
                # dispatch — never funnel a multi-element tensor through
                # `.item()`, which raises and would silently skip every
                # decode binding (root cause of stale-slot assertions
                # observed under --disable-radix-cache eval runs).
                row_is_multi = (
                    isinstance(row, torch.Tensor) and row.numel() > 1
                )
                if row_is_multi:
                    # Multi-row decode case: row, col, and values are 1-D
                    # tensors with matching length. Each (row[i], col[i])
                    # gets values[i].
                    if not isinstance(col, torch.Tensor):
                        return  # unexpected shape — skip silently
                    row_list = row.detach().cpu().tolist()
                    col_list = col.detach().cpu().tolist()
                    vals_list = (
                        values.detach().cpu().tolist()
                        if isinstance(values, torch.Tensor)
                        else list(values)
                    )
                    for v, r, p in zip(vals_list, row_list, col_list):
                        binder.bind_req_position(
                            v, row=int(r), col=int(p)
                        )
                    return
                # Single-row case: row is int / 0-D tensor.
                row_int = (
                    int(row.item())
                    if isinstance(row, torch.Tensor)
                    else int(row)
                )
                if isinstance(col, slice):
                    start = col.start or 0
                    step = col.step or 1
                    vals_list = (
                        values.detach().cpu().tolist()
                        if isinstance(values, torch.Tensor)
                        else list(values)
                    )
                    pos = start
                    for v in vals_list:
                        binder.bind_req_position(v, row=row_int, col=pos)
                        pos += step
                    return
                if isinstance(col, torch.Tensor):
                    col_list = col.detach().cpu().tolist()
                    vals_list = (
                        values.detach().cpu().tolist()
                        if isinstance(values, torch.Tensor)
                        else list(values)
                    )
                    for v, p in zip(vals_list, col_list):
                        binder.bind_req_position(v, row=row_int, col=p)
                    return
        except Exception:  # pragma: no cover — hooks never crash the hot path
            pass

    def hooked_write(indices, values):
        original_write(indices, values)
        if not binder.is_null():
            _extract_and_bind(indices, values)

    req_to_token_pool.write = hooked_write
    # Expose the binder so direct-write paths that bypass `pool.write()`
    # can still feed bind_req_position events. The Triton kernel in
    # `common.write_cache_indices` writes straight into the underlying
    # `req_to_token` tensor without going through this wrap; without
    # this back-reference, those slots would never be tracked by the
    # SlotBacktrack and the relocation flush would silently leave their
    # cells stale (root cause of the SWA stale-slot assertion observed
    # in eval_results_7).
    req_to_token_pool._slot_binder = binder

logger = logging.getLogger(__name__)
GB = 1024 * 1024 * 1024

# Todo: Once we support more than 2 sub-pools, the sub-pools in the middle may
# change their grow direction dynamically, so frozen=True no longer use
@dataclass(frozen=True, kw_only=True)
class SubPoolSpec(ABC):
    """Abstract per-slot layout of one sub-pool in a `SharedMemoryPool`.

    Every sub-pool is identified by `name`, holds state for `layer_num`
    layers, and picks a `grow_direction` (which end of the shared byte
    buffer its allocator grows from). Subclasses (e.g., `MHASubPoolSpec`,
    `MambaSubPoolSpec`) add shape / dtype specifics and must implement
    `entry_bytes`.

    Kept as a `dataclass(kw_only=True)` so subclasses can add fields without
    worrying about positional argument order. All call sites use keyword
    arguments.
    """

    name: str
    layer_num: int
    grow_direction: str  # "up" or "down"

    def __post_init__(self):
        assert self.grow_direction in ("up", "down"), (
            f"grow_direction must be 'up' or 'down'; got "
            f"{self.grow_direction!r}"
        )
        assert self.layer_num > 0, (
            f"layer_num must be positive; got {self.layer_num}"
        )

    @abstractmethod
    def entry_bytes(self) -> int:
        """Bytes consumed by one slot across all `layer_num` layers of this
        sub-pool (e.g., K + V for MHA; conv + temporal for Mamba).
        """
        raise NotImplementedError


@dataclass(frozen=True, kw_only=True)
class MHASubPoolSpec(SubPoolSpec):
    """Per-slot layout of one MHA-shaped sub-pool.

    `v_head_dim` may differ from `head_dim`: this matches `MHATokenToKVPool`'s
    ability to support different K and V per-head dimensions. When
    `v_head_dim is None`, the per-head V dimension falls back to `head_dim`.
    """

    head_num: int
    head_dim: int
    store_dtype: torch.dtype
    v_head_dim: Optional[int] = None

    def __post_init__(self):
        super().__post_init__()
        assert self.head_num > 0, (
            f"head_num must be positive; got {self.head_num}"
        )
        assert self.head_dim > 0, (
            f"head_dim must be positive; got {self.head_dim}"
        )
        # Resolve v_head_dim default.
        # Frozen dataclass requires object.__setattr__.
        if self.v_head_dim is None:
            object.__setattr__(self, "v_head_dim", self.head_dim)
        assert self.v_head_dim > 0, (
            f"v_head_dim must be positive; got {self.v_head_dim}"
        )

    def k_row_bytes(self) -> int:
        return self.head_num * self.head_dim * self.store_dtype.itemsize

    def v_row_bytes(self) -> int:
        return self.head_num * self.v_head_dim * self.store_dtype.itemsize

    def entry_bytes(self) -> int:
        """Bytes consumed by one slot across all layers (K + V).
        `layer_num * (k_row_bytes + v_row_bytes)`
        """
        return self.layer_num * (self.k_row_bytes() + self.v_row_bytes())


@dataclass(frozen=True, kw_only=True)
class MambaSubPoolSpec(SubPoolSpec):
    """Per-slot layout of one Mamba-shaped sub-pool.

    Note: `layer_num` here is the number of **Mamba** layers whose state is
    held (equivalent to the MambaPool `num_mamba_layers` concept).
    """

    conv_state_shapes: Tuple[Tuple[int, ...], ...]  # one shape per conv tensor
    conv_dtype: torch.dtype
    temporal_state_shape: Tuple[int, ...]
    temporal_dtype: torch.dtype

    def __post_init__(self):
        super().__post_init__()
        assert len(self.conv_state_shapes) > 0, (
            "conv_state_shapes must have at least one shape"
        )

    def conv_row_bytes(self, idx: int) -> int:
        return _prod(self.conv_state_shapes[idx]) * self.conv_dtype.itemsize

    def temporal_row_bytes(self) -> int:
        return _prod(self.temporal_state_shape) * self.temporal_dtype.itemsize

    def entry_bytes(self) -> int:
        """Bytes consumed by one slot across all layers, across
        `conv_state[0..len-1]` + `temporal_state`.
        """
        total = 0
        for i in range(len(self.conv_state_shapes)):
            total += self.layer_num * self.conv_row_bytes(i)
        total += self.layer_num * self.temporal_row_bytes()
        return total


class SharedMemoryPool:
    """
    One physical byte buffer shared by two (or more in future) sub-pools.

    Each sub-pool exposes its per-layer K/V or state tensors as strided views
    into the raw buffer. All sub-pools' views are anchored at byte 0; the
    allocators coordinate to keep their byte ranges disjoint / non-overlap
    as they grow in opposite directions; this class does not track usage.
    """

    def __init__(
        self,
        *,
        total_bytes: int,
        sub_pool_specs: List[SubPoolSpec],
        device: str,
        enable_memory_saver: bool,
    ):
        assert len(sub_pool_specs) == 2, (
            f"SharedMemoryPool currently supports exactly 2 sub-pools; got "
            f"{len(sub_pool_specs)}"
        )
        names = [s.name for s in sub_pool_specs]
        assert len(set(names)) == 2, (
            f"sub-pool names must be unique; got {names}"
        )
        directions = sorted(s.grow_direction for s in sub_pool_specs)
        assert directions == ["down", "up"], (
            f"SharedMemoryPool needs one grow-up and one grow-down sub-pool; "
            f"got {directions}"
        )

        self.device = device
        self.total_bytes = total_bytes
        self.sub_pool_specs = sub_pool_specs
        self._specs_by_name: Dict[str, SubPoolSpec] = {
            s.name: s for s in sub_pool_specs
        }

        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            self._raw = torch.empty(total_bytes, dtype=torch.uint8, device=device)
        # zero-init so unset slots read as zeros (matches current behavior).
        self._raw.zero_()

        self._max_slots: Dict[str, int] = {}
        self._anchor_bytes: Dict[str, int] = {}
        self._min_slot_index: Dict[str, int] = {}
        # MHA views: (k_buffer, v_buffer)
        # Mamba views: (conv_state_list, temporal_state)
        self._mha_views: Dict[
            str, Tuple[List[torch.Tensor], List[torch.Tensor]]
        ] = {}
        self._mamba_views: Dict[
            str, Tuple[List[torch.Tensor], torch.Tensor]
        ] = {}

        # Slot-0 padding-write safety.
        # Every pool's slot-0 dummy writes land in raw bytes [0, entry_i).
        # Their union is [0, entry_max). Each pool's first allocatable slot
        # index is chosen so its bytes start at ≥ entry_max — strictly above
        # any pool's slot-0 dummy-write range. This keeps real data safe.
        entry_max = max(s.entry_bytes() for s in sub_pool_specs)

        for spec in sub_pool_specs:
            entry_bytes = spec.entry_bytes()
            max_slots = total_bytes // entry_bytes
            # Ceiling division: min_slot_index * entry_bytes >= entry_max.
            min_slot_index = (entry_max + entry_bytes - 1) // entry_bytes
            if max_slots <= min_slot_index:
                raise RuntimeError(
                    f"SharedMemoryPool: sub-pool {spec.name!r} fits only "
                    f"{max_slots} slots in {total_bytes} bytes, but "
                    f"min_slot_index={min_slot_index} leaves no room for real "
                    f"data. Increase total_bytes."
                )
            # All sub-pool anchors are 0 — uniform view construction, no
            # alignment rounding needed (alignment falls out of
            # min_slot_index · entry_bytes, which is a multiple of entry_bytes
            # and hence of the dtype itemsize).
            anchor = 0
            self._max_slots[spec.name] = max_slots
            self._anchor_bytes[spec.name] = anchor
            self._min_slot_index[spec.name] = min_slot_index
            if isinstance(spec, MHASubPoolSpec):
                self._mha_views[spec.name] = self._build_mha_views(
                    spec, anchor, max_slots
                )
            elif isinstance(spec, MambaSubPoolSpec):
                self._mamba_views[spec.name] = self._build_mamba_views(
                    spec, anchor, max_slots
                )
            else:  # pragma: no cover
                raise TypeError(f"unsupported SubPoolSpec type: {type(spec)}")

        logger.info(
            "[shared-pool] SharedMemoryPool allocated: total_bytes=%.2f GB "
            "(=%d B), %d sub-pool(s)",
            total_bytes / GB,
            total_bytes,
            len(sub_pool_specs),
        )
        for s in sub_pool_specs:
            entry = s.entry_bytes()
            max_s = self._max_slots[s.name]
            anchor = self._anchor_bytes[s.name]
            min_idx = self._min_slot_index[s.name]
            logger.info(
                "[shared-pool]   sub-pool %r: kind=%s, layer_num=%d, "
                "grow_direction=%s, entry_bytes=%d, max_slots=%d, "
                "min_slot_index=%d (slots [0,%d) reserved), anchor_bytes=%d",
                s.name,
                type(s).__name__,
                s.layer_num,
                s.grow_direction,
                entry,
                max_s,
                min_idx,
                min_idx,
                anchor,
            )

    def spec(self, name: str) -> SubPoolSpec:
        """Return the sub-pool spec by name. The concrete type is
        `MHASubPoolSpec` or `MambaSubPoolSpec`; callers that need the specific
        subtype should `isinstance`-check or call `mha_spec` / `mamba_spec`."""
        return self._specs_by_name[name]

    def mha_spec(self, name: str) -> MHASubPoolSpec:
        """Like `spec`, but asserts MHA and narrows the type."""
        s = self._specs_by_name[name]
        assert isinstance(s, MHASubPoolSpec), (
            f"sub-pool {name!r} is {type(s).__name__}, expected MHASubPoolSpec"
        )
        return s

    def mamba_spec(self, name: str) -> MambaSubPoolSpec:
        """Like `spec`, but asserts Mamba and narrows the type."""
        s = self._specs_by_name[name]
        assert isinstance(s, MambaSubPoolSpec), (
            f"sub-pool {name!r} is {type(s).__name__}, expected MambaSubPoolSpec"
        )
        return s

    def max_slots(self, name: str) -> int:
        """Slot-index-space capacity for this sub-pool if it used the whole buffer."""
        return self._max_slots[name]

    def min_slot_index(self, name: str) -> int:
        """Smallest slot index the allocator is allowed to hand out for real
        data. Slots `[0, min_slot_index)` are reserved (slot 0 for padding
        dummy writes; slots 1..min-1 sit unused) so that every pool's real
        data begins at bytes ≥ `entry_max` — strictly above every pool's
        slot-0 dummy-write range."""
        return self._min_slot_index[name]

    def anchor_bytes(self, name: str) -> int:
        """Raw-buffer byte offset at which this sub-pool's slot-0 begins."""
        anchor = self._anchor_bytes[name]
        assert anchor == 0, (
            f"current design assumes all anchors are 0; got {anchor}"
        )
        return anchor

    def mha_views_for(
        self, name: str
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Returns (k_buffer, v_buffer) lists of per-layer strided views for an
        MHA sub-pool."""
        return self._mha_views[name]

    def mamba_views_for(
        self, name: str
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Returns (conv_state_list, temporal_state) views for a Mamba sub-pool."""
        return self._mamba_views[name]

    def _build_mha_views(
        self, spec: MHASubPoolSpec, anchor_bytes: int, max_slots: int
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        For grow-up: slot K at raw-byte offset anchor + K * entry_bytes.
        For grow-down: slot K at raw-byte offset anchor + K * entry_bytes.
          (The allocator does grow-down by starting watermark at max_slots-1
           and decrementing. The byte anchor is set so slot max_slots-1 sits
           at the high byte end.)

        Inside one slot's entry, layout is per-layer interleaved K/V (K first,
        V second per layer — preserves PDF "v[idx] adjacent to k[idx]" intent):
          [L0_K (k_row_bytes)][L0_V (v_row_bytes)][L1_K][L1_V]...[L_{N-1}_V]

        K and V may have different per-head dimensions (`head_dim` vs
        `v_head_dim`); the per-layer (k_row + v_row) byte step encodes that.
        """
        entry_bytes = spec.entry_bytes()
        k_row_bytes = spec.k_row_bytes()
        v_row_bytes = spec.v_row_bytes()
        layer_step_bytes = k_row_bytes + v_row_bytes
        itemsize = spec.store_dtype.itemsize
        assert entry_bytes % itemsize == 0
        assert anchor_bytes % itemsize == 0
        assert k_row_bytes % itemsize == 0
        assert v_row_bytes % itemsize == 0

        # Reinterpret raw bytes as the sub-pool's store dtype.
        as_dtype_view = self._raw.view(spec.store_dtype)
        slot_stride_elems = entry_bytes // itemsize
        anchor_elems = anchor_bytes // itemsize

        k_shape = (max_slots, spec.head_num, spec.head_dim)
        v_shape = (max_slots, spec.head_num, spec.v_head_dim)
        k_stride = (slot_stride_elems, spec.head_dim, 1)
        v_stride = (slot_stride_elems, spec.v_head_dim, 1)

        k_buffer: List[torch.Tensor] = []
        v_buffer: List[torch.Tensor] = []
        for l in range(spec.layer_num):
            # Layer l's K starts at byte (l * layer_step) within the slot;
            # layer l's V starts at byte (l * layer_step + k_row).
            k_offset_elems = anchor_elems + (l * layer_step_bytes) // itemsize
            v_offset_elems = (
                anchor_elems + (l * layer_step_bytes + k_row_bytes) // itemsize
            )
            k_tensor = torch.as_strided(
                as_dtype_view, size=k_shape, stride=k_stride,
                storage_offset=k_offset_elems,
            )
            v_tensor = torch.as_strided(
                as_dtype_view, size=v_shape, stride=v_stride,
                storage_offset=v_offset_elems,
            )
            k_buffer.append(k_tensor)
            v_buffer.append(v_tensor)

        return k_buffer, v_buffer

    def _build_mamba_views(
        self, spec: MambaSubPoolSpec, anchor_bytes: int, max_slots: int
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Per-slot byte layout for a Mamba entry:
          [ conv[0] rows × num_layers ]
          [ conv[1] rows × num_layers ]
          ...
          [ temporal rows × num_layers ]

        Each returned view has shape (num_layers, max_slots, *inner_shape) so
        it matches `MambaPool.State.conv[i]` / `.temporal` conventions.
        Strides:
          - layer stride = per_layer_bytes_for_this_tensor // itemsize
          - slot  stride = entry_bytes // itemsize
          - inner strides = standard C-contiguous over inner_shape
        """
        entry_bytes = spec.entry_bytes()
        N = spec.layer_num

        def c_strides(shape):
            # C-contiguous strides in elements.
            strides = []
            acc = 1
            for s in reversed(shape):
                strides.append(acc)
                acc *= s
            return tuple(reversed(strides))

        # Walk byte offsets: conv[0] for all layers, conv[1] for all layers, ..., temporal for all layers.
        conv_itemsize = spec.conv_dtype.itemsize
        assert entry_bytes % conv_itemsize == 0
        assert anchor_bytes % conv_itemsize == 0
        as_conv_dtype = self._raw.view(spec.conv_dtype)
        conv_slot_stride_elems = entry_bytes // conv_itemsize

        offset_bytes_within_entry = 0
        conv_views: List[torch.Tensor] = []
        for i, conv_shape in enumerate(spec.conv_state_shapes):
            inner_shape_bytes = spec.conv_row_bytes(i)
            assert inner_shape_bytes % conv_itemsize == 0
            offset_elems = (anchor_bytes + offset_bytes_within_entry) // conv_itemsize
            inner_strides = c_strides(conv_shape)
            stride = (
                inner_shape_bytes // conv_itemsize,
                conv_slot_stride_elems,
            ) + inner_strides
            shape = (N, max_slots) + tuple(conv_shape)
            view = torch.as_strided(
                as_conv_dtype, size=shape, stride=stride,
                storage_offset=offset_elems,
            )
            conv_views.append(view)
            offset_bytes_within_entry += N * inner_shape_bytes

        # temporal
        itemsize = spec.temporal_dtype.itemsize
        assert entry_bytes % itemsize == 0
        assert anchor_bytes % itemsize == 0
        inner_shape_bytes = spec.temporal_row_bytes()
        assert inner_shape_bytes % itemsize == 0
        offset_elems = (anchor_bytes + offset_bytes_within_entry) // itemsize
        as_dtype = self._raw.view(spec.temporal_dtype)
        inner_strides = c_strides(spec.temporal_state_shape)
        stride = (
            inner_shape_bytes // itemsize,
            entry_bytes // itemsize,
        ) + inner_strides
        shape = (N, max_slots) + tuple(spec.temporal_state_shape)
        temporal_view = torch.as_strided(
            as_dtype, size=shape, stride=stride,
            storage_offset=offset_elems,
        )

        return conv_views, temporal_view


class SharedMHATokenToKVPool(MHATokenToKVPool):
    """
    MHA KV pool whose `k_buffer` / `v_buffer` are strided views into a
    `SharedMemoryPool` instead of freshly-allocated contiguous tensors.

    - Buffer lifetime is owned by the SharedMemoryPool.
    - Relocation uses the native PyTorch move (strided tensors break the
      tiled Triton `data_strides` convention that assumes stride == row bytes).
    - `slot_backtrack_binder` is an instance attribute (defaults to the null
      binder) set by wiring code after construction.
    """

    def __init__(
        self,
        *,
        shared_buffer: SharedMemoryPool,
        sub_pool_name: str,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        enable_alt_stream: bool = True,
    ):
        from sglang.srt.mem_cache.relocation_log import null_backtrack_binder

        spec = shared_buffer.mha_spec(sub_pool_name)
        k_buffer, v_buffer = shared_buffer.mha_views_for(sub_pool_name)
        max_slots = shared_buffer.max_slots(sub_pool_name)

        # Stash the pre-built views before calling super().__init__ so that
        # our overridden _create_buffers can pick them up.
        self._shared_buffer = shared_buffer
        self._sub_pool_name = sub_pool_name
        self._preallocated_k_buffer = k_buffer
        self._preallocated_v_buffer = v_buffer

        super().__init__(
            size=max_slots - 1,  # -1 for reserved slot 0
            page_size=1,
            dtype=spec.store_dtype,
            head_num=spec.head_num,
            head_dim=spec.head_dim,
            layer_num=spec.layer_num,
            device=shared_buffer.device,
            enable_memory_saver=False,   # not used in MHA sub-pool
            v_head_dim=spec.v_head_dim,  # may differ from head_dim; resolved
                                         # to head_dim by spec.__post_init__
                                         # when the user passes None.
            start_layer=start_layer,
            end_layer=end_layer,
            enable_alt_stream=enable_alt_stream,
            enable_kv_cache_copy=False,  # strided views — force native move
        )

        # Slot-id write binder. Wiring code sets this post-construction.
        self.slot_backtrack_binder = null_backtrack_binder()

    # -- buffer lifecycle overrides --

    def _create_buffers(self):
        # Override parent allocation path. Use pre-built strided views from
        # SharedMemoryPool.
        self.k_buffer = self._preallocated_k_buffer
        self.v_buffer = self._preallocated_v_buffer

        # data_ptrs / data_strides are used by the Triton
        # `copy_all_layer_kv_cache_tiled` kernel. We do not use
        # self.memory_saver_adapter.region for these tensor allocation.
        # We force the native move path (see move_kv_cache override), so
        # these values aren't consumed by us, but we still populate them
        # for any external code that may inspect them.
        self.k_data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.k_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        self.v_data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.v_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        self.data_ptrs = torch.cat([self.k_data_ptrs, self.v_data_ptrs], dim=0)
        # Stride in BYTES from slot K to slot K+1 in each view.
        self.data_strides = torch.tensor(
            [x.stride(0) * x.dtype.itemsize for x in (self.k_buffer + self.v_buffer)],
            device=self.device,
        )

    def _clear_buffers(self):
        # Lifetime is owned by SharedMemoryPool; do not delete the views.
        pass

    # -- move_kv_cache: force native path (strided views incompatible with
    #    the tiled Triton kernel's stride==size assumption) --

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        if tgt_loc.numel() == 0:
            return
        move_kv_cache_native(self.k_buffer, self.v_buffer, tgt_loc, src_loc)

    # -- size reporting --

    def get_kv_size_bytes(self):
        # The shared buffer's total size is logged by SharedMemoryPool at
        # construction; per-sub-pool accounting would double-count.
        return 0, 0


class SharedSWAKVPool(SWAKVPool):
    """
    Shared-buffer replacement for `SWAKVPool`.

    Composes two `SharedMHATokenToKVPool` instances (full + swa) that alias
    the same physical byte buffer. Exposes the same interface as `SWAKVPool`
    so downstream attention/kernel code is unchanged.

    Inherits from `SWAKVPool` purely for the typing/contract relationship —
    `isinstance(kvcache, SWAKVPool)` is checked across attention backends
    (FlashAttn, AITER), the disagg path, models/utils, etc., to gate the
    SWA-aware kv-pool plumbing. We do NOT call the parent `__init__`: it
    would build static-partition `MHATokenToKVPool` instances, which is
    exactly what the shared pool replaces. The attribute layout the parent
    sets is replicated here against the shared buffer.
    """

    def __init__(
        self,
        *,
        shared_buffer: SharedMemoryPool,
        swa_attention_layer_ids: List[int],
        full_attention_layer_ids: List[int],
        enable_memory_saver: bool = False,
    ):
        # NOTE: do NOT call `super().__init__(...)`. The SWAKVPool body
        # would allocate two static-partition MHA pools.
        self.shared_buffer = shared_buffer
        self.swa_layer_nums = len(swa_attention_layer_ids)
        self.full_layer_nums = len(full_attention_layer_ids)
        self.start_layer = 0
        self.page_size = 1
        self.swa_loc: Optional[torch.Tensor] = None

        # The parent class exposes `size` / `size_swa` as plain attributes
        # (set in its __init__). Match that contract — these values are
        # constants of the SharedMemoryPool, fixed at allocation time.
        self.size = shared_buffer.max_slots("full") - 1
        self.size_swa = shared_buffer.max_slots("swa") - 1

        # `dtype` is read from MHASubPoolSpec.store_dtype; both sub-pools
        # share the same dtype because the SWA path is symmetric here.
        full_spec = shared_buffer.mha_spec("full")
        self.dtype = full_spec.store_dtype
        self.head_num = full_spec.head_num
        self.head_dim = full_spec.head_dim

        self.full_kv_pool = SharedMHATokenToKVPool(
            shared_buffer=shared_buffer,
            sub_pool_name="full",
        )
        self.swa_kv_pool = SharedMHATokenToKVPool(
            shared_buffer=shared_buffer,
            sub_pool_name="swa",
        )

        # for disagg with nvlink — currently disabled in shared-pool, but keep the
        # attributes present so any caller reading them doesn't AttributeError.
        self.enable_custom_mem_pool = False
        self.custom_mem_pool = None

        # {global_layer_id: (per-pool index, is_swa_layer)}
        self.layers_mapping: Dict[int, Tuple[int, bool]] = {}
        for idx, gid in enumerate(full_attention_layer_ids):
            self.layers_mapping[gid] = (idx, False)
        for idx, gid in enumerate(swa_attention_layer_ids):
            self.layers_mapping[gid] = (idx, True)

        self.full_to_swa_index_mapping: Optional[torch.Tensor] = None

        # For byte budget accounting only.
        self.mem_usage = shared_buffer.total_bytes / GB
        logger.info(
            "SharedSWAKVPool: total %.2f GB, full layers=%d, swa layers=%d, "
            "full max_slots=%d, swa max_slots=%d",
            self.mem_usage,
            self.full_layer_nums,
            self.swa_layer_nums,
            shared_buffer.max_slots("full"),
            shared_buffer.max_slots("swa"),
        )

    # -- mapping registration (mirrors SWAKVPool.register_mapping) --

    def register_mapping(self, full_to_swa_index_mapping: torch.Tensor) -> None:
        self.full_to_swa_index_mapping = full_to_swa_index_mapping

    # -- size/info --

    def get_kv_size_bytes(self):
        total = self.shared_buffer.total_bytes
        # Split reporting 50/50 as a convention (exact per-side accounting
        # requires the live allocator state).
        return total // 2, total // 2

    def get_contiguous_buf_infos(self):
        return self.full_kv_pool.get_contiguous_buf_infos()

    def get_state_buf_infos(self):
        return self.swa_kv_pool.get_contiguous_buf_infos()

    # -- buffer accessors --

    def get_key_buffer(self, layer_id: int):
        pool_layer_id, is_swa = self.layers_mapping[layer_id]
        pool = self.swa_kv_pool if is_swa else self.full_kv_pool
        return pool.get_key_buffer(pool_layer_id)

    def get_value_buffer(self, layer_id: int):
        pool_layer_id, is_swa = self.layers_mapping[layer_id]
        pool = self.swa_kv_pool if is_swa else self.full_kv_pool
        return pool.get_value_buffer(pool_layer_id)

    def get_kv_buffer(self, layer_id: int):
        pool_layer_id, is_swa = self.layers_mapping[layer_id]
        pool = self.swa_kv_pool if is_swa else self.full_kv_pool
        return pool.get_kv_buffer(pool_layer_id)

    # -- loc translation --

    def set_swa_loc(self, loc: torch.Tensor) -> None:
        self.swa_loc = loc

    def translate_loc_from_full_to_swa(self, kv_indices: torch.Tensor):
        assert self.full_to_swa_index_mapping is not None
        return self.full_to_swa_index_mapping[kv_indices].to(torch.int32)

    # -- kv writing --

    def set_kv_buffer(
        self,
        layer,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
    ):
        layer_id = layer.layer_id
        pool_layer_id, is_swa = self.layers_mapping[layer_id]
        if is_swa:
            if self.swa_loc is not None:
                loc = self.swa_loc
            elif self.full_to_swa_index_mapping is not None:
                loc = self.translate_loc_from_full_to_swa(loc)
            self.swa_kv_pool.set_kv_buffer(
                None, loc, cache_k, cache_v, k_scale, v_scale,
                layer_id_override=pool_layer_id,
            )
        else:
            self.full_kv_pool.set_kv_buffer(
                None, loc, cache_k, cache_v, k_scale, v_scale,
                layer_id_override=pool_layer_id,
            )

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        self.full_kv_pool.move_kv_cache(tgt_loc, src_loc)
        if self.full_to_swa_index_mapping is not None:
            tgt_swa = self.translate_loc_from_full_to_swa(tgt_loc)
            src_swa = self.translate_loc_from_full_to_swa(src_loc)
            self.swa_kv_pool.move_kv_cache(tgt_swa, src_swa)


class SharedMambaPool(MambaPool):
    """
    Mamba pool whose `conv_state` / `temporal_state` are strided views into a
    `SharedMemoryPool` instead of freshly-allocated contiguous tensors.

    alloc / free / clear / available_size delegate to an external
    `MultiEndedAllocator` (attached via `attach_allocator`) so the MHA and
    Mamba sub-pools share a single byte buffer with a dynamic boundary.

    Does NOT call `super().__init__()` — `MambaPool.__init__` allocates fresh
    tensors, which we replace wholesale. Instead we replicate the minimal
    state setup using the existing `MambaPool.State` / `SpeculativeState`
    dataclasses so inherited methods (`copy_from`, etc.) continue to work
    unchanged.
    """

    def __init__(
        self,
        *,
        shared_buffer: SharedMemoryPool,
        sub_pool_name: str,
        spec_state_size: int,
        mamba_layer_ids: List[int],
        enable_memory_saver: bool = False,
        speculative_num_draft_tokens: Optional[int] = None,
    ):
        from sglang.srt.mem_cache.relocation_log import null_backtrack_binder

        spec = shared_buffer.mamba_spec(sub_pool_name)
        assert spec.layer_num == len(mamba_layer_ids)
        conv_views, temporal_view = shared_buffer.mamba_views_for(sub_pool_name)
        max_slots = shared_buffer.max_slots(sub_pool_name)

        self._shared_buffer = shared_buffer
        self._sub_pool_name = sub_pool_name
        self._external_allocator = None  # set via attach_allocator

        # Replicate the state that MambaPool.__init__ would have set. `size`
        # is the pool's static usable-slot count (max_slots - 1 reserves slot
        # 0 for the padding sink). It is intentionally a plain attribute, not
        # a dynamic property: upstream code (leak checker, metrics, tests)
        # treats `pool.size` as a constant pool property, and byte-frontier
        # coordination with the peer sub-pool must NOT change this value.
        # Byte-coordinated capacity is exposed via `schedulable_available_size()`
        # for callers (e.g. the scheduler's alloc planner) that need it.
        self._max_size = max_slots - 1   # -1 for reserved slot 0
        self.size = self._max_size
        self.device = shared_buffer.device
        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )
        # for disagg with nvlink, which is not supported yet
        self.enable_custom_mem_pool = False
        self.custom_mem_pool = None

        # Keep the `num_mamba_layers` public attribute on self so that code
        # paths expecting a MambaPool interface (memory_pool_host.py, RDMA
        # registration helpers) don't break. This field on the spec is now
        # `layer_num`.
        self.num_mamba_layers = spec.layer_num

        # Pre-built conv/temporal views aliased over the shared byte buffer.
        assert conv_views[0].shape[0] == self.num_mamba_layers, (
            f"number of layers in conv_views={conv_views[0].shape[0]} vs "
            f"expected {self.num_mamba_layers}"
        )
        assert conv_views[0].shape[1] == self._max_size + 1, (
            f"shared buffer slots={conv_views[0].shape[1]} vs "
            f"expected {self._max_size + 1}"
        )

        # Optional per-draft-token intermediate buffers. These have a different
        # outer size (spec_state_size+1) so they aren't part of the shared
        # byte buffer; allocate locally.
        temporal_state_shape = spec.temporal_state_shape
        conv_state_shape = spec.conv_state_shapes
        conv_dtype = spec.conv_dtype
        ssm_dtype = spec.temporal_dtype
        if speculative_num_draft_tokens is not None:
            with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
                intermediate_ssm_state_cache = torch.zeros(
                    size=(
                        self.num_mamba_layers,
                        spec_state_size + 1,
                        speculative_num_draft_tokens,
                        temporal_state_shape[0],
                        temporal_state_shape[1],
                        temporal_state_shape[2],
                    ),
                    dtype=ssm_dtype,
                    device=shared_buffer.device,
                )
                intermediate_conv_window_cache = [
                    torch.zeros(
                        size=(
                            self.num_mamba_layers,
                            spec_state_size + 1,
                            speculative_num_draft_tokens,
                            conv_shape[0],
                            conv_shape[1],
                        ),
                        dtype=conv_dtype,
                        device=shared_buffer.device,
                    )
                    for conv_shape in conv_state_shape
                ]
            self.mamba_cache = self.SpeculativeState(
                conv=list(conv_views),
                temporal=temporal_view,
                intermediate_ssm=intermediate_ssm_state_cache,
                intermediate_conv_window=intermediate_conv_window_cache,
            )
        else:
            self.mamba_cache = self.State(
                conv=list(conv_views), temporal=temporal_view
            )

        # NOTE: Do NOT set `self.free_slots` here. In the non-shared
        # `MambaPool`, `free_slots` is the in-struct free-list that the pool
        # uses for alloc/free/clear/available_size. We override these functions
        # to delegate to an external `MultiEndedAllocator`, so we don't own a
        # free-list. Instead we expose `free_slots` as a property below that
        # derives from the allocator's watermark — keeps
        # `scheduler_runtime_checker_mixin` leak diagnostics and the
        # inherited `MambaPool.available_size()` reading accurate data.
        self.mem_usage = shared_buffer.total_bytes / GB
        self.layer_transfer_counter = None

        # Slot-id write binder. Wiring code sets this post-construction.
        self.slot_backtrack_binder = null_backtrack_binder()

        logger.info(
            "SharedMambaPool(%s) wrapped shared buffer: max_slots=%d, "
            "num_mamba_layers=%d",
            sub_pool_name,
            max_slots,
            self.num_mamba_layers,
        )

    # -- allocator delegation (MambaPool.copy_from / etc. are inherited) --

    def attach_allocator(self, allocator) -> None:
        """Wire an external `MultiEndedAllocator` to handle alloc/free/clear.
        When attached, SharedMambaPool.alloc/free delegate to the allocator
        (which coordinates byte ranges with the peer MHA sub-pool)."""
        self._external_allocator = allocator

    # -- free_slots / available_size proxies --
    #
    # External callers (scheduler memory-leak checker at
    # `scheduler_runtime_checker_mixin.py:162` and common capacity checks at
    # `common.py:305`) consult `mamba_pool.free_slots` and
    # `mamba_pool.available_size()`. In the parent MambaPool these are backed
    # by an in-struct free-list; here we route both to the external allocator
    # so the returned data reflects the true allocator state.

    @property
    def free_slots(self) -> torch.Tensor:
        """Slot ids currently available for allocation, derived from the
        external allocator's watermark / min_slot_index / max_slots.

        Before `attach_allocator` runs (construction bootstrap), falls back
        to the pre-shared-pool convention `arange(1, size+1)` so any caller
        that inspects the field during setup sees a sane default.
        """
        if self._external_allocator is None:
            return torch.arange(
                1, self._max_size + 1, dtype=torch.int64, device=self.device
            )
        alloc = self._external_allocator
        if alloc.grow_direction == "up":
            # Allocated: [min_slot_index, watermark). Free: [watermark, max_slots).
            start, end = alloc.watermark, alloc.max_slots
        else:
            # Allocated: (watermark, max_slots). Free: [min_slot_index, watermark+1).
            start, end = alloc.min_slot_index, alloc.watermark + 1
        if start >= end:
            return torch.empty((0,), dtype=torch.int64, device=self.device)
        return torch.arange(
            start, end, dtype=torch.int64, device=self.device
        )

    def available_size(self) -> int:
        """Slot-bound free count: `size - allocated_count`.

        This is the slots-conservation view used by the scheduler's leak check
        (`scheduler_runtime_checker_mixin._get_mamba_token_info`), where
        `num_used = size - available - evictable` must reduce to
        `allocated_count - evictable` for the conservation law to hold. It
        deliberately does NOT include byte-frontier coordination with the peer
        sub-pool — that constraint is exposed via `schedulable_available_size()`
        and only applies when planning a fresh allocation.
        """
        if self._external_allocator is None:
            return self._max_size
        return self._max_size - self._external_allocator.allocated_count()

    def schedulable_available_size(self) -> int:
        """Byte-coordinated free count, used by the scheduler's alloc planner
        (`common.alloc_req_slots`) when deciding whether to evict before
        allocating. May be smaller than `available_size()` when the peer
        sub-pool's byte usage tightens this side's effective capacity.
        """
        if self._external_allocator is None:
            return self._max_size
        return self._external_allocator.available_size()

    def alloc(self, need_size: int):
        if self._external_allocator is None:
            raise RuntimeError("SharedMambaPool.alloc called before attach_allocator")
        select_index = self._external_allocator.alloc(need_size)
        if select_index is None:
            return None

        # Clear newly-allocated conv/temporal rows (matches MambaPool.alloc
        # behavior at memory_pool.py's alloc, ~lines 350-360).
        for i in range(len(self.mamba_cache.conv)):
            t = self.mamba_cache.conv[i]
            z = torch.zeros(1, dtype=t.dtype, device=t.device).expand(
                t.shape[0], need_size, *t.shape[2:]
            )
            t[:, select_index] = z
        t = self.mamba_cache.temporal
        z = torch.zeros(1, dtype=t.dtype, device=t.device).expand(
            t.shape[0], need_size, *t.shape[2:]
        )
        t[:, select_index] = z

        return select_index

    def free(self, free_index: torch.Tensor):
        if self._external_allocator is None:
            raise RuntimeError("SharedMambaPool.free called before attach_allocator")
        # Under the immediate-apply refactor the binder is updated
        # synchronously inside `MultiEndedAllocator._apply_relocations`;
        # no per-free post-flush is needed.
        self._external_allocator.free(free_index)

    def clear(self):
        if self._external_allocator is None:
            raise RuntimeError("SharedMambaPool.clear called before attach_allocator")
        self._external_allocator.clear()

    def is_slot_allocated(self, slot: int) -> bool:
        """Forward to the external allocator for slot-validity checks. Used
        by the defensive validation backstop in `MambaRadixCache` to reject
        stale `req.mamba_pool_idx` values before they are laundered into
        the radix tree."""
        if self._external_allocator is None:
            return False
        return self._external_allocator.is_slot_allocated(int(slot))

    def allocator_state_str(self) -> str:
        """One-line allocator state summary for diagnostic warnings."""
        if self._external_allocator is None:
            return "<no external allocator attached>"
        return self._external_allocator.allocator_state_str()


class SharedHybridReqToTokenPool(HybridReqToTokenPool):
    """
    Hybrid req-to-token pool whose MambaPool is a `SharedMambaPool` aliasing a
    shared byte buffer. Also installs a `Req.mamba_pool_idx` write hook that
    routes into the Mamba SlotBacktrack for targeted relocation flushes.
    """

    def __init__(
        self,
        *,
        shared_buffer: SharedMemoryPool,
        mamba_sub_pool_name: str,
        size: int,
        max_context_len: int,
        device: str,
        enable_memory_saver: bool,
        cache_params,
        mamba_layer_ids: List[int],
        enable_mamba_extra_buffer: bool,
        speculative_num_draft_tokens: Optional[int] = None,
        enable_overlap_schedule: bool = True,
        start_layer: Optional[int] = None,
    ):
        self._shared_buffer = shared_buffer
        self._mamba_sub_pool_name = mamba_sub_pool_name
        # mamba_size matches SharedMemoryPool.max_slots - 1 (reserve slot 0).
        self._shared_mamba_size = (
            shared_buffer.max_slots(mamba_sub_pool_name) - 1
        )
        mamba_spec_state_size = speculative_num_draft_tokens or 0
        super().__init__(
            size=size,
            mamba_size=self._shared_mamba_size,
            mamba_spec_state_size=mamba_spec_state_size,
            max_context_len=max_context_len,
            device=device,
            enable_memory_saver=enable_memory_saver,
            cache_params=cache_params,
            mamba_layer_ids=mamba_layer_ids,
            enable_mamba_extra_buffer=enable_mamba_extra_buffer,
            speculative_num_draft_tokens=speculative_num_draft_tokens,
            enable_overlap_schedule=enable_overlap_schedule,
            start_layer=start_layer,
        )

    # Override to construct SharedMambaPool instead of plain MambaPool.
    def _init_mamba_pool(
        self,
        size: int,
        mamba_spec_state_size: int,
        cache_params,
        mamba_layer_ids: List[int],
        device: str,
        enable_mamba_extra_buffer: bool,
        speculative_num_draft_tokens: Optional[int] = None,
    ):
        # SharedMambaPool reads conv/temporal shapes from its sub-pool spec
        # (via shared_buffer.mamba_spec()), so cache_params is unused here
        # and is NOT forwarded — its __init__ does not accept that kwarg.
        self.mamba_pool = SharedMambaPool(
            shared_buffer=self._shared_buffer,
            sub_pool_name=self._mamba_sub_pool_name,
            spec_state_size=mamba_spec_state_size,
            mamba_layer_ids=mamba_layer_ids,
            enable_memory_saver=self.enable_memory_saver,
            speculative_num_draft_tokens=speculative_num_draft_tokens,
        )
        self.mamba_map = {
            layer_id: i for i, layer_id in enumerate(mamba_layer_ids)
        }

        self.device = device
        self.req_index_to_mamba_index_mapping: torch.Tensor = torch.zeros(
            self.size, dtype=torch.int32, device=self.device
        )
        if enable_mamba_extra_buffer:
            self.req_index_to_mamba_ping_pong_track_buffer_mapping: torch.Tensor = (
                torch.zeros(
                    (self.size, self.mamba_ping_pong_track_buffer_size),
                    dtype=torch.int32,
                    device=self.device,
                )
            )

    # -- bind/unbind helpers (no-op unless shared pool is on) --

    def _mamba_binder(self):
        from sglang.srt.mem_cache.relocation_log import null_backtrack_binder

        return getattr(
            self.mamba_pool, "slot_backtrack_binder", null_backtrack_binder()
        )

    @staticmethod
    def _slot_of(mid) -> int:
        return int(mid.item()) if hasattr(mid, "item") else int(mid)

    def _bind_req_mamba(self, req, mid) -> None:
        """Record that `req.mamba_pool_idx` currently holds slot `mid`."""
        binder = self._mamba_binder()
        if binder.is_null() or mid is None:
            return
        binder.bind_py_attr(self._slot_of(mid), req, "mamba_pool_idx")

    def _unbind_req_mamba(self, req, mid) -> None:
        binder = self._mamba_binder()
        if binder.is_null() or mid is None:
            return
        binder.unbind_py_attr(self._slot_of(mid), req, "mamba_pool_idx")

    # -- alloc override: parent assigns req.mamba_pool_idx on cache miss.
    #    We re-bind the assignment so the Mamba backtrack knows which Req
    #    currently holds each Mamba slot id.

    def alloc(self, reqs):
        # Capture per-req state BEFORE super().alloc so we can distinguish:
        #   (a) fresh-alloc reqs (req_pool_idx was None) — need new bindings.
        #   (b) cache-hit-reuse reqs (mamba_pool_idx was set by match_prefix
        #       BEFORE alloc, but req_pool_idx was None) — py_attr already
        #       bound by match_prefix, but aux still needs fresh binding.
        #   (c) chunked-prefill continuation reqs (BOTH mamba_pool_idx and
        #       req_pool_idx were already set) — `ReqToTokenPool.alloc`
        #       REUSES the existing req_pool_idx (memory_pool.py:155-179);
        #       py_attr and aux are already bound from the original alloc,
        #       so we MUST NOT re-bind or `bind_aux DUPLICATE` accumulates
        #       (one DUP per chunk; surfaced as ~30-60 DUPs per run in
        #       eval_results_15, even after the scheduler.py:2466 fix).
        prior_mid = [r.mamba_pool_idx for r in reqs]
        prior_req_pool_idx = [r.req_pool_idx for r in reqs]
        result = super().alloc(reqs)
        if result is None:
            return None
        for req, old in zip(reqs, prior_mid):
            if old is None and req.mamba_pool_idx is not None:
                # Parent just assigned a fresh mamba slot id.
                self._bind_req_mamba(req, req.mamba_pool_idx)
        # Bind the mapping-aux entry mapping[req_pool_idx] = mamba_slot ONLY
        # for reqs that didn't already have a req_pool_idx (i.e. are not the
        # chunked-prefill reuse case). For reuse, the binding is unchanged
        # from the prior alloc.
        binder = self._mamba_binder()
        if not binder.is_null():
            for req, prior_idx in zip(reqs, prior_req_pool_idx):
                mid = req.mamba_pool_idx
                if mid is None:
                    continue
                if prior_idx is not None:
                    # Chunked-prefill reuse: req_pool_idx unchanged, the aux
                    # binding (slot, mapping, idx) is already in the binder
                    # from the prior alloc — re-binding would be a no-op
                    # but is detected as DUPLICATE.
                    continue
                slot = self._slot_of(mid)
                binder.bind_aux(
                    slot,
                    self.req_index_to_mamba_index_mapping,
                    int(req.req_pool_idx),
                )
        return result

    # -- free_mamba_cache override: unbind the req's mamba_pool_idx.

    def free_mamba_cache(self, req, mamba_ping_pong_track_buffer_to_keep=None):
        old = req.mamba_pool_idx
        self._unbind_req_mamba(req, old)
        # Also unbind the aux binding on the mapping.
        binder = self._mamba_binder()
        if not binder.is_null() and old is not None:
            slot = self._slot_of(old)
            binder.unbind_aux(
                slot,
                self.req_index_to_mamba_index_mapping,
                int(req.req_pool_idx),
            )
        return super().free_mamba_cache(
            req, mamba_ping_pong_track_buffer_to_keep
        )

    # -- transfer_mamba_to_radix override: drop the req's binder entries
    #    WITHOUT freeing the slot. Used by `MambaRadixCache.cache_finished_req`
    #    when the new tree node takes ownership of the req's mamba slot
    #    (`mamba_exist=False` insert path). The slot stays allocated; the
    #    tree node's `tree_node` binder entry is now the only live reference.
    #
    #    Without this, the binder retains the alloc-time `py_attr(slot, req,
    #    'mamba_pool_idx')` and `aux(slot, mapping, req_pool_idx)` entries
    #    even after ownership has transferred to the tree. When the tree
    #    later evicts that node and frees the slot, the catch-all sees those
    #    entries as a live reference and poisons `req.mamba_pool_idx = -1` /
    #    `mapping[req_pool_idx] = -1`. The poisoned -1 then leaks into a
    #    subsequent `cache_finished_req` (via line 599's
    #    `req.mamba_pool_idx.unsqueeze(-1).clone()`) and back into the tree
    #    as `node.mamba_value=[-1]`, eventually crashing on a future
    #    `mamba_pool.free([-1])` (eval_results_1, all 3 failing Mamba cells).

    def transfer_mamba_to_radix(self, req):
        old = req.mamba_pool_idx
        if old is None:
            return
        self._unbind_req_mamba(req, old)
        binder = self._mamba_binder()
        if not binder.is_null() and req.req_pool_idx is not None:
            slot = self._slot_of(old)
            binder.unbind_aux(
                slot,
                self.req_index_to_mamba_index_mapping,
                int(req.req_pool_idx),
            )
        # Clear the dangling handle. If left, the
        # `if req.mamba_pool_idx is not None` reuse branch in
        # `HybridReqToTokenPool.alloc` (memory_pool.py:547) would relaunch
        # the req on a slot the tree now owns — clobbering the cached state.
        req.mamba_pool_idx = None


# =============================================================================
# Shared-memory-pool wiring helpers.
#
# `_init_pools` in `model_runner_kv_cache_mixin.py` dispatches into one of these
# helpers when `--enable-shared-memory-pool` is on. Keeping the wiring here (a)
# colocates all shared-pool construction in one file, (b) keeps `_init_pools`
# readable as a thin dispatcher.
# =============================================================================


@dataclass
class SharedPoolBundle:
    """Outputs from `init_shared_swa_pools` / `init_shared_mamba_pools`.

    `req_to_token_pool` is set only by the Mamba helper (which builds a
    `SharedHybridReqToTokenPool`); the SWA helper wraps the caller-provided
    standard `ReqToTokenPool` in place and leaves this field None.
    """

    shared_memory_pool: SharedMemoryPool
    token_to_kv_pool: object
    token_to_kv_pool_allocator: object
    relocation_log: object
    req_to_token_pool: Optional[object] = None


def _store_dtype_for(kv_cache_dtype: torch.dtype) -> torch.dtype:
    if kv_cache_dtype in (torch.float8_e5m2, torch.float8_e4m3fn):
        return torch.uint8
    return kv_cache_dtype


def init_shared_swa_pools(
    *,
    req_to_token_pool,
    device: str,
    kv_cache_dtype: torch.dtype,
    head_num: int,
    head_dim: int,
    full_attention_layer_ids: List[int],
    swa_attention_layer_ids: List[int],
    full_max_total_num_tokens: int,
    swa_max_total_num_tokens: int,
    enable_memory_saver: bool,
    swa_full_tokens_ratio: float,
    need_sort: bool,
    v_head_dim: Optional[int] = None,
    swa_head_num: Optional[int] = None,
    swa_head_dim: Optional[int] = None,
    swa_v_head_dim: Optional[int] = None,
) -> SharedPoolBundle:
    """Build the SWA shared-pool stack: SharedMemoryPool, SharedSWAKVPool,
    SlotBacktrack/binder per sub-pool, Relocator, and the
    SharedSWATokenToKVPoolAllocator. Wraps `req_to_token_pool.write` in place
    so slot-id writes feed the FULL binder.

    Heterogeneous-shape SWA (`is_hybrid_swa_compress` models) is supported via
    the `swa_*` kwargs — when provided, the SWA sub-pool spec is built with
    those dims. Each kwarg falls back to the corresponding full-pool dim when
    None, matching the symmetric SWA case.
    """
    from sglang.srt.mem_cache.multi_ended_allocator import (
        SharedSWATokenToKVPoolAllocator,
    )
    from sglang.srt.mem_cache.relocation_log import (
        Relocator,
        SlotBacktrack,
        SlotBacktrackBinder,
    )

    store_dtype = _store_dtype_for(kv_cache_dtype)
    full_spec = MHASubPoolSpec(
        name="full",
        layer_num=len(full_attention_layer_ids),
        head_num=head_num,
        head_dim=head_dim,
        v_head_dim=v_head_dim,
        store_dtype=store_dtype,
        grow_direction="up",
    )
    swa_spec = MHASubPoolSpec(
        name="swa",
        layer_num=len(swa_attention_layer_ids),
        head_num=swa_head_num if swa_head_num is not None else head_num,
        head_dim=swa_head_dim if swa_head_dim is not None else head_dim,
        v_head_dim=swa_v_head_dim if swa_v_head_dim is not None else v_head_dim,
        store_dtype=store_dtype,
        grow_direction="down",
    )
    total_bytes = (
        full_max_total_num_tokens * full_spec.entry_bytes()
        + swa_max_total_num_tokens * swa_spec.entry_bytes()
    )
    shared_pool = SharedMemoryPool(
        total_bytes=total_bytes,
        sub_pool_specs=[full_spec, swa_spec],
        device=device,
        enable_memory_saver=enable_memory_saver,
    )
    token_to_kv_pool = SharedSWAKVPool(
        shared_buffer=shared_pool,
        swa_attention_layer_ids=swa_attention_layer_ids,
        full_attention_layer_ids=full_attention_layer_ids,
        enable_memory_saver=enable_memory_saver,
    )

    full_backtrack = SlotBacktrack(size_total=shared_pool.max_slots("full"))
    swa_backtrack = SlotBacktrack(size_total=shared_pool.max_slots("swa"))
    relocation_log = Relocator(
        backtracks={"full": full_backtrack, "swa": swa_backtrack},
        device=device,
    )
    # Set the req_to_token_pool back-reference up-front so `apply()` (the
    # immediate-rewrite path called from MultiEndedAllocator._apply_relocations)
    # can read `req_to_token` without waiting for a first `flush()` call.
    relocation_log._req_to_token_pool_ref = req_to_token_pool
    full_binder = SlotBacktrackBinder(full_backtrack)
    swa_binder = SlotBacktrackBinder(swa_backtrack)
    token_to_kv_pool.full_kv_pool.slot_backtrack_binder = full_binder
    token_to_kv_pool.swa_kv_pool.slot_backtrack_binder = swa_binder

    # req_to_token stores full-pool slot ids; route position info into the
    # FULL binder so the immediate apply rewrites the right cells.
    wrap_req_to_token_pool_write(req_to_token_pool, full_binder)

    allocator = SharedSWATokenToKVPoolAllocator(
        shared_buffer=shared_pool,
        kvcache=token_to_kv_pool,
        relocation_log=relocation_log,
        device=device,
        full_max_total_num_tokens=full_max_total_num_tokens,
        swa_max_total_num_tokens=swa_max_total_num_tokens,
        need_sort=need_sort,
    )

    logger.info(
        "[shared-pool] ============================================================"
    )
    logger.info(
        "[shared-pool] SHARED MEMORY POOL ENABLED -- path=SWA hybrid"
    )
    logger.info(
        "[shared-pool]   full_layers=%d, swa_layers=%d, head_num=%d, "
        "head_dim=%d, v_head_dim=%s, swa_head_num=%s, swa_head_dim=%s, "
        "swa_v_head_dim=%s",
        len(full_attention_layer_ids),
        len(swa_attention_layer_ids),
        head_num,
        head_dim,
        v_head_dim,
        swa_head_num,
        swa_head_dim,
        swa_v_head_dim,
    )
    logger.info(
        "[shared-pool]   total_bytes=%d, full_max_total_num_tokens=%d, "
        "swa_max_total_num_tokens=%d",
        shared_pool.total_bytes,
        full_max_total_num_tokens,
        swa_max_total_num_tokens,
    )
    logger.info(
        "[shared-pool]   swa_full_tokens_ratio=%s governs total budget only, "
        "not the runtime split.",
        swa_full_tokens_ratio,
    )
    logger.info(
        "[shared-pool]   binder wired:  full=YES  swa=YES  req_to_token.write=WRAPPED"
    )
    logger.info(
        "[shared-pool] ============================================================"
    )
    return SharedPoolBundle(
        shared_memory_pool=shared_pool,
        token_to_kv_pool=token_to_kv_pool,
        token_to_kv_pool_allocator=allocator,
        relocation_log=relocation_log,
    )


def init_shared_mamba_pools(
    *,
    device: str,
    kv_cache_dtype: torch.dtype,
    head_num: int,
    head_dim: int,
    page_size: int,
    start_layer: int,
    end_layer: int,
    is_draft_worker: bool,
    use_mla_backend: bool,
    mamba_layer_ids: List[int],
    full_attention_layer_ids: List[int],
    mamba2_cache_params,
    model_context_len: int,
    extra_max_context_len: int,
    max_total_num_tokens: int,
    max_mamba_cache_size: int,
    max_num_reqs: int,
    enable_memory_saver: bool,
    enable_mamba_extra_buffer: bool,
    speculative_num_draft_tokens: Optional[int],
    disable_overlap_schedule: bool,
    mamba_full_memory_ratio: float,
    need_sort: bool,
) -> SharedPoolBundle:
    """Build the Mamba-hybrid shared-pool stack: SharedMemoryPool + the full
    sub-pool (`SharedMHATokenToKVPool` injected into a `HybridLinearKVPool`),
    `SharedHybridReqToTokenPool` with its `SharedMambaPool`, Relocator +
    binders, and the SharedMambaTokenToKVPoolAllocator.
    """
    from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool
    from sglang.srt.mem_cache.multi_ended_allocator import (
        SharedMambaTokenToKVPoolAllocator,
    )
    from sglang.srt.mem_cache.relocation_log import (
        Relocator,
        SlotBacktrack,
        SlotBacktrackBinder,
    )

    store_dtype = _store_dtype_for(kv_cache_dtype)
    full_spec = MHASubPoolSpec(
        name="full",
        layer_num=len(full_attention_layer_ids),
        head_num=head_num,
        head_dim=head_dim,
        store_dtype=store_dtype,
        grow_direction="up",
    )
    cp = mamba2_cache_params
    mamba_spec = MambaSubPoolSpec(
        name="mamba",
        layer_num=len(mamba_layer_ids),
        conv_state_shapes=tuple(tuple(s) for s in cp.shape.conv),
        conv_dtype=cp.dtype.conv,
        temporal_state_shape=tuple(cp.shape.temporal),
        temporal_dtype=cp.dtype.temporal,
        grow_direction="down",
    )
    total_bytes = (
        max_total_num_tokens * full_spec.entry_bytes()
        + max_mamba_cache_size * mamba_spec.entry_bytes()
    )
    shared_pool = SharedMemoryPool(
        total_bytes=total_bytes,
        sub_pool_specs=[full_spec, mamba_spec],
        device=device,
        enable_memory_saver=enable_memory_saver,
    )
    req_to_token_pool = SharedHybridReqToTokenPool(
        shared_buffer=shared_pool,
        mamba_sub_pool_name="mamba",
        size=max_num_reqs,
        max_context_len=model_context_len + extra_max_context_len,
        device=device,
        enable_memory_saver=enable_memory_saver,
        cache_params=mamba2_cache_params,
        mamba_layer_ids=mamba_layer_ids,
        enable_mamba_extra_buffer=enable_mamba_extra_buffer,
        speculative_num_draft_tokens=speculative_num_draft_tokens,
        enable_overlap_schedule=not disable_overlap_schedule,
        start_layer=start_layer,
    )

    # SharedMHATokenToKVPool reads dtype/head_num/head_dim/v_head_dim from
    # the spec (shared_pool.mha_spec("full")), and lifetime is owned by the
    # SharedMemoryPool — so neither `dtype` nor `enable_memory_saver` is in
    # its signature. Pass only what it accepts.
    shared_full_kv_pool = SharedMHATokenToKVPool(
        shared_buffer=shared_pool,
        sub_pool_name="full",
        start_layer=start_layer,
        end_layer=end_layer,
    )
    full_attn_layer_ids_for_pool = (
        [0] if is_draft_worker else list(full_attention_layer_ids)
    )
    token_to_kv_pool = HybridLinearKVPool(
        page_size=page_size,
        size=max_total_num_tokens,
        dtype=kv_cache_dtype,
        head_num=head_num,
        head_dim=head_dim,
        full_attention_layer_ids=full_attn_layer_ids_for_pool,
        enable_kvcache_transpose=False,
        device=device,
        mamba_pool=req_to_token_pool.mamba_pool,
        enable_memory_saver=enable_memory_saver,
        use_mla=use_mla_backend,
        start_layer=start_layer,
        full_kv_pool=shared_full_kv_pool,
    )

    full_backtrack = SlotBacktrack(size_total=shared_pool.max_slots("full"))
    mamba_backtrack = SlotBacktrack(size_total=shared_pool.max_slots("mamba"))
    relocation_log = Relocator(
        backtracks={"full": full_backtrack, "mamba": mamba_backtrack},
        device=device,
    )
    # Set the req_to_token_pool back-reference up-front so `apply()` (the
    # immediate-rewrite path called from MultiEndedAllocator._apply_relocations)
    # can read `req_to_token` without waiting for a first `flush()` call.
    relocation_log._req_to_token_pool_ref = req_to_token_pool
    full_binder = SlotBacktrackBinder(full_backtrack)
    mamba_binder = SlotBacktrackBinder(mamba_backtrack)
    token_to_kv_pool.full_kv_pool.slot_backtrack_binder = full_binder
    req_to_token_pool.mamba_pool.slot_backtrack_binder = mamba_binder
    wrap_req_to_token_pool_write(req_to_token_pool, full_binder)

    allocator = SharedMambaTokenToKVPoolAllocator(
        shared_buffer=shared_pool,
        kvcache=token_to_kv_pool,
        relocation_log=relocation_log,
        device=device,
        need_sort=need_sort,
    )

    logger.info(
        "[shared-pool] ============================================================"
    )
    logger.info(
        "[shared-pool] SHARED MEMORY POOL ENABLED -- path=Mamba hybrid"
    )
    logger.info(
        "[shared-pool]   full_layers=%d, mamba_layers=%d, head_num=%d, "
        "head_dim=%d, page_size=%d, use_mla=%s, is_draft_worker=%s",
        len(full_attention_layer_ids),
        len(mamba_layer_ids),
        head_num,
        head_dim,
        page_size,
        use_mla_backend,
        is_draft_worker,
    )
    logger.info(
        "[shared-pool]   total_bytes=%d, max_total_num_tokens=%d, "
        "max_mamba_cache_size=%d, max_num_reqs=%d, "
        "speculative_num_draft_tokens=%s",
        shared_pool.total_bytes,
        max_total_num_tokens,
        max_mamba_cache_size,
        max_num_reqs,
        speculative_num_draft_tokens,
    )
    logger.info(
        "[shared-pool]   mamba_full_memory_ratio=%s governs total budget only, "
        "not the runtime split.",
        mamba_full_memory_ratio,
    )
    logger.info(
        "[shared-pool]   binder wired:  full=YES  mamba=YES  "
        "req_to_token.write=WRAPPED"
    )
    logger.info(
        "[shared-pool] ============================================================"
    )
    return SharedPoolBundle(
        shared_memory_pool=shared_pool,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool=token_to_kv_pool,
        token_to_kv_pool_allocator=allocator,
        relocation_log=relocation_log,
    )
