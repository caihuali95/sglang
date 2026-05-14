"""SharedMemoryPool v2 — one physical byte buffer shared by ≥2 sub-pools.

See `shared_memory_pool_design.md` (same directory) for the full design. In short:

* One `uint8` buffer (`SharedMemoryPool._raw`) is split dynamically between
  sub-pools by `MultiEndedAllocator`s growing from opposite ends; `free` does
  eager compaction so each pool's allocated byte range stays hole-free.
* Per-slot layout is **slot/envelope-major** — a slot holds its data for all of
  that pool's layers in one contiguous byte envelope — so freeing a slot vacates
  a contiguous region the peer can grow into.
* Everything above the allocator stores **virtual** slot IDs (immutable for the
  slot's lifetime); the allocator keeps per-sub-pool `virtual_to_physical` /
  `physical_to_virtual` tables. On compaction `p_src → p_dst` only those two
  tables change — no reference rewriting. There is **no** `relocation_log` /
  `SlotBacktrack` / binder machinery (that was v1; this is v2).

Stage 1 (this revision) wires up the hybrid-Mamba family at `page_size == 1`.
`SharedSWAKVPool` / `init_shared_swa_pools` (Stage 2), N>2 sub-pools / DSV4
(Stage 4) and the disagg/spec bits (Stage 5) are not implemented here yet.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Optional, Tuple

import torch

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.mem_cache.memory_pool import (
    HybridReqToTokenPool,
    MambaPool,
    MHATokenToKVPool,
    move_kv_cache_native,
)
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

logger = logging.getLogger(__name__)

GB = 1024 * 1024 * 1024


def _prod(iterable) -> int:
    out = 1
    for x in iterable:
        out *= int(x)
    return out


def _store_dtype_for(kv_cache_dtype: torch.dtype) -> torch.dtype:
    if kv_cache_dtype in (torch.float8_e5m2, torch.float8_e4m3fn):
        return torch.uint8
    return kv_cache_dtype


# ---------------------------------------------------------------------------
# Sub-pool specs (pure per-slot layout math; no allocator/binder state)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class SubPoolSpec(ABC):
    """Abstract per-slot layout of one sub-pool in a `SharedMemoryPool`."""

    name: str
    layer_num: int
    grow_direction: str  # "up" or "down"

    def __post_init__(self):
        assert self.grow_direction in ("up", "down"), (
            f"grow_direction must be 'up' or 'down'; got {self.grow_direction!r}"
        )
        assert self.layer_num > 0, f"layer_num must be positive; got {self.layer_num}"

    @abstractmethod
    def entry_bytes(self) -> int:
        """Bytes consumed by one slot across all `layer_num` layers of this pool."""
        raise NotImplementedError


@dataclass(frozen=True, kw_only=True)
class MHASubPoolSpec(SubPoolSpec):
    """Per-slot layout of one MHA-shaped sub-pool. `v_head_dim` may differ from
    `head_dim` (matches `MHATokenToKVPool`); falls back to `head_dim` if None."""

    head_num: int
    head_dim: int
    store_dtype: torch.dtype
    v_head_dim: Optional[int] = None

    def __post_init__(self):
        super().__post_init__()
        assert self.head_num > 0, f"head_num must be positive; got {self.head_num}"
        assert self.head_dim > 0, f"head_dim must be positive; got {self.head_dim}"
        if self.v_head_dim is None:
            object.__setattr__(self, "v_head_dim", self.head_dim)
        assert self.v_head_dim > 0, f"v_head_dim must be positive; got {self.v_head_dim}"

    def k_row_bytes(self) -> int:
        return self.head_num * self.head_dim * self.store_dtype.itemsize

    def v_row_bytes(self) -> int:
        return self.head_num * self.v_head_dim * self.store_dtype.itemsize

    def entry_bytes(self) -> int:
        return self.layer_num * (self.k_row_bytes() + self.v_row_bytes())


@dataclass(frozen=True, kw_only=True)
class MambaSubPoolSpec(SubPoolSpec):
    """Per-slot layout of one Mamba-shaped sub-pool. `layer_num` = number of
    Mamba layers whose state is held (≡ `MambaPool` `num_mamba_layers`)."""

    conv_state_shapes: Tuple[Tuple[int, ...], ...]  # one shape per conv tensor
    conv_dtype: torch.dtype
    temporal_state_shape: Tuple[int, ...]
    temporal_dtype: torch.dtype

    def __post_init__(self):
        super().__post_init__()
        assert len(self.conv_state_shapes) > 0, "conv_state_shapes must be non-empty"

    def conv_row_bytes(self, idx: int) -> int:
        return _prod(self.conv_state_shapes[idx]) * self.conv_dtype.itemsize

    def temporal_row_bytes(self) -> int:
        return _prod(self.temporal_state_shape) * self.temporal_dtype.itemsize

    def entry_bytes(self) -> int:
        total = 0
        for i in range(len(self.conv_state_shapes)):
            total += self.layer_num * self.conv_row_bytes(i)
        total += self.layer_num * self.temporal_row_bytes()
        return total


# ---------------------------------------------------------------------------
# SharedMemoryPool — the byte buffer + the strided per-sub-pool views
# ---------------------------------------------------------------------------


class SharedMemoryPool:
    """One physical `uint8` byte buffer shared by 2 (Stage 4: ≥2) sub-pools.

    Each sub-pool exposes its per-layer K/V or conv/temporal tensors as strided
    views into the raw buffer (envelope layout, anchored at byte 0). Allocators
    coordinate to keep their byte ranges disjoint; this class does not track usage.
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
            f"{len(sub_pool_specs)} (N>2 is Stage 4)"
        )
        names = [s.name for s in sub_pool_specs]
        assert len(set(names)) == 2, f"sub-pool names must be unique; got {names}"
        directions = sorted(s.grow_direction for s in sub_pool_specs)
        assert directions == ["down", "up"], (
            f"SharedMemoryPool needs one grow-up and one grow-down sub-pool; "
            f"got {directions}"
        )

        self.device = device
        self.total_bytes = total_bytes
        self.sub_pool_specs = sub_pool_specs
        self._specs_by_name: Dict[str, SubPoolSpec] = {s.name: s for s in sub_pool_specs}

        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            self._raw = torch.empty(total_bytes, dtype=torch.uint8, device=device)
        self._raw.zero_()  # unset slots read as zeros (matches non-shared behavior)

        self._max_slots: Dict[str, int] = {}
        self._anchor_bytes: Dict[str, int] = {}
        self._min_slot_index: Dict[str, int] = {}
        # MHA views: (k_buffer, v_buffer); Mamba views: (conv_state_list, temporal_state)
        self._mha_views: Dict[str, Tuple[List[torch.Tensor], List[torch.Tensor]]] = {}
        self._mamba_views: Dict[str, Tuple[List[torch.Tensor], torch.Tensor]] = {}

        # Slot-0 padding-write safety: every pool's slot-0 dummy writes land in
        # raw bytes [0, entry_i). Their union is [0, entry_max). Each pool's first
        # allocatable slot index is chosen so its real data starts at ≥ entry_max.
        entry_max = max(s.entry_bytes() for s in sub_pool_specs)

        for spec in sub_pool_specs:
            entry_bytes = spec.entry_bytes()
            max_slots = total_bytes // entry_bytes
            min_slot_index = (entry_max + entry_bytes - 1) // entry_bytes  # ceil
            if max_slots <= min_slot_index:
                raise RuntimeError(
                    f"SharedMemoryPool: sub-pool {spec.name!r} fits only {max_slots} "
                    f"slots in {total_bytes} bytes, but min_slot_index={min_slot_index} "
                    f"leaves no room for real data. Increase total_bytes."
                )
            anchor = 0  # all anchors are 0 (uniform view construction)
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
            "[shared-pool] SharedMemoryPool allocated: total_bytes=%.2f GB (=%d B), "
            "%d sub-pool(s)",
            total_bytes / GB,
            total_bytes,
            len(sub_pool_specs),
        )
        for s in sub_pool_specs:
            logger.info(
                "[shared-pool]   sub-pool %r: kind=%s, layer_num=%d, grow=%s, "
                "entry_bytes=%d, max_slots=%d, min_slot_index=%d (slots [0,%d) reserved)",
                s.name,
                type(s).__name__,
                s.layer_num,
                s.grow_direction,
                s.entry_bytes(),
                self._max_slots[s.name],
                self._min_slot_index[s.name],
                self._min_slot_index[s.name],
            )

    # -- introspection --

    def spec(self, name: str) -> SubPoolSpec:
        return self._specs_by_name[name]

    def mha_spec(self, name: str) -> MHASubPoolSpec:
        s = self._specs_by_name[name]
        assert isinstance(s, MHASubPoolSpec), (
            f"sub-pool {name!r} is {type(s).__name__}, expected MHASubPoolSpec"
        )
        return s

    def mamba_spec(self, name: str) -> MambaSubPoolSpec:
        s = self._specs_by_name[name]
        assert isinstance(s, MambaSubPoolSpec), (
            f"sub-pool {name!r} is {type(s).__name__}, expected MambaSubPoolSpec"
        )
        return s

    def max_slots(self, name: str) -> int:
        return self._max_slots[name]

    def min_slot_index(self, name: str) -> int:
        return self._min_slot_index[name]

    def anchor_bytes(self, name: str) -> int:
        anchor = self._anchor_bytes[name]
        assert anchor == 0, f"current design assumes all anchors are 0; got {anchor}"
        return anchor

    def mha_views_for(self, name: str) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        return self._mha_views[name]

    def mamba_views_for(self, name: str) -> Tuple[List[torch.Tensor], torch.Tensor]:
        return self._mamba_views[name]

    # -- view construction (envelope layout) --

    def _build_mha_views(
        self, spec: MHASubPoolSpec, anchor_bytes: int, max_slots: int
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Inside one slot's envelope: per-layer interleaved K/V (K first, V second
        per layer): [L0_K | L0_V | L1_K | L1_V | ... | L_{N-1}_V]. K and V may have
        different per-head dims; the per-layer (k_row + v_row) byte step encodes it.
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
            k_offset_elems = anchor_elems + (l * layer_step_bytes) // itemsize
            v_offset_elems = anchor_elems + (l * layer_step_bytes + k_row_bytes) // itemsize
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
        """Per-slot envelope: [conv[0] rows × layers][conv[1] rows × layers]...
        [temporal rows × layers]. Each returned view has shape
        (num_layers, max_slots, *inner_shape) — matches `MambaPool.State.conv[i]`
        / `.temporal` conventions.
        """
        entry_bytes = spec.entry_bytes()
        N = spec.layer_num

        def c_strides(shape):
            strides = []
            acc = 1
            for s in reversed(shape):
                strides.append(acc)
                acc *= int(s)
            return tuple(reversed(strides))

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
                as_conv_dtype, size=shape, stride=stride, storage_offset=offset_elems,
            )
            conv_views.append(view)
            offset_bytes_within_entry += N * inner_shape_bytes

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
            as_dtype, size=shape, stride=stride, storage_offset=offset_elems,
        )
        return conv_views, temporal_view


# ---------------------------------------------------------------------------
# SharedMHATokenToKVPool — MHA pool whose buffers are views into a SharedMemoryPool
# ---------------------------------------------------------------------------


class SharedMHATokenToKVPool(MHATokenToKVPool):
    """MHA KV pool whose `k_buffer` / `v_buffer` are strided views into a
    `SharedMemoryPool`. Buffer lifetime is owned by the SharedMemoryPool;
    relocation uses the native move (strided views break the tiled Triton kernel
    that assumes stride == row bytes).

    When `attach_allocator` has been called (always, for the shared-pool path),
    `set_kv_buffer` receives **virtual** slot ids and translates them to physical
    via the allocator's `virtual_to_physical` table — or uses the per-batch
    precomputed `set_loc(...)` value when available.
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
        spec = shared_buffer.mha_spec(sub_pool_name)
        k_buffer, v_buffer = shared_buffer.mha_views_for(sub_pool_name)
        max_slots = shared_buffer.max_slots(sub_pool_name)

        self._shared_buffer = shared_buffer
        self._sub_pool_name = sub_pool_name
        self._preallocated_k_buffer = k_buffer
        self._preallocated_v_buffer = v_buffer
        self._external_allocator = None  # set via attach_allocator

        super().__init__(
            size=max_slots - 1,  # -1 for reserved slot 0
            page_size=1,
            dtype=spec.store_dtype,
            head_num=spec.head_num,
            head_dim=spec.head_dim,
            layer_num=spec.layer_num,
            device=shared_buffer.device,
            enable_memory_saver=False,  # buffer owned by SharedMemoryPool
            v_head_dim=spec.v_head_dim,
            start_layer=start_layer,
            end_layer=end_layer,
            enable_alt_stream=enable_alt_stream,
            enable_kv_cache_copy=False,  # strided views — force native move
        )

    # -- buffer lifecycle overrides --

    def _create_buffers(self):
        self.k_buffer = self._preallocated_k_buffer
        self.v_buffer = self._preallocated_v_buffer
        # data_ptrs / data_strides are populated for any external code that
        # inspects them; we force the native move path so they are not consumed.
        self.k_data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.k_buffer], dtype=torch.uint64, device=self.device,
        )
        self.v_data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.v_buffer], dtype=torch.uint64, device=self.device,
        )
        self.data_ptrs = torch.cat([self.k_data_ptrs, self.v_data_ptrs], dim=0)
        self.data_strides = torch.tensor(
            [x.stride(0) * x.dtype.itemsize for x in (self.k_buffer + self.v_buffer)],
            device=self.device,
        )

    def _clear_buffers(self):
        # Lifetime owned by SharedMemoryPool; do not delete the views.
        pass

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        # tgt_loc / src_loc are PHYSICAL slot ids (passed by the allocator's
        # _compact_pending). Force the native move (strided views).
        if tgt_loc.numel() == 0:
            return
        move_kv_cache_native(self.k_buffer, self.v_buffer, tgt_loc, src_loc)

    def get_kv_size_bytes(self):
        # The shared buffer's total size is logged by SharedMemoryPool once;
        # per-sub-pool accounting would double-count.
        return 0, 0

    # -- virtual->physical wiring --

    def attach_allocator(self, allocator) -> None:
        """Wire the `MultiEndedAllocator` whose `virtual_to_physical` table this
        pool uses to translate slot ids in `set_kv_buffer`."""
        self._external_allocator = allocator

    def set_kv_buffer(
        self,
        layer,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale=None,
        v_scale=None,
        layer_id_override: Optional[int] = None,
    ):
        # `loc` arrives as VIRTUAL slot ids (the shared-pool path); translate to
        # physical here. This is robust to sub-batched callers that pass a slice
        # of out_cache_loc (radix_attention.unified_attention_with_output).
        if self._external_allocator is not None:
            loc = self._external_allocator.virtual_to_physical[loc]
        super().set_kv_buffer(
            layer, loc, cache_k, cache_v, k_scale, v_scale,
            layer_id_override=layer_id_override,
        )


# ---------------------------------------------------------------------------
# SharedMambaPool — Mamba state pool whose buffers are views into a SharedMemoryPool
# ---------------------------------------------------------------------------


class SharedMambaPool(MambaPool):
    """Mamba pool whose `conv_state` / `temporal_state` are strided views into a
    `SharedMemoryPool`. alloc / free / clear / available_size delegate to an
    external `MultiEndedAllocator` (the id-owner of the per-request virtual-id
    space) attached via `attach_allocator`.

    Does NOT call `super().__init__()` (that allocates fresh tensors). Replicates
    the minimal `MambaPool` state against the shared buffer so inherited methods
    work. Slot-id-bearing public methods (`copy_from`, `fork_from`, `get_cpu_copy`,
    `load_cpu_copy`, `is_slot_allocated`) take **virtual** ids and translate.
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
        spec = shared_buffer.mamba_spec(sub_pool_name)
        assert spec.layer_num == len(mamba_layer_ids)
        conv_views, temporal_view = shared_buffer.mamba_views_for(sub_pool_name)
        max_slots = shared_buffer.max_slots(sub_pool_name)

        self._shared_buffer = shared_buffer
        self._sub_pool_name = sub_pool_name
        self._external_allocator = None  # set via attach_allocator

        # Replicate the state MambaPool.__init__ would have set.
        self._max_size = max_slots - 1  # -1 for reserved slot 0
        self.size = self._max_size
        self.device = shared_buffer.device
        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )
        self.enable_custom_mem_pool = False
        self.custom_mem_pool = None
        self.num_mamba_layers = spec.layer_num
        # Note: layer_transfer_counter is owned by HybridReqToTokenPool / MHA pools,
        # NOT MambaPool — don't add it here.

        assert conv_views[0].shape[0] == self.num_mamba_layers, (
            f"conv_views layers={conv_views[0].shape[0]} vs expected {self.num_mamba_layers}"
        )
        assert conv_views[0].shape[1] == self._max_size + 1, (
            f"conv_views slots={conv_views[0].shape[1]} vs expected {self._max_size + 1}"
        )

        # Optional per-draft-token intermediate buffers — different outer size
        # (spec_state_size+1), so NOT in the shared byte buffer; allocate locally.
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
                            cshape[0],
                            cshape[1],
                        ),
                        dtype=conv_dtype,
                        device=shared_buffer.device,
                    )
                    for cshape in conv_state_shape
                ]
            self.mamba_cache = self.SpeculativeState(
                conv=list(conv_views),
                temporal=temporal_view,
                intermediate_ssm=intermediate_ssm_state_cache,
                intermediate_conv_window=intermediate_conv_window_cache,
            )
        else:
            self.mamba_cache = self.State(conv=list(conv_views), temporal=temporal_view)

        self.mem_usage = shared_buffer.total_bytes / GB
        logger.info(
            "[shared-pool] SharedMambaPool(%s) wrapped shared buffer: max_slots=%d, "
            "num_mamba_layers=%d",
            sub_pool_name,
            max_slots,
            self.num_mamba_layers,
        )

    # -- allocator delegation --

    def attach_allocator(self, allocator) -> None:
        self._external_allocator = allocator

    @property
    def free_slots(self) -> torch.Tensor:
        """Physical-slot-space free list, derived from the allocator's watermark —
        consulted by the scheduler's leak checker. Falls back to the pre-shared
        convention before `attach_allocator` runs."""
        if self._external_allocator is None:
            return torch.arange(
                1, self._max_size + 1, dtype=torch.int64, device=self.device
            )
        alloc = self._external_allocator
        if alloc.grow_direction == "up":
            start, end = alloc.watermark_physical, alloc.max_slots
        else:
            start, end = alloc.min_slot_index, alloc.watermark_physical + 1
        if start >= end:
            return torch.empty((0,), dtype=torch.int64, device=self.device)
        return torch.arange(start, end, dtype=torch.int64, device=self.device)

    def available_size(self) -> int:
        """Slot-conservation free count: `size - allocated_count`.

        This is the view the scheduler's leak check expects — it composes into
        the invariant `available + evictable + protected + session_held == size`
        (`scheduler_runtime_checker_mixin._check_mamba_pool` /
        `_check_pool_invariant`), where `num_used = size - available - evictable`
        must reduce to `allocated_count - evictable`. The watchdog `dump_info`
        path can exercise this during an active workload, not just on idle, so
        returning anything byte-coordinated here would surface false leaks.

        It is deliberately NOT the right value for the alloc planner — when the
        peer pool has consumed the byte buffer, slot-wise availability overstates
        what `alloc(N)` will accept. Use `schedulable_available_size()` for that.
        """
        if self._external_allocator is None:
            return self._max_size
        return self._max_size - self._external_allocator.allocated_count()

    def schedulable_available_size(self) -> int:
        """Byte-coordinated free count — the contract is
        `schedulable_available_size() >= N  =>  alloc(N) succeeds`.

        Used by `common.alloc_req_slots` to decide whether to trigger mamba
        eviction before calling `req_to_token_pool.alloc(reqs)`. May be smaller
        than `available_size()` when the peer pool's byte usage tightens this
        side's effective capacity (in which case the planner needs to evict
        even though slot-wise there's room).
        """
        if self._external_allocator is None:
            return self._max_size
        return self._external_allocator.available_size()

    def alloc(self, need_size: int):
        if self._external_allocator is None:
            raise RuntimeError("SharedMambaPool.alloc called before attach_allocator")
        v = self._external_allocator.alloc(need_size)  # VIRTUAL ids
        if v is None:
            return None
        # Clear newly-allocated conv/temporal rows AT THE PHYSICAL SLOTS. The
        # allocator's `alloc` ran on the current stream (schedule_stream); the
        # bind's v2p write is on the same stream as this gather, so single-
        # stream ordering makes it visible.
        p = self._external_allocator.virtual_to_physical[v]
        for i in range(len(self.mamba_cache.conv)):
            t = self.mamba_cache.conv[i]
            z = torch.zeros(1, dtype=t.dtype, device=t.device).expand(
                t.shape[0], int(p.numel()), *t.shape[2:]
            )
            t[:, p] = z
        t = self.mamba_cache.temporal
        z = torch.zeros(1, dtype=t.dtype, device=t.device).expand(
            t.shape[0], int(p.numel()), *t.shape[2:]
        )
        t[:, p] = z
        return v

    def free(self, free_index: torch.Tensor):
        if self._external_allocator is None:
            raise RuntimeError("SharedMambaPool.free called before attach_allocator")
        self._external_allocator.free(free_index)  # VIRTUAL ids

    def clear(self):
        if self._external_allocator is None:
            raise RuntimeError("SharedMambaPool.clear called before attach_allocator")
        self._external_allocator.clear()

    # -- slot-id-bearing methods: translate virtual -> physical --

    def _copy_from_physical(self, src_index: torch.Tensor, dst_index: torch.Tensor):
        """Un-translated copy on PHYSICAL slot ids — used by the allocator's
        `_compact_pending` (which already has physical ids)."""
        MambaPool.copy_from(self, src_index, dst_index)

    def copy_from(self, src_index: torch.Tensor, dst_index: torch.Tensor):
        # Public: callers (radix-cache COW) pass VIRTUAL ids.
        v2p = self._external_allocator.virtual_to_physical
        return MambaPool.copy_from(self, v2p[src_index], v2p[dst_index])

    def fork_from(self, src_index: torch.Tensor):
        dst_v = self.alloc(1)  # VIRTUAL
        if dst_v is None:
            return None
        self.copy_from(src_index, dst_v)  # translates both
        return dst_v

    def get_cpu_copy(self, indices):
        v2p = self._external_allocator.virtual_to_physical
        return MambaPool.get_cpu_copy(self, v2p[indices])

    def load_cpu_copy(self, mamba_cache_cpu, indices):
        v2p = self._external_allocator.virtual_to_physical
        return MambaPool.load_cpu_copy(self, mamba_cache_cpu, v2p[indices])

    def is_slot_allocated(self, slot) -> bool:
        """Whether VIRTUAL id `slot` is currently in use."""
        if self._external_allocator is None:
            return False
        return self._external_allocator.is_slot_allocated(int(slot))

    def allocator_state_str(self) -> str:
        if self._external_allocator is None:
            return "<no external allocator attached>"
        return self._external_allocator.allocator_state_str()


# ---------------------------------------------------------------------------
# SharedHybridReqToTokenPool — HybridReqToTokenPool whose MambaPool is shared
# ---------------------------------------------------------------------------


class SharedHybridReqToTokenPool(HybridReqToTokenPool):
    """`HybridReqToTokenPool` whose `mamba_pool` is a `SharedMambaPool` aliasing a
    shared byte buffer. Everything else (alloc/get_mamba_indices/free_mamba_cache/
    ping-pong) is inherited unchanged — `req.mamba_pool_idx`,
    `req_index_to_mamba_index_mapping` and `TreeNode.mamba_value` now hold VIRTUAL
    mamba ids, which is exactly what they should hold. Adds `translate_mamba_indices`
    for the attention backend's per-batch virtual->physical translation.
    """

    def __init__(
        self,
        *,
        shared_buffer: SharedMemoryPool,
        mamba_sub_pool_name: str,
        size: int,
        mamba_spec_state_size: int,
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
        self._shared_mamba_size = shared_buffer.max_slots(mamba_sub_pool_name) - 1
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

    def _init_mamba_pool(
        self,
        mamba_size: int,
        mamba_spec_state_size: int,
        cache_params,
        mamba_layer_ids: List[int],
        device: str,
        enable_mamba_extra_buffer: bool,
        speculative_num_draft_tokens: Optional[int] = None,
    ):
        # Parent's contract: `mamba_size` is the source of truth for the mamba
        # pool's slot count. Under the shared pool that source of truth is
        # `SharedMemoryPool.max_slots("mamba") - 1` (= self._shared_mamba_size,
        # which __init__ passed as `mamba_size`). Re-assert the equality so a
        # future signature drift in the parent surfaces here, not later.
        assert mamba_size == self._shared_mamba_size, (
            f"SharedHybridReqToTokenPool._init_mamba_pool: mamba_size={mamba_size} "
            f"!= shared_buffer.max_slots({self._mamba_sub_pool_name!r}) - 1 "
            f"= {self._shared_mamba_size}"
        )
        # `cache_params` is consumed indirectly via the SharedMemoryPool's
        # MambaSubPoolSpec (built by init_shared_mamba_pools from the same
        # cache_params). Sanity-check the layer count matches.
        assert len(cache_params.layers) >= len(mamba_layer_ids), (
            f"cache_params.layers ({len(cache_params.layers)}) cannot supply "
            f"{len(mamba_layer_ids)} mamba layer ids"
        )
        # SharedMambaPool reads conv/temporal shapes from its sub-pool spec
        # (shared_buffer.mamba_spec(mamba_sub_pool_name)).
        self.mamba_pool = SharedMambaPool(
            shared_buffer=self._shared_buffer,
            sub_pool_name=self._mamba_sub_pool_name,
            spec_state_size=mamba_spec_state_size,
            mamba_layer_ids=mamba_layer_ids,
            enable_memory_saver=self.enable_memory_saver,
            speculative_num_draft_tokens=speculative_num_draft_tokens,
        )
        self.mamba_map = {layer_id: i for i, layer_id in enumerate(mamba_layer_ids)}
        self.device = device
        # Mirror the parent's sizing: indexed by req_pool_idx, so by the
        # req_to_token buffer's first dim — which is `self.size + 1`, NOT
        # `self.size` (ReqToTokenPool reserves index 0 as the padding row;
        # see ReqToTokenPool.__init__'s `_alloc_size = size + 1`). Using
        # `self.size` directly here would under-size the mapping by one row.
        req_pool_size = self.req_to_token.shape[0]
        self.req_index_to_mamba_index_mapping: torch.Tensor = torch.zeros(
            req_pool_size, dtype=torch.int32, device=self.device
        )
        if enable_mamba_extra_buffer:
            self.req_index_to_mamba_ping_pong_track_buffer_mapping: torch.Tensor = (
                torch.zeros(
                    (req_pool_size, self.mamba_ping_pong_track_buffer_size),
                    dtype=torch.int32,
                    device=self.device,
                )
            )

    def translate_mamba_indices(self, virtual_ids: torch.Tensor) -> torch.Tensor:
        """Virtual per-request mamba ids -> physical slot ids. Called once per
        batch by the linear-attention backend's metadata build."""
        v2p = self.mamba_pool._external_allocator.virtual_to_physical
        return v2p[virtual_ids].to(torch.int32)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class SharedPoolBundle(NamedTuple):
    shared_memory_pool: SharedMemoryPool
    token_to_kv_pool: object  # HybridLinearKVPool
    token_to_kv_pool_allocator: object  # SharedMambaTokenToKVPoolAllocator
    req_to_token_pool: object  # SharedHybridReqToTokenPool


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
    need_sort: bool,
    mamba_full_memory_ratio: Optional[float] = None,  # informational only
    forward_stream: Optional[torch.cuda.Stream] = None,
) -> SharedPoolBundle:
    """Build the Mamba-hybrid shared-pool stack: `SharedMemoryPool` (full + mamba
    sub-pools), `SharedHybridReqToTokenPool` (with its `SharedMambaPool`),
    `SharedMHATokenToKVPool` injected into a `HybridLinearKVPool`, and the
    `SharedMambaTokenToKVPoolAllocator`."""
    from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool
    from sglang.srt.mem_cache.multi_ended_allocator import (
        SharedMambaTokenToKVPoolAllocator,
    )

    assert not use_mla_backend, (
        "shared memory pool Stage 1 does not support MLA-hybrid-Mamba yet"
    )
    assert page_size == 1, "shared memory pool Stage 1 supports page_size == 1 only"

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
        conv_state_shapes=tuple(tuple(int(x) for x in s) for s in cp.shape.conv),
        conv_dtype=cp.dtype.conv,
        temporal_state_shape=tuple(int(x) for x in cp.shape.temporal),
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
        # Mirror model_runner_kv_cache_mixin._init_pools: the parent's
        # `mamba_spec_state_size` is `max_num_reqs` — it sizes the spec-decode
        # intermediate-state buffers' outer dimension (one slot per concurrent
        # request).
        mamba_spec_state_size=max_num_reqs,
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
    allocator = SharedMambaTokenToKVPoolAllocator(
        shared_buffer=shared_pool,
        kvcache=token_to_kv_pool,
        device=device,
        need_sort=need_sort,
        forward_stream=forward_stream,
    )

    logger.info(
        "[shared-pool] ============================================================"
    )
    logger.info("[shared-pool] SHARED MEMORY POOL ENABLED -- path=Mamba hybrid")
    logger.info(
        "[shared-pool]   full_layers=%d, mamba_layers=%d, head_num=%d, head_dim=%d, "
        "page_size=%d, is_draft_worker=%s",
        len(full_attention_layer_ids),
        len(mamba_layer_ids),
        head_num,
        head_dim,
        page_size,
        is_draft_worker,
    )
    logger.info(
        "[shared-pool]   total_bytes=%d, max_total_num_tokens=%d, max_mamba_cache_size=%d, "
        "max_num_reqs=%d, speculative_num_draft_tokens=%s",
        total_bytes,
        max_total_num_tokens,
        max_mamba_cache_size,
        max_num_reqs,
        speculative_num_draft_tokens,
    )
    if mamba_full_memory_ratio is not None:
        logger.info(
            "[shared-pool]   mamba_full_memory_ratio=%s governs the total budget only, "
            "not the runtime split.",
            mamba_full_memory_ratio,
        )
    logger.info(
        "[shared-pool] ============================================================"
    )
    return SharedPoolBundle(
        shared_memory_pool=shared_pool,
        token_to_kv_pool=token_to_kv_pool,
        token_to_kv_pool_allocator=allocator,
        req_to_token_pool=req_to_token_pool,
    )
