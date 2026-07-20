# Copyright 2023-2026 SGLang Team
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
"""UnifiedKVPool — one physical `uint8` byte buffer shared by N sub-pools.

Two end sub-pools grow from opposite ends (their `MultiEndedAllocator`s meet in
the middle); optional "float" middle sub-pools live between the two frontiers.
Eager-compacting `free` keeps each pool's byte range hole-free. Layout is
envelope-major (a slot's data for all its layers in one contiguous byte
envelope) so a freed slot vacates a region a neighbor can grow into. Everything
above the allocator stores virtual slot IDs; the allocator owns the
per-sub-pool virtual<->physical tables and compaction only mutates those (no
reference rewriting).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import msgspec
import torch
import triton
from torch.profiler import record_function

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.environ import envs
from sglang.srt.mem_cache.layout.page_major import (
    build_page_major_mamba_views,
    build_page_major_mha_views,
    build_page_major_mla_views,
    build_spec_state_views,
    mla_entry_bytes,
    spec_state_entry_bytes,
)
from sglang.srt.mem_cache.memory_pool import (
    HybridReqToTokenPool,
    KVCache,
    KVWriteLoc,
    MambaPool,
    MHATokenToKVPool,
    MLATokenToKVPool,
    move_kv_cache_native,
    unwrap_write_loc,
)
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.mem_cache.triton_ops.cache_move import store_cache_4d_kernel
from sglang.srt.mem_cache.triton_ops.mla_buffer import (
    get_mla_kv_buffer_page_major,
    set_mla_kv_buffer_page_major,
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


def _align_up(n: int, alignment: int) -> int:
    return (n + alignment - 1) // alignment * alignment


def _as_row_contiguous(t: torch.Tensor) -> torch.Tensor:
    """Make a KV source satisfy `store_cache_4d_kernel`'s row contract.

    The kernel flattens (head_num, head_dim) into one ROW_DIM addressed by a
    linear offset, so a row's elements must be adjacent in memory: for a 3-D
    (N, head_num, head_dim) source that means `stride[-1] == 1` and
    `stride[-2] == head_dim`. Only `stride(0)` is passed in, so any other
    layout is read with the wrong element spacing -- the store then lands the
    wrong bytes at the right slot, which no location-based check can see.

    A model may legitimately hand us a non-contiguous k/v (a slice of a fused
    QKV projection, a transpose). Copy those into a compact row layout instead
    of rejecting them; rows already compact are returned untouched, so the
    common path stays copy-free. `stride(0)` is deliberately NOT constrained --
    the kernel takes it as an argument, so a row-compact slice of a wider
    tensor is fine.
    """
    if t.ndim < 2:
        return t
    if t.stride(-1) == 1 and t.stride(-2) == t.shape[-1]:
        return t
    return t.contiguous()


def fused_entry_bytes(host_entry_bytes: int, draft_region: "MHARegionGeometry") -> int:
    """Per-slot bytes of a fused `[host KV | pad | draft KV]` envelope.

    SINGLE SOURCE OF TRUTH for both the physical layout
    (`MHASubPoolSpec.entry_bytes`) and the byte budget (the pool
    configurators' per-token cell) — the two must never drift, or the solve
    under- or over-provisions the buffer the views are built over."""
    return (
        _align_up(host_entry_bytes, draft_region.store_dtype.itemsize)
        + draft_region.entry_bytes()
    )


class MHARegionGeometry(msgspec.Struct, frozen=True, kw_only=True):
    """Geometry of one MHA-shaped byte region inside a fused slot envelope.

    Describes the DRAFT model's per-slot KV footprint when its KV is fused
    into a host sub-pool's entry (`[target KV | draft KV]`). Mirrors
    `MHASubPoolSpec`'s MHA fields but
    carries the draft's OWN geometry — EAGLE3/DFLASH drafts are separate
    checkpoints whose head_num/head_dim differ from the target's.
    `msgspec.Struct`, deliberately outside the grandfathered `SubPoolSpec`
    `@dataclass` hierarchy (same convention as `SpecStateSubPoolSpec`).
    """

    layer_num: int
    head_num: int
    head_dim: int
    v_head_dim: int
    store_dtype: torch.dtype

    def validate(self) -> None:
        assert self.layer_num > 0, f"layer_num must be positive; got {self.layer_num}"
        assert self.head_num > 0, f"head_num must be positive; got {self.head_num}"
        assert self.head_dim > 0, f"head_dim must be positive; got {self.head_dim}"
        assert (
            self.v_head_dim > 0
        ), f"v_head_dim must be positive; got {self.v_head_dim}"

    def k_row_bytes(self) -> int:
        return self.head_num * self.head_dim * self.store_dtype.itemsize

    def v_row_bytes(self) -> int:
        return self.head_num * self.v_head_dim * self.store_dtype.itemsize

    def entry_bytes(self) -> int:
        """Bytes for one slot's draft region across all draft layers."""
        return self.layer_num * (self.k_row_bytes() + self.v_row_bytes())


@dataclass(frozen=True, kw_only=True)
class SubPoolSpec(ABC):
    """Abstract per-slot layout of one sub-pool in a `UnifiedKVPool`."""

    name: str
    layer_num: int
    grow_direction: str  # "up" | "down"

    def __post_init__(self):
        assert self.grow_direction in (
            "up",
            "down",
        ), f"grow_direction must be 'up' or 'down'; got {self.grow_direction!r}"
        assert self.layer_num > 0, f"layer_num must be positive; got {self.layer_num}"

    @abstractmethod
    def entry_bytes(self) -> int:
        """Bytes for one slot across all `layer_num` layers."""
        raise NotImplementedError

    @abstractmethod
    def get_dtype(self) -> torch.dtype:
        """Storage dtype (informational). Multi-dtype subclasses return the dominant buffer's."""
        raise NotImplementedError


@dataclass(frozen=True, kw_only=True)
class MHASubPoolSpec(SubPoolSpec):
    """Per-slot layout of one MHA-shaped sub-pool. `v_head_dim` defaults to `head_dim`.

    With `draft_region` set, each slot is a FUSED envelope
    `[host KV | pad | draft KV]`: the draft model's KV rides at byte offset
    `draft_region_offset_bytes()` inside
    every slot, sharing this pool's slot ids, allocator, v2p mapping, and
    compaction moves. `draft_region is None` keeps the layout byte-identical
    to the pre-fusion one.
    """

    head_num: int
    head_dim: int
    store_dtype: torch.dtype
    v_head_dim: Optional[int] = None
    draft_region: Optional[MHARegionGeometry] = None

    def __post_init__(self):
        super().__post_init__()
        assert self.head_num > 0, f"head_num must be positive; got {self.head_num}"
        assert self.head_dim > 0, f"head_dim must be positive; got {self.head_dim}"
        if self.v_head_dim is None:
            object.__setattr__(self, "v_head_dim", self.head_dim)
        assert (
            self.v_head_dim > 0
        ), f"v_head_dim must be positive; got {self.v_head_dim}"
        if self.draft_region is not None:
            self.draft_region.validate()

    def k_row_bytes(self) -> int:
        return self.head_num * self.head_dim * self.store_dtype.itemsize

    def v_row_bytes(self) -> int:
        return self.head_num * self.v_head_dim * self.store_dtype.itemsize

    def host_entry_bytes(self) -> int:
        """Host (target-only) bytes for one slot — the pre-fusion entry."""
        return self.layer_num * (self.k_row_bytes() + self.v_row_bytes())

    def draft_region_offset_bytes(self) -> int:
        """Byte offset of the draft region within one slot's fused envelope
        (host entry aligned up to the draft dtype's itemsize so the draft
        views' element-space `storage_offset` never truncates)."""
        assert self.draft_region is not None
        return _align_up(
            self.host_entry_bytes(), self.draft_region.store_dtype.itemsize
        )

    def entry_bytes(self) -> int:
        if self.draft_region is None:
            return self.host_entry_bytes()
        # Via the shared helper so the configurators' cell math can never
        # drift from the physical layout.
        return fused_entry_bytes(self.host_entry_bytes(), self.draft_region)

    # Page-major byte math: within a page block K/V group per layer
    # [L0_K*ps | L0_V*ps | L1_K*ps | ...]; at ps==1 this collapses to the per-slot envelope.

    def page_bytes(self, page_size: int) -> int:
        return page_size * self.entry_bytes()

    def layer_k_offset_in_page(self, layer_id: int, page_size: int) -> int:
        return layer_id * page_size * (self.k_row_bytes() + self.v_row_bytes())

    def layer_v_offset_in_page(self, layer_id: int, page_size: int) -> int:
        return (
            self.layer_k_offset_in_page(layer_id, page_size)
            + page_size * self.k_row_bytes()
        )

    def get_dtype(self) -> torch.dtype:
        return self.store_dtype


@dataclass(frozen=True, kw_only=True)
class MambaSubPoolSpec(SubPoolSpec):
    """Per-slot layout of one Mamba-shaped sub-pool."""

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

    def get_dtype(self) -> torch.dtype:
        return self.conv_dtype  # representative state dtype; matches MambaPool.dtype


class SpecStateSubPoolSpec(msgspec.Struct, frozen=True, kw_only=True):
    """Per-slot layout of the spec-decode intermediate-state "float" sub-pool.

    One slot = one verify-batch row's intermediate SSM states + conv windows
    across all mamba layers and draft steps (DENSE conv layout). A float pool
    lives between the two end pools' frontiers as one contiguous, freely
    relocatable band. `msgspec.Struct`, deliberately outside the grandfathered
    `SubPoolSpec` `@dataclass` hierarchy — the sweep type-dispatches.
    """

    name: str
    layer_num: int
    num_draft_tokens: int
    conv_window_shapes: Tuple[Tuple[int, ...], ...]  # one shape per conv tensor
    conv_dtype: torch.dtype
    ssm_state_shape: Tuple[int, ...]
    ssm_dtype: torch.dtype
    grow_direction: str = "float"

    def validate(self) -> None:
        assert self.grow_direction == "float", (
            f"SpecStateSubPoolSpec must be a float middle pool; "
            f"got {self.grow_direction!r}"
        )
        assert self.layer_num > 0, f"layer_num must be positive; got {self.layer_num}"
        assert (
            self.num_draft_tokens > 0
        ), f"num_draft_tokens must be positive; got {self.num_draft_tokens}"
        assert len(self.conv_window_shapes) > 0, "conv_window_shapes must be non-empty"

    def entry_bytes(self) -> int:
        return spec_state_entry_bytes(
            layer_num=self.layer_num,
            num_draft_tokens=self.num_draft_tokens,
            conv_window_shapes=self.conv_window_shapes,
            conv_dtype=self.conv_dtype,
            ssm_state_shape=self.ssm_state_shape,
            ssm_dtype=self.ssm_dtype,
        )

    def get_dtype(self) -> torch.dtype:
        return self.ssm_dtype  # dominant buffer's dtype (informational)


class MLASubPoolSpec(msgspec.Struct, frozen=True, kw_only=True):
    """Layout spec for an MLA full-attention sub-pool: ONE latent region per
    layer (`kv_cache_dim = kv_lora_rank + qk_rope_head_dim`), no K/V pair.

    msgspec (not the grandfathered `@dataclass` ABC) per repo rule — the
    `SpecStateSubPoolSpec`/`MHARegionGeometry` precedent; `UnifiedKVPool`
    dispatches by isinstance, so nothing needs the ABC."""

    name: str
    layer_num: int
    kv_lora_rank: int
    qk_rope_head_dim: int
    store_dtype: torch.dtype
    grow_direction: str = "down"  # end pool, mirrors the MHA full spec

    def validate(self) -> None:
        assert self.layer_num > 0
        assert self.kv_lora_rank > 0 and self.qk_rope_head_dim > 0
        assert self.grow_direction in ("up", "down")

    def kv_cache_dim(self) -> int:
        return self.kv_lora_rank + self.qk_rope_head_dim

    def entry_bytes(self) -> int:
        # Single source of truth shared with the configurator's MLA pricing.
        return mla_entry_bytes(
            layer_num=self.layer_num,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            itemsize=self.store_dtype.itemsize,
        )

    def get_dtype(self) -> torch.dtype:
        return self.store_dtype


# Duck-typed union accepted by UnifiedKVPool: the grandfathered @dataclass
# hierarchy plus msgspec-based specs (`.claude/rules/no-dataclasses.md`).
SubPoolSpecLike = Union[SubPoolSpec, SpecStateSubPoolSpec, MLASubPoolSpec]


# ---------------------------------------------------------------------------
# UnifiedKVPool — the byte buffer + the strided per-sub-pool views
# ---------------------------------------------------------------------------


class UnifiedKVPool:
    """One physical `uint8` byte buffer shared by N sub-pools, each exposing
    strided per-layer views. Two end pools (one grow-up, one grow-down) plus
    optional "float" middle pools between their frontiers. Allocators keep
    byte ranges disjoint; no usage tracking here.
    """

    def __init__(
        self,
        *,
        total_bytes: int,
        sub_pool_specs: List[SubPoolSpecLike],
        device: str,
        enable_memory_saver: bool,
        page_size: int = 1,
    ):
        assert page_size >= 1, f"page_size must be >= 1; got {page_size}"
        assert (
            len(sub_pool_specs) >= 2
        ), f"UnifiedKVPool needs >= 2 sub-pools; got {len(sub_pool_specs)}"
        names = [s.name for s in sub_pool_specs]
        assert len(set(names)) == len(
            names
        ), f"sub-pool names must be unique; got {names}"
        for s in sub_pool_specs:
            if isinstance(s, (SpecStateSubPoolSpec, MLASubPoolSpec)):
                s.validate()
        up_specs = [s for s in sub_pool_specs if s.grow_direction == "up"]
        down_specs = [s for s in sub_pool_specs if s.grow_direction == "down"]
        float_specs = [s for s in sub_pool_specs if s.grow_direction == "float"]
        assert len(up_specs) == 1 and len(down_specs) == 1, (
            f"UnifiedKVPool needs exactly one grow-up and one grow-down end "
            f"sub-pool; got directions "
            f"{[s.grow_direction for s in sub_pool_specs]}"
        )
        assert len(up_specs) + len(down_specs) + len(float_specs) == len(
            sub_pool_specs
        ), (
            f"unknown grow_direction among "
            f"{[s.grow_direction for s in sub_pool_specs]}"
        )

        self.device = device
        self.total_bytes = total_bytes
        # Canonical chain order, low byte end -> high byte end: the grow-up end
        # pool, floating middles (input order preserved), the grow-down end
        # pool. Input list order is otherwise irrelevant (all access is
        # by-name); the chain order is what peer wiring follows.
        self.sub_pool_specs: List[SubPoolSpecLike] = [
            up_specs[0],
            *float_specs,
            down_specs[0],
        ]
        self._page_size = page_size
        self._specs_by_name: Dict[str, SubPoolSpecLike] = {
            s.name: s for s in sub_pool_specs
        }

        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            self._raw = torch.empty(total_bytes, dtype=torch.uint8, device=device)
        self._raw.zero_()  # unset slots must read as zeros (matches non-shared)

        self._max_slots: Dict[str, int] = {}
        self._anchor_bytes: Dict[str, int] = {}
        self._min_slot_index: Dict[str, int] = {}
        # MHA: (k_buffer, v_buffer); Mamba: (conv_state_list, temporal_state);
        # SpecState: (ssm_view, conv_window_views)
        self._mha_views: Dict[str, Tuple[List[torch.Tensor], List[torch.Tensor]]] = {}
        self._mamba_views: Dict[str, Tuple[List[torch.Tensor], torch.Tensor]] = {}
        self._spec_state_views: Dict[str, Tuple[torch.Tensor, List[torch.Tensor]]] = {}
        # MLA sub-pools: one latent view list per layer (no K/V pair).
        self._mla_views: Dict[str, List[torch.Tensor]] = {}
        # Draft-region K/V views of fused MHA sub-pools: same slot
        # ids and per-page stride as the host views, offset into each slot's
        # draft byte region.
        self._mha_draft_views: Dict[
            str, Tuple[List[torch.Tensor], List[torch.Tensor]]
        ] = {}

        # Slot-0 dummy writes for every pool land in [0, entry_max); each pool's
        # first allocatable slot is chosen so real data starts at >= entry_max.
        entry_max = max(s.entry_bytes() for s in self.sub_pool_specs)

        for spec in self.sub_pool_specs:
            entry_bytes = spec.entry_bytes()
            max_slots = total_bytes // entry_bytes
            min_slot_index = (entry_max + entry_bytes - 1) // entry_bytes  # ceil
            if max_slots <= min_slot_index:
                raise RuntimeError(
                    f"UnifiedKVPool: sub-pool {spec.name!r} fits only {max_slots} "
                    f"slots in {total_bytes} bytes, but min_slot_index={min_slot_index} "
                    f"leaves no room for real data. Increase total_bytes."
                )
            anchor = 0
            self._max_slots[spec.name] = max_slots
            self._anchor_bytes[spec.name] = anchor
            self._min_slot_index[spec.name] = min_slot_index
            if isinstance(spec, MHASubPoolSpec):
                self._mha_views[spec.name] = self._build_mha_views(
                    spec,
                    anchor,
                    max_slots,
                    page_size=page_size,
                )
                if spec.draft_region is not None:
                    self._mha_draft_views[spec.name] = self._build_mha_draft_views(
                        spec,
                        anchor,
                        max_slots,
                        page_size=page_size,
                    )
            elif isinstance(spec, MambaSubPoolSpec):
                self._mamba_views[spec.name] = self._build_mamba_views(
                    spec, anchor, max_slots
                )
            elif isinstance(spec, SpecStateSubPoolSpec):
                self._spec_state_views[spec.name] = self._build_spec_state_views(
                    spec, anchor, max_slots
                )
            elif isinstance(spec, MLASubPoolSpec):
                self._mla_views[spec.name] = self._build_mla_views(
                    spec,
                    anchor,
                    max_slots,
                    page_size=page_size,
                )
            else:  # pragma: no cover
                raise TypeError(f"unsupported SubPoolSpec type: {type(spec)}")

        logger.info(
            "[unified-memory-pool] UnifiedKVPool allocated: total_bytes=%.2f GB (=%d B), "
            "%d sub-pool(s)",
            total_bytes / GB,
            total_bytes,
            len(self.sub_pool_specs),
        )
        for s in self.sub_pool_specs:
            logger.info(
                "[unified-memory-pool]   sub-pool %r: kind=%s, layer_num=%d, grow=%s, "
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
            if isinstance(s, MHASubPoolSpec) and s.draft_region is not None:
                logger.info(
                    "[unified-memory-pool]     fused draft region: layer_num=%d, "
                    "head_num=%d, head_dim=%d/%d, entry_bytes=%d at slot offset %d "
                    "(host entry_bytes=%d)",
                    s.draft_region.layer_num,
                    s.draft_region.head_num,
                    s.draft_region.head_dim,
                    s.draft_region.v_head_dim,
                    s.draft_region.entry_bytes(),
                    s.draft_region_offset_bytes(),
                    s.host_entry_bytes(),
                )

    # -- introspection --

    def spec(self, name: str) -> SubPoolSpecLike:
        return self._specs_by_name[name]

    def mha_spec(self, name: str) -> MHASubPoolSpec:
        s = self._specs_by_name[name]
        assert isinstance(
            s, MHASubPoolSpec
        ), f"sub-pool {name!r} is {type(s).__name__}, expected MHASubPoolSpec"
        return s

    def mamba_spec(self, name: str) -> MambaSubPoolSpec:
        s = self._specs_by_name[name]
        assert isinstance(
            s, MambaSubPoolSpec
        ), f"sub-pool {name!r} is {type(s).__name__}, expected MambaSubPoolSpec"
        return s

    def spec_state_spec(self, name: str) -> SpecStateSubPoolSpec:
        s = self._specs_by_name[name]
        assert isinstance(
            s, SpecStateSubPoolSpec
        ), f"sub-pool {name!r} is {type(s).__name__}, expected SpecStateSubPoolSpec"
        return s

    def mla_spec(self, name: str) -> MLASubPoolSpec:
        s = self._specs_by_name[name]
        assert isinstance(
            s, MLASubPoolSpec
        ), f"sub-pool {name!r} is {type(s).__name__}, expected MLASubPoolSpec"
        return s

    def mla_views_for(self, name: str) -> List[torch.Tensor]:
        return self._mla_views[name]

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

    def has_draft_region(self, name: str) -> bool:
        """True when sub-pool `name` carries a fused draft KV region."""
        return name in self._mha_draft_views

    def draft_views_for(
        self, name: str
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Draft-region K/V views of fused sub-pool `name`."""
        return self._mha_draft_views[name]

    def mamba_views_for(self, name: str) -> Tuple[List[torch.Tensor], torch.Tensor]:
        return self._mamba_views[name]

    def spec_state_views_for(
        self, name: str
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        return self._spec_state_views[name]

    def _build_mha_views(
        self,
        spec: MHASubPoolSpec,
        anchor_bytes: int,
        max_slots: int,
        page_size: int,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        # page_stride_bytes uses the FUSED entry (== the host page bytes when
        # draft_region is None): with a draft region, consecutive pages are
        # entry_bytes() apart even for the host views.
        return build_page_major_mha_views(
            self._raw,
            layer_num=spec.layer_num,
            head_num=spec.head_num,
            head_dim=spec.head_dim,
            v_head_dim=spec.v_head_dim,
            store_dtype=spec.store_dtype,
            page_size=page_size,
            num_pages=max_slots // page_size,
            anchor_bytes=anchor_bytes,
            page_stride_bytes=page_size * spec.entry_bytes(),
        )

    def _build_mla_views(
        self,
        spec: MLASubPoolSpec,
        anchor_bytes: int,
        max_slots: int,
        page_size: int,
    ) -> List[torch.Tensor]:
        return build_page_major_mla_views(
            self._raw,
            layer_num=spec.layer_num,
            kv_lora_rank=spec.kv_lora_rank,
            qk_rope_head_dim=spec.qk_rope_head_dim,
            store_dtype=spec.store_dtype,
            page_size=page_size,
            num_pages=max_slots // page_size,
            anchor_bytes=anchor_bytes,
            page_stride_bytes=page_size * spec.entry_bytes(),
        )

    def _build_mha_draft_views(
        self,
        spec: MHASubPoolSpec,
        anchor_bytes: int,
        max_slots: int,
        page_size: int,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """K/V views of the draft byte region inside `spec`'s fused slots:
        same slot ids and page stride as the host views, based at the draft
        region's offset within each page block."""
        region = spec.draft_region
        assert region is not None
        return build_page_major_mha_views(
            self._raw,
            layer_num=region.layer_num,
            head_num=region.head_num,
            head_dim=region.head_dim,
            v_head_dim=region.v_head_dim,
            store_dtype=region.store_dtype,
            page_size=page_size,
            num_pages=max_slots // page_size,
            anchor_bytes=anchor_bytes
            + page_size * spec.draft_region_offset_bytes(),
            page_stride_bytes=page_size * spec.entry_bytes(),
        )

    def _build_mamba_views(
        self, spec: MambaSubPoolSpec, anchor_bytes: int, max_slots: int
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        return build_page_major_mamba_views(
            self._raw,
            layer_num=spec.layer_num,
            conv_state_shapes=spec.conv_state_shapes,
            conv_dtype=spec.conv_dtype,
            temporal_state_shape=spec.temporal_state_shape,
            temporal_dtype=spec.temporal_dtype,
            max_slots=max_slots,
            anchor_bytes=anchor_bytes,
        )

    def _build_spec_state_views(
        self, spec: SpecStateSubPoolSpec, anchor_bytes: int, max_slots: int
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        return build_spec_state_views(
            self._raw,
            layer_num=spec.layer_num,
            num_draft_tokens=spec.num_draft_tokens,
            conv_window_shapes=spec.conv_window_shapes,
            conv_dtype=spec.conv_dtype,
            ssm_state_shape=spec.ssm_state_shape,
            ssm_dtype=spec.ssm_dtype,
            max_slots=max_slots,
            anchor_bytes=anchor_bytes,
        )


class UnifiedMHATokenToKVPool(MHATokenToKVPool):
    """MHA KV pool whose `k_buffer`/`v_buffer` are strided views into a `UnifiedKVPool`.

    Relocation uses the native move (strided views break the tiled Triton kernel that
    assumes stride == row bytes). `set_kv_buffer` gets PHYSICAL slot ids; never translates.
    """

    def __init__(
        self,
        *,
        unified_buffer: UnifiedKVPool,
        sub_pool_name: str,
        page_size: int = 1,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        enable_alt_stream: bool = True,
    ):
        spec = unified_buffer.mha_spec(sub_pool_name)
        k_buffer, v_buffer = unified_buffer.mha_views_for(sub_pool_name)
        max_slots = unified_buffer.max_slots(sub_pool_name)

        self._unified_buffer = unified_buffer
        self._sub_pool_name = sub_pool_name
        self._k_views = k_buffer
        self._v_views = v_buffer
        self._page_size = page_size
        # Fused draft region: the host pool owns relocation for the
        # WHOLE slot envelope, so compaction/accept moves must carry the draft
        # bytes together with the host bytes.
        if unified_buffer.has_draft_region(sub_pool_name):
            draft_k, draft_v = unified_buffer.draft_views_for(sub_pool_name)
            self._draft_k_views: List[torch.Tensor] = draft_k
            self._draft_v_views: List[torch.Tensor] = draft_v
        else:
            self._draft_k_views = []
            self._draft_v_views = []

        super().__init__(
            size=max_slots - 1,  # -1 for reserved slot 0
            page_size=page_size,
            dtype=spec.store_dtype,
            head_num=spec.head_num,
            head_dim=spec.head_dim,
            layer_num=spec.layer_num,
            device=unified_buffer.device,
            enable_memory_saver=False,  # buffer owned by UnifiedKVPool
            v_head_dim=spec.v_head_dim,
            start_layer=start_layer,
            end_layer=end_layer,
            enable_alt_stream=enable_alt_stream,
            enable_kv_cache_copy=False,  # strided views — force native move
        )

    def _create_buffers(self):
        self.k_buffer = self._k_views
        self.v_buffer = self._v_views
        # For external inspectors only; the native move path doesn't consume them.
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
        self.data_strides = torch.tensor(
            [x.stride(0) * x.dtype.itemsize for x in (self.k_buffer + self.v_buffer)],
            device=self.device,
        )

    def _clear_buffers(self):
        # Lifetime owned by UnifiedKVPool; do not delete the views.
        pass

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        # tgt_loc/src_loc are PHYSICAL slot ids; native move only (strided views).
        # With a fused draft region the draft views ride in the same lists: one
        # move relocates the whole [host KV | draft KV] envelope atomically
        # (compaction, inverse-history rollback, and move_accept_kv all land
        # here). move_kv_cache_native zips k/v pairwise, so heterogeneous
        # host/draft shapes are fine.
        if tgt_loc.numel() == 0:
            return
        with record_function("UnifiedMHA.move_kv_cache"):
            move_kv_cache_native(
                self.k_buffer + self._draft_k_views,
                self.v_buffer + self._draft_v_views,
                tgt_loc,
                src_loc,
                page_size=self._page_size,
            )

    def set_kv_buffer_prefix_valid(
        self,
        layer,
        loc_2d: torch.Tensor,
        commit_lens: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale=None,
        v_scale=None,
        layer_id_override: Optional[int] = None,
    ):
        """Commit only each row's valid prefix (DFLASH's block write).

        MUST override: the parent's fused kernel writes through a packed
        `row_dim` stride, but our K/V are 4-D strided envelope views
        (stride[0] = page_bytes), so it would scatter to the WRONG addresses —
        silently, no crash. This is the same assumption that forced the
        `set_kv_buffer` override. Reduce to the valid rows and route through
        the stride-aware `set_kv_buffer` (exactly what the parent already does
        on non-CUDA). `loc_2d` is PHYSICAL — DFLASH translates at its choke
        point (`_append_target_hidden_to_draft_kv_by_loc`).
        """
        if loc_2d.ndim != 2:
            raise ValueError(f"loc_2d must be rank-2, got shape={tuple(loc_2d.shape)}.")
        row_offsets = torch.arange(loc_2d.shape[1], device=loc_2d.device)
        valid = row_offsets[None, :] < commit_lens.to(torch.int64)[:, None]
        valid_idx = torch.nonzero(valid.reshape(-1), as_tuple=False).flatten()
        if valid_idx.numel() == 0:
            return
        self.set_kv_buffer(
            layer,
            loc_2d.reshape(-1).index_select(0, valid_idx),
            cache_k.index_select(0, valid_idx),
            cache_v.index_select(0, valid_idx),
            k_scale,
            v_scale,
            layer_id_override=layer_id_override,
        )

    def get_contiguous_buf_infos(self):
        # PD-disaggregation KV transfer assumes densely packed rows; a fused
        # slot interleaves draft bytes between host rows, so the item/len
        # math would silently transfer garbage. Unified memory already
        # excludes PD (server_args gate) — fail loud if anything reaches here.
        if self._draft_k_views:
            raise NotImplementedError(
                "get_contiguous_buf_infos is not supported on a fused "
                "draft-KV sub-pool: rows are strided, not "
                "contiguous. PD transfer / HiCache must stay disabled with "
                "--enable-unified-memory + speculative decoding."
            )
        return super().get_contiguous_buf_infos()

    def get_kv_size_bytes(self):
        return 0, 0  # UnifiedKVPool logs the total; per-sub-pool would double-count

    def set_kv_buffer(
        self,
        layer,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale=None,
        v_scale=None,
        layer_id_override: Optional[int] = None,
        dcp_kv_mask: Optional[torch.Tensor] = None,
    ):
        # Decode context parallel (dcp_kv_mask) unsupported; fail loud.
        assert dcp_kv_mask is None, (
            "UnifiedMHATokenToKVPool.set_kv_buffer: decode context parallel "
            "(dcp_kv_mask) is not supported with --enable-unified-memory."
        )
        # Accept a bare loc OR a KVWriteLoc. Composite pools (HybridLinearKVPool,
        # UnifiedSWAKVPool) unwrap and hand us a bare PHYSICAL tensor, which is
        # the only way this pool was reached before fused draft KV. A fused DENSE
        # draft (DFLASH / EAGLE3 checkpoint) binds UnifiedDraftKVPool as the
        # runner's token_to_kv_pool DIRECTLY, so the backend's KVWriteLoc lands
        # here raw.
        # `full_loc` is the pre-translated PHYSICAL loc; a bare `loc` is already
        # physical.
        was_write_loc = isinstance(loc, KVWriteLoc)
        loc, _, full_loc = unwrap_write_loc(loc)
        if full_loc is not None:
            loc = full_loc
        elif was_write_loc:
            # A KVWriteLoc carrying no full_loc means the backend never ran the
            # virtual->physical translate, so `loc` holds VIRTUAL ids. Writing
            # them as physical does not crash -- it silently scribbles KV into
            # whatever slots the virtual ids happen to name, and only shows up
            # later as a degraded accept length. Backends that translate always
            # populate full_loc; a backend that does not (any non-Triton one) has
            # no business reaching this pool. Fail here rather than corrupt.
            raise RuntimeError(
                "unified pool received a KVWriteLoc with no physical loc "
                "(full_loc=None): the attention backend did not translate the "
                "virtual slot ids. The unified memory pool requires the Triton "
                "attention backend on every path, including "
                "--speculative-draft-attention-backend."
            )
        # Bypass super().set_kv_buffer: the parent's `k_cache.view(-1, row_dim)` can't
        # merge our 4-D layer-major view (stride[0]=page_bytes) at page_size>1. Call
        # store_cache_4d_kernel directly. `loc` is PHYSICAL token ids — no v2p translate.
        with record_function("UnifiedMHA.set_kv_buffer"):
            if cache_k.dtype != self.dtype:
                if k_scale is not None:
                    cache_k.div_(k_scale)
                if v_scale is not None:
                    cache_v.div_(v_scale)
                cache_k = cache_k.to(self.dtype)
                cache_v = cache_v.to(self.dtype)
            if self.store_dtype != self.dtype:
                cache_k = cache_k.view(self.store_dtype)
                cache_v = cache_v.view(self.store_dtype)

            layer_id = (
                layer.layer_id if layer_id_override is None else layer_id_override
            ) - self.start_layer
            assert 0 <= layer_id < len(self.k_buffer), (
                f"{type(self).__name__}: layer_id {layer_id} out of range for "
                f"{len(self.k_buffer)} layer view(s) (layer_id_override="
                f"{layer_id_override}, start_layer={self.start_layer}). A wrong "
                f"layer index writes this token's KV over another layer's."
            )
            k_view = self.k_buffer[layer_id]
            v_view = self.v_buffer[layer_id]
            ps = self._page_size
            N = loc.numel()
            if N == 0:
                return

            # We call `store_cache_4d_kernel` DIRECTLY (the `store_cache_4d`
            # wrapper cannot merge our 4-D layer-major view at page_size > 1), so
            # the wrapper's contract checks do not run and we must uphold them
            # here. The kernel reads `loc` and each KV row with raw pointer
            # arithmetic and walks FLAT memory -- a non-compact layout is read
            # with the wrong element spacing, so the store lands the WRONG BYTES
            # at the RIGHT slot. It does not fail: no index leaves its range, and
            # every torch-side check passes, because torch honours the very stride
            # the kernel ignores. The only symptom is degraded model quality.
            #
            # Both hazards are real, not theoretical:
            #   - source rows: a model may hand us a slice of a fused QKV
            #     projection, or a transpose;
            #   - `loc`: a per-step draft slot table built with
            #     permute(...).reshape(...) is a strided VIEW, and it made this
            #     kernel scatter draft KV across the wrong requests.
            # Normalize rather than reject -- both callers are legitimate.
            cache_k = _as_row_contiguous(cache_k)
            cache_v = _as_row_contiguous(cache_v)
            if not loc.is_contiguous():
                loc = loc.contiguous()
            assert k_view.stride(-1) == 1 and k_view.stride(-2) == k_view.shape[-1], (
                f"{type(self).__name__}: k_view trailing dims not contiguous; "
                f"stride={k_view.stride()}, shape={tuple(k_view.shape)}"
            )
            assert v_view.stride(-1) == 1 and v_view.stride(-2) == v_view.shape[-1], (
                f"{type(self).__name__}: v_view trailing dims not contiguous; "
                f"stride={v_view.stride()}, shape={tuple(v_view.shape)}"
            )
            head_num = k_view.shape[2]
            head_dim = k_view.shape[3]
            v_head_dim = v_view.shape[3]
            K_ROW_DIM = head_num * head_dim
            V_ROW_DIM = head_num * v_head_dim
            BLOCK = 128
            row_dim_max = K_ROW_DIM if K_ROW_DIM > V_ROW_DIM else V_ROW_DIM
            store_cache_4d_kernel[(N, triton.cdiv(row_dim_max, BLOCK), 2)](
                k_view,
                v_view,
                cache_k,
                cache_v,
                loc,
                k_view.stride(0),
                k_view.stride(1),
                v_view.stride(0),
                v_view.stride(1),
                cache_k.stride(0),
                cache_v.stride(0),
                K_ROW_DIM=K_ROW_DIM,
                V_ROW_DIM=V_ROW_DIM,
                PAGE_SIZE=ps,
                BLOCK=BLOCK,
                num_warps=4,
            )


class UnifiedMLATokenToKVPool(MLATokenToKVPool):
    """MLA latent-KV pool whose `kv_buffer` is a list of 4-D PAGE-MAJOR strided
    views into a `UnifiedKVPool` — the single-region analog of
    `UnifiedMHATokenToKVPool`.

    The views are `(num_pages, page_size, 1, kv_cache_dim)` and token
    addressing is two-level (`page = loc // ps`, `slot = loc % ps`) — NON-affine
    in the token id at page_size > 1, so every parent method that indexes
    `kv_buffer[loc]` token-affinely is overridden with a page-aware
    implementation. The layout is page-size-general by design (future ps > 1);
    today's only MLA-hybrid (Kimi Linear) runs at ps == 1 because the
    mamba-radix machinery forces it.

    `set_kv_buffer` gets PHYSICAL slot ids; never translates. Model-side entry
    points (`set_mla_kv_buffer`/`get_mla_kv_buffer`) also receive PHYSICAL
    ids: writes carry the ForwardBatch's rebound `out_cache_loc`
    (apply_unified_kv_loc_rebind), reads are translated at their production
    sites (fetch_mha_one_shot_kv_indices / prepare_chunked_kv_indices)."""

    def __init__(
        self,
        *,
        unified_buffer: UnifiedKVPool,
        sub_pool_name: str,
        page_size: int = 1,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
    ):
        spec = unified_buffer.mla_spec(sub_pool_name)
        assert spec.store_dtype in (torch.bfloat16, torch.float16), (
            "unified MLA sub-pool supports bf16/fp16 KV only (fp8/fp4 store "
            "juggling in the MLA base class is untested on strided views)"
        )
        self._unified_buffer = unified_buffer
        self._sub_pool_name = sub_pool_name
        self._kv_views = unified_buffer.mla_views_for(sub_pool_name)
        self._page_size = page_size
        max_slots = unified_buffer.max_slots(sub_pool_name)

        super().__init__(
            size=max_slots - 1,  # -1 for reserved slot 0 (dummy-write sink)
            page_size=page_size,
            dtype=spec.store_dtype,
            kv_lora_rank=spec.kv_lora_rank,
            qk_rope_head_dim=spec.qk_rope_head_dim,
            layer_num=spec.layer_num,
            device=unified_buffer.device,
            enable_memory_saver=False,  # buffer owned by UnifiedKVPool
            start_layer=start_layer,
            end_layer=end_layer,
        )

    def _create_buffers(self):
        self.kv_buffer = self._kv_views
        # For external inspectors only; nothing on the MLA path consumes them.
        self.data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.kv_buffer],
            dtype=torch.uint64,
            device=self.device,
        )

    def _clear_buffers(self):
        pass  # lifetime owned by UnifiedKVPool

    def get_kv_size_bytes(self):
        # SCALAR (the MLA contract — HybridLinearKVPool.__init__ divides it);
        # the UnifiedKVPool logs the real total centrally.
        return 0

    def get_contiguous_buf_infos(self):
        raise NotImplementedError(
            "UnifiedMLATokenToKVPool: PD disaggregation / host transfer needs "
            "contiguous per-layer buffers; the unified pool's views are "
            "strided. PD is rejected under --enable-unified-memory."
        )

    def get_cpu_copy(self, indices):
        raise NotImplementedError(
            "UnifiedMLATokenToKVPool: host offload (HiCache) is disabled under "
            "--enable-unified-memory; the parent's token-affine copy is "
            "invalid on page-major views."
        )

    def load_cpu_copy(self, kv_cache_cpu, indices):
        raise NotImplementedError(
            "UnifiedMLATokenToKVPool: host offload (HiCache) is disabled under "
            "--enable-unified-memory."
        )

    def _physical_loc(self, loc_info, ctx: str) -> torch.Tensor:
        """Unwrap a bare PHYSICAL loc or a KVWriteLoc WITH full_loc; RAISE on a
        KVWriteLoc without one (an untranslated backend) — the same load-bearing
        tripwire as `UnifiedMHATokenToKVPool.set_kv_buffer`."""
        was_write_loc = isinstance(loc_info, KVWriteLoc)
        loc, _, full_loc = unwrap_write_loc(loc_info)
        if full_loc is not None:
            return full_loc
        if was_write_loc:
            raise RuntimeError(
                f"unified MLA pool ({ctx}) received a KVWriteLoc with no "
                "physical loc (full_loc=None): the attention backend did not "
                "translate the virtual slot ids. The unified memory pool "
                "requires the Triton attention backend on every path."
            )
        return loc

    def set_kv_buffer(
        self,
        layer,
        loc_info,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        loc = self._physical_loc(loc_info, "set_kv_buffer")
        with record_function("UnifiedMLA.set_kv_buffer"):
            if cache_k.dtype != self.dtype:
                cache_k = cache_k.to(self.dtype)
            layer_id = layer.layer_id - self.start_layer
            assert 0 <= layer_id < len(self.kv_buffer), (
                f"UnifiedMLATokenToKVPool: layer_id {layer_id} out of range "
                f"for {len(self.kv_buffer)} layer view(s)"
            )
            kv_view = self.kv_buffer[layer_id]
            N = loc.numel()
            if N == 0:
                return
            # Reuse the proven page-aware store: `store_cache_4d_kernel`'s K/V
            # split is grid-controlled (axis 2), so a single-region launch
            # (grid z=1) writes only the "K" region — which here IS the whole
            # latent row. Same raw-pointer caller contract as the MHA pool:
            # compact source rows + contiguous loc, upheld here.
            cache_k = cache_k.reshape(N, -1)
            if not cache_k.is_contiguous():
                cache_k = cache_k.contiguous()
            if not loc.is_contiguous():
                loc = loc.contiguous()
            assert kv_view.stride(-1) == 1 and (
                kv_view.stride(1) == kv_view.shape[-1]
            ), (
                f"UnifiedMLATokenToKVPool: kv_view trailing dims not compact; "
                f"stride={kv_view.stride()}, shape={tuple(kv_view.shape)}"
            )
            ROW_DIM = kv_view.shape[2] * kv_view.shape[3]  # 1 * kv_cache_dim
            assert cache_k.shape[1] == ROW_DIM, (
                f"UnifiedMLATokenToKVPool: latent row dim {cache_k.shape[1]} "
                f"!= view row dim {ROW_DIM}"
            )
            BLOCK = 128
            store_cache_4d_kernel[(N, triton.cdiv(ROW_DIM, BLOCK), 1)](
                kv_view,
                kv_view,  # V pointer unused: grid z=1 never runs the V branch
                cache_k,
                cache_k,
                loc,
                kv_view.stride(0),
                kv_view.stride(1),
                kv_view.stride(0),
                kv_view.stride(1),
                cache_k.stride(0),
                cache_k.stride(0),
                K_ROW_DIM=ROW_DIM,
                V_ROW_DIM=ROW_DIM,
                PAGE_SIZE=self._page_size,
                BLOCK=BLOCK,
                num_warps=4,
            )

    def set_mla_kv_buffer(
        self,
        layer,
        loc: torch.Tensor,
        cache_k_nope: torch.Tensor,
        cache_k_rope: torch.Tensor,
    ):
        # `loc` is PHYSICAL: the ForwardBatch's out_cache_loc was rebound at
        # construction (apply_unified_kv_loc_rebind).
        layer_id = layer.layer_id - self.start_layer
        with record_function("UnifiedMLA.set_mla_kv_buffer"):
            if cache_k_nope.dtype != self.dtype:
                cache_k_nope = cache_k_nope.to(self.dtype)
                cache_k_rope = cache_k_rope.to(self.dtype)
            set_mla_kv_buffer_page_major(
                self.kv_buffer[layer_id],
                loc.contiguous(),
                cache_k_nope.reshape(loc.numel(), -1).contiguous(),
                cache_k_rope.reshape(loc.numel(), -1).contiguous(),
                page_size=self._page_size,
            )

    def get_mla_kv_buffer(
        self,
        layer,
        loc: torch.Tensor,
        dst_dtype: Optional[torch.dtype] = None,
    ):
        layer_id = layer.layer_id - self.start_layer
        n = loc.numel()
        dst_dtype = dst_dtype or self.dtype
        cache_k_nope = torch.empty(
            (n, 1, self.kv_lora_rank), dtype=dst_dtype, device=self.device
        )
        cache_k_rope = torch.empty(
            (n, 1, self.qk_rope_head_dim), dtype=dst_dtype, device=self.device
        )
        get_mla_kv_buffer_page_major(
            self.kv_buffer[layer_id],
            loc.contiguous(),
            cache_k_nope.view(n, -1),
            cache_k_rope.view(n, -1),
            page_size=self._page_size,
        )
        return cache_k_nope, cache_k_rope

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        """Page-aware single-region move for compaction / inverse-history /
        accept-move. `tgt_loc`/`src_loc` are PHYSICAL token ids; two-level
        advanced indexing is stride-correct on the 4-D views."""
        ps = self._page_size
        tgt_pg, tgt_in = tgt_loc // ps, tgt_loc % ps
        src_pg, src_in = src_loc // ps, src_loc % ps
        for kv in self.kv_buffer:
            kv[tgt_pg, tgt_in] = kv[src_pg, src_in]


class UnifiedDraftKVPool(UnifiedMHATokenToKVPool):
    """Draft-model KV pool over the DRAFT byte region of a fused host sub-pool.

    An envelope view into the target's `UnifiedKVPool._raw`: shares the host
    sub-pool's slot ids, allocator, v2p mapping, and compaction; owns NO
    memory. Draft-runner layer `i` maps to draft-region layer
    `layer_offset + i` (multi-layer EAGLE gives each per-step runner its own
    layer sub-range, keyed by `draft_model_idx`). Relocation is host-driven —
    the host pool's `move_kv_cache` carries the whole fused envelope — so this
    pool must never be asked to move or to describe contiguous buffers.
    Like `UnifiedMambaPool`, skips the immediate parent's `__init__`: it binds
    the draft views, then initializes `MHATokenToKVPool` state directly with
    the draft region's geometry.
    """

    def __init__(
        self,
        *,
        unified_buffer: UnifiedKVPool,
        sub_pool_name: str,
        page_size: int = 1,
        layer_offset: int = 0,
        layer_num: Optional[int] = None,
        enable_alt_stream: bool = True,
    ):
        spec = unified_buffer.mha_spec(sub_pool_name)
        region = spec.draft_region
        assert (
            region is not None
        ), f"sub-pool {sub_pool_name!r} has no fused draft region"
        draft_k, draft_v = unified_buffer.draft_views_for(sub_pool_name)
        if layer_num is None:
            layer_num = region.layer_num - layer_offset
        assert 0 <= layer_offset and layer_offset + layer_num <= region.layer_num, (
            f"draft layer sub-range [{layer_offset}, {layer_offset + layer_num}) "
            f"out of the region's {region.layer_num} layers"
        )
        max_slots = unified_buffer.max_slots(sub_pool_name)

        self._unified_buffer = unified_buffer
        self._sub_pool_name = sub_pool_name
        self._k_views = draft_k[layer_offset : layer_offset + layer_num]
        self._v_views = draft_v[layer_offset : layer_offset + layer_num]
        self._page_size = page_size
        # Never a mover (see move_kv_cache below); the host pool's fused move
        # lists carry this region.
        self._draft_k_views: List[torch.Tensor] = []
        self._draft_v_views: List[torch.Tensor] = []

        MHATokenToKVPool.__init__(
            self,
            size=max_slots - 1,  # -1 for reserved slot 0
            page_size=page_size,
            dtype=region.store_dtype,
            head_num=region.head_num,
            head_dim=region.head_dim,
            layer_num=layer_num,
            device=unified_buffer.device,
            enable_memory_saver=False,  # buffer owned by UnifiedKVPool
            v_head_dim=region.v_head_dim,
            start_layer=0,  # draft-runner layer ids are 0-based
            end_layer=layer_num,
            enable_alt_stream=enable_alt_stream,
            enable_kv_cache_copy=False,  # strided views — no tiled copy
        )

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        raise NotImplementedError(
            "UnifiedDraftKVPool never moves KV: the HOST sub-pool's "
            "move_kv_cache relocates the whole fused [host|draft] envelope."
        )

    def get_contiguous_buf_infos(self):
        raise NotImplementedError(
            "get_contiguous_buf_infos is not supported on the fused draft "
            "view pool (strided rows; PD transfer is excluded under "
            "--enable-unified-memory)."
        )


class UnifiedMambaPool(MambaPool):
    """Mamba state pool whose conv/temporal state are strided views into a `UnifiedKVPool`.

    Pure PHYSICAL store: slot lifecycle and the v<->p mapping live in the attached
    `UnifiedMambaSlotAllocator`. Does NOT call `super().__init__()` — replicates the
    minimal `MambaPool` state against the unified buffer so inherited methods work.
    """

    def __init__(
        self,
        *,
        unified_buffer: UnifiedKVPool,
        sub_pool_name: str,
        spec_state_size: int,
        mamba_layer_ids: List[int],
        enable_memory_saver: bool = False,
        speculative_num_draft_tokens: Optional[int] = None,
        spec_state_sub_pool: Optional[str] = None,
    ):
        spec = unified_buffer.mamba_spec(sub_pool_name)
        assert spec.layer_num == len(mamba_layer_ids)
        conv_views, temporal_view = unified_buffer.mamba_views_for(sub_pool_name)
        max_slots = unified_buffer.max_slots(sub_pool_name)

        self._unified_buffer = unified_buffer
        self._sub_pool_name = sub_pool_name

        # Replicate the state MambaPool.__init__ would have set.
        self._max_size = max_slots - 1  # -1 for reserved slot 0
        self.size = self._max_size
        self.device = unified_buffer.device
        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )
        self.enable_custom_mem_pool = False
        self.custom_mem_pool = None
        self.num_mamba_layers = spec.layer_num
        # GDN/KDA ReplaySSM unsupported; replicate parent's disabled-state attrs so
        # paths guarded by `replayssm_write_pos is not None` don't AttributeError.
        self.enable_linear_replayssm = False
        self.linear_replayssm_cache_len = 16
        self.replayssm_write_pos = None
        self.replayssm_is_kda = False

        assert (
            conv_views[0].shape[0] == self.num_mamba_layers
        ), f"conv_views layers={conv_views[0].shape[0]} vs expected {self.num_mamba_layers}"
        assert (
            conv_views[0].shape[1] == self._max_size + 1
        ), f"conv_views slots={conv_views[0].shape[1]} vs expected {self._max_size + 1}"

        # Spec-decode intermediates live in the shared buffer as the
        # "spec_state" float band sub-pool: DENSE envelope views addressed by
        # PHYSICAL band slots (row r of the current verify batch -> physical
        # slot band_start + r via the band allocator's v2p). The views span
        # the whole buffer (their outer dim is max_slots, > spec_state_size+1);
        # only the placed band's rows ever hold live data.
        if speculative_num_draft_tokens is not None:
            assert spec_state_sub_pool is not None, (
                "UnifiedMambaPool: speculative decoding needs a spec_state "
                "sub-pool in the unified buffer (spec_state_sub_pool=None)"
            )
            spec_spec = unified_buffer.spec_state_spec(spec_state_sub_pool)
            assert spec_spec.layer_num == self.num_mamba_layers
            assert spec_spec.num_draft_tokens == speculative_num_draft_tokens
            assert unified_buffer.max_slots(spec_state_sub_pool) >= (
                spec_state_size + 1
            ), (
                f"spec_state sub-pool fits {unified_buffer.max_slots(spec_state_sub_pool)} "
                f"slots < spec_state_size+1 = {spec_state_size + 1}"
            )
            ssm_view, conv_window_views = unified_buffer.spec_state_views_for(
                spec_state_sub_pool
            )
            self.mamba_cache = self.SpeculativeState(
                conv=list(conv_views),
                temporal=temporal_view,
                intermediate_ssm=ssm_view,
                intermediate_conv_window=list(conv_window_views),
            )
        else:
            self.mamba_cache = self.State(conv=list(conv_views), temporal=temporal_view)

        self.mem_usage = unified_buffer.total_bytes / GB
        logger.info(
            "[unified-memory-pool] UnifiedMambaPool(%s) wrapped unified buffer: max_slots=%d, "
            "num_mamba_layers=%d",
            sub_pool_name,
            max_slots,
            self.num_mamba_layers,
        )

    # Inherited MambaPool state ops (copy_from/clear_slots/get_cpu_copy/load_cpu_copy)
    # take PHYSICAL slot ids; callers translate via the slot allocator first.

    def _copy_from_physical(self, src_index: torch.Tensor, dst_index: torch.Tensor):
        # Physical-slot copy used by the allocator's `_compact_pending`.
        #
        # PER-LAYER, not the parent's all-layer `copy_from`: advanced indexing
        # materializes the gathered result, so a batched urgent flush moving
        # 100+ mamba states allocates (layers x n x inner) transiently — tens of
        # GiB, OOMing the probe-triggered flush. Looping layers
        # bounds the transient to 1/num_layers of the move while KEEPING the
        # batched gather-first semantics over the full src/dst sets per layer
        # (correct even when one move's dst page equals another's src —
        # slot-chunking would silently assume an ordering invariant instead).
        if src_index.numel() == 0:
            return
        for conv in self.mamba_cache.conv:
            for layer in range(conv.shape[0]):
                conv[layer][dst_index] = conv[layer][src_index]
        temporal = self.mamba_cache.temporal
        for layer in range(temporal.shape[0]):
            temporal[layer][dst_index] = temporal[layer][src_index]
        if self.replayssm_write_pos is not None:
            # Mirror MambaPool.copy_from: a copied checkpoint has no pending
            # ReplaySSM ring entries.
            self.replayssm_write_pos[dst_index] = 0


class UnifiedMambaSlotAllocator:
    """Mamba slot allocator (PHYSICAL view) for the unified memory pool.

    Owns slot alloc/free, sizing, and the v<->p mapping (``translate``), presenting the
    upstream ``MambaSlotAllocator`` interface. ``alloc()`` returns VIRTUAL ids and does
    NOT clear state — clearing is deferred to ``UnifiedMambaPool.clear_slots``.
    """

    def __init__(
        self, mea, max_size: int, device: str, reclaim_band=None, prefetch_cap=None
    ):
        self._multi_ended_allocator = mea
        self._max_size = max_size  # excludes reserved slot 0
        self._device = device
        self._alloc_iter = None  # active alloc_group batch iterator
        # Byte-budgeted slot count (max_mamba_cache_size) bounding the
        # alloc_group prefetch. NOT _max_size: that is the INDEX-space size
        # (total_bytes // entry, ~4x larger), which would let a deep-queue
        # prefetch legally claim most of the shared gap anyway.
        self._prefetch_cap = prefetch_cap if prefetch_cap is not None else max_size
        # Composite's zero-copy band-park (`_reclaim_spec_state_band`), or None
        # when spec is off. On a mamba-slot alloc shortfall the mamba frontier can
        # be blocked by the (unpinned) spec band even though `available_size()`
        # counts the band-transparent gap; park the band and retry once, mirroring
        # the full-KV alloc paths' `_alloc_with_spec_band_reclaim`.
        self._reclaim_band = reclaim_band

    def _alloc_with_band_reclaim(self, need_size: int):
        """Call the MEA alloc; on shortfall, park an unpinned band and retry once."""
        out = self._multi_ended_allocator.alloc(need_size)
        if out is None and self._reclaim_band is not None and self._reclaim_band():
            out = self._multi_ended_allocator.alloc(need_size)
        return out

    # -- translation (owns the v<->p mapping) --

    def translate(self, virtual_ids: torch.Tensor) -> torch.Tensor:
        # VIRTUAL -> PHYSICAL slot ids; page_size==1, so a direct v2p gather.
        return self._multi_ended_allocator.virtual_to_physical[virtual_ids]

    @property
    def virtual_to_physical(self) -> torch.Tensor:
        return self._multi_ended_allocator.virtual_to_physical

    # -- sizing / free-list --

    @property
    def size(self) -> int:
        return self._max_size

    def available_size(self) -> int:
        # Slot-conservation count (max - allocated): the leak-check view, NOT the
        # planner value (use schedulable_available_size for that).
        return self._max_size - self._multi_ended_allocator.allocated_count()

    def schedulable_available_size(self) -> int:
        # Byte-coordinated count (>= N => alloc(N) succeeds); credits the peer's
        # drainable holes since alloc flushes the peer before extending.
        return self._multi_ended_allocator.schedulable_available_size()

    @property
    def free_slots(self) -> torch.Tensor:
        # Watermark-derived physical free-list for the invariant checker.
        a = self._multi_ended_allocator
        assert a.page_size == 1, (
            "UnifiedMambaSlotAllocator.free_slots assumes page_size==1; got "
            f"{a.page_size}. Mamba state is per-request, orthogonal to paging."
        )
        if a.grow_direction == "up":
            start, end = a.watermark_physical, a.num_pages
        else:
            start, end = a.min_page_index, a.watermark_physical + 1
        if start >= end:
            return torch.empty((0,), dtype=torch.int64, device=self._device)
        return torch.arange(start, end, dtype=torch.int64, device=self._device)

    # -- slot management (delegates to the MultiEndedAllocator) --

    def alloc(self, need_size: int):
        # alloc_group fast path: single-slot draws from the prefetched batch.
        if self._alloc_iter is not None and need_size == 1:
            slot = next(self._alloc_iter, None)
            if slot is not None:
                return slot
        return self._alloc_with_band_reclaim(need_size)  # VIRTUAL ids

    def free(self, free_index: torch.Tensor):
        return self._multi_ended_allocator.free(free_index)

    def clear(self):
        self._alloc_iter = None
        return self._multi_ended_allocator.clear()

    def alloc_group_begin(self, num_reqs: int):
        """Pre-allocate a batch that ``alloc(1)`` then draws from."""
        self._alloc_iter = None
        # Cap the prefetch at the BYTE-BUDGETED remaining capacity
        # (max_mamba_cache_size − allocated). Mamba slots are BYTES in the
        # shared buffer (~139MB each); the scheduler calls this with
        # len(waiting_queue), which can be 100+ — an uncapped prefetch
        # transiently claims most of the shared gap during every admission
        # pass at small mem fractions (a deep queue x the per-state size can reach
        # tens of GiB), starving full-KV admission. Upstream's dense
        # pool hands out byte-decoupled slot IDs where over-prefetch was free;
        # here it is not. Surplus is returned at alloc_group_end; a shortfall
        # falls through to per-request alloc(1) — the cap never changes WHAT
        # gets allocated, only the transient's size.
        num_reqs = min(
            num_reqs,
            max(0, self._prefetch_cap - self._multi_ended_allocator.allocated_count()),
        )
        if num_reqs > 0:
            result = self._alloc_with_band_reclaim(num_reqs)
            if result is not None:
                self._alloc_iter = iter(result.split(1))

    def alloc_group_end(self):
        """Return any unused pre-allocated slots from the current group."""
        if self._alloc_iter is not None:
            remaining = list(self._alloc_iter)
            if remaining:
                self._multi_ended_allocator.free(torch.cat(remaining))
        self._alloc_iter = None

    def is_slot_allocated(self, slot) -> bool:
        return self._multi_ended_allocator.is_slot_allocated(int(slot))

    def allocator_state_str(self) -> str:
        return self._multi_ended_allocator.allocator_state_str()


class UnifiedHybridReqToTokenPool(HybridReqToTokenPool):
    """`HybridReqToTokenPool` whose `mamba_pool` is a `UnifiedMambaPool`. The inherited
    mamba-id state now holds VIRTUAL ids; adds `translate_mamba_indices` for v->p."""

    def __init__(
        self,
        *,
        unified_buffer: UnifiedKVPool,
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
        spec_state_sub_pool: Optional[str] = None,
    ):
        self._unified_buffer = unified_buffer
        self._mamba_sub_pool_name = mamba_sub_pool_name
        self._spec_state_sub_pool = spec_state_sub_pool
        self._shared_mamba_size = (
            unified_buffer.max_slots(mamba_sub_pool_name) - 1
        )  # reserve slot 0
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
        speculative_eagle_topk: Optional[int] = None,
        mamba_envelope_layout: bool = False,
        enable_linear_replayssm: bool = False,
        linear_replayssm_cache_len: int = 16,
    ):
        # mamba_envelope_layout / speculative_eagle_topk / enable_linear_replayssm /
        # linear_replayssm_cache_len: accepted to match the parent signature but NOT
        # forwarded — the shared pool's conv/temporal state are fixed-shape views.
        assert mamba_size == self._shared_mamba_size, (
            f"UnifiedHybridReqToTokenPool._init_mamba_pool: mamba_size={mamba_size} "
            f"!= unified_buffer.max_slots({self._mamba_sub_pool_name!r}) - 1 "
            f"= {self._shared_mamba_size}"
        )
        assert len(cache_params.layers) >= len(mamba_layer_ids), (
            f"cache_params.layers ({len(cache_params.layers)}) cannot supply "
            f"{len(mamba_layer_ids)} mamba layer ids"
        )
        self.mamba_pool = UnifiedMambaPool(
            unified_buffer=self._unified_buffer,
            sub_pool_name=self._mamba_sub_pool_name,
            spec_state_size=mamba_spec_state_size,
            mamba_layer_ids=mamba_layer_ids,
            enable_memory_saver=self.enable_memory_saver,
            speculative_num_draft_tokens=speculative_num_draft_tokens,
            spec_state_sub_pool=self._spec_state_sub_pool,
        )
        # Wired in by init_unified_mamba_pools once the mamba allocator exists.
        self.mamba_allocator = None
        self.mamba_map = {layer_id: i for i, layer_id in enumerate(mamba_layer_ids)}
        self.mamba_ckpt_pool = None  # int8 ckpt pool unused; None = feature off
        self.device = device
        # Sized by req_to_token's first dim (size + 1; row 0 is padding); self.size
        # would under-size by one row.
        req_pool_size = self.req_to_token.shape[0]
        self.req_index_to_mamba_index_mapping: torch.Tensor = torch.zeros(
            req_pool_size, dtype=torch.int32, device=self.device
        )
        if enable_mamba_extra_buffer:
            self.req_index_to_mamba_ping_pong_track_buffer_mapping: torch.Tensor = (
                torch.zeros(
                    (req_pool_size, self.mamba_ping_pong_track_buffer_size),
                    # int64 to match the parent's uncast index_put source (int32 dest
                    # would dtype-mismatch on the first radix prefill).
                    dtype=torch.int64,
                    device=self.device,
                )
            )

    def translate_mamba_indices(self, virtual_ids: torch.Tensor) -> torch.Tensor:
        """Virtual mamba ids -> physical slot ids."""
        return self.mamba_allocator.translate(virtual_ids).to(torch.int32)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class UnifiedPoolBundle(NamedTuple):
    unified_memory_pool: UnifiedKVPool
    token_to_kv_pool: object  # HybridLinearKVPool
    token_to_kv_pool_allocator: object  # UnifiedMambaTokenToKVPoolAllocator
    req_to_token_pool: object  # UnifiedHybridReqToTokenPool


def init_unified_mamba_pools(
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
    total_bytes: int,
    max_num_reqs: int,
    enable_memory_saver: bool,
    enable_mamba_extra_buffer: bool,
    speculative_num_draft_tokens: Optional[int],
    disable_overlap_schedule: bool,
    need_sort: bool,
    forward_stream: Optional[torch.cuda.Stream] = None,
    lazy_compaction: bool = False,
    spec_state_size: Optional[int] = None,
    draft_kv_geometry: Optional[MHARegionGeometry] = None,
    kv_lora_rank: Optional[int] = None,
    qk_rope_head_dim: Optional[int] = None,
) -> UnifiedPoolBundle:
    """Build the Mamba-hybrid unified-memory-pool stack."""
    from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool
    from sglang.srt.mem_cache.multi_ended_allocator import (
        SPEC_BAND_ALIGNMENT_MARGIN_SLOTS,
        UnifiedMambaTokenToKVPoolAllocator,
    )

    # Full sub-pool is page-aware; mamba stays page=1 (state is per-request).
    assert page_size >= 1, f"page_size must be >= 1, got {page_size}"

    store_dtype = _store_dtype_for(kv_cache_dtype)
    # full-attn at the high-byte end (grow-down), mamba at the low-byte end (grow-up).
    # With draft_kv_geometry (fused draft KV), each full slot carries
    # the draft model's KV region; the draft worker binds a UnifiedDraftKVPool
    # over the same slots.
    full_spec: SubPoolSpecLike
    if use_mla_backend:
        # MLA-hybrid (e.g. KDA + MLA): the full sub-pool stores one latent
        # region per layer. Constraints of this path, asserted here rather
        # than inherited from the CLI hooks (which can be bypassed):
        assert kv_lora_rank is not None and qk_rope_head_dim is not None, (
            "unified MLA-hybrid path needs the model's kv_lora_rank / "
            "qk_rope_head_dim"
        )
        # The fused draft-KV region geometry is MHA-shaped; no MLA-hybrid
        # draft checkpoint exists to drive an MLA-shaped one.
        assert draft_kv_geometry is None, (
            "unified MLA-hybrid path does not support a fused draft KV region"
        )
        # The mamba-radix machinery of the current MLA-hybrid models runs
        # non-overlap at page_size 1; the pool-side layout is page-size-general
        # but the runtime combination beyond this is unvalidated.
        assert page_size == 1, (
            f"unified MLA-hybrid path requires page_size == 1; got {page_size}"
        )
        assert disable_overlap_schedule, (
            "unified MLA-hybrid path requires --disable-overlap-schedule"
        )
        # Every MLA write path must route through the pool with the physical
        # loc (KVWriteLoc / the ForwardBatch rebind); non-CUDA platforms have
        # pool-bypassing MLA write kernels that no tripwire can catch.
        assert device.startswith("cuda"), (
            f"unified MLA-hybrid path is CUDA-only; got device={device!r}"
        )
        assert store_dtype in (torch.bfloat16, torch.float16), (
            f"unified MLA-hybrid path supports bf16/fp16 KV only; got "
            f"{store_dtype}"
        )
        full_spec = MLASubPoolSpec(
            name="full",
            layer_num=len(full_attention_layer_ids),
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            store_dtype=store_dtype,
            grow_direction="down",
        )
    else:
        full_spec = MHASubPoolSpec(
            name="full",
            layer_num=len(full_attention_layer_ids),
            head_num=head_num,
            head_dim=head_dim,
            store_dtype=store_dtype,
            grow_direction="down",
            draft_region=draft_kv_geometry,
        )
    cp = mamba2_cache_params
    mamba_spec = MambaSubPoolSpec(
        name="mamba",
        layer_num=len(mamba_layer_ids),
        conv_state_shapes=tuple(tuple(int(x) for x in s) for s in cp.shape.conv),
        conv_dtype=cp.dtype.conv,
        temporal_state_shape=tuple(int(x) for x in cp.shape.temporal),
        temporal_dtype=cp.dtype.temporal,
        grow_direction="up",
    )
    sub_pool_specs: List[SubPoolSpecLike] = [full_spec, mamba_spec]
    # `total_bytes` is the WHOLE profiled buffer budget B, passed in directly
    # (the same `unified_total_bytes` the SWA path carries). The
    # full-KV / mamba-state / spec-state frontiers all float within it, so we do
    # NOT reconstruct it from `max_total_num_tokens * e_f + slots * mamba_epp`:
    # that reconstruction re-imposes the N-shared partition floor, whereas
    # `max_total_num_tokens` is now only the single-request feasibility LABEL
    # (the per-request clamp / usage denominator). The mamba slots and the spec
    # band are carved from within B by their sub-pool specs below.
    assert total_bytes > 0, (
        f"unified mamba buffer needs a positive byte budget; got {total_bytes}"
    )
    # Spec decoding: insert the intermediate-state float band as a MIDDLE
    # sub-pool (spec_state_size + 1 slots — one per verify-batch row plus the
    # reserved slot 0 — plus SPEC_BAND_ALIGNMENT_MARGIN_SLOTS of alignment
    # headroom so a full-`bs` placement never fails on band-page rounding). Its
    # bytes live WITHIN B (already reserved by the admission solve); at runtime
    # the band occupies exactly the current verify bs and the rest sits in the
    # gaps, available to full/mamba growth.
    spec_state_sub_pool: Optional[str] = None
    if speculative_num_draft_tokens is not None:
        if spec_state_size is None:
            spec_state_size = max_num_reqs
        spec_state_sub_pool = "spec_state"
        spec_state_spec = SpecStateSubPoolSpec(
            name=spec_state_sub_pool,
            layer_num=len(mamba_layer_ids),
            num_draft_tokens=speculative_num_draft_tokens,
            conv_window_shapes=tuple(
                tuple(int(x) for x in s) for s in cp.shape.conv
            ),
            conv_dtype=cp.dtype.conv,
            ssm_state_shape=tuple(int(x) for x in cp.shape.temporal),
            ssm_dtype=cp.dtype.temporal,
        )
        sub_pool_specs.append(spec_state_spec)
    else:
        spec_state_size = max_num_reqs
    shared_pool = UnifiedKVPool(
        total_bytes=total_bytes,
        sub_pool_specs=sub_pool_specs,
        device=device,
        enable_memory_saver=enable_memory_saver,
        page_size=page_size,
    )
    req_to_token_pool = UnifiedHybridReqToTokenPool(
        unified_buffer=shared_pool,
        mamba_sub_pool_name="mamba",
        size=max_num_reqs,
        mamba_spec_state_size=spec_state_size,  # outer dim of spec-decode intermediates
        max_context_len=model_context_len + extra_max_context_len,
        device=device,
        enable_memory_saver=enable_memory_saver,
        cache_params=mamba2_cache_params,
        mamba_layer_ids=mamba_layer_ids,
        enable_mamba_extra_buffer=enable_mamba_extra_buffer,
        speculative_num_draft_tokens=speculative_num_draft_tokens,
        enable_overlap_schedule=not disable_overlap_schedule,
        start_layer=start_layer,
        spec_state_sub_pool=spec_state_sub_pool,
    )
    unified_full_kv_pool: KVCache
    if use_mla_backend:
        unified_full_kv_pool = UnifiedMLATokenToKVPool(
            unified_buffer=shared_pool,
            sub_pool_name="full",
            page_size=page_size,
            start_layer=start_layer,
            end_layer=end_layer,
        )
    else:
        unified_full_kv_pool = UnifiedMHATokenToKVPool(
            unified_buffer=shared_pool,
            sub_pool_name="full",
            page_size=page_size,
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
        device=device,
        mamba_pool=req_to_token_pool.mamba_pool,
        enable_memory_saver=enable_memory_saver,
        use_mla=use_mla_backend,
        start_layer=start_layer,
        full_kv_pool=unified_full_kv_pool,
    )
    allocator = UnifiedMambaTokenToKVPoolAllocator(
        unified_buffer=shared_pool,
        kvcache=token_to_kv_pool,
        device=device,
        page_size=page_size,
        need_sort=need_sort,
        forward_stream=forward_stream,
        lazy_compaction=lazy_compaction,
        spec_state_sub_pool=spec_state_sub_pool,
    )

    # Wrap the composite's mamba MultiEndedAllocator in a slot allocator (PHYSICAL view).
    # Share the composite's band-park so a mamba-slot alloc shortfall gets the same
    # "movable-under-pressure" retry the full-KV alloc paths already have.
    mamba_slot_allocator = UnifiedMambaSlotAllocator(
        allocator.mamba_allocator,
        max_size=req_to_token_pool._shared_mamba_size,
        device=device,
        reclaim_band=allocator._reclaim_spec_state_band,
        # Byte-budgeted capacity for the alloc_group prefetch cap; _shared_
        # mamba_size is INDEX space (~4x this) and must not bound the prefetch.
        prefetch_cap=max_mamba_cache_size,
    )
    # `_mamba_translate` feeds the HiCache offload path, GATED OFF here — wired but inert.
    req_to_token_pool.mamba_allocator = mamba_slot_allocator
    token_to_kv_pool._mamba_translate = mamba_slot_allocator.translate
    # NOTE: no full-KV translate hook is wired. The model-side MLA entry
    # points (`set_mla_kv_buffer`/`get_mla_kv_buffer`) receive PHYSICAL ids:
    # writes carry the ForwardBatch's rebound `out_cache_loc`
    # (apply_unified_kv_loc_rebind), reads are translated at their production
    # sites (fetch_mha_one_shot_kv_indices / prepare_chunked_kv_indices).

    logger.info(
        "[unified-memory-pool] ============================================================"
    )
    logger.info(
        "[unified-memory-pool] UNIFIED MEMORY POOL ENABLED -- path=Mamba hybrid"
    )
    logger.info(
        "[unified-memory-pool]   full_layers=%d, mamba_layers=%d, head_num=%d, head_dim=%d, "
        "page_size=%d, is_draft_worker=%s",
        len(full_attention_layer_ids),
        len(mamba_layer_ids),
        head_num,
        head_dim,
        page_size,
        is_draft_worker,
    )
    logger.info(
        "[unified-memory-pool]   total_bytes=%d, max_total_num_tokens=%d, max_mamba_cache_size=%d, "
        "max_num_reqs=%d, speculative_num_draft_tokens=%s",
        total_bytes,
        max_total_num_tokens,
        max_mamba_cache_size,
        max_num_reqs,
        speculative_num_draft_tokens,
    )
    logger.info(
        "[unified-memory-pool] ============================================================"
    )
    return UnifiedPoolBundle(
        unified_memory_pool=shared_pool,
        token_to_kv_pool=token_to_kv_pool,
        token_to_kv_pool_allocator=allocator,
        req_to_token_pool=req_to_token_pool,
    )


# ---------------------------------------------------------------------------
# UnifiedSWAKVPool — hybrid SWA on the shared byte buffer
# ---------------------------------------------------------------------------


class UnifiedSWAKVPool(SWAKVPool):
    """Shared-buffer replacement for `SWAKVPool`.

    Composes two `UnifiedMHATokenToKVPool` instances (full + swa) aliasing the same
    byte buffer. Inherits from `SWAKVPool` only for `isinstance`; does NOT call the
    parent `__init__` (it would build static-partition pools). The per-sub-pool v2p
    table IS the full->swa mapping, so `register_mapping` is a no-op.
    """

    def __init__(
        self,
        *,
        unified_buffer: UnifiedKVPool,
        swa_attention_layer_ids: List[int],
        full_attention_layer_ids: List[int],
        page_size: int = 1,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        enable_memory_saver: bool = False,
    ):
        # Do NOT call super().__init__ — it would allocate static-partition pools.
        self.unified_buffer = unified_buffer
        self.swa_layer_nums = len(swa_attention_layer_ids)
        self.full_layer_nums = len(full_attention_layer_ids)
        self.layer_num = self.full_layer_nums + self.swa_layer_nums
        self.start_layer = start_layer if start_layer is not None else 0
        self.page_size = page_size
        self.layer_transfer_counter = None

        self.size = unified_buffer.max_slots("full") - 1
        self.size_swa = unified_buffer.max_slots("swa") - 1

        full_spec = unified_buffer.mha_spec("full")
        swa_spec = unified_buffer.mha_spec("swa")
        assert full_spec.store_dtype == swa_spec.store_dtype, (
            "UnifiedSWAKVPool: full and swa sub-pools must share store_dtype; got "
            f"full={full_spec.store_dtype}, swa={swa_spec.store_dtype}"
        )
        self.dtype = full_spec.store_dtype
        self.head_num = full_spec.head_num
        self.head_dim = full_spec.head_dim
        self.device = unified_buffer.device

        self.full_kv_pool = UnifiedMHATokenToKVPool(
            unified_buffer=unified_buffer,
            sub_pool_name="full",
            page_size=page_size,
            start_layer=start_layer,
            end_layer=end_layer,
        )
        self.swa_kv_pool = UnifiedMHATokenToKVPool(
            unified_buffer=unified_buffer,
            sub_pool_name="swa",
            page_size=page_size,
            start_layer=start_layer,
            end_layer=end_layer,
        )

        # disagg/nvlink disabled; keep attrs present to avoid AttributeError.
        self.enable_custom_mem_pool = False
        self.custom_mem_pool = None

        # {global_layer_id: (per-pool index, is_swa_layer)}
        self.layers_mapping: Dict[int, Tuple[int, bool]] = {}
        for idx, gid in enumerate(full_attention_layer_ids):
            self.layers_mapping[gid] = (idx, False)
        for idx, gid in enumerate(swa_attention_layer_ids):
            self.layers_mapping[gid] = (idx, True)

        # None so dispatch routes through our v2p-table overrides, not a registered mapping.
        self.full_to_swa_index_mapping: Optional[torch.Tensor] = None

        self.mem_usage = 0.0  # cosmetic; UnifiedKVPool logs the real size

        # Wired in via attach_allocators.
        self._full_allocator = None
        self._swa_allocator = None

        logger.info(
            "[unified-memory-pool] UnifiedSWAKVPool wrapped unified buffer: "
            "full_layers=%d (max_slots=%d), swa_layers=%d (max_slots=%d), "
            "head_num=%d, head_dim=%d",
            self.full_layer_nums,
            unified_buffer.max_slots("full"),
            self.swa_layer_nums,
            unified_buffer.max_slots("swa"),
            self.head_num,
            self.head_dim,
        )

    # -- allocator wiring --

    def attach_allocators(self, *, full_allocator, swa_allocator) -> None:
        """Wire the two `MultiEndedAllocator`s whose v2p tables translate slot ids."""
        self._full_allocator = full_allocator
        self._swa_allocator = swa_allocator

    # -- BaseSWAKVPool ABC surface --

    def register_mapping(self, full_to_swa_index_mapping: torch.Tensor) -> None:
        return  # no-op in shared mode (the swa-side v2p IS the mapping)

    def translate_loc_from_full_to_swa(self, kv_indices: torch.Tensor):
        """Virtual token ids -> swa-physical token ids (int32)."""
        assert self._swa_allocator is not None, (
            "UnifiedSWAKVPool.translate_loc_from_full_to_swa called before "
            "attach_allocators"
        )
        ps = self._swa_allocator.page_size
        if ps == 1:
            return self._swa_allocator.virtual_to_physical[kv_indices].to(torch.int32)
        virt_pages = kv_indices // ps
        offsets = kv_indices % ps
        swa_phys_pages = self._swa_allocator.virtual_to_physical[virt_pages]
        return (swa_phys_pages * ps + offsets).to(torch.int32)

    def get_state_buf_infos(self):
        return self.swa_kv_pool.get_contiguous_buf_infos()

    # -- size/info --

    def get_kv_size_bytes(self):
        return 0, 0  # UnifiedKVPool logs the total; per-side would double-count

    def get_contiguous_buf_infos(self):
        return self.full_kv_pool.get_contiguous_buf_infos()

    # -- buffer accessors --

    def get_key_buffer(self, layer_id: int):
        self._wait_for_layer(layer_id)
        pool_layer_id, is_swa = self.layers_mapping[layer_id]
        pool = self.swa_kv_pool if is_swa else self.full_kv_pool
        return pool.get_key_buffer(pool_layer_id)

    def get_value_buffer(self, layer_id: int):
        self._wait_for_layer(layer_id)
        pool_layer_id, is_swa = self.layers_mapping[layer_id]
        pool = self.swa_kv_pool if is_swa else self.full_kv_pool
        return pool.get_value_buffer(pool_layer_id)

    def get_kv_buffer(self, layer_id: int):
        self._wait_for_layer(layer_id)
        pool_layer_id, is_swa = self.layers_mapping[layer_id]
        pool = self.swa_kv_pool if is_swa else self.full_kv_pool
        return pool.get_kv_buffer(pool_layer_id)

    # -- kv writing --

    def set_kv_buffer(
        self,
        layer,
        loc_info,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
    ):
        """Route to the right sub-pool. Both `swa_loc` and `full_loc` are PHYSICAL
        (pre-translated once per forward by the attention backend); never translates here.
        """
        _, swa_loc, full_loc = unwrap_write_loc(loc_info)
        layer_id = layer.layer_id
        pool_layer_id, is_swa = self.layers_mapping[layer_id]
        if is_swa:
            # swa_loc is ALREADY swa-physical. Routed through the UnifiedMHATokenToKVPool
            # override (its 4-D layer-major view can't take the parent's view(-1, row_dim)).
            assert swa_loc is not None, (
                "UnifiedSWAKVPool.set_kv_buffer: SWA layer received no swa_loc; the "
                "attention backend must bundle forward_metadata.swa_out_cache_loc."
            )
            self.swa_kv_pool.set_kv_buffer(
                None,
                swa_loc,
                cache_k,
                cache_v,
                k_scale,
                v_scale,
                layer_id_override=pool_layer_id,
            )
            return
        # Full layer: full_loc is full-physical, always precomputed (eager + cuda-graph).
        assert full_loc is not None, (
            "UnifiedSWAKVPool.set_kv_buffer: full layer received no full_loc; "
            "ForwardMetadata.out_cache_loc_full_physical must be precomputed for "
            "the unified memory pool."
        )
        self.full_kv_pool.set_kv_buffer(
            None,
            full_loc,
            cache_k,
            cache_v,
            k_scale,
            v_scale,
            layer_id_override=pool_layer_id,
        )

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        # Never called on the composite — compaction runs per-sub-pool via
        # UnifiedMHATokenToKVPool.move_kv_cache.
        raise NotImplementedError(
            "UnifiedSWAKVPool.move_kv_cache should not be called; compaction "
            "operates per-sub-pool via UnifiedMHATokenToKVPool.move_kv_cache."
        )

    # -- HiCache shims (translate virtual->physical, then delegate) --

    @staticmethod
    def _virt_tokens_to_phys_tokens(
        virt_tokens: torch.Tensor, allocator
    ) -> torch.Tensor:
        """Virtual TOKEN ids -> physical TOKEN ids (page-aware). Unbound pages yield
        negatives; callers filter via `swa_phys >= 0`."""
        ps = allocator.page_size
        if ps == 1:
            return allocator.virtual_to_physical[virt_tokens]
        virt_pages = virt_tokens // ps
        offsets = virt_tokens % ps
        phys_pages = allocator.virtual_to_physical[virt_pages]
        return phys_pages * ps + offsets

    def get_cpu_copy(self, indices, mamba_indices=None):
        assert self._full_allocator is not None
        assert self._swa_allocator is not None
        # `indices` are virtual TOKEN ids; translate per sub-pool.
        full_phys = self._virt_tokens_to_phys_tokens(indices, self._full_allocator)
        swa_phys = self._virt_tokens_to_phys_tokens(indices, self._swa_allocator)
        full_cpu = self.full_kv_pool.get_cpu_copy(full_phys)
        valid = swa_phys >= 0
        swa_cpu = None
        if bool(valid.any().item()):
            swa_cpu = self.swa_kv_pool.get_cpu_copy(swa_phys[valid])
        return {"full": full_cpu, "swa": swa_cpu}

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        assert self._full_allocator is not None
        full_phys = self._virt_tokens_to_phys_tokens(indices, self._full_allocator)
        self.full_kv_pool.load_cpu_copy(kv_cache_cpu["full"], full_phys)
        if kv_cache_cpu.get("swa") is not None:
            assert self._swa_allocator is not None
            swa_phys = self._virt_tokens_to_phys_tokens(indices, self._swa_allocator)
            self.swa_kv_pool.load_cpu_copy(kv_cache_cpu["swa"], swa_phys)


class UnifiedSWAPoolBundle(NamedTuple):
    unified_memory_pool: UnifiedKVPool
    token_to_kv_pool: object  # UnifiedSWAKVPool
    token_to_kv_pool_allocator: object  # UnifiedSWATokenToKVPoolAllocator


def init_unified_swa_pools(
    *,
    device: str,
    kv_cache_dtype: torch.dtype,
    head_num: int,
    head_dim: int,
    v_head_dim: int,
    swa_head_num: int,
    swa_head_dim: int,
    swa_v_head_dim: int,
    page_size: int,
    start_layer: int,
    end_layer: int,
    swa_attention_layer_ids: List[int],
    full_attention_layer_ids: List[int],
    full_max_total_num_tokens: int,
    swa_max_total_num_tokens: int,
    enable_memory_saver: bool,
    need_sort: bool,
    forward_stream: Optional[torch.cuda.Stream] = None,
    lazy_compaction: bool = False,
    draft_kv_geometry: Optional[MHARegionGeometry] = None,
    total_bytes: int,
) -> UnifiedSWAPoolBundle:
    """Build the SWA-hybrid unified-memory-pool stack.

    `total_bytes` is the buffer's byte budget DIRECTLY (required — the same
    `unified_total_bytes` contract as the mamba factory): the token counts are
    feasibility/label values, not a partition, and re-summing them would
    silently re-derive a split that no longer means anything. Every unified
    SWA configurator path sets it (`_solve_unified_pool_sizes`)."""
    from sglang.srt.mem_cache.multi_ended_allocator import (
        UnifiedSWATokenToKVPoolAllocator,
    )

    # Both sub-allocators are page-aware: one virtual ID space at PAGE granularity,
    # two physical sub-pools compacting pages independently.
    assert page_size >= 1, f"page_size must be >= 1, got {page_size}"
    assert (
        len(full_attention_layer_ids) > 0
    ), "SWA-hybrid with zero full-attention layers is degenerate"
    assert (
        len(swa_attention_layer_ids) > 0
    ), "SWA-hybrid with zero SWA-attention layers is degenerate"

    store_dtype = _store_dtype_for(kv_cache_dtype)
    # full-attn at the high-byte end (grow-down), swa at the low-byte end (grow-up).
    # With draft_kv_geometry (fused draft KV), each FULL slot carries
    # the draft model's KV region — the draft is dense, so it fuses into the
    # full sub-pool (a DSV4-style draft would instead attach to "swa").
    full_spec = MHASubPoolSpec(
        name="full",
        layer_num=len(full_attention_layer_ids),
        head_num=head_num,
        head_dim=head_dim,
        v_head_dim=v_head_dim,
        store_dtype=store_dtype,
        grow_direction="down",
        draft_region=draft_kv_geometry,
    )
    swa_spec = MHASubPoolSpec(
        name="swa",
        layer_num=len(swa_attention_layer_ids),
        head_num=swa_head_num,
        head_dim=swa_head_dim,
        v_head_dim=swa_v_head_dim,
        store_dtype=store_dtype,
        grow_direction="up",
    )
    # Fail loud rather than fall back to re-summing the token counts: a None/0
    # here means a configurator path skipped the unified feasibility solve
    # (e.g. an override that predates the unified branch) — exactly the
    # static-split regression this contract exists to prevent.
    assert total_bytes and total_bytes > 0, (
        "unified SWA buffer needs the byte budget (unified_total_bytes) from "
        f"the configurator's feasibility solve; got {total_bytes!r}"
    )
    shared_pool = UnifiedKVPool(
        total_bytes=total_bytes,
        sub_pool_specs=[full_spec, swa_spec],
        device=device,
        enable_memory_saver=enable_memory_saver,
        page_size=page_size,
    )
    token_to_kv_pool = UnifiedSWAKVPool(
        unified_buffer=shared_pool,
        swa_attention_layer_ids=swa_attention_layer_ids,
        full_attention_layer_ids=full_attention_layer_ids,
        page_size=page_size,
        start_layer=start_layer,
        end_layer=end_layer,
        enable_memory_saver=enable_memory_saver,
    )
    allocator = UnifiedSWATokenToKVPoolAllocator(
        unified_buffer=shared_pool,
        kvcache=token_to_kv_pool,
        device=device,
        full_max_total_num_tokens=full_max_total_num_tokens,
        swa_max_total_num_tokens=swa_max_total_num_tokens,
        page_size=page_size,
        need_sort=need_sort,
        forward_stream=forward_stream,
        lazy_compaction=lazy_compaction,
    )

    logger.info(
        "[unified-memory-pool] ============================================================"
    )
    logger.info("[unified-memory-pool] UNIFIED MEMORY POOL ENABLED -- path=SWA hybrid")
    logger.info(
        "[unified-memory-pool]   full_layers=%d, swa_layers=%d, head_num=%d, head_dim=%d, "
        "v_head_dim=%d, swa_head_num=%d, swa_head_dim=%d, swa_v_head_dim=%d, "
        "page_size=%d",
        len(full_attention_layer_ids),
        len(swa_attention_layer_ids),
        head_num,
        head_dim,
        v_head_dim,
        swa_head_num,
        swa_head_dim,
        swa_v_head_dim,
        page_size,
    )
    logger.info(
        "[unified-memory-pool]   total_bytes=%d (=%.2f GB), full_max_total_num_tokens=%d, "
        "swa_max_total_num_tokens=%d, joint_available=%d slots",
        total_bytes,
        total_bytes / GB,
        full_max_total_num_tokens,
        swa_max_total_num_tokens,
        allocator.available_size(),
    )
    logger.info(
        "[unified-memory-pool] ============================================================"
    )
    return UnifiedSWAPoolBundle(
        unified_memory_pool=shared_pool,
        token_to_kv_pool=token_to_kv_pool,
        token_to_kv_pool_allocator=allocator,
    )
