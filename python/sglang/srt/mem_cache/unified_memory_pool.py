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
"""UnifiedKVPool — one physical `uint8` byte buffer shared by 2 sub-pools.

Two `MultiEndedAllocator`s grow from opposite ends; eager-compacting `free`
keeps each pool's byte range hole-free. Layout is envelope-major (a slot's data
for all its layers in one contiguous byte envelope) so a freed slot vacates a
region the peer can grow into. Everything above the allocator stores virtual
slot IDs; the allocator owns the per-sub-pool virtual<->physical tables and
compaction only mutates those (no reference rewriting).
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
from sglang.srt.mem_cache.layout.page_major import (
    build_page_major_mamba_views,
    build_page_major_mha_views,
    build_page_major_mla_views,
    mla_entry_bytes,
)
from sglang.srt.mem_cache.memory_pool import (
    HybridReqToTokenPool,
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
    """Per-slot layout of one MHA-shaped sub-pool. `v_head_dim` defaults to `head_dim`."""

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
        assert (
            self.v_head_dim > 0
        ), f"v_head_dim must be positive; got {self.v_head_dim}"

    def k_row_bytes(self) -> int:
        return self.head_num * self.head_dim * self.store_dtype.itemsize

    def v_row_bytes(self) -> int:
        return self.head_num * self.v_head_dim * self.store_dtype.itemsize

    def entry_bytes(self) -> int:
        return self.layer_num * (self.k_row_bytes() + self.v_row_bytes())

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


class MLASubPoolSpec(msgspec.Struct, frozen=True, kw_only=True):
    """Layout spec for an MLA full-attention sub-pool: ONE latent region per
    layer (`kv_cache_dim = kv_lora_rank + qk_rope_head_dim`), no K/V pair.

    msgspec (not the grandfathered `@dataclass` hierarchy); `UnifiedKVPool`
    dispatches by isinstance, so nothing needs the shared base class."""

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
# hierarchy plus the msgspec-based MLA spec.
SubPoolSpecLike = Union[SubPoolSpec, MLASubPoolSpec]


# ---------------------------------------------------------------------------
# UnifiedKVPool — the byte buffer + the strided per-sub-pool views
# ---------------------------------------------------------------------------


class UnifiedKVPool:
    """One physical `uint8` byte buffer shared by 2 sub-pools, each exposing
    strided per-layer views. Allocators keep byte ranges disjoint; no usage tracking here.
    """

    def __init__(
        self,
        *,
        total_bytes: int,
        sub_pool_specs: List[SubPoolSpec],
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
            if isinstance(s, MLASubPoolSpec):
                s.validate()
        up_specs = [s for s in sub_pool_specs if s.grow_direction == "up"]
        down_specs = [s for s in sub_pool_specs if s.grow_direction == "down"]
        assert len(up_specs) == 1 and len(down_specs) == 1, (
            f"UnifiedKVPool needs exactly one grow-up and one grow-down end "
            f"sub-pool; got directions "
            f"{[s.grow_direction for s in sub_pool_specs]}"
        )
        names = [s.name for s in sub_pool_specs]
        assert len(set(names)) == 2, f"sub-pool names must be unique; got {names}"
        directions = sorted(s.grow_direction for s in sub_pool_specs)
        assert directions == ["down", "up"], (
            f"UnifiedKVPool needs one grow-up and one grow-down sub-pool; "
            f"got {directions}"
        )

        self.device = device
        self.total_bytes = total_bytes
        self.sub_pool_specs = sub_pool_specs
        self._page_size = page_size
        self._specs_by_name: Dict[str, SubPoolSpec] = {
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
        # MHA: (k_buffer, v_buffer); Mamba: (conv_state_list, temporal_state)
        self._mha_views: Dict[str, Tuple[List[torch.Tensor], List[torch.Tensor]]] = {}
        self._mamba_views: Dict[str, Tuple[List[torch.Tensor], torch.Tensor]] = {}
        # MLA sub-pools: one latent view list per layer (no K/V pair).
        self._mla_views: Dict[str, List[torch.Tensor]] = {}

        # Slot-0 dummy writes for both pools land in [0, entry_max); each pool's
        # first allocatable slot is chosen so real data starts at >= entry_max.
        entry_max = max(s.entry_bytes() for s in sub_pool_specs)

        for spec in sub_pool_specs:
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
            elif isinstance(spec, MambaSubPoolSpec):
                self._mamba_views[spec.name] = self._build_mamba_views(
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
            len(sub_pool_specs),
        )
        for s in sub_pool_specs:
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

    # -- introspection --

    def spec(self, name: str) -> SubPoolSpec:
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

    def mamba_views_for(self, name: str) -> Tuple[List[torch.Tensor], torch.Tensor]:
        return self._mamba_views[name]

    def _build_mha_views(
        self,
        spec: MHASubPoolSpec,
        anchor_bytes: int,
        max_slots: int,
        page_size: int,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
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
        if tgt_loc.numel() == 0:
            return
        with record_function("UnifiedMHA.move_kv_cache"):
            move_kv_cache_native(
                self.k_buffer,
                self.v_buffer,
                tgt_loc,
                src_loc,
                page_size=self._page_size,
            )

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
            k_view = self.k_buffer[layer_id]
            v_view = self.v_buffer[layer_id]
            ps = self._page_size
            N = loc.numel()
            if N == 0:
                return
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
    points (`set_mla_kv_buffer`/`get_mla_kv_buffer`) also receive PHYSICAL ids:
    `forward_batch.out_cache_loc` is rebound to physical once at ForwardBatch
    construction (see `apply_unified_kv_loc_rebind`), so every downstream
    consumer — backend and model code alike — sees only physical ids."""

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
        # `loc` is PHYSICAL (the ForwardBatch-level rebind).
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

        # Per-draft-token intermediate buffers have a different outer size
        # (spec_state_size+1), so they're NOT in the shared buffer; allocate locally.
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
                    device=unified_buffer.device,
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
                        device=unified_buffer.device,
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
        MambaPool.copy_from(self, src_index, dst_index)


class UnifiedMambaSlotAllocator:
    """Mamba slot allocator (PHYSICAL view) for the unified memory pool.

    Owns slot alloc/free, sizing, and the v<->p mapping (``translate``), presenting the
    upstream ``MambaSlotAllocator`` interface. ``alloc()`` returns VIRTUAL ids and does
    NOT clear state — clearing is deferred to ``UnifiedMambaPool.clear_slots``.
    """

    def __init__(self, mea, max_size: int, device: str):
        self._multi_ended_allocator = mea
        self._max_size = max_size  # excludes reserved slot 0
        self._device = device
        self._alloc_iter = None  # active alloc_group batch iterator

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
        return self._multi_ended_allocator.alloc(need_size)  # VIRTUAL ids

    def free(self, free_index: torch.Tensor):
        return self._multi_ended_allocator.free(free_index)

    def clear(self):
        self._alloc_iter = None
        return self._multi_ended_allocator.clear()

    def alloc_group_begin(self, num_reqs: int):
        """Pre-allocate a batch that ``alloc(1)`` then draws from."""
        self._alloc_iter = None
        if num_reqs > 0:
            result = self._multi_ended_allocator.alloc(num_reqs)
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
    ):
        self._unified_buffer = unified_buffer
        self._mamba_sub_pool_name = mamba_sub_pool_name
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
    max_num_reqs: int,
    enable_memory_saver: bool,
    enable_mamba_extra_buffer: bool,
    speculative_num_draft_tokens: Optional[int],
    disable_overlap_schedule: bool,
    need_sort: bool,
    mamba_full_memory_ratio: Optional[float] = None,  # informational only
    forward_stream: Optional[torch.cuda.Stream] = None,
    lazy_compaction: bool = False,
    kv_lora_rank: Optional[int] = None,
    qk_rope_head_dim: Optional[int] = None,
) -> UnifiedPoolBundle:
    """Build the Mamba-hybrid unified-memory-pool stack."""
    from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool
    from sglang.srt.mem_cache.multi_ended_allocator import (
        UnifiedMambaTokenToKVPoolAllocator,
    )

    # Full sub-pool is page-aware; mamba stays page=1 (state is per-request).
    assert page_size >= 1, f"page_size must be >= 1, got {page_size}"

    store_dtype = _store_dtype_for(kv_cache_dtype)
    # full-attn at the high-byte end (grow-down), mamba at the low-byte end (grow-up).
    full_spec: SubPoolSpecLike
    if use_mla_backend:
        # MLA-hybrid (e.g. KDA + MLA): the full sub-pool stores one latent
        # region per layer. Constraints of this path, asserted here rather
        # than inherited from the CLI hooks (which can be bypassed):
        assert kv_lora_rank is not None and qk_rope_head_dim is not None, (
            "unified MLA-hybrid path needs the model's kv_lora_rank / "
            "qk_rope_head_dim"
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
        # Every MLA write path must route through the pool doors, which the
        # ForwardBatch-level rebind feeds PHYSICAL ids; non-CUDA platforms have
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
    total_bytes = (
        max_total_num_tokens * full_spec.entry_bytes()
        + max_mamba_cache_size * mamba_spec.entry_bytes()
    )
    shared_pool = UnifiedKVPool(
        total_bytes=total_bytes,
        sub_pool_specs=[full_spec, mamba_spec],
        device=device,
        enable_memory_saver=enable_memory_saver,
        page_size=page_size,
    )
    req_to_token_pool = UnifiedHybridReqToTokenPool(
        unified_buffer=shared_pool,
        mamba_sub_pool_name="mamba",
        size=max_num_reqs,
        mamba_spec_state_size=max_num_reqs,  # outer dim of spec-decode intermediates
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
    )

    # Wrap the composite's mamba MultiEndedAllocator in a slot allocator (PHYSICAL view).
    mamba_slot_allocator = UnifiedMambaSlotAllocator(
        allocator.mamba_allocator,
        max_size=req_to_token_pool._shared_mamba_size,
        device=device,
    )
    # `_mamba_translate` feeds the HiCache offload path, GATED OFF here — wired but inert.
    req_to_token_pool.mamba_allocator = mamba_slot_allocator
    token_to_kv_pool._mamba_translate = mamba_slot_allocator.translate
    # NOTE: model-side MLA entry points (`set_mla_kv_buffer`/`get_mla_kv_buffer`
    # on zero-prefix extends) consume `forward_batch.out_cache_loc`, which the
    # ForwardBatch-level rebind (`apply_unified_kv_loc_rebind`) has already
    # translated to PHYSICAL — the pool doors never translate.

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
    if mamba_full_memory_ratio is not None:
        logger.info(
            "[unified-memory-pool]   mamba_full_memory_ratio=%s governs the total budget only, "
            "not the runtime split.",
            mamba_full_memory_ratio,
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
) -> UnifiedSWAPoolBundle:
    """Build the SWA-hybrid unified-memory-pool stack."""
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
    full_spec = MHASubPoolSpec(
        name="full",
        layer_num=len(full_attention_layer_ids),
        head_num=head_num,
        head_dim=head_dim,
        v_head_dim=v_head_dim,
        store_dtype=store_dtype,
        grow_direction="down",
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
    total_bytes = (
        full_max_total_num_tokens * full_spec.entry_bytes()
        + swa_max_total_num_tokens * swa_spec.entry_bytes()
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
