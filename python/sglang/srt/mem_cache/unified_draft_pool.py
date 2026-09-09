"""KV pools a DRAFT runner binds over the draft slots fused into the target's
unified pool: same pages, same slot ids, same v2p table as the target -- one
allocation, one free, one relocation."""

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from sglang.srt.mem_cache.layout.fused_draft import FusedDraftPlacement
from sglang.srt.mem_cache.memory_pool import (
    KVWriteLoc,
    MHATokenToKVPool,
    unwrap_write_loc,
    write_loc_id_space,
)
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.mem_cache.unified_memory_pool import UnifiedKVPool


class UnifiedDraftKVPool(MHATokenToKVPool):
    """Dense draft KV over the draft parts of one host sub-pool's entries.

    Per-layer `k_buffer` / `v_buffer` are views of the draft parts inside
    every slot of the host sub-pool (`UnifiedKVPool.build_dense_draft_views`);
    ``layer_slots`` maps each of this runner's layer ids to its region slot.
    Locs arriving through the KVCache API are the target's PHYSICAL token ids,
    produced by the allocator's translate (the id-space choke point binds it,
    see KVIndexTranslator); the pool exposes `host_allocator` for that
    binding. Relocation needs no method here: compaction moves whole page
    envelopes on the HOST pool, which carries the draft bytes; `move_kv_cache`
    raises so a stray per-slot move fails loudly instead of corrupting the
    fused layout.
    """

    requires_translated_write_loc = True

    def __init__(
        self,
        *,
        unified_buffer: UnifiedKVPool,
        host_sub_pool_name: str,
        host_allocator,
        layer_slots: Mapping[int, int],
        page_size: int = 1,
    ):
        region = unified_buffer.draft_host_spec(host_sub_pool_name).draft_region
        layer_ids = sorted(layer_slots)
        assert layer_ids, "UnifiedDraftKVPool binds at least one layer"
        start_layer = layer_ids[0]
        # The base pool indexes buffers by `layer_id - start_layer`.
        assert layer_ids == list(range(start_layer, start_layer + len(layer_ids))), (
            f"fused draft layer ids must be contiguous; got {layer_ids}"
        )
        slots = [layer_slots[layer_id] for layer_id in layer_ids]
        assert len(set(slots)) == len(slots) and all(
            0 <= s < region.layer_num for s in slots
        ), f"draft slots {slots} must be distinct and within range({region.layer_num})"
        k_views, v_views = unified_buffer.build_dense_draft_views(host_sub_pool_name)
        max_slots = unified_buffer.max_slots(host_sub_pool_name)

        self._unified_buffer = unified_buffer
        self._host_sub_pool_name = host_sub_pool_name
        # The id-space choke point (KVIndexTranslator) probes this: the draft
        # translates through the HOST allocator, exactly as the target does.
        self.host_allocator = host_allocator
        self.layer_slots: Dict[int, int] = dict(layer_slots)
        self._k_views: List[torch.Tensor] = [k_views[s] for s in slots]
        self._v_views: List[torch.Tensor] = [v_views[s] for s in slots]
        num_pages = max_slots // page_size

        super().__init__(
            size=num_pages * page_size - page_size,
            page_size=page_size,
            dtype=region.store_dtype,
            head_num=region.head_num,
            head_dim=region.head_dim,
            layer_num=len(layer_ids),
            device=unified_buffer.device,
            enable_memory_saver=False,  # buffer owned by UnifiedKVPool
            v_head_dim=region.resolved_v_head_dim(),
            start_layer=start_layer,
            end_layer=start_layer + len(layer_ids) - 1,
            enable_kv_cache_copy=False,
            # Same rationale as UnifiedMHATokenToKVPool: the env-driven layout
            # selectors must not re-shape buffers this pool builds itself.
            kv_cache_layout="page_major",
        )

    def _create_buffers(self):
        self.k_buffer = self._k_views
        self.v_buffer = self._v_views

    def _clear_buffers(self):
        pass  # lifetime owned by UnifiedKVPool

    def get_kv_size_bytes(self):
        return 0, 0  # fused into the host entries; UnifiedKVPool logs the total

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        raise NotImplementedError(
            "fused draft KV relocates with the HOST pool's whole-page move; a "
            "draft-side per-slot move would corrupt the fused layout"
        )

    def get_contiguous_buf_infos(self):
        raise NotImplementedError(
            "fused draft KV has no per-layer contiguous regions; KV transfer / "
            "disaggregation is unsupported."
        )

    def get_cpu_copy(self, indices, mamba_indices=None):
        raise NotImplementedError("CPU offloading is unsupported for fused draft KV.")

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        raise NotImplementedError("CPU offloading is unsupported for fused draft KV.")


class UnifiedDraftSWAKVPool(SWAKVPool):
    """A draft's KV pool when it has sliding-window layers: one dense draft
    pool per host sub-pool it fuses into ("full" and/or "swa"), routed per
    layer exactly like the target's `UnifiedSWAKVPool`, so the attention
    backends run their swa rail for it. Inherits `SWAKVPool` for `isinstance`
    only; never calls its `__init__` (it would allocate static pools)."""

    requires_translated_write_loc = True

    def __init__(
        self,
        *,
        unified_buffer: UnifiedKVPool,
        host_allocator,
        page_size: int,
        full_layer_slots: Mapping[int, int],
        swa_layer_slots: Mapping[int, int],
    ):
        assert swa_layer_slots, (
            "UnifiedDraftSWAKVPool binds at least one window layer; a full-only "
            "draft binds UnifiedDraftKVPool"
        )
        self.unified_buffer = unified_buffer
        self.host_allocator = host_allocator
        self.page_size = page_size
        self.device = unified_buffer.device
        self.layer_transfer_counter = None
        self.full_layer_nums = len(full_layer_slots)
        self.swa_layer_nums = len(swa_layer_slots)
        self.layer_num = self.full_layer_nums + self.swa_layer_nums
        self.start_layer = min([*full_layer_slots, *swa_layer_slots])
        self.size = unified_buffer.max_slots("full") - 1
        self.size_swa = unified_buffer.max_slots("swa") - 1

        self.full_kv_pool: Optional[UnifiedDraftKVPool] = None
        if full_layer_slots:
            self.full_kv_pool = self._side_pool("full", full_layer_slots)
        self.swa_kv_pool = self._side_pool("swa", swa_layer_slots)
        lead = self.full_kv_pool if self.full_kv_pool is not None else self.swa_kv_pool
        self.dtype = lead.dtype
        self.head_num = lead.head_num
        self.head_dim = lead.head_dim

        # {layer_id: (per-side index, is_swa_layer)}, sides indexed in layer order.
        self.layers_mapping: Dict[int, Tuple[int, bool]] = {}
        for idx, layer_id in enumerate(sorted(full_layer_slots)):
            self.layers_mapping[layer_id] = (idx, False)
        for idx, layer_id in enumerate(sorted(swa_layer_slots)):
            self.layers_mapping[layer_id] = (idx, True)
        # None so dispatch routes through the host's v2p tables, never a
        # registered mapping.
        self.full_to_swa_index_mapping: Optional[torch.Tensor] = None
        self.enable_custom_mem_pool = False
        self.custom_mem_pool = None
        self.dsa_kv_cache_store_fp8 = False
        self.kv_cache_dim = None
        self.index_head_dim = None
        self.mem_usage = 0.0  # fused into the host entries

    def _side_pool(
        self, host_sub_pool_name: str, layer_slots: Mapping[int, int]
    ) -> UnifiedDraftKVPool:
        # The side is indexed 0..n-1 in layer order (`layer_id_override`).
        return UnifiedDraftKVPool(
            unified_buffer=self.unified_buffer,
            host_sub_pool_name=host_sub_pool_name,
            host_allocator=self.host_allocator,
            layer_slots={
                idx: layer_slots[layer_id]
                for idx, layer_id in enumerate(sorted(layer_slots))
            },
            page_size=self.page_size,
        )

    def _side(self, layer_id: int) -> Tuple[UnifiedDraftKVPool, int]:
        pool_layer_id, is_swa = self.layers_mapping[layer_id]
        pool = self.swa_kv_pool if is_swa else self.full_kv_pool
        assert pool is not None, layer_id
        return pool, pool_layer_id

    # -- KVCache surface a side of None cannot answer --

    @property
    def post_capture_active(self) -> bool:
        return False  # the host buffer is fully backed at boot

    @property
    def post_capture_backed_bytes(self) -> int:
        return 0

    def finalize_backing(self, config) -> None:
        return

    def register_layer_transfer_counter(self, layer_transfer_counter):
        self.layer_transfer_counter = layer_transfer_counter

    # -- BaseSWAKVPool ABC surface --

    def register_mapping(self, full_to_swa_index_mapping: torch.Tensor) -> None:
        return  # the host's swa v2p table IS the mapping

    def translate_loc_from_full_to_swa(self, kv_indices: torch.Tensor):
        """Virtual token ids -> swa-physical token ids, through the host."""
        return self.host_allocator.translate_loc_from_full_to_swa(kv_indices)

    def get_state_buf_infos(self):
        raise NotImplementedError(
            "fused draft KV has no per-layer contiguous regions; KV transfer / "
            "disaggregation is unsupported."
        )

    # -- size/info --

    def get_kv_size_bytes(self):
        return 0, 0  # fused into the host entries; UnifiedKVPool logs the total

    def get_contiguous_buf_infos(self):
        raise NotImplementedError(
            "fused draft KV has no per-layer contiguous regions; KV transfer / "
            "disaggregation is unsupported."
        )

    def get_v_head_dim(self):
        lead = self.full_kv_pool if self.full_kv_pool is not None else self.swa_kv_pool
        return lead.get_value_buffer(lead.start_layer).shape[-1]

    # -- buffer accessors --

    def get_key_buffer(self, layer_id: int):
        pool, pool_layer_id = self._side(layer_id)
        return pool.get_key_buffer(pool_layer_id)

    def get_value_buffer(self, layer_id: int):
        pool, pool_layer_id = self._side(layer_id)
        return pool.get_value_buffer(pool_layer_id)

    def get_kv_buffer(self, layer_id: int):
        pool, pool_layer_id = self._side(layer_id)
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
        """Route to the right side. Both locs are kernel-facing already (the
        backend derives `swa_loc` once per forward); never translates here."""
        loc, swa_loc, full_loc = unwrap_write_loc(loc_info)
        id_space = write_loc_id_space(loc_info)
        pool, pool_layer_id = self._side(layer.layer_id)
        if pool is self.swa_kv_pool:
            assert swa_loc is not None, (
                "UnifiedDraftSWAKVPool.set_kv_buffer: window layer received no "
                "swa_loc; the attention backend must bundle "
                "forward_metadata.swa_out_cache_loc."
            )
            side_loc = swa_loc
        else:
            side_loc = loc if full_loc is None else full_loc
        pool.set_kv_buffer(
            None,
            KVWriteLoc(side_loc, id_space=id_space),
            cache_k,
            cache_v,
            k_scale,
            v_scale,
            layer_id_override=pool_layer_id,
        )

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        raise NotImplementedError(
            "fused draft KV relocates with the HOST pool's whole-page move; a "
            "draft-side per-slot move would corrupt the fused layout"
        )

    def get_cpu_copy(self, indices, mamba_indices=None, req_pool_index=None):
        raise NotImplementedError("CPU offloading is unsupported for fused draft KV.")

    def load_cpu_copy(
        self, kv_cache_cpu, indices, mamba_indices=None, req_pool_index=None
    ):
        raise NotImplementedError("CPU offloading is unsupported for fused draft KV.")


def fused_draft_host_allocator(token_to_kv_pool: Any) -> Optional[Any]:
    """The host allocator a fused draft pool translates through, or None for
    any other pool."""
    if isinstance(token_to_kv_pool, (UnifiedDraftKVPool, UnifiedDraftSWAKVPool)):
        return token_to_kv_pool.host_allocator
    return None


def draft_kv_layer_ids(model) -> List[int]:
    """Layer ids of every attention layer the draft model writes KV for, in
    layer order."""
    from sglang.srt.layers.radix_attention import RadixAttention

    return sorted(
        {m.layer_id for m in model.modules() if isinstance(m, RadixAttention)}
    )


def build_unified_draft_kv_pool(
    *,
    unified_buffer: UnifiedKVPool,
    host_allocator,
    placement: FusedDraftPlacement,
    runner: int,
    kv_layer_ids: Sequence[int],
    swa_layer_ids: Sequence[int],
    page_size: int,
) -> MHATokenToKVPool:
    """The KV pool draft runner ``runner`` binds over its fused slots.

    The placement sized the slots from the draft config; the model's real
    layer ids fill them in layer order, so a count mismatch is a loud boot
    failure, never a silent alias. Window layers follow the placement: they
    bind the swa sub-pool when it holds a region for them, else the full
    sub-pool, where a full lifetime covers any window.
    """
    swa_placed = placement.region("swa") is not None
    swa = set(swa_layer_ids) if swa_placed else set()
    full_ids = [layer_id for layer_id in kv_layer_ids if layer_id not in swa]
    swa_ids = [layer_id for layer_id in kv_layer_ids if layer_id in swa]
    full_slots = placement.slots_for(runner, "full")
    swa_slots = placement.slots_for(runner, "swa")
    assert len(full_ids) == len(full_slots) and len(swa_ids) == len(swa_slots), (
        f"draft runner {runner}: layers full={full_ids} swa={swa_ids} vs placed "
        f"slots full={list(full_slots)} swa={list(swa_slots)}"
    )
    if not swa_ids:
        return UnifiedDraftKVPool(
            unified_buffer=unified_buffer,
            host_sub_pool_name="full",
            host_allocator=host_allocator,
            layer_slots=dict(zip(full_ids, full_slots)),
            page_size=page_size,
        )
    return UnifiedDraftSWAKVPool(
        unified_buffer=unified_buffer,
        host_allocator=host_allocator,
        page_size=page_size,
        full_layer_slots=dict(zip(full_ids, full_slots)),
        swa_layer_slots=dict(zip(swa_ids, swa_slots)),
    )
