"""Draft-model regions nested inside a unified sub-pool's slot entry.

A fused draft stores its KV as extra parts of every host slot's entry, after
the host parts, so one slot id per token covers target and draft: one
allocation, one free, one whole-page relocation carry both. A region is not a
`SubPoolSpec`: it has no grow direction and no frontier logic, only the
geometry the host entry lays out at its `draft_offset_in_entry()`.
"""

from typing import List, Optional, Sequence, Tuple

import msgspec
import torch

from sglang.srt.mem_cache.layout.page_major import DensePart


class DenseDraftRegion(msgspec.Struct, frozen=True, kw_only=True):
    """Geometry of the DRAFT model's K/V rows fused into every slot of a host
    sub-pool. The draft is a separate checkpoint, so its head geometry and
    layer count differ from the host's; its K and V rows are two more parts of
    the host entry, indexed by the host's physical token id."""

    layer_num: int
    head_num: int
    head_dim: int
    store_dtype: torch.dtype
    v_head_dim: Optional[int] = None

    def validate(self) -> None:
        assert self.layer_num > 0, f"layer_num must be positive; got {self.layer_num}"
        assert self.head_num > 0, f"head_num must be positive; got {self.head_num}"
        assert self.head_dim > 0, f"head_dim must be positive; got {self.head_dim}"
        v = self.resolved_v_head_dim()
        assert v > 0, f"v_head_dim must be positive; got {v}"

    def resolved_v_head_dim(self) -> int:
        return self.head_dim if self.v_head_dim is None else self.v_head_dim

    def k_row_bytes(self) -> int:
        return self.head_num * self.head_dim * self.store_dtype.itemsize

    def v_row_bytes(self) -> int:
        return self.head_num * self.resolved_v_head_dim() * self.store_dtype.itemsize

    def entry_bytes(self) -> int:
        """Draft bytes per slot, before the host entry's alignment."""
        return self.layer_num * (self.k_row_bytes() + self.v_row_bytes())

    def parts(self, offset_bytes: int) -> Tuple[DensePart, DensePart]:
        """The draft's K and V parts, laid out from ``offset_bytes`` inside
        the host entry."""
        layer_stride = self.k_row_bytes() + self.v_row_bytes()
        return (
            DensePart(
                name="draft_k",
                offset_bytes=offset_bytes,
                layer_stride_bytes=layer_stride,
                layer_num=self.layer_num,
                row_shape=(self.head_num, self.head_dim),
                dtype=self.store_dtype,
            ),
            DensePart(
                name="draft_v",
                offset_bytes=offset_bytes + self.k_row_bytes(),
                layer_stride_bytes=layer_stride,
                layer_num=self.layer_num,
                row_shape=(self.head_num, self.resolved_v_head_dim()),
                dtype=self.store_dtype,
            ),
        )


_HOST_NAMES = ("full",)


class RunnerSlots(msgspec.Struct, frozen=True, kw_only=True):
    """One draft runner's slots inside each host region, as ``(start, count)``."""

    full: Tuple[int, int] = (0, 0)

    def range_for(self, host: str) -> range:
        if host == "full":
            start, count = self.full
            return range(start, start + count)
        return range(0)


class FusedDraftPlacement(msgspec.Struct, frozen=True, kw_only=True):
    """Where every draft runner's layers live inside the host sub-pools.

    One region per host sub-pool that carries draft slots (``None`` = the
    draft has no layer of that kind) plus each runner's slot ranges into it.
    Built once on the target, stored on the `UnifiedKVPool`, and read back by
    each draft runner, so the two sides cannot disagree on a slot.
    """

    runners: Tuple[RunnerSlots, ...]
    full: Optional[DenseDraftRegion] = None

    def __post_init__(self):
        assert len(self.runners) > 0, "a placement needs at least one draft runner"
        for host in _HOST_NAMES:
            region = self.region(host)
            slots = [s for r in self.runners for s in r.range_for(host)]
            if region is None:
                assert not slots, f"host {host!r} has draft slots but no region"
                continue
            assert slots == list(range(region.layer_num)), (
                f"host {host!r}: runner slot ranges {slots} must tile "
                f"range({region.layer_num}) in runner order"
            )

    def region(self, host: str) -> Optional[DenseDraftRegion]:
        if host == "full":
            return self.full
        return None

    def hosts(self) -> Tuple[str, ...]:
        return tuple(h for h in _HOST_NAMES if self.region(h) is not None)

    def slots_for(self, runner: int, host: str) -> range:
        return self.runners[runner].range_for(host)

    @classmethod
    def from_counts(
        cls, *, full_counts: Sequence[int], full: Optional[DenseDraftRegion]
    ) -> "FusedDraftPlacement":
        """Tile each host's region with the runners' layer counts, in runner order."""
        runners = []
        start = 0
        for count in full_counts:
            runners.append(RunnerSlots(full=(start, count)))
            start += count
        return cls(runners=tuple(runners), full=full)



class DraftKVGeometry(msgspec.Struct, frozen=True, kw_only=True):
    """Per-GPU K/V row geometry of one kind of draft attention layer."""

    head_num: int
    head_dim: int
    v_head_dim: int


class DraftKVProfile(msgspec.Struct, frozen=True, kw_only=True):
    """What the draft checkpoint asks of the host sub-pools.

    ``num_depths`` > 1 marks a per-depth head (one transformer block per MTP
    depth, served by one runner each under multi-layer EAGLE); otherwise
    every runner serves all ``num_layers`` layers.
    """

    num_layers: int
    full: DraftKVGeometry
    swa_layer_ids: Tuple[int, ...] = ()
    num_depths: int = 1
    num_state_layers: int = 0


def draft_kv_profile(
    draft_model_config, *, num_layers: int, attn_tp_size: int
) -> DraftKVProfile:
    """The profile of a draft `ModelConfig`, heads divided by attn_tp the way
    the target divides its own (drafts never join the DCP group)."""
    from sglang.srt.configs.hybrid_arch import mambaish_config

    mc = draft_model_config
    swa_layer_ids: Tuple[int, ...] = ()
    if mc.is_hybrid_swa and not mc.is_deepseek_v4_arch:
        swa_layer_ids = tuple(int(i) for i in mc.swa_attention_layer_ids)
    num_depths = mc.num_nextn_predict_layers
    mambaish = mambaish_config(mc)
    return DraftKVProfile(
        num_layers=int(num_layers),
        full=DraftKVGeometry(
            head_num=int(mc.get_num_kv_heads(attn_tp_size)),
            head_dim=int(mc.head_dim),
            v_head_dim=int(mc.v_head_dim),
        ),
        swa_layer_ids=swa_layer_ids,
        num_depths=1 if num_depths is None else int(num_depths),
        num_state_layers=(
            0 if mambaish is None else len(mambaish.mamba2_cache_params.layers)
        ),
    )


class FusedDraftDecision(msgspec.Struct, frozen=True, kw_only=True):
    """`place_fused_draft`'s answer: the placement, or why the draft keeps a
    private pool. Neither means fusion simply does not apply."""

    placement: Optional[FusedDraftPlacement] = None
    declined: Optional[str] = None


def _runner_layer_counts(
    profile: DraftKVProfile, num_runners: int
) -> Tuple[Optional[List[Tuple[int, int]]], Optional[str]]:
    """Per runner, its (full, swa) layer counts; or why no runner layout exists."""
    if profile.num_depths <= 1:
        num_swa = len(profile.swa_layer_ids)
        return [(profile.num_layers - num_swa, num_swa)] * num_runners, None
    if num_runners == 1:
        return None, (
            f"a per-depth draft head ({profile.num_depths} depths) needs one "
            "runner per depth (multi-layer EAGLE)"
        )
    if num_runners > profile.num_depths:
        return None, (
            f"{num_runners} draft runners exceed the head's "
            f"{profile.num_depths} depths"
        )
    swa = set(profile.swa_layer_ids)
    return [(0, 1) if r in swa else (1, 0) for r in range(num_runners)], None


def place_fused_draft(
    *,
    profile: DraftKVProfile,
    num_runners: int,
    host_names: Sequence[str],
    store_dtype: torch.dtype,
) -> FusedDraftDecision:
    """Assign every draft layer of every runner to the host sub-pool whose
    lifetime covers what the layer reads: a full-attention layer rides in
    ``"full"``. A layer kind no host arm serves declines the whole draft to
    its private pool."""
    counts, reason = _runner_layer_counts(profile, num_runners)
    if counts is None:
        return FusedDraftDecision(declined=reason)
    if profile.num_state_layers:
        return FusedDraftDecision(
            declined=(
                f"the draft has {profile.num_state_layers} recurrent-state "
                "layer(s) of its own, which no host state pool carries"
            )
        )
    num_swa = sum(swa for _, swa in counts)
    if num_swa:
        return FusedDraftDecision(
            declined=(
                f"the draft has {num_swa} SWA layer(s) of its own, which the "
                "fused dense pool cannot serve"
            )
        )
    geometry = profile.full
    if geometry.head_dim != geometry.v_head_dim:
        return FusedDraftDecision(
            declined=(
                "the draft's K/V rows are asymmetric "
                f"(head_dim={geometry.head_dim}, v_head_dim={geometry.v_head_dim}), "
                "which is not admitted yet"
            )
        )
    assert "full" in host_names, host_names
    full = DenseDraftRegion(
        layer_num=sum(full for full, _ in counts),
        head_num=geometry.head_num,
        head_dim=geometry.head_dim,
        v_head_dim=geometry.v_head_dim,
        store_dtype=store_dtype,
    )
    return FusedDraftDecision(
        placement=FusedDraftPlacement.from_counts(
            full_counts=[full for full, _ in counts], full=full
        )
    )
