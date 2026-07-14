import logging
from typing import Optional, Union

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.mamba.causal_conv1d_triton import PAD_SLOT_ID
from sglang.srt.layers.attention.mamba.mamba import MambaMixer2
from sglang.srt.layers.attention.mamba.mamba2_metadata import (
    ForwardMetadata,
    Mamba2Metadata,
)
from sglang.srt.layers.attention.mamba.mamba_state_scatter_triton import (
    fused_conv_window_scatter_with_mask,
    fused_mamba_state_scatter_with_mask,
    track_mamba_states_if_needed,
)
from sglang.srt.environ import envs
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.server_args import get_global_server_args
from sglang.srt.speculative.eagle_info import EagleDraftInput, EagleVerifyInput
from sglang.srt.speculative.spec_info import SpecInput

logger = logging.getLogger(__name__)

_DEBUG_SPEC_BAND = envs.SGLANG_DEBUG_SPEC_BAND.get()


class MambaAttnBackendBase(AttentionBackend):
    def __init__(self, model_runner: ModelRunner):
        super().__init__()
        self.pad_slot_id = PAD_SLOT_ID
        self.device = model_runner.device
        self.topk = model_runner.server_args.speculative_eagle_topk or 0
        self.is_draft_worker = model_runner.is_draft_worker
        self.req_to_token_pool: HybridReqToTokenPool = model_runner.req_to_token_pool
        self.token_to_kv_pool = model_runner.token_to_kv_pool
        self.enable_unified_memory = model_runner.server_args.enable_unified_memory
        self.forward_metadata: ForwardMetadata = None
        # Spec-decode intermediate index plumbing. Static max-rows int32 buffer
        # of per-verify-row indices into the intermediate_ssm /
        # intermediate_conv_window buffers: identity rows for the dense pool
        # (never rewritten), translated physical band slots under the unified
        # pool (refreshed in-place each verify — captured graphs read the
        # POINTER). getattr probe: only the unified mamba composite defines
        # spec_state_allocator.
        self._spec_state_allocator = getattr(
            model_runner.token_to_kv_pool_allocator, "spec_state_allocator", None
        )
        self.intermediate_state_indices_buf = torch.arange(
            self.req_to_token_pool.size, dtype=torch.int32, device=self.device
        )
        # High-water mark of rows the last unified refresh wrote: rows in
        # [n, high) hold stale ids and must be re-routed to the sink on shrink.
        # Starts at the full buffer (the initial identity contents are NOT
        # sink-safe under the unified pool).
        self._intermediate_rows_high = self.intermediate_state_indices_buf.shape[0]
        self.state_indices_list = []
        # Static (max_bs,) track-dest buffer captured by pointer, refreshed in-place
        # each replay; the captured track-save reads this, not the InputBuffer slot.
        self.mamba_track_indices_buf = None
        # Per-bs static write-cursor / force-flush buffers for cuda-graph; None
        # unless --enable-linear-replayssm is set.
        self.replayssm_write_pos_list = None
        self.replayssm_force_flush_list = None
        self.query_start_loc_list = []
        self.retrieve_next_token_list = []
        self.retrieve_next_sibling_list = []
        self.retrieve_parent_token_list = []
        self.cached_cuda_graph_decode_query_start_loc: torch.Tensor = None
        self.cached_cuda_graph_verify_query_start_loc: torch.Tensor = None
        self.conv_states_shape: tuple[int, int] = None

    def _execute_deferred_mamba_cow_and_clear(self, forward_batch: ForwardBatch):
        """Run deferred clear/COW ops on the forward stream to avoid races."""
        if (
            not forward_batch.forward_mode.is_extend()
            or forward_batch.forward_mode.is_target_verify()
            or forward_batch.forward_mode.is_draft_extend_v2()
            or self.is_draft_worker
        ):
            return
        if (
            forward_batch.mamba_clear_indices is not None
            and len(forward_batch.mamba_clear_indices) > 0
        ):
            # mamba_pool is a pure PHYSICAL store; translate before zeroing or
            # clear_slots zeroes the wrong physical slots.
            self.req_to_token_pool.mamba_pool.clear_slots(
                self._translate_mamba_indices(forward_batch.mamba_clear_indices)
            )
        if (
            forward_batch.mamba_cow_src_indices is not None
            and len(forward_batch.mamba_cow_src_indices) > 0
        ):
            ckpt_pool = getattr(self.req_to_token_pool, "mamba_ckpt_pool", None)
            if ckpt_pool is not None:
                # int8 checkpoints: dequantize src int8 ckpt slot into the active bf16 dst.
                ckpt_pool.load_to_active(
                    self.req_to_token_pool.mamba_pool,
                    forward_batch.mamba_cow_src_indices,
                    forward_batch.mamba_cow_dst_indices,
                )
            else:
                # mamba_pool is a pure PHYSICAL store; translate both COW slot ids.
                self.req_to_token_pool.mamba_pool.copy_from(
                    self._translate_mamba_indices(forward_batch.mamba_cow_src_indices),
                    self._translate_mamba_indices(forward_batch.mamba_cow_dst_indices),
                )
        forward_batch.mamba_clear_indices = None
        forward_batch.mamba_cow_src_indices = None
        forward_batch.mamba_cow_dst_indices = None

    def _translate_mamba_indices(self, mamba_indices: torch.Tensor) -> torch.Tensor:
        """Virtual->physical mamba slot-id translate (identity for the non-unified
        pool). Must run everywhere mamba ids feed the SSM/conv kernels or mamba-pool
        state ops, incl. the cuda-graph replay-prep copy into ``state_indices_list``."""
        return self.req_to_token_pool.translate_mamba_indices(mamba_indices)

    def _assert_verify_indices_fresh(
        self,
        mamba_cache_indices: torch.Tensor,
        intermediate_state_indices: torch.Tensor,
        bs: int,
    ) -> None:
        """DIAGNOSTIC (SGLANG_DEBUG_SPEC_BAND): at a TARGET_VERIFY metadata build,
        assert (a) every REAL translated mamba-cache slot points to a LIVE mamba
        page (mamba p2v != -1) — catches a STALE virtual->physical translation
        that reads a freed/wrong slot; and (b) the band's physical slots
        (intermediate_state_indices, which the verify WRITES) are byte-disjoint
        from the mamba-cache slots it READS this step — catches the band sitting
        on top of the very states being read/committed. Byte space, because band
        and mamba are distinct sub-pools with different entry_bytes.
        """
        band = self._spec_state_allocator
        if band is None or not hasattr(band, "low_peer"):
            return
        mamba = band.low_peer
        if mamba is None or not hasattr(mamba, "physical_to_virtual"):
            return
        # Real rows only: padded/tombstoned rows carry -1.
        mci = mamba_cache_indices[:bs]
        real = mci[mci >= 0].to(torch.int64)
        if real.numel() == 0:
            return
        # (a) freshness: translated physical slots must be occupied.
        p2v = mamba.physical_to_virtual
        in_range = (real >= 0) & (real < mamba.num_pages)
        if not bool(in_range.all().item()):
            msg = (
                f"[spec-band] verify mamba index OUT OF RANGE: "
                f"{real[~in_range][:8].tolist()} vs num_pages={mamba.num_pages}"
            )
            logger.error(msg)
            raise AssertionError(msg)
        live = p2v[real] != -1
        if not bool(live.all().item()):
            msg = (
                f"[spec-band] STALE mamba translation: verify reads physical slots "
                f"{real[~live][:8].tolist()} that are FREE (p2v==-1) — the "
                f"virtual->physical map is stale; the SSM/conv read gets garbage. "
                f"mamba={mamba.allocator_state_str()}"
            )
            logger.error(msg)
            raise AssertionError(msg)
        # (b) band-write vs mamba-read disjoint (byte space).
        isi = intermediate_state_indices[:bs].to(torch.int64)
        b_lo = isi * int(band.entry_bytes_per_page)
        b_hi = b_lo + int(band.entry_bytes_per_page)
        m_lo = real * int(mamba.entry_bytes_per_page)
        m_hi = m_lo + int(mamba.entry_bytes_per_page)
        # cross product (bs is small at verify): any band-row byte range hitting
        # any mamba-read byte range is corruption.
        overlap = (b_lo.unsqueeze(1) < m_hi.unsqueeze(0)) & (
            b_hi.unsqueeze(1) > m_lo.unsqueeze(0)
        )
        if bool(overlap.any().item()):
            msg = (
                f"[spec-band] BAND WRITE ALIASES MAMBA READ: this verify's band slots "
                f"(intermediate {isi[:8].tolist()}) share bytes with the mamba states "
                f"it reads/commits (physical {real[:8].tolist()}). The intermediate "
                f"write will clobber the recurrent state mid-verify. "
                f"band={band.allocator_state_str()} mamba={mamba.allocator_state_str()}"
            )
            logger.error(msg)
            raise AssertionError(msg)

    def _refresh_intermediate_state_indices(self, num_rows: int) -> torch.Tensor:
        """Refresh + return the static per-verify-row intermediate index buffer.

        Dense pool: the buffer holds identity rows permanently — return as-is.
        Unified pool: rows [0, num_rows) get the band's PHYSICAL slots. The
        band mapping is a CPU-known affine (`v2p[r] == band_start_page + r`,
        both plain ints from `place_band`), so the refresh is ONE
        `torch.arange(out=)` launch — no gather, no staging buffers. Rows
        beyond the band (shrink tail, padded cuda-graph rows, dummy verifies
        with an unplaced band) route to the physical slot-0 sink; the
        `_intermediate_rows_high` high-water mark bounds the tail zeroing to
        the actually-stale range (kernel-free in the steady-bs case).
        """
        buf = self.intermediate_state_indices_buf
        band = self._spec_state_allocator
        if band is None:
            return buf
        n = min(int(num_rows), buf.shape[0])
        # Pinned-band coverage: a PINNED band is a real verify placement and must cover
        # every real row; a mismatch means the bs changed after
        # place_spec_state_band_for_decode — fail loud, not silent sink
        # corruption. Unpinned (dummy/warmup/capture) verifies legitimately
        # run against an unplaced band: their rows go to the sink by design.
        num_placed = band.num_placed_slots
        assert not band._band_pinned or n <= num_placed, (
            f"pinned-band coverage violated: verify rows n={n} exceed the pinned band's "
            f"num_placed_slots={num_placed} ({band.allocator_state_str()}); "
            "the batch size changed between band placement and the verify "
            "metadata build."
        )
        k = min(n, num_placed)
        if k > 0:
            torch.arange(
                band.band_start_page,
                band.band_start_page + k,
                dtype=torch.int32,
                device=self.device,
                out=buf[:k],
            )
        if self._intermediate_rows_high > k:
            buf[k : self._intermediate_rows_high].zero_()  # stale rows -> sink
        self._intermediate_rows_high = k
        return buf

    def _forward_metadata(self, forward_batch: ForwardBatch):
        bs = forward_batch.batch_size

        retrieve_next_token = None
        retrieve_next_sibling = None
        retrieve_parent_token = None
        intermediate_state_indices = None
        track_conv_indices = None
        track_ssm_h_src = None
        track_ssm_h_dst = None
        track_ssm_final_src = None
        track_ssm_final_dst = None

        mamba_cache_indices = self.req_to_token_pool.get_mamba_indices(
            forward_batch.req_pool_indices
        )
        # Translate virtual->physical BEFORE the padding sentinel below, so the
        # gather reads only real ids; padded rows are then poisoned to -1 (skipped).
        mamba_cache_indices = self._translate_mamba_indices(mamba_cache_indices)
        if forward_batch.mamba_track_indices is not None:
            # This one is written BACK onto the ForwardBatch, because the track
            # helpers below read it from there rather than taking it as an
            # argument. That makes it the only translate here that is not
            # idempotent: translating twice gives v2p[v2p[v]], which is a valid
            # index into a wrong slot -- no crash, just wrong state.
            #
            # Nothing builds metadata twice for one ForwardBatch today, so this is
            # a latent hazard, not a live bug. Fail loudly if that ever changes
            # (a retry, a second backend in the decode group, a piecewise
            # re-build) instead of silently corrupting the mamba state.
            if getattr(forward_batch, "_mamba_track_indices_translated", False):
                raise RuntimeError(
                    "mamba_track_indices was already translated for this "
                    "ForwardBatch; a second translate would map v2p[v2p[v]] and "
                    "silently write the mamba state to the wrong slots. Hold the "
                    "physical ids in a backend-owned buffer instead of writing "
                    "them back onto the ForwardBatch."
                )
            forward_batch.mamba_track_indices = self._translate_mamba_indices(
                forward_batch.mamba_track_indices
            )
            forward_batch._mamba_track_indices_translated = True
        _real_bs = forward_batch._original_batch_size
        if _real_bs is not None and _real_bs < mamba_cache_indices.shape[0]:
            mamba_cache_indices = mamba_cache_indices.clone()
            mamba_cache_indices[_real_bs:] = -1

        replayssm_write_pos = None
        replayssm_force_flush = None
        if forward_batch.forward_mode.is_decode_or_idle():
            query_start_loc = torch.arange(
                0, bs + 1, dtype=torch.int32, device=self.device
            )
            # The ring cursor is a per-slot decode counter shared by all GDN layers;
            # manage it once here (snapshot, hand to layers, advance mod L), not per-layer.
            mamba_pool = getattr(self.req_to_token_pool, "mamba_pool", None)
            write_pos_buf = (
                getattr(mamba_pool, "replayssm_write_pos", None)
                if mamba_pool is not None
                else None
            )
            if write_pos_buf is not None:
                slots = mamba_cache_indices.to(torch.long)
                # Padded rows carry slot == -1; clamp the gather in-bounds (kernel
                # zeroes padded rows via state_idx < 0).
                safe_slots = slots.clamp(min=0)
                replayssm_write_pos = write_pos_buf[safe_slots].clone()
                L = mamba_pool.linear_replayssm_cache_len
                # KDA has no radix coordination: flush only on the natural write_pos
                # == L-1 wrap. GDN adds the radix-aligned force-flush below.
                is_kda = getattr(mamba_pool, "replayssm_is_kda", False)
                # Force-flush on the radix track's seq_lens % mamba_track_interval
                # == 0 boundary so the ring folds into temporal[slot] when read.
                if not is_kda:
                    force_flush_bool = self._replayssm_track_flush_mask(
                        forward_batch.seq_lens_cpu, bs
                    )
                    replayssm_force_flush = force_flush_bool.to(
                        device=self.device, dtype=torch.int32
                    )
                # Advance only valid slots, scattered over unique slots (dup-index
                # race; padded rows clamp to 0); a forced flush -> next write_pos 0.
                valid_mask = slots >= 0
                valid_slots = slots[valid_mask]
                if valid_slots.numel() > 0:
                    flushed = replayssm_write_pos == (L - 1)
                    if replayssm_force_flush is not None:
                        flushed = flushed | (replayssm_force_flush != 0)
                    next_pos = torch.where(
                        flushed,
                        torch.zeros_like(replayssm_write_pos),
                        (replayssm_write_pos + 1) % L,
                    )
                    # Dedup: rows sharing a slot share write_pos/flush, so the
                    # scattered value is identical regardless of which row wins.
                    uniq_slots, inv = torch.unique(valid_slots, return_inverse=True)
                    next_for_valid = next_pos[valid_mask]
                    new_vals = torch.empty(
                        uniq_slots.shape[0],
                        dtype=write_pos_buf.dtype,
                        device=write_pos_buf.device,
                    )
                    new_vals[inv] = next_for_valid.to(write_pos_buf.dtype)
                    write_pos_buf[uniq_slots] = new_vals
        elif forward_batch.forward_mode.is_extend(include_draft_extend_v2=True):
            if forward_batch.forward_mode.is_draft_extend_v2():
                # DRAFT_EXTEND_V2 runs only full-attn layers in the draft model;
                # skip mamba metadata.
                query_start_loc = None
            elif forward_batch.forward_mode.is_target_verify():
                query_start_loc = torch.arange(
                    0,
                    forward_batch.input_ids.shape[0] + 1,
                    step=forward_batch.spec_info.draft_token_num,
                    dtype=torch.int32,
                    device=forward_batch.input_ids.device,
                )
                intermediate_state_indices = self._refresh_intermediate_state_indices(
                    bs
                )
                if _DEBUG_SPEC_BAND:
                    self._assert_verify_indices_fresh(
                        mamba_cache_indices, intermediate_state_indices, bs
                    )

                if self.topk > 1:
                    retrieve_next_token = forward_batch.spec_info.retrieve_next_token
                    retrieve_next_sibling = (
                        forward_batch.spec_info.retrieve_next_sibling
                    )
                    # None during dummy run
                    if retrieve_next_token is not None:
                        # -1, NOT empty: the tree builder leaves a token's parent
                        # entry UNSET when it finds no parent for it (it warns
                        # "invalid eagle tree ... the token will be ignored" and
                        # skips). Uninitialized garbage there is not merely a wrong
                        # state read -- the GDN verify kernels multiply this value
                        # by the intermediate-state step stride (~1e7 elements) to
                        # form a pointer, so a junk entry is an illegal access. -1
                        # is the sentinel those kernels skip on. (The cuda-graph
                        # path allocates these zeroed, which is why only the eager
                        # path ever faulted.)
                        retrieve_parent_token = torch.full_like(
                            retrieve_next_token, -1
                        )
            else:
                query_start_loc = torch.empty(
                    (bs + 1,), dtype=torch.int32, device=self.device
                )
                query_start_loc[:bs] = forward_batch.extend_start_loc
                query_start_loc[bs] = (
                    forward_batch.extend_start_loc[-1]
                    + forward_batch.extend_seq_lens[-1]
                )
                if (
                    forward_batch.mamba_track_mask is not None
                    and forward_batch.mamba_track_mask.any()
                ):
                    track_conv_indices = self._init_track_conv_indices(
                        query_start_loc, forward_batch
                    )

                    (
                        track_ssm_h_src,
                        track_ssm_h_dst,
                        track_ssm_final_src,
                        track_ssm_final_dst,
                    ) = self._init_track_ssm_indices(mamba_cache_indices, forward_batch)
        else:
            raise ValueError(f"Invalid forward mode: {forward_batch.forward_mode=}")

        has_mamba_track_mask = bool(
            forward_batch.mamba_track_mask is not None
            and forward_batch.mamba_track_mask.any()
        )

        return ForwardMetadata(
            query_start_loc=query_start_loc,
            mamba_cache_indices=mamba_cache_indices,
            # Physical track destinations (None when tracking off); cuda-graph
            # supplies this via the static backend buffer in _replay_metadata.
            mamba_track_indices=getattr(forward_batch, "mamba_track_indices", None),
            retrieve_next_token=retrieve_next_token,
            retrieve_next_sibling=retrieve_next_sibling,
            retrieve_parent_token=retrieve_parent_token,
            intermediate_state_indices=intermediate_state_indices,
            track_conv_indices=track_conv_indices,
            track_ssm_h_src=track_ssm_h_src,
            track_ssm_h_dst=track_ssm_h_dst,
            track_ssm_final_src=track_ssm_final_src,
            track_ssm_final_dst=track_ssm_final_dst,
            has_mamba_track_mask=has_mamba_track_mask,
            replayssm_write_pos=replayssm_write_pos,
            replayssm_force_flush=replayssm_force_flush,
        )

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ):
        self.forward_metadata = self._replay_metadata(
            forward_batch.batch_size,
            forward_batch.req_pool_indices,
            forward_batch.forward_mode,
            forward_batch.spec_info,
            forward_batch.seq_lens_cpu if not in_capture else None,
            num_padding=(
                0 if in_capture else getattr(forward_batch, "num_padding", None)
            ),
            in_capture=in_capture,
            mamba_track_indices=getattr(forward_batch, "mamba_track_indices", None),
        )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        self._execute_deferred_mamba_cow_and_clear(forward_batch)
        self.forward_metadata = self._forward_metadata(forward_batch)

    def _init_track_conv_indices(
        self, query_start_loc: torch.Tensor, forward_batch: ForwardBatch
    ):
        """Flattened input positions of conv states to track during extend (up to
        the last complete chunk boundary, mamba_track_mask rows only)."""
        conv_state_len = self.conv_states_shape[-1]

        lens_to_track = (
            forward_batch.mamba_track_seqlens - forward_batch.extend_prefix_lens
        )
        mamba_cache_chunk_size = get_global_server_args().mamba_cache_chunk_size
        aligned_len = (lens_to_track // mamba_cache_chunk_size) * mamba_cache_chunk_size
        start_indices = query_start_loc[:-1] + aligned_len - conv_state_len
        start_indices = start_indices[forward_batch.mamba_track_mask]

        indices = start_indices.unsqueeze(-1) + torch.arange(
            conv_state_len,
            device=self.device,
            dtype=start_indices.dtype,
        )

        return indices.clamp(0, query_start_loc[-1] - 1)

    def _init_track_ssm_indices(
        self, mamba_cache_indices: torch.Tensor, forward_batch: ForwardBatch
    ):
        """src/dst indices to track SSM states for prefix caching: aligned seqs
        cache last_recurrent_state, unaligned cache intermediate `h` at the last
        chunk boundary."""
        mamba_cache_chunk_size = get_global_server_args().mamba_cache_chunk_size
        # CPU to avoid kernel launches for the masking ops
        mamba_track_mask = forward_batch.mamba_track_mask.cpu()
        extend_seq_lens = forward_batch.extend_seq_lens.cpu()
        mamba_track_indices = forward_batch.mamba_track_indices.cpu()
        mamba_cache_indices = mamba_cache_indices.cpu()
        mamba_track_seqlens = forward_batch.mamba_track_seqlens.cpu()
        prefix_lens = forward_batch.extend_prefix_lens.cpu()

        if isinstance(self, Mamba2AttnBackend):
            num_h_states = extend_seq_lens // mamba_cache_chunk_size
        else:
            num_h_states = (extend_seq_lens - 1) // mamba_cache_chunk_size + 1

        track_ssm_src_offset = torch.zeros_like(num_h_states)
        track_ssm_src_offset[1:] = torch.cumsum(num_h_states[:-1], dim=0)

        lens_to_track = mamba_track_seqlens - prefix_lens
        lens_masked = lens_to_track[mamba_track_mask]
        offset_masked = track_ssm_src_offset[mamba_track_mask]
        dst_masked = mamba_track_indices[mamba_track_mask]

        is_aligned = (lens_masked % mamba_cache_chunk_size) == 0

        # Aligned: last_recurrent_state from ssm_states.
        track_ssm_final_src = mamba_cache_indices[mamba_track_mask][is_aligned]
        track_ssm_final_dst = dst_masked[is_aligned]

        # Unaligned: intermediate state from h.
        # TODO: handle mamba_cache_chunk_size % page size != 0
        not_aligned = ~is_aligned
        track_ssm_h_src = offset_masked[not_aligned] + (
            lens_masked[not_aligned] // mamba_cache_chunk_size
        )
        track_ssm_h_dst = dst_masked[not_aligned]

        return (
            track_ssm_h_src.to(self.device, non_blocking=True),
            track_ssm_h_dst.to(self.device, non_blocking=True),
            track_ssm_final_src.to(self.device, non_blocking=True),
            track_ssm_final_dst.to(self.device, non_blocking=True),
        )

    def init_forward_metadata_capture_cpu_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
    ):
        self.forward_metadata = self._capture_metadata(
            bs, req_pool_indices, forward_mode, spec_info
        )

    def _replayssm_enabled(self) -> bool:
        """True iff --enable-linear-replayssm allocated the ring cursor
        (MambaPool.replayssm_write_pos doubles as the on/off gate)."""
        mamba_pool = getattr(self.req_to_token_pool, "mamba_pool", None)
        if mamba_pool is None:
            return False
        return getattr(mamba_pool, "replayssm_write_pos", None) is not None

    def _replayssm_track_flush_mask(
        self, seq_lens_cpu: torch.Tensor, bs: int
    ) -> torch.Tensor:
        """Per-row (length bs) bool flush mask = the radix track's seq_lens_cpu %
        mamba_track_interval == 0, so force-flush and snapshot fire on the same
        steps (no off-by-one)."""
        interval = get_global_server_args().mamba_track_interval
        if seq_lens_cpu is None:
            # Should not happen for the supported config; stay safe and never flush.
            return torch.zeros((bs,), dtype=torch.bool)
        mask = (seq_lens_cpu[:bs].to(torch.int64) % interval) == 0
        if mask.shape[0] < bs:
            pad = torch.zeros((bs - mask.shape[0],), dtype=torch.bool)
            mask = torch.cat([mask, pad])
        return mask.cpu()

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        assert (
            max_num_tokens % max_bs == 0
        ), f"max_num_tokens={max_num_tokens} must be divisible by max_bs={max_bs}"
        draft_token_num = max_num_tokens // max_bs
        # Per-bs static write-cursor / force-flush buffers, captured by pointer +
        # refreshed in-place each replay; sized like state_indices_list. None when off.
        self.replayssm_write_pos_list = [] if self._replayssm_enabled() else None
        self.replayssm_force_flush_list = [] if self._replayssm_enabled() else None
        # int64 to match DecodeInputBuffers.mamba_track_indices + the track-save
        # kernel's int64 index load. Refreshed in-place by _replay_metadata.
        self.mamba_track_indices_buf = torch.zeros(
            (max_bs,), dtype=torch.int64, device=self.device
        )
        for i in range(max_bs):
            self.state_indices_list.append(
                torch.full(
                    (i + 1,), self.pad_slot_id, dtype=torch.int32, device=self.device
                )
            )
            if self.replayssm_write_pos_list is not None:
                self.replayssm_write_pos_list.append(
                    torch.zeros((i + 1,), dtype=torch.int32, device=self.device)
                )
            if self.replayssm_force_flush_list is not None:
                self.replayssm_force_flush_list.append(
                    torch.zeros((i + 1,), dtype=torch.int32, device=self.device)
                )
            self.query_start_loc_list.append(
                torch.zeros((i + 2,), dtype=torch.int32, device=self.device)
            )
            self.retrieve_next_token_list.append(
                torch.zeros(
                    (i + 1, draft_token_num), dtype=torch.int32, device=self.device
                )
            )
            self.retrieve_next_sibling_list.append(
                torch.zeros(
                    (i + 1, draft_token_num), dtype=torch.int32, device=self.device
                )
            )
            self.retrieve_parent_token_list.append(
                torch.zeros(
                    (i + 1, draft_token_num), dtype=torch.int32, device=self.device
                )
            )
        self.cached_cuda_graph_decode_query_start_loc = torch.arange(
            0, max_bs + 1, dtype=torch.int32, device=self.device
        )
        self.cached_cuda_graph_verify_query_start_loc = torch.arange(
            0,
            max_bs * draft_token_num + 1,
            step=draft_token_num,
            dtype=torch.int32,
            device=self.device,
        )

    def init_cpu_graph_state(self, max_bs: int, max_num_tokens: int):
        assert (
            max_num_tokens % max_bs == 0
        ), f"max_num_tokens={max_num_tokens} must be divisible by max_bs={max_bs}"
        for i in range(max_bs):
            self.state_indices_list.append(
                torch.full(
                    (i + 1,), self.pad_slot_id, dtype=torch.int32, device=self.device
                )
            )
            self.query_start_loc_list.append(
                torch.empty((i + 2,), dtype=torch.int32, device=self.device)
            )
        self.cached_cuda_graph_decode_query_start_loc = torch.arange(
            0, max_bs + 1, dtype=torch.int32, device=self.device
        )

    def _capture_metadata(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
    ):
        if forward_mode.is_decode_or_idle():
            self.query_start_loc_list[bs - 1].copy_(
                self.cached_cuda_graph_decode_query_start_loc[: bs + 1]
            )
        elif forward_mode.is_target_verify():
            self.query_start_loc_list[bs - 1].copy_(
                self.cached_cuda_graph_verify_query_start_loc[: bs + 1]
            )
        else:
            raise ValueError(f"Invalid forward mode: {forward_mode=}")
        mamba_indices = self.req_to_token_pool.get_mamba_indices(req_pool_indices)
        # Captured Mamba kernels read state_indices_list as PHYSICAL ids; translate
        # before copying (no-op for non-unified pool).
        mamba_indices = self._translate_mamba_indices(mamba_indices)
        self.state_indices_list[bs - 1][: len(mamba_indices)].copy_(mamba_indices)

        # Capture records the pointer to the static per-bs buffers; their zeros are
        # overwritten in-place by _replay_metadata before each replay. None when off.
        replayssm_write_pos = (
            self.replayssm_write_pos_list[bs - 1]
            if self.replayssm_write_pos_list is not None
            else None
        )
        replayssm_force_flush = (
            self.replayssm_force_flush_list[bs - 1]
            if self.replayssm_force_flush_list is not None
            else None
        )

        # Capture records the POINTER to the static intermediate-index buffer;
        # contents are refreshed in-place by _replay_metadata before each
        # replay (identity/sink during the capture warm-run — harmless).
        intermediate_state_indices = (
            self.intermediate_state_indices_buf
            if forward_mode.is_target_verify()
            else None
        )

        if forward_mode.is_target_verify() and self.topk > 1:
            # retrieve_* are None during capture, so skip the copy.
            return ForwardMetadata(
                query_start_loc=self.query_start_loc_list[bs - 1],
                mamba_cache_indices=self.state_indices_list[bs - 1],
                retrieve_next_token=self.retrieve_next_token_list[bs - 1],
                retrieve_next_sibling=self.retrieve_next_sibling_list[bs - 1],
                retrieve_parent_token=self.retrieve_parent_token_list[bs - 1],
                intermediate_state_indices=intermediate_state_indices,
                replayssm_write_pos=replayssm_write_pos,
                replayssm_force_flush=replayssm_force_flush,
            )
        else:
            return ForwardMetadata(
                query_start_loc=self.query_start_loc_list[bs - 1],
                mamba_cache_indices=self.state_indices_list[bs - 1],
                intermediate_state_indices=intermediate_state_indices,
                replayssm_write_pos=replayssm_write_pos,
                replayssm_force_flush=replayssm_force_flush,
            )

    def _replay_metadata(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
        seq_lens_cpu: Optional[torch.Tensor],
        num_padding: Optional[int] = None,
        in_capture: bool = False,
        mamba_track_indices: Optional[torch.Tensor] = None,
    ):
        if num_padding is None:
            if seq_lens_cpu is None:
                num_padding = 0
            else:
                num_padding = torch.count_nonzero(
                    seq_lens_cpu == self.get_cuda_graph_seq_len_fill_value()
                )
        # Make sure forward metadata is correctly handled for padding reqs
        req_pool_indices[bs - num_padding :] = 0
        mamba_indices = self.req_to_token_pool.get_mamba_indices(req_pool_indices)
        # Translate using the LIVE v2p table BEFORE the padding sentinel below;
        # captured Mamba kernels read state_indices_list as PHYSICAL ids.
        mamba_indices = self._translate_mamba_indices(mamba_indices)
        mamba_indices[bs - num_padding :] = -1
        self.state_indices_list[bs - 1][: len(mamba_indices)].copy_(mamba_indices)
        # Refresh the static track-dest buffer in-place (translated); the captured
        # track-save reads it, leaving the handed-in InputBuffer slot read-only.
        track_buf = None
        if mamba_track_indices is not None:
            track_buf = self.mamba_track_indices_buf
            track_buf[: len(mamba_track_indices)].copy_(
                self._translate_mamba_indices(mamba_track_indices)
            )
        # Refresh the static write cursor in-place (mirrors the eager
        # snapshot-then-advance). Skip the advance during capture: dummy slots
        # would corrupt real ring positions.
        replayssm_write_pos = None
        replayssm_force_flush = None
        if self.replayssm_write_pos_list is not None:
            mamba_pool = self.req_to_token_pool.mamba_pool
            write_pos_buf = mamba_pool.replayssm_write_pos
            static_wp = self.replayssm_write_pos_list[bs - 1]
            static_ff = self.replayssm_force_flush_list[bs - 1]
            # Hand the full captured per-bs buffers to the kernel; it indexes per row.
            replayssm_write_pos = static_wp
            replayssm_force_flush = static_ff
            if write_pos_buf is not None:
                # this replay's per-row physical slots (padded rows == -1)
                slots = mamba_indices.to(torch.long)
                safe_slots = slots.clamp(min=0)
                # Snapshot this step's cursor into the captured buffer in-place
                # (copy_, never reassign the object).
                static_wp[: len(mamba_indices)].copy_(write_pos_buf[safe_slots])
                # Refresh the force-flush buffer in-place from this step's seq_lens
                # (same condition as the radix track). Zeroed during capture.
                force_flush_dev = None
                # KDA: no radix coordination -> leave zeroed so the advance is a pure wrap.
                is_kda = getattr(mamba_pool, "replayssm_is_kda", False)
                if (
                    not is_kda
                    and forward_mode.is_decode_or_idle()
                    and seq_lens_cpu is not None
                ):
                    ff_mask = self._replayssm_track_flush_mask(seq_lens_cpu, bs)
                    force_flush_dev = ff_mask.to(device=self.device, dtype=torch.int32)
                    static_ff.copy_(force_flush_dev)
                else:
                    static_ff.zero_()
                if not in_capture:
                    L = mamba_pool.linear_replayssm_cache_len
                    # Advance only valid (non-padded) slots; a forced flush empties
                    # the ring -> next write_pos 0, like the natural L-1 wrap.
                    valid_mask = slots >= 0
                    valid_slots = slots[valid_mask]
                    if valid_slots.numel() > 0:
                        cur_pos = write_pos_buf[safe_slots]
                        flushed = cur_pos == (L - 1)
                        if force_flush_dev is not None:
                            flushed = flushed | (force_flush_dev != 0)
                        next_pos = torch.where(
                            flushed,
                            torch.zeros_like(cur_pos),
                            (cur_pos + 1) % L,
                        )
                        # Dedup; rows sharing a slot share write_pos+flush.
                        uniq_slots, inv = torch.unique(valid_slots, return_inverse=True)
                        next_for_valid = next_pos[valid_mask]
                        new_vals = torch.empty(
                            uniq_slots.shape[0],
                            dtype=write_pos_buf.dtype,
                            device=write_pos_buf.device,
                        )
                        new_vals[inv] = next_for_valid.to(write_pos_buf.dtype)
                        write_pos_buf[uniq_slots] = new_vals
        if forward_mode.is_decode_or_idle():
            if num_padding == 0:
                self.query_start_loc_list[bs - 1].copy_(
                    self.cached_cuda_graph_decode_query_start_loc[: bs + 1]
                )
            else:
                self.query_start_loc_list[bs - 1][: bs - num_padding].copy_(
                    self.cached_cuda_graph_decode_query_start_loc[: bs - num_padding]
                )
                self.query_start_loc_list[bs - 1][bs - num_padding :].fill_(
                    bs - num_padding
                )
        elif forward_mode.is_target_verify():
            if num_padding == 0:
                self.query_start_loc_list[bs - 1].copy_(
                    self.cached_cuda_graph_verify_query_start_loc[: bs + 1]
                )
            else:
                self.query_start_loc_list[bs - 1][: bs - num_padding].copy_(
                    self.cached_cuda_graph_verify_query_start_loc[: bs - num_padding]
                )
                self.query_start_loc_list[bs - 1][bs - num_padding :].fill_(
                    (bs - num_padding) * spec_info.draft_token_num
                )
        else:
            raise ValueError(f"Invalid forward mode: {forward_mode=}")

        # Refresh the static intermediate-index buffer in-place (the captured
        # verify kernels read it by pointer): real rows get translated band
        # slots (identity for the dense pool), padded rows route to the sink.
        intermediate_state_indices = None
        if forward_mode.is_target_verify():
            intermediate_state_indices = self._refresh_intermediate_state_indices(
                bs - int(num_padding)
            )

        if forward_mode.is_target_verify() and self.topk > 1:
            if (
                spec_info is not None
                and getattr(spec_info, "retrieve_next_token", None) is not None
            ):
                bs_without_pad = spec_info.retrieve_next_token.shape[0]
                self.retrieve_next_token_list[bs - 1][:bs_without_pad].copy_(
                    spec_info.retrieve_next_token
                )
                self.retrieve_next_sibling_list[bs - 1][:bs_without_pad].copy_(
                    spec_info.retrieve_next_sibling
                )
            return ForwardMetadata(
                query_start_loc=self.query_start_loc_list[bs - 1],
                mamba_cache_indices=self.state_indices_list[bs - 1],
                mamba_track_indices=track_buf,
                retrieve_next_token=self.retrieve_next_token_list[bs - 1],
                retrieve_next_sibling=self.retrieve_next_sibling_list[bs - 1],
                retrieve_parent_token=self.retrieve_parent_token_list[bs - 1],
                intermediate_state_indices=intermediate_state_indices,
                replayssm_write_pos=replayssm_write_pos,
                replayssm_force_flush=replayssm_force_flush,
            )
        else:
            return ForwardMetadata(
                query_start_loc=self.query_start_loc_list[bs - 1],
                mamba_cache_indices=self.state_indices_list[bs - 1],
                mamba_track_indices=track_buf,
                intermediate_state_indices=intermediate_state_indices,
                replayssm_write_pos=replayssm_write_pos,
                replayssm_force_flush=replayssm_force_flush,
            )

    def get_cuda_graph_seq_len_fill_value(self):
        return 1  # Mamba attn does not use seq lens to index kv cache

    def get_cpu_graph_seq_len_fill_value(self):
        return 1

    def _track_mamba_state_decode(
        self,
        forward_batch: ForwardBatch,
        conv_states: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
    ):
        """Copy decode conv/SSM states to track slots for prefix caching. Track
        dests come from the metadata (under cuda-graph: the static buffer), so the
        InputBuffer registry slot is never mutated."""
        if forward_batch.mamba_track_mask is not None:
            track_mamba_states_if_needed(
                conv_states,
                ssm_states,
                cache_indices,
                forward_batch.mamba_track_mask,
                self.forward_metadata.mamba_track_indices,
                forward_batch.batch_size,
                check_freed_slots=self.enable_unified_memory,
            )

    def _track_mamba_state_extend(
        self,
        forward_batch: ForwardBatch,
        h: torch.Tensor,
        ssm_states: torch.Tensor,
        forward_metadata: ForwardMetadata,
    ):
        """Copy extend SSM state at the last chunk boundary to track slots (source
        depends on chunk alignment; see `_init_track_ssm_indices`)."""
        if forward_metadata.has_mamba_track_mask:
            h = h.squeeze(0)

            if forward_metadata.track_ssm_h_src.numel() > 0:
                ssm_states[forward_metadata.track_ssm_h_dst] = h[
                    forward_metadata.track_ssm_h_src
                ].to(ssm_states.dtype, copy=False)
            if forward_metadata.track_ssm_final_src.numel() > 0:
                ssm_states[forward_metadata.track_ssm_final_dst] = ssm_states[
                    forward_metadata.track_ssm_final_src
                ]


class Mamba2AttnBackend(MambaAttnBackendBase):
    """Attention backend wrapper for Mamba2Mixer kernels."""

    needs_cpu_seq_lens: bool = False

    def __init__(self, model_runner: ModelRunner):
        super().__init__(model_runner)
        config = model_runner.mamba2_config
        assert config is not None
        self.mamba_chunk_size = config.mamba_chunk_size
        self.conv_states_shape = (
            model_runner.req_to_token_pool.mamba_pool.mamba_cache.conv[0].shape
        )

        if model_runner.server_args.enable_mamba_extra_buffer():
            assert (
                self.conv_states_shape[-1] < self.mamba_chunk_size
            ), f"{self.conv_states_shape[-1]=} should be less than {self.mamba_chunk_size}"
            assert (
                model_runner.server_args.mamba_track_interval >= self.mamba_chunk_size
            ), f"mamba_track_interval ({model_runner.server_args.mamba_track_interval}) must be >= mamba_chunk_size ({self.mamba_chunk_size})"

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ):
        metadata = self._replay_metadata(
            forward_batch.batch_size,
            forward_batch.req_pool_indices,
            forward_batch.forward_mode,
            forward_batch.spec_info,
            forward_batch.seq_lens_cpu if not in_capture else None,
            num_padding=(
                0 if in_capture else getattr(forward_batch, "num_padding", None)
            ),
            in_capture=in_capture,
            mamba_track_indices=getattr(forward_batch, "mamba_track_indices", None),
        )
        spec_info = forward_batch.spec_info
        draft_token_num = spec_info.draft_token_num if spec_info is not None else 1
        self.forward_metadata = Mamba2Metadata.prepare_decode(
            metadata,
            forward_batch.seq_lens,
            is_target_verify=forward_batch.forward_mode.is_target_verify(),
            draft_token_num=draft_token_num,
        )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        self._execute_deferred_mamba_cow_and_clear(forward_batch)
        metadata = self._forward_metadata(forward_batch)
        self.forward_metadata = Mamba2Metadata.prepare_mixed(
            metadata,
            self.mamba_chunk_size,
            forward_batch,
        )

    def forward(
        self,
        mixer: MambaMixer2,
        hidden_states: torch.Tensor,
        output: Optional[torch.Tensor],
        layer_id: int,
        forward_batch: ForwardBatch,
        mup_vector: Optional[torch.Tensor] = None,
        use_triton_causal_conv: bool = False,
    ):
        assert isinstance(self.forward_metadata, Mamba2Metadata)
        # Two independent reasons need the stride-aware Triton causal-conv, either
        # of which forces it here (a model may also force it per-call):
        #  1. Page-major stores the conv state strided; the CUDA causal_conv1d
        #     assumes contiguous and garbles strided reads.
        #  2. Speculative decoding's target-verify path needs the per-draft-token
        #     intermediate conv windows to roll back to the last accepted token;
        #     only the Triton kernel emits them (see mamba.py `is_target_verify`).
        #     This covers every Mamba2-mixer target model without a per-model force.
        server_args = get_global_server_args()
        use_triton_causal_conv = (
            use_triton_causal_conv
            or server_args.enable_page_major_kv_layout
            or server_args.speculative_algorithm is not None
        )
        layer_cache = self.req_to_token_pool.mamba2_layer_cache(layer_id)
        mixer_out, intermediate_states = mixer.forward(
            hidden_states=hidden_states,
            output=output,
            layer_cache=layer_cache,
            metadata=self.forward_metadata,
            forward_batch=forward_batch,
            mup_vector=mup_vector,
            use_triton_causal_conv=use_triton_causal_conv,
        )

        if forward_batch.mamba_track_mask is not None:
            if (
                intermediate_states is not None
                and forward_batch.mamba_track_mask is not None
                and forward_batch.mamba_track_mask.any()
            ):
                self._track_mamba_state_extend(
                    forward_batch,
                    intermediate_states,
                    layer_cache.temporal,
                    self.forward_metadata,
                )

            if self.forward_metadata.num_decodes > 0:
                num_decodes = self.forward_metadata.num_decodes
                track_mamba_states_if_needed(
                    layer_cache.conv[0],
                    layer_cache.temporal,
                    self.forward_metadata.mamba_cache_indices[-num_decodes:],
                    forward_batch.mamba_track_mask[-num_decodes:],
                    self.forward_metadata.mamba_track_indices[-num_decodes:],
                    num_decodes,
                    check_freed_slots=self.enable_unified_memory,
                )

        return mixer_out

    def forward_decode(self, *args, **kwargs):
        raise NotImplementedError(
            "Mamba2AttnBackend's forward is called directly instead of through HybridLinearAttnBackend, as it supports mixed prefill and decode"
        )

    def forward_extend(self, *args, **kwargs):
        raise NotImplementedError(
            "Mamba2AttnBackend's forward is called directly instead of through HybridLinearAttnBackend, as it supports mixed prefill and decode"
        )


class HybridLinearAttnBackend(AttentionBackend):
    """Manages a full and linear attention backend"""

    def __init__(
        self,
        full_attn_backend: AttentionBackend,
        linear_attn_backend: MambaAttnBackendBase,
        full_attn_layers: list[int],
    ):
        self.full_attn_layers = full_attn_layers
        self.full_attn_backend = full_attn_backend
        self.linear_attn_backend = linear_attn_backend
        self.attn_backend_list = [full_attn_backend, linear_attn_backend]
        self.token_to_kv_pool = full_attn_backend.token_to_kv_pool
        self.req_to_token_pool = full_attn_backend.req_to_token_pool
        self.max_context_len = getattr(full_attn_backend, "max_context_len", None)
        self.needs_cpu_seq_lens = (
            full_attn_backend.needs_cpu_seq_lens
            or linear_attn_backend.needs_cpu_seq_lens
        )

    def _is_full_attn(
        self, layer: Optional[RadixAttention], layer_id: Optional[int] = None
    ) -> bool:
        if layer is not None:
            layer_id = layer.layer_id
        assert layer_id is not None, "either layer or layer_id must be provided"
        return layer_id in self.full_attn_layers

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ):
        for attn_backend in self.attn_backend_list:
            attn_backend.init_forward_metadata_out_graph(
                forward_batch, in_capture=in_capture
            )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        if forward_batch.forward_mode.is_draft_extend_v2():
            # DRAFT_EXTEND_V2 runs only full-attn layers in the draft model; skip
            # linear/mamba metadata (it requires query_start_loc).
            self.full_attn_backend.init_forward_metadata(forward_batch)
            return
        for attn_backend in self.attn_backend_list:
            attn_backend.init_forward_metadata(forward_batch)

    def init_mha_chunk_metadata(
        self, forward_batch: ForwardBatch, disable_flashinfer_ragged: bool = False
    ):
        # Hybrid MLA models resolve get_attn_backend() to this wrapper; delegate
        # so the full-attn backend plans its chunked-prefill metadata.
        init = getattr(self.full_attn_backend, "init_mha_chunk_metadata", None)
        if init is not None:
            init(forward_batch, disable_flashinfer_ragged)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        for attn_backend in self.attn_backend_list:
            attn_backend.init_cuda_graph_state(max_bs, max_num_tokens)

    def init_cpu_graph_state(self, max_bs: int, max_num_tokens: int):
        for attn_backend in self.attn_backend_list:
            attn_backend.init_cpu_graph_state(max_bs, max_num_tokens)

    def init_forward_metadata_capture_cpu_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
    ):
        for attn_backend in self.attn_backend_list:
            attn_backend.init_forward_metadata_capture_cpu_graph(
                bs,
                num_tokens,
                req_pool_indices,
                seq_lens,
                encoder_lens,
                forward_mode,
                spec_info,
            )

    def get_cuda_graph_seq_len_fill_value(self):
        return self.full_attn_backend.get_cuda_graph_seq_len_fill_value()

    def get_cpu_graph_seq_len_fill_value(self):
        return self.full_attn_backend.get_cpu_graph_seq_len_fill_value()

    def get_verify_buffers_to_fill_after_draft(self):
        # Forward to the full-attn sub-backend (the same non-propagation class
        # as translate_metadata_in_graph): the tree/chain mask lives in the
        # FULL-attention verify kernels. Without this forward the base class
        # returns [None, None], so the EAGLE draft loop falls back to
        # allocating a worst-case bs*max_context_len mask every step whenever
        # batch.seq_lens_sum is unavailable (the sync-free spec-overlap path),
        # and the cuda-graph replay copy then overflows Triton's
        # cuda_graph_custom_mask (sized max_num_tokens*max_context_len) by
        # bs*draft_token_num**2 — a boot-time RuntimeError on hybrid models
        # with EAGLE + cuda graphs (Qwen3.5 NEXTN).
        return self.full_attn_backend.get_verify_buffers_to_fill_after_draft()

    def update_verify_buffers_to_fill_after_draft(
        self, spec_info: SpecInput, cuda_graph_bs: Optional[int]
    ):
        self.full_attn_backend.update_verify_buffers_to_fill_after_draft(
            spec_info, cuda_graph_bs
        )

    def forward_decode(
        self,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        q: Optional[torch.Tensor] = None,  # For full attention
        k: Optional[torch.Tensor] = None,  # For full attention
        v: Optional[torch.Tensor] = None,  # For full attention
        mixed_qkv: Optional[torch.Tensor] = None,  # For linear attention
        a: Optional[torch.Tensor] = None,  # For GDN linear attention
        b: Optional[torch.Tensor] = None,  # For GDN linear attention
        **kwargs,
    ):
        if self._is_full_attn(layer, kwargs.get("layer_id")):
            return self.full_attn_backend.forward_decode(
                q, k, v, layer, forward_batch, save_kv_cache, **kwargs
            )
        return self.linear_attn_backend.forward_decode(
            q=q,
            k=k,
            v=v,
            layer=layer,
            forward_batch=forward_batch,
            save_kv_cache=save_kv_cache,
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            **kwargs,
        )

    def forward_extend(
        self,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        q: Optional[torch.Tensor] = None,  # For full attention
        k: Optional[torch.Tensor] = None,  # For full attention
        v: Optional[torch.Tensor] = None,  # For full attention
        mixed_qkv: Optional[torch.Tensor] = None,  # For linear attention
        a: Optional[torch.Tensor] = None,  # For GDN linear attention
        b: Optional[torch.Tensor] = None,  # For GDN linear attention
        **kwargs,
    ):
        if self._is_full_attn(layer, kwargs.get("layer_id")):
            return self.full_attn_backend.forward_extend(
                q, k, v, layer, forward_batch, save_kv_cache, **kwargs
            )
        return self.linear_attn_backend.forward_extend(
            q=q,
            k=k,
            v=v,
            layer=layer,
            forward_batch=forward_batch,
            save_kv_cache=save_kv_cache,
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            **kwargs,
        )

    def forward(
        self,
        q: Optional[torch.Tensor] = None,  # For full attention
        k: Optional[torch.Tensor] = None,  # For full attention
        v: Optional[torch.Tensor] = None,  # For full attention
        layer: RadixAttention = None,
        forward_batch: ForwardBatch = None,
        save_kv_cache: bool = True,
        mixed_qkv: Optional[torch.Tensor] = None,  # For linear attention
        a: Optional[torch.Tensor] = None,  # For linear attention
        b: Optional[torch.Tensor] = None,  # For linear attention
        **kwargs,
    ):
        is_linear_attn = not self._is_full_attn(layer, kwargs.get("layer_id"))

        if forward_batch.forward_mode.is_idle():
            if is_linear_attn:
                return mixed_qkv.new_empty(
                    mixed_qkv.shape[0], layer.num_v_heads, layer.head_v_dim
                )
            return q.new_empty(q.shape[0], layer.tp_q_head_num * layer.v_head_dim)
        elif forward_batch.forward_mode.is_decode():
            return self.forward_decode(
                layer,
                forward_batch,
                save_kv_cache,
                q,
                k,
                v,
                mixed_qkv,
                a,
                b,
                **kwargs,
            )
        else:
            return self.forward_extend(
                layer,
                forward_batch,
                save_kv_cache,
                q,
                k,
                v,
                mixed_qkv,
                a,
                b,
                **kwargs,
            )

    def update_mamba_state_after_mtp_verify(
        self,
        last_correct_step_indices: torch.Tensor,
        mamba_track_indices: Optional[torch.Tensor],
        mamba_steps_to_track: Optional[torch.Tensor],
        model,
    ):
        """Update mamba states after MTP verify via a fused gather-scatter kernel."""
        request_number = last_correct_step_indices.shape[0]

        state_indices_tensor = (
            self.linear_attn_backend.forward_metadata.mamba_cache_indices[
                :request_number
            ]
        )

        # `mamba_track_indices` arrives VIRTUAL: every spec commit caller
        # (spec_utils + dflash_worker_v2) passes the ScheduleBatch field, which
        # set_mamba_track_indices_from_reqs rebuilds from allocator-minted
        # ping-pong track ids. The scatters below address PHYSICAL state views,
        # so translate at this seam — the forward path's equivalent translate
        # (init_forward_metadata) only rebinds the ForwardBatch copy and never
        # reaches this call. Identity for the non-unified pool. Untranslated
        # ids are in-range physical slots, so the miss is SILENT: track states
        # land in the wrong slot once eviction/compaction makes v2p diverge
        # from identity — the radix×spec×mamba accuracy collapse.
        if mamba_track_indices is not None:
            mamba_track_indices = self.linear_attn_backend._translate_mamba_indices(
                mamba_track_indices
            )

        mamba_caches = (
            self.linear_attn_backend.req_to_token_pool.get_speculative_mamba2_params_all_layers()
        )

        conv_states = mamba_caches.conv[0]
        ssm_states = mamba_caches.temporal
        intermediate_state_cache = mamba_caches.intermediate_ssm
        intermediate_conv_window_cache = mamba_caches.intermediate_conv_window[0]

        # Unified pool: the scatter kernels hardcode src row == verify row
        # (src_idx = pid_req), so hand them the BAND SLICE of the full-span
        # intermediate views — row r of the slice IS physical slot
        # band_start + r. Legal because this runs OUTSIDE the captured verify
        # graph (fresh base pointer per eager launch), unlike the in-graph
        # write kernels which use the translated-indices path instead.
        band = self.linear_attn_backend._spec_state_allocator
        if band is not None:
            start = band.band_start_page
            stop = start + band.num_placed_slots
            intermediate_state_cache = intermediate_state_cache[:, start:stop]
            intermediate_conv_window_cache = intermediate_conv_window_cache[
                :, start:stop
            ]

        if _DEBUG_SPEC_BAND and band is not None:
            # CHECK 3 (commit freshness): the commit scatter targets the
            # PHYSICAL mamba slots snapshotted at this batch's metadata build.
            # Under OVERLAP, the scheduler thread may free/compact mamba slots
            # for the NEXT batch before this commit is enqueued — free()'s
            # wait_stream barrier serializes the copy only with kernels
            # enqueued SO FAR, not with this later commit. A moved/freed
            # commit target = lost state update (the radix-reuse corruption
            # class). Also assert the pin window still holds.
            mamba = getattr(band, "low_peer", None)
            if mamba is not None and hasattr(mamba, "physical_to_virtual"):
                if not band._band_pinned:
                    msg = (
                        "[spec-band] COMMIT OUTSIDE PIN WINDOW: the band was "
                        "unpinned before the commit scatter was enqueued "
                        f"{band.allocator_state_str()}"
                    )
                    logger.error(msg)
                    raise AssertionError(msg)
                _commit_phys = state_indices_tensor.to(torch.int64)
                # Also validate the track-state commit lane (the radix cached-
                # state writes) — rows that will actually scatter (steps != -1).
                if mamba_track_indices is not None and mamba_steps_to_track is not None:
                    _track_phys = mamba_track_indices.to(torch.int64)[
                        mamba_steps_to_track != -1
                    ]
                    _commit_phys = torch.cat([_commit_phys, _track_phys])
                _real = _commit_phys[
                    (_commit_phys >= 0) & (_commit_phys < mamba.num_pages)
                ]
                if _real.numel() > 0:
                    _stale = mamba.physical_to_virtual[_real] == -1
                    if bool(_stale.any().item()):
                        msg = (
                            "[spec-band] STALE COMMIT TARGET: the verify commit "
                            "scatter writes to physical mamba slots "
                            f"{_real[_stale][:8].tolist()} that are FREE now "
                            "(p2v==-1) — the slot was freed/compaction-moved "
                            "between this batch's metadata build and its "
                            "commit (overlap window). The accepted state is "
                            "lost / lands in reusable memory. "
                            f"mamba={mamba.allocator_state_str()}"
                        )
                        logger.error(msg)
                        raise AssertionError(msg)

        fused_mamba_state_scatter_with_mask(
            ssm_states,
            intermediate_state_cache,
            state_indices_tensor,
            last_correct_step_indices,
        )
        # conv intermediate uses the deduplicated sliding-window layout, so it
        # needs the strided-read scatter variant.
        fused_conv_window_scatter_with_mask(
            conv_states,
            intermediate_conv_window_cache,
            state_indices_tensor,
            last_correct_step_indices,
        )

        # Track indices for prefix cache
        if mamba_track_indices is not None:
            assert mamba_steps_to_track is not None
            fused_mamba_state_scatter_with_mask(
                ssm_states,
                intermediate_state_cache,
                mamba_track_indices,
                mamba_steps_to_track,
            )
            fused_conv_window_scatter_with_mask(
                conv_states,
                intermediate_conv_window_cache,
                mamba_track_indices,
                mamba_steps_to_track,
            )
