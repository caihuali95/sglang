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
"""Virtual<->physical slot Triton kernels for the unified memory pool."""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

# Fused take-physical-pages + bind for the alloc fast path. Invoked ONLY when
# `_hole_count == 0`; otherwise the slow path drains holes first (Invariant B,
# greedy hole reuse). Caller advances `watermark_physical` and checks overflow
# BEFORE launch, passing the PRE-extension watermark. Cuda-graph safe (no
# `.item()`, no tensor branching); runs on the scheduler thread.


@triton.jit
def alloc_bind_inplace_kernel(
    v_pages_ptr,  # in: [N] int64 — virtual page ids
    v2p_ptr,  # in/out: int64 — virtual_to_physical table
    p2v_ptr,  # in/out: int64 — physical_to_virtual table
    out_phys_ptr,  # out: [N] int64 — physical page ids
    N,  # runtime: number of pages to allocate
    start_phys,  # runtime: lowest physical page id in the new range
    BLOCK: tl.constexpr,
):
    """Fused: ascending arange + out_phys/v2p/p2v scatter.

    Caller pre-adjusts `start_phys` per direction so the range is always
    ascending (grow-up: start_wm; grow-down: start_wm - N + 1), making the
    v->p mapping byte-identical to the `torch.arange` slow path.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    phys = (start_phys + offs).to(tl.int64)
    v = tl.load(v_pages_ptr + offs, mask=mask, other=0).to(tl.int64)

    # Masked stores skip out-of-range lanes, and `other=0` keeps us off the
    # v2p[0]/p2v[0] padding-sink slot.
    tl.store(out_phys_ptr + offs, phys, mask=mask)
    tl.store(v2p_ptr + v, phys, mask=mask)
    tl.store(p2v_ptr + phys, v, mask=mask)


ALLOC_BIND_BLOCK = 128


def alloc_bind_inplace(
    v_pages: torch.Tensor,
    v2p: torch.Tensor,
    p2v: torch.Tensor,
    start_phys: int,
) -> torch.Tensor:
    """Allocate N ascending physical pages from `start_phys` and bind to `v_pages`.

    Caller must advance `watermark_physical` by N and verify overflow BEFORE
    calling; this launcher does neither.
    """
    N = int(v_pages.numel())
    if N == 0:
        return torch.empty(0, dtype=torch.int64, device=v_pages.device)
    if not v_pages.is_cuda:
        # Pure-torch CPU reference for the CUDA-only kernel.
        phys_pages = torch.arange(
            start_phys, start_phys + N, dtype=torch.int64, device=v_pages.device
        )
        v = v_pages.to(torch.int64)
        v2p[v] = phys_pages
        p2v[phys_pages] = v
        return phys_pages
    phys_pages = torch.empty(N, dtype=torch.int64, device=v_pages.device)
    grid = (triton.cdiv(N, ALLOC_BIND_BLOCK),)
    alloc_bind_inplace_kernel[grid](
        v_pages,
        v2p,
        p2v,
        phys_pages,
        N,
        start_phys,
        BLOCK=ALLOC_BIND_BLOCK,
    )
    return phys_pages


# Virtual -> physical TOKEN-id translate, run per decode step (read path and
# cuda-graph write-loc refill) on the scheduler thread. One kernel replaces the
# six-op tensor chain (`//`, `%`, index_select, `mul_`, `add_`, `clamp_`) the
# page math would otherwise cost, and — because each program reads and writes
# the SAME element index — it is safe when `out` aliases `virt_tokens`, which
# the canonical in-place caller does.


@triton.jit
def translate_v2p_kernel(
    virt_ptr,  # in: [N] virtual TOKEN ids
    v2p_ptr,  # in: int64 — virtual_to_physical PAGE table
    out_ptr,  # out: [N] int64 — physical (or dense) TOKEN ids
    N,  # runtime: number of ids
    page_stride,  # runtime: page_size * kernel_page_multiplier
    PAGE_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """``out = max(v2p[t // ps] * page_stride + t % ps, 0)``.

    ``page_stride`` carries the dense-view scale, so the same kernel serves the
    physical translate (multiplier 1) and the dense MLA translate.

    Both id classes that carry no location land on physical 0 — the reserved
    padding sink (the `min_slot_index` invariant keeps bytes [0, entry_max)
    free of real data), which a captured graph may legally read and write:
      - a TOMBSTONED page (v2p == -1) makes the product negative -> the clamp;
      - a PADDING id (t < 0) is routed explicitly, both to keep the gather in
        bounds and to match the pre-fusion table-sentinel behaviour.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    t = tl.load(virt_ptr + offs, mask=mask, other=0).to(tl.int64)
    is_padding = t < 0
    if PAGE_SIZE == 1:
        page = t
        offset = 0
    else:
        page = t // PAGE_SIZE
        offset = t % PAGE_SIZE
    page = tl.where(is_padding, 0, page)  # keep the gather in bounds
    phys = tl.load(v2p_ptr + page, mask=mask, other=0).to(tl.int64)
    res = tl.where(is_padding, 0, phys * page_stride + offset)
    # Narrow to the destination width last (int32 for the SWA read path, whose
    # kernels take int32 ids); page ids are bounded by the pool, so the
    # post-clamp value always fits.
    res = tl.maximum(res, 0).to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + offs, res, mask=mask)


TRANSLATE_BLOCK = 256


def translate_v2p(
    virt_tokens: torch.Tensor,
    v2p: torch.Tensor,
    *,
    page_size: int,
    page_stride: int,
    out: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.int64,
) -> torch.Tensor:
    """Translate virtual TOKEN ids through `v2p`; see `translate_v2p_kernel`.

    ``out`` may alias ``virt_tokens``. Returns ``out`` when given, else a fresh
    tensor of ``out_dtype`` (int64 for physical/dense ids, int32 for the SWA
    read path).
    """
    # The kernel indexes both sides flatly, so a strided view would read/write
    # the wrong elements (silently). `out` is caller-owned — fail loud rather
    # than write somewhere else; the input may be copied.
    assert out is None or out.is_contiguous(), "translate_v2p: out= must be contiguous"
    if not virt_tokens.is_contiguous():
        virt_tokens = virt_tokens.contiguous()
    if out is None:
        out = torch.empty(virt_tokens.shape, dtype=out_dtype, device=virt_tokens.device)
    N = int(virt_tokens.numel())
    if N == 0:
        return out
    if not virt_tokens.is_cuda:
        # Pure-torch CPU reference for the CUDA-only kernel.
        is_padding = virt_tokens < 0
        pages = torch.where(is_padding, 0, virt_tokens // page_size)
        offsets = virt_tokens % page_size
        res = torch.where(is_padding, 0, v2p[pages] * page_stride + offsets)
        out.copy_(torch.clamp_min(res, 0))
        return out
    grid = (triton.cdiv(N, TRANSLATE_BLOCK),)
    translate_v2p_kernel[grid](
        virt_tokens,
        v2p,
        out,
        N,
        page_stride,
        PAGE_SIZE=page_size,
        BLOCK=TRANSLATE_BLOCK,
    )
    return out


# Fused decode-step allocation for the unified SWA/tri composite: the upstream
# `alloc_decode_kernel` token-slot computation PLUS the dual virtual->physical
# binding (full end + swa side bind the SAME virtual pages into their own
# physical spaces). One launch replaces alloc_decode_kernel + two
# alloc_bind_inplace launches + two free-list clone snapshots per step.
#
# The physical side of each bind comes from a RESERVATION the caller planned in
# Python (`_reserve_phys_pages`): `phys(i) = holes[i] for i < n_drained, else
# start_phys + (i - n_drained)` — one formula covering the watermark/span fast
# path (no drain), the hole-recycling slow path, and the mixed case. Holes
# drain from the slice FRONT on every band type (matching `take_physical`).


# `free_page_ptr` and the two hole slices are all RE-SLICED after every
# allocation, so their `data_ptr()` flips between 16-byte-aligned and unaligned
# across calls. Triton bakes pointer alignment into the cache key, so each flip
# compiles another kernel variant and pays its JIT (~100ms) on the scheduler
# thread mid-serving — see the same rationale for `alloc_extend_kernel` in
# `ops/memory/allocator.py`. Skipping the specialization costs nothing here:
# every load through these three pointers is SCALAR, so the alignment hint
# enables no vectorization. `translate_v2p_kernel` is deliberately different:
# its loads ARE block-wide contiguous, so alignment genuinely matters there,
# and its pointers are [:n] slices off fixed buffers whose base never moves.
@triton.jit(
    do_not_specialize=[
        "free_page_ptr",
        "holes_a_ptr",
        "holes_b_ptr",
        "num_new_pages",
        "nd_a",
        "start_a",
        "nd_b",
        "start_b",
    ]
)
def alloc_decode_dual_bind_kernel(
    seq_lens_ptr,
    last_loc_ptr,
    free_page_ptr,  # in: virtual page free-list (front num_new_pages consumed)
    out_indices,  # out: [bs] virtual TOKEN ids
    v2p_a_ptr,  # in/out: side-A virtual_to_physical
    p2v_a_ptr,  # in/out: side-A physical_to_virtual
    holes_a_ptr,  # in: side-A drained hole slice (unread when nd_a == 0)
    v2p_b_ptr,  # in/out: side-B virtual_to_physical
    p2v_b_ptr,  # in/out: side-B physical_to_virtual
    holes_b_ptr,  # in: side-B drained hole slice (unread when nd_b == 0)
    num_new_pages,  # runtime: N pages consumed this step
    nd_a,  # runtime: side-A drained count
    start_a,  # runtime: side-A first extended physical page
    nd_b,  # runtime: side-B drained count
    start_b,  # runtime: side-B first extended physical page
    bs_upper: tl.constexpr,
    page_size: tl.constexpr,
):
    pid = tl.program_id(0)

    # --- token slot: byte-identical to `alloc_decode_kernel` ---
    load_offset = tl.arange(0, bs_upper)
    seq_lens = tl.load(seq_lens_ptr + load_offset, mask=load_offset <= pid)
    pre_lens = tl.where(load_offset <= pid, seq_lens - 1, seq_lens)

    seq_len = tl.load(seq_lens_ptr + pid)
    pre_len = seq_len - 1

    num_pages_after = (seq_lens + page_size - 1) // page_size
    num_pages_before = (pre_lens + page_size - 1) // page_size
    num_new_per_req = num_pages_after - num_pages_before

    num_page_start_loc_self = (seq_len + page_size - 1) // page_size - (
        pre_len + page_size - 1
    ) // page_size
    sum_num_new_pages = tl.sum(num_new_per_req)
    new_page_start_loc = sum_num_new_pages - num_page_start_loc_self

    if num_page_start_loc_self == 0:
        last_loc = tl.load(last_loc_ptr + pid)
        tl.store(out_indices + pid, last_loc + 1)
    else:
        page = tl.load(free_page_ptr + new_page_start_loc)
        tl.store(out_indices + pid, page * page_size)

    # --- dual bind: program pid binds new page #pid in BOTH spaces ---
    if pid < num_new_pages:
        v = tl.load(free_page_ptr + pid).to(tl.int64)
        if pid < nd_a:
            pa = tl.load(holes_a_ptr + pid).to(tl.int64)
        else:
            pa = (start_a + (pid - nd_a)).to(tl.int64)
        tl.store(v2p_a_ptr + v, pa)
        tl.store(p2v_a_ptr + pa, v)
        if pid < nd_b:
            pb = tl.load(holes_b_ptr + pid).to(tl.int64)
        else:
            pb = (start_b + (pid - nd_b)).to(tl.int64)
        tl.store(v2p_b_ptr + v, pb)
        tl.store(p2v_b_ptr + pb, v)


def _reserved_phys_ids(
    n: int, holes, n_drained: int, start_phys: int, device
) -> torch.Tensor:
    """CPU reference for the per-side phys formula (page ordinals 0..n-1)."""
    out = torch.empty(n, dtype=torch.int64, device=device)
    if n_drained > 0:
        out[:n_drained] = holes[:n_drained]
    if n > n_drained:
        out[n_drained:] = start_phys + torch.arange(
            n - n_drained, dtype=torch.int64, device=device
        )
    return out


def alloc_decode_dual_bind(
    *,
    seq_lens: torch.Tensor,
    last_loc: torch.Tensor,
    free_virtual_pages: torch.Tensor,
    num_new_pages: int,
    page_size: int,
    v2p_a: torch.Tensor,
    p2v_a: torch.Tensor,
    holes_a,  # Optional[torch.Tensor]
    nd_a: int,
    start_a: int,
    v2p_b: torch.Tensor,
    p2v_b: torch.Tensor,
    holes_b,  # Optional[torch.Tensor]
    nd_b: int,
    start_b: int,
) -> torch.Tensor:
    """One-launch decode alloc + dual bind; see the kernel docstring.

    `holes_*` may be None when `nd_* == 0` (never read). The caller owns all
    Python-side state updates (watermarks, hole-slice rebinds, free-list slice).
    """
    bs = int(seq_lens.shape[0])
    device = seq_lens.device
    out_indices = torch.empty((bs,), dtype=torch.int64, device=device)
    N = num_new_pages
    # Program `pid` binds new page #pid, so the grid (one program per request)
    # must cover every page. Decode consumes at most one page per request
    # (`get_num_new_pages(decode=True)` counts wrapping requests), so this holds
    # by construction — assert rather than trust it: an uncovered page would
    # stay v2p=-1 and route reads/writes to the slot-0 sink SILENTLY.
    assert N <= bs, (
        f"alloc_decode_dual_bind: num_new_pages ({N}) exceeds the program grid "
        f"({bs}); pages [{bs}, {N}) would never be bound"
    )
    if not seq_lens.is_cuda:
        # Pure-torch CPU reference for the CUDA-only kernel.
        pre = seq_lens - 1
        pages_after = (seq_lens + page_size - 1) // page_size
        pages_before = (pre + page_size - 1) // page_size
        need = (pages_after - pages_before).to(torch.int64)
        ordinal = torch.cumsum(need, 0) - need  # exclusive prefix count
        out_indices.copy_(last_loc + 1)
        wraps = need > 0
        if int(wraps.sum()) > 0:
            pages = free_virtual_pages[ordinal[wraps]]
            out_indices[wraps] = pages * page_size
        if N > 0:
            v = free_virtual_pages[:N].to(torch.int64)
            pa = _reserved_phys_ids(N, holes_a, nd_a, start_a, device)
            v2p_a[v] = pa
            p2v_a[pa] = v
            pb = _reserved_phys_ids(N, holes_b, nd_b, start_b, device)
            v2p_b[v] = pb
            p2v_b[pb] = v
        return out_indices
    # Dummy device pointers for unread hole slices (guarded by nd_* == 0).
    holes_a_t = holes_a if holes_a is not None else out_indices
    holes_b_t = holes_b if holes_b is not None else out_indices
    alloc_decode_dual_bind_kernel[(bs,)](
        seq_lens,
        last_loc,
        free_virtual_pages,
        out_indices,
        v2p_a,
        p2v_a,
        holes_a_t,
        v2p_b,
        p2v_b,
        holes_b_t,
        N,
        nd_a,
        start_a,
        nd_b,
        start_b,
        bs_upper=triton.next_power_of_2(bs),
        page_size=page_size,
    )
    return out_indices
