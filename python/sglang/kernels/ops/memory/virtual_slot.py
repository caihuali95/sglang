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
    tl.store(out_ptr + offs, tl.maximum(res, 0), mask=mask)


TRANSLATE_BLOCK = 256


def translate_v2p(
    virt_tokens: torch.Tensor,
    v2p: torch.Tensor,
    *,
    page_size: int,
    page_stride: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Translate virtual TOKEN ids through `v2p`; see `translate_v2p_kernel`.

    ``out`` may alias ``virt_tokens``. Returns ``out`` when given, else a fresh
    int64 tensor.
    """
    # The kernel indexes both sides flatly, so a strided view would read/write
    # the wrong elements (silently). `out` is caller-owned — fail loud rather
    # than write somewhere else; the input may be copied.
    assert out is None or out.is_contiguous(), "translate_v2p: out= must be contiguous"
    if not virt_tokens.is_contiguous():
        virt_tokens = virt_tokens.contiguous()
    if out is None:
        out = torch.empty(
            virt_tokens.shape, dtype=torch.int64, device=virt_tokens.device
        )
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
