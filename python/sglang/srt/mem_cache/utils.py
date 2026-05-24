# Copyright 2025 SGLang Team
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
"""Common utilities."""

import hashlib
from typing import Any, List, Optional, Tuple

import torch
import triton
import triton.language as tl

from sglang.jit_kernel.utils import is_arch_support_pdl
from sglang.srt.environ import envs


@triton.jit
def set_mla_kv_buffer_kernel(
    kv_buffer_ptr,
    cache_k_nope_ptr,
    cache_k_rope_ptr,
    loc_ptr,
    buffer_stride: tl.constexpr,
    nope_stride: tl.constexpr,
    rope_stride: tl.constexpr,
    nope_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    BLOCK: tl.constexpr,
    USE_GDC: tl.constexpr = False,
):
    pid_loc = tl.program_id(0)
    pid_blk = tl.program_id(1)

    base = pid_blk * BLOCK
    offs = base + tl.arange(0, BLOCK)
    total_dim = nope_dim + rope_dim
    mask = offs < total_dim

    if USE_GDC:
        tl.extra.cuda.gdc_wait()

    loc = tl.load(loc_ptr + pid_loc).to(tl.int64)
    dst_ptr = kv_buffer_ptr + loc * buffer_stride + offs

    # Three-way branch to handle boundary correctly while preserving fast path
    if base + BLOCK <= nope_dim:
        # Fast path: entire block is in nope region
        src = tl.load(
            cache_k_nope_ptr + pid_loc * nope_stride + offs,
            mask=mask,
        )
    elif base >= nope_dim:
        # Fast path: entire block is in rope region
        offs_rope = offs - nope_dim
        src = tl.load(
            cache_k_rope_ptr + pid_loc * rope_stride + offs_rope,
            mask=mask,
        )
    else:
        # Boundary case: block spans nope/rope boundary (e.g., FP8 with nope_dim=528)
        # Handle each offset individually to avoid negative indexing
        is_nope = offs < nope_dim
        is_rope = (offs >= nope_dim) & (offs < (nope_dim + rope_dim))

        src_nope = tl.load(
            cache_k_nope_ptr + pid_loc * nope_stride + offs,
            mask=mask & is_nope,
            other=0,
        )
        src_rope = tl.load(
            cache_k_rope_ptr + pid_loc * rope_stride + (offs - nope_dim),
            mask=mask & is_rope,
            other=0,
        )

        src = tl.where(is_nope, src_nope, src_rope)

    tl.store(dst_ptr, src, mask=mask)

    if USE_GDC:
        tl.extra.cuda.gdc_launch_dependents()


def set_mla_kv_buffer_triton(
    kv_buffer: torch.Tensor,
    loc: torch.Tensor,
    cache_k_nope: torch.Tensor,
    cache_k_rope: torch.Tensor,
):
    nope_dim = cache_k_nope.shape[-1]
    rope_dim = cache_k_rope.shape[-1]
    total_dim = nope_dim + rope_dim
    BLOCK = 128
    n_loc = loc.numel()
    grid = (n_loc, triton.cdiv(total_dim, BLOCK))

    pdl_kwargs = {"USE_GDC": True, "launch_pdl": True} if is_arch_support_pdl() else {}

    set_mla_kv_buffer_kernel[grid](
        kv_buffer,
        cache_k_nope,
        cache_k_rope,
        loc,
        kv_buffer.stride(0),
        cache_k_nope.stride(0),
        cache_k_rope.stride(0),
        nope_dim,
        rope_dim,
        BLOCK=BLOCK,
        **pdl_kwargs,
    )


# ---------------------------------------------------------------------------
# store_cache_4d — single-launch Triton write into the shared
# memory pool's 4-D envelope-strided K/V views.
#
# Replaces the per-call PyTorch advanced-indexing block in
# `SharedMHATokenToKVPool._do_set_kv_buffer`. At `PAGE_SIZE = 1` the kernel
# constexpr-folds to byte-identical addresses as the slot-major envelope
# view; at `PAGE_SIZE > 1` it uses the same `(page_id, tok_in_p)` split the
# attention read kernels (`_fwd_kernel_stage1`, etc.) use.
#
# See design-doc §19 for the §S361 motivation, falsification criteria, and
# eval matrix.
# ---------------------------------------------------------------------------


@triton.jit
def store_cache_4d_kernel(
    k_view_ptr,
    v_view_ptr,
    cache_k_ptr,
    cache_v_ptr,
    loc_ptr,
    # Strides in ELEMENTS (not bytes); wrapper passes view.stride(D)
    # directly. K and V may have different head_dim → different per-token
    # strides, so we carry both.
    stride_k_page,
    stride_k_tok,
    stride_v_page,
    stride_v_tok,
    stride_src_k_row,
    stride_src_v_row,
    # Constexpr shape parameters.
    K_ROW_DIM: tl.constexpr,  # head_num * head_dim
    V_ROW_DIM: tl.constexpr,  # head_num * v_head_dim
    PAGE_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Token-parallel Triton write into a 4-D envelope-strided K/V view.

    Grid: ``(N, ceil(max(K_ROW_DIM, V_ROW_DIM) / BLOCK), 2)`` where:
      - axis 0 → one program per token (loc[i])
      - axis 1 → blocks within one slot's K (or V) row
      - axis 2 → 0 = K, 1 = V (two-tensor write fused into one launch)

    For each token i, the kernel writes:
        page_id  = loc[i] // PAGE_SIZE
        tok_in_p = loc[i] %  PAGE_SIZE
        k_view[page_id, tok_in_p, :, :] = cache_k[i, :, :]
        v_view[page_id, tok_in_p, :, :] = cache_v[i, :, :]

    Cuda-graph safe: no Python branching on tensor values, no `.item()`,
    all shapes/strides known at launch time.
    """
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_kv = tl.program_id(2)

    # 1. Resolve destination slot in the 4-D view.
    loc = tl.load(loc_ptr + pid_n).to(tl.int64)
    if PAGE_SIZE == 1:
        page_id = loc
        tok_in_p = tl.zeros([], dtype=tl.int64)
    else:
        page_id = loc // PAGE_SIZE
        tok_in_p = loc % PAGE_SIZE

    # 2. Compute per-tensor source/dest pointers.
    base_off = pid_b * BLOCK + tl.arange(0, BLOCK)

    if pid_kv == 0:
        mask = base_off < K_ROW_DIM
        # The trailing (head_num, head_dim) axes of `k_view` are
        # contiguous: stride[-1]==1, stride[-2]==head_dim. So we can
        # treat them as a flat K_ROW_DIM dimension addressed by `base_off`.
        # The wrapper asserts this invariant.
        src_ptr = cache_k_ptr + pid_n * stride_src_k_row + base_off
        dst_ptr = (
            k_view_ptr
            + page_id * stride_k_page
            + tok_in_p * stride_k_tok
            + base_off
        )
    else:
        mask = base_off < V_ROW_DIM
        src_ptr = cache_v_ptr + pid_n * stride_src_v_row + base_off
        dst_ptr = (
            v_view_ptr
            + page_id * stride_v_page
            + tok_in_p * stride_v_tok
            + base_off
        )

    # 3. Copy.
    src = tl.load(src_ptr, mask=mask)
    tl.store(dst_ptr, src, mask=mask)


def store_cache_4d(
    k_view: torch.Tensor,
    v_view: torch.Tensor,
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    loc: torch.Tensor,
    page_size: int,
) -> None:
    """One-launch Triton replacement for the 4-D shared-pool KV write path.

    Writes ``cache_k[i]`` and ``cache_v[i]`` to
    ``k_view[loc[i]//ps, loc[i]%ps, :, :]`` (and analogously for V) for
    ``i in [0, N)``.

    Contract:
        - ``k_view``, ``v_view``: 4-D ``(num_pages, page_size, head_num,
          head_dim*)``, contiguous in the trailing ``(head_num, head_dim)``
          dims (i.e., ``stride[-1] == 1`` and ``stride[-2] == head_dim``).
        - ``cache_k``, ``cache_v``: 3-D ``(N, head_num, head_dim*)``,
          contiguous in the trailing ``(head_num, head_dim)`` dims.
        - ``loc``: 1-D int64 or int32, N elements, values in
          ``[0, num_pages * page_size)``. The caller is responsible for
          clamping tombstones to ≥ 0.
        - At ``page_size == 1`` the kernel produces byte-identical output
          to the legacy advanced-indexing path.

    Returns nothing; writes in place.
    """
    if loc.numel() == 0:
        return
    assert k_view.is_cuda and v_view.is_cuda, "store_cache_4d: CUDA only"
    assert k_view.ndim == 4 and v_view.ndim == 4, (
        f"store_cache_4d: k_view/v_view must be 4-D, "
        f"got {k_view.ndim}/{v_view.ndim}"
    )
    assert cache_k.ndim == 3 and cache_v.ndim == 3, (
        f"store_cache_4d: cache_k/cache_v must be 3-D, "
        f"got {cache_k.ndim}/{cache_v.ndim}"
    )
    assert cache_k.shape[0] == cache_v.shape[0] == loc.numel(), (
        "store_cache_4d: cache_k/cache_v/loc batch dim mismatch: "
        f"{cache_k.shape[0]}, {cache_v.shape[0]}, {loc.numel()}"
    )
    assert k_view.dtype == v_view.dtype == cache_k.dtype == cache_v.dtype, (
        "store_cache_4d: dtype mismatch: "
        f"k_view={k_view.dtype}, v_view={v_view.dtype}, "
        f"cache_k={cache_k.dtype}, cache_v={cache_v.dtype}"
    )
    # Stride invariants — the kernel addresses (head_num, head_dim) as one
    # flat ROW_DIM dimension; this requires the trailing two dims to be
    # contiguous. This is always true for views built by
    # `SharedMemoryPool._build_mha_views` (k_stride = (page_bytes/itemsize,
    # k_row_bytes/itemsize, head_dim, 1)). For cache_k/cache_v produced by
    # the model forward, the trailing dims are also contiguous.
    assert k_view.stride(-1) == 1 and k_view.stride(-2) == k_view.shape[-1], (
        f"store_cache_4d: k_view trailing dims must be contiguous; "
        f"got stride={k_view.stride()}, shape={tuple(k_view.shape)}"
    )
    assert v_view.stride(-1) == 1 and v_view.stride(-2) == v_view.shape[-1], (
        f"store_cache_4d: v_view trailing dims must be contiguous; "
        f"got stride={v_view.stride()}, shape={tuple(v_view.shape)}"
    )
    assert cache_k.stride(-1) == 1 and cache_k.stride(-2) == cache_k.shape[-1], (
        f"store_cache_4d: cache_k trailing dims must be contiguous; "
        f"got stride={cache_k.stride()}, shape={tuple(cache_k.shape)}"
    )
    assert cache_v.stride(-1) == 1 and cache_v.stride(-2) == cache_v.shape[-1], (
        f"store_cache_4d: cache_v trailing dims must be contiguous; "
        f"got stride={cache_v.stride()}, shape={tuple(cache_v.shape)}"
    )

    head_num = k_view.shape[2]
    head_dim = k_view.shape[3]
    v_head_dim = v_view.shape[3]
    K_ROW_DIM = head_num * head_dim
    V_ROW_DIM = head_num * v_head_dim
    BLOCK = 128
    N = loc.numel()
    row_dim_max = max(K_ROW_DIM, V_ROW_DIM)
    grid = (N, triton.cdiv(row_dim_max, BLOCK), 2)

    store_cache_4d_kernel[grid](
        k_view,
        v_view,
        cache_k,
        cache_v,
        loc,
        # K/V view strides in elements
        k_view.stride(0),
        k_view.stride(1),
        v_view.stride(0),
        v_view.stride(1),
        # Source-tensor row strides in elements
        cache_k.stride(0),
        cache_v.stride(0),
        K_ROW_DIM=K_ROW_DIM,
        V_ROW_DIM=V_ROW_DIM,
        PAGE_SIZE=page_size,
        BLOCK=BLOCK,
        num_warps=4,
    )


@triton.jit
def set_mla_kv_buffer_fp8_quant_kernel(
    kv_buffer_fp8_ptr,
    cache_k_nope_ptr,
    cache_k_rope_ptr,
    loc_ptr,
    buffer_stride: tl.constexpr,
    nope_stride: tl.constexpr,
    rope_stride: tl.constexpr,
    nope_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    BLOCK: tl.constexpr,
    USE_GDC: tl.constexpr = False,
):
    """Fuse BF16/FP16->FP8 cast with paged KV write."""
    pid_loc = tl.program_id(0)
    pid_blk = tl.program_id(1)

    base = pid_blk * BLOCK
    offs = base + tl.arange(0, BLOCK)
    total_dim = nope_dim + rope_dim
    mask = offs < total_dim

    if USE_GDC:
        tl.extra.cuda.gdc_wait()

    loc = tl.load(loc_ptr + pid_loc).to(tl.int64)
    dst_ptr = kv_buffer_fp8_ptr + loc * buffer_stride + offs

    if base + BLOCK <= nope_dim:
        src = tl.load(
            cache_k_nope_ptr + pid_loc * nope_stride + offs,
            mask=mask,
            other=0.0,
        )
    elif base >= nope_dim:
        offs_rope = offs - nope_dim
        src = tl.load(
            cache_k_rope_ptr + pid_loc * rope_stride + offs_rope,
            mask=mask,
            other=0.0,
        )
    else:
        is_nope = offs < nope_dim
        src_nope = tl.load(
            cache_k_nope_ptr + pid_loc * nope_stride + offs,
            mask=mask & is_nope,
            other=0.0,
        )
        src_rope = tl.load(
            cache_k_rope_ptr + pid_loc * rope_stride + (offs - nope_dim),
            mask=mask & ~is_nope,
            other=0.0,
        )
        src = tl.where(is_nope, src_nope, src_rope)

    # Destination pointer is FP8-typed view; tl.store performs downcast.
    tl.store(dst_ptr, src, mask=mask)

    if USE_GDC:
        tl.extra.cuda.gdc_launch_dependents()


def set_mla_kv_buffer_triton_fp8_quant(
    kv_buffer: torch.Tensor,
    loc: torch.Tensor,
    cache_k_nope: torch.Tensor,
    cache_k_rope: torch.Tensor,
    fp8_dtype: torch.dtype,
):
    """Fuse BF16/FP16 MLA K quantization with paged KV write."""
    kv_buffer_fp8 = kv_buffer.view(fp8_dtype)

    nope_dim = cache_k_nope.shape[-1]
    rope_dim = cache_k_rope.shape[-1]
    total_dim = nope_dim + rope_dim
    BLOCK = 128
    n_loc = loc.numel()
    grid = (n_loc, triton.cdiv(total_dim, BLOCK))

    pdl_kwargs = {"USE_GDC": True, "launch_pdl": True} if is_arch_support_pdl() else {}

    set_mla_kv_buffer_fp8_quant_kernel[grid](
        kv_buffer_fp8,
        cache_k_nope,
        cache_k_rope,
        loc,
        kv_buffer_fp8.stride(0),
        cache_k_nope.stride(0),
        cache_k_rope.stride(0),
        nope_dim,
        rope_dim,
        BLOCK=BLOCK,
        **pdl_kwargs,
    )


@triton.jit
def set_mla_kv_scale_buffer_kernel(
    kv_buffer_ptr,
    cache_k_nope_ptr,
    cache_k_rope_ptr,
    loc_ptr,
    buffer_stride: tl.constexpr,
    nope_stride: tl.constexpr,
    rope_stride: tl.constexpr,
    nope_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_loc = tl.program_id(0)
    pid_blk = tl.program_id(1)

    base = pid_blk * BLOCK
    offs = base + tl.arange(0, BLOCK)
    total_dim = nope_dim + rope_dim
    mask = offs < total_dim  # Make sure don't cross the boundary

    loc = tl.load(loc_ptr + pid_loc)
    dst_ptr = kv_buffer_ptr + loc * buffer_stride + offs

    # Check each offs should read 'nope' or 'rope'
    is_nope = offs < nope_dim
    src_nope = tl.load(
        cache_k_nope_ptr + pid_loc * nope_stride + offs, mask=mask & is_nope, other=0.0
    )
    src_rope = tl.load(
        cache_k_rope_ptr + pid_loc * rope_stride + (offs - nope_dim),
        mask=mask & ~is_nope,
        other=0.0,
    )

    # Combine nope + rope
    src = src_nope + src_rope
    tl.store(dst_ptr, src, mask=mask)


def set_mla_kv_scale_buffer_triton(
    kv_buffer: torch.Tensor,
    loc: torch.Tensor,
    cache_k_nope: torch.Tensor,
    cache_k_rope: torch.Tensor,
):
    nope_dim = cache_k_nope.shape[-1]
    rope_dim = cache_k_rope.shape[-1]
    total_dim = nope_dim + rope_dim
    BLOCK = 128  # Keep origin, works for smaller total_dim as well.
    n_loc = loc.numel()
    grid = (n_loc, triton.cdiv(total_dim, BLOCK))

    set_mla_kv_scale_buffer_kernel[grid](
        kv_buffer,
        cache_k_nope,
        cache_k_rope,
        loc,
        kv_buffer.stride(0),
        cache_k_nope.stride(0),
        cache_k_rope.stride(0),
        nope_dim,
        rope_dim,
        BLOCK=BLOCK,
    )


@triton.jit
def get_mla_kv_buffer_kernel(
    kv_buffer_ptr,
    cache_k_nope_ptr,
    cache_k_rope_ptr,
    loc_ptr,
    buffer_stride: tl.constexpr,
    nope_stride: tl.constexpr,
    rope_stride: tl.constexpr,
    nope_dim: tl.constexpr,
    rope_dim: tl.constexpr,
):
    pid_loc = tl.program_id(0)
    loc = tl.load(loc_ptr + pid_loc).to(tl.int64)
    loc_src_ptr = kv_buffer_ptr + loc * buffer_stride

    nope_offs = tl.arange(0, nope_dim)
    nope_src_ptr = loc_src_ptr + nope_offs
    nope_src = tl.load(nope_src_ptr)

    tl.store(
        cache_k_nope_ptr + pid_loc * nope_stride + nope_offs,
        nope_src,
    )

    rope_offs = tl.arange(0, rope_dim)
    rope_src_ptr = loc_src_ptr + nope_dim + rope_offs
    rope_src = tl.load(rope_src_ptr)
    tl.store(
        cache_k_rope_ptr + pid_loc * rope_stride + rope_offs,
        rope_src,
    )


def get_mla_kv_buffer_triton(
    kv_buffer: torch.Tensor,
    loc: torch.Tensor,
    cache_k_nope: torch.Tensor,
    cache_k_rope: torch.Tensor,
):
    # The source data type will be implicitly converted to the target data type.
    nope_dim = cache_k_nope.shape[-1]  # 512
    rope_dim = cache_k_rope.shape[-1]  # 64
    n_loc = loc.numel()
    grid = (n_loc,)

    get_mla_kv_buffer_kernel[grid](
        kv_buffer,
        cache_k_nope,
        cache_k_rope,
        loc,
        kv_buffer.stride(0),
        cache_k_nope.stride(0),
        cache_k_rope.stride(0),
        nope_dim,
        rope_dim,
    )


def maybe_init_custom_mem_pool(
    device: str,
) -> Tuple[bool, Optional[Any], Optional[str]]:
    """
    Initialize custom memory pool based on environment variable.

    This function can be modified to support more features that require a custom memory pool.

    Args:
        device: The device to allocate memory on

    Returns:
        Tuple of (enable_custom_mem_pool, custom_mem_pool, custom_mem_pool_type)
    """
    enable_custom_mem_pool = (
        True if envs.SGLANG_MOONCAKE_CUSTOM_MEM_POOL.get() is not None else False
    )

    if enable_custom_mem_pool:
        # Currently, only mooncake requires a custom mem pool for MNNVL/Barex PD disaggregation
        from sglang.srt.disaggregation.mooncake.utils import (
            init_mooncake_custom_mem_pool,
        )

        return init_mooncake_custom_mem_pool(device)
    else:
        return False, None, None


def convert_to_bigram_key(tokens: List[int]) -> List[Tuple[int, int]]:
    # EAGLE uses bigram keys in the radix tree since draft sequence is the one-token-shifted version of target
    # [1, 2, 3, 4] -> [(1,2), (2,3), (3,4)]
    if len(tokens) and isinstance(tokens[0], tuple):
        return tokens
    if len(tokens) < 2:
        return []
    return [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]


def get_hash_str(token_ids: List[int], prior_hash: Optional[str] = None) -> str:
    hasher = hashlib.sha256()

    if prior_hash:
        hasher.update(bytes.fromhex(prior_hash))

    for t in token_ids:
        if isinstance(t, tuple):
            # EAGLE bigram mode: hash both elements to uniquely identify the bigram
            for elem in t:
                hasher.update(elem.to_bytes(4, byteorder="little", signed=False))
        else:
            # Regular mode: single integer token
            hasher.update(t.to_bytes(4, byteorder="little", signed=False))

    return hasher.hexdigest()


def hash_str_to_int64(hash_str: str) -> int:
    """Convert SHA256 hex string to signed 64-bit integer for events.

    Takes first 16 hex characters (64 bits) and converts to signed int64 range.
    """
    uint64_val = int(hash_str[:16], 16)
    if uint64_val >= 2**63:
        return uint64_val - 2**64
    return uint64_val


def compute_node_hash_values(node: Any, page_size: int) -> List[str]:
    """Compute SHA256-based hash values for position-aware KV block IDs."""
    hash_values = []

    parent_hash = None
    if node.parent is not None and node.parent.hash_value is not None:
        if len(node.parent.key) > 0 and len(node.parent.hash_value) > 0:
            parent_hash = node.parent.hash_value[-1]

    logical_len = len(node.key)
    for start in range(0, logical_len, page_size):
        end = min(start + page_size, logical_len)
        if end <= start:
            continue
        hash_val = node.key.hash_page(start, end, parent_hash)
        hash_values.append(hash_val)
        parent_hash = hash_val
    return hash_values


def split_node_hash_value(
    child_hash_value: Optional[List[str]], split_len: int, page_size: int
) -> tuple[Optional[List[str]], Optional[List[str]]]:
    """Split hash_value between parent and child nodes during node splitting.

    Args:
        child_hash_value: The hash_value list from the child node being split
        split_len: The length at which to split (in tokens)
        page_size: The page size for calculating number of pages

    Returns:
        Tuple of (new_node_hash_value, updated_child_hash_value)
    """
    if child_hash_value is None:
        return None, None

    if page_size == 1:
        split_pages = split_len
    else:
        split_pages = split_len // page_size

    new_node_hash = child_hash_value[:split_pages]
    child_hash = child_hash_value[split_pages:]

    return new_node_hash, child_hash
