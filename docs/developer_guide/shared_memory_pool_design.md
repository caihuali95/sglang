# SharedMemoryPool & MultiEndedAllocator — Design Document

> **Audience**: engineers new to SGLang's memory subsystem AND core
> contributors who need a reference for the `--enable-shared-memory-pool` feature.
> The document explains the problem, the conceptual design, the concrete
> implementation, and the design trade-offs.

---

## Table of contents

1. [TL;DR](#1-tldr)
2. [Background](#2-background)
3. [Key concepts](#3-key-concepts)
4. [High-level design](#4-high-level-design)
5. [Memory layout details](#5-memory-layout-details)
6. [`MultiEndedAllocator` in detail](#6-multiendedallocator-in-detail)
7. [Relocation bookkeeping: `SlotBacktrack` & `Relocator`](#7-relocation-bookkeeping)
8. [Scheduler & model-runner integration](#8-scheduler--model-runner-integration)
9. [PD disaggregation compatibility (NOT YET SUPPORTED)](#9-pd-disaggregation-compatibility-not-yet-supported)
10. [Future extensions](#10-future-extensions)
11. [Implementation history: bugs found & fixes applied](#11-implementation-history-bugs-found--fixes-applied)
12. [Critical design flaws & open issues](#12-critical-design-flaws--open-issues)
13. [Glossary](#13-glossary)

> **Status note (this revision).** The relocation subsystem was redesigned
> from a *deferred per-tick flush* (`RelocationLog.flush` called from a
> scheduler hook) to **immediate, synchronous reference updates at free
> time** (`Relocator.apply`). The post-refactor debugging trail (§11.5–11.10)
> closed the last laundering bug (§11.9) and then stripped the dense
> diagnostic instrumentation it had accumulated (§11.10) — there is no more
> `-1` poisoning, no `SGLANG_SHARED_POOL_*_LOG` knobs, no `DOUBLE-ALLOCATION`/
> `DST-NOT-EMPTY`/`ORPHAN-TENSOR`/`IN-PLACE-MUTATION`/`FIRST-BIND` tripwires;
> the surviving checks are listed in §7.7. As of eval_47 the full Mamba + SWA
> eval matrix passes (crash-free, correct, at parity with the static-partition
> baseline on balanced workloads; slower on the deliberately overcommitted
> skewed workloads — §12 O1). Sections 4.3, 4.4, 7, 8 describe the current
> design; §11 the history; §12 the remaining flaws/issues. If you are reading
> old code comments that mention `RelocationLog`, `flush_relocations`,
> `resolve_to_current`, a per-tick flush, `poison_and_clear_slot`, or any
> `*_LOG` env knob, they refer to a since-removed earlier state.

---

## 1. TL;DR

**Problem.** SGLang's hybrid models (SWA — Gemma2 etc.; Mamba — Jamba, NemotronH,
Granite) need **two** KV/state memory pools. Today those pools are **statically
partitioned** at boot: one might run out of slots while the other has idle
capacity.

**Solution.** One flag, `--enable-shared-memory-pool`, replaces the two
partitioned pools with a single physical byte buffer split **dynamically**
by two cooperating allocators that grow from opposite ends:

```
   grow-up ──►                                       ◄── grow-down
   ┌─────────────────────┬─────────────────┬──────────────────────┐
   │ Pool A  (allocated) │   free middle   │ Pool B  (allocated)  │
   └─────────────────────┴─────────────────┴──────────────────────┘
   0                                                       total_bytes
```

When one side wants more space than the middle gap allows, the request is
rejected and the scheduler's existing radix-cache eviction path reclaims
slots; `free` then **eagerly compacts** the boundary inward so the allocated
range stays hole-free.

Slots that move must have their references (in `req_to_token`, `TreeNode.value`,
`TreeNode.mamba_value`, the SWA / Mamba mapping tensors, `Req.mamba_pool_idx`,
and `Req.prefix_indices`) updated. A reverse index — `SlotBacktrack` — is
maintained as writes happen so the update is *targeted* (O(num_relocations))
rather than a full scan of every index tensor. The update is applied
**immediately, inside `free()`**, by `Relocator.apply` — there is no deferred
log, no per-tick flush hook, and no chain-chasing. After `free()` returns,
every bound holder already names the post-compaction slot id.

---

## 2. Background

### 2.1 What is a KV cache?

Every transformer decoder produces per-layer Key and Value vectors for each
token it processes. Subsequent tokens attend over all earlier K/V rows, so
storing them — the **KV cache** — is essential for efficient serving.

Each **slot** in the cache holds the KV for one token across all layers. The
cache size (in slots) caps the total tokens we can keep warm across all
in-flight requests.

### 2.2 Hybrid attention models have two pools

Some modern models mix attention kinds in the same forward pass:

| Model family          | Layer mix                                      |
|-----------------------|------------------------------------------------|
| **Gemma2 / SWA**      | Some layers use full attention, others use sliding-window attention (SWA). |
| **Jamba / NemotronH / Granite** | Some layers are full attention, others are Mamba. |

SGLang handles this with two memory pools:

- **SWA case** — `SWAKVPool` holds:
  - `full_kv_pool` : KV for full-attention layers
  - `swa_kv_pool`  : KV for SWA layers
- **Mamba case** — `HybridLinearKVPool` + `HybridReqToTokenPool`:
  - `full_kv_pool` : MHA-style KV for full-attention layers
  - `mamba_pool`   : Mamba state (per-request conv + temporal, ≠ per-token)

### 2.3 The static-partition problem

Today those two pools are sized at startup via knobs like
`--swa-full-tokens-ratio` (SWA) and `--mamba-full-memory-ratio` (Mamba). The
split is **fixed for the life of the process**. A production workload rarely
matches the static ratio perfectly:

### 2.4 Goal

Keep the **same total budget**, but let the split float at runtime so neither
pool starves while the other is idle.

---

## 3. Key concepts

### 3.1 Memory pool vs. allocator

- **Memory pool** (e.g. `MHATokenToKVPool`, `MambaPool`) — owns the physical
  tensors and exposes per-layer `k_buffer[l]`, `v_buffer[l]`, etc. Attention
  kernels index into these.
- **Allocator** (e.g. `TokenToKVPoolAllocator`) — owns the **free-list** of
  slot indices. `alloc(N)` returns N unused slot ids; `free(indices)` returns
  slots to the free-list. The pool doesn't know which slots are alive (while `MambaPool` combining memory pool and allocator is an exception).

### 3.2 Slot index vs byte offset

A *slot* is an index into the pool's logical space (`[0, max_slots)`). Under
the hood, slot `K` for layer `l` lives at a specific **byte offset** inside
the tensor's storage:

```
  byte_offset(layer l, slot K, K-row) = base + l * row_bytes_per_layer
                                             + K * slot_stride_bytes
```

In the classic static-partition design, one pool is one set of tensors, so
slot `K` always names the same physical bytes. In the shared design, *two
pools share bytes* — so slot K of pool A and slot K of pool B may or may not
refer to the same physical bytes. The allocator tracks byte frontiers to keep
ranges disjoint.

### 3.3 Radix cache & `req_to_token`: where slot ids live

Slot ids are referenced from several structures. **Every container that
*persists* a slot id across forwards needs binding**, otherwise eager
compaction silently invalidates it (see §7.5 for the hazard, §11 for the
two real-world bugs this caused):

| Reference holder | How many slots? | Slot space | Bound? | Key observation |
|---|---|---|---|---|
| `ReqToTokenPool.req_to_token[req_idx, pos]` | many | full | ✅ via `req_position` | Multiple requests that share a **common prefix** will hold the same slot id at the **same column `pos`**. |
| `TreeNode.value` (radix cache) | many per node | full / SWA | ✅ via `tree_node` (attr=`"value"`) | A slot appears in **at most one** tree node. |
| `TreeNode.mamba_value` (`MambaRadixCache`) | 1 per node | mamba | ✅ via `tree_node` (attr=`"mamba_value"`) | Cloned tensor of `req.mamba_pool_idx`; without binding, eager compaction silently leaves it stale → assertion in `evict_mamba`. |
| `SWAKVPool.full_to_swa_index_mapping` | 1 entry per live full-slot | full → swa | ✅ via `aux` + eager mapping update | Maps a full-pool slot id → swa-pool slot id. |
| `HybridReqToTokenPool.req_index_to_mamba_index_mapping` | 1 per live req | mamba | ✅ via `aux` | Maps `req_pool_idx` → mamba slot id. |
| `Req.mamba_pool_idx` (Python attribute) | 1 per running req | mamba | ✅ via `py_attr` | CPU-side scalar holding the current mamba slot. |
| **`Req.prefix_indices`** | many per req | full | ✅ via `aux` (one entry per cell) | **Cloned** at radix-cache lookup time; persists across many ticks while a chunked req sits in the waiting queue. Originally treated as "transient" but actually persists across ticks; without binding it accumulates relocation drift. |
| Short-lived per-forward tensors: `out_cache_loc`, etc. | transient | full | ❌ — by design | Regenerated within a single forward and consumed before any `free()` can relocate their slots — they never persist into a later tick. (Note: a *clone* that escapes the forward — e.g. into `req.prefix_indices` or the local `value` clone in `_insert_helper` — is NOT in this row; see §12 F3.) |

### 3.4 Relocation

If slot K is moved to slot K' (data copied, free-list updated), **every
reference** to K above must become K' **before the next forward**. Otherwise
attention reads stale data.

### 3.5 Reverse mapping

To update references efficiently we need the opposite direction: given a
slot id K, find all places that reference K. We call this the
**`SlotBacktrack`** — see §7.

---

## 4. High-level design

### 4.1 One physical byte buffer, many logical views

```
  ┌─────────────────────────────────────────────────────────────────────┐
  │                    SharedMemoryPool._raw (uint8)                    │
  │                         total_bytes                                 │
  └─────────────────────────────────────────────────────────────────────┘
        ▲                                                       ▲
        │  aliased by reinterpret-view                          │
        │                                                       │
   ┌────┴───────────────────┐                     ┌─────────────┴──────────┐
   │ SharedMHATokenToKVPool │                     │ SharedMHATokenToKVPool │
   │   (sub-pool "full")    │                     │   (sub-pool "swa")     │
   │   slot 1,2,3,...grow↑  │                     │   ...,62,63 grow↓      │
   └────────────────────────┘                     └────────────────────────┘
   or SharedMambaPool                             or SharedMambaPool
```

Each sub-pool exposes the *same* per-layer `k_buffer[l]` / `v_buffer[l]` (or
`conv_state[i]` / `temporal_state`) API it did before — only now those
tensors are **strided views** into `_raw`, not fresh allocations.

### 4.2 `MultiEndedAllocator` — two-ended coordination

```
  Allocator A (grow-up)                Allocator B (grow-down)
     ▲                                           ▼
     │ next_slot = watermark_A                   │ next_slot = watermark_B
     │                                           │
     │──────────── shared byte space ────────────│
     [ A used ][      free middle      ][ B used ]
     0                                      total_bytes
```

- `alloc(N)` succeeds iff the gap between the two byte frontiers is ≥ N slots.
- `free(indices)` shrinks the watermark and **eagerly compacts** (copies the
  boundary slot into any freed interior hole). The allocated range stays
  contiguous; there are no holes between calls.

### 4.3 Immediate reference updates via `SlotBacktrack` + `Relocator`

When `free` compacts slot K→K', every reference to slot K must become K'.
We maintain a reverse index keyed by slot id, one per sub-pool:

```
  SlotBacktrack                                              cardinality
  ├── tree_node   : slot → _TreeNodeRef(node, position, attr)     1:1
  ├── req_position: slot → _ReqPosRef(col, rows: Set[req_idx])    1:(many reqs, one col)
  ├── aux         : slot → [_AuxRef(tensor, index), ...]          mapping/clone tensors
  └── py_attr     : slot → [(obj, "attr_name"), ...]              Req.mamba_pool_idx
```

This is *populated* by hook points at each write site (inside shared-mode
radix caches, inside `SharedHybridReqToTokenPool.alloc`, the `req_to_token`
write wrapper, etc.). When `free()` relocates `(src → dst)` it calls
`Relocator.apply(sub_pool, src, dst)` **right then** — looking up `src`'s
binder entries directly (O(num_relocations) instead of scanning all index
tensors) and rewriting every holder in place before `free()` returns.
There is no pending log and no later flush.

`Relocator` also owns a small **inverse history** (`List[(src_tensor,
dst_tensor)]` per allocator) so spec-decode `restore_state` can undo a batch
of relocations by replaying the inverse moves. The composite allocators
clear the consumed slice each tick so it doesn't grow unbounded.

A **free-time `clear_slot` catch-all** runs first: for every slot whose bytes
are about to be overwritten by eager compaction, `SlotBacktrack.clear_slot`
pops any binder entry it still carries (some caller path — notably
`_free_dedup_slots` for `req_position` — frees without unbinding). Cheap dict
pops, no warning, no sentinel poisoning. If a *holder* (not just the binder)
still names the freed slot, the next read of that holder trips the hard
stale-slot assertion in `_free_eager`. See §7.5.

### 4.4 Architecture diagram

```
                        ┌─────────────────────────────┐
                        │       SharedMemoryPool      │
                        │   (raw uint8 byte buffer)   │
                        └──────────┬─────────┬────────┘
                                   │ views   │ views
                      ┌────────────▼──┐   ┌──▼──────────────────┐
                      │ SharedMHA*    │   │ SharedMHA* / Mamba* │
                      │ (sub-pool A)  │   │ (sub-pool B)        │
                      └──┬────────────┘   └──────────┬──────────┘
                         │                           │
         ┌───────────────▼───────────────────────────▼──────────┐
         │        SharedSWA/MambaTokenToKVPoolAllocator        │
         │   ┌───────────────┐     ┌──────────────────┐         │
         │   │MultiEndedAlloc│◄───►│ MultiEndedAlloc  │         │
         │   │  (A: grow-up) │     │  (B: grow-down)  │         │
         │   └───────┬───────┘     └───────┬──────────┘         │
         │           │  free() ─► Relocator.apply(src,dst)      │
         │           └────────── Relocator ─────────┘           │
         │   (SlotBacktrack[A], SlotBacktrack[B], _req_to_token │
         │    back-ref, per-allocator _inverse_history)         │
         └──────────────────────┬──────────────────────────────┘
                                │ in-place, synchronous (no flush hook)
                                ▼
                      ┌────────────────────┐
                      │ req_to_token       │ ← writes hooked via
                      │ radix TreeNode.val │    wrap_req_to_token_pool_write
                      │  + .mamba_value    │   and BasePrefixCache._set_node_value /
                      │ full_to_swa / mamba│   _set_node_mamba_value
                      │ Req.mamba_pool_idx │   _set_req_mamba_pool_idx
                      │ Req.prefix_indices │   _set_req_prefix_indices
                      └────────────────────┘
```

---

## 5. Memory layout details

**Key fact.** In both supported hybrid families the two sub-pools have
**different per-slot byte sizes** (`entry_bytes_A ≠ entry_bytes_B`) — i.e.
both cases are **heterogeneous**. We treat them uniformly.

Why? A slot in a sub-pool holds the KV (or Mamba state) for one token (or
one request, for Mamba) across **that sub-pool's own layers**. The two
sub-pools own *disjoint subsets of the model's layers*:

| Case             | Sub-pool A ("full")                     | Sub-pool B                     |
|------------------|-----------------------------------------|---------------------------------|
| SWA hybrid       | full-attention layers (`N_full`)        | SWA layers (`N_swa`)            |
| Mamba hybrid     | full-attention layers (`N_full`)        | Mamba layers (`N_mamba`)        |

So `entry_bytes_A ∝ N_full` and `entry_bytes_B ∝ N_swa` (or `N_mamba`). For
real models `N_full ≠ N_swa` (Gemma2's typical 1:5 ratio; Jamba alternates
similarly), and Mamba per-layer bytes differ from MHA per-layer bytes
anyway. The **slot-index spaces of A and B are independent**: slot K of A and
slot K of B refer to *different* physical bytes and hold *different* data.

### 5.1 Per-slot byte layout inside one sub-pool

Inside one sub-pool's entry (use sub-pool A as an example, across all full attention layers):

```
  slot K entry:

    Byte offset inside entry:
      0                                                    entry_bytes_A
      │                                                           │
      ├─── L0 K row ── L0 V row ── L1 K row ── L1 V row ── … ─────┤

    Slot K of sub-pool A at raw byte = base_A + K * entry_bytes_A
    entry_bytes_A = 2 · layer_num_A · head_num · head_dim · dtype.itemsize
```


### 5.2 Two heterogeneous sub-pools sharing one buffer

```
  raw[0]                                                      raw[total_bytes]
  ┌─────────────────────────────────────────────────────────────────┐
  │                          (uint8)                                │
  └─────────────────────────────────────────────────────────────────┘
     ▲                                                       ▲
     │ reinterpreted as                                      │ reinterpreted as
     │ shape (N_full, total_slots_MHA, 2, heads, head_dim)   │ shape (N_mamba, total_slots_MAMBA, *)
     │ dtype = kv_dtype                                      │ dtype = conv_dtype / ssm_dtype
     │                                                       │
  ┌──┴──────────────┐                                 ┌──────┴────────────┐
  │  big_mha        │ ← sub-pool A (grow-up, anchor=0)│ big_mamba_conv[i] │ ← sub-pool B (grow-down)
  │                 │                                 │ big_mamba_temporal│
  └─────────────────┘                                 └───────────────────┘
```

- `entry_bytes_A` = per-slot-all-layers bytes for sub-pool A.
- `entry_bytes_B` = per-slot-all-layers-all-states bytes for sub-pool B.
  For SWA-B this is `2 · N_swa · head_num · head_dim · itemsize`;
  for Mamba-B this is `num_mamba_layers · (Σ conv_row_bytes + temporal_row_bytes)`.
- `max_slots_A = total_bytes // entry_bytes_A`; similarly for B.
- Both views **alias the same bytes**, but each allocator reports a
  different `max_slots` because slots of different sizes are compared.
- Non-overlap invariant (byte-space):

  ```
    allocated_count_A · entry_bytes_A  +  allocated_count_B · entry_bytes_B
                                       ≤ total_bytes
  ```

### 5.3 Grow-up vs grow-down

Both sub-pool anchors are `0` — views are constructed uniformly with
positive strides from the raw buffer's base. What distinguishes the two
pools is **allocator direction**:

- Grow-up allocator starts its watermark at `min_slot_index` (see §5.6)
  and hands out ascending slot ids.
- Grow-down allocator starts at `max_slots − 1` and hands out descending
  slot ids.

Allocated byte ranges grow toward each other from opposite ends of the
buffer; free bytes always form a single contiguous middle gap. `alloc`'s
fit check is O(1) — compare the two allocators' byte frontiers.

### 5.4 Byte-anchor alignment

PyTorch requires a tensor's `data_ptr()` to be a multiple of `dtype.itemsize`
for `.view(dtype)` / `as_strided(...)` to be well-defined. CUDA's
`cudaMalloc` returns memory aligned to **256 bytes** (CUDA's documented
guarantee), so the base `_raw.data_ptr()` is already 256-aligned.

With all sub-pool anchors = 0, every view's `data_ptr()` is just
`raw.data_ptr() + inner_offset`, where `inner_offset` is a multiple of
`row_bytes` (and therefore of `dtype.itemsize`). No explicit alignment
rounding is needed — alignment falls out of the view-construction math.

For real LLM configs (`head_num=8, head_dim=128, fp16` →
`row_bytes=2048`, `entry_bytes ≥ 4096`), every slot row is automatically
256-aligned. The per-view `anchor_bytes % itemsize == 0` assertion in
`_build_mha_views` / `_build_mamba_views` is trivially satisfied.

### 5.5 `intermediate_ssm` / `intermediate_conv_window` stay local

Speculative decoding needs per-draft-token intermediate state buffers whose
outer dimension is sized by `spec_state_size + 1`, **not** by `total_bytes /
entry_bytes`.
So these do not participate in the shared buffer — `SharedMambaPool` allocates them locally.
A future implementation can promote them to the shared buffer as the third sub-pool.

### 5.6 Slot-0 padding-write safety (`min_slot_index`)

**The problem.** SGLang reserves slot 0 of every KV pool as a "padding sink"
(`out_cache_loc[i] = 0` for padded positions after CUDA-graph zero-init).
`store_cache` ([kvcache.cuh:89-112](../../sgl-kernel/csrc/elementwise/kvcache.cuh#L89))
and the PyTorch fallback ([memory_pool.py:117-123](../../python/sglang/srt/mem_cache/memory_pool.py#L117))
both write unconditionally — no kernel branches on `loc == 0`. In a shared
byte buffer, pool `i`'s slot-0 dummy writes cover raw bytes `[0, entry_i)`.
If another pool `j` has `entry_j < entry_i` and allocates slot 1 at bytes
`[entry_j, 2 · entry_j)` — inside pool i's slot-0 dummy range — its real
data gets clobbered.

**The fix.** Let `entry_max = max(entry_i over all sub-pools)`. For each
sub-pool `i` define:

```
  min_slot_index_i = ceil(entry_max / entry_i)
```

The allocator **never hands out slot indices below `min_slot_index_i`**.
Slots `[0, min_slot_index_i)` are reserved:

- Slot 0 is the padding sink (unchanged behavior).
- Slots `1, 2, …, min_slot_index_i − 1` sit unused (no convention writes to
  them — `out_cache_loc` only produces `0` for padding).

**Why this is safe.** The union of every pool's slot-0 dummy-write region
is `[0, entry_max)`. Pool i's first allocatable slot is
`min_slot_index_i`, so its real data lives at bytes
`≥ min_slot_index_i · entry_i ≥ entry_max` — strictly above any pool's
slot-0 dummy range. No pool's real data can be clobbered. ✓

**Example** (`entry_L = 64`, `entry_H = 192`, so `entry_max = 192`):

- Pool L (grow-up, smaller entry): `min_L = ceil(192/64) = 3`. Slots 0, 1,
  2 reserved. First allocatable slot = 3 at bytes `[192, 256)`.
- Pool H (grow-down, larger entry): `min_H = ceil(192/192) = 1`. Only
  slot 0 reserved (unchanged). First allocatable slot = 1 (which the
  grow-down allocator reaches last — its first alloc returns `max_H − 1`
  at bytes near `total_bytes`).

Pool L's slot 3 at `[192, 256)` is disjoint from pool H's slot 0 at
`[0, 192)`. ✓

**Memory cost.** Each pool reserves `min_slot_index_i · entry_i` bytes at
its low-index end, which is `≥ entry_max` and `< entry_max + entry_i`.
Per pool this is at most one extra `entry_i` slot of overhead. Negligible
vs multi-GB KV budgets.

**Scalability to N > 2.** The invariant
`min_slot_index_i · entry_i ≥ entry_max` is computed **per-pool
independently** — every pool's real data begins at bytes `≥ entry_max`.
No positional asymmetry; the scheme applies uniformly to any number of
sub-pools sharing the buffer.

**Wiring.** `SharedMemoryPool.min_slot_index(name)` exposes the per-pool
value; `MultiEndedAllocator.__init__` reads it via
`shared_buffer.min_slot_index(sub_pool_name)` and initializes its watermark
accordingly. `allocated_count`, `_byte_low_frontier`, `_byte_high_frontier`,
`_allocated_range`, and the composite allocator's slot-space headroom
calculation all use `min_slot_index` in place of the hardcoded `1` that
used to denote "slot 0 reserved".

---

## 6. `MultiEndedAllocator` in detail

### 6.1 Invariants (enforced by eager compaction)

At any time between allocator calls:

- Grow-up allocator's live slots form the contiguous range `[1, watermark_A)`.
- Grow-down allocator's live slots form the contiguous range `(watermark_B, max_slots - 1]`.
- No holes, no free-list bookkeeping across calls.
- The complement is a single contiguous byte range in the middle ← **the**
  free gap.

### 6.2 `alloc(N)` — direct-peer byte-frontier check

```python
# Grow-up variant
gap_bytes = peer._byte_low_frontier() - self._byte_high_frontier()
if N > gap_bytes // entry_bytes: return None
select = torch.arange(watermark, watermark + N)
watermark += N
return select
```

Grow-down is symmetric (returns `[watermark - N + 1, watermark]`, watermark
decrements). Only the **direct peer** frontier is checked — for future implementation
of N > 2 sub-pools this is the correct local invariant (a request cannot reach bytes
on the far side of a direct neighbor even if those bytes happen to be free).

### 6.3 `free(indices)` — eager compaction

Given indices to free, walk from the boundary inward and swap the boundary
slot into each freed interior hole:

```
  Before free([2, 4]):     watermark = 7
  [ _ , 1 , 2 , 3 , 4 , 5 , 6 ,      _ ,  …  ]    ← grow-up, _ = reserved/free
        ↑                   ↑
        slot 1     slot 6 = boundary

  Step 1: free slot 4 → move slot 6 into slot 4
  [ _ , 1 , 2 , 3 , 6 , 5 , _ ,       _ , … ]
        ↑               ↑
        slot 1      new boundary

  Step 2: free slot 2 → move slot 5 into slot 2
  [ _ , 1 , 5 , 3 , 6 , _ , _ ,       _ , … ]
                        ↑
                        watermark = 5
```

Each non-boundary free triggers **one row-copy per layer** (batched across all
freed slots in a single `move_kv_cache` / `copy_from` call), then
`Relocator.apply(src, dst)` rewrites every bound reference **synchronously,
before `free()` returns** (§7.4). Each `(src, dst)` batch is also appended to
the allocator's `_inverse_history` for spec-decode rollback.

Why eager (not lazy)? From the PDF's design notes (lazy is crossed out):

- `alloc` check becomes a scalar compare (no free-list drain / hole tracking).
- Row-copy cost is small (~tens of KB per layer) vs attention compute.

#### 6.3.1 Boundary frees do NOT trigger a relocation

When the slot being freed `s` *equals* the boundary `b = watermark - 1`
(grow-up) or `watermark + 1` (grow-down), no data needs to move — the
allocator simply decrements `watermark`. Looking at the loop:

```python
b = self.watermark - 1
for s in uniq:                              # descending
    if s == b:
        b -= 1; self.watermark -= 1         # ← no relocation; no apply()
    elif s < b:
        src_list.append(b); dst_list.append(s)
        b -= 1; self.watermark -= 1         # ← (b → s) relocation
```

This is correct — there's nothing to rewrite because no data moved. But the
absence of a relocation has a subtle implication for **unbound containers**
that hold the slot id `s`: `Relocator.apply` is never invoked for `s`, so an
unbound container (e.g. a `node.mamba_value` clone, a `req.prefix_indices`
clone) holding `s` is silently abandoned, and a later free of `s` from that
container fires the out-of-range (`stale-slot`) assertion. The free-time
`clear_slot` catch-all *does* run for `s` (it pops any binder entry), but it
can't reach an *unbound* holder — that's why everything persistent must be
bound (§7.6) and clones that escape the bound containers must be re-validated
before reuse (§12 F3).

Under heavy concurrency a noticeable fraction of frees hit the boundary, so
this is not a corner case. **Bind every container that persists slot ids
across forwards** (§7.6); for clones that escape the bound containers,
re-validate against the allocator before reuse (§12 F3).

### 6.4 Size reporting

`available_size()` returns `gap_bytes // entry_bytes`, instead of the
sub-pool's own `max_slots - 1 - allocated_count`. When the peer allocates,
the peer's watermark moves and our reported `available_size` shrinks — the
scheduler's existing "evict if `available_size < need_size`" loop picks this
up automatically.

---

## 7. Relocation bookkeeping

> **This section describes the current (immediate-apply) design.** The
> deferred `RelocationLog.flush` design it replaced is summarised in §11.6.

### 7.1 What needs updating when a slot moves

Recall the reference holders from §3.3:

| Holder | Dimension that changes on relocation | Sub-pool |
|---|---|---|
| `req_to_token[req_idx, pos]` | int32 cell holding full-pool slot id | full |
| `TreeNode.value[i]` | int64 cell holding full-pool slot id | full / SWA |
| **`TreeNode.mamba_value[0]`** | int64 cell holding mamba-pool slot id | mamba |
| `full_to_swa_index_mapping[full_idx]` | both the **index** (full-slot id) and the **value** (swa-slot id) | full + swa |
| `req_index_to_mamba_index_mapping[req_pool_idx]` | **value** is a mamba-slot id | mamba |
| `Req.mamba_pool_idx` | CPU-side scalar tensor holding mamba-slot id | mamba |
| **`Req.prefix_indices[i]`** | int64 cell holding full-pool slot id | full |

### 7.2 `SlotBacktrack` — the reverse index

One `SlotBacktrack` per sub-pool. Each sub-pool's slot ids are integers in
its own `[0, max_slots)` space. The backtrack stores, per slot:

```
SlotBacktrack
 ├── tree_node   : Dict[int, _TreeNodeRef]       # 1:1
 ├── req_position: Dict[int, _ReqPosRef]         # 1:(many reqs, one column)
 ├── aux         : Dict[int, List[_AuxRef]]      # mapping / clone tensors, n:1
 └── py_attr     : Dict[int, List[(obj, attr_name)]]   # Req.mamba_pool_idx
```

`_TreeNodeRef(node, position, attr)` — `attr` is `"value"` or `"mamba_value"`
(§7.3.1). `Relocator.apply` writes `getattr(node, attr)[position] = dst`.

`_ReqPosRef(col, rows: Set[int])` — **row-tracked**. The original design
stored only the column; a slot bound at a different column in a later req
forced an `OVERWRITE` and left the prior req's `req_to_token` cell stale. By
tracking the exact set of `req_pool_indices` (rows) that hold this slot at
column `col`, `Relocator.apply` reads/writes only those cells (typically 1–5)
instead of scanning the whole column, and the multi-column failure mode
becomes unrepresentable: if `bind_req_position` ever observes a *different*
column for an already-known slot, that's an `assert` (a caller freed the slot
without going through the row-aware unbind — see §12).

`_AuxRef(tensor, index)` — a `(tensor, scalar-index)` cell in a mapping
tensor (`full_to_swa_index_mapping`, `req_index_to_mamba_index_mapping`) or
a clone tensor (`req.prefix_indices`).

There is no `transfer()`-then-return step any more: `Relocator.apply` pops
`src`'s entries, rewrites the holders in place, and re-binds them under `dst`
itself (§7.4).

### 7.3 `SlotBacktrackBinder` — null-safe facade

Every code site that *writes* a slot id into one of the holders above calls
the binder unconditionally. For non-shared-pool paths, the binder is the
null-binder (all methods are no-ops). For shared paths, a real binder
(`SlotBacktrackBinder(backtrack)`) routes calls into the reverse index.

```python
binder.bind_tree_node(slot, node, position, attr="value")  # node.value = X
binder.bind_tree_node(slot, node, 0, attr="mamba_value")   # node.mamba_value = X
binder.unbind_tree_node(slot, node, position, attr)        # match-aware (§7.3.2)
binder.bind_req_position(slot, row, col)                   # on req_to_token.write(...)
binder.unbind_req_position(slot, row)                      # row no longer holds slot
binder.bind_aux(slot, tensor, index)                       # on full_to_swa write,
                                                           # req_index_to_mamba write,
                                                           # req.prefix_indices cell
binder.unbind_aux(slot, tensor, index)                     # identity-keyed unbind
binder.bind_py_attr(slot, obj, attr_name)                  # on req.mamba_pool_idx = X
binder.unbind_py_attr(slot, obj, attr_name)
```

This avoids `isinstance` checks at hot sites. All `unbind_*` calls are
**match-aware / identity-keyed**: they only remove the binder entry if it
actually points at the `(node, position, attr)` / `(tensor, index)` /
`(obj, attr)` the caller names. Removing this discipline (an unconditional
`pop(slot)`) silently corrupted other holders' bindings when an intervening
relocation had already re-pointed the slot — see §11.8 / §12.

#### 7.3.1 The `attr` parameter on `bind_tree_node`

Originally `bind_tree_node` always wrote into `node.value`. The Mamba radix
tree stores a separate `node.mamba_value` (1-elem clone of
`req.mamba_pool_idx`) that also needs to be kept in sync. `_TreeNodeRef`
carries an `attr` field (default `"value"`), and the rewrite does
`getattr(node, attr)[pos] = dst` rather than hardcoding `node.value`. Each
sub-pool's binder is independent (`tree_node` dicts don't collide across the
full and mamba binders), so binding via `attr="mamba_value"` on the **mamba
binder** lets relocation of mamba slot ids reach the tree node's clone tensor.

#### 7.3.2 `bind_tree_node` aborts on `OVERWRITE`

A `tree_node` entry is 1:1 — a physical slot belongs to exactly one tree
node. If `bind_tree_node(slot, B, posB)` is called while `binder.tree_node[slot]`
already names a *different* node `A`, something laundered a slot that `A`
still owns into `B.value`. We do **not** silently replace the entry (that
would lose `A`'s binding and corrupt `A` on its next eviction); instead we
**abort** the new bind, keep `A`'s entry, and log `bind_tree_node OVERWRITE
ABORTED` with both caller stacks. Combined with the `_check_value_overlap`
pre-flight (which scans the incoming `value` tensor before any bind and warns
`VALUE-OVERLAP` / `VALUE-FREED-SLOT`, §7.7), this turns a silent corruption
into a loud, traceable diagnostic. Note: aborting the bind means `B.value`
holds a slot whose binder entry points at `A` — a residual inconsistency, but
strictly safer than the alternative and visible in the logs.

### 7.4 `Relocator.apply(sub_pool, src, dst)` — walkthrough

Called **synchronously from `MultiEndedAllocator._apply_relocations`**, which
`free()` calls right after the KV-buffer row copy (`move_kv_cache` /
`copy_from`). For each `(src[i], dst[i])` pair, in the boundary-inward order
`free` produced:

```python
def _apply_one(bt, src, dst, req_to_token):
    # Step 0 — pre-flight at dst. `dst` was just freed by the same
    #   _free_eager call, so `clear_slot(dst)` already dropped its binder
    #   entries. Pop any leftover anyway so step 6's re-bind can't
    #   silently overwrite.
    bt.tree_node.pop(dst); bt.req_position.pop(dst)
    bt.aux.pop(dst); bt.py_attr.pop(dst)

    # Step 1 — pop src's binder entries.
    tn, rp, ax, pa = (bt.tree_node.pop(src), bt.req_position.pop(src),
                      bt.aux.pop(src), bt.py_attr.pop(src))

    # Step 2 — tree_node: getattr(tn.node, tn.attr)[tn.position] = dst.

    # Step 3 — req_position: for each tracked row r in rp.rows, IF
    #   req_to_token[r, rp.col] == src, set it to dst. Rows whose cell is
    #   no longer src are stale (the row owner exited a non-cache-aware
    #   path); skipping them is correct — the rows set is a hint, the
    #   `== src` check is the safety net.

    # Step 4 — aux: for each _AuxRef(tensor, index): tensor[index] = dst.

    # Step 5 — py_attr: for each (obj, attr): setattr(obj, f"_{attr}",
    #   torch.tensor(dst, ...)).  (Private name dodges any property setter.)

    # Step 6 — re-bind src's entries under dst (the dicts at dst are empty
    #   by construction after Step 0).
    if tn: bt.tree_node[dst] = tn
    ...
```

Cost: **O(total number of relocations × tracked-rows-per-slot)** —
typically a handful of scalar writes per relocation. In the common
steady-state case (no relocations on a `free`), `apply` is never called.
There is no per-tick scan and no deferred work.

For `full_to_swa_index_mapping` specifically, `SharedSWATokenToKVPoolAllocator`
handles it **eagerly inside `_free_both` / `free_swa`** via
`_apply_new_relocations_to_mapping`, reading the suffix of
`MultiEndedAllocator._inverse_history` that the current `free` just produced:

- Full-side relocations: `mapping[dst] = mapping[src]; mapping[src] = 0`,
  in recorded order so chained moves resolve left-to-right.
- SWA-side relocations: build a chain-resolved `lookup[src] = terminal_dst`
  tensor and remap the mapping's values in one GPU op.

**Why a separate path and not the binder?** This predates the immediate-apply
refactor (the original failure: a later `_free_both` cleared `mapping[src]`
before the next-tick flush ran, so the deferred `mapping[dst] = mapping[src]`
copied a 0 and orphaned the swa slot — `swa_leaked=33` under skewed load on
gpt-oss-20b, §11.2). Even with immediate apply, the mapping needs the *value*
(swa id) at `mapping[src]` captured before any sibling `free` in the same
batch can clear it, so the eager in-`_free_both` handling is retained;
`_inverse_history` is the channel it reads.

### 7.5 The free-time `clear_slot` catch-all

Immediate apply closes the *deferred-flush drift* hazard that an earlier
revision of this doc spent a long section on (a slot id captured in an
unbound clone going stale across many ticks — see §11.2/§11.4/§11.6). But a
related invariant must still be enforced at `free()`:

> When `_free_eager` is about to overwrite a freed slot's bytes via eager
> compaction, **no binder entry may still name that slot**. Owners are
> supposed to have unbound it first (tree-side via `_unbind_*_value`;
> req-side via the row-aware unbinds; mamba-side via
> `transfer_mamba_to_radix` / `free_mamba_cache`). One path that *doesn't* —
> `_free_dedup_slots` — frees a request's fresh prefix slots without
> unbinding their `req_position`, so a freed slot can keep a `(col, {row})`
> entry; the *next* allocation of that slot would then trip `bind_req_position`'s
> multi-col assert.

So `_free_eager`, before doing any compaction, walks the freed slots and for
each one calls `SlotBacktrack.clear_slot(slot)` — pop the slot's `tree_node`
/ `req_position` / `aux` / `py_attr` entries. That's it: no warning, no
sentinel poisoning, no per-slot tensor writes. (An earlier revision also
poisoned every underlying holder with `-1` and emitted split-budget
diagnostics; that was removed once the laundering bugs were fixed — the per-
relocation device→CPU syncs it added dominated wall-clock on relocation-heavy
workloads, see §11.10. The plain `clear_slot` is cheap dict pops and is the
only part that was load-bearing.)

`clear_slot` is a backstop, not the workhorse: caller-side unbinds are still
correctness-load-bearing. If a caller frees a slot whose `tree_node` /
`aux` / `py_attr` was supposed to be unbound first, `clear_slot` drops the
binder entry, but the underlying holder (`node.value` cell, `req.prefix_indices`
cell, ...) still names the freed slot — and a later read of *that* will trip
the hard stale-slot assertion in `_free_eager`. That assertion's diagnostic
dump (binder state for the slot + recent `_inverse_history` + caller frames)
is the surviving "find the bug at the source" mechanism.

### 7.6 Binding-helper inventory (`base_prefix_cache.py`)

These helpers are the only callers of the binder methods outside the
pool/allocator infrastructure. They MUST be used at every assignment site
for the corresponding container; missing one lets a stale slot id survive
into the holder, which surfaces later as the hard stale-slot assertion
(§7.5 / the stale-slot dump in `_free_eager`).

| Helper | What it does | Sub-pool binder | Used at |
|---|---|---|---|
| `_set_node_value(node, new_value)` | match-aware unbind of `node.value`'s old slots → assign → bind new; runs `_check_value_overlap` pre-flight | full | `RadixCache._insert_helper`, `_split_node`, `MambaRadixCache._insert_helper` for the `value` field |
| `_bind_node_value(node)` | bind slots in `node.value`; `_check_value_overlap` pre-flight | full | After `_split_node` rewrites both halves |
| `_unbind_all_node_value(node) -> List[int]` | match-aware unbind of every `node.value` slot the binder agrees `node` owns; **returns those slot ids** | full | Tree-node eviction (then free only the returned ids), before `_split_node` rewrite |
| `_unbind_and_free_node_value(node) -> int` | `_unbind_all_node_value` then `free(returned ids)` only | full | `_evict_leaf_node`, `_iteratively_delete_tombstone_leaf`, plain `RadixCache` eviction loop |
| `_set_node_mamba_value(node, new_value)` | like `_set_node_value` for `node.mamba_value` | mamba | `MambaRadixCache._insert_helper` mamba field, tombstone revival |
| `_unbind_node_mamba_value(node) -> List[int]` | match-aware unbind, returns the unbound mamba slot id(s) | mamba | `_evict_leaf_node`, `evict_mamba`, before `_tombstone_internal_node`'s `mamba_value = None` |
| `_unbind_and_free_node_mamba_value(node) -> int` | unbind then `mamba_pool.free` only the returned ids | mamba | `_evict_leaf_node`, `evict_mamba` INTERNAL branch |
| `_safe_unbind_tree_node(...) -> List[int]` | the match-aware primitive the above use: returns `[slot]` if the binder agrees, `[]` (and a `DIVERGENCE`/`WRONG-NODE-AT-SLOT` warning) if it points elsewhere | full / mamba | internal |
| `_set_req_mamba_pool_idx(req, new_value)` | rebind `req.mamba_pool_idx` py-attr | mamba | radix cache assigns the attribute (cache-hit fork_from, etc.) |
| `_set_req_prefix_indices(req, new_value)` | cell-level rebind of every `req.prefix_indices` slot via `bind_aux` | full | all production assignment sites in `chunk_cache.py`, `mamba_radix_cache.py`, `radix_cache.py`, `swa_radix_cache.py`, `radix_cache_cpp.py`, `session_aware_cache.py`, `schedule_policy.py` |
| `_check_value_overlap(...)` | scan an about-to-be-bound `value` tensor for slots already bound to a different node (`VALUE-OVERLAP`) or already freed by the allocator (`VALUE-FREED-SLOT`); capped diagnostics, no behavior change | full / mamba | called from `_set_node_value` / `_bind_node_value` / `_set_node_mamba_value` |
| `_free_dedup_slots(value_seg, tree_value_seg)` | diff the two; free slots that differ AND aren't owned by a live tree node; skip (and warn) any slot already outside the allocator's range | full | `*RadixCache._insert_helper` dedup path |

Plus binder hooks fire automatically inside pool wrappers:

- `wrap_req_to_token_pool_write` — every `req_to_token_pool.write((row, col_or_slice), vals)`
  binds each written cell as `bind_req_position(slot, row, col)`.
- `wrap_req_to_token_pool_free` / `req_to_token_pool.free(req)` — zeroes the
  freed row's cells (so they can't be read back as stale slot ids) and
  drops the row's `req_position` bindings.
- `SharedHybridReqToTokenPool.alloc` — binds the freshly-assigned
  `req.mamba_pool_idx` (py-attr) and the
  `req_index_to_mamba_index_mapping[req_pool_idx]` aux entry.
- `SharedHybridReqToTokenPool.transfer_mamba_to_radix(req)` — on a mamba
  cache-finish where the tree took ownership of the slot, unbinds the req's
  py-attr + aux bindings and clears `req.mamba_pool_idx` **without freeing
  the slot** (ownership transfer, not a release).

For non-shared paths every helper resolves to the null binder and is a plain
assignment with no overhead.

### 7.7 Surviving correctness checks & diagnostics

The relocation subsystem went through a long debugging trail (§11). During
that trail a dense layer of *diagnostic tripwires* (`DOUBLE-ALLOCATION`,
`DST-NOT-EMPTY`, `ORPHAN-TENSOR`, `IN-PLACE-MUTATION`, `FIRST-BIND`,
`TREE-NODE-ORPHANED-ON-FREE`, and the `-1` poisoning) and a bag of
`SGLANG_SHARED_POOL_*_LOG` env knobs were added to localize each new failure
at the moment it happened. Once the laundering bugs were fixed (§11.9) they
were removed (§11.10) — they had served their purpose, several added a
device→CPU sync on the per-relocation hot path, and the env knobs no longer
exist. What remains is the small set of checks that are either *hard
correctness asserts* or *load-bearing aborts*, plus a couple of capped
warnings on cheap-to-check invariants:

| Check | Where | What it does |
|---|---|---|
| `stale-slot assertion` (hard) | `MultiEndedAllocator._free_eager` | `free()` was passed a slot outside the allocated range → dump binder state for the slot + recent `_inverse_history` + caller frames, then `raise`. The primary "find the bug at the source" mechanism. |
| `bind_req_position` multi-col (hard assert) | `SlotBacktrack.bind_req_position` | a slot is already known at a *different* column → a free path skipped the row-aware unbind / `clear_slot`. Should never fire. |
| `bind_tree_node` OVERWRITE-ABORT | `SlotBacktrack.bind_tree_node` | a slot is already bound to a *different* tree node → abort the new bind (keep the prior), log (capped count). A 1:1-invariant violation; should never fire. |
| `_free_dedup_slots` out-of-range skip | `BasePrefixCache._free_dedup_slots` | a slot about to be freed is already out of range → skip it (don't crash the watermark assert), log (capped). Defensive — should never fire after §11.9. |
| `_are_kv_indices_valid` / `_is_mamba_pool_idx_valid` aborts | `MambaRadixCache.cache_*_req` | the `req_to_token` row / `req.mamba_pool_idx` about to be cached holds a slot the allocator has already freed → refuse to launder it into the tree, skip/abort the cache for this req, log (capped). Backstop; should never fire after §11.9. |
| `_check_value_overlap` (`VALUE-OVERLAP` / `VALUE-FREED-SLOT`) | `BasePrefixCache._check_value_overlap` | pre-flight scan of a `value`/`mamba_value` tensor about to be bound — warn (capped, class-const cap) if any slot is owned by another node or already freed. Diagnostic only. |
| `_safe_unbind_tree_node` (`DIVERGENCE` / `WRONG-NODE-AT-SLOT`) | `BasePrefixCache._safe_unbind_tree_node` | an unbind names a slot whose binder entry points at a different node → pop the actually-bound slot(s) instead, warn (capped). |
| `bind_py_attr` MULTI-OWNER | `SlotBacktrack.bind_py_attr` | a *different* obj's `mamba_pool_idx` is being added while the slot already has live entries from another obj → warn (capped). Single-owner-invariant violation. |

A per-slot lifecycle ring buffer (`SlotBacktrack.history`, 16 events/slot,
always on, ~free) feeds the stale-slot assertion's dump.

---

## 8. Scheduler & model-runner integration

### 8.1 Pool gating in `_init_pools`

In `model_runner_kv_cache_mixin.py`, the existing branches for `is_hybrid_swa`
and `mambaish_config` gain a sibling that fires when
`server_args.enable_shared_memory_pool` is set:

```python
if self.is_hybrid_swa and enable_shared_memory_pool and not is_hybrid_swa_compress:
    shared_buffer = SharedMemoryPool(total_bytes=..., sub_pool_specs=[mha_spec, swa_spec], ...)
    self.token_to_kv_pool = SharedSWAKVPool(shared_buffer=shared_buffer, ...)
elif mambaish_config and enable_shared_memory_pool:
    shared_buffer = SharedMemoryPool(total_bytes=..., sub_pool_specs=[mha_spec, mamba_spec], ...)
    self.req_to_token_pool = SharedHybridReqToTokenPool(shared_buffer=shared_buffer, ...)
    shared_full_kv_pool = SharedMHATokenToKVPool(shared_buffer=shared_buffer, ...)
    self.token_to_kv_pool = HybridLinearKVPool(..., full_kv_pool=shared_full_kv_pool)
```

### 8.2 Allocator construction

A matching branch builds `SharedSWATokenToKVPoolAllocator` or
`SharedMambaTokenToKVPoolAllocator` (both composites over `MultiEndedAllocator`,
one per sub-pool), sharing one `Relocator`. The `Relocator` is wired with a
back-reference to the `ReqToTokenPool` (so `apply` can rewrite `req_to_token`
cells inline) and holds the `SlotBacktrack` for each sub-pool. The
`req_to_token_pool.write` wrapper and (for the Mamba path)
`SharedHybridReqToTokenPool` get the binder facade so writes register
`req_position` / `py_attr` / `aux` bindings.

### 8.3 No per-tick flush hook — immediate apply

The earlier design had a `flush_relocations()` scheduler hook called from
`run_batch` before each forward, which walked a per-tick `RelocationLog`.
**That hook is gone.** Under the immediate-apply design `free()` itself calls
`Relocator.apply` synchronously, so by the time `free()` returns, every bound
holder already names the post-compaction slot id — there is nothing to flush.
`scheduler.py` no longer calls `flush_relocations`, and the composite
allocators no longer expose it.

The one scheduler-visible interaction that remains is **spec-decode rollback**:
`MultiEndedAllocator.backup_state` / `restore_state` snapshot/replay against
the per-allocator `_inverse_history` (the `List[(src_tensor, dst_tensor)]`
batches `_apply_relocations` appended), replaying the inverse data moves and
the inverse `Relocator.apply(dst, src)` to undo the reference rewrites. The
composite allocators call `clear_inverse_history` after consuming the relevant
suffix (e.g. the SWA-mapping update) so the history doesn't grow unbounded.

### 8.4 Why ordering relative to `prepare_for_extend` no longer matters

The old design's most delicate property was that `prepare_for_extend` (which
*reads* `req.prefix_indices` and *writes* `req_to_token`) ran in the same
tick *before* the flush, so a stale id read from an unbound clone could only
be self-corrected if the relocation log still held an entry for it — and a
multi-tick-old relocation had already been flushed and cleared (the
`prefix_indices` drift class, §11.4). With immediate apply there is no
window: any relocation that ever happened has *already* rewritten every bound
holder. The remaining requirement is simply **bind every persistent holder**
(§7.6) so the relocation reaches it the moment it happens, and **re-validate
clones that escape the bound containers** before reusing them (the
`_insert_helper` `value` clone — §11.9). An *unbound* holder that goes stale
is still hazardous, but it can no longer silently "drift N relocations behind";
the first stale read trips the hard stale-slot assertion with full context.

---


## 9. PD disaggregation compatibility (NOT YET SUPPORTED)

`server_args.py` currently asserts
`enable_shared_memory_pool` is incompatible with `disaggregation_mode in
{"prefill", "decode"}`. This is conservative — the design *can* support
disaggregation, but two specific changes are needed first.

### 9.1 Gap 0 — RDMA byte-arithmetic mismatch (the binding gap)

The RDMA transfer protocol assumes **layer-monolithic** memory layout:

| | Non-shared (`MHATokenToKVPool`) | Shared pool (`SharedMHATokenToKVPool`) |
|---|---|---|
| Per-layer K storage | One contiguous tensor `(size, head_num, head_dim)` | Strided view into one shared buffer |
| Slot stride within K-tensor for layer L | `head_num · head_dim · itemsize` (one row) | `entry_bytes = layer_num · (k_row + v_row)` (whole per-slot envelope) |
| `k_buffer[L][0].nbytes` | Same as slot stride | `head_num · head_dim · itemsize` ≪ slot stride |

Mooncake's transfer formula
([conn.py:648](../../python/sglang/srt/disaggregation/mooncake/conn.py#L648))
is `dst_addr = dst_kv_ptrs[L] + slot * item_lens[L]`. Under non-shared,
`item_lens[L]` and slot stride are the same; under shared they differ by
`2 · layer_num`. The inherited `MHATokenToKVPool.get_contiguous_buf_infos`
returns `item_lens[L] = k_row` — **wrong for the shared layout**: a write
to slot 100 of layer 0 lands `(100 · k_row)` bytes from the layer base,
but slot 100's actual K data is at `(100 · entry_bytes)` bytes — silent
corruption.

**Fix shape.** Override `get_contiguous_buf_infos` /
`get_state_buf_infos` on the Shared* pools to return a **single fused
entry per sub-pool** whose `item_len = entry_bytes`. The RDMA
transfer engine then writes the entire per-slot envelope (all layers' K +
V interleaved) in one op per slot — strictly more efficient than the
per-layer transfer the inherited path would attempt. See the disagg-
compatibility plan in
[plans/in-the-current-implementation-composed-ullman.md](../../../.claude/plans/in-the-current-implementation-composed-ullman.md)
for the full proposal.

**Limitation in v1 of disagg support.** Heterogeneous-TP transfers
(`send_kvcache_slice`) need per-layer head-slicing within the per-slot
envelope, which can't be expressed by a single `(ptr, item_len)` pair.
v1 would assert prefill TP == decode TP when shared pool is enabled.

### 9.2 Gap 1 — Eager compaction during the RDMA transfer window

Decode pre-allocates dst slots, sends them to prefill via `send_metadata`,
then prefill RDMA-writes data into those slot byte ranges. If decode runs
`free()` on an unrelated request during the transfer window, eager
compaction may relocate the in-flight slot. The RDMA write then lands at
the OLD address — the relocated data, plus whatever the binder updated to
the NEW slot, both end up wrong.

**Fix shape.** Add a slot-pinning primitive on `MultiEndedAllocator`:

```python
allocator.pin_slots(indices)    # called at decode send_metadata / prefill
                                # forward-end
allocator.unpin_slots(indices)  # called at KVPoll.Success
```

`free()` either skips pinned slots entirely (allowing temporary holes
near the boundary) or pauses compaction while any pin is active. The
plan document linked above evaluates both designs; the lazy-compaction
variant (a redesign of `free` that defers all data movement to an
explicit `compact()` method) is the cleaner long-term answer because
it makes `free()` a no-op for in-flight slots regardless of pinning.

### 9.3 What already works

- `isinstance` polymorphism: `SharedSWAKVPool(SWAKVPool)` and
  `SharedHybridReqToTokenPool(HybridReqToTokenPool)` keep
  `state_type` detection in the disagg KVManager working unchanged.
- `min_slot_index` keeps slot-0 padding writes harmless (real data
  lives at bytes ≥ `entry_max`, well outside the transfer envelope's
  `dst_ptr + 0 * item_len` zone).
- Allocator-coordinated peer growth prevents the peer pool from
  allocating into bytes belonging to an in-flight slot — that's
  enforced by the existing byte-frontier check.
- Immediate `Relocator.apply` means there is no deferred-flush window
  during which a slot id could be "in transit"; the disagg gap is purely
  about *physical byte movement* (eager compaction relocating a slot whose
  bytes an RDMA transfer is mid-write to), §9.2 — not about reference
  bookkeeping lag.

---

## 10. Future extensions

- **N > 2 sub-pools** — e.g. promote `intermediate_ssm` /
  `intermediate_conv_window` into a third sub-pool over a second shared
  buffer (PDF "Potential Solution II"). `SubPoolSpec` is a list; the
  `MultiEndedAllocator` peer list design supports it.
- **`page_size > 1`** — paged variants of `MultiEndedAllocator` that track
  page frontiers instead of slot frontiers.
- **Disaggregation** — see §9.
- **Lazy compaction redesign** — replace `free()`'s eager compaction with
  bookkeeping-only free + an explicit `compact()` triggered before peer
  eviction. Strictly more efficient in the steady state (zero data moves
  on free) and makes the disagg pinning story trivial. Discussed in the
  plan file.
- **C++ tree integration** — if needed, add binder-callable hooks on the C++
  side; alternatively, keep a Python "shadow tree" for bookkeeping only.
- **Validation mode** — an env-guarded full sweep that, at idle, verifies
  every `req_to_token` cell, every `TreeNode.value`/`mamba_value`, every
  mapping cell and every `Req.prefix_indices` clone references a slot the
  allocator still considers allocated, AND that every allocated slot is
  referenced. The §7.7 tripwires are point checks at write/free/alloc time;
  this would be the periodic global cross-check that catches a *new*
  unbound-holder bug the moment it's introduced (rather than at the eval
  round that happens to exercise it).

---

## 11. Implementation history: bugs found & fixes applied

The shared-memory-pool feature shipped in stages. This section records
the non-obvious bugs caught after the initial v1 ship — each one revealed
a subtle interaction between the relocation system and an unbound
container that "looked transient" but actually persisted across forwards.

### 11.1 v1 ship: anchor-shift slot-0 clobbering

**Symptom (caught pre-merge).** With anchors at carefully-aligned bytes,
a smaller-entry pool's slot 0 (at byte 0) overlapped a larger-entry
pool's slot 0's dummy-write region.

**Fix.** Adopted the `min_slot_index` scheme (§5.6) — every pool's
real data starts at bytes ≥ `entry_max`, all anchors stay 0, and the
math falls out uniformly for any N. Cleaner, scalable, and the per-pool
overhead is at most one slot-equivalent of bytes.

### 11.2 SWA mapping race in `_free_both`

**Symptom.** Under skewed bench load on gpt-oss-20b: `swa_leaked=33`
intermittent leak detection.

**Cause.** A later `_free_both` / `free_swa` in the same batch cleared
`mapping[src]` to 0 before the next-tick flush ran. When flush later
applied `(src → dst)` with `mapping[dst] = mapping[src]`, it copied a 0
in place of the original swa pair, orphaning the swa slot.

**Fix.** Made the SWA mapping update **eager**: capture
`mapping[src]` at the moment compaction records the relocation (inside
`_free_both` / `free_swa`), and apply it to `mapping[dst]` immediately —
NOT via the relocation-log flush path. The relocation log still records
the move so the `req_to_token` and tree-node updates flow through flush
normally. See `_apply_new_relocations_to_mapping` in
[multi_ended_allocator.py](../../python/sglang/srt/mem_cache/multi_ended_allocator.py).

### 11.3 Mamba `evict_mamba` assertion: unbound `TreeNode.mamba_value`

**Symptom.** First eval run on `tiiuae/Falcon-H1-7B-Instruct`: 4 cells
crashed with
`AssertionError: MultiEndedAllocator.free(mamba): slot K is outside
allocated range [low, high)` from `MambaRadixCache.evict_mamba` →
`mamba_pool.free(x.mamba_value)`.

**Cause.** `_insert_helper` set
`mamba_value = req.mamba_pool_idx.unsqueeze(-1).clone()` and stored the
clone on the tree node. The clone is NOT the same Python object as
`req.mamba_pool_idx`, so it wasn't covered by the existing
`bind_py_attr` binding. When eager compaction relocated the mamba slot,
the clone was left holding the stale id.

**Fix.** Generalized `bind_tree_node` with an optional `attr` parameter
(see §7.3.1). `_TreeNodeRef` carries `attr` (default `"value"`); flush
writes via `getattr(node, attr)[pos] = dst`. Added
`_set_node_mamba_value` / `_unbind_node_mamba_value` helpers and routed
all `node.mamba_value = ...` and pre-free unbinds in
`mamba_radix_cache.py` through them.

**Result.** Zero `MultiEndedAllocator.free(mamba)` assertions in the
second eval run.

### 11.4 SWA leaks + full-pool assertion: unbound `Req.prefix_indices`

**Symptom.** Second eval run: 8 cells (4 SWA + 4 Mamba) all crashed,
some with `swa_leaked=N` (consistent N=8 or 9), others with
`AssertionError: MultiEndedAllocator.free(full): slot K is outside
allocated range [low, high)`. The diagnostic explicitly named
"req.prefix_indices" as the prime suspect.

**Cause.** `req.prefix_indices` is set as a *clone* of `kv_indices`
(or a `torch.cat` of slices) at radix-cache lookup / cache-finish time.
The clone persists across many ticks while a chunked req sits in the
waiting queue. Each tick's flush updates the *bound* references but not
this clone (no binding existed for it). When the req is finally picked
back, `prepare_for_extend` reads the stale clone, writes stale ids into
`req_to_token`, and the next tick's flush has no entry that touches
those cells (the relocations were applied and cleared many ticks ago —
see §7.5).

The SWA leak follows from the same cause: `_free_both` reads
`swa_indices = full_to_swa_mapping[stale_full_slot]`. With intervening
relocations, `mapping[stale_full_slot]` no longer points at this req's
swa slot — so the actual swa slot never gets freed.

**Fix.** Added `_set_req_prefix_indices(req, new_value)` helper in
`base_prefix_cache.py`. It uses cell-level `bind_aux(slot, tensor, pos)`
so each cell of `req.prefix_indices` is bound to its position; on flush
the binder writes `prefix_indices[pos] = dst` per relocation, keeping
the clone in sync. Routed all 11 production assignment sites through
the helper (see §7.6 for the table).

For non-shared paths, the helper resolves to the null binder and
becomes a plain assignment.

### 11.5 The deferred→immediate refactor (`RelocationLog` → `Relocator`)

**Why.** §11.4's `prefix_indices` fix worked, but the *class* of bug it
addressed kept recurring in new disguises across many eval rounds: any holder
of a slot id that wasn't bound, or was bound at a stale `(node, pos)` /
`(row, col)`, would drift while the per-tick `RelocationLog` cleared itself.
Two recurring concrete failures:

1. **Stale-slot asserts in `MambaRadixCache._insert_helper`.** The helper
   captures `value[:prefix_len]` *before* a `_free_dedup_slots` call; the free
   triggers eager compaction whose binder updates were *deferred*; the local
   Python clone `value` kept the pre-compaction ids and fed them back to the
   allocator. The pre-refactor workaround (`clear_pending=False` + a
   `resolve_to_current` chain-chase) was load-bearing and fragile.
2. **Multi-column `bind_req_position` "OVERWRITE" warnings** — `req_position`
   stored only the column; a slot reused at a different column left the prior
   req's `req_to_token` cell stale (see §11.7).

**What changed.** `RelocationLog` (+ `_pending`, `record`, `flush`,
`is_empty`, `checkpoint`/`rollback`, `resolve_to_current`) was replaced by
`Relocator` with a synchronous `apply(sub_pool, src, dst)` invoked from
`free()` (§7.4); the scheduler `flush_relocations` hook was deleted (§8.3);
`req_position` became row-tracked (§7.2/§11.7); spec-decode rollback moved to
a per-allocator `_inverse_history` (§8.3); and `_free_eager` gained the
`clear_slot` catch-all (§7.5). After the refactor: no `_pending`, no per-tick
flush, no chain-chasing, no `clear_pending=False`. (A dense layer of
diagnostic tripwires + `-1` poisoning was also added during the post-refactor
debugging; it was removed once the laundering bugs were fixed — §11.10.)

### 11.6 Mamba slot ownership: tombstones, `transfer_mamba_to_radix`, OVERWRITE-abort

**Symptoms (multiple eval rounds).** `mamba_pool.free([stale_id])` /
`mamba_pool.free([-1])` from `MambaRadixCache._evict_leaf_node` /
`evict_mamba`; `WRONG-NODE-AT-SLOT` cascades; `bind_tree_node OVERWRITE`
storms (cum > 100K) on `node.value` and `node.mamba_value`.

**Causes & fixes (cumulative):**
- `cache_finished_req(mamba_exist=False)` bound `req.mamba_pool_idx`'s slot to
  a tree node but left the req's py-attr/aux bindings pointing at it; a later
  tree evict freed the slot while the req still held a dangling handle to it,
  which then laundered back into a fresh `node.mamba_value`. **Fix:**
  `transfer_mamba_to_radix(req)` — hand ownership to the tree cleanly (unbind
  the req's bindings, clear `req.mamba_pool_idx`, **don't free**).
- `evict_mamba`'s INTERNAL branch tombstones internal nodes (`mamba_value =
  None`); `_tombstone_internal_node` redundantly re-unbound the mamba slot
  *after* the caller already did, and after `mamba_pool.free`'s eager-
  compaction had rebound that physical slot to a *different* node — corrupting
  it (`WRONG-NODE` cascade). **Fix:** removed the redundant unbind; made every
  `unbind_*` match-aware (§7.3 / `_safe_unbind_tree_node`).
- `_unbind_all_node_value` (and the mamba analogue) used an unconditional
  `pop(slot)`; called twice across an intervening apply-rebind it silently
  dropped *another* node's binding. **Fix:** match-aware unbind that *returns*
  the slot ids the binder agreed `node` owned; callers (`_unbind_and_free_*`)
  free **only** those, never the full `node.value` tensor — passing the full
  tensor advanced the watermark past slots still bound to other nodes (the
  eval_42 multi-col cascade).
- `bind_tree_node` *replacing* a `tree_node` entry silently lost the prior
  node's binding. **Fix:** abort on OVERWRITE, keep the prior entry, log both
  caller stacks (§7.3.2); added the `_check_value_overlap` pre-flight.
- Defensive backstops in `cache_finished_req` / `cache_unfinished_req`: refuse
  to launder a stale `req.mamba_pool_idx` into the tree (`_is_mamba_pool_idx_valid`),
  and `req_to_token_pool.free(req)` now zeroes the freed row.

**Result.** Mamba-pool cascades stopped recurring (no Mamba-pool crashes since
eval_42); the tp1/mfs0.85 and tp2/mfs0.85 Mamba cells run clean.

### 11.7 Row-tracked `req_position` (the multi-column assert)

**Symptom.** `bind_req_position multi-col for slot S: prior col=C0 (rows=[…]),
new (row=R, col=C1)` — and downstream, `req_to_token` cells holding freed slot
ids that crash later on `free()`.

**Cause.** `req_position` originally stored only `(col, rows)`. Under
prefix-sharing a physical slot is at the *same* column in every req that holds
it (an invariant) — but when a slot was freed and re-allocated to a different
req at a different column, the old entry survived and the prior req's
`req_to_token[row, C0]` cell was never rewritten/cleared, so it kept naming a
freed slot.

**Fix.** `_ReqPosRef(col, rows: Set[int])`. `bind_req_position(slot, row, col)`
asserts the column matches if the slot is already known (a different column ⇒
a free path skipped the row-aware unbind / `clear_slot` — a hard error, not a
tolerable race), and tracks the row set. `Relocator.apply` rewrites only the
tracked rows (with a `== src` verify), and the free-time `clear_slot` catch-all
drops the `_ReqPosRef` when the slot is freed. The multi-column representation
that made the bug possible no longer exists. Plumbed `row` into the two
`bind_req_position` call sites (`wrap_req_to_token_pool_write`, the Triton
`write_req_to_token_pool` manual fire); `req_to_token_pool.free(req)` now
zeroes the freed row and drops its `req_position` bindings.

### 11.8 Full-pool slots laundered into the radix tree (the `DOUBLE-ALLOCATION` saga)

**Symptom (eval_44–45, mfs=0.55 skewed-large, Mamba).**
`bind_req_position multi-col` and stale-slot crashes inside `_free_dedup_slots`.
A temporary `DOUBLE-ALLOCATION` tripwire (a cap-gated check in
`MultiEndedAllocator.alloc` that probes the binder for a just-handed-out slot,
since removed) caught the precise shape: `alloc()` handed out a contiguous run
of ~32 full-pool slots that were *still* bound — at consecutive positions — to
one radix `TreeNode.value` of a long streaming request, *and* to that
request's `req_to_token` row. I.e. that request's KV slots got into the tree
via `cache_unfinished_req` and *also* got freed and re-allocated while the
tree node still referenced them.

This was diagnosed in two stages. eval_44–45 added instrumentation to localize
it: `_free_dedup_slots` now *skips* (rather than crashes on) a slot already out
of range; `_check_value_overlap` got a `VALUE-FREED-SLOT` arm flagging a tree
node about to be bound to an already-freed slot, which pinpointed the
laundering *bind*; `cache_*_req` got the `_are_kv_indices_valid` backstop that
refuses to cache a `req_to_token` row containing freed ids. The first two were
diagnostic; the `_free_dedup_slots` skip and the `_are_kv_indices_valid`
backstop stayed. The actual root cause and fix is **§11.9**.

### 11.9 Laundering via the early `req_to_token` snapshot in `cache_unfinished_req`

> This is the root cause behind §11.8. Fixed in eval_46; verified clean in
> eval_47.

**The bug.** `MambaRadixCache.cache_unfinished_req` materialized the int64
`page_aligned_kv_indices` copy *near the top* of the function, then did
`mamba_pool.fork_from` and — if the (tiny, ~285-slot) mamba pool was full —
`self.evict(EvictParams(mamba_num=1))`, *then* called `insert()`. When that
`evict()` evicted a tree leaf, `_unbind_and_free_node_value` freed the leaf's
full-pool slots, which triggered eager compaction; eager compaction relocates
the *boundary* slots — and the boundary slots are precisely this request's
freshly-allocated suffix (it just did a multi-thousand-token prefill). `Relocator.apply`
rewrites `req_to_token` (bound) and the tree nodes (bound) — but **not** the
int64 copy, which isn't bound yet (`insert()` only `bind_value_as_aux`'s it a
few lines later). By the time it *is* bound, the copy's tail names slots that
are now above the watermark = freed. `_insert_helper` then `_set_node_value(new_node,
value.clone())` with that stale tail — the laundering moment. The aux binding
that *was* added earlier (`insert()`'s `_bind_value_as_aux` / `_unbind_value_as_aux`,
§7.6) only covers the `_insert_helper` walk; it doesn't cover the `fork_from` /
`evict` window before `insert()`. (`cache_finished_req` materialized its copy
*after* its frees, so it was never bitten — but its prefill-path caller has no
outer `free_group`, so its `_insert_helper` dedup-frees still relocated mid-walk.)

**Fix (eval_46).**
1. **Late snapshot.** `cache_unfinished_req` keeps a live `int32` *view* of
   `req_to_token` through `fork_from` / `evict` and materializes the int64 copy
   *immediately before* `insert()` — at that point `req_to_token` has already
   absorbed any `evict`-driven relocations.
2. **`free_group` wrapper.** The `insert()` + post-processing block in
   `cache_unfinished_req` / `cache_finished_req` runs inside
   `alloc.free_group_begin()` / `free_group_end()` — guarded by an
   `own_free_group` flag so only the *outermost* opener manages the group
   (the decode result path already wraps `release_kv_cache` in one; nesting
   would reset `free_group = []` and lose the outer batch). This defers the
   `_insert_helper` dedup-frees' eager compaction until after the new tree
   node is bound and `req_to_token` / `req.prefix_indices` are rewritten, so
   the in-flight `value` clone can't go stale while it's live.
3. SWA's `cache_*_req` got the same `free_group` wrapper for parity (it never
   had the late-snapshot bug — no `fork_from`/`evict` between snapshot and
   `insert` — but the wrapper is belt-and-suspenders).

**Result.** eval_46 ran the full Mamba matrix crash-free for the first time
(incl. the previously-fatal mfs=0.55 skewed-large cells); eval_47 added SWA and
also passed clean. Generalised lesson: a slot-id *clone* that escapes the bound
containers must be materialized as late as possible and re-validated before
reuse (§12 F3).

### 11.10 Removing the diagnostic instrumentation — and the perf win

Once §11.9 closed the laundering, the dense diagnostic layer accumulated during
the debugging trail was removed: the `DOUBLE-ALLOCATION` alloc-time probe
(`_check_alloc_not_bound`), the `DST-NOT-EMPTY` / `ORPHAN-TENSOR` /
`IN-PLACE-MUTATION` checks in `Relocator._apply_one`, the `FIRST-BIND` bind
tracing, the `TREE-NODE-ORPHANED-ON-FREE` / `free-with-refs` warnings, the
`-1` poisoning, and every `SGLANG_SHARED_POOL_*_LOG` env knob (`*_MAPPING_ASSERT`
stays). The free-time catch-all collapsed from `poison_and_clear_slot` (pop +
poison every holder + capped warning) to a plain `SlotBacktrack.clear_slot`
(pop the binder dicts). Kept: the hard stale-slot assertion, the
`bind_req_position` / `bind_tree_node`-overwrite asserts, the `_free_dedup_slots`
out-of-range skip, the `_are_kv_indices_valid` / `_is_mamba_pool_idx_valid`
aborts, and `_check_value_overlap` / `_safe_unbind_tree_node` warnings (§7.7).

This was also, unexpectedly, a major **performance fix**. The removed
instrumentation included *per-relocation device→CPU syncs* — `_check_tree_node_inplace_mutation`'s
`target[position].item()` and the orphan-tensor weakref check, both running on
*every* relocated slot — plus `poison_and_clear_slot`'s per-freed-slot CUDA
reads + masked writes. On the relocation-heavy skewed workloads (one giant
streaming request whose boundary slots are relocated on essentially every
`free()`), those dominated wall-clock. Median inter-token latency on the
Mamba shared path, eval_46 → eval_47: tp1/mfs0.55 skewed `15 592 ms → 388 ms`,
tp1/mfs0.55 skewed-large `14 061 ms → 245 ms`, tp1/mfs0.85 skewed `1 133 ms → 71 ms`,
tp2/mfs0.85 skewed `3 803 ms → 89 ms` — a 7–57× speedup, putting the skewed
workloads back in the "slower than baseline but usable" range (§12 O1 for the
current baseline vs shared numbers).

### 11.11 Lessons learned

1. **"Transient" is a load-bearing claim.** Any container that *outlives a
   single forward* — `req.prefix_indices`, the local `value` clone in
   `_insert_helper`, an int64 copy that's not bound yet — must be bound or
   re-validated against the allocator before reuse. The original §3.3 line
   calling `prefix_indices` "short-lived" was wrong and is corrected.
2. **Materialize escaped clones late.** A clone that leaves the bound
   containers (the `cache_unfinished_req` int64 copy) must be taken *after*
   every operation that could relocate the slots it names (`fork_from` /
   `evict`), and the work that consumes it should run inside a `free_group`
   so its own frees don't relocate mid-use.
3. **Boundary frees are silent.** A slot leaving the watermark via the
   `s == b` path produces no relocation pair — correct for bound holders, but
   an *unbound* holder of it is silently abandoned and surfaces later as the
   hard stale-slot assertion.
4. **Don't free more than you own.** The single most damaging recurring
   mistake was an eviction/abort path freeing `node.value` (or a `req_to_token`
   slice) wholesale when only *some* of those slots actually belonged to the
   freer. Free exactly the slots the binder agrees you own; skip the rest.
5. **Match-aware everything.** Unbinds and rebinds must be keyed by the full
   identity `(node, pos, attr)` / `(tensor, idx)` / `(obj, attr)` / `(row,
   col)`, never by slot id alone — an intervening relocation can have re-pointed
   the slot to a different owner since the caller last looked.
6. **Diagnostics belong off the hot path.** The instrumentation that found the
   bugs (§11.8/§11.9) was right to add and right to remove. A diagnostic that
   does a CUDA sync per relocation isn't free; once it's served its purpose,
   delete it (it cost ~40–200× on the worst workload — §11.10).

---

## 12. Critical design flaws & open issues

This section is the honest "what's still wrong / fragile" list, distilled from
§11. Read it before extending the feature.

### 12.1 Design flaws (inherent to the current approach)

**F1 — Reference holders are tracked by enumeration, not by construction.**
The whole `SlotBacktrack` machinery exists because slot ids are *copied* into
many independent containers (`req_to_token`, `TreeNode.value`/`mamba_value`,
two mapping tensors, `Req.mamba_pool_idx`, `Req.prefix_indices`, plus transient
clones). Every new container that persists a slot id is a new place to forget
to bind. There is no compiler/type-system help; a missing `bind_*` is invisible
until the eval round that exercises it. *Mitigation in place:* the hard
stale-slot assertion (§7.7) turns "invisible until much later" into "crash
with the binder dump and caller frames at the next free of the stale slot".
*Real fix (not done):* the "validation mode" full sweep (§10) run periodically
in CI/eval, and/or wrapping slot ids in a handle type whose copy is a binder
op.

**F2 — Eager compaction couples *physical byte movement* to *every* free.**
`free()` doing data copies + reference rewrites synchronously is what makes the
reference bookkeeping tractable (immediate apply), but it also means: (a) a
free of one request can move another request's slots (fine for bound holders,
fatal for unbound ones — F1); (b) the steady-state cost of `free` is
non-trivial on relocation-heavy workloads; (c) it's incompatible with
in-flight RDMA transfers (§9.2). *Real fix (not done):* the lazy-compaction
redesign (§10) — `free()` becomes bookkeeping-only, an explicit `compact()`
runs only when the peer actually needs the bytes. This also makes pinning
trivial and shrinks the "no live reference may name a slot whose bytes are
about to move" window to the `compact()` call.

**F3 — Radix-cache clones slot ids out of the bound containers.**
`MambaRadixCache._insert_helper` (and the SWA/plain analogues) take
`value = page_aligned_kv_indices` (an int64 copy of `req_to_token[row, :]`),
walk the tree, free a dedup'd subset, and bind the rest into new nodes. That
copy is *not* a bound container, so anything that relocates the slots it names
— while it's live but not yet `bind_value_as_aux`'d — leaves it stale, and the
stale tail gets laundered into a new tree node. §11.9 closed the two concrete
windows (`cache_unfinished_req`'s `fork_from`/`evict` window via the late
snapshot; the `_insert_helper` dedup-free window via the `free_group` wrapper),
and the `_are_kv_indices_valid` / `_free_dedup_slots`-out-of-range backstops
catch any residue. *Real fix (not done):* either resolve the clone through the
binder at use-time (use the live `req_to_token` view, not an int64 copy), or
make the dedup-free not relocate (lazy compaction, F2). The current code is
correct but relies on every `cache_*_req` path keeping its copy in the
"materialize late + run inside a `free_group`" discipline.

**F4 — The free-time `clear_slot` catch-all is a backstop, not a safety net.**
`_free_eager` pops every freed slot's binder entries — that's load-bearing for
the one path (`_free_dedup_slots`) that frees without unbinding `req_position`.
But it only clears the *binder*; if a caller also leaves a *holder*
(`node.value` cell, `req.prefix_indices` cell, a `req.mamba_pool_idx` attr)
naming the freed slot, `clear_slot` doesn't fix that — the next read of that
holder trips the hard stale-slot assertion. So the right number of "caller
forgot to unbind a holder" events is still zero; `clear_slot` just keeps the
*binder* consistent so the failure surfaces as a clean assertion rather than a
silent multi-col re-allocation.

### 12.2 Open issues

**O1 — Performance on relocation-heavy workloads.** On the skewed/skewed-large
workloads (64 concurrent multi-thousand-token requests overcommitting the KV
pool), shared-pool median ITL is slower than the static-partition baseline:
Mamba ≈ +18–89%, SWA ≈ +220–620% (eval_47). The SWA gap is the larger because
every `free()` on the SWA path additionally rewrites `full_to_swa_index_mapping`
(O(mapping_size) GPU work per swa-side relocation). This is the inherent F2
cost (`free()` does per-layer `move_kv_cache` + per-relocation `Relocator.apply`
bookkeeping). It is *not* a crash, and it improved ~7–57× when the per-relocation
diagnostic syncs were removed (§11.10) — balanced workloads are at parity. The
structural fix is the lazy-compaction redesign (F2/§10).

**O2 — PD disaggregation: still asserted off.** §9 — two concrete changes
(fused-entry `get_contiguous_buf_infos`, slot pinning during the RDMA window)
are required before the assert can be lifted; heterogeneous-TP transfers need
more.

**O3 — `page_size > 1` and N > 2 sub-pools are unimplemented** (§10) — the
allocator's peer-list and `SubPoolSpec` list are *designed* for them but no
code path constructs them.

*(Resolved: the "full-pool slots laundered into the radix tree" crash on
Mamba+low-`mfs`+skewed-large — root-caused and fixed in §11.9; eval_47 ran the
full Mamba+SWA matrix crash-free.)*

### 12.3 Where to look first when a new failure appears

1. **The hard `stale-slot assertion` dump** (`MultiEndedAllocator._free_eager`)
   is the primary forensic tool — it includes binder state for the offending
   slot, recent `_inverse_history`, the free-batch shape, and caller frames.
   Read it whole.
2. Grep the server log for the surviving warning prefixes — `VALUE-FREED-SLOT`,
   `VALUE-OVERLAP`, `WRONG-NODE-AT-SLOT`, `DIVERGENCE`, `STALE` (kv-indices /
   `req.mamba_pool_idx`), `OUTSIDE the allocator`, `OVERWRITE ABORTED`,
   `MULTI-OWNER`. Each carries a caller stack and a capped count.
3. If the crash is in `bind_req_position` / `bind_tree_node` / a `node.value`
   write — it's a *laundering* bug (F1/F3): something put a stale id into a
   container. Trace it back from the bind, not from the free; in particular
   check whether some `cache_*_req` / `_insert_helper` path materialized a
   slot-id copy before an operation that could relocate (§11.9).
4. If reproducing under stress is hard, temporarily re-add a per-relocation
   `assert getattr(tn.node, tn.attr).flatten()[tn.position].item() == src` in
   `Relocator._apply_one` step 2 (this is essentially the old IN-PLACE-MUTATION
   check) — it pins the corruption to the relocation that crosses it. Remove it
   afterward; it's a CUDA sync per relocation (see §11.10 for why that matters).

---

## 13. Glossary

| Term | Meaning |
|---|---|
| **Slot**          | Unit of allocation in a memory pool; holds KV (or Mamba state) for one "token" (MHA) or "request" (Mamba) across all layers. |
| **Sub-pool**      | One of the two (or more) logical pools sharing a single physical byte buffer. |
| **Byte frontier** | The highest (or lowest) byte offset currently occupied by one allocator's live slots. |
| **Watermark**     | The allocator's next-to-allocate slot id (grow-up: one past last allocated; grow-down: one below last allocated). |
| **Eager compaction** | In `free`, relocate the boundary slot into any freed interior hole so the allocated range stays contiguous. |
| **Boundary free** | The case `s == b` (the slot being freed equals the current boundary). The watermark just shrinks; **no relocation occurs** and `Relocator.apply` is not invoked. Hazardous for *unbound* holders of `s` — see §6.3.1, §12 F1. |
| **SlotBacktrack** | Per-sub-pool reverse index: `slot id → reference holder(s)`. Entries are `_TreeNodeRef(node, pos, attr, …)`, `_ReqPosRef(col, rows)`, `[_AuxRef(tensor, idx)]`, `[(obj, attr)]`. |
| **SlotBacktrackBinder** | Facade that routes slot-id writes/unbinds into a SlotBacktrack; `NullBacktrackBinder` is the no-op for static-partition paths. |
| **Relocator** | Owns the `SlotBacktrack`s + a `ReqToTokenPool` back-ref + per-allocator `_inverse_history`. `apply(sub_pool, src, dst)` rewrites every bound reference **synchronously** when `free()` relocates a slot; `SlotBacktrack.clear_slot` is the free-time catch-all. Replaced the old per-tick `RelocationLog`/`flush` (§11.5). |
| **Bound container** | A reference holder registered with the SlotBacktrack (via `bind_tree_node` / `bind_aux` / `bind_py_attr` / `bind_req_position`) so `Relocator.apply` rewrites it the moment a slot it holds is relocated. |
| **Unbound container** | Any Python container holding a slot id that is NOT registered with the binder (or is bound at a stale `(node,pos)` / `(row,col)`). The recurring root cause of the bugs in §11; now surfaces as the hard stale-slot assertion (with the binder dump) rather than silent drift. |
| **Binding helper** | The `_set_*` / `_unbind_*` / `_check_*` methods on `BasePrefixCache` that wrap raw assignments to bound containers (match-aware). Code MUST go through them — direct `node.value = X` or `req.prefix_indices = X` bypasses the binder. |
| **`clear_slot` catch-all** | On every `free()`, `_free_eager` calls `SlotBacktrack.clear_slot(s)` for each freed slot — pops its `tree_node`/`req_position`/`aux`/`py_attr` binder entries (cheap dict pops; no poisoning). Load-bearing for the `_free_dedup_slots`/`req_position` case (§7.5). |
| **Direct-peer check** | `alloc`'s fit check: compare byte frontier against the *immediate* neighbor, not a global sum over all peers. |
| **`min_slot_index`** | Per-sub-pool reservation count: slots `[0, min_slot_index)` are never handed out, so each pool's real data starts at bytes `≥ entry_max` — strictly above any pool's slot-0 dummy-write region. |

---

## Appendix A — File map

| File | Role |
|---|---|
| [shared_memory_pool.py](../../python/sglang/srt/mem_cache/shared_memory_pool.py) | `SharedMemoryPool`, sub-pool specs, `Shared{MHA,Mamba,SWA,Hybrid}*` classes, `wrap_req_to_token_pool_write` |
| [multi_ended_allocator.py](../../python/sglang/srt/mem_cache/multi_ended_allocator.py) | `MultiEndedAllocator`, `SharedSWATokenToKVPoolAllocator`, `SharedMambaTokenToKVPoolAllocator`, eager-mapping update for `full_to_swa_index_mapping` |
| [relocation_log.py](../../python/sglang/srt/mem_cache/relocation_log.py) | `SlotBacktrack` (+ `_TreeNodeRef` / `_ReqPosRef` / `_AuxRef`; `clear_slot`, the OVERWRITE-ABORT / MULTI-OWNER asserts, the per-slot `history` ring), `SlotBacktrackBinder`, `NullBacktrackBinder`, `Relocator` (`apply` / `_apply_one`). Filename retained from the pre-refactor `RelocationLog`. |
| [base_prefix_cache.py](../../python/sglang/srt/mem_cache/base_prefix_cache.py) | Binder helpers (§7.6): `_set_node_value` / `_bind_node_value` / `_unbind_all_node_value` / `_unbind_and_free_node_value` / `_safe_unbind_tree_node` (full); `_set_node_mamba_value` / `_unbind_node_mamba_value` / `_unbind_and_free_node_mamba_value` (mamba); `_set_req_mamba_pool_idx`; `_set_req_prefix_indices`; `_check_value_overlap`; `_free_dedup_slots` |
| [chunk_cache.py, mamba_radix_cache.py, radix_cache.py, swa_radix_cache.py, radix_cache_cpp.py, session_aware_cache.py](../../python/sglang/srt/mem_cache/) | Use the binder helpers at every assignment site. `mamba_radix_cache.py` also has the `cache_*_req` validation backstops (`_is_mamba_pool_idx_valid`, `_are_kv_indices_valid`, `transfer_mamba_to_radix`). |
| [schedule_policy.py](../../python/sglang/srt/managers/schedule_policy.py) | `PrefillAdder.add_chunked_req` calls `self.tree_cache._set_req_prefix_indices(...)` for `req.prefix_indices` extension. |
| [model_runner_kv_cache_mixin.py](../../python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py) | `_init_pools` gating and binder/`Relocator` wiring |
| [scheduler.py](../../python/sglang/srt/managers/scheduler.py) | No shared-pool-specific hook any more (the old `flush_relocations` pre-forward hook was removed when reference updates became immediate — §8.3). Spec-decode `backup_state`/`restore_state` live on the allocator. |
| [server_args.py](../../python/sglang/srt/server_args.py) | `--enable-shared-memory-pool` flag + validation asserts (currently incompatible with `--disaggregation-mode`; see §9) |

## Appendix B — Flow example: SWA alloc → free → apply (immediate)

```
  t=0  Initial:
       ┌─────────────────────────────────────────────┐
       │ [slot0 reserved]  free  [slot(N-1) reserved]│
       └─────────────────────────────────────────────┘
       watermark_full = 1       watermark_swa = N-1

  t=1  full.alloc(3):  returns [1,2,3]
       ┌─┬─┬─┬─┬──────────────────────────┬─┐
       │r│1│2│3│       free middle        │r│
       └─┴─┴─┴─┴──────────────────────────┴─┘
       watermark_full = 4

  t=2  swa.alloc(2):   returns [N-2, N-1]  (grow-down indexing)
       ┌─┬─┬─┬─┬───────────────────┬─┬─┬─┐
       │r│1│2│3│    free middle    │ │ │r│
       └─┴─┴─┴─┴───────────────────┴─┴─┴─┘
       watermark_swa = N-3

  t=3  full.free([2]):  ALL of this happens inside this one call —
       1. catch-all: SlotBacktrack.clear_slot(2) — pops any binder entry
          for slot 2 (it was owned by req X which already unbound it, so
          this is a no-op here; for a _free_dedup_slots-freed slot it
          drops a leftover req_position entry).
       2. eager compaction: move_kv_cache(dst=2, src=3)  (bytes copied)
       3. Relocator.apply("full", src=[3], dst=[2]):
            * tree_node[3]?  → getattr(node, attr)[pos] = 2 ; rebind under 2
            * req_position[3] = _ReqPosRef(col=C, rows={r1,r2,...})
              → for r in rows: if req_to_token[r, C] == 3: req_to_token[r,C] = 2
            * aux[3]? → each (tensor, idx): tensor[idx] = 2
            * py_attr[3]? → setattr(obj, "_attr", tensor(2))
            * full_to_swa_index_mapping handled eagerly in _free_both:
              mapping[2] = mapping[3]; mapping[3] = 0
       4. _inverse_history["full"].append(([3], [2]))   # for spec rollback
       ┌─┬─┬─┬──────────────────────┬─┬─┬─┐
       │r│1│3│       free middle    │ │ │r│
       └─┴─┴─┴──────────────────────┴─┴─┴─┘
       watermark_full = 3
       — free() returns here with EVERY bound holder already saying "2".

  t=4  model_runner.forward(...)     ← reads correct data. No flush step.
```

## Appendix C — Worked example: cross-tick `prefix_indices` drift (pre-refactor)

> **Historical.** This trace is from the deferred-flush era (it diagrams the
> bug fixed in §11.4) and is kept because it's the clearest illustration of
> *why every persistent holder must be bound*. Two things differ under the
> current immediate-apply design: there is no per-tick `flush()` step (the
> relocation is applied inside `free()` at the moment it happens, so the
> "WITH" column updates the bound `prefix_indices` cell synchronously); and an
> *unbound* holder of a relocated slot (the "WITHOUT" column) no longer
> "drifts N relocations behind" silently — its first stale read trips the
> hard stale-slot assertion with full diagnostic context (§7.5, §7.7). The
> lesson is identical; only the failure surface improved.

Imagine a chunked request `R` that needs three prefill chunks; between chunks,
other reqs cause heavy allocator churn.

```
─────────────────────────────────────────────────────────────────────────
Tick T (chunk 1 of R completes):

  process_batch_result:
    cache_unfinished_req(R):
      kv_indices = req_to_token[R.rpi, :len_so_far]   = [.., 7, 8, 9, ..]
      ┌───── WITHOUT BINDING ─────┐    ┌────── WITH _set_req_prefix_indices ──────┐
      R.prefix_indices =          │    self._set_req_prefix_indices(R, ...)
        kv_indices.clone()        │      → bind_aux(7, R.prefix_indices, p)
        (numerically: [.., 7, ..])│        bind_aux(8, R.prefix_indices, p+1)
                                  │        bind_aux(9, R.prefix_indices, p+2)
      └──────────────────────────-┘    └─────────────────────────────────────────┘

    free(some_other_req)  → eager compaction
      Slot 9 was the boundary; it gets freed.
      Watermark shrinks past 9.        log: []  (boundary free, no entry)
      Slot 8 also freed in this batch:
        boundary 8 → freed slot 5      log: [(8, 5)]

─────────────────────────────────────────────────────────────────────────
Tick T+1:
  flush() applies log:
    (8, 5):  req_to_token cells with 8 → 5
             TreeNode.value cells with 8 → 5
             ┌── WITHOUT ──┐    ┌── WITH ──┐
             prefix_indices    prefix_indices[p+1] = 5  ✓
             still has 8       (the bound aux entry was transferred)
                               bind_aux moved (8, R.pi, p+1) → (5, R.pi, p+1)

  log.clear()

  Other batches run, more relocations recorded; flushed at T+2; cleared.
  ... many ticks pass ...

─────────────────────────────────────────────────────────────────────────
Tick T+N (R is finally picked back):
  get_next_batch_to_run picks R for chunk 2.
  prepare_for_extend reads R.prefix_indices:
    ┌─── WITHOUT ───┐                    ┌─── WITH ───┐
    [.., 7, 8, 9, ..]  (7 maybe still    [.., 7, 5, ?, ..]  (cells were
    valid; 8 is stale; 9 is GONE —         updated in lockstep
    boundary-freed long ago)               by every flush)

  write_cache_indices writes prefix_indices into req_to_token[R.rpi, ...]:
    ┌─── WITHOUT ───┐                    ┌─── WITH ───┐
    Cells now hold stale 8, 9.           Cells hold current ids.
    bind_req_position(8, p+1)            bind_req_position(5, p+1)
    bind_req_position(9, p+2)            ...
    flush() — log has nothing for        flush() — bound cells stay current.
    8 or 9 (long since cleared).
    forward() reads slot 8, 9 — GARBAGE.
    Eventually: free(req_to_token[..])
    triggers AssertionError: slot 9
    is outside allocated range.
─────────────────────────────────────────────────────────────────────────
```

The without-binding column is what the assertion in
`eval_results_2/eval_*` second-run logs caught. The with-binding column
(post-§11.4 fix) avoids the failure: each tick's flush walks the
relocation log once and updates `R.prefix_indices` cells in lockstep
with `req_to_token` and `TreeNode.value` cells, so when `R` is finally
picked back the clone holds current ids.
