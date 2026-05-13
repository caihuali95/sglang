#!/bin/bash
# Shared-memory-pool server. Single-server only — pin the GPU(s) via
# `docker run --gpus device=N` (or device=0,1 for TP=2) on the host.
#
# Eval-mode env knobs:
#   TP_SIZE                   (default 1)    — passed as --tp-size
#   MEM_FRACTION_STATIC       (default 0.85)
#   SGLANG_DISABLE_RADIX_CACHE (default 0)   — when 1, append
#       --disable-radix-cache to the launch_server args. Used to test
#       the hypothesis that the radix-cache `match_prefix` → eviction
#       race (where a leaf's slots are freed while a Req still holds
#       them in `prefix_indices` / `mamba_pool_idx` between
#       `match_prefix` return and the later `inc_lock_ref` at
#       `cache_unfinished_req`) is the upstream defect causing the
#       Mamba leak in eval_results_20-22. With the radix tree off,
#       `cache_finished_req` falls into the `disable` branch and frees
#       all `kv_indices` directly — no boundary math, no premature
#       free of tree slots. If the leak persists with this flag, the
#       defect is elsewhere; if it goes away, we have a precise
#       upstream fix target.
#
# KV-copy path: ALWAYS native here, regardless of SGLANG_NATIVE_MOVE_KV_CACHE.
# `SharedMHATokenToKVPool.move_kv_cache` overrides the parent and routes to
# `move_kv_cache_native` because the shared pool's reinterpret-strided
# views don't satisfy the Triton tiled-kernel's pointer/stride invariants.
# This is the structural cost the eval is meant to measure.
set -euo pipefail
MODEL="${1:?model path required}"
PORT="${2:?port required}"

TP_SIZE="${TP_SIZE:-1}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.85}"
SGLANG_DISABLE_RADIX_CACHE="${SGLANG_DISABLE_RADIX_CACHE:-0}"

extra_args=()
if [ "$SGLANG_DISABLE_RADIX_CACHE" = "1" ]; then
  extra_args+=(--disable-radix-cache)
fi

echo "[launch_shared] shared memory pool"
echo "[launch_shared]   model         = $MODEL"
echo "[launch_shared]   port          = $PORT"
echo "[launch_shared]   tp-size       = $TP_SIZE"
echo "[launch_shared]   mem-fraction  = $MEM_FRACTION_STATIC"
echo "[launch_shared]   disable-radix = $SGLANG_DISABLE_RADIX_CACHE"
echo "[launch_shared]   KV-copy path  = NATIVE (forced by SharedMHATokenToKVPool.move_kv_cache)"

# See launch_baseline.sh for why --disable-piecewise-cuda-graph is on.
exec python3 -m sglang.launch_server \
  --model-path "$MODEL" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --page-size 1 \
  --tp-size "$TP_SIZE" \
  --mem-fraction-static "$MEM_FRACTION_STATIC" \
  --disable-cuda-graph \
  --disable-piecewise-cuda-graph \
  --enable-shared-memory-pool \
  "${extra_args[@]}"
