#!/bin/bash
# Static-partition baseline (no shared memory pool). Single-server only —
# pin the GPU(s) via `docker run --gpus device=N` (or device=0,1 for TP=2).
#
# Eval-mode env knobs:
#   TP_SIZE                       (default 1)  — passed as --tp-size
#   MEM_FRACTION_STATIC           (default 0.85)
#   SGLANG_NATIVE_MOVE_KV_CACHE   honored from env. The `run_eval.sh`
#                                 orchestrator sets this to '0' for the
#                                 baseline_triton variant and '1' for the
#                                 baseline_native variant. Default in
#                                 SGLang itself is '0' (Triton tiled
#                                 kernel) so an unset env stays on the
#                                 fast path.
set -euo pipefail
MODEL="${1:?model path required}"
PORT="${2:?port required}"

TP_SIZE="${TP_SIZE:-1}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.85}"

# Resolve the KV-copy variant (informational banner).
if [ "${SGLANG_NATIVE_MOVE_KV_CACHE:-0}" = "1" ]; then
  KV_COPY_VARIANT="NATIVE (forced via SGLANG_NATIVE_MOVE_KV_CACHE=1)"
else
  KV_COPY_VARIANT="TRITON tiled kernel (default)"
fi

echo "[launch_baseline] static partition pool"
echo "[launch_baseline]   model            = $MODEL"
echo "[launch_baseline]   port             = $PORT"
echo "[launch_baseline]   tp-size          = $TP_SIZE"
echo "[launch_baseline]   mem-fraction     = $MEM_FRACTION_STATIC"
echo "[launch_baseline]   KV-copy path     = $KV_COPY_VARIANT"

# --disable-cuda-graph and --disable-piecewise-cuda-graph: both off for
# determinism during shared-pool testing. Piecewise CUDA Graph is an
# experimental SGLang feature that has been observed to hit Dynamo trace
# errors on hybrid models (e.g. Falcon-H1's forward) — disabling it makes
# this test setup robust across the three model shapes.
exec python3 -m sglang.launch_server \
  --model-path "$MODEL" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --page-size 1 \
  --tp-size "$TP_SIZE" \
  --mem-fraction-static "$MEM_FRACTION_STATIC" \
  --disable-cuda-graph \
  --disable-piecewise-cuda-graph
