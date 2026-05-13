#!/bin/bash
# Generate an Nsight Systems trace of the shared-pool server under load.
# Output: /workspace/profiles/sglang-shared.nsys-rep — view in Nsight on
# your laptop. Mount /workspace/profiles to the host to retrieve.
set -euo pipefail

MODEL="${1:?model path required}"
PORT="${PORT:-${PORT_SHARED:-30001}}"
PROFILE_DIR="${PROFILE_DIR:-/workspace/profiles}"
mkdir -p "$PROFILE_DIR"

INPUT_LEN="${INPUT_LEN:-1024}"
OUTPUT_LEN="${OUTPUT_LEN:-256}"
NUM_PROMPTS="${NUM_PROMPTS:-100}"
CONCURRENCY="${CONCURRENCY:-16}"

LOG_DIR="${LOG_DIR:-/workspace/logs}"
mkdir -p "$LOG_DIR"

echo "==> Launching shared-pool server under nsys ..."
nsys profile \
  --output "$PROFILE_DIR/sglang-shared" \
  --trace=cuda,nvtx,osrt,cudnn \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --force-overwrite=true \
  -- \
  python3 -m sglang.launch_server \
    --model-path "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --page-size 1 \
    --mem-fraction-static "${MEM_FRACTION_STATIC:-0.85}" \
    --disable-cuda-graph \
    --disable-piecewise-cuda-graph \
    --enable-shared-memory-pool \
    > "$LOG_DIR/profile.log" 2>&1 &
SERVER=$!

cleanup() {
  kill -INT "$SERVER" 2>/dev/null || true
  wait "$SERVER" 2>/dev/null || true
}
trap cleanup EXIT

echo "==> Waiting for server to come up ..."
for i in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:${PORT}/health_generate" > /dev/null; then
    break
  fi
  sleep 2
done

if ! curl -sf "http://127.0.0.1:${PORT}/health_generate" > /dev/null; then
  echo "ERROR: server did not come up. Last 80 lines of log:"
  tail -n 80 "$LOG_DIR/profile.log" || true
  exit 1
fi

echo "==> Driving load (input=$INPUT_LEN, output=$OUTPUT_LEN, n=$NUM_PROMPTS) ..."
python3 -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 \
  --port "$PORT" \
  --dataset-name random \
  --random-input-len "$INPUT_LEN" \
  --random-output-len "$OUTPUT_LEN" \
  --num-prompts "$NUM_PROMPTS" \
  --max-concurrency "$CONCURRENCY"

echo
echo "==> Profile written to $PROFILE_DIR/sglang-shared.nsys-rep"
echo "    On the host: docker cp <container>:$PROFILE_DIR/sglang-shared.nsys-rep ."
echo "    Or volume-mount $PROFILE_DIR to retrieve automatically."
