#!/bin/bash
# End-to-end baseline run: launch a static-partition server, wait for it to
# be ready, run the benchmark suite against it, save results, shut it down.
#
# Single-server only. Pin the GPU on `docker run` with `--gpus device=N`.
# Override workload via env vars (INPUT_LEN / OUTPUT_LEN / NUM_PROMPTS /
# CONCURRENCY) or skip stages via SKIP_BALANCED=1 / SKIP_SKEWED=1.
set -euo pipefail

MODEL="${MODEL_ID:-tiiuae/Falcon-H1-0.5B-Instruct}"
PORT="${PORT_BASELINE:-30000}"

TS="$(date +%Y%m%d-%H%M%S)"
RESULT_DIR="${RESULT_DIR:-/workspace/results}/baseline_$TS"
LOG_DIR="$RESULT_DIR/logs"
mkdir -p "$RESULT_DIR" "$LOG_DIR"

echo "==> Result dir: $RESULT_DIR"
echo "==> Model:      $MODEL"
echo "==> Port:       $PORT"

echo "==> Launching baseline server ..."
/workspace/scripts/launch_baseline.sh "$MODEL" "$PORT" \
    > "$LOG_DIR/server.log" 2>&1 &
SERVER=$!

cleanup() {
  echo "==> Stopping baseline server (pid $SERVER) ..."
  kill "$SERVER" 2>/dev/null || true
  wait "$SERVER" 2>/dev/null || true
}
trap cleanup EXIT

echo "==> Waiting for /health_generate (up to 480s) ..."
for i in $(seq 1 240); do
  if curl -sf "http://127.0.0.1:${PORT}/health_generate" > /dev/null; then
    echo "==> baseline ready (port $PORT)"
    break
  fi
  sleep 2
done
if ! curl -sf "http://127.0.0.1:${PORT}/health_generate" > /dev/null; then
  echo "ERROR: baseline server did not come up within 480s"
  echo "--- last 80 lines of $LOG_DIR/server.log ---"
  tail -n 80 "$LOG_DIR/server.log" || true
  exit 1
fi

# ---- 1. Single-prompt sanity ----
echo
echo "==> [1/3] Single-prompt sanity ..."
curl -s -X POST "http://127.0.0.1:${PORT}/generate" \
  -H "Content-Type: application/json" \
  -d '{"text":"The capital of France is","sampling_params":{"max_new_tokens":16,"temperature":0}}' \
  | python3 -c 'import sys,json; print(json.load(sys.stdin)["text"])' \
  | tee "$RESULT_DIR/single_prompt.txt"

# ---- 2. Balanced bench ----
if [ "${SKIP_BALANCED:-0}" != "1" ]; then
  echo
  echo "==> [2/3] Balanced bench (input=1024 output=256 n=200 c=32) ..."
  PORT="$PORT" \
  INPUT_LEN="${INPUT_LEN:-1024}" \
  OUTPUT_LEN="${OUTPUT_LEN:-256}" \
  NUM_PROMPTS="${NUM_PROMPTS:-200}" \
  CONCURRENCY="${CONCURRENCY:-32}" \
    /workspace/scripts/bench_balanced.sh \
    2>&1 | tee "$RESULT_DIR/bench_balanced.txt"
fi

# ---- 3. Skewed bench (may OOM on baseline at default workload) ----
if [ "${SKIP_SKEWED:-0}" != "1" ]; then
  echo
  echo "==> [3/3] Skewed bench (input=4096 output=64 n=200 c=64) ..."
  PORT="$PORT" \
  INPUT_LEN="${SKEWED_INPUT_LEN:-4096}" \
  OUTPUT_LEN="${SKEWED_OUTPUT_LEN:-64}" \
  NUM_PROMPTS="${SKEWED_NUM_PROMPTS:-200}" \
  CONCURRENCY="${SKEWED_CONCURRENCY:-64}" \
    /workspace/scripts/bench_skewed.sh \
    > "$RESULT_DIR/bench_skewed.txt" 2>&1 \
    || echo "(skewed bench finished non-zero — see $RESULT_DIR/bench_skewed.txt)"
fi

echo
echo "==> Done. Results under $RESULT_DIR"
ls -la "$RESULT_DIR"
