#!/bin/bash
# End-to-end shared-memory-pool run: launch a server with
# --enable-shared-memory-pool, wait for it to be ready, run the benchmark
# suite against it, save results, shut it down.
#
# Single-server only. Pin the GPU on `docker run` with `--gpus device=N`.
# Override workload via env vars (INPUT_LEN / OUTPUT_LEN / NUM_PROMPTS /
# CONCURRENCY) or skip stages via SKIP_BALANCED=1 / SKIP_SKEWED=1.
set -euo pipefail

MODEL="${MODEL_ID:-tiiuae/Falcon-H1-0.5B-Instruct}"
PORT="${PORT_SHARED:-30001}"

TS="$(date +%Y%m%d-%H%M%S)"
RESULT_DIR="${RESULT_DIR:-/workspace/results}/shared_$TS"
LOG_DIR="$RESULT_DIR/logs"
mkdir -p "$RESULT_DIR" "$LOG_DIR"

echo "==> Result dir: $RESULT_DIR"
echo "==> Model:      $MODEL"
echo "==> Port:       $PORT"

echo "==> Launching shared-memory-pool server ..."
/workspace/scripts/launch_shared.sh "$MODEL" "$PORT" \
    > "$LOG_DIR/server.log" 2>&1 &
SERVER=$!

cleanup() {
  echo "==> Stopping shared server (pid $SERVER) ..."
  kill "$SERVER" 2>/dev/null || true
  wait "$SERVER" 2>/dev/null || true
}
trap cleanup EXIT

echo "==> Waiting for /health_generate (up to 480s) ..."
for i in $(seq 1 240); do
  if curl -sf "http://127.0.0.1:${PORT}/health_generate" > /dev/null; then
    echo "==> shared ready (port $PORT)"
    break
  fi
  sleep 2
done
if ! curl -sf "http://127.0.0.1:${PORT}/health_generate" > /dev/null; then
  echo "ERROR: shared server did not come up within 480s"
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
# Helper: when a bench step fails, print the last 60 lines of the server
# log so we can immediately tell post-bench unresponsiveness (no exception
# in the log, server alive but slow to answer the trailing /get_server_info)
# from a real scheduler crash (Python traceback / SIGQUIT in the log).
_dump_server_tail_on_fail() {
  local rc="$1" stage="$2"
  if [ "$rc" != "0" ]; then
    echo
    echo "==> [${stage}] bench exited non-zero (rc=${rc})."
    echo "    Last 60 lines of server log:"
    echo "    --- $LOG_DIR/server.log ---"
    tail -n 60 "$LOG_DIR/server.log" 2>/dev/null \
      | sed 's/^/    /' || true
    echo "    --- end ---"
  fi
}

# Both bench stages: (i) write a structured JSONL via `--output-file` so the
# numeric data survives even when the trailing /get_server_info GET fails,
# (ii) tolerate non-zero exit so the server isn't torn down by run_shared's
# EXIT trap when the bench client crashes after all requests succeed.

if [ "${SKIP_BALANCED:-0}" != "1" ]; then
  echo
  echo "==> [2/3] Balanced bench (input=1024 output=256 n=200 c=32) ..."
  # Disable fail-fast around the pipe so PIPESTATUS capture + on-fail tail
  # dump actually run when the bench client errors out (e.g., post-bench
  # `/get_server_info` GET racing with the server's last-cleanup burst).
  set +e
  PORT="$PORT" \
  INPUT_LEN="${INPUT_LEN:-1024}" \
  OUTPUT_LEN="${OUTPUT_LEN:-256}" \
  NUM_PROMPTS="${NUM_PROMPTS:-200}" \
  CONCURRENCY="${CONCURRENCY:-32}" \
  OUTPUT_FILE="$RESULT_DIR/bench_balanced.jsonl" \
    /workspace/scripts/bench_balanced.sh \
    2>&1 | tee "$RESULT_DIR/bench_balanced.txt"
  rc=${PIPESTATUS[0]}
  set -e
  _dump_server_tail_on_fail "$rc" "2/3 balanced"
fi

# ---- 3. Skewed bench (the workload shared pool should handle without OOM) ----
if [ "${SKIP_SKEWED:-0}" != "1" ]; then
  echo
  echo "==> [3/3] Skewed bench (input=4096 output=64 n=200 c=64) ..."
  set +e
  PORT="$PORT" \
  INPUT_LEN="${SKEWED_INPUT_LEN:-4096}" \
  OUTPUT_LEN="${SKEWED_OUTPUT_LEN:-64}" \
  NUM_PROMPTS="${SKEWED_NUM_PROMPTS:-200}" \
  CONCURRENCY="${SKEWED_CONCURRENCY:-64}" \
  OUTPUT_FILE="$RESULT_DIR/bench_skewed.jsonl" \
    /workspace/scripts/bench_skewed.sh \
    2>&1 | tee "$RESULT_DIR/bench_skewed.txt"
  rc=${PIPESTATUS[0]}
  set -e
  _dump_server_tail_on_fail "$rc" "3/3 skewed"
fi

echo
echo "==> Done. Results under $RESULT_DIR"
ls -la "$RESULT_DIR"
