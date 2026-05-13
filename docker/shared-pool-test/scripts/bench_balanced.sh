#!/bin/bash
# Balanced workload — used to verify the shared-pool steady-state cost is
# within ~2% of the static partition. Run against ONE port at a time
# (override with PORT=...; default 30001 = shared).
set -euo pipefail
PORT="${PORT:-${PORT_SHARED:-30001}}"
INPUT_LEN="${INPUT_LEN:-1024}"
OUTPUT_LEN="${OUTPUT_LEN:-256}"
NUM_PROMPTS="${NUM_PROMPTS:-200}"
CONCURRENCY="${CONCURRENCY:-32}"

# The new sglang.bench_serving `random` dataset samples token IDs from
# ShareGPT to build realistic-distribution prompts. When the image is built
# with the default DATASETS=1 prefetch, ShareGPT is baked at
# /workspace/assets/. Pass it explicitly so HF_HUB_OFFLINE=1 doesn't trigger
# a forbidden network fetch.
DATASET_PATH="${DATASET_PATH:-/workspace/assets/ShareGPT_V3_unfiltered_cleaned_split.json}"
extra=()
if [ -f "$DATASET_PATH" ]; then
  extra+=(--dataset-path "$DATASET_PATH")
fi
# `bench_serving --output-file` writes a JSONL summary BEFORE the trailing
# /get_server_info GET runs. Survive transient post-bench unresponsiveness
# by always asking for the file when OUTPUT_FILE is set.
if [ -n "${OUTPUT_FILE:-}" ]; then
  extra+=(--output-file "$OUTPUT_FILE")
fi

exec python3 -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 \
  --port "$PORT" \
  --dataset-name random \
  --random-input-len "$INPUT_LEN" \
  --random-output-len "$OUTPUT_LEN" \
  --num-prompts "$NUM_PROMPTS" \
  --max-concurrency "$CONCURRENCY" \
  "${extra[@]}"
