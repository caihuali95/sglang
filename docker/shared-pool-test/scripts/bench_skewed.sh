#!/bin/bash
# Skewed workload — long input, short output. For SWA hybrid models this
# pressures the full-attn pool (KV grows linearly with context); for
# Mamba-hybrid models the same shape pressures the full-attn pool while
# Mamba sits idle. Used to demonstrate the OOM-vs-no-OOM difference between
# the static partition and the shared pool.
set -euo pipefail
PORT="${PORT:-${PORT_SHARED:-30001}}"
INPUT_LEN="${INPUT_LEN:-4096}"
OUTPUT_LEN="${OUTPUT_LEN:-64}"
NUM_PROMPTS="${NUM_PROMPTS:-200}"
CONCURRENCY="${CONCURRENCY:-64}"

# See bench_balanced.sh for why --dataset-path / --output-file are passed.
DATASET_PATH="${DATASET_PATH:-/workspace/assets/ShareGPT_V3_unfiltered_cleaned_split.json}"
extra=()
if [ -f "$DATASET_PATH" ]; then
  extra+=(--dataset-path "$DATASET_PATH")
fi
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
