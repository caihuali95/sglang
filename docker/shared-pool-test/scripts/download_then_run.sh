#!/bin/bash
#
# Target-side EVAL bootstrapper. Pulls the latest `sglang-hybrid:mamba_swa_eval`
# image from TOS, then runs the eval matrix for two models in sequence:
#
#   1. Mamba-hybrid    : tiiuae/Falcon-H1-7B-Instruct  (SharedMambaPool path)
#   2. Symmetric SWA   : openai/gpt-oss-20b            (SharedSWAKVPool path)
#
# For EACH model, the eval iterates the cross-product of:
#   variants     {baseline_triton, baseline_native, shared}
#   tp sizes     EVAL_TP_SIZES         (default "1,2")
#   mem_fractions EVAL_MFS             (default "0.85,0.55")
#   workloads    EVAL_WORKLOADS        (default "balanced,balanced-large,
#                                       skewed,skewed-large")
#
# Output (per model):
#   ./results/eval_<model_slug>_<ts>/
#     runs/<cell>__<workload>.jsonl       (raw bench summaries)
#     summary.csv                          (long form, one row per JSONL)
#     summary.md                           (pivoted, per-cell deltas)
#     summary.console.txt                  (compact stdout snapshot)
#
# Skip individual models:
#   SKIP_MAMBA=1 ./download_then_run.sh
#   SKIP_SWA=1   ./download_then_run.sh
#
# Skip the image refresh and just re-run against an already-loaded image:
#   SKIP_IMAGE_PULL=1 ./download_then_run.sh
#
# Trim the matrix (faster runs, smaller eval surface):
#   EVAL_TP_SIZES=1 EVAL_MFS=0.85 ./download_then_run.sh
#   EVAL_WORKLOADS=balanced,skewed ./download_then_run.sh
#   EVAL_VARIANTS=baseline_triton,shared ./download_then_run.sh
#
# Capture per-cell nsys traces (requires INSTALL_PROFILING=1 image build):
#   RUN_EVAL_PROFILE=nsys ./download_then_run.sh
#
# All eval env vars are forwarded into the container.
#
# Don't fail-fast on a single model's failure — both runs proceed regardless.
set -uo pipefail

# ---- 1. Refresh image from TOS (skippable). ----
if [ "${SKIP_IMAGE_PULL:-0}" != "1" ]; then
  set -e

if [ -z "${TOS_BUCKET:-}" ]; then
  echo "ERROR: TOS_BUCKET environment variable is not set." >&2
  exit 1
fi

  rm -f ./sglang.tar.gz

  echo "==> Waiting for sglang.tar.gz to be uploaded to TOS..."
  while ! ./tosutil cp tos://${TOS_BUCKET}/sglang.tar.gz ./; do
    echo "    Image not available yet. Waiting 15s ..."
    sleep 15
  done
  echo "==> Image downloaded successfully!"

  echo "==> Extracting the container image..."
  rm -f ./sglang.tar
  gunzip ./sglang.tar.gz

  echo "==> Removing existing image / containers if they exist..."
  docker rm -f $(docker ps -a -q --filter ancestor=sglang-hybrid:mamba_swa_eval) 2>/dev/null || true
  docker image rm sglang-hybrid:mamba_swa_eval || true

  echo "==> Loading image into Docker..."
  docker load -i sglang.tar

  echo "==> Listing Docker images:"
  docker image ls

  echo "==> Cleaning up extracted tar..."
  rm ./sglang.tar

  set +e
else
  echo "==> SKIP_IMAGE_PULL=1; using already-loaded sglang-hybrid:mamba_swa_eval"
fi

mkdir -p ./results

# ---- 2. Per-model eval runner. ----
# Args: $1 slug ("mamba"|"swa"), $2 model id, $3 friendly description.
#
# Exposes BOTH GPUs to the container (`--gpus '"device=0,1"'`) so the
# tp_sizes=1 cells use one GPU and tp_sizes=2 cells use both. If the host
# only has one usable GPU, run with `EVAL_TP_SIZES=1` to skip the tp=2
# combinations.
#
# The inner double-quotes around `device=0,1` are required: without them
# the docker CLI parses the comma as a top-level option separator and
# interprets bare `1` as `count=1`, producing the daemon error
# "cannot set both Count and DeviceIDs on device request."
run_one() {
  local slug="$1" model="$2" desc="$3"
  local skip_var="SKIP_${slug^^}"
  if [ "${!skip_var:-0}" = "1" ]; then
    echo
    echo "==> [${slug}] SKIPPED (${skip_var}=1) — ${desc}"
    return 0
  fi

  echo
  echo "==> [${slug}] Running run_eval against ${desc}"
  echo "    model = ${model}"
  echo "    log   = ./${slug}_run_eval.log"
  echo "    (eval results land under ./results/eval_<model_slug>_<ts>/)"

  # Clean up any stale containers from a prior eval run that didn't exit
  # cleanly. Avoids "port is already allocated" surprises and prevents
  # zombie containers from accumulating across runs.
  local stale
  stale="$(docker ps -a -q --filter ancestor=sglang-hybrid:mamba_swa_eval 2>/dev/null)"
  if [ -n "$stale" ]; then
    echo "    cleaning up stale eval containers: $(echo "$stale" | tr '\n' ' ')"
    docker rm -f $stale 2>/dev/null || true
  fi

  # No `-p 30001:30001`: bench_serving lives inside this container and
  # talks to 127.0.0.1:30001 internally. Publishing the port would only
  # produce host-port conflicts when other processes share 30001.
  docker run --rm \
    --gpus '"device=0,1"' \
    --shm-size=8g --ipc=host \
    -v "$(pwd)/results:/workspace/results" \
    -e MODEL_ID="$model" \
    -e EVAL_VARIANTS="${EVAL_VARIANTS:-baseline_triton,baseline_native,shared}" \
    -e EVAL_TP_SIZES="${EVAL_TP_SIZES:-1,2}" \
    -e EVAL_MFS="${EVAL_MFS:-0.85,0.55}" \
    -e EVAL_WORKLOADS="${EVAL_WORKLOADS:-balanced,balanced-large,skewed,skewed-large}" \
    -e RUN_EVAL_PROFILE="${RUN_EVAL_PROFILE:-none}" \
    -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
    -e HF_HUB_OFFLINE=1 \
    -e TRANSFORMERS_OFFLINE=1 \
    -e SGLANG_SHARED_POOL_MAPPING_ASSERT="${SGLANG_SHARED_POOL_MAPPING_ASSERT:-1}" \
    -e SGLANG_DISABLE_RADIX_CACHE="${SGLANG_DISABLE_RADIX_CACHE:-0}" \
    -e EVAL_FAIL_FAST="${EVAL_FAIL_FAST:-0}" \
    sglang-hybrid:mamba_swa_eval run_eval \
    2>&1 | tee "${slug}_run_eval.log"
  local rc=${PIPESTATUS[0]}

  if [ "$rc" = "0" ]; then
    echo "==> [${slug}] OK"
  else
    echo "==> [${slug}] FAILED (rc=${rc}) — see ${slug}_run_eval.log; continuing"
  fi
  return 0
}

run_one mamba "tiiuae/Falcon-H1-7B-Instruct" \
  "Mamba-hybrid (SharedMambaPool path)"

run_one swa "openai/gpt-oss-20b" \
  "Symmetric SWA-hybrid (SharedSWAKVPool path)"

echo
echo "==> All requested eval runs completed."
echo "    Per-model stdout (one per non-skipped stage): ./<slug>_run_eval.log"
echo "    Per-model eval result dirs: ./results/eval_<model_slug>_<ts>/"
echo "    Each eval dir has summary.csv + summary.md for cross-variant deltas."
