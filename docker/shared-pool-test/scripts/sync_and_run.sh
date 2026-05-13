#!/bin/bash
# Target-side EVAL-mode runner.
#
# Each (model, tp, mfs) "cell" runs as ITS OWN container on its own GPU
# subset. With an 8-GPU pool and the default
# (tp ∈ {1,2}) × (mfs ∈ {0.55, 0.85}) matrix, all four cells of a single
# model run in parallel: tp=1 cells take one GPU each, tp=2 cells take
# two each (1+1+2+2 = 6 GPUs, fits in the 8-GPU pool with idle slack).
# Per-model wall-time drops from ~4 cell durations to ~1 cell duration.
# When both models run, the script does the mamba wave first, then the
# swa wave (so they share the same GPU pool sequentially).
#
# Models iterated:
#   1. Mamba-hybrid    : tiiuae/Falcon-H1-7B-Instruct  (SharedMambaPool)
#   2. Symmetric SWA   : openai/gpt-oss-20b            (SharedSWAKVPool)
#
# Standard env vars (knobs):
#   SKIP_MAMBA / SKIP_SWA          skip a model's wave entirely.
#   SKIP_SYNC                      reuse the previously-extracted source
#                                  instead of re-fetching the tarball.
#   EVAL_VARIANTS                  default "baseline_triton,baseline_native,shared".
#   EVAL_TP_SIZES                  default "1,2".
#   EVAL_MFS                       default "0.85,0.55".
#   EVAL_WORKLOADS                 default "balanced,balanced-large,skewed,skewed-large".
#   DEBUG_RERUN                    activates the broad failing-matrix profile
#                                  (shared-only, both tp ∈ {1,2}, both mfs,
#                                  3 historically-failing workloads).
#   DEBUG_WORKLOADS_{MAMBA,SWA}    per-model overrides under DEBUG_RERUN.
#   RUN_EVAL_PROFILE               "none" | "nsys" — enables per-cell
#                                  nsys traces (requires INSTALL_PROFILING=1
#                                  in the image).
#   EVAL_FAIL_FAST                 1 = stop at the first failed cell/wave.
#   SGLANG_DISABLE_RADIX_CACHE     1 = run with `--disable-radix-cache`.
#   SGLANG_SHARED_POOL_MAPPING_ASSERT
#                                  diagnostic assertion mode for the SWA
#                                  composite mapping update.
#
# Layout / scheduler env vars (with defaults that fit a fresh 8-GPU host):
#   GPUS              "0,1,2,3,4,5,6,7"   comma-separated CUDA device ids.
#                                          Override when only a subset is
#                                          available, or when running
#                                          alongside another job:
#                                              GPUS=4,5,6,7 ./sync_and_run.sh
#   SRC_DIR           "./sglang_source"   extracted source on the target.
#                                          The script bind-mounts
#                                          $SRC_DIR/sglang into the
#                                          container's dist-packages and
#                                          $SRC_DIR/scripts into
#                                          /workspace/scripts.
#   RESULTS_DIR       "./results" or       target-side aggregated results.
#                     "./results_<N>"      Each cell writes to a sub-
#                                          directory keyed by
#                                          {model_slug}__tp{tp}__mfs{mfs}
#                                          so concurrent cells never
#                                          share an output path. The
#                                          default is `./results` when
#                                          no positional N is given;
#                                          `./results_<N>` when N IS
#                                          given (see "Positional arg"
#                                          below). An explicit
#                                          RESULTS_DIR=... env override
#                                          wins over both.
#   TARBALL           "sglang-source.tar.gz"
#                                          name fetched from TOS.
#   EVAL_PARALLEL     "1"                  0 = sequential one-cell-per-
#                                          container (slower but cleaner
#                                          logs when diagnosing); 1 =
#                                          pack the per-model matrix into
#                                          a parallel wave on the GPU
#                                          pool (default).
#
# Positional arg (optional):
#   N    Non-negative integer. When given, `RESULTS_DIR` defaults to
#        `./results_<N>` instead of `./results`. This pairs with
#        `package_eval_results.sh <N>` (which reads from `./results_<N>`),
#        so a single integer ties together a run + its packaged zip.
#
# Typical invocations:
#     ./sync_and_run.sh                                 # full matrix, parallel; results → ./results
#     ./sync_and_run.sh 36                              # results → ./results_36
#     SKIP_SWA=1 ./sync_and_run.sh                      # Mamba only
#     SKIP_SWA=1 ./sync_and_run.sh 37                   # Mamba only; results → ./results_37
#     EVAL_FAIL_FAST=1 ./sync_and_run.sh                # stop on first crash
#     EVAL_PARALLEL=0 ./sync_and_run.sh                 # sequential (clean logs)
#     EVAL_TP_SIZES=1 EVAL_MFS=0.55 EVAL_WORKLOADS=skewed-large \
#         SKIP_SWA=1 ./sync_and_run.sh                  # narrow to one cell
#     GPUS=4,5,6,7 ./sync_and_run.sh                    # custom GPU subset
#     SKIP_SYNC=1 ./sync_and_run.sh                     # reuse extracted source
#
set -uo pipefail

# ---- 0. CLI arg parsing ---------------------------------------------------

usage() {
  echo "Usage: $0 [N]" >&2
  echo "  N    optional non-negative integer; when given, RESULTS_DIR" >&2
  echo "       defaults to ./results_<N> (matching package_eval_results.sh)." >&2
}

RESULTS_RUN_NUM=""
case "${1:-}" in
  ""|"--")
    : # no positional arg
    ;;
  --help|-h)
    usage
    exit 0
    ;;
  -*)
    echo "ERROR: unknown option '$1'." >&2
    usage
    exit 2
    ;;
  *)
    if ! [[ "$1" =~ ^[0-9]+$ ]]; then
      echo "ERROR: positional arg must be a non-negative integer, got '$1'." >&2
      usage
      exit 2
    fi
    RESULTS_RUN_NUM="$1"
    shift
    ;;
esac

# ---- 1. Knobs and defaults -------------------------------------------------

TARBALL="${TARBALL:-sglang-source.tar.gz}"
SRC_DIR="${SRC_DIR:-./sglang_source}"
# `RESULTS_DIR` precedence:
#   1. Explicit env var (highest priority — full override).
#   2. `./results_<N>` if a positional N was given.
#   3. `./results` (no N, no env override).
if [ -n "$RESULTS_RUN_NUM" ]; then
  RESULTS_DIR="${RESULTS_DIR:-./results_${RESULTS_RUN_NUM}}"
else
  RESULTS_DIR="${RESULTS_DIR:-./results}"
fi
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
EVAL_PARALLEL="${EVAL_PARALLEL:-1}"

# Container-name namespace. Each parallel container gets
# `eval_<RUN_ID>_<seq>` so a cleanup filter on the prefix kills only
# THIS invocation's containers — never another concurrent invocation
# (e.g., a sibling shell on the same host running a different cell).
RUN_ID="$(date +%Y%m%d-%H%M%S)-$$"
CONTAINER_PREFIX="eval_${RUN_ID}"

IFS=',' read -ra GPU_POOL <<< "$GPUS"
GPU_POOL_SIZE=${#GPU_POOL[@]}

echo "==> Runner configuration:"
echo "    RUN_ID      = $RUN_ID"
echo "    TARBALL     = $TARBALL"
echo "    SRC_DIR     = $SRC_DIR"
echo "    RESULTS_DIR = $RESULTS_DIR"
echo "    GPU pool    = ${GPU_POOL[*]} (size=$GPU_POOL_SIZE)"
echo "    parallel    = $EVAL_PARALLEL"
echo

# ---- 2. Refresh source from TOS (skippable) -------------------------------

if [ "${SKIP_SYNC:-0}" != "1" ]; then
  set -e

if [ -z "${TOS_BUCKET:-}" ]; then
  echo "ERROR: TOS_BUCKET environment variable is not set." >&2
  exit 1
fi
  echo "==> Downloading $TARBALL from TOS ..."
  rm -f "$TARBALL"
  while ! ./tosutil cp tos://${TOS_BUCKET}/${TARBALL} ./; do
    echo "    Not available yet; retrying in 10s ..."
    sleep 10
  done

  echo "==> Extracting to $SRC_DIR/ ..."
  rm -rf "$SRC_DIR"
  mkdir -p "$SRC_DIR"
  tar -xzf "$TARBALL" -C "$SRC_DIR"

  if [ ! -f "$SRC_DIR/sglang/__init__.py" ]; then
    echo "ERROR: $SRC_DIR/sglang/__init__.py not found after extracting." >&2
    exit 1
  fi
  if [ ! -d "$SRC_DIR/scripts" ]; then
    echo "ERROR: $SRC_DIR/scripts not found after extracting." >&2
    exit 1
  fi

  echo "==> Source ready at $SRC_DIR/"
  echo "    sglang  files : $(find $SRC_DIR/sglang  -type f -name '*.py' | wc -l)"
  echo "    sglang  size  : $(du -sh $SRC_DIR/sglang  | cut -f1)"
  echo "    scripts files : $(find $SRC_DIR/scripts -type f | wc -l)"
  echo "    scripts size  : $(du -sh $SRC_DIR/scripts | cut -f1)"
  chmod +x "$SRC_DIR/scripts"/*.sh 2>/dev/null || true
  set +e
else
  if [ ! -f "$SRC_DIR/sglang/__init__.py" ] || [ ! -d "$SRC_DIR/scripts" ]; then
    echo "ERROR: SKIP_SYNC=1 but $SRC_DIR is missing sglang/ or scripts/." >&2
    exit 1
  fi
  echo "==> SKIP_SYNC=1; using existing $SRC_DIR/{sglang,scripts}/"
fi

mkdir -p "$RESULTS_DIR"

# ---- 3. DEBUG_RERUN profile -----------------------------------------------

if [ "${DEBUG_RERUN:-0}" = "1" ]; then
  echo "==> DEBUG_RERUN=1: broad failing-matrix coverage."
  : "${EVAL_VARIANTS:=shared}"
  : "${EVAL_TP_SIZES:=1,2}"
  : "${EVAL_MFS:=0.55,0.85}"
  : "${DEBUG_WORKLOADS_MAMBA:=balanced-large,skewed,skewed-large}"
  : "${DEBUG_WORKLOADS_SWA:=balanced-large,skewed,skewed-large}"
fi

# Final resolved values for the per-model knobs. Used only to populate the
# per-cell env into each container — `run_eval.sh` inside the container
# does its own parsing.
EVAL_VARIANTS_FINAL="${EVAL_VARIANTS:-baseline_triton,baseline_native,shared}"
EVAL_TP_SIZES_FINAL="${EVAL_TP_SIZES:-1,2}"
EVAL_MFS_FINAL="${EVAL_MFS:-0.85,0.55}"

# ---- 4. Cell expansion -----------------------------------------------------
#
# Each "cell" is one (model, tp, mfs) triple — i.e., exactly one
# server boot inside its own container. We pin EVAL_VARIANTS at the
# cell level too so each container's run_eval iterates a single value
# (otherwise a tp=2 cell would also start a tp=1 container in the same
# pool slot, which is a separate cell). Variants are typically just
# "shared" for shared-pool testing, but we honor the user's EVAL_VARIANTS knob.

IFS=',' read -ra TP_ARR_FINAL  <<< "$EVAL_TP_SIZES_FINAL"
IFS=',' read -ra MFS_ARR_FINAL <<< "$EVAL_MFS_FINAL"

# Sanity: every tp must fit in the GPU pool — otherwise we cannot
# schedule that cell at all. (We support multi-wave scheduling for
# fitting cells whose total exceeds the pool, but a single cell whose
# tp > pool size is impossible.)
for tp in "${TP_ARR_FINAL[@]}"; do
  if [ "$tp" -gt "$GPU_POOL_SIZE" ]; then
    echo "ERROR: cell tp=$tp exceeds GPU pool size=$GPU_POOL_SIZE." >&2
    echo "       Either widen GPUS or trim EVAL_TP_SIZES." >&2
    exit 1
  fi
done

# ---- 5. Per-cell launcher --------------------------------------------------
#
# Args: $1 cell_idx, $2 model_slug, $3 model_id, $4 tp, $5 mfs,
#       $6 device-id-list (e.g. "2,3"), $7 workloads (comma-separated).
#
# Stdout/stderr is teed to a per-cell log so concurrent cells don't
# scramble each other's output.
launch_cell() {
  local cell_idx="$1"
  local model_slug="$2"
  local model_id="$3"
  local tp="$4"
  local mfs="$5"
  local devs="$6"
  local workloads="$7"

  local cell_name="${model_slug}__tp${tp}__mfs${mfs}"
  local cell_results="${RESULTS_DIR}/${cell_name}"
  local cell_log="${RESULTS_DIR}/${cell_name}.run_eval.log"
  local cname="${CONTAINER_PREFIX}_${cell_idx}_${cell_name}"

  mkdir -p "$cell_results"

  echo "==> [cell ${cell_idx}: ${cell_name}] launching on GPUs=${devs}"
  echo "    container   = ${cname}"
  echo "    results dir = ${cell_results}"
  echo "    log         = ${cell_log}"

  # `--gpus '"device=2,3"'` — the inner double-quotes are required, see
  # sync_and_run.sh comments for the rationale.
  docker run --rm \
    --name "$cname" \
    --label "eval_run=${CONTAINER_PREFIX}" \
    --gpus "\"device=${devs}\"" \
    --shm-size=8g --ipc=host \
    -v "$(pwd)/${cell_results}:/workspace/results" \
    -v "$(pwd)/${SRC_DIR}/sglang:/usr/local/lib/python3.12/dist-packages/sglang:ro" \
    -v "$(pwd)/${SRC_DIR}/scripts:/workspace/scripts:ro" \
    -e MODEL_ID="$model_id" \
    -e EVAL_VARIANTS="$EVAL_VARIANTS_FINAL" \
    -e EVAL_TP_SIZES="$tp" \
    -e EVAL_MFS="$mfs" \
    -e EVAL_WORKLOADS="$workloads" \
    -e RUN_EVAL_PROFILE="${RUN_EVAL_PROFILE:-none}" \
    -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
    -e HF_HUB_OFFLINE=1 \
    -e TRANSFORMERS_OFFLINE=1 \
    -e SGLANG_SHARED_POOL_MAPPING_ASSERT="${SGLANG_SHARED_POOL_MAPPING_ASSERT:-1}" \
    -e SGLANG_DISABLE_RADIX_CACHE="${SGLANG_DISABLE_RADIX_CACHE:-0}" \
    -e EVAL_FAIL_FAST="${EVAL_FAIL_FAST:-0}" \
    sglang-hybrid:mamba_swa_eval run_eval \
    > "$cell_log" 2>&1
  local rc=$?

  if [ "$rc" = "0" ]; then
    echo "==> [cell ${cell_idx}: ${cell_name}] OK"
  else
    echo "==> [cell ${cell_idx}: ${cell_name}] FAILED (rc=${rc}) — see ${cell_log}"
  fi
  return "$rc"
}

# ---- 6. Per-model wave runner ----------------------------------------------
#
# Args: $1 slug ("mamba"|"swa"), $2 model_id, $3 friendly description.
#
# Builds the per-model cell list, packs them onto the GPU pool, and
# either launches them in parallel (default) or sequentially. Returns
# the number of cells that failed.
run_one_model() {
  local slug="$1" model_id="$2" desc="$3"
  local skip_var="SKIP_${slug^^}"
  if [ "${!skip_var:-0}" = "1" ]; then
    echo "==> [${slug}] SKIPPED (${skip_var}=1) — ${desc}"
    return 0
  fi

  # Per-model workloads, same precedence as in sync_and_run.sh.
  local workloads
  if [ -n "${EVAL_WORKLOADS:-}" ]; then
    workloads="$EVAL_WORKLOADS"
  elif [ "${DEBUG_RERUN:-0}" = "1" ]; then
    local dvar="DEBUG_WORKLOADS_${slug^^}"
    workloads="${!dvar:-balanced-large}"
  else
    workloads="balanced,balanced-large,skewed,skewed-large"
  fi

  # Slug used by run_eval to build its result-dir name. Keeps cell
  # results inside ${RESULTS_DIR}/${cell_name}/eval_${MODEL_SLUG}_<TS>.
  local model_slug
  model_slug="$(echo "$model_id" | tr '/:' '__' | tr -c 'A-Za-z0-9._-' '_')"

  # Build the cell list as parallel arrays.
  local -a CELL_TPS=()
  local -a CELL_MFS=()
  for tp in "${TP_ARR_FINAL[@]}"; do
    for mfs in "${MFS_ARR_FINAL[@]}"; do
      CELL_TPS+=("$tp")
      CELL_MFS+=("$mfs")
    done
  done
  local n_cells=${#CELL_TPS[@]}

  echo
  echo "==> [${slug}] ${desc}"
  echo "    model      = ${model_id}"
  echo "    workloads  = ${workloads}"
  echo "    cells      = ${n_cells} (= ${EVAL_TP_SIZES_FINAL} tp × ${EVAL_MFS_FINAL} mfs)"
  echo "    GPU pool   = ${GPU_POOL[*]} (size ${GPU_POOL_SIZE})"

  # Schedule each cell in waves. Within a wave: cells run concurrently
  # in the background, each grabbing TP GPUs from the pool until it's
  # exhausted; then we wait, free the pool, start the next wave.
  local next_idx=0
  local cell_seq=0
  local n_failed=0
  local -a wave_pids=()

  while [ "$next_idx" -lt "$n_cells" ]; do
    local pool_used=0
    wave_pids=()

    while [ "$next_idx" -lt "$n_cells" ]; do
      local tp="${CELL_TPS[$next_idx]}"
      if [ $((pool_used + tp)) -gt "$GPU_POOL_SIZE" ]; then
        # No room in this wave for the next cell.
        break
      fi

      # Allocate `tp` consecutive GPUs from the pool.
      local devs=""
      for ((j = 0; j < tp; j++)); do
        if [ -n "$devs" ]; then devs="${devs},"; fi
        devs="${devs}${GPU_POOL[$((pool_used + j))]}"
      done
      pool_used=$((pool_used + tp))

      local mfs="${CELL_MFS[$next_idx]}"

      if [ "$EVAL_PARALLEL" = "1" ]; then
        # Background launch.
        launch_cell "$cell_seq" "$model_slug" "$model_id" "$tp" "$mfs" "$devs" "$workloads" &
        wave_pids+=($!)
      else
        # Sequential — block on each cell.
        launch_cell "$cell_seq" "$model_slug" "$model_id" "$tp" "$mfs" "$devs" "$workloads"
        local rc=$?
        if [ "$rc" != "0" ]; then
          n_failed=$((n_failed + 1))
          if [ "${EVAL_FAIL_FAST:-0}" = "1" ]; then
            echo "==> [${slug}] EVAL_FAIL_FAST=1 — stopping sequential wave after first failure."
            return "$n_failed"
          fi
        fi
      fi

      cell_seq=$((cell_seq + 1))
      next_idx=$((next_idx + 1))
    done

    if [ "$EVAL_PARALLEL" = "1" ] && [ "${#wave_pids[@]}" -gt 0 ]; then
      echo "==> [${slug}] wave running ${#wave_pids[@]} cell(s) in parallel; waiting ..."
      for pid in "${wave_pids[@]}"; do
        if ! wait "$pid"; then
          n_failed=$((n_failed + 1))
        fi
      done
      if [ "$n_failed" -gt 0 ] && [ "${EVAL_FAIL_FAST:-0}" = "1" ]; then
        echo "==> [${slug}] EVAL_FAIL_FAST=1 — stopping after first failed wave."
        return "$n_failed"
      fi
    fi
  done

  echo "==> [${slug}] all cells finished; failed=${n_failed}"
  return "$n_failed"
}

# ---- 7. Stale-container cleanup (LIMITED to this invocation only) ---------
#
# Only kills containers whose label matches THIS run's prefix. Other
# concurrent invocations of this script (different prefix) and any
# unrelated containers on the host are untouched.
cleanup_stale_containers() {
  local stale
  stale="$(docker ps -a -q --filter "label=eval_run=${CONTAINER_PREFIX}" 2>/dev/null)"
  if [ -n "$stale" ]; then
    echo "==> Cleaning up stale containers (label=eval_run=${CONTAINER_PREFIX}):"
    echo "    $(echo "$stale" | tr '\n' ' ')"
    docker rm -f $stale 2>/dev/null || true
  fi
}

trap cleanup_stale_containers EXIT

# ---- 8. Drive the matrix: mamba wave, then swa wave -----------------------

run_one_model mamba "tiiuae/Falcon-H1-7B-Instruct" \
  "Mamba-hybrid (SharedMambaPool path)"
mamba_failed=$?

if [ "${EVAL_FAIL_FAST:-0}" = "1" ] && [ "$mamba_failed" -gt 0 ] \
   && [ "${SKIP_MAMBA:-0}" != "1" ]; then
  echo
  echo "==> EVAL_FAIL_FAST=1 — mamba had ${mamba_failed} failed cell(s); skipping SWA."
  echo "    Inspect ${RESULTS_DIR}/*/run_eval.log."
  exit "$mamba_failed"
fi

run_one_model swa "openai/gpt-oss-20b" \
  "Symmetric SWA-hybrid (SharedSWAKVPool path)"
swa_failed=$?

if [ "${EVAL_FAIL_FAST:-0}" = "1" ] && [ "$swa_failed" -gt 0 ] \
   && [ "${SKIP_SWA:-0}" != "1" ]; then
  echo
  echo "==> EVAL_FAIL_FAST=1 — swa had ${swa_failed} failed cell(s)."
  echo "    Inspect ${RESULTS_DIR}/*/run_eval.log."
  exit "$swa_failed"
fi

total_failed=$((mamba_failed + swa_failed))

echo
echo "==> Eval wave complete."
echo "    Per-cell run_eval logs: ${RESULTS_DIR}/<cell_name>.run_eval.log"
echo "    Per-cell results dirs : ${RESULTS_DIR}/<cell_name>/eval_<model_slug>_<ts>/"
echo "    Failed cells          : ${total_failed}"

if [ "$total_failed" -gt 0 ]; then
  exit 1
fi
exit 0
