#!/bin/bash
# Eval-mode orchestrator. Runs ONE model through the cross-product of
# variants × TP sizes × MEM_FRACTION values × workloads, in a single
# container. The server is restarted at each (variant, tp, mfs) boundary
# (each combination needs its own server config); workloads run inside
# the same server.
#
# Variants captured (single comparison, three columns):
#   baseline_triton  static partition + Triton tiled KV-copy kernel
#   baseline_native  static partition + native (Python/torch) KV-copy
#   shared           shared pool      + native (forced by subclass override)
#
# Pairwise deltas:
#   triton  → native  : pure GPU-kernel-loss cost
#   native  → shared  : pure shared-pool-bookkeeping cost
#   triton  → shared  : total shared-pool cost
#
# Required env:
#   MODEL_ID              model to evaluate (no default — caller must set)
#
# Optional env (with defaults):
#   EVAL_VARIANTS         "baseline_triton,baseline_native,shared"
#   EVAL_TP_SIZES         "1,2"
#   EVAL_MFS              "0.85,0.55"
#   EVAL_WORKLOADS        "balanced,balanced-large,skewed,skewed-large"
#   EVAL_PORT             30001
#   EVAL_OUTPUT_ROOT      /workspace/results
#   EVAL_FAIL_FAST        "0" (default) | "1"
#                          When "1": stop the whole sweep as soon as ANY cell
#                          either crashes or has a workload-level failure.
#                          The verdict table + summary are still emitted so
#                          you can see exactly which cell broke. Useful for
#                          fast iteration on a hypothesis: a single
#                          ~1-min cell is enough; you don't have to wait for
#                          the full ~8-min broad sweep.
#   RUN_EVAL_PROFILE      "none" | "nsys"
#                          When "nsys": wraps the FIRST workload of each
#                          (variant, tp, mfs) under nsys profile, writing
#                          per-cell .nsys-rep traces under profiles/.
#                          Requires INSTALL_PROFILING=1 in the build.
#
# Workload definitions (input_len / output_len / num_prompts / concurrency):
#   balanced        1024 /  256 / 200 / 32
#   balanced-large  2048 /  512 / 200 / 64
#   skewed          4096 /   64 / 200 / 64
#   skewed-large    8192 /   64 / 100 / 64
#
# Output layout:
#   $EVAL_OUTPUT_ROOT/eval_<MODEL_SLUG>_<TS>/
#     runs/<model>__<variant>__tp<N>__mfs<MFS>__<workload>.jsonl
#     runs/<model>__<variant>__tp<N>__mfs<MFS>__<workload>.bench.txt
#     logs/<model>__<variant>__tp<N>__mfs<MFS>__server.log
#     single_prompts/<model>__<variant>__tp<N>__mfs<MFS>__single_prompt.txt
#     profiles/<model>__<variant>__tp<N>__mfs<MFS>.nsys-rep   (only if profiled)
#     summary.csv
#     summary.md
#
# Per-cell failures (server timeout, OOM, bench crash) are logged and the
# matrix continues. Final summary builds from whatever JSONLs exist.
set -uo pipefail

# ---- 1. Parameters ----
if [ -z "${MODEL_ID:-}" ]; then
  echo "ERROR: MODEL_ID env var is required." >&2
  exit 1
fi
MODEL="$MODEL_ID"
PORT="${EVAL_PORT:-30001}"
OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-/workspace/results}"
EVAL_VARIANTS="${EVAL_VARIANTS:-baseline_triton,baseline_native,shared}"
EVAL_TP_SIZES="${EVAL_TP_SIZES:-1,2}"
EVAL_MFS="${EVAL_MFS:-0.85,0.55}"
EVAL_WORKLOADS="${EVAL_WORKLOADS:-balanced,balanced-large,skewed,skewed-large}"
RUN_EVAL_PROFILE="${RUN_EVAL_PROFILE:-none}"
EVAL_FAIL_FAST="${EVAL_FAIL_FAST:-0}"

# Slug the model id for filenames.
MODEL_SLUG="$(echo "$MODEL" | tr '/:' '__' | tr -c 'A-Za-z0-9._-' '_')"

TS="$(date +%Y%m%d-%H%M%S)"
RESULT_DIR="$OUTPUT_ROOT/eval_${MODEL_SLUG}_${TS}"
RUNS_DIR="$RESULT_DIR/runs"
LOG_DIR="$RESULT_DIR/logs"
SP_DIR="$RESULT_DIR/single_prompts"
PROFILE_DIR="$RESULT_DIR/profiles"
mkdir -p "$RUNS_DIR" "$LOG_DIR" "$SP_DIR"
[ "$RUN_EVAL_PROFILE" = "nsys" ] && mkdir -p "$PROFILE_DIR"

echo "==> Eval result dir : $RESULT_DIR"
echo "==> Model           : $MODEL"
echo "==> Variants        : $EVAL_VARIANTS"
echo "==> TP sizes        : $EVAL_TP_SIZES"
echo "==> MEM fractions   : $EVAL_MFS"
echo "==> Workloads       : $EVAL_WORKLOADS"
echo "==> Profile mode    : $RUN_EVAL_PROFILE"
echo "==> Fail-fast       : $EVAL_FAIL_FAST"

# ---- 2. Workload table (input_len / output_len / num_prompts / concurrency) ----
declare -A WORKLOAD_INPUT_LEN
declare -A WORKLOAD_OUTPUT_LEN
declare -A WORKLOAD_NUM_PROMPTS
declare -A WORKLOAD_CONCURRENCY
WORKLOAD_INPUT_LEN[balanced]=1024;        WORKLOAD_OUTPUT_LEN[balanced]=256
WORKLOAD_NUM_PROMPTS[balanced]=200;       WORKLOAD_CONCURRENCY[balanced]=32
WORKLOAD_INPUT_LEN[balanced-large]=2048;  WORKLOAD_OUTPUT_LEN[balanced-large]=512
WORKLOAD_NUM_PROMPTS[balanced-large]=200; WORKLOAD_CONCURRENCY[balanced-large]=64
WORKLOAD_INPUT_LEN[skewed]=4096;          WORKLOAD_OUTPUT_LEN[skewed]=64
WORKLOAD_NUM_PROMPTS[skewed]=200;         WORKLOAD_CONCURRENCY[skewed]=64
WORKLOAD_INPUT_LEN[skewed-large]=8192;    WORKLOAD_OUTPUT_LEN[skewed-large]=64
WORKLOAD_NUM_PROMPTS[skewed-large]=100;   WORKLOAD_CONCURRENCY[skewed-large]=64

# ---- 3. Helpers ----
DATASET_PATH="${DATASET_PATH:-/workspace/assets/ShareGPT_V3_unfiltered_cleaned_split.json}"

_wait_ready() {
  local port="$1" timeout_s="$2"
  local i=0
  while [ "$i" -lt "$timeout_s" ]; do
    if curl -sf "http://127.0.0.1:${port}/health_generate" > /dev/null 2>&1; then
      return 0
    fi
    sleep 2
    i=$((i + 2))
  done
  return 1
}

_dump_server_tail_on_fail() {
  local rc="$1" cell="$2" log_file="$3"
  if [ "$rc" != "0" ]; then
    echo
    echo "==> [${cell}] FAILED (rc=${rc}). Last 120 lines of server log:"
    tail -n 120 "$log_file" 2>/dev/null | sed 's/^/    /' || true
    echo "    --- end ---"
  fi
}

# Returns 0 if the server PID is alive AND /health_generate responds. Else 1.
# Used between workloads to short-circuit a dead-server cell instead of
# wasting time on bench clients that all fail with TransferEncodingError.
_check_server_alive() {
  local port="$1" pid="$2"
  if [ -z "$pid" ]; then
    return 1
  fi
  if ! kill -0 "$pid" 2>/dev/null; then
    return 1
  fi
  # 3-second budget — between workloads, we want a fast verdict.
  if curl -sf --max-time 3 "http://127.0.0.1:${port}/health_generate" \
       > /dev/null 2>&1; then
    return 0
  fi
  return 1
}

# Snapshot GPU state to a sidecar file. Useful at failure time to
# disambiguate OOM vs allocator-state bug (the former shows ≥99% memory
# used; the latter typically much less).
_snapshot_gpu() {
  local out_file="$1"
  {
    date -u "+# captured-at: %Y-%m-%dT%H:%M:%SZ"
    if command -v nvidia-smi >/dev/null 2>&1; then
      nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
                 --format=csv 2>/dev/null
      echo
      nvidia-smi --query-compute-apps=pid,process_name,used_memory \
                 --format=csv 2>/dev/null
    else
      echo "# nvidia-smi unavailable"
    fi
  } > "$out_file" 2>&1 || true
}

_run_single_prompt() {
  local port="$1" out_file="$2"
  curl -s -X POST "http://127.0.0.1:${port}/generate" \
    -H "Content-Type: application/json" \
    -d '{"text":"The capital of France is","sampling_params":{"max_new_tokens":16,"temperature":0}}' \
    | python3 -c 'import sys,json; print(json.load(sys.stdin)["text"])' \
    > "$out_file" 2>&1 || true
}

_run_one_workload() {
  # Args: port, workload_name, jsonl_path, txt_path
  local port="$1" wname="$2" jsonl="$3" txt="$4"
  local input_len="${WORKLOAD_INPUT_LEN[$wname]}"
  local output_len="${WORKLOAD_OUTPUT_LEN[$wname]}"
  local num_prompts="${WORKLOAD_NUM_PROMPTS[$wname]}"
  local concurrency="${WORKLOAD_CONCURRENCY[$wname]}"

  local extra=()
  if [ -f "$DATASET_PATH" ]; then
    extra+=(--dataset-path "$DATASET_PATH")
  fi

  set +e
  python3 -m sglang.bench_serving \
    --backend sglang \
    --host 127.0.0.1 \
    --port "$port" \
    --dataset-name random \
    --random-input-len "$input_len" \
    --random-output-len "$output_len" \
    --num-prompts "$num_prompts" \
    --max-concurrency "$concurrency" \
    --output-file "$jsonl" \
    "${extra[@]}" \
    2>&1 | tee "$txt"
  local rc=${PIPESTATUS[0]}
  set -e
  return "$rc"
}

# ---- 4. The main loop. ----
# Server-restart count = |variants| × |tp_sizes| × |mfs|
IFS=',' read -ra VARIANTS_ARR <<< "$EVAL_VARIANTS"
IFS=',' read -ra TP_ARR       <<< "$EVAL_TP_SIZES"
IFS=',' read -ra MFS_ARR      <<< "$EVAL_MFS"
IFS=',' read -ra WORKLOAD_ARR <<< "$EVAL_WORKLOADS"

set -e

# Per-cell verdicts collected during the run; printed at the end so a
# 48-cell sweep doesn't bury the outcome in 10k lines of bench output.
declare -A CELL_VERDICT      # cell -> "ok" | "crashed" | "partial"
declare -A CELL_OK_COUNT     # cell -> # workloads that succeeded
declare -A CELL_FAIL_COUNT   # cell -> # workloads that failed
declare -A CELL_SERVER_LOG   # cell -> path to its server log

# Set when EVAL_FAIL_FAST=1 and a cell has crashed/partially-failed. After
# a failed cell, any later cells are skipped via `continue` → the loop ends
# quickly and the verdict table / summary still get emitted. We don't `break`
# nested loops directly because the outer two are independent — using a flag
# keeps the skip path uniform.
FAIL_FAST_TRIGGERED=0

for tp in "${TP_ARR[@]}"; do
  for mfs in "${MFS_ARR[@]}"; do
    for variant in "${VARIANTS_ARR[@]}"; do
      if [ "$FAIL_FAST_TRIGGERED" = "1" ]; then
        break 3
      fi
      cell="${MODEL_SLUG}__${variant}__tp${tp}__mfs${mfs}"
      server_log="$LOG_DIR/${cell}__server.log"
      sp_file="$SP_DIR/${cell}__single_prompt.txt"

      echo
      echo "==> ============================================================"
      echo "==> Cell: $cell"
      echo "==> ============================================================"

      # Resolve launcher + KV-copy env per variant.
      case "$variant" in
        baseline_triton)
          launcher=/workspace/scripts/launch_baseline.sh
          export SGLANG_NATIVE_MOVE_KV_CACHE=0
          ;;
        baseline_native)
          launcher=/workspace/scripts/launch_baseline.sh
          export SGLANG_NATIVE_MOVE_KV_CACHE=1
          ;;
        shared)
          launcher=/workspace/scripts/launch_shared.sh
          # SGLang env doesn't matter for shared (subclass overrides force
          # native), but unset for cleanliness.
          unset SGLANG_NATIVE_MOVE_KV_CACHE
          ;;
        *)
          echo "==> [${cell}] UNKNOWN variant '$variant'; skipping" >&2
          continue
          ;;
      esac

      # Optionally wrap server start under nsys for the FIRST workload.
      profile_arg=""
      if [ "$RUN_EVAL_PROFILE" = "nsys" ]; then
        nsys_out="$PROFILE_DIR/${cell}"
        # Generate trace ONLY for the first workload by setting a marker.
        # We do this by wrapping the server in nsys; the server runs the
        # full workload set under the trace.
        set +e
        command -v nsys >/dev/null 2>&1 && profile_arg=1 || profile_arg=0
        set -e
        if [ "$profile_arg" = "0" ]; then
          echo "==> [${cell}] WARN: RUN_EVAL_PROFILE=nsys but nsys not found"
          echo "                  rebuild with --build-arg INSTALL_PROFILING=1"
          profile_arg=""
        fi
      fi

      # Launch server.
      echo "==> [${cell}] launching server (variant=${variant}, tp=${tp}, mfs=${mfs}) ..."
      set +e
      if [ -n "$profile_arg" ]; then
        nsys profile \
          --output "$PROFILE_DIR/${cell}" \
          --trace=cuda,nvtx,osrt,cudnn \
          --force-overwrite=true \
          -- \
          env TP_SIZE="$tp" MEM_FRACTION_STATIC="$mfs" \
            bash "$launcher" "$MODEL" "$PORT" \
          > "$server_log" 2>&1 &
      else
        env TP_SIZE="$tp" MEM_FRACTION_STATIC="$mfs" \
          bash "$launcher" "$MODEL" "$PORT" \
          > "$server_log" 2>&1 &
      fi
      SERVER_PID=$!
      set -e

      # Per-cell EXIT trap to ensure the server stops between cells.
      _stop_server() {
        if [ -n "${SERVER_PID:-}" ]; then
          kill "$SERVER_PID" 2>/dev/null || true
          # Give it a few seconds to flush; then SIGKILL if still alive.
          for i in 1 2 3 4 5; do
            kill -0 "$SERVER_PID" 2>/dev/null || break
            sleep 1
          done
          kill -9 "$SERVER_PID" 2>/dev/null || true
          wait "$SERVER_PID" 2>/dev/null || true
          unset SERVER_PID
        fi
      }
      trap _stop_server RETURN

      # Wait for ready (longer timeout for big models / cold start).
      if ! _wait_ready "$PORT" 600; then
        echo "==> [${cell}] server did not come up within 600s — skipping cell"
        _dump_server_tail_on_fail 1 "$cell" "$server_log"
        _stop_server
        continue
      fi

      # Single-prompt sanity (per cell).
      _run_single_prompt "$PORT" "$sp_file"

      # Iterate workloads inside this server's lifetime.
      cell_dead=0
      cell_ok_workloads=0
      cell_failed_workloads=0
      for wname in "${WORKLOAD_ARR[@]}"; do
        run_slug="${cell}__${wname}"
        jsonl="$RUNS_DIR/${run_slug}.jsonl"
        txt="$RUNS_DIR/${run_slug}.bench.txt"

        # Skip remaining workloads in this cell if a previous workload
        # killed the server. Continuing would just produce a stream of
        # TransferEncodingError noise from the bench client and burn
        # 60+ seconds per workload waiting for tcp resets.
        if [ "$cell_dead" = "1" ]; then
          echo "==> [${run_slug}] SKIPPED — server died earlier in this cell"
          cell_failed_workloads=$((cell_failed_workloads + 1))
          continue
        fi

        echo "==> [${cell}] workload=${wname} -> $(basename $jsonl)"
        rc=0
        _run_one_workload "$PORT" "$wname" "$jsonl" "$txt" || rc=$?
        if [ "$rc" != "0" ]; then
          echo "==> [${run_slug}] bench rc=$rc (data may still be in $jsonl)"
          # Determine whether the server is still up. If it crashed (which
          # is the most common cause of mid-stream TransferEncodingError on
          # the client), surface the server traceback and skip the rest of
          # this cell.
          if ! _check_server_alive "$PORT" "${SERVER_PID:-}"; then
            cell_dead=1
            gpu_snap="$LOG_DIR/${cell}__gpu_at_crash.txt"
            _snapshot_gpu "$gpu_snap"
            echo "==> [${run_slug}] server is DEAD after this workload."
            echo "==> [${run_slug}] GPU snapshot saved to: $gpu_snap"
            _dump_server_tail_on_fail "$rc" "$run_slug" "$server_log"
          else
            echo "==> [${run_slug}] server still responsive — bench-side flake; continuing"
          fi
          cell_failed_workloads=$((cell_failed_workloads + 1))
        else
          cell_ok_workloads=$((cell_ok_workloads + 1))
        fi
      done

      _stop_server
      trap - RETURN

      CELL_OK_COUNT[$cell]=$cell_ok_workloads
      CELL_FAIL_COUNT[$cell]=$cell_failed_workloads
      CELL_SERVER_LOG[$cell]="$server_log"
      if [ "$cell_dead" = "1" ]; then
        CELL_VERDICT[$cell]="crashed"
        echo "==> [${cell}] done (server crashed; ok=${cell_ok_workloads} failed=${cell_failed_workloads})"
      elif [ "$cell_failed_workloads" != "0" ]; then
        CELL_VERDICT[$cell]="partial"
        echo "==> [${cell}] done (partial; ok=${cell_ok_workloads} failed=${cell_failed_workloads})"
      else
        CELL_VERDICT[$cell]="ok"
        echo "==> [${cell}] done (ok=${cell_ok_workloads} failed=${cell_failed_workloads})"
      fi

      # Fail-fast: the moment a cell didn't fully succeed, mark and skip the
      # remaining cells. The verdict table + summary still build below from
      # whatever ran. Useful when the user is iterating on a single
      # hypothesis and just needs the first failure to land.
      if [ "$EVAL_FAIL_FAST" = "1" ] && [ "${CELL_VERDICT[$cell]}" != "ok" ]; then
        echo "==> [${cell}] EVAL_FAIL_FAST=1 — stopping the sweep after this cell."
        FAIL_FAST_TRIGGERED=1
        sleep 5
        break 3
      fi

      # Brief inter-cell grace: give CUDA / network sockets time to
      # release. Without this we've seen the next cell's server come up
      # against partially-freed GPU memory.
      sleep 5
    done
  done
done

# ---- 5. Per-cell verdict table ----
# Surfaces which cells fully succeeded, partially succeeded, or crashed.
# Without this, in a 48-cell sweep a single mid-run server crash is buried
# under thousands of bench-output lines.
verdict_file="$RESULT_DIR/cell_verdicts.txt"
{
  echo "# Per-cell verdicts (populated as the eval runs)"
  echo "# Columns: verdict cell ok_workloads failed_workloads server_log"
  for cell in "${!CELL_VERDICT[@]}"; do
    printf "%-8s %s ok=%s fail=%s log=%s\n" \
      "${CELL_VERDICT[$cell]}" \
      "$cell" \
      "${CELL_OK_COUNT[$cell]}" \
      "${CELL_FAIL_COUNT[$cell]}" \
      "${CELL_SERVER_LOG[$cell]}"
  done | sort
} > "$verdict_file"

echo
echo "==> Per-cell verdicts -> $verdict_file"
cat "$verdict_file"

# Highlight crashed cells with the location of their server logs so the
# user knows exactly where to look.
crashed_any=0
for cell in "${!CELL_VERDICT[@]}"; do
  if [ "${CELL_VERDICT[$cell]}" = "crashed" ]; then
    crashed_any=1
    break
  fi
done
if [ "$crashed_any" = "1" ]; then
  echo
  echo "==> CRASHED cells (inspect server logs for the actual traceback):"
  for cell in "${!CELL_VERDICT[@]}"; do
    if [ "${CELL_VERDICT[$cell]}" = "crashed" ]; then
      echo "    ${CELL_SERVER_LOG[$cell]}"
    fi
  done | sort -u
fi

# ---- 6. Build summary ----
echo
echo "==> Building summary tables ..."
set +e
python3 /workspace/scripts/compare_eval_results.py "$RESULT_DIR" \
  | tee "$RESULT_DIR/summary.console.txt"
set -e

echo
echo "==> Eval complete. All artifacts under: $RESULT_DIR"
ls -la "$RESULT_DIR"

# Fail-fast: propagate failure to the caller so the outer wrapper
# (`sync_and_run.sh`) can skip the next model in the sweep.
if [ "$EVAL_FAIL_FAST" = "1" ] && [ "$FAIL_FAST_TRIGGERED" = "1" ]; then
  echo "==> EVAL_FAIL_FAST=1 triggered; exiting with rc=2."
  exit 2
fi
