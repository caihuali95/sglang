#!/bin/bash
# Dispatch to a named test mode. See ./scripts/ for the actual runners.
#
# Single-server only. Pin the GPU on `docker run` with `--gpus device=N`;
# multi-server orchestration has been removed.
set -euo pipefail

MODE="${1:-run_shared}"
shift || true

# Defaults can be overridden by `-e` flags on `docker run`.
# MODEL_ID picks which baked model to load. Default = the smallest hybrid
# (Falcon-H1-0.5B-Instruct → Mamba shared path). Override to test SWA paths:
#   -e MODEL_ID=openai/gpt-oss-20b              # SWA shared (symmetric)
#   -e MODEL_ID=XiaomiMiMo/MiMo-V2-Flash        # SWA shared (compress)
export MODEL_ID="${MODEL_ID:-tiiuae/Falcon-H1-0.5B-Instruct}"
export PORT_BASELINE="${PORT_BASELINE:-30000}"
export PORT_SHARED="${PORT_SHARED:-30001}"
export MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.85}"

case "$MODE" in
  shell)
    exec /bin/bash "$@"
    ;;
  prefetch)
    # Add more models / re-download datasets at runtime.
    # Usage: docker run ... prefetch MODELS="repo1,repo2" DATASETS=1
    exec /workspace/scripts/prefetch.sh "$@"
    ;;
  baseline)
    # Launch only — server runs until the container is stopped.
    exec /workspace/scripts/launch_baseline.sh "$MODEL_ID" "$PORT_BASELINE"
    ;;
  shared)
    # Launch only — server runs until the container is stopped.
    exec /workspace/scripts/launch_shared.sh "$MODEL_ID" "$PORT_SHARED"
    ;;
  run_baseline|run-baseline)
    # Launch baseline server, run benchmarks, save results, shut down.
    exec /workspace/scripts/run_baseline.sh
    ;;
  run_shared|run-shared)
    # Launch shared-pool server, run benchmarks, save results, shut down.
    exec /workspace/scripts/run_shared.sh
    ;;
  run_eval|run-eval)
    # Iterate the eval matrix:
    #   variants × tp_sizes × mem_fractions × workloads
    # for $MODEL_ID. Server restarts at each (variant, tp, mfs) boundary.
    # Knobs (env vars, see run_eval.sh for the full list):
    #   EVAL_VARIANTS / EVAL_TP_SIZES / EVAL_MFS / EVAL_WORKLOADS
    #   RUN_EVAL_PROFILE  ("nsys" wraps the server in nsys profile)
    exec /workspace/scripts/run_eval.sh
    ;;
  bench-balanced)
    # Run balanced workload against an externally-managed server. Set
    # PORT=... to point at the right port (default $PORT_SHARED).
    exec /workspace/scripts/bench_balanced.sh
    ;;
  bench-skewed)
    # Run skewed workload against an externally-managed server. Set PORT=...
    exec /workspace/scripts/bench_skewed.sh
    ;;
  profile)
    exec /workspace/scripts/profile_shared.sh "$MODEL_ID"
    ;;
  list)
    echo "Baked models (HF cache at $HF_HOME):"
    if [ -d "$HF_HOME/hub" ]; then
      ls "$HF_HOME/hub" | sed 's/^/  /'
    else
      echo "  (none)"
    fi
    echo
    echo "Baked datasets (in $ASSETS_DIR):"
    if [ -d "$ASSETS_DIR" ]; then
      ls -lh "$ASSETS_DIR" | sed 's/^/  /'
    else
      echo "  (none)"
    fi
    ;;
  *)
    cat <<EOF
unknown mode: $MODE

Available modes (single-server only):
  shell           Drop into bash.
  list            Show baked models / datasets.
  prefetch        Download more models / datasets into a running container.

  baseline        Launch a static-partition server on \$PORT_BASELINE
                  (no benchmark; runs until container stops).
  shared          Launch a --enable-shared-memory-pool server on
                  \$PORT_SHARED (no benchmark; runs until container stops).

  run_baseline    Launch baseline + run the benchmark suite + save
                  results under /workspace/results/baseline_<ts>/ + exit.
  run_shared      Launch shared + run the benchmark suite + save
                  results under /workspace/results/shared_<ts>/ + exit.
  run_eval        Iterate the full eval matrix for \$MODEL_ID:
                  variants {baseline_triton, baseline_native, shared}
                  × tp sizes (\$EVAL_TP_SIZES, default "1,2")
                  × mem-fractions (\$EVAL_MFS, default "0.85,0.55")
                  × workloads (\$EVAL_WORKLOADS, default
                              "balanced,balanced-large,skewed,skewed-large").
                  Outputs runs/*.jsonl + summary.csv + summary.md
                  under /workspace/results/eval_<model>_<ts>/.
                  Set RUN_EVAL_PROFILE=nsys to capture per-cell traces
                  (requires --build-arg INSTALL_PROFILING=1).

  bench-balanced  Run balanced workload against an already-up server
                  (set PORT=... to point at it).
  bench-skewed    Run skewed workload against an already-up server.
  profile         Generate an nsys trace of the shared-pool server
                  under load (single-server, self-managed).
EOF
    exit 1
    ;;
esac
