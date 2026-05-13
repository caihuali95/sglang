#!/bin/bash
set -euo pipefail

if [ -z "${TOS_BUCKET:-}" ]; then
  echo "ERROR: TOS_BUCKET environment variable is not set." >&2
  exit 1
fi

SKIP_GPT_OSS=0
SKIP_FALCON=0

usage() {
  echo "Usage: $0 [--skip-gpt-oss] [--skip-falcon] <N>" >&2
  echo "Packages the eval results into ./eval_results_N.zip and uploads it." >&2
  echo "" >&2
  echo "Inputs (produced by 'sync_and_run.sh <N>'):" >&2
  echo "  ./results_<N>/<cell_name>/             — per-cell result dir" >&2
  echo "  ./results_<N>/<cell_name>.run_eval.log — per-cell run_eval stdout" >&2
  echo "  ./out.log                              — host-orchestrator stdout" >&2
  echo "" >&2
  echo "  --skip-gpt-oss   Skip gpt-oss-20b cell dirs and *.run_eval.log files." >&2
  echo "  --skip-falcon    Skip Falcon-H1-7B cell dirs and *.run_eval.log files." >&2
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --skip-gpt-oss)
      SKIP_GPT_OSS=1
      shift
      ;;
    --skip-falcon)
      SKIP_FALCON=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    -*)
      echo "ERROR: unknown option '$1'." >&2
      usage
      exit 2
      ;;
    *)
      break
      ;;
  esac
done

if [ "$SKIP_GPT_OSS" = "1" ] && [ "$SKIP_FALCON" = "1" ]; then
  echo "ERROR: --skip-gpt-oss and --skip-falcon together leave nothing to package." >&2
  exit 2
fi

if [ "$#" -ne 1 ]; then
  usage
  exit 2
fi

if ! [[ "$1" =~ ^[0-9]+$ ]]; then
  echo "ERROR: N must be a non-negative integer, got '$1'." >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

RESULTS_DIR="./results_$1"
TARGET_DIR="./eval_results_$1"
TARGET_ZIP="${TARGET_DIR}.zip"
TOSUTIL="./tosutil"
TOS_DEST="tos://${TOS_BUCKET}"

if [ ! -d "$RESULTS_DIR" ]; then
  echo "ERROR: ${RESULTS_DIR} does not exist." >&2
  exit 1
fi

if [ ! -x "$TOSUTIL" ]; then
  echo "ERROR: ${TOSUTIL} is missing or not executable." >&2
  exit 1
fi

if [ -e "$TARGET_DIR" ] || [ -e "$TARGET_ZIP" ]; then
  echo "ERROR: target ${TARGET_DIR} or ${TARGET_ZIP} already exists." >&2
  exit 1
fi

# Layout produced by `sync_and_run.sh <N>` (parallel-cell mode):
#   ./results_<N>/<model_slug>__tp<TP>__mfs<MFS>/
#     ├── eval_<MODEL_SLUG>_<TS>/        (run_eval.sh's per-cell output)
#     │   ├── cell_verdicts.txt
#     │   ├── logs/, runs/, single_prompts/, summary.csv, summary.md, ...
#     ├── (cell-level files: each container writes here)
#   ./results_<N>/<model_slug>__tp<TP>__mfs<MFS>.run_eval.log
#                                         (one per cell — captures the
#                                          container's stdout/stderr)
#
# We package, per skip flag:
#   * cell directories whose name starts with the model's slug
#   * the matching .run_eval.log files
#   * ./out.log (from the host orchestrator)
GPT_OSS_PREFIX="openai_gpt-oss-20b"          # matches model slug from run_eval.sh
FALCON_PREFIX="tiiuae_Falcon-H1-7B-Instruct"

if [ ! -f ./out.log ]; then
  echo "ERROR: ./out.log not found." >&2
  exit 1
fi

# Resolve cell directories per model. Each model produces one directory
# per (tp, mfs) combination, e.g. <slug>__tp1__mfs0.55.
shopt -s nullglob
gpt_oss_cell_dirs=()
falcon_cell_dirs=()
for cell_dir in "$RESULTS_DIR"/*/; do
  cell_dir="${cell_dir%/}"
  base="$(basename "$cell_dir")"
  case "$base" in
    "${GPT_OSS_PREFIX}__"*)
      gpt_oss_cell_dirs+=("$cell_dir")
      ;;
    "${FALCON_PREFIX}__"*)
      falcon_cell_dirs+=("$cell_dir")
      ;;
  esac
done

# Resolve per-cell run_eval logs (sit alongside the cell dirs, with .run_eval.log).
gpt_oss_logs=("$RESULTS_DIR/${GPT_OSS_PREFIX}__"*.run_eval.log)
falcon_logs=("$RESULTS_DIR/${FALCON_PREFIX}__"*.run_eval.log)
shopt -u nullglob

if [ "$SKIP_GPT_OSS" != "1" ] \
   && [ "${#gpt_oss_cell_dirs[@]}" -eq 0 ] \
   && [ "${#gpt_oss_logs[@]}" -eq 0 ]; then
  echo "ERROR: no gpt-oss-20b results found under ${RESULTS_DIR}/. " \
       "(Looked for '${GPT_OSS_PREFIX}__*' cell dirs and " \
       "'${GPT_OSS_PREFIX}__*.run_eval.log' files.)" >&2
  exit 1
fi
if [ "$SKIP_FALCON" != "1" ] \
   && [ "${#falcon_cell_dirs[@]}" -eq 0 ] \
   && [ "${#falcon_logs[@]}" -eq 0 ]; then
  echo "ERROR: no Falcon-H1 results found under ${RESULTS_DIR}/. " \
       "(Looked for '${FALCON_PREFIX}__*' cell dirs and " \
       "'${FALCON_PREFIX}__*.run_eval.log' files.)" >&2
  exit 1
fi

mkdir "$TARGET_DIR"
ls "$RESULTS_DIR"

if [ "$SKIP_GPT_OSS" != "1" ]; then
  for d in "${gpt_oss_cell_dirs[@]}"; do
    cp -r "$d" "$TARGET_DIR/"
  done
  for f in "${gpt_oss_logs[@]}"; do
    cp "$f" "$TARGET_DIR/"
  done
fi
if [ "$SKIP_FALCON" != "1" ]; then
  for d in "${falcon_cell_dirs[@]}"; do
    cp -r "$d" "$TARGET_DIR/"
  done
  for f in "${falcon_logs[@]}"; do
    cp "$f" "$TARGET_DIR/"
  done
fi
cp ./out.log "$TARGET_DIR/"
ls "$TARGET_DIR"
zip -r "$TARGET_ZIP" "$TARGET_DIR"
"$TOSUTIL" cp "$TARGET_ZIP" "$TOS_DEST"
