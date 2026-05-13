#!/bin/bash
# Pre-download models and datasets into the image's HF cache + assets dir.
#
# Used at build time (via the Dockerfile) and re-runnable inside the
# container to add more assets after the image is built.
#
# Env / args:
#   MODELS         comma-separated HF repo IDs (e.g.
#                  "google/gemma-2-2b-it,ai21labs/AI21-Jamba-Mini-1.6").
#                  Empty = skip model prefetch.
#   DATASETS       "1" (default) to download ShareGPT, "0" to skip.
#   HF_TOKEN       HuggingFace token for gated models (gemma, llama, etc.).
#   ASSETS_DIR     Where to put the ShareGPT JSON. Default /workspace/assets.

set -euo pipefail

MODELS="${MODELS:-${1:-}}"
DATASETS="${DATASETS:-1}"
ASSETS_DIR="${ASSETS_DIR:-/workspace/assets}"
HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
export HF_HOME

mkdir -p "$ASSETS_DIR" "$HF_HOME"

# ---- Models ----
# IMPORTANT: do NOT ignore *.bin — some HF model repos still ship PyTorch
# pickle weights (no safetensors). Ignoring *.bin would download metadata
# only and cause a runtime "no checkpoint found" failure. We DO ignore
# alternate-framework formats (TF/JAX/TFLite/GGUF) since SGLang doesn't
# consume them.
if [ -n "$MODELS" ]; then
  echo "==> Prefetching models: $MODELS"
  IFS=',' read -ra MODEL_LIST <<< "$MODELS"
  for repo in "${MODEL_LIST[@]}"; do
    repo="$(echo "$repo" | xargs)"  # trim
    [ -z "$repo" ] && continue
    echo "  - $repo"
    python3 - <<PYEOF
import os
from huggingface_hub import snapshot_download
tok = os.environ.get("HF_TOKEN") or None
local_dir = snapshot_download(
    repo_id="$repo",
    token=tok,
    ignore_patterns=["*.h5", "*.msgpack", "*.tflite", "*.gguf"],
)
# Sanity check: at least one weight file must be present. If a repo ships
# only safetensors and we successfully downloaded, we're good. If we got
# only metadata (config.json + tokenizer files) the runtime would fail.
import glob
weight_globs = ["*.safetensors", "*.bin", "*.pt", "*.pth"]
found = []
for g in weight_globs:
    found.extend(glob.glob(os.path.join(local_dir, g)))
    found.extend(glob.glob(os.path.join(local_dir, "*", g)))
if not found:
    raise SystemExit(
        f"prefetch: no weight files found under {local_dir} for $repo. "
        f"Aborting build."
    )
print(f"    OK ({len(found)} weight file(s)) -> {local_dir}")
PYEOF
  done
else
  echo "==> Skipping model prefetch (MODELS empty)"
fi

# ---- Datasets ----
if [ "$DATASETS" = "1" ]; then
  SHAREGPT_URL="${SHAREGPT_URL:-https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json}"
  SHAREGPT_PATH="$ASSETS_DIR/ShareGPT_V3_unfiltered_cleaned_split.json"
  if [ -f "$SHAREGPT_PATH" ]; then
    echo "==> ShareGPT already at $SHAREGPT_PATH (skipping download)"
  else
    echo "==> Downloading ShareGPT to $SHAREGPT_PATH"
    # -L follows redirects (HF uses CDN). When CURL_INSECURE=1 (set by the
    # Dockerfile when building behind an SSL-inspecting proxy), pass -k so
    # curl skips TLS verification — build-time only, the artifact is
    # unaffected.
    curl_args=(-L --fail --retry 3 --retry-delay 5)
    if [ "${CURL_INSECURE:-0}" = "1" ]; then
      curl_args+=(-k)
      echo "    (CURL_INSECURE=1: TLS verification disabled for this download)"
    fi
    curl "${curl_args[@]}" -o "$SHAREGPT_PATH.tmp" "$SHAREGPT_URL"
    mv "$SHAREGPT_PATH.tmp" "$SHAREGPT_PATH"
  fi
else
  echo "==> Skipping dataset prefetch (DATASETS=0)"
fi

echo "==> Prefetch complete."
echo "    Models in:   $HF_HOME/hub/"
echo "    Datasets in: $ASSETS_DIR/"
