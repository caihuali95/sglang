#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

if [ -z "${TOS_BUCKET:-}" ]; then
  echo "ERROR: TOS_BUCKET environment variable is not set." >&2
  exit 1
fi

# Change to the workspace directory
if [ -z "${WORKSPACE_DIR:-}" ]; then
  echo "ERROR: WORKSPACE_DIR environment variable is not set." >&2
  exit 1
fi
cd "${WORKSPACE_DIR}"

echo "Checking HF_TOKEN..."
# HF_TOKEN must be exported by the caller (e.g. `export HF_TOKEN=...` in
# your shell, or sourced from a credentials file you do NOT commit).
# It is consumed only by the prefetch step inside the Docker build to
# download gated models; it is not persisted into any image layer.
if [ -z "${HF_TOKEN:-}" ]; then
  echo "ERROR: HF_TOKEN is not set. Export it before running this script." >&2
  echo "       e.g.  export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx" >&2
  exit 1
fi

echo "Removing sglang.tar.gz in TOS..."
yes | ../tosutil rm tos://${TOS_BUCKET}/sglang.tar.gz || true

echo "Removing old image if it exists..."
docker image rm sglang-hybrid:mamba_swa_eval || true

echo "Building Docker image..."
DOCKER_BUILDKIT=1 docker build \
  --platform linux/amd64 \
  -f docker/shared-pool-test/Dockerfile \
  --build-arg PIP_TRUSTED_HOSTS="pypi.org files.pythonhosted.org" \
  --build-arg INSECURE_DOWNLOADS=1 \
  --build-arg HF_TOKEN="$HF_TOKEN" \
  -t sglang-hybrid:mamba_swa_eval \
  .

echo "Running test command..."
docker run --rm sglang-hybrid:mamba_swa_eval list
docker run --rm sglang-hybrid:mamba_swa_eval shell \
  -c 'python3 -c "from sglang.benchmark.datasets import DatasetRow; print(DatasetRow.__module__)"'

echo "Saving and compressing Docker image..."
# Remove existing file if it exists to avoid gzip asking for confirmation
rm -f sglang.tar sglang.tar.gz
docker save -o sglang.tar sglang-hybrid:mamba_swa_eval

# At this point the image's full byte content lives in sglang.tar — the
# Docker-managed copy under /var/lib/docker is redundant. Drop it BEFORE
# gzip so the host has ~10+ GB of free disk to compress into. (gzip
# in-place rewrites the file, which can briefly need extra headroom on
# tight disks.) The downstream `gzip` and `tosutil cp` only need
# sglang.tar — they don't touch Docker.
echo "Removing the in-Docker image to free disk before gzip..."
docker image rm sglang-hybrid:mamba_swa_eval || true

gzip sglang.tar
echo "Uploading to TOS..."
yes | ../tosutil rm tos://${TOS_BUCKET}/sglang.tar.gz || true
../tosutil cp ./sglang.tar.gz tos://${TOS_BUCKET}/

echo "Successfully built and uploaded the container image to TOS!"
