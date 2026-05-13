# syntax=docker/dockerfile:1.6
#
# Self-contained image for testing --enable-shared-memory-pool. Bundles:
#   - this branch of SGLang (editable install)
#   - profiling tools: nsight-systems-cli, py-spy, memray, gpustat, nvitop
#   - test runner scripts (smoke / bench-* / profile / run-all)
# Optional bake-ins (default on):
#   - one or more model checkpoints
#   - ShareGPT V3 dataset (~50 MB)
#
# Build context MUST be the repo root, e.g.:
#   docker build -f docker/shared-pool-test/Dockerfile -t sglang-shared-test .
#
# Default BAKE_MODELS — every entry is a hybrid model that actually triggers
# a shared-memory-pool code path. (Non-hybrid models like gemma-2-2b-it are
# deliberately NOT included: --enable-shared-memory-pool is a no-op for them
# in SGLang, so they would just bloat the image without testing anything.)
#
#   tiiuae/Falcon-H1-0.5B-Instruct (~1 GB, ungated)
#       Mamba-hybrid. Triggers SharedHybridReqToTokenPool + SharedMambaPool
#       + SharedMambaTokenToKVPoolAllocator. Smallest path to exercise; runs
#       on any GPU.
#   openai/gpt-oss-20b (~12 GB, ungated; ships native MXFP4)
#       SWA-hybrid (symmetric). Triggers SharedSWAKVPool +
#       SharedSWATokenToKVPoolAllocator over two MHA sub-pools with the same
#       head dims. Needs ~24 GB GPU. Used by SGLang's own SWA radix tests.
#   XiaomiMiMo/MiMo-V2-Flash (~14 GB, may be gated)
#       SWA-hybrid (compress). Triggers the heterogeneous-shape path
#       (`is_hybrid_swa_compress`) we just enabled — full and SWA pools
#       have different head dims. Needs ~24 GB GPU. Pass HF_TOKEN if gated.
#
# Total bake ~27 GB + ShareGPT V3 (~50 MB). Override BAKE_MODELS to slim
# down (e.g. just Falcon-H1-0.5B for a CPU-friendly Mamba-only build).
#
# Build-arg examples:
#   --build-arg BAKE_MODELS=""                                    # skip all models
#   --build-arg BAKE_MODELS="tiiuae/Falcon-H1-0.5B-Instruct"     # Mamba only, smallest
#   --build-arg BAKE_DATASETS=0                                   # skip ShareGPT
#   --build-arg HF_TOKEN=hf_xxx                                   # gated models
#
# Build-host network behind an SSL-inspecting corporate proxy (the build
# fails with "[SSL: CERTIFICATE_VERIFY_FAILED]")? Pass:
#   --build-arg PIP_TRUSTED_HOSTS="pypi.org files.pythonhosted.org" \
#   --build-arg INSECURE_DOWNLOADS=1
# That makes pip skip cert verification for those hosts and tells
# huggingface_hub + curl to skip TLS verification during model/dataset
# prefetch. The image artifact itself is not affected — only the build-time
# downloads. For a proper fix, mount your corporate CA cert into the build
# instead (see PIP_INDEX_URL / CA_CERT_PATH usage further below).

ARG SGLANG_BASE_TAG=latest
FROM lmsysorg/sglang:${SGLANG_BASE_TAG}

# EVAL-MODE BAKE
# This image targets the perf-eval workflow (`run_eval` mode):
#   * Mamba-hybrid          → tiiuae/Falcon-H1-7B-Instruct  (~14 GB)
#   * Symmetric SWA-hybrid  → openai/gpt-oss-20b            (~12 GB)
# Total bake ~26 GB. The 7B Falcon-H1 (instead of 0.5B) is chosen so that
# perf measurements actually surface meaningful per-stage costs — the
# 0.5B variant is too small for the bench's bandwidth/compute profile to
# be representative.
#
# Asymmetric SWA (MiMoV2Flash) is intentionally OUT of the eval image —
# the eval target is a baseline-vs-shared comparison on the two main
# paths. Re-add via build-arg if you ever need to perf-eval the compress
# path too.
#
# Override examples:
#   --build-arg BAKE_MODELS="tiiuae/Falcon-H1-3B-Instruct,openai/gpt-oss-20b"
#       (smaller Mamba model — faster load, lighter eval)
#   --build-arg BAKE_MODELS="tiiuae/Falcon-H1-7B-Instruct"
#       (Mamba-only eval; SKIP_SWA=1 on the run side)
ARG BAKE_MODELS=tiiuae/Falcon-H1-7B-Instruct,openai/gpt-oss-20b
ARG BAKE_DATASETS=1
ARG HF_TOKEN=

# Profiling tools (nsys + py-spy + memray + gpustat + nvitop). DEFAULT ON
# for the eval image — `run_eval` with `RUN_EVAL_PROFILE=nsys` needs them
# for the deep-dive shared-pool overhead analysis. Override to 0 for a
# slimmer image without profiling support.
#
# NOTE: INSTALL_PROFILING=1 requires apt-get update to reach Ubuntu
# mirrors. If the build host's apt is firewalled (corp proxy returning
# 401), set to 0 to take the apt-skip path; you'll lose nsys but build
# will succeed.
ARG INSTALL_PROFILING=1

# Network-trust knobs. All optional. Defaults assume direct PyPI / HF access.
ARG PIP_INDEX_URL=
ARG PIP_TRUSTED_HOSTS=
ARG INSECURE_DOWNLOADS=0

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface \
    ASSETS_DIR=/workspace/assets \
    RESULT_DIR=/workspace/results \
    # The base image pins flashinfer at one patch level, but installing this
    # branch's "python[all]" extras can transitively pull a slightly newer
    # flashinfer-jit-cache (e.g. post2 vs post3+cuXXX). The package's strict
    # version check then aborts at import time. Bypass — the patch deltas in
    # the 0.6.x series are ABI-compatible. Override with `-e
    # FLASHINFER_DISABLE_VERSION_CHECK=0` if you need strict matching.
    FLASHINFER_DISABLE_VERSION_CHECK=1

# 0a. Configure pip if we're behind a proxy / mirror. Writes /etc/pip.conf
#     so all subsequent pip invocations (in this Dockerfile and inside the
#     editable install) pick it up.
#
#     NOTE: pip uses Python's configparser, which forbids duplicate keys.
#     Multiple trusted hosts MUST share a single space-separated value:
#         trusted-host = pypi.org files.pythonhosted.org
#     not one line per host. PIP_TRUSTED_HOSTS is already space-separated, so
#     we just emit it verbatim.
RUN if [ -n "$PIP_TRUSTED_HOSTS" ] || [ -n "$PIP_INDEX_URL" ]; then \
      { \
        echo "[global]"; \
        if [ -n "$PIP_INDEX_URL" ]; then echo "index-url = $PIP_INDEX_URL"; fi; \
        if [ -n "$PIP_TRUSTED_HOSTS" ]; then \
          echo "trusted-host = $PIP_TRUSTED_HOSTS"; \
        fi; \
      } > /etc/pip.conf; \
      echo "wrote /etc/pip.conf:"; cat /etc/pip.conf; \
    fi

# 0b. Build-time TLS verification toggle for huggingface_hub + curl. Used
#     ONLY at build time (during prefetch.sh). The runtime container does
#     not need network access, so this leaks no risk past the build.
ENV HF_HUB_DISABLE_SSL_VERIFICATION=${INSECURE_DOWNLOADS}
ENV CURL_INSECURE=${INSECURE_DOWNLOADS}

# 1. Replace the upstream sglang wheel with this branch.
#
# Non-editable install on purpose. Setuptools 64+ "redirect-mode" editable
# installs have a known footgun with our layout (package source under
# python/sglang/ but pyproject.toml at python/): the editable finder ends
# up mapping `sglang -> /workspace/sglang/` (the repo root) instead of
# /workspace/sglang/python/sglang/, so `import sglang.X` for submodules
# added in recent commits (e.g. sglang.benchmark.datasets) silently 404s.
# A regular wheel install copies the package tree into site-packages and
# avoids the issue entirely. We don't gain anything from editable mode here
# because the image isn't rebuilt on the target machine.
#
# IMAGE-SIZE OPTIMIZATION: install from a BuildKit bind mount of the build
# context instead of `COPY . /workspace/sglang`. The bind mount exposes the
# source files only for the duration of this RUN — no files leak into any
# image layer. Compared to the prior COPY approach this saves ~200 MB
# (the .git tree alone is ~150 MB) per image.
#
# Same RUN also nukes:
#   - residual sglang* / __editable__*sglang* / __editable___sglang* under
#     site-packages (the base image `lmsysorg/sglang:latest` ships sglang
#     preinstalled, and `pip uninstall` doesn't always purge metadata).
#   - /opt/sglang and /sgl-workspace/sglang (preinstalled clones in the
#     base image; not used by anything in this harness).
#
# All cleanup must happen in the SAME RUN as `pip install` so the deletions
# apply within the same layer (otherwise we'd just stack a delete-layer on
# top of the bloated base layer, no size win).
WORKDIR /workspace
RUN --mount=type=bind,source=.,target=/build,rw \
    --mount=type=cache,target=/root/.cache/pip \
    pip uninstall -y sglang || true \
 && find /usr/local/lib/python*/dist-packages /usr/lib/python*/dist-packages \
        -maxdepth 1 \( -name 'sglang*' -o -name '__editable__*sglang*' -o -name '__editable___sglang*' \) \
        -exec rm -rf {} + 2>/dev/null || true \
 && rm -rf /opt/sglang /sgl-workspace/sglang \
 && pip install --no-cache-dir "/build/python[all]" \
 && python3 -c "import sglang; assert sglang.__file__, f'sglang installed as namespace pkg: {sglang.__path__}'; print('install ok:', sglang.__file__)" \
 && python3 -c "from sglang.benchmark.datasets import DatasetRow, get_dataset; print('bench import ok:', DatasetRow.__module__)"

# 2. System tools.
#
# `curl` and `ca-certificates` are the runtime always-needed pieces (health
# checks, prefetch.sh dataset download). The lmsysorg/sglang base image
# already ships both, so we deliberately AVOID `apt-get update` on the
# default path — it's a major flake source on hosts behind corporate apt
# proxies that can't reach archive.ubuntu.com (returns 401, breaks build).
# The `command -v curl` guard fails the build loudly if a base image
# without curl is ever swapped in.
#
# Profiling stack (`nsight-systems-cli`, `linux-tools-common`) is several
# hundred MB and only used by entrypoint.sh's `profile` mode. It needs
# apt-get update; opt in with `--build-arg INSTALL_PROFILING=1`.
RUN if [ "$INSTALL_PROFILING" = "1" ]; then \
      apt-get update && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        nsight-systems-cli \
        linux-tools-common \
      && rm -rf /var/lib/apt/lists/*; \
    else \
      command -v curl >/dev/null 2>&1 || { \
        echo "ERROR: base image is missing 'curl', which is required for" >&2; \
        echo "       prefetch.sh / health checks. Either swap to a base" >&2; \
        echo "       image that ships curl, or rebuild with" >&2; \
        echo "       --build-arg INSTALL_PROFILING=1 to install via apt." >&2; \
        exit 1; \
      }; \
      [ -f /etc/ssl/certs/ca-certificates.crt ] || { \
        echo "ERROR: base image is missing ca-certificates. Same options as" >&2; \
        echo "       above (use a base image that has it, or set" >&2; \
        echo "       INSTALL_PROFILING=1)." >&2; \
        exit 1; \
      }; \
      echo "default path: skipping apt; curl + ca-certificates already present in base image"; \
    fi

# Runtime Python dependencies. Only `huggingface_hub` is used by the test
# harness directly (prefetch.sh). pynvml is used by SGLang itself (the base
# image already has it, but pinning here avoids surprises). Profiling-only
# tools (py-spy / memray / gpustat / nvitop) are gated.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir \
        pynvml \
        huggingface_hub \
    && if [ "$INSTALL_PROFILING" = "1" ]; then \
         pip install --no-cache-dir \
             py-spy memray gpustat nvitop; \
       fi

# 3. Test runners (copied early so prefetch.sh is available for step 4).
COPY docker/shared-pool-test/entrypoint.sh /workspace/entrypoint.sh
COPY docker/shared-pool-test/scripts/ /workspace/scripts/
RUN chmod +x /workspace/entrypoint.sh /workspace/scripts/*.sh

# 4. Bake models + datasets into the image. NOTE: do NOT use a build cache
#    mount on $HF_HOME here — cache mounts are not persisted into image
#    layers, which would defeat the whole point of "self-contained image".
#    HF_TOKEN is consumed only here; do NOT echo or persist it.
RUN HF_TOKEN="${HF_TOKEN}" \
    MODELS="${BAKE_MODELS}" \
    DATASETS="${BAKE_DATASETS}" \
    /workspace/scripts/prefetch.sh

# 5. Air-gapped runtime knobs. Set AFTER prefetch so build-time downloads
#    still work, but baked into the image so `docker run` on a network-
#    isolated host doesn't try to call out:
#
#    - HF_HUB_OFFLINE=1: huggingface_hub uses only the local HF cache
#      (no `file_exists` / `hf_hub_download` HTTP calls). This is the
#      standard one for SGLang / transformers >= 0.20.
#    - TRANSFORMERS_OFFLINE=1: legacy transformers code paths honor this
#      instead. Setting both is harmless and bulletproofs either path.
#
#    Override at runtime with `-e HF_HUB_OFFLINE=0` if you ever need the
#    container to reach HF (e.g. to add a model not in the bake).
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

WORKDIR /workspace
ENTRYPOINT ["/workspace/entrypoint.sh"]
CMD ["run_shared"]
