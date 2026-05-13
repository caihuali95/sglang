#!/bin/bash
# Fast iteration helper for the shared-pool test image.
#
# Packages just the `python/sglang/` package tree into a small tarball and
# uploads it to TOS as `sglang-source.tar.gz`. Used together with
# `sync_and_run.sh` on the target host to iterate on Python-only source
# changes WITHOUT rebuilding / re-uploading the full container image.
#
# When to use this:
#   * You changed Python files under python/sglang/ (allocator, mem_cache,
#     scheduler, etc.).
#   * You changed shell helpers under docker/shared-pool-test/scripts/
#     (run_shared.sh, bench_*.sh, launch_*.sh, etc.). The companion
#     sync_and_run.sh on the target bind-mounts both into the running
#     container, so neither needs an image rebuild.
#
# When NOT to use this:
#   * You changed dependencies (pyproject.toml), the Dockerfile, the
#     entrypoint.sh (top-level, NOT the scripts/ helpers), or any other
#     non-Python file outside the two trees above. Run `build_and_upload.sh`
#     instead.
#   * You touched compiled extensions (sgl_kernel, etc.) — those are in
#     separate packages and aren't covered here.
set -euo pipefail

if [ -z "${TOS_BUCKET:-}" ]; then
  echo "ERROR: TOS_BUCKET environment variable is not set." >&2
  exit 1
fi

if [ -z "${WORKSPACE_DIR:-}" ]; then
  echo "ERROR: WORKSPACE_DIR environment variable is not set." >&2
  exit 1
fi
cd "${WORKSPACE_DIR}"

if [ ! -d python/sglang ]; then
  echo "ERROR: python/sglang not found. Run from the sglang repo root." >&2
  exit 1
fi
if [ ! -d docker/shared-pool-test/scripts ]; then
  echo "ERROR: docker/shared-pool-test/scripts not found." >&2
  exit 1
fi

# `TARBALL` is the tarball filename uploaded to TOS. The default
# matches what the companion target-side `sync_and_run.sh` fetches.
#
# To stage a side-channel tarball alongside an in-flight eval (so the
# target host can keep using its existing tarball without disruption),
# override with a different name:
#
#     TARBALL=sglang-source-side.tar.gz ./sync_source.sh
#
# Then on the target side, set the matching env var:
#
#     TARBALL=sglang-source-side.tar.gz \
#         SRC_DIR=./sglang_source_side \
#         RESULTS_DIR=./results_side \
#         ./scripts/sync_and_run.sh
TARBALL="${TARBALL:-sglang-source.tar.gz}"

echo "==> Removing old tarball in TOS: $TARBALL ..."
yes | ../tosutil rm tos://${TOS_BUCKET}/${TARBALL} || true

echo "==> Packing python/sglang/ + docker/shared-pool-test/scripts/ ..."
rm -f "$TARBALL"
# Two top-level entries in the archive:
#   * sglang/  (from python/)        → mounted to dist-packages/sglang
#   * scripts/ (from docker/.../)    → mounted to /workspace/scripts
# tar's `-C` switches the working dir for the FOLLOWING file argument, so
# this gives a flat archive with `sglang/` and `scripts/` at the top.
COPYFILE_DISABLE=1 tar --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='*.pyo' \
    -czf "$TARBALL" \
    -C python sglang \
    -C ../docker/shared-pool-test scripts

ls -lh "$TARBALL"

echo "==> Uploading to TOS ..."
../tosutil cp "./${TARBALL}" tos://${TOS_BUCKET}/

echo
echo "==> Done. On the target host, run:"
if [ "$TARBALL" = "sglang-source.tar.gz" ]; then
  echo "    ./scripts/sync_and_run.sh"
else
  echo "    TARBALL=$TARBALL ./scripts/sync_and_run.sh"
  echo "    # (also set SRC_DIR / RESULTS_DIR if you want a separate workspace.)"
fi
