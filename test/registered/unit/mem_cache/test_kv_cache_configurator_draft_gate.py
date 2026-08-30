# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""The unified-pool fast path is target-only.

BUG REGRESSION. The fast path was gated on `req_to_token_pool is None` alone,
but a compact-window DFLASH draft (`--speculative-draft-window-size`) also
passes None — it builds a private req_to_token of its own — so the draft
worker would allocate a SECOND unified byte buffer at boot. A draft-shaped
configurator must fall through to the normal pool build instead.

    python -m pytest test/registered/unit/mem_cache/test_kv_cache_configurator_draft_gate.py -v
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.mem_cache import kv_cache_configurator as kcc
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _ReachedNormalBuild(Exception):
    """Sentinel: control flow fell past the unified fast path."""


class TestUnifiedFastPathDraftGate(CustomTestCase):
    def _run(self, *, is_draft_worker: bool):
        cfg = kcc.KVCacheConfigurator.__new__(kcc.KVCacheConfigurator)
        cfg.is_draft_worker = is_draft_worker
        cfg.mambaish_config = object()  # would select the mamba unified arm
        sizes = SimpleNamespace(max_running_requests=8, max_total_num_tokens=64)
        taken = []
        with (
            patch.object(
                kcc,
                "get_memory",
                return_value=SimpleNamespace(enable_unified_memory=True),
            ),
            patch.object(
                kcc,
                "get_disagg",
                return_value=SimpleNamespace(disaggregation_mode="null"),
            ),
            patch.object(
                kcc.KVCacheConfigurator,
                "_init_unified_mamba_pools",
                lambda self, **kw: taken.append("mamba") or None,
            ),
            patch.object(
                kcc.KVCacheConfigurator,
                "_init_unified_swa_pools",
                lambda self, **kw: taken.append("swa") or None,
            ),
            patch.object(
                kcc.KVCacheConfigurator,
                "_build_req_to_token_pool",
                side_effect=_ReachedNormalBuild,
            ),
        ):
            try:
                cfg._init_pools(
                    sizes=sizes,
                    req_to_token_pool=None,
                    token_to_kv_pool_allocator=None,
                )
            except _ReachedNormalBuild:
                return "normal", taken
            except (AttributeError, TypeError):
                # The stubbed unified arm returned None; anything past the
                # fast-path dispatch counts as having taken it.
                return "unified", taken
        return "unified", taken

    def test_draft_worker_falls_through_to_the_normal_build(self):
        path, taken = self._run(is_draft_worker=True)
        self.assertEqual(path, "normal")
        self.assertEqual(taken, [])

    def test_target_worker_still_takes_the_fast_path(self):
        """The guard must narrow to drafts only — a target regression here
        silently turns --enable-unified-memory into a no-op."""
        path, taken = self._run(is_draft_worker=False)
        self.assertEqual(taken, ["mamba"])


if __name__ == "__main__":
    unittest.main()
