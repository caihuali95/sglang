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
"""`--enable-unified-memory` speculative-decoding allow-list.

Two audited arms (see `_handle_unified_memory_pool`):
  * DSPARK: chain draft with a private draft pool; linear chain only
    (`--speculative-eagle-topk` in {None, 1}) on the spec-verify-audited
    backend set.
  * NGRAM: target-only verify (no draft worker, no draft KV) on mamba-family
    targets ONLY — the family whose target-verify snapshots land in the
    pool's spec-state scratch region. Backends demanded EXPLICITLY from the
    same audited set (an unset backend would default to fa3/flashinfer later
    in resolution, silently leaving the audited envelope).

Everything else stays refused until its verify id rails are audited. Pinned
so no arm silently widens to an unaudited algorithm, family, or backend —
and so the NGRAM arm's addition never perturbs the DSPARK arm.

    python -m pytest test/registered/unit/server_args/test_unified_memory_spec_gate.py -v
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _accepts(
    algorithm: str | None,
    *,
    is_mambaish: bool = True,
    topk: int | None = 1,
    backend: str | None = "triton",
) -> bool:
    """Run just `_handle_unified_memory_pool` against a minimal stand-in.

    ServerArgs' real constructor pulls in a model config; this exercises the
    single handler under test with the fields it reads (disaggregation off,
    so only the speculative / cache / dcp / cuda-graph checks run). The
    mamba-family probe (`mambaish_config`) is patched at its source module —
    the handler imports it at call time.
    """
    sa = ServerArgs.__new__(ServerArgs)
    for name, value in {
        "enable_unified_memory": True,
        "disaggregation_mode": "null",
        "speculative_algorithm": algorithm,
        "speculative_eagle_topk": topk,
        "enable_hierarchical_cache": False,
        "enable_lmcache": False,
        "dcp_size": 1,
        "cuda_graph_config": None,
    }.items():
        object.__setattr__(sa, name, value)
    sa._resolved_attention_backends = lambda: [backend]
    sa.get_model_config = lambda: SimpleNamespace()
    with patch(
        "sglang.srt.configs.hybrid_arch.mambaish_config",
        lambda model_config: object() if is_mambaish else None,
    ):
        try:
            ServerArgs._handle_unified_memory_pool(sa)
            return True
        except AssertionError:
            return False


class TestUnifiedMemorySpecGate(unittest.TestCase):
    # Verify-audited backends (shared by both arms).
    AUDITED_BACKENDS = ("triton", "trtllm_mla", "cutedsl_mla", "tokenspeed_mla")
    # Algorithms with no audited unified-pool verify rails.
    UNAUDITED_ALGORITHMS = ("EAGLE", "EAGLE3", "DFLASH", "STANDALONE", "NEXTN")

    def test_ngram_admitted_on_mamba_family(self):
        """NGRAM on a mamba-family target with an audited, explicitly set
        backend is the spec-state-scratch configuration."""
        for backend in self.AUDITED_BACKENDS:
            self.assertTrue(
                _accepts("NGRAM", backend=backend),
                f"NGRAM should pass on audited backend {backend}",
            )

    def test_ngram_refused_off_mamba_family(self):
        """No spec-state scratch outside the mamba composites: other unified
        families must be refused, not fail at verify time."""
        self.assertFalse(_accepts("NGRAM", is_mambaish=False))

    def test_ngram_refused_unaudited_or_unset_backend(self):
        """fa3/flashinfer do not translate speculative verify indices, and an
        unset backend (None) would default to them later in resolution — the
        NGRAM arm demands an explicit audited backend, never a None slip."""
        for backend in ("fa3", "fa4", "flashinfer", "trtllm_mha", None):
            self.assertFalse(_accepts("NGRAM", backend=backend))

    def test_dspark_arm_unchanged(self):
        """The NGRAM addition must not perturb DSPARK: its audited set
        passes, its chain constraint holds, unaudited backends refuse, and
        it never consults the model family."""
        for backend in self.AUDITED_BACKENDS:
            self.assertTrue(_accepts("DSPARK", backend=backend))
        self.assertTrue(_accepts("DSPARK", is_mambaish=False))
        self.assertFalse(_accepts("DSPARK", backend="fa3"))
        self.assertFalse(_accepts("DSPARK", topk=4))

    def test_spec_off_admitted(self):
        """The gate constrains only speculative configurations; spec-off must
        keep booting regardless of family."""
        for is_mambaish in (True, False):
            self.assertTrue(_accepts(None, is_mambaish=is_mambaish))

    def test_unaudited_algorithms_refused(self):
        """Every other algorithm stays out until its verify id rails are
        audited — on every family."""
        for algorithm in self.UNAUDITED_ALGORITHMS:
            for is_mambaish in (True, False):
                self.assertFalse(_accepts(algorithm, is_mambaish=is_mambaish))


if __name__ == "__main__":
    unittest.main()
