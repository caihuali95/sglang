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

Three audited arms (see `handle_unified_memory_pool`), each with its own
constraints because each rides a different draft-KV story:
  * DSPARK: chain draft with a private draft pool; verify runs on the MLA
    backend family (`triton` / `trtllm_mla` / `cutedsl_mla` / `tokenspeed_mla`
    / `flashmla` / `flashinfer`)
    and the draft chain must be linear (`--speculative-eagle-topk` in
    {None, 1}).
  * NGRAM: no draft model and no draft KV -- target-verify rails only, on the
    same audited backend set as DSPARK. No chain-shape constraint: the KV
    placement is chain-identical at any bfs breadth (the tree lives in the
    verify custom mask).
  * EAGLE/EAGLE3: hybrid-SWA targets ONLY -- the draft's KV lives fused inside
    the full pool's page envelope (`DenseDraftRegion`), which only the
    hybrid-SWA unified composite provisions; mamba-family targets additionally
    need per-step verify state slots the pool does not provision yet. Chain
    only, and verify is audited on `triton` / `flashinfer` -- demanded
    EXPLICITLY (an unset backend would default to fa3 later in resolution),
    for the draft worker too (its backend resolves separately: explicit flag
    first, else it inherits the target's). The MLA verify backends must not
    leak into this arm, and fa3 does not translate speculative verify
    indices to the dense id space.

Everything else (DFLASH / STANDALONE / registered customs)
stays refused until its verify id rails are audited. Pinned so no arm silently
widens to an unaudited algorithm, family, tree shape, or backend -- and so the
EAGLE arm's addition never perturbs the DSPARK arm.

    python -m pytest test/registered/unit/server_args/test_unified_memory_spec_gate.py -v
"""

import unittest
from types import SimpleNamespace

from sglang.srt.arg_groups.kv_cache_hook import handle_unified_memory_pool
from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _accepts(
    algorithm: str | None,
    *,
    is_hybrid_swa: bool = True,
    topk: int | None = 1,
    backend: str | None = "triton",
    draft_backend: str | None = None,
) -> bool:
    """Run just `handle_unified_memory_pool` against a minimal stand-in.

    ServerArgs' real constructor pulls in a model config; this exercises the
    single handler under test with the fields it reads (disaggregation off, so
    only the speculative / cache / dcp / cuda-graph checks run).
    """
    sa = ServerArgs.__new__(ServerArgs)
    for name, value in {
        "enable_unified_memory": True,
        "disaggregation_mode": "null",
        "speculative_algorithm": algorithm,
        "speculative_eagle_topk": topk,
        "speculative_draft_attention_backend": draft_backend,
        "enable_hierarchical_cache": False,
        "enable_lmcache": False,
        "dcp_size": 1,
        "cuda_graph_config": None,
        "attention_backend": backend,
        "prefill_attention_backend": None,
        "decode_attention_backend": None,
    }.items():
        object.__setattr__(sa, name, value)
    object.__setattr__(
        sa,
        "_model_config",
        SimpleNamespace(
            is_hybrid_swa=is_hybrid_swa,
            attention_arch=AttentionArch.MHA,
        ),
    )
    try:
        handle_unified_memory_pool(sa)
        return True
    except AssertionError:
        return False


class TestUnifiedMemorySpecGate(unittest.TestCase):
    # Verify-audited backends for the DSPARK (MLA-family) arm.
    DSPARK_BACKENDS = (
        "triton",
        "trtllm_mla",
        "cutedsl_mla",
        "tokenspeed_mla",
        "flashmla",
        "flashinfer",
    )
    # Verify-audited backends for the EAGLE (fused-draft) arm.
    EAGLE_BACKENDS = ("triton", "flashinfer")
    # Algorithms with no audited unified-pool verify rails.
    # "NEXTN" is deliberately absent: the CLI alias collapses it to
    # "EAGLE" in handle_speculative_decoding BEFORE this gate runs
    # (arg_groups/pipeline.py orders the hooks), so the raw string can
    # never reach the gate -- a case on it would test an impossible
    # input. The aliased spelling is covered by the EAGLE cases.
    UNAUDITED_ALGORITHMS = ("DFLASH", "STANDALONE")

    def test_eagle_family_admitted_on_hybrid_swa(self):
        """EAGLE/EAGLE3 chain on a hybrid-SWA target with audited verify
        backends is the fused-draft-KV configuration -- both spellings,
        resolved or unset topk, every audited backend."""
        for algorithm in ("EAGLE", "EAGLE3"):
            for topk in (None, 1):
                for backend in self.EAGLE_BACKENDS:
                    self.assertTrue(
                        _accepts(algorithm, topk=topk, backend=backend),
                        f"{algorithm} topk={topk} backend={backend} should "
                        "pass on a hybrid-SWA target",
                    )

    def test_eagle_refused_off_hybrid_swa(self):
        """No fused draft region outside the hybrid-SWA composite: a
        mamba-family (or dense) target must be refused, not fail at boot."""
        for algorithm in ("EAGLE", "EAGLE3"):
            self.assertFalse(_accepts(algorithm, is_hybrid_swa=False))

    def test_eagle_refused_tree_topk(self):
        """Tree verify is not audited for the unified pool; only a linear
        chain passes."""
        for topk in (2, 4, 8):
            self.assertFalse(_accepts("EAGLE", topk=topk))

    def test_eagle_refused_unaudited_backends(self):
        """fa3 does not translate speculative verify indices, and the
        MLA verify set from the DSPARK arm must not leak into the EAGLE arm."""
        for backend in ("fa3", "fa4", "trtllm_mha") + tuple(
            b for b in self.DSPARK_BACKENDS if b not in self.EAGLE_BACKENDS
        ):
            self.assertFalse(_accepts("EAGLE", backend=backend))

    def test_eagle_refused_unset_backend(self):
        """An unset backend defaults to fa3/flashinfer later in resolution --
        the EAGLE arm demands an explicit triton, never a None slip."""
        self.assertFalse(_accepts("EAGLE", backend=None))

    def test_eagle_draft_backend_pinned(self):
        """The draft worker resolves its own backend: unset inherits the
        target's triton, explicit triton passes, anything else refuses."""
        self.assertTrue(_accepts("EAGLE", draft_backend=None))
        for draft_backend in self.EAGLE_BACKENDS:
            self.assertTrue(_accepts("EAGLE", draft_backend=draft_backend))
        for draft_backend in ("fa3", "trtllm_mha"):
            self.assertFalse(_accepts("EAGLE", draft_backend=draft_backend))

    def test_dspark_arm_unchanged(self):
        """The EAGLE addition must not perturb DSPARK: its MLA verify set
        passes, its chain constraint holds, and unaudited backends refuse."""
        for backend in self.DSPARK_BACKENDS:
            self.assertTrue(
                _accepts("DSPARK", backend=backend),
                f"DSPARK should pass on verify-audited backend {backend}",
            )
        self.assertFalse(_accepts("DSPARK", backend="fa3"))
        self.assertFalse(_accepts("DSPARK", topk=4))

    def test_ngram_admitted_on_the_verify_audited_backends(self):
        """NGRAM is target-verify only: the DSPARK backend set passes,
        unaudited backends refuse, and the chain-shape constraint does NOT
        apply (any bfs breadth -- the tree lives in the verify mask, KV
        placement is chain-identical)."""
        for backend in self.DSPARK_BACKENDS:
            self.assertTrue(
                _accepts("NGRAM", backend=backend),
                f"NGRAM should pass on verify-audited backend {backend}",
            )
        self.assertFalse(_accepts("NGRAM", backend="fa3"))
        self.assertTrue(_accepts("NGRAM", topk=4))

    def test_spec_off_admitted(self):
        """The gate constrains only speculative configurations; spec-off must
        keep booting regardless of family."""
        for is_hybrid_swa in (True, False):
            self.assertTrue(_accepts(None, is_hybrid_swa=is_hybrid_swa))

    def test_unaudited_algorithms_refused(self):
        """Every other algorithm stays out until its verify id rails are
        audited -- on every family."""
        for algorithm in self.UNAUDITED_ALGORITHMS:
            for is_hybrid_swa in (True, False):
                self.assertFalse(_accepts(algorithm, is_hybrid_swa=is_hybrid_swa))


if __name__ == "__main__":
    unittest.main()
