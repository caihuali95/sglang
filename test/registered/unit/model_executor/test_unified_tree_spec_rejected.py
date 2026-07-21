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
"""Under the unified pool, tree drafting (speculative_eagle_topk > 1) at
page_size > 1 runs the two-pass page-tree cascade, whose draft-decode expand
pass duplicates each branch's prefix-tail page into its first-page holes via a
draft-side ``move_kv_cache`` (``duplicate_prefix_tail_to_draft_branches``). The
fused draft-KV layout has no such draft-side move -- ``UnifiedDraftKVPool``
slots are relocated only by the host pool's whole-envelope move -- so the
cascade would fault mid-run. ``ServerArgs._handle_unified_memory_pool`` rejects
the combination at config time instead. These tests pin that guard: it fires
for tree + large page, and stays silent for every legitimate neighbour (chain
drafting, tree at page_size == 1, DFLASH -- forced to topk == 1 -- at large
page, and spec-off).
"""

import unittest

from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _run_unified_guard(speculative_eagle_topk, page_size, speculative_algorithm="EAGLE"):
    """Bind the real ``_handle_unified_memory_pool`` to a minimal object that
    carries only the attributes the method reads, with every guard BEFORE the
    tree check pre-satisfied, so the call reaches (and only reaches) the
    tree/page-size guard under test. Avoids the heavy full ``__post_init__``."""
    sa = ServerArgs.__new__(ServerArgs)
    sa.enable_unified_memory = True
    sa.disaggregation_mode = "null"
    sa.speculative_algorithm = speculative_algorithm
    sa.speculative_eagle_topk = speculative_eagle_topk
    sa.page_size = page_size
    sa.enable_hierarchical_cache = False
    sa.enable_lmcache = False
    sa.dcp_size = 1
    sa.cuda_graph_config = None  # skips the piecewise-prefill guard
    sa._handle_unified_memory_pool()


class TestUnifiedTreeSpecRejected(CustomTestCase):
    def test_tree_at_large_page_rejected(self):
        """topk > 1 AND page_size > 1 -> loud config-time rejection."""
        for topk, page_size in ((2, 256), (4, 256), (4, 64), (2, 2)):
            with self.subTest(topk=topk, page_size=page_size):
                with self.assertRaisesRegex(
                    ValueError, "tree speculative decoding"
                ):
                    _run_unified_guard(topk, page_size)

    def test_tree_at_page_size_one_allowed(self):
        """The prefix-tail duplication only exists for page_size > 1, so tree
        drafting at page_size == 1 has no partial-tail page to move -> allowed."""
        for topk in (2, 4, 8):
            with self.subTest(topk=topk):
                _run_unified_guard(topk, page_size=1)  # must not raise

    def test_chain_at_large_page_allowed(self):
        """Chain drafting (topk == 1) never runs the cascade -> allowed at any
        page size."""
        for page_size in (1, 64, 256):
            with self.subTest(page_size=page_size):
                _run_unified_guard(1, page_size)  # must not raise

    def test_dflash_topk_one_at_large_page_allowed(self):
        """DFLASH is normalized to topk == 1 by handle_speculative_decoding
        (which runs before this guard), so a DFLASH run at page_size == 256 must
        pass -- it is the shipping product config."""
        _run_unified_guard(1, page_size=256, speculative_algorithm="DFLASH")

    def test_spec_off_at_large_page_allowed(self):
        """Spec-off leaves speculative_eagle_topk as None; the ``(x or 0)`` guard
        must treat that as not-a-tree and stay silent."""
        _run_unified_guard(None, page_size=256, speculative_algorithm=None)


if __name__ == "__main__":
    unittest.main()
