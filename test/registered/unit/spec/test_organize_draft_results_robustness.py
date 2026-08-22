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
"""`organize_draft_results` must emit an ancestor-closed selection.

`build_tree_kernel_efficient` walks each selected token's parent chain on
device; a selection containing a token whose ancestor was NOT selected sends
that walk out of the row or into a spin — a GPU hang with no error. NaN draft
scores are the reachable trigger: NaN is unordered, so a naive topk promotes
NaN candidates and displaces real parents. The function must (a) rank NaN
below every valid score and (b) repair any remaining closure violation.

    python -m pytest test/registered/unit/spec/test_organize_draft_results_robustness.py -v
"""

import unittest

import torch

from sglang.srt.speculative.eagle_utils import organize_draft_results
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

TOPK = 2
STEPS = 3
NUM_DRAFT_TOKEN = 5  # selects num_draft_token - 1 = 4 candidates


def _closure_violations(selected: torch.Tensor, parent_list: torch.Tensor) -> int:
    """Count selected candidates whose parent candidate is not selected."""
    violations = 0
    sel = set(selected[0].tolist())
    for c in sel:
        tb = c // TOPK
        if tb == 0:
            continue  # step-0 candidate: parent is the verified root
        if int(parent_list[0, tb]) not in sel:
            violations += 1
    return violations


def _naive_selection(score_flat: torch.Tensor) -> torch.Tensor:
    # The pre-fix algorithm: plain topk + sort, no NaN handling, no repair.
    idx = torch.topk(score_flat, NUM_DRAFT_TOKEN - 1, dim=-1).indices
    return torch.sort(idx).values


class TestOrganizeDraftResultsRobustness(unittest.TestCase):
    def _inputs(self, poison_nan: bool):
        # Candidate space: step0 = flat [0, 2), later steps = flat [2, 10).
        # parent_list column tb is the parent (flat index) of candidates
        # [tb*TOPK, tb*TOPK + TOPK).
        score_list = [
            torch.tensor([[[0.80, 0.70]]]),  # step0: flat 0, 1
            torch.tensor([[[0.50, 0.05], [0.02, 0.01]]]),  # flat 2..5
            torch.tensor([[[0.90, 0.03], [0.04, 0.02]]]),  # flat 6..9
        ]
        if poison_nan:
            # A childless late candidate turns NaN: naive topk promotes it and
            # displaces flat 2 — the PARENT of the still-selected flat 6.
            score_list[2] = score_list[2].clone()
            score_list[2][0, 1, 0] = float("nan")  # flat 8
        token_list = [
            torch.arange(2).view(1, 2) + 100,
            torch.arange(4).view(1, 4) + 200,
            torch.arange(4).view(1, 4) + 300,
        ]
        # parents_list: last entry unused (dropped by cat(parents_list[:-1])).
        # parent_list = [step0 root markers | parents of flat 6..9] = 6 cols;
        # col 3 = parent of flat {6, 7} = flat 2; col 4 = parent of {8, 9} = 3.
        parents_list = [
            torch.tensor([[0, 0]]),
            torch.tensor([[2, 2, 3, 3]]),
            torch.tensor([[6, 6, 7, 7]]),
        ]
        return score_list, token_list, parents_list

    def test_nan_scores_yield_an_ancestor_closed_selection(self):
        score_list, token_list, parents_list = self._inputs(poison_nan=True)

        # The pre-fix algorithm violates closure on this input (documents the
        # failure mode this function now prevents).
        naive = _naive_selection(torch.cat(score_list, dim=1).flatten(1))
        parent_list_ref = torch.cat(parents_list[:-1], dim=1)
        self.assertGreater(
            _closure_violations(naive, parent_list_ref),
            0,
            "fixture no longer reproduces the naive orphan selection",
        )

        parent_list, selected, draft_tokens = organize_draft_results(
            score_list, token_list, parents_list, NUM_DRAFT_TOKEN, TOPK
        )
        self.assertEqual(_closure_violations(selected, parent_list), 0)
        # NaN must not be selected: flat 8 ranks below every valid score.
        self.assertNotIn(8, selected[0].tolist())
        self.assertEqual(draft_tokens.shape, (1, NUM_DRAFT_TOKEN - 1))

    def test_all_nan_scores_still_terminate_with_valid_indices(self):
        score_list, token_list, parents_list = self._inputs(poison_nan=False)
        score_list = [torch.full_like(s, float("nan")) for s in score_list]
        parent_list, selected, _ = organize_draft_results(
            score_list, token_list, parents_list, NUM_DRAFT_TOKEN, TOPK
        )
        self.assertEqual(_closure_violations(selected, parent_list), 0)
        self.assertTrue(bool((selected >= 0).all()))
        self.assertTrue(bool((selected < 10).all()))

    def test_clean_input_selection_is_unchanged(self):
        score_list, token_list, parents_list = self._inputs(poison_nan=False)
        naive = _naive_selection(torch.cat(score_list, dim=1).flatten(1))
        _, selected, draft_tokens = organize_draft_results(
            score_list, token_list, parents_list, NUM_DRAFT_TOKEN, TOPK
        )
        torch.testing.assert_close(selected, naive, rtol=0, atol=0)
        ss = torch.cat(token_list, dim=1)
        torch.testing.assert_close(
            draft_tokens, torch.gather(ss, index=selected, dim=1), rtol=0, atol=0
        )


if __name__ == "__main__":
    unittest.main()
