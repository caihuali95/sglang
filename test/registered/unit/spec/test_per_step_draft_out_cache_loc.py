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
"""`per_step_draft_out_cache_loc` must hand out CONTIGUOUS per-step rows.

Its rows are consumed with raw pointer arithmetic (the DSv4 metadata kernel
bakes the step's compression write target from `raw_out_loc_ptr + batch_id`);
the naive permute+reshape is a stride-(1, num_steps) VIEW, so a raw reader
sees step-interleaved locations — silent wrong-slot KV writes, no crash.

    python -m pytest test/registered/unit/spec/test_per_step_draft_out_cache_loc.py -v
"""

import unittest

import torch

from sglang.srt.speculative.eagle_utils import per_step_draft_out_cache_loc
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestPerStepDraftOutCacheLoc(unittest.TestCase):
    def _cases(self):
        return [(1, 1, 2), (3, 1, 4), (2, 4, 3), (5, 8, 7)]

    def test_rows_are_contiguous_for_raw_pointer_readers(self):
        for bs, topk, steps in self._cases():
            loc = torch.arange(bs * topk * steps, dtype=torch.int64)
            per_step = per_step_draft_out_cache_loc(loc, bs, topk, steps)
            for s in range(steps):
                self.assertTrue(
                    per_step[s].is_contiguous(),
                    f"step row must be contiguous (bs={bs}, topk={topk}, "
                    f"steps={steps}, step={s}); a strided row step-interleaves "
                    "raw-pointer readers",
                )

    def test_layout_contract_values_unchanged(self):
        # The materialization must not change WHICH locations each step gets:
        # step s row == buffer.view(bs, topk, steps)[:, :, s].flatten().
        for bs, topk, steps in self._cases():
            loc = torch.arange(bs * topk * steps, dtype=torch.int64)
            per_step = per_step_draft_out_cache_loc(loc, bs, topk, steps)
            ref = loc.view(bs, topk, steps)
            for s in range(steps):
                torch.testing.assert_close(
                    per_step[s], ref[:, :, s].reshape(-1), rtol=0, atol=0
                )


if __name__ == "__main__":
    unittest.main()
