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
"""`per_step_draft_out_cache_loc` must return CONTIGUOUS per-step rows.

The permute-then-reshape it performs is merge-able, so without an explicit
`.contiguous()` the reshape returns a strided VIEW whose rows have element
stride == num_steps. Torch consumers honour that stride; the Triton store
kernels those rows feed do NOT -- they read the loc with raw pointer arithmetic
(`tl.load(loc_ptr + i)`), walking flat memory. A strided row therefore makes
the kernel consume the step-INTERLEAVED window: row r's KV lands at row
r//num_steps's slot, (num_steps-1)/num_steps of the intended slots are never
written, and (for the last step's row) the flat window runs past the buffer
end into arbitrary memory. This was the fused-EAGLE accept-length regression:
forensics showed the exact 1-correct / (N/steps - 1)-permuted-with-pairs-
(r, steps*r) / rest-UNWRITTEN signature, and repairing the stores restored
baseline accept parity.

Every stride-aware check passes on the strided version -- only contiguity
distinguishes it -- so this pins BOTH the values and the memory layout.

CPU-only: pure tensor layout arithmetic.
"""

import unittest

import torch

from sglang.srt.speculative.eagle_utils import per_step_draft_out_cache_loc
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestPerStepOutCacheLoc(CustomTestCase):
    def _reference(self, flat, bs, topk, steps):
        """Independent layout reference: step i's row = [req0_topk0_stepi,
        req0_topk1_stepi, ..., reqB_topkK_stepi] from the (bs, topk, steps)
        request-major buffer."""
        return flat.view(bs, topk, steps).permute(2, 0, 1).reshape(steps, -1)

    def test_values_and_contiguity(self):
        for bs, topk, steps in ((1, 1, 3), (30, 1, 3), (5, 4, 3), (7, 2, 5)):
            flat = torch.arange(bs * topk * steps, dtype=torch.int64)
            out = per_step_draft_out_cache_loc(flat, bs, topk, steps)
            ref = self._reference(flat, bs, topk, steps)
            with self.subTest(bs=bs, topk=topk, steps=steps):
                self.assertTrue(torch.equal(out, ref))  # values unchanged
                # The load-bearing property: the WHOLE tensor and every
                # per-step row must be contiguous, because the rows are handed
                # to raw-pointer Triton kernels.
                self.assertTrue(out.is_contiguous())
                for i in range(steps):
                    # torch's is_contiguous IS the property the raw-pointer
                    # reader needs (size-1 dims legitimately report an
                    # arbitrary stride, and a 1-element row is trivially safe).
                    self.assertTrue(out[i].is_contiguous())
                    if out[i].numel() > 1:
                        self.assertEqual(out[i].stride(), (1,))

    def test_the_strided_trap_exists_without_contiguous(self):
        # Documents WHY the .contiguous() is there: the raw permute+reshape IS a
        # view with row stride == steps, and its flat memory really is the
        # step-interleaved window the kernel would misread. If torch ever stops
        # returning a view here, the .contiguous() becomes a free no-op and this
        # test can be retired.
        bs, topk, steps = 30, 1, 3
        flat = torch.arange(bs * topk * steps, dtype=torch.int64)
        trap = flat.view(bs, topk, steps).permute(2, 0, 1).reshape(steps, -1)
        self.assertFalse(trap.is_contiguous())
        self.assertEqual(trap[0].stride(), (steps,))
        # What a raw-pointer reader would consume for step 0: flat[0:bs] --
        # the interleave of ALL steps for the first bs//steps requests...
        kernel_would_read = flat[: bs * topk]
        # ...which matches the torch-level row only at position 0 (0 == 0*steps).
        torch_row = trap[0]
        agree = kernel_would_read == torch_row
        self.assertEqual(int(agree.sum()), 1)
        self.assertTrue(bool(agree[0]))


if __name__ == "__main__":
    unittest.main()
