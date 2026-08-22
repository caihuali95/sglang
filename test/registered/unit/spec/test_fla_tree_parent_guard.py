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
"""The FLA tree-verify kernels must bounds-guard the parent-state walk.

Tree target-verify derives each step's parent from the draft tree; a corrupted
tree (unset or garbage parent entries, e.g. after NaN draft scores) yields an
out-of-range parent_step_idx, and the unguarded `step_offset = parent_step_idx
* HV * K * V` walks off the intermediate-state allocation — an illegal address
or silent wrong-state reads. mamba_ssm's walk is guarded; pin the same
`>= 0 and < T` guard on the two FLA kernels so a refactor cannot drop it.
(Functional coverage runs on the GPU spec lanes; these kernels need a device.)

    python -m pytest test/registered/unit/spec/test_fla_tree_parent_guard.py -v
"""

import unittest
from pathlib import Path

from sglang.kernels.ops.attention import fla as _fla_pkg
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_FLA_DIR = Path(next(iter(_fla_pkg.__path__)))

_GUARD = "if parent_step_idx >= 0 and parent_step_idx < T:"
_OFFSET = "step_offset = parent_step_idx"

_KERNEL_FILES = [
    "fused_recurrent.py",
    "fused_sigmoid_gating_recurrent.py",
]


class TestFlaTreeParentGuard(unittest.TestCase):
    def test_every_parent_offset_sits_under_the_bounds_guard(self):
        for name in _KERNEL_FILES:
            lines = (_FLA_DIR / name).read_text().splitlines()
            offset_lines = [i for i, l in enumerate(lines) if _OFFSET in l]
            self.assertGreater(
                len(offset_lines), 0, f"{name}: parent-offset site not found"
            )
            for i in offset_lines:
                window = lines[max(0, i - 12) : i]
                self.assertTrue(
                    any(_GUARD in l for l in window),
                    f"{name}:{i + 1}: `{_OFFSET}` is not preceded by the "
                    f"bounds guard `{_GUARD}` — an out-of-range parent walks "
                    "off the intermediate-state allocation",
                )


if __name__ == "__main__":
    unittest.main()
