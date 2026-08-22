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
"""Hybrid-mamba models must request the stride-aware Triton causal-conv.

Speculative target-verify reconstructs the intermediate conv state, which only
the Triton causal-conv path supports; `MambaMixer2.forward` asserts
`use_triton_causal_conv=True` when speculative decoding is active. A hybrid
model that omits the flag boots fine and then dies at its first verify decode
(Falcon-H1 did). Pin the flag at each model's linear-attn forward call so a
refactor cannot silently drop it.

    python -m pytest test/registered/unit/spec/test_hybrid_mamba_verify_conv_flag.py -v
"""

import ast
import unittest
from pathlib import Path

from sglang.srt import models as _models_pkg
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_MODELS_DIR = Path(next(iter(_models_pkg.__path__)))

# Hybrid-mamba models whose decoder layer calls the shared
# Mamba2AttnBackend.forward; each must pass use_triton_causal_conv=True there.
_HYBRID_MAMBA_MODELS = [
    "falcon_h1.py",
    "granitemoehybrid.py",
    "nemotron_h.py",
]


def _forward_calls_with_flag(path: Path) -> int:
    """Count call sites passing use_triton_causal_conv=True to a .forward()."""
    tree = ast.parse(path.read_text())
    count = 0
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr != "forward":
            continue
        for kw in node.keywords:
            if (
                kw.arg == "use_triton_causal_conv"
                and isinstance(kw.value, ast.Constant)
                and kw.value.value is True
            ):
                count += 1
    return count


class TestHybridMambaVerifyConvFlag(unittest.TestCase):
    def test_every_hybrid_mamba_model_requests_the_triton_conv(self):
        missing = [
            name
            for name in _HYBRID_MAMBA_MODELS
            if _forward_calls_with_flag(_MODELS_DIR / name) == 0
        ]
        self.assertEqual(
            missing,
            [],
            "these hybrid-mamba models never pass use_triton_causal_conv=True "
            "to the linear-attn backend forward; their speculative verify will "
            "trip the MambaMixer2 assert at the first verify decode",
        )


if __name__ == "__main__":
    unittest.main()
