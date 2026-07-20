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
"""KDA speculative-decoding layout + capability-gate contracts.

Three invariants that make KDA target-verify safe, each of which a plausible
refactor could silently break:

1. KDA conv shapes are stored SWAPPED, ``(win, dim)``, versus GDN's
   ``(dim, win)`` — the reason the dedup conv-window layout (which unpacks
   ``conv_dim, win = conv_shape``) must NOT be used for KDA.
2. ``conv_window_dedup_enabled`` therefore returns False whenever
   ``is_kda=True`` (dense fallback), while GDN chain keeps dedup.
3. The boot-time spec capability gate probes the kernel dispatcher's
   ``target_verify`` METHOD (duck-typed like the commit dispatch); GDN's and
   KDA's dispatchers must both satisfy it, and a dispatcher without the method
   must be rejected loudly — never allowed to reach the silent-corruption
   verify path.
"""

import unittest
from types import SimpleNamespace

from sglang.srt.configs.mamba_utils import KimiLinearStateShape, Mamba2StateShape
from sglang.srt.mem_cache.memory_pool import conv_window_dedup_enabled
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

try:
    from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
        MambaAttnBackendBase,
    )
    from sglang.srt.layers.attention.linear.gdn_backend import GDNKernelDispatcher
    from sglang.srt.layers.attention.linear.kda_backend import KDAKernelDispatcher

    BACKENDS_IMPORTABLE = True
except ImportError:
    BACKENDS_IMPORTABLE = False


class TestKDAConvShapeOrientation(CustomTestCase):
    def test_kda_conv_shape_is_win_major(self):
        """KimiLinearStateShape stores conv as (win, dim) — the transposed
        orientation every KDA kernel call site compensates for. If this ever
        flips to GDN's (dim, win), the dedup exclusion and the verify branch's
        transposes both become wrong — this pin forces that change through
        review."""
        shape = KimiLinearStateShape.create(
            tp_world_size=1,
            num_heads=32,
            head_dim=128,
            conv_kernel_size=4,
        )
        (conv_shape,) = shape.conv
        win, dim = conv_shape
        self.assertEqual(win, 4 - 1)  # conv_kernel - 1
        self.assertEqual(dim, 32 * 128 * 3)  # q + k + v projections

    def test_gdn_conv_shape_is_dim_major(self):
        """The GDN/mamba2 orientation the dedup layout was built for."""
        shape = Mamba2StateShape.create(
            tp_world_size=1,
            intermediate_size=4096,
            n_groups=2,
            num_heads=32,
            head_dim=128,
            state_size=128,
            conv_kernel=4,
        )
        (conv_shape,) = shape.conv
        dim, win = conv_shape
        self.assertEqual(win, 4 - 1)
        self.assertGreater(dim, win)  # dim-major: feature axis first


class TestConvWindowDedupEnabled(CustomTestCase):
    def test_truth_table(self):
        cases = [
            # (is_npu, is_cpu, topk, is_kda) -> expected
            ((False, False, None, False), True),  # GDN spec chain (topk unset)
            ((False, False, 1, False), True),  # GDN chain
            ((False, False, 4, False), False),  # GDN tree -> dense
            ((False, False, 1, True), False),  # KDA chain -> dense (swapped conv)
            ((False, False, 10, True), False),  # KDA tree -> dense
            ((True, False, 1, False), False),  # NPU -> dense
            ((False, True, 1, False), False),  # CPU -> dense
        ]
        for args, expected in cases:
            with self.subTest(args=args):
                self.assertEqual(conv_window_dedup_enabled(*args), expected)

    def test_is_kda_defaults_false(self):
        """Pre-existing GDN call sites that don't pass is_kda keep dedup."""
        self.assertTrue(conv_window_dedup_enabled(False, False, 1))


@unittest.skipUnless(BACKENDS_IMPORTABLE, "attention backends not importable")
class TestSpecTargetVerifyCapabilityGate(CustomTestCase):
    @staticmethod
    def _run_gate(dispatcher, spec_algorithm="NGRAM", is_draft_worker=False):
        mock_self = SimpleNamespace(
            is_draft_worker=is_draft_worker, kernel_dispatcher=dispatcher
        )
        mock_runner = SimpleNamespace(
            server_args=SimpleNamespace(speculative_algorithm=spec_algorithm)
        )
        MambaAttnBackendBase._check_spec_target_verify_support(
            mock_self, mock_runner
        )

    def test_both_real_dispatchers_advertise_target_verify(self):
        """The probe is hasattr on the METHOD — pin that both shipping
        dispatchers expose it (class-level check, no kernel construction)."""
        for cls in (GDNKernelDispatcher, KDAKernelDispatcher):
            with self.subTest(dispatcher=cls.__name__):
                self.assertTrue(callable(getattr(cls, "target_verify", None)))

    def test_dispatcher_with_method_passes(self):
        class _VerifyCapable:
            def target_verify(self, **kwargs):
                pass

        self._run_gate(_VerifyCapable())  # must not raise

    def test_dispatcher_without_method_rejected(self):
        class _NoVerify:
            pass

        with self.assertRaisesRegex(ValueError, "target-verify"):
            self._run_gate(_NoVerify())

    def test_spec_off_is_noop(self):
        class _NoVerify:
            pass

        self._run_gate(_NoVerify(), spec_algorithm=None)  # must not raise

    def test_draft_worker_is_noop(self):
        class _NoVerify:
            pass

        self._run_gate(_NoVerify(), is_draft_worker=True)  # must not raise


if __name__ == "__main__":
    unittest.main()
