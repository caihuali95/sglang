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
"""The unified-pool mamba path admits model families by POSITIVE allow-list
(`check_unified_mamba_family_admissible`) -- every admitted family's
state-touching kernels have been audited for the envelope-strided slot
addressing the unified views require. These tests pin BOTH sides:

  - admitted: the Qwen-family GDN configs (incl. subclasses), the mamba2
    family, and the KDA + MLA hybrid (Kimi Linear);
  - rejected BY NAME: JetNemotron / JetVLM (their decode state-update kernel
    addresses the initial state densely -- no slot stride -- which corrupts
    state silently under strided views) and the hybrid-lightning Bailing
    config (mambaish AND MLA-capable; its lightning backend + overlap path is
    unvetted). If someone "helpfully" adds one of these to the allow-list,
    this test is the reviewable place that decision must go through.
"""

import unittest

from sglang.srt.configs import (
    BailingHybridConfig,
    FalconH1Config,
    InternS2PreviewConfig,
    JetNemotronConfig,
    JetVLMConfig,
    KimiLinearConfig,
    Lfm2Config,
    NemotronHConfig,
    Qwen3_5Config,
    Qwen3_5MoeConfig,
    Qwen3NextConfig,
    ZayaConfig,
)
from sglang.srt.model_executor.model_runner_kv_cache_mixin import (
    check_unified_mamba_family_admissible,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestUnifiedMambaFamilyAllowlist(CustomTestCase):
    def test_admitted_families_pass(self):
        for cls in (
            # GDN, Qwen family (InternS2Preview subclasses Qwen3_5MoeConfig).
            Qwen3NextConfig,
            Qwen3_5Config,
            Qwen3_5MoeConfig,
            InternS2PreviewConfig,
            # Mamba2 family.
            FalconH1Config,
            NemotronHConfig,
            Lfm2Config,
            ZayaConfig,
            # KDA + MLA hybrid.
            KimiLinearConfig,
        ):
            with self.subTest(config=cls.__name__):
                check_unified_mamba_family_admissible(cls())  # must not raise

    def test_jet_nemotron_family_rejected(self):
        """JetNemotron/JetVLM are mambaish and pass the generic hybrid gates,
        but their decode kernel is dense-h0 -- must stay rejected."""
        for cls in (JetNemotronConfig, JetVLMConfig):
            with self.subTest(config=cls.__name__):
                with self.assertRaisesRegex(ValueError, cls.__name__):
                    check_unified_mamba_family_admissible(cls())

    def test_hybrid_lightning_rejected(self):
        """The hybrid-lightning Bailing config is mambaish AND MLA-capable --
        it would become admissible to the MLA-hybrid path by accident if the
        gate were family-generic. Keep it named here."""
        with self.assertRaisesRegex(ValueError, "BailingHybridConfig"):
            check_unified_mamba_family_admissible(BailingHybridConfig())


if __name__ == "__main__":
    unittest.main()
