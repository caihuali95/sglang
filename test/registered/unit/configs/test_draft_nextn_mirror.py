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
"""Draft num_nextn_predict_layers must reach the text config on wrappers.

The draft-arch overrides in `_config_draft_model` write `hf_config`; the
reader takes `hf_text_config`. On wrapper configs (multimodal /
ConditionalGeneration) the two are distinct objects — without the mirror the
reader falls back to None and the MTP draft worker is sized at the FULL model
depth (an oversized draft KV pool and an under-charged target cell).

    python -m pytest test/registered/unit/configs/test_draft_nextn_mirror.py -v
"""

import unittest
from types import SimpleNamespace

from sglang.srt.configs.model_config import _mirror_draft_nextn_layers
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDraftNextnMirror(unittest.TestCase):
    def test_wrapper_config_gets_the_mirror(self):
        hf = SimpleNamespace(num_nextn_predict_layers=1)
        text = SimpleNamespace()
        _mirror_draft_nextn_layers(
            hf_config=hf, hf_text_config=text, is_draft_model=True
        )
        self.assertEqual(text.num_nextn_predict_layers, 1)

    def test_shared_object_is_left_alone(self):
        hf = SimpleNamespace(num_nextn_predict_layers=1)
        _mirror_draft_nextn_layers(hf_config=hf, hf_text_config=hf, is_draft_model=True)
        self.assertEqual(hf.num_nextn_predict_layers, 1)

    def test_non_draft_and_non_mtp_do_not_write(self):
        hf = SimpleNamespace(num_nextn_predict_layers=1)
        text = SimpleNamespace()
        _mirror_draft_nextn_layers(
            hf_config=hf, hf_text_config=text, is_draft_model=False
        )
        self.assertFalse(hasattr(text, "num_nextn_predict_layers"))

        hf_plain = SimpleNamespace()  # EAGLE-style draft: no nextn attr at all
        _mirror_draft_nextn_layers(
            hf_config=hf_plain, hf_text_config=text, is_draft_model=True
        )
        self.assertFalse(hasattr(text, "num_nextn_predict_layers"))


if __name__ == "__main__":
    unittest.main()
