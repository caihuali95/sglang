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
"""The write-loc diagnostic (`SGLANG_DEBUG_FUSED_WRITE_LOCS`) must actually FIRE.

A silent wrong-slot KV write never crashes -- it just degrades accept length --
so the check that hunts it has to be proven to catch each failure mode, or a
clean run means nothing. Pins all three:

  (a) a virtual loc reaching the pool with no pre-translated physical partner,
  (b) a physical loc that disagrees with the live v2p map (stale / mis-sliced),
  (c) a physical slot outside the sub-pool's addressable range,

plus the two must-NOT-fire cases: an identity-v2p write (the trap -- a wrong-slot
bug is invisible while v2p is identity) and a correct write under a DIVERGED v2p.

CPU-only: pure index arithmetic against a fake translate.
"""

import unittest

import torch

from sglang.srt.mem_cache.unified_memory_pool import _check_write_locs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

MIN_SLOT, MAX_SLOT = 4, 1024


def _v2p(mapping):
    """A translate fn backed by an explicit virtual->physical dict."""

    def translate(virt, *, out=None):
        return torch.tensor(
            [mapping[int(v)] for v in virt.tolist()], dtype=torch.int64
        )

    return translate


def _check(virt, phys, mapping):
    _check_write_locs(
        sub_pool_name="full",
        translate=_v2p(mapping),
        virt_loc=virt,
        phys_loc=phys,
        min_slot=MIN_SLOT,
        max_slot=MAX_SLOT,
    )


class TestWriteLocCheck(CustomTestCase):
    def test_correct_write_under_diverged_v2p_passes(self):
        # The realistic steady state: v2p is a permutation, and the write agrees.
        mapping = {10: 77, 11: 5, 12: 900}
        virt = torch.tensor([10, 11, 12])
        _check(virt, torch.tensor([77, 5, 900]), mapping)

    def test_identity_v2p_cannot_distinguish(self):
        # The trap this whole hunt lives in: while v2p is identity, writing the
        # VIRTUAL id is indistinguishable from writing the physical one. Pinned so
        # nobody "fixes" the check into firing here.
        mapping = {10: 10, 11: 11}
        virt = torch.tensor([10, 11])
        _check(virt, virt.clone(), mapping)

    def test_missing_physical_loc_fires(self):
        # (a) full_loc=None -> the pool would use VIRTUAL ids as physical slots.
        with self.assertRaises(AssertionError) as cm:
            _check(torch.tensor([10, 11]), None, {10: 77, 11: 5})
        self.assertIn("NO pre-translated physical loc", str(cm.exception))

    def test_wrong_slot_write_fires_and_names_the_tokens(self):
        # (b) the exact Bug-B shape: the physical loc disagrees with the live map.
        mapping = {10: 77, 11: 5, 12: 900}
        virt = torch.tensor([10, 11, 12])
        wrong = torch.tensor([77, 900, 5])  # tokens 1 and 2 swapped
        with self.assertRaises(AssertionError) as cm:
            _check(virt, wrong, mapping)
        msg = str(cm.exception)
        self.assertIn("WRONG-SLOT KV WRITE", msg)
        self.assertIn("2/3 tokens", msg)  # must report HOW MANY, not just that
        self.assertIn("(1, 11, 900, 5)", msg)  # (idx, virtual, written-to, should-be)

    def test_shape_mismatch_fires(self):
        # A mis-sliced per-step write buffer: right values, wrong length.
        with self.assertRaises(AssertionError) as cm:
            _check(torch.tensor([10, 11]), torch.tensor([77]), {10: 77, 11: 5})
        self.assertIn("mis-sliced", str(cm.exception))

    def test_out_of_range_slot_fires(self):
        # (c) a physical slot outside [min_slot, max_slot) -- the IMA precursor.
        # Checked even though the values agree with the map, so a corrupt v2p is
        # caught rather than blindly trusted.
        for bad in (MIN_SLOT - 1, MAX_SLOT):
            with self.assertRaises(AssertionError) as cm:
                _check(torch.tensor([10]), torch.tensor([bad]), {10: bad})
            self.assertIn("OUT OF RANGE", str(cm.exception))

    def test_empty_write_is_a_noop(self):
        _check(torch.tensor([], dtype=torch.int64), None, {})


if __name__ == "__main__":
    unittest.main()
