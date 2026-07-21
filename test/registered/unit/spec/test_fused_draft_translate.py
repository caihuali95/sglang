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
"""In-kernel v2p translate of `generate_draft_decode_kv_indices` (fused draft KV
fused draft KV): with HAS_V2P the kernel must emit exactly
`clamp_min(v2p[id // ps] * ps + id % ps, 0)` for every id it writes, and
write NOTHING it didn't write before (no garbage-tail translation).

Equivalence test: run the kernel without v2p, post-translate the touched
positions as a reference, run with v2p, compare — over tombstoned pages
(v2p == -1 -> slot-0 sink clamp), topk > 1, and page_size in {1, 4}.

CUDA-only (Triton kernel).
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")

_HAS_CUDA = torch.cuda.is_available()


def _reference_translate(ids, v2p, page_size):
    pages = ids // page_size
    offs = ids % page_size
    return torch.clamp_min(v2p[pages] * page_size + offs, 0)


@unittest.skipUnless(_HAS_CUDA, "Triton kernel requires CUDA")
class TestFusedDraftTranslate(CustomTestCase):
    def _run(self, *, page_size, topk, num_steps, seq_lens):
        from sglang.srt.speculative.triton_ops.cache_locs import (
            generate_draft_decode_kv_indices,
        )
        from sglang.srt.utils import next_power_of_2

        device = "cuda"
        num_seqs = len(seq_lens)
        bs = num_seqs * topk
        pool_len = 512
        num_pages = 128  # virtual pages

        # req_to_token rows hold page-structured virtual ids covering the
        # prefix + the per-topk draft slots (mirror the real layout loosely --
        # the kernel only requires ids at the positions it reads).
        torch.manual_seed(7)
        req_to_token = torch.randint(
            0,
            num_pages * page_size,
            (num_seqs, pool_len),
            dtype=torch.int64,
            device=device,
        )
        req_pool_indices = torch.arange(num_seqs, dtype=torch.int64, device=device)
        seq_lens_t = torch.tensor(seq_lens, dtype=torch.int64, device=device)
        positions = torch.randint(
            1, 64, (bs,), dtype=torch.int64, device=device
        )

        # v2p: permutation of pages with some tombstones (-1).
        v2p = torch.randperm(num_pages, dtype=torch.int64, device=device)
        v2p[::7] = -1  # tombstoned pages -> sink clamp

        width = (max(seq_lens) + num_steps) * bs + 64
        SENTINEL = -7

        def launch(with_v2p):
            kv_indices = torch.full(
                (num_steps, width), SENTINEL, dtype=torch.int64, device=device
            )
            kv_indptr = torch.zeros(
                (num_steps, bs + 2), dtype=torch.int32, device=device
            )
            generate_draft_decode_kv_indices[(num_steps, num_seqs, topk)](
                req_pool_indices,
                req_to_token,
                seq_lens_t,
                kv_indices,
                kv_indptr,
                positions,
                pool_len,
                width,
                kv_indptr.shape[1],
                next_power_of_2(num_seqs),
                next_power_of_2(num_steps),
                next_power_of_2(bs),
                page_size,
                v2p_ptr=v2p if with_v2p else None,
                HAS_V2P=with_v2p,
            )
            return kv_indices

        plain = launch(with_v2p=False)
        fused = launch(with_v2p=True)

        touched = plain != SENTINEL
        # Same write footprint: the translate must not add or drop stores.
        self.assertTrue(
            torch.equal(touched, fused != SENTINEL),
            "HAS_V2P changed the kernel's write footprint",
        )
        # Every written id is the reference translate of the plain id.
        expected = _reference_translate(plain[touched], v2p, page_size)
        self.assertTrue(
            torch.equal(fused[touched], expected),
            "in-kernel translate disagrees with the reference "
            "clamp_min(v2p[page]*ps + off, 0)",
        )
        # Tombstone clamp actually exercised and landed in the sink page.
        plain_pages = plain[touched] // page_size
        hit_tombstone = v2p[plain_pages] < 0
        if hit_tombstone.any():
            self.assertTrue(
                torch.all(fused[touched][hit_tombstone] == 0),
                "tombstoned ids must clamp to the slot-0 sink",
            )

    def test_ps1_topk1_chain(self):
        self._run(page_size=1, topk=1, num_steps=3, seq_lens=[5, 17, 33])

    def test_ps1_topk4_tree(self):
        self._run(page_size=1, topk=4, num_steps=3, seq_lens=[9, 21])

    def test_ps4_topk1(self):
        self._run(page_size=4, topk=1, num_steps=4, seq_lens=[8, 16, 24])


if __name__ == "__main__":
    unittest.main()
