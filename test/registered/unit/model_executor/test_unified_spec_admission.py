"""Unit tests for the unified-pool speculative admission solve
(`solve_unified_spec_admission` in model_runner_kv_cache_mixin.py) -- CPU
only, pure integer arithmetic.

The solve charges, per admitted request: hard mamba slots + D spec-band rows
+ a token floor priced at the fused cell (the full-KV slot entry, which
already carries the draft's per-token bytes -- the draft KV is fused into the
slot layout, so there is no separate draft pool and nothing else to reserve).
These tests pin:
  1. the solve against a verbatim transcription of its formula (an executable
     spec: any accounting drift in either direction fails byte-exactly),
  2. boundary behavior (requested cap, explicit mamba slots, infeasible),
  3. the radix-headroom grant's base-ratio cap.
"""

import unittest

from sglang.srt.model_executor.model_runner_kv_cache_mixin import (
    UNIFIED_SPEC_MIN_TOKENS_PER_REQ,
    solve_unified_spec_admission,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

GIB = 1 << 30
MIB = 1 << 20


def _reference_solve(
    *,
    rest_bytes,
    requested,
    mamba_bytes_per_req,
    num_draft_tokens,
    hard_slots,
    mamba_ratio,
    cell_size,
    margin_slots,
    explicit_mamba_slots=None,
):
    """The solve's formula, transcribed verbatim as an executable spec."""
    token_floor_bytes = UNIFIED_SPEC_MIN_TOKENS_PER_REQ * cell_size
    band_const_bytes = (1 + margin_slots) * num_draft_tokens * mamba_bytes_per_req
    per_req_bytes = (
        hard_slots + num_draft_tokens
    ) * mamba_bytes_per_req + token_floor_bytes
    n = (rest_bytes - band_const_bytes) // per_req_bytes
    n = min(requested, n)
    if explicit_mamba_slots is not None:
        n = min(n, explicit_mamba_slots // hard_slots)
    if n <= 0:
        return n, 0, 0
    band_bytes = (n + 1 + margin_slots) * num_draft_tokens * mamba_bytes_per_req
    hard_mamba_slots = n * hard_slots
    leftover = (
        rest_bytes
        - band_bytes
        - hard_mamba_slots * mamba_bytes_per_req
        - n * token_floor_bytes
    )
    headroom_slots = min(
        max(0, leftover) // mamba_bytes_per_req,
        n * mamba_ratio - hard_mamba_slots,
    )
    mamba_slots = int(hard_mamba_slots + headroom_slots)
    if explicit_mamba_slots is not None:
        mamba_slots = min(mamba_slots, explicit_mamba_slots)
    deducted = band_bytes + mamba_slots * mamba_bytes_per_req
    return n, mamba_slots, deducted


# Production-shaped cells: Qwen3.5-9B-class GDN. The fused cell = host entry
# 32768 B/token + the draft region: EAGLE +4096 (1 MTP layer), DFLASH +24576
# (6 layers). NGRAM carries no draft KV, so its cell is the host entry alone.
QWEN_EAGLE = dict(
    mamba_bytes_per_req=49 * MIB,
    num_draft_tokens=4,
    hard_slots=3,
    mamba_ratio=5,
    cell_size=36864,
    margin_slots=3,
)
QWEN_DFLASH = dict(
    mamba_bytes_per_req=49 * MIB,
    num_draft_tokens=16,
    hard_slots=1,
    mamba_ratio=1,
    cell_size=57344,
    margin_slots=3,
)
QWEN_NGRAM = dict(
    mamba_bytes_per_req=49 * MIB,
    num_draft_tokens=4,
    hard_slots=3,
    mamba_ratio=5,
    cell_size=32768,
    margin_slots=3,
)


class TestUnifiedSpecAdmissionSolve(CustomTestCase):
    def test_matches_reference_formula(self):
        """Byte-exact against the transcribed formula, across fused-EAGLE,
        fused-DFLASH and no-draft-KV cells and tight/ample/clamped budgets."""
        for params in (QWEN_EAGLE, QWEN_DFLASH, QWEN_NGRAM):
            for rest_gib in (2, 10, 35, 55):
                for requested in (8, 48, 256):
                    for explicit in (None, 40):
                        kwargs = dict(
                            rest_bytes=rest_gib * GIB,
                            requested=requested,
                            explicit_mamba_slots=explicit,
                            **params,
                        )
                        a = solve_unified_spec_admission(**kwargs)
                        n, mamba_slots, deducted = _reference_solve(**kwargs)
                        with self.subTest(
                            cell=params["cell_size"],
                            rest_gib=rest_gib,
                            requested=requested,
                            explicit=explicit,
                        ):
                            self.assertEqual(a.max_num_reqs, n)
                            if n > 0:
                                self.assertEqual(a.mamba_slots, mamba_slots)
                                self.assertEqual(a.deducted_bytes, deducted)
                                self.assertEqual(
                                    a.token_budget_bytes,
                                    rest_gib * GIB - deducted,
                                )

    def test_token_floor_fits_the_budget(self):
        """Whatever the solve admits, the per-request token floor must fit in
        the token budget it leaves behind -- the floor is a guarantee, not an
        aspiration."""
        for params, rest_gib, requested in (
            (QWEN_EAGLE, 35, 256),
            (QWEN_DFLASH, 55, 256),
            (QWEN_DFLASH, 36, 48),
            (QWEN_EAGLE, 6, 48),
        ):
            a = solve_unified_spec_admission(
                rest_bytes=rest_gib * GIB, requested=requested, **params
            )
            with self.subTest(rest_gib=rest_gib, cell=params["cell_size"]):
                self.assertGreater(a.max_num_reqs, 0)
                tokens = a.token_budget_bytes // params["cell_size"]
                self.assertGreaterEqual(
                    tokens, a.max_num_reqs * UNIFIED_SPEC_MIN_TOKENS_PER_REQ
                )

    def test_heavier_fused_cell_admits_fewer(self):
        """A heavier draft region (DFLASH's 24576 B/token vs EAGLE's 4096)
        raises the fused cell and must shrink admission monotonically -- the
        cell is the ONLY place the draft's cost enters the solve."""
        rest_bytes = 55 * GIB
        base = dict(QWEN_EAGLE, num_draft_tokens=4, hard_slots=3, mamba_ratio=5)
        light = solve_unified_spec_admission(
            rest_bytes=rest_bytes, requested=1024, **base
        )
        heavy = solve_unified_spec_admission(
            rest_bytes=rest_bytes,
            requested=1024,
            **{**base, "cell_size": QWEN_DFLASH["cell_size"]},
        )
        self.assertLess(heavy.max_num_reqs, light.max_num_reqs)
        self.assertGreater(heavy.max_num_reqs, 0)

    def test_requested_caps_admission(self):
        a = solve_unified_spec_admission(
            rest_bytes=55 * GIB, requested=48, **QWEN_DFLASH
        )
        self.assertLessEqual(a.max_num_reqs, 48)
        b = solve_unified_spec_admission(
            rest_bytes=55 * GIB, requested=8, **QWEN_DFLASH
        )
        self.assertEqual(b.max_num_reqs, 8)

    def test_explicit_mamba_slots_cap(self):
        a = solve_unified_spec_admission(
            rest_bytes=35 * GIB,
            requested=256,
            explicit_mamba_slots=30,
            **QWEN_EAGLE,
        )
        # 30 explicit slots / 3 hard slots per request -> at most 10 admitted,
        # and the grant itself never exceeds the explicit cap.
        self.assertLessEqual(a.max_num_reqs, 10)
        self.assertLessEqual(a.mamba_slots, 30)

    def test_infeasible_returns_nonpositive(self):
        """A budget below one request's floor must return n <= 0 (the caller
        raises with the itemization) and zeroed reservations."""
        a = solve_unified_spec_admission(
            rest_bytes=1 * GIB, requested=48, **QWEN_DFLASH
        )
        self.assertLessEqual(a.max_num_reqs, 0)
        self.assertEqual(a.deducted_bytes, 0)
        self.assertEqual(a.band_bytes, 0)

    def test_headroom_capped_by_base_ratio(self):
        """With an ample budget the radix-headroom grant stops at the base
        ratio's allowance (n x ratio total slots), not at the leftover bytes."""
        a = solve_unified_spec_admission(
            rest_bytes=35 * GIB, requested=4, **QWEN_EAGLE
        )
        self.assertEqual(a.max_num_reqs, 4)
        self.assertEqual(a.hard_mamba_slots, 4 * QWEN_EAGLE["hard_slots"])
        self.assertEqual(
            a.mamba_slots, 4 * QWEN_EAGLE["mamba_ratio"]
        )  # leftover >> cap


if __name__ == "__main__":
    unittest.main()
