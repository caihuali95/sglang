"""Unit tests for the unified-pool speculative admission solve
(`solve_unified_spec_admission` in model_runner_kv_cache_mixin.py) -- CPU
only, pure integer arithmetic.

The solve charges, per admitted request: hard mamba slots + D spec-band rows
(both inflated by cell/base to back the virtual-id draft pool) + a token
floor priced at the draft-scaled cell. These tests pin:
  1. d=0 (no separate draft pool, e.g. NGRAM) is byte-identical to the
     legacy formula the solve replaced,
  2. the fit-by-construction guarantee: target buffer + draft pool
     (sized to the buffer's virtual-id capacity) never exceeds rest,
  3. boundary behavior (requested cap, explicit mamba slots, infeasible).
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


def _legacy_solve(
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
    """The pre-draft-reserve solve, transcribed verbatim. With cell == base the
    new solve must reproduce it byte-for-byte."""
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


# Production-shaped bases: Qwen3.5-9B-class GDN. base cell 32768 B/token;
# EAGLE draft +4096 (1 MTP layer), DFLASH draft +24576 (6 layers).
QWEN_EAGLE = dict(
    mamba_bytes_per_req=49 * MIB,
    num_draft_tokens=4,
    hard_slots=3,
    mamba_ratio=5,
    cell_size=36864,
    base_cell_size=32768,
    margin_slots=3,
)
QWEN_DFLASH = dict(
    mamba_bytes_per_req=49 * MIB,
    num_draft_tokens=16,
    hard_slots=1,
    mamba_ratio=1,
    cell_size=57344,
    base_cell_size=32768,
    margin_slots=3,
)


class TestUnifiedSpecAdmissionSolve(CustomTestCase):
    def test_no_draft_pool_matches_legacy_formula(self):
        """d = 0 (cell == base, e.g. NGRAM): the draft-backing inflation is
        the identity and the solve must equal the pre-reserve formula exactly
        -- across a grid including tight, ample, and clamped budgets."""
        for rest_gib in (2, 10, 35, 55):
            for requested in (8, 48, 256):
                for explicit in (None, 40):
                    kwargs = dict(
                        rest_bytes=rest_gib * GIB,
                        requested=requested,
                        mamba_bytes_per_req=49 * MIB,
                        num_draft_tokens=4,
                        hard_slots=3,
                        mamba_ratio=5,
                        cell_size=32768,
                        margin_slots=3,
                        explicit_mamba_slots=explicit,
                    )
                    a = solve_unified_spec_admission(
                        base_cell_size=32768, **kwargs
                    )
                    n, mamba_slots, deducted = _legacy_solve(**kwargs)
                    with self.subTest(rest_gib=rest_gib, requested=requested):
                        self.assertEqual(a.max_num_reqs, n)
                        if n > 0:
                            self.assertEqual(a.mamba_slots, mamba_slots)
                            self.assertEqual(a.deducted_bytes, deducted)
                            self.assertEqual(a.draft_reserve_bytes, 0)

    def test_draft_pool_fits_by_construction(self):
        """The core draft-reserve guarantee: reconstruct the factory's buffer
        (tokens x base + band + mamba) and the draft pool (virtual-id
        capacity x draft bytes/token); their sum must fit in rest. Without the
        reserve, a heavy DFLASH draft pool (a large fraction of the buffer)
        overflows and OOMs."""
        for params, rest_gib, requested in (
            (QWEN_EAGLE, 35, 256),  # EAGLE high-mfs-shaped
            (QWEN_DFLASH, 55, 256),  # DFLASH high-mfs-shaped (the OOM case)
            (QWEN_DFLASH, 36, 48),  # DFLASH mid-mfs-shaped
            (QWEN_EAGLE, 6, 48),  # tight budget
        ):
            rest_bytes = rest_gib * GIB
            a = solve_unified_spec_admission(
                rest_bytes=rest_bytes, requested=requested, **params
            )
            with self.subTest(rest_gib=rest_gib, cell=params["cell_size"]):
                self.assertGreater(a.max_num_reqs, 0)
                cell = params["cell_size"]
                base = params["base_cell_size"]
                tokens = a.token_budget_bytes // cell
                buffer_bytes = (
                    tokens * base
                    + a.band_bytes
                    + a.mamba_slots * params["mamba_bytes_per_req"]
                )
                draft_pool_bytes = (buffer_bytes // base) * (cell - base)
                # <= rest + one token's bytes (integer-flooring slack).
                self.assertLessEqual(buffer_bytes + draft_pool_bytes, rest_bytes + cell)
                # The floor tokens must actually fit the budget.
                self.assertGreaterEqual(
                    tokens, a.max_num_reqs * UNIFIED_SPEC_MIN_TOKENS_PER_REQ
                )
                if cell > base:
                    self.assertGreater(a.draft_reserve_bytes, 0)

    def test_dflash_reserve_admits_fewer_than_unreserved(self):
        """Charging the draft backing must shrink admission (that shrink IS
        the reserve); DFLASH (q=0.75) shrinks much harder than EAGLE
        (q=0.125)."""
        rest_bytes = 55 * GIB
        dflash = solve_unified_spec_admission(
            rest_bytes=rest_bytes, requested=1024, **QWEN_DFLASH
        )
        no_reserve = solve_unified_spec_admission(
            rest_bytes=rest_bytes,
            requested=1024,
            **{**QWEN_DFLASH, "cell_size": QWEN_DFLASH["base_cell_size"]},
        )
        self.assertLess(dflash.max_num_reqs, no_reserve.max_num_reqs)
        # q=0.75: byte terms inflate ~1.75x, so admission lands well below
        # 2/3 of the unreserved count.
        self.assertLess(dflash.max_num_reqs * 3, no_reserve.max_num_reqs * 2)

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
