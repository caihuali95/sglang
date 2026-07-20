"""Unit tests for the unified-pool joint admission solve
(`solve_unified_spec_admission` in model_runner_kv_cache_mixin.py) -- CPU
only, pure integer arithmetic. Covers spec ON and spec OFF (D=0).

The solve charges, per admitted request: hard mamba slots + D spec-band rows
+ a token floor priced at the fused cell (the full-KV slot entry, which
already carries the draft's per-token bytes -- the draft KV is fused into the
slot layout, so there is no separate draft pool and nothing else to reserve).
At D=0 (spec OFF) every band term degenerates to exactly zero and the solve
reduces to hard mamba slots + the token floor.
These tests pin:
  1. the solve against a verbatim transcription of its formula (an executable
     spec: any accounting drift in either direction fails byte-exactly),
  2. boundary behavior (requested cap, explicit mamba slots, infeasible),
  3. the radix-headroom grant's base-ratio cap,
  4. the D=0 degeneracy (band bytes exactly zero, deduction = mamba only).
"""

import unittest

from sglang.srt.model_executor.model_runner_kv_cache_mixin import (
    UNIFIED_MIN_TOKENS_PER_REQ,
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
    token_floor_bytes = UNIFIED_MIN_TOKENS_PER_REQ * cell_size
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


def _deducted_bytes(a, mamba_bytes_per_req):
    """The N-shared reservation (band + all admitted mamba slots), derived from
    the surviving struct fields. The dedicated `deducted_bytes` /
    `token_budget_bytes` fields were removed once nothing in production read
    them; the solve's arithmetic is still pinned here, just recomputed."""
    return a.band_bytes + a.mamba_slots * mamba_bytes_per_req


# Production-shaped cells: Qwen3.5-9B-class GDN. The fused cell = host entry
# 32768 B/token + the draft region: EAGLE +4096 (1 MTP layer), DFLASH +24576
# (6 layers). NGRAM carries no draft KV, so its cell is the host entry alone.
QWEN_EAGLE = dict(
    mamba_bytes_per_req=49 * MIB,
    num_draft_tokens=4,
    # Floor = the FULL mamba_ratio (active + track + radix-checkpoint slots:
    # the checkpoint slots are locked by running chunked requests, so they
    # are part of the per-request floor, not leftover headroom).
    hard_slots=5,
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
# Spec-OFF cells (D=0): no draft KV in the cell, no band. The caller passes
# the FULL mamba_ratio as the floor (active + radix-checkpoint slots; plain
# radix ratio 3, overlap+lazy extra buffer ratio 4). disable_radix_cache is
# ratio 1 -> floor 1 (the DFLASH cell above covers that shape).
QWEN_SPEC_OFF = dict(
    mamba_bytes_per_req=49 * MIB,
    num_draft_tokens=0,
    hard_slots=3,
    mamba_ratio=3,
    cell_size=32768,
    margin_slots=3,
)
QWEN_SPEC_OFF_EXTRA_BUF = dict(
    mamba_bytes_per_req=49 * MIB,
    num_draft_tokens=0,
    hard_slots=4,
    mamba_ratio=4,
    cell_size=32768,
    margin_slots=3,
)
# Kimi-Linear-48B-class KDA + MLA hybrid at TP2, spec OFF. The MLA cell is an
# order of magnitude lighter than an MHA cell (7 MLA layers x 576-dim latent
# x bf16 = 8064 B/token, single latent region -- no K/V pair); the KDA state
# (~20 KDA layers, fp32 temporal) dominates the per-request charge. The
# no_buffer mamba strategy forces non-overlap, so the plain radix ratio 3
# floor applies.
KIMI_SPEC_OFF = dict(
    mamba_bytes_per_req=21 * MIB,
    num_draft_tokens=0,
    hard_slots=3,
    mamba_ratio=3,
    cell_size=8064,
    margin_slots=3,
)


class TestUnifiedSpecAdmissionSolve(CustomTestCase):
    def test_matches_reference_formula(self):
        """Byte-exact against the transcribed formula, across fused-EAGLE,
        fused-DFLASH, no-draft-KV and spec-OFF (D=0) cells and
        tight/ample/clamped budgets."""
        for params in (
            QWEN_EAGLE,
            QWEN_DFLASH,
            QWEN_NGRAM,
            QWEN_SPEC_OFF,
            QWEN_SPEC_OFF_EXTRA_BUF,
            KIMI_SPEC_OFF,
        ):
            for rest_gib in (2, 10, 35, 55):
                for requested in (8, 48, 256, 4096):
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
                            d=params["num_draft_tokens"],
                            hard=params["hard_slots"],
                            rest_gib=rest_gib,
                            requested=requested,
                            explicit=explicit,
                        ):
                            self.assertEqual(a.max_num_reqs, n)
                            if n > 0:
                                self.assertEqual(a.mamba_slots, mamba_slots)
                                # N-shared reservation, byte-exact vs reference.
                                self.assertEqual(
                                    _deducted_bytes(
                                        a, params["mamba_bytes_per_req"]
                                    ),
                                    deducted,
                                )
                                # One request's floor = 1x hard slots + the
                                # fixed band; never larger than the N-shared
                                # deduction (which carries N floors + headroom).
                                self.assertEqual(
                                    a.single_req_floor_bytes,
                                    params["hard_slots"]
                                    * params["mamba_bytes_per_req"]
                                    + a.band_bytes,
                                )
                                self.assertLessEqual(
                                    a.single_req_floor_bytes,
                                    _deducted_bytes(
                                        a, params["mamba_bytes_per_req"]
                                    ),
                                )

    def test_spec_off_degenerates_band_to_zero(self):
        """D=0 (spec OFF): every band term is EXACTLY zero, so the deduction is
        the mamba grant alone. This is the arithmetic contract the spec-OFF gate
        flip relies on."""
        for params in (QWEN_SPEC_OFF, QWEN_SPEC_OFF_EXTRA_BUF, KIMI_SPEC_OFF):
            for rest_gib, requested in ((2, 48), (35, 256), (55, 4096)):
                a = solve_unified_spec_admission(
                    rest_bytes=rest_gib * GIB, requested=requested, **params
                )
                with self.subTest(
                    hard=params["hard_slots"],
                    rest_gib=rest_gib,
                    requested=requested,
                ):
                    self.assertGreater(a.max_num_reqs, 0)
                    self.assertEqual(a.band_bytes, 0)
                    # No band -> the deduction is exactly the mamba grant, and
                    # the single-request floor is exactly one request's slots.
                    self.assertEqual(
                        _deducted_bytes(a, params["mamba_bytes_per_req"]),
                        a.mamba_slots * params["mamba_bytes_per_req"],
                    )
                    self.assertEqual(
                        a.single_req_floor_bytes,
                        params["hard_slots"] * params["mamba_bytes_per_req"],
                    )
                    # The hard grant backs every admitted request's slots.
                    self.assertEqual(
                        a.hard_mamba_slots,
                        a.max_num_reqs * params["hard_slots"],
                    )

    def test_spec_off_headroom_still_ratio_capped(self):
        """The radix-headroom grant keeps its `n x mamba_ratio` cap at D=0 --
        leftover bytes beyond it stay in the shared gap (token budget)."""
        a = solve_unified_spec_admission(
            rest_bytes=35 * GIB, requested=4, **QWEN_SPEC_OFF
        )
        self.assertEqual(a.max_num_reqs, 4)
        self.assertEqual(a.mamba_slots, 4 * QWEN_SPEC_OFF["mamba_ratio"])

    def test_token_floor_fits_the_budget(self):
        """Whatever the solve admits, the per-request token floor must fit in
        the token budget it leaves behind -- the floor is a guarantee, not an
        aspiration."""
        for params, rest_gib, requested in (
            (QWEN_EAGLE, 35, 256),
            (QWEN_DFLASH, 55, 256),
            (QWEN_DFLASH, 36, 48),
            (QWEN_EAGLE, 6, 48),
            (QWEN_SPEC_OFF, 6, 4096),
            (QWEN_SPEC_OFF_EXTRA_BUF, 35, 4096),
            # Kimi-48B at TP2 leaves a narrow KV budget after ~48 GB/rank of
            # weights -- the solve must still admit a healthy batch there.
            (KIMI_SPEC_OFF, 12, 256),
            (KIMI_SPEC_OFF, 4, 48),
        ):
            a = solve_unified_spec_admission(
                rest_bytes=rest_gib * GIB, requested=requested, **params
            )
            with self.subTest(rest_gib=rest_gib, cell=params["cell_size"]):
                self.assertGreater(a.max_num_reqs, 0)
                token_budget = rest_gib * GIB - _deducted_bytes(
                    a, params["mamba_bytes_per_req"]
                )
                tokens = token_budget // params["cell_size"]
                self.assertGreaterEqual(
                    tokens, a.max_num_reqs * UNIFIED_MIN_TOKENS_PER_REQ
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
        # 30 explicit slots / 5 floor slots per request -> at most 6 admitted,
        # and the grant itself never exceeds the explicit cap.
        self.assertLessEqual(a.max_num_reqs, 6)
        self.assertLessEqual(a.mamba_slots, 30)

    def test_infeasible_returns_nonpositive(self):
        """A budget below one request's floor must return n <= 0 (the caller
        raises with the itemization) and zeroed reservations."""
        a = solve_unified_spec_admission(
            rest_bytes=1 * GIB, requested=48, **QWEN_DFLASH
        )
        self.assertLessEqual(a.max_num_reqs, 0)
        self.assertEqual(a.mamba_slots, 0)
        self.assertEqual(a.band_bytes, 0)
        self.assertEqual(a.single_req_floor_bytes, 0)

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


class TestSingleRequestFeasibilityLabel(CustomTestCase):
    """The handler advertises max_total_num_tokens as `(rest -
    single_req_floor_bytes) // cell` — a SINGLE-request feasibility ceiling
    (subtract ONE request's floor), decoupled from the N-shared token budget.
    These pin the arithmetic the handler relies on (see
    _handle_max_mamba_cache_unified); the ModelRunner wiring is exercised by the
    e2e GPU lanes."""

    def _feasibility_tokens(self, a, params, rest_bytes):
        return (rest_bytes - a.single_req_floor_bytes) // params["cell_size"]

    def _nshared_tokens(self, a, params, rest_bytes):
        token_budget = rest_bytes - _deducted_bytes(
            a, params["mamba_bytes_per_req"]
        )
        return token_budget // params["cell_size"]

    def test_label_never_below_nshared(self):
        """The feasibility label is >= the old N-shared value everywhere:
        subtracting 1x a floor can only leave MORE room than subtracting Nx.
        This is the whole point of the change — the label never regresses."""
        for params in (
            QWEN_EAGLE,
            QWEN_DFLASH,
            QWEN_NGRAM,
            QWEN_SPEC_OFF,
            KIMI_SPEC_OFF,
        ):
            for rest_gib in (10, 35, 55):
                for requested in (48, 256, 4096):
                    rest = rest_gib * GIB
                    a = solve_unified_spec_admission(
                        rest_bytes=rest, requested=requested, **params
                    )
                    if a.max_num_reqs <= 0:
                        continue
                    with self.subTest(
                        cell=params["cell_size"],
                        rest_gib=rest_gib,
                        requested=requested,
                    ):
                        self.assertGreaterEqual(
                            self._feasibility_tokens(a, params, rest),
                            self._nshared_tokens(a, params, rest),
                        )

    def test_large_state_label_strictly_larger(self):
        """Kimi-class (large mamba state, light MLA cell): when N is byte-bound
        the N mamba floors dominate the budget, so the N-shared label collapses
        toward N*token_floor while the single-request label stays near the full
        buffer. Concretely the ratio is ~1 + hard*per_req/token_floor (≈5x for
        KDA: 3*21MiB / (2048*8064B)). This is the bug the change fixes."""
        rest = 35 * GIB
        # requested large so admission is BYTE-bound (N maximal); at a small
        # requested cap N is artificially low and mamba does not yet dominate.
        a = solve_unified_spec_admission(
            rest_bytes=rest, requested=4096, **KIMI_SPEC_OFF
        )
        self.assertGreater(a.max_num_reqs, 1)
        feasibility = self._feasibility_tokens(a, KIMI_SPEC_OFF, rest)
        nshared = self._nshared_tokens(a, KIMI_SPEC_OFF, rest)
        # The freed bytes = (mamba_slots - hard_slots) * per_req; for KDA that is
        # several multiples of the N-shared count.
        self.assertGreater(feasibility, 3 * nshared)

    def test_small_state_label_parity(self):
        """Tiny mamba state (per_req << token_floor bytes): the 1x-vs-Nx
        distinction is a rounding error, so the feasibility label matches the
        N-shared value within one request's floor. GDN-class behavior is
        unchanged."""
        tiny = dict(
            mamba_bytes_per_req=64 * (1 << 10),  # 64 KiB — negligible state
            num_draft_tokens=0,
            hard_slots=3,
            mamba_ratio=3,
            cell_size=32768,
            margin_slots=3,
        )
        rest = 35 * GIB
        a = solve_unified_spec_admission(rest_bytes=rest, requested=256, **tiny)
        feasibility = self._feasibility_tokens(a, tiny, rest)
        nshared = self._nshared_tokens(a, tiny, rest)
        # Within the N-shared deduction expressed in tokens (a few tokens).
        self.assertLessEqual(
            feasibility - nshared,
            _deducted_bytes(a, tiny["mamba_bytes_per_req"]) // tiny["cell_size"]
            + 1,
        )

    def test_one_request_fits_its_floor_plus_token_floor(self):
        """The feasibility budget must hold at least one request's full-KV token
        floor after its own mamba floor + band — else the per-request clamp
        would admit a request that cannot be scheduled alone."""
        for params, rest_gib, requested in (
            (QWEN_EAGLE, 35, 256),
            (QWEN_DFLASH, 55, 256),
            (KIMI_SPEC_OFF, 12, 256),
            (KIMI_SPEC_OFF, 4, 48),
            (QWEN_SPEC_OFF, 6, 4096),
        ):
            rest = rest_gib * GIB
            a = solve_unified_spec_admission(
                rest_bytes=rest, requested=requested, **params
            )
            with self.subTest(rest_gib=rest_gib, cell=params["cell_size"]):
                self.assertGreater(a.max_num_reqs, 0)
                self.assertGreaterEqual(
                    self._feasibility_tokens(a, params, rest),
                    UNIFIED_MIN_TOKENS_PER_REQ,
                )


if __name__ == "__main__":
    unittest.main()
