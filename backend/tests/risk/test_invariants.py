"""Construction-time invariants for the Gate's value types.

Every rejection below is a value that would make some later number a lie: a
float where money belongs, a naive timestamp in a system that is UTC end to end,
a negative count. They are cheap to check at construction and expensive to
discover in a verdict, so they are checked at construction.

`InvariantViolation` at build time is the house style inherited from
`alphagate.core`: a constructed domain object is always valid.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest

from alphagate.core.errors import InvariantViolation
from alphagate.risk import (
    DEFAULT_LIMITS,
    Approved,
    CheckResult,
    OpenPosition,
    PortfolioSnapshot,
    Vetoed,
    VetoReason,
    evaluate,
)
from tests.risk.conftest import NOW, credit_spread, position, proposal, snapshot

NAIVE = datetime(2026, 9, 1, 15, 30)  # noqa: DTZ001 — the point of the test

NON_FINITE = ["NaN", "sNaN", "Infinity", "-Infinity"]
"""Every shape of non-number a Decimal can take. All must be domain errors."""


class TestOpenPosition:
    def test_quantity_must_be_positive(self) -> None:
        with pytest.raises(InvariantViolation, match="quantity must be positive"):
            replace(position(), quantity=0)

    def test_max_loss_must_be_decimal(self) -> None:
        """A float here would make `open_risk` a float, and heat is money."""
        with pytest.raises(InvariantViolation, match="must be Decimal"):
            replace(position(), max_loss=350.0)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad", ["0", "-1", "NaN", "sNaN", "Infinity", "-Infinity"])
    def test_max_loss_must_be_finite_and_positive(self, bad: str) -> None:
        """`InvariantViolation`, never `InvalidOperation`.

        `Decimal("NaN") <= 0` raises instead of answering, so the finiteness half
        of this invariant is asked first. See `tests/options/test_non_finite.py`
        for the full statement of why the exception type is load-bearing.
        """
        with pytest.raises(InvariantViolation, match="finite and positive"):
            replace(position(), max_loss=Decimal(bad))

    def test_opened_at_must_be_tz_aware(self) -> None:
        with pytest.raises(InvariantViolation, match="tz-aware"):
            replace(position(), opened_at=NAIVE)

    def test_it_reports_its_underlying(self) -> None:
        assert position().underlying == credit_spread().underlying


class TestPortfolioSnapshot:
    def test_equity_must_be_decimal(self) -> None:
        with pytest.raises(InvariantViolation, match="must be Decimal"):
            PortfolioSnapshot(
                equity=100_000.0,  # type: ignore[arg-type]
                positions=(),
                drawdown_pct=Decimal(0),
                fills_today=0,
            )

    @pytest.mark.parametrize("bad", NON_FINITE)
    def test_equity_must_be_finite(self, bad: str) -> None:
        with pytest.raises(InvariantViolation, match="equity must be finite"):
            snapshot(equity=Decimal(bad))

    @pytest.mark.parametrize("bad", NON_FINITE)
    def test_drawdown_must_be_finite(self, bad: str) -> None:
        with pytest.raises(InvariantViolation, match="drawdown_pct must be finite"):
            snapshot(drawdown=bad)

    def test_drawdown_must_not_be_negative(self) -> None:
        """A negative drawdown is an account at a new high, which is `0`."""
        with pytest.raises(InvariantViolation, match="drawdown_pct"):
            snapshot(drawdown="-0.01")

    def test_fills_today_must_not_be_negative(self) -> None:
        with pytest.raises(InvariantViolation, match="fills_today"):
            snapshot(fills_today=-1)

    def test_an_empty_book_has_no_risk_and_no_exposure(self) -> None:
        empty = snapshot()
        assert empty.open_risk == Decimal(0)
        assert empty.net_delta == 0.0
        assert empty.open_structures == 0


class TestTradeProposal:
    def test_risk_as_of_must_be_tz_aware(self) -> None:
        with pytest.raises(InvariantViolation, match="risk_as_of must be tz-aware"):
            proposal(risk_as_of=NAIVE)

    def test_total_max_loss_scales_with_quantity(self) -> None:
        assert proposal(quantity=1).total_max_loss == Decimal(350)
        assert proposal(quantity=4).total_max_loss == Decimal(1400)


class TestRiskLimits:
    def test_percentages_must_be_decimal(self) -> None:
        with pytest.raises(InvariantViolation, match="must be Decimal"):
            replace(DEFAULT_LIMITS, max_spread_pct=0.05)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad", NON_FINITE)
    def test_percentages_must_be_finite(self, bad: str) -> None:
        with pytest.raises(InvariantViolation, match="must be finite"):
            replace(DEFAULT_LIMITS, max_drawdown_pct=Decimal(bad))

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_float_knobs_must_be_finite_too(self, bad: float) -> None:
        """Floats fail the opposite way round and are more dangerous for it.

        `nan <= 0` is False and `nan > x` is False, so a NaN would pass every
        magnitude check below it and then silently disable the check it
        configures — a freshness limit that never trips reads exactly like a
        freshness limit that is never breached.
        """
        with pytest.raises(InvariantViolation, match="must be finite"):
            replace(DEFAULT_LIMITS, max_quote_age=bad)
        with pytest.raises(InvariantViolation, match="bounds must be finite"):
            replace(DEFAULT_LIMITS, delta_band=(-0.30, bad))

    def test_the_dte_floor_must_not_be_negative(self) -> None:
        """An expired contract is not a short-dated one."""
        with pytest.raises(InvariantViolation, match="must not be negative"):
            replace(DEFAULT_LIMITS, dte_range=(-1, 21))

    def test_bands_scale_linearly_with_equity(self) -> None:
        small = DEFAULT_LIMITS.scaled_delta_band(Decimal(50_000))
        large = DEFAULT_LIMITS.scaled_delta_band(Decimal(100_000))
        assert large[1] == pytest.approx(small[1] * 2)
        assert small[0] == pytest.approx(-small[1])


class TestVerdictSurface:
    def test_a_veto_reports_itself_as_not_approved(self) -> None:
        veto = Vetoed(
            reasons=(VetoReason("liquidity", "too wide"),),
            checks=(CheckResult("liquidity", False, "too wide"),),
        )
        assert veto.is_approved is False

    def test_an_approval_reports_itself_as_approved(self) -> None:
        verdict = evaluate(proposal(), snapshot(), DEFAULT_LIMITS, NOW)
        assert isinstance(verdict, Approved)
        assert verdict.is_approved is True

    def test_the_check_tape_is_a_tuple_not_a_list(self) -> None:
        """Immutable on the way out: the journal records what the Gate decided,
        not what a later caller edited."""
        verdict = evaluate(proposal(), snapshot(), DEFAULT_LIMITS, NOW)
        assert isinstance(verdict.checks, tuple)


def test_a_position_opened_in_the_future_is_still_constructible() -> None:
    """The Gate does not police the caller's bookkeeping, only its own limits."""
    ahead: Any = replace(position(), opened_at=NOW + timedelta(days=1))
    assert isinstance(ahead, OpenPosition)
