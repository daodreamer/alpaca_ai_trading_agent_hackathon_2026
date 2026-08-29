"""Non-finite money is refused as a *domain* error, not an arithmetic one.

`Decimal("NaN")` does not behave like a number under comparison: `NaN <= 0`
raises `InvalidOperation` rather than answering, and `Decimal("Infinity")`
answers but corrupts every total it enters. Both must be refused at
construction, and — this is the point of the file — they must be refused with
`InvariantViolation`.

The distinction is not cosmetic. `InvariantViolation` is the type the domain
documents, the type callers catch, and the type the agent loop turns into a
journal entry. An `InvalidOperation` escaping from inside a validator is an
unhandled arithmetic error at the top of the agent loop, thrown from a line that
was supposed to be the thing preventing exactly that.

Every assertion below is therefore `pytest.raises(InvariantViolation)` and never
the broader `DomainError` or a bare `Exception` — a test that accepts any
exception cannot tell a working guard from a broken one.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal, InvalidOperation

import pytest

from alphagate.core.errors import DomainError, InvariantViolation
from alphagate.core.identifiers import ticker
from alphagate.options import (
    Cover,
    Greeks,
    Leg,
    OptionContract,
    OptionQuote,
    OptionStructure,
    Right,
    Side,
    StructureKind,
    StructureRisk,
)

AAPL = ticker("AAPL")
SEP = date(2026, 9, 18)
NOW = datetime(2026, 9, 1, 15, 30, tzinfo=UTC)

NON_FINITE = [
    pytest.param(Decimal("NaN"), id="nan"),
    pytest.param(Decimal("-NaN"), id="negative-nan"),
    pytest.param(Decimal("sNaN"), id="signalling-nan"),
    pytest.param(Decimal("Infinity"), id="infinity"),
    pytest.param(Decimal("-Infinity"), id="negative-infinity"),
]


def call_contract(strike: Decimal) -> OptionContract:
    return OptionContract(AAPL, SEP, strike, Right.CALL)


GOOD = call_contract(Decimal("150"))


def test_invariant_violation_is_the_documented_type() -> None:
    """It is a `DomainError`, and it is emphatically not an arithmetic error."""
    assert issubclass(InvariantViolation, DomainError)
    assert not issubclass(InvariantViolation, ArithmeticError)
    assert not issubclass(InvalidOperation, DomainError)


@pytest.mark.parametrize("value", NON_FINITE)
class TestNonFiniteIsADomainError:
    """Each constructor that validates a `Decimal`, one per money field."""

    def test_contract_strike(self, value: Decimal) -> None:
        with pytest.raises(InvariantViolation, match="strike must be finite"):
            call_contract(value)

    def test_quote_bid(self, value: Decimal) -> None:
        with pytest.raises(InvariantViolation, match="bid must be finite"):
            OptionQuote(GOOD, NOW, bid=value, ask=Decimal("2.00"))

    def test_quote_ask(self, value: Decimal) -> None:
        with pytest.raises(InvariantViolation, match="ask must be finite"):
            OptionQuote(GOOD, NOW, bid=Decimal("1.00"), ask=value)

    def test_cover_basis(self, value: Decimal) -> None:
        with pytest.raises(InvariantViolation, match="cover basis must be finite"):
            Cover(shares=100, basis=value)

    def test_cover_cash(self, value: Decimal) -> None:
        with pytest.raises(InvariantViolation, match="cover cash must be finite"):
            Cover(cash=value)

    def test_structure_risk_max_loss(self, value: Decimal) -> None:
        with pytest.raises(InvariantViolation, match="max_loss must be positive and finite"):
            StructureRisk(
                net_premium=Decimal(150),
                max_loss=value,
                max_profit=Decimal(150),
                breakevens=(),
                net_greeks=None,
                worst_spread_pct=Decimal("0.02"),
                days_to_expiry=10,
                quote_age_seconds=0.0,
            )


@pytest.mark.parametrize("value", NON_FINITE)
def test_no_validator_leaks_invalid_operation(value: Decimal) -> None:
    """The regression this file exists for.

    Before the fix, `max_loss <= 0` was evaluated before `is_finite()`, so a NaN
    left the constructor as `InvalidOperation`. Construction was still refused —
    the invariant held — but the caller was handed an exception it had no reason
    to expect from a validator.
    """
    builders = [
        lambda: call_contract(value),
        lambda: OptionQuote(GOOD, NOW, bid=value, ask=Decimal("2.00")),
        lambda: Cover(shares=100, basis=value),
        lambda: StructureRisk(
            net_premium=Decimal(150),
            max_loss=value,
            max_profit=Decimal(150),
            breakevens=(),
            net_greeks=None,
            worst_spread_pct=Decimal("0.02"),
            days_to_expiry=10,
            quote_age_seconds=0.0,
        ),
    ]
    for build in builders:
        try:
            build()
        except InvariantViolation:
            continue
        except ArithmeticError as exc:  # pragma: no cover - the bug this pins
            pytest.fail(f"validator raised {type(exc).__name__} instead of InvariantViolation")
        else:  # pragma: no cover - construction must not succeed
            pytest.fail(f"a non-finite {value} was accepted")


class TestFiniteValuesStillWork:
    """The guard must not have narrowed anything that was legal before."""

    def test_a_zero_bid_is_a_real_market(self) -> None:
        quote = OptionQuote(GOOD, NOW, bid=Decimal(0), ask=Decimal("0.05"))
        assert quote.bid == 0

    def test_a_very_small_strike_is_still_a_strike(self) -> None:
        assert call_contract(Decimal("0.001")).strike == Decimal("0.001")

    def test_a_normal_structure_still_builds(self) -> None:
        structure = OptionStructure(
            StructureKind.VERTICAL_CREDIT,
            (
                Leg(call_contract(Decimal("150")), Side.SELL),
                Leg(call_contract(Decimal("155")), Side.BUY),
            ),
        )
        assert structure.width == Decimal(5)

    def test_greeks_are_floats_and_are_not_policed_here(self) -> None:
        """Deliberate: greeks are estimates, and a NaN delta is caught downstream
        by the Gate's `net_delta_budget`, which vetoes an exposure it cannot
        compare against its band. Money is the thing that must be exact."""
        assert Greeks(float("nan"), 0.0, 0.0, 0.0, 0.0, 0.25).delta != 0.0
