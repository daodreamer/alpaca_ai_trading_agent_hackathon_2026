"""02 D1–D2 — contract identity, OCC rendering, quotes and greeks.

RED first. Every test here encodes a sentence from the spec; if a test is
deleted, a spec clause becomes unenforced.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest
from hypothesis import given
from hypothesis import strategies as st

from alphagate.core.errors import InvariantViolation
from alphagate.core.identifiers import ticker
from alphagate.options import (
    MAX_QUOTE_AGE,
    Greeks,
    OptionContract,
    OptionQuote,
    Right,
    format_occ,
    parse_occ,
)

AAPL = ticker("AAPL")
EXPIRY = date(2026, 9, 18)


def contract(
    strike: str = "150", right: Right = Right.CALL, expiry: date = EXPIRY
) -> OptionContract:
    return OptionContract(underlying=AAPL, expiry=expiry, strike=Decimal(strike), right=right)


class TestContractInvariants:
    def test_a_valid_contract_constructs(self) -> None:
        c = contract()
        assert c.strike == Decimal("150")
        assert c.multiplier == 100

    @pytest.mark.parametrize("strike", ["0", "-1", "-150.00"])
    def test_a_non_positive_strike_is_rejected(self, strike: str) -> None:
        with pytest.raises(InvariantViolation, match="strike"):
            contract(strike)

    @pytest.mark.parametrize("multiplier", [0, -100])
    def test_a_non_positive_multiplier_is_rejected(self, multiplier: int) -> None:
        with pytest.raises(InvariantViolation, match="multiplier"):
            OptionContract(AAPL, EXPIRY, Decimal("150"), Right.CALL, multiplier)

    def test_a_strike_finer_than_a_tenth_of_a_cent_is_rejected(self) -> None:
        """OCC encodes the strike in thousandths. A strike that cannot round-trip
        through that encoding is not a real contract, whatever a provider says."""
        with pytest.raises(InvariantViolation, match="strike"):
            contract("150.0001")

    def test_a_strike_too_large_for_the_encoding_is_rejected(self) -> None:
        with pytest.raises(InvariantViolation, match="strike"):
            contract("100000")

    def test_a_float_strike_is_refused(self) -> None:
        """Money is Decimal end to end; Decimal(0.1) is not 0.1."""
        with pytest.raises((InvariantViolation, TypeError)):
            OptionContract(AAPL, EXPIRY, 150.0, Right.CALL)  # type: ignore[arg-type]

    def test_contracts_are_frozen_and_hashable(self) -> None:
        from dataclasses import FrozenInstanceError

        c = contract()
        assert {c, contract()} == {c}
        with pytest.raises(FrozenInstanceError):
            c.strike = Decimal("160")  # type: ignore[misc]


class TestOccRendering:
    """D1 — OccSymbol is a rendering of a contract, not its identity."""

    def test_the_documented_example(self) -> None:
        assert format_occ(contract("150", Right.CALL)) == "AAPL260918C00150000"

    def test_a_put_renders_with_p(self) -> None:
        assert format_occ(contract("150", Right.PUT)) == "AAPL260918P00150000"

    def test_a_fractional_strike_renders_in_thousandths(self) -> None:
        assert format_occ(contract("152.50")) == "AAPL260918C00152500"
        assert format_occ(contract("7.5")) == "AAPL260918C00007500"

    def test_a_high_strike_renders_without_overflow(self) -> None:
        assert format_occ(contract("99999")) == "AAPL260918C99999000"

    def test_parse_recovers_every_field(self) -> None:
        c = parse_occ("AAPL260918C00152500")
        assert c.underlying == AAPL
        assert c.expiry == date(2026, 9, 18)
        assert c.right is Right.CALL
        assert c.strike == Decimal("152.5")

    @pytest.mark.parametrize(
        "bad",
        [
            "",
            "AAPL",
            "AAPL260918X00150000",  # unknown right
            "AAPL2609180C0150000",  # misplaced right
            "AAPL260918C0015000",  # seven strike digits
            "AAPL261318C00150000",  # month 13
            "AAPL260918C00000000",  # zero strike
            "260918C00150000",  # no root
        ],
    )
    def test_malformed_symbols_are_rejected(self, bad: str) -> None:
        with pytest.raises(InvariantViolation):
            parse_occ(bad)

    @given(
        strike_thousandths=st.integers(min_value=1, max_value=99_999_999),
        days=st.integers(min_value=0, max_value=3650),
        right=st.sampled_from(list(Right)),
        root=st.sampled_from(["A", "AAPL", "SPY", "BRKB", "GOOGL", "ABCDEF"]),
    )
    def test_round_trip(self, strike_thousandths: int, days: int, right: Right, root: str) -> None:
        """D1 — parse(format(c)) == c for any valid contract."""
        c = OptionContract(
            underlying=ticker(root),
            expiry=date(2020, 1, 1) + timedelta(days=days),
            strike=Decimal(strike_thousandths).scaleb(-3),
            right=right,
        )
        assert parse_occ(format_occ(c)) == c

    def test_the_symbol_is_not_the_identity(self) -> None:
        """Two contracts equal by field are equal, regardless of how they were made."""
        assert parse_occ("AAPL260918C00150000") == contract("150")


class TestQuote:
    def now(self) -> datetime:
        return datetime(2026, 9, 1, 15, 30, tzinfo=UTC)

    def quote(self, bid: str = "1.00", ask: str = "1.10", age: int = 0) -> OptionQuote:
        return OptionQuote(
            contract=contract(),
            as_of=self.now() - timedelta(seconds=age),
            bid=Decimal(bid),
            ask=Decimal(ask),
        )

    def test_mid_is_quantised_to_a_cent(self) -> None:
        assert self.quote("1.00", "1.10").mid == Decimal("1.05")

    def test_a_half_cent_mid_rounds_half_even(self) -> None:
        """1.025 goes to 1.02, not 1.03.

        Half-even rather than half-up, matching `core.numeric`: rounding ties
        consistently upward biases every mid in the same direction, and a system
        that quotes thousands of spreads accumulates that bias into its P&L.
        """
        assert self.quote("1.00", "1.05").mid == Decimal("1.02")
        assert self.quote("1.00", "1.07").mid == Decimal("1.04")

    def test_spread_pct_is_a_fraction_of_mid(self) -> None:
        q = self.quote("1.00", "1.10")
        assert q.spread_pct == pytest.approx(Decimal("0.0952"), abs=Decimal("0.0001"))

    def test_a_crossed_market_is_rejected(self) -> None:
        with pytest.raises(InvariantViolation, match="bid"):
            self.quote("1.20", "1.10")

    def test_a_negative_bid_is_rejected(self) -> None:
        with pytest.raises(InvariantViolation):
            self.quote("-0.01", "1.10")

    def test_a_naive_timestamp_is_rejected(self) -> None:
        """Rule 5: all times tz-aware UTC."""
        with pytest.raises(InvariantViolation, match="tz-aware"):
            OptionQuote(
                contract(),
                datetime(2026, 9, 1, 15, 30),  # noqa: DTZ001 - naive on purpose
                Decimal("1"),
                Decimal("2"),
            )

    def test_age_is_measured_against_a_supplied_time(self) -> None:
        """D5 / Rule 5 — the domain never reads the clock."""
        q = self.quote(age=90)
        assert q.age_seconds(self.now()) == pytest.approx(90.0)

    def test_staleness_is_reported_not_enforced(self) -> None:
        """D2 — the domain reports age; the Gate decides what is too old."""
        fresh, stale = self.quote(age=10), self.quote(age=90)
        assert not fresh.is_stale(self.now())
        assert stale.is_stale(self.now())
        assert stale.is_stale(self.now(), max_age=MAX_QUOTE_AGE)
        # Still constructible: the domain does not veto.
        assert stale.mid > 0

    def test_a_quote_from_the_future_has_negative_age(self) -> None:
        """Clock skew must be visible, not silently clamped to zero."""
        q = self.quote(age=-30)
        assert q.age_seconds(self.now()) == pytest.approx(-30.0)


class TestGreeks:
    def test_greeks_carry_iv(self) -> None:
        g = Greeks(delta=0.5, gamma=0.01, theta=-0.05, vega=0.2, rho=0.01, iv=0.35)
        assert g.iv == 0.35

    def test_greeks_are_floats_not_decimals(self) -> None:
        """01 D3 — greeks and IV are estimates, not money."""
        g = Greeks(0.5, 0.01, -0.05, 0.2, 0.01, 0.35)
        assert all(isinstance(v, float) for v in (g.delta, g.gamma, g.theta, g.vega, g.rho, g.iv))

    def test_greeks_scale_and_add(self) -> None:
        g = Greeks(0.5, 0.01, -0.05, 0.2, 0.01, 0.35)
        doubled = g.scaled(2)
        assert doubled.delta == pytest.approx(1.0)
        assert doubled.iv == pytest.approx(0.35), "IV is a rate, not an extensive quantity"
        summed = g.plus(g)
        assert summed.vega == pytest.approx(0.4)

    def test_a_quote_may_have_no_greeks(self) -> None:
        """D2 — missing greeks are None, never zero."""
        q = OptionQuote(contract(), datetime(2026, 9, 1, tzinfo=UTC), Decimal("1"), Decimal("2"))
        assert q.greeks is None
