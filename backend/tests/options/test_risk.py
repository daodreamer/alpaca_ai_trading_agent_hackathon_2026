"""02 D4–D5 — the numbers the Gate needs.

Every arithmetic test uses a hand-computed fixture with exact Decimals. A
tolerance here would hide precisely the class of bug this module can have: an
off-by-a-multiplier, a sign flip, a float creeping into money.

The invariant that matters most is D4's: **`max_loss` is finite and positive for
every constructible structure.** That is asserted directly, and then again over
generated inputs, because it is the sentence the Gate's whole defined-risk claim
rests on.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from alphagate.core.errors import InvariantViolation
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
    compute_risk,
)

AAPL = ticker("AAPL")
SEP = date(2026, 9, 18)
NOW = datetime(2026, 9, 1, 15, 30, tzinfo=UTC)
G = Greeks(delta=0.30, gamma=0.01, theta=-0.05, vega=0.10, rho=0.01, iv=0.25)


def c(strike: str, right: Right = Right.CALL) -> OptionContract:
    return OptionContract(AAPL, SEP, Decimal(strike), right)


def leg(strike: str, side: Side, right: Right = Right.CALL, qty: int = 1) -> Leg:
    return Leg(c(strike, right), side, qty)


def q(
    contract: OptionContract,
    bid: str,
    ask: str,
    *,
    greeks: Greeks | None = G,
    age: int = 0,
) -> OptionQuote:
    return OptionQuote(contract, NOW - timedelta(seconds=age), Decimal(bid), Decimal(ask), greeks)


class TestVerticalCredit:
    """Short 150 call / long 155 call for a 1.50 credit, one contract."""

    def structure(self, qty: int = 1) -> OptionStructure:
        return OptionStructure(
            StructureKind.VERTICAL_CREDIT,
            (leg("150", Side.SELL, qty=qty), leg("155", Side.BUY, qty=qty)),
        )

    def quotes(self) -> dict[OptionContract, OptionQuote]:
        return {
            c("150"): q(c("150"), "3.00", "3.00"),
            c("155"): q(c("155"), "1.50", "1.50"),
        }

    def test_net_premium_is_a_positive_credit(self) -> None:
        r = compute_risk(self.structure(), self.quotes(), NOW)
        assert r.net_premium == Decimal("150.00"), "3.00 - 1.50, times 100"
        assert r.is_credit

    def test_max_loss_is_the_width_less_the_credit(self) -> None:
        r = compute_risk(self.structure(), self.quotes(), NOW)
        assert r.max_loss == Decimal("350.00"), "5.00 wide x 100, less 150 credit"

    def test_max_profit_is_the_credit(self) -> None:
        assert compute_risk(self.structure(), self.quotes(), NOW).max_profit == Decimal("150.00")

    def test_everything_scales_with_quantity(self) -> None:
        one = compute_risk(self.structure(1), self.quotes(), NOW)
        three = compute_risk(self.structure(3), self.quotes(), NOW)
        assert three.net_premium == one.net_premium * 3
        assert three.max_loss == one.max_loss * 3

    def test_the_breakeven_is_the_short_strike_plus_the_credit(self) -> None:
        r = compute_risk(self.structure(), self.quotes(), NOW)
        assert r.breakevens == (Decimal("151.50"),)

    def test_return_on_risk(self) -> None:
        r = compute_risk(self.structure(), self.quotes(), NOW)
        assert r.return_on_risk == pytest.approx(Decimal("150") / Decimal("350"))

    def test_days_to_expiry_comes_from_the_supplied_time(self) -> None:
        r = compute_risk(self.structure(), self.quotes(), NOW)
        assert r.days_to_expiry == 17

    def test_worst_spread_is_the_widest_leg(self) -> None:
        quotes = {
            c("150"): q(c("150"), "2.90", "3.10"),  # ~6.7% of 3.00
            c("155"): q(c("155"), "1.49", "1.51"),  # ~1.3% of 1.50
        }
        r = compute_risk(self.structure(), quotes, NOW)
        assert r.worst_spread_pct > Decimal("0.06")

    def test_quote_age_is_the_oldest_leg(self) -> None:
        quotes = {
            c("150"): q(c("150"), "3.00", "3.00", age=5),
            c("155"): q(c("155"), "1.50", "1.50", age=45),
        }
        assert compute_risk(self.structure(), quotes, NOW).quote_age_seconds == 45.0


class TestVerticalDebit:
    """Long 150 call / short 155 call for a 1.50 debit."""

    def structure(self) -> OptionStructure:
        return OptionStructure(
            StructureKind.VERTICAL_DEBIT,
            (leg("150", Side.BUY), leg("155", Side.SELL)),
        )

    def quotes(self) -> dict[OptionContract, OptionQuote]:
        return {
            c("150"): q(c("150"), "3.00", "3.00"),
            c("155"): q(c("155"), "1.50", "1.50"),
        }

    def test_net_premium_is_a_negative_debit(self) -> None:
        r = compute_risk(self.structure(), self.quotes(), NOW)
        assert r.net_premium == Decimal("-150.00")
        assert not r.is_credit

    def test_max_loss_is_the_debit_paid(self) -> None:
        assert compute_risk(self.structure(), self.quotes(), NOW).max_loss == Decimal("150.00")

    def test_max_profit_is_the_width_less_the_debit(self) -> None:
        assert compute_risk(self.structure(), self.quotes(), NOW).max_profit == Decimal("350.00")


class TestIronCondor:
    def structure(self) -> OptionStructure:
        return OptionStructure(
            StructureKind.IRON_CONDOR,
            (
                leg("140", Side.BUY, Right.PUT),
                leg("145", Side.SELL, Right.PUT),
                leg("155", Side.SELL, Right.CALL),
                leg("160", Side.BUY, Right.CALL),
            ),
        )

    def quotes(self) -> dict[OptionContract, OptionQuote]:
        return {
            c("140", Right.PUT): q(c("140", Right.PUT), "0.50", "0.50"),
            c("145", Right.PUT): q(c("145", Right.PUT), "1.20", "1.20"),
            c("155", Right.CALL): q(c("155", Right.CALL), "1.30", "1.30"),
            c("160", Right.CALL): q(c("160", Right.CALL), "0.60", "0.60"),
        }

    def test_net_premium_sums_both_wings(self) -> None:
        r = compute_risk(self.structure(), self.quotes(), NOW)
        # (1.20 - 0.50) + (1.30 - 0.60) = 1.40, times 100
        assert r.net_premium == Decimal("140.00")

    def test_max_loss_uses_one_wing_not_both(self) -> None:
        """Only one side can finish in the money. Charging both would double-count."""
        r = compute_risk(self.structure(), self.quotes(), NOW)
        assert r.max_loss == Decimal("360.00"), "5.00 wide x 100, less 140 credit"

    def test_there_are_two_breakevens(self) -> None:
        r = compute_risk(self.structure(), self.quotes(), NOW)
        assert r.breakevens == (Decimal("143.60"), Decimal("156.40"))


class TestCoveredStructures:
    def test_a_cash_secured_put_loses_to_zero_not_to_infinity(self) -> None:
        """D4 — 'assignment to zero, not unlimited'."""
        s = OptionStructure(
            StructureKind.CASH_SECURED_PUT,
            (leg("150", Side.SELL, Right.PUT),),
            cover=Cover(cash=Decimal("15000")),
        )
        quotes = {c("150", Right.PUT): q(c("150", Right.PUT), "2.00", "2.00")}
        r = compute_risk(s, quotes, NOW)
        assert r.net_premium == Decimal("200.00")
        assert r.max_loss == Decimal("14800.00"), "150 x 100 assignment, less 200 credit"
        assert r.breakevens == (Decimal("148.00"),)

    def test_a_covered_call_loses_the_stock_basis(self) -> None:
        s = OptionStructure(
            StructureKind.COVERED_CALL,
            (leg("150", Side.SELL, Right.CALL),),
            cover=Cover(shares=100, basis=Decimal("140")),
        )
        quotes = {c("150"): q(c("150"), "2.00", "2.00")}
        r = compute_risk(s, quotes, NOW)
        assert r.max_loss == Decimal("13800.00"), "140 x 100 basis, less 200 credit"
        assert r.max_profit == Decimal("1200.00"), "10 x 100 appreciation, plus 200 credit"


class TestGreeks:
    def structure(self) -> OptionStructure:
        return OptionStructure(
            StructureKind.VERTICAL_CREDIT,
            (leg("150", Side.SELL), leg("155", Side.BUY)),
        )

    def test_net_greeks_are_signed_and_scaled_by_the_multiplier(self) -> None:
        quotes = {
            c("150"): q(
                c("150"), "3.00", "3.00", greeks=Greeks(0.40, 0.01, -0.05, 0.10, 0.01, 0.25)
            ),
            c("155"): q(
                c("155"), "1.50", "1.50", greeks=Greeks(0.20, 0.01, -0.03, 0.08, 0.01, 0.22)
            ),
        }
        r = compute_risk(self.structure(), quotes, NOW)
        assert r.net_greeks is not None
        # short 0.40 + long 0.20 = -40 + 20 = -20 deltas
        assert r.net_greeks.delta == pytest.approx(-20.0)
        assert r.net_greeks.vega == pytest.approx(-2.0)

    def test_one_missing_leg_makes_the_net_none(self) -> None:
        """D4 — never a partial sum, which would understate exposure."""
        quotes = {
            c("150"): q(c("150"), "3.00", "3.00", greeks=None),
            c("155"): q(c("155"), "1.50", "1.50"),
        }
        assert compute_risk(self.structure(), quotes, NOW).net_greeks is None

    def test_missing_greeks_are_never_read_as_zero(self) -> None:
        quotes = {
            c("150"): q(c("150"), "3.00", "3.00", greeks=None),
            c("155"): q(c("155"), "1.50", "1.50", greeks=None),
        }
        r = compute_risk(self.structure(), quotes, NOW)
        assert r.net_greeks is None, "a delta-unknown structure read as delta-neutral"


class TestContract:
    def test_a_missing_quote_is_an_error_not_a_zero(self) -> None:
        s = OptionStructure(
            StructureKind.VERTICAL_CREDIT, (leg("150", Side.SELL), leg("155", Side.BUY))
        )
        with pytest.raises(InvariantViolation, match="no quote"):
            compute_risk(s, {c("150"): q(c("150"), "3.00", "3.00")}, NOW)

    def test_a_naive_as_of_is_refused(self) -> None:
        s = OptionStructure(
            StructureKind.VERTICAL_CREDIT, (leg("150", Side.SELL), leg("155", Side.BUY))
        )
        quotes = {c("150"): q(c("150"), "3.00", "3.00"), c("155"): q(c("155"), "1.50", "1.50")}
        with pytest.raises(InvariantViolation, match="tz-aware"):
            compute_risk(s, quotes, datetime(2026, 9, 1, 15, 30))  # noqa: DTZ001

    def test_money_never_becomes_a_float(self) -> None:
        s = OptionStructure(
            StructureKind.VERTICAL_CREDIT, (leg("150", Side.SELL), leg("155", Side.BUY))
        )
        quotes = {c("150"): q(c("150"), "3.00", "3.00"), c("155"): q(c("155"), "1.50", "1.50")}
        r = compute_risk(s, quotes, NOW)
        for value in (r.net_premium, r.max_loss, r.max_profit, r.worst_spread_pct, *r.breakevens):
            assert isinstance(value, Decimal)


class TestDeterminismAndInvariants:
    """D5 and the D4 max_loss invariant, over generated inputs."""

    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    @given(
        short_strike=st.integers(min_value=50, max_value=400),
        width=st.integers(min_value=1, max_value=50),
        credit_cents=st.integers(min_value=1, max_value=400),
        qty=st.integers(min_value=1, max_value=20),
    )
    def test_credit_spread_max_loss_is_always_finite_and_positive(
        self, short_strike: int, width: int, credit_cents: int, qty: int
    ) -> None:
        long_strike = short_strike + width
        credit = Decimal(credit_cents) / 100
        # A credit above the width is arbitrage, not a market we need to model.
        assume(credit < Decimal(width))

        s = OptionStructure(
            StructureKind.VERTICAL_CREDIT,
            (
                leg(str(short_strike), Side.SELL, qty=qty),
                leg(str(long_strike), Side.BUY, qty=qty),
            ),
        )
        quotes = {
            c(str(short_strike)): q(c(str(short_strike)), str(credit + 1), str(credit + 1)),
            c(str(long_strike)): q(c(str(long_strike)), "1.00", "1.00"),
        }
        r = compute_risk(s, quotes, NOW)
        assert r.max_loss > 0
        assert r.max_loss.is_finite()
        assert r.max_loss + r.max_profit == s.width * s.multiplier * qty

    def test_identical_inputs_give_identical_output(self) -> None:
        s = OptionStructure(
            StructureKind.VERTICAL_CREDIT, (leg("150", Side.SELL), leg("155", Side.BUY))
        )
        quotes = {c("150"): q(c("150"), "3.00", "3.00"), c("155"): q(c("155"), "1.50", "1.50")}
        first = compute_risk(s, quotes, NOW)
        for _ in range(100):
            assert compute_risk(s, quotes, NOW) == first

    def test_leg_order_does_not_change_the_result(self) -> None:
        legs = (leg("150", Side.SELL), leg("155", Side.BUY))
        quotes = {c("150"): q(c("150"), "3.00", "3.00"), c("155"): q(c("155"), "1.50", "1.50")}
        a = compute_risk(OptionStructure(StructureKind.VERTICAL_CREDIT, legs), quotes, NOW)
        b = compute_risk(
            OptionStructure(StructureKind.VERTICAL_CREDIT, tuple(reversed(legs))), quotes, NOW
        )
        assert a == b
