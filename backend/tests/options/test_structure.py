"""02 D3 — structure, not leg.

The claim under test is the strongest one in the whole architecture:

    "If it cannot be built, it cannot be proposed, and it cannot be sent."

So most of these tests assert a *refusal*. A naked short is not something the
Gate vetoes; it is something that has no representation. If any test in
`TestNakedIsUnrepresentable` starts passing by constructing a structure, the
project's central safety claim has quietly become false.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from alphagate.core.errors import InvariantViolation
from alphagate.core.identifiers import Ticker, ticker
from alphagate.options import (
    Cover,
    Leg,
    OptionContract,
    OptionStructure,
    Right,
    Side,
    StructureKind,
)

AAPL = ticker("AAPL")
MSFT = ticker("MSFT")
SEP = date(2026, 9, 18)
OCT = date(2026, 10, 16)


def c(
    strike: str, right: Right = Right.CALL, expiry: date = SEP, sym: Ticker = AAPL
) -> OptionContract:
    return OptionContract(sym, expiry, Decimal(strike), right)


def leg(
    strike: str,
    side: Side,
    right: Right = Right.CALL,
    expiry: date = SEP,
    qty: int = 1,
    sym: Ticker = AAPL,
) -> Leg:
    return Leg(contract=c(strike, right, expiry, sym), side=side, quantity=qty)


def call_credit(qty: int = 1) -> OptionStructure:
    """Bear call spread: short the near strike, long the far one."""
    return OptionStructure(
        kind=StructureKind.VERTICAL_CREDIT,
        legs=(
            leg("150", Side.SELL, Right.CALL, qty=qty),
            leg("155", Side.BUY, Right.CALL, qty=qty),
        ),
    )


def put_credit(qty: int = 1) -> OptionStructure:
    """Bull put spread: short the near strike, long the far one."""
    return OptionStructure(
        kind=StructureKind.VERTICAL_CREDIT,
        legs=(
            leg("150", Side.SELL, Right.PUT, qty=qty),
            leg("145", Side.BUY, Right.PUT, qty=qty),
        ),
    )


def iron_condor(qty: int = 1) -> OptionStructure:
    return OptionStructure(
        kind=StructureKind.IRON_CONDOR,
        legs=(
            leg("140", Side.BUY, Right.PUT, qty=qty),
            leg("145", Side.SELL, Right.PUT, qty=qty),
            leg("155", Side.SELL, Right.CALL, qty=qty),
            leg("160", Side.BUY, Right.CALL, qty=qty),
        ),
    )


class TestLeg:
    def test_quantity_must_be_positive(self) -> None:
        """Direction lives in `side`. A negative quantity would give two ways to
        say "short", and they would eventually disagree."""
        with pytest.raises(InvariantViolation, match="quantity"):
            Leg(c("150"), Side.SELL, 0)
        with pytest.raises(InvariantViolation, match="quantity"):
            Leg(c("150"), Side.BUY, -1)


class TestVerticalCredit:
    def test_a_call_credit_spread_constructs(self) -> None:
        s = call_credit()
        assert s.kind is StructureKind.VERTICAL_CREDIT
        assert s.underlying == AAPL
        assert s.expiry == SEP

    def test_a_put_credit_spread_constructs(self) -> None:
        assert put_credit().kind is StructureKind.VERTICAL_CREDIT

    def test_the_short_call_must_be_the_lower_strike(self) -> None:
        """Selling the far strike and buying the near one is a debit, not a
        credit. Mislabelling it would make max_loss arithmetic silently wrong."""
        with pytest.raises(InvariantViolation, match="credit"):
            OptionStructure(
                StructureKind.VERTICAL_CREDIT,
                (leg("155", Side.SELL, Right.CALL), leg("150", Side.BUY, Right.CALL)),
            )

    def test_the_short_put_must_be_the_higher_strike(self) -> None:
        with pytest.raises(InvariantViolation, match="credit"):
            OptionStructure(
                StructureKind.VERTICAL_CREDIT,
                (leg("145", Side.SELL, Right.PUT), leg("150", Side.BUY, Right.PUT)),
            )

    def test_one_leg_does_not_construct(self) -> None:
        with pytest.raises(InvariantViolation, match="two legs"):
            OptionStructure(StructureKind.VERTICAL_CREDIT, (leg("150", Side.SELL),))

    def test_mismatched_underlyings_do_not_construct(self) -> None:
        with pytest.raises(InvariantViolation, match="underlying"):
            OptionStructure(
                StructureKind.VERTICAL_CREDIT,
                (leg("150", Side.SELL), leg("155", Side.BUY, sym=MSFT)),
            )

    def test_mismatched_expiries_do_not_construct(self) -> None:
        """A calendar spread has different risk. It is not this kind."""
        with pytest.raises(InvariantViolation, match="expir"):
            OptionStructure(
                StructureKind.VERTICAL_CREDIT,
                (leg("150", Side.SELL), leg("155", Side.BUY, expiry=OCT)),
            )

    def test_mismatched_rights_do_not_construct(self) -> None:
        with pytest.raises(InvariantViolation, match="right"):
            OptionStructure(
                StructureKind.VERTICAL_CREDIT,
                (leg("150", Side.SELL, Right.CALL), leg("155", Side.BUY, Right.PUT)),
            )

    def test_identical_strikes_do_not_construct(self) -> None:
        """Zero width means zero defined risk, which is a naked position wearing
        two legs."""
        with pytest.raises(InvariantViolation, match="strike"):
            OptionStructure(
                StructureKind.VERTICAL_CREDIT,
                (leg("150", Side.SELL), leg("150", Side.BUY)),
            )

    def test_unequal_leg_quantities_do_not_construct(self) -> None:
        """A ratio spread is unbounded on one side."""
        with pytest.raises(InvariantViolation, match="quantit"):
            OptionStructure(
                StructureKind.VERTICAL_CREDIT,
                (leg("150", Side.SELL, qty=2), leg("155", Side.BUY, qty=1)),
            )

    def test_two_shorts_do_not_construct(self) -> None:
        with pytest.raises(InvariantViolation):
            OptionStructure(
                StructureKind.VERTICAL_CREDIT,
                (leg("150", Side.SELL), leg("155", Side.SELL)),
            )

    def test_width_is_the_strike_distance(self) -> None:
        assert call_credit().width == Decimal("5")
        assert put_credit().width == Decimal("5")


class TestVerticalDebit:
    def test_a_call_debit_spread_constructs(self) -> None:
        s = OptionStructure(
            StructureKind.VERTICAL_DEBIT,
            (leg("150", Side.BUY, Right.CALL), leg("155", Side.SELL, Right.CALL)),
        )
        assert s.width == Decimal("5")

    def test_a_credit_shaped_debit_does_not_construct(self) -> None:
        with pytest.raises(InvariantViolation, match="debit"):
            OptionStructure(
                StructureKind.VERTICAL_DEBIT,
                (leg("150", Side.SELL, Right.CALL), leg("155", Side.BUY, Right.CALL)),
            )


class TestIronCondor:
    def test_a_condor_constructs(self) -> None:
        s = iron_condor()
        assert len(s.legs) == 4
        assert s.underlying == AAPL

    def test_leg_order_does_not_affect_validity(self) -> None:
        """The same four legs in any order are the same condor."""
        shuffled = OptionStructure(
            StructureKind.IRON_CONDOR,
            (
                leg("145", Side.SELL, Right.PUT),
                leg("140", Side.BUY, Right.PUT),
                leg("160", Side.BUY, Right.CALL),
                leg("155", Side.SELL, Right.CALL),
            ),
        )
        assert shuffled == iron_condor()

    def test_the_short_strikes_must_sit_inside_the_long_ones(self) -> None:
        """Longs inside shorts is a long condor: opposite risk, different kind."""
        with pytest.raises(InvariantViolation, match="inverted"):
            OptionStructure(
                StructureKind.IRON_CONDOR,
                (
                    leg("140", Side.SELL, Right.PUT),
                    leg("145", Side.BUY, Right.PUT),
                    leg("160", Side.SELL, Right.CALL),
                    leg("155", Side.BUY, Right.CALL),
                ),
            )

    def test_three_legs_do_not_construct(self) -> None:
        with pytest.raises(InvariantViolation, match="four legs"):
            OptionStructure(StructureKind.IRON_CONDOR, iron_condor().legs[:3])

    def test_the_put_side_must_sit_below_the_call_side(self) -> None:
        with pytest.raises(InvariantViolation):
            OptionStructure(
                StructureKind.IRON_CONDOR,
                (
                    leg("160", Side.BUY, Right.PUT),
                    leg("165", Side.SELL, Right.PUT),
                    leg("145", Side.SELL, Right.CALL),
                    leg("140", Side.BUY, Right.CALL),
                ),
            )

    def test_width_is_the_wider_wing(self) -> None:
        s = OptionStructure(
            StructureKind.IRON_CONDOR,
            (
                leg("130", Side.BUY, Right.PUT),
                leg("145", Side.SELL, Right.PUT),
                leg("155", Side.SELL, Right.CALL),
                leg("160", Side.BUY, Right.CALL),
            ),
        )
        assert s.width == Decimal("15"), "max loss is set by the wider side"


class TestCoveredStructures:
    def test_a_covered_call_needs_shares(self) -> None:
        with pytest.raises(InvariantViolation, match="cover"):
            OptionStructure(StructureKind.COVERED_CALL, (leg("150", Side.SELL, Right.CALL),))

    def test_a_covered_call_with_enough_shares_constructs(self) -> None:
        s = OptionStructure(
            StructureKind.COVERED_CALL,
            (leg("150", Side.SELL, Right.CALL),),
            cover=Cover(shares=100, basis=Decimal("140")),
        )
        assert s.cover is not None
        assert s.cover.shares == 100

    def test_too_few_shares_do_not_construct(self) -> None:
        """99 shares against one contract leaves one share-equivalent naked."""
        with pytest.raises(InvariantViolation, match="cover"):
            OptionStructure(
                StructureKind.COVERED_CALL,
                (leg("150", Side.SELL, Right.CALL),),
                cover=Cover(shares=99, basis=Decimal("140")),
            )

    def test_a_covered_call_needs_a_basis(self) -> None:
        """Without the basis, max loss is unstateable."""
        with pytest.raises(InvariantViolation, match="basis"):
            OptionStructure(
                StructureKind.COVERED_CALL,
                (leg("150", Side.SELL, Right.CALL),),
                cover=Cover(shares=100),
            )

    def test_a_cash_secured_put_needs_the_full_assignment_cost(self) -> None:
        with pytest.raises(InvariantViolation, match="cover"):
            OptionStructure(
                StructureKind.CASH_SECURED_PUT,
                (leg("150", Side.SELL, Right.PUT),),
                cover=Cover(cash=Decimal("14000")),
            )

    def test_a_fully_secured_put_constructs(self) -> None:
        s = OptionStructure(
            StructureKind.CASH_SECURED_PUT,
            (leg("150", Side.SELL, Right.PUT),),
            cover=Cover(cash=Decimal("15000")),
        )
        assert s.cover is not None

    def test_a_covered_call_must_be_short(self) -> None:
        with pytest.raises(InvariantViolation):
            OptionStructure(
                StructureKind.COVERED_CALL,
                (leg("150", Side.BUY, Right.CALL),),
                cover=Cover(shares=100, basis=Decimal("140")),
            )


class TestNakedIsUnrepresentable:
    """The safety claim. Each of these must stay a refusal, forever."""

    def test_there_is_no_custom_kind(self) -> None:
        assert not any(k.name in ("CUSTOM", "NAKED", "FREEFORM") for k in StructureKind)

    def test_every_kind_is_one_of_the_five_in_the_spec(self) -> None:
        assert {k.name for k in StructureKind} == {
            "VERTICAL_CREDIT",
            "VERTICAL_DEBIT",
            "IRON_CONDOR",
            "COVERED_CALL",
            "CASH_SECURED_PUT",
        }

    @pytest.mark.parametrize("kind", list(StructureKind))
    def test_a_lone_uncovered_short_never_constructs(self, kind: StructureKind) -> None:
        with pytest.raises(InvariantViolation):
            OptionStructure(kind, (leg("150", Side.SELL, Right.CALL),))

    @pytest.mark.parametrize("kind", list(StructureKind))
    def test_a_lone_uncovered_short_put_never_constructs(self, kind: StructureKind) -> None:
        with pytest.raises(InvariantViolation):
            OptionStructure(kind, (leg("150", Side.SELL, Right.PUT),))

    def test_no_structure_has_more_short_than_long_contracts_without_cover(self) -> None:
        """Every constructible spread is balanced or explicitly covered."""
        for s in (call_credit(), put_credit(), iron_condor()):
            shorts = sum(leg.quantity for leg in s.legs if leg.side is Side.SELL)
            longs = sum(leg.quantity for leg in s.legs if leg.side is Side.BUY)
            assert shorts == longs, f"{s.kind} is unbalanced and uncovered"


class TestStructureProperties:
    def test_structures_are_frozen_and_hashable(self) -> None:
        assert {call_credit(), call_credit()} == {call_credit()}

    def test_leg_order_does_not_change_identity(self) -> None:
        """Two orderings of the same legs are the same structure."""
        a = call_credit()
        b = OptionStructure(StructureKind.VERTICAL_CREDIT, tuple(reversed(a.legs)))
        assert a == b, "leg order leaked into structure identity"

    def test_quantity_is_the_common_leg_quantity(self) -> None:
        assert call_credit(qty=3).quantity == 3

    def test_days_to_expiry_takes_the_date_as_an_argument(self) -> None:
        assert call_credit().days_to_expiry(date(2026, 9, 1)) == 17

    def test_short_legs_are_reported(self) -> None:
        assert len(call_credit().short_legs) == 1
        assert len(iron_condor().short_legs) == 2
