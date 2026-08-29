"""02 D4, test plan item 5 — `max_loss` over generated inputs, for **every** kind.

The hand-computed fixtures in `test_risk.py` pin the arithmetic; this pins the
invariant. They are not the same job. A fixture proves the number is right for
the inputs someone thought of, and the inputs nobody thinks of are exactly where
a `max_loss` of zero, a negative one, or a `NaN` would come from.

That matters more here than it would elsewhere, because of what sits downstream:
`risk/checks.py` decides defined risk with `max_loss.is_finite() and max_loss > 0`
and nothing else. The entire "defined-risk structures only" claim — a hard gate
in specs/00, and the reason this project can say a naked short is unrepresentable
— reduces to that one expression being handed an honest number.

Until this file existed the property ran over `VERTICAL_CREDIT` alone, so the
four other kinds had the claim made on their behalf with no evidence behind it.

These were checked by mutation rather than assumed to work. Three of four
injected bugs are caught here; the one that matters is the fourth, because it is
caught here **and nowhere else**: dropping the quantity scaling from the
cash-secured put arm leaves every other test in `tests/options/` green, since
they all use one contract. A ten-lot would have reported the maximum loss of a
one-lot, and the Gate would have budgeted an order of magnitude too much risk
against a number that looked entirely reasonable. That is why the generators
draw a quantity.

Every generator produces a *constructible* structure at *non-arbitrage* prices.
The bounds are not decoration: a credit wider than the spread is free money, not
a market, and generating it would test what `compute_risk` does with impossible
input rather than what it does with real input. The Gate refuses a non-positive
`max_loss` anyway, which is the correct fail-closed answer to a quote that
strange.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

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

type Quotes = dict[OptionContract, OptionQuote]
type Generated = tuple[OptionStructure, Quotes]


def contract(strike: int, right: Right) -> OptionContract:
    return OptionContract(AAPL, SEP, Decimal(strike), right)


def priced(target: OptionContract, cents: int) -> OptionQuote:
    """A quote with no spread, so the mid is exact and the test is about risk.

    A generated bid/ask spread would make every assertion below a statement
    about rounding as well as about the invariant, and only one of those is
    under test here.
    """
    money = Decimal(cents) / 100
    return OptionQuote(target, NOW - timedelta(seconds=5), money, money, G)


# --------------------------------------------------------------------- #
# One generator per kind. Adding a kind means adding one here — enforced
# below, because a kind this property silently skipped would be a kind
# whose defined-risk claim nobody checked.
# --------------------------------------------------------------------- #


@st.composite
def vertical_credits(draw: st.DrawFn) -> Generated:
    """Short the near call, long the far one, credit bounded under the width."""
    short_strike = draw(st.integers(min_value=50, max_value=400))
    width = draw(st.integers(min_value=1, max_value=50))
    long_cents = draw(st.integers(min_value=1, max_value=2_000))
    credit_cents = draw(st.integers(min_value=1, max_value=width * 100 - 1))
    qty = draw(st.integers(min_value=1, max_value=20))

    short = contract(short_strike, Right.CALL)
    long = contract(short_strike + width, Right.CALL)
    return (
        OptionStructure(
            StructureKind.VERTICAL_CREDIT,
            (Leg(short, Side.SELL, qty), Leg(long, Side.BUY, qty)),
        ),
        {
            short: priced(short, long_cents + credit_cents),
            long: priced(long, long_cents),
        },
    )


@st.composite
def vertical_debits(draw: st.DrawFn) -> Generated:
    """Long the near call, short the far one. The debit is the maximum loss."""
    long_strike = draw(st.integers(min_value=50, max_value=400))
    width = draw(st.integers(min_value=1, max_value=50))
    short_cents = draw(st.integers(min_value=1, max_value=2_000))
    debit_cents = draw(st.integers(min_value=1, max_value=width * 100 - 1))
    qty = draw(st.integers(min_value=1, max_value=20))

    long = contract(long_strike, Right.CALL)
    short = contract(long_strike + width, Right.CALL)
    return (
        OptionStructure(
            StructureKind.VERTICAL_DEBIT,
            (Leg(long, Side.BUY, qty), Leg(short, Side.SELL, qty)),
        ),
        {
            long: priced(long, short_cents + debit_cents),
            short: priced(short, short_cents),
        },
    )


@st.composite
def iron_condors(draw: st.DrawFn) -> Generated:
    """Both wings, short strikes apart, total credit under the *wider* wing.

    The two wings get independent widths deliberately. `max_loss` charges the
    wider one and not both — only one side can finish in the money — so a
    generator that only ever produced symmetric condors would never exercise
    the branch that has to choose, which is the branch where double-counting
    would hide.
    """
    put_long_strike = draw(st.integers(min_value=50, max_value=300))
    put_width = draw(st.integers(min_value=1, max_value=40))
    gap = draw(st.integers(min_value=1, max_value=50))
    call_width = draw(st.integers(min_value=1, max_value=40))
    qty = draw(st.integers(min_value=1, max_value=20))

    put_short_strike = put_long_strike + put_width
    call_short_strike = put_short_strike + gap
    call_long_strike = call_short_strike + call_width

    widest = max(put_width, call_width)
    total_credit = draw(st.integers(min_value=2, max_value=widest * 100 - 1))
    put_credit = draw(st.integers(min_value=1, max_value=total_credit - 1))
    call_credit = total_credit - put_credit
    wing_cents = draw(st.integers(min_value=1, max_value=1_000))

    put_long = contract(put_long_strike, Right.PUT)
    put_short = contract(put_short_strike, Right.PUT)
    call_short = contract(call_short_strike, Right.CALL)
    call_long = contract(call_long_strike, Right.CALL)
    return (
        OptionStructure(
            StructureKind.IRON_CONDOR,
            (
                Leg(put_long, Side.BUY, qty),
                Leg(put_short, Side.SELL, qty),
                Leg(call_short, Side.SELL, qty),
                Leg(call_long, Side.BUY, qty),
            ),
        ),
        {
            put_long: priced(put_long, wing_cents),
            put_short: priced(put_short, wing_cents + put_credit),
            call_long: priced(call_long, wing_cents),
            call_short: priced(call_short, wing_cents + call_credit),
        },
    )


@st.composite
def covered_calls(draw: st.DrawFn) -> Generated:
    """Short call against stock. The loss is the basis, less the credit.

    Not the strike: the position loses the whole value of the shares if the
    underlying goes to zero, and what that is worth depends on what was paid.
    """
    strike = draw(st.integers(min_value=10, max_value=400))
    basis = draw(st.integers(min_value=1, max_value=400))
    qty = draw(st.integers(min_value=1, max_value=20))
    premium_cents = draw(st.integers(min_value=1, max_value=basis * 100 - 1))

    call = contract(strike, Right.CALL)
    return (
        OptionStructure(
            StructureKind.COVERED_CALL,
            (Leg(call, Side.SELL, qty),),
            cover=Cover(shares=100 * qty, basis=Decimal(basis)),
        ),
        {call: priced(call, premium_cents)},
    )


@st.composite
def cash_secured_puts(draw: st.DrawFn) -> Generated:
    """Short put, fully funded. Assignment to zero, less the credit — never
    unlimited, which is the whole reason the kind is representable at all."""
    strike = draw(st.integers(min_value=10, max_value=400))
    qty = draw(st.integers(min_value=1, max_value=20))
    premium_cents = draw(st.integers(min_value=1, max_value=strike * 100 - 1))

    put = contract(strike, Right.PUT)
    return (
        OptionStructure(
            StructureKind.CASH_SECURED_PUT,
            (Leg(put, Side.SELL, qty),),
            cover=Cover(cash=Decimal(strike) * 100 * qty),
        ),
        {put: priced(put, premium_cents)},
    )


BUILDERS: dict[StructureKind, st.SearchStrategy[Generated]] = {
    StructureKind.VERTICAL_CREDIT: vertical_credits(),
    StructureKind.VERTICAL_DEBIT: vertical_debits(),
    StructureKind.IRON_CONDOR: iron_condors(),
    StructureKind.COVERED_CALL: covered_calls(),
    StructureKind.CASH_SECURED_PUT: cash_secured_puts(),
}

EVERY_KIND = pytest.mark.parametrize("kind", list(StructureKind), ids=lambda k: k.value)
PROPERTY = settings(max_examples=150, suppress_health_check=[HealthCheck.too_slow])


def test_every_kind_has_a_generator() -> None:
    """"Adding a member is a change to the safety argument" — `StructureKind`
    says so itself. It must not be possible to add one this file skips."""
    assert set(BUILDERS) == set(StructureKind)


class TestMaxLossIsAlwaysDefined:
    """The sentence the Gate's defined-risk claim rests on."""

    @EVERY_KIND
    @PROPERTY
    @given(data=st.data())
    def test_it_is_finite_and_positive(
        self, kind: StructureKind, data: st.DataObject
    ) -> None:
        structure, quotes = data.draw(BUILDERS[kind])
        risk = compute_risk(structure, quotes, NOW)

        assert risk.max_loss.is_finite(), (
            f"{kind.value} produced a non-finite max_loss; `defined_risk` in the Gate "
            "calls .is_finite() on this and a NaN would have to answer, not raise"
        )
        assert risk.max_loss > 0, (
            f"{kind.value} produced max_loss {risk.max_loss} at non-arbitrage prices; "
            "the Gate reads a non-positive figure as undefined risk and vetoes, so this "
            "is the arithmetic being wrong rather than the trade being bad"
        )

    @EVERY_KIND
    @PROPERTY
    @given(data=st.data())
    def test_it_is_never_smaller_than_the_credit_taken(
        self, kind: StructureKind, data: st.DataObject
    ) -> None:
        """A structure cannot risk less than nothing while being paid.

        Catches the sign error that a finiteness check would sail past: a
        `max_loss` computed as `credit - width` instead of `width - credit`
        stays finite and positive over plenty of inputs, and is the maximum
        *profit* wearing the wrong name.
        """
        structure, quotes = data.draw(BUILDERS[kind])
        risk = compute_risk(structure, quotes, NOW)
        if risk.is_credit:
            assert risk.max_loss + risk.net_premium > 0

    @EVERY_KIND
    @PROPERTY
    @given(data=st.data())
    def test_the_numbers_the_gate_reads_are_decimal(
        self, kind: StructureKind, data: st.DataObject
    ) -> None:
        """Money is `Decimal` end to end — specs/01 Rule 3. A float that reached
        `max_loss` would compare and sum fine, and be wrong in the last place."""
        structure, quotes = data.draw(BUILDERS[kind])
        risk = compute_risk(structure, quotes, NOW)
        assert isinstance(risk.max_loss, Decimal)
        assert isinstance(risk.net_premium, Decimal)
        assert all(isinstance(point, Decimal) for point in risk.breakevens)

    @EVERY_KIND
    @PROPERTY
    @given(data=st.data())
    def test_it_is_deterministic(self, kind: StructureKind, data: st.DataObject) -> None:
        """Same structure, same quotes, same `as_of`, same answer — specs/01
        Rule 7, on the numbers the rest of the pipeline is downstream of."""
        structure, quotes = data.draw(BUILDERS[kind])
        assert compute_risk(structure, quotes, NOW) == compute_risk(structure, quotes, NOW)

    @EVERY_KIND
    @PROPERTY
    @given(data=st.data())
    def test_the_gate_would_accept_it_as_defined_risk(
        self, kind: StructureKind, data: st.DataObject
    ) -> None:
        """The property stated as the Gate actually asks it.

        `defined_risk` is one expression, and everything above is only useful
        because of what that expression does with the answer. Asserting it in
        the Gate's own words means a change to either side has to break this.
        """
        structure, quotes = data.draw(BUILDERS[kind])
        risk = compute_risk(structure, quotes, NOW)
        assert risk.max_loss.is_finite()
        assert risk.max_loss > 0
