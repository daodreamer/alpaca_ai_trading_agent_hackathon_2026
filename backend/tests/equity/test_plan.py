"""Planning a rebalance — specs/09 D2, D3, D4, D6; test plan items 3–6.

The baseline (see `conftest.py`) is a book that is already held exactly, so the
plan is empty. Every test below moves one thing and asserts on the one order, or
the one skip, that results.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import datetime
from decimal import Decimal

import pytest

from alphagate.core.errors import InvariantViolation
from alphagate.core.identifiers import Ticker
from alphagate.equity import (
    EquityPolicy,
    EquitySide,
    Holding,
    Mark,
    OrderIntent,
    RebalancePlan,
    Skipped,
    SkipReason,
    TargetBook,
    plan_rebalance,
)
from tests.equity.conftest import AAA, BBB, CCC, EQUITY, NOW


def plan(
    book: TargetBook,
    holdings: Sequence[Holding],
    marks: Mapping[Ticker, Mark],
    policy: EquityPolicy,
    **overrides: object,
) -> RebalancePlan:
    return plan_rebalance(
        book,
        holdings=holdings,
        marks=marks,
        equity=overrides.pop("equity", EQUITY),
        policy=policy,
        as_of=overrides.pop("as_of", NOW),
        **overrides,
    )


def only(plan_result: RebalancePlan) -> OrderIntent:
    assert len(plan_result.intents) == 1, [i.label() for i in plan_result.intents]
    return plan_result.intents[0]


def skip_for(plan_result: RebalancePlan, symbol: Ticker) -> Skipped:
    matches = [s for s in plan_result.skipped if s.symbol == symbol]
    assert matches, f"{symbol} was neither traded nor skipped"
    return matches[0]


# --------------------------------------------------------------------- #
# The band — specs/09 D3, test plan item 5
# --------------------------------------------------------------------- #


def test_a_book_already_held_produces_no_orders(
    book: TargetBook,
    holdings: list[Holding],
    marks: dict[Ticker, Mark],
    policy: EquityPolicy,
) -> None:
    """Four days in five, this is the whole answer.

    And it is journalled rather than silent: every symbol comes back as a stated
    reason there is no order.
    """
    result = plan(book, holdings, marks, policy)
    assert result.is_empty
    assert result.counts() == {"inside_band": 3}


def test_drift_inside_the_band_is_left_alone(
    book: TargetBook,
    holdings: list[Holding],
    marks: dict[Ticker, Mark],
    policy: EquityPolicy,
) -> None:
    """$1,000 of drift on a $10,000 position, against a 20% band. Churn."""
    drifted = [replace(holdings[0], shares=Decimal(90)), *holdings[1:]]
    result = plan(book, drifted, marks, policy)
    assert result.is_empty
    assert skip_for(result, AAA).reason is SkipReason.INSIDE_BAND


def test_drift_exactly_at_the_band_is_traded(
    book: TargetBook,
    holdings: list[Holding],
    marks: dict[Ticker, Mark],
    policy: EquityPolicy,
) -> None:
    """Test plan item 5. The boundary is inclusive on the trading side.

    AAA's target is $10,000, so its band is 20% of that — $2,000, or 20 shares
    of a $100 stock. Holding 80 leaves the target exactly $2,000 away.
    """
    drifted = [replace(holdings[0], shares=Decimal(80)), *holdings[1:]]
    result = plan(book, drifted, marks, policy)
    intent = only(result)
    assert intent.symbol == AAA
    assert intent.side is EquitySide.BUY
    assert intent.shares == Decimal(20)


def test_each_skip_carries_the_band_it_was_measured_against(
    book: TargetBook,
    holdings: list[Holding],
    marks: dict[Ticker, Mark],
    policy: EquityPolicy,
) -> None:
    """The band is per position, so one global number on the plan would be wrong
    about most of its own lines."""
    result = plan(book, holdings, marks, policy)
    assert result.band_pct == Decimal("0.20")
    assert skip_for(result, AAA).threshold == Decimal(2_000)  # 20% of $10,000
    assert skip_for(result, CCC).threshold == Decimal(800)  # 20% of $4,000


def test_a_sleeve_sized_position_can_be_established(
    book: TargetBook,
    marks: dict[Ticker, Mark],
    policy: EquityPolicy,
) -> None:
    """The regression that motivated the proportional band — specs/09 D3.

    Under a band of 0.25% of equity, a $192 sleeve position sat permanently
    inside a $253 threshold and could never be bought. The real book holds a
    hundred such names, so the executable book was a tenth of the strategy and
    nothing reported an error.
    """
    from dataclasses import replace as _replace

    sleeve_weight = Decimal("0.00192")
    small = _replace(book, weights={**book.weights, CCC: sleeve_weight})
    result = plan(small, [], marks, policy)
    assert any(i.symbol == CCC for i in result.intents), (
        "a $192 position must be establishable; it is the shape of 100 of the 104"
    )


def test_the_absolute_floor_binds_on_a_tiny_position(policy: EquityPolicy) -> None:
    """20% of a $20 position is $4, and a $4 order is all spread."""
    assert policy.threshold(Decimal(20), Decimal(0)) == policy.min_order_notional


def test_the_band_is_measured_against_the_larger_side(policy: EquityPolicy) -> None:
    """One rule for establishing, drifting and exiting.

    Exiting is the case that needs the `max`: the target is zero, so a band
    proportional to the *target* alone would be the floor, and a $40,000 position
    would be closed by a $25 order every time its price moved.
    """
    assert policy.threshold(Decimal(0), Decimal(40_000)) == Decimal(8_000)
    assert policy.threshold(Decimal(40_000), Decimal(0)) == Decimal(8_000)


# --------------------------------------------------------------------- #
# Sizing — specs/09 D2
# --------------------------------------------------------------------- #


def test_weights_become_shares_at_the_marked_price(
    book: TargetBook,
    marks: dict[Ticker, Mark],
    policy: EquityPolicy,
) -> None:
    """A 0.10 weight on $100k is $10,000; at $100 a share that is 100 shares."""
    result = plan(book, [], marks, policy)
    aaa = next(i for i in result.intents if i.symbol == AAA)
    assert aaa.side is EquitySide.BUY
    assert aaa.shares == Decimal(100)
    assert aaa.notional == Decimal(10_000)


def test_rounding_is_toward_zero_so_a_buy_never_overshoots(
    book: TargetBook,
    marks: dict[Ticker, Mark],
    policy: EquityPolicy,
) -> None:
    """`ROUND_DOWN` on a `Decimal` is toward zero, which is safe on both sides:
    a buy stops short of the target, a sell stops short of the holding."""
    priced = {**marks, AAA: replace(marks[AAA], price=Decimal(3))}
    result = plan(book, [], priced, policy)
    aaa = next(i for i in result.intents if i.symbol == AAA)
    # 10000 / 3 = 3333.3333..., truncated at four places.
    assert aaa.shares == Decimal("3333.3333")
    assert aaa.notional < Decimal(10_000)


def test_a_whole_share_asset_is_floored(
    book: TargetBook,
    marks: dict[Ticker, Mark],
    policy: EquityPolicy,
) -> None:
    priced = {**marks, CCC: replace(marks[CCC], fractionable=False, price=Decimal(30))}
    result = plan(book, [], priced, policy)
    ccc = next(i for i in result.intents if i.symbol == CCC)
    assert ccc.shares == Decimal(133)  # 4000 / 30 = 133.33
    assert ccc.shares == ccc.shares.to_integral_value()


def test_a_whole_share_target_below_one_share_is_a_recorded_hole(
    book: TargetBook,
    holdings: list[Holding],
    marks: dict[Ticker, Mark],
    policy: EquityPolicy,
) -> None:
    """Test plan item 6. A $192 sleeve position in a $500 non-fractionable name
    cannot be held, and the record says so rather than dropping it.

    Here the target is $4,000 and one share costs $25,000, so the book asks for
    0.16 of a share of something that does not divide.
    """
    small = replace(marks[CCC], fractionable=False, price=Decimal(25_000))
    unheld = [h for h in holdings if h.symbol != CCC]
    result = plan(book, unheld, {**marks, CCC: small}, policy)
    skipped = skip_for(result, CCC)
    assert skipped.reason is SkipReason.ROUNDS_TO_ZERO
    assert "less than one share" in skipped.detail


# --------------------------------------------------------------------- #
# Exits — specs/09 D2's third point
# --------------------------------------------------------------------- #


def test_a_symbol_that_left_the_book_is_sold_to_zero(
    book: TargetBook,
    holdings: list[Holding],
    marks: dict[Ticker, Mark],
    policy: EquityPolicy,
) -> None:
    """The exit path, and it is the same arithmetic as everything else.

    There is no separate close routine to forget to call: a symbol absent from
    the book has a target of zero, so its delta is the whole position.
    """
    held = [*holdings, Holding(ticker_ddd := _ddd(), Decimal(40), Decimal(9), Decimal(400))]
    priced = {**marks, ticker_ddd: Mark(ticker_ddd, Decimal(10), 1.0, True, True)}
    result = plan(book, held, priced, policy)
    exits = [i for i in result.intents if i.symbol == ticker_ddd]
    assert len(exits) == 1
    assert exits[0].side is EquitySide.SELL
    assert exits[0].shares == Decimal(40)
    assert exits[0].resulting_shares == Decimal(0)


def test_a_symbol_neither_held_nor_wanted_is_not_an_order(
    book: TargetBook,
    holdings: list[Holding],
    marks: dict[Ticker, Mark],
    policy: EquityPolicy,
) -> None:
    result = plan(book, holdings, marks, policy)
    assert all(s.reason is not SkipReason.ALREADY_FLAT for s in result.skipped)


# --------------------------------------------------------------------- #
# No shorts — specs/09 D6
# --------------------------------------------------------------------- #


def test_a_sell_is_clamped_to_what_is_held(
    book: TargetBook,
    marks: dict[Ticker, Mark],
    policy: EquityPolicy,
) -> None:
    """The planner is the first of three layers that refuse a short."""
    held = [Holding(_ddd(), Decimal(5), Decimal(10), Decimal(50))]
    priced = {**marks, _ddd(): Mark(_ddd(), Decimal(100), 1.0, True, True)}
    result = plan(book, held, priced, policy)
    sells = [i for i in result.intents if i.side is EquitySide.SELL]
    assert all(i.shares <= i.held_shares for i in sells)
    assert all(i.resulting_shares >= 0 for i in sells)


def test_an_intent_that_would_open_a_short_is_unconstructible() -> None:
    """The second layer. The type refuses even if a caller builds one by hand."""
    from alphagate.equity import OrderIntent

    with pytest.raises(InvariantViolation, match="would open a short"):
        OrderIntent(
            symbol=AAA,
            side=EquitySide.SELL,
            shares=Decimal(10),
            reference_price=Decimal(100),
            target_weight=Decimal(0),
            held_weight=Decimal(0),
            held_shares=Decimal(5),
            fractionable=True,
            mark_age_seconds=1.0,
        )


# --------------------------------------------------------------------- #
# Sequence — specs/09 D4, test plan item 4
# --------------------------------------------------------------------- #


def test_sells_precede_buys(
    book: TargetBook,
    marks: dict[Ticker, Mark],
    policy: EquityPolicy,
) -> None:
    """Sells release the buying power the buys then spend.

    The other order, on a fully-invested book, refuses the first buys for want of
    cash and leaves the last sells sitting in it — a rebalance that half
    happened, which is worse than either end state.
    """
    held = [
        Holding(AAA, Decimal(200), Decimal(90), Decimal(20_000)),
        Holding(_ddd(), Decimal(300), Decimal(100), Decimal(30_000)),
    ]
    priced = {**marks, _ddd(): Mark(_ddd(), Decimal(100), 1.0, True, True)}
    result = plan(book, held, priced, policy)
    sides = [i.side for i in result.intents]
    assert sides.index(EquitySide.BUY) > max(
        i for i, s in enumerate(sides) if s is EquitySide.SELL
    )


def test_within_a_side_the_largest_notional_goes_first(
    book: TargetBook,
    marks: dict[Ticker, Mark],
    policy: EquityPolicy,
) -> None:
    result = plan(book, [], marks, policy)
    buys = [i for i in result.intents if i.side is EquitySide.BUY]
    assert [i.symbol for i in buys] == [AAA, BBB, CCC]
    assert buys == sorted(buys, key=lambda i: -i.notional)


def test_the_plan_is_deterministic(
    book: TargetBook,
    marks: dict[Ticker, Mark],
    policy: EquityPolicy,
) -> None:
    """Test plan item 3. Same inputs, same sequence — every time.

    Built from two differently-ordered holding lists so that a dict or set walked
    in insertion order would produce two different plans.
    """
    forward = [
        Holding(AAA, Decimal(100), Decimal(90), Decimal(10_000)),
        Holding(BBB, Decimal(100), Decimal(45), Decimal(5_000)),
        Holding(CCC, Decimal(100), Decimal(20), Decimal(2_500)),
    ]
    first = plan(book, forward, marks, policy)
    second = plan(book, list(reversed(forward)), marks, policy)
    assert [i.label() for i in first.intents] == [i.label() for i in second.intents]
    assert [(s.symbol, s.reason) for s in first.skipped] == [
        (s.symbol, s.reason) for s in second.skipped
    ]


# --------------------------------------------------------------------- #
# Refusing to guess
# --------------------------------------------------------------------- #


def test_an_unpriced_symbol_is_held_not_traded(
    book: TargetBook,
    holdings: list[Holding],
    marks: dict[Ticker, Mark],
    policy: EquityPolicy,
) -> None:
    """We do not know what it is worth, so we do not trade it."""
    del marks[AAA]
    result = plan(book, holdings, marks, policy)
    assert skip_for(result, AAA).reason is SkipReason.NO_MARK


def test_a_stale_mark_is_not_traded_on(
    book: TargetBook,
    marks: dict[Ticker, Mark],
    policy: EquityPolicy,
) -> None:
    stale = {**marks, AAA: replace(marks[AAA], age_seconds=policy.max_quote_age + 1)}
    result = plan(book, [], stale, policy)
    assert skip_for(result, AAA).reason is SkipReason.STALE_MARK


def test_a_mark_exactly_at_the_age_limit_is_still_fresh(
    book: TargetBook,
    marks: dict[Ticker, Mark],
    policy: EquityPolicy,
) -> None:
    """Inclusive on the safe side, same convention as the options Gate."""
    edge = {**marks, AAA: replace(marks[AAA], age_seconds=policy.max_quote_age)}
    result = plan(book, [], edge, policy)
    assert any(i.symbol == AAA for i in result.intents)


def test_an_untradeable_symbol_is_skipped(
    book: TargetBook,
    marks: dict[Ticker, Mark],
    policy: EquityPolicy,
) -> None:
    halted = {**marks, AAA: replace(marks[AAA], tradeable=False)}
    result = plan(book, [], halted, policy)
    assert skip_for(result, AAA).reason is SkipReason.NOT_TRADEABLE


def test_a_naive_timestamp_is_refused(
    book: TargetBook,
    holdings: list[Holding],
    marks: dict[Ticker, Mark],
    policy: EquityPolicy,
) -> None:
    """Nothing in this layer reads a clock, so a naive `as_of` is a caller that
    lost its timezone rather than a caller that meant UTC."""
    with pytest.raises(InvariantViolation, match="tz-aware"):
        plan(book, holdings, marks, policy, as_of=datetime(2026, 8, 28, 13, 45))  # noqa: DTZ001


def test_a_float_equity_is_refused(
    book: TargetBook,
    holdings: list[Holding],
    marks: dict[Ticker, Mark],
    policy: EquityPolicy,
) -> None:
    with pytest.raises(InvariantViolation, match="must be Decimal"):
        plan(book, holdings, marks, policy, equity=100_000.0)


def test_turnover_is_the_sum_of_both_sides(
    book: TargetBook,
    marks: dict[Ticker, Mark],
    policy: EquityPolicy,
) -> None:
    held = [Holding(AAA, Decimal(500), Decimal(90), Decimal(50_000))]
    result = plan(book, held, marks, policy)
    assert result.turnover == result.buy_notional + result.sell_notional
    assert result.sell_notional > 0
    assert result.buy_notional > 0


def _ddd() -> Ticker:
    from alphagate.core.identifiers import ticker

    return ticker("DDD")


# --------------------------------------------------------------------- #
# Blindness — an empty plan that decided nothing
# --------------------------------------------------------------------- #


def test_a_plan_that_priced_nothing_is_blind(
    book: TargetBook,
    holdings: list[Holding],
    marks: dict[Ticker, Mark],
    policy: EquityPolicy,
) -> None:
    """Every symbol stale is not "the book is already held" — it is "no idea".

    The pre-open case, and the one that cost a session: the free feed does not
    tick before 13:30, so a pass at 13:26 skips all of them and looks, from the
    stage alone, exactly like a quiet day.
    """
    blind = {
        symbol: replace(mark, age_seconds=policy.max_quote_age + 1)
        for symbol, mark in marks.items()
    }
    result = plan(book, holdings, blind, policy)
    assert result.is_empty
    assert result.is_blind


def test_a_plan_with_nothing_to_do_is_not_blind(
    book: TargetBook,
    holdings: list[Holding],
    marks: dict[Ticker, Mark],
    policy: EquityPolicy,
) -> None:
    """The common case. Empty, but it saw every price it needed to."""
    result = plan(book, holdings, marks, policy)
    assert result.is_empty
    assert not result.is_blind


def test_one_readable_price_is_enough_to_not_be_blind(
    book: TargetBook,
    holdings: list[Holding],
    marks: dict[Ticker, Mark],
    policy: EquityPolicy,
) -> None:
    """Blindness is total, not proportional.

    A partly stale book is an ordinary session — names come and go stale all
    day. Only a pass that could read *nothing* has failed to decide.
    """
    partly = {
        **{
            symbol: replace(mark, age_seconds=policy.max_quote_age + 1)
            for symbol, mark in marks.items()
        },
        AAA: marks[AAA],
    }
    result = plan(book, holdings, partly, policy)
    assert result.is_empty
    assert not result.is_blind


def test_a_missing_mark_counts_as_blindness_too(
    book: TargetBook,
    holdings: list[Holding],
    policy: EquityPolicy,
) -> None:
    """`NO_MARK` and `STALE_MARK` are the same failure seen twice."""
    result = plan(book, holdings, {}, policy)
    assert result.is_empty
    assert result.is_blind


def test_a_plan_with_orders_is_never_blind(
    book: TargetBook,
    marks: dict[Ticker, Mark],
    policy: EquityPolicy,
) -> None:
    result = plan(book, [], marks, policy)
    assert not result.is_empty
    assert not result.is_blind


def test_an_untradeable_book_is_not_blind(
    book: TargetBook,
    holdings: list[Holding],
    marks: dict[Ticker, Mark],
    policy: EquityPolicy,
) -> None:
    """A halted symbol is a price we can read and a market we cannot reach.

    Retrying it in thirty seconds is not the remedy, so it must not reopen the
    day the way a stale quote does.
    """
    halted = {
        symbol: replace(mark, tradeable=False) for symbol, mark in marks.items()
    }
    result = plan(book, holdings, halted, policy)
    assert result.is_empty
    assert not result.is_blind
