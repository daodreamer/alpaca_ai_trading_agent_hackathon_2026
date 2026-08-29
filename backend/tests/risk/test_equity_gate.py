"""The equity Risk Gate — specs/09 D5, test plan items 7–9.

The baseline is an intent that passes every check with room to spare. Each test
then breaks exactly one thing, because a test that fails for two reasons cannot
tell you which check it is exercising, and this suite exists to pin twelve
checks individually.

`conftest.py` for the equity fixtures lives in `tests/equity/`, and this module
imports from it rather than duplicating a second book payload. Two fixtures for
one artefact would drift, and the drift would be between the thing the planner
is tested against and the thing the Gate is tested against — which is the pair
that most needs to agree.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
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
    TargetBook,
)
from alphagate.risk import (
    EQUITY_CHECKS,
    UNWAIVABLE,
    ApprovedEquity,
    EquityPortfolio,
    EquityVerdict,
    GatedEquityOrder,
    evaluate_equity,
)
from tests.equity.conftest import AAA, BBB, FINGERPRINT, NOW, portfolio_for


def buy_intent(**overrides: object) -> OrderIntent:
    """A $2,000 buy of AAA — 20 shares at $100, well inside every budget.

    The baseline holds 100 shares, so the result is 120 shares worth $12,000
    against a $15,000 concentration cap: comfortably inside, and close enough
    that `test_position_cap_vetoes_a_concentrated_result` does not have to invent
    an absurd order to cross it.
    """
    defaults = {
        "symbol": AAA,
        "side": EquitySide.BUY,
        "shares": Decimal(20),
        "reference_price": Decimal(100),
        "target_weight": Decimal("0.12"),
        "held_weight": Decimal("0.10"),
        "held_shares": Decimal(100),
        "fractionable": True,
        "mark_age_seconds": 1.0,
    }
    return OrderIntent(**{**defaults, **overrides})


def sell_intent(**overrides: object) -> OrderIntent:
    return buy_intent(
        side=EquitySide.SELL, target_weight=Decimal("0.08"), **overrides
    )


def judge(
    intent: OrderIntent,
    book: TargetBook,
    portfolio: EquityPortfolio,
    policy: EquityPolicy,
    *,
    as_of: datetime = NOW,
    pinned: str = FINGERPRINT,
) -> EquityVerdict:
    return evaluate_equity(
        intent, book, portfolio, policy, as_of, pinned_fingerprint=pinned
    )


def veto_names(verdict: EquityVerdict) -> set[str]:
    return {reason.check for reason in verdict.reasons}


# --------------------------------------------------------------------- #
# The shape of the answer
# --------------------------------------------------------------------- #


def test_a_clean_intent_is_approved_with_the_whole_tape(
    book: TargetBook,
    portfolio: EquityPortfolio,
    policy: EquityPolicy,
) -> None:
    verdict = judge(buy_intent(), book, portfolio, policy)
    assert verdict.is_approved
    assert isinstance(verdict, ApprovedEquity)
    assert len(verdict.checks) == len(EQUITY_CHECKS)
    assert all(check.passed for check in verdict.checks)
    assert verdict.order.symbol == AAA
    assert verdict.order.shares == Decimal(20)
    assert verdict.order.fingerprint == FINGERPRINT


def test_every_check_runs_even_after_one_fails(
    book: TargetBook,
    holdings: list[Holding],
    marks: dict[Ticker,
    Mark],
    policy: EquityPolicy,
) -> None:
    """No short-circuit. A refusal with one reason and a refusal with five are
    different situations, and the journal should be able to show which."""
    broken = portfolio_for(
        holdings, marks, drawdown_pct=Decimal("0.5"), buying_power=Decimal(1)
    )
    verdict = judge(buy_intent(), book, broken, policy)
    assert not verdict.is_approved
    assert len(verdict.checks) == len(EQUITY_CHECKS)
    assert {"drawdown_killswitch", "buying_power"} <= veto_names(verdict)


def test_the_verdict_is_deterministic(
    book: TargetBook,
    portfolio: EquityPortfolio,
    policy: EquityPolicy,
) -> None:
    """Including the order of `checks` and `reasons` — both come from a tuple."""
    first = judge(buy_intent(), book, portfolio, policy)
    second = judge(buy_intent(), book, portfolio, policy)
    assert [c.name for c in first.checks] == [c.name for c in second.checks]
    assert [c.name for c in first.checks] == [c.__name__ for c in EQUITY_CHECKS]


def test_a_naive_timestamp_is_refused(
    book: TargetBook,
    portfolio: EquityPortfolio,
    policy: EquityPolicy,
) -> None:
    naive = datetime(2026, 8, 28, 13)  # noqa: DTZ001 — the whole point of the test
    with pytest.raises(InvariantViolation, match="tz-aware"):
        judge(buy_intent(), book, portfolio, policy, as_of=naive)


# --------------------------------------------------------------------- #
# One test per check — test plan item 7
# --------------------------------------------------------------------- #


def test_book_is_pinned_vetoes_a_plan_for_another_strategy(
    book: TargetBook,
    portfolio: EquityPortfolio,
    policy: EquityPolicy,
) -> None:
    """The book was checked at load. This is the assertion that the plan reaching
    the Gate came from *that* book — two different objects, and a wiring mistake
    between them is invisible everywhere else."""
    verdict = judge(buy_intent(), book, portfolio, policy, pinned="0" * 16)
    assert "book_is_pinned" in veto_names(verdict)


def test_book_is_fresh_vetoes_a_stale_book(
    book: TargetBook,
    portfolio: EquityPortfolio,
    policy: EquityPolicy,
) -> None:
    late = NOW + timedelta(days=policy.max_book_age_days + 2)
    verdict = judge(buy_intent(), book, portfolio, policy, as_of=late)
    assert "book_is_fresh" in veto_names(verdict)


def test_a_book_exactly_at_the_age_limit_passes(
    book: TargetBook,
    portfolio: EquityPortfolio,
    policy: EquityPolicy,
) -> None:
    """Inclusive on the safe side."""
    midnight = datetime.min.time()
    edge = datetime.combine(book.as_of, midnight, tzinfo=UTC) + timedelta(
        days=policy.max_book_age_days
    )
    verdict = judge(buy_intent(), book, portfolio, policy, as_of=edge)
    check = next(c for c in verdict.checks if c.name == "book_is_fresh")
    assert check.passed


def test_price_is_fresh_vetoes_a_stale_mark(
    book: TargetBook,
    portfolio: EquityPortfolio,
    policy: EquityPolicy,
) -> None:
    stale = buy_intent(mark_age_seconds=policy.max_quote_age + 0.1)
    assert "price_is_fresh" in veto_names(judge(stale, book, portfolio, policy))


def test_a_mark_exactly_at_the_age_limit_passes(
    book: TargetBook,
    portfolio: EquityPortfolio,
    policy: EquityPolicy,
) -> None:
    edge = buy_intent(mark_age_seconds=policy.max_quote_age)
    assert judge(edge, book, portfolio, policy).is_approved


def test_no_short_selling_vetoes_a_sell_past_the_holding(
    book: TargetBook,
    portfolio: EquityPortfolio,
    policy: EquityPolicy,
) -> None:
    """Test plan item 8's second half, and the third of three layers.

    The intent type refuses this outright, so the Gate cannot be handed one
    through the normal path — which is exactly why the check has to be tested by
    reaching around it. A check that can only be exercised by the bug it exists
    to catch is still worth having.
    """
    intent = sell_intent(shares=Decimal(10), held_shares=Decimal(10))
    short = object.__new__(OrderIntent)
    for name, value in (
        ("symbol", AAA), ("side", EquitySide.SELL), ("shares", Decimal(50)),
        ("reference_price", Decimal(100)), ("target_weight", Decimal(0)),
        ("held_weight", Decimal(0)), ("held_shares", Decimal(10)),
        ("fractionable", True), ("mark_age_seconds", 1.0),
    ):
        object.__setattr__(short, name, value)
    verdict = judge(short, book, portfolio, policy)
    assert "no_short_selling" in veto_names(verdict)
    assert judge(intent, book, portfolio, policy).is_approved


def test_order_is_material_vetoes_a_trivial_order(
    book: TargetBook,
    portfolio: EquityPortfolio,
    policy: EquityPolicy,
) -> None:
    tiny = buy_intent(shares=Decimal("0.1"))  # $10 against a $25 floor
    assert "order_is_material" in veto_names(judge(tiny, book, portfolio, policy))


def test_an_order_exactly_at_the_floor_passes(
    book: TargetBook,
    portfolio: EquityPortfolio,
    policy: EquityPolicy,
) -> None:
    edge = buy_intent(shares=policy.min_order_notional / Decimal(100))
    check = next(
        c for c in judge(edge, book, portfolio, policy).checks
        if c.name == "order_is_material"
    )
    assert check.passed


def test_position_cap_vetoes_a_concentrated_result(
    book: TargetBook,
    portfolio: EquityPortfolio,
    policy: EquityPolicy,
) -> None:
    """AAA is already 100 shares; buying 100 more takes it to $20,000 — past the
    $15,000 that 15% of a $100,000 account allows in one name."""
    heavy = buy_intent(shares=Decimal(100))
    assert "position_cap" in veto_names(judge(heavy, book, portfolio, policy))


def test_gross_exposure_cap_vetoes_leverage(
    book: TargetBook,
    marks: dict[Ticker,
    Mark],
    policy: EquityPolicy,
) -> None:
    """A fully invested book cannot buy more without borrowing.

    Built explicitly rather than from the baseline holdings, which are 20% of the
    account on purpose so that the buying-power tests have room to move. Leverage
    only becomes expressible once there is no cash left.
    """
    invested = [
        Holding(AAA, Decimal(100), Decimal(90), Decimal(10_000)),
        Holding(BBB, Decimal(1_800), Decimal(45), Decimal(90_000)),
    ]
    full = portfolio_for(invested, marks, buying_power=Decimal(1_000_000))
    assert full.gross_exposure == Decimal(1)
    verdict = judge(buy_intent(), book, full, policy)
    assert "gross_exposure_cap" in veto_names(verdict)


def test_buying_power_vetoes_an_unpayable_buy(
    book: TargetBook,
    holdings: list[Holding],
    marks: dict[Ticker,
    Mark],
    policy: EquityPolicy,
) -> None:
    poor = portfolio_for(holdings, marks, buying_power=Decimal(100))
    assert "buying_power" in veto_names(judge(buy_intent(), book, poor, policy))


def test_a_sell_never_needs_buying_power(
    book: TargetBook,
    holdings: list[Holding],
    marks: dict[Ticker,
    Mark],
    policy: EquityPolicy,
) -> None:
    poor = portfolio_for(holdings, marks, buying_power=Decimal(0))
    check = next(
        c for c in judge(sell_intent(), book, poor, policy).checks
        if c.name == "buying_power"
    )
    assert check.passed


def test_daily_turnover_cap_vetoes_a_rebuilt_book(
    book: TargetBook,
    holdings: list[Holding],
    marks: dict[Ticker,
    Mark],
    policy: EquityPolicy,
) -> None:
    spent = portfolio_for(holdings, marks, turnover_today=Decimal(119_500))
    assert "daily_turnover_cap" in veto_names(judge(buy_intent(), book, spent, policy))


def test_daily_order_cap_is_a_circuit_breaker(
    book: TargetBook,
    holdings: list[Holding],
    marks: dict[Ticker,
    Mark],
    policy: EquityPolicy,
) -> None:
    busy = portfolio_for(holdings, marks, orders_today=policy.max_daily_orders)
    assert "daily_order_cap" in veto_names(judge(buy_intent(), book, busy, policy))


def test_drawdown_killswitch_vetoes_past_the_limit(
    book: TargetBook,
    holdings: list[Holding],
    marks: dict[Ticker,
    Mark],
    policy: EquityPolicy,
) -> None:
    deep = portfolio_for(holdings, marks, drawdown_pct=policy.max_drawdown_pct + Decimal("0.01"))
    assert "drawdown_killswitch" in veto_names(judge(buy_intent(), book, deep, policy))


def test_the_latch_vetoes_even_when_the_drawdown_has_recovered(
    book: TargetBook,
    holdings: list[Holding],
    marks: dict[Ticker,
    Mark],
    policy: EquityPolicy,
) -> None:
    """A pure function cannot remember yesterday, so the latch is carried in.

    Re-arming is a deliberate human act, not a restart.
    """
    latched = portfolio_for(holdings, marks, drawdown_pct=Decimal(0), killswitch_tripped=True)
    verdict = judge(buy_intent(), book, latched, policy)
    assert "drawdown_killswitch" in veto_names(verdict)
    assert "latched" in next(
        r.detail for r in verdict.reasons if r.check == "drawdown_killswitch"
    )


def test_a_drawdown_exactly_at_the_limit_passes(
    book: TargetBook,
    holdings: list[Holding],
    marks: dict[Ticker,
    Mark],
    policy: EquityPolicy,
) -> None:
    edge = portfolio_for(holdings, marks, drawdown_pct=policy.max_drawdown_pct)
    check = next(
        c for c in judge(buy_intent(), book, edge, policy).checks
        if c.name == "drawdown_killswitch"
    )
    assert check.passed


# --------------------------------------------------------------------- #
# The waiver — specs/09 D5, test plan item 8
# --------------------------------------------------------------------- #


def test_a_sell_is_waived_past_a_budget_veto(
    book: TargetBook,
    holdings: list[Holding],
    marks: dict[Ticker,
    Mark],
    policy: EquityPolicy,
) -> None:
    """Once the kill switch has tripped, the only orders that should still be
    possible are the ones that make the book smaller."""
    latched = portfolio_for(holdings, marks, killswitch_tripped=True)
    verdict = judge(sell_intent(), book, latched, policy)
    assert verdict.is_approved
    assert "drawdown_killswitch" in {r.check for r in verdict.waived}


def test_a_waived_reason_is_recorded_rather_than_dropped(
    book: TargetBook,
    holdings: list[Holding],
    marks: dict[Ticker,
    Mark],
    policy: EquityPolicy,
) -> None:
    """"This sell was allowed past the turnover cap" is a fact somebody reading
    the journal after a bad week needs to be able to find."""
    spent = portfolio_for(holdings, marks, turnover_today=Decimal(119_999))
    verdict = judge(sell_intent(), book, spent, policy)
    assert verdict.is_approved
    assert {r.check for r in verdict.waived} == {"daily_turnover_cap"}


def test_a_sell_is_not_waived_past_a_stale_price(
    book: TargetBook,
    portfolio: EquityPortfolio,
    policy: EquityPolicy,
) -> None:
    """A sell of a name we cannot price is not an exit, it is a rejection that
    leaves the position on."""
    stale = sell_intent(mark_age_seconds=policy.max_quote_age + 1)
    assert not judge(stale, book, portfolio, policy).is_approved


def test_a_sell_is_not_waived_past_the_wrong_book(
    book: TargetBook,
    portfolio: EquityPortfolio,
    policy: EquityPolicy,
) -> None:
    verdict = judge(sell_intent(), book, portfolio, policy, pinned="0" * 16)
    assert not verdict.is_approved


def test_the_unwaivable_set_is_exactly_the_documented_five() -> None:
    """A check moved out of this set silently weakens the exit waiver into a
    general bypass, which is the failure this constant exists to prevent."""
    assert {
        "book_is_pinned",
        "book_is_fresh",
        "symbol_is_tradeable",
        "price_is_fresh",
        "no_short_selling",
    } == UNWAIVABLE


def test_a_buy_is_never_waived(
    book: TargetBook,
    holdings: list[Holding],
    marks: dict[Ticker,
    Mark],
    policy: EquityPolicy,
) -> None:
    latched = portfolio_for(holdings, marks, killswitch_tripped=True)
    assert not judge(buy_intent(), book, latched, policy).is_approved


# --------------------------------------------------------------------- #
# The key — specs/09 D7, test plan item 9
# --------------------------------------------------------------------- #


def test_a_gated_equity_order_cannot_be_forged() -> None:
    """The runtime half of guard 5. `tests/test_boundaries.py` is the static half."""
    with pytest.raises(InvariantViolation, match="may only be constructed"):
        GatedEquityOrder(
            symbol=AAA,
            side=EquitySide.BUY,
            shares=Decimal(1),
            reference_price=Decimal(100),
            fractionable=True,
            fingerprint=FINGERPRINT,
            book_as_of="2026-08-27",
            approved_at=NOW,
        )


def test_an_approval_cannot_be_forged_either(
    book: TargetBook,
    portfolio: EquityPortfolio,
    policy: EquityPolicy,
) -> None:
    """Otherwise a forged `ApprovedEquity` around a real order is a second door."""
    real = judge(buy_intent(), book, portfolio, policy).order
    with pytest.raises(InvariantViolation, match="may only be constructed"):
        ApprovedEquity(order=real, checks=())


def test_a_gated_order_cannot_be_copied_into_a_second_order(
    book: TargetBook,
    portfolio: EquityPortfolio,
    policy: EquityPolicy,
) -> None:
    """An order that can be cloned is an order that can be submitted twice."""
    order = judge(buy_intent(), book, portfolio, policy).order
    with pytest.raises(InvariantViolation):
        replace(order, shares=Decimal(999))


def test_a_veto_carries_no_order(
    book: TargetBook,
    holdings: list[Holding],
    marks: dict[Ticker,
    Mark],
    policy: EquityPolicy,
) -> None:
    poor = portfolio_for(holdings, marks, buying_power=Decimal(0))
    verdict = judge(buy_intent(), book, poor, policy)
    assert not hasattr(verdict, "order")


# --------------------------------------------------------------------- #
# The portfolio snapshot
# --------------------------------------------------------------------- #


def test_gross_exposure_is_measured_at_the_gates_own_marks(
    holdings: list[Holding],
    marks: dict[Ticker,
    Mark],
) -> None:
    """Not at the broker's `market_value`, which may be minutes old.

    A check about a *resulting* position must be arithmetic on the same price the
    order was sized from, or it compares two different valuations.
    """
    portfolio = portfolio_for(holdings, marks)
    assert portfolio.gross_notional == Decimal(20_000)
    assert portfolio.gross_exposure == Decimal("0.2")


def test_an_unmarked_position_falls_back_to_market_value_not_zero(
    marks: dict[Ticker,
    Mark],
) -> None:
    """A position valued at nothing would pass a concentration check it should fail."""
    held = [Holding(BBB, Decimal(100), Decimal(45), Decimal(4_500))]
    portfolio = portfolio_for(held, {})
    assert portfolio.notional_of(BBB) == Decimal(4_500)


def test_a_float_equity_snapshot_is_refused(
    holdings: list[Holding],
    marks: dict[Ticker,
    Mark],
) -> None:
    with pytest.raises(InvariantViolation, match="must be Decimal"):
        portfolio_for(holdings, marks, equity=100_000.0)
