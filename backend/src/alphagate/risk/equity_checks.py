"""The equity checks — specs/09 D5.

PURE. Stdlib plus `alphagate.core`, `alphagate.equity` and the sibling risk
modules. No LLM, no I/O, no clock read, no network.

Each check is a pure predicate over `(intent, book, portfolio, policy, as_of)`,
wrapped in a `Context` so the shared arithmetic is computed once and every check
sees the same numbers. They are registered in `EQUITY_CHECKS`, a tuple, and that
tuple *is* the declared order — no set, no dict, no sorting at call time.

The three rules carried over from [03](../../../../specs/03-risk-gate.md)
unchanged, because they were right there and are right here:

**Every check runs.** No short-circuit, no early return. A refusal with one
reason and a refusal with five are different situations and the journal should
be able to show which.

**Boundaries are inclusive on the safe side.** A value exactly at its limit
passes; a value past it vetoes.

**A risk-reducing order is not blocked by a budget.** The checks are still
computed — the dashboard wants the numbers — but a failure on a sell is waived
rather than turned into a veto. Two checks are exempt from the waiver and say so
in `WAIVABLE`: one would open a short, the other cannot be filled, and neither
becomes acceptable because the order happens to be a sell.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Final

from alphagate.equity.book import TargetBook
from alphagate.equity.plan import EquitySide, OrderIntent
from alphagate.equity.policy import EquityPolicy
from alphagate.risk.equity_portfolio import EquityPortfolio
from alphagate.risk.verdict import CheckResult

__all__ = [
    "EQUITY_CHECKS",
    "UNWAIVABLE",
    "EquityCheck",
    "EquityContext",
    "run_equity_checks",
]


@dataclass(frozen=True, slots=True)
class EquityContext:
    """The five arguments, plus the arithmetic every check would otherwise redo."""

    intent: OrderIntent
    book: TargetBook
    portfolio: EquityPortfolio
    policy: EquityPolicy
    as_of: datetime
    pinned_fingerprint: str

    @property
    def notional(self) -> Decimal:
        return self.intent.notional

    @property
    def reduces_risk(self) -> bool:
        return self.intent.side.reduces_risk

    @property
    def resulting_notional(self) -> Decimal:
        """What this symbol will be worth once the order fills, at the same price
        the order was sized from."""
        return self.intent.resulting_notional

    @property
    def resulting_gross(self) -> Decimal:
        """Book gross after the fill, as a fraction of equity.

        Computed as a delta on the current gross rather than by re-summing the
        book, because the current gross is measured at the portfolio's marks and
        re-summing with one symbol swapped for a differently-priced version of
        itself would compare two different valuations.
        """
        delta = self.notional if self.intent.side is EquitySide.BUY else -self.notional
        return (self.portfolio.gross_notional + delta) / self.portfolio.equity

    @property
    def book_age_days(self) -> int:
        return self.book.age_days(self.as_of.date())


type EquityCheck = Callable[[EquityContext], CheckResult]


# --------------------------------------------------------------------- #
# Provenance — whether this order is serving the strategy we said it was.
# No configuration can switch these off.
# --------------------------------------------------------------------- #


def book_is_pinned(ctx: EquityContext) -> CheckResult:
    """The order must serve the pinned strategy, not merely *a* strategy.

    The book was already checked at load (specs/09 D1). This is the assertion
    that the plan reaching the Gate came from that book and not another — the
    two are different objects and a wiring mistake between them would be
    invisible everywhere else.
    """
    actual = ctx.book.fingerprint
    ok = bool(actual) and actual == ctx.pinned_fingerprint
    return CheckResult(
        name="book_is_pinned",
        passed=ok,
        detail=(
            f"serving {actual}"
            if ok
            else f"plan is for {actual!r}, pinned strategy is {ctx.pinned_fingerprint!r}"
        ),
        observed=actual,
        limit=ctx.pinned_fingerprint,
    )


def book_is_fresh(ctx: EquityContext) -> CheckResult:
    """A book older than the rebalance period describes a portfolio the strategy
    no longer holds.

    Measured against the Gate's `as_of` argument, never a clock read.
    """
    age = ctx.book_age_days
    limit = ctx.policy.max_book_age_days
    ok = age <= limit
    return CheckResult(
        name="book_is_fresh",
        passed=ok,
        detail=(
            f"book is {age}d old (as of {ctx.book.as_of.isoformat()}), limit {limit}d"
        ),
        observed=age,
        limit=limit,
    )


def symbol_is_tradeable(ctx: EquityContext) -> CheckResult:
    """An order in a name the broker will not trade is a rejection with extra steps.

    Not waivable for a sell. A halted name cannot be sold either, and pretending
    otherwise would produce a plan that believes it has exited a position it
    still holds.
    """
    # The planner refuses to build an intent for an untradeable mark, so in a
    # healthy system this cannot fail. It stays as the assertion that the
    # upstream guarantee actually held, and it costs one comparison.
    ok = ctx.intent.reference_price > 0
    return CheckResult(
        name="symbol_is_tradeable",
        passed=ok,
        detail=(
            f"{ctx.intent.symbol} priced at {ctx.intent.reference_price}"
            if ok
            else f"{ctx.intent.symbol} has no usable price"
        ),
        observed=ctx.intent.reference_price,
    )


def price_is_fresh(ctx: EquityContext) -> CheckResult:
    """Stale marks make every other number in this file a guess.

    Inclusive: a quote exactly at the limit is still fresh. The age travels on
    the intent from the mark it was sized against, so this is a comparison
    between two arguments rather than a clock read.
    """
    age = ctx.intent.mark_age_seconds
    limit = ctx.policy.max_quote_age
    ok = age <= limit
    return CheckResult(
        name="price_is_fresh",
        passed=ok,
        detail=f"mark is {age:.1f}s old, limit {limit:.1f}s",
        observed=age,
        limit=limit,
    )


def no_short_selling(ctx: EquityContext) -> CheckResult:
    """A sell may not exceed what is held. specs/09 D6.

    The third of three layers: the loader refuses a negative weight, the planner
    clamps the quantity, and this refuses the result anyway. Never waived — the
    waiver exists for orders that reduce risk, and a short does the opposite
    however it is labelled.
    """
    if ctx.intent.side is EquitySide.BUY:
        return CheckResult(
            name="no_short_selling",
            passed=True,
            detail="a buy cannot open a short",
            observed=Decimal(0),
        )
    resulting = ctx.intent.resulting_shares
    ok = resulting >= 0
    return CheckResult(
        name="no_short_selling",
        passed=ok,
        detail=(
            f"leaves {resulting} shares"
            if ok
            else f"would leave {resulting} shares — a short position"
        ),
        observed=resulting,
        limit=Decimal(0),
    )


# --------------------------------------------------------------------- #
# Budgeted — fractions of equity, waivable for a risk-reducing order.
# --------------------------------------------------------------------- #


def order_is_material(ctx: EquityContext) -> CheckResult:
    """An order below the floor is all spread. specs/09 D3, asserted again here.

    The planner already applied the band, so this is the assertion that it did.
    A duplicated cheap check at a boundary is worth more than a comment saying
    the boundary is safe.
    """
    floor = ctx.policy.min_order_notional
    ok = ctx.notional >= floor
    return CheckResult(
        name="order_is_material",
        passed=ok,
        detail=f"order notional {_money(ctx.notional)}, floor {_money(floor)}",
        observed=ctx.notional,
        limit=floor,
    )


def position_cap(ctx: EquityContext) -> CheckResult:
    """No single name may end up above its share of the account.

    The book's largest weight is 8%, so a failure here means the book is
    malformed or the equity read is wrong — which is exactly what a cap should
    catch and a strategy parameter should not.
    """
    resulting = ctx.resulting_notional
    limit = ctx.policy.max_position(ctx.portfolio.equity)
    ok = resulting <= limit
    return CheckResult(
        name="position_cap",
        passed=ok,
        detail=(
            f"{ctx.intent.symbol} would be {_money(resulting)}, cap {_money(limit)} "
            f"({ctx.policy.max_position_pct:%} of equity)"
        ),
        observed=resulting,
        limit=limit,
    )


def gross_exposure_cap(ctx: EquityContext) -> CheckResult:
    """Fully invested is the most this may be. Anything above is leverage."""
    resulting = ctx.resulting_gross
    limit = ctx.policy.max_gross
    ok = resulting <= limit
    return CheckResult(
        name="gross_exposure_cap",
        passed=ok,
        detail=f"gross would be {resulting:.4f}x equity, cap {limit}x",
        observed=resulting,
        limit=limit,
    )


def buying_power(ctx: EquityContext) -> CheckResult:
    """A buy must be payable now, not after the sells that follow it.

    Deliberately pessimistic. The plan places sells first precisely so the cash
    is there (specs/09 D4), but the Gate judges one order against the account as
    it stands — a Gate that budgeted against orders it assumed would fill is a
    Gate reasoning about a book that does not exist yet.
    """
    if ctx.intent.side is EquitySide.SELL:
        return CheckResult(
            name="buying_power",
            passed=True,
            detail="a sell releases buying power",
            observed=ctx.portfolio.buying_power,
        )
    available = ctx.portfolio.buying_power
    ok = ctx.notional <= available
    return CheckResult(
        name="buying_power",
        passed=ok,
        detail=f"needs {_money(ctx.notional)}, {_money(available)} available",
        observed=ctx.notional,
        limit=available,
    )


def daily_turnover_cap(ctx: EquityContext) -> CheckResult:
    """One session may turn the book over, not repeatedly rebuild it."""
    resulting = ctx.portfolio.turnover_today + ctx.notional
    limit = ctx.policy.max_daily_turnover(ctx.portfolio.equity)
    ok = resulting <= limit
    return CheckResult(
        name="daily_turnover_cap",
        passed=ok,
        detail=f"today's turnover would be {_money(resulting)}, cap {_money(limit)}",
        observed=resulting,
        limit=limit,
    )


def daily_order_cap(ctx: EquityContext) -> CheckResult:
    """A circuit breaker on a runaway plan, not a limit on a rebalance.

    A full build of this book is 104 orders, so the cap is well above a normal
    session and well below a loop.
    """
    resulting = ctx.portfolio.orders_today + 1
    limit = ctx.policy.max_daily_orders
    ok = resulting <= limit
    return CheckResult(
        name="daily_order_cap",
        passed=ok,
        detail=f"this would be order {resulting} of {limit} today",
        observed=resulting,
        limit=limit,
    )


def drawdown_killswitch(ctx: EquityContext) -> CheckResult:
    """Stop opening risk once the account is far enough below its high-water mark.

    Vetoes on either the live drawdown crossing the limit **or** the latch
    already being set. A pure function cannot remember yesterday, so the latch
    is state the caller carries in — and re-arming is a deliberate human act,
    not a restart ([03](../../../../specs/03-risk-gate.md) D4).

    Waivable, and that matters more here than anywhere else in the file: once
    the switch has tripped the only orders that should still be possible are the
    ones that make the book smaller.
    """
    drawdown = ctx.portfolio.drawdown_pct
    limit = ctx.policy.max_drawdown_pct
    latched = ctx.portfolio.killswitch_tripped
    ok = not latched and drawdown <= limit
    return CheckResult(
        name="drawdown_killswitch",
        passed=ok,
        detail=(
            f"drawdown {drawdown:.4f} against a {limit} limit"
            + (" — latched; only a human re-arms it" if latched else "")
        ),
        observed=drawdown,
        limit=limit,
    )


EQUITY_CHECKS: Final[tuple[EquityCheck, ...]] = (
    book_is_pinned,
    book_is_fresh,
    symbol_is_tradeable,
    price_is_fresh,
    no_short_selling,
    order_is_material,
    position_cap,
    gross_exposure_cap,
    buying_power,
    daily_turnover_cap,
    daily_order_cap,
    drawdown_killswitch,
)
"""The declared order. A tuple, and this tuple *is* the order specs/09 D5
promises — the verdict's `checks` and `reasons` both come out in it."""

UNWAIVABLE: Final = frozenset(
    {
        "book_is_pinned",
        "book_is_fresh",
        "symbol_is_tradeable",
        "price_is_fresh",
        "no_short_selling",
    }
)
"""Checks a risk-reducing order does not get waived past.

`no_short_selling` because the waiver is for orders that shrink the book and a
short grows it. `symbol_is_tradeable` and `price_is_fresh` because a sell of an
unpriceable name is not an exit, it is a rejection that leaves the position on.
The two provenance checks because an order that is not serving the pinned,
current book has no business being placed in either direction."""


def run_equity_checks(ctx: EquityContext) -> tuple[CheckResult, ...]:
    """Run every check, in `EQUITY_CHECKS` order. Total and deterministic."""
    return tuple(check(ctx) for check in EQUITY_CHECKS)


def waive_for_reduction(result: CheckResult, *, reduces_risk: bool) -> bool:
    """Whether a failing check should be waived rather than vetoed.

    A separate function rather than a condition inside the Gate, so the rule is
    nameable, testable and visible in one place.
    """
    return reduces_risk and result.name not in UNWAIVABLE


def _money(value: Decimal) -> str:
    return f"${value.quantize(Decimal('0.01'))}"
