"""The checks — specs/03 D4.

PURE. Stdlib plus `alphagate.core`, `alphagate.options` and the sibling risk
modules. No LLM, no I/O, no clock read, no network.

Each check is a pure predicate over `(proposal, portfolio, limits, as_of)`,
wrapped up as a `Context` so the shared arithmetic is computed once and every
check sees the same numbers. They are registered in `CHECKS`, a tuple, and that
tuple *is* the declared order from specs/03 D6 — no set, no dict, no sorting at
call time.

Three design points that are easy to get wrong:

**Every check runs.** There is no short-circuit and no early return from the
sequence. The Gate returns the whole tape.

**Boundaries are inclusive on the safe side.** A value exactly at its limit
passes; a value past it vetoes. Stated once here and asserted once per check,
because "> or >=" is the kind of thing that gets flipped during a refactor and
noticed during a drawdown.

**The Gate never blocks an exit.** For `Intent.CLOSE` the checks are still
*computed* — the dashboard wants the numbers — but a failure is waived rather
than turned into a veto. See `waive_for_exit`. That is specs/03 D4's closing
sentence, and it outranks every budget in this file.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Final

from alphagate.risk.limits import RiskLimits
from alphagate.risk.portfolio import PortfolioSnapshot
from alphagate.risk.proposal import Intent, TradeProposal
from alphagate.risk.verdict import CheckResult

__all__ = ["CHECKS", "Check", "Context", "run_all", "waive_for_exit"]


@dataclass(frozen=True, slots=True)
class Context:
    """The four arguments, plus the arithmetic every check would otherwise redo."""

    proposal: TradeProposal
    portfolio: PortfolioSnapshot
    limits: RiskLimits
    as_of: datetime

    @property
    def trade_loss(self) -> Decimal:
        """Maximum loss of the whole proposed order."""
        return self.proposal.risk.max_loss * self.proposal.quantity

    @property
    def opens_risk(self) -> bool:
        return self.proposal.intent.opens_risk

    @property
    def quote_age(self) -> float:
        """Oldest leg quote age at *the Gate's* `as_of`, not the agent's.

        `StructureRisk.quote_age_seconds` was true when the risk was computed.
        The proposal then travelled through a model call, so the Gate ages it
        forward by the elapsed time. When perception and gating share one `as_of`
        — the normal case inside a single tick — the elapsed term is zero and
        this is exactly the number `compute_risk` produced.
        """
        elapsed = (self.as_of - self.proposal.risk_as_of).total_seconds()
        return self.proposal.risk.quote_age_seconds + elapsed


type Check = Callable[[Context], CheckResult]


# --------------------------------------------------------------------- #
# Structural — whether the trade is knowable at all. No configuration can
# switch these off.
# --------------------------------------------------------------------- #


def defined_risk(ctx: Context) -> CheckResult:
    """A structure whose loss is unbounded must never reach an exchange.

    specs/02 D3 already makes such a structure unconstructible, so in a healthy
    system this check cannot fail. It stays anyway: it is the assertion that the
    type-level guarantee actually held, and it costs one comparison.
    """
    max_loss = ctx.proposal.risk.max_loss
    ok = max_loss.is_finite() and max_loss > 0
    return CheckResult(
        name="defined_risk",
        passed=ok,
        detail=(
            f"maximum loss is {max_loss}"
            if ok
            else f"maximum loss {max_loss} is not finite and positive"
        ),
        observed=max_loss,
    )


def known_greeks(ctx: Context) -> CheckResult:
    """Refuse to open exposure nobody can state.

    Opens only. Closing a position whose greeks a provider failed to return is
    still the right thing to do — arguably more so.
    """
    greeks = ctx.proposal.risk.net_greeks
    ok = greeks is not None or ctx.proposal.intent is not Intent.OPEN
    return CheckResult(
        name="known_greeks",
        passed=ok,
        detail=(
            "net greeks present"
            if greeks is not None
            else "net greeks are unknown; an opening trade must state its exposure"
        ),
        observed="present" if greeks is not None else "missing",
    )


def fresh_quotes(ctx: Context) -> CheckResult:
    """Stale quotes make every other number in this file a guess.

    Age is the *oldest* leg's age (specs/02 D4), aged forward to the `as_of`
    argument and never measured against a clock read. Inclusive: a quote exactly
    `max_quote_age` seconds old is still fresh.
    """
    age = ctx.quote_age
    limit = ctx.limits.max_quote_age
    ok = age <= limit
    return CheckResult(
        name="fresh_quotes",
        passed=ok,
        detail=f"oldest leg quote is {age:.1f}s old, limit {limit:.1f}s",
        observed=age,
        limit=limit,
    )


# --------------------------------------------------------------------- #
# Budgeted — configurable, but never disabled.
# --------------------------------------------------------------------- #


def per_trade_loss(ctx: Context) -> CheckResult:
    """One trade may not risk more than its slice of equity."""
    observed = ctx.trade_loss
    limit = ctx.limits.max_trade_loss(ctx.portfolio.equity)
    return CheckResult(
        name="per_trade_loss",
        passed=observed <= limit,
        detail=f"order risks {observed}, per-trade limit {limit}",
        observed=observed,
        limit=limit,
    )


def portfolio_heat(ctx: Context) -> CheckResult:
    """Total risk on the book after this fill.

    Maximum loss rather than mark-to-market, on purpose: heat answers "what if
    everything goes wrong at once", and that question does not care what the
    position happens to be worth right now.
    """
    observed = ctx.portfolio.open_risk + ctx.trade_loss
    limit = ctx.limits.max_portfolio_loss(ctx.portfolio.equity)
    return CheckResult(
        name="portfolio_heat",
        passed=observed <= limit,
        detail=f"book would carry {observed} of risk, limit {limit}",
        observed=observed,
        limit=limit,
    )


def position_count(ctx: Context) -> CheckResult:
    """A cap on how many things can go wrong independently.

    It is also a cap on how much one human can reconcile by hand at the end of a
    competition day, which is why it is 8 and not 80.
    """
    observed = ctx.portfolio.open_structures
    limit = ctx.limits.max_open_structures
    return CheckResult(
        name="position_count",
        passed=observed < limit,
        detail=f"{observed} structures open, limit {limit}",
        observed=observed,
        limit=limit,
    )


def underlying_concentration(ctx: Context) -> CheckResult:
    """Eight positions in one name is one position wearing eight hats."""
    underlying = ctx.proposal.structure.underlying
    observed = ctx.portfolio.exposure_to(underlying) + ctx.trade_loss
    limit = ctx.limits.max_per_underlying(ctx.portfolio.equity)
    return CheckResult(
        name="underlying_concentration",
        passed=observed <= limit,
        detail=f"{underlying} exposure would be {observed}, limit {limit}",
        observed=observed,
        limit=limit,
    )


def net_delta_budget(ctx: Context) -> CheckResult:
    """Directional exposure after the fill, against a band scaled to equity.

    Unknown exposure fails. This is `known_greeks` applied to the book: a
    portfolio delta of `None` is not a portfolio delta of zero, and treating it
    as zero is how a directional book convinces itself it is neutral.
    """
    return _greek_budget(ctx, "net_delta_budget", "delta")


def net_vega_budget(ctx: Context) -> CheckResult:
    """Volatility exposure after the fill. Same rule, same scaling."""
    return _greek_budget(ctx, "net_vega_budget", "vega")


def _greek_budget(ctx: Context, name: str, greek: str) -> CheckResult:
    if greek == "delta":
        band = ctx.limits.scaled_delta_band(ctx.portfolio.equity)
        book = ctx.portfolio.net_delta
    else:
        band = ctx.limits.scaled_vega_band(ctx.portfolio.equity)
        book = ctx.portfolio.net_vega
    low, high = band
    rendered_band = f"[{low:.4g}, {high:.4g}]"
    proposed = ctx.proposal.risk.net_greeks

    if book is None or proposed is None:
        missing = "the book" if book is None else "the proposal"
        return CheckResult(
            name=name,
            passed=False,
            detail=f"net {greek} is unknown: {missing} has a position without greeks",
            observed=None,
            limit=rendered_band,
        )

    after = book + float(getattr(proposed, greek)) * ctx.proposal.quantity
    return CheckResult(
        name=name,
        passed=low <= after <= high,
        detail=f"net {greek} would be {after:.4g}, band {rendered_band}",
        observed=after,
        limit=rendered_band,
    )


def liquidity(ctx: Context) -> CheckResult:
    """The spread is the risk, and it is the part of the loss that is certain.

    Measured on the widest leg, because a multi-leg order crosses all of them.
    """
    observed = ctx.proposal.risk.worst_spread_pct
    limit = ctx.limits.max_spread_pct
    return CheckResult(
        name="liquidity",
        passed=observed <= limit,
        detail=f"widest leg spread is {observed}, limit {limit}",
        observed=observed,
        limit=limit,
    )


def expiry_window(ctx: Context) -> CheckResult:
    """Days to expiry, inclusive on both ends.

    The lower bound excludes 0DTE; the upper bound keeps a position able to
    round-trip inside the scored window. As much a strategy claim as a risk one
    — specs/07 D5.
    """
    observed = ctx.proposal.risk.days_to_expiry
    low, high = ctx.limits.dte_range
    return CheckResult(
        name="expiry_window",
        passed=low <= observed <= high,
        detail=f"{observed} days to expiry, window [{low}, {high}]",
        observed=observed,
        limit=f"[{low}, {high}]",
    )


def drawdown_killswitch(ctx: Context) -> CheckResult:
    """The check that matters for the P&L criterion.

    It trips at the threshold and *latches*: once tripped, the flag rides in on
    every later snapshot until a human clears it, so recovering a little equity
    does not quietly re-arm the strategy that lost it. Opens are refused
    unconditionally; closes are waived like every other check, because the Gate
    must never block an exit.
    """
    observed = ctx.portfolio.drawdown_pct
    limit = ctx.limits.max_drawdown_pct
    latched = ctx.portfolio.killswitch_tripped
    tripped = latched or observed >= limit
    detail = (
        "kill switch latched; opens stay blocked until it is re-armed by hand"
        if latched
        else f"drawdown {observed}, kill switch at {limit}"
    )
    return CheckResult(
        name="drawdown_killswitch",
        passed=not tripped,
        detail=detail,
        observed=observed,
        limit=limit,
    )


def daily_trade_cap(ctx: Context) -> CheckResult:
    """A ceiling on how fast a bad day can compound.

    Counts fills, not proposals: a Gate that counted its own vetoes could talk
    itself out of trading by refusing to trade.
    """
    observed = ctx.portfolio.fills_today
    limit = ctx.limits.max_daily_trades
    return CheckResult(
        name="daily_trade_cap",
        passed=observed < limit,
        detail=f"{observed} fills today, cap {limit}",
        observed=observed,
        limit=limit,
    )


CHECKS: Final[tuple[Check, ...]] = (
    # Structural first: if the trade is not knowable, the budget arithmetic
    # below is arithmetic over numbers nobody should trust.
    defined_risk,
    known_greeks,
    fresh_quotes,
    # Budgeted, in the order they appear in specs/03 D4.
    per_trade_loss,
    portfolio_heat,
    position_count,
    underlying_concentration,
    net_delta_budget,
    net_vega_budget,
    liquidity,
    expiry_window,
    drawdown_killswitch,
    daily_trade_cap,
)
"""The declared order. This tuple is the determinism guarantee of specs/03 D6."""

_EXIT_WAIVER: Final = "waived: this is an exit, and the Gate never blocks an exit (specs/03 D4)"


def waive_for_exit(result: CheckResult) -> CheckResult:
    """Turn a failing check into a passing one for a closing order.

    The observation is kept exactly as measured — the dashboard still shows that
    the book was over its heat limit while the close went through. Only the
    verdict changes, and the detail says why.
    """
    if result.passed:
        return result
    return CheckResult(
        name=result.name,
        passed=True,
        detail=f"{result.detail} — {_EXIT_WAIVER}",
        observed=result.observed,
        limit=result.limit,
    )


def run_all(ctx: Context) -> tuple[CheckResult, ...]:
    """Every check, in the declared order, with no short-circuit."""
    results = tuple(check(ctx) for check in CHECKS)
    if ctx.opens_risk:
        return results
    return tuple(waive_for_exit(result) for result in results)
