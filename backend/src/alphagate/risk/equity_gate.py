"""The equity Risk Gate — specs/09 D5.

PURE. Stdlib plus `alphagate.core`, `alphagate.equity` and its own siblings.
**No LLM, no I/O, no clock read, no network.**

One public function. Everything it needs is an argument, *including the time* —
which is what lets a replay and a live run take the identical code path, the
only difference between them being what hour they say it is.

This module is the only place in the codebase where a `GatedEquityOrder` can be
minted; `equity_verdict.py` enforces that structurally rather than by
convention, exactly as `verdict.py` does for the options door. If you find
yourself wanting to construct one somewhere else, the thing you actually want is
to call `evaluate_equity`.

`CONSUMER_MUST_SUPPLY` in the researcher's target book names "an equity-shaped
risk gate — AlphaGate's is options-shaped and does not apply". This is that
gate. It is not a translation of the options one: the checks are different
because the risks are, and the only things the two share are the discipline and
the `CheckResult` type.
"""

from __future__ import annotations

from datetime import datetime

from alphagate.core.errors import InvariantViolation
from alphagate.equity.book import TargetBook
from alphagate.equity.plan import OrderIntent
from alphagate.equity.policy import EquityPolicy
from alphagate.risk.equity_checks import EquityContext, run_equity_checks, waive_for_reduction
from alphagate.risk.equity_portfolio import EquityPortfolio
from alphagate.risk.equity_verdict import (
    ApprovedEquity,
    EquityVerdict,
    GatedEquityOrder,
    VetoedEquity,
)
from alphagate.risk.verdict import CheckResult, VetoReason

__all__ = ["evaluate_equity"]


def evaluate_equity(
    intent: OrderIntent,
    book: TargetBook,
    portfolio: EquityPortfolio,
    policy: EquityPolicy,
    as_of: datetime,
    *,
    pinned_fingerprint: str,
) -> EquityVerdict:
    """Judge one intent. Pure, total, and deterministic.

    Every check runs, always — the Gate does not short-circuit on the first
    veto. A refusal with one reason and a refusal with five are different
    situations, and the journal should be able to show which.

    Determinism: the same arguments produce the same verdict, including the
    order of `checks`, `reasons` and `waived`. All three come from
    `EQUITY_CHECKS`, a tuple. Nothing here iterates a set or relies on dict
    ordering, and nothing here reads a clock.

    A **sell is waived past a budget veto** but not past the five checks in
    `UNWAIVABLE`. The waived reasons travel on the approval rather than being
    dropped: "this order was allowed past the turnover cap" is a fact somebody
    reading the journal after a bad week needs to find.
    """
    if as_of.tzinfo is None:
        raise InvariantViolation(
            f"as_of must be tz-aware UTC, got {as_of!r}; the Gate never reads a clock, "
            "so a naive timestamp here is a caller that lost its timezone"
        )

    results = run_equity_checks(
        EquityContext(
            intent=intent,
            book=book,
            portfolio=portfolio,
            policy=policy,
            as_of=as_of,
            pinned_fingerprint=pinned_fingerprint,
        )
    )

    reduces = intent.side.reduces_risk
    failures = [result for result in results if not result.passed]
    waived = tuple(
        VetoReason(check=result.name, detail=result.detail)
        for result in failures
        if waive_for_reduction(result, reduces_risk=reduces)
    )
    reasons = tuple(
        VetoReason(check=result.name, detail=result.detail)
        for result in failures
        if not waive_for_reduction(result, reduces_risk=reduces)
    )
    if reasons:
        return VetoedEquity(reasons=reasons, checks=results)
    return _approve(intent, book, results, waived, as_of)


def _approve(
    intent: OrderIntent,
    book: TargetBook,
    checks: tuple[CheckResult, ...],
    waived: tuple[VetoReason, ...],
    as_of: datetime,
) -> ApprovedEquity:
    """Mint the order. Called only from `evaluate_equity`, only when nothing
    unwaivable failed.

    The order carries the book's fingerprint and session rather than a reference
    to the book itself, because those two strings are what the idempotency key
    is derived from — and an order that could not name the book it serves is an
    order nobody can reconcile against a plan.
    """
    order = GatedEquityOrder(
        symbol=intent.symbol,
        side=intent.side,
        shares=intent.shares,
        reference_price=intent.reference_price,
        fractionable=intent.fractionable,
        fingerprint=book.fingerprint,
        book_as_of=book.as_of.isoformat(),
        approved_at=as_of,
    )
    return ApprovedEquity(order=order, checks=checks, waived=waived)
