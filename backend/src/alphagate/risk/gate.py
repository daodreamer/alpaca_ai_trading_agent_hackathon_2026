"""The Risk Gate — specs/03 D1 and D6.

PURE. Stdlib plus `alphagate.core`, `alphagate.options` and its own siblings.
**No LLM, no I/O, no clock read, no network.** This is the layer that makes the
project's claim true, so it is the layer with the least freedom.

One public function. Everything it needs is an argument, *including the time* —
which is what lets a backtest and a live run take the identical code path, the
only difference between them being what hour they say it is (specs/01 Rule 4).

This module is also the only place in the codebase where a `GatedOrder` can be
minted; `verdict.py` enforces that structurally rather than by convention. If
you find yourself wanting to construct one somewhere else, the thing you
actually want is to call `evaluate`.
"""

from __future__ import annotations

from datetime import datetime

from alphagate.core.errors import InvariantViolation
from alphagate.risk.checks import Context, run_all
from alphagate.risk.limits import RiskLimits
from alphagate.risk.portfolio import PortfolioSnapshot
from alphagate.risk.proposal import TradeProposal
from alphagate.risk.verdict import (
    Approved,
    CheckResult,
    GatedOrder,
    Verdict,
    Vetoed,
    VetoReason,
)

__all__ = ["evaluate"]


def evaluate(
    proposal: TradeProposal,
    portfolio: PortfolioSnapshot,
    limits: RiskLimits,
    as_of: datetime,
) -> Verdict:
    """Judge one proposal. Pure, total, and deterministic.

    Every check runs, always — the Gate does not short-circuit on the first
    veto. A refusal with one reason and a refusal with five are different
    situations, and the journal (specs/06) should be able to show which.

    Determinism (specs/03 D6): same `(proposal, portfolio, limits, as_of)`
    produces the same verdict, including the order of `checks` and `reasons`.
    Both orders come from `checks.CHECKS`, a tuple. Nothing here iterates a set
    or relies on dict ordering, and nothing here reads a clock.
    """
    if as_of.tzinfo is None:
        raise InvariantViolation(
            f"as_of must be tz-aware UTC, got {as_of!r}; the Gate never reads a clock, "
            "so a naive timestamp here is a caller that lost its timezone"
        )

    results = run_all(Context(proposal, portfolio, limits, as_of))
    reasons = tuple(
        VetoReason(check=result.name, detail=result.detail)
        for result in results
        if not result.passed
    )
    if reasons:
        return Vetoed(reasons=reasons, checks=results)
    return _approve(proposal, results, as_of)


def _approve(
    proposal: TradeProposal, checks: tuple[CheckResult, ...], as_of: datetime
) -> Approved:
    """Mint the order. Called only from `evaluate`, only when nothing failed.

    `limit_price` carries the **domain** sign convention — a credit is positive.
    Alpaca inverts it, and that flip happens in exactly one named function in the
    execution adapter (specs/04 D2). Flipping it here as well would be flipping
    it nowhere.
    """
    order = GatedOrder(
        structure=proposal.structure,
        quantity=proposal.quantity,
        intent=proposal.intent,
        limit_price=proposal.risk.net_premium,
        approved_at=as_of,
        proposal_id=proposal.proposal_id,
    )
    return Approved(order=order, checks=checks)
