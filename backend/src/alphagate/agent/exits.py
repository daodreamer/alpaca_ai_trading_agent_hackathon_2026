"""Exits — specs/05 D8. Pure, deterministic, and **not the model's decision**.

> "Exits are evaluated every cycle and are not the model's decision: profit
> target, stop, and DTE-based close are deterministic rules. The Gate never
> blocks an exit."

The asymmetry between this module and `proposer.py` is the point. Entering is a
judgement call, so a model gets a vote; leaving is a rule, so it does not. A
model asked whether to close a losing position will find a reason to wait, every
time, and it will be articulate about it. That is the single most expensive
failure mode available to an agent with an open position, and the defence is not
a better prompt — it is that there is no prompt.

Three rules, checked in a fixed order, first match wins:

| Rule | Fires when |
| --- | --- |
| `PROFIT_TARGET` | the credit has decayed to `profit_target` of what was taken in |
| `STOP` | the loss reaches `stop_multiple` times the credit taken in |
| `DTE_CLOSE` | days to expiry falls to `min_dte` or below |

`DTE_CLOSE` is the one that is easy to leave out and expensive to omit. Gamma
rises sharply into the last week: a spread that has behaved for two weeks can
lose its whole width in an afternoon, and being right about direction does not
help if the position is closed by assignment instead of by us.

The rules are evaluated against a **current** `StructureRisk` — the position
marked to market now — and the credit recorded when it was opened. Both are
arguments; nothing here reads a clock or a quote.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Final

from alphagate.core.errors import InvariantViolation
from alphagate.options import OptionStructure, StructureRisk
from alphagate.risk import Intent, OpenPosition

__all__ = ["ExitDecision", "ExitPolicy", "ExitRule", "evaluate_exit"]


class ExitRule(Enum):
    """Why a position is being closed. Recorded in the journal verbatim."""

    PROFIT_TARGET = "profit_target"
    STOP = "stop"
    DTE_CLOSE = "dte_close"
    HOLD = "hold"

    @property
    def closes(self) -> bool:
        return self is not ExitRule.HOLD


@dataclass(frozen=True, slots=True)
class ExitPolicy:
    """The thresholds. One place, logged at startup, shown in the dashboard."""

    profit_target: Decimal = Decimal("0.50")
    """Close when half the credit has been earned. Not 100%: the last half of a
    credit spread's profit takes most of the remaining time and carries all of
    the remaining risk, and a strategy that always holds to expiry converts a
    good win rate into an occasional maximum loss."""

    stop_multiple: Decimal = Decimal("2.0")
    """Close when the loss reaches twice the credit taken in. With a 50% target
    that is a 1:4 win/loss ratio per trade, which needs a high win rate — which
    is what selling premium at a distance is for. Wider stops on defined-risk
    spreads mostly mean discovering the maximum loss more often."""

    min_dte: int = 2
    """Close at two days to expiry regardless. See the module docstring on gamma."""

    def __post_init__(self) -> None:
        if not Decimal(0) < self.profit_target <= Decimal(1):
            raise InvariantViolation(
                f"profit_target must be a fraction in (0, 1], got {self.profit_target}"
            )
        if self.stop_multiple <= 0:
            raise InvariantViolation(f"stop_multiple must be positive, got {self.stop_multiple}")
        if self.min_dte < 0:
            raise InvariantViolation(f"min_dte must not be negative, got {self.min_dte}")


DEFAULT_EXIT_POLICY: Final = ExitPolicy()


@dataclass(frozen=True, slots=True)
class ExitDecision:
    """Whether to close, why, and the numbers behind it."""

    rule: ExitRule
    structure: OptionStructure
    detail: str
    unrealised: Decimal
    """Profit if positive, loss if negative, in account currency."""
    fraction_of_credit: Decimal | None
    """Unrealised as a fraction of the credit taken in. `None` for a debit
    structure, where "fraction of the credit" names nothing."""
    days_to_expiry: int

    @property
    def should_close(self) -> bool:
        return self.rule.closes

    @property
    def intent(self) -> Intent:
        return Intent.CLOSE


def evaluate_exit(
    position: OpenPosition,
    current: StructureRisk,
    entry_premium: Decimal,
    *,
    as_of: datetime,
    policy: ExitPolicy = DEFAULT_EXIT_POLICY,
) -> ExitDecision:
    """Decide whether one open position should be closed. Pure.

    `entry_premium` follows the domain sign convention (specs/02 D4): a credit
    received is positive. `current.net_premium` is what it would cost to be flat
    now, in the same convention — so for a credit structure the unrealised gain
    is `entry - current`, because buying it back for less than you sold it is
    the profit.

    A structure opened for a debit has no "fraction of the credit", so the
    profit and stop rules are expressed against the debit paid instead, and
    `fraction_of_credit` reports `None` rather than a number that would read as
    a percentage of something that does not exist.
    """
    if as_of.tzinfo is None:
        raise InvariantViolation(f"as_of must be tz-aware, got {as_of!r}")

    unrealised = (entry_premium - current.net_premium) * position.quantity
    basis = abs(entry_premium) * position.quantity
    fraction = None if basis <= 0 else (unrealised / basis)
    dte = current.days_to_expiry

    def decide(rule: ExitRule, detail: str) -> ExitDecision:
        return ExitDecision(
            rule=rule,
            structure=position.structure,
            detail=detail,
            unrealised=unrealised,
            fraction_of_credit=fraction if entry_premium > 0 else None,
            days_to_expiry=dte,
        )

    # Fixed order, first match wins. Profit before stop so that a position that
    # somehow satisfies both — a wild mark, a crossed quote — is read the safe
    # way round: taking a profit that is not there costs a commission, taking a
    # loss that is not there costs the loss.
    if fraction is not None and fraction >= policy.profit_target:
        return decide(
            ExitRule.PROFIT_TARGET,
            f"{fraction:.0%} of the premium earned, target {policy.profit_target:.0%}",
        )
    if fraction is not None and fraction <= -policy.stop_multiple:
        return decide(
            ExitRule.STOP,
            f"loss is {abs(fraction):.1f}x the premium, stop at {policy.stop_multiple:.1f}x",
        )
    if dte <= policy.min_dte:
        return decide(
            ExitRule.DTE_CLOSE,
            f"{dte} days to expiry, closing at {policy.min_dte} — gamma risk into expiry",
        )
    return decide(
        ExitRule.HOLD,
        f"holding: {fraction:.0%} of premium, {dte} days left"
        if fraction is not None
        else f"holding: {dte} days left",
    )
