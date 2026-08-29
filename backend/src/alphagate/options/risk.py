"""Structure risk — specs/02 D4 and D5.

The numbers the Gate needs, computed purely from a structure and a quote set.

**The sign convention is the finance one and it is not Alpaca's.** A credit is
positive here. Alpaca's `limit_price` says the opposite, and the flip happens in
exactly one named function in the execution adapter (specs/04 D2). Do not
"correct" the convention in this module to match the wire format: the arithmetic
below reads correctly only this way, and a sign flip applied in two places is a
sign flip applied nowhere.

**`max_loss` is always finite and positive.** That is a type-level invariant,
not a check: a structure whose loss is unbounded cannot be constructed
(specs/02 D3), so there is no arm of this function that can return `None`. The
Gate's `defined_risk` check is therefore a belt-and-braces assertion rather than
the primary defence, which is the correct ordering.

**Missing greeks propagate as `None`.** If any leg lacks greeks, the net is
`None` — never a partial sum, and never zero. A partial sum would understate
exposure by exactly the legs we know least about.

D5: pure function of `(structure, quotes, as_of)`. No clock, no randomness.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal

from alphagate.core.errors import InvariantViolation
from alphagate.options.contract import OptionContract, Right, Side
from alphagate.options.quote import Greeks, OptionQuote
from alphagate.options.structure import OptionStructure, StructureKind

__all__ = ["StructureRisk", "compute_risk"]


@dataclass(frozen=True, slots=True)
class StructureRisk:
    """Everything the Gate needs to judge a structure, and nothing else."""

    net_premium: Decimal
    """Credit received is positive; debit paid is negative."""
    max_loss: Decimal
    """Always finite and > 0."""
    max_profit: Decimal
    breakevens: tuple[Decimal, ...]
    net_greeks: Greeks | None
    worst_spread_pct: Decimal
    days_to_expiry: int
    quote_age_seconds: float
    """Age of the *oldest* leg quote. The Gate's freshness check reads this."""

    def __post_init__(self) -> None:
        # Order matters: `Decimal("NaN") <= 0` raises InvalidOperation, so the
        # finiteness half of this invariant has to be asked first or the caller
        # gets an arithmetic error where a domain error was promised.
        if not self.max_loss.is_finite() or self.max_loss <= 0:
            raise InvariantViolation(
                f"max_loss must be positive and finite, got {self.max_loss}; "
                "an unbounded structure should not have been constructible"
            )

    @property
    def is_credit(self) -> bool:
        return self.net_premium > 0

    @property
    def return_on_risk(self) -> Decimal:
        """Max profit as a fraction of max loss. The strategy's ranking metric."""
        return self.max_profit / self.max_loss


def _leg_quote(
    quotes: Mapping[OptionContract, OptionQuote], contract: OptionContract
) -> OptionQuote:
    quote = quotes.get(contract)
    if quote is None:
        raise InvariantViolation(f"no quote for {contract}; risk cannot be computed without one")
    return quote


def compute_risk(
    structure: OptionStructure,
    quotes: Mapping[OptionContract, OptionQuote],
    as_of: datetime,
) -> StructureRisk:
    """Pure. Same inputs, same output, no clock read.

    ``as_of`` is used only to age the quotes and to compute days to expiry. It is
    an argument rather than a `now()` call so a backtest and a live run take the
    identical code path — the only difference between them being what time they
    say it is.
    """
    if as_of.tzinfo is None:
        raise InvariantViolation(f"as_of must be tz-aware, got {as_of!r}")

    legs = structure.legs
    multiplier = structure.multiplier

    # Premium: a sold leg brings money in, a bought leg takes it out.
    net_premium = Decimal(0)
    worst_spread = Decimal(0)
    oldest_age = float("-inf")
    greeks: Greeks | None = Greeks(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    for leg in legs:
        quote = _leg_quote(quotes, leg.contract)
        cash = quote.mid * leg.quantity * multiplier
        net_premium += -leg.side.signum * cash
        worst_spread = max(worst_spread, quote.spread_pct)
        oldest_age = max(oldest_age, quote.age_seconds(as_of))

        if greeks is None or quote.greeks is None:
            # One missing leg poisons the whole net. Anything else would report
            # an exposure smaller than the real one.
            greeks = None
        else:
            greeks = greeks.plus(
                quote.greeks.scaled(leg.side.signum * leg.quantity * multiplier)
            )

    expiry_days = structure.days_to_expiry(as_of.date())
    max_loss, max_profit = _bounds(structure, net_premium)
    breakevens = _breakevens(structure, net_premium)

    return StructureRisk(
        net_premium=net_premium,
        max_loss=max_loss,
        max_profit=max_profit,
        breakevens=breakevens,
        net_greeks=greeks,
        worst_spread_pct=worst_spread,
        days_to_expiry=expiry_days,
        quote_age_seconds=oldest_age,
    )


def _bounds(structure: OptionStructure, net_premium: Decimal) -> tuple[Decimal, Decimal]:
    """Maximum loss and maximum profit, in account currency.

    Each arm is the textbook identity for its kind. They are written out rather
    than generalised because a general formula over arbitrary legs would have to
    handle shapes this module refuses to construct, and the branch that handles
    an impossible shape is the branch nobody tests.
    """
    notional = Decimal(structure.multiplier * structure.quantity)
    kind = structure.kind

    if kind is StructureKind.VERTICAL_CREDIT:
        # Risk the width, keep the credit.
        return structure.width * notional - net_premium, net_premium

    if kind is StructureKind.VERTICAL_DEBIT:
        # Lose the debit, make the width less the debit.
        debit = -net_premium
        return debit, structure.width * notional - debit

    if kind is StructureKind.IRON_CONDOR:
        # Only one wing can finish in the money, so the wider one bounds the loss.
        return structure.width * notional - net_premium, net_premium

    if kind is StructureKind.CASH_SECURED_PUT:
        # Assignment to zero, not "unlimited" (specs/02 D4).
        return structure.legs[0].contract.strike * notional - net_premium, net_premium

    # Covered call: the stock going to zero, offset by the credit taken in.
    basis = _cover_basis(structure)
    leg = structure.legs[0]
    shares = Decimal(structure.cover.shares)  # type: ignore[union-attr]
    called_away = (leg.contract.strike - basis) * shares + net_premium
    return basis * shares - net_premium, called_away


def _breakevens(structure: OptionStructure, net_premium: Decimal) -> tuple[Decimal, ...]:
    """Underlying prices at expiry where the position breaks even, per share."""
    notional = Decimal(structure.multiplier * structure.quantity)
    per_share = net_premium / notional
    kind = structure.kind

    if kind in (StructureKind.VERTICAL_CREDIT, StructureKind.VERTICAL_DEBIT):
        short = next(leg for leg in structure.legs if leg.side is Side.SELL)
        long = next(leg for leg in structure.legs if leg.side is Side.BUY)
        anchor = short if kind is StructureKind.VERTICAL_CREDIT else long
        if anchor.contract.right is Right.CALL:
            return (anchor.contract.strike + abs(per_share),)
        return (anchor.contract.strike - abs(per_share),)

    if kind is StructureKind.IRON_CONDOR:
        put_short = max(
            (leg for leg in structure.short_legs if leg.contract.right is Right.PUT),
            key=lambda leg: leg.contract.strike,
        )
        call_short = min(
            (leg for leg in structure.short_legs if leg.contract.right is Right.CALL),
            key=lambda leg: leg.contract.strike,
        )
        return (
            put_short.contract.strike - per_share,
            call_short.contract.strike + per_share,
        )

    if kind is StructureKind.CASH_SECURED_PUT:
        return (structure.legs[0].contract.strike - per_share,)

    return (_cover_basis(structure) - per_share,)


def _cover_basis(structure: OptionStructure) -> Decimal:
    """The stock basis backing a covered call.

    Guaranteed present by `OptionStructure` validation. Re-checked rather than
    asserted because an `assert` vanishes under `python -O`, and this value is
    the denominator of a loss figure.
    """
    if structure.cover is None or structure.cover.basis is None:
        raise InvariantViolation(
            f"{structure.kind.name} reached risk computation without a cover basis"
        )
    return structure.cover.basis


def expiry_date(structure: OptionStructure) -> date:
    return structure.expiry
