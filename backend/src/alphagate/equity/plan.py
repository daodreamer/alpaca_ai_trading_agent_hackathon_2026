"""Weights → shares → a deterministic sequence of orders. specs/09 D2–D4.

PURE. Stdlib plus `alphagate.core` and its own siblings. No clock, no I/O, no
prices fetched — `as_of` and the marks are arguments, which is what lets a
replay and a live pass take the identical code path.

The three things this module decides, and the reason each is here rather than
upstream:

**Sizing.** The book carries weights and nothing else, on purpose: the
researcher does not know the account's equity and inventing one would have been
its first step towards placing the order. So equity, prices, and what is
actually held all meet for the first time in this function.

**The no-trade band.** Between rebalances the targets are constant and the
holdings drift with price. Diffing them daily would trade the whole book every
session and pay costs the backtest never charged — so a symbol is left alone
until its drift is worth an order (specs/09 D3).

**A symbol held but no longer in the book is sold to zero.** That is the exit
path, and it is the same arithmetic as every other: its target weight is
absent, so its target notional is zero, so the delta is the whole position.
There is no separate "close" code path to forget to call.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import ROUND_DOWN, Decimal
from enum import Enum

from alphagate.core.errors import InvariantViolation
from alphagate.core.identifiers import Ticker
from alphagate.equity.book import TargetBook
from alphagate.equity.policy import EquityPolicy

__all__ = [
    "EquitySide",
    "Holding",
    "Mark",
    "OrderIntent",
    "RebalancePlan",
    "SkipReason",
    "Skipped",
    "plan_rebalance",
]


class EquitySide(Enum):
    """Which way one order goes.

    Deliberately not `alphagate.options.Side`. An option leg's side is part of a
    structure whose risk is computed from four legs at once; a share order's is
    a direction and a sign. Sharing the type would invite sharing the code that
    interprets it, and the two mean different things.
    """

    BUY = "buy"
    SELL = "sell"

    @property
    def reduces_risk(self) -> bool:
        """Whether an order of this side can only make the book smaller.

        Because the planner never emits a sell beyond the held quantity
        (specs/09 D6), a sell is always a reduction — which is what earns it the
        Gate's exit waiver, the same way a closing option order does in
        [03](../../../../specs/03-risk-gate.md) D4.
        """
        return self is EquitySide.SELL


class SkipReason(Enum):
    """Why a symbol in the book produced no order.

    An enum rather than a string because these are counted and rendered, and a
    tally keyed on free text is a tally that silently splits when somebody
    rewords a message.
    """

    INSIDE_BAND = "inside_band"
    """The drift was not worth an order. The common case, four days in five."""
    NO_MARK = "no_mark"
    """No usable price. Held, loudly — see the module docstring's third point in
    reverse: we do not know what it is worth, so we do not trade it."""
    STALE_MARK = "stale_mark"
    NOT_TRADEABLE = "not_tradeable"
    ROUNDS_TO_ZERO = "rounds_to_zero"
    """A whole-share asset whose target is smaller than one share. A real hole
    in the book, and the record says so rather than dropping it."""
    ALREADY_FLAT = "already_flat"


@dataclass(frozen=True, slots=True)
class Holding:
    """One equity line as the broker reports it.

    `shares` is signed for honesty about what the account contains, even though
    nothing here can produce a negative one. A short position that arrived some
    other way must be visible rather than parsed away.
    """

    symbol: Ticker
    shares: Decimal
    average_price: Decimal
    market_value: Decimal

    def __post_init__(self) -> None:
        for name in ("shares", "average_price", "market_value"):
            value = getattr(self, name)
            if not isinstance(value, Decimal):
                raise InvariantViolation(f"{name} must be Decimal, got {type(value).__name__}")
            if not value.is_finite():
                raise InvariantViolation(f"{name} must be finite, got {value}")


@dataclass(frozen=True, slots=True)
class Mark:
    """What one symbol is worth right now, and whether it can be traded.

    `age_seconds` travels with the price rather than being recomputed from a
    clock, so the Gate's freshness check is a comparison between two arguments
    and never a clock read (specs/03 D1, applied here).
    """

    symbol: Ticker
    price: Decimal
    age_seconds: float
    tradeable: bool
    fractionable: bool

    def __post_init__(self) -> None:
        if not isinstance(self.price, Decimal):
            raise InvariantViolation(f"price must be Decimal, got {type(self.price).__name__}")
        if not self.price.is_finite() or self.price <= 0:
            raise InvariantViolation(
                f"{self.symbol}: price must be finite and positive, got {self.price}"
            )


@dataclass(frozen=True, slots=True)
class OrderIntent:
    """One order the plan wants placed, before any Gate has seen it.

    Not a `GatedEquityOrder` and deliberately nothing like one: this type is
    constructible anywhere, carries no approval, and `execution` will not accept
    it. It becomes submittable only by passing `risk.equity_gate.evaluate`.
    """

    symbol: Ticker
    side: EquitySide
    shares: Decimal
    reference_price: Decimal
    target_weight: Decimal
    held_weight: Decimal
    held_shares: Decimal
    fractionable: bool
    mark_age_seconds: float

    def __post_init__(self) -> None:
        if self.shares <= 0:
            raise InvariantViolation(
                f"{self.symbol}: an order intent must move a positive number of "
                f"shares, got {self.shares}; direction lives in `side`"
            )
        if self.side is EquitySide.SELL and self.shares > self.held_shares:
            raise InvariantViolation(
                f"{self.symbol}: selling {self.shares} against {self.held_shares} held "
                "would open a short. specs/09 D6 — not expressible here"
            )

    @property
    def notional(self) -> Decimal:
        return self.shares * self.reference_price

    @property
    def resulting_shares(self) -> Decimal:
        return (
            self.held_shares + self.shares
            if self.side is EquitySide.BUY
            else self.held_shares - self.shares
        )

    @property
    def resulting_notional(self) -> Decimal:
        return self.resulting_shares * self.reference_price

    def label(self) -> str:
        return f"{self.side.value} {self.shares} {self.symbol} @ ~{self.reference_price}"


@dataclass(frozen=True, slots=True)
class Skipped:
    """A symbol the plan considered and did not trade, and why."""

    symbol: Ticker
    reason: SkipReason
    detail: str
    drift_notional: Decimal = Decimal(0)


@dataclass(frozen=True, slots=True)
class RebalancePlan:
    """Everything one rebalance pass decided, including the parts that decided nothing.

    `skipped` is not diagnostic clutter. On this strategy the plan is empty four
    days in five, and "why did it not trade today" is the question the journal
    exists to answer ([06](../../../../specs/06-journal.md) D1). A plan that
    recorded only its orders would answer it with silence.
    """

    fingerprint: str
    book_as_of: str
    book_digest: str
    as_of: datetime
    equity: Decimal
    intents: tuple[OrderIntent, ...]
    skipped: tuple[Skipped, ...]
    band: Decimal
    """The threshold this pass used, in currency. Recorded because it is the
    number that explains every `INSIDE_BAND` line below it."""

    @property
    def is_empty(self) -> bool:
        return not self.intents

    @property
    def buy_notional(self) -> Decimal:
        return sum(
            (i.notional for i in self.intents if i.side is EquitySide.BUY), Decimal(0)
        )

    @property
    def sell_notional(self) -> Decimal:
        return sum(
            (i.notional for i in self.intents if i.side is EquitySide.SELL), Decimal(0)
        )

    @property
    def turnover(self) -> Decimal:
        return self.buy_notional + self.sell_notional

    def counts(self) -> dict[str, int]:
        tally: dict[str, int] = {}
        for skip in self.skipped:
            tally[skip.reason.value] = tally.get(skip.reason.value, 0) + 1
        return tally


# --------------------------------------------------------------------------- #
# The plan
# --------------------------------------------------------------------------- #


def plan_rebalance(
    book: TargetBook,
    *,
    holdings: Sequence[Holding],
    marks: Mapping[Ticker, Mark],
    equity: Decimal,
    policy: EquityPolicy,
    as_of: datetime,
) -> RebalancePlan:
    """Turn a book, an account and a set of prices into an ordered sequence of orders.

    Deterministic: the same arguments produce the same plan, intent for intent
    and skip for skip, including the order of both. Every iteration below is
    over a sorted sequence — nothing here walks a dict or a set in insertion
    order (CLAUDE.md §3 rule 7).

    `as_of` is carried into the plan rather than read, so a replayed pass and a
    live one are the same call with a different argument.
    """
    if as_of.tzinfo is None:
        raise InvariantViolation(
            f"as_of must be tz-aware UTC, got {as_of!r}; nothing in this layer reads "
            "a clock, so a naive timestamp is a caller that lost its timezone"
        )
    if not isinstance(equity, Decimal):
        raise InvariantViolation(
            f"equity must be Decimal, got {type(equity).__name__}; every target "
            "notional in this function is a fraction of it, and a float "
            "denominator makes every one of them approximate"
        )
    # Finiteness before the comparison: `Decimal("NaN") <= 0` raises rather than
    # answering, and an invariant check must return a verdict.
    if not equity.is_finite() or equity <= 0:
        raise InvariantViolation(f"equity must be finite and positive, got {equity}")

    held = _held_by_symbol(holdings)
    band = policy.band(equity)

    # The union, sorted. A symbol held but absent from the book has a target of
    # zero and is sold out — the exit path, arrived at by the same arithmetic as
    # everything else rather than by a branch that could be forgotten.
    universe = sorted(set(book.weights) | set(held))

    intents: list[OrderIntent] = []
    skipped: list[Skipped] = []

    for symbol in universe:
        target_weight = book.weights.get(symbol, Decimal(0))
        held_shares = held.get(symbol, Decimal(0))
        outcome = _decide(
            symbol,
            target_weight=target_weight,
            held_shares=held_shares,
            mark=marks.get(symbol),
            equity=equity,
            band=band,
            policy=policy,
        )
        if isinstance(outcome, OrderIntent):
            intents.append(outcome)
        else:
            skipped.append(outcome)

    return RebalancePlan(
        fingerprint=book.fingerprint,
        book_as_of=book.as_of.isoformat(),
        book_digest=book.digest,
        as_of=as_of,
        equity=equity,
        intents=tuple(_sequenced(intents)),
        skipped=tuple(skipped),
        band=band,
    )


def _decide(
    symbol: Ticker,
    *,
    target_weight: Decimal,
    held_shares: Decimal,
    mark: Mark | None,
    equity: Decimal,
    band: Decimal,
    policy: EquityPolicy,
) -> OrderIntent | Skipped:
    """One symbol. Returns the order to place, or the reason there is none."""
    if mark is None:
        return Skipped(
            symbol,
            SkipReason.NO_MARK,
            "no price; held rather than traded on a guess",
        )
    if mark.age_seconds > policy.max_quote_age:
        return Skipped(
            symbol,
            SkipReason.STALE_MARK,
            f"quote is {mark.age_seconds:.0f}s old, limit {policy.max_quote_age:.0f}s",
        )
    if not mark.tradeable:
        return Skipped(
            symbol,
            SkipReason.NOT_TRADEABLE,
            "the broker reports this asset untradeable",
        )

    target_notional = target_weight * equity
    held_notional = held_shares * mark.price
    drift = target_notional - held_notional

    if target_weight == 0 and held_shares == 0:
        return Skipped(symbol, SkipReason.ALREADY_FLAT, "not held and not wanted")

    if abs(drift) < band:
        return Skipped(
            symbol,
            SkipReason.INSIDE_BAND,
            f"drift {_money(drift)} is inside the {_money(band)} band",
            drift_notional=drift,
        )

    shares = _shares_for(drift, mark, policy)
    if shares == 0:
        return Skipped(
            symbol,
            SkipReason.ROUNDS_TO_ZERO,
            (
                f"{_money(abs(drift))} of a whole-share asset at {mark.price} is "
                "less than one share; the book cannot be held exactly"
                if not mark.fractionable
                else f"{_money(abs(drift))} rounds to nothing at {mark.price}"
            ),
            drift_notional=drift,
        )

    side = EquitySide.BUY if drift > 0 else EquitySide.SELL
    if side is EquitySide.SELL:
        # Never sell more than is held. The planner is the first of three layers
        # that refuse a short (specs/09 D6); `OrderIntent.__post_init__` is the
        # second and the Gate is the third.
        shares = min(shares, held_shares)
        if shares <= 0:
            return Skipped(
                symbol,
                SkipReason.ALREADY_FLAT,
                "nothing held to sell",
                drift_notional=drift,
            )

    return OrderIntent(
        symbol=symbol,
        side=side,
        shares=shares,
        reference_price=mark.price,
        target_weight=target_weight,
        held_weight=held_notional / equity,
        held_shares=held_shares,
        fractionable=mark.fractionable,
        mark_age_seconds=mark.age_seconds,
    )


def _shares_for(drift: Decimal, mark: Mark, policy: EquityPolicy) -> Decimal:
    """How many shares that much money buys, rounded so we never overshoot.

    `ROUND_DOWN` is toward zero for a `Decimal`, which is the direction that is
    safe on both sides: a buy stops short of the target rather than exceeding
    the position cap, and a sell stops short of the holding rather than tipping
    it short.
    """
    raw = abs(drift) / mark.price
    if mark.fractionable:
        quantum = Decimal(1).scaleb(-policy.fractional_places)
        return raw.quantize(quantum, rounding=ROUND_DOWN)
    return raw.quantize(Decimal(1), rounding=ROUND_DOWN)


def _sequenced(intents: Sequence[OrderIntent]) -> list[OrderIntent]:
    """Sells first, then buys; within each, largest notional first, then symbol.

    specs/09 D4. Sells release the buying power the buys then spend, and on a
    fully-invested book the other order means the first buys are refused for
    want of cash while the last sells leave the account holding it — a rebalance
    that half-happened, which is worse than either end state.

    The symbol tiebreak is what makes the sequence a function of the inputs
    rather than of dictionary order.
    """
    def key(intent: OrderIntent) -> tuple[int, Decimal, str]:
        return (
            0 if intent.side is EquitySide.SELL else 1,
            -intent.notional,
            str(intent.symbol),
        )

    return sorted(intents, key=key)


def _held_by_symbol(holdings: Sequence[Holding]) -> dict[Ticker, Decimal]:
    """Collapse the broker's lines to one share count per symbol.

    Alpaca reports one line per symbol, so this is a dict comprehension in every
    real case. It is written as an accumulation anyway, because a duplicate line
    silently overwriting the first would understate a position, and understating
    a position is how a sell becomes a short.
    """
    totals: dict[Ticker, Decimal] = {}
    for holding in sorted(holdings, key=lambda h: str(h.symbol)):
        totals[holding.symbol] = totals.get(holding.symbol, Decimal(0)) + holding.shares
    return totals


def _money(value: Decimal) -> str:
    return f"${value.quantize(Decimal('0.01'))}"
