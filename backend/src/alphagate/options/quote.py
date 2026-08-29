"""Quotes and greeks — specs/02 D2.

Two rules from the spec drive every decision here.

**Money is Decimal; greeks and IV are float.** A price is a fact with an exact
value. A delta is an estimate produced by a model from an implied volatility
that was itself backed out of a price. Giving them the same type invites
arithmetic that treats a guess as a fact.

**Missing greeks are None, never zero.** A provider that omits delta must not be
readable as delta-neutral. The Gate's `known_greeks` check exists precisely to
refuse an opening trade whose exposure nobody can state, and it can only do that
if absence is representable.

**The domain never reads the clock.** Age is measured against a supplied
timestamp. `is_stale` reports; the Gate decides.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import ROUND_HALF_EVEN, Decimal
from typing import Final

from alphagate.core.errors import InvariantViolation
from alphagate.options.contract import OptionContract

__all__ = ["MAX_QUOTE_AGE", "Greeks", "OptionQuote"]

MAX_QUOTE_AGE: Final = 60.0
"""Seconds after which a quote is stale by default (specs/03 D5)."""

_CENT: Final = Decimal("0.01")


@dataclass(frozen=True, slots=True)
class Greeks:
    """Per-contract sensitivities. Floats: these are estimates, not money."""

    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    iv: float

    def scaled(self, factor: float) -> Greeks:
        """Scale the extensive sensitivities by ``factor``.

        IV is deliberately not scaled. Delta and vega are per-contract exposures
        and double when you hold two; implied volatility is a rate describing the
        underlying, and doubling it would be meaningless.
        """
        return Greeks(
            delta=self.delta * factor,
            gamma=self.gamma * factor,
            theta=self.theta * factor,
            vega=self.vega * factor,
            rho=self.rho * factor,
            iv=self.iv,
        )

    def plus(self, other: Greeks) -> Greeks:
        """Add two exposures. The resulting ``iv`` is the notional-free mean.

        Summing IV would be nonsense, and picking one leg's IV would be
        arbitrary, so the aggregate reports the average as a summary statistic.
        Nothing in the Gate keys off aggregate IV; the strategy reads IV rank
        from the underlying, not from a structure.
        """
        return Greeks(
            delta=self.delta + other.delta,
            gamma=self.gamma + other.gamma,
            theta=self.theta + other.theta,
            vega=self.vega + other.vega,
            rho=self.rho + other.rho,
            iv=(self.iv + other.iv) / 2.0,
        )


@dataclass(frozen=True, slots=True)
class OptionQuote:
    """A two-sided market for one contract at one instant."""

    contract: OptionContract
    as_of: datetime
    bid: Decimal
    ask: Decimal
    greeks: Greeks | None = None
    open_interest: int | None = None
    volume: int | None = None

    def __post_init__(self) -> None:
        if self.as_of.tzinfo is None:
            raise InvariantViolation(f"quote timestamp must be tz-aware, got {self.as_of!r}")
        for name in ("bid", "ask"):
            value = getattr(self, name)
            if not isinstance(value, Decimal):
                raise InvariantViolation(f"{name} must be Decimal, got {type(value).__name__}")
            # Before any comparison: a non-finite Decimal makes `<` raise
            # InvalidOperation instead of answering (see OptionContract).
            if not value.is_finite():
                raise InvariantViolation(f"{name} must be finite, got {value}")
            if value < 0:
                raise InvariantViolation(f"{name} must not be negative, got {value}")
        if self.bid > self.ask:
            raise InvariantViolation(
                f"crossed market: bid {self.bid} exceeds ask {self.ask}"
            )

    @property
    def mid(self) -> Decimal:
        """Midpoint, quantised to a cent."""
        return ((self.bid + self.ask) / 2).quantize(_CENT, rounding=ROUND_HALF_EVEN)

    @property
    def spread_pct(self) -> Decimal:
        """Bid/ask width as a fraction of mid.

        A zero mid means a market with no value on either side; it reports as
        fully wide rather than dividing by zero, which is the conservative
        reading and the one the liquidity check should veto on.
        """
        mid = self.mid
        if mid <= 0:
            return Decimal(1)
        return (self.ask - self.bid) / mid

    def age_seconds(self, as_of: datetime) -> float:
        """Seconds between this quote and ``as_of``. Negative if it is ahead.

        Clock skew is left visible rather than clamped: a feed running ahead of
        the local clock is a real operational problem, and silently reporting
        zero would hide it.
        """
        if as_of.tzinfo is None:
            raise InvariantViolation(f"as_of must be tz-aware, got {as_of!r}")
        return (as_of - self.as_of).total_seconds()

    def is_stale(self, as_of: datetime, *, max_age: float = MAX_QUOTE_AGE) -> bool:
        return self.age_seconds(as_of) > max_age
