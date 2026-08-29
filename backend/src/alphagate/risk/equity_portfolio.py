"""The equity book as the Gate sees it — specs/09 D5.

PURE. Stdlib plus `alphagate.core` and `alphagate.equity`.

A *snapshot*, not a live account object. It is assembled by the layer above from
a broker read and handed to `evaluate_equity` as a value; the Gate never queries
it, never refreshes it, and never reads a clock to decide how old it is.

**The kill-switch latch lives here, not in the Gate**, for the reason
[03](../../../../specs/03-risk-gate.md) gives: a pure function cannot remember
that it tripped yesterday. The caller carries the flag in, and the check vetoes
on either the live drawdown crossing the limit or the latch already being set.

**Today's turnover and order count are carried, not inferred.** The Gate cannot
read the journal, so the two facts that make a daily cap meaningful arrive as
fields. A snapshot that guessed them would enforce a cap against a number
nobody measured.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal

from alphagate.core.errors import InvariantViolation
from alphagate.core.identifiers import Ticker
from alphagate.equity.plan import Holding

__all__ = ["EquityPortfolio"]


@dataclass(frozen=True, slots=True)
class EquityPortfolio:
    """Everything about the account the equity Gate is allowed to know."""

    equity: Decimal
    cash: Decimal
    buying_power: Decimal
    holdings: tuple[Holding, ...]
    marks: Mapping[Ticker, Decimal]
    """Current price per symbol, for valuing the book. Separate from `holdings`
    because a holding's `market_value` is the broker's own figure and may be
    minutes old, while a Gate check about a *resulting* position must be
    arithmetic on the same price the order was sized from."""

    drawdown_pct: Decimal
    """Peak-to-current as a fraction, e.g. `Decimal("0.03")` for 3%."""
    orders_today: int
    turnover_today: Decimal
    killswitch_tripped: bool = False
    """Latched by the caller. Cleared only by a human re-arming it."""

    def __post_init__(self) -> None:
        for name in ("equity", "cash", "buying_power", "turnover_today"):
            value = getattr(self, name)
            if not isinstance(value, Decimal):
                raise InvariantViolation(f"{name} must be Decimal, got {type(value).__name__}")
            if not value.is_finite():
                raise InvariantViolation(f"{name} must be finite, got {value}")
        if self.equity <= 0:
            raise InvariantViolation(
                f"equity must be positive, got {self.equity}; every budgeted limit "
                "is a fraction of it"
            )
        if self.turnover_today < 0:
            raise InvariantViolation(
                f"turnover_today must not be negative, got {self.turnover_today}"
            )
        if not self.drawdown_pct.is_finite():
            raise InvariantViolation(f"drawdown_pct must be finite, got {self.drawdown_pct}")
        if self.drawdown_pct < 0:
            raise InvariantViolation(
                f"drawdown_pct must not be negative, got {self.drawdown_pct}"
            )
        if self.orders_today < 0:
            raise InvariantViolation(f"orders_today must not be negative, got {self.orders_today}")

    def shares_of(self, symbol: Ticker) -> Decimal:
        return sum(
            (h.shares for h in self.holdings if h.symbol == symbol), Decimal(0)
        )

    def notional_of(self, symbol: Ticker) -> Decimal:
        """What one position is worth at the Gate's own marks.

        Falls back to the broker's `market_value` when no mark is available,
        rather than to zero: a position valued at nothing would pass a
        concentration check it should fail.
        """
        price = self.marks.get(symbol)
        if price is not None:
            return self.shares_of(symbol) * price
        return sum((h.market_value for h in self.holdings if h.symbol == symbol), Decimal(0))

    @property
    def gross_notional(self) -> Decimal:
        """Total long exposure, at the Gate's marks."""
        return sum(
            (self.notional_of(symbol) for symbol in self._symbols()), Decimal(0)
        )

    @property
    def gross_exposure(self) -> Decimal:
        """Gross as a fraction of equity. 1.0 is fully invested."""
        return self.gross_notional / self.equity

    @property
    def positions(self) -> int:
        return len(self._symbols())

    def _symbols(self) -> Sequence[Ticker]:
        """Sorted and de-duplicated, so anything summing over it is deterministic."""
        return sorted({h.symbol for h in self.holdings if h.shares != 0})
