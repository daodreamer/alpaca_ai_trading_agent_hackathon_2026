"""The book as the Gate sees it — specs/03 D1.

PURE. Stdlib plus `alphagate.core` and `alphagate.options`.

This is a *snapshot*, not a live account object: it is assembled by the layer
above from a reconciled Alpaca read and then handed to `evaluate` as a value.
The Gate never queries it, never refreshes it, and never reads a clock to decide
how old it is. Everything the Gate needs is either a field here or an argument.

**The kill switch latch lives here, not in the Gate.** A pure function cannot
remember that it tripped yesterday. So the tripped flag is persisted state that
the caller carries in, and `drawdown_killswitch` vetoes on *either* the live
drawdown crossing the limit or the latch already being set. Re-arming is a
deliberate act by a human that clears the flag before the next snapshot is
built — specs/03 D4.

**Missing greeks poison the aggregate, exactly as in specs/02 D4.** If any open
position's exposure is unknown, the portfolio's net delta is `None`, never a
partial sum. A partial sum understates the book by precisely the positions we
know least about.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from alphagate.core.errors import InvariantViolation
from alphagate.core.identifiers import Ticker
from alphagate.options.quote import Greeks
from alphagate.options.structure import OptionStructure

__all__ = ["OpenPosition", "PortfolioSnapshot"]


@dataclass(frozen=True, slots=True)
class OpenPosition:
    """One live structure and the risk still on the table for it."""

    structure: OptionStructure
    quantity: int
    max_loss: Decimal
    """Remaining maximum loss for the whole position, in account currency."""
    net_greeks: Greeks | None
    opened_at: datetime

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise InvariantViolation(f"position quantity must be positive, got {self.quantity}")
        if not isinstance(self.max_loss, Decimal):
            raise InvariantViolation(
                f"max_loss must be Decimal, got {type(self.max_loss).__name__}"
            )
        # Finiteness first: `Decimal("NaN") <= 0` raises InvalidOperation rather
        # than answering, and an invariant check must return a verdict, not a
        # different exception than the one the caller is told to expect.
        if not self.max_loss.is_finite() or self.max_loss <= 0:
            raise InvariantViolation(
                f"open position max_loss must be finite and positive, got {self.max_loss}"
            )
        if self.opened_at.tzinfo is None:
            raise InvariantViolation(f"opened_at must be tz-aware, got {self.opened_at!r}")

    @property
    def underlying(self) -> Ticker:
        return self.structure.underlying


@dataclass(frozen=True, slots=True)
class PortfolioSnapshot:
    """Everything about the account the Gate is allowed to know."""

    equity: Decimal
    positions: tuple[OpenPosition, ...]
    drawdown_pct: Decimal
    """Peak-to-current drawdown as a fraction, e.g. ``Decimal("0.03")`` for 3%."""
    fills_today: int
    killswitch_tripped: bool = False
    """Latched by the caller. Cleared only by a human re-arming it."""

    def __post_init__(self) -> None:
        if not isinstance(self.equity, Decimal):
            raise InvariantViolation(f"equity must be Decimal, got {type(self.equity).__name__}")
        if not self.equity.is_finite():
            raise InvariantViolation(f"equity must be finite, got {self.equity}")
        if self.equity <= 0:
            raise InvariantViolation(
                f"equity must be positive, got {self.equity}; "
                "every budgeted limit is a fraction of it"
            )
        if not self.drawdown_pct.is_finite():
            raise InvariantViolation(f"drawdown_pct must be finite, got {self.drawdown_pct}")
        if self.drawdown_pct < 0:
            raise InvariantViolation(f"drawdown_pct must not be negative, got {self.drawdown_pct}")
        if self.fills_today < 0:
            raise InvariantViolation(f"fills_today must not be negative, got {self.fills_today}")

    @property
    def open_structures(self) -> int:
        return len(self.positions)

    @property
    def open_risk(self) -> Decimal:
        """Total maximum loss across every open position."""
        return sum((position.max_loss for position in self.positions), Decimal(0))

    def exposure_to(self, underlying: Ticker) -> Decimal:
        """Maximum loss concentrated in one underlying."""
        return sum(
            (p.max_loss for p in self.positions if p.underlying == underlying), Decimal(0)
        )

    @property
    def net_delta(self) -> float | None:
        """Portfolio delta, or `None` if any position's exposure is unknown."""
        return self._net("delta")

    @property
    def net_vega(self) -> float | None:
        return self._net("vega")

    def _net(self, greek: str) -> float | None:
        total = 0.0
        for position in self.positions:
            if position.net_greeks is None:
                return None
            total += float(getattr(position.net_greeks, greek))
        return total
