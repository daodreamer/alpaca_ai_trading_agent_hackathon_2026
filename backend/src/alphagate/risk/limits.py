"""The configuration the Gate judges against — specs/03 D5.

PURE. Stdlib only.

Every budgeted limit is expressed as a **fraction of equity**, not an absolute
figure. A dollar cap written for a $100k paper account silently becomes a
different strategy on a $30k one, and the competition account size is not the
thing this system is claiming to be robust to.

Greek bands are expressed **per $1,000 of equity** and scaled the same way, for
the same reason. They are `float` because greeks are `float` (specs/01 Rule 3):
a delta band is a tolerance on an estimate, not an amount of money.

`DEFAULT_LIMITS` is the competition configuration. It lives in one module, is
logged at startup, and is rendered in the dashboard — a risk limit nobody can
read is a risk limit nobody is enforcing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal
from typing import Final

from alphagate.core.errors import InvariantViolation

__all__ = ["DEFAULT_LIMITS", "RiskLimits"]

_EQUITY_UNIT: Final = Decimal(1000)
"""Greek bands are quoted per $1k of equity."""


@dataclass(frozen=True, slots=True)
class RiskLimits:
    """One immutable configuration. Never mutated; replaced wholesale."""

    max_trade_loss_pct: Decimal
    max_portfolio_loss_pct: Decimal
    max_open_structures: int
    max_per_underlying_pct: Decimal
    delta_band: tuple[float, float]
    """Inclusive band per $1k of equity."""
    vega_band: tuple[float, float]
    max_spread_pct: Decimal
    dte_range: tuple[int, int]
    """Inclusive on both ends. Lower bound excludes 0DTE — specs/03 D5."""
    max_drawdown_pct: Decimal
    max_daily_trades: int
    max_quote_age: float
    """Seconds. Inclusive: a quote exactly this old is still fresh."""

    def __post_init__(self) -> None:
        for name in (
            "max_trade_loss_pct",
            "max_portfolio_loss_pct",
            "max_per_underlying_pct",
            "max_spread_pct",
            "max_drawdown_pct",
        ):
            value = getattr(self, name)
            if not isinstance(value, Decimal):
                raise InvariantViolation(f"{name} must be Decimal, got {type(value).__name__}")
            if not value.is_finite():
                raise InvariantViolation(f"{name} must be finite, got {value}")
            if value <= 0:
                raise InvariantViolation(
                    f"{name} must be positive, got {value}; a limit of zero is not a "
                    "conservative limit, it is a disabled system"
                )
        for name in ("max_open_structures", "max_daily_trades"):
            count = getattr(self, name)
            if count <= 0:
                raise InvariantViolation(f"{name} must be positive, got {count}")
        # Floats fail the other way round: `nan <= 0` is False and `nan > x` is
        # False, so a NaN limit would sail through every comparison below and
        # then silently disable the check it configures. Ask explicitly.
        if not math.isfinite(self.max_quote_age):
            raise InvariantViolation(f"max_quote_age must be finite, got {self.max_quote_age}")
        if self.max_quote_age <= 0:
            raise InvariantViolation(f"max_quote_age must be positive, got {self.max_quote_age}")
        for name in ("delta_band", "vega_band"):
            low, high = getattr(self, name)
            if not (math.isfinite(low) and math.isfinite(high)):
                raise InvariantViolation(f"{name} bounds must be finite, got ({low}, {high})")
            if low > high:
                raise InvariantViolation(f"{name} is inverted: ({low}, {high})")
        low_dte, high_dte = self.dte_range
        if low_dte < 0:
            raise InvariantViolation(f"dte_range lower bound must not be negative: {low_dte}")
        if low_dte > high_dte:
            raise InvariantViolation(f"dte_range is inverted: {self.dte_range}")

    # --------------------------------------------------------------- #
    # Absolute limits, derived from the equity they are a fraction of
    # --------------------------------------------------------------- #

    def max_trade_loss(self, equity: Decimal) -> Decimal:
        return self.max_trade_loss_pct * equity

    def max_portfolio_loss(self, equity: Decimal) -> Decimal:
        return self.max_portfolio_loss_pct * equity

    def max_per_underlying(self, equity: Decimal) -> Decimal:
        return self.max_per_underlying_pct * equity

    def scaled_delta_band(self, equity: Decimal) -> tuple[float, float]:
        return self._scaled(self.delta_band, equity)

    def scaled_vega_band(self, equity: Decimal) -> tuple[float, float]:
        return self._scaled(self.vega_band, equity)

    @staticmethod
    def _scaled(band: tuple[float, float], equity: Decimal) -> tuple[float, float]:
        units = float(equity / _EQUITY_UNIT)
        low, high = band
        return (low * units, high * units)


DEFAULT_LIMITS: Final = RiskLimits(
    max_trade_loss_pct=Decimal("0.01"),
    max_portfolio_loss_pct=Decimal("0.05"),
    max_open_structures=8,
    max_per_underlying_pct=Decimal("0.02"),
    delta_band=(-0.30, 0.30),
    vega_band=(-50.0, 50.0),
    max_spread_pct=Decimal("0.05"),
    # No 0DTE: over a four-day scored window it turns P&L into a coin flip.
    # Capped at 21: a 45-DTE position cannot round-trip inside the window and
    # would be graded purely on mark-to-market. specs/03 D5, specs/07 D5.
    dte_range=(3, 21),
    max_drawdown_pct=Decimal("0.05"),
    max_daily_trades=15,
    max_quote_age=60.0,
)
