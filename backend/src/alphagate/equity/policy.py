"""Executor policy — the numbers that are ours, not the strategy's.

PURE. Stdlib plus `alphagate.core`.

Everything here is a decision about *placing* a book, never about *choosing*
one. The strategy is what `ai_quant_researcher` validated; the band, the floor
and the caps below are the cost of turning its weights into orders, and keeping
them in a separate type from anything the researcher produced is what stops the
two from being confused.

Budgeted limits are fractions of equity for the same reason
[03](../../../../specs/03-risk-gate.md) D5 gives: a dollar cap written for a
$100k paper account silently becomes a different policy on a $30k one.

`DEFAULT_EQUITY_POLICY` is the competition configuration. One module, logged at
startup, rendered in the dashboard — a limit nobody can read is a limit nobody
is enforcing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal
from typing import Final

from alphagate.core.errors import InvariantViolation

__all__ = ["DEFAULT_EQUITY_POLICY", "EquityPolicy"]


@dataclass(frozen=True, slots=True)
class EquityPolicy:
    """One immutable configuration. Never mutated; replaced wholesale."""

    no_trade_band_pct: Decimal
    """Minimum drift, as a fraction of equity, before a symbol is traded.

    specs/09 D3. The strategy rebalances every five sessions; between them the
    targets do not move but the holdings drift with price, so a naive diff would
    trade every name every day and pay costs no backtest ever charged."""

    min_order_notional: Decimal
    """Absolute floor on one order, in account currency.

    A band expressed only as a fraction of equity still admits a $3 order on a
    small account, and a $3 order is all spread."""

    max_position_pct: Decimal
    """Ceiling on one resulting position. The book's own largest core weight is
    8%, so this is a guard against a malformed book rather than a constraint on
    a healthy one — which is exactly what a Gate check should be."""

    max_gross: Decimal
    """Total exposure ceiling. 1.0 is fully invested; anything above is
    leverage, and no run upstream measured leverage."""

    max_daily_turnover_pct: Decimal
    """Traded notional in one session, as a fraction of equity. The first
    session builds the whole book, so this is generous by necessity; what it
    stops is a loop that rebuilds it repeatedly."""

    max_daily_orders: int
    """The book holds 104 names, so a full build is 104 orders. This is a
    circuit breaker on a runaway plan, not a limit on a rebalance."""

    max_book_age_days: int
    """Calendar days between the book's `as_of` and today. A book from last
    month describes a portfolio the strategy no longer holds."""

    max_quote_age: float
    """Seconds. Inclusive: a quote exactly this old is still fresh. `float`
    because it is a duration, not money."""

    max_drawdown_pct: Decimal
    """Peak-to-current, and the kill switch's threshold. Latched by the caller —
    a pure policy cannot remember that it tripped yesterday."""

    fractional_places: int
    """Decimal places for a fractional share quantity. Alpaca accepts nine;
    four is well inside that and keeps the journal readable."""

    def __post_init__(self) -> None:
        for name in (
            "no_trade_band_pct",
            "min_order_notional",
            "max_position_pct",
            "max_gross",
            "max_daily_turnover_pct",
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
        for name in ("max_daily_orders", "max_book_age_days", "fractional_places"):
            count = getattr(self, name)
            if count <= 0:
                raise InvariantViolation(f"{name} must be positive, got {count}")
        # NaN fails every comparison silently, so ask explicitly rather than
        # letting a NaN limit disable the check it configures.
        if not math.isfinite(self.max_quote_age):
            raise InvariantViolation(f"max_quote_age must be finite, got {self.max_quote_age}")
        if self.max_quote_age <= 0:
            raise InvariantViolation(f"max_quote_age must be positive, got {self.max_quote_age}")
        if self.fractional_places > 9:
            raise InvariantViolation(
                f"fractional_places is {self.fractional_places}; Alpaca accepts nine"
            )

    # --------------------------------------------------------------- #
    # Absolute limits, derived from the equity they are a fraction of
    # --------------------------------------------------------------- #

    def band(self, equity: Decimal) -> Decimal:
        """The drift a symbol must exceed before it is traded, in currency.

        The larger of the two thresholds, so both bind: the fractional band on a
        large account and the absolute floor on a small one.
        """
        return max(self.no_trade_band_pct * equity, self.min_order_notional)

    def max_position(self, equity: Decimal) -> Decimal:
        return self.max_position_pct * equity

    def max_daily_turnover(self, equity: Decimal) -> Decimal:
        return self.max_daily_turnover_pct * equity


DEFAULT_EQUITY_POLICY: Final = EquityPolicy(
    # 0.25% of a $100k account is $250. Below that on a 104-name book the order
    # is smaller than a single sleeve position, so trading it is churn.
    no_trade_band_pct=Decimal("0.0025"),
    min_order_notional=Decimal(25),
    # Comfortably above the book's 8.2% largest name and far below anything that
    # would read as a concentrated bet.
    max_position_pct=Decimal("0.15"),
    max_gross=Decimal("1.00"),
    # A first-session build is one turnover of the whole account. 1.2 leaves
    # room for that plus a rebalance, and stops a third pass.
    max_daily_turnover_pct=Decimal("1.20"),
    max_daily_orders=250,
    # Five sessions is the rebalance period; seven calendar days covers it
    # across a weekend without admitting a book from the session before last.
    max_book_age_days=7,
    max_quote_age=90.0,
    max_drawdown_pct=Decimal("0.10"),
    fractional_places=4,
)
