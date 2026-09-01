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

__all__ = ["DEFAULT_LIMITS", "OPTIONS_SLEEVE_ALLOCATION", "SLEEVE_LIMITS", "RiskLimits"]

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
"""The whole-account configuration, kept as the reference point.

Every fraction here is a fraction of a $100,000 account. It is retained because
the backtest and the existing tests are calibrated against it, and because
`SLEEVE_LIMITS` below is best read as a diff against it.
"""


OPTIONS_SLEEVE_ALLOCATION: Final = Decimal(10_000)
"""The capital assigned to the options agent — 10% of a $100,000 account.

A fixed figure, not a fraction of live equity, for the reason `risk.sleeve`
gives: a fraction would let the equity book's overnight mark resize the options
agent's budgets. The operator splits the account once and the split holds.

**Raised from $5,000 once the researched rule was priced against a real chain.**
specs/07 D1's rule sells a 0.16-delta SPY put against a 0.08-delta wing, and on
2026-08-28 that is a 15-point spread: $1,389 of maximum loss per contract. At a
$5,000 sleeve the per-trade budget was $1,000, `agent/sizing.py` floored the
quantity to **zero**, and the rule could never have opened a position — which
would have looked like a quiet market in the journal rather than like an account
too small for its own strategy.

$10,000 is not a round number chosen for comfort. At `max_trade_loss_pct` of
0.20 it makes the per-trade budget $2,000, which is *exactly* the sizing the
research ran at (2% of the $100,000 the backtest and the sealed run were sized
against, specs/10 D8a). The sleeve is the smallest one that executes the
measured rule rather than a cheaper approximation of it.
"""


SLEEVE_LIMITS: Final = RiskLimits(
    # $2,000 a trade on the $10,000 sleeve, and the fraction is unchanged from
    # when the sleeve was $5,000 -- the budget moved because the base moved, not
    # because this control was loosened.
    #
    # $2,000 is the number that matters: it is what specs/10 D8a's 2% of
    # $100,000 came to, so the live per-trade size is the size the sealed run
    # measured. It funds exactly one contract of the researched structure
    # ($1,389 max loss on 2026-08-28) and refuses a second, which is the
    # research's own cadence of one entry per session.
    max_trade_loss_pct=Decimal("0.20"),
    # $8,000 of the $10,000 may be at risk at once. For defined-risk structures
    # maximum loss *is* the capital committed -- Alpaca holds exactly that as
    # buying power -- so this is the sleeve's deployment ceiling, not merely a
    # risk cap. 0.80 rather than 1.00 leaves room for the last trade to be
    # refused by a budget rather than by a broker rejection, which journals a
    # reason instead of an exception.
    max_portfolio_loss_pct=Decimal("0.80"),
    max_open_structures=8,
    # Equal to the heat cap, because specs/07 D2 gives this strategy one
    # underlying. With a single-name universe this check and `portfolio_heat`
    # measure the same quantity, so anything tighter is not a concentration
    # limit -- it is a second, lower heat cap wearing a concentration limit's
    # name, and it would silently cap the sleeve at a quarter of what it was
    # allocated while `max_portfolio_loss_pct` read as the binding number.
    #
    # Set equal rather than removed: the universe is configuration, and on the
    # day a second underlying is added this check must already be present and
    # already correct rather than needing to be remembered.
    max_per_underlying_pct=Decimal("0.80"),
    # Quoted per $1k, so this is +/- 12.0 net delta on the $10,000 sleeve and was
    # +/- 6.0 on the $5,000 one. Stating it as a rate rather than as an absolute
    # is what lets the sleeve be resized without silently changing which greek
    # binds -- the reason the vega band below is written the same way.
    #
    # The account-scaled band was +/- 30, which on a sleeve this size is not a
    # band at all: 30 delta is $19,500 of SPY notional.
    #
    # **This is the number most likely to need a live adjustment.** It is the
    # only limit here whose calibration comes from arithmetic rather than from
    # an observed run, and the failure it produces is quiet -- candidates are
    # dropped before the model sees them (`agent/candidates.py`), so a band set
    # too tight looks like a market with no setups rather than like a
    # misconfiguration. `preflight` prints it for that reason.
    delta_band=(-1.20, 1.20),
    # +/- 250 vega. Non-binding for one-wide verticals either way; kept at the
    # same per-$1k rate so that a change of sleeve size does not silently
    # change which greek is the binding one.
    vega_band=(-50.0, 50.0),
    max_spread_pct=Decimal("0.05"),
    dte_range=(3, 21),
    # A fifth of the sleeve -- $2,000. Under the account-scaled 5% this switch
    # measured the *account* and was therefore tripped by the equity book: an 8%
    # market move against a $90,000 stock sleeve is a 5% account drawdown, and
    # the options agent -- having lost nothing -- latched shut until a human
    # cleared it. Measured against the sleeve, 5% would be $500, which one
    # spread reaching its stop can produce; that is a trade going wrong, not a
    # strategy failing. 20% is four such trades in a row and a real signal that
    # something is not working.
    max_drawdown_pct=Decimal("0.20"),
    max_daily_trades=15,
    max_quote_age=60.0,
)
"""The options sleeve's configuration — specs/03 D6.

Read as a diff against `DEFAULT_LIMITS`: the base is `OPTIONS_SLEEVE_ALLOCATION`
rather than account equity, so every fraction here is a fraction of $10,000. Two
of the absolute figures are held where they were (per-trade, and the greek rate)
and three are deliberately moved (heat, concentration, drawdown). Each comment
above says which and why.
"""
