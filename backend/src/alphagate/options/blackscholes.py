"""Black–Scholes pricing and implied-volatility inversion.

PURE. Stdlib `math` only. No clock, no I/O, no provider types.

This exists for one reason: **to reconstruct an implied-volatility history**.
Alpaca serves current greeks but no historical IV, and `iv_rank` (specs/05 D2)
is meaningless without a trailing series to rank against. Historical *option
prices* are available, so the series is derived by inverting the model that
produced them.

Everything here is `float`, deliberately, and it is the one place in the system
where money touches floats. Prices go in as floats and volatilities come out as
floats because both are inputs to and outputs of a *model* — an implied
volatility is not a quantity of money, it is a parameter fitted to one. The
conversion happens at this boundary and nowhere else; `Decimal` prices are
converted on the way in by the caller, and nothing here returns a price that
becomes money again.

**Inversion is by bisection on a bracketed interval, not by Newton's method.**
Newton from a bad starting guess wanders off deep in the wings, where vega is
nearly zero and the derivative is meaningless; a bisection that has already
bracketed a root cannot. Price is monotone in volatility, so bracketing is
sufficient and the iteration count is bounded by construction. This runs offline
over a few hundred bars, so the extra iterations cost nothing worth having.

An option priced below its intrinsic value, or above its theoretical maximum,
has **no** implied volatility. That returns `None` rather than a clamped
boundary value: a stale or crossed print is a fact about the print, and
recording it as "IV = 0.01" would put a fabricated point into the very series
`iv_rank` is computed from.
"""

from __future__ import annotations

import math
from typing import Final

__all__ = [
    "MAX_VOL",
    "MIN_VOL",
    "call_price",
    "implied_volatility",
    "put_price",
    "time_to_expiry_years",
]

MIN_VOL: Final = 0.001
"""0.1%. Below this the price is flat in vol and the number is noise."""
MAX_VOL: Final = 5.0
"""500%. Above this we are fitting a model to something that is not an option."""

_TOLERANCE: Final = 1e-8
_MIN_EXTRINSIC: Final = 1e-6
"""Extrinsic value, as a fraction of the larger of spot and strike, below which
a price carries no information about volatility.

Found by a failing test rather than by reasoning ahead: a deep in-the-money call
at 5% volatility has a time value of about 1e-12, and an absolute convergence
tolerance of 1e-8 "converges" on the first midpoint it tries. The function then
returns a confidently wrong number — the worst possible outcome for a series
that an IV rank is computed from. On a $766 underlying this threshold is about
eight hundredths of a cent, far below any real quote, so it excludes only the
prices that were never going to answer."""

_MAX_ITERATIONS: Final = 100
TRADING_DAYS: Final = 252.0
CALENDAR_DAYS: Final = 365.0


def time_to_expiry_years(days: int) -> float:
    """Calendar days to expiry, in years.

    Calendar rather than trading days, to match `OptionContract.days_to_expiry`
    and because time value decays over weekends. Using a 252-day year here and a
    calendar-day count there would put a systematic ~1.4× error into every
    inverted volatility — small enough to look plausible and large enough to
    reorder an IV rank.
    """
    return max(days, 0) / CALENDAR_DAYS


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _d1_d2(
    spot: float, strike: float, vol: float, years: float, rate: float
) -> tuple[float, float]:
    scaled = vol * math.sqrt(years)
    d1 = (math.log(spot / strike) + (rate + 0.5 * vol * vol) * years) / scaled
    return d1, d1 - scaled


def call_price(spot: float, strike: float, vol: float, years: float, rate: float = 0.0) -> float:
    """European call, no dividend. Degenerate inputs return intrinsic value."""
    if years <= 0 or vol <= 0 or spot <= 0 or strike <= 0:
        return max(spot - strike * math.exp(-rate * years), 0.0)
    d1, d2 = _d1_d2(spot, strike, vol, years, rate)
    return spot * _norm_cdf(d1) - strike * math.exp(-rate * years) * _norm_cdf(d2)


def put_price(spot: float, strike: float, vol: float, years: float, rate: float = 0.0) -> float:
    """European put, by put–call parity off `call_price`.

    Derived rather than written out so the two can never drift apart: a parity
    violation between two independently coded formulas is a bug that only shows
    up as a skew that is not there.
    """
    if years <= 0 or vol <= 0 or spot <= 0 or strike <= 0:
        return max(strike * math.exp(-rate * years) - spot, 0.0)
    return call_price(spot, strike, vol, years, rate) - spot + strike * math.exp(-rate * years)


def implied_volatility(
    price: float,
    spot: float,
    strike: float,
    years: float,
    *,
    is_call: bool,
    rate: float = 0.0,
) -> float | None:
    """Invert Black–Scholes for volatility. `None` when there is no root.

    Returns `None` — never a clamped bound — when the price is at or below
    intrinsic, at or above the theoretical maximum, or the option has expired.
    Each of those is a real observation about a bad print, and a fabricated
    volatility would enter the series that `iv_rank` ranks against.
    """
    if price <= 0 or spot <= 0 or strike <= 0 or years <= 0:
        return None

    discount = math.exp(-rate * years)
    intrinsic = (
        max(spot - strike * discount, 0.0)
        if is_call
        else max(strike * discount - spot, 0.0)
    )
    ceiling = spot if is_call else strike * discount
    if price <= intrinsic or price >= ceiling:
        return None

    # Deep in or out of the money, the price is flat in volatility: all of it is
    # intrinsic, and the little that is not is below the resolution of any
    # root-finder. Refusing is the honest answer; converging is not.
    if price - intrinsic < _MIN_EXTRINSIC * max(spot, strike):
        return None

    price_at = call_price if is_call else put_price

    def error(vol: float) -> float:
        return price_at(spot, strike, vol, years, rate) - price

    low, high = MIN_VOL, MAX_VOL
    error_low, error_high = error(low), error(high)
    if error_low > 0 or error_high < 0:
        # The root is outside the interval we are prepared to believe in.
        return None

    # Bisection. Monotone in vol, so bracketing is enough and the iteration
    # count is bounded by construction rather than by hope.
    # Relative to the price being matched, not absolute: 1e-8 of a $200 option
    # and 1e-8 of a $0.05 option are different questions.
    tolerance = max(_TOLERANCE, price * _TOLERANCE)
    for _ in range(_MAX_ITERATIONS):
        mid = 0.5 * (low + high)
        value = error(mid)
        if abs(value) < tolerance or (high - low) < _TOLERANCE:
            return mid
        if value < 0:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)
