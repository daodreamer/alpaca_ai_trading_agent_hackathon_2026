"""Black-Scholes — specs/10-options-research.md D1a.

**This model never moves money.** Entry prices are the vendor's bid and ask;
settlement is the underlying's close on the expiration date. Both are quoted
numbers. What lives here produces the equity curve *between* those two events,
and nothing else in the system may call it for any other purpose.

Why it exists at all, given that D1 refuses modelled prices in the exit: a book
marked only on its cash flows has a beta of zero and no drawdown between entry
and expiry, by construction. Handing that to
[`alpha.py`](../backtest/alpha.py) would credit the whole return of a short put
spread to alpha, which is precisely the failure that module was written to
prevent. Cash accounting looks conservative and is not; it is wrong in the
direction that flatters the strategy.

Two deliberate simplifications, both named in D1a rather than buried:

**Zero rates by default.** Over 2019–2024 the front-end rate went from 2.4% to
0 to 5.3%, so any single constant is wrong somewhere. It is a parameter, it
defaults to zero, and it moves a 28-day mark by cents.

**No dividends, European exercise.** SPY pays about 1.3% a year and its options
are American. The bias is toward optimism on short ITM legs, and it is recorded
with every result rather than corrected by a second model.
"""

from __future__ import annotations

import math
from typing import Literal

__all__ = ["black_scholes", "intrinsic"]

Right = Literal["call", "put"]


def intrinsic(right: Right, *, spot: float, strike: float) -> float:
    """What the contract is worth if it expired right now."""
    return max(spot - strike, 0.0) if right == "call" else max(strike - spot, 0.0)


def _normal_cdf(x: float) -> float:
    """Φ(x) from the standard library, so the price is bit-identical everywhere.

    An approximation polynomial would make two machines disagree in the sixth
    decimal, and a determinism test that compares equity curves exactly would
    fail for a reason that has nothing to do with the strategy.
    """
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def black_scholes(
    right: Right,
    *,
    spot: float,
    strike: float,
    years: float,
    iv: float,
    rate: float = 0.0,
) -> float:
    """Theoretical value of one contract, per share.

    ``years`` is time to expiry in years; ``iv`` the annualised volatility as a
    decimal. Multiply by 100 for a contract's value — the multiplier belongs to
    the position, not to the price.
    """
    if years < 0:
        raise ValueError(f"years must be >= 0, got {years}; a mark dated after expiry is a bug")
    if spot <= 0 or strike <= 0:
        raise ValueError(f"spot and strike must be positive, got {spot} and {strike}")

    # At expiry, or with no volatility, the option *is* its intrinsic value.
    # Not a special case bolted on: sigma*sqrt(T) is the denominator of d1, and
    # the limit of the formula as it goes to zero is exactly this.
    if years == 0.0 or iv <= 0.0:
        discounted = strike * math.exp(-rate * years)
        return (
            max(spot - discounted, 0.0) if right == "call" else max(discounted - spot, 0.0)
        )

    sigma_root_t = iv * math.sqrt(years)
    d1 = (math.log(spot / strike) + (rate + 0.5 * iv * iv) * years) / sigma_root_t
    d2 = d1 - sigma_root_t
    discounted = strike * math.exp(-rate * years)
    if right == "call":
        return spot * _normal_cdf(d1) - discounted * _normal_cdf(d2)
    return discounted * _normal_cdf(-d2) - spot * _normal_cdf(-d1)
