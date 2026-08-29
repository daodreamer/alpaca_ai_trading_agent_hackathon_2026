"""IV rank, IV/HV, and realised volatility — specs/05 D2's headline input.

PURE. Stdlib plus `alphagate.core`. No clock, no I/O.

`iv_rank` is *where the current implied volatility sits inside its own trailing
range*, not the level of implied volatility. The distinction is the whole point
and it is easy to get wrong: SPY at 15% IV can be at the 90th percentile of its
own year or the 10th, and those are opposite trades. The first version of the
live smoke test handed the model a mean IV of 15.79 labelled `iv_rank`, and the
model dutifully reasoned that "IV rank is low (15.79)" — a wrong answer produced
faithfully from a wrong input. This module exists so that cannot happen again.

**Insufficient history returns `None`, never a default.** A rank computed from
four observations is not a rank; it is noise wearing a percentile's clothing.
`MarketRead.iv_rank` is therefore optional, and the strategy screen refuses to
produce a setup without it. That is the same discipline as specs/02 D2's missing
greeks and specs/05's unmeasured trend: absence is representable, and never
reads as a middling value.

Two definitions are provided because both are in common use and they disagree:

* **rank** (`iv_rank`) — the classic min/max position: `(iv - min) / (max - min)`.
  Sensitive to a single outlier at either end.
* **percentile** (`iv_percentile`) — the fraction of the window below the current
  value. Robust, and usually the more informative of the two.

Both are returned, both are journalled, and the strategy says which it uses.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Final

from alphagate.core.errors import InvariantViolation

__all__ = [
    "MIN_HISTORY",
    "VolatilityRead",
    "iv_percentile",
    "iv_rank",
    "iv_vs_hv",
    "realised_volatility",
    "summarise_volatility",
]

MIN_HISTORY: Final = 20
"""Trading days of IV history before a rank means anything.

Twenty is a month of sessions: enough that one wild print cannot define the
range, few enough to be reachable inside a competition week from reconstructed
history. It is a floor, not a target — a year is better and the store keeps
whatever it is given."""

_ANNUALISATION: Final = 252.0


def realised_volatility(closes: Sequence[Decimal], *, window: int = 20) -> float | None:
    """Annualised close-to-close volatility over the last `window` returns.

    `None` when there are not enough closes. Log returns, sample standard
    deviation (`n-1`), annualised by trading days — the convention that makes the
    result comparable with an implied volatility, which is what `iv_vs_hv` needs
    it for.
    """
    if window < 2 or len(closes) < window + 1:
        return None
    recent = [float(close) for close in closes[-(window + 1) :]]
    if any(price <= 0 for price in recent):
        raise InvariantViolation("close prices must be positive to take log returns")

    returns = [math.log(recent[i] / recent[i - 1]) for i in range(1, len(recent))]
    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    return math.sqrt(variance) * math.sqrt(_ANNUALISATION)


def iv_rank(
    current: float, history: Sequence[float], *, minimum: int = MIN_HISTORY
) -> float | None:
    """Where `current` sits between the low and the high of `history`, in [0, 1].

    `None` when the history is too short, or when it is entirely flat — a range
    of zero has no inside, and dividing by it would produce a number rather than
    an answer.
    """
    usable = _usable(history, minimum)
    if usable is None:
        return None
    low, high = min(usable), max(usable)
    if high - low <= 0:
        return None
    return max(0.0, min(1.0, (current - low) / (high - low)))


def iv_percentile(
    current: float, history: Sequence[float], *, minimum: int = MIN_HISTORY
) -> float | None:
    """The fraction of `history` strictly below `current`, in [0, 1].

    Robust where `iv_rank` is not: one panic spike three months ago moves the
    rank a long way and the percentile barely at all.
    """
    usable = _usable(history, minimum)
    if usable is None:
        return None
    return sum(1 for value in usable if value < current) / len(usable)


def iv_vs_hv(implied: float, realised: float | None) -> float | None:
    """Implied over realised. Above 1 means options are pricing more movement
    than the underlying has recently delivered.

    `None` propagates: an unknown realised volatility makes the ratio unknown,
    not 1.0. Reporting 1.0 would say "options are fairly priced", which is a
    claim, not a missing value.
    """
    if realised is None or realised <= 0 or implied <= 0:
        return None
    return implied / realised


def _usable(history: Sequence[float], minimum: int) -> list[float] | None:
    finite = [value for value in history if math.isfinite(value) and value > 0]
    if len(finite) < max(minimum, 2):
        return None
    return finite


@dataclass(frozen=True, slots=True)
class VolatilityRead:
    """Everything the market read says about volatility, and what it could not.

    Every field is optional and each `None` is a distinct, recorded fact. A read
    that could not compute a rank is different from one that computed a low
    rank, and the journal must be able to tell a judge which.
    """

    implied: float | None
    realised: float | None
    rank: float | None
    percentile: float | None
    ratio: float | None
    observations: int
    """How many historical points the rank was computed from. Recorded so a
    thin history is visible rather than merely reflected in a shaky number."""

    @property
    def is_complete(self) -> bool:
        return self.implied is not None and self.rank is not None

    def as_decimal(self, value: float | None, places: str = "0.0001") -> Decimal | None:
        if value is None:
            return None
        return Decimal(str(value)).quantize(Decimal(places))


def summarise_volatility(
    implied: float | None,
    closes: Sequence[Decimal],
    history: Sequence[float],
    *,
    hv_window: int = 20,
    minimum: int = MIN_HISTORY,
) -> VolatilityRead:
    """Build the whole volatility picture from what is actually available."""
    realised = realised_volatility(closes, window=hv_window)
    if implied is None:
        return VolatilityRead(None, realised, None, None, None, len(history))
    return VolatilityRead(
        implied=implied,
        realised=realised,
        rank=iv_rank(implied, history, minimum=minimum),
        percentile=iv_percentile(implied, history, minimum=minimum),
        ratio=iv_vs_hv(implied, realised),
        observations=len(history),
    )
