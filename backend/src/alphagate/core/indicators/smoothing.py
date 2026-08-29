"""Recursive averaging kernels — ADR 0007 D5.

Two smoothers, deliberately separated from the indicators that use them, because
several indicators share one and *which* one they share is the thing ADR 0007
had to pin down:

* `EmaCore` — alpha = 2/(n+1). Used by EMA and by both MACD lines.
* `WilderCore` — alpha = 1/n, Wilder's RMA. Used by RSI and ATR, and the reason
  those two agree with every mainstream charting package.

Both are seeded with the simple mean of their first `n` inputs and emit nothing
before that, so no value is ever computed from a partially-filled window. They
take plain floats, not bars: MACD's signal line smooths the MACD line, which is
not a price.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from alphagate.core.indicators.base import RollingWindow, require_period

__all__ = ["EmaCore", "WilderCore"]


class _SeededSmoother(ABC):
    """Shared seeding: collect `period` values, average them, then recurse."""

    __slots__ = ("_seed", "_value", "period")

    def __init__(self, period: int) -> None:
        self.period = require_period(period)
        self._seed = RollingWindow(period)
        self._value: float | None = None

    def feed(self, value: float) -> float | None:
        """Advance by one input; return the smoothed value, or `None` while seeding."""
        if self._value is None:
            self._value = self._seed.push(value)
        else:
            self._value = self._step(self._value, value)
        return self._value

    @abstractmethod
    def _step(self, previous: float, value: float) -> float: ...


class EmaCore(_SeededSmoother):
    """Exponential moving average, alpha = 2/(period+1), SMA-seeded."""

    __slots__ = ("_alpha",)

    def __init__(self, period: int) -> None:
        super().__init__(period)
        self._alpha = 2.0 / (self.period + 1)

    def _step(self, previous: float, value: float) -> float:
        return self._alpha * value + (1.0 - self._alpha) * previous


class WilderCore(_SeededSmoother):
    """Wilder's smoothing (RMA): `(prev * (n-1) + x) / n`, mean-seeded.

    Equivalent to an EMA with alpha = 1/n. Written in Wilder's own form rather
    than as an alpha so that a reader comparing against the 1978 worked example
    sees the same expression.
    """

    __slots__ = ()

    def _step(self, previous: float, value: float) -> float:
        return (previous * (self.period - 1) + value) / self.period
