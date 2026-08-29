"""RSI and MACD — ADR 0007 D4/D5.

Contract for both:

* input — one price per bar, selected by `PriceSource` (default close);
* output — RSI: a bounded `float` in [0, 100]; MACD: a `MacdValue`, whose line
  is price-valued and whose signal/histogram may be `None` for longer than the
  line's own warmup;
* warmup — RSI `period + 1` (n changes need n+1 closes); MACD `slow` for the
  line and `slow + signal - 1` for the signal;
* missing data — bar-indexed; a gap is not interpolated;
* duplicates/out-of-order — refused by the base class (ADR 0007 D6);
* tolerance — rel 1e-9 internally, 1e-6 against a third-party implementation
  (ADR 0005 D7);
* causal — yes.
"""

from __future__ import annotations

import dataclasses

from alphagate.core.bar import Bar
from alphagate.core.errors import InvariantViolation
from alphagate.core.indicators.base import (
    IndicatorSpec,
    OnlineIndicator,
    PriceSource,
    require_period,
)
from alphagate.core.indicators.smoothing import EmaCore, WilderCore

__all__ = ["Macd", "MacdValue", "Rsi"]

NEUTRAL_RSI = 50.0
"""What RSI reads when nothing moved at all — see `_rsi`."""


def _rsi(average_gain: float, average_loss: float) -> float:
    """RSI from Wilder's two smoothed averages, including the degenerate cases.

    `100 - 100/(1 + gain/loss)` is undefined when there were no losses, and
    uninformative when there were neither gains nor losses. Both are real market
    states, not missing data, so both get a value rather than a `None`
    (ADR 0007 D5).
    """
    if average_loss == 0.0:
        return NEUTRAL_RSI if average_gain == 0.0 else 100.0
    return 100.0 - 100.0 / (1.0 + average_gain / average_loss)


class Rsi(OnlineIndicator[float]):
    """Relative Strength Index with Wilder's smoothing, default period 14."""

    def __init__(
        self,
        *,
        period: int = 14,
        source: PriceSource = PriceSource.CLOSE,
        allow_partial: bool = False,
    ) -> None:
        super().__init__(allow_partial=allow_partial)
        self.period = require_period(period)
        self.source = source
        self._gains = WilderCore(self.period)
        self._losses = WilderCore(self.period)
        self._previous: float | None = None

    @property
    def spec(self) -> IndicatorSpec:
        return IndicatorSpec("rsi", (("period", str(self.period)), ("source", self.source.value)))

    @property
    def warmup(self) -> int:
        return self.period + 1

    def _consume(self, bar: Bar) -> float | None:
        current = self.source.of(bar)
        previous, self._previous = self._previous, current
        if previous is None:
            return None

        change = current - previous
        average_gain = self._gains.feed(max(change, 0.0))
        average_loss = self._losses.feed(max(-change, 0.0))
        if average_gain is None or average_loss is None:
            return None
        return _rsi(average_gain, average_loss)


@dataclasses.dataclass(frozen=True, slots=True)
class MacdValue:
    """One MACD sample.

    `signal` and `histogram` are `None` between the line's warmup and the
    signal's: the line exists and is drawable, the crossover does not exist yet.
    A rule that needs the crossover must say so by checking (ADR 0007 D5).
    """

    macd: float
    signal: float | None = None
    histogram: float | None = None


class Macd(OnlineIndicator[MacdValue]):
    """Moving Average Convergence/Divergence, default 12/26/9.

    `macd = EMA(fast) - EMA(slow)`, `signal = EMA(signal_period)` of the MACD
    line itself, `histogram = macd - signal`.
    """

    def __init__(
        self,
        *,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
        source: PriceSource = PriceSource.CLOSE,
        allow_partial: bool = False,
    ) -> None:
        super().__init__(allow_partial=allow_partial)
        self.fast = require_period(fast, field="fast period")
        self.slow = require_period(slow, field="slow period")
        self.signal_period = require_period(signal, field="signal period")
        if self.fast >= self.slow:
            raise InvariantViolation(
                f"fast period ({self.fast}) must be shorter than the slow period "
                f"({self.slow}); otherwise the MACD line is inverted or always zero"
            )
        self.source = source
        self._fast = EmaCore(self.fast)
        self._slow = EmaCore(self.slow)
        self._signal = EmaCore(self.signal_period)

    @property
    def spec(self) -> IndicatorSpec:
        return IndicatorSpec(
            "macd",
            (
                ("fast", str(self.fast)),
                ("slow", str(self.slow)),
                ("signal", str(self.signal_period)),
                ("source", self.source.value),
            ),
        )

    @property
    def warmup(self) -> int:
        """Bars before the MACD line appears. The signal needs `signal_warmup`."""
        return self.slow

    @property
    def signal_warmup(self) -> int:
        return self.slow + self.signal_period - 1

    def _consume(self, bar: Bar) -> MacdValue | None:
        price = self.source.of(bar)
        fast = self._fast.feed(price)
        slow = self._slow.feed(price)
        if fast is None or slow is None:
            return None

        line = fast - slow
        signal = self._signal.feed(line)
        histogram = None if signal is None else line - signal
        return MacdValue(macd=line, signal=signal, histogram=histogram)
