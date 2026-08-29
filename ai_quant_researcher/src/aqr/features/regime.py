"""Market regime classification (architecture section 7).

``regime_robustness`` is 10% of the evaluator's weight and needs a label per
bar. Until now only :class:`SyntheticProvider` could supply one, because only
the simulator knows the regime it generated; on real data the pipeline passed no
labels, the report was never built, and the evaluator fell back to 0.5. A tenth
of every real-data score was a constant standing in for a measurement.

Four regimes, matching the simulator's own vocabulary so that a strategy scored
on synthetic bars and on real bars is scored against the same four questions:

``TREND_BULL``      a move up that is large relative to the noise
``TREND_BEAR``      the same, down
``RANGE_LOW_VOL``   no directional edge, and quiet
``RANGE_HIGH_VOL``  no directional edge, and violent

plus ``UNKNOWN`` for the warm-up, where there is not yet enough history to say
anything. Guessing there would attribute early trades to a regime that was never
measured, which is worse than declining to judge them: ``regime_robustness``
skips trades whose regime is unknown.

**How trend is decided.** Not by a moving-average crossover, which says
"trending" whenever price wanders slightly in one direction, but by asking
whether the recent move is large *relative to the volatility of this
instrument*. The statistic is the drift's t-value::

    z = log(close[t] / close[t - LOOKBACK]) / (sigma * sqrt(LOOKBACK / 252))

A z of 3 means the move is three standard deviations of noise; a z of 0.4 means
the market went nowhere in a way that happens to have a sign. This is scale-free,
so a 12%-vol utility and a 90%-vol biotech are judged on the same axis without
per-symbol tuning -- and per-symbol tuning is how a regime classifier becomes
another fitted parameter nobody counted.

**How volatility is decided.** Recent realised vol against an *expanding*
baseline -- the mean of every reading this instrument has produced so far --
rather than against a trailing window of its own.

The trailing-window version fails exactly when it matters. Measure 20-bar vol
against 252-bar vol and a market that has been violent for a year scores 1.0:
the denominator has absorbed the crisis, so the classifier calls it normal. A
percentile of the series against itself fails the same way for the same reason.
The expanding baseline cannot be captured like that, because the calm years stay
in it. It is still causal -- an expanding window is a trailing window that never
forgets.

**Causality.** Every input is a trailing window and every indicator here is
prefix-stable, so the label at bar ``t`` cannot change when bar ``t+1`` arrives.
That is pinned by a test, not by inspection: a regime classifier that peeks is
contaminated in the most flattering possible direction, since it would know
which regime the market was about to be in.
"""

from __future__ import annotations

import numpy as np

from aqr.core.indicators import realized_vol
from aqr.data.bars import Bars

__all__ = ["REGIMES", "classify", "regime_series"]

REGIMES = ("TREND_BULL", "TREND_BEAR", "RANGE_LOW_VOL", "RANGE_HIGH_VOL", "UNKNOWN")

# Bars over which the drift is measured. A quarter: long enough that noise
# averages out, short enough that a regime change is visible while it matters.
LOOKBACK = 60

# The recent-volatility window, and how many readings the expanding baseline
# needs before it means anything. A "normal" built from thirty bars is not a
# normal, and comparing against it produces regime labels that flip on noise.
VOL_FAST = 20
MIN_VOL_HISTORY = 252

# |z| at or above this is a trend. 1.5 is roughly a one-sided 93% confidence
# that the drift is not noise -- deliberately not 2.0, which is so strict that
# most genuine trends spend their lives classified as ranges.
TREND_Z = 1.5

# Recent vol this many times the baseline is a high-volatility regime: a third
# above this instrument's own long-run normal.
HIGH_VOL_RATIO = 1.3

TRADING_DAYS = 252.0


def _labels(bars: Bars) -> list[str]:
    n = len(bars)
    if n == 0:
        return []

    close = np.asarray(bars.close, dtype=np.float64)
    out = ["UNKNOWN"] * n

    fast = realized_vol(close, VOL_FAST, annualize=int(TRADING_DAYS))

    # Expanding mean of every volatility reading so far. Cumulative sums keep it
    # O(n) and, more importantly, keep it exactly causal: the value at t is a
    # function of readings at or before t and nothing else.
    seen = np.isfinite(fast)
    count = np.cumsum(seen)
    total = np.cumsum(np.where(seen, fast, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        baseline = np.where(
            count >= MIN_VOL_HISTORY, total / np.maximum(count, 1), np.nan
        )

    # The drift's t-value. Both terms are trailing: the numerator ends at t, the
    # denominator is an estimate made from bars up to t.
    with np.errstate(divide="ignore", invalid="ignore"):
        move = np.full(n, np.nan)
        if n > LOOKBACK:
            move[LOOKBACK:] = np.log(close[LOOKBACK:] / close[:-LOOKBACK])
        scale = fast * np.sqrt(LOOKBACK / TRADING_DAYS)
        z = np.where(scale > 0, move / scale, np.nan)
        ratio = np.where(baseline > 0, fast / baseline, np.nan)

    for i in range(n):
        z_i, ratio_i = z[i], ratio[i]
        if not np.isfinite(z_i) or not np.isfinite(ratio_i):
            continue  # still warming up; UNKNOWN is the honest answer
        if z_i >= TREND_Z:
            out[i] = "TREND_BULL"
        elif z_i <= -TREND_Z:
            out[i] = "TREND_BEAR"
        elif ratio_i >= HIGH_VOL_RATIO:
            out[i] = "RANGE_HIGH_VOL"
        else:
            out[i] = "RANGE_LOW_VOL"
    return out


def regime_series(bars: Bars) -> list[str]:
    """A regime label for every bar, computed only from that bar and earlier."""
    return _labels(bars)


def classify(bars: Bars, index: int) -> str:
    """The regime at one bar.

    Equivalent to ``regime_series(bars)[index]`` and tested to agree with it.
    Provided for the caller that wants a single label without materialising the
    whole series -- reporting, mostly.
    """
    n = len(bars)
    if not -n <= index < n:
        raise IndexError(f"bar {index} out of range for {n} bars")
    return _labels(bars)[index]
