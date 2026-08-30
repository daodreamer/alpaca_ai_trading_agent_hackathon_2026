"""The residual-alpha regression pairs per-session returns, not per-bar ones.

The stitched out-of-sample curve moves once per *bar*; the benchmark series
from ``benchmark_returns`` moves once per *session*. On daily bars the two
coincide. On hourly bars the old pairing handed every bar of a day the same
daily benchmark return -- four strategy observations against one market
observation, repeated four times -- and the regression measured something
other than alpha.

The fix compounds each session's bars into one strategy return before
pairing. These tests pin both halves: on hourly bars the pairing recovers a
known alpha and beta exactly, and on daily bars the result is bit-for-bit the
regression the old code produced.
"""

from __future__ import annotations

import numpy as np
import pytest

from aqr.backtest.alpha import residual_alpha
from aqr.data.bars import Bars
from aqr.pipeline import _residual
from aqr.validation.holdout import benchmark_returns
from aqr.validation.walkforward import WalkForwardReport

SESSIONS = 130  # days of benchmark history
BARS_PER_SESSION = 4
EDGE = 0.0005  # the strategy's known daily alpha over the benchmark

# Day ordinals, consecutive so the session spacing is exactly one day.
DAYS = np.arange(18_000, 18_000 + SESSIONS, dtype=np.int64)


def _daily_returns() -> np.ndarray:
    """One benchmark return per session transition, with genuine variance."""
    rng = np.random.default_rng(41)
    return rng.normal(0.0003, 0.008, SESSIONS - 1)


def _benchmark_bars(daily: np.ndarray) -> Bars:
    """Hourly bars whose *last bar of each session* follows ``daily``.

    ``benchmark_returns`` reads one close per session, so only that last bar
    matters to it; the earlier bars of the day sit flat at the previous close.
    Session ``i`` closes at ``100 * prod(1 + daily[:i])``, so the return into
    session ``i`` is exactly ``daily[i - 1]``.
    """
    last = np.concatenate(([100.0], 100.0 * np.exp(np.cumsum(np.log1p(daily)))))
    stamps: list[int] = []
    closes: list[float] = []
    for i, day in enumerate(DAYS):
        closes.extend([float(last[i])] * BARS_PER_SESSION)
        stamps.extend(day * 86_400 + h * 3600 for h in range(BARS_PER_SESSION))
    c = np.array(closes, dtype=np.float64)
    return Bars(
        symbol="SPY",
        timeframe="1h",
        event_time=np.array(stamps, dtype=np.int64),
        open=c,
        high=c,
        low=c,
        close=c,
        volume=np.full(c.size, 1e6),
    )


def _stitched(daily: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """An hourly stitched curve compounding to ``daily + EDGE`` per session."""
    target = daily + EDGE
    timeline: list[int] = []
    equity: list[float] = []
    level = 1.0
    for i, day in enumerate(DAYS[1:]):
        timeline.extend(int(day * 86_400 + h * 3600) for h in range(BARS_PER_SESSION))
        # Flat through the session, the whole move on the last bar.
        equity.extend([level] * (BARS_PER_SESSION - 1))
        level *= 1.0 + float(target[i])
        equity.append(level)
    return np.array(equity), np.array(timeline, dtype=np.int64)


def _report(equity: np.ndarray, timeline: np.ndarray) -> WalkForwardReport:
    return WalkForwardReport(
        strategy="known_alpha_v1",
        fingerprint="test",
        symbols=("SPY",),
        stitched_equity=equity,
        stitched_timeline=timeline,
    )


def test_hourly_bars_are_compounded_to_sessions_before_pairing() -> None:
    daily = _daily_returns()
    equity, timeline = _stitched(daily)
    result = _residual(_report(equity, timeline), {"SPY": _benchmark_bars(daily)})

    assert result is not None
    # One observation per strategy session, not one per bar.
    assert result.observations == SESSIONS - 1
    # The exact linear case: the edge is recovered, not the rented beta.
    assert result.beta == pytest.approx(1.0)
    assert result.alpha == pytest.approx(EDGE * 252.0, rel=1e-6)


def test_daily_bars_pair_exactly_as_before() -> None:
    """One bar per session: session compounding is the identity, and the
    regression is the old per-bar pairing under another name."""
    daily = _daily_returns()
    last = 100.0 * np.exp(np.cumsum(np.log1p(daily)))
    closes = np.concatenate(([100.0], last))
    bars = Bars(
        symbol="SPY",
        timeframe="1D",
        event_time=DAYS * 86_400,
        open=closes,
        high=closes,
        low=closes,
        close=closes,
        volume=np.full(DAYS.size, 1e6),
    )
    target = daily + EDGE
    equity = np.concatenate(([1.0], np.cumprod(1.0 + target)))
    result = _residual(_report(equity, DAYS * 86_400), {"SPY": bars})

    series = benchmark_returns({"SPY": bars})
    assert series is not None
    _, bench = series
    # What the old code regressed: per-bar strategy returns (identical to the
    # session returns here) against the per-session benchmark, annualised on
    # the stitched timeline's own spacing.
    from aqr.backtest.metrics import periods_per_year

    strat = equity[1:] / equity[:-1] - 1.0
    expected = residual_alpha(
        strat, bench, periods_per_year=periods_per_year(DAYS * 86_400)
    )

    assert result is not None and expected is not None
    assert result.as_dict() == expected.as_dict()
