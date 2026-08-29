"""Causal technical indicators over 1-D float arrays.

Conventions used throughout:

* Input and output are ``numpy`` float64 arrays of identical length.
* The warm-up entries are ``NaN``. Callers must treat ``NaN`` as "not yet
  computable", never as zero.
* No function reads ``x[i + 1]`` when producing ``out[i]``. This is the single
  property that keeps the backtester honest, and it is tested directly by
  ``tests/test_causality.py``, which truncates the input and asserts the prefix
  of the output is unchanged.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

__all__ = [
    "adx",
    "atr",
    "bollinger",
    "ema",
    "macd",
    "percentile_rank",
    "realized_vol",
    "roc",
    "rolling_max",
    "rolling_min",
    "rsi",
    "rvol",
    "sma",
    "stochastic_k",
    "true_range",
    "vwap_session",
]


def _as_array(x: Array | list[float]) -> Array:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 1:
        raise ValueError(f"expected a 1-D array, got shape {a.shape}")
    return a


def _validate_period(period: int, name: str) -> None:
    if period < 1:
        raise ValueError(f"{name}: period must be >= 1, got {period}")


def sma(x: Array, period: int) -> Array:
    """Simple moving average."""
    x = _as_array(x)
    _validate_period(period, "sma")
    out = np.full(x.size, np.nan)
    if period > x.size:
        return out
    csum = np.cumsum(np.insert(x, 0, 0.0))
    out[period - 1 :] = (csum[period:] - csum[:-period]) / period
    return out


def ema(x: Array, period: int) -> Array:
    """Exponential moving average, seeded with the SMA of the first window.

    Seeding with an SMA rather than with ``x[0]`` keeps the series independent of
    how much history happens to be loaded, which matters when the same strategy
    is walked forward over different windows.
    """
    x = _as_array(x)
    _validate_period(period, "ema")
    out = np.full(x.size, np.nan)
    if period > x.size:
        return out
    alpha = 2.0 / (period + 1.0)
    prev = float(np.mean(x[:period]))
    out[period - 1] = prev
    for i in range(period, x.size):
        prev = alpha * float(x[i]) + (1.0 - alpha) * prev
        out[i] = prev
    return out


def _wilder(x: Array, period: int) -> Array:
    """Wilder's smoothing, the RMA underlying ATR / ADX / RSI."""
    out = np.full(x.size, np.nan)
    if period > x.size:
        return out
    prev = float(np.mean(x[:period]))
    out[period - 1] = prev
    for i in range(period, x.size):
        prev = (prev * (period - 1) + float(x[i])) / period
        out[i] = prev
    return out


def rsi(x: Array, period: int = 14) -> Array:
    """Relative Strength Index (Wilder). Range [0, 100]."""
    x = _as_array(x)
    _validate_period(period, "rsi")
    out = np.full(x.size, np.nan)
    if x.size < period + 1:
        return out
    delta = np.diff(x)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = _wilder(gain, period)
    avg_loss = _wilder(loss, period)
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = np.where(avg_loss == 0.0, np.inf, avg_gain / avg_loss)
        val = 100.0 - 100.0 / (1.0 + rs)
    val = np.where(np.isnan(avg_gain), np.nan, val)
    out[1:] = val
    return out


def true_range(high: Array, low: Array, close: Array) -> Array:
    """True range. The first bar falls back to its own high-low range."""
    high, low, close = _as_array(high), _as_array(low), _as_array(close)
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    return np.maximum(
        high - low,
        np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)),
    )


def atr(high: Array, low: Array, close: Array, period: int = 14) -> Array:
    """Average True Range (Wilder-smoothed)."""
    _validate_period(period, "atr")
    return _wilder(true_range(high, low, close), period)


def adx(high: Array, low: Array, close: Array, period: int = 14) -> Array:
    """Average Directional Index: trend *strength*, direction-agnostic."""
    high, low, close = _as_array(high), _as_array(low), _as_array(close)
    _validate_period(period, "adx")
    n = high.size
    out = np.full(n, np.nan)
    if n < 2 * period + 1:
        return out

    up = np.diff(high, prepend=high[0])
    down = -np.diff(low, prepend=low[0])
    up[0] = 0.0
    down[0] = 0.0
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)

    tr_s = _wilder(true_range(high, low, close), period)
    plus_s = _wilder(plus_dm, period)
    minus_s = _wilder(minus_dm, period)

    with np.errstate(divide="ignore", invalid="ignore"):
        plus_di = 100.0 * plus_s / tr_s
        minus_di = 100.0 * minus_s / tr_s
        denom = plus_di + minus_di
        dx = np.where(denom == 0.0, 0.0, 100.0 * np.abs(plus_di - minus_di) / denom)

    first = period - 1
    out[first:] = _wilder(np.nan_to_num(dx[first:], nan=0.0), period)
    out[: 2 * period - 2] = np.nan
    return out


def macd(x: Array, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[Array, Array, Array]:
    """Returns ``(macd_line, signal_line, histogram)``."""
    x = _as_array(x)
    line = ema(x, fast) - ema(x, slow)
    sig = np.full(x.size, np.nan)
    finite = ~np.isnan(line)
    if finite.any():
        start = int(np.argmax(finite))
        sig[start:] = ema(line[start:], signal)
    return line, sig, line - sig


def bollinger(x: Array, period: int = 20, k: float = 2.0) -> tuple[Array, Array, Array]:
    """Returns ``(lower, middle, upper)``. Population std, per the classic definition."""
    x = _as_array(x)
    _validate_period(period, "bollinger")
    mid = sma(x, period)
    dev = np.full(x.size, np.nan)
    if period <= x.size:
        win = np.lib.stride_tricks.sliding_window_view(x, period)
        dev[period - 1 :] = win.std(axis=-1)
    return mid - k * dev, mid, mid + k * dev


def roc(x: Array, period: int = 10) -> Array:
    """Rate of change as a fraction: 0.05 means +5%."""
    x = _as_array(x)
    _validate_period(period, "roc")
    out = np.full(x.size, np.nan)
    if period >= x.size:
        return out
    prev = x[:-period]
    with np.errstate(divide="ignore", invalid="ignore"):
        out[period:] = np.where(prev == 0.0, np.nan, (x[period:] - prev) / prev)
    return out


def rolling_max(x: Array, period: int) -> Array:
    x = _as_array(x)
    _validate_period(period, "rolling_max")
    out = np.full(x.size, np.nan)
    if period <= x.size:
        win = np.lib.stride_tricks.sliding_window_view(x, period)
        out[period - 1 :] = win.max(axis=-1)
    return out


def rolling_min(x: Array, period: int) -> Array:
    x = _as_array(x)
    _validate_period(period, "rolling_min")
    out = np.full(x.size, np.nan)
    if period <= x.size:
        win = np.lib.stride_tricks.sliding_window_view(x, period)
        out[period - 1 :] = win.min(axis=-1)
    return out


def stochastic_k(high: Array, low: Array, close: Array, period: int = 14) -> Array:
    """Stochastic %K in [0, 100]."""
    hi = rolling_max(_as_array(high), period)
    lo = rolling_min(_as_array(low), period)
    close = _as_array(close)
    with np.errstate(divide="ignore", invalid="ignore"):
        rng = hi - lo
        return np.where(rng == 0.0, 50.0, 100.0 * (close - lo) / rng)


def realized_vol(close: Array, period: int = 20, annualize: int = 252) -> Array:
    """Annualised realised volatility of log returns."""
    close = _as_array(close)
    _validate_period(period, "realized_vol")
    out = np.full(close.size, np.nan)
    if close.size < 2 or period > close.size - 1:
        return out
    with np.errstate(divide="ignore", invalid="ignore"):
        logret = np.diff(np.log(close))
    win = np.lib.stride_tricks.sliding_window_view(logret, period)
    out[period:] = win.std(axis=-1) * np.sqrt(annualize)
    return out


def rvol(volume: Array, period: int = 20) -> Array:
    """Relative volume: this bar over the mean of the *preceding* window.

    The window deliberately excludes the current bar. Including it would let a
    volume spike dilute the very signal it is supposed to raise.
    """
    volume = _as_array(volume)
    _validate_period(period, "rvol")
    out = np.full(volume.size, np.nan)
    if period >= volume.size:
        return out
    avg = sma(volume, period)
    prior = np.roll(avg, 1)
    prior[0] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where((prior == 0.0) | np.isnan(prior), np.nan, volume / prior)
    return out


def percentile_rank(x: Array, period: int = 252) -> Array:
    """Fraction of the trailing window that ``x[i]`` exceeds, in [0, 1]."""
    x = _as_array(x)
    _validate_period(period, "percentile_rank")
    out = np.full(x.size, np.nan)
    if period > x.size:
        return out
    win = np.lib.stride_tricks.sliding_window_view(x, period)
    current = x[period - 1 :][:, None]
    out[period - 1 :] = (win < current).sum(axis=-1) / period
    return out


def vwap_session(
    high: Array, low: Array, close: Array, volume: Array, session: NDArray[Any]
) -> Array:
    """Volume-weighted average price, reset whenever ``session`` changes.

    ``session`` is any array constant within a session (a date ordinal, for
    daily-reset intraday VWAP). On daily bars each bar is its own session and
    this degenerates to the typical price.
    """
    high, low, close, volume = (_as_array(a) for a in (high, low, close, volume))
    typical = (high + low + close) / 3.0
    out = np.full(close.size, np.nan)
    cum_pv = 0.0
    cum_v = 0.0
    prev: float | None = None
    for i in range(close.size):
        marker = float(session[i])
        if prev is None or marker != prev:
            cum_pv = 0.0
            cum_v = 0.0
            prev = marker
        cum_pv += float(typical[i]) * float(volume[i])
        cum_v += float(volume[i])
        out[i] = cum_pv / cum_v if cum_v > 0 else typical[i]
    return out
