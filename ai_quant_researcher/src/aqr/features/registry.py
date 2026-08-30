"""The whitelist of computable features.

This registry is the *entire* vocabulary the strategy DSL can reach. An LLM that
invents ``ema_of_tomorrow(20)`` gets a validation error naming the closest known
feature, not a silent NaN and not an exception three layers down in the
backtester. Adding a feature is a deliberate act of extending this table.

Each entry declares:

* ``arity``    how many integer/float arguments it takes.
* ``build``    ``(Bars, args) -> float array`` aligned to the bar index.
* ``warmup``   bars of history the feature needs before it is meaningful. The
                backtester refuses to trade until the largest warmup has elapsed.
* ``doc``      one line, surfaced to the LLM in the prompt.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from aqr.core import indicators as ind
from aqr.data.bars import Array, Bars, bars_per_year

if TYPE_CHECKING:
    from aqr.features.cross_section import CrossSection

__all__ = ["FeatureSpec", "REGISTRY", "cross_sectional_names", "feature_names", "resolve"]

Builder = Callable[[Bars, tuple[float, ...]], Array]

# A cross-sectional feature cannot be computed from one symbol's bars: it
# needs the universe. Rather than widen every builder's signature -- 29 of
# them, none of which want the argument -- these declare themselves and take
# a prepared CrossSection instead.
CrossBuilder = Callable[["CrossSection", str, tuple[float, ...]], Array]


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    name: str
    arity: int
    build: Builder | None
    warmup: Callable[[tuple[float, ...]], int]
    doc: str
    build_cross: CrossBuilder | None = None
    """Set instead of ``build`` for features that need the whole universe."""

    @property
    def cross_sectional(self) -> bool:
        return self.build_cross is not None


def _p(args: tuple[float, ...], i: int, default: float) -> float:
    return args[i] if len(args) > i else default


def _int(args: tuple[float, ...], i: int, default: int) -> int:
    value = _p(args, i, float(default))
    if value != int(value):
        raise ValueError(f"expected an integer argument, got {value}")
    return int(value)


REGISTRY: dict[str, FeatureSpec] = {}


def _register(
    name: str,
    arity: int,
    build: Builder,
    warmup: Callable[[tuple[float, ...]], int],
    doc: str,
) -> None:
    REGISTRY[name] = FeatureSpec(name=name, arity=arity, build=build, warmup=warmup, doc=doc)


def _register_cross(
    name: str,
    arity: int,
    build: CrossBuilder,
    warmup: Callable[[tuple[float, ...]], int],
    doc: str,
) -> None:
    REGISTRY[name] = FeatureSpec(
        name=name, arity=arity, build=None, warmup=warmup, doc=doc, build_cross=build
    )


# --------------------------------------------------------------------------- #
# Raw price and volume
# --------------------------------------------------------------------------- #

for _field in ("open", "high", "low", "close", "volume"):

    def _raw(bars: Bars, _args: tuple[float, ...], _f: str = _field) -> Array:
        return np.asarray(getattr(bars, _f), dtype=np.float64)

    _register(_field, 0, _raw, lambda _a: 1, f"raw {_field} of the current bar")

# --------------------------------------------------------------------------- #
# Trend
# --------------------------------------------------------------------------- #

_register(
    "sma",
    1,
    lambda b, a: ind.sma(b.close, _int(a, 0, 20)),
    lambda a: _int(a, 0, 20),
    "sma(n): simple moving average of close",
)
_register(
    "ema",
    1,
    lambda b, a: ind.ema(b.close, _int(a, 0, 20)),
    lambda a: _int(a, 0, 20),
    "ema(n): exponential moving average of close",
)
_register(
    "adx",
    1,
    lambda b, a: ind.adx(b.high, b.low, b.close, _int(a, 0, 14)),
    lambda a: 2 * _int(a, 0, 14) + 1,
    "adx(n): trend strength in [0, 100]; > 20-25 is commonly read as trending",
)
_register(
    "macd_hist",
    3,
    lambda b, a: ind.macd(b.close, _int(a, 0, 12), _int(a, 1, 26), _int(a, 2, 9))[2],
    lambda a: _int(a, 1, 26) + _int(a, 2, 9),
    "macd_hist(fast, slow, signal): MACD histogram",
)
_register(
    "roc",
    1,
    lambda b, a: ind.roc(b.close, _int(a, 0, 10)),
    lambda a: _int(a, 0, 10) + 1,
    "roc(n): n-bar rate of change as a fraction (0.05 = +5%)",
)
_register(
    "slope",
    1,
    lambda b, a: _slope(b.close, _int(a, 0, 20)),
    lambda a: _int(a, 0, 20),
    "slope(n): least-squares slope of close over n bars, normalised by price",
)

# --------------------------------------------------------------------------- #
# Mean reversion / oscillators
# --------------------------------------------------------------------------- #

_register(
    "rsi",
    1,
    lambda b, a: ind.rsi(b.close, _int(a, 0, 14)),
    lambda a: _int(a, 0, 14) + 1,
    "rsi(n): Wilder RSI in [0, 100]",
)
_register(
    "stoch_k",
    1,
    lambda b, a: ind.stochastic_k(b.high, b.low, b.close, _int(a, 0, 14)),
    lambda a: _int(a, 0, 14),
    "stoch_k(n): stochastic %K in [0, 100]",
)
_register(
    "bb_upper",
    2,
    lambda b, a: ind.bollinger(b.close, _int(a, 0, 20), _p(a, 1, 2.0))[2],
    lambda a: _int(a, 0, 20),
    "bb_upper(n, k): upper Bollinger band",
)
_register(
    "bb_lower",
    2,
    lambda b, a: ind.bollinger(b.close, _int(a, 0, 20), _p(a, 1, 2.0))[0],
    lambda a: _int(a, 0, 20),
    "bb_lower(n, k): lower Bollinger band",
)
_register(
    "zscore",
    1,
    lambda b, a: _zscore(b.close, _int(a, 0, 20)),
    lambda a: _int(a, 0, 20),
    "zscore(n): (close - sma(n)) / rolling std(n)",
)

# --------------------------------------------------------------------------- #
# Volatility
# --------------------------------------------------------------------------- #

_register(
    "atr",
    1,
    lambda b, a: ind.atr(b.high, b.low, b.close, _int(a, 0, 14)),
    lambda a: _int(a, 0, 14),
    "atr(n): average true range in price units",
)
_register(
    "atr_pct",
    1,
    lambda b, a: ind.atr(b.high, b.low, b.close, _int(a, 0, 14)) / b.close,
    lambda a: _int(a, 0, 14),
    "atr_pct(n): ATR as a fraction of price",
)
_register(
    "realized_vol",
    1,
    lambda b, a: ind.realized_vol(
        b.close, _int(a, 0, 20), annualize=bars_per_year(b.timeframe)
    ),
    lambda a: _int(a, 0, 20) + 1,
    "realized_vol(n): annualised realised volatility of log returns, n bars",
)
_register(
    "vol_pct",
    2,
    lambda b, a: ind.percentile_rank(
        ind.realized_vol(b.close, _int(a, 0, 20), annualize=bars_per_year(b.timeframe)),
        _int(a, 1, round(bars_per_year(b.timeframe))),
    ),
    lambda a: _int(a, 0, 20) + _int(a, 1, 252) + 1,
    "vol_pct(n, lookback): percentile rank of realised vol within its own "
    "history; the default lookback is one year of bars at this timeframe",
)

# --------------------------------------------------------------------------- #
# Volume
# --------------------------------------------------------------------------- #

_register(
    "rvol",
    1,
    lambda b, a: ind.rvol(b.volume, _int(a, 0, 20)),
    lambda a: _int(a, 0, 20) + 1,
    "rvol(n): volume over the mean of the n preceding bars",
)
_register(
    "vwap",
    0,
    lambda b, _a: ind.vwap_session(b.high, b.low, b.close, b.volume, b.session),
    lambda _a: 1,
    "vwap(): session volume-weighted average price",
)

# --------------------------------------------------------------------------- #
# Structure
# --------------------------------------------------------------------------- #

_register(
    "highest",
    1,
    lambda b, a: ind.rolling_max(b.high, _int(a, 0, 20)),
    lambda a: _int(a, 0, 20),
    "highest(n): highest high of the last n bars, current bar included",
)
_register(
    "lowest",
    1,
    lambda b, a: ind.rolling_min(b.low, _int(a, 0, 20)),
    lambda a: _int(a, 0, 20),
    "lowest(n): lowest low of the last n bars, current bar included",
)
_register(
    "dist_to_high",
    1,
    lambda b, a: (ind.rolling_max(b.high, _int(a, 0, 20)) - b.close) / b.close,
    lambda a: _int(a, 0, 20),
    "dist_to_high(n): fractional distance below the n-bar high",
)
_register(
    "dist_to_low",
    1,
    lambda b, a: (b.close - ind.rolling_min(b.low, _int(a, 0, 20))) / b.close,
    lambda a: _int(a, 0, 20),
    "dist_to_low(n): fractional distance above the n-bar low",
)


# --------------------------------------------------------------------------- #
# Overnight
#
# The DSL had no way to reach a previous bar's value, so an overnight gap --
# open[t] against close[t-1] -- was not expressible at all. Models proposed
# overnight hypotheses repeatedly anyway and approximated them with conditions
# that meant something else, which mostly fired on nothing.
#
# Two named features rather than a general lag operator. `prev(n)` would invite
# `prev(1) > prev(2) > prev(3)`: three degrees of freedom spent rediscovering
# `slope(3)`. The overfitting detector charges for parameters, but the cheapest
# defence is a vocabulary that does not suggest the mistake.
# --------------------------------------------------------------------------- #


def _prev_close(bars: Bars, _args: tuple[float, ...]) -> Array:
    close = np.asarray(bars.close, dtype=np.float64)
    out = np.full(close.size, np.nan)
    out[1:] = close[:-1]
    return out


def _gap(bars: Bars, _args: tuple[float, ...]) -> Array:
    """Open against the previous close, as a fraction.

    The first bar is NaN, not zero. Zero would read as "opened flat", which is a
    claim about a session nobody observed; NaN keeps the bar out of every
    comparison it appears in.
    """
    previous = _prev_close(bars, ())
    open_ = np.asarray(bars.open, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(previous > 0, open_ / previous - 1.0, np.nan)


_register(
    "prev_close",
    0,
    _prev_close,
    lambda _a: 2,
    "prev_close(): the close of the previous bar",
)
_register(
    "gap",
    0,
    _gap,
    lambda _a: 2,
    "gap(): overnight gap, this bar's open over the previous close, as a fraction "
    "(0.02 = opened 2% higher)",
)
_register(
    "overnight_return",
    0,
    _gap,
    lambda _a: 2,
    "overnight_return(): the same number as gap(), under the name a "
    "close-to-open hypothesis usually gives it",
)


def _slope(close: Array, period: int) -> Array:
    out = np.full(close.size, np.nan)
    if period < 2 or period > close.size:
        return out
    x = np.arange(period, dtype=np.float64)
    x -= x.mean()
    denom = float((x * x).sum())
    win = np.lib.stride_tricks.sliding_window_view(close, period)
    beta = (win - win.mean(axis=-1, keepdims=True)) @ x / denom
    with np.errstate(divide="ignore", invalid="ignore"):
        out[period - 1 :] = beta / close[period - 1 :]
    return out


def _zscore(close: Array, period: int) -> Array:
    out = np.full(close.size, np.nan)
    if period < 2 or period > close.size:
        return out
    win = np.lib.stride_tricks.sliding_window_view(close, period)
    std = win.std(axis=-1)
    mean = win.mean(axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        out[period - 1 :] = np.where(std == 0.0, 0.0, (close[period - 1 :] - mean) / std)
    return out


# --------------------------------------------------------------------------- #
# Cross-sectional
#
# The vocabulary's largest gap until now: "buy the strongest names in a strong
# market" is the oldest documented equity anomaly there is, and no rule could
# express it, because every rule saw one instrument at a time.
#
# Point-in-time correctness lives in aqr.features.cross_section: at bar t
# the peer set is exactly the symbols that had a bar at t, so a rank is never
# taken against a universe assembled with hindsight.
# --------------------------------------------------------------------------- #

_register_cross(
    "rs_rank",
    1,
    lambda cs, sym, a: cs.rank(sym, _int(a, 0, 60)),
    lambda a: _int(a, 0, 60) + 1,
    "rs_rank(n): percentile rank of this symbol's n-bar return across the "
    "universe, 0 weakest to 1 strongest",
)
_register_cross(
    "rel_return",
    1,
    lambda cs, sym, a: cs.relative_return(sym, _int(a, 0, 60)),
    lambda a: _int(a, 0, 60) + 1,
    "rel_return(n): this symbol's n-bar return minus the universe median, as "
    "a fraction",
)
_register_cross(
    "breadth",
    1,
    lambda cs, sym, a: cs.breadth(sym, _int(a, 0, 60)),
    lambda a: _int(a, 0, 60) + 1,
    "breadth(n): fraction of the universe with a positive n-bar return -- a "
    "market-state feature, the same for every symbol",
)


def cross_sectional_names() -> list[str]:
    """Features that cannot be computed from one symbol's bars alone."""
    return sorted(n for n, spec in REGISTRY.items() if spec.cross_sectional)


def feature_names() -> list[str]:
    return sorted(REGISTRY)


def resolve(name: str) -> FeatureSpec:
    """Look up a feature, or raise with a suggestion.

    The suggestion matters: it is what turns an LLM's near-miss into a
    self-correcting retry rather than a dead experiment.
    """
    if name in REGISTRY:
        return REGISTRY[name]
    import difflib

    close = difflib.get_close_matches(name, REGISTRY, n=3, cutoff=0.6)
    hint = f" Did you mean: {', '.join(close)}?" if close else ""
    raise KeyError(f"unknown feature {name!r}.{hint}")
