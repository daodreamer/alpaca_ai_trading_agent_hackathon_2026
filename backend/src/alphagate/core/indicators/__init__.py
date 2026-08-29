"""Indicator engine — specs/05-indicators.md, ADR 0007.

Indicators are deterministic numerical functions of bars. They produce numbers
and nothing else: no alerts, no trade judgements, no notifications (CLAUDE.md
§4). Every one of them is an online state machine, and whole-series computation
is a fold of that machine (`compute_series`), so a replayed value and a live
value are the same value.

MVP set (specs/19-acceptance.md):

    Sma, Ema            moving averages, any PriceSource
    Rsi                 Wilder-smoothed, default 14
    Macd                12/26/9, line + signal + histogram
    Atr                 Wilder-smoothed true range, default 14
    SessionVwap         session-anchored, calendar-aware via the bar
    VolumeSma           moving average of volume

Conventions — seeding, warmup, degenerate cases, intrabar behaviour — are stated
in `adr/0007-indicator-conventions.md` rather than left to each implementation.
"""

from alphagate.core.indicators.base import (
    IndicatorPoint,
    IndicatorSeries,
    IndicatorSpec,
    OnlineIndicator,
    PriceSource,
    RollingWindow,
    compute_series,
    require_period,
)
from alphagate.core.indicators.moving_averages import Ema, Sma
from alphagate.core.indicators.oscillators import NEUTRAL_RSI, Macd, MacdValue, Rsi
from alphagate.core.indicators.smoothing import EmaCore, WilderCore
from alphagate.core.indicators.volatility import Atr, true_range
from alphagate.core.indicators.volume import SessionVwap, VolumeSma

__all__ = [
    "NEUTRAL_RSI",
    "Atr",
    "Ema",
    "EmaCore",
    "IndicatorPoint",
    "IndicatorSeries",
    "IndicatorSpec",
    "Macd",
    "MacdValue",
    "OnlineIndicator",
    "PriceSource",
    "RollingWindow",
    "Rsi",
    "SessionVwap",
    "Sma",
    "VolumeSma",
    "WilderCore",
    "compute_series",
    "require_period",
    "true_range",
]
