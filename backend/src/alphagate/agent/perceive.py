"""Step 1 — perception. specs/05 D1, D2.

Turns real bars and a real chain into a `MarketRead`. This is the reuse dividend
from adr/0001 made concrete: the median entrant pastes OHLC into a prompt, and
we hand the model a trend state machine's output, an ATR, and an implied
volatility ranked against its own history.

Impure only in that it *calls* a `MarketData` port; it reads no clock, and
`as_of` arrives as an argument like everywhere else in this system.

**Everything that could not be measured comes back `None`.** Not zero, not a
neutral default. A `MarketRead` with `iv_rank=None` says "nobody has enough
history to rank this" and the screen refuses to build a setup from it; a
`MarketRead` with `iv_rank=0.5` would say "premium is exactly mid-range", which
is a claim. The first live smoke test made precisely this mistake — it labelled a
mean IV level `iv_rank`, the model read "IV rank is low (15.79)" and reasoned
faithfully from it — and the whole shape of this module is a response to that.

The IV history is **reconstructed**, not stored from nowhere. Alpaca serves
current greeks but no historical implied volatility, so `iv_history` takes one
long-dated contract's daily closes and inverts Black–Scholes against the
underlying's closes for the same days. That is a real derivation with a real
caveat, stated in `iv_history`'s own docstring rather than buried here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Final

from alphagate.agent.earnings import EarningsCalendar, NoEarningsCalendar, earnings_within
from alphagate.agent.levels import LevelRead, read_confluence, read_levels
from alphagate.agent.model import MarketRead
from alphagate.agent.trend import TrendRead, read_trend
from alphagate.core.bar import Bar
from alphagate.core.confluence import Confluence
from alphagate.core.identifiers import Ticker
from alphagate.core.indicators.volatility import Atr
from alphagate.core.time_model import Timeframe
from alphagate.marketdata.port import MarketData
from alphagate.options import OptionContract, OptionQuote, Right
from alphagate.options.blackscholes import implied_volatility, time_to_expiry_years
from alphagate.options.volatility import (
    MIN_HISTORY,
    VolatilityRead,
    iv_rank,
    realised_volatility,
    summarise_volatility,
)

__all__ = [
    "ATR_PERIOD",
    "HISTORY_DAYS",
    "OPTIONS_HISTORY_FLOOR",
    "atm_implied_volatility",
    "atr_percent",
    "iv_history",
    "perceive",
]

ATR_PERIOD: Final = 14
HISTORY_DAYS: Final = 180
"""Calendar days of daily bars pulled for volatility and trend work.

Roughly 125 sessions: comfortably past `MIN_HISTORY` and past the trend engine's
warmup, even after holidays and missing prints."""

OPTIONS_HISTORY_FLOOR: Final = date(2024, 2, 5)
"""Alpaca serves no options data before this. Any request reaching back further
returns nothing for the earlier part of the window rather than failing, which is
the kind of silent short series that makes an indicator confidently wrong — so
callers reconstructing IV history clamp to it rather than discovering it."""

RISK_FREE: Final = 0.04
"""A flat 4% discount rate. Held as a named constant rather than fetched: over a
9-to-21-day option the rate moves implied volatility in the fourth decimal, and
a live curve would be three more failure modes for no change in any decision."""


def atr_percent(bars: Sequence[Bar], *, period: int = ATR_PERIOD) -> Decimal | None:
    """ATR as a percentage of the last close. `None` until the indicator is warm.

    A percentage rather than an absolute, because the model compares it against
    breakevens expressed in percentage terms and because $8 of ATR means one
    thing on SPY and another on a $30 stock.
    """
    indicator = Atr(period=period)
    for bar in bars:
        indicator.update(bar)
    if indicator.value is None or not bars:
        return None
    last_close = bars[-1].close
    if last_close <= 0:  # pragma: no cover - Bar refuses non-positive prices
        return None
    return (Decimal(str(indicator.value)) / last_close * 100).quantize(Decimal("0.01"))


def atm_implied_volatility(
    quotes: Mapping[OptionContract, OptionQuote], spot: Decimal
) -> float | None:
    """The implied volatility of the contract nearest the money.

    Taken from the provider's own greeks rather than re-derived: it is the same
    number the chain is quoted against, and inverting our own would introduce a
    second opinion where the point is to have one.

    `None` when no near-the-money contract carries greeks — specs/02 D2's rule
    that absence is representable, applied one level up.
    """
    candidates = [
        (abs(contract.strike - spot), quote)
        for contract, quote in quotes.items()
        if quote.greeks is not None and quote.greeks.iv > 0
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda pair: pair[0])
    return candidates[0][1].greeks.iv  # type: ignore[union-attr]


def iv_history(
    data: MarketData,
    contract: OptionContract,
    closes_by_day: Mapping[date, Decimal],
    *,
    start: date,
    end: date,
    rate: float = RISK_FREE,
) -> list[float]:
    """Reconstruct a daily implied-volatility series from one contract's bars.

    For each session where both the option closed and the underlying closed, the
    Black–Scholes price is inverted for volatility. Days that produce no root —
    a print at or below intrinsic, a crossed market, an expired contract — are
    **dropped, not interpolated**: a fabricated point would enter the very series
    the rank is computed from.

    **The caveat, stated plainly.** One fixed contract is not a constant-maturity
    at-the-money series. As spot drifts the contract moves along the skew, and as
    time passes its maturity shortens, so the level of this series is not
    directly comparable with today's ATM implied volatility. Choosing a
    long-dated, near-the-money contract keeps both effects modest over a few
    months, and ranking is a comparison *within* the series rather than against
    an external level — but this is an approximation and the journal records it
    as a reconstruction rather than as an observation.
    """
    # Alpaca has no options data before February 2024; asking for more returns a
    # shorter series, not an error, and a shorter series than you think you have
    # is how a rank gets computed over the wrong window.
    bars = data.option_daily_bars(contract, start=max(start, OPTIONS_HISTORY_FLOOR), end=end)
    series: list[float] = []
    for bar in bars:
        underlying_close = closes_by_day.get(bar.session_date)
        if underlying_close is None or bar.close <= 0:
            continue
        years = time_to_expiry_years(contract.days_to_expiry(bar.session_date))
        if years <= 0:
            continue
        vol = implied_volatility(
            price=float(bar.close),
            spot=float(underlying_close),
            strike=float(contract.strike),
            years=years,
            is_call=contract.right is Right.CALL,
            rate=rate,
        )
        if vol is not None:
            series.append(vol)
    return series


@dataclass(frozen=True, slots=True)
class Perception:
    """A `MarketRead` plus the volatility detail behind it.

    The detail is returned separately rather than folded into `MarketRead`
    because the journal wants it — how many observations the rank was computed
    from, and which of rank and percentile were computable — while the model is
    better served by the two headline numbers.
    """

    read: MarketRead
    volatility: VolatilityRead
    trend: TrendRead
    levels: LevelRead
    confluence: Confluence | None
    bars: tuple[Bar, ...]

    @property
    def is_tradeable(self) -> bool:
        """Whether perception produced enough to act on. Fail closed."""
        return self.read.is_complete


def perceive(
    data: MarketData,
    symbol: Ticker,
    *,
    as_of: datetime,
    iv_reference: OptionContract | None = None,
    chain: Mapping[OptionContract, OptionQuote] | None = None,
    history: Sequence[float] | None = None,
    calendar: EarningsCalendar | None = None,
    holding_through: date | None = None,
    intraday: Mapping[Timeframe, Sequence[Bar]] | None = None,
    history_days: int = HISTORY_DAYS,
    minimum_history: int = MIN_HISTORY,
) -> Perception:
    """Build a `MarketRead` from real data.

    Implied-volatility history arrives one of two ways, and neither is invented
    here:

    * `history` — observations accumulated by `IvHistoryStore`, one per session.
      This is the path that works on a Basic data plan.
    * `iv_reference` — a contract whose daily bars are inverted through
      Black–Scholes. Needs OPRA data; returns nothing without it.

    With neither, `iv_rank` comes back `None` and renders as `unmeasured`. That
    is the correct behaviour on the first run of a new underlying, and it is
    visibly so rather than quietly so.
    """
    end = as_of.date()
    start = end - timedelta(days=history_days)
    bars = data.daily_bars(symbol, start=start, end=end)
    closes = [bar.close for bar in bars]
    closes_by_day = {bar.session_date: bar.close for bar in bars}

    spot = closes[-1] if closes else data.latest_price(symbol)
    implied = atm_implied_volatility(chain, spot) if chain else None

    observations = list(history) if history else []
    if iv_reference is not None:
        # Reconstruction supplements the accumulated store rather than replacing
        # it: on an entitled account both are real observations of the same
        # thing, and throwing away the accumulated half would reset the history
        # every time the reference contract expired.
        observations = [
            *iv_history(data, iv_reference, closes_by_day, start=start, end=end),
            *observations,
        ]
    volatility = summarise_volatility(
        implied, closes, observations, minimum=minimum_history
    )
    # Realised volatility ranked against its own trailing range. Needs only
    # stock bars, so unlike `iv_rank` it is available today.
    hv_series = _hv_series(closes)
    hv = iv_rank(hv_series[-1], hv_series[:-1], minimum=minimum_history) if hv_series else None

    trend = read_trend(bars)

    atr = atr_percent(bars)
    raw_atr = _atr_absolute(bars)
    levels = read_levels(bars, symbol=symbol, as_of=as_of, atr=raw_atr)

    # Confluence over every timeframe the caller supplied bars for, and no
    # others: reporting "the timeframes agree" from one timeframe is not a
    # weaker claim, it is a different one.
    folds: dict[Timeframe, Sequence[Bar]] = {Timeframe.D1: bars}
    if intraday:
        folds.update(intraday)
    confluence = read_confluence(folds, symbol=symbol, as_of=as_of)

    through = holding_through if holding_through is not None else end + timedelta(days=21)
    earnings = earnings_within(
        calendar if calendar is not None else NoEarningsCalendar(),
        symbol,
        as_of=end,
        through=through,
    )

    read = MarketRead(
        underlying=symbol,
        as_of=as_of,
        atr_pct=atr,
        iv_rank=_as_percent(volatility.rank),
        iv_percentile=_as_percent(volatility.percentile),
        iv_vs_hv=_as_decimal(volatility.ratio),
        hv_rank=_as_percent(hv),
        earnings_within_dte=earnings,
        spot=spot,
        trend=trend.state,
        confluence=confluence,
        levels=levels.levels,
    )
    return Perception(
        read=read,
        volatility=volatility,
        trend=trend,
        levels=levels,
        confluence=confluence,
        bars=bars,
    )


def _atr_absolute(bars: Sequence[Bar], *, period: int = ATR_PERIOD) -> float | None:
    """ATR in price terms, which is what the level engine expresses zones in.

    `atr_percent` normalises for the model's benefit; the engine wants the raw
    number, and converting back from a rounded percentage would lose precision
    the zone widths are computed from.
    """
    indicator = Atr(period=period)
    for bar in bars:
        indicator.update(bar)
    return indicator.value


def _hv_series(closes: Sequence[Decimal], *, window: int = 20) -> list[float]:
    """A rolling realised-volatility series, so today's HV can be ranked.

    Recomputed from scratch per window rather than kept online: this runs once a
    cycle over ~125 closes, and a rolling variance that drifts from the batch one
    is a bug that only shows up as a rank being slightly wrong forever.
    """
    series: list[float] = []
    for end in range(window + 1, len(closes) + 1):
        value = realised_volatility(closes[:end], window=window)
        if value is not None and value > 0:
            series.append(value)
    return series


def _as_percent(value: float | None) -> Decimal | None:
    """A rank in [0, 1] rendered as 0–100, which is how traders read it."""
    if value is None:
        return None
    return (Decimal(str(value)) * 100).quantize(Decimal("0.01"))


def _as_decimal(value: float | None) -> Decimal | None:
    if value is None:
        return None
    return Decimal(str(value)).quantize(Decimal("0.0001"))
