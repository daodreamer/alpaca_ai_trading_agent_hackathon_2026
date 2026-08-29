"""The indicator engine contract — ADR 0007 D1/D3/D4/D6/D7/D8.

These tests are about the *frame* every indicator sits in: what the output axis
looks like, which bars are allowed in, and what happens on a partial bar. The
arithmetic of each indicator is pinned in its own module.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from alphagate.core.bar import AdjustmentMode, Bar, Feed
from alphagate.core.errors import InvariantViolation
from alphagate.core.identifiers import ticker
from alphagate.core.indicators import (
    Atr,
    Ema,
    Macd,
    OnlineIndicator,
    PriceSource,
    Rsi,
    SessionVwap,
    Sma,
    VolumeSma,
    compute_series,
)
from alphagate.core.time_model import Timeframe
from tests.core.indicators.synthetic import bar_at, close_bars, ohlc_bars

RAMP = [str(value) for value in range(100, 140)]

type Factory = Callable[[], OnlineIndicator[Any]]

FACTORIES: dict[str, Factory] = {
    "sma": lambda: Sma(period=3),
    "ema": lambda: Ema(period=3),
    "rsi": lambda: Rsi(period=3),
    "macd": lambda: Macd(fast=2, slow=3, signal=2),
    "atr": lambda: Atr(period=3),
    "session_vwap": SessionVwap,
    "volume_sma": lambda: VolumeSma(period=3),
}


@pytest.fixture(params=sorted(FACTORIES))
def factory(request: pytest.FixtureRequest) -> Factory:
    """Builds a fresh instance of one MVP indicator, for contract-wide checks."""
    name: str = request.param
    return FACTORIES[name]


# -- output axis ------------------------------------------------------------


def test_series_has_exactly_one_point_per_input_bar(factory: Factory) -> None:
    bars = close_bars(RAMP)
    series = compute_series(factory(), bars)

    assert len(series.points) == len(bars)
    assert [point.start_time_utc for point in series.points] == [b.start_time_utc for b in bars]


def test_series_carries_the_bars_identity_and_the_indicator_spec() -> None:
    bars = close_bars(RAMP)
    series = compute_series(Ema(period=5), bars)

    assert series.symbol == bars[0].symbol
    assert series.timeframe is Timeframe.M1
    assert series.spec.key == "ema(period=5,source=CLOSE)"
    assert series.warmup == 5


def test_values_are_none_before_warmup_and_present_after(factory: Factory) -> None:
    series = compute_series(factory(), close_bars(RAMP))
    warmup = series.warmup

    assert all(point.value is None for point in series.points[: warmup - 1])
    assert all(point.value is not None for point in series.points[warmup - 1 :])


def test_a_series_shorter_than_warmup_is_all_none() -> None:
    series = compute_series(Sma(period=10), close_bars(RAMP[:4]))

    assert series.values == (None,) * 4
    assert not series.is_warm


def test_empty_input_is_refused_rather_than_producing_an_unattributable_series() -> None:
    with pytest.raises(InvariantViolation, match="at least one bar"):
        compute_series(Sma(period=3), ())


def test_compute_series_refuses_an_already_used_indicator() -> None:
    indicator = Sma(period=3)
    compute_series(indicator, close_bars(RAMP))

    with pytest.raises(InvariantViolation, match="fresh"):
        compute_series(indicator, close_bars(RAMP))


# -- online == batch --------------------------------------------------------


def test_streaming_one_bar_at_a_time_reproduces_the_batch_series(factory: Factory) -> None:
    """ADR 0007 D1: batch is a fold of `update`, so the two cannot disagree."""
    bars = close_bars(RAMP)
    indicator = factory()
    streamed = tuple(indicator.update(bar) for bar in bars)

    assert streamed == compute_series(factory(), bars).values


def test_value_property_tracks_the_last_committed_update() -> None:
    indicator = Sma(period=3)
    bars = close_bars(["10", "20", "30", "40"])

    assert indicator.value is None
    assert not indicator.is_warm

    for bar in bars[:2]:
        indicator.update(bar)
    assert indicator.value is None

    indicator.update(bars[2])
    assert indicator.is_warm
    assert indicator.value == pytest.approx(20.0)

    indicator.update(bars[3])
    assert indicator.value == pytest.approx(30.0)


# -- input discipline (D6) --------------------------------------------------


def test_a_repeated_bar_is_refused(factory: Factory) -> None:
    indicator = factory()
    bars = close_bars(RAMP[:5])
    for bar in bars:
        indicator.update(bar)

    with pytest.raises(InvariantViolation, match="strictly increasing"):
        indicator.update(bars[-1])


def test_an_out_of_order_bar_is_refused(factory: Factory) -> None:
    indicator = factory()
    bars = close_bars(RAMP[:5])
    for bar in bars:
        indicator.update(bar)

    with pytest.raises(InvariantViolation, match="strictly increasing"):
        indicator.update(bars[2])


FOREIGN_BARS: dict[str, Callable[[], Bar]] = {
    "symbol": lambda: bar_at(9, ("100", "100", "100", "100"), symbol=ticker("MSFT")),
    "timeframe": lambda: bar_at(9, ("100", "100", "100", "100"), timeframe=Timeframe.M5),
    "feed": lambda: bar_at(9, ("100", "100", "100", "100"), feed=Feed.IEX),
    "adjustment": lambda: bar_at(
        9, ("100", "100", "100", "100"), adjustment_mode=AdjustmentMode.ADJUSTED
    ),
}


@pytest.mark.parametrize("difference", sorted(FOREIGN_BARS))
def test_a_bar_from_another_series_is_refused(difference: str) -> None:
    indicator = Ema(period=3)
    for bar in close_bars(RAMP[:3]):
        indicator.update(bar)

    with pytest.raises(InvariantViolation, match="series"):
        indicator.update(FOREIGN_BARS[difference]())


# -- partial bars (D7) ------------------------------------------------------


def test_a_partial_bar_is_refused_by_default(factory: Factory) -> None:
    indicator = factory()
    for bar in close_bars(RAMP[:5]):
        indicator.update(bar)

    with pytest.raises(InvariantViolation, match="partial"):
        indicator.update(bar_at(5, ("140", "141", "139", "140"), is_final=False))


def test_a_provisional_update_does_not_change_committed_state() -> None:
    indicator = Ema(period=3, allow_partial=True)
    for bar in close_bars(["10", "20", "30"]):
        indicator.update(bar)
    committed = indicator.value

    provisional = indicator.update(bar_at(3, ("99", "99", "99", "99"), is_final=False))

    assert provisional is not None
    assert provisional != committed
    assert indicator.value == committed


def test_intrabar_ticks_leave_no_trace_once_the_bar_closes() -> None:
    """The committed value is what it would have been had the ticks never arrived."""
    bars = close_bars(RAMP[:8])

    quiet = Ema(period=3)
    for bar in bars:
        quiet.update(bar)

    ticked = Ema(period=3, allow_partial=True)
    for index, bar in enumerate(bars):
        for tick in ("1", "999", "50"):
            ticked.update(bar_at(index, (tick, tick, tick, tick), is_final=False))
        ticked.update(bar)

    assert ticked.value == quiet.value


def test_a_provisional_update_before_warmup_may_still_be_none() -> None:
    indicator = Sma(period=5, allow_partial=True)
    indicator.update(close_bars(["10"])[0])

    assert indicator.update(bar_at(1, ("11", "11", "11", "11"), is_final=False)) is None


# -- parameter identity (D8) ------------------------------------------------


SPEC_KEYS: list[tuple[Factory, str]] = [
    (lambda: Sma(period=20), "sma(period=20,source=CLOSE)"),
    (lambda: Ema(period=200, source=PriceSource.HLC3), "ema(period=200,source=HLC3)"),
    (lambda: Rsi(period=14), "rsi(period=14,source=CLOSE)"),
    (Macd, "macd(fast=12,slow=26,signal=9,source=CLOSE)"),
    (lambda: Atr(period=14), "atr(period=14)"),
    (SessionVwap, "session_vwap(sessions=REGULAR)"),
    (lambda: VolumeSma(period=20), "volume_sma(period=20)"),
]


@pytest.mark.parametrize(("build", "key"), SPEC_KEYS, ids=[key for _, key in SPEC_KEYS])
def test_spec_key_names_the_configuration(build: Factory, key: str) -> None:
    assert build().spec.key == key


def test_a_spec_renders_as_its_key() -> None:
    assert str(Sma(period=20).spec) == "sma(period=20,source=CLOSE)"


def test_an_indicator_reports_its_key_and_progress_when_logged() -> None:
    indicator = Sma(period=3)
    for bar in close_bars(RAMP[:2]):
        indicator.update(bar)

    assert repr(indicator) == "<sma(period=3,source=CLOSE) committed=2>"


def test_a_series_length_is_its_point_count() -> None:
    assert len(compute_series(Sma(period=3), close_bars(RAMP))) == len(RAMP)


def test_specs_with_the_same_parameters_are_equal() -> None:
    assert Ema(period=20).spec == Ema(period=20).spec
    assert Ema(period=20).spec != Ema(period=50).spec


@pytest.mark.parametrize("period", [0, -1])
def test_a_non_positive_period_is_refused(period: int) -> None:
    with pytest.raises(InvariantViolation, match="period"):
        Sma(period=period)


def test_macd_requires_a_fast_period_shorter_than_the_slow_one() -> None:
    with pytest.raises(InvariantViolation, match="fast"):
        Macd(fast=26, slow=12)


# -- price sources ----------------------------------------------------------


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (PriceSource.OPEN, 10.0),
        (PriceSource.HIGH, 14.0),
        (PriceSource.LOW, 8.0),
        (PriceSource.CLOSE, 12.0),
        (PriceSource.HL2, 11.0),
        (PriceSource.HLC3, (14.0 + 8.0 + 12.0) / 3),
        (PriceSource.OHLC4, (10.0 + 14.0 + 8.0 + 12.0) / 4),
    ],
)
def test_price_source_extracts_the_documented_combination(
    source: PriceSource, expected: float
) -> None:
    bar = ohlc_bars([("10", "14", "8", "12")])[0]

    assert source.of(bar) == pytest.approx(expected)
