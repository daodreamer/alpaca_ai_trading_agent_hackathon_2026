"""Indicator invariants over generated sequences — specs/14-testing.md.

The unit tests pin chosen numbers; these check the claims that must hold for
every input, which is where an off-by-one in a rolling window shows up.
"""

from __future__ import annotations

from decimal import Decimal

from hypothesis import given
from hypothesis import strategies as st

from alphagate.core.indicators import (
    Atr,
    Ema,
    Macd,
    Rsi,
    SessionVwap,
    Sma,
    VolumeSma,
    compute_series,
)
from tests.core.indicators.synthetic import bar_at, close_bars, ohlc_bars

closes = st.lists(
    st.decimals(min_value=Decimal("1"), max_value=Decimal("1000"), places=2).map(str),
    min_size=1,
    max_size=60,
)


@st.composite
def ohlc_rows(draw: st.DrawFn) -> list[tuple[str, str, str, str]]:
    """Generated bars that satisfy the OHLC invariants by construction."""
    size = draw(st.integers(min_value=1, max_value=40))
    rows: list[tuple[str, str, str, str]] = []
    for _ in range(size):
        low, first, second, high = sorted(
            draw(
                st.lists(
                    st.decimals(min_value=Decimal("1"), max_value=Decimal("500"), places=2),
                    min_size=4,
                    max_size=4,
                )
            )
        )
        open_, close = (first, second) if draw(st.booleans()) else (second, first)
        rows.append((str(open_), str(high), str(low), str(close)))
    return rows


@given(closes)
def test_output_length_always_matches_the_input_axis(prices: list[str]) -> None:
    bars = close_bars(prices)

    for series in (
        compute_series(Sma(period=5), bars),
        compute_series(Ema(period=5), bars),
        compute_series(Rsi(period=5), bars),
        compute_series(Macd(fast=3, slow=5, signal=3), bars),
        compute_series(VolumeSma(period=5), bars),
    ):
        assert len(series.values) == len(prices)


@given(closes)
def test_the_same_input_always_produces_the_same_output(prices: list[str]) -> None:
    """CLAUDE.md §11 — determinism, asserted rather than assumed."""
    bars = close_bars(prices)

    assert compute_series(Ema(period=7), bars).values == compute_series(Ema(period=7), bars).values
    assert compute_series(Rsi(period=7), bars).values == compute_series(Rsi(period=7), bars).values


@given(closes)
def test_sma_stays_inside_the_range_of_its_window(prices: list[str]) -> None:
    period = 4
    bars = close_bars(prices)
    series = compute_series(Sma(period=period), bars)

    for index, value in enumerate(series.values):
        if value is None:
            continue
        window = [float(bar.close) for bar in bars[index - period + 1 : index + 1]]
        assert min(window) - 1e-9 <= value <= max(window) + 1e-9


@given(closes)
def test_ema_stays_inside_the_range_of_the_prices_it_has_seen(prices: list[str]) -> None:
    bars = close_bars(prices)
    series = compute_series(Ema(period=4), bars)

    for index, value in enumerate(series.values):
        if value is None:
            continue
        seen = [float(bar.close) for bar in bars[: index + 1]]
        assert min(seen) - 1e-9 <= value <= max(seen) + 1e-9


@given(closes)
def test_rsi_stays_within_its_bounds(prices: list[str]) -> None:
    series = compute_series(Rsi(period=5), close_bars(prices))

    assert all(0.0 <= value <= 100.0 for value in series.values if value is not None)


@given(ohlc_rows())
def test_atr_is_never_negative(rows: list[tuple[str, str, str, str]]) -> None:
    series = compute_series(Atr(period=5), ohlc_bars(rows))

    assert all(value >= 0.0 for value in series.values if value is not None)


@given(ohlc_rows())
def test_session_vwap_stays_within_the_sessions_price_range(
    rows: list[tuple[str, str, str, str]],
) -> None:
    bars = ohlc_bars(rows)
    series = compute_series(SessionVwap(), bars)
    low = min(float(bar.low) for bar in bars)
    high = max(float(bar.high) for bar in bars)

    assert all(low - 1e-9 <= value <= high + 1e-9 for value in series.values if value is not None)


@given(closes)
def test_streaming_matches_batch_for_every_sequence(prices: list[str]) -> None:
    bars = close_bars(prices)
    online = Macd(fast=3, slow=5, signal=3)

    assert (
        tuple(online.update(bar) for bar in bars)
        == compute_series(Macd(fast=3, slow=5, signal=3), bars).values
    )


@given(closes)
def test_a_provisional_reading_never_disturbs_the_committed_series(prices: list[str]) -> None:
    bars = close_bars(prices)
    quiet = compute_series(Ema(period=5), bars).values

    ticked = Ema(period=5, allow_partial=True)
    streamed: list[float | None] = []
    for index, bar in enumerate(bars):
        ticked.update(bar_at(index, ("1", "1", "1", "1"), is_final=False))
        streamed.append(ticked.update(bar))

    assert tuple(streamed) == quiet
