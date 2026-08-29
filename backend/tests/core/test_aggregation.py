"""Session grid and bar aggregation — ADR 0004 D4/D5/D8.

The grid is the thing that makes a 1h bar mean the same on every provider, so
these tests pin the boundaries against real NYSE sessions rather than a
synthetic calendar.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest

from alphagate.core.aggregation import (
    aggregate_bars,
    fold_bars,
    next_close_after,
    session_grid,
)
from alphagate.core.bar import AdjustmentMode, Bar, Feed
from alphagate.core.errors import InvariantViolation
from alphagate.core.identifiers import Ticker, ticker
from alphagate.core.time_model import SessionKind, SessionWindow, Timeframe
from alphagate.infra.calendars import PandasMarketCalendar
from alphagate.infra.clock import ReplayClock

NYSE = "XNYS"
AAPL: Ticker = ticker("AAPL")

NORMAL_DAY = date(2026, 11, 25)  # Wednesday, full session
HALF_DAY = date(2026, 11, 27)  # day after Thanksgiving, 13:00 ET close
HOLIDAY = date(2026, 11, 26)  # Thanksgiving

OPEN = datetime(2026, 11, 25, 14, 30, tzinfo=UTC)
CLOSE = datetime(2026, 11, 25, 21, 0, tzinfo=UTC)

FULL_WEEK_MONDAY = date(2026, 11, 16)  # Mon–Fri, five full sessions
MLK_MONDAY = date(2026, 1, 19)  # shut; the week opens on the Tuesday
# NORMAL_DAY/HOLIDAY/HALF_DAY all fall in the Thanksgiving week: Mon–Wed full,
# Thursday shut, Friday closing at 13:00 ET.
WEEK_OPEN = datetime(2026, 11, 23, 14, 30, tzinfo=UTC)
WEEK_CLOSE = datetime(2026, 11, 27, 18, 0, tzinfo=UTC)


class ClosedCalendar:
    """A calendar whose exchange never opens.

    NYSE has no week without a session, so the empty-week branch has no real
    date to be tested at. A stub is the honest way to reach it.
    """

    def sessions_for(self, exchange: str, day: date) -> Sequence[SessionWindow]:
        return ()

    def trading_day_of(self, exchange: str, instant: datetime) -> date:
        raise AssertionError("not reached")

    def next_session_open(self, exchange: str, instant: datetime) -> datetime:
        raise AssertionError("not reached")


@pytest.fixture
def calendar() -> PandasMarketCalendar:
    return PandasMarketCalendar(clock=ReplayClock(datetime(2026, 8, 13, tzinfo=UTC)))


def minute_bar(
    start: datetime,
    *,
    open_: str = "100",
    high: str = "101",
    low: str = "99",
    close: str = "100.5",
    volume: str = "10",
    session: SessionKind = SessionKind.REGULAR,
    session_date: date = NORMAL_DAY,
    is_final: bool = True,
    **overrides: object,
) -> Bar:
    fields: dict[str, object] = {
        "symbol": AAPL,
        "timeframe": Timeframe.M1,
        "start_time_utc": start,
        "end_time_utc": start + timedelta(minutes=1),
        "session_date": session_date,
        "session": session,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "source": "fake",
        "feed": Feed.SYNTHETIC,
        "adjustment_mode": AdjustmentMode.UNADJUSTED,
        "is_final": is_final,
    }
    return Bar(**{**fields, **overrides})  # type: ignore[arg-type]


def minutes(start: datetime, count: int, **kwargs: object) -> list[Bar]:
    return [minute_bar(start + timedelta(minutes=i), **kwargs) for i in range(count)]  # type: ignore[arg-type]


class TestSessionGridRegular:
    def test_hourly_grid_is_anchored_at_the_session_open(
        self, calendar: PandasMarketCalendar
    ) -> None:
        cells = [
            c
            for c in session_grid(calendar, NYSE, NORMAL_DAY, Timeframe.H1)
            if c.session is SessionKind.REGULAR
        ]
        # 09:30 ET, not the 13:00 UTC hour boundary — the whole point of ADR 0004 D4.
        assert cells[0].start == OPEN
        assert cells[0].end == OPEN + timedelta(hours=1)

    def test_hourly_regular_session_yields_seven_cells(
        self, calendar: PandasMarketCalendar
    ) -> None:
        cells = [
            c
            for c in session_grid(calendar, NYSE, NORMAL_DAY, Timeframe.H1)
            if c.session is SessionKind.REGULAR
        ]
        assert len(cells) == 7

    def test_last_regular_cell_is_truncated_at_the_close(
        self, calendar: PandasMarketCalendar
    ) -> None:
        cells = [
            c
            for c in session_grid(calendar, NYSE, NORMAL_DAY, Timeframe.H1)
            if c.session is SessionKind.REGULAR
        ]
        last = cells[-1]
        assert last.start == CLOSE - timedelta(minutes=30)
        assert last.end == CLOSE
        assert last.duration == timedelta(minutes=30)

    def test_cells_are_contiguous_and_non_overlapping(self, calendar: PandasMarketCalendar) -> None:
        cells = session_grid(calendar, NYSE, NORMAL_DAY, Timeframe.H1)
        for earlier, later in itertools.pairwise(cells):
            assert earlier.end == later.start

    def test_fifteen_minute_grid_matches_the_epoch_grid_on_a_normal_day(
        self, calendar: PandasMarketCalendar
    ) -> None:
        # 13:30 UTC is an exact multiple of 15 minutes, so session anchoring and
        # epoch alignment coincide here. Only 1h/4h actually differ.
        cells = [
            c
            for c in session_grid(calendar, NYSE, NORMAL_DAY, Timeframe.M15)
            if c.session is SessionKind.REGULAR
        ]
        assert len(cells) == 26
        assert all(c.duration == timedelta(minutes=15) for c in cells)


class TestSessionGridExtendedHours:
    def test_pre_market_cells_are_tagged_and_never_merged(
        self, calendar: PandasMarketCalendar
    ) -> None:
        cells = session_grid(calendar, NYSE, NORMAL_DAY, Timeframe.H1)
        pre = [c for c in cells if c.session is SessionKind.PRE]
        assert pre
        assert all(c.end <= OPEN for c in pre)
        assert pre[-1].end == OPEN

    def test_first_pre_market_cell_is_truncated_at_the_pre_open(
        self, calendar: PandasMarketCalendar
    ) -> None:
        pre = [
            c
            for c in session_grid(calendar, NYSE, NORMAL_DAY, Timeframe.H1)
            if c.session is SessionKind.PRE
        ]
        # Pre-market opens 09:00 UTC; the grid line above it is 09:30.
        assert pre[0].start == datetime(2026, 11, 25, 9, 0, tzinfo=UTC)
        assert pre[0].end == datetime(2026, 11, 25, 9, 30, tzinfo=UTC)

    def test_post_market_cells_follow_the_close(self, calendar: PandasMarketCalendar) -> None:
        post = [
            c
            for c in session_grid(calendar, NYSE, NORMAL_DAY, Timeframe.H1)
            if c.session is SessionKind.POST
        ]
        assert post[0].start == CLOSE
        assert post[-1].end == datetime(2026, 11, 26, 1, 0, tzinfo=UTC)

    def test_every_cell_carries_the_trading_day(self, calendar: PandasMarketCalendar) -> None:
        for cell in session_grid(calendar, NYSE, NORMAL_DAY, Timeframe.H1):
            assert cell.session_date == NORMAL_DAY


class TestSessionGridEdges:
    def test_holiday_has_no_grid(self, calendar: PandasMarketCalendar) -> None:
        assert session_grid(calendar, NYSE, HOLIDAY, Timeframe.H1) == ()

    def test_half_day_yields_four_hourly_cells(self, calendar: PandasMarketCalendar) -> None:
        cells = [
            c
            for c in session_grid(calendar, NYSE, HALF_DAY, Timeframe.H1)
            if c.session is SessionKind.REGULAR
        ]
        assert len(cells) == 4
        assert cells[-1].duration == timedelta(minutes=30)

    def test_daily_cell_spans_the_regular_session_only(
        self, calendar: PandasMarketCalendar
    ) -> None:
        cells = session_grid(calendar, NYSE, NORMAL_DAY, Timeframe.D1)
        assert len(cells) == 1
        assert (cells[0].start, cells[0].end) == (OPEN, CLOSE)
        assert cells[0].session is SessionKind.REGULAR


class TestSessionGridWeekly:
    """ADR 0004 D6: a `1W` bar opens at the session open of the week's first
    trading day and closes at the session close of its last."""

    def test_full_week_spans_monday_open_to_friday_close(
        self, calendar: PandasMarketCalendar
    ) -> None:
        cells = session_grid(calendar, NYSE, FULL_WEEK_MONDAY, Timeframe.W1)
        assert len(cells) == 1
        assert cells[0].start == datetime(2026, 11, 16, 14, 30, tzinfo=UTC)
        assert cells[0].end == datetime(2026, 11, 20, 21, 0, tzinfo=UTC)
        assert cells[0].session is SessionKind.REGULAR

    def test_every_day_of_the_week_yields_the_same_cell(
        self, calendar: PandasMarketCalendar
    ) -> None:
        """What lets `aggregate_bars` group daily bars keyed by their own
        `session_date` into one weekly cell without knowing about weeks."""
        cells = {
            session_grid(calendar, NYSE, FULL_WEEK_MONDAY + timedelta(days=offset), Timeframe.W1)
            for offset in range(7)
        }
        assert len(cells) == 1

    def test_the_cell_is_dated_to_the_weeks_first_trading_day(
        self, calendar: PandasMarketCalendar
    ) -> None:
        cells = session_grid(calendar, NYSE, FULL_WEEK_MONDAY, Timeframe.W1)
        assert cells[0].session_date == FULL_WEEK_MONDAY

    def test_a_holiday_shortens_the_week_rather_than_breaking_it(
        self, calendar: PandasMarketCalendar
    ) -> None:
        """Thanksgiving week: Thursday is shut and Friday closes at 13:00 ET.
        The weekly bar ends at the last close that happened, not at the one the
        calendar would have had on a full week."""
        cells = session_grid(calendar, NYSE, HOLIDAY, Timeframe.W1)
        assert len(cells) == 1
        assert cells[0].start == datetime(2026, 11, 23, 14, 30, tzinfo=UTC)
        assert cells[0].end == datetime(2026, 11, 27, 18, 0, tzinfo=UTC)

    def test_a_shut_monday_moves_the_week_open_to_tuesday(
        self, calendar: PandasMarketCalendar
    ) -> None:
        cells = session_grid(calendar, NYSE, MLK_MONDAY, Timeframe.W1)
        assert cells[0].start == datetime(2026, 1, 20, 14, 30, tzinfo=UTC)
        assert cells[0].session_date == date(2026, 1, 20)

    def test_a_week_with_no_sessions_has_no_grid(self) -> None:
        assert session_grid(ClosedCalendar(), NYSE, FULL_WEEK_MONDAY, Timeframe.W1) == ()


class TestAggregation:
    def test_empty_input_yields_nothing(self, calendar: PandasMarketCalendar) -> None:
        assert (
            aggregate_bars(
                [], target=Timeframe.H1, calendar=calendar, exchange=NYSE, watermark=CLOSE
            )
            == ()
        )

    def test_sixty_minutes_become_one_hourly_bar(self, calendar: PandasMarketCalendar) -> None:
        result = aggregate_bars(
            minutes(OPEN, 60),
            target=Timeframe.H1,
            calendar=calendar,
            exchange=NYSE,
            watermark=CLOSE,
        )
        assert len(result) == 1
        assert result[0].timeframe is Timeframe.H1
        assert result[0].start_time_utc == OPEN
        assert result[0].end_time_utc == OPEN + timedelta(hours=1)

    def test_ohlc_comes_from_the_right_bars(self, calendar: PandasMarketCalendar) -> None:
        source = [
            minute_bar(OPEN, open_="100", high="102", low="99", close="101"),
            minute_bar(OPEN + timedelta(minutes=1), open_="101", high="105", low="98", close="99"),
            minute_bar(OPEN + timedelta(minutes=2), open_="99", high="103", low="97", close="102"),
        ]
        bar = aggregate_bars(
            source, target=Timeframe.H1, calendar=calendar, exchange=NYSE, watermark=CLOSE
        )[0]
        assert bar.open == Decimal("100.00000000")  # first bar's open
        assert bar.close == Decimal("102.00000000")  # last bar's close
        assert bar.high == Decimal("105.00000000")
        assert bar.low == Decimal("97.00000000")

    def test_volume_is_summed(self, calendar: PandasMarketCalendar) -> None:
        source = minutes(OPEN, 3, volume="10")
        bar = aggregate_bars(
            source, target=Timeframe.H1, calendar=calendar, exchange=NYSE, watermark=CLOSE
        )[0]
        assert bar.volume == Decimal("30.00000000")

    def test_vwap_is_volume_weighted(self, calendar: PandasMarketCalendar) -> None:
        source = [
            minute_bar(OPEN, volume="100", vwap="100"),
            minute_bar(OPEN + timedelta(minutes=1), volume="300", vwap="101"),
        ]
        bar = aggregate_bars(
            source, target=Timeframe.H1, calendar=calendar, exchange=NYSE, watermark=CLOSE
        )[0]
        # (100*100 + 300*101) / 400 = 100.75 — weighted, not the plain mean of 100.5
        assert bar.vwap == Decimal("100.75000000")

    def test_vwap_is_dropped_when_a_source_bar_lacks_it(
        self, calendar: PandasMarketCalendar
    ) -> None:
        source = [
            minute_bar(OPEN, volume="100", vwap="100"),
            minute_bar(OPEN + timedelta(minutes=1), volume="100"),
        ]
        bar = aggregate_bars(
            source, target=Timeframe.H1, calendar=calendar, exchange=NYSE, watermark=CLOSE
        )[0]
        assert bar.vwap is None

    def test_trade_count_is_summed_when_present(self, calendar: PandasMarketCalendar) -> None:
        source = [
            minute_bar(OPEN, trade_count=5),
            minute_bar(OPEN + timedelta(minutes=1), trade_count=7),
        ]
        bar = aggregate_bars(
            source, target=Timeframe.H1, calendar=calendar, exchange=NYSE, watermark=CLOSE
        )[0]
        assert bar.trade_count == 12

    def test_full_session_aggregates_to_seven_hourly_bars(
        self, calendar: PandasMarketCalendar
    ) -> None:
        source = minutes(OPEN, 390)  # 6.5h of minutes
        result = aggregate_bars(
            source, target=Timeframe.H1, calendar=calendar, exchange=NYSE, watermark=CLOSE
        )
        assert len(result) == 7
        assert result[-1].duration == timedelta(minutes=30)
        assert result[-1].is_truncated

    def test_gaps_in_source_are_tolerated(self, calendar: PandasMarketCalendar) -> None:
        # Illiquid minutes simply do not print; the hour still aggregates.
        source = [minute_bar(OPEN), minute_bar(OPEN + timedelta(minutes=45))]
        result = aggregate_bars(
            source, target=Timeframe.H1, calendar=calendar, exchange=NYSE, watermark=CLOSE
        )
        assert len(result) == 1
        assert result[0].volume == Decimal("20.00000000")


class TestAggregationFinality:
    def test_bar_is_final_once_the_watermark_passes_the_cell_end(
        self, calendar: PandasMarketCalendar
    ) -> None:
        source = minutes(OPEN, 60)
        bar = aggregate_bars(
            source,
            target=Timeframe.H1,
            calendar=calendar,
            exchange=NYSE,
            watermark=OPEN + timedelta(hours=1),
        )[0]
        assert bar.is_final

    def test_bar_is_partial_while_the_watermark_is_inside_the_cell(
        self, calendar: PandasMarketCalendar
    ) -> None:
        source = minutes(OPEN, 30)
        bar = aggregate_bars(
            source,
            target=Timeframe.H1,
            calendar=calendar,
            exchange=NYSE,
            watermark=OPEN + timedelta(minutes=30),
        )[0]
        assert not bar.is_final

    def test_a_non_final_source_bar_keeps_the_target_partial(
        self, calendar: PandasMarketCalendar
    ) -> None:
        source = [*minutes(OPEN, 59), minute_bar(OPEN + timedelta(minutes=59), is_final=False)]
        bar = aggregate_bars(
            source, target=Timeframe.H1, calendar=calendar, exchange=NYSE, watermark=CLOSE
        )[0]
        assert not bar.is_final


class TestAggregationValidation:
    def test_mixed_feeds_are_rejected(self, calendar: PandasMarketCalendar) -> None:
        source = [
            minute_bar(OPEN, feed=Feed.IEX),
            minute_bar(OPEN + timedelta(minutes=1), feed=Feed.SIP),
        ]
        with pytest.raises(InvariantViolation, match="feed"):
            aggregate_bars(
                source, target=Timeframe.H1, calendar=calendar, exchange=NYSE, watermark=CLOSE
            )

    def test_mixed_adjustment_modes_are_rejected(self, calendar: PandasMarketCalendar) -> None:
        source = [
            minute_bar(OPEN, adjustment_mode=AdjustmentMode.ADJUSTED),
            minute_bar(OPEN + timedelta(minutes=1), adjustment_mode=AdjustmentMode.UNADJUSTED),
        ]
        with pytest.raises(InvariantViolation, match="adjustment"):
            aggregate_bars(
                source, target=Timeframe.H1, calendar=calendar, exchange=NYSE, watermark=CLOSE
            )

    def test_mixed_symbols_are_rejected(self, calendar: PandasMarketCalendar) -> None:
        source = [
            minute_bar(OPEN),
            minute_bar(OPEN + timedelta(minutes=1), symbol=ticker("MSFT")),
        ]
        with pytest.raises(InvariantViolation, match="symbol"):
            aggregate_bars(
                source, target=Timeframe.H1, calendar=calendar, exchange=NYSE, watermark=CLOSE
            )

    def test_source_must_be_finer_than_the_target(self, calendar: PandasMarketCalendar) -> None:
        hourly = minute_bar(OPEN, timeframe=Timeframe.H1, end_time_utc=OPEN + timedelta(hours=1))
        with pytest.raises(InvariantViolation, match="finer"):
            aggregate_bars(
                [hourly],
                target=Timeframe.M15,
                calendar=calendar,
                exchange=NYSE,
                watermark=CLOSE,
            )

    def test_every_canonical_timeframe_pair_divides_evenly(self) -> None:
        """The "whole multiple" guard in `_check_timeframes` cannot fire today.

        Every canonical intraday timeframe divides every coarser one, so the
        check is insurance against a future addition (a 90m or 2h bar would
        break it). This test is what makes that addition fail loudly here rather
        than silently misalign a grid.
        """
        intraday = [tf for tf in Timeframe if tf.is_intraday]
        for finer in intraday:
            for coarser in intraday:
                if finer.nominal_duration < coarser.nominal_duration:
                    assert coarser.nominal_duration % finer.nominal_duration == timedelta(0), (
                        f"{coarser.code} is not a whole multiple of {finer.code}"
                    )

    def test_a_bar_outside_any_session_is_rejected(self, calendar: PandasMarketCalendar) -> None:
        stray = minute_bar(datetime(2026, 11, 25, 4, 0, tzinfo=UTC))  # before pre-market
        with pytest.raises(InvariantViolation, match="no session"):
            aggregate_bars(
                [stray], target=Timeframe.H1, calendar=calendar, exchange=NYSE, watermark=CLOSE
            )

    def test_partial_source_bars_may_not_straddle_a_cell_boundary(
        self, calendar: PandasMarketCalendar
    ) -> None:
        straddling = minute_bar(
            OPEN + timedelta(minutes=59),
            timeframe=Timeframe.M5,
            end_time_utc=OPEN + timedelta(minutes=64),
        )
        with pytest.raises(InvariantViolation, match="straddle"):
            aggregate_bars(
                [straddling],
                target=Timeframe.H1,
                calendar=calendar,
                exchange=NYSE,
                watermark=CLOSE,
            )


def daily_bar(
    session_date: date,
    *,
    open_: str = "100",
    high: str = "101",
    low: str = "99",
    close: str = "100.5",
    volume: str = "1000",
    is_final: bool = True,
    calendar: PandasMarketCalendar,
) -> Bar:
    """A `1D` bar placed on the real session, the way the daily path builds one."""
    regular = next(
        w for w in calendar.sessions_for(NYSE, session_date) if w.kind is SessionKind.REGULAR
    )
    return Bar(
        symbol=AAPL,
        timeframe=Timeframe.D1,
        start_time_utc=regular.open_utc,
        end_time_utc=regular.close_utc,
        session_date=session_date,
        session=SessionKind.REGULAR,
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
        source="fake",
        feed=Feed.SYNTHETIC,
        adjustment_mode=AdjustmentMode.UNADJUSTED,
        is_final=is_final,
    )


class TestWeeklyAggregation:
    """`1W` folds `1D`, and only `1D` (ADR 0021)."""

    def test_a_week_of_daily_bars_becomes_one_weekly_bar(
        self, calendar: PandasMarketCalendar
    ) -> None:
        days = [
            daily_bar(date(2026, 11, 16) + timedelta(days=i), calendar=calendar) for i in range(5)
        ]
        weekly = aggregate_bars(
            days,
            target=Timeframe.W1,
            calendar=calendar,
            exchange=NYSE,
            watermark=datetime(2026, 11, 21, tzinfo=UTC),
        )
        assert len(weekly) == 1
        assert weekly[0].timeframe is Timeframe.W1
        assert weekly[0].start_time_utc == datetime(2026, 11, 16, 14, 30, tzinfo=UTC)
        assert weekly[0].end_time_utc == datetime(2026, 11, 20, 21, 0, tzinfo=UTC)
        assert weekly[0].session_date == date(2026, 11, 16)
        assert weekly[0].is_final

    def test_ohlc_comes_from_the_first_and_last_session_of_the_week(
        self, calendar: PandasMarketCalendar
    ) -> None:
        days = [
            daily_bar(
                date(2026, 11, 16), open_="100", high="103", low="98", close="99", calendar=calendar
            ),
            daily_bar(
                date(2026, 11, 17),
                open_="99",
                high="110",
                low="97",
                close="105",
                calendar=calendar,
            ),
            daily_bar(
                date(2026, 11, 18),
                open_="105",
                high="106",
                low="90",
                close="92",
                calendar=calendar,
            ),
        ]
        weekly = aggregate_bars(
            days,
            target=Timeframe.W1,
            calendar=calendar,
            exchange=NYSE,
            watermark=datetime(2026, 11, 21, tzinfo=UTC),
        )[0]
        assert (weekly.open, weekly.close) == (Decimal("100"), Decimal("92"))
        assert (weekly.high, weekly.low) == (Decimal("110"), Decimal("90"))
        assert weekly.volume == Decimal("3000")

    def test_two_weeks_do_not_merge(self, calendar: PandasMarketCalendar) -> None:
        days = [
            daily_bar(date(2026, 11, 19), calendar=calendar),
            daily_bar(date(2026, 11, 20), calendar=calendar),
            daily_bar(date(2026, 11, 23), calendar=calendar),
            daily_bar(date(2026, 11, 24), calendar=calendar),
        ]
        weekly = aggregate_bars(
            days,
            target=Timeframe.W1,
            calendar=calendar,
            exchange=NYSE,
            watermark=datetime(2026, 11, 28, tzinfo=UTC),
        )
        assert [bar.session_date for bar in weekly] == [date(2026, 11, 16), date(2026, 11, 23)]

    def test_a_holiday_week_closes_at_the_half_day(self, calendar: PandasMarketCalendar) -> None:
        days = [
            daily_bar(day, calendar=calendar)
            for day in (
                date(2026, 11, 23),
                date(2026, 11, 24),
                date(2026, 11, 25),
                date(2026, 11, 27),
            )
        ]
        weekly = aggregate_bars(
            days,
            target=Timeframe.W1,
            calendar=calendar,
            exchange=NYSE,
            watermark=datetime(2026, 11, 28, tzinfo=UTC),
        )[0]
        assert (weekly.start_time_utc, weekly.end_time_utc) == (WEEK_OPEN, WEEK_CLOSE)

    def test_the_week_in_progress_is_not_final(self, calendar: PandasMarketCalendar) -> None:
        """Wednesday of a live week: three sessions in, and a weekly bar that
        cannot be treated as closed (CLAUDE.md §12)."""
        days = [
            daily_bar(date(2026, 11, 16) + timedelta(days=i), calendar=calendar) for i in range(3)
        ]
        weekly = aggregate_bars(
            days,
            target=Timeframe.W1,
            calendar=calendar,
            exchange=NYSE,
            watermark=datetime(2026, 11, 18, 21, 0, tzinfo=UTC),
        )[0]
        assert not weekly.is_final

    def test_an_unfinished_daily_bar_keeps_the_week_open(
        self, calendar: PandasMarketCalendar
    ) -> None:
        days = [
            daily_bar(date(2026, 11, 16) + timedelta(days=i), calendar=calendar) for i in range(4)
        ] + [daily_bar(date(2026, 11, 20), is_final=False, calendar=calendar)]
        weekly = aggregate_bars(
            days,
            target=Timeframe.W1,
            calendar=calendar,
            exchange=NYSE,
            watermark=datetime(2026, 11, 21, tzinfo=UTC),
        )[0]
        assert not weekly.is_final

    def test_only_daily_bars_may_be_folded_into_a_week(
        self, calendar: PandasMarketCalendar
    ) -> None:
        """A minute bar divides seven days evenly and would place inside the
        weekly cell, so nothing but this rule stops a pre-market print from
        entering a regular-session weekly bar (ADR 0021)."""
        with pytest.raises(InvariantViolation, match="1D"):
            aggregate_bars(
                minutes(OPEN, 5),
                target=Timeframe.W1,
                calendar=calendar,
                exchange=NYSE,
                watermark=CLOSE,
            )


class TestFoldBars:
    """The whole path from a provider's native bars to a target — ADR 0021.

    `aggregate_bars` folds one step. This is the step *sequence*, which exists
    because `1W` cannot be reached in one fold and because both the Alpaca
    adapter and `AggregatingBarSource` were otherwise going to spell the same
    chain out twice.
    """

    def test_a_single_step_target_matches_aggregate_bars(
        self, calendar: PandasMarketCalendar
    ) -> None:
        source = minutes(OPEN, 60)
        assert fold_bars(
            source, target=Timeframe.H1, calendar=calendar, exchange=NYSE, watermark=CLOSE
        ) == aggregate_bars(
            source, target=Timeframe.H1, calendar=calendar, exchange=NYSE, watermark=CLOSE
        )

    def test_minutes_reach_a_week_through_the_daily_grid(
        self, calendar: PandasMarketCalendar
    ) -> None:
        source = [
            minute_bar(
                datetime(2026, 11, 16 + offset, 14, 30, tzinfo=UTC),
                session_date=date(2026, 11, 16 + offset),
                high=str(101 + offset),
            )
            for offset in range(5)
        ]
        weekly = fold_bars(
            source,
            target=Timeframe.W1,
            calendar=calendar,
            exchange=NYSE,
            watermark=datetime(2026, 11, 21, tzinfo=UTC),
        )
        assert len(weekly) == 1
        assert weekly[0].timeframe is Timeframe.W1
        assert weekly[0].high == Decimal("105")
        assert weekly[0].end_time_utc == datetime(2026, 11, 20, 21, 0, tzinfo=UTC)

    def test_daily_bars_fold_straight_into_the_week(self, calendar: PandasMarketCalendar) -> None:
        days = [
            daily_bar(date(2026, 11, 16) + timedelta(days=i), calendar=calendar) for i in range(5)
        ]
        assert fold_bars(
            days,
            target=Timeframe.W1,
            calendar=calendar,
            exchange=NYSE,
            watermark=datetime(2026, 11, 21, tzinfo=UTC),
        ) == aggregate_bars(
            days,
            target=Timeframe.W1,
            calendar=calendar,
            exchange=NYSE,
            watermark=datetime(2026, 11, 21, tzinfo=UTC),
        )

    def test_extended_hours_minutes_never_enter_a_daily_bar(
        self, calendar: PandasMarketCalendar
    ) -> None:
        pre = minute_bar(datetime(2026, 11, 25, 13, 0, tzinfo=UTC), session=SessionKind.PRE)
        daily = fold_bars(
            [pre, *minutes(OPEN, 5)],
            target=Timeframe.D1,
            calendar=calendar,
            exchange=NYSE,
            watermark=datetime(2026, 11, 26, tzinfo=UTC),
        )
        assert len(daily) == 1
        assert daily[0].start_time_utc == OPEN

    def test_extended_hours_minutes_never_enter_a_weekly_bar(
        self, calendar: PandasMarketCalendar
    ) -> None:
        """The reason the chain goes through `1D` at all: a weekly cell spans
        the nights, so a pre-market print would place inside it."""
        pre = minute_bar(
            datetime(2026, 11, 17, 13, 0, tzinfo=UTC),
            session=SessionKind.PRE,
            session_date=date(2026, 11, 17),
            high="999",
        )
        source = [
            pre,
            *[
                minute_bar(
                    datetime(2026, 11, 16 + offset, 14, 30, tzinfo=UTC),
                    session_date=date(2026, 11, 16 + offset),
                )
                for offset in range(5)
            ],
        ]
        weekly = fold_bars(
            source,
            target=Timeframe.W1,
            calendar=calendar,
            exchange=NYSE,
            watermark=datetime(2026, 11, 21, tzinfo=UTC),
        )
        assert weekly[0].high == Decimal("101")

    def test_nothing_in_yields_nothing_out(self, calendar: PandasMarketCalendar) -> None:
        assert (
            fold_bars([], target=Timeframe.W1, calendar=calendar, exchange=NYSE, watermark=CLOSE)
            == ()
        )


class TestNextCloseAfter:
    """When the next bar of a series can possibly close — ADR 0023.

    The monitor asks this to decide whether a series has anything new to fetch.
    Getting it *late* means a missed bar, so every case here is about the
    boundary rather than the middle.
    """

    def test_inside_a_session_it_is_the_next_grid_line(
        self, calendar: PandasMarketCalendar
    ) -> None:
        assert next_close_after(
            calendar, NYSE, timeframe=Timeframe.H1, instant=OPEN + timedelta(hours=1)
        ) == OPEN + timedelta(hours=2)

    def test_a_truncated_last_cell_is_not_a_whole_period_away(
        self, calendar: PandasMarketCalendar
    ) -> None:
        """The case that makes "last close + nominal duration" wrong. The final
        regular hourly cell is 20:30–21:00, half an hour long; waiting an hour
        after 20:30 would notice the 21:00 bar thirty minutes late."""
        assert next_close_after(calendar, NYSE, timeframe=Timeframe.H1, instant=CLOSE) > CLOSE
        assert (
            next_close_after(
                calendar, NYSE, timeframe=Timeframe.H1, instant=CLOSE - timedelta(minutes=30)
            )
            == CLOSE
        )

    def test_the_close_instant_itself_belongs_to_the_cell_that_ended(
        self, calendar: PandasMarketCalendar
    ) -> None:
        """Strictly after: a bar that ended at 21:00 has nothing more to wait for
        at 21:00, so the answer is the *following* cell."""
        assert next_close_after(calendar, NYSE, timeframe=Timeframe.D1, instant=CLOSE) == datetime(
            2026, 11, 27, 18, 0, tzinfo=UTC
        )

    def test_a_holiday_is_skipped_rather_than_answered_with_a_shut_day(
        self, calendar: PandasMarketCalendar
    ) -> None:
        """Wednesday's daily bar closes at 21:00; Thursday is Thanksgiving, so
        the next daily close is Friday's half-day at 13:00 ET."""
        assert next_close_after(calendar, NYSE, timeframe=Timeframe.D1, instant=CLOSE) == datetime(
            2026, 11, 27, 18, 0, tzinfo=UTC
        )

    def test_a_weekly_series_waits_a_week(self, calendar: PandasMarketCalendar) -> None:
        assert next_close_after(
            calendar,
            NYSE,
            timeframe=Timeframe.W1,
            instant=datetime(2026, 11, 20, 21, 0, tzinfo=UTC),
        ) == datetime(2026, 11, 27, 18, 0, tzinfo=UTC)

    def test_mid_week_the_answer_is_this_weeks_close(self, calendar: PandasMarketCalendar) -> None:
        assert next_close_after(calendar, NYSE, timeframe=Timeframe.W1, instant=OPEN) == datetime(
            2026, 11, 27, 18, 0, tzinfo=UTC
        )

    def test_overnight_the_answer_is_in_the_next_session(
        self, calendar: PandasMarketCalendar
    ) -> None:
        """The saving this exists for: between the post-market close and the next
        pre-market open there is nothing to ask any provider for."""
        after_hours = datetime(2026, 11, 25, 3, 0, tzinfo=UTC)
        found = next_close_after(calendar, NYSE, timeframe=Timeframe.M5, instant=after_hours)
        assert found is not None
        assert found > after_hours

    def test_nothing_within_the_horizon_is_reported_as_unknown(self) -> None:
        """An exchange that never opens has no next close. `None` means "no
        answer", and the monitor treats that as due — never skipping a series on
        the strength of not knowing."""
        assert (
            next_close_after(ClosedCalendar(), NYSE, timeframe=Timeframe.M5, instant=OPEN) is None
        )


class TestAggregationDeterminism:
    def test_input_order_does_not_change_the_result(self, calendar: PandasMarketCalendar) -> None:
        source = minutes(OPEN, 60)
        forward = aggregate_bars(
            source, target=Timeframe.H1, calendar=calendar, exchange=NYSE, watermark=CLOSE
        )
        backward = aggregate_bars(
            list(reversed(source)),
            target=Timeframe.H1,
            calendar=calendar,
            exchange=NYSE,
            watermark=CLOSE,
        )
        assert forward == backward

    def test_repeated_aggregation_is_identical(self, calendar: PandasMarketCalendar) -> None:
        source = minutes(OPEN, 390)
        first = aggregate_bars(
            source, target=Timeframe.H1, calendar=calendar, exchange=NYSE, watermark=CLOSE
        )
        second = aggregate_bars(
            source, target=Timeframe.H1, calendar=calendar, exchange=NYSE, watermark=CLOSE
        )
        assert first == second
