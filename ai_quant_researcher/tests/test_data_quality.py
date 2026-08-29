"""Data quality checks.

These exist because of a real incident, not a hypothetical one. Pulling SPY
daily bars from Alpaca's IEX feed returned 1530 bars over a window containing
roughly 1960 sessions, with a single 634-day hole in the middle. Nothing
downstream noticed: the backtester indexes positionally, so a two-year gap is
simply two adjacent bars, and a strategy is scored on a history that never
happened.

The checks are cheap and the failure they catch is expensive, so they run on
every pull rather than on request.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from aqr.data.bars import Bars
from aqr.data.quality import inspect

SESSION = timedelta(days=1)


def _bars(
    days: list[int],
    *,
    volume: float = 1e6,
    close: float = 100.0,
    timeframe: str = "1D",
) -> Bars:
    base = datetime(2024, 1, 1, tzinfo=UTC)
    stamps = np.array(
        [int((base + timedelta(days=d)).timestamp()) for d in days], dtype=np.int64
    )
    n = len(days)
    return Bars(
        symbol="TEST",
        timeframe=timeframe,
        event_time=stamps,
        open=np.full(n, close - 1.0),
        high=np.full(n, close + 1.0),
        low=np.full(n, close - 2.0),
        close=np.full(n, close),
        volume=np.full(n, volume),
    )


class TestContinuity:
    def test_a_clean_weekday_series_reports_no_gap(self) -> None:
        # Mon-Fri for three weeks, weekends absent as they should be.
        days = [d for d in range(21) if (datetime(2024, 1, 1) + timedelta(days=d)).weekday() < 5]
        report = inspect(_bars(days))
        assert report.ok
        assert report.max_gap_days <= 3

    def test_a_weekend_is_not_a_gap(self) -> None:
        # Friday to Monday is three calendar days and entirely normal.
        report = inspect(_bars([4, 7]))
        assert report.ok

    def test_a_long_holiday_weekend_is_not_a_gap(self) -> None:
        # Thursday to the following Monday: a Friday holiday, four days.
        report = inspect(_bars([3, 7]))
        assert report.ok

    def test_a_two_year_hole_is_reported(self) -> None:
        report = inspect(_bars([0, 1, 2, 636, 637]))
        assert not report.ok
        assert report.max_gap_days >= 630
        assert "634" in str(report) or "633" in str(report)

    def test_the_gap_is_located_so_it_can_be_refetched(self) -> None:
        report = inspect(_bars([0, 1, 300]))
        assert len(report.gaps) == 1
        start, end, _ = report.gaps[0]
        assert start.date() == datetime(2024, 1, 2, tzinfo=UTC).date()
        assert end.date() == datetime(2024, 10, 27, tzinfo=UTC).date()


class TestKnownClosures:
    def test_hurricane_sandy_is_not_a_data_gap(self) -> None:
        """The market shut for two consecutive sessions in October 2012. Every
        clean series from 2010 on contains this five-day span, and a check that
        flags it teaches its reader to ignore it."""
        stamps = [
            int(datetime(2012, 10, d, tzinfo=UTC).timestamp()) for d in (25, 26, 31)
        ]
        bars = Bars(
            symbol="TEST",
            timeframe="1D",
            event_time=np.array(stamps, dtype=np.int64),
            open=np.full(3, 99.0),
            high=np.full(3, 101.0),
            low=np.full(3, 98.0),
            close=np.full(3, 100.0),
            volume=np.full(3, 1e6),
        )
        assert inspect(bars).gaps == []

    def test_a_five_day_hole_with_no_closure_behind_it_is_still_a_gap(self) -> None:
        stamps = [
            int(datetime(2014, 10, d, tzinfo=UTC).timestamp()) for d in (25, 26, 31)
        ]
        bars = Bars(
            symbol="TEST",
            timeframe="1D",
            event_time=np.array(stamps, dtype=np.int64),
            open=np.full(3, 99.0),
            high=np.full(3, 101.0),
            low=np.full(3, 98.0),
            close=np.full(3, 100.0),
            volume=np.full(3, 1e6),
        )
        assert len(inspect(bars).gaps) == 1


class TestCoverage:
    def test_a_series_missing_a_quarter_of_its_sessions_is_flagged(self) -> None:
        """Bars can be dense enough to hide gaps individually and still be
        missing a fifth of the history."""
        days = [d for d in range(0, 400, 2)]  # every other calendar day
        report = inspect(_bars(days))
        assert report.expected_sessions > report.bar_count
        assert report.coverage < 0.75
        assert not report.ok

    def test_coverage_is_not_computed_for_intraday(self) -> None:
        # The expected count depends on the session length and the venue's
        # calendar; guessing it would produce a warning nobody could act on.
        report = inspect(_bars([0, 1, 2], timeframe="5m"))
        assert report.coverage is None


class TestDegenerateBars:
    def test_zero_volume_bars_are_counted(self) -> None:
        good = _bars([0, 1, 2])
        holed = Bars(
            symbol=good.symbol,
            timeframe=good.timeframe,
            event_time=good.event_time,
            open=good.open,
            high=good.high,
            low=good.low,
            close=good.close,
            volume=np.array([1e6, 0.0, 0.0]),
        )
        assert inspect(holed).zero_volume_bars == 2

    def test_non_positive_prices_are_fatal_not_advisory(self) -> None:
        good = _bars([0, 1, 2])
        broken = Bars(
            symbol=good.symbol,
            timeframe=good.timeframe,
            event_time=good.event_time,
            open=good.open,
            high=good.high,
            low=np.array([99.0, 0.0, 98.0]),
            close=good.close,
            volume=good.volume,
        )
        report = inspect(broken)
        assert not report.ok
        assert report.non_positive_prices == 1

    def test_an_impossible_overnight_move_is_flagged_as_an_unadjusted_split(self) -> None:
        """A 4:1 split in unadjusted bars looks like a -75% return the strategy
        never took, and a mean-reversion rule will find it every time."""
        base = _bars([0, 1, 2])
        split = Bars(
            symbol=base.symbol,
            timeframe=base.timeframe,
            event_time=base.event_time,
            open=np.array([100.0, 100.0, 25.0]),
            high=np.array([101.0, 101.0, 26.0]),
            low=np.array([99.0, 99.0, 24.0]),
            close=np.array([100.0, 100.0, 25.0]),
            volume=base.volume,
        )
        report = inspect(split)
        assert report.suspected_splits == 1
        assert not report.ok


class TestReporting:
    def test_a_clean_report_is_short_and_says_so(self) -> None:
        days = [d for d in range(21) if (datetime(2024, 1, 1) + timedelta(days=d)).weekday() < 5]
        assert "ok" in str(inspect(_bars(days))).lower()

    def test_an_empty_series_is_not_ok(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            inspect(_bars([]))


class TestShortOfRequest:
    """A provider that returns less history than it was asked for.

    Measured, not imagined. An IBKR pull for 2005 onward returned PEP starting
    2017-12-20 and AZN starting 2026-02-02 -- 8 years and 7 months against the
    22 requested -- and every one of those series passed every check, because
    internally they are complete. The window was short, not broken, and nothing
    said so. Research then runs on a fifth of the history someone believes they
    have.

    Not an error: PLTR listing in 2020 is a fact about PLTR. It has to be
    visible, which is a different requirement from being fatal.
    """

    def test_a_series_starting_when_requested_is_not_flagged(self) -> None:
        days = list(range(0, 400, 1))
        report = inspect(_bars(days), requested_start=datetime(2024, 1, 1, tzinfo=UTC))
        assert report.starts_late_days == 0

    def test_a_series_starting_a_decade_late_says_by_how_much(self) -> None:
        report = inspect(
            _bars(list(range(0, 400))), requested_start=datetime(2014, 1, 1, tzinfo=UTC)
        )
        assert report.starts_late_days > 3600

    def test_being_short_is_reported_but_is_not_a_failure(self) -> None:
        # A listing date is legitimate. Refusing to cache a symbol for having
        # one would drop every recent IPO from the universe.
        report = inspect(
            _bars(list(range(0, 400))), requested_start=datetime(2014, 1, 1, tzinfo=UTC)
        )
        assert report.ok
        assert "short" in str(report).lower() or "late" in str(report).lower()

    def test_without_a_requested_start_nothing_is_claimed(self) -> None:
        report = inspect(_bars(list(range(0, 400))))
        assert report.starts_late_days == 0
        assert "late" not in str(report).lower()
