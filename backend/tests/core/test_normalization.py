"""BarSeries — duplicates, out-of-order arrival, revisions, gaps.

specs/03-market-data.md responsibilities 4 and 7, and the data test list in
specs/14-testing.md.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from alphagate.core.bar import AdjustmentMode, Bar, Feed
from alphagate.core.errors import InvariantViolation
from alphagate.core.identifiers import Ticker, ticker
from alphagate.core.normalization import BarSeries, IngestOutcome
from alphagate.core.time_model import SessionKind, Timeframe

AAPL: Ticker = ticker("AAPL")
DAY = date(2026, 11, 25)
OPEN = datetime(2026, 11, 25, 14, 30, tzinfo=UTC)


def hour_bar(start: datetime, **overrides: object) -> Bar:
    fields: dict[str, object] = {
        "symbol": AAPL,
        "timeframe": Timeframe.H1,
        "start_time_utc": start,
        "end_time_utc": start + timedelta(hours=1),
        "session_date": DAY,
        "session": SessionKind.REGULAR,
        "open": "100",
        "high": "101",
        "low": "99",
        "close": "100.5",
        "volume": "1000",
        "source": "fake",
        "feed": Feed.SYNTHETIC,
        "adjustment_mode": AdjustmentMode.UNADJUSTED,
        "is_final": True,
    }
    return Bar(**{**fields, **overrides})  # type: ignore[arg-type]


def series() -> BarSeries:
    return BarSeries(
        symbol=AAPL,
        timeframe=Timeframe.H1,
        feed=Feed.SYNTHETIC,
        adjustment_mode=AdjustmentMode.UNADJUSTED,
    )


class TestAccepting:
    def test_starts_empty(self) -> None:
        assert len(series()) == 0
        assert series().bars == ()

    def test_accepts_a_bar(self) -> None:
        s = series()
        assert s.ingest(hour_bar(OPEN)) is IngestOutcome.ACCEPTED
        assert len(s) == 1

    def test_keeps_bars_in_chronological_order(self) -> None:
        s = series()
        for offset in (2, 0, 1):
            s.ingest(hour_bar(OPEN + timedelta(hours=offset)))
        assert [b.start_time_utc for b in s.bars] == [
            OPEN,
            OPEN + timedelta(hours=1),
            OPEN + timedelta(hours=2),
        ]

    def test_latest_returns_the_most_recent_bar(self) -> None:
        s = series()
        s.ingest(hour_bar(OPEN))
        s.ingest(hour_bar(OPEN + timedelta(hours=1)))
        assert s.latest is not None
        assert s.latest.start_time_utc == OPEN + timedelta(hours=1)

    def test_latest_is_none_when_empty(self) -> None:
        assert series().latest is None


class TestSeriesIdentity:
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("symbol", ticker("MSFT")),
            ("feed", Feed.IEX),
            ("adjustment_mode", AdjustmentMode.ADJUSTED),
        ],
    )
    def test_a_bar_from_another_series_is_rejected(self, field: str, value: object) -> None:
        s = series()
        with pytest.raises(InvariantViolation, match=field):
            s.ingest(hour_bar(OPEN, **{field: value}))

    def test_a_15m_bar_does_not_belong_in_an_hourly_series(self) -> None:
        s = series()
        with pytest.raises(InvariantViolation, match="timeframe"):
            s.ingest(
                hour_bar(OPEN, timeframe=Timeframe.M15, end_time_utc=OPEN + timedelta(minutes=15))
            )


class TestDuplicates:
    def test_an_identical_repeat_is_ignored(self) -> None:
        s = series()
        bar = hour_bar(OPEN)
        s.ingest(bar)
        assert s.ingest(bar) is IngestOutcome.DUPLICATE
        assert len(s) == 1
        assert s.duplicate_count == 1

    def test_a_conflicting_repeat_on_a_closed_window_is_rejected(self) -> None:
        s = series()
        s.ingest(hour_bar(OPEN, close="100.5"))
        with pytest.raises(InvariantViolation, match="revision 0"):
            s.ingest(hour_bar(OPEN, close="100.75"))

    def test_the_conflict_message_names_the_bar(self) -> None:
        s = series()
        s.ingest(hour_bar(OPEN))
        with pytest.raises(InvariantViolation, match="14:30"):
            s.ingest(hour_bar(OPEN, close="100.75"))


class TestPartialBars:
    def test_a_forming_bar_is_updated_in_place(self) -> None:
        # The ordinary intrabar case: same window, no revision bump, new values.
        s = series()
        s.ingest(hour_bar(OPEN, close="100.5", is_final=False))
        assert s.ingest(hour_bar(OPEN, close="100.75", is_final=False)) is IngestOutcome.UPDATED
        assert len(s) == 1
        assert s.bars[0].close == hour_bar(OPEN, close="100.75").close

    def test_closing_a_forming_bar_needs_no_revision_bump(self) -> None:
        s = series()
        s.ingest(hour_bar(OPEN, is_final=False))
        assert s.ingest(hour_bar(OPEN, close="100.75", is_final=True)) is IngestOutcome.UPDATED
        assert s.bars[0].is_final

    def test_an_update_is_not_counted_as_a_revision(self) -> None:
        s = series()
        s.ingest(hour_bar(OPEN, is_final=False))
        s.ingest(hour_bar(OPEN, close="100.75", is_final=True))
        assert s.revision_count == 0


class TestRevisions:
    def test_a_higher_revision_replaces_the_bar(self) -> None:
        s = series()
        s.ingest(hour_bar(OPEN, close="100.5"))
        assert s.ingest(hour_bar(OPEN, close="100.75", revision=1)) is IngestOutcome.REVISED
        assert len(s) == 1
        assert s.bars[0].close == hour_bar(OPEN, close="100.75").close
        assert s.revision_count == 1

    def test_a_stale_revision_is_ignored(self) -> None:
        s = series()
        s.ingest(hour_bar(OPEN, close="100.75", revision=2))
        assert s.ingest(hour_bar(OPEN, close="100.5", revision=1)) is IngestOutcome.STALE
        assert s.bars[0].revision == 2

    def test_finalizing_a_partial_bar_is_a_revision(self) -> None:
        s = series()
        s.ingest(hour_bar(OPEN, is_final=False))
        assert s.ingest(hour_bar(OPEN, is_final=True, revision=1)) is IngestOutcome.REVISED
        assert s.bars[0].is_final


class TestOutOfOrder:
    def test_a_late_bar_is_accepted_and_counted(self) -> None:
        s = series()
        s.ingest(hour_bar(OPEN + timedelta(hours=1)))
        assert s.ingest(hour_bar(OPEN)) is IngestOutcome.ACCEPTED
        assert s.out_of_order_count == 1
        assert s.bars[0].start_time_utc == OPEN

    def test_in_order_arrival_is_not_counted(self) -> None:
        s = series()
        s.ingest(hour_bar(OPEN))
        s.ingest(hour_bar(OPEN + timedelta(hours=1)))
        assert s.out_of_order_count == 0

    def test_a_revision_of_an_older_bar_is_not_out_of_order(self) -> None:
        s = series()
        s.ingest(hour_bar(OPEN))
        s.ingest(hour_bar(OPEN + timedelta(hours=1)))
        s.ingest(hour_bar(OPEN, close="100.75", revision=1))
        assert s.out_of_order_count == 0


class TestGaps:
    def test_no_gaps_when_every_cell_is_present(self) -> None:
        s = series()
        expected = [OPEN, OPEN + timedelta(hours=1), OPEN + timedelta(hours=2)]
        for start in expected:
            s.ingest(hour_bar(start))
        assert s.missing_starts(expected) == ()

    def test_missing_cells_are_reported_in_order(self) -> None:
        s = series()
        s.ingest(hour_bar(OPEN))
        s.ingest(hour_bar(OPEN + timedelta(hours=2)))
        expected = [OPEN + timedelta(hours=i) for i in range(4)]
        assert s.missing_starts(expected) == (
            OPEN + timedelta(hours=1),
            OPEN + timedelta(hours=3),
        )

    def test_a_partial_bar_does_not_count_as_present(self) -> None:
        s = series()
        s.ingest(hour_bar(OPEN, is_final=False))
        assert s.missing_starts([OPEN]) == (OPEN,)


class TestSnapshot:
    def test_bars_is_an_immutable_snapshot(self) -> None:
        s = series()
        s.ingest(hour_bar(OPEN))
        snapshot = s.bars
        s.ingest(hour_bar(OPEN + timedelta(hours=1)))
        assert len(snapshot) == 1
        assert len(s.bars) == 2

    def test_final_bars_excludes_the_partial_one(self) -> None:
        s = series()
        s.ingest(hour_bar(OPEN))
        s.ingest(hour_bar(OPEN + timedelta(hours=1), is_final=False))
        assert len(s.final_bars) == 1
        assert s.final_bars[0].start_time_utc == OPEN


class TestDeterminism:
    def test_arrival_order_does_not_change_the_result(self) -> None:
        starts = [OPEN + timedelta(hours=i) for i in range(5)]

        forward = series()
        for start in starts:
            forward.ingest(hour_bar(start))

        shuffled = series()
        for index in (3, 0, 4, 1, 2):
            shuffled.ingest(hour_bar(starts[index]))

        assert forward.bars == shuffled.bars
