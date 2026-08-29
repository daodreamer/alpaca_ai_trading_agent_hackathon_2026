"""Comparing two vendors' idea of the same instrument.

Written because eyeballing stopped scaling. An IBKR ``ADJUSTED_LAST`` pull of
the NASDAQ-50 produced four unadjusted 2:1 splits (AAPL 2005-02-28, ADBE
2005-05-24, ORLY 2005-06-16, SBUX 2005-10-24 -- every date an exact split date),
a 4.01x discontinuity in FTNT, and a 910-day hole in TMUS. Each was found by
reading a list and recognising a date, which works for fifty symbols and for
nobody's benefit at five hundred.

The invariant that makes this mechanical:

    Two correctly adjusted series for the same instrument may disagree about the
    *price level* -- different adjustment bases, different dividend treatment --
    but they cannot disagree about a *daily return*.

So the comparison is on returns, never on prices. A day where one vendor says
-49.7% and the other says +0.3% is not a rounding difference or a stale close.
It is one of them failing to adjust a split, and the disagreement says which
dates to look at without anyone having to know the corporate calendar.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from aqr.data.bars import Bars
from aqr.data.cross_source import compare


def _series(closes: list[float], *, start_day: int = 0, symbol: str = "TEST") -> Bars:
    base = datetime(2020, 1, 1, tzinfo=UTC)
    n = len(closes)
    stamps = np.array(
        [int((base + timedelta(days=start_day + i)).timestamp()) for i in range(n)],
        dtype=np.int64,
    )
    c = np.array(closes, dtype=np.float64)
    return Bars(
        symbol=symbol,
        timeframe="1D",
        event_time=stamps,
        open=c,
        high=c * 1.01,
        low=c * 0.99,
        close=c,
        volume=np.full(n, 1e6),
    )


class TestAgreement:
    def test_identical_series_agree(self) -> None:
        bars = _series([100.0, 101.0, 99.0, 103.0])
        report = compare(bars, bars)
        assert report.agrees
        assert report.disagreements == []

    def test_a_constant_price_factor_is_not_a_disagreement(self) -> None:
        """The whole point. Two vendors adjusting from different bases produce
        price levels that differ by a factor and returns that do not differ at
        all."""
        closes = [100.0, 101.0, 99.0, 103.0]
        report = compare(_series(closes), _series([c / 4 for c in closes]))
        assert report.agrees

    def test_tiny_rounding_differences_are_tolerated(self) -> None:
        a = _series([100.0, 101.0, 99.0])
        b = _series([100.0, 101.0001, 99.0002])
        assert compare(a, b).agrees


class TestDisagreement:
    def test_an_unadjusted_split_in_one_source_is_located(self) -> None:
        # B fails to adjust a 2:1 split on the third bar: the same story AAPL
        # 2005-02-28 tells.
        a = _series([100.0, 101.0, 102.0, 103.0])
        b = _series([200.0, 202.0, 102.0, 103.0])
        report = compare(a, b)

        assert not report.agrees
        assert len(report.disagreements) == 1
        moment, left, right = report.disagreements[0]
        assert moment.date() == datetime(2020, 1, 3, tzinfo=UTC).date()
        assert left == pytest.approx(0.0099, abs=1e-3)
        assert right == pytest.approx(-0.4950, abs=1e-3)

    def test_the_worst_disagreement_is_reported(self) -> None:
        a = _series([100.0, 101.0, 102.0, 103.0])
        b = _series([100.0, 101.5, 102.0, 206.0])
        report = compare(a, b)
        assert report.worst > 0.9

    def test_only_overlapping_dates_are_compared(self) -> None:
        """One vendor starting later is a different complaint, already reported
        by the quality check. Comparing a date only one of them has would
        manufacture a disagreement out of a listing date."""
        a = _series([100.0, 101.0, 102.0, 103.0])
        b = _series([102.0, 103.0], start_day=2)
        report = compare(a, b)
        assert report.agrees
        assert report.overlap == 2

    def test_no_overlap_is_reported_rather_than_called_agreement(self) -> None:
        a = _series([100.0, 101.0])
        b = _series([100.0, 101.0], start_day=500)
        report = compare(a, b)
        assert report.overlap == 0
        assert not report.agrees
        assert "no overlap" in str(report).lower()


class TestReporting:
    def test_the_report_names_both_sources(self) -> None:
        a = _series([100.0, 101.0, 102.0])
        b = _series([100.0, 101.0, 51.0])
        report = compare(a, b, left_name="yahoo", right_name="ibkr")
        text = str(report)
        assert "yahoo" in text and "ibkr" in text

    def test_a_clean_comparison_says_how_many_days_matched(self) -> None:
        bars = _series([100.0, 101.0, 99.0, 103.0])
        assert "3" in str(compare(bars, bars))

    def test_it_says_which_side_looks_wrong_when_one_is_a_clean_ratio(self) -> None:
        """A disagreement where one side moved by exactly -1/2, -1/3 or -1/4 and
        the other did not is a split, and naming it saves the next person a trip
        to a corporate-actions calendar."""
        a = _series([100.0, 101.0, 102.0, 103.0])
        b = _series([200.0, 202.0, 102.0, 103.0])
        assert "split" in str(compare(a, b, left_name="yahoo", right_name="ibkr")).lower()


class TestSessionAlignment:
    """Vendors stamp the same session at different instants.

    Yahoo puts a daily bar at 00:00Z; Alpaca puts it at the session open, 05:00Z;
    IBKR hands back a bare date. Matching on the raw instant found nothing in
    common between Yahoo and Alpaca across sixteen overlapping years and reported
    it as "no overlap" for all 54 symbols -- a comparison that silently checks
    nothing is worse than no comparison, because it reads as a clean bill.
    """

    def test_the_same_session_matches_across_different_stamp_conventions(self) -> None:
        closes = [100.0, 101.0, 99.0, 103.0]
        midnight = _series(closes)
        # The same four sessions, stamped at the 05:00Z open instead.
        opening = Bars(
            symbol="TEST",
            timeframe="1D",
            event_time=midnight.event_time + 5 * 3600,
            open=midnight.open,
            high=midnight.high,
            low=midnight.low,
            close=midnight.close,
            volume=midnight.volume,
        )
        report = compare(midnight, opening)
        assert report.overlap == 4
        assert report.agrees

    def test_a_disagreement_is_still_found_across_conventions(self) -> None:
        a = _series([100.0, 101.0, 102.0, 103.0])
        shifted = _series([100.0, 101.0, 51.0, 51.5])
        b = Bars(
            symbol="TEST",
            timeframe="1D",
            event_time=shifted.event_time + 5 * 3600,
            open=shifted.open,
            high=shifted.high,
            low=shifted.low,
            close=shifted.close,
            volume=shifted.volume,
        )
        report = compare(a, b)
        assert not report.agrees
        assert report.worst > 0.4

    def test_genuinely_disjoint_windows_still_report_no_overlap(self) -> None:
        a = _series([100.0, 101.0])
        b = _series([100.0, 101.0], start_day=500)
        assert compare(a, b).overlap == 0


class TestComparable:
    def test_no_overlap_is_not_the_same_as_disagreement(self) -> None:
        """A summary that counts "nothing to compare" as "disagrees" turns a
        broken comparison into a data problem and sends the reader after the
        wrong bug."""
        a = _series([100.0, 101.0])
        b = _series([100.0, 101.0], start_day=500)
        report = compare(a, b)
        assert not report.comparable
        assert not report.disagrees

    def test_a_real_disagreement_is_both_comparable_and_disagreeing(self) -> None:
        a = _series([100.0, 101.0, 102.0])
        b = _series([100.0, 101.0, 51.0])
        report = compare(a, b)
        assert report.comparable
        assert report.disagrees

    def test_agreement_is_comparable_and_not_disagreeing(self) -> None:
        bars = _series([100.0, 101.0, 102.0])
        report = compare(bars, bars)
        assert report.comparable
        assert not report.disagrees
