"""Features that compare a symbol to the rest of the universe.

The largest gap in the vocabulary. Every rule so far sees one instrument at a
time, so "buy the strongest names in a strong market" -- the oldest documented
equity anomaly there is -- was not expressible at all. Neither was any form of
"this stock is moving and its sector is not".

Three properties carry the whole thing, and each has a way of failing silently.

**Point in time.** The cross-section at session t contains exactly the symbols
that had a bar at t. Including a symbol that listed in 2020 in the 2010
cross-section is look-ahead of the worst kind: the ranks would be computed
against a universe assembled with hindsight, and the resulting anomaly would be
"the stocks that were later added to the index outperformed".

**Causality.** A rank at t may use returns ending at t and nothing after. The
prefix-stability test that every other feature passes applies here too.

**Alignment.** Symbols do not share a bar count or a calendar. The output for a
symbol must line up with *its own* bars, whatever its peers were doing.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from aqr.data.bars import Bars
from aqr.features.cross_section import CrossSection


def _bars(closes: list[float], *, symbol: str, start_day: int = 0) -> Bars:
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


def _universe(**named: list[float]) -> dict[str, Bars]:
    return {name: _bars(closes, symbol=name) for name, closes in named.items()}


class TestRank:
    def test_the_strongest_name_ranks_highest(self) -> None:
        universe = _universe(
            WINNER=[100.0, 100.0, 130.0],
            MIDDLE=[100.0, 100.0, 110.0],
            LOSER=[100.0, 100.0, 90.0],
        )
        cs = CrossSection(universe)
        assert cs.rank("WINNER", 1)[2] == pytest.approx(1.0)
        assert cs.rank("LOSER", 1)[2] == pytest.approx(0.0)
        assert 0.0 < cs.rank("MIDDLE", 1)[2] < 1.0

    def test_the_rank_is_aligned_to_the_symbols_own_bars(self) -> None:
        universe = _universe(A=[100.0, 101.0, 102.0], B=[100.0, 99.0, 98.0])
        assert len(CrossSection(universe).rank("A", 1)) == 3

    def test_bars_without_a_lookback_return_are_not_ranked(self) -> None:
        universe = _universe(A=[100.0, 101.0, 102.0], B=[100.0, 99.0, 98.0])
        rank = CrossSection(universe).rank("A", 2)
        assert np.isnan(rank[0])
        assert np.isnan(rank[1])
        assert np.isfinite(rank[2])

    def test_a_universe_of_one_has_no_cross_section(self) -> None:
        """Ranking a symbol against itself always returns the same number, which
        would read as a signal and is not one."""
        rank = CrossSection(_universe(ONLY=[100.0, 101.0, 102.0])).rank("ONLY", 1)
        assert np.all(np.isnan(rank))


class TestPointInTime:
    def test_a_symbol_that_has_not_listed_yet_is_absent_from_the_cross_section(
        self,
    ) -> None:
        """The failure this guards against produces a beautiful anomaly: rank
        against a universe assembled with hindsight and you have discovered that
        the stocks later added to the index outperformed."""
        early = _bars([100.0, 100.0, 130.0], symbol="EARLY")
        # Lists on the third session, then rockets.
        late = _bars([100.0, 500.0], symbol="LATE", start_day=2)
        cs = CrossSection({"EARLY": early, "LATE": late})

        # On session 2 (index 1) only EARLY exists, so there is nothing to rank
        # against and the rank is undefined rather than 1.0.
        assert np.isnan(cs.rank("EARLY", 1)[1])

    def test_a_symbol_that_delists_stops_contributing(self) -> None:
        long_lived = _bars([100.0] * 6, symbol="LONG")
        short = _bars([100.0, 130.0], symbol="SHORT")
        cs = CrossSection({"LONG": long_lived, "SHORT": short})
        rank = cs.rank("LONG", 1)
        assert np.isfinite(rank[1])  # SHORT still around
        assert np.isnan(rank[4])  # nobody left to compare with


class TestCausality:
    def test_ranks_are_prefix_stable(self) -> None:
        rng = np.random.default_rng(3)
        universe = {
            name: _bars(list(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 200)))), symbol=name)
            for name in ("A", "B", "C", "D")
        }
        full = CrossSection(universe).rank("A", 20)

        truncated_universe = {k: v.slice(0, 120) for k, v in universe.items()}
        truncated = CrossSection(truncated_universe).rank("A", 20)

        np.testing.assert_allclose(truncated, full[:120], equal_nan=True)

    def test_a_later_peer_move_cannot_change_an_earlier_rank(self) -> None:
        base = _universe(A=[100.0, 110.0, 100.0], B=[100.0, 100.0, 100.0])
        before = CrossSection(base).rank("A", 1)[1]

        # B does something dramatic, but only on the last bar.
        moved = _universe(A=[100.0, 110.0, 100.0], B=[100.0, 100.0, 500.0])
        after = CrossSection(moved).rank("A", 1)[1]
        assert before == pytest.approx(after)


class TestRelativeReturn:
    def test_it_is_the_symbols_return_less_the_universe_median(self) -> None:
        universe = _universe(
            A=[100.0, 120.0],  # +20%
            B=[100.0, 110.0],  # +10%
            C=[100.0, 100.0],  # 0%
        )
        cs = CrossSection(universe)
        assert cs.relative_return("A", 1)[1] == pytest.approx(0.10)
        assert cs.relative_return("C", 1)[1] == pytest.approx(-0.10)

    def test_a_symbol_alone_on_a_session_has_no_relative_return(self) -> None:
        """Arithmetically the median of one number is that number, so the answer
        would be exactly 0.0 -- "precisely average" on a day when there is no
        average. Worse, 0.0 is a value that `relative_return(20) > 0` evaluates
        happily, while NaN correctly takes the bar out of the comparison."""
        cs = CrossSection(
            {
                "A": _bars([100.0, 120.0], symbol="A"),
                "LATE": _bars([100.0, 500.0], symbol="LATE", start_day=5),
            }
        )
        assert np.isnan(cs.relative_return("A", 1)[1])

    def test_the_median_is_taken_over_present_symbols_only(self) -> None:
        universe = _universe(A=[100.0, 120.0], B=[100.0, 110.0], C=[100.0, 100.0])
        late = _bars([100.0, 500.0], symbol="LATE", start_day=5)
        cs = CrossSection({**universe, "LATE": late})
        # LATE has not listed, so the median is B's +10% and A sits 10% above it.
        assert cs.relative_return("A", 1)[1] == pytest.approx(0.10)


class TestBreadth:
    def test_it_is_the_fraction_of_the_universe_that_rose(self) -> None:
        universe = _universe(
            A=[100.0, 110.0],
            B=[100.0, 105.0],
            C=[100.0, 95.0],
            D=[100.0, 90.0],
        )
        assert CrossSection(universe).breadth("A", 1)[1] == pytest.approx(0.5)

    def test_breadth_is_the_same_number_for_every_symbol(self) -> None:
        universe = _universe(A=[100.0, 110.0], B=[100.0, 105.0], C=[100.0, 95.0])
        cs = CrossSection(universe)
        assert cs.breadth("A", 1)[1] == pytest.approx(cs.breadth("C", 1)[1])


class TestUnknownSymbol:
    def test_asking_about_a_symbol_outside_the_universe_raises(self) -> None:
        with pytest.raises(KeyError, match="NOPE"):
            CrossSection(_universe(A=[100.0, 101.0])).rank("NOPE", 1)


class TestCaching:
    def test_the_same_request_is_computed_once(self) -> None:
        universe = _universe(A=[100.0, 101.0, 102.0], B=[100.0, 99.0, 98.0])
        cs = CrossSection(universe)
        first = cs.rank("A", 1)
        assert cs.rank("A", 1) is first

    def test_the_ranking_is_shared_by_every_symbol_not_redone_per_symbol(self) -> None:
        """The cost that made a 661-name universe unusable.

        Everything under the symbol argument is universe-wide. Asking a second
        symbol for the same period must read the grid the first one built, not
        rank the universe again -- otherwise the work is quadratic in the
        universe size and a full-index sweep does not finish.
        """
        universe = _universe(
            A=[100.0, 101.0, 102.0],
            B=[100.0, 99.0, 98.0],
            C=[100.0, 103.0, 97.0],
        )
        cs = CrossSection(universe)
        cs.rank("A", 1)
        built = cs._rank_grid(1)
        cs.rank("B", 1)
        cs.rank("C", 1)
        assert cs._rank_grid(1) is built

    def test_a_different_period_gets_its_own_grid(self) -> None:
        universe = _universe(
            A=[100.0, 101.0, 102.0, 103.0],
            B=[100.0, 99.0, 98.0, 97.0],
            C=[100.0, 103.0, 97.0, 105.0],
        )
        cs = CrossSection(universe)
        assert cs._rank_grid(1) is not cs._rank_grid(2)


class TestRankDefinition:
    def test_it_matches_counting_every_pair(self) -> None:
        """The sorted-and-searched rank is the pairwise rank, exactly.

        The fast form replaced an ``n**2`` comparison matrix. It is worth
        pinning that it replaced it and did not merely approximate it --
        including the tie, where both names take the lower value.
        """
        rng = np.random.default_rng(7)
        grid = rng.normal(size=(40, 9))
        grid[3, 2] = grid[3, 5]  # a tie
        grid[7, :] = np.nan  # a session nobody traded
        grid[8, 1:] = np.nan  # a session with one lone name
        grid[9, 4] = np.nan  # one absentee among peers

        present = np.isfinite(grid)
        expected = np.full(grid.shape, np.nan)
        for i in range(grid.shape[0]):
            finite = np.flatnonzero(present[i])
            if finite.size < 2:
                continue
            values = grid[i, finite]
            pairwise = (values[:, None] > values[None, :]).sum(axis=1)
            expected[i, finite] = pairwise / (values.size - 1)

        np.testing.assert_array_equal(CrossSection._rank_rows(grid), expected)


def _intraday_bars(closes: list[float], *, symbol: str) -> Bars:
    """Hourly bars inside a single session (and the next), all same day."""
    base = datetime(2020, 1, 1, tzinfo=UTC)
    n = len(closes)
    stamps = np.array(
        [int((base + timedelta(hours=i)).timestamp()) for i in range(n)],
        dtype=np.int64,
    )
    c = np.array(closes, dtype=np.float64)
    return Bars(
        symbol=symbol,
        timeframe="1h",
        event_time=stamps,
        open=c,
        high=c * 1.01,
        low=c * 0.99,
        close=c,
        volume=np.full(n, 1e6),
    )


class TestIntraday:
    """A session grid is a look-ahead grid on intraday bars.

    Four hourly bars share one session. Averaging them into one row hands the
    10:00 bar the cross-section of the 13:00 close -- a rank built from returns
    that will not exist for another three hours. The grid is keyed on exact bar
    timestamps so each bar gets the cross-section of its own close; on daily
    data every symbol's bar closes at the same timestamp, so nothing there
    changes.
    """

    def test_bars_within_a_session_get_their_own_cross_section(self) -> None:
        universe = {
            "A": _intraday_bars([100.0, 110.0, 100.0, 100.0], symbol="A"),
            "B": _intraday_bars([100.0, 100.0, 110.0, 100.0], symbol="B"),
        }
        rank = CrossSection(universe).rank("A", 1)
        # Bar 1: A rose 10% against a flat B -- the strongest name.
        assert rank[1] == pytest.approx(1.0)
        # Bar 2: A fell back while B rose 10% -- now the weakest. A session
        # grid would repeat bar 1's (or the day's last) rank here.
        assert rank[2] == pytest.approx(0.0)

    def test_a_later_bar_in_the_same_session_cannot_change_an_earlier_rank(
        self,
    ) -> None:
        base = {
            "A": _intraday_bars([100.0, 101.0, 102.0, 103.0], symbol="A"),
            "B": _intraday_bars([100.0, 99.0, 98.0, 97.0], symbol="B"),
        }
        before = CrossSection(base).rank("A", 1)[1]

        # B does something dramatic, but only on the last bar of the session.
        moved = {
            "A": _intraday_bars([100.0, 101.0, 102.0, 103.0], symbol="A"),
            "B": _intraday_bars([100.0, 99.0, 98.0, 500.0], symbol="B"),
        }
        after = CrossSection(moved).rank("A", 1)[1]
        assert before == pytest.approx(after)

    def test_relative_return_uses_only_returns_known_at_that_bar(self) -> None:
        universe = {
            "A": _intraday_bars([100.0, 110.0, 100.0, 100.0], symbol="A"),
            "B": _intraday_bars([100.0, 100.0, 95.0, 100.0], symbol="B"),
        }
        cs = CrossSection(universe)
        # Bar 1: A +10%, B flat -- the median of two is their midpoint, so A
        # sits 5 points above it.
        assert cs.relative_return("A", 1)[1] == pytest.approx(0.05)
        # Bar 2: A -1/11, B -5%; a session grid would still show bar 1's gap.
        assert cs.relative_return("A", 1)[2] == pytest.approx(
            ((-1.0 / 11.0) - (-0.05)) / 2.0
        )
        # Breadth moves with the bars instead of freezing at the day's last
        # reading: one gainer at bar 1, none at bar 2.
        assert cs.breadth("A", 1)[1] == pytest.approx(0.5)
        assert cs.breadth("A", 1)[2] == pytest.approx(0.0)
