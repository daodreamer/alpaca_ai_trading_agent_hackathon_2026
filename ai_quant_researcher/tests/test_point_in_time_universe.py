"""Membership as a function of time.

`NASDAQ_50` is a tuple of tickers and an `as_of` date, and its own docstring says
what is wrong with that: these are the names that survived to be listed today, so
backtesting them over 2016-2024 measures, in part, the strategy's ability to
profit from already knowing who won. The docstring has been there since the
universe was written. Nothing enforced it, because `resolve()` has no date
parameter -- membership is a constant.

That is survivable for a rule that looks at one symbol at a time and fatal for one
that ranks. A cross-sectional score asks "how does this name compare to its
peers", and if the peer set is the set of eventual winners then the comparison is
contaminated at its root: `rs_rank(60)` computed against survivors is not the
number a researcher in 2016 could have computed. The rule discovers that the
companies which were not acquired went up.

So a universe becomes an interval table: each ticker with the spans during which
it was actually a member. Three things the type has to get right, all of which
are ways to reintroduce the bias by accident:

* **A name contributes only inside its intervals.** Not from its first bar --
  bars exist before a company joins an index and after it leaves.
* **Re-entry is two intervals, not one.** Merging them would make a company a
  member during a period when it was not, which is the original error in
  miniature.
* **Boundaries are half-open.** The removal date is the first day the name is
  *not* a member, because that is how the source records it, and a convention
  chosen twice is a convention chosen wrong once.
"""

from __future__ import annotations

from datetime import date

import pytest

from aqr.data.universes import Interval, Membership, PointInTimeUniverse


def _universe() -> PointInTimeUniverse:
    return PointInTimeUniverse(
        name="probe",
        source="test",
        window=(date(2016, 1, 1), date(2024, 8, 31)),
        members=(
            Membership("AAPL", "Apple", (Interval(date(2016, 1, 1), None),)),
            Membership("ATVI", "Activision", (Interval(date(2016, 1, 1), date(2023, 10, 13)),)),
            Membership("TSLA", "Tesla", (Interval(date(2020, 12, 21), None),)),
            Membership(
                "NWL",
                "Newell",
                (
                    Interval(date(2016, 1, 1), date(2019, 6, 3)),
                    Interval(date(2021, 4, 5), None),
                ),
            ),
        ),
    )


# --------------------------------------------------------------------------
# Membership


def test_a_member_is_in_on_the_day_it_joins() -> None:
    m = Membership("TSLA", "Tesla", (Interval(date(2020, 12, 21), None),))
    assert m.member_on(date(2020, 12, 21)) is True
    assert m.member_on(date(2020, 12, 18)) is False


def test_the_removal_date_is_the_first_day_out() -> None:
    """Half-open. The source records the effective date of the removal, so on
    that date the name is already gone."""
    m = Membership("ATVI", "Activision", (Interval(date(2016, 1, 1), date(2023, 10, 13)),))
    assert m.member_on(date(2023, 10, 12)) is True
    assert m.member_on(date(2023, 10, 13)) is False


def test_re_entry_leaves_a_hole() -> None:
    """Merging two intervals into one would make the company a member during a
    period when it was not -- the original bias, in miniature."""
    m = _universe().by_ticker("NWL")
    assert m.member_on(date(2019, 1, 1)) is True
    assert m.member_on(date(2020, 1, 1)) is False
    assert m.member_on(date(2022, 1, 1)) is True


def test_intervals_must_not_overlap() -> None:
    with pytest.raises(ValueError, match="overlap"):
        Membership(
            "X",
            "X",
            (Interval(date(2016, 1, 1), date(2020, 1, 1)), Interval(date(2019, 1, 1), None)),
        )


def test_an_interval_cannot_end_before_it_starts() -> None:
    with pytest.raises(ValueError, match="before"):
        Interval(date(2020, 1, 1), date(2019, 1, 1))


def test_only_one_interval_may_be_open() -> None:
    """Two open-ended intervals means the name is a member of two overlapping
    infinite spans, which is not a state the source can express."""
    with pytest.raises(ValueError):
        Membership(
            "X", "X", (Interval(date(2016, 1, 1), None), Interval(date(2020, 1, 1), None))
        )


# --------------------------------------------------------------------------
# The universe


def test_symbols_on_a_date_are_the_members_that_day() -> None:
    u = _universe()
    assert set(u.symbols_on(date(2018, 6, 1))) == {"AAPL", "ATVI", "NWL"}
    assert set(u.symbols_on(date(2020, 6, 1))) == {"AAPL", "ATVI"}
    assert set(u.symbols_on(date(2024, 1, 1))) == {"AAPL", "TSLA", "NWL"}


def test_every_symbol_ever_a_member_is_what_gets_pulled() -> None:
    """The download list is the union across time, not the list as of today --
    otherwise the delisted names are missing and the whole exercise is pointless."""
    assert set(_universe().all_symbols()) == {"AAPL", "ATVI", "TSLA", "NWL"}


def test_the_symbol_order_is_stable() -> None:
    """A universe whose order depends on dict iteration makes a backtest
    irreproducible through the tie-break in the ranking."""
    a = _universe().all_symbols()
    b = _universe().all_symbols()
    assert a == b == tuple(sorted(a))


def test_a_universe_reports_its_own_survivorship() -> None:
    """The bias as a number rather than a paragraph.

    Four names ever, three still in at the end, one left: 25% of the universe
    is invisible to a today's-list version of it.
    """
    stats = _universe().survivorship()
    assert stats["ever"] == 4
    assert stats["still_in"] == 3
    assert stats["left"] == 1
    assert stats["left_fraction"] == pytest.approx(0.25)


def test_a_today_s_list_universe_reports_zero_survivorship_and_says_so() -> None:
    """`NASDAQ_50` converted to this type must not look unbiased just because it
    has no removals -- it has none *because* it was chosen with hindsight."""
    u = PointInTimeUniverse(
        name="today",
        source="current constituents only",
        window=(date(2016, 1, 1), date(2024, 8, 31)),
        members=(Membership("AAPL", "Apple", (Interval(date(2016, 1, 1), None),)),),
    )
    stats = u.survivorship()
    assert stats["left"] == 0
    assert stats["reconstructed"] is False


def test_a_reconstructed_universe_says_so() -> None:
    assert _universe().survivorship()["reconstructed"] is True


# --------------------------------------------------------------------------
# The old universe still works


def test_the_legacy_universe_is_untouched() -> None:
    """Every result in the registry was measured on it. It stays resolvable."""
    from aqr.data.universes import resolve

    symbols = resolve("nasdaq50")
    assert len(symbols) == 50
    assert "AAPL" in symbols
