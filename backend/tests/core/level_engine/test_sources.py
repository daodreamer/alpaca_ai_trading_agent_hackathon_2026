"""Where candidates come from — specs/07-levels.md MVP source list.

Each source is a pure function over bars or confirmed structure. The recurring
question in these tests is what each candidate is *dated to*: recency and
look-ahead are both measured from that date (ADR 0009 D5).
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal

import pytest

from alphagate.core.errors import InvariantViolation
from alphagate.core.level_engine import (
    moving_average_candidate,
    previous_day_candidates,
    previous_week_candidates,
    swing_candidates,
    year_extreme_candidates,
)
from alphagate.core.levels import LevelKind, LevelSource
from alphagate.core.structure import SwingKind, SwingPoint, SwingStatus
from tests.core.level_engine.synthetic import daily_bar, daily_series, start_of

# 2026-08-13 is a Thursday: ISO week 33 runs Mon 10th – Fri 14th, week 32 the 3rd – 7th.
LAST_WEEK = [
    (date(2026, 8, 3), "104", "96"),
    (date(2026, 8, 4), "108", "99"),
    (date(2026, 8, 5), "106", "94"),
]
THIS_WEEK = [
    (date(2026, 8, 10), "112", "101"),
    (date(2026, 8, 11), "115", "103"),
]


def swing(kind: SwingKind, price: str, *, index: int = 1) -> SwingPoint:
    return SwingPoint(
        kind=kind,
        status=SwingStatus.CONFIRMED,
        bar_start_utc=start_of(index),
        detected_at_utc=start_of(index + 2),
        price=price,
    )


# -- swings -----------------------------------------------------------------


def test_a_swing_high_becomes_resistance_and_a_swing_low_support() -> None:
    candidates = swing_candidates(
        [swing(SwingKind.HIGH, "120"), swing(SwingKind.LOW, "100", index=4)]
    )

    assert [(c.kind, c.source, c.price) for c in candidates] == [
        (LevelKind.RESISTANCE, LevelSource.SWING_HIGH, Decimal("120")),
        (LevelKind.SUPPORT, LevelSource.SWING_LOW, Decimal("100")),
    ]


def test_a_swing_candidate_is_dated_to_its_pivot_not_its_detection() -> None:
    (level_candidate,) = swing_candidates([swing(SwingKind.HIGH, "120", index=1)])

    assert level_candidate.observed_at_utc == start_of(1)


def test_an_unconfirmed_swing_is_not_evidence() -> None:
    forming = SwingPoint(
        kind=SwingKind.HIGH,
        status=SwingStatus.CANDIDATE,
        bar_start_utc=start_of(3),
        detected_at_utc=start_of(3),
        price="130",
    )

    assert swing_candidates([forming]) == ()


# -- session extremes -------------------------------------------------------


def test_previous_day_uses_the_most_recent_completed_session() -> None:
    high, low = previous_day_candidates(daily_series([*LAST_WEEK, *THIS_WEEK]))

    assert (high.source, high.price) == (LevelSource.PREV_DAY_HIGH, Decimal("115"))
    assert (low.source, low.price) == (LevelSource.PREV_DAY_LOW, Decimal("103"))


def test_an_unfinished_session_is_not_a_previous_day() -> None:
    bars = (
        *daily_series(THIS_WEEK),
        daily_bar(date(2026, 8, 12), high="999", low="1", is_final=False),
    )

    high, _ = previous_day_candidates(bars)

    assert high.price == Decimal("115")


def test_previous_day_needs_at_least_one_completed_session() -> None:
    assert previous_day_candidates(()) == ()


def test_previous_week_spans_the_last_completed_iso_week() -> None:
    high, low = previous_week_candidates(daily_series([*LAST_WEEK, *THIS_WEEK]))

    assert (high.source, high.price) == (LevelSource.PREV_WEEK_HIGH, Decimal("108"))
    assert (low.source, low.price) == (LevelSource.PREV_WEEK_LOW, Decimal("94"))


def test_the_week_in_progress_is_not_the_previous_week() -> None:
    assert previous_week_candidates(daily_series(THIS_WEEK)) == ()


def test_year_extremes_look_back_one_year_and_no_further() -> None:
    bars = daily_series(
        [
            (date(2025, 1, 2), "500", "480"),  # older than a year: ignored
            (date(2026, 3, 2), "150", "60"),
            *LAST_WEEK,
            *THIS_WEEK,
        ]
    )

    high, low = year_extreme_candidates(bars)

    assert (high.source, high.price) == (LevelSource.YEAR_HIGH, Decimal("150"))
    assert (low.source, low.price) == (LevelSource.YEAR_LOW, Decimal("60"))


def test_a_year_extreme_is_dated_to_the_session_that_made_it() -> None:
    bars = daily_series([(date(2026, 3, 2), "150", "60"), *THIS_WEEK])

    high, _ = year_extreme_candidates(bars)

    assert high.observed_at_utc == datetime(2026, 3, 2, 13, 30, tzinfo=UTC)


def test_year_extremes_need_bars() -> None:
    assert year_extreme_candidates(()) == ()


def test_previous_week_needs_bars() -> None:
    assert previous_week_candidates(()) == ()


def test_duplicate_daily_bars_are_refused_rather_than_silently_averaged() -> None:
    bars = (daily_bar(date(2026, 8, 10), high="1", low="1"), *daily_series(THIS_WEEK))

    with pytest.raises(InvariantViolation, match="one bar per session"):
        previous_day_candidates(bars)


def test_session_sources_do_not_care_about_input_order() -> None:
    rows = [*LAST_WEEK, *THIS_WEEK]
    forwards = previous_day_candidates(daily_series(rows))
    backwards = previous_day_candidates(tuple(reversed(daily_series(rows))))

    assert forwards == backwards


# -- moving averages --------------------------------------------------------


def test_a_moving_average_below_price_is_support() -> None:
    level_candidate = moving_average_candidate(100.0, price=105.0, observed_at_utc=start_of(3))

    assert level_candidate.kind is LevelKind.SUPPORT
    assert level_candidate.source is LevelSource.MOVING_AVERAGE
    assert level_candidate.price == Decimal("100")


def test_a_moving_average_above_price_is_resistance() -> None:
    level_candidate = moving_average_candidate(100.0, price=95.0, observed_at_utc=start_of(3))

    assert level_candidate.kind is LevelKind.RESISTANCE


def test_price_exactly_on_the_average_reads_as_resistance() -> None:
    """The STRICT reading used for structure labels, applied here too."""
    level_candidate = moving_average_candidate(100.0, price=100.0, observed_at_utc=start_of(3))

    assert level_candidate.kind is LevelKind.RESISTANCE


def test_a_moving_average_crosses_into_the_exact_domain_deliberately() -> None:
    """float -> Decimal through `from_approximate`, never `Decimal(float)`."""
    level_candidate = moving_average_candidate(100.1, price=105.0, observed_at_utc=start_of(3))

    assert str(level_candidate.price) == "100.10000000"
