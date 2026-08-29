"""Level lifecycle beyond the MVP break — specs/07-levels.md, ADR 0025 D5/D6.

Decay, expiry and polarity flip. All three are off unless configured, so the
first thing these tests establish is that a default configuration still behaves
exactly like the MVP did.
"""

from __future__ import annotations

import math

import pytest

from alphagate.core.level_engine import (
    LevelConfig,
    LifecycleConfig,
    ZoneWidth,
    advance,
    bars_since_touch,
    build_levels,
    flip,
)
from alphagate.core.levels import Level, LevelKind, LevelSource, LevelStatus
from alphagate.core.time_model import Timeframe
from tests.core.level_engine.synthetic import AAPL, candidate, minute_bar, start_of

AS_OF = start_of(0)
"""When the levels under test were built.

The session's own clock, not an arbitrary later instant: `advance` only looks at
bars that closed *after* a level was built, so a level dated to the end of the
day would ignore a window from the morning of it."""

ABSOLUTE = LevelConfig(
    zone_width=ZoneWidth.absolute("0.5"),
    merge_distance=ZoneWidth.absolute("1"),
    max_cluster_width=ZoneWidth.absolute("3"),
)

# Resistance at 100, zone [99.5, 100.5].
BELOW = [minute_bar(index, high="98", low="97") for index in range(6)]
"""Bars that neither touch the zone nor close beyond it."""


def resistance(**kwargs: object) -> Level:
    (level,) = build_levels(
        [candidate("100")],
        symbol=AAPL,
        timeframe=Timeframe.M1,
        as_of=AS_OF,
        config=ABSOLUTE,
        **kwargs,  # type: ignore[arg-type]
    )
    return level


# -- counting the silence ---------------------------------------------------


def test_a_level_price_is_sitting_in_was_touched_zero_bars_ago() -> None:
    level = resistance()
    bars = [*BELOW, minute_bar(6, high="100", low="99.6")]

    assert bars_since_touch(level, bars) == 0


def test_bars_since_a_touch_counts_back_to_the_last_one() -> None:
    level = resistance()
    bars = [minute_bar(0, high="100", low="99.6"), *BELOW[:3]]

    assert bars_since_touch(level, bars) == 3


def test_a_level_nothing_in_the_window_ever_touched_is_as_old_as_the_window() -> None:
    assert bars_since_touch(resistance(), BELOW) == len(BELOW)


# -- the default is still the MVP -------------------------------------------


def test_by_default_an_untested_level_just_stands_there() -> None:
    """No decay, no expiry, no flip unless asked for (ADR 0025 D6)."""
    level = resistance(bars=BELOW)

    (carried,) = advance(level, bars=BELOW)

    assert carried == level


def test_a_close_beyond_the_zone_still_invalidates_whatever_else_is_configured() -> None:
    level = resistance()
    broke = [*BELOW, minute_bar(6, high="102", low="99", close="101")]

    (carried,) = advance(level, bars=broke)

    assert carried.status is LevelStatus.INVALIDATED


def test_an_already_invalidated_level_is_left_alone() -> None:
    level = resistance()
    dead = advance(level, bars=[minute_bar(6, high="102", low="99", close="101")])[0]

    assert advance(dead, bars=BELOW, config=LifecycleConfig(untouched_decay=0.5)) == (dead,)


def test_a_close_from_before_the_level_existed_cannot_have_broken_it() -> None:
    """Otherwise a level built today dies instantly on last week's rally."""
    late = build_levels(
        [candidate("100")],
        symbol=AAPL,
        timeframe=Timeframe.M1,
        as_of=start_of(9),
        config=ABSOLUTE,
    )[0]
    earlier_break = [minute_bar(2, high="102", low="99", close="101")]

    (carried,) = advance(late, bars=earlier_break)

    assert carried.status is LevelStatus.ACTIVE


def test_advancing_is_a_projection_of_the_window_not_an_accumulation() -> None:
    """Re-deriving from the level as built must give the same answer, however
    many times it is done — the property `build_levels` has (ADR 0025 D6)."""
    level = resistance(bars=BELOW)
    config = LifecycleConfig(untouched_decay=0.9)

    once = advance(level, bars=BELOW, config=config)[0]
    again = advance(level, bars=BELOW, config=config)[0]

    assert once == again


def test_a_longer_silence_decays_further_than_a_shorter_one() -> None:
    level = resistance(bars=BELOW)
    config = LifecycleConfig(untouched_decay=0.9)
    longer = [*BELOW, *[minute_bar(index, high="98", low="97") for index in range(6, 12)]]

    brief = advance(level, bars=BELOW, config=config)[0]
    extended = advance(level, bars=longer, config=config)[0]

    assert extended.strength < brief.strength


def test_a_window_with_no_closed_bar_changes_nothing() -> None:
    level = resistance()
    forming = [minute_bar(6, high="102", low="99", close="101", is_final=False)]

    assert advance(level, bars=forming) == (level,)


# -- decay ------------------------------------------------------------------


def test_an_untested_level_decays_by_the_bars_it_went_untested() -> None:
    level = resistance(bars=BELOW)
    config = LifecycleConfig(untouched_decay=0.9)

    (decayed,) = advance(level, bars=BELOW, config=config)

    assert decayed.strength == pytest.approx(level.strength * 0.9 ** len(BELOW))


def test_a_level_price_is_still_testing_does_not_decay() -> None:
    bars = [*BELOW, minute_bar(6, high="100", low="99.6")]
    level = resistance(bars=bars)

    (carried,) = advance(level, bars=bars, config=LifecycleConfig(untouched_decay=0.5))

    assert carried.strength == level.strength


def test_a_decayed_breakdown_still_sums_to_the_strength_it_explains() -> None:
    level = resistance(bars=BELOW)

    (decayed,) = advance(level, bars=BELOW, config=LifecycleConfig(untouched_decay=0.8))

    assert math.fsum(value for _, value in decayed.strength_breakdown) == pytest.approx(
        decayed.strength
    )
    assert decayed.strength_breakdown != level.strength_breakdown


def test_decay_moves_the_update_stamp_and_leaves_the_creation_one() -> None:
    level = resistance(bars=BELOW)

    (decayed,) = advance(level, bars=BELOW, config=LifecycleConfig(untouched_decay=0.9))

    assert decayed.created_at == level.created_at
    assert decayed.updated_at >= level.updated_at


# -- expiry -----------------------------------------------------------------


def test_a_level_nothing_has_tested_for_long_enough_is_invalidated() -> None:
    level = resistance(bars=BELOW)
    config = LifecycleConfig(expire_after_untouched_bars=len(BELOW))

    (expired,) = advance(level, bars=BELOW, config=config)

    assert expired.status is LevelStatus.INVALIDATED


def test_a_level_tested_recently_enough_survives() -> None:
    level = resistance(bars=BELOW)
    config = LifecycleConfig(expire_after_untouched_bars=len(BELOW) + 1)

    (survived,) = advance(level, bars=BELOW, config=config)

    assert survived.status is LevelStatus.ACTIVE


def test_expiry_never_produces_a_flipped_counterpart() -> None:
    """Nothing traded through this level; flipping it would claim a rejection
    that never happened (ADR 0025 D6)."""
    level = resistance(bars=BELOW)
    config = LifecycleConfig(expire_after_untouched_bars=1, polarity_flip=True)

    carried = advance(level, bars=BELOW, config=config)

    assert len(carried) == 1


# -- polarity flip ----------------------------------------------------------


def test_broken_resistance_comes_back_as_support() -> None:
    level = resistance()
    broke = [minute_bar(6, high="102", low="99", close="101")]

    dead, flipped = advance(level, bars=broke, config=LifecycleConfig(polarity_flip=True))

    assert dead.status is LevelStatus.INVALIDATED
    assert flipped.kind is LevelKind.SUPPORT
    assert flipped.status is LevelStatus.ACTIVE
    assert flipped.zone == level.zone


def test_a_flipped_level_says_which_level_it_is_the_other_side_of() -> None:
    """Kind is part of the derived identity, so continuity has to be explicit
    (ADR 0025 D5)."""
    level = resistance()
    broke = [minute_bar(6, high="102", low="99", close="101")]

    _, flipped = advance(level, bars=broke, config=LifecycleConfig(polarity_flip=True))

    assert flipped.flipped_from == level.id
    assert flipped.id != level.id


def test_a_flip_carries_the_evidence_unchanged() -> None:
    level = resistance(bars=BELOW)
    broke = [*BELOW, minute_bar(6, high="102", low="99", close="101")]

    _, flipped = advance(level, bars=broke, config=LifecycleConfig(polarity_flip=True))

    assert flipped.sources == level.sources
    assert flipped.strength == level.strength
    assert flipped.strength_breakdown == level.strength_breakdown


def test_broken_support_comes_back_as_resistance() -> None:
    (level,) = build_levels(
        [candidate("100", kind=LevelKind.SUPPORT, source=LevelSource.SWING_LOW)],
        symbol=AAPL,
        timeframe=Timeframe.M1,
        as_of=AS_OF,
        config=ABSOLUTE,
    )
    broke = [minute_bar(6, high="100", low="98", close="98.5")]

    _, flipped = advance(level, bars=broke, config=LifecycleConfig(polarity_flip=True))

    assert flipped.kind is LevelKind.RESISTANCE


def test_a_flip_is_dated_to_the_break_that_caused_it() -> None:
    level = resistance()
    breaking = minute_bar(6, high="102", low="99", close="101")

    _, flipped = advance(level, bars=[breaking], config=LifecycleConfig(polarity_flip=True))

    assert flipped.created_at == breaking.end_time_utc
    assert flipped.updated_at == breaking.end_time_utc


def test_flipping_twice_is_flipping_back() -> None:
    level = resistance()

    once = flip(level, at=AS_OF)
    twice = flip(once, at=AS_OF)

    assert twice.kind is level.kind
    assert twice.id == level.id
    assert twice.flipped_from == once.id


def test_nothing_flips_unless_it_is_asked_for() -> None:
    level = resistance()
    broke = [minute_bar(6, high="102", low="99", close="101")]

    assert len(advance(level, bars=broke)) == 1


# -- configuration ----------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("untouched_decay", 0.0),
        ("untouched_decay", 1.5),
        ("expire_after_untouched_bars", 0),
    ],
)
def test_an_impossible_lifecycle_configuration_is_refused(field: str, value: float) -> None:
    from alphagate.core.errors import InvariantViolation

    with pytest.raises(InvariantViolation, match=field):
        LifecycleConfig(**{field: value})  # type: ignore[arg-type]


def test_the_lifecycle_is_the_same_however_often_it_is_run() -> None:
    level = resistance(bars=BELOW)
    config = LifecycleConfig(untouched_decay=0.9, polarity_flip=True)

    assert advance(level, bars=BELOW, config=config) == advance(level, bars=BELOW, config=config)


def test_a_decayed_level_keeps_its_identity() -> None:
    """Decay changes the level's standing, not its evidence or its zone — so the
    derived id must not move (ADR 0009 D2)."""
    level = resistance(bars=BELOW)

    (decayed,) = advance(level, bars=BELOW, config=LifecycleConfig(untouched_decay=0.5))

    assert decayed.id == level.id
    assert decayed.price == level.price
    assert decayed.zone == level.zone


def test_a_zone_a_price_never_reached_decays_toward_zero_but_not_below() -> None:
    level = resistance(bars=BELOW)
    long_silence = [minute_bar(index, high="98", low="97") for index in range(200)]

    (decayed,) = advance(level, bars=long_silence, config=LifecycleConfig(untouched_decay=0.5))

    assert decayed.strength >= 0.0
    assert decayed.strength < 1.0
