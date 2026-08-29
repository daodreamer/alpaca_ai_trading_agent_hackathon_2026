"""Level assembly, scoring and invalidation — ADR 0009 D2/D6/D7/D8.

The phase's exit criterion is "deterministic levels", which here means something
stronger than "stable output": two rebuilds of the same evidence produce levels
that compare *equal*, identity included.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from alphagate.core.errors import InvariantViolation
from alphagate.core.level_engine import (
    LevelConfig,
    ProfileConfig,
    StrengthWeights,
    ZoneWidth,
    build_levels,
    build_volume_profile,
    count_touch_episodes,
    invalidate,
    level_id,
)
from alphagate.core.levels import LevelKind, LevelSource, LevelStatus, Zone
from alphagate.core.time_model import Timeframe
from tests.core.level_engine.synthetic import AAPL, candidate, minute_bar, start_of

AS_OF = datetime(2026, 8, 13, 20, 0, tzinfo=UTC)

ABSOLUTE = LevelConfig(
    zone_width=ZoneWidth.absolute("0.5"),
    merge_distance=ZoneWidth.absolute("1"),
    max_cluster_width=ZoneWidth.absolute("3"),
)


def build(
    *candidates: object, config: LevelConfig = ABSOLUTE, **kwargs: object
) -> tuple[object, ...]:
    return build_levels(
        candidates,  # type: ignore[arg-type]
        symbol=AAPL,
        timeframe=Timeframe.M1,
        as_of=AS_OF,
        config=config,
        **kwargs,  # type: ignore[arg-type]
    )


# -- assembly ---------------------------------------------------------------


def test_a_cluster_becomes_one_level_with_a_zone_around_its_centre() -> None:
    (level,) = build(candidate("100"), candidate("101", index=1))

    assert level.symbol == AAPL
    assert level.kind is LevelKind.RESISTANCE
    assert level.price == Decimal("100.5")  # equal quality and no decay: the plain mean
    assert level.zone == Zone(low=Decimal("100"), high=Decimal("101"))  # centre ± 0.5
    assert level.status is LevelStatus.ACTIVE


def test_a_level_records_every_source_that_supported_it() -> None:
    (level,) = build(
        candidate("100", source=LevelSource.SWING_HIGH),
        candidate("100.2", source=LevelSource.PREV_DAY_HIGH, index=1),
    )

    assert level.sources == frozenset({LevelSource.SWING_HIGH, LevelSource.PREV_DAY_HIGH})


def test_the_centre_is_weighted_by_source_quality() -> None:
    """A 52-week high pulls the centre harder than a moving average does."""
    (level,) = build(
        candidate("100", source=LevelSource.YEAR_HIGH),
        candidate("101", source=LevelSource.MOVING_AVERAGE, index=1),
    )

    expected = (100 * 0.9 + 101 * 0.5) / (0.9 + 0.5)
    assert float(level.price) == pytest.approx(expected, abs=1e-8)


def test_a_level_has_no_zone_when_the_width_cannot_be_known() -> None:
    """ATR-based width before ATR is warm: an exact level, not a fabricated band."""
    (level,) = build(candidate("100"), config=LevelConfig())

    assert level.zone is None
    assert level.effective_zone == Zone(low=Decimal("100"), high=Decimal("100"))


def test_a_zone_never_extends_to_zero() -> None:
    config = LevelConfig(
        zone_width=ZoneWidth.absolute("5"),
        merge_distance=ZoneWidth.absolute("1"),
        max_cluster_width=ZoneWidth.absolute("3"),
    )

    (level,) = build(candidate("1"), config=config)

    assert level.zone is not None
    assert level.zone.low > 0


def test_levels_are_returned_in_price_order() -> None:
    levels = build(candidate("105"), candidate("100", index=1), candidate("110", index=2))

    assert [level.price for level in levels] == [Decimal("100"), Decimal("105"), Decimal("110")]


# -- identity ---------------------------------------------------------------


def test_identity_is_derived_from_the_zone_so_rebuilds_compare_equal() -> None:
    first = build(candidate("100"), candidate("101", index=1))
    second = build(candidate("100"), candidate("101", index=1))

    assert first == second
    assert first[0].id == second[0].id


def test_the_id_names_the_symbol_timeframe_kind_and_zone() -> None:
    identity = level_id(
        symbol=AAPL,
        timeframe=Timeframe.H1,
        kind=LevelKind.SUPPORT,
        zone=Zone(low=Decimal("99.5"), high=Decimal("100.5")),
    )

    assert identity == "AAPL:1h:SUPPORT:99.50000000:100.50000000"


def test_support_and_resistance_at_one_price_are_different_levels() -> None:
    levels = build(
        candidate("100", kind=LevelKind.RESISTANCE, source=LevelSource.SWING_HIGH),
        candidate("100", kind=LevelKind.SUPPORT, source=LevelSource.SWING_LOW, index=1),
    )

    assert len({level.id for level in levels}) == 2


# -- touches ----------------------------------------------------------------

ZONE = Zone(low=Decimal("99"), high=Decimal("101"))


def test_a_touch_is_an_episode_not_a_bar_count() -> None:
    """Ten bars resting in a zone is one test of it, not ten (ADR 0009 D6)."""
    bars = [
        minute_bar(0, high="105", low="103"),
        minute_bar(1, high="101", low="99"),
        minute_bar(2, high="100.5", low="99.5"),
        minute_bar(3, high="100.2", low="99.8"),
        minute_bar(4, high="105", low="103"),
        minute_bar(5, high="100", low="98"),
    ]

    assert count_touch_episodes(ZONE, bars) == 2


def test_a_bar_that_never_reaches_the_zone_is_not_a_touch() -> None:
    assert count_touch_episodes(ZONE, [minute_bar(0, high="98.9", low="98")]) == 0


def test_touching_the_exact_zone_edge_counts() -> None:
    """The zone is closed, so an edge touch is a touch (levels.Zone)."""
    assert count_touch_episodes(ZONE, [minute_bar(0, high="99", low="97")]) == 1


def test_no_bars_means_no_episodes() -> None:
    assert count_touch_episodes(ZONE, []) == 0


# -- strength and confidence ------------------------------------------------


def test_the_breakdown_sums_to_the_strength_it_explains() -> None:
    bars = [minute_bar(index, high="101", low="99") for index in range(3)]

    (level,) = build(
        candidate("100", source=LevelSource.SWING_HIGH),
        candidate("100.2", source=LevelSource.PREV_DAY_HIGH, index=1),
        bars=bars,
    )

    assert math.fsum(value for _, value in level.strength_breakdown) == pytest.approx(
        level.strength
    )
    assert dict(level.strength_breakdown).keys() == {
        "source_quality",
        "touches",
        "recency",
        "confluence",
    }


def test_more_independent_sources_score_higher() -> None:
    bars = [minute_bar(0, high="101", low="99")]
    lonely = build(candidate("100", source=LevelSource.SWING_HIGH), bars=bars)[0]
    crowded = build(
        candidate("100", source=LevelSource.SWING_HIGH),
        candidate("100.1", source=LevelSource.PREV_DAY_HIGH, index=1),
        candidate("100.2", source=LevelSource.PREV_WEEK_HIGH, index=2),
        bars=bars,
    )[0]

    assert crowded.strength > lonely.strength


def test_more_touches_score_higher() -> None:
    once = [minute_bar(0, high="101", low="99")]
    thrice = [
        minute_bar(0, high="101", low="99"),
        minute_bar(1, high="120", low="110"),
        minute_bar(2, high="101", low="99"),
        minute_bar(3, high="120", low="110"),
        minute_bar(4, high="101", low="99"),
    ]

    tested_once = build(candidate("100"), bars=once)[0]
    tested_thrice = build(candidate("100"), bars=thrice)[0]

    assert tested_thrice.strength > tested_once.strength


def test_strength_stays_within_its_bounds() -> None:
    perfect = LevelConfig(
        zone_width=ZoneWidth.absolute("0.5"),
        merge_distance=ZoneWidth.absolute("1"),
        max_cluster_width=ZoneWidth.absolute("3"),
        source_quality={LevelSource.USER: 1.0, LevelSource.SWING_HIGH: 1.0},
        touch_saturation=1,
        confluence_saturation=1,
    )
    bars = [minute_bar(0, high="101", low="99")]

    (level,) = build(candidate("100"), bars=bars, config=perfect)

    assert level.strength == pytest.approx(100.0)


def test_confidence_reports_how_much_evidence_could_be_evaluated() -> None:
    """Touches need bars; with none supplied the component is undefined, not zero."""
    with_bars = build(candidate("100"), bars=[minute_bar(0, high="101", low="99")])[0]
    without_bars = build(candidate("100"))[0]

    assert with_bars.confidence == pytest.approx(1.0)
    assert without_bars.confidence == pytest.approx(0.75)
    assert "touches" not in dict(without_bars.strength_breakdown)


def test_a_requested_but_unavailable_component_lowers_confidence() -> None:
    config = LevelConfig(
        zone_width=ZoneWidth.absolute("0.5"),
        merge_distance=ZoneWidth.absolute("1"),
        max_cluster_width=ZoneWidth.absolute("3"),
        weights=StrengthWeights(volume_context=1.0),
    )

    (level,) = build(candidate("100"), bars=[minute_bar(0, high="101", low="99")], config=config)

    assert level.confidence == pytest.approx(4 / 5)
    assert "volume_context" not in dict(level.strength_breakdown)


def test_a_level_scored_with_nothing_available_is_honest_about_it() -> None:
    """Only an un-computable component is requested: score 0, confidence 0."""
    config = LevelConfig(
        zone_width=ZoneWidth.absolute("0.5"),
        merge_distance=ZoneWidth.absolute("1"),
        max_cluster_width=ZoneWidth.absolute("3"),
        weights=StrengthWeights(
            source_quality=0.0,
            touches=0.0,
            recency=0.0,
            confluence=0.0,
            volume_context=1.0,
        ),
    )

    (level,) = build(candidate("100"), config=config)

    assert level.strength == 0.0
    assert level.confidence == 0.0
    assert level.strength_breakdown == ()


def test_evidence_whose_weight_has_decayed_away_still_prices_the_level() -> None:
    """Underflowed weights fall back to the plain mean rather than dividing by zero."""
    bars = [minute_bar(index, high="101", low="99") for index in range(30)]
    config = LevelConfig(
        zone_width=ZoneWidth.absolute("0.5"),
        merge_distance=ZoneWidth.absolute("1"),
        max_cluster_width=ZoneWidth.absolute("3"),
        recency_decay=0.001,
    )

    (level,) = build(candidate("100", index=0), candidate("101", index=1), bars=bars, config=config)

    assert level.price == Decimal("100.5")


def test_recency_decay_favours_the_newer_evidence() -> None:
    bars = [minute_bar(index, high="101", low="99") for index in range(6)]
    config = LevelConfig(
        zone_width=ZoneWidth.absolute("0.5"),
        merge_distance=ZoneWidth.absolute("1"),
        max_cluster_width=ZoneWidth.absolute("3"),
        recency_decay=0.5,
    )

    stale = build(candidate("100", index=0), bars=bars, config=config)[0]
    fresh = build(candidate("100", index=5), bars=bars, config=config)[0]

    assert fresh.strength > stale.strength


# -- volume context (F2) ----------------------------------------------------

VOLUME_WEIGHTED = LevelConfig(
    zone_width=ZoneWidth.absolute("0.5"),
    merge_distance=ZoneWidth.absolute("1"),
    max_cluster_width=ZoneWidth.absolute("3"),
    weights=StrengthWeights(volume_context=1.0),
)


def busy_profile() -> object:
    """Volume concentrated at 100 and almost none at 110."""
    bars = [
        minute_bar(0, high="100.5", low="100.5", volume="10000"),
        minute_bar(1, high="110.5", low="110.5", volume="10"),
    ]
    return build_volume_profile(bars, config=ProfileConfig(bin_width=ZoneWidth.absolute("1")))


def test_a_level_where_the_volume_is_scores_higher_than_one_where_it_is_not() -> None:
    """The component `specs/07-levels.md` has carried at weight 0 since the MVP
    is computable once there is a profile (ADR 0025 D8)."""
    profile = busy_profile()
    bars = [minute_bar(0, high="101", low="99")]

    busy = build(candidate("100"), bars=bars, config=VOLUME_WEIGHTED, profile=profile)[0]
    quiet = build(candidate("110"), bars=bars, config=VOLUME_WEIGHTED, profile=profile)[0]

    assert dict(busy.strength_breakdown)["volume_context"] > 0
    assert busy.strength > quiet.strength


def test_volume_context_stays_undefined_without_a_profile() -> None:
    """Undefined, not zero — the level is reported as less confident instead of
    silently scored as if nothing traded there (ADR 0009 D7)."""
    bars = [minute_bar(0, high="101", low="99")]

    (level,) = build(candidate("100"), bars=bars, config=VOLUME_WEIGHTED)

    assert "volume_context" not in dict(level.strength_breakdown)
    assert level.confidence == pytest.approx(4 / 5)


def test_a_profile_does_not_move_a_level_whose_weights_ignore_it() -> None:
    """Weight 0 is the default, so supplying a profile changes nothing until
    somebody asks for it."""
    bars = [minute_bar(0, high="101", low="99")]

    with_profile = build(candidate("100"), bars=bars, profile=busy_profile())
    without = build(candidate("100"), bars=bars)

    assert with_profile == without


# -- invalidation -----------------------------------------------------------


def test_resistance_dies_on_a_close_above_its_zone() -> None:
    (level,) = build(candidate("100"))

    survived = invalidate(level, minute_bar(9, high="100.4", low="99", close="100.2"))
    broken = invalidate(level, minute_bar(9, high="102", low="99", close="101"))

    assert survived.status is LevelStatus.ACTIVE
    assert broken.status is LevelStatus.INVALIDATED
    assert broken.updated_at >= level.updated_at


def test_support_dies_on_a_close_below_its_zone() -> None:
    (level,) = build(candidate("100", kind=LevelKind.SUPPORT, source=LevelSource.SWING_LOW))

    broken = invalidate(level, minute_bar(9, high="100", low="98", close="98.5"))

    assert broken.status is LevelStatus.INVALIDATED


def test_a_wick_through_a_level_does_not_invalidate_it() -> None:
    (level,) = build(candidate("100"))

    survived = invalidate(level, minute_bar(9, high="120", low="99", close="100.1"))

    assert survived.status is LevelStatus.ACTIVE


def test_an_already_invalidated_level_is_left_alone() -> None:
    (level,) = build(candidate("100"))
    broken = invalidate(level, minute_bar(9, high="102", low="99", close="101"))

    assert invalidate(broken, minute_bar(10, high="103", low="102", close="102.5")) is broken


def test_a_partial_bar_cannot_invalidate_a_level() -> None:
    from dataclasses import replace

    (level,) = build(candidate("100"))
    partial = replace(minute_bar(9, high="102", low="99", close="101"), is_final=False)

    with pytest.raises(InvariantViolation, match="partial bar"):
        invalidate(level, partial)


def test_invalidation_returns_a_new_level_and_mutates_nothing() -> None:
    (level,) = build(candidate("100"))

    invalidate(level, minute_bar(9, high="102", low="99", close="101"))

    assert level.status is LevelStatus.ACTIVE


# -- determinism ------------------------------------------------------------


def test_the_same_evidence_always_produces_identical_levels() -> None:
    bars = [minute_bar(index, high="101", low="99") for index in range(4)]
    evidence = [
        candidate("100", source=LevelSource.SWING_HIGH),
        candidate("100.3", source=LevelSource.PREV_DAY_HIGH, index=1),
        candidate("104", source=LevelSource.YEAR_HIGH, index=2),
    ]

    assert build(*evidence, bars=bars) == build(*evidence, bars=bars)


def test_shuffled_evidence_produces_the_same_levels() -> None:
    evidence = [
        candidate("100", source=LevelSource.SWING_HIGH),
        candidate("100.3", source=LevelSource.PREV_DAY_HIGH, index=1),
        candidate("104", source=LevelSource.YEAR_HIGH, index=2),
    ]

    assert build(*evidence) == build(*reversed(evidence))


def test_candidates_dated_outside_the_bar_window_are_treated_as_the_oldest() -> None:
    """A daily level under an intraday series still has to score somehow."""
    bars = [minute_bar(index, high="101", low="99") for index in range(3)]
    config = LevelConfig(
        zone_width=ZoneWidth.absolute("0.5"),
        merge_distance=ZoneWidth.absolute("1"),
        max_cluster_width=ZoneWidth.absolute("3"),
        recency_decay=0.5,
    )

    outside = build(candidate("100", index=99), bars=bars, config=config)[0]
    oldest_inside = build(candidate("100", index=0), bars=bars, config=config)[0]

    assert outside.strength < oldest_inside.strength


def test_as_of_stamps_both_timestamps() -> None:
    (level,) = build(candidate("100"))

    assert level.created_at == AS_OF
    assert level.updated_at == AS_OF


def test_no_candidates_means_no_levels() -> None:
    assert build() == ()


def test_a_naive_as_of_is_refused() -> None:
    with pytest.raises(InvariantViolation, match="timezone"):
        build_levels(
            [candidate("100")],
            symbol=AAPL,
            timeframe=Timeframe.M1,
            as_of=datetime(2026, 8, 13, 20, 0),  # noqa: DTZ001 — the point of the test
        )


def test_the_default_configuration_is_used_when_none_is_given() -> None:
    levels = build_levels(
        [candidate("100")], symbol=AAPL, timeframe=Timeframe.M1, as_of=AS_OF, atr=2.0
    )

    (level,) = levels
    assert level.zone == Zone(low=Decimal("99.5"), high=Decimal("100.5"))  # 0.25 x ATR 2


def test_start_of_helper_keeps_candidate_dates_inside_the_session() -> None:
    assert start_of(0).hour == 13
