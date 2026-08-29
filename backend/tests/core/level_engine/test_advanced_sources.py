"""F2 level sources — specs/07-levels.md "Full version", ADR 0025.

Anchored VWAP, Fibonacci retracements, unfilled gaps and volume-profile nodes,
plus the one function that decides which of them a configuration consults.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from alphagate.core.level_engine import (
    AdvancedSources,
    LevelConfig,
    ProfileConfig,
    ZoneWidth,
    anchored_vwap,
    anchored_vwap_candidates,
    build_volume_profile,
    collect_candidates,
    fibonacci_candidates,
    gap_candidates,
    volume_profile_candidates,
)
from alphagate.core.levels import LevelKind, LevelSource
from alphagate.core.structure import SwingKind, SwingStatus
from tests.core.level_engine.synthetic import minute_bar, start_of, swing

RATIOS = (Decimal("0.5"), Decimal("0.618"))


def flat_bars(count: int) -> list[object]:
    """A run of bars that all trade at 100 — a VWAP with a knowable answer."""
    return [
        minute_bar(index, high="100", low="100", close="100", volume="100")
        for index in range(count)
    ]


# -- anchored VWAP ----------------------------------------------------------


def test_anchored_vwap_is_the_volume_weighted_typical_price_since_the_anchor() -> None:
    bars = [
        minute_bar(0, high="90", low="90", close="90", volume="1000"),
        minute_bar(1, high="102", low="99", close="102", volume="100"),
        minute_bar(2, high="106", low="103", close="106", volume="300"),
    ]

    value = anchored_vwap(bars, anchor_start_utc=start_of(1))

    # (101 x 100 + 105 x 300) / 400 — the bar before the anchor is excluded.
    assert value == Decimal("104")


def test_anchored_vwap_ignores_bars_before_its_anchor() -> None:
    bars = [
        minute_bar(0, high="200", low="200", close="200", volume="9999"),
        minute_bar(1, high="100", low="100", close="100", volume="100"),
    ]

    assert anchored_vwap(bars, anchor_start_utc=start_of(1)) == Decimal("100")


def test_anchored_vwap_of_no_volume_is_undefined() -> None:
    """A volume-weighted average of no volume is not zero."""
    bars = [minute_bar(0, high="100", low="100", volume="0")]

    assert anchored_vwap(bars, anchor_start_utc=start_of(0)) is None


def test_anchored_vwap_ignores_a_bar_that_has_not_closed() -> None:
    bars = [
        minute_bar(0, high="100", low="100", close="100", volume="100"),
        minute_bar(1, high="200", low="200", close="200", volume="100", is_final=False),
    ]

    assert anchored_vwap(bars, anchor_start_utc=start_of(0)) == Decimal("100")


def test_a_vwap_below_price_is_support_and_above_it_is_resistance() -> None:
    bars = flat_bars(4)
    swings = (swing(0, price="100", kind=SwingKind.LOW),)

    below = anchored_vwap_candidates(
        bars, swings, price=Decimal("120"), observed_at_utc=start_of(3)
    )
    above = anchored_vwap_candidates(bars, swings, price=Decimal("80"), observed_at_utc=start_of(3))

    assert [candidate.kind for candidate in below] == [LevelKind.SUPPORT]
    assert [candidate.kind for candidate in above] == [LevelKind.RESISTANCE]
    assert below[0].source is LevelSource.ANCHORED_VWAP


def test_a_vwap_is_anchored_only_at_confirmed_swings() -> None:
    """A candidate swing may still be revoked; anchoring at one would move the
    level under its own consumers (CLAUDE.md §12)."""
    bars = flat_bars(4)
    swings = (swing(0, price="100", status=SwingStatus.CANDIDATE),)

    assert (
        anchored_vwap_candidates(bars, swings, price=Decimal("99"), observed_at_utc=start_of(3))
        == ()
    )


def test_a_vwap_is_anchored_once_each_way() -> None:
    bars = flat_bars(6)
    swings = (
        swing(0, price="100", kind=SwingKind.LOW),
        swing(2, price="100", kind=SwingKind.HIGH),
    )

    candidates = anchored_vwap_candidates(
        bars, swings, price=Decimal("120"), observed_at_utc=start_of(5)
    )

    assert len(candidates) == 2


# -- Fibonacci --------------------------------------------------------------


def test_an_up_leg_retraces_to_support() -> None:
    swings = (
        swing(0, price="100", kind=SwingKind.LOW),
        swing(4, price="110", kind=SwingKind.HIGH),
    )

    candidates = fibonacci_candidates(swings, ratios=RATIOS)

    assert [candidate.price for candidate in candidates] == [Decimal("105"), Decimal("103.82")]
    assert {candidate.kind for candidate in candidates} == {LevelKind.SUPPORT}
    assert {candidate.source for candidate in candidates} == {LevelSource.FIBONACCI}


def test_a_down_leg_retraces_to_resistance() -> None:
    swings = (
        swing(0, price="110", kind=SwingKind.HIGH),
        swing(4, price="100", kind=SwingKind.LOW),
    )

    candidates = fibonacci_candidates(swings, ratios=RATIOS)

    assert [candidate.price for candidate in candidates] == [Decimal("105"), Decimal("106.18")]
    assert {candidate.kind for candidate in candidates} == {LevelKind.RESISTANCE}


def test_a_retracement_is_dated_to_the_pivot_that_completed_the_leg() -> None:
    """Dating it to the first pivot would claim the level was knowable before
    the swing that defines it existed (CLAUDE.md §12)."""
    swings = (
        swing(0, price="100", kind=SwingKind.LOW),
        swing(4, price="110", kind=SwingKind.HIGH),
    )

    (candidate, *_) = fibonacci_candidates(swings, ratios=RATIOS)

    assert candidate.observed_at_utc == start_of(4)


def test_the_leg_is_the_most_recent_one() -> None:
    swings = (
        swing(0, price="90", kind=SwingKind.LOW),
        swing(4, price="120", kind=SwingKind.HIGH),
        swing(8, price="100", kind=SwingKind.LOW),
    )

    candidates = fibonacci_candidates(swings, ratios=(Decimal("0.5"),))

    # The 120 → 100 leg, not the 90 → 120 one.
    assert [candidate.price for candidate in candidates] == [Decimal("110")]


def test_one_pivot_is_not_a_leg() -> None:
    assert fibonacci_candidates((swing(0, price="100"),), ratios=RATIOS) == ()


def test_two_pivots_the_same_way_are_not_a_leg() -> None:
    swings = (swing(0, price="100"), swing(4, price="110"))

    assert fibonacci_candidates(swings, ratios=RATIOS) == ()


def test_a_leg_of_no_height_has_no_geometry() -> None:
    swings = (
        swing(0, price="100", kind=SwingKind.LOW),
        swing(4, price="100", kind=SwingKind.HIGH),
    )

    assert fibonacci_candidates(swings, ratios=RATIOS) == ()


def test_a_ratio_outside_its_leg_is_refused() -> None:
    from alphagate.core.errors import InvariantViolation

    with pytest.raises(InvariantViolation, match="fibonacci ratio"):
        AdvancedSources(fibonacci_ratios=(Decimal("1.618"),))


# -- gaps -------------------------------------------------------------------


def test_a_gap_up_leaves_support_at_both_of_its_edges() -> None:
    bars = [
        minute_bar(0, high="100", low="99", close="100"),
        minute_bar(1, high="105", low="103", close="105"),
    ]

    candidates = gap_candidates(bars)

    assert [candidate.price for candidate in candidates] == [Decimal("100"), Decimal("103")]
    assert {candidate.kind for candidate in candidates} == {LevelKind.SUPPORT}
    assert {candidate.source for candidate in candidates} == {LevelSource.GAP}


def test_a_gap_down_leaves_resistance() -> None:
    bars = [
        minute_bar(0, high="105", low="103", close="103"),
        minute_bar(1, high="100", low="99", close="99"),
    ]

    candidates = gap_candidates(bars)

    assert [candidate.price for candidate in candidates] == [Decimal("100"), Decimal("103")]
    assert {candidate.kind for candidate in candidates} == {LevelKind.RESISTANCE}


def test_a_gap_price_has_traded_back_through_is_not_a_level() -> None:
    bars = [
        minute_bar(0, high="100", low="99", close="100"),
        minute_bar(1, high="105", low="103", close="105"),
        minute_bar(2, high="104", low="98", close="99"),  # back through the far edge
    ]

    assert gap_candidates(bars) == ()


def test_a_partly_filled_gap_keeps_the_edges_it_had() -> None:
    """Half a gap is still a gap; shrinking the zone would move the level
    without the evidence changing (ADR 0025 D4)."""
    bars = [
        minute_bar(0, high="100", low="99", close="100"),
        minute_bar(1, high="105", low="103", close="105"),
        minute_bar(2, high="105", low="101.5", close="104"),  # into it, not through
    ]

    assert [candidate.price for candidate in gap_candidates(bars)] == [
        Decimal("100"),
        Decimal("103"),
    ]


def test_a_gap_narrower_than_the_floor_is_not_a_level() -> None:
    """Without a floor every one-tick hole becomes a zone."""
    bars = [
        minute_bar(0, high="100", low="99", close="100"),
        minute_bar(1, high="102", low="100.01", close="102"),
    ]

    assert gap_candidates(bars, minimum=ZoneWidth.percent("0.1")) == ()
    assert gap_candidates(bars, minimum=None) != ()


def test_a_gap_is_dated_to_the_bar_that_opened_it() -> None:
    bars = [
        minute_bar(0, high="100", low="99", close="100"),
        minute_bar(1, high="105", low="103", close="105"),
    ]

    assert {candidate.observed_at_utc for candidate in gap_candidates(bars)} == {start_of(1)}


def test_touching_bars_leave_no_gap() -> None:
    bars = [
        minute_bar(0, high="100", low="99", close="100"),
        minute_bar(1, high="102", low="100", close="102"),
    ]

    assert gap_candidates(bars) == ()


def test_one_bar_cannot_gap() -> None:
    assert gap_candidates([minute_bar(0, high="100", low="99")]) == ()


# -- volume profile candidates ----------------------------------------------


def test_profile_nodes_become_candidates_on_the_side_price_is_on() -> None:
    bars = [
        minute_bar(0, high="100.5", low="100.5", volume="1000"),
        minute_bar(1, high="105.5", low="105.5", volume="1000"),
    ]
    profile = build_volume_profile(bars, config=ProfileConfig(bin_width=ZoneWidth.absolute("1")))

    assert profile is not None
    candidates = volume_profile_candidates(
        profile, price=Decimal("110"), observed_at_utc=start_of(1)
    )

    sources = {candidate.source for candidate in candidates}
    assert LevelSource.VOLUME_POC in sources
    assert LevelSource.VOLUME_LVN in sources
    # Everything traded below 110, so everything is support.
    assert {candidate.kind for candidate in candidates} == {LevelKind.SUPPORT}


def test_a_node_exactly_at_price_reads_as_resistance() -> None:
    """The STRICT convention used for structure labels and moving averages."""
    bars = [minute_bar(0, high="100.5", low="100.5", volume="1000")]
    profile = build_volume_profile(bars, config=ProfileConfig(bin_width=ZoneWidth.absolute("1")))

    assert profile is not None
    (candidate,) = volume_profile_candidates(
        profile, price=Decimal("100.5"), observed_at_utc=start_of(0)
    )

    assert candidate.kind is LevelKind.RESISTANCE


# -- collection -------------------------------------------------------------


BARS = [
    minute_bar(0, high="100", low="99", close="100", volume="1000"),
    minute_bar(1, high="105", low="103", close="105", volume="1000"),
    minute_bar(2, high="106", low="104", close="104", volume="1000"),
]
SWINGS = (
    swing(0, price="99", kind=SwingKind.LOW),
    swing(2, price="106", kind=SwingKind.HIGH),
)


def sources_of(config: LevelConfig) -> set[LevelSource]:
    profile = build_volume_profile(BARS, config=config.profile)
    return {
        candidate.source
        for candidate in collect_candidates(BARS, swings=SWINGS, profile=profile, config=config)
    }


def test_every_advanced_source_contributes_by_default() -> None:
    found = sources_of(LevelConfig())

    assert {
        LevelSource.SWING_HIGH,
        LevelSource.SWING_LOW,
        LevelSource.GAP,
        LevelSource.FIBONACCI,
        LevelSource.ANCHORED_VWAP,
        LevelSource.VOLUME_POC,
    } <= found


def test_a_switched_off_source_contributes_nothing() -> None:
    config = LevelConfig(
        advanced=AdvancedSources(
            volume_profile=False, anchored_vwap=False, fibonacci=False, gaps=False
        )
    )

    assert sources_of(config) == {LevelSource.SWING_HIGH, LevelSource.SWING_LOW}


def test_collection_without_a_profile_still_collects_everything_else() -> None:
    found = {
        candidate.source
        for candidate in collect_candidates(BARS, swings=SWINGS, profile=None, config=LevelConfig())
    }

    assert LevelSource.VOLUME_POC not in found
    assert LevelSource.GAP in found


def test_without_bars_only_the_sources_that_need_none_contribute() -> None:
    """The dynamic sources are dated to the newest bar. With no bars there is no
    "now" to date them to, and inventing one would date evidence to a close that
    never happened."""
    found = {candidate.source for candidate in collect_candidates([], swings=SWINGS)}

    assert found == {LevelSource.SWING_HIGH, LevelSource.SWING_LOW}


def test_dynamic_sources_are_dated_to_the_newest_closed_bar() -> None:
    """A VWAP recomputed every bar is evidence about now, not about its anchor
    (ADR 0025 D5)."""
    dynamic = [
        candidate
        for candidate in collect_candidates(BARS, swings=SWINGS, config=LevelConfig())
        if candidate.source is LevelSource.ANCHORED_VWAP
    ]

    assert dynamic
    assert {candidate.observed_at_utc for candidate in dynamic} == {start_of(2)}
