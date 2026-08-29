"""Clustering — ADR 0009 D4.

Two claims carry the phase's "deterministic levels" exit criterion: the result
never depends on the order candidates arrived in, and a chain of nearby
candidates cannot merge into one arbitrarily wide zone.
"""

from __future__ import annotations

import random
from decimal import Decimal

from alphagate.core.level_engine import LevelConfig, ZoneWidth, cluster_candidates
from alphagate.core.levels import LevelKind, LevelSource
from tests.core.level_engine.synthetic import candidate

ATR = Decimal("2")

# merge within 1.0, cap the total width at 3.0
CONFIG = LevelConfig(
    merge_distance=ZoneWidth.absolute("1"),
    max_cluster_width=ZoneWidth.absolute("3"),
)


def cluster(*candidates: object) -> tuple[tuple[Decimal, ...], ...]:
    groups = cluster_candidates(candidates, config=CONFIG, atr=ATR)  # type: ignore[arg-type]
    return tuple(tuple(member.price for member in group) for group in groups)


def test_candidates_within_the_merge_distance_form_one_cluster() -> None:
    assert cluster(candidate("100"), candidate("100.5", index=1)) == (
        (Decimal("100"), Decimal("100.5")),
    )


def test_candidates_beyond_the_merge_distance_stay_apart() -> None:
    assert cluster(candidate("100"), candidate("102", index=1)) == (
        (Decimal("100"),),
        (Decimal("102"),),
    )


def test_a_gap_exactly_equal_to_the_merge_distance_still_merges() -> None:
    assert cluster(candidate("100"), candidate("101", index=1)) == (
        (Decimal("100"), Decimal("101")),
    )


def test_a_chain_cannot_grow_past_the_width_cap() -> None:
    """Single linkage without a cap would make one 100–104 zone out of this."""
    chain = [candidate(str(100 + step), index=step) for step in range(5)]

    assert cluster(*chain) == (
        (Decimal("100"), Decimal("101"), Decimal("102"), Decimal("103")),
        (Decimal("104"),),
    )


def test_support_and_resistance_never_merge() -> None:
    """A swing low and a swing high at one price are two facts, not one level."""
    groups = cluster_candidates(
        (
            candidate("100", kind=LevelKind.RESISTANCE, source=LevelSource.SWING_HIGH),
            candidate("100", kind=LevelKind.SUPPORT, source=LevelSource.SWING_LOW, index=1),
        ),
        config=CONFIG,
        atr=ATR,
    )

    assert len(groups) == 2
    assert {group[0].kind for group in groups} == {LevelKind.SUPPORT, LevelKind.RESISTANCE}


def test_clustering_does_not_depend_on_input_order() -> None:
    candidates = [candidate(str(100 + step * 0.5), index=step) for step in range(8)]
    expected = cluster_candidates(candidates, config=CONFIG, atr=ATR)

    shuffler = random.Random(20260813)  # noqa: S311 — shuffling test input, not crypto
    for _ in range(10):
        shuffled = candidates[:]
        shuffler.shuffle(shuffled)
        assert cluster_candidates(shuffled, config=CONFIG, atr=ATR) == expected


def test_candidates_at_the_same_price_are_ordered_by_source_then_time() -> None:
    """A total order, so no tie can fall back on arrival order."""
    groups = cluster_candidates(
        (
            candidate("100", source=LevelSource.YEAR_HIGH, index=5),
            candidate("100", source=LevelSource.PREV_DAY_HIGH, index=9),
            candidate("100", source=LevelSource.PREV_DAY_HIGH, index=2),
        ),
        config=CONFIG,
        atr=ATR,
    )

    (group,) = groups
    assert [(member.source.value, member.observed_at_utc.minute) for member in group] == [
        ("PREV_DAY_HIGH", 32),
        ("PREV_DAY_HIGH", 39),
        ("YEAR_HIGH", 35),
    ]


def test_clusters_are_returned_in_price_order() -> None:
    prices = cluster(candidate("105"), candidate("100", index=1), candidate("102.5", index=2))

    assert prices == ((Decimal("100"),), (Decimal("102.5"),), (Decimal("105"),))


def test_no_candidates_means_no_clusters() -> None:
    assert cluster_candidates((), config=CONFIG, atr=ATR) == ()


def test_atr_based_distances_fall_back_to_no_limit_before_atr_is_warm() -> None:
    """With no ATR the distance is unknowable; candidates are not split on a guess."""
    config = LevelConfig(
        merge_distance=ZoneWidth.atr("0.5"), max_cluster_width=ZoneWidth.atr("1.5")
    )
    groups = cluster_candidates(
        (candidate("100"), candidate("140", index=1)), config=config, atr=None
    )

    assert len(groups) == 1
