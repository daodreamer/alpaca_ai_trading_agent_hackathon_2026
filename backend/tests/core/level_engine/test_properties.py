"""Level invariants over generated evidence — specs/14-testing.md.

The claims here are the ones the Phase 4 exit criterion rests on: clustering
partitions its input, never exceeds the configured width, and does not depend on
the order the evidence arrived in.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from decimal import Decimal

from hypothesis import given
from hypothesis import strategies as st

from alphagate.core.level_engine import (
    LevelCandidate,
    LevelConfig,
    ZoneWidth,
    build_levels,
    cluster_candidates,
)
from alphagate.core.levels import LevelKind, LevelSource
from alphagate.core.time_model import Timeframe
from tests.core.level_engine.synthetic import AAPL, start_of

AS_OF = datetime(2026, 8, 13, 20, 0, tzinfo=UTC)

CONFIG = LevelConfig(
    zone_width=ZoneWidth.absolute("0.5"),
    merge_distance=ZoneWidth.absolute("1"),
    max_cluster_width=ZoneWidth.absolute("3"),
)

evidence = st.lists(
    st.tuples(
        st.decimals(min_value=Decimal("10"), max_value=Decimal("500"), places=2),
        st.sampled_from(
            [
                (LevelKind.RESISTANCE, LevelSource.SWING_HIGH),
                (LevelKind.RESISTANCE, LevelSource.PREV_DAY_HIGH),
                (LevelKind.SUPPORT, LevelSource.SWING_LOW),
                (LevelKind.SUPPORT, LevelSource.YEAR_LOW),
            ]
        ),
        st.integers(min_value=0, max_value=30),
    ),
    max_size=25,
)


def candidates_from(rows: list[tuple[Decimal, tuple[LevelKind, LevelSource], int]]):  # noqa: ANN201
    return [
        LevelCandidate(price=price, kind=kind, source=source, observed_at_utc=start_of(index))
        for price, (kind, source), index in rows
    ]


@given(evidence)
def test_clustering_partitions_its_input(rows: list) -> None:  # type: ignore[type-arg]
    candidates = candidates_from(rows)
    clusters = cluster_candidates(candidates, config=CONFIG, atr=None)

    flattened = [member for group in clusters for member in group]
    assert sorted(flattened, key=lambda c: c.sort_key) == sorted(
        candidates, key=lambda c: c.sort_key
    )


@given(evidence)
def test_no_cluster_is_wider_than_the_cap(rows: list) -> None:  # type: ignore[type-arg]
    clusters = cluster_candidates(candidates_from(rows), config=CONFIG, atr=None)

    for group in clusters:
        assert group[-1].price - group[0].price <= Decimal("3")


@given(evidence)
def test_a_cluster_never_mixes_support_with_resistance(rows: list) -> None:  # type: ignore[type-arg]
    clusters = cluster_candidates(candidates_from(rows), config=CONFIG, atr=None)

    for group in clusters:
        assert len({member.kind for member in group}) == 1


@given(evidence, st.randoms(use_true_random=False))
def test_clustering_ignores_input_order(rows: list, shuffler: object) -> None:  # type: ignore[type-arg]
    candidates = candidates_from(rows)
    shuffled = candidates[:]
    shuffler.shuffle(shuffled)  # type: ignore[attr-defined]

    assert cluster_candidates(shuffled, config=CONFIG, atr=None) == cluster_candidates(
        candidates, config=CONFIG, atr=None
    )


@given(evidence)
def test_every_level_scores_inside_its_declared_bounds(rows: list) -> None:  # type: ignore[type-arg]
    levels = build_levels(
        candidates_from(rows),
        symbol=AAPL,
        timeframe=Timeframe.M1,
        as_of=AS_OF,
        config=CONFIG,
    )

    for level in levels:
        assert 0.0 <= level.strength <= 100.0
        assert 0.0 <= level.confidence <= 1.0
        assert math.fsum(value for _, value in level.strength_breakdown) <= 100.0 + 1e-9


@given(evidence)
def test_a_levels_price_always_lies_inside_its_own_zone(rows: list) -> None:  # type: ignore[type-arg]
    levels = build_levels(
        candidates_from(rows),
        symbol=AAPL,
        timeframe=Timeframe.M1,
        as_of=AS_OF,
        config=CONFIG,
    )

    for level in levels:
        assert level.effective_zone.contains(level.price)


@given(evidence)
def test_rebuilding_produces_equal_levels_not_merely_similar_ones(rows: list) -> None:  # type: ignore[type-arg]
    candidates = candidates_from(rows)

    def run() -> tuple[object, ...]:
        return build_levels(
            candidates,
            symbol=AAPL,
            timeframe=Timeframe.M1,
            as_of=AS_OF,
            config=CONFIG,
        )

    assert run() == run()


@given(evidence)
def test_level_ids_are_unique_within_one_build(rows: list) -> None:  # type: ignore[type-arg]
    levels = build_levels(
        candidates_from(rows),
        symbol=AAPL,
        timeframe=Timeframe.M1,
        as_of=AS_OF,
        config=CONFIG,
    )

    assert len({level.id for level in levels}) == len(levels)
