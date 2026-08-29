"""Swing detection and structure labels — ADR 0008 D2/D3/D4/D5.

The canonical fixture is a zig-zag written as a list of centre prices; each bar
spans `centre ± 1`, so a swing high prices at `centre + 1` and a swing low at
`centre - 1`.
"""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal

import pytest

from alphagate.core.bar import Bar
from alphagate.core.errors import InvariantViolation
from alphagate.core.structure import (
    EqualPricePolicy,
    StructureEngine,
    StructureLabel,
    SwingKind,
    SwingStatus,
)
from tests.core.structure.synthetic import bar_at, start_of, swing_bars

ZIGZAG = ["10", "12", "10", "14", "8", "13", "7", "15", "6"]


def run(engine: StructureEngine, bars: Sequence[Bar]) -> StructureEngine:
    for bar in bars:
        engine.update(bar)
    return engine


def prices(engine: StructureEngine, kind: SwingKind) -> list[Decimal]:
    return [swing.price for swing in engine.swings if swing.kind is kind]


def labels(engine: StructureEngine, kind: SwingKind) -> list[StructureLabel | None]:
    return [swing.label for swing in engine.swings if swing.kind is kind]


# -- pivots -----------------------------------------------------------------


def test_a_swing_high_is_confirmed_right_bars_after_its_pivot() -> None:
    engine = StructureEngine(left_bars=1, right_bars=1)
    bars = swing_bars(["10", "12", "10"])

    assert engine.update(bars[0]).confirmed_swings == ()
    assert engine.update(bars[1]).confirmed_swings == ()

    (swing,) = engine.update(bars[2]).confirmed_swings
    assert swing.kind is SwingKind.HIGH
    assert swing.status is SwingStatus.CONFIRMED
    assert swing.price == Decimal("13")  # centre 12 + margin 1
    assert swing.bar_start_utc == start_of(1)
    assert swing.detected_at_utc == start_of(2)


def test_a_swing_low_mirrors_the_high_rule() -> None:
    engine = run(StructureEngine(left_bars=1, right_bars=1), swing_bars(["12", "10", "12"]))

    (swing,) = engine.swings
    assert swing.kind is SwingKind.LOW
    assert swing.price == Decimal("9")  # centre 10 - margin 1
    assert swing.bar_start_utc == start_of(1)


def test_the_full_zigzag_produces_the_expected_swings() -> None:
    engine = run(StructureEngine(left_bars=1, right_bars=1), swing_bars(ZIGZAG))

    assert prices(engine, SwingKind.HIGH) == [Decimal(p) for p in ("13", "15", "14", "16")]
    assert prices(engine, SwingKind.LOW) == [Decimal(p) for p in ("9", "7", "6")]


def test_swings_are_reported_in_pivot_order() -> None:
    engine = run(StructureEngine(left_bars=1, right_bars=1), swing_bars(ZIGZAG))
    starts = [swing.bar_start_utc for swing in engine.swings]

    assert starts == sorted(starts)


def test_a_plateau_of_equal_highs_yields_one_pivot_at_its_last_bar() -> None:
    """Left-inclusive, right-strict (ADR 0008 D2): a flat top is one swing, not three."""
    engine = run(
        StructureEngine(left_bars=1, right_bars=1), swing_bars(["10", "12", "12", "12", "10"])
    )

    (swing,) = engine.swings
    assert swing.kind is SwingKind.HIGH
    assert swing.bar_start_utc == start_of(3)


def test_an_outside_bar_is_both_a_swing_high_and_a_swing_low() -> None:
    bars = (
        bar_at(0, open_="10", high="10", low="9", close="10"),
        bar_at(1, open_="10", high="12", low="7", close="10"),
        bar_at(2, open_="10", high="10", low="9", close="10"),
    )
    engine = run(StructureEngine(left_bars=1, right_bars=1), bars)

    assert {swing.kind for swing in engine.swings} == {SwingKind.HIGH, SwingKind.LOW}
    assert all(swing.bar_start_utc == start_of(1) for swing in engine.swings)


def test_wider_windows_need_more_bars_on_both_sides() -> None:
    engine = StructureEngine(left_bars=2, right_bars=2)
    bars = swing_bars(["9", "10", "12", "10", "9"])

    run(engine, bars[:4])
    assert engine.swings == ()

    engine.update(bars[4])
    (swing,) = [s for s in engine.swings if s.kind is SwingKind.HIGH]
    assert swing.bar_start_utc == start_of(2)
    assert swing.detected_at_utc == start_of(4)


def test_warmup_is_the_pivot_window() -> None:
    assert StructureEngine(left_bars=3, right_bars=2).warmup == 6


@pytest.mark.parametrize(("left", "right"), [(0, 1), (1, 0), (-1, 2)])
def test_non_positive_windows_are_refused(left: int, right: int) -> None:
    with pytest.raises(InvariantViolation, match="bars"):
        StructureEngine(left_bars=left, right_bars=right)


# -- candidates -------------------------------------------------------------


def test_a_forming_swing_is_offered_as_a_candidate_before_it_confirms() -> None:
    engine = StructureEngine(left_bars=1, right_bars=1)
    bars = swing_bars(["10", "12", "10"])

    engine.update(bars[0])
    update = engine.update(bars[1])

    assert update.confirmed_swings == ()
    candidate = update.candidate_high
    assert candidate is not None
    assert candidate.status is SwingStatus.CANDIDATE
    assert candidate.bar_start_utc == start_of(1)
    assert candidate.detected_at_utc == start_of(1)
    assert candidate.label is None


def test_a_candidate_is_not_recorded_as_structure() -> None:
    engine = run(StructureEngine(left_bars=1, right_bars=1), swing_bars(["10", "12"]))

    assert engine.swings == ()
    assert engine.structural_high is None


def test_a_candidate_that_is_exceeded_never_becomes_a_swing() -> None:
    engine = run(StructureEngine(left_bars=1, right_bars=1), swing_bars(["10", "12", "14", "16"]))

    assert engine.swings == ()


# -- labels -----------------------------------------------------------------


def test_labels_compare_against_the_previous_swing_of_the_same_kind() -> None:
    engine = run(StructureEngine(left_bars=1, right_bars=1), swing_bars(ZIGZAG))

    assert labels(engine, SwingKind.HIGH) == [
        None,
        StructureLabel.HH,
        StructureLabel.LH,
        StructureLabel.HH,
    ]
    assert labels(engine, SwingKind.LOW) == [None, StructureLabel.LL, StructureLabel.LL]


def test_the_first_swing_of_each_kind_has_no_label() -> None:
    engine = run(StructureEngine(left_bars=1, right_bars=1), swing_bars(["10", "12", "10", "12"]))

    assert engine.swings[0].label is None


def test_a_higher_low_is_labelled_hl() -> None:
    engine = run(
        StructureEngine(left_bars=1, right_bars=1), swing_bars(["12", "8", "14", "10", "15"])
    )

    assert labels(engine, SwingKind.LOW) == [None, StructureLabel.HL]


EQUAL_HIGHS = ["10", "12", "10", "12", "10"]
EQUAL_LOWS = ["12", "10", "12", "10", "12"]


def test_an_equal_high_is_a_lower_high_under_the_strict_policy() -> None:
    engine = run(
        StructureEngine(left_bars=1, right_bars=1, equal_prices=EqualPricePolicy.STRICT),
        swing_bars(EQUAL_HIGHS),
    )

    assert labels(engine, SwingKind.HIGH) == [None, StructureLabel.LH]


def test_an_equal_high_is_a_higher_high_under_the_inclusive_policy() -> None:
    engine = run(
        StructureEngine(left_bars=1, right_bars=1, equal_prices=EqualPricePolicy.INCLUSIVE),
        swing_bars(EQUAL_HIGHS),
    )

    assert labels(engine, SwingKind.HIGH) == [None, StructureLabel.HH]


def test_an_equal_low_follows_the_same_policy() -> None:
    strict = run(
        StructureEngine(left_bars=1, right_bars=1, equal_prices=EqualPricePolicy.STRICT),
        swing_bars(EQUAL_LOWS),
    )
    inclusive = run(
        StructureEngine(left_bars=1, right_bars=1, equal_prices=EqualPricePolicy.INCLUSIVE),
        swing_bars(EQUAL_LOWS),
    )

    assert labels(strict, SwingKind.LOW)[1] is StructureLabel.LL
    assert labels(inclusive, SwingKind.LOW)[1] is StructureLabel.HL


# -- minimum move -----------------------------------------------------------

# ATR(2) is warm from bar 2, so the first pivot here sits at bar 3 deliberately:
# a filtered engine must not accept a pivot it cannot measure (ADR 0008 D4).
NOISY = ["10", "10", "10", "14", "10", "14.01", "10", "30", "10"]


def test_the_noise_filter_is_off_by_default() -> None:
    engine = run(StructureEngine(left_bars=1, right_bars=1), swing_bars(NOISY))

    assert prices(engine, SwingKind.HIGH) == [Decimal(p) for p in ("15", "15.01", "31")]


def test_a_swing_too_close_to_the_previous_same_kind_swing_is_dropped() -> None:
    """ADR 0008 D4 — the move is measured against the previous high, in ATR."""
    engine = run(
        StructureEngine(left_bars=1, right_bars=1, minimum_move_atr=1.0, atr_period=2),
        swing_bars(NOISY),
    )

    assert prices(engine, SwingKind.HIGH) == [Decimal("15"), Decimal("31")]


def test_a_pivot_arriving_before_atr_is_warm_is_dropped_when_the_filter_is_on() -> None:
    """A filter that switches on mid-series would make structure depend on the cut."""
    engine = run(
        StructureEngine(left_bars=1, right_bars=1, minimum_move_atr=1.0, atr_period=50),
        swing_bars(ZIGZAG),
    )

    assert engine.swings == ()


def test_the_filter_warmup_includes_atr() -> None:
    filtered = StructureEngine(left_bars=1, right_bars=1, minimum_move_atr=1.0, atr_period=14)
    unfiltered = StructureEngine(left_bars=1, right_bars=1, atr_period=14)

    assert filtered.warmup == 15
    assert unfiltered.warmup == 3
