"""Break of structure — ADR 0008 D7.

Bars are written out one by one here rather than generated: what makes a break
test meaningful is the exact relationship between a close, a high and a level,
and a helper that hides it would hide the test.
"""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal

from alphagate.core.bar import Bar
from alphagate.core.structure import BreakKind, BreakPolicy, StructureEngine
from tests.core.structure.synthetic import bar_at, start_of

# b1 is a swing high at 12, confirmed by b2. b3 stays under it.
SETUP_HIGH = (
    bar_at(0, open_="10", high="11", low="9", close="10"),
    bar_at(1, open_="11", high="12", low="10", close="11"),
    bar_at(2, open_="10", high="11", low="9", close="10"),
    bar_at(3, open_="11", high="11.5", low="10", close="11"),
)

# Mirror: b1 is a swing low at 8, confirmed by b2.
SETUP_LOW = (
    bar_at(0, open_="10", high="11", low="9", close="10"),
    bar_at(1, open_="9", high="10", low="8", close="9"),
    bar_at(2, open_="10", high="11", low="9", close="10"),
    bar_at(3, open_="9.5", high="10.5", low="8.5", close="9.5"),
)


def run(engine: StructureEngine, bars: Sequence[Bar]) -> StructureEngine:
    for bar in bars:
        engine.update(bar)
    return engine


def test_a_close_beyond_the_structural_high_is_a_breakout() -> None:
    engine = run(StructureEngine(left_bars=1, right_bars=1), SETUP_HIGH)
    assert engine.structural_high is not None
    assert engine.structural_high.price == Decimal("12")

    update = engine.update(bar_at(4, open_="12", high="12.5", low="11", close="12.4"))

    (event,) = update.breaks
    assert event.kind is BreakKind.BREAKOUT
    assert event.level_price == Decimal("12")
    assert event.level_bar_start_utc == start_of(1)
    assert event.bar_start_utc == start_of(4)
    assert event.closes == 1
    assert not event.volume_confirmed


def test_a_close_beyond_the_structural_low_is_a_breakdown() -> None:
    engine = run(StructureEngine(left_bars=1, right_bars=1), SETUP_LOW)
    assert engine.structural_low is not None
    assert engine.structural_low.price == Decimal("8")

    update = engine.update(bar_at(4, open_="9", high="9", low="7.5", close="7.9"))

    (event,) = update.breaks
    assert event.kind is BreakKind.BREAKDOWN
    assert event.level_price == Decimal("8")


def test_a_close_exactly_on_the_level_does_not_break_it() -> None:
    engine = run(StructureEngine(left_bars=1, right_bars=1), SETUP_HIGH)

    update = engine.update(bar_at(4, open_="11", high="12.5", low="11", close="12"))

    assert update.breaks == ()


def test_a_wick_through_the_level_does_not_break_it_by_default() -> None:
    engine = run(StructureEngine(left_bars=1, right_bars=1), SETUP_HIGH)

    update = engine.update(bar_at(4, open_="11", high="13", low="11", close="11.5"))

    assert update.breaks == ()


def test_a_wick_breaks_the_level_when_the_profile_enables_it() -> None:
    engine = run(
        StructureEngine(left_bars=1, right_bars=1, breaks=BreakPolicy(use_wicks=True)),
        SETUP_HIGH,
    )

    update = engine.update(bar_at(4, open_="11", high="13", low="11", close="11.5"))

    (event,) = update.breaks
    assert event.kind is BreakKind.BREAKOUT


def test_several_closes_can_be_required() -> None:
    engine = run(
        StructureEngine(left_bars=1, right_bars=1, breaks=BreakPolicy(closes_required=2)),
        SETUP_HIGH,
    )

    first = engine.update(bar_at(4, open_="12", high="12.5", low="11", close="12.4"))
    second = engine.update(bar_at(5, open_="12", high="12.6", low="11", close="12.5"))

    assert first.breaks == ()
    (event,) = second.breaks
    assert event.closes == 2
    assert event.bar_start_utc == start_of(5)


def test_a_close_back_inside_resets_the_count() -> None:
    engine = run(
        StructureEngine(left_bars=1, right_bars=1, breaks=BreakPolicy(closes_required=2)),
        SETUP_HIGH,
    )

    engine.update(bar_at(4, open_="12", high="12.5", low="11", close="12.4"))
    engine.update(bar_at(5, open_="12", high="12.6", low="11", close="11"))
    third = engine.update(bar_at(6, open_="11", high="12.7", low="10.5", close="12.4"))
    fourth = engine.update(bar_at(7, open_="12", high="12.8", low="11", close="12.5"))

    assert third.breaks == ()
    (event,) = fourth.breaks
    assert event.bar_start_utc == start_of(7)


def test_a_broken_level_is_consumed_and_cannot_break_again() -> None:
    """ADR 0008 D7 — otherwise price hovering above it emits an event every bar."""
    engine = run(StructureEngine(left_bars=1, right_bars=1), SETUP_HIGH)

    engine.update(bar_at(4, open_="12", high="12.5", low="11", close="12.4"))
    after = engine.update(bar_at(5, open_="12.4", high="12.6", low="11", close="12.6"))

    assert after.breaks == ()
    assert engine.structural_high is None
    assert len(engine.breaks) == 1


def test_no_break_is_possible_before_a_level_exists() -> None:
    engine = StructureEngine(left_bars=1, right_bars=1)

    updates = [engine.update(bar) for bar in SETUP_HIGH]

    assert all(update.breaks == () for update in updates)


# -- volume confirmation ----------------------------------------------------

VOLUME_POLICY = BreakPolicy(volume_multiple=2.0, volume_period=3)


def test_volume_confirmation_rejects_a_break_on_ordinary_volume() -> None:
    engine = run(StructureEngine(left_bars=1, right_bars=1, breaks=VOLUME_POLICY), SETUP_HIGH)

    update = engine.update(
        bar_at(4, open_="12", high="12.5", low="11", close="12.4", volume="1000")
    )

    assert update.breaks == ()


def test_volume_confirmation_accepts_a_break_on_heavy_volume() -> None:
    engine = run(StructureEngine(left_bars=1, right_bars=1, breaks=VOLUME_POLICY), SETUP_HIGH)

    update = engine.update(
        bar_at(4, open_="12", high="12.5", low="11", close="12.4", volume="2500")
    )

    (event,) = update.breaks
    assert event.volume_confirmed


def test_an_unverifiable_volume_condition_is_not_a_satisfied_one() -> None:
    """The average is not warm yet, so the break cannot be confirmed."""
    engine = run(
        StructureEngine(
            left_bars=1,
            right_bars=1,
            breaks=BreakPolicy(volume_multiple=2.0, volume_period=50),
        ),
        SETUP_HIGH,
    )

    update = engine.update(
        bar_at(4, open_="12", high="12.5", low="11", close="12.4", volume="1000000")
    )

    assert update.breaks == ()


def test_the_volume_average_excludes_the_breaking_bar() -> None:
    """Otherwise a heavy bar raises the very average it is measured against."""
    engine = run(StructureEngine(left_bars=1, right_bars=1, breaks=VOLUME_POLICY), SETUP_HIGH)

    # Preceding three bars average 1000; 2000 is exactly the 2.0 multiple. Were the
    # breaking bar included, the average would be 1250 and 2000 would fall short.
    update = engine.update(
        bar_at(4, open_="12", high="12.5", low="11", close="12.4", volume="2000")
    )

    (event,) = update.breaks
    assert event.volume_confirmed
