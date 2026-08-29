"""Engine-level guarantees — ADR 0008 D1/D3/D6/D8.

The Phase 4 exit criterion is "no look-ahead bias in replay tests". That is what
most of this module is: replay the fixture one bar at a time and assert that
nothing the engine said could have been said any earlier, and that nothing it
said was ever taken back.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import timedelta

import pytest

from alphagate.core.bar import AdjustmentMode, Bar, Feed
from alphagate.core.errors import InvariantViolation
from alphagate.core.identifiers import ticker
from alphagate.core.structure import StructureEngine, SwingStatus
from alphagate.core.time_model import Timeframe
from tests.core.structure.synthetic import bar_at, swing_bars

ZIGZAG = swing_bars(["10", "12", "10", "14", "8", "13", "7", "15", "6", "16", "5"])


def run(engine: StructureEngine, bars: Sequence[Bar]) -> StructureEngine:
    for bar in bars:
        engine.update(bar)
    return engine


# -- no look-ahead ----------------------------------------------------------


def test_nothing_is_reported_before_the_bar_that_could_know_it() -> None:
    engine = StructureEngine(left_bars=1, right_bars=1)

    for bar in ZIGZAG:
        update = engine.update(bar)

        for swing in update.confirmed_swings:
            assert swing.detected_at_utc == bar.start_time_utc
            assert swing.bar_start_utc < bar.start_time_utc
        for event in update.breaks:
            assert event.bar_start_utc == bar.start_time_utc
        for swing in engine.swings:
            assert swing.detected_at_utc <= bar.start_time_utc


def test_confirmation_lags_the_pivot_by_exactly_the_right_window() -> None:
    for right in (1, 2, 3):
        engine = run(StructureEngine(left_bars=1, right_bars=right), ZIGZAG)

        for swing in engine.swings:
            lag = swing.detected_at_utc - swing.bar_start_utc
            assert lag == timedelta(minutes=right)


def test_a_break_is_never_dated_before_the_level_it_broke() -> None:
    engine = run(StructureEngine(left_bars=1, right_bars=1), ZIGZAG)

    for event in engine.breaks:
        assert event.level_bar_start_utc < event.bar_start_utc


def test_every_break_refers_to_a_swing_the_engine_had_already_confirmed() -> None:
    engine = run(StructureEngine(left_bars=1, right_bars=1), ZIGZAG)
    confirmed = {(swing.bar_start_utc, swing.price) for swing in engine.swings}

    for event in engine.breaks:
        assert (event.level_bar_start_utc, event.level_price) in confirmed


# -- append-only ------------------------------------------------------------


def test_confirmed_structure_is_never_revised_or_withdrawn() -> None:
    """ADR 0008 D6 — each snapshot extends the previous one, never rewrites it."""
    engine = StructureEngine(left_bars=1, right_bars=1)
    previous: tuple[object, ...] = ()

    for bar in ZIGZAG:
        engine.update(bar)
        current = engine.swings
        assert current[: len(previous)] == previous
        previous = current


def test_every_recorded_swing_is_confirmed() -> None:
    engine = run(StructureEngine(left_bars=1, right_bars=1), ZIGZAG)

    assert all(swing.status is SwingStatus.CONFIRMED for swing in engine.swings)


def test_break_events_accumulate_in_order() -> None:
    engine = run(StructureEngine(left_bars=1, right_bars=1), ZIGZAG)
    starts = [event.bar_start_utc for event in engine.breaks]

    assert starts == sorted(starts)


# -- determinism ------------------------------------------------------------


def test_the_same_bars_produce_the_same_structure() -> None:
    first = run(StructureEngine(left_bars=2, right_bars=2, minimum_move_atr=0.5), ZIGZAG)
    second = run(StructureEngine(left_bars=2, right_bars=2, minimum_move_atr=0.5), ZIGZAG)

    assert first.swings == second.swings
    assert first.breaks == second.breaks


def test_an_update_is_returned_for_every_bar_even_when_nothing_happened() -> None:
    engine = StructureEngine(left_bars=1, right_bars=1)

    update = engine.update(ZIGZAG[0])

    assert update is not None
    assert update.confirmed_swings == ()
    assert update.breaks == ()
    assert update.candidate_high is None
    assert update.candidate_low is None


# -- stream discipline (inherited from BarConsumer) --------------------------


def test_a_repeated_bar_is_refused() -> None:
    engine = run(StructureEngine(left_bars=1, right_bars=1), ZIGZAG[:4])

    with pytest.raises(InvariantViolation, match="strictly increasing"):
        engine.update(ZIGZAG[3])


def test_a_bar_from_another_series_is_refused() -> None:
    engine = run(StructureEngine(left_bars=1, right_bars=1), ZIGZAG[:4])
    foreign = bar_at(9, open_="10", high="11", low="9", close="10", symbol=ticker("MSFT"))

    with pytest.raises(InvariantViolation, match="different series"):
        engine.update(foreign)


@pytest.mark.parametrize(
    "difference",
    [
        {"timeframe": Timeframe.M5},
        {"feed": Feed.IEX},
        {"adjustment_mode": AdjustmentMode.ADJUSTED},
    ],
    ids=["timeframe", "feed", "adjustment"],
)
def test_the_series_key_covers_more_than_the_symbol(difference: dict[str, object]) -> None:
    engine = run(StructureEngine(left_bars=1, right_bars=1), ZIGZAG[:4])
    foreign = bar_at(9, open_="10", high="11", low="9", close="10", **difference)  # type: ignore[arg-type]

    with pytest.raises(InvariantViolation, match="different series"):
        engine.update(foreign)


def test_a_partial_bar_is_refused_by_default() -> None:
    engine = run(StructureEngine(left_bars=1, right_bars=1), ZIGZAG[:4])
    partial = bar_at(4, open_="10", high="11", low="9", close="10", is_final=False)

    with pytest.raises(InvariantViolation, match="partial"):
        engine.update(partial)


def test_a_provisional_update_leaves_the_engine_untouched() -> None:
    engine = run(StructureEngine(left_bars=1, right_bars=1, allow_partial=True), ZIGZAG[:4])
    before = (engine.swings, engine.breaks, engine.committed_bars)

    engine.update(bar_at(4, open_="99", high="99", low="99", close="99", is_final=False))

    assert (engine.swings, engine.breaks, engine.committed_bars) == before


def test_intrabar_ticks_do_not_change_what_the_closed_bar_produces() -> None:
    quiet = run(StructureEngine(left_bars=1, right_bars=1), ZIGZAG)

    ticked = StructureEngine(left_bars=1, right_bars=1, allow_partial=True)
    for index, bar in enumerate(ZIGZAG):
        for tick in ("1", "999"):
            ticked.update(
                bar_at(index, open_=tick, high=tick, low=tick, close=tick, is_final=False)
            )
        ticked.update(bar)

    assert ticked.swings == quiet.swings
    assert ticked.breaks == quiet.breaks


# -- identity ---------------------------------------------------------------


def test_the_spec_key_records_the_whole_configuration() -> None:
    engine = StructureEngine(left_bars=3, right_bars=3)

    assert engine.spec.key == (
        "structure(left=3,right=3,min_move_atr=0.0,atr=14,equal=STRICT,closes=1,wicks=False)"
    )


def test_the_spec_key_reflects_a_changed_profile() -> None:
    from alphagate.core.structure import BreakPolicy, EqualPricePolicy

    engine = StructureEngine(
        left_bars=2,
        right_bars=4,
        minimum_move_atr=1.5,
        atr_period=20,
        equal_prices=EqualPricePolicy.INCLUSIVE,
        breaks=BreakPolicy(closes_required=2, use_wicks=True),
    )

    assert engine.spec.key == (
        "structure(left=2,right=4,min_move_atr=1.5,atr=20,equal=INCLUSIVE,closes=2,wicks=True)"
    )
