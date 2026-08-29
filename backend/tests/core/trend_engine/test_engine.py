"""Engine-level guarantees — ADR 0010 D1/D5/D6/D7/D10/D11/D12.

The Phase 5 exit criteria are "confirmed trend events are stable under one-bar
noise" and "explanations available for UI". Most of this module is those two
sentences turned into assertions, plus the stream discipline every `BarConsumer`
inherits.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
from collections.abc import Sequence

import pytest

from alphagate.core.bar import AdjustmentMode, Bar, Feed
from alphagate.core.errors import InvariantViolation
from alphagate.core.identifiers import ticker
from alphagate.core.structure import StructureEngine
from alphagate.core.time_model import Timeframe
from alphagate.core.trend import TrendDirection, TrendPhase, is_valid_transition
from alphagate.core.trend_engine import (
    StrengthComponent,
    TrendConfig,
    TrendEngine,
    TrendEventKind,
    TrendUpdate,
    phase_steps,
)
from tests.core.trend_engine.synthetic import FAST, bar_at, bars_from, flat, ramp

RISING = bars_from(ramp(100, 14))
LONG_RISE = bars_from(ramp(100, 24))
FALLING = bars_from(ramp(140, 14, step=-1))
SIDEWAYS = bars_from(flat(100, 14))
REVERSING = bars_from(ramp(100, 12) + ramp(111, 14, step=-1))
CALM_THEN_RISING = bars_from(flat(100, 10) + ramp(101, 14))


def engine(**overrides: object) -> TrendEngine:
    config = dataclasses.replace(FAST, **overrides)  # type: ignore[arg-type]
    return TrendEngine(config=config, structure=StructureEngine(left_bars=1, right_bars=1))


def feed(instance: TrendEngine, bar: Bar) -> TrendUpdate:
    update = instance.update(bar)
    assert update is not None
    return update


def run(instance: TrendEngine, bars: Sequence[Bar]) -> list[TrendUpdate]:
    return [feed(instance, bar) for bar in bars]


def phases(updates: Sequence[TrendUpdate]) -> list[TrendPhase]:
    return [update.state.state for update in updates]


# -- the states themselves --------------------------------------------------


def test_the_engine_says_unknown_until_it_has_evidence() -> None:
    updates = run(engine(), RISING)

    assert updates[0].state.state is TrendPhase.UNKNOWN
    assert updates[0].state.reason_codes == ()
    assert updates[0].bias is None


def test_a_rising_market_confirms_a_bullish_trend() -> None:
    final = run(engine(), RISING)[-1].state

    assert final.direction is TrendDirection.BULLISH
    assert final.state is TrendPhase.STRONG_BULLISH
    assert final.confirmed_at is not None


def test_a_falling_market_confirms_the_mirror_trend() -> None:
    final = run(engine(), FALLING)[-1].state

    assert final.direction is TrendDirection.BEARISH
    assert final.state is TrendPhase.STRONG_BEARISH


def test_a_flat_market_settles_on_neutral_and_stays_there() -> None:
    """specs/08-trend.md — the sideways regime is a state, not an absence of one."""
    updates = run(engine(), SIDEWAYS)

    assert updates[-1].state.state is TrendPhase.NEUTRAL
    assert all(phase is not TrendPhase.UNKNOWN for phase in phases(updates)[-6:])
    assert not any(update.events for update in updates)


def visited(updates: Sequence[TrendUpdate]) -> list[TrendPhase]:
    """Every phase the engine passed through, intra-bar steps included.

    A bar may move several rungs at once (ADR 0010 D6), so the per-bar states
    are not the whole walk — and it is the walk that must not skip.
    """
    path = [TrendPhase.UNKNOWN]
    for update in updates:
        path.extend(phase_steps(path[-1], update.state.state))
    return path


def test_the_engine_walks_the_ladder_rather_than_jumping() -> None:
    """A multi-rung move is a sequence of events, one per rung (ADR 0010 D6)."""
    for fixture in (RISING, FALLING, SIDEWAYS, REVERSING):
        previous = TrendPhase.UNKNOWN
        for update in run(engine(), fixture):
            current = update.state.state
            steps = phase_steps(previous, current)
            if steps and TrendPhase.UNKNOWN not in (previous, current):
                assert [event.to_phase for event in update.events] == list(steps)
            previous = current


def test_a_reversal_passes_through_neutral() -> None:
    """CLAUDE.md §12 / specs/08-trend.md — no bull-to-bear without the middle."""
    path = visited(run(engine(), REVERSING))

    for before, after in itertools.pairwise(path):
        assert is_valid_transition(before, after)
        if before.direction.is_directional and after.direction.is_directional:
            assert before.direction is after.direction, (
                f"{before.value} flipped straight to {after.value}"
            )


# -- confirmation and hysteresis --------------------------------------------


def first_known(updates: Sequence[TrendUpdate]) -> int:
    return next(i for i, phase in enumerate(phases(updates)) if phase is not TrendPhase.UNKNOWN)


def test_confirmation_delays_the_move() -> None:
    """specs/08-trend.md — evidence must hold for N consecutive finalized bars."""
    prompt = run(engine(confirmation_bars=1), LONG_RISE)
    patient = run(engine(confirmation_bars=4), LONG_RISE)

    assert first_known(patient) > first_known(prompt)


def test_one_bar_of_noise_does_not_move_a_confirmed_state() -> None:
    """The Phase 5 exit criterion, stated as a test."""
    spike = bar_at(9, open_="100", high="130.5", low="99.5", close="130")
    calm = SIDEWAYS[:9]

    steady = engine(confirmation_bars=2)
    run(steady, calm)
    before = steady.state
    after_spike = feed(steady, spike).state
    after_next = feed(steady, bars_from(flat(100, 11))[10]).state

    assert before is not None
    assert after_spike.state is before.state is TrendPhase.NEUTRAL
    assert after_next.state is TrendPhase.NEUTRAL
    assert after_spike.strength > 0.0  # the score saw it; the state did not move


def test_without_confirmation_the_same_spike_does_move_the_state() -> None:
    """Confirmation is doing the work — not the thresholds on their own."""
    spike = bar_at(9, open_="100", high="130.5", low="99.5", close="130")

    twitchy = engine(confirmation_bars=1)
    run(twitchy, SIDEWAYS[:9])

    assert feed(twitchy, spike).state.direction is TrendDirection.BULLISH


def test_the_exit_margin_holds_a_phase_against_evidence_that_does_not_clear_it() -> None:
    """ADR 0010 D5 — the second half of hysteresis, at the engine level.

    A margin wide enough that no bias can clear the neutral band: confirmation
    alone would have moved this stream, and does at the default margin.
    """
    stubborn = run(engine(confirmation_bars=1, exit_margin=0.9), CALM_THEN_RISING)
    ordinary = run(engine(confirmation_bars=1), CALM_THEN_RISING)

    assert stubborn[-1].state.state is TrendPhase.NEUTRAL
    assert ordinary[-1].state.direction is TrendDirection.BULLISH


# -- missing evidence -------------------------------------------------------


def test_evidence_below_the_confidence_floor_is_unknown() -> None:
    """ADR 0010 D7 — a flat market never prints a swing, so structure is missing."""
    demanding = run(engine(minimum_confidence=1.0), SIDEWAYS)

    assert all(update.state.state is TrendPhase.UNKNOWN for update in demanding)
    assert all(update.state.confidence < 1.0 for update in demanding)


def test_the_same_market_is_readable_at_the_default_floor() -> None:
    tolerant = run(engine(), SIDEWAYS)

    assert tolerant[-1].state.state is TrendPhase.NEUTRAL
    assert 0.5 <= tolerant[-1].state.confidence < 1.0


def test_dropping_to_unknown_needs_no_confirmation_and_emits_nothing() -> None:
    updates = run(engine(minimum_confidence=1.0, confirmation_bars=5), SIDEWAYS)

    assert not any(update.events for update in updates)


def test_a_held_trend_falls_back_to_unknown_the_moment_evidence_thins() -> None:
    """ADR 0010 D7 — no confirmation wait, and no event: nothing happened to
    the market, something happened to the data.

    Warm indicators do not go cold on their own, so the floor is raised on a
    running engine to stand in for the input loss a real provider can deliver.
    """
    instance = engine(confirmation_bars=5)
    run(instance, RISING)
    held = instance.state
    assert held is not None
    assert held.direction is TrendDirection.BULLISH

    instance.config = dataclasses.replace(instance.config, minimum_confidence=1.0)
    update = feed(instance, bars_from(ramp(100, 15))[14])

    assert update.state.state is TrendPhase.UNKNOWN
    assert update.events == ()


# -- explanation ------------------------------------------------------------


def test_every_readable_state_explains_itself() -> None:
    """specs/08-trend.md — the score must be explainable, per component."""
    for update in run(engine(), RISING):
        state = update.state
        if state.state is TrendPhase.UNKNOWN:
            continue
        assert state.reason_codes
        assert state.strength_components is not None
        assert math.fsum(state.strength_components.values()) == pytest.approx(state.strength)


def test_the_breakdown_uses_the_component_names_the_spec_gives() -> None:
    final = run(engine(), RISING)[-1].state

    assert final.strength_components is not None
    assert set(final.strength_components) <= {member.value for member in StrengthComponent}


def test_a_reason_code_is_produced_for_every_readable_piece_of_evidence() -> None:
    update = run(engine(), RISING)[-1]
    readable = [reading for reading in update.evidence if reading.vote is not None]

    assert len(update.state.reason_codes) == len(readable)
    assert set(update.state.reason_codes) == {
        reading.reason_code for reading in readable if reading.reason_code is not None
    }


def test_every_bar_reports_one_reading_per_evidence_kind() -> None:
    for update in run(engine(), RISING):
        kinds = [reading.kind for reading in update.evidence]
        assert len(kinds) == len(set(kinds))


# -- events -----------------------------------------------------------------


def test_entering_a_trend_from_nothing_is_a_confirmation() -> None:
    events = [event for update in run(engine(), RISING) for event in update.events]

    assert events[0].kind is TrendEventKind.TREND_CONFIRMED
    assert events[0].from_phase is TrendPhase.UNKNOWN


def test_turning_around_is_a_reversal() -> None:
    events = [event for update in run(engine(), REVERSING) for event in update.events]
    kinds = [event.kind for event in events]

    assert TrendEventKind.TREND_REVERSAL in kinds
    reversal = events[kinds.index(TrendEventKind.TREND_REVERSAL)]
    assert reversal.to_phase.direction is TrendDirection.BEARISH


def test_a_stable_trend_does_not_emit_an_event_every_bar() -> None:
    """specs/08-trend.md, in as many words."""
    updates = run(engine(), RISING)
    settled = updates[len(updates) // 2 :]

    assert not any(update.events for update in settled)


def test_events_are_dated_to_the_bar_that_produced_them() -> None:
    for update in run(engine(), REVERSING):
        for event in update.events:
            assert event.bar_start_utc == update.state.as_of


def test_a_multi_rung_move_is_emitted_one_rung_at_a_time() -> None:
    """ADR 0010 D6 — the event stream stays ordered and each step is explainable."""
    for update in run(engine(), REVERSING):
        for before, after in zip(update.events, update.events[1:], strict=False):
            assert before.to_phase is after.from_phase
            assert is_valid_transition(before.to_phase, after.to_phase)


def test_the_last_event_of_a_bar_lands_on_that_bar_s_state() -> None:
    for update in run(engine(), REVERSING):
        if update.events:
            assert update.events[-1].to_phase is update.state.state


def test_events_accumulate_in_order() -> None:
    instance = engine()
    run(instance, REVERSING)
    stamps = [event.bar_start_utc for event in instance.events]

    assert stamps == sorted(stamps)


@pytest.mark.parametrize(
    ("current", "target", "expected"),
    [
        (TrendPhase.NEUTRAL, TrendPhase.NEUTRAL, ()),
        (TrendPhase.NEUTRAL, TrendPhase.WATCH_BULL, (TrendPhase.WATCH_BULL,)),
        (
            TrendPhase.NEUTRAL,
            TrendPhase.STRONG_BULLISH,
            (TrendPhase.WATCH_BULL, TrendPhase.BULLISH, TrendPhase.STRONG_BULLISH),
        ),
        (
            TrendPhase.WATCH_BULL,
            TrendPhase.WATCH_BEAR,
            (TrendPhase.NEUTRAL, TrendPhase.WATCH_BEAR),
        ),
        (TrendPhase.UNKNOWN, TrendPhase.STRONG_BEARISH, (TrendPhase.STRONG_BEARISH,)),
        (TrendPhase.BULLISH, TrendPhase.UNKNOWN, (TrendPhase.UNKNOWN,)),
    ],
)
def test_the_stepper_walks_the_ladder(
    current: TrendPhase, target: TrendPhase, expected: tuple[TrendPhase, ...]
) -> None:
    assert phase_steps(current, target) == expected


# -- no look-ahead, determinism ---------------------------------------------


def test_a_prefix_of_the_stream_produces_the_prefix_of_the_states() -> None:
    """CLAUDE.md §12 — nothing the engine says depends on a bar it has not seen."""
    whole = phases(run(engine(), REVERSING))

    for length in range(1, len(REVERSING) + 1):
        assert phases(run(engine(), REVERSING[:length])) == whole[:length]


def test_the_same_bars_produce_the_same_states_and_the_same_hashes() -> None:
    first = run(engine(), REVERSING)
    second = run(engine(), REVERSING)

    assert [update.state for update in first] == [update.state for update in second]
    assert [update.events for update in first] == [update.events for update in second]


def test_the_snapshot_hash_follows_the_evidence() -> None:
    rising = run(engine(), RISING)[-1].state
    falling = run(engine(), FALLING)[-1].state

    assert rising.input_snapshot_hash != falling.input_snapshot_hash
    assert rising.input_snapshot_hash


def test_the_snapshot_hash_follows_the_configuration() -> None:
    default = run(engine(), RISING)[-1].state
    retuned = run(engine(rsi_midpoint=45.0), RISING)[-1].state

    assert default.input_snapshot_hash != retuned.input_snapshot_hash


# -- stream discipline (inherited from BarConsumer) -------------------------


def test_a_repeated_bar_is_refused() -> None:
    instance = engine()
    run(instance, RISING[:5])

    with pytest.raises(InvariantViolation, match="strictly increasing"):
        instance.update(RISING[4])


def test_a_bar_from_another_series_is_refused() -> None:
    instance = engine()
    run(instance, RISING[:5])
    foreign = bar_at(9, open_="100", high="101", low="99", close="100", symbol=ticker("MSFT"))

    with pytest.raises(InvariantViolation, match="different series"):
        instance.update(foreign)


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
    instance = engine()
    run(instance, RISING[:5])
    foreign = bar_at(9, open_="100", high="101", low="99", close="100", **difference)  # type: ignore[arg-type]

    with pytest.raises(InvariantViolation, match="different series"):
        instance.update(foreign)


def test_a_partial_bar_is_refused_by_default() -> None:
    instance = engine()
    run(instance, RISING[:5])
    partial = bar_at(5, open_="105", high="106", low="104", close="105", is_final=False)

    with pytest.raises(InvariantViolation, match="partial"):
        instance.update(partial)


def test_intrabar_ticks_do_not_change_what_the_closed_bar_produces() -> None:
    quiet = engine()
    quiet_states = [update.state for update in run(quiet, RISING)]

    ticked = TrendEngine(
        config=FAST,
        structure=StructureEngine(left_bars=1, right_bars=1),
        allow_partial=True,
    )
    ticked_states = []
    for index, bar in enumerate(RISING):
        for tick in ("1", "999"):
            ticked.update(
                bar_at(index, open_=tick, high=tick, low=tick, close=tick, is_final=False)
            )
        ticked_states.append(feed(ticked, bar).state)

    assert ticked_states == quiet_states


def test_a_provisional_reading_leaves_the_engine_untouched() -> None:
    instance = TrendEngine(
        config=FAST,
        structure=StructureEngine(left_bars=1, right_bars=1),
        allow_partial=True,
    )
    run(instance, RISING[:8])
    before = (instance.state, instance.events, instance.committed_bars)

    instance.update(bar_at(8, open_="1", high="1", low="1", close="1", is_final=False))

    assert (instance.state, instance.events, instance.committed_bars) == before


# -- identity and construction ----------------------------------------------


def test_the_spec_key_records_the_whole_configuration() -> None:
    instance = TrendEngine()

    assert instance.spec.key == (
        "trend(fast=20,slow=50,slope=5,slope_atr=0.0,atr=14,rsi=14,rsi_mid=50.0,"
        "rsi_band=0.0,macd=12/26/9,src=CLOSE,thresholds=0.2/0.5/0.8,margin=0.05,"
        "confirm=2,min_conf=0.5,weights=1.0/1.0/1.0/1.0/1.0/1.0/0.0,"
        "structure=structure(left=3,right=3,min_move_atr=0.0,atr=14,equal=STRICT,"
        "closes=1,wicks=False))"
    )


def test_the_identity_covers_the_composed_structure_engine() -> None:
    """Structure feeds the trend, so a re-tuned structure engine is a different trend."""
    wide = TrendEngine(structure=StructureEngine(left_bars=5, right_bars=5))

    assert wide.spec.key != TrendEngine().spec.key


def test_a_retuned_engine_has_a_different_identity() -> None:
    assert TrendEngine().spec.key != TrendEngine(config=TrendConfig(rsi_period=21)).spec.key


def test_warmup_is_the_bar_at_which_the_confidence_floor_can_first_be_met() -> None:
    """ADR 0010 D11 — not the slowest input, which would over-report."""
    instance = engine()
    updates = run(instance, RISING)

    assert instance.warmup == first_known(updates) + 1


def test_demanding_full_confidence_pushes_warmup_out_to_the_slowest_input() -> None:
    assert engine(minimum_confidence=1.0).warmup > engine().warmup


def test_the_composed_structure_engine_is_readable() -> None:
    """A UI draws the swings the trend was reasoned from, not a second set."""
    instance = engine()
    run(instance, REVERSING)

    assert instance.structure.swings
    assert instance.structure.committed_bars == len(REVERSING)


def test_a_structure_engine_that_has_already_run_is_refused() -> None:
    used = StructureEngine(left_bars=1, right_bars=1)
    used.update(RISING[0])

    with pytest.raises(InvariantViolation, match="fresh"):
        TrendEngine(config=FAST, structure=used)
