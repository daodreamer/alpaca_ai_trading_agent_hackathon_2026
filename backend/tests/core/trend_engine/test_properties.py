"""Trend invariants over generated evidence and price paths — specs/14-testing.md.

Hand-written fixtures show the engine doing the right thing on a shape someone
imagined. These check the things that must hold on every shape: the score stays
in range and stays decomposable, the ladder is never skipped, and no event is
ever emitted without a state change behind it.
"""

from __future__ import annotations

import math
from decimal import Decimal

from hypothesis import given
from hypothesis import strategies as st

from alphagate.core.bar import Bar
from alphagate.core.structure import StructureEngine
from alphagate.core.trend import TrendPhase, is_valid_transition
from alphagate.core.trend_engine import (
    EvidenceKind,
    EvidenceReading,
    PhaseThresholds,
    TrendEngine,
    TrendWeights,
    phase_steps,
    score_evidence,
)
from tests.core.trend_engine.synthetic import FAST, bar_at

votes = st.sampled_from([None, -1.0, -0.5, 0.0, 0.5, 1.0])
biases = st.floats(min_value=-1.0, max_value=1.0)
phases = st.sampled_from(list(TrendPhase))
paths = st.lists(
    st.decimals(min_value=Decimal("10"), max_value=Decimal("200"), places=2),
    min_size=1,
    max_size=40,
)

DEFAULTS = PhaseThresholds()


def readings(values: list[float | None]) -> tuple[EvidenceReading, ...]:
    return tuple(
        EvidenceReading(kind=kind, vote=value)
        for kind, value in zip(EvidenceKind, values, strict=True)
    )


def walk(path: list[Decimal]) -> tuple[Bar, ...]:
    return tuple(
        bar_at(
            index,
            open_=str(centre),
            high=str(centre + 1),
            low=str(centre - 1),
            close=str(centre),
        )
        for index, centre in enumerate(path)
    )


# -- scoring ----------------------------------------------------------------


@given(st.lists(votes, min_size=len(EvidenceKind), max_size=len(EvidenceKind)))
def test_the_score_is_always_reportable(values: list[float | None]) -> None:
    score = score_evidence(readings(values), weights=TrendWeights())

    assert 0.0 <= score.strength <= 100.0
    assert 0.0 <= score.confidence <= 1.0
    assert score.bias is None or -1.0 <= score.bias <= 1.0


@given(st.lists(votes, min_size=len(EvidenceKind), max_size=len(EvidenceKind)))
def test_the_breakdown_always_adds_up(values: list[float | None]) -> None:
    """The invariant `TrendState` enforces, checked before it gets there."""
    score = score_evidence(readings(values), weights=TrendWeights())

    assert math.isclose(math.fsum(score.components.values()), score.strength, abs_tol=1e-9)


@given(st.lists(votes, min_size=len(EvidenceKind), max_size=len(EvidenceKind)))
def test_mirroring_the_evidence_mirrors_the_bias(values: list[float | None]) -> None:
    score = score_evidence(readings(values), weights=TrendWeights())
    mirrored = score_evidence(
        readings([None if value is None else -value for value in values]),
        weights=TrendWeights(),
    )

    if score.bias is None:
        assert mirrored.bias is None
    else:
        assert mirrored.bias is not None
        assert math.isclose(mirrored.bias, -score.bias, abs_tol=1e-12)
        assert math.isclose(mirrored.strength, score.strength, abs_tol=1e-9)


# -- thresholds and stepping ------------------------------------------------


@given(biases, phases, st.floats(min_value=0.0, max_value=1.0))
def test_a_target_is_either_the_phase_held_or_the_phase_the_bias_names(
    bias: float, current: TrendPhase, margin: float
) -> None:
    target = DEFAULTS.target_from(bias, current=current, exit_margin=margin)

    assert target in (current, DEFAULTS.phase_for(bias))


@given(phases, phases)
def test_the_stepper_produces_a_chain_of_legal_single_moves(
    current: TrendPhase, target: TrendPhase
) -> None:
    steps = phase_steps(current, target)

    if current is target:
        assert steps == ()
        return

    assert steps[-1] is target
    previous = current
    for step in steps:
        assert is_valid_transition(previous, step)
        previous = step


# -- the engine over a random walk ------------------------------------------


@given(paths)
def test_the_engine_never_skips_a_rung(path: list[Decimal]) -> None:
    """A bar may move several rungs, but only as a sequence of single steps."""
    engine = TrendEngine(config=FAST, structure=StructureEngine(left_bars=1, right_bars=1))
    previous = TrendPhase.UNKNOWN

    for bar in walk(path):
        update = engine.update(bar)
        assert update is not None
        current = update.state.state
        steps = phase_steps(previous, current)
        if steps and TrendPhase.UNKNOWN not in (previous, current):
            assert [event.to_phase for event in update.events] == list(steps)
        for step in steps:
            assert is_valid_transition(previous, step)
            previous = step
        previous = current


@given(paths)
def test_an_event_is_never_emitted_without_a_state_change(path: list[Decimal]) -> None:
    engine = TrendEngine(config=FAST, structure=StructureEngine(left_bars=1, right_bars=1))
    previous = TrendPhase.UNKNOWN

    for bar in walk(path):
        update = engine.update(bar)
        assert update is not None
        if update.events:
            assert update.state.state is not previous
            assert update.events[0].from_phase is previous
        previous = update.state.state


@given(paths)
def test_a_state_is_produced_for_every_bar_and_never_moves_backwards(
    path: list[Decimal],
) -> None:
    engine = TrendEngine(config=FAST, structure=StructureEngine(left_bars=1, right_bars=1))
    stamps = []

    for bar in walk(path):
        update = engine.update(bar)
        assert update is not None
        stamps.append(update.state.as_of)

    assert stamps == sorted(stamps)
    assert len(stamps) == len(path)
