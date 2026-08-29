"""TrendState and the transition ladder — specs/08-trend.md."""

from __future__ import annotations

import dataclasses
import itertools
from datetime import UTC, datetime

import pytest

from alphagate.core.errors import InvalidTransition, InvariantViolation
from alphagate.core.identifiers import ticker
from alphagate.core.time_model import Timeframe
from alphagate.core.trend import (
    TrendDirection,
    TrendPhase,
    TrendState,
    is_valid_transition,
)

AAPL = ticker("AAPL")
NOW = datetime(2026, 8, 13, 14, 0, tzinfo=UTC)


def make_state(**overrides: object) -> TrendState:
    defaults: dict[str, object] = {
        "symbol": AAPL,
        "timeframe": Timeframe.H1,
        "state": TrendPhase.NEUTRAL,
        "strength": 50.0,
        "confidence": 0.5,
        "as_of": NOW,
        "confirmed_at": None,
        "reason_codes": ("EMA_ALIGNED",),
        "input_snapshot_hash": "0" * 64,
    }
    return TrendState(**{**defaults, **overrides})  # type: ignore[arg-type]


class TestPhaseLadder:
    def test_ladder_is_ordered_bear_to_bull(self) -> None:
        assert [p.name for p in TrendPhase.ladder()] == [
            "STRONG_BEARISH",
            "BEARISH",
            "WATCH_BEAR",
            "NEUTRAL",
            "WATCH_BULL",
            "BULLISH",
            "STRONG_BULLISH",
        ]

    def test_unknown_is_not_on_the_ladder(self) -> None:
        assert TrendPhase.UNKNOWN not in TrendPhase.ladder()

    @pytest.mark.parametrize(
        ("phase", "direction"),
        [
            (TrendPhase.STRONG_BEARISH, TrendDirection.BEARISH),
            (TrendPhase.BEARISH, TrendDirection.BEARISH),
            (TrendPhase.WATCH_BEAR, TrendDirection.BEARISH),
            (TrendPhase.NEUTRAL, TrendDirection.NEUTRAL),
            (TrendPhase.WATCH_BULL, TrendDirection.BULLISH),
            (TrendPhase.BULLISH, TrendDirection.BULLISH),
            (TrendPhase.STRONG_BULLISH, TrendDirection.BULLISH),
            (TrendPhase.UNKNOWN, TrendDirection.UNKNOWN),
        ],
    )
    def test_direction_is_derived_from_the_phase(
        self, phase: TrendPhase, direction: TrendDirection
    ) -> None:
        assert phase.direction is direction


class TestTransitionRules:
    def test_single_step_moves_are_allowed(self) -> None:
        assert is_valid_transition(TrendPhase.NEUTRAL, TrendPhase.WATCH_BULL)
        assert is_valid_transition(TrendPhase.WATCH_BULL, TrendPhase.BULLISH)
        assert is_valid_transition(TrendPhase.BULLISH, TrendPhase.WATCH_BULL)

    def test_staying_put_is_allowed(self) -> None:
        for phase in TrendPhase:
            assert is_valid_transition(phase, phase)

    def test_skipping_a_rung_is_rejected(self) -> None:
        assert not is_valid_transition(TrendPhase.NEUTRAL, TrendPhase.BULLISH)
        assert not is_valid_transition(TrendPhase.NEUTRAL, TrendPhase.STRONG_BULLISH)

    def test_crossing_sides_without_passing_through_neutral_is_rejected(self) -> None:
        # This is the hysteresis requirement (specs/08-trend.md) expressed structurally:
        # a one-tick crossover cannot flip bull to bear.
        assert not is_valid_transition(TrendPhase.BULLISH, TrendPhase.BEARISH)
        assert not is_valid_transition(TrendPhase.WATCH_BULL, TrendPhase.WATCH_BEAR)
        assert not is_valid_transition(TrendPhase.STRONG_BULLISH, TrendPhase.STRONG_BEARISH)

    def test_watch_states_reach_neutral_in_one_step(self) -> None:
        assert is_valid_transition(TrendPhase.WATCH_BULL, TrendPhase.NEUTRAL)
        assert is_valid_transition(TrendPhase.WATCH_BEAR, TrendPhase.NEUTRAL)

    def test_unknown_may_cold_start_into_any_phase(self) -> None:
        for phase in TrendPhase:
            assert is_valid_transition(TrendPhase.UNKNOWN, phase)

    def test_any_phase_may_fall_back_to_unknown(self) -> None:
        # Indicator data can go missing mid-stream (specs/08-trend.md testing list).
        for phase in TrendPhase:
            assert is_valid_transition(phase, TrendPhase.UNKNOWN)

    def test_every_ordered_pair_is_decided(self) -> None:
        for a, b in itertools.product(TrendPhase, repeat=2):
            assert isinstance(is_valid_transition(a, b), bool)

    def test_ladder_transitions_are_symmetric(self) -> None:
        for a, b in itertools.product(TrendPhase.ladder(), repeat=2):
            assert is_valid_transition(a, b) == is_valid_transition(b, a)


class TestTrendState:
    def test_direction_tracks_the_phase(self) -> None:
        assert make_state(state=TrendPhase.BULLISH).direction is TrendDirection.BULLISH

    @pytest.mark.parametrize("strength", [-0.1, 100.1])
    def test_strength_is_bounded(self, strength: float) -> None:
        with pytest.raises(InvariantViolation, match="strength"):
            make_state(strength=strength)

    @pytest.mark.parametrize("confidence", [-0.1, 1.1])
    def test_confidence_is_bounded(self, confidence: float) -> None:
        with pytest.raises(InvariantViolation, match="confidence"):
            make_state(confidence=confidence)

    def test_reason_codes_are_required(self) -> None:
        # Every state must be explainable (specs/00-vision.md principle 2).
        with pytest.raises(InvariantViolation, match="reason_codes"):
            make_state(reason_codes=())

    def test_unknown_phase_needs_no_reason_codes(self) -> None:
        assert make_state(state=TrendPhase.UNKNOWN, reason_codes=()).reason_codes == ()

    def test_snapshot_hash_is_required_for_determinism_audits(self) -> None:
        with pytest.raises(InvariantViolation, match="input_snapshot_hash"):
            make_state(input_snapshot_hash="")

    def test_is_frozen(self) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            make_state().strength = 1.0  # type: ignore[misc]

    def test_strength_components_must_sum_to_the_strength(self) -> None:
        with pytest.raises(InvariantViolation, match="components"):
            make_state(
                strength=50.0,
                strength_components={"ema_alignment": 25.0, "structure": 10.0},
            )

    def test_strength_components_are_accepted_when_consistent(self) -> None:
        state = make_state(
            strength=50.0,
            strength_components={"ema_alignment": 25.0, "structure": 25.0},
        )
        assert state.strength_components == {"ema_alignment": 25.0, "structure": 25.0}


class TestTransitionTo:
    def test_legal_transition_returns_a_new_state(self) -> None:
        state = make_state(state=TrendPhase.NEUTRAL)
        moved = state.transition_to(
            TrendPhase.WATCH_BULL,
            as_of=NOW,
            strength=55.0,
            confidence=0.6,
            reason_codes=("EMA20_ABOVE_EMA50",),
            input_snapshot_hash="1" * 64,
        )
        assert moved.state is TrendPhase.WATCH_BULL
        assert state.state is TrendPhase.NEUTRAL  # original untouched

    def test_illegal_transition_raises(self) -> None:
        with pytest.raises(InvalidTransition, match="BULLISH"):
            make_state(state=TrendPhase.BEARISH).transition_to(
                TrendPhase.BULLISH,
                as_of=NOW,
                strength=70.0,
                confidence=0.7,
                reason_codes=("X",),
                input_snapshot_hash="1" * 64,
            )

    def test_time_must_not_run_backwards(self) -> None:
        with pytest.raises(InvariantViolation, match="as_of"):
            make_state().transition_to(
                TrendPhase.WATCH_BULL,
                as_of=NOW.replace(hour=13),
                strength=55.0,
                confidence=0.6,
                reason_codes=("X",),
                input_snapshot_hash="1" * 64,
            )

    def test_entering_a_directional_phase_stamps_confirmed_at(self) -> None:
        moved = make_state(state=TrendPhase.WATCH_BULL).transition_to(
            TrendPhase.BULLISH,
            as_of=NOW,
            strength=70.0,
            confidence=0.7,
            reason_codes=("STRUCTURE_HH_HL",),
            input_snapshot_hash="1" * 64,
        )
        assert moved.confirmed_at == NOW

    def test_staying_in_the_same_phase_keeps_the_original_confirmation_time(self) -> None:
        earlier = make_state(state=TrendPhase.BULLISH, confirmed_at=NOW)
        later = earlier.transition_to(
            TrendPhase.BULLISH,
            as_of=NOW.replace(hour=15),
            strength=72.0,
            confidence=0.75,
            reason_codes=("STRUCTURE_HH_HL",),
            input_snapshot_hash="2" * 64,
        )
        assert later.confirmed_at == NOW

    def test_returning_to_neutral_clears_confirmation(self) -> None:
        confirmed = make_state(state=TrendPhase.WATCH_BULL, confirmed_at=NOW)
        neutral = confirmed.transition_to(
            TrendPhase.NEUTRAL,
            as_of=NOW.replace(hour=15),
            strength=50.0,
            confidence=0.5,
            reason_codes=("EVIDENCE_LOST",),
            input_snapshot_hash="3" * 64,
        )
        assert neutral.confirmed_at is None
