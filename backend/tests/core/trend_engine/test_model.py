"""Configuration, thresholds and events — ADR 0010 D4/D5/D10.

The threshold band tests are where the hysteresis rule is actually pinned: the
engine tests show it survives a bar stream, but the arithmetic of "how far past
the boundary must bias go before the phase moves" is decided here.
"""

from __future__ import annotations

import dataclasses

import pytest

from alphagate.core.errors import InvariantViolation
from alphagate.core.time_model import Timeframe
from alphagate.core.trend import TrendPhase
from alphagate.core.trend_engine import (
    EvidenceKind,
    EvidenceReading,
    PhaseThresholds,
    StrengthComponent,
    TrendConfig,
    TrendEvent,
    TrendEventKind,
    TrendWeights,
)
from tests.core.trend_engine.synthetic import AAPL, SESSION_OPEN

DEFAULTS = PhaseThresholds()


# -- thresholds -------------------------------------------------------------


@pytest.mark.parametrize(
    ("bias", "expected"),
    [
        (1.0, TrendPhase.STRONG_BULLISH),
        (0.9, TrendPhase.STRONG_BULLISH),
        (0.7, TrendPhase.BULLISH),
        (0.3, TrendPhase.WATCH_BULL),
        (0.0, TrendPhase.NEUTRAL),
        (-0.3, TrendPhase.WATCH_BEAR),
        (-0.7, TrendPhase.BEARISH),
        (-1.0, TrendPhase.STRONG_BEARISH),
    ],
)
def test_bias_maps_onto_the_ladder(bias: float, expected: TrendPhase) -> None:
    assert DEFAULTS.phase_for(bias) is expected


@pytest.mark.parametrize(
    ("bias", "expected"),
    [
        (0.8, TrendPhase.STRONG_BULLISH),
        (0.5, TrendPhase.BULLISH),
        (0.2, TrendPhase.WATCH_BULL),
        (-0.2, TrendPhase.WATCH_BEAR),
        (-0.5, TrendPhase.BEARISH),
        (-0.8, TrendPhase.STRONG_BEARISH),
    ],
)
def test_a_boundary_belongs_to_the_stronger_phase(bias: float, expected: TrendPhase) -> None:
    """ADR 0010 D4 — "≥ threshold" reads the way the spec's prose does."""
    assert DEFAULTS.phase_for(bias) is expected


def test_the_mapping_is_symmetric() -> None:
    ladder = TrendPhase.ladder()
    for tenth in range(-10, 11):
        bias = tenth / 10
        here = ladder.index(DEFAULTS.phase_for(bias))
        mirrored = ladder.index(DEFAULTS.phase_for(-bias))
        assert here + mirrored == len(ladder) - 1


def test_every_directional_phase_has_a_band_and_unknown_does_not() -> None:
    for phase in TrendPhase.ladder():
        low, high = DEFAULTS.band_of(phase)
        assert low < high

    with pytest.raises(InvariantViolation, match="UNKNOWN"):
        DEFAULTS.band_of(TrendPhase.UNKNOWN)


def test_thresholds_must_be_ordered() -> None:
    with pytest.raises(InvariantViolation, match="ordered"):
        PhaseThresholds(watch=0.6, trend=0.5, strong=0.8)


def test_thresholds_must_stay_inside_the_bias_range() -> None:
    with pytest.raises(InvariantViolation, match=r"\(0, 1\]"):
        PhaseThresholds(watch=0.2, trend=0.5, strong=1.5)


# -- hysteresis -------------------------------------------------------------


def test_leaving_a_phase_requires_clearing_its_band_by_the_margin() -> None:
    """ADR 0010 D5 — bias sitting on a boundary must not flip-flop."""
    held = TrendPhase.WATCH_BULL

    assert DEFAULTS.target_from(0.50, current=held, exit_margin=0.05) is held
    assert DEFAULTS.target_from(0.54, current=held, exit_margin=0.05) is held
    assert DEFAULTS.target_from(0.55, current=held, exit_margin=0.05) is TrendPhase.BULLISH


def test_the_margin_applies_to_the_downside_of_the_band_too() -> None:
    held = TrendPhase.BULLISH

    assert DEFAULTS.target_from(0.46, current=held, exit_margin=0.05) is held
    assert DEFAULTS.target_from(0.45, current=held, exit_margin=0.05) is TrendPhase.WATCH_BULL


def test_re_entering_the_band_needs_no_margin_at_all() -> None:
    """The margin defends the phase you hold; it is not a toll on returning."""
    assert DEFAULTS.target_from(0.30, current=TrendPhase.BULLISH, exit_margin=0.05) is (
        TrendPhase.WATCH_BULL
    )
    assert DEFAULTS.target_from(0.30, current=TrendPhase.WATCH_BULL, exit_margin=0.05) is (
        TrendPhase.WATCH_BULL
    )


def test_unknown_holds_no_band_so_any_bias_moves_off_it() -> None:
    assert DEFAULTS.target_from(0.0, current=TrendPhase.UNKNOWN, exit_margin=0.5) is (
        TrendPhase.NEUTRAL
    )


def test_a_margin_of_zero_is_the_plain_mapping() -> None:
    for tenth in range(-10, 11):
        bias = tenth / 10
        for phase in TrendPhase.ladder():
            assert DEFAULTS.target_from(bias, current=phase, exit_margin=0.0) is (
                DEFAULTS.phase_for(bias)
            )


# -- weights ----------------------------------------------------------------


def test_volume_is_declared_at_zero_weight() -> None:
    """ADR 0010 D2 — in the table, so enabling it is configuration."""
    weights = TrendWeights()

    assert weights.of(EvidenceKind.VOLUME) == 0.0
    assert EvidenceKind.VOLUME not in weights.requested()


def test_every_evidence_kind_has_a_weight_and_a_component() -> None:
    weights = TrendWeights()

    for kind in EvidenceKind:
        assert weights.of(kind) >= 0.0
        assert isinstance(kind.component, StrengthComponent)


def test_weighting_volume_is_refused_while_it_cannot_be_computed() -> None:
    """A weight on evidence that never votes would only depress confidence."""
    with pytest.raises(InvariantViolation, match="volume"):
        TrendWeights(volume=1.0)


def test_a_negative_weight_is_refused() -> None:
    with pytest.raises(InvariantViolation, match="negative"):
        TrendWeights(rsi=-1.0)


def test_all_weights_zero_is_refused() -> None:
    with pytest.raises(InvariantViolation, match="positive"):
        TrendWeights(
            ema_alignment=0.0,
            ema_slope=0.0,
            price_location=0.0,
            structure=0.0,
            rsi=0.0,
            macd_histogram=0.0,
        )


# -- config -----------------------------------------------------------------


def test_the_fast_average_must_be_faster_than_the_slow_one() -> None:
    with pytest.raises(InvariantViolation, match="slow_period"):
        TrendConfig(fast_period=50, slow_period=20)


def test_the_macd_pair_must_be_ordered_too() -> None:
    with pytest.raises(InvariantViolation, match="macd_slow"):
        TrendConfig(macd_fast=26, macd_slow=12)


@pytest.mark.parametrize(
    ("override", "expected"),
    [
        ({"slope_minimum_atr": -0.5}, "slope_minimum_atr"),
        ({"rsi_midpoint": 0.0}, "rsi_midpoint"),
        ({"rsi_midpoint": 100.0}, "rsi_midpoint"),
        ({"rsi_band": -1.0}, "rsi_band"),
        ({"rsi_midpoint": 95.0, "rsi_band": 10.0}, "bullish side"),
    ],
    ids=["slope", "midpoint-low", "midpoint-high", "band-negative", "band-swallows-bull"],
)
def test_the_evidence_thresholds_must_stay_inside_their_own_scales(
    override: dict[str, float], expected: str
) -> None:
    with pytest.raises(InvariantViolation, match=expected):
        TrendConfig(**override)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "override",
    [
        {"confirmation_bars": 0},
        {"slope_lookback": 0},
        {"rsi_period": 0},
        {"atr_period": 0},
    ],
    ids=["confirmation", "slope", "rsi", "atr"],
)
def test_periods_must_be_at_least_one_bar(override: dict[str, int]) -> None:
    with pytest.raises(InvariantViolation, match="at least 1 bar"):
        TrendConfig(**override)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_minimum_confidence_is_a_fraction(value: float) -> None:
    with pytest.raises(InvariantViolation, match="minimum_confidence"):
        TrendConfig(minimum_confidence=value)


def test_the_exit_margin_may_not_be_negative() -> None:
    with pytest.raises(InvariantViolation, match="exit_margin"):
        TrendConfig(exit_margin=-0.01)


def test_the_rsi_band_may_not_swallow_the_whole_scale() -> None:
    with pytest.raises(InvariantViolation, match="rsi_band"):
        TrendConfig(rsi_band=60.0)


# -- readings ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("vote", "expected"),
    [(1.0, "RSI_BULL"), (-1.0, "RSI_BEAR"), (0.0, "RSI_FLAT"), (None, None)],
)
def test_a_reading_names_itself(vote: float | None, expected: str | None) -> None:
    reading = EvidenceReading(kind=EvidenceKind.RSI, vote=vote)

    assert reading.reason_code == expected


def test_a_vote_outside_the_range_is_refused() -> None:
    with pytest.raises(InvariantViolation, match=r"\[-1, 1\]"):
        EvidenceReading(kind=EvidenceKind.RSI, vote=1.5)


# -- events -----------------------------------------------------------------


def make_event(**overrides: object) -> TrendEvent:
    defaults: dict[str, object] = {
        "kind": TrendEventKind.TREND_CONFIRMED,
        "symbol": AAPL,
        "timeframe": Timeframe.M1,
        "from_phase": TrendPhase.NEUTRAL,
        "to_phase": TrendPhase.WATCH_BULL,
        "bar_start_utc": SESSION_OPEN,
        "strength": 42.0,
        "reason_codes": ("RSI_BULL",),
        "input_snapshot_hash": "abc123",
    }
    return TrendEvent(**{**defaults, **overrides})  # type: ignore[arg-type]


def test_an_event_records_both_ends_of_the_move() -> None:
    event = make_event()

    assert event.from_phase is TrendPhase.NEUTRAL
    assert event.to_phase is TrendPhase.WATCH_BULL


def test_an_event_that_changes_nothing_is_refused() -> None:
    with pytest.raises(InvariantViolation, match="change"):
        make_event(to_phase=TrendPhase.NEUTRAL)


def test_an_event_may_not_skip_a_rung() -> None:
    with pytest.raises(InvariantViolation, match="one rung"):
        make_event(from_phase=TrendPhase.NEUTRAL, to_phase=TrendPhase.STRONG_BULLISH)


def test_an_event_carries_a_score_on_the_same_scale_as_the_state() -> None:
    with pytest.raises(InvariantViolation, match=r"\[0, 100\]"):
        make_event(strength=101.0)


def test_an_event_must_carry_its_evidence() -> None:
    with pytest.raises(InvariantViolation, match="input_snapshot_hash"):
        make_event(input_snapshot_hash="")


def test_an_event_is_frozen() -> None:
    event = make_event()

    with pytest.raises(dataclasses.FrozenInstanceError):
        event.strength = 1.0  # type: ignore[misc]
