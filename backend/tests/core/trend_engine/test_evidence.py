"""Evidence reading and scoring — ADR 0010 D2/D3/D8/D9.

Pure functions over a snapshot of indicator readings, so every case the spec
lists — missing values, exactly-on-boundary values, disagreement — is expressible
without building a bar stream.
"""

from __future__ import annotations

import math

import pytest

from alphagate.core.structure import StructureLabel
from alphagate.core.trend_engine import (
    EvidenceInputs,
    EvidenceKind,
    EvidenceReading,
    StrengthComponent,
    TrendConfig,
    TrendWeights,
    read_evidence,
    score_evidence,
)

BASE = EvidenceInputs(
    close=100.0,
    fast_ema=100.0,
    slow_ema=100.0,
    slow_ema_before=100.0,
    atr=1.0,
    rsi=50.0,
    macd_histogram=0.0,
    swing_high_label=None,
    swing_low_label=None,
)


def vars_of(inputs: EvidenceInputs) -> dict[str, object]:
    return {field: getattr(inputs, field) for field in inputs.__slots__}


def vote_for(
    kind: EvidenceKind, *, config: TrendConfig | None = None, **overrides: object
) -> float | None:
    inputs = EvidenceInputs(**{**vars_of(BASE), **overrides})  # type: ignore[arg-type]
    read = read_evidence(inputs, config=config if config is not None else TrendConfig())
    return next(reading.vote for reading in read if reading.kind is kind)


# -- one reading per kind, every bar ----------------------------------------


def test_every_kind_is_reported_even_when_it_cannot_be_read() -> None:
    readings = read_evidence(BASE, config=TrendConfig())

    assert tuple(reading.kind for reading in readings) == tuple(EvidenceKind)


def test_volume_is_declared_but_never_computed_in_the_mvp() -> None:
    assert vote_for(EvidenceKind.VOLUME) is None


# -- EMA alignment and price location ---------------------------------------


@pytest.mark.parametrize(
    ("fast", "slow", "expected"),
    [(101.0, 100.0, 1.0), (99.0, 100.0, -1.0), (100.0, 100.0, 0.0)],
)
def test_ema_alignment_reads_the_pair(fast: float, slow: float, expected: float) -> None:
    assert vote_for(EvidenceKind.EMA_ALIGNMENT, fast_ema=fast, slow_ema=slow) == expected


@pytest.mark.parametrize("missing", ["fast_ema", "slow_ema"])
def test_ema_alignment_is_unreadable_while_either_average_warms(missing: str) -> None:
    assert vote_for(EvidenceKind.EMA_ALIGNMENT, **{missing: None}) is None


@pytest.mark.parametrize(("close", "expected"), [(101.0, 1.0), (99.0, -1.0), (100.0, 0.0)])
def test_price_location_reads_price_against_the_fast_average(close: float, expected: float) -> None:
    assert vote_for(EvidenceKind.PRICE_LOCATION, close=close) == expected


def test_price_location_is_unreadable_while_the_fast_average_warms() -> None:
    assert vote_for(EvidenceKind.PRICE_LOCATION, fast_ema=None) is None


# -- slope ------------------------------------------------------------------


@pytest.mark.parametrize(("before", "expected"), [(99.0, 1.0), (101.0, -1.0), (100.0, 0.0)])
def test_slope_is_the_slow_average_against_its_own_past(before: float, expected: float) -> None:
    assert vote_for(EvidenceKind.EMA_SLOPE, slow_ema_before=before) == expected


def test_slope_is_unreadable_before_the_lookback_has_filled() -> None:
    assert vote_for(EvidenceKind.EMA_SLOPE, slow_ema_before=None) is None


def test_a_slope_inside_the_atr_deadband_is_flat_not_bullish() -> None:
    config = TrendConfig(slope_minimum_atr=0.5)

    assert vote_for(EvidenceKind.EMA_SLOPE, config=config, slow_ema_before=99.6) == 0.0
    assert vote_for(EvidenceKind.EMA_SLOPE, config=config, slow_ema_before=99.4) == 1.0


def test_a_deadband_that_needs_atr_is_unreadable_without_it() -> None:
    """ADR 0010 D2 — unmeasurable is not the same as flat."""
    config = TrendConfig(slope_minimum_atr=0.5)

    assert vote_for(EvidenceKind.EMA_SLOPE, config=config, atr=None, slow_ema_before=99.0) is None
    assert vote_for(EvidenceKind.EMA_SLOPE, atr=None, slow_ema_before=99.0) == 1.0


# -- structure --------------------------------------------------------------


@pytest.mark.parametrize(
    ("high", "low", "expected"),
    [
        (StructureLabel.HH, StructureLabel.HL, 1.0),
        (StructureLabel.LH, StructureLabel.LL, -1.0),
        (StructureLabel.HH, StructureLabel.LL, 0.0),
        (StructureLabel.LH, StructureLabel.HL, 0.0),
    ],
)
def test_structure_needs_both_sides_to_agree(
    high: StructureLabel, low: StructureLabel, expected: float
) -> None:
    assert vote_for(EvidenceKind.STRUCTURE, swing_high_label=high, swing_low_label=low) == expected


@pytest.mark.parametrize("present", ["swing_high_label", "swing_low_label"])
def test_structure_is_unreadable_with_only_one_label(present: str) -> None:
    assert vote_for(EvidenceKind.STRUCTURE, **{present: StructureLabel.HH}) is None


# -- momentum ---------------------------------------------------------------


@pytest.mark.parametrize(("rsi", "expected"), [(60.0, 1.0), (40.0, -1.0), (50.0, 0.0)])
def test_rsi_reads_against_the_midpoint(rsi: float, expected: float) -> None:
    assert vote_for(EvidenceKind.RSI, rsi=rsi) == expected


def test_the_rsi_band_is_exclusive_at_its_own_edge() -> None:
    config = TrendConfig(rsi_band=10.0)

    assert vote_for(EvidenceKind.RSI, config=config, rsi=60.0) == 0.0
    assert vote_for(EvidenceKind.RSI, config=config, rsi=60.5) == 1.0
    assert vote_for(EvidenceKind.RSI, config=config, rsi=40.0) == 0.0
    assert vote_for(EvidenceKind.RSI, config=config, rsi=39.5) == -1.0


def test_rsi_is_unreadable_while_it_warms() -> None:
    assert vote_for(EvidenceKind.RSI, rsi=None) is None


@pytest.mark.parametrize(
    ("histogram", "expected"), [(0.5, 1.0), (-0.5, -1.0), (0.0, 0.0), (None, None)]
)
def test_the_macd_histogram_votes_by_sign(histogram: float | None, expected: float | None) -> None:
    assert vote_for(EvidenceKind.MACD_HISTOGRAM, macd_histogram=histogram) == expected


# -- scoring ----------------------------------------------------------------


def readings(**votes: float | None) -> tuple[EvidenceReading, ...]:
    """Readings for every kind; anything unnamed is unavailable."""
    return tuple(
        EvidenceReading(kind=kind, vote=votes.get(kind.value.lower())) for kind in EvidenceKind
    )


ALL_BULLISH = readings(
    ema_alignment=1.0,
    ema_slope=1.0,
    price_location=1.0,
    structure=1.0,
    rsi=1.0,
    macd_histogram=1.0,
)


def test_unanimous_evidence_is_full_bias_and_full_strength() -> None:
    score = score_evidence(ALL_BULLISH, weights=TrendWeights())

    assert score.bias == pytest.approx(1.0)
    assert score.strength == pytest.approx(100.0)
    assert score.confidence == pytest.approx(1.0)


def test_the_mirror_case_is_the_mirror_score() -> None:
    bearish = readings(
        ema_alignment=-1.0,
        ema_slope=-1.0,
        price_location=-1.0,
        structure=-1.0,
        rsi=-1.0,
        macd_histogram=-1.0,
    )
    score = score_evidence(bearish, weights=TrendWeights())

    assert score.bias == pytest.approx(-1.0)
    assert score.strength == pytest.approx(100.0)


def test_the_components_sum_to_the_score() -> None:
    """specs/04-domain-model.md — the invariant TrendState enforces."""
    mixed = readings(ema_alignment=1.0, ema_slope=1.0, structure=-1.0, rsi=1.0)
    score = score_evidence(mixed, weights=TrendWeights())

    assert math.fsum(score.components.values()) == pytest.approx(score.strength)


def test_dissenting_evidence_contributes_a_negative_share() -> None:
    """ADR 0010 D8 — "structure: -18" is a statement, not a rounding error."""
    mixed = readings(ema_alignment=1.0, ema_slope=1.0, structure=-1.0, rsi=1.0)
    score = score_evidence(mixed, weights=TrendWeights())

    assert score.components[StrengthComponent.STRUCTURE.value] < 0
    assert score.components[StrengthComponent.EMA_ALIGNMENT.value] > 0


def test_momentum_gathers_both_of_its_indicators() -> None:
    score = score_evidence(ALL_BULLISH, weights=TrendWeights())
    per_kind = 100.0 / len(TrendWeights().requested())

    assert score.components[StrengthComponent.MOMENTUM.value] == pytest.approx(2 * per_kind)
    assert score.components[StrengthComponent.PRICE_LOCATION.value] == pytest.approx(per_kind)


def test_unavailable_evidence_leaves_the_denominator_alone() -> None:
    """Half the evidence, all of it bullish, is still full bias — at half confidence."""
    partial = readings(ema_alignment=1.0, ema_slope=1.0, rsi=1.0)
    score = score_evidence(partial, weights=TrendWeights())

    assert score.bias == pytest.approx(1.0)
    assert score.confidence == pytest.approx(0.5)


def test_zero_weight_evidence_is_not_counted_as_missing() -> None:
    score = score_evidence(ALL_BULLISH, weights=TrendWeights())

    assert score.confidence == pytest.approx(1.0)


def test_no_readable_evidence_has_no_bias_at_all() -> None:
    score = score_evidence(readings(), weights=TrendWeights())

    assert score.bias is None
    assert score.strength == 0.0
    assert score.confidence == 0.0
    assert score.components == {}


def test_perfectly_balanced_evidence_scores_zero_strength() -> None:
    balanced = readings(ema_alignment=1.0, structure=-1.0)
    score = score_evidence(balanced, weights=TrendWeights())

    assert score.bias == pytest.approx(0.0)
    assert score.strength == pytest.approx(0.0)


def test_weights_change_the_bias() -> None:
    mixed = readings(ema_alignment=1.0, structure=-1.0)
    score = score_evidence(mixed, weights=TrendWeights(ema_alignment=3.0, structure=1.0))

    assert score.bias == pytest.approx(0.5)


def test_strength_never_leaves_its_range() -> None:
    for reading in (ALL_BULLISH, readings(), readings(rsi=-1.0)):
        score = score_evidence(reading, weights=TrendWeights())
        assert 0.0 <= score.strength <= 100.0
