"""Reading and scoring evidence — specs/08-trend.md, ADR 0010 D2/D3/D8/D9.

Pure functions over a snapshot. The engine is stateful because a bar stream is
recursive; deciding what six numbers mean is not, so it is separated out and can
be tested by writing the numbers down.

Nothing here decides a state. It produces votes, a bias and a score; the ladder
and its hysteresis live in `PhaseThresholds`, and the state machine in
`alphagate.core.trend`.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from types import MappingProxyType

from alphagate.core.structure import structural_bias
from alphagate.core.trend_engine.model import (
    EvidenceInputs,
    EvidenceKind,
    EvidenceReading,
    EvidenceScore,
    TrendConfig,
    TrendWeights,
)

__all__ = ["read_evidence", "score_evidence"]


def read_evidence(inputs: EvidenceInputs, *, config: TrendConfig) -> tuple[EvidenceReading, ...]:
    """One reading per evidence kind, in a fixed order, on every bar.

    Every kind is reported even when it cannot be read: "no opinion" is part of
    the explanation, and a reading that vanished would be indistinguishable from
    one that was never configured.
    """
    votes: dict[EvidenceKind, float | None] = {
        EvidenceKind.EMA_ALIGNMENT: _difference(inputs.fast_ema, inputs.slow_ema),
        EvidenceKind.PRICE_LOCATION: _difference(inputs.close, inputs.fast_ema),
        EvidenceKind.EMA_SLOPE: _slope(inputs, config=config),
        EvidenceKind.STRUCTURE: structural_bias(inputs.swing_high_label, inputs.swing_low_label),
        EvidenceKind.RSI: _rsi(inputs.rsi, config=config),
        EvidenceKind.MACD_HISTOGRAM: _sign(inputs.macd_histogram),
        # Declared, not computed: the MVP has no volume-context definition.
        EvidenceKind.VOLUME: None,
    }
    return tuple(EvidenceReading(kind=kind, vote=votes[kind]) for kind in EvidenceKind)


def _sign(value: float | None) -> float | None:
    if value is None:
        return None
    if value > 0:
        return 1.0
    return -1.0 if value < 0 else 0.0


def _difference(above: float | None, below: float | None) -> float | None:
    if above is None or below is None:
        return None
    return _sign(above - below)


def _slope(inputs: EvidenceInputs, *, config: TrendConfig) -> float | None:
    """The slow average against its own past, outside a deadband.

    The deadband is measured in ATR, so "flat" means flat relative to how much
    this instrument moves. When it is configured and ATR is still warming the
    slope is unreadable rather than assumed flat — unmeasurable is not an
    observation (ADR 0010 D2).
    """
    if inputs.slow_ema is None or inputs.slow_ema_before is None:
        return None

    change = inputs.slow_ema - inputs.slow_ema_before
    if config.slope_minimum_atr == 0.0:
        return _sign(change)

    if inputs.atr is None:
        return None
    deadband = config.slope_minimum_atr * inputs.atr
    return 0.0 if abs(change) < deadband else _sign(change)


def _rsi(value: float | None, *, config: TrendConfig) -> float | None:
    if value is None:
        return None
    if value > config.rsi_midpoint + config.rsi_band:
        return 1.0
    if value < config.rsi_midpoint - config.rsi_band:
        return -1.0
    return 0.0


def score_evidence(readings: Iterable[EvidenceReading], *, weights: TrendWeights) -> EvidenceScore:
    """Fold readings into a bias, a strength and its breakdown (ADR 0010 D3/D8).

    Unavailable evidence leaves both the numerator and the denominator, so bias
    is never dragged toward neutral by data that is merely still warming.
    """
    collected: Sequence[EvidenceReading] = tuple(readings)
    requested = weights.requested()
    votes = {reading.kind: reading.vote for reading in collected if reading.vote is not None}
    available = tuple(kind for kind in requested if kind in votes)

    confidence = len(available) / len(requested) if requested else 0.0
    if not available:
        return EvidenceScore(
            bias=None, strength=0.0, confidence=confidence, components=MappingProxyType({})
        )

    total_weight = math.fsum(weights.of(kind) for kind in available)
    bias = math.fsum(weights.of(kind) * _vote(votes, kind) for kind in available) / total_weight

    # Contributions are signed by the bias, not by the phase the engine holds:
    # phase lags evidence by design, and a breakdown that flipped sign whenever
    # the state machine lagged would be reporting the lag (ADR 0010 D8).
    direction = -1.0 if bias < 0 else 1.0
    components: dict[str, float] = {}
    for kind in available:
        share = 100.0 * direction * weights.of(kind) * _vote(votes, kind) / total_weight
        name = kind.component.value
        components[name] = components.get(name, 0.0) + share

    strength = min(max(math.fsum(components.values()), 0.0), 100.0)
    return EvidenceScore(
        bias=bias,
        strength=strength,
        confidence=confidence,
        components=MappingProxyType(components),
    )


def _vote(votes: Mapping[EvidenceKind, float | None], kind: EvidenceKind) -> float:
    vote = votes[kind]
    assert vote is not None  # noqa: S101 — `votes` holds only readable evidence
    return vote
