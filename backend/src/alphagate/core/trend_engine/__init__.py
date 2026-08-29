"""Trend engine — specs/08-trend.md, ADR 0010.

Turns indicators, market structure and price location into an explainable state
on a seven-rung ladder, plus a 0–100 strength score whose components are
returned alongside it. Not a crossover: a crossover is one of the six pieces of
evidence, and the state moves only when enough of them agree for long enough.

Three properties everything here is built around:

* **nothing is asserted that cannot be measured** — evidence that is still
  warming votes `None`, never zero, and `confidence` reports how much of the
  requested evidence was actually readable;
* **the state lags the evidence on purpose** — an exit margin and a confirmation
  count together are what "a one-tick crossover must not flip the main state"
  means in code;
* **every state explains itself** — reason codes, a per-component breakdown that
  sums to the score, and a hash of the evidence that produced both.

The state machine itself lives in `alphagate.core.trend`. This package decides what
to feed it. It raises no alerts and notifies no one (CLAUDE.md §8/§9); it emits
`TrendEvent`, and the alert engine decides whether anyone hears about it.
"""

from alphagate.core.trend_engine.engine import TrendEngine, phase_steps
from alphagate.core.trend_engine.evidence import read_evidence, score_evidence
from alphagate.core.trend_engine.model import (
    EvidenceInputs,
    EvidenceKind,
    EvidenceReading,
    EvidenceScore,
    PhaseThresholds,
    StrengthComponent,
    TrendConfig,
    TrendEvent,
    TrendEventKind,
    TrendUpdate,
    TrendWeights,
)

__all__ = [
    "EvidenceInputs",
    "EvidenceKind",
    "EvidenceReading",
    "EvidenceScore",
    "PhaseThresholds",
    "StrengthComponent",
    "TrendConfig",
    "TrendEngine",
    "TrendEvent",
    "TrendEventKind",
    "TrendUpdate",
    "TrendWeights",
    "phase_steps",
    "read_evidence",
    "score_evidence",
]
