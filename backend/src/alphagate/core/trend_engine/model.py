"""Trend-engine inputs, configuration and outputs — specs/08-trend.md, ADR 0010.

Everything that decides what the engine says about a market lives here, as
configuration: thresholds, weights, bands, deadbands, confirmation and margin.
`specs/08-trend.md` requires that the strength score be explainable; a number
that changes the answer and is not visible in one place cannot be explained.

The state machine itself is not here — it is `alphagate.core.trend`, where it has
been since Phase 1. This module supplies what feeds it.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from datetime import datetime
from enum import Enum
from types import MappingProxyType
from typing import Final

from alphagate.core.errors import InvariantViolation
from alphagate.core.identifiers import Ticker
from alphagate.core.indicators import PriceSource
from alphagate.core.streaming import require_period
from alphagate.core.structure import StructureLabel
from alphagate.core.time_model import Timeframe, ensure_utc
from alphagate.core.trend import TrendPhase, TrendState, is_valid_transition

__all__ = [
    "EvidenceInputs",
    "EvidenceKind",
    "EvidenceReading",
    "EvidenceScore",
    "PhaseThresholds",
    "StrengthComponent",
    "TrendConfig",
    "TrendEvent",
    "TrendEventKind",
    "TrendUpdate",
    "TrendWeights",
]


class StrengthComponent(Enum):
    """The score breakdown `specs/08-trend.md` names.

    Several pieces of evidence can share one component — RSI and the MACD
    histogram are both momentum — because the breakdown is for a reader, and a
    reader wants five lines, not seven.
    """

    EMA_ALIGNMENT = "ema_alignment"
    STRUCTURE = "structure"
    MOMENTUM = "momentum"
    PRICE_LOCATION = "price_location"
    VOLUME = "volume"


class EvidenceKind(Enum):
    """One configurable condition from `specs/08-trend.md`.

    The bullish and bearish lists in the spec are the same conditions read in
    opposite directions, so they are one enum, not two (ADR 0010 D2).
    """

    EMA_ALIGNMENT = "EMA_ALIGNMENT"
    EMA_SLOPE = "EMA_SLOPE"
    PRICE_LOCATION = "PRICE_LOCATION"
    STRUCTURE = "STRUCTURE"
    RSI = "RSI"
    MACD_HISTOGRAM = "MACD_HISTOGRAM"
    VOLUME = "VOLUME"
    """Declared, never computed in the MVP: the spec's breakdown names a volume
    component while its evidence list has none (ADR 0010 D2)."""

    @property
    def component(self) -> StrengthComponent:
        return _COMPONENTS[self]

    @property
    def weight_field(self) -> str:
        return self.value.lower()


_COMPONENTS: Final[Mapping[EvidenceKind, StrengthComponent]] = MappingProxyType(
    {
        EvidenceKind.EMA_ALIGNMENT: StrengthComponent.EMA_ALIGNMENT,
        EvidenceKind.EMA_SLOPE: StrengthComponent.EMA_ALIGNMENT,
        EvidenceKind.PRICE_LOCATION: StrengthComponent.PRICE_LOCATION,
        EvidenceKind.STRUCTURE: StrengthComponent.STRUCTURE,
        EvidenceKind.RSI: StrengthComponent.MOMENTUM,
        EvidenceKind.MACD_HISTOGRAM: StrengthComponent.MOMENTUM,
        EvidenceKind.VOLUME: StrengthComponent.VOLUME,
    }
)


@dataclasses.dataclass(frozen=True, slots=True)
class EvidenceReading:
    """One condition's opinion: `+1` bullish, `-1` bearish, `0` undecided.

    `None` is not `0`. A warming EMA has no opinion; a flat EMA has the opinion
    that nothing is happening, and collapsing the two would make an engine at
    bar three look like an engine in a dead market (ADR 0010 D2).
    """

    kind: EvidenceKind
    vote: float | None

    def __post_init__(self) -> None:
        if self.vote is not None and not -1.0 <= self.vote <= 1.0:
            raise InvariantViolation(
                f"{self.kind.value}: a vote must be within [-1, 1], got {self.vote}"
            )

    @property
    def reason_code(self) -> str | None:
        """What this reading contributes to a state's explanation."""
        if self.vote is None:
            return None
        if self.vote > 0:
            return f"{self.kind.value}_BULL"
        if self.vote < 0:
            return f"{self.kind.value}_BEAR"
        return f"{self.kind.value}_FLAT"


@dataclasses.dataclass(frozen=True, slots=True)
class EvidenceInputs:
    """Everything the evidence table reads, as of one bar.

    A snapshot rather than an engine handle, so every case the spec asks to be
    tested — a missing value, a value exactly on a boundary — can be written
    down instead of arranged for.
    """

    close: float
    fast_ema: float | None
    slow_ema: float | None
    slow_ema_before: float | None
    atr: float | None
    rsi: float | None
    macd_histogram: float | None
    swing_high_label: StructureLabel | None
    swing_low_label: StructureLabel | None


@dataclasses.dataclass(frozen=True, slots=True)
class EvidenceScore:
    """What the readings add up to (ADR 0010 D3/D8/D9)."""

    bias: float | None
    """Weighted mean of the available votes, in [-1, 1]. `None` when nothing is
    readable at all."""

    strength: float
    """`100 * |bias|` — the magnitude of directional agreement, 0–100."""

    confidence: float
    """Data sufficiency: readable evidence over requested evidence."""

    components: Mapping[str, float]
    """Per-component contributions, summing to `strength` by construction. A
    component that argues against the prevailing bias contributes a negative
    number, which is a statement rather than a rounding error."""


@dataclasses.dataclass(frozen=True, slots=True)
class TrendWeights:
    """How much each condition counts (specs/08-trend.md, ADR 0010 D2)."""

    ema_alignment: float = 1.0
    ema_slope: float = 1.0
    price_location: float = 1.0
    structure: float = 1.0
    rsi: float = 1.0
    macd_histogram: float = 1.0
    volume: float = 0.0
    """Zero, and refused above zero: the MVP has no definition for volume
    evidence. Keeping the field means enabling it later is a configuration
    change and one deleted guard, not a schema change."""

    def __post_init__(self) -> None:
        for field in dataclasses.fields(self):
            weight = getattr(self, field.name)
            if weight < 0:
                raise InvariantViolation(f"{field.name}: weight must not be negative, got {weight}")
        if self.volume > 0:
            raise InvariantViolation(
                "volume: declared but not computed in the MVP, so weighting it would "
                "only lower confidence — see ADR 0010 D2"
            )
        if not self.requested():
            raise InvariantViolation("at least one evidence weight must be positive")

    def of(self, kind: EvidenceKind) -> float:
        weight: float = getattr(self, kind.weight_field)
        return weight

    def requested(self) -> tuple[EvidenceKind, ...]:
        """The evidence that actually counts — everything with a weight."""
        return tuple(kind for kind in EvidenceKind if self.of(kind) > 0)

    @property
    def label(self) -> str:
        return "/".join(str(getattr(self, field.name)) for field in dataclasses.fields(self))


@dataclasses.dataclass(frozen=True, slots=True)
class PhaseThresholds:
    """Where the ladder's rungs sit on the bias scale (ADR 0010 D4).

    Symmetric by construction: one set of thresholds serves both sides, so a
    bullish market and its mirror image produce mirror-image phases.
    """

    watch: float = 0.2
    trend: float = 0.5
    strong: float = 0.8

    def __post_init__(self) -> None:
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if not 0.0 < value <= 1.0:
                raise InvariantViolation(
                    f"{field.name}: a threshold must be within (0, 1], got {value}"
                )
        if not self.watch < self.trend < self.strong:
            raise InvariantViolation(
                "thresholds must be ordered watch < trend < strong, got "
                f"{self.watch} / {self.trend} / {self.strong}"
            )

    @property
    def label(self) -> str:
        return f"{self.watch}/{self.trend}/{self.strong}"

    def phase_for(self, bias: float) -> TrendPhase:
        """The phase this bias names, ignoring whatever phase is held.

        A boundary belongs to the stronger phase: exactly `trend` of bullish
        bias is BULLISH, which is how "≥ threshold" reads in the spec's prose.
        """
        if bias >= self.strong:
            return TrendPhase.STRONG_BULLISH
        if bias >= self.trend:
            return TrendPhase.BULLISH
        if bias >= self.watch:
            return TrendPhase.WATCH_BULL
        if bias > -self.watch:
            return TrendPhase.NEUTRAL
        if bias > -self.trend:
            return TrendPhase.WATCH_BEAR
        if bias > -self.strong:
            return TrendPhase.BEARISH
        return TrendPhase.STRONG_BEARISH

    def band_of(self, phase: TrendPhase) -> tuple[float, float]:
        """The bias range `phase_for` maps onto `phase`.

        Unbounded at the ends, because bias is bounded and the outermost phases
        have nowhere further to go.
        """
        match phase:
            case TrendPhase.STRONG_BULLISH:
                return (self.strong, math.inf)
            case TrendPhase.BULLISH:
                return (self.trend, self.strong)
            case TrendPhase.WATCH_BULL:
                return (self.watch, self.trend)
            case TrendPhase.NEUTRAL:
                return (-self.watch, self.watch)
            case TrendPhase.WATCH_BEAR:
                return (-self.trend, -self.watch)
            case TrendPhase.BEARISH:
                return (-self.strong, -self.trend)
            case TrendPhase.STRONG_BEARISH:
                return (-math.inf, -self.strong)
            case TrendPhase.UNKNOWN:
                raise InvariantViolation(
                    "UNKNOWN is off the ladder and holds no band: it is left the moment "
                    "there is evidence, and entered the moment there is not"
                )

    def target_from(self, bias: float, *, current: TrendPhase, exit_margin: float) -> TrendPhase:
        """The phase to aim for, given the one already held (ADR 0010 D5).

        Leaving the held phase requires clearing its band by `exit_margin`;
        returning to it requires nothing. The margin defends the phase you hold,
        it is not a toll on coming back.
        """
        if current is TrendPhase.UNKNOWN:
            return self.phase_for(bias)

        low, high = self.band_of(current)
        if bias >= high + exit_margin or bias <= low - exit_margin:
            return self.phase_for(bias)
        return current


@dataclasses.dataclass(frozen=True, slots=True)
class TrendConfig:
    """Everything the trend engine is allowed to decide with."""

    fast_period: int = 20
    slow_period: int = 50
    slope_lookback: int = 5
    slope_minimum_atr: float = 0.0
    """Slope deadband, in ATR. Zero — off — by default: a deadband is a policy,
    and a policy that is on by default is a hidden constant."""

    atr_period: int = 14
    rsi_period: int = 14
    rsi_midpoint: float = 50.0
    rsi_band: float = 0.0
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    source: PriceSource = PriceSource.CLOSE
    weights: TrendWeights = dataclasses.field(default_factory=TrendWeights)
    thresholds: PhaseThresholds = dataclasses.field(default_factory=PhaseThresholds)
    exit_margin: float = 0.05
    confirmation_bars: int = 2
    minimum_confidence: float = 0.5
    """Below this share of readable evidence the state is UNKNOWN — the engine
    says "I cannot see" rather than guessing from what is left."""

    def __post_init__(self) -> None:
        for field in (
            "fast_period",
            "slow_period",
            "slope_lookback",
            "atr_period",
            "rsi_period",
            "macd_fast",
            "macd_slow",
            "macd_signal",
            "confirmation_bars",
        ):
            require_period(getattr(self, field), field=field)

        if self.fast_period >= self.slow_period:
            raise InvariantViolation(
                f"slow_period ({self.slow_period}) must be longer than fast_period "
                f"({self.fast_period}); otherwise the alignment reads backwards"
            )
        if self.macd_fast >= self.macd_slow:
            raise InvariantViolation(
                f"macd_slow ({self.macd_slow}) must be longer than macd_fast ({self.macd_fast})"
            )
        if self.slope_minimum_atr < 0:
            raise InvariantViolation(
                f"slope_minimum_atr: must not be negative, got {self.slope_minimum_atr}"
            )
        if not 0.0 < self.rsi_midpoint < 100.0:
            raise InvariantViolation(
                f"rsi_midpoint: must sit inside the RSI scale, got {self.rsi_midpoint}"
            )
        if self.rsi_band < 0 or self.rsi_midpoint - self.rsi_band <= 0.0:
            raise InvariantViolation(
                f"rsi_band: must be non-negative and leave a bearish side, got {self.rsi_band}"
            )
        if self.rsi_midpoint + self.rsi_band >= 100.0:
            raise InvariantViolation(f"rsi_band: must leave a bullish side, got {self.rsi_band}")
        if self.exit_margin < 0:
            raise InvariantViolation(f"exit_margin: must not be negative, got {self.exit_margin}")
        if not 0.0 <= self.minimum_confidence <= 1.0:
            raise InvariantViolation(
                f"minimum_confidence: must be within [0, 1], got {self.minimum_confidence}"
            )


class TrendEventKind(Enum):
    """The only state changes worth reporting (specs/08-trend.md)."""

    TREND_CONFIRMED = "TREND_CONFIRMED"
    TREND_REVERSAL = "TREND_REVERSAL"
    TREND_STRENGTH_INCREASED = "TREND_STRENGTH_INCREASED"
    TREND_STRENGTH_DECREASED = "TREND_STRENGTH_DECREASED"


@dataclasses.dataclass(frozen=True, slots=True)
class TrendEvent:
    """One rung of movement, with the evidence that moved it.

    A domain event, not an alert. Whether it reaches anyone — cooldown, debounce,
    re-arm, delivery — is the alert engine's decision (CLAUDE.md §8).
    """

    kind: TrendEventKind
    symbol: Ticker
    timeframe: Timeframe
    from_phase: TrendPhase
    to_phase: TrendPhase
    bar_start_utc: datetime
    strength: float
    reason_codes: tuple[str, ...]
    input_snapshot_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "bar_start_utc", ensure_utc(self.bar_start_utc, field="bar_start_utc")
        )
        if self.from_phase is self.to_phase:
            raise InvariantViolation(
                f"{self.from_phase.value}: an event must record a change of phase"
            )
        if not is_valid_transition(self.from_phase, self.to_phase):
            raise InvariantViolation(
                f"{self.from_phase.value} -> {self.to_phase.value}: an event carries one rung "
                "of movement; a multi-rung move is a sequence of events (ADR 0010 D6)"
            )
        if not 0.0 <= self.strength <= 100.0:
            raise InvariantViolation(f"strength: must be within [0, 100], got {self.strength}")
        if not self.input_snapshot_hash:
            raise InvariantViolation(
                "input_snapshot_hash is required so an event can be tied back to the "
                "evidence that raised it"
            )


@dataclasses.dataclass(frozen=True, slots=True)
class TrendUpdate:
    """What one bar produced.

    Returned for every bar, because "the trend did not change" is an answer. The
    readings and the bias travel with the state so a UI can show why without
    asking the engine a second question.
    """

    state: TrendState
    events: tuple[TrendEvent, ...]
    evidence: tuple[EvidenceReading, ...]
    bias: float | None
    target: TrendPhase
    """The phase the evidence asks for, which is not yet the phase held while
    confirmation is still counting."""
