"""The trend engine — specs/08-trend.md, ADR 0010.

Composes indicators and the structure engine into an explainable state machine.
It computes no indicator of its own (CLAUDE.md §4), identifies no structure
(§5), and raises no alerts (§8): it decides which rung of the ladder the
evidence supports, how confident it is, and which of the four spec'd events that
movement is.

Everything it emits carries the bar that produced it and a hash of the evidence
behind it, so "why is this bullish" is answerable after the fact.
"""

from __future__ import annotations

import hashlib
import math
from collections import deque

from alphagate.core.bar import Bar
from alphagate.core.errors import InvariantViolation
from alphagate.core.indicators import Atr, Ema, Macd, Rsi
from alphagate.core.streaming import BarConsumer, ComponentSpec
from alphagate.core.structure import StructureEngine, StructureLabel, SwingKind
from alphagate.core.trend import TrendDirection, TrendPhase, TrendState
from alphagate.core.trend_engine.evidence import read_evidence, score_evidence
from alphagate.core.trend_engine.model import (
    EvidenceInputs,
    EvidenceKind,
    EvidenceReading,
    EvidenceScore,
    TrendConfig,
    TrendEvent,
    TrendEventKind,
    TrendUpdate,
)

__all__ = ["TrendEngine", "phase_steps"]


def phase_steps(current: TrendPhase, target: TrendPhase) -> tuple[TrendPhase, ...]:
    """The single-rung moves that take `current` to `target` (ADR 0010 D6).

    A gap through a threshold is applied as a sequence rather than one leap, so
    the event stream stays ordered and each step carries its own reason. UNKNOWN
    is exempt in both directions: it is off the ladder, so a cold start enters
    directly and a loss of evidence drops straight out.
    """
    if current is target:
        return ()
    if current is TrendPhase.UNKNOWN or target is TrendPhase.UNKNOWN:
        return (target,)

    here, there = current.rung, target.rung
    if here is None or there is None:  # pragma: no cover — UNKNOWN handled above
        return (target,)

    ladder = TrendPhase.ladder()
    step = 1 if there > here else -1
    return tuple(ladder[rung] for rung in range(here + step, there + step, step))


class TrendEngine(BarConsumer[TrendUpdate]):
    """Trend state, strength and events for one symbol on one timeframe."""

    def __init__(
        self,
        *,
        config: TrendConfig | None = None,
        structure: StructureEngine | None = None,
        allow_partial: bool = False,
    ) -> None:
        super().__init__(allow_partial=allow_partial)
        self.config = config if config is not None else TrendConfig()

        if structure is not None and structure.committed_bars:
            raise InvariantViolation(
                "the trend engine needs a fresh StructureEngine: one that has already "
                f"consumed {structure.committed_bars} bars would make the trend depend "
                "on history this engine never saw"
            )
        self._structure = structure if structure is not None else StructureEngine()

        settings = self.config
        self._fast = Ema(period=settings.fast_period, source=settings.source)
        self._slow = Ema(period=settings.slow_period, source=settings.source)
        self._atr = Atr(period=settings.atr_period)
        self._rsi = Rsi(period=settings.rsi_period, source=settings.source)
        self._macd = Macd(
            fast=settings.macd_fast,
            slow=settings.macd_slow,
            signal=settings.macd_signal,
            source=settings.source,
        )

        self._slow_history: deque[float | None] = deque(maxlen=settings.slope_lookback + 1)
        self._labels: dict[SwingKind, StructureLabel | None] = {
            SwingKind.HIGH: None,
            SwingKind.LOW: None,
        }

        self._state: TrendState | None = None
        self._events: list[TrendEvent] = []
        self._last_direction: TrendDirection = TrendDirection.UNKNOWN
        self._pending: TrendPhase = TrendPhase.UNKNOWN
        self._pending_bars = 0

    # -- identity ----------------------------------------------------------

    @property
    def spec(self) -> ComponentSpec:
        settings = self.config
        return ComponentSpec(
            "trend",
            (
                ("fast", str(settings.fast_period)),
                ("slow", str(settings.slow_period)),
                ("slope", str(settings.slope_lookback)),
                ("slope_atr", str(settings.slope_minimum_atr)),
                ("atr", str(settings.atr_period)),
                ("rsi", str(settings.rsi_period)),
                ("rsi_mid", str(settings.rsi_midpoint)),
                ("rsi_band", str(settings.rsi_band)),
                ("macd", f"{settings.macd_fast}/{settings.macd_slow}/{settings.macd_signal}"),
                ("src", settings.source.value),
                ("thresholds", settings.thresholds.label),
                ("margin", str(settings.exit_margin)),
                ("confirm", str(settings.confirmation_bars)),
                ("min_conf", str(settings.minimum_confidence)),
                ("weights", settings.weights.label),
                ("structure", self._structure.spec.key),
            ),
        )

    @property
    def warmup(self) -> int:
        """The bar at which the confidence floor can first be met (ADR 0010 D11).

        Not the slowest input: with `minimum_confidence` below 1 the engine
        legitimately produces a state before every indicator is warm, and a
        warmup that over-reported would have callers discard valid states.
        """
        requested = self.config.weights.requested()
        needed = sorted(self._warmup_of(kind) for kind in requested)
        enough = math.ceil(self.config.minimum_confidence * len(needed))
        index = min(max(enough, 1), len(needed)) - 1
        return needed[index] + self.config.confirmation_bars - 1

    def _warmup_of(self, kind: EvidenceKind) -> int:
        match kind:
            case EvidenceKind.EMA_ALIGNMENT:
                return max(self._fast.warmup, self._slow.warmup)
            case EvidenceKind.PRICE_LOCATION:
                return self._fast.warmup
            case EvidenceKind.EMA_SLOPE:
                return self._slow.warmup + self.config.slope_lookback
            case EvidenceKind.STRUCTURE:
                return self._structure.warmup
            case EvidenceKind.RSI:
                return self._rsi.warmup
            case EvidenceKind.MACD_HISTOGRAM:
                return self._macd.warmup
            case EvidenceKind.VOLUME:  # pragma: no cover — TrendWeights forbids the weight
                raise InvariantViolation(
                    "volume evidence has no warmup because it is never computed (ADR 0010 D2)"
                )

    # -- reading -----------------------------------------------------------

    @property
    def state(self) -> TrendState | None:
        """The trend as of the last committed bar, or `None` before the first."""
        return self._state

    @property
    def events(self) -> tuple[TrendEvent, ...]:
        """Every event emitted so far, in order. Append-only."""
        return tuple(self._events)

    @property
    def structure(self) -> StructureEngine:
        """The composed structure engine — readable, so a UI can draw the swings."""
        return self._structure

    # -- writing -----------------------------------------------------------

    def _consume(self, bar: Bar) -> TrendUpdate:
        readings = read_evidence(self._observe(bar), config=self.config)
        score = score_evidence(readings, weights=self.config.weights)
        digest = self._digest(bar, readings, score.bias)

        held = self._state.state if self._state is not None else TrendPhase.UNKNOWN
        target = self._target(score, held=held)
        moves = self._moves(held, target)

        state = self._state if self._state is not None else _initial(bar, digest)
        events: list[TrendEvent] = []
        # With no move to make, the state is still restamped: strength, evidence
        # and confidence are this bar's, even when the phase is last bar's.
        for step in moves if moves else (state.state,):
            before = state.state
            state = state.transition_to(
                step,
                as_of=bar.start_time_utc,
                strength=score.strength,
                confidence=score.confidence,
                reason_codes=_reason_codes(readings),
                input_snapshot_hash=digest,
                strength_components=score.components if score.components else None,
            )
            event = self._event(before, step)
            if event is not None:
                events.append(
                    TrendEvent(
                        kind=event,
                        symbol=bar.symbol,
                        timeframe=bar.timeframe,
                        from_phase=before,
                        to_phase=step,
                        bar_start_utc=bar.start_time_utc,
                        strength=score.strength,
                        reason_codes=_reason_codes(readings),
                        input_snapshot_hash=digest,
                    )
                )

        self._state = state
        self._events.extend(events)
        return TrendUpdate(
            state=state,
            events=tuple(events),
            evidence=readings,
            bias=score.bias,
            target=target,
        )

    # -- evidence ----------------------------------------------------------

    def _observe(self, bar: Bar) -> EvidenceInputs:
        """Advance every composed component by one bar and snapshot the result."""
        fast = self._fast.update(bar)
        slow = self._slow.update(bar)
        atr = self._atr.update(bar)
        rsi = self._rsi.update(bar)
        macd = self._macd.update(bar)

        # Structure answers on every committed bar; the `None` arm exists only
        # because `BarConsumer.update` is generic over components that warm up.
        structure = self._structure.update(bar)
        for swing in structure.confirmed_swings if structure is not None else ():
            self._labels[swing.kind] = swing.label

        self._slow_history.append(slow)
        before = (
            self._slow_history[0] if len(self._slow_history) == self._slow_history.maxlen else None
        )

        return EvidenceInputs(
            close=self.config.source.of(bar),
            fast_ema=fast,
            slow_ema=slow,
            slow_ema_before=before,
            atr=atr,
            rsi=rsi,
            macd_histogram=macd.histogram if macd is not None else None,
            swing_high_label=self._labels[SwingKind.HIGH],
            swing_low_label=self._labels[SwingKind.LOW],
        )

    # -- state -------------------------------------------------------------

    @property
    def structure_labels(self) -> tuple[StructureLabel | None, StructureLabel | None]:
        """The latest confirmed swing high and low labels, in that order.

        Already held here as `STRUCTURE` evidence; exposed because the
        multi-timeframe confluence reads the same pair (ADR 0024) and deriving it
        a second time from the structure engine would be a second answer to the
        same question.
        """
        return (self._labels[SwingKind.HIGH], self._labels[SwingKind.LOW])

    def _target(self, score: EvidenceScore, *, held: TrendPhase) -> TrendPhase:
        if score.bias is None or score.confidence < self.config.minimum_confidence:
            return TrendPhase.UNKNOWN
        return self.config.thresholds.target_from(
            score.bias, current=held, exit_margin=self.config.exit_margin
        )

    def _moves(self, held: TrendPhase, target: TrendPhase) -> tuple[TrendPhase, ...]:
        """Apply confirmation, then hand back the rungs to walk (ADR 0010 D5/D7)."""
        if target is held:
            self._pending, self._pending_bars = held, 0
            return ()

        if target is TrendPhase.UNKNOWN:
            # Losing evidence is not a trend judgement, so it does not wait for
            # confirmation: waiting would mean reporting a trend nothing can see.
            self._pending, self._pending_bars = TrendPhase.UNKNOWN, 0
            return phase_steps(held, target)

        if target is self._pending:
            self._pending_bars += 1
        else:
            self._pending, self._pending_bars = target, 1

        if self._pending_bars < self.config.confirmation_bars:
            return ()

        self._pending_bars = 0
        return phase_steps(held, target)

    def _event(self, before: TrendPhase, after: TrendPhase) -> TrendEventKind | None:
        """Which of the four spec'd events one rung of movement is (ADR 0010 D10).

        Also the only place the engine's *stance* is recorded — the last side it
        took — which is what distinguishes a reversal from a first confirmation.
        """
        if before is after or after is TrendPhase.UNKNOWN:
            return None

        if not after.direction.is_directional:
            # Reaching NEUTRAL from a trend is that trend fading; reaching it from
            # UNKNOWN is the engine warming up, which is nobody's news.
            if before is TrendPhase.UNKNOWN:
                return None
            return TrendEventKind.TREND_STRENGTH_DECREASED

        if before.direction is not after.direction:
            reversed_stance = (
                self._last_direction.is_directional and self._last_direction is not after.direction
            )
            self._last_direction = after.direction
            return (
                TrendEventKind.TREND_REVERSAL if reversed_stance else TrendEventKind.TREND_CONFIRMED
            )

        self._last_direction = after.direction
        return _stronger(before, after)

    def _digest(self, bar: Bar, readings: tuple[EvidenceReading, ...], bias: float | None) -> str:
        """A fingerprint of the evidence, not of the bars (ADR 0010 D12).

        Covers the engine's identity, the bar, every vote — including the ones
        that could not be read — and the bias they produced. Equal across a
        rebuild, and enough to tie a state back to what caused it.
        """
        parts = [self.spec.key, bar.start_time_utc.isoformat()]
        parts += [
            f"{reading.kind.value}={'' if reading.vote is None else repr(reading.vote)}"
            for reading in readings
        ]
        parts.append(f"bias={'' if bias is None else repr(bias)}")
        return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def _initial(bar: Bar, digest: str) -> TrendState:
    """The state before any evidence: unknown, and honest about it."""
    return TrendState(
        symbol=bar.symbol,
        timeframe=bar.timeframe,
        state=TrendPhase.UNKNOWN,
        strength=0.0,
        confidence=0.0,
        as_of=bar.start_time_utc,
        reason_codes=(),
        input_snapshot_hash=digest,
    )


def _reason_codes(readings: tuple[EvidenceReading, ...]) -> tuple[str, ...]:
    return tuple(reading.reason_code for reading in readings if reading.reason_code is not None)


def _stronger(before: TrendPhase, after: TrendPhase) -> TrendEventKind:
    """Movement along the ladder *is* the trend's strength, as far as events go."""
    if _from_neutral(after) > _from_neutral(before):
        return TrendEventKind.TREND_STRENGTH_INCREASED
    return TrendEventKind.TREND_STRENGTH_DECREASED


def _from_neutral(phase: TrendPhase) -> int:
    """How far up or down the ladder a phase sits. Never called for UNKNOWN."""
    ladder = TrendPhase.ladder()
    return abs(ladder.index(phase) - ladder.index(TrendPhase.NEUTRAL))
