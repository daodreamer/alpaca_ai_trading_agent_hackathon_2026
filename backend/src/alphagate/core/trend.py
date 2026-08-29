"""Trend state and its transition rules — specs/08-trend.md.

This module holds the *state machine*, not the engine. It answers "is this
transition legal and what does the resulting state look like", and knows nothing
about EMAs, structure or confirmation counting. The TrendEngine that evaluates
evidence arrives in Phase 5.

specs/08-trend.md says "not every transition is allowed" without enumerating
them, so the rule is derived from the shape of the state set instead of listed
by hand. The seven directional phases form an ordered ladder:

    STRONG_BEARISH — BEARISH — WATCH_BEAR — NEUTRAL — WATCH_BULL — BULLISH — STRONG_BULLISH

A transition may move at most one rung. That single rule delivers what the spec
asks for in prose:

* hysteresis — bullish cannot become bearish without passing through NEUTRAL,
  so a one-tick crossover cannot flip the main state;
* no state skipping — reaching STRONG_BULLISH requires actually passing through
  WATCH_BULL and BULLISH, which keeps the emitted event sequence explainable.

UNKNOWN sits off the ladder. Any phase may be entered from it (cold start after
warmup) and any phase may fall back to it (indicator data went missing).

When evidence jumps several rungs at once — a gap through a threshold — the
engine applies the move as a sequence of single steps rather than one leap, so
the event stream stays ordered.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from datetime import datetime
from enum import Enum
from types import MappingProxyType
from typing import Final, Self

from alphagate.core.errors import InvalidTransition, InvariantViolation
from alphagate.core.identifiers import Ticker
from alphagate.core.time_model import Timeframe, ensure_utc

__all__ = [
    "TrendDirection",
    "TrendPhase",
    "TrendState",
    "is_valid_transition",
]

STRENGTH_COMPONENT_TOLERANCE: Final = 1e-9
"""Approximate-domain tolerance from ADR 0005 D7."""


class TrendDirection(Enum):
    UNKNOWN = "UNKNOWN"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"
    BULLISH = "BULLISH"

    @property
    def is_directional(self) -> bool:
        """Whether this direction takes a side. NEUTRAL and UNKNOWN do not."""
        return self in (TrendDirection.BULLISH, TrendDirection.BEARISH)


class TrendPhase(Enum):
    UNKNOWN = "UNKNOWN"
    STRONG_BEARISH = "STRONG_BEARISH"
    BEARISH = "BEARISH"
    WATCH_BEAR = "WATCH_BEAR"
    NEUTRAL = "NEUTRAL"
    WATCH_BULL = "WATCH_BULL"
    BULLISH = "BULLISH"
    STRONG_BULLISH = "STRONG_BULLISH"

    @staticmethod
    def ladder() -> tuple[TrendPhase, ...]:
        """The seven directional phases, ordered most bearish to most bullish."""
        return _LADDER

    @property
    def direction(self) -> TrendDirection:
        return _DIRECTIONS[self]

    @property
    def rung(self) -> int | None:
        """Position on the ladder, or None for UNKNOWN."""
        return _LADDER.index(self) if self in _LADDER else None


_LADDER: Final[tuple[TrendPhase, ...]] = (
    TrendPhase.STRONG_BEARISH,
    TrendPhase.BEARISH,
    TrendPhase.WATCH_BEAR,
    TrendPhase.NEUTRAL,
    TrendPhase.WATCH_BULL,
    TrendPhase.BULLISH,
    TrendPhase.STRONG_BULLISH,
)

_DIRECTIONS: Final[Mapping[TrendPhase, TrendDirection]] = MappingProxyType(
    {
        TrendPhase.UNKNOWN: TrendDirection.UNKNOWN,
        TrendPhase.STRONG_BEARISH: TrendDirection.BEARISH,
        TrendPhase.BEARISH: TrendDirection.BEARISH,
        TrendPhase.WATCH_BEAR: TrendDirection.BEARISH,
        TrendPhase.NEUTRAL: TrendDirection.NEUTRAL,
        TrendPhase.WATCH_BULL: TrendDirection.BULLISH,
        TrendPhase.BULLISH: TrendDirection.BULLISH,
        TrendPhase.STRONG_BULLISH: TrendDirection.BULLISH,
    }
)


def is_valid_transition(current: TrendPhase, target: TrendPhase) -> bool:
    """Whether the machine may move from `current` to `target` in one step."""
    if current is TrendPhase.UNKNOWN or target is TrendPhase.UNKNOWN:
        return True
    here, there = current.rung, target.rung
    if here is None or there is None:  # UNKNOWN, already handled above
        return True
    return abs(there - here) <= 1


def _check_score(value: float, *, field: str, upper: float) -> None:
    if not 0.0 <= value <= upper:
        raise InvariantViolation(f"{field}: must be within [0, {upper}], got {value}")


@dataclasses.dataclass(frozen=True, slots=True)
class TrendState:
    """The trend of one symbol on one timeframe, at one instant.

    `direction` is derived rather than stored: two fields that must agree are two
    fields that can disagree.
    """

    symbol: Ticker
    timeframe: Timeframe
    state: TrendPhase
    strength: float
    confidence: float
    as_of: datetime
    reason_codes: tuple[str, ...]
    input_snapshot_hash: str
    confirmed_at: datetime | None = None
    strength_components: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        _check_score(self.strength, field="strength", upper=100.0)
        _check_score(self.confidence, field="confidence", upper=1.0)

        object.__setattr__(self, "as_of", ensure_utc(self.as_of, field="as_of"))
        if self.confirmed_at is not None:
            object.__setattr__(
                self, "confirmed_at", ensure_utc(self.confirmed_at, field="confirmed_at")
            )

        if self.state is not TrendPhase.UNKNOWN and not self.reason_codes:
            raise InvariantViolation(
                f"{self.state.value}: reason_codes are required — every state must be "
                "explainable (specs/00-vision.md)"
            )
        if not self.input_snapshot_hash:
            raise InvariantViolation(
                "input_snapshot_hash is required so a state can be tied back to the "
                "inputs that produced it (CLAUDE.md §11)"
            )

        if self.strength_components is not None:
            total = sum(self.strength_components.values())
            if abs(total - self.strength) > STRENGTH_COMPONENT_TOLERANCE:
                raise InvariantViolation(
                    f"strength components sum to {total}, which does not match "
                    f"strength {self.strength}"
                )
            object.__setattr__(
                self, "strength_components", MappingProxyType(dict(self.strength_components))
            )

    @property
    def direction(self) -> TrendDirection:
        return self.state.direction

    def __deepcopy__(self, memo: dict[int, object]) -> Self:
        """A deep copy of an immutable value is the value.

        Frozen, with immutable fields throughout, so sharing is safe. Stated
        explicitly because `strength_components` is a `mappingproxy`, which
        `copy.deepcopy` refuses outright — and `BarConsumer._provisional` copies
        whatever holds a state in order to evaluate a partial bar.
        """
        return self

    def transition_to(
        self,
        target: TrendPhase,
        *,
        as_of: datetime,
        strength: float,
        confidence: float,
        reason_codes: tuple[str, ...],
        input_snapshot_hash: str,
        strength_components: Mapping[str, float] | None = None,
    ) -> Self:
        """Return the state that follows this one.

        Pure: the receiver is unchanged. Raises rather than clamping, because a
        rejected transition is a bug in the caller, not a value to round off.
        """
        if not is_valid_transition(self.state, target):
            raise InvalidTransition(
                f"{self.state.value} -> {target.value} is not a legal transition; "
                "the ladder moves one rung at a time and direction flips must pass "
                "through NEUTRAL (specs/08-trend.md)"
            )
        moment = ensure_utc(as_of, field="as_of")
        if moment < self.as_of:
            raise InvariantViolation(
                f"as_of ({moment.isoformat()}) must not precede the current state's "
                f"as_of ({self.as_of.isoformat()})"
            )

        return dataclasses.replace(
            self,
            state=target,
            as_of=moment,
            strength=strength,
            confidence=confidence,
            reason_codes=reason_codes,
            input_snapshot_hash=input_snapshot_hash,
            strength_components=strength_components,
            confirmed_at=self._confirmation_time(target, moment),
        )

    def _confirmation_time(self, target: TrendPhase, moment: datetime) -> datetime | None:
        """When the *current* directional stance was first established.

        Cleared on returning to NEUTRAL/UNKNOWN, restamped when the side changes,
        and otherwise carried forward — so "bullish since 10:30" survives a move
        from BULLISH to STRONG_BULLISH.
        """
        if not target.direction.is_directional:
            return None
        if self.direction is not target.direction or self.confirmed_at is None:
            return moment
        return self.confirmed_at
