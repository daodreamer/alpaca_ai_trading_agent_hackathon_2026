"""What the structure engine says — ADR 0008 D3/D5/D7.

Every statement here carries two instants: the bar it is *about*, and the bar at
which it became knowable. That pair is the whole anti-look-ahead argument, and
it is data rather than a comment so a test can assert on it (CLAUDE.md §12).
"""

from __future__ import annotations

import dataclasses
from datetime import datetime
from decimal import Decimal
from enum import Enum

from alphagate.core.errors import InvariantViolation
from alphagate.core.numeric import price as exact_price
from alphagate.core.time_model import ensure_utc

__all__ = [
    "BreakEvent",
    "BreakKind",
    "EqualPricePolicy",
    "StructureLabel",
    "StructureUpdate",
    "SwingKind",
    "SwingPoint",
    "SwingStatus",
    "structural_bias",
]


class SwingKind(Enum):
    HIGH = "HIGH"
    LOW = "LOW"


class SwingStatus(Enum):
    CANDIDATE = "CANDIDATE"
    """Satisfies the left window; the right window has not filled in yet.

    Drawable, never structural: labels, the noise filter and break detection all
    ignore candidates (ADR 0008 D3).
    """

    CONFIRMED = "CONFIRMED"


class StructureLabel(Enum):
    """Relation to the previous swing of the same kind (specs/06-structure.md)."""

    HH = "HH"
    HL = "HL"
    LH = "LH"
    LL = "LL"


def structural_bias(high: StructureLabel | None, low: StructureLabel | None) -> float | None:
    """Which way a pair of swing labels points: `+1`, `-1`, `0` or unreadable.

    Confirmed HH *and* HL is bullish; LH *and* LL is bearish; a mix is neither.
    Both sides are required because `specs/08-trend.md` names the pair — one
    higher high beside a lower low is a widening range, not a trend.

    `None` is not `0`. A missing label has no opinion; a mixed pair has the
    opinion that structure is not taking a side.

    Lives here rather than inside an engine because two of them read it — the
    trend engine's `STRUCTURE` evidence and the multi-timeframe confluence
    (ADR 0024) — and a rule spelled out twice is a rule that will disagree with
    itself.
    """
    if high is None or low is None:
        return None
    if high is StructureLabel.HH and low is StructureLabel.HL:
        return 1.0
    if high is StructureLabel.LH and low is StructureLabel.LL:
        return -1.0
    return 0.0


class EqualPricePolicy(Enum):
    """How an exactly equal high or low is labelled (ADR 0008 D5)."""

    STRICT = "STRICT"
    """"Higher" requires strictly higher: an equal high is LH, an equal low LL."""

    INCLUSIVE = "INCLUSIVE"
    """An equal high is HH, an equal low HL."""


class BreakKind(Enum):
    BREAKOUT = "BREAKOUT"
    BREAKDOWN = "BREAKDOWN"


@dataclasses.dataclass(frozen=True, slots=True)
class SwingPoint:
    """A pivot high or low.

    `price` is the pivot bar's high for a swing high and its low for a swing low
    — the extreme itself, not the close.
    """

    kind: SwingKind
    status: SwingStatus
    bar_start_utc: datetime
    detected_at_utc: datetime
    price: Decimal
    label: StructureLabel | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "price", exact_price(self.price, field="price"))
        if self.price <= 0:
            raise InvariantViolation(f"price: must be positive, got {self.price}")

        pivot = ensure_utc(self.bar_start_utc, field="bar_start_utc")
        detected = ensure_utc(self.detected_at_utc, field="detected_at_utc")
        object.__setattr__(self, "bar_start_utc", pivot)
        object.__setattr__(self, "detected_at_utc", detected)

        if detected < pivot:
            raise InvariantViolation(
                f"detected_at_utc ({detected.isoformat()}) must not precede the pivot "
                f"({pivot.isoformat()}): a swing cannot be known before it happened"
            )
        if self.status is SwingStatus.CONFIRMED and detected == pivot:
            raise InvariantViolation(
                "a confirmed swing needs right-side bars, so it cannot be detected on "
                "its own pivot bar (ADR 0008 D3)"
            )
        if self.status is SwingStatus.CANDIDATE and self.label is not None:
            raise InvariantViolation("a candidate is not structure and carries no label")


@dataclasses.dataclass(frozen=True, slots=True)
class BreakEvent:
    """A confirmed break of a structural level.

    Immutable, like every domain event: a level that broke stays broken, and the
    level itself is consumed so it cannot break twice (ADR 0008 D7).
    """

    kind: BreakKind
    level_price: Decimal
    level_bar_start_utc: datetime
    bar_start_utc: datetime
    closes: int
    volume_confirmed: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "level_price", exact_price(self.level_price, field="level_price"))
        level_start = ensure_utc(self.level_bar_start_utc, field="level_bar_start_utc")
        break_start = ensure_utc(self.bar_start_utc, field="bar_start_utc")
        object.__setattr__(self, "level_bar_start_utc", level_start)
        object.__setattr__(self, "bar_start_utc", break_start)

        if break_start <= level_start:
            raise InvariantViolation(
                f"a break at {break_start.isoformat()} cannot precede or coincide with "
                f"the level it broke ({level_start.isoformat()})"
            )
        if self.closes < 1:
            raise InvariantViolation(f"closes: a break needs at least one, got {self.closes}")


@dataclasses.dataclass(frozen=True, slots=True)
class StructureUpdate:
    """What one bar changed.

    Returned for every bar, empty when nothing happened: for structure "no new
    swing" is an answer, not a warmup gap.
    """

    confirmed_swings: tuple[SwingPoint, ...] = ()
    breaks: tuple[BreakEvent, ...] = ()
    candidate_high: SwingPoint | None = None
    candidate_low: SwingPoint | None = None
