"""Price levels and zones — specs/04-domain-model.md and specs/07-levels.md.

A `Level` is technical evidence at a price, not advice. `strength` ranks how
much evidence supports it; it is never a signal to buy or sell.

Upstream this module also carried `UserLevel` — a line a human drew on a chart,
with a priority and an alert policy. AlphaGate has no human in the loop and no
chart to draw on, so it was pruned with the rest of the monitoring product
(adr/0001 D5). Every level here is evidence the engine derived and may
invalidate on its own.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime
from decimal import Decimal
from enum import Enum

from alphagate.core.errors import InvariantViolation
from alphagate.core.identifiers import LevelId, Ticker
from alphagate.core.numeric import DOMAIN_CONTEXT, ExactInput
from alphagate.core.numeric import price as exact_price
from alphagate.core.time_model import Timeframe, ensure_utc

__all__ = [
    "Level",
    "LevelKind",
    "LevelSource",
    "LevelStatus",
    "Zone",
]


class LevelKind(Enum):
    SUPPORT = "SUPPORT"
    RESISTANCE = "RESISTANCE"


class LevelSource(Enum):
    """Where the evidence for a level came from (specs/07-levels.md)."""

    SWING_HIGH = "SWING_HIGH"
    SWING_LOW = "SWING_LOW"
    PREV_DAY_HIGH = "PREV_DAY_HIGH"
    PREV_DAY_LOW = "PREV_DAY_LOW"
    PREV_WEEK_HIGH = "PREV_WEEK_HIGH"
    PREV_WEEK_LOW = "PREV_WEEK_LOW"
    YEAR_HIGH = "YEAR_HIGH"
    YEAR_LOW = "YEAR_LOW"
    MOVING_AVERAGE = "MOVING_AVERAGE"
    USER = "USER"

    # -- F2, advanced levels (ADR 0025) --------------------------------------
    VOLUME_POC = "VOLUME_POC"
    """Point of control: the price the most volume traded at."""

    VOLUME_HVN = "VOLUME_HVN"
    """A high-volume node other than the point of control."""

    VOLUME_LVN = "VOLUME_LVN"
    """A low-volume node — a price band the market crossed quickly."""

    ANCHORED_VWAP = "ANCHORED_VWAP"
    FIBONACCI = "FIBONACCI"
    GAP = "GAP"


class LevelStatus(Enum):
    CANDIDATE = "CANDIDATE"
    """Detected but not yet confirmed by a finalized bar.

    Kept distinct from ACTIVE so that a structure awaiting confirmation can
    never be consumed as though it were established (CLAUDE.md §12).
    """

    ACTIVE = "ACTIVE"
    INVALIDATED = "INVALIDATED"


@dataclasses.dataclass(frozen=True, slots=True)
class Zone:
    """A closed price band, `[low, high]`.

    Closed rather than half-open, unlike a bar's time window: a touch of the
    exact zone edge is a touch, and excluding it would make `LEVEL_TOUCHED`
    depend on floating-point luck.
    """

    low: Decimal
    high: Decimal

    def __post_init__(self) -> None:
        object.__setattr__(self, "low", exact_price(self.low, field="zone_low"))
        object.__setattr__(self, "high", exact_price(self.high, field="zone_high"))
        if self.low <= 0:
            raise InvariantViolation(f"zone_low: must be positive, got {self.low}")
        if self.low > self.high:
            raise InvariantViolation(
                f"zone_low ({self.low}) must not exceed zone_high ({self.high})"
            )

    @property
    def width(self) -> Decimal:
        return self.high - self.low

    @property
    def midpoint(self) -> Decimal:
        # Divide in the explicit domain context so the result never depends on
        # ambient decimal state (CLAUDE.md §11).
        midpoint = DOMAIN_CONTEXT.divide(self.low + self.high, Decimal(2))
        return exact_price(midpoint, field="zone_midpoint")

    def contains(self, value: ExactInput) -> bool:
        return self.low <= exact_price(value, field="value") <= self.high

    def distance_to(self, value: ExactInput) -> Decimal:
        """Distance from `value` to the nearest edge; zero when inside."""
        target = exact_price(value, field="value")
        if target < self.low:
            return self.low - target
        if target > self.high:
            return target - self.high
        return exact_price(0)


def _check_price_within_zone(price_value: Decimal, zone: Zone | None) -> None:
    if zone is not None and not zone.contains(price_value):
        raise InvariantViolation(
            f"price ({price_value}) must lie within its zone "
            f"[{zone.low}, {zone.high}] (specs/04-domain-model.md)"
        )


def _check_score(value: float, *, field: str, upper: float) -> None:
    if not 0.0 <= value <= upper:
        raise InvariantViolation(f"{field}: must be within [0, {upper}], got {value}")


@dataclasses.dataclass(frozen=True, slots=True)
class Level:
    """A system-generated level."""

    id: LevelId
    symbol: Ticker
    timeframe: Timeframe
    kind: LevelKind
    price: Decimal
    zone: Zone | None
    sources: frozenset[LevelSource]
    strength: float
    confidence: float
    status: LevelStatus
    created_at: datetime
    updated_at: datetime
    strength_breakdown: tuple[tuple[str, float], ...] = ()
    """Per-component contributions to `strength`, summing to it (ADR 0009 D6).

    Ordered pairs rather than a mapping so the level stays frozen and hashable,
    and so the order the components are explained in is itself deterministic.
    Empty for a level built without scoring.
    """

    flipped_from: LevelId | None = None
    """The level this one is the other side of, when polarity flip is on.

    A broken resistance that comes back as support is a *new* level under the
    derived-identity rule — its kind is part of its id (ADR 0009 D2) — so the
    continuity `specs/07-levels.md` asks for has to be carried explicitly. This
    is that link, and it is the only thing connecting the two (ADR 0025 D5).
    """

    def __post_init__(self) -> None:
        object.__setattr__(self, "price", exact_price(self.price, field="price"))
        if self.price <= 0:
            raise InvariantViolation(f"price: must be positive, got {self.price}")
        _check_price_within_zone(self.price, self.zone)

        if not self.sources:
            raise InvariantViolation("a level needs at least one source")

        _check_score(self.strength, field="strength", upper=100.0)
        _check_score(self.confidence, field="confidence", upper=1.0)

        created = ensure_utc(self.created_at, field="created_at")
        updated = ensure_utc(self.updated_at, field="updated_at")
        object.__setattr__(self, "created_at", created)
        object.__setattr__(self, "updated_at", updated)
        if updated < created:
            raise InvariantViolation(
                f"updated_at ({updated.isoformat()}) must not precede "
                f"created_at ({created.isoformat()})"
            )

    @property
    def effective_zone(self) -> Zone:
        """The zone, or a degenerate zone at the price when none was assigned.

        Lets proximity logic treat exact levels and bands uniformly instead of
        branching on `zone is None` at every call site.
        """
        return self.zone if self.zone is not None else Zone(low=self.price, high=self.price)

    def distance_to(self, value: ExactInput) -> Decimal:
        """Distance from `value` to the nearest edge of the effective zone."""
        return self.effective_zone.distance_to(value)
