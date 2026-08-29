"""Alert rules, policies and events — specs/09-alerts.md.

The engine that evaluates these arrives in Phase 6. What lives here is the
vocabulary: what can be asked for, and what an emission looks like once it has
happened.

Two design points worth stating explicitly.

**Suppression is recorded, not swallowed.** A rule that fires while in cooldown
still produces an `AlertEvent`, carrying `EmissionStatus.SUPPRESSED_COOLDOWN`.
specs/01-product.md requires the alert center to show suppressed events, and an
event that was never written cannot be shown or audited.

**Acknowledgement is not a field on the event.** specs/04-domain-model.md lists
`status` on AlertEvent and then states that events are immutable facts while
acknowledgement is separate mutable state. Both hold once `status` is read as
*what happened at emission time* — which never changes — and acknowledgement
moves to its own `AlertAcknowledgement` record.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from types import MappingProxyType
from typing import Any, Final

from alphagate.core.errors import InvariantViolation
from alphagate.core.identifiers import AlertEventId, AlertRuleId, LevelId, Ticker, UserId
from alphagate.core.numeric import price as exact_price
from alphagate.core.time_model import Timeframe, ensure_utc
from alphagate.core.trend import TrendPhase

__all__ = [
    "AlertAcknowledgement",
    "AlertCondition",
    "AlertEvent",
    "AlertEventType",
    "AlertPolicy",
    "AlertRule",
    "AlertScope",
    "ConfirmationPolicy",
    "Cooldown",
    "EmissionStatus",
    "NotificationChannel",
    "ProximityKind",
    "ProximityThreshold",
    "RearmMode",
    "RearmPolicy",
]


class AlertEventType(Enum):
    """The domain events a rule can watch for.

    `LEVEL_BROKEN` from specs/02-architecture.md is split by direction here, as
    specs/09-alerts.md asks for "breaking upward" and "breaking downward"
    separately — a rule usually wants one and not the other.
    """

    LEVEL_APPROACHING = "LEVEL_APPROACHING"
    LEVEL_TOUCHED = "LEVEL_TOUCHED"
    LEVEL_BROKEN_UP = "LEVEL_BROKEN_UP"
    LEVEL_BROKEN_DOWN = "LEVEL_BROKEN_DOWN"
    LEVEL_ZONE_EXITED = "LEVEL_ZONE_EXITED"

    TREND_CONFIRMED = "TREND_CONFIRMED"
    TREND_REVERSAL = "TREND_REVERSAL"
    TREND_STRENGTH_INCREASED = "TREND_STRENGTH_INCREASED"
    TREND_STRENGTH_DECREASED = "TREND_STRENGTH_DECREASED"

    BREAKOUT_CONFIRMED = "BREAKOUT_CONFIRMED"
    BREAKOUT_FAILED = "BREAKOUT_FAILED"

    @property
    def is_level_event(self) -> bool:
        return self in _LEVEL_EVENTS

    @property
    def is_trend_event(self) -> bool:
        return self in _TREND_EVENTS


_LEVEL_EVENTS: Final = frozenset(
    {
        AlertEventType.LEVEL_APPROACHING,
        AlertEventType.LEVEL_TOUCHED,
        AlertEventType.LEVEL_BROKEN_UP,
        AlertEventType.LEVEL_BROKEN_DOWN,
        AlertEventType.LEVEL_ZONE_EXITED,
    }
)

_TREND_EVENTS: Final = frozenset(
    {
        AlertEventType.TREND_CONFIRMED,
        AlertEventType.TREND_REVERSAL,
        AlertEventType.TREND_STRENGTH_INCREASED,
        AlertEventType.TREND_STRENGTH_DECREASED,
    }
)

_STRENGTH_EVENTS: Final = frozenset(
    {
        AlertEventType.TREND_STRENGTH_INCREASED,
        AlertEventType.TREND_STRENGTH_DECREASED,
    }
)


class ProximityKind(Enum):
    """How "close" is measured (specs/09-alerts.md)."""

    ABSOLUTE = "ABSOLUTE"
    PERCENT = "PERCENT"
    ATR_MULTIPLE = "ATR_MULTIPLE"


@dataclasses.dataclass(frozen=True, slots=True)
class ProximityThreshold:
    """A distance, in whichever unit `kind` names.

    Exact-domain even for ATR multiples: the value is user configuration that
    must round-trip through storage and the API unchanged.
    """

    kind: ProximityKind
    value: Decimal

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", exact_price(self.value, field="threshold"))
        if self.value <= 0:
            raise InvariantViolation(f"threshold: must be positive, got {self.value}")


@dataclasses.dataclass(frozen=True, slots=True)
class Cooldown:
    """Silence window following a trigger.

    Prevents repeated notifications while the same condition stays true.
    """

    duration: timedelta

    def __post_init__(self) -> None:
        if self.duration <= timedelta(0):
            raise InvariantViolation(f"cooldown duration must be positive, got {self.duration}")

    def expires_at(self, triggered_at: datetime) -> datetime:
        return ensure_utc(triggered_at, field="triggered_at") + self.duration

    def is_active(self, *, triggered_at: datetime, now: datetime) -> bool:
        """Half-open, matching bar windows: the expiry instant is already clear."""
        return ensure_utc(now, field="now") < self.expires_at(triggered_at)


class RearmMode(Enum):
    ON_ZONE_EXIT = "ON_ZONE_EXIT"
    """Price leaves the zone by the hysteresis distance."""

    ON_STATE_CHANGE = "ON_STATE_CHANGE"
    """Trend returns to neutral, or to any other state."""

    AFTER_TIME = "AFTER_TIME"
    """A window passes and the condition has reset."""


@dataclasses.dataclass(frozen=True, slots=True)
class RearmPolicy:
    """When a fired rule becomes eligible to fire again."""

    mode: RearmMode
    hysteresis: ProximityThreshold | None = None
    window: timedelta | None = None

    def __post_init__(self) -> None:
        required, forbidden = _REARM_PARAMS[self.mode]
        for name in required:
            if getattr(self, name) is None:
                raise InvariantViolation(f"{self.mode.value} re-arm requires {name}")
        for name in forbidden:
            if getattr(self, name) is not None:
                raise InvariantViolation(f"{self.mode.value} re-arm does not use {name}")
        if self.window is not None and self.window <= timedelta(0):
            raise InvariantViolation(f"re-arm window must be positive, got {self.window}")


_REARM_PARAMS: Final[Mapping[RearmMode, tuple[tuple[str, ...], tuple[str, ...]]]] = (
    MappingProxyType(
        {
            RearmMode.ON_ZONE_EXIT: (("hysteresis",), ("window",)),
            RearmMode.ON_STATE_CHANGE: ((), ("hysteresis", "window")),
            RearmMode.AFTER_TIME: (("window",), ("hysteresis",)),
        }
    )
)


@dataclasses.dataclass(frozen=True, slots=True)
class ConfirmationPolicy:
    """How much evidence a condition needs before it counts.

    `allow_intrabar` defaults to False: confirming from a partial bar is
    look-ahead unless the profile asked for it (CLAUDE.md §12).
    """

    bars: int
    allow_intrabar: bool = False

    def __post_init__(self) -> None:
        if self.bars < 1:
            raise InvariantViolation(f"confirmation bars must be at least 1, got {self.bars}")


class NotificationChannel(Enum):
    """Where an alert can be sent.

    `DESKTOP` was here and is gone (`adr/0020-telegram-only.md`). Removing the
    member rather than retiring it means no stored row may still name it, which
    is why migration `c31d8b7f0a42` rewrites the rules and preferences that did
    and deletes the delivery rows: a value this enum cannot parse would fail on
    read rather than sit harmlessly.

    What a deployment can actually deliver on is `Services.available_channels` —
    the dispatchers that were wired — not this enum, which is what a rule is
    allowed to ask for.
    """

    TELEGRAM = "TELEGRAM"
    EMAIL = "EMAIL"
    WEBHOOK = "WEBHOOK"


@dataclasses.dataclass(frozen=True, slots=True)
class AlertScope:
    """What a rule watches."""

    symbol: Ticker
    timeframe: Timeframe | None = None
    level_id: LevelId | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class AlertCondition:
    """The thing being watched for, plus whatever parameter it needs."""

    event_type: AlertEventType
    proximity: ProximityThreshold | None = None
    strength_threshold: float | None = None

    def __post_init__(self) -> None:
        if self.event_type is AlertEventType.LEVEL_APPROACHING and self.proximity is None:
            raise InvariantViolation(
                "LEVEL_APPROACHING needs a proximity threshold — without one there is "
                "no definition of 'approaching'"
            )
        if self.event_type in _STRENGTH_EVENTS:
            if self.strength_threshold is None:
                raise InvariantViolation(
                    f"{self.event_type.value} needs a strength_threshold to cross"
                )
            if not 0.0 <= self.strength_threshold <= 100.0:
                raise InvariantViolation(
                    f"strength_threshold must be within [0, 100], got {self.strength_threshold}"
                )


@dataclasses.dataclass(frozen=True, slots=True)
class AlertPolicy:
    """Per-level alerting preferences attached to a user level.

    A compact subset of a rule: which events the user wants from this particular
    line, and how often. The AlertEngine expands it into rules.
    """

    event_types: frozenset[AlertEventType]
    proximity: ProximityThreshold | None = None
    cooldown: Cooldown | None = None

    def __post_init__(self) -> None:
        if not self.event_types:
            raise InvariantViolation("an alert policy needs at least one event type")
        if AlertEventType.LEVEL_APPROACHING in self.event_types and self.proximity is None:
            raise InvariantViolation("LEVEL_APPROACHING needs a proximity threshold")


@dataclasses.dataclass(frozen=True, slots=True)
class AlertRule:
    id: AlertRuleId
    user_id: UserId
    scope: AlertScope
    condition: AlertCondition
    confirmation: ConfirmationPolicy
    cooldown: Cooldown
    rearm: RearmPolicy
    channels: frozenset[NotificationChannel]
    enabled: bool = True

    def __post_init__(self) -> None:
        if not self.channels:
            raise InvariantViolation("a rule needs at least one notification channel")
        if self.condition.event_type.is_level_event and self.scope.level_id is None:
            raise InvariantViolation(
                f"{self.condition.event_type.value} is a level event; scope needs a level_id"
            )
        if self.condition.event_type.is_trend_event and self.scope.timeframe is None:
            raise InvariantViolation(
                f"{self.condition.event_type.value} is a trend event; scope needs a timeframe "
                "— trend has no meaning without one"
            )


class EmissionStatus(Enum):
    """What the AlertEngine did with a condition that became true.

    An immutable fact about the moment of emission, not a mutable workflow state.
    """

    TRIGGERED = "TRIGGERED"
    SUPPRESSED_COOLDOWN = "SUPPRESSED_COOLDOWN"
    SUPPRESSED_DUPLICATE = "SUPPRESSED_DUPLICATE"
    SUPPRESSED_DISARMED = "SUPPRESSED_DISARMED"


@dataclasses.dataclass(frozen=True, slots=True)
class AlertEvent:
    """An immutable record that a condition became true.

    Append-only (specs/12-storage.md). Nothing mutates an event after emission;
    later facts about it are separate records.
    """

    id: AlertEventId
    rule_id: AlertRuleId
    symbol: Ticker
    timeframe: Timeframe
    event_type: AlertEventType
    occurred_at: datetime
    price: Decimal
    reason_codes: tuple[str, ...]
    input_snapshot: Mapping[str, Any]
    status: EmissionStatus
    level_id: LevelId | None = None
    trend_state_before: TrendPhase | None = None
    trend_state_after: TrendPhase | None = None

    def __post_init__(self) -> None:
        if not self.rule_id:
            raise InvariantViolation("an event must reference the rule_id that caused it")
        object.__setattr__(self, "occurred_at", ensure_utc(self.occurred_at, field="occurred_at"))
        object.__setattr__(self, "price", exact_price(self.price, field="price"))
        if self.price <= 0:
            raise InvariantViolation(f"price: must be positive, got {self.price}")

        if not self.reason_codes:
            raise InvariantViolation(
                "reason_codes are required — every alert must be explainable in the UI "
                "(specs/01-product.md)"
            )

        if self.event_type.is_level_event and self.level_id is None:
            raise InvariantViolation(f"{self.event_type.value} must reference a level_id")
        if self.event_type.is_trend_event and (
            self.trend_state_before is None or self.trend_state_after is None
        ):
            raise InvariantViolation(
                f"{self.event_type.value} must carry trend_state_before and "
                "trend_state_after so the change is auditable"
            )

        # Copy: the caller's dict must not be able to rewrite history.
        object.__setattr__(self, "input_snapshot", MappingProxyType(dict(self.input_snapshot)))

    @property
    def was_delivered_to_channels(self) -> bool:
        """Whether this emission should reach the notification router.

        Suppressed events are recorded and shown in the alert center, but they
        are not notifications.
        """
        return self.status is EmissionStatus.TRIGGERED


@dataclasses.dataclass(frozen=True, slots=True)
class AlertAcknowledgement:
    """A user marking an event as seen.

    Separate from the event because events are immutable and acknowledgement is
    not (specs/04-domain-model.md).
    """

    event_id: AlertEventId
    user_id: UserId
    acknowledged_at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "acknowledged_at", ensure_utc(self.acknowledged_at, field="acknowledged_at")
        )
