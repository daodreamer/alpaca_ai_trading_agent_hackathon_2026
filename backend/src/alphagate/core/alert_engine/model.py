"""Alert-engine inputs, per-rule state and derived identity — ADR 0011.

The vocabulary an alert is expressed in — `AlertRule`, `Cooldown`, `RearmPolicy`,
`AlertEvent` — has lived in `alphagate.core.alerts` since Phase 1. What lives here is
what the engine needs in order to *evaluate* one: the observation it reads, the
level view it reads it against, the state it carries between bars, and the two
derivations that make an event's identity a function of its cause rather than of
when it happened to be created.
"""

from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Mapping
from datetime import datetime
from decimal import Decimal
from types import MappingProxyType
from typing import Any, Final, Self

from alphagate.core.alerts import AlertEventType, AlertRule
from alphagate.core.bar import Bar
from alphagate.core.errors import InvariantViolation
from alphagate.core.identifiers import AlertEventId, AlertRuleId, LevelId
from alphagate.core.levels import Level, LevelKind, UserLevel, Zone
from alphagate.core.numeric import ExactInput
from alphagate.core.numeric import price as exact_price
from alphagate.core.time_model import ensure_utc
from alphagate.core.trend import TrendState
from alphagate.core.trend_engine import TrendEvent

__all__ = [
    "USER_LEVEL_PREFIX",
    "AlertObservation",
    "ConditionReading",
    "RuleState",
    "WatchedLevel",
    "alert_event_id",
    "event_key",
    "rule_fingerprint",
    "user_level_id",
]

USER_LEVEL_PREFIX: Final = "user:"
"""Namespace for user levels inside the engine's level id space (ADR 0011 D11).

`Level` ids are derived from evidence and `UserLevel` ids are assigned by
persistence, so nothing stops the two spaces from colliding. This makes them
disjoint by construction rather than by hope.
"""


def user_level_id(raw: str) -> LevelId:
    """The engine-facing level id of a user level."""
    return LevelId(f"{USER_LEVEL_PREFIX}{raw}")


@dataclasses.dataclass(frozen=True, slots=True)
class WatchedLevel:
    """What the alert engine sees of a level, whatever produced it.

    Always carries a zone: an exact level is a degenerate zone at its own price,
    which lets every predicate be written once instead of once per shape
    (ADR 0011 D11).
    """

    id: LevelId
    kind: LevelKind
    price: Decimal
    zone: Zone

    def __post_init__(self) -> None:
        object.__setattr__(self, "price", exact_price(self.price, field="price"))
        if not self.zone.contains(self.price):
            raise InvariantViolation(
                f"{self.id}: price ({self.price}) must lie within its zone "
                f"[{self.zone.low}, {self.zone.high}]"
            )

    @classmethod
    def of_level(cls, level: Level) -> Self:
        return cls(id=level.id, kind=level.kind, price=level.price, zone=level.effective_zone)

    @classmethod
    def of_user_level(cls, level: UserLevel) -> Self:
        return cls(
            id=user_level_id(level.id),
            kind=level.kind,
            price=level.price,
            zone=level.effective_zone,
        )

    def contains(self, value: ExactInput) -> bool:
        return self.zone.contains(value)

    def distance_to(self, value: ExactInput) -> Decimal:
        return self.zone.distance_to(value)


@dataclasses.dataclass(frozen=True, slots=True)
class AlertObservation:
    """Everything the alert engine may look at, as of one bar (ADR 0011 D1).

    A snapshot of work already done: the levels in force, the trend as the trend
    engine reported it, the events that bar raised, and ATR. The alert engine
    computes none of it.
    """

    bar: Bar
    levels: tuple[WatchedLevel, ...] = ()
    trend: TrendState | None = None
    trend_events: tuple[TrendEvent, ...] = ()
    atr: Decimal | None = None

    def __post_init__(self) -> None:
        seen: set[LevelId] = set()
        for level in self.levels:
            if level.id in seen:
                raise InvariantViolation(
                    f"level id {level.id!r} appears twice in one observation; a rule "
                    "scoped to it could not say which one it meant"
                )
            seen.add(level.id)
        if self.atr is not None:
            object.__setattr__(self, "atr", exact_price(self.atr, field="atr"))

    def level(self, level_id: LevelId | None) -> WatchedLevel | None:
        """The watched level with this id, or `None` when it is not in force."""
        if level_id is None:
            return None
        return next((level for level in self.levels if level.id == level_id), None)


@dataclasses.dataclass(frozen=True, slots=True)
class ConditionReading:
    """One verdict on one condition, for one bar.

    `holds` is three-valued: `True`, `False`, and `None` for "not measurable"
    (ADR 0011 D2). A `None` moves no rule state, because a rule that forgot what
    it was counting every time ATR was cold would never confirm anything.
    """

    holds: bool | None
    reason_codes: tuple[str, ...] = ()
    level: WatchedLevel | None = None
    trend_event: TrendEvent | None = None
    detail: Mapping[str, Any] = dataclasses.field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        object.__setattr__(self, "detail", MappingProxyType(dict(self.detail)))


@dataclasses.dataclass(frozen=True, slots=True)
class RuleState:
    """What one rule remembers between bars.

    Frozen and replaced rather than mutated: a rule's state is a value, and
    replacing it makes every transition visible at the one call site that
    decides it.
    """

    consecutive: int = 0
    """Bars the condition has held. The emission edge is when it *reaches*
    `confirmation.bars`, which is what stops a persisting condition from
    alerting every bar (ADR 0011 D3)."""

    armed: bool = True
    triggered_at: datetime | None = None
    reset_seen: bool = False
    """Whether the condition has read `False` since the last trigger."""

    inside_zone: bool = False
    """Whether the previous evaluation's close sat inside the watched zone.
    Only `LEVEL_ZONE_EXITED` reads it, and it is the only history it needs."""

    def __post_init__(self) -> None:
        if self.consecutive < 0:
            raise InvariantViolation(f"consecutive: must not be negative, got {self.consecutive}")
        if self.triggered_at is not None:
            object.__setattr__(
                self, "triggered_at", ensure_utc(self.triggered_at, field="triggered_at")
            )


def event_key(rule_id: AlertRuleId, event_type: AlertEventType, bar_start_utc: datetime) -> str:
    """The identity of an emission, before it is told apart from its duplicates.

    Two emissions sharing a key are the same alert reported twice — which is
    reachable, because ADR 0010 D6 lets one bar walk several rungs of the trend
    ladder and raise the same kind of event more than once.
    """
    return f"{rule_id}|{event_type.value}|{bar_start_utc.isoformat()}"


def alert_event_id(key: str, ordinal: int) -> AlertEventId:
    """Derived, so a rebuild produces the same ids (ADR 0011 D7).

    The ordinal exists so that a duplicate is still storable: an append-only log
    cannot record a second event under a colliding id.
    """
    if ordinal < 0:
        raise InvariantViolation(f"ordinal: must not be negative, got {ordinal}")
    return AlertEventId(f"{key}#{ordinal}")


def rule_fingerprint(rule: AlertRule) -> str:
    """A hash of everything about a rule that changes what it fires on.

    `specs/09-alerts.md` requires an event to carry a "configuration/profile id".
    With no profile table yet, the honest answer is a fingerprint of the rule as
    it stood: it answers "was this the same rule when it fired?" across an edit,
    which is the question the id was for.

    Deliberately excludes `enabled` and `channels`: neither changes whether the
    condition became true, and a routing change should not make history look
    like it came from a different rule.
    """
    condition = rule.condition
    parts = [
        f"scope={rule.scope.symbol}/{rule.scope.timeframe.code if rule.scope.timeframe else ''}"
        f"/{rule.scope.level_id or ''}",
        f"event={condition.event_type.value}",
        "proximity="
        + (
            f"{condition.proximity.kind.value}:{condition.proximity.value}"
            if condition.proximity is not None
            else ""
        ),
        f"strength={condition.strength_threshold}",
        f"confirm={rule.confirmation.bars}/{rule.confirmation.allow_intrabar}",
        f"cooldown={rule.cooldown.duration.total_seconds()}",
        "rearm="
        + f"{rule.rearm.mode.value}:"
        + (
            f"{rule.rearm.hysteresis.kind.value}:{rule.rearm.hysteresis.value}"
            if rule.rearm.hysteresis is not None
            else ""
        )
        + f":{rule.rearm.window.total_seconds() if rule.rearm.window is not None else ''}",
    ]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
