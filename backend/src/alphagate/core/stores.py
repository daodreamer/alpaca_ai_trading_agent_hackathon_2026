"""Persistence ports for user-owned configuration — specs/12-storage.md.

`UserLevelStore` already lives in `alphagate.core.level_store`; these are the rest of
the shapes a use case needs to read and write without importing PostgreSQL.

They are deliberately narrow. A repository can do more — joins, bulk selects,
whatever a report needs — and a port that exposed all of it would make every
in-memory implementation a database emulator. What is here is what the monitor,
the router and the API actually call.

`NotificationPreferences` is the one *entity* in this module rather than a port.
`specs/13-notifications.md` lists per-user notification settings and no spec
models them, so this is the minimum shape those settings need: which channels are
on, what priority is worth interrupting for, and when not to interrupt at all.
"""

from __future__ import annotations

import dataclasses
from datetime import time, timedelta
from typing import Protocol

from alphagate.core.accounts import Watchlist
from alphagate.core.alerts import AlertAcknowledgement, AlertRule, NotificationChannel
from alphagate.core.errors import InvariantViolation
from alphagate.core.identifiers import AlertEventId, AlertRuleId, UserId, WatchlistId
from alphagate.core.levels import Priority
from alphagate.core.operations import NotificationDelivery

__all__ = [
    "AcknowledgementStore",
    "AlertRuleStore",
    "DeliveryStore",
    "NotificationPreferences",
    "PreferencesStore",
    "WatchlistStore",
]


class AlertRuleStore(Protocol):
    def upsert(self, rule: AlertRule) -> None: ...

    def delete(self, rule_id: AlertRuleId) -> None: ...

    def get(self, rule_id: AlertRuleId) -> AlertRule | None: ...

    def for_user(self, user_id: UserId) -> tuple[AlertRule, ...]: ...

    def enabled_rules(self) -> tuple[AlertRule, ...]:
        """Every rule the monitor should evaluate.

        Not filtered by user: the monitor runs for the deployment, and a rule
        belonging to nobody who is currently looking at a screen still fires."""
        ...


class AcknowledgementStore(Protocol):
    """Who has read which alert.

    Separate from the event store because an acknowledgement is a *later* fact
    about an event, and alert events are append-only: marking one as read must
    not rewrite it (`specs/04-domain-model.md`).
    """

    def acknowledge(self, acknowledgement: AlertAcknowledgement) -> None: ...

    def get(self, event_id: AlertEventId) -> AlertAcknowledgement | None: ...


class WatchlistStore(Protocol):
    def save(self, watchlist: Watchlist) -> None: ...

    def get(self, watchlist_id: WatchlistId) -> Watchlist | None: ...

    def for_user(self, user_id: UserId) -> tuple[Watchlist, ...]: ...


class DeliveryStore(Protocol):
    """Where notification attempts are recorded.

    Identity is `(event_id, channel)`, which is the idempotency key
    `specs/02-architecture.md` requires. A retry updates the record; it does not
    create a second one."""

    def record(self, delivery: NotificationDelivery) -> None: ...

    def get(
        self, event_id: AlertEventId, channel: NotificationChannel
    ) -> NotificationDelivery | None: ...

    def for_event(self, event_id: AlertEventId) -> tuple[NotificationDelivery, ...]: ...


@dataclasses.dataclass(frozen=True, slots=True)
class NotificationPreferences:
    """Who gets told what, and when not to be told at all."""

    user_id: UserId
    channels: frozenset[NotificationChannel] = frozenset({NotificationChannel.TELEGRAM})
    minimum_priority: Priority = Priority.LOW
    """Below this, an event is recorded and shown in the alert center but does
    not interrupt. The alert center is not a channel — it is the record."""

    quiet_from: time | None = None
    quiet_until: time | None = None
    """A local-time window during which nothing is delivered. Stored as wall
    times rather than instants because "not after 10pm" is a daily rule, not a
    one-off. Both or neither."""

    quiet_offset: timedelta = timedelta(0)
    """The user's offset from UTC, applied when reading the quiet window.

    Explicit rather than a timezone name: the MVP is single-user and a fixed
    offset is honest about what it does. A named zone with DST is a settings
    feature, not a notification one."""

    def __post_init__(self) -> None:
        if (self.quiet_from is None) != (self.quiet_until is None):
            raise InvariantViolation(
                "a quiet window needs both quiet_from and quiet_until, or neither"
            )

    @property
    def has_quiet_hours(self) -> bool:
        return self.quiet_from is not None and self.quiet_until is not None


class PreferencesStore(Protocol):
    def get(self, user_id: UserId) -> NotificationPreferences | None: ...

    def save(self, preferences: NotificationPreferences) -> None: ...
