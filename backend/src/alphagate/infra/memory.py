"""In-memory alert event storage — ADR 0011 D12.

Satisfies `alphagate.core.alert_engine.AlertEventStore` without a database, which is
what Phase 6 needs: the roadmap asks for the *abstraction*, and PostgreSQL is
Phase 7. It is also what replay and integration tests store into, so the append
rules below are exercised on every fixture run rather than only in Phase 7.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime

from alphagate.core.accounts import Watchlist
from alphagate.core.alerts import (
    AlertAcknowledgement,
    AlertEvent,
    AlertRule,
    NotificationChannel,
)
from alphagate.core.bar import AdjustmentMode, Bar, Feed
from alphagate.core.errors import InvariantViolation
from alphagate.core.identifiers import (
    AlertEventId,
    AlertRuleId,
    Ticker,
    UserId,
    UserLevelId,
    WatchlistId,
)
from alphagate.core.levels import UserLevel
from alphagate.core.operations import NotificationDelivery
from alphagate.core.stores import NotificationPreferences
from alphagate.core.time_model import Timeframe, ensure_utc

__all__ = [
    "InMemoryAcknowledgementStore",
    "InMemoryAlertEventStore",
    "InMemoryAlertRuleStore",
    "InMemoryBarStore",
    "InMemoryDeliveryStore",
    "InMemoryPreferencesStore",
    "InMemoryUserLevelStore",
    "InMemoryWatchlistStore",
]


class InMemoryAlertEventStore:
    """Append-only, ordered, and idempotent on identity."""

    def __init__(self) -> None:
        self._events: list[AlertEvent] = []
        self._by_id: dict[AlertEventId, AlertEvent] = {}

    def __len__(self) -> int:
        return len(self._events)

    def __repr__(self) -> str:
        return f"<InMemoryAlertEventStore events={len(self._events)}>"

    def append(self, event: AlertEvent) -> None:
        """Store one event.

        Re-appending an identical event is a no-op — event ids are derived from
        the bar and the rule (ADR 0011 D7), so a replay of the same bars is
        expected to produce them again. Anything *different* under the same id
        would be a rewritten fact, and alert events are append-only
        (specs/12-storage.md).
        """
        existing = self._by_id.get(event.id)
        if existing is not None:
            if existing != event:
                raise InvariantViolation(
                    f"{event.id}: alert events are append-only; an event already exists "
                    "under this id and differs from the one being stored"
                )
            return
        self._by_id[event.id] = event
        self._events.append(event)

    def extend(self, events: Iterable[AlertEvent]) -> None:
        for event in events:
            self.append(event)

    def all_events(self) -> tuple[AlertEvent, ...]:
        return tuple(self._events)

    def events_for(self, rule_id: AlertRuleId) -> tuple[AlertEvent, ...]:
        return tuple(event for event in self._events if event.rule_id == rule_id)

    def delivered(self) -> tuple[AlertEvent, ...]:
        """Only what reached the notification router.

        The alert center shows suppressed events too; this is the other view.
        """
        return tuple(event for event in self._events if event.was_delivered_to_channels)

    def last_triggered_per_rule(self) -> dict[AlertRuleId, AlertEvent]:
        """The most recent TRIGGERED event per rule (ADR 0012 D6).

        Insertion order is emission order, so the last one wins by overwrite —
        no comparison, and therefore no way for two events at the same instant
        to swap places between runs.
        """
        latest: dict[AlertRuleId, AlertEvent] = {}
        for event in self._events:
            if event.was_delivered_to_channels:
                latest[event.rule_id] = event
        return latest


class InMemoryBarStore:
    """Satisfies `alphagate.core.market_data.BarStore` without a database.

    The bar cache (ADR 0022) is worth having even with no `PMM_DATABASE_URL`: a
    chart redrawn ten times in a session should cost the provider one backfill,
    and that is true whether or not the bars outlive the process.

    Keyed by the whole series identity, and the revision rule is the one the
    database enforces with the `where` clause on its upsert — written twice
    because the two stores must not disagree about which copy of a bar wins.
    """

    def __init__(self) -> None:
        self._bars: dict[tuple[Ticker, str, str, str], dict[datetime, Bar]] = {}

    def __len__(self) -> int:
        return sum(len(series) for series in self._bars.values())

    def __repr__(self) -> str:
        return f"<InMemoryBarStore series={len(self._bars)} bars={len(self)}>"

    def get_range(
        self,
        *,
        symbol: Ticker,
        timeframe: Timeframe,
        feed: Feed,
        adjustment_mode: AdjustmentMode,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> tuple[Bar, ...]:
        series = self._bars.get(_series_key(symbol, timeframe, feed, adjustment_mode), {})
        lower = ensure_utc(start, field="start") if start is not None else None
        upper = ensure_utc(end, field="end") if end is not None else None
        return tuple(
            sorted(
                (
                    bar
                    for instant, bar in series.items()
                    if (lower is None or instant >= lower) and (upper is None or instant < upper)
                ),
                key=lambda bar: bar.start_time_utc,
            )
        )

    def upsert_many(self, incoming: Iterable[Bar]) -> int:
        written = 0
        for bar in incoming:
            key = _series_key(bar.symbol, bar.timeframe, bar.feed, bar.adjustment_mode)
            series = self._bars.setdefault(key, {})
            held = series.get(bar.start_time_utc)
            if held is not None and bar.revision < held.revision:
                continue
            series[bar.start_time_utc] = bar
            written += 1
        return written


def _series_key(
    symbol: Ticker, timeframe: Timeframe, feed: Feed, adjustment_mode: AdjustmentMode
) -> tuple[Ticker, str, str, str]:
    return (symbol, timeframe.code, feed.value, adjustment_mode.value)


class InMemoryUserLevelStore:
    """Satisfies `alphagate.core.level_store.UserLevelStore` without a database.

    What the API falls back to when no `PMM_DATABASE_URL` is configured, so the
    chart can be developed and demonstrated without standing up PostgreSQL.

    It is honest about what it is: a level survives a page reload, because the
    process outlives the page, but it does not survive a restart. That is the
    difference `specs/12-storage.md` exists to remove, and the composition root
    warns when this is the store in use.
    """

    def __init__(self) -> None:
        self._levels: dict[UserLevelId, UserLevel] = {}

    def __len__(self) -> int:
        return len(self._levels)

    def __repr__(self) -> str:
        return f"<InMemoryUserLevelStore levels={len(self._levels)}>"

    def upsert(self, level: UserLevel) -> None:
        self._levels[level.id] = level

    def delete(self, level_id: UserLevelId) -> None:
        self._levels.pop(level_id, None)

    def get(self, level_id: UserLevelId) -> UserLevel | None:
        return self._levels.get(level_id)

    def for_user(self, user_id: UserId) -> tuple[UserLevel, ...]:
        return self._ordered(level for level in self._levels.values() if level.user_id == user_id)

    def for_symbol(self, user_id: UserId, symbol: Ticker) -> tuple[UserLevel, ...]:
        return self._ordered(
            level
            for level in self._levels.values()
            if level.user_id == user_id and level.symbol == symbol
        )

    @staticmethod
    def _ordered(levels: Iterable[UserLevel]) -> tuple[UserLevel, ...]:
        """By id, matching what the repository's `ORDER BY id` returns.

        Two stores that answer the same question in different orders would make
        the API's output depend on which one is wired in, and a chart test would
        pass against one and fail against the other.
        """
        return tuple(sorted(levels, key=lambda level: level.id))


class InMemoryAlertRuleStore:
    """Satisfies `alphagate.core.stores.AlertRuleStore` without a database."""

    def __init__(self) -> None:
        self._rules: dict[AlertRuleId, AlertRule] = {}

    def __len__(self) -> int:
        return len(self._rules)

    def upsert(self, rule: AlertRule) -> None:
        self._rules[rule.id] = rule

    def delete(self, rule_id: AlertRuleId) -> None:
        self._rules.pop(rule_id, None)

    def get(self, rule_id: AlertRuleId) -> AlertRule | None:
        return self._rules.get(rule_id)

    def for_user(self, user_id: UserId) -> tuple[AlertRule, ...]:
        return self._ordered(rule for rule in self._rules.values() if rule.user_id == user_id)

    def enabled_rules(self) -> tuple[AlertRule, ...]:
        return self._ordered(rule for rule in self._rules.values() if rule.enabled)

    @staticmethod
    def _ordered(rules: Iterable[AlertRule]) -> tuple[AlertRule, ...]:
        return tuple(sorted(rules, key=lambda rule: rule.id))


class InMemoryWatchlistStore:
    """Satisfies `alphagate.core.stores.WatchlistStore` without a database."""

    def __init__(self) -> None:
        self._lists: dict[WatchlistId, Watchlist] = {}

    def save(self, watchlist: Watchlist) -> None:
        self._lists[watchlist.id] = watchlist

    def get(self, watchlist_id: WatchlistId) -> Watchlist | None:
        return self._lists.get(watchlist_id)

    def for_user(self, user_id: UserId) -> tuple[Watchlist, ...]:
        return tuple(
            sorted(
                (item for item in self._lists.values() if item.user_id == user_id),
                key=lambda item: item.id,
            )
        )


class InMemoryDeliveryStore:
    """Satisfies `alphagate.core.stores.DeliveryStore` without a database.

    Keyed by `(event_id, channel)` — the idempotency key from
    `specs/02-architecture.md`. A retry replaces the record; it never adds one.
    """

    def __init__(self) -> None:
        self._deliveries: dict[tuple[AlertEventId, NotificationChannel], NotificationDelivery] = {}

    def __len__(self) -> int:
        return len(self._deliveries)

    def record(self, delivery: NotificationDelivery) -> None:
        self._deliveries[(delivery.event_id, delivery.channel)] = delivery

    def get(
        self, event_id: AlertEventId, channel: NotificationChannel
    ) -> NotificationDelivery | None:
        return self._deliveries.get((event_id, channel))

    def for_event(self, event_id: AlertEventId) -> tuple[NotificationDelivery, ...]:
        return tuple(
            sorted(
                (
                    delivery
                    for (stored, _), delivery in self._deliveries.items()
                    if stored == event_id
                ),
                key=lambda delivery: delivery.channel.value,
            )
        )


class InMemoryPreferencesStore:
    """Satisfies `alphagate.core.stores.PreferencesStore` without a database."""

    def __init__(self) -> None:
        self._preferences: dict[UserId, NotificationPreferences] = {}

    def get(self, user_id: UserId) -> NotificationPreferences | None:
        return self._preferences.get(user_id)

    def save(self, preferences: NotificationPreferences) -> None:
        self._preferences[preferences.user_id] = preferences


class InMemoryAcknowledgementStore:
    """Satisfies `alphagate.core.stores.AcknowledgementStore` without a database.

    Acknowledging twice keeps the first time: when the user read it is the fact,
    and re-reading it does not change when they first did.
    """

    def __init__(self) -> None:
        self._acknowledgements: dict[AlertEventId, AlertAcknowledgement] = {}

    def acknowledge(self, acknowledgement: AlertAcknowledgement) -> None:
        self._acknowledgements.setdefault(acknowledgement.event_id, acknowledgement)

    def get(self, event_id: AlertEventId) -> AlertAcknowledgement | None:
        return self._acknowledgements.get(event_id)

    def all_acknowledgements(self) -> tuple[AlertAcknowledgement, ...]:
        return tuple(sorted(self._acknowledgements.values(), key=lambda item: item.acknowledged_at))
