"""The alert event persistence port — ADR 0011 D12, specs/12-storage.md.

Declared in the Domain because it is part of a contract the Domain states:
`AlertEvent` is append-only, and "append-only" is a property of whatever holds
the events, not of the dataclass. A caller must be able to depend on that
without importing an infrastructure module.

The engine does not hold a store. It returns events; something above it decides
where they go (CLAUDE.md §8).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, runtime_checkable

from alphagate.core.alerts import AlertEvent
from alphagate.core.identifiers import AlertRuleId

__all__ = ["AlertEventStore", "AlertJournal"]


@runtime_checkable
class AlertEventStore(Protocol):
    """Append-only storage for emitted alert events.

    Implementations must be idempotent on identity: appending an event that is
    already stored, unchanged, is a no-op — which is what lets a deterministic
    replay be run twice. Appending a *different* event under an id that already
    exists is an `InvariantViolation`, because that would rewrite a fact.
    """

    def append(self, event: AlertEvent) -> None: ...

    def extend(self, events: Iterable[AlertEvent]) -> None: ...

    def all_events(self) -> tuple[AlertEvent, ...]:
        """Every stored event, in insertion order."""
        ...

    def events_for(self, rule_id: AlertRuleId) -> tuple[AlertEvent, ...]: ...

    def delivered(self) -> tuple[AlertEvent, ...]:
        """Only the events that reached the notification router."""
        ...


@runtime_checkable
class AlertJournal(AlertEventStore, Protocol):
    """A store that can also answer what each rule last fired.

    Separate from `AlertEventStore` because the two are needed by different
    callers for different reasons: the alert engine's output goes to a store,
    while restoring a rule's gates after a restart (ADR 0012 D6) is a *read*, and
    a component that only appends should not have to implement it.
    """

    def last_triggered_per_rule(self) -> dict[AlertRuleId, AlertEvent]:
        """The most recent `TRIGGERED` event for each rule that has ever fired.

        What `pmm.application.recovery` turns back into cooldown state. Rules
        that have never triggered are absent rather than mapped to `None`: a
        fresh `RuleState` is already the right answer for them.
        """
        ...
