"""Alert evaluation — specs/09-alerts.md, ADR 0011.

Alerting is an event state machine. A condition becoming true is necessary and
not sufficient: confirmation decides whether it counts, cooldown and re-arm
decide whether it is heard, and every one of those decisions is written down as
an `AlertEvent` so the alert center can answer "why didn't I get an alert?".

The engine emits events and nothing else. Mapping an event onto a channel is the
notification router's job (CLAUDE.md §8/§9), and persisting one is a port.
"""

from __future__ import annotations

from alphagate.core.alert_engine.conditions import (
    is_discrete,
    is_supported,
    read_conditions,
    resolve_threshold,
)
from alphagate.core.alert_engine.engine import AlertEngine
from alphagate.core.alert_engine.model import (
    USER_LEVEL_PREFIX,
    AlertObservation,
    ConditionReading,
    RuleState,
    WatchedLevel,
    alert_event_id,
    event_key,
    rule_fingerprint,
    user_level_id,
)
from alphagate.core.alert_engine.policy import expand_policy
from alphagate.core.alert_engine.store import AlertEventStore, AlertJournal

__all__ = [
    "USER_LEVEL_PREFIX",
    "AlertEngine",
    "AlertEventStore",
    "AlertJournal",
    "AlertObservation",
    "ConditionReading",
    "RuleState",
    "WatchedLevel",
    "alert_event_id",
    "event_key",
    "expand_policy",
    "is_discrete",
    "is_supported",
    "read_conditions",
    "resolve_threshold",
    "rule_fingerprint",
    "user_level_id",
]
