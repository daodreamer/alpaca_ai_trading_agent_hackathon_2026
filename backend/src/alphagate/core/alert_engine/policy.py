"""Expanding a user level's `AlertPolicy` into rules — ADR 0011 D11.

`AlertPolicy` is the compact thing a user attaches to a line they drew: which
events they want from it, how close counts as close, how often. A rule is the
thing the engine evaluates. This is the one-way function between them.

One rule per event type, so each carries its own cooldown and re-arm state.
Grouping several event types under one cooldown is `specs/09-alerts.md`'s
"alert groups", which is Full-version work.
"""

from __future__ import annotations

from alphagate.core.alert_engine.model import user_level_id
from alphagate.core.alerts import (
    AlertCondition,
    AlertEventType,
    AlertPolicy,
    AlertRule,
    AlertScope,
    ConfirmationPolicy,
    Cooldown,
    NotificationChannel,
    RearmPolicy,
)
from alphagate.core.errors import InvariantViolation
from alphagate.core.identifiers import AlertRuleId
from alphagate.core.levels import UserLevel

__all__ = ["expand_policy"]


def expand_policy(
    level: UserLevel,
    *,
    confirmation: ConfirmationPolicy,
    rearm: RearmPolicy,
    channels: frozenset[NotificationChannel],
    cooldown: Cooldown | None = None,
) -> tuple[AlertRule, ...]:
    """The rules a user level's alert policy asks for, ordered by id.

    `cooldown` is the fallback when the policy does not name one. Neither being
    present is an error rather than a default: a silence window nobody chose is
    a hidden constant, and this codebase does not keep those.
    """
    policy = level.alert_policy
    if policy is None:
        raise InvariantViolation(
            f"{level.id}: no alert_policy to expand; a level with no policy raises "
            "no alerts, which is a valid state and not a rule set"
        )

    effective_cooldown = policy.cooldown if policy.cooldown is not None else cooldown
    if effective_cooldown is None:
        raise InvariantViolation(
            f"{level.id}: no cooldown — neither the policy nor the caller supplied one, "
            "and inventing a silence window would hide a decision"
        )

    scope = AlertScope(
        symbol=level.symbol,
        timeframe=level.timeframe,
        level_id=user_level_id(level.id),
    )

    rules = [
        AlertRule(
            id=AlertRuleId(f"{user_level_id(level.id)}:{event_type.value}"),
            user_id=level.user_id,
            scope=scope,
            condition=AlertCondition(event_type=event_type, proximity=policy.proximity),
            confirmation=confirmation,
            cooldown=effective_cooldown,
            rearm=rearm,
            channels=channels,
            enabled=level.active,
        )
        for event_type in _level_events(level, policy)
    ]
    return tuple(sorted(rules, key=lambda rule: rule.id))


def _level_events(level: UserLevel, policy: AlertPolicy) -> tuple[AlertEventType, ...]:
    non_level = sorted(
        event_type.value for event_type in policy.event_types if not event_type.is_level_event
    )
    if non_level:
        raise InvariantViolation(
            f"{level.id}: {non_level} are not level events; a policy attached to a "
            "level cannot ask for a trend alert"
        )
    return tuple(sorted(policy.event_types, key=lambda event_type: event_type.value))
