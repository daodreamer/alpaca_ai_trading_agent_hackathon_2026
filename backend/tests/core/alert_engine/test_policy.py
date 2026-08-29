"""Expanding a user level's `AlertPolicy` into rules — ADR 0011 D11."""

from __future__ import annotations

from datetime import timedelta

import pytest

from alphagate.core.alert_engine import expand_policy
from alphagate.core.alerts import (
    AlertEventType,
    AlertPolicy,
    ConfirmationPolicy,
    Cooldown,
    NotificationChannel,
    RearmMode,
    RearmPolicy,
)
from alphagate.core.errors import InvariantViolation
from alphagate.core.levels import LevelKind, UserLevel
from tests.core.alert_engine.synthetic import AAPL, USER, absolute

CHANNELS = frozenset({NotificationChannel.TELEGRAM})
CONFIRMATION = ConfirmationPolicy(bars=1)
REARM = RearmPolicy(mode=RearmMode.ON_STATE_CHANGE)


def user_level(policy: AlertPolicy | None) -> UserLevel:
    return UserLevel(
        id="ul-1",
        user_id=USER,
        symbol=AAPL,
        kind=LevelKind.RESISTANCE,
        price="103.20",
        alert_policy=policy,
    )


def expand(policy: AlertPolicy, **overrides: object) -> tuple[object, ...]:
    kwargs: dict[str, object] = {
        "confirmation": CONFIRMATION,
        "rearm": REARM,
        "channels": CHANNELS,
        "cooldown": Cooldown(timedelta(minutes=15)),
    }
    return expand_policy(user_level(policy), **{**kwargs, **overrides})  # type: ignore[arg-type]


class TestExpansion:
    def test_one_rule_per_event_type(self) -> None:
        policy = AlertPolicy(
            event_types=frozenset({AlertEventType.LEVEL_TOUCHED, AlertEventType.LEVEL_BROKEN_UP})
        )
        rules = expand(policy)
        assert len(rules) == 2
        assert {rule.condition.event_type for rule in rules} == policy.event_types  # type: ignore[attr-defined]

    def test_rules_are_ordered_deterministically(self) -> None:
        policy = AlertPolicy(
            event_types=frozenset({AlertEventType.LEVEL_TOUCHED, AlertEventType.LEVEL_BROKEN_UP})
        )
        first = [rule.id for rule in expand(policy)]  # type: ignore[attr-defined]
        second = [rule.id for rule in expand(policy)]  # type: ignore[attr-defined]
        assert first == second == sorted(first)

    def test_rules_point_at_the_namespaced_user_level(self) -> None:
        policy = AlertPolicy(event_types=frozenset({AlertEventType.LEVEL_TOUCHED}))
        (rule,) = expand(policy)
        assert rule.scope.level_id == "user:ul-1"  # type: ignore[attr-defined]
        assert rule.user_id == USER  # type: ignore[attr-defined]

    def test_the_policy_cooldown_wins_over_the_default(self) -> None:
        policy = AlertPolicy(
            event_types=frozenset({AlertEventType.LEVEL_TOUCHED}),
            cooldown=Cooldown(timedelta(minutes=45)),
        )
        (rule,) = expand(policy)
        assert rule.cooldown.duration == timedelta(minutes=45)  # type: ignore[attr-defined]

    def test_the_proximity_threshold_is_carried_to_approaching_rules(self) -> None:
        policy = AlertPolicy(
            event_types=frozenset({AlertEventType.LEVEL_APPROACHING}),
            proximity=absolute("0.75"),
        )
        (rule,) = expand(policy)
        assert rule.condition.proximity == absolute("0.75")  # type: ignore[attr-defined]


class TestRefusals:
    def test_a_level_with_no_policy_cannot_be_expanded(self) -> None:
        with pytest.raises(InvariantViolation, match="alert_policy"):
            expand_policy(
                user_level(None),
                confirmation=CONFIRMATION,
                rearm=REARM,
                channels=CHANNELS,
                cooldown=Cooldown(timedelta(minutes=15)),
            )

    def test_no_cooldown_anywhere_is_refused_rather_than_invented(self) -> None:
        policy = AlertPolicy(event_types=frozenset({AlertEventType.LEVEL_TOUCHED}))
        with pytest.raises(InvariantViolation, match="cooldown"):
            expand(policy, cooldown=None)

    def test_a_trend_event_type_is_not_a_level_policy(self) -> None:
        policy = AlertPolicy(event_types=frozenset({AlertEventType.TREND_CONFIRMED}))
        with pytest.raises(InvariantViolation, match="level"):
            expand(policy)
