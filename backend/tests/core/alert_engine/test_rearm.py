"""Cooldown, re-arm and the E2E scenario in specs/14-testing.md.

Cooldown and arming are two gates, not one (ADR 0011 D5). A rule fires only when
both are open, and the tests below open them one at a time.
"""

from __future__ import annotations

from datetime import timedelta

from alphagate.core.alert_engine import AlertEngine
from alphagate.core.alerts import (
    AlertEvent,
    AlertEventType,
    Cooldown,
    EmissionStatus,
    RearmMode,
    RearmPolicy,
)
from tests.core.alert_engine.synthetic import (
    RULE,
    absolute,
    bars_from,
    make_rule,
    observation,
)

INSIDE = "103.20"
JUST_BELOW = "102.70"
FAR_BELOW = "102.00"


def replay(engine: AlertEngine, closes: list[str]) -> tuple[AlertEvent, ...]:
    events: list[AlertEvent] = []
    for bar in bars_from(closes):
        events.extend(engine.evaluate(observation(bar)))
    return tuple(events)


def statuses(events: tuple[AlertEvent, ...]) -> list[EmissionStatus]:
    return [event.status for event in events]


class TestOnStateChange:
    def test_the_condition_going_false_rearms_the_rule(self) -> None:
        rule = make_rule(
            cooldown=Cooldown(timedelta(minutes=1)),
            rearm=RearmPolicy(mode=RearmMode.ON_STATE_CHANGE),
        )
        engine = AlertEngine((rule,))
        # Bars are one minute apart, so the cooldown expires while price is away.
        events = replay(engine, [FAR_BELOW, INSIDE, FAR_BELOW, FAR_BELOW, INSIDE])
        assert statuses(events) == [EmissionStatus.TRIGGERED, EmissionStatus.TRIGGERED]

    def test_cooldown_still_applies_after_rearming(self) -> None:
        rule = make_rule(
            cooldown=Cooldown(timedelta(hours=1)),
            rearm=RearmPolicy(mode=RearmMode.ON_STATE_CHANGE),
        )
        engine = AlertEngine((rule,))
        events = replay(engine, [FAR_BELOW, INSIDE, FAR_BELOW, INSIDE])
        assert statuses(events) == [
            EmissionStatus.TRIGGERED,
            EmissionStatus.SUPPRESSED_COOLDOWN,
        ]


class TestOnZoneExit:
    def test_leaving_by_less_than_the_hysteresis_does_not_rearm(self) -> None:
        rule = make_rule(
            cooldown=Cooldown(timedelta(minutes=1)),
            rearm=RearmPolicy(mode=RearmMode.ON_ZONE_EXIT, hysteresis=absolute("0.50")),
        )
        engine = AlertEngine((rule,))
        # 103.00 - 102.70 = 0.30, still inside the 0.50 hysteresis.
        events = replay(engine, [FAR_BELOW, INSIDE, JUST_BELOW, JUST_BELOW, INSIDE])
        assert statuses(events) == [
            EmissionStatus.TRIGGERED,
            EmissionStatus.SUPPRESSED_DISARMED,
        ]

    def test_leaving_by_the_hysteresis_distance_rearms(self) -> None:
        rule = make_rule(
            cooldown=Cooldown(timedelta(minutes=1)),
            rearm=RearmPolicy(mode=RearmMode.ON_ZONE_EXIT, hysteresis=absolute("0.50")),
        )
        engine = AlertEngine((rule,))
        # 103.00 - 102.00 = 1.00, clear of the hysteresis.
        events = replay(engine, [FAR_BELOW, INSIDE, FAR_BELOW, FAR_BELOW, INSIDE])
        assert statuses(events) == [EmissionStatus.TRIGGERED, EmissionStatus.TRIGGERED]


class TestAfterTime:
    def test_the_window_alone_does_not_rearm_a_condition_that_never_reset(self) -> None:
        """specs/09-alerts.md: the window passes *and* the condition has reset."""
        rule = make_rule(
            event_type=AlertEventType.LEVEL_APPROACHING,
            cooldown=Cooldown(timedelta(minutes=1)),
            rearm=RearmPolicy(mode=RearmMode.AFTER_TIME, window=timedelta(minutes=2)),
        )
        engine = AlertEngine((rule,))
        # Approaching stays true throughout: one edge, therefore one event.
        events = replay(engine, [FAR_BELOW, *["102.60"] * 10])
        assert statuses(events) == [EmissionStatus.TRIGGERED]

    def test_a_reset_before_the_window_expires_is_still_disarmed(self) -> None:
        rule = make_rule(
            cooldown=Cooldown(timedelta(minutes=1)),
            rearm=RearmPolicy(mode=RearmMode.AFTER_TIME, window=timedelta(minutes=30)),
        )
        engine = AlertEngine((rule,))
        events = replay(engine, [FAR_BELOW, INSIDE, FAR_BELOW, INSIDE])
        assert statuses(events) == [
            EmissionStatus.TRIGGERED,
            EmissionStatus.SUPPRESSED_DISARMED,
        ]

    def test_a_reset_and_an_expired_window_rearm(self) -> None:
        rule = make_rule(
            cooldown=Cooldown(timedelta(minutes=1)),
            rearm=RearmPolicy(mode=RearmMode.AFTER_TIME, window=timedelta(minutes=2)),
        )
        engine = AlertEngine((rule,))
        events = replay(engine, [FAR_BELOW, INSIDE, FAR_BELOW, FAR_BELOW, FAR_BELOW, INSIDE])
        assert statuses(events) == [EmissionStatus.TRIGGERED, EmissionStatus.TRIGGERED]


class TestProductScenario:
    def test_the_e2e_zone_scenario_from_specs_14(self) -> None:
        """Approach, alert, sit in the zone, exit, re-enter, alert again."""
        rule = make_rule(
            cooldown=Cooldown(timedelta(minutes=2)),
            rearm=RearmPolicy(mode=RearmMode.ON_ZONE_EXIT, hysteresis=absolute("0.25")),
        )
        engine = AlertEngine((rule,))
        path = [
            *["101.00"] * 3,  # far away
            *[INSIDE] * 6,  # inside the zone — one alert, no storm
            *["102.00"] * 4,  # out by more than the hysteresis — re-armed
            *[INSIDE] * 3,  # back in — a second alert
        ]
        events = replay(engine, path)
        assert statuses(events) == [EmissionStatus.TRIGGERED, EmissionStatus.TRIGGERED]
        assert engine.state_of(RULE).armed is False
