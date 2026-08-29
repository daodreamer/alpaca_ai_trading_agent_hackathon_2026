"""The alert engine — specs/09-alerts.md, ADR 0011.

The engine decides whether a condition that became true reaches anyone. It never
decides what a trend is, and it never sends anything.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from alphagate.core.alert_engine import AlertEngine, AlertObservation, WatchedLevel
from alphagate.core.alerts import (
    AlertCondition,
    AlertEvent,
    AlertEventType,
    AlertScope,
    ConfirmationPolicy,
    Cooldown,
    EmissionStatus,
    RearmMode,
    RearmPolicy,
)
from alphagate.core.errors import InvariantViolation
from alphagate.core.identifiers import AlertRuleId, LevelId, ticker
from alphagate.core.levels import Level, LevelKind, LevelSource, LevelStatus, UserLevel, Zone
from alphagate.core.time_model import Timeframe
from tests.core.alert_engine.synthetic import (
    AAPL,
    RULE,
    USER,
    ZONE_ID,
    absolute,
    bar_at,
    bars_from,
    end_of,
    make_rule,
    observation,
    resistance,
    start_of,
)

INSIDE = "103.20"
BELOW = "102.00"
NEAR = "102.60"
ABOVE = "103.90"


def replay(engine: AlertEngine, closes: list[str]) -> tuple[AlertEvent, ...]:
    events: list[AlertEvent] = []
    for bar in bars_from(closes):
        events.extend(engine.evaluate(observation(bar)))
    return tuple(events)


def triggered(events: tuple[AlertEvent, ...]) -> tuple[AlertEvent, ...]:
    return tuple(event for event in events if event.status is EmissionStatus.TRIGGERED)


# -- construction -----------------------------------------------------------


class TestRegistration:
    def test_a_confirmation_count_a_discrete_condition_can_never_meet_is_refused(self) -> None:
        rule = make_rule(
            event_type=AlertEventType.LEVEL_ZONE_EXITED,
            confirmation=ConfirmationPolicy(bars=2),
        )
        with pytest.raises(InvariantViolation, match="discrete"):
            AlertEngine((rule,))

    def test_an_undefined_event_type_is_refused_rather_than_never_fired(self) -> None:
        rule = make_rule(
            condition=AlertCondition(event_type=AlertEventType.BREAKOUT_CONFIRMED),
            scope=AlertScope(symbol=AAPL, timeframe=Timeframe.M1, level_id=ZONE_ID),
        )
        with pytest.raises(InvariantViolation, match="no MVP definition"):
            AlertEngine((rule,))

    def test_zone_exit_rearm_without_a_level_is_refused(self) -> None:
        rule = make_rule(
            event_type=AlertEventType.TREND_CONFIRMED,
            scope=AlertScope(symbol=AAPL, timeframe=Timeframe.M1),
            rearm=RearmPolicy(mode=RearmMode.ON_ZONE_EXIT, hysteresis=absolute("0.50")),
        )
        with pytest.raises(InvariantViolation):
            AlertEngine((rule,))

    def test_rules_for_two_symbols_cannot_share_an_engine(self) -> None:
        other = make_rule(
            id=AlertRuleId("r-2"),
            scope=AlertScope(symbol=ticker("MSFT"), timeframe=Timeframe.M1, level_id=ZONE_ID),
        )
        with pytest.raises(InvariantViolation, match="one symbol"):
            AlertEngine((make_rule(), other))

    def test_duplicate_rule_ids_are_refused(self) -> None:
        with pytest.raises(InvariantViolation, match="rule id"):
            AlertEngine((make_rule(), make_rule()))


class TestSeriesGuards:
    def test_a_partial_bar_is_refused(self) -> None:
        engine = AlertEngine((make_rule(),))
        bar = bar_at(0, open_=INSIDE, high=INSIDE, low=INSIDE, close=INSIDE, is_final=False)
        with pytest.raises(InvariantViolation, match="partial"):
            engine.evaluate(observation(bar))

    def test_a_bar_that_does_not_advance_is_refused(self) -> None:
        engine = AlertEngine((make_rule(),))
        bar = bar_at(0, open_=INSIDE, high=INSIDE, low=INSIDE, close=INSIDE)
        engine.evaluate(observation(bar))
        with pytest.raises(InvariantViolation, match="strictly increasing"):
            engine.evaluate(observation(bar))

    def test_a_bar_from_another_symbol_is_refused(self) -> None:
        engine = AlertEngine((make_rule(),))
        bar = bar_at(0, open_=INSIDE, high=INSIDE, low=INSIDE, close=INSIDE, symbol=ticker("MSFT"))
        with pytest.raises(InvariantViolation, match="different series"):
            engine.evaluate(observation(bar))

    def test_a_bar_from_another_timeframe_is_refused(self) -> None:
        engine = AlertEngine((make_rule(),))
        bar = bar_at(0, open_=INSIDE, high=INSIDE, low=INSIDE, close=INSIDE, timeframe=Timeframe.H1)
        with pytest.raises(InvariantViolation, match="timeframe"):
            engine.evaluate(observation(bar))


# -- emission ---------------------------------------------------------------


class TestEmissionEdge:
    def test_a_condition_that_becomes_true_fires_once(self) -> None:
        engine = AlertEngine((make_rule(),))
        events = replay(engine, [BELOW, INSIDE])
        assert len(events) == 1
        assert events[0].status is EmissionStatus.TRIGGERED
        assert events[0].event_type is AlertEventType.LEVEL_TOUCHED

    def test_a_condition_that_stays_true_does_not_fire_again(self) -> None:
        """The phase exit criterion: no alert storm while a condition persists."""
        engine = AlertEngine((make_rule(),))
        events = replay(engine, [BELOW, *[INSIDE] * 20])
        assert len(events) == 1

    def test_nothing_fires_before_the_confirmation_count_is_met(self) -> None:
        engine = AlertEngine((make_rule(confirmation=ConfirmationPolicy(bars=3)),))
        assert replay(engine, [BELOW, INSIDE, INSIDE]) == ()

    def test_the_event_lands_on_the_bar_the_count_is_met(self) -> None:
        engine = AlertEngine((make_rule(confirmation=ConfirmationPolicy(bars=3)),))
        events = replay(engine, [BELOW, INSIDE, INSIDE, INSIDE, INSIDE])
        assert len(events) == 1
        assert events[0].occurred_at == end_of(3)

    def test_the_count_restarts_when_the_condition_lapses(self) -> None:
        engine = AlertEngine((make_rule(confirmation=ConfirmationPolicy(bars=3)),))
        assert replay(engine, [INSIDE, INSIDE, BELOW, INSIDE, INSIDE]) == ()

    def test_an_unreadable_condition_moves_no_state(self) -> None:
        engine = AlertEngine((make_rule(confirmation=ConfirmationPolicy(bars=2)),))
        first, second, third = bars_from([INSIDE, INSIDE, INSIDE])
        engine.evaluate(observation(first))
        engine.evaluate(observation(second, levels=()))  # level gone: unreadable
        assert engine.state_of(RULE).consecutive == 1
        assert len(engine.evaluate(observation(third))) == 1


class TestSuppression:
    def test_a_second_edge_inside_the_cooldown_is_recorded_not_swallowed(self) -> None:
        engine = AlertEngine((make_rule(cooldown=Cooldown(timedelta(minutes=30))),))
        events = replay(engine, [BELOW, INSIDE, BELOW, INSIDE])
        assert [event.status for event in events] == [
            EmissionStatus.TRIGGERED,
            EmissionStatus.SUPPRESSED_COOLDOWN,
        ]

    def test_a_suppressed_event_never_reaches_the_notification_router(self) -> None:
        engine = AlertEngine((make_rule(cooldown=Cooldown(timedelta(minutes=30))),))
        events = replay(engine, [BELOW, INSIDE, BELOW, INSIDE])
        assert [event.was_delivered_to_channels for event in events] == [True, False]

    def test_out_of_cooldown_but_not_rearmed_reads_as_disarmed(self) -> None:
        rule = make_rule(
            cooldown=Cooldown(timedelta(minutes=3)),
            rearm=RearmPolicy(mode=RearmMode.AFTER_TIME, window=timedelta(hours=2)),
        )
        engine = AlertEngine((rule,))
        events = replay(engine, [BELOW, INSIDE, BELOW, INSIDE, BELOW, INSIDE])
        assert [event.status for event in events] == [
            EmissionStatus.TRIGGERED,
            EmissionStatus.SUPPRESSED_COOLDOWN,
            EmissionStatus.SUPPRESSED_DISARMED,
        ]

    def test_every_suppressed_event_still_explains_itself(self) -> None:
        engine = AlertEngine((make_rule(cooldown=Cooldown(timedelta(minutes=30))),))
        events = replay(engine, [BELOW, INSIDE, BELOW, INSIDE])
        assert events[1].reason_codes
        assert "SUPPRESSED_BY_COOLDOWN" in events[1].reason_codes
        assert events[1].input_snapshot["cooldown_until"] is not None


class TestEventPayload:
    def test_an_event_carries_everything_the_spec_requires(self) -> None:
        engine = AlertEngine((make_rule(),))
        (event,) = replay(engine, [BELOW, INSIDE])

        assert event.rule_id == RULE
        assert event.symbol == AAPL
        assert event.timeframe is Timeframe.M1
        assert event.event_type is AlertEventType.LEVEL_TOUCHED
        assert event.price == bars_from([INSIDE])[0].close
        assert event.level_id == ZONE_ID
        assert event.reason_codes
        assert event.input_snapshot["rule_fingerprint"]
        assert event.input_snapshot["bar_start_utc"] == start_of(1).isoformat()

    def test_an_event_is_dated_to_the_close_of_its_bar(self) -> None:
        engine = AlertEngine((make_rule(),))
        (event,) = replay(engine, [BELOW, INSIDE])
        assert event.occurred_at == end_of(1)

    def test_ids_are_derived_and_unique(self) -> None:
        engine = AlertEngine((make_rule(cooldown=Cooldown(timedelta(minutes=30))),))
        events = replay(engine, [BELOW, INSIDE, BELOW, INSIDE])
        assert len({event.id for event in events}) == len(events)
        assert RULE in events[0].id
        assert AlertEventType.LEVEL_TOUCHED.value in events[0].id

    def test_the_snapshot_is_json_shaped(self) -> None:
        import json

        engine = AlertEngine((make_rule(),))
        (event,) = replay(engine, [BELOW, INSIDE])
        assert json.dumps(dict(event.input_snapshot))


class TestOrdering:
    def test_events_from_one_bar_are_ordered_by_rule_id(self) -> None:
        second = make_rule(
            id=AlertRuleId("r-2"),
            event_type=AlertEventType.LEVEL_APPROACHING,
            scope=AlertScope(symbol=AAPL, timeframe=Timeframe.M1, level_id=ZONE_ID),
        )
        first = make_rule(id=AlertRuleId("r-1"), event_type=AlertEventType.LEVEL_TOUCHED)
        engine = AlertEngine((second, first))
        bar = bar_at(0, open_=NEAR, high="103.05", low="102.50", close=NEAR)
        events = engine.evaluate(observation(bar))
        assert [event.rule_id for event in events] == ["r-1", "r-2"]


class TestDisabledRules:
    def test_a_disabled_rule_is_not_evaluated(self) -> None:
        engine = AlertEngine((make_rule(enabled=False),))
        assert replay(engine, [BELOW, INSIDE]) == ()


class TestHistory:
    def test_the_engine_keeps_an_append_only_log(self) -> None:
        engine = AlertEngine((make_rule(cooldown=Cooldown(timedelta(minutes=30))),))
        events = replay(engine, [BELOW, INSIDE, BELOW, INSIDE])
        assert engine.events == events


class TestWatchedLevel:
    def test_a_system_level_converts(self) -> None:
        level = Level(
            id=LevelId("lvl-9"),
            symbol=AAPL,
            timeframe=Timeframe.M1,
            kind=LevelKind.RESISTANCE,
            price="103.20",
            zone=Zone(low="103.00", high="103.40"),
            sources=frozenset({LevelSource.SWING_HIGH}),
            strength=50.0,
            confidence=1.0,
            status=LevelStatus.ACTIVE,
            created_at=start_of(0),
            updated_at=start_of(0),
        )
        watched = WatchedLevel.of_level(level)
        assert watched.id == LevelId("lvl-9")
        assert watched.zone == level.zone

    def test_a_user_level_is_namespaced_so_ids_cannot_collide(self) -> None:
        user_level = UserLevel(
            id="ul-9",
            user_id=USER,
            symbol=AAPL,
            kind=LevelKind.RESISTANCE,
            price="103.20",
        )
        watched = WatchedLevel.of_user_level(user_level)
        assert watched.id == "user:ul-9"
        assert watched.zone.low == watched.zone.high == watched.price

    def test_an_observation_refuses_two_levels_with_one_id(self) -> None:
        bar = bar_at(0, open_=INSIDE, high=INSIDE, low=INSIDE, close=INSIDE)
        with pytest.raises(InvariantViolation, match="level id"):
            AlertObservation(bar=bar, levels=(resistance(), resistance()))
