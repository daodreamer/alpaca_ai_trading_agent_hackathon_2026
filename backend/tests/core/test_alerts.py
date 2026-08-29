"""AlertRule, AlertEvent and the policy value objects — specs/09-alerts.md."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from alphagate.core.alerts import (
    AlertAcknowledgement,
    AlertCondition,
    AlertEvent,
    AlertEventType,
    AlertPolicy,
    AlertRule,
    AlertScope,
    ConfirmationPolicy,
    Cooldown,
    EmissionStatus,
    NotificationChannel,
    ProximityKind,
    ProximityThreshold,
    RearmMode,
    RearmPolicy,
)
from alphagate.core.errors import InvariantViolation
from alphagate.core.identifiers import (
    AlertEventId,
    AlertRuleId,
    LevelId,
    UserId,
    UserLevelId,
    ticker,
)
from alphagate.core.levels import LevelKind, UserLevel
from alphagate.core.time_model import Timeframe
from alphagate.core.trend import TrendPhase

AAPL = ticker("AAPL")
NOW = datetime(2026, 8, 13, 14, 0, tzinfo=UTC)


class TestProximityThreshold:
    def test_absolute_threshold_carries_an_exact_price(self) -> None:
        threshold = ProximityThreshold(kind=ProximityKind.ABSOLUTE, value="0.50")
        assert threshold.value == Decimal("0.50000000")

    @pytest.mark.parametrize(
        "kind", [ProximityKind.ABSOLUTE, ProximityKind.PERCENT, ProximityKind.ATR_MULTIPLE]
    )
    def test_value_must_be_positive(self, kind: ProximityKind) -> None:
        with pytest.raises(InvariantViolation, match="positive"):
            ProximityThreshold(kind=kind, value="0")

    def test_rejects_float_value(self) -> None:
        with pytest.raises(TypeError):
            ProximityThreshold(kind=ProximityKind.ATR_MULTIPLE, value=0.25)  # type: ignore[arg-type]

    def test_atr_multiple_example_from_the_spec(self) -> None:
        # "approach if distance(price, zone) <= 0.25 * ATR(14)"
        threshold = ProximityThreshold(kind=ProximityKind.ATR_MULTIPLE, value="0.25")
        assert threshold.kind is ProximityKind.ATR_MULTIPLE
        assert threshold.value == Decimal("0.25000000")


class TestCooldown:
    def test_duration_must_be_positive(self) -> None:
        with pytest.raises(InvariantViolation, match="positive"):
            Cooldown(duration=timedelta(0))

    def test_expires_at_is_relative_to_the_trigger(self) -> None:
        cooldown = Cooldown(duration=timedelta(minutes=30))
        assert cooldown.expires_at(NOW) == NOW + timedelta(minutes=30)

    def test_is_active_is_half_open(self) -> None:
        cooldown = Cooldown(duration=timedelta(minutes=30))
        assert cooldown.is_active(triggered_at=NOW, now=NOW)
        assert cooldown.is_active(triggered_at=NOW, now=NOW + timedelta(minutes=29))
        assert not cooldown.is_active(triggered_at=NOW, now=NOW + timedelta(minutes=30))


class TestRearmPolicy:
    def test_zone_exit_mode_requires_a_hysteresis_distance(self) -> None:
        with pytest.raises(InvariantViolation, match="hysteresis"):
            RearmPolicy(mode=RearmMode.ON_ZONE_EXIT)

    def test_after_time_mode_requires_a_window(self) -> None:
        with pytest.raises(InvariantViolation, match="window"):
            RearmPolicy(mode=RearmMode.AFTER_TIME)

    def test_state_change_mode_needs_neither(self) -> None:
        assert RearmPolicy(mode=RearmMode.ON_STATE_CHANGE).hysteresis is None

    def test_window_must_be_positive(self) -> None:
        with pytest.raises(InvariantViolation, match="window"):
            RearmPolicy(mode=RearmMode.AFTER_TIME, window=timedelta(0))

    def test_zone_exit_mode_rejects_an_unused_window(self) -> None:
        with pytest.raises(InvariantViolation, match="window"):
            RearmPolicy(
                mode=RearmMode.ON_ZONE_EXIT,
                hysteresis=ProximityThreshold(kind=ProximityKind.PERCENT, value="0.5"),
                window=timedelta(hours=1),
            )


class TestAlertPolicy:
    def test_requires_at_least_one_event_type(self) -> None:
        with pytest.raises(InvariantViolation, match="event type"):
            AlertPolicy(event_types=frozenset())

    def test_approaching_requires_a_proximity_threshold(self) -> None:
        with pytest.raises(InvariantViolation, match="proximity"):
            AlertPolicy(event_types=frozenset({AlertEventType.LEVEL_APPROACHING}))

    def test_touch_only_policy_needs_no_proximity(self) -> None:
        policy = AlertPolicy(event_types=frozenset({AlertEventType.LEVEL_TOUCHED}))
        assert policy.proximity is None
        assert policy.cooldown is None

    def test_attaches_to_a_user_level(self) -> None:
        policy = AlertPolicy(
            event_types=frozenset({AlertEventType.LEVEL_APPROACHING}),
            proximity=ProximityThreshold(kind=ProximityKind.ATR_MULTIPLE, value="0.25"),
            cooldown=Cooldown(duration=timedelta(hours=1)),
        )
        level = UserLevel(
            id=UserLevelId("ul-1"),
            user_id=UserId("u-1"),
            symbol=AAPL,
            kind=LevelKind.SUPPORT,
            price="95",
            alert_policy=policy,
        )
        assert level.alert_policy is policy

    def test_is_frozen(self) -> None:
        policy = AlertPolicy(event_types=frozenset({AlertEventType.LEVEL_TOUCHED}))
        with pytest.raises(dataclasses.FrozenInstanceError):
            policy.cooldown = None  # type: ignore[misc]


class TestConfirmationPolicy:
    def test_requires_at_least_one_bar(self) -> None:
        with pytest.raises(InvariantViolation, match="bars"):
            ConfirmationPolicy(bars=0)

    def test_defaults_to_finalized_bars_only(self) -> None:
        # CLAUDE.md §12 — no look-ahead, and no intrabar confirmation unless asked for.
        assert ConfirmationPolicy(bars=2).allow_intrabar is False

    def test_intrabar_must_be_opted_into(self) -> None:
        assert ConfirmationPolicy(bars=1, allow_intrabar=True).allow_intrabar is True


class TestAlertScope:
    def test_level_scoped_rule_names_a_level(self) -> None:
        scope = AlertScope(symbol=AAPL, timeframe=Timeframe.H1, level_id=LevelId("lvl-1"))
        assert scope.level_id == "lvl-1"

    def test_timeframe_is_optional(self) -> None:
        assert AlertScope(symbol=AAPL).timeframe is None


def make_rule(**overrides: object) -> AlertRule:
    defaults: dict[str, object] = {
        "id": AlertRuleId("rule-1"),
        "user_id": UserId("u-1"),
        "scope": AlertScope(symbol=AAPL, timeframe=Timeframe.H1, level_id=LevelId("lvl-1")),
        "condition": AlertCondition(
            event_type=AlertEventType.LEVEL_APPROACHING,
            proximity=ProximityThreshold(kind=ProximityKind.ATR_MULTIPLE, value="0.25"),
        ),
        "confirmation": ConfirmationPolicy(bars=1),
        "cooldown": Cooldown(duration=timedelta(minutes=30)),
        "rearm": RearmPolicy(
            mode=RearmMode.ON_ZONE_EXIT,
            hysteresis=ProximityThreshold(kind=ProximityKind.PERCENT, value="0.5"),
        ),
        "channels": frozenset({NotificationChannel.TELEGRAM}),
        "enabled": True,
    }
    return AlertRule(**{**defaults, **overrides})  # type: ignore[arg-type]


class TestAlertCondition:
    def test_level_events_require_a_proximity_threshold(self) -> None:
        with pytest.raises(InvariantViolation, match="proximity"):
            AlertCondition(event_type=AlertEventType.LEVEL_APPROACHING)

    def test_touch_and_break_events_need_no_proximity(self) -> None:
        assert AlertCondition(event_type=AlertEventType.LEVEL_TOUCHED).proximity is None
        assert AlertCondition(event_type=AlertEventType.LEVEL_BROKEN_UP).proximity is None

    def test_strength_crossing_requires_a_threshold(self) -> None:
        with pytest.raises(InvariantViolation, match="strength_threshold"):
            AlertCondition(event_type=AlertEventType.TREND_STRENGTH_INCREASED)

    def test_strength_threshold_is_bounded(self) -> None:
        with pytest.raises(InvariantViolation, match="strength_threshold"):
            AlertCondition(
                event_type=AlertEventType.TREND_STRENGTH_INCREASED, strength_threshold=101.0
            )

    def test_trend_confirmation_needs_no_threshold(self) -> None:
        assert AlertCondition(event_type=AlertEventType.TREND_CONFIRMED).proximity is None


class TestAlertRule:
    def test_builds_a_valid_rule(self) -> None:
        assert make_rule().enabled

    def test_requires_at_least_one_channel(self) -> None:
        with pytest.raises(InvariantViolation, match="channel"):
            make_rule(channels=frozenset())

    def test_level_scoped_conditions_require_a_level_id(self) -> None:
        with pytest.raises(InvariantViolation, match="level_id"):
            make_rule(scope=AlertScope(symbol=AAPL, timeframe=Timeframe.H1))

    def test_trend_conditions_require_a_timeframe(self) -> None:
        with pytest.raises(InvariantViolation, match="timeframe"):
            make_rule(
                scope=AlertScope(symbol=AAPL),
                condition=AlertCondition(event_type=AlertEventType.TREND_CONFIRMED),
            )

    def test_is_frozen(self) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            make_rule().enabled = False  # type: ignore[misc]


def make_event(**overrides: object) -> AlertEvent:
    defaults: dict[str, object] = {
        "id": AlertEventId("evt-1"),
        "rule_id": AlertRuleId("rule-1"),
        "symbol": AAPL,
        "timeframe": Timeframe.H1,
        "event_type": AlertEventType.LEVEL_APPROACHING,
        "occurred_at": NOW,
        "price": "104.50",
        "level_id": LevelId("lvl-1"),
        "reason_codes": ("DISTANCE_WITHIN_0.25_ATR",),
        "input_snapshot": {"atr14": 2.0, "close": "104.50"},
        "status": EmissionStatus.TRIGGERED,
    }
    return AlertEvent(**{**defaults, **overrides})  # type: ignore[arg-type]


class TestAlertEvent:
    def test_builds_a_valid_event(self) -> None:
        assert make_event().price == Decimal("104.50000000")

    def test_must_reference_the_rule_that_caused_it(self) -> None:
        with pytest.raises(InvariantViolation, match="rule_id"):
            make_event(rule_id=AlertRuleId(""))

    def test_reason_codes_are_required(self) -> None:
        # specs/01-product.md: every alert has a reason payload viewable in the UI.
        with pytest.raises(InvariantViolation, match="reason_codes"):
            make_event(reason_codes=())

    def test_level_events_must_reference_a_level(self) -> None:
        with pytest.raises(InvariantViolation, match="level_id"):
            make_event(level_id=None)

    def test_trend_events_must_carry_the_state_change(self) -> None:
        with pytest.raises(InvariantViolation, match="trend_state"):
            make_event(
                event_type=AlertEventType.TREND_CONFIRMED,
                level_id=None,
            )

    def test_trend_event_with_before_and_after_is_valid(self) -> None:
        event = make_event(
            event_type=AlertEventType.TREND_CONFIRMED,
            level_id=None,
            trend_state_before=TrendPhase.WATCH_BULL,
            trend_state_after=TrendPhase.BULLISH,
        )
        assert event.trend_state_after is TrendPhase.BULLISH

    def test_occurred_at_must_be_timezone_aware(self) -> None:
        with pytest.raises(InvariantViolation, match="timezone"):
            make_event(occurred_at=datetime(2026, 8, 13, 14, 0))  # noqa: DTZ001

    def test_is_frozen(self) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            make_event().price = Decimal("1")  # type: ignore[misc]

    def test_input_snapshot_is_not_shared_with_the_caller(self) -> None:
        snapshot = {"atr14": 2.0}
        event = make_event(input_snapshot=snapshot)
        snapshot["atr14"] = 99.0
        assert event.input_snapshot["atr14"] == 2.0

    def test_suppressed_events_are_recorded_as_facts(self) -> None:
        # specs/01-product.md: the alert center shows suppressed/cooldown events.
        event = make_event(status=EmissionStatus.SUPPRESSED_COOLDOWN)
        assert event.status is EmissionStatus.SUPPRESSED_COOLDOWN
        assert not event.was_delivered_to_channels

    def test_triggered_events_are_deliverable(self) -> None:
        assert make_event().was_delivered_to_channels


class TestAcknowledgement:
    def test_acknowledgement_is_separate_mutable_state(self) -> None:
        # specs/04-domain-model.md: events are immutable facts; acknowledgement is not
        # a field on the event.
        assert not hasattr(make_event(), "acknowledged")
        ack = AlertAcknowledgement(
            event_id=AlertEventId("evt-1"),
            user_id=UserId("u-1"),
            acknowledged_at=NOW,
        )
        assert ack.event_id == "evt-1"

    def test_acknowledgement_time_must_be_aware(self) -> None:
        with pytest.raises(InvariantViolation, match="timezone"):
            AlertAcknowledgement(
                event_id=AlertEventId("evt-1"),
                user_id=UserId("u-1"),
                acknowledged_at=datetime(2026, 8, 13, 14, 0),  # noqa: DTZ001
            )
