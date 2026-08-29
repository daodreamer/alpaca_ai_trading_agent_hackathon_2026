"""Boundaries and refusals that the happy paths never reach.

Every one of these is a rule stated somewhere in ADR 0011 or in the Phase 1
vocabulary; a rule with no test is prose.
"""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

import pytest

from alphagate.core.alert_engine import (
    AlertEngine,
    AlertObservation,
    RuleState,
    WatchedLevel,
    alert_event_id,
    read_conditions,
    rule_fingerprint,
    user_level_id,
)
from alphagate.core.alerts import (
    AlertEventType,
    AlertScope,
    ConfirmationPolicy,
    Cooldown,
    ProximityKind,
    ProximityThreshold,
    RearmMode,
    RearmPolicy,
)
from alphagate.core.bar import Feed
from alphagate.core.errors import InvariantViolation
from alphagate.core.identifiers import AlertRuleId, LevelId
from alphagate.core.levels import LevelKind, Zone
from alphagate.core.time_model import Timeframe
from alphagate.core.trend import TrendPhase, TrendState
from alphagate.core.trend_engine import TrendEvent, TrendEventKind
from alphagate.infra.memory import InMemoryAlertEventStore
from tests.core.alert_engine.synthetic import (
    AAPL,
    ZONE_ID,
    absolute,
    bar_at,
    make_rule,
    observation,
    resistance,
    start_of,
)


class TestEngineRefusals:
    def test_an_engine_with_no_rules_is_refused(self) -> None:
        with pytest.raises(InvariantViolation, match="at least one rule"):
            AlertEngine(())

    def test_asking_for_an_unregistered_rule_is_refused(self) -> None:
        engine = AlertEngine((make_rule(),))
        with pytest.raises(InvariantViolation, match="no rule"):
            engine.state_of(AlertRuleId("nope"))

    def test_rules_on_two_timeframes_cannot_share_an_engine(self) -> None:
        other = make_rule(
            id=AlertRuleId("r-2"),
            scope=AlertScope(symbol=AAPL, timeframe=Timeframe.H1, level_id=ZONE_ID),
        )
        with pytest.raises(InvariantViolation, match="one timeframe"):
            AlertEngine((make_rule(), other))

    def test_a_bar_from_another_feed_is_refused(self) -> None:
        engine = AlertEngine((make_rule(),))
        engine.evaluate(observation(bar_at(0, open_="102", high="102", low="102", close="102")))
        wrong_feed = bar_at(1, open_="102", high="102", low="102", close="102", feed=Feed.IEX)
        with pytest.raises(InvariantViolation, match="different series"):
            engine.evaluate(observation(wrong_feed))

    def test_the_engine_describes_itself(self) -> None:
        engine = AlertEngine((make_rule(),))
        assert "AlertEngine" in repr(engine)
        assert engine.rules == (make_rule(),)


class TestZoneExitRearmWithoutAZone:
    def test_a_missing_level_cannot_rearm_the_rule(self) -> None:
        """The level went out of force while the rule was disarmed."""
        rule = make_rule(
            cooldown=Cooldown(timedelta(minutes=1)),
            rearm=RearmPolicy(mode=RearmMode.ON_ZONE_EXIT, hysteresis=absolute("0.10")),
        )
        engine = AlertEngine((rule,))
        engine.evaluate(
            observation(bar_at(0, open_="103.2", high="103.3", low="103.1", close="103.2"))
        )
        # Level gone: nothing to leave, so nothing re-arms.
        engine.evaluate(
            observation(bar_at(1, open_="101", high="101.1", low="100.9", close="101"), levels=())
        )
        assert engine.state_of(rule.id).armed is False

    def test_an_unresolvable_hysteresis_cannot_rearm_the_rule(self) -> None:
        rule = make_rule(
            cooldown=Cooldown(timedelta(minutes=1)),
            rearm=RearmPolicy(
                mode=RearmMode.ON_ZONE_EXIT,
                hysteresis=ProximityThreshold(kind=ProximityKind.ATR_MULTIPLE, value=Decimal("1")),
            ),
        )
        engine = AlertEngine((rule,))
        engine.evaluate(
            observation(bar_at(0, open_="103.2", high="103.3", low="103.1", close="103.2"))
        )
        engine.evaluate(observation(bar_at(1, open_="101", high="101.1", low="100.9", close="101")))
        assert engine.state_of(rule.id).armed is False


class TestValueObjects:
    def test_a_level_price_outside_its_own_zone_is_refused(self) -> None:
        with pytest.raises(InvariantViolation, match="must lie within its zone"):
            WatchedLevel(
                id=LevelId("lvl-x"),
                kind=LevelKind.RESISTANCE,
                price=Decimal("99"),
                zone=Zone(low=Decimal("103"), high=Decimal("104")),
            )

    def test_a_negative_confirmation_count_is_refused(self) -> None:
        with pytest.raises(InvariantViolation, match="consecutive"):
            RuleState(consecutive=-1)

    def test_a_negative_ordinal_is_refused(self) -> None:
        with pytest.raises(InvariantViolation, match="ordinal"):
            alert_event_id("key", -1)

    def test_the_user_level_namespace_is_applied_once(self) -> None:
        assert user_level_id("ul-1") == "user:ul-1"

    def test_the_fingerprint_changes_when_the_rule_does(self) -> None:
        original = make_rule()
        retuned = make_rule(confirmation=ConfirmationPolicy(bars=3))
        assert rule_fingerprint(original) != rule_fingerprint(retuned)

    def test_the_fingerprint_ignores_routing(self) -> None:
        """A channel change is not a different rule (ADR 0011 D9)."""
        original = make_rule()
        rerouted = make_rule(enabled=False)
        assert rule_fingerprint(original) == rule_fingerprint(rerouted)

    def test_an_observation_carries_atr_in_the_exact_domain(self) -> None:
        bar = bar_at(0, open_="102", high="102", low="102", close="102")
        assert AlertObservation(bar=bar, atr=Decimal("0.5")).atr == Decimal("0.50000000")


class TestStrengthDecrease:
    def test_a_fading_trend_below_the_threshold_holds(self) -> None:
        rule = make_rule(
            event_type=AlertEventType.TREND_STRENGTH_DECREASED,
            scope=AlertScope(symbol=AAPL, timeframe=Timeframe.M1),
        )
        state = TrendState(
            symbol=AAPL,
            timeframe=Timeframe.M1,
            state=TrendPhase.WATCH_BULL,
            strength=20.0,
            confidence=1.0,
            as_of=start_of(0),
            reason_codes=("EMA_ALIGNMENT_BULL",),
            input_snapshot_hash="hash-1",
        )
        fading = TrendEvent(
            kind=TrendEventKind.TREND_STRENGTH_DECREASED,
            symbol=AAPL,
            timeframe=Timeframe.M1,
            from_phase=TrendPhase.BULLISH,
            to_phase=TrendPhase.WATCH_BULL,
            bar_start_utc=start_of(0),
            strength=20.0,
            reason_codes=("EMA_ALIGNMENT_BULL",),
            input_snapshot_hash="hash-1",
        )
        bar = bar_at(0, open_="102", high="102", low="102", close="102")
        (reading,) = read_conditions(
            rule,
            AlertObservation(bar=bar, trend=state, trend_events=(fading,)),
            inside_zone=False,
        )
        assert reading.holds is True
        assert "STRENGTH_AT_OR_BELOW_THRESHOLD" in reading.reason_codes


class TestStoreDunders:
    def test_the_store_reports_its_size(self) -> None:
        engine = AlertEngine((make_rule(),))
        store = InMemoryAlertEventStore()
        store.extend(
            engine.evaluate(
                observation(bar_at(0, open_="103.2", high="103.3", low="103.1", close="103.2"))
            )
        )
        assert len(store) == 1
        assert "InMemoryAlertEventStore" in repr(store)


def test_the_default_zone_is_the_one_the_tests_assume() -> None:
    assert resistance().zone == Zone(low=Decimal("103.00"), high=Decimal("103.40"))
