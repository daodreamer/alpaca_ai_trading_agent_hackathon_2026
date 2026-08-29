"""Trend-sourced alerts, and the duplicate ADR 0010 D6 hands to this phase."""

from __future__ import annotations

from alphagate.core.alert_engine import AlertEngine, AlertObservation
from alphagate.core.alerts import AlertEventType, AlertScope, EmissionStatus
from alphagate.core.identifiers import AlertRuleId
from alphagate.core.time_model import Timeframe
from alphagate.core.trend import TrendPhase, TrendState
from alphagate.core.trend_engine import TrendEvent, TrendEventKind
from tests.core.alert_engine.synthetic import AAPL, bar_at, make_rule, start_of


def state(phase: TrendPhase, *, strength: float, index: int = 0) -> TrendState:
    return TrendState(
        symbol=AAPL,
        timeframe=Timeframe.M1,
        state=phase,
        strength=strength,
        confidence=1.0,
        as_of=start_of(index),
        reason_codes=("EMA_ALIGNMENT_BULL", "STRUCTURE_BULL"),
        input_snapshot_hash="hash-1",
    )


def event(
    kind: TrendEventKind,
    *,
    from_phase: TrendPhase,
    to_phase: TrendPhase,
    index: int = 0,
    strength: float = 60.0,
) -> TrendEvent:
    return TrendEvent(
        kind=kind,
        symbol=AAPL,
        timeframe=Timeframe.M1,
        from_phase=from_phase,
        to_phase=to_phase,
        bar_start_utc=start_of(index),
        strength=strength,
        reason_codes=("EMA_ALIGNMENT_BULL", "STRUCTURE_BULL"),
        input_snapshot_hash="hash-1",
    )


def trend_rule(event_type: AlertEventType, **overrides: object) -> object:
    return make_rule(
        event_type=event_type,
        scope=AlertScope(symbol=AAPL, timeframe=Timeframe.M1),
        **overrides,
    )


def observe(index: int, **extra: object) -> AlertObservation:
    bar = bar_at(index, open_="100", high="101", low="99", close="100")
    return AlertObservation(bar=bar, **extra)  # type: ignore[arg-type]


class TestTrendConfirmed:
    def test_a_confirmation_event_becomes_an_alert(self) -> None:
        engine = AlertEngine((trend_rule(AlertEventType.TREND_CONFIRMED),))  # type: ignore[arg-type]
        confirmed = event(
            TrendEventKind.TREND_CONFIRMED,
            from_phase=TrendPhase.NEUTRAL,
            to_phase=TrendPhase.WATCH_BULL,
        )
        (alert,) = engine.evaluate(
            observe(
                0,
                trend=state(TrendPhase.WATCH_BULL, strength=60.0),
                trend_events=(confirmed,),
            )
        )
        assert alert.status is EmissionStatus.TRIGGERED
        assert alert.trend_state_before is TrendPhase.NEUTRAL
        assert alert.trend_state_after is TrendPhase.WATCH_BULL
        assert "EMA_ALIGNMENT_BULL" in alert.reason_codes

    def test_a_quiet_bar_produces_nothing(self) -> None:
        engine = AlertEngine((trend_rule(AlertEventType.TREND_CONFIRMED),))  # type: ignore[arg-type]
        assert engine.evaluate(observe(0, trend=state(TrendPhase.BULLISH, strength=60.0))) == ()


class TestDuplicate:
    def test_the_second_rung_of_one_bar_is_recorded_as_a_duplicate(self) -> None:
        """ADR 0010 D6: one bar can walk two rungs. It is one alert, not two."""
        engine = AlertEngine((trend_rule(AlertEventType.TREND_STRENGTH_INCREASED),))  # type: ignore[arg-type]
        first = event(
            TrendEventKind.TREND_STRENGTH_INCREASED,
            from_phase=TrendPhase.WATCH_BULL,
            to_phase=TrendPhase.BULLISH,
        )
        second = event(
            TrendEventKind.TREND_STRENGTH_INCREASED,
            from_phase=TrendPhase.BULLISH,
            to_phase=TrendPhase.STRONG_BULLISH,
            strength=90.0,
        )
        events = engine.evaluate(
            observe(
                0,
                trend=state(TrendPhase.STRONG_BULLISH, strength=90.0),
                trend_events=(first, second),
            )
        )
        assert [alert.status for alert in events] == [
            EmissionStatus.TRIGGERED,
            EmissionStatus.SUPPRESSED_DUPLICATE,
        ]

    def test_duplicate_outranks_cooldown(self) -> None:
        engine = AlertEngine((trend_rule(AlertEventType.TREND_STRENGTH_INCREASED),))  # type: ignore[arg-type]
        rungs = (
            event(
                TrendEventKind.TREND_STRENGTH_INCREASED,
                from_phase=TrendPhase.WATCH_BULL,
                to_phase=TrendPhase.BULLISH,
            ),
            event(
                TrendEventKind.TREND_STRENGTH_INCREASED,
                from_phase=TrendPhase.BULLISH,
                to_phase=TrendPhase.STRONG_BULLISH,
                strength=90.0,
            ),
        )
        events = engine.evaluate(
            observe(0, trend=state(TrendPhase.STRONG_BULLISH, strength=90.0), trend_events=rungs)
        )
        assert events[1].status is EmissionStatus.SUPPRESSED_DUPLICATE
        assert "DUPLICATE_EMISSION" in events[1].reason_codes

    def test_a_duplicate_still_gets_its_own_id(self) -> None:
        engine = AlertEngine((trend_rule(AlertEventType.TREND_STRENGTH_INCREASED),))  # type: ignore[arg-type]
        rungs = (
            event(
                TrendEventKind.TREND_STRENGTH_INCREASED,
                from_phase=TrendPhase.WATCH_BULL,
                to_phase=TrendPhase.BULLISH,
            ),
            event(
                TrendEventKind.TREND_STRENGTH_INCREASED,
                from_phase=TrendPhase.BULLISH,
                to_phase=TrendPhase.STRONG_BULLISH,
                strength=90.0,
            ),
        )
        events = engine.evaluate(
            observe(0, trend=state(TrendPhase.STRONG_BULLISH, strength=90.0), trend_events=rungs)
        )
        assert events[0].id != events[1].id


class TestStrengthThreshold:
    def test_a_ladder_step_below_the_threshold_is_not_an_alert(self) -> None:
        engine = AlertEngine((trend_rule(AlertEventType.TREND_STRENGTH_INCREASED),))  # type: ignore[arg-type]
        step = event(
            TrendEventKind.TREND_STRENGTH_INCREASED,
            from_phase=TrendPhase.NEUTRAL,
            to_phase=TrendPhase.WATCH_BULL,
            strength=30.0,
        )
        assert (
            engine.evaluate(
                observe(0, trend=state(TrendPhase.WATCH_BULL, strength=30.0), trend_events=(step,))
            )
            == ()
        )


class TestSeparateRules:
    def test_two_rules_on_one_bar_each_get_their_own_event(self) -> None:
        engine = AlertEngine(
            (
                trend_rule(AlertEventType.TREND_CONFIRMED, id=AlertRuleId("r-a")),  # type: ignore[arg-type]
                trend_rule(AlertEventType.TREND_REVERSAL, id=AlertRuleId("r-b")),  # type: ignore[arg-type]
            )
        )
        reversal = event(
            TrendEventKind.TREND_REVERSAL,
            from_phase=TrendPhase.NEUTRAL,
            to_phase=TrendPhase.WATCH_BULL,
        )
        events = engine.evaluate(
            observe(0, trend=state(TrendPhase.WATCH_BULL, strength=60.0), trend_events=(reversal,))
        )
        assert [alert.rule_id for alert in events] == ["r-b"]
