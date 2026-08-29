"""Condition predicates — ADR 0011 D2.

One bar in, one verdict out. `None` is a third answer and means "not
measurable", which is never the same as "false".
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from alphagate.core.alert_engine import ConditionReading, is_discrete, read_conditions
from alphagate.core.alerts import (
    AlertCondition,
    AlertEventType,
    AlertScope,
    ProximityKind,
    ProximityThreshold,
)
from alphagate.core.identifiers import LevelId
from alphagate.core.time_model import Timeframe
from alphagate.core.trend import TrendPhase, TrendState
from alphagate.core.trend_engine import TrendEvent, TrendEventKind
from tests.core.alert_engine.synthetic import (
    AAPL,
    ZONE_ID,
    bar_at,
    make_rule,
    observation,
    resistance,
    start_of,
)


def only(readings: tuple[ConditionReading, ...]) -> ConditionReading:
    assert len(readings) == 1
    return readings[0]


# -- zone geometry ----------------------------------------------------------
# The default watched zone is resistance at [103.00, 103.40].


class TestApproaching:
    def test_holds_when_the_close_is_within_the_threshold(self) -> None:
        # 103.00 - 102.60 = 0.40, inside the 0.50 threshold.
        bar = bar_at(0, open_="102.60", high="102.70", low="102.50", close="102.60")
        rule = make_rule(event_type=AlertEventType.LEVEL_APPROACHING)
        assert only(read_conditions(rule, observation(bar), inside_zone=False)).holds is True

    def test_does_not_hold_when_further_away_than_the_threshold(self) -> None:
        bar = bar_at(0, open_="102.00", high="102.10", low="101.90", close="102.00")
        rule = make_rule(event_type=AlertEventType.LEVEL_APPROACHING)
        assert only(read_conditions(rule, observation(bar), inside_zone=False)).holds is False

    def test_inside_the_zone_is_not_approaching_it(self) -> None:
        bar = bar_at(0, open_="103.20", high="103.30", low="103.10", close="103.20")
        rule = make_rule(event_type=AlertEventType.LEVEL_APPROACHING)
        assert only(read_conditions(rule, observation(bar), inside_zone=False)).holds is False

    def test_exactly_at_the_threshold_counts_as_approaching(self) -> None:
        bar = bar_at(0, open_="102.50", high="102.60", low="102.40", close="102.50")
        rule = make_rule(event_type=AlertEventType.LEVEL_APPROACHING)
        assert only(read_conditions(rule, observation(bar), inside_zone=False)).holds is True

    def test_percent_threshold_is_resolved_against_the_close(self) -> None:
        # 1% of 102.00 is 1.02, and the zone is 1.00 away.
        threshold = ProximityThreshold(kind=ProximityKind.PERCENT, value=Decimal("1"))
        rule = make_rule(
            condition=AlertCondition(
                event_type=AlertEventType.LEVEL_APPROACHING, proximity=threshold
            )
        )
        bar = bar_at(0, open_="102.00", high="102.10", low="101.90", close="102.00")
        assert only(read_conditions(rule, observation(bar), inside_zone=False)).holds is True

    def test_atr_threshold_is_unreadable_while_atr_is_warming(self) -> None:
        threshold = ProximityThreshold(kind=ProximityKind.ATR_MULTIPLE, value=Decimal("0.25"))
        rule = make_rule(
            condition=AlertCondition(
                event_type=AlertEventType.LEVEL_APPROACHING, proximity=threshold
            )
        )
        bar = bar_at(0, open_="102.90", high="102.95", low="102.85", close="102.90")
        reading = only(read_conditions(rule, observation(bar, atr=None), inside_zone=False))
        assert reading.holds is None

    def test_atr_threshold_resolves_once_atr_is_warm(self) -> None:
        threshold = ProximityThreshold(kind=ProximityKind.ATR_MULTIPLE, value=Decimal("2"))
        rule = make_rule(
            condition=AlertCondition(
                event_type=AlertEventType.LEVEL_APPROACHING, proximity=threshold
            )
        )
        bar = bar_at(0, open_="102.60", high="102.70", low="102.50", close="102.60")
        reading = only(
            read_conditions(rule, observation(bar, atr=Decimal("0.25")), inside_zone=False)
        )
        assert reading.holds is True


class TestTouched:
    def test_a_wick_into_the_zone_is_a_touch(self) -> None:
        bar = bar_at(0, open_="102.80", high="103.05", low="102.70", close="102.90")
        rule = make_rule(event_type=AlertEventType.LEVEL_TOUCHED)
        assert only(read_conditions(rule, observation(bar), inside_zone=False)).holds is True

    def test_a_bar_entirely_below_the_zone_is_not_a_touch(self) -> None:
        bar = bar_at(0, open_="102.80", high="102.95", low="102.70", close="102.90")
        rule = make_rule(event_type=AlertEventType.LEVEL_TOUCHED)
        assert only(read_conditions(rule, observation(bar), inside_zone=False)).holds is False

    def test_touching_the_exact_edge_counts(self) -> None:
        bar = bar_at(0, open_="102.80", high="103.00", low="102.70", close="102.90")
        rule = make_rule(event_type=AlertEventType.LEVEL_TOUCHED)
        assert only(read_conditions(rule, observation(bar), inside_zone=False)).holds is True


class TestBreaks:
    def test_a_close_above_the_zone_breaks_upward(self) -> None:
        bar = bar_at(0, open_="103.30", high="103.60", low="103.20", close="103.50")
        rule = make_rule(event_type=AlertEventType.LEVEL_BROKEN_UP)
        assert only(read_conditions(rule, observation(bar), inside_zone=False)).holds is True

    def test_a_wick_above_the_zone_does_not_break_it(self) -> None:
        bar = bar_at(0, open_="103.10", high="103.90", low="103.00", close="103.20")
        rule = make_rule(event_type=AlertEventType.LEVEL_BROKEN_UP)
        assert only(read_conditions(rule, observation(bar), inside_zone=False)).holds is False

    def test_a_close_exactly_on_the_edge_does_not_break_it(self) -> None:
        bar = bar_at(0, open_="103.30", high="103.45", low="103.20", close="103.40")
        rule = make_rule(event_type=AlertEventType.LEVEL_BROKEN_UP)
        assert only(read_conditions(rule, observation(bar), inside_zone=False)).holds is False

    def test_a_close_below_the_zone_breaks_downward(self) -> None:
        bar = bar_at(0, open_="103.10", high="103.20", low="102.50", close="102.60")
        rule = make_rule(event_type=AlertEventType.LEVEL_BROKEN_DOWN)
        assert only(read_conditions(rule, observation(bar), inside_zone=False)).holds is True


class TestZoneExit:
    def test_leaving_the_zone_holds_only_if_price_was_inside(self) -> None:
        bar = bar_at(0, open_="103.10", high="103.20", low="102.50", close="102.60")
        rule = make_rule(event_type=AlertEventType.LEVEL_ZONE_EXITED)
        assert only(read_conditions(rule, observation(bar), inside_zone=True)).holds is True
        assert only(read_conditions(rule, observation(bar), inside_zone=False)).holds is False

    def test_staying_inside_is_not_an_exit(self) -> None:
        bar = bar_at(0, open_="103.10", high="103.20", low="103.05", close="103.10")
        rule = make_rule(event_type=AlertEventType.LEVEL_ZONE_EXITED)
        assert only(read_conditions(rule, observation(bar), inside_zone=True)).holds is False


class TestMissingLevel:
    def test_a_level_that_is_not_in_the_observation_is_unreadable(self) -> None:
        bar = bar_at(0, open_="103.10", high="103.20", low="103.05", close="103.10")
        rule = make_rule(event_type=AlertEventType.LEVEL_TOUCHED)
        reading = only(read_conditions(rule, observation(bar, levels=()), inside_zone=False))
        assert reading.holds is None

    def test_a_different_level_in_the_observation_does_not_substitute(self) -> None:
        bar = bar_at(0, open_="103.10", high="103.20", low="103.05", close="103.10")
        other = resistance(level_id=LevelId("lvl-other"))
        rule = make_rule(event_type=AlertEventType.LEVEL_TOUCHED)
        reading = only(read_conditions(rule, observation(bar, levels=(other,)), inside_zone=False))
        assert reading.holds is None


# -- trend ------------------------------------------------------------------


def trend_state(phase: TrendPhase, *, strength: float) -> TrendState:
    return TrendState(
        symbol=AAPL,
        timeframe=Timeframe.M1,
        state=phase,
        strength=strength,
        confidence=1.0,
        as_of=start_of(0),
        reason_codes=("EMA_ALIGNMENT_BULL",),
        input_snapshot_hash="deadbeef",
    )


def trend_event(
    kind: TrendEventKind,
    *,
    from_phase: TrendPhase = TrendPhase.NEUTRAL,
    to_phase: TrendPhase = TrendPhase.WATCH_BULL,
) -> TrendEvent:
    return TrendEvent(
        kind=kind,
        symbol=AAPL,
        timeframe=Timeframe.M1,
        from_phase=from_phase,
        to_phase=to_phase,
        bar_start_utc=start_of(0),
        strength=60.0,
        reason_codes=("EMA_ALIGNMENT_BULL",),
        input_snapshot_hash="deadbeef",
    )


def trend_rule(event_type: AlertEventType, **overrides: object) -> object:
    return make_rule(
        event_type=event_type,
        scope=AlertScope(symbol=AAPL, timeframe=Timeframe.M1),
        **overrides,
    )


class TestTrendConditions:
    def test_a_matching_trend_event_holds_and_is_carried(self) -> None:
        bar = bar_at(0, open_="100", high="101", low="99", close="100")
        event = trend_event(TrendEventKind.TREND_CONFIRMED)
        reading = only(
            read_conditions(
                trend_rule(AlertEventType.TREND_CONFIRMED),  # type: ignore[arg-type]
                observation(
                    bar,
                    levels=(),
                    trend=trend_state(TrendPhase.WATCH_BULL, strength=60.0),
                    trend_events=(event,),
                ),
                inside_zone=False,
            )
        )
        assert reading.holds is True
        assert reading.trend_event is event

    def test_a_bar_with_no_trend_event_does_not_hold(self) -> None:
        bar = bar_at(0, open_="100", high="101", low="99", close="100")
        reading = only(
            read_conditions(
                trend_rule(AlertEventType.TREND_CONFIRMED),  # type: ignore[arg-type]
                observation(
                    bar, levels=(), trend=trend_state(TrendPhase.WATCH_BULL, strength=60.0)
                ),
                inside_zone=False,
            )
        )
        assert reading.holds is False

    def test_no_trend_state_at_all_is_unreadable(self) -> None:
        bar = bar_at(0, open_="100", high="101", low="99", close="100")
        reading = only(
            read_conditions(
                trend_rule(AlertEventType.TREND_CONFIRMED),  # type: ignore[arg-type]
                observation(bar, levels=()),
                inside_zone=False,
            )
        )
        assert reading.holds is None

    def test_a_wrong_kind_of_trend_event_does_not_hold(self) -> None:
        bar = bar_at(0, open_="100", high="101", low="99", close="100")
        event = trend_event(TrendEventKind.TREND_REVERSAL)
        reading = only(
            read_conditions(
                trend_rule(AlertEventType.TREND_CONFIRMED),  # type: ignore[arg-type]
                observation(
                    bar,
                    levels=(),
                    trend=trend_state(TrendPhase.WATCH_BULL, strength=60.0),
                    trend_events=(event,),
                ),
                inside_zone=False,
            )
        )
        assert reading.holds is False

    def test_strength_threshold_filters_a_ladder_step(self) -> None:
        bar = bar_at(0, open_="100", high="101", low="99", close="100")
        event = trend_event(
            TrendEventKind.TREND_STRENGTH_INCREASED,
            from_phase=TrendPhase.WATCH_BULL,
            to_phase=TrendPhase.BULLISH,
        )
        rule = trend_rule(AlertEventType.TREND_STRENGTH_INCREASED)

        weak = only(
            read_conditions(
                rule,  # type: ignore[arg-type]
                observation(
                    bar,
                    levels=(),
                    trend=trend_state(TrendPhase.BULLISH, strength=40.0),
                    trend_events=(event,),
                ),
                inside_zone=False,
            )
        )
        strong = only(
            read_conditions(
                rule,  # type: ignore[arg-type]
                observation(
                    bar,
                    levels=(),
                    trend=trend_state(TrendPhase.BULLISH, strength=60.0),
                    trend_events=(event,),
                ),
                inside_zone=False,
            )
        )
        assert weak.holds is False
        assert strong.holds is True

    def test_one_bar_that_walks_two_rungs_reads_twice(self) -> None:
        """ADR 0010 D6 hands this case to the alert engine; it must not be lost."""
        bar = bar_at(0, open_="100", high="101", low="99", close="100")
        first = trend_event(
            TrendEventKind.TREND_STRENGTH_INCREASED,
            from_phase=TrendPhase.WATCH_BULL,
            to_phase=TrendPhase.BULLISH,
        )
        second = trend_event(
            TrendEventKind.TREND_STRENGTH_INCREASED,
            from_phase=TrendPhase.BULLISH,
            to_phase=TrendPhase.STRONG_BULLISH,
        )
        readings = read_conditions(
            trend_rule(AlertEventType.TREND_STRENGTH_INCREASED),  # type: ignore[arg-type]
            observation(
                bar,
                levels=(),
                trend=trend_state(TrendPhase.STRONG_BULLISH, strength=90.0),
                trend_events=(first, second),
            ),
            inside_zone=False,
        )
        assert [reading.holds for reading in readings] == [True, True]


class TestDiscreteness:
    @pytest.mark.parametrize(
        "event_type",
        [
            AlertEventType.LEVEL_ZONE_EXITED,
            AlertEventType.TREND_CONFIRMED,
            AlertEventType.TREND_REVERSAL,
            AlertEventType.TREND_STRENGTH_INCREASED,
            AlertEventType.TREND_STRENGTH_DECREASED,
        ],
    )
    def test_momentary_conditions_are_discrete(self, event_type: AlertEventType) -> None:
        assert is_discrete(event_type) is True

    @pytest.mark.parametrize(
        "event_type",
        [
            AlertEventType.LEVEL_APPROACHING,
            AlertEventType.LEVEL_TOUCHED,
            AlertEventType.LEVEL_BROKEN_UP,
            AlertEventType.LEVEL_BROKEN_DOWN,
        ],
    )
    def test_persisting_conditions_are_continuous(self, event_type: AlertEventType) -> None:
        assert is_discrete(event_type) is False


class TestReasonCodes:
    def test_a_holding_condition_explains_itself(self) -> None:
        bar = bar_at(0, open_="103.30", high="103.60", low="103.20", close="103.50")
        rule = make_rule(event_type=AlertEventType.LEVEL_BROKEN_UP)
        reading = only(read_conditions(rule, observation(bar), inside_zone=False))
        assert "CLOSE_ABOVE_ZONE_HIGH" in reading.reason_codes

    def test_the_snapshot_fragment_carries_prices_as_strings(self) -> None:
        bar = bar_at(0, open_="103.30", high="103.60", low="103.20", close="103.50")
        rule = make_rule(event_type=AlertEventType.LEVEL_BROKEN_UP)
        reading = only(read_conditions(rule, observation(bar), inside_zone=False))
        level = reading.detail["level"]
        assert isinstance(level["zone_high"], str)
        assert level["id"] == ZONE_ID


def test_datetimes_in_this_module_are_utc() -> None:
    assert start_of(0).tzinfo is UTC
    assert isinstance(start_of(0), datetime)
