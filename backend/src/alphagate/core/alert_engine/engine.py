"""The alert engine — specs/09-alerts.md, ADR 0011.

Decides whether a condition that became true reaches anyone. It identifies no
structure, computes no indicator, judges no trend and sends no notification
(CLAUDE.md §8): it consumes an observation, applies confirmation, cooldown and
re-arm, and returns `AlertEvent`s — including the ones that were suppressed,
because "why didn't I get an alert?" is a question the alert center has to be
able to answer.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any, Final

from alphagate.core.alert_engine.conditions import (
    is_discrete,
    is_supported,
    read_conditions,
    resolve_threshold,
)
from alphagate.core.alert_engine.model import (
    AlertObservation,
    ConditionReading,
    RuleState,
    WatchedLevel,
    alert_event_id,
    event_key,
    rule_fingerprint,
)
from alphagate.core.alerts import AlertEvent, AlertRule, EmissionStatus, RearmMode
from alphagate.core.bar import AdjustmentMode, Bar, Feed
from alphagate.core.errors import InvariantViolation
from alphagate.core.identifiers import AlertRuleId, Ticker
from alphagate.core.numeric import format_exact
from alphagate.core.time_model import Timeframe

__all__ = ["AlertEngine"]

_SUPPRESSION_CODES: Final[dict[EmissionStatus, str]] = {
    EmissionStatus.SUPPRESSED_COOLDOWN: "SUPPRESSED_BY_COOLDOWN",
    EmissionStatus.SUPPRESSED_DISARMED: "RULE_DISARMED",
    EmissionStatus.SUPPRESSED_DUPLICATE: "DUPLICATE_EMISSION",
}

type _SeriesKey = tuple[Ticker, Timeframe, Feed, AdjustmentMode]


class AlertEngine:
    """Rule evaluation for one symbol on one timeframe.

    Stateful, because cooldown and re-arm are memory, and single-series for the
    same reason every other engine here is: a rule counting confirmation bars
    cannot be fed two markets.
    """

    def __init__(self, rules: Sequence[AlertRule]) -> None:
        self._rules = tuple(sorted(rules, key=lambda rule: rule.id))
        self._check_rules()

        self._symbol: Ticker = self._rules[0].scope.symbol
        self._timeframe: Timeframe | None = next(
            (rule.scope.timeframe for rule in self._rules if rule.scope.timeframe is not None),
            None,
        )

        self._states: dict[AlertRuleId, RuleState] = {rule.id: RuleState() for rule in self._rules}
        self._series: _SeriesKey | None = None
        self._last_start: datetime | None = None
        self._ordinals: dict[str, int] = {}
        self._events: list[AlertEvent] = []

    # -- identity ----------------------------------------------------------

    def __repr__(self) -> str:
        return f"<AlertEngine {self._symbol} rules={len(self._rules)} events={len(self._events)}>"

    @property
    def rules(self) -> tuple[AlertRule, ...]:
        """The registered rules, ordered by id — which is also emission order."""
        return self._rules

    @property
    def events(self) -> tuple[AlertEvent, ...]:
        """Every event emitted so far, in order. Append-only."""
        return tuple(self._events)

    def state_of(self, rule_id: AlertRuleId) -> RuleState:
        if rule_id not in self._states:
            raise InvariantViolation(f"no rule {rule_id!r} is registered with this engine")
        return self._states[rule_id]

    def restore(self, states: Mapping[AlertRuleId, RuleState]) -> None:
        """Adopt persisted gate state after a restart (ADR 0012 D6).

        An engine that came back armed with an empty cooldown would re-fire every
        condition that is still true, which is the storm Phase 6 exists to
        prevent. What may be restored is a decision the caller has already made;
        this method only refuses to apply it at a point where it would be a lie.

        Only before the first bar: rule state that arrived mid-stream would be
        state for bars this engine has already judged.
        """
        if self._last_start is not None:
            raise InvariantViolation(
                "restore before the first bar — adopting rule state after the engine "
                "has judged bars would silently rewrite decisions it already made"
            )
        unknown = sorted(set(states) - set(self._states))
        if unknown:
            raise InvariantViolation(
                f"restoring state for rules this engine does not have: {unknown}"
            )
        self._states.update(states)

    # -- registration ------------------------------------------------------

    def _check_rules(self) -> None:
        if not self._rules:
            raise InvariantViolation(
                "an alert engine needs at least one rule; a symbol nobody watches has "
                "no engine rather than an empty one"
            )

        seen: set[AlertRuleId] = set()
        for rule in self._rules:
            if rule.id in seen:
                raise InvariantViolation(f"rule id {rule.id!r} is registered twice")
            seen.add(rule.id)

            event_type = rule.condition.event_type
            if not is_supported(event_type):
                raise InvariantViolation(
                    f"{rule.id}: {event_type.value} has no MVP definition "
                    "(specs/09-alerts.md lists neither breakout type among the MVP "
                    "alert types); refusing a rule that could never fire"
                )
            if is_discrete(event_type) and rule.confirmation.bars > 1:
                raise InvariantViolation(
                    f"{rule.id}: {event_type.value} is a discrete condition — it happens "
                    f"on one bar, so it can never hold for {rule.confirmation.bars} "
                    "consecutive bars (ADR 0011 D4)"
                )
            if rule.rearm.mode is RearmMode.ON_ZONE_EXIT and rule.scope.level_id is None:
                raise InvariantViolation(
                    f"{rule.id}: ON_ZONE_EXIT re-arm needs a level to leave; the rule's "
                    "scope has no level_id"
                )

        symbols = {rule.scope.symbol for rule in self._rules}
        if len(symbols) > 1:
            raise InvariantViolation(
                f"one engine follows one symbol, got {sorted(symbols)}; run one engine per symbol"
            )
        timeframes = {
            rule.scope.timeframe for rule in self._rules if rule.scope.timeframe is not None
        }
        if len(timeframes) > 1:
            raise InvariantViolation(
                "one engine follows one timeframe, got "
                f"{sorted(timeframe.code for timeframe in timeframes)}"
            )

    # -- evaluation --------------------------------------------------------

    def evaluate(self, observation: AlertObservation) -> tuple[AlertEvent, ...]:
        """Judge every rule against one bar and return what it produced.

        The returned tuple contains suppressed events as well as triggered ones.
        Only `AlertEvent.was_delivered_to_channels` decides what a notification
        router should look at.
        """
        bar = observation.bar
        self._check_bar(bar)

        events: list[AlertEvent] = []
        for rule in self._rules:
            if rule.enabled:
                events.extend(self._evaluate_rule(rule, observation))

        self._last_start = bar.start_time_utc
        self._events.extend(events)
        return tuple(events)

    def _check_bar(self, bar: Bar) -> None:
        if not bar.is_final:
            raise InvariantViolation(
                f"refusing a partial bar at {bar.start_time_utc.isoformat()}: intrabar "
                "alerting is not implemented, and a provisional trigger that a closing "
                "bar contradicts would still have started a cooldown (ADR 0011)"
            )
        if bar.symbol != self._symbol:
            raise InvariantViolation(
                f"bar belongs to a different series — this engine follows {self._symbol}, "
                f"the bar is {bar.symbol}"
            )
        if self._timeframe is not None and bar.timeframe is not self._timeframe:
            raise InvariantViolation(
                f"bar timeframe {bar.timeframe.code} does not match the rules' timeframe "
                f"{self._timeframe.code}"
            )

        key: _SeriesKey = (bar.symbol, bar.timeframe, bar.feed, bar.adjustment_mode)
        if self._series is None:
            self._series = key
        elif key != self._series:
            held = self._series
            raise InvariantViolation(
                f"bar belongs to a different series — this engine follows {held[0]} "
                f"{held[1].code} {held[2].value} {held[3].value}, the bar is {key[0]} "
                f"{key[1].code} {key[2].value} {key[3].value}"
            )

        if self._last_start is not None and bar.start_time_utc <= self._last_start:
            raise InvariantViolation(
                f"bar starts at {bar.start_time_utc.isoformat()}, at or before the last "
                f"evaluated bar at {self._last_start.isoformat()}; bar starts must be "
                "strictly increasing. A duplicate, late or revised bar means rebuilding, "
                "not replaying into cooldown state"
            )

    def _evaluate_rule(
        self, rule: AlertRule, observation: AlertObservation
    ) -> tuple[AlertEvent, ...]:
        state = self._states[rule.id]
        level = observation.level(rule.scope.level_id)
        now = observation.bar.end_time_utc

        readings = read_conditions(rule, observation, inside_zone=state.inside_zone)
        state = self._rearmed(rule, state, level=level, observation=observation, now=now)

        events: list[AlertEvent] = []
        for reading in readings:
            if reading.holds is None:
                continue
            if not reading.holds:
                state = dataclasses.replace(state, consecutive=0, reset_seen=True)
                continue

            # A discrete condition is its own edge every time it happens: one bar
            # can walk two rungs of the trend ladder (ADR 0010 D6), and the
            # second must be judged, not absorbed into a running count.
            held = 1 if is_discrete(rule.condition.event_type) else state.consecutive + 1
            state = dataclasses.replace(state, consecutive=held)
            # Strictly `==`: the edge is reported once, and a condition that goes
            # on holding says nothing new (ADR 0011 D3).
            if state.consecutive != rule.confirmation.bars:
                continue

            state, event = self._emit(rule, state, reading, observation, now=now)
            events.append(event)

        if level is not None:
            state = dataclasses.replace(state, inside_zone=level.contains(observation.bar.close))
        self._states[rule.id] = state
        return tuple(events)

    # -- gates -------------------------------------------------------------

    def _rearmed(
        self,
        rule: AlertRule,
        state: RuleState,
        *,
        level: WatchedLevel | None,
        observation: AlertObservation,
        now: datetime,
    ) -> RuleState:
        """Re-arm the rule if its re-arm policy is satisfied (ADR 0011 D5).

        Independent of cooldown: a rule can be armed and still silent. Keeping
        the two gates apart is what lets a suppressed event name which one held
        it back.
        """
        if state.armed or state.triggered_at is None:
            return state
        if not self._is_rearm_ready(rule, state, level=level, observation=observation, now=now):
            return state
        return dataclasses.replace(state, armed=True, reset_seen=False)

    def _is_rearm_ready(
        self,
        rule: AlertRule,
        state: RuleState,
        *,
        level: WatchedLevel | None,
        observation: AlertObservation,
        now: datetime,
    ) -> bool:
        triggered_at = state.triggered_at
        assert triggered_at is not None  # noqa: S101 — checked by the caller

        match rule.rearm.mode:
            case RearmMode.ON_STATE_CHANGE:
                return state.reset_seen
            case RearmMode.AFTER_TIME:
                window = rule.rearm.window
                assert window is not None  # noqa: S101 — RearmPolicy requires it
                # Both, because specs/09-alerts.md says both: a window alone
                # would be a second cooldown wearing a different name.
                return state.reset_seen and now >= triggered_at + window
            case RearmMode.ON_ZONE_EXIT:
                return self._left_the_zone(rule, level=level, observation=observation)

    def _left_the_zone(
        self, rule: AlertRule, *, level: WatchedLevel | None, observation: AlertObservation
    ) -> bool:
        hysteresis = rule.rearm.hysteresis
        assert hysteresis is not None  # noqa: S101 — RearmPolicy requires it
        if level is None:
            return False
        close = observation.bar.close
        distance = resolve_threshold(hysteresis, price=close, atr=observation.atr)
        if distance is None:
            return False
        return level.distance_to(close) >= distance

    def _status(
        self, rule: AlertRule, state: RuleState, *, key: str, now: datetime
    ) -> EmissionStatus:
        """Which gate, if any, this emission fails (ADR 0011 D6)."""
        if key in self._ordinals:
            return EmissionStatus.SUPPRESSED_DUPLICATE
        if state.triggered_at is not None and rule.cooldown.is_active(
            triggered_at=state.triggered_at, now=now
        ):
            return EmissionStatus.SUPPRESSED_COOLDOWN
        if not state.armed:
            return EmissionStatus.SUPPRESSED_DISARMED
        return EmissionStatus.TRIGGERED

    # -- emission ----------------------------------------------------------

    def _emit(
        self,
        rule: AlertRule,
        state: RuleState,
        reading: ConditionReading,
        observation: AlertObservation,
        *,
        now: datetime,
    ) -> tuple[RuleState, AlertEvent]:
        bar = observation.bar
        event_type = rule.condition.event_type
        key = event_key(rule.id, event_type, bar.start_time_utc)
        status = self._status(rule, state, key=key, now=now)

        ordinal = self._ordinals.get(key, 0)
        self._ordinals[key] = ordinal + 1

        cooldown_until = (
            rule.cooldown.expires_at(state.triggered_at).isoformat()
            if state.triggered_at is not None
            else None
        )
        trend_event = reading.trend_event
        event = AlertEvent(
            id=alert_event_id(key, ordinal),
            rule_id=rule.id,
            symbol=bar.symbol,
            timeframe=bar.timeframe,
            event_type=event_type,
            occurred_at=now,
            price=bar.close,
            reason_codes=_reason_codes(rule, reading, status),
            input_snapshot=_snapshot(
                rule,
                reading,
                observation,
                status=status,
                cooldown_until=cooldown_until,
            ),
            status=status,
            level_id=reading.level.id if reading.level is not None else None,
            trend_state_before=trend_event.from_phase if trend_event is not None else None,
            trend_state_after=trend_event.to_phase if trend_event is not None else None,
        )

        if status is EmissionStatus.TRIGGERED:
            state = dataclasses.replace(state, armed=False, triggered_at=now, reset_seen=False)
        return state, event


def _reason_codes(
    rule: AlertRule, reading: ConditionReading, status: EmissionStatus
) -> tuple[str, ...]:
    """Symbolic, with the numbers one field away in the snapshot (ADR 0011 D10)."""
    codes = [rule.condition.event_type.value, *reading.reason_codes]
    if rule.confirmation.bars > 1:
        codes.append(f"CONFIRMED_{rule.confirmation.bars}_BARS")
    suppression = _SUPPRESSION_CODES.get(status)
    if suppression is not None:
        codes.append(suppression)
    return tuple(codes)


def _snapshot(
    rule: AlertRule,
    reading: ConditionReading,
    observation: AlertObservation,
    *,
    status: EmissionStatus,
    cooldown_until: str | None,
) -> dict[str, Any]:
    """The audit record specs/17-observability.md asks every alert to carry.

    Exact-domain values are strings, per specs/12-storage.md: a snapshot is
    persistence-shaped, and a price that becomes a JSON number has already lost.
    """
    bar = observation.bar
    snapshot: dict[str, Any] = {
        "bar_start_utc": bar.start_time_utc.isoformat(),
        "bar_end_utc": bar.end_time_utc.isoformat(),
        "open": format_exact(bar.open),
        "high": format_exact(bar.high),
        "low": format_exact(bar.low),
        "close": format_exact(bar.close),
        "atr": None if observation.atr is None else format_exact(observation.atr),
        "status": status.value,
        "confirmation": {
            "required": rule.confirmation.bars,
            "event_type": rule.condition.event_type.value,
        },
        "cooldown_seconds": rule.cooldown.duration.total_seconds(),
        "cooldown_until": cooldown_until,
        "rearm": rule.rearm.mode.value,
        "rule_id": rule.id,
        "rule_fingerprint": rule_fingerprint(rule),
    }
    snapshot.update(reading.detail)
    return snapshot
