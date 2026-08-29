"""Condition predicates — specs/09-alerts.md, ADR 0011 D2.

Pure functions over one observation. Each returns what the condition says about
this bar and the evidence for saying it; nothing here knows about cooldown,
arming or delivery.

Two price conventions are borrowed rather than invented, because the same
question is already answered elsewhere in the Domain:

* a **touch** is the bar's range against the zone, as in `count_touch_episodes`
  (ADR 0009 D6) — a wick into a zone is a test of it;
* a **break** is the close beyond the zone, as in `invalidate` (ADR 0009 D8) — a
  wick through a level is not a break of it.
"""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal
from typing import Any, Final

from alphagate.core.alert_engine.model import AlertObservation, ConditionReading, WatchedLevel
from alphagate.core.alerts import AlertEventType, AlertRule, ProximityKind, ProximityThreshold
from alphagate.core.errors import InvariantViolation
from alphagate.core.numeric import DOMAIN_CONTEXT, format_exact
from alphagate.core.numeric import price as exact_price
from alphagate.core.trend_engine import TrendEvent, TrendEventKind

__all__ = [
    "is_discrete",
    "is_supported",
    "read_conditions",
    "resolve_threshold",
]

_DISCRETE: Final = frozenset(
    {
        AlertEventType.LEVEL_ZONE_EXITED,
        AlertEventType.TREND_CONFIRMED,
        AlertEventType.TREND_REVERSAL,
        AlertEventType.TREND_STRENGTH_INCREASED,
        AlertEventType.TREND_STRENGTH_DECREASED,
    }
)
"""Conditions that happen rather than persist (ADR 0011 D2/D4)."""

_UNSUPPORTED: Final = frozenset({AlertEventType.BREAKOUT_CONFIRMED, AlertEventType.BREAKOUT_FAILED})
"""In `AlertEventType` because specs/02-architecture.md lists them; not among the
MVP alert types in specs/09-alerts.md, and failed breakouts are F3 work."""

_TREND_KINDS: Final[Mapping[AlertEventType, TrendEventKind]] = {
    AlertEventType.TREND_CONFIRMED: TrendEventKind.TREND_CONFIRMED,
    AlertEventType.TREND_REVERSAL: TrendEventKind.TREND_REVERSAL,
    AlertEventType.TREND_STRENGTH_INCREASED: TrendEventKind.TREND_STRENGTH_INCREASED,
    AlertEventType.TREND_STRENGTH_DECREASED: TrendEventKind.TREND_STRENGTH_DECREASED,
}


def is_discrete(event_type: AlertEventType) -> bool:
    """Whether the condition is momentary, and so cannot be confirmed over bars."""
    return event_type in _DISCRETE


def is_supported(event_type: AlertEventType) -> bool:
    return event_type not in _UNSUPPORTED


def resolve_threshold(
    threshold: ProximityThreshold, *, price: Decimal, atr: Decimal | None
) -> Decimal | None:
    """The threshold in price terms, or `None` when it cannot be known yet.

    `None` happens only in ATR mode before ATR is warm. Substituting a made-up
    distance would be inventing the very number the user configured.
    """
    match threshold.kind:
        case ProximityKind.ABSOLUTE:
            return threshold.value
        case ProximityKind.PERCENT:
            return exact_price(
                DOMAIN_CONTEXT.divide(
                    DOMAIN_CONTEXT.multiply(price, threshold.value), Decimal(100)
                ),
                field="threshold",
            )
        case ProximityKind.ATR_MULTIPLE:
            if atr is None:
                return None
            return exact_price(DOMAIN_CONTEXT.multiply(atr, threshold.value), field="threshold")


def read_conditions(
    rule: AlertRule, observation: AlertObservation, *, inside_zone: bool
) -> tuple[ConditionReading, ...]:
    """What `rule`'s condition says about this bar.

    Usually one reading. A trend rule returns one per matching `TrendEvent`,
    because a bar that walks two rungs of the ladder raises two events and the
    engine has to decide about each (ADR 0010 D6).
    """
    event_type = rule.condition.event_type
    if not is_supported(event_type):  # pragma: no cover — refused at registration
        raise InvariantViolation(f"{event_type.value}: no MVP definition")
    if event_type.is_trend_event:
        return _read_trend(rule, observation)
    return (_read_level(rule, observation, inside_zone=inside_zone),)


# -- levels -----------------------------------------------------------------


def _read_level(
    rule: AlertRule, observation: AlertObservation, *, inside_zone: bool
) -> ConditionReading:
    level = observation.level(rule.scope.level_id)
    if level is None:
        return ConditionReading(holds=None, reason_codes=("LEVEL_NOT_IN_FORCE",))

    close = observation.bar.close
    distance = level.distance_to(close)
    detail: dict[str, Any] = {"level": _level_detail(level, distance=distance)}

    match rule.condition.event_type:
        case AlertEventType.LEVEL_APPROACHING:
            return _read_approaching(rule, observation, level=level, distance=distance)
        case AlertEventType.LEVEL_TOUCHED:
            bar = observation.bar
            holds = bar.low <= level.zone.high and bar.high >= level.zone.low
            code = "BAR_RANGE_INTERSECTS_ZONE"
        case AlertEventType.LEVEL_BROKEN_UP:
            holds = close > level.zone.high
            code = "CLOSE_ABOVE_ZONE_HIGH"
        case AlertEventType.LEVEL_BROKEN_DOWN:
            holds = close < level.zone.low
            code = "CLOSE_BELOW_ZONE_LOW"
        case AlertEventType.LEVEL_ZONE_EXITED:
            holds = inside_zone and not level.contains(close)
            code = "CLOSE_LEFT_ZONE"
        case unreachable:  # pragma: no cover — every level event is handled above
            raise InvariantViolation(f"{unreachable.value}: not a level condition")

    return ConditionReading(
        holds=holds,
        reason_codes=(code,) if holds else (),
        level=level,
        detail=detail,
    )


def _read_approaching(
    rule: AlertRule,
    observation: AlertObservation,
    *,
    level: WatchedLevel,
    distance: Decimal,
) -> ConditionReading:
    threshold = rule.condition.proximity
    if threshold is None:  # pragma: no cover — AlertCondition requires one
        raise InvariantViolation("LEVEL_APPROACHING needs a proximity threshold")

    resolved = resolve_threshold(threshold, price=observation.bar.close, atr=observation.atr)
    detail: dict[str, Any] = {
        "level": _level_detail(level, distance=distance),
        "threshold": {
            "kind": threshold.kind.value,
            "value": format_exact(threshold.value),
            "resolved": None if resolved is None else format_exact(resolved),
        },
    }
    if resolved is None:
        return ConditionReading(
            holds=None,
            reason_codes=("THRESHOLD_NOT_RESOLVABLE",),
            level=level,
            detail=detail,
        )

    # Zero distance means price is *in* the zone, which is a touch and not an
    # approach; a rule that wanted both would ask for both.
    holds = Decimal(0) < distance <= resolved
    return ConditionReading(
        holds=holds,
        reason_codes=("CLOSE_WITHIN_PROXIMITY",) if holds else (),
        level=level,
        detail=detail,
    )


def _level_detail(level: WatchedLevel, *, distance: Decimal) -> Mapping[str, Any]:
    return {
        "id": level.id,
        "kind": level.kind.value,
        "price": format_exact(level.price),
        "zone_low": format_exact(level.zone.low),
        "zone_high": format_exact(level.zone.high),
        "distance": format_exact(distance),
    }


# -- trend ------------------------------------------------------------------


def _read_trend(rule: AlertRule, observation: AlertObservation) -> tuple[ConditionReading, ...]:
    state = observation.trend
    if state is None:
        return (ConditionReading(holds=None, reason_codes=("TREND_NOT_AVAILABLE",)),)

    kind = _TREND_KINDS[rule.condition.event_type]
    matching = tuple(event for event in observation.trend_events if event.kind is kind)
    if not matching:
        return (ConditionReading(holds=False, detail={"trend": _trend_detail(observation)}),)

    return tuple(_read_trend_event(rule, observation, event=event) for event in matching)


def _read_trend_event(
    rule: AlertRule, observation: AlertObservation, *, event: TrendEvent
) -> ConditionReading:
    threshold = rule.condition.strength_threshold
    state = observation.trend
    assert state is not None  # noqa: S101 — checked by the caller

    holds = True
    codes = [event.kind.value, *event.reason_codes]
    if threshold is not None:
        if rule.condition.event_type is AlertEventType.TREND_STRENGTH_INCREASED:
            holds = state.strength >= threshold
            codes.append("STRENGTH_AT_OR_ABOVE_THRESHOLD")
        else:
            holds = state.strength <= threshold
            codes.append("STRENGTH_AT_OR_BELOW_THRESHOLD")

    return ConditionReading(
        holds=holds,
        reason_codes=tuple(codes) if holds else (),
        trend_event=event,
        detail={
            "trend": _trend_detail(observation),
            "trend_event": {
                "kind": event.kind.value,
                "from": event.from_phase.value,
                "to": event.to_phase.value,
                "strength": event.strength,
                "bar_start_utc": event.bar_start_utc.isoformat(),
                "input_snapshot_hash": event.input_snapshot_hash,
            },
            "strength_threshold": threshold,
        },
    )


def _trend_detail(observation: AlertObservation) -> Mapping[str, Any] | None:
    state = observation.trend
    if state is None:  # pragma: no cover — callers check first
        return None
    return {
        "phase": state.state.value,
        "strength": state.strength,
        "confidence": state.confidence,
        "as_of": state.as_of.isoformat(),
        "input_snapshot_hash": state.input_snapshot_hash,
    }
