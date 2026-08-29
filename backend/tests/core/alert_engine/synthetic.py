"""Bars, levels, rules and observations for alert tests.

An alert test is usually "this price did this thing near this zone, and this is
what the rule should say about it", so the helpers take the price path and a
zone and fill in everything else with the least interesting valid value.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Any

from alphagate.core.alert_engine import AlertObservation, WatchedLevel
from alphagate.core.alerts import (
    AlertCondition,
    AlertEventType,
    AlertRule,
    AlertScope,
    ConfirmationPolicy,
    Cooldown,
    NotificationChannel,
    ProximityKind,
    ProximityThreshold,
    RearmMode,
    RearmPolicy,
)
from alphagate.core.bar import AdjustmentMode, Bar, Feed
from alphagate.core.identifiers import AlertRuleId, LevelId, Ticker, UserId, ticker
from alphagate.core.levels import LevelKind, Zone
from alphagate.core.time_model import SessionKind, Timeframe

AAPL: Ticker = ticker("AAPL")
USER: UserId = UserId("u-1")
RULE: AlertRuleId = AlertRuleId("r-1")
ZONE_ID: LevelId = LevelId("lvl-1")

SESSION_DATE = date(2026, 8, 13)
SESSION_OPEN = datetime(2026, 8, 13, 13, 30, tzinfo=UTC)

MINUTE = timedelta(minutes=1)


def start_of(index: int) -> datetime:
    return SESSION_OPEN + index * MINUTE


def end_of(index: int) -> datetime:
    return start_of(index) + MINUTE


def bar_at(
    index: int,
    *,
    open_: str,
    high: str,
    low: str,
    close: str,
    volume: str = "1000",
    is_final: bool = True,
    symbol: Ticker = AAPL,
    timeframe: Timeframe = Timeframe.M1,
    feed: Feed = Feed.SYNTHETIC,
) -> Bar:
    return Bar(
        symbol=symbol,
        timeframe=timeframe,
        start_time_utc=start_of(index),
        end_time_utc=end_of(index),
        session_date=SESSION_DATE,
        session=SessionKind.REGULAR,
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
        source="test",
        feed=feed,
        adjustment_mode=AdjustmentMode.UNADJUSTED,
        is_final=is_final,
    )


def bars_from(closes: Sequence[str], *, margin: str = "0.10") -> tuple[Bar, ...]:
    """One bar per close, spanning `close ± margin` and opening at its close."""
    step = Decimal(margin)
    return tuple(
        bar_at(
            index,
            open_=close,
            high=str(Decimal(close) + step),
            low=str(Decimal(close) - step),
            close=close,
        )
        for index, close in enumerate(closes)
    )


def resistance(
    low: str = "103.00", high: str = "103.40", *, level_id: LevelId = ZONE_ID
) -> WatchedLevel:
    zone = Zone(low=low, high=high)
    return WatchedLevel(id=level_id, kind=LevelKind.RESISTANCE, price=zone.midpoint, zone=zone)


def observation(bar: Bar, **overrides: Any) -> AlertObservation:
    defaults: dict[str, Any] = {"bar": bar, "levels": (resistance(),)}
    return AlertObservation(**{**defaults, **overrides})


def absolute(value: str) -> ProximityThreshold:
    return ProximityThreshold(kind=ProximityKind.ABSOLUTE, value=Decimal(value))


def make_rule(**overrides: Any) -> AlertRule:
    """A level rule on the default resistance zone, armed and unconstrained."""
    event_type = overrides.pop("event_type", AlertEventType.LEVEL_TOUCHED)
    condition = overrides.pop(
        "condition",
        AlertCondition(
            event_type=event_type,
            proximity=(
                absolute("0.50") if event_type is AlertEventType.LEVEL_APPROACHING else None
            ),
            strength_threshold=(
                50.0
                if event_type
                in (
                    AlertEventType.TREND_STRENGTH_INCREASED,
                    AlertEventType.TREND_STRENGTH_DECREASED,
                )
                else None
            ),
        ),
    )
    scope = overrides.pop(
        "scope",
        AlertScope(
            symbol=AAPL,
            timeframe=Timeframe.M1,
            level_id=ZONE_ID if condition.event_type.is_level_event else None,
        ),
    )
    defaults: dict[str, Any] = {
        "id": RULE,
        "user_id": USER,
        "scope": scope,
        "condition": condition,
        "confirmation": ConfirmationPolicy(bars=1),
        "cooldown": Cooldown(timedelta(minutes=5)),
        "rearm": RearmPolicy(mode=RearmMode.ON_STATE_CHANGE),
        "channels": frozenset({NotificationChannel.TELEGRAM}),
    }
    return AlertRule(**{**defaults, **overrides})
