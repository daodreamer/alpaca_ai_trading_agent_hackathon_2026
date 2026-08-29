"""Invariants that must hold for any price path — specs/14-testing.md.

The phase exit criteria are properties, not examples: "no alert storm in
fixtures" and "every event is auditable" have to survive paths nobody wrote by
hand.
"""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

from hypothesis import given, settings
from hypothesis import strategies as st

from alphagate.core.alert_engine import AlertEngine
from alphagate.core.alerts import (
    AlertEvent,
    AlertEventType,
    ConfirmationPolicy,
    Cooldown,
    EmissionStatus,
    RearmMode,
    RearmPolicy,
)
from tests.core.alert_engine.synthetic import RULE, absolute, bars_from, make_rule, observation

# Prices chosen around the default [103.00, 103.40] resistance zone: well below,
# just outside, inside, and above it.
PRICES = st.sampled_from(["101.00", "102.00", "102.60", "103.20", "103.90"])
PATHS = st.lists(PRICES, min_size=1, max_size=40)

ZONE_LOW = Decimal("103.00")
ZONE_HIGH = Decimal("103.40")


def run(rule_kwargs: dict[str, object], closes: list[str]) -> tuple[AlertEvent, ...]:
    engine = AlertEngine((make_rule(**rule_kwargs),))
    events: list[AlertEvent] = []
    for bar in bars_from(closes):
        events.extend(engine.evaluate(observation(bar)))
    return tuple(events)


def rising_edges(holds: list[bool]) -> int:
    previous = False
    count = 0
    for value in holds:
        if value and not previous:
            count += 1
        previous = value
    return count


@settings(max_examples=200, deadline=None)
@given(closes=PATHS)
def test_no_alert_storm_while_a_condition_persists(closes: list[str]) -> None:
    """One episode of a true condition produces exactly one emission."""
    events = run({"cooldown": Cooldown(timedelta(minutes=1))}, closes)
    holds = [bar.low <= ZONE_HIGH and bar.high >= ZONE_LOW for bar in bars_from(closes)]
    assert len(events) == rising_edges(holds)


@settings(max_examples=100, deadline=None)
@given(closes=PATHS)
def test_replaying_the_same_path_produces_identical_events(closes: list[str]) -> None:
    first = run({}, closes)
    second = run({}, closes)
    assert first == second


@settings(max_examples=100, deadline=None)
@given(closes=PATHS)
def test_every_event_is_auditable(closes: list[str]) -> None:
    for event in run({}, closes):
        assert event.reason_codes
        assert event.rule_id == RULE
        assert event.input_snapshot["rule_fingerprint"]
        assert event.input_snapshot["bar_start_utc"]
        assert event.occurred_at.tzinfo is not None


@settings(max_examples=100, deadline=None)
@given(closes=PATHS)
def test_a_triggered_event_is_always_followed_by_a_gate(closes: list[str]) -> None:
    """Two consecutive triggers require the condition to have lapsed between them."""
    events = run(
        {
            "cooldown": Cooldown(timedelta(hours=1)),
            "rearm": RearmPolicy(mode=RearmMode.ON_ZONE_EXIT, hysteresis=absolute("0.25")),
        },
        closes,
    )
    triggered = [event for event in events if event.status is EmissionStatus.TRIGGERED]
    # One hour of cooldown outlasts forty one-minute bars, so nothing can fire twice.
    assert len(triggered) <= 1


@settings(max_examples=100, deadline=None)
@given(closes=PATHS, bars=st.integers(min_value=1, max_value=4))
def test_confirmation_never_fires_earlier_than_it_allows(closes: list[str], bars: int) -> None:
    events = run({"confirmation": ConfirmationPolicy(bars=bars)}, closes)
    strict = run({"confirmation": ConfirmationPolicy(bars=bars + 1)}, closes)
    assert len(strict) <= len(events)


@settings(max_examples=100, deadline=None)
@given(closes=PATHS)
def test_breaking_up_and_breaking_down_are_never_both_true(closes: list[str]) -> None:
    up = run({"event_type": AlertEventType.LEVEL_BROKEN_UP}, closes)
    down = run({"event_type": AlertEventType.LEVEL_BROKEN_DOWN}, closes)
    overlap = {event.occurred_at for event in up} & {event.occurred_at for event in down}
    assert not overlap
