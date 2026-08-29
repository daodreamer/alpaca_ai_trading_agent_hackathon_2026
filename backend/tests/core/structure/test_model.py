"""Invariants of what the structure engine emits — ADR 0008 D3/D7.

These are the claims a consumer is allowed to rely on without checking: a swing
was not known before it happened, a confirmed swing had right-side bars, a break
came after the level it broke.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from alphagate.core.errors import InvariantViolation
from alphagate.core.structure import (
    BreakEvent,
    BreakKind,
    BreakPolicy,
    StructureLabel,
    SwingKind,
    SwingPoint,
    SwingStatus,
)
from tests.core.structure.synthetic import start_of


def swing(**overrides: object) -> SwingPoint:
    fields: dict[str, object] = {
        "kind": SwingKind.HIGH,
        "status": SwingStatus.CONFIRMED,
        "bar_start_utc": start_of(1),
        "detected_at_utc": start_of(2),
        "price": "12",
    }
    return SwingPoint(**{**fields, **overrides})  # type: ignore[arg-type]


def break_event(**overrides: object) -> BreakEvent:
    fields: dict[str, object] = {
        "kind": BreakKind.BREAKOUT,
        "level_price": "12",
        "level_bar_start_utc": start_of(1),
        "bar_start_utc": start_of(4),
        "closes": 1,
    }
    return BreakEvent(**{**fields, **overrides})  # type: ignore[arg-type]


def test_a_swing_price_must_be_positive() -> None:
    with pytest.raises(InvariantViolation, match="positive"):
        swing(price="0")


def test_a_swing_cannot_be_detected_before_its_pivot() -> None:
    with pytest.raises(InvariantViolation, match="before it happened"):
        swing(bar_start_utc=start_of(5), detected_at_utc=start_of(4))


def test_a_confirmed_swing_cannot_be_detected_on_its_own_bar() -> None:
    with pytest.raises(InvariantViolation, match="right-side bars"):
        swing(bar_start_utc=start_of(3), detected_at_utc=start_of(3))


def test_a_candidate_may_be_detected_on_its_own_bar() -> None:
    candidate = swing(
        status=SwingStatus.CANDIDATE, bar_start_utc=start_of(3), detected_at_utc=start_of(3)
    )

    assert candidate.detected_at_utc == candidate.bar_start_utc


def test_a_candidate_carries_no_label() -> None:
    with pytest.raises(InvariantViolation, match="candidate"):
        swing(
            status=SwingStatus.CANDIDATE,
            bar_start_utc=start_of(3),
            detected_at_utc=start_of(3),
            label=StructureLabel.HH,
        )


def test_a_swing_price_is_quantized_on_construction() -> None:
    assert str(swing(price="12.123456789").price) == "12.12345679"


def test_a_break_must_follow_the_level_it_broke() -> None:
    with pytest.raises(InvariantViolation, match="cannot precede"):
        break_event(level_bar_start_utc=start_of(4), bar_start_utc=start_of(4))


def test_a_break_needs_at_least_one_close() -> None:
    with pytest.raises(InvariantViolation, match="closes"):
        break_event(closes=0)


def test_break_events_are_frozen_facts() -> None:
    event = break_event()

    with pytest.raises(AttributeError):
        event.closes = 2  # type: ignore[misc]


def test_a_swing_carries_a_lag_its_consumer_can_measure() -> None:
    point = swing(bar_start_utc=start_of(1), detected_at_utc=start_of(4))

    assert point.detected_at_utc - point.bar_start_utc == timedelta(minutes=3)


# -- break policy -----------------------------------------------------------


@pytest.mark.parametrize("closes", [0, -1])
def test_a_policy_needs_at_least_one_close(closes: int) -> None:
    with pytest.raises(InvariantViolation, match="closes_required"):
        BreakPolicy(closes_required=closes)


def test_a_policy_volume_period_must_be_positive() -> None:
    with pytest.raises(InvariantViolation, match="volume_period"):
        BreakPolicy(volume_period=0)


@pytest.mark.parametrize("multiple", [0.0, -1.5])
def test_a_volume_multiple_must_be_positive_when_set(multiple: float) -> None:
    with pytest.raises(InvariantViolation, match="volume_multiple"):
        BreakPolicy(volume_multiple=multiple)


def test_a_negative_noise_threshold_is_refused() -> None:
    from alphagate.core.structure import StructureEngine

    with pytest.raises(InvariantViolation, match="minimum_move_atr"):
        StructureEngine(minimum_move_atr=-0.5)
