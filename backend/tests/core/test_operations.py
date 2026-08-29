"""Provider sync state and notification delivery — ADR 0012."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from alphagate.core.alerts import NotificationChannel
from alphagate.core.bar import AdjustmentMode, Feed
from alphagate.core.errors import InvariantViolation
from alphagate.core.identifiers import AlertEventId, ticker
from alphagate.core.operations import DeliveryStatus, NotificationDelivery, ProviderSyncState
from alphagate.core.time_model import Timeframe

MOMENT = datetime(2026, 8, 13, 12, 0, tzinfo=UTC)
AAPL = ticker("AAPL")


def sync_state(**overrides: object) -> ProviderSyncState:
    defaults: dict[str, object] = {
        "provider": "alpaca",
        "symbol": AAPL,
        "timeframe": Timeframe.M1,
        "feed": Feed.IEX,
        "adjustment_mode": AdjustmentMode.UNADJUSTED,
        "synced_at_utc": MOMENT,
    }
    return ProviderSyncState(**{**defaults, **overrides})  # type: ignore[arg-type]


class TestProviderSyncState:
    def test_timestamps_are_normalized_to_utc(self) -> None:
        state = sync_state(last_bar_start_utc=MOMENT)
        assert state.synced_at_utc.tzinfo is not None
        assert state.last_bar_start_utc is not None

    def test_an_empty_provider_is_refused(self) -> None:
        with pytest.raises(InvariantViolation, match="provider"):
            sync_state(provider="  ")

    def test_an_unsynced_series_has_no_last_bar(self) -> None:
        assert sync_state().last_bar_start_utc is None

    def test_the_same_symbol_on_two_feeds_is_two_states(self) -> None:
        """Feed is part of a series identity (ADR 0004), so it is part of this one."""
        assert sync_state(feed=Feed.IEX) != sync_state(feed=Feed.SIP)


class TestNotificationDelivery:
    def test_a_delivery_defaults_to_one_attempt(self) -> None:
        delivery = NotificationDelivery(
            event_id=AlertEventId("e-1"),
            channel=NotificationChannel.TELEGRAM,
            status=DeliveryStatus.SENT,
            attempted_at=MOMENT,
        )
        assert delivery.attempts == 1

    def test_a_delivery_must_reference_an_event(self) -> None:
        with pytest.raises(InvariantViolation, match="alert event"):
            NotificationDelivery(
                event_id=AlertEventId(""),
                channel=NotificationChannel.TELEGRAM,
                status=DeliveryStatus.PENDING,
                attempted_at=MOMENT,
            )

    def test_zero_attempts_is_not_an_attempt(self) -> None:
        with pytest.raises(InvariantViolation, match="attempts"):
            NotificationDelivery(
                event_id=AlertEventId("e-1"),
                channel=NotificationChannel.EMAIL,
                status=DeliveryStatus.FAILED,
                attempted_at=MOMENT,
                attempts=0,
            )
