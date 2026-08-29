"""Provider sync state and notification delivery — ADR 0012.

Two small records that `specs/12-storage.md` names as tables and no spec models.
Both are operational facts rather than market facts: where a backfill got to, and
whether a notification went out.

`NotificationDelivery` lives in the Domain and knows nothing about Telegram,
email or webhooks. It is the record that a delivery was attempted, which is what
`specs/02-architecture.md`'s "delivery must be idempotent per
`alert_event_id + channel`" needs somewhere to be written down. The adapters that
actually send arrive in Phase 14.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime
from enum import Enum

from alphagate.core.alerts import NotificationChannel
from alphagate.core.bar import AdjustmentMode, Feed
from alphagate.core.errors import InvariantViolation
from alphagate.core.identifiers import AlertEventId, Ticker
from alphagate.core.time_model import Timeframe, ensure_utc

__all__ = ["DeliveryStatus", "NotificationDelivery", "ProviderSyncState"]


@dataclasses.dataclass(frozen=True, slots=True)
class ProviderSyncState:
    """How far a provider has been synced for one series.

    Keyed by the full series identity rather than by symbol alone: the same
    symbol on two feeds is two backfills, for the same reason it is two bar
    series (ADR 0004).
    """

    provider: str
    symbol: Ticker
    timeframe: Timeframe
    feed: Feed
    adjustment_mode: AdjustmentMode
    synced_at_utc: datetime
    last_bar_start_utc: datetime | None = None
    detail: str | None = None

    def __post_init__(self) -> None:
        if not self.provider.strip():
            raise InvariantViolation("provider is required")
        object.__setattr__(
            self, "synced_at_utc", ensure_utc(self.synced_at_utc, field="synced_at_utc")
        )
        if self.last_bar_start_utc is not None:
            object.__setattr__(
                self,
                "last_bar_start_utc",
                ensure_utc(self.last_bar_start_utc, field="last_bar_start_utc"),
            )


class DeliveryStatus(Enum):
    PENDING = "PENDING"
    SENT = "SENT"
    FAILED = "FAILED"


@dataclasses.dataclass(frozen=True, slots=True)
class NotificationDelivery:
    """One attempt to deliver one alert event to one channel.

    Identity is `(event_id, channel)`, which is exactly the idempotency key
    `specs/02-architecture.md` requires. Retries update the status of this
    record; they do not create a second one.
    """

    event_id: AlertEventId
    channel: NotificationChannel
    status: DeliveryStatus
    attempted_at: datetime
    attempts: int = 1
    detail: str | None = None

    def __post_init__(self) -> None:
        if not self.event_id:
            raise InvariantViolation("a delivery must reference an alert event")
        object.__setattr__(
            self, "attempted_at", ensure_utc(self.attempted_at, field="attempted_at")
        )
        if self.attempts < 1:
            raise InvariantViolation(f"attempts: must be at least 1, got {self.attempts}")
