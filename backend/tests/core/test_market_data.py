"""Market data port value objects — specs/03-market-data.md."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from alphagate.core.bar import AdjustmentMode, Feed
from alphagate.core.errors import InvariantViolation
from alphagate.core.identifiers import ticker
from alphagate.core.market_data import (
    BarRange,
    FeedHealth,
    FeedStatus,
    ProviderCapabilities,
)
from alphagate.core.time_model import Timeframe

AAPL = ticker("AAPL")
NOW = datetime(2026, 11, 25, 15, 0, tzinfo=UTC)


class TestFeedStatus:
    def test_records_when_the_last_message_arrived(self) -> None:
        status = FeedStatus(
            health=FeedHealth.DELAYED,
            feed=Feed.IEX,
            as_of=NOW,
            last_message_at=NOW - timedelta(seconds=20),
        )
        assert status.last_message_at == NOW - timedelta(seconds=20)

    def test_last_message_is_optional(self) -> None:
        status = FeedStatus(health=FeedHealth.DISCONNECTED, feed=Feed.IEX, as_of=NOW)
        assert status.last_message_at is None

    def test_timestamps_must_be_aware(self) -> None:
        with pytest.raises(InvariantViolation, match="timezone"):
            FeedStatus(
                health=FeedHealth.LIVE,
                feed=Feed.IEX,
                as_of=datetime(2026, 11, 25, 15, 0),  # noqa: DTZ001
            )

    def test_last_message_timestamp_must_be_aware(self) -> None:
        with pytest.raises(InvariantViolation, match="timezone"):
            FeedStatus(
                health=FeedHealth.LIVE,
                feed=Feed.IEX,
                as_of=NOW,
                last_message_at=datetime(2026, 11, 25, 14, 0),  # noqa: DTZ001
            )

    def test_delayed_is_distinguishable_from_live(self) -> None:
        # specs/03-market-data.md: the UI must be able to stop a user believing
        # delayed data is real-time.
        live = FeedStatus(health=FeedHealth.LIVE, feed=Feed.SIP, as_of=NOW)
        delayed = FeedStatus(health=FeedHealth.DELAYED, feed=Feed.IEX, as_of=NOW)
        assert live.health is not delayed.health


class TestProviderCapabilities:
    def make(self, **overrides: object) -> ProviderCapabilities:
        defaults: dict[str, object] = {
            "name": "alpaca",
            "feeds": frozenset({Feed.IEX}),
            "timeframes": frozenset({Timeframe.M1, Timeframe.D1}),
            "adjustment_modes": frozenset({AdjustmentMode.ADJUSTED}),
        }
        return ProviderCapabilities(**{**defaults, **overrides})  # type: ignore[arg-type]

    def test_reports_supported_timeframes(self) -> None:
        caps = self.make()
        assert caps.supports(Timeframe.M1)
        assert not caps.supports(Timeframe.H4)

    def test_a_provider_must_expose_a_feed(self) -> None:
        with pytest.raises(InvariantViolation, match="feed"):
            self.make(feeds=frozenset())

    def test_a_provider_must_support_a_timeframe(self) -> None:
        with pytest.raises(InvariantViolation, match="timeframe"):
            self.make(timeframes=frozenset())

    def test_extended_hours_and_streaming_default_off(self) -> None:
        # Capability is claimed, never assumed.
        caps = self.make()
        assert not caps.supports_extended_hours
        assert not caps.supports_streaming

    def test_earliest_available_is_optional(self) -> None:
        assert self.make().earliest_available is None
        assert self.make(earliest_available=date(2016, 1, 4)).earliest_available == date(2016, 1, 4)


class TestBarRange:
    def test_is_half_open(self) -> None:
        window = BarRange(
            symbol=AAPL, timeframe=Timeframe.H1, start=NOW, end=NOW + timedelta(hours=2)
        )
        assert window.contains(NOW)
        assert window.contains(NOW + timedelta(hours=1, minutes=59))
        assert not window.contains(NOW + timedelta(hours=2))

    def test_end_must_follow_start(self) -> None:
        with pytest.raises(InvariantViolation, match="end"):
            BarRange(symbol=AAPL, timeframe=Timeframe.H1, start=NOW, end=NOW)

    def test_timestamps_must_be_aware(self) -> None:
        with pytest.raises(InvariantViolation, match="timezone"):
            BarRange(
                symbol=AAPL,
                timeframe=Timeframe.H1,
                start=datetime(2026, 11, 25, 15, 0),  # noqa: DTZ001
                end=NOW + timedelta(hours=1),
            )
