"""Market data ports — specs/03-market-data.md, ADR 0002.

Split into small protocols rather than one fat `MarketDataProvider`, honouring
the ISP note in the spec: a CSV fixture source can supply history without
pretending to stream, and the replay driver can consume history without
depending on a streaming API that does not exist for it.

Capability is data, not a subclass. Whether a provider has SIP or only IEX,
whether it can serve extended hours, and how far back it goes are runtime facts
that change with a subscription — so they are reported, not encoded in the type.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Iterator, Sequence
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Protocol

from alphagate.core.bar import AdjustmentMode, Bar, Feed
from alphagate.core.errors import InvariantViolation
from alphagate.core.identifiers import Ticker
from alphagate.core.time_model import Timeframe, ensure_utc

__all__ = [
    "BarRange",
    "BarStore",
    "FeedHealth",
    "FeedStatus",
    "HistoricalBarSource",
    "ProviderCapabilities",
    "StreamingBarSource",
    "StreamingTradeSource",
    "Trade",
]


class FeedHealth(Enum):
    """What the UI needs to stop a user believing delayed data is live.

    specs/03-market-data.md is explicit that this must be surfaced.
    """

    LIVE = "LIVE"
    DELAYED = "DELAYED"
    STALE = "STALE"
    """Connected, but no data has arrived when it should have."""

    DISCONNECTED = "DISCONNECTED"


@dataclasses.dataclass(frozen=True, slots=True)
class FeedStatus:
    health: FeedHealth
    feed: Feed
    as_of: datetime
    last_message_at: datetime | None = None
    detail: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "as_of", ensure_utc(self.as_of, field="as_of"))
        if self.last_message_at is not None:
            object.__setattr__(
                self,
                "last_message_at",
                ensure_utc(self.last_message_at, field="last_message_at"),
            )


@dataclasses.dataclass(frozen=True, slots=True)
class ProviderCapabilities:
    """What this provider can actually do, right now, on this subscription."""

    name: str
    feeds: frozenset[Feed]
    timeframes: frozenset[Timeframe]
    adjustment_modes: frozenset[AdjustmentMode]
    supports_extended_hours: bool = False
    supports_streaming: bool = False
    earliest_available: date | None = None

    def __post_init__(self) -> None:
        if not self.feeds:
            raise InvariantViolation(f"{self.name}: a provider must expose at least one feed")
        if not self.timeframes:
            raise InvariantViolation(f"{self.name}: a provider must support at least one timeframe")

    def supports(self, timeframe: Timeframe) -> bool:
        return timeframe in self.timeframes


@dataclasses.dataclass(frozen=True, slots=True)
class BarRange:
    """A half-open request window, matching bar semantics: `[start, end)`."""

    symbol: Ticker
    timeframe: Timeframe
    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "start", ensure_utc(self.start, field="start"))
        object.__setattr__(self, "end", ensure_utc(self.end, field="end"))
        if self.end <= self.start:
            raise InvariantViolation(
                f"end ({self.end.isoformat()}) must follow start ({self.start.isoformat()})"
            )

    def contains(self, instant: datetime) -> bool:
        return self.start <= ensure_utc(instant, field="instant") < self.end


class HistoricalBarSource(Protocol):
    """Anything that can answer for a closed time range."""

    def capabilities(self) -> ProviderCapabilities: ...

    def get_historical_bars(self, request: BarRange) -> Sequence[Bar]:
        """Finalized bars covering `request`, chronological, no duplicates."""
        ...


class BarStore(Protocol):
    """Somewhere canonical bars are kept — ADR 0022, specs/12-storage.md.

    Narrow on purpose, like the ports in `alphagate.core.stores`: a repository can do
    more, and a port exposing all of it would make every in-memory
    implementation a database emulator. Read a range, write some bars.

    A series is identified by all four of symbol, timeframe, feed and adjustment
    mode. Two of those are easy to drop and both are wrong to: an IEX bar and a
    SIP bar are different observations, and mixing adjusted with unadjusted is
    how a split becomes a fake breakout.
    """

    def get_range(
        self,
        *,
        symbol: Ticker,
        timeframe: Timeframe,
        feed: Feed,
        adjustment_mode: AdjustmentMode,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> tuple[Bar, ...]:
        """Bars in `[start, end)`, chronologically — the same half-open window a
        `BarRange` uses, so a stored range and a requested range mean one thing."""
        ...

    def upsert_many(self, incoming: Iterable[Bar]) -> int:
        """Store bars, keeping the higher revision when one already exists.

        A correction replaces what it corrects; a late copy of an older revision
        is dropped rather than allowed to undo it."""
        ...


class StreamingBarSource(Protocol):
    """Anything that can emit bars as they form."""

    def capabilities(self) -> ProviderCapabilities: ...

    def feed_status(self) -> FeedStatus: ...

    def stream_bars(self, symbols: Iterable[Ticker], timeframe: Timeframe) -> Iterator[Bar]:
        """Bars in non-decreasing `start_time_utc` order.

        May yield the same window more than once as a partial bar fills, ending
        with `is_final=True`. Consumers should feed the output through a
        `BarSeries`, which is what resolves the repeats.
        """
        ...


@dataclasses.dataclass(frozen=True, slots=True)
class Trade:
    """One print, for display only — `adr/0016-latest-trade-stream.md`.

    Deliberately not a `Bar`: it carries no session, no aggregation window, no
    revision. Nothing in `alphagate.core` outside this module may import it, and no
    engine may consume it — an indicator, a structure or a trend reads finalized
    bars, never "whatever it last traded at just now". This type exists to
    answer one question a bar-close-only view cannot: is this symbol still
    trading, and at what price, right now.
    """

    symbol: Ticker
    price: Decimal
    size: Decimal
    timestamp: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", ensure_utc(self.timestamp, field="timestamp"))


class StreamingTradeSource(Protocol):
    """Anything that can emit trades as they print."""

    def capabilities(self) -> ProviderCapabilities: ...

    def feed_status(self) -> FeedStatus: ...

    def stream_trades(self, symbols: Iterable[Ticker]) -> Iterator[Trade]:
        """Trades in the order printed. Display only — see `Trade`."""
        ...

    def close(self) -> None:
        """Ask `stream_trades` to return control at its next checkpoint.

        `stream_trades` is a generator that reconnects on its own and may not
        yield for a while on a quiet stream, so a caller running it on a worker
        thread needs a way to ask it to stop that does not depend on it having
        just produced a value.
        """
        ...
