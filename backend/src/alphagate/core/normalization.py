"""Bar ingestion and normalization — specs/03-market-data.md.

A `BarSeries` is the canonical store for one `(symbol, timeframe, feed,
adjustment_mode)` combination — the uniqueness key from `specs/12-storage.md`.
It is the thing that turns a messy provider stream into the ordered, deduplicated
sequence the engines expect.

Every ingest returns an `IngestOutcome` rather than a bare bool. "The bar was
already there", "it superseded an older copy" and "it was stale and dropped" are
different facts, and a reconnect-and-backfill path needs to tell them apart to
report honestly on what it did.

The one case that raises rather than reports is a *conflict*: the same bar, same
revision, different values. That is a provider contradicting itself, and picking
a winner silently would put an unexplainable number into the bar store.
"""

from __future__ import annotations

import bisect
from collections.abc import Iterable, Sequence
from datetime import datetime
from enum import Enum

from alphagate.core.bar import AdjustmentMode, Bar, Feed
from alphagate.core.errors import InvariantViolation
from alphagate.core.identifiers import Ticker
from alphagate.core.time_model import Timeframe, ensure_utc

__all__ = ["BarSeries", "IngestOutcome"]


class IngestOutcome(Enum):
    ACCEPTED = "ACCEPTED"
    """A bar not previously seen."""

    DUPLICATE = "DUPLICATE"
    """Byte-identical to what is already held; dropped."""

    REVISED = "REVISED"
    """A higher revision replaced the stored bar."""

    UPDATED = "UPDATED"
    """A partial bar was replaced by a fuller version of itself.

    Distinct from REVISED: no revision number changed, the bar simply had not
    closed yet. This is the ordinary intrabar streaming case, not a correction.
    """

    STALE = "STALE"
    """An older revision arrived after a newer one; dropped."""


class BarSeries:
    """Ordered, deduplicated bars for one series key.

    Mutable by design — it accumulates a stream. The bars it holds are immutable,
    and `bars` hands out a snapshot, so a consumer iterating cannot be surprised
    by a concurrent ingest.
    """

    __slots__ = (
        "_by_start",
        "_duplicate_count",
        "_out_of_order_count",
        "_revision_count",
        "_starts",
        "adjustment_mode",
        "feed",
        "symbol",
        "timeframe",
    )

    def __init__(
        self,
        *,
        symbol: Ticker,
        timeframe: Timeframe,
        feed: Feed,
        adjustment_mode: AdjustmentMode,
    ) -> None:
        self.symbol = symbol
        self.timeframe = timeframe
        self.feed = feed
        self.adjustment_mode = adjustment_mode

        self._by_start: dict[datetime, Bar] = {}
        self._starts: list[datetime] = []
        self._duplicate_count = 0
        self._revision_count = 0
        self._out_of_order_count = 0

    # -- reading -----------------------------------------------------------

    def __len__(self) -> int:
        return len(self._starts)

    @property
    def bars(self) -> tuple[Bar, ...]:
        """Chronological snapshot."""
        return tuple(self._by_start[start] for start in self._starts)

    @property
    def final_bars(self) -> tuple[Bar, ...]:
        """Only the finalized bars.

        What indicators consume: a partial bar must not seed a recursive
        calculation (CLAUDE.md §12).
        """
        return tuple(bar for bar in self.bars if bar.is_final)

    @property
    def latest(self) -> Bar | None:
        if not self._starts:
            return None
        return self._by_start[self._starts[-1]]

    @property
    def duplicate_count(self) -> int:
        return self._duplicate_count

    @property
    def revision_count(self) -> int:
        return self._revision_count

    @property
    def out_of_order_count(self) -> int:
        return self._out_of_order_count

    # -- writing -----------------------------------------------------------

    def ingest(self, bar: Bar) -> IngestOutcome:
        self._check_belongs(bar)
        start = bar.start_time_utc
        existing = self._by_start.get(start)

        if existing is None:
            self._insert(start, bar)
            return IngestOutcome.ACCEPTED

        if bar.revision > existing.revision:
            self._by_start[start] = bar
            self._revision_count += 1
            return IngestOutcome.REVISED

        if bar.revision < existing.revision:
            return IngestOutcome.STALE

        if bar == existing:
            self._duplicate_count += 1
            return IngestOutcome.DUPLICATE

        if not existing.is_final:
            # The stored bar was still forming. A new value for an open window is
            # the window filling in, not the provider changing its mind.
            self._by_start[start] = bar
            return IngestOutcome.UPDATED

        raise InvariantViolation(
            f"{self.symbol} {self.timeframe.code} {start.isoformat()}: two different "
            f"bars arrived at revision {existing.revision} for the same closed "
            "window. The provider is contradicting itself; a correction must bump "
            "the revision."
        )

    def ingest_all(self, bars: Iterable[Bar]) -> tuple[IngestOutcome, ...]:
        return tuple(self.ingest(bar) for bar in bars)

    def _insert(self, start: datetime, bar: Bar) -> None:
        position = bisect.bisect_left(self._starts, start)
        if position != len(self._starts):
            self._out_of_order_count += 1
        self._starts.insert(position, start)
        self._by_start[start] = bar

    def _check_belongs(self, bar: Bar) -> None:
        if bar.symbol != self.symbol:
            raise InvariantViolation(
                f"symbol mismatch: series holds {self.symbol}, bar is {bar.symbol}"
            )
        if bar.timeframe is not self.timeframe:
            raise InvariantViolation(
                f"timeframe mismatch: series holds {self.timeframe.code}, bar is "
                f"{bar.timeframe.code}"
            )
        if bar.feed is not self.feed:
            raise InvariantViolation(
                f"feed mismatch: series holds {self.feed.value}, bar is {bar.feed.value}"
            )
        if bar.adjustment_mode is not self.adjustment_mode:
            raise InvariantViolation(
                f"adjustment_mode mismatch: series holds {self.adjustment_mode.value}, "
                f"bar is {bar.adjustment_mode.value}"
            )

    # -- gaps --------------------------------------------------------------

    def missing_starts(self, expected: Sequence[datetime]) -> tuple[datetime, ...]:
        """Which expected cell starts have no finalized bar.

        Takes the expected starts rather than deriving them, so the caller
        decides which sessions and which session kinds are in scope — a series
        with no pre-market subscription is not full of gaps.
        """
        present = {start for start, bar in self._by_start.items() if bar.is_final}
        return tuple(
            moment
            for moment in (ensure_utc(e, field="expected") for e in expected)
            if moment not in present
        )
