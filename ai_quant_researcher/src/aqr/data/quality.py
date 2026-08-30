"""Is this series actually usable, or does it merely look like data?

Written after a concrete failure. A pull of SPY daily bars from Alpaca's IEX
feed returned 1530 bars across a window holding roughly 1960 sessions, with a
single 634-day hole in the middle. Nothing downstream noticed, and nothing
downstream *could*: the backtester indexes positionally, so a two-year gap is
just two adjacent bars, and every metric computed on top of it is arithmetic
performed on a history that did not happen.

The checks here are the cheapest possible defence -- counting, differencing and
one ratio -- and they run on every pull rather than on request, because a data
problem that requires someone to remember to look for it is a data problem that
ships.

Nothing here judges a *strategy*. It judges the bars.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta

import numpy as np

from aqr.data.bars import Bars

__all__ = ["QualityReport", "inspect"]

# A Thursday-to-Tuesday holiday weekend is four calendar days. Five is the first
# span a normal US equity calendar cannot produce, so that is where "gap" starts.
MAX_NORMAL_GAP_DAYS = 4

# Days the US equity market was closed unexpectedly. Without these, every clean
# series from 2010 onward reports a five-day hole at Hurricane Sandy and the
# check trains its reader to ignore it -- which is worse than not checking.
#
# The table is deliberately short and only covers *unscheduled* closures. Fixed
# holidays never produce a gap over four days on their own, so they need no
# entry, and maintaining a full exchange calendar to refine a warning threshold
# would buy precision nobody uses.
UNSCHEDULED_CLOSURES: dict[date, str] = {
    date(2001, 9, 11): "9/11",
    date(2001, 9, 12): "9/11",
    date(2001, 9, 13): "9/11",
    date(2001, 9, 14): "9/11",
    date(2004, 6, 11): "Reagan state funeral",
    date(2007, 1, 2): "Ford national day of mourning",
    date(2012, 10, 29): "Hurricane Sandy",
    date(2012, 10, 30): "Hurricane Sandy",
    date(2018, 12, 5): "Bush national day of mourning",
    date(2025, 1, 9): "Carter national day of mourning",
}

# Sessions per calendar year, for the coverage ratio. Deliberately the round
# convention rather than a real calendar: this number decides whether to print a
# warning, and importing a holiday database to refine a warning threshold buys
# precision nobody uses.
SESSIONS_PER_YEAR = 252.0

# Below this, the series is missing enough of its own history that any backtest
# on it is describing a different market.
MIN_COVERAGE = 0.85

# Coverage over a short span is noise: two bars a weekend apart are "67% of the
# three sessions expected", which is true and useless. Below this span the ratio
# is not reported at all rather than reported as a warning.
MIN_SPAN_DAYS_FOR_COVERAGE = 180

# How much later than requested a series may start before it is worth saying
# so. A month absorbs holiday and weekend edges; a year does not.
MIN_LATE_START_DAYS = 60

# An overnight move this large in adjusted bars is almost always a split that
# was not adjusted -- a 2:1 is -50%, a 4:1 is -75%. Real single-session moves of
# this size exist, which is why this is a warning and not an exception.
SPLIT_RATIO = 0.4


@dataclass(frozen=True, slots=True)
class QualityReport:
    """What is wrong with a bar series, and whether it is bad enough to stop."""

    symbol: str
    timeframe: str
    bar_count: int
    first: datetime
    last: datetime
    max_gap_days: int
    gaps: list[tuple[datetime, datetime, int]] = field(default_factory=list)
    """``(last good bar, next bar, calendar days between)`` for each hole."""
    expected_sessions: int = 0
    coverage: float | None = None
    """Bars present over sessions expected. ``None`` for intraday, where the
    expected count depends on the venue's session length and guessing it would
    produce a warning nobody could act on."""
    zero_volume_bars: int = 0
    non_positive_prices: int = 0
    suspected_splits: int = 0
    starts_late_days: int = 0
    """Days between the requested start and the first bar actually returned.

    Reported, never fatal. A listing date is a fact about the instrument, and
    refusing to cache a symbol for having one would drop every recent IPO from
    the universe. But a provider quietly answering a 22-year request with 8
    years is how research runs on a fifth of the history someone believes it has."""

    @property
    def ok(self) -> bool:
        return not self.problems()

    def problems(self) -> list[str]:
        """Every reason not to research on this series, worst first."""
        out: list[str] = []
        if self.non_positive_prices:
            out.append(f"{self.non_positive_prices} bar(s) with a non-positive price")
        if self.gaps:
            worst = max(g[2] for g in self.gaps)
            out.append(f"{len(self.gaps)} gap(s), largest {worst} calendar days")
        if self.coverage is not None and self.coverage < MIN_COVERAGE:
            out.append(
                f"{self.coverage:.0%} coverage: {self.bar_count} bars where "
                f"{self.expected_sessions} sessions were expected"
            )
        if self.suspected_splits:
            out.append(
                f"{self.suspected_splits} overnight move(s) over {SPLIT_RATIO:.0%} -- "
                "these bars may not be split-adjusted"
            )
        return out

    def notes(self) -> list[str]:
        """Things worth seeing that are not reasons to reject the series."""
        out: list[str] = []
        if self.starts_late_days >= MIN_LATE_START_DAYS:
            out.append(
                f"starts {self.starts_late_days // 365} year(s) later than requested "
                f"-- short history, not a broken series"
            )
        if self.zero_volume_bars:
            out.append(f"{self.zero_volume_bars} zero-volume bar(s)")
        return out

    def __str__(self) -> str:
        span = f"{self.first.date()} -> {self.last.date()}"
        head = f"{self.symbol} {self.timeframe}: {self.bar_count} bars {span}"
        problems = self.problems()
        notes = self.notes()
        if not problems:
            head = f"{head} -- ok"
        lines = [head]
        lines += [f"  ! {p}" for p in problems]
        lines += [f"  - {n}" for n in notes]
        return "\n".join(lines)

    def as_dict(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "bars": self.bar_count,
            "first": self.first.isoformat(),
            "last": self.last.isoformat(),
            "max_gap_days": self.max_gap_days,
            "coverage": self.coverage,
            "zero_volume_bars": self.zero_volume_bars,
            "non_positive_prices": self.non_positive_prices,
            "suspected_splits": self.suspected_splits,
            "starts_late_days": self.starts_late_days,
            "problems": self.problems(),
            "notes": self.notes(),
        }


def _closed_days_between(left: datetime, right: datetime) -> int:
    """Unscheduled closure days strictly between two bars."""
    day = left.date() + timedelta(days=1)
    last = right.date()
    closed = 0
    while day < last:
        if day in UNSCHEDULED_CLOSURES:
            closed += 1
        day += timedelta(days=1)
    return closed


def inspect(bars: Bars, requested_start: datetime | None = None) -> QualityReport:
    """Check a series for the holes that survive every downstream defence.

    ``requested_start`` is what the caller asked for. Supplying it is what
    lets a silently-truncated pull be seen at all: a series that begins a
    decade after the request is internally perfect and answers a different
    question from the one that was asked."""
    n = len(bars)
    if n == 0:
        raise ValueError("cannot inspect an empty series")

    stamps = bars.event_time
    first = datetime.fromtimestamp(int(stamps[0]), tz=UTC)
    last = datetime.fromtimestamp(int(stamps[-1]), tz=UTC)

    gaps: list[tuple[datetime, datetime, int]] = []
    max_gap = 0
    if n > 1:
        deltas = np.diff(stamps) // 86_400
        max_gap = int(deltas.max())
        for i in np.flatnonzero(deltas > MAX_NORMAL_GAP_DAYS):
            left = datetime.fromtimestamp(int(stamps[i]), tz=UTC)
            right = datetime.fromtimestamp(int(stamps[i + 1]), tz=UTC)
            span = int(deltas[i])
            if span - _closed_days_between(left, right) <= MAX_NORMAL_GAP_DAYS:
                continue  # explained: the market was shut, the data is not missing
            gaps.append((left, right, span))

    expected = 0
    coverage: float | None = None
    span_days = (last - first).total_seconds() / 86_400
    if bars.timeframe in ("1D", "1W") and span_days >= MIN_SPAN_DAYS_FOR_COVERAGE:
        years = span_days / 365.25
        per_year = SESSIONS_PER_YEAR if bars.timeframe == "1D" else 52.0
        expected = max(1, round(years * per_year))
        coverage = min(1.0, n / expected)

    lows = np.asarray(bars.low, dtype=np.float64)
    opens = np.asarray(bars.open, dtype=np.float64)
    non_positive = int(np.count_nonzero((lows <= 0) | (opens <= 0) | ~np.isfinite(lows)))

    closes = np.asarray(bars.close, dtype=np.float64)
    splits = 0
    if n > 1:
        previous = closes[:-1]
        moves = np.where(previous > 0, np.abs(closes[1:] - previous) / previous, 0.0)
        splits = int(np.count_nonzero(moves > SPLIT_RATIO))

    return QualityReport(
        symbol=bars.symbol,
        timeframe=bars.timeframe,
        bar_count=n,
        first=first,
        last=last,
        max_gap_days=max_gap,
        gaps=gaps,
        expected_sessions=expected,
        coverage=coverage,
        zero_volume_bars=int(np.count_nonzero(np.asarray(bars.volume) <= 0)),
        non_positive_prices=non_positive,
        suspected_splits=splits,
        starts_late_days=(
            max(0, int((first - requested_start).total_seconds() // 86_400))
            if requested_start
            else 0
        ),
    )
