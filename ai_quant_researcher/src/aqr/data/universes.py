"""Named universes.

A hard-coded ticker list is a dated artefact: constituents change, companies are
acquired, tickers are reused. So every list here carries an ``as_of`` date, and
:func:`resolve` verifies against the data source rather than trusting the
constant. A symbol that no longer returns bars is dropped with a warning instead
of silently producing an empty series that the backtester would happily trade.

The survivorship warning matters more than the list. ``NASDAQ_50`` is the
*current* top 50 — every one of them got there by going up. Backtesting a
strategy on today's winners over 2010–2024 measures, in part, the strategy's
ability to profit from the fact that we already know who won. Treat any result
on this universe as an upper bound, never as an expectation.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from functools import cache
from pathlib import Path
from typing import Any

__all__ = [
    "NASDAQ_50",
    "PIT_CSV_ROOTS",
    "PIT_UNIVERSE_FILES",
    "UNIVERSES",
    "Interval",
    "Membership",
    "PointInTimeUniverse",
    "Universe",
    "is_point_in_time",
    "load_point_in_time",
    "resolve",
    "universe_names",
]


@dataclass(frozen=True, slots=True)
class Universe:
    name: str
    symbols: tuple[str, ...]
    as_of: date
    note: str = ""

    def __len__(self) -> int:
        return len(self.symbols)

    def __str__(self) -> str:
        return f"{self.name}: {len(self.symbols)} symbols as of {self.as_of.isoformat()}"


# The 50 largest Nasdaq-listed companies by market capitalisation. Ordered
# roughly by size at the stated date, which is the order in which a partial slice
# should be taken.
NASDAQ_50 = Universe(
    name="nasdaq50",
    as_of=date(2026, 8, 26),
    note=(
        "Current constituents. Survivorship-biased for any backtest that starts "
        "before as_of -- these are the names that survived to be listed today."
    ),
    symbols=(
        "NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "AVGO", "TSLA", "NFLX", "COST",
        "PLTR", "ASML", "AMD", "CSCO", "TMUS", "AZN", "LIN", "INTU", "PEP", "TXN",
        "QCOM", "AMGN", "ISRG", "BKNG", "ADBE", "AMAT", "PDD", "GILD", "HON", "MU",
        "LRCX", "ADP", "VRTX", "KLAC", "PANW", "SBUX", "MELI", "ADI", "CRWD", "INTC",
        "CEG", "MDLZ", "CTAS", "ORLY", "MRVL", "CDNS", "SNPS", "FTNT", "ABNB", "REGN",
    ),
)

UNIVERSES: dict[str, Universe] = {NASDAQ_50.name: NASDAQ_50}

# Point-in-time universes are held as JSON rather than as a constant: an interval
# table for six hundred names is data, and a reconstruction that records its own
# corrections and its own unresolved cases cannot be a tuple of tickers.
#
# The path is anchored to the package, not to the working directory, so that a
# universe resolves the same way from a CLI run, a test and a notebook.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]

PIT_UNIVERSE_FILES: dict[str, Path] = {
    "sp500_pit": _PROJECT_ROOT / "data-universes" / "sp500_pit.json",
}

# Which cache root holds the bars for each point-in-time universe. The pull that
# built ``data-sp500`` was driven by the union across time, so it is the only
# root that has the names that left the index -- and those are precisely the
# names whose absence is the survivorship bias.
PIT_CSV_ROOTS: dict[str, str] = {"sp500_pit": "data-sp500"}


def universe_names() -> list[str]:
    """Every name ``resolve`` accepts, point-in-time ones included."""
    return sorted({*UNIVERSES, *PIT_UNIVERSE_FILES})


def is_point_in_time(name: str) -> bool:
    return name.strip().lower() in PIT_UNIVERSE_FILES


@cache
def load_point_in_time(name: str) -> PointInTimeUniverse:
    """The membership table for a registered point-in-time universe.

    Cached because it is read once and consulted per symbol per session, and
    because :class:`PointInTimeUniverse` is frozen, so sharing one instance
    across callers cannot make two runs differ.
    """
    key = name.strip().lower()
    path = PIT_UNIVERSE_FILES.get(key)
    if path is None:
        raise KeyError(
            f"{name!r} is not a point-in-time universe; known: "
            f"{sorted(PIT_UNIVERSE_FILES)}"
        )
    if not path.exists():
        raise FileNotFoundError(
            f"{key} is registered but its interval table is missing: {path}"
        )
    return PointInTimeUniverse.from_json(path, name=key)


def resolve(name: str, limit: int | None = None) -> list[str]:
    """Symbols for a named universe, optionally truncated to the largest ``limit``.

    For a point-in-time universe this is the union across time -- every ticker
    ever a member -- which is what has to be loaded. Restricting a given session
    to that session's members is the membership table's job, not this one's.
    """
    key = name.strip().lower()
    if key in PIT_UNIVERSE_FILES:
        if limit:
            # The constant lists are ordered by size, so "the largest N" means
            # something there. A point-in-time universe has no size order, so a
            # limit would silently mean "the alphabetically first N" -- a
            # selection made on the ticker string, which is not a universe.
            raise ValueError(
                f"{key} is point-in-time and has no size order, so a limit of "
                f"{limit} would select on the ticker string; drop the limit"
            )
        return list(load_point_in_time(key).all_symbols())
    if key not in UNIVERSES:
        raise KeyError(f"unknown universe {name!r}; known: {universe_names()}")
    symbols = list(UNIVERSES[key].symbols)
    return symbols[:limit] if limit else symbols


# --------------------------------------------------------------------------- #
# Membership as a function of time
# --------------------------------------------------------------------------- #
#
# The docstring at the top of this file has always said what is wrong with a
# constant ticker list. Nothing enforced it, because ``resolve()`` has no date
# parameter: membership was a constant and the survivorship warning was a
# paragraph.
#
# That is survivable for a rule that looks at one symbol at a time and fatal for
# one that ranks. A cross-sectional score asks how a name compares to its peers,
# and if the peer set is the set of eventual winners then the comparison is
# contaminated at its root -- ``rs_rank(60)`` computed against survivors is not a
# number any researcher in 2016 could have computed.


@dataclass(frozen=True, slots=True)
class Interval:
    """A half-open membership span: ``start`` inclusive, ``end`` exclusive.

    ``end is None`` means still a member. Half-open because that is how the
    source records a removal -- as the effective date, the first day the name is
    *not* in the index -- and a boundary convention chosen twice is chosen wrong
    once.
    """

    start: date
    end: date | None = None

    def __post_init__(self) -> None:
        if self.end is not None and self.end < self.start:
            raise ValueError(f"interval ends before it starts: {self.start} .. {self.end}")

    def contains(self, day: date) -> bool:
        return self.start <= day and (self.end is None or day < self.end)


@dataclass(frozen=True, slots=True)
class Membership:
    """One ticker and every span during which it was a member."""

    ticker: str
    name: str
    intervals: tuple[Interval, ...]

    def __post_init__(self) -> None:
        if not self.intervals:
            raise ValueError(f"{self.ticker}: a member needs at least one interval")
        ordered = sorted(self.intervals, key=lambda i: i.start)
        open_ended = [i for i in ordered if i.end is None]
        if len(open_ended) > 1:
            raise ValueError(
                f"{self.ticker}: two open-ended intervals; a name cannot be a member "
                f"of two unbounded spans at once"
            )
        for earlier, later in zip(ordered, ordered[1:], strict=False):
            if earlier.end is None or earlier.end > later.start:
                raise ValueError(
                    f"{self.ticker}: intervals overlap ({earlier.start}..{earlier.end} "
                    f"and {later.start}..{later.end})"
                )
        object.__setattr__(self, "intervals", tuple(ordered))

    def member_on(self, day: date) -> bool:
        return any(i.contains(day) for i in self.intervals)

    @property
    def left(self) -> bool:
        """Whether the name is out at the end of its last interval."""
        return self.intervals[-1].end is not None

    def member_between(self, first: date, last: date) -> bool:
        """Whether any interval overlaps the half-open span ``[first, last)``."""
        return any(
            interval.start < last and (interval.end is None or interval.end > first)
            for interval in self.intervals
        )


@dataclass(frozen=True, slots=True)
class PointInTimeUniverse:
    """A universe whose membership is a function of the date.

    ``all_symbols`` is the union across time, which is what has to be
    downloaded: a pull driven by today's members is missing precisely the names
    whose absence created the bias.
    """

    name: str
    source: str
    window: tuple[date, date]
    members: tuple[Membership, ...]

    def by_ticker(self, ticker: str) -> Membership:
        for member in self.members:
            if member.ticker == ticker:
                return member
        raise KeyError(f"{ticker} was never a member of {self.name}")

    def symbols_on(self, day: date) -> tuple[str, ...]:
        return tuple(sorted(m.ticker for m in self.members if m.member_on(day)))

    def unexplained_absences(
        self, symbols: Iterable[str], first: date, last: date
    ) -> list[str]:
        """Which of ``symbols`` have no business being absent over ``[first, last)``.

        A universe with a fixed ticker list is a contract: a name the strategy
        claims to trade but has no bars for is a data hole, and running on the
        rest would quietly evaluate a different strategy. That reasoning holds
        for most of a point-in-time universe too -- but not for the part of it
        that is *supposed* to be absent.

        A name that was not a member at any point in the span had not listed
        yet, or had already been acquired. Its absence is what the table
        predicts, and treating it as a failure makes any check built on it a
        function of how large the universe is rather than of the data. A name
        that *was* a member in the span and still has no bars is a hole exactly
        as before, and is what this returns.

        A ticker the table has never heard of is not reported: the membership
        mask already treats it as never a member, so the run cannot trade it
        and its absence agrees with the gate rather than contradicting it.
        """
        absent: list[str] = []
        for symbol in symbols:
            try:
                member = self.by_ticker(symbol)
            except KeyError:
                continue
            if member.member_between(first, last):
                absent.append(symbol)
        return absent

    def all_symbols(self) -> tuple[str, ...]:
        """Every ticker ever a member, sorted.

        Sorted because the portfolio engine breaks ranking ties by symbol, so an
        order that depended on dict iteration would make a backtest
        irreproducible through the tie-break.
        """
        return tuple(sorted(m.ticker for m in self.members))

    def survivorship(self) -> dict[str, Any]:
        """The bias as a number rather than a paragraph.

        ``reconstructed`` is False for a universe assembled from today's list
        alone. Such a universe reports zero departures, and it is important that
        this reads as "not measured" rather than as "no bias": it has no
        removals *because* it was chosen with hindsight.
        """
        ever = len(self.members)
        left = sum(1 for m in self.members if m.left)
        return {
            "ever": ever,
            "still_in": ever - left,
            "left": left,
            "left_fraction": (left / ever) if ever else 0.0,
            "reconstructed": left > 0,
            "window": [self.window[0].isoformat(), self.window[1].isoformat()],
            "source": self.source,
        }

    @classmethod
    def from_json(cls, path: Path | str, *, name: str = "") -> PointInTimeUniverse:
        """Load the interval table produced by the membership reconstruction."""
        body = json.loads(Path(path).read_text(encoding="utf-8"))
        window = body["window"]
        members = tuple(
            Membership(
                ticker=str(entry["ticker"]),
                name=str(entry.get("name", "")),
                intervals=tuple(
                    Interval(
                        start=date.fromisoformat(span["from"]),
                        end=date.fromisoformat(span["to"]) if span.get("to") else None,
                    )
                    for span in entry["intervals"]
                ),
            )
            for entry in body["members"]
        )
        return cls(
            name=name or body.get("name", "sp500_pit"),
            source=str(body.get("source", {}).get("changes_url", "unknown")),
            window=(date.fromisoformat(window["start"]), date.fromisoformat(window["end"])),
            members=members,
        )
