"""Earnings dates — specs/05 D2's `earnings_within_dte`.

Alpaca has no earnings calendar. Its corporate-actions endpoint covers
dividends, splits, mergers and spinoffs; a probe against it returns
`{"cash_dividends": 1}` and nothing else. So this field cannot be computed from
the data we have, and the question is what to do about that rather than whether
to pretend otherwise.

Three answers, in descending order of honesty, and this module offers the first
two:

**Structurally never.** An index ETF has no earnings announcement. SPY and QQQ
are trusts holding baskets; there is no quarter for them to report. That is not
a lookup, it is a fact about the instrument, and `ETF_UNDERLYINGS` records it.

**Explicitly, by hand.** For a four-day competition on a handful of large caps,
a hand-maintained mapping is not a shortcut — it is more reliable than a scraped
calendar and it is auditable. `StaticEarningsCalendar` takes one, and the dates
in it are dates a human checked.

**Unknown.** Anything else. `next_earnings` returns `None`, `MarketRead
.earnings_within_dte` is `None`, the prompt renders `unmeasured`, and the model
is told to prefer declining when something it needs is unmeasured. An earnings
print inside the holding period is the single most common way a defined-risk
premium sale becomes a maximum loss, so "we did not check" must never render as
"there is none".

The distinction that costs money is between `False` and `None`, and it is
exactly the one a `bool` cannot make. That is why the field is `bool | None`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from typing import Final, Protocol, runtime_checkable

from alphagate.core.identifiers import Ticker, ticker

__all__ = [
    "ETF_UNDERLYINGS",
    "EarningsCalendar",
    "NoEarningsCalendar",
    "StaticEarningsCalendar",
    "earnings_within",
]

ETF_UNDERLYINGS: Final[frozenset[Ticker]] = frozenset(
    ticker(symbol)
    for symbol in ("SPY", "QQQ", "IWM", "DIA", "XLF", "XLE", "XLK", "EEM", "GLD", "TLT")
)
"""Index and sector ETFs: no earnings, ever. A fact about the instrument."""


@runtime_checkable
class EarningsCalendar(Protocol):
    """When does this underlying next report? `None` means nobody knows."""

    def next_earnings(self, symbol: Ticker, on_or_after: date) -> date | None: ...

    def is_known(self, symbol: Ticker) -> bool:
        """Whether this calendar can answer for `symbol` at all.

        Separate from `next_earnings` returning `None`, because "reports in no
        upcoming window" and "we have no idea" are different answers and only
        one of them is safe to trade on.
        """
        ...


@dataclass(frozen=True, slots=True)
class NoEarningsCalendar:
    """Knows about ETFs and nothing else. The fail-closed default.

    Every single-name underlying comes back unknown, which means every
    single-name trade carries an `unmeasured` earnings field into the prompt.
    That is the correct behaviour for a system with no calendar, and it is
    visibly so rather than quietly so.
    """

    def next_earnings(self, symbol: Ticker, on_or_after: date) -> date | None:
        return None

    def is_known(self, symbol: Ticker) -> bool:
        return symbol in ETF_UNDERLYINGS


@dataclass(frozen=True, slots=True)
class StaticEarningsCalendar:
    """A hand-maintained mapping. Auditable, and checked by a person.

    For a competition watchlist of half a dozen names over four days this beats
    a scraped feed on every axis that matters: it cannot silently go stale
    without someone noticing, and the provenance of every date is "a human
    looked it up".
    """

    dates: Mapping[Ticker, tuple[date, ...]] = field(default_factory=dict)

    def next_earnings(self, symbol: Ticker, on_or_after: date) -> date | None:
        upcoming = sorted(day for day in self.dates.get(symbol, ()) if day >= on_or_after)
        return upcoming[0] if upcoming else None

    def is_known(self, symbol: Ticker) -> bool:
        return symbol in ETF_UNDERLYINGS or symbol in self.dates


def earnings_within(
    calendar: EarningsCalendar,
    symbol: Ticker,
    *,
    as_of: date,
    through: date,
) -> bool | None:
    """Does an earnings print fall inside the holding period?

    `True` yes, `False` no, **`None` nobody checked** — and the caller must not
    collapse the last two. An earnings print inside the window is the most
    common way a defined-risk premium sale becomes a maximum loss.
    """
    if symbol in ETF_UNDERLYINGS:
        return False
    if not calendar.is_known(symbol):
        return None
    day = calendar.next_earnings(symbol, as_of)
    if day is None:
        return None
    return as_of <= day <= through
