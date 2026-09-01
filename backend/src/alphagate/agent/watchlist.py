"""The competition watchlist — specs/05 D8, specs/07 D2.

**One entry: SPY.**

Not a placeholder and not a starting point. specs/07 D2 requires four things of
an underlying at once, and SPY is the only instrument that has all four:

1. **A market tight enough to trade.** The Gate refuses anything wider than 5%
   (specs/03 D5), and on a thin chain that check does all the work and no trade
   ever happens — the first live runs of this system were vetoed on `liquidity`
   repeatedly. SPY quotes 0.53% on the strikes this strategy sells; the sector
   ETFs reached for as breadth quote 10% to 200%.
2. **No earnings, structurally.** An index trust has no quarter to report, so
   `earnings.ETF_UNDERLYINGS` answers `earnings_within_dte` as a fact about the
   instrument rather than a lookup. Alpaca serves no earnings calendar, so a
   single name's field is `None` and the screen fail-closes on it.
3. **Weekly expiries**, so the 3-21 day window is reachable rather than rounding
   to whatever monthly happens to be nearby.
4. **Free history to backtest on**, back to 2019 with bid, ask, implied
   volatility and greeks. QQQ and IWM are tradeable and have none.

One place, logged at startup, rendered in the dashboard — the same treatment the
risk limits get (specs/03 D5), and for the same reason: a configuration that
lives at its call sites is a configuration nobody can review.

**What one name costs.** Concurrent positions differ by expiry and short strike
on the same underlying, so they are not independent bets: one adverse move in
SPY moves all of them. specs/07 D7 reports the P&L as a sample of one
underlying's window rather than of a strategy across a market. It also makes
`underlying_concentration` and `portfolio_heat` measure the same quantity, which
is why specs/03 D6 sets them equal rather than leaving a concentration limit to
silently become the binding cap.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Final

from alphagate.agent.earnings import StaticEarningsCalendar
from alphagate.core.identifiers import Ticker, ticker

__all__ = [
    "COMPETITION_EARNINGS",
    "WATCHLIST",
    "Underlying",
    "tradeable_today",
]


@dataclass(frozen=True, slots=True)
class Underlying:
    """One name, and the parameters its chain is enumerated with.

    `spread_width` is per-name because five dollars is a sensible wing on a $700
    index and an absurd one on a $40 stock — it would be most of the underlying.
    Expressing it as a fraction of spot instead would be neater and wrong: strike
    ladders are quoted in fixed increments, and a computed width lands between
    strikes that do not exist.
    """

    symbol: Ticker
    spread_width: Decimal
    note: str

    def __str__(self) -> str:
        return f"{self.symbol} ({self.spread_width}-wide)"


WATCHLIST: Final[tuple[Underlying, ...]] = (
    Underlying(
        ticker("SPY"),
        Decimal(5),
        "S&P 500. 0.53% quoted spread, weekly expiries, no earnings, 2019- history.",
    ),
)


COMPETITION_EARNINGS: Final = StaticEarningsCalendar({})
"""Hand-maintained earnings dates. **Empty, and under specs/07 D2 it stays so.**

The watchlist holds one index ETF, which has no quarter to report, so
`StaticEarningsCalendar` answers it from `ETF_UNDERLYINGS` and this mapping is
never consulted. It is kept rather than deleted because it is what a single name
would need before the screen could admit it: Alpaca has no earnings calendar
(see `earnings.py`), an earnings print inside the holding period is the single
most common way a defined-risk premium sale becomes a maximum loss, and the date
has to be one a human checked. Adding a name without adding its date is the
mistake this empty mapping is positioned to catch."""


def tradeable_today(
    calendar: StaticEarningsCalendar = COMPETITION_EARNINGS,
) -> tuple[Underlying, ...]:
    """The names the screen can actually admit right now.

    Not a filter applied silently at the top of the loop — call it, print what it
    returns, and know which names are sitting out and why. A watchlist that
    quietly shrinks is a watchlist that is lying about its breadth.
    """
    return tuple(entry for entry in WATCHLIST if calendar.is_known(entry.symbol))


def next_earnings_gaps(
    calendar: StaticEarningsCalendar = COMPETITION_EARNINGS, *, on: date | None = None
) -> tuple[Ticker, ...]:
    """Which watchlist names have no earnings date on file. For the startup log."""
    del on
    return tuple(entry.symbol for entry in WATCHLIST if not calendar.is_known(entry.symbol))
