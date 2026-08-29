"""The competition watchlist — specs/05 D8.

> "A handful of liquid optionable underlyings (SPY, QQQ and a few large caps).
> Breadth is not the point; enough fills to escape single-trade noise is."

One place, logged at startup, rendered in the dashboard — the same treatment the
risk limits get (specs/03 D5), and for the same reason: a configuration that
lives at its call sites is a configuration nobody can review.

**Why these, and why so few.** The scoring window is four days and the goal is
roughly thirty fills. Thirty fills across six names is five trades a name, which
is enough for the P&L to be about the strategy rather than about one spread;
thirty fills across forty names is one trade each, which is a sampling of the
market and not a demonstration of anything. Breadth would also break the
per-underlying concentration limit's usefulness — a limit that is never near
binding is a limit that tells you nothing.

Every entry is chosen for the same three properties, in this order:

1. **Penny-wide or near-penny option markets.** The Gate refuses anything wider
   than 5% (specs/03 D5), and on a thin chain that check does all the work and
   no trade ever happens. This was not hypothetical — the first live runs were
   vetoed on `liquidity` repeatedly.
2. **Weekly expiries**, so the 3–21 day window is actually reachable rather than
   rounding to whatever monthly happens to be nearby.
3. **A price that makes a 5-wide spread a sensible size** against a 1%-of-equity
   per-trade limit.

The ETFs carry the load. They have no earnings — a fact about the instrument,
not a lookup (see `earnings.py`) — which means they are the only names that can
satisfy the screen at all while this account has no earnings calendar. The
single names are listed, and will be declined by the screen until someone fills
in `COMPETITION_EARNINGS`. That is the correct behaviour and it is deliberately
visible rather than quietly excluded.
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
    Underlying(ticker("SPY"), Decimal(5), "S&P 500. Penny-wide, daily expiries, no earnings."),
    Underlying(ticker("QQQ"), Decimal(5), "Nasdaq 100. Same, slightly more volatility."),
    Underlying(ticker("IWM"), Decimal(1), "Russell 2000. Cheaper, so a narrower wing."),
    Underlying(ticker("AAPL"), Decimal(5), "Large cap. Needs an earnings date to trade."),
    Underlying(ticker("MSFT"), Decimal(5), "Large cap. Needs an earnings date to trade."),
    Underlying(ticker("NVDA"), Decimal(5), "Large cap, high IV. Needs an earnings date."),
)

COMPETITION_EARNINGS: Final = StaticEarningsCalendar({})
"""Hand-maintained earnings dates for the single names. **Deliberately empty.**

Alpaca has no earnings calendar (see `earnings.py`), so until a human fills this
in, every single-name read carries `earnings_within_dte=None`, the screen
refuses it, and the journal says why. An empty mapping that declines is honest;
a populated one that was guessed at is not.

To trade the single names: look up the next report date for each, put it here,
and the screen starts admitting them. That is one line of work and it is a line
a person has to sign for."""


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
