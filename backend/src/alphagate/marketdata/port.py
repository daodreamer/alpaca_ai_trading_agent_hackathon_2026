"""The market-data seam — adr/0002 D2.

Market data comes in over a **direct REST adapter**, not over MCP, for one
load-bearing reason: the backtest and the live agent must run the same code path
with only the clock differing (specs/01 Rule 4). A recorded-payload transport
seam gives us that; an MCP session does not replay.

So this module declares a `MarketData` protocol and two implementations live
behind it — `AlpacaMarketData` over HTTPS, and `RecordedMarketData` over a
directory of captured payloads. The suite uses the second exclusively and never
opens a socket.

The second reason is smaller and still real: MCP wraps every response in an
`_alpaca_mcp_security` envelope. That envelope earns its keep on the write path,
where model-facing output lives (specs/04 D7), and is pure overhead on a bar
fetch.

**GET only.** Nothing in this package can place, cancel or amend anything. That
is not a convention — there is no method here that could, and a boundary test
asserts no `post`/`put`/`delete` call appears in the package. Orders leave
through one door (specs/01 Rule 1b) and this is not it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime
from decimal import Decimal
from typing import Protocol, runtime_checkable

from alphagate.core.bar import Bar
from alphagate.core.identifiers import Ticker
from alphagate.options import OptionContract, OptionQuote

__all__ = ["MarketData", "OptionBar"]


class OptionBar:
    """One daily bar for one option contract, for IV reconstruction.

    Deliberately not `core.bar.Bar`: that type carries a session, a feed and an
    adjustment mode, all of which are claims about an *equity* series that would
    be invented here. A contract's daily close is a smaller fact and gets a
    smaller type.
    """

    __slots__ = ("close", "contract", "session_date", "volume")

    def __init__(
        self, contract: OptionContract, session_date: date, close: Decimal, volume: int
    ) -> None:
        self.contract = contract
        self.session_date = session_date
        self.close = close
        self.volume = volume

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"OptionBar({self.contract}, {self.session_date}, close={self.close})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, OptionBar):
            return NotImplemented
        return (
            self.contract == other.contract
            and self.session_date == other.session_date
            and self.close == other.close
            and self.volume == other.volume
        )

    def __hash__(self) -> int:
        return hash((self.contract, self.session_date, self.close, self.volume))


@runtime_checkable
class MarketData(Protocol):
    """Everything the perception step reads. Read-only, by construction."""

    def daily_bars(self, symbol: Ticker, *, start: date, end: date) -> tuple[Bar, ...]:
        """Adjusted daily bars, oldest first, closed bars only."""
        ...

    def intraday_bars(
        self, symbol: Ticker, *, timeframe: str, start: datetime, end: datetime
    ) -> tuple[Bar, ...]: ...

    def latest_price(self, symbol: Ticker) -> Decimal:
        """Mid of the current top of book."""
        ...

    def option_chain(
        self,
        symbol: Ticker,
        *,
        expiry_from: date,
        expiry_to: date,
        strike_from: Decimal | None = None,
        strike_to: Decimal | None = None,
        right: str | None = None,
    ) -> Mapping[OptionContract, OptionQuote]: ...

    def option_daily_bars(
        self, contract: OptionContract, *, start: date, end: date
    ) -> Sequence[OptionBar]:
        """Daily closes for one contract. The raw material for IV history."""
        ...
