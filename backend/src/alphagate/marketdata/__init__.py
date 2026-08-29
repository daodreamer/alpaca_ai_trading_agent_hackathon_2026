"""Market data — adr/0002 D2.

**Read-only.** Nothing in this package can place, cancel or amend an order.
Market data comes in over a direct REST adapter rather than over MCP, because
the backtest and the live agent must run the same code path with only the clock
differing (specs/01 Rule 4) — and a recorded-payload seam replays where an MCP
session does not.

`alpaca` is not re-exported: importing this package should not require `httpx`
to be reachable, the same way `alphagate.execution` does not re-export `stdio`.
"""

from alphagate.marketdata.port import MarketData, OptionBar, StockSnapshot
from alphagate.marketdata.recorded import RecordedMarketData

__all__ = ["MarketData", "OptionBar", "RecordedMarketData", "StockSnapshot"]
