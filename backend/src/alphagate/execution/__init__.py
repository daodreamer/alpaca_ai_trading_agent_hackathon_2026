"""Execution — specs/04, adr/0002.

The layer between an `Approved` verdict and a live Alpaca order.

`submit` accepts a `GatedOrder` and nothing else, and a `GatedOrder` can only be
minted inside `risk.gate`. That is the whole architecture of this package in one
sentence: there is exactly one door, and the Gate holds the only key.

`stdio` is deliberately not re-exported here. It is the one module that imports
`fastmcp`, and importing this package should not start caring whether that
library is installed — the tests never touch it, and the backtest never opens a
subprocess.
"""

from alphagate.execution.account import (
    ACCOUNT_TOOL,
    CLOCK_TOOL,
    POSITIONS_TOOL,
    AccountRead,
    LegPosition,
    MarketClock,
    read_account,
    read_clock,
    read_positions,
    to_account,
    to_clock,
    to_leg_positions,
)
from alphagate.execution.credentials import load_env_file, mcp_environment, require_paper_account
from alphagate.execution.equity import (
    PLACE_STOCK_ORDER_TOOL,
    Tradeability,
    cancel_equity,
    equity_arguments_for,
    equity_client_order_id,
    equity_order_fingerprint,
    read_back_equity,
    read_share_positions,
    read_tradeability,
    share_positions_from,
    submit_equity,
    to_stock_arguments,
)
from alphagate.execution.errors import (
    ExecutionError,
    MalformedToolOutput,
    PartialFillBreach,
    ToolTimeout,
    TransportFailure,
    UnsubmittableOrder,
)
from alphagate.execution.idempotency import client_order_id, order_fingerprint, trading_day_of
from alphagate.execution.lifecycle import LegStatus, OrderId, OrderStatus, Submission
from alphagate.execution.mapping import position_intent, to_tool_arguments
from alphagate.execution.pricing import (
    alpaca_limit_price,
    alpaca_limit_price_inverse,
    net_premium_per_unit,
)
from alphagate.execution.session import (
    McpSession,
    RecordedSession,
    SecurityEnvelope,
    ToolResult,
    unwrap,
)
from alphagate.execution.submit import arguments_for, cancel, read_back, submit

__all__ = [
    "ACCOUNT_TOOL",
    "CLOCK_TOOL",
    "PLACE_STOCK_ORDER_TOOL",
    "POSITIONS_TOOL",
    "AccountRead",
    "ExecutionError",
    "LegPosition",
    "LegStatus",
    "MalformedToolOutput",
    "MarketClock",
    "McpSession",
    "OrderId",
    "OrderStatus",
    "PartialFillBreach",
    "RecordedSession",
    "SecurityEnvelope",
    "Submission",
    "ToolResult",
    "ToolTimeout",
    "Tradeability",
    "TransportFailure",
    "UnsubmittableOrder",
    "alpaca_limit_price",
    "alpaca_limit_price_inverse",
    "arguments_for",
    "cancel",
    "cancel_equity",
    "client_order_id",
    "equity_arguments_for",
    "equity_client_order_id",
    "equity_order_fingerprint",
    "load_env_file",
    "mcp_environment",
    "net_premium_per_unit",
    "order_fingerprint",
    "position_intent",
    "read_account",
    "read_back",
    "read_back_equity",
    "read_clock",
    "read_positions",
    "read_share_positions",
    "read_tradeability",
    "require_paper_account",
    "share_positions_from",
    "submit",
    "submit_equity",
    "to_account",
    "to_clock",
    "to_leg_positions",
    "to_stock_arguments",
    "to_tool_arguments",
    "trading_day_of",
    "unwrap",
]
