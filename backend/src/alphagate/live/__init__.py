"""Running the agent for real — the composition root and the CLI.

The only package that knows a live Alpaca account exists. Everything under
`core`, `options`, `risk`, `agent` and `journal` is tested offline and has no
idea this package is here; that is the point of the layering in specs/01, and it
is why the test suite runs without a network.

* `wiring` — assembles the tested parts into a running session.
* `cli` — `python -m alphagate <command>`, including the pre-flight check that
  verifies the four hard gates in specs/00 before a trading day starts.
"""

from alphagate.live.wiring import (
    LiveContext,
    SessionState,
    build_market_data,
    expiry_window,
    gather_for,
    mcp_session,
    slots_for,
)

__all__ = [
    "LiveContext",
    "SessionState",
    "build_market_data",
    "expiry_window",
    "gather_for",
    "mcp_session",
    "slots_for",
]
