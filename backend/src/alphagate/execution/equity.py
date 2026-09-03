"""The equity door — specs/09 D7.

```python
def submit_equity(order: GatedEquityOrder, mcp: McpSession) -> Submission
def read_share_positions(mcp: McpSession) -> tuple[SharePosition, ...]
def read_tradeability(mcp: McpSession, symbols) -> Mapping[Ticker, Tradeability]
```

`GatedEquityOrder` is constructible only inside `risk.equity_gate`
([09](../../../../specs/09-equity-execution.md) D5), and `submit_equity`
accepts nothing else. Second door, same rule: there is no REST fallback,
because a fallback is a bypass.

Every field below was read off the live `place_stock_order` schema
(alpaca-mcp-server 3.4.7), not inferred from documentation.

**Market, day, no extended hours.** A limit order on a rebalance is a rebalance
that may not happen. The book is a set of weights to be *holding*, not a price
to be got, and an unfilled leg leaves the account in a state neither the
backtest nor the plan describes. Fractional quantities require market/day
anyway, so the two constraints agree.

**A timeout is not a failure.** Identical discipline to
[04](../../../../specs/04-execution.md) D4: it is an unknown outcome, resolved
by reading back the derived `client_order_id`, never by resubmitting. This is
the only control-flow decision in the module that matters, and it is why
`ToolTimeout` is not in the retry loop.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from hashlib import blake2b
from typing import Any, Final

from alphagate.core.errors import InvariantViolation
from alphagate.core.identifiers import Ticker, ticker
from alphagate.equity.plan import Holding
from alphagate.execution.errors import (
    BrokerRefused,
    ExecutionError,
    MalformedToolOutput,
    ToolTimeout,
    TransportFailure,
    UnsubmittableOrder,
)
from alphagate.execution.idempotency import PREFIX, trading_day_of
from alphagate.execution.lifecycle import OrderId, Submission, submission_from
from alphagate.execution.session import McpSession, ToolArgument, ToolResult
from alphagate.risk.equity_verdict import GatedEquityOrder

__all__ = [
    "ASSETS_TOOL",
    "MAX_ATTEMPTS",
    "PLACE_STOCK_ORDER_TOOL",
    "SNAPSHOT_TOOL",
    "TIME_IN_FORCE",
    "Tradeability",
    "equity_arguments_for",
    "equity_client_order_id",
    "equity_order_fingerprint",
    "read_share_positions",
    "share_positions_from",
    "submit_equity",
    "to_stock_arguments",
]

PLACE_STOCK_ORDER_TOOL: Final = "place_stock_order"
ASSETS_TOOL: Final = "get_asset"
SNAPSHOT_TOOL: Final = "get_stock_snapshot"
POSITIONS_TOOL: Final = "get_all_positions"
READ_BACK_TOOL: Final = "get_order_by_client_id"
CANCEL_TOOL: Final = "cancel_order_by_id"

TIME_IN_FORCE: Final = "day"
"""Not a default — the only value a fractional market order accepts."""

ORDER_TYPE: Final = "market"
MAX_ATTEMPTS: Final = 3
BACKOFF_BASE: Final = 0.5
"""Seconds. 0.5, 1.0 — two waits, because there are three attempts."""

_EQUITY_CLASS: Final = "us_equity"


# --------------------------------------------------------------------------- #
# Idempotency
# --------------------------------------------------------------------------- #


def equity_order_fingerprint(order: GatedEquityOrder, trading_day: date) -> str:
    """The canonical string an id is derived from. Stable and human-readable.

    Returned rather than kept private so a mismatch is debuggable: when two
    orders that should share an id do not, you want to diff these, not two hex
    digests.

    **The quantity is not in it.** Two passes on the same day that both want to
    buy the same name are the same rebalance decision arrived at twice, and the
    second must be refused by the broker rather than doubled — which is exactly
    what a duplicate `client_order_id` does. Putting the share count in the key
    would make a re-plan after a price tick into a second position.

    The book's own session *is* in it, so tomorrow's book legitimately produces
    a different order for the same name.
    """
    return "/".join(
        (
            order.fingerprint,
            order.book_as_of,
            str(order.symbol),
            order.side.value,
            trading_day.isoformat(),
        )
    )


def equity_client_order_id(
    order: GatedEquityOrder, trading_day: date | None = None
) -> str:
    """A stable, derived idempotency key. Never random.

    A UUID regenerated on retry is not an idempotency key, it is a second order.
    `trading_day` defaults to the **Eastern** date of approval, for the reason
    [04](../../../../specs/04-execution.md) D4 gives: hashing the UTC date would
    split one afternoon across two keys.
    """
    day = trading_day if trading_day is not None else trading_day_of(order.approved_at)
    digest = blake2b(
        equity_order_fingerprint(order, day).encode("utf-8"), digest_size=16
    ).hexdigest()
    return f"{PREFIX}-eq-{digest[:21]}"


# --------------------------------------------------------------------------- #
# Mapping
# --------------------------------------------------------------------------- #


def to_stock_arguments(order: GatedEquityOrder) -> dict[str, ToolArgument]:
    """Render a gated equity order as `place_stock_order` arguments.

    `qty` is a string, as the schema wants, and carries the fractional places
    the Gate approved. A non-fractionable asset is asserted whole here as well
    as clamped in the planner: sending `12.5000` for a name Alpaca will not
    split is a rejection at 14:31, and the planner's rounding is the kind of
    thing a later refactor quietly relaxes.

    `notional` is deliberately never sent. It is mutually exclusive with `qty`,
    and reconciliation is in shares — an order placed in dollars comes back as
    a share count nobody predicted, which turns the next pass's diff into a
    guess.
    """
    if order.shares <= 0:
        raise UnsubmittableOrder(
            f"{order.symbol}: refusing to send a non-positive quantity {order.shares}"
        )
    if not order.fractionable and order.shares != order.shares.to_integral_value():
        raise UnsubmittableOrder(
            f"{order.symbol}: {order.shares} shares of a non-fractionable asset. "
            "The planner rounds these to whole shares; a fractional one here means "
            "that rounding was bypassed."
        )
    return {
        "symbol": str(order.symbol),
        "side": order.side.value,
        "qty": _quantity(order),
        "type": ORDER_TYPE,
        "time_in_force": TIME_IN_FORCE,
    }


def _quantity(order: GatedEquityOrder) -> str:
    """Shares as Alpaca wants them: a plain decimal string, no exponent.

    `Decimal.__str__` produces `1E+2` for `Decimal("100").normalize()` and
    Alpaca's parser does not accept it. Normalising the *representation* rather
    than the value is what `to_integral_value` and the explicit format below are
    for.
    """
    if order.shares == order.shares.to_integral_value():
        return str(int(order.shares))
    return format(order.shares.normalize(), "f")


def equity_arguments_for(
    order: GatedEquityOrder, trading_day: date | None = None
) -> Mapping[str, ToolArgument]:
    """The exact payload `submit_equity` would send. For the journal and for tests."""
    return {
        **to_stock_arguments(order),
        "client_order_id": equity_client_order_id(order, trading_day),
    }


# --------------------------------------------------------------------------- #
# The door
# --------------------------------------------------------------------------- #


def submit_equity(
    order: GatedEquityOrder,
    mcp: McpSession,
    *,
    trading_day: date | None = None,
    sleep: Any = time.sleep,
) -> Submission:
    """Send one gated equity order. The only way a share order reaches Alpaca.

    `sleep` is injected so the retry path is testable without waiting; it is not
    a knob for production.
    """
    if not isinstance(order, GatedEquityOrder):
        raise TypeError(
            f"submit_equity() accepts a GatedEquityOrder, got {type(order).__name__}. "
            "specs/09 D7: every equity order reaching Alpaca passed the equity Gate. "
            "There is no bypass — construct one with risk.equity_gate.evaluate_equity."
        )

    key = equity_client_order_id(order, trading_day)
    arguments: dict[str, ToolArgument] = {
        **to_stock_arguments(order),
        "client_order_id": key,
    }

    last_failure: TransportFailure | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            result = mcp.call(PLACE_STOCK_ORDER_TOOL, arguments)
        except ToolTimeout:
            # Unknown outcome, not a failure. Ask what happened; never resend.
            return read_back_equity(
                key, mcp, submitted_at=order.approved_at, attempts=attempt
            )
        except TransportFailure as failure:
            last_failure = failure
            if attempt == MAX_ATTEMPTS:
                break
            sleep(BACKOFF_BASE * (2 ** (attempt - 1)))
            continue

        _reject_error_payload(result)
        return submission_from(
            result,
            client_order_id=key,
            submitted_at=order.approved_at,
            attempts=attempt,
        )

    raise TransportFailure(
        f"{PLACE_STOCK_ORDER_TOOL} failed {MAX_ATTEMPTS} times for {key}: {last_failure}"
    )


def read_back_equity(
    key: str,
    mcp: McpSession,
    *,
    submitted_at: datetime,
    attempts: int = 1,
) -> Submission:
    """Resolve an unknown outcome by asking what the broker has.

    A `get` that itself fails is left to raise. Guessing here would mean deciding
    between "no order exists" and "an order exists that we cannot see", and those
    two guesses have opposite consequences.

    There is no partial-fill guard, and the difference from the options door is
    the point: a single-leg share order that fills 40 of 100 shares is a
    perfectly ordinary outcome, not a naked leg. The next pass diffs against
    what is actually held and finishes the job.
    """
    result = mcp.call(READ_BACK_TOOL, {"client_order_id": key})
    if _looks_like_not_found(result):
        raise ExecutionError(
            f"equity order {key} timed out and cannot be read back; it may or may "
            "not exist. Reconcile by hand before submitting anything else — "
            "resubmitting blind is how one position becomes two (specs/04 D4)."
        )
    return submission_from(
        result,
        client_order_id=key,
        submitted_at=submitted_at,
        attempts=attempts,
        resolved_by_readback=True,
    )


def cancel_equity(order_id: OrderId, mcp: McpSession) -> None:
    """Cancel a live order. Idempotent at the broker; not retried here."""
    _reject_error_payload(mcp.call(CANCEL_TOOL, {"order_id": str(order_id)}))


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Tradeability:
    """What the broker will let us do with one symbol.

    `fractionable` is the field that decides whether a $192 sleeve position is
    expressible at all. On a $500 non-fractionable name it is not, and the
    planner records the resulting hole rather than rounding it away silently
    (specs/09 D2).
    """

    symbol: Ticker
    tradeable: bool
    fractionable: bool


def read_share_positions(mcp: McpSession) -> tuple[Holding, ...]:
    """Ask the broker what shares are held. Equity lines only.

    The mirror image of `account.read_positions`, which keeps the option legs
    and drops these. Two readers over one payload rather than one reader with a
    mode, because the two produce different types and a function that returns
    either is a function every caller has to narrow.
    """
    return share_positions_from(mcp.call(POSITIONS_TOOL, {}))


def share_positions_from(result: ToolResult) -> tuple[Holding, ...]:
    """Parse `get_all_positions`, keeping the equity lines. Pure.

    A line we cannot parse **raises**. Silently skipping a holding understates
    the book, and an understated holding is how the next pass's sell becomes a
    short.
    """
    holdings: list[Holding] = []
    for row in _rows(result):
        if str(row.get("asset_class", "")) != _EQUITY_CLASS:
            continue
        holdings.append(_holding(row, tool=result.tool))
    return tuple(sorted(holdings, key=lambda h: str(h.symbol)))


def _holding(row: Mapping[str, Any], *, tool: str) -> Holding:
    symbol = str(row.get("symbol", ""))
    try:
        name = ticker(symbol)
    except InvariantViolation as exc:
        raise MalformedToolOutput(f"{tool} returned an unparseable symbol {symbol!r}") from exc
    shares = _decimal(row.get("qty"), field="qty", symbol=symbol, tool=tool)
    side = str(row.get("side", "")).lower()
    if side == "short" and shares > 0:
        shares = -shares
    if side == "long" and shares < 0:
        raise MalformedToolOutput(
            f"{tool} reports {symbol} as long with quantity {shares}; "
            "refusing to guess which field is right"
        )
    return Holding(
        symbol=name,
        shares=shares,
        average_price=_decimal(
            row.get("avg_entry_price"), field="avg_entry_price", symbol=symbol,
            tool=tool, default=Decimal(0),
        ),
        market_value=_decimal(
            row.get("market_value"), field="market_value", symbol=symbol,
            tool=tool, default=Decimal(0),
        ),
    )


def read_tradeability(
    mcp: McpSession, symbols: Iterable[Ticker]
) -> dict[Ticker, Tradeability]:
    """Ask the broker about each symbol, one call per name.

    One call per symbol because `get_asset` takes one. On a 104-name book that
    is 104 round trips, which is why the caller caches this for a session — the
    answers change on the order of once a year, and re-asking every heartbeat
    would spend the whole slot on it.

    A symbol the broker does not know is **absent from the result**, not
    defaulted to tradeable. The planner then skips it with `NOT_TRADEABLE`,
    which is the honest reading of "we could not find out".
    """
    found: dict[Ticker, Tradeability] = {}
    for symbol in sorted(set(symbols), key=str):
        try:
            result = mcp.call(ASSETS_TOOL, {"symbol_or_asset_id": str(symbol)})
        except (TransportFailure, ToolTimeout, MalformedToolOutput):
            continue
        data = result.data
        if not data or data.get("error"):
            continue
        found[symbol] = Tradeability(
            symbol=symbol,
            tradeable=bool(data.get("tradable", False)),
            fractionable=bool(data.get("fractionable", False)),
        )
    return found


# --------------------------------------------------------------------------- #


def _reject_error_payload(result: ToolResult) -> None:
    """Turn a tool-level error object into an exception.

    The MCP server answers some failures with `{"error": ...}` and HTTP 200. A
    response that says "error" and is read as an order is an order that does not
    exist being recorded as one that does.

    `BrokerRefused` rather than a bare `MalformedToolOutput`: the payload is
    perfectly readable and says the broker declined. The distinction is only for
    whoever reads the journal afterwards — see the class docstring — and the two
    are caught identically everywhere.
    """
    data = result.data
    error = data.get("error") or data.get("detail")
    if error and "status" not in data:
        raise BrokerRefused(f"{result.tool} returned an error: {error}")


def _looks_like_not_found(result: ToolResult) -> bool:
    if "status" in result.data:
        return False
    blob = result.raw.lower()
    return "not found" in blob or "404" in blob


def _rows(result: ToolResult) -> Iterator[Mapping[str, Any]]:
    data: Any = result.data
    if isinstance(data, Mapping):
        data = data.get("result", data)
    if isinstance(data, Mapping):
        data = data.values()
    if not isinstance(data, Sequence | list | tuple) and not hasattr(data, "__iter__"):
        raise MalformedToolOutput(f"{result.tool} returned no position list")
    for row in data:
        if isinstance(row, Mapping):
            yield row


def _decimal(
    raw: Any,
    *,
    field: str,
    symbol: str,
    tool: str,
    default: Decimal | None = None,
) -> Decimal:
    if raw is None or raw == "":
        if default is not None:
            return default
        raise MalformedToolOutput(f"{tool}: {symbol} has no {field!r}")
    try:
        return Decimal(str(raw))
    except InvalidOperation as exc:
        raise MalformedToolOutput(
            f"{tool}: {symbol} {field} is not a number: {raw!r}"
        ) from exc
