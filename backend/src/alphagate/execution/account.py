"""Reading the account and the book — the two GET tools the agent needs.

`submit.py` writes; this only reads. It parses the two payloads every slot needs
before it can propose anything: how much equity the limits are a fraction of,
and what is already on.

Three things are deliberate.

**Money is `Decimal`, parsed from the string Alpaca sent.** Alpaca returns
`"98234.17"`, and the only way to get a float in here is to ask for one. Equity
is the denominator of every budgeted limit in specs/03 D5; a float denominator
makes every limit approximate.

**Option positions come back as legs, because that is what Alpaca has.** There
is no "iron condor" in the positions endpoint — there are four option lines with
a sign and a quantity. This module does not try to guess which four belong
together; pairing legs by staring at strikes is how you end up believing in a
condor that is actually two unrelated spreads. `agent/book.py` does the pairing,
using the journal, which is the only place that actually knows.

**Drawdown is measured against the high-water mark, not against yesterday.**
`last_equity` is what Alpaca gives, and using it alone would report a fresh 0%
drawdown every morning — which would make the kill switch in specs/03 D4
unarmable across days, since the thing it watches would reset overnight. The
peak is the caller's to carry; this returns what the broker said and nothing
more.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Final

from alphagate.core.identifiers import Ticker, ticker
from alphagate.execution.errors import MalformedToolOutput
from alphagate.execution.session import McpSession, SecurityEnvelope, ToolResult
from alphagate.options import OptionContract, Right

__all__ = [
    "ACCOUNT_TOOL",
    "CLOCK_TOOL",
    "POSITIONS_TOOL",
    "AccountRead",
    "LegPosition",
    "MarketClock",
    "read_account",
    "read_clock",
    "read_positions",
    "to_account",
    "to_clock",
    "to_leg_positions",
]

ACCOUNT_TOOL: Final = "get_account_info"
POSITIONS_TOOL: Final = "get_all_positions"
CLOCK_TOOL: Final = "get_clock"

_OPTION_CLASS: Final = "us_option"


@dataclass(frozen=True, slots=True)
class AccountRead:
    """The account, as the broker currently sees it.

    No account id and no account number: specs/06 D4 forbids writing them, and
    the cheapest way to keep a secret out of a journal is not to carry it into
    the process in the first place.
    """

    equity: Decimal
    last_equity: Decimal
    buying_power: Decimal
    options_buying_power: Decimal
    options_level: int
    """Alpaca's **effective** options level — `options_trading_level`, which is
    the minimum of the approved level and any configured cap. Level 2 buys and
    sells covered; level 3 is what spreads need. Below 3 every vertical in the
    menu is unfillable, and finding that out from a rejection at 14:30 is
    finding it out too late.

    Deliberately not `options_approved_level`: an account approved for 3 but
    capped at 2 would read as tradeable and reject every order."""
    cash: Decimal
    multiplier: int
    """Margin classification, and the **only** pattern-day-trader signal the
    account object carries: `4` means PDT. There is no `pattern_day_trader`,
    `daytrade_count` or `daytrading_buying_power` field on this payload, so
    nothing may read them."""
    is_blocked: bool
    envelope: SecurityEnvelope | None
    observed_at: datetime

    @property
    def can_trade_spreads(self) -> bool:
        return self.options_level >= 3 and not self.is_blocked

    @property
    def is_pattern_day_trader(self) -> bool:
        """A 4x multiplier is PDT. Reported, never acted on — the agent places
        a handful of defined-risk option spreads a day and cannot approach the
        threshold, so this is a status line rather than a constraint."""
        return self.multiplier >= 4

    @property
    def session_change(self) -> Decimal:
        """Today's P&L as the broker computes it. For the dashboard, not the Gate."""
        return self.equity - self.last_equity


@dataclass(frozen=True, slots=True)
class LegPosition:
    """One option line in the positions endpoint. Not a structure.

    `quantity` is signed: positive is long, negative is short. Alpaca reports
    `qty` as a signed string and `side` as a word; both are read, and they must
    agree — a disagreement means we are misreading the payload, and a misread
    short leg is a naked position we think is covered.
    """

    contract: OptionContract
    quantity: int
    average_price: Decimal
    market_value: Decimal
    unrealised: Decimal

    @property
    def is_short(self) -> bool:
        return self.quantity < 0

    @property
    def underlying(self) -> Ticker:
        return self.contract.underlying


@dataclass(frozen=True, slots=True)
class MarketClock:
    """The exchange's own clock, which is the only one that knows about holidays.

    A fixed 13:30-20:00 UTC session is wrong three times a year — a half day
    closes at 17:00 UTC — and wrong every weekend, which is when most of the
    debugging happens. `wiring.market_session` is the fallback for when this is
    unreachable, and it says so.

    Nothing in the domain reads this. It reaches the Gate, if at all, as the
    `as_of` argument the caller passes; a clock read inside a pure layer is the
    thing specs/01 Rule 4 exists to prevent.
    """

    is_open: bool
    next_open: datetime
    next_close: datetime
    observed_at: datetime

    @property
    def opens_today(self) -> bool:
        """Whether there is still a session on the observer's own date."""
        return self.is_open or self.next_open.date() == self.observed_at.date()


def read_clock(mcp: McpSession, *, observed_at: datetime) -> MarketClock:
    """Ask the exchange whether it is open. One call, no retry."""
    return to_clock(mcp.call(CLOCK_TOOL, {}), observed_at=observed_at)


def to_clock(result: ToolResult, *, observed_at: datetime) -> MarketClock:
    """Parse `get_clock`. Pure.

    Alpaca answers with offset-aware Eastern timestamps
    (`2026-08-31T09:30:00-04:00`), which are converted to UTC here rather than
    carried around in a second timezone. specs/01 Rule 5: all times are tz-aware
    UTC, and the boundary is where that becomes true.
    """
    data = result.data
    return MarketClock(
        is_open=bool(data.get("is_open")),
        next_open=_moment(data, "next_open", default=observed_at),
        next_close=_moment(data, "next_close", default=observed_at),
        observed_at=observed_at,
    )


def _moment(data: Mapping[str, Any], key: str, *, default: datetime) -> datetime:
    raw = data.get(key)
    if not isinstance(raw, str) or not raw:
        return default
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return default
    return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def read_account(mcp: McpSession, *, observed_at: datetime) -> AccountRead:
    """Ask the broker for the account. One call, no retry.

    A failure here is not recoverable by trying again in the same slot: without
    equity there is no budget, and every limit in specs/03 D5 is a fraction of
    it. The caller skips the slot and journals why.
    """
    return to_account(mcp.call(ACCOUNT_TOOL, {}), observed_at=observed_at)


def read_positions(mcp: McpSession) -> tuple[LegPosition, ...]:
    """Ask the broker what is open. Option legs only; equities are ignored."""
    return to_leg_positions(mcp.call(POSITIONS_TOOL, {}))


def to_account(result: ToolResult, *, observed_at: datetime) -> AccountRead:
    """Parse `get_account_info`. Pure."""
    data = result.data
    return AccountRead(
        equity=_money(data, "equity"),
        last_equity=_money(data, "last_equity", default=_money(data, "equity")),
        buying_power=_money(data, "buying_power", default=Decimal(0)),
        options_buying_power=_money(data, "options_buying_power", default=Decimal(0)),
        options_level=_int(data.get("options_trading_level")),
        cash=_money(data, "cash", default=Decimal(0)),
        multiplier=_int(data.get("multiplier")),
        is_blocked=bool(data.get("account_blocked")) or bool(data.get("trading_blocked")),
        envelope=result.envelope,
        observed_at=observed_at,
    )


def to_leg_positions(result: ToolResult) -> tuple[LegPosition, ...]:
    """Parse `get_all_positions`, keeping the option lines.

    Equity positions are dropped rather than rejected: the competition account
    may hold shares from a covered call, and a stock line is not an error. What
    would be an error is an option line we cannot parse, and that raises —
    silently skipping an open short leg is the single worst failure available
    to this function.
    """
    legs: list[LegPosition] = []
    for row in _rows(result):
        if str(row.get("asset_class", "")) != _OPTION_CLASS:
            continue
        legs.append(_leg(row, tool=result.tool))
    return tuple(legs)


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


def _leg(row: Mapping[str, Any], *, tool: str) -> LegPosition:
    symbol = str(row.get("symbol", ""))
    contract = _contract(symbol, tool=tool)
    quantity = _int(row.get("qty"))
    side = str(row.get("side", "")).lower()
    if side == "short" and quantity > 0:
        quantity = -quantity
    if side == "long" and quantity < 0:
        raise MalformedToolOutput(
            f"{tool} reports {symbol} as long with quantity {quantity}; "
            "refusing to guess which field is right"
        )
    if quantity == 0:
        raise MalformedToolOutput(f"{tool} reports {symbol} with zero quantity")
    return LegPosition(
        contract=contract,
        quantity=quantity,
        average_price=_money(row, "avg_entry_price", default=Decimal(0)),
        market_value=_money(row, "market_value", default=Decimal(0)),
        unrealised=_money(row, "unrealized_pl", default=Decimal(0)),
    )


def _contract(symbol: str, *, tool: str) -> OptionContract:
    """Parse an OCC symbol: `SPY260904P00752000`.

    Root is variable length, then six digits of date, then C or P, then eight
    digits of strike in thousandths. Parsed from the right, because the root is
    the only part whose length is not fixed.
    """
    if len(symbol) < 16:
        raise MalformedToolOutput(f"{tool} returned an unparseable option symbol {symbol!r}")
    body = symbol[-15:]
    root = symbol[: -len(body)]
    try:
        expiry = date(
            2000 + int(body[0:2]),
            int(body[2:4]),
            int(body[4:6]),
        )
        right = Right.CALL if body[6].upper() == "C" else Right.PUT
        strike = Decimal(int(body[7:])) / 1000
    except (ValueError, InvalidOperation) as exc:
        raise MalformedToolOutput(
            f"{tool} returned an unparseable option symbol {symbol!r}"
        ) from exc
    if body[6].upper() not in {"C", "P"}:
        raise MalformedToolOutput(f"{tool} returned {symbol!r} with no call/put marker")
    return OptionContract(ticker(root), expiry, strike, right)


def _money(data: Mapping[str, Any], key: str, *, default: Decimal | None = None) -> Decimal:
    raw = data.get(key)
    if raw is None or raw == "":
        if default is not None:
            return default
        raise MalformedToolOutput(f"account payload has no {key!r}")
    try:
        return Decimal(str(raw))
    except InvalidOperation as exc:
        raise MalformedToolOutput(f"{key} is not a number: {raw!r}") from exc


def _int(value: object) -> int:
    if value is None or value == "":
        return 0
    try:
        return int(Decimal(str(value)))
    except (InvalidOperation, ValueError):
        return 0
