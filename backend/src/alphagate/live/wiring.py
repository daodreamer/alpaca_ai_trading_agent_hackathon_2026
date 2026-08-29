"""The composition root — where the tested parts become a running agent.

Everything below this line has been unit-tested offline for a week. This module
is the first place any of it touches a real socket, and it is deliberately the
only one: nothing here is clever, and nothing anywhere else knows that a live
Alpaca account exists.

Four things it owns and nothing else does.

**The paper guard runs before anything opens.** `require_paper_account` is
checked at the top of `session_for`, not somewhere polite in the middle. The one
rule this project cannot negotiate is that no order reaches real money, and the
place to enforce it is before the transport exists rather than after.

**The book is re-read every slot.** specs/05's runner promises "nothing is
inferred about positions", and this is what makes that true: account and
positions come off the broker at the top of every gather, get matched against
the journal by `agent/book.py`, and produce a fresh `PortfolioSnapshot`. One
extra pair of calls per fifteen minutes is not a budget worth economising
against; a position we believe in and the broker does not is.

**One underlying per slot, round-robin.** The runner runs one cycle per slot and
a cycle is about one name. With three tradeable ETFs and a slot every fifteen
minutes, each name is considered roughly every forty-five — which spreads the
day's fills across the watchlist instead of concentrating them in whichever name
happened to be first. Depth per name would be the other choice; breadth is worth
more when the P&L sample is four days long and the risk of one name's bad
afternoon dominating it is real.

**The peak equity is persisted.** specs/03 D4's kill switch watches drawdown
from the high-water mark, and a mark that resets at midnight is a kill switch
that cannot latch across the days it exists for. It lives in a small JSON file
beside the journal.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

from alphagate.agent import (
    CycleInputs,
    IvHistoryStore,
    MarketRead,
    Slot,
    Underlying,
    build_candidates,
    next_slot,
    perceive,
    session_slots,
    tradeable_today,
    vertical_credit_spreads,
)
from alphagate.agent.book import BookRead, HeldPosition, read_book
from alphagate.agent.cycle import CycleRecord, run_exit_cycle
from alphagate.agent.exits import DEFAULT_EXIT_POLICY, ExitPolicy
from alphagate.core.bar import Feed
from alphagate.core.errors import InvariantViolation
from alphagate.core.identifiers import Ticker
from alphagate.execution import (
    AccountRead,
    McpSession,
    read_account,
    read_positions,
    require_paper_account,
)
from alphagate.journal import Journal
from alphagate.live.status import build_status, write_status
from alphagate.marketdata import MarketData
from alphagate.marketdata.alpaca import AlpacaMarketData
from alphagate.options import OptionContract, OptionQuote, Right, StructureRisk, compute_risk
from alphagate.risk import DEFAULT_LIMITS, RiskLimits

__all__ = [
    "LiveContext",
    "SessionState",
    "build_market_data",
    "expiry_window",
    "gather_for",
    "market_session",
    "mcp_session",
]

EXIT_SEQUENCE_BASE = 500
"""Exit cycle ids start here.

A session has at most a few dozen slots, so nothing above 500 collides with an
entry — and an exit and an entry in the same slot minting the same `cycle_id`
would collapse two decisions into one journal line (specs/06 D2)."""

DTE_MIN = 3
DTE_MAX = 21
"""The expiry window candidates are drawn from — specs/07 D5.

Wider than the Gate's own window on purpose: the chain request should return
everything the Gate might accept, and let the Gate refuse. A request narrower
than the rule would hide candidates from the menu and call it a risk decision.
"""


def build_market_data(env: dict[str, str], *, feed: str | None = None) -> AlpacaMarketData:
    """The read-only REST client. GET only — adr/0002 D2."""
    kwargs: dict[str, Any] = {
        "key_id": env["ALPACA_API_KEY_ID"],
        "secret_key": env["ALPACA_API_SECRET_KEY"],
    }
    if env.get("ALPACA_DATA_URL"):
        kwargs["base_url"] = env["ALPACA_DATA_URL"]
    if feed:
        kwargs["feed"] = Feed(feed)
    return AlpacaMarketData(**kwargs)


def mcp_session(env: dict[str, str], *, timeout: float = 30.0) -> Any:
    """Start the Alpaca MCP server as a subprocess. Paper-guarded first.

    The guard is the first statement rather than a check somewhere inside: a
    transport that could reach a live account should not come into existence.
    """
    require_paper_account(env)
    from alphagate.execution.credentials import mcp_environment
    from alphagate.execution.stdio import StdioSession

    return StdioSession(env=mcp_environment(env), timeout=timeout)


@dataclass
class SessionState:
    """What must survive a restart, and where it lives.

    Small on purpose. Anything that can be recovered from the journal is not in
    here — the journal is the record, and a second source of truth for the same
    fact is a second source of disagreement.
    """

    path: Path
    peak_equity: Decimal | None = None
    killswitch_tripped: bool = False

    @classmethod
    def load(cls, path: Path) -> SessionState:
        if not path.is_file():
            return cls(path=path)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return cls(path=path)
        peak = raw.get("peak_equity")
        return cls(
            path=path,
            peak_equity=Decimal(str(peak)) if peak is not None else None,
            killswitch_tripped=bool(raw.get("killswitch_tripped")),
        )

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                {
                    "peak_equity": str(self.peak_equity) if self.peak_equity else None,
                    "killswitch_tripped": self.killswitch_tripped,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    def observe(self, equity: Decimal) -> None:
        """Raise the high-water mark. Never lowers it — that is the whole job."""
        if self.peak_equity is None or equity > self.peak_equity:
            self.peak_equity = equity
            self.save()


@dataclass
class LiveContext:
    """Everything one live session needs, assembled once."""

    data: MarketData
    mcp: McpSession | None
    journal: Journal
    iv: IvHistoryStore
    state: SessionState
    limits: RiskLimits = DEFAULT_LIMITS
    exit_policy: ExitPolicy = DEFAULT_EXIT_POLICY
    """The exit thresholds — specs/07 D6.

    On the context rather than defaulted inside `run_exit_cycle`, because
    `ExitPolicy` says of itself: "one place, logged at startup, shown in the
    dashboard". A threshold reachable only as a function default is a threshold
    nobody can print."""
    universe: tuple[Underlying, ...] = field(default_factory=tradeable_today)
    right: Right = Right.PUT
    """Put credit spreads by default — specs/07 D4's bullish/neutral expression.

    A single default rather than a switch: the direction the strategy takes is
    a strategy decision (07 D4), and threading it through the wiring as an
    option would put it somewhere nobody reads."""

    last_book: BookRead | None = None
    """The most recent book read, kept for the caller's reporting."""

    last_account: AccountRead | None = None
    """The account behind `last_book`. Kept so the status snapshot reports the
    same read the Gate budgeted against, rather than a second one taken a
    moment later that would disagree with it by a tick."""

    next_slot_at: datetime | None = None
    """When the following cycle is due. Set by the runner's gather so the status
    page can say *when*, not merely *what* — "idle" and "idle, next at 15:05"
    are different amounts of reassurance."""

    def underlying_for(self, slot: Slot) -> Underlying:
        """Round-robin across the watchlist. See the module docstring."""
        return self.universe[slot.sequence % len(self.universe)]

    def account(self, *, as_of: datetime) -> AccountRead:
        if self.mcp is None:
            raise RuntimeError("no MCP session: cannot read the account")
        return read_account(self.mcp, observed_at=as_of)

    def book(self, *, as_of: datetime, fills_today: int = 0) -> BookRead:
        """Re-read the account and the book from the broker. Never inferred."""
        if self.mcp is None:
            raise RuntimeError("no MCP session: cannot read the book")
        account = read_account(self.mcp, observed_at=as_of)
        self.last_account = account
        self.state.observe(account.equity)
        legs = read_positions(self.mcp)
        book = read_book(
            account,
            legs,
            self.journal.read(as_of.date()),
            peak_equity=self.state.peak_equity,
            fills_today=fills_today,
            killswitch_tripped=self.state.killswitch_tripped,
        )
        self.last_book = book
        return book


def expiry_window(as_of: datetime) -> tuple[date, date]:
    today = as_of.date()
    return today + timedelta(days=DTE_MIN), today + timedelta(days=DTE_MAX)


def gather_for(
    context: LiveContext,
    *,
    fills: Sequence[int] = (),
    submit_exits: bool = True,
    slots: Sequence[Slot] = (),
) -> Any:
    """Build the runner's `gather` callable.

    One closure over one context, because `run_session` wants a function of a
    slot and everything else it needs is fixed for the session. `fills` is a
    one-element mutable counter the caller can read — the Gate's daily fill cap
    (specs/03 D5) is a portfolio fact and the snapshot has to carry it.
    """
    counter = list(fills) or [0]
    schedule = tuple(slots)

    def gather(slot: Slot) -> CycleInputs:
        """Everything one slot needs — including the exits, evaluated first.

        Exits are computed here rather than in the runner because the runner is
        deliberately ignorant of market data (specs/05 D1), and re-pricing an
        open position needs a chain. `CycleInputs.exits` is the seam the runner
        exposes for exactly this, and leaving it empty — as the first version of
        this file did — produces an agent that opens positions and never closes
        one."""
        underlying = context.underlying_for(slot)
        symbol = underlying.symbol
        low, high = expiry_window(slot.at)

        chain = context.data.option_chain(
            symbol,
            expiry_from=low,
            expiry_to=high,
            right=_api_right(context.right),
        )
        perception = perceive(
            context.data,
            symbol,
            as_of=slot.at,
            chain=chain,
            history=context.iv.observations(symbol),
        )
        upcoming = next_slot(schedule, slot.at) if schedule else None
        context.next_slot_at = upcoming.at if upcoming else None
        book = context.book(as_of=slot.at, fills_today=counter[0])
        account = context.last_account
        if account is None:  # pragma: no cover - book() always sets it
            raise RuntimeError('book() did not record an account read')

        structures = vertical_credit_spreads(
            chain, right=context.right, width=underlying.spread_width, as_of=slot.at
        )
        candidates = build_candidates(
            structures,
            limits=context.limits,
            equity=book.snapshot.equity,
            as_of=slot.at,
            book_delta=book.snapshot.net_delta or 0.0,
        )
        marks: dict[str, StructureRisk] = {}
        exits = _exit_cycles(
            context,
            book,
            as_of=slot.at,
            base_sequence=slot.sequence,
            mcp=context.mcp if submit_exits else None,
            marks=marks,
        )
        _publish_status(
            context,
            book,
            account,
            slot=slot,
            marks=marks,
            note=f"{len(candidates)} candidates on {symbol}",
        )
        return CycleInputs(
            read=perception.read,
            candidates=candidates,
            portfolio=book.snapshot,
            exits=exits,
        )

    return gather


_API_RIGHT = {Right.PUT: "put", Right.CALL: "call"}
"""`Right.value` is the OCC letter — `P`, `C` — because that is what an option
symbol is built from. Alpaca's snapshot endpoint wants the English word and
answers `400 Bad Request` to the letter.

Two vocabularies for one fact, and the translation belongs at the boundary that
needs it rather than in the domain type. Found by running it: the first version
sent `type=p` and the whole chain request failed, which would have been a silent
`NO_CANDIDATES` for the entire trading day."""


def _api_right(right: Right) -> str:
    return _API_RIGHT[right]


def _quotes_for(
    data: MarketData, held: Sequence[HeldPosition], *, as_of: datetime
) -> Mapping[OptionContract, OptionQuote]:
    """Fresh quotes for every leg of every open position.

    One chain request per (underlying, expiry) rather than per position, because
    two spreads on the same expiry share a request and a slot with four
    positions should not cost four round trips.

    The window is the position's own expiry, not the entry window: a position
    held into its last week is exactly the one the DTE rule is about, and a
    request built from the *entry* range would stop returning it precisely when
    it matters most.
    """
    wanted: set[tuple[Ticker, date]] = {
        (item.position.underlying, item.position.structure.expiry) for item in held
    }
    quotes: dict[OptionContract, OptionQuote] = {}
    for symbol, expiry in sorted(wanted, key=lambda pair: (str(pair[0]), pair[1])):
        quotes.update(
            data.option_chain(symbol, expiry_from=expiry, expiry_to=expiry)
        )
    del as_of
    return quotes


def _exit_cycles(
    context: LiveContext,
    book: BookRead,
    *,
    as_of: datetime,
    base_sequence: int,
    mcp: McpSession | None,
    marks: dict[str, StructureRisk] | None = None,
) -> tuple[CycleRecord, ...]:
    """Evaluate every open position, and close the ones the policy says to.

    A position we cannot re-price is **held, loudly**, not closed. Missing
    quotes mean we do not know what it is worth, and closing on that basis is
    guessing at a price; the alternative — holding a position whose exit rule
    may have fired — is the one that a later slot can still fix.

    Sequences start well above the entry cycles' so an exit and an entry in the
    same slot cannot mint the same `cycle_id` (specs/06 D2).
    """
    if not book.held:
        return ()

    try:
        quotes = _quotes_for(context.data, book.held, as_of=as_of)
    except Exception:
        return ()

    records: list[CycleRecord] = []
    for offset, item in enumerate(book.held):
        legs = [leg.contract for leg in item.position.structure.legs]
        if any(leg not in quotes for leg in legs):
            continue
        try:
            current = compute_risk(item.position.structure, quotes, as_of)
        except InvariantViolation:
            continue
        if marks is not None:
            marks[item.cycle_id] = current
        record = run_exit_cycle(
            held=item,
            current=current,
            read=_position_read(item, as_of),
            portfolio=book.snapshot,
            limits=context.limits,
            as_of=as_of,
            mcp=mcp,
            sequence=EXIT_SEQUENCE_BASE + base_sequence * 10 + offset,
            policy=context.exit_policy,
        )
        if record is not None:
            records.append(record)
    return tuple(records)


def _position_read(item: HeldPosition, as_of: datetime) -> MarketRead:
    """The minimum honest `MarketRead` for an exit line.

    An exit is not a perception-driven decision — it reads the position's own
    marks, not the trend engine — so inventing a full read here would put
    numbers in the journal that no engine produced. Spot is the one field a read
    may never omit, and the position's own short strike is the closest thing to
    a price this path actually knows.
    """
    return MarketRead(
        underlying=item.position.underlying,
        as_of=as_of,
        spot=item.position.structure.legs[0].contract.strike,
    )


def _publish_status(
    context: LiveContext,
    book: BookRead,
    account: AccountRead,
    *,
    slot: Slot,
    marks: dict[str, StructureRisk],
    note: str,
) -> None:
    """Drop a snapshot for the dashboard. Never raises.

    Status is a convenience; trading is not. A disk that is full or a directory
    that is read-only must cost the operator a stale page, not a trading slot.
    """
    try:
        snapshot = build_status(
            account=account,
            book=book,
            limits=context.limits,
            policy=context.exit_policy,
            marks=marks,
            as_of=slot.at,
            next_slot=context.next_slot_at,
            slot_sequence=slot.sequence,
            universe=tuple(str(u.symbol) for u in context.universe),
            peak_equity=context.state.peak_equity,
            stage_counts=_stages_today(context, slot.at),
            note=note,
        )
        write_status(snapshot, directory=context.journal.directory)
    except Exception:
        return


def _stages_today(context: LiveContext, as_of: datetime) -> dict[str, int]:
    counts: dict[str, int] = {}
    try:
        records = context.journal.read(as_of.date())
    except OSError:
        return counts
    for record in records:
        stage = str(record.get("stage", ""))
        if stage:
            counts[stage] = counts.get(stage, 0) + 1
    return counts


def market_session(as_of: datetime | None = None) -> tuple[datetime, datetime]:
    """Today's regular trading hours, in UTC.

    A fixed 13:30–20:00 UTC is *wrong* three times a year — a half day closes at
    17:00 UTC — and the correct source is the exchange clock. This is the
    fallback for when the clock is unreachable; `cli.py` prefers the broker's
    own `get_clock` and says which one it used.
    """
    moment = as_of or datetime.now(UTC)
    day = moment.date()
    return (
        datetime(day.year, day.month, day.day, 13, 30, tzinfo=UTC),
        datetime(day.year, day.month, day.day, 20, 0, tzinfo=UTC),
    )


def slots_for(open_at: datetime, close_at: datetime) -> tuple[Slot, ...]:
    return session_slots(open_at, close_at)
