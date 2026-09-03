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
from hashlib import sha256
from pathlib import Path
from typing import Any, Final

from alphagate.agent import (
    CycleInputs,
    IvHistoryStore,
    MarketRead,
    OptionBook,
    OptionRule,
    Slot,
    Underlying,
    UnusableOptionBook,
    build_candidates,
    load_option_book,
    next_slot,
    perceive,
    session_slots,
    spreads_by_delta,
    tradeable_today,
    vertical_credit_spreads,
)
from alphagate.agent.book import BookRead, HeldPosition, read_book
from alphagate.agent.cycle import CycleRecord, run_exit_cycle
from alphagate.agent.exits import DEFAULT_EXIT_POLICY, ExitPolicy
from alphagate.agent.screen import BookScreen, DefaultScreen, Screen
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
from alphagate.interface.read import stage_tally
from alphagate.journal import Journal
from alphagate.live.equity import UnpinnedBook, find_latest_book, unpinned_books
from alphagate.live.status import build_status, write_status
from alphagate.marketdata import MarketData
from alphagate.marketdata.alpaca import AlpacaMarketData
from alphagate.options import (
    OptionContract,
    OptionQuote,
    OptionStructure,
    Right,
    Side,
    StructureRisk,
    compute_risk,
)
from alphagate.risk import RiskLimits
from alphagate.risk.limits import OPTIONS_SLEEVE_ALLOCATION, SLEEVE_LIMITS

__all__ = [
    "LiveContext",
    "SessionState",
    "build_market_data",
    "expiry_window",
    "gather_for",
    "load_pinned_option_book",
    "market_session",
    "mcp_session",
    "option_book_window",
    "publish_startup_status",
    "right_for_structure",
    "screen_for",
]

EXIT_SEQUENCE_BASE = 500
"""Exit cycle ids start here.

A session has at most a few dozen slots, so nothing above 500 collides with an
entry — and an exit and an entry in the same slot minting the same `cycle_id`
would collapse two decisions into one journal line (specs/06 D2)."""

DTE_MIN = 3
DTE_MAX = 21
"""The expiry window candidates are drawn from, absent a book — specs/07 D5.

Wider than the Gate's own window on purpose: the chain request should return
everything the Gate might accept, and let the Gate refuse. A request narrower
than the rule would hide candidates from the menu and call it a risk decision.

Once a book is loaded, `option_book_window` below replaces this with the
book's own window intersected against the Gate's — the book is the rule the
research validated, and this pair is only the fallback for a context built
with no book at all (offline tests, and any caller that pre-dates one)."""


def right_for_structure(structure: str) -> Right:
    """Which side of the chain a rule's structure trades — specs/07 D5.

    Only the two verticals `agent/candidates.py` actually builds.
    `option_book.py`'s loader also admits `iron_condor` — it is defined risk on
    its own terms, and refusing it there would be refusing a shape this
    executor might one day build rather than one it cannot represent. But
    nothing here builds one: `spreads_by_delta` resolves a single vertical, not
    two. A book naming `iron_condor` is therefore refused here, at the one
    place that would otherwise silently trade half of it and call the other
    half a quiet market.
    """
    if structure == "put_credit_spread":
        return Right.PUT
    if structure == "call_credit_spread":
        return Right.CALL
    raise InvariantViolation(
        f"structure {structure!r} has no vertical-spread executor in this build "
        "(agent/candidates.py builds put_credit_spread and call_credit_spread "
        "only); refusing rather than resolving half of it"
    )


def option_book_window(rule: OptionRule, limits: RiskLimits) -> tuple[int, int]:
    """The chain request's DTE window: the book's own, intersected with the
    Gate's `dte_range`.

    Requesting the book's window unmodified is a live bug waiting to happen.
    This book's tolerance reaches 24 DTE (14 target, ±10 — `rule.dte_window()`)
    and `SLEEVE_LIMITS.dte_range` stops at 21 (specs/03 D5): a 24-DTE candidate
    would price, size and rank onto the menu only to be vetoed by the Gate
    every single cycle — exactly the "menu whose ranking fights the Gate"
    failure `agent/candidates.py`'s own module docstring describes, just
    arriving from the expiry axis instead of the delta one. The book is not
    wrong to want 24; the account's own risk limit is simply narrower than the
    researched window on this one edge, and the chain request should ask for
    what the Gate can actually approve rather than for what would be vetoed.

    This is the mirror image of `DTE_MIN`/`DTE_MAX`, which asks *wider* than
    the Gate on purpose so the Gate does the refusing. Here the *book* is the
    wider side, so the request narrows to match — the Gate's own range is
    never widened to accommodate the book (CLAUDE.md rule: do not widen the
    Gate).
    """
    book_low, book_high = rule.dte_window()
    gate_low, gate_high = limits.dte_range
    return max(book_low, gate_low), min(book_high, gate_high)


def load_pinned_option_book(
    directory: Path, pinned_fingerprint: str
) -> tuple[OptionBook | None, str]:
    """The pinned option rule, or the reason there is none — specs/07 D1.

    Mirrors `equity.find_latest_book` plus `equity.read_book`, adapted to
    return a reason instead of raising: `preflight`, `once` and `run` each
    report a missing or unusable book in their own voice, and there is
    deliberately no hand-written fallback rule on any of these paths — a book
    that cannot be loaded means the agent places no orders this session, said
    plainly, never a quieter strategy standing in for the one the research
    validated.

    Digested over the file's own **bytes**, not a re-serialised mapping, for
    the reason `equity.py` gives: the pin is a claim about one specific file,
    and hashing a copy re-encoded by this process's JSON settings would hash
    this process rather than the artefact `aqr option-book` wrote.
    """
    path = find_latest_book(directory, pinned_fingerprint)
    if path is None:
        advice = _option_pin_advice(unpinned_books(directory, pinned_fingerprint))
        return None, (
            f"no option book for {pinned_fingerprint} in {directory}; "
            + (advice or "run `aqr option-book` first")
        )
    try:
        raw_bytes = path.read_bytes()
        payload = json.loads(raw_bytes.decode("utf-8"))
        if not isinstance(payload, dict):
            raise UnusableOptionBook(
                f"{path.name} is a {type(payload).__name__}, expected an object"
            )
        book = load_option_book(
            payload,
            pinned_fingerprint=pinned_fingerprint,
            digest=sha256(raw_bytes).hexdigest(),
        )
    except (UnusableOptionBook, OSError, UnicodeDecodeError, ValueError) as failure:
        return None, f"{path.name}: {failure}"
    return book, ""


def _option_pin_advice(found: Sequence[UnpinnedBook]) -> str:
    """The option-sleeve twin of `equity.changed_pin_advice`.

    Not reused verbatim: that function's own wording names
    `ALPHAGATE_STRATEGY_FINGERPRINT` and specs/09 D1, and printing the wrong
    environment variable at the moment an operator is trying to fix a stale pin
    is worse than printing nothing.
    """
    if not found:
        return ""
    return "\n".join(
        [
            "No book for the pinned option rule, but these are on disk:",
            *(
                f"  {book.name or '(unnamed)'} [{book.fingerprint}] "
                f"as of {book.as_of or '?'}  — {book.path.name}"
                for book in found
            ),
            "If this account should be executing one of them, change "
            "ALPHAGATE_OPTION_FINGERPRINT in .env.local to that fingerprint "
            "and restart. Until then the pin stands and these are ignored — "
            "specs/07 D1.",
        ]
    )


def screen_for(context: LiveContext) -> Screen:
    """The screen this session runs — specs/07 D1.

    `BookScreen` once a rule is loaded. `DefaultScreen` is only reachable from
    a `LiveContext` built with no book at all — `cli.py`'s composition root
    never produces one of those for `once` or `run` (a missing book exits
    before a session forms, see `_context` there) — but offline tests of this
    module, and any future caller that has not adopted a book yet, still build
    one routinely, and `DefaultScreen`'s fail-closed default is the right thing
    to fall back to rather than a screen that trades with no rule at all.
    """
    if context.option_book is not None:
        return BookScreen(context.option_book.rule)
    return DefaultScreen()


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


ACCOUNT_BASIS: Final = "account"
"""The pre-sleeve basis. A state file with no `basis` key was written under it,
which is why it is the default rather than an error."""

OPTIONS_SLEEVE_BASIS: Final = "options-sleeve"

EQUITY_SLEEVE_BASIS: Final = "equity-sleeve/marked-on-the-sleeve"
"""Bumped from the bare `equity-sleeve`, which is a label the files on disk
still carry while holding a number that is not one.

The rebalance pass marked the sleeve, as the label says. The thirty-second
heartbeat re-marked the same field with *account* equity, and it runs two
hundred times as often, so every state file written before that was fixed holds
an account-scale peak under a sleeve-scale name — read back, a standing ~10%
drawdown that never happened, which on 2026-09-03 latched the kill switch and
refused the day's rebalance.

Changing the string is what makes `SessionState.load` discard those marks
instead of believing them. See its docstring for why they are discarded rather
than rescaled."""


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
    basis: str = ACCOUNT_BASIS
    """What `peak_equity` was measured against — specs/03 D6.

    Persisted, and checked on load. Before sleeves existed every high-water mark
    was a mark on account equity; now each strategy marks its own sleeve, and
    the two are not comparable. A file written under one basis and read under
    another hands the kill switch a threshold measured on a different quantity:
    on this account an untouched $95,000 sleeve, read against a $100,175 account
    peak, reports a 5% drawdown that never happened."""

    discarded_peak: Decimal | None = None
    """A high-water mark dropped by a basis change, kept for one message.

    Not persisted. The CLI prints it once, because a kill switch that quietly
    forgot its history is exactly the failure worth being loud about."""

    @classmethod
    def load(cls, path: Path, *, basis: str = ACCOUNT_BASIS) -> SessionState:
        """Read the state, discarding a peak measured against something else.

        **Discarded, not rescaled.** Rescaling an account peak into a sleeve
        peak means subtracting what the other sleeve was worth at that moment,
        and nothing in this file records it. It happens to equal the allocation
        today, because the options sleeve has not traded — and a migration that
        is correct only while some fact remains true is one that will be wrong
        silently later.

        Dropping the mark can only understate a drawdown, and only until the
        sleeve makes a new high. Rescaling it wrongly could overstate one and
        latch a kill switch on arithmetic nobody would think to check.
        """
        if not path.is_file():
            return cls(path=path, basis=basis)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return cls(path=path, basis=basis)
        peak = raw.get("peak_equity")
        stored = Decimal(str(peak)) if peak is not None else None
        latched = bool(raw.get("killswitch_tripped"))
        if str(raw.get("basis", ACCOUNT_BASIS)) != basis:
            return cls(
                path=path,
                peak_equity=None,
                killswitch_tripped=latched,
                basis=basis,
                discarded_peak=stored,
            )
        return cls(path=path, peak_equity=stored, killswitch_tripped=latched, basis=basis)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                {
                    "peak_equity": str(self.peak_equity) if self.peak_equity else None,
                    "killswitch_tripped": self.killswitch_tripped,
                    "basis": self.basis,
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
    sleeve_allocation: Decimal = OPTIONS_SLEEVE_ALLOCATION
    """The capital this agent is allowed to commit — specs/03 D6.

    The base every budget in `limits` is a fraction of. Held here rather than
    read from the account, so that what the options agent may risk does not move
    because the equity book had a good morning."""
    limits: RiskLimits = SLEEVE_LIMITS
    """`SLEEVE_LIMITS`, not `DEFAULT_LIMITS`: the fractions are fractions of
    `sleeve_allocation`. Pairing the account-scaled limits with a sleeve base
    would apply the 5% twice and size every candidate to zero."""
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
    option would put it somewhere nobody reads. Once a book is loaded, `cli.py`
    sets this from `right_for_structure(option_book.rule.structure)` rather
    than leaving the two free to disagree about which side of the chain is
    being traded."""

    option_book: OptionBook | None = None
    """The validated rule this session executes — specs/07 D1.

    Named `option_book` rather than `book` because `LiveContext.book()` above
    is already a method — the broker/journal read the rest of this module
    calls "the book" in a different sense (positions and P&L, not a rule), and
    reusing the word for both would make every reference to either ambiguous
    at the call site.

    `None` only for a context built with no book at all: the offline tests of
    this module, and nothing `cli.py`'s composition root produces for `once` or
    `run` — a book that fails to load exits before a `LiveContext` exists
    (specs/07 D1's "no book, no orders"). `screen_for` and `gather_for` both
    read this to decide whether to run the researched rule or fall back to the
    pre-book defaults."""

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
        """Re-read the account and the sleeve from the broker. Never inferred.

        Three orderings matter here and each was a bug waiting to happen.

        **The whole journal, not today's file.** `read_through` because a spread
        opened on Monday and still held on Wednesday has no fill in Wednesday's
        file: reading one day would report its legs as unexplained and drop the
        position out of the Gate's risk model. Realised P&L is cumulative for
        the same reason.

        **Bounded at `as_of`**, so a replay of an earlier day cannot see a later
        one.

        **The high-water mark is observed after the read, and on the sleeve.**
        Before, it recorded account equity — which made the options kill switch
        a function of what the equity book did overnight (specs/03 D6). It is
        raised after the book is built rather than before because the drawdown
        must be measured against the peak as it stood, and a new high raises the
        mark for the *next* cycle. At a new high the two orderings agree, since
        `Sleeve.drawdown` already returns zero above the peak; at a new low they
        do not, and observing first would quietly rebase the loss to zero.
        """
        if self.mcp is None:
            raise RuntimeError("no MCP session: cannot read the book")
        account = read_account(self.mcp, observed_at=as_of)
        self.last_account = account
        legs = read_positions(self.mcp)
        book = read_book(
            account,
            legs,
            self.journal.read_through(as_of.date()),
            sleeve_allocation=self.sleeve_allocation,
            peak_equity=self.state.peak_equity,
            fills_today=fills_today,
            killswitch_tripped=self.state.killswitch_tripped,
        )
        self.state.observe(book.snapshot.equity)
        self.last_book = book
        return book


def expiry_window(
    as_of: datetime, *, dte_window: tuple[int, int] = (DTE_MIN, DTE_MAX)
) -> tuple[date, date]:
    """The chain request's date range, `dte_window` days out from `as_of`.

    Defaults to the pre-book fallback so every existing caller and test is
    unchanged; `gather_for` passes `option_book_window(...)` once a book is
    loaded, which is a *narrower* pair than the default on the high end (the
    Gate's `dte_range` tops out at 21 where a book may reach further).
    """
    today = as_of.date()
    low, high = dte_window
    return today + timedelta(days=low), today + timedelta(days=high)


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
        window = (
            option_book_window(context.option_book.rule, context.limits)
            if context.option_book is not None
            else (DTE_MIN, DTE_MAX)
        )
        low, high = expiry_window(slot.at, dte_window=window)

        chain = context.data.option_chain(
            symbol,
            expiry_from=low,
            expiry_to=high,
            right=_api_right(context.right),
        )
        latest_iv = context.iv.latest(symbol)
        perception = perceive(
            context.data,
            symbol,
            as_of=slot.at,
            chain=chain,
            history=context.iv.observations(symbol),
            history_as_of=latest_iv[0] if latest_iv is not None else None,
        )
        upcoming = next_slot(schedule, slot.at) if schedule else None
        context.next_slot_at = upcoming.at if upcoming else None
        book = context.book(as_of=slot.at, fills_today=counter[0])
        account = context.last_account
        if account is None:  # pragma: no cover - book() always sets it
            raise RuntimeError('book() did not record an account read')

        if context.option_book is not None:
            rule = context.option_book.rule
            structures = spreads_by_delta(
                chain,
                right=context.right,
                anchor_delta=rule.anchor_delta,
                anchor_tolerance=rule.anchor_tolerance,
                width_delta=rule.width_delta,
                as_of=slot.at,
            )
        else:
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

    spots = _spots_for(context.data, book.held)
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
            read=_position_read(item, as_of, spot=spots.get(item.position.underlying)),
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


def _spots_for(data: MarketData, held: Sequence[HeldPosition]) -> dict[Ticker, Decimal]:
    """The underlying's price for each held position. Best effort, never raises.

    One quote per distinct underlying, so an exit line can journal what the
    market was actually doing rather than a stand-in. A symbol the port cannot
    answer for is simply absent: `_position_read` falls back, and a quote
    failure must not be able to stop an exit from being evaluated — the
    position's own option marks are what the policy reads, and they came from a
    different call.
    """
    spots: dict[Ticker, Decimal] = {}
    for symbol in sorted({item.position.underlying for item in held}, key=str):
        try:
            spots[symbol] = data.latest_price(symbol)
        except Exception as failure:
            # Not logged and not raised. The caller renders the fallback, and
            # the one thing this must not do is prevent the exit below it from
            # being evaluated over a field that decides nothing.
            del failure
    return spots


def _position_read(
    item: HeldPosition, as_of: datetime, *, spot: Decimal | None = None
) -> MarketRead:
    """The minimum honest `MarketRead` for an exit line.

    An exit is not a perception-driven decision — it reads the position's own
    marks, not the trend engine — so inventing a full read here would put
    numbers in the journal that no engine produced. Every optional field stays
    `None` for that reason.

    `spot` is the one field a read may never omit. It is the underlying's real
    price when `_spots_for` could get one, and the position's **short strike**
    otherwise — the strike the position is defined around, and the closest thing
    to a price this path knows on its own. The fallback used to be `legs[0]`,
    which on a put credit spread is the long wing: journalled and rendered as
    "spot", it read as a market forty points from where SPY actually was.
    """
    return MarketRead(
        underlying=item.position.underlying,
        as_of=as_of,
        spot=spot if spot is not None else _short_strike(item.position.structure),
    )


def _short_strike(structure: OptionStructure) -> Decimal:
    """The strike of the first short leg — every kind in specs/02 D3 has one."""
    for leg in structure.legs:
        if leg.side is Side.SELL:
            return leg.contract.strike
    return structure.legs[0].contract.strike  # pragma: no cover - no such kind


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


def publish_startup_status(
    context: LiveContext, *, as_of: datetime, slots: Sequence[Slot]
) -> None:
    """A status snapshot the instant a session starts, before any slot runs.

    `_publish_status` above is only ever called from inside `gather_for`'s
    per-slot closure, so a session that only just started publishes nothing
    until its first slot fires -- up to a full `CYCLE_INTERVAL` (15 minutes)
    later. `interface/status.py`'s staleness rule reads whatever snapshot was
    already on disk in the meantime, so a freshly started, perfectly healthy
    `run` reports **not running** on the dashboard for that whole gap. Measured
    live: 30 minutes of "not running" while the process was alive throughout,
    and the moment an operator who just typed the command is watching most
    closely.

    **A full snapshot, not a bare "alive" marker.** Building one means reading
    the account and the sleeve the same way a slot would (`context.book`), and
    that read is most of the value here: a bad credential, a blocked account or
    a dead MCP subprocess shows up now, at startup, instead of silently inside
    the first slot's own `except Exception: return` fifteen minutes later.

    **Decides and journals nothing** (specs/06). No screen, no proposer, no
    Gate runs, and no `Slot` from `slots` is consumed or advanced past --
    `slot_sequence` is written as `-1` because no cycle has run to have one,
    and there is nothing here for a journal line to record. `_stages_today`
    still runs, so the page's today-so-far counts are right from the first
    paint rather than empty until the first slot writes them.

    Never raises to the caller -- `supervised_run` restarts a session that
    threw, and a startup publish that could itself kill the session it is
    trying to make visible would be a strange kind of fix. But a swallowed
    failure here reproduces exactly the bug this function exists to close --
    the page silently not appearing -- so unlike `_publish_status`, a failure
    is printed to the operator's console rather than passed over in silence.
    """
    try:
        book = context.book(as_of=as_of)
        account = context.last_account
        if account is None:  # pragma: no cover - book() always sets it
            raise RuntimeError("book() did not record an account read")
        upcoming = next_slot(tuple(slots), as_of)
        context.next_slot_at = upcoming.at if upcoming else None
        snapshot = build_status(
            account=account,
            book=book,
            limits=context.limits,
            policy=context.exit_policy,
            marks={},
            as_of=as_of,
            next_slot=context.next_slot_at,
            slot_sequence=-1,
            universe=tuple(str(u.symbol) for u in context.universe),
            peak_equity=context.state.peak_equity,
            stage_counts=_stages_today(context, as_of),
            note="session starting -- no cycle has run yet",
        )
        write_status(snapshot, directory=context.journal.directory)
    except Exception as exc:
        print(f"! could not publish a startup status: {type(exc).__name__}: {exc}")


def _stages_today(context: LiveContext, as_of: datetime) -> dict[str, int]:
    """Today's stage tally for the options page.

    Counted through `interface.read.stage_tally`, which is what shapes the same
    field for the journal page — one implementation of "count the stages,
    options only". Both agents write to one daily file, and a tally that added
    an equity pass's `submitted` to this agent's would be a number about
    neither sleeve, printed on this agent's own status card.
    """
    try:
        records = context.journal.read(as_of.date())
    except OSError:
        return {}
    return stage_tally(records)


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
