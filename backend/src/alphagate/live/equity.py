"""The equity composition root — specs/09 D8 and D9.

Where the tested parts become a running rebalancer. Everything below the
`equity`, `risk` and `execution` line has been unit-tested offline; this module
is the first place any of it meets a real account, and it is deliberately the
only one.

The pass, in order, and every step is a separate testable function:

1. **Find the book.** The newest artefact `aqr target-book` wrote for the pinned
   fingerprint. Read the *bytes*, hash them, then parse — so the digest is of
   the file and not of this process's JSON formatting.
2. **Copy it.** Into `journal/books/`, verbatim. `aqr` regenerates its output
   directory daily, and a journal line pointing at a path whose contents have
   since changed is a record of nothing (specs/09 D9).
3. **Read the account and the book from the broker.** Never inferred, every
   pass, the same discipline the options runner keeps.
4. **Mark every symbol.** One batched snapshot request for the union of what is
   held and what is wanted.
5. **Plan.** Pure, deterministic, sells before buys.
6. **Gate each intent, then submit it.** One at a time, and the portfolio
   snapshot is *advanced* after each fill so the daily caps see the orders this
   pass has already placed. A pass that budgeted every order against the
   account as it stood at the top would let a hundred orders each pass a check
   the hundred of them together fail.
7. **Journal.** One record, whatever happened, including "nothing".

**A pass is idempotent by day.** The client order id derives from the book's
session and the trading day, so a restart at 14:00 that re-plans the same book
produces the same keys, and Alpaca refuses the duplicates. The journal is
checked first anyway, so the normal path does not lean on the broker for that.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from hashlib import sha256
from pathlib import Path
from typing import Any, Final

from alphagate.core.identifiers import Ticker
from alphagate.equity import (
    DEFAULT_EQUITY_POLICY,
    EquityPolicy,
    EquitySide,
    Holding,
    Mark,
    OrderIntent,
    RebalancePlan,
    Skipped,
    TargetBook,
    UnusableBook,
    load_target_book,
    plan_rebalance,
)
from alphagate.execution import (
    AccountRead,
    ExecutionError,
    McpSession,
    Submission,
    Tradeability,
    read_account,
    read_share_positions,
    read_tradeability,
    submit_equity,
)
from alphagate.journal import Journal
from alphagate.marketdata import MarketData, StockSnapshot
from alphagate.risk import (
    ApprovedEquity,
    EquityPortfolio,
    EquityVerdict,
    VetoedEquity,
    evaluate_equity,
)

__all__ = [
    "BOOK_ARCHIVE",
    "EquityContext",
    "EquityCycleRecord",
    "EquityOrderRecord",
    "EquityStage",
    "UnpinnedBook",
    "already_decided",
    "archive_book",
    "changed_pin_advice",
    "cycle_id_for",
    "digest_of",
    "find_latest_book",
    "marks_from",
    "read_book",
    "run_equity_cycle",
    "today_totals",
    "unpinned_books",
]

BOOK_ARCHIVE: Final = "books"
"""Subdirectory of the journal where executed books are kept, verbatim."""

CYCLE_KIND: Final = "equity"
"""Distinguishes these records from the options cycles in the same daily file.

One journal, two agents. A separate file per agent would have meant two answers
to "what happened on 2026-08-31", and the question is about the account rather
than about which process asked."""


class EquityStage(Enum):
    """How far a rebalance pass got. Every pass records one.

    `NO_TRADES` is the majority and it is the point: the strategy rebalances
    every five sessions, so four days in five the honest answer is "the book is
    already held", and a journal that only contained trades could not say so.
    """

    NO_BOOK = "no_book"
    """No usable artefact. Either none was generated, or the one on disk was
    refused — a different strategy, an unspent seal, a refuted rule."""
    STALE_BOOK = "stale_book"
    NO_MARKS = "no_marks"
    """The pass could not read a single price it needed, so it decided nothing.

    Non-terminal, and that is its whole reason to exist. It looks like
    `NO_TRADES` — an empty plan, every symbol accounted for — but the account of
    each one is "no usable quote" rather than "already held", and the remedy is
    to look again in thirty seconds rather than to call the day done. See
    `RebalancePlan.is_blind`."""
    NO_TRADES = "no_trades"
    PLANNED = "planned"
    """Gated and deliberately not sent. What `equity-plan` produces, and a
    first-class outcome rather than a flag threaded through the trading path."""
    SUBMITTED = "submitted"
    VETOED = "vetoed"
    """Every intent was refused. Distinct from `NO_TRADES`, which means there was
    nothing to refuse."""
    HALTED = "halted"
    """Stopped part-way — a transport failure or an unresolvable timeout. The
    orders already placed stand and are journalled; the rest were not attempted,
    and the next pass will re-derive them from what is actually held."""

    @property
    def traded(self) -> bool:
        return self in {EquityStage.SUBMITTED, EquityStage.HALTED}

    @property
    def decided(self) -> bool:
        """Whether this stage settles the book it was reached on.

        See `DECIDED_STAGES`.
        """
        return self.value in DECIDED_STAGES


DECIDED_STAGES: Final = frozenset(
    {"submitted", "no_trades", "planned", "vetoed", "halted", "stale_book"}
)
"""The stages that settle a book. A session will not run a second pass on a book
that already reached one of these today.

An allowlist rather than a denylist, so a stage nobody has classified is quiet
rather than terminal. Two are deliberately absent: `no_book`, where the artefact
was missing or unreadable, and `no_marks`, where no price could be read for any
symbol in it. Neither is an opinion about the book, and both are fixed by
looking again a little later."""


def cycle_id_for(as_of: datetime, sequence: int) -> str:
    """`YYYY-MM-DD-EQ-NNN` — the same shape as specs/06 D2's option cycle ids.

    `EQ` where an option cycle carries its underlying, because a rebalance is
    about the whole book rather than one name. Collision-free against the option
    ids for the same reason it is readable: no ticker is `EQ`.
    """
    return f"{as_of.date().isoformat()}-EQ-{sequence:03d}"


# --------------------------------------------------------------------------- #
# The book
# --------------------------------------------------------------------------- #


def find_latest_book(directory: Path, fingerprint: str) -> Path | None:
    """The newest artefact for one strategy, by the session it describes.

    Sorted by **filename**, not by mtime. `aqr target-book` names its output
    `<name>-<fingerprint>-<as_of>.json`, so the lexicographic maximum is the
    latest session — whereas mtime would prefer whichever file was regenerated
    most recently, which on a re-run of an old date is the wrong one.
    """
    if not directory.is_dir():
        return None
    candidates = sorted(directory.glob(f"*-{fingerprint}-*.json"))
    return candidates[-1] if candidates else None


@dataclass(frozen=True, slots=True)
class UnpinnedBook:
    """A book on disk that the pin does not admit. A report, never an instruction.

    Deliberately *not* a `TargetBook`: nothing here has been validated and
    nothing here may be executed. It carries the three fields an operator needs
    to decide whether the pin is stale, and no weights at all — so there is no
    version of this type that could be handed to a planner by accident.
    """

    path: Path
    name: str
    fingerprint: str
    as_of: str
    """The session as the file spells it, unparsed. A book we will not execute
    does not get to fail on a malformed date."""


def unpinned_books(directory: Path, pinned_fingerprint: str) -> tuple[UnpinnedBook, ...]:
    """Books in the directory belonging to some strategy other than the pinned one.

    This exists because the pin is invisible from the inside.
    `find_latest_book` globs on the pinned fingerprint, so a book for a strategy
    the pin does not name does not merely get refused — it is never looked at,
    and the pass reports "no target book" while a perfectly good one sits in the
    same directory. Refusing it is correct (specs/09 D1). Being silent about it
    is what sends somebody to read source at 09:20.

    Keyed on the file's **contents**, not its name, so a file named for the
    pinned strategy whose `spec_fingerprint` says otherwise is reported here —
    which is exactly the case `load_target_book` refuses, now with an
    explanation attached.

    Newest per strategy: `aqr` writes one book per session, and a month of them
    is a wall of text in which the only candidate is the latest. Sorted newest
    first with the fingerprint breaking ties, so the same directory gives the
    same answer in the same order (CLAUDE.md rule 7).

    Never raises. It runs on the failure path — a directory with a half-written
    file in it is a directory we still report on.
    """
    if not directory.is_dir():
        return ()
    newest: dict[str, UnpinnedBook] = {}
    for path in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(path.read_bytes().decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        fingerprint = str(payload.get("spec_fingerprint", ""))
        if not fingerprint or fingerprint == pinned_fingerprint:
            continue
        found = UnpinnedBook(
            path=path,
            name=str(payload.get("spec_name", "")),
            fingerprint=fingerprint,
            as_of=str(payload.get("as_of", "")),
        )
        seen = newest.get(fingerprint)
        if seen is None or found.as_of > seen.as_of:
            newest[fingerprint] = found
    return tuple(
        sorted(newest.values(), key=lambda b: (_descending(b.as_of), b.fingerprint))
    )


def _descending(as_of: str) -> tuple[int, ...]:
    """Sort key that puts the latest session first without reversing the tie-break."""
    return tuple(-ord(character) for character in as_of)


def changed_pin_advice(found: Sequence[UnpinnedBook]) -> str:
    """What to tell the operator when the pin matched nothing on disk.

    One question, asked at one moment: the pinned fingerprint found no book, so
    is the directory empty, or is there a book here for a strategy the pin does
    not name? Those are completely different problems and `find_latest_book`
    reports them with the same silence — it globs on the fingerprint, so a book
    for another strategy is not refused, it is never looked at.

    Called only on that path. When the pin finds its book there is nothing to
    say and nothing is said, even if the directory holds five other strategies —
    that is what a research repository looks like on an ordinary day, and a
    warning printed on a healthy morning teaches its reader to skim past the one
    morning it is real.

    It reports and stops. Nothing here changes the pin and nothing offers to:
    the pin is the only checkable form of "this account executes the one
    strategy the research validated" (specs/09 D1), and a system that changed it
    on its own would have deleted the property it was reporting on.

    Empty when there is nothing on disk either, so callers can print it
    unconditionally and fall back to "run `aqr target-book`".
    """
    if not found:
        return ""
    return "\n".join(
        [
            "No book for the pinned strategy, but these are on disk:",
            *(
                f"  {book.name or '(unnamed)'} [{book.fingerprint}] "
                f"as of {book.as_of or '?'}  — {book.path.name}"
                for book in found
            ),
            "If this account should be executing one of them, change "
            "ALPHAGATE_STRATEGY_FINGERPRINT in .env.local to that fingerprint "
            "and restart. Until then the pin stands and these are ignored — "
            "specs/09 D1.",
        ]
    )


def digest_of(path: Path) -> str | None:
    """The digest of a book on disk, without parsing or validating it.

    The session guard runs every heartbeat and only needs to answer "is this the
    same file I already acted on", so it must not pay for a full load — and must
    not refuse a book for being unusable, which is the pass's job to journal.

    `None` when the file is not readable. `aqr target-book` rewrites its output
    in place, so a read landing inside that window is a *not yet* rather than a
    failure: the next heartbeat is thirty seconds away.

    Hashes the same bytes `read_book` does, and the tests hold the two together.
    A guard that hashed a normalised copy would never match the journal and
    would therefore start a pass on every beat.
    """
    try:
        return sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def read_book(path: Path, *, pinned_fingerprint: str) -> tuple[TargetBook, str]:
    """Load one artefact and return it beside the bytes it came from.

    The raw text is returned rather than discarded because two things need it:
    the digest, which must be of the file, and the archive copy, which must be
    byte-identical to what was executed. Re-serialising the parsed mapping would
    give a digest of this process's JSON settings and an archive that differs
    from the original in whitespace.

    Read as **bytes** and decoded here, rather than through `read_text`. On
    Windows `read_text` translates CRLF to LF, so the digest would be of a
    normalised copy rather than of the file — and `archive_book` writing that
    copy back through `write_text` would translate it again, producing an
    archive whose hash matches neither the source nor the record. Byte-exact in
    and byte-exact out is the only version of this that means anything.
    """
    raw_bytes = path.read_bytes()
    digest = sha256(raw_bytes).hexdigest()
    raw = raw_bytes.decode("utf-8")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise UnusableBook(f"{path.name} is a {type(payload).__name__}, expected an object")
    book = load_target_book(payload, pinned_fingerprint=pinned_fingerprint, digest=digest)
    return book, raw


def archive_book(raw: str, book: TargetBook, *, directory: Path) -> Path:
    """Copy the executed book into the journal. Verbatim, never re-serialised.

    Named by fingerprint, session **and digest**. The first two alone were a
    hole: `aqr` names its output by session, so a book regenerated for the same
    `as_of` with different weights landed on the same archive path and replaced
    the evidence of the pass before it. That was survivable while a session held
    one pass. It is not survivable now that a new book may be acted on the same
    day — the journal line would name a digest no file on disk had.

    Writing the digest into the name also keeps this idempotent for the reason
    it always was: the same bytes archived twice is one file, because the same
    bytes cannot have two digests.
    """
    archive = directory / BOOK_ARCHIVE
    archive.mkdir(parents=True, exist_ok=True)
    path = (
        archive
        / f"{book.fingerprint}-{book.as_of.isoformat()}-{book.digest[:12]}.json"
    )
    path.write_bytes(raw.encode("utf-8"))
    return path


# --------------------------------------------------------------------------- #
# Marks
# --------------------------------------------------------------------------- #


def marks_from(
    snapshots: Mapping[Ticker, StockSnapshot],
    tradeability: Mapping[Ticker, Tradeability],
    *,
    as_of: datetime,
) -> dict[Ticker, Mark]:
    """Join prices to tradeability, and age each price against `as_of`.

    The age is computed here, once, from an argument — not inside the planner
    and not from a clock read. That is what makes the freshness check downstream
    a comparison between two values the caller supplied, and it is why a replay
    of this pass produces the same skips as the live one did.

    A symbol with a price but no asset record is **not tradeable**. That is the
    honest reading of "we could not find out", and defaulting it the other way
    would send an order to find out — which is a strange way to ask.
    """
    marks: dict[Ticker, Mark] = {}
    for symbol in sorted(snapshots, key=str):
        snapshot = snapshots[symbol]
        asset = tradeability.get(symbol)
        marks[symbol] = Mark(
            symbol=symbol,
            price=snapshot.price,
            age_seconds=max(0.0, (as_of - snapshot.as_of).total_seconds()),
            tradeable=asset is not None and asset.tradeable,
            fractionable=asset is not None and asset.fractionable,
        )
    return marks


# --------------------------------------------------------------------------- #
# The record
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class EquityOrderRecord:
    """One intent, its verdict, and what became of it.

    The verdict is carried whole — the full check tape, not just the answer — so
    the dashboard can show the near-misses (specs/09 D10). A book sitting at 14%
    of a 15% concentration cap is a different situation from one at 2%, and only
    one of them is a reason to look.
    """

    symbol: str
    side: str
    shares: Decimal
    reference_price: Decimal
    notional: Decimal
    target_weight: Decimal
    held_weight: Decimal
    verdict: EquityVerdict
    submission: Submission | None
    outcome: str

    @property
    def approved(self) -> bool:
        return self.verdict.is_approved


@dataclass(frozen=True, slots=True)
class EquityCycleRecord:
    """Everything one rebalance pass did, whether or not it traded.

    Written once, at the end. Later facts — a fill minutes after submission —
    arrive as separate amendment lines keyed by `cycle_id`, never as an edit to
    this one (specs/06 D3).
    """

    cycle_id: str
    kind: str
    as_of: datetime
    stage: EquityStage
    strategy: Mapping[str, Any]
    """Fingerprint, name, session, digest, and the sealed-run numbers. Copied
    into every record rather than referenced, because a journal line has to be
    readable without the database that produced the book."""
    equity: Decimal
    band_pct: Decimal
    """The proportional no-trade band this pass used. The currency threshold
    differs per position and travels on each skipped line."""
    orders: tuple[EquityOrderRecord, ...]
    skipped: tuple[Mapping[str, Any], ...]
    turnover: Decimal
    note: str = ""

    @property
    def traded(self) -> bool:
        return self.stage.traded

    @property
    def submitted(self) -> int:
        return sum(1 for order in self.orders if order.submission is not None)


def strategy_view(book: TargetBook) -> dict[str, Any]:
    """The provenance a record carries, and the dashboard renders — specs/09 D10.

    Every number here is *recorded*, never acted on. The sealed run decided
    whether this strategy may run at all, and that decision was made before this
    process started; nothing downstream sizes a position by an alpha or a `t`.

    `looks` and `can_confirm` are included because leaving them out is how a
    +2.22 becomes "significant" in somebody's summary. The bar it had to clear
    depends on how many candidates the sealed window has screened, and the window
    can refute but never confirm.
    """
    sealed = book.sealed
    return {
        "fingerprint": book.fingerprint,
        "name": book.name,
        "as_of": book.as_of.isoformat(),
        "digest": book.digest,
        "status": book.status,
        "universe": book.universe,
        "dataset_version": book.dataset_version,
        "hypothesis": book.hypothesis,
        "selection_rule": book.selection_rule,
        "distinct_hypotheses": book.distinct_hypotheses,
        "positions": len(book.weights),
        "gross": book.gross,
        "sealed": {
            "return": sealed.strategy_return,
            "sharpe": sealed.strategy_sharpe,
            "benchmark_sharpe": sealed.benchmark_sharpe,
            "max_drawdown": sealed.max_drawdown,
            "trades": sealed.trades,
            "observations": sealed.observations,
            "alpha": sealed.alpha,
            "beta": sealed.beta,
            "t_alpha": sealed.t_alpha,
            "information_ratio": sealed.information_ratio,
            "is_significant": sealed.is_significant,
            "refuted": sealed.refuted,
            "can_confirm": sealed.can_confirm,
            "looks": sealed.looks,
            "window": f"{sealed.first_session[:10]} to {sealed.last_session[:10]}",
        },
    }


def _skip_view(skipped: Skipped) -> dict[str, Any]:
    return {
        "symbol": str(skipped.symbol),
        "reason": skipped.reason.value,
        "detail": skipped.detail,
        "drift": skipped.drift_notional,
        "threshold": skipped.threshold,
    }


# --------------------------------------------------------------------------- #
# Today, so far
# --------------------------------------------------------------------------- #


def today_totals(journal: Journal, day: date) -> tuple[int, Decimal]:
    """Orders placed and notional traded on the equity path today.

    Read off the journal rather than kept in memory, so a restart at 14:00
    resumes with the caps it left at rather than with a fresh budget. The Gate
    cannot read a file — that is the point of it — so these arrive as fields on
    the portfolio snapshot (specs/09 D5).
    """
    orders = 0
    turnover = Decimal(0)
    for record in journal.read(day):
        if record.get("kind") != CYCLE_KIND:
            continue
        for order in record.get("orders", ()):
            if not isinstance(order, Mapping) or order.get("submission") is None:
                continue
            orders += 1
            turnover += _money(order.get("notional"))
    return orders, turnover


def already_decided(journal: Journal, day: date, digest: str) -> bool:
    """Whether this exact book has already been passed on today.

    Keyed on the *digest*, not on the fingerprint and not on the session: a book
    regenerated mid-morning with new weights is a different instruction and
    should be acted on, while the same bytes seen twice is a restart.

    This is the whole of the idempotence rule. A day-keyed guard was simpler and
    wrong in a way that matters to this system: research is the point, and a
    strategy that improves at eleven o'clock should not have to wait for
    tomorrow's open to be held. What must not happen is the *same* instruction
    being executed twice, and the digest is exactly that instruction's identity.

    Bounded by the Gate rather than by this function: `max_daily_orders` and
    `max_daily_turnover_pct` count across the whole session, so a second pass
    spends a budget the first one has already drawn on
    ([09](../../../../specs/09-equity-execution.md) D5). That is where a runaway
    regeneration loop stops, and it stops it by cost rather than by clock.
    """
    return any(
        record.get("kind") == CYCLE_KIND
        and str(record.get("strategy", {}).get("digest", "")) == digest
        and str(record.get("stage", "")) in DECIDED_STAGES
        for record in journal.read(day)
        if isinstance(record.get("strategy"), Mapping)
    )


# --------------------------------------------------------------------------- #
# The context
# --------------------------------------------------------------------------- #


@dataclass
class EquityContext:
    """Everything one equity session needs, assembled once.

    `pinned_fingerprint` is the load-bearing field. It comes from configuration
    and never from a book, which is what makes "only the strategy the researcher
    validated" a property rather than a hope (specs/09 D1).
    """

    data: MarketData
    mcp: McpSession | None
    journal: Journal
    books: Path
    """Where `aqr target-book` writes. Read-only from here — this process never
    generates a book, because generating one means running the strategy, and the
    engine that runs it lives in the other project."""
    pinned_fingerprint: str
    policy: EquityPolicy = DEFAULT_EQUITY_POLICY
    peak_equity: Decimal | None = None
    killswitch_tripped: bool = False

    _tradeability: dict[Ticker, Tradeability] = field(default_factory=dict)
    """Cached for the session. `get_asset` takes one symbol, so a 104-name book
    is 104 round trips; the answers change on the order of once a year, and
    re-asking every heartbeat would spend the whole slot on it."""

    last_account: AccountRead | None = None
    last_plan: RebalancePlan | None = None
    last_book: TargetBook | None = None
    last_marks: dict[Ticker, Mark] = field(default_factory=dict)
    last_holdings: tuple[Holding, ...] = ()
    last_note: str = ""
    """The most recent pass's one-line summary, carried onto the status page.

    The heartbeat runs far more often than a pass does, so without this the page
    would show a live account under a blank explanation for most of the day."""

    def observe(self, equity: Decimal) -> None:
        """Raise the high-water mark. Never lowers it — that is the whole job."""
        if self.peak_equity is None or equity > self.peak_equity:
            self.peak_equity = equity

    def drawdown(self, equity: Decimal) -> Decimal:
        if self.peak_equity is None or self.peak_equity <= 0:
            return Decimal(0)
        return max(Decimal(0), (self.peak_equity - equity) / self.peak_equity)

    def tradeability_for(self, symbols: Iterable[Ticker]) -> dict[Ticker, Tradeability]:
        """Ask the broker about names not already known. Cached for the session."""
        if self.mcp is None:
            return dict(self._tradeability)
        missing = [s for s in sorted(set(symbols), key=str) if s not in self._tradeability]
        if missing:
            self._tradeability.update(read_tradeability(self.mcp, missing))
        return dict(self._tradeability)


# --------------------------------------------------------------------------- #
# The pass
# --------------------------------------------------------------------------- #


def run_equity_cycle(
    context: EquityContext,
    *,
    as_of: datetime,
    submit: bool,
    sequence: int = 0,
) -> EquityCycleRecord:
    """One rebalance pass. Returns a record on every path, including failure.

    `submit=False` is dry: the Gate still runs and every verdict is still
    recorded, but nothing is sent. That is what `equity-plan` does, and it is a
    first-class outcome rather than a flag threaded through the trading path —
    so the journal never has to be read as "submitted, but not really".
    """
    cycle_id = cycle_id_for(as_of, sequence)

    path = find_latest_book(context.books, context.pinned_fingerprint)
    if path is None:
        # The directory is not necessarily empty — it may hold a book for a
        # strategy the pin does not name, which globs right past us. Say so, or
        # "no target book" is a true sentence that sends the reader nowhere.
        advice = changed_pin_advice(
            unpinned_books(context.books, context.pinned_fingerprint)
        )
        return _empty(
            cycle_id,
            as_of,
            EquityStage.NO_BOOK,
            note=(
                f"no target book for {context.pinned_fingerprint} in {context.books}; "
                + (advice.replace("\n", " ") if advice else "run `aqr target-book` first")
            ),
        )
    try:
        book, raw = read_book(path, pinned_fingerprint=context.pinned_fingerprint)
    except (UnusableBook, json.JSONDecodeError, OSError) as failure:
        return _empty(
            cycle_id, as_of, EquityStage.NO_BOOK, note=f"{path.name}: {failure}"
        )

    context.last_book = book
    archive_book(raw, book, directory=context.journal.directory)

    age = book.age_days(as_of.date())
    if age > context.policy.max_book_age_days:
        return _empty(
            cycle_id,
            as_of,
            EquityStage.STALE_BOOK,
            strategy=strategy_view(book),
            note=(
                f"book is {age}d old (as of {book.as_of.isoformat()}), limit "
                f"{context.policy.max_book_age_days}d; refresh it before trading"
            ),
        )

    if context.mcp is None:
        return _empty(
            cycle_id,
            as_of,
            EquityStage.NO_BOOK,
            strategy=strategy_view(book),
            note="no broker session; cannot read the account the book is a fraction of",
        )

    account = read_account(context.mcp, observed_at=as_of)
    context.last_account = account
    context.observe(account.equity)
    holdings = read_share_positions(context.mcp)
    context.last_holdings = holdings

    symbols = sorted(set(book.weights) | {h.symbol for h in holdings}, key=str)
    snapshots = context.data.stock_snapshots(symbols)
    marks = marks_from(snapshots, context.tradeability_for(symbols), as_of=as_of)
    context.last_marks = marks

    plan = plan_rebalance(
        book,
        holdings=holdings,
        marks=marks,
        equity=account.equity,
        policy=context.policy,
        as_of=as_of,
    )
    context.last_plan = plan

    orders_today, turnover_today = today_totals(context.journal, as_of.date())
    portfolio = EquityPortfolio(
        equity=account.equity,
        cash=account.cash,
        buying_power=account.buying_power,
        holdings=tuple(holdings),
        marks={symbol: mark.price for symbol, mark in marks.items()},
        drawdown_pct=context.drawdown(account.equity),
        orders_today=orders_today,
        turnover_today=turnover_today,
        killswitch_tripped=context.killswitch_tripped,
    )

    if plan.is_empty:
        context.last_note = _no_trade_note(plan)
        return EquityCycleRecord(
            cycle_id=cycle_id,
            kind=CYCLE_KIND,
            as_of=as_of,
            stage=EquityStage.NO_MARKS if plan.is_blind else EquityStage.NO_TRADES,
            strategy=strategy_view(book),
            equity=account.equity,
            band_pct=plan.band_pct,
            orders=(),
            skipped=tuple(_skip_view(s) for s in plan.skipped),
            turnover=Decimal(0),
            note=_no_trade_note(plan),
        )

    orders, stage, note = _place(
        plan, book, portfolio, context, as_of=as_of, submit=submit
    )
    context.last_note = note
    return EquityCycleRecord(
        cycle_id=cycle_id,
        kind=CYCLE_KIND,
        as_of=as_of,
        stage=stage,
        strategy=strategy_view(book),
        equity=account.equity,
        band_pct=plan.band_pct,
        orders=orders,
        skipped=tuple(_skip_view(s) for s in plan.skipped),
        turnover=sum(
            (o.notional for o in orders if o.submission is not None), Decimal(0)
        ),
        note=note,
    )


def _place(
    plan: RebalancePlan,
    book: TargetBook,
    portfolio: EquityPortfolio,
    context: EquityContext,
    *,
    as_of: datetime,
    submit: bool,
) -> tuple[tuple[EquityOrderRecord, ...], EquityStage, str]:
    """Gate and send each intent in plan order, advancing the snapshot as we go.

    The advancing is the subtle part. A pass that judged every order against the
    account as it stood at the top would let a hundred orders each pass the
    daily-turnover check that the hundred of them together fail — the cap would
    be enforced between passes and not within one, which is not what it says.

    A transport failure stops the pass rather than skipping to the next order.
    The orders already placed stand and are journalled; the rest are simply not
    attempted, and the next pass re-derives them from what is actually held,
    which is the one source that cannot be wrong about what happened.
    """
    records: list[EquityOrderRecord] = []
    live = portfolio
    approved = 0

    for intent in plan.intents:
        verdict = evaluate_equity(
            intent,
            book,
            live,
            context.policy,
            as_of,
            pinned_fingerprint=context.pinned_fingerprint,
        )
        if isinstance(verdict, VetoedEquity):
            records.append(_order_record(intent, verdict, None, "vetoed"))
            continue
        approved += 1

        if not submit or context.mcp is None:
            records.append(_order_record(intent, verdict, None, "dry_run"))
            live = _advance(live, intent)
            continue

        try:
            submission = submit_equity(verdict.order, context.mcp)
        except ExecutionError as failure:
            records.append(_order_record(intent, verdict, None, f"halted: {failure}"))
            return (
                tuple(records),
                EquityStage.HALTED,
                (
                    f"stopped at {intent.symbol} after {len(records) - 1} orders: "
                    f"{failure}. The next pass re-derives the rest from the book "
                    "the broker actually holds."
                ),
            )
        records.append(
            _order_record(intent, verdict, submission, submission.status.value)
        )
        live = _advance(live, intent)

    if approved == 0:
        return (
            tuple(records),
            EquityStage.VETOED,
            f"all {len(plan.intents)} intents refused by the Gate",
        )
    stage = EquityStage.SUBMITTED if submit and context.mcp else EquityStage.PLANNED
    verb = "submitted" if stage is EquityStage.SUBMITTED else "gated, not sent"
    return (
        tuple(records),
        stage,
        f"{approved} of {len(plan.intents)} intents {verb}",
    )


def _advance(portfolio: EquityPortfolio, intent: OrderIntent) -> EquityPortfolio:
    """The snapshot as it will be once this order fills.

    Assuming the fill is the right assumption *within* a pass — a market order on
    a liquid name during regular hours fills, and the alternative is a hundred
    orders each budgeting against an account none of them have touched. It is the
    wrong assumption *between* passes, which is why the next pass reads the book
    from the broker instead of carrying this forward.
    """
    delta = intent.shares if intent.side is EquitySide.BUY else -intent.shares
    holdings = list(portfolio.holdings)
    for index, holding in enumerate(holdings):
        if holding.symbol == intent.symbol:
            holdings[index] = Holding(
                symbol=holding.symbol,
                shares=holding.shares + delta,
                average_price=holding.average_price,
                market_value=(holding.shares + delta) * intent.reference_price,
            )
            break
    else:
        holdings.append(
            Holding(
                symbol=intent.symbol,
                shares=delta,
                average_price=intent.reference_price,
                market_value=delta * intent.reference_price,
            )
        )
    spend = intent.notional if intent.side is EquitySide.BUY else -intent.notional
    return EquityPortfolio(
        equity=portfolio.equity,
        cash=portfolio.cash - spend,
        buying_power=portfolio.buying_power - spend,
        holdings=tuple(h for h in holdings if h.shares != 0),
        marks=portfolio.marks,
        drawdown_pct=portfolio.drawdown_pct,
        orders_today=portfolio.orders_today + 1,
        turnover_today=portfolio.turnover_today + intent.notional,
        killswitch_tripped=portfolio.killswitch_tripped,
    )


def _order_record(
    intent: OrderIntent,
    verdict: EquityVerdict,
    submission: Submission | None,
    outcome: str,
) -> EquityOrderRecord:
    return EquityOrderRecord(
        symbol=str(intent.symbol),
        side=intent.side.value,
        shares=intent.shares,
        reference_price=intent.reference_price,
        notional=intent.notional,
        target_weight=intent.target_weight,
        held_weight=intent.held_weight,
        verdict=verdict,
        submission=submission,
        outcome=outcome,
    )


def _empty(
    cycle_id: str,
    as_of: datetime,
    stage: EquityStage,
    *,
    strategy: Mapping[str, Any] | None = None,
    note: str = "",
) -> EquityCycleRecord:
    """A pass that decided nothing, and says why. Still journalled."""
    return EquityCycleRecord(
        cycle_id=cycle_id,
        kind=CYCLE_KIND,
        as_of=as_of,
        stage=stage,
        strategy=strategy or {},
        equity=Decimal(0),
        band_pct=Decimal(0),
        orders=(),
        skipped=(),
        turnover=Decimal(0),
        note=note,
    )


def _no_trade_note(plan: RebalancePlan) -> str:
    counts = plan.counts()
    inside = counts.get("inside_band", 0)
    parts = [f"{inside} symbols inside the {plan.band_pct:.0%} band"]
    for reason in sorted(counts):
        if reason != "inside_band":
            parts.append(f"{counts[reason]} {reason}")
    return "; ".join(parts)


def _money(value: Any) -> Decimal:
    """Journal values come back as strings — specs/06's `encode`, in reverse."""
    try:
        return Decimal(str(value))
    except (TypeError, ArithmeticError, ValueError):
        return Decimal(0)


def approved_orders(record: EquityCycleRecord) -> Sequence[EquityOrderRecord]:
    """Convenience for the CLI and the status page."""
    return [order for order in record.orders if isinstance(order.verdict, ApprovedEquity)]
