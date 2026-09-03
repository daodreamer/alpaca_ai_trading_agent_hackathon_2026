"""One session, driven — specs/05 D1 and D8.

The thin part. `schedule.py` decides *when*, `cycle.py` decides *what*, and this
walks one against the other: for each slot, evaluate exits, then — if the slot
allows opening — perceive, screen, enumerate, propose, gate, submit, record.

It is deliberately small and deliberately dull, because everything interesting
has already been made pure somewhere else. The only genuinely stateful decisions
here are three:

**Exits go first, every slot.** Before any thought is given to opening
something new. A cycle that spends its model call deciding what to buy and then
discovers it should have closed a position twenty minutes ago has the ordering
backwards, and the ordering is free to get right.

**A breach stops the session.** A partial fill is a naked leg (specs/04 D5); the
kill switch latches, opens are refused from that moment, and the runner stops
proposing rather than continuing to be refused. Closes remain available, which
is the whole point of the Gate never blocking an exit.

**Nothing is inferred about positions.** Every slot re-reads them from the
broker. A position we believe in but the broker does not is the worse of the two
ways to be wrong, and one cheap call per fifteen minutes is not a budget worth
economising against.

`sleep` and `now` are injected, so a whole trading day runs in a test in
milliseconds and the live path differs by two arguments.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Final

from alphagate.agent.cycle import CycleRecord, run_cycle
from alphagate.agent.model import Candidate, MarketRead, Setup, Stage
from alphagate.agent.proposer import DEFAULT_PROPOSER, Proposer
from alphagate.agent.schedule import Slot
from alphagate.agent.screen import DefaultScreen, Screen
from alphagate.execution import McpSession
from alphagate.journal import Journal, ReconcileResult, reconcile
from alphagate.risk import Intent, PortfolioSnapshot, RiskLimits

__all__ = ["CycleInputs", "SessionResult", "run_session"]

MAX_CONSECUTIVE_ERRORS: Final = 3
"""Stop the session after this many slots fail outright.

Not zero: a single transient is not a reason to abandon a trading day. Not
unbounded: a session that fails every slot for six hours and journals nothing
useful is worse than one that stops and says so."""


@dataclass(frozen=True, slots=True)
class CycleInputs:
    """Everything one slot needs, gathered by the caller.

    A callable rather than a bundle of ports, so the runner does not have to
    know whether the chain came from REST, a replay or a fixture — and so the
    whole of this module stays testable without a market.
    """

    read: MarketRead
    candidates: tuple[Candidate, ...]
    portfolio: PortfolioSnapshot
    exits: tuple[CycleRecord, ...] = ()
    """Exit cycles already run for this slot, ready to be journalled."""
    entry_block: str = ""
    """Why the option book's own rule forbids an entry this slot, if it does.

    Decided by the gatherer, because the caps are the *book's* and this module
    has never read a book — `agent.option_book.entry_refusal` writes the
    sentence and `live/wiring.py` supplies it the two numbers. Non-empty means
    the cycle stops at the screen with this as its reason, and no model is
    asked: there is nothing to ask about a menu the rule will not allow."""


type Gather = Callable[[Slot], CycleInputs]
type Sleeper = Callable[[float], None]
type Clock = Callable[[], datetime]


@dataclass
class SessionResult:
    """What one session did. Every slot that ran is in here."""

    records: list[CycleRecord] = field(default_factory=list)
    stopped_early: str | None = None
    """Why the session ended before its last slot, if it did."""
    reconciled: int = 0
    """Outcome amendments written during the session — specs/06 D3."""
    unreadable: list[tuple[str, str]] = field(default_factory=list)
    """`(cycle_id, why)` for orders the broker could not be asked about.

    Surfaced on the result rather than only in the journal because an order we
    have lost track of is the one thing an operator must see before the next
    session opens, and specs/04 D4 forbids guessing what it was."""

    @property
    def fills(self) -> int:
        return sum(1 for record in self.records if record.stage is Stage.FILLED)

    @property
    def by_stage(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in self.records:
            counts[record.stage.value] = counts.get(record.stage.value, 0) + 1
        return counts

    def summary(self) -> str:
        parts = ", ".join(f"{stage}={count}" for stage, count in sorted(self.by_stage.items()))
        tail = f" — stopped: {self.stopped_early}" if self.stopped_early else ""
        amended = f", {self.reconciled} amended" if self.reconciled else ""
        lost = f", {len(self.unreadable)} UNREADABLE" if self.unreadable else ""
        return f"{len(self.records)} cycles ({parts}){amended}{lost}{tail}"


def run_session(
    slots: Sequence[Slot],
    gather: Gather,
    *,
    limits: RiskLimits,
    journal: Journal,
    proposer: Proposer = DEFAULT_PROPOSER,
    screen: Screen | None = None,
    mcp: McpSession | None = None,
    sleep: Sleeper = time.sleep,
    now: Clock = lambda: datetime.now(UTC),
    reconcile_open_orders: bool = True,
) -> SessionResult:
    """Walk one session's slots. Journals every cycle, including the failures.

    Returns rather than raises on a slot that blows up: a session that dies
    halfway through with an exception has lost the record of why, and the record
    is the product.

    **Open orders are reconciled once per slot**, before anything else. A limit
    submitted at 14:30 that fills at 15:05 is a fact the 14:30 line cannot
    contain, and specs/06 D3 says it arrives as an amendment — so something has
    to go and ask. Doing it here, at the top of the slot, means the exit logic
    below sees fills the broker already has rather than a book we believe in;
    and because amendments converge rather than overwrite, a slot that asks
    about an order it already resolved costs nothing but the call it skips.

    Set `reconcile_open_orders=False` for a dry run, where there are no orders
    to ask about and the extra call would be a lie about what the session did.
    """
    chosen_screen = screen if screen is not None else DefaultScreen()
    result = SessionResult()
    consecutive_errors = 0
    killswitch_latched = False

    for slot in slots:
        _wait_for(slot, sleep=sleep, now=now)

        try:
            inputs = gather(slot)
        except Exception as exc:
            consecutive_errors += 1
            result.stopped_early = f"gather failed at {slot}: {type(exc).__name__}: {exc}"
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                return result
            continue
        consecutive_errors = 0

        # 0. Reconcile what the broker already knows — specs/06 D3.
        if mcp is not None and reconcile_open_orders:
            _absorb(result, _reconcile_quietly(journal, mcp, slot.at))

        # 1. Exits first, always, and on every slot including the late ones.
        for exit_record in inputs.exits:
            journal.append(exit_record)
            result.records.append(exit_record)
            if exit_record.stage is Stage.BREACHED:
                killswitch_latched = True

        if killswitch_latched:
            result.stopped_early = "partial fill breach — opens blocked, reconcile by hand"
            return result

        # 2. Entries, only where the schedule and the book allow them.
        if not slot.kind.may_open:
            continue
        if inputs.portfolio.killswitch_tripped:
            result.stopped_early = "kill switch latched on the incoming snapshot"
            return result

        # The book's own cadence and concurrency caps, before the screen: a
        # slot the rule forbids is not a slot to read the market for.
        setup = None if inputs.entry_block else chosen_screen.screen(inputs.read)
        record = run_cycle(
            read=inputs.read,
            setup=setup,
            candidates=inputs.candidates,
            portfolio=inputs.portfolio,
            limits=limits,
            as_of=slot.at,
            mcp=mcp,
            proposer=proposer,
            sequence=slot.sequence,
            intent=Intent.OPEN,
            screen_reason=_screen_reason(chosen_screen, inputs, setup),
        )
        journal.append(record)
        result.records.append(record)

        if record.stage is Stage.BREACHED:
            result.stopped_early = "partial fill breach — opens blocked, reconcile by hand"
            return result

    return result


def _screen_reason(screen: Screen, inputs: CycleInputs, setup: Setup | None) -> str:
    """The `NO_SETUP` note: the rule's refusal, the screen's, or nothing.

    The book's own caps come first because they were decided first — the screen
    never ran when they refuse, so asking it to explain a decision it did not
    make would put an invented reason in the journal.
    """
    if inputs.entry_block:
        return inputs.entry_block
    return "" if setup is not None else screen.explain(inputs.read)


def _reconcile_quietly(journal: Journal, mcp: McpSession, at: datetime) -> ReconcileResult:
    """Ask the broker about open orders, and never let that end the session.

    A reconciliation pass is housekeeping. If it raises — a transport failure, a
    day file that will not parse — the trading session must carry on and the
    failure must be visible, because the alternative is an agent that stops
    trading because it could not update a record.
    """
    try:
        return reconcile(journal, at.date(), mcp, as_of=at)
    except Exception as exc:
        return ReconcileResult(failures=[("*", f"{type(exc).__name__}: {exc}")])


def _absorb(result: SessionResult, pass_result: ReconcileResult) -> None:
    result.reconciled += len(pass_result.amended)
    result.unreadable.extend(pass_result.failures)


def _wait_for(slot: Slot, *, sleep: Sleeper, now: Clock) -> None:
    """Sleep until the slot is due. Never sleeps for a slot already past.

    A negative wait means the previous cycle overran — a slow model call, a
    retried submission. Running immediately is right: skipping to the next slot
    would silently drop a cycle from the journal, and the journal is meant to
    have one line per scheduled cycle whether or not it was punctual.
    """
    delay = (slot.at - now()).total_seconds()
    if delay > 0:
        sleep(delay)


def pending(slots: Sequence[Slot], after: datetime) -> Iterator[Slot]:
    """The slots still to come. For a runner resuming mid-session."""
    return (slot for slot in slots if slot.at > after)


def equity_of(portfolio: PortfolioSnapshot) -> Decimal:
    """Convenience for callers assembling the next slot's inputs."""
    return portfolio.equity
