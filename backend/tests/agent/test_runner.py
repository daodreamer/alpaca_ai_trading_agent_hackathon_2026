"""05 D1 and D8 — one session, driven.

A whole trading day in milliseconds: `sleep` and `now` are injected, so the
runner's real behaviour is exercised without waiting for it. That is the payoff
for making the schedule pure — the alternative is a test that either waits six
hours or tests nothing.

The three decisions with teeth, and each gets its own class:

* exits run first on every slot, including the late ones;
* a breach stops the session with the kill switch latched;
* a slot that blows up is journalled and survived, until too many are.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from alphagate.agent import Choice, Stage, session_slots
from alphagate.agent.cycle import CycleRecord
from alphagate.agent.runner import (
    MAX_CONSECUTIVE_ERRORS,
    CycleInputs,
    SessionResult,
    run_session,
)
from alphagate.agent.schedule import CycleKind, Slot
from alphagate.execution import RecordedSession, ToolTimeout
from alphagate.execution.submit import READ_BACK_TOOL
from alphagate.journal import Journal
from alphagate.risk import DEFAULT_LIMITS, Intent, PortfolioSnapshot
from tests.agent.conftest import EQUITY, SPY, StubProposer, book, menu, read, setup

OPEN = datetime(2026, 8, 26, 13, 30, tzinfo=UTC)
CLOSE = datetime(2026, 8, 26, 20, 0, tzinfo=UTC)
DAY = OPEN.date()
FILLED = (
    Path(__file__).resolve().parents[1] / "fixtures" / "mcp" / "order_filled.json"
).read_text(encoding="utf-8")


PRIOR_SEQUENCE = 99
"""Far from the sequences this session will use.

The first draft used 0, which is the sequence of the session's own first slot —
so the runner's fresh record and the pre-existing one shared a `cycle_id` and
`read` collapsed them (specs/06 D2, and the reason `duplicate_cycles` exists).
The amendment was applied correctly and then buried, which is the failure that
looks exactly like the amendment not working."""


def submitted_cycle() -> CycleRecord:
    """A decision already on disk whose order is still live at the broker."""
    from tests.journal.conftest import at_stage

    return at_stage(Stage.SUBMITTED, sequence=PRIOR_SEQUENCE)


def prior(journal: Journal) -> str:
    record = submitted_cycle()
    journal.append(record)
    return record.cycle_id


def by_id(journal: Journal, cycle_id: str) -> dict[str, Any]:
    return next(line for line in journal.read(DAY) if line["cycle_id"] == cycle_id)


class Recorder:
    """Injected clock and sleeper. Advances only when the runner sleeps."""

    def __init__(self, start: datetime) -> None:
        self.moment = start
        self.slept: list[float] = []

    def now(self) -> datetime:
        return self.moment

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.moment += timedelta(seconds=seconds)


class Gatherer:
    """A `Gather` that records what it was asked for, and can be told to fail."""

    def __init__(
        self,
        *,
        portfolio: PortfolioSnapshot | None = None,
        exits: tuple[CycleRecord, ...] = (),
        fail_on: set[int] | None = None,
        entry_block: str = "",
    ) -> None:
        self.portfolio = portfolio if portfolio is not None else book()
        self.exits = exits
        self.fail_on = fail_on or set()
        self.entry_block = entry_block
        self.seen: list[Slot] = []

    def __call__(self, slot: Slot) -> CycleInputs:
        self.seen.append(slot)
        if slot.sequence in self.fail_on:
            raise RuntimeError(f"chain fetch failed at slot {slot.sequence}")
        return CycleInputs(
            read=read(),
            candidates=menu(),
            portfolio=self.portfolio,
            exits=self.exits,
            entry_block=self.entry_block,
        )


def gather_for(
    *,
    portfolio: PortfolioSnapshot | None = None,
    exits: tuple[CycleRecord, ...] = (),
    fail_on: set[int] | None = None,
    entry_block: str = "",
) -> Gatherer:
    return Gatherer(
        portfolio=portfolio, exits=exits, fail_on=fail_on, entry_block=entry_block
    )



def exit_record(stage: Stage = Stage.FILLED, sequence: int = 0) -> CycleRecord:
    return CycleRecord(
        cycle_id=f"2026-08-26-SPY-exit-{sequence:03d}",
        as_of=OPEN,
        stage=stage,
        read=read(),
        setup=setup(),
        candidates=(),
        choice=None,
        call=None,
        proposal=None,
        verdict=None,
        submission=None,
        note="deterministic exit",
    )


def run(
    slots: Sequence[Slot], journal: Journal, **kwargs: Any
) -> tuple[SessionResult, Recorder]:
    clock = Recorder(OPEN)
    defaults: dict[str, Any] = {
        "limits": DEFAULT_LIMITS,
        "journal": journal,
        "proposer": StubProposer(choice=Choice(None, "no edge", 0.1)),
        "sleep": clock.sleep,
        "now": clock.now,
    }
    defaults.update(kwargs)
    gather = defaults.pop("gather")
    return run_session(slots, gather, **defaults), clock


class TestOneSession:
    def test_it_runs_every_slot(self, tmp_path: Path) -> None:
        slots = session_slots(OPEN, CLOSE)[:6]
        journal = Journal(directory=tmp_path)
        result, _ = run(slots, journal, gather=gather_for())
        assert len(result.records) == 6
        assert result.stopped_early is None

    def test_every_cycle_is_journalled(self, tmp_path: Path) -> None:
        """specs/05 D1 step 8 through the runner: one line per scheduled cycle."""
        slots = session_slots(OPEN, CLOSE)[:5]
        journal = Journal(directory=tmp_path)
        run(slots, journal, gather=gather_for())
        assert len(journal.read(OPEN.date())) == 5

    def test_it_sleeps_until_each_slot(self, tmp_path: Path) -> None:
        slots = session_slots(OPEN, CLOSE)[:3]
        _, clock = run(slots, Journal(directory=tmp_path), gather=gather_for())
        # 20 minutes to the first, then 15 to each of the next two.
        assert clock.slept == [1200.0, 900.0, 900.0]

    def test_a_late_cycle_runs_immediately_rather_than_being_skipped(
        self, tmp_path: Path
    ) -> None:
        """"Skipping to the next slot would silently drop a cycle from the
        journal, and the journal is meant to have one line per scheduled
        cycle whether or not it was punctual." """
        slots = session_slots(OPEN, CLOSE)[:3]
        journal = Journal(directory=tmp_path)
        clock = Recorder(CLOSE)  # every slot is already in the past
        result = run_session(
            slots,
            gather_for(),
            limits=DEFAULT_LIMITS,
            journal=journal,
            proposer=StubProposer(choice=Choice(None, "no", 0.0)),
            sleep=clock.sleep,
            now=clock.now,
        )
        assert clock.slept == []
        assert len(result.records) == 3

    def test_the_summary_counts_by_stage(self, tmp_path: Path) -> None:
        slots = session_slots(OPEN, CLOSE)[:4]
        result, _ = run(slots, Journal(directory=tmp_path), gather=gather_for())
        assert result.by_stage == {"declined": 4}
        assert "declined=4" in result.summary()


class TestExitsGoFirst:
    def test_exits_are_journalled_before_the_entry(self, tmp_path: Path) -> None:
        """"A cycle that spends its model call deciding what to buy and then
        discovers it should have closed a position twenty minutes ago has the
        ordering backwards." """
        slots = session_slots(OPEN, CLOSE)[:1]
        journal = Journal(directory=tmp_path)
        result, _ = run(
            slots, journal, gather=gather_for(exits=(exit_record(),))
        )
        assert [record.stage for record in result.records] == [
            Stage.FILLED,
            Stage.DECLINED,
        ]

    def test_exits_run_on_slots_that_may_not_open(self, tmp_path: Path) -> None:
        """The late slots. specs/03 D4 through the scheduler: closing at 15:55
        is exactly when you want to be able to."""
        late = [slot for slot in session_slots(OPEN, CLOSE) if not slot.kind.may_open]
        assert late, "the fixture session must have exits-only slots"
        journal = Journal(directory=tmp_path)
        result, _ = run(late, journal, gather=gather_for(exits=(exit_record(),)))
        assert {record.stage for record in result.records} == {Stage.FILLED}

    def test_no_entry_is_attempted_on_an_exits_only_slot(self, tmp_path: Path) -> None:
        late = [slot for slot in session_slots(OPEN, CLOSE) if not slot.kind.may_open]
        proposer = StubProposer(choice=Choice(0, "would open", 0.9))
        gatherer = gather_for()
        result, _ = run(
            late, Journal(directory=tmp_path), gather=gatherer, proposer=proposer
        )
        assert result.records == []
        assert proposer.seen == [], "the model must not be asked on an exits-only slot"
        assert len(gatherer.seen) == len(late), "the slot still ran; it just did not open"


class TestBreachStopsTheSession:
    def test_a_breached_exit_latches_and_stops(self, tmp_path: Path) -> None:
        slots = session_slots(OPEN, CLOSE)[:5]
        journal = Journal(directory=tmp_path)
        result, _ = run(
            slots,
            journal,
            gather=gather_for(exits=(exit_record(Stage.BREACHED),)),
        )
        assert len(result.records) == 1
        assert result.stopped_early is not None
        assert "naked leg" in result.stopped_early or "breach" in result.stopped_early

    def test_a_latched_snapshot_stops_before_proposing(self, tmp_path: Path) -> None:
        """The Gate would refuse every open anyway; continuing to ask is a way
        of spending model calls on a foregone conclusion."""
        slots = session_slots(OPEN, CLOSE)[:5]
        proposer = StubProposer(choice=Choice(0, "would open", 0.9))
        result, _ = run(
            slots,
            Journal(directory=tmp_path),
            gather=gather_for(portfolio=book(killswitch_tripped=True)),
            proposer=proposer,
        )
        assert result.records == []
        assert result.stopped_early == "kill switch latched on the incoming snapshot"
        assert proposer.seen == []


class TestFailures:
    def test_one_bad_slot_does_not_end_the_day(self, tmp_path: Path) -> None:
        slots = session_slots(OPEN, CLOSE)[:5]
        journal = Journal(directory=tmp_path)
        result, _ = run(slots, journal, gather=gather_for(fail_on={0}))
        assert len(result.records) == 4

    def test_enough_bad_slots_do(self, tmp_path: Path) -> None:
        """"A session that fails every slot for six hours and journals nothing
        useful is worse than one that stops and says so." """
        slots = session_slots(OPEN, CLOSE)[:8]
        result, _ = run(
            slots,
            Journal(directory=tmp_path),
            gather=gather_for(fail_on={0, 1, 2, 3}),
        )
        assert result.records == []
        assert result.stopped_early is not None
        assert MAX_CONSECUTIVE_ERRORS == 3

    def test_the_counter_resets_on_a_good_slot(self, tmp_path: Path) -> None:
        slots = session_slots(OPEN, CLOSE)[:8]
        result, _ = run(
            slots,
            Journal(directory=tmp_path),
            gather=gather_for(fail_on={0, 1, 3, 4}),
        )
        assert len(result.records) == 4, "slots 2, 5, 6 and 7 ran"

    def test_a_failure_never_raises_out_of_the_session(self, tmp_path: Path) -> None:
        """"A session that dies halfway through with an exception has lost the
        record of why, and the record is the product." """
        slots = session_slots(OPEN, CLOSE)[:3]
        result, _ = run(
            slots, Journal(directory=tmp_path), gather=gather_for(fail_on={0, 1, 2})
        )
        assert isinstance(result.stopped_early, str)


class TestTheBooksOwnCadence:
    """A slot the rule forbids an entry on is journalled, and asks no model.

    The refusal is computed where the book and the journal are (`gather`), and
    arrives as `CycleInputs.entry_block`. The runner turns it into the same
    `NO_SETUP` line a screen refusal produces — one line per slot, with the
    reason in it — and never reaches the proposer, because there is nothing to
    ask about a menu the rule will not let it take.
    """

    def test_the_slot_is_journalled_with_the_rules_reason(self, tmp_path: Path) -> None:
        journal = Journal(directory=tmp_path)
        result, _ = run(
            session_slots(OPEN, CLOSE)[:1],
            journal,
            gather=gather_for(entry_block="the rule allows 3 concurrent position(s)"),
        )
        (record,) = result.records
        assert record.stage is Stage.NO_SETUP
        assert record.note == "the rule allows 3 concurrent position(s)"

    def test_no_model_is_asked(self, tmp_path: Path) -> None:
        proposer = StubProposer()
        run(
            session_slots(OPEN, CLOSE)[:1],
            Journal(directory=tmp_path),
            gather=gather_for(entry_block="the rule wants 1 session(s) between entries"),
            proposer=proposer,
        )
        assert proposer.seen == []

    def test_exits_still_run(self, tmp_path: Path) -> None:
        """The cadence is about *entries*. A rule that stopped managing what it
        already holds would be a rule that opens positions and abandons them."""
        result, _ = run(
            session_slots(OPEN, CLOSE)[:1],
            Journal(directory=tmp_path),
            gather=gather_for(entry_block="the rule allows 3", exits=(exit_record(),)),
        )
        assert [r.stage for r in result.records] == [Stage.FILLED, Stage.NO_SETUP]

    def test_an_unblocked_slot_still_screens(self, tmp_path: Path) -> None:
        proposer = StubProposer()
        run(
            session_slots(OPEN, CLOSE)[:1],
            Journal(directory=tmp_path),
            gather=gather_for(),
            proposer=proposer,
        )
        assert proposer.seen != []


class TestOpenOrdersAreReconciled:
    """specs/06 D3, driven. The runner is the thing that drives it.

    D3 describes amendments; `journal.reconcile` produces them; without this
    wiring nothing in production ever calls it, and "a fill hours after
    submission" stays a sentence in a spec. These tests are about the wiring
    specifically — the reconciler's own behaviour is covered in
    `tests/journal/test_reconcile.py`.
    """

    def test_a_fill_that_lands_later_is_amended_onto_the_decision(
        self, tmp_path: Path
    ) -> None:
        journal = Journal(directory=tmp_path / "j")
        cycle_id = prior(journal)
        slots = session_slots(OPEN, CLOSE)[:1]

        result, _ = run(
            slots,
            journal,
            gather=Gatherer(),
            mcp=RecordedSession.scripted(**{READ_BACK_TOOL: FILLED}),
        )

        assert result.reconciled == 1
        assert by_id(journal, cycle_id)["outcome"]["status"] == "filled"
        assert by_id(journal, cycle_id)["stage"] == "filled", (
            "and the cycle now reads as a fill, which is what puts its legs in "
            "the book -- see `journal.writer._with_final_stage`"
        )
        assert [
            line["stage"]
            for line in journal.raw_lines(DAY)
            if line.get("cycle_id") == cycle_id and "stage" in line
        ] == ["submitted"], "the decision line on disk stands, untouched"

    def test_a_dry_run_asks_the_broker_nothing(self, tmp_path: Path) -> None:
        """`mcp=None` means there is nothing submitted to reconcile, and a call
        would be a lie about what the session did."""
        journal = Journal(directory=tmp_path / "j")
        prior(journal)
        result, _ = run(session_slots(OPEN, CLOSE)[:2], journal, gather=Gatherer())
        assert result.reconciled == 0

    def test_it_can_be_switched_off(self, tmp_path: Path) -> None:
        journal = Journal(directory=tmp_path / "j")
        prior(journal)
        session = RecordedSession.scripted(**{READ_BACK_TOOL: FILLED})
        result, _ = run(
            session_slots(OPEN, CLOSE)[:1],
            journal,
            gather=Gatherer(),
            mcp=session,
            reconcile_open_orders=False,
        )
        assert result.reconciled == 0
        assert session.calls == []

    def test_a_reconciliation_failure_never_stops_the_session(
        self, tmp_path: Path
    ) -> None:
        """Housekeeping. An agent that stops trading because it could not update
        a record has the priorities backwards — but the failure must be loud."""
        journal = Journal(directory=tmp_path / "j")
        prior(journal)
        slots = session_slots(OPEN, CLOSE)[:3]

        result, _ = run(
            slots,
            journal,
            gather=Gatherer(),
            mcp=RecordedSession.scripted(
                **{READ_BACK_TOOL: [ToolTimeout("no answer")] * 3}
            ),
        )
        assert len(result.records) == len(slots), "the day ran to the end"
        assert result.stopped_early is None
        assert result.unreadable, "and said so"
        assert "UNREADABLE" in result.summary()

    def test_it_runs_before_exits_so_the_book_is_current(
        self, tmp_path: Path
    ) -> None:
        """An exit decided against a position the broker already filled out from
        under us is a decision made on a stale book."""
        journal = Journal(directory=tmp_path / "j")
        cycle_id = prior(journal)
        session = RecordedSession.scripted(**{READ_BACK_TOOL: FILLED})
        gatherer = Gatherer(exits=(exit_record(),))

        run(
            session_slots(OPEN, CLOSE)[:1],
            journal,
            gather=gatherer,
            mcp=session,
        )
        lines = journal.raw_lines(DAY)
        kinds = [
            "amendment" if line.get("amendment") else str(line.get("cycle_id", ""))
            for line in lines
        ]
        assert kinds[0] == cycle_id, "the decision was already there"
        assert kinds[1] == "amendment", "then the fill"
        assert kinds[2].endswith("exit-000"), "then the exit"

    def test_a_second_slot_does_not_ask_again_about_a_resolved_order(
        self, tmp_path: Path
    ) -> None:
        """Amendments converge; the queue shrinks. Otherwise every slot of the
        day re-asks about every order the day ever placed."""
        journal = Journal(directory=tmp_path / "j")
        prior(journal)
        session = RecordedSession.scripted(**{READ_BACK_TOOL: [FILLED]})

        result, _ = run(
            session_slots(OPEN, CLOSE)[:4], journal, gather=Gatherer(), mcp=session
        )
        assert result.reconciled == 1
        assert len(session.calls) == 1, "asked once, then never again"


class TestEntriesUseTheOpenIntent:
    def test_the_runner_opens_rather_than_closes(self, tmp_path: Path) -> None:
        """Exits are `evaluate_exit`'s job and arrive pre-made; the entry path
        is unconditionally an open."""
        slots = session_slots(OPEN, CLOSE)[:1]
        rich = replace(DEFAULT_LIMITS)
        result, _ = run(
            slots,
            Journal(directory=tmp_path),
            gather=gather_for(),
            limits=rich,
            proposer=StubProposer(choice=Choice(0, "take it", 0.6)),
        )
        assert result.records[0].proposal is not None
        assert result.records[0].proposal.intent is Intent.OPEN


def test_the_fixture_session_has_both_slot_kinds() -> None:
    kinds = {slot.kind for slot in session_slots(OPEN, CLOSE)}
    assert kinds == {CycleKind.FULL, CycleKind.EXITS_ONLY}


def test_the_book_fixture_is_the_size_the_menu_assumes() -> None:
    assert book().equity == EQUITY
    assert book().equity == Decimal(100_000)
    assert read().underlying == SPY


@pytest.mark.parametrize("count", [1, 3, 10])
def test_slot_counts_are_respected(count: int, tmp_path: Path) -> None:
    slots = session_slots(OPEN, CLOSE)[:count]
    result, _ = run(slots, Journal(directory=tmp_path), gather=gather_for())
    assert len(result.records) == count
