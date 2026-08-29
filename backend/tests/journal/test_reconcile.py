"""The producer for D3's later facts — `journal.reconcile`.

D3 describes a mechanism. A mechanism nobody drives is a mechanism nobody has
tested against reality, so this covers the thing that drives it: read the day
back, find the orders still live, ask the broker, append one amendment each.

The properties worth having, in the order they would hurt to lose:

1. **Re-running is safe.** It runs on a timer. If a second pass could duplicate
   a fill or overwrite a line, nobody could run it without thinking first.
2. **An unreadable order is left alone, loudly.** specs/04 D4 refuses to guess
   between "no order exists" and "an order exists we cannot see". Reconciliation
   inherits the refusal rather than softening it into a default.
3. **One bad order does not abandon the rest.** A session with ten open orders
   and one unreadable must reconcile nine.

`RecordedSession` replays captured payloads; nothing here opens a socket.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from alphagate.agent import Stage
from alphagate.execution import RecordedSession, ToolTimeout
from alphagate.journal import Journal, outcome_from, reconcile, unresolved
from tests.journal.conftest import DAY, at_stage, cycle, payload, submission

AS_OF = datetime(2026, 8, 26, 18, 0, tzinfo=UTC)
READ_BACK_TOOL = "get_order_by_client_id"


def submitted(journal: Journal, *, sequence: int = 0) -> str:
    """Journal one cycle that reached the broker and is still live.

    A real `run_cycle` against a scripted session, so the `client_order_id` the
    reconciler reads back is the one `submit` actually minted rather than one a
    test chose. If the fingerprint scheme (specs/04 D5) ever changed shape, the
    reconciler would stop finding its own orders and this would catch it.
    """
    record = at_stage(Stage.SUBMITTED, sequence=sequence)
    journal.append(record)
    return record.cycle_id


class TestFindingWhatIsStillOpen:
    """`unresolved` is pure: a file in, a work list out."""

    def test_a_submitted_cycle_is_pending(self, journal: Journal) -> None:
        cycle_id = submitted(journal)
        pending = unresolved(journal.read(DAY))
        assert [item.cycle_id for item in pending] == [cycle_id]
        assert pending[0].client_order_id == "alphagate-1204108820879f25366cd5b6"

    def test_a_cycle_that_never_reached_the_broker_is_not(self, journal: Journal) -> None:
        """`VETOED` and `DRY_RUN` have no order to ask about. Asking would be a
        call per quiet cycle, and the quiet cycles are the majority."""
        journal.append(cycle())
        journal.append(at_stage(Stage.VETOED, sequence=1))
        assert unresolved(journal.read(DAY)) == ()

    def test_an_already_filled_cycle_drops_out(self, journal: Journal) -> None:
        """Which is why this reads `Journal.read` and not `raw_lines`: the
        amendments are the record of what we already learned."""
        cycle_id = submitted(journal)
        journal.record_outcome(
            outcome_from(submission("order_filled"), cycle_id=cycle_id, observed_at=AS_OF)
        )
        assert unresolved(journal.read(DAY)) == ()

    def test_an_unknown_status_stays_in_the_queue(self, journal: Journal) -> None:
        """specs/04 D4: an unrecognised status is an unresolved order, and
        calling it done is how one goes missing."""
        cycle_id = submitted(journal)
        journal.amend(cycle_id, outcome={"status": "something_new"})
        assert [item.cycle_id for item in unresolved(journal.read(DAY))] == [cycle_id]

    def test_it_is_deterministic_and_in_journal_order(self, journal: Journal) -> None:
        for sequence in range(3):
            submitted(journal, sequence=sequence)
        records = journal.read(DAY)
        assert unresolved(records) == unresolved(records)
        assert [item.cycle_id[-3:] for item in unresolved(records)] == ["000", "001", "002"]


class TestReconciling:
    def test_a_fill_lands_as_an_amendment(self, journal: Journal) -> None:
        cycle_id = submitted(journal)
        session = RecordedSession.scripted(**{READ_BACK_TOOL: payload("order_filled")})

        result = reconcile(journal, DAY, session, as_of=AS_OF)

        assert [outcome.cycle_id for outcome in result.amended] == [cycle_id]
        assert result.resolved == 1
        assert journal.read(DAY)[0]["outcome"]["status"] == "filled"

    def test_the_decision_line_is_untouched(self, journal: Journal) -> None:
        submitted(journal)
        before = journal.path_for(DAY).read_text(encoding="utf-8")
        session = RecordedSession.scripted(**{READ_BACK_TOOL: payload("order_filled")})

        reconcile(journal, DAY, session, as_of=AS_OF)
        assert journal.path_for(DAY).read_text(encoding="utf-8").startswith(before)

    def test_running_it_twice_is_safe(self, journal: Journal) -> None:
        """It runs on a timer. The second pass must find nothing to do."""
        submitted(journal)
        session = RecordedSession.scripted(
            **{READ_BACK_TOOL: [payload("order_filled"), payload("order_filled")]}
        )
        first = reconcile(journal, DAY, session, as_of=AS_OF)
        second = reconcile(journal, DAY, session, as_of=AS_OF)

        assert len(first.amended) == 1
        assert second.amended == [], "already terminal — nothing left to ask about"
        assert len(journal.raw_lines(DAY)) == 2

    def test_a_still_live_order_is_amended_and_asked_again_next_pass(
        self, journal: Journal
    ) -> None:
        """A `new` order is a real answer, worth recording, and not the end."""
        submitted(journal)
        session = RecordedSession.scripted(
            **{
                READ_BACK_TOOL: [
                    payload("get_order_by_client_id"),
                    payload("order_filled"),
                ]
            }
        )
        first = reconcile(journal, DAY, session, as_of=AS_OF)
        assert first.resolved == 0

        second = reconcile(journal, DAY, session, as_of=AS_OF)
        assert second.resolved == 1
        assert journal.read(DAY)[0]["outcome"]["status"] == "filled"

    def test_an_unreadable_order_is_reported_not_guessed_at(
        self, journal: Journal
    ) -> None:
        """specs/04 D4. Guessing here means choosing between losing a position
        and duplicating one, and those have opposite consequences."""
        cycle_id = submitted(journal)
        session = RecordedSession.scripted(**{READ_BACK_TOOL: ToolTimeout("no answer")})

        result = reconcile(journal, DAY, session, as_of=AS_OF)

        assert result.amended == []
        assert [failed for failed, _ in result.failures] == [cycle_id]
        assert "outcome" not in journal.read(DAY)[0], "nothing was invented"

    def test_one_unreadable_order_does_not_abandon_the_others(
        self, journal: Journal
    ) -> None:
        for sequence in range(3):
            submitted(journal, sequence=sequence)
        session = RecordedSession.scripted(
            **{
                READ_BACK_TOOL: [
                    payload("order_filled"),
                    ToolTimeout("no answer"),
                    payload("order_filled"),
                ]
            }
        )
        result = reconcile(journal, DAY, session, as_of=AS_OF)
        assert len(result.amended) == 2
        assert len(result.failures) == 1

    def test_it_reads_no_clock(self, journal: Journal) -> None:
        """specs/01 Rule 5 has no exception for the code that is genuinely about
        the passage of time — which is where the temptation is strongest."""
        submitted(journal)
        session = RecordedSession.scripted(**{READ_BACK_TOOL: payload("order_filled")})
        result = reconcile(journal, DAY, session, as_of=AS_OF)
        assert result.amended[0].observed_at == AS_OF

    def test_a_quiet_day_asks_the_broker_nothing(self, journal: Journal) -> None:
        journal.append(cycle(with_setup=False))
        session = RecordedSession.scripted()
        assert reconcile(journal, DAY, session, as_of=AS_OF).amended == []
        assert session.calls == []

    def test_the_summary_says_what_happened(self, journal: Journal) -> None:
        submitted(journal)
        session = RecordedSession.scripted(**{READ_BACK_TOOL: payload("order_filled")})
        assert reconcile(journal, DAY, session, as_of=AS_OF).summary() == (
            "1 amended (1 terminal)"
        )


class TestOrphans:
    def test_an_amendment_with_no_record_is_reported_not_dropped(
        self, journal: Journal
    ) -> None:
        """A fill for a decision this file does not contain is a reconciliation
        question. Answerable beats lost."""
        journal.append(cycle())
        journal.amend("2026-08-26-QQQ-004", outcome={"status": "filled"})
        assert journal.orphaned_amendments(DAY) == ("2026-08-26-QQQ-004",)

    def test_a_day_that_adds_up_has_no_orphans(self, journal: Journal) -> None:
        cycle_id = submitted(journal)
        journal.amend(cycle_id, outcome={"status": "filled"})
        assert journal.orphaned_amendments(DAY) == ()


@pytest.mark.parametrize("stage", [Stage.NO_SETUP, Stage.DECLINED, Stage.FILLED])
def test_only_submitted_cycles_are_worth_a_call(journal: Journal, stage: Stage) -> None:
    """`FILLED` is already terminal; the rest never reached the broker."""
    journal.append(at_stage(stage))
    assert unresolved(journal.read(DAY)) == ()
