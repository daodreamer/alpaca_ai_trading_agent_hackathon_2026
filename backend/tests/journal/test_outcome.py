"""06 D2 `outcome` and D3 — the later facts, as amendments.

The discipline under test is one sentence of the spec: *"The original decision
therefore stays exactly as it was made, with no hindsight leaking backwards into
it."* Everything here is a way of trying to make that false.

The interesting cases are not "does an amendment work" — `test_writer.py`
answers that — but the ones where a shortcut would be tempting: a second
read-back, a fill and then a close, a partial that resolves. Each of those is a
place where "just update the line" would be easier and would quietly turn the
journal into a database with one row per cycle and no memory.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal

from alphagate.agent import Stage
from alphagate.execution import OrderStatus
from alphagate.journal import Fill, Journal, outcome_from, realised
from tests.journal.conftest import DAY, at_stage, submission

FILLED_AT = datetime(2026, 8, 26, 18, 0, tzinfo=UTC)
CLOSED_AT = datetime(2026, 8, 28, 19, 30, tzinfo=UTC)


class TestReadingABrokerAnswer:
    def test_it_carries_the_fills_leg_by_leg(self) -> None:
        outcome = outcome_from(
            submission("order_filled"), cycle_id="2026-08-26-SPY-000", observed_at=FILLED_AT
        )
        assert outcome.status is OrderStatus.FILLED
        assert [fill.symbol for fill in outcome.fills] == [
            "SPY260904P00747000",
            "SPY260904P00752000",
        ]
        assert outcome.filled_quantity == 2

    def test_prices_stay_decimal(self) -> None:
        """A fill price that has been through a float is a fill price that
        disagrees with the order it records."""
        outcome = outcome_from(
            submission("order_filled"), cycle_id="2026-08-26-SPY-000", observed_at=FILLED_AT
        )
        assert [fill.price for fill in outcome.fills] == [Decimal("1.31"), Decimal("1.91")]

    def test_it_is_pure(self) -> None:
        """Same submission, same `observed_at`, same record. Twice."""
        args = {"cycle_id": "2026-08-26-SPY-000", "observed_at": FILLED_AT}
        assert outcome_from(submission("order_filled"), **args) == outcome_from(
            submission("order_filled"), **args
        )

    def test_realised_pl_starts_as_none_never_zero(self) -> None:
        """An open position has no realised P&L. Zero would read as a scratch."""
        outcome = outcome_from(
            submission("order_filled"), cycle_id="2026-08-26-SPY-000", observed_at=FILLED_AT
        )
        assert outcome.realised_pl is None
        assert outcome.closed_at is None

    def test_a_live_order_is_not_terminal(self) -> None:
        outcome = outcome_from(
            submission("get_order_by_client_id"),
            cycle_id="2026-08-26-SPY-000",
            observed_at=FILLED_AT,
        )
        assert outcome.status is OrderStatus.NEW
        assert not outcome.is_terminal

    def test_an_unfilled_leg_has_no_price_rather_than_a_zero_one(self) -> None:
        assert Fill(symbol="X", quantity=0, price=None, status="new").notional is None
        assert Fill(symbol="X", quantity=2, price=Decimal("1.25"), status="filled").notional == (
            Decimal("2.50")
        )


class TestOutcomesAreAmendments:
    """specs/06 D3, on the type that exists to be one."""

    def test_the_decision_line_is_byte_identical_afterwards(self, journal: Journal) -> None:
        record = at_stage(Stage.SUBMITTED)
        journal.append(record)
        before = journal.path_for(DAY).read_text(encoding="utf-8")

        journal.record_outcome(
            outcome_from(
                submission("order_filled"), cycle_id=record.cycle_id, observed_at=FILLED_AT
            )
        )
        after = journal.path_for(DAY).read_text(encoding="utf-8")
        assert after.startswith(before)
        assert len(after.strip().split("\n")) == 2

    def test_the_fill_reaches_the_final_state(self, journal: Journal) -> None:
        record = at_stage(Stage.SUBMITTED)
        journal.append(record)
        journal.record_outcome(
            outcome_from(
                submission("order_filled"), cycle_id=record.cycle_id, observed_at=FILLED_AT
            )
        )
        line = journal.read(DAY)[0]
        assert line["stage"] == "submitted", "the stage records what we knew then"
        assert line["outcome"]["status"] == "filled", "the outcome records what happened"

    def test_the_original_line_never_learns_the_fill(self, journal: Journal) -> None:
        """The one that would be trivially easy to get wrong and impossible to
        notice: hindsight leaking backwards into the decision."""
        record = at_stage(Stage.SUBMITTED)
        journal.append(record)
        journal.record_outcome(
            outcome_from(
                submission("order_filled"), cycle_id=record.cycle_id, observed_at=FILLED_AT
            )
        )
        first = json.loads(journal.path_for(DAY).read_text(encoding="utf-8").split("\n")[0])
        assert "outcome" not in first

    def test_a_second_read_back_converges_without_overwriting(
        self, journal: Journal
    ) -> None:
        """A reconciler is expected to run on a timer. Two passes must leave two
        lines and one answer."""
        record = at_stage(Stage.SUBMITTED)
        journal.append(record)
        journal.record_outcome(
            outcome_from(
                submission("get_order_by_client_id"),
                cycle_id=record.cycle_id,
                observed_at=datetime(2026, 8, 26, 15, 0, tzinfo=UTC),
            )
        )
        journal.record_outcome(
            outcome_from(
                submission("order_filled"), cycle_id=record.cycle_id, observed_at=FILLED_AT
            )
        )
        assert len(journal.raw_lines(DAY)) == 3
        assert journal.read(DAY)[0]["outcome"]["status"] == "filled"

    def test_the_close_arrives_as_its_own_line_later_still(self, journal: Journal) -> None:
        """"a fill hours after submission, realised P&L on close" — two facts,
        two lines, days apart."""
        record = at_stage(Stage.SUBMITTED)
        journal.append(record)
        fill = outcome_from(
            submission("order_filled"), cycle_id=record.cycle_id, observed_at=FILLED_AT
        )
        journal.record_outcome(fill)
        journal.record_outcome(
            realised(fill, Decimal("-42.50"), closed_at=CLOSED_AT), day=DAY
        )

        final = journal.read(DAY)[0]["outcome"]
        assert final["realised_pl"] == "-42.50"
        assert final["closed_at"] == CLOSED_AT.isoformat()
        assert final["status"] == "filled", "the close does not lose the fill"

    def test_the_close_is_filed_under_the_day_of_the_decision(
        self, journal: Journal
    ) -> None:
        """A P&L line filed under the day it was learned would leave the
        decision it belongs to permanently unamended."""
        record = at_stage(Stage.SUBMITTED)
        journal.append(record)
        fill = outcome_from(
            submission("order_filled"), cycle_id=record.cycle_id, observed_at=FILLED_AT
        )
        path = journal.record_outcome(realised(fill, Decimal("60.00"), closed_at=CLOSED_AT))
        assert path.name == "2026-08-26.jsonl", "the cycle id names the file, not the clock"

    def test_an_outcome_without_a_cycle_is_refused(self, journal: Journal) -> None:
        """Rather than being written somewhere plausible and read by nobody.

        The realistic failure is not a wrong type — `Amendable` and mypy handle
        that before the code runs — it is a real `OutcomeRecord` built with an
        empty id, which types fine and would file an amendment nothing can ever
        match."""
        import pytest

        orphan = outcome_from(
            submission("order_filled"), cycle_id="", observed_at=FILLED_AT
        )
        with pytest.raises(ValueError, match="must name the cycle"):
            journal.record_outcome(orphan)

    def test_realised_keeps_the_provenance_of_the_fill_it_closes(self) -> None:
        fill = outcome_from(
            submission("order_filled"), cycle_id="2026-08-26-SPY-000", observed_at=FILLED_AT
        )
        closed = realised(fill, Decimal("60.00"), closed_at=CLOSED_AT)
        assert closed.envelope == fill.envelope
        assert closed.fills == fill.fills
        assert closed.order_id == fill.order_id
