"""05 D8 — the exit cycle, and the order that follows from it.

`exits.py` decides; this is the path that acts on the decision, and until it
existed the agent could open a position and never close one. `evaluate_exit` was
fully implemented and fully tested and reachable from nothing — which is the
worst shape a safety mechanism can be in, because the tests all pass.

Four properties, in the order they would hurt to lose:

1. **A hold writes nothing.** Twenty-six lines a day per position saying "still
   fine" is a journal nobody reads.
2. **A close goes through the Gate.** Not exempt, not fast-pathed. That is what
   makes `execution`'s "only a `GatedOrder`" rule hold on this path too.
3. **No model is consulted.** The decision to take a loss is the one you least
   want improvised.
4. **The quantity and structure are the ones already held.** An exit that closes
   a different size leaves a naked remainder.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from alphagate.agent import Stage
from alphagate.agent.book import HeldPosition
from alphagate.agent.cycle import CycleRecord, run_exit_cycle
from alphagate.agent.exits import DEFAULT_EXIT_POLICY, ExitPolicy
from alphagate.agent.model import MarketRead
from alphagate.execution import RecordedSession
from alphagate.execution.mapping import PLACE_ORDER_TOOL
from alphagate.options import (
    Leg,
    OptionContract,
    OptionQuote,
    OptionStructure,
    Right,
    Side,
    StructureKind,
    compute_risk,
)
from alphagate.risk import DEFAULT_LIMITS, Intent, OpenPosition
from tests.agent.conftest import EXPIRY, SPY, book
from tests.journal.conftest import payload

NOW = datetime(2026, 8, 26, 14, 30, tzinfo=UTC)
OPENED = NOW - timedelta(days=3)


def contract(strike: str) -> OptionContract:
    return OptionContract(SPY, EXPIRY, Decimal(strike), Right.PUT)


def spread() -> OptionStructure:
    """Short the 752 put, long the 747 — **one contract a leg**.

    A *unit* structure, because that is the only kind the pipeline builds:
    `candidates.vertical_credit_spreads` uses `Leg(contract, side)` with the
    default quantity, and size is carried separately in `TradeProposal.quantity`
    and `OpenPosition.quantity`. A fixture with three-contract legs would be
    testing a shape that never reaches this code, and would quietly cancel the
    double-counting this file exists to catch.
    """
    return OptionStructure(
        StructureKind.VERTICAL_CREDIT,
        (Leg(contract("752"), Side.SELL), Leg(contract("747"), Side.BUY)),
    )


def quotes(short: str, long: str, *, as_of: datetime = NOW) -> dict:
    def quote(target: OptionContract, price: str) -> OptionQuote:
        return OptionQuote(target, as_of, Decimal(price), Decimal(price), None)

    return {contract("752"): quote(contract("752"), short),
            contract("747"): quote(contract("747"), long)}


def held(
    *, quantity: int = 1, entry_premium: str = "60.00", structure: OptionStructure | None = None
) -> HeldPosition:
    used = structure if structure is not None else spread()
    return HeldPosition(
        position=OpenPosition(
            structure=used,
            quantity=quantity,
            max_loss=Decimal("440.00") * quantity,
            net_greeks=None,
            opened_at=OPENED,
        ),
        entry_premium=Decimal(entry_premium),
        cycle_id="2026-08-23-SPY-004",
    )


def read() -> MarketRead:
    return MarketRead(underlying=SPY, as_of=NOW, spot=Decimal("766.00"))


def run(
    *,
    position: HeldPosition | None = None,
    short: str = "0.30",
    long: str = "0.10",
    as_of: datetime = NOW,
    mcp: object = None,
    policy: ExitPolicy = DEFAULT_EXIT_POLICY,
) -> CycleRecord | None:
    item = position if position is not None else held()
    current = compute_risk(item.position.structure, quotes(short, long, as_of=as_of), as_of)
    return run_exit_cycle(
        held=item,
        current=current,
        read=read(),
        portfolio=book(),
        limits=DEFAULT_LIMITS,
        as_of=as_of,
        mcp=mcp,  # type: ignore[arg-type]
        policy=policy,
    )


class TestHoldingWritesNothing:
    def test_a_position_that_should_be_held_produces_no_record(self) -> None:
        """The majority case, every slot, for every open position."""
        assert run(short="1.40", long="0.90") is None

    def test_it_is_the_policy_that_decides_not_this_function(self) -> None:
        """Same numbers, a stricter target, a different answer — so the rule
        lives in `ExitPolicy` and not in a threshold buried here."""
        entered = Decimal("60.00")
        assert run(short="1.40", long="0.90", position=held(entry_premium=str(entered))) is None
        eager = ExitPolicy(profit_target=Decimal("0.05"))
        assert run(short="1.40", long="0.90", policy=eager) is not None


class TestClosing:
    def test_a_profit_target_produces_a_close(self) -> None:
        record = run(short="0.30", long="0.10")
        assert record is not None
        assert record.proposal is not None
        assert record.proposal.intent is Intent.CLOSE
        assert "profit_target" in record.note

    def test_a_stop_produces_a_close(self) -> None:
        """Taking the loss is a decision the policy makes and the agent obeys."""
        record = run(short="2.60", long="0.40")
        assert record is not None
        assert "stop" in record.note

    def test_expiry_pressure_produces_a_close(self) -> None:
        """specs/07 D6: close at two days out regardless of the mark."""
        late = datetime.combine(EXPIRY, datetime.min.time(), tzinfo=UTC) - timedelta(days=1)
        # A mark that fires neither the profit target nor the stop, so the only
        # rule left is the calendar.
        record = run(short="1.40", long="0.90", as_of=late)
        assert record is not None
        assert "dte" in record.note

    def test_it_closes_the_structure_that_is_held(self) -> None:
        """Not a re-derived one. An exit that closes something else leaves the
        position open and adds a second."""
        item = held(quantity=3)
        record = run(position=item, short="0.30", long="0.10")
        assert record is not None
        assert record.proposal is not None
        assert record.proposal.structure == item.position.structure

    def test_it_closes_the_whole_size(self) -> None:
        """A partial close on a defined-risk spread is a different position with
        a different maximum loss, and nobody asked for one."""
        record = run(position=held(quantity=3), short="0.30", long="0.10")
        assert record is not None
        assert record.proposal is not None
        assert record.proposal.quantity == 3

    def test_the_rule_and_its_numbers_reach_the_journal(self) -> None:
        """specs/06: a judge reads why. "closed" is not why."""
        record = run(short="0.30", long="0.10")
        assert record is not None
        assert record.proposal is not None
        assert record.proposal.rationale.startswith("profit_target")
        assert record.note


class TestTheGateStillRuns:
    def test_a_close_carries_a_verdict(self) -> None:
        """Not exempt and not fast-pathed — that is what keeps `execution`'s
        only-a-`GatedOrder` rule true on this path."""
        record = run(short="0.30", long="0.10")
        assert record is not None
        assert record.verdict is not None
        assert len(record.verdict.checks) == 13

    def test_the_gate_does_not_block_an_exit(self, ) -> None:
        """specs/03 D4. Even with the kill switch latched and the day's fill cap
        used up, a close goes through — that is the whole point of the Gate
        distinguishing intent."""
        item = held()
        current = compute_risk(item.position.structure, quotes("0.30", "0.10"), NOW)
        record = run_exit_cycle(
            held=item,
            current=current,
            read=read(),
            portfolio=book(killswitch_tripped=True, fills_today=99, drawdown="0.40"),
            limits=DEFAULT_LIMITS,
            as_of=NOW,
        )
        assert record is not None
        assert record.stage is not Stage.VETOED


class TestSubmission:
    def test_without_a_session_it_is_a_dry_run(self) -> None:
        record = run(short="0.30", long="0.10")
        assert record is not None
        assert record.stage is Stage.DRY_RUN
        assert record.submission is None

    def test_with_a_session_the_order_goes_out(self) -> None:
        session = RecordedSession.scripted(**{PLACE_ORDER_TOOL: payload("order_filled")})
        record = run(short="0.30", long="0.10", mcp=session)
        assert record is not None
        assert record.stage is Stage.FILLED
        assert record.submission is not None
        assert len(session.calls) == 1

    def test_the_order_is_a_close_on_the_wire(self) -> None:
        """`position_intent` has to say so, or the broker opens a new position
        in the opposite direction instead of flattening this one."""
        session = RecordedSession.scripted(**{PLACE_ORDER_TOOL: payload("order_filled")})
        run(short="0.30", long="0.10", mcp=session)
        _, arguments = session.calls[0]
        legs = arguments["legs"]
        assert all("close" in leg["position_intent"] for leg in legs)  # type: ignore[index,union-attr]

    def test_a_rejected_close_is_recorded_not_swallowed(self) -> None:
        """A close that did not happen is a position still open. Nothing about
        that may be quiet."""
        import json

        rejected = json.loads(payload("order_filled"))
        rejected["data"]["status"] = "rejected"
        rejected["data"]["reject_reason"] = "no such position"
        session = RecordedSession.scripted(**{PLACE_ORDER_TOOL: json.dumps(rejected)})
        record = run(short="0.30", long="0.10", mcp=session)
        assert record is not None
        assert record.stage is Stage.REJECTED


class TestNoModelIsInvolved:
    def test_the_record_has_no_model_call(self) -> None:
        record = run(short="0.30", long="0.10")
        assert record is not None
        assert record.call is None
        assert record.choice is None
        assert record.candidates == ()

    def test_it_is_attributed_to_the_policy(self) -> None:
        record = run(short="0.30", long="0.10")
        assert record is not None
        assert record.proposal is not None
        assert record.proposal.proposed_by == "exit-policy"

    def test_it_is_deterministic(self) -> None:
        """Same position, same marks, same instant, same answer."""
        first, second = run(short="0.30", long="0.10"), run(short="0.30", long="0.10")
        assert first is not None
        assert second is not None
        assert first.cycle_id == second.cycle_id
        assert first.note == second.note


@pytest.mark.parametrize("quantity", [1, 2, 10])
def test_the_decision_does_not_depend_on_the_size(quantity: int) -> None:
    """A three-lot and a one-lot at the same mark reach the same conclusion.

    The entry premium is per unit and so is the mark, so the *fraction* earned —
    which is what the policy thresholds are expressed in — is independent of
    size. A one-lot that closes and a ten-lot that holds would mean one of the
    two figures had been scaled and the other had not, which is precisely the
    bug this pairing was written to prevent."""
    item = held(quantity=quantity)
    record = run(position=item, short="0.30", long="0.10")
    assert record is not None
    assert "profit_target" in record.note
