"""05 D1, D3, D5, D6 — the cycle. Test plan items 1, 2, 3, 5.

The first class is the one that matters most. "The cycle always reaches step 8"
is not a nice property, it is the thing that makes the journal evidence: a
record that exists only when the agent traded cannot answer "why didn't it trade
at 14:30?", and that is the question the whole project is arranged to answer.

Everything here runs offline. The proposer is a stub, the session replays
captured payloads, and `as_of` is a constant.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest

from alphagate.agent import Choice, Stage, run_cycle
from alphagate.agent.cycle import CycleRecord, cycle_id_for, portfolio_after
from alphagate.agent.proposer import (
    DecliningProposer,
    DeterministicProposer,
    RecordedProposer,
)
from alphagate.execution import RecordedSession, ToolTimeout, TransportFailure
from alphagate.execution.mapping import PLACE_ORDER_TOOL
from alphagate.execution.submit import READ_BACK_TOOL
from alphagate.risk import DEFAULT_LIMITS, Intent, RiskLimits
from tests.agent.conftest import NOW, SPY, StubProposer, book, menu, read, setup
from tests.execution.conftest import payload, payload_json
from tests.execution.test_submit import envelope_wrap

ACCEPTED = payload("place_option_order")
FILLED = payload("order_filled")


def order_response(**fields: object) -> str:
    """A captured order payload with fields overridden, still enveloped.

    Going through `envelope_wrap` matters: a payload without the
    `_alpaca_mcp_security` wrapper takes a different branch in `unwrap`, and a
    test that skips it is exercising a shape the server never sends.
    """
    data = dict(payload_json("place_option_order")["data"])
    data.update(fields)
    return envelope_wrap(data)


def run(**overrides: Any) -> CycleRecord:
    kwargs: dict[str, Any] = {
        "read": read(),
        "setup": setup(),
        "candidates": menu(),
        "portfolio": book(),
        "limits": DEFAULT_LIMITS,
        "as_of": NOW,
        "proposer": StubProposer(),
    }
    kwargs.update(overrides)
    return run_cycle(**kwargs)


class TestEveryPathRecords:
    """specs/05 test plan item 1. The load-bearing property of the layer."""

    def test_no_setup(self) -> None:
        record = run(setup=None)
        assert record.stage is Stage.NO_SETUP
        assert record.cycle_id
        assert record.read is not None

    def test_no_candidates(self) -> None:
        record = run(candidates=())
        assert record.stage is Stage.NO_CANDIDATES
        assert record.candidates == ()

    def test_declined(self) -> None:
        record = run(proposer=StubProposer(choice=Choice(None, "iv too low", 0.1)))
        assert record.stage is Stage.DECLINED
        assert record.choice is not None
        assert record.choice.rationale == "iv too low"

    def test_vetoed(self) -> None:
        record = run(portfolio=book(drawdown="0.20"))
        assert record.stage is Stage.VETOED
        assert "drawdown_killswitch" in record.veto_reasons
        assert record.verdict is not None

    def test_submitted(self) -> None:
        session = RecordedSession.scripted(**{PLACE_ORDER_TOOL: ACCEPTED})
        record = run(mcp=session)
        assert record.stage is Stage.SUBMITTED
        assert record.submission is not None

    def test_filled(self) -> None:
        session = RecordedSession.scripted(**{PLACE_ORDER_TOOL: FILLED})
        record = run(mcp=session)
        assert record.stage is Stage.FILLED
        assert record.traded

    def test_every_stage_is_reachable(self) -> None:
        """A stage nothing can produce is a stage the dashboard renders and
        nobody has ever seen — worse than not having it."""
        produced = {
            run(setup=None).stage,
            run(candidates=()).stage,
            run(proposer=StubProposer(choice=Choice(None, "no", 0.0))).stage,
            run(portfolio=book(drawdown="0.20")).stage,
            run().stage,
            run(mcp=RecordedSession.scripted(**{PLACE_ORDER_TOOL: ACCEPTED})).stage,
            run(mcp=RecordedSession.scripted(**{PLACE_ORDER_TOOL: FILLED})).stage,
        }
        assert {
            Stage.NO_SETUP,
            Stage.NO_CANDIDATES,
            Stage.DECLINED,
            Stage.VETOED,
            Stage.DRY_RUN,
            Stage.SUBMITTED,
            Stage.FILLED,
        } <= produced

    def test_the_record_always_carries_the_read(self) -> None:
        """Even on the paths that never got near a trade."""
        for record in (run(setup=None), run(candidates=()), run()):
            assert record.read.underlying == SPY
            assert record.as_of == NOW


class TestTheModelCannotOverstepD3:
    """specs/05 D3 and test plan item 2."""

    @pytest.mark.parametrize("index", [99, 12, -1, -5])
    def test_an_out_of_range_index_is_a_decline_not_a_retry(self, index: int) -> None:
        """"An index outside the candidate range is a decline, not an error to
        retry around." A model naming a candidate that does not exist has not
        chosen, and asking again is asking it to guess harder."""
        record = run(proposer=StubProposer(choice=Choice(index, "pick that one", 0.9)))
        assert record.stage is Stage.DECLINED
        assert record.proposal is None
        assert str(index) in record.note

    def test_the_rationale_survives_an_unusable_index(self) -> None:
        """The journal still shows what it said, even though nothing was done."""
        record = run(proposer=StubProposer(choice=Choice(42, "very confident", 1.0)))
        assert record.choice is not None
        assert record.choice.rationale == "very confident"

    def test_a_valid_index_selects_that_exact_candidate(self) -> None:
        candidates = menu()
        record = run(candidates=candidates, proposer=StubProposer(choice=Choice(2, "third", 0.5)))
        assert record.proposal is not None
        assert record.proposal.structure is candidates[2].structure

    def test_the_model_has_no_channel_for_a_symbol(self) -> None:
        """The structural claim, as a test over the type rather than a comment."""
        fields = set(Choice.__dataclass_fields__)
        assert fields == {"candidate_index", "rationale", "self_reported_confidence"}


class TestFailClosed:
    """specs/05 D6 and test plan item 3."""

    def test_no_proposer_configured_means_no_trade(self) -> None:
        record = run(proposer=DecliningProposer())
        assert record.stage is Stage.DECLINED
        assert record.proposal is None

    def test_a_proposer_that_errors_still_declines(self) -> None:
        stub = StubProposer(choice=Choice(None, "declined: timeout", 0.0), error="timeout")
        record = run(proposer=stub)
        assert record.stage is Stage.DECLINED
        assert record.call is not None
        assert record.call.error == "timeout"

    def test_a_submission_failure_is_recorded_not_raised(self) -> None:
        session = RecordedSession.scripted(
            **{PLACE_ORDER_TOOL: TransportFailure("connection reset")}
        )
        record = run(mcp=session)
        assert record.stage is Stage.REJECTED
        assert "TransportFailure" in record.note
        assert record.verdict is not None, "the gate's tape survives a failed submission"

    def test_a_rejection_carries_its_reason(self) -> None:
        wrapped = order_response(
            status="rejected", reject_reason="insufficient options buying power"
        )
        record = run(mcp=RecordedSession.scripted(**{PLACE_ORDER_TOOL: wrapped}))
        assert record.stage is Stage.REJECTED
        assert record.submission is not None
        assert record.submission.reason == "insufficient options buying power"

    def test_a_timeout_resolves_by_read_back_inside_the_cycle(self) -> None:
        session = RecordedSession.scripted(
            **{
                PLACE_ORDER_TOOL: ToolTimeout("no answer"),
                READ_BACK_TOOL: payload("get_order_by_client_id"),
            }
        )
        record = run(mcp=session)
        assert record.submission is not None
        assert record.submission.resolved_by_readback
        assert len(session.calls_to(PLACE_ORDER_TOOL)) == 1

    def test_a_partial_fill_latches_the_kill_switch(self) -> None:
        """specs/04 D5 meeting specs/03 D4: the Gate is pure, so the latch rides
        in on the next snapshot."""
        legs = payload_json("place_option_order")["data"]["legs"]
        half = [dict(legs[0], status="filled", filled_qty="1"), dict(legs[1])]
        session = RecordedSession.scripted(
            **{PLACE_ORDER_TOOL: order_response(status="partially_filled", legs=half)}
        )
        record = run(mcp=session)
        assert record.stage is Stage.BREACHED
        assert "naked leg" in record.note
        assert portfolio_after(book(), record).killswitch_tripped

    def test_the_kill_switch_blocks_the_next_open(self, limits: RiskLimits) -> None:
        record = run(portfolio=book(killswitch_tripped=True))
        assert record.stage is Stage.VETOED
        assert "drawdown_killswitch" in record.veto_reasons

    def test_a_dry_run_gates_but_does_not_submit(self) -> None:
        record = run(mcp=None)
        assert record.stage is Stage.DRY_RUN
        assert record.verdict is not None
        assert record.submission is None


class TestConfidenceIsNeverActedOn:
    """specs/05 D5 and test plan item 5."""

    @pytest.mark.parametrize("confidence", [0.0, 0.01, 0.5, 1.0])
    def test_quantity_does_not_move_with_confidence(self, confidence: float) -> None:
        record = run(proposer=StubProposer(choice=Choice(0, "reasoning", confidence)))
        assert record.proposal is not None
        assert record.proposal.quantity == menu()[0].quantity

    def test_the_verdict_does_not_move_with_confidence(self) -> None:
        low = run(proposer=StubProposer(choice=Choice(0, "r", 0.01)))
        high = run(proposer=StubProposer(choice=Choice(0, "r", 0.99)))
        assert type(low.verdict) is type(high.verdict)
        assert low.proposal is not None
        assert high.proposal is not None
        assert low.proposal.quantity == high.proposal.quantity

    def test_it_is_still_recorded(self) -> None:
        record = run(proposer=StubProposer(choice=Choice(0, "r", 0.83)))
        assert record.choice is not None
        assert record.choice.self_reported_confidence == 0.83


class TestTheProposalCarriesProvenance:
    def test_the_rationale_reaches_the_trade_proposal(self) -> None:
        """specs/03 D2: `rationale` is evidence for the journal, never Gate input."""
        record = run(proposer=StubProposer(choice=Choice(0, "because of X", 0.4)))
        assert record.proposal is not None
        assert record.proposal.rationale == "because of X"

    def test_the_model_id_is_recorded_as_the_proposer(self) -> None:
        record = run(proposer=StubProposer(model="deepseek-chat"))
        assert record.proposal is not None
        assert record.proposal.proposed_by == "deepseek-chat"

    def test_the_cycle_id_is_the_proposal_id(self) -> None:
        """Which is what ties the journal line to the broker's order."""
        record = run()
        assert record.proposal is not None
        assert record.proposal.proposal_id == record.cycle_id


class TestCycleIds:
    def test_the_shape_is_date_ticker_sequence(self) -> None:
        assert cycle_id_for(NOW, "SPY", 3) == "2026-08-26-SPY-003"

    def test_they_do_not_collide_across_underlyings_or_days(self) -> None:
        ids = {
            cycle_id_for(NOW + timedelta(days=day), ticker, sequence)
            for day in range(4)
            for ticker in ("SPY", "QQQ", "AAPL")
            for sequence in range(26)
        }
        assert len(ids) == 4 * 3 * 26


class TestReplay:
    """specs/05 D7 and test plan item 6."""

    def test_a_recorded_choice_reproduces_the_order(self) -> None:
        live = run(proposer=StubProposer(choice=Choice(1, "second best", 0.6)))
        assert live.proposal is not None

        replay = run(
            proposer=RecordedProposer({live.cycle_id: Choice(1, "second best", 0.6)})
        )
        assert replay.proposal is not None
        assert replay.proposal.structure == live.proposal.structure
        assert replay.proposal.quantity == live.proposal.quantity

    def test_an_unrecorded_cycle_declines_rather_than_guessing(self) -> None:
        record = run(proposer=RecordedProposer({}))
        assert record.stage is Stage.DECLINED
        assert record.choice is not None
        assert "no recorded choice" in record.choice.rationale

    def test_the_deterministic_proposer_needs_no_network(self) -> None:
        record = run(proposer=DeterministicProposer())
        assert record.stage is Stage.DRY_RUN
        assert record.call is not None
        assert record.call.model == "deterministic-v1"

    def test_the_deterministic_proposer_declines_on_cheap_premium(self) -> None:
        record = run(read=read(iv_rank="20"), proposer=DeterministicProposer())
        assert record.stage is Stage.DECLINED
        assert record.choice is not None
        assert "below" in record.choice.rationale

    def test_two_identical_cycles_agree(self) -> None:
        """Steps 5 and 6 are pure, so with the proposer fixed the whole tail is."""
        a = run(proposer=RecordedProposer({cycle_id_for(NOW, "SPY", 0): Choice(0, "r", 0.5)}))
        b = run(proposer=RecordedProposer({cycle_id_for(NOW, "SPY", 0): Choice(0, "r", 0.5)}))
        assert a.stage is b.stage
        assert a.proposal == b.proposal
        assert a.verdict == b.verdict


class TestPortfolioAdvance:
    def test_a_fill_increments_the_daily_count(self) -> None:
        record = run(mcp=RecordedSession.scripted(**{PLACE_ORDER_TOOL: FILLED}))
        assert portfolio_after(book(fills_today=2), record).fills_today == 3

    def test_a_decline_does_not(self) -> None:
        record = run(proposer=StubProposer(choice=Choice(None, "no", 0.0)))
        assert portfolio_after(book(fills_today=2), record).fills_today == 2

    def test_positions_are_not_inferred(self) -> None:
        """A position we believe in but the broker does not is the worse of the
        two ways to be wrong, so positions are re-read rather than advanced."""
        before = book()
        record = run(mcp=RecordedSession.scripted(**{PLACE_ORDER_TOOL: FILLED}))
        assert portfolio_after(before, record).positions == before.positions


class TestExits:
    def test_a_close_is_gated_but_never_blocked(self) -> None:
        """specs/03 D4 through the cycle: a hostile book still lets an exit out."""
        record = run(
            portfolio=book(drawdown="0.40", fills_today=99, killswitch_tripped=True),
            intent=Intent.CLOSE,
            mcp=RecordedSession.scripted(**{PLACE_ORDER_TOOL: FILLED}),
        )
        assert record.stage is Stage.FILLED
        assert record.proposal is not None
        assert record.proposal.intent is Intent.CLOSE
