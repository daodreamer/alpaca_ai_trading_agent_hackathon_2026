"""05 D7 / 06 D6 — a journalled day replays to an identical order set.

Test plan item 6, done the strict way: the first pass writes real JSONL to disk,
the second pass reads the choices back **out of that file** and replays them.
Not from an in-memory dict — from the bytes. That is the difference between
testing the replay mechanism and testing the artefact the submission actually
ships, and only one of them is evidence.

> "If replay diverges, something impure leaked into steps 1–3, 5 or 6. The
> replay test is therefore also the determinism test for the whole pipeline,
> which is why it is worth more than its size suggests."

Everything replays from captured payloads. No network, no model, no clock.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from alphagate.agent import (
    Choice,
    CycleRecord,
    DefaultScreen,
    Stage,
    build_candidates,
    cycle_id_for,
    perceive,
    run_cycle,
    session_slots,
)
from alphagate.agent.candidates import vertical_credit_spreads
from alphagate.agent.proposer import Proposer, RecordedProposer
from alphagate.agent.replay import recorded_choices, replay_proposer
from alphagate.core.identifiers import Ticker, ticker
from alphagate.execution import McpSession, RecordedSession
from alphagate.execution.mapping import PLACE_ORDER_TOOL
from alphagate.execution.submit import READ_BACK_TOOL
from alphagate.journal import Journal, reconcile, unresolved
from alphagate.marketdata import RecordedMarketData
from alphagate.options import OptionContract, OptionQuote, Right
from alphagate.risk import DEFAULT_LIMITS, PortfolioSnapshot
from tests.agent.conftest import StubProposer

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "marketdata"
MCP_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "mcp"
FILLED = (MCP_FIXTURES / "order_filled.json").read_text(encoding="utf-8")
SPY: Ticker = ticker("SPY")
OPEN = datetime(2026, 8, 26, 13, 30, tzinfo=UTC)
CLOSE = datetime(2026, 8, 26, 20, 0, tzinfo=UTC)
EQUITY = Decimal(100_000)
DAY = date(2026, 8, 26)


@pytest.fixture
def data() -> RecordedMarketData:
    return RecordedMarketData(directory=FIXTURES)


def chain(data: RecordedMarketData) -> Mapping[OptionContract, OptionQuote]:
    return data.option_chain(
        SPY, expiry_from=date(2026, 9, 4), expiry_to=date(2026, 9, 11), right="put"
    )


def run_session(
    data: RecordedMarketData,
    proposer: Proposer,
    journal: Journal,
    *,
    slots: int = 4,
    mcp: McpSession | None = None,
) -> list[CycleRecord]:
    """Run several cycles of one session and journal every one of them.

    The market data is a fixed capture, so each slot sees the same chain; what
    varies is `as_of` and therefore the cycle id. That is deliberate — it makes
    the test about the *identity* of each cycle rather than about the data
    changing underneath it.
    """
    schedule = session_slots(OPEN, CLOSE)[:slots]
    screen = DefaultScreen()
    records: list[CycleRecord] = []
    quotes = chain(data)

    for slot in schedule:
        result = perceive(data, SPY, as_of=slot.at, chain=quotes)
        structures = vertical_credit_spreads(
            quotes, right=Right.PUT, width=Decimal(5), as_of=slot.at
        )
        candidates = build_candidates(
            structures, limits=DEFAULT_LIMITS, equity=EQUITY, as_of=slot.at
        )
        record = run_cycle(
            read=result.read,
            setup=screen.screen(result.read),
            candidates=candidates,
            portfolio=PortfolioSnapshot(
                equity=EQUITY, positions=(), drawdown_pct=Decimal(0), fills_today=0
            ),
            limits=DEFAULT_LIMITS,
            as_of=slot.at,
            mcp=mcp,
            proposer=proposer,
            sequence=slot.sequence,
        )
        journal.append(record)
        records.append(record)
    return records


choices_from = recorded_choices
"""Rebuild the proposer's answers from the journal file. From the bytes.

This is the function that makes the test worth something: if the journal does
not carry enough to reconstruct a `Choice`, the record is narration and the
replay claim is false — which is why it now lives in `alphagate.agent.replay`
and is imported here rather than defined here. A helper only the test can reach
proves the mechanism works and proves nothing about whether the shipped system
can do it, and D6 is a claim about the shipped system.
"""


def orders_of(records: Sequence[CycleRecord]) -> list[tuple[str, str, int, str, str]]:
    """The order set: what was proposed, at what size, how it ended — and the
    `client_order_id` that went to the broker.

    The last field is what makes this an *order* set rather than a summary. The
    id is a fingerprint of the order and the trading day (specs/04 D5), so two
    runs agreeing on it is the strongest available statement that they would
    have placed the same order at the broker, not merely reached the same
    conclusion by coincidence.
    """
    return [
        (
            record.cycle_id,
            str(record.proposal.structure) if record.proposal else "-",
            record.proposal.quantity if record.proposal else 0,
            record.stage.value,
            record.submission.client_order_id if record.submission else "-",
        )
        for record in records
    ]


def filling_broker(count: int = 8) -> RecordedSession:
    """A session that fills every order it is given, from the real payload."""
    return RecordedSession.scripted(**{PLACE_ORDER_TOOL: [FILLED] * count})


class TestReplayThroughTheSubmissionPath:
    """specs/06 D6 and test plan item 7, on a day that actually traded.

    Every other test in this file runs with `mcp=None`, so every cycle stops at
    `DRY_RUN` and the "identical order set" being compared has never contained a
    submitted order. That is a determinism claim about steps 1-6 dressed up as a
    claim about the whole pipeline — the half of the spec sentence that says
    "**or 5 or 6**" was covered and step 7 was not.

    These run the same day twice through a scripted broker.
    """

    def test_a_traded_day_replays_to_an_identical_order_set(
        self, data: RecordedMarketData, tmp_path: Path
    ) -> None:
        live_journal = Journal(directory=tmp_path / "live")
        live = run_session(
            data,
            StubProposer(choice=Choice(1, "second on the menu", 0.6)),
            live_journal,
            mcp=filling_broker(),
        )
        assert {record.stage for record in live} == {Stage.FILLED}, (
            "the fixture must actually trade, or this tests nothing new"
        )

        replay = run_session(
            data,
            replay_proposer(live_journal, DAY),
            Journal(directory=tmp_path / "replay"),
            mcp=filling_broker(),
        )
        assert orders_of(replay) == orders_of(live)

    def test_the_client_order_ids_are_identical_across_the_two_passes(
        self, data: RecordedMarketData, tmp_path: Path
    ) -> None:
        """The strongest form of the claim. specs/04 D5 makes the id a
        fingerprint of the order and the trading day, so two runs agreeing on it
        means they would have placed the same order at the broker — not merely
        reached the same conclusion by two different routes."""
        live_journal = Journal(directory=tmp_path / "live")
        live = run_session(
            data, StubProposer(choice=Choice(1, "x", 0.5)), live_journal, mcp=filling_broker()
        )
        replay = run_session(
            data,
            replay_proposer(live_journal, DAY),
            Journal(directory=tmp_path / "replay"),
            mcp=filling_broker(),
        )
        live_ids = [r.submission.client_order_id for r in live if r.submission]
        replay_ids = [r.submission.client_order_id for r in replay if r.submission]
        assert live_ids == replay_ids

    def test_the_ids_that_went_on_the_wire_are_distinct_and_reproducible(
        self, data: RecordedMarketData, tmp_path: Path
    ) -> None:
        """Distinctness has to be asserted on what was *sent*, not on what the
        record shows.

        `submission_from` prefers the broker's echo of `client_order_id` over the
        one we minted, which is right — the broker's copy is the one that exists
        — but it means a replayed fixture answering four cycles with one captured
        payload makes all four records agree. The ids we generated are on the
        session's call log, and those are the ones specs/04 D5 is about.

        They differ per slot here because the chosen structure differs per slot;
        two cycles proposing the *same* structure on the same day would
        deliberately share an id, which is the idempotency property that stops a
        retry becoming a second position."""
        live_broker = filling_broker()
        journal = Journal(directory=tmp_path / "live")
        run_session(
            data, StubProposer(choice=Choice(1, "x", 0.5)), journal, mcp=live_broker
        )
        sent = [args["client_order_id"] for _, args in live_broker.calls]
        assert len(set(sent)) == len(sent) == 4

        replay_broker = filling_broker()
        run_session(
            data,
            replay_proposer(journal, DAY),
            Journal(directory=tmp_path / "replay"),
            mcp=replay_broker,
        )
        assert [args["client_order_id"] for _, args in replay_broker.calls] == sent

    def test_the_broker_saw_byte_identical_arguments_both_times(
        self, data: RecordedMarketData, tmp_path: Path
    ) -> None:
        """D6 says *byte-identical*. Comparing records compares what we decided;
        comparing the tool arguments compares what would have left the process,
        which is the thing the spec is actually promising."""
        live_journal = Journal(directory=tmp_path / "live")
        live_broker = filling_broker()
        run_session(
            data, StubProposer(choice=Choice(1, "x", 0.5)), live_journal, mcp=live_broker
        )

        replay_broker = filling_broker()
        run_session(
            data,
            replay_proposer(live_journal, DAY),
            Journal(directory=tmp_path / "replay"),
            mcp=replay_broker,
        )
        assert replay_broker.calls == live_broker.calls
        assert live_broker.calls, "the fixture must have called the broker"

    def test_a_rejected_day_replays_as_a_rejected_day(
        self, data: RecordedMarketData, tmp_path: Path
    ) -> None:
        """A refusal from the broker is part of the order set too. specs/04 test
        plan item 8 wants rejections in the journal with their reason; this is
        that reason surviving into a replay."""
        rejected = json.dumps(
            {
                **json.loads(FILLED),
                "data": {
                    **json.loads(FILLED)["data"],
                    "status": "rejected",
                    "reject_reason": "insufficient options buying power",
                },
            }
        )
        broker = RecordedSession.scripted(**{PLACE_ORDER_TOOL: [rejected] * 8})
        journal = Journal(directory=tmp_path / "live")
        live = run_session(
            data, StubProposer(choice=Choice(1, "x", 0.5)), journal, mcp=broker
        )
        assert {record.stage for record in live} == {Stage.REJECTED}

        replay = run_session(
            data,
            replay_proposer(journal, DAY),
            Journal(directory=tmp_path / "replay"),
            mcp=RecordedSession.scripted(**{PLACE_ORDER_TOOL: [rejected] * 8}),
        )
        assert orders_of(replay) == orders_of(live)
        assert journal.read(DAY)[0]["submission"]["reason"] == (
            "insufficient options buying power"
        )

    def test_the_journalled_day_reconciles_against_the_broker(
        self, data: RecordedMarketData, tmp_path: Path
    ) -> None:
        """The last link: a day journalled through submission is a day the
        reconciler can pick up. If `unresolved` could not find these orders, D3's
        amendment path would be unreachable from the live writer."""
        journal = Journal(directory=tmp_path / "live")
        pending = json.dumps(
            {
                **json.loads(FILLED),
                "data": {**json.loads(FILLED)["data"], "status": "new", "legs": []},
            }
        )
        run_session(
            data,
            StubProposer(choice=Choice(1, "x", 0.5)),
            journal,
            mcp=RecordedSession.scripted(**{PLACE_ORDER_TOOL: [pending] * 8}),
        )
        still_open = unresolved(journal.read(DAY))
        assert len(still_open) == 4

        result = reconcile(
            journal,
            DAY,
            RecordedSession.scripted(**{READ_BACK_TOOL: [FILLED] * 4}),
            as_of=CLOSE,
        )
        assert result.resolved == 4
        assert all(line["outcome"]["status"] == "filled" for line in journal.read(DAY))


class TestReplayFromDisk:
    def test_a_journalled_day_replays_to_an_identical_order_set(
        self, data: RecordedMarketData, tmp_path: Path
    ) -> None:
        live_journal = Journal(directory=tmp_path / "live")
        live = run_session(
            data, StubProposer(choice=Choice(1, "second on the menu", 0.6)), live_journal
        )

        replayed_journal = Journal(directory=tmp_path / "replay")
        recovered = choices_from(live_journal, DAY)
        assert len(recovered) == len(live), "every cycle must be recoverable"

        replay = run_session(data, RecordedProposer(recovered), replayed_journal)
        assert orders_of(replay) == orders_of(live)

    def test_the_verdicts_replay_too_not_just_the_orders(
        self, data: RecordedMarketData, tmp_path: Path
    ) -> None:
        """A refusal is part of the order set. Reproducing the trades and not
        the vetoes would mean the journal could not explain a quiet day."""
        journal = Journal(directory=tmp_path / "live")
        live = run_session(data, StubProposer(choice=Choice(1, "second", 0.6)), journal)
        replay = run_session(
            data, RecordedProposer(choices_from(journal, DAY)), Journal(directory=tmp_path / "r")
        )
        for original, again in zip(live, replay, strict=True):
            assert again.verdict == original.verdict
            assert again.veto_reasons == original.veto_reasons

    def test_the_rationale_survives_the_round_trip(
        self, data: RecordedMarketData, tmp_path: Path
    ) -> None:
        """"A rationale without the model id and prompt version behind it is a
        quote with no source" — so all three have to come back."""
        journal = Journal(directory=tmp_path / "live")
        run_session(data, StubProposer(choice=Choice(1, "iv rich, trend up", 0.42)), journal)
        lines = journal.read(DAY)
        assert lines[0]["choice"]["rationale"] == "iv rich, trend up"
        assert lines[0]["choice"]["self_reported_confidence"] == 0.42
        assert lines[0]["call"]["model"] == "stub-model"
        assert lines[0]["call"]["prompt_version"] == "test"

    def test_a_declining_day_replays_as_a_declining_day(
        self, data: RecordedMarketData, tmp_path: Path
    ) -> None:
        """The majority case. specs/05 D1: `DECLINED` entries are the point."""
        journal = Journal(directory=tmp_path / "live")
        live = run_session(data, StubProposer(choice=Choice(None, "no edge", 0.1)), journal)
        assert {record.stage for record in live} == {Stage.DECLINED}

        replay = run_session(
            data, RecordedProposer(choices_from(journal, DAY)), Journal(directory=tmp_path / "r")
        )
        assert orders_of(replay) == orders_of(live)

    def test_cycle_ids_are_stable_across_the_two_passes(
        self, data: RecordedMarketData, tmp_path: Path
    ) -> None:
        """Which is what lets the recorded proposer find the right answer."""
        journal = Journal(directory=tmp_path / "live")
        live = run_session(data, StubProposer(choice=Choice(1, "x", 0.5)), journal)
        expected = [
            cycle_id_for(slot.at, "SPY", slot.sequence)
            for slot in session_slots(OPEN, CLOSE)[: len(live)]
        ]
        assert [record.cycle_id for record in live] == expected

    def test_an_extra_cycle_shortens_the_order_set_visibly(
        self, data: RecordedMarketData, tmp_path: Path
    ) -> None:
        """"Replaying a day that has grown an extra cycle should produce a
        shorter order set, visibly, not a different one quietly." """
        journal = Journal(directory=tmp_path / "live")
        run_session(data, StubProposer(choice=Choice(1, "x", 0.5)), journal, slots=2)

        longer = run_session(
            data,
            RecordedProposer(choices_from(journal, DAY)),
            Journal(directory=tmp_path / "r"),
            slots=4,
        )
        traded = [record for record in longer if record.proposal is not None]
        declined = [record for record in longer if record.stage is Stage.DECLINED]
        assert len(traded) == 2
        assert len(declined) == 2

    def test_replay_proposer_is_the_one_argument_that_differs(
        self, data: RecordedMarketData, tmp_path: Path
    ) -> None:
        """specs/06 D6, as the shipped API rather than as a test helper.

        `replay_proposer(journal, day)` is a drop-in for the live proposer, so
        replaying a day differs from trading it by the value of one argument and
        the value of `as_of`. If that were not true, every replay would go
        through a bespoke path and the determinism claim would be about the
        bespoke path."""
        live_journal = Journal(directory=tmp_path / "live")
        live = run_session(data, StubProposer(choice=Choice(1, "second", 0.6)), live_journal)
        replay = run_session(
            data, replay_proposer(live_journal, DAY), Journal(directory=tmp_path / "r")
        )
        assert orders_of(replay) == orders_of(live)

    def test_a_cycle_that_never_reached_a_model_replays_as_a_decline(
        self, tmp_path: Path
    ) -> None:
        """Absent, not invented. We have no record of the model choosing
        anything, so replaying it as a choice would be manufacturing one."""
        journal = Journal(directory=tmp_path / "live")
        journal.append({"cycle_id": "2026-08-26-SPY-000", "as_of": OPEN.isoformat()})
        assert recorded_choices(journal, DAY) == {}

    def test_the_journal_line_carries_the_whole_menu(
        self, data: RecordedMarketData, tmp_path: Path
    ) -> None:
        """specs/06 D2: "the full menu, not just the pick". Without it the
        rationale is unfalsifiable — you cannot see what was passed over."""
        journal = Journal(directory=tmp_path / "live")
        live = run_session(data, StubProposer(choice=Choice(1, "x", 0.5)), journal, slots=1)
        line = journal.read(DAY)[0]
        assert len(line["candidates"]) == len(live[0].candidates)
        assert len(line["candidates"]) > 1

    def test_every_check_is_journalled_passed_and_failed(
        self, data: RecordedMarketData, tmp_path: Path
    ) -> None:
        """specs/06 D2 again: the dashboard renders near-misses, which is how
        you show a risk system working rather than merely present."""
        journal = Journal(directory=tmp_path / "live")
        run_session(data, StubProposer(choice=Choice(1, "x", 0.5)), journal, slots=1)
        checks = journal.read(DAY)[0]["verdict"]["checks"]
        assert len(checks) == 13
        assert all("observed" in check for check in checks)
        assert any(check["passed"] for check in checks)

    def test_no_credentials_reach_the_journal(
        self, data: RecordedMarketData, tmp_path: Path
    ) -> None:
        """specs/06 D4. The demo video shows this file."""
        journal = Journal(directory=tmp_path / "live")
        run_session(data, StubProposer(choice=Choice(1, "x", 0.5)), journal)
        blob = journal.path_for(DAY).read_text(encoding="utf-8")
        assert "APCA" not in blob
        assert "Bearer" not in blob
        assert "sk-" not in blob
