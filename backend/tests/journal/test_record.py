"""06 D2 and D5 — the record itself. Test plan items 1, 6 and 8.

`test_writer.py` covers the file format against a toy dataclass, which is the
right subject for questions about truncation and redaction. This file asks the
harder question: does the *real* `CycleRecord` survive the round trip with
everything specs/06 D2 says it must carry, at every stage a cycle can stop at?

Those are different failures. A writer that works perfectly and a record that
loses its verdict on the way to disk produce the same green suite unless
something runs the actual type through the actual encoder.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from alphagate.agent import Choice, Stage, cycle_id_for
from alphagate.journal import Journal, outcome_from, untrusted_sources
from tests.journal.conftest import (
    DAY,
    FixedProposer,
    at_stage,
    cycle,
    envelope_json,
    submission,
)


class TestEveryStageRoundTrips:
    """specs/06 test plan item 1."""

    @pytest.mark.parametrize("stage", list(Stage))
    def test_a_record_at_any_stage_writes_and_reads_back(
        self, journal: Journal, stage: Stage
    ) -> None:
        journal.append(at_stage(stage))
        loaded = journal.read(DAY)
        assert len(loaded) == 1
        assert loaded[0]["stage"] == stage.value

    def test_the_parametrisation_covers_the_enum(self) -> None:
        """A new `Stage` that nobody journalled would slip through the test above
        silently, because the parametrisation is generated from the enum. This
        asserts the enum is what we think it is, so adding a stage is a visible
        event rather than a quiet one."""
        assert {stage.value for stage in Stage} == {
            "no_setup",
            "no_candidates",
            "declined",
            "vetoed",
            "submitted",
            "rejected",
            "filled",
            "dry_run",
            "breached",
        }

    def test_the_quiet_stages_are_journalled_like_any_other(
        self, journal: Journal
    ) -> None:
        """specs/06 D2: "Every cycle, not every trade." `NO_SETUP` and
        `DECLINED` are the majority and they are the point."""
        journal.append(cycle(with_setup=False))
        journal.append(
            cycle(proposer=FixedProposer(Choice(None, "no edge here", 0.1)), sequence=1)
        )
        stages = [line["stage"] for line in journal.read(DAY)]
        assert stages == ["no_setup", "declined"]

    def test_a_no_setup_line_still_carries_the_market_read(
        self, journal: Journal
    ) -> None:
        """Otherwise "why didn't it trade at 14:30?" is answered with "it
        didn't", which is the question restated rather than answered."""
        journal.append(cycle(with_setup=False))
        line = journal.read(DAY)[0]
        assert line["read"]["spot"] == "766.00"
        assert line["read"]["iv_rank"] == "62"
        assert line["note"]


class TestTheWholeMenu:
    """specs/06 D2: "The whole menu, not the pick.\""""

    def test_every_candidate_is_written_not_just_the_chosen_one(
        self, journal: Journal
    ) -> None:
        record = cycle()
        journal.append(record)
        line = journal.read(DAY)[0]
        assert len(line["candidates"]) == len(record.candidates) > 1

    def test_each_candidate_carries_its_structure_risk(self, journal: Journal) -> None:
        """"Without them the rationale is unfalsifiable; with them you can see
        what was passed over.\""""
        journal.append(cycle())
        for candidate in journal.read(DAY)[0]["candidates"]:
            risk = candidate["risk"]
            assert Decimal(risk["max_loss"]) > 0
            assert "breakevens" in risk
            assert risk["days_to_expiry"] >= 0


class TestEveryCheck:
    """specs/06 D2: "Every check, passed and failed.\""""

    def test_passed_checks_are_recorded_alongside_the_failed_ones(
        self, journal: Journal
    ) -> None:
        journal.append(cycle())
        checks = journal.read(DAY)[0]["verdict"]["checks"]
        assert any(check["passed"] for check in checks)
        assert all({"observed", "limit", "name"} <= set(check) for check in checks)

    def test_a_near_miss_is_visible_in_the_line(self, journal: Journal) -> None:
        """"The dashboard renders near-misses, which is how you show a risk
        system working rather than merely present.\""""
        journal.append(cycle())
        checks = journal.read(DAY)[0]["verdict"]["checks"]
        measured = [c for c in checks if c["observed"] is not None and c["limit"] is not None]
        assert measured, "at least one check must report a number, not just a verdict"


class TestTheSecurityEnvelopeTravels:
    """specs/06 D5 and test plan item 6. Also specs/04 test plan item 9."""

    def test_the_envelope_survives_write_and_read(self, journal: Journal) -> None:
        journal.append(at_stage(Stage.FILLED))
        envelope = journal.read(DAY)[0]["submission"]["envelope"]
        assert envelope == envelope_json("order_filled")

    def test_the_trust_marker_is_the_one_the_server_sent(self, journal: Journal) -> None:
        journal.append(at_stage(Stage.FILLED))
        envelope = journal.read(DAY)[0]["submission"]["envelope"]
        assert envelope["trust"] == "untrusted_tool_output"
        assert envelope["risk"] == "api_structured"
        assert envelope["tool_name"] == "get_order_by_client_id"

    def test_it_stays_attached_to_the_data_it_described(self, journal: Journal) -> None:
        """specs/06 D5: "The wrapper is kept in the record, attached to the data
        it described." Not hoisted to the top of the line, where it would be a
        claim about the cycle rather than about these bytes."""
        journal.append(at_stage(Stage.FILLED))
        line = journal.read(DAY)[0]
        assert "envelope" not in line
        assert line["submission"]["envelope"]["trust"] == "untrusted_tool_output"

    def test_it_survives_onto_an_outcome_amendment_too(self, journal: Journal) -> None:
        """A fill read back hours later is still bytes from outside. Dropping
        the marker one hop downstream is exactly where the boundary stops being
        one."""
        record = at_stage(Stage.SUBMITTED)
        journal.append(record)
        journal.record_outcome(
            outcome_from(
                submission("order_filled"),
                cycle_id=record.cycle_id,
                observed_at=datetime(2026, 8, 26, 18, 0, tzinfo=UTC),
            )
        )
        assert journal.read(DAY)[0]["outcome"]["envelope"] == envelope_json("order_filled")

    def test_the_journal_can_say_which_bytes_came_from_outside(
        self, journal: Journal
    ) -> None:
        """specs/06 D5's actual claim: "which bytes in this decision came from
        outside the trust boundary?" — answered from the file."""
        journal.append(at_stage(Stage.FILLED))
        sources = untrusted_sources(journal.read(DAY)[0])
        assert {source.kind for source in sources} == {"model", "mcp"}
        assert [source.path for source in sources] == [
            "call.raw_response",
            "submission.raw",
        ]

    def test_a_quiet_cycle_has_no_untrusted_bytes_at_all(self, journal: Journal) -> None:
        """`NO_SETUP` never called a model and never reached the broker. The
        answer is empty, and that is a real answer."""
        journal.append(cycle(with_setup=False))
        assert untrusted_sources(journal.read(DAY)[0]) == ()

    def test_the_raw_response_is_kept_verbatim(self, journal: Journal) -> None:
        """"A paraphrase of a rejection reason is not a rejection reason.\""""
        journal.append(at_stage(Stage.FILLED))
        raw = journal.read(DAY)[0]["submission"]["raw"]
        assert json.loads(raw)["data"]["status"] == "filled"


class TestCycleIds:
    """specs/06 test plan item 8: deterministic and collision-free."""

    def test_it_is_deterministic(self) -> None:
        at = datetime(2026, 8, 26, 14, 30, tzinfo=UTC)
        assert cycle_id_for(at, "SPY", 3) == cycle_id_for(at, "SPY", 3) == "2026-08-26-SPY-003"

    def test_the_time_within_the_day_does_not_change_it(self) -> None:
        """The sequence carries position within the session, not the clock. A
        cycle that runs three minutes late must keep its identity, or a replay
        looks for a choice under an id the journal does not contain."""
        morning = datetime(2026, 8, 26, 13, 50, tzinfo=UTC)
        assert cycle_id_for(morning, "SPY", 0) == cycle_id_for(
            morning + timedelta(minutes=3), "SPY", 0
        )

    def test_underlyings_do_not_collide(self) -> None:
        at = datetime(2026, 8, 26, 14, 30, tzinfo=UTC)
        ids = {cycle_id_for(at, name, 0) for name in ("SPY", "QQQ", "IWM")}
        assert len(ids) == 3

    def test_days_do_not_collide(self) -> None:
        ids = {
            cycle_id_for(datetime(2026, 8, day, 14, 30, tzinfo=UTC), "SPY", 0)
            for day in (26, 27, 28)
        }
        assert len(ids) == 3

    def test_a_full_session_of_ids_is_collision_free(self) -> None:
        """The property that matters: within one day, across the watchlist,
        every scheduled cycle has its own line."""
        at = datetime(2026, 8, 26, 13, 50, tzinfo=UTC)
        ids = [
            cycle_id_for(at, name, sequence)
            for name in ("SPY", "QQQ", "IWM", "TLT")
            for sequence in range(26)
        ]
        assert len(set(ids)) == len(ids)

    def test_it_sorts_by_sequence_not_lexically_by_accident(self) -> None:
        """Zero-padded, so `-010-` sorts after `-009-`. A journal a judge reads
        top to bottom must be in the order the day happened."""
        at = datetime(2026, 8, 26, 13, 50, tzinfo=UTC)
        ids = [cycle_id_for(at, "SPY", n) for n in range(12)]
        assert sorted(ids) == ids

    def test_a_reused_sequence_is_reported_rather_than_collapsed_silently(
        self, journal: Journal
    ) -> None:
        """The failure mode the real 2026-08-26 journal actually has.

        An operator restarting the agent mid-session runs sequence 3 twice. The
        ids are still collision-free by construction — the same inputs give the
        same id, which is the point — but `read` keys on `cycle_id`, so the two
        decisions collapse to one and the first is on disk and not in the day.
        A journal whose first claim is one record per cycle must not lose one
        quietly."""
        journal.append(at_stage(Stage.VETOED, sequence=3))
        journal.append(at_stage(Stage.DRY_RUN, sequence=3))
        journal.append(at_stage(Stage.DRY_RUN, sequence=4))

        assert len(journal.raw_lines(DAY)) == 3
        assert len(journal.read(DAY)) == 2, "the collapse is real"
        assert journal.duplicate_cycles(DAY) == {"2026-08-26-SPY-003": 2}

    def test_a_clean_day_reports_no_duplicates(self, journal: Journal) -> None:
        for sequence in range(4):
            journal.append(at_stage(Stage.DRY_RUN, sequence=sequence))
        assert journal.duplicate_cycles(DAY) == {}

    def test_amendments_are_not_mistaken_for_duplicates(self, journal: Journal) -> None:
        """Two lines with one cycle id are the normal case, not the broken one,
        whenever the second is an amendment."""
        record = at_stage(Stage.SUBMITTED)
        journal.append(record)
        journal.amend(record.cycle_id, outcome={"status": "filled"})
        assert journal.duplicate_cycles(DAY) == {}

    def test_the_id_names_the_day_the_line_is_filed_under(self, journal: Journal) -> None:
        """Which is what lets an amendment with no `as_of` find its own file."""
        record = cycle()
        assert journal.append(record).stem == record.cycle_id[:10]
