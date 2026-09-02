"""06 D1, D3, D4 — the journal. Test plan items 1, 2, 3, 4, 5.

The redaction class is the one that matters outside the codebase. The demo video
shows this file; a leaked key on screen is unrecoverable, and no amount of "we
meant to strip that" fixes it afterwards. So the test runs against a fixture
that deliberately contains credential-shaped strings, and asserts on the bytes
that actually reach disk rather than on the objects that went in.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path

import pytest

from alphagate.agent import Stage
from alphagate.execution import OrderStatus
from alphagate.journal import REDACTED, Journal, encode, outcome_from, redact
from alphagate.journal.writer import _FINAL_STAGE_FOR_STATUS, _LIVE_STAGE
from tests.journal.conftest import at_stage, submission

MCP_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "mcp"

DAY = date(2026, 8, 26)
NOW = datetime(2026, 8, 26, 14, 30, tzinfo=UTC)

OBSERVED_AT = datetime(2026, 8, 26, 18, 0, tzinfo=UTC)
SPY_000 = "2026-08-26-SPY-000"

FAKE_ALPACA_KEY = "PKTESTTESTTESTTESTTEST99"
FAKE_SECRET = "sk-abcdefghijklmnop0123456789"


class Colour(Enum):
    RED = "red"


@dataclass(frozen=True, slots=True)
class Nested:
    price: Decimal
    when: datetime


@dataclass(frozen=True, slots=True)
class Entry:
    cycle_id: str
    as_of: datetime
    stage: Colour
    amount: Decimal
    nested: Nested
    tags: tuple[str, ...]


def entry(cycle_id: str = "2026-08-26-SPY-000") -> Entry:
    return Entry(
        cycle_id=cycle_id,
        as_of=NOW,
        stage=Colour.RED,
        amount=Decimal("0.1"),
        nested=Nested(price=Decimal("752.005"), when=NOW),
        tags=("a", "b"),
    )


@pytest.fixture
def journal(tmp_path: Path) -> Journal:
    return Journal(directory=tmp_path / "journal")


class TestWritingAndReading:
    def test_one_file_per_trading_day(self, journal: Journal) -> None:
        path = journal.append(entry())
        assert path.name == "2026-08-26.jsonl"
        assert path.is_file()

    def test_a_record_round_trips(self, journal: Journal) -> None:
        journal.append(entry())
        loaded = journal.read(DAY)
        assert len(loaded) == 1
        assert loaded[0]["cycle_id"] == "2026-08-26-SPY-000"
        assert loaded[0]["stage"] == "red"

    def test_it_is_one_object_per_line(self, journal: Journal) -> None:
        for index in range(3):
            journal.append(entry(f"2026-08-26-SPY-{index:03d}"))
        lines = journal.path_for(DAY).read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 3
        assert all(json.loads(line) for line in lines)

    def test_records_keep_their_order(self, journal: Journal) -> None:
        for index in range(5):
            journal.append(entry(f"2026-08-26-SPY-{index:03d}"))
        assert [r["cycle_id"][-3:] for r in journal.read(DAY)] == [
            "000",
            "001",
            "002",
            "003",
            "004",
        ]

    def test_reading_an_absent_day_is_empty_not_an_error(self, journal: Journal) -> None:
        assert journal.read(date(2020, 1, 1)) == []


class TestMoneyStaysExact:
    def test_decimals_become_strings_never_floats(self, journal: Journal) -> None:
        """"Round-tripping money through a float is exactly the mistake specs/01
        Rule 3 exists to prevent, and a journal is the last place to make it —
        the record would disagree with the order it records." """
        journal.append(entry())
        raw = journal.path_for(DAY).read_text(encoding="utf-8")
        assert '"amount": "0.1"' in raw
        assert '"amount": 0.1' not in raw

    def test_precision_survives(self, journal: Journal) -> None:
        journal.append(entry())
        loaded = journal.read(DAY)[0]
        assert Decimal(loaded["nested"]["price"]) == Decimal("752.005")

    def test_enums_and_datetimes_encode_readably(self) -> None:
        assert encode(Colour.RED) == "red"
        assert encode(NOW) == "2026-08-26T14:30:00+00:00"
        assert encode(DAY) == "2026-08-26"


class TestTruncatedLines:
    """specs/06 test plan item 2."""

    def test_a_truncated_final_line_is_skipped_not_fatal(self, journal: Journal) -> None:
        journal.append(entry("2026-08-26-SPY-000"))
        journal.append(entry("2026-08-26-SPY-001"))
        path = journal.path_for(DAY)
        path.write_text(path.read_text(encoding="utf-8")[:-25], encoding="utf-8")

        loaded = journal.read(DAY)
        assert len(loaded) == 1, "the intact record survives"
        assert loaded[0]["cycle_id"] == "2026-08-26-SPY-000"

    def test_a_day_with_only_a_broken_line_still_loads(self, journal: Journal) -> None:
        path = journal.path_for(DAY)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"cycle_id": "2026-08', encoding="utf-8")
        assert journal.read(DAY) == []

    def test_blank_lines_are_ignored(self, journal: Journal) -> None:
        journal.append(entry())
        path = journal.path_for(DAY)
        path.write_text(path.read_text(encoding="utf-8") + "\n\n", encoding="utf-8")
        assert len(journal.read(DAY)) == 1


class TestAmendments:
    """specs/06 D3 and test plan items 3 and 4."""

    def test_an_amendment_is_a_new_line_never_an_edit(self, journal: Journal) -> None:
        journal.append(entry())
        original = journal.path_for(DAY).read_text(encoding="utf-8")
        journal.amend("2026-08-26-SPY-000", outcome="filled", realised=Decimal("60.00"))
        after = journal.path_for(DAY).read_text(encoding="utf-8")
        assert after.startswith(original), "the original line is byte-identical"
        assert len(after.strip().split("\n")) == 2

    def test_amendments_apply_by_cycle_id(self, journal: Journal) -> None:
        journal.append(entry())
        journal.amend("2026-08-26-SPY-000", outcome="filled", realised=Decimal("60.00"))
        record = journal.read(DAY)[0]
        assert record["outcome"] == "filled"
        assert record["realised"] == "60.00"
        assert record["stage"] == "red", "the original facts are untouched"

    def test_an_amendment_for_an_unknown_cycle_is_ignored(self, journal: Journal) -> None:
        journal.append(entry())
        journal.amend("2026-08-26-SPY-999", outcome="filled")
        loaded = journal.read(DAY)
        assert len(loaded) == 1
        assert "outcome" not in loaded[0]

    def test_later_amendments_win(self, journal: Journal) -> None:
        journal.append(entry())
        journal.amend("2026-08-26-SPY-000", outcome="submitted")
        journal.amend("2026-08-26-SPY-000", outcome="filled")
        assert journal.read(DAY)[0]["outcome"] == "filled"

    def test_amendments_do_not_reorder_the_day(self, journal: Journal) -> None:
        journal.append(entry("2026-08-26-SPY-000"))
        journal.append(entry("2026-08-26-SPY-001"))
        journal.amend("2026-08-26-SPY-000", outcome="filled")
        assert [r["cycle_id"][-3:] for r in journal.read(DAY)] == ["000", "001"]

    def test_an_amendment_that_arrives_first_still_applies(self, journal: Journal) -> None:
        """specs/06 test plan item 4. File order is the order the facts were
        *learned*, not the order they were caused: a reconciler restarted
        mid-morning can flush a fill before the day it belongs to has been
        replayed, and a reader that dropped it would lose the fill silently."""
        journal.amend("2026-08-26-SPY-000", outcome="filled")
        journal.append(entry())
        assert journal.read(DAY)[0]["outcome"] == "filled"

    def test_out_of_order_amendments_reach_the_same_final_state(
        self, journal: Journal
    ) -> None:
        """The same three facts, written in three orders. One answer."""
        def final(write: object) -> dict[str, object]:
            j = Journal(directory=journal.directory / str(id(write)))
            write(j)  # type: ignore[operator]
            return j.read(DAY)[0]

        def forwards(j: Journal) -> None:
            j.append(entry())
            j.amend("2026-08-26-SPY-000", outcome="submitted")
            j.amend("2026-08-26-SPY-000", outcome="filled", realised=Decimal("60.00"))

        def record_late(j: Journal) -> None:
            j.amend("2026-08-26-SPY-000", outcome="submitted")
            j.append(entry())
            j.amend("2026-08-26-SPY-000", outcome="filled", realised=Decimal("60.00"))

        def record_last(j: Journal) -> None:
            j.amend("2026-08-26-SPY-000", outcome="submitted")
            j.amend("2026-08-26-SPY-000", outcome="filled", realised=Decimal("60.00"))
            j.append(entry())

        states = [final(write) for write in (forwards, record_late, record_last)]
        assert states[0] == states[1] == states[2]
        assert states[0]["outcome"] == "filled"
        assert states[0]["realised"] == "60.00"

    def test_the_later_fact_still_wins_when_the_record_is_late(
        self, journal: Journal
    ) -> None:
        """Amendments are applied in file order among themselves whatever order
        the originals appear in. Otherwise "the same final state" is only
        approximately true, which is the worst kind of true."""
        journal.amend("2026-08-26-SPY-000", outcome="submitted")
        journal.amend("2026-08-26-SPY-000", outcome="filled")
        journal.append(entry())
        assert journal.read(DAY)[0]["outcome"] == "filled"

    def test_amendments_for_two_cycles_interleave_without_crossing(
        self, journal: Journal
    ) -> None:
        journal.append(entry("2026-08-26-SPY-000"))
        journal.amend("2026-08-26-SPY-001", outcome="filled")
        journal.append(entry("2026-08-26-SPY-001"))
        journal.amend("2026-08-26-SPY-000", outcome="vetoed")

        first, second = journal.read(DAY)
        assert first["outcome"] == "vetoed"
        assert second["outcome"] == "filled"

    def test_an_early_amendment_is_still_never_an_edit(self, journal: Journal) -> None:
        """It is held in memory on read, not written into the line on disk."""
        journal.amend("2026-08-26-SPY-000", outcome="filled")
        journal.append(entry())
        lines = journal.path_for(DAY).read_text(encoding="utf-8").strip().splitlines()
        assert "outcome" not in json.loads(lines[1])

    def test_no_hindsight_leaks_into_the_original_line(self, journal: Journal) -> None:
        """specs/01 Rule 4, applied to the record of what we knew."""
        journal.append(entry())
        journal.amend("2026-08-26-SPY-000", realised_pl=Decimal("-120.00"))
        first_line = json.loads(
            journal.path_for(DAY).read_text(encoding="utf-8").split("\n")[0]
        )
        assert "realised_pl" not in first_line


class TestTheFinalStageFollowsTheOrder:
    """specs/06 D3 applied to `stage` — test plan item 3, the half that was missing.

    A cycle is journalled the instant its order is placed, and at that instant
    the broker usually says `pending_new`: `agent.cycle._stage_of` reads that
    and writes `SUBMITTED`. The fill lands seconds or hours later and arrives as
    an amendment. So `stage` has always been *what the broker said, last we
    looked* — and a reader that ignored the later look reported every
    slower-than-instant fill as an order still in flight.

    That is not cosmetic. `agent.book.open_positions` claims broker legs for the
    cycles whose stage is `filled`; left at `submitted`, a real spread became a
    leg "the journal cannot explain", dropped out of the Gate's risk model and
    out of the exit policy that was supposed to close it.

    The rule is deliberately narrow: only a cycle still at `submitted` is
    promoted, and only by a *terminal* status. A live order, a latched
    `BREACHED`, an unknown status — all left exactly as they were.
    """

    def fill_later(self, journal: Journal, *, sequence: int = 0) -> str:
        """One real submitted cycle, filled by a later read-back."""
        record = at_stage(Stage.SUBMITTED, sequence=sequence)
        journal.append(record)
        journal.record_outcome(
            outcome_from(
                submission("order_filled"),
                cycle_id=record.cycle_id,
                observed_at=OBSERVED_AT,
            )
        )
        return record.cycle_id

    def test_a_fill_that_landed_later_reads_as_filled(self, journal: Journal) -> None:
        self.fill_later(journal)
        record = journal.read(DAY)[0]
        assert record["stage"] == "filled"
        assert record["outcome"]["status"] == "filled"

    def test_the_line_on_disk_still_says_submitted(self, journal: Journal) -> None:
        """The promotion happens on read. No hindsight is written backwards into
        the decision line — specs/01 Rule 4, and the whole reason D3 exists."""
        self.fill_later(journal)
        assert journal.raw_lines(DAY)[0]["stage"] == "submitted"

    def test_the_whole_history_read_promotes_too(self, journal: Journal) -> None:
        """`read_through` is what `agent.book.read_book` actually calls: a
        position opened on Monday is claimed from Wednesday's session, or not at
        all."""
        self.fill_later(journal)
        assert journal.read_through(DAY)[0]["stage"] == "filled"

    def test_a_still_live_order_keeps_its_stage(self, journal: Journal) -> None:
        """`new` is not an answer, it is the absence of one. Promoting it would
        claim a fill nobody reported."""
        journal.append(at_stage(Stage.SUBMITTED))
        journal.amend(SPY_000, outcome={"status": "new"})
        assert journal.read(DAY)[0]["stage"] == "submitted"

    def test_an_unrecognised_status_keeps_its_stage(self, journal: Journal) -> None:
        journal.append(at_stage(Stage.SUBMITTED))
        journal.amend(SPY_000, outcome={"status": "sideways"})
        assert journal.read(DAY)[0]["stage"] == "submitted"

    def test_a_rejection_read_back_later_says_rejected(self, journal: Journal) -> None:
        """A broker refusing an order after accepting it is the same fact as
        refusing it at the door, and every view buckets it the same way."""
        journal.append(at_stage(Stage.SUBMITTED))
        journal.amend(SPY_000, outcome={"status": "rejected"})
        assert journal.read(DAY)[0]["stage"] == "rejected"

    def test_a_cancelled_order_is_left_at_submitted(self, journal: Journal) -> None:
        """Deliberate. No `Stage` means "came back without a fill", and inventing
        one here would change the taxonomy every view is built on rather than
        report a fact. `submitted` stays true — it *was* submitted — and
        `outcome.status` is right there for anyone asking what became of it."""
        journal.append(at_stage(Stage.SUBMITTED))
        journal.amend(SPY_000, outcome={"status": "canceled"})
        assert journal.read(DAY)[0]["stage"] == "submitted"

    def test_a_latched_breach_is_never_promoted(self, journal: Journal) -> None:
        """specs/04 D5. `BREACHED` is a kill switch a human clears, and a
        read-back that says `filled` must not clear it on their behalf."""
        journal.append(at_stage(Stage.BREACHED))
        journal.amend(SPY_000, outcome={"status": "filled"})
        assert journal.read(DAY)[0]["stage"] == "breached"

    def test_a_cycle_that_never_reached_the_broker_is_untouched(
        self, journal: Journal
    ) -> None:
        journal.append(at_stage(Stage.VETOED))
        journal.amend(SPY_000, outcome={"status": "filled"})
        assert journal.read(DAY)[0]["stage"] == "vetoed"

    def test_an_amendment_that_arrives_first_still_promotes(
        self, journal: Journal
    ) -> None:
        """The held-amendment path (test plan item 4) reaches the same state."""
        journal.amend(SPY_000, outcome={"status": "filled"})
        journal.append(at_stage(Stage.SUBMITTED))
        assert journal.read(DAY)[0]["stage"] == "filled"

    def test_the_last_amendment_decides(self, journal: Journal) -> None:
        journal.append(at_stage(Stage.SUBMITTED))
        journal.amend(SPY_000, outcome={"status": "new"})
        journal.amend(SPY_000, outcome={"status": "filled"})
        assert journal.read(DAY)[0]["stage"] == "filled"

    def test_an_outcome_that_is_not_a_mapping_is_ignored(self, journal: Journal) -> None:
        """One malformed amendment must not stop the day from loading."""
        journal.append(at_stage(Stage.SUBMITTED))
        journal.amend(SPY_000, outcome="filled")
        assert journal.read(DAY)[0]["stage"] == "submitted"

    def test_an_equity_pass_keeps_its_own_stage(self, journal: Journal) -> None:
        """The equity sleeve journals its outcomes per order rather than at the
        top level, and `EquityStage` is a different vocabulary. Nothing here
        reaches it — `interface/read.py` is where the two sleeves are told
        apart."""
        journal.append(
            {
                "cycle_id": "2026-08-26-EQ-000",
                "as_of": NOW,
                "kind": "equity",
                "stage": "submitted",
                "orders": [{"symbol": "HD", "outcome": "filled"}],
            }
        )
        assert journal.read(DAY)[0]["stage"] == "submitted"

    def test_the_promotion_map_names_real_statuses_and_real_stages(self) -> None:
        """The anti-drift guard.

        `journal` cannot import `agent` — `agent.runner` imports `journal`, so
        that dependency only runs one way — which is why the rule is written in
        the vocabularies' *strings*. This is the test that keeps those strings
        honest: every key is a terminal `OrderStatus`, every value is a real
        `Stage`, the stage promoted *from* is the one the reconciler considers
        live, and the terminal statuses left out are exactly the three with no
        `Stage` of their own.
        """
        for status, stage in _FINAL_STAGE_FOR_STATUS.items():
            assert OrderStatus(status).is_terminal, f"{status} is not terminal"
            assert Stage(stage), f"{stage} is not a Stage"
        assert Stage(_LIVE_STAGE) is Stage.SUBMITTED
        unmapped = {status for status in OrderStatus if status.is_terminal} - {
            OrderStatus(status) for status in _FINAL_STAGE_FOR_STATUS
        }
        assert unmapped == {
            OrderStatus.CANCELED,
            OrderStatus.EXPIRED,
            OrderStatus.REPLACED,
        }, "a terminal status with no Stage of its own must stay unpromoted"


class TestRedaction:
    """specs/06 D4 and test plan item 5. The one that must not fail."""

    def test_the_account_id_never_reaches_the_file(self, journal: Journal) -> None:
        """specs/06 D4 names it explicitly: "or the account id".

        Alpaca returns it as a bare `id`, and an order has a bare `id` that D4
        requires us to *keep*. Both are plain UUIDs, so nothing about the value
        distinguishes them — only the object they sit in does. This is the test
        that was missing: the captured account fixture had been hand-redacted
        before it was committed, so the suite had never seen the real shape and
        the leak was invisible.
        """
        payload = json.loads(
            (MCP_FIXTURES / "account_with_identity.json").read_text(encoding="utf-8")
        )
        journal.append(
            {"cycle_id": "2026-08-26-SPY-000", "as_of": NOW.isoformat(), **payload}
        )
        raw = journal.path_for(DAY).read_text(encoding="utf-8")
        assert "9d8e7f6a-1234-4321-abcd-0123456789ab" not in raw
        assert "PA3ABCDEFG" not in raw

    def test_the_order_id_beside_it_still_survives(self, journal: Journal) -> None:
        """The other half of the same decision. A redaction that took both would
        be safe and useless — you cannot reconcile against it."""
        journal.append(
            {
                "cycle_id": "2026-08-26-SPY-000",
                "as_of": NOW.isoformat(),
                "account": {"id": "9d8e7f6a-1234-4321-abcd-0123456789ab"},
                "submission": {
                    "id": "36048703-4d61-43cd-9914-7cf3163c8f86",
                    "client_order_id": "alphagate-1204108820879f25366cd5b6",
                    "status": "filled",
                },
            }
        )
        line = journal.read(DAY)[0]
        assert line["account"]["id"] == REDACTED
        assert line["submission"]["id"] == "36048703-4d61-43cd-9914-7cf3163c8f86"

    def test_an_account_is_recognised_by_its_shape_not_its_key(
        self, journal: Journal
    ) -> None:
        """It arrives from `unwrap` as the top-level object, with no `account`
        key above it to give it away. Markers are the only signal there is."""
        cleaned = redact(
            {"id": "9d8e7f6a-1234", "buying_power": "100000", "cash": "50000"}
        )
        assert cleaned["id"] == REDACTED
        assert cleaned["buying_power"] == "100000", "balances are not secrets"

    def test_an_account_number_pasted_into_prose_is_stripped(self) -> None:
        """The dangerous appearance is never the one in a field called
        `account_number`. It is this one."""
        assert "PA3ABCDEFG" not in redact("the model mentioned account PA3ABCDEFG")

    def test_a_leg_inside_an_order_keeps_its_identifiers(self, journal: Journal) -> None:
        """Redaction recurses; the account marker must not leak downward into
        objects that merely happen to be nested under one."""
        cleaned = redact(
            {
                "buying_power": "1",
                "id": "acct-uuid",
                "orders": [{"id": "order-uuid", "symbol": "SPY260904P00752000"}],
            }
        )
        assert cleaned["id"] == REDACTED
        assert cleaned["orders"][0]["id"] == "order-uuid"

    def test_secret_keys_are_stripped_by_name(self) -> None:
        assert redact({"api_key": "anything"})["api_key"] == REDACTED
        assert redact({"account_number": "PA123"})["account_number"] == REDACTED
        assert redact({"Authorization": "Bearer x"})["Authorization"] == REDACTED

    def test_credential_shapes_are_stripped_wherever_they_appear(self) -> None:
        """A deny-list of names cannot catch a key pasted into a rationale."""
        text = f"the model said: use {FAKE_ALPACA_KEY} and {FAKE_SECRET}"
        cleaned = redact(text)
        assert FAKE_ALPACA_KEY not in cleaned
        assert FAKE_SECRET not in cleaned

    def test_a_fixture_full_of_credentials_produces_no_matching_bytes(
        self, journal: Journal
    ) -> None:
        journal.append(
            {
                "cycle_id": "2026-08-26-SPY-000",
                "as_of": NOW.isoformat(),
                "api_key": FAKE_ALPACA_KEY,
                "account_number": "PA3ABCDEFG",
                "nested": {"secret": FAKE_SECRET, "note": f"header was Bearer {FAKE_SECRET}"},
                "rationale": f"pasted {FAKE_ALPACA_KEY} by mistake",
            }
        )
        raw = journal.path_for(DAY).read_text(encoding="utf-8")
        assert FAKE_ALPACA_KEY not in raw
        assert FAKE_SECRET not in raw
        assert "PA3ABCDEFG" not in raw

    def test_order_ids_survive_because_reconciliation_needs_them(
        self, journal: Journal
    ) -> None:
        """specs/06 D4 draws the line at account identity. A journal you cannot
        reconcile against is decoration."""
        journal.append(
            {
                "cycle_id": "2026-08-26-SPY-000",
                "as_of": NOW.isoformat(),
                "order_id": "36048703-4d61-43cd-9914-7cf3163c8f86",
                "client_order_id": "alphagate-1204108820879f25366cd5b6",
            }
        )
        raw = journal.path_for(DAY).read_text(encoding="utf-8")
        assert "36048703-4d61-43cd-9914-7cf3163c8f86" in raw
        assert "alphagate-1204108820879f25366cd5b6" in raw

    def test_redaction_reaches_inside_lists(self) -> None:
        cleaned = redact({"calls": [{"api_key": "x"}, {"fine": "y"}]})
        assert cleaned["calls"][0]["api_key"] == REDACTED
        assert cleaned["calls"][1]["fine"] == "y"

    def test_it_is_applied_on_the_encoded_form(self, journal: Journal) -> None:
        """So it sees the strings the file will contain, not the objects."""
        journal.append(
            {"cycle_id": "2026-08-26-SPY-000", "as_of": NOW.isoformat(), "secret": Decimal("1")}
        )
        assert journal.read(DAY)[0]["secret"] == REDACTED


class TestDayResolution:
    def test_the_day_comes_from_as_of(self, journal: Journal) -> None:
        assert journal.append(entry()).name == "2026-08-26.jsonl"

    def test_it_falls_back_to_the_cycle_id(self, journal: Journal) -> None:
        path = journal.append({"cycle_id": "2026-09-01-QQQ-004"})
        assert path.name == "2026-09-01.jsonl"

    def test_an_unattributable_record_is_refused(self, journal: Journal) -> None:
        """Rather than being filed under today, which is a clock read and a lie."""
        with pytest.raises(ValueError, match="cannot determine the trading day"):
            journal.append({"note": "no identity"})

    def test_an_explicit_day_wins(self, journal: Journal) -> None:
        assert journal.append({"note": "x"}, day=DAY).name == "2026-08-26.jsonl"
