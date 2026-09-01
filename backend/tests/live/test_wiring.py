"""The composition root — `live/wiring.py`.

This is the only package that knows a live account exists, so it is the only one
whose tests could accidentally open a socket. They do not: `RecordedSession` and
`RecordedMarketData` stand in for both ends, which is the whole reason those
seams exist (adr/0002 D4).

What is worth testing here is not the assembly — that is obvious code — but the
four decisions the module actually makes, each of which is silent when wrong:

* the paper guard runs *before* a transport exists;
* the high-water mark rises and never falls;
* the watchlist rotates rather than concentrating on one name;
* the chain request speaks Alpaca's vocabulary, not the domain's.

That last one is here because it shipped broken. `Right.PUT.value` is `"P"` and
the snapshot endpoint answers `400` to it, which would have been a silent
`NO_CANDIDATES` for an entire trading day. It was found by running the thing,
not by reading it, and this is the test that would have found it first.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from alphagate.agent import Slot, Underlying
from alphagate.agent.option_book import EntryRule, OptionRule
from alphagate.agent.schedule import CycleKind
from alphagate.agent.screen import BookScreen, DefaultScreen
from alphagate.core.errors import InvariantViolation
from alphagate.core.identifiers import ticker
from alphagate.execution import ExecutionError
from alphagate.execution.session import ToolResult
from alphagate.interface.status import STALE_AFTER, _age_of, read_status
from alphagate.journal import Journal
from alphagate.live.wiring import (
    DTE_MAX,
    DTE_MIN,
    LiveContext,
    SessionState,
    _api_right,
    expiry_window,
    gather_for,
    load_pinned_option_book,
    market_session,
    mcp_session,
    option_book_window,
    publish_startup_status,
    right_for_structure,
    screen_for,
)
from alphagate.marketdata import RecordedMarketData
from alphagate.options import Right
from alphagate.risk import DEFAULT_LIMITS

NOW = datetime(2026, 8, 26, 14, 30, tzinfo=UTC)

PAPER = {
    "ALPACA_API_KEY_ID": "PKTESTTESTTESTTESTTEST99",
    "ALPACA_API_SECRET_KEY": "secret",
    "ALPACA_TRADING_URL": "https://paper-api.alpaca.markets",
}
LIVE = {
    "ALPACA_API_KEY_ID": "AKLIVELIVELIVELIVELIVE99",
    "ALPACA_API_SECRET_KEY": "secret",
    "ALPACA_TRADING_URL": "https://api.alpaca.markets",
}


class TestThePaperGuardRunsFirst:
    """The one rule this project cannot negotiate."""

    def test_a_live_key_never_gets_a_transport(self) -> None:
        """It raises before `StdioSession` is constructed, so a transport that
        could reach real money does not come into existence."""
        with pytest.raises(ExecutionError):
            mcp_session(LIVE)

    def test_a_live_trading_url_is_refused_even_with_a_paper_key(self) -> None:
        """Two independent signals, because either alone can be edited by
        accident."""
        mixed = {**PAPER, "ALPACA_TRADING_URL": "https://api.alpaca.markets"}
        with pytest.raises(ExecutionError):
            mcp_session(mixed)


class TestThePeakEquityPersists:
    """specs/03 D4. A mark that resets at midnight is a kill switch that cannot
    latch across the days it exists for."""

    def test_it_rises(self, tmp_path: Path) -> None:
        state = SessionState(path=tmp_path / "state.json")
        state.observe(Decimal("100000"))
        state.observe(Decimal("101000"))
        assert state.peak_equity == Decimal("101000")

    def test_it_never_falls(self, tmp_path: Path) -> None:
        state = SessionState(path=tmp_path / "state.json")
        state.observe(Decimal("101000"))
        state.observe(Decimal("95000"))
        assert state.peak_equity == Decimal("101000"), "that is the whole job"

    def test_it_survives_a_restart(self, tmp_path: Path) -> None:
        path = tmp_path / "state.json"
        SessionState(path=path).observe(Decimal("101726.48"))
        assert SessionState.load(path).peak_equity == Decimal("101726.48")

    def test_money_stays_decimal_across_the_file(self, tmp_path: Path) -> None:
        path = tmp_path / "state.json"
        SessionState(path=path).observe(Decimal("100000.01"))
        assert isinstance(SessionState.load(path).peak_equity, Decimal)
        assert json.loads(path.read_text(encoding="utf-8"))["peak_equity"] == "100000.01"

    def test_the_latch_survives_too(self, tmp_path: Path) -> None:
        path = tmp_path / "state.json"
        state = SessionState(path=path, killswitch_tripped=True)
        state.save()
        assert SessionState.load(path).killswitch_tripped

    def test_a_missing_file_is_no_history_not_a_crash(self, tmp_path: Path) -> None:
        assert SessionState.load(tmp_path / "absent.json").peak_equity is None

    def test_a_corrupt_file_is_no_history_not_a_crash(self, tmp_path: Path) -> None:
        """Losing the high-water mark is bad; refusing to start the trading day
        because a JSON file has a stray brace is worse."""
        path = tmp_path / "state.json"
        path.write_text("{not json", encoding="utf-8")
        assert SessionState.load(path).peak_equity is None


class TestTheUniverseRotates:
    def test_each_slot_takes_the_next_name(self) -> None:
        context = _context(("SPY", "QQQ", "IWM"))
        picked = [str(context.underlying_for(_slot(n)).symbol) for n in range(6)]
        assert picked == ["SPY", "QQQ", "IWM", "SPY", "QQQ", "IWM"]

    def test_a_single_name_universe_does_not_divide_by_zero(self) -> None:
        context = _context(("SPY",))
        assert str(context.underlying_for(_slot(7)).symbol) == "SPY"

    def test_it_is_deterministic(self) -> None:
        """Same slot, same name, so a replayed day picks what the live one did."""
        context = _context(("SPY", "QQQ", "IWM"))
        assert context.underlying_for(_slot(4)) == context.underlying_for(_slot(4))


class TestTheChainRequest:
    def test_the_api_wants_the_word_not_the_occ_letter(self) -> None:
        """`Right.PUT.value` is `"P"` because that is what an option symbol is
        built from; Alpaca's snapshot endpoint answers 400 to it. Two
        vocabularies for one fact, translated at the boundary that needs it."""
        assert _api_right(Right.PUT) == "put"
        assert _api_right(Right.CALL) == "call"

    def test_the_window_is_wider_than_the_gate_will_accept(self) -> None:
        """The chain request should return everything the Gate might accept and
        let the Gate refuse. A narrower request hides candidates from the menu
        and calls it a risk decision."""
        low, high = expiry_window(NOW)
        assert (low - NOW.date()).days == DTE_MIN
        assert (high - NOW.date()).days == DTE_MAX

    def test_the_window_moves_with_the_day(self) -> None:
        assert expiry_window(NOW)[0] < expiry_window(
            datetime(2026, 8, 27, 14, 30, tzinfo=UTC)
        )[0]


class TestTheSessionFallback:
    def test_regular_hours_are_tz_aware_utc(self) -> None:
        open_at, close_at = market_session(NOW)
        assert open_at.tzinfo is not None
        assert (open_at.hour, open_at.minute) == (13, 30)
        assert (close_at.hour, close_at.minute) == (20, 0)

    def test_it_is_todays_session(self) -> None:
        assert market_session(NOW)[0].date() == date(2026, 8, 26)


def _slot(sequence: int) -> Slot:
    return Slot(at=NOW, kind=CycleKind.FULL, sequence=sequence)


def _context(symbols: tuple[str, ...]) -> LiveContext:
    from alphagate.agent import IvHistoryStore
    from alphagate.journal import Journal
    from alphagate.marketdata import RecordedMarketData

    return LiveContext(
        data=RecordedMarketData(directory=Path(".")),
        mcp=None,
        journal=Journal(directory=Path(".")),
        iv=IvHistoryStore(directory=Path(".")),
        state=SessionState(path=Path("state.json")),
        universe=tuple(
            Underlying(ticker(symbol), Decimal(5), "test") for symbol in symbols
        ),
    )


class TestItRefusesToGuessWithoutABroker:
    def test_reading_the_book_needs_a_session(self) -> None:
        """A dry run has no account to read, and inventing one would put a made
        up equity into every budgeted limit."""
        with pytest.raises(RuntimeError, match="no MCP session"):
            _context(("SPY",)).book(as_of=NOW)


def _option_rule(structure: str = "put_credit_spread") -> OptionRule:
    return OptionRule(
        underlying=ticker("SPY"),
        structure=structure,
        entry=EntryRule(expression="iv_rank() < 15", clauses=(("iv_rank", "<", 15.0),)),
        dte_target=14,
        dte_tolerance=10,
        anchor_delta=0.16,
        anchor_tolerance=0.06,
        width_delta=0.08,
        min_sessions_between_entries=1,
        risk_per_trade=Decimal("0.02"),
        max_concurrent=3,
    )


class TestRightForStructure:
    """specs/07 D5: which side of the chain a rule's structure trades."""

    def test_put_credit_spread_trades_puts(self) -> None:
        assert right_for_structure("put_credit_spread") is Right.PUT

    def test_call_credit_spread_trades_calls(self) -> None:
        assert right_for_structure("call_credit_spread") is Right.CALL

    def test_an_iron_condor_is_refused(self) -> None:
        """`option_book.py`'s loader admits `iron_condor` as defined risk on its
        own terms, but nothing in `agent/candidates.py` builds one —
        `spreads_by_delta` resolves a single vertical. A book naming it must
        not silently trade half of it."""
        with pytest.raises(InvariantViolation, match="no vertical-spread executor"):
            right_for_structure("iron_condor")


class TestOptionBookWindow:
    """specs/07 D5 meets specs/03 D5: the book's window, capped by the Gate's."""

    def test_a_narrower_book_window_passes_through(self) -> None:
        narrow = replace(_option_rule(), dte_target=10, dte_tolerance=2)
        assert narrow.dte_window() == (8, 12)
        assert option_book_window(narrow, DEFAULT_LIMITS) == (8, 12)

    def test_a_book_window_wider_than_the_gate_is_clipped(self) -> None:
        """This book's own window is (4, 24) and `SLEEVE_LIMITS.dte_range` is
        (3, 21) — the 24 must never be requested, or a candidate built at that
        expiry would be vetoed by the Gate on every cycle."""
        rule = _option_rule()
        assert rule.dte_window() == (4, 24)
        assert option_book_window(rule, DEFAULT_LIMITS) == (4, 21)

    def test_the_gates_own_range_is_never_widened(self) -> None:
        """The book being wider must narrow the request, never the other way:
        this function must not be usable to loosen the Gate's own limit."""
        tight = replace(DEFAULT_LIMITS, dte_range=(5, 10))
        assert option_book_window(_option_rule(), tight) == (5, 10)


class TestScreenFor:
    def test_a_context_with_a_book_gets_a_book_screen(self) -> None:
        context = _context(("SPY",))
        context.option_book = _fake_book()
        screen = screen_for(context)
        assert isinstance(screen, BookScreen)
        assert screen.rule is context.option_book.rule

    def test_a_context_with_no_book_falls_back_to_the_default(self) -> None:
        assert isinstance(screen_for(_context(("SPY",))), DefaultScreen)


def _fake_book() -> Any:
    """A minimal stand-in carrying only the `.rule` `screen_for` reads.

    Building a real `OptionBook` needs the full loader's provenance and sealed
    fields, which `test_option_book.py` already exercises end to end; this
    file only needs something with the one attribute `screen_for` touches.
    """
    return SimpleNamespace(rule=_option_rule())


class TestLoadPinnedOptionBook:
    """specs/07 D1: no book, no orders — and the reason is a value, not a
    traceback, so every caller can report it in its own voice."""

    def test_an_empty_directory_is_a_reason_not_a_book(self, tmp_path: Path) -> None:
        book, reason = load_pinned_option_book(tmp_path, "cc197008e0deb097")
        assert book is None
        assert "no option book" in reason

    def test_a_book_for_another_fingerprint_is_reported_by_name(self, tmp_path: Path) -> None:
        """The pin is invisible from the inside: a book on disk for a
        different strategy is not refused, it is never looked at by
        `find_latest_book`'s glob — so the reason must say what *is* there."""
        (tmp_path / "some_other_rule-deadbeefcafe0000-2024-08-30.json").write_text(
            json.dumps({"schema_version": 1, "spec_fingerprint": "deadbeefcafe0000",
                        "spec_name": "some_other_rule", "as_of": "2024-08-30"}),
            encoding="utf-8",
        )
        book, reason = load_pinned_option_book(tmp_path, "cc197008e0deb097")
        assert book is None
        assert "some_other_rule" in reason
        assert "ALPHAGATE_OPTION_FINGERPRINT" in reason

    def test_a_malformed_file_is_a_reason_not_a_crash(self, tmp_path: Path) -> None:
        name = "iv_rank_low_sticky_put_credit_spread_v1-cc197008e0deb097-2024-08-30.json"
        (tmp_path / name).write_text("{not json", encoding="utf-8")
        book, reason = load_pinned_option_book(tmp_path, "cc197008e0deb097")
        assert book is None
        assert reason

    def test_the_real_committed_book_loads(self) -> None:
        """The artefact this whole feature is built to execute."""
        repo_root = Path(__file__).resolve().parents[3]
        directory = repo_root / "ai_quant_researcher" / "runs" / "option_books"
        if not directory.is_dir():
            pytest.skip("no option book checked out at this path")
        book, reason = load_pinned_option_book(directory, "cc197008e0deb097")
        assert book is not None, reason
        assert book.rule.structure == "put_credit_spread"
        assert book.digest, "the digest must be computed, not left blank"


# --------------------------------------------------------------------------- #
# `once`'s timing bug: gather must judge freshness against when quotes were
# actually fetched, not against a slot that has not happened yet.
# --------------------------------------------------------------------------- #


class FakeMcp:
    """Enough of `McpSession` to answer `LiveContext.book()` -- one account,
    no open positions, nothing else. What the book actually contains is
    irrelevant here; this suite is about quote freshness, not account state.
    """

    def call(self, tool: str, arguments: Mapping[str, object]) -> ToolResult:
        del arguments
        if tool == "get_account_info":
            return ToolResult(
                tool=tool,
                data={"equity": "500000", "cash": "500000", "options_trading_level": 3},
                envelope=None,
                raw="{}",
            )
        if tool == "get_all_positions":
            return ToolResult(tool=tool, data={"result": []}, envelope=None, raw="{}")
        raise AssertionError(f"unexpected tool call in this test: {tool}")  # pragma: no cover


class TestGatherJudgesFreshnessAsOfTheSlotItIsGiven:
    """The live bug `_adhoc_slot` (`live/cli.py`) fixes, pinned at the level
    `gather_for`'s closure actually sees it: the same fetched chain produces a
    real menu when judged as of when it was fetched, and nothing at all when
    judged 85 seconds later -- the exact gap a live run measured between `now`
    and the next scheduled slot, which `once` used to hand `gather` before the
    fix. A test that only asserted "N candidates" for one clock reading would
    pass for the wrong reason on a quiet market; this asserts the *same* chain
    behaves differently under the *two* clock readings the bug and the fix
    actually produce.
    """

    FRESH_AS_OF = datetime(2026, 8, 26, 18, 12, 11, tzinfo=UTC)
    """Inside 60s (`SLEEVE_LIMITS.max_quote_age`) of every quote in the
    `chain_SPY.json` fixture -- its oldest quote is stamped 18:11:11.9Z."""

    def _context(self, tmp_path: Path) -> LiveContext:
        from alphagate.agent import IvHistoryStore

        fixtures = Path(__file__).resolve().parents[1] / "fixtures" / "marketdata"
        return LiveContext(
            data=RecordedMarketData(directory=fixtures),
            mcp=FakeMcp(),
            journal=Journal(directory=tmp_path / "journal"),
            iv=IvHistoryStore(directory=tmp_path / "iv"),
            state=SessionState(path=tmp_path / "state.json"),
        )

    def _slot(self, at: datetime) -> Slot:
        return Slot(at=at, kind=CycleKind.FULL, sequence=0)

    def test_the_same_chain_survives_freshness_only_as_of_when_it_was_fetched(
        self, tmp_path: Path
    ) -> None:
        context = self._context(tmp_path)
        gather = gather_for(context, slots=())

        fresh = gather(self._slot(self.FRESH_AS_OF))
        assert len(fresh.candidates) > 0, "the fixture must genuinely have a menu"

        # The exact gap a live run measured: `now` to the next scheduled slot.
        judged_as_a_future_slot_would_be = self._slot(
            self.FRESH_AS_OF + timedelta(seconds=85)
        )
        stale = gather(judged_as_a_future_slot_would_be)
        assert stale.candidates == (), (
            "the same chain, judged 85s later, must fail freshness -- this is "
            "exactly what handing `gather` a future slot's timestamp used to do"
        )


# --------------------------------------------------------------------------- #
# `run` reporting "not running" for up to a full slot interval after it
# actually starts, because nothing was published before the first slot fired.
# --------------------------------------------------------------------------- #


class TestPublishStartupStatus:
    """`publish_startup_status` is what makes the dashboard say *running* the
    moment a session starts, rather than up to `CYCLE_INTERVAL` later. See its
    own docstring in `wiring.py` for the live symptom this closes.
    """

    def _context(self, tmp_path: Path, *, mcp: Any = None) -> LiveContext:
        from alphagate.agent import IvHistoryStore

        fixtures = Path(__file__).resolve().parents[1] / "fixtures" / "marketdata"
        return LiveContext(
            data=RecordedMarketData(directory=fixtures),
            mcp=mcp if mcp is not None else FakeMcp(),
            journal=Journal(directory=tmp_path / "journal"),
            iv=IvHistoryStore(directory=tmp_path / "iv"),
            state=SessionState(path=tmp_path / "state.json"),
        )

    def test_a_snapshot_exists_before_any_slot_runs_and_reads_as_running(
        self, tmp_path: Path
    ) -> None:
        """The behaviour that matters: a status file exists immediately, its
        `as_of` is the real start time, and the *dashboard's own* staleness
        rule (`STALE_AFTER`, `_age_of`) -- not a duplicated threshold in this
        test -- calls it running."""
        context = self._context(tmp_path)
        now = datetime.now(UTC)
        upcoming = (
            Slot(at=now + timedelta(minutes=5), kind=CycleKind.FULL, sequence=0),
            Slot(at=now + timedelta(minutes=20), kind=CycleKind.FULL, sequence=1),
        )

        publish_startup_status(context, as_of=now, slots=upcoming)

        snapshot = read_status(tmp_path / "journal")
        assert snapshot is not None, "no status was published at session start"
        assert snapshot["as_of"] == now.isoformat()
        age = _age_of(snapshot["as_of"])
        assert age is not None
        assert age < STALE_AFTER, "the dashboard's own rule must call this running"
        assert snapshot["next_slot"] == upcoming[0].at.isoformat()

    def test_it_writes_no_journal_record(self, tmp_path: Path) -> None:
        """No screen, no proposer, no Gate ran here -- there is nothing to
        journal and no cycle sequence was spent (specs/06)."""
        context = self._context(tmp_path)
        publish_startup_status(context, as_of=datetime.now(UTC), slots=())
        assert list((tmp_path / "journal").glob("*.jsonl")) == []

    def test_the_note_reads_as_a_heartbeat_not_a_completed_cycle(
        self, tmp_path: Path
    ) -> None:
        context = self._context(tmp_path)
        publish_startup_status(context, as_of=datetime.now(UTC), slots=())
        snapshot = read_status(tmp_path / "journal")
        assert snapshot is not None
        note = str(snapshot["note"]).lower()
        assert "starting" in note
        assert "candidates" not in note, "must not read like a completed cycle"

    def test_a_publish_failure_is_printed_not_swallowed(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Unlike the per-slot `_publish_status`, a startup failure must reach
        the operator's console: silently not appearing is the exact bug being
        fixed, and a supervised restart that failed to publish and said
        nothing would reproduce it."""

        class BrokenMcp:
            def call(self, tool: str, arguments: Mapping[str, object]) -> ToolResult:
                del tool, arguments
                raise RuntimeError("transport is gone")

        context = self._context(tmp_path, mcp=BrokenMcp())
        publish_startup_status(context, as_of=datetime.now(UTC), slots=())
        assert "could not publish" in capsys.readouterr().out
        assert read_status(tmp_path / "journal") is None
