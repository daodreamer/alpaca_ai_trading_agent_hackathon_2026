"""The equity composition root, offline — specs/09 D8, D9; test plan item 12.

The whole chain from a target-book file to a placed order, against a
`RecordedSession` and a stub `MarketData`. Nothing here opens a socket.

This is the suite that covers the parts no unit test can: that the book found on
disk is the one submitted, that the portfolio snapshot is advanced between
orders so the daily caps bind *within* a pass, that a transport failure stops
the pass instead of skipping an order, and that a pass which decides nothing is
still journalled.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest

from alphagate.core.identifiers import Ticker, ticker
from alphagate.equity import (
    DEFAULT_EQUITY_POLICY,
    EQUITY_SLEEVE_ALLOCATION,
    UnusableBook,
)
from alphagate.execution import RecordedSession, TransportFailure
from alphagate.execution.account import read_account
from alphagate.execution.equity import PLACE_STOCK_ORDER_TOOL
from alphagate.journal import Journal
from alphagate.live.equity import (
    BOOK_ARCHIVE,
    DECIDED_STAGES,
    EquityContext,
    EquityStage,
    already_decided,
    archive_book,
    changed_pin_advice,
    cycle_id_for,
    digest_of,
    find_latest_book,
    marks_from,
    read_book,
    run_equity_cycle,
    today_totals,
    unpinned_books,
)
from alphagate.live.equity_cli import _heartbeat
from alphagate.live.wiring import EQUITY_SLEEVE_BASIS, SessionState
from alphagate.marketdata import StockSnapshot
from tests.equity.conftest import AAA, BBB, CCC, FINGERPRINT

NOW = datetime(2026, 8, 28, 13, 45, tzinfo=UTC)
POSITIONS_TOOL = "get_all_positions"
ACCOUNT_TOOL = "get_account_info"
ASSET_TOOL = "get_asset"


# --------------------------------------------------------------------- #
# Stubs
# --------------------------------------------------------------------- #


class StubMarketData:
    """A `MarketData` that answers `stock_snapshots` and refuses everything else.

    Deliberately not `RecordedMarketData`: this suite is about the composition
    root, and a fixture directory of captured payloads would make every test
    here also a test of the replay adapter. What it *does* share is the
    `StockSnapshot` type, so a change to that shape fails here too.
    """

    def __init__(self, prices: Mapping[Ticker, Decimal], *, age_seconds: float = 1.0) -> None:
        self.prices = dict(prices)
        self.age_seconds = age_seconds
        self.asked: list[tuple[str, ...]] = []

    def stock_snapshots(
        self, symbols: Sequence[Ticker]
    ) -> Mapping[Ticker, StockSnapshot]:
        self.asked.append(tuple(str(s) for s in symbols))
        stamp = datetime.fromtimestamp(
            NOW.timestamp() - self.age_seconds, tz=UTC
        )
        return {
            symbol: StockSnapshot(symbol, price, stamp, from_quote=True)
            for symbol, price in self.prices.items()
            if symbol in set(symbols)
        }

    def __getattr__(self, name: str) -> Any:  # pragma: no cover - guard, not behaviour
        raise AssertionError(f"the equity path must not call {name}()")


def account_payload(equity: str = "100000", cash: str = "100000") -> str:
    return json.dumps(
        {
            "equity": equity,
            "last_equity": equity,
            "cash": cash,
            "buying_power": cash,
            "options_buying_power": "0",
            "options_trading_level": 3,
            "multiplier": "2",
            "account_blocked": False,
        }
    )


def positions_payload(rows: list[dict[str, Any]]) -> str:
    return json.dumps({"result": rows})


def asset_payload(symbol: str, *, fractionable: bool = True) -> str:
    return json.dumps({"symbol": symbol, "tradable": True, "fractionable": fractionable})


def order_payload(status: str = "filled") -> str:
    return json.dumps(
        {"id": "00000000-0000-4000-8000-000000000001", "status": status, "filled_qty": "1"}
    )


def session(
    *,
    equity: str = "100000",
    cash: str = "100000",
    positions: list[dict[str, Any]] | None = None,
    order: Any = None,
) -> RecordedSession:
    return RecordedSession.scripted(
        **{
            ACCOUNT_TOOL: account_payload(equity, cash),
            POSITIONS_TOOL: positions_payload(positions or []),
            ASSET_TOOL: asset_payload("X"),
            PLACE_STOCK_ORDER_TOOL: order if order is not None else order_payload(),
        }
    )


@pytest.fixture
def books(tmp_path: Path, book_payload: dict[str, Any]) -> Path:
    """A directory holding one valid artefact, named the way `aqr` names them."""
    directory = tmp_path / "target_books"
    directory.mkdir()
    path = directory / f"rs_test-{FINGERPRINT}-2026-08-27.json"
    path.write_text(json.dumps(book_payload, indent=2), encoding="utf-8")
    return directory


@pytest.fixture
def context(tmp_path: Path, books: Path) -> EquityContext:
    return EquityContext(
        data=StubMarketData(
            {AAA: Decimal(100), BBB: Decimal(50), CCC: Decimal(25)}
        ),  # type: ignore[arg-type]
        mcp=session(),
        journal=Journal(directory=tmp_path / "journal"),
        books=books,
        pinned_fingerprint=FINGERPRINT,
        policy=DEFAULT_EQUITY_POLICY,
    )


# --------------------------------------------------------------------- #
# Finding and reading the book
# --------------------------------------------------------------------- #


def test_the_latest_book_is_chosen_by_session_not_by_mtime(books: Path) -> None:
    """`aqr` regenerates old dates on demand, so mtime prefers the wrong file.

    The filename carries the session, and the session is what "latest" means.
    """
    older = books / f"rs_test-{FINGERPRINT}-2026-08-20.json"
    older.write_text("{}", encoding="utf-8")
    # Written second, so its mtime is newest — and it is still not the latest.
    found = find_latest_book(books, FINGERPRINT)
    assert found is not None
    assert found.name.endswith("2026-08-27.json")


def test_a_book_for_another_fingerprint_is_not_even_found(books: Path) -> None:
    assert find_latest_book(books, "0" * 16) is None


def test_a_missing_directory_is_not_an_error(tmp_path: Path) -> None:
    """The researcher may not have run yet. That is a state, not a crash."""
    assert find_latest_book(tmp_path / "nothing", FINGERPRINT) is None


def test_the_digest_is_of_the_file_not_of_a_reserialisation(books: Path) -> None:
    """Hashing the parsed mapping would hash this process's JSON settings.

    Two readers with different `indent` would then disagree about whether the
    same file had been executed.

    Compared against `read_bytes` on purpose. The first version hashed
    `read_text`, which translates CRLF to LF on Windows — so the digest was of a
    normalised copy and the archive, written back through `write_text`, matched
    neither.
    """
    from hashlib import sha256

    path = find_latest_book(books, FINGERPRINT)
    assert path is not None
    book, raw = read_book(path, pinned_fingerprint=FINGERPRINT)
    assert book.digest == sha256(path.read_bytes()).hexdigest()
    assert raw.encode("utf-8") == path.read_bytes()


def test_the_archive_copy_is_byte_identical(books: Path, tmp_path: Path) -> None:
    """`aqr` overwrites its output directory; the journal's copy is the record.

    A journal line pointing at a path whose contents have since changed is a
    record of nothing (specs/09 D9).
    """
    path = find_latest_book(books, FINGERPRINT)
    assert path is not None
    book, raw = read_book(path, pinned_fingerprint=FINGERPRINT)
    copied = archive_book(raw, book, directory=tmp_path / "journal")
    assert copied.read_bytes() == path.read_bytes()
    assert copied.parent.name == BOOK_ARCHIVE


def test_a_book_for_the_wrong_strategy_is_refused_on_read(books: Path) -> None:
    path = find_latest_book(books, FINGERPRINT)
    assert path is not None
    with pytest.raises(UnusableBook, match="pinned strategy"):
        read_book(path, pinned_fingerprint="0" * 16)


# --------------------------------------------------------------------- #
# Marks
# --------------------------------------------------------------------- #


def test_a_symbol_with_a_price_but_no_asset_record_is_not_tradeable() -> None:
    """"We could not find out" is not "yes".

    Defaulting the other way would send an order to find out, which is a strange
    way to ask.
    """
    stamp = datetime.fromtimestamp(NOW.timestamp() - 5, tz=UTC)
    marks = marks_from(
        {AAA: StockSnapshot(AAA, Decimal(100), stamp, from_quote=True)},
        {},
        as_of=NOW,
    )
    assert marks[AAA].tradeable is False
    assert marks[AAA].fractionable is False


def test_the_age_is_computed_from_the_argument_not_a_clock() -> None:
    """Which is what makes a replayed pass produce the same skips as the live one."""
    stamp = datetime.fromtimestamp(NOW.timestamp() - 42, tz=UTC)
    marks = marks_from(
        {AAA: StockSnapshot(AAA, Decimal(100), stamp, from_quote=True)}, {}, as_of=NOW
    )
    assert marks[AAA].age_seconds == pytest.approx(42.0)


def test_a_future_stamped_quote_reports_zero_age_rather_than_negative() -> None:
    """Clock skew between the broker and this process is real and small.

    A negative age would pass every freshness check by being absurdly fresh,
    which is the wrong direction to fail in.
    """
    ahead = datetime.fromtimestamp(NOW.timestamp() + 10, tz=UTC)
    marks = marks_from(
        {AAA: StockSnapshot(AAA, Decimal(100), ahead, from_quote=True)}, {}, as_of=NOW
    )
    assert marks[AAA].age_seconds == 0.0


# --------------------------------------------------------------------- #
# The pass
# --------------------------------------------------------------------- #


def test_a_pass_with_no_book_is_still_journalled(tmp_path: Path) -> None:
    """"Why did it not trade" has an answer on disk even when the answer is
    "the researcher had not written a book yet"."""
    context = EquityContext(
        data=StubMarketData({}),  # type: ignore[arg-type]
        mcp=session(),
        journal=Journal(directory=tmp_path / "journal"),
        books=tmp_path / "empty",
        pinned_fingerprint=FINGERPRINT,
    )
    record = run_equity_cycle(context, as_of=NOW, submit=False)
    assert record.stage is EquityStage.NO_BOOK
    assert "no target book" in record.note
    assert record.cycle_id == cycle_id_for(NOW, 0)


def test_an_unreadable_book_is_reported_not_raised(tmp_path: Path, books: Path) -> None:
    (books / f"rs_test-{FINGERPRINT}-2026-08-28.json").write_text("{oops", encoding="utf-8")
    context = EquityContext(
        data=StubMarketData({}),  # type: ignore[arg-type]
        mcp=session(),
        journal=Journal(directory=tmp_path / "journal"),
        books=books,
        pinned_fingerprint=FINGERPRINT,
    )
    record = run_equity_cycle(context, as_of=NOW, submit=False)
    assert record.stage is EquityStage.NO_BOOK


def test_a_stale_book_is_refused_before_the_account_is_read(
    tmp_path: Path, books: Path
) -> None:
    """The order matters: a book nobody may execute is not a reason to spend a
    round trip finding out how much equity there is."""
    context = EquityContext(
        data=StubMarketData({}),  # type: ignore[arg-type]
        mcp=session(),
        journal=Journal(directory=tmp_path / "journal"),
        books=books,
        pinned_fingerprint=FINGERPRINT,
    )
    late = datetime(2026, 9, 30, 13, 45, tzinfo=UTC)
    record = run_equity_cycle(context, as_of=late, submit=False)
    assert record.stage is EquityStage.STALE_BOOK
    assert record.strategy["fingerprint"] == FINGERPRINT
    assert context.mcp is not None
    assert not context.mcp.calls_to(ACCOUNT_TOOL)  # type: ignore[union-attr]


def test_a_book_already_held_produces_no_orders(context: EquityContext) -> None:
    """The common case, and it is journalled rather than silent."""
    context.mcp = session(
        positions=[
            _row("AAA", "100"), _row("BBB", "120"), _row("CCC", "160"),
        ]
    )
    record = run_equity_cycle(context, as_of=NOW, submit=True)
    assert record.stage is EquityStage.NO_TRADES
    assert record.orders == ()
    assert "inside the 20% band" in record.note
    assert context.mcp.calls_to(PLACE_STOCK_ORDER_TOOL) == []


def test_a_pass_that_priced_nothing_is_not_a_no_trade_day(
    tmp_path: Path, books: Path
) -> None:
    """The 2026-08-31 failure, in one test.

    Before the open the free feed does not tick, so every quote is minutes old
    and every symbol is skipped. That is not "the book is already held"; it is
    "we could not look". Recording it as `NO_TRADES` closed the session and the
    book was never built.
    """
    context = EquityContext(
        data=StubMarketData(
            {AAA: Decimal(100), BBB: Decimal(50), CCC: Decimal(25)},
            age_seconds=DEFAULT_EQUITY_POLICY.max_quote_age + 1,
        ),  # type: ignore[arg-type]
        mcp=session(),
        journal=Journal(directory=tmp_path / "journal"),
        books=books,
        pinned_fingerprint=FINGERPRINT,
        policy=DEFAULT_EQUITY_POLICY,
    )
    record = run_equity_cycle(context, as_of=NOW, submit=True)
    assert record.stage is EquityStage.NO_MARKS
    assert record.orders == ()
    assert "3 stale_mark" in record.note
    assert context.mcp is not None
    assert context.mcp.calls_to(PLACE_STOCK_ORDER_TOOL) == []  # type: ignore[union-attr]


def test_a_blind_pass_is_still_journalled(tmp_path: Path, books: Path) -> None:
    """Not deciding is a thing that happened and the journal must say so.

    `NO_MARKS` differs from `NO_TRADES` only in whether the day stays open; both
    are a line in the file, because "the process was up and saw nothing" is an
    answer and silence is not.
    """
    journal = Journal(directory=tmp_path / "journal")
    context = EquityContext(
        data=StubMarketData(
            {AAA: Decimal(100)},
            age_seconds=DEFAULT_EQUITY_POLICY.max_quote_age + 1,
        ),  # type: ignore[arg-type]
        mcp=session(),
        journal=journal,
        books=books,
        pinned_fingerprint=FINGERPRINT,
        policy=DEFAULT_EQUITY_POLICY,
    )
    journal.append(run_equity_cycle(context, as_of=NOW, submit=True))
    written = journal.read(NOW.date())
    assert [row["stage"] for row in written] == ["no_marks"]


def test_no_marks_does_not_count_as_having_traded() -> None:
    """The stage is only useful if the session guard agrees with it."""
    assert EquityStage.NO_MARKS.value not in DECIDED_STAGES
    assert EquityStage.NO_TRADES.value in DECIDED_STAGES


def test_a_dry_pass_gates_everything_and_sends_nothing(context: EquityContext) -> None:
    """A first-class outcome, not a flag threaded through the trading path — so
    the journal never has to be read as "submitted, but not really"."""
    record = run_equity_cycle(context, as_of=NOW, submit=False)
    assert record.stage is EquityStage.PLANNED
    assert len(record.orders) == 3
    assert all(order.approved for order in record.orders)
    assert all(order.submission is None for order in record.orders)
    assert context.mcp is not None
    assert context.mcp.calls_to(PLACE_STOCK_ORDER_TOOL) == []  # type: ignore[union-attr]


def test_a_live_pass_submits_in_plan_order(context: EquityContext) -> None:
    record = run_equity_cycle(context, as_of=NOW, submit=True)
    assert record.stage is EquityStage.SUBMITTED
    assert context.mcp is not None
    sent = context.mcp.calls_to(PLACE_STOCK_ORDER_TOOL)  # type: ignore[union-attr]
    assert [call["symbol"] for call in sent] == ["AAA", "BBB", "CCC"]
    assert all(call["type"] == "market" for call in sent)
    assert all(order.submission is not None for order in record.orders)


def test_an_off_book_holding_is_sold_before_anything_is_bought(
    context: EquityContext,
) -> None:
    """The exit path, end to end. Sells release the buying power buys then spend."""
    context.mcp = session(positions=[_row("ZZZ", "40")])
    context.data = StubMarketData(  # type: ignore[assignment]
        {
            AAA: Decimal(100),
            BBB: Decimal(50),
            CCC: Decimal(25),
            ticker("ZZZ"): Decimal(200),
        }
    )
    run_equity_cycle(context, as_of=NOW, submit=True)
    sent = context.mcp.calls_to(PLACE_STOCK_ORDER_TOOL)
    assert sent[0]["symbol"] == "ZZZ"
    assert sent[0]["side"] == "sell"
    assert sent[0]["qty"] == "40"
    assert all(call["side"] == "buy" for call in sent[1:])


def test_the_snapshot_is_advanced_between_orders(context: EquityContext) -> None:
    """The daily caps must bind *within* a pass, not only between them.

    A pass that judged every order against the account as it stood at the top
    would let a hundred orders each pass a turnover check that the hundred of
    them together fail. Asserted through the buying-power check, which is the
    one that can be exhausted by a plan this small: $12,000 of cash against
    $20,000 of buys means the last order is refused for want of it.
    """
    context.mcp = session(cash="12000")
    record = run_equity_cycle(context, as_of=NOW, submit=True)
    vetoed = [order for order in record.orders if not order.approved]
    assert vetoed, "with $12k against $20k of buys, something must be refused"
    assert any(
        reason.check == "buying_power"
        for order in vetoed
        for reason in order.verdict.reasons  # type: ignore[union-attr]
    )


def test_a_transport_failure_stops_the_pass_rather_than_skipping_an_order(
    context: EquityContext,
) -> None:
    """The orders already placed stand and are journalled; the rest are not
    attempted, and the next pass re-derives them from what is actually held —
    which is the one source that cannot be wrong about what happened."""
    context.mcp = RecordedSession.scripted(
        **{
            ACCOUNT_TOOL: account_payload(),
            POSITIONS_TOOL: positions_payload([]),
            ASSET_TOOL: asset_payload("X"),
            PLACE_STOCK_ORDER_TOOL: [
                order_payload(),
                TransportFailure("connection reset"),
            ],
        }
    )
    record = run_equity_cycle(context, as_of=NOW, submit=True)
    assert record.stage is EquityStage.HALTED
    assert "stopped at" in record.note
    assert record.orders[0].submission is not None
    assert record.orders[-1].submission is None


def test_the_record_carries_the_provenance_the_dashboard_renders(
    context: EquityContext,
) -> None:
    """specs/09 D10. The block that makes the demo a claim rather than a
    screenshot — and every number in it is recorded, never acted on."""
    record = run_equity_cycle(context, as_of=NOW, submit=False)
    sealed = record.strategy["sealed"]
    assert record.strategy["fingerprint"] == FINGERPRINT
    assert sealed["t_alpha"] == pytest.approx(2.2202313842814925)
    assert sealed["looks"] == 1
    assert sealed["can_confirm"] is False
    assert sealed["refuted"] is False


def test_the_book_is_archived_by_the_pass(context: EquityContext) -> None:
    """Named by fingerprint, session and digest, so a book regenerated for the
    same session cannot land on top of the one that was executed."""
    record = run_equity_cycle(context, as_of=NOW, submit=False)
    archive = context.journal.directory / BOOK_ARCHIVE
    copied = list(archive.iterdir())
    assert len(copied) == 1
    assert copied[0].name == (
        f"{FINGERPRINT}-2026-08-27-{record.strategy['digest'][:12]}.json"
    )


def test_marks_are_requested_for_the_union_of_held_and_wanted(
    context: EquityContext,
) -> None:
    """A held name absent from the book still needs a price — it is about to be
    sold, and a sale priced by guesswork is the thing the planner refuses."""
    context.mcp = session(positions=[_row("ZZZ", "5")])
    run_equity_cycle(context, as_of=NOW, submit=False)
    asked = context.data.asked[-1]  # type: ignore[attr-defined]
    assert set(asked) == {"AAA", "BBB", "CCC", "ZZZ"}


# --------------------------------------------------------------------- #
# Today, so far
# --------------------------------------------------------------------- #


def test_todays_totals_are_read_off_the_journal(context: EquityContext) -> None:
    """So a restart at 14:00 resumes with the caps it left at, rather than with
    a fresh budget. The Gate cannot read a file — that is the point of it — so
    these arrive as fields on the snapshot."""
    record = run_equity_cycle(context, as_of=NOW, submit=True)
    context.journal.append(record)
    orders, turnover = today_totals(context.journal, NOW.date())
    assert orders == 3
    # $18,000, not $20,000: the book is sized against the equity *sleeve*
    # (specs/03 D6), not against this fixture's $100,000 account. The rest is
    # the options agent's allocation and this strategy may not spend it. Every
    # target scales by the same ratio, so that ratio is the whole of the change
    # -- which is why it is written as one here rather than as a literal. The
    # split moved from 95/5 to 90/10 when the researched option rule turned out
    # to need a $2,000 per-trade budget to fund a single contract.
    scale = EQUITY_SLEEVE_ALLOCATION / Decimal(100_000)
    assert turnover == Decimal(20_000) * scale


def test_an_options_cycle_does_not_count_towards_the_equity_caps(
    context: EquityContext,
) -> None:
    """One journal, two agents. The caps are per agent because the budgets are."""
    context.journal.append(
        {"cycle_id": "2026-08-28-SPY-001", "as_of": NOW.isoformat(), "stage": "filled"}
    )
    orders, turnover = today_totals(context.journal, NOW.date())
    assert (orders, turnover) == (0, Decimal(0))


# --------------------------------------------------------------------- #
# The drawdown, and the two numbers that have to be the same one
# --------------------------------------------------------------------- #


def _status(context: EquityContext) -> dict[str, Any]:
    return json.loads(
        (context.journal.directory / "equity-status.json").read_text(encoding="utf-8")
    )


class TestTheDrawdownIsMeasuredOnTheSleeve:
    """specs/03 D6, and the day it cost a rebalance.

    Two things compute a drawdown for this sleeve: the Gate's
    `drawdown_killswitch`, off `sleeve.drawdown(peak)`, and the status page. They
    have to be the same number, because one of them stops trading and the other
    is the only place a human would look to see it coming.

    They were not. The rebalance pass marked the high-water mark on the sleeve
    (~$90k, correct), the thirty-second heartbeat re-marked it on the *account*
    (~$101k), and the heartbeat runs two hundred times more often. So the stored
    peak was account-scale while the drawdown was measured sleeve-scale, giving a
    standing ~10.3% loss that never happened — and on 2026-09-03 it crossed the
    10% kill switch and refused the day's only rebalance while the page beside it
    read 0.08%.
    """

    def test_the_heartbeat_marks_the_sleeve_not_the_account(
        self, context: EquityContext
    ) -> None:
        _heartbeat(context, as_of=NOW, next_pass=None, sequence=1)
        assert context.peak_equity == EQUITY_SLEEVE_ALLOCATION, (
            "an untraded sleeve is worth its allocation; the account is worth "
            "that plus the options sleeve, and marking the account here is what "
            "made the kill switch measure one against the other"
        )

    def test_the_page_reports_what_the_gate_will_measure(
        self, context: EquityContext, books: Path, tmp_path: Path
    ) -> None:
        """Mark the peak, then lose $1,000 of it. The sleeve is down 1.11% of
        $90,000 — not 1.00% of the $100,000 account."""
        _heartbeat(context, as_of=NOW, next_pass=None, sequence=1)
        context.mcp = session(equity="99000", cash="99000")
        _heartbeat(context, as_of=NOW, next_pass=None, sequence=2)

        published = _status(context)
        account = read_account(context.mcp, observed_at=NOW)  # type: ignore[arg-type]
        sleeve = context.sleeve(account)
        assert published["equity"] == "99000", "the page still shows the account"
        assert Decimal(published["peak_equity"]) == EQUITY_SLEEVE_ALLOCATION
        assert Decimal(published["drawdown_pct"]) == sleeve.drawdown(
            peak=context.peak_equity
        ), "the page's drawdown is the Gate's drawdown"
        assert Decimal(published["drawdown_pct"]) > Decimal("0.011")

    def test_a_peak_stored_under_the_pre_fix_label_is_discarded(
        self, tmp_path: Path
    ) -> None:
        """The state files written before this fix say `equity-sleeve` and hold
        an account-scale number. The basis check exists for exactly this: the
        label was right about what it should have been and wrong about what was
        recorded, so the mark is dropped rather than believed."""
        path = tmp_path / "equity-state.json"
        path.write_text(
            json.dumps(
                {
                    "peak_equity": "101367.36",
                    "killswitch_tripped": False,
                    "basis": "equity-sleeve",
                }
            ),
            encoding="utf-8",
        )
        state = SessionState.load(path, basis=EQUITY_SLEEVE_BASIS)
        assert state.peak_equity is None
        assert state.discarded_peak == Decimal("101367.36")

    def test_the_sleeve_basis_says_what_it_measures(self) -> None:
        """A guard on the constant itself. Reusing the old label would silently
        adopt the poisoned marks it was bumped to discard."""
        assert EQUITY_SLEEVE_BASIS != "equity-sleeve"
        assert EQUITY_SLEEVE_BASIS.startswith("equity-sleeve")


def _row(symbol: str, qty: str) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "asset_class": "us_equity",
        "qty": qty,
        "side": "long",
        "avg_entry_price": "1",
        "market_value": "1",
    }


# --------------------------------------------------------------------- #
# The session guard — one pass per book, not one per day
# --------------------------------------------------------------------- #


def journalled(
    journal: Journal, digest: str, stage: EquityStage, *, sequence: int = 0
) -> None:
    """A minimal cycle line — only the fields the guard reads."""
    journal.append(
        {
            "kind": "equity",
            "cycle_id": cycle_id_for(NOW, sequence),
            "as_of": NOW.isoformat(),
            "stage": stage.value,
            "strategy": {"fingerprint": FINGERPRINT, "digest": digest},
        }
    )


def test_the_same_book_is_not_rebalanced_twice(tmp_path: Path) -> None:
    """A restart at 14:00 must not replay the morning."""
    journal = Journal(directory=tmp_path)
    journalled(journal, "abc123", EquityStage.SUBMITTED)
    assert already_decided(journal, NOW.date(), "abc123")


def test_a_regenerated_book_is_a_new_instruction(tmp_path: Path) -> None:
    """The point of keying on the digest.

    A book rebuilt mid-session with different weights is a different set of
    orders, and the research having improved is not a reason to wait until
    tomorrow to hold what it now says to hold.
    """
    journal = Journal(directory=tmp_path)
    journalled(journal, "abc123", EquityStage.SUBMITTED)
    assert not already_decided(journal, NOW.date(), "def456")


def test_a_blind_pass_does_not_settle_a_book(tmp_path: Path) -> None:
    journal = Journal(directory=tmp_path)
    journalled(journal, "abc123", EquityStage.NO_MARKS)
    assert not already_decided(journal, NOW.date(), "abc123")


def test_a_quiet_pass_does_settle_a_book(tmp_path: Path) -> None:
    """`NO_TRADES` about this book is an answer about this book."""
    journal = Journal(directory=tmp_path)
    journalled(journal, "abc123", EquityStage.NO_TRADES)
    assert already_decided(journal, NOW.date(), "abc123")


def test_a_book_refused_as_stale_is_not_retried(tmp_path: Path) -> None:
    """Re-reading the same too-old file cannot make it younger. A *new* file
    gets a new digest and its own pass."""
    journal = Journal(directory=tmp_path)
    journalled(journal, "abc123", EquityStage.STALE_BOOK)
    assert already_decided(journal, NOW.date(), "abc123")


def test_yesterdays_pass_does_not_settle_todays_book(tmp_path: Path) -> None:
    """Same bytes, new session: the holdings drifted overnight."""
    journal = Journal(directory=tmp_path)
    journalled(journal, "abc123", EquityStage.SUBMITTED)
    assert not already_decided(journal, NOW.date() + timedelta(days=1), "abc123")


def test_an_options_cycle_is_not_an_equity_decision(tmp_path: Path) -> None:
    """One journal, two agents — specs/06. The kind is load-bearing."""
    journal = Journal(directory=tmp_path)
    journal.append(
        {
            "kind": "option",
            "cycle_id": "2026-08-28-SPY-001",
            "as_of": NOW.isoformat(),
            "stage": "submitted",
            "strategy": {"digest": "abc123"},
        }
    )
    assert not already_decided(journal, NOW.date(), "abc123")


def test_a_line_without_a_strategy_is_stepped_over(tmp_path: Path) -> None:
    """`NO_BOOK` carries no strategy at all, and a heartbeat carries no stage."""
    journal = Journal(directory=tmp_path)
    stub = {"cycle_id": cycle_id_for(NOW, 0), "as_of": NOW.isoformat()}
    journal.append({**stub, "kind": "equity", "stage": "no_book"})
    journal.append(
        {**stub, "kind": "equity", "stage": "submitted", "strategy": None}
    )
    assert not already_decided(journal, NOW.date(), "abc123")


# --------------------------------------------------------------------- #
# The digest the guard keys on
# --------------------------------------------------------------------- #


def test_the_guards_digest_is_the_one_the_journal_records(books: Path) -> None:
    """Two ways of hashing the same file is one way of hashing it wrong.

    The guard cannot afford to parse and validate a book every thirty seconds,
    so it hashes the bytes directly — and that hash has to be the number
    `read_book` puts in the journal, or the guard compares against something
    that is never there and every heartbeat starts a pass.
    """
    path = find_latest_book(books, FINGERPRINT)
    assert path is not None
    book, _ = read_book(path, pinned_fingerprint=FINGERPRINT)
    assert digest_of(path) == book.digest


def test_a_book_that_is_not_there_has_no_digest(tmp_path: Path) -> None:
    """`aqr` rewrites its output in place. Reading during that window has to be
    a "not yet", not a crash."""
    assert digest_of(tmp_path / "gone.json") is None


# --------------------------------------------------------------------- #
# The archive, now that one session can hold two books
# --------------------------------------------------------------------- #


def test_two_books_for_the_same_session_do_not_overwrite_each_other(
    tmp_path: Path, books: Path
) -> None:
    """The archive is the evidence of what was executed.

    `aqr` names its output by session, and the archive used to copy that shape —
    so a book regenerated for the same `as_of` silently replaced the record of
    the pass before it. Now that a session can hold two passes, the digest has
    to be in the name.
    """
    path = find_latest_book(books, FINGERPRINT)
    assert path is not None
    book, raw = read_book(path, pinned_fingerprint=FINGERPRINT)

    revised_raw = raw + "\n"
    revised = replace(book, digest=sha256(revised_raw.encode("utf-8")).hexdigest())

    first = archive_book(raw, book, directory=tmp_path / "journal")
    second = archive_book(revised_raw, revised, directory=tmp_path / "journal")

    assert first != second
    assert first.read_bytes().decode("utf-8") == raw
    assert second.read_bytes().decode("utf-8") == revised_raw


def test_the_same_book_archived_twice_is_one_file(tmp_path: Path, books: Path) -> None:
    """Idempotent on a restart — the archive is a set of books, not a log of
    passes."""
    path = find_latest_book(books, FINGERPRINT)
    assert path is not None
    book, raw = read_book(path, pinned_fingerprint=FINGERPRINT)
    first = archive_book(raw, book, directory=tmp_path / "journal")
    second = archive_book(raw, book, directory=tmp_path / "journal")
    assert first == second
    assert len(list(first.parent.iterdir())) == 1


# --------------------------------------------------------------------- #
# The pin — what is on disk that we are not allowed to execute
# --------------------------------------------------------------------- #


OTHER = "96cbc95ab6f09a60"


def write_book(
    directory: Path,
    payload: dict[str, Any],
    *,
    name: str,
    fingerprint: str,
    as_of: str,
) -> Path:
    """A book for some *other* strategy, named the way `aqr` names them."""
    revised = {
        **payload,
        "spec_name": name,
        "spec_fingerprint": fingerprint,
        "as_of": as_of,
    }
    path = directory / f"{name}-{fingerprint}-{as_of}.json"
    path.write_text(json.dumps(revised, indent=2), encoding="utf-8")
    return path


def test_a_directory_holding_only_the_pinned_book_is_quiet(books: Path) -> None:
    """No advice is the common case and it must cost nothing to say."""
    assert unpinned_books(books, FINGERPRINT) == ()


def test_a_book_for_another_strategy_is_reported(
    books: Path, book_payload: dict[str, Any]
) -> None:
    """The failure this exists for.

    `find_latest_book` globs on the pinned fingerprint, so a book for a strategy
    the pin does not name is *invisible* — the pass says "no target book" while
    a perfectly good one sits in the same directory. The pin is right to refuse
    it. Saying nothing about it is not.
    """
    write_book(
        books, book_payload, name="low_vol_rs_carry_v5", fingerprint=OTHER,
        as_of="2026-08-28",
    )
    found = unpinned_books(books, FINGERPRINT)
    assert [(b.name, b.fingerprint, b.as_of) for b in found] == [
        ("low_vol_rs_carry_v5", OTHER, "2026-08-28")
    ]


def test_the_pin_is_read_from_the_contents_not_the_filename(
    books: Path, book_payload: dict[str, Any]
) -> None:
    """A file *named* for the pinned strategy whose contents say otherwise is
    exactly the case `load_target_book` refuses, and the operator deserves to be
    told why rather than left with a rejection."""
    path = books / f"mislabelled-{FINGERPRINT}-2026-08-28.json"
    path.write_text(
        json.dumps({**book_payload, "spec_fingerprint": OTHER}), encoding="utf-8"
    )
    assert [b.fingerprint for b in unpinned_books(books, FINGERPRINT)] == [OTHER]


def test_only_the_newest_book_per_strategy_is_reported(
    books: Path, book_payload: dict[str, Any]
) -> None:
    """`aqr` writes one per session. A month of them is a wall of text, and the
    only one that could possibly be meant is the latest."""
    for as_of in ("2026-08-25", "2026-08-28", "2026-08-26"):
        write_book(
            books, book_payload, name="v5", fingerprint=OTHER, as_of=as_of
        )
    found = unpinned_books(books, FINGERPRINT)
    assert [b.as_of for b in found] == ["2026-08-28"]


def test_the_report_is_newest_first_and_deterministic(
    books: Path, book_payload: dict[str, Any]
) -> None:
    """Same directory, same answer, same order — CLAUDE.md rule 7."""
    write_book(books, book_payload, name="b", fingerprint="b" * 16, as_of="2026-08-26")
    write_book(books, book_payload, name="a", fingerprint="a" * 16, as_of="2026-08-28")
    write_book(books, book_payload, name="c", fingerprint="c" * 16, as_of="2026-08-26")
    once = unpinned_books(books, FINGERPRINT)
    assert [b.name for b in once] == ["a", "b", "c"]
    assert unpinned_books(books, FINGERPRINT) == once


def test_a_malformed_file_is_stepped_over(books: Path) -> None:
    """This runs on the failure path. It must not have a failure path of its
    own — a directory with junk in it is a directory we still report on."""
    (books / "half-written.json").write_text("{not json", encoding="utf-8")
    (books / "empty.json").write_text("", encoding="utf-8")
    assert unpinned_books(books, FINGERPRINT) == ()


def test_a_missing_directory_reports_nothing(tmp_path: Path) -> None:
    assert unpinned_books(tmp_path / "nope", FINGERPRINT) == ()


# --------------------------------------------------------------------- #
# ...and what we tell the operator to do about it
# --------------------------------------------------------------------- #


def test_advice_names_the_variable_and_the_file(
    books: Path, book_payload: dict[str, Any]
) -> None:
    """The whole value is being actionable. "Fingerprint mismatch" sends someone
    reading source; this sends them to one line of one file."""
    write_book(
        books, book_payload, name="low_vol_rs_carry_v5", fingerprint=OTHER,
        as_of="2026-08-28",
    )
    advice = changed_pin_advice(unpinned_books(books, FINGERPRINT))
    assert "ALPHAGATE_STRATEGY_FINGERPRINT" in advice
    assert ".env.local" in advice
    assert OTHER in advice
    assert "low_vol_rs_carry_v5" in advice


def test_advice_is_empty_when_there_is_nothing_on_disk(books: Path) -> None:
    """An empty directory is a different problem, and the caller says so
    instead."""
    assert changed_pin_advice(unpinned_books(books, FINGERPRINT)) == ""


def test_advice_does_not_tell_anyone_it_switched(
    books: Path, book_payload: dict[str, Any]
) -> None:
    """It reports and stops. Changing the pin is a person's decision — that is
    the whole of what the pin is for (specs/09 D1), and advice that read like an
    action taken would undo it."""
    write_book(books, book_payload, name="v5", fingerprint=OTHER, as_of="2026-08-28")
    advice = changed_pin_advice(unpinned_books(books, FINGERPRINT))
    assert "change" in advice.lower()
    assert "switched" not in advice.lower()


# --------------------------------------------------------------------- #
# ...and where the operator actually meets it
# --------------------------------------------------------------------- #


def test_a_pass_with_no_pinned_book_names_the_one_it_found(
    tmp_path: Path, book_payload: dict[str, Any]
) -> None:
    """`equity-run` journals this, so the answer is in the file at 09:20 rather
    than in whoever was watching the terminal."""
    directory = tmp_path / "target_books"
    directory.mkdir()
    write_book(
        directory, book_payload, name="low_vol_rs_carry_v5", fingerprint=OTHER,
        as_of="2026-08-28",
    )
    context = EquityContext(
        data=StubMarketData({}),  # type: ignore[arg-type]
        mcp=session(),
        journal=Journal(directory=tmp_path / "journal"),
        books=directory,
        pinned_fingerprint=FINGERPRINT,
    )
    record = run_equity_cycle(context, as_of=NOW, submit=False)
    assert record.stage is EquityStage.NO_BOOK
    assert OTHER in record.note
    assert "ALPHAGATE_STRATEGY_FINGERPRINT" in record.note


def test_a_pass_with_an_empty_directory_still_says_run_aqr(
    tmp_path: Path
) -> None:
    """No book at all is a different problem, and must not be dressed up as a
    pin that needs changing."""
    directory = tmp_path / "target_books"
    directory.mkdir()
    context = EquityContext(
        data=StubMarketData({}),  # type: ignore[arg-type]
        mcp=session(),
        journal=Journal(directory=tmp_path / "journal"),
        books=directory,
        pinned_fingerprint=FINGERPRINT,
    )
    record = run_equity_cycle(context, as_of=NOW, submit=False)
    assert record.stage is EquityStage.NO_BOOK
    assert "aqr target-book" in record.note
    assert "ALPHAGATE_STRATEGY_FINGERPRINT" not in record.note


def test_a_healthy_pass_never_mentions_the_pin(
    tmp_path: Path, books: Path, book_payload: dict[str, Any]
) -> None:
    """The correction that matters.

    The pin is right, the book is found, and another strategy happens to sit in
    the same directory — which is what a research repository looks like on any
    ordinary day, for as long as that book stays in force. A book's `as_of` is
    not a freshness signal: the same file can be the one in force for weeks.
    So there is no comparison to make and nothing to report; the guard speaks
    only when the pin matched nothing at all.
    """
    write_book(books, book_payload, name="v5", fingerprint=OTHER, as_of="2026-09-30")
    context = EquityContext(
        data=StubMarketData(
            {AAA: Decimal(100), BBB: Decimal(50), CCC: Decimal(25)}
        ),  # type: ignore[arg-type]
        mcp=session(),
        journal=Journal(directory=tmp_path / "journal"),
        books=books,
        pinned_fingerprint=FINGERPRINT,
        policy=DEFAULT_EQUITY_POLICY,
    )
    record = run_equity_cycle(context, as_of=NOW, submit=False)
    assert record.stage is EquityStage.PLANNED
    assert "ALPHAGATE_STRATEGY_FINGERPRINT" not in record.note
    assert OTHER not in record.note
