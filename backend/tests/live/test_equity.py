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
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from alphagate.core.identifiers import Ticker, ticker
from alphagate.equity import DEFAULT_EQUITY_POLICY, UnusableBook
from alphagate.execution import RecordedSession, TransportFailure
from alphagate.execution.equity import PLACE_STOCK_ORDER_TOOL
from alphagate.journal import Journal
from alphagate.live.equity import (
    BOOK_ARCHIVE,
    EquityContext,
    EquityStage,
    archive_book,
    cycle_id_for,
    find_latest_book,
    marks_from,
    read_book,
    run_equity_cycle,
    today_totals,
)
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
    run_equity_cycle(context, as_of=NOW, submit=False)
    archive = context.journal.directory / BOOK_ARCHIVE
    assert (archive / f"{FINGERPRINT}-2026-08-27.json").is_file()


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
    assert turnover == Decimal(20_000)


def test_an_options_cycle_does_not_count_towards_the_equity_caps(
    context: EquityContext,
) -> None:
    """One journal, two agents. The caps are per agent because the budgets are."""
    context.journal.append(
        {"cycle_id": "2026-08-28-SPY-001", "as_of": NOW.isoformat(), "stage": "filled"}
    )
    orders, turnover = today_totals(context.journal, NOW.date())
    assert (orders, turnover) == (0, Decimal(0))


def _row(symbol: str, qty: str) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "asset_class": "us_equity",
        "qty": qty,
        "side": "long",
        "avg_entry_price": "1",
        "market_value": "1",
    }
