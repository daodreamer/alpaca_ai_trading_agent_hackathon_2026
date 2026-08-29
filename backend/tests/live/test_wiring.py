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
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from alphagate.agent import Slot, Underlying
from alphagate.agent.schedule import CycleKind
from alphagate.core.identifiers import ticker
from alphagate.execution import ExecutionError
from alphagate.live.wiring import (
    DTE_MAX,
    DTE_MIN,
    LiveContext,
    SessionState,
    _api_right,
    expiry_window,
    market_session,
    mcp_session,
)
from alphagate.options import Right

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
