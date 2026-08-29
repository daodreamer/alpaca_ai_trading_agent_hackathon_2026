"""The Interactive Brokers bar adapter.

No test here starts TWS. The ``IB`` handle is injected, which is what makes the
parts worth testing testable: chunking a twenty-year request into pieces the API
will actually answer, obeying the pacing limit without a real clock, and
refusing a bar size IBKR does not have.

The pacing test matters more than it looks. IBKR's limit is enforced by
disconnecting the client, and a puller that trips it halfway through a universe
leaves a half-written cache that looks like a complete one.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any

import pytest

from aqr.data.ibkr import IbkrProvider, ibkr_bar_size, ibkr_duration

START = datetime(2020, 1, 1, tzinfo=UTC)
END = datetime(2024, 1, 1, tzinfo=UTC)


@dataclass
class _Bar:
    date: Any
    open: float
    high: float
    low: float
    close: float
    volume: float


class _StubIB:
    """Enough of ``ib_async.IB`` to drive the adapter."""

    def __init__(self, pages: list[list[_Bar]] | None = None, connected: bool = False) -> None:
        self.pages = pages if pages is not None else []
        self.requests: list[dict[str, Any]] = []
        self.connect_calls: list[dict[str, Any]] = []
        self._connected = connected
        self.disconnected = False

    def isConnected(self) -> bool:  # noqa: N802 - the real API spells it this way
        return self._connected

    def connect(self, host: str, port: int, clientId: int, **kw: Any) -> None:  # noqa: N803
        self.connect_calls.append({"host": host, "port": port, "clientId": clientId, **kw})
        self._connected = True

    def disconnect(self) -> None:
        self.disconnected = True
        self._connected = False

    def qualifyContracts(self, contract: Any) -> list[Any]:  # noqa: N802
        return [contract]

    def reqHistoricalData(self, contract: Any, **kw: Any) -> list[_Bar]:  # noqa: N802
        self.requests.append({"contract": contract, **kw})
        return self.pages.pop(0) if self.pages else []


def _daily(day: date, close: float = 100.0) -> _Bar:
    return _Bar(date=day, open=close - 1, high=close + 1, low=close - 2, close=close, volume=1e6)


class TestParameterMapping:
    @pytest.mark.parametrize(
        ("ours", "theirs"),
        [("1m", "1 min"), ("5m", "5 mins"), ("1h", "1 hour"), ("1D", "1 day"), ("1W", "1 week")],
    )
    def test_maps_the_timeframes_the_rest_of_the_system_uses(
        self, ours: str, theirs: str
    ) -> None:
        assert ibkr_bar_size(ours) == theirs

    def test_refuses_a_bar_size_ibkr_does_not_have(self) -> None:
        with pytest.raises(ValueError, match="4h"):
            ibkr_bar_size("4h")

    def test_duration_never_exceeds_what_the_api_will_answer(self) -> None:
        # Asking for 20 years of daily bars in one call gets an error, not 20
        # years of bars.
        assert ibkr_duration("1D", 20 * 365) == "1 Y"
        assert ibkr_duration("1D", 200) == "200 D"
        assert ibkr_duration("5m", 200) == "5 D"


class TestChunking:
    """Only unadjusted modes chunk: ADJUSTED_LAST refuses an end date, so it
    cannot be split into pieces at all (see TestUndatedMode)."""

    def test_a_long_window_is_split_into_requests_the_api_accepts(self) -> None:
        pages = [
            [_daily(date(2020 + i, 6, d)) for d in range(1, 5)] for i in range(4)
        ]
        ib = _StubIB(list(reversed(pages)), connected=True)
        provider = IbkrProvider(what_to_show="TRADES", ib=ib, sleep=lambda _s: None)

        bars = provider.load("AAPL", START, END, "1D")

        assert len(ib.requests) == 4, "four years of daily bars needs four requests"
        assert len(bars) == 16
        assert list(bars.event_time) == sorted(bars.event_time)

    def test_walks_backwards_from_the_end_so_the_newest_bars_are_never_missing(self) -> None:
        ib = _StubIB([[_daily(date(2023, 6, 1))], []], connected=True)
        provider = IbkrProvider(what_to_show="TRADES", ib=ib, sleep=lambda _s: None)
        provider.load("AAPL", START, END, "1D")

        first = ib.requests[0]["endDateTime"]
        assert first.year == 2024, f"first request should end at the window end, got {first}"

    def test_an_empty_chunk_does_not_end_the_pull(self) -> None:
        # A holiday-heavy or pre-listing chunk returns nothing. Stopping there
        # would silently truncate a symbol's history at an arbitrary year.
        ib = _StubIB(
            [[], [_daily(date(2022, 6, 1))], [], [_daily(date(2020, 6, 1))]], connected=True
        )
        provider = IbkrProvider(what_to_show="TRADES", ib=ib, sleep=lambda _s: None)
        bars = provider.load("AAPL", START, END, "1D")
        assert len(bars) == 2
        assert len(ib.requests) == 4


class TestPacing:
    def test_waits_between_requests_rather_than_being_disconnected(self) -> None:
        """IBKR enforces its limit by dropping the client, which leaves a
        half-written cache that looks complete."""
        slept: list[float] = []
        ib = _StubIB([[_daily(date(2020 + i, 6, 1))] for i in range(4)], connected=True)
        provider = IbkrProvider(
            what_to_show="TRADES", ib=ib, sleep=slept.append, pacing_seconds=11.0
        )

        provider.load("AAPL", START, END, "1D")

        assert slept, "a multi-chunk pull must pace itself"
        assert all(s == 11.0 for s in slept)
        assert len(slept) == len(ib.requests) - 1, "no wait is needed before the first request"


class TestPointInTime:
    def test_a_daily_bar_is_not_knowable_until_its_session_closes(self) -> None:
        ib = _StubIB([[_daily(date(2023, 6, 1))]], connected=True)
        provider = IbkrProvider(ib=ib, sleep=lambda _s: None)
        bars = provider.load("AAPL", datetime(2023, 1, 1, tzinfo=UTC), END, "1D")
        assert int(bars.available_time[0]) > int(bars.event_time[0])

    def test_asks_for_split_and_dividend_adjusted_bars(self) -> None:
        ib = _StubIB([[_daily(date(2023, 6, 1))]], connected=True)
        provider = IbkrProvider(ib=ib, sleep=lambda _s: None)
        provider.load("AAPL", datetime(2023, 1, 1, tzinfo=UTC), END, "1D")
        assert ib.requests[0]["whatToShow"] == "ADJUSTED_LAST"

    def test_reports_what_produced_the_bars(self) -> None:
        provider = IbkrProvider(ib=_StubIB(connected=True), sleep=lambda _s: None)
        assert provider.dataset_version("1D") == "ibkr:ADJUSTED_LAST:rth:1D"


class TestConnection:
    def test_connects_only_when_the_handle_is_not_already_connected(self) -> None:
        ib = _StubIB([[_daily(date(2023, 6, 1))]], connected=True)
        provider = IbkrProvider(ib=ib, sleep=lambda _s: None)
        provider.load("AAPL", datetime(2023, 1, 1, tzinfo=UTC), END, "1D")
        assert ib.connect_calls == []

    def test_connects_with_the_configured_endpoint_when_it_is_not(self) -> None:
        ib = _StubIB([[_daily(date(2023, 6, 1))]], connected=False)
        provider = IbkrProvider(
            host="10.0.0.2", port=4001, client_id=42, ib=ib, sleep=lambda _s: None
        )
        provider.load("AAPL", datetime(2023, 1, 1, tzinfo=UTC), END, "1D")
        call = ib.connect_calls[0]
        assert (call["host"], call["port"], call["clientId"]) == ("10.0.0.2", 4001, 42)

    def test_the_client_declares_itself_read_only(self) -> None:
        """Not a workaround for a TWS dialog -- a statement of what this class
        does. It calls reqHistoricalData and nothing else, so the handshake
        should not be asking for write access it will never use.

        The default is the opposite: ib_async connects read-write and binds an
        order-id sequence during the handshake, which TWS refuses outright when
        the user has Read-Only API enabled. Declaring read-only means the safest
        TWS configuration is also the one that works."""
        ib = _StubIB([[_daily(date(2023, 6, 1))]], connected=False)
        IbkrProvider(ib=ib, sleep=lambda _s: None).load(
            "AAPL", datetime(2023, 1, 1, tzinfo=UTC), END, "1D"
        )
        assert ib.connect_calls[0]["readonly"] is True

    def test_the_handshake_asks_for_no_account_or_order_state(self) -> None:
        """ib_async downloads positions, open orders, completed orders, account
        updates and executions on connect by default. A bar provider has no
        business holding any of it."""
        ib = _StubIB([[_daily(date(2023, 6, 1))]], connected=False)
        IbkrProvider(ib=ib, sleep=lambda _s: None).load(
            "AAPL", datetime(2023, 1, 1, tzinfo=UTC), END, "1D"
        )
        fetch = ib.connect_calls[0].get("fetchFields")
        assert fetch is not None, "the default fetch set must be overridden explicitly"
        assert not bool(fetch), f"expected an empty startup fetch, got {fetch!r}"

    def test_an_injected_handle_is_never_disconnected_by_the_provider(self) -> None:
        # The caller owns a handle it passed in; closing it under them would
        # break the next symbol in a multi-symbol pull.
        ib = _StubIB([[_daily(date(2023, 6, 1))]], connected=True)
        provider = IbkrProvider(ib=ib, sleep=lambda _s: None)
        provider.load("AAPL", datetime(2023, 1, 1, tzinfo=UTC), END, "1D")
        provider.close()
        assert ib.disconnected is False


class TestFailures:
    def test_an_empty_window_says_so(self) -> None:
        ib = _StubIB([[], [], [], []], connected=True)
        provider = IbkrProvider(ib=ib, sleep=lambda _s: None)
        with pytest.raises(ValueError, match="no bars"):
            provider.load("NOPE", START, END, "1D")

    def test_duplicate_bars_across_chunks_are_dropped(self) -> None:
        ib = _StubIB(
            [[_daily(date(2023, 6, 1))], [_daily(date(2023, 6, 1))]], connected=True
        )
        provider = IbkrProvider(ib=ib, sleep=lambda _s: None)
        bars = provider.load("AAPL", datetime(2022, 1, 1, tzinfo=UTC), END, "1D")
        assert len(bars) == 1


class TestUndatedMode:
    """ADJUSTED_LAST refuses an explicit end date, confirmed against a live TWS:

        ADJUSTED_LAST with end date  FAILED
        ADJUSTED_LAST no end date    ok  251 bars
        TRADES        with end date  ok  251 bars

    So it cannot be chunked. All of its history has to arrive in one request,
    which ends at "now" whatever window was asked for.
    """

    def test_adjusted_last_sends_one_undated_request(self) -> None:
        ib = _StubIB([[_daily(date(2023, 6, 1))]], connected=True)
        provider = IbkrProvider(ib=ib, sleep=lambda _s: None, now=lambda: END)

        provider.load("AAPL", START, END, "1D")

        assert len(ib.requests) == 1, "chunking an undated request is not possible"
        assert ib.requests[0]["endDateTime"] == ""

    def test_the_duration_covers_the_whole_window(self) -> None:
        ib = _StubIB([[_daily(date(2023, 6, 1))]], connected=True)
        provider = IbkrProvider(ib=ib, sleep=lambda _s: None, now=lambda: END)

        provider.load("AAPL", datetime(2005, 1, 1, tzinfo=UTC), END, "1D")
        assert ib.requests[0]["durationStr"] == "19 Y"

    def test_the_duration_is_measured_to_now_not_to_the_requested_end(self) -> None:
        """An undated request always ends at the present. A window ending in
        2015 still needs every year since, or the years in between arrive empty
        and the window is silently short."""
        ib = _StubIB([[_daily(date(2013, 6, 3))]], connected=True)
        provider = IbkrProvider(
            ib=ib, sleep=lambda _s: None, now=lambda: datetime(2026, 1, 1, tzinfo=UTC)
        )
        provider.load("AAPL", datetime(2010, 1, 1, tzinfo=UTC), datetime(2015, 1, 1, tzinfo=UTC))
        assert ib.requests[0]["durationStr"] == "16 Y"

    def test_bars_outside_the_requested_window_are_still_discarded(self) -> None:
        ib = _StubIB(
            [[_daily(date(2013, 6, 3)), _daily(date(2020, 6, 1))]], connected=True
        )
        provider = IbkrProvider(
            ib=ib, sleep=lambda _s: None, now=lambda: datetime(2026, 1, 1, tzinfo=UTC)
        )
        bars = provider.load(
            "AAPL", datetime(2010, 1, 1, tzinfo=UTC), datetime(2015, 1, 1, tzinfo=UTC)
        )
        assert len(bars) == 1

    def test_an_undated_pull_pays_no_pacing_delay(self) -> None:
        # One request cannot trip a rate limit, and an eleven-second wait per
        # symbol across fifty symbols is nine minutes of nothing.
        slept: list[float] = []
        ib = _StubIB([[_daily(date(2023, 6, 1))]], connected=True)
        IbkrProvider(ib=ib, sleep=slept.append, now=lambda: END).load("AAPL", START, END)
        assert slept == []

    def test_trades_still_chunks_because_it_accepts_an_end_date(self) -> None:
        ib = _StubIB([[_daily(date(2020 + i, 6, 1))] for i in range(4)], connected=True)
        provider = IbkrProvider(
            what_to_show="TRADES", ib=ib, sleep=lambda _s: None, now=lambda: END
        )
        provider.load("AAPL", START, END, "1D")
        assert len(ib.requests) == 4
        assert all(r["endDateTime"] for r in ib.requests)
