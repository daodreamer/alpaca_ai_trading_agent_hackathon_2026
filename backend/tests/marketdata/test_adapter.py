"""The market-data adapter — adr/0002 D2.

Every payload replayed here was captured from Alpaca's REST API on 2026-08-26:
125 daily SPY bars, 200 option-chain snapshots, one top-of-book quote. The
parsing is tested against what the API actually sends, not against what the
documentation describes, and `RecordedMarketData` runs the *same* `to_bar` and
`to_option_quote` the live adapter does — a replay with its own parser would be
a second implementation quietly diverging from the first.

The two properties worth stating up front:

**Money is exact at the boundary.** Alpaca sends prices as JSON numbers, which
arrive as floats. Every one is converted through `str` into a `Decimal` here, at
the edge, because doing it later is doing it after the error is baked in.

**Nothing in this package writes.** There is no method that could place, cancel
or amend anything, and a boundary test in `tests/test_boundaries.py` asserts no
write verb appears in the package at all.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

import httpx
import pytest

from alphagate.core.bar import AdjustmentMode, Feed
from alphagate.core.errors import InvariantViolation
from alphagate.core.identifiers import Ticker, ticker
from alphagate.core.time_model import SessionKind, Timeframe
from alphagate.marketdata import RecordedMarketData
from alphagate.marketdata.alpaca import AlpacaMarketData, to_bar, to_option_quote
from alphagate.marketdata.port import MarketData
from alphagate.options import OptionContract, Right

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "marketdata"
SPY: Ticker = ticker("SPY")
END = date(2026, 8, 26)
START = date(2026, 2, 27)
FAKE_KEY = "PKTESTTESTTESTTESTTEST99"
FAKE_SECRET = "abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGH"


@pytest.fixture
def data() -> RecordedMarketData:
    return RecordedMarketData(directory=FIXTURES)


def payload(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


class TestDailyBars:
    def test_it_reads_the_captured_history(self, data: RecordedMarketData) -> None:
        bars = data.daily_bars(SPY, start=START, end=END)
        assert len(bars) == 125
        assert bars[0].session_date < bars[-1].session_date, "oldest first"

    def test_prices_are_exact_decimals(self, data: RecordedMarketData) -> None:
        """"Doing it later is doing it after the error is baked in." """
        bar = data.daily_bars(SPY, start=START, end=END)[-1]
        for field in (bar.open, bar.high, bar.low, bar.close):
            assert isinstance(field, Decimal)

    def test_a_price_survives_the_float_that_carried_it(self) -> None:
        """`Decimal(0.1)` is not `0.1`. Through `str`, always."""
        bar = to_bar(
            {"t": "2026-08-26T04:00:00Z", "o": 0.1, "h": 0.3, "l": 0.1, "c": 0.1, "v": 1},
            symbol=SPY,
            timeframe=Timeframe.D1,
            feed=Feed.IEX,
            adjustment=AdjustmentMode.ADJUSTED,
        )
        assert bar.close == Decimal("0.1")
        # RUF032 is right in general and deliberately wrong here: constructing
        # the bad value is how the test shows the good one differs from it.
        assert bar.close != Decimal(0.1)  # noqa: RUF032

    def test_the_bar_carries_its_identity_not_a_guess(self) -> None:
        """"The same window on IEX and on SIP are different observations." """
        raw = payload("bars_SPY_1Day")["bars"]["SPY"][0]
        sip = to_bar(
            raw,
            symbol=SPY,
            timeframe=Timeframe.D1,
            feed=Feed.SIP,
            adjustment=AdjustmentMode.UNADJUSTED,
        )
        assert sip.feed is Feed.SIP
        assert sip.adjustment_mode is AdjustmentMode.UNADJUSTED
        assert sip.source == "alpaca-rest"

    def test_a_daily_bar_is_a_session_not_a_day(self) -> None:
        """ADR 0004 D6: `D1` spans the 6.5-hour regular session. A 24-hour span
        would exceed the timeframe's nominal duration and `Bar` would refuse it."""
        bar = to_bar(
            payload("bars_SPY_1Day")["bars"]["SPY"][0],
            symbol=SPY,
            timeframe=Timeframe.D1,
            feed=Feed.IEX,
            adjustment=AdjustmentMode.ADJUSTED,
        )
        assert bar.end_time_utc - bar.start_time_utc == datetime(
            2026, 1, 1, 6, 30, tzinfo=UTC
        ) - datetime(2026, 1, 1, tzinfo=UTC)
        assert bar.session is SessionKind.REGULAR

    def test_windowing_happens_on_read_not_in_the_fixture(
        self, data: RecordedMarketData
    ) -> None:
        """One capture serves every test that wants a different slice."""
        narrow = data.daily_bars(SPY, start=date(2026, 8, 1), end=END)
        assert 0 < len(narrow) < 125

    def test_it_records_what_was_asked_for(self, data: RecordedMarketData) -> None:
        data.daily_bars(SPY, start=START, end=END)
        assert data.requests == [("daily_bars", "SPY")]

    def test_a_missing_capture_is_loud(self, tmp_path: Path) -> None:
        """"Re-record the fixture rather than falling back to a live call." """
        empty = RecordedMarketData(directory=tmp_path)
        with pytest.raises(InvariantViolation, match="no recorded payload"):
            empty.daily_bars(SPY, start=START, end=END)


class TestOptionChain:
    def test_it_reads_the_captured_snapshots(self, data: RecordedMarketData) -> None:
        chain = data.option_chain(
            SPY, expiry_from=date(2026, 9, 4), expiry_to=date(2026, 9, 11)
        )
        assert len(chain) > 100
        assert all(isinstance(c, OptionContract) for c in chain)

    def test_quotes_carry_greeks_when_the_provider_sends_them(
        self, data: RecordedMarketData
    ) -> None:
        chain = data.option_chain(
            SPY, expiry_from=date(2026, 9, 4), expiry_to=date(2026, 9, 11)
        )
        with_greeks = [q for q in chain.values() if q.greeks is not None]
        assert len(with_greeks) > 50
        assert all(q.greeks.iv > 0 for q in with_greeks)  # type: ignore[union-attr]

    def test_a_one_sided_market_is_not_a_quote(self) -> None:
        """A zero bid is not a market; it is a contract nobody is pricing.
        Dropping it here keeps a fake 50% spread out of the candidate menu."""
        assert to_option_quote(
            "SPY260904P00700000",
            {"latestQuote": {"bp": 0, "ap": 0.05, "t": "2026-08-26T14:00:00Z"}},
        ) is None

    def test_partial_greeks_are_refused_rather_than_half_read(self) -> None:
        """"A partial greeks object would report an exposure smaller than the
        real one" — specs/02 D4."""
        quote = to_option_quote(
            "SPY260904P00750000",
            {
                "latestQuote": {"bp": 1.0, "ap": 1.1, "t": "2026-08-26T14:00:00Z"},
                "greeks": {"delta": -0.2, "gamma": 0.01},
            },
        )
        assert quote is not None
        assert quote.greeks is None

    def test_filters_apply_on_read(self, data: RecordedMarketData) -> None:
        puts = data.option_chain(
            SPY,
            expiry_from=date(2026, 9, 4),
            expiry_to=date(2026, 9, 11),
            strike_from=Decimal(740),
            strike_to=Decimal(760),
            right="put",
        )
        assert puts
        assert all(c.right is Right.PUT for c in puts)
        assert all(Decimal(740) <= c.strike <= Decimal(760) for c in puts)

    def test_quote_timestamps_are_tz_aware(self, data: RecordedMarketData) -> None:
        chain = data.option_chain(
            SPY, expiry_from=date(2026, 9, 4), expiry_to=date(2026, 9, 11)
        )
        assert all(q.as_of.tzinfo is not None for q in chain.values())


class TestLatestPrice:
    def test_it_is_the_midpoint(self, data: RecordedMarketData) -> None:
        quote = payload("quote_SPY")["quotes"]["SPY"]
        expected = (Decimal(str(quote["bp"])) + Decimal(str(quote["ap"]))) / 2
        assert data.latest_price(SPY) == expected


class TestTheLiveAdapterIsGetOnly:
    """adr/0002 D2 and specs/01 Rule 1b. Orders leave by one door; not this one."""

    def test_it_only_ever_issues_get(self) -> None:
        methods: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            methods.append(request.method)
            body = (
                payload("quote_SPY")
                if "quotes/latest" in str(request.url)
                else payload("bars_SPY_1Day")
            )
            return httpx.Response(200, json=body)

        client = httpx.Client(transport=httpx.MockTransport(handler))
        adapter = AlpacaMarketData(key_id=FAKE_KEY, secret_key=FAKE_SECRET, client=client)
        adapter.daily_bars(SPY, start=START, end=END)
        adapter.latest_price(SPY)
        adapter.option_chain(SPY, expiry_from=START, expiry_to=END)
        assert len(methods) >= 3
        assert set(methods) == {"GET"}

    def test_it_satisfies_the_port(self) -> None:
        adapter = AlpacaMarketData(key_id=FAKE_KEY, secret_key=FAKE_SECRET)
        assert isinstance(adapter, MarketData)
        assert isinstance(RecordedMarketData(directory=FIXTURES), MarketData)

    def test_the_repr_never_shows_the_secret(self) -> None:
        """A repr ends up in a traceback, and a traceback in a log."""
        adapter = AlpacaMarketData(key_id=FAKE_KEY, secret_key=FAKE_SECRET)
        assert FAKE_SECRET not in repr(adapter)
        assert FAKE_KEY not in repr(adapter)

    def test_credentials_go_in_the_headers(self) -> None:
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200, json={"bars": {"SPY": []}})

        client = httpx.Client(transport=httpx.MockTransport(handler))
        AlpacaMarketData(
            key_id=FAKE_KEY, secret_key=FAKE_SECRET, client=client
        ).daily_bars(SPY, start=START, end=END)
        assert seen[0].headers["apca-api-key-id"] == FAKE_KEY

    def test_pagination_is_followed_to_the_end(self) -> None:
        """"A truncated bar series is a chart with a hole in it, and every
        indicator downstream would report confidently on the wrong window." """
        pages = [
            {"bars": {"SPY": [_raw("2026-08-24")]}, "next_page_token": "a"},
            {"bars": {"SPY": [_raw("2026-08-25")]}, "next_page_token": "b"},
            {"bars": {"SPY": [_raw("2026-08-26")]}, "next_page_token": None},
        ]
        served = iter(pages)

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=next(served))

        client = httpx.Client(transport=httpx.MockTransport(handler))
        bars = AlpacaMarketData(
            key_id=FAKE_KEY, secret_key=FAKE_SECRET, client=client
        ).daily_bars(SPY, start=START, end=END)
        assert len(bars) == 3

    def test_an_endless_cursor_is_refused_rather_than_looped(self) -> None:
        """A server that keeps handing back a cursor is a bug somewhere, and an
        unbounded loop turns it into a hang at the open."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200, json={"bars": {"SPY": [_raw("2026-08-26")]}, "next_page_token": "x"}
            )

        client = httpx.Client(transport=httpx.MockTransport(handler))
        with pytest.raises(InvariantViolation, match="refusing to loop"):
            AlpacaMarketData(
                key_id=FAKE_KEY, secret_key=FAKE_SECRET, client=client
            ).daily_bars(SPY, start=START, end=END)

    def test_a_one_sided_top_of_book_is_refused(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"quotes": {"SPY": {"bp": 0, "ap": 766.0}}})

        client = httpx.Client(transport=httpx.MockTransport(handler))
        with pytest.raises(InvariantViolation, match="one-sided market"):
            AlpacaMarketData(
                key_id=FAKE_KEY, secret_key=FAKE_SECRET, client=client
            ).latest_price(SPY)


class TestNoSecretsInTheFixtures:
    @pytest.mark.parametrize("name", ["bars_SPY_1Day", "chain_SPY", "quote_SPY"])
    def test_the_captures_are_clean(self, name: str) -> None:
        blob = (FIXTURES / f"{name}.json").read_text(encoding="utf-8")
        assert "APCA" not in blob
        assert "secret" not in blob.lower()


def _raw(day: str) -> dict:
    return {"t": f"{day}T04:00:00Z", "o": 760, "h": 770, "l": 755, "c": 766, "v": 1000}
