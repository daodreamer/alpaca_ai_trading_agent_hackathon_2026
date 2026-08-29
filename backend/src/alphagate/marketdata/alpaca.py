"""Alpaca market data over REST — adr/0002 D2.

GET only. There is no method in this file that can place, cancel or amend an
order, and `tests/test_boundaries.py` asserts that no write verb appears in the
package. Orders leave through exactly one door and this is not it.

Two things this adapter does that a thinner one would skip, and both of them
matter more than they look.

**It converts money at the boundary.** Alpaca sends prices as JSON numbers,
which arrive in Python as floats. Every one of them is converted through `str`
into a `Decimal` here, at the edge, so that `0.1` is `0.1` and not
`0.1000000000000000055511151231257827`. Doing it later is doing it after the
error has been baked in. This is also why the Alpaca SDK is deliberately not
used (specs/01 Rule 3): its models carry float prices.

**It refuses to invent a bar's identity.** A `core.bar.Bar` carries a feed and an
adjustment mode, and both are part of its identity — "the same window on IEX and
on SIP are different observations, not duplicates". So the feed and the
adjustment are passed in and recorded as what they are, never defaulted to
whatever the response happened to contain.

Pagination is followed to the end. A silently truncated bar series is a chart
with a hole in it, and every indicator downstream would report confidently on
the wrong window.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Any, Final

import httpx

from alphagate.core.bar import AdjustmentMode, Bar, Feed
from alphagate.core.errors import InvariantViolation
from alphagate.core.identifiers import Ticker
from alphagate.core.time_model import SessionKind, Timeframe
from alphagate.marketdata.port import OptionBar
from alphagate.options import Greeks, OptionContract, OptionQuote, parse_occ

__all__ = ["DEFAULT_DATA_URL", "AlpacaMarketData", "to_bar", "to_option_quote"]

DEFAULT_DATA_URL: Final = "https://data.alpaca.markets"
SOURCE: Final = "alpaca-rest"
_BAR_PAGE_LIMIT: Final = 10_000
_SNAPSHOT_PAGE_LIMIT: Final = 1_000
"""The ceilings the API actually enforces, which differ per endpoint. Sending
10,000 to the options snapshot route is a 400: `"invalid limit: larger than the
allowed maximum of 1000"`. Found by asking it, not by reading about it."""

_MAX_PAGES: Final = 50
"""A cap on the pagination loop. A server that keeps handing back a cursor is a
bug somewhere, and an unbounded loop turns it into a hang at the open."""


def _decimal(value: Any, *, field_name: str) -> Decimal:
    """Convert a JSON number to an exact value. Through `str`, always.

    `Decimal(0.1)` is not `0.1`. Going through the shortest round-trip repr is
    the one supervised crossing from the approximate domain to the exact one.
    """
    if value is None:
        raise InvariantViolation(f"{field_name} is missing")
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def _optional_decimal(value: Any) -> Decimal | None:
    return None if value is None else Decimal(str(value))


def _timestamp(value: Any, *, field_name: str) -> datetime:
    if not isinstance(value, str):
        raise InvariantViolation(f"{field_name} must be an ISO timestamp, got {value!r}")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def to_bar(
    payload: Mapping[str, Any],
    *,
    symbol: Ticker,
    timeframe: Timeframe,
    feed: Feed,
    adjustment: AdjustmentMode,
) -> Bar:
    """One Alpaca bar object into a domain `Bar`.

    The end time is the start plus the timeframe's nominal duration, except for
    `D1` where the domain models a bar as the 6.5-hour regular session rather
    than 24 hours (ADR 0004 D6) — a full day would exceed the nominal span and
    `Bar` would refuse it, correctly.
    """
    start = _timestamp(payload.get("t"), field_name="bar timestamp")
    span = (
        timedelta(hours=6, minutes=30)
        if timeframe is Timeframe.D1
        else timeframe.nominal_duration
    )
    return Bar(
        symbol=symbol,
        timeframe=timeframe,
        start_time_utc=start,
        end_time_utc=start + span,
        session_date=start.date(),
        session=SessionKind.REGULAR,
        open=_decimal(payload.get("o"), field_name="open"),
        high=_decimal(payload.get("h"), field_name="high"),
        low=_decimal(payload.get("l"), field_name="low"),
        close=_decimal(payload.get("c"), field_name="close"),
        volume=_decimal(payload.get("v", 0), field_name="volume"),
        source=SOURCE,
        feed=feed,
        adjustment_mode=adjustment,
        is_final=True,
        vwap=_optional_decimal(payload.get("vw")),
        trade_count=payload.get("n"),
    )


def to_option_quote(symbol: str, snapshot: Mapping[str, Any]) -> OptionQuote | None:
    """One chain snapshot into an `OptionQuote`. `None` when unquotable.

    A missing side, or a zero on either side, is not a market: it is a contract
    nobody is making a price in. Returning `None` keeps it out of the candidate
    menu entirely rather than letting a `0.00` bid become a `50%` spread that the
    liquidity check then has to argue with.

    Greeks are carried when present and left `None` when absent — specs/02 D2.
    A provider that omits delta must not be readable as delta-neutral.
    """
    quote = snapshot.get("latestQuote")
    if not isinstance(quote, Mapping):
        return None
    bid, ask = quote.get("bp"), quote.get("ap")
    if not bid or not ask:
        return None

    raw_greeks = snapshot.get("greeks")
    greeks: Greeks | None = None
    if isinstance(raw_greeks, Mapping):
        try:
            greeks = Greeks(
                delta=float(raw_greeks["delta"]),
                gamma=float(raw_greeks["gamma"]),
                theta=float(raw_greeks["theta"]),
                vega=float(raw_greeks["vega"]),
                rho=float(raw_greeks["rho"]),
                iv=float(snapshot.get("impliedVolatility", 0.0)),
            )
        except (KeyError, TypeError, ValueError):
            # A partial greeks object is worse than none: it would report an
            # exposure smaller than the real one (specs/02 D4).
            greeks = None

    return OptionQuote(
        contract=parse_occ(symbol),
        as_of=_timestamp(quote.get("t"), field_name="quote timestamp"),
        bid=_decimal(bid, field_name="bid"),
        ask=_decimal(ask, field_name="ask"),
        greeks=greeks,
    )


@dataclass
class AlpacaMarketData:
    """Read-only Alpaca data client. Nothing here writes."""

    key_id: str
    secret_key: str
    base_url: str = DEFAULT_DATA_URL
    feed: Feed = Feed.IEX
    """The paper account's entitlement. Recorded on every bar, never guessed —
    a SIP bar and an IEX bar of the same window are different observations."""
    adjustment: AdjustmentMode = AdjustmentMode.ADJUSTED
    timeout: float = 30.0
    client: httpx.Client | None = field(default=None, repr=False)

    def __repr__(self) -> str:
        """Redacted: a repr ends up in a traceback, and a traceback in a log."""
        return f"AlpacaMarketData(base_url={self.base_url!r}, feed={self.feed.value})"

    # ------------------------------------------------------------------ #
    # Transport — GET only
    # ------------------------------------------------------------------ #

    def _get(self, path: str, params: Mapping[str, Any]) -> dict[str, Any]:
        headers = {
            "APCA-API-KEY-ID": self.key_id,
            "APCA-API-SECRET-KEY": self.secret_key,
        }
        url = f"{self.base_url.rstrip('/')}{path}"
        cleaned = {k: v for k, v in params.items() if v is not None}
        if self.client is not None:
            response = self.client.get(url, params=cleaned, headers=headers, timeout=self.timeout)
        else:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(url, params=cleaned, headers=headers)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise InvariantViolation(
                f"{path} returned {type(payload).__name__}, expected an object"
            )
        return payload

    def _paged(
        self,
        path: str,
        params: Mapping[str, Any],
        key: str,
        *,
        page_limit: int = _BAR_PAGE_LIMIT,
    ) -> Iterator[dict[str, Any]]:
        """Follow `next_page_token` to the end.

        A truncated series is a chart with a hole in it, and every indicator
        downstream would report confidently on the wrong window.
        """
        cursor: str | None = None
        for _ in range(_MAX_PAGES):
            payload = self._get(path, {**params, "page_token": cursor, "limit": page_limit})
            body = payload.get(key)
            if isinstance(body, Mapping):
                yield dict(body)
            cursor = payload.get("next_page_token")
            if not cursor:
                return
        raise InvariantViolation(f"{path} paginated past {_MAX_PAGES} pages; refusing to loop")

    # ------------------------------------------------------------------ #
    # The port
    # ------------------------------------------------------------------ #

    def daily_bars(self, symbol: Ticker, *, start: date, end: date) -> tuple[Bar, ...]:
        return self._bars(symbol, Timeframe.D1, start.isoformat(), end.isoformat())

    def intraday_bars(
        self, symbol: Ticker, *, timeframe: str, start: datetime, end: datetime
    ) -> tuple[Bar, ...]:
        return self._bars(
            symbol, Timeframe.from_code(timeframe), start.isoformat(), end.isoformat()
        )

    def _bars(self, symbol: Ticker, timeframe: Timeframe, start: str, end: str) -> tuple[Bar, ...]:
        params = {
            "symbols": str(symbol),
            "timeframe": _alpaca_timeframe(timeframe),
            "start": start,
            "end": end,
            "adjustment": "split" if self.adjustment is AdjustmentMode.ADJUSTED else "raw",
            "feed": self.feed.value.lower(),
            "sort": "asc",
        }
        collected: list[Bar] = []
        for page in self._paged("/v2/stocks/bars", params, "bars"):
            for raw in page.get(str(symbol), []):
                collected.append(
                    to_bar(
                        raw,
                        symbol=symbol,
                        timeframe=timeframe,
                        feed=self.feed,
                        adjustment=self.adjustment,
                    )
                )
        return tuple(collected)

    def latest_price(self, symbol: Ticker) -> Decimal:
        payload = self._get("/v2/stocks/quotes/latest", {"symbols": str(symbol)})
        quote = payload.get("quotes", {}).get(str(symbol))
        if not isinstance(quote, Mapping):
            raise InvariantViolation(f"no quote for {symbol}")
        bid = _decimal(quote.get("bp"), field_name="bid")
        ask = _decimal(quote.get("ap"), field_name="ask")
        if bid <= 0 or ask <= 0:
            raise InvariantViolation(f"{symbol} has a one-sided market: bid {bid}, ask {ask}")
        return (bid + ask) / 2

    def option_chain(
        self,
        symbol: Ticker,
        *,
        expiry_from: date,
        expiry_to: date,
        strike_from: Decimal | None = None,
        strike_to: Decimal | None = None,
        right: str | None = None,
    ) -> Mapping[OptionContract, OptionQuote]:
        params = {
            "expiration_date_gte": expiry_from.isoformat(),
            "expiration_date_lte": expiry_to.isoformat(),
            "strike_price_gte": None if strike_from is None else str(strike_from),
            "strike_price_lte": None if strike_to is None else str(strike_to),
            "type": right,
            "feed": "indicative",
        }
        quotes: dict[OptionContract, OptionQuote] = {}
        for page in self._paged(
            f"/v1beta1/options/snapshots/{symbol}",
            params,
            "snapshots",
            page_limit=_SNAPSHOT_PAGE_LIMIT,
        ):
            for occ, snapshot in page.items():
                if not isinstance(snapshot, Mapping):
                    continue
                quote = to_option_quote(occ, snapshot)
                if quote is not None:
                    quotes[quote.contract] = quote
        return quotes

    def option_daily_bars(
        self, contract: OptionContract, *, start: date, end: date
    ) -> Sequence[OptionBar]:
        from alphagate.options import format_occ

        occ = format_occ(contract)
        params = {
            "symbols": occ,
            "timeframe": "1Day",
            "start": start.isoformat(),
            "end": end.isoformat(),
            "sort": "asc",
        }
        bars: list[OptionBar] = []
        for page in self._paged(
            "/v1beta1/options/bars", params, "bars", page_limit=_SNAPSHOT_PAGE_LIMIT
        ):
            for raw in page.get(occ, []):
                bars.append(
                    OptionBar(
                        contract=contract,
                        session_date=_timestamp(
                            raw.get("t"), field_name="bar timestamp"
                        ).date(),
                        close=_decimal(raw.get("c"), field_name="close"),
                        volume=int(raw.get("v", 0)),
                    )
                )
        return tuple(bars)


def _alpaca_timeframe(timeframe: Timeframe) -> str:
    """Domain code to Alpaca's own spelling. `1m` is `1Min`, not `1m`."""
    return {
        Timeframe.M1: "1Min",
        Timeframe.M5: "5Min",
        Timeframe.M15: "15Min",
        Timeframe.M30: "30Min",
        Timeframe.H1: "1Hour",
        Timeframe.H4: "4Hour",
        Timeframe.D1: "1Day",
        Timeframe.W1: "1Week",
    }[timeframe]
