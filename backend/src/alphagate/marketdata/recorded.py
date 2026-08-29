"""Replay recorded payloads — adr/0002 D2, specs/01 Rule 4.

The reason the market-data path is REST and not MCP: **this replays and an MCP
session does not.** The backtest and the live agent run the same parsing code
over the same payload shapes, and the only difference between them is where the
bytes came from and what time it is.

`RecordedMarketData` is deliberately built on the *same* `to_bar` and
`to_option_quote` functions the live adapter uses. A replay that parsed payloads
its own way would be a second implementation quietly diverging from the first,
and the day it diverged would be the day the backtest stopped describing the
system.

Captures are plain JSON files, one per request shape, in a directory. They are
committed, diffable, and can be re-recorded with `capture_to`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from alphagate.core.bar import AdjustmentMode, Bar, Feed
from alphagate.core.errors import InvariantViolation
from alphagate.core.identifiers import Ticker
from alphagate.core.time_model import Timeframe
from alphagate.marketdata.alpaca import to_bar, to_option_quote, to_stock_snapshot
from alphagate.marketdata.port import OptionBar, StockSnapshot
from alphagate.options import OptionContract, OptionQuote, format_occ

__all__ = ["RecordedMarketData"]


@dataclass
class RecordedMarketData:
    """A `MarketData` backed by captured JSON. Never opens a socket."""

    directory: Path
    feed: Feed = Feed.IEX
    adjustment: AdjustmentMode = AdjustmentMode.ADJUSTED
    requests: list[tuple[str, str]] = field(default_factory=list)
    """What was asked for, in order. Lets a test assert that the perception step
    fetched what it claimed to, rather than quietly using a stale local."""

    def _load(self, name: str) -> dict[str, Any]:
        path = self.directory / f"{name}.json"
        if not path.is_file():
            raise InvariantViolation(
                f"no recorded payload {name!r} in {self.directory}; "
                "re-record the fixture rather than falling back to a live call"
            )
        parsed = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(parsed, dict):
            raise InvariantViolation(f"{name} is a {type(parsed).__name__}, expected an object")
        return parsed

    # ------------------------------------------------------------------ #

    def daily_bars(self, symbol: Ticker, *, start: date, end: date) -> tuple[Bar, ...]:
        self.requests.append(("daily_bars", str(symbol)))
        return self._bars(f"bars_{symbol}_1Day", symbol, Timeframe.D1, start, end)

    def intraday_bars(
        self, symbol: Ticker, *, timeframe: str, start: datetime, end: datetime
    ) -> tuple[Bar, ...]:
        self.requests.append(("intraday_bars", f"{symbol}:{timeframe}"))
        return self._bars(
            f"bars_{symbol}_{timeframe}",
            symbol,
            Timeframe.from_code(timeframe),
            start.date(),
            end.date(),
        )

    def _bars(
        self, name: str, symbol: Ticker, timeframe: Timeframe, start: date, end: date
    ) -> tuple[Bar, ...]:
        payload = self._load(name)
        raw = payload.get("bars", {}).get(str(symbol), [])
        bars = tuple(
            to_bar(
                item,
                symbol=symbol,
                timeframe=timeframe,
                feed=self.feed,
                adjustment=self.adjustment,
            )
            for item in raw
        )
        # Windowing happens here, not in the fixture, so one capture serves every
        # test that wants a different slice of the same history.
        return tuple(b for b in bars if start <= b.session_date <= end)

    def latest_price(self, symbol: Ticker) -> Decimal:
        self.requests.append(("latest_price", str(symbol)))
        quote = self._load(f"quote_{symbol}").get("quotes", {}).get(str(symbol))
        if not isinstance(quote, Mapping):
            raise InvariantViolation(f"no recorded quote for {symbol}")
        return (Decimal(str(quote["bp"])) + Decimal(str(quote["ap"]))) / 2

    def stock_snapshots(
        self, symbols: Sequence[Ticker]
    ) -> Mapping[Ticker, StockSnapshot]:
        """Marks from one captured `snapshots` payload — specs/09 D2.

        One fixture for the whole batch, keyed by symbol, because that is what
        the live route returns and a per-symbol file would let a test replay a
        combination the API cannot produce.

        A symbol with no entry is absent from the result, exactly as it is live.
        That is the case the planner's `NO_MARK` skip exists for, and it has to
        be reachable offline or it is only tested by hoping.
        """
        self.requests.append(("stock_snapshots", ",".join(sorted(str(s) for s in symbols))))
        body = self._load("snapshots").get("snapshots", {})
        if not isinstance(body, Mapping):
            raise InvariantViolation("recorded snapshots payload has no 'snapshots' object")
        wanted = {str(symbol) for symbol in symbols}
        found: dict[Ticker, StockSnapshot] = {}
        for raw_symbol, snapshot in body.items():
            if str(raw_symbol) not in wanted or not isinstance(snapshot, Mapping):
                continue
            mark = to_stock_snapshot(str(raw_symbol), snapshot)
            if mark is not None:
                found[mark.symbol] = mark
        return found

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
        self.requests.append(("option_chain", str(symbol)))
        payload = self._load(f"chain_{symbol}")
        snapshots = payload.get("snapshots", payload.get("data", {}).get("snapshots", {}))
        quotes: dict[OptionContract, OptionQuote] = {}
        for occ, snapshot in snapshots.items():
            if not isinstance(snapshot, Mapping):
                continue
            quote = to_option_quote(occ, snapshot)
            if quote is None:
                continue
            contract = quote.contract
            if not expiry_from <= contract.expiry <= expiry_to:
                continue
            if strike_from is not None and contract.strike < strike_from:
                continue
            if strike_to is not None and contract.strike > strike_to:
                continue
            if right is not None and contract.right.value != right[0].upper():
                continue
            quotes[contract] = quote
        return quotes

    def option_daily_bars(
        self, contract: OptionContract, *, start: date, end: date
    ) -> Sequence[OptionBar]:
        occ = format_occ(contract)
        self.requests.append(("option_daily_bars", occ))
        payload = self._load(f"optionbars_{occ}")
        raw = payload.get("bars", {}).get(occ, [])
        bars = [
            OptionBar(
                contract=contract,
                session_date=datetime.fromisoformat(
                    str(item["t"]).replace("Z", "+00:00")
                ).date(),
                close=Decimal(str(item["c"])),
                volume=int(item.get("v", 0)),
            )
            for item in raw
        ]
        return tuple(b for b in bars if start <= b.session_date <= end)
