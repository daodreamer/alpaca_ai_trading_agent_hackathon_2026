"""Daily and intraday bars from Alpaca's market data API.

Why a second real provider when Yahoo already works: the same account that
prices the trades supplies the bars, the universe is every US-listed equity
rather than whatever Yahoo will hand a scraper, and the intraday timeframes the
architecture's v0.2 needs (5m, 15m) are available without a gateway process.

Two details that are not adapter boilerplate.

**Availability, not just occurrence.** Alpaca stamps a bar with the instant the
period *opened*. A daily bar labelled ``2024-01-02T05:00:00Z`` describes a
session that had not happened yet at that instant. Copying that stamp into
``available_time`` would tell the point-in-time loader that today's close was
knowable at today's open, which is look-ahead entering through the data layer --
where none of the backtester's defences can see it. So ``available_time`` is the
*end* of the bar's period.

**Adjusted by default.** An unadjusted 4:1 split is a -75% return the strategy
never took, and a mean-reversion rule will find it every time.

**Regular hours by default.** Alpaca serves extended-hours intraday bars and
has no parameter to say otherwise: a 15-minute SPY pull returned 59 bars per
session where 09:30-16:00 holds 26, the first stamped 04:00 New York and the
last 19:45. Those bars are thin, their spreads are a multiple of the
session's, and the backtester treats every bar alike -- so they are filtered
out here unless asked for. See :mod:`aqr.data.sessions`.

**SIP by default, not IEX.** IEX is one venue with a few percent of the volume,
and its historical bars are correspondingly sparse: a 2016-2026 pull of SPY came
back with 1530 bars covering roughly 1960 sessions, including one 634-day hole.
The same request on the consolidated tape returned 2677 bars with no gap larger
than a holiday weekend. An account without a SIP entitlement gets an error here,
which is the right outcome -- silently researching on a series with a two-year
hole in it is worse than not researching at all. ``aqr pull`` checks every
series it writes for exactly this, via :mod:`aqr.data.quality`.

The HTTP client is injected, so nothing here needs a network to be tested. It is
also why ``httpx`` is an optional extra rather than a hard dependency: this
module is reachable from ``aqr.data``, which the offline research loop uses.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, Protocol, cast

import numpy as np

from aqr.config import load_env_files
from aqr.data.bars import Bars, bar_duration, ensure_utc
from aqr.data.sessions import regular_hours_only as _regular_hours_only

__all__ = ["AlpacaProvider", "alpaca_timeframe"]

DATA_URL = "https://data.alpaca.markets"

# Statuses that mean "later", not "no". 429 is the rate limiter and 5xx is
# the vendor having a moment; both clear on their own. Everything else --
# 403 for an entitlement the account lacks, 404 for a symbol that never
# traded -- is a fact, and retrying a fact only makes the failure slower.
_TRANSIENT = frozenset({429, 500, 502, 503, 504})

# Only timeframes the rest of the system can actually use. ``4h`` is absent
# because Alpaca has no such bar, and quietly substituting ``1h`` would produce
# a backtest that is internally consistent and answers a question nobody asked.
_TIMEFRAMES: dict[str, str] = {
    "1m": "1Min",
    "5m": "5Min",
    "15m": "15Min",
    "30m": "30Min",
    "1h": "1Hour",
    "1D": "1Day",
    "1W": "1Week",
}


def alpaca_timeframe(timeframe: str) -> str:
    """Our timeframe name in Alpaca's spelling."""
    if timeframe not in _TIMEFRAMES:
        raise ValueError(
            f"Alpaca has no bar for timeframe {timeframe!r}; known: {sorted(_TIMEFRAMES)}"
        )
    return _TIMEFRAMES[timeframe]


class _Response(Protocol):
    status_code: int
    text: str

    def json(self) -> Any: ...


class HttpClient(Protocol):
    """The slice of an HTTP client this adapter needs. ``httpx.Client`` satisfies it."""

    def get(
        self, url: str, *, params: dict[str, Any], headers: dict[str, str]
    ) -> _Response: ...


class AlpacaProvider:
    """Point-in-time bars from Alpaca.

    Credentials come from the same ``.env.local`` the rest of the repository
    uses. They are sent as headers and never appear in a URL, a parameter or an
    error message.
    """

    def __init__(
        self,
        key_id: str | None = None,
        secret: str | None = None,
        *,
        feed: str = "sip",
        adjustment: str = "all",
        base_url: str = DATA_URL,
        client: HttpClient | None = None,
        page_limit: int = 10_000,
        regular_hours_only: bool = True,
        max_retries: int = 6,
        retry_backoff: float = 1.5,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.regular_hours_only = regular_hours_only
        self.feed = feed
        self.adjustment = adjustment
        self.base_url = base_url.rstrip("/")
        self.page_limit = page_limit
        self._client = client
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        # Injected so the retry tests do not spend real seconds sleeping.
        self._sleep = sleep
        self._key_id, self._secret = _credentials(key_id, secret)

    def _get(
        self,
        client: HttpClient,
        url: str,
        params: dict[str, Any],
        symbol: str,
        interval: str,
    ) -> Any:
        """One page, retrying the statuses that mean "later".

        Building a 682-ticker cache is 682 sequential requests and the rate
        limiter will be hit. Treating that as a hard failure left tickers
        permanently unfetchable -- repeating the pull re-hit the limit at the
        same point -- and a missing ticker is a hole in the universe rather than
        a slow download.
        """
        delay = self.retry_backoff
        for attempt in range(self.max_retries + 1):
            response = client.get(url, params=params, headers=self._headers())
            if response.status_code == 200:
                return response.json()
            if response.status_code not in _TRANSIENT or attempt == self.max_retries:
                # The body may echo the request back. Report the status and the
                # symbol; never the credential.
                raise RuntimeError(
                    f"Alpaca returned {response.status_code} for {symbol} {interval}"
                )
            self._sleep(delay)
            delay *= 2
        raise AssertionError("unreachable")

    def dataset_version(self, timeframe: str) -> str:
        """What produced these bars. Recorded with every experiment.

        Two runs against different feeds are not comparable, and comparing them
        anyway is how a data change gets attributed to a strategy change.
        """
        session = "rth" if self.regular_hours_only else "all-hours"
        return f"alpaca:{self.feed}:{self.adjustment}:{session}:{timeframe}"

    def load(self, symbol: str, start: datetime, end: datetime, timeframe: str = "1D") -> Bars:
        start, end = ensure_utc(start), ensure_utc(end)
        if end <= start:
            raise ValueError("end must be after start")

        interval = alpaca_timeframe(timeframe)
        duration = int(bar_duration(timeframe).total_seconds())
        client = self._require_client()
        url = f"{self.base_url}/v2/stocks/bars"
        params: dict[str, Any] = {
            "symbols": symbol,
            "timeframe": interval,
            "start": start.isoformat().replace("+00:00", "Z"),
            "end": end.isoformat().replace("+00:00", "Z"),
            "adjustment": self.adjustment,
            "feed": self.feed,
            "limit": self.page_limit,
            "sort": "asc",
        }

        rows: dict[int, tuple[float, float, float, float, float]] = {}
        seen_tokens: set[str] = set()
        token: str | None = None
        while True:
            page_params = dict(params)
            if token:
                page_params["page_token"] = token
            payload = self._get(client, url, page_params, symbol, interval)
            for raw in (payload.get("bars") or {}).get(symbol, []) or []:
                event = _epoch(raw["t"])
                # Later pages win on a duplicate: Alpaca's page boundaries can
                # repeat a bar, and Bars refuses non-increasing timestamps.
                rows[event] = (
                    float(raw["o"]),
                    float(raw["h"]),
                    float(raw["l"]),
                    float(raw["c"]),
                    float(raw["v"]),
                )
            token = payload.get("next_page_token") or None
            if token is None:
                break
            if token in seen_tokens:
                raise RuntimeError(
                    f"Alpaca repeated page token {token!r} for {symbol}; refusing to loop"
                )
            seen_tokens.add(token)

        if not rows:
            raise ValueError(f"{symbol}: no bars from Alpaca for {start}..{end} at {interval}")

        stamps = sorted(rows)
        cols = list(zip(*(rows[t] for t in stamps), strict=True))
        event_time = np.array(stamps, dtype=np.int64)
        series = Bars(
            symbol=symbol,
            timeframe=timeframe,
            event_time=event_time,
            # A bar is knowable when its period ends, not when it began.
            available_time=event_time + duration,
            open=np.array(cols[0], dtype=np.float64),
            high=np.array(cols[1], dtype=np.float64),
            low=np.array(cols[2], dtype=np.float64),
            close=np.array(cols[3], dtype=np.float64),
            volume=np.array(cols[4], dtype=np.float64),
        )
        return _regular_hours_only(series) if self.regular_hours_only else series

    # ------------------------------------------------------------------ #

    def _headers(self) -> dict[str, str]:
        return {
            "APCA-API-KEY-ID": self._key_id,
            "APCA-API-SECRET-KEY": self._secret,
            "Accept": "application/json",
        }

    def _require_client(self) -> HttpClient:
        if self._client is not None:
            return self._client
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                "AlpacaProvider needs the 'alpaca' extra: uv sync --extra alpaca"
            ) from exc
        # httpx types its own parameters far more narrowly than this adapter
        # needs; the cast records that the structural match is real and that the
        # Protocol exists for the tests, not for httpx.
        client = cast(HttpClient, httpx.Client(timeout=30.0))
        self._client = client
        return client


def _credentials(key_id: str | None, secret: str | None) -> tuple[str, str]:
    if key_id and secret:
        return key_id, secret
    load_env_files()
    resolved_key = key_id or os.environ.get("ALPACA_API_KEY_ID", "").strip()
    resolved_secret = secret or os.environ.get("ALPACA_API_SECRET_KEY", "").strip()
    if not resolved_key or not resolved_secret:
        raise RuntimeError(
            "ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY are not set. Put them in "
            ".env.local at the repository root."
        )
    return resolved_key, resolved_secret


def _epoch(stamp: str) -> int:
    return int(datetime.fromisoformat(stamp.replace("Z", "+00:00")).astimezone(UTC).timestamp())
