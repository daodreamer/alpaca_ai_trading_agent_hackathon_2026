"""Bars from Interactive Brokers, through TWS or IB Gateway.

IBKR is the deepest history available to a retail account: two decades of daily
bars on names Yahoo has forgotten and Alpaca never listed, plus non-US venues.
The cost is that it is not an HTTP API. A gateway process must be running and
logged in, and the connection is stateful, which is why this adapter takes an
injected ``IB`` handle -- one connection serves a whole universe pull, and
nothing here needs a gateway to be tested.

Three things this module exists to get right.

**Chunking.** IBKR answers a bounded duration per request: roughly a year of
daily bars, a handful of days of minute bars. A twenty-year request does not
return twenty years, it returns an error. So the window is walked *backwards*
from its end in acceptable pieces -- backwards because a truncated pull should
be missing 2004, not last week.

**Pacing.** The limit is around sixty historical requests per ten minutes, and
IBKR enforces it by disconnecting the client. A puller that trips it halfway
through a universe leaves a half-written cache that looks exactly like a
complete one. So requests are spaced, and the delay is injectable so the tests
do not have to wait.

**Adjustment.** ``ADJUSTED_LAST`` rather than ``TRADES``: an unadjusted 4:1
split is a -75% return the strategy never took.

**ADJUSTED_LAST cannot be chunked.** Confirmed against a live TWS rather than
read off a page: an ``ADJUSTED_LAST`` request carrying an explicit
``endDateTime`` comes back with nothing, the same request with that field
empty returns bars, and ``TRADES`` accepts either. So the adjusted mode
issues exactly one request -- which necessarily ends at the present -- and
the requested window is applied by filtering afterwards. Unadjusted modes
keep the chunked path.

**Read-only, declared.** ``ib_async`` connects read-write by default and binds
an order-id sequence during the handshake, which TWS refuses outright when the
user has Read-Only API enabled -- so the default turns the safest TWS
configuration into the one that does not work. This class calls
``reqHistoricalData`` and nothing else, so it says so: ``readonly=True``, and an
empty startup fetch. The default startup fetch downloads positions, open orders,
completed orders, account updates and executions; a bar provider has no business
holding any of it.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any, Protocol, cast

import numpy as np

from aqr.data.bars import Bars, bar_duration, ensure_utc

__all__ = [
    "IbkrProvider",
    "ProbeResult",
    "ibkr_bar_size",
    "ibkr_duration",
    "probe_durations",
    "probe_requests",
]

# The gateway's default ports. 7497/7496 are TWS paper/live; 4002/4001 are the
# headless Gateway's. Paper is the default here for the same reason it is
# everywhere else in this repository.
TWS_PAPER_PORT = 7497
GATEWAY_PAPER_PORT = 4002

_BAR_SIZES: dict[str, str] = {
    "1m": "1 min",
    "5m": "5 mins",
    "15m": "15 mins",
    "30m": "30 mins",
    "1h": "1 hour",
    "1D": "1 day",
    "1W": "1 week",
}

# Calendar days of history IBKR will return in a single request, per bar size.
# Conservative on purpose: the failure mode of asking for too much is a rejected
# request, and the cost of asking for too little is one extra round trip.
_MAX_DAYS_PER_REQUEST: dict[str, int] = {
    "1m": 1,
    "5m": 5,
    "15m": 14,
    "30m": 28,
    "1h": 28,
    # 366, not 365: IBKR's "1 Y" covers a leap year too, and a 365-day chunk
    # turns any four-year window into five requests for the sake of one day.
    "1D": 366,
    "1W": 366,
}


# Modes IBKR will not answer with an explicit end date. Measured, not assumed:
# aqr ibkr-check sends every shape and reports which came back with bars.
REQUIRES_EMPTY_END = frozenset({"ADJUSTED_LAST"})


def ibkr_bar_size(timeframe: str) -> str:
    """Our timeframe name in IBKR's spelling."""
    if timeframe not in _BAR_SIZES:
        raise ValueError(
            f"IBKR has no bar size for timeframe {timeframe!r}; known: {sorted(_BAR_SIZES)}"
        )
    return _BAR_SIZES[timeframe]


def ibkr_duration(timeframe: str, days: int) -> str:
    """A duration string for ``days`` of history, clamped to what IBKR answers.

    Clamping rather than raising: the caller's window is legitimate, it just
    cannot arrive in one request. :meth:`IbkrProvider.load` splits it.
    """
    if timeframe not in _MAX_DAYS_PER_REQUEST:
        raise ValueError(f"unsupported timeframe {timeframe!r}")
    capped = max(1, min(int(days), _MAX_DAYS_PER_REQUEST[timeframe]))
    return "1 Y" if capped >= 365 else f"{capped} D"


class IB(Protocol):
    """The slice of ``ib_async.IB`` this adapter uses."""

    def isConnected(self) -> bool: ...  # noqa: N802 - the real API spells it this way

    def connect(  # noqa: N803
        self, host: str, port: int, clientId: int, **kwargs: Any
    ) -> Any: ...

    def disconnect(self) -> None: ...

    def qualifyContracts(self, contract: Any) -> list[Any]: ...  # noqa: N802

    def reqHistoricalData(self, contract: Any, **kwargs: Any) -> list[Any]: ...  # noqa: N802


class IbkrProvider:
    """Historical bars from a running TWS or IB Gateway.

    ``ib`` may be supplied by the caller, in which case this class never
    connects it and never closes it: a universe pull opens one connection and
    reuses it, and closing it under the caller would break the next symbol.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = TWS_PAPER_PORT,
        client_id: int = 17,
        *,
        what_to_show: str = "ADJUSTED_LAST",
        use_rth: bool = True,
        ib: IB | None = None,
        sleep: Callable[[float], None] | None = None,
        pacing_seconds: float = 11.0,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.client_id = client_id
        self.what_to_show = what_to_show
        self.use_rth = use_rth
        self.pacing_seconds = pacing_seconds
        self._owns_connection = ib is None
        self._ib = ib
        self._sleep = sleep if sleep is not None else _default_sleep
        # An undated request always ends at the present, so the duration has
        # to be measured against a clock. Injected, so the tests do not need
        # a real one.
        self._now = now if now is not None else _utc_now

    def dataset_version(self, timeframe: str) -> str:
        """What produced these bars. Recorded with every experiment."""
        session = "rth" if self.use_rth else "all"
        return f"ibkr:{self.what_to_show}:{session}:{timeframe}"

    def close(self) -> None:
        """Disconnect, but only a connection this provider opened."""
        if self._owns_connection and self._ib is not None and self._ib.isConnected():
            self._ib.disconnect()

    def load(self, symbol: str, start: datetime, end: datetime, timeframe: str = "1D") -> Bars:
        start, end = ensure_utc(start), ensure_utc(end)
        if end <= start:
            raise ValueError("end must be after start")

        bar_size = ibkr_bar_size(timeframe)
        step_seconds = int(bar_duration(timeframe).total_seconds())
        chunk_days = _MAX_DAYS_PER_REQUEST[timeframe]
        ib = self._connect()
        contract = self._contract(symbol)

        rows: dict[int, tuple[float, float, float, float, float]] = {}

        if self.what_to_show in REQUIRES_EMPTY_END:
            # One request, ending at the present because there is no way to
            # say otherwise. The duration is measured from start to *now*
            # rather than to end: a window closing in 2015 still needs every
            # year since, or those years arrive empty and the window comes
            # back silently short by a decade.
            span_days = max(
                1, math.ceil((self._now() - start).total_seconds() / 86_400)
            )
            raw = ib.reqHistoricalData(
                contract,
                endDateTime="",
                durationStr=_years_duration(span_days),
                barSizeSetting=bar_size,
                whatToShow=self.what_to_show,
                useRTH=self.use_rth,
                formatDate=2,
            )
            self._collect(raw, rows, start, end)
            return self._build(
                symbol, timeframe, rows, step_seconds, bar_size, start, end
            )

        total_days = max(1, math.ceil((end - start).total_seconds() / 86_400))
        chunks = max(1, math.ceil(total_days / chunk_days))

        cursor = end
        for index in range(chunks):
            if index:
                # Not before the first request: a single-chunk pull should not
                # pay a pacing delay it does not owe.
                self._sleep(self.pacing_seconds)
            remaining = max(1, math.ceil((cursor - start).total_seconds() / 86_400))
            raw = ib.reqHistoricalData(
                contract,
                endDateTime=cursor,
                durationStr=ibkr_duration(timeframe, remaining),
                barSizeSetting=bar_size,
                whatToShow=self.what_to_show,
                useRTH=self.use_rth,
                formatDate=2,
            )
            self._collect(raw, rows, start, end)
            # Step back a whole chunk regardless of what came back. An empty
            # chunk is a holiday stretch or a pre-listing gap, not the end of
            # the symbol's history -- stopping here would truncate it at an
            # arbitrary year.
            cursor = cursor - timedelta(days=chunk_days)
            if cursor <= start:
                break

        return self._build(symbol, timeframe, rows, step_seconds, bar_size, start, end)

    @staticmethod
    def _collect(
        raw: list[Any] | None,
        rows: dict[int, tuple[float, float, float, float, float]],
        start: datetime,
        end: datetime,
    ) -> None:
        """Keep the bars inside the requested window, and only those."""
        for item in raw or []:
            event = _epoch(item.date)
            if not start.timestamp() <= event < end.timestamp():
                continue
            rows[event] = (
                float(item.open),
                float(item.high),
                float(item.low),
                float(item.close),
                float(max(item.volume, 0.0)),
            )

    def _build(
        self,
        symbol: str,
        timeframe: str,
        rows: dict[int, tuple[float, float, float, float, float]],
        step_seconds: int,
        bar_size: str,
        start: datetime,
        end: datetime,
    ) -> Bars:
        if not rows:
            raise ValueError(f"{symbol}: no bars from IBKR for {start}..{end} at {bar_size}")

        stamps = sorted(rows)
        cols = list(zip(*(rows[t] for t in stamps), strict=True))
        event_time = np.array(stamps, dtype=np.int64)
        return Bars(
            symbol=symbol,
            timeframe=timeframe,
            event_time=event_time,
            # A bar is knowable when its period ends, not when it began.
            available_time=event_time + step_seconds,
            open=np.array(cols[0], dtype=np.float64),
            high=np.array(cols[1], dtype=np.float64),
            low=np.array(cols[2], dtype=np.float64),
            close=np.array(cols[3], dtype=np.float64),
            volume=np.array(cols[4], dtype=np.float64),
        )

    # ------------------------------------------------------------------ #

    def _connect(self) -> IB:
        ib = self._ib
        if ib is None:
            try:
                from ib_async import IB as _IB
            except ImportError as exc:  # pragma: no cover - depends on optional extra
                raise RuntimeError(
                    "IbkrProvider needs the 'ibkr' extra: uv sync --extra ibkr. "
                    "It also needs TWS or IB Gateway running and logged in."
                ) from exc
            # ib_async types its own handshake far more narrowly than this
            # adapter needs; the cast records that the structural match is real
            # and that the Protocol exists for the tests, not for ib_async.
            ib = cast(IB, _IB())
            self._ib = ib
        if not ib.isConnected():
            ib.connect(self.host, self.port, clientId=self.client_id, **_readonly_handshake())
        return ib

    def _contract(self, symbol: str) -> Any:
        assert self._ib is not None
        return _qualify(self._ib, symbol)


@dataclass(frozen=True, slots=True)
class ProbeResult:
    """What one request shape did when it was actually sent."""

    what_to_show: str
    dated: bool
    """Whether an explicit ``endDateTime`` was sent, or the field left empty."""
    bars: int
    error: str | None = None

    def __str__(self) -> str:
        end = "with end date" if self.dated else "no end date "
        if self.error:
            return f"  {self.what_to_show:<13} {end}  FAILED  {self.error[:70]}"
        return f"  {self.what_to_show:<13} {end}  ok      {self.bars} bars"


def probe_requests(
    ib: IB,
    symbol: str = "AAPL",
    *,
    end: datetime | None = None,
    duration: str = "1 Y",
    bar_size: str = "1 day",
    timeout: float = 30.0,
) -> list[ProbeResult]:
    """Send every plausible request shape and report which ones work.

    A live ``reqHistoricalData`` failure is ambiguous: ``ADJUSTED_LAST``
    refusing an explicit end date, a missing market-data subscription, an
    unaccepted duration and a short timeout all surface as the same timeout plus
    "Error 366: No historical data query found". Testing them one round trip at
    a time is slow and each round trip needs a human. This asks all four at
    once.

    Reads only. It qualifies a contract and requests bars; it touches no account
    or order state.
    """
    contract = _qualify(ib, symbol)
    results: list[ProbeResult] = []
    for what_to_show in ("ADJUSTED_LAST", "TRADES"):
        for dated in (True, False):
            # IBKR's documented restriction is that ADJUSTED_LAST wants an empty
            # end date. "Documented" and "true of the running build" are not the
            # same claim, which is why both are sent rather than one assumed.
            end_value: Any = (end or datetime.now(UTC)) if dated else ""
            try:
                raw = ib.reqHistoricalData(
                    contract,
                    endDateTime=end_value,
                    durationStr=duration,
                    barSizeSetting=bar_size,
                    whatToShow=what_to_show,
                    useRTH=True,
                    formatDate=2,
                    timeout=timeout,
                )
            except Exception as exc:
                results.append(ProbeResult(what_to_show, dated, 0, f"{type(exc).__name__}: {exc}"))
                continue
            count = len(raw or [])
            results.append(
                ProbeResult(
                    what_to_show,
                    dated,
                    count,
                    None if count else "returned no bars and no exception",
                )
            )
    return results


# Windows to try when finding out how much history one request will answer.
# IBKR's published step table says a "1 day" bar size accepts "1 Y"; practice is
# widely reported to be more generous, and the difference decides whether
# ADJUSTED_LAST -- which cannot be chunked -- can reach past one year at all.
_DURATION_LADDER = ("1 Y", "5 Y", "10 Y", "20 Y", "30 Y")


def probe_durations(
    ib: IB,
    symbol: str = "AAPL",
    *,
    what_to_show: str = "ADJUSTED_LAST",
    bar_size: str = "1 day",
    timeout: float = 60.0,
) -> list[ProbeResult]:
    """How much history one undated request returns, window by window.

    Always undated, because that is the shape ``ADJUSTED_LAST`` requires; a
    dated request here would measure something else. Reads only.
    """
    contract = _qualify(ib, symbol)
    results: list[ProbeResult] = []
    for duration in _DURATION_LADDER:
        try:
            raw = ib.reqHistoricalData(
                contract,
                endDateTime="",
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow=what_to_show,
                useRTH=True,
                formatDate=2,
                timeout=timeout,
            )
        except Exception as exc:
            results.append(ProbeResult(duration, False, 0, f"{type(exc).__name__}: {exc}"))
            continue
        count = len(raw or [])
        results.append(
            ProbeResult(
                duration, False, count, None if count else "returned no bars and no exception"
            )
        )
    return results


def _readonly_handshake() -> dict[str, Any]:
    """Connection options that ask TWS for nothing this provider will not use.

    ``fetchFields`` is an ib_async flag enum; the empty value is the falsy zero
    member, which is not exported by name. If the enum is missing -- an older
    ib_async, or the stubbed handle the tests inject -- ``readonly`` alone still
    carries the part that matters.
    """
    options: dict[str, Any] = {"readonly": True}
    try:
        from ib_async.ib import StartupFetch
    except ImportError:  # pragma: no cover - depends on optional extra
        return options
    options["fetchFields"] = StartupFetch(0)
    return options


def _years_duration(days: int) -> str:
    """A whole-year IBKR duration covering ``days``, rounded up.

    Years rather than days because an undated request is asking for all of
    it, and IBKR answers a coarse unit over a long span more readily than a
    fine one.
    """
    return f"{max(1, math.ceil(days / 365.25))} Y"


def _utc_now() -> datetime:  # pragma: no cover - trivial
    return datetime.now(UTC)


def _qualify(ib: IB, symbol: str) -> Any:
    """A US stock contract IBKR has confirmed it recognises."""
    try:
        from ib_async import Stock
    except ImportError:  # pragma: no cover - the stub path, used by the tests
        return {"symbol": symbol, "exchange": "SMART", "currency": "USD"}
    qualified = ib.qualifyContracts(Stock(symbol, "SMART", "USD"))
    if not qualified:
        raise ValueError(f"IBKR could not qualify a US stock contract for {symbol!r}")
    return qualified[0]


def _default_sleep(seconds: float) -> None:  # pragma: no cover - trivial
    import time

    time.sleep(seconds)


def _epoch(stamp: Any) -> int:
    """IBKR hands back a datetime, a date, or an epoch string depending on the
    bar size and ``formatDate``. All three mean the same thing."""
    if isinstance(stamp, datetime):
        moment = stamp if stamp.tzinfo else stamp.replace(tzinfo=UTC)
        return int(moment.astimezone(UTC).timestamp())
    if isinstance(stamp, date):
        return int(datetime(stamp.year, stamp.month, stamp.day, tzinfo=UTC).timestamp())
    text = str(stamp).strip()
    if text.isdigit():
        return int(text)
    return int(datetime.fromisoformat(text).replace(tzinfo=UTC).timestamp())
