"""Where bars come from.

Three providers ship with the MVP:

``SyntheticProvider``  deterministic regime-switching prices. No network, seeded,
                       reproducible -- this is what the tests and the offline
                       research loop run against.
``CsvProvider``        ``timestamp,open,high,low,close,volume`` files on disk.
``YFinanceProvider``   daily bars from Yahoo, if the optional ``data`` extra is
                       installed. Purely an adapter; it adds no semantics.

A provider's only job is to return a validated :class:`Bars`. Anything that
looks like analysis belongs in ``features/``.
"""

from __future__ import annotations

import csv
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from aqr.data.bars import Array, Bars, bar_duration, bars_per_year, ensure_utc

__all__ = ["CsvProvider", "Provider", "SyntheticProvider", "YFinanceProvider"]


class Provider(Protocol):
    """Anything that can hand us a point-in-time bar series."""

    def load(self, symbol: str, start: datetime, end: datetime, timeframe: str) -> Bars: ...


# --------------------------------------------------------------------------- #
# Synthetic
# --------------------------------------------------------------------------- #

# Annualised drift and volatility per regime, plus the per-bar probability of
# leaving it. Deliberately coarse: the point is to produce data on which a
# regime filter *can* help, not to model any real market.
_REGIMES: dict[str, tuple[float, float, float]] = {
    "TREND_BULL": (0.18, 0.14, 0.010),
    "TREND_BEAR": (-0.15, 0.26, 0.014),
    "RANGE_LOW_VOL": (0.02, 0.09, 0.012),
    "RANGE_HIGH_VOL": (0.00, 0.32, 0.020),
}
_REGIME_NAMES = tuple(_REGIMES)


class SyntheticProvider:
    """Regime-switching geometric Brownian motion with a per-symbol seed.

    Same symbol and same window always yield the same bars, so an experiment is
    reproducible from its recorded inputs alone. Each symbol gets its own seed,
    so cross-asset robustness testing sees genuinely different paths rather than
    one path four times.
    """

    def __init__(self, seed: int = 7, annual_bars: float | None = None) -> None:
        self.seed = seed
        # Bars per year the regime drifts and vols are annualised against.
        # None derives it from the timeframe at load: 252 daily, 1638 hourly.
        # Hard-coding 252 on intraday bars scales every bar's variance by
        # bars_per_year / 252 -- hourly bars would carry a day's worth of
        # noise each.
        self.annual_bars = annual_bars

    def _symbol_seed(self, symbol: str) -> int:
        acc = self.seed
        for ch in symbol:
            acc = (acc * 131 + ord(ch)) % (2**31 - 1)
        return acc

    def load(self, symbol: str, start: datetime, end: datetime, timeframe: str = "1D") -> Bars:
        start, end = ensure_utc(start), ensure_utc(end)
        if end <= start:
            raise ValueError("end must be after start")
        step = bar_duration(timeframe)

        stamps: list[int] = []
        cursor = start
        while cursor < end:
            # Weekdays only, so daily bars line up with a real trading calendar.
            if step < timedelta(days=1) or cursor.weekday() < 5:
                stamps.append(int(cursor.timestamp()))
            cursor += step
        n = len(stamps)
        if n == 0:
            raise ValueError(f"{symbol}: window {start}..{end} contains no bars")

        rng = np.random.default_rng(self._symbol_seed(symbol))
        annual_bars = (
            bars_per_year(timeframe) if self.annual_bars is None else self.annual_bars
        )
        dt = 1.0 / annual_bars

        regime = self._regime_path(rng, n)
        mus = np.array([_REGIMES[_REGIME_NAMES[r]][0] for r in regime])
        sigmas = np.array([_REGIMES[_REGIME_NAMES[r]][1] for r in regime])

        shocks = rng.standard_normal(n)
        logret = (mus - 0.5 * sigmas**2) * dt + sigmas * np.sqrt(dt) * shocks
        close = 100.0 * np.exp(np.cumsum(logret))

        prev_close = np.concatenate(([100.0], close[:-1]))
        gap = rng.normal(0.0, 0.15, n) * sigmas * np.sqrt(dt)
        open_ = prev_close * np.exp(gap)
        span = np.abs(rng.normal(0.0, 1.0, n)) * sigmas * np.sqrt(dt) * close
        high = np.maximum(open_, close) + span * 0.6
        low = np.minimum(open_, close) - span * 0.6
        surprise = 1.0 + 3.0 * np.abs(logret) / (sigmas * dt**0.5)
        volume = np.round(1e6 * np.exp(rng.normal(0.0, 0.35, n)) * surprise)

        return Bars(
            symbol=symbol,
            timeframe=timeframe,
            event_time=np.array(stamps, dtype=np.int64),
            open=open_.astype(np.float64),
            high=high.astype(np.float64),
            low=low.astype(np.float64),
            close=close.astype(np.float64),
            volume=volume.astype(np.float64),
        )

    @staticmethod
    def _regime_path(rng: np.random.Generator, n: int) -> NDArray[np.int64]:
        """The hidden regime state per bar. Shared by ``load`` and ``regimes``
        so the two can never drift out of sync."""
        path = np.empty(n, dtype=np.int64)
        current = int(rng.integers(0, len(_REGIME_NAMES)))
        for i in range(n):
            if rng.random() < _REGIMES[_REGIME_NAMES[current]][2]:
                current = int(rng.integers(0, len(_REGIME_NAMES)))
            path[i] = current
        return path

    def regimes(
        self, symbol: str, start: datetime, end: datetime, timeframe: str = "1D"
    ) -> list[str]:
        """The generator's own regime labels — ground truth for regime tests."""
        n = len(self.load(symbol, start, end, timeframe))
        rng = np.random.default_rng(self._symbol_seed(symbol))
        return [_REGIME_NAMES[r] for r in self._regime_path(rng, n)]


# --------------------------------------------------------------------------- #
# CSV
# --------------------------------------------------------------------------- #


class CsvProvider:
    """Reads ``<root>/<timeframe>/<symbol>.csv``.

    Required header: ``timestamp,open,high,low,close,volume``. An optional
    ``available_time`` column carries point-in-time semantics for restated data.
    Timestamps are ISO-8601 and must be tz-aware.
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def path_for(self, symbol: str, timeframe: str) -> Path:
        return self.root / timeframe / f"{symbol}.csv"

    def load(self, symbol: str, start: datetime, end: datetime, timeframe: str = "1D") -> Bars:
        path = self.path_for(symbol, timeframe)
        if not path.exists():
            raise FileNotFoundError(f"no CSV for {symbol} {timeframe} at {path}")
        start, end = ensure_utc(start), ensure_utc(end)

        rows: list[tuple[int, int, float, float, float, float, float]] = []
        with path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                event = ensure_utc(datetime.fromisoformat(row["timestamp"]))
                if not (start <= event < end):
                    continue
                raw_avail = row.get("available_time") or row["timestamp"]
                avail = ensure_utc(datetime.fromisoformat(raw_avail))
                rows.append(
                    (
                        int(event.timestamp()),
                        int(avail.timestamp()),
                        float(row["open"]),
                        float(row["high"]),
                        float(row["low"]),
                        float(row["close"]),
                        float(row["volume"]),
                    )
                )
        if not rows:
            raise ValueError(f"{symbol}: no rows in {path} within {start}..{end}")
        rows.sort(key=lambda r: r[0])
        cols = list(zip(*rows, strict=True))
        return Bars(
            symbol=symbol,
            timeframe=timeframe,
            event_time=np.array(cols[0], dtype=np.int64),
            available_time=np.array(cols[1], dtype=np.int64),
            open=np.array(cols[2], dtype=np.float64),
            high=np.array(cols[3], dtype=np.float64),
            low=np.array(cols[4], dtype=np.float64),
            close=np.array(cols[5], dtype=np.float64),
            volume=np.array(cols[6], dtype=np.float64),
        )

    def write(self, bars: Bars) -> Path:
        path = self.path_for(bars.symbol, bars.timeframe)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                ["timestamp", "open", "high", "low", "close", "volume", "available_time"]
            )
            for i in range(len(bars)):
                writer.writerow(
                    [
                        datetime.fromtimestamp(int(bars.event_time[i]), tz=UTC).isoformat(),
                        f"{bars.open[i]:.6f}",
                        f"{bars.high[i]:.6f}",
                        f"{bars.low[i]:.6f}",
                        f"{bars.close[i]:.6f}",
                        f"{bars.volume[i]:.0f}",
                        datetime.fromtimestamp(int(bars.available_time[i]), tz=UTC).isoformat(),
                    ]
                )
        return path


# --------------------------------------------------------------------------- #
# Yahoo
# --------------------------------------------------------------------------- #


class YFinanceProvider:
    """Daily bars from Yahoo Finance. Requires the optional ``data`` extra."""

    _INTERVALS = {"1D": "1d", "1h": "1h", "1W": "1wk", "5m": "5m", "15m": "15m", "30m": "30m"}

    def load(self, symbol: str, start: datetime, end: datetime, timeframe: str = "1D") -> Bars:
        try:
            import yfinance
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                "YFinanceProvider needs the 'data' extra: uv sync --extra data"
            ) from exc
        if timeframe not in self._INTERVALS:
            raise ValueError(f"yfinance adapter does not map timeframe {timeframe!r}")

        start, end = ensure_utc(start), ensure_utc(end)
        frame = yfinance.download(
            symbol,
            start=start.date().isoformat(),
            end=end.date().isoformat(),
            interval=self._INTERVALS[timeframe],
            auto_adjust=True,
            progress=False,
        )
        if frame is None or frame.empty:
            raise ValueError(f"yfinance returned nothing for {symbol} {start}..{end}")
        if hasattr(frame.columns, "nlevels") and frame.columns.nlevels > 1:
            frame.columns = frame.columns.get_level_values(0)

        raw = frame.index
        index = raw.tz_localize("UTC") if raw.tz is None else raw.tz_convert("UTC")
        stamps = np.array([int(ts.timestamp()) for ts in index], dtype=np.int64)

        def col(name: str) -> Array:
            return np.asarray(frame[name].to_numpy(), dtype=np.float64).ravel()

        return Bars(
            symbol=symbol,
            timeframe=timeframe,
            event_time=stamps,
            open=col("Open"),
            high=col("High"),
            low=col("Low"),
            close=col("Close"),
            volume=col("Volume"),
        )
