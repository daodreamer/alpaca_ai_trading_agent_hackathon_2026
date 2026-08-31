"""End-of-day option chains, from DoltHub's free ``post-no-preference/options``.

The equities side of this project pulls bars from Alpaca or IBKR. Neither will
sell this account an option chain: Alpaca gates historical option data behind an
OPRA agreement, and IBKR serves no historical option data at all -- probed
against a live TWS, ``reqHistoricalData`` on a listed SPY put returns zero bars
for TRADES, MIDPOINT and BID_ASK alike.

DoltHub hosts a public database that does, at no cost and with no key:

    option_chain        date, act_symbol, expiration, strike, call_put,
                        bid, ask, vol, delta, gamma, theta, vega, rho
    volatility_history  date, act_symbol, iv_current, iv_year_high, iv_year_low,
                        hv_current, ... and the week/month-ago columns

``volatility_history`` is the more valuable of the two per byte: ``iv_current``
against ``iv_year_low .. iv_year_high`` **is** IV rank, which the options agent
has never been able to compute and which its strategy conditions on.

Why this is two phases
----------------------

The CSV endpoint generates its answer on the fly. It returns no
``content-length``, ``HEAD`` times out, and it honours no ``Range`` header --
all three confirmed against the live service. So a transfer that dies at minute
fifty cannot be resumed, and a bug in the parsing would otherwise cost a second
full download.

Phase one therefore does nothing but move bytes to disk, gzipped. Phase two
reads that local file and splits it, and may be re-run as often as the splitting
logic changes. The raw file is also the audit artefact: it is what the vendor
actually said, before this module had an opinion about it.

Why not the SQL API
-------------------

There is one, and it is unusable for bulk. Aggregates (``count``, ``min``,
``max``, ``group by``) exceed the server's query deadline on a table this size,
and a plain ``select`` silently returns **zero rows** above roughly a thousand
-- not an error, an empty success. A puller built on it would produce a short
cache that looks exactly like a complete one. Point lookups for one symbol on
one date do work, and are what a coverage probe should use.

What this data is not
---------------------

**End of day only.** One snapshot per session, so ``1D`` research is possible
and ``1h``/``4h`` are not. Intraday option data starts at about $40/month
elsewhere.

**A subset of each chain.** Roughly 200 contracts per symbol per session,
against the 13,514 CBOE lists for SPY. The strike ladder around the money is
there; the far wings and most expirations are not. Adequate for delta-targeted
structures, inadequate for surface work, and the difference is not visible from
inside a row.
"""

from __future__ import annotations

import csv
import gzip
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

__all__ = [
    "CHAIN_COLUMNS",
    "DOLTHUB_REPO",
    "VOLATILITY_COLUMNS",
    "DownloadResult",
    "SplitResult",
    "csv_url",
    "download_table",
    "read_header",
    "split_table",
]

DOLTHUB_REPO = "post-no-preference/options"
"""The database. Public, unauthenticated, and not ours -- pinned by name so a
reader can go and look at the same thing this cache came from."""

CHAIN_COLUMNS = (
    "date",
    "act_symbol",
    "expiration",
    "strike",
    "call_put",
    "bid",
    "ask",
    "vol",
    "delta",
    "gamma",
    "theta",
    "vega",
    "rho",
)
"""``option_chain``'s header, as the service emits it.

Checked on split rather than assumed. A vendor that added a column between two
existing ones would otherwise shift every field one place and produce a cache
full of plausible, wrong numbers."""

VOLATILITY_COLUMNS = ("date", "act_symbol", "iv_current", "iv_year_high", "iv_year_low")
"""The subset of ``volatility_history`` that IV rank needs.

The table carries more -- the hv series, the week- and month-ago columns -- and
the splitter keeps whatever the header actually contains. This tuple is what is
*required* for the cache to be worth having."""


def csv_url(table: str, *, repo: str = DOLTHUB_REPO, branch: str = "master") -> str:
    return f"https://www.dolthub.com/csv/{repo}/{branch}/{table}"


class ByteStream(Protocol):
    """The slice of an HTTP response this module needs.

    A Protocol so the tests never open a socket -- the same reason
    ``data/providers.py`` has one.
    """

    def iter_bytes(self) -> Iterator[bytes]: ...


class StreamOpener(Protocol):
    """Something that yields a context-managed :class:`ByteStream` for a URL."""

    def __call__(self, url: str) -> Any: ...


@dataclass(frozen=True, slots=True)
class DownloadResult:
    path: Path
    bytes_read: int
    seconds: float

    @property
    def rate_mb_s(self) -> float:
        return (self.bytes_read / 1e6) / self.seconds if self.seconds > 0 else 0.0


@dataclass(slots=True)
class SplitResult:
    """What a split produced, and what it passed over."""

    rows_read: int = 0
    rows_kept: int = 0
    symbols: set[str] = field(default_factory=set)
    unknown_symbols: set[str] = field(default_factory=set)
    """Symbols in the vendor file that the requested universe did not name.

    Counted rather than dropped silently, because "the vendor covers less than
    we hoped" and "our universe spells symbols differently" produce the same
    small file count and want opposite responses."""

    malformed_rows: int = 0
    """Lines whose field count did not match the header. Skipped, and counted:
    a truncated final line is normal, and ten thousand of them is a corrupt
    download that must not pass for a good one."""

    first_date: str = ""
    last_date: str = ""

    def observe_date(self, day: str) -> None:
        if not self.first_date or day < self.first_date:
            self.first_date = day
        if day > self.last_date:
            self.last_date = day


def download_table(
    table: str,
    dest: Path,
    *,
    open_stream: StreamOpener,
    on_progress: Callable[[int, float], None] | None = None,
    progress_every: int = 50_000_000,
) -> DownloadResult:
    """Phase one: move bytes to ``dest``, gzipped. No parsing, no opinions.

    Written to a ``.part`` file and renamed only on a clean finish, so an
    interrupted transfer cannot be mistaken for a complete one by the next run.
    There is no resume -- the service honours no ``Range`` -- so a failure means
    starting over, and the rename is what makes that unambiguous rather than a
    half file the splitter would happily read.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".part")
    started = time.monotonic()
    read = 0
    marker = progress_every
    with gzip.open(partial, "wb") as out, open_stream(csv_url(table)) as response:
        for chunk in response.iter_bytes():
            out.write(chunk)
            read += len(chunk)
            if on_progress is not None and read >= marker:
                on_progress(read, time.monotonic() - started)
                marker += progress_every
    partial.replace(dest)
    return DownloadResult(path=dest, bytes_read=read, seconds=time.monotonic() - started)


def split_table(
    source: Path,
    root: Path,
    *,
    symbols: Iterable[str],
    required_columns: tuple[str, ...],
    flush_rows: int = 2_000_000,
    on_progress: Callable[[SplitResult], None] | None = None,
    progress_every: int = 10_000_000,
) -> SplitResult:
    """Phase two: ``<root>/<symbol>.csv``, one file per symbol, header preserved.

    Local and repeatable. Reads the gzipped vendor file, keeps the rows whose
    ``act_symbol`` the universe named, and writes them under ``root`` in the
    per-symbol layout ``CsvProvider`` already uses for bars.

    **Buffered, not one handle per symbol.** The vendor file is ordered by date,
    so one symbol's rows are scattered across the whole of it. Holding six
    hundred file handles open for an hour to avoid re-opening them trades a
    bounded cost for an unbounded one. Rows accumulate in memory and every
    symbol is appended at once when the buffer fills, which fixes both the
    handle count and the memory ceiling.

    A symbol's file is truncated on its first flush and appended to afterwards,
    so re-running replaces a cache rather than doubling it.
    """
    wanted = {s.upper() for s in symbols}
    root.mkdir(parents=True, exist_ok=True)
    result = SplitResult()
    buffer: dict[str, list[list[str]]] = {}
    buffered = 0
    opened: set[str] = set()
    marker = progress_every

    with gzip.open(source, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError(f"{source.name} is empty") from None
        missing = [column for column in required_columns if column not in header]
        if missing:
            raise ValueError(
                f"{source.name} is missing {missing}; the vendor's schema has moved and "
                f"this cache would be silently misaligned. Header was: {header}"
            )
        symbol_at = header.index("act_symbol")
        date_at = header.index("date")

        def flush() -> None:
            nonlocal buffered
            for symbol, rows in buffer.items():
                path = root / f"{symbol}.csv"
                first = symbol not in opened
                with path.open("w" if first else "a", newline="", encoding="utf-8") as out:
                    writer = csv.writer(out)
                    if first:
                        writer.writerow(header)
                        opened.add(symbol)
                    writer.writerows(rows)
            buffer.clear()
            buffered = 0

        for row in reader:
            result.rows_read += 1
            if len(row) != len(header):
                result.malformed_rows += 1
                continue
            symbol = row[symbol_at]
            if symbol not in wanted:
                result.unknown_symbols.add(symbol)
                continue
            buffer.setdefault(symbol, []).append(row)
            buffered += 1
            result.rows_kept += 1
            result.symbols.add(symbol)
            result.observe_date(row[date_at])
            if buffered >= flush_rows:
                flush()
            if on_progress is not None and result.rows_read >= marker:
                on_progress(result)
                marker += progress_every
        flush()
    return result


def read_header(source: Path) -> list[str]:
    """The vendor header of a downloaded file, without reading the rest."""
    with gzip.open(source, "rt", encoding="utf-8", newline="") as handle:
        try:
            return next(csv.reader(handle))
        except StopIteration:
            raise ValueError(f"{source.name} is empty") from None
