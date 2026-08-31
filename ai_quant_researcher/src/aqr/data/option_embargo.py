"""The embargo, applied to option chains.

`data/embargo.py` does this for bars. This is the same three locks over a
different shape, and it exists as its own module because the shape is genuinely
different: a chain row is a contract on a session, not a bar in a series, so
`Bars` never sees it and `Bars.__post_init__` -- the sensor the whole seal rests
on -- never fires for it.

**That gap is the reason this file exists.** Caching option data without it
would leave a route into the process that the seal cannot see, which is exactly
the failure `aqr.seal` was built to make impossible.

Three locks, in decreasing order of how much they are relied on:

**Physical.** `data-options/` is truncated at the embargo on disk. The rows are
not there. Research code configured against that root cannot read them however
wrong it is, and :func:`audit_option_root` makes that checkable with `csv`
alone -- auditing the research root must not itself be the peek it looks for.

**Structural.** :class:`OptionChain` is the only container option rows enter,
and its `__post_init__` reports every session it holds to the seal. The sensor
is on the type rather than on any loader, for the reason
[seal.py](../seal.py) gives about `Bars`: a check on each loader covers the
loaders somebody remembered, and a check on the type covers the route nobody
anticipated, which is the only route that was ever going to be the problem.

**Procedural.** :func:`load_sealed` is the only way to read the untruncated
root, and it needs the sealed phase. The research loader cannot reach those
rows because they are not in its root.

## The boundary is the same one the bars use

`EMBARGO_START` is 2024-09-01 and the free option history runs to 2026-08-28,
so the reserved window is very nearly two years -- the same reservation, on the
same date, as the equity side. Sharing it is deliberate: two embargo dates in
one repository is a question every later reader has to re-answer, and a
strategy that ever reads both would have to reconcile them.

## What the canary is, here

`__CANARY__` is a symbol that exists only after the embargo, so nothing
legitimate loads it and any load is a peek by construction. It is written into
the **research** root only -- a tripwire placed where a peek cannot happen
catches nothing, and in the sealed root, where embargoed rows are expected, it
would only cry wolf.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from aqr.data.options_chain import CHAIN_COLUMNS, VOLATILITY_COLUMNS
from aqr.seal import CANARY_SYMBOL, EMBARGO_START, Phase
from aqr.seal import current as current_seal

__all__ = [
    "CHAIN_TABLE",
    "OPTIONS_RESEARCH_ROOT",
    "OPTIONS_SEALED_ROOT",
    "VOLATILITY_TABLE",
    "OptionCacheAudit",
    "OptionChain",
    "audit_option_root",
    "load_research",
    "load_sealed",
    "split_at_embargo",
    "write_option_canary",
]

CHAIN_TABLE = "option_chain"
VOLATILITY_TABLE = "volatility_history"


def _repo_root() -> Path:
    """The project directory, anchored on this file rather than on the cwd.

    A cache root that moves with the working directory is a cache root that can
    silently become the wrong one.
    """
    return Path(__file__).resolve().parents[3]


OPTIONS_RESEARCH_ROOT = _repo_root() / "data-options"
"""Truncated at the embargo. What a search may read."""

OPTIONS_SEALED_ROOT = _repo_root() / "data-options-sealed"
"""The full history. Reachable only in the sealed phase."""


@dataclass(frozen=True, slots=True)
class OptionChain:
    """One symbol's end-of-day option rows, and the seal's sensor.

    Deliberately not parsed into numbers. A chain row's meaning lives in the
    combination of expiry, strike and right, and a container that flattened it
    into arrays would have to decide a layout before anything is known about
    what will consume it. What this type owes the rest of the system is the one
    thing no consumer can be trusted to remember: telling the seal which
    sessions it holds.
    """

    symbol: str
    table: str
    columns: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    session: NDArray[np.int64] = field(repr=False)
    """Epoch seconds of each row's `date`, in the shape `Seal.observe` takes."""

    def __post_init__(self) -> None:
        if self.session.size != len(self.rows):
            raise ValueError(
                f"{self.symbol}: {self.session.size} sessions for {len(self.rows)} rows"
            )
        current_seal().observe(self.symbol, self.session)

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def dates(self) -> list[date]:
        return [datetime.fromtimestamp(int(t), tz=UTC).date() for t in self.session]

    def as_dicts(self) -> list[dict[str, str]]:
        return [dict(zip(self.columns, row, strict=True)) for row in self.rows]


def _epoch(day: str) -> int:
    return int(datetime.fromisoformat(day).replace(tzinfo=UTC).timestamp())


def _read(path: Path, symbol: str, table: str) -> OptionChain:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        columns = tuple(next(reader))
        at = columns.index("date")
        rows = [tuple(row) for row in reader if len(row) == len(columns)]
    session = np.array([_epoch(row[at]) for row in rows], dtype=np.int64)
    return OptionChain(
        symbol=symbol, table=table, columns=columns, rows=tuple(rows), session=session
    )


def load_research(
    symbol: str, *, table: str = CHAIN_TABLE, root: Path | None = None
) -> OptionChain:
    """Read from the truncated root. Safe in the research phase by construction.

    Safe because the rows are not on disk, not because this function is careful:
    the physical lock is the one that survives being called from somewhere
    nobody reviewed.
    """
    base = OPTIONS_RESEARCH_ROOT if root is None else Path(root)
    path = base / table / f"{symbol}.csv"
    if not path.exists():
        raise FileNotFoundError(f"no {table} cache for {symbol} at {path}")
    return _read(path, symbol, table)


def load_sealed(
    symbol: str, *, table: str = CHAIN_TABLE, root: Path | None = None
) -> OptionChain:
    """Read the untruncated root. Refused outside the sealed phase.

    Two locks rather than one, as `data/embargo.py` argues: the failure guarded
    against is unrecoverable, because once the embargoed sessions have informed
    a choice no later run can un-inform it.
    """
    seal = current_seal()
    if seal.phase is not Phase.SEALED:
        raise PermissionError(
            f"the sealed option cache is readable only in the sealed phase; "
            f"the seal is in {seal.phase.name}"
        )
    base = OPTIONS_SEALED_ROOT if root is None else Path(root)
    path = base / table / f"{symbol}.csv"
    if not path.exists():
        raise FileNotFoundError(f"no sealed {table} cache for {symbol} at {path}")
    return _read(path, symbol, table)


@dataclass(frozen=True, slots=True)
class OptionCacheAudit:
    """What an option cache root holds, in dates rather than in promises."""

    root: Path
    latest: date | None
    offenders: tuple[str, ...] = ()
    files: int = 0
    rows: int = 0
    canary_present: bool = False
    canary_tables: tuple[str, ...] = ()
    """Which tables the tripwire is armed in. A root-level boolean cannot say:
    one armed table must not read as cover for the other."""

    @property
    def clean(self) -> bool:
        return not self.offenders


def audit_option_root(
    root: Path | str, *, embargo: datetime = EMBARGO_START
) -> OptionCacheAudit:
    """Report the latest session in a root, and any symbol past the embargo.

    Reads with `csv` and never builds an :class:`OptionChain`, so auditing the
    research root is not itself the peek it is looking for. Otherwise nobody
    could ever check.
    """
    root = Path(root)
    cutoff = embargo.astimezone(UTC).date()
    latest: date | None = None
    offenders: list[str] = []
    canary_tables: list[str] = []
    files = 0
    rows = 0

    for path in sorted(root.rglob("*.csv")):
        files += 1
        symbol = path.stem
        if symbol == CANARY_SYMBOL:
            # The tripwire is *supposed* to sit in the research root holding
            # embargoed rows. Reported, never counted as contamination, and it
            # carries synthetic values so nothing real is exposed by it.
            table = path.parent.name
            if table not in canary_tables:
                canary_tables.append(table)
            continue
        newest: date | None = None
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                stamp = row.get("date")
                if not stamp:
                    continue
                rows += 1
                when = date.fromisoformat(stamp)
                if newest is None or when > newest:
                    newest = when
        if newest is None:
            continue
        if latest is None or newest > latest:
            latest = newest
        if newest >= cutoff and symbol not in offenders:
            offenders.append(symbol)

    return OptionCacheAudit(
        root=root,
        latest=latest,
        offenders=tuple(offenders),
        files=files,
        rows=rows,
        canary_present=bool(canary_tables),
        canary_tables=tuple(canary_tables),
    )


def write_option_canary(root: Path | str, table: str) -> Path:
    """Arm the tripwire: `<root>/<table>/__CANARY__.csv`.

    Synthetic throughout -- every price and greek 1, one strike, one expiry --
    so the file exposes nothing real, and every session stamped past the embargo
    so it cannot be mistaken for a series that simply ends early. Idempotent:
    the content is a pure function of the table, so a cache rebuild re-arms it
    byte for byte.
    """
    columns = CHAIN_COLUMNS if table == CHAIN_TABLE else _volatility_header()
    lines = [",".join(columns)]
    for offset in range(30):  # about a month past the embargo
        day = (EMBARGO_START + timedelta(days=offset)).date().isoformat()
        values = {"date": day, "act_symbol": CANARY_SYMBOL}
        if table == CHAIN_TABLE:
            values |= {"expiration": day, "call_put": "Put"}
        lines.append(",".join(str(values.get(name, 1)) for name in columns))
    path = Path(root) / table / f"{CANARY_SYMBOL}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _volatility_header() -> tuple[str, ...]:
    return VOLATILITY_COLUMNS


@dataclass(frozen=True, slots=True)
class SplitResult:
    symbol: str
    table: str
    research_rows: int
    sealed_rows: int
    research_last: str
    sealed_last: str


def split_at_embargo(
    source: Path,
    *,
    symbol: str,
    table: str,
    research_root: Path,
    sealed_root: Path,
    embargo: datetime = EMBARGO_START,
) -> SplitResult:
    """Write the truncated root and the full root from one cached file.

    The sealed root gets **everything**, the research root everything strictly
    before the embargo. Read with `csv` rather than through
    :class:`OptionChain`, because building one here would report the embargoed
    sessions to the seal and taint the very run that is arranging not to see
    them.
    """
    cutoff = embargo.astimezone(UTC).date()
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        at = header.index("date")
        rows = [row for row in reader if len(row) == len(header)]

    before = [row for row in rows if date.fromisoformat(row[at]) < cutoff]

    def write(root: Path, payload: list[list[str]]) -> str:
        path = root / table / f"{symbol}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as out:
            writer = csv.writer(out)
            writer.writerow(header)
            writer.writerows(payload)
        return max((row[at] for row in payload), default="")

    return SplitResult(
        symbol=symbol,
        table=table,
        research_rows=len(before),
        sealed_rows=len(rows),
        research_last=write(research_root, before),
        sealed_last=write(sealed_root, rows),
    )
