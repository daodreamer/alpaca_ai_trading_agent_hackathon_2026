"""The two cache roots, and the wrappers that keep them apart.

``aqr.seal`` detects a peek. This module makes the common ones impossible, which
is strictly better: an alarm that fires after a campaign has finished has still
cost the campaign, and the decision it contaminated has already been made.

Three locks, in decreasing order of how much they are relied on:

**Physical.** ``data-sp500/`` is truncated at the embargo on disk. The rows
are not there. Search code configured against that root cannot read them however
wrong it is, and :func:`audit_cache_root` makes that checkable without loading
anything into the process doing the checking.

**Structural.** :class:`ResearchProvider` clamps the end date *before* the
underlying call, not after. Truncating the response is not enough when the
response crosses a network: a vendor request for the embargoed years is logged on
the vendor's side, is paid for, and on a delayed feed can arrive after the
truncation decision was taken.

**Procedural.** :class:`SealedProvider` is the only thing that returns the
embargoed years, and it needs both an explicit :class:`SealToken` and the sealed
phase. Two locks rather than one because the failure they guard is
unrecoverable: once the embargoed years have informed a choice, no later run can
un-inform it.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

from aqr.data.bars import Bars, bar_duration, ensure_utc
from aqr.data.providers import Provider
from aqr.seal import CANARY_SYMBOL, EMBARGO_START, LoadRecord, Phase, Seal
from aqr.seal import current as current_seal

__all__ = [
    "SP500_RESEARCH_ROOT",
    "SP500_SEALED_ROOT",
    "CacheAudit",
    "ResearchProvider",
    "SealToken",
    "SealedProvider",
    "audit_cache_root",
    "write_canary",
]


def _repo_root() -> Path:
    """The project directory, anchored on this file rather than on the cwd.

    A cache root that moves with the working directory is a cache root that can
    silently become the wrong one.
    """
    return Path(__file__).resolve().parents[3]


SP500_RESEARCH_ROOT = _repo_root() / "data-sp500"
"""The point-in-time S&P 500, truncated at the embargo. What every campaign reads."""

SP500_SEALED_ROOT = _repo_root() / "data-sp500-sealed"
"""The same tickers, full history including the embargoed years. Read once, by
one process. The pair audited by ``aqr seal-check``.

A research/sealed pair rather than a single directory: the audit reports per
root, and a root that mixed truncated and full series would have nothing useful
to say about either."""


@dataclass(frozen=True, slots=True)
class SealToken:
    """Permission to read the embargoed years.

    Deliberately trivial to construct and deliberately hard to reach: the lock is
    not cryptographic, it is that
    ``test_no_module_outside_the_embargo_layer_constructs_a_seal_token`` fails the
    build if the name appears anywhere but here and the sealed entry point. A
    lock the research loop can pick is not a lock, and the import graph is what
    does the picking.
    """


class ResearchProvider:
    """Wraps any provider so that it cannot return embargoed bars.

    A request that overruns the embargo is clamped rather than refused. Refusing
    would kill a forty-hypothesis campaign over a request that was entirely
    reasonable to make -- "everything up to today" is what a researcher means and
    the embargo is what the system means by today.
    """

    def __init__(self, inner: Provider, *, label: str = "") -> None:
        self._inner = inner
        self._label = label or type(inner).__name__

    def dataset_version(self, timeframe: str) -> str:
        version: str = self._inner.dataset_version(timeframe)  # type: ignore[attr-defined]
        return f"{version}@embargo-{EMBARGO_START.date().isoformat()}"

    def load(
        self, symbol: str, start: datetime, end: datetime, timeframe: str = "1D"
    ) -> Bars:
        seal: Seal = current_seal()
        seal.require(Phase.RESEARCH)
        start, end = ensure_utc(start), ensure_utc(end)
        capped = min(end, EMBARGO_START)

        if capped <= start:
            # Entirely inside the embargo. A mistake, but a recoverable one: the
            # campaign gets nothing and the ledger records that it got nothing,
            # which is more useful than a traceback.
            bars = _empty(symbol, timeframe)
        else:
            bars = self._inner.load(symbol, start, capped, timeframe)

        seal.record_load(
            LoadRecord(
                source=f"research:{self._label}",
                symbol=symbol,
                requested_start=int(start.timestamp()),
                requested_end=int(end.timestamp()),
                rows=len(bars),
                max_event_time=int(bars.event_time[-1]) if len(bars) else 0,
            )
        )
        return bars


class SealedProvider:
    """The only route to the embargoed years.

    Needs a token *and* the sealed phase. The phase check is what stops a token
    that leaked into research code from being enough on its own.
    """

    def __init__(self, inner: Provider, *, token: SealToken, label: str = "") -> None:
        if not isinstance(token, SealToken):
            raise TypeError("SealedProvider requires an explicit SealToken")
        self._inner = inner
        self._label = label or type(inner).__name__

    def dataset_version(self, timeframe: str) -> str:
        return str(self._inner.dataset_version(timeframe))  # type: ignore[attr-defined]

    def load(
        self, symbol: str, start: datetime, end: datetime, timeframe: str = "1D"
    ) -> Bars:
        seal: Seal = current_seal()
        seal.require(Phase.SEALED)
        start, end = ensure_utc(start), ensure_utc(end)
        bars = self._inner.load(symbol, start, end, timeframe)
        seal.record_load(
            LoadRecord(
                source=f"sealed:{self._label}",
                symbol=symbol,
                requested_start=int(start.timestamp()),
                requested_end=int(end.timestamp()),
                rows=len(bars),
                max_event_time=int(bars.event_time[-1]) if len(bars) else 0,
            )
        )
        return bars


def _empty(symbol: str, timeframe: str) -> Bars:
    import numpy as np

    return Bars(
        symbol=symbol,
        timeframe=timeframe,
        event_time=np.empty(0, dtype=np.int64),
        open=np.empty(0),
        high=np.empty(0),
        low=np.empty(0),
        close=np.empty(0),
        volume=np.empty(0),
    )


@dataclass(frozen=True, slots=True)
class CacheAudit:
    """What a cache root actually holds, in timestamps rather than in promises."""

    root: Path
    latest: datetime | None
    offenders: tuple[str, ...] = field(default=())
    files: int = 0
    canary_present: bool = False
    """Whether the tripwire is armed in this root. Reported, never an offence."""
    canary_timeframes: tuple[str, ...] = field(default=())
    """Which timeframes the tripwire is armed in. A root-level boolean cannot
    say: one armed timeframe must not read as cover for the others."""

    @property
    def clean(self) -> bool:
        return not self.offenders


def audit_cache_root(root: Path | str, *, embargo: datetime = EMBARGO_START) -> CacheAudit:
    """Report the latest bar in a cache root, and any symbol past the embargo.

    Reads timestamps with :mod:`csv` and never builds a :class:`Bars`, so
    auditing the research root is not itself the peek it is looking for.
    Otherwise nobody could ever check.
    """
    root = Path(root)
    latest: datetime | None = None
    offenders: list[str] = []
    files = 0
    canary_tfs: list[str] = []

    for path in sorted(root.rglob("*.csv")):
        files += 1
        symbol = path.stem
        if symbol == CANARY_SYMBOL:
            # The tripwire is *supposed* to sit in the research root holding
            # embargoed rows -- a canary placed anywhere a peek cannot happen
            # catches nothing. It is reported, not counted as contamination, and
            # it carries synthetic prices so nothing real is exposed by it.
            timeframe = path.parent.name
            if timeframe not in canary_tfs:
                canary_tfs.append(timeframe)
            continue
        newest: datetime | None = None
        with path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                stamp = row.get("timestamp")
                if not stamp:
                    continue
                when = datetime.fromisoformat(stamp).astimezone(UTC)
                if newest is None or when > newest:
                    newest = when
        if newest is None:
            continue
        if latest is None or newest > latest:
            latest = newest
        if newest >= embargo:
            offenders.append(symbol)

    return CacheAudit(
        root=root,
        latest=latest,
        offenders=tuple(offenders),
        files=files,
        canary_present=bool(canary_tfs),
        canary_timeframes=tuple(canary_tfs),
    )


def write_canary(root: Path | str, timeframe: str) -> Path:
    """Arm the tripwire: write ``<root>/<timeframe>/__CANARY__.csv``.

    The canary is a symbol that exists only after the embargo, so nothing
    legitimate ever loads it and any load is a peek by construction --
    ``Bars.__post_init__`` taints the seal on sight. It belongs in the
    research root, because a tripwire placed where a peek cannot happen
    catches nothing, and never in the sealed root, where embargoed rows are
    expected and it would only cry wolf.

    The bars are synthetic -- every price 1, volume 100 -- so the file
    exposes nothing real, and stamped past the embargo so it cannot be
    mistaken for a truncated series that simply ends early. Timestamps are
    spaced by the timeframe's nominal bar duration and not aligned to real
    sessions; the tripwire only needs to sit inside the embargo window.
    Idempotent: the content is a pure function of the timeframe, so a cache
    rebuild re-arms it byte for byte.
    """
    step = bar_duration(timeframe)
    rows = int(timedelta(days=30) / step)  # about a month past the embargo
    lines = ["timestamp,open,high,low,close,volume,available_time"]
    for i in range(rows):
        stamp = (EMBARGO_START + i * step).isoformat()
        lines.append(f"{stamp},1,1,1,1,100,{stamp}")
    path = Path(root) / timeframe / f"{CANARY_SYMBOL}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
