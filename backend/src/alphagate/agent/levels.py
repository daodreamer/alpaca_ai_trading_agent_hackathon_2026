"""Levels and multi-timeframe confluence — specs/05 D2's last two fields.

Two folds over the same bars, both delegating to engines the core already owns
(adr/0001). Nothing here computes a swing, a zone or a score; it decides what
evidence to hand the engines and how to say so when there isn't enough.

**Levels** come from confirmed swings, plus the session and year extremes the
bar history makes available. The engine clusters, prices, scores and dedupes;
this module's only real decision is which candidates exist and what `atr` to
express zone widths in.

**Confluence** is the same trend fold on several timeframes, put side by side.
It is the field most likely to be faked and hardest to fake usefully: reporting
"the timeframes agree" from one timeframe is not a weaker claim, it is a
different one. So `read_confluence` is given a mapping of timeframe to bars and
reports on exactly the timeframes it was given — a caller with only daily bars
gets a one-row confluence and a `confidence` that says so, never an inferred
hourly view.

The near-level distance the model actually uses is computed here too, because
"support 1.2% below spot" is the form the question takes and a list of prices is
not. Everything is `None` when unmeasurable, as everywhere else in this layer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Final

from alphagate.agent.trend import read_trend
from alphagate.core.bar import Bar
from alphagate.core.confluence import Confluence, TimeframeInput, build_confluence
from alphagate.core.errors import InvariantViolation
from alphagate.core.identifiers import Ticker
from alphagate.core.level_engine import LevelCandidate, build_levels
from alphagate.core.levels import Level, LevelKind, LevelSource
from alphagate.core.structure import StructureEngine
from alphagate.core.structure.model import SwingKind
from alphagate.core.time_model import Timeframe

__all__ = [
    "NEAR_LEVELS",
    "LevelRead",
    "describe_level",
    "read_confluence",
    "read_levels",
]

NEAR_LEVELS: Final = 4
"""How many levels either side of spot reach the prompt.

Four. The model is choosing between structures whose breakevens sit within an
ATR or two of spot; a level six percent away does not bear on that choice, and
every extra row is prompt the model has to ignore correctly."""


@dataclass(frozen=True, slots=True)
class LevelRead:
    """Nearby levels, or an account of why there are none."""

    levels: tuple[Level, ...]
    reason: str

    @property
    def is_measured(self) -> bool:
        return bool(self.levels)

    def nearest(self, spot: Decimal, kind: LevelKind) -> Level | None:
        """The closest level of one kind on the correct side of spot.

        Support above spot is not support, and resistance below it is not
        resistance — a level that price has already passed through is evidence
        about the past. Filtering by side here keeps the model from reading a
        broken level as an intact one.
        """
        candidates = [
            level
            for level in self.levels
            if level.kind is kind
            and (level.price <= spot if kind is LevelKind.SUPPORT else level.price >= spot)
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda level: abs(level.price - spot))


def read_levels(
    bars: Sequence[Bar],
    *,
    symbol: Ticker,
    as_of: datetime,
    atr: float | None = None,
    timeframe: Timeframe = Timeframe.D1,
) -> LevelRead:
    """Confirmed swings and range extremes, clustered and scored by the engine.

    `atr` is what zone widths and merge distances are expressed in; passing
    `None` is legitimate while it warms, and the engine widens its own defaults
    rather than pretending to a precision it does not have.
    """
    if not bars:
        return LevelRead((), "no bars")

    structure = StructureEngine()
    for bar in bars:
        if bar.is_final:
            try:
                structure.update(bar)
            except InvariantViolation as exc:
                return LevelRead((), f"series break: {exc}")

    candidates: list[LevelCandidate] = [
        LevelCandidate(
            price=swing.price,
            kind=(
                LevelKind.RESISTANCE if swing.kind is SwingKind.HIGH else LevelKind.SUPPORT
            ),
            source=(
                LevelSource.SWING_HIGH if swing.kind is SwingKind.HIGH else LevelSource.SWING_LOW
            ),
            observed_at_utc=swing.bar_start_utc,
        )
        for swing in structure.swings
    ]

    # The range extremes of the window. Cheap, and they are the levels a human
    # would name first — which matters when the journal has to be readable.
    highest = max(bars, key=lambda bar: bar.high)
    lowest = min(bars, key=lambda bar: bar.low)
    candidates.append(
        LevelCandidate(
            price=highest.high,
            kind=LevelKind.RESISTANCE,
            source=LevelSource.YEAR_HIGH,
            observed_at_utc=highest.start_time_utc,
        )
    )
    candidates.append(
        LevelCandidate(
            price=lowest.low,
            kind=LevelKind.SUPPORT,
            source=LevelSource.YEAR_LOW,
            observed_at_utc=lowest.start_time_utc,
        )
    )

    levels = build_levels(
        candidates,
        symbol=symbol,
        timeframe=timeframe,
        bars=bars,
        atr=atr,
        as_of=as_of,
    )
    if not levels:
        return LevelRead((), f"no levels from {len(candidates)} candidates")
    return LevelRead(levels, "measured")


def read_confluence(
    bars_by_timeframe: Mapping[Timeframe, Sequence[Bar]],
    *,
    symbol: Ticker,
    as_of: datetime,
) -> Confluence | None:
    """Fold each timeframe, then ask the core how much they agree.

    Returns `None` for an empty mapping rather than an empty report: "no
    timeframes were read" and "the timeframes disagree" are different facts, and
    a `Confluence` with no stances would render as the second.

    A timeframe whose trend could not be measured is still passed in, with
    `trend=None`. The core is explicit that a row which vanished would be
    indistinguishable from a timeframe nobody looked at — so it is handed the
    absence rather than the omission.
    """
    if not bars_by_timeframe:
        return None

    views = [
        TimeframeInput(timeframe=timeframe, trend=read_trend(bars).state)
        for timeframe, bars in sorted(
            bars_by_timeframe.items(), key=lambda pair: pair[0].nominal_duration
        )
    ]
    return build_confluence(symbol, views, as_of=as_of)


def describe_level(level: Level, spot: Decimal) -> dict[str, object]:
    """One level, as the model should see it: a distance, not just a price."""
    distance = (level.price - spot) / spot * 100 if spot > 0 else Decimal(0)
    return {
        "kind": level.kind.value,
        "price": str(level.price),
        "distance_pct": f"{distance:+.2f}",
        "strength_0_100": round(float(level.strength), 1),
        "sources": sorted({source.value for source in level.sources}),
    }
