"""Multi-timeframe confluence — `specs/18-roadmap.md` F1, ADR 0024.

One pure function over a snapshot. `build_confluence` takes what the
per-timeframe engines have already decided — a `TrendState`, a pair of structure
labels and the levels in force, per timeframe — and says how much of it agrees.

What it deliberately does **not** do:

* compute anything. No indicator, no swing, no level is produced here; every
  input arrives already confirmed by the engine that owns it, on a bar that has
  already closed. Confluence adds no time semantics of its own and therefore no
  new way to look ahead (CLAUDE.md §12).
* feed anything back. `specs/07-levels.md` keeps a `multi_timeframe` component in
  the level-strength table at weight 0; this does not switch it on. A level's
  score still comes from its own timeframe (ADR 0024 D5).
* blend into one number. Trend agreement, structure agreement and level
  clustering are reported side by side, because a single blended score would be
  a weighting nobody chose and would hide which of the three was talking.

The scoring conventions are the ones the rest of this system already uses:
unreadable inputs leave both sides of the average rather than being read as
zero, contributions sum to the number they explain, and a component arguing the
other way contributes a negative number.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from decimal import Decimal
from types import MappingProxyType

from alphagate.core.errors import InvariantViolation
from alphagate.core.identifiers import LevelId, Ticker
from alphagate.core.levels import Level, LevelKind, LevelStatus
from alphagate.core.numeric import DOMAIN_CONTEXT
from alphagate.core.numeric import price as exact_price
from alphagate.core.structure import StructureLabel, structural_bias
from alphagate.core.time_model import Timeframe, ensure_utc
from alphagate.core.trend import TrendDirection, TrendPhase, TrendState

__all__ = [
    "Confluence",
    "ConfluenceConfig",
    "LevelCluster",
    "TimeframeInput",
    "TimeframeStance",
    "build_confluence",
]

_NEUTRAL_RUNG = 3
"""NEUTRAL's place on the seven-rung ladder in `specs/08-trend.md`."""

_RUNG_SPAN = 3.0
"""Rungs from NEUTRAL to either end, which is what scales a rung into [-1, 1]."""


@dataclasses.dataclass(frozen=True, slots=True)
class ConfluenceConfig:
    """Every policy this engine applies, in one place.

    `weights` is empty by default, meaning **every timeframe counts the same**.
    A weighting that favours the coarse timeframes is the obvious first thing to
    configure and is deliberately not the default: a policy that is on by default
    is a hidden constant, which is the argument `specs/07-levels.md` already makes
    about recency decay.
    """

    weights: Mapping[Timeframe, float] = dataclasses.field(default_factory=dict)
    default_weight: float = 1.0
    neutral_band: float = 0.2
    """How far from zero the alignment must be before it takes a side.

    Matches the `watch` threshold in `specs/08-trend.md`, so "leaning" means the
    same distance here as it does on one timeframe."""

    merge_distance_pct: Decimal = Decimal("0.0025")
    """How close two levels must be to join one cluster, as a fraction of price.

    A fraction rather than an ATR multiple because the levels being compared come
    from different timeframes, and there is no one ATR that belongs to all of
    them."""

    max_cluster_width_pct: Decimal = Decimal("0.0075")
    """The cap that makes single linkage safe (ADR 0009). Without it a chain of
    levels each close to the next merges into one arbitrarily wide band."""

    def __post_init__(self) -> None:
        if self.default_weight <= 0:
            raise InvariantViolation(f"default weight must be positive, got {self.default_weight}")
        for timeframe, weight in self.weights.items():
            if weight < 0:
                raise InvariantViolation(
                    f"{timeframe.code}: weight must not be negative, got {weight}"
                )
        if not 0.0 <= self.neutral_band < 1.0:
            raise InvariantViolation(f"neutral_band must fall in [0, 1), got {self.neutral_band}")
        for name in ("merge_distance_pct", "max_cluster_width_pct"):
            value = getattr(self, name)
            if value <= 0:
                raise InvariantViolation(f"{name} must be positive, got {value}")
        if self.max_cluster_width_pct < self.merge_distance_pct:
            raise InvariantViolation(
                "max_cluster_width_pct must be at least merge_distance_pct, or no two "
                "levels could ever merge"
            )
        object.__setattr__(self, "weights", MappingProxyType(dict(self.weights)))

    def weight_of(self, timeframe: Timeframe) -> float:
        return self.weights.get(timeframe, self.default_weight)


@dataclasses.dataclass(frozen=True, slots=True)
class TimeframeInput:
    """What one timeframe has to say, as its own engines left it."""

    timeframe: Timeframe
    trend: TrendState | None = None
    swing_high_label: StructureLabel | None = None
    swing_low_label: StructureLabel | None = None
    levels: tuple[Level, ...] = ()


@dataclasses.dataclass(frozen=True, slots=True)
class TimeframeStance:
    """One timeframe's row in the explanation.

    Every requested timeframe gets one, including the ones that could not be
    read: a row that vanished would be indistinguishable from a timeframe nobody
    asked about.
    """

    timeframe: Timeframe
    phase: TrendPhase
    weight: float
    trend_bias: float | None
    trend_contribution: float
    structure_bias: float | None
    structure_contribution: float
    as_of: datetime | None

    @property
    def direction(self) -> TrendDirection:
        return self.phase.direction


@dataclasses.dataclass(frozen=True, slots=True)
class LevelCluster:
    """Levels from one or more timeframes standing at the same price."""

    kind: LevelKind
    price: Decimal
    zone_low: Decimal
    zone_high: Decimal
    timeframes: tuple[Timeframe, ...]
    level_ids: tuple[LevelId, ...]

    @property
    def timeframe_count(self) -> int:
        """How many timeframes agree this price matters. One is just a level."""
        return len(self.timeframes)


@dataclasses.dataclass(frozen=True, slots=True)
class Confluence:
    """What the timeframes, together, currently say about one symbol."""

    symbol: Ticker
    as_of: datetime
    stances: tuple[TimeframeStance, ...]
    trend_alignment: float | None
    """Weighted mean of each timeframe's place on the ladder, in [-1, 1].

    `None` when no timeframe could be read at all — which is not zero, and the
    difference is the point."""

    trend_agreement: float | None
    """Share of the directional weight sitting on the leading side, in [0, 1].

    Separate from alignment because alignment is zero both when everything is
    neutral and when half is bullish and half bearish. `None` when nothing is
    taking a side, so there is nothing to agree about."""

    structure_alignment: float | None
    direction: TrendDirection
    confidence: float
    """Readable timeframes over requested ones — data sufficiency, not strength."""

    conflicts: tuple[Timeframe, ...]
    """Timeframes whose direction opposes the leading one, coarsest last."""

    clusters: tuple[LevelCluster, ...]


def build_confluence(
    symbol: Ticker,
    views: Iterable[TimeframeInput],
    *,
    as_of: datetime,
    config: ConfluenceConfig | None = None,
) -> Confluence:
    """Synthesise the timeframes for one symbol. Pure, and deterministic.

    Ordered by timeframe duration rather than by arrival, so two callers holding
    the same facts in different orders get the same answer — identity included.
    """
    settings = config or ConfluenceConfig()
    ordered = _ordered(symbol, views)
    moment = ensure_utc(as_of, field="as_of")

    stances = tuple(_stance(view, config=settings) for view in ordered)
    trend_alignment = _weighted_mean(
        (s.weight, s.trend_bias) for s in stances if s.trend_bias is not None
    )
    structure_alignment = _weighted_mean(
        (s.weight, s.structure_bias) for s in stances if s.structure_bias is not None
    )
    stances = tuple(
        _with_contributions(
            stance,
            trend_total=_weight_total(stances, "trend_bias"),
            structure_total=_weight_total(stances, "structure_bias"),
        )
        for stance in stances
    )

    readable = [s for s in stances if s.trend_bias is not None]
    direction = _direction(trend_alignment, band=settings.neutral_band)
    return Confluence(
        symbol=symbol,
        as_of=moment,
        stances=stances,
        trend_alignment=trend_alignment,
        trend_agreement=_agreement(readable),
        structure_alignment=structure_alignment,
        direction=direction,
        confidence=len(readable) / len(stances) if stances else 0.0,
        conflicts=_conflicts(readable, direction),
        clusters=_clusters(ordered, config=settings),
    )


# -- timeframes --------------------------------------------------------------


def _ordered(symbol: Ticker, views: Iterable[TimeframeInput]) -> tuple[TimeframeInput, ...]:
    collected = tuple(views)
    if not collected:
        raise InvariantViolation("confluence needs at least one timeframe to synthesise")

    seen: set[Timeframe] = set()
    for view in collected:
        if view.timeframe in seen:
            raise InvariantViolation(
                f"{view.timeframe.code} appears twice; a timeframe has one stance"
            )
        seen.add(view.timeframe)
        if view.trend is not None and view.trend.symbol != symbol:
            raise InvariantViolation(
                f"{view.trend.symbol}: trend state does not belong to {symbol}"
            )
        for level in view.levels:
            if level.symbol != symbol:
                raise InvariantViolation(f"{level.symbol}: level does not belong to {symbol}")
    return tuple(sorted(collected, key=lambda view: view.timeframe.nominal_duration))


def _stance(view: TimeframeInput, *, config: ConfluenceConfig) -> TimeframeStance:
    phase = view.trend.state if view.trend is not None else TrendPhase.UNKNOWN
    return TimeframeStance(
        timeframe=view.timeframe,
        phase=phase,
        weight=config.weight_of(view.timeframe),
        trend_bias=_ladder_bias(phase),
        trend_contribution=0.0,
        structure_bias=structural_bias(view.swing_high_label, view.swing_low_label),
        structure_contribution=0.0,
        as_of=view.trend.as_of if view.trend is not None else None,
    )


def _ladder_bias(phase: TrendPhase) -> float | None:
    """Where a phase sits on the ladder, scaled to [-1, 1].

    `specs/08-trend.md` already decided the rung *is* the trend's strength as far
    as anything downstream is concerned — it is what events are emitted on and
    what the margin and confirmation count protect. Reading the rung rather than
    the 0–100 score keeps confluence on that same scale instead of inventing a
    second one, and keeps a state whose score is still settling from moving the
    answer before its phase does.
    """
    rung = phase.rung
    return None if rung is None else (rung - _NEUTRAL_RUNG) / _RUNG_SPAN


def _weight_total(stances: Sequence[TimeframeStance], field: str) -> float:
    return math.fsum(s.weight for s in stances if getattr(s, field) is not None)


def _with_contributions(
    stance: TimeframeStance, *, trend_total: float, structure_total: float
) -> TimeframeStance:
    """Each row's share of the number it helps explain.

    They sum to the alignment by construction, and a timeframe arguing the other
    way carries a negative one — a statement about the market rather than a
    rounding error (`specs/08-trend.md`).
    """
    return dataclasses.replace(
        stance,
        trend_contribution=(
            0.0
            if stance.trend_bias is None or trend_total == 0.0
            else stance.weight * stance.trend_bias / trend_total
        ),
        structure_contribution=(
            0.0
            if stance.structure_bias is None or structure_total == 0.0
            else stance.weight * stance.structure_bias / structure_total
        ),
    )


def _weighted_mean(pairs: Iterable[tuple[float, float | None]]) -> float | None:
    collected = [(weight, value) for weight, value in pairs if value is not None]
    total = math.fsum(weight for weight, _ in collected)
    if not collected or total == 0.0:
        return None
    return math.fsum(weight * value for weight, value in collected) / total


def _direction(alignment: float | None, *, band: float) -> TrendDirection:
    if alignment is None:
        return TrendDirection.UNKNOWN
    if alignment >= band:
        return TrendDirection.BULLISH
    if alignment <= -band:
        return TrendDirection.BEARISH
    return TrendDirection.NEUTRAL


def _agreement(readable: Sequence[TimeframeStance]) -> float | None:
    """How much of the directional weight sits on the leading side.

    Only timeframes that take a side count, on both sides of the fraction. With
    none the answer is `None` rather than 1.0: a market where every timeframe is
    neutral is not one where they all agree on a direction, and reporting it as
    total agreement would be the loudest possible way to say nothing.
    """
    directional = [s for s in readable if s.direction.is_directional]
    if not directional:
        return None
    total = math.fsum(s.weight for s in directional)
    bullish = math.fsum(s.weight for s in directional if s.direction is TrendDirection.BULLISH)
    return max(bullish, total - bullish) / total


def _conflicts(
    readable: Sequence[TimeframeStance], direction: TrendDirection
) -> tuple[Timeframe, ...]:
    if not direction.is_directional:
        return ()
    return tuple(
        s.timeframe for s in readable if s.direction.is_directional and s.direction is not direction
    )


# -- levels ------------------------------------------------------------------


def _clusters(
    views: Sequence[TimeframeInput], *, config: ConfluenceConfig
) -> tuple[LevelCluster, ...]:
    """Levels standing at the same price, whatever timeframe produced them.

    Sorted single linkage with a width cap, over a total order (`price`,
    `timeframe`, `id`) so no decision can depend on the order the levels arrived
    in — the same shape `specs/07-levels.md` specifies for candidates inside one
    timeframe, and separate runs for support and resistance for the same reason:
    a level that was its own opposite would mean nothing.
    """
    in_force = [
        level for view in views for level in view.levels if level.status is LevelStatus.ACTIVE
    ]
    clusters: list[LevelCluster] = []
    for kind in (LevelKind.SUPPORT, LevelKind.RESISTANCE):
        members = sorted(
            (level for level in in_force if level.kind is kind),
            key=lambda level: (level.price, level.timeframe.nominal_duration, level.id),
        )
        clusters.extend(_link(members, kind=kind, config=config))
    return tuple(sorted(clusters, key=lambda cluster: (cluster.price, cluster.kind.value)))


def _link(
    members: Sequence[Level], *, kind: LevelKind, config: ConfluenceConfig
) -> list[LevelCluster]:
    groups: list[list[Level]] = []
    for level in members:
        if groups and _joins(groups[-1], level, config=config):
            groups[-1].append(level)
        else:
            groups.append([level])
    return [_fold(group, kind=kind) for group in groups]


def _joins(group: Sequence[Level], candidate: Level, *, config: ConfluenceConfig) -> bool:
    reference = group[-1].price
    if candidate.price - reference > reference * config.merge_distance_pct:
        return False
    # The cap is what makes single linkage safe: a chain of levels each within
    # the merge distance of the next would otherwise become one arbitrarily wide
    # band, silently turning "support at 100" into "support between 95 and 105".
    widest = candidate.price - group[0].price
    return widest <= group[0].price * config.max_cluster_width_pct


def _fold(group: Sequence[Level], *, kind: LevelKind) -> LevelCluster:
    """One cluster from its members.

    The centre is the plain mean of the member prices and the band is the union
    of their zones. Neither is weighted: strength is scored per timeframe against
    that timeframe's own evidence, and using it here would be a cross-timeframe
    weighting nobody chose (ADR 0024 D5).
    """
    total = group[0].price
    for level in group[1:]:
        total = DOMAIN_CONTEXT.add(total, level.price)
    centre = DOMAIN_CONTEXT.divide(total, Decimal(len(group)))
    return LevelCluster(
        kind=kind,
        price=exact_price(centre, field="price"),
        zone_low=min(level.effective_zone.low for level in group),
        zone_high=max(level.effective_zone.high for level in group),
        timeframes=tuple(
            sorted(
                {level.timeframe for level in group},
                key=lambda timeframe: timeframe.nominal_duration,
            )
        ),
        level_ids=tuple(level.id for level in group),
    )
