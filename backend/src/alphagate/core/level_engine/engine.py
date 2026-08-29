"""Clustering, scoring and level assembly — specs/07-levels.md, ADR 0009.

Pure functions over a snapshot. Structure and indicators are stateful because
they are recursive; levels are not, so the same candidates and configuration
always produce the same levels — which is what "deterministic levels" in
`specs/18-roadmap.md` has to mean if it is to be testable.

A level is technical evidence at a price, ranked. It is never advice
(CLAUDE.md §15).
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from decimal import Decimal

from alphagate.core.bar import Bar
from alphagate.core.errors import InvariantViolation
from alphagate.core.identifiers import LevelId, Ticker
from alphagate.core.level_engine.model import LevelCandidate, LevelConfig, LifecycleConfig
from alphagate.core.level_engine.volume_profile import VolumeProfile
from alphagate.core.levels import Level, LevelKind, LevelStatus, Zone
from alphagate.core.numeric import DOMAIN_CONTEXT, format_exact, from_approximate
from alphagate.core.numeric import price as exact_price
from alphagate.core.time_model import Timeframe, ensure_utc

__all__ = [
    "advance",
    "bars_since_touch",
    "build_levels",
    "cluster_candidates",
    "count_touch_episodes",
    "decay",
    "flip",
    "invalidate",
    "level_id",
    "score_cluster",
]

_MIN_ZONE_PRICE = Decimal("0.00000001")
"""A zone never reaches zero: prices are positive, and so are zone bounds."""


# -- clustering -------------------------------------------------------------


def cluster_candidates(
    candidates: Iterable[LevelCandidate],
    *,
    config: LevelConfig,
    atr: Decimal | None,
) -> tuple[tuple[LevelCandidate, ...], ...]:
    """Group nearby candidates, deterministically (ADR 0009 D4).

    Sorted single linkage with a width cap. Support and resistance never mix:
    a swing low and a swing high at one price are two pieces of evidence, and a
    level that was its own opposite would mean nothing.
    """
    clusters: list[tuple[LevelCandidate, ...]] = []

    for kind in (LevelKind.SUPPORT, LevelKind.RESISTANCE):
        ordered = sorted(
            (candidate for candidate in candidates if candidate.kind is kind),
            key=lambda candidate: candidate.sort_key,
        )
        current: list[LevelCandidate] = []

        for candidate in ordered:
            if current and _starts_new_cluster(current, candidate, config=config, atr=atr):
                clusters.append(tuple(current))
                current = []
            current.append(candidate)

        if current:
            clusters.append(tuple(current))

    return tuple(sorted(clusters, key=lambda group: (group[0].price, group[0].kind.value)))


def _starts_new_cluster(
    current: Sequence[LevelCandidate],
    candidate: LevelCandidate,
    *,
    config: LevelConfig,
    atr: Decimal | None,
) -> bool:
    gap = candidate.price - current[-1].price
    merge = config.merge_distance.resolve(price=candidate.price, atr=atr)
    if merge is not None and gap > merge:
        return True

    cap = config.max_cluster_width.resolve(price=candidate.price, atr=atr)
    # The cap is what makes single linkage safe: without it a chain of candidates
    # each within `merge` of the next collapses into one arbitrarily wide zone.
    return cap is not None and candidate.price - current[0].price > cap


# -- scoring ----------------------------------------------------------------


def count_touch_episodes(zone: Zone, bars: Iterable[Bar]) -> int:
    """How many times price *visited* the zone, not how many bars sat in it.

    A maximal run of consecutive touching bars is one episode. Ten bars resting
    inside a zone is one test of it; counting bars would make strength a
    function of the timeframe rather than of the market (ADR 0009 D6).
    """
    episodes = 0
    inside = False
    for bar in bars:
        touching = bar.low <= zone.high and bar.high >= zone.low
        if touching and not inside:
            episodes += 1
        inside = touching
    return episodes


def score_cluster(
    cluster: Sequence[LevelCandidate],
    *,
    zone: Zone,
    bars: Sequence[Bar],
    config: LevelConfig,
    recency: float,
    profile: VolumeProfile | None = None,
) -> tuple[float, float, tuple[tuple[str, float], ...]]:
    """Return `(strength, confidence, breakdown)` for one cluster.

    A component that cannot be evaluated is `None`, not zero, and is left out of
    both the weighted sum and its denominator; `confidence` then reports how much
    of the requested evidence was actually available (ADR 0009 D6/D7).
    """
    distinct = {candidate.source for candidate in cluster}
    touches = count_touch_episodes(zone, bars) if bars else None

    components: dict[str, float | None] = {
        "source_quality": _mean(config.quality_of(source) for source in distinct),
        "touches": None if touches is None else _saturate(touches, config.touch_saturation),
        "recency": recency,
        "confluence": _saturate(len(distinct), config.confluence_saturation),
        # Computable since F2, and still weight 0 by default: switching it on
        # rescales every level's strength, which is a decision for whoever runs
        # the instance rather than a side effect of the data becoming available
        # (ADR 0025 D8). Without a profile it stays undefined, exactly as before.
        "volume_context": None if profile is None else profile.context(zone),
        # Declared, not computed: reported by `alphagate.core.confluence` and
        # deliberately not fed back (ADR 0024 D5).
        "multi_timeframe": None,
    }

    weights = config.weights.as_mapping()
    requested = [name for name, weight in weights.items() if weight > 0]
    available = [name for name in requested if components[name] is not None]
    if not requested:  # pragma: no cover — StrengthWeights forbids it
        raise InvariantViolation("at least one strength weight must be positive")

    confidence = len(available) / len(requested)
    if not available:
        return 0.0, confidence, ()

    total_weight = math.fsum(weights[name] for name in available)
    breakdown = tuple(
        (name, 100.0 * weights[name] * _value(components[name]) / total_weight)
        for name in available
    )
    strength = math.fsum(contribution for _, contribution in breakdown)
    return min(strength, 100.0), confidence, breakdown


def _value(component: float | None) -> float:
    assert component is not None  # noqa: S101 — filtered by `available`
    return component


def _mean(values: Iterable[float]) -> float:
    collected = list(values)
    return math.fsum(collected) / len(collected) if collected else 0.0


def _saturate(count: int, saturation: int) -> float:
    return min(count, saturation) / saturation


# -- assembly ---------------------------------------------------------------


def level_id(*, symbol: Ticker, timeframe: Timeframe, kind: LevelKind, zone: Zone) -> LevelId:
    """Identity derived from the evidence, not assigned (ADR 0009 D2).

    Two rebuilds of the same bars therefore produce equal levels rather than
    merely similar ones. Not a database key: a level whose zone shifts is a
    different level under this rule.
    """
    return LevelId(
        f"{symbol}:{timeframe.code}:{kind.value}:{format_exact(zone.low)}:{format_exact(zone.high)}"
    )


def build_levels(
    candidates: Iterable[LevelCandidate],
    *,
    symbol: Ticker,
    timeframe: Timeframe,
    bars: Sequence[Bar] = (),
    atr: float | None = None,
    as_of: datetime,
    config: LevelConfig | None = None,
    profile: VolumeProfile | None = None,
) -> tuple[Level, ...]:
    """Cluster, price, score and assemble levels from candidate evidence.

    `bars` is the history touches are counted over and recency is measured in;
    `atr` is the volatility the zone width and merge distances are expressed in,
    `None` while it is still warming. `profile` is the volume-at-price the
    `volume_context` component is read from, `None` when there is none — the
    component is then undefined rather than zero (ADR 0025 D8).
    """
    settings = config if config is not None else LevelConfig()
    moment = ensure_utc(as_of, field="as_of")
    volatility = from_approximate(atr, field="atr") if atr is not None else None
    ages = _bar_ages(bars)

    levels = [
        _build_one(
            cluster,
            symbol=symbol,
            timeframe=timeframe,
            bars=bars,
            atr=volatility,
            as_of=moment,
            config=settings,
            ages=ages,
            profile=profile,
        )
        for cluster in cluster_candidates(candidates, config=settings, atr=volatility)
    ]
    return tuple(levels)


def _build_one(
    cluster: Sequence[LevelCandidate],
    *,
    symbol: Ticker,
    timeframe: Timeframe,
    bars: Sequence[Bar],
    atr: Decimal | None,
    as_of: datetime,
    config: LevelConfig,
    ages: Mapping[datetime, int],
    profile: VolumeProfile | None = None,
) -> Level:
    weights = [_weight_of(candidate, config=config, ages=ages) for candidate in cluster]
    centre = _weighted_centre(cluster, weights)
    zone = _zone_around(centre, atr=atr, config=config)
    effective = zone if zone is not None else Zone(low=centre, high=centre)
    recency = max(_recency_of(candidate, config=config, ages=ages) for candidate in cluster)

    strength, confidence, breakdown = score_cluster(
        cluster, zone=effective, bars=bars, config=config, recency=recency, profile=profile
    )

    return Level(
        id=level_id(symbol=symbol, timeframe=timeframe, kind=cluster[0].kind, zone=effective),
        symbol=symbol,
        timeframe=timeframe,
        kind=cluster[0].kind,
        price=centre,
        zone=zone,
        sources=frozenset(candidate.source for candidate in cluster),
        strength=strength,
        confidence=confidence,
        status=LevelStatus.ACTIVE,
        created_at=as_of,
        updated_at=as_of,
        strength_breakdown=breakdown,
    )


def _bar_ages(bars: Sequence[Bar]) -> Mapping[datetime, int]:
    """How many bars ago each bar start is, counted from the newest bar."""
    last = len(bars) - 1
    return {bar.start_time_utc: last - index for index, bar in enumerate(bars)}


def _bars_ago(candidate: LevelCandidate, ages: Mapping[datetime, int]) -> int:
    age = ages.get(candidate.observed_at_utc)
    if age is not None:
        return age
    # Dated outside the supplied window — a daily level under an intraday series,
    # or evidence older than the history at hand. Treated as the oldest thing
    # present rather than as brand new.
    return len(ages)


def _recency_of(
    candidate: LevelCandidate, *, config: LevelConfig, ages: Mapping[datetime, int]
) -> float:
    if config.recency_decay == 1.0:
        return 1.0
    return config.recency_decay ** _bars_ago(candidate, ages)


def _weight_of(
    candidate: LevelCandidate, *, config: LevelConfig, ages: Mapping[datetime, int]
) -> float:
    return config.quality_of(candidate.source) * _recency_of(candidate, config=config, ages=ages)


def _weighted_centre(cluster: Sequence[LevelCandidate], weights: Sequence[float]) -> Decimal:
    exact = [from_approximate(weight, field="cluster weight") for weight in weights]
    total = sum(exact, Decimal(0))
    if total <= 0:
        # Every candidate decayed to nothing; fall back to the plain mean rather
        # than dividing by zero or dropping the cluster.
        exact = [Decimal(1)] * len(cluster)
        total = Decimal(len(cluster))

    notional = sum(
        (
            DOMAIN_CONTEXT.multiply(candidate.price, weight)
            for candidate, weight in zip(cluster, exact, strict=True)
        ),
        Decimal(0),
    )
    return exact_price(DOMAIN_CONTEXT.divide(notional, total), field="level price")


def _zone_around(centre: Decimal, *, atr: Decimal | None, config: LevelConfig) -> Zone | None:
    half = config.zone_width.resolve(price=centre, atr=atr)
    if half is None or half <= 0:
        # No width to apply — an exact level, not a fabricated band (ADR 0009 D3).
        return None
    return Zone(low=max(centre - half, _MIN_ZONE_PRICE), high=centre + half)


# -- lifecycle --------------------------------------------------------------


def invalidate(level: Level, bar: Bar, *, at: datetime | None = None) -> Level:
    """Invalidate a level that `bar` closed decisively beyond (ADR 0009 D8).

    Resistance dies on a close above the zone, support on a close below it.
    Returns the level unchanged when it survives; nothing mutates in place, and
    nothing decays on a timer.
    """
    if level.status is not LevelStatus.ACTIVE:
        return level
    if not bar.is_final:
        raise InvariantViolation(
            f"{level.id}: a partial bar cannot invalidate a level; wait for the close"
        )

    zone = level.effective_zone
    broken = bar.close > zone.high if level.kind is LevelKind.RESISTANCE else bar.close < zone.low
    if not broken:
        return level

    moment = ensure_utc(at if at is not None else bar.end_time_utc, field="at")
    return dataclasses.replace(
        level,
        status=LevelStatus.INVALIDATED,
        updated_at=max(moment, level.updated_at),
    )


def bars_since_touch(level: Level, bars: Sequence[Bar]) -> int:
    """How many finalized bars have closed since anything last reached the zone.

    `0` while price is in it; `len(bars)` when nothing in the window ever
    touched it, which is the honest answer for "at least this long" — the
    window is all the evidence there is.
    """
    zone = level.effective_zone
    final = [bar for bar in bars if bar.is_final]
    for age, bar in enumerate(reversed(final)):
        if bar.low <= zone.high and bar.high >= zone.low:
            return age
    return len(final)


def decay(level: Level, *, untouched: int, config: LifecycleConfig, at: datetime) -> Level:
    """Weaken a standing level that nothing has tested (ADR 0025 D6).

    Build-time recency ages the *evidence*; this ages the level's *standing*. A
    shelf nobody has come back to for two hundred bars is weaker than the day it
    formed even though the swing that made it has not moved, and the two decays
    answer different questions — which is why this one is off by default too.

    The breakdown is scaled with the strength, so it still sums to the number it
    explains (`specs/07-levels.md`).
    """
    if config.untouched_decay >= 1.0 or untouched <= 0 or level.strength <= 0.0:
        return level

    factor = config.untouched_decay**untouched
    return dataclasses.replace(
        level,
        strength=level.strength * factor,
        strength_breakdown=tuple(
            (name, value * factor) for name, value in level.strength_breakdown
        ),
        updated_at=max(ensure_utc(at, field="at"), level.updated_at),
    )


def flip(level: Level, *, at: datetime) -> Level:
    """The other side of a level a close has just broken (ADR 0025 D5).

    Broken resistance becoming support is the oldest claim in technical
    analysis and the MVP deliberately refused to make it (ADR 0009 D8). What
    makes it sayable now is `flipped_from`: the flipped level is a genuinely new
    level — its kind is part of its identity — and the link back is what stops
    "the same band, other polarity" from being unrecoverable.

    The evidence does not change, so neither do the sources, the strength or the
    breakdown. What changes is which side of the band price is on.
    """
    moment = ensure_utc(at, field="at")
    zone = level.effective_zone
    kind = LevelKind.SUPPORT if level.kind is LevelKind.RESISTANCE else LevelKind.RESISTANCE
    return dataclasses.replace(
        level,
        id=level_id(symbol=level.symbol, timeframe=level.timeframe, kind=kind, zone=zone),
        kind=kind,
        status=LevelStatus.ACTIVE,
        created_at=moment,
        updated_at=moment,
        flipped_from=level.id,
    )


def advance(
    level: Level,
    *,
    bars: Sequence[Bar],
    config: LifecycleConfig | None = None,
    at: datetime | None = None,
) -> tuple[Level, ...]:
    """Carry one level forward over the bars that have closed since it was built.

    Returns what the level became — one level ordinarily, two when a break
    produces a flipped counterpart. Order is `(the level, its flip)`, and the
    caller decides what to do with either.

    **A projection, not an accumulation.** `advance` is applied to the level as
    `build_levels` produced it, and re-applied to that same level as the window
    grows; it is not applied to its own output. Decay compounds with the bars,
    not with the number of times this function has been called, which is what
    keeps the answer a pure function of `(level, bars, config)` — the same
    property `build_levels` has and for the same reason (ADR 0025 D6).

    Only bars that closed *after* the level was built are considered. A window
    reaching back before it exists to count touches, and closes from before a
    level existed cannot have broken it.

    The three effects are tried in the order they overrule each other: a break
    settles it outright, expiry applies only to a level nothing has broken, and
    decay only to one that is neither. A level that is already invalidated is
    left alone — the same rule `invalidate` follows.
    """
    settings = config if config is not None else LifecycleConfig()
    since = [bar for bar in bars if bar.is_final and bar.end_time_utc > level.created_at]
    if level.status is not LevelStatus.ACTIVE or not since:
        return (level,)

    for bar in since:
        broken = invalidate(level, bar, at=at)
        if broken.status is LevelStatus.INVALIDATED:
            # Dated to the bar that broke it, not to the newest one: the level
            # died when it died, and re-running this later must not move that.
            moment = broken.updated_at
            return (broken, flip(broken, at=moment)) if settings.polarity_flip else (broken,)

    moment = ensure_utc(at if at is not None else since[-1].end_time_utc, field="at")
    untouched = bars_since_touch(level, since)
    expiry = settings.expire_after_untouched_bars
    if expiry is not None and untouched >= expiry:
        # Expiry is not a break: nothing traded through this level, it simply
        # stopped mattering. Flipping it would claim a rejection that never
        # happened, so expiry never produces a counterpart.
        return (
            dataclasses.replace(
                level,
                status=LevelStatus.INVALIDATED,
                updated_at=max(moment, level.updated_at),
            ),
        )

    return (decay(level, untouched=untouched, config=settings, at=moment),)
