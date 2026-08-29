"""Where level candidates come from — specs/07-levels.md source list.

Pure functions, one per source. They read bars and confirmed structure and
produce dated candidates; they do not cluster, score or decide anything.

Every candidate is dated to the bar that produced the evidence, never to "now",
because recency and look-ahead are both measured from that date (ADR 0009 D5).
The exception is a *dynamic* source — a moving average, an anchored VWAP, a
volume-profile node — whose value is recomputed on every bar and is therefore
evidence about the present, not about the bar it was anchored at. Those are
dated to the newest bar, and the distinction is spelled out in ADR 0025 D5
rather than left for a reader to infer from two different call sites.

The MVP sources come first; the F2 sources (ADR 0025) follow, and
`collect_candidates` is the one place that knows which of them a configuration
has switched on.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from datetime import date, datetime, timedelta
from decimal import Decimal

from alphagate.core.bar import Bar
from alphagate.core.errors import InvariantViolation
from alphagate.core.level_engine.model import LevelCandidate, LevelConfig, ZoneWidth
from alphagate.core.level_engine.volume_profile import VolumeProfile
from alphagate.core.levels import LevelKind, LevelSource
from alphagate.core.numeric import DOMAIN_CONTEXT, from_approximate
from alphagate.core.numeric import price as exact_price
from alphagate.core.structure import SwingKind, SwingPoint, SwingStatus

__all__ = [
    "anchored_vwap",
    "anchored_vwap_candidates",
    "collect_candidates",
    "fibonacci_candidates",
    "gap_candidates",
    "moving_average_candidate",
    "previous_day_candidates",
    "previous_week_candidates",
    "swing_candidates",
    "volume_profile_candidates",
    "year_extreme_candidates",
]

_YEAR = timedelta(days=365)
_THREE = Decimal(3)


def swing_candidates(swings: Iterable[SwingPoint]) -> tuple[LevelCandidate, ...]:
    """Confirmed swings become levels: a high is resistance, a low is support.

    Candidates are silently skipped for anything that is not confirmed — a
    candidate swing is not evidence yet (ADR 0008 D3).
    """
    return tuple(
        LevelCandidate(
            price=swing.price,
            kind=LevelKind.RESISTANCE if swing.kind is SwingKind.HIGH else LevelKind.SUPPORT,
            source=(
                LevelSource.SWING_HIGH if swing.kind is SwingKind.HIGH else LevelSource.SWING_LOW
            ),
            observed_at_utc=swing.bar_start_utc,
        )
        for swing in swings
        if swing.status is SwingStatus.CONFIRMED
    )


def previous_day_candidates(daily_bars: Sequence[Bar]) -> tuple[LevelCandidate, ...]:
    """High and low of the most recent completed session.

    Pass completed daily bars only: the caller knows which session is still
    open, and a level built from a half-formed day would move under its own
    consumers.
    """
    completed = _completed(daily_bars)
    if not completed:
        return ()
    previous = completed[-1]
    return (
        LevelCandidate(
            price=previous.high,
            kind=LevelKind.RESISTANCE,
            source=LevelSource.PREV_DAY_HIGH,
            observed_at_utc=previous.start_time_utc,
        ),
        LevelCandidate(
            price=previous.low,
            kind=LevelKind.SUPPORT,
            source=LevelSource.PREV_DAY_LOW,
            observed_at_utc=previous.start_time_utc,
        ),
    )


def previous_week_candidates(daily_bars: Sequence[Bar]) -> tuple[LevelCandidate, ...]:
    """High and low of the last *completed* ISO week.

    The week the most recent bar belongs to is still in progress, so it is
    excluded — the same reason `previous_day_candidates` wants completed bars.
    """
    completed = _completed(daily_bars)
    if not completed:
        return ()

    weeks = {_iso_week(bar.session_date) for bar in completed}
    current = _iso_week(completed[-1].session_date)
    earlier = sorted(week for week in weeks if week < current)
    if not earlier:
        return ()

    target = earlier[-1]
    members = [bar for bar in completed if _iso_week(bar.session_date) == target]
    anchor = members[-1].start_time_utc
    return (
        LevelCandidate(
            price=max(bar.high for bar in members),
            kind=LevelKind.RESISTANCE,
            source=LevelSource.PREV_WEEK_HIGH,
            observed_at_utc=anchor,
        ),
        LevelCandidate(
            price=min(bar.low for bar in members),
            kind=LevelKind.SUPPORT,
            source=LevelSource.PREV_WEEK_LOW,
            observed_at_utc=anchor,
        ),
    )


def year_extreme_candidates(daily_bars: Sequence[Bar]) -> tuple[LevelCandidate, ...]:
    """52-week high and low, dated to the sessions that actually made them.

    Dating them to the extreme's own session rather than to today is what lets
    recency decay treat a year-old high as a year-old high.
    """
    completed = _completed(daily_bars)
    if not completed:
        return ()

    # The most recent session is always inside its own trailing year, so the
    # window is never empty once there is a completed bar at all.
    cutoff = completed[-1].session_date - _YEAR
    window = [bar for bar in completed if bar.session_date > cutoff]

    highest = max(window, key=lambda bar: (bar.high, bar.start_time_utc))
    lowest = min(window, key=lambda bar: (bar.low, bar.start_time_utc))
    return (
        LevelCandidate(
            price=highest.high,
            kind=LevelKind.RESISTANCE,
            source=LevelSource.YEAR_HIGH,
            observed_at_utc=highest.start_time_utc,
        ),
        LevelCandidate(
            price=lowest.low,
            kind=LevelKind.SUPPORT,
            source=LevelSource.YEAR_LOW,
            observed_at_utc=lowest.start_time_utc,
        ),
    )


def moving_average_candidate(
    value: float, *, price: float, observed_at_utc: datetime
) -> LevelCandidate:
    """A moving average configured as a dynamic level (specs/07-levels.md source 5).

    Support when price is above it, resistance when below: a dynamic level's
    polarity is decided by where price currently sits, not by history. Exactly
    at the average it is treated as resistance, matching the `STRICT` reading
    used for structure labels (ADR 0008 D5).
    """
    return LevelCandidate(
        price=from_approximate(value, field="moving average"),
        kind=LevelKind.SUPPORT if price > value else LevelKind.RESISTANCE,
        source=LevelSource.MOVING_AVERAGE,
        observed_at_utc=observed_at_utc,
    )


# -- F2: advanced sources (ADR 0025) ----------------------------------------


def volume_profile_candidates(
    profile: VolumeProfile, *, price: Decimal, observed_at_utc: datetime
) -> tuple[LevelCandidate, ...]:
    """The point of control and the nodes either side of it.

    Polarity follows the same rule a moving average uses: a node below price is
    support, one at or above it is resistance. A shelf of volume does not
    remember which way price came at it — where price sits now is the only thing
    that decides which side of it the shelf is on (ADR 0025 D2).
    """
    nodes = (
        (profile.poc, LevelSource.VOLUME_POC),
        *((node, LevelSource.VOLUME_HVN) for node in profile.high_nodes),
        *((node, LevelSource.VOLUME_LVN) for node in profile.low_nodes),
    )
    return tuple(
        LevelCandidate(
            price=node.price,
            kind=_dynamic_kind(price=price, value=node.price),
            source=source,
            observed_at_utc=observed_at_utc,
        )
        for node, source in nodes
    )


def anchored_vwap(bars: Sequence[Bar], *, anchor_start_utc: datetime) -> Decimal | None:
    """Volume-weighted average price from an anchor bar to the end of `bars`.

    `None` when nothing traded since the anchor: a volume-weighted average of no
    volume is undefined, and `SessionVwap` says the same thing the same way.

    Exact arithmetic, unlike `SessionVwap`'s float: this value becomes a level
    price, and a level's identity is derived from its price (ADR 0009 D2). A
    float here would make the id depend on summation order.
    """
    anchored = [bar for bar in bars if bar.is_final and bar.start_time_utc >= anchor_start_utc]
    notional = Decimal(0)
    volume = Decimal(0)
    for bar in anchored:
        if bar.volume <= 0:
            continue
        typical = DOMAIN_CONTEXT.divide(bar.high + bar.low + bar.close, _THREE)
        notional += DOMAIN_CONTEXT.multiply(typical, bar.volume)
        volume += bar.volume
    if volume <= 0:
        return None
    return exact_price(DOMAIN_CONTEXT.divide(notional, volume), field="anchored vwap")


def anchored_vwap_candidates(
    bars: Sequence[Bar],
    swings: Iterable[SwingPoint],
    *,
    price: Decimal,
    observed_at_utc: datetime,
    anchors: int = 1,
) -> tuple[LevelCandidate, ...]:
    """VWAPs anchored at the most recent confirmed swings, one each way.

    The anchor is a confirmed swing rather than a date, because "where the
    average holder since the last low is" is the question the indicator is
    usually being asked. Confirmed only: anchoring at a swing that may yet be
    revoked would move the level under its own consumers (CLAUDE.md §12).
    """
    if anchors < 1:
        return ()

    confirmed = sorted(
        (swing for swing in swings if swing.status is SwingStatus.CONFIRMED),
        key=lambda swing: swing.bar_start_utc,
    )
    chosen: list[datetime] = []
    for kind in (SwingKind.HIGH, SwingKind.LOW):
        matching = [swing for swing in confirmed if swing.kind is kind]
        chosen.extend(swing.bar_start_utc for swing in matching[-anchors:])

    candidates = []
    for anchor in sorted(set(chosen)):
        value = anchored_vwap(bars, anchor_start_utc=anchor)
        if value is None or value <= 0:
            continue
        candidates.append(
            LevelCandidate(
                price=value,
                kind=_dynamic_kind(price=price, value=value),
                source=LevelSource.ANCHORED_VWAP,
                observed_at_utc=observed_at_utc,
            )
        )
    return tuple(candidates)


def fibonacci_candidates(
    swings: Iterable[SwingPoint], *, ratios: Sequence[Decimal]
) -> tuple[LevelCandidate, ...]:
    """Retracements of the most recent completed leg (ADR 0025 D3).

    The leg is the last confirmed swing and the nearest confirmed swing of the
    other kind before it. Retracements of an up-leg are support and of a
    down-leg resistance — the level is where the market would be *coming back
    to*, so its polarity is the direction the leg travelled.

    Dated to the later of the two pivots: the leg does not exist until its second
    end is confirmed, and dating it to the first would claim the retracement was
    knowable before the swing that defines it (CLAUDE.md §12).
    """
    confirmed = sorted(
        (swing for swing in swings if swing.status is SwingStatus.CONFIRMED),
        key=lambda swing: swing.bar_start_utc,
    )
    if len(confirmed) < 2 or not ratios:
        return ()

    latest = confirmed[-1]
    earlier = next(
        (swing for swing in reversed(confirmed[:-1]) if swing.kind is not latest.kind), None
    )
    if earlier is None:
        return ()

    high = max(latest.price, earlier.price)
    low = min(latest.price, earlier.price)
    span = high - low
    if span <= 0:
        # A leg of zero height retraces to itself; there is no geometry here.
        return ()

    rising = latest.kind is SwingKind.HIGH
    kind = LevelKind.SUPPORT if rising else LevelKind.RESISTANCE
    return tuple(
        LevelCandidate(
            price=(
                high - DOMAIN_CONTEXT.multiply(span, ratio)
                if rising
                else low + DOMAIN_CONTEXT.multiply(span, ratio)
            ),
            kind=kind,
            source=LevelSource.FIBONACCI,
            observed_at_utc=latest.bar_start_utc,
        )
        for ratio in ratios
    )


def gap_candidates(
    bars: Sequence[Bar], *, minimum: ZoneWidth | None = None, atr: Decimal | None = None
) -> tuple[LevelCandidate, ...]:
    """Both edges of every gap price has not yet traded back through.

    A gap up leaves unvisited air below: support at the previous high and at the
    new low. A gap down leaves it above, as resistance. Two candidates per gap
    rather than one, because the gap *is* a band — and letting the clustering
    build that band from its edges is better than inventing a midpoint the
    market never traded at.

    A gap is dropped once any later bar trades back through its far edge. Partial
    fills leave the zone as it was: half a gap is still a gap, and shrinking it
    would mean the level moved without the evidence changing (ADR 0025 D4).
    """
    final = sorted((bar for bar in bars if bar.is_final), key=lambda bar: bar.start_time_utc)
    if len(final) < 2:
        return ()

    # Suffix extremes: how far price came back after each bar, in one pass rather
    # than a scan per gap.
    lowest_after: list[Decimal] = [final[-1].low] * len(final)
    highest_after: list[Decimal] = [final[-1].high] * len(final)
    for index in range(len(final) - 2, -1, -1):
        lowest_after[index] = min(final[index].low, lowest_after[index + 1])
        highest_after[index] = max(final[index].high, highest_after[index + 1])

    candidates: list[LevelCandidate] = []
    for index in range(1, len(final)):
        previous, current = final[index - 1], final[index]
        if current.low > previous.high:
            edges, kind, filled = (
                (previous.high, current.low),
                LevelKind.SUPPORT,
                lowest_after[index] <= previous.high,
            )
        elif current.high < previous.low:
            edges, kind, filled = (
                (current.high, previous.low),
                LevelKind.RESISTANCE,
                highest_after[index] >= previous.low,
            )
        else:
            continue

        if filled or not _wide_enough(edges, minimum=minimum, atr=atr):
            continue
        candidates.extend(
            LevelCandidate(
                price=edge,
                kind=kind,
                source=LevelSource.GAP,
                observed_at_utc=current.start_time_utc,
            )
            for edge in edges
        )
    return tuple(candidates)


def _wide_enough(
    edges: tuple[Decimal, Decimal], *, minimum: ZoneWidth | None, atr: Decimal | None
) -> bool:
    if minimum is None:
        return True
    floor = minimum.resolve(price=edges[0], atr=atr)
    # An unresolvable floor (ATR warming) filters nothing rather than everything:
    # refusing to answer is not the same as answering "too small".
    return floor is None or edges[1] - edges[0] >= floor


def _dynamic_kind(*, price: Decimal, value: Decimal) -> LevelKind:
    """Support below price, resistance at or above it.

    The `STRICT` reading used for structure labels and for moving averages
    (ADR 0008 D5): exactly on the level counts as resistance.
    """
    return LevelKind.SUPPORT if price > value else LevelKind.RESISTANCE


def collect_candidates(
    bars: Sequence[Bar],
    *,
    swings: Iterable[SwingPoint] = (),
    daily_bars: Sequence[Bar] = (),
    atr: Decimal | None = None,
    profile: VolumeProfile | None = None,
    config: LevelConfig | None = None,
) -> tuple[LevelCandidate, ...]:
    """Every source a configuration has switched on, over one window.

    One place decides which sources are consulted, so the chart and the alert
    engine cannot end up reading different level sets from the same bars — which
    they would the moment two call sites assembled this list themselves.

    `daily_bars` are the completed sessions the previous-day/week and 52-week
    extremes come from; a five-minute series does not have them and is handed
    them by whoever is also running the daily one.
    """
    settings = config if config is not None else LevelConfig()
    final = [bar for bar in bars if bar.is_final]
    swings = tuple(swings)

    candidates = swing_candidates(swings)
    if daily_bars:
        candidates += (
            previous_day_candidates(daily_bars)
            + previous_week_candidates(daily_bars)
            + year_extreme_candidates(daily_bars)
        )
    if not final:
        return candidates

    advanced = settings.advanced
    latest = final[-1]
    # Dynamic sources are evidence about the present and are dated to the newest
    # closed bar; static ones keep the date of the bar that made them.
    now = latest.start_time_utc

    if advanced.volume_profile and profile is not None:
        candidates += volume_profile_candidates(profile, price=latest.close, observed_at_utc=now)
    if advanced.anchored_vwap:
        candidates += anchored_vwap_candidates(
            final,
            swings,
            price=latest.close,
            observed_at_utc=now,
            anchors=advanced.vwap_anchors,
        )
    if advanced.fibonacci:
        candidates += fibonacci_candidates(swings, ratios=advanced.fibonacci_ratios)
    if advanced.gaps:
        candidates += gap_candidates(final, minimum=advanced.minimum_gap, atr=atr)
    return candidates


def _completed(bars: Sequence[Bar]) -> list[Bar]:
    ordered = sorted(bars, key=lambda bar: bar.start_time_utc)
    final = [bar for bar in ordered if bar.is_final]
    for previous, current in itertools.pairwise(final):
        if previous.session_date == current.session_date:
            raise InvariantViolation(
                f"two daily bars for {current.session_date.isoformat()}; session extremes "
                "need one bar per session"
            )
    return final


def _iso_week(day: date) -> tuple[int, int]:
    calendar = day.isocalendar()
    return (calendar.year, calendar.week)
