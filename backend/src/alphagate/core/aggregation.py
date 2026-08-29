"""Session-anchored bar grids and aggregation — ADR 0004 D4/D5/D8.

Two pure functions:

* `session_grid` — the bar boundaries for one trading day and timeframe;
* `aggregate_bars` — finer bars folded into a coarser timeframe on that grid.

Both take a `TradingCalendar`, never exchange hours. The grid is anchored at the
**regular-session open**, not the UTC epoch, which is what makes a 1h bar mean
the same thing regardless of which provider supplied the minutes. Cells are
clipped at session-window boundaries, so the last bar of a session is short and
pre-market prints never leak into a regular-session bar.

Nothing here reads a clock. Finality is decided by an explicit `watermark`
supplied by the caller — a bar is final only once time has demonstrably passed
its end. That keeps look-ahead out by construction (CLAUDE.md §12).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Sequence
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Final

from alphagate.core.bar import AdjustmentMode, Bar, Feed
from alphagate.core.errors import InvariantViolation
from alphagate.core.identifiers import Ticker
from alphagate.core.numeric import DOMAIN_CONTEXT
from alphagate.core.numeric import price as exact_price
from alphagate.core.time_model import SessionKind, Timeframe, TradingCalendar, ensure_utc

__all__ = [
    "GridCell",
    "aggregate_bars",
    "fold_bars",
    "next_close_after",
    "session_grid",
]

SEARCH_HORIZON: Final = timedelta(days=21)
"""How far `next_close_after` will look before giving up.

Long enough to cross a weekly cell from the Friday close plus the longest run of
shut days a calendar realistically has, and short enough that an exchange which
never opens produces an answer rather than a scan."""


@dataclasses.dataclass(frozen=True, slots=True)
class GridCell:
    """One bar-shaped slot on the session grid: `[start, end)`."""

    start: datetime
    end: datetime
    session: SessionKind
    session_date: date

    @property
    def duration(self) -> timedelta:
        return self.end - self.start

    def contains(self, instant: datetime) -> bool:
        return self.start <= instant < self.end


def session_grid(
    calendar: TradingCalendar,
    exchange: str,
    day: date,
    timeframe: Timeframe,
) -> tuple[GridCell, ...]:
    """Bar boundaries for one trading day, in chronological order.

    Empty when the exchange is closed. `1D` produces a single cell spanning the
    regular session — not the calendar day (ADR 0004 D6).
    """
    if timeframe is Timeframe.W1:
        return _week_grid(calendar, exchange, day)

    windows = calendar.sessions_for(exchange, day)
    if not windows:
        return ()

    regular = next((w for w in windows if w.kind is SessionKind.REGULAR), None)
    if regular is None:
        return ()

    if timeframe is Timeframe.D1:
        return (
            GridCell(
                start=regular.open_utc,
                end=regular.close_utc,
                session=SessionKind.REGULAR,
                session_date=day,
            ),
        )

    step = timeframe.nominal_duration
    anchor = regular.open_utc
    cells: list[GridCell] = []

    for window in windows:
        # Step back to the last grid line at or before the window opens, so the
        # grid is one continuous ruler across pre/regular/post rather than three
        # independently anchored ones.
        offset = (window.open_utc - anchor) // step
        cursor = anchor + offset * step
        while cursor < window.close_utc:
            cells.append(
                GridCell(
                    start=max(cursor, window.open_utc),
                    end=min(cursor + step, window.close_utc),
                    session=window.kind,
                    session_date=day,
                )
            )
            cursor += step

    return tuple(cells)


def next_close_after(
    calendar: TradingCalendar,
    exchange: str,
    *,
    timeframe: Timeframe,
    instant: datetime,
) -> datetime | None:
    """When the next bar of this series can first be closed — ADR 0023.

    The end of the first grid cell that ends **strictly after** `instant`. A bar
    that ended at 21:00 has nothing more to wait for at 21:00; the answer is the
    following cell.

    Derived from the calendar rather than from `nominal_duration`, because the
    two disagree exactly where it matters. The last regular hourly cell of a NYSE
    session is 20:30–21:00 — half an hour — so "the previous close plus an hour"
    would notice that bar thirty minutes late, and an alert on it would be thirty
    minutes late with it. Holidays and half-days are the same failure at a larger
    scale.

    `None` when nothing closes within `SEARCH_HORIZON`: an exchange that is shut
    for longer than the horizon, or one that never opens. It means "no answer",
    and a caller must read it as such — skipping work on the strength of not
    knowing is how a series stops being watched.
    """
    moment = ensure_utc(instant, field="instant")
    # One day early: a post-market cell belongs to a trading day whose UTC date
    # can already be the next one, so the cell containing `moment` may be found
    # only under yesterday's grid.
    day = moment.date() - timedelta(days=1)
    last = (moment + SEARCH_HORIZON).date()

    while day <= last:
        for cell in session_grid(calendar, exchange, day, timeframe):
            if cell.end > moment:
                return cell.end
        day += timedelta(days=1)
    return None


def _week_grid(
    calendar: TradingCalendar,
    exchange: str,
    day: date,
) -> tuple[GridCell, ...]:
    """The single cell covering the ISO week `day` falls in — ADR 0004 D6.

    Every day of the week produces the *same* cell, which is what lets
    `aggregate_bars` group daily bars keyed by their own `session_date` without
    knowing that weeks exist. The cell is dated to the week's first trading day,
    matching where it opens.

    Regular sessions only, exactly as `1D` is. The cell therefore spans the
    intervening nights and extended-hours windows without owning them; nothing
    but a `1D` bar may be folded into it (`_check_timeframes`).
    """
    monday = day - timedelta(days=day.weekday())
    regular = [
        (window, monday + timedelta(days=offset))
        for offset in range(7)
        for window in calendar.sessions_for(exchange, monday + timedelta(days=offset))
        if window.kind is SessionKind.REGULAR
    ]
    if not regular:
        return ()

    first, first_day = regular[0]
    last, _ = regular[-1]
    return (
        GridCell(
            start=first.open_utc,
            end=last.close_utc,
            session=SessionKind.REGULAR,
            session_date=first_day,
        ),
    )


def aggregate_bars(
    bars: Iterable[Bar],
    *,
    target: Timeframe,
    calendar: TradingCalendar,
    exchange: str,
    watermark: datetime,
) -> tuple[Bar, ...]:
    """Fold finer bars into `target`, on the session grid.

    `watermark` is how far time has advanced. A target bar is final only when the
    watermark has passed its end *and* every contributing source bar is final;
    there is no other way to declare a bar closed, which is what keeps a partial
    bar from being mistaken for a settled one.

    Source bars may be sparse — an illiquid minute simply does not print — but
    they must be homogeneous in symbol, feed and adjustment mode, and must not
    straddle a cell boundary.
    """
    source = sorted(bars, key=lambda b: b.start_time_utc)
    if not source:
        return ()

    limit = ensure_utc(watermark, field="watermark")
    identity = _shared_identity(source)
    _check_timeframes(source[0].timeframe, target)

    grouped: dict[GridCell, list[Bar]] = {}
    grids: dict[date, tuple[GridCell, ...]] = {}

    for bar in source:
        session_day = bar.session_date
        if session_day not in grids:
            grids[session_day] = session_grid(calendar, exchange, session_day, target)
        cell = _cell_for(grids[session_day], bar)
        grouped.setdefault(cell, []).append(bar)

    return tuple(
        _fold(cell, members, target=target, identity=identity, watermark=limit)
        for cell, members in sorted(grouped.items(), key=lambda item: item[0].start)
    )


def fold_bars(
    bars: Iterable[Bar],
    *,
    target: Timeframe,
    calendar: TradingCalendar,
    exchange: str,
    watermark: datetime,
) -> tuple[Bar, ...]:
    """Fold all the way to `target`, however many grids that takes — ADR 0021.

    `aggregate_bars` is one step. Two targets need more than the step:

    * `1D` is the regular session (ADR 0004 D6), so extended-hours source bars
      are dropped rather than left to fail placement;
    * `1W` cannot be reached from minutes at all — a weekly cell spans the
      nights between sessions, so a pre-market print would place inside it — and
      is therefore folded via `1D`, which is already confined to the session.

    Both callers that turn a provider's native bars into ours — the Alpaca
    adapter and `AggregatingBarSource` — go through here, so the chain is
    written once.
    """
    source = tuple(bars)
    if not source:
        return ()

    limit = ensure_utc(watermark, field="watermark")
    stages = (
        (Timeframe.D1, target) if _needs_daily_stage(source[0].timeframe, target) else (target,)
    )

    folded = source
    for stage in stages:
        if stage is Timeframe.D1:
            folded = tuple(bar for bar in folded if bar.session is SessionKind.REGULAR)
            if not folded:
                return ()
        if folded and folded[0].timeframe is stage:
            continue
        folded = aggregate_bars(
            folded, target=stage, calendar=calendar, exchange=exchange, watermark=limit
        )
    return folded


def _needs_daily_stage(source: Timeframe, target: Timeframe) -> bool:
    return target is Timeframe.W1 and source is not Timeframe.D1


# -- internals --------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class _Identity:
    symbol: Ticker
    source: str
    feed: Feed
    adjustment_mode: AdjustmentMode


def _shared_identity(bars: Sequence[Bar]) -> _Identity:
    first = bars[0]
    for bar in bars:
        if bar.symbol != first.symbol:
            raise InvariantViolation(
                f"cannot aggregate across symbols: {first.symbol} and {bar.symbol}"
            )
        if bar.feed is not first.feed:
            raise InvariantViolation(
                f"cannot aggregate across feeds: {first.feed.value} and {bar.feed.value} "
                "— they are different observations, not duplicates"
            )
        if bar.adjustment_mode is not first.adjustment_mode:
            raise InvariantViolation(
                f"cannot aggregate across adjustment modes: "
                f"{first.adjustment_mode.value} and {bar.adjustment_mode.value}"
            )
        if bar.timeframe is not first.timeframe:
            raise InvariantViolation(
                f"cannot aggregate mixed source timeframes: "
                f"{first.timeframe.code} and {bar.timeframe.code}"
            )
    return _Identity(
        symbol=first.symbol,
        source=first.source,
        feed=first.feed,
        adjustment_mode=first.adjustment_mode,
    )


def _check_timeframes(source: Timeframe, target: Timeframe) -> None:
    if source is target:
        raise InvariantViolation(f"source timeframe {source.code} must be finer than the target")
    if source.nominal_duration > target.nominal_duration:
        raise InvariantViolation(
            f"source timeframe {source.code} must be finer than the target {target.code}"
        )
    if target is Timeframe.W1 and source is not Timeframe.D1:
        # A weekly cell spans the whole week, nights included, so a pre-market
        # minute would place inside it and enter a bar that is defined as the
        # regular sessions only. Folding days — which are already regular-session
        # only — is what makes the week mean what ADR 0004 D6 says (ADR 0021).
        raise InvariantViolation(
            f"1W is aggregated from 1D, not from {source.code}: a weekly cell "
            "spans the nights between sessions, and only a daily bar is already "
            "confined to the regular session"
        )
    if target is not Timeframe.D1 and (
        target.nominal_duration % source.nominal_duration != timedelta(0)
    ):
        raise InvariantViolation(
            f"{target.code} is not a whole multiple of {source.code}; the grids would not line up"
        )


def _cell_for(grid: tuple[GridCell, ...], bar: Bar) -> GridCell:
    for cell in grid:
        if cell.contains(bar.start_time_utc):
            if bar.end_time_utc > cell.end:
                raise InvariantViolation(
                    f"source bar {bar.start_time_utc.isoformat()}–"
                    f"{bar.end_time_utc.isoformat()} would straddle the cell boundary "
                    f"at {cell.end.isoformat()}"
                )
            return cell
    raise InvariantViolation(
        f"no session cell contains {bar.start_time_utc.isoformat()} on "
        f"{bar.session_date.isoformat()}"
    )


def _fold(
    cell: GridCell,
    members: list[Bar],
    *,
    target: Timeframe,
    identity: _Identity,
    watermark: datetime,
) -> Bar:
    ordered = sorted(members, key=lambda b: b.start_time_utc)
    volume = sum((b.volume for b in ordered), Decimal(0))

    return Bar(
        symbol=identity.symbol,
        timeframe=target,
        start_time_utc=cell.start,
        end_time_utc=cell.end,
        session_date=cell.session_date,
        session=cell.session,
        open=ordered[0].open,
        high=max(b.high for b in ordered),
        low=min(b.low for b in ordered),
        close=ordered[-1].close,
        volume=volume,
        vwap=_weighted_vwap(ordered, volume),
        trade_count=_total_trades(ordered),
        source=identity.source,
        feed=identity.feed,
        adjustment_mode=identity.adjustment_mode,
        is_final=watermark >= cell.end and all(b.is_final for b in ordered),
    )


def _weighted_vwap(bars: Sequence[Bar], volume: Decimal) -> Decimal | None:
    """Volume-weighted mean of the source VWAPs.

    Dropped entirely if any source bar lacks one: a VWAP computed from a subset
    of the period is not the period's VWAP, and quietly returning it would be
    worse than returning nothing.
    """
    if volume <= 0 or any(b.vwap is None for b in bars):
        return None
    notional = sum(
        (DOMAIN_CONTEXT.multiply(b.vwap, b.volume) for b in bars if b.vwap is not None),
        Decimal(0),
    )
    return exact_price(DOMAIN_CONTEXT.divide(notional, volume), field="vwap")


def _total_trades(bars: Sequence[Bar]) -> int | None:
    counts = [b.trade_count for b in bars]
    if any(c is None for c in counts):
        return None
    return sum(c for c in counts if c is not None)
