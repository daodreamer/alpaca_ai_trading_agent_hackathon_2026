"""The cross-sectional engine: always invested, ranked, rebalanced, 80/20.

``engine.run_backtest`` simulates a trigger: something fires, a position opens,
a stop or a target or a bar count closes it. That form was never going to beat a
long benchmark, and the registry says so in fourteen promoted strategies that
all lost to buy and hold. The arithmetic is not subtle -- a book that is out of
the market part of the time forfeits part of the drift it is measured against,
so winning on Sharpe requires cutting volatility by more than it cut return.

This engine takes the other form:

* stay invested, so beta is roughly one and the drift is not given away;
* rank the universe and hold the top of it, so the excess return is a
  cross-sectional spread rather than a bet on being out at the right moments;
* rebalance on a fixed cadence, so breadth is the number of names rather than
  the number of concurrent positions.

Two engines rather than one, deliberately. The state machines share almost
nothing -- there are no stops here, no targets, no holding-period limit, no
per-position sizing -- and merging them would make both harder to read while
putting the existing green tests at risk of a refactor they did not ask for.

**The 80/20 split.** The core holds ``1 - sleeve.budget``; the sleeve holds the
rest and is reserved for event-driven deviations that do not exist yet. While
idle it holds the benchmark rather than cash, which is the difference between a
deviation budget and a permanent handicap: at the benchmark's historical CAGR an
idle 20% costs several percent a year, more than the entire realistic alpha
budget.

**What counts as a trade.** Only the core leg. The sleeve holds every eligible
name at all times, so if the sleeve counted, nothing would ever leave the book
and a rebalancing strategy would report no round-trips at all -- failing the
evaluator's minimum-trades gate for a reason that has nothing to do with the
rule. The strategy's decisions are core selections, and those are what is
reported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import numpy as np

from aqr.backtest.engine import BacktestConfig, BacktestResult, Trade
from aqr.data.bars import Bars
from aqr.data.universes import PointInTimeUniverse
from aqr.dsl.expr import evaluate
from aqr.dsl.schema import StrategySpec
from aqr.features.cross_section import CrossSection
from aqr.features.engine import FeatureFrame

__all__ = ["PortfolioResult", "run_portfolio"]


@dataclass(slots=True)
class PortfolioResult(BacktestResult):
    """A portfolio run.

    Two kinds of weight are reported and they answer different questions.
    :meth:`weights_at` gives the *target* weights in force at a step -- the
    book's intent as of the last rebalance, which is what a selection rule is
    judged on. ``exposure``, ``core_weight`` and ``sleeve_weight`` give the
    realised mark-to-market fractions, which drift with prices between
    rebalances and are what an allocator would see.
    """

    exposure: np.ndarray = field(default_factory=lambda: np.zeros(0))
    core_weight: np.ndarray = field(default_factory=lambda: np.zeros(0))
    sleeve_weight: np.ndarray = field(default_factory=lambda: np.zeros(0))
    benchmark_equity: np.ndarray | None = None
    first_decision_step: int | None = None
    first_fill_step: int | None = None
    _core_targets: list[dict[str, float]] = field(default_factory=list, repr=False)
    _sleeve_targets: list[dict[str, float]] = field(default_factory=list, repr=False)

    def core_weights_at(self, step: int) -> dict[str, float]:
        """Target core weights in force at ``step``."""
        return dict(self._core_targets[step])

    def sleeve_weights_at(self, step: int) -> dict[str, float]:
        return dict(self._sleeve_targets[step])

    def weights_at(self, step: int) -> dict[str, float]:
        out = dict(self._sleeve_targets[step])
        for symbol, weight in self._core_targets[step].items():
            out[symbol] = out.get(symbol, 0.0) + weight
        return out


@dataclass(slots=True)
class _Leg:
    """One open core holding, tracked so a round-trip can be reported."""

    shares: float
    entry_price: float
    entry_time: int
    entry_step: int
    fees: float
    slippage: float


def _aligned(
    values: np.ndarray, step_of_bar: np.ndarray, length: int, fill: float
) -> np.ndarray:
    """Scatter a per-bar series onto the shared timeline.

    Symbols that list late, delist early or simply have a shorter history leave
    gaps, and a gap must read as absence. Filling it with a number would let a
    name that was not trading take part in the cross-section.
    """
    out = np.full(length, fill, dtype=np.float64)
    out[step_of_bar] = values
    return out


def run_portfolio(
    spec: StrategySpec,
    data: dict[str, Bars] | Bars,
    config: BacktestConfig | None = None,
    *,
    peers: dict[str, Bars] | None = None,
    membership: PointInTimeUniverse | None = None,
) -> PortfolioResult:
    """Simulate ``spec`` in portfolio mode.

    ``peers`` is the market that peer-relative features rank against, and
    defaults to the traded universe -- same contract as ``run_backtest``, so a
    result stays reproducible from the spec alone.

    ``membership`` restricts what may be held, ranked and benchmarked to the
    names actually in the index on each day. Truncating each symbol's bars to
    its intervals would get most of this for free, but not re-entry: a company
    dropped in 2019 and readmitted in 2021 keeps trading throughout, so its bars
    have no gap and bar-presence says "member" for the whole span. The book would
    then hold it during two years when it was not in the universe -- picked,
    necessarily, because we know it came back.

    Omitting it reproduces the previous behaviour exactly, which every result in
    the registry was measured under.
    """
    if spec.mode != "portfolio":
        raise ValueError(f"{spec.name}: run_portfolio needs mode 'portfolio', got {spec.mode!r}")
    config = config or BacktestConfig()
    if isinstance(data, Bars):
        data = {data.symbol: data}

    universe = [s for s in spec.universe.symbols if s in data]
    missing = [s for s in spec.universe.symbols if s not in data]
    if not universe:
        raise ValueError(f"{spec.name}: no data for any symbol in {list(spec.universe.symbols)}")

    timeline = np.unique(np.concatenate([data[s].event_time for s in universe]))
    steps = int(timeline.size)

    # A missing symbol is a data hole -- unless the membership table says the
    # name was not in the index over the span these bars cover, in which case
    # having no bars is what the table predicts. Checked after the timeline is
    # built, because the span is what decides it.
    if missing:
        if membership is None:
            raise ValueError(f"{spec.name}: missing bars for {missing}")
        holes = membership.unexplained_absences(
            missing,
            datetime.fromtimestamp(int(timeline[0]), tz=UTC).date(),
            datetime.fromtimestamp(int(timeline[-1]), tz=UTC).date() + timedelta(days=1),
        )
        if holes:
            raise ValueError(f"{spec.name}: missing bars for {holes}")
    cross_section = CrossSection(peers if peers else {s: data[s] for s in universe})

    rank: dict[str, np.ndarray] = {}
    screen: dict[str, np.ndarray] = {}
    last_step: dict[str, int] = {}
    # Built once. Converting the timeline to dates inside the per-symbol loop is
    # 800 symbols times 2,200 steps of `datetime.fromtimestamp` for a calendar
    # that is identical every time.
    # Ordinals rather than ``date`` objects: the mask below is a range lookup,
    # and a range lookup on 2,200 Python dates per symbol is 1.5 million
    # comparisons for an answer that two binary searches give exactly.
    calendar: np.ndarray = (
        np.array(
            [datetime.fromtimestamp(int(t), tz=UTC).date().toordinal() for t in timeline],
            dtype=np.int64,
        )
        if membership is not None
        else np.empty(0, dtype=np.int64)
    )
    open_px: dict[str, np.ndarray] = {}
    close_px: dict[str, np.ndarray] = {}
    # What a held name is marked at on each step: the close, carried forward
    # across gaps. Precomputed because it is read once per holding per step.
    mark_px: dict[str, np.ndarray] = {}
    warmup = 0

    assert spec.rank_ast is not None  # guaranteed by StrategySpec validation
    for symbol in universe:
        bars = data[symbol]
        frame = FeatureFrame(bars, cross_section)
        warmup = max(warmup, frame.warmup(spec.features()))
        step_of_bar = np.searchsorted(timeline, bars.event_time)

        values = np.asarray(evaluate(spec.rank_ast, frame), dtype=np.float64)
        rank[symbol] = _aligned(values, step_of_bar, steps, np.nan)

        if spec.screen_ast is not None:
            mask = np.asarray(evaluate(spec.screen_ast, frame), dtype=bool)
            screen[symbol] = _aligned(mask.astype(np.float64), step_of_bar, steps, 0.0) > 0.5
        else:
            present = np.zeros(steps, dtype=bool)
            present[step_of_bar] = True
            screen[symbol] = present
        if membership is not None:
            screen[symbol] &= _membership_mask(membership, symbol, calendar)

        last_step[symbol] = int(step_of_bar[-1]) if step_of_bar.size else -1
        open_px[symbol] = _aligned(
            np.asarray(bars.open, dtype=np.float64), step_of_bar, steps, np.nan
        )
        close_px[symbol] = _aligned(
            np.asarray(bars.close, dtype=np.float64), step_of_bar, steps, np.nan
        )
        mark_px[symbol] = _carry_forward(close_px[symbol], open_px[symbol])

    core_budget = spec.sleeve.core_budget
    sleeve_budget = spec.sleeve.budget
    costs = config.costs

    cash = float(config.initial_equity)
    legs: dict[str, _Leg] = {}
    sleeve_shares: dict[str, float] = {}
    trades: list[Trade] = []

    equity = np.full(steps, float(config.initial_equity), dtype=np.float64)
    exposure = np.zeros(steps, dtype=np.float64)
    core_weight = np.zeros(steps, dtype=np.float64)
    sleeve_weight = np.zeros(steps, dtype=np.float64)
    core_targets: list[dict[str, float]] = [{} for _ in range(steps)]
    sleeve_targets: list[dict[str, float]] = [{} for _ in range(steps)]

    pending: tuple[dict[str, float], dict[str, float]] | None = None
    live_core: dict[str, float] = {}
    live_sleeve: dict[str, float] = {}
    first_decision: int | None = None
    first_fill: int | None = None

    for step in range(steps):
        # ---- fill yesterday's decision at today's open -------------------
        if pending is not None:
            target_core, target_sleeve = pending
            pending = None
            held = _holdings(legs, sleeve_shares)
            marked = cash + sum(sh * float(mark_px[s][step]) for s, sh in held)
            cash = _rebalance(
                step=step,
                timeline=timeline,
                target_core=target_core,
                target_sleeve=target_sleeve,
                open_px=open_px,
                equity=marked,
                cash=cash,
                legs=legs,
                sleeve_shares=sleeve_shares,
                trades=trades,
                costs=costs,
                allow_fractional=config.allow_fractional_shares,
            )
            live_core, live_sleeve = target_core, target_sleeve
            if first_fill is None:
                first_fill = step

        # ---- a final bar is an exit --------------------------------------
        #
        # Prices are carried forward across gaps, because a name with no bar
        # today has not become worthless -- it simply did not trade. That
        # reasoning stops when the bars stop for good. Without this, a delisted
        # holding is never sold: the rebalance only trades symbols that have a
        # price at the current step, so the position sits in the book at its
        # final mark, contributing a flat, riskless, permanently profitable line
        # for the rest of the run. It is the most flattering failure available,
        # and it only appears on a universe that contains the companies that
        # were acquired or wiped out -- which is the only kind worth building.
        for symbol in sorted(set(legs) | set(sleeve_shares)):
            final = last_step.get(symbol, -1)
            stops_trading = final == step and final < steps - 1
            leaves_index = (
                membership is not None
                and step + 1 < steps
                and screen[symbol][step]
                and not screen[symbol][step + 1]
                and not stops_trading
            )
            if not (stops_trading or leaves_index):
                # The end of the data is not a delisting. Every symbol's last
                # bar is the final step, and liquidating there would rewrite
                # every open position at the end of every run as an exit.
                continue
            reason = "delisted" if stops_trading else "left_universe"
            price = float(close_px[symbol][step])
            if np.isnan(price):
                continue
            shares = sleeve_shares.pop(symbol, 0.0)
            leg = legs.pop(symbol, None)
            if leg is not None:
                shares += leg.shares
                trades.append(
                    _round_trip(
                        symbol,
                        leg,
                        exit_price=price,
                        exit_time=int(timeline[step]),
                        exit_step=step,
                        fee=0.0,
                        reason=reason,
                    )
                )
            if shares:
                cash += shares * price
            live_core.pop(symbol, None)
            live_sleeve.pop(symbol, None)

        core_targets[step] = dict(live_core)
        sleeve_targets[step] = dict(live_sleeve)

        # ---- mark to market ----------------------------------------------
        core_value = sum(leg.shares * float(mark_px[s][step]) for s, leg in legs.items())
        sleeve_value = sum(sh * float(mark_px[s][step]) for s, sh in sleeve_shares.items())
        total = cash + core_value + sleeve_value
        equity[step] = total
        if total > 0:
            exposure[step] = (core_value + sleeve_value) / total
            core_weight[step] = core_value / total
            sleeve_weight[step] = sleeve_value / total

        # ---- decide, for tomorrow ----------------------------------------
        if step >= warmup and (step - warmup) % spec.rebalance_every == 0 and step + 1 < steps:
            eligible = [
                s
                for s in universe
                if screen[s][step]
                and not np.isnan(rank[s][step])
                and not np.isnan(open_px[s][step + 1])
            ]
            # Sorted by rank descending, then by symbol. The tie-break is not
            # cosmetic: without it a flat cross-section makes the book depend on
            # dictionary order, and the run stops being reproducible.
            ordered = sorted(eligible, key=lambda s: (-float(rank[s][step]), s))
            chosen = ordered[: spec.hold]

            # Divided by ``hold``, not by how many happened to qualify. Putting
            # two names' worth of capital into one name is a different strategy
            # with twice the idiosyncratic risk, arrived at by accident.
            per_name = core_budget / spec.hold if spec.hold else 0.0
            target_core = {s: per_name for s in chosen}

            spare = core_budget - per_name * len(chosen)
            if spec.sleeve.idle == "benchmark" and eligible:
                # The unfilled core budget joins the sleeve rather than sitting
                # in cash: the book stays invested without concentrating.
                share = (sleeve_budget + spare) / len(eligible)
                target_sleeve = {s: share for s in eligible}
            else:
                target_sleeve = {}

            pending = (target_core, target_sleeve)
            if first_decision is None:
                first_decision = step

    benchmark = _benchmark(universe, close_px, screen, float(config.initial_equity))

    return PortfolioResult(
        spec_fingerprint=spec.fingerprint(),
        strategy=spec.name,
        symbols=tuple(universe),
        timeline=timeline,
        equity=equity,
        trades=trades,
        warmup_bars=warmup,
        exposure=exposure,
        core_weight=core_weight,
        sleeve_weight=sleeve_weight,
        benchmark_equity=benchmark,
        first_decision_step=first_decision,
        first_fill_step=first_fill,
        _core_targets=core_targets,
        _sleeve_targets=sleeve_targets,
    )


def _holdings(
    legs: dict[str, _Leg], sleeve_shares: dict[str, float]
) -> list[tuple[str, float]]:
    out: dict[str, float] = {s: leg.shares for s, leg in legs.items()}
    for symbol, shares in sleeve_shares.items():
        out[symbol] = out.get(symbol, 0.0) + shares
    return list(out.items())


def _carry_forward(closes: np.ndarray, opens: np.ndarray) -> np.ndarray:
    """Last known close for a symbol at each step, carried forward across gaps.

    A name with no bar today has not become worthless; it simply did not trade.
    Marking it at zero would produce an equity curve with holes in it. Before
    its first bar there is no close to carry, so the bar's own open stands in,
    and a step with neither is 0.0.

    Computed once per symbol rather than per mark. Resolving it per lookup meant
    rescanning the symbol's history from the beginning on every step it was
    held, which is quadratic in the run length and was a fifth of the cost of a
    large-universe backtest.
    """
    carried = np.asarray(closes, dtype=np.float64).copy()
    known = ~np.isnan(carried)
    # Index of the most recent known close at or before each step, or -1.
    last_known = np.maximum.accumulate(np.where(known, np.arange(carried.size), -1))
    fillable = ~known & (last_known >= 0)
    carried[fillable] = carried[last_known[fillable]]
    unknown = ~known & (last_known < 0)
    if unknown.any():
        fallback = np.asarray(opens, dtype=np.float64)[unknown]
        carried[unknown] = np.where(np.isnan(fallback), 0.0, fallback)
    return carried


def _rebalance(
    *,
    step: int,
    timeline: np.ndarray,
    target_core: dict[str, float],
    target_sleeve: dict[str, float],
    open_px: dict[str, np.ndarray],
    equity: float,
    cash: float,
    legs: dict[str, _Leg],
    sleeve_shares: dict[str, float],
    trades: list[Trade],
    costs: object,
    allow_fractional: bool,
) -> float:
    """Trade the book to its targets at this bar's open, and charge for it."""
    symbols = sorted(set(target_core) | set(target_sleeve) | set(legs) | set(sleeve_shares))
    when = int(timeline[step])

    for symbol in symbols:
        price = float(open_px[symbol][step]) if symbol in open_px else np.nan
        if np.isnan(price) or price <= 0:
            continue

        want_core = _shares(target_core.get(symbol, 0.0), equity, price, allow_fractional)
        want_sleeve = _shares(target_sleeve.get(symbol, 0.0), equity, price, allow_fractional)
        have_core = legs[symbol].shares if symbol in legs else 0.0
        have_sleeve = sleeve_shares.get(symbol, 0.0)

        delta = (want_core + want_sleeve) - (have_core + have_sleeve)
        if abs(delta) > 1e-12:
            side = "buy" if delta > 0 else "sell"
            fill = costs.fill_price(price, side)  # type: ignore[attr-defined]
            fee = costs.commission(abs(delta), price)  # type: ignore[attr-defined]
            cash -= delta * fill
            cash -= fee
        else:
            fee = 0.0

        # The core leg is the strategy's decision, so it is the leg that
        # reports round-trips. See the module docstring.
        if want_core <= 0 and have_core > 0:
            leg = legs.pop(symbol)
            trades.append(
                _round_trip(symbol, leg, exit_price=price, exit_time=when, exit_step=step, fee=fee)
            )
        elif want_core > 0 and have_core <= 0:
            legs[symbol] = _Leg(
                shares=want_core,
                entry_price=price,
                entry_time=when,
                entry_step=step,
                fees=fee,
                slippage=0.0,
            )
        elif want_core > 0 and symbol in legs:
            legs[symbol].shares = want_core

        if want_sleeve > 0:
            sleeve_shares[symbol] = want_sleeve
        else:
            sleeve_shares.pop(symbol, None)

    return cash


def _shares(weight: float, equity: float, price: float, allow_fractional: bool) -> float:
    if weight <= 0 or price <= 0 or equity <= 0:
        return 0.0
    raw = weight * equity / price
    return raw if allow_fractional else float(int(raw))


def _round_trip(
    symbol: str,
    leg: _Leg,
    *,
    exit_price: float,
    exit_time: int,
    exit_step: int,
    fee: float,
    reason: str = "rebalance",
) -> Trade:
    gross = (exit_price - leg.entry_price) * leg.shares
    return Trade(
        symbol=symbol,
        direction="long",
        entry_time=leg.entry_time,
        entry_price=leg.entry_price,
        exit_time=exit_time,
        exit_price=exit_price,
        quantity=leg.shares,
        gross_pnl=gross,
        fees=leg.fees + fee,
        slippage=leg.slippage,
        bars_held=exit_step - leg.entry_step,
        exit_reason=reason,
        mae=0.0,
        mfe=0.0,
    )


def _benchmark(
    universe: list[str],
    close_px: dict[str, np.ndarray],
    eligible: dict[str, np.ndarray],
    initial: float,
) -> np.ndarray:
    """Equal-weight the eligible names, rebalanced every bar, no costs.

    The same universe the strategy chose from, so beta cancels out of the
    comparison rather than being smuggled in by an index that happens to be
    handy. Deliberately generous -- frictionless and perfectly rebalanced -- so
    that a strategy which cannot beat it is not close.

    Eligibility matters here for the same reason it matters to the book: a
    benchmark holding names that were not in the index is something nobody could
    have owned, and measuring a strategy against it flatters or penalises it for
    a reason that has nothing to do with the rule.
    """
    steps = len(next(iter(close_px.values()))) if close_px else 0
    curve = np.full(steps, initial, dtype=np.float64)
    if steps == 0:
        return curve

    # One ``steps x symbols`` view instead of a Python loop over the universe
    # inside a Python loop over the sessions. Column order is ``universe``
    # order, and each session's returns are still gathered in that order, so
    # the mean is taken over exactly the values, in exactly the sequence, that
    # the per-symbol form produced.
    prices = np.column_stack([close_px[symbol] for symbol in universe])
    live = np.column_stack([eligible[symbol] for symbol in universe])

    for step in range(1, steps):
        prev, now = prices[step - 1], prices[step]
        taking = live[step] & ~np.isnan(prev) & ~np.isnan(now) & (prev > 0)
        if taking.any():
            rets = now[taking] / prev[taking] - 1.0
            curve[step] = curve[step - 1] * (1.0 + float(np.mean(rets)))
        else:
            curve[step] = curve[step - 1]
    return curve


def _membership_mask(
    membership: PointInTimeUniverse, symbol: str, calendar: np.ndarray
) -> np.ndarray:
    """Per-step membership for one symbol, over a pre-built calendar of ordinals.

    A symbol the table has never heard of is not a member on any day. Defaulting
    it to True would silently readmit exactly the names the table exists to
    exclude.

    An interval is a contiguous run of sessions, so it is filled by two binary
    searches on the (ascending) calendar rather than by asking every session
    whether it falls inside. Asking per session cost 1.5 million Python calls
    per backtest, which is most of a leave-one-out pass over a large universe.
    ``Interval`` is half-open, and ``side="left"`` at both ends is what makes
    the slice half-open too: ``start`` is included, ``end`` is not.
    """
    try:
        member = membership.by_ticker(symbol)
    except KeyError:
        return np.zeros(calendar.size, dtype=bool)
    mask = np.zeros(calendar.size, dtype=bool)
    for interval in member.intervals:
        lo = int(np.searchsorted(calendar, interval.start.toordinal(), side="left"))
        hi = (
            calendar.size
            if interval.end is None
            else int(np.searchsorted(calendar, interval.end.toordinal(), side="left"))
        )
        mask[lo:hi] = True
    return mask
