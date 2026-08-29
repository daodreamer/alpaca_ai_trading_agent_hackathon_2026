"""Walk-forward validation (architecture section 13).

A single train/test split answers "did this work once". Walk-forward asks the
question that matters: *does re-fitting this rule on what was knowable at the
time keep working, repeatedly, on what came next.*

Each fold does exactly that. Parameters are chosen on the fold's train window
only, then applied unchanged to the fold's test window. The reported number is
the aggregate of the test windows -- train performance is recorded solely so the
overfitting detector can measure the gap between the two.

Fold boundaries are defined in *time*, not in bar positions, and each symbol is
then sliced by timestamp. Positional folds would silently misalign a universe
whose members have different histories -- fold 3 covering 2019 for one symbol
and 2021 for another -- and every cross-sectional number computed from them
would be meaningless.

Warm-up is handled by widening each window backwards so indicators are warm at
the first tradable bar, while trades opened inside the warm-up region are
excluded from the fold's result.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np

from aqr.backtest.engine import BacktestConfig, BacktestResult, Trade
from aqr.backtest.metrics import Metrics, compute_metrics
from aqr.backtest.run import run_strategy
from aqr.data.bars import Bars
from aqr.data.universes import PointInTimeUniverse
from aqr.dsl.schema import StrategySpec
from aqr.features.cross_section import CrossSection
from aqr.features.engine import FeatureFrame
from aqr.validation.params import apply_params
from aqr.validation.splits import TEST_BARS, TRAIN_BARS, Fold, Split, walk_forward_folds

__all__ = ["FoldResult", "WalkForwardReport", "run_walk_forward", "warmup_for"]

# Cap on grid size. A search this wide is already closer to data mining than to
# research, and the experiment record should show a refusal rather than a
# suspiciously good number.
_MAX_GRID = 512

# Below this, a train-window result is treated as no evidence at all rather than
# as a weak signal -- otherwise a two-trade fluke wins the parameter search.
_MIN_TRAIN_TRADES = 5


@dataclass(slots=True)
class FoldResult:
    fold: Fold
    params: dict[str, float]
    train: Metrics
    test: Metrics
    test_trades: list[Trade]
    candidates_tried: int

    @property
    def gap(self) -> float:
        """Train Sharpe minus test Sharpe. Large and positive means fitted."""
        return self.train.sharpe - self.test.sharpe


@dataclass(slots=True)
class WalkForwardReport:
    strategy: str
    fingerprint: str
    symbols: tuple[str, ...]
    folds: list[FoldResult] = field(default_factory=list)
    stitched: Metrics | None = None
    """Metrics over the chained test windows -- what an allocator who re-fitted
    on schedule and stayed invested would actually have experienced."""
    stitched_equity: np.ndarray | None = None
    """The curve those metrics summarise. Kept because the residual-alpha
    regression needs returns rather than a Sharpe: only the series can separate
    what the rule added from the exposure it happened to run."""
    stitched_timeline: np.ndarray | None = None
    """Timestamps for ``stitched_equity``, so it can be aligned with the
    benchmark by session rather than by position. Two series that merely have
    the same length are not necessarily the same days."""
    stitched_zero_cost: Metrics | None = None
    """The same test windows, the same selected parameters, no friction.

    Present only when a zero-cost config was asked for. It exists so that "how
    much of the edge do costs remove" can be answered by dividing two numbers
    that differ *only* in costs. Comparing the out-of-sample result against a
    frictionless full-period in-sample run instead answered a different
    question, and attributed out-of-sample decay to spreads."""

    @property
    def oos_sharpe(self) -> float:
        """Sharpe of the chained out-of-sample curve.

        Not the mean of the per-fold Sharpes. A mean over folds discards what
        happens across a fold boundary -- which is the whole reason ``stitched``
        exists -- and it runs systematically higher: on the four control
        baselines it exceeded the chained figure by 0.06 to 0.32, enough to
        report two of them as beating buy and hold when the continuous curve
        says they trailed.

        Falls back to the per-fold mean only when there is no chained curve to
        take, which means no fold produced one.
        """
        if self.stitched is not None:
            return self.stitched.sharpe
        return self.mean_fold_sharpe

    @property
    def mean_fold_sharpe(self) -> float:
        """Average Sharpe across folds. Reported, never compared: see
        :attr:`oos_sharpe` for why it is not the out-of-sample number."""
        return float(np.mean([f.test.sharpe for f in self.folds])) if self.folds else 0.0

    @property
    def oos_sharpe_std(self) -> float:
        if len(self.folds) < 2:
            return 0.0
        return float(np.std([f.test.sharpe for f in self.folds], ddof=1))

    @property
    def positive_fold_rate(self) -> float:
        """Share of folds with a positive test return. Consistency, not size."""
        if not self.folds:
            return 0.0
        return float(np.mean([f.test.total_return > 0 for f in self.folds]))

    @property
    def mean_gap(self) -> float:
        return float(np.mean([f.gap for f in self.folds])) if self.folds else 0.0

    @property
    def total_test_trades(self) -> int:
        return sum(f.test.num_trades for f in self.folds)

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "fingerprint": self.fingerprint,
            "symbols": list(self.symbols),
            "folds": len(self.folds),
            "oos_sharpe": self.oos_sharpe,
            "mean_fold_sharpe": self.mean_fold_sharpe,
            "oos_sharpe_std": self.oos_sharpe_std,
            "positive_fold_rate": self.positive_fold_rate,
            "mean_train_test_gap": self.mean_gap,
            "total_test_trades": self.total_test_trades,
            "stitched": self.stitched.as_dict() if self.stitched else None,
            "stitched_zero_cost": (
                self.stitched_zero_cost.as_dict() if self.stitched_zero_cost else None
            ),
            "per_fold": [
                {
                    "index": f.fold.index,
                    "params": f.params,
                    "train_sharpe": f.train.sharpe,
                    "test_sharpe": f.test.sharpe,
                    "test_return": f.test.total_return,
                    "test_trades": f.test.num_trades,
                }
                for f in self.folds
            ],
        }

    def __str__(self) -> str:
        head = (
            f"{self.strategy} on {'+'.join(self.symbols)}: {len(self.folds)} folds, "
            f"OOS Sharpe {self.oos_sharpe:.2f} +/- {self.oos_sharpe_std:.2f}, "
            f"{self.positive_fold_rate:.0%} folds positive, "
            f"train-test gap {self.mean_gap:+.2f}"
        )
        rows = [
            f"  fold {f.fold.index}: train {f.train.sharpe:6.2f} | test {f.test.sharpe:6.2f} "
            f"| ret {f.test.total_return:+7.1%} | {f.test.num_trades:3d} trades"
            for f in self.folds
        ]
        return "\n".join([head, *rows])


def warmup_for(spec: StrategySpec, data: dict[str, Bars]) -> int:
    """Bars of history the strategy needs before it can trade anywhere."""
    cross_section = CrossSection(data)
    return max(
        FeatureFrame(bars, cross_section).warmup(spec.features())
        for bars in data.values()
    )


def _expand_grid(grid: dict[str, list[float]]) -> list[dict[str, float]]:
    if not grid:
        return [{}]
    keys = sorted(grid)
    combos = list(itertools.product(*(grid[k] for k in keys)))
    if len(combos) > _MAX_GRID:
        raise ValueError(
            f"parameter grid has {len(combos)} combinations, over the {_MAX_GRID} limit; "
            "narrow the search or the result will be a data-mining artefact"
        )
    return [dict(zip(keys, combo, strict=True)) for combo in combos]


def _score(metrics: Metrics) -> float:
    """Selection criterion applied to the *train* window only."""
    if metrics.num_trades < _MIN_TRAIN_TRADES:
        return float("-inf")
    return metrics.sharpe


def _window(
    data: dict[str, Bars], start_ts: int, stop_ts: int, warmup: int
) -> tuple[dict[str, Bars], int]:
    """Slice every symbol to ``[start_ts, stop_ts)``, widened back by ``warmup``.

    Returns the sliced data and the timestamp at which the fold proper begins,
    so the caller can discard trades that opened inside the warm-up history.
    """
    out: dict[str, Bars] = {}
    for symbol, bars in data.items():
        stop = bars.index_of_ts(stop_ts)
        start = bars.index_of_ts(start_ts)
        # ``index_of_ts`` is the first bar at or after a timestamp, so for a
        # series that ended years ago it returns the series end for *both*
        # bounds. Widening then produced a slice of the symbol's final bars,
        # and a name delisted in 2016 joined a 2024 window carrying 2016
        # prices. The engine builds its timeline from the union of event
        # times, so one such name stretched the window back to its own last
        # bar. A symbol with no bar inside the window is simply not in it.
        if stop - start < 1:
            continue
        widened = max(0, start - warmup)
        if stop - widened < 2:
            continue
        out[symbol] = bars.slice(widened, stop)
    return out, start_ts


def _unexplained_absences(
    missing: list[str],
    membership: PointInTimeUniverse | None,
    data: dict[str, Bars],
    start_ts: int,
    stop_ts: int,
) -> list[str]:
    """Absent symbols whose absence nothing accounts for -- the real data holes.

    Two things account for an absence, and they are different questions.

    **The bars.** A symbol whose series does not reach the window had not begun
    trading; one whose series ended before it could not be traded during it. In
    neither case is there anything to fetch, and neither is a defect.

    **The table.** A symbol that was not in the index at any point in the span
    is *supposed* to be absent, whatever its bars do. See
    :meth:`PointInTimeUniverse.unexplained_absences`.

    What is left over is a name that could have traded and should have been in
    the index, with no bars: a hole, and still a hard failure.

    Both halves earn their place. Without the table, failing on every absence
    discarded ten folds of eleven on the 680-name point-in-time S&P universe --
    a name that had not listed yet is absent by construction -- and every
    strategy scored a ``positive_fold_rate`` of exactly 9%, which measured the
    calendar and not the rule. Without the bars, seven names still failed:
    ALXN, CXO, ETFC, FLIR, NBL, TIF and VAR were all acquired, and Wikipedia
    records the index removal one to ten days after the stock's last trade. The
    table calls them members for a few days in which they could not be bought.
    """
    unresolved: list[str] = []
    for symbol in missing:
        bars = data.get(symbol)
        if bars is not None and len(bars) and (
            int(bars.event_time[-1]) < start_ts or int(bars.event_time[0]) >= stop_ts
        ):
            continue
        unresolved.append(symbol)
    if membership is None:
        return unresolved
    return membership.unexplained_absences(
        unresolved,
        datetime.fromtimestamp(start_ts, tz=UTC).date(),
        datetime.fromtimestamp(stop_ts, tz=UTC).date(),
    )


def _narrowed(spec: StrategySpec, available: dict[str, Bars]) -> StrategySpec:
    """The spec restricted to the names that exist in this window.

    Only ever called once every absence has been accounted for, so this drops
    names that could not have been traded rather than names that are missing.
    The engine would narrow the universe anyway; doing it here keeps
    ``run_portfolio``'s own guard meaningful for callers that have not already
    answered the question.
    """
    symbols = tuple(s for s in spec.universe.symbols if s in available)
    if len(symbols) == len(spec.universe.symbols):
        return spec
    return spec.with_params(
        universe=spec.universe.__class__(
            symbols=symbols, timeframe=spec.universe.timeframe
        ),
        hold=min(spec.hold, max(1, len(symbols) - 1)),
    )


def _run_window(
    spec: StrategySpec,
    data: dict[str, Bars],
    start_ts: int,
    stop_ts: int,
    warmup: int,
    config: BacktestConfig,
    membership: PointInTimeUniverse | None = None,
) -> tuple[Metrics, BacktestResult]:
    """Backtest one window; trades opened during warm-up do not count."""
    sliced, cutoff = _window(data, start_ts, stop_ts, warmup)
    missing = [s for s in spec.universe.symbols if s not in sliced]
    if missing:
        if _unexplained_absences(missing, membership, data, start_ts, stop_ts):
            empty = _empty_result(spec, tuple(data))
            return compute_metrics(empty), empty
        spec = _narrowed(spec, sliced)

    result = run_strategy(spec, sliced, config, membership=membership)
    keep = int(np.searchsorted(result.timeline, cutoff, side="left"))
    if keep <= 0:
        return compute_metrics(result), result
    if keep >= result.timeline.size:
        empty = _empty_result(spec, tuple(sliced))
        return compute_metrics(empty), empty

    trimmed = BacktestResult(
        spec_fingerprint=result.spec_fingerprint,
        strategy=result.strategy,
        symbols=result.symbols,
        timeline=result.timeline[keep:],
        equity=result.equity[keep:],
        trades=[t for t in result.trades if t.entry_time >= cutoff],
        warmup_bars=result.warmup_bars,
        halted=result.halted,
        halt_reason=result.halt_reason,
    )
    return compute_metrics(trimmed), trimmed


def _empty_result(spec: StrategySpec, symbols: tuple[str, ...]) -> BacktestResult:
    return BacktestResult(
        spec_fingerprint=spec.fingerprint(),
        strategy=spec.name,
        symbols=symbols,
        timeline=np.array([], dtype=np.int64),
        equity=np.array([], dtype=np.float64),
        trades=[],
        warmup_bars=0,
    )


def _bounds(reference: Bars, split: Split) -> tuple[int, int]:
    """Positional split on the reference series -> a half-open time range."""
    n = len(reference)
    start = int(reference.event_time[min(split.start, n - 1)])
    if split.stop >= n:
        stop = int(reference.event_time[n - 1]) + 1
    else:
        stop = int(reference.event_time[split.stop])
    return start, stop


def run_walk_forward(
    spec: StrategySpec,
    data: dict[str, Bars] | Bars,
    *,
    train_bars: int = TRAIN_BARS,
    test_bars: int = TEST_BARS,
    grid: dict[str, list[float]] | None = None,
    anchored: bool = True,
    embargo_bars: int | None = None,
    config: BacktestConfig | None = None,
    zero_cost_config: BacktestConfig | None = None,
    membership: PointInTimeUniverse | None = None,
) -> WalkForwardReport:
    """Walk ``spec`` forward over ``data``.

    ``grid``          optional parameter search, run independently on each
                      fold's train window. Omit it to evaluate a fixed rule.
    ``embargo_bars``  defaults to the strategy's own holding period, so trades
                      opened at the end of a train window cannot resolve inside
                      the test window that follows.
    ``zero_cost_config``
                      when given, each fold's *test* window is run a second time
                      with these costs and the parameters already selected, and
                      the result chained into ``stitched_zero_cost``. Same
                      windows, same parameters, so the difference between the
                      two curves is friction and nothing else.
    """
    config = config or BacktestConfig()
    if isinstance(data, Bars):
        data = {data.symbol: data}
    traded = {s: data[s] for s in spec.universe.symbols if s in data}
    if not traded:
        raise ValueError(f"{spec.name}: no data for the universe {list(spec.universe.symbols)}")

    report = WalkForwardReport(
        strategy=spec.name, fingerprint=spec.fingerprint(), symbols=tuple(traded)
    )

    # The longest series defines the calendar; every symbol is cut to the same
    # dates, so a fold means the same period for all of them.
    reference = max(traded.values(), key=len)
    warmup = warmup_for(spec, traded)
    embargo = spec.exit.max_holding_bars if embargo_bars is None else embargo_bars

    folds = walk_forward_folds(
        len(reference),
        train_bars=train_bars,
        test_bars=test_bars,
        anchored=anchored,
        embargo_bars=embargo,
    )
    if not folds:
        return report

    combos = _expand_grid(grid or {})
    stitched_trades: list[Trade] = []
    segments: list[np.ndarray] = []
    stamps: list[np.ndarray] = []
    free_segments: list[np.ndarray] = []
    free_stamps: list[np.ndarray] = []

    for fold in folds:
        train_start, train_stop = _bounds(reference, fold.train)
        test_start, test_stop = _bounds(reference, fold.test)

        best_params: dict[str, float] = {}
        best_spec = spec
        best_train: Metrics | None = None
        best_score = float("-inf")

        for params in combos:
            candidate = apply_params(spec, params) if params else spec
            train_metrics, _ = _run_window(
                candidate, traded, train_start, train_stop, warmup, config, membership
            )
            score = _score(train_metrics)
            if score > best_score or best_train is None:
                best_score = score
                best_params = params
                best_spec = candidate
                best_train = train_metrics

        assert best_train is not None
        test_metrics, test_result = _run_window(
            best_spec, traded, test_start, test_stop, warmup, config, membership
        )
        report.folds.append(
            FoldResult(
                fold=fold,
                params=dict(best_params),
                train=best_train,
                test=test_metrics,
                test_trades=list(test_result.trades),
                candidates_tried=len(combos),
            )
        )
        stitched_trades.extend(test_result.trades)
        if test_result.equity.size > 1:
            segments.append(test_result.equity)
            stamps.append(test_result.timeline)

        if zero_cost_config is not None:
            # The same window and the same spec the selection already chose.
            # Re-selecting under zero costs would let the parameters move, and
            # the comparison would no longer isolate friction.
            _, free_result = _run_window(
                best_spec, traded, test_start, test_stop, warmup, zero_cost_config,
                membership,
            )
            if free_result.equity.size > 1:
                free_segments.append(free_result.equity)
                free_stamps.append(free_result.timeline)

    free = _stitch(free_segments, free_stamps, config.initial_equity)
    if free is not None:
        free_timeline, free_equity = free
        report.stitched_zero_cost = compute_metrics(
            BacktestResult(
                spec_fingerprint=spec.fingerprint(),
                strategy=spec.name,
                symbols=tuple(traded),
                timeline=free_timeline,
                equity=free_equity,
                trades=stitched_trades,
                warmup_bars=warmup,
            )
        )

    stitched = _stitch(segments, stamps, config.initial_equity)
    if stitched is not None:
        timeline, equity = stitched
        report.stitched_timeline = timeline
        report.stitched_equity = equity
        report.stitched = compute_metrics(
            BacktestResult(
                spec_fingerprint=spec.fingerprint(),
                strategy=spec.name,
                symbols=tuple(traded),
                timeline=timeline,
                equity=equity,
                trades=stitched_trades,
                warmup_bars=warmup,
            )
        )
    return report


def _stitch(
    segments: list[np.ndarray], times: list[np.ndarray], initial_equity: float
) -> tuple[np.ndarray, np.ndarray] | None:
    """Chain the fold test windows into one continuous equity curve.

    Each fold restarts from the same notional capital, so the segments are
    chained by *return*, not by level. This is what exposes a drawdown that
    spans a fold boundary -- something per-fold averages hide completely.
    """
    returns: list[np.ndarray] = []
    marks: list[np.ndarray] = []
    for equity, timeline in zip(segments, times, strict=True):
        if equity.size < 2:
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            step = np.diff(equity) / np.where(equity[:-1] == 0, np.nan, equity[:-1])
        good = np.isfinite(step)
        returns.append(step[good])
        marks.append(timeline[1:][good])
    if not returns:
        return None

    chained = np.concatenate(returns)
    stamps = np.concatenate(marks)
    if chained.size == 0:
        return None
    curve = initial_equity * np.cumprod(1.0 + chained)
    curve = np.insert(curve, 0, initial_equity)
    lead = stamps[0] - (stamps[1] - stamps[0] if stamps.size > 1 else 86_400)
    return np.insert(stamps, 0, lead), curve
