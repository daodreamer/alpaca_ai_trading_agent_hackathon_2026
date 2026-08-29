"""Robustness tests (architecture section 14).

A strategy that works only at exactly ``EMA(20)``, only on SPY, and only in a
bull market has not found an edge -- it has found a coincidence. These four
tests are the ones the architecture names, and each answers a different way of
being fooled:

``parameter_stability``  is the result a peak or a plateau?
``regime_robustness``    does it survive when the market changes character?
``asset_robustness``     is it a property of the market or of one ticker?
``monte_carlo``          how much of the equity curve was luck of ordering?

Each returns a score in [0, 1] alongside the evidence, because the evaluator
needs a number and a human needs the reason.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from aqr.backtest.engine import BacktestConfig, Trade
from aqr.backtest.metrics import Metrics, compute_metrics
from aqr.backtest.run import run_strategy
from aqr.data.bars import Bars
from aqr.data.universes import PointInTimeUniverse
from aqr.dsl.schema import StrategySpec
from aqr.seal import Contamination
from aqr.seal import current as current_seal
from aqr.validation.params import Slot, neighbours, slots

__all__ = [
    "UNSCORED_REGIME",
    "AssetReport",
    "MonteCarloReport",
    "ParameterReport",
    "RegimeReport",
    "asset_robustness",
    "monte_carlo",
    "parameter_stability",
    "regime_robustness",
    "score_regimes",
]

# The label a classifier uses for "there is not yet enough history to say".
# It is not a regime, and scoring it as one gives a strategy credit for being
# profitable during its own warm-up. Trades that fall in it stay visible in the
# report -- knowing how many there were is useful -- but they do not vote.
UNSCORED_REGIME = "UNKNOWN"

_DEFAULT_FACTORS = (0.75, 0.9, 1.1, 1.25)


def _sharpe(spec: StrategySpec, data: dict[str, Bars], config: BacktestConfig) -> Metrics:
    return compute_metrics(run_strategy(spec, data, config))


# --------------------------------------------------------------------------- #
# 14.1 Parameter perturbation
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class ParameterReport:
    baseline: float
    per_slot: dict[str, list[float]] = field(default_factory=dict)
    score: float = 0.0
    fragile: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "baseline_sharpe": self.baseline,
            "score": self.score,
            "fragile_parameters": self.fragile,
            "per_slot": self.per_slot,
        }

    def __str__(self) -> str:
        head = f"parameter stability {self.score:.2f} (baseline Sharpe {self.baseline:.2f})"
        if self.fragile:
            head += f" -- fragile: {', '.join(self.fragile)}"
        return head


def parameter_stability(
    spec: StrategySpec,
    data: dict[str, Bars],
    *,
    factors: tuple[float, ...] = _DEFAULT_FACTORS,
    config: BacktestConfig | None = None,
    only: list[str] | None = None,
) -> ParameterReport:
    """Nudge each parameter and see whether the edge survives.

    The score is the fraction of neighbouring parameter values that keep at
    least half the baseline Sharpe. A sharp peak -- good at 20, worthless at 19
    and 21 -- scores near zero, which is the architecture's worked example of
    severe overfitting.
    """
    config = config or BacktestConfig()
    baseline = _sharpe(spec, data, config).sharpe
    report = ParameterReport(baseline=baseline)

    candidates: list[Slot] = [s for s in slots(spec) if only is None or s.path in only]
    if not candidates:
        report.score = 1.0
        return report

    # A baseline with no edge has nothing to be fragile about; scoring it high
    # would let a flat strategy look robust.
    threshold = 0.5 * baseline if baseline > 0 else None
    survivors = 0
    total = 0

    for slot in candidates:
        results: list[float] = []
        for variant in neighbours(spec, slot, factors):
            try:
                sharpe = _sharpe(variant, data, config).sharpe
            except ValueError:
                continue
            results.append(sharpe)
            total += 1
            if threshold is not None and sharpe >= threshold:
                survivors += 1
        if results:
            report.per_slot[slot.path] = results
            if threshold is not None and max(results) < threshold:
                report.fragile.append(slot.path)

    report.score = float(survivors / total) if total else 1.0
    return report


# --------------------------------------------------------------------------- #
# 14.2 Market / regime test
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class RegimeReport:
    per_regime: dict[str, Metrics] = field(default_factory=dict)
    score: float = 0.0
    scored_regimes: tuple[str, ...] = ()
    """Which regimes actually voted: those with enough trades, excluding
    ``UNSCORED_REGIME``. Reported so that a score of 0.5 can be read as "one of
    two" instead of leaving the denominator a mystery."""

    def as_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "scored_regimes": list(self.scored_regimes),
            "per_regime": {k: v.as_dict() for k, v in self.per_regime.items()},
        }

    def __str__(self) -> str:
        rows = []
        for name, m in sorted(self.per_regime.items()):
            note = "" if name in self.scored_regimes else "   (not scored)"
            rows.append(
                f"  {name:<16} ret {m.total_return:+7.1%}  {m.num_trades:3d} trades{note}"
            )
        return "\n".join([f"regime robustness {self.score:.2f}", *rows])


def score_regimes(buckets: dict[str, list[Trade]], min_trades: int = 5) -> float:
    """Fraction of judged regimes in which the strategy made money.

    A regime with too few trades is not judged -- with three trades the answer
    is the sign of one of them. ``UNSCORED_REGIME`` is never judged at all: it
    means "not measured", and counting it would give a strategy credit for being
    profitable during its own warm-up.
    """
    scored = 0
    positive = 0
    for label, trades in buckets.items():
        if label == UNSCORED_REGIME or len(trades) < min_trades:
            continue
        scored += 1
        if sum(t.net_pnl for t in trades) > 0:
            positive += 1
    return float(positive / scored) if scored else 0.0


def regime_robustness(
    spec: StrategySpec,
    data: dict[str, Bars],
    labels: dict[str, list[str]],
    *,
    config: BacktestConfig | None = None,
    min_trades: int = 5,
) -> RegimeReport:
    """Attribute trades to the regime they were opened in.

    Splitting the *bars* by regime and backtesting each piece would fabricate
    entries and exits at every boundary. Attributing completed trades to the
    regime at entry keeps every trade real; the cost is that regimes with few
    trades are simply not judged.
    """
    config = config or BacktestConfig()
    result = run_strategy(spec, data, config)
    report = RegimeReport()
    if not result.trades:
        return report

    index: dict[str, dict[int, str]] = {}
    for symbol, series in labels.items():
        if symbol not in data:
            continue
        bars = data[symbol]
        if len(series) != len(bars):
            raise ValueError(
                f"{symbol}: {len(series)} regime labels for {len(bars)} bars"
            )
        index[symbol] = {int(t): series[i] for i, t in enumerate(bars.event_time)}

    buckets: dict[str, list[Trade]] = {}
    for trade in result.trades:
        label = index.get(trade.symbol, {}).get(trade.entry_time)
        if label is None:
            continue
        buckets.setdefault(label, []).append(trade)

    for label, trades in buckets.items():
        report.per_regime[label] = _metrics_from_trades(trades, config.initial_equity)
    report.scored_regimes = tuple(
        sorted(
            label
            for label, trades in buckets.items()
            if label != UNSCORED_REGIME and len(trades) >= min_trades
        )
    )
    report.score = score_regimes(buckets, min_trades)
    return report


def _metrics_from_trades(trades: list[Trade], equity: float) -> Metrics:
    """Trade-level metrics for a subset, without a synthetic equity curve.

    Curve-derived fields (CAGR, Sharpe, drawdown) are left at zero rather than
    invented from a reconstructed curve that never existed.
    """
    from aqr.backtest.metrics import _trade_stats  # local: private by intent

    pnl = np.array([t.net_pnl for t in trades], dtype=np.float64)
    stats = _trade_stats(trades)
    return Metrics(
        total_return=float(pnl.sum() / equity) if equity else 0.0,
        cagr=0.0,
        sharpe=0.0,
        sortino=0.0,
        max_drawdown=0.0,
        calmar=0.0,
        num_trades=len(trades),
        exposure=0.0,
        turnover=0.0,
        **stats,
    )


# --------------------------------------------------------------------------- #
# 14.3 Asset test
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class AssetReport:
    per_symbol: dict[str, Metrics] = field(default_factory=dict)
    score: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "per_symbol": {k: v.as_dict() for k, v in self.per_symbol.items()},
        }

    def __str__(self) -> str:
        rows = [
            f"  {sym:<8} sharpe {m.sharpe:6.2f}  ret {m.total_return:+7.1%}  "
            f"{m.num_trades:3d} trades"
            for sym, m in sorted(self.per_symbol.items())
        ]
        return "\n".join([f"asset robustness {self.score:.2f}", *rows])


def asset_robustness(
    spec: StrategySpec,
    data: dict[str, Bars],
    *,
    config: BacktestConfig | None = None,
    min_trades: int = 5,
    membership: PointInTimeUniverse | None = None,
    workers: int | None = None,
) -> AssetReport:
    """Ask whether the edge depends on one ticker, in the way the mode allows.

    Section 14.3's point: an edge tested only on SPY is a claim about SPY. How
    to ask it depends on what the strategy's unit of decision is.

    **Signal mode: one symbol at a time.** The rule opens positions per name, so
    running it on one name is a coherent question. The peer set stays the whole
    universe while the traded set shrinks, which for a peer-relative rule is the
    difference between a measurement and a fiction: ranked against itself every
    cross-sectional feature is undefined, the rule fires on nothing, every
    symbol is skipped, and the score comes out 0.0 -- not as a placeholder but
    as an active penalty for using the feature at all.

    **Portfolio mode: leave one out.** A book that holds ten names cannot be run
    on one. Weight per name is ``core_budget / hold``, so a single-symbol run
    puts 8% into the only name available, leaves the sleeve nowhere to go, and
    measures a portfolio that is 72% cash -- which is not an answer to the
    question. The universe minus one name is. If the edge survives every
    deletion it does not live in any single ticker; if one deletion destroys it,
    that name *was* the finding. Entries are keyed by the symbol **removed**.
    """
    config = config or BacktestConfig()
    if spec.mode == "portfolio":
        return _leave_one_out(
            spec,
            data,
            config=config,
            min_trades=min_trades,
            membership=membership,
            workers=workers,
        )

    report = AssetReport()
    scored = 0
    positive = 0
    for symbol, bars in data.items():
        single = spec.with_params(
            universe=spec.universe.__class__(symbols=(symbol,), timeframe=spec.universe.timeframe)
        )
        try:
            metrics = compute_metrics(
                run_strategy(single, {symbol: bars}, config, peers=dict(data))
            )
        except ValueError:
            continue
        report.per_symbol[symbol] = metrics
        if metrics.num_trades >= min_trades:
            scored += 1
            if metrics.total_return > 0:
                positive += 1
    report.score = float(positive / scored) if scored else 0.0
    return report


# One deletion per core. The 680 deletions a point-in-time universe asks for are
# independent pure functions of the same inputs, so the only reason to run them
# one after another is that nobody had written this yet.
#
# Two cores left free: a research loop should not make the machine it runs on
# unusable, and the twentieth core of twenty buys five percent.
DEFAULT_WORKERS = max(1, (os.cpu_count() or 2) - 2)

# Below this there is nothing to win: starting processes and handing each one a
# copy of the bars costs more than the deletions save.
MIN_PARALLEL_DELETIONS = 32


def _deletion_spec(spec: StrategySpec, dropped: str, kept_size: int) -> StrategySpec:
    """The strategy as it would have been had ``dropped`` never existed."""
    return spec.with_params(
        universe=spec.universe.__class__(
            symbols=tuple(s for s in spec.universe.symbols if s != dropped),
            timeframe=spec.universe.timeframe,
        ),
        # The peer set shrinks with the traded set here, deliberately: the
        # question is what the strategy would have been had that name never
        # existed, and a peer universe still containing it would answer a
        # different one.
        hold=min(spec.hold, max(1, kept_size)),
    )


def _run_deletion(
    spec: StrategySpec,
    data: dict[str, Bars],
    dropped: str,
    config: BacktestConfig,
    membership: PointInTimeUniverse | None,
) -> Metrics | None:
    """One deletion. A pure function of its arguments, which is what lets the
    caller decide whether to run it here or in another process."""
    kept = {symbol: bars for symbol, bars in data.items() if symbol != dropped}
    try:
        return compute_metrics(
            run_strategy(
                _deletion_spec(spec, dropped, len(kept)),
                kept,
                config,
                membership=membership,
            )
        )
    except ValueError:
        return None


# --------------------------------------------------------------------------- #
# The worker side
# --------------------------------------------------------------------------- #
#
# The seal is a per-process singleton, so a worker gets its own. That would be a
# hole if a worker could observe anything: its taint would die with it and never
# reach the parent ledger or hash chain, which is precisely the silent
# contamination the seal exists to make impossible.
#
# It cannot. A worker is handed bars the parent has already loaded and already
# put past its seal. It opens no file, reaches no provider, and builds no series
# of its own -- and unpickling a frozen slotted dataclass bypasses
# ``__post_init__``, where the sensor lives, so a worker could not observe
# anything even if it tried.
#
# That is the kind of "true today" that stops being true quietly, so it is
# checked rather than trusted, twice: every worker reports whether its own seal
# came back tainted and the parent refuses the whole result if any did, and
# ``TestLeaveOneOutIsSealNeutral`` pins the same property on the serial path.

_WORKER: dict[str, Any] = {}


def _worker_init(
    spec: StrategySpec,
    data: dict[str, Bars],
    config: BacktestConfig,
    membership: PointInTimeUniverse | None,
) -> None:
    """Hand a worker the inputs once, rather than with every deletion."""
    _WORKER["spec"] = spec
    _WORKER["data"] = data
    _WORKER["config"] = config
    _WORKER["membership"] = membership


def _worker_deletion(dropped: str) -> tuple[str, Metrics | None, bool]:
    metrics = _run_deletion(
        _WORKER["spec"],
        _WORKER["data"],
        dropped,
        _WORKER["config"],
        _WORKER["membership"],
    )
    return dropped, metrics, current_seal().tainted


def _parallel_deletions(
    spec: StrategySpec,
    data: dict[str, Bars],
    names: list[str],
    config: BacktestConfig,
    membership: PointInTimeUniverse | None,
    workers: int,
) -> dict[str, Metrics | None]:
    """Every deletion, across ``workers`` processes, keyed by the name removed."""
    results: dict[str, Metrics | None] = {}
    tainted: list[str] = []
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_worker_init,
        initargs=(spec, data, config, membership),
    ) as pool:
        for dropped, metrics, worker_tainted in pool.map(_worker_deletion, names):
            results[dropped] = metrics
            if worker_tainted:
                tainted.append(dropped)
    if tainted:
        raise Contamination(
            f"{len(tainted)} leave-one-out worker(s) returned a tainted seal "
            f"(first: {tainted[0]}). A worker is handed bars the parent has "
            f"already vetted and must materialise none of its own; something "
            f"in this path now does, and its taint cannot reach the ledger."
        )
    return results


def _leave_one_out(
    spec: StrategySpec,
    data: dict[str, Bars],
    *,
    config: BacktestConfig,
    min_trades: int,
    membership: PointInTimeUniverse | None = None,
    workers: int | None = None,
) -> AssetReport:
    """Re-run the book on the universe minus each name in turn.

    ``workers`` spreads the deletions over processes. The report does not depend
    on how many there are: each deletion is a pure function of the same inputs,
    and the answers are keyed by the name removed and read back in sorted order,
    so completion order cannot reach it. ``workers=1`` forces the serial path,
    and ``TestLeaveOneOutInParallel`` pins the two against each other.
    """
    report = AssetReport()
    if len(data) < 3:
        # Leaving one out of two leaves one, which is the degenerate
        # single-name book this exists to avoid. Report nothing rather than a
        # number that means nothing.
        return report

    names = sorted(data)
    count = DEFAULT_WORKERS if workers is None else max(1, workers)
    if count > 1 and len(names) >= MIN_PARALLEL_DELETIONS:
        results = _parallel_deletions(spec, data, names, config, membership, count)
    else:
        results = {
            dropped: _run_deletion(spec, data, dropped, config, membership)
            for dropped in names
        }

    scored = 0
    positive = 0
    for dropped in names:
        metrics = results.get(dropped)
        if metrics is None:
            continue
        report.per_symbol[dropped] = metrics
        if metrics.num_trades >= min_trades:
            scored += 1
            if metrics.total_return > 0:
                positive += 1
    report.score = float(positive / scored) if scored else 0.0
    return report


# --------------------------------------------------------------------------- #
# 14.4 Monte Carlo
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class MonteCarloReport:
    trials: int
    median_drawdown: float
    worst_drawdown: float
    p95_drawdown: float
    probability_of_ruin: float
    probability_of_loss: float
    median_return: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "trials": self.trials,
            "median_drawdown": self.median_drawdown,
            "p95_drawdown": self.p95_drawdown,
            "worst_drawdown": self.worst_drawdown,
            "probability_of_ruin": self.probability_of_ruin,
            "probability_of_loss": self.probability_of_loss,
            "median_return": self.median_return,
        }

    def __str__(self) -> str:
        return (
            f"monte carlo ({self.trials} trials): median DD {self.median_drawdown:.1%}, "
            f"p95 DD {self.p95_drawdown:.1%}, worst {self.worst_drawdown:.1%}, "
            f"P(loss) {self.probability_of_loss:.0%}, P(ruin) {self.probability_of_ruin:.1%}"
        )


def monte_carlo(
    trades: list[Trade],
    initial_equity: float = 100_000.0,
    *,
    trials: int = 1000,
    ruin_threshold: float = 0.5,
    seed: int = 20260826,
) -> MonteCarloReport:
    """Resample the trade sequence to separate edge from ordering luck.

    Trades are drawn *with* replacement, which additionally samples the
    uncertainty in the trade distribution itself, not merely its ordering. The
    realised drawdown is one path out of many; what an allocator needs to know
    is the shape of the rest of that distribution.

    The seed is fixed. A robustness test whose verdict changes between runs is
    not a test.
    """
    if not trades:
        return MonteCarloReport(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    returns = np.array([t.return_pct for t in trades], dtype=np.float64)
    weights = np.array(
        [t.entry_price * t.quantity / initial_equity for t in trades], dtype=np.float64
    )
    # Per-trade equity impact, so a big position's loss is not counted as if it
    # had been sized like a small one.
    impact = returns * np.clip(weights, 0.0, 1.0)

    rng = np.random.default_rng(seed)
    n = impact.size
    draws = rng.integers(0, n, size=(trials, n))
    paths = np.cumprod(1.0 + impact[draws], axis=1)

    peaks = np.maximum.accumulate(paths, axis=1)
    drawdowns = np.max((peaks - paths) / peaks, axis=1)
    finals = paths[:, -1]

    return MonteCarloReport(
        trials=trials,
        median_drawdown=float(np.median(drawdowns)),
        worst_drawdown=float(drawdowns.max()),
        p95_drawdown=float(np.percentile(drawdowns, 95)),
        probability_of_ruin=float(np.mean(drawdowns >= ruin_threshold)),
        probability_of_loss=float(np.mean(finals < 1.0)),
        median_return=float(np.median(finals) - 1.0),
    )
