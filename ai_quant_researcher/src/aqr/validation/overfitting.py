"""The overfitting detector (architecture section 16).

This is the module that argues *against* every strategy, and it is the reason
the research loop is allowed to run unattended. An LLM proposing hypotheses in a
tight loop will eventually produce something with a beautiful backtest. The only
defence is to make the cost of that search visible and to charge for it.

Nine signals, each returning a penalty in [0, 1] with a human-readable reason:

1.  parameter count relative to the number of trades
2.  the search cost -- how many backtests bought this result
3.  Sharpe inflation from the multiple-comparisons correction
4.  train vs out-of-sample gap
5.  concentration of profit in a handful of trades
6.  concentration in one asset
7.  concentration in one period
8.  sensitivity to transaction costs
9.  sample size

The verdict is not a veto; it is evidence attached to the experiment. A high
score with a stated reason is far more useful to the next iteration than a
silent rejection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from aqr.backtest.engine import BacktestResult, Trade
from aqr.backtest.metrics import Metrics, periods_per_year
from aqr.dsl.schema import StrategySpec

__all__ = ["OverfittingReport", "Signal", "deflated_sharpe", "detect_overfitting"]


@dataclass(frozen=True, slots=True)
class Signal:
    name: str
    penalty: float
    reason: str

    def __str__(self) -> str:
        return f"{self.name:<24} {self.penalty:4.2f}  {self.reason}"


@dataclass(slots=True)
class OverfittingReport:
    signals: list[Signal] = field(default_factory=list)
    score: float = 0.0
    """0 = no evidence of overfitting, 1 = overwhelming."""

    @property
    def verdict(self) -> str:
        if self.score >= 0.6:
            return "LIKELY_OVERFIT"
        if self.score >= 0.35:
            return "SUSPECT"
        return "PLAUSIBLE"

    @property
    def worst(self) -> list[Signal]:
        return sorted(self.signals, key=lambda s: -s.penalty)[:3]

    def as_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "verdict": self.verdict,
            "signals": [
                {"name": s.name, "penalty": s.penalty, "reason": s.reason} for s in self.signals
            ],
        }

    def __str__(self) -> str:
        rows = [str(s) for s in sorted(self.signals, key=lambda s: -s.penalty)]
        return "\n".join([f"overfitting {self.score:.2f} -> {self.verdict}", *rows])


def deflated_sharpe(
    observed: float, trials: int, sample: int, periods_per_year: float = 252.0
) -> float:
    """Sharpe adjusted for how many attempts it took to find it.

    Searching N independent strategies over a sample of n returns produces a
    best-of-N Sharpe near ``sqrt(2 ln N / n)`` even when every one of them is
    worthless. Subtracting that expectation is the difference between "we found
    an edge" and "we ran a lot of backtests".

    ``periods_per_year`` must be the same factor the observed Sharpe was
    annualised with. It used to be the literal 252 while the observed Sharpe
    was annualised from the bar spacing, which left the two halves of one
    subtraction in different units -- and on 5-minute bars deflated about
    seventy-eight times too little, in the one direction that matters.

    A simplification of Bailey & Lopez de Prado's deflated Sharpe: it assumes
    independent trials, which overstates the deflation when candidates are
    correlated variants of one idea. Erring toward scepticism is the intended
    direction of the error.
    """
    if trials <= 1 or sample <= 1:
        return observed
    factor = periods_per_year if periods_per_year > 0 else 252.0
    expected_max = math.sqrt(2.0 * math.log(trials) / sample)
    return observed - expected_max * math.sqrt(factor)


def _clip(value: float) -> float:
    return float(min(max(value, 0.0), 1.0))


def detect_overfitting(
    spec: StrategySpec,
    result: BacktestResult,
    metrics: Metrics,
    *,
    train_sharpe: float | None = None,
    oos_sharpe: float | None = None,
    backtests_run: int = 1,
    zero_cost_sharpe: float | None = None,
) -> OverfittingReport:
    """Weigh the evidence that ``metrics`` is an artefact rather than an edge.

    ``backtests_run``      how many candidates were evaluated to arrive here.
                           This is the single most informative input and the one
                           most often omitted; the experiment database exists so
                           it can be supplied honestly.
    ``zero_cost_sharpe``   the same strategy under a frictionless model. A result
                           that only exists when costs are ignored is not a
                           result.
    """
    report = OverfittingReport()
    trades = result.trades
    n = len(trades)

    # 1. Degrees of freedom against evidence.
    params = spec.parameter_count()
    per_param = n / params if params else float(n)
    report.signals.append(
        Signal(
            "parameter_count",
            _clip(1.0 - per_param / 30.0),
            f"{params} parameters fitted against {n} trades ({per_param:.1f} trades per parameter)",
        )
    )

    # 2. The cost of the search.
    report.signals.append(
        Signal(
            "search_cost",
            _clip(math.log10(max(backtests_run, 1)) / 3.0),
            f"{backtests_run} backtest(s) evaluated to reach this result",
        )
    )

    # 3. Sharpe inflation.
    #
    # Both arguments are timeframe-aware on purpose. Six years is six years:
    # slicing it into 5-minute bars multiplies the sample count by seventy-
    # eight without buying any more evidence about whether an edge is real,
    # and the annualisation has to match the one the Sharpe already used.
    periods = int(result.equity.size)
    factor = periods_per_year(result.timeline)
    deflated = deflated_sharpe(metrics.sharpe, backtests_run, periods, factor)
    inflation = metrics.sharpe - deflated
    report.signals.append(
        Signal(
            "sharpe_inflation",
            _clip(inflation / max(abs(metrics.sharpe), 0.5)),
            f"Sharpe {metrics.sharpe:.2f} deflates to {deflated:.2f} after {backtests_run} trials",
        )
    )

    # 4. Train vs out-of-sample.
    if train_sharpe is not None and oos_sharpe is not None:
        gap = train_sharpe - oos_sharpe
        report.signals.append(
            Signal(
                "train_oos_gap",
                _clip(gap / 2.0),
                f"train Sharpe {train_sharpe:.2f} vs OOS {oos_sharpe:.2f} (gap {gap:+.2f})",
            )
        )

    # 5-7. Concentration: does the result rest on a few trades, one name, one month?
    report.signals.append(_profit_concentration(trades))
    report.signals.append(_asset_concentration(trades))
    report.signals.append(_time_concentration(trades))

    # 8. Cost sensitivity.
    if zero_cost_sharpe is not None:
        erosion = 1.0 - metrics.sharpe / zero_cost_sharpe if zero_cost_sharpe > 0 else 0.0
        report.signals.append(
            Signal(
                "cost_sensitivity",
                _clip(erosion),
                f"Sharpe {zero_cost_sharpe:.2f} frictionless vs {metrics.sharpe:.2f} with costs",
            )
        )

    # 9. Sample size. Everything above is unreliable without this one.
    report.signals.append(
        Signal(
            "sample_size",
            _clip(1.0 - n / 100.0),
            f"{n} trades" + (" -- too few to conclude anything" if n < 30 else ""),
        )
    )

    report.score = float(np.mean([s.penalty for s in report.signals]))
    return report


def _profit_concentration(trades: list[Trade]) -> Signal:
    """How much of the profit came from the best few trades."""
    if not trades:
        return Signal("profit_concentration", 1.0, "no trades")
    pnl = np.array([t.net_pnl for t in trades], dtype=np.float64)
    total = pnl.sum()
    if total <= 0:
        return Signal("profit_concentration", 0.0, "no profit to concentrate")
    top = int(max(1, round(0.05 * pnl.size)))
    share = float(np.sort(pnl)[-top:].sum() / total)
    return Signal(
        "profit_concentration",
        _clip((share - 0.3) / 0.7),
        f"top {top} of {pnl.size} trades produced {share:.0%} of net profit",
    )


def _asset_concentration(trades: list[Trade]) -> Signal:
    if not trades:
        return Signal("asset_concentration", 1.0, "no trades")
    symbols = {t.symbol for t in trades}
    if len(symbols) == 1:
        return Signal(
            "asset_concentration", 0.5, f"all trades on one symbol ({next(iter(symbols))})"
        )
    by_symbol: dict[str, float] = {}
    for trade in trades:
        by_symbol[trade.symbol] = by_symbol.get(trade.symbol, 0.0) + trade.net_pnl
    total = sum(v for v in by_symbol.values() if v > 0)
    if total <= 0:
        return Signal("asset_concentration", 0.0, "no profit to concentrate")
    best = max(by_symbol.values())
    share = best / total
    return Signal(
        "asset_concentration",
        _clip((share - 1.0 / len(symbols)) / (1.0 - 1.0 / len(symbols))),
        f"largest of {len(symbols)} symbols contributed {share:.0%} of gross profit",
    )


def _time_concentration(trades: list[Trade]) -> Signal:
    """Whether the equity curve is one good quarter and a long flat line."""
    if len(trades) < 4:
        return Signal("time_concentration", 0.5, "too few trades to assess timing")
    times = np.array([t.exit_time for t in trades], dtype=np.float64)
    pnl = np.array([t.net_pnl for t in trades], dtype=np.float64)
    total = pnl.sum()
    if total <= 0:
        return Signal("time_concentration", 0.0, "no profit to concentrate")
    span = times.max() - times.min()
    if span <= 0:
        return Signal("time_concentration", 1.0, "every trade closed at the same instant")
    buckets = min(10, max(2, len(trades) // 5))
    edges = np.linspace(times.min(), times.max() + 1, buckets + 1)
    sums = np.array(
        [pnl[(times >= edges[i]) & (times < edges[i + 1])].sum() for i in range(buckets)]
    )
    share = float(sums.max() / total)
    return Signal(
        "time_concentration",
        _clip((share - 1.0 / buckets) / (1.0 - 1.0 / buckets)),
        f"best of {buckets} periods produced {share:.0%} of net profit",
    )
