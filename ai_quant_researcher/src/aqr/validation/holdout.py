"""Re-testing a strategy on data the search never touched.

Every other measurement in this project is taken on the data that selected the
strategy. Walk-forward helps -- each test fold is unseen *within* a run -- but
the universe, the vendor and the window were fixed before the first hypothesis
was proposed, and by the end of four campaigns 153 hypotheses had been tried
against them. A best-of-153 result on one universe is a claim about that
universe.

A holdout is the one test the search cannot have leaked into: different symbols,
a different decade, or a different vendor. It answers the question the score
cannot, because the score is computed from the bars that chose the strategy.

**The benchmark is not optional.** Fourteen strategies promoted to PAPER kept
a positive Sharpe on twenty-eight symbols they had never been selected on --
12 of 14, which reads like a success and was reported as one. Equal-weight
buy-and-hold on the same symbols over the same window returned +4263% at
Sharpe 1.16; the best strategy returned +57.5% at Sharpe 0.65. Every one of
them lost to doing nothing, and nothing in the pipeline had asked.

A long-only rule on NASDAQ names over 2010-2026 makes money almost whatever
the rule is. Without the benchmark printed beside it, that fact reads as
evidence about the rule. So the benchmark is computed here, always, from the
same bars over the same window.

The asymmetry is worth stating. A strategy that survives a holdout is not
thereby proven -- one holdout is one sample, and surviving it is what a lucky
strategy does one time in however many. A strategy that collapses has said
something definite. The second outcome is both more common and more useful, and
this module is built expecting it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from aqr.backtest.engine import BacktestConfig
from aqr.backtest.metrics import Metrics, compute_metrics
from aqr.backtest.run import run_strategy
from aqr.data.bars import Bars
from aqr.dsl.schema import StrategySpec, Universe

__all__ = ["HoldoutResult", "buy_and_hold", "run_holdout"]


@dataclass(frozen=True, slots=True)
class HoldoutResult:
    """What one rule did on bars it was never selected against."""

    strategy: str
    fingerprint: str
    symbols: tuple[str, ...]
    traded: bool
    metrics: Metrics | None = None
    selected_sharpe: float | None = None
    """Sharpe on the data that chose this strategy, if the caller knows it."""
    benchmark: Metrics | None = None
    """Equal-weight buy-and-hold over the same symbols and window: what the
    holdout period paid for doing nothing at all."""
    note: str = ""

    @property
    def beat_benchmark(self) -> bool | None:
        """Whether the rule beat holding the same symbols, risk-adjusted.

        On Sharpe, not on total return. A rule in the market a tenth of the
        time will lose on return to one that is always in it, and that on its
        own says nothing about whether the rule is any good.
        """
        if self.metrics is None or self.benchmark is None:
            return None
        return bool(self.metrics.sharpe > self.benchmark.sharpe)

    @property
    def retained(self) -> float | None:
        """Share of the selected-on Sharpe that survived, or ``None``.

        ``None`` when there is nothing to compare against, and also when the
        selected-on Sharpe was not positive: dividing by a number that was never
        an edge produces a ratio that looks like one.
        """
        if self.metrics is None or self.selected_sharpe is None:
            return None
        if self.selected_sharpe <= 0:
            return None
        return float(self.metrics.sharpe / self.selected_sharpe)

    def __str__(self) -> str:
        head = f"{self.strategy} [{self.fingerprint}] holdout on {len(self.symbols)} symbol(s)"
        if self.symbols:
            head += f": {', '.join(self.symbols[:8])}"
            if len(self.symbols) > 8:
                head += f" +{len(self.symbols) - 8} more"
        if not self.traded:
            return f"{head}\n  no trades -- {self.note}"
        assert self.metrics is not None
        line = (
            f"  sharpe {self.metrics.sharpe:+5.2f}  return {self.metrics.total_return:+7.1%}  "
            f"maxDD {self.metrics.max_drawdown:5.1%}  {self.metrics.num_trades:4d} trades"
        )
        retained = self.retained
        if retained is not None:
            line += f"  |  kept {retained:+.0%} of the selected-on Sharpe"
        lines = [head, line]
        if self.benchmark is not None:
            verdict = "BEAT" if self.beat_benchmark else "LOST TO"
            lines.append(
                f"  {verdict} buy and hold: sharpe {self.benchmark.sharpe:+5.2f}  "
                f"return {self.benchmark.total_return:+8.1%}  "
                f"maxDD {self.benchmark.max_drawdown:5.1%}"
            )
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "fingerprint": self.fingerprint,
            "symbols": list(self.symbols),
            "traded": self.traded,
            "metrics": self.metrics.as_dict() if self.metrics else None,
            "selected_sharpe": self.selected_sharpe,
            "retained": self.retained,
            "benchmark": self.benchmark.as_dict() if self.benchmark else None,
            "beat_benchmark": self.beat_benchmark,
            "note": self.note,
        }


def run_holdout(
    spec: StrategySpec,
    data: dict[str, Bars],
    *,
    selected_sharpe: float | None = None,
    config: BacktestConfig | None = None,
) -> HoldoutResult:
    """Run ``spec`` unchanged over ``data``, whatever universe it named.

    The rule is not modified -- only the instruments it is pointed at -- so the
    fingerprint is preserved and the result can be filed against the original
    experiment. Substituting the universe is the whole mechanism: a strategy
    fitted to twenty symbols has to be asked about thirty others by *some* means,
    and rewriting the rule would be asking about a different strategy.
    """
    base = HoldoutResult(
        strategy=spec.name,
        fingerprint=spec.fingerprint(),
        symbols=tuple(sorted(data)),
        traded=False,
        selected_sharpe=selected_sharpe,
    )
    if not data:
        return _with(base, note="no data for any held-out symbol")

    pointed = spec.with_params(
        universe=Universe(symbols=tuple(sorted(data)), timeframe=spec.universe.timeframe)
    )
    try:
        result = run_strategy(pointed, data, config or BacktestConfig())
    except ValueError as exc:
        return _with(base, note=str(exc))

    benchmark = buy_and_hold(data)
    metrics = compute_metrics(result)
    if metrics.num_trades == 0:
        # Distinct from losing money. Reporting a zero return here invites
        # "it lost its edge elsewhere", which is not what the data said.
        return _with(base, note="the rule never fired on the held-out symbols")
    return HoldoutResult(
        strategy=spec.name,
        fingerprint=spec.fingerprint(),
        symbols=tuple(result.symbols),
        traded=True,
        metrics=metrics,
        selected_sharpe=selected_sharpe,
        benchmark=benchmark,
    )


def benchmark_returns(data: dict[str, Bars]) -> tuple[np.ndarray, np.ndarray] | None:
    """Equal-weight portfolio returns per session, and the sessions.

    Split out of :func:`buy_and_hold` because the residual-alpha regression needs
    the series rather than its summary: a Sharpe difference cannot separate what
    a rule added from the exposure it ran, and only the returns can.

    Returns ``(sessions, portfolio_returns)`` where ``sessions`` has one more
    element than ``portfolio_returns`` -- there is no return on the first
    session, and inventing a flat day to make the arrays line up moves both the
    mean and the standard deviation.
    """
    if not data:
        return None
    sessions = np.unique(np.concatenate([b.session for b in data.values()]))
    if sessions.size < 3:
        return None
    closes = np.full((sessions.size, len(data)), np.nan)
    for column, (_symbol, bars) in enumerate(sorted(data.items())):
        closes[np.searchsorted(sessions, bars.session), column] = np.asarray(
            bars.close, dtype=np.float64
        )
    with np.errstate(divide="ignore", invalid="ignore"):
        step = closes[1:] / closes[:-1] - 1.0
    present = np.isfinite(step).sum(axis=1)
    totals = np.nansum(np.where(np.isfinite(step), step, 0.0), axis=1)
    # A session where every symbol is absent is a flat day for the portfolio,
    # not an undefined one.
    portfolio = np.where(present > 0, totals / np.maximum(present, 1), 0.0)
    if portfolio.size < 2:
        return None
    return sessions, portfolio


def buy_and_hold(data: dict[str, Bars]) -> Metrics | None:
    """Equal-weight, rebalanced daily, no costs -- what doing nothing paid.

    Deliberately generous to the benchmark: no transaction costs, no slippage,
    perfect rebalancing. A strategy that cannot beat an unrealistically good
    version of doing nothing is not close.

    A symbol contributes from its first bar, so a name that lists mid-window
    joins then rather than being back-filled at zero.
    """
    series = benchmark_returns(data)
    if series is None:
        return None
    sessions, portfolio = series

    equity = 100_000.0 * np.cumprod(1.0 + portfolio)
    ppy = _periods_per_year(sessions)

    # Computed here rather than through ``compute_metrics``. That function
    # refuses to report a ratio on fewer than five trades, which is right for a
    # strategy -- a Sharpe from three trades is noise -- and meaningless for
    # buy-and-hold, which has no trades by construction and thousands of daily
    # returns. Routing this curve through it reported Sharpe 0.00 on a series
    # that returned +4263%, and every comparison then read "BEAT buy and hold".
    std = float(portfolio.std(ddof=1))
    sharpe = float(portfolio.mean() / std * np.sqrt(ppy)) if std > 0 else 0.0
    downside = portfolio[portfolio < 0]
    dstd = float(downside.std(ddof=1)) if downside.size > 1 else 0.0
    sortino = float(portfolio.mean() / dstd * np.sqrt(ppy)) if dstd > 0 else sharpe

    total_return = float(np.prod(1.0 + portfolio) - 1.0)
    years = portfolio.size / ppy
    cagr = float((1.0 + total_return) ** (1.0 / years) - 1.0) if years > 0 else 0.0
    peak = np.maximum.accumulate(equity)
    max_dd = float(np.max((peak - equity) / np.where(peak > 0, peak, 1.0)))

    return Metrics(
        total_return=total_return,
        cagr=cagr,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=max_dd,
        calmar=float(cagr / max_dd) if max_dd > 1e-9 else 0.0,
        win_rate=0.0,
        profit_factor=0.0,
        expectancy=0.0,
        average_win=0.0,
        average_loss=0.0,
        num_trades=0,
        average_holding_bars=0.0,
        # Always in the market: that is the whole of the benchmark, and the
        # number a low-exposure strategy is really being compared against.
        exposure=1.0,
        turnover=0.0,
        total_fees=0.0,
        total_slippage=0.0,
        best_trade=0.0,
        worst_trade=0.0,
    )


def _periods_per_year(sessions: np.ndarray) -> float:
    """Annualisation factor from the session spacing, as the metrics layer does."""
    step = float(np.median(np.diff(sessions)))
    return 252.0 if step <= 0 else max(1.0, 252.0 / step)


def _with(base: HoldoutResult, *, note: str) -> HoldoutResult:
    return HoldoutResult(
        strategy=base.strategy,
        fingerprint=base.fingerprint,
        symbols=base.symbols,
        traded=False,
        selected_sharpe=base.selected_sharpe,
        note=note,
    )
