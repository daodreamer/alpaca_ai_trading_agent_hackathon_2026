"""Performance metrics (architecture section 10.1).

Two properties are load-bearing:

*Degenerate inputs return finite, honest values.* A run with two trades has no
meaningful Sharpe; it returns 0.0 rather than an inflated number produced by a
near-zero denominator. The strategy evaluator treats "not enough evidence" and
"no edge" the same way, which is the correct default when the alternative is
promoting noise.

*Everything is computed from the equity curve or the trade list*, never from the
strategy that produced them. A metric that knows what it is measuring is a
metric that can be gamed.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from aqr.backtest.engine import BacktestResult, Trade

__all__ = [
    "MIN_PERIODS_FOR_RATIOS",
    "MIN_TRADES_FOR_RATIOS",
    "periods_per_year", "Metrics", "compute_metrics", "drawdown_series"]

MIN_TRADES_FOR_RATIOS = 5
MIN_PERIODS_FOR_RATIOS = 20
"""Below either floor, ``sharpe`` and ``sortino`` are returned as 0.0 -- and
that zero means *declined to compute*, not *measured and flat*.

Public because a report that prints the number has to be able to tell those two
apart. An options walk-forward fold can hold two trades (specs/10 D8: a
structure held to expiry produces evidence only when it closes), and seven folds
each printing "sharpe 0.00" beside a stitched "0.96" reads as a contradiction
when it is actually a refusal. ``options/walkforward.py`` reads these to print
``n/a`` instead.
"""

# The old private spellings, kept so nothing outside this module breaks.
_MIN_TRADES_FOR_RATIOS = MIN_TRADES_FOR_RATIOS
_MIN_PERIODS_FOR_RATIOS = MIN_PERIODS_FOR_RATIOS


@dataclass(frozen=True, slots=True)
class Metrics:
    total_return: float
    cagr: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    win_rate: float
    profit_factor: float
    expectancy: float
    average_win: float
    average_loss: float
    num_trades: int
    average_holding_bars: float
    exposure: float
    turnover: float
    total_fees: float
    total_slippage: float
    best_trade: float
    worst_trade: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __str__(self) -> str:
        return (
            f"trades={self.num_trades} ret={self.total_return:+.1%} cagr={self.cagr:+.1%} "
            f"sharpe={self.sharpe:.2f} sortino={self.sortino:.2f} "
            f"maxdd={self.max_drawdown:.1%} pf={self.profit_factor:.2f} "
            f"win={self.win_rate:.0%}"
        )


def drawdown_series(equity: np.ndarray) -> np.ndarray:
    """Fractional drawdown from the running peak, at every point."""
    if equity.size == 0:
        return equity
    peak = np.maximum.accumulate(equity)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(peak > 0, (equity - peak) / peak, 0.0)


def periods_per_year(timeline: np.ndarray) -> float:
    """Infer the annualisation factor from the bar spacing itself.

    Reading it off the data rather than a config field means a 5-minute
    backtest cannot be accidentally annualised as if it were daily -- a mistake
    that inflates Sharpe by roughly eight times.

    Public because the overfitting detector needs the *same* number. Two
    copies of an annualisation factor is exactly how the observed Sharpe and
    its search-cost deflation came to be denominated in different units.
    """
    if timeline.size < 3:
        return 252.0
    step = float(np.median(np.diff(timeline)))
    if step <= 0:
        return 252.0
    seconds_per_trading_year = 252.0 * 6.5 * 3600.0
    if step >= 86_400:
        return max(1.0, 365.0 * 86_400.0 / step * (252.0 / 365.0))
    return max(1.0, seconds_per_trading_year / step)


def compute_metrics(result: BacktestResult) -> Metrics:
    equity = np.asarray(result.equity, dtype=np.float64)
    trades = result.trades
    start_equity = result.initial_equity or 1.0

    if equity.size == 0:
        return _empty()

    total_return = float(equity[-1] / start_equity - 1.0)

    ppy = periods_per_year(result.timeline)
    years = equity.size / ppy
    if years > 0 and equity[-1] > 0 and start_equity > 0:
        cagr = float((equity[-1] / start_equity) ** (1.0 / years) - 1.0)
    else:
        cagr = 0.0

    with np.errstate(divide="ignore", invalid="ignore"):
        rets = np.diff(equity) / np.where(equity[:-1] == 0, np.nan, equity[:-1])
    rets = rets[np.isfinite(rets)]

    enough = rets.size >= _MIN_PERIODS_FOR_RATIOS and len(trades) >= _MIN_TRADES_FOR_RATIOS
    sharpe = sortino = 0.0
    if enough:
        std = float(rets.std(ddof=1))
        if std > 0:
            sharpe = float(rets.mean() / std * math.sqrt(ppy))
        downside = rets[rets < 0]
        dstd = float(downside.std(ddof=1)) if downside.size > 1 else 0.0
        if dstd > 0:
            sortino = float(rets.mean() / dstd * math.sqrt(ppy))
        elif rets.mean() > 0:
            # No losing period at all. Real, but not a licence to report infinity.
            sortino = sharpe

    max_dd = float(-drawdown_series(equity).min()) if equity.size else 0.0
    calmar = float(cagr / max_dd) if max_dd > 1e-9 else 0.0

    stats = _trade_stats(trades)
    exposure = _exposure(result)
    turnover = _turnover(result, start_equity)

    return Metrics(
        total_return=total_return,
        cagr=cagr,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=max_dd,
        calmar=calmar,
        num_trades=len(trades),
        exposure=exposure,
        turnover=turnover,
        **stats,
    )


def _trade_stats(trades: list[Trade]) -> dict[str, float]:
    if not trades:
        return {
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "average_win": 0.0,
            "average_loss": 0.0,
            "average_holding_bars": 0.0,
            "total_fees": 0.0,
            "total_slippage": 0.0,
            "best_trade": 0.0,
            "worst_trade": 0.0,
        }
    pnl = np.array([t.net_pnl for t in trades], dtype=np.float64)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    gross_win = float(wins.sum())
    gross_loss = float(-losses.sum())
    # No losses at all is a small-sample artefact far more often than an edge,
    # so it reports as a high-but-finite factor rather than infinity.
    profit_factor = 0.0
    if gross_loss > 1e-9:
        profit_factor = gross_win / gross_loss
    elif gross_win > 0:
        profit_factor = 10.0
    return {
        "win_rate": float(wins.size / pnl.size),
        "profit_factor": float(profit_factor),
        "expectancy": float(pnl.mean()),
        "average_win": float(wins.mean()) if wins.size else 0.0,
        "average_loss": float(losses.mean()) if losses.size else 0.0,
        "average_holding_bars": float(np.mean([t.bars_held for t in trades])),
        "total_fees": float(sum(t.fees for t in trades)),
        "total_slippage": float(sum(t.slippage for t in trades)),
        "best_trade": float(pnl.max()),
        "worst_trade": float(pnl.min()),
    }


def _exposure(result: BacktestResult) -> float:
    """Fraction of the timeline during which capital was at risk."""
    if result.timeline.size == 0 or not result.trades:
        return 0.0
    held = np.zeros(result.timeline.size, dtype=bool)
    for trade in result.trades:
        start = int(np.searchsorted(result.timeline, trade.entry_time, side="left"))
        stop = int(np.searchsorted(result.timeline, trade.exit_time, side="right"))
        held[start:stop] = True
    return float(held.mean())


def _turnover(result: BacktestResult, start_equity: float) -> float:
    """Traded notional per year, as a multiple of starting equity."""
    if not result.trades or start_equity <= 0:
        return 0.0
    notional = sum(t.quantity * (t.entry_price + t.exit_price) for t in result.trades)
    ppy = periods_per_year(result.timeline)
    years = max(result.timeline.size / ppy, 1e-9)
    return float(notional / start_equity / years)


def _empty() -> Metrics:
    return Metrics(
        total_return=0.0,
        cagr=0.0,
        sharpe=0.0,
        sortino=0.0,
        max_drawdown=0.0,
        calmar=0.0,
        win_rate=0.0,
        profit_factor=0.0,
        expectancy=0.0,
        average_win=0.0,
        average_loss=0.0,
        num_trades=0,
        average_holding_bars=0.0,
        exposure=0.0,
        turnover=0.0,
        total_fees=0.0,
        total_slippage=0.0,
        best_trade=0.0,
        worst_trade=0.0,
    )
