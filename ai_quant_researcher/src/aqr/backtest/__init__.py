"""Causal simulation with costs. Pure: no LLM, no network, no wall clock."""

from aqr.backtest.costs import ZERO_COST, CostModel
from aqr.backtest.engine import BacktestConfig, BacktestResult, Trade, run_backtest
from aqr.backtest.metrics import Metrics, compute_metrics, drawdown_series

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "CostModel",
    "Metrics",
    "Trade",
    "ZERO_COST",
    "compute_metrics",
    "drawdown_series",
    "run_backtest",
]
