"""The one place that decides which engine runs a spec.

There are two engines and eight callers -- walk-forward, four robustness passes,
the holdout, the pipeline and the CLI. Letting each caller branch on
``spec.mode`` would work until one of them was missed, and the miss would be
silent: running a portfolio spec through the event-driven engine does not raise,
it simulates a strategy nobody wrote. So the branch lives here, once, and the
callers do not know there are two engines.
"""

from __future__ import annotations

from aqr.backtest.engine import BacktestConfig, BacktestResult, run_backtest
from aqr.backtest.portfolio import run_portfolio
from aqr.data.bars import Bars
from aqr.data.universes import PointInTimeUniverse
from aqr.dsl.schema import StrategySpec

__all__ = ["run_strategy"]


def run_strategy(
    spec: StrategySpec,
    data: dict[str, Bars] | Bars,
    config: BacktestConfig | None = None,
    *,
    peers: dict[str, Bars] | None = None,
    membership: PointInTimeUniverse | None = None,
) -> BacktestResult:
    """Simulate ``spec`` with whichever engine its mode calls for.

    Returns a :class:`BacktestResult` either way -- ``PortfolioResult`` is one --
    so metrics, stitching and the overfitting detector need no special case.

    ``membership`` is accepted for every spec and used only by the portfolio
    engine, which is the only one that has a concept of an index to belong to.
    Raising for a signal spec would mean every caller had to know which engine it
    was about to hit, which is the knowledge this function exists to hold.
    """
    if spec.mode == "portfolio":
        return run_portfolio(spec, data, config, peers=peers, membership=membership)
    return run_backtest(spec, data, config, peers=peers)
