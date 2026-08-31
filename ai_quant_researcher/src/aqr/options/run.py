"""The one place that assembles a chain, an underlying and a volatility
history into a run — specs/10-options-research.md D9's diagram, made literal.

``backtest/run.py`` exists because there are two equity engines and eight
callers, and letting each caller branch on ``spec.mode`` is how a branch gets
missed silently. The options side has only one engine so far, but it already
has more than one caller that needs the same three objects assembled in the
same order: the walk-forward in ``options/walkforward.py`` slices a fold's
chain and re-runs; the robustness passes in ``options/robustness.py`` slice a
year or swap a DTE target and re-run; the six hand-built structures this spec
was written to let a person evaluate all call ``run_option_backtest`` with the
same three arguments in the same order. Three positional parameters in the
same order, called from four places, is exactly the shape that goes wrong
quietly when a fifth caller is added and gets the order right for
``run_option_backtest`` but wrong for a sibling that takes them differently.

:class:`OptionMarket` is the one names to remember instead of the one order to
remember. It also gives every caller downstream of a load (walk-forward,
robustness, the eventual agent proposer) one object to slice and pass around
rather than three that have to be kept in step with each other by hand.
"""

from __future__ import annotations

from dataclasses import dataclass

from aqr.data.bars import Bars
from aqr.options.chain import ChainIndex
from aqr.options.engine import OptionBacktestConfig, OptionBacktestResult, run_option_backtest
from aqr.options.features import VolatilityHistory
from aqr.options.spec import OptionSpec

__all__ = ["OptionMarket", "run_option_strategy"]


@dataclass(frozen=True, slots=True)
class OptionMarket:
    """Everything :func:`run_option_strategy` needs about one underlying.

    Not a loader and not a cache -- this module stays in ``options/``, a pure
    layer (``tests/test_boundaries.py``), so it never opens a file itself. A
    caller builds one from objects it already loaded (``ChainIndex.from_rows``,
    a bars provider, ``VolatilityHistory.from_rows``) exactly as it would have
    assembled the three positional arguments to ``run_option_backtest`` by
    hand; the difference is that the assembly happens once, at the call site
    that did the loading, instead of being re-derived at every place that
    wants to run a spec.
    """

    underlying: Bars
    chain: ChainIndex
    volatility: VolatilityHistory | None = None

    def with_chain(self, chain: ChainIndex) -> OptionMarket:
        """The same market, over a different slice of the chain.

        The one mutation a fold or a robustness pass actually needs: which
        sessions may fill an entry. ``underlying`` and ``volatility`` travel
        unchanged, because neither one gates *when* an entry can happen --
        the engine's own session loop walks ``chain.sessions`` and nothing
        else, so narrowing the chain is what narrows the window (see
        ``ChainIndex.slice_dates``), and both other series stay exactly as
        wide as the data actually is. A backward-looking feature (``sma(200)``,
        ``iv_rank()``) reads only its own history at the decision bar
        regardless of how much series sits after that bar in the array --
        that property is asserted directly in ``tests/test_option_features.py``
        and ``tests/test_causality.py``'s options-side equivalent -- so
        leaving the underlying and the volatility history at full width does
        not let a fold see anything past its own chain window; it only lets a
        position entered near a fold's edge find the real close it settles
        against, which every position needs regardless of which fold entered
        it.
        """
        return OptionMarket(underlying=self.underlying, chain=chain, volatility=self.volatility)


def run_option_strategy(
    spec: OptionSpec,
    market: OptionMarket,
    config: OptionBacktestConfig | None = None,
) -> OptionBacktestResult:
    """Simulate ``spec`` over ``market``.

    Returns the same :class:`~aqr.options.engine.OptionBacktestResult`
    ``run_option_backtest`` returns -- this function does no more than name
    its three positional arguments and hand them through, which is the whole
    point: a caller that has a market bundle never has to remember that the
    chain is second and the underlying is third.
    """
    return run_option_backtest(
        spec, market.chain, market.underlying, config, volatility=market.volatility
    )
