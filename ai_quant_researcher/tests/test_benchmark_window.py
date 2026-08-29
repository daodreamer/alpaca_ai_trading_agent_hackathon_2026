"""The benchmark has to cover the period the strategy was judged on.

A methodological hole in the first version, found while reading a result rather
than a test. The strategy's number is ``oos_sharpe``: the mean over walk-forward
*test* folds, which begin only after the first training window. The benchmark
was computed over the whole loaded window.

On a 1995-2026 pull that gap is the first 756 bars -- 1995 through 1998, three of
the strongest years technology stocks have ever had. Every "lost to buy and hold"
line was comparing a strategy measured from 1998 against an index measured from
1995, and the error ran in the direction that flatters the benchmark.

Either direction would have been wrong. A comparison whose two halves cover
different decades is not a comparison, and the fact that it happened to support
the conclusion already reached is exactly why it needed a test.
"""

from __future__ import annotations

import numpy as np
import pytest

from aqr.data.bars import Bars
from aqr.dsl.loader import loads
from aqr.validation.holdout import buy_and_hold

SPEC = """
strategy:
  name: window_probe_v1
  universe:
    symbols: [SPY, QQQ, IWM]
    timeframe: 1D
  entry: rsi(14) < 45
  exit:
    stop_loss: {type: atr, multiplier: 2.0, period: 14}
    max_holding_bars: 20
  sizing:
    risk_per_trade: 0.005
"""


class TestWindowSlicing:
    def test_the_benchmark_can_be_restricted_to_a_bar_range(
        self, universe: dict[str, Bars]
    ) -> None:
        whole = buy_and_hold(universe)
        later = buy_and_hold({s: b.slice(756, len(b)) for s, b in universe.items()})
        assert whole is not None and later is not None
        assert whole.sharpe != later.sharpe

    def test_omitting_the_training_years_changes_the_answer(
        self, universe: dict[str, Bars]
    ) -> None:
        """The size of the hole, on the fixture. If this ever comes out equal,
        the slicing has stopped doing anything."""
        whole = buy_and_hold(universe)
        later = buy_and_hold({s: b.slice(756, len(b)) for s, b in universe.items()})
        assert whole is not None and later is not None
        assert abs(whole.sharpe - later.sharpe) > 0.01


class TestPipelineUsesTheOosWindow:
    def test_the_recorded_benchmark_matches_the_out_of_sample_span(
        self, universe: dict[str, Bars]
    ) -> None:
        from aqr.pipeline import evaluate_candidate

        spec = loads(SPEC)
        outcome = evaluate_candidate(
            spec, universe, registry=None, run_robustness=False, train_bars=756, test_bars=252
        )
        assert outcome.walk_forward is not None and outcome.walk_forward.folds
        assert outcome.benchmark is not None

        first_test = outcome.walk_forward.folds[0].fold.test.start
        expected = buy_and_hold({s: b.slice(first_test, len(b)) for s, b in universe.items()})
        assert expected is not None
        assert outcome.benchmark.sharpe == pytest.approx(expected.sharpe, rel=1e-9)

    def test_it_is_not_the_whole_window(self, universe: dict[str, Bars]) -> None:
        from aqr.pipeline import evaluate_candidate

        outcome = evaluate_candidate(
            loads(SPEC), universe, registry=None, run_robustness=False
        )
        whole = buy_and_hold(universe)
        assert outcome.benchmark is not None and whole is not None
        assert outcome.benchmark.sharpe != pytest.approx(whole.sharpe)

    def test_without_folds_there_is_no_benchmark_to_report(
        self, universe: dict[str, Bars]
    ) -> None:
        """A strategy rejected before the walk-forward has no out-of-sample
        period, so there is no window to measure an index over either."""
        from aqr.pipeline import evaluate_candidate

        short = {s: b.slice(0, 300) for s, b in universe.items()}
        outcome = evaluate_candidate(loads(SPEC), short, registry=None, run_robustness=False)
        assert outcome.rejected_early
        assert outcome.benchmark is None


class TestFairness:
    def test_both_numbers_start_on_the_same_bar(self, universe: dict[str, Bars]) -> None:
        """The property the whole thing rests on, stated as a test rather than
        as a comment someone can stop believing."""
        from aqr.pipeline import evaluate_candidate

        outcome = evaluate_candidate(
            loads(SPEC), universe, registry=None, run_robustness=False
        )
        assert outcome.walk_forward is not None
        assert outcome.benchmark is not None

        start = outcome.walk_forward.folds[0].fold.test.start
        sliced = {s: b.slice(start, len(b)) for s, b in universe.items()}
        reference = buy_and_hold(sliced)
        assert reference is not None
        assert outcome.benchmark.total_return == pytest.approx(reference.total_return, rel=1e-9)


def test_slicing_preserves_alignment(universe: dict[str, Bars]) -> None:
    sliced = {s: b.slice(756, len(b)) for s, b in universe.items()}
    for symbol, bars in sliced.items():
        assert len(bars) == len(universe[symbol]) - 756
        assert np.all(np.diff(bars.event_time) > 0)
