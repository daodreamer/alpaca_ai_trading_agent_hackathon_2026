"""Can this pipeline recognise an edge that is actually there?

Two hundred hypotheses have been evaluated across five campaigns and not one
beat buy-and-hold. That is the expected outcome -- most quantitative research
finds nothing, and a system that says so is worth more than one that produces
false positives -- but it is also exactly what a *broken* pipeline produces. A
rejection machine and a correct evaluator are indistinguishable from the outside
until the evaluator is shown accepting something real.

So: data with an edge planted in it, by construction, that the DSL can express.
Price mean-reverts after a large down day, strongly enough to survive costs. If
the gauntlet cannot pass that, its rejections say nothing about the market.

This is a test of the *instrument*, not of any strategy. It is deliberately
generous -- the effect is far larger than anything real -- because the question
is whether the needle moves at all, not how sensitive it is.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from aqr.backtest.engine import run_backtest
from aqr.backtest.metrics import compute_metrics
from aqr.data.bars import Bars
from aqr.dsl.loader import loads
from aqr.pipeline import evaluate_candidate

SPEC = """
strategy:
  name: planted_reversal_v1
  hypothesis: A large down day is followed by a bounce, by construction.
  universe:
    symbols: [AAA, BBB, CCC, DDD]
    timeframe: 1D
  entry: roc(1) < -0.03
  exit:
    stop_loss: {type: atr, multiplier: 3.0, period: 14}
    take_profit: {type: risk_reward, ratio: 1.0}
    max_holding_bars: 3
  sizing:
    risk_per_trade: 0.01
    max_position_pct: 0.5
  max_positions: 4
"""


def _planted(symbol: str, seed: int, *, n: int = 4000, edge: float = 0.02) -> Bars:
    """A random walk in which a 3% down day is followed by a bounce.

    The bounce is added to the *next* bar, so nothing about bar ``t`` reveals it:
    a strategy has to actually take the position to collect it, which is what
    makes this a test of the pipeline rather than of a look-ahead.
    """
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0002, 0.018, n)
    for i in range(1, n):
        if steps[i - 1] < -0.03:
            steps[i] += edge

    close = 100.0 * np.exp(np.cumsum(steps))
    prev = np.concatenate(([100.0], close[:-1]))
    open_ = prev * (1.0 + rng.normal(0.0, 0.001, n))
    span = np.abs(steps) * close * 0.4 + close * 0.002
    base = datetime(2000, 1, 3, tzinfo=UTC)
    return Bars(
        symbol=symbol,
        timeframe="1D",
        event_time=np.array(
            [int((base + timedelta(days=i)).timestamp()) for i in range(n)], dtype=np.int64
        ),
        open=open_,
        high=np.maximum(open_, close) + span,
        low=np.minimum(open_, close) - span,
        close=close,
        volume=np.full(n, 3e6),
    )


@pytest.fixture(scope="module")
def planted() -> dict[str, Bars]:
    return {
        name: _planted(name, seed)
        for name, seed in (("AAA", 1), ("BBB", 2), ("CCC", 3), ("DDD", 4))
    }


@pytest.fixture(scope="module")
def flat() -> dict[str, Bars]:
    """The same generator with the edge removed. The control's control."""
    return {
        name: _planted(name, seed, edge=0.0)
        for name, seed in (("AAA", 1), ("BBB", 2), ("CCC", 3), ("DDD", 4))
    }


class TestTheEdgeIsThere:
    def test_the_planted_data_pays_the_rule(self, planted: dict[str, Bars]) -> None:
        metrics = compute_metrics(run_backtest(loads(SPEC), planted))
        assert metrics.num_trades > 100
        assert metrics.sharpe > 1.0, "the edge was not planted deeply enough to test with"

    def test_the_same_rule_finds_nothing_without_it(self, flat: dict[str, Bars]) -> None:
        """Same generator, same seeds, same rule. Only the planted bounce
        differs, so anything the rule earns here is the generator's drift."""
        metrics = compute_metrics(run_backtest(loads(SPEC), flat))
        assert metrics.sharpe < 0.5


class TestThePipelineSeesIt:
    def test_a_real_edge_is_not_rejected(self, planted: dict[str, Bars]) -> None:
        outcome = evaluate_candidate(loads(SPEC), planted, registry=None)
        assert outcome.rejected_early is None, outcome.rejected_early
        assert outcome.verdict != "REJECT", outcome.summary()

    def test_it_scores_well(self, planted: dict[str, Bars]) -> None:
        outcome = evaluate_candidate(loads(SPEC), planted, registry=None)
        assert outcome.score >= 55, f"scored {outcome.score:.1f}\n{outcome.summary()}"

    def test_it_survives_out_of_sample(self, planted: dict[str, Bars]) -> None:
        outcome = evaluate_candidate(loads(SPEC), planted, registry=None)
        assert outcome.walk_forward is not None
        assert outcome.walk_forward.oos_sharpe > 0.5
        assert outcome.walk_forward.positive_fold_rate > 0.6

    def test_it_beats_the_benchmark(self, planted: dict[str, Bars]) -> None:
        """The comparison that failed for all two hundred real hypotheses. If it
        could not be passed even by a planted edge, it would be measuring
        something other than what it claims."""
        outcome = evaluate_candidate(loads(SPEC), planted, registry=None)
        assert outcome.evaluation is not None
        assert outcome.evaluation.excess_sharpe is not None
        assert outcome.evaluation.beats_benchmark is True, (
            f"strategy {outcome.walk_forward.oos_sharpe:.2f} vs benchmark "
            f"{outcome.evaluation.benchmark_sharpe:.2f}"
        )


class TestTheControlIsRejected:
    def test_the_same_rule_on_edgeless_data_does_not_pass(
        self, flat: dict[str, Bars]
    ) -> None:
        """The other half of the instrument check. A pipeline that accepts the
        planted edge is only informative if it also declines the version with
        nothing in it."""
        outcome = evaluate_candidate(loads(SPEC), flat, registry=None)
        assert outcome.verdict in ("REJECT", "REVIEW"), outcome.summary()
        assert outcome.score < 55


class TestOverfittingDetectorIsNotTheBlocker:
    def test_a_real_edge_is_not_called_overfit(self, planted: dict[str, Bars]) -> None:
        outcome = evaluate_candidate(loads(SPEC), planted, registry=None)
        assert outcome.overfitting is not None
        assert outcome.overfitting.verdict != "LIKELY_OVERFIT", str(outcome.overfitting)
