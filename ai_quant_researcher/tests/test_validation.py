"""Splits, walk-forward, robustness and the overfitting detector.

The theme: these are the components whose job is to say *no*, so the tests are
mostly about whether they say no when they should. A validation suite that
approves everything is worse than none, because it launders a guess into a
verified result.
"""

from __future__ import annotations

import numpy as np
import pytest

from aqr.backtest.engine import BacktestConfig, Trade, run_backtest
from aqr.backtest.metrics import compute_metrics
from aqr.dsl.schema import StrategySpec, Universe
from aqr.validation.overfitting import deflated_sharpe, detect_overfitting
from aqr.validation.params import apply_params, get_param, neighbours, set_param, slots
from aqr.validation.robustness import (
    asset_robustness,
    monte_carlo,
    parameter_stability,
    regime_robustness,
)
from aqr.validation.splits import purge_overlap, three_way_split, walk_forward_folds
from aqr.validation.walkforward import run_walk_forward


class TestSplits:
    def test_three_way_split_is_chronological_and_contiguous(self) -> None:
        train, validation, test = three_way_split(1000)
        assert (len(train), len(validation), len(test)) == (600, 200, 200)
        assert train.stop == validation.start
        assert validation.stop == test.start

    def test_walk_forward_windows_do_not_overlap(self) -> None:
        folds = walk_forward_folds(3000, train_bars=756, test_bars=252, embargo_bars=0)
        assert len(folds) > 3
        for a, b in zip(folds, folds[1:], strict=False):
            assert a.test.stop <= b.test.start, "test windows overlap; OOS is not OOS"

    def test_train_always_precedes_test(self) -> None:
        for fold in walk_forward_folds(3000, 756, 252, embargo_bars=20):
            assert fold.train.stop <= fold.test.start

    def test_embargo_leaves_a_gap(self) -> None:
        folds = walk_forward_folds(3000, 756, 252, embargo_bars=20)
        assert folds[0].test.start - folds[0].train.stop == 20

    def test_anchored_windows_grow_sliding_windows_do_not(self) -> None:
        anchored = walk_forward_folds(3000, 756, 252, anchored=True)
        sliding = walk_forward_folds(3000, 756, 252, anchored=False)
        assert all(f.train.start == 0 for f in anchored)
        assert len({len(f.train) for f in sliding}) == 1

    def test_too_little_history_yields_no_folds(self) -> None:
        assert walk_forward_folds(500, 756, 252) == []

    def test_purge_drops_folds_that_shrink_too_far(self) -> None:
        folds = walk_forward_folds(3000, 756, 252)
        purged = purge_overlap(folds, holding_bars=200)
        assert len(purged) <= len(folds)
        for fold in purged:
            assert len(fold.test) >= 1


class TestParams:
    def test_slots_finds_both_structured_and_literal_parameters(self, spec) -> None:
        paths = {s.path for s in slots(spec)}
        assert "exit.stop_loss.multiplier" in paths
        assert "entry#0" in paths  # the 20 in ema(20)
        assert "regime#0" in paths  # the 200 in ema(200)

    def test_setting_a_literal_rewrites_only_that_occurrence(self) -> None:
        spec = StrategySpec(
            name="dup",
            entry="ema(20) > 20 and rsi(20) > 40",
            universe=Universe(symbols=("SPY",)),
        )
        changed = set_param(spec, "entry#1", 25)
        assert changed.entry == "ema(20) > 25 and rsi(20) > 40"

    def test_integer_parameters_stay_integers(self, spec) -> None:
        changed = set_param(spec, "exit.max_holding_bars", 12.6)
        assert changed.exit.max_holding_bars == 13

    def test_apply_params_is_order_independent(self, spec) -> None:
        changes = {"exit.stop_loss.multiplier": 3.0, "exit.take_profit.ratio": 1.5}
        a = apply_params(spec, changes)
        b = apply_params(spec, dict(reversed(list(changes.items()))))
        assert a.fingerprint() == b.fingerprint()

    def test_get_param_returns_none_for_inapplicable_knobs(self, spec) -> None:
        from aqr.dsl.schema import TakeProfit

        no_target = spec.with_params(exit=spec.exit.__class__(take_profit=TakeProfit(type="none")))
        assert get_param(no_target, "exit.take_profit.ratio") is None

    def test_neighbours_drops_variants_that_cannot_be_built(self, spec) -> None:
        slot = next(s for s in slots(spec) if s.path == "exit.stop_loss.multiplier")
        variants = neighbours(spec, slot, (0.0, 0.5, 2.0))
        # A zero multiplier is invalid and must be silently skipped, not raised.
        assert all(v.exit.stop_loss.multiplier > 0 for v in variants)
        assert len({v.fingerprint() for v in variants}) == len(variants)


class TestWalkForward:
    def test_produces_folds_and_out_of_sample_metrics(self, spec, spy) -> None:
        report = run_walk_forward(spec, {"SPY": spy}, train_bars=756, test_bars=252)
        assert len(report.folds) >= 2
        assert report.stitched is not None
        assert 0.0 <= report.positive_fold_rate <= 1.0

    def test_a_grid_search_reports_the_parameters_it_chose(self, spec, spy) -> None:
        report = run_walk_forward(
            spec,
            {"SPY": spy},
            train_bars=756,
            test_bars=252,
            grid={"exit.stop_loss.multiplier": [1.5, 2.5]},
        )
        assert all(f.candidates_tried == 2 for f in report.folds)
        assert all(f.params["exit.stop_loss.multiplier"] in (1.5, 2.5) for f in report.folds)

    def test_an_oversized_grid_is_refused(self, spec, spy) -> None:
        huge = {"exit.stop_loss.multiplier": [1.0 + i / 100 for i in range(600)]}
        with pytest.raises(ValueError, match="data-mining"):
            run_walk_forward(spec, {"SPY": spy}, grid=huge)

    def test_is_deterministic(self, spec, spy) -> None:
        a = run_walk_forward(spec, {"SPY": spy})
        b = run_walk_forward(spec, {"SPY": spy})
        assert a.oos_sharpe == b.oos_sharpe
        assert [f.test.num_trades for f in a.folds] == [f.test.num_trades for f in b.folds]

    def test_multi_symbol_folds_cover_the_same_dates(self, spec, universe) -> None:
        wide = spec.with_params(universe=Universe(symbols=("SPY", "QQQ", "IWM")))
        report = run_walk_forward(wide, universe, train_bars=756, test_bars=252)
        assert report.symbols == ("SPY", "QQQ", "IWM")
        assert report.folds


class TestRobustness:
    def test_parameter_stability_is_a_fraction(self, spec, spy) -> None:
        report = parameter_stability(
            spec, {"SPY": spy}, only=["exit.stop_loss.multiplier", "exit.take_profit.ratio"]
        )
        assert 0.0 <= report.score <= 1.0
        assert report.per_slot

    def test_asset_robustness_tests_each_symbol_separately(self, spec, universe) -> None:
        report = asset_robustness(spec, universe)
        assert set(report.per_symbol) == set(universe)
        assert 0.0 <= report.score <= 1.0

    def test_regime_attribution_covers_the_trades_it_can_place(
        self, spec, universe, regimes
    ) -> None:
        report = regime_robustness(spec, {"SPY": universe["SPY"]}, {"SPY": regimes["SPY"]})
        total = sum(m.num_trades for m in report.per_regime.values())
        actual = len(run_backtest(spec, {"SPY": universe["SPY"]}, BacktestConfig()).trades)
        assert total == actual, "trades went missing during regime attribution"

    def test_mismatched_regime_labels_are_rejected(self, spec, universe) -> None:
        with pytest.raises(ValueError, match="regime labels"):
            regime_robustness(spec, {"SPY": universe["SPY"]}, {"SPY": ["TREND_BULL"] * 5})

    def test_monte_carlo_is_reproducible(self) -> None:
        trades = _fake_trades()
        a = monte_carlo(trades, trials=200)
        b = monte_carlo(trades, trials=200)
        assert a.p95_drawdown == b.p95_drawdown, "a robustness test that changes is not a test"

    def test_monte_carlo_p95_drawdown_is_at_least_the_median(self) -> None:
        report = monte_carlo(_fake_trades(), trials=500)
        assert report.p95_drawdown >= report.median_drawdown
        assert report.worst_drawdown >= report.p95_drawdown

    def test_monte_carlo_handles_no_trades(self) -> None:
        assert monte_carlo([], trials=10).trials == 0


class TestOverfitting:
    def test_deflation_grows_with_the_number_of_trials(self) -> None:
        one = deflated_sharpe(2.0, trials=1, sample=1000)
        many = deflated_sharpe(2.0, trials=1000, sample=1000)
        assert one == 2.0
        assert many < one, "searching 1000 strategies did not discount the winner"

    def test_a_thin_result_from_a_wide_search_is_flagged(self, spec, spy) -> None:
        result = run_backtest(spec, {"SPY": spy}, BacktestConfig())
        metrics = compute_metrics(result)
        honest = detect_overfitting(spec, result, metrics, backtests_run=1)
        mined = detect_overfitting(spec, result, metrics, backtests_run=5000)
        assert mined.score > honest.score
        assert any(s.name == "search_cost" for s in mined.signals)

    def test_a_large_train_oos_gap_is_flagged(self, spec, spy) -> None:
        result = run_backtest(spec, {"SPY": spy}, BacktestConfig())
        metrics = compute_metrics(result)
        report = detect_overfitting(
            spec, result, metrics, train_sharpe=2.5, oos_sharpe=0.1, backtests_run=10
        )
        gap = next(s for s in report.signals if s.name == "train_oos_gap")
        assert gap.penalty > 0.9

    def test_cost_sensitivity_is_flagged(self, spec, spy) -> None:
        result = run_backtest(spec, {"SPY": spy}, BacktestConfig())
        metrics = compute_metrics(result)
        report = detect_overfitting(
            spec, result, metrics, backtests_run=1, zero_cost_sharpe=metrics.sharpe + 2.0
        )
        cost = next(s for s in report.signals if s.name == "cost_sensitivity")
        assert cost.penalty > 0.5

    def test_every_penalty_is_bounded(self, spec, spy) -> None:
        result = run_backtest(spec, {"SPY": spy}, BacktestConfig())
        report = detect_overfitting(
            spec,
            result,
            compute_metrics(result),
            train_sharpe=9.0,
            oos_sharpe=-9.0,
            backtests_run=10**9,
            zero_cost_sharpe=50.0,
        )
        assert all(0.0 <= s.penalty <= 1.0 for s in report.signals)
        assert 0.0 <= report.score <= 1.0

    def test_verdict_thresholds(self, spec, spy) -> None:
        result = run_backtest(spec, {"SPY": spy}, BacktestConfig())
        report = detect_overfitting(spec, result, compute_metrics(result), backtests_run=1)
        assert report.verdict in ("PLAUSIBLE", "SUSPECT", "LIKELY_OVERFIT")


def _fake_trades(n: int = 60) -> list[Trade]:
    """A deterministic trade list with a mix of winners and losers."""
    rng = np.random.default_rng(11)
    out: list[Trade] = []
    for i in range(n):
        ret = float(rng.normal(0.004, 0.02))
        entry = 100.0
        exit_price = entry * (1 + ret)
        out.append(
            Trade(
                symbol="TEST",
                direction="long",
                entry_time=1_600_000_000 + i * 86_400 * 3,
                entry_price=entry,
                exit_time=1_600_000_000 + i * 86_400 * 3 + 86_400,
                exit_price=exit_price,
                quantity=100.0,
                gross_pnl=(exit_price - entry) * 100.0,
                fees=1.0,
                slippage=0.5,
                bars_held=1,
                exit_reason="target" if ret > 0 else "stop",
                mae=min(0.0, ret),
                mfe=max(0.0, ret),
            )
        )
    return out
