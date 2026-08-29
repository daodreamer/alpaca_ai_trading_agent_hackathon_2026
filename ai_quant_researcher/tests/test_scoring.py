"""Metrics, the evaluator, and the registry.

Two failure modes are being guarded against here:

*Metrics that flatter small samples.* A two-trade run with no losses should not
report an infinite profit factor and a Sharpe of 6. Degenerate inputs must
produce finite, sceptical numbers.

*A scorer that can be talked around.* The evaluator's fatal conditions must be
fatal — no combination of strong components may rescue a strategy with eleven
trades or a 40% drawdown.
"""

from __future__ import annotations

import numpy as np
import pytest

from aqr.backtest.engine import BacktestConfig, BacktestResult, run_backtest
from aqr.backtest.metrics import compute_metrics, drawdown_series
from aqr.evaluator.score import WEIGHTS, EvaluationInput, evaluate_strategy
from aqr.registry.db import ExperimentRecord, Registry
from tests.test_validation import _fake_trades


def _result(equity: np.ndarray, trades=None) -> BacktestResult:
    n = equity.size
    return BacktestResult(
        spec_fingerprint="test",
        strategy="test",
        symbols=("TEST",),
        timeline=np.array([1_600_000_000 + i * 86_400 for i in range(n)], dtype=np.int64),
        equity=equity,
        trades=trades or [],
        warmup_bars=0,
    )


class TestMetrics:
    def test_drawdown_is_zero_on_a_monotonic_curve(self) -> None:
        assert drawdown_series(np.arange(1.0, 100.0)).min() == pytest.approx(0.0)

    def test_max_drawdown_matches_a_hand_computed_value(self) -> None:
        equity = np.array([100.0, 120.0, 60.0, 90.0])
        metrics = compute_metrics(_result(equity))
        assert metrics.max_drawdown == pytest.approx(0.5)  # 120 -> 60

    def test_total_return_matches_the_curve(self) -> None:
        metrics = compute_metrics(_result(np.array([100.0, 110.0, 150.0])))
        assert metrics.total_return == pytest.approx(0.5)

    def test_ratios_are_zero_without_enough_evidence(self) -> None:
        """Three trades cannot support a Sharpe, so none is reported."""
        equity = np.linspace(100.0, 130.0, 10)
        metrics = compute_metrics(_result(equity, _fake_trades(3)))
        assert metrics.sharpe == 0.0
        assert metrics.sortino == 0.0

    def test_profit_factor_is_finite_when_there_are_no_losses(self) -> None:
        winners = [t for t in _fake_trades(80) if t.net_pnl > 0]
        metrics = compute_metrics(_result(np.linspace(100.0, 200.0, 300), winners))
        assert np.isfinite(metrics.profit_factor)
        assert metrics.profit_factor <= 10.0

    def test_empty_result_is_all_zeros_not_an_exception(self) -> None:
        metrics = compute_metrics(_result(np.array([], dtype=np.float64)))
        assert metrics.num_trades == 0
        assert metrics.sharpe == 0.0

    def test_annualisation_is_inferred_from_bar_spacing(self) -> None:
        """A daily and an hourly curve with identical shape must not both be
        annualised as if they were daily."""
        equity = 100.0 * np.cumprod(1 + np.full(400, 0.001))
        trades = _fake_trades(40)
        daily = compute_metrics(_result(equity, trades))
        hourly = BacktestResult(
            spec_fingerprint="t",
            strategy="t",
            symbols=("T",),
            timeline=np.array(
                [1_600_000_000 + i * 3600 for i in range(equity.size)], dtype=np.int64
            ),
            equity=equity,
            trades=trades,
            warmup_bars=0,
        )
        assert compute_metrics(hourly).sharpe != pytest.approx(daily.sharpe)

    def test_exposure_is_a_fraction(self, spec, spy) -> None:
        metrics = compute_metrics(run_backtest(spec, {"SPY": spy}, BacktestConfig()))
        assert 0.0 <= metrics.exposure <= 1.0


def _input(**overrides) -> EvaluationInput:
    from aqr.backtest.alpha import ResidualAlpha
    from aqr.backtest.metrics import Metrics

    defaults = dict(
        total_return=0.4,
        cagr=0.12,
        sharpe=1.2,
        sortino=1.6,
        max_drawdown=0.12,
        calmar=1.0,
        win_rate=0.5,
        profit_factor=1.6,
        expectancy=100.0,
        average_win=400.0,
        average_loss=-250.0,
        num_trades=120,
        average_holding_bars=9.0,
        exposure=0.4,
        turnover=3.0,
        total_fees=500.0,
        total_slippage=300.0,
        best_trade=2000.0,
        worst_trade=-900.0,
    )
    defaults.update(overrides.pop("oos_fields", {}))
    data = EvaluationInput(
        strategy="candidate",
        fingerprint="abc123",
        oos=Metrics(**defaults),  # type: ignore[arg-type]
        oos_sharpe=1.2,
        positive_fold_rate=0.7,
        parameter_stability=0.8,
        asset_robustness=0.75,
        regime_robustness=0.6,
        # A candidate with no benchmark regression is rejected outright, so the
        # baseline fixture carries one that clears the gate. Tests about the
        # gate itself pass their own.
        residual=ResidualAlpha(
            alpha=0.05,
            beta=0.9,
            t_alpha=3.2,
            information_ratio=0.6,
            residual_vol=0.08,
            excess_sharpe=0.1,
            r_squared=0.7,
            observations=1500,
        ),
    )
    for key, value in overrides.items():
        setattr(data, key, value)
    return data


class TestEvaluator:
    def test_weights_sum_to_one(self) -> None:
        assert sum(WEIGHTS.values()) == pytest.approx(1.0)

    def test_a_solid_candidate_is_not_rejected(self) -> None:
        result = evaluate_strategy(_input())
        assert result.verdict in ("ACCEPT", "PAPER")
        assert result.score > 50

    def test_too_few_trades_is_fatal_regardless_of_score(self) -> None:
        result = evaluate_strategy(
            _input(oos_fields={"num_trades": 11}, oos_sharpe=3.0, parameter_stability=1.0)
        )
        assert result.verdict == "REJECT"
        assert any("below the 30 minimum" in r for r in result.reasons)

    def test_an_excessive_drawdown_is_fatal(self) -> None:
        result = evaluate_strategy(_input(oos_fields={"max_drawdown": 0.45}))
        assert result.verdict == "REJECT"

    def test_a_negative_oos_sharpe_is_fatal(self) -> None:
        assert evaluate_strategy(_input(oos_sharpe=-0.1)).verdict == "REJECT"

    def test_an_edge_that_only_exists_without_costs_is_fatal(self) -> None:
        result = evaluate_strategy(_input(oos_sharpe=0.4, zero_cost_sharpe=2.0))
        assert result.verdict == "REJECT"
        assert any("without spreads" in r for r in result.reasons)

    def test_in_sample_performance_cannot_rescue_a_weak_candidate(self) -> None:
        """There is no field for in-sample Sharpe. That is the point.

        The weakness is expressed through the information ratio rather than
        through ``oos_sharpe``, which no longer carries weight: on a rising
        universe a long rule's absolute Sharpe is mostly the market's.
        """
        from aqr.backtest.alpha import ResidualAlpha

        weak = evaluate_strategy(
            _input(
                oos_sharpe=0.05,
                parameter_stability=0.1,
                asset_robustness=0.0,
                residual=ResidualAlpha(
                    alpha=0.002,
                    beta=1.0,
                    t_alpha=2.1,
                    information_ratio=0.03,
                    residual_vol=0.07,
                    excess_sharpe=-0.4,
                    r_squared=0.9,
                    observations=1500,
                ),
            )
        )
        assert weak.score < 40

    def test_inconsistent_folds_are_reported_but_not_fatal(self) -> None:
        result = evaluate_strategy(_input(positive_fold_rate=0.2))
        assert any("not consistent across time" in r for r in result.reasons)
        assert result.verdict != "REJECT"

    def test_score_is_bounded(self) -> None:
        best = evaluate_strategy(
            _input(oos_sharpe=99.0, parameter_stability=5.0, asset_robustness=5.0)
        )
        assert 0.0 <= best.score <= 100.0


class TestRegistry:
    def test_re_registering_the_same_rule_does_not_duplicate_it(self, tmp_path, spec) -> None:
        with Registry(tmp_path / "r.sqlite") as reg:
            a = reg.upsert_strategy(spec)
            b = reg.upsert_strategy(spec.with_params(name="renamed"))
            assert a == b
            assert len(reg.strategies()) == 1

    def test_lifecycle_transitions_are_enforced(self, tmp_path, spec) -> None:
        with Registry(tmp_path / "r.sqlite") as reg:
            fingerprint = reg.upsert_strategy(spec)
            with pytest.raises(ValueError, match="cannot go"):
                reg.set_status(fingerprint, "LIVE")
            reg.set_status(fingerprint, "PAPER", "passed evaluation")
            reg.set_status(fingerprint, "LIVE", "30 days of paper")
            assert reg.get_strategy(fingerprint).status == "LIVE"

    def test_a_terminal_state_is_terminal(self, tmp_path, spec) -> None:
        with Registry(tmp_path / "r.sqlite") as reg:
            fingerprint = reg.upsert_strategy(spec)
            reg.set_status(fingerprint, "REJECTED", "no edge")
            with pytest.raises(ValueError, match="terminal"):
                reg.set_status(fingerprint, "PAPER")

    def test_backtest_count_accumulates_across_experiments(self, tmp_path, spec) -> None:
        with Registry(tmp_path / "r.sqlite") as reg:
            reg.upsert_strategy(spec)
            for i in range(3):
                reg.record_experiment(
                    ExperimentRecord(
                        fingerprint=spec.fingerprint(),
                        strategy_name=f"run{i}",
                        symbols=("SPY",),
                        timeframe="1D",
                        data_start="2010-01-01",
                        data_end="2024-01-01",
                        dataset_version="test",
                        backtests_run=10,
                    )
                )
            assert reg.total_backtests() == 30

    def test_has_tried_detects_a_repeat(self, tmp_path, spec) -> None:
        with Registry(tmp_path / "r.sqlite") as reg:
            assert not reg.has_tried(spec.fingerprint())
            reg.upsert_strategy(spec)
            reg.record_experiment(
                ExperimentRecord(
                    fingerprint=spec.fingerprint(),
                    strategy_name=spec.name,
                    symbols=("SPY",),
                    timeframe="1D",
                    data_start="",
                    data_end="",
                    dataset_version="test",
                )
            )
            assert reg.has_tried(spec.fingerprint())

    def test_memory_includes_failures(self, tmp_path, spec) -> None:
        """The failures are the part that stops the next hypothesis repeating."""
        with Registry(tmp_path / "r.sqlite") as reg:
            reg.upsert_strategy(spec)
            reg.record_experiment(
                ExperimentRecord(
                    fingerprint=spec.fingerprint(),
                    strategy_name="broken",
                    symbols=("SPY",),
                    timeframe="1D",
                    data_start="",
                    data_end="",
                    dataset_version="test",
                    error="never fires",
                )
            )
            memory = reg.memory()
            assert memory[0]["verdict"] == "ERROR"
            assert memory[0]["error"] == "never fires"
