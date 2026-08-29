"""The benchmark, and the gate it became.

Fourteen strategies reached PAPER. All fourteen lost to equal-weight
buy-and-hold on the symbols they were tested on -- Sharpe 1.16 against a best of
0.65 -- and nothing in the pipeline had asked, so nothing in the record said so.

When that was first fixed the comparison was recorded and deliberately *not*
gated, on the reasoning that the losers are the control group. That reasoning
was right about the registry and wrong about the verdict, and this file now
draws the line where it belongs:

* **Recording is unchanged.** Every experiment still carries its benchmark and
  its excess, winners and losers alike. A research log that keeps only winners
  is the log that made the winners look inevitable, and rejection is not
  deletion -- a REJECT verdict is still written to the registry in full.
* **Promotion is gated.** A rule that added nothing to owning the universe is
  not a candidate, whatever its Sharpe. The gate is the residual regression
  rather than the Sharpe difference, because a Sharpe difference charges a
  low-exposure rule for exposure it never took.

`excess_sharpe` survives as a reported number. It is no longer the number that
decides.
"""

from __future__ import annotations

import numpy as np
import pytest

from aqr.backtest.alpha import ResidualAlpha
from aqr.backtest.metrics import Metrics
from aqr.evaluator.score import EvaluationInput, evaluate_strategy


def _metrics(**overrides: float) -> Metrics:
    base = dict(
        total_return=0.5,
        cagr=0.08,
        sharpe=0.8,
        sortino=1.0,
        max_drawdown=0.15,
        calmar=0.5,
        win_rate=0.55,
        profit_factor=1.4,
        expectancy=50.0,
        average_win=200.0,
        average_loss=-140.0,
        num_trades=200,
        average_holding_bars=8.0,
        exposure=0.3,
        turnover=5.0,
        total_fees=100.0,
        total_slippage=50.0,
        best_trade=900.0,
        worst_trade=-600.0,
    )
    base.update(overrides)
    return Metrics(**base)  # type: ignore[arg-type]


def _input(**overrides: object) -> EvaluationInput:
    base: dict[str, object] = dict(
        strategy="probe_v1",
        fingerprint="deadbeefdeadbeef",
        oos=_metrics(),
        oos_sharpe=0.8,
        positive_fold_rate=0.7,
        parameter_stability=0.7,
        asset_robustness=0.6,
        regime_robustness=0.6,
        overfitting=None,
        monte_carlo_p95_drawdown=0.2,
        zero_cost_sharpe=1.0,
    )
    base.setdefault("residual", _residual())
    base.update(overrides)
    return EvaluationInput(**base)  # type: ignore[arg-type]


def _residual(**overrides: float) -> ResidualAlpha:
    base: dict[str, float] = dict(
        alpha=0.04,
        beta=0.9,
        t_alpha=3.0,
        information_ratio=0.55,
        residual_vol=0.07,
        excess_sharpe=0.05,
        r_squared=0.7,
        observations=1500,
    )
    base.update(overrides)
    return ResidualAlpha(**base)  # type: ignore[arg-type]


class TestExcessSharpe:
    def test_it_is_the_strategy_less_the_benchmark(self) -> None:
        evaluation = evaluate_strategy(_input(oos_sharpe=0.8, benchmark_sharpe=1.16))
        assert evaluation.excess_sharpe == pytest.approx(0.8 - 1.16)

    def test_without_a_benchmark_it_is_unknown_not_zero(self) -> None:
        """Zero would read as "exactly matched the market", which is a finding.
        Not having looked is not a finding."""
        assert evaluate_strategy(_input()).excess_sharpe is None

    def test_beating_the_market_is_recorded(self) -> None:
        evaluation = evaluate_strategy(_input(oos_sharpe=1.4, benchmark_sharpe=1.16))
        assert evaluation.excess_sharpe is not None
        assert evaluation.excess_sharpe > 0
        assert evaluation.beats_benchmark is True

    def test_losing_to_the_market_is_recorded(self) -> None:
        evaluation = evaluate_strategy(_input(oos_sharpe=0.65, benchmark_sharpe=1.16))
        assert evaluation.beats_benchmark is False


class TestItIsAGate:
    def test_adding_nothing_is_fatal(self) -> None:
        """The reversal. A rule whose return is the exposure it ran is not a
        candidate, however respectable its Sharpe looks beside the index."""
        beaten = evaluate_strategy(
            _input(
                oos_sharpe=0.8,
                benchmark_sharpe=1.16,
                residual=_residual(alpha=-0.01, t_alpha=-0.7),
            )
        )
        assert beaten.verdict == "REJECT"

    def test_a_rejected_strategy_is_still_a_full_record(self) -> None:
        """Rejection is not deletion. The control group survives in the
        registry; what it no longer does is get promoted."""
        beaten = evaluate_strategy(
            _input(benchmark_sharpe=1.16, residual=_residual(alpha=-0.01, t_alpha=-0.7))
        )
        body = beaten.as_dict()
        assert body["benchmark_sharpe"] == pytest.approx(1.16)
        assert body["excess_sharpe"] is not None
        assert body["alpha"] is not None and body["beta"] is not None

    def test_the_sharpe_difference_no_longer_decides(self) -> None:
        """A low-exposure rule with real alpha survives even though its absolute
        Sharpe trails the index. Under a Sharpe-difference gate it would not."""
        low_beta = evaluate_strategy(
            _input(
                oos_sharpe=0.7,
                benchmark_sharpe=1.16,
                residual=_residual(alpha=0.04, beta=0.35, t_alpha=3.4),
            )
        )
        assert low_beta.verdict != "REJECT"

    def test_a_fatal_flaw_still_rejects(self) -> None:
        # The existing gates are untouched.
        rejected = evaluate_strategy(
            _input(oos=_metrics(num_trades=11), benchmark_sharpe=0.0)
        )
        assert rejected.verdict == "REJECT"


class TestItIsVisible:
    def test_the_comparison_appears_in_the_reasons(self) -> None:
        evaluation = evaluate_strategy(_input(oos_sharpe=0.65, benchmark_sharpe=1.16))
        joined = " ".join(evaluation.reasons).lower()
        assert "buy and hold" in joined or "benchmark" in joined

    def test_the_reason_says_which_way_it_went(self) -> None:
        lost = " ".join(evaluate_strategy(_input(oos_sharpe=0.3, benchmark_sharpe=1.16)).reasons)
        won = " ".join(evaluate_strategy(_input(oos_sharpe=1.9, benchmark_sharpe=1.16)).reasons)
        assert lost != won

    def test_it_serialises(self) -> None:
        payload = evaluate_strategy(_input(benchmark_sharpe=1.16)).as_dict()
        assert payload["benchmark_sharpe"] == pytest.approx(1.16)
        assert payload["excess_sharpe"] is not None


class TestLongOnlyBias:
    def test_a_long_only_rule_in_a_bull_market_is_flagged_not_praised(self) -> None:
        """The finding this exists for: on NASDAQ names over 2010-2026 a
        long-only rule makes money almost whatever the rule is, and a positive
        Sharpe on its own reads as evidence about the rule."""
        evaluation = evaluate_strategy(_input(oos_sharpe=0.65, benchmark_sharpe=1.16))
        assert evaluation.beats_benchmark is False
        assert any("hold" in r.lower() or "benchmark" in r.lower() for r in evaluation.reasons)


def test_metrics_fixture_is_sane() -> None:
    assert np.isfinite(_metrics().sharpe)
