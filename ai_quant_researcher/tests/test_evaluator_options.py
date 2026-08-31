"""The options evaluator profile — specs/10-options-research.md D8.

Two things are checked here that matter more than the score arithmetic, which
``evaluate_option_strategy`` shares with ``evaluate_strategy`` almost verbatim
(``tests/test_scoring.py`` already pins that shape):

**The gate reads cycles, not trades.** Thirty overlapping trades that
collapse to a handful of independent cycles must be rejected exactly the way
a 71-cycle, 400-trade run with genuinely independent evidence must not be.

**The equity path is untouched.** ``evaluate_strategy``, ``EvaluationInput``
and ``WEIGHTS`` are imported and exercised here unmodified, so a regression in
either profile's wiring would show up as a failure in the *other* profile's
assertions too -- which is the point: nothing about adding the options
profile may perturb the equity one, and this file is one more thing that would
notice if it did.
"""

from __future__ import annotations

import pytest

from aqr.backtest.alpha import ResidualAlpha
from aqr.backtest.engine import Trade
from aqr.backtest.metrics import Metrics
from aqr.evaluator.score import (
    AFFORDABILITY_BOUND_THRESHOLD,
    MIN_INDEPENDENT_CYCLES,
    OPTION_WEIGHTS,
    WEIGHTS,
    EvaluationInput,
    OptionEvaluationInput,
    evaluate_option_strategy,
    evaluate_strategy,
)
from aqr.validation.cycles import independent_cycles

DAY = 86_400


def _trade(entry_day: int, exit_day: int, *, pnl: float = 500.0) -> Trade:
    return Trade(
        symbol="SPY",
        direction="short",
        entry_time=entry_day * DAY,
        entry_price=900.0,
        exit_time=exit_day * DAY,
        exit_price=800.0,
        quantity=100.0,
        gross_pnl=pnl,
        fees=1.0,
        slippage=0.5,
        bars_held=exit_day - entry_day,
        exit_reason="expiry",
        mae=0.0,
        mfe=0.0,
    )


def _residual(**overrides: object) -> ResidualAlpha:
    defaults = dict(
        alpha=0.05,
        beta=0.9,
        t_alpha=3.2,
        information_ratio=0.6,
        residual_vol=0.08,
        excess_sharpe=0.1,
        r_squared=0.7,
        observations=1500,
    )
    defaults.update(overrides)
    return ResidualAlpha(**defaults)  # type: ignore[arg-type]


def _metrics(**overrides: object) -> Metrics:
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
        num_trades=60,
        average_holding_bars=28.0,
        exposure=0.4,
        turnover=3.0,
        total_fees=500.0,
        total_slippage=300.0,
        best_trade=2000.0,
        worst_trade=-900.0,
    )
    defaults.update(overrides)
    return Metrics(**defaults)  # type: ignore[arg-type]


def _independent_trades(n: int) -> list[Trade]:
    """``n`` structures, each entered only after the last one closed -- every
    one an independent cycle by construction."""
    return [_trade(i * 28, i * 28 + 28) for i in range(n)]


def _option_input(**overrides: object) -> OptionEvaluationInput:
    oos_trades = overrides.pop("oos_trades", _independent_trades(40))
    defaults: dict[str, object] = dict(
        strategy="candidate",
        fingerprint="abc123",
        oos=_metrics(),
        oos_sharpe=1.2,
        oos_trades=oos_trades,
        positive_fold_rate=0.7,
        parameter_stability=0.8,
        year_dte_robustness=0.75,
        regime_robustness=0.6,
        affordability_bound_fraction=0.0,
        residual=_residual(),
    )
    defaults.update(overrides)
    return OptionEvaluationInput(**defaults)  # type: ignore[arg-type]


class TestOptionWeights:
    def test_option_weights_sum_to_one(self) -> None:
        assert sum(OPTION_WEIGHTS.values()) == pytest.approx(1.0)

    def test_option_weights_swap_asset_for_year_dte_robustness_at_the_same_weight(self) -> None:
        assert "asset_robustness" not in OPTION_WEIGHTS
        assert "year_dte_robustness" in OPTION_WEIGHTS
        assert OPTION_WEIGHTS["year_dte_robustness"] == WEIGHTS["asset_robustness"]
        # Every other weight is untouched.
        shared = set(WEIGHTS) & set(OPTION_WEIGHTS)
        assert shared == {
            "information_ratio",
            "profit_factor",
            "max_drawdown",
            "parameter_stability",
            "regime_robustness",
        }
        for key in shared:
            assert OPTION_WEIGHTS[key] == WEIGHTS[key]


class TestIndependentCycleGate:
    def test_a_run_with_enough_independent_cycles_is_not_rejected_for_sample_size(self) -> None:
        trades = _independent_trades(MIN_INDEPENDENT_CYCLES)
        result = evaluate_option_strategy(_option_input(oos_trades=trades))
        assert not any("independent" in r and "below" in r for r in result.reasons)

    def test_too_few_independent_cycles_is_fatal_even_with_many_trades(self) -> None:
        """400 trades, one session apart, 28 DTE each: heavily overlapping,
        so the independent-cycle count is far below the trade count -- and it
        is the cycle count the gate must read."""
        overlapping = [_trade(i, i + 28) for i in range(400)]
        cycles = independent_cycles(overlapping)
        assert cycles < MIN_INDEPENDENT_CYCLES < 400  # confirms the fixture proves the point

        result = evaluate_option_strategy(
            _option_input(
                oos=_metrics(num_trades=400),
                oos_trades=overlapping,
                oos_sharpe=3.0,
                parameter_stability=1.0,
                year_dte_robustness=1.0,
            )
        )
        assert result.verdict == "REJECT"
        assert any(f"below the {MIN_INDEPENDENT_CYCLES} minimum" in r for r in result.reasons)

    def test_a_report_states_the_trade_count_next_to_the_cycle_count(self) -> None:
        trades = _independent_trades(30)
        result = evaluate_option_strategy(_option_input(oos_trades=trades))
        assert any(
            "30 out-of-sample trades" in r and "30 independent cycles" in r
            for r in result.reasons
        )

    def test_the_gate_and_the_report_never_disagree_on_what_a_cycle_is(self) -> None:
        """Same trade list, same number, computed by the same function -- the
        whole point of D8's "one function" instruction."""
        trades = [_trade(i * 5, i * 5 + 28) for i in range(60)]
        result = evaluate_option_strategy(_option_input(oos_trades=trades))
        expected = independent_cycles(trades)
        assert any(f"{expected} independent cycles" in r for r in result.reasons)


class TestAffordabilityBoundD8a:
    """specs/10 D8a: a low cycle count decided by the account rather than the
    market must never be presented as an ordinary verdict on the rule.

    ``_option_input``'s own default (``affordability_bound_fraction=0.0``)
    already exercises the "not bound" path in every test above this class --
    what is new here is the "bound" path, and the requirement that the
    fraction is visible either way.
    """

    def test_the_fraction_is_always_reported_regardless_of_the_gate(self) -> None:
        """D8a: "a reader can never see a cycle count without the fraction
        beside it" -- checked here on a *passing* run, where nothing else
        would force the fraction into the reasons list at all."""
        trades = _independent_trades(40)
        result = evaluate_option_strategy(
            _option_input(oos_trades=trades, affordability_bound_fraction=0.37)
        )
        assert result.verdict != "REJECT"  # nothing else about this fixture is fatal
        assert any("37%" in r and "affordability-bound" in r for r in result.reasons)

    def test_a_low_cycle_count_below_the_threshold_reads_as_an_ordinary_rejection(self) -> None:
        """37% of skips being unaffordable does not clear
        ``AFFORDABILITY_BOUND_THRESHOLD`` (50%), so the combined "this REJECT
        is a verdict on the account" sentence must not appear -- most of the
        shortfall here really was the rule or the ladder, not the account."""
        overlapping = [_trade(i, i + 28) for i in range(400)]
        assert independent_cycles(overlapping) < MIN_INDEPENDENT_CYCLES
        result = evaluate_option_strategy(
            _option_input(
                oos=_metrics(num_trades=400),
                oos_trades=overlapping,
                oos_sharpe=3.0,
                affordability_bound_fraction=0.37,
            )
        )
        assert result.verdict == "REJECT"
        assert not any("verdict on the account" in r for r in result.reasons)
        assert any("37%" in r for r in result.reasons)  # still reported, per the test above

    def test_a_low_cycle_count_above_the_threshold_names_the_account_not_the_rule(self) -> None:
        """specs/10 D8a's own worked example: 94.9% of skips at
        ``risk_per_trade=0.01`` were affordability, not the rule declining to
        trade. The gate still fires -- REJECT, not loosened -- but the
        reason has to say the run measured the account."""
        overlapping = [_trade(i, i + 28) for i in range(400)]
        result = evaluate_option_strategy(
            _option_input(
                oos=_metrics(num_trades=400),
                oos_trades=overlapping,
                oos_sharpe=3.0,
                affordability_bound_fraction=0.949,
            )
        )
        assert result.verdict == "REJECT"  # D8a: the >= 25 gate is not loosened
        assert any(
            "verdict on the account" in r and f"below the {MIN_INDEPENDENT_CYCLES} minimum" in r
            for r in result.reasons
        )

    def test_the_gate_is_never_loosened_by_a_high_affordability_fraction(self) -> None:
        """The one thing D8a explicitly forbids: turning REJECT into PAPER or
        ACCEPT because the shortfall was explained away. A verdict on an
        account-bound sample is not evidence either way."""
        overlapping = [_trade(i, i + 28) for i in range(400)]
        result = evaluate_option_strategy(
            _option_input(
                oos=_metrics(num_trades=400),
                oos_trades=overlapping,
                oos_sharpe=3.0,
                parameter_stability=1.0,
                year_dte_robustness=1.0,
                regime_robustness=1.0,
                affordability_bound_fraction=1.0,
            )
        )
        assert result.verdict == "REJECT"

    def test_the_evaluation_carries_the_fraction_and_the_derived_flag(self) -> None:
        result = evaluate_option_strategy(_option_input(affordability_bound_fraction=0.949))
        assert result.affordability_bound_fraction == pytest.approx(0.949)
        assert result.is_affordability_bound is True

    def test_the_derived_flag_respects_the_threshold_in_both_directions(self) -> None:
        below = evaluate_option_strategy(_option_input(affordability_bound_fraction=0.1))
        above = evaluate_option_strategy(
            _option_input(affordability_bound_fraction=AFFORDABILITY_BOUND_THRESHOLD)
        )
        assert below.is_affordability_bound is False
        assert above.is_affordability_bound is True  # the boundary itself counts as bound

    def test_the_equity_path_never_reports_an_affordability_fraction(self) -> None:
        """``evaluate_strategy`` has no notion of D8a -- the equity engine
        never refuses an entry for being too small to cover one contract's
        defined-risk loss the way a spread does -- so both the fraction and
        the derived flag must read as "not applicable," not "measured at
        zero." Zero would be a claim; this rule was never asked."""
        data = EvaluationInput(
            strategy="candidate",
            fingerprint="abc123",
            oos=_metrics(),
            oos_sharpe=1.2,
            positive_fold_rate=0.7,
            parameter_stability=0.8,
            asset_robustness=0.75,
            regime_robustness=0.6,
            residual=_residual(),
        )
        result = evaluate_strategy(data)
        assert result.affordability_bound_fraction is None
        assert result.is_affordability_bound is None


class TestOptionEvaluatorShapeMatchesEquity:
    def test_a_solid_candidate_is_not_rejected(self) -> None:
        result = evaluate_option_strategy(_option_input())
        assert result.verdict in ("ACCEPT", "PAPER")
        assert result.score > 50

    def test_an_excessive_drawdown_is_fatal(self) -> None:
        result = evaluate_option_strategy(_option_input(oos=_metrics(max_drawdown=0.45)))
        assert result.verdict == "REJECT"

    def test_a_negative_oos_sharpe_is_fatal(self) -> None:
        assert evaluate_option_strategy(_option_input(oos_sharpe=-0.1)).verdict == "REJECT"

    def test_no_benchmark_regression_is_fatal(self) -> None:
        assert evaluate_option_strategy(_option_input(residual=None)).verdict == "REJECT"

    def test_non_positive_alpha_is_fatal(self) -> None:
        result = evaluate_option_strategy(
            _option_input(residual=_residual(alpha=-0.01, t_alpha=1.0))
        )
        assert result.verdict == "REJECT"

    def test_score_is_bounded(self) -> None:
        best = evaluate_option_strategy(
            _option_input(oos_sharpe=99.0, parameter_stability=5.0, year_dte_robustness=5.0)
        )
        assert 0.0 <= best.score <= 100.0


class TestEquityPathIsUnperturbed:
    """These exercise ``evaluate_strategy`` directly, mirroring
    ``tests/test_scoring.py``'s own baseline fixture, so a change that broke
    the equity path while wiring in the options one would fail here too."""

    def test_equity_weights_still_sum_to_one(self) -> None:
        assert sum(WEIGHTS.values()) == pytest.approx(1.0)

    def test_equity_evaluator_still_gates_on_trade_count(self) -> None:
        from aqr.backtest.alpha import ResidualAlpha
        from aqr.backtest.metrics import Metrics

        data = EvaluationInput(
            strategy="candidate",
            fingerprint="abc123",
            oos=Metrics(
                total_return=0.4, cagr=0.12, sharpe=1.2, sortino=1.6, max_drawdown=0.12,
                calmar=1.0, win_rate=0.5, profit_factor=1.6, expectancy=100.0,
                average_win=400.0, average_loss=-250.0, num_trades=11,
                average_holding_bars=9.0, exposure=0.4, turnover=3.0, total_fees=500.0,
                total_slippage=300.0, best_trade=2000.0, worst_trade=-900.0,
            ),
            oos_sharpe=3.0,
            positive_fold_rate=0.7,
            parameter_stability=1.0,
            asset_robustness=0.75,
            regime_robustness=0.6,
            residual=ResidualAlpha(
                alpha=0.05, beta=0.9, t_alpha=3.2, information_ratio=0.6,
                residual_vol=0.08, excess_sharpe=0.1, r_squared=0.7, observations=1500,
            ),
        )
        result = evaluate_strategy(data)
        assert result.verdict == "REJECT"
        assert any("below the 30 minimum" in r for r in result.reasons)
