"""The evaluator, once residual alpha is what it scores.

Two changes with one motivation. The scorer's heaviest component was raw
out-of-sample Sharpe and the benchmark comparison was written into ``reasons``
where it changed nothing, so on a universe that drifted upward for sixteen years
the search was rewarded for renting beta. It obliged: 177 short and 37
market-neutral proposals produced no promotions at all, while 69 long-only ones
produced all fourteen.

So the ``oos_sharpe`` component becomes the information ratio of the part beta
does not explain, and ``t(alpha) > 0`` becomes fatal. The consequence is
intended and worth stating plainly: every strategy currently at PAPER fails the
new gate, because not one of the seventy-four benchmarked experiments has a
positive excess return once exposure is accounted for.
"""

from __future__ import annotations

import pytest

from aqr.backtest.alpha import ResidualAlpha
from aqr.backtest.metrics import Metrics
from aqr.evaluator.score import WEIGHTS, EvaluationInput, evaluate_strategy


def _metrics(**over: float) -> Metrics:
    base: dict[str, float] = {
        "total_return": 0.5,
        "cagr": 0.12,
        "sharpe": 0.9,
        "sortino": 1.2,
        "max_drawdown": 0.15,
        "calmar": 0.8,
        "win_rate": 0.55,
        "profit_factor": 1.6,
        "expectancy": 0.004,
        "average_win": 0.02,
        "average_loss": -0.012,
        "num_trades": 200,
        "average_holding_bars": 12.0,
        "exposure": 0.95,
        "turnover": 3.0,
        "total_fees": 900.0,
        "total_slippage": 400.0,
        "best_trade": 0.09,
        "worst_trade": -0.05,
    }
    base.update(over)
    return Metrics(**{k: v for k, v in base.items()})  # type: ignore[arg-type]


def _alpha(**over: float) -> ResidualAlpha:
    base: dict[str, float] = {
        "alpha": 0.03,
        "beta": 1.0,
        "t_alpha": 3.0,
        "information_ratio": 0.6,
        "residual_vol": 0.05,
        "excess_sharpe": 0.1,
        "r_squared": 0.8,
        "observations": 2000,
    }
    base.update(over)
    return ResidualAlpha(**base)  # type: ignore[arg-type]


def _input(**over: object) -> EvaluationInput:
    base: dict[str, object] = {
        "strategy": "candidate",
        "fingerprint": "abc123",
        "oos": _metrics(),
        "oos_sharpe": 0.9,
        "positive_fold_rate": 0.7,
        "parameter_stability": 0.8,
        "asset_robustness": 0.7,
        "regime_robustness": 0.7,
        "residual": _alpha(),
    }
    base.update(over)
    return EvaluationInput(**base)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# The gate


def test_negative_alpha_is_fatal_however_good_the_rest_looks() -> None:
    """The registry's central finding, turned into a rule.

    A strategy can post a strong Sharpe, a clean drawdown and stable parameters
    and still have added nothing -- the fourteen at PAPER all did.
    """
    result = evaluate_strategy(_input(residual=_alpha(alpha=-0.02, t_alpha=-1.1)))
    assert result.verdict == "REJECT"
    assert any("added nothing" in r or "alpha" in r for r in result.reasons)


def test_a_high_return_high_beta_book_is_rejected() -> None:
    """27.9% a year against the benchmark's 26.9%, at beta 1.13. Rented, not
    earned, and the old scorer would have promoted it."""
    result = evaluate_strategy(
        _input(
            oos_sharpe=1.08,
            residual=_alpha(alpha=-0.016, beta=1.13, t_alpha=-0.60, information_ratio=-0.16),
        )
    )
    assert result.verdict == "REJECT"


def test_positive_but_insignificant_alpha_is_not_promoted() -> None:
    """Two years of daily bars cannot resolve an information ratio below about
    1.4. An alpha that cannot be distinguished from zero is not evidence, and a
    verdict that treats it as evidence is how a sealed window gets spent."""
    result = evaluate_strategy(_input(residual=_alpha(alpha=0.01, t_alpha=0.8)))
    assert result.verdict in ("REVIEW", "REJECT")
    assert result.verdict != "PAPER"


def test_significant_alpha_survives_the_gate() -> None:
    result = evaluate_strategy(_input(residual=_alpha(alpha=0.06, t_alpha=3.5)))
    assert result.verdict in ("ACCEPT", "PAPER")


def test_a_missing_regression_is_not_a_pass() -> None:
    """No benchmark means the question was never asked. Scoring it as though the
    answer were yes is exactly the failure this replaces."""
    result = evaluate_strategy(_input(residual=None))
    assert result.verdict not in ("ACCEPT", "PAPER")
    assert any("benchmark" in r or "alpha" in r for r in result.reasons)


# --------------------------------------------------------------------------
# The score


def test_the_heaviest_component_is_the_information_ratio() -> None:
    assert "information_ratio" in WEIGHTS
    assert "oos_sharpe" not in WEIGHTS
    assert WEIGHTS["information_ratio"] == max(WEIGHTS.values())
    assert sum(WEIGHTS.values()) == pytest.approx(1.0)


def test_raw_sharpe_no_longer_moves_the_score() -> None:
    """Two strategies identical except for how much beta they ran must score the
    same. Under the old weights the leveraged one scored higher for it."""
    modest = evaluate_strategy(_input(oos_sharpe=0.6))
    levered = evaluate_strategy(_input(oos_sharpe=1.4))
    assert modest.score == pytest.approx(levered.score)


def test_a_better_information_ratio_scores_higher() -> None:
    weak = evaluate_strategy(_input(residual=_alpha(information_ratio=0.2, t_alpha=2.5)))
    strong = evaluate_strategy(_input(residual=_alpha(information_ratio=0.9, t_alpha=4.0)))
    assert strong.score > weak.score


def test_the_evaluation_records_the_decomposition() -> None:
    result = evaluate_strategy(_input())
    body = result.as_dict()
    assert body["alpha"] == pytest.approx(0.03)
    assert body["beta"] == pytest.approx(1.0)
    assert body["t_alpha"] == pytest.approx(3.0)
    assert body["information_ratio"] == pytest.approx(0.6)


def test_the_old_gates_still_bite() -> None:
    """Residual alpha is an addition, not an amnesty. Too few trades is still
    fatal however good the regression looks."""
    result = evaluate_strategy(
        _input(oos=_metrics(num_trades=11), residual=_alpha(t_alpha=6.0, information_ratio=1.5))
    )
    assert result.verdict == "REJECT"
    assert any("trades" in r for r in result.reasons)
