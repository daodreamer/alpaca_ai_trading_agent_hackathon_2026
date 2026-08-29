"""Residual alpha: the number that decides whether a rule added anything.

The registry holds fourteen strategies promoted to PAPER and seventy-four
experiments carrying a benchmark. Not one has a positive excess Sharpe. The
scorer's first component was raw out-of-sample Sharpe, so on a universe that
drifted up for sixteen years the search was rewarded for holding beta and the
benchmark line was written into ``reasons`` where it changed nothing.

A plain Sharpe difference is not the fix, because it is unfair in the other
direction: a rule that is in the market a fifth of the time has a beta near 0.2,
and comparing its Sharpe to a fully-invested benchmark charges it for exposure
it never took. The honest question is a regression --

    r_strategy = alpha + beta * r_benchmark + residual

-- where ``alpha`` is what the rule added, ``beta`` is the market exposure it
happened to run, and ``t(alpha)`` says whether the alpha is distinguishable from
zero at all.

The end-to-end run that motivated this file: a 12-1 momentum book returned 27.9%
a year against the benchmark's 26.9% and would have been recorded as beating the
market by a point. Its beta was 1.13, its alpha ``-1.61%`` a year, and its
drawdown 7.5 points deeper. The extra return was rented, not earned.
"""

from __future__ import annotations

import numpy as np
import pytest

from aqr.backtest.alpha import ResidualAlpha, residual_alpha

RNG = np.random.default_rng(20260828)
N = 2520  # ten years of daily bars


def _market(n: int = N, vol: float = 0.011, drift: float = 0.0004) -> np.ndarray:
    """A market whose *realised* mean is exactly ``drift``.

    A fixture that relies on a sample mean landing near its expectation is a
    fixture that fails one run in twenty for reasons having nothing to do with
    the code under test. Over 2520 draws at 1.1% daily vol the standard error of
    the mean is half the drift itself, so this matters here.
    """
    raw = RNG.normal(0.0, vol, n)
    return raw - raw.mean() + drift


# --------------------------------------------------------------------------
# The regression


def test_pure_beta_has_no_alpha() -> None:
    """A leveraged copy of the benchmark added nothing, however good it looks.

    This is the case the old scorer got wrong: 1.13x the market returns more than
    the market, and a raw Sharpe comparison reads that as skill.
    """
    market = _market()
    result = residual_alpha(1.13 * market, market)
    assert result.beta == pytest.approx(1.13, abs=0.01)
    assert result.alpha == pytest.approx(0.0, abs=0.005)
    assert abs(result.t_alpha) < 2.0


def test_a_constant_edge_shows_up_as_alpha() -> None:
    market = _market()
    edge = 0.0002  # 2bp a day, about 5% a year
    result = residual_alpha(market + edge, market)
    assert result.beta == pytest.approx(1.0, abs=0.01)
    assert result.alpha == pytest.approx(edge * 252, rel=0.05)
    assert result.t_alpha > 5.0


def test_a_low_exposure_rule_is_not_charged_for_exposure_it_never_took() -> None:
    """Half the market plus a real edge, and a *lower* absolute return.

    Judged on level it lost. Judged on what it added, it did not: it took half
    the exposure and beat what that exposure was worth.
    """
    market = _market()
    # Built off the realised mean rather than the population one: a fixture that
    # depends on a sample mean landing near its expectation is a flaky test.
    noise = RNG.normal(0.0, 0.002, N)
    strategy = 0.5 * market + 0.4 * market.mean() + (noise - noise.mean())
    result = residual_alpha(strategy, market)
    assert result is not None
    assert result.beta == pytest.approx(0.5, abs=0.02)
    assert strategy.mean() < market.mean(), "the naive read is that this lost"
    assert result.alpha > 0
    assert result.t_alpha > 3.0


def test_a_higher_return_can_still_be_negative_alpha() -> None:
    """The case the old scorer got wrong, and the one the registry is full of.

    A 12-1 momentum book returned 27.9% a year against the benchmark's 26.9%
    and would have been recorded as beating the market by a point. Its beta was
    1.13. The extra return was rented.
    """
    market = _market()
    noise = RNG.normal(0.0, 0.001, N)
    strategy = 1.13 * market - 0.02 * market.mean() + (noise - noise.mean())
    result = residual_alpha(strategy, market)
    assert result is not None
    assert strategy.mean() > market.mean(), "it out-returned the benchmark"
    assert result.beta > 1.0
    assert result.alpha < 0, "and still added nothing"
    assert result.is_significant is False


def test_negative_alpha_is_reported_as_negative() -> None:
    market = _market()
    result = residual_alpha(market - 0.0002, market)
    assert result.alpha < 0
    assert result.t_alpha < -5.0


def test_the_information_ratio_is_alpha_over_residual_risk() -> None:
    market = _market()
    strategy = market + RNG.normal(0.0002, 0.004, N)
    result = residual_alpha(strategy, market)
    expected = result.alpha / result.residual_vol
    assert result.information_ratio == pytest.approx(expected, rel=1e-9)


# --------------------------------------------------------------------------
# Degenerate inputs, which are the ones that produce confident nonsense


def test_too_few_observations_refuses_to_answer() -> None:
    """A beta from five days is a number, not an estimate. ``None`` beats a
    plausible-looking figure nobody can check."""
    market = _market(5)
    assert residual_alpha(market * 1.2, market) is None


def test_a_flat_benchmark_refuses_to_answer() -> None:
    """No variance in the regressor means beta is undefined. Returning zero would
    silently reclassify every return as alpha."""
    flat = np.zeros(N)
    assert residual_alpha(_market(), flat) is None


def test_mismatched_lengths_are_refused_rather_than_truncated() -> None:
    """Silently aligning two series of different length is how an off-by-one
    becomes an edge."""
    with pytest.raises(ValueError, match="length"):
        residual_alpha(_market(N), _market(N - 1))


def test_non_finite_observations_are_dropped_in_pairs() -> None:
    """Dropping a NaN from one series but not its partner shifts every later
    observation against the wrong day."""
    market = _market()
    strategy = market + 0.0002
    strategy[100] = np.nan
    market[500] = np.inf
    result = residual_alpha(strategy, market)
    assert result is not None
    assert result.observations == N - 2


def test_the_result_is_json_safe() -> None:
    import json

    result = residual_alpha(_market() * 1.1, _market())
    assert result is not None
    json.dumps(result.as_dict())


def test_the_computation_is_deterministic() -> None:
    market = _market()
    strategy = market * 0.9 + 0.0001
    a = residual_alpha(strategy, market)
    b = residual_alpha(strategy, market)
    assert a == b


def test_alpha_is_annualised_by_the_periods_it_was_measured_in() -> None:
    """Daily by default. A weekly series annualised at 252 overstates by five."""
    market = _market()
    daily = residual_alpha(market + 0.0002, market)
    weekly = residual_alpha(market + 0.0002, market, periods_per_year=52.0)
    assert daily is not None and weekly is not None
    assert daily.alpha == pytest.approx(weekly.alpha * 252 / 52, rel=1e-6)


def test_a_result_knows_whether_it_is_evidence() -> None:
    """``t_alpha`` alone invites eyeballing. The threshold is written down once."""
    market = _market()
    assert residual_alpha(market + 0.0003, market).is_significant is True  # type: ignore[union-attr]
    assert residual_alpha(1.13 * market, market).is_significant is False  # type: ignore[union-attr]


def test_the_type_carries_what_the_record_needs() -> None:
    result = residual_alpha(_market() * 1.1, _market())
    assert isinstance(result, ResidualAlpha)
    for field in ("alpha", "beta", "t_alpha", "information_ratio", "excess_sharpe"):
        assert field in result.as_dict()
