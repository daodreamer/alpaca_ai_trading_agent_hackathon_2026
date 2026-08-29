"""Return decomposition: what a rule added, separated from what it rented.

The evaluator's first component used to be raw out-of-sample Sharpe. On a
universe that drifted upward for sixteen years that rewarded holding beta, and
the search obliged: every strategy this project promoted was long-only, and none
of the seventy-four experiments carrying a benchmark had a positive excess
Sharpe.

Subtracting the benchmark's Sharpe is not the fix. It is unfair the other way --
a rule invested a fifth of the time runs a beta near 0.2, and charging it
against a fully-invested benchmark bills it for exposure it never took. Both
distortions come from comparing summary statistics instead of returns, so this
module regresses one on the other:

    r_strategy = alpha + beta * r_benchmark + residual

``beta`` is the exposure the rule happened to run. ``alpha`` is what is left,
which is the only part the rule can claim. ``t_alpha`` is whether that
remainder is distinguishable from zero -- and over the two-year sealed window it
usually will not be, which is a fact about the sample size rather than about the
rule, and is why the sealed run can refute a strategy but never confirm one.

Everything here refuses rather than guesses. Too few observations, a benchmark
with no variance, mismatched lengths: each returns ``None`` or raises. A beta
estimated from five days is a number and not an estimate, and a plausible-looking
figure nobody can check is worse than an absent one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["MIN_OBSERVATIONS", "SIGNIFICANCE_T", "ResidualAlpha", "residual_alpha"]

MIN_OBSERVATIONS = 60
"""Below this the regression is not reported at all."""

SIGNIFICANCE_T = 2.0
"""Written down once, so nobody has to eyeball a t-statistic in a review."""

TRADING_DAYS = 252.0


@dataclass(frozen=True, slots=True)
class ResidualAlpha:
    """One regression of strategy returns on benchmark returns."""

    alpha: float
    """Annualised intercept: the return the rule added beyond its exposure."""
    beta: float
    """Market exposure the rule ran, whether or not it meant to."""
    t_alpha: float
    """Whether ``alpha`` is distinguishable from zero on this sample."""
    information_ratio: float
    """``alpha`` per unit of residual risk -- the ranking number."""
    residual_vol: float
    """Annualised standard deviation of the part beta does not explain."""
    excess_sharpe: float
    """Naive Sharpe difference. Kept because it is what a reader expects to see,
    and because the cases where it disagrees with ``alpha`` are the interesting
    ones."""
    r_squared: float
    observations: int

    @property
    def is_significant(self) -> bool:
        return self.t_alpha > SIGNIFICANCE_T

    def as_dict(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "t_alpha": self.t_alpha,
            "information_ratio": self.information_ratio,
            "residual_vol": self.residual_vol,
            "excess_sharpe": self.excess_sharpe,
            "r_squared": self.r_squared,
            "observations": self.observations,
            "is_significant": self.is_significant,
        }


def _sharpe(returns: np.ndarray, periods_per_year: float) -> float:
    sd = float(returns.std(ddof=1)) if returns.size > 1 else 0.0
    if sd <= 0:
        return 0.0
    return float(returns.mean() / sd * np.sqrt(periods_per_year))


def residual_alpha(
    strategy: np.ndarray,
    benchmark: np.ndarray,
    *,
    periods_per_year: float = TRADING_DAYS,
) -> ResidualAlpha | None:
    """Regress ``strategy`` on ``benchmark``. Both are period returns, not levels.

    Returns ``None`` when the sample cannot support the estimate. Raises only
    when the inputs are incoherent -- two series of different length are an
    alignment bug, and truncating them silently is how an off-by-one becomes an
    edge.
    """
    strategy = np.asarray(strategy, dtype=np.float64)
    benchmark = np.asarray(benchmark, dtype=np.float64)
    if strategy.shape != benchmark.shape:
        raise ValueError(
            f"strategy and benchmark returns must have the same length, "
            f"got {strategy.size} and {benchmark.size}"
        )

    # Dropped in pairs. Removing a NaN from one series but not its partner
    # shifts every later observation against the wrong day.
    good = np.isfinite(strategy) & np.isfinite(benchmark)
    x, y = benchmark[good], strategy[good]
    n = int(x.size)
    if n < MIN_OBSERVATIONS:
        return None

    var_x = float(x.var(ddof=1))
    if var_x <= 0:
        # No variance in the regressor: beta is undefined. Returning zero would
        # silently reclassify every return as alpha.
        return None

    beta = float(np.cov(y, x, ddof=1)[0, 1] / var_x)
    intercept = float(y.mean() - beta * x.mean())
    residual = y - (intercept + beta * x)

    # Two parameters fitted, so n - 2 degrees of freedom.
    dof = n - 2
    resid_sd = float(np.sqrt(residual @ residual / dof)) if dof > 0 else 0.0

    # An exact linear relationship leaves a residual made entirely of
    # floating-point noise, and dividing by that produces a t-statistic out of
    # nothing at all -- the most confident kind of wrong. What separates the two
    # exact cases is the *intercept*, not the residual: a levered copy of the
    # benchmark has neither alpha nor a way to estimate one, while the benchmark
    # plus a constant has an alpha that is measured perfectly.
    scale = float(np.abs(y).mean())
    floor = 1e-9 * scale
    if scale > 0 and resid_sd < floor:
        if abs(intercept) < floor:
            return ResidualAlpha(
                alpha=0.0,
                beta=beta,
                t_alpha=0.0,
                information_ratio=0.0,
                residual_vol=0.0,
                excess_sharpe=_sharpe(y, periods_per_year) - _sharpe(x, periods_per_year),
                r_squared=1.0,
                observations=n,
            )
        # Perfectly measured, so the t-statistic is unbounded. Floored rather
        # than infinite: an infinity does not survive a JSON round-trip into the
        # experiment record, and every consumer would need a special case.
        resid_sd = floor
    se_intercept = (
        resid_sd * float(np.sqrt(1.0 / n + x.mean() ** 2 / (var_x * (n - 1))))
        if resid_sd > 0
        else 0.0
    )

    alpha = intercept * periods_per_year
    residual_vol = resid_sd * float(np.sqrt(periods_per_year))
    t_alpha = intercept / se_intercept if se_intercept > 0 else 0.0
    ir = alpha / residual_vol if residual_vol > 0 else 0.0
    var_y = float(y.var(ddof=1))
    r2 = 1.0 - float(residual.var(ddof=1)) / var_y if var_y > 0 else 0.0

    return ResidualAlpha(
        alpha=alpha,
        beta=beta,
        t_alpha=float(t_alpha),
        information_ratio=float(ir),
        residual_vol=residual_vol,
        excess_sharpe=_sharpe(y, periods_per_year) - _sharpe(x, periods_per_year),
        r_squared=r2,
        observations=n,
    )
