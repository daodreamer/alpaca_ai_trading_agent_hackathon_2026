"""Strategy scoring (architecture section 15).

    30%  information ratio of residual alpha
    20%  profit factor
    15%  max drawdown
    15%  parameter stability
    10%  cross-asset robustness
    10%  regime robustness

Three deliberate departures from a naive weighted sum:

*Out-of-sample only.* Every component is measured on data the strategy was not
fitted to. An in-sample Sharpe of 3 contributes nothing here, by design.

*Gates before weights.* A strategy with 11 trades, or one that only works
without transaction costs, does not get a mediocre score -- it gets rejected.
Averaging lets a fatal flaw be offset by an unrelated strength, and there is no
weight at which "too few trades to know" becomes acceptable.

*Residual, not absolute.* The heaviest component used to be raw out-of-sample
Sharpe, and the benchmark comparison was recorded in ``reasons`` where it
changed nothing. On a universe that drifted upward for sixteen years that
rewarded renting beta, and the search obliged: 177 short and 37 market-neutral
proposals produced no promotions between them, while 69 long-only ones produced
all fourteen. The component is now the information ratio of the part beta does
not explain, and a non-positive ``t(alpha)`` is fatal.

That change demotes every strategy currently at PAPER. It should: not one of the
seventy-four benchmarked experiments added anything once exposure was accounted
for, and the previous scorer said otherwise only because it never asked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from aqr.backtest.alpha import ResidualAlpha
from aqr.backtest.metrics import Metrics
from aqr.validation.overfitting import OverfittingReport

__all__ = ["Evaluation", "EvaluationInput", "Verdict", "evaluate_strategy"]

Verdict = Literal["ACCEPT", "PAPER", "REVIEW", "REJECT"]

WEIGHTS: dict[str, float] = {
    "information_ratio": 0.30,
    "profit_factor": 0.20,
    "max_drawdown": 0.15,
    "parameter_stability": 0.15,
    "asset_robustness": 0.10,
    "regime_robustness": 0.10,
}

MIN_TRADES = 30
MAX_ACCEPTABLE_DRAWDOWN = 0.35
MIN_POSITIVE_FOLD_RATE = 0.4

GOOD_INFORMATION_RATIO = 0.5
"""Where the information-ratio component saturates.

Deliberately modest. A sustained IR of 0.5 on liquid US equities after costs
would be a real result; setting the bar where a fitted backtest can reach it
would make the component a measure of how hard the search tried."""


@dataclass(slots=True)
class EvaluationInput:
    """Everything the evaluator is allowed to look at."""

    strategy: str
    fingerprint: str
    oos: Metrics
    """Out-of-sample metrics -- stitched walk-forward, or the untouched test split."""
    oos_sharpe: float
    """Kept for the record and for the cost comparison. No longer scored: it is
    mostly the market's Sharpe whenever the rule is long and the market rose."""
    positive_fold_rate: float
    parameter_stability: float
    asset_robustness: float
    regime_robustness: float
    residual: ResidualAlpha | None = None
    """The regression of strategy returns on the benchmark's.

    ``None`` means nobody asked, which is not the same as a pass and is not
    scored as one -- a strategy with no benchmark cannot be promoted."""
    overfitting: OverfittingReport | None = None
    monte_carlo_p95_drawdown: float | None = None
    zero_cost_sharpe: float | None = None
    benchmark_sharpe: float | None = None
    """Equal-weight buy-and-hold over the same symbols and window.

    Recorded, never a gate. A strategy that loses to holding the index is
    kept on purpose: without the ones that failed there is nothing to measure
    the ones that succeed against, and a research log that keeps only winners
    is the log that made the winners look inevitable."""


@dataclass(slots=True)
class Evaluation:
    strategy: str
    fingerprint: str
    score: float
    verdict: Verdict
    components: dict[str, float] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    residual: ResidualAlpha | None = None
    benchmark_sharpe: float | None = None
    excess_sharpe: float | None = None
    """Out-of-sample Sharpe less the benchmark's, or ``None`` if nobody
    looked. Not zero: zero would read as "exactly matched the market", which
    is a finding, and not having looked is not."""

    @property
    def beats_benchmark(self) -> bool | None:
        return None if self.excess_sharpe is None else bool(self.excess_sharpe > 0)

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "fingerprint": self.fingerprint,
            "score": self.score,
            "verdict": self.verdict,
            "components": self.components,
            "reasons": self.reasons,
            "benchmark_sharpe": self.benchmark_sharpe,
            "excess_sharpe": self.excess_sharpe,
            "beats_benchmark": self.beats_benchmark,
            "alpha": self.residual.alpha if self.residual else None,
            "beta": self.residual.beta if self.residual else None,
            "t_alpha": self.residual.t_alpha if self.residual else None,
            "information_ratio": (self.residual.information_ratio if self.residual else None),
            "residual": self.residual.as_dict() if self.residual else None,
        }

    def __str__(self) -> str:
        rows = [f"  {k:<20} {v:5.1f}/100" for k, v in sorted(self.components.items())]
        reasons = [f"  - {r}" for r in self.reasons]
        return "\n".join(
            [f"{self.strategy} [{self.fingerprint}]  score {self.score:.0f}/100  {self.verdict}"]
            + rows
            + (["  reasons:"] if reasons else [])
            + reasons
        )


def _curve(value: float, good: float, cap: float = 1.0) -> float:
    """Map a metric onto [0, 1], saturating at ``good``.

    Saturating matters: without it, one component with an extreme value can
    dominate the weighted sum, and extreme values in backtests are usually
    artefacts rather than achievements.
    """
    if not np.isfinite(value) or value <= 0:
        return 0.0
    return float(min(value / good, cap))


def evaluate_strategy(data: EvaluationInput) -> Evaluation:
    """Score a strategy and decide what happens to it next."""
    ir = data.residual.information_ratio if data.residual is not None else 0.0
    components = {
        "information_ratio": 100.0 * _curve(ir, good=GOOD_INFORMATION_RATIO),
        "profit_factor": 100.0 * _curve(data.oos.profit_factor - 1.0, good=0.8),
        "max_drawdown": 100.0 * _drawdown_component(data.oos.max_drawdown),
        "parameter_stability": 100.0 * _clip01(data.parameter_stability),
        "asset_robustness": 100.0 * _clip01(data.asset_robustness),
        "regime_robustness": 100.0 * _clip01(data.regime_robustness),
    }
    score = sum(components[k] * w for k, w in WEIGHTS.items())

    reasons: list[str] = []
    fatal = False

    if data.oos.num_trades < MIN_TRADES:
        fatal = True
        reasons.append(
            f"{data.oos.num_trades} out-of-sample trades is below the {MIN_TRADES} minimum; "
            "the result is not distinguishable from noise"
        )
    if data.oos.max_drawdown > MAX_ACCEPTABLE_DRAWDOWN:
        fatal = True
        reasons.append(
            f"out-of-sample drawdown {data.oos.max_drawdown:.0%} exceeds the "
            f"{MAX_ACCEPTABLE_DRAWDOWN:.0%} limit"
        )
    if data.oos_sharpe <= 0:
        fatal = True
        reasons.append(f"out-of-sample Sharpe {data.oos_sharpe:.2f} is not positive")
    if data.residual is None:
        fatal = True
        reasons.append(
            "no benchmark regression: whether this rule added anything to owning "
            "the universe was never asked, and an unasked question is not a pass"
        )
    elif data.residual.alpha <= 0 or data.residual.t_alpha <= 0:
        fatal = True
        reasons.append(
            f"alpha {data.residual.alpha:+.1%}/yr at beta {data.residual.beta:.2f} "
            f"(t={data.residual.t_alpha:+.2f}) -- this rule added nothing to owning "
            f"the universe; its return is the exposure it ran"
        )
    elif not data.residual.is_significant:
        reasons.append(
            f"alpha {data.residual.alpha:+.1%}/yr is positive but not significant "
            f"(t={data.residual.t_alpha:+.2f}, IR {data.residual.information_ratio:+.2f}); "
            f"positive and unmeasurable is not evidence"
        )
    if data.positive_fold_rate < MIN_POSITIVE_FOLD_RATE:
        reasons.append(
            f"only {data.positive_fold_rate:.0%} of walk-forward folds were profitable; "
            "the edge is not consistent across time"
        )
    if data.zero_cost_sharpe is not None and data.zero_cost_sharpe > 0:
        retained = data.oos_sharpe / data.zero_cost_sharpe
        if retained < 0.5:
            fatal = True
            reasons.append(
                f"transaction costs remove {1 - retained:.0%} of the frictionless edge; "
                "this strategy exists only in a world without spreads"
            )
    if data.monte_carlo_p95_drawdown is not None and data.monte_carlo_p95_drawdown > 0.5:
        reasons.append(
            f"the 95th-percentile resampled drawdown is {data.monte_carlo_p95_drawdown:.0%}; "
            "the realised curve was a favourable ordering"
        )
    if data.overfitting is not None:
        if data.overfitting.verdict == "LIKELY_OVERFIT":
            fatal = True
        reasons.append(
            f"overfitting {data.overfitting.score:.2f} ({data.overfitting.verdict}): "
            + "; ".join(s.reason for s in data.overfitting.worst)
        )

    excess: float | None = None
    if data.benchmark_sharpe is not None:
        excess = float(data.oos_sharpe - data.benchmark_sharpe)
        if excess > 0:
            reasons.append(
                f"beat buy and hold on the same symbols: Sharpe {data.oos_sharpe:.2f} "
                f"against {data.benchmark_sharpe:.2f} (+{excess:.2f})"
            )
        else:
            # Descriptive, not a verdict. The regression above is what decides
            # whether anything was added, and the two genuinely disagree when
            # beta is far from one: a rule at beta 0.5 trails a fully-invested
            # benchmark on Sharpe while adding real alpha. This line used to say
            # "added nothing" in the same breath as the regression reporting
            # +5.5%, which is not a contradiction a reader should have to
            # resolve for themselves.
            exposure = (
                f" at beta {data.residual.beta:.2f}" if data.residual is not None else ""
            )
            reasons.append(
                f"trails buy and hold on absolute Sharpe: {data.oos_sharpe:.2f} against "
                f"{data.benchmark_sharpe:.2f} ({excess:.2f}){exposure}"
            )

    verdict: Verdict
    significant = data.residual is not None and data.residual.is_significant
    if fatal:
        verdict = "REJECT"
    elif not significant:
        # Promotion means "worth spending sealed data on". An alpha that cannot
        # be distinguished from zero on the search window will not become
        # distinguishable on two years of it.
        verdict = "REVIEW"
    elif score >= 70:
        verdict = "ACCEPT"
    elif score >= 55:
        verdict = "PAPER"
    else:
        verdict = "REVIEW"

    return Evaluation(
        strategy=data.strategy,
        fingerprint=data.fingerprint,
        score=float(score),
        verdict=verdict,
        components=components,
        reasons=reasons,
        residual=data.residual,
        benchmark_sharpe=data.benchmark_sharpe,
        excess_sharpe=excess,
    )


def _drawdown_component(drawdown: float) -> float:
    """Full marks below 10%, zero at 40%, linear in between."""
    if drawdown <= 0.10:
        return 1.0
    if drawdown >= 0.40:
        return 0.0
    return float(1.0 - (drawdown - 0.10) / 0.30)


def _clip01(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(min(max(value, 0.0), 1.0))
