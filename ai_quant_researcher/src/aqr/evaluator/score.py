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
from aqr.backtest.engine import Trade
from aqr.backtest.metrics import Metrics
from aqr.validation.cycles import independent_cycles
from aqr.validation.overfitting import OverfittingReport

__all__ = [
    "AFFORDABILITY_BOUND_THRESHOLD",
    "Evaluation",
    "EvaluationInput",
    "MIN_INDEPENDENT_CYCLES",
    "OPTION_WEIGHTS",
    "OptionEvaluationInput",
    "Verdict",
    "evaluate_option_strategy",
    "evaluate_strategy",
]

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
    affordability_bound_fraction: float | None = None
    """specs/10 D8a. Set by :func:`evaluate_option_strategy`; left ``None`` by
    :func:`evaluate_strategy`, which has no analogous pathology -- the equity
    engine buys fractional-of-a-position dollar amounts and never refuses an
    entry for being too small to cover one contract's defined-risk maximum
    loss the way a spread does. ``None`` here means "not applicable", not
    "measured at zero" -- the same distinction ``excess_sharpe`` already
    draws for "nobody looked."""

    @property
    def beats_benchmark(self) -> bool | None:
        return None if self.excess_sharpe is None else bool(self.excess_sharpe > 0)

    @property
    def is_affordability_bound(self) -> bool | None:
        """Whether D8a's fraction clears :data:`AFFORDABILITY_BOUND_THRESHOLD`.

        ``None`` when :attr:`affordability_bound_fraction` is ``None``
        (the equity path, or an options run whose caller never measured it)
        -- mirrors :attr:`beats_benchmark`'s three-valued shape for the same
        reason: a reader must be able to tell "not bound" from "not asked."
        """
        if self.affordability_bound_fraction is None:
            return None
        return self.affordability_bound_fraction >= AFFORDABILITY_BOUND_THRESHOLD

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
            "affordability_bound_fraction": self.affordability_bound_fraction,
            "is_affordability_bound": self.is_affordability_bound,
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


# --------------------------------------------------------------------------- #
# The options profile — specs/10-options-research.md D8
# --------------------------------------------------------------------------- #
#
# A sibling entry point, not a rewrite of the one above. Every existing test
# against ``evaluate_strategy`` / ``EvaluationInput`` / ``WEIGHTS`` /
# ``MIN_TRADES`` keeps passing unchanged, because none of those names are
# touched below -- D8 changes what "enough evidence" and "robust against what"
# mean for a structure held to expiry on one underlying, and both of those are
# genuinely different questions from the ones the equity profile answers, not
# the same question answered more strictly.

MIN_INDEPENDENT_CYCLES = 25
"""D8's gate, replacing ``MIN_TRADES``. specs/10 measured 71 non-overlapping
28-DTE cycles across the whole 5.55-year research window -- ``MIN_TRADES``'s
30 *trades* is close to meaningless against that, because thirty overlapping
credit spreads (``max_concurrent`` > 1, or a cadence shorter than the DTE)
can be eight independent bets wearing thirty tickets. The gate counts what
:func:`~aqr.validation.cycles.independent_cycles` counts -- greedily walk
entries in order, credit one, skip every later entry that opened before the
credited one's own expiry -- so a report and this gate can never disagree
about what a cycle is; they call the same function."""

AFFORDABILITY_BOUND_THRESHOLD = 0.5
"""specs/10 D8a. Below this fraction, most of what kept a rule's cycle count
low was the rule declining to trade or the ladder declining to offer a leg;
above it, most of what kept the cycle count low was the account being too
small to afford the structure the rule already wanted. Measured on the real
cache at ``risk_per_trade=0.01`` (``put_credit_spread`` / 28 DTE / 0.16
anchor / 0.06 width): 578 affordability skips against 31 trades is 94.9%,
nowhere near this line; the line exists to separate that run from one where
affordability explains a fifth or a third of the shortfall rather than
essentially all of it. Set at the midpoint for the same reason
``GREEK_CONSISTENCY_TOLERANCE`` is: this is a fraction of a *count*, not a
measurement with sampling noise on both sides of a boundary, so there is no
data-driven way to place it more precisely than "clearly more than not."

A REJECT below this line is still worded as ordinary -- the fraction is
always in ``reasons[0]`` regardless, per D8a's requirement that a reader
never sees the cycle count without the fraction beside it -- but only above
this line does the REJECT's own reason say the run measured the account
rather than the rule."""

OPTION_WEIGHTS: dict[str, float] = {
    "information_ratio": 0.30,
    "profit_factor": 0.20,
    "max_drawdown": 0.15,
    "parameter_stability": 0.15,
    "year_dte_robustness": 0.10,
    "regime_robustness": 0.10,
}
"""``WEIGHTS`` with one swap: D8 replaces ``asset_robustness`` (undefined on
one underlying -- specs/10 D10) with the combined leave-one-year-out /
DTE-bucket-agreement score from ``options/robustness.py``, at the same 10%
weight ``asset_robustness`` held. Every other component and every other
weight is identical to the equity profile on purpose -- an information ratio,
a profit factor and a drawdown mean the same thing regardless of what
produced the trades, and D8 does not ask for those three to be re-derived."""


@dataclass(slots=True)
class OptionEvaluationInput:
    """:class:`EvaluationInput`'s options counterpart.

    Carries ``oos_trades`` rather than a pre-computed cycle count on purpose:
    D8 requires the evaluator and every report to agree on what a cycle is,
    and the only way two call sites cannot drift apart on that question is if
    they call the same function on the same list. ``evaluate_option_strategy``
    calls :func:`~aqr.validation.cycles.independent_cycles` on this field
    itself rather than trusting a number a caller already computed.
    """

    strategy: str
    fingerprint: str
    oos: Metrics
    oos_sharpe: float
    oos_trades: list[Trade]
    positive_fold_rate: float
    parameter_stability: float
    year_dte_robustness: float
    """D8's replacement for ``asset_robustness`` -- see ``OPTION_WEIGHTS``."""
    regime_robustness: float
    affordability_bound_fraction: float
    """specs/10 D8a, from
    :func:`~aqr.options.engine.affordability_bound_fraction` (or
    :attr:`~aqr.options.walkforward.OptionWalkForwardReport.oos_affordability_bound_fraction`
    for a walk-forward run) -- computed on the same out-of-sample sample this
    input's ``oos_trades`` came from, never on the full-window run, so the
    number a caller passes here always describes the sample this evaluation
    is actually judging. No default: a caller that has not measured it must
    say so at the call site rather than have this silently read as "not
    affordability-bound," which is a claim, not an absence of one."""
    residual: ResidualAlpha | None = None
    overfitting: OverfittingReport | None = None
    monte_carlo_p95_drawdown: float | None = None
    zero_cost_sharpe: float | None = None
    benchmark_sharpe: float | None = None


def evaluate_option_strategy(data: OptionEvaluationInput) -> Evaluation:
    """Score an option strategy and decide what happens to it next.

    Same shape as :func:`evaluate_strategy` -- gates before weights,
    out-of-sample only, residual rather than absolute Sharpe -- with two
    substitutions D8 requires: the sample-size gate reads independent cycles
    instead of trade count, and the cross-sectional-robustness slot reads
    year/DTE-bucket robustness instead of asset robustness. Returns the same
    :class:`Evaluation` type ``evaluate_strategy`` returns, so nothing
    downstream of a verdict needs to know which profile produced it.
    """
    cycles = independent_cycles(data.oos_trades)
    ir = data.residual.information_ratio if data.residual is not None else 0.0
    components = {
        "information_ratio": 100.0 * _curve(ir, good=GOOD_INFORMATION_RATIO),
        "profit_factor": 100.0 * _curve(data.oos.profit_factor - 1.0, good=0.8),
        "max_drawdown": 100.0 * _drawdown_component(data.oos.max_drawdown),
        "parameter_stability": 100.0 * _clip01(data.parameter_stability),
        "year_dte_robustness": 100.0 * _clip01(data.year_dte_robustness),
        "regime_robustness": 100.0 * _clip01(data.regime_robustness),
    }
    score = sum(components[k] * w for k, w in OPTION_WEIGHTS.items())

    reasons: list[str] = [
        f"{len(data.oos_trades)} out-of-sample trades, {cycles} independent cycles "
        "(the gate: specs/10 D8 counts a cycle, not a trade); "
        f"{data.affordability_bound_fraction:.0%} of skipped entries were "
        "affordability-bound (D8a: unaffordable, not unwanted by the rule)"
    ]
    fatal = False

    if cycles < MIN_INDEPENDENT_CYCLES:
        fatal = True
        if data.affordability_bound_fraction >= AFFORDABILITY_BOUND_THRESHOLD:
            # D8a's own requirement: a reader must never see this REJECT
            # without also seeing that most of the shortfall was the account,
            # not the market -- so the two facts are stated in the same
            # sentence rather than trusting reasons[0], above, to be read
            # first. Still fatal, still REJECT: the >= 25 gate is not
            # loosened by this, because a run that could not afford enough
            # cycles has not measured the rule either way, and PAPER or
            # ACCEPT would claim it had.
            reasons.append(
                f"{cycles} independent out-of-sample cycles is below the "
                f"{MIN_INDEPENDENT_CYCLES} minimum, but {data.affordability_bound_fraction:.0%} "
                "of the sessions that reached sizing were skipped for being unaffordable "
                "at this risk_per_trade, not because the rule declined to trade or the "
                "ladder had nothing to offer (D8a) -- this REJECT is a verdict on the "
                "account this run was sized against, not on the rule; it does not become "
                "evidence against the rule until it is re-run at a sizing the account can "
                "actually afford"
            )
        else:
            reasons.append(
                f"{cycles} independent out-of-sample cycles is below the "
                f"{MIN_INDEPENDENT_CYCLES} minimum; a structure held to expiry only "
                "produces new evidence once it closes, and this run has not closed "
                "enough of them to distinguish an edge from noise"
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
            "the underlying was never asked, and an unasked question is not a pass"
        )
    elif data.residual.alpha <= 0 or data.residual.t_alpha <= 0:
        fatal = True
        reasons.append(
            f"alpha {data.residual.alpha:+.1%}/yr at beta {data.residual.beta:.2f} "
            f"(t={data.residual.t_alpha:+.2f}) -- this rule added nothing to owning "
            f"the underlying; its return is the exposure it ran"
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
            "the edge is not consistent across calendar time"
        )
    if data.zero_cost_sharpe is not None and data.zero_cost_sharpe > 0:
        retained = data.oos_sharpe / data.zero_cost_sharpe
        if retained < 0.5:
            fatal = True
            reasons.append(
                f"transaction costs remove {1 - retained:.0%} of the frictionless edge; "
                "the crossed spread (D2) is not optional and this strategy exists only "
                "without it"
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
                f"beat buy and hold on the same underlying: Sharpe {data.oos_sharpe:.2f} "
                f"against {data.benchmark_sharpe:.2f} (+{excess:.2f})"
            )
        else:
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
        affordability_bound_fraction=data.affordability_bound_fraction,
    )
