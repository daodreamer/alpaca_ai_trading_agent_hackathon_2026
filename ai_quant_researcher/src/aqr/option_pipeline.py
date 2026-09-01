"""The option research pipeline: hypothesis in, verdict out — specs/10 D8.

[`pipeline.py`](pipeline.py)'s counterpart, in the same order and for the same
reason — cheap rejections first, expensive tests reserved for candidates that
survived them:

    run -> walk-forward -> robustness -> overfitting -> residual alpha ->
    evaluation -> registry

A sibling module rather than a branch inside ``evaluate_candidate``. The two
pipelines share their *shape* and almost none of their steps: there is no
cross-sectional asset robustness on one underlying (D8 replaces it with
leave-one-year-out and DTE-bucket agreement), the walk-forward is cut on the
calendar rather than in bars (``options/walkforward.py``), the sample-size gate
counts independent cycles rather than trades, and the frictionless comparison
can only remove commission because the spread is in the quotes and D2 charges
it in full. A single function carrying both would branch at every one of those
points, and the branch nobody updated would be the one that silently scored an
option rule by equity rules.

What is genuinely shared is shared: :class:`~aqr.backtest.metrics.Metrics`,
``residual_alpha``, ``detect_overfitting``, the registry, and the
:class:`~aqr.evaluator.score.Evaluation` type a verdict comes back in. That is
specs/10's one architectural commitment — the options engine emits the same
``BacktestResult`` — cashed in.

Nothing here calls an LLM. The pipeline judges a rule; where the rule came from
is not its concern.
"""

from __future__ import annotations

import hashlib
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any

import numpy as np

from aqr.backtest.alpha import ResidualAlpha, residual_alpha
from aqr.backtest.metrics import Metrics, compute_metrics, periods_per_year
from aqr.data.bars import Bars
from aqr.evaluator.score import (
    Evaluation,
    OptionEvaluationInput,
    evaluate_option_strategy,
)
from aqr.options.costs import ZERO_OPTION_COST, OptionCostModel
from aqr.options.engine import OptionBacktestConfig, OptionBacktestResult
from aqr.options.robustness import (
    DteBucketReport,
    YearReport,
    dte_bucket_agreement,
    leave_one_year_out,
    option_parameter_stability,
    option_regime_robustness,
)
from aqr.options.run import OptionMarket, run_option_strategy
from aqr.options.spec import OptionSpec
from aqr.options.walkforward import OptionWalkForwardReport, run_option_walk_forward
from aqr.registry.db import ExperimentRecord, Registry
from aqr.seal import current as current_seal
from aqr.validation.cycles import independent_cycles
from aqr.validation.holdout import benchmark_returns, buy_and_hold
from aqr.validation.overfitting import OverfittingReport, detect_overfitting
from aqr.validation.robustness import ParameterReport, RegimeReport

__all__ = [
    "OptionResearchOutcome",
    "evaluate_option_candidate",
    "option_code_hash",
]


def option_code_hash() -> str:
    """Fingerprint of the option evaluation code itself.

    Its own hash rather than ``pipeline.code_hash()``: two option experiments
    with identical metrics and different engine code are not comparable, and the
    modules that decide an option result — the chain selector, the structure
    arithmetic, the engine — are not the modules ``pipeline.code_hash`` reads.
    """
    from pathlib import Path

    from aqr.backtest import metrics
    from aqr.options import chain, engine, features, structure

    digest = hashlib.sha256()
    for module in (chain, structure, engine, features, metrics):
        source = getattr(module, "__file__", None)
        if source:
            digest.update(Path(source).read_bytes())
    return digest.hexdigest()[:16]


@dataclass(slots=True)
class OptionResearchOutcome:
    """Everything learned about one option rule."""

    spec: OptionSpec
    rejected_early: str | None = None
    in_sample: Metrics | None = None
    zero_cost: Metrics | None = None
    """The same run with commission and fees removed.

    Note what this *cannot* remove: the crossed spread. D2 charges the quoted
    bid and ask in full and offers no mid-price option, so the cost-retention
    ratio here answers "how much of the edge does commission take" and not "how
    much does friction take". On a $1.50 credit spread at the median 1.5%
    relative spread, friction is mostly the part this number leaves in — which
    makes a strategy that fails this gate a strategy that could not even pay
    its commission.
    """
    result: OptionBacktestResult | None = None
    walk_forward: OptionWalkForwardReport | None = None
    parameters: ParameterReport | None = None
    years: YearReport | None = None
    dte_buckets: DteBucketReport | None = None
    regimes: RegimeReport | None = None
    overfitting: OverfittingReport | None = None
    benchmark: Metrics | None = None
    """Buy and hold on the underlying over the out-of-sample window.

    The only honest benchmark for a rule on one underlying: a short put spread
    is a levered, capped long position, so most of what it earns is the drift it
    was exposed to and only the residual is the rule's own."""
    residual: ResidualAlpha | None = None
    evaluation: Evaluation | None = None
    warnings: list[str] = field(default_factory=list)
    backtests_run: int = 0

    @property
    def verdict(self) -> str:
        if self.rejected_early:
            return "REJECT"
        return self.evaluation.verdict if self.evaluation else "UNKNOWN"

    @property
    def score(self) -> float:
        return self.evaluation.score if self.evaluation else 0.0

    @property
    def result_has_trades(self) -> bool:
        """Whether the full-window run opened anything at all.

        Distinct from ``rejected_early is None``: a rule can be rejected early
        for having no walk-forward folds while still having opened positions,
        and only the first of those is worth sending back to a proposer as
        "your rule cannot fire". The second is a fact about the window.
        """
        return bool(self.result is not None and self.result.trades)

    @property
    def year_dte_robustness(self) -> float:
        """D8's replacement for ``asset_robustness``, as one number.

        The mean of the two, and neither one alone. Leave-one-year-out asks
        whether the rule survives the removal of any single year — the 2020 and
        2022 volatility episodes dominate this window, and a rule that is one
        of them is not a rule. DTE-bucket agreement asks whether it keeps its
        sign at 14, 28 and 49 days, because a rule that works at exactly one DTE
        target found a DTE target rather than a premium. Both are necessary and
        neither is sufficient, so they average rather than either one gating.
        """
        scores = [
            report.score for report in (self.years, self.dte_buckets) if report is not None
        ]
        return float(sum(scores) / len(scores)) if scores else 0.5

    def summary(self) -> str:
        if self.rejected_early:
            return f"{self.spec.name}: REJECT -- {self.rejected_early}"
        lines = [str(self.evaluation)] if self.evaluation else []
        if self.walk_forward:
            lines.append(str(self.walk_forward))
        for report in (self.parameters, self.years, self.dte_buckets, self.regimes):
            if report is not None:
                lines.append(str(report))
        if self.overfitting:
            lines.append(str(self.overfitting))
        if self.result is not None:
            lines.append(str(self.result.skip_census))
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.spec.name,
            "fingerprint": self.spec.fingerprint(),
            "verdict": self.verdict,
            "score": self.score,
            "rejected_early": self.rejected_early,
            "in_sample": self.in_sample.as_dict() if self.in_sample else None,
            "walk_forward": self.walk_forward.as_dict() if self.walk_forward else None,
            "warnings": self.warnings,
            "backtests_run": self.backtests_run,
            "benchmark": self.benchmark.as_dict() if self.benchmark else None,
            "skip_census": (
                self.result.skip_census.as_dict() if self.result is not None else None
            ),
        }


def evaluate_option_candidate(
    spec: OptionSpec,
    market: OptionMarket,
    *,
    config: OptionBacktestConfig | None = None,
    registry: Registry | None = None,
    llm_model: str | None = None,
    prompt_hash: str | None = None,
    dataset_version: str = "options-cache",
    run_robustness: bool = True,
) -> OptionResearchOutcome:
    """Put one option rule through the full gauntlet.

    ``registry`` is optional and strongly recommended: without it the overfitting
    detector cannot see how many hypotheses preceded this one. The count it is
    given is the **option** search's own (``distinct_hypotheses(family="option")``),
    never the combined total — D8 is explicit that a multiplicity denominator
    mixing the 414-hypothesis equity search with a 20-hypothesis option search
    makes both bars wrong.
    """
    config = config or OptionBacktestConfig()
    outcome = OptionResearchOutcome(spec=spec)

    # --- the full-window run: cheap, and it is also the first gate --------- #
    result = run_option_strategy(spec, market, config)
    outcome.result = result
    outcome.in_sample = compute_metrics(result)
    outcome.backtests_run += 1

    if not result.trades:
        # Not "no trades" as a bare fact. The census says which of the six
        # mutually exclusive reasons it was, and the difference between "the
        # rule never fired" and "the account could not afford it" (D8a) is the
        # difference between a bad hypothesis and a badly sized run.
        outcome.rejected_early = (
            f"the rule opened no positions in {len(market.chain.sessions)} sessions: "
            f"{result.skip_census}"
        )
        _record(outcome, market, registry, llm_model, prompt_hash, dataset_version, config.costs)
        return outcome

    zero_config = OptionBacktestConfig(
        initial_equity=config.initial_equity,
        costs=ZERO_OPTION_COST,
        rate=config.rate,
        max_reference_staleness_days=config.max_reference_staleness_days,
        settle_before=config.settle_before,
    )
    outcome.zero_cost = compute_metrics(run_option_strategy(spec, market, zero_config))
    outcome.backtests_run += 1

    # --- walk-forward: the number that actually counts --------------------- #
    wf = run_option_walk_forward(spec, market, config)
    outcome.walk_forward = wf
    outcome.backtests_run += len(wf.folds)

    if not wf.folds:
        first, last = market.chain.sessions[0], market.chain.sessions[-1]
        outcome.rejected_early = (
            f"{first.isoformat()} -> {last.isoformat()} is not enough calendar time "
            f"for a {wf.train_months}+{wf.test_months}-month walk-forward"
        )
        _record(outcome, market, registry, llm_model, prompt_hash, dataset_version, config.costs)
        return outcome

    oos = wf.stitched or _worst_fold(wf)

    # --- robustness -------------------------------------------------------- #
    if run_robustness:
        outcome.parameters = option_parameter_stability(spec, market, config)
        outcome.backtests_run += sum(len(v) for v in outcome.parameters.per_slot.values())
        outcome.years = leave_one_year_out(spec, market, config)
        outcome.backtests_run += len(outcome.years.per_year)
        outcome.dte_buckets = dte_bucket_agreement(spec, market, config)
        outcome.backtests_run += len(outcome.dte_buckets.per_bucket)
        outcome.regimes = option_regime_robustness(
            result.trades, market.underlying, initial_equity=config.initial_equity
        )

    # --- overfitting, informed by the option search's own history ---------- #
    prior = registry.distinct_hypotheses(family="option") if registry else 0
    outcome.overfitting = detect_overfitting(
        spec,
        result,
        outcome.in_sample,
        train_sharpe=float(sum(f.train.sharpe for f in wf.folds) / len(wf.folds)),
        oos_sharpe=wf.oos_sharpe,
        backtests_run=prior + 1,
        zero_cost_sharpe=outcome.zero_cost.sharpe if outcome.zero_cost else None,
    )

    # --- what owning the underlying would have paid ------------------------ #
    #
    # Over the out-of-sample window only, for the same reason the equity
    # pipeline restricts it: a strategy judged from the first test fold and a
    # benchmark measured from the start of the data are not measuring the same
    # years, and on this window those years are 2020 and 2022.
    oos_bars = _from(market.underlying, wf.folds[0].fold.test_start)
    outcome.benchmark = buy_and_hold({spec.underlying: oos_bars})
    outcome.residual = _residual(wf, {spec.underlying: oos_bars})

    # --- verdict ----------------------------------------------------------- #
    outcome.evaluation = evaluate_option_strategy(
        OptionEvaluationInput(
            strategy=spec.name,
            fingerprint=spec.fingerprint(),
            oos=oos,
            oos_sharpe=wf.oos_sharpe,
            oos_trades=wf.oos_trades,
            positive_fold_rate=wf.positive_fold_rate,
            parameter_stability=outcome.parameters.score if outcome.parameters else 0.5,
            year_dte_robustness=outcome.year_dte_robustness,
            regime_robustness=outcome.regimes.score if outcome.regimes else 0.5,
            # Measured on the same out-of-sample sample ``oos_trades`` came
            # from, never on the full-window run: D8a's whole point is that the
            # fraction has to describe the sample the verdict is about.
            affordability_bound_fraction=wf.oos_affordability_bound_fraction,
            residual=outcome.residual,
            overfitting=outcome.overfitting,
            zero_cost_sharpe=_frictionless_oos_sharpe(wf, outcome),
            benchmark_sharpe=(
                outcome.benchmark.sharpe if outcome.benchmark is not None else None
            ),
        )
    )
    _record(outcome, market, registry, llm_model, prompt_hash, dataset_version, config.costs)
    return outcome


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _frictionless_oos_sharpe(
    wf: OptionWalkForwardReport, outcome: OptionResearchOutcome
) -> float | None:
    """The denominator of "how much of the edge does commission remove".

    Falls back to the full-window frictionless run, which is the honest thing to
    offer: the options walk-forward does not re-run each fold without costs the
    way the equity one does, so there is no fold-matched frictionless curve to
    divide by. Named here rather than silently substituted -- the ratio is then
    across different windows, which flatters or penalises depending on where the
    costs actually fell, and the evaluator's cost gate is fatal.
    """
    return outcome.zero_cost.sharpe if outcome.zero_cost else None


def _worst_fold(wf: OptionWalkForwardReport) -> Metrics:
    """Fallback out-of-sample metrics when the folds could not be stitched."""
    from aqr.backtest.metrics import _empty

    if not wf.folds:
        return _empty()
    return min(wf.folds, key=lambda f: f.test.sharpe).test


def _from(bars: Bars, start: date) -> Bars:
    """The underlying from ``start`` onward, sliced by date rather than by index.

    The walk-forward's folds are cut on the calendar (specs/10 D8) and the bars
    are indexed by position, so the boundary has to be translated once, here,
    rather than at each call site that wants "the out-of-sample bars".
    """
    boundary = int(datetime(start.year, start.month, start.day, tzinfo=UTC).timestamp())
    at = int(np.searchsorted(bars.event_time, boundary, side="left"))
    return bars.slice(at, len(bars))


def _residual(
    wf: OptionWalkForwardReport, oos_bars: dict[str, Bars]
) -> ResidualAlpha | None:
    """Regress the stitched out-of-sample curve on the underlying over the same days.

    Aligned by session rather than by position, exactly as ``pipeline._residual``
    is and for the same reason: the stitched curve covers the test windows and
    the benchmark covers every session in the span, so lining them up by index
    would compare Tuesday against Wednesday and call the difference alpha.

    The options curve is a *marked* one (specs/10 D1a): between entry and expiry
    it moves with the Black-Scholes valuation of an open structure, which is
    what gives a short-premium book the beta and the drawdown that cash
    accounting destroys. Regressing a cash-only curve here would attribute the
    whole of a short put spread's return to alpha, which is the exact failure
    ``alpha.py`` exists to prevent.
    """
    if wf.stitched_equity is None or wf.stitched_timeline is None:
        return None
    if wf.stitched_equity.size < 3:
        return None
    series = benchmark_returns(oos_bars)
    if series is None:
        return None
    sessions, bench_returns = series

    equity = wf.stitched_equity
    with np.errstate(divide="ignore", invalid="ignore"):
        bar_returns = equity[1:] / equity[:-1] - 1.0
    bar_sessions = (wf.stitched_timeline // 86_400).astype(np.int64)[1:]

    strat_by_session: dict[int, float] = {}
    for r, day in zip(bar_returns.tolist(), bar_sessions.tolist(), strict=True):
        key = int(day)
        previous = strat_by_session.get(key)
        strat_by_session[key] = (
            r if previous is None else (1.0 + previous) * (1.0 + r) - 1.0
        )

    # ``sessions`` has one more entry than ``bench_returns``: the return at index
    # i belongs to the transition *into* session i+1.
    bench_by_session = dict(zip(sessions[1:].tolist(), bench_returns.tolist(), strict=True))
    paired = [
        (day, strat_r, bench_by_session[day])
        for day, strat_r in strat_by_session.items()
        if day in bench_by_session
    ]
    if len(paired) < 3:
        return None
    strat = np.array([s for _, s, _ in paired], dtype=np.float64)
    bench = np.array([b for _, _, b in paired], dtype=np.float64)
    timeline = np.array([day for day, _, _ in paired], dtype=np.int64) * 86_400
    return residual_alpha(strat, bench, periods_per_year=periods_per_year(timeline))


def _record(
    outcome: OptionResearchOutcome,
    market: OptionMarket,
    registry: Registry | None,
    llm_model: str | None,
    prompt_hash: str | None,
    dataset_version: str,
    costs: OptionCostModel,
) -> None:
    """Write the experiment down -- successes and failures alike.

    Filed under ``family="option"``, which is what keeps this search's
    multiplicity denominator separate from the equity one (D8). Everything else
    is the equity ``_record``'s shape: the same table, the same seal
    certificate, the same append-only status transitions.
    """
    if registry is None:
        return
    spec = outcome.spec
    sessions = market.chain.sessions
    start = sessions[0].isoformat() if sessions else ""
    end = sessions[-1].isoformat() if sessions else ""

    registry.upsert_option_strategy(spec)
    registry.record_experiment(
        ExperimentRecord(
            fingerprint=spec.fingerprint(),
            strategy_name=spec.name,
            hypothesis=spec.hypothesis,
            symbols=(spec.underlying,),
            # Not a bar size: the option chain's grid is one snapshot per
            # session and is not the bar grid at all (D0). Recording "1D" here
            # would say the walk-forward was cut in daily bars, which is the one
            # thing ``options/walkforward.py`` exists not to do.
            timeframe="option_chain",
            data_start=start,
            data_end=end,
            dataset_version=dataset_version,
            family="option",
            # Cost retention is a fatal gate here as on the equity side, so the
            # schedule is part of the verdict rather than context for it -- and
            # an option schedule and an equity one are not the same units
            # ($/contract/leg against $/share), so a reader comparing two
            # verdicts has to be able to see which was charged.
            costs=costs.as_dict(),
            train_metrics=outcome.in_sample.as_dict() if outcome.in_sample else None,
            oos_metrics=(
                outcome.walk_forward.stitched.as_dict()
                if outcome.walk_forward and outcome.walk_forward.stitched
                else None
            ),
            robustness={
                "parameters": outcome.parameters.as_dict() if outcome.parameters else None,
                "years": outcome.years.as_dict() if outcome.years else None,
                "dte_buckets": (
                    outcome.dte_buckets.as_dict() if outcome.dte_buckets else None
                ),
                "regimes": outcome.regimes.as_dict() if outcome.regimes else None,
                "benchmark": outcome.benchmark.as_dict() if outcome.benchmark else None,
                "walk_forward": (
                    outcome.walk_forward.as_dict() if outcome.walk_forward else None
                ),
                "skip_census": (
                    outcome.result.skip_census.as_dict() if outcome.result else None
                ),
                "independent_cycles": (
                    independent_cycles(outcome.walk_forward.oos_trades)
                    if outcome.walk_forward
                    else 0
                ),
            },
            overfitting=outcome.overfitting.as_dict() if outcome.overfitting else None,
            evaluation=outcome.evaluation.as_dict() if outcome.evaluation else None,
            verdict=outcome.verdict,
            score=outcome.score,
            backtests_run=max(outcome.backtests_run, 1),
            llm_model=llm_model,
            prompt_hash=prompt_hash,
            code_hash=option_code_hash(),
            error=outcome.rejected_early,
            seal=current_seal().certificate(),
        )
    )
    if outcome.evaluation:
        registry.set_score(spec.fingerprint(), outcome.evaluation.score)
    if outcome.verdict == "REJECT":
        with suppress(ValueError):
            registry.set_status(
                spec.fingerprint(), "REJECTED", outcome.rejected_early or "score"
            )
    elif outcome.verdict in ("ACCEPT", "PAPER"):
        with suppress(ValueError):
            registry.set_status(spec.fingerprint(), "PAPER", f"score {outcome.score:.0f}")
