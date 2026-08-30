"""The research pipeline: hypothesis in, verdict out.

This is the sequence from architecture section 44, wired together in the order
that makes each step's failure cheap:

    validate -> in-sample backtest -> walk-forward -> robustness ->
    overfitting -> evaluation -> registry

Cheap rejections come first. A strategy that never fires is discarded before any
walk-forward runs; one that needs more history than exists never reaches the
robustness suite. The expensive tests are reserved for candidates that have
already survived the cheap ones.

Nothing here calls an LLM. The pipeline judges a strategy; where the strategy
came from is not its concern, which is what lets the same code path evaluate a
hand-written rule and a model-generated one identically.
"""

from __future__ import annotations

import hashlib
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np

from aqr.backtest.alpha import ResidualAlpha, residual_alpha
from aqr.backtest.costs import ZERO_COST, CostModel
from aqr.backtest.engine import BacktestConfig
from aqr.backtest.metrics import Metrics, compute_metrics, periods_per_year
from aqr.backtest.run import run_strategy
from aqr.data.bars import Bars
from aqr.data.universes import PointInTimeUniverse
from aqr.dsl.schema import StrategySpec
from aqr.dsl.validator import validate_against
from aqr.evaluator.score import Evaluation, EvaluationInput, evaluate_strategy
from aqr.features.cross_section import CrossSection
from aqr.features.regime import regime_series
from aqr.registry.db import ExperimentRecord, Registry
from aqr.seal import current as current_seal
from aqr.validation.holdout import benchmark_returns, buy_and_hold
from aqr.validation.overfitting import OverfittingReport, detect_overfitting
from aqr.validation.robustness import (
    AssetReport,
    MonteCarloReport,
    ParameterReport,
    RegimeReport,
    asset_robustness,
    monte_carlo,
    parameter_stability,
    regime_robustness,
)
from aqr.validation.splits import window_bars
from aqr.validation.walkforward import WalkForwardReport, run_walk_forward

__all__ = ["ResearchOutcome", "code_hash", "evaluate_candidate"]


def code_hash() -> str:
    """Fingerprint of the evaluation code itself.

    Two experiments with identical metrics but different code hashes are not
    comparable, and quietly comparing them is how a backtester bug becomes a
    year of misleading research.
    """
    from pathlib import Path

    from aqr.backtest import engine, metrics
    from aqr.core import indicators

    digest = hashlib.sha256()
    for module in (indicators, engine, metrics):
        source = getattr(module, "__file__", None)
        if source:
            digest.update(Path(source).read_bytes())
    return digest.hexdigest()[:16]


@dataclass(slots=True)
class ResearchOutcome:
    """Everything learned about one candidate."""

    spec: StrategySpec
    rejected_early: str | None = None
    in_sample: Metrics | None = None
    zero_cost: Metrics | None = None
    walk_forward: WalkForwardReport | None = None
    parameters: ParameterReport | None = None
    assets: AssetReport | None = None
    regimes: RegimeReport | None = None
    monte_carlo: MonteCarloReport | None = None
    overfitting: OverfittingReport | None = None
    benchmark: Metrics | None = None
    """Equal-weight buy-and-hold on the traded universe over the same window.

    Fourteen strategies reached PAPER before this existed and all fourteen lost
    to it -- Sharpe 1.16 against a best of 0.65 -- because a long-only rule on
    these names over this window makes money almost whatever the rule is."""
    residual: ResidualAlpha | None = None
    """The strategy's out-of-sample returns regressed on the benchmark's.

    This is what the verdict now turns on. The Sharpe comparison above says
    which number was bigger; only the regression says whether the difference was
    earned or rented, and on this universe over this window it is almost always
    rented."""
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

    def summary(self) -> str:
        if self.rejected_early:
            return f"{self.spec.name}: REJECT -- {self.rejected_early}"
        lines = [str(self.evaluation)] if self.evaluation else []
        if self.walk_forward:
            lines.append(str(self.walk_forward))
        for report in (self.parameters, self.assets, self.regimes, self.monte_carlo):
            if report is not None:
                lines.append(str(report))
        if self.overfitting:
            lines.append(str(self.overfitting))
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
        }


def _frictionless_oos_sharpe(
    wf: WalkForwardReport, outcome: ResearchOutcome
) -> float | None:
    """The denominator of "how much of the edge do costs remove".

    It has to be the same windows and the same parameters as the numerator,
    differing only in friction, or the ratio is not about friction. Dividing an
    out-of-sample, with-costs Sharpe by an in-sample, frictionless, full-period
    one charged out-of-sample decay to spreads -- and that gate is fatal, so it
    was rejecting strategies for a reason it had not measured.

    Falls back to the full-period frictionless run only when the walk-forward
    produced no chained curve, where there is nothing better to offer.
    """
    if wf.stitched_zero_cost is not None:
        return wf.stitched_zero_cost.sharpe
    return outcome.zero_cost.sharpe if outcome.zero_cost else None


def evaluate_candidate(
    spec: StrategySpec,
    data: dict[str, Bars],
    *,
    regime_labels: dict[str, list[str]] | None = None,
    membership: PointInTimeUniverse | None = None,
    config: BacktestConfig | None = None,
    train_bars: int | None = None,
    test_bars: int | None = None,
    grid: dict[str, list[float]] | None = None,
    registry: Registry | None = None,
    llm_model: str | None = None,
    prompt_hash: str | None = None,
    dataset_version: str = "synthetic-v1",
    run_robustness: bool = True,
) -> ResearchOutcome:
    """Put one strategy through the full gauntlet.

    ``registry`` is optional but strongly recommended: without it the
    overfitting detector cannot see how many backtests preceded this one, and
    the resulting score will be optimistic in exactly the way this project
    exists to prevent.
    """
    config = config or BacktestConfig()
    outcome = ResearchOutcome(spec=spec)
    universe = [s for s in spec.universe.symbols if s in data]
    if not universe:
        outcome.rejected_early = (
            f"no data for the universe {list(spec.universe.symbols)}"
        )
        _record(
            outcome, spec, data, registry, llm_model, prompt_hash,
            dataset_version, config.costs,
        )
        return outcome
    subset = {s: data[s] for s in universe}
    primary = subset[universe[0]]

    # The spec's bar size and the data's must agree. The timeframe is part of
    # where bars live (``CsvProvider.path_for``), so a mismatch here means the
    # caller loaded the wrong directory -- and would otherwise backtest an
    # hourly rule against daily bars without a word.
    mismatched = {
        s: b.timeframe for s, b in subset.items() if b.timeframe != spec.universe.timeframe
    }
    if mismatched:
        detail = ", ".join(f"{s} on {tf}" for s, tf in sorted(mismatched.items()))
        outcome.rejected_early = (
            f"spec is on {spec.universe.timeframe} bars but the loaded data is {detail}"
        )
        _record(
            outcome, spec, data, registry, llm_model, prompt_hash,
            dataset_version, config.costs,
        )
        return outcome

    # The walk-forward geometry is denominated in *this* bar size: 504 daily
    # bars is two years of fit, 504 hourly bars is seven weeks, and one number
    # cannot mean both. Explicit arguments override the per-timeframe default.
    default_train, default_test = window_bars(spec.universe.timeframe)
    if train_bars is None:
        train_bars = default_train
    if test_bars is None:
        test_bars = default_test

    # --- cheap gate: does this strategy even make sense on this data? ----- #
    report = validate_against(spec, primary, CrossSection(subset))
    outcome.warnings.extend(report.warnings)
    if not report.ok:
        outcome.rejected_early = "; ".join(report.errors)
        _record(
            outcome, spec, data, registry, llm_model, prompt_hash,
            dataset_version, config.costs,
        )
        return outcome

    # --- in-sample, and the same run without friction --------------------- #
    result = run_strategy(spec, subset, config, membership=membership)
    outcome.in_sample = compute_metrics(result)
    outcome.backtests_run += 1

    zero_config = BacktestConfig(
        initial_equity=config.initial_equity,
        costs=ZERO_COST,
        max_portfolio_drawdown=config.max_portfolio_drawdown,
        allow_fractional_shares=config.allow_fractional_shares,
    )
    outcome.zero_cost = compute_metrics(
        run_strategy(spec, subset, zero_config, membership=membership)
    )
    outcome.backtests_run += 1

    if outcome.in_sample.num_trades == 0:
        outcome.rejected_early = "the strategy produced no trades in sample"
        _record(
            outcome, spec, data, registry, llm_model, prompt_hash,
            dataset_version, config.costs,
        )
        return outcome

    # --- walk-forward: the number that actually counts -------------------- #
    wf = run_walk_forward(
        spec,
        subset,
        train_bars=train_bars,
        test_bars=test_bars,
        grid=grid,
        config=config,
        zero_cost_config=zero_config,
        membership=membership,
    )
    outcome.walk_forward = wf
    # +2 per fold, not +1: the test window is run again without friction so
    # the cost gate has something to divide by that differs only in costs.
    outcome.backtests_run += sum(f.candidates_tried + 2 for f in wf.folds)

    if not wf.folds:
        outcome.rejected_early = (
            f"only {max(len(b) for b in subset.values())} bars: not enough history "
            f"for a {train_bars}+{test_bars}-bar walk-forward"
        )
        _record(
            outcome, spec, data, registry, llm_model, prompt_hash,
            dataset_version, config.costs,
        )
        return outcome

    oos = wf.stitched or _worst_case(wf)

    # --- robustness ------------------------------------------------------- #
    if run_robustness:
        outcome.parameters = parameter_stability(spec, subset, config=config)
        outcome.backtests_run += sum(len(v) for v in outcome.parameters.per_slot.values())
        outcome.assets = asset_robustness(spec, data, config=config, membership=membership)
        outcome.backtests_run += len(outcome.assets.per_symbol)
        # Derived from the bars when the caller supplies none, rather than left
        # absent for the evaluator to substitute 0.5 for. That substitution is
        # how a tenth of every real-data score once became a constant standing
        # in for a measurement, and an optional argument is what let it happen
        # again from any caller that did not know to pass one.
        #
        # The argument survives as an override for the simulator, which knows
        # the regime it generated and can supply truth where the classifier can
        # only estimate.
        labels = regime_labels or {
            symbol: regime_series(bars) for symbol, bars in subset.items()
        }
        outcome.regimes = regime_robustness(spec, subset, labels, config=config)
        outcome.backtests_run += 1
        outcome.monte_carlo = monte_carlo(result.trades, config.initial_equity)

    # --- overfitting, informed by the whole search history ---------------- #
    #
    # Distinct rules, not backtests. A parameter perturbation is a diagnostic
    # on a hypothesis already chosen, not another place the search could have
    # got lucky, and counting the ~57 of them per candidate inflated the
    # multiple-comparisons denominator by the same factor.
    prior = registry.distinct_hypotheses() if registry else 0
    outcome.overfitting = detect_overfitting(
        spec,
        result,
        outcome.in_sample,
        train_sharpe=float(
            sum(f.train.sharpe for f in wf.folds) / len(wf.folds)
        ),
        oos_sharpe=wf.oos_sharpe,
        backtests_run=prior + 1,
        zero_cost_sharpe=outcome.zero_cost.sharpe if outcome.zero_cost else None,
    )

    # --- what doing nothing would have paid -------------------------------- #
    #
    # On the traded universe, over *the out-of-sample period only*.
    #
    # Both halves have to cover the same bars. The strategy's number is the mean
    # over walk-forward test folds, which begin after the first training window;
    # measuring the index from the start of the data instead would compare a
    # strategy judged from 1998 against an index measured from 1995, and those
    # three years were among the strongest technology stocks have ever had.
    #
    # Measured on the same survivorship-biased symbol set as the strategy, which
    # is deliberate: the bias is then in both numbers and most of it cancels out
    # of the difference.
    oos_start = wf.folds[0].fold.test.start
    oos_bars = {symbol: bars.slice(oos_start, len(bars)) for symbol, bars in subset.items()}
    outcome.benchmark = buy_and_hold(oos_bars)
    outcome.residual = _residual(wf, oos_bars)

    # --- verdict ---------------------------------------------------------- #
    outcome.evaluation = evaluate_strategy(
        EvaluationInput(
            strategy=spec.name,
            fingerprint=spec.fingerprint(),
            oos=oos,
            oos_sharpe=wf.oos_sharpe,
            positive_fold_rate=wf.positive_fold_rate,
            parameter_stability=outcome.parameters.score if outcome.parameters else 0.5,
            asset_robustness=outcome.assets.score if outcome.assets else 0.5,
            regime_robustness=outcome.regimes.score if outcome.regimes else 0.5,
            overfitting=outcome.overfitting,
            monte_carlo_p95_drawdown=(
                outcome.monte_carlo.p95_drawdown if outcome.monte_carlo else None
            ),
            zero_cost_sharpe=_frictionless_oos_sharpe(wf, outcome),
            benchmark_sharpe=(
                outcome.benchmark.sharpe if outcome.benchmark is not None else None
            ),
            residual=outcome.residual,
        )
    )
    _record(
            outcome, spec, data, registry, llm_model, prompt_hash,
            dataset_version, config.costs,
        )
    return outcome


def _residual(wf: WalkForwardReport, oos_bars: dict[str, Bars]) -> ResidualAlpha | None:
    """Regress the stitched out-of-sample curve on the benchmark over the same days.

    Aligned by session rather than by position. The stitched curve covers the
    walk-forward test windows and the benchmark covers every session in the
    out-of-sample span, and two series of the same length are not necessarily
    the same days -- lining them up by index would compare Tuesday against
    Wednesday and call the difference alpha.
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

    # The stitched curve is per *bar* and the benchmark per *session*. On
    # intraday data a session holds several bars, so compound each session's
    # bars into one strategy return before pairing; otherwise every bar of a
    # day is regressed against the same daily return. On daily bars each
    # session holds exactly one bar and the compounding is the identity.
    strat_by_session: dict[int, float] = {}
    for r, day in zip(bar_returns.tolist(), bar_sessions.tolist(), strict=True):
        key = int(day)
        previous = strat_by_session.get(key)
        # A first entry stored verbatim keeps the daily case -- one bar per
        # session -- bit-for-bit identical to the unaggregated return.
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
    # The paired series is per session, so annualise against session spacing,
    # not bar spacing -- on hourly bars the bar factor would inflate alpha
    # six-and-a-half times.
    paired_timeline = np.array([day for day, _, _ in paired], dtype=np.int64) * 86_400
    return residual_alpha(strat, bench, periods_per_year=periods_per_year(paired_timeline))


def _worst_case(wf: WalkForwardReport) -> Metrics:
    """Fallback out-of-sample metrics when the folds could not be stitched."""
    if not wf.folds:
        from aqr.backtest.metrics import _empty

        return _empty()
    return min(wf.folds, key=lambda f: f.test.sharpe).test


def _record(
    outcome: ResearchOutcome,
    spec: StrategySpec,
    data: dict[str, Bars],
    registry: Registry | None,
    llm_model: str | None,
    prompt_hash: str | None,
    dataset_version: str,
    costs: CostModel,
) -> None:
    """Write the experiment down -- successes and failures alike."""
    if registry is None:
        return
    symbols = tuple(spec.universe.symbols)
    known = [data[s] for s in symbols if s in data]
    if known:
        start = datetime.fromtimestamp(int(known[0].event_time[0]), tz=UTC).isoformat()
        end = datetime.fromtimestamp(int(known[0].event_time[-1]), tz=UTC).isoformat()
    else:
        start = end = ""

    registry.upsert_strategy(spec)
    registry.record_experiment(
        ExperimentRecord(
            fingerprint=spec.fingerprint(),
            strategy_name=spec.name,
            hypothesis=spec.hypothesis,
            symbols=symbols,
            timeframe=spec.universe.timeframe,
            data_start=start,
            data_end=end,
            dataset_version=dataset_version,
            # Cost retention is a fatal gate, so the schedule is part of the
            # verdict rather than context for it.
            costs=costs.as_dict(),
            train_metrics=outcome.in_sample.as_dict() if outcome.in_sample else None,
            oos_metrics=(
                outcome.walk_forward.stitched.as_dict()
                if outcome.walk_forward and outcome.walk_forward.stitched
                else None
            ),
            robustness={
                "parameters": outcome.parameters.as_dict() if outcome.parameters else None,
                "assets": outcome.assets.as_dict() if outcome.assets else None,
                "regimes": outcome.regimes.as_dict() if outcome.regimes else None,
                "monte_carlo": outcome.monte_carlo.as_dict() if outcome.monte_carlo else None,
                "benchmark": outcome.benchmark.as_dict() if outcome.benchmark else None,
                "walk_forward": outcome.walk_forward.as_dict() if outcome.walk_forward else None,
            },
            overfitting=outcome.overfitting.as_dict() if outcome.overfitting else None,
            evaluation=outcome.evaluation.as_dict() if outcome.evaluation else None,
            verdict=outcome.verdict,
            score=outcome.score,
            backtests_run=max(outcome.backtests_run, 1),
            llm_model=llm_model,
            prompt_hash=prompt_hash,
            code_hash=code_hash(),
            error=outcome.rejected_early,
            # The certificate of the process that ran this experiment, so the
            # ancestry taint check is a query rather than an act of trust. Read
            # here rather than passed in: a caller that could supply its own
            # certificate could supply a clean one.
            seal=current_seal().certificate(),
        )
    )
    if outcome.evaluation:
        registry.set_score(spec.fingerprint(), outcome.evaluation.score)
    # A strategy already in a terminal state stays there; research history is
    # append-only, so a repeat verdict must not rewrite it.
    if outcome.verdict == "REJECT":
        with suppress(ValueError):
            registry.set_status(spec.fingerprint(), "REJECTED", outcome.rejected_early or "score")
    elif outcome.verdict in ("ACCEPT", "PAPER"):
        with suppress(ValueError):
            registry.set_status(spec.fingerprint(), "PAPER", f"score {outcome.score:.0f}")
