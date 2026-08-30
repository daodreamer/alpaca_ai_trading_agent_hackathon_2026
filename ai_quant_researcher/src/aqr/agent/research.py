"""The research loop (architecture sections 8 and 25).

    propose -> compile -> validate -> evaluate -> record -> remember -> propose

The loop is the product. Any one hypothesis is probably wrong; what makes this
research rather than guessing is that every attempt is written down, every
attempt raises the multiple-comparisons bar for the next one, and promotion is
decided by out-of-sample evidence rather than by whoever is watching.

Three properties are load-bearing:

*The proposer never touches the verdict.* It receives memory and returns fields.
Scoring happens in code it cannot reach.

*A malformed proposal costs one iteration, not the run.* Models produce invalid
output; the loop records the failure as an experiment and continues.

*Nothing is promoted past PAPER here.* Live promotion is a human decision, and
the registry's state machine will not accept the jump regardless.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from aqr.agent.proposer import (
    HeuristicProposer,
    Proposal,
    Proposer,
    build_spec,
    spec_to_proposal_fields,
)
from aqr.backtest.engine import BacktestConfig
from aqr.data.bars import Bars
from aqr.data.universes import PointInTimeUniverse
from aqr.dsl.loader import save_file
from aqr.dsl.schema import StrategySpec
from aqr.dsl.validator import validate_against
from aqr.features.cross_section import CrossSection
from aqr.pipeline import ResearchOutcome, evaluate_candidate
from aqr.registry.db import ExperimentRecord, Registry
from aqr.validation.splits import window_bars

__all__ = ["ResearchConfig", "ResearchLoop", "ResearchStep", "saved_filename"]


def saved_filename(spec: StrategySpec) -> str:
    """Where a promoted strategy is written.

    Name *and* fingerprint. Models reuse names: a live campaign produced two
    different rules both called ``breadth_extreme_reversal_long_v1``, with
    different verdicts. The registry handled it -- it is keyed by content --
    but both were written to one file, so the survivor on disk was whichever
    finished last and the other was simply gone.
    """
    return f"{spec.name}-{spec.fingerprint()}.yaml"


@dataclass(slots=True)
class ResearchConfig:
    symbols: list[str]
    timeframe: str = "1D"
    """The default granularity: what the proposer is told to fall back to and
    what a proposal without a ``timeframe`` field compiles to."""
    timeframes: tuple[str, ...] = ("1D",)
    """The allowed set a proposal may choose from. A choice outside it is a
    compile failure, fed back to the model like any other invalid field."""
    iterations: int = 8
    risk_per_trade: float = 0.0075
    max_positions: int = 3
    train_bars: int | None = None
    test_bars: int | None = None
    """Walk-forward geometry. ``None`` derives it from each candidate's own bar
    size via ``window_bars`` -- 504 daily bars is two years, 504 hourly bars is
    seven weeks, and one number cannot mean both. Explicit values override."""
    memory_depth: int = 20
    mutate_best_every: int = 3
    """Every Nth iteration, refine the best strategy so far instead of proposing
    a new mechanism. Pure exploration never converges; pure refinement never
    discovers anything."""
    dataset_version: str = "synthetic-v1"
    save_accepted_to: str | None = "strategies"


@dataclass(slots=True)
class ResearchStep:
    iteration: int
    proposal: Proposal
    spec: StrategySpec | None = None
    outcome: ResearchOutcome | None = None
    error: str | None = None

    @property
    def verdict(self) -> str:
        if self.error:
            return "ERROR"
        return self.outcome.verdict if self.outcome else "UNKNOWN"

    @property
    def score(self) -> float:
        return self.outcome.score if self.outcome else 0.0

    def __str__(self) -> str:
        name = self.spec.name if self.spec else self.proposal.fields.get("name", "?")
        if self.error:
            return f"[{self.iteration}] {name}: ERROR -- {self.error}"
        return f"[{self.iteration}] {name}: {self.verdict} ({self.score:.0f}/100)"


@dataclass(slots=True)
class ResearchLoop:
    data: dict[str, Bars]
    registry: Registry
    config: ResearchConfig
    proposer: Proposer = field(default_factory=HeuristicProposer)
    backtest_config: BacktestConfig = field(default_factory=BacktestConfig)
    regime_labels: dict[str, list[str]] | None = None
    data_by_timeframe: dict[str, dict[str, Bars]] | None = None
    """Bars per allowed granularity, for campaigns that let the proposer choose.
    ``data`` stays the default granularity's set, so a single-timeframe run
    needs nothing more."""
    regime_labels_by_timeframe: dict[str, dict[str, list[str]]] | None = None
    membership: PointInTimeUniverse | None = None
    """Which names were index members on each session. ``None`` means the whole
    of ``data`` is tradable throughout -- correct for a synthetic run, and the
    survivorship bias itself for a universe drawn from today's constituents."""
    steps: list[ResearchStep] = field(default_factory=list)

    def run(self) -> list[ResearchStep]:
        """Run the configured number of iterations. Never raises for a bad idea."""
        for i in range(1, self.config.iterations + 1):
            self.steps.append(self._iterate(i))
        return self.steps

    def best(self) -> ResearchStep | None:
        scored = [s for s in self.steps if s.outcome and not s.error]
        return max(scored, key=lambda s: s.score) if scored else None

    def summary(self) -> str:
        lines = [str(step) for step in self.steps]
        best = self.best()
        if best:
            lines.append("")
            lines.append(f"best: {best.spec.name if best.spec else '?'} at {best.score:.0f}/100")
        return "\n".join(lines)

    # ----------------------------------------------------------------- #

    def _data_for(self, timeframe: str) -> dict[str, Bars]:
        """The bars a candidate on ``timeframe`` is evaluated against."""
        if self.data_by_timeframe is None:
            return self.data
        return self.data_by_timeframe.get(timeframe, self.data)

    def _labels_for(self, timeframe: str) -> dict[str, list[str]] | None:
        """Regime labels are per bar, so a label set fits one granularity only.
        A candidate off the default gets ``None`` and the pipeline estimates
        labels from the bars it is handed."""
        if self.regime_labels_by_timeframe is not None:
            return self.regime_labels_by_timeframe.get(timeframe)
        return self.regime_labels if timeframe == self.config.timeframe else None

    def _windows_for(self, timeframe: str) -> tuple[int, int]:
        """Walk-forward geometry for a candidate on ``timeframe``."""
        default = window_bars(timeframe)
        train = self.config.train_bars if self.config.train_bars is not None else default[0]
        test = self.config.test_bars if self.config.test_bars is not None else default[1]
        return train, test

    def _iterate(self, iteration: int) -> ResearchStep:
        memory = self.registry.memory(self.config.memory_depth)
        parent = self._parent_for(iteration)

        try:
            proposal = self.proposer.propose(
                symbols=self.config.symbols,
                timeframe=self.config.timeframe,
                memory=memory,
                parent=parent,
                timeframes=self.config.timeframes,
            )
        except Exception as exc:  # a proposer failure must not end the run
            placeholder = Proposal(fields={"name": f"proposal_failed_{iteration}"}, source="error")
            step = ResearchStep(iteration=iteration, proposal=placeholder, error=str(exc))
            self._record_failure(step)
            return step

        step = ResearchStep(iteration=iteration, proposal=proposal)

        try:
            spec = build_spec(
                proposal,
                self.config.symbols,
                self.config.timeframe,
                allowed_timeframes=self.config.timeframes,
                risk_per_trade=self.config.risk_per_trade,
                max_positions=self.config.max_positions,
                parent_fingerprint=(parent or {}).get("fingerprint"),
            )
        except (ValueError, KeyError, TypeError) as exc:
            # An unparseable proposal is a real result: it says the prompt or the
            # model needs work, and the record is what makes that visible.
            step.error = f"proposal did not compile: {exc}"
            self._record_failure(step)
            return step

        step.spec = spec

        # A rule that compiles and fires on nothing is the cheapest failure to
        # detect and the most wasteful one to accept. In a 40-hypothesis
        # campaign on real daily bars, 15 proposals landed here. Send it back
        # once with the reason before spending the pipeline on it.
        spec, first_attempt = self._repair_dead_rule(spec, proposal)
        if first_attempt is not None:
            # The dead attempt is on the record. Hiding it would understate the
            # multiple-comparisons denominator, which is the one number the
            # overfitting detector cannot do without.
            self._record_dead_attempt(first_attempt, step)
            step.spec = spec

        if self.registry.has_tried(spec.fingerprint()):
            step.error = "identical strategy already evaluated; skipping the backtest"
            self._record_failure(step)
            return step

        train_bars, test_bars = self._windows_for(spec.universe.timeframe)
        step.outcome = evaluate_candidate(
            spec,
            self._data_for(spec.universe.timeframe),
            regime_labels=self._labels_for(spec.universe.timeframe),
            membership=self.membership,
            config=self.backtest_config,
            train_bars=train_bars,
            test_bars=test_bars,
            registry=self.registry,
            llm_model=proposal.model,
            prompt_hash=proposal.prompt_hash,
            dataset_version=self.config.dataset_version,
        )
        if step.outcome.verdict in ("ACCEPT", "PAPER") and self.config.save_accepted_to:
            save_file(spec, f"{self.config.save_accepted_to}/{saved_filename(spec)}")
        return step

    def _repair_dead_rule(
        self, spec: StrategySpec, proposal: Proposal
    ) -> tuple[StrategySpec, StrategySpec | None]:
        """One repair turn for a rule that cannot fire.

        Returns ``(spec_to_evaluate, dead_first_attempt_or_None)``. One turn,
        not five: a model that cannot fix an unsatisfiable condition after being
        shown it is not about to produce a good hypothesis on the third try.

        A proposer with no ``repair`` -- the offline heuristic one, which has no
        model to ask -- is left exactly as it was.
        """
        report = self._satisfiable(spec)
        if report is None:
            return spec, None
        repair = getattr(self.proposer, "repair", None)
        if not callable(repair):
            return spec, None
        try:
            repaired = repair(proposal=proposal, problems=report)
            candidate = build_spec(
                repaired,
                self.config.symbols,
                self.config.timeframe,
                allowed_timeframes=self.config.timeframes,
                risk_per_trade=self.config.risk_per_trade,
                max_positions=self.config.max_positions,
            )
        except Exception:
            # A failed repair is not a failed run. Evaluate the original and let
            # the pipeline reject it with the reason it deserves.
            return spec, None
        if candidate.fingerprint() == spec.fingerprint():
            # The model handed back the same rule. That is a failed repair, not
            # a second attempt: recording it twice would inflate the search-cost
            # denominator with an experiment nobody ran.
            return spec, None
        return candidate, spec

    def _satisfiable(self, spec: StrategySpec) -> list[str] | None:
        """The validator's complaints about ``spec``, or ``None`` if it is fine."""
        data = self._data_for(spec.universe.timeframe)
        traded = {s: data[s] for s in spec.universe.symbols if s in data}
        if not traded:
            return None
        primary = next(iter(traded.values()))
        report = validate_against(spec, primary, CrossSection(traded))
        return None if report.ok else list(report.errors)

    def _record_dead_attempt(self, dead: StrategySpec, step: ResearchStep) -> None:
        """Write down a rule that never fired, before its replacement is tried."""
        self.registry.upsert_strategy(dead)
        self.registry.record_experiment(
            ExperimentRecord(
                fingerprint=dead.fingerprint(),
                strategy_name=dead.name,
                hypothesis=dead.hypothesis,
                symbols=tuple(self.config.symbols),
                timeframe=dead.universe.timeframe,
                data_start="",
                data_end="",
                dataset_version=self.config.dataset_version,
                verdict="REJECT",
                backtests_run=0,
                llm_model=step.proposal.model,
                prompt_hash=step.proposal.prompt_hash,
                error="never fires; sent back to the proposer for one repair turn",
            )
        )

    def _parent_for(self, iteration: int) -> dict[str, Any] | None:
        """The strategy to refine, on refinement iterations."""
        every = self.config.mutate_best_every
        if every <= 0 or iteration % every != 0:
            return None
        best = self.best()
        if best is None or best.spec is None:
            return None
        fields = spec_to_proposal_fields(best.spec)
        fields["fingerprint"] = best.spec.fingerprint()
        fields["achieved_score"] = best.score
        return fields

    def _record_failure(self, step: ResearchStep) -> None:
        """Failures are experiments too, and they count toward the search cost."""
        spec = step.spec
        self.registry.record_experiment(
            ExperimentRecord(
                fingerprint=spec.fingerprint() if spec else "n/a",
                strategy_name=(
                    spec.name if spec else str(step.proposal.fields.get("name", "unknown"))
                ),
                hypothesis=str(step.proposal.fields.get("hypothesis", "")),
                symbols=tuple(self.config.symbols),
                # The granularity the candidate actually asked for, not the run
                # default: a failed 1h proposal filed under 1D would tell the
                # next prompt the wrong thing was tried.
                timeframe=(
                    spec.universe.timeframe
                    if spec
                    else str(
                        step.proposal.fields.get("timeframe") or self.config.timeframe
                    )
                ),
                data_start="",
                data_end="",
                dataset_version=self.config.dataset_version,
                verdict="ERROR",
                backtests_run=0,
                llm_model=step.proposal.model,
                prompt_hash=step.proposal.prompt_hash,
                error=step.error,
            )
        )
