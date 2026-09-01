"""The option research loop — specs/10-options-research.md D8.

    propose -> compile -> run -> evaluate -> record -> remember -> propose

The same machine [`research.py`](research.py) runs, with one difference that is
the whole reason this module exists: **the search budget is a gate, not a
default.**

specs/10 D8 measured 71 non-overlapping 28-DTE cycles in the 5.55-year research
window. The equity campaign spent 414 hypotheses and deflated its Sharpe to 0.74
for it; 400 trials against 71 cycles produces a number with no information in
it, because the best of 400 draws from a null distribution looks exactly like an
edge at that sample size. So :data:`OPTION_SEARCH_BUDGET` is enforced against
the registry's own count of option hypotheses — not against this run's
``iterations``, which a second invocation would reset — and a campaign that
would cross it is refused rather than truncated silently.

The three properties ``research.py`` names hold here unchanged: the proposer
never touches the verdict, a malformed proposal costs one iteration rather than
the run, and nothing is promoted past PAPER.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from aqr.agent.option_proposer import (
    OptionProposer,
    TemplateOptionProposer,
    build_option_spec,
    option_spec_to_proposal_fields,
)
from aqr.agent.proposer import Proposal
from aqr.option_pipeline import OptionResearchOutcome, evaluate_option_candidate
from aqr.options.engine import OptionBacktestConfig
from aqr.options.run import OptionMarket
from aqr.options.spec import OptionSpec, dumps_option_spec
from aqr.registry.db import OPTION, ExperimentRecord, Registry

__all__ = [
    "OPTION_SEARCH_BUDGET",
    "OptionResearchConfig",
    "OptionResearchLoop",
    "OptionResearchStep",
    "option_saved_filename",
]

OPTION_SEARCH_BUDGET = 20
"""specs/10 D8's cap on the option search, and it is a hard gate.

Not a suggested default and not a per-run limit: the count that matters is how
many *distinct option hypotheses this registry has ever evaluated*, because the
multiple-comparisons problem does not reset when a process exits. Twenty against
71 independent cycles is already a Bonferroni factor of twenty on a sample that
small; forty would mean the winner has to clear a bar the data cannot resolve,
and four hundred would mean the exercise stopped being measurement.
"""


def option_saved_filename(spec: OptionSpec) -> str:
    """Where a promoted option rule is written. Name *and* fingerprint.

    The same lesson the equity side learned the hard way: models reuse names, and
    two different rules sharing one filename means the survivor on disk is
    whichever finished last.
    """
    return f"{spec.name}-{spec.fingerprint()}.yaml"


@dataclass(slots=True)
class OptionResearchConfig:
    underlying: str = "SPY"
    iterations: int = 8
    risk_per_trade: float = 0.02
    """A fraction of equity, against the structure's own maximum loss.

    2%, not D5's worked-example 1%, and the reason is measured rather than
    preferred. specs/10 D8a: at 1% of a $100,000 account the median
    ``put_credit_spread`` max loss of $892 leaves most sessions unable to afford
    a single contract, and 578 of 598 skips were affordability rather than the
    market — a search run there would reject every rule for a fact about the
    account. At 2% the same rule produces 57 independent cycles instead of 21,
    with nothing about the rule changed. The number is recorded with every
    verdict and the evaluator reports the affordability fraction on every run,
    so this is a stated choice rather than a hidden one.
    """
    max_concurrent: int = 3
    memory_depth: int = 20
    mutate_best_every: int = 4
    """Every Nth iteration, refine the best rule so far instead of proposing a
    new mechanism. Sparser than the equity loop's 3 because the budget is 20
    rather than 414: refinement iterations are the ones least likely to teach
    anything new, and here each one is 5% of the whole search."""
    dataset_version: str = "options-cache"
    save_accepted_to: str | None = "strategies/options"

    def __post_init__(self) -> None:
        if self.iterations < 1:
            raise ValueError("iterations must be >= 1")
        if self.iterations > OPTION_SEARCH_BUDGET:
            raise ValueError(
                f"iterations={self.iterations} exceeds the option search budget of "
                f"{OPTION_SEARCH_BUDGET} (specs/10 D8). The window holds about 71 "
                "independent 28-DTE cycles; a search wider than this produces a "
                "winner that cannot be distinguished from the luckiest draw."
            )


@dataclass(slots=True)
class OptionResearchStep:
    iteration: int
    proposal: Proposal
    spec: OptionSpec | None = None
    outcome: OptionResearchOutcome | None = None
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
        cycles = ""
        if self.outcome and self.outcome.walk_forward is not None:
            cycles = f", {self.outcome.walk_forward.oos_cycles} cycles"
        return f"[{self.iteration}] {name}: {self.verdict} ({self.score:.0f}/100{cycles})"


@dataclass(slots=True)
class OptionResearchLoop:
    market: OptionMarket
    registry: Registry
    config: OptionResearchConfig
    proposer: OptionProposer = field(default_factory=TemplateOptionProposer)
    backtest_config: OptionBacktestConfig = field(default_factory=OptionBacktestConfig)
    steps: list[OptionResearchStep] = field(default_factory=list)

    def run(self) -> list[OptionResearchStep]:
        """Run the configured iterations, or as many as the budget still allows.

        Never raises for a bad idea, and never spends past
        :data:`OPTION_SEARCH_BUDGET`. Stopping early is reported as a step with
        a reason rather than as a silent short list, because "the campaign ended
        at 14 of 20" and "the budget was already spent" are different facts and
        a reader of the log needs to be able to tell them apart.
        """
        for i in range(1, self.config.iterations + 1):
            spent = self.registry.distinct_hypotheses(family=OPTION)
            if spent >= OPTION_SEARCH_BUDGET:
                self.steps.append(
                    OptionResearchStep(
                        iteration=i,
                        proposal=Proposal(
                            fields={"name": "budget_exhausted"}, source="budget"
                        ),
                        error=(
                            f"the option search budget is spent: {spent} distinct "
                            f"hypotheses have been evaluated against a window holding "
                            f"about 71 independent cycles, and specs/10 D8 caps it at "
                            f"{OPTION_SEARCH_BUDGET}. Nothing was proposed."
                        ),
                    )
                )
                break
            self.steps.append(self._iterate(i, spent))
        return self.steps

    def best(self) -> OptionResearchStep | None:
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

    def _iterate(self, iteration: int, spent: int) -> OptionResearchStep:
        memory = self.registry.memory(self.config.memory_depth, family=OPTION)
        parent = self._parent_for(iteration)

        try:
            proposal = self.proposer.propose(
                underlying=self.config.underlying,
                memory=memory,
                parent=parent,
                budget=(spent, OPTION_SEARCH_BUDGET),
            )
        except Exception as exc:  # a proposer failure must not end the run
            placeholder = Proposal(fields={"name": f"proposal_failed_{iteration}"}, source="error")
            step = OptionResearchStep(iteration=iteration, proposal=placeholder, error=str(exc))
            self._record_failure(step)
            return step

        step = OptionResearchStep(iteration=iteration, proposal=proposal)
        try:
            spec = self._build(proposal, parent)
        except (ValueError, KeyError, TypeError) as exc:
            # An unusable proposal is a real result: it says the prompt or the
            # model needs work, and the record is what makes that visible.
            step.error = f"proposal did not compile: {exc}"
            self._record_failure(step)
            return step
        step.spec = spec

        if self.registry.has_tried(spec.fingerprint()):
            step.error = "identical rule already evaluated; skipping the backtest"
            self._record_failure(step)
            return step

        outcome = self._evaluate(spec, proposal)

        # A rule that compiled and opened nothing is the cheapest failure to
        # detect and the most wasteful to accept -- one of twenty iterations
        # spent on a rule that produced no evidence in either direction. Send it
        # back once with the census attached, which is what tells the model
        # whether it wrote an unsatisfiable condition or asked for a structure
        # the ladder never offered.
        if outcome.rejected_early and not outcome.result_has_trades:
            repaired = self._repair(proposal, outcome, parent)
            if repaired is not None:
                step.spec, repaired_proposal = repaired
                step.proposal = repaired_proposal
                outcome = self._evaluate(step.spec, repaired_proposal)

        step.outcome = outcome
        if outcome.verdict in ("ACCEPT", "PAPER") and self.config.save_accepted_to:
            self._save(step.spec or spec)
        return step

    def _build(self, proposal: Proposal, parent: dict[str, Any] | None) -> OptionSpec:
        return build_option_spec(
            proposal,
            self.config.underlying,
            risk_per_trade=self.config.risk_per_trade,
            max_concurrent=self.config.max_concurrent,
            parent_fingerprint=(parent or {}).get("fingerprint"),
        )

    def _evaluate(self, spec: OptionSpec, proposal: Proposal) -> OptionResearchOutcome:
        return evaluate_option_candidate(
            spec,
            self.market,
            config=self.backtest_config,
            registry=self.registry,
            llm_model=proposal.model,
            prompt_hash=proposal.prompt_hash,
            dataset_version=self.config.dataset_version,
        )

    def _repair(
        self,
        proposal: Proposal,
        outcome: OptionResearchOutcome,
        parent: dict[str, Any] | None,
    ) -> tuple[OptionSpec, Proposal] | None:
        """One repair turn for a rule that opened nothing. ``None`` if it failed.

        One turn, not five: a model that cannot fix an unsatisfiable condition
        after being shown the skip census is not about to produce a good
        hypothesis on the third try, and against a budget of twenty the next
        iteration is a cheaper place to spend the attempt.

        A proposer with no ``repair`` — the offline template one, which has no
        model to ask — is left exactly as it was. The dead attempt stays on the
        record either way: the pipeline already wrote it, and hiding it would
        understate the multiple-comparisons denominator, which is the one number
        the overfitting detector cannot do without.
        """
        repair = getattr(self.proposer, "repair", None)
        if not callable(repair):
            return None
        problems = [outcome.rejected_early or "the rule opened no positions"]
        try:
            repaired = repair(proposal=proposal, problems=problems)
            candidate = self._build(repaired, parent)
        except Exception:
            # A failed repair is not a failed run. The original's rejection
            # stands, with the reason it earned.
            return None
        if candidate.fingerprint() == outcome.spec.fingerprint():
            # The model handed back the same rule. That is a failed repair, not
            # a second attempt.
            return None
        return candidate, repaired

    def _parent_for(self, iteration: int) -> dict[str, Any] | None:
        """The rule to refine, on refinement iterations."""
        every = self.config.mutate_best_every
        if every <= 0 or iteration % every != 0:
            return None
        best = self.best()
        if best is None or best.spec is None:
            return None
        fields = option_spec_to_proposal_fields(best.spec)
        fields["fingerprint"] = best.spec.fingerprint()
        fields["achieved_score"] = best.score
        return fields

    def _save(self, spec: OptionSpec) -> None:
        """Write a surviving rule to disk, beside the equity ones.

        Its own subdirectory. The two file formats are not interchangeable —
        ``dsl/loader.py`` would read an option rule as a spec with no entry
        condition rather than refusing it — so keeping them in one directory
        would make ``aqr backtest strategies/*.yaml`` a loaded gun.
        """
        from pathlib import Path

        target = Path(self.config.save_accepted_to or "strategies/options")
        target.mkdir(parents=True, exist_ok=True)
        (target / option_saved_filename(spec)).write_text(
            dumps_option_spec(spec), encoding="utf-8"
        )

    def _record_failure(self, step: OptionResearchStep) -> None:
        """Failures are experiments too, and they count toward the search cost."""
        spec = step.spec
        if spec is not None:
            self.registry.upsert_option_strategy(spec)
        self.registry.record_experiment(
            ExperimentRecord(
                fingerprint=spec.fingerprint() if spec else "n/a",
                strategy_name=(
                    spec.name if spec else str(step.proposal.fields.get("name", "unknown"))
                ),
                hypothesis=str(step.proposal.fields.get("hypothesis", "")),
                symbols=(self.config.underlying,),
                timeframe="option_chain",
                data_start="",
                data_end="",
                dataset_version=self.config.dataset_version,
                family=OPTION,
                verdict="ERROR",
                backtests_run=0,
                llm_model=step.proposal.model,
                prompt_hash=step.proposal.prompt_hash,
                error=step.error,
            )
        )
