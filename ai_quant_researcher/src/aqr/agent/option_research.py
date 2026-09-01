"""The option research loop — specs/10-options-research.md D8.

    propose -> compile -> run -> evaluate -> record -> remember -> propose

The same machine [`research.py`](research.py) runs, with one difference that is
the whole reason this module exists: **the search is counted separately, and
what it costs is reported rather than capped away.**

specs/10 D8 measured 71 non-overlapping 28-DTE cycles in the 5.55-year research
window and argued for a cap of 20 hypotheses, on the grounds that the best of
many draws from a null distribution looks like an edge at that sample size. The
premise is right; the conclusion that a *cap* is the defence is not, and the
first real campaign showed why. At twenty trials the highest-scoring rule --
97/100, every robustness component maxed, beating buy-and-hold on Sharpe -- was
rejected because its 0.67 Sharpe deflated to **-0.28** once the search that
found it was paid for. The deflation term did that, not the cap.

So :data:`OPTION_SEARCH_BUDGET` is a guardrail against a runaway loop rather
than a statistical control. What keeps a wide search honest is that its width is
priced into every verdict and recorded in every artefact.

**The denominator is the campaign's, not the database's.** One search is one
campaign, and a verdict is deflated against the trials *that search* took. The
argument: deflation asks how many draws bought this maximum, and a sweep
exploring iron condors in March did not buy the maximum of a sweep exploring
calendar effects in January; charging the second for the first is naive
Bonferroni applied across a research programme rather than across an experiment,
and it ends with a programme unable to conclude anything. The argument against is
real and is not hidden: somebody who runs ten campaigns and reports the best rule
from the tenth has looked at all ten. Nothing here stops that and nothing here
conceals it — ``aqr campaigns`` lists every search that has ever run, and every
book carries the campaign count, the campaign's own denominator and the all-time
figure side by side.

The three properties ``research.py`` names hold here unchanged: the proposer
never touches the verdict, a malformed proposal costs one iteration rather than
the run, and nothing is promoted past PAPER.

One thing this loop does that the equity one does not: it hands the proposer the
**measured range of every feature** (:meth:`OptionResearchLoop.span`). That is
not a nicety. A campaign without it lost seven of twenty slots to conditions no
session could satisfy — ``term_slope() > 5`` against a maximum of 0.052 — and
five more to repair turns that made the same mistake again.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from aqr.agent.option_proposer import (
    OptionProposer,
    TemplateOptionProposer,
    build_option_spec,
    catalogue_spans,
    option_spec_to_proposal_fields,
)
from aqr.agent.proposer import Proposal
from aqr.features.engine import FeatureKey
from aqr.option_pipeline import OptionResearchOutcome, evaluate_option_candidate
from aqr.options.engine import OptionBacktestConfig
from aqr.options.features import OptionFeatureFrame, feature_span
from aqr.options.run import OptionMarket
from aqr.options.spec import OptionSpec, dumps_option_spec
from aqr.registry.db import OPTION, ExperimentRecord, Registry
from aqr.seal import current as current_seal

__all__ = [
    "OPTION_SEARCH_BUDGET",
    "WIDE_SEARCH_WARNING_AT",
    "OptionResearchConfig",
    "OptionResearchLoop",
    "OptionResearchStep",
    "option_saved_filename",
]

OPTION_SEARCH_BUDGET = 1000
"""The ceiling on one campaign, in distinct hypotheses.

**specs/10 D8 argues for 20, and this is 1000.** The divergence is deliberate
and it is worth being precise about what it does and does not give up. D8's
reasoning is that the research window holds about 71 non-overlapping 28-DTE
cycles, so a wide search produces a winner indistinguishable from the luckiest
draw. That reasoning is correct and nothing here contradicts it. What it does
not establish is that a *cap* is the defence, because the defence was never the
cap: it is the deflation term, and the deflation term already scales with the
trial count. Measured on the first real campaign, at twenty trials, the
best-scoring rule's Sharpe of 0.67 deflated to **-0.28** -- the machinery priced
the search and rejected the rule without the cap being involved at all. At two
hundred trials it prices it harder, and at a thousand harder still.

So the cap is a guardrail against an accident (a loop left running, a script
with a typo in its iteration count), not the statistical control. The control is
that every verdict carries ``sharpe_inflation``, every book carries the
campaign's denominator *and* the all-time count, and neither can be spent
without being recorded. A wide search is allowed to be wide; it is not allowed
to be quiet.

The honest cost of raising it: a search this wide will find something that
looks good, and the deflated number is then the only thing standing between
that and a promotion. Read it.
"""

WIDE_SEARCH_WARNING_AT = 50
"""Where the campaign starts saying out loud what the search is costing.

Not a gate and not a prompt -- a line of output. Past this many trials the
deflation term is large enough that a reader who has not thought about it will
misread a high score, and the cheapest place to make that hard is next to the
score."""


_DUPLICATE_RETRIES = 3
"""How many times to ask again when a proposal repeats a rule already evaluated.

Three, not one and not ten. A proposal is cheap next to a backtest, so retrying
is nearly free; but a proposer that has offered the same rule four times is
either exhausted (the offline library) or not reading its memory (a model), and
neither is fixed by a fifth ask."""

_ALREADY_TRIED = (
    "That rule has already been evaluated in this database: {name}, a "
    "{structure} with entry `{entry}`. Its result is already known and running "
    "it again would produce the identical numbers. Propose something "
    "different -- and prefer a different STRUCTURE, EXPIRY or ANCHOR DELTA "
    "rather than another entry condition on the same structure, which is the "
    "axis every campaign so far has over-explored."
)

_campaigns_started = 0


def _next_campaign_id() -> str:
    """``run-<seal>-<n>``, unique per loop and traceable to its process."""
    global _campaigns_started
    _campaigns_started += 1
    return f"run-{current_seal().run_id[:8]}-{_campaigns_started}"


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
    memory_depth: int = 40
    """How many past experiments the proposer is shown.

    Raised from the equity loop's 20 because an option campaign can now run to
    hundreds: a model shown the last twenty of two hundred hypotheses has no way
    to avoid re-proposing the hundred and eightieth. Repeats are caught on the
    fingerprint and cost no budget, but they do cost an iteration and an API
    call."""
    mutate_best_every: int = 4
    """Every Nth iteration, refine the best rule so far instead of proposing a
    new mechanism. Pure exploration never converges; pure refinement never
    discovers anything."""
    dataset_version: str = "options-cache"
    save_accepted_to: str | None = "strategies/options"
    campaign: str = ""
    """The name this search is recorded and deflated under.

    One search is one campaign, and the multiple-comparisons denominator is
    counted within it: a search exploring iron condors today is not charged for
    a search that explored calendar effects last month. Empty means "generate
    one", which is what the CLI does.

    Naming one explicitly is how a campaign gets *resumed*: pass the same name
    twice and the second run continues the first's denominator instead of
    starting a fresh one. That is the honest option when a run was interrupted,
    and the dishonest one when it is used to launder a second look, which is why
    ``aqr campaigns`` lists every search that has ever run."""

    def __post_init__(self) -> None:
        if self.iterations < 1:
            raise ValueError("iterations must be >= 1")
        if self.iterations > OPTION_SEARCH_BUDGET:
            raise ValueError(
                f"iterations={self.iterations} exceeds the option search ceiling of "
                f"{OPTION_SEARCH_BUDGET}. That ceiling is a guardrail against a "
                "runaway loop, not a statistical control -- the control is the "
                "deflation term, which scales with the trial count and is reported "
                "with every verdict."
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
    _frame: OptionFeatureFrame | None = field(default=None, init=False, repr=False)
    _campaign: str = field(default="", init=False, repr=False)

    @property
    def campaign(self) -> str:
        """This search's identity, generated on first use and then fixed.

        Read once and cached: a campaign that changed its name halfway through
        would split its own denominator, which is the one failure this whole
        mechanism cannot have.

        The generated form is ``run-<seal>-<n>``: the process seal's ``run_id``
        so a campaign can be traced to the process that ran it and lines up with
        the ancestry-taint grouping, plus an ordinal because **one search is one
        campaign and a process may run several**. Deriving it from the seal
        alone was the first attempt and it was wrong -- a script running three
        sweeps in one process would have deflated all three against one
        denominator, which is exactly the merging this mechanism exists to
        prevent, arriving from the other direction.
        """
        if not self._campaign:
            self._campaign = self.config.campaign or _next_campaign_id()
        return self._campaign

    def span(self, key: FeatureKey) -> tuple[float, float] | None:
        """What ``key`` actually ranges over on this campaign's own market.

        Bound to the market the loop was handed, which for a search is always
        the research market -- ``research_option_market`` is what the CLI
        builds, and ``aqr.seal`` is what makes that checkable rather than a
        convention. Handing a proposer spans measured on the sealed years would
        put embargoed information into a prompt.

        The frame is built once and caches per key, so the twentieth proposal
        pays nothing for the range check the first one paid for.
        """
        if self._frame is None:
            self._frame = OptionFeatureFrame(
                bars=self.market.underlying,
                chain=self.market.chain,
                volatility=self.market.volatility,
            )
        return feature_span(self._frame, key)

    def _catalogue_spans(self) -> dict[str, tuple[float, float]]:
        """Measured once per campaign, then reused.

        ``self.span`` caches per key inside the frame, so the second call is
        free; the dict is rebuilt each iteration only because a campaign is
        allowed to be handed a different market between runs and a cached copy
        would then describe the wrong one.
        """
        return catalogue_spans(self.span)

    def run(self) -> list[OptionResearchStep]:
        """Run the configured iterations, or as many as the budget still allows.

        Never raises for a bad idea, and never spends past
        :data:`OPTION_SEARCH_BUDGET`. Stopping early is reported as a step with
        a reason rather than as a silent short list, because "the campaign ended
        at 14 of 20" and "the budget was already spent" are different facts and
        a reader of the log needs to be able to tell them apart.
        """
        for i in range(1, self.config.iterations + 1):
            spent = self.registry.distinct_hypotheses(
                family=OPTION, campaign=self.campaign
            )
            if spent >= OPTION_SEARCH_BUDGET:
                self.steps.append(
                    OptionResearchStep(
                        iteration=i,
                        proposal=Proposal(
                            fields={"name": "budget_exhausted"}, source="budget"
                        ),
                        error=(
                            f"this campaign has reached the ceiling: {spent} distinct "
                            f"hypotheses in {self.campaign}, against a ceiling of "
                            f"{OPTION_SEARCH_BUDGET}. That ceiling is a guardrail "
                            "against a runaway loop, not a statistical control. "
                            "Nothing was proposed."
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
                span=self.span,
                spans=self._catalogue_spans(),
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
            resolved = self._ask_for_something_new(proposal, spec, parent, memory)
            if resolved is None:
                # Not recorded, and that is the point. Nothing was evaluated, so
                # no draw was taken and this campaign's denominator must not
                # grow. Recording it would inflate the count a deflation is
                # computed against with rules the campaign never ran -- which is
                # exactly the merging a per-campaign denominator exists to stop,
                # arriving through the back door. The iteration is still spent
                # and the step still says so; the ledger already holds the
                # original evaluation, under the campaign that did run it.
                step.error = (
                    f"identical rule already evaluated ({spec.fingerprint()}); "
                    f"the proposer offered nothing new in "
                    f"{_DUPLICATE_RETRIES + 1} attempts, so this iteration is "
                    "spent and no experiment is recorded"
                )
                return step
            proposal, spec = resolved
            step.proposal, step.spec = proposal, spec

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

    def _ask_for_something_new(
        self,
        proposal: Proposal,
        spec: OptionSpec,
        parent: dict[str, Any] | None,
        memory: list[dict[str, Any]],
    ) -> tuple[Proposal, OptionSpec] | None:
        """Ask again when a proposal repeats a rule the ledger already holds.

        A duplicate is not a result -- the answer is already recorded, and
        re-running it would produce the identical numbers. Before this, it cost
        a whole iteration: a second campaign against the offline template
        library spent five of six that way. Bounded retries are cheap (a
        proposal, not a backtest) and turn most of those back into real work.

        ``None`` when the proposer keeps offering the same thing, which for the
        offline library means it is exhausted and for a model means the memory
        it was shown is not steering it.
        """
        seen = {spec.fingerprint()}
        for _ in range(_DUPLICATE_RETRIES):
            try:
                candidate = self.proposer.propose(
                    underlying=self.config.underlying,
                    memory=memory,
                    parent=parent,
                    instruction=_ALREADY_TRIED.format(
                        name=spec.name, entry=spec.entry, structure=spec.structure.type
                    ),
                    budget=(
                        self.registry.distinct_hypotheses(
                            family=OPTION, campaign=self.campaign
                        ),
                        OPTION_SEARCH_BUDGET,
                    ),
                    span=self.span,
                    spans=self._catalogue_spans(),
                )
                rebuilt = self._build(candidate, parent)
            except Exception:
                return None
            fingerprint = rebuilt.fingerprint()
            if fingerprint in seen:
                return None
            if not self.registry.has_tried(fingerprint):
                return candidate, rebuilt
            seen.add(fingerprint)
        return None

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
            campaign=self.campaign,
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
            repaired = repair(proposal=proposal, problems=problems, span=self.span)
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
                campaign=self.campaign,
                verdict="ERROR",
                backtests_run=0,
                llm_model=step.proposal.model,
                prompt_hash=step.proposal.prompt_hash,
                error=step.error,
            )
        )
