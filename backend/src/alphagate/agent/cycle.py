"""One pass of the agent — specs/05 D1.

```
 1. perceive    core engines over fresh bars     → MarketRead      pure
 2. screen      deterministic pre-filter         → Setup | None    pure
 3. enumerate   build candidate structures       → Candidates      pure
 4. propose     the LLM picks one, or none       → Choice          impure
 5. size        risk budget → quantity           → TradeProposal   pure
 6. gate        risk.evaluate                    → Verdict         pure
 7. submit      execution.submit, if Approved    → Submission      impure
 8. record      every step, always               → CycleRecord     impure
```

Steps 1–3 happen before `run_cycle` is called — they need market data, which is
the caller's business — so this function starts at the point where all the
inputs are values. That keeps the orchestration itself pure enough to test
without a network, and it is why every test in `tests/agent/` runs offline.

**The cycle always produces a record.** There is no early `return None`, no
exception path that escapes without one, and no branch that logs instead. A
journal containing only trades cannot answer "why didn't it trade at 14:30?",
and that is the question the whole project is arranged to answer.

`as_of` is an argument. Nothing here reads a clock, so a replayed cycle and a
live cycle differ in the value of one parameter and nothing else.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from alphagate.agent.book import HeldPosition
from alphagate.agent.exits import DEFAULT_EXIT_POLICY, ExitPolicy, evaluate_exit
from alphagate.agent.model import Candidate, Choice, MarketRead, ModelCall, Setup, Stage
from alphagate.agent.proposer import DEFAULT_PROPOSER, Proposer
from alphagate.core.errors import InvariantViolation
from alphagate.execution import (
    ExecutionError,
    McpSession,
    PartialFillBreach,
    Submission,
    submit,
)
from alphagate.options import StructureRisk
from alphagate.risk import (
    Approved,
    Intent,
    PortfolioSnapshot,
    RiskLimits,
    TradeProposal,
    Verdict,
    Vetoed,
    evaluate,
)

__all__ = ["CycleRecord", "cycle_id_for", "run_cycle", "run_exit_cycle"]


def cycle_id_for(as_of: datetime, underlying: str, sequence: int) -> str:
    """`YYYY-MM-DD-TICKER-NNN` — specs/06 D2.

    Deterministic and collision-free across underlyings and days, which is what
    lets `RecordedProposer` find the right choice on replay.
    """
    return f"{as_of.date().isoformat()}-{underlying}-{sequence:03d}"


@dataclass(frozen=True, slots=True)
class CycleRecord:
    """Everything one pass did, whether or not it traded — specs/06 D2.

    Written once, at the end of the cycle. Later facts (a fill hours after
    submission, realised P&L on close) arrive as separate amendment lines keyed
    by `cycle_id`, never as an edit to this one: the original decision stays
    exactly as it was made, with no hindsight leaking backwards (specs/06 D3).
    """

    cycle_id: str
    as_of: datetime
    stage: Stage
    read: MarketRead
    setup: Setup | None
    candidates: tuple[Candidate, ...]
    choice: Choice | None
    call: ModelCall | None
    proposal: TradeProposal | None
    verdict: Verdict | None
    submission: Submission | None
    note: str = ""
    """Why the cycle ended where it did, in one line, for the dashboard."""

    @property
    def traded(self) -> bool:
        return self.stage.traded

    @property
    def veto_reasons(self) -> tuple[str, ...]:
        if isinstance(self.verdict, Vetoed):
            return tuple(reason.check for reason in self.verdict.reasons)
        return ()


def run_cycle(
    *,
    read: MarketRead,
    setup: Setup | None,
    candidates: Sequence[Candidate],
    portfolio: PortfolioSnapshot,
    limits: RiskLimits,
    as_of: datetime,
    mcp: McpSession | None = None,
    proposer: Proposer = DEFAULT_PROPOSER,
    sequence: int = 0,
    intent: Intent = Intent.OPEN,
    screen_reason: str = "",
) -> CycleRecord:
    """Steps 4 through 8. Returns a record on every path.

    `mcp=None` is dry-run: the Gate still runs and the verdict is still recorded,
    but nothing is submitted. That is the mode the pre-open check uses, and it is
    a first-class outcome rather than a flag threaded through the trading path.

    `screen_reason` is the screen's own explanation for a `None` setup —
    specs/07 D1. It is threaded through rather than re-derived here because
    only the screen that actually ran knows whether the `None` meant "the rule
    said no" or "a feature was unmeasured", and a `BookScreen`'s `explain`
    carries exactly that distinction (`agent/screen.py`). Left empty, the
    `NO_SETUP` note falls back to the generic text — which is what every caller
    that has not been taught to supply a reason still gets, unchanged.
    """
    cycle_id = cycle_id_for(as_of, str(read.underlying), sequence)
    menu = tuple(candidates)

    def record(
        stage: Stage,
        *,
        note: str,
        choice: Choice | None = None,
        call: ModelCall | None = None,
        proposal: TradeProposal | None = None,
        verdict: Verdict | None = None,
        submission: Submission | None = None,
    ) -> CycleRecord:
        return CycleRecord(
            cycle_id=cycle_id,
            as_of=as_of,
            stage=stage,
            read=read,
            setup=setup,
            candidates=menu,
            choice=choice,
            call=call,
            proposal=proposal,
            verdict=verdict,
            submission=submission,
            note=note,
        )

    # -- 2. screen ------------------------------------------------------ #
    if setup is None:
        return record(
            Stage.NO_SETUP, note=screen_reason or "the screen found nothing to trade"
        )

    # -- 3. enumerate --------------------------------------------------- #
    if not menu:
        return record(
            Stage.NO_CANDIDATES,
            note="no structure survived pricing, freshness, spread, DTE and sizing",
        )

    # -- 4. propose (impure) -------------------------------------------- #
    proposal_result = proposer.propose(read, menu, cycle_id=cycle_id)
    choice, call = proposal_result.choice, proposal_result.call
    chosen = choice.resolve(menu)
    if chosen is None:
        return record(
            Stage.DECLINED,
            note=_decline_note(choice, call, len(menu)),
            choice=choice,
            call=call,
        )

    # -- 5. size (pure; the quantity was decided before the model saw it) - #
    trade = TradeProposal(
        structure=chosen.structure,
        risk=chosen.risk,
        quantity=chosen.quantity,
        intent=intent,
        rationale=choice.rationale,
        proposed_by=call.model,
        proposal_id=cycle_id,
        risk_as_of=read.as_of,
    )

    # -- 6. gate (pure) -------------------------------------------------- #
    verdict = evaluate(trade, portfolio, limits, as_of)
    if isinstance(verdict, Vetoed):
        return record(
            Stage.VETOED,
            note="gate vetoed: " + ", ".join(r.check for r in verdict.reasons),
            choice=choice,
            call=call,
            proposal=trade,
            verdict=verdict,
        )
    assert isinstance(verdict, Approved)  # noqa: S101 - the union has two arms

    # -- 7. submit (impure) ---------------------------------------------- #
    if mcp is None:
        return record(
            Stage.DRY_RUN,
            note="approved, not submitted: no session (dry run)",
            choice=choice,
            call=call,
            proposal=trade,
            verdict=verdict,
        )

    try:
        submission = submit(verdict.order, mcp)
    except PartialFillBreach as breach:
        # specs/04 D5. A spread half filled is a naked leg. Latch the kill
        # switch and stop; do not try to leg out.
        return record(
            Stage.BREACHED,
            note=str(breach),
            choice=choice,
            call=call,
            proposal=trade,
            verdict=verdict,
            submission=breach.submission,
        )
    except (ExecutionError, InvariantViolation) as failure:
        return record(
            Stage.REJECTED,
            note=f"submission failed: {type(failure).__name__}: {failure}",
            choice=choice,
            call=call,
            proposal=trade,
            verdict=verdict,
        )

    stage = _stage_of(submission)
    return record(
        stage,
        note=submission.reason or f"order {submission.raw_status}",
        choice=choice,
        call=call,
        proposal=trade,
        verdict=verdict,
        submission=submission,
    )


def _stage_of(submission: Submission) -> Stage:
    if submission.is_rejected:
        return Stage.REJECTED
    if submission.status.is_filled:
        return Stage.FILLED
    return Stage.SUBMITTED


def _decline_note(choice: Choice, call: ModelCall, menu_size: int) -> str:
    """Say *why* it declined, distinguishing the three different reasons.

    A model that chose nothing, a model that named something that does not
    exist, and a model that never answered are three different situations, and
    a journal that renders all of them as "declined" cannot tell you which of
    your components is broken.
    """
    if call.error:
        return f"declined ({call.error})"
    if choice.candidate_index is not None:
        return f"declined: index {choice.candidate_index} outside the menu of {menu_size}"
    return "declined by the model"


def run_exit_cycle(
    *,
    held: HeldPosition,
    current: StructureRisk,
    read: MarketRead,
    portfolio: PortfolioSnapshot,
    limits: RiskLimits,
    as_of: datetime,
    mcp: McpSession | None = None,
    sequence: int = 0,
    policy: ExitPolicy = DEFAULT_EXIT_POLICY,
) -> CycleRecord | None:
    """One exit decision, and the order that follows from it — specs/05 D8.

    Returns `None` when the position should be held. That is the majority case
    and it is deliberately *not* journalled: a line per open position per slot
    would be twenty-six lines a day per position saying "still fine", and a
    journal nobody can read is a journal nobody reads. What is journalled is
    every actual close, with the rule that fired and the numbers behind it.

    **No model is consulted and none can be.** `evaluate_exit` is pure and
    deterministic (specs/07 D6), the structure to close is the one already held,
    and the quantity is the one already on. There is nothing here for a model to
    choose, which is the point: the decision to take a loss is exactly the
    decision you do not want a text generator improvising.

    **The Gate still runs.** A close is not exempt — it goes through `evaluate`
    like any other order, which is what makes `execution`'s "only a `GatedOrder`"
    rule hold on this path too. The Gate never blocks an exit (specs/03 D4), so
    this is a formality in the good case and a tripwire in the bad one: an exit
    that somehow arrived as an OPEN would be caught here rather than at the
    broker.
    """
    decision = evaluate_exit(
        held.position, current, held.entry_premium, as_of=as_of, policy=policy
    )
    if not decision.should_close:
        return None

    cycle_id = cycle_id_for(as_of, str(held.position.underlying), sequence)
    trade = TradeProposal(
        structure=held.position.structure,
        risk=current,
        quantity=held.position.quantity,
        intent=Intent.CLOSE,
        rationale=f"{decision.rule.value}: {decision.detail}",
        proposed_by="exit-policy",
        proposal_id=cycle_id,
        risk_as_of=as_of,
    )

    def record(
        stage: Stage, *, note: str, verdict: Verdict | None, submission: Submission | None
    ) -> CycleRecord:
        return CycleRecord(
            cycle_id=cycle_id,
            as_of=as_of,
            stage=stage,
            read=read,
            setup=None,
            candidates=(),
            choice=None,
            call=None,
            proposal=trade,
            verdict=verdict,
            submission=submission,
            note=note,
        )

    verdict = evaluate(trade, portfolio, limits, as_of)
    if isinstance(verdict, Vetoed):
        # specs/03 D4 says the Gate never blocks an exit, so this branch should
        # be unreachable. It is recorded rather than asserted because an exit we
        # believe was refused and a position we believe is closed are different
        # kinds of wrong, and only one of them is visible in the morning.
        return record(
            Stage.VETOED,
            note="gate vetoed an exit: " + ", ".join(r.check for r in verdict.reasons),
            verdict=verdict,
            submission=None,
        )
    assert isinstance(verdict, Approved)  # noqa: S101 - the union has two arms

    if mcp is None:
        return record(
            Stage.DRY_RUN,
            note=f"would close ({decision.rule.value}): {decision.detail}",
            verdict=verdict,
            submission=None,
        )

    try:
        submission = submit(verdict.order, mcp)
    except PartialFillBreach as breach:
        return record(
            Stage.BREACHED, note=str(breach), verdict=verdict, submission=breach.submission
        )
    except (ExecutionError, InvariantViolation) as failure:
        return record(
            Stage.REJECTED,
            note=f"exit submission failed: {type(failure).__name__}: {failure}",
            verdict=verdict,
            submission=None,
        )

    return record(
        _stage_of(submission),
        note=f"{decision.rule.value}: {decision.detail}",
        verdict=verdict,
        submission=submission,
    )


def portfolio_after(
    portfolio: PortfolioSnapshot, record: CycleRecord, risk: StructureRisk | None = None
) -> PortfolioSnapshot:
    """The snapshot the next cycle should see, given what this one did.

    Only the counters the Gate reads are advanced; positions are re-read from
    the broker rather than inferred, because a position we believe in but the
    broker does not is the worst of the two ways to be wrong.

    A breach latches the kill switch. That is specs/04 D5's "blocks new opens
    until a human clears it" and specs/03 D4's latch, meeting: the Gate is pure,
    so the latch has to ride in on the snapshot.
    """
    del risk
    fills = portfolio.fills_today + (1 if record.stage is Stage.FILLED else 0)
    return PortfolioSnapshot(
        equity=portfolio.equity,
        positions=portfolio.positions,
        drawdown_pct=portfolio.drawdown_pct,
        fills_today=fills,
        killswitch_tripped=portfolio.killswitch_tripped or record.stage is Stage.BREACHED,
    )
