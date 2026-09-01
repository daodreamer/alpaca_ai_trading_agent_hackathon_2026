"""Turning journal lines into what a dashboard renders. Pure.

Separate from `app.py` on purpose: every interesting decision about *what a
judge sees* is here, where it can be tested without starting a web server, and
`app.py` is left holding nothing but routing.

The shaping rules are all consequences of one thing specs/06 says the journal is
for — "a judge who opens any fill and reads the reasoning that produced it,
including the checks that nearly stopped it".

**Near-misses are computed, not just displayed.** A check that passed at 96% of
its limit is the most interesting line in the whole record: it is the evidence
that the Gate is load-bearing rather than decorative. So `headroom` is a number
on every measurable check, and the view sorts by it.

**The quiet cycles are first-class.** `NO_SETUP` and `DECLINED` are the majority
(specs/06 D2) and a dashboard that lists only fills answers the easy question.
The day view carries every cycle and the stage counts alongside them.

**Nothing here reaches the broker.** The dashboard reads a directory of text
files. It cannot place an order, cancel one, or move money, and that is a
property worth having in the thing you point at a screen during a demo.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from alphagate.journal import Journal, trust_report, untrusted_sources

__all__ = [
    "CATEGORY_DETAIL",
    "CATEGORY_LABELS",
    "CheckView",
    "CycleView",
    "DayView",
    "available_days",
    "day_records_with_category",
    "day_view",
    "to_cycle",
]

_TRADED = frozenset({"submitted", "filled"})
_QUIET = frozenset({"no_setup", "no_candidates", "declined"})
_GATE_VETO = frozenset({"vetoed", "breached"})
_BROKER_REJECTED = frozenset({"rejected"})
_MODEL_DECLINED = frozenset({"declined"})

CATEGORY_LABELS: dict[str, str] = {
    "traded": "traded",
    "approved_not_sent": "approved, not sent",
    "gate_veto": "stopped by the Risk Gate",
    "broker_rejected": "rejected by the broker",
    "model_declined": "model declined the menu",
    "undecidable": "entry undecidable — iv_rank unmeasured",
    "no_setup": "no qualifying setup",
    "no_candidates": "no priceable structure",
    "other": "unclassified",
}
"""One label per bucket a judge can tell apart — see `CycleView.category`."""

CATEGORY_DETAIL: dict[str, str] = {
    "traded": "the proposal passed every check and reached the broker",
    "approved_not_sent": (
        "the Gate approved this, but the run intentionally did not submit it — "
        "the pre-open dry run works this way"
    ),
    "gate_veto": "a proposal was built and the Risk Gate refused it — see the checks below",
    "broker_rejected": "the Gate approved this and the broker refused it",
    "model_declined": "the model was shown a menu of candidates and chose not to trade",
    "undecidable": (
        "iv_rank could not be measured yet, so the entry rule `iv_rank() < 15` cannot "
        "be decided — the agent stands aside rather than guessing at the number the "
        "rule was researched on (specs/07 D1)"
    ),
    "no_setup": "the market was read and did not meet the entry rule — quiet, not undecided",
    "no_candidates": (
        "the entry rule fired, but nothing survived pricing, freshness, spread, DTE "
        "and sizing — a menu problem, not an entry problem, and a different fact from "
        "either `no_setup` bucket above"
    ),
    "other": "",
}
"""A static, note-independent sentence per bucket — shown even when the journal's
own `note` is empty, so the distinction specs/06 D2 asks for holds regardless of
how much the writer said that cycle."""


@dataclass(frozen=True, slots=True)
class CheckView:
    """One Gate check, ready to render — specs/03 D3."""

    name: str
    passed: bool
    observed: str
    limit: str
    detail: str
    headroom: float | None
    """How close this check came to failing, 0–1, or `None` where it is not a
    measurable quantity.

    1.0 means it used none of its budget; 0.0 means it failed. The number exists
    because "passed" and "passed with 4% to spare" are very different facts
    about a risk system and only one of them is evidence that it works."""

    @property
    def is_near_miss(self) -> bool:
        return self.passed and self.headroom is not None and self.headroom < 0.15


@dataclass(frozen=True, slots=True)
class CycleView:
    """One journalled cycle, shaped for a template."""

    cycle_id: str
    as_of: str
    stage: str
    note: str
    underlying: str
    spot: str
    iv_rank: str
    trend: str
    rationale: str
    model: str
    prompt_version: str
    confidence: str
    candidate_count: int
    chosen_index: int | None
    structure: str
    quantity: int
    max_loss: str
    credit: str
    checks: tuple[CheckView, ...]
    outcome_status: str
    realised: str
    trust: str
    untrusted_paths: tuple[str, ...]
    raw: Mapping[str, Any]

    @property
    def traded(self) -> bool:
        return self.stage in _TRADED

    @property
    def is_quiet(self) -> bool:
        return self.stage in _QUIET

    @property
    def failed_checks(self) -> tuple[CheckView, ...]:
        return tuple(check for check in self.checks if not check.passed)

    @property
    def near_misses(self) -> tuple[CheckView, ...]:
        """Passed checks that nearly did not, tightest first."""
        near = [check for check in self.checks if check.is_near_miss]
        near.sort(key=lambda check: check.headroom if check.headroom is not None else 1.0)
        return tuple(near)

    @property
    def category(self) -> str:
        """Why nothing happened, in a bucket a judge can tell apart from the rest.

        This is the whole point of requirement 3: a cycle where the entry rule
        could not be decided (`iv_rank` unmeasured, specs/07 D1) is not the same
        fact as a market that was read and simply did not qualify, and neither
        is a Risk Gate veto. `stage` alone conflates the first two — `no_setup`
        covers both — so this property adds the one distinction `stage` cannot
        make, from a field the journal already carries.
        """
        if self.stage in ("filled", "submitted"):
            return "traded"
        if self.stage == "dry_run":
            return "approved_not_sent"
        if self.stage in _GATE_VETO:
            return "gate_veto"
        if self.stage in _BROKER_REJECTED:
            return "broker_rejected"
        if self.stage in _MODEL_DECLINED:
            return "model_declined"
        if self.stage == "no_candidates":
            # A different failure mode from `no_setup`, whatever `iv_rank` says:
            # the entry already fired (a `Setup` exists) and the menu-building
            # screens -- pricing, freshness, spread, DTE, sizing -- found nothing
            # to propose. Folding this into `no_setup` would erase exactly the
            # distinction requirement 3 asks for between "the entry could not
            # qualify" and "the entry qualified but nothing was tradeable".
            return "no_candidates"
        if self.stage == "no_setup":
            return "undecidable" if self.iv_rank == "unmeasured" else "no_setup"
        return "other"

    @property
    def category_label(self) -> str:
        return CATEGORY_LABELS.get(self.category, self.category)

    @property
    def category_detail(self) -> str:
        return CATEGORY_DETAIL.get(self.category, "")


@dataclass(frozen=True, slots=True)
class DayView:
    """One trading day."""

    day: date
    cycles: tuple[CycleView, ...]
    stage_counts: dict[str, int]
    duplicates: dict[str, int]
    orphans: tuple[str, ...]

    @property
    def fills(self) -> int:
        return self.stage_counts.get("filled", 0)

    @property
    def vetoes(self) -> int:
        return self.stage_counts.get("vetoed", 0)

    @property
    def quiet(self) -> int:
        return sum(self.stage_counts.get(stage, 0) for stage in _QUIET)

    @property
    def realised(self) -> Decimal:
        """Realised P&L for the day, from outcome amendments only.

        Only closed positions count. An open position's mark is not a result,
        and specs/07 D7 is explicit that unrealised and realised are reported
        separately rather than added together into a number that flatters."""
        total = Decimal(0)
        for cycle in self.cycles:
            value = _decimal_or_none(cycle.realised)
            if value is not None:
                total += value
        return total

    @property
    def has_warnings(self) -> bool:
        return bool(self.duplicates or self.orphans)


def available_days(journal: Journal) -> tuple[date, ...]:
    """Every day with a journal file, newest first."""
    if not journal.directory.is_dir():
        return ()
    days: list[date] = []
    for path in journal.directory.glob("*.jsonl"):
        try:
            days.append(date.fromisoformat(path.stem))
        except ValueError:
            continue
    return tuple(sorted(days, reverse=True))


def day_view(journal: Journal, day: date) -> DayView:
    records = journal.read(day)
    cycles = tuple(to_cycle(record) for record in records)
    counts: dict[str, int] = {}
    for cycle in cycles:
        counts[cycle.stage] = counts.get(cycle.stage, 0) + 1
    return DayView(
        day=day,
        cycles=cycles,
        stage_counts=counts,
        duplicates=journal.duplicate_cycles(day),
        orphans=journal.orphaned_amendments(day),
    )


def day_records_with_category(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """The day's raw records, each carrying its own decline classification.

    This is the one place `category` / `category_label` / `category_detail`
    reach JSON. `/api/day/{day}` used to hand back the line off disk unchanged;
    a browser had no way to learn why a cycle declined except by re-deriving
    `CycleView.category` in TypeScript, which is a second implementation of a
    classification with nothing keeping the two in agreement -- and a drift
    between them would not announce itself, it would just make the React page
    and the server-rendered fallback disagree about the one thing this panel
    exists to answer. Serializing the Python answer is the fix; the frontend
    now reads these three fields rather than recomputing them.

    Only non-equity records are stamped. `category` is specs/07's decline
    taxonomy -- built entirely out of options `Stage` values and `iv_rank` --
    and running it over an equity cycle's `no_trades` / `submitted` stages
    would not classify anything, it would mislabel it under a taxonomy that
    was never about that sleeve.

    Returns new dicts. The caller's records are never mutated, and every
    original key survives untouched -- this adds three keys, it does not
    reshape the record into `CycleView`'s field names.
    """
    enriched: list[dict[str, Any]] = []
    for record in records:
        if record.get("kind") == "equity":
            enriched.append(dict(record))
            continue
        view = to_cycle(record)
        enriched.append(
            {
                **record,
                "category": view.category,
                "category_label": view.category_label,
                "category_detail": view.category_detail,
            }
        )
    return enriched


def to_cycle(record: Mapping[str, Any]) -> CycleView:
    read = _mapping(record.get("read"))
    call = _mapping(record.get("call"))
    choice = _mapping(record.get("choice"))
    proposal = _mapping(record.get("proposal"))
    risk = _mapping(proposal.get("risk"))
    verdict = _mapping(record.get("verdict"))
    outcome = _mapping(record.get("outcome"))
    candidates = record.get("candidates")

    return CycleView(
        cycle_id=str(record.get("cycle_id", "")),
        as_of=_clock(record.get("as_of")),
        stage=str(record.get("stage", "unknown")),
        note=str(record.get("note", "")),
        underlying=str(read.get("underlying", "—")),
        spot=_number(read.get("spot")),
        iv_rank=_number(read.get("iv_rank"), unmeasured="unmeasured"),
        trend=_trend(read.get("trend")),
        rationale=str(choice.get("rationale", "")),
        model=str(call.get("model", "—")),
        prompt_version=str(call.get("prompt_version", "—")),
        confidence=_number(choice.get("self_reported_confidence"), unmeasured="—"),
        candidate_count=len(candidates) if isinstance(candidates, Sequence) else 0,
        chosen_index=_int_or_none(choice.get("candidate_index")),
        structure=_structure_label(proposal),
        quantity=_int_or_none(proposal.get("quantity")) or 0,
        max_loss=_number(risk.get("max_loss")),
        credit=_number(risk.get("net_premium")),
        checks=_checks(verdict),
        outcome_status=str(outcome.get("status", "")) or _implied_outcome(record),
        realised=_number(outcome.get("realised_pl"), unmeasured=""),
        trust=trust_report(record),
        untrusted_paths=tuple(source.path for source in untrusted_sources(record)),
        raw=record,
    )


# ------------------------------------------------------------------ #


def _checks(verdict: Mapping[str, Any]) -> tuple[CheckView, ...]:
    raw = verdict.get("checks")
    if not isinstance(raw, Sequence):
        return ()
    return tuple(_check(item) for item in raw if isinstance(item, Mapping))


def _check(item: Mapping[str, Any]) -> CheckView:
    observed = _decimal_or_none(item.get("observed"))
    limit = _decimal_or_none(item.get("limit"))
    return CheckView(
        name=str(item.get("name", "")),
        passed=bool(item.get("passed")),
        observed=_number(item.get("observed"), unmeasured="—"),
        limit=_number(item.get("limit"), unmeasured="—"),
        detail=str(item.get("detail", "")),
        headroom=_headroom(observed, limit, passed=bool(item.get("passed"))),
    )


def _headroom(observed: Decimal | None, limit: Decimal | None, *, passed: bool) -> float | None:
    """Fraction of the budget left. `None` where the check is not a magnitude.

    A failed check is zero by definition rather than a negative number: "how
    much room was left" has one answer once there was none, and a scale that
    runs below zero would make the sort order say a badly failed check nearly
    passed.
    """
    if observed is None or limit is None or limit == 0:
        return None
    if not passed:
        return 0.0
    try:
        used = abs(observed) / abs(limit)
    except (InvalidOperation, ZeroDivisionError):
        return None
    return max(0.0, min(1.0, 1.0 - float(used)))


def _structure_label(proposal: Mapping[str, Any]) -> str:
    structure = _mapping(proposal.get("structure"))
    kind = str(structure.get("kind", "")).replace("_", " ")
    legs = structure.get("legs")
    if not isinstance(legs, Sequence) or not legs:
        return kind or "—"
    strikes: list[str] = []
    for leg in legs:
        contract = _mapping(_mapping(leg).get("contract"))
        strike = _decimal_or_none(contract.get("strike"))
        if strike is not None:
            strikes.append(f"{strike.normalize():f}")
    return f"{kind} {'/'.join(strikes)}" if strikes else kind


def _implied_outcome(record: Mapping[str, Any]) -> str:
    """What the submission said, where no amendment has arrived yet."""
    submission = _mapping(record.get("submission"))
    return str(submission.get("raw_status", ""))


def _trend(value: Any) -> str:
    if isinstance(value, Mapping):
        state = value.get("state") or value.get("phase")
        strength = value.get("strength")
        if state is not None and strength is not None:
            return f"{state} {strength}"
        if state is not None:
            return str(state)
    if value is None:
        return "unmeasured"
    return str(value)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _number(value: Any, *, unmeasured: str = "—") -> str:
    if value is None or value == "":
        return unmeasured
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def _decimal_or_none(value: Any) -> Decimal | None:
    if value is None or value == "" or isinstance(value, bool):
        return None
    try:
        return Decimal(str(value))
    except InvalidOperation:
        return None


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _clock(value: Any) -> str:
    if not isinstance(value, str):
        return "—"
    try:
        return datetime.fromisoformat(value).strftime("%H:%M:%S")
    except ValueError:
        return value
