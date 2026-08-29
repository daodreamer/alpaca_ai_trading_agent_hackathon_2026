"""Closing the loop on D3 — the fill that lands hours after the decision.

D3 says later facts arrive as separate amendment lines. That is a mechanism, and
a mechanism with no producer is a mechanism nobody has tested against reality.
This module is the producer: it reads a day back, finds the cycles whose orders
were still live when the line was written, asks the broker what happened, and
appends one amendment each.

The whole thing is arranged so the interesting part is pure:

* `unresolved` takes journalled dicts and returns the cycles still open. No
  session, no clock, no I/O — so "which orders are we still waiting on?" is
  answerable from a file alone, which is what makes it answerable in a replay.
* `reconcile` is the thin impure shell: one `read_back` per unresolved cycle,
  one amendment per answer.

**Re-running is safe and is expected to happen.** A second pass writes a second
amendment; the later one wins on read (D3), the earlier one stays on disk, and
the decision line is never touched. That is the property that lets this run on a
timer without anyone having to reason about whether it already ran.

**An order that cannot be read back is left alone, loudly.** `read_back` raises
rather than guessing between "no order exists" and "an order exists we cannot
see" (specs/04 D4), and this catches that into a `failures` list rather than
letting one unreadable order abandon the other nine. Nothing is invented to fill
the gap: an unamended cycle reads as unresolved, which is exactly what it is.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from alphagate.execution import ExecutionError, McpSession, PartialFillBreach, read_back
from alphagate.journal.outcome import OutcomeRecord, outcome_from
from alphagate.journal.writer import Journal

__all__ = ["Pending", "ReconcileResult", "reconcile", "unresolved"]

_LIVE_STAGES = frozenset({"submitted"})
"""Only `SUBMITTED` is worth asking about.

`FILLED` is already terminal, `REJECTED` and `VETOED` never reached the broker,
and `BREACHED` is a latched kill switch that a human clears — asking the broker
about it would be asking a question whose answer we are forbidden to act on.
"""


@dataclass(frozen=True, slots=True)
class Pending:
    """A cycle whose order was still live when its line was written."""

    cycle_id: str
    client_order_id: str
    submitted_at: datetime


@dataclass
class ReconcileResult:
    """What one pass learned. Both halves matter."""

    amended: list[OutcomeRecord] = field(default_factory=list)
    failures: list[tuple[str, str]] = field(default_factory=list)
    """`(cycle_id, why)` for orders that could not be read back. Reported, never
    guessed at — see the module docstring."""

    @property
    def resolved(self) -> int:
        return sum(1 for outcome in self.amended if outcome.is_terminal)

    def summary(self) -> str:
        tail = f", {len(self.failures)} unreadable" if self.failures else ""
        return f"{len(self.amended)} amended ({self.resolved} terminal){tail}"


def unresolved(records: Sequence[Mapping[str, Any]]) -> tuple[Pending, ...]:
    """The cycles still waiting on the broker, in journal order. Pure.

    Reads the *final* state of each cycle, so a cycle already amended to
    `filled` by an earlier pass drops out here rather than being asked about
    again. That is why this takes `Journal.read` output and not `raw_lines`:
    amendments are the record of what we have already learned, and ignoring them
    would make every pass ask every question again.
    """
    pending: list[Pending] = []
    for record in records:
        outcome = record.get("outcome")
        if isinstance(outcome, Mapping) and _is_terminal(outcome):
            continue
        if str(record.get("stage", "")) not in _LIVE_STAGES:
            continue
        submission = record.get("submission")
        if not isinstance(submission, Mapping):
            continue
        key = str(submission.get("client_order_id") or "")
        submitted_at = _timestamp(submission.get("submitted_at"))
        if not key or submitted_at is None:
            continue
        pending.append(
            Pending(
                cycle_id=str(record.get("cycle_id", "")),
                client_order_id=key,
                submitted_at=submitted_at,
            )
        )
    return tuple(pending)


def reconcile(
    journal: Journal,
    day: date,
    mcp: McpSession,
    *,
    as_of: datetime,
) -> ReconcileResult:
    """Ask the broker about every unresolved cycle and amend the day.

    `as_of` is a parameter, not a clock read — the reconciler is the one place
    where the temptation to reach for `now()` is strongest, because it is
    genuinely about the passage of time, and specs/01 Rule 5 does not carve out
    an exception for the cases that feel like they deserve one.
    """
    result = ReconcileResult()
    for item in unresolved(journal.read(day)):
        try:
            submission = read_back(
                item.client_order_id, mcp, submitted_at=item.submitted_at
            )
        except PartialFillBreach as breach:
            # specs/04 D5. A spread half filled is a naked leg. Record what the
            # broker says and let the operator see it; do not try to leg out.
            outcome = outcome_from(
                breach.submission, cycle_id=item.cycle_id, observed_at=as_of
            )
            journal.record_outcome(outcome, day=day)
            result.amended.append(outcome)
            result.failures.append((item.cycle_id, f"partial fill: {breach}"))
            continue
        except ExecutionError as failure:
            result.failures.append((item.cycle_id, f"{type(failure).__name__}: {failure}"))
            continue

        outcome = outcome_from(submission, cycle_id=item.cycle_id, observed_at=as_of)
        journal.record_outcome(outcome, day=day)
        result.amended.append(outcome)
    return result


def _is_terminal(outcome: Mapping[str, Any]) -> bool:
    """Terminal by the same rule `OrderStatus.is_terminal` uses, read off disk.

    `unknown` is deliberately absent from the set: an unrecognised status is an
    unresolved order, so it stays in the queue and gets asked about again.
    """
    return str(outcome.get("status", "")) in {
        "filled",
        "canceled",
        "expired",
        "rejected",
        "replaced",
    }


def _timestamp(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None
