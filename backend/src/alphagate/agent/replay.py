"""Reconstruct a cycle from the journal alone — specs/06 D6.

> "The journal is sufficient to reconstruct a cycle without a network or a
> model: `market_read` and `candidates` are the pure inputs, `choice` is the
> recorded model output, and `RecordedProposer` supplies it."

That claim was being demonstrated by a helper that lived inside a test file,
which is a slightly awkward place for the load-bearing half of a spec decision:
a helper only the test can reach proves the mechanism works and proves nothing
about whether the shipped system can do it. So it lives here, the test imports
it, and the backtest and the dashboard can too.

It lives in `agent/` rather than `journal/` on purpose. `Choice` is an agent
type and `runner.py` already imports the journal; putting the reader downstream
would close the loop into an import cycle and, worse, would give the journal a
reason to know what a model response looks like. The journal writes records. The
agent knows how to read itself back out of them.

Nothing here opens a socket or reads a clock.
"""

from __future__ import annotations

from datetime import date

from alphagate.agent.model import Choice
from alphagate.agent.proposer import RecordedProposer
from alphagate.journal import Journal

__all__ = ["recorded_choices", "replay_proposer"]


def recorded_choices(journal: Journal, day: date) -> dict[str, Choice]:
    """Every journalled model decision for a day, keyed by `cycle_id`.

    From the bytes on disk, not from an in-memory record. That is the difference
    between testing the replay mechanism and testing the artefact the submission
    ships, and only one of them is evidence — if the journal does not carry
    enough to rebuild a `Choice`, the record is narration and D6 is false.

    Cycles that never reached a model (`NO_SETUP`, `NO_CANDIDATES`) have no
    choice to recover and are simply absent. `RecordedProposer` treats a missing
    cycle as a decline, which is the honest reading: we have no record of the
    model choosing anything, so replaying it as a choice would be inventing one.
    """
    recovered: dict[str, Choice] = {}
    for line in journal.read(day):
        choice = line.get("choice")
        if not isinstance(choice, dict):
            continue
        cycle_id = str(line.get("cycle_id", ""))
        if not cycle_id:
            continue
        index = choice.get("candidate_index")
        recovered[cycle_id] = Choice(
            candidate_index=int(index) if isinstance(index, int) else None,
            rationale=str(choice.get("rationale", "")),
            self_reported_confidence=float(choice.get("self_reported_confidence") or 0.0),
        )
    return recovered


def replay_proposer(journal: Journal, day: date) -> RecordedProposer:
    """A proposer that answers from the file — specs/05 D7, specs/06 D6.

    Drop-in for the live proposer in `run_cycle`, so replaying a day differs
    from trading it by the value of one argument and the value of `as_of`.
    """
    return RecordedProposer(recorded_choices(journal, day))
