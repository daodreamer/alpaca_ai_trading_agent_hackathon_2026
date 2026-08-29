"""The decision journal — specs/06.

One record per agent cycle, append-only, JSONL. The artefact the demo is built
on: a judge who opens any fill and reads the reasoning that produced it —
including the checks that nearly stopped it — is looking at Technology
Implementation and Presentation at the same time.

* `writer` — D1 storage, D3 amendment, D4 redaction.
* `outcome` — D2's `OutcomeRecord`: the fill and the realised P&L, arriving
  later as amendments rather than as edits.
* `reconcile` — the producer for those amendments. Asks the broker what became
  of the orders that were still live when their lines were written.
* `trust` — D5. Answers "which bytes in this decision came from outside the
  trust boundary?" from the file, which is where a judge would look.

D6's replay guarantee belongs with the agent, because a `Choice` is an agent
type and the journal has no business knowing what a model response looks like:
see `alphagate.agent.replay`, tested in `tests/agent/test_replay.py`.
"""

from alphagate.journal.outcome import Fill, OutcomeRecord, outcome_from, realised
from alphagate.journal.reconcile import Pending, ReconcileResult, reconcile, unresolved
from alphagate.journal.trust import UntrustedSource, trust_report, untrusted_sources
from alphagate.journal.writer import REDACTED, Journal, encode, redact

__all__ = [
    "REDACTED",
    "Fill",
    "Journal",
    "OutcomeRecord",
    "Pending",
    "ReconcileResult",
    "UntrustedSource",
    "encode",
    "outcome_from",
    "realised",
    "reconcile",
    "redact",
    "trust_report",
    "unresolved",
    "untrusted_sources",
]
