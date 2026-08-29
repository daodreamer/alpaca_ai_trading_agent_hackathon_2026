# 06 — The decision journal

One record per agent cycle. Append-only. The single source of truth for what the
system did and why.

This is the artefact the demo is built on. A judge who opens any fill and reads
the reasoning that produced it — including the checks that nearly stopped it —
is looking at Technology Implementation and Presentation at the same time
([00](00-brief.md)).

## D1 — Storage

JSONL, one object per line, one file per trading day:
`journal/YYYY-MM-DD.jsonl`.

Not a database. Postgres was deliberately left behind with the rest of the
upstream persistence layer ([adr/0001](../adr/0001-core-reuse.md) D4): a
seven-day project does not need migrations, and an append-only text file is
trivially replayable, diffable, and shippable with the submission as evidence.

Writes are append-then-flush. A crash mid-cycle loses at most the current line,
and a truncated final line is skipped on read rather than failing the load.

## D2 — The record

```python
@dataclass(frozen=True, slots=True)
class CycleRecord:                 # alphagate.agent.cycle
    cycle_id: str                  # deterministic: f"{date}-{underlying}-{seq}"
    as_of: datetime                # tz-aware UTC
    stage: Stage                   # NO_SETUP | NO_CANDIDATES | DECLINED | VETOED
                                   # | DRY_RUN | SUBMITTED | REJECTED | FILLED
                                   # | BREACHED
    read: MarketRead               # what the engines saw          — 05 D2
    setup: Setup | None            # what the screen matched, if anything
    candidates: tuple[Candidate, ...]   # the full menu, not just the pick
    choice: Choice | None          # index, rationale, confidence  — 05 D3
    call: ModelCall | None         # prompt version, model id, response, latency
    proposal: TradeProposal | None
    verdict: Verdict | None        # ALL checks, passed and failed — 03 D3
    submission: Submission | None
    note: str                      # why the cycle ended where it did, in one line
```

Two deliberate departures from the first draft of this spec, both of which the
code got right and the prose had wrong:

**The domain types are reused, not mirrored.** The draft named `CandidateRecord`,
`VerdictRecord` and `SubmissionRecord`. Inventing a parallel record type per
domain type buys nothing here — the encoder walks dataclasses generically — and
costs a conversion layer that can drift from what was actually decided. The
journal stores `Candidate`, `Verdict` and `Submission` themselves.

**`outcome` is not a field.** The draft listed `outcome: OutcomeRecord | None` on
the record. It cannot be one: the record is written once, at the end of the
cycle, and the outcome is by definition not known then. `OutcomeRecord` exists
(`alphagate.journal.outcome`) and reaches the journal as a **D3 amendment**, so
it appears in the merged state a reader gets back from `Journal.read` and never
in the line written at decision time. That is the difference D3 is about, and
putting the field on the record would have quietly undone it.

Two more stages exist than the draft listed. `DRY_RUN` is approval without
submission — the pre-open check — and `BREACHED` is a partial fill on a
multi-leg order (04 D5). Both are first-class outcomes rather than flags,
so the journal never has to be read as "submitted, but not really".

Three things this format insists on:

**Every cycle, not every trade.** `NO_SETUP` and `DECLINED` entries are the
majority and they are the point. "Why didn't it trade at 14:30?" is answerable.

**The whole menu, not the pick.** `candidates` holds all 6–12 structures the
model chose between, each with its `StructureRisk`. Without them the rationale
is unfalsifiable; with them you can see what was passed over.

**Every check, passed and failed.** `verdict` carries the full `checks` tuple
([03](03-risk-gate.md) D3), each with `observed` and `limit`. The dashboard
renders near-misses, which is how you show a risk system working rather than
merely present.

## D3 — Amendment, not mutation

A record is written once, at the end of the cycle. Later facts — a fill hours
after submission, realised P&L on close — arrive as **separate amendment lines**
keyed by `cycle_id`, never as an edit to the original.

Reading applies amendments in file order. The original decision therefore stays
exactly as it was made, with no hindsight leaking backwards into it. This is the
same discipline as [01](01-architecture.md) Rule 4: no look-ahead, including in
the record of what we knew.

## D4 — Redaction

Never written: API keys, secrets, account numbers, the raw `Authorization`
header, or the account id. `client_order_id` and Alpaca order ids are fine and
necessary for reconciliation.

**The account id is the hard one, and it is a structural problem rather than a
pattern-matching one.** Alpaca returns it as a bare `id`; an order also has a
bare `id` that the line above explicitly requires us to keep; both are plain
UUIDs. No regex can separate them, and no deny-list of key names can either. The
only thing that distinguishes them is the object they are sitting in, so
`redact` tracks where it is in the document: an `id` inside an object carrying
`buying_power` or `account_number` is an account id and goes; an `id` beside
`client_order_id` is an order id and stays.

It fails safe in the direction that matters. A false positive redacts an order
id and breaks a reconciliation — loud, and fixable in a minute. A false negative
writes the account id into the file that goes on video.

A test asserts that no journal line matches the credential patterns, run against
a fixture whose inputs deliberately contain a fake key **and a fake account
identity of the real shape**. That last clause is not decoration: the captured
`get_account_info.json` had been hand-redacted before it was committed, so for a
while the suite had never seen an account payload with its identity fields
present and could not have caught this. The demo video shows this file; a leaked
key on screen is unrecoverable.

## D5 — The security envelope travels

MCP tool output arrives wrapped in `_alpaca_mcp_security` with
`"trust": "untrusted_tool_output"` ([04](04-execution.md) D7). The wrapper is
kept in the record, attached to the data it described.

The journal can then answer a question most agent projects cannot: *which bytes
in this decision came from outside the trust boundary?*

## D6 — Replay

The journal is sufficient to reconstruct a cycle without a network or a model:
`market_read` and `candidates` are the pure inputs, `choice` is the recorded
model output, and `RecordedProposer` ([05](05-agent.md) D7) supplies it. Replaying
a day must produce a byte-identical order set.

If replay diverges, something impure leaked into steps 1–3, 5 or 6. The replay
test is therefore also the determinism test for the whole pipeline, which is why
it is worth more than its size suggests.

## D7 — Reconciliation drives the amendments

D3 describes a mechanism; something has to drive it. `journal.reconcile` reads a
day back, finds the cycles whose orders were still live when their lines were
written, asks the broker, and appends one amendment each. The runner calls it at
the top of every slot, before exits, so the exit logic sees fills the broker
already has rather than a book we believe in.

Two properties make it safe to run on a timer:

* **Re-running converges.** A second pass writes a second amendment, the later
  one wins on read, and no earlier line is touched. Nobody has to reason about
  whether it already ran.
* **An unreadable order is reported, never guessed at.** [04](04-execution.md)
  D4 refuses to choose between "no order exists" and "an order exists we cannot
  see"; reconciliation inherits the refusal, surfaces it on the session result,
  and leaves the cycle unamended — which reads as unresolved, because it is.

A reconciliation failure never ends a trading session. An agent that stops
trading because it could not update a record has the priorities backwards.

## Test plan (RED first)

1. Every `Stage` round-trips through write and read — on records the pipeline
   genuinely *reached*, not records assigned a stage after the fact. A `FILLED`
   fixture with no submission proves the encoder handles nine enum values and
   nothing about whether the journal can hold what the agent produces.
2. A truncated final line is skipped, not fatal.
3. Amendments apply by `cycle_id` and never mutate the original line.
4. Reading a day with amendments out of order still yields the same final state,
   including an amendment that arrives *before* the record it amends.
5. Redaction: a fixture containing a fake key produces no matching journal bytes;
   and an account payload with its identity fields present loses the account id
   while the order id beside it survives.
6. The security envelope survives write and read, onto outcome amendments as
   well as submissions.
7. Full-day replay reproduces an identical order set — **through submission**,
   not only to the Gate. The broker must see byte-identical tool arguments on
   both passes; a replay that only reproduces `DRY_RUN` verdicts is a
   determinism claim about steps 1–6 wearing a claim about step 7.
8. `cycle_id` is deterministic and collision-free across underlyings and days.
   Where a sequence *is* reused — an operator restarting mid-session — the
   collapse is reported by `duplicate_cycles` rather than silently losing a line.
9. Reconciliation is idempotent, survives an unreadable order, and never ends
   the session.
