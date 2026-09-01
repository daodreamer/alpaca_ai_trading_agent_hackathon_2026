# PLAN — options, from a research loop to a placed order

Written to survive a context reset. Everything here is executable from a cold
start; nothing depends on remembering a conversation. Companion to
[`ai_quant_researcher/PLAN.md`](ai_quant_researcher/PLAN.md), which covers the
equity side and is done.

Read [`specs/10-options-research.md`](specs/10-options-research.md) before
touching `ai_quant_researcher/src/aqr/options/`, and
[`specs/05-agent.md`](specs/05-agent.md) + [`specs/07-strategy.md`](specs/07-strategy.md)
before touching `backend/src/alphagate/agent/`.

---

## Where things stand — verified 2026-09-01

| | |
|---|---|
| `ai_quant_researcher` | **1294 tests, ruff + mypy clean.** Options engine, feature layer, walk-forward, robustness and evaluator profile all green |
| `backend` | **2411 tests, ruff + mypy clean.** Options agent (specs/05 8-step cycle) and equity book both present |
| Journal | **84 equity orders on 2026-08-31. Zero options records, ever.** |

Commits: `7a8e3c8` (engine), `4f50c4f` (validation chain), `f940308` (sleeves).

### The one urgent fact

[`specs/00-brief.md`](specs/00-brief.md) hard gate 3: **the strategy must
incorporate options trading**, and the submission closes **2026-09-04 15:00
UTC**. There are three sessions left — 09-01, 09-02, 09-03 — and the options
agent has never placed an order. Everything in Phase O4..O7 below is worth
nothing if that stays true, so **Phase A runs first and runs today**, and it
does not depend on any of the research work.

Phase A trades the hand-written rule from specs/07, which already exists and is
already gated. The research track replaces *which rule* the agent runs, not
*whether* it runs.

### What the research has actually concluded so far

Six hand-written structures, full chain, out of sample, at a sizing where the
account is not the binding constraint:

| structure | cycles | OOS return | alpha/yr | t(α) | verdict |
|---|--:|--:|--:|--:|---|
| put credit spread, unconditional | 36 | +14.4% | +0.7% | +0.26 | REVIEW |
| put credit spread, `close > sma(200)` | 30 | +9.8% | +0.4% | +0.13 | REVIEW |
| put credit spread, `iv_rank() > 50` | 14 | +2.9% | +0.7% | +0.37 | REJECT |
| iron condor | 35 | +4.6% | +0.9% | +0.22 | REVIEW |
| long put | 36 | −98.4% | −58.5% | −2.23 | REJECT |
| call credit spread | 35 | −10.1% | +2.1% | +0.42 | REJECT |

Nothing accepted. `REVIEW` means "alpha not distinguishable from zero", never a
promotion. This is the expected outcome — specs/07 D0 cites the literature that
says the variance risk premium's alpha has decayed, and that section was written
before any of this ran.

**Design consequence, and it is not pessimism:** the pipeline must treat "the
research promoted nothing" as a first-class outcome, not a blocked state. Phase
O7 below therefore keeps the hand-written rule running when no researched rule
has earned promotion. Building a handoff that only works when a strategy passes
would mean the honest answer breaks the product.

---

## Phase A — Options trading live, today

**Owner: whoever reads this first. Nothing else starts until A3 is done.**

### A1 — Prove the loop end to end, without placing

```bash
uv run --directory backend python -m alphagate preflight
uv run --directory backend python -m alphagate once            # defaults to NOT placing
uv run --directory backend python -m alphagate show -v
```

`once` must reach step 8 of the specs/05 cycle and write a `CycleRecord`
whatever happens — a screen that finds nothing, a model that declines and a Gate
that vetoes each write a record. If `show` prints nothing, the cycle is not
reaching step 8 and that is the bug to fix first.

Acceptance: a journal record exists for today with a reason, whether or not a
trade was proposed.

### A2 — Place

```bash
uv run --directory backend python -m alphagate run              # NO --dry-run: this places
```

`run` without `--dry-run` places real paper orders. Read
[`CLAUDE.md`](CLAUDE.md) §6 before running it.

Acceptance: at least one option structure submitted, gated, and journalled, with
its `GatedOrder` provenance intact.

### A3 — Keep it running for the three remaining sessions

Target **≥ 30 fills** (specs/00). With three sessions left that means several
cycles per session, so schedule `run` rather than invoking `once` by hand.
Check `journal/` grows every session and `alphagate status` reflects it.

Acceptance: fills accumulating, `equity-status` and `status` both truthful, kill
switch never latched by the *other* sleeve (that is what `f940308` fixed —
verify it holds under a real drawdown, do not assume).

---

## Phase O4 — The LLM proposes option hypotheses

Everything below is in `ai_quant_researcher/`. Commands run from that directory.
`uv run --extra dev pytest` / `ruff check .` / `mypy` must be green at the end of
every step.

### O4.1 — The proposer

New `src/aqr/agent/option_proposer.py` and `option_prompt.py`, alongside the
equity `proposer.py` — **not inside it**. The two vocabularies are deliberately
unmixed (CLAUDE.md §2b), and the option prompt must enumerate specs/10 D6's
feature table and D4's structure whitelist, not the equity registry.

The model emits `OptionSpec` **fields**, never code, never a price, never a
size. Reuse the equity proposer's repair loop: a spec that fails validation goes
back with the error, a bounded number of times, and every attempt is recorded.

What the prompt must state, because the model will otherwise propose things the
data cannot price:

- there is **no exit** — structures are held to expiry (specs/10 D1). No stop,
  no target, no roll, no "manage at 50%".
- the width is named **by delta**, not by points (D5). A 10-point wing resolves
  on 23% of sessions; a delta wing on 98%.
- expiries are ~14 / ~28 / ~49 DTE only. No 0DTE, no weeklies, no LEAPS.
- one underlying, SPY. There is no cross-section to rank.

Acceptance: an offline proposer run produces valid, varied `OptionSpec`s with no
API key; every rejection is recorded with its reason.

### O4.2 — The campaign, capped at 20 hypotheses

```bash
uv run aqr option-research --iterations 20 --provider deepseek
```

**The cap is a hard gate, not a default.** specs/10 D8: there are 71
non-overlapping cycles in 5.55 years. The equity search spent 414 hypotheses and
deflated its Sharpe to 0.74 for it; 400 trials against 71 cycles produces a
number with no information in it.

### O4.3 — Keep the search denominators apart

The registry must count option trials **separately** from the 414 equity ones.
The multiplicity bar (`multiplicity_bar`, Bonferroni on a 5% family-wise rate)
is computed per window, and a denominator that mixes the two searches makes both
bars wrong.

Acceptance: `aqr experiments` and `aqr registry` distinguish option experiments;
a test asserts the two counts never merge.

---

## Phase O5 — Pre-registration and the one sealed run

### O5.1 — The sealed option cache

The sealed roots exist (`data-options-sealed/`, 1,260 sessions to 2026-08-28)
but **the sealed underlying does not**. Settlement needs SPY's raw closes past
the embargo, and `aqr pull` clamps at it by design.

Pull it through the sealed entry point, with `--adjustment raw` — the reason is
specs/10 D0 and D2a, and getting this wrong is the bug that cost a day:

```bash
uv run python -m aqr.cli_sealed pull --symbols SPY --adjustment raw \
    --csv-root data-options-underlying-sealed --timeframe 1D
```

If `cli_sealed pull` does not accept `--adjustment`, add it there the same way
`4f50c4f` added it to `aqr pull`. Verify with the parity check in
`tests/test_option_cache_claims.py` pointed at the sealed roots: implied spot
must agree with the close to within 2%.

### O5.2 — Pre-registration for option specs

Extend `registry/db.py`'s `preregistration` table and `aqr preregister` to
option fingerprints. Ancestry taint is checked by campaign, as on the equity
side.

### O5.3 — Spend one shot

```bash
uv run aqr preregister --rule "..." <FINGERPRINT>
uv run python -m aqr.cli_sealed option-run <FINGERPRINT>
```

Only for a candidate that earned it out of sample. If nothing did, **do not
spend the shot** — record that the search promoted nothing, which is a result
and is the one the evidence currently supports.

The sealed window is ~25 independent cycles. It can refute and it cannot
confirm; no artefact may word it otherwise.

---

## Phase O6 — The handoff artefact

New `src/aqr/option_book.py`, modelled on `target_book.py` but **not** an
extension of it: an option book is a list of legs, not a vector of weights, and
stretching `TargetBook` to cover both would make every consumer branch.

It must carry, per specs/09 D0's reasoning:

- schema version, generated-at, spec fingerprint, spec name and version
- the rule itself — structure kind, DTE target, anchor delta, width delta,
  cadence, sizing — because the executor rebuilds the structure from live
  quotes rather than from stale strikes
- the session it was produced from, the dataset version (which now names the
  price adjustment), the seal state and the sealed-run verdict
- `CONSUMER_MUST_SUPPLY`, carried inside the artefact

**Strikes are deliberately not in the book.** A book written from a Tuesday
close naming strike 5480 is wrong by Wednesday's open. The executor gets the
*rule* and resolves it against a live chain — which is also why the backend
needs its own delta-selection code and cannot import `aqr.options.chain`.

```bash
uv run aqr option-book <FINGERPRINT>
```

Refuses for the same reasons `target-book` does: unspent seal, refuted sealed
run, wrong status.

Acceptance: `backend/tests/test_boundaries.py` guard 9 still passes in both
directions — neither project imports the other, and `scripts/pipeline.py` calls
both CLIs by subprocess only.

---

## Phase O7 — The backend executes the researched rule

### O7.1 — Load and refuse

`backend/src/alphagate/agent/option_book.py`: pure, takes a parsed mapping and
not a path, and refuses in the same enumerated way `load_target_book` does —
reporting every fault at once, not the first.

**The fingerprint is pinned in configuration, not read from the book.** Add
`ALPHAGATE_OPTION_FINGERPRINT` to `.env.local` with no default, exactly as
`ALPHAGATE_STRATEGY_FINGERPRINT` has none. That pin is the only checkable
meaning of "it only trades the rule the research validated".

### O7.2 — The rule drives the screen, the Gate does not move

Today the agent's structure choice comes from specs/07's hand-written constants.
Make those come from the loaded book instead — the DTE target, the anchor delta,
the width delta, the cadence, the sleeve allocation.

**Nothing about the Risk Gate changes.** Every order still goes through it,
there is still no bypass, `execution/` still accepts only a `GatedOrder`, and
naked short structures are still unrepresentable. A researched rule is an input
to the agent, never an exemption from the gate.

### O7.3 — No book, no problem

When no book is present, or the pinned fingerprint has nothing promoted behind
it, the agent runs specs/07's hand-written rule and says so in the journal. The
research refusing to promote anything must not stop the system trading.

### O7.4 — The pipeline

Extend `scripts/pipeline.py` with the options chain, mirroring the equity one:

```
aqr options-pull → aqr option-book → alphagate preflight → alphagate run
```

subprocess only, both directions, no imports.

---

## Definition of done for the whole track

- `uv run --directory backend --extra dev pytest` / `ruff check .` / `mypy` green
- `uv run --extra dev pytest` / `ruff check .` / `mypy` green in `ai_quant_researcher`
- ≥ 30 option fills journalled, every one through the Gate
- The dashboard shows what was traded and **why**, including the cycles that
  declined
- Every claim in the demo traceable to a journal record or a registry row

## Known risks, carried rather than hidden

**The research will probably promote nothing.** Six structures, all REJECT or
REVIEW. That is the honest state of the evidence and O7.3 exists so it is not
also a broken pipeline. The submission's claim is the one specs/07 D0 already
frames: we harvest a documented premium whose alpha has decayed, and the
contribution is risk control and execution discipline, not prediction.

**Three sessions is not 30 fills at one cycle a day.** Phase A3 needs several
cycles per session, which means scheduling and it means the kill switch and the
sleeve accounting get exercised for real.

**The sealed option window is ~25 independent cycles.** It refutes; it does not
confirm. Do not let a demo script word it as validation.
