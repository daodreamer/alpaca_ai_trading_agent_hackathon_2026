# AlphaGate

**Trading agents that can be overruled.**

Built for the [Alpaca AI Trading Agents Hackathon](https://lablab.ai/ai-hackathons/alpaca-ai-trading-agents-hackathon)
(28 Aug – 4 Sep 2026), Options Alpha Agents track.

## The idea

Most LLM trading agents put the model in the decision seat: prompt in, order out.
That fails in options for a specific reason — the loss function is asymmetric and
the model has no calibrated sense of how much it can lose. A hallucinated ticker
costs you a bad fill; a hallucinated naked short strangle costs you the account.

AlphaGate splits the job:

- The **LLM proposes structure**. Given a market read, which options structure
  expresses it — credit spread, debit spread, condor, which expiry, which strikes.
  This is a judgement task with a bounded, checkable output.
- **Deterministic code disposes.** Every proposal passes a Risk Gate that can
  veto it. The Gate is pure, tested, and has no model in it. It owns defined-risk
  enforcement, position limits, greeks budget, and the kill switch.
- **Perception is not a prompt.** Trend state, market structure, and price levels
  come from a deterministic engine (`alphagate.core`), not from asking a model to
  read a chart. The model receives facts, not pixels.

Every order carries a decision record: the inputs the agent saw, what it proposed,
what the Gate said, and why. You can open any fill in the dashboard and read the
reasoning that produced it.

The same split runs a second time, one level up, over equities. There the
proposer is not a model but a *research pipeline* — the sibling project in this
repository, which searches hypotheses, tries to destroy them, and hands over the
one that survived a pre-registered out-of-sample window. It hands over weights
and nothing else, because it does not know the account's equity and inventing
one would be its first step towards placing the order. Everything between that
file and a share order — sizing, reconciliation, caps, a kill switch, a fill
journal, and a second Risk Gate shaped for equities — is on this side of the
line.

## Two agents, one account

There are **two** trading paths in this repository, and they are different
answers to the same brief.

**The options agent** is the one the first half of this README describes: the
LLM proposes a structure, the Risk Gate disposes. Its strategy layer is
incomplete — specs/07 D4 and D5 are unimplemented, so the live path builds
fixed-width put credit spreads whatever the trend says — and it has no backtest,
so nothing about it is a claim about edge yet.

**The equity agent** executes a strategy that *has* been validated, and it
executes only that one. `ai_quant_researcher/` searched 324 hypotheses, put the
survivor through walk-forward, pre-registered it, and spent a one-shot sealed
window on it:

```
rs_volatility_consistency_neutral_v1 [3f6e2c8a9309068b]
sealed window 2024-09-03 → 2026-08-27   (498 sessions, never read during the search)
  strategy   return +56.39%  sharpe +1.86  maxDD -10.4%  trades 561
  residual   alpha +16.72%/yr  beta 0.43  t +2.22  IR +1.58
```

That window **was not refuted**, which is the strongest verdict 498 sessions can
produce — `can_confirm` is `False` by construction, because the standard error
on an annualised Sharpe there is about ±0.71. The researcher writes the weights
it holds to a file and stops. AlphaGate reads that file, prices it, gates every
resulting order, and places what survives.

[specs/09](specs/09-equity-execution.md) is the contract between them, and
`scripts/pipeline.py` runs the whole chain:

```bash
python scripts/pipeline.py            # refresh the cache → rebuild the book → trade it
python scripts/pipeline.py --dry-run  # rebuild the book, plan against it, place nothing
```

**Neither project imports the other.** The seam is the JSON artefact, and
`tests/test_boundaries.py` fails the build if an import appears in either
direction, or in the pipeline driver that runs both. The two hold different
invariants — the researcher holds no money and places no orders, so it has no
`Decimal` rule and no Risk Gate; AlphaGate has both — and an import would make
one project's invariants the other's problem.

### Status

| | |
| --- | --- |
| **Equity chain** — research → sealed validation → target book → priced, gated, journalled, and rendered on the dashboard. Verified end to end against the live paper account. | working |
| **First live equity submission** — the plan, the Gate and the door are tested offline and against a closed market; no share order has met the broker yet. | next session |
| **Options strategy** — specs/07 D4 and D5 unimplemented. | blocked on research |
| **Options backtest** — spec 08 not written. [specs/00](specs/00-brief.md) says this is what turns "up 2% in four days" into a claim about edge. | blocked on the above |

2,344 tests green (pytest / ruff / mypy; frontend eslint / tsc), all offline.

Development runs against a pre-existing paper account. The competition account is
a **new, dedicated** one, switched over on 28 Aug — hard gate 4 in
[specs/00-brief.md](specs/00-brief.md), and the reason `preflight` refuses to pass
that line until you assert it by hand.

## Manual

Everything runs from the repository root. Paths default to `.env.local` and
`journal/` here, so none of these need flags.

### Setup, once

Credentials live in `.env.local` (see [.env.example](.env.example)). The Alpaca
keys are required.

`DEEPSEEK_API_KEY` is needed for the model to propose anything. Without it the
agent **refuses to start rather than quietly trading without a model** — a silent
fallback would mean believing you were running the LLM path when you were not.
Pass `--no-model` to ask for the deterministic proposer on purpose.

### Before every trading day

```bash
uv run --directory backend python -m alphagate preflight
```

Checks the four hard gates against the live account instead of against memory: a
live key, an account that is not the dedicated one, an options level below 3.
None of those announce themselves otherwise — you find out from a rejection at
14:30, by which point the trading day is half gone.

The "dedicated new paper account" line fails by design until you confirm it:

```bash
uv run --directory backend python -m alphagate preflight --confirm-dedicated
```

Only pass that once it is genuinely the competition account.

### Running the options agent

```bash
# one cycle, right now — reads the market, builds a menu, gates it, journals it
uv run --directory backend python -m alphagate once
uv run --directory backend python -m alphagate once --no-model   # skip the LLM call

# a whole session, on the 15-minute schedule, from now to the close
uv run --directory backend python -m alphagate run --dry-run     # gate everything, submit nothing
uv run --directory backend python -m alphagate run               # place paper orders
uv run --directory backend python -m alphagate run --no-supervise  # one session, no restarts
```

**`once` is dry by default; `run` is not.** A debugging command that places
orders is a debugging command that places orders by accident. `run` is what a
trading day is, so asking it to trade is not a surprise — but `--dry-run` first
is the cheap habit.

Both journal every cycle, including the ones that decided nothing. That is the
point: "why didn't it trade at 14:30?" has an answer on disk.

**Every slot evaluates exits before it considers opening anything.** Each open
position is re-priced from a fresh chain and put through the exit policy — take
half the credit, stop at twice it, close at two days to expiry. A position that
should be held produces no journal line; a close produces one with the rule that
fired and the numbers behind it. No model is consulted on the way out, and none
can be: the decision to take a loss is the one you least want improvised.

**`run` supervises itself.** A dropped connection at 14:00 should not cost the
afternoon, so a session that dies is resumed — from the slots still ahead, never
replaying one that already has a journal line. Two things stop it for good
instead: a partial-fill breach and a latched kill switch. Both mean a naked leg
or a book a human has not looked at, and resuming through either would be worse
than stopping.

### Trading the validated equity strategy

The whole chain, from bars to orders:

```bash
python scripts/pipeline.py              # refresh → rebuild the book → trade it
python scripts/pipeline.py --dry-run    # rebuild the book, plan against it, send nothing
python scripts/pipeline.py book trade   # skip the pull; the cache is already current
```

Or one stage at a time, which is what you type when something is wrong:

```bash
uv run --directory backend python -m alphagate equity-preflight
uv run --directory backend python -m alphagate equity-plan        # gated, nothing sent
uv run --directory backend python -m alphagate equity-rebalance   # orders placed
uv run --directory backend python -m alphagate equity-run         # a session
uv run --directory backend python -m alphagate equity-status
```

**`equity-plan` is dry and `equity-rebalance` is not** — the same asymmetry
`once` and `run` keep, for the same reason.

**Pin the strategy before any of this works.** `.env.local` needs

```
ALPHAGATE_STRATEGY_FINGERPRINT=3f6e2c8a9309068b
```

and a target book naming any other fingerprint is refused by name. There is
deliberately no default: the pin is the whole of "only the strategy the
researcher validated", and a default would make it a property of whichever file
happened to be newest.

`equity-preflight` checks the six refusals of [specs/09](specs/09-equity-execution.md)
D1 before the open rather than during a rebalance — the schema, the pin, the
registry lifecycle state, whether the seal was spent, whether the sealed window
refuted the rule, and whether the book is fresh. Then the account, and whether
the market is open at all.

**`equity-run` heartbeats every thirty seconds and rebalances once**, fifteen
minutes after the open. The heartbeat is most of what it does and it is not
decoration: this strategy rebalances every five sessions, so four days in five
the plan is empty, and a process that only woke to trade would be
indistinguishable from one that had died.

**The band is proportional.** A position is left alone until its drift exceeds
20% of its own size, floored at $25. Not a fraction of equity — the book's
sleeve positions are $194 each on a $100k account, and a 0.25%-of-equity band
would have been $253, which is to say the hundred sleeve names could never have
been established at all. That was the first version, and
[specs/09 D3](specs/09-equity-execution.md) records it.

**A symbol the book no longer wants is sold to zero.** There is no separate exit
rule: an absent target is a target of zero, so the whole position is the drift.
Nothing to forget to run.

### Watching it

```bash
uv run --directory backend python -m alphagate status         # right now, in the terminal
uv run --directory backend python -m alphagate serve          # http://127.0.0.1:8000
uv run --directory backend python -m alphagate show -v        # a journalled day
uv run --directory backend python -m alphagate show --day 2026-08-28
```

`status` answers the question the journal cannot: **is it running, what does it
hold, and how close is it to a limit.** Equity and today's P&L, every open
position with its current mark and how far it is from the profit target and the
stop, the four budgeted limits against what is used, and any broker legs the
journal cannot account for.

The **dashboard** is the same information with a Live tab and a Journal tab, and
it is the thing to open during a demo:

- **Options** — health, money, positions with a bar showing each one travelling
  between its stop and its target, room left before the Gate refuses its own
  next proposal, and the day's cycle counts. Polls every 15 seconds.
- **Equity** — the strategy that earned the account, first: name, fingerprint,
  hypothesis, and the sealed alpha, beta and `t` beside the two denominators
  that qualify them. Then the book — target weight against held weight, with
  each position's own no-trade band — and today's orders with their verdicts.
- **Journal** — every cycle, quiet ones included. Expand one for the market read,
  the model's rationale, and all thirteen Gate checks **sorted tightest-first**,
  so a check that passed with 4% of its budget left sits at the top where it
  belongs.

Two properties are structural rather than intended, and
[tests/test_boundaries.py](backend/tests/test_boundaries.py) enforces both:

**The dashboard cannot trade.** `alphagate.interface` imports the journal and
nothing else — no MCP session, no market data client, no `alphagate.live`. There
is no code path from a browser to an order.

**It learns the live book from a file.** The agent writes `journal/status.json`
each slot; the page reads it. That keeps the guard above intact, and it fails
honestly: if the agent stops, the file stops being rewritten and the page says
*not running* instead of showing a stale book with a confident face.

### Building the dashboard

The Live tab is a React app (Vite, Tailwind v4, shadcn/ui) that builds into the
Python package, so one process serves the page and the API:

```bash
cd frontend && npm install && npm run build   # → backend/src/alphagate/interface/static/
npm run dev                                   # hot reload, proxying /api to :8000
```

The build is optional. With no `static/` directory the server still runs and
falls back to server-rendered journal pages that need no toolchain at all — a
dashboard that refused to start because a frontend was not compiled would be a
bad thing to discover at 09:20.

### What lands on disk

| Path | What it is |
| --- | --- |
| `journal/YYYY-MM-DD.jsonl` | One line per cycle, append-only. The record the submission ships. |
| `journal/state.json` | High-water equity and the kill-switch latch, carried across days. |
| `journal/status.json` | What the agent is doing right now, rewritten every slot. The dashboard's only live source. |
| `journal/iv/` | Implied-volatility history, one file per underlying, accumulated a session at a time. |
| `journal/equity-status.json` | The equity book right now, rewritten every heartbeat. The Equity tab's only live source. |
| `journal/books/` | Every target book that was actually executed, byte-for-byte. `aqr` regenerates its own output daily; this is the copy that can still say what ran. |

No API key, account number or account id is ever written to any of them — that
is [specs/06](specs/06-journal.md) D4, and it is tested against a fixture that
deliberately contains credential-shaped values.

### Development

```bash
uv run --directory backend --extra dev pytest
uv run --directory backend --extra dev ruff check .
uv run --directory backend --extra dev mypy
```

The whole suite runs offline. Nothing in `tests/` opens a socket or a
subprocess — market data and the broker are both replayed from captured
payloads, which is why it takes twenty seconds.

See [specs/](specs/) for the contracts.

## Repository layout

| Directory | What it is |
| --- | --- |
| [backend/](backend/) | **AlphaGate itself.** The options agent, the Risk Gate, execution, the journal. This is the competition entry. |
| [specs/](specs/) | Contracts, written before the code they govern. |
| [frontend/](frontend/) | The dashboard's Live tab — Vite + React + shadcn/ui, built into the Python package. |
| [journal/](journal/) | The decision records the agent writes — one line per cycle, append-only. Committed: this is the evidence the submission rests on ([specs/06](specs/06-journal.md) D1). |
| [adr/](adr/) | Decisions and the reasoning behind them. |
| [ai_quant_researcher/](ai_quant_researcher/) | A separate system sharing the repository: an equities strategy-research lab that implements [specs/trading_strategy_architecture.md](specs/trading_strategy_architecture.md). It produces the strategy the equity agent executes, and imports nothing from AlphaGate. |
| [scripts/](scripts/) | `pipeline.py` — refresh, rebuild the book, trade it. Runs both projects' CLIs as subprocesses and imports neither. |

`ai_quant_researcher/` is a **sibling project, not part of AlphaGate**. It has
its own `pyproject.toml`, its own virtualenv, its own test suite, and it imports
nothing from `alphagate`. The two answer different questions — AlphaGate asks
"should this order be allowed through", the researcher asks "does this equities
rule have an edge that survives out of sample" — and they are kept apart so
neither's invariants leak into the other.

They are nonetheless **connected**, through a file. The researcher's
`CONSUMER_MUST_SUPPLY` names six things it will not do — sizing, reconciliation,
turnover caps, an equity-shaped risk gate, a kill switch, a fill journal — and
[specs/09](specs/09-equity-execution.md) is AlphaGate supplying all six. What
travels between them is a target book: weights per symbol, plus the fingerprint,
the seal state, the sealed measurement and both multiplicity denominators, so a
reader who has never seen `aqr` can audit where the weights came from.

Concretely, the rules in [CLAUDE.md](CLAUDE.md) §3 govern `backend/` and do not
apply to `ai_quant_researcher/`: it holds no money (so no `Decimal` requirement),
places no orders (so no Risk Gate), and has its own LLM boundary enforced by its
own [boundary tests](ai_quant_researcher/tests/test_boundaries.py). Do not
"reconcile" the two.

## Provenance

`alphagate.core` is extracted from the author's existing open-source project
*Personal Market Monitor* — a deterministic, UI-free market analysis engine
(indicators, market structure, levels, trend state machine). It predates this
hackathon and is reused as a library. Everything else in this repository —
the options domain, the Risk Gate, execution, the agent, and the dashboard —
is built during the competition window. See [adr/0001-core-reuse.md](adr/0001-core-reuse.md).
