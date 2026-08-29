# AlphaGate

**An options trading agent that can be overruled.**

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

## Status

Machine built and wired; **not yet trading**. Specs 01–07 written, 01–06
implemented, 2,198 tests green (pytest / ruff / mypy; frontend eslint / tsc).
Pre-flight passes every hard gate that can be checked from code.

The whole loop runs end to end against the live paper account: perceive, screen,
enumerate a menu, ask the model, size, gate, submit, journal, reconcile, and
evaluate exits on every slot. What it has not done is place an order — the first
real submission will be the first trading day.

Still open, in priority order:

| | |
| --- | --- |
| **Strategy** — specs/07 D4 (direction → structure, condor on disagreement) and D5 (strike selection by delta target) are unimplemented. The live path builds fixed-width put credit spreads only, whatever the trend says. | blocked on strategy research |
| **Backtest** — spec 08 is not written. [specs/00](specs/00-brief.md) says this is what turns "up 2% in four days" into a claim about edge. | blocked on the above |
| **First live submission** — the multi-leg mapping, the partial-fill breach and the timeout read-back are tested offline and have never met a broker. | 28 Aug |

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

### Running the agent

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

- **Live** — health, money, positions with a bar showing each one travelling
  between its stop and its target, room left before the Gate refuses its own
  next proposal, and the day's cycle counts. Polls every 15 seconds.
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
| [ai_quant_researcher/](ai_quant_researcher/) | A separate system sharing the repository: an equities strategy-research lab that implements [specs/trading_strategy_architecture.md](specs/trading_strategy_architecture.md). |

`ai_quant_researcher/` is a **sibling project, not part of AlphaGate**. It has
its own `pyproject.toml`, its own virtualenv, its own test suite, and it imports
nothing from `alphagate`. The two answer different questions — AlphaGate asks
"should this options order be allowed through", the researcher asks "does this
equities rule have an edge that survives out of sample" — and they are kept
apart so neither's invariants leak into the other.

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
