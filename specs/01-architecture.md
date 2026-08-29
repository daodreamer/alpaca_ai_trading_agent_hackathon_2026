# 01 — Architecture and the core boundary

## Layers

```
alphagate/
  core/          PURE. Extracted from Personal Market Monitor. Stdlib only.
                 Bars, indicators, structure, levels, trend. Deterministic.
  options/       PURE. Options domain: contracts, structures, greeks budget.  → 02
  risk/          PURE. The Risk Gate. Proposal in, verdict out.               → 03
  agent/         Orchestration. Perception → proposal → gate → execution.
                 The only layer that talks to an LLM.                         → 05
  execution/     Alpaca Trading API adapter: orders, account, positions.
                 Owns the MCP transport (`stdio.py`) and the paper guard.     → 04
  marketdata/    Read-only Alpaca REST adapter, plus a recorded-payload seam
                 the tests and the backtest replay. Writes nothing.
  journal/       Decision records. Append-only, one per cycle.                → 06
  infra/         Clocks and exchange calendars. The impure edges core needs.
  live/          The composition root. Assembles the tested parts into a
                 running agent, and the `python -m alphagate` CLI.
  interface/     FastAPI + the dashboard. Read-only over the journal and the
                 status snapshot; cannot reach a broker.
```

The last two are the only packages that know a live Alpaca account exists.
Everything above them is tested offline, which is why the suite needs no
network and no subprocess.

Dependency direction is strictly downward. `core`, `options` and `risk` import
nothing from the layers above them and nothing outside the standard library.

## Rule 1 — the LLM lives in exactly one layer

`agent/` is the only package permitted to import an LLM SDK or construct a
prompt. If a model call appears in `risk/`, the Gate is no longer a gate.

## Rule 1b — orders leave through one door

Order submission goes through the MCP server and nowhere else
(adr/0002-execution-via-mcp.md D1). Market data comes in over a direct REST
adapter, because the backtest replays recorded payloads and an MCP session does
not replay. The CLI writes nothing.

## Rule 2 — the Gate is deterministic and total

Every order reaching Alpaca passed the Gate. There is no bypass path, no
`force=True`, no debug flag. Enforced by a boundary test: `execution/` may only
be called with a `GatedOrder`, a type the Gate alone can construct.

## Rule 3 — no float money

Prices, premiums, and P&L are `Decimal` end to end, converted at the ingest
boundary. Inherited from core (ADR 0005 in the upstream project). Greeks and
IV are `float` — they are estimates, not money.

## Rule 4 — no look-ahead

Same rule the core already enforces. A structure that needs a future bar to
confirm is a *candidate*, not *confirmed*. The backtest — spec 08, not yet
written — must use the same code path as live, with the clock as the only
difference.

## The core boundary — what is reused, what is not

`alphagate.core` is a verbatim extraction of `pmm.domain` from Personal Market
Monitor, with the package renamed. It is pure (its only internal imports are
`pmm.domain.*`, verified before extraction) and arrives with its full test
suite (~11k lines) and its boundary guard test.

**In scope for reuse (the agent's perception layer):**

| Module | Role in the agent |
| --- | --- |
| `numeric`, `bar`, `symbol`, `time_model`, `identifiers`, `clock` | Decimal/UTC primitives the options model builds on |
| `normalization`, `aggregation`, `streaming` | Underlying bar cleaning, dedupe, timeframe folding |
| `indicators` | ATR, EMA, RSI, MACD, VWAP — inputs to the market read |
| `structure`, `level_engine`, `levels` | Swings, BOS, support/resistance zones |
| `trend_engine`, `trend`, `confluence` | Multi-timeframe trend state — the headline perception signal |

**Carried but unused for now:** `alerts`, `alert_engine`, `news`, `stores`,
`operations`, `accounts`. Extracted with the package because splitting a passing
test suite costs more than it saves. Pruned after the competition, not during.

**Explicitly not reused:** the upstream `application/`, `interface/`,
`persistence/` and provider layers. They are shaped for a monitoring product,
not an agent. The upstream Alpaca adapter is *market data only* (`data.alpaca.markets`,
GET-only transport); `execution/` is new work — see [04-execution.md](04-execution.md).

## Working discipline for a seven-day solo build

The core arrives strict (mypy strict, `filterwarnings = ["error"]`) and passing.
Keep it that way — it costs nothing, it is already written.

New layers: strict typing in `options/` and `risk/` (they are the correctness
surface), pragmatic elsewhere. Do not spend competition hours on type stubs for
the dashboard.
