<!-- Language: **English** · [简体中文](ARCHITECTURE.zh-CN.md) -->

*Language:* **English** · [简体中文](ARCHITECTURE.zh-CN.md)

# Architecture and workflow

This document explains **how the system is put together and what happens when it
runs**. For "what is this and how do I start it", read the
[README](../README.md). For the contracts each piece must satisfy, read
[`specs/`](../specs/).

---

## 1. The shape of the whole thing

Two independent systems share this repository. They answer different questions,
hold different invariants, and are connected by exactly one thing: a file.

```mermaid
flowchart LR
    LAB["<b>ai_quant_researcher</b> — the laboratory<br/><br/>an LLM proposes a falsifiable hypothesis<br/>a deterministic pipeline tries to destroy it<br/>one survivor, validated out of sample<br/><br/><i>holds no money · places no orders</i>"]
    FILE[("<b>target book</b><br/>weights + provenance<br/>a JSON file")]
    DESK["<b>AlphaGate</b> — the trading desk<br/><br/>load, and refuse anything unvalidated<br/>weights → shares<br/>the Risk Gate<br/>the broker<br/><br/><i>holds money · places orders</i>"]

    LAB --> FILE --> DESK
```

The laboratory holds no money and places no orders. The desk holds both.

**Why they are separate.** The laboratory has no `Decimal` rule because it holds
no money, and no Risk Gate because it places no orders. The desk has both. An
import in either direction would make one system's invariants the other's
problem. `backend/tests/test_boundaries.py` fails the build if `alphagate`
imports `aqr`, if `aqr` imports `alphagate`, or if the pipeline driver that runs
both imports either.

**Why they are connected by a file.** A file can be hashed, copied, committed,
and read a year later by a program that has never heard of either project. The
target book carries the strategy's fingerprint, its seal state, its sealed
measurement and both multiplicity denominators — so a reader can audit where the
weights came from without trusting this document.

---

## 2. The laboratory: from a guess to a validated strategy

```mermaid
flowchart TB
    A["LLM researcher<br/><i>proposes DSL fields, never code</i>"]
    B["Strategy DSL<br/><i>parsed, whitelisted, content-hashed</i>"]
    C{"Validator<br/><i>does it ever fire?</i>"}
    D["<b>Try to destroy it</b><br/>backtest, next-bar fills and real costs<br/>walk-forward, judged on unseen data<br/>robustness over parameters, assets, regimes<br/>residual alpha, regressed on the benchmark<br/>overfitting, charged against search cost"]
    I{"Evaluator<br/><i>gates before weights</i>"}
    K["Pre-register<br/><i>selection rule recorded first</i>"]
    L["Sealed run<br/><i>ONE shot, on data never read</i>"]
    M{"Refuted?"}
    N["Registry: PAPER"]
    O["Registry: REJECT<br/><i>kept, never deleted</i>"]
    P[("Target book")]

    A --> B --> C
    C -->|"dead rule, one repair turn"| A
    C -->|"alive"| D --> I
    I -->|"REJECT / REVIEW"| O
    I -->|"ACCEPT"| K --> L --> M
    M -->|"yes"| O
    M -->|"no"| N --> P
```

### The three things that make this more than a backtest

**The LLM emits a hypothesis, never code.** An expression like
`close <= ema(20) * 1.01 and rsi(14) > 40` is tokenised into an AST whose only
leaves are numbers and names from a feature registry. There is nothing to
sandbox because there is nothing dangerous to express.

**Next-bar fills, with no override.** A decision made from bar *t* fills at bar
*t+1*'s open, always. There is no configuration flag that relaxes it, because a
look-ahead switch is a look-ahead bug with a rationale attached.

**The last two years are sealed.** They are physically absent from the search
cache, the taint bit lives on the `Bars` type rather than on each provider, and
an append-only ledger records what was read. Each candidate gets **one** run
against that window, and the count of candidates screened is charged as a
multiplicity penalty — a `t` of +2.22 clears the bar as the first look and would
not clear it as the seventh.

### What the sealed window can and cannot say

```mermaid
flowchart LR
    W["498 sessions<br/>2024-09 → 2026-08"]
    W --> R["CAN refute<br/><i>the rule stopped working</i>"]
    W --> C["CANNOT confirm<br/><i>SE on annualised Sharpe ≈ ±0.71</i>"]
    W --> D["CANNOT prove the embargoed<br/><i>period</i> did not inform a decision<br/><i>— only that the data was not read</i>"]
```

`can_confirm` is a property that returns `False` by construction. The dashboard
prints that sentence beside the numbers, because a page that said "confirmed"
would be claiming something nobody measured.

---

## 3. The desk: from a file to a share order

### 3.1 One rebalance pass, end to end

```mermaid
sequenceDiagram
    autonumber
    participant R as run
    participant D as disk
    participant Q as quotes
    participant PL as planner
    participant G as Gate
    participant B as broker

    R->>D: newest book for the PINNED fingerprint
    D-->>R: bytes
    R->>R: hash, parse, refuse in 7 ways
    R->>D: copy the book in, verbatim
    R->>B: account + positions
    B-->>R: equity, cash, holdings
    R->>Q: snapshots, 200 symbols per request
    Q-->>R: mid, or last trade
    R->>PL: book, holdings, marks, equity, policy, as_of
    PL-->>R: intents, sells first + a reason per skip

    loop each intent, in order
        R->>G: intent, book, portfolio, policy, as_of
        alt approved
            G-->>R: GatedEquityOrder + check tape
            R->>B: place_stock_order, market, day
            B-->>R: submission
            R->>R: advance the portfolio snapshot
        else vetoed
            G-->>R: reasons + check tape
        end
    end

    R->>D: one journal record
```

Two details in that loop are load-bearing.

**The snapshot is advanced after each order.** A pass that judged every order
against the account as it stood at the top would let a hundred orders each pass
a daily-turnover check that the hundred of them together fail — the cap would
bind between passes and not within one, which is not what it says.

**A transport failure stops the pass rather than skipping an order.** What was
placed stands and is journalled; the rest are simply not attempted, and the next
pass re-derives them from what the broker actually holds — the one source that
cannot be wrong about what happened.

### 3.2 How a weight becomes a number of shares

```mermaid
flowchart TB
    W["target weight<br/>e.g. 0.08"]
    E["account equity<br/>e.g. $100,000"]
    P["mark<br/>quote mid, or last trade"]
    H["held shares<br/>read from the broker"]

    W --> TN["target notional<br/>= weight × equity"]
    E --> TN
    TN --> DR["drift = target − held value"]
    H --> HV["held value = shares × mark"]
    P --> HV
    HV --> DR

    DR --> Q{"drift big enough?"}
    Q -->|"no"| SK["skip: INSIDE_BAND<br/><i>journalled with the reason</i>"]
    Q -->|"yes"| SH["shares = drift / mark"]
    SH --> RD{"fractionable?"}
    RD -->|"yes"| F4["round toward zero, 4 dp"]
    RD -->|"no"| FI["floor to a whole share"]
    FI --> Z{"= 0?"}
    Z -->|"yes"| SK2["skip: ROUNDS_TO_ZERO<br/><i>a real hole in the book, recorded</i>"]
    Z -->|"no"| OUT["OrderIntent"]
    F4 --> OUT
```

**The band is proportional to the position, not to the account.** The threshold
is `max(20% × max(target, held), $25)`, measured against the larger of the two
so that one rule covers establishing a position, letting it drift, and exiting
it.

The first version made it a fraction of equity — 0.25%, which is $253 on a $100k
account. The book's sleeve positions are $194 each, so every one of the hundred
sleeve names sat permanently inside the band and could never be established. The
executable book would have been the ten core names in an otherwise-cash account:
a tenth of the strategy, with nothing reporting an error.

**A symbol the book no longer wants is sold to zero by the same arithmetic.** Its
target weight is absent, so its target notional is zero, so its drift is the
whole position. There is no separate exit rule, and therefore none to forget to
call.

### 3.3 The two Gates

```mermaid
flowchart TB
    subgraph OPT["Options path"]
        OP["TradeProposal"] --> OG["risk.gate.evaluate<br/><i>13 checks</i>"]
        OG -->|approved| OO["GatedOrder"]
        OO --> OS["execution.submit"]
        OS --> OA["place_option_order"]
    end

    subgraph EQ["Equity path"]
        EP["OrderIntent"] --> EG["risk.equity_gate.evaluate_equity<br/><i>12 checks</i>"]
        EG -->|approved| EO["GatedEquityOrder"]
        EO --> ES["execution.submit_equity"]
        ES --> EA["place_stock_order"]
    end

    OG -.->|vetoed| J["journal — with the full check tape"]
    EG -.->|vetoed| J
```

They are **not** the same Gate with different numbers. The options Gate judges
short strikes, days to expiry and defined-risk width; none of those exist for a
share order. The equity Gate judges concentration, gross exposure, buying power,
turnover, order count and drawdown; none of those are what a spread is about.

What they *do* share is the discipline, and it is identical in both:

| Rule | Meaning |
| --- | --- |
| Pure | stdlib only. No clock, no I/O, no network, no model. |
| Total | every check runs; there is no short-circuit on the first veto. |
| Deterministic | the check tuple *is* the order of `checks` and `reasons`. |
| Inclusive boundaries | a value exactly at its limit passes. |
| One minting site | the gated type is constructible in exactly one module. |
| Exits are waived | a risk-reducing order is not blocked by a budget. |

### 3.4 The one door, twice

```mermaid
flowchart LR
    X1["anything else"] -.->|"TypeError"| DOOR
    G1["risk.gate<br/><i>the only place a<br/>GatedOrder is minted</i>"] --> DOOR["execution.submit*"]
    G2["risk.equity_gate<br/><i>the only place a<br/>GatedEquityOrder is minted</i>"] --> DOOR
    DOOR --> BROKER[("Alpaca")]
```

`GatedOrder.__post_init__` walks the call stack and refuses construction unless
the calling frame belongs to `alphagate.risk.gate`. `GatedEquityOrder` does the
same for `alphagate.risk.equity_gate`. Frame inspection is unusual and chosen
deliberately: a module-private token can be imported, a convention can be
forgotten on day five, and a code review cannot be run by CI.

Two consequences worth knowing before they surprise you: `copy.deepcopy`,
`pickle` and `dataclasses.replace` of a gated order all raise. That is correct —
an order that can be cloned into a second order is an order that can be
submitted twice.

The static half lives in `test_boundaries.py`, which scans every function in
`execution/` whose name starts with `submit` and asserts that its first
parameter is one of the gated types, and that every gated type has exactly one
door. The first version of that guard knew only the exact word `submit`, so
`submit_equity` would have walked straight past it.

---

## 4. Layering, and what enforces it

```mermaid
flowchart TB
    subgraph PURE["PURE — stdlib only · no clock · no I/O · no network · no model"]
        direction LR
        CORE["core<br/><i>indicators, structure,<br/>levels, trend</i>"]
        OPTS["options<br/><i>contracts, greeks,<br/>structures, risk</i>"]
        RISK["risk<br/><i>both Gates</i>"]
        EQTY["equity<br/><i>book, planner, policy</i>"]
    end

    AGENT["agent<br/><i>THE ONLY LAYER<br/>THAT MAY CALL AN LLM</i>"]
    EXEC["execution<br/><i>MCP adapter, the doors</i>"]
    MKTD["marketdata<br/><i>REST, GET only</i>"]
    LIVE["live<br/><i>composition root — the only<br/>module that knows a real<br/>account exists</i>"]
    JRNL["journal<br/><i>append-only JSONL</i>"]
    IFCE["interface<br/><i>dashboard</i>"]

    AGENT --> PURE
    EXEC --> RISK
    MKTD --> CORE
    LIVE --> AGENT
    LIVE --> EXEC
    LIVE --> MKTD
    LIVE --> JRNL
    IFCE --> JRNL
    IFCE -. "MUST NOT import<br/>execution · marketdata · live" .-x LIVE
```

Every arrow above is checked by a test rather than trusted:

| Guard | What it refuses |
| --- | --- |
| 1 | a pure layer importing anything but the standard library and its siblings |
| 2 | an LLM SDK anywhere outside `agent/` |
| 3 | a network library in a pure layer |
| 4 | `Decimal` built from a float literal |
| 5 | a gated type minted outside its one module, or a door taking an ungated type |
| 6 | a write verb in `marketdata` — it may only issue GETs |
| 7 | the model's self-reported confidence reaching sizing or gating |
| 8 | `interface` importing `execution`, `marketdata` or `live` |
| 9 | either project importing the other, or the pipeline driver importing either |

Guard 8 is the one that matters on demo day: **there is no code path from a
browser to an order.** The dashboard learns the live book from a JSON file the
agent writes, which is also why it fails honestly — if the agent stops, the file
stops being rewritten and the page says *not running* instead of showing a stale
book with a confident face.

---

## 5. The record

```mermaid
flowchart TB
    C1["cycle decides"] --> W1["append one JSONL line"]
    F1["a fill arrives<br/>ten minutes later"] --> W2["append an amendment line<br/>keyed by cycle_id"]
    W1 --> FILE["journal/YYYY-MM-DD.jsonl"]
    W2 --> FILE
    FILE --> RD["read: apply amendments<br/>in file order"]
    RD --> OUT["the day, as it was decided,<br/>plus what became of it"]
```

**Amendment, not mutation.** A record is written once. Later facts arrive as
separate lines, and reading applies them in file order — so the original
decision stays exactly as it was made and no hindsight leaks backwards into it.

**Nothing here contains a credential.** Every line is passed through `redact` on
the way out, which strips by key name, by shape (`PK…`, `sk-…`, `PA3…`), and by
containing object — the last of those because Alpaca returns an account id as a
bare `id`, and an *order* also has a bare `id` that reconciliation needs. No
regex can tell two UUIDs apart; only the object around them can.

**Quiet cycles are journalled too**, and on the equity strategy they are the
majority: it rebalances every five sessions, so four days in five the honest
answer is "the book is already held". A journal that only contained trades could
not say so.

What lands on disk:

| Path | What |
| --- | --- |
| `journal/YYYY-MM-DD.jsonl` | one line per cycle, both agents, append-only |
| `journal/books/` | every target book that was actually executed, byte-for-byte |
| `journal/status.json` | the options agent, right now |
| `journal/equity-status.json` | the equity book, right now |
| `journal/state.json`, `journal/equity-state.json` | high-water equity and the kill-switch latch, across days |

---

## 6. The daily loop

```mermaid
flowchart TB
    S(["session opens"]) --> RF["refresh<br/><i>pull bars through yesterday</i>"]
    RF --> BK["book<br/><i>re-run the validated strategy</i>"]
    BK --> CK{"market open?"}
    CK -->|"no"| Q["say so, journal it, wait"]
    CK -->|"yes"| WT["wait for open + 15 min"]
    WT --> PS["one rebalance pass"]
    PS --> HB["heartbeat every 30s<br/><i>re-read account, re-mark book,<br/>rewrite status.json</i>"]
    HB --> HB
    HB --> CL(["session closes"])
```

`python scripts/pipeline.py` runs all three stages. The driver executes both
projects' CLIs as subprocesses and imports neither.

**Fifteen minutes after the open**, because the opening auction needs to settle
before a snapshot mid is a price rather than an artefact — and because a book
placed at lunchtime is a different book from the one the strategy decided on.

**The heartbeat is most of what the process does**, and it is not decoration. On
a strategy that rebalances every five sessions, a process that only woke to
trade would be indistinguishable from one that had died.

**The stage that is deliberately absent is `research`.** A campaign burns looks
against the sealed window — the multiplicity denominator every claim in this
repository is deflated by — and a nightly cron that quietly screened another
seven candidates would invalidate the `t` printed on the dashboard without
anyone noticing. Running one is a decision a person makes.

---

## 7. Repository map

| Path | What it is |
| --- | --- |
| `backend/src/alphagate/core/` | deterministic market analysis. Extracted from an existing project; **do not rewrite** ([adr/0001](../adr/0001-core-reuse.md)) |
| `backend/src/alphagate/options/` | contracts, greeks, structures, risk. Pure |
| `backend/src/alphagate/equity/` | target book, planner, executor policy. Pure |
| `backend/src/alphagate/risk/` | both Gates, both verdict types. Pure |
| `backend/src/alphagate/agent/` | perception, menu, prompt, proposer. **The only LLM layer** |
| `backend/src/alphagate/execution/` | MCP adapter, idempotency, both doors |
| `backend/src/alphagate/marketdata/` | REST market data. GET only |
| `backend/src/alphagate/journal/` | append-only record, redaction, reconciliation |
| `backend/src/alphagate/live/` | composition roots and the CLI |
| `backend/src/alphagate/interface/` | the dashboard. Imports the journal and nothing else |
| `ai_quant_researcher/src/aqr/` | the laboratory. Imports nothing from `alphagate` |
| `frontend/` | the dashboard's React app, built into the Python package |
| `scripts/pipeline.py` | the three-stage driver. Imports neither project |
| `specs/` | the contracts, written before the code they govern |
| `adr/` | decisions, and the reasoning behind them |
| `journal/` | the record the submission ships |

---

## 8. Where to read next

| Question | File |
| --- | --- |
| How do I run this? | [README](../README.md) |
| What must the Risk Gate refuse? | [specs/03](../specs/03-risk-gate.md) |
| How does an order reach Alpaca? | [specs/04](../specs/04-execution.md) |
| What is in the journal, and why? | [specs/06](../specs/06-journal.md) |
| How does a validated strategy become positions? | [specs/09](../specs/09-equity-execution.md) |
| How is a strategy validated in the first place? | [ai_quant_researcher/README](../ai_quant_researcher/README.md) |
| Why is `core/` reused rather than written? | [adr/0001](../adr/0001-core-reuse.md) |
| Why MCP for orders and REST for data? | [adr/0002](../adr/0002-execution-via-mcp.md) |
