# 09 — Equity execution: the handoff, and what happens after it

The chain this spec closes:

```
aqr research → walk-forward → pre-register → sealed run → target book (a file)
                                                              │
                                                              ▼
                                            AlphaGate: load → plan → gate → submit
                                                              │
                                                              ▼
                                                journal + status.json → dashboard
```

Spec 08 stays reserved for the options backtest. This is the equity path, and it
exists because [`ai_quant_researcher/`](../ai_quant_researcher/) produces a
validated *equities* strategy and then stops — deliberately. Its
`CONSUMER_MUST_SUPPLY` names six things it will not do, and every one of them is
a decision below.

## D0 — The seam is a file, and it stays a file

`ai_quant_researcher` does not import `alphagate`. `alphagate` does not import
`aqr`. Neither knows the other's package exists. The interface is a target-book
JSON artefact on disk, and `tests/test_boundaries.py` fails the build if either
direction of import appears.

That is not fastidiousness. The two projects answer different questions and hold
different invariants — the researcher holds no money and places no orders, so it
has no `Decimal` rule and no Risk Gate; AlphaGate holds both. An import would
make one project's invariants the other's problem, and
[CLAUDE.md](../CLAUDE.md) §2b already says not to reconcile them.

The file is also what makes the claim auditable. A book carries its own
fingerprint, seal state, sealed-run measurement and both multiplicity
denominators, so a reader who has never seen `aqr` can check where the weights
came from.

## D1 — Only a book that earned it may be loaded

`load_target_book` is pure — it takes a parsed mapping, not a path — and it
refuses in seven ways. All seven are checked, and every fault is reported at
once, because a book that is wrong in three ways should not be fixed three
times.

| Refusal | Why |
| --- | --- |
| `schema_version != 1` | fields may mean something new under the same number |
| `spec_fingerprint` ≠ the pinned one | the pin is what makes "only this strategy" true |
| `provenance.status` not in `{PAPER, LIVE}` | a CANDIDATE has not earned paper money |
| `sealed_look < 1` | the seal is unspent; the rule still owes an out-of-sample verdict |
| `sealed_measurement.refuted` | the sealed window refuted it |
| any weight < 0 | a short leg needs a locate and a different risk shape (D6) |
| `gross` > 1.0 + tolerance | leverage the researcher never measured |

**The fingerprint is pinned in configuration, not read from the book.** A book
names the strategy it describes; if the backend believed that name, swapping the
file would swap the strategy and nothing would notice. So the operator pins
`ALPHAGATE_STRATEGY_FINGERPRINT` once and a book that disagrees is refused by
name. This is the whole of "only the strategy `ai_quant_researcher` provides".

`provenance.status` comes from the registry's `CANDIDATE → PAPER → LIVE` state
machine, which refuses to skip a step. Reading it here means the lifecycle rule
is enforced on the execution side too, rather than trusted.

**The pin refuses silently, so it also has to report.** `find_latest_book` globs
on the pinned fingerprint, which means a book for a strategy the pin does not
name is not merely refused — it is never looked at. "No target book" and "your
pin is stale" come out as the same sentence, and the second one sends the reader
to source to find out which they got.

So when **no** book matches the pin, `equity-preflight` and the `NO_BOOK` journal
line list what *is* on disk under some other fingerprint — newest per strategy,
name and `as_of` and file — and name the one line that would admit it:
`ALPHAGATE_STRATEGY_FINGERPRINT` in `.env.local`.

**When the pinned book is found, this is silent**, however many other strategies
sit beside it. That is what a research repository looks like on an ordinary day.
There is deliberately no cleverness about which book is *newer*: a target book
stays in force for as long as the strategy does, so its `as_of` is the session
the weights describe and not a freshness signal, and a rule built on comparing
them would fire on healthy mornings — which teaches its reader to skim past the
one morning it is real.

It reports and stops there. Nothing changes the pin, nothing offers to, and
`equity-preflight` does not fail on it. Changing which strategy this account
executes is a person's decision, and a system that made it automatically would
have deleted the only property the pin exists to provide.

## D2 — Weights become shares, and the executor owns every step

The book carries weights. Turning one into an order needs four things it
deliberately does not have: equity, a price, what is already held, and a policy.

```
target_notional[s] = weight[s] × equity
target_shares[s]   = target_notional[s] / price[s]
delta_shares[s]    = target_shares[s] − held_shares[s]
```

Rounded per asset: fractional to 4 dp when Alpaca reports the asset
`fractionable`, floored to a whole share otherwise. A whole-share floor that
produces zero is a **skip with a journalled reason**, not a silent drop — on a
$100k account a 0.19% sleeve position is $192, and a non-fractionable $500 name
rounds to nothing. That is a real hole in the book and the record should say so.

`price` is the snapshot mid, or the last trade when there is no two-sided quote.
Never the book's own bars: the book is as of yesterday's close and the order
fills today.

## D3 — The no-trade band, and why it is proportional

The strategy rebalances every 5 sessions and holds 10 core names. Between
rebalances the target weights do not change, but the *held* weights drift with
price, so a naive diff would trade every name every day and pay costs the
backtest never paid.

So an intent is emitted only when

```
|delta_notional| ≥ max(drift_band_pct × max(target_notional, held_notional),
                       min_order_notional)
```

Default band 20% of the **position**, floor $25. Both are executor policy, both
are recorded in the journal beside the plan, and neither is a strategy
parameter — the strategy is what `aqr` validated, and this is the cost of
placing it.

**The first version made the band a fraction of equity, and it was wrong in a
way worth recording.** 0.25% of a $100k account is $253. The book's sleeve
positions are 0.192% of equity — $194 each — so every sleeve position was
permanently inside the band: the hundred names the strategy holds could never be
established, the ten core names would have been bought into an otherwise-cash
account, and the result would have been a tenth of the strategy with nothing
reporting an error. It was visible only because `equity-status` prints what is
*outside* the band, and the sleeve was not in the list.

A proportional band cannot fail that way, because the threshold for establishing
a position is a fraction of the position rather than a constant it might be
smaller than. Measured against the larger of target and holding, so one rule
covers establishing, drifting and exiting.

## D4 — Sells before buys, and a deterministic order

The plan is a sequence, and the sequence is part of the contract:

1. **sells first**, largest notional first, ties broken by symbol;
2. **buys second**, largest notional first, ties broken by symbol.

Sells release buying power that buys then spend. Placing them the other way
round on a fully-invested book means the first buys are refused for want of cash
and the last sells leave the account holding it — a rebalance that half-happened
and is worse than either end state.

Ties broken by symbol so the same inputs produce the same sequence. Determinism
is [CLAUDE.md](../CLAUDE.md) §3 rule 7 and it is what lets a plan be replayed.

## D5 — The equity Risk Gate

The options Gate does not apply — its checks are about short strikes, DTE and
defined-risk width, none of which an equity order has. `CONSUMER_MUST_SUPPLY`
says so explicitly. So there is a second gate, with the same shape and the same
guarantees: pure, no clock, no I/O, no model, every check runs, no
short-circuit, `CHECKS` is a tuple and that tuple is the order.

| Check | Refuses |
| --- | --- |
| `book_is_pinned` | a plan whose book fingerprint is not the pinned one |
| `book_is_fresh` | a book older than `max_book_age_days` sessions |
| `symbol_is_tradeable` | an asset the broker reports untradable or halted |
| `price_is_fresh` | a quote older than `max_quote_age` seconds |
| `order_is_material` | a notional under the floor — D3, asserted again here |
| `position_cap` | a resulting position over `max_position_pct` of equity |
| `gross_exposure_cap` | a resulting gross over `max_gross` |
| `buying_power` | a buy whose notional exceeds available buying power |
| `daily_turnover_cap` | today's traded notional over `max_daily_turnover_pct` |
| `daily_order_cap` | more than `max_daily_orders` orders today |
| `no_short_selling` | any sell that would take the position below zero |
| `drawdown_killswitch` | drawdown past the limit, or the latch already set |

Boundaries are inclusive on the safe side: a value exactly at its limit passes.

**A sell that reduces or closes is never blocked by a budget.** Same rule as
[03](03-risk-gate.md) D4's exit waiver, for the same reason: the checks are
computed so the dashboard has the numbers, and a failure on a risk-reducing
order is waived rather than turned into a veto. `no_short_selling` and
`symbol_is_tradeable` are not waivable — one would open a short leg, the other
cannot be filled.

## D6 — No short selling, and the structure that makes it unexpressible

`direction: market_neutral` in the DSL means the *ranking* is cross-sectional.
The book it produced holds 104 long positions and no negative weight. A short
equity leg needs a locate, accrues borrow, and has unbounded loss — a different
risk shape entirely, and one nothing here has measured.

So `EquityOrderIntent` carries `shares: Decimal` and a `side`, and the planner
never emits a sell larger than the held quantity. D1 refuses a negative weight
at load; the planner cannot construct one; the Gate checks it anyway. Three
layers, same as [02](02-options-domain.md) D3's treatment of naked options.

## D7 — The one door, again

```python
def submit_equity(order: GatedEquityOrder, mcp: McpSession) -> Submission
```

`GatedEquityOrder` is minted only inside `risk.equity_gate`, enforced by the
same frame check that guards `GatedOrder` ([03](03-risk-gate.md) D3), and
`tests/test_boundaries.py` guard 5 is extended to cover **both** gated types and
**both** doors. A second execution path added without a second guard is the
failure mode this whole spec is arranged against.

Orders are `market`, `day`, `extended_hours: false`. A limit order on a
rebalance is a rebalance that may not happen: the book is a set of weights to be
holding, not a price to be got, and an unfilled leg leaves the account in a
state neither the backtest nor the plan describes. Fractional quantities require
market/day anyway.

`client_order_id` is derived from `(fingerprint, as_of, symbol, side, trading
day)`. A timeout is resolved by reading it back, never by resending —
[04](04-execution.md) D4, unchanged.

## D8 — The cadence

`equity-run` is a long-lived process:

* **one rebalance pass per book**, the first no earlier than a configured offset
  after the open (default 15 minutes — enough for the opening auction to settle,
  early enough that the book is not a lunchtime book);
* **a heartbeat every 30 seconds** that re-reads account and positions, re-marks
  the book, and rewrites `equity-status.json`, so the dashboard is live even on
  the four days in five when the plan is empty;
* **a reconcile pass** that reads back every order the session submitted and
  amends its journal line with the fill.

**The pass is idempotent by book, not by day.** The guard is the journal, keyed
on the book's digest: a restart at 14:00 finds today's line for the book on disk
and does not replay it, while a book *regenerated* at 14:00 with different
weights is a different instruction and gets its own pass.

Keying on the day was the first version and it was wrong for this project.
Research is the point of the system, and a strategy that improves at eleven
o'clock should not have to wait for tomorrow's open to be held. What must not
happen is the same instruction being executed twice, and the digest is precisely
that instruction's identity — the fingerprint is the *strategy*, the `as_of` is
the *session*, and neither distinguishes two books that disagree.

Nothing here is a rate limit, and it deliberately is not one. A second pass is
bounded by the Gate: `max_daily_orders` and `max_daily_turnover_pct` count
across the whole session (D5), so the second pass spends a budget the first has
already drawn on. `max_daily_turnover_pct` is 1.20 for exactly this reason —
one full build plus a rebalance, and a third pass runs out of room. A
regeneration loop is stopped by cost, which is measurable, rather than by a
clock, which is arbitrary.

Whether a pass settles its book is decided by an allowlist of **deciding
stages**, so a stage nobody has classified is quiet rather than terminal. Two
are deliberately outside it: `NO_BOOK`, where the artefact was missing or
unreadable, and `NO_MARKS`, where the pass could not read a usable price for any
symbol in it. Neither is an opinion about the book, and both are fixed by looking
again a little later. Every other stage settles it.

`NO_MARKS` exists because without it a blind pass is indistinguishable from a
quiet one. Both produce an empty plan with every symbol accounted for; the
difference is whether the account of each symbol reads "already held" or "no
usable quote", and only the first is an answer. On 2026-08-31 a pass ran four
minutes before the open against a feed that does not tick pre-market, skipped
all 87 names as `stale_mark`, recorded `NO_TRADES`, and closed the session. The
book was never built.

Blindness is total rather than proportional: a pass is blind only when *every*
skip is `no_mark` or `stale_mark`. Names go stale one at a time all session and
that is ordinary trading, so a single readable price is enough to make the pass
a decision — no threshold, and nothing to tune. `not_tradeable` is not
blindness: a halted symbol is a price we could read and a market we cannot
reach, and waiting thirty seconds is not the remedy.

## D9 — What lands on disk

| Path | What |
| --- | --- |
| `journal/YYYY-MM-DD.jsonl` | the same file. Equity cycles carry `kind: "equity"` |
| `journal/equity-status.json` | the live equity book, rewritten every heartbeat |
| `journal/books/<fingerprint>-<as_of>-<digest12>.json` | the book as loaded, copied verbatim |

The book is copied rather than referenced. `aqr` regenerates
`runs/target_books/` daily; a journal line pointing at a path whose contents
have since changed is a record of nothing. The copy is hashed and the digest
goes in the journal line, so a book on disk that no longer matches what was
executed can be identified as such.

The digest is in the archive's **filename** too, and it has to be. `aqr` names
its output by session, so a book regenerated for the same `as_of` overwrites its
own file upstream — and an archive that copied only fingerprint and session
would overwrite the evidence of the pass that ran before it, leaving a journal
line naming a digest no file on disk had. That was survivable while a session
held one pass. Now that D8 admits a second, it is not. Two books, two files; the
same book archived twice is still one file, because the same bytes cannot have
two digests.

A cycle line carries the plan, every intent, every intent's verdict with the
full check tape, and the submission. `NO_TRADES` is journalled like any other
outcome — "why did it not rebalance today" is the question the journal exists
for, and on this strategy it is the answer four days in five.

## D10 — The dashboard shows the provenance, not just the positions

The Live tab gains an **Equity** view with four blocks:

* **the strategy** — name, fingerprint, `as_of`, and the sealed-run numbers:
  alpha, beta, `t`, information ratio, the looks count and the significance bar
  it had to clear. This is the block that makes the demo a claim rather than a
  screenshot;
* **the book** — target weight against held weight per symbol, the drift, and
  whether it is inside the no-trade band;
* **today** — every intent, its verdict, and its fill;
* **the gate** — the check tape, tightest-first, same as the options view.

Read from `equity-status.json` and the journal. `alphagate.interface` still
imports neither `execution` nor `live` nor `marketdata`, so there is still no
code path from a browser to an order.

## Test plan (RED first)

1. `load_target_book` refuses each of D1's seven faults, and reports several at once.
2. A book with the wrong fingerprint is refused even though it is otherwise valid.
3. `plan_rebalance` is deterministic: same inputs, same sequence, byte-identical.
4. Sells precede buys, and both are ordered by notional then symbol.
5. A drift inside the band emits no intent; one exactly at the band emits one.
6. A non-fractionable asset whose target rounds to zero produces a skip with a reason.
7. Each Gate check vetoes on its own and passes at its exact boundary.
8. A reducing sell is waived past a budget veto; it is not waived past `no_short_selling`.
9. `GatedEquityOrder` cannot be constructed outside `risk.equity_gate`.
10. `submit_equity` refuses anything that is not a `GatedEquityOrder`.
11. A timeout produces a read-back, never a second `place_stock_order`.
12. The whole chain runs offline from a fixture book and a `RecordedSession`.
