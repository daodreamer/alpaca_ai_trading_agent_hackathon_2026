# AI Quant Researcher

> An automated quantitative research lab: an LLM proposes trading hypotheses, a
> deterministic pipeline tries to destroy them, and only what survives gets
> written down as a strategy.

This is an implementation of MVP v0.1 from
[`specs/trading_strategy_architecture.md`](../specs/trading_strategy_architecture.md)
— phases 1–7 of the recommended build order, plus the experiment database and
strategy registry that the later phases depend on.

The central claim of that architecture is worth restating, because every design
decision below follows from it:

**The LLM is a researcher, not a trader.** It never predicts a price, never
sizes a position and never emits code. It proposes a falsifiable hypothesis in a
constrained language. Everything after that is deterministic.

---

## What it does

```
                  ┌──────────────────┐
                  │  LLM Researcher  │  proposes a hypothesis as DSL fields
                  └────────┬─────────┘  (schema-constrained; cannot emit code)
                           ▼
                  ┌──────────────────┐
                  │   Strategy DSL   │  parsed, whitelisted, content-hashed
                  └────────┬─────────┘
                           ▼
                  ┌──────────────────┐
                  │    Validator     │  never fires? unsatisfiable? too few bars?
                  └────────┬─────────┘
                           ▼
                  ┌──────────────────┐
                  │   Backtester     │  next-bar fills, real costs, no look-ahead
                  └────────┬─────────┘  two engines: trigger, or ranked portfolio
                           ▼
                  ┌──────────────────┐
                  │  Walk-forward    │  fit on train, judged on unseen test
                  └────────┬─────────┘
                           ▼
                  ┌──────────────────┐
                  │   Robustness     │  parameters · assets · regimes · Monte Carlo
                  └────────┬─────────┘
                           ▼
                  ┌──────────────────┐
                  │  Residual alpha  │  regressed on the benchmark: what was added,
                  └────────┬─────────┘  separated from the exposure that was rented
                           ▼
                  ┌──────────────────┐
                  │   Overfitting    │  9 signals, charged against search cost
                  └────────┬─────────┘
                           ▼
                  ┌──────────────────┐
                  │    Evaluator     │  score 0–100, gates before weights
                  └────────┬─────────┘
                           ▼
              ACCEPT / PAPER / REVIEW / REJECT
                           │
                           ▼
                  ┌──────────────────┐
                  │    Registry      │  every experiment, including the failures
                  └──────────────────┘
```

---

## Quick start

```bash
uv venv && uv pip install -e ".[dev]"

uv run aqr features                          # the DSL vocabulary
uv run aqr backtest examples/trend_pullback.yaml
uv run aqr walkforward examples/trend_pullback.yaml
uv run aqr evaluate examples/trend_pullback.yaml
uv run aqr research --iterations 8           # the loop, offline, no API key
uv run aqr experiments                       # what has been tried
uv run aqr seal-check                        # prove the research cache holds no embargoed bars
uv run aqr costs --positions 110             # what one order costs, under each schedule
uv run aqr target-book FINGERPRINT           # the handoff. Nothing here places it
```

Everything runs offline by default against a seeded regime-switching price
simulator. To research on real bars, pull them into the cache once:

```bash
uv pip install -e ".[alpaca]"
uv run aqr pull --universe nasdaq50 --source alpaca --start 2016-01-01 --end 2026-08-27
uv run aqr research --source csv --universe nasdaq50 --iterations 40
```

`aqr pull` is the only command that reaches a network for market data, and it
checks everything it writes (see *Data you can research on*, below). Sources:
`alpaca` (consolidated tape, 2016 onward, intraday available), `ibkr` (deeper
history and non-US venues; needs TWS or IB Gateway running), `yahoo` (2010
onward, no credentials). `--source csv --csv-root <dir>` reads the cache.

To have a model propose the hypotheses:

```bash
uv pip install -e ".[llm]"
uv run aqr providers                          # which keys are visible (never prints one)
uv run aqr research --provider deepseek --iterations 10 --source yahoo
uv run aqr research --provider anthropic --model claude-opus-5
```

Keys are read from `.env.local` at the repository root (searched upward), then
`.env`. A variable already exported in the real environment always wins.

---

## The embargo, and why it is not a convention

The last two years are not available to the search. They are reserved for one
pre-registered validation run, and the separation is mechanical because a
convention does not survive a debugging session.

**The sensor is on the type, not on the I/O.** Every series in this system
becomes a `Bars` before anything can compute on it -- CSV, Yahoo, Alpaca, IBKR,
the simulator, a test fixture, a provider nobody has written yet. A check on
each provider covers the providers somebody remembered; a check in
`Bars.__post_init__` covers the route nobody anticipated, which is the only
route that was ever going to be the problem.

**The bit is monotone.** Clean to tainted, never back, no public setter. A flag
that can be cleared records the intention of whoever cleared it.

**The ledger says what was read.** The bit answers *whether*; an append-only
ledger with a hash chain answers *what*, so an audit is a query rather than an
act of trust. Editing a stored verdict without replaying the loads that produced
it does not reproduce the digest.

**The rows are not on the disk.** `data-sp500/` is physically truncated at
the embargo; `data-sp500-sealed/` holds the full history and only `SealedProvider`
can read it, which needs an explicit token *and* the sealed phase. `aqr
seal-check` verifies the claim without loading anything into the process doing
the checking:

```
root               files  latest bar  past embargo  canary
data-sp500         681    2024-08-30  none          armed (1D, 1h, 4h)
data-sp500-sealed  682    2026-08-27  598           -
```

**And a canary**, for the routes the first four do not anticipate: `__CANARY__`
is a symbol that exists only after the embargo, in the research root and nowhere
else. Nothing legitimate loads it, so its appearance anywhere is physical
evidence rather than an inference. It is reported by the audit and not counted
as a violation, because a tripwire placed where a peek cannot happen catches
nothing. The tripwire is armed in every timeframe — 1D, 1h and 4h — and the
audit reports them individually, because one armed timeframe must not read as
cover for the others. The bars are synthetic and written by
`scripts/pull_sp500_intraday.py` after each research pull, so a rebuilt cache
re-arms them; and they are committed, because a tripwire nobody can see is not
evidence.

**What none of it proves.** The seal shows the embargoed *data* was not read. It
cannot show the embargoed *period* did not inform a decision: the researcher
lived through it and every model in `aqr providers` has a training cutoff after
it. So the certificate records that exposure rather than denying it, and the
claim is worded as "the data was not read", never as "uncontaminated".

**The sealed window can refute, not confirm.** Two years is about 500 sessions,
where the standard error of an annualised Sharpe is around 0.7. An excess Sharpe
of 0.5 measured there is indistinguishable from zero. The evidence has to come
from walk-forward inside the search window; the sealed run is the check on
whether the rule has since stopped working.

**One shot is per candidate, not per window — and the looks are counted.** This
is the part that makes a research loop possible at all. The LLM proposes a
genuinely new hypothesis, that hypothesis gets its own sealed run, and the loop
continues; a protocol that burned the window on the first strategy would be a
one-strategy protocol. What is still refused, and always will be, is a *second*
run on the *same* fingerprint — re-running one rule until it passes is the thing
the whole apparatus exists to make impossible.

The cost of the second look is multiplicity, and it is charged rather than
ignored. A window that has screened seven candidates has performed a seven-way
selection, so the bar for the alpha rises with the count:

| looks | t the alpha must clear |
|---|---|
| 1 | 1.96 |
| 2 | 2.24 |
| 5 | 2.58 |
| 10 | 2.81 |
| 20 | 3.02 |

Bonferroni on a 5% family-wise rate. `+2.22` — what the promoted strategy
actually returned — clears the bar as the only candidate ever screened and does
not clear it as the seventh. Every sealed record carries its `looks`,
`significance_bar` and `alpha_clears_bar`, `aqr preregistered` prints the running
count, and every target book carries **two** denominators: the hypotheses the
search compared, and the candidates the sealed window has screened.

Nothing here forbids the seventh look. Counting it is the defence.

---

## Two engines, and why the second one exists

Fourteen strategies reached PAPER. All fourteen lost to buy and hold, and the
reason was structural rather than a bad rule.

The original engine simulates a trigger: something fires, a position opens, a
stop or a target or a bar count closes it, at most `max_positions` at a time.
That form is out of the market a great deal of the time, so it forfeits part of
the drift it is measured against. Winning on Sharpe then requires cutting
volatility by more than it cut return, which almost nothing does. The registry
bears this out by direction:

| direction | proposed | promoted | mean score |
|---|---|---|---|
| long | 69 | **14** | 50.1 |
| short | 177 | 0 | 14.5 |
| market_neutral | 37 | 0 | 20.7 |

The model was not refusing to propose market-neutral rules. It proposed 214 of
them and the scorer killed every one, because the scorer's heaviest component
was raw Sharpe and long-only rules collect beta for free.

So there is a second engine. It stays invested, ranks the universe, and holds
the top of it:

```yaml
strategy:
  mode: portfolio
  rank_by: rs_rank(60) - rs_rank(5)   # numeric, best first
  screen: rvol(20) > 0.8              # optional eligibility
  hold: 10
  rebalance_every: 21
  sleeve: {budget: 0.20, idle: benchmark}
```

Beta stays near one, so the drift is not given away, and the excess return is a
cross-sectional spread rather than a bet on being out at the right moments.
Breadth becomes the number of names rather than the number of concurrent
positions -- which matters, because every strategy in the registry ran with
`max_positions: 3` over 19 symbols, and three concurrent bets cannot produce a
measurable information ratio whatever the signal quality.

`rank_by` must be a number. `close > ema(20)` parses, runs, and sorts the book
into true and false -- an arbitrary two-tier ordering that looks exactly like a
ranking -- so the schema refuses it.

### The 80/20 split, and the mistake it nearly was

The core holds 80%; the sleeve holds 20% and is reserved for event-driven
deviations. What the sleeve does while idle is the whole design decision.

The obvious reading -- keep 20% in cash until an event -- is a trap. Equal-weight
NASDAQ-50 compounded at 26.6% a year over 2010–2026, so an idle fifth costs
about 5.3% a year, which is more than twice the entire realistic alpha budget
for a cross-sectional equity strategy. The strategy would be structurally behind
before its first rebalance.

So an idle sleeve holds the benchmark, and the budget is a *deviation* budget
rather than a cash bucket. Beta stays at one, every sleeve action is a pure
active bet, and capital released by a reduction returns to the benchmark instead
of landing as cash. `idle: cash` exists so the drag can be measured rather than
taken on trust; it is not the default and should not become one.

**Only the core leg reports trades.** The sleeve holds every eligible name at all
times, so if it counted, nothing would ever leave the book and every portfolio
strategy would fail the minimum-trades gate for a reason having nothing to do
with its rule.

---

## The score: what was added, not what was returned

The evaluator's heaviest component used to be raw out-of-sample Sharpe, and the
benchmark comparison went into `reasons` where it changed nothing. That was
deliberate at the time -- the losers are the control group -- and it was right
about the registry and wrong about the verdict.

It now regresses returns instead of comparing summaries:

    r_strategy = alpha + beta * r_benchmark + residual

`beta` is the exposure the rule happened to run; `alpha` is what is left, which
is the only part the rule can claim; `t(alpha)` is whether that remainder is
distinguishable from zero. A non-positive alpha is fatal, and a positive but
insignificant one caps the verdict at REVIEW: promotion means "worth spending
sealed data on", and an alpha that cannot be measured on fourteen years will not
become measurable on two.

**Subtracting the benchmark's Sharpe would not have worked**, and the reason
matters. It is unfair in the other direction: a rule invested a fifth of the
time runs a beta near 0.2, and charging it against a fully-invested benchmark
bills it for exposure it never took. Both distortions come from comparing
statistics instead of returns.

Here is the case that motivated all of it, measured on the real cache:

| | CAGR | Sharpe | maxDD | beta | alpha | t |
|---|---|---|---|---|---|---|
| 12-1 momentum, top 10 | 27.9% | 1.08 | 36.5% | 1.13 | **−1.6%/yr** | −0.59 |
| equal-weight benchmark | 26.9% | 1.23 | 29.0% | — | — | — |

It out-returned the benchmark by a point and would have been recorded as
beating the market. Its beta was 1.13 and its drawdown seven points deeper. The
extra return was rented, not earned.

And the reverse case, from the same cache -- a low-volatility ranking:

```
alpha +5.51%/yr   beta 0.53   t +1.17   IR +0.71
trails buy and hold on absolute Sharpe: 0.37 against 1.27 (-0.90) at beta 0.53
```

Positive alpha, and it *trails the benchmark on Sharpe*, because it runs half
the exposure. A Sharpe-difference gate would have thrown it away. It is still
rejected -- costs remove 72% of the frictionless edge, and t=1.17 is not
evidence -- but for reasons that are about the strategy.

**Gates before weights, still.** That low-volatility book scores 92.7/100 and is
rejected anyway. A weighted sum lets a fatal flaw be offset by an unrelated
strength, and there is no weight at which "only works without spreads" becomes
acceptable.

### What the portfolio engine needed before the rest of the machinery would work

Two things, neither free, both the kind of bug that produces a flattering number
rather than a crash:

**Parameters the perturbation test can find.** `slots()` looked at `entry`,
`regime`, `signal_exit` and the exit/sizing knobs. A portfolio spec has none of
those -- its parameters are `rank_by`, `screen`, `hold` and `rebalance_every`.
Left alone, every perturbation would have returned the identical strategy, every
result would have been identical, and `parameter_stability` (15% of the score)
would have paid full marks for being unperturbable.

**An asset-robustness question that can be asked.** The README already records
what happened when single-symbol runs met peer-relative features: the score came
out 0.0, an active penalty for using the feature. Portfolio mode brought the same
failure back in a new shape, and the first real run showed it -- `asset_robustness
0.0` on a strategy with +5.5% alpha. A book that holds ten names cannot be run on
one: weight per name is `core_budget / hold`, so a single-symbol run puts 8% into
the only name available and measures a portfolio that is 72% cash. The question
is now asked as **leave-one-out** -- the universe minus each name in turn -- and
entries are keyed by the symbol *removed*, so a low score points at the name the
edge depended on. The same strategy now scores 100.

---

## The pieces

| Layer | What it is | Purity |
|---|---|---|
| [`core/`](src/aqr/core) | Causal technical indicators over float arrays | numpy + stdlib |
| [`data/`](src/aqr/data) | Point-in-time bars; synthetic, CSV, Yahoo, Alpaca and IBKR providers, plus the quality checks every pull runs | no LLM |
| [`features/`](src/aqr/features) | The whitelist of computable features, memoised, and the causal regime classifier | pure |
| [`dsl/`](src/aqr/dsl) | Tokenizer, parser, spec, validator, content hash | pure |
| [`backtest/`](src/aqr/backtest) | Next-bar-fill simulation (trigger and portfolio), cost model, metrics, residual alpha | pure |
| [`validation/`](src/aqr/validation) | Splits, walk-forward, robustness, overfitting | pure |
| [`evaluator/`](src/aqr/evaluator) | The 0–100 score and the verdict | pure |
| [`registry/`](src/aqr/registry) | SQLite experiment log and lifecycle state machine | no LLM |
| [`agent/`](src/aqr/agent) | Prompt construction and proposal generation | **the only LLM layer** |
| [`seal.py`](src/aqr/seal.py) | The embargo: taint bit, load ledger, hash chain, canary | no I/O, no clock |
| [`backtest/costs.py`](src/aqr/backtest/costs.py) | Named broker schedules, and what one order of a given size actually costs | pure |
| [`target_book.py`](src/aqr/target_book.py) | The handoff: target weights as a versioned file, for something else to execute | places nothing |

The purity column is enforced by [`tests/test_boundaries.py`](tests/test_boundaries.py),
not by convention.

---

## Six decisions worth explaining

### 1. The LLM emits DSL, never code

An expression like `close <= ema(20) * 1.01 and rsi(14) > 40` is tokenized and
parsed into an AST whose only leaves are numbers and names from the feature
registry. The grammar has no attribute access, no indexing, no function
definition and no way to name anything that is not a registered feature. There is
nothing to sandbox because there is nothing dangerous to express.

A model that invents `ema_of_tomorrow(20)` gets a parse error naming the three
closest real features — which turns a dead experiment into a self-correcting
retry. That retry is real: with an endpoint that cannot enforce a schema
(DeepSeek's JSON mode guarantees valid JSON and nothing more), the proposal is
parsed against the grammar and one repair turn is sent back carrying the exact
error. One repair, not five — a model that cannot fill in a ten-field form after
being told what it got wrong is not about to produce a good hypothesis on the
third try.

### 2. Next-bar fills, with no override

A decision made from bar `t` is filled at bar `t+1`'s open, always. There is no
configuration flag that relaxes this, because a look-ahead switch is a
look-ahead bug with a rationale attached.

Two consequences the tests pin down:

- **Every indicator is prefix-stable.** Truncate the input, recompute, and the
  overlapping values must be bit-identical. A function that peeks cannot pass.
- **Truncating the data cannot change past trades.** A backtest over the first
  1500 bars produces exactly the trades that the full backtest produced in that
  window — same entries, same exits, same fills.

### 3. Pessimistic intrabar resolution

If a bar's range contains both the stop and the target, the stop is taken.
Without the tick sequence, assuming the favourable one is how backtests learn to
lie. If a bar *gaps through* the stop, the fill is the open — not the stop price.
Stops are not guarantees, and pretending otherwise under-reports precisely the
risk the stop exists to measure.

### 4. The search cost gates the verdict — it does not scale the score

The overfitting detector's most informative input is *how many hypotheses bought
this result*. Searching N strategies over n periods produces a best-of-N Sharpe
near `sqrt(2·ln N / n)` even when all N are worthless, so the reported Sharpe is
deflated by that expectation.

Two things about how that number is used, both of which were initially wrong
here and are worth stating precisely rather than gesturing at.

**What counts as a trial.** Not backtests. One hypothesis costs about 57 of them
— an in-sample run, a frictionless one, ten walk-forward folds, forty parameter
perturbations, one per symbol for asset robustness — and not one of those forty
perturbations is a hypothesis anyone chose between. They are diagnostics on a
rule already selected. The denominator is `distinct_hypotheses()`, which counted
153 where the backtest total said 8,705, and the difference is about a third of
the deflation term. `total_backtests()` still exists and still only grows; it
measures compute, not multiplicity.

**Where the deflation lands.** On the *verdict*, as a gate: `LIKELY_OVERFIT`
rejects outright. It does **not** scale the weighted score, and the `oos_sharpe`
component is the raw walk-forward Sharpe.

That is a deliberate limitation, not an oversight, and the reason is
uncomfortable. Over ten years of daily bars, 153 independent tries produce a
best-of Sharpe near 1.0 out of nothing whatsoever — while the best out-of-sample
Sharpe this project has produced in four campaigns is 0.60. Fold the deflation
into the weighted sum and every strategy scores zero, because by this standard
every strategy *is* zero.

So the score ranks candidates and the verdict tells the truth about them. A
strategy found in 100 hypotheses and one found in 8,000 score the same; only one
of them survives the gate. Anyone tempted to read a 67/100 as evidence of an edge
should read the `sharpe_inflation` line in the same record, which says in plain
numbers what the search cost was.

This is only honest if every attempt is recorded, including the ones that
crashed. That is why the registry writes failures as experiments, and why a rule
sent back for repair leaves its dead first attempt on the record.

### 5. Gates before weights

The evaluator's weights come straight from the architecture (30% OOS Sharpe, 20%
profit factor, 15% drawdown, 15% parameter stability, 10% cross-asset, 10%
regime). But a weighted sum lets a fatal flaw be offset by an unrelated strength,
and there is no weight at which "11 trades" becomes acceptable evidence.

So four conditions reject outright regardless of score: fewer than 30
out-of-sample trades, drawdown over 35%, non-positive OOS Sharpe, and losing more
than half the edge to transaction costs.

Every component is measured out-of-sample. An in-sample Sharpe of 3 contributes
exactly zero.

### 6. Promotion is a state machine

`CANDIDATE → PAPER → LIVE` and nothing skips a step. The registry refuses
`CANDIDATE → LIVE` outright — the shortcut this whole project exists to prevent
is the one where a good backtest becomes a live position without the paper
trading in between.

---

## Data you can research on

**The bar caches are not in git.** `data-sp500/` and `data-sp500-sealed/`
together are over a gigabyte, so they live outside the repository (local disk
or cloud storage) and are listed in `.gitignore`. A fresh clone has the code
and the universe file (`data-universes/sp500_pit.json`) but no bars. To rebuild
them, point your Alpaca credentials at the repo (`.env.local`) and run the
driver, which pulls 1h for both caches and resamples it into 4h:

```
uv run python ../scripts/pull_sp500_intraday.py --timeframe all
```

The same driver also re-arms the canaries — the synthetic tripwire symbol in
the research root — across all three timeframes, so a fresh clone that rebuilds
its caches ends with the seal's tripwires in place too. `--canary-only` does
just that step, without touching the network.

`data-sp500/` is truncated at the embargo (the search window);
`data-sp500-sealed/` holds the full history (the sealed window). The two are
kept disjoint — the research cache ends the session before the sealed cache
begins — so a strategy is never tuned and validated on the same bar.

A backtest is arithmetic performed on whatever you hand it, and it cannot tell
the difference between a market and a hole in a file. So every pull is checked
before it is cached, and the sources are checked against each other.

**Every pull is checked before it is written.** Gaps, coverage against the
expected session count, zero-volume bars, non-positive prices, overnight moves
large enough to be an unadjusted split, and whether the series starts anywhere
near the date requested. A series that fails is reported and not cached;
`--keep-suspect` caches it anyway, which is the right answer once a human has
looked and the move was real.

The check knows what a real calendar looks like: Hurricane Sandy shut the US
market for two consecutive sessions in October 2012, so every clean series from
before then contains a five-day span, and a checker that flags it teaches its
reader to ignore it. Unscheduled closures live in a table and are not gaps.

**Sources are checked against each other.** `aqr compare` reads two caches and
compares *daily returns*, never prices:

> Two correctly adjusted series for one instrument may disagree about the price
> *level* — different adjustment bases, different dividend treatment — but they
> cannot disagree about a daily *return*.

It cannot say which vendor is right. It says which dates to look at, which is
the part that does not scale by hand. Sessions are matched by date, not by
timestamp: Yahoo stamps a daily bar at 00:00Z, Alpaca at the 05:00Z open, IBKR
with a bare date, and matching on the instant found nothing in common between
Yahoo and Alpaca across sixteen overlapping years while reporting it as "no
overlap" — a comparison that silently checks nothing reads as a clean bill.

### What the three sources actually contain

Measured on the NASDAQ-50, not taken from anyone's documentation.

| Source | Depth | Confirmed defects |
|---|---|---|
| **Yahoo** (`data-yahoo/`) | 2010– | none found |
| Alpaca SIP (`data/`) | 2016– | closes wrong on halt and auction days |
| IBKR `ADJUSTED_LAST` (`data-ibkr/`) | 2005– | six kinds, below |

**IBKR is the deepest and the least trustworthy.** `ADJUSTED_LAST` returned four
unadjusted 2:1 splits — AAPL 2005-02-28, ADBE 2005-05-24, ORLY 2005-06-16,
SBUX 2005-10-24, every date an exact split date — a 4.01× discontinuity in FTNT
on 2014-01-13, a 31.7% break in ASML, a 910-day hole in TMUS across 2013–2015,
143 bars for AZN where twenty years were asked for, and seven series truncated
by a decade with no corporate action behind it. Thirteen of fifty symbols.

It also refuses an explicit end date, which is not a defect but does mean its
history cannot be chunked and must arrive in one request. `aqr ibkr-check`
establishes that and how deep one request reaches (30 years, on this account).

**Alpaca is clean except where it matters most.** Its daily closes diverge from
the official close on exactly the days a strategy's tail risk lives: the March
2020 circuit-breaker sessions, and TSLA's S&P 500 inclusion on 2020-12-18, where
the official close of $695.00 against the prior $655.90 is +5.96%. Yahoo reports
+6.0%; Alpaca reports +0.4%. Same story on SPY 2020-03-12: −9.6% against −6.9%,
where the record says −9.57%.

Its IEX feed is worse and is not the default: a 2016–2026 pull of SPY came back
with 1530 bars covering ~1960 sessions, including one **634-day hole**. The
consolidated tape returned 2677 bars whose largest gap is a holiday weekend.

**So Yahoo is the primary research dataset**, and the deeper sources are kept
for cross-checking rather than merged in. Merging would produce a series with a
seam in it that no later run could see, and the five extra years IBKR offers are
not worth thirteen corrupted symbols.

## Regimes are measured, not assumed

`regime_robustness` is 10% of the score and needs a label per bar. Only the
simulator could supply one — it knows the regime it generated — so on real data
the pipeline passed no labels, the report was never built, and the evaluator
substituted 0.5. A tenth of every real-data score was a constant standing in for
a measurement. Experiment records from before the fix show it plainly:
`"regime_robustness": 50.0`, on every single one.

The classifier ([`features/regime.py`](src/aqr/features/regime.py)) asks two
questions, both scale-free so that a 12%-vol utility and a 90%-vol biotech are
judged on the same axis without per-symbol tuning — per-symbol tuning being how
a regime classifier becomes another fitted parameter nobody counted:

- **Is it trending?** Not "did a moving average cross", which says yes whenever
  price wanders slightly in one direction, but the drift's t-value: is the
  60-bar move large relative to this instrument's own noise?
- **Is it violent?** Recent realised vol against an **expanding** baseline — the
  mean of every reading so far — not against a trailing window of itself. The
  trailing version fails exactly when it matters: after a year of crisis the
  denominator has absorbed the crisis, and the classifier calls it normal.

Warm-up bars are labelled `UNKNOWN` rather than guessed, and `regime_robustness`
skips those trades. And the labels are prefix-stable: truncate the series,
reclassify, and every overlapping label is identical. A regime classifier that
peeks is contaminated in the most flattering possible direction, because it
would know which regime the market was about to be in.

## The benchmark, and the finding that put it there

Fourteen strategies reached PAPER. `aqr holdout` re-ran all fourteen on
twenty-eight NASDAQ-50 symbols that no campaign had ever loaded — a set the
search could not have leaked into, because it was never in a universe.

```
12 of 14 kept a positive Sharpe on symbols they were never selected on
```

That reads like a result. Then the benchmark went in beside it:

```
14 of 14  LOST TO buy and hold: sharpe +1.16  return +4262.7%  maxDD 29.5%
best strategy:                  sharpe +0.65  return   +57.5%  maxDD  9.3%
```

Not one of them beat holding the same symbols over the same window. The best
managed half the Sharpe and a seventy-fourth of the return. The reason is not
subtle: they are all long-only rules, and the window was 2010–2026. On NASDAQ
names over that stretch a long-only rule makes money almost whatever the rule
is, so its Sharpe is mostly the market's and says almost nothing about the rule.

Nothing in the pipeline had asked. A strategy could score 68.7/100 and reach
PAPER while being comprehensively beaten by doing nothing.

**It was recorded and not gated, and that half was wrong.** Keeping the losers
is right: without the ones that failed there is nothing to measure the ones that
succeed against, and a research log that keeps only winners is the log that made
the winners look inevitable. Every experiment carries `benchmark_sharpe` and
`excess_sharpe` for exactly that reason, and rejection is not deletion — a
REJECT verdict is still written to the registry in full.

But recording an answer the verdict ignores is how fourteen strategies reached
PAPER having added nothing. Promotion is now gated on the *residual regression*
rather than on this Sharpe difference — see [The score](#the-score-what-was-added-not-what-was-returned)
above for why the difference itself is the wrong gate, and what happens to a
low-exposure rule under it.

The benchmark is deliberately generous — equal-weight, rebalanced daily, no
costs, no slippage. A strategy that cannot beat an unrealistically good version
of doing nothing is not close.

### Two bugs this found on the way in

The first benchmark implementation reported **Sharpe +0.00 on a series that
returned +4263%**, so every line printed "BEAT buy and hold" — the exact
opposite of the truth, in the one line the comparison exists to produce. The
cause was reuse: the curve was wrapped in a synthetic `BacktestResult` and passed
to `compute_metrics`, which refuses a ratio on fewer than five trades. That guard
is right for a strategy — a Sharpe from three trades is noise — and meaningless
for buy-and-hold, which has no trades by construction and four thousand daily
returns.

The second: a zero was padded onto the front of the portfolio return series so
it would line up with the sessions. There is no return on the first session, and
inventing a flat day that never happened moves both the mean and the standard
deviation.

## The window has to contain a bear market

Campaigns 1–4 ran on 2010–2026, which was chosen carelessly and contains two
shallow V-shaped drawdowns. Yahoo has these names back to 1995, which covers the
dot-com collapse, 2008, 2020 and 2022 — and changes what the benchmark looks
like:

| window | benchmark Sharpe | CAGR | max drawdown |
|---|---|---|---|
| 2010–2026 | +1.23 | +26.6% | 29.0% |
| 1995–2026 | +1.08 | +27.1% | **51.7%** |

The bear markets show up in drawdown, not in Sharpe. That is the point: a
strategy tested only on 2010–2026 has never been asked what it does when the
index halves.

**Depth makes survivorship bias worse, not better.** `NASDAQ_50` is the list as
it stands *today*; every constituent earned its place partly by not going
bankrupt in 2001. Backtesting them through the dot-com crash measures, in part,
the fact that we already know who survived.

The benchmark comparison is what makes this bearable. Strategy and benchmark are
measured on the same biased symbol set, so most of the bias is in both numbers
and cancels out of the difference. The absolute return over 1995–2026 is close to
meaningless; the excess over buy-and-hold is not.

## Long, short, and market-neutral

`direction` takes `long`, `short`, or `market_neutral`. The last needs a separate
`short_entry` expression and holds both sides at once.

This exists because of the holdout finding rather than for completeness. A
long-only rule's Sharpe is mostly beta; a rule that is long and short in similar
size has no such crutch, and no such excuse. It is also the form the
cross-sectional features were built for — `rs_rank(60) > 0.7` long against
`rs_rank(60) < 0.3` short is the classic construction, and until now it could not
be written down.

Two things the implementation gets right on purpose:

**Direction belongs to the position, not the run.** `long` used to be decided
once and threaded through sizing, stop placement, intrabar exit resolution and
borrow cost. Every one of those now asks the position. A stop above the entry is
correct for a short and catastrophic for a long, and that is not a distinction
that can be got right halfway.

**A short leg is not the long leg negated.** The schema refuses `short_entry` on
a one-directional strategy and refuses a `market_neutral` strategy without one.
Mirroring the long rule would assert that one condition predicts up moves and its
negation predicts down ones — precisely the symmetry equities do not have, given
that they drift upward, that losses on the short side are unbounded, and that a
borrow fee accrues every day. A bar satisfying both legs is treated as a
contradiction and produces no trade at all: picking one would make the result
depend on evaluation order.

## The universe, not just the symbol

Every rule used to see one instrument at a time. So "buy the strongest names in
a broad advance" — the oldest documented equity anomaly there is — could not be
written down at all, and neither could any form of "this stock is moving and its
peers are not". Three features close that:

| | |
|---|---|
| `rs_rank(n)` | percentile rank of this symbol's *n*-bar return across the universe, 0 weakest to 1 strongest |
| `rel_return(n)` | this symbol's *n*-bar return minus the universe median |
| `breadth(n)` | fraction of the universe with a positive *n*-bar return — a market-state feature, identical for every symbol |

The median rather than the mean, because one name up 400% should not redefine
what "average" means for the other forty-nine.

### The three ways this fails silently

**Ranking against a universe assembled with hindsight.** If a symbol that listed
in 2020 takes part in the 2010 cross-section, the anomaly discovered is "the
stocks later added to the index outperformed". So a symbol contributes from its
first bar and stops at its last, and a session with fewer than two names present
produces NaN rather than a rank.

NaN, specifically, and not 0.5 or 0.0. A rank against nobody is not a weak
signal, it is no signal — and `rel_return(20) > 0` evaluates a 0.0 happily while
NaN correctly takes the bar out of the comparison.

**Losing prefix-stability.** The rank at bar *t* is built from returns ending at
*t*; a peer doing something dramatic afterwards cannot reach back and change it.
The generic causality test covers these features like any other, truncating the
whole universe together — which is the honest prefix test here, since on the day
the data ends nobody knew what any peer would do next either.

**Scoring them with machinery that assumes one symbol.** `asset_robustness` asks
"is this an edge or a property of one ticker?" and used to answer by re-running
the rule on each symbol alone. For a peer-relative rule that question is
unanswerable as posed: ranked against itself every feature is undefined, the rule
fires on nothing, every symbol is skipped, and the score comes out **0.0** — not
as a placeholder but as an active penalty for using the feature at all, on 10% of
the total.

So `run_backtest` now separates the *traded* universe from the *peer* universe.
Asset robustness holds the market definition fixed and shrinks the traded set to
one name, which is the question it meant to ask. The peer set still defaults to
the traded universe, so a result stays reproducible from the spec alone: whatever
else happens to be loaded cannot change the answer.

## The universe was survivorship-biased, and it was worth 16 points a year

Every result before this section was measured against the wrong benchmark, and
the correction is the largest single number in this repository.

`data/` was pulled from *today's* constituent list. Every company that was
dropped, acquired or went bankrupt is missing from it, so a backtest over
2016–2024 on that cache holds only the names that made it to 2026. That is not a
small bias. The old NASDAQ-50 benchmark showed CAGR 28.2% with a -29.3% maximum
drawdown. The point-in-time S&P 500, equal-weighted, over the same window:

| | CAGR | Sharpe | maxDD |
|---|---|---|---|
| NASDAQ-50, survivors only | 28.2% | — | -29.3% |
| **S&P 500, point-in-time** | **12.2%** | **0.68** | **-40.0%** |

**Sixteen points of CAGR a year, and eleven points of drawdown.** Every earlier
"beat the market" judgement in this project was made against a baseline inflated
by roughly that much — which is to say most of them were not beating anything.

So `data-universes/sp500_pit.json` holds **682 tickers with dated membership
intervals**, 503–504 members on any given session, reconstructed from the index
change log. 31.8% of the names in it left the index inside the window. They are
kept, because the ones that left are exactly the ones whose absence creates the
bias.

Membership gates three separate things, and missing any one of them leaks:

- **what may be held** — a name dropped in 2019 and readmitted in 2021 keeps
  trading throughout, so its bars have no gap and bar-presence says "member" for
  the whole span. The book would hold it for two years it was not in the index,
  picked necessarily because we know it came back;
- **what the cross-section ranks against** — ranking today's decision against
  next year's index members is look-ahead in the ranking rather than in the
  price;
- **what the benchmark is** — an equal-weight benchmark built from survivors is
  the same bias wearing the benchmark's clothes.

Measured, same rule, same bars, gate on and off:

```
without --universe (rank against every name ever a member)   CAGR 14.3%  Sharpe 0.81
with    --universe sp500_pit (rank against that day's index)  CAGR 10.5%  Sharpe 0.67
```

**3.8 points of CAGR a year was look-ahead.**

Three names are recorded in the file as *not corrected* rather than quietly
fixed: `WRK` is missing entirely — WestRock was a member for most of the window
and the change log has no row for it or its successor `SW`, and inventing
interval boundaries from no source is worse than a documented hole — and `D`,
`SRE` and `TROW` have unverified interval starts.

### Alpaca does not adjust for spin-offs or merger share exchanges

Splits and dividends, yes. A company handing its shareholders stock in a
subsidiary, no. The price simply drops, and to a backtest a drop is a drop:

| | date | apparent move | what actually happened |
|---|---|---|---|
| `RTX` | 2020-04-03 | **-70.3%** | Raytheon/UTC merger and the Carrier + Otis spin-offs |
| `TGNA` | 2017-06-01 | **-79.2%** | Cars.com spun off |

Seven names carry an unadjusted discontinuity inside a tradable interval. They
are **not corrected**, and that is a deliberate choice with a stated defence: the
overfitting detector's profit-concentration signal is what catches a result that
leans on one, and it does so without anybody having to guess a corporate-action
ratio. `data-sp500` keeps them, so `data-sp500-sealed` keeps them too — the
sealed window has to differ from the search window in its dates and in nothing
else.

Ten times a strategy fires on the day the tape falls 70% because a spin-off was
not adjusted, that strategy has a profit concentrated in ten trades, and the
detector says so. That is the check. It is weaker than correcting the data and
it is honest about being weaker.

## A dead rule gets one more turn

The first 40-hypothesis DeepSeek campaign over 19 real symbols produced **15
rejections for "never fires"** — 37% of the research budget. Every one of them
compiled cleanly, referenced only real features, and was unsatisfiable on daily
bars: opening drives, midday VWAP reversions, funding rates. Sensible ideas
about a market the data does not describe.

Two things were tried. Only one of them worked, and the log says which.

**Telling the model what a bar contains — did not help.** The feature catalogue
answers "will this parse"; it never answered "is there anything there to see", so
a paragraph naming what a daily bar cannot show was added to every prompt. The
dead-rule rate went *up* the next campaign, from 15/40 to 23/40. The likeliest
reason is a confound the note cannot beat: research memory now holds dozens of
rejections, and the prompt asks for hypotheses unlike what has been tried, which
pushes the model toward more exotic and less expressible ideas. The paragraph is
still there — it is true, and it costs nothing — but it is not a fix and is not
described as one.

**Sending the dead rule back once, with the reason — worked.** It is the same
recoverable failure the parse-error path already handles, and it arrives with a
validator complaint worth forwarding. The repair prompt carries the offending
expressions and asks for the *same mechanism* with a workable condition — not
the research memory again, which would invite the model to abandon the idea
instead of fixing it.

| | dead rules | recovered by repair | reached the pipeline |
|---|---|---|---|
| campaign 1 (before) | 15 | — | 24 / 40 |
| campaign 2 | 23 | 12 | 28 / 40 |
| campaign 3 | 24 | 14 | 29 / 40 |

Roughly half of dead rules come back alive. The rate of *producing* them is
unchanged, so this is a recovery mechanism and not a cure.

The failed first attempt is still written to the experiment log. Hiding it would
understate the multiple-comparisons denominator, which is the one number the
overfitting detector cannot do without. A repair that returns the identical rule
is a failed repair, not a second attempt, and is not recorded twice.

## The overnight features, and what they found

The campaign logs showed the model reaching for overnight hypotheses again and
again — `short_up_gap_fade`, `post_gap_fill_rejection`,
`long_overnight_return_reversal` — and none of them could be written down. The
DSL had no way to reach a previous bar's value, so `open[t] / close[t-1]` was
not expressible, and the model kept approximating it with conditions that meant
something else. So `gap()`, `overnight_return()` and `prev_close()` were added.

The model adopted them immediately: **22 of 54 proposals** in the next campaign
used one. Eleven of those survived long enough to be evaluated properly.

**All eleven were rejected.** The best managed an out-of-sample Sharpe of 0.14
on six trades; the rest were negative. On this universe, over this window, after
costs and after the search cost is charged, the overnight effect is not there.

That is the feature addition paying for itself. It converted a family of
hypotheses from *unexpressible, therefore untested* into *tested, and rejected* —
which is the only form of answer this project is built to produce.

## Writing a strategy by hand

```yaml
strategy:
  name: trend_pullback_v1
  hypothesis: >
    In an established uptrend a shallow pullback is profit-taking rather than a
    change of trend, and buyers who missed the move supply demand at the mean.

  universe:
    symbols: [SPY, QQQ, IWM]
    timeframe: 1D

  regime: close > ema(200) and adx(14) > 20
  entry:  close <= ema(20) * 1.01 and rsi(14) > 40 and rvol(20) > 1.2

  exit:
    stop_loss:   {type: atr, multiplier: 2.0, period: 14}
    take_profit: {type: risk_reward, ratio: 2.0}
    max_holding_bars: 20

  sizing:
    risk_per_trade: 0.005      # fraction of equity at risk per position
    max_position_pct: 0.25

  max_positions: 3
```

Unknown fields are errors, not noise: `stop_los:` gets rejected rather than
silently giving you the default stop you did not ask for.

`aqr features` lists the full vocabulary (32 features across trend, mean
reversion, volatility, volume, structure, overnight and cross-section).

A market-neutral strategy adds a `short_entry` beside `entry` and sets
`direction: market_neutral`.

Intraday works the same way: set `timeframe: 1h` or `4h` and every lookback is
counted in those bars. The hourly cache holds six bars per regular session
(10:00–16:00 ET — the 09:30–10:00 half-hour is not in the data), the 4h cache
two. The research loop can also offer the model a choice:
`aqr research --timeframes 1D,1h,4h` lets each hypothesis carry its own
`timeframe` field, restricted to that allowed set, and the walk-forward
geometry and history requirements scale to whichever bar size was picked.

---

## The sealed run: what actually happened when the envelope was opened

`rs_volatility_consistency_neutral_v1` was the sole ACCEPT of campaign 07. It was
pre-registered on 2026-08-28 — selection rule recorded, seal digest recorded,
before a single embargoed bar was read — and then run once, in a separate
process, against 2024-09 through 2026-08.

```
rs_volatility_consistency_neutral_v1 [3f6e2c8a9309068b]
sealed window 2024-09-03 -> 2026-08-27   (498 sessions)
  strategy   return +56.39%  sharpe +1.86  maxDD -20.4%  trades 300
  residual   alpha +16.72%/yr  beta 0.43  t +2.22  IR +1.58
```

**It was not refuted.** That is the strongest verdict this window is capable of
producing, and the wording is not modesty — `can_confirm` is a property that
returns `False` by construction. The standard error on an annualised Sharpe over
498 sessions is ±0.71, which is larger than most of the Sharpe differences
anybody argues about.

The part that was genuinely falsifiable in advance is the beta. The search window
predicted 0.46; the sealed window returned **0.43**. A rule that had been fitting
noise had no reason to reproduce its market exposure two years later, and this
one did.

What this does **not** upgrade: the deflated Sharpe is still 0.10 after 324
trials. A t of +2.22 clears the bar as the only candidate ever screened against
this window and would not clear it as the seventh — see the multiplicity table
above. And the seal proves the embargoed *data* was not read; it cannot prove the
embargoed *period* did not inform a decision, since the researcher lived through
it and every model in `aqr providers` has a training cutoff after it. The
certificate records that exposure under `knowledge_exposure` rather than denying
it.

The stored result carries the measurement, the seal certificate, the declaration
and the ancestry report, so the run can be audited without trusting this
paragraph.

## What a trade costs depends on how many names you hold

The cost model was calibrated for a three-position event-driven book. The
strategy that survived holds 110 names and rebalances every five sessions, and
two of the four charges do not scale with notional. So the same rule, on the same
bars, under the same model, is priced differently depending on the size of the
account:

```
      equity     CAGR   Sharpe   frictionless   Sharpe retained
     100,000   15.94%     1.15           1.58              73%
   1,000,000   20.50%     1.43           1.58              91%
  10,000,000   20.85%     1.46           1.58              92%
```

Nothing about the strategy changed across those rows. At $100k a sleeve position
is $192, and a $1.00 per-order floor on $192 is **52 basis points** against the
3bp the spread and slippage charge. The floor bound on 1262 of 2317 core
round-trips. Cost retention is a *fatal* gate, so this is not cosmetic: it
decides verdicts, and it decides them on `initial_equity` — a number nobody
thinks of as a cost parameter.

**The fix is not a cheaper model.** A trader with $100k and a hundred-name book
at a broker charging a dollar an order really would pay 52bp, and a model that
hid it would be lying in the flattering direction. What was actually wrong is
that the schedule was implied rather than named, the per-order cost could not be
asked for, and the schedule was not recorded with the verdict it decided. All
three are fixed:

```bash
aqr costs --equity 100000 --positions 110      # what one order actually costs
```

| schedule | fee | spread+slip | all-in | floor binds |
|---|---|---|---|---|
| `alpaca` | $0.00 (0.0bp) | $0.27 | 3.0bp | — |
| `ibkr_fixed` | $1.00 (11.0bp) | $0.27 | **14.0bp** | yes |
| `zero` | $0.00 | $0.00 | 0.0bp | — |

That table spreads the account evenly over 110 names — $909 each. The real book
is not even: 10 core names at 8% and about 100 sleeve names at 0.19%, and it is
the sleeve names, at $192, that reach 52bp. At $1M the same table reads 4.1bp for
`ibkr_fixed`. `--costs alpaca` selects the
commission-free schedule — the venue these bars come from and whose paper account
the target book is built for. A named schedule wins outright rather than merging
with the individual `--spread-bps` knobs: half a preset and half a set of
overrides is a schedule no broker offers, and the record would name one that was
not used.

**The default is unchanged**, and deliberately. `IBKR_FIXED` *is* `CostModel()` —
the same numbers 324 recorded experiments were judged under. Recalibrating by
editing the defaults would silently reinterpret every one of them. What changed
is that the schedule now travels with the experiment record, the same way
`dataset_version` does, so a verdict can be traced to the cost assumptions that
produced it.

## The handoff: a target book, not an order

A validated strategy has to become positions somewhere, and that somewhere is not
here. `aqr target-book` writes one file and stops:

```bash
aqr target-book 3f6e2c8a9309068b
# 604 of 680 symbols, 2023-05-17 -> 2026-08-29
#
# rs_volatility_consistency_neutral_v1 [3f6e2c8a9309068b] as of 2026-08-27
#   104 positions, gross 1.0000 (core 0.8000, sleeve 0.2000)
#
# runs/target_books/rs_volatility_consistency_neutral_v1-...-2026-08-27.json
```

There is no `--dry-run`, because there is nothing to be dry about: no code path
in this project sends anything anywhere. Four things make the file worth
executing.

**It comes off the validated code path.** The weights are
`run_strategy(...).weights_at(last)` — the same entry point the backtest, the
walk-forward, the robustness passes and the sealed run all go through. The
handoff and the backtest differ in where the data stops and in nothing else, and
`test_the_book_matches_what_the_backtest_held` fails the build if they ever
diverge.

**Weights, not shares and not notionals.** This project does not know the
account's equity, and inventing one would be the first step towards it placing
the order. The schema refuses a `shares`, `quantity`, `notional`, `equity` or
`orders` field by name. What the consumer owes is listed *inside* the artefact,
under `consumer_must_supply`: sizing, reconciliation against real positions,
turnover and notional caps, a kill switch, a fill journal, and an equity-shaped
risk gate — AlphaGate's gate is options-shaped (short strikes, DTE, defined-risk
width) and does not apply to an equity book. Naming that gap is part of the
handoff; filling it is not.

**It is only written for something that earned it.** A book is refused unless the
registry knows the strategy, its seal has been spent, and the sealed run did not
refute it. Handing off before the seal is spent would read the embargoed years
for a rule that still owes an out-of-sample verdict, which is the loophole
pre-registration exists to close. The book that *is* written carries the
declaration, the selection rule, the sealed measurement, and both
denominators — the 324 hypotheses the search compared and the number of
candidates the sealed window has screened — so the claim it rests on travels with
it.

**Producing one taints the seal, and the certificate says so.** A target book
means reading the present, so the process that writes one is tainted by
construction. That is the correct record rather than a defect: the taint says
"this process read past the embargo", which is the whole job. What that process
must never also be is the search — and it is not, because the seal has already
been spent before a book can be written.

One property is a real cost and is named in the artefact rather than hidden. The
engine decides at a close and fills at the next open, so the weights in force on
the final session were decided on the session before it: an executor placing them
at the next open acts one session later than the backtest did. Reporting the
decision made *at* the final session instead would mean re-implementing the
selection outside the engine, which is the one thing the handoff exists not to
do.

---

## What this does *not* do yet

Deliberately out of scope for v0.1, in the order the architecture recommends
adding them:

- **News / event engine** (§6, §22) — LLM event extraction, event scoring. The
  20% sleeve is built and reserved for it; until it exists the sleeve holds the
  benchmark, which is to say it is switched off without costing anything.
- **ML prediction layer** (§23) — LightGBM over the feature set.
- **Multi-agent research** (§26–27) — Critic and Risk agents as separate roles.
- **Order placement** — and it never will, here. This project finds strategies;
  execution is a different system with different invariants, and the registry's
  `CANDIDATE → PAPER → LIVE` states describe research standing, not positions.
  What it produces instead is a [target book](#the-handoff-a-target-book-not-an-order),
  and `tests/test_boundaries.py` fails the build if a trading host, an order
  path or a broker SDK ever appears under `src/aqr/`.
- **Intraday costs** — the loop itself now researches on 1h and 4h bars as well
  as daily (`aqr research --timeframes 1D,1h,4h`; the model picks a granularity
  per hypothesis, inside that set). What has not been re-measured is the cost
  model, which is still calibrated for daily holding periods — treat intraday
  scores with that caveat until it is.

The pieces that would be hardest to retrofit — point-in-time data, causal
features, the search-cost accounting, the lifecycle state machine — are the ones
that are here.

---

## Development

```bash
uv run pytest                    # 973 tests, all offline, all deterministic
uv run ruff check .
uv run mypy
```

The suite is fast and hermetic by design. A test that reaches the network fails
for reasons unrelated to the code; one that depends on live market data cannot be
re-run next year and give the same answer.

`tests/test_end_to_end.py` runs the whole chain — `research → evaluate →
preregister → sealed run → target-book` — through the real CLI commands against a
temporary cache and registry, asserting after each stage that the artefact it was
supposed to leave behind is there and carries the fingerprint the next stage will
look for. Offline, so it needs no API key, and part of the suite rather than a
script somebody has to remember to run.

Every stage of it runs in its own `scope(Seal())`, because in real life every
stage is its own process — and the first version of that test proved why. Writing
its fixture cache taints whichever seal is ambient, because the cache spans the
embargo, and sharing that seal with the research stage made `preregister` refuse
the candidate for a tainted ancestry. The seal was right and the test was
wrong.

---

## Licence

Personal research project. Not investment advice; no part of this produces a
recommendation to buy or sell anything.
