# PLAN — from search to a placed order

Written to survive a context reset. Everything here is executable from a cold
start; nothing depends on remembering a conversation.

## Where things stand

**Done and green (979 tests, ruff, mypy clean). All five phases complete.
`aqr evaluate` on the full 680-name universe: 1m08s.**

**The seal has been opened five times.** The executed strategy is the fourth,
`low_vol_relative_strength_carry_v1_improved` [`9b4ac85c149ec6db`], run against
the embargoed years: alpha +20.40%/yr, beta 0.35, t +3.73, IR +2.66 over 498
sessions, maxDD -4.3%. It was **not refuted**, which is the only verdict a
498-session window is entitled to reach — the standard error on an annualised
Sharpe here is ±0.71.

**Its declaration says "backfill test 3".** The seal was spent exercising the
sealed-cache backfill rather than choosing a strategy, and the rule was pinned
afterwards, once the result was visible. Selecting on the outcome is what
pre-registration exists to stop, and no bar this project computes charges for it,
so the `t` should be read with a discount the table does not apply. What survives
the discount is the margin: five looks put the bar at 2.576 and `t +3.73` clears
the bar for any look count up to 261. See `runs/README.md` for all five runs and
the bar each faced.

The first candidate through the window, `rs_volatility_consistency_neutral_v1`,
returned `t +2.22` and does not clear the bar it would face today; it is retired.
Nothing about any of this upgrades the deflated Sharpe, which is 0.74 after 411
trials, against a search denominator of 414 distinct hypotheses.

**Against SPY rather than against the universe**, the excess return is +21.5pp,
not +34.5pp: SPY buy-and-hold returned +42.05% over the same 498 sessions where
the equal-weight universe returned +29.02%. The gap between the two benchmarks is
cap-weighting, and it is most of the headline. What survives the switch is the
risk side -- maxDD -4.3% against -18.8%, Sharpe +2.70 against +1.16.
`scripts/report_benchmark.py` prints all three rows; the SPY row is for reading
and feeds no verdict.

Phase 4 is done: `aqr target-book` writes a target book off the same
`run_strategy` path the sealed run used, refuses to write one for a candidate
whose seal is unspent or whose sealed run refuted it, and records it in the
registry against the fingerprint. Nothing under `src/aqr/` can reach a trading
API, and a boundary test now fails the build if that changes.

Phase 5 is done. The README carries the four sections it owed; the cost model
names its schedules, can price an order, and travels with the verdict it decided
(the default is unchanged on purpose); and `tests/test_end_to_end.py` runs the
whole chain offline, asserting each artefact as it lands.

| Piece | State |
|---|---|
| Embargo seal | `src/aqr/seal.py` — sensor in `Bars.__post_init__`, monotone taint, hash-chained load ledger, `__CANARY__` tripwire, per-campaign `run_id`, one-way `enter_sealed()` refused once anything has been read |
| Repeat looks | One shot is per *candidate*, not per window: a new hypothesis earns its own sealed run, so the research loop can continue. The looks are counted and the alpha's significance bar rises with the count (`multiplicity_bar`, Bonferroni on a 5% family-wise rate) |
| Pre-registration | `preregistration` table + `sealed_run_at` on `strategies`. `aqr preregister` declares, `aqr-sealed run` spends, exactly once. Ancestry taint is checked by campaign, not by backtest |
| Cost calibration | `IBKR_FIXED` (the unchanged default) / `ALPACA_EQUITIES` / `ZERO_COST`, `CostModel.price_order`, `aqr costs`. The schedule is recorded with every experiment — cost retention is a fatal gate, so it is part of the verdict |
| End-to-end | `tests/test_end_to_end.py` — research → evaluate → preregister → sealed run → target-book, offline, every artefact asserted |
| Target-book handoff | `src/aqr/target_book.py` + `aqr target-book` — weights only, off `run_strategy`, refused unless the seal was spent and not refuted, recorded in `target_books` with a sha256 |
| Sealed entry point | `src/aqr/cli_sealed.py` — a separate binary that cannot import `aqr.agent` or `aqr.cli`, reads only through `SealedProvider`, and promotes the process only after every refusal has already had its chance |
| Cache separation | `data-sp500/` (Alpaca PIT, 681 series, truncated at embargo), `data-sp500-sealed/` (682 series, full history through 2026-08-27). `aqr seal-check` audits both |
| Portfolio engine | `src/aqr/backtest/portfolio.py` — always invested, `rank_by` ranking, 80% core / 20% sleeve (sleeve holds the benchmark while idle), forced exits `delisted` / `left_universe` |
| Residual alpha | `src/aqr/backtest/alpha.py` — regression, not Sharpe difference. `t(alpha) <= 0` is fatal; positive-but-insignificant caps the verdict at REVIEW |
| PIT universe | `data-universes/sp500_pit.json` — 682 members, 503–504 per day, 31.8% left the index in-window. Corrections and non-corrections recorded in the file |
| Membership gating | Threaded through `run_strategy` → walk-forward → leave-one-out → `evaluate_candidate` |
| Proposer | Knows `mode: portfolio`, `rank_by`, `hold`, `rebalance_every`; mode-specific field validation |
| Cross-section cost | `_rank_rows` is `n log n` and every per-session reduction is memoised per period. 680-name backtest of `rs_rank(126) - rs_rank(21)`: 9.2s (§1.1) |
| PIT universe in the CLI | `--universe sp500_pit` resolves, gates by membership, and defaults `--csv-root` to `data-sp500` (§1.2) |
| Walk-forward geometry | `TRAIN_BARS = 504` / `TEST_BARS = 126` in `validation/splits.py`: 13 folds on 2180 sessions, 11 with embargo (§1.3) |
| Baseline specs | `strategies/baselines/*.yaml` — four hand-written portfolio rules over the 680 usable PIT names, for the control campaign |

**Measured baseline on the PIT universe** (2016-01-04 → 2024-08-30, 661 symbols):

```
benchmark (equal-weight, PIT)   CAGR 12.2%   Sharpe 0.68   maxDD -40.0%
mom 12-1     CAGR 10.6%  beta 0.92  alpha -0.16%/yr  t -0.04
reversal 1m  CAGR 15.7%  beta 1.20  alpha -0.30%/yr  t -0.09   <- highest return, no alpha
low vol      CAGR 12.1%  beta 0.65  alpha +2.76%/yr  t +1.09  IR +0.37
```

The old NASDAQ-50 benchmark showed CAGR 28.2% / maxDD -29.3%. Every earlier
"beat the market" judgement was made against a baseline inflated by roughly 16
points a year.

**Verify these before starting** (cheap, and they catch a stale tree):

```bash
uv run --directory ai_quant_researcher --extra dev pytest -q      # expect 979 passed
uv run --directory ai_quant_researcher aqr seal-check             # research root clean, canary armed
ls ai_quant_researcher/data-sp500/1D/*.csv | wc -l                # expect 681
```

---

## Phase 1 — Make the search runnable on the PIT universe

Blocking everything downstream.

### 1.1 Cross-sectional feature cost on 661 symbols — **DONE**

The cost was in `CrossSection._rank_rows`, twice over. It built an `n x n`
boolean comparison matrix per session, *and* it was never memoised: only the
return grid was cached, so the whole universe was re-ranked once per symbol.
Total cost `n**2` per session times `n` symbols. Measured: 25 names 0.6s, 50
names 1.8s, 100 names 7.4s — extrapolating to 661 gives ~35 minutes, which is
the reported failure.

Fixed in `src/aqr/features/cross_section.py`:

- ranks come from a sort plus `searchsorted` (`n log n`) instead of the pairwise
  matrix, and `tests/test_cross_section.py::TestRankDefinition` pins that the
  fast form equals the pairwise one exactly, ties included;
- every per-session reduction — ranks, medians, breadth — is memoised per
  period alongside the return grid, so it is computed once for the universe
  rather than once per symbol;
- column lookup no longer re-sorts the universe on every call.

**Acceptance met.** Feature layer: 661 symbols × 2180 bars, both periods, every
symbol — 0.26s. Full backtest of `rs_rank(126) - rs_rank(21)` over the 680-name
PIT universe: **9.2 seconds** (was: did not finish in 40 minutes). 792 tests
green, ruff and mypy clean.

### 1.2 Register the PIT universe in the CLI — **DONE** (acceptance pending 1.4)

- `PIT_UNIVERSE_FILES`, `load_point_in_time`, `is_point_in_time` and
  `universe_names` in `src/aqr/data/universes.py`; `resolve("sp500_pit")` returns
  the 682-name union across time. The JSON path is anchored to the package, not
  to the working directory. A `--universe-limit` on a point-in-time universe
  raises rather than silently selecting the alphabetically first N.
- `_load()` in `cli.py` takes the universe *name* and returns
  `(bars, labels, membership, version)`. `research`, `evaluate` and `backtest`
  pass the membership table into the pipeline; `ResearchLoop` carries it too.
- `--csv-root` defaults to `data-sp500` for `sp500_pit`, and says so when it
  does. An explicit `--csv-root` is still obeyed.

**The gate bites, measurably.** `baseline_rs_rank_spread` over 2016-01-04 →
2024-08-30, same bars, same rule:

```
without --universe (rank against every name ever a member)  CAGR 14.3%  Sharpe 0.81
with    --universe sp500_pit (rank against that day's index) CAGR 10.5%  Sharpe 0.67
```

3.8 points of CAGR a year was look-ahead: ranking against, and holding, names
that were not in the index on the session being traded.

Full-`evaluate` acceptance is blocked on 1.4 below, not on this work.

### 1.3 Walk-forward geometry for a 2180-bar window

The defaults (`train_bars=756`, `test_bars=252`) yield about 5 folds on 8.7
years, against 10 on the old window. Fewer folds means a noisier
`positive_fold_rate`, which is already the single most common rejection reason.

- Decide and document the geometry. Candidate: `train_bars=504`, `test_bars=126`
  → roughly 13 folds, at the cost of a shorter fit window.
- Do **not** loosen `MIN_POSITIVE_FOLD_RATE` to compensate. If the shorter window
  cannot support the gate, that is a finding about the window.
- **Acceptance:** the chosen geometry is a named constant with a comment giving
  the fold count and the reasoning, and `aqr evaluate` on the PIT universe
  produces ≥ 8 folds.

**DONE.** `TRAIN_BARS = 504` / `TEST_BARS = 126` in
`src/aqr/validation/splits.py`, with the fold arithmetic and the reasoning in a
comment above them. Measured on 2180 sessions: 13 folds, 11 once the 21-bar
embargo a monthly rebalance implies is applied — against 6 and 5 under the old
756/252. Threaded through `run_walk_forward`, `evaluate_candidate`,
`ResearchConfig` and the CLI options, so there is one geometry rather than four
copies of a literal. `MIN_POSITIVE_FOLD_RATE` is untouched at 0.4. All 792 tests
pass unchanged under the new default.

### 1.4 Leave-one-out on a 680-name universe — **DONE**

Found while taking 1.2's acceptance. Not anticipated when this plan was written,
because leave-one-out was only ever run on 19–50 names.

`asset_robustness` in portfolio mode re-runs the whole book once per name
deleted: 680 full backtests. At 3.21s each that was **36 minutes per
`evaluate`**, before walk-forward and everything else — roughly a day of it for
a 40-iteration campaign.

**Decision taken: the gate is not weakened.** Deleting one name at a time is the
honest form of "does the edge live in one place", and a sampled, blocked or
deleted version of that question answers a weaker one. The count stays at 680;
the cost per deletion is what gets fixed.

A profile said most of a deletion was spent recomputing things that are the same
for every deletion:

| fix | what was wrong | file |
|---|---|---|
| membership mask | `member_on(day)` per symbol per session — 1.5M Python calls. An interval is a contiguous run of sessions, so two binary searches fill it | `backtest/portfolio.py` |
| benchmark curve | a Python loop over the universe inside a Python loop over the sessions | `backtest/portfolio.py` |
| carry-forward marks | `_price` rescanned the symbol's history from bar 0 on every mark, 561k times — quadratic in run length | `backtest/portfolio.py` |
| unused ATR | portfolio mode never reads `spec.exit`, yet `features()` declared the stop-loss ATR, so it was computed for all 679 names. Also latent correctness: warm-up is the max over declared features, so a short-lookback portfolio spec sat out sessions waiting for an indicator nothing consults | `dsl/schema.py` |

```
per deletion   3.21s -> 0.85s        (3.8x)
all 680        36 min -> 10 min
```

Every number is unchanged: `baseline_rs_rank_spread` still ends at 237,118 with
3,903 trades and 7,891 in fees, before and after. `tests/test_membership_gating.py`
pins the mask against the per-day predicate and the carry-forward against the
rescan it replaced, so the equivalence is asserted rather than observed. 798
tests green.

**Parallelised, and the seal survives it.** The profile was flat after the fixes
above -- rank-grid rebuild 30%, `_rebalance` 22%, engine body the rest -- so the
only lever left was the machine's other 19 cores. 680 deletions are 680
independent pure functions of the same inputs.

The reason this was not done immediately: **the seal is a per-process
singleton**. A worker gets its own, so a taint it incurred would die with the
worker and never reach the parent's ledger or hash chain -- silent
contamination, the one failure the seal exists to prevent. Worse, measured:

```
seal is a module global                      : True
does unpickling a Bars re-run the sensor?    : NO -- __post_init__ is bypassed
```

`Bars` is a frozen slotted dataclass, so pickle restores it through
`__setstate__` and never calls `__post_init__`, which is where the sensor
lives. A worker would not merely fail to report contamination; it would fail to
notice it.

So the design is that **workers never touch the seal at all**, rather than that
the seal learns to merge across processes. A worker is handed bars the parent
has already loaded and already observed; it opens no file, reaches no provider
and builds no series. Measured on the serial path first, to check the premise
rather than assume it:

```
seal before a leave-one-out pass: max_event_time / ledger / digest / tainted
seal after  a leave-one-out pass: unchanged, all four
```

A deletion materialises nothing, so there is nothing to merge back. That is the
kind of "true today" that stops being true quietly, so it is checked from both
ends: every worker returns whether its own seal came back tainted and
`_parallel_deletions` raises `Contamination` on the whole result if any did,
and `TestLeaveOneOutIsSealNeutral` pins the property on the serial path so that
a change which starts slicing bars in here fails loudly.

```
per deletion   3.21s -> 0.85s     (engine, above)
all 680        36 min -> 10 min -> 1m08s     (18 cores; DEFAULT_WORKERS = cpu_count - 2)
```

`aqr evaluate` on the full 680-name universe now takes **1 minute 8 seconds**,
against roughly eleven. Every number is identical to the serial run to the last
digit -- verdict, score `35.80966894292672`, alpha `-0.04467891296456766`,
every component, and the reason strings. Results are keyed by the name removed
and read back in sorted order, so worker count and completion order cannot
reach the report; `TestLeaveOneOutInParallel` pins serial against parallel on a
universe deliberately over `MIN_PARALLEL_DELETIONS`, because below it the
comparison would be serial against serial and would prove nothing.

**Still open, separately from speed:** whether per-name deletion *measures*
anything at this size. Dropping 1 name of 680 that a 50-name book ranks over
changes the held set on a handful of sessions, so the gate is very unlikely to
fail — not because the strategy is robust but because the deletion is small. The
speed fix above does not address this, and it should not be confused with a fix
for it. Worth reporting alongside the score: how many deletions changed the book
at all.


---

## Phase 2 — Run the search

### 2.1 Offline control campaign — **DONE on the third run**

Before spending LLM calls, run the full gauntlet on hand-written specs — the
four in the baseline table above plus a few variants.

- **Acceptance:** every one produces a complete experiment record; the low-vol
  rule reaches REVIEW (positive alpha, insignificant t) rather than PAPER; the
  reversal rule is REJECTed despite the highest raw return.

**Ran all four** (`strategies/baselines/*.yaml`, ~745 backtests each, recorded in
`runs/control-campaign.sqlite`). All four came back REJECT:

```
strategy                 verdict  score   alpha   beta      t     IR   obs
baseline_momentum_12_1   REJECT    96.7   11.1%   0.70   0.78   1.13   125
baseline_reversal_1m     REJECT    50.2  -17.2%   0.66  -2.03  -2.92   125
baseline_low_volatility  REJECT   100.0    7.2%   0.37   1.12   1.61   125
baseline_rs_rank_spread  REJECT    76.4    0.6%   0.55   0.07   0.11   125
```

**The acceptance cannot be judged from this run, because the evidence is
wrong.** Three things gave it away: every strategy reported *exactly* the same
`positive_fold_rate` of 9%, every regression used *exactly* 125 observations,
and 125 daily observations is six months when the window is 8.7 years.

**Defect A — a fold is discarded if any one universe member lacks bars.**
`_run_window` in `validation/walkforward.py`:

```python
missing = [s for s in spec.universe.symbols if s not in sliced]
if missing:
    return compute_metrics(empty), empty   # the whole fold, gone
```

`_window` drops a symbol with under 2 bars in the widened window. On a
point-in-time universe a name that had not yet listed, or had already been
acquired, is absent *by construction*. Measured per fold on
`baseline_momentum_12_1`:

```
fold  present  absent  of which not index members that day
   0      654      26      21
   3      665      15      12
   6      670      10      10
   9      677       3       3
  10      680       0       0
```

Only fold 10 survives. So `positive_fold_rate` is 1/11 = 9% **for any strategy
whatsoever** — it is measuring the calendar, not the rule. The check is right
for a fixed universe, where a missing symbol means a data hole; it is exactly
wrong for a universe whose membership is a function of the date. Note the
column that is not explained by membership (26 absent, 21 non-member): those
five *are* genuine data holes, and telling the two apart is the fix, not
relaxing the check.

**Defect B — the residual regression therefore runs on one fold.** With ten
folds discarded, `_stitch` receives a single segment, and
`stitched_timeline` is 126 entries. Every `alpha` / `beta` / `t` in the table
above is estimated from six months. That is why no strategy can be significant:
`t = IR * sqrt(n / periods_per_year) = IR * sqrt(125/252) = 0.70 * IR`, so an IR
of 1.6 reports t = 1.12 and the gate says "positive but unmeasurable" about a
sample that was never given to it. The regression maths in `backtest/alpha.py`
is correct and was checked; it is being fed a truncated sample.

**Defect C — `_window` clamps instead of excluding.** `bars.index_of_ts` returns
the series end for a timestamp past the last bar, so a delisted name is not
dropped from a later window — it contributes its final 253 bars from years
earlier. In fold 10 (test window 2023-11-20 .. 2024-05-22), **88 of the 680
symbols have sliced bars that end before the fold begins**, some in 2016:

```
ACE    2016-01-04 .. 2016-01-14   (9 bars)
AET    2017-11-28 .. 2018-11-28   (253 bars)
BBBY   2022-04-29 .. 2023-05-02   (253 bars)
earliest bar anywhere in this 'fold': 2016-01-04
```

The engine builds its timeline from the union of event times, so the one
surviving fold is an eight-year backtest whose first eight years are trimmed
off afterwards. The reported window is right; the run behind it is not what
was asked for.

**Fixed.** An absence is now accounted for by two different questions, and what
survives both is a real data hole.

*The bars* (`_window`, `_unexplained_absences`). `_window` excludes a symbol
with no bar *inside* the window instead of clamping to the series end, which
ends defect C. And a symbol whose series does not reach the window had not
begun trading; one whose series ended before it could not be traded during it.
Neither is a defect and neither fails a fold.

*The table* (`PointInTimeUniverse.unexplained_absences`, shared by
`validation/walkforward.py` and `backtest/portfolio.py`, which had the same
check at two depths). A name that was not in the index at any point in the span
is supposed to be absent.

A name that could have traded and should have been in the index, with no bars,
still fails the fold — that is the half of the check worth keeping, and a
blanket relaxation would have thrown it away.

**The bars half is not redundant.** With only the table, seven names still
failed: ALXN, CXO, ETFC, FLIR, NBL, TIF and VAR were all acquired, and
Wikipedia records the index removal one to ten days *after* the stock's last
trade:

```
ALXN  bars end 2021-07-20   table says member until 2021-07-21   (+1 day)
NBL   bars end 2020-10-02   table says member until 2020-10-12   (+10 days)
```

For those days the table calls them members and there is nothing to buy. This
is a source-boundary artefact, not a data hole, and it is resolved by believing
the bars rather than by editing the universe file — the point-in-time record
stays as the source wrote it.

**Result on `baseline_momentum_12_1`, 680 names:**

```
                     before      after
folds that ran         1/11      11/11
positive_fold_rate       9%        73%
stitched observations   126      1,376
```

The residual regression now sees 5.5 years of out-of-sample returns instead of
six months.

Pinned by `tests/test_membership_pipeline.py`: a late-listing name lets every
fold run (1/6 folds under the old rule, 6/6 under the new one — the same
pathology in miniature); an acquired name whose table entry outlasts its bars
lets every fold run; a member with no bars at all still stops all six. Plus
`TestAWindowExcludesRatherThanClamps` for defect C.

**Run 2 fixed the folds and exposed a second problem.** Two strategies were
told they had beaten buy and hold when the continuous out-of-sample curve says
they trailed. `oos_sharpe` was the *mean of the per-fold Sharpes*, compared
against a benchmark Sharpe computed over the whole window as one series. The
mean over folds ran higher on all four, by 0.06 to 0.32 — and `_stitch`'s own
docstring says why: per-fold averages hide a drawdown that spans a fold
boundary, which is the reason the chained curve exists at all.

That mismatch also sat inside a *fatal* gate. Cost sensitivity was

```
out-of-sample, with costs   /   in-sample, frictionless, full period
```

whose numerator and denominator differ by friction *and* by out-of-sample
decay, with the whole difference charged to spreads. It is why run 1 rejected
all four for "transaction costs remove 56–94% of the frictionless edge": that
was the truncated six-month sample, not costs.

**Run 3** takes `oos_sharpe` from the chained curve (`mean_fold_sharpe` keeps
the old meaning, reported but never compared), and gives the cost gate a
denominator that differs only in costs: each fold's test window is run a second
time, with the parameters the selection already chose and the costs set to
zero, chained into `stitched_zero_cost`. Re-selecting under zero costs would
let the parameters move and the comparison would stop being about friction.
Cost retention is then 85–90% across the four — costs remove 10–15%, not
56–94%. Each fold now costs one extra backtest (~9s per evaluate), and
`backtests_run` counts it, because that number is the multiple-comparisons
denominator.

The overfitting detector's own cost signal was left alone: it already compares
in-sample-with-costs against in-sample-frictionless over the same period, which
is like for like. The mismatch was only in `evaluator/score.py`, and only that
was changed.

```
strategy                    run 1             run 2             run 3
momentum_12_1      REJECT a=+11.1%   REJECT a=-4.5%    REJECT a=-4.5%
reversal_1m        REJECT a=-17.2%   REJECT a=+0.8%    REJECT a=+0.8%
low_volatility     REJECT a= +7.2%   REVIEW a=+0.5%    REVIEW a=+0.5%
rs_rank_spread     REJECT a= +0.6%   REJECT a=-2.4%    REJECT a=-2.4%

regression observations       125             1,375             1,375
cost gate fired for       all four              none              none
false "beat buy and hold"       --       two of four              none
```

**Acceptance met, on run 3.**

1. *A complete experiment record for every one* — four records, non-null
   `alpha` / `beta` / `t_alpha`, 11 folds, 1,375 out-of-sample observations.
2. *Low-vol reaches REVIEW rather than PAPER* — exactly REVIEW, for exactly the
   predicted reason: "alpha +0.5%/yr is positive but not significant
   (t=+0.16)". Beta 0.48, the lowest of the four.
3. *Reversal REJECTed despite the highest raw return* — highest CAGR (12.6%)
   and highest chained out-of-sample return (+104%), rejected on a 46%
   out-of-sample drawdown against the 35% limit, and on alpha of +0.8% at
   **beta 0.98**: its return is the exposure it ran.

Two things worth carrying forward. The three 73% fold rates are not another
structural artefact — a long-only book at beta 0.48–0.98 is profitable in
roughly the windows the market rose in, and 8 of 11 rose; `rs_rank` differing
at 64% is the one that discriminates. And the honest `oos_sharpe` let the
overfitting detector see something it could not before: low-vol now reports
"train Sharpe 0.92 vs OOS 0.60 (gap +0.32)".

Pinned by `TestTheCostGateComparesLikeWithLike`: the frictionless curve covers
the same folds, must beat the costed one once 40bp of spread is charged, is
absent unless asked for, and `oos_sharpe == stitched.sharpe != mean_fold_sharpe`.
807 tests green.


### 2.2 LLM campaign — **DONE**, all three acceptance criteria met

`aqr research --provider deepseek --universe sp500_pit --iterations 40`.

- Confirm the proposer actually emits `mode: portfolio` — the system prompt now
  pushes toward it, but the registry's 283 historical proposals are all signal
  mode, so research memory will pull the other way.
- **Acceptance:** ≥ 60% of proposals are portfolio mode; every experiment carries
  a residual regression; the campaign log records the dead-rule and repair rates
  for comparison against the 15/40, 23/40, 24/40 series in the README.

Run as campaign 07 into `runs/research.sqlite`. 40 proposals over 680 symbols:
1 ACCEPT, 3 REVIEW, 27 REJECT, 9 ERROR, summarised in `runs/README.md`. The
console log it was originally cited from is in git history.

**`--min-bars 0`, deliberately.** The default of 1500 drops 91 of the 681 cached
names, and precisely the short-lived ones — ACE at 9 bars, BRCM and PCP at 19,
GMCR at 41. Those are the acquired non-survivors: the default would have
reinstalled, in the last step, exactly the survivorship bias the whole
point-in-time apparatus exists to remove. Membership and each feature's own
warm-up decide who can be ranked; a bar count is a crude proxy for it and a
biased one.

**1. Portfolio mode: 25 of 35 compiled proposals, 71%.** Met, and met against
the pull of research memory rather than with it.

**2. Residual regression on every experiment: 28 of 28 evaluated.** None was
rejected before reaching the regression.

**3. Dead rules and repair, against the README's series:**

```
                     dead rules   recovered by repair   reached the pipeline
campaign 1 (before)      15               --                  24 / 40
campaign 2               23               12                  28 / 40
campaign 3               24               14                  29 / 40
campaign 7 (this)         5                5                  28 / 40
```

**The dead-rule rate collapsed, and portfolio mode is why.** "Never fires" is a
signal-mode failure: a boolean entry condition can be unsatisfiable on daily
bars, which is what produced 15 to 24 of them per campaign. A ranking always
produces a value, so a portfolio spec cannot be dead in that way. 71% portfolio
proposals removed most of the failure mode rather than fixing it — worth saying
in those words, because the repair mechanism did not get better.

All five that did die came back alive (5/5, against roughly half before), but on
five cases that is not a rate worth quoting.

**A new failure took its place, and it is the actionable finding.** 8 of 40
proposals never compiled, and **5 were the same mistake**: `direction:
market_neutral` with an empty `short_entry`. The proposer already retries once,
so each of those burned two LLM calls and still failed — 20% of the budget on
malformed proposals, half of it on one repeated schema error. The model reaches
for market-neutral now that it is thinking cross-sectionally and does not supply
the short leg. That is a prompt fix, not a code fix, and it is the obvious
lever for campaign 08.

Remaining errors: 2 rules whose screen never passes, 1 duplicate.

**The one ACCEPT, and why it should not be believed yet.**

```
rs_volatility_consistency_neutral_v1        ACCEPT 83.6/100, saved as PAPER
  mode      portfolio, hold 10, rebalance every 5
  rank_by   rs_rank(60) + rs_rank(20)
  screen    atr_pct(20) < 0.03 and rs_rank(60) > 0.5 and rs_rank(20) > 0.5

  alpha +10.13%/yr   beta 0.46   t +2.02   IR +0.87   R2 0.44   obs 1,375
  Sharpe 1.11 against the benchmark's 0.71
```

Economically it is the low-volatility anomaly and cross-sectional momentum in
one book: hold calm names that are above the median on both a quarter and a
month. Beta 0.46 is consistent with the volatility screen, and it is the first
rule in this project to clear `t > 2` on a full residual regression.

**And the detector says it is probably the search.** `Sharpe 1.25 deflates to
0.10 after 305 trials`. A t of +2.02 is the best of 305 hypotheses; the
threshold was written for one. This is exactly the case Phase 3 exists for, and
the right next move is to pre-register it and spend the seal — not to believe
the 83.6.

Note also that the ACCEPT is recorded against a registry whose
`distinct_hypotheses()` is now 305 and rising, and that this is the honest
denominator: the 288 that preceded this campaign were real selection events on
this project, even though most were on the old universe.

### 2.3 Search-cost accounting for portfolio mode — **DONE**

`distinct_hypotheses()` is the multiple-comparisons denominator. Confirm a
portfolio spec's fingerprint changes with `rank_by` / `hold` / `rebalance_every`
and that perturbations are not counted as hypotheses.

- **Acceptance:** a test pins that 40 parameter perturbations of one portfolio
  spec add 1 to `distinct_hypotheses()`, not 40.

**The accounting was already right, and is now pinned.** A portfolio spec's
fingerprint moves with each of `rank_by`, `hold` and `rebalance_every`, so two
different rules cannot collide and under-count the search. Evaluating one
portfolio rule runs dozens of backtests and adds exactly **1** to
`distinct_hypotheses()`. `TestPerturbationsAreDiagnosticsNotHypotheses` also
pins that the perturbations *would* count if they were recorded — each
neighbour is a distinct fingerprint — so the acceptance is not passing because
the variants happen to collide, but because nobody chose between them.

**What checking it turned up: two thirds of a portfolio spec's parameter
stability was paid out for knobs its engine never reads.** `slots()` returned
every field on the dataclass, so a portfolio spec exposed ten:

```
hold                        2 neighbours   2/2 change the book
rebalance_every             4              4/4
rank_by#0, rank_by#1        6              6/6
exit.stop_loss.multiplier   4              NO-OP
exit.stop_loss.period       4              NO-OP
exit.take_profit.ratio      4              NO-OP
exit.max_holding_bars       4              NO-OP
sizing.risk_per_trade       4              NO-OP
sizing.max_position_pct     4              NO-OP
```

`run_portfolio` consults neither `exit` nor `sizing` — the book is chosen by
rank and turned over on the rebalance, so no stop fires and no position is
sized on its own. Perturbing those six produced bit-identical backtests, and
`parameter_stability` — 15% of the score — reads identical as *stable*. 24 of
36 perturbations were marks awarded for being unperturbable.

The comment on `_STRUCTURED` already names this exact failure and was written
when `hold` and `rebalance_every` were added to fix it; leaving the dead knobs
in was the other half of the same bug. The mirror image was true too: signal
specs were perturbing `hold` and `rebalance_every`, which the signal engine
ignores.

`slots()` now returns only what the spec's mode actually reads. Every
perturbation moves the book — pinned by
`TestEverySlotIsOneTheEngineReads::test_every_perturbation_actually_moves_the_book`,
which fails if any neighbour ever produces an identical equity curve again.
A side effect: 24 fewer wasted backtests per portfolio evaluation
(`backtests_run` 67 → 43 on the test spec).

821 tests green.

**Carried forward, not fixed:** `run_portfolio` also ignores `spec.regime`, and
nothing in the schema rejects a `regime` on a portfolio spec. It is excluded
from `slots()` now, so it no longer inflates stability, but a spec can still
declare a regime filter that is silently not applied. `max_positions` is the
opposite case — a knob the signal engine does read that `_STRUCTURED` does not
list, so it is never perturbed.

---

## Phase 3 — Pre-registration and the sealed run — **DONE**

The embargo apparatus existed and had never been spent. It has now been spent,
once, on `rs_volatility_consistency_neutral_v1`. **The candidate was not
refuted.**

```
rs_volatility_consistency_neutral_v1 [3f6e2c8a9309068b]
sealed window 2024-09-03 -> 2026-08-27, 498 sessions, 628 of 680 symbols

  strategy   return +56.39%   sharpe +1.86   maxDD -10.40%   trades 561
  benchmark  return +32.38%   sharpe +0.98
  residual   alpha +16.72%/yr  beta 0.43  t +2.22  IR +1.58  R2 0.30
```

**What this is and is not evidence for.** 498 sessions puts the standard error
on an annualised Sharpe at ±0.71, which is most of the gap being reported. This
run could have refuted the rule and it did not; that is the whole of what it
establishes. The evidence for the rule remains the walk-forward inside the
search window, and that evidence is still discounted by
`Sharpe 1.25 deflates to 0.10 after 305 trials`. `SealedMeasurement.can_confirm`
returns `False` unconditionally so nothing downstream can quietly upgrade this.

The checkable part is the part that was predicted in advance and came back the
same: **beta 0.43 against 0.46 in the search window.** A volatility screen
producing a low-beta book is a mechanism, and the mechanism held on unseen data.
Alpha rising from +10.13%/yr to +16.72%/yr is not corroboration of anything — it
is one draw from a distribution whose standard error is wider than the change.

### 3.1 Registry: pre-registration and one-shot enforcement — **DONE**

New `preregistration` table (`fingerprint` as primary key, `declared_at`,
`selection_rule`, `seal_digest`), new `sealed_run_at` and `sealed_result`
columns on `strategies`, and `PreregistrationError` as its own type so the CLI
can distinguish "wrong fingerprint" from "already spent".

`aqr preregister FINGERPRINT --rule "..."` declares; `python -m aqr.cli_sealed
run FINGERPRINT` spends. Two binaries, so the ordering is visible in the shell
history and not only in the database. `aqr preregistered` lists what has been
declared and what has been spent.

Refused, each with a test: a blank selection rule, an unknown fingerprint, a
second declaration, a sealed run on an undeclared candidate, and a second sealed
run. The second sealed run is refused rather than overwritten — once the first
result can be discarded, "we re-ran it" and "we re-ran it until it worked" are
indistinguishable from outside.

**Old databases migrate rather than break.** `_MIGRATIONS` adds the new columns
with `ALTER TABLE` when they are absent, pinned by a test that builds a
pre-Phase-3 schema by hand and opens it. 18MB of campaign history is the
multiple-comparisons denominator and is the one number here that cannot be
reconstructed.

### 3.2 Ancestry taint check — **DONE**

The seal gained a `run_id`: one process, one seal, one campaign. It is stamped
into `certificate()` and deliberately kept out of the digest, so two campaigns
that read the same bars in the same order still hash the same.

`Registry.ancestry_taint(fingerprint)` walks every experiment on that
fingerprint plus every sibling sharing a `run_id` with one of them, because
**the unit of contamination is the campaign, not the backtest**: a process that
read an embargoed bar while evaluating hypothesis 12 was still contaminated when
it evaluated hypothesis 13. `pipeline._record` now writes
`current().certificate()` with every experiment, so the check is a query rather
than an act of trust.

**Reported honestly on the history that predates it.** `TaintReport` carries
`unrecorded` separately from `tainted`, and the real database returns
`clean: 2 experiments across 0 campaign(s), 2 with no seal recorded`. Silence is
not evidence of innocence, so the sealed run refuses to proceed on an unrecorded
ancestry unless `--allow-unrecorded-ancestry` is passed — which it was, and which
is on the record inside the stored result. Every campaign from here carries
certificates and the flag stops being needed.

### 3.3 `cli_sealed.py` — the separate entry point — **DONE**

`aqr-sealed` (`python -m aqr.cli_sealed`), with `audit`, `pull`, `declared` and
`run`. Boundary tests: it cannot import `aqr.agent`, it cannot import `aqr.cli`
(which carries `research`), and it must construct a `SealedProvider`.

`Seal.enter_sealed()` is the phase transition, one-way, with no `enter_research`
and no public setter — and **refused once the process has read anything**.
Without that condition the sequence "search, then promote, then read the answer"
would produce a certificate saying `phase: sealed, tainted: false`, which is
precisely the claim the seal exists to make unfalsifiable. The promotion is
folded into the digest, so a research certificate can never be mistaken for a
sealed one. `test_only_the_sealed_entry_point_promotes_the_process` fails the
build if any other module calls it.

`run` checks in protocol order — declared, then unspent, then untainted, *then*
promote and read. A rejected candidate costs no seal at all, because nothing has
been read when the rejection happens.

The measurement itself is `src/aqr/validation/sealed.py`, separate from the
process allowed to run it, so it is testable without any of the phase
machinery.

**The bars start before the window; the measurement does not.** `rs_rank(60)`
needs sixty sessions of history, so the backtest is handed 1200 calendar days of
warm-up and only the *scoring* is clipped to the embargoed span. Truncating the
bars instead would leave the first sixty sealed sessions trading an unwarmed
signal — a different strategy than the one that was validated. Pinned by
`test_the_warmup_history_is_used_but_not_measured`. The benchmark is measured
over the same window for the same reason: a two-year strategy Sharpe against a
nine-year benchmark Sharpe compares two market regimes wearing one label.

### 3.4 Build the sealed cache — **DONE**

`data-sp500-sealed/` holds 682 series through 2026-08-27, 598 of them with rows
past the embargo. `aqr seal-check` now audits the pair by default:

```
root               files  latest bar  past embargo  canary
data-sp500         681    2024-08-30  none          armed
data-sp500-sealed  682    2026-08-27  598           -
```

Its expectation rule was also wrong and is fixed: it keyed on `"research" in
name`, which marked every research root without that word in its name red.
It now keys on `"sealed" not in name`.

**Three defects surfaced only because this pull was attempted, and all three
would have silently produced a smaller universe.**

1. **The tape is served on a delay.** A request whose end is inside that delay
   is refused `403` for *every* symbol, not partially — the first attempt
   fetched 0 of 682. `FEED_DELAY_DAYS = 1`: "today" means yesterday here, and
   the pull and the run share the constant so the run cannot score sessions the
   cache lacks.
2. **`429` was treated as a hard failure.** 682 sequential requests hit the rate
   limiter, and repeating the pull re-hit it at the same point: after ten runs,
   41 tickers were permanently unfetchable. `AlpacaProvider` now retries the
   statuses that mean *later* (`429`, `5xx`) with exponential backoff and does
   not retry the ones that mean *no* (`403` entitlement, `404`). One run now
   fetches everything. The clock is injected, so the tests cost no seconds.
3. **78 series fail the quality check** — the unadjusted spin-offs and mergers
   already recorded under *Known gaps*. `data-sp500` kept them, so the sealed
   cache must keep them too: the sealed window has to differ from the search
   window in dates and in nothing else. Pulled with `--keep-suspect`.

Two remaining asymmetries between the pair, neither touching the strategy's
universe: `PSKY` and `SNDK` now return bars and are in the sealed root only;
`RE` is in `data-sp500` only and is not a member of the PIT universe at all (it
was renamed `EG`).

### 3.5 Spend it once — **DONE**

Run above. 52 of the 680 universe symbols returned no sealed bars — every one of
them delisted before the 2021-05-20 warm-up start, so this is the point-in-time
universe working rather than a hole in the cache.

`distinct_hypotheses()` read **324** at the time of this declaration, not the
305 quoted earlier: campaign 07 and the control campaigns added more. 324 is the
honest denominator for that look and is what the declaration records. It has
since grown to **414** — later campaigns, and the four looks that followed this
one. Each declaration records the denominator that applied to it; the numbers do
not get retconned.

The stored result carries the measurement, the seal certificate, the
declaration, and the ancestry report, so the run can be audited without trusting
this write-up. `runs/research.sqlite.pre-phase3.bak` is the database as it stood
before the seal was spent.

**Recorded, not fixed:** the seal proves the embargoed *data* was not read. It
does not prove the embargoed *period* did not inform a decision. The researcher
lived through 2024-09 to 2026-08 and every model in `aqr providers` has a
training cutoff after it. `certificate()["knowledge_exposure"]` records that
rather than denying it, and no claim built on this run may be worded as
"uncontaminated".

---

## Phase 4 — Hand the strategy off; do not place the order — **DONE**

**Decided 2026-08-28: `ai_quant_researcher` does not place orders.** The question
the previous draft of this phase left open has been answered, and the answer is
the opposite of what that draft assumed. There is no `src/aqr/live/` and no
Alpaca client in this project. What this project produces is a *target book*;
placing it is the outer layer's job.

That keeps the README sentence under *What this does not do yet* — "Order
placement — and it never will, here. This project finds strategies; execution is
a different system with different invariants" — true as written, rather than
requiring it to be argued away.

It also keeps `CLAUDE.md` §2 intact in the direction that matters. §2 says the
two projects are isolated and `ai_quant_researcher` must not import `alphagate`.
A handoff respects that: this project writes a file, and whatever executes reads
it. Neither imports the other, and the file is the whole interface.

**What the outer layer inherits, and what it must supply.** AlphaGate's
`GatedOrder` and every rule in `specs/03-risk-gate.md` are options-shaped —
short strikes, DTE, defined-risk width — and none of it applies to an equity
book. So the outer layer is not "AlphaGate as it stands"; whatever consumes the
target book needs an equity-shaped risk gate of its own. Naming that gap is part
of this phase. Filling it is not.

### 4.1 Target book from the same code path — **DONE**

`src/aqr/target_book.py`. `build_target_book` calls `run_strategy` — the same
entry point the backtest, the walk-forward, the robustness passes and the sealed
run all go through — and takes `weights_at(last)`. There is no second
implementation of the selection anywhere.

`test_the_book_matches_what_the_backtest_held` is the acceptance test: bars
truncated at a historical session, handed off, and compared against the weights
the full backtest held on that session. Core, sleeve and combined all match.

A signal-mode spec is refused. A trigger strategy's positions are a consequence
of fills rather than a set of weights, and inventing weights for one would hand
off a portfolio interpretation of a strategy nobody validated as a portfolio. A
run too short to clear its warm-up is refused separately: an empty book reads as
"hold nothing", which is a position, and a run that never filled has not decided
to hold nothing — it has not decided anything.

**One session of lag, named rather than hidden.** The engine decides at a close
and fills at the next open, so the weights in force on the final session were
decided on the session before it. An executor placing them at the next open acts
one session later than the backtest did. The alternative — reporting the
decision made *at* the final session — means re-implementing the selection
outside the engine, which is the one thing 4.1 exists to prevent. Recorded in
the artefact's `fill_convention` field so the consumer reads it rather than
inferring it.

### 4.2 The handoff artefact — **DONE**

JSON, `schema_version: 1`, validated on write as well as on read: a malformed
book that reaches disk is a book something downstream will try to execute, and
the cheapest place to stop it is before it is written.

It carries the fingerprint, the spec name and version, the `as_of` session (the
session the weights were in force on — not the wall clock), the dataset version,
the universe, the timeframe, how many of the declared symbols actually loaded,
the seal certificate, and the provenance from 4.4. Weights are reported three
ways — combined, core, sleeve — all sorted, so the file is byte-reproducible and
`book_digest` can detect one that was edited after the handoff.

`consumer_must_supply` is a field *inside* the artefact rather than a line in
this document: a boundary recorded only in a design document is one the first
person to consume the file will not see.

**Acceptance, both halves.** The schema test refuses a missing or mistyped
field, a version it cannot interpret, a position count that disagrees with the
weights, and — by name — any `shares`, `quantity`, `notional`, `equity` or
`orders` field, because the first one to appear would be this project sizing a
position. The boundary test in `tests/test_boundaries.py` refuses a trading host
(`api.`, `paper-api.`, `broker-api.alpaca.markets`), an order path
(`/v2/orders`, `/v2/positions`, `/v2/account`), an account credential name, and
an `alpaca` / `alpaca_trade_api` / `alphagate` / `ib_insync` import, anywhere
under `src/aqr/`. `data.alpaca.markets` is allowed, and a second test states
that allowance explicitly rather than leaving it implied — reading bars is the
job.

### 4.3 The runner — **DONE**

`aqr target-book <fingerprint-or-path>` writes the artefact and prints its path.
No `--dry-run`. A path is accepted for convenience and then held to the same
standard as a fingerprint: it has to resolve to a strategy the registry knows.

Run on the promoted candidate against `data-sp500-sealed`:

```
604 of 680 symbols, 2023-05-17 -> 2026-08-29
rs_volatility_consistency_neutral_v1 [3f6e2c8a9309068b] as of 2026-08-27
  104 positions, gross 1.0000 (core 0.8000, sleeve 0.2000)
  core: ABNB AMGN BDX IVZ MPC MRK PSX REGN TGT VRTX, 0.08 each
runs/target_books/rs_volatility_consistency_neutral_v1-3f6e2c8a9309068b-2026-08-27.json
```

**Three refusals, in this order:** the registry has to know the strategy, its
seal has to have been spent, and the sealed run must not have refuted it. The
middle one is the one that matters. A book handed off before the seal is spent
would read the embargoed years for a rule that still owes an out-of-sample
verdict — the exact loophole pre-registration exists to close.

**Producing a book taints the seal, and the certificate says so.** Reading the
present is what a handoff *is*; the taint records that this process read past the
embargo, which is honest rather than a defect. What the process must never also
be is the search, and it cannot be: the seal is already spent before a book can
be written, so there is no unspent answer left to leak.

### 4.4 Provenance — **DONE**

A `target_books` table in the registry: fingerprint, `as_of`, path, sha256
digest, and the whole book. Appended, never replaced — two books for the same
session are two handoffs, and if the second differs from the first, overwriting
is the one thing that would hide it. `aqr target-books` lists them.

The artefact carries its own provenance as well, so a reader who has never heard
of `aqr` needs neither the database nor this file: the hypothesis, the registry
status and score, the pre-registration and its selection rule, the sealed
measurement, the sealed run's own seal certificate, and both denominators the
claim has to be discounted by — the 324 hypotheses the search compared
(`distinct_hypotheses`) and the number of candidates the sealed window has
screened (`sealed_looks_total`).

### 4.5 Not in this project

Recorded so the boundary is a decision and not an omission: reconciliation
against live positions, order sizing, turnover and notional caps, the kill
switch, the fill journal, and the equity risk gate. All of these belong to
whatever consumes the artefact from 4.2 — and all of them are listed inside the
artefact under `consumer_must_supply`, so the consumer inherits the list rather
than the assumption.

---

## Phase 5 — Ship — **DONE**

### 5.1 README — **DONE**

Four sections, each promised by an earlier phase and each now written where it
was promised:

- **The universe was survivorship-biased, and it was worth 16 points a year.**
  The NASDAQ-50 survivor benchmark showed CAGR 28.2% / maxDD -29.3%; the
  point-in-time S&P 500 shows **12.2% / -40.0%**. Every earlier "beat the market"
  judgement was made against a baseline inflated by roughly sixteen points. The
  section also states what membership gates (holdings, the ranking cross-section,
  *and* the benchmark), the 3.8-point measured effect of the gate, and the three
  names recorded as *not corrected* rather than quietly fixed.
- **Alpaca does not adjust for spin-offs or merger share exchanges.** `RTX`
  -70.3% on 2020-04-03 (Raytheon/UTC merger plus the Carrier and Otis spin-offs)
  and `TGNA` -79.2% on 2017-06-01 (Cars.com) are corporate actions, not market
  moves. Seven names carry one inside a tradable interval. Not corrected, with
  the defence stated: the profit-concentration signal is what catches a result
  leaning on one, it is weaker than fixing the data, and it says so.
- **The sealed run.** The measurement as it came out, what "not refuted" is and
  is not entitled to mean, and the one part that was falsifiable in advance —
  beta predicted 0.46, returned 0.43.
- **The handoff.** Written in Phase 4, under the sentence *What this does not do
  yet* already promised.

### 5.2 Cost model recalibration — **DONE**

Measured before it was changed. The model was calibrated for a three-position
event-driven book; the strategy that survived holds 110 names. Two of the four
charges do not scale with notional, so the same rule on the same bars is priced
differently by account size:

```
      equity     CAGR   Sharpe   frictionless   Sharpe retained
     100,000   15.94%     1.15           1.58              73%
   1,000,000   20.50%     1.43           1.58              91%
  10,000,000   20.85%     1.46           1.58              92%
```

At $100k a sleeve position is $192 and a $1.00 order floor on $192 is **52 basis
points**, against the 3bp the spread and slippage charge. The floor bound on 1262
of 2317 core round-trips. At `rebalance_every: 1` the fatal gate flips outright
(-67% retained at $100k, 32% at $1M) — the model is doing its job on genuine
turnover, but the *magnitude* is set by `initial_equity`, a number nobody thinks
of as a cost parameter.

**What was not done: making the model cheaper.** A trader with $100k and a
hundred-name book at a broker charging a dollar an order really would pay 52bp,
and a model that hid it would be lying in the flattering direction. Three things
were actually wrong, and all three are fixed:

- **The schedule was implied.** Now named. `IBKR_FIXED` *is* `CostModel()` —
  byte-identical defaults, so the 324 recorded verdicts stay reproducible;
  `ALPACA_EQUITIES` is commission-free, the venue these bars come from and whose
  paper account the target book is built for. Spread and slippage are unchanged
  between them: they are properties of the market rather than of the broker, and
  they are where the pessimism belongs. `--costs alpaca` selects one, and a named
  schedule wins outright rather than merging with the individual knobs.
- **The per-order cost could not be asked for.** `CostModel.price_order` returns
  the fee, the adverse charge, the all-in bps and whether the floor bound.
  `aqr costs --equity 100000 --positions 110` prints it for every schedule.
- **The schedule was not recorded with the verdict.** Cost retention is a fatal
  gate, so the schedule is part of the verdict rather than context for it. It is
  now written into every experiment, the same way `dataset_version` is.

**The default is deliberately unchanged.** Recalibrating by editing the defaults
would silently reinterpret every experiment already recorded, and a cost model
that gets cheaper the year a strategy needs it to is not a cost model.
`test_the_default_schedule_is_unchanged` pins it.

### 5.3 End-to-end — **DONE**

`tests/test_end_to_end.py`: `research → evaluate → preregister → sealed run →
target-book`, the real CLI commands, against a temporary cache and a temporary
registry, asserting after each stage that the artefact it was supposed to leave
behind is there and carries the fingerprint the next stage will look for.

Offline throughout, so it needs no API key and no network, and it is part of the
suite rather than a script somebody has to remember to run.

**Every stage runs in its own `scope(Seal())`, because in real life every stage
is its own process** — and the first version of this test proved why. Writing the
fixture cache taints whichever seal is ambient, because the cache spans the
embargo; sharing that seal with the research stage made `preregister` refuse the
candidate for a tainted ancestry. The seal was right and the test was wrong. A
process that has seen the embargoed years may not also be the process that
searches, and building the cache is exactly such a process.

The last stage asserts the ambient seal is still `RESEARCH` afterwards: if either
sealed stage leaked, the boundary the whole project rests on would have been
broken by its own test suite.

One convention it pins on the way past: the data window is half-open, so
`--end 2026-08-27` stops at the session *before* it, and `as_of` names the last
session the book actually saw rather than the date that was asked for. An
executor reconciling against a session that was never in the book is the failure
that field exists to prevent.

---

## Known gaps carried forward, not silently

- **`WRK` is missing** from the universe. WestRock was an index member for most
  of the window; Wikipedia's change log has no row for it or its successor `SW`.
  Filling it would mean inventing interval boundaries from no source. Recorded in
  `data-universes/sp500_pit.json` under `not_corrected`.
- **`D`, `SRE`, `TROW`** have unverified interval starts, same file.
- **Alpaca does not adjust for spin-offs or merger share exchanges.** Seven names
  carry an unadjusted discontinuity inside a tradable interval — `RTX` -70.3% on
  2020-04-03, `TGNA` -79.2% on 2017-06-01. Not corrected; the overfitting
  detector's profit-concentration signal is what would catch a result that leans
  on one. Written up in the README rather than only here.
- **No 2008-scale event.** The window holds three real drawdowns — 2018 Q4
  (-19.8%), 2020 (-29.3% on NDX-50, -40.0% on the PIT S&P), 2022 (a full year at
  Sharpe -0.53) — but nothing of 2008's depth. IBKR cannot supply it: its
  delisted contracts resolve by conId but carry no EOD chart data, so a 2008 run
  there would be the most survivorship-biased possible.
- **`PSKY` and `SNDK`** returned no Alpaca bars from 2016; 681 of 683 cached.
- **A target book is one session behind the backtest.** The engine decides at a
  close and fills at the next open, so the weights in force on the final session
  were decided on the session before it, and an executor placing them at the next
  open acts one session later than the simulation did. Closing that gap means
  re-implementing the selection outside `run_portfolio`, which would cost the
  guarantee that the handoff and the backtest describe the same strategy. Named
  in the artefact's `fill_convention`, not fixed.
- **Cost is a function of book breadth and account size, and only one of those
  is a property of the strategy.** The per-order floor makes `initial_equity` a
  cost parameter: Sharpe retained goes 73% at $100k and 92% at $10M on the same
  rule. Named schedules and `aqr costs` make it visible and the record makes it
  auditable; nothing makes it go away, because at a small account on a wide book
  it is real. A book of this shape needs either a larger account or a
  commission-free venue, and that is a fact about the strategy worth carrying.
- **The sealed window is a screen, and every use of it narrows what a pass
  means.** One shot is per candidate, so the research loop may keep proposing new
  hypotheses and validating them — but the survivor of an N-way screen is a
  weaker claim than the survivor of a one-way screen. The count is recorded and
  the alpha's bar rises with it (1.96 at one look, 2.58 at five, 3.02 at twenty);
  nothing refuses the Nth look. The only real way back to a one-look window is to
  wait for new sessions to accumulate.
- **The seal proves data was not read, not that knowledge was not used.** The
  researcher lived through the embargoed period and every model in `aqr
  providers` has a training cutoff after it.
