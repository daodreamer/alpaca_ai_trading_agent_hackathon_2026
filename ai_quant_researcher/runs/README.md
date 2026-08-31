# `runs/` — what is here, and what it is evidence of

This directory used to hold seven console logs, 254 KB of scrollback across
sixteen retried pull invocations. The information in them mattered; the format
did not. This file is that information, and the logs are in git history if
anyone needs the per-symbol detail.

Everything below is derived from the registry and from the caches on disk, not
retyped from the logs. Regenerate the tables with `scripts/report_benchmark.py`
and a read of `runs/research.sqlite`.

---

## Files

| File | What it is |
| --- | --- |
| `research.sqlite.gz` | The registry, gzipped. 413 strategies, 441 experiments, 8 pre-registrations, 5 sealed runs. `python scripts/pack_registry.py --unpack` restores it. |
| `target_books/` | Every target book handed to the execution side, one file per generation. This is the seam: `backend/` reads these and imports nothing else from here. |
| `README.md` | This file. |

Ignored, local-only: `research.sqlite` (the live 51 MB database),
`research.sqlite.pre-phase3.bak`, `control-campaign*.sqlite`, `control2/`,
`control3/`, and any `campaign-*.log` a future run writes.

---

## The caches

None of these are in the repository. The two embargo roots are 1.3 GB, and
Alpaca restates history, so a clone cannot rebuild them byte for byte — which is
why the coverage is written down here rather than left to be re-derived.

| Root | Symbols | Range | Bars |
| --- | --- | --- | --- |
| `data-sp500/1D` | 682 | 2016-01-04 → 2024-08-27 | 1,332,011 |
| `data-sp500/1h` | 628 | 2016-01-04 → 2024-08-27 | 7,330,501 |
| `data-sp500-sealed/1D` | 682 | 2021-05-20 → 2026-08-27 | 780,109 |
| `data-sp500-sealed/1h` | 580 | 2024-08-28 → 2026-08-28 | 1,686,049 |
| `data-benchmark/1D` | 1 (SPY) | 2016-01-04 → 2026-08-28 | 2,679 |
| `data-benchmark/1h` | 1 (SPY) | 2016-01-04 → 2026-08-28 | 16,074 |

`data-benchmark/` is the only one committed — 1.9 MB, so
`scripts/report_benchmark.py` runs from a fresh clone.

**Why the counts differ.** 682 is every ticker ever a member of `sp500_pit`,
including the ones that left; that is the point, since the names that left are
exactly the ones whose absence creates survivorship bias. The shortfalls are:

- **`data-sp500/1h`, 628 of 682.** Intraday history is shorter than daily, and
  54 names either delisted before Alpaca's 1h coverage begins or failed the
  quality check (a gap in the series is worse than no series — the backtester
  indexes positionally, so a hole is invisible to every defence downstream).
- **`data-sp500-sealed/1h`, 580 of 682.** Same, plus a narrower window: the
  sealed 1h pull was restarted twice, from 2016-01-01 to 2021-05-01 to a final
  2024-06-01, which is two years of out-of-sample plus three months of warm-up.
- **`data-sp500-sealed/1D` starts 2021-05-20, not 2016.** The sealed cache only
  needs the sealed window plus enough warm-up for the longest feature.

**The 1D sealed pull took sixteen invocations.** Alpaca returned 429 under a
682-symbol sweep, so the pull was re-run until it converged: 428 fetched, then
106, 30, 19, 8, 7, 5, 2, and then six runs that fetched nothing and settled at
605 cached. A seventeenth pass found 77 series that had failed the quality check
and refetched them. Each invocation recorded its own seal digest; the last was
`6937345987ccb8f5`, and the quality-check repass was `14970a744b049382`.

Retrying does not weaken the embargo. `aqr pull` clamps every request at
2024-09-01 before it is sent, and the sealed root is filled by a separate binary
(`aqr-sealed`) in a separate process that cannot search.

---

## The sealed runs

Five, all on 1D bars, all against the same 498-session window
2024-09-03 → 2026-08-27. **Each look raises the significance bar for the next**,
which is why the table has a `bar` column and why a raw `t` is not enough.

| # | When | Strategy | Declared as | return | sharpe | maxDD | t | bar | clears |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 08-28 22:27 | `rs_volatility_consistency_neutral_v1` | sole ACCEPT of campaign 07 | +56.39% | +1.86 | -10.40% | +2.22 | — | — |
| 2 | 08-30 18:59 | `low_vol_rs_carry_v3` | "highest scoring candidate" | +23.28% | +2.07 | -2.19% | +2.63 | 2.241 | yes |
| 3 | 08-30 19:06 | `low_vol_relative_strength_carry_v1` | "test warmup output" | +18.69% | +1.42 | -3.88% | +1.63 | 2.394 | **no** |
| 4 | 08-30 19:15 | `low_vol_relative_strength_carry_v1_improved` | "backfill test 3" | +63.50% | +2.70 | -4.28% | +3.73 | 2.498 | yes |
| 5 | 08-30 20:16 | **`low_vol_rs_carry_v5`** | "low_vol_rs_carry_v5 with 99 score" | +51.88% | +2.22 | -3.71% | +2.94 | 2.576 | yes |

None was refuted. Run 5 is the pinned strategy —
`ALPHAGATE_STRATEGY_FINGERPRINT=96cbc95ab6f09a60`.

**Three things this table admits rather than hides.**

*Runs 2 through 4 were declared with debug selection rules.* "test warmup
output", "backfill test 3" — those are not selection rules, they are the
messages someone types while developing the sealed-cache backfill. Spending a
one-shot window to test a code path is exactly the "look until you like it"
pattern pre-registration exists to prevent. What kept it honest is that the
machinery charged for every look anyway: the bar climbed 2.241 → 2.394 → 2.498 →
2.576 across them, and run 5 had to clear the highest of them.

*Run 4 beats run 5 on every line.* Higher return, higher Sharpe, a larger t. It
is not the pinned strategy because it was declared as "backfill test 3" — a
candidate whose declaration says it was a debug invocation cannot then be
promoted on the strength of the result it happened to produce. Picking it after
seeing the number is the thing the seal exists to stop.

*Run 1's `t` of +2.22 would not clear today's bar.* It cleared as the only
candidate ever screened against this window. Once the window had been opened
five times, 2.22 sits under 2.576. It is retired, and that is why.

**What none of this upgrades.** `can_confirm` is `False` by construction — the
standard error on an annualised Sharpe over 498 sessions is ±0.71. The search
denominator is 414 distinct hypotheses and `v5`'s Sharpe of 1.97 deflates to
0.74 after 411 trials. A sealed window that cannot confirm cannot repair a
search that was wide.

---

## Campaign 07

Cited by `PLAN.md` as the campaign that produced the first sealed candidate. 40
proposals over 680 symbols on 1D bars: **1 ACCEPT, 3 REVIEW, 27 REJECT, 9
ERROR**. The ACCEPT was `rs_volatility_consistency_neutral_v1`
[`3f6e2c8a9309068b`] at 84/100 — run 1 in the table above, now retired.

The registry across all campaigns reads 12 ACCEPT, 14 PAPER, 39 REVIEW, 348
REJECT, 28 ERROR over 441 experiments. Failures are recorded on purpose:
forgetting the attempts that went nowhere is the mechanism by which a search
looks luckier than it was.
