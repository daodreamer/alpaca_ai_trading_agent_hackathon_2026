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
| `research.sqlite.gz` | The registry, gzipped. 585 strategies, 613 experiments, 9 pre-registrations, 6 sealed runs (5 equity, 1 option). `python scripts/pack_registry.py --unpack` restores it. |
| `target_books/` | Every target book handed to the execution side, one file per generation. This is the seam: `backend/` reads these and imports nothing else from here. |
| `option_books/` | The same seam for the option sleeve: the rule, deliberately without strikes. |
| `README.md` | This file. |

The counts above are the packed database as committed. `python
scripts/pack_registry.py` re-packs it from the live `runs/research.sqlite`, and
should be re-run whenever the live one has moved — the two drifted once already,
which is why this note exists.

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

Six in total, against **two separate windows with two separate counters**. The
equity window has been opened five times; the option window once. They are
counted apart because specs/10 D8 requires it — charging the option side for the
equity side's five looks would be Bonferroni applied across two different
searches — and `aqr preregistered` prints the two bars separately for the same
reason.

### The equity window

Five, all on 1D bars, all against the same 498-session window
2024-09-03 → 2026-08-27. **Each look raises the significance bar for the next**,
which is why the table has a `bar` column and why a raw `t` is not enough.

| # | When | Strategy | Declared as | return | sharpe | maxDD | t | bar | clears |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 08-28 22:27 | `rs_volatility_consistency_neutral_v1` | sole ACCEPT of campaign 07 | +56.39% | +1.86 | -10.40% | +2.22 | — | — |
| 2 | 08-30 18:59 | `low_vol_rs_carry_v3` | "highest scoring candidate" | +23.28% | +2.07 | -2.19% | +2.63 | 2.241 | yes |
| 3 | 08-30 19:06 | `low_vol_relative_strength_carry_v1` | "test warmup output" | +18.69% | +1.42 | -3.88% | +1.63 | 2.394 | **no** |
| 4 | 08-30 19:15 | **`low_vol_relative_strength_carry_v1_improved`** | "backfill test 3" | +63.50% | +2.70 | -4.28% | +3.73 | 2.498 | yes |
| 5 | 08-30 20:16 | `low_vol_rs_carry_v5` | "low_vol_rs_carry_v5 with 99 score" | +51.88% | +2.22 | -3.71% | +2.94 | 2.576 | yes |

None was refuted. **Run 4 is the pinned strategy** —
`ALPHAGATE_STRATEGY_FINGERPRINT=9b4ac85c149ec6db`.

The bar is `z(1 - 0.05 / looks / 2)`: a two-sided 5% level split Bonferroni-wise
across every look the window has taken. Two looks put it at 2.241, five at
2.576, and it keeps climbing for as long as anyone keeps opening the envelope.

**Three things this table admits rather than hides.**

*Runs 2 through 4 were declared with debug selection rules.* "test warmup
output", "backfill test 3" — those are not selection rules, they are the
messages someone types while developing the sealed-cache backfill. Spending a
one-shot window to test a code path is exactly the "look until you like it"
pattern pre-registration exists to prevent. What the machinery does charge for
is the looks themselves: the bar climbed 2.241 → 2.394 → 2.498 → 2.576 across
them, so each debug run made the next candidate's job harder.

*Run 4 is pinned, and it was chosen after its result was visible.* This is the
weak point in the chain and it is not going to be dressed up. Its declaration
says "backfill test 3": the seal was spent exercising the cache-backfill code
path, and at the time the operator could not see what the run had produced. The
number was read later, it was the best of the five, and the pin moved to it.
That is selection on the outcome, which is precisely what pre-registration
exists to prevent, and none of the bars in the table above charges for it —
`looks` counts how many times the window was opened, not how many times a result
was looked at before choosing.

What can honestly be said in its defence is the size of the margin rather than
the process. `t +3.73` clears the bar at six looks (2.638), at eight (2.734),
and on out to **261 looks** before the Bonferroni correction catches it. A
selection effect large enough to manufacture that from noise would have to be
two orders of magnitude bigger than the search this project actually ran. The
procedural objection stands; the statistical one does not survive the
arithmetic.

*Run 1's `t` of +2.22 would not clear today's bar.* It cleared as the only
candidate ever screened against this window. Once the window had been opened
five times, 2.22 sits under 2.576. It is retired, and that is why.

**What none of this upgrades.** `can_confirm` is `False` by construction — the
standard error on an annualised Sharpe over 498 sessions is ±0.71. The search
denominator is 414 distinct hypotheses, and the pinned rule's own record says
its search-window Sharpe of 1.81 deflates to **0.59 after 402 trials**. A sealed
window that cannot confirm cannot repair a search that was wide.

Its one genuinely reassuring line is `train_oos_gap`: train Sharpe 1.84 against
OOS 2.00, a gap of **-0.15**. The rule did better out of sample than in it,
which is not what an overfit rule does.

### The option window

One, spent 2026-09-01 on `iv_rank_low_sticky_put_credit_spread_v1`
[`cc197008e0deb097`] — the pinned option rule,
`ALPHAGATE_OPTION_FINGERPRINT=cc197008e0deb097`.

| Window | 2024-09-03 → 2026-08-31, 500 observations |
| --- | --- |
| Declared as | the only ACCEPT among 40 hypotheses in campaign `run-b8818e2a-1`; 172 option hypotheses across 3 campaigns all time |
| Residual alpha | **+2.52%/yr**, `t` +1.11, beta 0.09, IR 0.79 |
| Max drawdown | -2.05% |
| Verdict | **not refuted**, and `alpha_clears_bar` is `false` |

**This window can refute and cannot confirm, and unlike the equity side that is
not a caveat — it is the design.** Only 32 independent cycles settled inside
500 sessions, because a 14-DTE structure entered at most once a session still
overlaps itself, and `t +1.11` on 32 cycles is not evidence of an edge. What the
run establishes is the negative: the rule was given its one shot at two years it
had never seen, and those two years did not produce the significantly negative
alpha that would have killed it.

The rule is therefore executed on a **survived-refutation** basis, not a
confirmed one. specs/10 D8 says so, the option book carries `can_confirm: false`
in its own `sealed_measurement`, and the dashboard's Options tab prints it
beside the rule rather than under it.

## Campaign 07

Cited by `PLAN.md` as the campaign that produced the first sealed candidate. 40
proposals over 680 symbols on 1D bars: **1 ACCEPT, 3 REVIEW, 27 REJECT, 9
ERROR**. The ACCEPT was `rs_volatility_consistency_neutral_v1`
[`3f6e2c8a9309068b`] at 84/100 — run 1 in the table above, now retired.

The registry across all campaigns reads 13 ACCEPT, 14 PAPER, 48 REVIEW, 510
REJECT, 28 ERROR over 613 experiments. Failures are recorded on purpose:
forgetting the attempts that went nowhere is the mechanism by which a search
looks luckier than it was.

Those 613 experiments are **586 distinct hypotheses**, and that figure splits
before it is used: 414 on the equity side, 172 on the option side. specs/10 D8
requires the split — the two searches explore different spaces against different
sample sizes, and one denominator covering both would be far too strict for the
option side and far too loose for the equity side. The combined number is for
display only; every counting caller passes a family.
