"""Prompt construction for the *option* research agent — specs/10 D4, D5, D6.

A sibling of [`prompts.py`](prompts.py), deliberately not an extension of it.
The two vocabularies are unmixed on purpose (CLAUDE.md §2): an equity prompt
enumerates ``REGISTRY`` and talks about stops, targets and holding periods, and
every one of those words is a lie on a structure held to expiry. A model handed
the equity prompt with three fields renamed would propose ``manage at 50%`` and
this system would have to explain, downstream, why the number it produced is
about a strategy nobody can trade.

So the whole prompt is rebuilt around what specs/10 measured the data can
actually answer, and the four facts a model will otherwise get wrong are stated
in the system prompt rather than left to be inferred from the schema:

1.  **There is no exit.** D1: a specific contract is re-quoted on 1-3% of later
    sessions, so no stop, target, roll or "manage at 50%" can be priced. The
    schema has no field for one, and the prompt says why.
2.  **The width is named by delta, not by points.** D5: a delta-selected wing
    resolves on 98% of sessions and an exact 10-point wing on 23%, because the
    cache samples about 24 rungs from a ladder that lists hundreds.
3.  **Three expiries exist**, at rolling ~14 / ~28 / ~49 DTE. No 0DTE, no
    weeklies, no LEAPS.
4.  **One underlying.** There is no cross-section to rank, so there is no
    portfolio mode and no ``rank_by``.

The feature catalogue is generated from :data:`~aqr.options.features.OPTION_FEATURES`
and the bar :data:`~aqr.features.registry.REGISTRY` together, because that union
is exactly what ``OptionSpec``'s entry expression parses against
(``options/features.py``'s ``resolve_entry_feature``). Hand-writing it would let
it drift, and a drifted catalogue is a model proposing features that do not
exist.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from aqr.evaluator.score import MIN_INDEPENDENT_CYCLES
from aqr.features.registry import REGISTRY
from aqr.options.features import OPTION_FEATURES

__all__ = [
    "ALLOWED_DTE_TARGETS",
    "CYCLE_BUDGET",
    "OPTION_PROPOSAL_SCHEMA",
    "OPTION_SYSTEM_PROMPT",
    "SESSIONS_PER_YEAR",
    "STRUCTURE_CATALOGUE",
    "build_option_user_prompt",
    "cycle_budget_table",
    "option_feature_catalogue",
    "structure_catalogue",
]

DTE_BAND = (11, 66)
"""Every days-to-expiry the cache actually lists, measured rather than assumed.

specs/10 D0 describes the vendor as carrying "three rolling targets at ~14 / ~28
/ ~49", and the first version of this schema turned that into an enum of exactly
those three. That was over-restrictive and it narrowed the search for no reason:
the sampled expiries run 11 to 66 days out across 39 distinct values, and with
the engine's ±10-day tolerance a target anywhere in the band resolves. Measured
on the 753 research sessions:

    target 11/14/21/28   753 of 753 sessions resolve
    target 35            751
    target 49            582
    target 56            711
    target 42 / 63 / 66  334 / 419 / 326

So the band is open and the coverage is published, which is the honest
arrangement: a model choosing 42 is choosing a target that resolves on 44% of
sessions, and it should be able to make that trade deliberately rather than
being refused for choosing a number that is not one of three."""

DTE_COVERAGE: tuple[tuple[int, int], ...] = (
    (11, 753),
    (14, 753),
    (21, 753),
    (28, 753),
    (35, 751),
    (42, 334),
    (49, 582),
    (56, 711),
    (63, 419),
    (66, 326),
)
"""How many of the 753 research sessions resolve an expiry within 10 days of a
target. Shown to the model so a thin bucket is a choice rather than a surprise;
asserted against the cache by ``tests/test_option_cache_claims.py``."""

ALLOWED_DTE_TARGETS = tuple(range(DTE_BAND[0], DTE_BAND[1] + 1))
"""Kept as a name because callers validate against it; now the whole band."""

SESSIONS_PER_YEAR = 136
"""Chain sessions per year, not trading days per year.

The cache holds 753 sessions across 5.56 years and the median gap between two
of them is **two calendar days** (448 gaps of 2, 226 of 3, 45 of 7 -- 2019 is
one Saturday snapshot a week, 2020 onward is roughly Mon/Wed/Fri). So a session
is not a day, and ``min_sessions_between_entries: 5`` is about ten calendar days
apart rather than a week. The model was never told this and could not have
inferred it from "one snapshot per session", which reads as daily."""

CYCLE_BUDGET: tuple[tuple[int, int, int, int], ...] = (
    (11, 144, 91, 7),
    (14, 144, 91, 7),
    (21, 126, 89, 7),
    (28, 71, 46, 10),
    (35, 71, 46, 10),
    (42, 33, 21, 0),
    (49, 33, 21, 0),
    (56, 33, 21, 0),
    (63, 32, 20, 0),
)
"""``(dte_target, cycles in the window, cycles out of sample, min firing rate %)``.

The first three columns are the ceiling, not a forecast: what a rule with **no
entry condition at all** gets, entering on every session available, holding to
expiry, counted by the same non-overlapping walk the verdict uses
(:func:`~aqr.validation.cycles.independent_cycles`). A real rule fires on some
fraction of sessions and lands below its row.

The fourth column is the number a proposer actually needs and is the reason this
table has four columns rather than three: the smallest share of sessions an entry
condition may fire on and still reach 25 OOS cycles, measured by resampling the
real session grid (median of 25 draws). ``0`` means unreachable at any
selectivity.

Two facts fall out, and they are different facts:

* **42 DTE and longer cannot clear the gate at all.** 21 cycles with the
  condition removed entirely is below 25, so every hypothesis in that bucket is
  a REJECT decided before the backtest runs. The 100-hypothesis campaign spent
  eight slots there.
* Between 14 and 28 DTE the *threshold* barely moves -- 7% against 10% -- because
  at that selectivity the binding constraint is how often the rule fires, not
  how much its holdings overlap. What doubles is the *headroom* (91 against 46),
  and headroom past the gate is what buys statistical power rather than a bare
  pass. Prefer the short expiry, but do not tell a model the short expiry is the
  difference between passing and failing at 10%: it is not.

The band stays open (the previous narrowing was wrong for its own reasons), but
an open band with a published ceiling is a choice; an open band without one is a
trap. Asserted against the cache by ``tests/test_option_cache_claims.py``."""

STRUCTURE_CATALOGUE: tuple[tuple[str, str], ...] = (
    (
        "put_credit_spread",
        "sell the anchor put, buy a further-out put. Credit; risk = width - credit.",
    ),
    (
        "call_credit_spread",
        "sell the anchor call, buy a further-out call. Credit; risk = width - credit.",
    ),
    (
        "put_debit_spread",
        "buy the anchor put, sell a further-out put. Debit; risk = premium paid.",
    ),
    (
        "call_debit_spread",
        "buy the anchor call, sell a further-out call. Debit; risk = premium paid.",
    ),
    (
        "iron_condor",
        "a put credit spread and a call credit spread together. Credit; "
        "risk = wider wing - credit.",
    ),
    ("long_put", "buy one put. Debit; risk = premium paid. No width."),
    ("long_call", "buy one call. Debit; risk = premium paid. No width."),
)
"""D4's whitelist, and nothing else.

There is no ``custom``, no lone short leg and no kind whose maximum loss is
unbounded. ``covered_call`` and ``cash_secured_put`` are absent because each
needs a stock position or a cash reserve to be defined-risk, which makes it a
portfolio statement rather than a structure.
"""


OPTION_SYSTEM_PROMPT = """\
You are a quantitative researcher proposing testable hypotheses about SPY \
option structures. You do not predict prices and you do not place trades.

Every proposal is compiled into a deterministic rule and backtested on \
end-of-day option chains with the full quoted spread charged on every leg, \
walk-forward validated on calendar folds you have never seen, and put through \
leave-one-year-out and DTE-bucket robustness tests. You cannot flatter your way \
past any of this.

THE GATE YOU MUST DESIGN FOR, FIRST, BEFORE ANYTHING ELSE: the out-of-sample \
period must contain AT LEAST 25 INDEPENDENT CYCLES. Not 25 trades -- 25 \
non-overlapping ones. A structure held to expiry produces new evidence only \
when it closes, so entries opened while an earlier one is still running are \
correlated draws on the same outcome and are not counted again: thirty \
overlapping spreads can be eight independent bets. Below 25 the verdict is \
REJECT and nothing else about your hypothesis is even read.

Two knobs decide whether you can reach 25. `dte_target` sets the CEILING, \
because it fixes how long each cycle occupies the calendar; your entry condition \
then takes some fraction of what the ceiling left. The cycle budget table in the \
catalogue below prints both -- the ceiling for each target, and the smallest \
share of sessions your condition may fire on and still reach 25. Read your \
target's row before you write anything else. A target of 42 or more has a \
ceiling of 21, which is under the gate with no condition at all: those \
hypotheses are rejected before the backtest runs, and eight of the last hundred \
were spent there.

The previous campaign of 100 hypotheses produced zero passes and its best rule \
reached 4 cycles against the 25 it needed. It did not fail on its ideas. It \
failed on arithmetic nobody showed it: a three-clause `and` fires on roughly 4% \
of sessions, and 4% is under the floor at every expiry this cache lists.

Six things about this data decide what you may propose. They are measured \
facts, not preferences:

1. THERE IS NO EXIT. A structure is opened and held to expiry, and settles \
against the underlying's close on the expiration date. A specific contract is \
re-quoted on only 1-3% of later sessions, so the data cannot price a stop, a \
profit target, a signal exit, a roll, or "manage at 50% of max profit". There \
is no field for any of them. Do not describe one in your hypothesis either: a \
verdict from this system is a verdict on the UNMANAGED version of your rule.

2. THE WIDTH IS NAMED BY DELTA, NOT BY POINTS. The cache samples about 24 \
strikes from a ladder that lists hundreds, so the listed distance below a \
16-delta put is 8, 9, 10, 18, 25, 35 or 45 points depending on the session. A \
delta-selected wing resolves on 98% of sessions; an exact 10-point wing on \
23%. Name `width_delta` (the protective leg's own delta), which must be \
SMALLER than `anchor_delta` because the protective leg is further out of the \
money.

3. EXPIRIES RUN 11 TO 66 DAYS OUT and no further, and THE SHORTEST ONE LISTED \
IS 14 DAYS. No 0DTE, no weeklies, no LEAPS, no fixed calendar dates -- the \
expiries roll with the observation date. On 632 of the 753 sessions the nearest \
expiry is exactly 14 days out; on the rest it is 11, 13, 15 or 16. So the \
shortest position this data can express is a two-week one, and "hold for a few \
days" does not exist here at any setting. Any target in the band is allowed and \
the engine takes the nearest listed expiry within 10 days, but the choice costs \
you sample size and the cycle budget table below prices it: 14 DTE buys 91 \
possible OOS cycles, 28 buys 46, and 49 buys 21 against a gate of 25.

4. A SESSION IS NOT A DAY. The cache holds 753 sessions across 5.56 years -- \
about 136 a year, against roughly 252 trading days. The median gap between two \
sessions is TWO calendar days (2019 is one snapshot a week; 2020 onward is \
roughly Monday/Wednesday/Friday). `min_sessions_between_entries` counts these \
sessions, so 5 is about ten calendar days apart, not a week. Entering on every \
session available -- cadence 1 -- is roughly an entry every other day, and that \
is the fastest this data permits anything to happen.

5. THERE IS ONE UNDERLYING: SPY. There is no cross-section to rank, no \
relative-strength claim to make, and no portfolio mode.

6. THE VOLATILITY FEATURES ARE DECIMAL FRACTIONS, NOT PERCENTAGE POINTS. \
`iv_current()` of 0.156 means 15.6% vol. `term_slope()` never exceeds 0.052, so \
`term_slope() > 5` is false on every session that has ever existed and the rule \
opens nothing. `iv_hv_spread()` runs -0.38 to 0.17; a five-point variance \
premium is 0.05. `iv_rank()` is the ONE feature on a 0..100 scale, and its being \
the exception is exactly what makes this easy to get wrong. Every feature in the \
catalogue below states its units and its measured range -- read them before you \
write a number. A threshold outside a feature's range is not a strict \
hypothesis, it is a rule that cannot fire, and it costs a slot of the search \
budget for no information at all.

What makes a good proposal:

- It states a REASON the premium should exist -- a risk-transfer, structural or \
behavioural argument. "Short volatility earns the variance risk premium" is a \
real mechanism; "credit spreads win 84% of the time" is a description of the \
payoff, not a reason to be paid for it.
- It fires often enough to produce independent cycles. Do this arithmetic \
BEFORE you commit to a condition. Measured on this cache at 14 DTE with cadence \
1, entry conditions of varying selectivity produce, out of sample:

      fires on 100% of sessions -> 91 OOS cycles      fires on 15% -> 43
      fires on  50% of sessions -> 76 OOS cycles      fires on 10% -> 33
      fires on  30% of sessions -> 62 OOS cycles      fires on  5% -> 19
      fires on  20% of sessions -> 50 OOS cycles

  At 14 DTE a condition true on one session in ten still clears the gate of 25; \
the measured floor is 7%. At 28 DTE the floor is 10% -- close, because at that \
selectivity what limits you is how often the rule fires, not how much its \
holdings overlap. What the short expiry really buys is HEADROOM: 91 possible \
cycles against 46, and cycles past the gate are what turn a bare pass into a \
measurable one. At 42 DTE and beyond there is no floor, because the ceiling \
itself is under the gate.

  A three-clause `and` on features that are each true a third of the time fires \
on about 4% of sessions. That is below every floor in the table, and it is the \
shape that produced 4 cycles and a REJECT one hundred times in a row. Two \
clauses, or one clause and a well-chosen structure, is what a passing rule \
looks like.
- It has few parameters. Every numeric literal in your entry expression, plus \
the DTE target, the anchor delta, the width and the cadence, is a degree of \
freedom the overfitting detector charges you for.
- It is different in kind from what has already been tried. Nudging a threshold \
on a rejected hypothesis is not research.
- VARY THE STRUCTURE, NOT ONLY THE CONDITION. This is the failure mode of every \
campaign so far: fifteen hypotheses that all wrote `dte_target: 28`, \
`anchor_delta: 0.16` and a three-clause `and` in the entry, differing only in \
which features the clauses named. The entry condition is one of five degrees of \
freedom you have. The structure, the expiry, how far out of the money the anchor \
sits, how wide the wing is and how often you are allowed to enter are the other \
four, and they change the payoff far more than a threshold does. A 0.30-delta \
spread at 14 days and a 0.08-delta spread at 56 days are different strategies; \
two entry conditions on a 0.16-delta 28-day spread are usually the same strategy \
asked twice. The memory below lists the knobs every past attempt used -- look at \
what has been pinned and move it.

Direction is expressed by the STRUCTURE, not by a field. A put credit spread is \
a bullish, short-volatility position; a call credit spread is bearish and short \
volatility; a long put is bearish and long volatility; an iron condor is \
directionally neutral and short volatility. Choose the structure your mechanism \
implies, and say in the hypothesis which of those you meant.

What you must not do:

- Do not write Python, shell, SQL or any code. You emit only the schema fields.
- Do not reference the future. Your entry condition is evaluated at the close of \
session t-1 and filled from session t's chain.
- Do not ask for data that is not in the feature list. There is no order flow, \
no open interest surface, no dealer gamma, no VIX term structure beyond the \
`term_slope()` this cache can compute, and no earnings calendar.
- Do not name a strike, a price, a number of contracts or a dollar amount. \
Sizing is the run's configuration, and strikes are resolved from the live \
ladder by delta.
"""


def structure_catalogue() -> str:
    """D4's whitelist, rendered for the prompt."""
    lines = ["Structures (this list is exhaustive -- nothing else constructs):"]
    lines += [f"  {kind:<20} {note}" for kind, note in STRUCTURE_CATALOGUE]
    lines.append("")
    lines.append(
        "long_call and long_put take no width. Every other kind requires "
        "width_delta. iron_condor may additionally set call_anchor_delta and "
        "call_width_delta; leaving them empty mirrors the put side."
    )
    return "\n".join(lines)


def option_feature_catalogue(
    spans: Mapping[str, tuple[float, float]] | None = None,
) -> str:
    """Every feature an option entry expression can name.

    Both halves of it: specs/10 D6's option table *and* the unchanged bar
    registry, because ``resolve_entry_feature`` resolves from the union and a
    rule may say ``iv_rank() > 50 and close > sma(200)`` in one expression.
    Generated rather than written down, so the prompt cannot describe a
    vocabulary the parser does not have.
    """
    lines = ["Option features (from the volatility history and the chain):"]
    for name in sorted(OPTION_FEATURES):
        lines.append(f"  {OPTION_FEATURES[name].doc}")
        # Measured on this run's own market rather than taken from the doc
        # string, when a caller has one. The docs carry a range too, and a
        # re-pull can move it; the number a model reads should be the number the
        # engine will use.
        rendered = (spans or {}).get(name)
        if rendered is not None:
            lines.append(
                f"      measured this run: {rendered[0]:.4g} .. {rendered[1]:.4g}"
            )
    lines.append("")
    lines.append(
        "Underlying-bar features (SPY daily bars, the same registry the equity side uses):"
    )
    for name in sorted(REGISTRY):
        lines.append(f"  {REGISTRY[name].doc}")
    lines.append("")
    lines.append(
        "Expressions support + - * / , comparisons (< <= > >= == !=), and the "
        "keywords and / or / not. Feature arguments must be plain numbers. "
        "Example: iv_rank() > 40 and close > sma(200)"
    )
    lines.append("")
    lines.append(
        "Expiry coverage -- of 753 research sessions, how many resolve an expiry "
        "within 10 days of each target:"
    )
    lines.append(
        "  " + "  ".join(f"{target}d:{count}" for target, count in DTE_COVERAGE)
    )
    lines.append(
        "An option feature is NaN when the vendor row is missing or more than 5 "
        "calendar days stale, and NaN compares false in every direction -- so a "
        "condition on iv_rank() simply does not fire on those sessions rather "
        "than firing on a stale number."
    )
    lines.append("")
    lines.append(cycle_budget_table())
    return "\n".join(lines)


def cycle_budget_table() -> str:
    """:data:`CYCLE_BUDGET`, rendered with the gate next to it.

    Printed in the user prompt rather than only described in the system prompt
    because it is the one table the model has to arithmetic against, and a
    number it has to recall from four paragraphs earlier is a number it will
    approximate.
    """
    lines = [
        f"Cycle budget. The gate is {MIN_INDEPENDENT_CYCLES} out-of-sample "
        "independent cycles. The first two columns are the CEILING -- what a rule",
        "with no entry condition at all would get. The last column is the one to "
        "design against.",
        "",
        "  dte    max cycles   max OOS cycles   your entry may fire on as little as",
    ]
    for target, window, oos, floor in CYCLE_BUDGET:
        room = (
            "NOTHING -- under the gate with no condition at all, always REJECT"
            if floor == 0
            else f"{floor}% of sessions"
        )
        lines.append(f"  {target:<6} {window:<12} {oos:<16} {room}")
    lines.append("")
    lines.append(
        "So: 42 DTE and longer is an automatic REJECT whatever you write. At 35 and "
        "shorter, a condition firing on under 10% of sessions is the failure mode "
        "that rejected 100 hypotheses in a row -- a three-clause `and` usually "
        "lands near 4%. Two clauses, or one, is the shape that survives."
    )
    return "\n".join(lines)


OPTION_PROPOSAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": "snake_case identifier, e.g. iv_rank_put_credit_spread_v1",
        },
        "hypothesis": {
            "type": "string",
            "description": (
                "Two or three sentences: which premium this harvests or which "
                "mispricing it exploits, and why it should persist. State the "
                "mechanism, not the payoff diagram."
            ),
        },
        "entry": {
            "type": "string",
            "description": (
                "The entry condition, evaluated at the close of the previous "
                "session. Must be a comparison or a combination of them, e.g. "
                "'iv_rank() > 40 and close > sma(200)'. Required: a rule with no "
                "condition opens a position every session it can, which is a "
                "schedule rather than a hypothesis. It must also FIRE OFTEN "
                "ENOUGH: see the cycle budget table for the minimum share of "
                "sessions your dte_target leaves you (7% at 14 DTE, 10% at 28). "
                "Each `and` clause multiplies the share down, so three of them on "
                "features that are each true a third of the time lands near 4% "
                "and is rejected on sample size before anything else is read."
            ),
        },
        "structure_type": {
            "type": "string",
            "enum": [kind for kind, _ in STRUCTURE_CATALOGUE],
            "description": "Which defined-risk structure. Direction lives here.",
        },
        "dte_target": {
            "type": "integer",
            "minimum": DTE_BAND[0],
            "maximum": DTE_BAND[1],
            "description": (
                "Days to expiry, anywhere in 11..66 -- the whole band the cache "
                "lists. The engine takes the nearest listed expiry within 10 days. "
                "This is the field that sets your sample-size ceiling, so choose "
                "it before the entry condition: 14 leaves room for 91 "
                "out-of-sample cycles, 28 leaves 46, and 42 or longer leaves 21 "
                "against a gate of 25 -- an automatic REJECT whatever the rule "
                "says. The shortest expiry the cache lists is 14 days; there is no "
                "way to hold for less than about two weeks."
            ),
        },
        "anchor_delta": {
            "type": "number",
            "description": (
                "The magnitude of the delta of the leg the rule names, in (0, 1). "
                "0.16 is the usual short-premium choice; 0.30 is closer to the "
                "money and 0.05 is far out of it. Whether it is bought or sold "
                "follows from structure_type, not from you."
            ),
        },
        "width_delta": {
            "type": "number",
            "description": (
                "The protective leg, by its OWN delta. Must be strictly less than "
                "anchor_delta. 0 for long_call and long_put, which have one leg."
            ),
        },
        "call_anchor_delta": {
            "type": "number",
            "description": (
                "iron_condor only: the call side's anchor delta. 0 mirrors the put "
                "side. Must be 0 for every other structure."
            ),
        },
        "call_width_delta": {
            "type": "number",
            "description": (
                "iron_condor only: the call side's wing delta. 0 mirrors the put "
                "side. Must be 0 for every other structure."
            ),
        },
        "min_sessions_between_entries": {
            "type": "integer",
            "description": (
                "Cadence: how many CHAIN SESSIONS must pass between entries -- and "
                "a session is about two calendar days, not one, so 5 means about "
                "ten days apart. Raising it cannot raise your independent-cycle "
                "count, because overlapping entries were never counted twice in "
                "the first place; it can only lower the number of trades. Leave it "
                "at 1 unless the mechanism genuinely needs spacing."
            ),
        },
        "expected_cycles_per_year": {
            "type": "integer",
            "description": (
                "Your own estimate of independent (non-overlapping) cycles per "
                "year. Compute it, do not guess it: take your dte_target's row in "
                "the cycle budget table, multiply the ceiling by the share of "
                "sessions you think your entry fires on, and divide by 5.6 years. "
                "If the answer is under 5 per year your hypothesis cannot clear "
                "the gate and you should change the expiry or the condition now, "
                "not propose it and find out."
            ),
        },
    },
    "required": [
        "name",
        "hypothesis",
        "entry",
        "structure_type",
        "dte_target",
        "anchor_delta",
        "width_delta",
        "call_anchor_delta",
        "call_width_delta",
        "min_sessions_between_entries",
        "expected_cycles_per_year",
    ],
    "additionalProperties": False,
}


def _render_memory(memory: list[dict[str, Any]]) -> str:
    """Past option experiments, most recent first.

    Failures included and rendered with their reason, for the same reason the
    equity prompt does it: a list of successes pushes the model toward variations
    on what already worked, which is how a research loop overfits its own
    history. The one addition here is the cycle count — an option result that
    reported its trade count alone would invite the model to propose something
    that trades more, when the binding constraint is how many of those trades
    were independent.
    """
    if not memory:
        return "No option experiments have been run yet. This is the first hypothesis."
    lines = ["Option experiments already run (most recent first). Do not repeat these:"]
    for item in memory:
        verdict = item.get("verdict") or "UNKNOWN"
        detail: list[str] = []
        sharpe = item.get("oos_sharpe")
        if sharpe is not None:
            detail.append(f"OOS Sharpe {sharpe:.2f}")
        if item.get("oos_trades") is not None:
            detail.append(f"{item['oos_trades']} trades")
        cycles = item.get("oos_cycles")
        if cycles is not None:
            # Against the gate, not on its own. "4 independent cycles" is a fact
            # the model cannot act on; "4/25 cycles -- REJECTED on sample size"
            # names the constraint that actually decided the verdict, which is
            # the whole reason the failures are shown at all.
            verdict_note = (
                " -- REJECTED on sample size, the rule was never measured"
                if cycles < MIN_INDEPENDENT_CYCLES
                else " -- cleared the sample-size gate"
            )
            detail.append(f"{cycles}/{MIN_INDEPENDENT_CYCLES} cycles{verdict_note}")
        if item.get("overfitting"):
            detail.append(f"overfitting {item['overfitting']}")
        if item.get("error"):
            detail.append(f"failed: {item['error']}")
        suffix = f" ({', '.join(detail)})" if detail else ""
        lines.append(f"  [{verdict}] {item.get('name')}{suffix}")
        # The knobs, not just the claim. Fifteen hypotheses in the first
        # campaign wrote dte 28 / delta 0.16 and varied only the entry
        # condition; a memory that showed the claim alone gave the model no way
        # to notice it was exploring one corner of a five-dimensional space.
        knobs = (item.get("structure") or "").strip()
        if knobs:
            lines.append(f"      structure: {knobs}")
        hypothesis = (item.get("hypothesis") or "").strip()
        if hypothesis:
            lines.append(f"      claim: {hypothesis}")
        entry = (item.get("entry") or "").strip()
        if entry:
            lines.append(f"      entry: {entry}")
    return "\n".join(lines)


def build_option_user_prompt(
    *,
    underlying: str,
    memory: list[dict[str, Any]],
    instruction: str | None = None,
    parent: dict[str, Any] | None = None,
    budget: tuple[int, int] | None = None,
    spans: Mapping[str, tuple[float, float]] | None = None,
) -> str:
    """The volatile half of the prompt: catalogue, memory, and this turn's ask.

    ``budget`` is ``(spent, cap)`` and is shown to the model on purpose: a model
    that does not know how much of the search is left will keep proposing small
    variations. Telling it how many attempts remain is the cheapest way to buy
    back some variety in the ones that are left.

    ``spans`` is the measured range of each option feature on this run's own
    market. Optional, and supplying it is strongly recommended: a campaign
    without it lost seven of twenty slots to thresholds no session could satisfy
    (``iv_hv_spread() > 5`` against a maximum of 0.17), which is the failure
    :func:`~aqr.agent.option_proposer.unreachable_thresholds` now catches after
    the fact and this prevents before it.
    """
    parts = [
        structure_catalogue(),
        "",
        option_feature_catalogue(spans),
        "",
        f"Underlying: {underlying}. End-of-day chains only -- one snapshot per "
        "session, no intraday, about 24 strikes per expiry.",
        "",
        _render_memory(memory),
        "",
    ]
    if budget is not None:
        spent, cap = budget
        parts += [
            f"Search budget: this is hypothesis {spent + 1} of {cap}, and every "
            "one of them is charged against the deflated Sharpe -- a wide search "
            "over a window this short produces a best-of-N number with no "
            "information in it, and the overfitting report says so out loud. Do "
            "not treat the cap as an allowance to spend. Spend attempts on "
            "mechanisms that differ from each other, not on thresholds, and not "
            "on rules the cycle budget already says cannot pass.",
            "",
        ]
    if parent:
        parts += [
            "You are improving on this rule. Change one thing and say why:",
            json.dumps(parent, indent=2, sort_keys=True),
            "",
        ]
    parts.append(
        instruction
        or (
            "Propose one new hypothesis. Prefer a mechanism that is different in "
            "kind from what has already been tried, not a re-parameterisation of it."
        )
    )
    return "\n".join(parts)
