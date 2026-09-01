"""Prompt construction for the *option* research agent — specs/10 D4, D5, D6.

A sibling of [`prompts.py`](prompts.py), deliberately not an extension of it.
The two vocabularies are unmixed on purpose (CLAUDE.md §2b): an equity prompt
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

from aqr.features.registry import REGISTRY
from aqr.options.features import OPTION_FEATURES

__all__ = [
    "ALLOWED_DTE_TARGETS",
    "OPTION_PROPOSAL_SCHEMA",
    "OPTION_SYSTEM_PROMPT",
    "STRUCTURE_CATALOGUE",
    "build_option_user_prompt",
    "option_feature_catalogue",
    "structure_catalogue",
]

ALLOWED_DTE_TARGETS = (14, 28, 49)
"""D0: the vendor carries three rolling expiry targets and nothing else. Not a
policy — a fact about the cache, so a proposal outside it cannot be priced."""

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
leave-one-year-out and DTE-bucket robustness tests. The verdict gates on \
INDEPENDENT CYCLES, not on trade count: a structure held to expiry produces \
new evidence only when it closes, so thirty overlapping spreads can be eight \
independent bets. The whole research window holds about 71 non-overlapping \
28-day cycles across five and a half years. You cannot flatter your way past \
any of this.

Four things about this data decide what you may propose. They are measured \
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

3. THERE ARE THREE EXPIRIES, at rolling targets near 14, 28 and 49 days. No \
0DTE, no weeklies, no LEAPS, and no fixed calendar dates -- the expiries move \
with the observation date.

4. THERE IS ONE UNDERLYING: SPY. There is no cross-section to rank, no \
relative-strength claim to make, and no portfolio mode.

5. THE VOLATILITY FEATURES ARE DECIMAL FRACTIONS, NOT PERCENTAGE POINTS. \
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
- It fires often enough to produce independent cycles. Your entry condition and \
your cadence together decide the sample size, and a condition that is true on \
18% of sessions at a 28-day holding period leaves roughly twelve independent \
bets in five years. That is not evidence, whatever it measures.
- It has few parameters. Every numeric literal in your entry expression, plus \
the DTE target, the anchor delta, the width and the cadence, is a degree of \
freedom the overfitting detector charges you for.
- It is different in kind from what has already been tried. Nudging a threshold \
on a rejected hypothesis is not research.

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
    lines.append(
        "An option feature is NaN when the vendor row is missing or more than 5 "
        "calendar days stale, and NaN compares false in every direction -- so a "
        "condition on iv_rank() simply does not fire on those sessions rather "
        "than firing on a stale number."
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
                "schedule rather than a hypothesis."
            ),
        },
        "structure_type": {
            "type": "string",
            "enum": [kind for kind, _ in STRUCTURE_CATALOGUE],
            "description": "Which defined-risk structure. Direction lives here.",
        },
        "dte_target": {
            "type": "integer",
            "enum": list(ALLOWED_DTE_TARGETS),
            "description": (
                "Days to expiry, at one of the three rolling targets the cache "
                "carries. Nothing else can be priced."
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
                "Cadence: how many chain sessions must pass between entries. This "
                "is your only control over how correlated your trades are, and the "
                "verdict counts independent cycles. Below the DTE target, entries "
                "overlap."
            ),
        },
        "expected_cycles_per_year": {
            "type": "integer",
            "description": (
                "Your own estimate of independent (non-overlapping) cycles per "
                "year. Used to check your intuition against what actually happened."
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
        if item.get("oos_cycles") is not None:
            detail.append(f"{item['oos_cycles']} independent cycles")
        if item.get("overfitting"):
            detail.append(f"overfitting {item['overfitting']}")
        if item.get("error"):
            detail.append(f"failed: {item['error']}")
        suffix = f" ({', '.join(detail)})" if detail else ""
        lines.append(f"  [{verdict}] {item.get('name')}{suffix}")
        hypothesis = (item.get("hypothesis") or "").strip()
        if hypothesis:
            lines.append(f"      claim: {hypothesis}")
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
            f"Search budget: this is hypothesis {spent + 1} of {cap}. The cap is "
            "small because the window holds about 71 independent cycles, and a "
            "search wide enough to overwhelm that denominator produces a number "
            "with no information in it. Spend the remaining attempts on "
            "mechanisms that differ from each other, not on thresholds.",
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
