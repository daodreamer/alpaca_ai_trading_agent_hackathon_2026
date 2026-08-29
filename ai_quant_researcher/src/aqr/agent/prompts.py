"""Prompt construction for the research agent.

Three things go into every proposal request, and the order matters for prompt
caching: the stable parts first, the volatile parts last.

1.  The system prompt and the feature catalogue -- identical on every call.
2.  Research memory -- what has already been tried, and how it went.
3.  The specific instruction for this iteration.

The catalogue is generated from the feature registry rather than written by
hand, so it cannot drift out of sync with what the DSL will actually accept. A
model that proposes a feature is proposing one that exists.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from aqr.features.registry import REGISTRY

__all__ = [
    "PROPOSAL_SCHEMA",
    "SYSTEM_PROMPT",
    "build_user_prompt",
    "feature_catalogue",
    "observability_note",
    "prompt_hash",
]

SYSTEM_PROMPT = """\
You are a quantitative researcher. Your job is to propose testable trading \
hypotheses, not to predict prices and not to place trades.

Every proposal you make will be compiled into a deterministic rule, backtested \
with realistic transaction costs, walk-forward validated on data you have never \
seen, and put through parameter-perturbation, cross-asset and Monte Carlo \
robustness tests. An overfitting detector will weigh how many backtests have \
already been run against how good your result looks. You cannot flatter your way \
past any of this, so do not try.

What makes a good proposal:

- It states a *reason* the edge should exist -- a behavioural, structural or \
risk-premium argument -- not just a combination of indicators that would have \
worked.
- It has few parameters. Every numeric literal you write is a degree of freedom \
the overfitting detector will charge you for.
- It fires often enough to be measurable (aim for well over 30 trades across the \
test period) but is not simply long-the-market in disguise.
- It is different from what has already been tried. Repeating a rejected \
hypothesis with one threshold nudged is not research.

On which form to use:

- Two are available. `signal` opens a position when a condition fires and \
closes it on a stop, a target or a bar count. `portfolio` ranks the universe \
every `rebalance_every` bars, holds the top `hold` names, and stays invested.
- **Prefer `portfolio` when your hypothesis is about which names to own rather \
than when to be in the market.** Anything phrased as "the strongest names \
outperform", "cheap names mean-revert", or "low-volatility names earn more per \
unit of risk" is a cross-sectional claim and belongs in a ranked book.
- This is a measured finding, not a style preference. A signal rule is out of \
the market much of the time, so it forfeits part of the drift it is measured \
against; every strategy this project promoted was one, and all fourteen lost \
to simply holding the same symbols. A ranked book keeps its market exposure \
and earns its difference on selection, which is what your hypothesis is about.
- In portfolio mode, `rank_by` must be a NUMBER, not a condition. \
`rs_rank(60) - rs_rank(5)` is a ranking; `rs_rank(60) > 0.7` is a filter that \
sorts the book into true and false. Put a filter in `screen` instead.
- You will be scored on alpha *after* the market exposure you ran is \
subtracted. Returning more than the index by holding more of it earns nothing.

On direction:

- Three are available: `long`, `short`, and `market_neutral`.
- A short is not a long with the sign flipped. Equities drift upward, losses are \
unbounded, and a borrow fee accrues every day the position stays open. So a \
short needs a sharper mechanism than "the mirror worked": name the *forced* or \
*price-insensitive* seller, or the trapped buyer whose stops supply the move.
- Check that your regime filter agrees with your direction. A long entry gated \
on `close < ema(200)` will never fire, and that is a wasted iteration.

**Prefer `market_neutral` when your hypothesis is about relative strength.** \
This is not a style preference, it is a measured finding. Fourteen long-only \
strategies from earlier campaigns reached PAPER and every one of them lost to \
simply holding the same symbols: on this universe over this window a long-only \
rule makes money almost whatever the rule is, so its Sharpe is mostly the \
market's and says almost nothing about the rule. A rule that is long and short \
at the same time has no such crutch, and its result means something.

A `market_neutral` proposal must supply `short_entry` as well as `entry`. Make \
them genuinely different conditions, not one negated: if `rs_rank(60) > 0.7` is \
your long leg, `rs_rank(60) < 0.3` is a defensible short leg only if you can say \
why weakness persists, which is a different claim from why strength persists. If \
you cannot, propose a one-directional strategy instead and say so.

What you must not do:

- Do not write Python, shell, SQL, or any code. You emit only the fields of the \
schema.
- Do not reference the future in any form. Every condition is evaluated at the \
close of a bar and filled at the next bar's open.
- Do not ask for data that is not in the feature list.
"""


# What one bar of each timeframe actually contains. The feature catalogue says
# what *parses*; this says what is *observable*, and the gap between the two is
# expensive. In a 40-hypothesis campaign on daily bars, 15 proposals compiled
# cleanly against real features and then fired on nothing -- opening drives,
# midday VWAP reversions, funding rates. Every one of them is a sensible idea
# about a market the data does not describe.
_OBSERVABILITY: dict[str, str] = {
    "1D": (
        "You are working with DAILY bars: one bar per session, carrying only "
        "open, high, low, close and volume. Within a session you cannot see "
        "anything -- not the opening drive, not a midday reversal, not where "
        "price sat relative to VWAP at 11am, not the order book, not the tape. "
        "`vwap` on daily bars is a session-anchored average of daily bars, not "
        "an intraday one. There is also no funding rate, no options flow, no "
        "short interest, no index breadth and no earnings calendar: if it is "
        "not in the feature list, this system genuinely cannot observe it. "
        "A rule that needs intraday structure will compile and then fire on "
        "nothing, which costs an iteration and teaches nobody anything."
    ),
    "1W": (
        "You are working with WEEKLY bars: one bar per week, open/high/low/"
        "close/volume only. Nothing inside the week is observable."
    ),
}

_INTRADAY_NOTE = (
    "You are working with {timeframe} bars, so intraday structure IS visible "
    "at that resolution -- but nothing finer than one {timeframe} bar is, and "
    "there is still no order book, no tape, no funding rate and no news feed. "
    "If it is not in the feature list, this system cannot observe it."
)


def observability_note(timeframe: str) -> str:
    """What a bar of this timeframe can and cannot show the model.

    Separate from the feature catalogue on purpose. The catalogue answers "will
    this parse"; this answers "is there anything there to see", and a model that
    knows only the first will keep proposing perfectly valid rules about
    structure the data does not contain.
    """
    note = _OBSERVABILITY.get(timeframe)
    return note if note else _INTRADAY_NOTE.format(timeframe=timeframe)


def feature_catalogue() -> str:
    """The complete DSL vocabulary, rendered for the prompt."""
    lines = ["Available features (this list is exhaustive -- nothing else parses):"]
    for name in sorted(REGISTRY):
        lines.append(f"  {REGISTRY[name].doc}")
    lines.append("")
    lines.append(
        "Expressions support + - * / , comparisons (< <= > >= == !=), and the "
        "keywords and / or / not. Feature arguments must be plain numbers. "
        "Example: close <= ema(20) * 1.01 and rsi(14) > 40 and rvol(20) > 1.2"
    )
    return "\n".join(lines)


PROPOSAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": "snake_case identifier, e.g. vol_contraction_breakout_v1",
        },
        "hypothesis": {
            "type": "string",
            "description": (
                "Two or three sentences: what inefficiency this exploits and why it "
                "should persist. State the mechanism, not the indicators."
            ),
        },
        "mode": {
            "type": "string",
            "enum": ["signal", "portfolio"],
            "description": (
                "Which form the hypothesis takes. 'signal' opens a position when a "
                "condition fires and closes it on a stop, a target or a bar count. "
                "'portfolio' ranks the whole universe every few bars and holds the "
                "top names, staying invested throughout."
            ),
        },
        "rank_by": {
            "type": "string",
            "description": (
                "Portfolio mode only. A NUMBER the universe is sorted by, best "
                "first -- for example 'rs_rank(60) - rs_rank(5)'. Not a condition: "
                "'close > ema(20)' sorts the book into true and false. Empty string "
                "in signal mode."
            ),
        },
        "screen": {
            "type": "string",
            "description": (
                "Portfolio mode only. Optional condition a name must satisfy to be "
                "eligible for the ranking. Empty string for none."
            ),
        },
        "hold": {
            "type": "integer",
            "description": "Portfolio mode only. How many names the book holds.",
        },
        "rebalance_every": {
            "type": "integer",
            "description": "Portfolio mode only. Bars between rebalances.",
        },
        "direction": {
            "type": "string",
            "enum": ["long", "short", "market_neutral"],
        },
        "short_entry": {
            "type": "string",
            "description": (
                "The short leg. Required when direction is market_neutral, and must "
                "be empty otherwise. A separate condition, not the entry negated."
            ),
        },
        "regime": {
            "type": "string",
            "description": (
                "Optional market-state filter gating all entries, e.g. "
                "'close > ema(200)'. Empty string for no filter."
            ),
        },
        "entry": {
            "type": "string",
            "description": "The entry condition. Must be a comparison or a combination of them.",
        },
        "signal_exit": {
            "type": "string",
            "description": "Optional condition that closes the position. Empty string for none.",
        },
        "stop_loss_atr_multiple": {
            "type": "number",
            "description": "Stop distance in ATR(14) multiples. Typically 1.0 to 4.0.",
        },
        "take_profit_r_multiple": {
            "type": "number",
            "description": "Target as a multiple of the stop distance. 0 means no target.",
        },
        "max_holding_bars": {
            "type": "integer",
            "description": "Hard time stop in bars.",
        },
        "expected_trades_per_year": {
            "type": "integer",
            "description": "Your own estimate. Used to check your intuition against reality.",
        },
    },
    "required": [
        "name",
        "hypothesis",
        "mode",
        "direction",
        "regime",
        "short_entry",
        "signal_exit",
        "stop_loss_atr_multiple",
        "take_profit_r_multiple",
        "max_holding_bars",
        "expected_trades_per_year",
    ],
    "additionalProperties": False,
}


def _render_memory(memory: list[dict[str, Any]]) -> str:
    """Past experiments, most recent first.

    Failures are included deliberately, and rendered with their reason. A list of
    successes only would push the model toward variations on what already worked
    -- which is precisely how a research loop overfits its own history.
    """
    if not memory:
        return "No experiments have been run yet. This is the first hypothesis."
    lines = ["Experiments already run (most recent first). Do not repeat these:"]
    for item in memory:
        verdict = item.get("verdict") or "UNKNOWN"
        sharpe = item.get("oos_sharpe")
        trades = item.get("oos_trades")
        detail = []
        if sharpe is not None:
            detail.append(f"OOS Sharpe {sharpe:.2f}")
        if trades is not None:
            detail.append(f"{trades} trades")
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


def build_user_prompt(
    *,
    symbols: list[str],
    timeframe: str,
    memory: list[dict[str, Any]],
    instruction: str | None = None,
    parent: dict[str, Any] | None = None,
) -> str:
    """The volatile half of the prompt: memory, universe, and this turn's ask."""
    parts = [
        feature_catalogue(),
        "",
        observability_note(timeframe),
        "",
        f"Universe: {', '.join(symbols)} on {timeframe} bars.",
        "",
        _render_memory(memory),
        "",
    ]
    if parent:
        parts += [
            "You are improving on this strategy. Change one thing and say why:",
            json.dumps(parent, indent=2, sort_keys=True),
            "",
        ]
    parts.append(
        instruction
        or (
            "Propose one new hypothesis. Prefer a mechanism that is different in kind "
            "from what has already been tried, not a re-parameterisation of it."
        )
    )
    return "\n".join(parts)


def prompt_hash(system: str, user: str) -> str:
    """Content hash of the exact prompt, recorded with the experiment.

    Without it, a result cannot be reproduced: the same model and the same data
    give a different answer when the prompt has quietly changed.
    """
    digest = hashlib.sha256()
    digest.update(system.encode())
    digest.update(b"\x00")
    digest.update(user.encode())
    return digest.hexdigest()[:16]
