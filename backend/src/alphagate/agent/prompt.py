"""The prompt, versioned — specs/05 D3 and D7.

`PROMPT_VERSION` goes into every journal record. **A prompt edit is a version
bump**, without exception: a rationale from v1 and a rationale from v2 are not
comparable, and a journal that cannot tell them apart cannot support the claim
that the day replays.

What the model is shown is engine output, not prices to interpret (specs/05 D2),
and what it may return is an index or null (D3). The schema below is the whole
channel. Note what it does not contain: no symbol, no strike, no quantity, no
price. A model cannot hallucinate a contract into this system because there is
no field in which to write one.

The instruction text is deliberately blunt about declining. Models are agreeable,
and an agreeable model asked to pick from a list will pick from the list. Saying
"null is a good answer" twice, and giving it a named reason to use, is what makes
`Stage.DECLINED` a real outcome rather than a theoretical one.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, Final

from alphagate.agent.model import Candidate, MarketRead

__all__ = [
    "PROMPT_VERSION",
    "RESPONSE_SCHEMA",
    "SYSTEM_PROMPT",
    "build_user_message",
]

PROMPT_VERSION: Final = "v5"
"""v2 renamed the volatility fields to say what they are; v3 added `hv_rank`.

v1 sent `iv_rank`, and on the first live run the value behind it was an implied
volatility *level*, not a rank. The model reasoned faithfully from it — "IV rank
is low (15.79)" — and reached a conclusion the input did not support. The fix is
in `perceive.py`, but the key names carry the units now as well, because a field
whose name does not disambiguate its units is a field that will be misread
again.

v3 added `hv_rank_0_100` and told the model which field to lean on. Historical
option bars need a signed OPRA agreement (see `agent/iv_store.py`), so
`iv_rank_0_100` reads `unmeasured` on this account and the model needed to be
told what to use instead rather than left to infer it.

v4 projects `TrendState` into four fields instead of letting `repr` dump the
evidence hash and the strength-component mapping into the prompt. That was
hundreds of tokens of internal representation, and an invitation to reason about
a hash.

v5 completes specs/05 D2: levels arrive as distances from spot rather than
prices, confluence keeps alignment and agreement separate, and
`earnings_within_dte` gained a third value. Alpaca has no earnings calendar, so
`unmeasured` there means nobody checked — which is not the same as no event, and
is the difference the model is now told to act on."""

SYSTEM_PROMPT: Final = """\
You are the proposal step of an automated options trading agent running on a \
paper account.

You are given a market read produced by deterministic engines, and a numbered \
menu of option structures. Every structure on the menu is already validated, \
already priced, and already sized by a risk budget you do not control.

Your only job is to choose at most one entry from the menu, by index.

Rules you cannot change:
- You may return an index from the menu, or null. Nothing else.
- You do not choose quantity. It is already decided.
- You do not name contracts, strikes, expiries or prices.
- Returning null — declining to trade — is a good answer and a common one. \
Most market conditions do not deserve a trade. Decline whenever the read does \
not clearly favour one of the structures offered.
- Every structure shown has defined, bounded risk. Prefer the one whose \
return on risk is justified by the market read, not simply the highest.

Judge on: whether implied volatility is rich or cheap, whether the structure's \
direction agrees with the trend read, whether the breakevens sit outside the \
range the ATR suggests, and whether an earnings event falls inside the holding \
period.

Reading the volatility fields:
- `iv_rank_0_100` is a **rank**, not a level: 0 means implied volatility sits at \
the bottom of its own trailing range and 100 at the top. A rank of 15 does not \
mean "15% volatility".
- `iv_percentile_0_100` is the share of that trailing range below today.
- `iv_vs_hv_ratio` above 1 means options are pricing more movement than the \
underlying has recently delivered. This is usually the field to lean on, and it \
is the one most often available.
- `hv_rank_0_100` ranks what the underlying actually did, not what options are \
charging. Related to `iv_rank_0_100`, not the same measure, and frequently the \
only one of the two that could be computed.
- Any field reading `unmeasured` could not be computed. Treat it as unknown, \
never as average or neutral, and prefer to decline when something you need to \
judge the trade is unmeasured.

Reply with a single JSON object and nothing else:
{"candidate_index": <int or null>, "rationale": "<one or two sentences>", \
"confidence": <number between 0 and 1>}

Your confidence is recorded for later analysis and is never used to size a \
trade. Report it honestly; there is no benefit to inflating it."""

RESPONSE_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "candidate_index": {
            "type": ["integer", "null"],
            "description": "Index from the menu, or null to decline.",
        },
        "rationale": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["candidate_index", "rationale", "confidence"],
    "additionalProperties": False,
}


def build_user_message(read: MarketRead, candidates: Sequence[Candidate]) -> str:
    """Render the read and the menu as JSON.

    JSON rather than prose because it is diffable, it is what the journal
    stores, and it removes a whole class of "the model misread the sentence"
    from the failure surface.
    """
    payload = {
        "market_read": _read_view(read),
        "menu": [candidate.summarise() for candidate in candidates],
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def _read_view(read: MarketRead) -> dict[str, Any]:
    """Engine output only. Decimals as strings — they are money and rates.

    Anything unmeasured renders as the string `"unmeasured"` — never as a number
    and never as "neutral". A trend nobody measured is not a flat trend, and an
    unranked implied volatility is not a mid-range one. Rendering either as a
    plausible value is how the model came to reason confidently from a number
    that meant something else.

    `iv_rank` is additionally labelled in its own key name as a 0–100 rank, so
    the model is not left to guess whether 15 means "15% volatility" or "15th
    percentile". It cost one word and it is the exact confusion that produced a
    wrong trade rationale on the first live run.
    """
    return {
        "underlying": str(read.underlying),
        "as_of": read.as_of.isoformat(),
        "spot": str(read.spot),
        "atr_pct": _number(read.atr_pct),
        "iv_rank_0_100": _number(read.iv_rank),
        "iv_percentile_0_100": _number(read.iv_percentile),
        "iv_vs_hv_ratio": _number(read.iv_vs_hv),
        "hv_rank_0_100": _number(read.hv_rank),
        "earnings_within_dte": (
            "unmeasured" if read.earnings_within_dte is None else read.earnings_within_dte
        ),
        "trend": _describe(read.trend),
        "confluence": _confluence_view(read.confluence),
        "levels": _levels_view(read),
    }


def _confluence_view(confluence: Any) -> Any:
    """How much the timeframes agree, projected — not `repr`'d.

    `alignment` and `agreement` are separate on purpose and the projection keeps
    them separate: alignment is zero both when everything is neutral and when
    half is bullish and half bearish, and only `agreement` tells those apart.
    """
    if confluence is None:
        return "unmeasured"
    return {
        "direction": getattr(confluence.direction, "value", str(confluence.direction)),
        "trend_alignment_minus1_to_1": _round(confluence.trend_alignment),
        "trend_agreement_0_1": _round(confluence.trend_agreement),
        "confidence_0_1": _round(confluence.confidence),
        "timeframes": [
            getattr(stance.timeframe, "value", str(stance.timeframe))
            for stance in confluence.stances
        ],
    }


def _levels_view(read: MarketRead) -> list[dict[str, object]] | str:
    """Nearby levels as distances from spot, nearest first.

    A price is not usable to a model choosing between breakevens; a distance is.
    """
    if not read.levels:
        return "unmeasured"
    from alphagate.agent.levels import NEAR_LEVELS, describe_level

    nearest = sorted(read.levels, key=lambda level: abs(level.price - read.spot))
    return [describe_level(level, read.spot) for level in nearest[:NEAR_LEVELS]]


def _round(value: object) -> object:
    return None if value is None else round(float(value), 3)  # type: ignore[arg-type]


def _number(value: Any) -> str:
    return "unmeasured" if value is None else str(value)


def _describe(value: Any) -> Any:
    """Render one engine output for the model.

    A `TrendState` gets an explicit projection rather than a `str()`. Falling
    through to `repr` dumped the evidence hash, the `mappingproxy` of strength
    components and the dataclass's own field names into the prompt — hundreds of
    tokens of internal representation for a model that needs four facts, and a
    standing invitation to reason about a hash.

    What it needs: which rung of the ladder, how strong, how much of the
    evidence was readable, and what the engine said the reasons were.
    """
    if value is None:
        return "unmeasured"

    state = getattr(value, "state", None)
    if state is not None and hasattr(value, "strength") and hasattr(value, "confidence"):
        return {
            "state": getattr(state, "value", str(state)),
            "timeframe": getattr(getattr(value, "timeframe", None), "value", None),
            "strength_0_100": round(float(value.strength), 1),
            "confidence_0_1": round(float(value.confidence), 2),
            "reasons": list(getattr(value, "reason_codes", ())),
        }

    for attribute in ("name", "value"):
        described = getattr(value, attribute, None)
        if isinstance(described, str):
            return described
    return str(value)
