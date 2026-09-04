"""Where option hypotheses come from — specs/10-options-research.md D4, D5.

The same three-part shape as [`proposer.py`](proposer.py), and deliberately not
the same module: the two vocabularies stay unmixed (CLAUDE.md §2), so an equity
proposal cannot acquire a ``structure_type`` and an option proposal cannot
acquire a ``stop_loss_atr_multiple``. What *is* shared is the part worth
sharing — the :class:`~aqr.agent.proposer.Proposal` container, the repair loop's
shape, and ``prompt_hash`` — because those are about talking to a model, not
about what is being proposed.

``TemplateOptionProposer``  a deterministic, offline library of the structures
                            specs/07 and specs/10 were written about, plus
                            mutation of the best rule so far.
``AnthropicOptionProposer`` asks Claude, constrained by a JSON schema.
``OpenAICompatOptionProposer`` / ``DeepSeekOptionProposer`` for endpoints whose
                            JSON mode guarantees only that the bytes parse.

The offline one is not a toy. It makes the option research loop runnable with no
API key, in CI, and reproducibly — and it is the baseline the model has to beat.
It also has a second job the equity side does not need: its first seven
templates are exactly the hand-written structures whose results are recorded in
PLAN-OPTIONS.md, so an offline run reproduces the campaign that produced them.

What no proposer here can do: emit code, choose the underlying, choose its own
risk budget, or name a strike. It returns a small dictionary of fields;
:func:`build_option_spec` turns that into an :class:`~aqr.options.spec.OptionSpec`
and ``OptionSpec``'s own validation gets the final say.
"""

from __future__ import annotations

import json
import os
import random
from collections.abc import Callable, Mapping
from contextlib import suppress
from typing import Any, Protocol

from aqr.agent.option_prompt import (
    DTE_BAND,
    OPTION_PROPOSAL_SCHEMA,
    OPTION_SYSTEM_PROMPT,
    STRUCTURE_CATALOGUE,
    build_option_user_prompt,
)
from aqr.agent.prompts import prompt_hash
from aqr.agent.proposer import Proposal
from aqr.dsl.expr import (  # noqa: I001
    Binary,
    Call,
    Compare,
    Expr,
    Logic,
    Not,
    Number,
    ParseError,
    Unary,
    feature_keys,
    parse,
)
from aqr.features.engine import FeatureKey
from aqr.options.features import resolve_entry_feature
from aqr.options.spec import Anchor, Cadence, DteTarget, OptionSizing, OptionSpec, StructureSpec

__all__ = [
    "AnthropicOptionProposer",
    "DeepSeekOptionProposer",
    "OpenAICompatOptionProposer",
    "OptionProposer",
    "TemplateOptionProposer",
    "CATALOGUE_KEYS",
    "SpanResolver",
    "build_option_spec",
    "catalogue_spans",
    "check_option_proposal",
    "option_spec_to_proposal_fields",
    "unreachable_thresholds",
]

DEFAULT_MODEL = "claude-opus-5"

_KINDS = tuple(kind for kind, _ in STRUCTURE_CATALOGUE)
_ONE_LEG = frozenset({"long_call", "long_put"})


class OptionProposer(Protocol):
    def propose(
        self,
        *,
        underlying: str,
        memory: list[dict[str, Any]],
        instruction: str | None = None,
        parent: dict[str, Any] | None = None,
        budget: tuple[int, int] | None = None,
        span: SpanResolver | None = None,
        spans: Mapping[str, tuple[float, float]] | None = None,
    ) -> Proposal: ...


CATALOGUE_KEYS: tuple[tuple[str, tuple[float, ...]], ...] = (
    ("iv_rank", ()),
    ("iv_current", ()),
    ("hv_current", ()),
    ("iv_hv_spread", ()),
    ("iv_change", (5.0,)),
    ("atm_iv", (28.0,)),
    ("term_slope", ()),
    ("skew_25d", ()),
)
"""The eight option features, at a representative argument where they take one.

``span`` answers about one key a rule actually named; this is the fixed list the
*catalogue* shows before any rule exists, so a model reads the scale of every
feature before it writes a number rather than after. The arity-1 pair is measured
at the arguments a rule most often uses -- 5 for ``iv_change`` (the vendor's week
column) and 28 for ``atm_iv`` (the cache's median DTE, and that feature's own
default).
"""


def catalogue_spans(span: SpanResolver) -> dict[str, tuple[float, float]]:
    """Measured ranges for :data:`CATALOGUE_KEYS`, for the prompt."""
    out: dict[str, tuple[float, float]] = {}
    for name, args in CATALOGUE_KEYS:
        bounds = span(FeatureKey(name, args))
        if bounds is not None:
            out[name] = bounds
    return out


# --------------------------------------------------------------------------- #
# Proposal -> OptionSpec
# --------------------------------------------------------------------------- #


def _number(fields: dict[str, Any], name: str, default: float = 0.0) -> float:
    value = fields.get(name)
    if value is None or isinstance(value, bool):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def build_option_spec(
    proposal: Proposal,
    underlying: str = "SPY",
    *,
    risk_per_trade: float = 0.02,
    max_concurrent: int = 3,
    anchor_tolerance: float = 0.06,
    dte_tolerance: int = 10,
    parent_fingerprint: str | None = None,
) -> OptionSpec:
    """Turn a proposal into a validated :class:`OptionSpec`.

    The underlying, the risk budget and the concurrency limit come from the run
    configuration, never from the proposal — the same rule the equity
    ``build_spec`` follows, for the same reason: a model that could choose its
    own risk budget could buy its own statistical significance. specs/10 D8a
    measured exactly that, on the real cache: the same rule at 1% and 2% of
    equity produces 21 and 57 independent cycles, with nothing about the rule
    changed.

    The two *tolerances* are configuration rather than proposal fields as well,
    and that is a narrower claim than it looks. ``anchor.tolerance`` and
    ``dte.tolerance`` do not describe the hypothesis; they describe how far the
    engine may stray from it before refusing, which is a property of how thin
    the vendor's ladder is. A model tuning them would be tuning the cache's
    sampling, and specs/10 D5 is explicit that selecting on the vendor's
    sampling is not research.
    """
    fields = proposal.fields
    kind = str(fields.get("structure_type") or "").strip()
    width_delta = _number(fields, "width_delta")
    call_anchor_delta = _number(fields, "call_anchor_delta")
    call_width_delta = _number(fields, "call_width_delta")

    structure = StructureSpec(
        type=kind,  # type: ignore[arg-type]
        dte=DteTarget(target=int(fields.get("dte_target") or 28), tolerance=dte_tolerance),
        anchor=Anchor(
            delta=_number(fields, "anchor_delta", 0.16), tolerance=anchor_tolerance
        ),
        width_delta=width_delta or None,
        call_anchor=(
            Anchor(delta=call_anchor_delta, tolerance=anchor_tolerance)
            if call_anchor_delta
            else None
        ),
        call_width_delta=call_width_delta or None,
    )
    return OptionSpec(
        name=str(fields["name"]).strip(),
        underlying=underlying.upper(),
        entry=str(fields["entry"]).strip(),
        structure=structure,
        sizing=OptionSizing(risk_per_trade=risk_per_trade, max_concurrent=max_concurrent),
        cadence=Cadence(
            min_sessions_between_entries=int(
                fields.get("min_sessions_between_entries") or 5
            )
        ),
        hypothesis=str(fields.get("hypothesis") or "").strip(),
        parent=parent_fingerprint,
    )


# --------------------------------------------------------------------------- #
# Checking what a model sent back
# --------------------------------------------------------------------------- #


def check_option_proposal(
    fields: dict[str, Any], *, span: SpanResolver | None = None
) -> list[str]:
    """Problems with a raw option proposal, phrased so a model can act on them.

    Runs *before* the spec is built, and its output is fed straight back to the
    model as a repair instruction — which is why every message names the
    offending value and says what to do instead. The point is not to log a
    complaint; it is to make the next attempt succeed.

    Deliberately duplicates a few of ``StructureSpec``'s own invariants rather
    than catching its ``ValueError``. The dataclass raises on the first fault
    and stops; a model that got three fields wrong deserves to hear about all
    three in one turn, because the alternative is three round trips out of a
    twenty-hypothesis budget.

    ``span`` is what turns "this rule happens not to fire" into "this rule
    *cannot* fire", and it is the difference between a slot spent and a slot
    saved -- see :func:`unreachable_thresholds`. Optional because a caller with
    no market to measure against (a schema test, a hand-written spec) can still
    check everything else; absent, the range check simply does not run.

    Returns an empty list when the proposal is usable.
    """
    problems: list[str] = []
    if not isinstance(fields, dict):
        return [f"expected a JSON object, got {type(fields).__name__}"]

    for required in ("name", "hypothesis"):
        value = fields.get(required)
        if not isinstance(value, str) or not value.strip():
            problems.append(f"field {required!r} is missing or empty")

    kind = fields.get("structure_type")
    if kind not in _KINDS:
        problems.append(
            f"structure_type must be one of {list(_KINDS)}, got {kind!r}. There is no "
            "'custom' and no structure with an uncovered short leg: if it cannot be "
            "constructed it cannot be measured (specs/10 D4)."
        )

    dte = fields.get("dte_target")
    if not isinstance(dte, int) or isinstance(dte, bool):
        problems.append(f"dte_target must be a whole number of days, got {dte!r}")
    elif not DTE_BAND[0] <= dte <= DTE_BAND[1]:
        problems.append(
            f"dte_target must be between {DTE_BAND[0]} and {DTE_BAND[1]} days, got "
            f"{dte}; that is the whole range of expiries the cache lists, and one "
            "outside it names a contract that cannot be priced"
        )

    anchor = fields.get("anchor_delta")
    if isinstance(anchor, bool) or not isinstance(anchor, (int, float)):
        problems.append(f"anchor_delta must be a number, got {anchor!r}")
    elif not 0.0 < float(anchor) < 1.0:
        problems.append(
            f"anchor_delta is a magnitude in (0, 1), got {anchor}. A put's delta is "
            "negative in the data; name 0.16 for either right."
        )

    width = fields.get("width_delta")
    if isinstance(width, bool) or not isinstance(width, (int, float)):
        problems.append(f"width_delta must be a number, got {width!r}")
    elif kind in _ONE_LEG and float(width) != 0.0:
        problems.append(
            f"{kind} has one leg and no width; set width_delta to 0"
        )
    elif kind in _KINDS and kind not in _ONE_LEG:
        if float(width) <= 0.0:
            problems.append(
                f"{kind} requires width_delta > 0: risk is the width less the credit, "
                "so a spread without one has no maximum loss to size against"
            )
        elif (
            isinstance(anchor, (int, float))
            and not isinstance(anchor, bool)
            and float(width) >= float(anchor)
        ):
            problems.append(
                f"width_delta {width} is not below anchor_delta {anchor}: the "
                "protective leg is further out of the money than the leg it "
                "protects, so it has the smaller delta"
            )

    for name in ("call_anchor_delta", "call_width_delta"):
        value = fields.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            problems.append(f"{name} must be a number ({0} when unused), got {value!r}")
            continue
        if float(value) and kind != "iron_condor":
            problems.append(f"{name} is iron-condor only; set it to 0 for {kind}")
        if float(value) < 0.0:
            problems.append(f"{name} must not be negative, got {value}")
    call_anchor = _number(fields, "call_anchor_delta")
    call_width = _number(fields, "call_width_delta")
    if call_anchor and call_width and call_width >= call_anchor:
        problems.append(
            f"call_width_delta {call_width} is not below call_anchor_delta "
            f"{call_anchor}: the call side's wing is further out of the money too"
        )

    cadence = fields.get("min_sessions_between_entries")
    if not isinstance(cadence, int) or isinstance(cadence, bool) or cadence < 1:
        problems.append(
            f"min_sessions_between_entries must be a whole number >= 1, got {cadence!r}"
        )

    cycles = fields.get("expected_cycles_per_year")
    if cycles is not None and (isinstance(cycles, bool) or not isinstance(cycles, int)):
        problems.append(f"expected_cycles_per_year must be a whole number, got {cycles!r}")

    # The one field that has to survive a real parser. A near-miss here is the
    # common failure and it is entirely recoverable, so it is checked against the
    # combined option/bar vocabulary the spec itself will use rather than against
    # a list in the prompt.
    entry = fields.get("entry")
    if not isinstance(entry, str) or not entry.strip():
        problems.append(
            "field 'entry' is missing or empty. A rule with no condition opens a "
            "position every session it can, which is a schedule rather than a "
            "hypothesis."
        )
    else:
        try:
            node = parse(entry, resolve_feature=resolve_entry_feature)
        except ParseError as exc:
            problems.append(f"entry does not parse: {exc}")
            if _mentions_exit(entry):
                # The most likely reason a model reached for a feature that does
                # not exist, and the parser's own "unknown feature 'stop_loss'"
                # reads as a misspelling rather than as "that concept is absent
                # from this system on purpose". Appended rather than substituted:
                # the parser's message names the token, this one says why there
                # is no token to name.
                problems.append(
                    "entry names something that reads like an exit. There is no exit "
                    "here and no feature for one: every structure is held to expiry "
                    "(specs/10 D1). Put the condition that OPENS the position in "
                    "'entry' and nothing else."
                )
        else:
            if not isinstance(node, (Compare, Logic, Not)):
                problems.append(
                    f"entry is an expression, not a condition: {entry!r}. Compare it "
                    f"to something, for example '{entry} > 0'."
                )
            elif span is not None:
                # Last, and only once the expression is known to be a condition:
                # a threshold nothing can satisfy is a rule that opens nothing,
                # and finding that out from the engine costs a slot of the search
                # budget. Finding it out here costs a retry.
                problems += unreachable_thresholds(entry, span)
    return problems


def _named_ranges(entry: str, span: SpanResolver | None) -> str:
    """What every feature the rule named actually ranges over."""
    if span is None or not entry.strip():
        return ""
    try:
        node = parse(entry, resolve_feature=resolve_entry_feature)
    except ParseError:
        return ""
    lines: list[str] = []
    for key in sorted(feature_keys(node), key=str):
        bounds = span(key)
        if bounds is not None:
            lines.append(f"  {key}: {bounds[0]:.4g} .. {bounds[1]:.4g}")
    if not lines:
        return ""
    return (
        "The features your condition names, and the range each one actually "
        "takes over the research window:\n" + "\n".join(lines) + "\n\n"
    )


_EXIT_WORDS = ("stop", "target", "roll", "exit", "manage")


def _mentions_exit(entry: str) -> bool:
    """Whether an entry expression smuggles in a management rule.

    Cheap and deliberately shallow: the DSL has no exit vocabulary, so anything
    matching here already failed to parse or parsed as something else entirely.
    It exists to turn a confusing ``unknown feature 'stop_loss'`` into a message
    that says why the field does not exist rather than that it was misspelled.
    """
    lowered = entry.lower()
    return any(word in lowered for word in _EXIT_WORDS)


def _repair_instruction(problems: list[str], raw: str) -> str:
    joined = "\n".join(f"- {p}" for p in problems)
    return (
        "Your previous answer was rejected. Fix these problems and return the "
        "complete JSON object again:\n"
        f"{joined}\n\n"
        f"Your previous answer was:\n{raw[:1500]}"
    )


def _dead_rule_instruction(
    fields: dict[str, Any], problems: list[str], span: SpanResolver | None = None
) -> str:
    """Ask for a fix to a rule that compiled and then opened nothing.

    Narrow on purpose, and the option version says one thing the equity version
    cannot: a rule here can be perfectly satisfiable and still trade nothing,
    because the ladder had no leg at the named delta or the account could not
    afford the structure (specs/10 D8a). Handing the model "loosen the
    condition" for an affordability skip would have it weaken a hypothesis to
    fix an account, so the census is quoted and the model is told which of the
    two it is looking at.
    """
    joined = "\n".join(f"- {p}" for p in problems)
    # The ranges of the features THIS rule named, not a fixed example. The first
    # version of this instruction quoted iv_rank()'s range regardless, and in the
    # campaign that motivated it iv_rank() was the one feature that was never
    # the problem: the model read a range for a feature it had not misused and
    # repeated its unit error on the ones it had.
    ranges = _named_ranges(str(fields.get("entry") or ""), span)
    return (
        "Your hypothesis compiled correctly but produced no usable evidence.\n\n"
        f"{joined}\n\n"
        f"{ranges}"
        "Here is what you proposed:\n"
        f"  entry:      {fields.get('entry', '')!r}\n"
        f"  structure:  {fields.get('structure_type', '')} at "
        f"{fields.get('dte_target', '')} DTE, anchor delta "
        f"{fields.get('anchor_delta', '')}, width delta {fields.get('width_delta', '')}\n"
        f"  cadence:    every {fields.get('min_sessions_between_entries', '')} sessions\n\n"
        "Keep the same mechanism and the same structure. Change the condition, the "
        "cadence or the anchor delta so the rule can actually open positions -- "
        "check every threshold against the ranges above, and check that a cadence "
        "shorter than the DTE target is what you meant. Return the complete JSON "
        "object again."
    )


# --------------------------------------------------------------------------- #
# Claude
# --------------------------------------------------------------------------- #


class AnthropicOptionProposer:
    """Asks Claude for an option hypothesis, constrained by the schema.

    The schema is the containment boundary, exactly as on the equity side: the
    model cannot return code because there is no field to put code in, cannot
    invent a structure because ``structure_type`` is an enum, and cannot invent
    a feature because the entry expression is parsed against the real
    vocabulary before anything runs.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        api_key: str | None = None,
        max_tokens: int = 8000,
        effort: str = "high",
    ) -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                "AnthropicOptionProposer needs the 'llm' extra: uv sync --extra llm"
            ) from exc
        self._client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort

    def propose(
        self,
        *,
        underlying: str,
        memory: list[dict[str, Any]],
        instruction: str | None = None,
        parent: dict[str, Any] | None = None,
        budget: tuple[int, int] | None = None,
        span: SpanResolver | None = None,
        spans: Mapping[str, tuple[float, float]] | None = None,
    ) -> Proposal:
        user = build_option_user_prompt(
            underlying=underlying,
            memory=memory,
            instruction=instruction,
            parent=parent,
            budget=budget,
            spans=spans,
        )
        # Anthropic's json_schema mode constrains the *shape* of the reply and
        # can say nothing about whether a number is inside a feature's range, so
        # the range check needs its own turn here exactly as it does on the
        # OpenAI-compatible path.
        return self._ask(user, span=span)

    def repair(
        self,
        *,
        proposal: Proposal,
        problems: list[str],
        span: SpanResolver | None = None,
    ) -> Proposal:
        """One more turn for a rule that opened nothing."""
        return self._ask(
            _dead_rule_instruction(proposal.fields, problems, span), span=span
        )

    def _ask(self, user: str, *, span: SpanResolver | None = None) -> Proposal:
        text = self._turn([{"role": "user", "content": user}])
        try:
            fields = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"model returned non-JSON output: {text[:400]}") from exc

        # ``json_schema`` mode constrains the *shape* of the reply and can say
        # nothing about whether a number lies inside a feature's range, so the
        # range check needs a turn of its own here exactly as it does on the
        # OpenAI-compatible path. One retry, not a loop: a model that cannot put
        # a threshold inside a range it was just handed will not manage it on
        # the third attempt, and the next iteration is a cheaper place to spend
        # the tokens.
        problems = check_option_proposal(fields, span=span) if span is not None else []
        if problems:
            retry = self._turn(
                [
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": text},
                    {"role": "user", "content": _repair_instruction(problems, text)},
                ]
            )
            # A retry that comes back unparseable leaves the original standing:
            # the loop then evaluates it and it is rejected with the reason it
            # earned, which is a better record than an exception here.
            with suppress(json.JSONDecodeError):
                fields = json.loads(retry)
                text = retry
        return Proposal(
            fields=fields,
            source="anthropic",
            model=self.model,
            prompt_hash=prompt_hash(OPTION_SYSTEM_PROMPT, user),
            raw=text,
        )

    def _turn(self, messages: list[dict[str, str]]) -> str:
        """One request, and the one place the SDK's TypedDicts are widened.

        The SDK types ``messages``, ``thinking`` and ``output_config`` as
        TypedDicts. Importing them would pull ``anthropic`` to module scope and
        break the optional-extra design, so the widening happens here, at the
        boundary, and nowhere else -- which is also why the retry above goes
        through this rather than repeating the call.
        """
        widened: Any = messages
        thinking: Any = {"type": "adaptive"}
        output_config: Any = {
            "effort": self.effort,
            "format": {"type": "json_schema", "schema": OPTION_PROPOSAL_SCHEMA},
        }
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=OPTION_SYSTEM_PROMPT,
            messages=widened,
            thinking=thinking,
            output_config=output_config,
        )
        if getattr(response, "stop_reason", None) == "refusal":
            details = getattr(response, "stop_details", None)
            raise RuntimeError(f"model declined to answer: {details}")
        return "".join(block.text for block in response.content if block.type == "text")


# --------------------------------------------------------------------------- #
# OpenAI-compatible endpoints
# --------------------------------------------------------------------------- #

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"


class OpenAICompatOptionProposer:
    """Any chat endpoint that speaks the OpenAI protocol.

    Same difference in guarantees the equity version documents: Anthropic's
    ``json_schema`` mode constrains the shape of the output, DeepSeek's JSON
    mode guarantees only that the bytes parse. So the schema is rendered into
    the prompt, the reply is checked against the real grammar and the real
    structure whitelist, and a rejected proposal gets a bounded number of repair
    turns with the specific errors attached.
    """

    def __init__(
        self,
        model: str = DEEPSEEK_MODEL,
        *,
        base_url: str = DEEPSEEK_BASE_URL,
        api_key: str | None = None,
        provider: str = "deepseek",
        max_tokens: int = 4000,
        temperature: float = 1.0,
        repair_attempts: int = 1,
        timeout: float = 180.0,
        client: Any | None = None,
    ) -> None:
        # ``client`` is an injection point for tests. The repair loop -- the part
        # most worth testing -- is independent of the network, and requiring an
        # API key to exercise it would mean it never got tested.
        if client is not None:
            self._client = client
        else:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - depends on optional extra
                raise RuntimeError(
                    "OpenAICompatOptionProposer needs the 'llm' extra: "
                    "uv pip install -e '.[llm]'"
                ) from exc
            if api_key is None:
                from aqr.config import credential

                api_key = credential(provider)
            self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self.model = model
        self.provider = provider
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.repair_attempts = max(0, repair_attempts)

    def _system_prompt(self) -> str:
        return (
            f"{OPTION_SYSTEM_PROMPT}\n"
            "Return a single JSON object and nothing else -- no prose, no code "
            "fence. It must match this JSON schema exactly:\n"
            f"{json.dumps(OPTION_PROPOSAL_SCHEMA, indent=2)}"
        )

    def propose(
        self,
        *,
        underlying: str,
        memory: list[dict[str, Any]],
        instruction: str | None = None,
        parent: dict[str, Any] | None = None,
        budget: tuple[int, int] | None = None,
        span: SpanResolver | None = None,
        spans: Mapping[str, tuple[float, float]] | None = None,
    ) -> Proposal:
        system = self._system_prompt()
        user = build_option_user_prompt(
            underlying=underlying,
            memory=memory,
            instruction=instruction,
            parent=parent,
            budget=budget,
            spans=spans,
        )
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        last_error = "the model returned nothing usable"
        raw = ""
        for attempt in range(self.repair_attempts + 1):
            raw = self._call(messages)
            try:
                fields = json.loads(raw)
            except json.JSONDecodeError as exc:
                problems = [f"the reply was not valid JSON: {exc}"]
            else:
                problems = check_option_proposal(fields, span=span)
                if not problems:
                    return Proposal(
                        fields=fields,
                        source=self.provider,
                        model=self.model,
                        prompt_hash=prompt_hash(system, user),
                        raw=raw,
                    )
            last_error = "; ".join(problems)
            if attempt < self.repair_attempts:
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": _repair_instruction(problems, raw)})

        raise ValueError(f"{self.provider} produced an unusable option proposal: {last_error}")

    def repair(
        self,
        *,
        proposal: Proposal,
        problems: list[str],
        span: SpanResolver | None = None,
    ) -> Proposal:
        """One more turn, carrying the reason the rule produced nothing.

        Raises rather than returning something unchecked: the research loop
        treats a raised repair as "no repair" and evaluates the original, which
        keeps the real rejection reason attached to the real attempt.
        """
        system = self._system_prompt()
        user = _dead_rule_instruction(proposal.fields, problems, span)
        raw = self._call(
            [{"role": "system", "content": system}, {"role": "user", "content": user}]
        )
        try:
            fields = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{self.provider} repair returned non-JSON: {raw[:200]}") from exc
        problems_now = check_option_proposal(fields, span=span)
        if problems_now:
            raise ValueError(
                f"{self.provider} repair is still unusable: {'; '.join(problems_now)}"
            )
        return Proposal(
            fields=fields,
            source=self.provider,
            model=self.model,
            prompt_hash=prompt_hash(system, user),
            raw=raw,
        )

    def _call(self, messages: list[dict[str, str]]) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format={"type": "json_object"},
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return (response.choices[0].message.content or "").strip()


def DeepSeekOptionProposer(
    model: str = DEEPSEEK_MODEL, **kwargs: Any
) -> OpenAICompatOptionProposer:
    """DeepSeek preset. ``deepseek-chat`` for the loop, ``deepseek-reasoner`` to think harder."""
    kwargs.setdefault("base_url", DEEPSEEK_BASE_URL)
    kwargs.setdefault("provider", "deepseek")
    return OpenAICompatOptionProposer(model, **kwargs)


# --------------------------------------------------------------------------- #
# Offline
# --------------------------------------------------------------------------- #

# Each template is a *mechanism* with a stated rationale, and the first seven are
# the structures PLAN-OPTIONS.md records results for. An offline campaign
# therefore reproduces that table rather than exploring somewhere new, which is
# what makes it a baseline: a model that cannot beat these has not added
# research value, and without them that would be impossible to notice.
_TEMPLATES: list[dict[str, Any]] = [
    {
        "name": "put_credit_spread_unconditional",
        "hypothesis": (
            "Index puts carry a variance risk premium: buyers pay more than the "
            "realised distribution justifies because they are hedging a portfolio "
            "they already own, and that demand is price-insensitive. Selling a "
            "defined-risk put spread harvests the premium without an unbounded tail."
        ),
        "structure_type": "put_credit_spread",
        "dte_target": 28,
        "anchor_delta": 0.16,
        "width_delta": 0.06,
        "min_sessions_between_entries": 5,
    },
    {
        "name": "put_credit_spread_in_uptrend",
        "hypothesis": (
            "The variance risk premium is largest when the hedging demand is not "
            "being validated by the market. Conditioning the same short put spread "
            "on an uptrend avoids selling insurance into the regime where the "
            "insurer is most likely to pay out."
        ),
        "entry": "close > sma(200)",
        "structure_type": "put_credit_spread",
        "dte_target": 28,
        "anchor_delta": 0.16,
        "width_delta": 0.06,
        "min_sessions_between_entries": 5,
    },
    {
        "name": "put_credit_spread_high_iv_rank",
        "hypothesis": (
            "The premium paid for index puts is a spread over realised risk, so it "
            "should be widest when implied volatility is high relative to its own "
            "recent range. Selling only into a high IV rank should collect a larger "
            "credit for the same distance out of the money."
        ),
        "entry": "iv_rank() > 50",
        "structure_type": "put_credit_spread",
        "dte_target": 28,
        "anchor_delta": 0.16,
        "width_delta": 0.06,
        "min_sessions_between_entries": 5,
    },
    {
        "name": "iron_condor_neutral",
        "hypothesis": (
            "If the variance risk premium is a premium on the whole distribution "
            "rather than on the downside alone, then selling both tails should "
            "collect twice and the two directional exposures should largely cancel, "
            "leaving a position paid for bearing realised variance."
        ),
        "structure_type": "iron_condor",
        "dte_target": 28,
        "anchor_delta": 0.16,
        "width_delta": 0.06,
        "min_sessions_between_entries": 5,
    },
    {
        "name": "long_put_crash_hedge",
        "hypothesis": (
            "The mirror of the premium claim, run as a control: if index puts are "
            "expensive then buying them systematically must lose money, and the "
            "size of that loss is the size of the premium the short side is being "
            "paid. A campaign that only tests the profitable direction has not "
            "measured the premium, it has assumed it."
        ),
        "structure_type": "long_put",
        "dte_target": 28,
        "anchor_delta": 0.16,
        "width_delta": 0.0,
        "min_sessions_between_entries": 5,
    },
    {
        "name": "call_credit_spread_unconditional",
        "hypothesis": (
            "The upside call skew is thinner than the downside put skew, so if the "
            "premium is compensation for variance rather than for crash risk "
            "specifically, a short call spread should be paid too -- and if it is "
            "not, the premium is directional and the short put spread's result is "
            "about the drift, not about variance."
        ),
        "structure_type": "call_credit_spread",
        "dte_target": 28,
        "anchor_delta": 0.16,
        "width_delta": 0.06,
        "min_sessions_between_entries": 5,
    },
    {
        "name": "put_credit_spread_iv_over_hv",
        "hypothesis": (
            "The variance risk premium is the gap between implied and realised "
            "volatility, so condition directly on that gap rather than on implied "
            "volatility's own rank: a high IV rank in a genuinely volatile market "
            "is not a premium, it is a forecast."
        ),
        "entry": "iv_hv_spread() > 0.02",
        "structure_type": "put_credit_spread",
        "dte_target": 28,
        "anchor_delta": 0.16,
        "width_delta": 0.06,
        "min_sessions_between_entries": 5,
    },
    {
        "name": "put_credit_spread_short_dated",
        "hypothesis": (
            "Time decay is convex in the last weeks of a contract's life, so a "
            "short-dated short put spread should collect a larger fraction of its "
            "maximum profit per unit of time at risk -- at the cost of a thinner "
            "cushion against the same move."
        ),
        "structure_type": "put_credit_spread",
        "dte_target": 14,
        "anchor_delta": 0.16,
        "width_delta": 0.06,
        "min_sessions_between_entries": 3,
    },
    {
        "name": "put_credit_spread_term_slope",
        "hypothesis": (
            "An upward-sloping term structure says the market expects volatility to "
            "rise from a calm present. Selling the near expiry into that slope is "
            "being paid the forward's premium over the spot, which is a different "
            "claim from selling into a high absolute level."
        ),
        "entry": "term_slope() > 0.01",
        "structure_type": "put_credit_spread",
        "dte_target": 28,
        "anchor_delta": 0.16,
        "width_delta": 0.06,
        "min_sessions_between_entries": 5,
    },
    {
        "name": "iron_condor_calm_regime",
        "hypothesis": (
            "Selling both tails is a bet on realised variance staying inside the "
            "implied range, which is a bet against volatility clustering. "
            "Conditioning on a low IV rank selects the regime where the market "
            "itself expects the range to hold."
        ),
        "entry": "iv_rank() < 30",
        "structure_type": "iron_condor",
        "dte_target": 28,
        "anchor_delta": 0.16,
        "width_delta": 0.06,
        "min_sessions_between_entries": 5,
    },
    {
        "name": "put_credit_spread_after_a_drop",
        "hypothesis": (
            "Insurance is bought after the accident. A sharp fall raises hedging "
            "demand from holders who did not hedge beforehand, so the premium on "
            "downside protection should be widest just after a drop rather than "
            "before one."
        ),
        "entry": "roc(5) < -0.02 and close > sma(200)",
        "structure_type": "put_credit_spread",
        "dte_target": 28,
        "anchor_delta": 0.16,
        "width_delta": 0.06,
        "min_sessions_between_entries": 5,
    },
    {
        "name": "put_debit_spread_momentum",
        "hypothesis": (
            "A control on the direction rather than on the premium: if a short put "
            "spread makes money in a downtrend as well as an uptrend, its result is "
            "the premium; if a LONG put spread makes money in a downtrend, the "
            "result was the drift all along."
        ),
        "entry": "close < sma(200)",
        "structure_type": "put_debit_spread",
        "dte_target": 28,
        "anchor_delta": 0.30,
        "width_delta": 0.16,
        "min_sessions_between_entries": 5,
    },
]


class TemplateOptionProposer:
    """Deterministic offline search: the template library, then mutation.

    Phase one walks the library so every mechanism gets one honest test. Phase
    two perturbs the best rule found so far, one knob at a time — the same
    arrangement the equity ``HeuristicProposer`` uses, and for the same reason:
    a mutation that changes three parameters and improves the result has taught
    nobody which of the three mattered.

    Seeded, so an offline campaign is reproducible. ``random`` is allowed in
    this layer; what is not allowed is an unseeded generator.
    """

    def __init__(self, seed: int = 20260901) -> None:
        self._rng = random.Random(seed)
        self._used: set[str] = set()
        self._lineage: dict[str, int] = {}
        # Seeded with the library itself: a recombination that reproduces a
        # template is not a new hypothesis, and the registry would reject it on
        # the fingerprint anyway -- one iteration later, and one iteration is 5%
        # of this search's whole budget.
        self._seen: set[tuple[Any, ...]] = {_signature(_complete(t)) for t in _TEMPLATES}

    def propose(
        self,
        *,
        underlying: str,
        memory: list[dict[str, Any]],
        instruction: str | None = None,
        parent: dict[str, Any] | None = None,
        budget: tuple[int, int] | None = None,
        span: SpanResolver | None = None,
        spans: Mapping[str, tuple[float, float]] | None = None,
    ) -> Proposal:
        if parent:
            return self._mutate(parent)
        for template in _TEMPLATES:
            if template["name"] not in self._used:
                self._used.add(template["name"])
                return Proposal(fields=_complete(template), source="offline-template")
        # Library exhausted. Recombine an entry condition with a different
        # structure, which is the cheapest way to ask a genuinely new question
        # of the same data without inventing a mechanism nobody argued for.
        #
        # Only from templates that *have* a condition, and only onto a structure
        # that does not already carry it. Sampling freely produced pairs whose
        # combination was byte-identical to a template already tried -- the loop
        # catches those on the fingerprint and skips the backtest, but the
        # iteration is still gone, and against a budget of twenty that is 5% of
        # the search spent rediscovering something.
        conditioned = [t for t in _TEMPLATES if t.get("entry")]
        pairs = [
            (left, right)
            for left in conditioned
            for right in _TEMPLATES
            if _signature(_recombine(left, right)) not in self._seen
        ]
        if pairs:
            left, right = self._rng.choice(pairs)
            fields = _recombine(left, right)
            self._seen.add(_signature(fields))
            return Proposal(fields=fields, source="offline-template")
        # Everything expressible from the library has been asked. Say so rather
        # than emitting a duplicate the loop will reject a moment later.
        raise ValueError(
            "the offline template library is exhausted: every structure and every "
            "combination of its conditions has been proposed. Run with --provider "
            "deepseek or anthropic for hypotheses the library does not contain."
        )

    def _mutate(self, parent: dict[str, Any]) -> Proposal:
        """One knob changed, everything else held fixed.

        The knobs are the anchor delta, the DTE target and the cadence, in that
        order of how much they change the hypothesis: moving the anchor changes
        which premium is being harvested, moving the DTE changes which term of
        it, and moving the cadence changes only how many independent bets the
        run gets. The width is not mutated on its own, because a width that
        crosses the anchor is not a variant of the rule -- it is a different
        structure -- so it moves with the anchor instead.
        """
        fields = _complete(parent)
        knob = self._rng.choice(["anchor_delta", "dte_target", "min_sessions_between_entries"])
        if knob == "dte_target":
            current = int(fields["dte_target"])
            # Move to a different *bucket* rather than to an adjacent day: 28 to
            # 29 is not a variant anyone can distinguish, and the engine's
            # ±10-day tolerance would resolve both to the same expiry.
            others = [d for d in (14, 21, 28, 35, 49, 56) if abs(d - current) >= 7]
            fields["dte_target"] = self._rng.choice(others)
            change = f"DTE {parent.get('dte_target')} -> {fields['dte_target']}"
        elif knob == "anchor_delta":
            factor = self._rng.choice([0.7, 0.85, 1.2, 1.5])
            anchor = min(0.45, max(0.05, round(float(fields["anchor_delta"]) * factor, 3)))
            fields["anchor_delta"] = anchor
            if float(fields["width_delta"]):
                # Keep the wing a fixed fraction of the anchor so the mutation
                # stays a variant rather than becoming an invalid structure.
                fields["width_delta"] = round(anchor * 0.375, 3)
            change = f"anchor delta {parent.get('anchor_delta')} -> {anchor}"
        else:
            factor = self._rng.choice([0.5, 2.0, 3.0])
            cadence = max(1, int(round(int(fields["min_sessions_between_entries"]) * factor)))
            fields["min_sessions_between_entries"] = cadence
            change = f"cadence {parent.get('min_sessions_between_entries')} -> {cadence}"

        base = str(parent.get("name", "option_rule")).rsplit("_v", 1)[0]
        # A monotonic per-lineage counter, not parent_version + 1: two mutations
        # of the same parent would otherwise both be "_v2".
        self._lineage[base] = self._lineage.get(base, 1) + 1
        fields["name"] = f"{base}_v{self._lineage[base]}"
        fields["hypothesis"] = f"{parent.get('hypothesis', '')} Variant: {change}."
        return Proposal(fields=fields, source="offline-mutation")


def _signature(fields: dict[str, Any]) -> tuple[Any, ...]:
    """What makes two offline proposals the same *rule*.

    Deliberately not the name and not the hypothesis: ``OptionSpec.fingerprint``
    drops both for exactly this reason — renaming a strategy must not make it
    look new. This is the same equivalence, computed before a spec is built, so
    the proposer can avoid emitting the duplicate rather than the loop having to
    catch it.
    """
    return (
        str(fields.get("entry", "")),
        str(fields.get("structure_type", "")),
        int(fields.get("dte_target", 0)),
        float(fields.get("anchor_delta", 0.0)),
        float(fields.get("width_delta", 0.0)),
        float(fields.get("call_anchor_delta", 0.0)),
        float(fields.get("call_width_delta", 0.0)),
        int(fields.get("min_sessions_between_entries", 0)),
    )


def _recombine(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """``right``'s structure under ``left``'s condition."""
    fields = _complete(right)
    fields["name"] = f"{right['name']}_under_{left['name']}"
    fields["entry"] = str(left["entry"])
    fields["hypothesis"] = (
        f"{right['hypothesis']} Tested under a different condition: {fields['entry']}."
    )
    return fields


def _complete(template: dict[str, Any]) -> dict[str, Any]:
    """Fill a template's unstated fields with the schema's own defaults.

    ``entry`` defaults to ``close > 0`` — true on every session with a bar —
    because the templates that state no condition are the unconditional ones,
    and ``OptionSpec`` refuses an empty entry outright. Writing it as a literal
    keeps "this rule has no condition" visible in the artefact instead of
    hiding it in a missing field.
    """
    fields = dict(template)
    fields.setdefault("entry", "close > 0")
    fields.setdefault("width_delta", 0.0)
    fields.setdefault("call_anchor_delta", 0.0)
    fields.setdefault("call_width_delta", 0.0)
    fields.setdefault("min_sessions_between_entries", 5)
    fields.setdefault("expected_cycles_per_year", 12)
    return fields


def option_spec_to_proposal_fields(spec: OptionSpec) -> dict[str, Any]:
    """Render a spec back into proposal fields, for mutation and for the prompt.

    A rule that could not be described back to the model would be invisible to
    the search that produced it, and the model would keep re-proposing it.
    """
    structure = spec.structure
    return {
        "name": spec.name,
        "hypothesis": spec.hypothesis,
        "entry": spec.entry,
        "structure_type": structure.type,
        "dte_target": structure.dte.target,
        "anchor_delta": structure.anchor.delta,
        "width_delta": structure.width_delta or 0.0,
        "call_anchor_delta": (
            structure.call_anchor.delta if structure.call_anchor is not None else 0.0
        ),
        "call_width_delta": structure.call_width_delta or 0.0,
        "min_sessions_between_entries": spec.cadence.min_sessions_between_entries,
        "expected_cycles_per_year": 12,
    }


# --------------------------------------------------------------------------- #
# Thresholds that can never be true
# --------------------------------------------------------------------------- #

SpanResolver = Callable[[FeatureKey], "tuple[float, float] | None"]
"""What a feature's value actually ranges over, or ``None`` if unknown.

A callable rather than a mapping because the set of keys is not known until a
proposal names them: ``realized_vol(20)`` and ``realized_vol(60)`` are different
keys and a campaign cannot enumerate the arguments a model might choose.
:func:`~aqr.options.features.feature_span` is the implementation; the research
loop binds it to the research market.
"""


def unreachable_thresholds(entry: str, span: SpanResolver) -> list[str]:
    """Comparisons in ``entry`` that no session can satisfy.

    The failure this exists for was measured on a real campaign, not imagined.
    Seven of twenty hypotheses died writing ``term_slope() > 5``,
    ``iv_hv_spread() > 2``, ``iv_current() > 25`` and ``realized_vol(20) > 15``:
    the volatility features here are decimal fractions and the model wrote
    percentage points, because ``iv_rank()`` is the one feature on a 0..100
    scale and it was the only one whose documentation said so. ``term_slope()``
    never exceeds 0.052, so ``> 5`` is off by a factor of ninety-six.

    Every one of those is a *good hypothesis in the wrong unit*, and every one
    cost a slot of a fixed search budget -- plus, for five of them, a second slot
    on a repair turn that reproduced the same class of error. Catching it here
    costs nothing: this runs inside the proposer's own retry loop, before a spec
    exists and before anything is recorded.

    Deliberately narrow. Only ``feature <op> number`` (and the mirrored form) is
    checked, and only against the *whole observed range* -- not a percentile, not
    a "this looks rare" warning. A rule firing on 2% of sessions is a small
    sample and the evaluator already says so with the cycle gate; a rule firing
    on 0% of them is a typo. The difference between "few" and "none" is the only
    one this function is competent to judge, and widening it would put the
    proposer in the business of preferring rules that trade more.
    """
    try:
        node = parse(entry, resolve_feature=resolve_entry_feature)
    except ParseError:
        return []  # the parser's own message is the useful one
    problems: list[str] = []
    for call, op, number in _threshold_comparisons(node):
        bounds = span(call.key)
        if bounds is None:
            continue
        low, high = bounds
        if op in (">", ">=") and number > high:
            problems.append(_never(call, op, number, low, high, "above"))
        elif op in ("<", "<=") and number < low:
            problems.append(_never(call, op, number, low, high, "below"))
    return problems


_DECIMAL_HINT = (
    "Volatility features here are DECIMAL fractions -- iv_current() of 0.156 is "
    "15.6% vol, and a five-point term slope is 0.05, not 5. iv_rank() is the "
    "only feature on a 0..100 scale."
)


def _never(call: Call, op: str, number: float, low: float, high: float, side: str) -> str:
    # ``call.key`` rather than ``call``: ``Call.__str__`` drops the parentheses
    # on an arity-0 feature, and the catalogue the model read writes
    # ``term_slope()``. Two spellings of one name in the same conversation is a
    # small thing that costs a retry.
    name = call.key
    return (
        f"{name} is never {side} {number:g}: over the whole research window it "
        f"ranges {low:.4g} .. {high:.4g}, so `{name} {op} {number:g}` is false on "
        f"every session and the rule would open nothing. {_DECIMAL_HINT}"
    )


_MIRRORED_OP = {"<": ">", "<=": ">=", ">": "<", ">=": "<=", "==": "==", "!=": "!="}


def _threshold_comparisons(node: Expr) -> list[tuple[Call, str, float]]:
    """Every ``feature <op> number`` in the tree, with the mirrored form flipped.

    ``5 < term_slope()`` and ``term_slope() > 5`` are the same claim, and a check
    that caught one and not the other would be one a model could walk around
    without meaning to.
    """
    found: list[tuple[Call, str, float]] = []
    stack: list[Expr] = [node]
    while stack:
        current = stack.pop()
        if isinstance(current, Compare):
            left, right = current.left, current.right
            if isinstance(left, Call) and isinstance(right, Number):
                found.append((left, current.op, right.value))
            elif isinstance(right, Call) and isinstance(left, Number):
                found.append((right, _MIRRORED_OP[current.op], left.value))
            stack += [left, right]
        elif isinstance(current, Logic | Binary):
            stack += [current.left, current.right]
        elif isinstance(current, Not | Unary):
            stack.append(current.operand)
    return found
