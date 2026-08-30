"""Where hypotheses come from.

Two proposers ship with the MVP, behind one interface:

``AnthropicProposer``  asks Claude for a hypothesis, constrained by a JSON schema.
``HeuristicProposer``  a deterministic, offline search over a template library
                       plus mutation of the best strategy so far.

The second is not a toy. It makes the research loop runnable with no API key, in
CI, and reproducibly -- and it gives the LLM something to beat. A generator that
cannot outperform a template library is not adding research value, and without
the baseline that would be impossible to notice.

Note what neither proposer can do: emit code, choose its own universe, or reach
the test data. They return a small dictionary of fields; this module turns that
into a :class:`StrategySpec`, and the validator gets the final say.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Any, Protocol

from aqr.agent.prompts import PROPOSAL_SCHEMA, SYSTEM_PROMPT, build_user_prompt, prompt_hash
from aqr.dsl.expr import Compare, Logic, Not, ParseError, parse
from aqr.dsl.schema import ExitRules, Sizing, StopLoss, StrategySpec, TakeProfit, Universe
from aqr.dsl.validator import validate
from aqr.validation.params import slots

__all__ = [
    "AnthropicProposer",
    "DeepSeekProposer",
    "HeuristicProposer",
    "OpenAICompatProposer",
    "Proposal",
    "Proposer",
    "build_spec",
    "check_proposal",
    "spec_to_proposal_fields",
]

DEFAULT_MODEL = "claude-opus-5"


@dataclass(slots=True)
class Proposal:
    """A raw hypothesis, before it becomes a strategy."""

    fields: dict[str, Any]
    source: str
    model: str | None = None
    prompt_hash: str | None = None
    raw: str | None = None


class Proposer(Protocol):
    def propose(
        self,
        *,
        symbols: list[str],
        timeframe: str,
        memory: list[dict[str, Any]],
        instruction: str | None = None,
        parent: dict[str, Any] | None = None,
        timeframes: tuple[str, ...] | None = None,
    ) -> Proposal: ...


# --------------------------------------------------------------------------- #
# Proposal -> StrategySpec
# --------------------------------------------------------------------------- #


def build_spec(
    proposal: Proposal,
    symbols: list[str],
    timeframe: str = "1D",
    *,
    allowed_timeframes: tuple[str, ...] | None = None,
    risk_per_trade: float = 0.0075,
    max_positions: int = 3,
    parent_fingerprint: str | None = None,
) -> StrategySpec:
    """Turn a proposal into a validated spec.

    The universe, the risk budget and the position limit come from the run
    configuration, never from the proposal. A model that could choose its own
    universe could choose the one its idea happens to work on.

    The timeframe is the one degree of freedom a proposal *may* choose, and
    only within ``allowed_timeframes`` -- the same philosophy as the universe:
    the run configuration bounds the choice, the model makes it. An
    out-of-set choice raises, which the research loop records as a compile
    failure and feeds back to the model.
    """
    fields = proposal.fields
    allowed = allowed_timeframes or (timeframe,)
    chosen = str(fields.get("timeframe") or "").strip() or timeframe
    if chosen not in allowed:
        raise ValueError(f"timeframe must be one of {list(allowed)}, got {chosen!r}")
    timeframe = chosen
    ratio = float(fields.get("take_profit_r_multiple") or 0.0)
    take_profit = (
        TakeProfit(type="none", ratio=1.0)
        if ratio <= 0
        else TakeProfit(type="risk_reward", ratio=ratio)
    )
    regime = (fields.get("regime") or "").strip() or None
    signal_exit = (fields.get("signal_exit") or "").strip() or None
    direction = str(fields.get("direction") or "long")
    # Only carried for a market-neutral proposal: the spec rejects a short
    # leg on a one-directional strategy, and a model that fills the field
    # anyway should not have its whole proposal thrown away for it.
    short_entry = (fields.get("short_entry") or "").strip() or None
    if direction != "market_neutral":
        short_entry = None

    mode = str(fields.get("mode") or "signal")
    if mode == "portfolio":
        # ``hold`` is the hypothesis; the universe is the run configuration. A
        # model asking for the top 20 of a 4-name universe has made a reasonable
        # request against something it cannot see, so clamp rather than reject:
        # rejecting spends an iteration on an accident of the run.
        requested = int(fields.get("hold") or 10)
        spec = StrategySpec(
            name=str(fields["name"]).strip(),
            entry="",
            universe=Universe(symbols=tuple(s.upper() for s in symbols), timeframe=timeframe),
            sizing=Sizing(risk_per_trade=risk_per_trade, max_position_pct=0.25),
            regime=regime,
            mode="portfolio",
            rank_by=str(fields.get("rank_by") or "").strip() or None,
            screen=str(fields.get("screen") or "").strip() or None,
            hold=max(1, min(requested, len(symbols))),
            rebalance_every=max(1, int(fields.get("rebalance_every") or 5)),
            hypothesis=str(fields.get("hypothesis") or "").strip(),
            parent=parent_fingerprint,
        )
        validate(spec).raise_if_failed()
        return spec

    spec = StrategySpec(
        name=str(fields["name"]).strip(),
        entry=str(fields["entry"]).strip(),
        universe=Universe(symbols=tuple(s.upper() for s in symbols), timeframe=timeframe),
        exit=ExitRules(
            stop_loss=StopLoss(
                type="atr",
                multiplier=float(fields.get("stop_loss_atr_multiple") or 2.0),
                period=14,
            ),
            take_profit=take_profit,
            max_holding_bars=int(fields.get("max_holding_bars") or 20),
            signal_exit=signal_exit,
        ),
        sizing=Sizing(risk_per_trade=risk_per_trade, max_position_pct=0.25),
        direction=direction,  # type: ignore[arg-type]
        regime=regime,
        short_entry=short_entry,
        max_positions=max_positions,
        hypothesis=str(fields.get("hypothesis") or "").strip(),
        parent=parent_fingerprint,
    )
    validate(spec).raise_if_failed()
    return spec


# --------------------------------------------------------------------------- #
# Checking what a model sent back
# --------------------------------------------------------------------------- #

_NUMERIC_FIELDS = ("stop_loss_atr_multiple", "take_profit_r_multiple", "max_holding_bars")


def check_proposal(fields: dict[str, Any]) -> list[str]:
    """Problems with a raw proposal, phrased so a model can act on them.

    This runs *before* the strategy is built, and its output is fed straight back
    to the model as a repair instruction. That is why the messages name the
    offending value and, for an unknown feature, the closest real one: the point
    is not to log a complaint but to make the next attempt succeed.

    Returns an empty list when the proposal is usable.
    """
    problems: list[str] = []
    if not isinstance(fields, dict):
        return [f"expected a JSON object, got {type(fields).__name__}"]

    mode = fields.get("mode", "signal")
    if mode not in ("signal", "portfolio"):
        return [f"mode must be 'signal' or 'portfolio', got {mode!r}"]

    for required in ("name", "hypothesis"):
        value = fields.get(required)
        if not isinstance(value, str) or not value.strip():
            problems.append(f"field {required!r} is missing or empty")

    # The two forms are different machines, and a proposal carrying fields from
    # both has not chosen one. Ignoring the extra field would run a strategy
    # nobody proposed -- silently, since neither engine would complain.
    if mode == "portfolio":
        if not str(fields.get("rank_by") or "").strip():
            problems.append(
                "mode is 'portfolio' but rank_by is empty; without a ranking there "
                "is nothing to hold the top of"
            )
        if str(fields.get("entry") or "").strip():
            problems.append(
                "mode is 'portfolio' but entry is set; a ranked book is chosen by "
                "rank_by, not opened by a trigger. Leave entry empty."
            )
        for name in ("hold", "rebalance_every"):
            value = fields.get(name)
            if value is not None and (not isinstance(value, int) or value < 1):
                problems.append(f"field {name!r} must be a whole number >= 1, got {value!r}")
    else:
        if not str(fields.get("entry") or "").strip():
            problems.append("field 'entry' is missing or empty")
        for name in ("rank_by", "screen"):
            if str(fields.get(name) or "").strip():
                problems.append(
                    f"{name} is only meaningful in portfolio mode; set mode to "
                    f"'portfolio' or leave {name} empty"
                )

    direction = fields.get("direction", "long")
    if direction not in ("long", "short", "market_neutral"):
        problems.append(
            f"direction must be 'long', 'short' or 'market_neutral', got {direction!r}"
        )
    if direction == "market_neutral" and not str(fields.get("short_entry") or "").strip():
        problems.append(
            "direction is 'market_neutral' but short_entry is empty; a neutral "
            "strategy needs both legs, or it is a long-only strategy wearing the label"
        )

    tf = str(fields.get("timeframe") or "").strip()
    known = PROPOSAL_SCHEMA["properties"]["timeframe"]["enum"]
    if tf and tf not in known:
        problems.append(
            f"timeframe must be one of {known}, got {tf!r}; every parameter is "
            "counted in bars of the timeframe you choose"
        )

    for name in _NUMERIC_FIELDS:
        value = fields.get(name)
        if value is not None and not isinstance(value, (int, float)):
            problems.append(f"field {name!r} must be a number, got {value!r}")

    # Parse every expression against the real grammar and feature registry. A
    # near-miss here is the common failure, and it is entirely recoverable.
    for name in ("entry", "regime", "signal_exit", "short_entry", "screen"):
        source = fields.get(name)
        if not isinstance(source, str) or not source.strip():
            continue
        try:
            node = parse(source)
        except ParseError as exc:
            problems.append(f"{name} does not parse: {exc}")
            continue
        if not isinstance(node, (Compare, Logic, Not)):
            problems.append(
                f"{name} is an expression, not a condition: {source!r}. "
                f"Compare it to something, for example '{source} > 0'."
            )

    # The ranking is checked the other way round, and the message says what to
    # do rather than what went wrong: the repair turn gets one attempt.
    ranking = fields.get("rank_by")
    if isinstance(ranking, str) and ranking.strip():
        try:
            node = parse(ranking)
        except ParseError as exc:
            problems.append(f"rank_by does not parse: {exc}")
        else:
            if isinstance(node, (Compare, Logic, Not)):
                problems.append(
                    f"rank_by is a condition, not a number: {ranking!r} sorts the "
                    f"book into true and false. Rank by the quantity itself -- if "
                    f"you meant it as a filter, put it in 'screen' instead."
                )
    return problems


def _dead_rule_instruction(fields: dict[str, Any], problems: list[str]) -> str:
    """Ask for a fix to a rule that compiled and then fired on nothing.

    Deliberately narrow. The rule is not wrong in form, so the model is shown
    the offending expressions and the validator's complaint and asked to change
    the *condition* -- not handed the research memory again, which would invite
    it to abandon the idea instead of repairing it.
    """
    joined = "\n".join(f"- {p}" for p in problems)
    entry = fields.get("entry", "")
    regime = fields.get("regime", "") or "(none)"
    return (
        "Your hypothesis compiled correctly but its entry condition never "
        "becomes true on the data, so it produced no trades at all. That is a "
        "wasted experiment, not a result.\n\n"
        f"{joined}\n\n"
        "Here is what you proposed:\n"
        f"  entry:  {entry!r}\n"
        f"  regime: {regime!r}\n\n"
        "Keep the same mechanism and the same direction. Loosen or correct the "
        "condition so it can actually occur -- check that every threshold is "
        "inside the feature's real range, and that the regime filter does not "
        "contradict the entry. Return the complete JSON object again."
    )


def _repair_instruction(problems: list[str], raw: str) -> str:
    joined = "\n".join(f"- {p}" for p in problems)
    return (
        "Your previous answer was rejected. Fix these problems and return the "
        "complete JSON object again:\n"
        f"{joined}\n\n"
        f"Your previous answer was:\n{raw[:1500]}"
    )


# --------------------------------------------------------------------------- #
# Claude
# --------------------------------------------------------------------------- #


class AnthropicProposer:
    """Asks Claude for a hypothesis, constrained by :data:`PROPOSAL_SCHEMA`.

    The schema is the containment boundary. The model cannot return code because
    there is no field to put code in; it cannot invent a feature because the
    expression it returns is parsed against the registry before anything runs.
    Structured output turns "please behave" into "cannot misbehave".
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
                "AnthropicProposer needs the 'llm' extra: uv sync --extra llm"
            ) from exc
        self._client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort

    def propose(
        self,
        *,
        symbols: list[str],
        timeframe: str,
        memory: list[dict[str, Any]],
        instruction: str | None = None,
        parent: dict[str, Any] | None = None,
        timeframes: tuple[str, ...] | None = None,
    ) -> Proposal:
        user = build_user_prompt(
            symbols=symbols,
            timeframe=timeframe,
            memory=memory,
            instruction=instruction,
            parent=parent,
            timeframes=timeframes,
        )
        # The SDK types these as TypedDicts. Importing them would pull anthropic
        # to module scope and break the optional-extra design, so the widening
        # happens here, at the boundary, and nowhere else.
        messages: Any = [{"role": "user", "content": user}]
        thinking: Any = {"type": "adaptive"}
        output_config: Any = {
            "effort": self.effort,
            "format": {"type": "json_schema", "schema": PROPOSAL_SCHEMA},
        }
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            messages=messages,
            thinking=thinking,
            output_config=output_config,
        )
        if getattr(response, "stop_reason", None) == "refusal":
            details = getattr(response, "stop_details", None)
            raise RuntimeError(f"model declined to answer: {details}")

        text = "".join(block.text for block in response.content if block.type == "text")
        try:
            fields = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"model returned non-JSON output: {text[:400]}") from exc
        return Proposal(
            fields=fields,
            source="anthropic",
            model=self.model,
            prompt_hash=prompt_hash(SYSTEM_PROMPT, user),
            raw=text,
        )


    def repair(self, *, proposal: Proposal, problems: list[str]) -> Proposal:
        """One more turn for a rule that compiled and fired on nothing."""
        user = _dead_rule_instruction(proposal.fields, problems)
        messages: Any = [{"role": "user", "content": user}]
        thinking: Any = {"type": "adaptive"}
        output_config: Any = {
            "effort": self.effort,
            "format": {"type": "json_schema", "schema": PROPOSAL_SCHEMA},
        }
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            messages=messages,
            thinking=thinking,
            output_config=output_config,
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        try:
            fields = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"repair returned non-JSON output: {text[:200]}") from exc
        return Proposal(
            fields=fields,
            source="anthropic",
            model=self.model,
            prompt_hash=prompt_hash(SYSTEM_PROMPT, user),
            raw=text,
        )


# --------------------------------------------------------------------------- #
# OpenAI-compatible endpoints (DeepSeek and anything speaking the same protocol)
# --------------------------------------------------------------------------- #

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"


class OpenAICompatProposer:
    """Any chat endpoint that speaks the OpenAI protocol.

    The important difference from :class:`AnthropicProposer` is what the API can
    guarantee. Anthropic's ``json_schema`` mode constrains the *shape* of the
    output, so a malformed field is impossible. DeepSeek's JSON mode guarantees
    only that the bytes parse -- the model can still omit a field or write
    ``ema200`` where ``ema(200)`` was meant.

    So the schema is rendered into the prompt, the reply is checked against the
    real grammar, and a rejected proposal gets exactly one repair attempt with
    the specific errors attached. One, not five: a model that cannot fill in a
    ten-field form after being told precisely what it got wrong is not about to
    produce a good hypothesis on the third try, and the loop's next iteration is
    a cheaper place to spend those tokens.
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
        # ``client`` is an injection point for tests. Everything about the repair
        # loop -- the part most worth testing -- is independent of the network,
        # and requiring an API key to exercise it would mean it never got tested.
        if client is not None:
            self._client = client
        else:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - depends on optional extra
                raise RuntimeError(
                    "OpenAICompatProposer needs the 'llm' extra: uv pip install -e '.[llm]'"
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
        """The shared system prompt plus the schema the endpoint cannot enforce."""
        return (
            f"{SYSTEM_PROMPT}\n"
            "Return a single JSON object and nothing else -- no prose, no code "
            "fence. It must match this JSON schema exactly:\n"
            f"{json.dumps(PROPOSAL_SCHEMA, indent=2)}"
        )

    def propose(
        self,
        *,
        symbols: list[str],
        timeframe: str,
        memory: list[dict[str, Any]],
        instruction: str | None = None,
        parent: dict[str, Any] | None = None,
        timeframes: tuple[str, ...] | None = None,
    ) -> Proposal:
        system = self._system_prompt()
        user = build_user_prompt(
            symbols=symbols,
            timeframe=timeframe,
            memory=memory,
            instruction=instruction,
            parent=parent,
            timeframes=timeframes,
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
                problems = check_proposal(fields)
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

        raise ValueError(f"{self.provider} produced an unusable proposal: {last_error}")

    def repair(self, *, proposal: Proposal, problems: list[str]) -> Proposal:
        """One more turn, carrying the reason the rule was unusable.

        Raises rather than returning something unchecked: the research loop
        treats a raised repair as "no repair" and evaluates the original, which
        keeps the real rejection reason attached to the real attempt.
        """
        system = self._system_prompt()
        user = _dead_rule_instruction(proposal.fields, problems)
        raw = self._call(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
        )
        try:
            fields = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{self.provider} repair returned non-JSON: {raw[:200]}") from exc
        problems_now = check_proposal(fields)
        if problems_now:
            raise ValueError(f"{self.provider} repair is still unusable: {'; '.join(problems_now)}")
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


def DeepSeekProposer(model: str = DEEPSEEK_MODEL, **kwargs: Any) -> OpenAICompatProposer:
    """DeepSeek preset. ``deepseek-chat`` for the loop, ``deepseek-reasoner`` to think harder."""
    kwargs.setdefault("base_url", DEEPSEEK_BASE_URL)
    kwargs.setdefault("provider", "deepseek")
    return OpenAICompatProposer(model, **kwargs)


# --------------------------------------------------------------------------- #
# Offline
# --------------------------------------------------------------------------- #

# Each template is a *mechanism* with a stated rationale, not a random indicator
# combination. The offline loop's job is to explore this space systematically;
# the LLM's job is to propose mechanisms that are not in it.
_TEMPLATES: list[dict[str, Any]] = [
    {
        "name": "trend_pullback",
        "hypothesis": (
            "In an established uptrend, a shallow pullback is profit-taking rather than "
            "a change of trend, and buyers who missed the move supply demand at the mean."
        ),
        "regime": "close > ema(200)",
        "entry": "close <= ema(20) * 1.01 and rsi(14) > 40",
        "stop_loss_atr_multiple": 2.0,
        "take_profit_r_multiple": 2.0,
        "max_holding_bars": 20,
    },
    {
        "name": "vol_contraction_breakout",
        "hypothesis": (
            "Volatility clusters. A compression in realised volatility resolves into "
            "an expansion, and the breakout direction tends to continue for several bars."
        ),
        "regime": "vol_pct(20, 252) < 0.3",
        "entry": "close > highest(20) * 0.999 and rvol(20) > 1.5",
        "stop_loss_atr_multiple": 2.5,
        "take_profit_r_multiple": 2.5,
        "max_holding_bars": 15,
    },
    {
        "name": "momentum_continuation",
        "hypothesis": (
            "Cross-sectional and time-series momentum persist at horizons of one to "
            "twelve months, a premium usually attributed to under-reaction to news."
        ),
        "regime": "close > ema(200) and adx(14) > 25",
        "entry": "roc(60) > 0.08 and close > ema(50)",
        "stop_loss_atr_multiple": 3.0,
        "take_profit_r_multiple": 0.0,
        "max_holding_bars": 60,
    },
    {
        "name": "oversold_reversion",
        "hypothesis": (
            "Short-horizon selling in an otherwise healthy market overshoots, because "
            "forced sellers are price-insensitive and liquidity providers demand a premium."
        ),
        "regime": "close > ema(200)",
        "entry": "rsi(3) < 15 and close > ema(200)",
        "stop_loss_atr_multiple": 2.0,
        "take_profit_r_multiple": 1.5,
        "max_holding_bars": 8,
    },
    {
        "name": "band_reversion",
        "hypothesis": (
            "A close below the lower Bollinger band in a non-trending market is a "
            "statistical outlier that mean-reverts more often than it continues."
        ),
        "regime": "adx(14) < 20",
        "entry": "close < bb_lower(20, 2) and rsi(14) < 30",
        "stop_loss_atr_multiple": 2.0,
        "take_profit_r_multiple": 1.5,
        "max_holding_bars": 10,
    },
    {
        "name": "volume_thrust",
        "hypothesis": (
            "An unusually heavy up-bar marks institutional accumulation, which is "
            "executed over days rather than instantly and so leaves a short-lived drift."
        ),
        "regime": "close > ema(100)",
        "entry": "rvol(20) > 2.0 and close > open and roc(1) > 0.01",
        "stop_loss_atr_multiple": 2.0,
        "take_profit_r_multiple": 2.0,
        "max_holding_bars": 10,
    },
    {
        "name": "range_breakout",
        "hypothesis": (
            "A breach of a multi-week high resolves the uncertainty that kept the range "
            "intact; stops above the range add fuel to the initial move."
        ),
        "regime": "close > ema(200)",
        "entry": "close > highest(50) * 0.998 and adx(14) > 20",
        "stop_loss_atr_multiple": 2.5,
        "take_profit_r_multiple": 3.0,
        "max_holding_bars": 30,
    },
    {
        "name": "gap_fade",
        "hypothesis": (
            "A large opening gap with no follow-through overshoots the information that "
            "caused it, and the excess is retraced within days."
        ),
        "regime": "vol_pct(20, 252) > 0.5",
        "entry": "roc(1) < -0.04 and close > ema(200) and rvol(20) > 1.5",
        "stop_loss_atr_multiple": 2.5,
        "take_profit_r_multiple": 1.5,
        "max_holding_bars": 6,
    },
    # --- short side ---------------------------------------------------------
    # Shorts are not longs with the sign flipped. The asymmetries are real:
    # losses are unbounded, borrow costs accrue daily, and the long-run drift of
    # equities is against the position. A short therefore needs a sharper
    # mechanism than "the mirror image worked", which is why each of these names
    # a specific reason the *downside* should follow through.
    {
        "name": "downtrend_rally_fade",
        "hypothesis": (
            "In a confirmed downtrend a rally into resistance is short covering and "
            "bargain hunting, not a change of trend. Supply overhead from holders "
            "waiting to get out at breakeven caps the bounce."
        ),
        "direction": "short",
        "regime": "close < ema(200) and adx(14) > 20",
        "entry": "close >= ema(20) * 0.99 and rsi(14) < 60",
        "stop_loss_atr_multiple": 2.0,
        "take_profit_r_multiple": 2.0,
        "max_holding_bars": 15,
    },
    {
        "name": "breakdown_continuation",
        "hypothesis": (
            "A break of a multi-week low forces liquidation from stop orders and "
            "risk-parity deleveraging, both of which are price-insensitive sellers "
            "whose flow continues after the initial break."
        ),
        "direction": "short",
        "regime": "close < ema(100)",
        "entry": "close < lowest(50) * 1.002 and rvol(20) > 1.5",
        "stop_loss_atr_multiple": 2.5,
        "take_profit_r_multiple": 2.5,
        "max_holding_bars": 20,
    },
    {
        "name": "blowoff_exhaustion",
        "hypothesis": (
            "A vertical move on extreme volume far above the mean marks late, "
            "momentum-chasing buyers with no remaining marginal bid behind them. "
            "The excess mean-reverts once that flow stops."
        ),
        "direction": "short",
        "regime": "vol_pct(20, 252) > 0.6",
        "entry": "zscore(20) > 2.5 and rvol(20) > 2.5 and rsi(5) > 85",
        "stop_loss_atr_multiple": 2.0,
        "take_profit_r_multiple": 1.5,
        "max_holding_bars": 8,
    },
    {
        "name": "failed_breakout_short",
        "hypothesis": (
            "A break above a multi-week high that closes back inside the range traps "
            "the buyers who chased it. Their stops sit just below, and triggering "
            "them supplies the move down."
        ),
        "direction": "short",
        "regime": "adx(14) < 25",
        "entry": "highest(20) > close * 1.02 and close < ema(20) and roc(3) < -0.01",
        "stop_loss_atr_multiple": 2.0,
        "take_profit_r_multiple": 2.0,
        "max_holding_bars": 10,
    },
]


class HeuristicProposer:
    """Deterministic offline search: templates first, then mutation of the best.

    Phase one walks the template library so every mechanism gets one honest test.
    Phase two perturbs the best strategy found so far -- the evolution loop of
    architecture section 25, with the important constraint that a mutation is
    only ever accepted on out-of-sample evidence, which is enforced by the
    evaluator rather than here.

    The heuristic proposer does not explore across granularities: it stamps
    every proposal with the run's default timeframe and leaves cross-timeframe
    choice to the model proposers. Note the template parameters are daily-scale
    semantics -- ema(200), roc(60), a 20-bar holding period -- so on a non-1D
    default they measure a *shorter* horizon than their hypotheses describe.
    """

    def __init__(self, seed: int = 20260826) -> None:
        self._rng = random.Random(seed)
        self._used: set[str] = set()
        self._lineage: dict[str, int] = {}

    def propose(
        self,
        *,
        symbols: list[str],
        timeframe: str,
        memory: list[dict[str, Any]],
        instruction: str | None = None,
        parent: dict[str, Any] | None = None,
        timeframes: tuple[str, ...] | None = None,
    ) -> Proposal:
        if parent:
            return self._mutate(parent)
        for template in _TEMPLATES:
            if template["name"] not in self._used:
                self._used.add(template["name"])
                fields = dict(template)
                fields["name"] = f"{template['name']}_v1"
                fields.setdefault("direction", "long")
                fields["signal_exit"] = ""
                fields["expected_trades_per_year"] = 20
                fields["timeframe"] = timeframe
                return Proposal(fields=fields, source="heuristic")
        # Library exhausted: recombine a regime with a different entry, which is
        # the cheapest way to ask a genuinely new question of the same data.
        # Recombine only within one direction: a long entry under a downtrend
        # regime is not a new hypothesis, it is a contradiction that will simply
        # never fire.
        direction = self._rng.choice(["long", "short"])
        pool = [t for t in _TEMPLATES if t.get("direction", "long") == direction]
        left, right = self._rng.sample(pool, 2)
        fields = dict(right)
        fields["name"] = f"{right['name']}_in_{left['name']}_regime"
        fields["regime"] = left["regime"]
        fields["hypothesis"] = (
            f"{right['hypothesis']} Tested under a different market filter: "
            f"{left['regime']}."
        )
        fields["direction"] = direction
        fields["signal_exit"] = ""
        fields["expected_trades_per_year"] = 15
        fields["timeframe"] = timeframe
        return Proposal(fields=fields, source="heuristic")

    def _mutate(self, parent: dict[str, Any]) -> Proposal:
        """One parameter changed, everything else held fixed.

        Changing one thing at a time is what makes the result attributable. A
        mutation that alters three parameters and improves the Sharpe has taught
        us nothing about which of the three mattered.
        """
        fields = dict(parent)
        knobs = ["stop_loss_atr_multiple", "take_profit_r_multiple", "max_holding_bars"]
        knob = self._rng.choice(knobs)
        factor = self._rng.choice([0.75, 0.85, 1.15, 1.3])
        current = float(fields.get(knob) or 0.0)
        if current <= 0:
            knob, current = "stop_loss_atr_multiple", float(
                fields.get("stop_loss_atr_multiple") or 2.0
            )
        updated = current * factor
        fields[knob] = int(round(updated)) if knob == "max_holding_bars" else round(updated, 2)
        base = str(parent.get("name", "strategy")).rsplit("_v", 1)[0]
        # A monotonic per-lineage counter, not parent_version + 1: two mutations
        # of the same parent would otherwise both be "_v2" and be impossible to
        # tell apart in the experiment log.
        self._lineage[base] = self._lineage.get(base, 1) + 1
        fields["name"] = f"{base}_v{self._lineage[base]}"
        fields["hypothesis"] = (
            f"{parent.get('hypothesis', '')} Variant: {knob} {current:g} -> {fields[knob]}."
        )
        return Proposal(fields=fields, source="heuristic-mutation")


def spec_to_proposal_fields(spec: StrategySpec) -> dict[str, Any]:
    """Render a spec back into proposal fields, for mutation and for the prompt.

    Research memory is built from these. A portfolio strategy that could not be
    described back to the model would be invisible to the search that produced
    it, and the model would keep re-proposing it.
    """
    if spec.mode == "portfolio":
        return {
            "name": spec.name,
            "hypothesis": spec.hypothesis,
            "mode": "portfolio",
            "direction": spec.direction,
            "timeframe": spec.universe.timeframe,
            "regime": spec.regime or "",
            "rank_by": spec.rank_by or "",
            "screen": spec.screen or "",
            "hold": spec.hold,
            "rebalance_every": spec.rebalance_every,
            "parameters": [str(s) for s in slots(spec)],
        }
    return {
        "name": spec.name,
        "hypothesis": spec.hypothesis,
        "mode": "signal",
        "direction": spec.direction,
        "timeframe": spec.universe.timeframe,
        "regime": spec.regime or "",
        "entry": spec.entry,
        "signal_exit": spec.exit.signal_exit or "",
        "stop_loss_atr_multiple": spec.exit.stop_loss.multiplier,
        "take_profit_r_multiple": (
            0.0 if spec.exit.take_profit.type == "none" else spec.exit.take_profit.ratio
        ),
        "max_holding_bars": spec.exit.max_holding_bars,
        "parameters": [str(s) for s in slots(spec)],
    }
