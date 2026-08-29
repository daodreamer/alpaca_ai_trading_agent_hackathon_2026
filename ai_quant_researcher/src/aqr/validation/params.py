"""Addressing and rewriting a strategy's free parameters.

Both parameter search (section 13) and perturbation testing (section 14.1) need
the same primitive: name every knob a strategy has, then set one to a different
value and get a new, valid strategy back.

The knobs are of two kinds:

* *Structured* fields -- ``exit.stop_loss.multiplier``, ``sizing.risk_per_trade``.
* *Literals inside expressions* -- the ``200`` in ``close > ema(200)``.

The second kind is addressed positionally: ``entry#0`` is the first numeric
literal in the entry expression. Rewriting goes through the tokenizer, never
through string replacement, so ``ema(20)`` in ``close <= ema(20) * 1.02`` is
distinguishable from the ``20`` that might appear elsewhere in the same line.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from aqr.dsl.expr import tokenize
from aqr.dsl.schema import StrategySpec

__all__ = ["Slot", "apply_params", "get_param", "neighbours", "set_param", "slots"]

_STRUCTURED: dict[str, tuple[str, ...]] = {
    # Portfolio mode. Without these a portfolio spec has no perturbable
    # parameters at all, every neighbour comes back identical, and
    # ``parameter_stability`` -- 15% of the score -- pays out full marks for
    # being unperturbable. That is worse than not measuring it.
    "hold": ("hold",),
    "rebalance_every": ("rebalance_every",),
    "exit.stop_loss.multiplier": ("exit", "stop_loss", "multiplier"),
    "exit.stop_loss.period": ("exit", "stop_loss", "period"),
    "exit.take_profit.ratio": ("exit", "take_profit", "ratio"),
    "exit.max_holding_bars": ("exit", "max_holding_bars"),
    "sizing.risk_per_trade": ("sizing", "risk_per_trade"),
    "sizing.max_position_pct": ("sizing", "max_position_pct"),
}

_INTEGER_SLOTS = {
    "exit.stop_loss.period",
    "exit.max_holding_bars",
    "hold",
    "rebalance_every",
}

_EXPRESSION_FIELDS = ("entry", "regime", "signal_exit", "rank_by", "screen")

# Which knobs each engine actually consults.
#
# ``run_portfolio`` reads ``rank_by``, ``screen``, ``hold`` and
# ``rebalance_every`` and nothing else: it never touches ``exit`` or ``sizing``,
# because the book is chosen by rank and turned over on the rebalance, so no
# stop fires and no position is sized on its own. The signal engine is the
# mirror image -- it has no ranking and no rebalance clock.
#
# Perturbing a knob the engine does not read produces a bit-identical backtest,
# and ``parameter_stability`` -- 15% of the score -- reads identical as
# *stable*. Six of a portfolio spec's ten slots were dead that way, so two
# thirds of its stability mark was paid out for being unperturbable. The
# comment on ``_STRUCTURED`` above already names this failure mode; leaving the
# dead knobs in was the other half of it.
_PORTFOLIO_STRUCTURED = frozenset({"hold", "rebalance_every"})
_PORTFOLIO_EXPRESSIONS = frozenset({"rank_by", "screen"})
_SIGNAL_STRUCTURED = frozenset(_STRUCTURED) - _PORTFOLIO_STRUCTURED
_SIGNAL_EXPRESSIONS = frozenset({"entry", "regime", "signal_exit"})


def _readable(spec: StrategySpec) -> tuple[frozenset[str], frozenset[str]]:
    """The structured paths and expression fields this spec's engine reads."""
    if spec.mode == "portfolio":
        return _PORTFOLIO_STRUCTURED, _PORTFOLIO_EXPRESSIONS
    return _SIGNAL_STRUCTURED, _SIGNAL_EXPRESSIONS


@dataclass(frozen=True, slots=True)
class Slot:
    """One addressable parameter."""

    path: str
    value: float
    kind: str  # "structured" | "literal"
    integral: bool

    def __str__(self) -> str:
        rendered = int(self.value) if self.integral else self.value
        return f"{self.path}={rendered}"


def _expression_of(spec: StrategySpec, field: str) -> str | None:
    if field == "entry":
        return spec.entry
    if field == "regime":
        return spec.regime
    if field == "signal_exit":
        return spec.exit.signal_exit
    if field == "rank_by":
        return spec.rank_by
    if field == "screen":
        return spec.screen
    raise KeyError(field)


def slots(spec: StrategySpec) -> list[Slot]:
    """Every parameter of ``spec`` its own engine reads, in a stable order.

    Restricted to what the mode consults, because a slot the engine ignores is
    not a parameter of the strategy -- it is a field that happens to be on the
    dataclass. Perturbing one answers "is the result stable" with "the result is
    the same object", which is not the same claim.
    """
    found: list[Slot] = []
    structured, expressions = _readable(spec)
    for path, _ in _STRUCTURED.items():
        if path not in structured:
            continue
        value = get_param(spec, path)
        if value is None:
            continue
        found.append(
            Slot(path=path, value=float(value), kind="structured", integral=path in _INTEGER_SLOTS)
        )
    for field in _EXPRESSION_FIELDS:
        if field not in expressions:
            continue
        source = _expression_of(spec, field)
        if not source:
            continue
        numbers = [t for t in tokenize(source) if t.kind == "number"]
        for i, token in enumerate(numbers):
            value = float(token.text)
            found.append(
                Slot(
                    path=f"{field}#{i}",
                    value=value,
                    kind="literal",
                    integral=value.is_integer() and "." not in token.text,
                )
            )
    return found


def get_param(spec: StrategySpec, path: str) -> float | None:
    """Current value at ``path``, or ``None`` if the strategy has no such knob."""
    if "#" in path:
        field, _, index = path.partition("#")
        source = _expression_of(spec, field)
        if not source:
            return None
        numbers = [t for t in tokenize(source) if t.kind == "number"]
        position = int(index)
        if position >= len(numbers):
            return None
        return float(numbers[position].text)

    if path not in _STRUCTURED:
        raise KeyError(f"unknown parameter {path!r}; known: {sorted(_STRUCTURED)}")
    if path in ("hold", "rebalance_every") and spec.mode != "portfolio":
        return None
    if path == "exit.stop_loss.period" and spec.exit.stop_loss.type != "atr":
        return None
    if path == "exit.take_profit.ratio" and spec.exit.take_profit.type == "none":
        return None
    node: Any = spec
    for part in _STRUCTURED[path]:
        node = getattr(node, part)
    return float(node)


def set_param(spec: StrategySpec, path: str, value: float) -> StrategySpec:
    """A copy of ``spec`` with one parameter changed.

    The result is constructed through the normal spec constructor, so an
    out-of-range perturbation raises here rather than producing a strategy that
    is invalid in a way only the backtester would notice.
    """
    if "#" in path:
        field, _, index = path.partition("#")
        source = _expression_of(spec, field)
        if not source:
            raise KeyError(f"{spec.name} has no {field} expression")
        rewritten = _rewrite_literal(source, int(index), value)
        if field == "entry":
            return replace(spec, entry=rewritten)
        if field == "regime":
            return replace(spec, regime=rewritten)
        if field == "rank_by":
            return replace(spec, rank_by=rewritten)
        if field == "screen":
            return replace(spec, screen=rewritten)
        return replace(spec, exit=replace(spec.exit, signal_exit=rewritten))

    if path not in _STRUCTURED:
        raise KeyError(f"unknown parameter {path!r}; known: {sorted(_STRUCTURED)}")

    # Dispatched explicitly rather than through **{field: value}. Dynamic kwargs
    # would let a typo in _STRUCTURED reach the dataclass as an unexpected
    # keyword at run time; written out, the type checker catches it here.
    number = float(value)
    whole = int(round(value))

    if path == "hold":
        return replace(spec, hold=whole)
    if path == "rebalance_every":
        return replace(spec, rebalance_every=whole)
    if path == "exit.stop_loss.multiplier":
        stop = replace(spec.exit.stop_loss, multiplier=number)
        return replace(spec, exit=replace(spec.exit, stop_loss=stop))
    if path == "exit.stop_loss.period":
        stop = replace(spec.exit.stop_loss, period=whole)
        return replace(spec, exit=replace(spec.exit, stop_loss=stop))
    if path == "exit.take_profit.ratio":
        target = replace(spec.exit.take_profit, ratio=number)
        return replace(spec, exit=replace(spec.exit, take_profit=target))
    if path == "exit.max_holding_bars":
        return replace(spec, exit=replace(spec.exit, max_holding_bars=whole))
    if path == "sizing.risk_per_trade":
        return replace(spec, sizing=replace(spec.sizing, risk_per_trade=number))
    if path == "sizing.max_position_pct":
        return replace(spec, sizing=replace(spec.sizing, max_position_pct=number))
    raise KeyError(f"parameter {path!r} is listed but not settable")


def apply_params(spec: StrategySpec, params: dict[str, float]) -> StrategySpec:
    """Apply several parameter changes at once, in sorted path order.

    Sorting makes the result independent of dict iteration order, which is what
    lets a grid search be reproducible from its recorded parameter map.
    """
    out = spec
    for path in sorted(params):
        out = set_param(out, path, params[path])
    return out


def _rewrite_literal(source: str, index: int, value: float) -> str:
    """Replace the ``index``-th numeric literal, preserving everything else."""
    numbers = [t for t in tokenize(source) if t.kind == "number"]
    if index >= len(numbers):
        raise IndexError(f"{source!r} has {len(numbers)} literals; no #{index}")
    token = numbers[index]
    integral = value.is_integer() and "." not in token.text
    rendered = str(int(round(value))) if integral else f"{value:g}"
    return source[: token.pos] + rendered + source[token.pos + len(token.text) :]


def neighbours(spec: StrategySpec, slot: Slot, factors: tuple[float, ...]) -> list[StrategySpec]:
    """Variants of ``spec`` with one slot scaled by each factor.

    Variants that fail construction -- a multiplier driven to zero, a period
    below one -- are dropped rather than raised, because a perturbation test
    that dies on its own edge case tells us nothing about the strategy.
    """
    out: list[StrategySpec] = []
    seen: set[str] = set()
    for factor in factors:
        candidate = slot.value * factor
        if slot.integral:
            candidate = float(max(1, int(round(candidate))))
        if candidate == slot.value:
            continue
        try:
            variant = set_param(spec, slot.path, candidate)
        except (ValueError, IndexError, KeyError):
            continue
        fingerprint = variant.fingerprint()
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        out.append(variant)
    return out
