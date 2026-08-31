"""A tiny expression language: tokenizer, parser, evaluator.

The architecture is explicit that an LLM must never hand us Python to execute.
So it hands us this instead:

    close > ema(200)
    close <= ema(20) * 1.01
    rsi(14) > 40 and rvol(20) > 1.2

Grammar (precedence climbing, lowest first)::

    or_expr   := and_expr ("or" and_expr)*
    and_expr  := not_expr ("and" not_expr)*
    not_expr  := "not" not_expr | comparison
    comparison:= sum ( ("<"|"<="|">"|">="|"=="|"!=") sum )?
    sum       := product (("+"|"-") product)*
    product   := unary (("*"|"/") unary)*
    unary     := "-" unary | atom
    atom      := NUMBER | NAME | NAME "(" [args] ")" | "(" or_expr ")"

There is no attribute access, no indexing, no assignment, no lambda and no name
that is not in the feature registry. That is the whole security argument: the
grammar cannot express anything dangerous, so no sandbox is required.

Evaluation is vectorised over the whole bar series at once and yields either a
float array (arithmetic) or a boolean mask (comparisons/logic). ``NaN`` operands
propagate to ``False`` in comparisons, which is what keeps a strategy from
trading during its own warm-up.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from aqr.features.engine import FeatureKey
from aqr.features.registry import resolve

__all__ = [
    "Binary",
    "Call",
    "Compare",
    "Expr",
    "FeatureLookup",
    "FeatureSource",
    "Logic",
    "Not",
    "Number",
    "ParseError",
    "Unary",
    "evaluate",
    "feature_keys",
    "parse",
]


Values = NDArray[Any]
"""A float array (arithmetic) or a boolean mask (comparisons and logic)."""


class FeatureLookup(Protocol):
    """What `parse` needs from a feature-table entry: its arity, to reject too
    many arguments before a backtest ever runs. `aqr.features.registry.FeatureSpec`
    satisfies this structurally, and so does
    `aqr.options.features.OptionFeatureSpec` -- neither module imports the other.

    ``arity`` is a read-only property on this Protocol rather than a plain
    attribute: a frozen dataclass field (both concrete implementations are
    frozen) is get-only from the outside, and a Protocol declared with a plain
    attribute demands a *settable* one, which would reject both of them.
    """

    @property
    def arity(self) -> int: ...


class FeatureSource(Protocol):
    """What `evaluate` needs to read a feature: enough for a bar-only
    `aqr.features.engine.FeatureFrame`, and enough for anything that layers more
    vocabulary on top of one -- `aqr.options.features.OptionFeatureFrame` mixes
    option features into the same entry expression this way (specs/10 D6).
    Structural, not a base class, so this module stays ignorant of both."""

    def __len__(self) -> int: ...
    def get(self, key: FeatureKey) -> Values: ...


class ParseError(ValueError):
    """Raised for anything the grammar cannot accept. Message is LLM-facing."""


# --------------------------------------------------------------------------- #
# Tokenizer
# --------------------------------------------------------------------------- #

_TOKEN = re.compile(
    r"""
    (?P<space>\s+)
  | (?P<number>\d+\.\d+|\.\d+|\d+)
  | (?P<name>[A-Za-z_][A-Za-z_0-9]*)
  | (?P<op><=|>=|==|!=|[<>+\-*/(),])
    """,
    re.VERBOSE,
)

_KEYWORDS = {"and", "or", "not"}


@dataclass(frozen=True, slots=True)
class Token:
    kind: str  # "number" | "name" | "op" | "kw" | "end"
    text: str
    pos: int


def tokenize(source: str) -> list[Token]:
    tokens: list[Token] = []
    i = 0
    while i < len(source):
        match = _TOKEN.match(source, i)
        if match is None:
            raise ParseError(f"unexpected character {source[i]!r} at position {i} in {source!r}")
        i = match.end()
        kind = match.lastgroup
        assert kind is not None
        if kind == "space":
            continue
        text = match.group()
        if kind == "name" and text in _KEYWORDS:
            kind = "kw"
        tokens.append(Token(kind, text, match.start()))
    tokens.append(Token("end", "", len(source)))
    return tokens


# --------------------------------------------------------------------------- #
# AST
# --------------------------------------------------------------------------- #


class Expr:
    """Base node. Subclasses are frozen dataclasses so specs stay hashable."""

    def __str__(self) -> str:  # pragma: no cover - trivial
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class Number(Expr):
    value: float

    def __str__(self) -> str:
        return str(int(self.value)) if float(self.value).is_integer() else str(self.value)


@dataclass(frozen=True, slots=True)
class Call(Expr):
    """A feature reference. ``close`` parses to ``Call("close", ())``."""

    name: str
    args: tuple[float, ...]

    def __str__(self) -> str:
        if not self.args:
            return self.name
        rendered = ", ".join(str(Number(a)) for a in self.args)
        return f"{self.name}({rendered})"

    @property
    def key(self) -> FeatureKey:
        return FeatureKey(self.name, self.args)


@dataclass(frozen=True, slots=True)
class Unary(Expr):
    op: str
    operand: Expr

    def __str__(self) -> str:
        return f"-{self.operand}"


@dataclass(frozen=True, slots=True)
class Binary(Expr):
    op: str
    left: Expr
    right: Expr

    def __str__(self) -> str:
        return f"{self.left} {self.op} {self.right}"


@dataclass(frozen=True, slots=True)
class Compare(Expr):
    op: str
    left: Expr
    right: Expr

    def __str__(self) -> str:
        return f"{self.left} {self.op} {self.right}"


@dataclass(frozen=True, slots=True)
class Logic(Expr):
    op: str  # "and" | "or"
    left: Expr
    right: Expr

    def __str__(self) -> str:
        return f"({self.left}) {self.op} ({self.right})"


@dataclass(frozen=True, slots=True)
class Not(Expr):
    operand: Expr

    def __str__(self) -> str:
        return f"not ({self.operand})"


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #

_COMPARISONS = {"<", "<=", ">", ">=", "==", "!="}


class _Parser:
    def __init__(
        self, source: str, *, resolve_feature: Callable[[str], FeatureLookup] = resolve
    ) -> None:
        self.source = source
        self.tokens = tokenize(source)
        self.i = 0
        self._resolve_feature = resolve_feature

    @property
    def current(self) -> Token:
        return self.tokens[self.i]

    def _advance(self) -> Token:
        token = self.tokens[self.i]
        self.i += 1
        return token

    def _accept(self, kind: str, *texts: str) -> Token | None:
        token = self.current
        if token.kind == kind and (not texts or token.text in texts):
            return self._advance()
        return None

    def _expect(self, kind: str, *texts: str) -> Token:
        token = self._accept(kind, *texts)
        if token is None:
            wanted = " or ".join(repr(t) for t in texts) or kind
            raise ParseError(
                f"expected {wanted} at position {self.current.pos} "
                f"in {self.source!r}, got {self.current.text!r}"
            )
        return token

    def parse(self) -> Expr:
        node = self._or()
        if self.current.kind != "end":
            raise ParseError(
                f"unexpected {self.current.text!r} at position {self.current.pos} "
                f"in {self.source!r}"
            )
        return node

    def _or(self) -> Expr:
        node = self._and()
        while self._accept("kw", "or"):
            node = Logic("or", node, self._and())
        return node

    def _and(self) -> Expr:
        node = self._not()
        while self._accept("kw", "and"):
            node = Logic("and", node, self._not())
        return node

    def _not(self) -> Expr:
        if self._accept("kw", "not"):
            return Not(self._not())
        return self._comparison()

    def _comparison(self) -> Expr:
        node = self._sum()
        token = self.current
        if token.kind == "op" and token.text in _COMPARISONS:
            self._advance()
            return Compare(token.text, node, self._sum())
        return node

    def _sum(self) -> Expr:
        node = self._product()
        while True:
            token = self._accept("op", "+", "-")
            if token is None:
                return node
            node = Binary(token.text, node, self._product())

    def _product(self) -> Expr:
        node = self._unary()
        while True:
            token = self._accept("op", "*", "/")
            if token is None:
                return node
            node = Binary(token.text, node, self._unary())

    def _unary(self) -> Expr:
        if self._accept("op", "-"):
            return Unary("-", self._unary())
        return self._atom()

    def _atom(self) -> Expr:
        if self._accept("op", "("):
            node = self._or()
            self._expect("op", ")")
            return node

        token = self._accept("number")
        if token is not None:
            return Number(float(token.text))

        token = self._accept("name")
        if token is None:
            raise ParseError(
                f"expected a feature, number or '(' at position {self.current.pos} "
                f"in {self.source!r}, got {self.current.text!r}"
            )
        name = token.text

        args: list[float] = []
        # A bare name is a zero-argument feature; "(" opens an argument list,
        # and "()" is the empty one.
        if self._accept("op", "(") and self._accept("op", ")") is None:
            while True:
                args.append(self._number_literal())
                if self._accept("op", ",") is None:
                    break
            self._expect("op", ")")

        try:
            spec = self._resolve_feature(name)
        except KeyError as exc:
            raise ParseError(str(exc.args[0])) from exc
        if len(args) > spec.arity:
            raise ParseError(
                f"{name} takes at most {spec.arity} argument(s), got {len(args)} in {self.source!r}"
            )
        return Call(name, tuple(args))

    def _number_literal(self) -> float:
        """Feature arguments must be literals.

        Allowing ``ema(n)`` where ``n`` is itself an expression would make the
        warm-up of a strategy depend on its own data, which is unanswerable
        before the backtest runs. Parameters come from the spec, not the market.
        """
        negative = self._accept("op", "-") is not None
        token = self._expect("number")
        value = float(token.text)
        return -value if negative else value


def parse(source: str, *, resolve_feature: Callable[[str], FeatureLookup] = resolve) -> Expr:
    """Parse one expression. Raises :class:`ParseError` with an LLM-readable message.

    ``resolve_feature`` is the feature table, and it is the *only* thing D5
    means by "same tokenizer, same whitelist... only the feature table
    changes": swapping it in for `aqr.options.features.resolve_entry_feature`
    is what lets an `OptionSpec` write ``iv_rank() > 50 and close > sma(200)``
    without this module, or any of its other callers, knowing an option
    feature exists. The default is the unchanged equity registry, so every
    existing caller parses exactly as it did before this parameter existed.
    """
    if not source or not source.strip():
        raise ParseError("empty expression")
    return _Parser(source, resolve_feature=resolve_feature).parse()


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #


def feature_keys(node: Expr) -> set[FeatureKey]:
    """Every feature the expression touches — used to compute warm-up up front."""
    if isinstance(node, Call):
        return {node.key}
    if isinstance(node, Number):
        return set()
    if isinstance(node, (Unary, Not)):
        return feature_keys(node.operand)
    if isinstance(node, (Binary, Compare, Logic)):
        return feature_keys(node.left) | feature_keys(node.right)
    raise TypeError(f"unhandled node {type(node).__name__}")


def evaluate(node: Expr, frame: FeatureSource) -> Values:
    """Evaluate over every bar at once.

    Returns a float array for arithmetic nodes and a boolean array for
    comparison/logic nodes. ``NaN`` compares ``False`` in every direction, so a
    strategy is silent until its indicators are warm.
    """
    if isinstance(node, Number):
        return np.full(len(frame), node.value, dtype=np.float64)

    if isinstance(node, Call):
        return frame.get(node.key)

    if isinstance(node, Unary):
        return -evaluate(node.operand, frame)

    if isinstance(node, Binary):
        left = evaluate(node.left, frame)
        right = evaluate(node.right, frame)
        with np.errstate(divide="ignore", invalid="ignore"):
            if node.op == "+":
                return left + right
            if node.op == "-":
                return left - right
            if node.op == "*":
                return left * right
            if node.op == "/":
                return np.where(right == 0.0, np.nan, left / np.where(right == 0.0, 1.0, right))
        raise ParseError(f"unknown operator {node.op!r}")

    if isinstance(node, Compare):
        left = evaluate(node.left, frame)
        right = evaluate(node.right, frame)
        known = ~(np.isnan(left) | np.isnan(right))
        with np.errstate(invalid="ignore"):
            table = {
                "<": left < right,
                "<=": left <= right,
                ">": left > right,
                ">=": left >= right,
                "==": left == right,
                "!=": left != right,
            }
        if node.op not in table:
            raise ParseError(f"unknown comparison {node.op!r}")
        return np.asarray(table[node.op] & known, dtype=bool)

    if isinstance(node, Logic):
        left = _as_mask(evaluate(node.left, frame), node.left)
        right = _as_mask(evaluate(node.right, frame), node.right)
        return left & right if node.op == "and" else left | right

    if isinstance(node, Not):
        return ~_as_mask(evaluate(node.operand, frame), node.operand)

    raise TypeError(f"unhandled node {type(node).__name__}")


def _as_mask(values: Values, node: Expr) -> Values:
    """Refuse to coerce a number into a condition.

    ``rsi(14) and close > 5`` is almost certainly a mistake by whoever (or
    whatever) wrote it, and silently reading it as "RSI is non-zero" would hide
    that mistake inside a plausible-looking equity curve.
    """
    if values.dtype != bool:
        raise ParseError(
            f"{node} is a number, not a condition; compare it to something "
            f"(for example '{node} > 0')"
        )
    return values
