"""What an LLM is allowed to propose about options — specs/10-options-research.md D5.

A separate type from [`dsl/schema.py`](../dsl/schema.py)'s ``StrategySpec``, not
new fields on it. The equity spec's ``stop_loss``, ``take_profit`` and
``max_holding_bars`` have no meaning on a structure held to expiry, and its
``mode: portfolio`` has no cross-section to rank on one underlying. Widening one
type to cover both would make every validator branch on which half of itself was
in use, and the LLM prompt would have to explain which fields to ignore.

**There is no ``exit`` block, and its absence is the design.** No stop, no
target, no signal exit, no roll. specs/10 D0 measured that a specific contract
is re-quoted on 1–3% of later sessions, so none of those can be priced. A field
that exists but whose semantics are fabricated is worse than a missing one: the
model will fill it in, and every number downstream will quietly describe a
strategy nobody can trade.

The entry expression is parsed by the *existing* DSL — same tokenizer, same
whitelist, same ``feature_keys`` walk — against the combined table
``options/features.py`` builds, so a rule can say
``iv_rank() > 50 and close > sma(200)`` in one expression: ``close`` and
``sma`` resolve through the unchanged bar registry, ``iv_rank`` through
specs/10 D6's option feature table, and ``OptionSpec`` never has to know which
is which.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from typing import Any

import yaml

from aqr.dsl.expr import Expr, feature_keys, parse, tokenize
from aqr.features.engine import FeatureKey
from aqr.options.features import resolve_entry_feature
from aqr.options.structure import StructureKind

__all__ = [
    "Anchor",
    "Cadence",
    "DteTarget",
    "OptionSizing",
    "OptionSpec",
    "StructureSpec",
    "dumps_option_spec",
    "loads_option_spec",
    "option_spec_from_dict",
    "spec_to_dict",
]

_SPREAD_KINDS = frozenset(
    {
        "put_credit_spread",
        "call_credit_spread",
        "put_debit_spread",
        "call_debit_spread",
        "iron_condor",
    }
)

_CREDIT_KINDS = frozenset({"put_credit_spread", "call_credit_spread", "iron_condor"})


@dataclass(frozen=True, slots=True)
class DteTarget:
    """Which expiry, by days to it rather than by date.

    The vendor samples three rolling targets near 14, 28 and 49 days, and the
    dates move every session, so naming a date would name a contract that
    exists on one session in twenty.
    """

    target: int = 28
    tolerance: int = 10

    def __post_init__(self) -> None:
        if self.target < 1:
            raise ValueError("dte.target must be >= 1")
        if self.tolerance < 0:
            raise ValueError("dte.tolerance must be >= 0")


@dataclass(frozen=True, slots=True)
class Anchor:
    """The leg the rule names, by the magnitude of its delta.

    Whether it is bought or sold follows from the structure kind, not from a
    field: a ``put_credit_spread`` sells its anchor and a ``put_debit_spread``
    buys it, and letting a spec say otherwise would let it name a structure
    whose maximum loss is not the one its kind implies.
    """

    delta: float = 0.16
    tolerance: float = 0.06

    def __post_init__(self) -> None:
        if not 0.0 < self.delta < 1.0:
            raise ValueError(
                f"anchor.delta is a magnitude in (0, 1), got {self.delta}. A put's "
                f"delta is negative in the data; the rule names 0.16 for either right."
            )
        if not 0.0 < self.tolerance <= 0.5:
            raise ValueError("anchor.tolerance must be in (0, 0.5]")


@dataclass(frozen=True, slots=True)
class StructureSpec:
    """Which structure, at which expiry, anchored on which delta."""

    type: StructureKind
    dte: DteTarget = field(default_factory=DteTarget)
    anchor: Anchor = field(default_factory=Anchor)
    width_delta: float | None = None
    """The protective leg, named by its own delta. **Prefer this.**

    Measured on the SPY research window against a 16-delta short put: a
    delta-selected wing resolves on 98% of sessions and a fixed 10-point wing on
    23%, because the cache samples about 24 rungs from a ladder that lists
    hundreds. A point width is not a rule this data can express, and a search
    told to use one would be selecting on the vendor's sampling.
    """
    width_points: float | None = None
    """An exact strike distance. Refuses when the ladder does not list it."""
    call_anchor: Anchor | None = None
    """Iron condor only. Defaults to the put side's anchor, mirrored."""
    call_width_delta: float | None = None
    call_width_points: float | None = None

    def __post_init__(self) -> None:
        spread = self.type in _SPREAD_KINDS
        named = [w for w in (self.width_delta, self.width_points) if w]
        if spread and not named:
            raise ValueError(
                f"{self.type}: a width is required -- width_delta (preferred) or "
                f"width_points. Risk is the width less the credit, so a spread "
                f"without one has no maximum loss to size against."
            )
        if len(named) > 1:
            raise ValueError(
                f"{self.type}: name the width once, as delta or as points, not both"
            )
        if not spread and named:
            raise ValueError(f"{self.type}: has one leg and no width")
        widths = (("width_points", self.width_points), ("width_delta", self.width_delta))
        for label, value in widths:
            if value is not None and value <= 0:
                raise ValueError(f"{label} must be > 0")
        if self.width_delta is not None and self.width_delta >= self.anchor.delta:
            raise ValueError(
                f"width_delta {self.width_delta} is not below the anchor's "
                f"{self.anchor.delta}: the protective leg is further out of the money "
                f"than the leg it protects, so it has the smaller delta"
            )
        if self.type != "iron_condor" and (
            self.call_anchor or self.call_width_points or self.call_width_delta
        ):
            raise ValueError(f"{self.type}: the call_* fields are iron-condor only")

    @property
    def is_credit(self) -> bool:
        return self.type in _CREDIT_KINDS

    @property
    def call_side_anchor(self) -> Anchor:
        return self.call_anchor or self.anchor

    @property
    def call_side_width_delta(self) -> float | None:
        return self.call_width_delta or self.width_delta

    @property
    def call_side_width_points(self) -> float | None:
        return self.call_width_points or self.width_points


@dataclass(frozen=True, slots=True)
class OptionSizing:
    """Risk-first, against the structure's own maximum loss.

    The denominator is what the position can actually lose, not its notional
    and not its credit. A rule that risked a fraction of the credit would take
    nine times the intended risk on a 10-wide spread collecting 1.00.
    """

    risk_per_trade: float = 0.01
    max_concurrent: int = 3

    def __post_init__(self) -> None:
        if not 0 < self.risk_per_trade <= 0.1:
            raise ValueError("sizing.risk_per_trade must be in (0, 0.1]")
        if self.max_concurrent < 1:
            raise ValueError("sizing.max_concurrent must be >= 1")


@dataclass(frozen=True, slots=True)
class Cadence:
    """How far apart entries must sit, in sessions.

    This is the only control the spec has over the *effective sample size*.
    Overlapping entries produce correlated trades, and specs/10 D8 gates on
    non-overlapping cycles rather than on the trade count for exactly that
    reason.
    """

    min_sessions_between_entries: int = 5

    def __post_init__(self) -> None:
        if self.min_sessions_between_entries < 1:
            raise ValueError("cadence.min_sessions_between_entries must be >= 1")


@dataclass(frozen=True, slots=True)
class OptionSpec:
    """A complete, executable option rule. Immutable and content-addressed."""

    name: str
    underlying: str
    entry: str
    structure: StructureSpec
    sizing: OptionSizing = field(default_factory=OptionSizing)
    cadence: Cadence = field(default_factory=Cadence)
    version: int = 1
    hypothesis: str = ""
    parent: str | None = None

    _entry_ast: Expr = field(init=False, repr=False, compare=False, default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("strategy.name must not be empty")
        if not self.underlying:
            raise ValueError("strategy.underlying must not be empty")
        if not self.entry:
            raise ValueError(
                "strategy.entry is required. A rule with no condition opens a position "
                "every session it can, which is a schedule rather than a hypothesis."
            )
        # Parse eagerly: a spec that cannot be compiled must not exist. The
        # combined resolver (bar registry + specs/10 D6's option table) is
        # what lets `entry` say `iv_rank() > 50 and close > sma(200)` --
        # dsl/expr.py's tokenizer, grammar and whitelist are otherwise
        # untouched (options/features.py's module docstring).
        object.__setattr__(
            self, "_entry_ast", parse(self.entry, resolve_feature=resolve_entry_feature)
        )

    @property
    def entry_ast(self) -> Expr:
        return self._entry_ast

    def features(self) -> set[FeatureKey]:
        return feature_keys(self._entry_ast)

    def parameter_count(self) -> int:
        """Free parameters, for the overfitting detector.

        The structure's knobs count: a delta target, a width and a DTE are three
        numbers chosen by search exactly like a lookback is.
        """
        literals = sum(1 for token in tokenize(self.entry) if token.kind == "number")
        structure = self.structure
        knobs = 2  # the DTE target and the anchor's delta
        if structure.width_delta or structure.width_points:
            knobs += 1
        if structure.call_anchor:
            knobs += 1
        if structure.call_width_delta or structure.call_width_points:
            knobs += 1
        return literals + knobs + 1  # +1 for the cadence

    def fingerprint(self) -> str:
        """Stable content hash. Two specs that trade identically share a hash."""
        payload = json.dumps(spec_to_dict(self, canonical=True), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def with_params(self, **changes: Any) -> OptionSpec:
        return replace(self, **changes)


def spec_to_dict(spec: OptionSpec, *, canonical: bool = False) -> dict[str, Any]:
    """Plain-dict form for YAML/JSON.

    ``canonical=True`` drops provenance, so the hash identifies *behaviour*.
    Renaming a strategy must not make it look new.
    """
    out: dict[str, Any] = {
        "underlying": spec.underlying,
        "entry": spec.entry,
        "structure": asdict(spec.structure),
        "sizing": asdict(spec.sizing),
        "cadence": asdict(spec.cadence),
    }
    if canonical:
        return out
    out["name"] = spec.name
    out["version"] = spec.version
    out["hypothesis"] = spec.hypothesis
    out["parent"] = spec.parent
    return out


# --------------------------------------------------------------------------- #
# YAML round trip
# --------------------------------------------------------------------------- #
#
# The registry stores a spec as text, so an option rule needs the same round
# trip ``dsl/loader.py`` gives an equity one. It lives here rather than there
# for the reason the two spec types are separate at all: ``dsl/`` is the equity
# DSL's home and importing ``options`` into it would put the option vocabulary
# inside the module an equity ``StrategySpec`` is parsed by, which is exactly
# the mixing ``options/features.py`` documents avoiding.
#
# Strings in, strings out. Nothing here opens a file -- ``options/`` is a pure
# layer (tests/test_boundaries.py) and the one caller that needs a path is the
# CLI, which already has one.

_TOP_LEVEL_KEY = "option_strategy"
"""Not ``strategy``. A file that could be loaded by either loader would be
loaded by whichever one the caller happened to reach for, and the two produce
rules with different exits, different sizing and different verdicts."""


def option_spec_from_dict(raw: Mapping[str, Any]) -> OptionSpec:
    """Rebuild a spec from its plain-dict form. The inverse of :func:`spec_to_dict`.

    Accepts the artefact with or without its ``option_strategy`` wrapper, so a
    caller holding the inner mapping (the registry, which stored exactly what
    ``spec_to_dict`` produced) does not have to re-wrap it to read it back.
    """
    body = raw.get(_TOP_LEVEL_KEY, raw)
    if not isinstance(body, Mapping):
        raise ValueError(f"{_TOP_LEVEL_KEY} must be a mapping, got {type(body).__name__}")

    structure_raw = body.get("structure")
    if not isinstance(structure_raw, Mapping):
        raise ValueError("structure is required and must be a mapping")

    def _anchor(value: Any) -> Anchor | None:
        if not isinstance(value, Mapping):
            return None
        return Anchor(
            delta=float(value.get("delta", 0.16)),
            tolerance=float(value.get("tolerance", 0.06)),
        )

    dte_raw = structure_raw.get("dte")
    dte = (
        DteTarget(
            target=int(dte_raw.get("target", 28)),
            tolerance=int(dte_raw.get("tolerance", 10)),
        )
        if isinstance(dte_raw, Mapping)
        else DteTarget()
    )
    structure = StructureSpec(
        type=str(structure_raw.get("type", "")),  # type: ignore[arg-type]
        dte=dte,
        anchor=_anchor(structure_raw.get("anchor")) or Anchor(),
        width_delta=_optional_float(structure_raw.get("width_delta")),
        width_points=_optional_float(structure_raw.get("width_points")),
        call_anchor=_anchor(structure_raw.get("call_anchor")),
        call_width_delta=_optional_float(structure_raw.get("call_width_delta")),
        call_width_points=_optional_float(structure_raw.get("call_width_points")),
    )

    sizing_raw = body.get("sizing")
    sizing = (
        OptionSizing(
            risk_per_trade=float(sizing_raw.get("risk_per_trade", 0.01)),
            max_concurrent=int(sizing_raw.get("max_concurrent", 3)),
        )
        if isinstance(sizing_raw, Mapping)
        else OptionSizing()
    )
    cadence_raw = body.get("cadence")
    cadence = (
        Cadence(
            min_sessions_between_entries=int(
                cadence_raw.get("min_sessions_between_entries", 5)
            )
        )
        if isinstance(cadence_raw, Mapping)
        else Cadence()
    )
    return OptionSpec(
        name=str(body.get("name", "")),
        underlying=str(body.get("underlying", "")),
        entry=str(body.get("entry", "")),
        structure=structure,
        sizing=sizing,
        cadence=cadence,
        version=int(body.get("version", 1)),
        hypothesis=str(body.get("hypothesis") or ""),
        parent=(str(body["parent"]) if body.get("parent") else None),
    )


def _optional_float(value: Any) -> float | None:
    """``None`` for absent *and* for zero.

    A width of zero is not a width, and the dataclass would refuse it with a
    message about a positive number when what the caller meant was "there is no
    wing on this structure". The proposer writes 0 for an absent width because a
    JSON schema cannot express "omit this field", so the two spellings have to
    mean the same thing by the time they reach ``StructureSpec``.
    """
    if value is None:
        return None
    number = float(value)
    return number or None


def dumps_option_spec(spec: OptionSpec) -> str:
    """The spec as YAML, provenance included."""
    return yaml.safe_dump(
        {_TOP_LEVEL_KEY: spec_to_dict(spec)},
        sort_keys=False,
        allow_unicode=True,
        width=100,
    )


def loads_option_spec(text: str) -> OptionSpec:
    """Parse YAML text into a spec, with the source problem kept in the error."""
    try:
        raw: Any = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(f"option strategy YAML is malformed: {exc}") from exc
    if raw is None:
        raise ValueError("option strategy YAML is empty")
    if not isinstance(raw, Mapping):
        raise ValueError(f"expected a mapping, got {type(raw).__name__}")
    return option_spec_from_dict(raw)
