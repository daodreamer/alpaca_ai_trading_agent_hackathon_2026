"""The strategy specification: what an LLM is allowed to propose.

A :class:`StrategySpec` is the complete, self-contained description of a
tradable rule set. It is frozen, hashable and round-trips through YAML, so an
experiment can be replayed years later from its recorded spec alone.

Section 9 of the architecture lists what a strategy must contain. Every item on
that list is a required field here rather than a convention, because a missing
stop-loss is not a strategy with a gap -- it is a different, much worse strategy.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal

from aqr.dsl.expr import Compare, Expr, Logic, Not, feature_keys, parse
from aqr.features.engine import FeatureKey

__all__ = [
    "ExitRules",
    "Sizing",
    "StopLoss",
    "StrategySpec",
    "TakeProfit",
    "Universe",
    "spec_from_dict",
    "spec_to_dict",
]

Direction = Literal["long", "short", "market_neutral"]

Mode = Literal["signal", "portfolio"]
"""Which engine runs the spec.

``signal``     event-driven: a boolean entry opens a position that a stop, a
               target or a bar count later closes. At most ``max_positions`` at
               once, and out of the market whenever nothing fires.
``portfolio``  cross-sectional: rank the universe by ``rank_by``, hold the top
               ``hold`` names, rebalance every ``rebalance_every`` bars, stay
               invested throughout.

The distinction is not a convenience. A book that is out of the market part of
the time forfeits part of the drift it is measured against, which is why every
strategy this project has promoted lost to buy and hold. Beating a long
benchmark needs a form that keeps beta and earns the difference on selection.
"""


@dataclass(frozen=True, slots=True)
class Universe:
    """What the strategy is allowed to trade."""

    symbols: tuple[str, ...]
    timeframe: str = "1D"

    def __post_init__(self) -> None:
        if not self.symbols:
            raise ValueError("universe.symbols must not be empty")


@dataclass(frozen=True, slots=True)
class StopLoss:
    """Where the position is wrong.

    ``atr``     distance = multiplier * atr(period)
    ``percent`` distance = value * entry price
    """

    type: Literal["atr", "percent"] = "atr"
    multiplier: float = 2.0
    period: int = 14

    def __post_init__(self) -> None:
        if self.multiplier <= 0:
            raise ValueError("stop_loss.multiplier must be > 0")
        if self.type == "atr" and self.period < 1:
            raise ValueError("stop_loss.period must be >= 1")


@dataclass(frozen=True, slots=True)
class TakeProfit:
    """Where the position is right.

    ``risk_reward`` distance = ratio * stop distance
    ``percent``     distance = ratio * entry price
    ``none``        no target; the exit is the stop or the holding limit
    """

    type: Literal["risk_reward", "percent", "none"] = "risk_reward"
    ratio: float = 2.0

    def __post_init__(self) -> None:
        if self.type != "none" and self.ratio <= 0:
            raise ValueError("take_profit.ratio must be > 0")


@dataclass(frozen=True, slots=True)
class Sizing:
    """Risk-first position sizing (architecture section 21).

    Size follows from the stop distance, never the other way round: risking a
    fixed fraction of equity means a wider stop buys fewer shares rather than
    quietly taking more risk.
    """

    type: Literal["fixed_fractional", "fixed_notional"] = "fixed_fractional"
    risk_per_trade: float = 0.005
    max_position_pct: float = 0.20

    def __post_init__(self) -> None:
        if not 0 < self.risk_per_trade <= 0.1:
            raise ValueError("sizing.risk_per_trade must be in (0, 0.1]")
        if not 0 < self.max_position_pct <= 1.0:
            raise ValueError("sizing.max_position_pct must be in (0, 1]")


@dataclass(frozen=True, slots=True)
class ExitRules:
    stop_loss: StopLoss = field(default_factory=StopLoss)
    take_profit: TakeProfit = field(default_factory=TakeProfit)
    max_holding_bars: int = 20
    signal_exit: str | None = None
    """Optional expression; when true the position is closed at the next open."""

    def __post_init__(self) -> None:
        if self.max_holding_bars < 1:
            raise ValueError("exit.max_holding_bars must be >= 1")


@dataclass(frozen=True, slots=True)
class Sleeve:
    """The capital held back from the systematic core.

    Reserved for event-driven deviations. What it does while idle is the whole
    design decision: at the benchmark historical CAGR a 20% cash bucket costs
    several percent a year, which is more than the entire realistic alpha
    budget, so the strategy would be structurally behind before its first
    rebalance. An idle sleeve therefore holds the benchmark, and the budget is a
    *deviation* budget rather than a cash bucket. Beta stays at one and every
    sleeve action is a pure active bet.

    ``cash`` exists so that drag can be measured rather than taken on trust. It
    is not the default and should not become one.
    """

    budget: float = 0.20
    idle: Literal["benchmark", "cash"] = "benchmark"

    def __post_init__(self) -> None:
        if not 0.0 <= self.budget < 1.0:
            raise ValueError("strategy.sleeve.budget must be in [0, 1)")
        if self.idle not in ("benchmark", "cash"):
            raise ValueError("strategy.sleeve.idle must be 'benchmark' or 'cash'")

    @property
    def core_budget(self) -> float:
        return 1.0 - self.budget


@dataclass(frozen=True, slots=True)
class StrategySpec:
    """A complete, executable strategy. Immutable and content-addressed."""

    name: str
    entry: str
    universe: Universe
    exit: ExitRules = field(default_factory=ExitRules)
    sizing: Sizing = field(default_factory=Sizing)
    direction: Direction = "long"
    regime: str | None = None
    """Optional expression gating *all* entries, e.g. ``close > ema(200)``."""
    short_entry: str | None = None
    """The short leg, for ``direction: market_neutral`` only.

    A separate expression rather than the long rule negated. Mirroring would
    express a hypothesis nobody proposed -- that one condition predicts up
    moves and its negation predicts down ones -- and that symmetry is exactly
    what equities do not have: they drift upward, losses on the short side are
    unbounded, and a borrow fee accrues every day."""
    max_positions: int = 5
    mode: Mode = "signal"
    rank_by: str | None = None
    """Portfolio mode: a *numeric* expression the universe is sorted by, best first."""
    screen: str | None = None
    """Portfolio mode: an optional condition a name must satisfy to be eligible."""
    hold: int = 20
    """Portfolio mode: how many names the core holds."""
    rebalance_every: int = 5
    """Portfolio mode: bars between rebalances."""
    sleeve: Sleeve = field(default_factory=Sleeve)
    version: int = 1
    hypothesis: str = ""
    parent: str | None = None
    """Fingerprint of the spec this one was evolved from, if any."""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("strategy.name must not be empty")
        if self.max_positions < 1:
            raise ValueError("strategy.max_positions must be >= 1")
        if self.mode == "portfolio":
            if not self.rank_by:
                raise ValueError(
                    "strategy.rank_by is required in portfolio mode; without a "
                    "ranking there is nothing to hold the top of"
                )
            if self.entry:
                raise ValueError(
                    "strategy.entry is not used in portfolio mode: the book is "
                    "chosen by rank_by, not opened by a trigger"
                )
            if self.hold < 1:
                raise ValueError("strategy.hold must be >= 1")
            if self.rebalance_every < 1:
                raise ValueError("strategy.rebalance_every must be >= 1")
        else:
            for _field in ("rank_by", "screen"):
                if getattr(self, _field):
                    raise ValueError(
                        f"strategy.{_field} is only meaningful in portfolio mode; "
                        f"set mode: portfolio or remove it"
                    )
            if not self.entry:
                raise ValueError("strategy.entry is required in signal mode")
        if self.direction == "market_neutral" and not self.short_entry:
            raise ValueError(
                "strategy.short_entry is required for direction 'market_neutral'; "
                "without it this is a long-only strategy wearing the label"
            )
        if self.direction != "market_neutral" and self.short_entry:
            raise ValueError(
                f"strategy.short_entry is only meaningful for direction "
                f"'market_neutral', not {self.direction!r}"
            )
        # Parse eagerly: a spec that cannot be compiled must not exist.
        if self.entry:
            object.__setattr__(self, "_entry_ast", parse(self.entry))
        if self.rank_by:
            rank = parse(self.rank_by)
            if isinstance(rank, Compare | Logic | Not):
                # ``rank_by: close > ema(20)`` sorts the book by True and False.
                # It parses, it runs, and it produces an arbitrary two-tier
                # ordering that looks exactly like a ranking.
                raise ValueError(
                    f"strategy.rank_by must be a number, not a condition: "
                    f"{self.rank_by!r} yields true/false. Rank by the quantity "
                    f"itself, for example 'close / ema(20)'."
                )
            object.__setattr__(self, "_rank_ast", rank)
        if self.screen:
            object.__setattr__(self, "_screen_ast", parse(self.screen))
        if self.regime:
            object.__setattr__(self, "_regime_ast", parse(self.regime))
        if self.exit.signal_exit:
            object.__setattr__(self, "_exit_ast", parse(self.exit.signal_exit))
        if self.short_entry:
            object.__setattr__(self, "_short_ast", parse(self.short_entry))

    # ``slots=True`` needs the cached ASTs declared; they are derived, not data.
    _entry_ast: Expr = field(init=False, repr=False, compare=False, default=None)  # type: ignore[assignment]
    _regime_ast: Expr | None = field(init=False, repr=False, compare=False, default=None)
    _exit_ast: Expr | None = field(init=False, repr=False, compare=False, default=None)
    _short_ast: Expr | None = field(init=False, repr=False, compare=False, default=None)
    _rank_ast: Expr | None = field(init=False, repr=False, compare=False, default=None)
    _screen_ast: Expr | None = field(init=False, repr=False, compare=False, default=None)

    @property
    def entry_ast(self) -> Expr:
        return self._entry_ast

    @property
    def regime_ast(self) -> Expr | None:
        return self._regime_ast

    @property
    def exit_ast(self) -> Expr | None:
        return self._exit_ast

    @property
    def short_entry_ast(self) -> Expr | None:
        return self._short_ast

    @property
    def rank_ast(self) -> Expr | None:
        return self._rank_ast

    @property
    def screen_ast(self) -> Expr | None:
        return self._screen_ast

    def features(self) -> set[FeatureKey]:
        """Every feature the strategy touches, stop-loss ATR included.

        In portfolio mode it is *not* included, because the cross-sectional
        engine never reads ``exit``: the book is chosen by ``rank_by`` and
        turned over on the rebalance, and no stop fires. Declaring the ATR there
        anyway cost more than the arithmetic -- warm-up is the maximum over
        every declared feature, so a spec ranking on a short lookback would sit
        out extra sessions waiting for an indicator nothing consults.
        """
        keys = feature_keys(self._entry_ast) if self._entry_ast is not None else set()
        if self._rank_ast is not None:
            keys |= feature_keys(self._rank_ast)
        if self._screen_ast is not None:
            keys |= feature_keys(self._screen_ast)
        if self._short_ast is not None:
            keys |= feature_keys(self._short_ast)
        if self._regime_ast is not None:
            keys |= feature_keys(self._regime_ast)
        if self._exit_ast is not None:
            keys |= feature_keys(self._exit_ast)
        if self.mode != "portfolio" and self.exit.stop_loss.type == "atr":
            keys.add(FeatureKey("atr", (float(self.exit.stop_loss.period),)))
        return keys

    def parameter_count(self) -> int:
        """Free parameters, for the overfitting detector.

        Counts every numeric literal in the rule expressions plus the exit and
        sizing knobs. This is the denominator in "how many degrees of freedom did
        we spend to get this Sharpe".
        """
        literals = 0
        for text in (
            self.entry,
            self.regime or "",
            self.exit.signal_exit or "",
            self.short_entry or "",
        ):
            if not text:
                continue
            from aqr.dsl.expr import tokenize

            literals += sum(1 for t in tokenize(text) if t.kind == "number")
        exits = 2 if self.exit.take_profit.type != "none" else 1
        return literals + exits + 1  # +1 for max_holding_bars

    def fingerprint(self) -> str:
        """Stable content hash. Two specs that trade identically share a hash."""
        payload = json.dumps(spec_to_dict(self, canonical=True), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def with_params(self, **changes: Any) -> StrategySpec:
        """A copy with fields replaced — used by perturbation and evolution."""
        return replace(self, **changes)


# --------------------------------------------------------------------------- #
# (de)serialisation
# --------------------------------------------------------------------------- #


def spec_to_dict(spec: StrategySpec, *, canonical: bool = False) -> dict[str, Any]:
    """Plain-dict form for YAML/JSON.

    ``canonical=True`` drops provenance (name, hypothesis, parent) so the hash
    identifies *behaviour*. Renaming a strategy must not make it look new.
    """
    out: dict[str, Any] = {
        "entry": spec.entry,
        "short_entry": spec.short_entry,
        "direction": spec.direction,
        "regime": spec.regime,
        "max_positions": spec.max_positions,
        "mode": spec.mode,
        "rank_by": spec.rank_by,
        "screen": spec.screen,
        "hold": spec.hold,
        "rebalance_every": spec.rebalance_every,
        "sleeve": asdict(spec.sleeve),
        "universe": asdict(spec.universe),
        "exit": {
            "stop_loss": asdict(spec.exit.stop_loss),
            "take_profit": asdict(spec.exit.take_profit),
            "max_holding_bars": spec.exit.max_holding_bars,
            "signal_exit": spec.exit.signal_exit,
        },
        "sizing": asdict(spec.sizing),
    }
    out["universe"]["symbols"] = list(spec.universe.symbols)
    if canonical:
        # Symbol order is not behaviour; symbol set is.
        out["universe"]["symbols"] = sorted(out["universe"]["symbols"])
        return out
    out["name"] = spec.name
    out["version"] = spec.version
    out["hypothesis"] = spec.hypothesis
    out["parent"] = spec.parent
    return out


_UNIVERSE_FIELDS = {"symbols", "timeframe"}
_EXIT_FIELDS = {"stop_loss", "take_profit", "max_holding_bars", "signal_exit"}
_TOP_FIELDS = {
    "name",
    "entry",
    "short_entry",
    "universe",
    "exit",
    "sizing",
    "direction",
    "regime",
    "max_positions",
    "mode",
    "rank_by",
    "screen",
    "hold",
    "rebalance_every",
    "sleeve",
    "version",
    "hypothesis",
    "parent",
}


def _reject_unknown(got: dict[str, Any], allowed: set[str], where: str) -> None:
    """Unknown keys are errors, not noise.

    An LLM that writes ``stop_los:`` should be told, not silently given the
    default stop it did not ask for.
    """
    unknown = set(got) - allowed
    if unknown:
        raise ValueError(f"{where}: unknown field(s) {sorted(unknown)}; allowed: {sorted(allowed)}")


def spec_from_dict(raw: dict[str, Any]) -> StrategySpec:
    """Build a spec from parsed YAML, rejecting anything unexpected."""
    if not isinstance(raw, dict):
        raise ValueError(f"strategy must be a mapping, got {type(raw).__name__}")
    if "strategy" in raw and len(raw) == 1:
        raw = raw["strategy"]
    _reject_unknown(raw, _TOP_FIELDS, "strategy")

    mode = str(raw.get("mode", "signal"))
    needed = ("name", "universe") if mode == "portfolio" else ("name", "entry", "universe")
    for key in needed:
        if key not in raw:
            raise ValueError(f"strategy.{key} is required")

    uni_raw = dict(raw["universe"])
    _reject_unknown(uni_raw, _UNIVERSE_FIELDS, "strategy.universe")
    symbols = uni_raw.get("symbols") or []
    if isinstance(symbols, str):
        symbols = [symbols]
    universe = Universe(
        symbols=tuple(str(s).upper() for s in symbols),
        timeframe=str(uni_raw.get("timeframe", "1D")),
    )

    exit_raw = dict(raw.get("exit") or {})
    _reject_unknown(exit_raw, _EXIT_FIELDS, "strategy.exit")
    stop_raw = dict(exit_raw.get("stop_loss") or {})
    _reject_unknown(stop_raw, {"type", "multiplier", "period"}, "strategy.exit.stop_loss")
    tp_raw = dict(exit_raw.get("take_profit") or {})
    _reject_unknown(tp_raw, {"type", "ratio"}, "strategy.exit.take_profit")
    exits = ExitRules(
        stop_loss=StopLoss(**stop_raw),
        take_profit=TakeProfit(**tp_raw),
        max_holding_bars=int(exit_raw.get("max_holding_bars", 20)),
        signal_exit=exit_raw.get("signal_exit") or None,
    )

    sleeve_raw = dict(raw.get("sleeve") or {})
    _reject_unknown(sleeve_raw, {"budget", "idle"}, "strategy.sleeve")

    sizing_raw = dict(raw.get("sizing") or {})
    _reject_unknown(
        sizing_raw, {"type", "risk_per_trade", "max_position_pct"}, "strategy.sizing"
    )

    return StrategySpec(
        name=str(raw["name"]),
        entry=str(raw.get("entry") or ""),
        universe=universe,
        exit=exits,
        sizing=Sizing(**sizing_raw),
        direction=str(raw.get("direction", "long")),  # type: ignore[arg-type]
        regime=raw.get("regime") or None,
        short_entry=(raw.get("short_entry") or None),
        max_positions=int(raw.get("max_positions", 5)),
        mode=mode,  # type: ignore[arg-type]
        rank_by=(raw.get("rank_by") or None),
        screen=(raw.get("screen") or None),
        hold=int(raw.get("hold", 20)),
        rebalance_every=int(raw.get("rebalance_every", 5)),
        sleeve=Sleeve(**sleeve_raw),
        version=int(raw.get("version", 1)),
        hypothesis=str(raw.get("hypothesis", "")),
        parent=raw.get("parent") or None,
    )
