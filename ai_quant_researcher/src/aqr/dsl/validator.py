"""Checks a spec must pass before any compute is spent on it.

Parsing proves a strategy is *expressible*. Validation asks whether it is
*sane*: does it have enough history to warm up, does it risk a survivable
fraction, is the entry condition ever true, is the parameter count wildly out of
proportion to the sample. Rejecting here is cheap; discovering it after a
walk-forward run is not.

The distinction between an error and a warning is deliberate. Errors are
structural and block execution. Warnings are judgement calls that travel with
the experiment record so a human -- or the Critic agent -- can weigh them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from aqr.data.bars import Bars
from aqr.dsl.expr import evaluate
from aqr.dsl.schema import StrategySpec
from aqr.features.cross_section import CrossSection
from aqr.features.engine import FeatureFrame

__all__ = ["ValidationReport", "validate", "validate_against"]

# A strategy whose entry fires on nearly every bar is not a signal, it is a
# constant. The threshold is loose on purpose: filtering research too early is
# its own failure mode.
_MAX_SIGNAL_RATE = 0.60
_MIN_SIGNAL_COUNT = 10


@dataclass(slots=True)
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_if_failed(self) -> None:
        if self.errors:
            joined = "\n  - ".join(self.errors)
            raise ValueError(f"strategy failed validation:\n  - {joined}")

    def __str__(self) -> str:
        lines = [f"ERROR   {e}" for e in self.errors]
        lines += [f"WARNING {w}" for w in self.warnings]
        return "\n".join(lines) or "OK"


def validate(spec: StrategySpec) -> ValidationReport:
    """Data-independent checks. Cheap enough to run on every LLM proposal."""
    report = ValidationReport()

    if spec.direction not in ("long", "short", "market_neutral"):
        report.errors.append(
            f"direction must be 'long', 'short' or 'market_neutral', got {spec.direction!r}"
        )
    if spec.direction == "market_neutral" and not spec.short_entry:
        report.errors.append(
            "a market-neutral strategy needs a short_entry as well as an entry"
        )

    # The entry must be a condition, not a number. evaluate() enforces this at
    # run time; catching it here turns a mid-backtest crash into a clean reject.
    #
    # Portfolio specs have no entry: they are chosen by rank, not opened by a
    # trigger. Applying the entry rule to them rejected every one before the
    # pipeline had run anything -- the rule the validator had, correctly applied
    # to a spec it did not cover.
    if spec.mode == "portfolio":
        if spec.rank_by and _is_boolean(spec.rank_by):
            report.errors.append(
                f"rank_by must be a number, not a condition: {spec.rank_by!r} sorts "
                f"the book into true and false"
            )
        if spec.screen and not _is_boolean(spec.screen):
            report.errors.append(f"screen must be a condition, got {spec.screen!r}")
    elif not _is_boolean(spec.entry):
        report.errors.append(f"entry must be a condition, got the expression {spec.entry!r}")
    if spec.regime and not _is_boolean(spec.regime):
        report.errors.append(f"regime must be a condition, got {spec.regime!r}")
    if spec.exit.signal_exit and not _is_boolean(spec.exit.signal_exit):
        report.errors.append(f"exit.signal_exit must be a condition, got {spec.exit.signal_exit!r}")

    if spec.sizing.risk_per_trade > 0.02:
        report.warnings.append(
            f"risk_per_trade {spec.sizing.risk_per_trade:.1%} exceeds 2%; "
            "a 10-loss streak would cost more than 18% of equity"
        )
    if spec.exit.take_profit.type == "none" and spec.exit.max_holding_bars > 250:
        report.warnings.append(
            "no take-profit and a holding limit over a year: exits will be dominated by the stop"
        )
    if spec.parameter_count() > 12:
        report.warnings.append(
            f"{spec.parameter_count()} free parameters is a lot of degrees of freedom "
            "for a single hypothesis"
        )
    return report


def validate_against(
    spec: StrategySpec, bars: Bars, cross_section: CrossSection | None = None
) -> ValidationReport:
    """Data-dependent checks: warm-up, signal frequency, tradability.

    ``cross_section`` is required only by strategies that use peer-relative
    features. Omitting it for one that does turns into a validation error
    naming the problem, rather than an exception three layers up.
    """
    report = validate(spec)
    if not report.ok:
        return report

    frame = FeatureFrame(bars, cross_section)
    try:
        warmup = frame.warmup(spec.features())
    except KeyError as exc:
        report.errors.append(str(exc.args[0]))
        return report
    except ValueError as exc:
        # A cross-sectional feature with no universe behind it. A real
        # finding about the call, not about the strategy.
        report.errors.append(str(exc))
        return report

    n = len(bars)
    if warmup >= n:
        report.errors.append(
            f"{spec.name} needs {warmup} bars to warm up but only {n} are available"
        )
        return report
    usable = n - warmup
    if usable < 60:
        report.warnings.append(
            f"only {usable} tradable bars after a {warmup}-bar warm-up; "
            "results will not be statistically meaningful"
        )

    if spec.mode == "portfolio":
        return _validate_portfolio(spec, bars, frame, report, warmup, cross_section)

    mask = np.asarray(evaluate(spec.entry_ast, frame), dtype=bool)
    if spec.regime_ast is not None:
        mask &= np.asarray(evaluate(spec.regime_ast, frame), dtype=bool)
    mask[:warmup] = False

    fired = int(mask.sum())
    if fired == 0:
        report.errors.append(
            f"{spec.name} never fires on {bars.symbol}: the entry condition is unsatisfiable here"
        )
    elif fired < _MIN_SIGNAL_COUNT:
        report.warnings.append(
            f"entry fires only {fired} times on {bars.symbol}; too few to evaluate"
        )
    rate = fired / usable if usable else 0.0
    if rate > _MAX_SIGNAL_RATE:
        report.warnings.append(
            f"entry is true on {rate:.0%} of tradable bars on {bars.symbol}; "
            "this is closer to buy-and-hold than to a signal"
        )
    return report


def _validate_portfolio(
    spec: StrategySpec,
    bars: Bars,
    frame: FeatureFrame,
    report: ValidationReport,
    warmup: int,
    cross_section: CrossSection | None,
) -> ValidationReport:
    """The portfolio equivalents of "does the entry ever fire".

    Two questions, not one. A ranking can be well-defined and still useless, and
    the useless case is the dangerous one because it produces a plausible curve.
    """
    assert spec.rank_ast is not None  # StrategySpec refuses a portfolio spec without one
    values = np.asarray(evaluate(spec.rank_ast, frame), dtype=np.float64)[warmup:]

    if values.size == 0 or not np.isfinite(values).any():
        report.errors.append(
            f"{spec.name}: rank_by is never defined on {bars.symbol}; the book would "
            f"be empty at every rebalance and the equity curve a flat line"
        )
        return report

    finite = values[np.isfinite(values)]
    if finite.size and float(finite.std()) == 0.0:
        # Constant across time here; whether it is constant across the universe
        # needs the peers, checked below when they are available.
        report.warnings.append(
            f"{spec.name}: rank_by is the same value on every bar of {bars.symbol}"
        )

    if cross_section is not None:
        spread = _cross_sectional_spread(spec, cross_section, warmup)
        if spread is not None and spread == 0.0:
            report.errors.append(
                f"{spec.name}: rank_by is constant across the universe, so it cannot "
                f"distinguish one name from another; the book would be sorted by symbol"
            )

    if spec.screen_ast is not None:
        passes = np.asarray(evaluate(spec.screen_ast, frame), dtype=bool)[warmup:]
        if not passes.any():
            report.errors.append(
                f"{spec.name}: the screen never passes on {bars.symbol}; nothing is eligible"
            )

    if spec.hold >= len(spec.universe.symbols):
        report.warnings.append(
            f"hold={spec.hold} is at least the size of the {len(spec.universe.symbols)}-name "
            f"universe, so the core is the universe -- which is the benchmark, and a "
            f"strategy that is the benchmark has no alpha to find"
        )
    return report


def _cross_sectional_spread(
    spec: StrategySpec, cross_section: CrossSection, warmup: int
) -> float | None:
    """Mean dispersion of the ranking across the universe, or None if unknowable."""
    assert spec.rank_ast is not None
    columns: list[np.ndarray] = []
    for symbol in spec.universe.symbols:
        peer = cross_section.peers.get(symbol)
        if peer is None:
            return None
        values = np.asarray(
            evaluate(spec.rank_ast, FeatureFrame(peer, cross_section)), dtype=np.float64
        )[warmup:]
        columns.append(values)
    if len(columns) < 2:
        return None
    width = min(c.size for c in columns)
    if width == 0:
        return None
    stacked = np.vstack([c[:width] for c in columns])
    with np.errstate(invalid="ignore"):
        spread = np.nanstd(stacked, axis=0)
    finite = spread[np.isfinite(spread)]
    return float(finite.mean()) if finite.size else None


def _is_boolean(source: str) -> bool:
    """Whether an expression yields a mask, decided structurally without data."""
    from aqr.dsl.expr import Compare, Logic, Not, ParseError, parse

    try:
        node = parse(source)
    except ParseError:
        return False
    return isinstance(node, (Compare, Logic, Not))
