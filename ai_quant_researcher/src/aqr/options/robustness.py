"""Robustness, D8's replacements for a single-underlying cache — specs/10 D8.

``validation/robustness.py``'s ``asset_robustness`` asks whether an edge
depends on one ticker by leaving tickers out, one at a time, and re-running on
the rest. That question has no answer here: ``data-options/`` holds SPY and
nothing else (specs/10 D10), so "leave SPY out" leaves zero underlyings and
"leave one of several out" has no several to choose from. D8 is explicit that
the component must not be zero-filled to make the weighted sum balance --
zero would read as "this edge failed every asset test", which is not what
"there was no asset test" means -- so it is replaced by two things the data
actually supports:

**Leave-one-year-out.** The mechanical analogue of leave-one-asset-out with
*year* standing in for *ticker*: for each of 2019-2024, remove that year's
entry opportunities and re-run on what remains. If dropping any single year
kills the edge, that year -- 2020's crash, 2022's bear market, whichever it
turns out to be -- was not a year the strategy survived, it was the finding.

**DTE-bucket agreement.** specs/10 D0 measured that the vendor samples three
rolling DTE targets, ~14 / ~28 / ~49 days, on every session. A rule is usually
proposed at one of them; running the *same* rule at all three and checking
whether the sign of the result holds is the options-specific version of "does
this survive a parameter nudge" -- except here the three values are not an
arbitrary perturbation grid, they are the only three DTE regimes this cache
actually offers. A rule that only works at its own DTE target found a DTE
target, not a premium (D8's own words).

Parameter perturbation is here too, because "the anchor delta, the width, the
DTE target, the cadence" (D8) are not the knobs ``validation/params.py``
addresses -- that module is written entirely in terms of ``StrategySpec``'s
own field paths (``exit.stop_loss.multiplier``, ``sizing.max_position_pct``,
literals inside an equity expression) and has no notion of an
``OptionSpec.structure.anchor``. Reusing its machinery would mean teaching it
a second spec shape it was never built to address; reusing its *result type*
(``ParameterReport``) instead keeps the report identical in shape to the
equity side's without pretending the two specs share knobs they do not.

Monte Carlo needs no options-specific version at all: ``validation.robustness.
monte_carlo`` already operates on the generic ``Trade`` view and nothing
option-shaped, so it is reused directly rather than wrapped here.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date
from typing import Any

import numpy as np

from aqr.backtest.engine import Trade
from aqr.backtest.metrics import (
    Metrics,
    _trade_stats,  # local: private by intent, as validation/robustness.py already does
    compute_metrics,
)
from aqr.data.bars import Bars
from aqr.features.regime import regime_series
from aqr.options.engine import OptionBacktestConfig
from aqr.options.run import OptionMarket, run_option_strategy
from aqr.options.spec import OptionSpec
from aqr.validation.cycles import independent_cycles
from aqr.validation.robustness import (
    UNSCORED_REGIME,
    ParameterReport,
    RegimeReport,
    score_regimes,
)

__all__ = [
    "DEFAULT_DTE_BUCKETS",
    "DEFAULT_YEARS",
    "DteBucketReport",
    "YearReport",
    "dte_bucket_agreement",
    "leave_one_year_out",
    "option_parameter_stability",
    "option_regime_robustness",
]

DEFAULT_YEARS: tuple[int, ...] = (2019, 2020, 2021, 2022, 2023, 2024)
"""D8's "6 folds over 2019-2024" -- every calendar year the research chain
touches, 2019 (from 02-09) through 2024 (to 08-30) inclusive."""

DEFAULT_DTE_BUCKETS: tuple[int, ...] = (14, 28, 49)
"""specs/10 D0's own rolling targets -- not a perturbation grid chosen for
this test, the three DTE regimes the vendor actually samples every session."""

_MIN_CYCLES_TO_SCORE = 3
"""Below this a year's or a bucket's own sign is one or two trades' sign, not
evidence -- the same reasoning D8 applies to the whole run, applied per slice."""

_DEFAULT_FACTORS = (0.75, 0.9, 1.1, 1.25)
"""validation/robustness.py's own default nudge factors, reused unchanged: the
question -- does the edge survive a +/-25% and +/-10% nudge -- does not
depend on which spec shape is being nudged."""


# --------------------------------------------------------------------------- #
# Leave-one-year-out
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class YearReport:
    per_year: dict[int, Metrics] = field(default_factory=dict)
    cycles: dict[int, int] = field(default_factory=dict)
    score: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "per_year": {y: m.as_dict() for y, m in self.per_year.items()},
            "cycles": dict(self.cycles),
        }

    def __str__(self) -> str:
        rows = [
            f"  drop {y}: ret {m.total_return:+7.1%}  {self.cycles.get(y, 0):3d} cycles "
            f"of the remaining years"
            for y, m in sorted(self.per_year.items())
        ]
        return "\n".join([f"leave-one-year-out {self.score:.2f}", *rows])


def leave_one_year_out(
    spec: OptionSpec,
    market: OptionMarket,
    config: OptionBacktestConfig | None = None,
    *,
    years: tuple[int, ...] = DEFAULT_YEARS,
    min_cycles: int = _MIN_CYCLES_TO_SCORE,
) -> YearReport:
    """Re-run with one calendar year's entries removed, for each year in turn.

    A year with no sessions in this chain at all (the sealed window, or a year
    outside the research cache) is skipped rather than scored as a deletion
    that changed nothing -- there was nothing to delete.
    """
    config = config or OptionBacktestConfig()
    report = YearReport()
    scored = 0
    positive = 0
    for year in years:
        start, stop = date(year, 1, 1), date(year + 1, 1, 1)
        if not market.chain.slice_dates(start, stop).sessions:
            continue
        deletion_chain = market.chain.exclude_dates(start, stop)
        result = run_option_strategy(spec, market.with_chain(deletion_chain), config)
        metrics = compute_metrics(result)
        cycles = independent_cycles(result.trades)
        report.per_year[year] = metrics
        report.cycles[year] = cycles
        if cycles >= min_cycles:
            scored += 1
            if metrics.total_return > 0:
                positive += 1
    report.score = float(positive / scored) if scored else 0.0
    return report


# --------------------------------------------------------------------------- #
# DTE-bucket agreement
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class DteBucketReport:
    per_bucket: dict[int, Metrics] = field(default_factory=dict)
    cycles: dict[int, int] = field(default_factory=dict)
    score: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "per_bucket": {d: m.as_dict() for d, m in self.per_bucket.items()},
            "cycles": dict(self.cycles),
        }

    def __str__(self) -> str:
        rows = [
            f"  ~{d} DTE: ret {m.total_return:+7.1%}  {self.cycles.get(d, 0):3d} cycles"
            for d, m in sorted(self.per_bucket.items())
        ]
        return "\n".join([f"DTE-bucket agreement {self.score:.2f}", *rows])


def dte_bucket_agreement(
    spec: OptionSpec,
    market: OptionMarket,
    config: OptionBacktestConfig | None = None,
    *,
    targets: tuple[int, ...] = DEFAULT_DTE_BUCKETS,
    min_cycles: int = _MIN_CYCLES_TO_SCORE,
) -> DteBucketReport:
    """Re-run ``spec`` at each of the vendor's three DTE targets in turn.

    Only ``structure.dte.target`` moves; ``structure.dte.tolerance`` is left
    exactly as the spec named it, because the point is to ask "does this rule
    still work at 14 or 49 days" and not "does this rule still work at 14 or
    49 days under a tolerance band that was never tuned for them" -- the
    latter would conflate two different perturbations in one score.
    """
    config = config or OptionBacktestConfig()
    report = DteBucketReport()
    scored = 0
    positive = 0
    for target in targets:
        variant = spec.with_params(
            structure=replace(
                spec.structure, dte=replace(spec.structure.dte, target=target)
            )
        )
        result = run_option_strategy(variant, market, config)
        metrics = compute_metrics(result)
        cycles = independent_cycles(result.trades)
        report.per_bucket[target] = metrics
        report.cycles[target] = cycles
        if cycles >= min_cycles:
            scored += 1
            if metrics.total_return > 0:
                positive += 1
    report.score = float(positive / scored) if scored else 0.0
    return report


# --------------------------------------------------------------------------- #
# Parameter perturbation
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class _Slot:
    path: str
    value: float
    integral: bool


def _option_slots(spec: OptionSpec) -> list[_Slot]:
    """The knobs D8 names: the anchor delta, the width, the DTE target, the
    cadence. Whichever width form the spec used (``width_delta`` is preferred
    per specs/10 D5; ``width_points`` stays available) is the one perturbed --
    perturbing the one the spec did not set would silently test a different
    rule."""
    slots = [
        _Slot("structure.anchor.delta", spec.structure.anchor.delta, integral=False),
        _Slot("structure.dte.target", float(spec.structure.dte.target), integral=True),
        _Slot(
            "cadence.min_sessions_between_entries",
            float(spec.cadence.min_sessions_between_entries),
            integral=True,
        ),
    ]
    if spec.structure.width_delta is not None:
        slots.append(_Slot("structure.width_delta", spec.structure.width_delta, integral=False))
    elif spec.structure.width_points is not None:
        slots.append(
            _Slot("structure.width_points", spec.structure.width_points, integral=False)
        )
    return slots


def _with_slot(spec: OptionSpec, slot: _Slot, value: float) -> OptionSpec:
    if slot.path == "structure.anchor.delta":
        anchor = replace(spec.structure.anchor, delta=value)
        return spec.with_params(structure=replace(spec.structure, anchor=anchor))
    if slot.path == "structure.dte.target":
        dte = replace(spec.structure.dte, target=int(value))
        return spec.with_params(structure=replace(spec.structure, dte=dte))
    if slot.path == "structure.width_delta":
        return spec.with_params(structure=replace(spec.structure, width_delta=value))
    if slot.path == "structure.width_points":
        return spec.with_params(structure=replace(spec.structure, width_points=value))
    if slot.path == "cadence.min_sessions_between_entries":
        cadence = replace(spec.cadence, min_sessions_between_entries=int(value))
        return spec.with_params(cadence=cadence)
    raise KeyError(slot.path)  # pragma: no cover -- _option_slots names nothing else


def option_parameter_stability(
    spec: OptionSpec,
    market: OptionMarket,
    config: OptionBacktestConfig | None = None,
    *,
    factors: tuple[float, ...] = _DEFAULT_FACTORS,
) -> ParameterReport:
    """Nudge each of D8's four knobs and see whether the edge survives.

    Mirrors ``validation.robustness.parameter_stability``'s score exactly --
    the fraction of neighbours keeping at least half the baseline Sharpe --
    over a different, options-shaped set of neighbours.
    """
    config = config or OptionBacktestConfig()
    baseline = compute_metrics(run_option_strategy(spec, market, config)).sharpe
    report = ParameterReport(baseline=baseline)
    threshold = 0.5 * baseline if baseline > 0 else None
    survivors = 0
    total = 0

    for slot in _option_slots(spec):
        results: list[float] = []
        seen: set[str] = set()
        for factor in factors:
            candidate = slot.value * factor
            if slot.integral:
                candidate = float(max(1, round(candidate)))
            if candidate == slot.value:
                continue
            try:
                variant = _with_slot(spec, slot, candidate)
            except ValueError:
                continue
            fingerprint = variant.fingerprint()
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            try:
                sharpe = compute_metrics(run_option_strategy(variant, market, config)).sharpe
            except ValueError:
                continue
            results.append(sharpe)
            total += 1
            if threshold is not None and sharpe >= threshold:
                survivors += 1
        if results:
            report.per_slot[slot.path] = results
            if threshold is not None and max(results) < threshold:
                report.fragile.append(slot.path)

    report.score = float(survivors / total) if total else 1.0
    return report


# --------------------------------------------------------------------------- #
# Regime robustness (reused machinery, options-side bucketing)
# --------------------------------------------------------------------------- #


def _metrics_from_trades(trades: list[Trade], equity: float) -> Metrics:
    """Trade-level metrics for a regime bucket -- curve-derived fields (CAGR,
    Sharpe, drawdown) left at zero, exactly as
    ``validation.robustness._metrics_from_trades`` does, because no
    reconstructed curve exists for a bucket of non-contiguous trades."""
    pnl = np.array([t.net_pnl for t in trades], dtype=np.float64)
    stats = _trade_stats(trades)
    return Metrics(
        total_return=float(pnl.sum() / equity) if equity else 0.0,
        cagr=0.0,
        sharpe=0.0,
        sortino=0.0,
        max_drawdown=0.0,
        calmar=0.0,
        num_trades=len(trades),
        exposure=0.0,
        turnover=0.0,
        **stats,
    )


def option_regime_robustness(
    trades: list[Trade],
    underlying: Bars,
    *,
    initial_equity: float = 100_000.0,
    min_trades: int = 5,
) -> RegimeReport:
    """Attribute a trade list to the regime SPY was in at entry, and ask
    whether the rule made money in more than one of them.

    ``regime_series`` classifies the *underlying* bars this engine already
    settles against, so no options-specific regime classifier is needed --
    the market SPY options are written on and the market ``features/regime.py``
    already classifies are the same series. Trades are attributed by entry
    time exactly the way ``validation.robustness.regime_robustness`` attributes
    equity trades; ``score_regimes`` is the same scoring function, reused
    rather than re-derived, so the two sides answer "more than one regime
    worked" the same way.
    """
    report = RegimeReport()
    if not trades:
        return report
    labels = regime_series(underlying)
    index = {int(t): labels[i] for i, t in enumerate(underlying.event_time)}

    buckets: dict[str, list[Trade]] = {}
    for trade in trades:
        label = index.get(trade.entry_time)
        if label is None:
            continue
        buckets.setdefault(label, []).append(trade)

    for label, bucket_trades in buckets.items():
        report.per_regime[label] = _metrics_from_trades(bucket_trades, initial_equity)
    report.scored_regimes = tuple(
        sorted(
            label
            for label, bucket_trades in buckets.items()
            if label != UNSCORED_REGIME and len(bucket_trades) >= min_trades
        )
    )
    report.score = score_regimes(buckets, min_trades)
    return report
