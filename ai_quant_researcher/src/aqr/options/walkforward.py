"""Walk-forward, on calendar time — specs/10-options-research.md D8.

``validation/walkforward.py``'s ``TRAIN_BARS`` / ``TEST_BARS`` (504 / 126) are
denominated in *bars*, and D0 is explicit about why that unit fails here: the
option chain's session grid is not the bar grid. 753 sessions cover 5.55
years at a cadence that is weekly in 2019 and Mon/Wed/Fri from 2020 on, so 504
of them is a different amount of calendar time depending on which 504 a fold
happens to land on. Worse, a bar-counted fold does not even see the thing
that actually limits this data: D8 measured **71** non-overlapping 28-DTE
cycles across the whole research window, not 753 tradable sessions, because a
structure is held to expiry and two entries that overlap are not two
independent bets. A fold defined in bars can hold three real cycles and call
itself a fifth of a five-year study.

So folds are cut on the calendar instead, and every fold reports the cycle
count next to the trade count -- D8's own words -- so a reader never has to
take "126 bars of test data" on faith when the honest unit is "this many
independent cycles, and here they are."

**Geometry.** Two years train, six months test, stepping six months: a fixed
2-year window slides across the research span rather than growing from a
fixed start, so a fold from 2023 is judged with the same two years of prior
history as a fold from 2020, not with everything back to 2019. Over 2019-02-09
.. 2024-08-30 that produces on the order of 7-8 folds, each covering a test
window with roughly a tenth of the whole window's 71 independent cycles --
already a small number, and the report says exactly how small for the spec
this run actually used, never a number borrowed from D0's table for a
different structure or a different DTE target.

**What makes a fold's number honest.** The chain is what is sliced, not the
result. ``ChainIndex.slice_dates`` restricts which *sessions* the engine may
enter from before the engine ever runs, so a position from the train half of
a fold's own window cannot be attributed to the test half by construction --
see ``ChainIndex.slice_dates``'s own docstring for the failure this replaces.
The underlying and the volatility history are *not* sliced per fold: both are
already causal at the decision bar (specs/10 D2, D6 -- proven by
``tests/test_causality.py`` and ``tests/test_option_features.py``'s
``TestNoLookAhead``), so widening either one past a fold's own boundary lets a
position that fold *did* open find the real close it settles against, and it
lets nothing else through, because no feature this engine can evaluate reads
a bar after its own decision index. The one thing sliced is the one thing
that gates *when* an entry can happen.

**A documented approximation.** ``stitched`` chains each fold's own test
segment by return, the same technique ``validation/walkforward.py`` uses for
the equity side. Two fold boundaries can be closer together than a
structure's own DTE, so a position opened near the end of one fold's test
window can still be marked-to-model a few sessions into the next fold's
calendar span before it settles -- both segments are honest on their own
(each is this fold's *own* re-run, entries gated to this fold alone), the
approximation is only in treating the chained curve as one continuous
allocator experience across that handful of overlapping sessions. Reported
here rather than hidden: it is a curve-continuity wrinkle, not a look-ahead
leak, and it is bounded by one DTE window (specs/10 caps DTE at 66 days) out
of a 6-month test span.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any

import numpy as np

from aqr.backtest.engine import BacktestResult, Trade
from aqr.backtest.metrics import MIN_TRADES_FOR_RATIOS, Metrics, compute_metrics
from aqr.data.bars import Bars
from aqr.options.engine import (
    OptionBacktestConfig,
    OptionBacktestResult,
    OptionTrade,
    SkipCensus,
    SkippedEntry,
    affordability_bound_fraction,
)
from aqr.options.run import OptionMarket, run_option_strategy
from aqr.options.spec import OptionSpec
from aqr.validation.cycles import independent_cycles

__all__ = [
    "STEP_MONTHS",
    "TEST_MONTHS",
    "TRAIN_MONTHS",
    "OptionFold",
    "OptionFoldResult",
    "OptionWalkForwardReport",
    "option_walk_forward_folds",
    "run_option_walk_forward",
]

TRAIN_MONTHS = 24
TEST_MONTHS = 6
STEP_MONTHS = 6
"""specs/10 D8's suggested geometry, adopted: a fixed two-year fit window, a
six-month judgement window, stepping by the judgement window's own width so
consecutive test windows tile the calendar with no gap and no overlap. Over
the 5.55-year research window this yields roughly seven or eight folds --
computed and reported per run rather than pinned as a constant, because it
depends on the actual first and last chain session, which the sealed window
and the research window do not share."""

_MIN_FOLD_TEST_DAYS = 30
"""A test window trimmed by the data's own end to under a month is a stub:
reporting it as a fold gives it the same visual weight as a full six-month
window while holding perhaps one cycle. Dropped rather than padded."""


def _add_months(d: date, months: int) -> date:
    """Calendar-month arithmetic, clamped to the shorter month's last day.

    ``date`` has no month-add of its own and this project has no calendar
    dependency to reach for one; the clamp is the only subtlety -- 2024-08-31
    plus one month is 2024-09-30, not an overflow into October.
    """
    total = d.month - 1 + months
    year = d.year + total // 12
    month = total % 12 + 1
    next_month = date(year + (1 if month == 12 else 0), 1 if month == 12 else month + 1, 1)
    days_in_month = (next_month - date(year, month, 1)).days
    return date(year, month, min(d.day, days_in_month))


def _midnight(d: date) -> datetime:
    return datetime(d.year, d.month, d.day, tzinfo=UTC)


@dataclass(frozen=True, slots=True)
class OptionFold:
    """One walk-forward step, cut on the calendar rather than on bar position."""

    index: int
    train_start: date
    train_stop: date
    test_start: date
    test_stop: date

    def __str__(self) -> str:
        return (
            f"fold {self.index}: train {self.train_start}..{self.train_stop} -> "
            f"test {self.test_start}..{self.test_stop}"
        )


def option_walk_forward_folds(
    first: date,
    last: date,
    *,
    train_months: int = TRAIN_MONTHS,
    test_months: int = TEST_MONTHS,
    step_months: int = STEP_MONTHS,
) -> list[OptionFold]:
    """Rolling, fixed-width calendar folds over ``[first, last)``.

    ``step_months`` equal to ``test_months`` (the default) is what makes
    consecutive test windows partition the span exactly: fold *i*'s test
    window ends precisely where fold *i+1*'s begins, so summing cycles or
    trades across every fold's test window double counts nothing and skips
    nothing, as long as an entry session actually falls in ``[first, last)``.
    A caller after a different property -- overlapping test windows, a
    growing rather than rolling train window -- can still get it by passing a
    different ``step_months`` or widening ``train_months``` fold by fold
    outside this function; the default here is D8's own suggestion.
    """
    if train_months < 1 or test_months < 1 or step_months < 1:
        raise ValueError("train_months, test_months and step_months must all be >= 1")
    if last <= first:
        raise ValueError(f"last {last} must be after first {first}")

    folds: list[OptionFold] = []
    train_start = first
    index = 0
    while True:
        train_stop = _add_months(train_start, train_months)
        test_start = train_stop
        test_stop = min(_add_months(test_start, test_months), last)
        if test_start >= last or (test_stop - test_start).days < _MIN_FOLD_TEST_DAYS:
            break
        folds.append(OptionFold(index, train_start, train_stop, test_start, test_stop))
        index += 1
        train_start = _add_months(train_start, step_months)
    return folds


@dataclass(slots=True)
class OptionFoldResult:
    fold: OptionFold
    train: Metrics
    test: Metrics
    train_trades: list[Trade] = field(default_factory=list)
    test_trades: list[Trade] = field(default_factory=list)
    train_cycles: int = 0
    test_cycles: int = 0
    """Independent cycles per :func:`aqr.validation.cycles.independent_cycles`
    -- the one function the evaluator gates on, so a reader never sees this
    report and the evaluator's own reasons disagree about what a cycle is."""
    train_skip_census: SkipCensus = field(default_factory=SkipCensus)
    test_skip_census: SkipCensus = field(default_factory=SkipCensus)
    """D8a, bucketed to this fold's own window by :func:`_bucket_skips`. The
    test half is what :attr:`OptionWalkForwardReport.oos_skip_census` pools
    across folds; the train half is kept for the same reason ``train_cycles``
    is -- a rule can be affordability-bound on its fit window and not on its
    judgement window, or the reverse, and a report that only carried one half
    could not tell which."""

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.fold.index,
            "train_start": self.fold.train_start.isoformat(),
            "train_stop": self.fold.train_stop.isoformat(),
            "test_start": self.fold.test_start.isoformat(),
            "test_stop": self.fold.test_stop.isoformat(),
            "train_trades": self.train.num_trades,
            "train_cycles": self.train_cycles,
            "test_trades": self.test.num_trades,
            "test_cycles": self.test_cycles,
            "test_return": self.test.total_return,
            "test_sharpe": self.test.sharpe,
            "test_skip_census": self.test_skip_census.as_dict(),
            "test_affordability_bound_fraction": affordability_bound_fraction(
                self.test_skip_census, self.test.num_trades
            ),
        }

    @property
    def test_sharpe_measured(self) -> bool:
        """Whether this fold has enough trades for ``compute_metrics`` to have
        computed a ratio at all.

        Below :data:`~aqr.backtest.metrics.MIN_TRADES_FOR_RATIOS` it returns
        0.0, and that zero means *declined* rather than *flat*. A six-month
        options fold routinely holds two trades -- a structure held to expiry
        produces evidence only when it closes (specs/10 D8) -- so this is the
        common case here rather than an edge one, and a report that printed
        "0.00" seven times beside a stitched "0.96" would read as a
        contradiction when it is a refusal.
        """
        return self.test.num_trades >= MIN_TRADES_FOR_RATIOS

    def __str__(self) -> str:
        sharpe = (
            f"sharpe {self.test.sharpe:.2f}"
            if self.test_sharpe_measured
            else f"sharpe n/a ({self.test.num_trades} trades, under the "
            f"{MIN_TRADES_FOR_RATIOS}-trade ratio floor)"
        )
        return (
            f"{self.fold}: train {self.train.num_trades} trades/{self.train_cycles} cycles "
            f"-> test {self.test.num_trades} trades/{self.test_cycles} cycles, "
            f"ret {self.test.total_return:+.1%}, {sharpe}"
        )


@dataclass(slots=True)
class OptionWalkForwardReport:
    strategy: str
    fingerprint: str
    underlying: str
    train_months: int
    test_months: int
    step_months: int
    folds: list[OptionFoldResult] = field(default_factory=list)
    stitched: Metrics | None = None
    stitched_equity: np.ndarray | None = None
    stitched_timeline: np.ndarray | None = None

    @property
    def oos_trades(self) -> list[Trade]:
        return [trade for fold in self.folds for trade in fold.test_trades]

    @property
    def oos_cycles(self) -> int:
        """The number the evaluator's gate reads (D8: independent cycles >= 25).

        Recomputed over the pooled out-of-sample trade list rather than summed
        per fold: two cycles from adjacent folds can still overlap in holding
        period near a fold boundary (a position opened at the very end of one
        fold's test window and one opened at the very start of the next), and
        summing each fold's own count would double count that overlap instead
        of resolving it the way :func:`independent_cycles` does everywhere
        else.
        """
        return independent_cycles(self.oos_trades)

    @property
    def oos_sharpe(self) -> float:
        """Sharpe of the chained test curve; falls back to the fold mean only
        when no fold produced a chainable segment. Mirrors
        ``WalkForwardReport.oos_sharpe``'s reasoning on the equity side: a
        mean over folds is measurably higher than the continuous curve's own
        Sharpe because it discards what happens at each fold boundary."""
        if self.stitched is not None:
            return self.stitched.sharpe
        if not self.folds:
            return 0.0
        return float(np.mean([f.test.sharpe for f in self.folds]))

    @property
    def oos_skip_census(self) -> SkipCensus:
        """D8a's census, pooled the same way ``oos_trades`` is: a plain sum
        of each fold's own ``test_skip_census`` rather than a recount over a
        re-concatenated entry list.

        A skip belongs to exactly one session, and D8's default geometry
        (``step_months == test_months``) tiles the test windows with no
        overlap, so summing the per-fold counts and recounting a pooled list
        agree. They are not required to agree under a non-default,
        overlapping geometry -- but neither is ``len(self.oos_trades)``
        above, which has the identical caveat and is the existing precedent
        for pooling by concatenation rather than by re-deriving overlap
        resolution a second way.
        """
        return SkipCensus(
            affordability=sum(f.test_skip_census.affordability for f in self.folds),
            no_leg_or_wing=sum(f.test_skip_census.no_leg_or_wing for f in self.folds),
            no_expiry_in_band=sum(f.test_skip_census.no_expiry_in_band for f in self.folds),
            greek_consistency=sum(f.test_skip_census.greek_consistency for f in self.folds),
            embargo_refusal=sum(f.test_skip_census.embargo_refusal for f in self.folds),
            stale_underlying=sum(f.test_skip_census.stale_underlying for f in self.folds),
        )

    @property
    def oos_affordability_bound_fraction(self) -> float:
        """D8a's fraction, measured over the pooled out-of-sample sample this
        report's own OOS metrics were computed from -- the same function
        ``options/engine.py`` defines, so a reader of this report and a
        reader of one fold's own :class:`OptionFoldResult` can never see two
        different definitions of "affordability-bound" for the same run."""
        return affordability_bound_fraction(self.oos_skip_census, len(self.oos_trades))

    @property
    def positive_fold_rate(self) -> float:
        if not self.folds:
            return 0.0
        return float(np.mean([f.test.total_return > 0 for f in self.folds]))

    @property
    def mean_gap(self) -> float:
        if not self.folds:
            return 0.0
        return float(np.mean([f.train.sharpe - f.test.sharpe for f in self.folds]))

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "fingerprint": self.fingerprint,
            "underlying": self.underlying,
            "geometry": {
                "train_months": self.train_months,
                "test_months": self.test_months,
                "step_months": self.step_months,
            },
            "folds": [f.as_dict() for f in self.folds],
            "oos_trades": len(self.oos_trades),
            "oos_cycles": self.oos_cycles,
            "oos_sharpe": self.oos_sharpe,
            "positive_fold_rate": self.positive_fold_rate,
            "mean_train_test_gap": self.mean_gap,
            "stitched": self.stitched.as_dict() if self.stitched else None,
            "oos_skip_census": self.oos_skip_census.as_dict(),
            "oos_affordability_bound_fraction": self.oos_affordability_bound_fraction,
        }

    def __str__(self) -> str:
        head = (
            f"{self.strategy} on {self.underlying}: {len(self.folds)} folds "
            f"({self.train_months}mo train / {self.test_months}mo test / "
            f"{self.step_months}mo step), {len(self.oos_trades)} OOS trades / "
            f"{self.oos_cycles} independent cycles, OOS sharpe {self.oos_sharpe:.2f}, "
            f"{self.positive_fold_rate:.0%} folds positive, "
            f"{self.oos_affordability_bound_fraction:.0%} of OOS skips affordability-bound (D8a)"
        )
        return "\n".join([head, *(f"  {f}" for f in self.folds)])


def _bucket(
    result: OptionBacktestResult, window_start: date, window_stop: date
) -> tuple[list[OptionTrade], list[Trade]]:
    """The subset of one fold's own trades entered in ``[window_start, window_stop)``.

    ``option_trades`` and ``trades`` are built together, one generic
    :class:`~aqr.backtest.engine.Trade` per :class:`OptionTrade`, in the same
    order (``options/engine.py``'s ``_as_trade`` comprehension) -- zipping
    them is what lets this filter on the option-side ``entry_session`` date,
    which is unambiguous, rather than reconstructing a date from the generic
    view's epoch ``entry_time``.
    """
    options: list[OptionTrade] = []
    trades: list[Trade] = []
    for option, trade in zip(result.option_trades, result.trades, strict=True):
        if window_start <= option.entry_session < window_stop:
            options.append(option)
            trades.append(trade)
    return options, trades


def _bucket_skips(
    result: OptionBacktestResult, window_start: date, window_stop: date
) -> list[SkippedEntry]:
    """The subset of one fold's own skips whose *session* falls in the window --
    specs/10 D8a's counterpart to :func:`_bucket`.

    Skips are attributed by the same field trades are (``entry_session`` /
    ``session``, both the calendar date the engine's own loop was standing on
    when it decided), so a fold's affordability fraction is measured over
    exactly the sessions that fold's own re-run offered a chance to trade --
    not the full-window run's census, which was sized against a different
    equity trajectory and a different train/test split and would report a
    fraction this fold's own trade count does not explain.
    """
    return [entry for entry in result.skips if window_start <= entry.session < window_stop]


def _trimmed_result(
    spec: OptionSpec,
    underlying: Bars,
    full: OptionBacktestResult,
    window_start: date,
    options: list[OptionTrade],
    trades: list[Trade],
) -> BacktestResult:
    """One window's own slice of a fold's equity curve, for ``compute_metrics``.

    Trimmed to ``[window_start, last settlement in this window]`` rather than
    to the window's own nominal stop: a position opened near the end of a
    window is still this window's trade, and its marked path and its
    settlement both belong in the metric that judges this window, even though
    both land after the window's calendar end. Trimmed forward at all, rather
    than left to run to the end of the dataset, because ``full``'s equity
    array does not reset when the window ends -- it just goes flat, and a flat
    tail spanning years would dilute every ratio in ``compute_metrics`` with a
    return of exactly zero that this window never earned.
    """
    start_idx = underlying.index_of(_midnight(window_start))
    if options:
        last_expiry = max(o.expiration for o in options)
        stop_idx = underlying.index_of(_midnight(last_expiry)) + 1
    else:
        stop_idx = start_idx
    stop_idx = max(stop_idx, start_idx)
    return BacktestResult(
        spec_fingerprint=spec.fingerprint(),
        strategy=spec.name,
        symbols=(spec.underlying,),
        timeline=full.timeline[start_idx:stop_idx],
        equity=full.equity[start_idx:stop_idx],
        trades=trades,
        warmup_bars=0,
    )


def run_option_walk_forward(
    spec: OptionSpec,
    market: OptionMarket,
    config: OptionBacktestConfig | None = None,
    *,
    train_months: int = TRAIN_MONTHS,
    test_months: int = TEST_MONTHS,
    step_months: int = STEP_MONTHS,
) -> OptionWalkForwardReport:
    """Walk ``spec`` forward over ``market``'s chain, on calendar folds.

    No parameter search: each fold re-runs the same, fixed ``spec`` -- D8's
    apparatus is about measuring whether a rule's out-of-sample behaviour
    holds up through calendar time, not about re-fitting it every six months
    on a budget of twenty hypotheses total (D8's search cap). A caller that
    wants a fitted walk-forward can build one from the pieces here the way
    ``validation/walkforward.py`` does for equities; this function answers the
    question D8 actually asks.
    """
    config = config or OptionBacktestConfig()
    report = OptionWalkForwardReport(
        strategy=spec.name,
        fingerprint=spec.fingerprint(),
        underlying=spec.underlying,
        train_months=train_months,
        test_months=test_months,
        step_months=step_months,
    )
    first, last = market.chain.first(), market.chain.last()
    if first is None or last is None:
        return report

    folds = option_walk_forward_folds(
        first, last, train_months=train_months, test_months=test_months, step_months=step_months
    )
    segments: list[np.ndarray] = []
    stamps: list[np.ndarray] = []

    for fold in folds:
        fold_chain = market.chain.slice_dates(fold.train_start, fold.test_stop)
        fold_result = run_option_strategy(spec, market.with_chain(fold_chain), config)

        train_options, train_trades = _bucket(fold_result, fold.train_start, fold.test_start)
        test_options, test_trades = _bucket(fold_result, fold.test_start, fold.test_stop)
        train_bt = _trimmed_result(
            spec, market.underlying, fold_result, fold.train_start, train_options, train_trades
        )
        test_bt = _trimmed_result(
            spec, market.underlying, fold_result, fold.test_start, test_options, test_trades
        )
        train_skips = _bucket_skips(fold_result, fold.train_start, fold.test_start)
        test_skips = _bucket_skips(fold_result, fold.test_start, fold.test_stop)

        report.folds.append(
            OptionFoldResult(
                fold=fold,
                train=compute_metrics(train_bt),
                test=compute_metrics(test_bt),
                train_trades=train_trades,
                test_trades=test_trades,
                train_cycles=independent_cycles(train_trades),
                test_cycles=independent_cycles(test_trades),
                train_skip_census=SkipCensus.from_entries(train_skips),
                test_skip_census=SkipCensus.from_entries(test_skips),
            )
        )
        if test_bt.equity.size > 1:
            segments.append(test_bt.equity)
            stamps.append(test_bt.timeline)

    stitched = _stitch(segments, stamps, config.initial_equity)
    if stitched is not None:
        timeline, equity = stitched
        report.stitched_timeline = timeline
        report.stitched_equity = equity
        report.stitched = compute_metrics(
            BacktestResult(
                spec_fingerprint=spec.fingerprint(),
                strategy=spec.name,
                symbols=(spec.underlying,),
                timeline=timeline,
                equity=equity,
                trades=report.oos_trades,
                warmup_bars=0,
            )
        )
    return report


def _stitch(
    segments: list[np.ndarray], times: list[np.ndarray], initial_equity: float
) -> tuple[np.ndarray, np.ndarray] | None:
    """Chain each fold's own test segment by return, not by level.

    The same technique ``validation/walkforward.py``'s private ``_stitch``
    uses for the equity side, duplicated rather than imported: what it does is
    generic numpy, the interesting property (each segment is a fold-local,
    leak-free re-run) lives in :func:`run_option_walk_forward` and
    ``ChainIndex.slice_dates`` instead, and importing a private, underscored
    name across modules would tie this file to that one's internals for a
    dozen lines of arithmetic.
    """
    returns: list[np.ndarray] = []
    marks: list[np.ndarray] = []
    for equity, timeline in zip(segments, times, strict=True):
        if equity.size < 2:
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            step = np.diff(equity) / np.where(equity[:-1] == 0, np.nan, equity[:-1])
        good = np.isfinite(step)
        returns.append(step[good])
        marks.append(timeline[1:][good])
    if not returns:
        return None
    chained = np.concatenate(returns)
    stamps = np.concatenate(marks)
    if chained.size == 0:
        return None
    curve = initial_equity * np.cumprod(1.0 + chained)
    curve = np.insert(curve, 0, initial_equity)
    lead = stamps[0] - (stamps[1] - stamps[0] if stamps.size > 1 else 86_400)
    return np.insert(stamps, 0, lead), curve
