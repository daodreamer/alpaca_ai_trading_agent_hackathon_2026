"""What the sealed *option* run measures — specs/10-options-research.md D8.

[`validation/sealed.py`](../validation/sealed.py)'s counterpart, and it returns
that module's :class:`~aqr.validation.sealed.SealedMeasurement` unchanged. That
is deliberate and it is the same commitment specs/10 makes about the engine: the
sealed window is scored by the same object, with the same
:attr:`~aqr.validation.sealed.SealedMeasurement.can_confirm` of ``False``, the
same multiplicity bar and the same ``refuted`` rule, so an option verdict and an
equity verdict cannot quietly mean different things.

What differs is only what could not be shared:

**The backtest.** ``run_option_strategy`` rather than ``run_strategy``, over an
:class:`~aqr.options.run.OptionMarket` rather than a dict of bars.

**The benchmark.** Buy and hold on the one underlying, because there is no
universe. A short put spread is a levered, capped long position, so most of what
it earns is the drift it was exposed to — regressing on the underlying is the
only way to separate the rule from its exposure, and it is what
``backtest/alpha.py`` exists for.

**The embargo boundary is also a settlement boundary.** ``OptionBacktestConfig``
carries ``settle_before``, and in the sealed phase it is the sealed root's own
end rather than the research embargo (D3). A caller that left it at the default
would refuse every entry in the window it came to measure, which is why this
function takes the config and does not build one.

**And the sentence the window is entitled to.** 2024-09-01 → 2026-08-28 is about
25 independent 28-DTE cycles. It can say "this stopped working". It cannot say
"this works", and no artefact reading this measurement may word it otherwise.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np

from aqr.backtest.alpha import residual_alpha
from aqr.backtest.metrics import periods_per_year
from aqr.data.bars import Bars
from aqr.options.engine import OptionBacktestConfig
from aqr.options.run import OptionMarket, run_option_strategy
from aqr.options.spec import OptionSpec
from aqr.validation.cycles import independent_cycles
from aqr.validation.holdout import benchmark_returns
from aqr.validation.sealed import (
    MIN_SESSIONS,
    SealedMeasurement,
    _max_drawdown,
    _sharpe,
)

# ``_max_drawdown`` and ``_sharpe`` are imported rather than re-derived, and
# that is the point rather than an oversight: an option Sharpe computed with a
# different degrees-of-freedom convention than the equity one would make the two
# sealed measurements incomparable while looking identical in the report.
# Sharing the arithmetic is what makes ``SealedMeasurement`` mean one thing.

__all__ = ["measure_sealed_option_window"]


def measure_sealed_option_window(
    spec: OptionSpec,
    market: OptionMarket,
    *,
    since: datetime,
    config: OptionBacktestConfig | None = None,
    looks: int = 1,
) -> SealedMeasurement:
    """Score ``spec`` on the sessions from ``since`` onward, warmed by everything before.

    ``market`` must hold the full history, not just the embargoed span. The
    backtest runs over all of it and the measurement is clipped, which is the
    only arrangement in which the sealed window trades the same rule the search
    window validated — the equity version's own argument, and it is stronger
    here: a structure held to 28 days that was opened before the boundary
    settles inside it, and truncating the chain would silently drop those.

    ``looks`` is this reading's ordinal within the **option** family. A parameter
    rather than a lookup, for the same reason the clock is: this layer stays
    offline and deterministic, and the count lives in the registry, which is the
    caller's to query.
    """
    result = run_option_strategy(spec, market, config)
    boundary = int(since.timestamp())
    fingerprint = spec.fingerprint()

    equity = np.asarray(result.equity, dtype=np.float64)
    timeline = np.asarray(result.timeline, dtype=np.int64)
    if equity.size < 2:
        return _empty(spec, fingerprint, since, 0, "the backtest produced no equity curve", looks)

    # Returns are indexed by the session they land *in*, so a return belongs to
    # the window when its closing session does. Slicing the curve instead would
    # drop the first day of the window or keep the last day before it, depending
    # on which end the slice was taken from.
    with np.errstate(divide="ignore", invalid="ignore"):
        step = equity[1:] / equity[:-1] - 1.0
    step_time = timeline[1:]
    inside = step_time >= boundary

    strat = step[inside]
    strat_time = step_time[inside]
    if strat.size == 0:
        return _empty(
            spec,
            fingerprint,
            since,
            int(timeline.size),
            f"no sessions on or after {since.date().isoformat()}",
            looks,
        )

    series = benchmark_returns({spec.underlying: _from(market.underlying, boundary)})
    if series is None:
        return _empty(
            spec, fingerprint, since, int(timeline.size), "the benchmark could not be built", looks
        )
    sessions, bench = series
    # ``sessions`` has one more entry than ``bench``: the return at index i is
    # the move *into* session i + 1.
    bench_by_session = dict(zip(sessions[1:].tolist(), bench.tolist(), strict=True))
    paired = [
        (float(r), bench_by_session[int(day)])
        for r, day in zip(strat.tolist(), (strat_time // 86_400).tolist(), strict=True)
        if int(day) in bench_by_session
    ]
    strat_paired = np.array([a for a, _ in paired], dtype=np.float64)
    bench_paired = np.array([b for _, b in paired], dtype=np.float64)

    ppy = periods_per_year(timeline)
    observations = int(strat_paired.size)
    window_equity = equity[np.concatenate(([False], inside))]

    # Trades that *settled* inside the window, which is when an option position
    # produces its evidence. A structure opened in August 2024 and expiring in
    # September is a sealed-window result even though its entry is not, and the
    # equity version's ``exit_time`` filter says the same thing about a position
    # closed after the boundary.
    settled = [t for t in result.trades if int(t.exit_time) >= boundary]

    residual = (
        residual_alpha(strat_paired, bench_paired, periods_per_year=ppy)
        if observations >= MIN_SESSIONS
        else None
    )
    note = (
        ""
        if residual is not None
        else f"{observations} paired observations, below the {MIN_SESSIONS} the regression needs"
    )
    if residual is not None:
        cycles = independent_cycles(settled)
        note = (
            f"{cycles} independent cycles settled inside the window; specs/10 D8 -- this "
            "window can refute and cannot confirm"
        )

    return SealedMeasurement(
        strategy=spec.name,
        fingerprint=fingerprint,
        since=since,
        first_session=datetime.fromtimestamp(int(strat_time[0]), tz=UTC),
        last_session=datetime.fromtimestamp(int(strat_time[-1]), tz=UTC),
        observations=observations,
        backtest_sessions=int(timeline.size),
        strategy_return=float(np.prod(1.0 + strat_paired) - 1.0) if observations else 0.0,
        strategy_sharpe=_sharpe(strat_paired, ppy),
        max_drawdown=_max_drawdown(window_equity),
        benchmark_return=float(np.prod(1.0 + bench_paired) - 1.0) if observations else 0.0,
        benchmark_sharpe=_sharpe(bench_paired, ppy),
        trades=len(settled),
        residual=residual,
        note=note,
        looks=looks,
    )


def _from(bars: Bars, boundary: int) -> Bars:
    """The underlying from the boundary onward.

    The benchmark has to cover the same sessions the strategy is scored over and
    no more, or a two-year strategy return is compared against a seven-year
    index return and the difference is attributed to the rule.
    """
    at = int(np.searchsorted(bars.event_time, boundary, side="left"))
    return bars.slice(max(0, at - 1), len(bars))


def _empty(
    spec: OptionSpec,
    fingerprint: str,
    since: datetime,
    backtest_sessions: int,
    note: str,
    looks: int,
) -> SealedMeasurement:
    """A measurement that could not be made, saying so rather than reporting zeroes."""
    return SealedMeasurement(
        strategy=spec.name,
        fingerprint=fingerprint,
        since=since,
        first_session=None,
        last_session=None,
        observations=0,
        backtest_sessions=backtest_sessions,
        strategy_return=0.0,
        strategy_sharpe=0.0,
        max_drawdown=0.0,
        benchmark_return=0.0,
        benchmark_sharpe=0.0,
        trades=0,
        residual=None,
        note=note,
        looks=looks,
    )
