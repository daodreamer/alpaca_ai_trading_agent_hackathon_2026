"""What the sealed run measures, separated from the process that is allowed to run it.

The sealed run is one backtest and one regression. Nothing here is novel and
nothing here should be: the whole claim of the embargo protocol is that the
sealed window is scored by *the same code* that scored the search window, so
that the only difference between the two numbers is the data.

Two details carry all the weight.

**The bars start before the window; the measurement does not.** ``rs_rank(60)``
needs sixty sessions of history, and a strategy whose signal is unwarmed on its
first sixty sealed sessions is a different strategy than the one that was
validated. So the backtest is handed everything and only the *scoring* is
clipped to the embargoed span. The alternative -- truncating the bars -- silently
changes the rule.

**The benchmark is measured over the same window.** Comparing a two-year
strategy Sharpe against a nine-year benchmark Sharpe compares two market
regimes wearing one label, and the difference would be attributed to the rule.

And the limitation, which belongs on the object rather than in a report somebody
might not write: two years is roughly 500 sessions, the standard error of an
annualised Sharpe on 500 sessions is near 0.7, and therefore this measurement can
*refute* a candidate and cannot confirm one. :attr:`SealedMeasurement.can_confirm`
is False by construction. The evidence for a rule is the walk-forward inside the
search window; this is the check on whether it has since stopped working.

**The window can be read more than once, and the reading is counted.** The
one-shot rule is per candidate, not per window: a second, genuinely new
hypothesis gets its own sealed run, which is what makes an ongoing research loop
possible at all. But a window read seven times has screened seven candidates, and
the survivor of a seven-way screen is a weaker claim than the survivor of a
one-way screen -- the same multiple-comparisons problem as the search, moved up a
level. So every measurement carries :attr:`SealedMeasurement.looks`, and
:attr:`SealedMeasurement.significance_bar` raises the t an alpha must clear as
that count grows. Counting it is the whole defence; nothing here forbids the
seventh look.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from statistics import NormalDist
from typing import Any

import numpy as np

from aqr.backtest.alpha import SIGNIFICANCE_T, ResidualAlpha, residual_alpha
from aqr.backtest.engine import BacktestConfig
from aqr.backtest.metrics import periods_per_year
from aqr.backtest.run import run_strategy
from aqr.data.bars import Bars
from aqr.data.universes import PointInTimeUniverse
from aqr.dsl.schema import StrategySpec
from aqr.validation.holdout import benchmark_returns

__all__ = ["SealedMeasurement", "measure_sealed_window", "multiplicity_bar"]

MIN_SESSIONS = 60
"""Below this the regression is not attempted. Mirrors ``alpha.MIN_OBSERVATIONS``."""

FAMILY_WISE_ALPHA = 0.05
"""The error rate held across *all* readings of the window, not each one."""


def multiplicity_bar(looks: int) -> float:
    """The ``t`` an alpha must clear when the window has been read ``looks`` times.

    Bonferroni: hold the family-wise error at 5% across every candidate the
    sealed window has screened, rather than at 5% per candidate. Read it seven
    times at a per-look threshold and the chance of at least one false pass is
    about 30%, which is a screen that passes noise rather than a check on it.

    A normal quantile rather than a Student t. The window is around 500 sessions
    and the difference at that many degrees of freedom is under half a percent --
    far smaller than the standard error the measurement already carries, and not
    worth a scipy dependency.

    One look gives 1.96, which is the same threshold as
    :data:`~aqr.backtest.alpha.SIGNIFICANCE_T` (2.0) before rounding.
    """
    n = max(1, int(looks))
    return float(NormalDist().inv_cdf(1.0 - FAMILY_WISE_ALPHA / (2.0 * n)))


def _sharpe(returns: np.ndarray, ppy: float) -> float:
    if returns.size < 2:
        return 0.0
    sd = float(returns.std(ddof=1))
    if sd <= 0:
        return 0.0
    return float(returns.mean() / sd * np.sqrt(ppy))


def _max_drawdown(equity: np.ndarray) -> float:
    if equity.size == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    with np.errstate(divide="ignore", invalid="ignore"):
        drawdown = np.where(peak > 0, equity / peak - 1.0, 0.0)
    return float(drawdown.min())


@dataclass(frozen=True, slots=True)
class SealedMeasurement:
    """One pre-registered candidate, scored on the embargoed years. Once."""

    strategy: str
    fingerprint: str
    since: datetime
    """The embargo boundary. Nothing before it is scored."""
    first_session: datetime | None
    last_session: datetime | None
    observations: int
    """Paired strategy/benchmark returns inside the window. The sample size."""
    backtest_sessions: int
    """Sessions the backtest ran over, warm-up included. Always the larger number."""
    strategy_return: float
    strategy_sharpe: float
    max_drawdown: float
    benchmark_return: float
    benchmark_sharpe: float
    trades: int
    residual: ResidualAlpha | None
    """``None`` when the window cannot support the regression. Never a guess."""
    note: str = ""
    """Why the residual is absent, when it is."""
    looks: int = 1
    """How many times the embargoed window has been read, this run included.

    One-shot is per candidate, not per window: a genuinely new hypothesis gets
    its own sealed run, which is what makes a research loop possible. But the
    survivor of a seven-way screen is a weaker claim than the survivor of a
    one-way screen, and this is the number that says which one is being made.
    """

    @property
    def significance_bar(self) -> float:
        """The ``t`` this alpha must clear, given how many looks preceded it."""
        return multiplicity_bar(self.looks)

    @property
    def alpha_clears_bar(self) -> bool:
        """Did the alpha clear the multiplicity-adjusted threshold?

        Necessary, never sufficient -- :attr:`can_confirm` is still False. An
        alpha that fails this bar is one the sealed window cannot distinguish
        from the luckiest of ``looks`` draws; one that clears it has only earned
        the right not to be dismissed on that particular ground.
        """
        if self.residual is None:
            return False
        return self.residual.t_alpha >= self.significance_bar

    @property
    def sharpe_standard_error(self) -> float:
        """Roughly ``sqrt(periods_per_year / n)``: what one window can resolve.

        On 500 daily sessions this is near 0.7, which is larger than most of the
        Sharpe differences anybody argues about.
        """
        if self.observations < 2:
            return float("inf")
        return float(np.sqrt(252.0 / self.observations))

    @property
    def can_confirm(self) -> bool:
        """False, always, and it is a property rather than a comment on purpose.

        A single two-year window is too small to establish an edge at any
        plausible effect size. Anything that reads this object has to handle the
        answer being no.
        """
        return False

    @property
    def refuted(self) -> bool:
        """The rule did not merely fail to repeat -- it went the other way.

        Negative alpha that clears the significance threshold in the wrong
        direction. The asymmetry with :attr:`can_confirm` is the point: refuting
        needs one bad window, confirming needs many good ones.
        """
        if self.residual is None:
            return False
        return self.residual.alpha < 0 and self.residual.t_alpha <= -SIGNIFICANCE_T

    def summary(self) -> str:
        window = "no sessions"
        if self.first_session and self.last_session:
            window = (
                f"{self.first_session.date().isoformat()} -> "
                f"{self.last_session.date().isoformat()}"
            )
        lines = [
            f"{self.strategy} [{self.fingerprint}] sealed window {window} "
            f"({self.observations} sessions)",
            f"  strategy   return {self.strategy_return:+.2%}  "
            f"sharpe {self.strategy_sharpe:+.2f}  maxDD {self.max_drawdown:.2%}  "
            f"trades {self.trades}",
            f"  benchmark  return {self.benchmark_return:+.2%}  "
            f"sharpe {self.benchmark_sharpe:+.2f}",
        ]
        if self.residual is None:
            lines.append(f"  residual   not estimated -- {self.note}")
        else:
            r = self.residual
            lines.append(
                f"  residual   alpha {r.alpha:+.2%}/yr  beta {r.beta:.2f}  "
                f"t {r.t_alpha:+.2f}  IR {r.information_ratio:+.2f}  "
                f"R2 {r.r_squared:.2f}"
            )
        lines.append(
            f"  this window can refute and cannot confirm: the standard error on an "
            f"annualised Sharpe here is +/-{self.sharpe_standard_error:.2f}"
        )
        if self.looks > 1:
            lines.append(
                f"  look {self.looks} at this window: {self.looks} candidates have now "
                f"been screened against it, so the bar for the alpha is "
                f"t >= {self.significance_bar:.2f}, not {multiplicity_bar(1):.2f}"
            )
            if self.residual is not None and not self.alpha_clears_bar:
                lines.append(
                    f"  t {self.residual.t_alpha:+.2f} does not clear that bar: this "
                    "result is what the luckiest of these looks would produce anyway"
                )
        if self.refuted:
            lines.append("  REFUTED: alpha is negative and significant on the sealed years")
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "fingerprint": self.fingerprint,
            "since": self.since.isoformat(),
            "first_session": self.first_session.isoformat() if self.first_session else None,
            "last_session": self.last_session.isoformat() if self.last_session else None,
            "observations": self.observations,
            "backtest_sessions": self.backtest_sessions,
            "strategy_return": self.strategy_return,
            "strategy_sharpe": self.strategy_sharpe,
            "max_drawdown": self.max_drawdown,
            "benchmark_return": self.benchmark_return,
            "benchmark_sharpe": self.benchmark_sharpe,
            "trades": self.trades,
            "residual": self.residual.as_dict() if self.residual else None,
            "sharpe_standard_error": self.sharpe_standard_error,
            "can_confirm": self.can_confirm,
            "refuted": self.refuted,
            "looks": self.looks,
            "significance_bar": self.significance_bar,
            "alpha_clears_bar": self.alpha_clears_bar,
            "note": self.note,
        }


def measure_sealed_window(
    spec: StrategySpec,
    data: dict[str, Bars],
    *,
    since: datetime,
    membership: PointInTimeUniverse | None = None,
    config: BacktestConfig | None = None,
    looks: int = 1,
) -> SealedMeasurement:
    """Score ``spec`` on the sessions from ``since`` onward, warmed by everything before.

    ``data`` must contain the full history, not just the embargoed span. The
    backtest runs over all of it and the measurement is clipped, which is the
    only arrangement in which the sealed window trades the same rule the search
    window validated.

    ``looks`` is this reading's ordinal -- how many candidates the window has
    screened, this one included. A parameter rather than a lookup, for the same
    reason the clock is: this layer stays offline and deterministic, and the
    count lives in the registry, which is the caller's to query.
    """
    universe = {s: data[s] for s in spec.universe.symbols if s in data}
    if not universe:
        raise ValueError(f"no data for the universe {list(spec.universe.symbols)}")

    result = run_strategy(spec, universe, config, membership=membership)
    boundary = int(since.timestamp())

    equity = np.asarray(result.equity, dtype=np.float64)
    timeline = np.asarray(result.timeline, dtype=np.int64)
    fingerprint = spec.fingerprint()

    if equity.size < 2:
        return _empty(
            spec, fingerprint, since, 0, "the backtest produced no equity curve", looks
        )

    # Returns are indexed by the session they land *in*, so a return belongs to
    # the window when its closing session does. Slicing the equity curve instead
    # would drop the first day of the window or keep the last day before it,
    # depending on which end the slice was taken from.
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

    series = benchmark_returns(universe)
    if series is None:
        return _empty(
            spec,
            fingerprint,
            since,
            int(timeline.size),
            "the benchmark could not be built",
            looks,
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
    first = datetime.fromtimestamp(int(strat_time[0]), tz=UTC)
    last = datetime.fromtimestamp(int(strat_time[-1]), tz=UTC)

    window_equity = equity[np.concatenate(([False], inside))]
    residual = (
        residual_alpha(strat_paired, bench_paired, periods_per_year=ppy)
        if observations >= MIN_SESSIONS
        else None
    )
    note = (
        ""
        if residual is not None
        else (
            f"{observations} paired observations, below the {MIN_SESSIONS} "
            "the regression needs"
        )
    )

    return SealedMeasurement(
        strategy=spec.name,
        fingerprint=fingerprint,
        since=since,
        first_session=first,
        last_session=last,
        observations=observations,
        backtest_sessions=int(timeline.size),
        strategy_return=float(np.prod(1.0 + strat_paired) - 1.0) if observations else 0.0,
        strategy_sharpe=_sharpe(strat_paired, ppy),
        max_drawdown=_max_drawdown(window_equity),
        benchmark_return=float(np.prod(1.0 + bench_paired) - 1.0) if observations else 0.0,
        benchmark_sharpe=_sharpe(bench_paired, ppy),
        trades=sum(1 for t in result.trades if int(t.exit_time) >= boundary),
        residual=residual,
        note=note,
        looks=looks,
    )


def _empty(
    spec: StrategySpec,
    fingerprint: str,
    since: datetime,
    backtest_sessions: int,
    note: str,
    looks: int = 1,
) -> SealedMeasurement:
    """A measurement that could not be made, saying so rather than reporting zeroes.

    Every field is zero and ``residual`` is None. A plausible-looking figure
    nobody can check is worse than an absent one.
    """
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
