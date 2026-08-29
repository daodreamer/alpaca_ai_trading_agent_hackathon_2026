"""Membership reaching the rest of the gauntlet.

`run_portfolio` takes a membership table; nothing else does. That is enough to
run one backtest correctly and not enough to evaluate a strategy, because every
number the verdict rests on comes from somewhere else: walk-forward folds,
parameter perturbations, leave-one-out asset robustness, the regime split. Each
of those re-runs the strategy, and a re-run without the table is a re-run on a
survivorship-biased universe.

The failure would not look like a failure. The headline backtest would be
correct, the folds would quietly hold names that had left the index, and the
out-of-sample Sharpe -- the number the score leans on hardest -- would be the
contaminated one.

So it threads through, and the test that matters is the boring one: the same
strategy evaluated with and without the table must not produce the same
out-of-sample number.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import numpy as np

from aqr.backtest.costs import ZERO_COST, CostModel
from aqr.backtest.engine import BacktestConfig
from aqr.backtest.run import run_strategy
from aqr.data.bars import Bars
from aqr.data.universes import Interval, Membership, PointInTimeUniverse
from aqr.dsl.schema import spec_from_dict
from aqr.pipeline import evaluate_candidate
from aqr.seal import current as current_seal
from aqr.validation.robustness import (
    MIN_PARALLEL_DELETIONS,
    AssetReport,
    _leave_one_out,
    _worker_deletion,
    _worker_init,
    asset_robustness,
)
from aqr.validation.walkforward import _window, run_walk_forward

SYMBOLS = [f"S{i:02d}" for i in range(10)]
N = 1000
T0 = int(datetime(2016, 1, 4, tzinfo=UTC).timestamp())


def _data() -> dict[str, Bars]:
    rng = np.random.default_rng(5)
    t = np.arange(T0, T0 + N * 86_400, 86_400, dtype=np.int64)
    out: dict[str, Bars] = {}
    for i, s in enumerate(SYMBOLS):
        steps = rng.normal(0.0004, 0.013, N) + np.sin(
            np.arange(N) * 2 * np.pi / 80.0 + i
        ) * 0.0014
        close = 100.0 * np.exp(np.cumsum(steps))
        out[s] = Bars(
            symbol=s,
            timeframe="1D",
            event_time=t,
            open=close * 0.999,
            high=close * 1.011,
            low=close * 0.989,
            close=close,
            volume=np.full(N, 2e6),
        )
    return out


def _day(step: int) -> date:
    return datetime.fromtimestamp(T0 + step * 86_400, tz=UTC).date()


def _membership() -> PointInTimeUniverse:
    """Half the universe leaves at the midpoint. Crude, and the point is only
    that the gate has something to bite on."""
    members = []
    for i, s in enumerate(SYMBOLS):
        span = (
            (Interval(_day(0), _day(N // 2)),)
            if i % 2 == 0
            else (Interval(_day(0), None),)
        )
        members.append(Membership(s, s, span))
    return PointInTimeUniverse(
        name="probe", source="test", window=(_day(0), _day(N - 1)), members=tuple(members)
    )


def _spec():
    return spec_from_dict(
        {
            "strategy": {
                "name": "xs_probe",
                "mode": "portfolio",
                "rank_by": "roc(40) - roc(5)",
                "hold": 3,
                "rebalance_every": 10,
                "universe": {"symbols": SYMBOLS, "timeframe": "1D"},
            }
        }
    )


CONFIG = BacktestConfig(initial_equity=1_000_000.0, allow_fractional_shares=True)
ZERO_FREE = BacktestConfig(
    initial_equity=1_000_000.0, allow_fractional_shares=True, costs=ZERO_COST
)


# --------------------------------------------------------------------------


def test_run_strategy_forwards_membership() -> None:
    with_gate = run_strategy(_spec(), _data(), CONFIG, membership=_membership())
    without = run_strategy(_spec(), _data(), CONFIG)
    assert not np.allclose(with_gate.equity, without.equity)


def test_a_signal_spec_ignores_membership_without_complaining() -> None:
    """The event-driven engine has no membership concept yet. Passing a table it
    cannot use must not be an error -- the pipeline passes one for every spec."""
    spec = spec_from_dict(
        {
            "strategy": {
                "name": "trigger",
                "entry": "close > ema(20)",
                "universe": {"symbols": SYMBOLS, "timeframe": "1D"},
            }
        }
    )
    result = run_strategy(spec, _data(), CONFIG, membership=_membership())
    assert result.equity.size > 0


def test_walk_forward_folds_respect_membership() -> None:
    """The out-of-sample number the score leans on hardest."""
    kwargs = dict(train_bars=400, test_bars=200, config=CONFIG)
    gated = run_walk_forward(_spec(), _data(), membership=_membership(), **kwargs)
    plain = run_walk_forward(_spec(), _data(), **kwargs)
    assert gated.folds and plain.folds
    assert [f.test.sharpe for f in gated.folds] != [f.test.sharpe for f in plain.folds]


def test_leave_one_out_respects_membership() -> None:
    gated = asset_robustness(_spec(), _data(), config=CONFIG, membership=_membership())
    plain = asset_robustness(_spec(), _data(), config=CONFIG)
    assert gated.per_symbol.keys() == plain.per_symbol.keys()
    assert any(
        gated.per_symbol[s].total_return != plain.per_symbol[s].total_return
        for s in gated.per_symbol
    )


def test_the_pipeline_carries_it_end_to_end() -> None:
    gated = evaluate_candidate(
        _spec(), _data(), config=CONFIG, train_bars=400, test_bars=200,
        membership=_membership(),
    )
    plain = evaluate_candidate(
        _spec(), _data(), config=CONFIG, train_bars=400, test_bars=200,
    )
    assert gated.evaluation is not None and plain.evaluation is not None
    assert gated.evaluation.components != plain.evaluation.components


# --------------------------------------------------------------------------
# Which absences a fold is allowed to survive
# --------------------------------------------------------------------------
#
# Every fold used to be discarded if any one name in the universe had no bars in
# its window. On a fixed ten-name universe that is right: a missing name is a
# data hole. On a universe whose membership is a function of the date it is
# catastrophic -- a name that had not listed yet, or had already been acquired,
# is absent by construction. On the 680-name point-in-time S&P universe it threw
# away ten folds out of eleven, and every strategy scored a `positive_fold_rate`
# of exactly 9%: the number was measuring the calendar, not the rule.


def _bars_between(symbol: str, first_step: int, stop_step: int) -> Bars:
    """One symbol that trades only over ``[first_step, stop_step)``.

    Used for the two shapes a point-in-time universe is full of: a name that
    lists part-way through, and a name whose series stops for good.
    """
    n = stop_step - first_step
    rng = np.random.default_rng(11)
    t = np.arange(
        T0 + first_step * 86_400, T0 + stop_step * 86_400, 86_400, dtype=np.int64
    )
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.013, n)))
    return Bars(
        symbol=symbol,
        timeframe="1D",
        event_time=t,
        open=close * 0.999,
        high=close * 1.011,
        low=close * 0.989,
        close=close,
        volume=np.full(n, 2e6),
    )


def _bars_ending_at(symbol: str, stop_step: int) -> Bars:
    return _bars_between(symbol, 0, stop_step)


class TestAbsenceInAFold:
    def test_a_name_that_was_not_a_member_may_be_absent(self) -> None:
        """The bug that made every fold rate 9%. `LATE` joins the index only at
        the very end, so it is absent from the early folds by construction."""
        listed = N - 60
        data = _data()
        data["LATE"] = _bars_between("LATE", listed, N)
        symbols = [*SYMBOLS, "LATE"]
        spec = spec_from_dict(
            {
                "strategy": {
                    "name": "late_joiner",
                    "mode": "portfolio",
                    "rank_by": "roc(40) - roc(5)",
                    "hold": 3,
                    "rebalance_every": 10,
                    "universe": {"symbols": symbols, "timeframe": "1D"},
                }
            }
        )
        members = [
            Membership(s, s, (Interval(_day(0), None),)) for s in SYMBOLS
        ]
        members.append(Membership("LATE", "LATE", (Interval(_day(listed), None),)))
        table = PointInTimeUniverse(
            name="probe",
            source="test",
            window=(_day(0), _day(N - 1)),
            members=tuple(members),
        )
        report = run_walk_forward(
            spec, data, train_bars=300, test_bars=100, config=CONFIG, membership=table
        )
        assert len(report.folds) >= 3
        # Every fold before LATE listed must still run. Under the old rule they
        # were all discarded, because LATE had no bars to be sliced.
        assert all(f.test.num_trades > 0 for f in report.folds)

    def test_a_member_with_no_bars_is_still_a_hard_failure(self) -> None:
        """The half of the check worth keeping. `GHOST` is a member throughout
        and has no bars at all, which is a data hole and not a calendar fact."""
        data = _data()
        symbols = [*SYMBOLS, "GHOST"]
        spec = spec_from_dict(
            {
                "strategy": {
                    "name": "ghost",
                    "mode": "portfolio",
                    "rank_by": "roc(40) - roc(5)",
                    "hold": 3,
                    "rebalance_every": 10,
                    "universe": {"symbols": symbols, "timeframe": "1D"},
                }
            }
        )
        members = [Membership(s, s, (Interval(_day(0), None),)) for s in symbols]
        table = PointInTimeUniverse(
            name="probe",
            source="test",
            window=(_day(0), _day(N - 1)),
            members=tuple(members),
        )
        report = run_walk_forward(
            spec, data, train_bars=300, test_bars=100, config=CONFIG, membership=table
        )
        # Every fold refuses, so nothing trades and nothing is stitched.
        assert all(f.test.num_trades == 0 for f in report.folds)

    def test_a_delisted_name_may_be_absent_even_while_the_table_calls_it_a_member(
        self,
    ) -> None:
        """ALXN, CXO, ETFC, FLIR, NBL, TIF and VAR, in miniature.

        A stock stops trading when its merger closes; the index records the
        removal one to ten days later. For those few days the table says
        "member" and there is nothing to buy, so the bars, not the table, are
        what explain the absence.
        """
        stop = N // 2
        data = _data()
        data["ACQ"] = _bars_ending_at("ACQ", stop)
        symbols = [*SYMBOLS, "ACQ"]
        spec = spec_from_dict(
            {
                "strategy": {
                    "name": "acquired",
                    "mode": "portfolio",
                    "rank_by": "roc(40) - roc(5)",
                    "hold": 3,
                    "rebalance_every": 10,
                    "universe": {"symbols": symbols, "timeframe": "1D"},
                }
            }
        )
        members = [Membership(s, s, (Interval(_day(0), None),)) for s in SYMBOLS]
        # The table outlasts the last bar, exactly as Wikipedia does.
        members.append(
            Membership("ACQ", "ACQ", (Interval(_day(0), _day(stop + 8)),))
        )
        table = PointInTimeUniverse(
            name="probe",
            source="test",
            window=(_day(0), _day(N - 1)),
            members=tuple(members),
        )
        report = run_walk_forward(
            spec, data, train_bars=300, test_bars=100, config=CONFIG, membership=table
        )
        traded = [f for f in report.folds if f.test.num_trades > 0]
        assert traded, "folds after the acquisition must still run"


class TestAWindowExcludesRatherThanClamps:
    """`index_of_ts` returns the series end for a timestamp past the last bar,
    so both bounds of a late window collapsed onto it and a symbol delisted
    years earlier was sliced to its own final bars and handed to the fold. The
    engine builds its timeline from the union of event times, so one such name
    dragged the window back to its last trading day."""

    def test_a_series_that_ended_before_the_window_is_not_in_it(self) -> None:
        stop = 300
        data = {"OLD": _bars_ending_at("OLD", stop), **_data()}
        late_start = int(data["S00"].event_time[800])
        late_stop = int(data["S00"].event_time[900])
        sliced, _ = _window(data, late_start, late_stop, warmup=50)
        assert "OLD" not in sliced

    def test_the_names_that_remain_all_reach_into_the_window(self) -> None:
        data = {"OLD": _bars_ending_at("OLD", 300), **_data()}
        start = int(data["S00"].event_time[800])
        stop = int(data["S00"].event_time[900])
        sliced, _ = _window(data, start, stop, warmup=50)
        assert sliced
        for symbol, bars in sliced.items():
            assert int(bars.event_time[-1]) >= start, symbol


# --------------------------------------------------------------------------
# What "costs remove X% of the edge" is allowed to divide
# --------------------------------------------------------------------------


class TestTheCostGateComparesLikeWithLike:
    """`oos_sharpe / zero_cost_sharpe < 0.5` is a fatal gate.

    Its numerator is out-of-sample and pays costs. Its denominator used to be a
    frictionless run over the *whole* period, in sample -- so the ratio mixed
    friction with out-of-sample decay and charged the total to spreads. Now the
    denominator is the same test windows under the same selected parameters,
    with the costs set to zero, and nothing else differs.
    """

    @staticmethod
    def _report(config: BacktestConfig, zero: BacktestConfig | None):
        return run_walk_forward(
            _spec(), _data(), train_bars=300, test_bars=100,
            config=config, zero_cost_config=zero, membership=_membership(),
        )

    def test_it_is_absent_unless_asked_for(self) -> None:
        assert self._report(CONFIG, None).stitched_zero_cost is None

    def test_it_covers_the_same_windows_as_the_costed_curve(self) -> None:
        costly = BacktestConfig(
            initial_equity=1_000_000.0,
            allow_fractional_shares=True,
            costs=CostModel(spread_bps=40.0, slippage_bps=25.0, commission_per_share=0.02),
        )
        report = self._report(costly, ZERO_FREE)
        assert report.stitched is not None
        assert report.stitched_zero_cost is not None
        # Same folds, so the same number of days is being judged.
        assert report.stitched.num_trades == report.stitched_zero_cost.num_trades

    def test_removing_friction_cannot_lower_the_frictionless_curve(self) -> None:
        """The direction is the whole point: whatever the strategy is, paying
        forty basis points of spread cannot beat paying none of it."""
        costly = BacktestConfig(
            initial_equity=1_000_000.0,
            allow_fractional_shares=True,
            costs=CostModel(spread_bps=40.0, slippage_bps=25.0, commission_per_share=0.02),
        )
        report = self._report(costly, ZERO_FREE)
        assert report.stitched is not None and report.stitched_zero_cost is not None
        assert report.stitched_zero_cost.total_return > report.stitched.total_return

    def test_oos_sharpe_is_the_chained_curve_not_the_mean_of_the_folds(self) -> None:
        """A mean over folds discards what happens across a fold boundary, which
        is the reason the chained curve exists at all."""
        report = self._report(CONFIG, None)
        assert report.stitched is not None
        assert report.oos_sharpe == report.stitched.sharpe
        assert report.oos_sharpe != report.mean_fold_sharpe


# --------------------------------------------------------------------------
# Leave-one-out across processes
# --------------------------------------------------------------------------


WIDE = [f"W{i:02d}" for i in range(MIN_PARALLEL_DELETIONS + 8)]


def _wide_data() -> dict[str, Bars]:
    """A universe big enough that the parallel path is actually taken."""
    rng = np.random.default_rng(23)
    t = np.arange(T0, T0 + 400 * 86_400, 86_400, dtype=np.int64)
    out: dict[str, Bars] = {}
    for i, sym in enumerate(WIDE):
        steps = rng.normal(0.0004, 0.013, 400) + np.sin(
            np.arange(400) * 2 * np.pi / 70.0 + i
        ) * 0.0016
        close = 100.0 * np.exp(np.cumsum(steps))
        out[sym] = Bars(
            symbol=sym,
            timeframe="1D",
            event_time=t,
            open=close * 0.999,
            high=close * 1.011,
            low=close * 0.989,
            close=close,
            volume=np.full(400, 2e6),
        )
    return out


def _wide_spec():
    return spec_from_dict(
        {
            "strategy": {
                "name": "wide_probe",
                "mode": "portfolio",
                "rank_by": "roc(40) - roc(5)",
                "hold": 5,
                "rebalance_every": 10,
                "universe": {"symbols": WIDE, "timeframe": "1D"},
            }
        }
    )


class TestLeaveOneOutInParallel:
    """Forty deletions is forty independent pure functions of the same inputs.

    Spreading them over cores is only legitimate if the report cannot tell. The
    answers are keyed by the name removed and read back in sorted order, so
    completion order has nowhere to leak in -- but that is an argument, and
    these are the check. The universe is deliberately over
    ``MIN_PARALLEL_DELETIONS``, or the comparison would be serial against
    serial and would prove nothing.
    """

    @staticmethod
    def _report(workers: int) -> AssetReport:
        return _leave_one_out(
            _wide_spec(),
            _wide_data(),
            config=CONFIG,
            min_trades=1,
            workers=workers,
        )

    def test_the_universe_is_large_enough_to_take_the_parallel_path(self) -> None:
        assert len(WIDE) >= MIN_PARALLEL_DELETIONS

    def test_the_score_does_not_depend_on_the_number_of_workers(self) -> None:
        assert self._report(1).score == self._report(4).score

    def test_every_per_symbol_result_is_identical(self) -> None:
        serial, parallel = self._report(1), self._report(4)
        assert sorted(serial.per_symbol) == sorted(parallel.per_symbol)
        for symbol, metrics in serial.per_symbol.items():
            assert parallel.per_symbol[symbol] == metrics, symbol

    def test_a_small_universe_stays_on_the_serial_path(self) -> None:
        """Handing workers a copy of the bars to save ten short backtests is a
        loss. The threshold is a performance choice, so it must not become a
        correctness one: the answer is the same on either side of it."""
        assert len(SYMBOLS) < MIN_PARALLEL_DELETIONS
        small = _leave_one_out(
            _spec(), _data(), config=CONFIG, min_trades=1,
            membership=_membership(), workers=8,
        )
        serial = _leave_one_out(
            _spec(), _data(), config=CONFIG, min_trades=1,
            membership=_membership(), workers=1,
        )
        assert small.score == serial.score


class TestLeaveOneOutIsSealNeutral:
    """Why running deletions in other processes does not open a hole.

    The seal is a per-process singleton. A worker gets its own, so any taint it
    incurred would die with it and never reach the parent's ledger or hash
    chain -- silent contamination, which is the one thing the seal exists to
    prevent.

    It is safe because a deletion materialises nothing: it re-runs on bars the
    parent already loaded and already observed. This pins that, so that a change
    which starts slicing bars in here fails loudly instead of quietly voiding
    the guarantee. The workers check the same property from their side and the
    parent refuses a tainted result.
    """

    @staticmethod
    def _state() -> tuple[int | None, int, str, bool]:
        seal = current_seal()
        return (seal.max_event_time, len(seal.loads), seal.digest, seal.tainted)

    def test_a_deletion_pass_leaves_the_seal_untouched(self) -> None:
        # Build the bars first. Constructing a series is exactly what the seal
        # is supposed to notice, so materialising the fixture inside the
        # measurement would be measuring the fixture.
        spec, data, table = _spec(), _data(), _membership()
        before = self._state()
        _leave_one_out(
            spec, data, config=CONFIG, min_trades=1, membership=table, workers=1
        )
        assert self._state() == before

    def test_a_worker_reports_the_taint_it_would_have_hidden(self) -> None:
        """The parent cannot see into a worker's seal, so the worker returns the
        flag alongside its answer and `_parallel_deletions` refuses the whole
        result if any came back tainted.

        Asserted as "the deletion did not change it" rather than "it is False":
        the seal is a process-global singleton, so whether it is already
        tainted depends on what else has run in this session -- which is the
        very property that makes a per-process seal worth checking.
        """
        spec, data, table = _spec(), _data(), _membership()
        expected = current_seal().tainted
        _worker_init(spec, data, CONFIG, table)
        dropped, metrics, tainted = _worker_deletion(sorted(data)[0])
        assert dropped == sorted(data)[0]
        assert metrics is not None
        assert tainted is expected
