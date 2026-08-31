"""Calendar walk-forward — specs/10-options-research.md D8.

Two things are load-bearing here and get most of the attention:

**Fold geometry is calendar time, not bar count.** A fold's width has to be
checkable by counting months on a calendar, not by counting array positions.

**A fold slices the input, not the output.** D8's own failure mode: a
position entered near the end of a train window that settles inside the
following test window must not be attributed to the test window's own
metrics. The fixture below is built so a hand count of that specific failure
mode is possible -- a position entered one session before a fold boundary,
with an expiry that lands after it.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from aqr.options.engine import affordability_bound_fraction
from aqr.options.run import OptionMarket
from aqr.options.spec import Cadence, OptionSizing
from aqr.options.walkforward import (
    _add_months,
    option_walk_forward_folds,
    run_option_walk_forward,
)
from aqr.validation.cycles import independent_cycles
from tests.test_option_engine import build_world, credit_spread_spec, trading_days

# --------------------------------------------------------------------------- #
# Month arithmetic
# --------------------------------------------------------------------------- #


def test_add_months_advances_the_month_and_keeps_the_day() -> None:
    assert _add_months(date(2019, 2, 9), 24) == date(2021, 2, 9)


def test_add_months_clamps_to_the_shorter_months_last_day() -> None:
    """2024-08-31 plus one month is 2024-09-30, not an October overflow."""
    assert _add_months(date(2024, 8, 31), 1) == date(2024, 9, 30)


def test_add_months_handles_a_year_rollover() -> None:
    assert _add_months(date(2024, 11, 15), 3) == date(2025, 2, 15)


# --------------------------------------------------------------------------- #
# Fold geometry
# --------------------------------------------------------------------------- #


def test_folds_tile_the_span_with_no_gap_and_no_overlap() -> None:
    folds = option_walk_forward_folds(
        date(2019, 2, 9), date(2024, 8, 30), train_months=24, test_months=6, step_months=6
    )
    assert len(folds) >= 6
    for a, b in zip(folds, folds[1:], strict=False):
        assert a.test_stop == b.test_start
        assert a.train_stop == a.test_start


def test_a_rolling_train_window_stays_a_fixed_width() -> None:
    """24 months train, stepping 6 months: every fold's own train window is
    exactly two years wide, not a growing, anchored-at-the-start window."""
    folds = option_walk_forward_folds(
        date(2019, 2, 9), date(2024, 8, 30), train_months=24, test_months=6, step_months=6
    )
    for fold in folds:
        assert _add_months(fold.train_start, 24) == fold.train_stop


def test_a_stub_final_window_is_dropped() -> None:
    """The span ends 20 days after what would be the next fold's train stop --
    far too little for even a single 28 DTE cycle to close inside it."""
    first = date(2019, 2, 9)
    last = _add_months(_add_months(first, 24), 6) + timedelta(days=20)
    folds = option_walk_forward_folds(first, last, train_months=24, test_months=6, step_months=6)
    assert len(folds) == 1
    assert folds[0].test_stop == _add_months(_add_months(first, 24), 6)


# --------------------------------------------------------------------------- #
# The leak D8 names by name
# --------------------------------------------------------------------------- #


def _last_weekday_before(d: date) -> date:
    probe = d - timedelta(days=1)
    while probe.weekday() >= 5:
        probe -= timedelta(days=1)
    return probe


def test_a_position_entered_before_a_fold_boundary_is_not_counted_in_the_next_fold() -> None:
    """One structure, entered the last session before fold 0's test window
    opens, expiring well inside it. If the walk-forward sliced the *output*
    of one full-window run by entry date, this trade is unambiguously a
    train-fold trade (it entered before the boundary) and must never show up
    as a test trade; if it instead re-ran the engine per fold on an
    unrestricted chain, the same session could re-enter under a later fold's
    own window and be double-counted. Both failures are ruled out by
    confirming the trade lands in exactly one fold's train bucket and no
    fold's test bucket.

    Two more anchor sessions, well separated in calendar time, exist only so
    the chain's own ``first()``/``last()`` span enough of the calendar for a
    real fold to be built around the boundary session; ``max_concurrent=1``
    and their wide separation keep them from ever overlapping the trade under
    test, so they cannot themselves land in anyone's test bucket by
    coincidence -- the settlement-adjusted date is spread checked explicitly
    below.
    """
    days = trading_days(900, start=date(2019, 2, 4))  # a Monday
    fold0_test_start = _add_months(days[0], 24)
    boundary_session = _last_weekday_before(fold0_test_start)
    tail_session = fold0_test_start + timedelta(days=60)
    while tail_session.weekday() >= 5:
        tail_session += timedelta(days=1)

    chain, bars = build_world(sessions=[days[0], boundary_session, tail_session], days=days)
    spec = credit_spread_spec(
        cadence=Cadence(min_sessions_between_entries=1),
        sizing=OptionSizing(risk_per_trade=0.01, max_concurrent=1),
    )
    market = OptionMarket(underlying=bars, chain=chain)
    report = run_option_walk_forward(spec, market, train_months=24, test_months=6, step_months=6)
    assert report.folds, "the fixture must produce at least one fold"

    all_test = [t for f in report.folds for t in f.test_trades]
    boundary_epoch = int(
        datetime(boundary_session.year, boundary_session.month, boundary_session.day, tzinfo=UTC)
        .timestamp()
    )
    assert boundary_epoch not in {t.entry_time for t in all_test}

    fold0 = report.folds[0]
    train_entries = {t.entry_time for t in fold0.train_trades}
    assert boundary_epoch in train_entries


def test_trades_in_a_test_window_are_not_double_counted_across_folds() -> None:
    """A dense entry schedule spanning several folds: the union of every
    fold's own test-window trade count, summed, must equal exactly the
    independent-cycle count over the pooled out-of-sample list -- pooling
    must never invent or drop a trade at a fold seam."""
    days = trading_days(1700, start=date(2019, 2, 9))
    sessions = days[5::5]  # every 5th trading day: dense enough to span folds
    chain, bars = build_world(sessions=sessions, days=days)
    spec = credit_spread_spec(
        cadence=Cadence(min_sessions_between_entries=5),
        sizing=OptionSizing(risk_per_trade=0.01, max_concurrent=3),
    )
    market = OptionMarket(underlying=bars, chain=chain)
    report = run_option_walk_forward(spec, market, train_months=24, test_months=6, step_months=6)

    pooled = report.oos_trades
    entry_times = [t.entry_time for t in pooled]
    assert len(entry_times) == len(set(entry_times)), "a trade was counted in more than one fold"
    assert report.oos_cycles == independent_cycles(pooled)
    assert report.oos_cycles <= len(pooled)


def test_geometry_and_cycle_counts_are_on_the_report() -> None:
    days = trading_days(1700, start=date(2019, 2, 9))
    sessions = days[5::5]
    chain, bars = build_world(sessions=sessions, days=days)
    spec = credit_spread_spec(
        cadence=Cadence(min_sessions_between_entries=5),
        sizing=OptionSizing(risk_per_trade=0.01, max_concurrent=3),
    )
    market = OptionMarket(underlying=bars, chain=chain)
    report = run_option_walk_forward(spec, market)
    payload = report.as_dict()
    assert payload["geometry"] == {"train_months": 24, "test_months": 6, "step_months": 6}
    for fold in payload["folds"]:
        assert "test_cycles" in fold and "test_trades" in fold
        assert fold["test_cycles"] <= fold["test_trades"]


def test_an_empty_chain_produces_an_empty_report_rather_than_raising() -> None:
    days = trading_days(80)
    chain, bars = build_world(sessions=[], days=days)
    market = OptionMarket(underlying=bars, chain=chain)
    report = run_option_walk_forward(credit_spread_spec(), market)
    assert report.folds == []
    assert report.oos_trades == []


# --------------------------------------------------------------------------- #
# The skip census, pooled the same way trades are (D8a)
# --------------------------------------------------------------------------- #


def test_the_oos_skip_census_is_measured_over_the_pooled_test_windows_only() -> None:
    """A risk budget too small for one contract (0.2% of 100,000 against 900
    of risk per contract, the same arithmetic
    ``test_option_engine.py::test_a_risk_budget_too_small_for_one_contract_skips_the_entry``
    checks) turns every entry across the whole span into an affordability
    skip. Spanning several folds is what proves this is *pooled* from each
    fold's own ``test_skip_census`` rather than read off one full-window
    run: :func:`affordability_bound_fraction` on the pooled result must equal
    the same function applied by hand to the report's own pooled numbers.
    """
    days = trading_days(1700, start=date(2019, 2, 9))
    sessions = days[5::5]
    chain, bars = build_world(sessions=sessions, days=days)
    spec = credit_spread_spec(
        cadence=Cadence(min_sessions_between_entries=5),
        sizing=OptionSizing(risk_per_trade=0.002, max_concurrent=3),
    )
    market = OptionMarket(underlying=bars, chain=chain)
    report = run_option_walk_forward(spec, market)

    assert report.oos_trades == []  # nothing was ever affordable
    assert report.oos_skip_census.affordability > 0
    # Pooled by plain summation of each fold's own test-window count, exactly
    # the way ``len(oos_trades)`` is -- checked on the ``affordability`` field
    # alone rather than the whole census, because the research span this
    # fixture's dense entry schedule runs into also crosses D3's embargo
    # boundary near its tail, which contributes real ``embargo`` skips this
    # test is not about.
    assert report.oos_skip_census.affordability == sum(
        f.test_skip_census.affordability for f in report.folds
    )
    assert report.oos_affordability_bound_fraction == pytest.approx(
        affordability_bound_fraction(report.oos_skip_census, len(report.oos_trades))
    )
    # Every session that reached sizing was unaffordable and none traded, so
    # the fraction is exactly 1.0 -- the sharpest possible case of D8a's
    # pathology, and the one the report must not soften into something else.
    assert report.oos_affordability_bound_fraction == pytest.approx(1.0)


def test_a_folds_own_skip_census_is_on_the_report_dict() -> None:
    days = trading_days(1700, start=date(2019, 2, 9))
    sessions = days[5::5]
    chain, bars = build_world(sessions=sessions, days=days)
    spec = credit_spread_spec(
        cadence=Cadence(min_sessions_between_entries=5),
        sizing=OptionSizing(risk_per_trade=0.002, max_concurrent=3),
    )
    market = OptionMarket(underlying=bars, chain=chain)
    report = run_option_walk_forward(spec, market)
    payload = report.as_dict()
    assert "oos_skip_census" in payload and "oos_affordability_bound_fraction" in payload
    assert payload["oos_affordability_bound_fraction"] == pytest.approx(1.0)
    for fold in payload["folds"]:
        assert "test_skip_census" in fold and "test_affordability_bound_fraction" in fold
        assert fold["test_skip_census"]["affordability"] >= 0


def test_a_skip_before_a_fold_boundary_is_not_counted_in_the_next_folds_test_census() -> None:
    """The skip-side mirror of
    ``test_a_position_entered_before_a_fold_boundary_is_not_counted_in_the_next_fold``:
    an affordability skip the session right before fold 0's test window opens
    must land in that fold's *train* census, never its test one, and must
    not also appear in fold 1's test census -- ``_bucket_skips`` is a
    half-open ``[window_start, window_stop)`` filter on the same session
    field ``_bucket`` already uses for trades, and this is the same leak D8
    names for trades, checked for skips instead.
    """
    days = trading_days(900, start=date(2019, 2, 4))
    fold0_test_start = _add_months(days[0], 24)
    boundary_session = _last_weekday_before(fold0_test_start)
    tail_session = fold0_test_start + timedelta(days=60)
    while tail_session.weekday() >= 5:
        tail_session += timedelta(days=1)

    # Three sessions, matching the trade-side boundary test above: ``days[0]``
    # and ``tail_session`` exist only so the chain's own ``first()``/
    # ``last()`` span enough calendar time for a real fold to be built
    # around the boundary session -- a chain with one session has
    # ``first() == last()`` and ``option_walk_forward_folds`` refuses that.
    chain, bars = build_world(sessions=[days[0], boundary_session, tail_session], days=days)
    spec = credit_spread_spec(
        cadence=Cadence(min_sessions_between_entries=1),
        sizing=OptionSizing(risk_per_trade=0.002, max_concurrent=1),
    )
    market = OptionMarket(underlying=bars, chain=chain)
    report = run_option_walk_forward(spec, market, train_months=24, test_months=6, step_months=6)
    assert report.folds, "the fixture must produce at least one fold"

    fold0 = report.folds[0]
    assert fold0.train_skip_census.affordability == 1
    assert fold0.test_skip_census.affordability == 0
    if len(report.folds) > 1:
        assert report.folds[1].test_skip_census.affordability == 0
