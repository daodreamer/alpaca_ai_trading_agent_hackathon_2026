"""D8's asset-robustness replacements — specs/10-options-research.md D8.

``asset_robustness`` (leave-one-asset-out) has nothing to leave out on a
single-underlying cache. These tests check the three things D8 substitutes:
leave-one-year-out, DTE-bucket agreement, and options-shaped parameter
perturbation -- plus the thin regime-robustness bucketing this module reuses
``validation.robustness.score_regimes`` for.

Most of the arithmetic these functions lean on (selection, settlement, cycle
counting, ``ParameterReport``/``RegimeReport`` scoring) is already pinned by
other suites; what is under test here is the *wiring* -- that a year is
actually excluded rather than merely relabelled, that a DTE bucket actually
resolves at the requested target, and that a perturbation actually touches
the knob it claims to.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from aqr.options.chain import ChainIndex
from aqr.options.robustness import (
    dte_bucket_agreement,
    leave_one_year_out,
    option_parameter_stability,
    option_regime_robustness,
)
from aqr.options.run import OptionMarket, run_option_strategy
from aqr.options.spec import Cadence, OptionSizing
from tests.test_option_engine import chain_row, credit_spread_spec, make_underlying, trading_days

# --------------------------------------------------------------------------- #
# Leave-one-year-out
# --------------------------------------------------------------------------- #


def _multi_year_world(years: tuple[int, ...] = (2019, 2020, 2021)) -> tuple[ChainIndex, object]:
    """A session in January of each requested year, all pricing identically
    (build_world's own fixture), so a deletion changes only *which* years are
    represented, never the shape of an individual trade."""
    days = trading_days(1500, start=date(years[0], 1, 7))
    rows = []
    sessions = []
    for year in years:
        # index >= 6 so a decision bar always exists (the engine correctly
        # refuses an entry whose reference bar is the series' first).
        session = next(d for i, d in enumerate(days) if d.year == year and d.month == 1 and i >= 6)
        sessions.append(session)
        expiry = session + timedelta(days=28)
        while expiry.weekday() >= 5:
            expiry += timedelta(days=1)
        rows += [
            chain_row(session, expiry, 390.0, "put", bid=1.50, ask=1.60, delta=-0.16),
            chain_row(session, expiry, 380.0, "put", bid=0.45, ask=0.50, delta=-0.09),
        ]
    return ChainIndex.from_rows(rows), make_underlying({}, days)


def test_leave_one_year_out_removes_exactly_that_years_entries() -> None:
    chain, bars = _multi_year_world((2019, 2020, 2021))
    market = OptionMarket(underlying=bars, chain=chain)
    spec = credit_spread_spec(
        cadence=Cadence(min_sessions_between_entries=1),
        sizing=OptionSizing(risk_per_trade=0.01, max_concurrent=3),
    )
    baseline = run_option_strategy(spec, market)
    assert {t.entry_session.year for t in baseline.option_trades} == {2019, 2020, 2021}

    report = leave_one_year_out(spec, market, years=(2019, 2020, 2021), min_cycles=1)
    assert set(report.per_year) == {2019, 2020, 2021}
    for dropped_year, metrics in report.per_year.items():
        # A deletion run's own trades must never include the dropped year.
        deletion_chain = market.chain.exclude_dates(
            date(dropped_year, 1, 1), date(dropped_year + 1, 1, 1)
        )
        deletion_result = run_option_strategy(spec, market.with_chain(deletion_chain))
        assert dropped_year not in {t.entry_session.year for t in deletion_result.option_trades}
        assert metrics.num_trades == len(deletion_result.option_trades)


def test_leave_one_year_out_skips_a_year_absent_from_the_chain() -> None:
    chain, bars = _multi_year_world((2020,))
    market = OptionMarket(underlying=bars, chain=chain)
    spec = credit_spread_spec(cadence=Cadence(min_sessions_between_entries=1))
    report = leave_one_year_out(spec, market, years=(2019, 2020), min_cycles=1)
    assert 2019 not in report.per_year, "2019 has no sessions in this chain at all"
    assert 2020 in report.per_year


def test_leave_one_year_out_scores_only_years_with_enough_cycles() -> None:
    chain, bars = _multi_year_world((2019, 2020, 2021))
    market = OptionMarket(underlying=bars, chain=chain)
    spec = credit_spread_spec(cadence=Cadence(min_sessions_between_entries=1))
    report = leave_one_year_out(spec, market, years=(2019, 2020, 2021), min_cycles=1_000_000)
    assert report.score == 0.0  # nothing clears an impossible cycle bar


# --------------------------------------------------------------------------- #
# DTE-bucket agreement
# --------------------------------------------------------------------------- #


def _three_bucket_world(sessions: list[date], days: list[date]) -> ChainIndex:
    """Every session offers a 16-delta put at all three of D0's DTE targets,
    so a spec variant asking for 14, 28 or 49 has something to resolve."""
    rows = []
    for session in sessions:
        for dte in (14, 28, 49):
            expiry = session + timedelta(days=dte)
            while expiry.weekday() >= 5:
                expiry += timedelta(days=1)
            rows += [
                chain_row(session, expiry, 390.0, "put", bid=1.50, ask=1.60, delta=-0.16),
                chain_row(session, expiry, 380.0, "put", bid=0.45, ask=0.50, delta=-0.09),
            ]
    return ChainIndex.from_rows(rows)


def test_dte_bucket_agreement_trades_at_each_requested_target() -> None:
    days = trading_days(300)
    sessions = days[5:60:5]
    chain = _three_bucket_world(sessions, days)
    market = OptionMarket(underlying=make_underlying({}, days), chain=chain)
    spec = credit_spread_spec(
        cadence=Cadence(min_sessions_between_entries=1),
        sizing=OptionSizing(risk_per_trade=0.01, max_concurrent=5),
    )
    report = dte_bucket_agreement(spec, market, targets=(14, 28, 49), min_cycles=1)
    assert set(report.per_bucket) == {14, 28, 49}
    for target, metrics in report.per_bucket.items():
        assert metrics.num_trades > 0, f"the {target} DTE bucket traded nothing"


def test_dte_bucket_agreement_leaves_the_original_spec_untouched() -> None:
    days = trading_days(200)
    sessions = days[5:20:5]
    chain = _three_bucket_world(sessions, days)
    market = OptionMarket(underlying=make_underlying({}, days), chain=chain)
    spec = credit_spread_spec(cadence=Cadence(min_sessions_between_entries=1))
    dte_bucket_agreement(spec, market, targets=(14, 28, 49), min_cycles=1)
    assert spec.structure.dte.target == 28  # the fixture default, unmutated


# --------------------------------------------------------------------------- #
# Parameter perturbation
# --------------------------------------------------------------------------- #


def test_option_parameter_stability_perturbs_the_dte_delta_width_and_cadence() -> None:
    days = trading_days(300)
    sessions = days[5:60:5]
    chain = _three_bucket_world(sessions, days)
    market = OptionMarket(underlying=make_underlying({}, days), chain=chain)
    spec = credit_spread_spec(
        cadence=Cadence(min_sessions_between_entries=5),
        sizing=OptionSizing(risk_per_trade=0.01, max_concurrent=5),
    )
    report = option_parameter_stability(spec, market)
    assert "structure.anchor.delta" in report.per_slot
    assert "structure.dte.target" in report.per_slot
    assert "cadence.min_sessions_between_entries" in report.per_slot
    assert "structure.width_points" in report.per_slot  # credit_spread_spec uses width_points
    assert 0.0 <= report.score <= 1.0


def test_option_parameter_stability_baseline_matches_a_direct_backtest() -> None:
    from aqr.backtest.metrics import compute_metrics

    days = trading_days(200)
    sessions = days[5:20:5]
    chain = _three_bucket_world(sessions, days)
    market = OptionMarket(underlying=make_underlying({}, days), chain=chain)
    spec = credit_spread_spec(cadence=Cadence(min_sessions_between_entries=1))
    expected = compute_metrics(run_option_strategy(spec, market)).sharpe
    report = option_parameter_stability(spec, market)
    assert report.baseline == pytest.approx(expected)


# --------------------------------------------------------------------------- #
# Regime robustness (bucketing reused from validation/robustness.py)
# --------------------------------------------------------------------------- #


def test_option_regime_robustness_is_empty_without_trades() -> None:
    days = trading_days(60)
    report = option_regime_robustness([], make_underlying({}, days))
    assert report.per_regime == {}
    assert report.score == 0.0


def test_option_regime_robustness_buckets_every_trade_it_can_label() -> None:
    days = trading_days(300)
    sessions = days[5:60:5]
    chain = _three_bucket_world(sessions, days)
    market = OptionMarket(underlying=make_underlying({}, days), chain=chain)
    spec = credit_spread_spec(
        cadence=Cadence(min_sessions_between_entries=1),
        sizing=OptionSizing(risk_per_trade=0.01, max_concurrent=5),
    )
    result = run_option_strategy(spec, market)
    report = option_regime_robustness(result.trades, market.underlying, min_trades=1)
    bucketed = sum(m.num_trades for m in report.per_regime.values())
    assert bucketed <= len(result.trades)
    assert bucketed > 0
