"""The option pipeline — specs/10-options-research.md D7, D8, D8a.

``evaluate_option_candidate`` is the option side of ``evaluate_candidate``, and
the things worth pinning are the ones where it differs rather than the ones it
shares. The shared parts (metrics, residual alpha, the overfitting signals, the
``Evaluation`` type) are already pinned by the equity suites and by
``test_evaluator_options.py``.

What is asserted here:

* a rule that opens nothing is rejected with the **census**, not with "no
  trades" -- the difference between "the rule never fired" and "the account
  could not afford it" is the difference between a bad hypothesis and a badly
  sized run (D8a),
* the multiplicity denominator handed to the overfitting detector is the option
  search's own count and never the combined one (D8),
* ``asset_robustness`` is replaced rather than zero-filled, by the mean of
  leave-one-year-out and DTE-bucket agreement,
* and the cost schedule travels with the verdict, because cost retention is a
  fatal gate and an option schedule ($/contract/leg) and an equity one ($/share)
  are not the same units.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pytest

from aqr.option_pipeline import evaluate_option_candidate, option_code_hash
from aqr.options.chain import ChainIndex
from aqr.options.costs import ALPACA_OPTIONS, IBKR_OPTIONS
from aqr.options.engine import OptionBacktestConfig
from aqr.options.run import OptionMarket
from aqr.options.spec import Cadence, OptionSizing, OptionSpec, StructureSpec
from aqr.registry.db import EQUITY, OPTION, ExperimentRecord, Registry
from tests.test_option_engine import chain_row, credit_spread_spec, make_underlying, trading_days


def _market(first: date = date(2019, 2, 4), last: date = date(2024, 6, 28)) -> OptionMarket:
    days = trading_days(1500, start=first - timedelta(days=30))
    sessions = [d for i, d in enumerate(days) if i >= 10 and i % 10 == 0 and d <= last]
    rows: list[dict[str, str]] = []
    for session in sessions:
        expiry = session + timedelta(days=28)
        while expiry.weekday() >= 5:
            expiry += timedelta(days=1)
        rows += [
            chain_row(session, expiry, 390.0, "put", bid=1.50, ask=1.60, delta=-0.16),
            chain_row(session, expiry, 380.0, "put", bid=0.45, ask=0.50, delta=-0.09),
            chain_row(session, expiry, 370.0, "put", bid=0.20, ask=0.25, delta=-0.04),
        ]
    return OptionMarket(underlying=make_underlying({}, days), chain=ChainIndex.from_rows(rows))


@pytest.fixture(scope="module")
def market() -> OptionMarket:
    return _market()


@pytest.fixture
def registry(tmp_path: Path) -> Any:
    with Registry(tmp_path / "research.sqlite") as reg:
        yield reg


def _spec(**overrides: Any) -> OptionSpec:
    fields: dict[str, Any] = {
        "name": "pipeline_probe",
        "hypothesis": "Index puts carry a variance risk premium.",
        "structure": StructureSpec(type="put_credit_spread", width_delta=0.06),
        "cadence": Cadence(min_sessions_between_entries=2),
        "sizing": OptionSizing(risk_per_trade=0.02, max_concurrent=3),
    }
    fields.update(overrides)
    return credit_spread_spec(**fields)


def _config(**overrides: Any) -> OptionBacktestConfig:
    fields: dict[str, Any] = {"initial_equity": 1_000_000.0}
    fields.update(overrides)
    return OptionBacktestConfig(**fields)


# --------------------------------------------------------------------------- #


def test_a_rule_that_opens_nothing_is_rejected_with_the_census(
    market: OptionMarket,
) -> None:
    """D8a: "no trades" is not a finding. Six mutually exclusive reasons are, and
    only one of them is about the rule."""
    outcome = evaluate_option_candidate(
        _spec(entry="close > 100000"), market, config=_config(), run_robustness=False
    )
    assert outcome.verdict == "REJECT"
    assert "opened no positions" in (outcome.rejected_early or "")
    assert "affordability" in (outcome.rejected_early or "")
    assert outcome.result_has_trades is False


def test_an_unaffordable_rule_is_distinguishable_from_an_unwanted_one(
    market: OptionMarket,
) -> None:
    """The same rule, the same window, a hundredth of the account. Every skip is
    an affordability skip, and the census says so rather than reporting the same
    "no trades" a never-firing condition would."""
    outcome = evaluate_option_candidate(
        _spec(), market, config=_config(initial_equity=100.0), run_robustness=False
    )
    assert outcome.result is not None
    census = outcome.result.skip_census
    assert census.affordability > 0
    assert census.affordability > census.no_leg_or_wing


def test_the_overfitting_denominator_is_the_option_searchs_own(
    registry: Registry, market: OptionMarket
) -> None:
    """D8: 414 equity hypotheses in the same ledger must not deflate an option
    result. The searches explored different spaces against different sample
    sizes, and one denominator across both makes both bars wrong."""
    for i in range(40):
        registry.record_experiment(
            ExperimentRecord(
                fingerprint=f"equity{i:016d}",
                strategy_name=f"equity_{i}",
                symbols=("SPY",),
                timeframe="1D",
                data_start="",
                data_end="",
                dataset_version="synthetic-v1",
                family=EQUITY,
                verdict="REJECT",
            )
        )

    outcome = evaluate_option_candidate(
        _spec(), market, config=_config(), registry=registry, run_robustness=False
    )
    assert outcome.overfitting is not None
    search_cost = next(s for s in outcome.overfitting.signals if s.name == "search_cost")
    # One option hypothesis, not forty-one.
    assert "1 backtest" in search_cost.reason


def test_year_and_dte_robustness_replaces_asset_robustness(
    market: OptionMarket,
) -> None:
    """``asset_robustness`` is undefined on one underlying and must not be
    faked. The replacement is the mean of the two D8 names, and neither one
    alone: a rule that survives every year but only at 28 DTE found a DTE
    target, not a premium."""
    outcome = evaluate_option_candidate(_spec(), market, config=_config())
    assert outcome.years is not None
    assert outcome.dte_buckets is not None
    expected = (outcome.years.score + outcome.dte_buckets.score) / 2
    assert outcome.year_dte_robustness == pytest.approx(expected)
    assert outcome.evaluation is not None
    assert outcome.evaluation.components["year_dte_robustness"] == pytest.approx(
        100.0 * min(max(expected, 0.0), 1.0)
    )
    assert "asset_robustness" not in outcome.evaluation.components


def test_an_unmeasured_run_reports_the_midpoint_rather_than_a_zero(
    market: OptionMarket,
) -> None:
    """With robustness skipped there is no measurement, and 0.0 would read as
    "measured, and it failed"."""
    outcome = evaluate_option_candidate(
        _spec(), market, config=_config(), run_robustness=False
    )
    assert outcome.years is None
    assert outcome.year_dte_robustness == 0.5


def test_the_cost_schedule_travels_with_the_verdict(
    registry: Registry, market: OptionMarket
) -> None:
    """Cost retention is a fatal gate, so the schedule is part of the verdict
    rather than context for it -- and an option schedule and an equity one are
    not the same units, so a reader comparing two verdicts has to see which was
    charged."""
    evaluate_option_candidate(
        _spec(),
        market,
        config=_config(costs=ALPACA_OPTIONS),
        registry=registry,
        run_robustness=False,
    )
    (row,) = registry.experiments(5, family=OPTION)
    assert ALPACA_OPTIONS.name in (row["costs"] or "")
    assert IBKR_OPTIONS.name not in (row["costs"] or "")


def test_the_experiment_records_the_independent_cycle_count(
    registry: Registry, market: OptionMarket
) -> None:
    """It is what the evaluator gates on, so a log that showed only the trade
    count would let a reader believe a sample that does not exist."""
    evaluate_option_candidate(
        _spec(), market, config=_config(), registry=registry, run_robustness=False
    )
    (row,) = registry.memory(5, family=OPTION)
    assert row["oos_cycles"] is not None
    assert row["oos_trades"] is None or row["oos_cycles"] <= row["oos_trades"]


def test_the_code_hash_covers_the_option_engine_not_the_equity_one() -> None:
    """Two option experiments with identical metrics and different engine code
    are not comparable, and the modules that decide an option result are not the
    ones ``pipeline.code_hash`` reads."""
    from aqr.pipeline import code_hash

    assert option_code_hash() != code_hash()
    assert option_code_hash() == option_code_hash()


def test_the_same_spec_on_the_same_market_gives_the_same_verdict(
    market: OptionMarket,
) -> None:
    """D5: two runs of the same spec on the same cache produce the same trades,
    byte for byte, or the engine is broken -- and therefore the same score."""
    left = evaluate_option_candidate(_spec(), market, config=_config(), run_robustness=False)
    right = evaluate_option_candidate(_spec(), market, config=_config(), run_robustness=False)
    assert left.score == right.score
    assert left.verdict == right.verdict
    assert left.as_dict()["walk_forward"] == right.as_dict()["walk_forward"]
