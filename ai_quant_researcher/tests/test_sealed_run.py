"""The measurement the sealed run makes, separated from the process that runs it.

The sealed run is one backtest and one regression, and the only thing that makes
it different from any other backtest is *where it starts measuring*. That is the
part worth testing: a window that leaked one pre-embargo session would report
in-sample performance under a sealed banner, which is the single most misleading
number this project could produce.

What it cannot do is also asserted here. Two years is roughly 500 sessions, and
the standard error of an annualised Sharpe on 500 sessions is near 0.7. The
result carries that number so nobody has to remember it, and
``can_confirm`` is False by construction: this run can refute a candidate and
cannot confirm one.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from aqr.data.bars import Bars
from aqr.data.providers import SyntheticProvider
from aqr.dsl.loader import loads
from aqr.dsl.schema import StrategySpec
from aqr.validation.sealed import SealedMeasurement, measure_sealed_window

START = datetime(2015, 1, 1, tzinfo=UTC)
END = datetime(2024, 1, 1, tzinfo=UTC)
SYMBOLS = ("SPY", "QQQ", "IWM", "DIA", "EFA")

PORTFOLIO = """
strategy:
  name: sealed_probe
  hypothesis: Calm winners keep winning.
  mode: portfolio
  universe: {symbols: [SPY, QQQ, IWM, DIA, EFA], timeframe: 1D}
  rank_by: rs_rank(60) + rs_rank(20)
  hold: 2
  rebalance_every: 5
  exit:
    stop_loss: {type: atr, multiplier: 2.0, period: 14}
    take_profit: {type: risk_reward, ratio: 2.0}
    max_holding_bars: 20
  sizing: {risk_per_trade: 0.01, max_position_pct: 0.25}
"""


@pytest.fixture(scope="module")
def data() -> dict[str, Bars]:
    provider = SyntheticProvider()
    return {s: provider.load(s, START, END) for s in SYMBOLS}


@pytest.fixture(scope="module")
def spec() -> StrategySpec:
    return loads(PORTFOLIO)


def _measure(spec: StrategySpec, data: dict[str, Bars], since: datetime) -> SealedMeasurement:
    return measure_sealed_window(spec, data, since=since)


# --------------------------------------------------------------------------
# The window


def test_the_measurement_starts_where_it_was_told_to(
    spec: StrategySpec, data: dict[str, Bars]
) -> None:
    since = datetime(2022, 1, 1, tzinfo=UTC)
    measured = _measure(spec, data, since)
    assert measured.first_session >= since
    assert measured.observations > 0


def test_nothing_before_the_window_reaches_the_result(
    spec: StrategySpec, data: dict[str, Bars]
) -> None:
    """A single leaked pre-embargo session is in-sample performance reported
    under a sealed banner, so the boundary is asserted rather than assumed."""
    since = datetime(2022, 1, 1, tzinfo=UTC)
    measured = _measure(spec, data, since)
    expected = sum(
        1
        for stamp in data["SPY"].timestamps
        if stamp >= since
    )
    # One session carries no return, and the regression pairs returns.
    assert measured.observations <= expected


def test_a_later_window_is_a_strict_subset_of_an_earlier_one(
    spec: StrategySpec, data: dict[str, Bars]
) -> None:
    early = _measure(spec, data, datetime(2021, 1, 1, tzinfo=UTC))
    late = _measure(spec, data, datetime(2022, 1, 1, tzinfo=UTC))
    assert late.observations < early.observations
    assert late.first_session > early.first_session


def test_the_warmup_history_is_used_but_not_measured(
    spec: StrategySpec, data: dict[str, Bars]
) -> None:
    """``rs_rank(60)`` needs sixty sessions of history it must not be scored on.

    The backtest runs over everything it is given; only the measurement window
    is clipped. Truncating the *bars* instead would leave the first sixty
    sessions of the sealed window trading on an unwarmed signal, which is a
    different strategy than the one that was validated.
    """
    since = datetime(2022, 1, 1, tzinfo=UTC)
    measured = _measure(spec, data, since)
    assert measured.backtest_sessions > measured.observations


def test_a_window_with_too_little_data_refuses_rather_than_guessing(
    spec: StrategySpec, data: dict[str, Bars]
) -> None:
    """A beta estimated from five days is a number and not an estimate."""
    measured = _measure(spec, data, datetime(2023, 12, 1, tzinfo=UTC))
    assert measured.residual is None
    assert "observations" in (measured.note or "")


def test_a_window_after_the_last_bar_refuses(
    spec: StrategySpec, data: dict[str, Bars]
) -> None:
    measured = _measure(spec, data, datetime(2030, 1, 1, tzinfo=UTC))
    assert measured.residual is None
    assert measured.observations == 0


# --------------------------------------------------------------------------
# The regression


def test_the_residual_is_reported_against_the_same_sessions(
    spec: StrategySpec, data: dict[str, Bars]
) -> None:
    measured = _measure(spec, data, datetime(2020, 1, 1, tzinfo=UTC))
    assert measured.residual is not None
    assert measured.residual.observations == measured.observations
    assert np.isfinite(measured.residual.beta)


def test_the_benchmark_is_measured_over_the_window_too(
    spec: StrategySpec, data: dict[str, Bars]
) -> None:
    """Comparing a two-year strategy Sharpe to a nine-year benchmark Sharpe is
    a comparison of two different market regimes wearing one label."""
    early = _measure(spec, data, datetime(2018, 1, 1, tzinfo=UTC))
    late = _measure(spec, data, datetime(2022, 1, 1, tzinfo=UTC))
    assert early.benchmark_sharpe != late.benchmark_sharpe


def test_the_measurement_is_deterministic(
    spec: StrategySpec, data: dict[str, Bars]
) -> None:
    since = datetime(2021, 1, 1, tzinfo=UTC)
    a, b = _measure(spec, data, since), _measure(spec, data, since)
    assert a.as_dict() == b.as_dict()


# --------------------------------------------------------------------------
# What it cannot do


def test_the_result_states_its_own_standard_error(
    spec: StrategySpec, data: dict[str, Bars]
) -> None:
    """Two years is around 500 sessions, and the standard error of an annualised
    Sharpe on 500 sessions is near 0.7. Carried on the result so the report
    cannot quietly omit it."""
    measured = _measure(spec, data, datetime(2022, 1, 1, tzinfo=UTC))
    assert 0.4 < measured.sharpe_standard_error < 1.2


def test_the_result_never_claims_to_confirm(
    spec: StrategySpec, data: dict[str, Bars]
) -> None:
    for since in (datetime(2016, 1, 1, tzinfo=UTC), datetime(2022, 1, 1, tzinfo=UTC)):
        assert _measure(spec, data, since).can_confirm is False


def test_a_refuted_candidate_is_named_as_refuted(
    spec: StrategySpec, data: dict[str, Bars]
) -> None:
    """The one verdict the sealed run is entitled to reach.

    ``refuted`` is true when the residual alpha is negative and significant --
    the rule did not merely fail to repeat, it went the other way by more than
    the sample can explain.
    """
    measured = _measure(spec, data, datetime(2020, 1, 1, tzinfo=UTC))
    assert measured.residual is not None
    expected = measured.residual.alpha < 0 and measured.residual.t_alpha <= -2.0
    assert measured.refuted is expected


def test_the_summary_mentions_the_limitation(
    spec: StrategySpec, data: dict[str, Bars]
) -> None:
    text = _measure(spec, data, datetime(2022, 1, 1, tzinfo=UTC)).summary()
    assert "refute" in text.lower()
