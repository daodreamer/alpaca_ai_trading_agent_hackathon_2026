"""Cross-sectional strategies through the rest of the machinery.

The feature computation is tested next door. What is tested here is everything
that quietly assumed a rule could be evaluated one symbol at a time.

``asset_robustness`` is the sharp case. It asks "is this an edge, or a property
of one ticker?" and answers it by re-running the rule on each symbol alone. For
a peer-relative rule that question is unanswerable as posed: with one symbol
there is no cross-section, every feature is NaN, the rule fires on nothing, and
the symbol is skipped. Skip them all and the score is 0.0 -- so every
cross-sectional strategy would be silently docked a tenth of its total, not as a
placeholder but as an active penalty for using the feature.

The fix is to ask the question properly: hold the market definition fixed and
trade one name out of it.
"""

from __future__ import annotations

import numpy as np
import pytest

from aqr.backtest.engine import run_backtest
from aqr.data.bars import Bars
from aqr.dsl.loader import loads
from aqr.dsl.schema import StrategySpec
from aqr.dsl.validator import validate_against
from aqr.features.cross_section import CrossSection
from aqr.validation.robustness import asset_robustness

SPEC_YAML = """
strategy:
  name: cross_sectional_momentum_v1
  hypothesis: The strongest names in a broad advance keep going.
  universe:
    symbols: [SPY, QQQ, IWM]
    timeframe: 1D
  entry: rs_rank(60) > 0.6 and breadth(60) > 0.5
  exit:
    stop_loss: {type: atr, multiplier: 2.0, period: 14}
    take_profit: {type: risk_reward, ratio: 2.0}
    max_holding_bars: 20
  sizing:
    risk_per_trade: 0.005
    max_position_pct: 0.25
  max_positions: 3
"""


@pytest.fixture
def cross_spec() -> StrategySpec:
    return loads(SPEC_YAML)


class TestItRunsAtAll:
    def test_a_peer_relative_rule_backtests(
        self, cross_spec: StrategySpec, universe: dict[str, Bars]
    ) -> None:
        result = run_backtest(cross_spec, universe)
        assert result.trades, "a cross-sectional rule should be able to trade"

    def test_it_validates_when_given_the_universe(
        self, cross_spec: StrategySpec, universe: dict[str, Bars]
    ) -> None:
        report = validate_against(cross_spec, universe["SPY"], CrossSection(universe))
        assert report.ok, report.errors

    def test_validating_without_the_universe_is_an_error_not_a_silence(
        self, cross_spec: StrategySpec, universe: dict[str, Bars]
    ) -> None:
        report = validate_against(cross_spec, universe["SPY"])
        assert not report.ok
        assert any("cross-sectional" in e for e in report.errors)


class TestPeerSet:
    def test_the_peer_set_defaults_to_the_traded_universe(
        self, cross_spec: StrategySpec, universe: dict[str, Bars]
    ) -> None:
        """Reproducible from the spec alone: two runs of one strategy must rank
        against the same market, whatever else happens to be loaded."""
        extra = dict(universe)
        extra["EXTRA"] = universe["SPY"]  # loaded but not in the spec's universe
        assert _fingerprint(run_backtest(cross_spec, universe)) == _fingerprint(
            run_backtest(cross_spec, extra)
        )

    def test_an_explicit_peer_set_is_used_for_the_features(
        self, cross_spec: StrategySpec, universe: dict[str, Bars]
    ) -> None:
        one = cross_spec.with_params(
            universe=cross_spec.universe.__class__(symbols=("SPY",), timeframe="1D")
        )
        alone = run_backtest(one, {"SPY": universe["SPY"]})
        with_peers = run_backtest(one, {"SPY": universe["SPY"]}, peers=universe)

        assert not alone.trades, "one symbol is no cross-section"
        assert with_peers.trades, "the same symbol ranked against a real universe trades"

    def test_peers_may_include_symbols_that_are_never_traded(
        self, cross_spec: StrategySpec, universe: dict[str, Bars]
    ) -> None:
        one = cross_spec.with_params(
            universe=cross_spec.universe.__class__(symbols=("SPY",), timeframe="1D")
        )
        result = run_backtest(one, {"SPY": universe["SPY"]}, peers=universe)
        assert set(result.symbols) == {"SPY"}


class TestAssetRobustness:
    def test_a_cross_sectional_strategy_is_not_silently_scored_zero(
        self, cross_spec: StrategySpec, universe: dict[str, Bars]
    ) -> None:
        report = asset_robustness(cross_spec, universe)
        assert report.per_symbol, "every symbol was skipped -- the score would be a fiction"

    def test_each_symbol_is_judged_against_the_whole_universe(
        self, cross_spec: StrategySpec, universe: dict[str, Bars]
    ) -> None:
        report = asset_robustness(cross_spec, universe)
        for symbol in universe:
            assert symbol in report.per_symbol
            assert report.per_symbol[symbol].num_trades > 0

    def test_a_single_symbol_strategy_is_unaffected(
        self, spec: StrategySpec, universe: dict[str, Bars]
    ) -> None:
        # The ordinary path must behave exactly as it did before.
        report = asset_robustness(spec, universe)
        assert report.per_symbol


def _fingerprint(result: object) -> tuple[tuple[str, int, float], ...]:
    return tuple(
        (t.symbol, t.entry_time, round(t.quantity, 6))
        for t in getattr(result, "trades", [])
    )


class TestWarmup:
    def test_the_lookback_is_charged_as_warmup(
        self, cross_spec: StrategySpec, universe: dict[str, Bars]
    ) -> None:
        """rs_rank(60) cannot be known until 60 bars have passed, and trading
        before that would be trading on a NaN."""
        result = run_backtest(cross_spec, universe)
        assert result.warmup_bars >= 60
        first_entry = min((t.entry_time for t in result.trades), default=None)
        assert first_entry is not None
        spy = universe["SPY"]
        assert first_entry >= int(spy.event_time[60])


class TestDeterminism:
    def test_two_identical_runs_produce_identical_trades(
        self, cross_spec: StrategySpec, universe: dict[str, Bars]
    ) -> None:
        a = run_backtest(cross_spec, universe)
        b = run_backtest(cross_spec, universe)
        assert _fingerprint(a) == _fingerprint(b)
        np.testing.assert_array_equal(a.equity, b.equity)
