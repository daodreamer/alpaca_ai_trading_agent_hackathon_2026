"""Re-testing a promoted strategy on data the search never touched.

Everything else in this project measures a strategy against the data it was
selected on. Walk-forward helps -- the test fold is unseen *within* a run -- but
the universe, the vendor and the window were fixed before the first hypothesis
was proposed, and 153 hypotheses were tried against them. A best-of-153 result on
one universe is a claim about that universe.

A holdout is the only test the search cannot have leaked into: different symbols,
or a different decade, or a different vendor. It answers the one question the
score cannot, because the score is computed from the same bars that chose the
strategy.

The comparison has to be honest in both directions. A strategy that survives is
not thereby proven -- one holdout is one sample. A strategy that collapses has
told you something definite, and that is the more common and more useful result.
"""

from __future__ import annotations

import numpy as np
import pytest

from aqr.dsl.loader import loads
from aqr.dsl.schema import StrategySpec
from aqr.validation.holdout import HoldoutResult, run_holdout

SPEC = """
strategy:
  name: holdout_probe_v1
  hypothesis: A rule to re-test elsewhere.
  universe:
    symbols: [SPY, QQQ]
    timeframe: 1D
  entry: rsi(14) < 45
  exit:
    stop_loss: {type: atr, multiplier: 2.0, period: 14}
    take_profit: {type: risk_reward, ratio: 2.0}
    max_holding_bars: 20
  sizing:
    risk_per_trade: 0.005
    max_position_pct: 0.25
  max_positions: 2
"""


@pytest.fixture
def spec() -> StrategySpec:
    return loads(SPEC)


class TestUniverseSubstitution:
    def test_it_evaluates_on_the_held_out_symbols(self, spec, universe) -> None:
        held = {"IWM": universe["IWM"]}
        result = run_holdout(spec, held)
        assert result.symbols == ("IWM",)

    def test_the_original_universe_is_not_consulted(self, spec, universe) -> None:
        """The point of a holdout is that the selected-on data plays no part."""
        held = {"IWM": universe["IWM"]}
        result = run_holdout(spec, held)
        assert "SPY" not in result.symbols
        assert "QQQ" not in result.symbols

    def test_the_rule_itself_is_unchanged(self, spec, universe) -> None:
        result = run_holdout(spec, {"IWM": universe["IWM"]})
        assert result.fingerprint == spec.fingerprint(), (
            "substituting the universe must not change what rule is being tested"
        )


class TestVerdict:
    def test_a_rule_that_trades_reports_metrics(self, spec, universe) -> None:
        result = run_holdout(spec, {"IWM": universe["IWM"]})
        assert result.metrics is not None
        assert result.metrics.num_trades > 0

    def test_a_rule_that_never_fires_says_so_rather_than_scoring_zero(
        self, universe
    ) -> None:
        """Zero trades and a zero return are different findings. Reporting the
        second when the first happened invites "it lost money elsewhere", which
        is not what the data said."""
        dead = loads(SPEC.replace("rsi(14) < 45", "rsi(14) > 150"))
        result = run_holdout(dead, {"IWM": universe["IWM"]})
        assert result.traded is False
        assert result.note

    def test_missing_bars_are_reported_not_silently_skipped(self, spec) -> None:
        result = run_holdout(spec, {})
        assert result.traded is False
        assert "no data" in result.note.lower()


class TestComparison:
    def test_it_compares_against_the_selected_on_result(self, spec, universe) -> None:
        held = {"IWM": universe["IWM"]}
        result = run_holdout(spec, held, selected_sharpe=1.20)
        assert result.selected_sharpe == pytest.approx(1.20)
        assert result.retained is not None

    def test_retention_is_the_share_of_the_edge_that_survived(self, spec, universe) -> None:
        held = {"IWM": universe["IWM"]}
        result = run_holdout(spec, held, selected_sharpe=1.0)
        assert result.metrics is not None
        assert result.retained == pytest.approx(result.metrics.sharpe / 1.0, abs=1e-9)

    def test_a_negative_selected_sharpe_yields_no_retention(self, spec, universe) -> None:
        # Dividing by a number that was not an edge produces a ratio that looks
        # like one.
        result = run_holdout(spec, {"IWM": universe["IWM"]}, selected_sharpe=-0.3)
        assert result.retained is None

    def test_without_a_selected_sharpe_it_still_reports_the_holdout(
        self, spec, universe
    ) -> None:
        result = run_holdout(spec, {"IWM": universe["IWM"]})
        assert result.retained is None
        assert result.metrics is not None


class TestReporting:
    def test_the_summary_names_the_symbols_and_the_outcome(self, spec, universe) -> None:
        text = str(run_holdout(spec, {"IWM": universe["IWM"]}, selected_sharpe=1.0))
        assert "IWM" in text
        assert "holdout" in text.lower()

    def test_a_dead_rule_summary_says_it_did_not_trade(self, spec, universe) -> None:
        dead = loads(SPEC.replace("rsi(14) < 45", "rsi(14) > 150"))
        assert "no trades" in str(run_holdout(dead, {"IWM": universe["IWM"]})).lower()

    def test_it_serialises(self, spec, universe) -> None:
        payload = run_holdout(spec, {"IWM": universe["IWM"]}).as_dict()
        assert payload["fingerprint"] == spec.fingerprint()
        assert "symbols" in payload


class TestDeterminism:
    def test_two_runs_agree(self, spec, universe) -> None:
        held = {"IWM": universe["IWM"]}
        a, b = run_holdout(spec, held), run_holdout(spec, held)
        assert a.metrics is not None and b.metrics is not None
        assert a.metrics.sharpe == b.metrics.sharpe
        assert a.metrics.num_trades == b.metrics.num_trades


def test_the_result_is_a_dataclass_not_a_dict() -> None:
    # Typed, so a renamed field is a type error rather than a KeyError in a
    # report someone reads once a month.
    assert hasattr(HoldoutResult, "__dataclass_fields__")


class TestBenchmark:
    """What you get for doing nothing.

    The finding that prompted this. Fourteen strategies promoted to PAPER kept a
    positive Sharpe on twenty-eight symbols they had never been selected on --
    12 of 14, which reads like a success. Equal-weight buy-and-hold on the same
    symbols over the same window returned +4263% at Sharpe 1.16. The best
    strategy returned +57.5% at Sharpe 0.65.

    Every one of them lost to doing nothing, and nothing in the pipeline asked.
    A long-only rule on NASDAQ names over 2010-2026 makes money almost whatever
    the rule is; without the benchmark beside it, that fact reads as evidence.
    """

    def test_it_reports_what_holding_the_same_symbols_would_have_done(
        self, spec, universe
    ) -> None:
        result = run_holdout(spec, {"IWM": universe["IWM"], "QQQ": universe["QQQ"]})
        assert result.benchmark is not None
        assert result.benchmark.num_trades == 0, "buy and hold is not a trading strategy"

    def test_the_benchmark_covers_the_same_window(self, spec, universe) -> None:
        held = {"IWM": universe["IWM"]}
        result = run_holdout(spec, held)
        assert result.benchmark is not None
        assert np.isfinite(result.benchmark.sharpe)

    def test_beating_the_benchmark_is_stated_plainly(self, spec, universe) -> None:
        result = run_holdout(spec, {"IWM": universe["IWM"], "QQQ": universe["QQQ"]})
        assert result.beat_benchmark is not None
        assert "buy and hold" in str(result).lower()

    def test_a_rule_that_does_not_trade_has_no_benchmark_comparison(
        self, universe
    ) -> None:
        dead = loads(SPEC.replace("rsi(14) < 45", "rsi(14) > 150"))
        result = run_holdout(dead, {"IWM": universe["IWM"]})
        assert result.beat_benchmark is None

    def test_the_comparison_is_on_risk_adjusted_return(self, spec, universe) -> None:
        """Not total return. A rule in the market a tenth of the time will lose
        on return to one that is always in it, and that on its own says nothing
        about whether the rule is any good."""
        result = run_holdout(spec, {"IWM": universe["IWM"], "QQQ": universe["QQQ"]})
        assert result.benchmark is not None and result.metrics is not None
        assert result.beat_benchmark == (result.metrics.sharpe > result.benchmark.sharpe)

    def test_it_serialises_the_benchmark(self, spec, universe) -> None:
        payload = run_holdout(spec, {"IWM": universe["IWM"]}).as_dict()
        assert "benchmark" in payload


class TestBenchmarkArithmetic:
    """The benchmark's Sharpe has to be a number.

    It was 0.00 on a series that returned +4263%, so every comparison printed
    "BEAT buy and hold" -- the exact opposite of the truth, in the one line the
    whole holdout exists to produce.

    The cause was reuse: the benchmark curve was wrapped in a fake
    ``BacktestResult`` and handed to ``compute_metrics``, which refuses to
    report a ratio on fewer than five trades. That guard is right for a strategy
    -- a Sharpe from three trades is noise -- and meaningless for buy-and-hold,
    which has no trades by construction and four thousand daily returns.
    """

    def test_a_rising_series_has_a_positive_benchmark_sharpe(self) -> None:
        from datetime import UTC, datetime, timedelta

        from aqr.data.bars import Bars
        from aqr.validation.holdout import buy_and_hold

        n = 500
        close = 100.0 * np.exp(np.arange(n) * 0.001)
        base = datetime(2020, 1, 1, tzinfo=UTC)
        rising = Bars(
            symbol="UP",
            timeframe="1D",
            event_time=np.array(
                [int((base + timedelta(days=i)).timestamp()) for i in range(n)],
                dtype=np.int64,
            ),
            open=close,
            high=close * 1.01,
            low=close * 0.99,
            close=close,
            volume=np.full(n, 1e6),
        )
        metrics = buy_and_hold({"UP": rising})
        assert metrics is not None
        assert metrics.total_return > 0
        assert metrics.sharpe > 0, "a rising curve cannot have a zero Sharpe"

    def test_the_benchmark_sharpe_matches_a_direct_computation(self, universe) -> None:
        from aqr.validation.holdout import buy_and_hold

        bars = universe["SPY"]
        close = np.asarray(bars.close, dtype=float)
        rets = close[1:] / close[:-1] - 1.0
        expected = rets.mean() / rets.std(ddof=1) * np.sqrt(252.0)

        metrics = buy_and_hold({"SPY": bars})
        assert metrics is not None
        assert metrics.sharpe == pytest.approx(expected, rel=1e-6)

    def test_zero_trades_does_not_suppress_the_ratio(self, universe) -> None:
        from aqr.validation.holdout import buy_and_hold

        metrics = buy_and_hold({"SPY": universe["SPY"]})
        assert metrics is not None
        assert metrics.num_trades == 0
        assert metrics.sharpe != 0.0

    def test_full_exposure_is_reported(self, universe) -> None:
        from aqr.validation.holdout import buy_and_hold

        metrics = buy_and_hold({"SPY": universe["SPY"]})
        assert metrics is not None
        assert metrics.exposure == pytest.approx(1.0)

    def test_the_comparison_now_points_the_right_way(self, spec, universe) -> None:
        # SPY over this window is a hard benchmark; a mean-reversion rule that
        # is in the market part of the time should lose to it.
        result = run_holdout(spec, {"SPY": universe["SPY"]})
        assert result.benchmark is not None and result.metrics is not None
        assert result.beat_benchmark == (result.metrics.sharpe > result.benchmark.sharpe)
        assert result.benchmark.sharpe != 0.0
