"""The market regime classifier (architecture section 7).

This exists to close a hole rather than to add a feature. ``regime_robustness``
is 10% of the evaluator's weight, and it needs a label per bar. Only the
synthetic provider could supply one, because only the simulator knows the regime
it generated. On real data the pipeline passed no labels at all, the report was
never built, and the evaluator substituted 0.5 -- so a tenth of every real-data
score was a constant standing in for a measurement.

The property that matters more than the labels themselves is causality. A regime
classifier is a feature like any other: if the label at bar ``t`` moves when bars
after ``t`` arrive, then every regime-attributed result is contaminated, and it
is contaminated in the most flattering possible way -- the classifier would know
which regime the market was *about to* be in.
"""

from __future__ import annotations

import numpy as np
import pytest

from aqr.data.bars import Bars
from aqr.data.providers import SyntheticProvider
from aqr.features.regime import REGIMES, classify, regime_series


@pytest.fixture
def spy(provider: SyntheticProvider) -> Bars:
    from datetime import UTC, datetime

    return provider.load("SPY", datetime(2010, 1, 1, tzinfo=UTC), datetime(2024, 1, 1, tzinfo=UTC))


class TestCausality:
    def test_labels_are_prefix_stable(self, spy: Bars) -> None:
        """Truncate the input, reclassify, and every overlapping label must be
        identical. A classifier that peeks cannot pass this."""
        full = regime_series(spy)
        for cut in (500, 1200, 2500):
            truncated = regime_series(spy.slice(0, cut))
            assert truncated == full[:cut], f"labels changed when truncated at {cut}"

    def test_appending_a_bar_never_rewrites_history(self, spy: Bars) -> None:
        before = regime_series(spy.slice(0, 1000))
        after = regime_series(spy.slice(0, 1001))
        assert after[:1000] == before

    def test_the_classifier_does_not_read_the_clock(self) -> None:
        import ast
        import inspect as py_inspect

        from aqr.features import regime

        tree = ast.parse(py_inspect.getsource(regime))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert node.func.attr not in ("now", "utcnow", "today")


class TestLabels:
    def test_every_bar_gets_a_label_from_the_known_set(self, spy: Bars) -> None:
        labels = regime_series(spy)
        assert len(labels) == len(spy)
        assert set(labels) <= set(REGIMES)

    def test_warmup_bars_are_labelled_unknown_not_guessed(self, spy: Bars) -> None:
        """A trend estimate built from four bars is not a trend estimate. Saying
        so lets ``regime_robustness`` skip those trades instead of attributing
        them to a regime that was never measured."""
        labels = regime_series(spy)
        assert labels[0] == "UNKNOWN"
        assert "UNKNOWN" not in labels[600:]

    def test_a_steady_uptrend_is_labelled_bull(self) -> None:
        bars = _ramp(1200, drift=0.0008, noise=0.002)
        labels = regime_series(bars)
        tail = labels[-400:]
        assert tail.count("TREND_BULL") / len(tail) > 0.7

    def test_a_steady_downtrend_is_labelled_bear(self) -> None:
        bars = _ramp(1200, drift=-0.0008, noise=0.002)
        labels = regime_series(bars)
        tail = labels[-400:]
        assert tail.count("TREND_BEAR") / len(tail) > 0.7

    def test_a_flat_quiet_market_is_labelled_low_vol_range(self) -> None:
        bars = _ramp(1200, drift=0.0, noise=0.001)
        tail = regime_series(bars)[-400:]
        assert tail.count("RANGE_LOW_VOL") / len(tail) > 0.6

    def test_a_flat_violent_market_is_labelled_high_vol_range(self) -> None:
        bars = _ramp(1200, drift=0.0, noise=0.001, shock_from=800, shock=0.045)
        tail = regime_series(bars)[-200:]
        assert tail.count("RANGE_HIGH_VOL") / len(tail) > 0.5


class TestDeterminism:
    def test_the_same_bars_always_produce_the_same_labels(self, spy: Bars) -> None:
        assert regime_series(spy) == regime_series(spy)

    def test_classify_agrees_with_the_series_at_every_index(self, spy: Bars) -> None:
        window = spy.slice(0, 900)
        series = regime_series(window)
        for i in (300, 600, 899):
            assert classify(window, i) == series[i]

    def test_classify_refuses_an_index_past_the_end(self, spy: Bars) -> None:
        with pytest.raises(IndexError):
            classify(spy, len(spy))


class TestShortSeries:
    def test_a_series_shorter_than_the_warmup_is_all_unknown(self) -> None:
        labels = regime_series(_ramp(20, drift=0.001, noise=0.001))
        assert set(labels) == {"UNKNOWN"}

    def test_an_empty_series_yields_no_labels(self) -> None:
        empty = _ramp(300, drift=0.0, noise=0.001).slice(0, 0)
        assert regime_series(empty) == []


def _ramp(
    n: int,
    *,
    drift: float,
    noise: float,
    shock_from: int | None = None,
    shock: float = 0.0,
) -> Bars:
    """A deterministic price path with a known character.

    Seeded rather than random: a regime test that passes on some seeds is a
    regime test that will fail in CI on a day nobody changed anything.
    """
    rng = np.random.default_rng(11)
    steps = np.full(n, drift) + rng.normal(0.0, noise, n)
    if shock_from is not None:
        steps[shock_from:] = rng.normal(0.0, shock, n - shock_from)
    close = 100.0 * np.exp(np.cumsum(steps))
    span = np.abs(steps) * close + close * 0.001
    return Bars(
        symbol="RAMP",
        timeframe="1D",
        event_time=np.arange(n, dtype=np.int64) * 86_400 + 1_600_000_000,
        open=close - steps * close * 0.5,
        high=np.maximum(close, close - steps * close * 0.5) + span,
        low=np.minimum(close, close - steps * close * 0.5) - span,
        close=close,
        volume=np.full(n, 1e6),
    )


class TestUnknownIsNotARegime:
    """``UNKNOWN`` means "not measured", and scoring it as a regime gives a
    strategy credit for being profitable during its own warm-up."""

    def test_unknown_trades_are_reported_but_not_scored(
        self, universe: dict[str, Bars]
    ) -> None:
        from aqr.dsl.schema import StrategySpec, Universe
        from aqr.validation.robustness import UNSCORED_REGIME, regime_robustness

        spec = StrategySpec(
            name="always_in_v1",
            entry="close > 0",
            universe=Universe(symbols=tuple(sorted(universe))),
        )
        labels = {s: regime_series(b) for s, b in universe.items()}
        report = regime_robustness(spec, universe, labels)

        assert UNSCORED_REGIME == "UNKNOWN"
        assert UNSCORED_REGIME not in report.scored_regimes
        if UNSCORED_REGIME in report.per_regime:
            assert report.per_regime[UNSCORED_REGIME].num_trades > 0, (
                "if unknown-regime trades exist they must still be visible"
            )

    def test_a_strategy_profitable_only_during_warmup_scores_zero(self) -> None:
        from aqr.validation.robustness import score_regimes

        winner = [_trade(pnl=100.0) for _ in range(10)]
        loser = [_trade(pnl=-100.0) for _ in range(10)]
        assert score_regimes({"UNKNOWN": winner, "TREND_BULL": loser}) == 0.0
        assert score_regimes({"UNKNOWN": loser, "TREND_BULL": winner}) == 1.0


def _trade(*, pnl: float) -> object:
    from aqr.backtest.engine import Trade

    return Trade(
        symbol="TEST",
        direction="long",
        entry_time=0,
        entry_price=100.0,
        exit_time=86_400,
        exit_price=100.0 + pnl,
        quantity=1.0,
        gross_pnl=pnl,
        fees=0.0,
        slippage=0.0,
        bars_held=1,
        exit_reason="target",
        mae=0.0,
        mfe=0.0,
    )


class TestTimeframeScaling:
    """The windows are wall-clock spans, not fixed bar counts.

    Sixty bars is a quarter on a daily chart and nine days on an hourly one.
    The classifier converts with ``bars_per_year`` so the same hypothesis --
    a quarter of drift, a year of vol history -- is tested at every bar size.
    On daily bars the conversion factor is exactly one, so every label above
    is computed by the same arithmetic as before.
    """

    def test_daily_windows_are_the_daily_constants(self) -> None:
        from aqr.features.regime import LOOKBACK, MIN_VOL_HISTORY, TRADING_DAYS, _windows

        assert _windows("1D") == (LOOKBACK, MIN_VOL_HISTORY, TRADING_DAYS)

    def test_hourly_windows_cover_the_same_wall_clock(self) -> None:
        from aqr.features.regime import _windows

        lookback, min_vol_history, per_year = _windows("1h")
        assert (lookback, min_vol_history, per_year) == (390, 1638, 1638.0)
        # A quarter and a year of trading time, whatever the bar size.
        assert lookback / per_year == pytest.approx(60 / 252.0)
        assert min_vol_history / per_year == pytest.approx(1.0)

    def test_an_hourly_uptrend_is_measured_over_an_hourly_quarter(self) -> None:
        """z is drift * sqrt(lookback) / per-bar noise. With the daily 60-bar
        lookback this path scores z ~ 1.2 and reads as a range; over the
        correct 390-bar quarter it is z ~ 3, a trend."""
        rng = np.random.default_rng(23)
        n = 2400
        steps = 0.0006 + rng.normal(0.0, 0.004, n)
        close = 100.0 * np.exp(np.cumsum(steps))
        bars = Bars(
            symbol="HRLY",
            timeframe="1h",
            event_time=np.arange(n, dtype=np.int64) * 3600 + 1_600_000_000,
            open=close,
            high=close * 1.001,
            low=close * 0.999,
            close=close,
            volume=np.full(n, 1e6),
        )
        tail = regime_series(bars)[-200:]
        assert tail.count("TREND_BULL") / len(tail) > 0.7
        assert "UNKNOWN" not in tail
