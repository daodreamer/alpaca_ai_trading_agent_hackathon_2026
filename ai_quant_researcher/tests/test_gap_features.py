"""Overnight features, and why the vocabulary was missing them.

Driven by the campaign log rather than by taste. Across two DeepSeek campaigns
the model proposed at least six overnight hypotheses -- ``short_up_gap_fade``,
``post_gap_fill_rejection``, ``long_overnight_return_reversal``,
``gap_fill_momentum_reversal`` and others -- and two of them scored well enough
to reach REVIEW, so the theme carries signal.

None of them could be written down. The DSL had no way to reach a *previous*
bar's value: ``close`` means this bar's close, and there is no lag. An overnight
gap is ``open[t] / close[t-1] - 1``, and that expression was not expressible. So
the model approximated it with conditions that meant something else, and the
approximations mostly fired on nothing.

The fix is two features, not a general lag operator. ``prev(n)`` would let a
model write ``prev(1) > prev(2) > prev(3)`` and spend three degrees of freedom
rediscovering ``slope(3)``; the overfitting detector charges for parameters, but
the cheapest defence is a vocabulary that does not invite the mistake.
"""

from __future__ import annotations

import numpy as np
import pytest

from aqr.data.bars import Bars
from aqr.features.engine import FeatureFrame, FeatureKey
from aqr.features.registry import REGISTRY


def _bars(opens: list[float], closes: list[float]) -> Bars:
    n = len(closes)
    o = np.array(opens, dtype=np.float64)
    c = np.array(closes, dtype=np.float64)
    return Bars(
        symbol="TEST",
        timeframe="1D",
        event_time=np.arange(n, dtype=np.int64) * 86_400 + 1_600_000_000,
        open=o,
        high=np.maximum(o, c) * 1.01,
        low=np.minimum(o, c) * 0.99,
        close=c,
        volume=np.full(n, 1e6),
    )


def _value(bars: Bars, name: str, *args: float) -> np.ndarray:
    return FeatureFrame(bars).get(FeatureKey(name, tuple(args)))


class TestRegistered:
    @pytest.mark.parametrize("name", ["gap", "overnight_return", "prev_close"])
    def test_the_feature_exists_and_is_documented(self, name: str) -> None:
        assert name in REGISTRY
        assert REGISTRY[name].doc.strip()


class TestGap:
    def test_a_flat_open_is_a_zero_gap(self) -> None:
        gap = _value(_bars([100.0, 100.0, 100.0], [100.0, 100.0, 100.0]), "gap")
        assert gap[1] == pytest.approx(0.0)
        assert gap[2] == pytest.approx(0.0)

    def test_an_up_gap_is_positive_and_scaled_by_the_previous_close(self) -> None:
        # Closed at 100, opened at 105: a +5% gap.
        gap = _value(_bars([100.0, 105.0], [100.0, 103.0]), "gap")
        assert gap[1] == pytest.approx(0.05)

    def test_a_down_gap_is_negative(self) -> None:
        gap = _value(_bars([100.0, 96.0], [100.0, 97.0]), "gap")
        assert gap[1] == pytest.approx(-0.04)

    def test_the_first_bar_has_no_previous_close_and_is_not_guessed(self) -> None:
        """Zero would read as "opened flat", which is a claim about a session
        nobody observed. NaN keeps the bar out of every comparison instead."""
        gap = _value(_bars([100.0, 105.0], [100.0, 103.0]), "gap")
        assert np.isnan(gap[0])


class TestOvernightReturn:
    def test_it_measures_close_to_next_open_not_close_to_close(self) -> None:
        # The distinction is the whole point: close-to-close is roc(1).
        bars = _bars([100.0, 110.0], [100.0, 101.0])
        assert _value(bars, "overnight_return")[1] == pytest.approx(0.10)
        assert _value(bars, "gap")[1] == pytest.approx(0.10)

    def test_it_is_the_same_quantity_as_gap(self) -> None:
        """Two names for one number, because a model reaching for "overnight
        return" should not have to guess that we called it "gap"."""
        bars = _bars([100.0, 105.0, 99.0], [100.0, 103.0, 100.0])
        np.testing.assert_allclose(
            _value(bars, "gap"), _value(bars, "overnight_return"), equal_nan=True
        )


class TestPrevClose:
    def test_it_is_the_close_of_the_bar_before(self) -> None:
        prev = _value(_bars([100.0, 105.0, 99.0], [100.0, 103.0, 101.0]), "prev_close")
        assert np.isnan(prev[0])
        assert prev[1] == pytest.approx(100.0)
        assert prev[2] == pytest.approx(103.0)

    def test_it_never_reaches_forward(self) -> None:
        bars = _bars([100.0, 105.0, 99.0], [100.0, 103.0, 101.0])
        prev = _value(bars, "prev_close")
        assert prev[1] != pytest.approx(101.0), "that is the *next* bar's close"


class TestCausality:
    @pytest.mark.parametrize("name", ["gap", "overnight_return", "prev_close"])
    def test_the_feature_is_prefix_stable(self, name: str, spy: Bars) -> None:
        """Truncate the bars, recompute, and the overlap must be bit-identical.
        Everything in the registry is held to this; a new feature is not
        exempt."""
        full = _value(spy, name)
        for cut in (200, 900, 1800):
            np.testing.assert_array_equal(_value(spy.slice(0, cut), name), full[:cut])

    @pytest.mark.parametrize("name", ["gap", "overnight_return", "prev_close"])
    def test_one_bar_of_warmup_is_declared(self, name: str) -> None:
        # The backtester refuses to trade until the largest warmup has elapsed.
        # A feature that under-declares its warmup trades on a NaN.
        assert REGISTRY[name].warmup(()) >= 2


class TestUsableInAStrategy:
    def test_a_gap_rule_compiles_and_fires(self, spy: Bars) -> None:
        from aqr.dsl.schema import StrategySpec, Universe
        from aqr.dsl.validator import validate_against

        spec = StrategySpec(
            name="gap_fade_v1",
            entry="gap() < -0.01 and rsi(14) < 45",
            universe=Universe(symbols=("SPY",)),
            hypothesis="An overnight fall is partly liquidity, not information.",
        )
        report = validate_against(spec, spy)
        assert report.ok, report.errors
