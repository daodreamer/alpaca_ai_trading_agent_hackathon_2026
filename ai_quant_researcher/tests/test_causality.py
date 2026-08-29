"""The look-ahead tests.

Architecture section 11 calls look-ahead bias the highest-priority failure mode,
and it is the one that cannot be caught by inspection -- an indicator that peeks
one bar ahead looks completely normal and simply produces better results.

The test here is mechanical and hard to fool: compute a series on the full data,
compute it again on a truncated prefix, and assert the overlapping values are
identical. A function that reads the future cannot pass, because on the truncated
input the future is not there.

The same property is then asserted end to end: a backtest over the first N bars
must produce exactly the trades that a backtest over all bars produced in that
window. If future bars influenced anything, the two diverge.
"""

from __future__ import annotations

import numpy as np
import pytest

from aqr.backtest.engine import BacktestConfig, run_backtest
from aqr.core import indicators as ind
from aqr.data.bars import Bars
from aqr.dsl.expr import evaluate, parse
from aqr.features.engine import FeatureFrame, FeatureKey
from aqr.features.registry import REGISTRY

CUT = 400


def _assert_prefix_stable(full: np.ndarray, prefix: np.ndarray, label: str) -> None:
    overlap = prefix.size
    a, b = full[:overlap], prefix
    both_nan = np.isnan(a) & np.isnan(b)
    close = np.isclose(a, b, rtol=1e-9, atol=1e-9, equal_nan=True)
    bad = np.flatnonzero(~(both_nan | close))
    assert bad.size == 0, (
        f"{label} changed when future bars were removed, first at index {bad[:5].tolist()}: "
        f"{a[bad[:3]]} vs {b[bad[:3]]} -- this is look-ahead bias"
    )


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_every_feature_is_causal(name: str, spy: Bars, universe: dict[str, Bars]) -> None:
    """No feature's value at bar i may depend on bars after i.

    Cross-sectional features are held to the same standard, with the whole
    universe truncated together -- which is the honest prefix test for them: on
    the day the data ends, nobody knew what any peer would do next either.
    """
    from aqr.features.cross_section import CrossSection

    args_by_arity: dict[int, tuple[float, ...]] = {
        0: (),
        1: (14.0,),
        2: (20.0, 2.0),
        3: (12.0, 26.0, 9.0),
    }
    spec = REGISTRY[name]
    key = FeatureKey(name, args_by_arity[spec.arity])

    if spec.cross_sectional:
        full_frame = FeatureFrame(universe["SPY"], CrossSection(universe))
        cut = {s: b.slice(0, CUT) for s, b in universe.items()}
        prefix_frame = FeatureFrame(cut["SPY"], CrossSection(cut))
    else:
        full_frame = FeatureFrame(spy)
        prefix_frame = FeatureFrame(spy.slice(0, CUT))

    _assert_prefix_stable(full_frame.get(key), prefix_frame.get(key), f"feature {key}")


def test_a_cross_sectional_feature_refuses_a_lone_symbol(spy: Bars) -> None:
    """Silently returning NaN would make the strategy fire on nothing and be
    rejected for "never fires" -- a true statement about the wrong problem, and
    one that sends the next person hunting for a bad threshold."""
    with pytest.raises(ValueError, match="cross-sectional"):
        FeatureFrame(spy).get(FeatureKey("rs_rank", (60.0,)))


def test_vol_pct_is_causal(spy: Bars) -> None:
    """A percentile rank is the classic place a whole-sample statistic sneaks in."""
    key = FeatureKey("vol_pct", (20.0, 252.0))
    full = FeatureFrame(spy).get(key)
    prefix = FeatureFrame(spy.slice(0, 800)).get(key)
    _assert_prefix_stable(full, prefix, "vol_pct(20, 252)")


def test_percentile_rank_uses_only_trailing_window() -> None:
    """A value that is the maximum of its window must rank at the top, and a
    later spike must not retroactively lower an earlier rank."""
    x = np.array([1.0, 2.0, 3.0, 100.0, 1.0], dtype=np.float64)
    ranks = ind.percentile_rank(x, period=3)
    assert np.isnan(ranks[0]) and np.isnan(ranks[1])
    assert ranks[2] == pytest.approx(2 / 3)  # 3.0 beats 1.0 and 2.0
    assert ranks[3] == pytest.approx(2 / 3)  # 100 beats 2.0 and 3.0


def test_expression_evaluation_is_causal(spy: Bars) -> None:
    node = parse("close > ema(200) and rsi(14) > 50 and rvol(20) > 1.2")
    full = np.asarray(evaluate(node, FeatureFrame(spy)), dtype=bool)
    prefix = np.asarray(evaluate(node, FeatureFrame(spy.slice(0, CUT))), dtype=bool)
    assert (full[:CUT] == prefix).all(), "entry signal changed once the future was removed"


def test_backtest_prefix_produces_identical_trades(spec, spy: Bars) -> None:
    """The end-to-end guarantee: truncating the data cannot change past trades."""
    config = BacktestConfig()
    full = run_backtest(spec, {"SPY": spy}, config)
    cut = 1500
    prefix = run_backtest(spec, {"SPY": spy.slice(0, cut)}, config)
    boundary = int(spy.event_time[cut - 1])

    # Compare only trades that had fully resolved before the truncation point;
    # a position still open at the cut is closed early by definition.
    full_closed = [t for t in full.trades if t.exit_time < boundary]
    prefix_closed = [t for t in prefix.trades if t.exit_time < boundary]

    assert len(full_closed) == len(prefix_closed)
    for a, b in zip(full_closed, prefix_closed, strict=True):
        assert a.entry_time == b.entry_time
        assert a.exit_time == b.exit_time
        assert a.entry_price == pytest.approx(b.entry_price)
        assert a.exit_price == pytest.approx(b.exit_price)
        assert a.quantity == pytest.approx(b.quantity)
        assert a.exit_reason == b.exit_reason


def test_as_of_filters_on_availability_not_event_time() -> None:
    """A bar that had happened but had not yet reached us must be invisible."""
    from datetime import UTC, datetime

    import numpy as np

    stamps = np.array([1_600_000_000, 1_600_086_400], dtype=np.int64)
    # The second bar happened on day two but only became available on day three.
    available = np.array([1_600_000_000, 1_600_172_800], dtype=np.int64)
    bars = Bars(
        symbol="LATE",
        timeframe="1D",
        event_time=stamps,
        available_time=available,
        open=np.array([10.0, 11.0]),
        high=np.array([10.5, 11.5]),
        low=np.array([9.5, 10.5]),
        close=np.array([10.0, 11.0]),
        volume=np.array([1e6, 1e6]),
    )
    at_day_two = datetime.fromtimestamp(1_600_086_400, tz=UTC)
    assert len(bars.as_of(at_day_two)) == 1, "a not-yet-published bar leaked into the past"
    at_day_three = datetime.fromtimestamp(1_600_172_800, tz=UTC)
    assert len(bars.as_of(at_day_three)) == 2
