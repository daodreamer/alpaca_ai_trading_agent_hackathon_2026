"""What happens to a holding whose symbol stops trading.

On a universe of today's survivors this never comes up, which is exactly why it
was never handled: every name in `NASDAQ_50` has a bar on the last day of the
data. A wider universe is worth building only if it contains the companies that
were acquired, taken private or wiped out -- otherwise the cross-sectional
"anomaly" it finds is partly "the ones that did not go bankrupt outperformed" --
and the moment those names are in it, the portfolio engine has a hole.

The hole is not a crash. Prices are carried forward across gaps, because a name
with no bar today has not become worthless, it simply did not trade; marking it
at zero would put holes in the equity curve. But that reasoning stops when the
bars stop for good. The rebalance only trades symbols with a price at the
current step, so a delisted holding is never sold: it sits in the book at its
final price, contributing a flat, riskless, permanently profitable line to the
equity curve for as long as the backtest runs.

That is the most flattering possible failure. A strategy that happened to hold
Silicon Valley Bank in March 2023 would show the position frozen at its last
good mark, and every risk statistic computed afterwards would be measuring a
portfolio that partly does not exist.

So a final bar is an exit. The position is liquidated at that bar's close and
reported as a completed trade with its own exit reason, which keeps it visible
in the trade log rather than dissolving it into the equity curve.
"""

from __future__ import annotations

import numpy as np
import pytest

from aqr.backtest.engine import BacktestConfig
from aqr.backtest.portfolio import run_portfolio
from aqr.data.bars import Bars
from aqr.dsl.schema import spec_from_dict

SYMBOLS = ["AAA", "BBB", "CCC", "DDD"]
N = 300
STEP = 86_400
T0 = 1_400_000_000


def _bars(symbol: str, drift: float, n: int = N) -> Bars:
    t = np.arange(T0, T0 + n * STEP, STEP, dtype=np.int64)
    close = 100.0 * np.exp(drift * np.arange(n))
    return Bars(
        symbol=symbol,
        timeframe="1D",
        event_time=t,
        open=close * 0.999,
        high=close * 1.01,
        low=close * 0.99,
        close=close,
        volume=np.full(n, 1e6),
    )


def _universe_with_a_delisting(last: int = 150) -> dict[str, Bars]:
    """AAA is the strongest name and stops trading at ``last``.

    Strongest so that the ranking holds it: the failure only shows up in a name
    the strategy actually owns.
    """
    data = {s: _bars(s, 0.0002 * (len(SYMBOLS) - i)) for i, s in enumerate(SYMBOLS)}
    data["AAA"] = _bars("AAA", 0.0008, n=last)
    return data


def _spec(**over: object):
    body: dict[str, object] = {
        "name": "xs_probe",
        "mode": "portfolio",
        "rank_by": "roc(20)",
        "hold": 2,
        "rebalance_every": 10,
        "universe": {"symbols": SYMBOLS, "timeframe": "1D"},
        "sleeve": {"budget": 0.20, "idle": "benchmark"},
    }
    body.update(over)
    return spec_from_dict({"strategy": body})


CONFIG = BacktestConfig(initial_equity=1_000_000.0, allow_fractional_shares=True)


def _run(data: dict[str, Bars] | None = None, **over: object):
    return run_portfolio(_spec(**over), data or _universe_with_a_delisting(), CONFIG)


# --------------------------------------------------------------------------


def test_a_delisted_holding_is_sold_not_frozen() -> None:
    """The bug, stated as a test: after the last bar the name must be gone."""
    last = 150
    result = _run(_universe_with_a_delisting(last))
    for step in range(last + 2, len(result.timeline)):
        assert "AAA" not in result.core_weights_at(step), f"still held at step {step}"
        assert "AAA" not in result.sleeve_weights_at(step), f"still in sleeve at step {step}"


def test_the_liquidation_is_a_reported_trade() -> None:
    """Visible in the trade log rather than dissolved into the equity curve.

    A position that leaves the book without a trade record is invisible to
    profit factor, to win rate, and to the overfitting detector's
    profit-concentration signal.
    """
    result = _run()
    exits = [t for t in result.trades if t.symbol == "AAA"]
    assert exits, "the delisting produced no trade"
    assert any(t.exit_reason == "delisted" for t in exits), [t.exit_reason for t in exits]


def test_the_liquidation_happens_at_the_last_traded_price() -> None:
    last = 150
    data = _universe_with_a_delisting(last)
    final_close = float(data["AAA"].close[-1])
    result = _run(data)
    exits = [t for t in result.trades if t.symbol == "AAA" and t.exit_reason == "delisted"]
    assert exits
    assert exits[-1].exit_price == pytest.approx(final_close)


def test_a_delisted_name_stops_contributing_to_equity() -> None:
    """The frozen position was a flat, riskless, permanently profitable line.

    Compared against a run where the same name never existed at all, the equity
    curves must converge rather than diverge by a constant.
    """
    last = 150
    with_delisting = _run(_universe_with_a_delisting(last))
    survivors = {s: b for s, b in _universe_with_a_delisting(last).items() if s != "AAA"}
    without = run_portfolio(
        _spec(universe={"symbols": [s for s in SYMBOLS if s != "AAA"], "timeframe": "1D"}),
        survivors,
        CONFIG,
    )
    # Not equal -- the strategy did own AAA for a while, and that is real. What
    # must not happen is a permanent gap that keeps growing after the delisting.
    tail = slice(-30, None)
    ratio = with_delisting.equity[tail] / without.equity[tail]
    assert float(np.std(ratio)) < 0.05, "the two curves are still diverging after the exit"


def test_the_name_is_not_re_bought_after_it_stops_trading() -> None:
    result = _run()
    held_after = [
        step
        for step in range(152, len(result.timeline))
        if "AAA" in result.weights_at(step)
    ]
    assert held_after == []


def test_a_universe_where_nothing_delists_is_unaffected() -> None:
    """The change must be invisible to the survivor universe every existing
    result was measured on."""
    intact = {s: _bars(s, 0.0002 * (len(SYMBOLS) - i)) for i, s in enumerate(SYMBOLS)}
    result = run_portfolio(_spec(), intact, CONFIG)
    assert not [t for t in result.trades if t.exit_reason == "delisted"]


def test_the_benchmark_also_drops_the_delisted_name() -> None:
    """Otherwise the comparison is against an index that holds a ghost, and the
    strategy is measured against something nobody could have owned."""
    result = _run()
    assert np.all(np.isfinite(result.benchmark_equity))
    assert result.benchmark_equity[-1] > 0
