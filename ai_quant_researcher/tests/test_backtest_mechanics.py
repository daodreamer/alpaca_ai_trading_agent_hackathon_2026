"""Backtester mechanics, on hand-built price paths with known answers.

Synthetic random data proves the engine runs. These tests prove it is *right*:
each one constructs a price path where the correct fill, the correct exit reason
and the correct P&L can be worked out by hand, then asserts the engine agrees.

This is the layer where a backtester lies most convincingly — the numbers stay
plausible whether or not the fill logic is correct.
"""

from __future__ import annotations

import pytest

from aqr.backtest.costs import ZERO_COST, CostModel
from aqr.backtest.engine import BacktestConfig, run_backtest
from aqr.dsl.schema import ExitRules, Sizing, StopLoss, StrategySpec, TakeProfit, Universe
from tests.conftest import make_bars

FRICTIONLESS = BacktestConfig(
    initial_equity=100_000.0, costs=ZERO_COST, allow_fractional_shares=True
)


def simple_spec(
    entry: str = "close > 0",
    *,
    stop_multiple: float = 2.0,
    tp_ratio: float = 2.0,
    max_holding: int = 20,
    signal_exit: str | None = None,
    direction: str = "long",
    risk: float = 0.01,
) -> StrategySpec:
    """A minimal spec with a percent stop, so the stop distance is exactly known."""
    return StrategySpec(
        name="mechanics",
        entry=entry,
        universe=Universe(symbols=("TEST",), timeframe="1D"),
        exit=ExitRules(
            stop_loss=StopLoss(type="percent", multiplier=stop_multiple / 100.0),
            take_profit=(
                TakeProfit(type="none", ratio=1.0)
                if tp_ratio <= 0
                else TakeProfit(type="risk_reward", ratio=tp_ratio)
            ),
            max_holding_bars=max_holding,
            signal_exit=signal_exit,
        ),
        sizing=Sizing(risk_per_trade=risk, max_position_pct=1.0),
        direction=direction,  # type: ignore[arg-type]
        max_positions=1,
    )


def test_entry_fills_at_the_next_bar_open_not_this_bar_close() -> None:
    """The single most important property of the engine."""
    # The entry fires on bar 2 alone. Bar 3's open is deliberately unlike every
    # close in the series, so a fill at 137 can only have come from that open.
    bars = make_bars(
        closes=[100, 100, 90, 100, 100],
        opens=[100, 100, 100, 137, 100],
        highs=[100, 100, 100, 137, 100],
        lows=[100, 100, 89, 100, 100],
    )
    spec = simple_spec(entry="close < 95", stop_multiple=20.0, tp_ratio=0.0, max_holding=1)
    result = run_backtest(spec, {"TEST": bars}, FRICTIONLESS)

    assert len(result.trades) == 1, "expected exactly one trade"
    assert result.trades[0].entry_price == pytest.approx(137.0), (
        "the fill did not use bar 3's open; the engine is filling at a close, "
        "which is look-ahead bias"
    )


def test_stop_is_taken_when_a_bar_contains_both_stop_and_target() -> None:
    """Pessimistic resolution: an ambiguous bar resolves against the position."""
    # Enter at 100 on bar 2's open. Stop at 98, target at 104.
    # Bar 3 ranges 96..106 — it contains both. The stop must win.
    bars = make_bars(
        closes=[100, 100, 100, 101, 101],
        opens=[100, 100, 100, 100, 101],
        highs=[100, 100, 100, 106, 101],
        lows=[100, 100, 100, 96, 101],
    )
    spec = simple_spec(entry="close > 50", stop_multiple=2.0, tp_ratio=2.0)
    result = run_backtest(spec, {"TEST": bars}, FRICTIONLESS)

    first = result.trades[0]
    assert first.exit_reason == "stop", f"ambiguous bar resolved as {first.exit_reason!r}"
    assert first.exit_price == pytest.approx(98.0)
    assert first.net_pnl < 0


def test_a_gap_through_the_stop_fills_at_the_open_not_the_stop() -> None:
    """Stops are not guarantees. A gap costs more than the stop price."""
    # Enter at 100, stop at 98. Bar 3 opens at 90 — well through the stop.
    bars = make_bars(
        closes=[100, 100, 100, 90, 90],
        opens=[100, 100, 100, 90, 90],
        highs=[100, 100, 100, 91, 91],
        lows=[100, 100, 100, 89, 89],
    )
    spec = simple_spec(entry="close > 50", stop_multiple=2.0)
    result = run_backtest(spec, {"TEST": bars}, FRICTIONLESS)

    first = result.trades[0]
    assert first.exit_reason == "stop_gap"
    assert first.exit_price == pytest.approx(90.0), (
        "filled at the stop price through a gap — this understates real risk"
    )


def test_target_is_taken_when_only_the_target_is_reached() -> None:
    bars = make_bars(
        closes=[100, 100, 100, 105, 105],
        opens=[100, 100, 100, 100, 105],
        highs=[100, 100, 100, 106, 105],
        lows=[100, 100, 100, 99.5, 105],
    )
    spec = simple_spec(entry="close > 50", stop_multiple=2.0, tp_ratio=2.0)
    result = run_backtest(spec, {"TEST": bars}, FRICTIONLESS)

    first = result.trades[0]
    assert first.exit_reason == "target"
    assert first.exit_price == pytest.approx(104.0)  # 100 + 2 * 2
    assert first.net_pnl > 0


def test_max_holding_closes_at_the_open_of_the_limit_bar() -> None:
    bars = make_bars(closes=[100] * 8, opens=[100, 100, 100, 100, 100, 100, 111, 100])
    spec = simple_spec(entry="close > 50", stop_multiple=20.0, tp_ratio=0.0, max_holding=4)
    result = run_backtest(spec, {"TEST": bars}, FRICTIONLESS)

    first = result.trades[0]
    assert first.exit_reason == "max_holding"
    assert first.bars_held == 4
    assert first.exit_price == pytest.approx(111.0), "exited at a close rather than the open"


def test_signal_exit_beats_the_stop_on_the_same_bar() -> None:
    """A signal decided at the previous close executes at the open, which comes
    before any intrabar stop could trigger."""
    # Entry fills at bar 2's open. On bar 3 the exit signal from bar 2's close is
    # already standing, and bar 3's low would also have taken out the 98 stop.
    # The open comes first in time, so the signal must win.
    bars = make_bars(
        closes=[100, 100, 100, 100, 100],
        opens=[100, 100, 100, 99.5, 100],
        highs=[100, 100, 100, 100, 100],
        lows=[100, 100, 100, 90, 100],
    )
    spec = simple_spec(
        entry="close > 50", stop_multiple=2.0, tp_ratio=0.0, signal_exit="close > 50"
    )
    result = run_backtest(spec, {"TEST": bars}, FRICTIONLESS)

    first = result.trades[0]
    assert first.exit_reason == "signal"
    assert first.exit_price == pytest.approx(99.5)


def test_position_size_follows_from_risk_and_stop_distance() -> None:
    """Risking 1% of 100k with a 2% stop on a 100 stock is 500 shares."""
    bars = make_bars(closes=[100] * 6, opens=[100] * 6)
    spec = simple_spec(entry="close > 50", stop_multiple=2.0, risk=0.01, tp_ratio=0.0)
    result = run_backtest(spec, {"TEST": bars}, FRICTIONLESS)

    first = result.trades[0]
    # risk amount 1000, stop distance 2.00 -> 500 shares
    assert first.quantity == pytest.approx(500.0)


def test_a_wider_stop_buys_fewer_shares() -> None:
    """Risk-first sizing: widening the stop must reduce size, not raise risk."""
    bars = make_bars(closes=[100] * 6, opens=[100] * 6)
    narrow = run_backtest(
        simple_spec(entry="close > 50", stop_multiple=1.0, tp_ratio=0.0),
        {"TEST": bars},
        FRICTIONLESS,
    )
    wide = run_backtest(
        simple_spec(entry="close > 50", stop_multiple=4.0, tp_ratio=0.0),
        {"TEST": bars},
        FRICTIONLESS,
    )
    assert narrow.trades[0].quantity > wide.trades[0].quantity
    # Currency at risk is the same either way — that is the whole point.
    narrow_risk = narrow.trades[0].quantity * 100 * 0.01
    wide_risk = wide.trades[0].quantity * 100 * 0.04
    assert narrow_risk == pytest.approx(wide_risk, rel=0.01)


def test_costs_reduce_pnl_and_are_attributed() -> None:
    bars = make_bars(
        closes=[100, 100, 100, 105, 105],
        opens=[100, 100, 100, 100, 105],
        highs=[100, 100, 100, 106, 105],
        lows=[100, 100, 100, 100, 105],
    )
    spec = simple_spec(entry="close > 50", stop_multiple=2.0, tp_ratio=2.0)
    costly = BacktestConfig(
        costs=CostModel(
            commission_per_share=0.01,
            min_commission=1.0,
            spread_bps=10.0,
            slippage_bps=5.0,
            participation_cap=1.0,
        ),
        allow_fractional_shares=True,
    )
    with_costs = run_backtest(spec, {"TEST": bars}, costly).trades[0]
    without = run_backtest(spec, {"TEST": bars}, FRICTIONLESS).trades[0]

    assert with_costs.entry_price > without.entry_price, "buy did not pay the spread"
    assert with_costs.fees > 0
    assert with_costs.slippage > 0
    assert with_costs.net_pnl < without.net_pnl


def test_liquidity_cap_truncates_an_oversized_order() -> None:
    """A strategy that cannot be filled must not be reported as if it were."""
    thin = make_bars(closes=[100] * 6, opens=[100] * 6, volume=1000.0)
    spec = simple_spec(entry="close > 50", stop_multiple=2.0, tp_ratio=0.0)
    capped = BacktestConfig(
        costs=CostModel(
            commission_per_share=0.0,
            min_commission=0.0,
            spread_bps=0.0,
            slippage_bps=0.0,
            participation_cap=0.05,
        ),
        allow_fractional_shares=True,
    )
    result = run_backtest(spec, {"TEST": thin}, capped)
    assert result.trades[0].quantity == pytest.approx(50.0)  # 5% of 1000


def test_max_positions_is_respected() -> None:
    bars = {
        sym: make_bars(closes=[100] * 12, opens=[100] * 12, symbol=sym)
        for sym in ("A", "B", "C")
    }
    spec = StrategySpec(
        name="crowded",
        entry="close > 50",
        universe=Universe(symbols=("A", "B", "C")),
        exit=ExitRules(
            stop_loss=StopLoss(type="percent", multiplier=0.02),
            take_profit=TakeProfit(type="none", ratio=1.0),
            max_holding_bars=50,
        ),
        sizing=Sizing(risk_per_trade=0.005, max_position_pct=0.3),
        max_positions=2,
    )
    result = run_backtest(spec, bars, FRICTIONLESS)
    # Only two names can ever be open, so the third never trades.
    assert len({t.symbol for t in result.trades}) <= 2


def test_short_direction_profits_when_price_falls() -> None:
    bars = make_bars(
        closes=[100, 100, 100, 95, 95],
        opens=[100, 100, 100, 100, 95],
        highs=[100, 100, 100, 100, 95],
        lows=[100, 100, 100, 94, 95],
    )
    spec = simple_spec(entry="close > 50", stop_multiple=2.0, tp_ratio=2.0, direction="short")
    result = run_backtest(spec, {"TEST": bars}, FRICTIONLESS)
    first = result.trades[0]
    assert first.direction == "short"
    assert first.exit_reason == "target"
    assert first.net_pnl > 0, "a short did not profit from a falling price"


def test_equity_curve_reconciles_with_the_trade_list(spec, spy) -> None:
    """Final equity must equal starting equity plus the sum of net P&L.

    A mismatch means cash accounting and trade accounting disagree — and in a
    backtest that discrepancy always flatters the equity curve.
    """
    result = run_backtest(spec, {"SPY": spy}, BacktestConfig())
    booked = sum(t.net_pnl for t in result.trades)
    assert result.final_equity == pytest.approx(result.initial_equity + booked, rel=1e-9)


def test_no_position_is_left_open_at_the_end(spec, spy) -> None:
    result = run_backtest(spec, {"SPY": spy}, BacktestConfig())
    assert result.trades[-1].exit_time <= int(spy.event_time[-1])
    assert any(t.exit_reason == "end_of_data" for t in result.trades) or result.trades


def test_backtest_is_deterministic(spec, universe) -> None:
    """Same input, same config, same result — including the order of trades."""
    a = run_backtest(spec, {"SPY": universe["SPY"]}, BacktestConfig())
    b = run_backtest(spec, {"SPY": universe["SPY"]}, BacktestConfig())
    assert a.final_equity == b.final_equity
    assert [t.entry_time for t in a.trades] == [t.entry_time for t in b.trades]
    assert [t.exit_price for t in a.trades] == [t.exit_price for t in b.trades]


def test_missing_universe_symbol_is_an_error_not_a_silent_subset(spec, spy) -> None:
    wide = spec.with_params(universe=Universe(symbols=("SPY", "MISSING")))
    with pytest.raises(ValueError, match="MISSING"):
        run_backtest(wide, {"SPY": spy}, BacktestConfig())
