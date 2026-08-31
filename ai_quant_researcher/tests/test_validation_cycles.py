"""``independent_cycles`` — specs/10-options-research.md D8.

The one definition the options evaluator's gate and every walk-forward /
robustness report share. Fixtures use bare epoch seconds rather than real
dates so the arithmetic is checkable by eye: a "day" here is just 86,400,
which is also what lets the same tests stand in for the equity side, since
``independent_cycles`` reads only ``entry_time`` / ``exit_time`` off the
generic ``Trade`` view.
"""

from __future__ import annotations

from aqr.backtest.engine import Trade
from aqr.validation.cycles import independent_cycles

DAY = 86_400


def trade(entry_day: int, exit_day: int) -> Trade:
    return Trade(
        symbol="SPY",
        direction="short",
        entry_time=entry_day * DAY,
        entry_price=9.0,
        exit_time=exit_day * DAY,
        exit_price=8.0,
        quantity=100.0,
        gross_pnl=100.0,
        fees=1.0,
        slippage=0.5,
        bars_held=exit_day - entry_day,
        exit_reason="expiry",
        mae=0.0,
        mfe=0.0,
    )


def test_no_trades_is_zero_cycles() -> None:
    assert independent_cycles([]) == 0


def test_a_single_trade_is_one_cycle() -> None:
    assert independent_cycles([trade(0, 28)]) == 1


def test_back_to_back_non_overlapping_trades_are_each_a_cycle() -> None:
    """Entered only after the last one closed: every entry is new evidence."""
    trades = [trade(0, 28), trade(28, 56), trade(56, 84)]
    assert independent_cycles(trades) == 3


def test_entries_opened_while_the_first_is_still_open_do_not_add_cycles() -> None:
    """Five-session cadence, 28 DTE: the classic overlapping put-spread book.
    Every entry until day 28 is correlated with the first and must not count
    as a fresh, independent bet."""
    trades = [trade(day, day + 28) for day in range(0, 25, 5)]  # 0, 5, 10, 15, 20
    # Only day 0 counts; day 28 (the first free day) is not among the entries.
    assert independent_cycles(trades) == 1


def test_a_new_cycle_starts_once_something_actually_closes() -> None:
    trades = [
        trade(0, 28),  # counted; free_from = 28
        trade(10, 38),  # entered before day 28: correlated, skipped
        trade(28, 56),  # entered exactly at day 28: counted; free_from = 56
        trade(40, 70),  # entered before day 56: correlated, skipped
    ]
    assert independent_cycles(trades) == 2


def test_the_count_does_not_depend_on_the_callers_list_order() -> None:
    ordered = [trade(0, 28), trade(28, 56), trade(56, 84)]
    assert independent_cycles(list(reversed(ordered))) == independent_cycles(ordered) == 3


def test_same_session_entries_are_the_least_independent_pair_not_the_most() -> None:
    """Two structures opened on the same day: crediting both as separate
    cycles would treat the *most* correlated pair in the whole dataset as
    though it were two independent bets."""
    trades = [trade(0, 28), trade(0, 30)]
    assert independent_cycles(trades) == 1


def test_matches_the_71_cycle_measurement_shape_on_a_synthetic_28_dte_program() -> None:
    """The same greedy walk ``tests/test_option_cache_claims.py`` uses to
    measure 71 real cycles, run here on a hand-built 28-day-cadence program
    over about two years: entries land every 28 sessions, so every one is
    its own cycle and the count is exactly the number of entries."""
    trades = [trade(day, day + 28) for day in range(0, 28 * 20, 28)]
    assert independent_cycles(trades) == 20
