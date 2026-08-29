"""What a schedule charges a book of a given shape, and why that had to be askable.

The cost model was calibrated for a 3-position event-driven book. The strategy
that survived is a 110-name book rebalanced every five sessions, and two of the
four charges — ``commission_per_share`` and ``min_commission`` — do not scale
with notional. So the same rule, on the same bars, under the same model, is
priced differently depending on how many names it holds and how large the
account is.

That matters because cost retention is a *fatal* gate. Measured on the promoted
strategy, Sharpe retained against the frictionless run goes 73% at $100k and 91%
at $1M, and the only thing that changed was ``initial_equity`` — a number nobody
thinks of as a cost parameter.

The fix is not a cheaper model. A trader with $100k and a hundred-name book at a
broker charging a dollar an order really would pay that. The fix is that the
schedule is nameable, the per-order cost is askable, and the schedule is recorded
with the verdict it decided.
"""

from __future__ import annotations

import pytest

from aqr.backtest.costs import (
    ALPACA_EQUITIES,
    IBKR_FIXED,
    PRESETS,
    ZERO_COST,
    CostModel,
    preset,
)

# --------------------------------------------------------------------------
# The defaults are load-bearing history


def test_the_default_schedule_is_unchanged() -> None:
    """324 experiments were judged under these numbers.

    Recalibrating by editing the defaults would silently reinterpret every one
    of them, and a cost model that gets cheaper the year a strategy needs it to
    is not a cost model. ``IBKR_FIXED`` names what the defaults always were; it
    does not change them.
    """
    assert CostModel() == IBKR_FIXED
    assert IBKR_FIXED.commission_per_share == 0.005
    assert IBKR_FIXED.min_commission == 1.0
    assert IBKR_FIXED.spread_bps == 2.0
    assert IBKR_FIXED.slippage_bps == 1.0


def test_the_commission_free_preset_is_not_the_default() -> None:
    """Making it the default would improve every recorded result at a stroke."""
    assert CostModel() != ALPACA_EQUITIES
    assert ALPACA_EQUITIES.min_commission == 0.0
    assert ALPACA_EQUITIES.commission_per_share == 0.0


def test_the_market_assumptions_survive_the_change_of_broker() -> None:
    """Spread and slippage are properties of the market, not of the broker, and
    they are where the pessimism belongs. Only the commission differs."""
    assert ALPACA_EQUITIES.spread_bps == IBKR_FIXED.spread_bps
    assert ALPACA_EQUITIES.slippage_bps == IBKR_FIXED.slippage_bps


def test_an_unknown_preset_raises_rather_than_defaulting() -> None:
    """A typo that silently priced a run under a different broker than the one
    asked for is the failure this naming exists to prevent."""
    with pytest.raises(ValueError, match="unknown cost preset"):
        preset("robinhood")
    assert preset("ALPACA ") is ALPACA_EQUITIES
    assert set(PRESETS) == {"ibkr_fixed", "alpaca", "zero"}


# --------------------------------------------------------------------------
# The question the model could not answer


def test_the_floor_dominates_a_small_position() -> None:
    """$1.00 on a $192 sleeve position is 52 basis points — against the 3bp the
    spread and slippage charge. That is the mis-calibration, in one number."""
    order = IBKR_FIXED.price_order(notional=192.0, price=100.0)
    assert order.floor_binds
    assert order.fee == pytest.approx(1.0)
    assert order.fee_bps == pytest.approx(52.0, abs=0.5)
    assert order.bps == pytest.approx(55.0, abs=0.5)


def test_the_same_schedule_is_cheap_on_a_concentrated_book() -> None:
    """The three-position book the model was calibrated for. Nothing is wrong
    with the schedule here, which is exactly why the defect went unnoticed."""
    order = IBKR_FIXED.price_order(notional=33_000.0, price=100.0)
    assert not order.floor_binds
    assert order.fee_bps == pytest.approx(0.5, abs=0.1)
    assert order.bps == pytest.approx(3.5, abs=0.1)


def test_the_commission_free_schedule_does_not_depend_on_book_size() -> None:
    """The property that makes a breadth strategy judgeable at all: cost scales
    with turnover and with nothing else."""
    small = ALPACA_EQUITIES.price_order(notional=192.0, price=100.0)
    large = ALPACA_EQUITIES.price_order(notional=33_000.0, price=100.0)
    assert small.bps == pytest.approx(large.bps)
    assert small.bps == pytest.approx(3.0)
    assert not small.floor_binds


def test_the_adverse_charge_scales_with_notional() -> None:
    ten = IBKR_FIXED.price_order(notional=10_000.0, price=50.0)
    twenty = IBKR_FIXED.price_order(notional=20_000.0, price=50.0)
    assert twenty.adverse == pytest.approx(2 * ten.adverse)


def test_a_zero_or_negative_order_costs_nothing() -> None:
    for notional, price in ((0.0, 100.0), (-5.0, 100.0), (100.0, 0.0)):
        order = IBKR_FIXED.price_order(notional=notional, price=price)
        assert order.total == 0.0
        assert order.bps == 0.0
        assert not order.floor_binds


def test_the_frictionless_model_charges_nothing_at_any_size() -> None:
    assert ZERO_COST.price_order(notional=192.0, price=100.0).total == 0.0


def test_shares_follow_from_notional_and_price() -> None:
    order = IBKR_FIXED.price_order(notional=1_000.0, price=40.0)
    assert order.shares == pytest.approx(25.0)


# --------------------------------------------------------------------------
# The schedule travels with the verdict


def test_the_schedule_serialises_for_the_record() -> None:
    """Two runs under different schedules are not comparable, and comparing them
    anyway is how a change of broker gets attributed to a change of strategy —
    the same reason ``dataset_version`` is written with every experiment."""
    payload = IBKR_FIXED.as_dict()
    assert payload["min_commission"] == 1.0
    assert payload["spread_bps"] == 2.0
    assert list(payload) == sorted(payload)
    assert CostModel(**payload) == IBKR_FIXED


def test_the_schedules_are_distinguishable_once_recorded() -> None:
    assert IBKR_FIXED.as_dict() != ALPACA_EQUITIES.as_dict()
