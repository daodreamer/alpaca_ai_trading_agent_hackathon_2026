"""Structures and their money — specs/10-options-research.md D4, test plan 9–11.

The rule D4 states is that defined risk lives in the type: if it cannot be
constructed, it cannot be proposed and it cannot be measured into existence. So
these tests are half arithmetic and half refusals.

The arithmetic is worth being fussy about because every number downstream is
divided by it. ``fixed_risk`` sizing takes maximum loss as its denominator, so a
maximum loss that is wrong by the width of a spread sizes the whole book wrong
by the same factor, and nothing later in the pipeline can see that it happened.
"""

from __future__ import annotations

from datetime import date

import pytest

from aqr.options.chain import Quote
from aqr.options.structure import Leg, Structure

EXPIRY = date(2023, 9, 29)


def leg(strike: float, right: str, side: str, *, bid: float, ask: float, iv: float = 0.15) -> Leg:
    return Leg(
        quote=Quote(
            expiration=EXPIRY,
            strike=strike,
            right=right,
            bid=bid,
            ask=ask,
            iv=iv,
            delta=-0.16 if right == "put" else 0.16,
            gamma=0.001,
            theta=-0.05,
            vega=0.09,
            rho=0.02,
        ),
        side=side,
    )


def put_credit_spread() -> Structure:
    """Sell the 433 put at 1.50, buy the 423 put at 0.50. Width 10, credit 1.00."""
    return Structure(
        kind="put_credit_spread",
        legs=(
            leg(433.0, "put", "sell", bid=1.50, ask=1.60),
            leg(423.0, "put", "buy", bid=0.45, ask=0.50),
        ),
    )


# --------------------------------------------------------------------------- #
# Entry cash: the full spread is crossed, every time
# --------------------------------------------------------------------------- #


def test_a_credit_spread_collects_the_bid_and_pays_the_ask() -> None:
    """1.50 in on the short leg, 0.50 out on the long one. Not 1.55 and 0.475.

    Crossing the spread in full is D2 and there is no parameter that softens it.
    At the cache's p90 relative spread of 40%, a mid-price fill assumption is
    not optimism, it is a different instrument.
    """
    assert put_credit_spread().entry_cash == pytest.approx(1.00)


def test_a_debit_structure_reports_negative_entry_cash() -> None:
    """Sign convention, pinned once: cash in is positive, cash out is negative.
    Every P&L downstream is ``entry_cash + settlement``, so a flipped sign here
    turns every loss into a gain."""
    long_put = Structure(kind="long_put", legs=(leg(433.0, "put", "buy", bid=1.50, ask=1.60),))
    assert long_put.entry_cash == pytest.approx(-1.60)


# --------------------------------------------------------------------------- #
# Maximum loss is computed, not looked up
# --------------------------------------------------------------------------- #


def test_a_credit_spread_risks_the_width_less_the_credit() -> None:
    assert put_credit_spread().max_loss == pytest.approx(10.0 - 1.00)


def test_a_long_option_risks_exactly_the_premium_paid() -> None:
    long_call = Structure(kind="long_call", legs=(leg(433.0, "call", "buy", bid=2.00, ask=2.20),))
    assert long_call.max_loss == pytest.approx(2.20)


def test_an_iron_condor_risks_its_wider_wing_less_the_credit() -> None:
    """Both sides cannot finish in the money, so the risk is one wing, not two.
    A naive sum of the two spreads' risk would size the position at half what
    the rule asked for."""
    condor = Structure(
        kind="iron_condor",
        legs=(
            leg(433.0, "put", "sell", bid=1.50, ask=1.60),
            leg(423.0, "put", "buy", bid=0.45, ask=0.50),
            leg(460.0, "call", "sell", bid=1.20, ask=1.30),
            leg(475.0, "call", "buy", bid=0.30, ask=0.35),
        ),
    )
    credit = 1.50 - 0.50 + 1.20 - 0.35
    assert condor.entry_cash == pytest.approx(credit)
    assert condor.max_loss == pytest.approx(15.0 - credit)


def test_maximum_loss_is_found_by_evaluating_the_payoff_not_by_a_per_kind_formula() -> None:
    """A formula per structure kind is seven chances to be wrong and no way to
    notice. The payoff is piecewise linear with kinks only at the strikes, so
    its minimum is at a strike or at a boundary, and evaluating there is both
    exact and the same code for every kind."""
    spread = put_credit_spread()
    worst = min(
        spread.entry_cash + spread.settlement_value(spot)
        for spot in (0.0, 400.0, 423.0, 428.0, 433.0, 500.0)
    )
    assert -worst == pytest.approx(spread.max_loss)


# --------------------------------------------------------------------------- #
# Settlement
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("spot", "expected"),
    [
        (450.0, 1.00),  # both expire worthless: keep the credit
        (433.0, 1.00),  # exactly at the short strike: still worthless
        (428.0, -4.00),  # halfway through: short is 5 in the money
        (423.0, -9.00),  # at the long strike: maximum loss
        (400.0, -9.00),  # beyond it: capped, which is the whole point
    ],
)
def test_a_credit_spread_settles_against_the_underlying_close(
    spot: float, expected: float
) -> None:
    spread = put_credit_spread()
    assert spread.entry_cash + spread.settlement_value(spot) == pytest.approx(expected)


def test_a_loss_never_exceeds_the_maximum_loss_at_any_settlement_price() -> None:
    spread = put_credit_spread()
    for spot in range(300, 600, 7):
        assert spread.entry_cash + spread.settlement_value(float(spot)) >= -spread.max_loss - 1e-9


# --------------------------------------------------------------------------- #
# What must not construct (D4)
# --------------------------------------------------------------------------- #


def test_a_naked_short_call_does_not_construct() -> None:
    """Unbounded loss. There is no kind that admits it, and the payoff check
    catches it even if a kind were added carelessly."""
    with pytest.raises(ValueError):
        Structure(kind="call_credit_spread", legs=(leg(433.0, "call", "sell", bid=2.0, ask=2.2),))


def test_a_credit_spread_needs_both_legs() -> None:
    with pytest.raises(ValueError, match="two legs"):
        Structure(kind="put_credit_spread", legs=(leg(433.0, "put", "sell", bid=1.5, ask=1.6),))


def test_a_put_credit_spread_must_sell_the_higher_strike() -> None:
    """Selling the lower strike is a debit spread wearing a credit spread's
    name, with the opposite directional bet and a different maximum loss."""
    with pytest.raises(ValueError, match="higher strike"):
        Structure(
            kind="put_credit_spread",
            legs=(
                leg(423.0, "put", "sell", bid=0.45, ask=0.50),
                leg(433.0, "put", "buy", bid=1.50, ask=1.60),
            ),
        )


def test_legs_of_one_structure_share_an_expiry() -> None:
    """A calendar spread cannot be priced by this engine: the near leg would
    need a quote on the far leg's expiry date, which D0 says is not there."""
    far = Leg(
        quote=Quote(
            expiration=date(2023, 10, 20),
            strike=423.0,
            right="put",
            bid=0.90,
            ask=1.00,
            iv=0.15,
            delta=-0.10,
            gamma=0.001,
            theta=-0.05,
            vega=0.09,
            rho=0.02,
        ),
        side="buy",
    )
    with pytest.raises(ValueError, match="expiry"):
        Structure(kind="put_credit_spread", legs=(leg(433.0, "put", "sell", bid=1.5, ask=1.6), far))


def test_a_structure_whose_short_leg_has_no_bid_does_not_construct() -> None:
    """Opening it means selling into a zero bid, which fills at nothing."""
    with pytest.raises(ValueError, match="bid"):
        Structure(
            kind="put_credit_spread",
            legs=(
                leg(433.0, "put", "sell", bid=0.0, ask=0.05),
                leg(423.0, "put", "buy", bid=0.45, ask=0.50),
            ),
        )


def test_a_structure_with_no_risk_does_not_construct() -> None:
    """A credit larger than the width is an arbitrage, which on this cache means
    a stale or crossed quote rather than free money. ``fixed_risk`` sizing would
    divide by it and ask for an unbounded number of contracts."""
    with pytest.raises(ValueError, match="risk"):
        Structure(
            kind="put_credit_spread",
            legs=(
                leg(433.0, "put", "sell", bid=12.00, ask=12.10),
                leg(423.0, "put", "buy", bid=0.45, ask=0.50),
            ),
        )


# --------------------------------------------------------------------------- #
# The mark (D1a) — a model, walled off from the money
# --------------------------------------------------------------------------- #


def test_the_mark_sits_between_the_extremes_of_the_payoff() -> None:
    spread = put_credit_spread()
    value = spread.mark(spot=440.0, years=0.05)
    assert -spread.max_loss <= spread.entry_cash + value <= spread.entry_cash


def test_the_mark_at_expiry_equals_settlement() -> None:
    """The model and the money must agree at the one moment both are defined,
    or the equity curve jumps on the settlement date for no reason."""
    spread = put_credit_spread()
    assert spread.mark(spot=428.0, years=0.0) == pytest.approx(spread.settlement_value(428.0))


def test_a_short_put_spread_loses_on_the_mark_when_spot_falls() -> None:
    """The property D1a exists for. Under cash accounting this is exactly zero,
    the position shows no beta, and ``alpha.py`` credits the entire return of a
    short premium book to skill."""
    spread = put_credit_spread()
    assert spread.mark(spot=420.0, years=0.05) < spread.mark(spot=450.0, years=0.05)
