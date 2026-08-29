"""04 D2 — the sign flip. Test plan item 1.

This is the file that stands between the project and an order that is wrong by
twice the premium and fills instantly. Everything here is hand-computed; there
is not a single assertion that recomputes the value under test using the code
under test.

Two conversions, both dangerous, both tested in both directions:

* **the flip** — domain says credit-positive, Alpaca says debit-positive;
* **the scale** — `net_premium` is total cash for the structure, `limit_price`
  is per share for one strategy unit. Getting this wrong is a factor of a
  hundred, in the direction of "why did that fill immediately".
"""

from __future__ import annotations

from decimal import Decimal

import pytest
from hypothesis import given
from hypothesis import strategies as st

from alphagate.core.errors import InvariantViolation
from alphagate.execution.pricing import (
    alpaca_limit_price,
    alpaca_limit_price_inverse,
    net_premium_per_unit,
)
from alphagate.risk import Intent
from tests.execution.conftest import call_debit_spread, gated, put_credit_spread


class TestTheFlip:
    """Hand-computed, both directions. specs/04 D2's table, as assertions."""

    def test_a_credit_goes_out_negative(self) -> None:
        """Domain: +1.50 received. Alpaca: -1.50, "negative = proceeds"."""
        assert alpaca_limit_price(Decimal("1.50")) == "-1.50"

    def test_a_debit_goes_out_positive(self) -> None:
        """Domain: -2.36 paid. Alpaca: +2.36, "positive = cost"."""
        assert alpaca_limit_price(Decimal("-2.36")) == "2.36"

    def test_reading_back_a_credit(self) -> None:
        assert alpaca_limit_price_inverse("-1.50") == Decimal("1.50")

    def test_reading_back_a_debit(self) -> None:
        assert alpaca_limit_price_inverse("2.36") == Decimal("-2.36")

    def test_zero_never_goes_out_as_negative_zero(self) -> None:
        """`-Decimal(0)` formats as `-0.00`, which is a price that means nothing
        and looks alarming in a journal."""
        assert alpaca_limit_price(Decimal(0)) == "0.00"

    def test_the_wire_value_is_never_scientific_notation(self) -> None:
        """`str(Decimal("0E-2"))` is `"0E-2"`, which is not a price."""
        assert alpaca_limit_price(Decimal("0.00")) == "0.00"
        assert "E" not in alpaca_limit_price(Decimal("0.001"))

    @pytest.mark.parametrize("bad", [1.5, "1.50", None])
    def test_a_non_decimal_price_is_refused(self, bad: object) -> None:
        """A float here is a rounding error on a live order."""
        with pytest.raises(InvariantViolation, match="must be Decimal"):
            alpaca_limit_price(bad)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad", ["NaN", "Infinity", "-Infinity"])
    def test_a_non_finite_price_is_refused(self, bad: str) -> None:
        with pytest.raises(InvariantViolation, match="must be finite"):
            alpaca_limit_price(Decimal(bad))


class TestRoundTrip:
    """The property the spec names: inverse ∘ forward is identity."""

    @given(
        cents=st.integers(min_value=-100_000, max_value=100_000),
    )
    def test_flip_then_unflip_is_identity(self, cents: int) -> None:
        original = Decimal(cents).scaleb(-2)
        assert alpaca_limit_price_inverse(alpaca_limit_price(original)) == original

    @given(cents=st.integers(min_value=-100_000, max_value=100_000))
    def test_unflip_then_flip_is_identity(self, cents: int) -> None:
        wire = f"{Decimal(cents).scaleb(-2):.2f}"
        assert alpaca_limit_price(alpaca_limit_price_inverse(wire)) == wire

    @given(cents=st.integers(min_value=1, max_value=100_000))
    def test_the_flip_always_changes_the_sign(self, cents: int) -> None:
        credit = Decimal(cents).scaleb(-2)
        assert alpaca_limit_price(credit).startswith("-")
        assert not alpaca_limit_price(-credit).startswith("-")


class TestTheScale:
    """Total structure cash → per-share price for one strategy unit."""

    def test_a_one_contract_credit_spread(self) -> None:
        """0.60/share × 100 = 60.00 total cash; the wire wants 0.60 back."""
        order = gated()
        assert order.limit_price == Decimal(60)
        assert net_premium_per_unit(order) == Decimal("0.60")

    def test_a_debit_keeps_its_sign_through_the_division(self) -> None:
        structure, quotes = call_debit_spread()
        order = gated(structure=structure, quotes=quotes, intent=Intent.OPEN)
        assert order.limit_price == Decimal("-236.00")
        assert net_premium_per_unit(order) == Decimal("-2.36")

    def test_the_structures_own_quantity_divides_out(self) -> None:
        """A 3-contract structure has 3× the cash and the *same* unit price.

        This is the one that catches a double-count: `qty` on the wire already
        carries the multiplier, so leaving it in the price too would ask for
        three times the credit.
        """
        # A three-contract structure risks 1,320, which is over 1% of the
        # default 100k book — the Gate refuses it, correctly. The scale test is
        # about arithmetic, so it runs on an account that can carry the size.
        rich = Decimal(500_000)
        one, one_q = put_credit_spread(qty=1)
        three, three_q = put_credit_spread(qty=3)
        single = gated(structure=one, quotes=one_q, equity=rich)
        triple = gated(structure=three, quotes=three_q, equity=rich)
        assert triple.limit_price == single.limit_price * 3
        assert net_premium_per_unit(triple) == net_premium_per_unit(single)

    def test_the_orders_quantity_does_not_divide_out(self) -> None:
        """`GatedOrder.quantity` is the wire's `qty`. It must not touch price."""
        rich = Decimal(500_000)
        one = gated(quantity=1, equity=rich)
        four = gated(quantity=4, equity=rich)
        assert net_premium_per_unit(four) == net_premium_per_unit(one)

    def test_the_result_is_quantised_to_a_cent(self) -> None:
        """Sub-penny limits are rejected by the API."""
        assert net_premium_per_unit(gated()).as_tuple().exponent == -2
