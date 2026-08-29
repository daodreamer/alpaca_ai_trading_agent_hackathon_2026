"""04 D4 — the derived idempotency key. Test plan item 5.

The property under test is a pair, and both halves matter equally:

* the same order submitted twice produces the **same** key, so the API's
  duplicate rejection turns a retry into a no-op;
* an order that differs in any way that makes it a *different order* produces a
  **different** key, so a legitimate second trade is not silently swallowed.

A key that is too stable loses trades. A key that is too unstable doubles
positions. Only the second failure costs money, which is why the day boundary is
the coarsest window that still separates today's order from tomorrow's.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest

from alphagate.execution.idempotency import (
    MARKET_TZ,
    PREFIX,
    client_order_id,
    order_fingerprint,
    trading_day_of,
)
from alphagate.risk import Intent
from tests.execution.conftest import call_debit_spread, gated, put_credit_spread

RICH = Decimal(500_000)
DAY = date(2026, 8, 26)


class TestStability:
    def test_the_same_order_gives_the_same_key(self) -> None:
        assert client_order_id(gated()) == client_order_id(gated())

    def test_two_separately_built_but_identical_orders_agree(self) -> None:
        """The key is a function of the facts, not of object identity."""
        a, b = gated(), gated()
        assert a is not b
        assert client_order_id(a) == client_order_id(b)

    def test_leg_order_does_not_change_the_key(self) -> None:
        """`OptionStructure` normalises leg order at construction, so two
        orderings of one spread are one order and must share a key."""
        structure, quotes = put_credit_spread()
        reversed_legs = type(structure)(structure.kind, tuple(reversed(structure.legs)))
        assert client_order_id(gated(structure=structure, quotes=quotes)) == client_order_id(
            gated(structure=reversed_legs, quotes=quotes)
        )

    def test_the_price_is_not_in_the_key(self) -> None:
        """Repricing an unfilled order is a replace, not a new order. Putting the
        limit in the key would let a one-cent adjustment open a second position."""
        structure, quotes = put_credit_spread()
        cheap = gated(structure=structure, quotes=quotes)
        assert "limit" not in order_fingerprint(cheap, DAY)
        assert str(cheap.limit_price) not in order_fingerprint(cheap, DAY)


class TestSeparation:
    def test_a_different_quantity_is_a_different_order(self) -> None:
        assert client_order_id(gated(quantity=1, equity=RICH)) != client_order_id(
            gated(quantity=2, equity=RICH)
        )

    def test_a_different_intent_is_a_different_order(self) -> None:
        assert client_order_id(gated(intent=Intent.OPEN)) != client_order_id(
            gated(intent=Intent.CLOSE)
        )

    def test_a_different_day_is_a_different_order(self) -> None:
        """The same proposal, legitimately re-proposed tomorrow, must go through."""
        order = gated()
        assert client_order_id(order, DAY) != client_order_id(order, DAY + timedelta(days=1))

    def test_a_different_structure_is_a_different_order(self) -> None:
        credit, credit_quotes = put_credit_spread()
        debit, debit_quotes = call_debit_spread()
        assert client_order_id(gated(structure=credit, quotes=credit_quotes)) != client_order_id(
            gated(structure=debit, quotes=debit_quotes)
        )

    def test_a_different_proposal_is_a_different_order(self) -> None:
        assert client_order_id(gated(proposal_id="a")) != client_order_id(gated(proposal_id="b"))


class TestTradingDay:
    """The day is Eastern, not UTC. specs/04 D4."""

    def test_an_afternoon_utc_timestamp_is_the_same_eastern_day(self) -> None:
        """20:30 UTC is 16:30 ET — still 26 August, not the 27th."""
        assert trading_day_of(datetime(2026, 8, 26, 20, 30, tzinfo=UTC)) == DAY

    def test_a_late_evening_utc_timestamp_is_still_the_same_session(self) -> None:
        """23:00 UTC on the 26th is 19:00 ET on the 26th. Hashing the UTC date
        would split one session's orders across two keys every afternoon."""
        assert trading_day_of(datetime(2026, 8, 26, 23, 0, tzinfo=UTC)) == DAY

    def test_after_midnight_eastern_is_the_next_day(self) -> None:
        assert trading_day_of(datetime(2026, 8, 27, 5, 0, tzinfo=UTC)) == date(2026, 8, 27)

    def test_the_default_comes_from_the_approval_timestamp(self) -> None:
        order = gated()
        assert client_order_id(order) == client_order_id(
            order, trading_day_of(order.approved_at)
        )

    def test_the_market_timezone_is_eastern(self) -> None:
        assert str(MARKET_TZ) == "America/New_York"


class TestShape:
    def test_it_is_prefixed_and_short_enough_for_the_api(self) -> None:
        """Alpaca caps `client_order_id` at 128 characters."""
        key = client_order_id(gated())
        assert key.startswith(f"{PREFIX}-")
        assert len(key) <= 128

    def test_it_is_url_and_log_safe(self) -> None:
        key = client_order_id(gated())
        assert key.replace("-", "").isalnum()

    def test_the_fingerprint_is_readable(self) -> None:
        """Returned rather than hidden so a mismatch can be diffed as text."""
        fingerprint = order_fingerprint(gated(), DAY)
        assert "SPY260904P00752000:sell:1" in fingerprint
        assert "vertical_credit" in fingerprint
        assert "2026-08-26" in fingerprint

    @pytest.mark.parametrize("quantity", [1, 2, 3])
    def test_keys_do_not_collide_across_a_realistic_day(self, quantity: int) -> None:
        keys = {
            client_order_id(gated(quantity=quantity, proposal_id=f"p-{i}", equity=RICH))
            for i in range(50)
        }
        assert len(keys) == 50
