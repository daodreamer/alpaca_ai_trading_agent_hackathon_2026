"""04 D3 — structure to tool call. Test plan items 2 and 3.

One golden fixture per `StructureKind`, written out in full rather than
assembled by a helper. A golden test whose expected value is computed shares the
bug it is meant to catch; these are typed out.

The `qty` / `ratio_qty` distinction gets its own section because it is the field
most likely to be misread as a contract count. It is not: it is a strategy
multiplier, and the two multiply.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from alphagate.execution.errors import UnsubmittableOrder
from alphagate.execution.mapping import MAX_LEGS, position_intent, to_tool_arguments
from alphagate.options import Side, StructureKind
from alphagate.risk import Intent
from tests.execution.conftest import (
    STRUCTURES,
    call_debit_spread,
    cash_secured_put,
    covered_call,
    gated,
    iron_condor,
    put_credit_spread,
)

RICH = Decimal(500_000)


class TestGoldenArguments:
    """One per kind. specs/04 test plan item 2."""

    def test_vertical_credit(self) -> None:
        structure, quotes = put_credit_spread()
        assert to_tool_arguments(gated(structure=structure, quotes=quotes)) == {
            "qty": "1",
            "type": "limit",
            "time_in_force": "day",
            "order_class": "mleg",
            "limit_price": "-0.60",  # credit → negative on the wire (D2)
            "legs": [
                {
                    "symbol": "SPY260904P00747000",
                    "ratio_qty": "1",
                    "side": "buy",
                    "position_intent": "buy_to_open",
                },
                {
                    "symbol": "SPY260904P00752000",
                    "ratio_qty": "1",
                    "side": "sell",
                    "position_intent": "sell_to_open",
                },
            ],
        }

    def test_vertical_debit(self) -> None:
        structure, quotes = call_debit_spread()
        arguments = to_tool_arguments(gated(structure=structure, quotes=quotes))
        assert arguments["limit_price"] == "2.36"  # debit → positive
        assert arguments["order_class"] == "mleg"
        assert [leg["symbol"] for leg in arguments["legs"]] == [  # type: ignore[union-attr]
            "SPY260904C00770000",
            "SPY260904C00775000",
        ]

    def test_iron_condor_sends_four_legs(self) -> None:
        structure, quotes = iron_condor()
        arguments = to_tool_arguments(gated(structure=structure, quotes=quotes, equity=RICH))
        legs = arguments["legs"]
        assert isinstance(legs, list)
        assert len(legs) == MAX_LEGS
        assert arguments["limit_price"] == "-0.80"
        assert [leg["symbol"][-15:] for leg in legs] == [
            "260904C00790000",
            "260904C00795000",
            "260904P00740000",
            "260904P00745000",
        ]
        # Leg order is the domain's normalisation (specs/02 D3): calls before
        # puts, then strike ascending. Not the order they were written in.
        assert [leg["side"] for leg in legs] == ["sell", "buy", "buy", "sell"]

    def test_covered_call_is_single_leg_and_carries_no_order_class(self) -> None:
        """`symbol`/`side` are single-leg only; `legs` and `order_class` are absent."""
        structure, quotes = covered_call()
        arguments = to_tool_arguments(gated(structure=structure, quotes=quotes, equity=RICH))
        assert arguments == {
            "qty": "1",
            "type": "limit",
            "time_in_force": "day",
            "symbol": "SOFI260904C00022000",
            "side": "sell",
            "position_intent": "sell_to_open",
            "limit_price": "0.24",
        }

    def test_cash_secured_put_prices_positive_despite_being_a_credit(self) -> None:
        """The single-leg branch. This is the one that would go out negative.

        A short put at `-0.98` is not a short put at `0.98`; on a single-leg
        order the direction lives in `side`, and the price is the option's own.
        """
        structure, quotes = cash_secured_put()
        arguments = to_tool_arguments(gated(structure=structure, quotes=quotes, equity=RICH))
        assert arguments["limit_price"] == "0.24"
        assert not str(arguments["limit_price"]).startswith("-")
        assert arguments["side"] == "sell"
        assert "legs" not in arguments

    @pytest.mark.parametrize("kind", list(StructureKind))
    def test_every_kind_maps(self, kind: StructureKind) -> None:
        """The parametrisation is over the enum, so adding a kind fails here."""
        structure, quotes = STRUCTURES[kind]()
        arguments = to_tool_arguments(gated(structure=structure, quotes=quotes, equity=RICH))
        assert arguments["type"] == "limit"
        assert arguments["time_in_force"] == "day"
        assert "limit_price" in arguments


class TestQuantitySemantics:
    """specs/04 test plan item 3. `qty` is a multiplier, not a contract count."""

    def test_qty_is_the_order_quantity(self) -> None:
        arguments = to_tool_arguments(gated(quantity=2, equity=RICH))
        assert arguments["qty"] == "2"

    def test_ratio_qty_is_the_leg_quantity(self) -> None:
        structure, quotes = put_credit_spread(qty=2)
        arguments = to_tool_arguments(gated(structure=structure, quotes=quotes, equity=RICH))
        legs = arguments["legs"]
        assert isinstance(legs, list)
        assert {leg["ratio_qty"] for leg in legs} == {"2"}

    def test_they_multiply(self) -> None:
        """`qty="10"` with `ratio_qty="2"` is twenty contracts on that leg —
        the schema's own worked example."""
        structure, quotes = put_credit_spread(qty=2)
        arguments = to_tool_arguments(
            gated(structure=structure, quotes=quotes, quantity=10, equity=Decimal(50_000_000))
        )
        legs = arguments["legs"]
        assert isinstance(legs, list)
        assert arguments["qty"] == "10"
        assert legs[0]["ratio_qty"] == "2"
        # 10 × 2 = 20 contracts a leg, and the price stays per-share per-unit.
        assert arguments["limit_price"] == "-0.60"


class TestPositionIntent:
    """Optional in the API, mandatory here — assignment behaviour depends on it."""

    @pytest.mark.parametrize(
        ("intent", "side", "expected"),
        [
            (Intent.OPEN, Side.BUY, "buy_to_open"),
            (Intent.OPEN, Side.SELL, "sell_to_open"),
            (Intent.CLOSE, Side.BUY, "buy_to_close"),
            (Intent.CLOSE, Side.SELL, "sell_to_close"),
            (Intent.ROLL, Side.BUY, "buy_to_open"),
            (Intent.ROLL, Side.SELL, "sell_to_open"),
        ],
    )
    def test_every_combination(self, intent: Intent, side: Side, expected: str) -> None:
        assert position_intent(intent, side) == expected

    def test_a_close_reverses_the_intent_on_every_leg(self) -> None:
        arguments = to_tool_arguments(gated(intent=Intent.CLOSE))
        legs = arguments["legs"]
        assert isinstance(legs, list)
        assert {leg["position_intent"] for leg in legs} == {"buy_to_close", "sell_to_close"}

    def test_a_roll_opens(self) -> None:
        """A roll spans two expiries, which specs/02 D3 makes unconstructible in
        one structure — so a roll is two orders, and this is the opening half."""
        arguments = to_tool_arguments(gated(intent=Intent.ROLL))
        legs = arguments["legs"]
        assert isinstance(legs, list)
        assert all(leg["position_intent"].endswith("_to_open") for leg in legs)


class TestClosingReversesTheOrder:
    """specs/04 D3, the half that had never been sent.

    A close is journalled as the structure that is *held* — `run_exit_cycle`
    passes `held.position.structure` with `Intent.CLOSE`, because the position
    is the thing being closed and inventing a mirrored `OptionStructure` would
    put a shape in the journal that was never held. Turning that into an order
    is this module's job, and the order is the mirror: every leg goes the other
    way, and a credit structure costs a debit to buy back.

    Getting only half of it right is worse than getting none: an order with the
    held sides and `_to_close` intents is a request to open a second copy of the
    position wearing a label that says it closes one. Alpaca happens to check
    that pair and refuse it -- `422 position intent mismatch, inferred:
    buy_to_open, specified: buy_to_close`, which is what this class was written
    from, 39 of them in one session. A broker that only recorded the label would
    have opened 39 more spreads instead.
    """

    def test_a_put_credit_spread_closes_as_the_mirror_order(self) -> None:
        """The golden one, typed out. Compare it with `test_vertical_credit`
        above: same structure, every side inverted, and the credit that opened
        it is the debit that closes it."""
        structure, quotes = put_credit_spread()
        assert to_tool_arguments(
            gated(structure=structure, quotes=quotes, intent=Intent.CLOSE)
        ) == {
            "qty": "1",
            "type": "limit",
            "time_in_force": "day",
            "order_class": "mleg",
            "limit_price": "0.60",  # buying the spread back costs a debit (D2)
            "legs": [
                {
                    "symbol": "SPY260904P00747000",
                    "ratio_qty": "1",
                    "side": "sell",  # held long; sold to close
                    "position_intent": "sell_to_close",
                },
                {
                    "symbol": "SPY260904P00752000",
                    "ratio_qty": "1",
                    "side": "buy",  # held short; bought back
                    "position_intent": "buy_to_close",
                },
            ],
        }

    def test_a_single_leg_close_flips_its_side_too(self) -> None:
        """A covered call is closed by buying the call back. The price stays
        positive on a single leg — the direction lives in `side`."""
        structure, quotes = covered_call()
        opening = to_tool_arguments(gated(structure=structure, quotes=quotes, equity=RICH))
        closing = to_tool_arguments(
            gated(structure=structure, quotes=quotes, intent=Intent.CLOSE, equity=RICH)
        )
        assert opening["side"] == "sell"
        assert opening["position_intent"] == "sell_to_open"
        assert closing["side"] == "buy"
        assert closing["position_intent"] == "buy_to_close"
        assert closing["limit_price"] == opening["limit_price"], (
            "a single-leg price is the option's own and never signed; only the "
            "side carries the direction"
        )

    @pytest.mark.parametrize("kind", sorted(STRUCTURES, key=lambda k: k.value))
    def test_every_kind_reverses_every_leg(self, kind: StructureKind) -> None:
        """The property, over all five kinds: the order that closes a position
        is the one that would have opened its opposite."""
        structure, quotes = STRUCTURES[kind]()
        order = gated(
            structure=structure, quotes=quotes, intent=Intent.CLOSE, equity=RICH
        )
        arguments = to_tool_arguments(order)
        legs = arguments.get("legs")
        wire = (
            [(leg["side"], leg["position_intent"]) for leg in legs]  # type: ignore[index,union-attr]
            if isinstance(legs, list)
            else [(arguments["side"], arguments["position_intent"])]
        )
        held = [leg.side for leg in structure.legs]
        assert [side for side, _ in wire] == [s.opposite.value for s in held]
        for side, intent in wire:
            assert intent == f"{side}_to_close", "the intent's verb is the wire side's"

    def test_the_sign_flip_is_the_only_difference_in_price(self) -> None:
        """Closing prices at the same magnitude, on the other side of zero. A
        close that priced at the same signed number would be an order to sell
        the spread again, at the price we would pay to buy it."""
        structure, quotes = put_credit_spread()
        opening = to_tool_arguments(gated(structure=structure, quotes=quotes))
        closing = to_tool_arguments(
            gated(structure=structure, quotes=quotes, intent=Intent.CLOSE)
        )
        assert Decimal(str(opening["limit_price"])) == -Decimal(str(closing["limit_price"]))

    def test_an_open_is_left_exactly_as_it_was(self) -> None:
        """The regression guard. Only `CLOSE` mirrors."""
        structure, quotes = put_credit_spread()
        legs = to_tool_arguments(gated(structure=structure, quotes=quotes))["legs"]
        assert isinstance(legs, list)
        assert [leg["side"] for leg in legs] == [leg.side.value for leg in structure.legs]

    def test_a_roll_opens_and_does_not_mirror(self) -> None:
        """A roll is two orders and this is the opening half, so its sides are
        the sides of the structure it is opening."""
        structure, quotes = put_credit_spread()
        legs = to_tool_arguments(
            gated(structure=structure, quotes=quotes, intent=Intent.ROLL)
        )["legs"]
        assert isinstance(legs, list)
        assert [leg["side"] for leg in legs] == [leg.side.value for leg in structure.legs]
        assert all(leg["position_intent"].endswith("_to_open") for leg in legs)


class TestLegCeiling:
    def test_no_constructible_structure_exceeds_four_legs(self) -> None:
        """The domain enforces it; the adapter only has to agree."""
        for kind in StructureKind:
            structure, _ = STRUCTURES[kind]()
            assert len(structure.legs) <= MAX_LEGS

    def test_the_tripwire_is_wired(self) -> None:
        """`UnsubmittableOrder` exists for a kind added without reading D3."""
        assert issubclass(UnsubmittableOrder, Exception)


class TestPurity:
    def test_mapping_is_deterministic(self) -> None:
        order = gated()
        assert to_tool_arguments(order) == to_tool_arguments(order)

    def test_every_value_is_a_string_or_a_list_of_strings(self) -> None:
        """The schema takes strings. An int on the wire is a 400 at best."""
        for kind in StructureKind:
            structure, quotes = STRUCTURES[kind]()
            for key, value in to_tool_arguments(
                gated(structure=structure, quotes=quotes, equity=RICH)
            ).items():
                if isinstance(value, list):
                    for leg in value:
                        assert all(isinstance(v, str) for v in leg.values()), key
                else:
                    assert isinstance(value, str), key
