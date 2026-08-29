"""Structure → tool call — specs/04 D3.

Every field below was read off the live `place_option_order` schema
(alpaca-mcp-server 2.3.0), not inferred from documentation.

| Field | Value |
| --- | --- |
| `qty` | the **strategy multiplier**, not a contract count |
| `legs` | one dict per leg, max 4; `symbol` + `ratio_qty`, plus side and intent |
| `order_class` | `"mleg"`, sent explicitly — an inferred value is one nobody reviewed |
| `type` | `"limit"`, always: on options the spread *is* the risk |
| `limit_price` | net debit/credit, sign per D2 |
| `time_in_force` | `"day"` only. Options support nothing else. |
| `position_intent` | per leg; optional in the API, mandatory here |
| `symbol`, `side` | single-leg only; omitted for `mleg` |

Two branches, because there are two order shapes and conflating them is a live
mispricing:

**Multi-leg** (vertical, condor) sends a *net* price that carries the credit /
debit sign. **Single-leg** (covered call, cash-secured put) sends the option's
own price, which is always positive — the direction lives in `side`. Sending a
signed net price on a single-leg order submits a negative limit, and a short put
priced at `-1.50` is not a short put priced at `1.50`.

The four-leg ceiling is enforced in the *domain*: no `StructureKind` in specs/02
D3 exceeds four legs, so an unsendable structure is unconstructible. It is
asserted here anyway, as a tripwire for a future kind added without reading this
file.
"""

from __future__ import annotations

from typing import Final

from alphagate.execution.errors import UnsubmittableOrder
from alphagate.execution.pricing import alpaca_limit_price, net_premium_per_unit
from alphagate.execution.session import ToolArgument
from alphagate.options import Leg, Side, format_occ
from alphagate.risk import GatedOrder, Intent

__all__ = [
    "MAX_LEGS",
    "PLACE_ORDER_TOOL",
    "TIME_IN_FORCE",
    "position_intent",
    "to_tool_arguments",
]

PLACE_ORDER_TOOL: Final = "place_option_order"
MAX_LEGS: Final = 4
TIME_IN_FORCE: Final = "day"
"""Options support nothing else. Not a default — the only legal value."""


def position_intent(intent: Intent, side: Side) -> str:
    """`buy_to_open` and friends, per leg.

    Optional in the API and mandatory here, because assignment behaviour and
    position netting both depend on it. Letting the broker guess whether this
    sell is opening a short or closing a long is letting the broker guess what
    the position is.

    `ROLL` maps to opening intents. A roll spans two expiries, which specs/02 D3
    makes unconstructible in one structure, so a roll is always *two* orders: a
    close of the old structure and an open of the new one. This is the open half;
    the close half arrives as its own proposal with `Intent.CLOSE`.
    """
    verb = "buy" if side is Side.BUY else "sell"
    match intent:
        case Intent.OPEN | Intent.ROLL:
            return f"{verb}_to_open"
        case Intent.CLOSE:
            return f"{verb}_to_close"
    raise UnsubmittableOrder(f"no position_intent mapping for {intent}")  # pragma: no cover


def to_tool_arguments(order: GatedOrder) -> dict[str, ToolArgument]:
    """Render a gated order as `place_option_order` arguments.

    Pure and total: same order in, same dict out, no clock and no I/O. Every
    value is a string, as the schema requires.
    """
    legs = order.structure.legs
    if len(legs) > MAX_LEGS:
        raise UnsubmittableOrder(
            f"{order.structure.kind.name} has {len(legs)} legs; Alpaca accepts at "
            f"most {MAX_LEGS}. specs/02 D3 should have made this unconstructible — "
            "a new StructureKind was added without reading specs/04 D3."
        )
    if order.quantity <= 0:  # pragma: no cover - guaranteed by specs/03 D2
        raise UnsubmittableOrder(f"order quantity must be positive, got {order.quantity}")

    arguments: dict[str, ToolArgument] = {
        "qty": str(order.quantity),
        "type": "limit",
        "time_in_force": TIME_IN_FORCE,
    }

    if len(legs) == 1:
        arguments.update(_single_leg(order, legs[0]))
    else:
        arguments.update(_multi_leg(order, legs))
    return arguments


def _single_leg(order: GatedOrder, leg: Leg) -> dict[str, ToolArgument]:
    """A covered call or a cash-secured put.

    The price is the option's own, always positive: `abs` of the per-unit net
    premium. The sign that says credit-or-debit is not dropped, it is expressed
    by `side` — selling a put at 1.50 *is* the credit.
    """
    per_unit = net_premium_per_unit(order)
    return {
        "symbol": format_occ(leg.contract),
        "side": leg.side.value,
        "position_intent": position_intent(order.intent, leg.side),
        "limit_price": f"{abs(per_unit):.2f}",
    }


def _multi_leg(order: GatedOrder, legs: tuple[Leg, ...]) -> dict[str, ToolArgument]:
    """A vertical or a condor: one net price, signed per D2.

    `ratio_qty` is the leg's own quantity, and `qty` (set by the caller) is the
    strategy multiplier that scales it. `qty="10"` with `ratio_qty="2"` is twenty
    contracts on that leg. The two multiply — which is why `net_premium_per_unit`
    divides out the structure's quantity but not the order's.
    """
    return {
        "order_class": "mleg",
        "limit_price": alpaca_limit_price(net_premium_per_unit(order)),
        "legs": [
            {
                "symbol": format_occ(leg.contract),
                "ratio_qty": str(leg.quantity),
                "side": leg.side.value,
                "position_intent": position_intent(order.intent, leg.side),
            }
            for leg in legs
        ],
    }
