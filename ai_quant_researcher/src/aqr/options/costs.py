"""What an option order costs, beyond the spread — specs/10-options-research.md D7.

A separate model from [`backtest/costs.py`](../backtest/costs.py), and the
reason is not tidiness. That one charges a *modelled* spread in basis points,
because an equity backtest fills at a bar's open and has no book to cross. This
engine crosses a **real** book: the bid and the ask are in the cache and D2
charges both in full. Routing option fills through the equity model would
therefore charge the spread twice — once from the quotes and once from
``spread_bps`` — and the second charge would be invisible.

So what is left here is only what the quotes do not already say: commission,
regulatory fees, and the fee for a leg that finishes in the money.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

__all__ = ["ALPACA_OPTIONS", "IBKR_OPTIONS", "PRESETS", "ZERO_OPTION_COST", "OptionCostModel"]


@dataclass(frozen=True, slots=True)
class OptionCostModel:
    """Per-contract charges. The spread is not here; it is in the fill (D2)."""

    name: str = "IBKR_OPTIONS"
    commission_per_contract: float = 0.65
    regulatory_per_contract: float = 0.0
    """OCC, ORF and exchange fees, rolled into one per-contract number."""
    exercise_per_contract: float = 0.0
    """Charged per *leg* that finishes in the money, on the settlement date."""

    def __post_init__(self) -> None:
        for field_name in ("commission_per_contract", "regulatory_per_contract",
                           "exercise_per_contract"):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} must be >= 0")

    def entry_fee(self, *, legs: int, contracts: int) -> float:
        """Charged per leg per contract: a four-legged condor pays four times."""
        per = self.commission_per_contract + self.regulatory_per_contract
        return per * legs * contracts

    def settlement_fee(self, *, itm_legs: int, contracts: int) -> float:
        return self.exercise_per_contract * itm_legs * contracts

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


IBKR_OPTIONS = OptionCostModel()
"""$0.65 a contract, the schedule this project's equity default already assumes."""

ALPACA_OPTIONS = OptionCostModel(
    name="ALPACA_OPTIONS",
    commission_per_contract=0.0,
    regulatory_per_contract=0.05,
    exercise_per_contract=0.0,
)
"""No commission, but the pass-through fees are real and are not zero.

Modelled at 5c a contract, which is the rough sum of the OCC clearing fee, the
ORF and the exchange charges on a retail-sized order. It is an estimate and is
labelled as one; what it must not be is zero, because a free-to-trade
assumption is exactly the kind that survives research and fails in production.
"""

ZERO_OPTION_COST = OptionCostModel(
    name="ZERO_OPTION_COST",
    commission_per_contract=0.0,
    regulatory_per_contract=0.0,
    exercise_per_contract=0.0,
)
"""For the frictionless comparison the evaluator's cost-retention gate needs.

Not a trading assumption. A strategy that only works here dies in research.
"""

PRESETS: dict[str, OptionCostModel] = {
    model.name: model for model in (IBKR_OPTIONS, ALPACA_OPTIONS, ZERO_OPTION_COST)
}
