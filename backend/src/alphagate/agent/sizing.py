"""Sizing — specs/05 D4. Pure, and the model gets no vote.

```
quantity = floor(limits.max_trade_loss / structure.max_loss)
```

That is the whole rule. It is here, in eight lines of arithmetic, rather than in
a prompt, because sizing is where an LLM's lack of calibration does the most
damage per unit of plausibility. A model that is wrong about direction loses one
trade's risk. A model that is wrong about size loses several.

If the quantity is zero the candidate is **dropped before the model ever sees
it** (specs/05 D1 step 3). Showing a model something it cannot legally trade and
relying on it not to pick that one is not a control.
"""

from __future__ import annotations

from decimal import ROUND_FLOOR, Decimal

from alphagate.options import StructureRisk
from alphagate.risk import RiskLimits

__all__ = ["size_for"]


def size_for(risk: StructureRisk, limits: RiskLimits, equity: Decimal) -> int:
    """Contracts this structure may be traded in, given the per-trade budget.

    Returns 0 when even one unit is too large. Zero is a real answer — it is how
    an account too small for a structure declines it without anyone deciding to.

    Floor, never round: rounding up would put the order over the limit and hand
    the Gate a veto that sizing could have avoided. The Gate would catch it
    (`per_trade_loss`, specs/03 D4), which is exactly why sizing must not rely on
    that — a control that is routinely tripped by our own code stops being read.
    """
    budget = limits.max_trade_loss(equity)
    if not budget.is_finite() or budget <= 0:  # pragma: no cover - RiskLimits guards this
        return 0
    per_unit = risk.max_loss
    if not per_unit.is_finite() or per_unit <= 0:  # pragma: no cover - specs/02 D4 guards this
        return 0
    return int((budget / per_unit).to_integral_value(rounding=ROUND_FLOOR))
