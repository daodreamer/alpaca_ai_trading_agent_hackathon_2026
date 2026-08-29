"""The Risk Gate — specs/03.

PURE. Stdlib plus `alphagate.core` and `alphagate.options`. No LLM, no I/O, no
clock read, no network. Enforced by `tests/test_boundaries.py`, not by goodwill.

Three sentences carry the whole layer:

1. `evaluate` is one pure function and every input, including the time, is an
   argument.
2. Every check runs, always. The tape is the product, not just the answer.
3. A `GatedOrder` can only be minted inside `alphagate.risk.gate`, and
   `execution` accepts nothing else — so "every order passed the Gate" is a
   property of the type system rather than a promise in a README.
"""

from alphagate.risk.checks import CHECKS, Check, Context, run_all
from alphagate.risk.gate import evaluate
from alphagate.risk.limits import DEFAULT_LIMITS, RiskLimits
from alphagate.risk.portfolio import OpenPosition, PortfolioSnapshot
from alphagate.risk.proposal import Intent, TradeProposal
from alphagate.risk.verdict import (
    Approved,
    CheckResult,
    GatedOrder,
    Observation,
    Verdict,
    Vetoed,
    VetoReason,
)

__all__ = [
    "CHECKS",
    "DEFAULT_LIMITS",
    "Approved",
    "Check",
    "CheckResult",
    "Context",
    "GatedOrder",
    "Intent",
    "Observation",
    "OpenPosition",
    "PortfolioSnapshot",
    "RiskLimits",
    "TradeProposal",
    "Verdict",
    "VetoReason",
    "Vetoed",
    "evaluate",
    "run_all",
]
