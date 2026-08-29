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

There are **two** gates, because there are two kinds of order and one type that
meant either would prove neither. The options Gate judges structures — short
strikes, DTE, defined-risk width. The equity Gate (specs/09 D5) judges share
orders against a validated target book, and mints `GatedEquityOrder`, which
`execution.submit_equity` accepts and nothing else does. Same discipline, same
frame guard, different checks; `tests/test_boundaries.py` covers both doors.
"""

from alphagate.risk.checks import CHECKS, Check, Context, run_all
from alphagate.risk.equity_checks import (
    EQUITY_CHECKS,
    UNWAIVABLE,
    EquityCheck,
    EquityContext,
    run_equity_checks,
)
from alphagate.risk.equity_gate import evaluate_equity
from alphagate.risk.equity_portfolio import EquityPortfolio
from alphagate.risk.equity_verdict import (
    ApprovedEquity,
    EquityVerdict,
    GatedEquityOrder,
    VetoedEquity,
)
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
    "EQUITY_CHECKS",
    "UNWAIVABLE",
    "Approved",
    "ApprovedEquity",
    "Check",
    "CheckResult",
    "Context",
    "EquityCheck",
    "EquityContext",
    "EquityPortfolio",
    "EquityVerdict",
    "GatedEquityOrder",
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
    "VetoedEquity",
    "evaluate",
    "evaluate_equity",
    "run_all",
    "run_equity_checks",
]
