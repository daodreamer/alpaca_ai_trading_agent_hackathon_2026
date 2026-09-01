"""The equity domain — specs/09.

PURE. Stdlib plus `alphagate.core` and its own siblings. No LLM, no I/O, no
clock read, no network. Enforced by `tests/test_boundaries.py`.

Three sentences carry the layer:

1. A target book is loaded only if it names the **pinned** strategy, its seal is
   spent, and the sealed window did not refute it — so "we execute only what the
   researcher validated" is a checked property, not a claim.
2. Weights become shares here and nowhere else, because this is the first place
   equity, prices and holdings meet.
3. `plan_rebalance` is deterministic and total: the same inputs give the same
   sequence, and every symbol it considered comes back either as an order or as
   a stated reason there is none.

Nothing in this package can place an order. The plan it produces is a sequence
of `OrderIntent`s, which `execution` will not accept — they become submittable
only by passing `risk.equity_gate.evaluate`.
"""

from alphagate.equity.book import (
    BOOK_SCHEMA_VERSION,
    SealedRun,
    TargetBook,
    UnusableBook,
    load_target_book,
)
from alphagate.equity.plan import (
    EquitySide,
    Holding,
    Mark,
    OrderIntent,
    RebalancePlan,
    Skipped,
    SkipReason,
    plan_rebalance,
)
from alphagate.equity.policy import DEFAULT_EQUITY_POLICY, EQUITY_SLEEVE_ALLOCATION, EquityPolicy

__all__ = [
    "BOOK_SCHEMA_VERSION",
    "DEFAULT_EQUITY_POLICY",
    "EQUITY_SLEEVE_ALLOCATION",
    "EquityPolicy",
    "EquitySide",
    "Holding",
    "Mark",
    "OrderIntent",
    "RebalancePlan",
    "SealedRun",
    "SkipReason",
    "Skipped",
    "TargetBook",
    "UnusableBook",
    "load_target_book",
    "plan_rebalance",
]
