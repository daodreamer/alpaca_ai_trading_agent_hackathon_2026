"""The read surface — the dashboard over the journal.

**Read-only, structurally.** This package imports `alphagate.journal` and
nothing else from the system: no `McpSession`, no `submit`, no market data
client. There is no code path from a browser to an order, and on demo day that
property is worth more than any feature it might otherwise have. It is enforced
by `tests/test_boundaries.py`, not by intention.

* `read` — journal lines into what a page renders. Pure, and where every
  decision about what a judge sees actually lives.
* `app` — routing and HTML. Deliberately dull.
"""

from alphagate.interface.read import CheckView, CycleView, DayView, available_days, day_view

__all__ = ["CheckView", "CycleView", "DayView", "available_days", "day_view"]
