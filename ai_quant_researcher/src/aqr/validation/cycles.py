"""Independent, non-overlapping cycles among a trade list — specs/10-options-research.md D8.

A structure held to expiry (specs/10 D1) makes "trades" and "independent
evidence" two different counts. Thirty overlapping put spreads, each opened
five sessions after the last while three earlier ones are still open, are not
thirty independent bets on the premium: they are correlated draws on the same
handful of realised outcomes, because whatever moved SPY between two
overlapping entries moved all of them together. ``MIN_TRADES`` in the equity
evaluator is close to meaningless applied to that -- specs/10 measured a 28 DTE
program at 753 sessions over 5.55 years and only **71** of the trades it could
support are actually independent of each other.

The count that matters is the number of non-overlapping cycles: enter, hold to
expiry, and only then does the next counted entry start its own clock. That is
a greedy walk in entry order -- take a trade, then skip every later trade that
opened before the one just taken has closed -- and it is deliberately the
*only* place that walk is written, because the options evaluator (D8's
independent-cycle gate) and every report that prints a cycle count next to a
trade count must agree on what a cycle is. Two independent implementations of
"non-overlapping" are two chances for one to quietly count differently after
an edit, and the disagreement would not announce itself -- it would just make
one report's gate and another report's table tell different stories about the
same run.

Works on the generic :class:`~aqr.backtest.engine.Trade` view rather than on
:class:`~aqr.options.engine.OptionTrade`, because every option structure is
held to expiry (D1): a trade's ``exit_time`` *is* its expiry, so the greedy
walk needs nothing option-specific and the equity side could use the same
function on any held-to-a-fixed-date trade list without this module knowing
options exist.
"""

from __future__ import annotations

from collections.abc import Sequence

from aqr.backtest.engine import Trade

__all__ = ["independent_cycles"]


def independent_cycles(trades: Sequence[Trade]) -> int:
    """Greedily count trades whose holding periods do not overlap.

    Sorted by entry time first, because the walk is order-dependent and a
    caller's own list order (insertion order, symbol order, whatever a prior
    step happened to produce) must not change the count. Once a trade is
    counted, every later trade that entered before this one's ``exit_time``
    is correlated with it and is not counted as separate evidence -- not
    discarded from the underlying trade list, just not double counted here.
    Two structures opened on the same session are the least independent pair
    possible, not the most: the first one counted sets ``free_from`` to its
    own expiry, and the second -- entering no later than the first, so
    strictly before that expiry -- is always skipped by the same check. A
    third entry only starts counting again once something has actually
    closed.
    """
    ordered = sorted(trades, key=lambda t: t.entry_time)
    cycles = 0
    free_from: int | None = None
    for trade in ordered:
        if free_from is not None and trade.entry_time < free_from:
            continue
        cycles += 1
        free_from = trade.exit_time
    return cycles
