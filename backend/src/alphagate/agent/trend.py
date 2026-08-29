"""Wiring the trend engine — specs/05 D2, adr/0001.

This is the reuse dividend made concrete. The median entrant pastes OHLC into a
prompt and asks the model what it thinks; we fold the same bars through a
seven-rung state machine that lags its evidence on purpose, and hand the model
the state, its strength, its confidence, and the reason codes that produced it.

Three things this module is careful about, all inherited from the core's own
discipline rather than invented here.

**A fresh engine per read.** `TrendEngine` is a `BarConsumer`: it accumulates,
it is bound to one `(symbol, timeframe, feed, adjustment)` series, and it
refuses a `StructureEngine` that has already seen bars. Reusing an instance
across cycles would make today's trend depend on which bars yesterday's process
happened to have consumed — a hidden input, and the end of a deterministic
rebuild.

**A trend below the confidence floor is `None`, not `UNKNOWN` dressed up.** The
engine reports how much of the evidence it could actually read. A state produced
from a third of the evidence is not a weaker opinion, it is an opinion about
something else, and `MarketRead.trend` renders `None` as `unmeasured` precisely
so the model does not average it into a view.

**Multi-timeframe is not free.** `read_trend` does one timeframe. Confluence
(specs/05 D2's `confluence` field) needs the same fold on several and then
`core.confluence.build_confluence` over the results; that is wired here only for
the timeframes the caller supplies bars for, and reports `None` for the rest
rather than inferring a daily trend from hourly bars.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from alphagate.core.bar import Bar
from alphagate.core.errors import InvariantViolation
from alphagate.core.trend import TrendPhase, TrendState
from alphagate.core.trend_engine import TrendConfig, TrendEngine, TrendUpdate

__all__ = ["MIN_CONFIDENCE", "TrendRead", "read_trend"]

MIN_CONFIDENCE: Final = 0.5
"""Share of the requested evidence that must be readable for a state to count.

Half. The engine already refuses to move the state on thin evidence; this is the
caller's own floor on top of that, and it exists because a trend read from a
handful of warm indicators is the kind of number that looks like an answer in a
prompt. Below it, `read_trend` reports `None`."""


@dataclass(frozen=True, slots=True)
class TrendRead:
    """A trend state, or an honest account of why there isn't one."""

    state: TrendState | None
    bars_consumed: int
    reason: str
    """Why `state` is `None`, when it is. Recorded, so "not enough bars" and
    "evidence too thin to trust" are distinguishable in the journal."""

    @property
    def phase(self) -> TrendPhase | None:
        return None if self.state is None else self.state.state

    @property
    def is_measured(self) -> bool:
        return self.state is not None

    def describe(self) -> str:
        """One line for a human. The model gets the state object itself."""
        if self.state is None:
            return f"unmeasured ({self.reason})"
        return (
            f"{self.state.state.value} strength={self.state.strength:.0f} "
            f"confidence={self.state.confidence:.2f}"
        )


def read_trend(
    bars: Sequence[Bar],
    *,
    config: TrendConfig | None = None,
    minimum_confidence: float = MIN_CONFIDENCE,
) -> TrendRead:
    """Fold a bar series into a trend state. Pure over its inputs.

    A fresh engine every call — see the module docstring. The cost is one pass
    over ~125 daily bars per cycle, which is nothing next to what a hidden
    dependency on process history would cost the first time a rebuild disagreed
    with the journal.
    """
    if not bars:
        return TrendRead(None, 0, "no bars")

    engine = TrendEngine(config=config)
    update: TrendUpdate | None = None
    consumed = 0
    for bar in bars:
        if not bar.is_final:
            # A partial bar must not seed a recursive calculation unless the
            # caller asked for it, and here nobody did.
            continue
        try:
            update = engine.update(bar)
        except InvariantViolation as exc:
            # A series break — a gap, a duplicate, a feed change mid-history.
            # Reporting it is better than folding across it and calling the
            # result a trend.
            return TrendRead(None, consumed, f"series break at bar {consumed}: {exc}")
        consumed += 1

    if update is None or update.state is None:
        return TrendRead(None, consumed, f"engine still warming after {consumed} bars")

    state = update.state
    if state.confidence < minimum_confidence:
        return TrendRead(
            None,
            consumed,
            f"confidence {state.confidence:.2f} below the {minimum_confidence:.2f} floor",
        )
    return TrendRead(state, consumed, "measured")
