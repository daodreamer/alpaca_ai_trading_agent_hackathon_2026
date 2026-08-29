"""Market structure — specs/06-structure.md, ADR 0008.

Swings, HH/HL/LH/LL labels and breaks of structure. The engine identifies
structure; it does not judge trend and it does not notify (CLAUDE.md §5).

Two properties everything here is built around:

* **nothing is known before it could be** — a pivot needs `right_bars` after it,
  so every statement carries both the bar it is about (`bar_start_utc`) and the
  bar that established it (`detected_at_utc`);
* **nothing is taken back** — confirmed swings and break events are append-only,
  so an alert raised on structure can never refer to a fact that later evaporates.
"""

from alphagate.core.structure.engine import BreakPolicy, StructureEngine
from alphagate.core.structure.model import (
    BreakEvent,
    BreakKind,
    EqualPricePolicy,
    StructureLabel,
    StructureUpdate,
    SwingKind,
    SwingPoint,
    SwingStatus,
    structural_bias,
)

__all__ = [
    "BreakEvent",
    "BreakKind",
    "BreakPolicy",
    "EqualPricePolicy",
    "StructureEngine",
    "StructureLabel",
    "StructureUpdate",
    "SwingKind",
    "SwingPoint",
    "SwingStatus",
    "structural_bias",
]
