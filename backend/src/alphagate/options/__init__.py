"""Options domain — specs/02.

PURE. Stdlib plus `alphagate.core`. No provider types, no LLM, no I/O, no clock.

The load-bearing idea is in D3: the unit the agent proposes and the Gate judges
is a *structure*, never a bare leg. There is no naked-short structure kind, so a
naked short is not something the Gate has to refuse — it is something the type
system cannot express.
"""

from alphagate.options.contract import (
    STRIKE_PLACES,
    OptionContract,
    Right,
    Side,
    format_occ,
    parse_occ,
)
from alphagate.options.quote import MAX_QUOTE_AGE, Greeks, OptionQuote
from alphagate.options.risk import StructureRisk, compute_risk
from alphagate.options.structure import Cover, Leg, OptionStructure, StructureKind

__all__ = [
    "MAX_QUOTE_AGE",
    "STRIKE_PLACES",
    "Cover",
    "Greeks",
    "Leg",
    "OptionContract",
    "OptionQuote",
    "OptionStructure",
    "Right",
    "Side",
    "StructureKind",
    "StructureRisk",
    "compute_risk",
    "format_occ",
    "parse_occ",
]
