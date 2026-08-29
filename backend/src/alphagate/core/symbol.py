"""The tradable instrument — specs/04-domain-model.md.

Domain identity is the ticker, not a database surrogate key. Persistence is free
to assign an integer id; the Domain has no use for it and refusing to carry it
keeps `Symbol` constructible in a test without a database.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from enum import Enum
from types import MappingProxyType
from typing import Final

from alphagate.core.errors import InvariantViolation
from alphagate.core.identifiers import Ticker

__all__ = ["AssetType", "Symbol"]

MVP_CURRENCY: Final = "USD"
"""ADR 0005 D5 — currency is carried on the Symbol, and asserted on ingest.

MVP is US equities and ETFs. A `Money` value object across the whole Domain is
deliberately not introduced for a single-currency product; multi-currency work
gets its own ADR.
"""


class AssetType(Enum):
    STOCK = "STOCK"
    ETF = "ETF"


# eq=False: identity is the ticker alone, so the generated field-wise __eq__ and
# __hash__ would be wrong. Two records for AAPL are the same symbol.
@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class Symbol:
    ticker: Ticker
    name: str
    asset_type: AssetType
    exchange: str
    currency: str = MVP_CURRENCY
    active: bool = True
    provider_mapping: Mapping[str, str] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.exchange.strip():
            raise InvariantViolation(f"{self.ticker}: exchange is required")
        if self.currency != MVP_CURRENCY:
            raise InvariantViolation(
                f"{self.ticker}: currency {self.currency!r} is not supported; "
                f"MVP is {MVP_CURRENCY} only (ADR 0005 D5)"
            )
        object.__setattr__(self, "provider_mapping", MappingProxyType(dict(self.provider_mapping)))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Symbol):
            return NotImplemented
        return self.ticker == other.ticker

    def __hash__(self) -> int:
        return hash(self.ticker)

    def provider_symbol(self, provider: str) -> str:
        """How `provider` spells this symbol, falling back to the canonical ticker.

        The fallback is the common case — most providers use the plain ticker —
        so an adapter does not need a mapping entry per symbol to work.
        """
        return self.provider_mapping.get(provider, self.ticker)
