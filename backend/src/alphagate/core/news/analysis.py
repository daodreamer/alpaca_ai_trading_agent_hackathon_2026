"""What the pipeline asks of whatever explains a story, and what it gets back.

The port lives here, and the thing that speaks HTTP to a particular vendor does
not. That split is what lets the application layer depend on "something can
explain a release" without depending on DeepSeek, and it is why swapping the
model later is an adapter change rather than a rewrite.

`NewsAnalysis` records the model id, the prompt version and a hash of the raw
response alongside the reading. A language model is not deterministic, so it is
treated like any other external source: the result is data that was received,
not a computation that can be repeated. A replay reads the stored result, which
is how `CLAUDE.md` §11 is satisfied without pretending otherwise.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

__all__ = ["Confidence", "Direction", "NewsAnalysis", "NewsAnalyst"]


class Direction(StrEnum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    UNCLEAR = "UNCLEAR"


class Confidence(StrEnum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


@dataclass(frozen=True, slots=True)
class NewsAnalysis:
    """What a model made of one release, and what produced it."""

    primary_ticker: str
    company: str
    direction: Direction
    why: str
    confidence: Confidence
    also_affected: tuple[str, ...]

    model: str
    prompt_version: str
    response_hash: str
    """SHA-256 of the raw response body. The audit trail for a surprising alert."""


class NewsAnalyst(Protocol):
    """Explains a screened item, or returns `None` and lets the alert go plain.

    Returning `None` rather than raising is the contract that matters: an
    explanation is an enrichment, and losing it must never cost the alert.
    """

    def analyse(
        self,
        *,
        headline: str,
        teaser: str,
        candidates: Sequence[str],
        categories: Sequence[str],
    ) -> NewsAnalysis | None: ...
