"""News as a domain concern: what a story is, and what makes one material.

Screening and ticker attribution are pure functions over text — deterministic,
stdlib only, and testable without a wire in sight. They live here rather than
beside the RSS adapter because they are the judgements, and the adapter is only
plumbing: swapping Business Wire for a paid feed must not be able to change what
counts as news.

The `NewsAnalyst` port lives here too, for the same reason `HistoricalBarSource`
does — the application layer depends on the shape of an explanation, never on
which vendor produced it.
"""

from .analysis import Confidence, Direction, NewsAnalysis, NewsAnalyst
from .models import RawNewsItem, normalise_headline
from .screening import NewsCategory, ScreenHit, screen
from .tickers import TickerMention, extract_tickers, primary_ticker

__all__ = [
    "Confidence",
    "Direction",
    "NewsAnalysis",
    "NewsAnalyst",
    "NewsCategory",
    "RawNewsItem",
    "ScreenHit",
    "TickerMention",
    "extract_tickers",
    "normalise_headline",
    "primary_ticker",
    "screen",
]
