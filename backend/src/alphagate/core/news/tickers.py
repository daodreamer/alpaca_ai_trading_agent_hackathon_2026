"""Which listed company a press release is about.

Wire services write the ticker into the first sentence — "Uber Technologies,
Inc (NYSE: UBER) today announced" — because their customers are newsrooms that
need it. That convention is the whole reason this module is thirty lines of
regex instead of a company-name index with fuzzy matching behind it.

It also does a second job for free. A release with no exchange-qualified ticker
in it is almost always about a private company, and a monitor for *stocks* has
nothing to say about a Series B. So failing to find a ticker is a filter, not
an error: the item is dropped rather than escalated to something cleverer.
That direction is deliberate — a missed alert costs one story, and a wrong
ticker on a confident alert costs trust in every alert after it.
"""

from __future__ import annotations

import re

__all__ = ["US_EXCHANGES", "TickerMention", "extract_tickers"]

US_EXCHANGES: frozenset[str] = frozenset(
    {
        "NYSE",
        "NYSE AMERICAN",
        "NYSEAMERICAN",
        "NYSE ARCA",
        "NYSEARCA",
        "AMEX",
        "NASDAQ",
        "NASDAQ GLOBAL SELECT MARKET",
        "NASDAQ GLOBAL MARKET",
        "NASDAQ CAPITAL MARKET",
        "NASDAQGS",
        "NASDAQGM",
        "NASDAQCM",
        "CBOE",
        "BATS",
        "OTCQB",
        "OTCQX",
        "OTC",
        "OTCMKTS",
        "OTC MARKETS",
    }
)
"""Venues whose symbols are tradeable in the account this monitor watches.

Everything else is recognised and then discarded. That matters more than it
looks: "(NASDAQ: BIDU and HKEX: 9888)" is one company on two venues, and an
alert naming `9888` would be unactionable noise.
"""

# The exchange half is matched loosely and validated afterwards against
# US_EXCHANGES, rather than being spelled out as an alternation. Wires invent
# new venue spellings faster than a pattern can be maintained, and an unknown
# venue must be *dropped*, which requires having parsed it first.
_MENTION = re.compile(
    r"""
    \(?                                  # the opening paren, when there is one
    \b(?P<exchange>[A-Za-z][A-Za-z./\ ]{1,28}?)  # "NASDAQ", "NYSE American", "Nasdaq/TASE"
    \s*:\s*
    (?P<symbol>[A-Z]{1,5}(?:[.\-][A-Z]{1,2})?)   # BRK.B, RDS-A
    \b
    """,
    re.VERBOSE,
)

_SYMBOL_OK = re.compile(r"^[A-Z]{1,5}(?:[.\-][A-Z]{1,2})?$")


class TickerMention:
    """One `EXCHANGE: SYMBOL` found in the text, and where it was found.

    Position is kept because order carries meaning on a wire: the subject of a
    release is named first, and any company mentioned later is usually a
    partner, an acquirer's target or a boilerplate "about" paragraph.
    """

    __slots__ = ("exchange", "position", "symbol")

    def __init__(self, exchange: str, symbol: str, position: int) -> None:
        self.exchange = exchange
        self.symbol = symbol
        self.position = position

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"TickerMention({self.exchange}:{self.symbol}@{self.position})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TickerMention):
            return NotImplemented
        return (self.exchange, self.symbol, self.position) == (
            other.exchange,
            other.symbol,
            other.position,
        )

    def __hash__(self) -> int:
        return hash((self.exchange, self.symbol, self.position))


def _us_venue(raw: str) -> str | None:
    """The US venue named in an exchange label, if one is.

    Wires write a dual listing as one label — "Nasdaq/TASE: ODYS" — and the
    symbol that follows is the US one. Reading the label whole would classify
    that release as foreign and silently drop a tradeable event, which is how a
    real offering announcement was lost the first time this ran against a live
    feed.
    """
    for part in raw.split("/"):
        venue = " ".join(part.replace(".", "").split()).upper()
        if venue in US_EXCHANGES:
            return venue
    return None


def extract_tickers(text: str) -> list[TickerMention]:
    """Every US-listed ticker named in `text`, first mention first.

    De-duplicated by symbol, keeping the earliest position, so a company named
    in both the lede and the boilerplate counts once and keeps the rank its
    lede gave it.
    """
    seen: dict[str, TickerMention] = {}
    for match in _MENTION.finditer(text):
        exchange = _us_venue(match.group("exchange"))
        if exchange is None:
            continue
        symbol = match.group("symbol")
        if not _SYMBOL_OK.match(symbol):
            continue
        if symbol not in seen:
            seen[symbol] = TickerMention(exchange, symbol, match.start("symbol"))
    return sorted(seen.values(), key=lambda m: m.position)


def primary_ticker(text: str) -> str | None:
    """The one symbol an alert should be filed under, if any.

    The first US-listed mention. On a press release that is the issuer, because
    the wire's own house style puts the issuing company first.
    """
    mentions = extract_tickers(text)
    return mentions[0].symbol if mentions else None
