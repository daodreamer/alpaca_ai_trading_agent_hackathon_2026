"""What a news source hands over, before anything has judged it.

`RawNewsItem` is deliberately close to the wire: a headline, a teaser, a link
and two timestamps. It carries no score, no event type and no opinion, because
those are decisions and this is evidence. Whatever classifies it later — a
keyword screen, a language model, a person reading it — is replaceable, and the
record of what arrived is not.

The two timestamps are the reason this type exists at all. `published_at` is
what the wire claims; `received_at` is when this process actually saw it. Only
the second one is a fact about our system, and only the second one is safe to
measure latency or backtest against (`CLAUDE.md` §12). Wires backfill and
correct `published_at`; nothing can retroactively change when we read the feed.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime

__all__ = ["RawNewsItem", "normalise_headline"]

_PUNCTUATION = re.compile(r"[^\w\s]+")
_WHITESPACE = re.compile(r"\s+")


def normalise_headline(headline: str) -> str:
    """A headline reduced to what makes it the same story.

    Wires reissue a release to correct a typo, add a boilerplate line or push a
    translated copy, and every reissue is a new `external_id`. Comparing the
    text as delivered would let all of them through as separate alerts, which is
    exactly the failure a person notices — the same FDA decision arriving five
    times.

    Case, punctuation, accents and runs of whitespace are all dropped: none of
    them distinguish two stories, and each of them differs between a release and
    its correction.
    """
    folded = unicodedata.normalize("NFKD", headline)
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    folded = _PUNCTUATION.sub(" ", folded.casefold())
    return _WHITESPACE.sub(" ", folded).strip()


@dataclass(frozen=True, slots=True)
class RawNewsItem:
    """One story as one source delivered it.

    Frozen because this is the ingest record. Anything downstream that wants to
    add a judgement builds its own object rather than mutating the evidence.
    """

    source: str
    """Which adapter produced this — `businesswire`, `prnewswire`, `massive`."""

    external_id: str
    """The source's own identifier, usually its permalink. Half of the dedupe key."""

    headline: str
    summary: str
    url: str

    received_at: datetime
    """When this process read it. Always tz-aware UTC; the only honest clock here."""

    published_at: datetime | None = None
    """When the wire says it went out. `None` when the feed omitted or mangled it."""

    def __post_init__(self) -> None:
        # A naive timestamp here would silently become "whatever the server's
        # locale is" the first time it is compared or stored, and this feed is
        # read by a process whose timezone nobody has thought about. `CLAUDE.md`
        # §5 asks for explicit timezones; this is where that is cheap to enforce.
        _require_utc("received_at", self.received_at)
        if self.published_at is not None:
            _require_utc("published_at", self.published_at)
        if not self.external_id:
            raise ValueError("external_id must not be empty: it is half the dedupe key")

    @property
    def dedupe_key(self) -> tuple[str, str]:
        """Identity within one source: the source and its own id."""
        return (self.source, self.external_id)

    @property
    def story_key(self) -> str:
        """Identity *across* sources.

        The same press release reaches us from Business Wire and PR Newswire
        under different ids and different links, and a person wants one alert.
        Hashing the normalised headline is crude but it is the only field the
        two copies reliably share.
        """
        return hashlib.sha256(normalise_headline(self.headline).encode()).hexdigest()

    @property
    def latency_seconds(self) -> float | None:
        """How far behind the wire we were, when the wire said anything.

        This is the number that decides whether the whole feature is worth
        running, so it is computed from the record rather than logged in passing.
        """
        if self.published_at is None:
            return None
        return (self.received_at - self.published_at).total_seconds()

    @property
    def text(self) -> str:
        """Headline and teaser together — what a screen or a model reads."""
        return f"{self.headline}\n{self.summary}".strip()


def _require_utc(field: str, value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    if value.utcoffset() != UTC.utcoffset(None):
        raise ValueError(f"{field} must be in UTC, got offset {value.utcoffset()}")
