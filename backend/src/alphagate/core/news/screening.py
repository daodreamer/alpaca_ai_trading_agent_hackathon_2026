r"""The cheap first pass: is this story the kind that moves a stock in a day?

Wires carry a few thousand releases a day and almost all of them are conference
appearances, board appointments and product launches. Sending every one to a
language model would cost real money and, worse, would add queueing delay to the
handful that matter — the ones where being two minutes late is the whole
difference.

So a regex screen runs first and rejects most of the volume in microseconds. It
is deliberately *loose*: its job is to be sure it never drops a real FDA
approval, not to be sure everything it passes is one. Precision is the model's
job downstream, where there is context to be precise with.

Adding a category is a product decision with a spec behind it, not a regex
someone appends on a hunch. The categories here cover two sectors on purpose:
regulatory and clinical events for biotech issuers, operating and capital events
for everyone else. Polling a technology wire while only screening for FDA
decisions would mean reading a hundred semiconductor releases a day and seeing
none of them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

__all__ = ["NewsCategory", "ScreenHit", "screen"]


class NewsCategory(StrEnum):
    """What kind of event a headline looks like.

    A *look*, not a finding: `CLINICAL_RESULT` means "this mentions a trial
    readout", not "a trial succeeded". Direction is never inferred here — a
    Phase 3 headline reads the same whether the drug worked or failed, and a
    guidance headline reads the same whether it was raised or cut. Guessing from
    keywords is how a monitor tells someone the opposite of the truth.
    """

    # Regulatory and clinical — what moves a biotech issuer.
    FDA_DECISION = "FDA_DECISION"
    CLINICAL_RESULT = "CLINICAL_RESULT"

    # Capital structure and control — sector-independent. An acquisition, a
    # dilutive raise or a bankruptcy moves any issuer the day it is announced.
    MERGER_ACQUISITION = "MERGER_ACQUISITION"
    OFFERING = "OFFERING"
    BANKRUPTCY = "BANKRUPTCY"

    # Operating outlook and demand — where a technology issuer actually moves.
    # A chipmaker does not get an FDA approval; it re-rates on guidance and on
    # the size of what it just sold.
    GUIDANCE = "GUIDANCE"
    MAJOR_CONTRACT = "MAJOR_CONTRACT"

    # Forced flow. An index addition moves a stock through mechanical buying
    # rather than through anything changing in the business.
    INDEX_CHANGE = "INDEX_CHANGE"


_PATTERNS: dict[NewsCategory, re.Pattern[str]] = {
    NewsCategory.FDA_DECISION: re.compile(
        r"""
        \bFDA\b
        | \bU\.?S\.?\s+Food\s+and\s+Drug\s+Administration\b
        | \bcomplete\s+response\s+letter\b | \bCRL\b
        | \b(?:510\(k\)|PMA|BLA|NDA|sNDA|sBLA|ANDA|IND)\b
        | \bbreakthrough\s+therapy\b | \bfast\s+track\b
        | \borphan\s+drug\b | \bpriority\s+review\b
        | \bPDUFA\b
        | \bmarketing\s+authoris?z?ation\b
        | \bclinical\s+hold\b
        """,
        re.IGNORECASE | re.VERBOSE,
    ),
    NewsCategory.CLINICAL_RESULT: re.compile(
        r"""
        \bPhase\s*(?:1|2|3|I{1,3})\b
        | \btop-?line\b
        | \bprimary\s+endpoint\b
        | \bpivotal\s+(?:trial|study)\b
        | \binterim\s+analysis\b
        | \bstatistically\s+significant\b
        """,
        re.IGNORECASE | re.VERBOSE,
    ),
    NewsCategory.MERGER_ACQUISITION: re.compile(
        r"""
        \bdefinitive\s+(?:merger\s+)?agreement\b
        | \bto\s+(?:be\s+)?acquir\w+\b | \bacquisition\s+of\b
        | \bagrees?\s+to\s+acquire\b | \bwill\s+acquire\b
        | \bmerger\b | \bto\s+merge\s+with\b
        | \btender\s+offer\b
        | \bstrategic\s+alternatives\b
        | \bbuyout\b | \btake[\s-]private\b
        """,
        re.IGNORECASE | re.VERBOSE,
    ),
    NewsCategory.OFFERING: re.compile(
        r"""
        \bpublic\s+offering\b | \bprivate\s+placement\b
        | \bregistered\s+direct\s+offering\b
        | \bat[\s-]the[\s-]market\s+offering\b | \bATM\s+program\b
        | \bpricing\s+of\b.{0,40}\boffering\b
        | \bconvertible\s+(?:senior\s+)?notes\b
        | \bshelf\s+registration\b
        | \bproposed\s+offering\b
        """,
        re.IGNORECASE | re.VERBOSE,
    ),
    NewsCategory.BANKRUPTCY: re.compile(
        r"""
        \bchapter\s*(?:7|11|15)\b
        | \bbankruptcy\s+protection\b | \bfiles?\s+for\s+bankruptcy\b
        | \bgoing\s+concern\b
        | \breceivership\b | \bliquidation\b
        | \bnotice\s+of\s+non-?compliance\b
        | \bdelisting\b | \bdeficiency\s+letter\b
        | \bdefaults?\s+on\b[^.]{0,30}?\b(?:notes|loan|debt)\b
        """,
        re.IGNORECASE | re.VERBOSE,
    ),
    NewsCategory.GUIDANCE: re.compile(
        r"""
        \b(?:raises|raised|lowers|lowered|cuts|reduces|reaffirms|reiterates
           |updates|withdraws|suspends|revises|revised)\b
          [^.]{0,40}?\b(?:guidance|outlook|forecast|expectations)\b
        | \b(?:guidance|outlook|forecast)\b
          [^.]{0,30}?\b(?:raised|lowered|increased|reduced|updated|withdrawn)\b
        | \bpreliminary\b[^.]{0,40}?\b(?:results|revenue|earnings)\b
        | \bpre-?announces\b
        | \bprofit\s+warning\b
        """,
        re.IGNORECASE | re.VERBOSE,
    ),
    NewsCategory.MAJOR_CONTRACT: re.compile(
        r"""
        # A contract only matters at a size worth naming, so either a figure or
        # a government counterparty is required. "Wins contract" with neither is
        # a press release about a sales call, and tech wires are full of them.
        \b(?:awarded|awards|wins|secures|receives|selected\s+for)\b
          [^.]{0,60}?
          (?: \$\s?\d[\d,.]*\s*(?:million|billion)
            | \b(?:U\.?S\.?\s+)?(?:Air\s+Force|Army|Navy|Space\s+Force
               |Department\s+of\s+Defense|DoD|NASA|DARPA|Pentagon)\b )
        | \$\s?\d[\d,.]*\s*(?:million|billion)
          [^.]{0,40}?\b(?:contract|order|purchase\s+agreement|award)\b
        | \bIDIQ\b | \bindefinite\s+delivery\b
        """,
        re.IGNORECASE | re.VERBOSE,
    ),
    NewsCategory.INDEX_CHANGE: re.compile(
        r"""
        \b(?:to\s+join|set\s+to\s+join|will\s+join|added\s+to|will\s+replace)\b
          [^.]{0,25}?
          \b(?:S&P\s*(?:500|400|600)|Nasdaq-?100|Russell\s*(?:1000|2000))\b
        | \b(?:S&P\s*(?:500|400|600)|Nasdaq-?100|Russell\s*(?:1000|2000))\b
          [^.]{0,30}?\b(?:addition|inclusion|will\s+replace)\b
        """,
        re.IGNORECASE | re.VERBOSE,
    ),
}

# Releases that match a category by vocabulary while announcing nothing. On a
# biotech wire the worst offender is "to present Phase 3 data at a conference";
# on a technology wire it is the product launch and the analyst-report award.
# Both are announced constantly and neither moves a stock.
_NON_EVENT = re.compile(
    r"""
    \bto\s+present\b | \bwill\s+present\b | \bpresentation\s+at\b
    | \bposter\s+session\b | \babstract\b
    | \binvestor\s+(?:conference|day)\b | \bwebcast\b
    | \bfireside\s+chat\b
    | \bconference\s+call\s+to\s+discuss\b
    | \bto\s+report\b[^.]{0,30}?\bresults\s+on\b
    | \bappoint\w*\s+(?:to\s+)?(?:the\s+)?board\b
    | \bnames?\s+\w+\s+as\s+(?:chief|vice\s+president)\b
    | \bannual\s+meeting\s+of\s+(?:stockholders|shareholders)\b
    | \bawarded?\b.{0,30}\b(?:prize|award\s+for)\b
    | \bnamed?\s+an?\s+leader\b | \bmagic\s+quadrant\b
    | \bwins\b[^.]{0,30}?\baward\b
    | \blaunch(?:es|ed|ing)\b | \bunveil(?:s|ed|ing)\b
    | \bintroduc(?:es|ed|ing)\b
    | \bstrategic\s+partnership\b
    | \bexpands?\b[^.]{0,30}?\b(?:footprint|presence|operations)\b
    # Trial operations. These share every keyword with a trial readout while
    # announcing nothing about whether the drug works, which is the only part
    # that moves a stock. One of them reached a phone before this existed.
    | \bselects?\b[^.]{0,40}?\b(?:CRO|CDMO|manufacturing\s+partner)\b
    | \bdoses?\s+(?:the\s+)?first\s+patient\b
    | \bfirst\s+patient\s+(?:dosed|enrolled|treated)\b
    | \bcompletes?\s+enrollment\b | \benrollment\s+(?:is\s+)?complete\b
    | \binitiat(?:es|ed|ion\s+of)\b[^.]{0,40}?\b(?:trial|study)\b
    | \b(?:begins?|commences?)\b[^.]{0,40}?\b(?:trial|study|dosing)\b
    | \bIRB\s+approval\b | \bethics\s+committee\b
    | \bexpands?\b[^.]{0,40}?\b(?:trial|study)\b
    | \bsite\s+activation\b
    """,
    re.IGNORECASE | re.VERBOSE,
)


@dataclass(frozen=True, slots=True)
class ScreenHit:
    """Why the screen let a story through."""

    categories: frozenset[NewsCategory]
    matched: tuple[str, ...]
    """The literal substrings that matched, so a surprising alert can be explained."""


def screen(text: str) -> ScreenHit | None:
    """`None` when the story is not worth a model call.

    A story can land in more than one category — an acquisition of a company
    with a Phase 3 asset is both — and all of them are kept. Narrowing to one is
    a judgement, and the screen does not make judgements.
    """
    if _NON_EVENT.search(text):
        return None

    categories: set[NewsCategory] = set()
    matched: list[str] = []
    for category, pattern in _PATTERNS.items():
        found = pattern.search(text)
        if found is not None:
            categories.add(category)
            matched.append(found.group(0).strip())

    if not categories:
        return None
    return ScreenHit(categories=frozenset(categories), matched=tuple(matched))
