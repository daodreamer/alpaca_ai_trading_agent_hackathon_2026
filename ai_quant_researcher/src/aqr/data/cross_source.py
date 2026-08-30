"""Do two vendors tell the same story about the same instrument?

Written because eyeballing stopped scaling. An IBKR ``ADJUSTED_LAST`` pull of
the NASDAQ-50 contained four unadjusted 2:1 splits -- AAPL 2005-02-28, ADBE
2005-05-24, ORLY 2005-06-16, SBUX 2005-10-24, every date an exact split date --
plus a 4.01x discontinuity in FTNT and a 910-day hole in TMUS. All six were
found by reading a list and recognising a date. That works for fifty symbols and
for nobody at five hundred.

The invariant that makes it mechanical:

    Two correctly adjusted series for one instrument may disagree about the
    *price level* -- different adjustment bases, different dividend treatment --
    but they cannot disagree about a *daily return*.

So the comparison is on returns and never on prices. A day where one vendor says
-49.7% and the other says +0.3% is not a rounding difference or a stale close;
it is one of them failing to adjust a split. The comparison cannot say which
vendor is right, but it says exactly which dates to look at, which is the part
that does not scale by hand.

Sessions are matched by *date*, not by timestamp. Vendors stamp the same
session at different instants -- Yahoo at 00:00Z, Alpaca at the 05:00Z open,
IBKR with a bare date -- and matching on the raw instant found nothing in
common between Yahoo and Alpaca across sixteen overlapping years. A
comparison that silently checks nothing is worse than none at all, because it
reads as a clean bill of health.

Nothing here fetches anything. It compares two :class:`Bars` that are already
in hand, which keeps it usable against the CSV cache with no network at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np

from aqr.data.bars import Bars

__all__ = ["ComparisonReport", "compare"]

# Return difference on one day above which two vendors are telling different
# stories. Well above any plausible rounding or close-timing difference, and far
# below the smallest split (a 2:1 moves a return by ~0.5).
TOLERANCE = 0.01

# Ratios that betray an unadjusted split rather than a data error: the price
# halves, thirds or quarters overnight and nothing else in the market does.
_SPLIT_RETURNS = {
    -0.5: "2:1",
    -2 / 3: "3:1",
    -0.75: "4:1",
    -0.8: "5:1",
    -0.9: "10:1",
}
_SPLIT_TOLERANCE = 0.02


@dataclass(frozen=True, slots=True)
class ComparisonReport:
    """Where two vendors' daily returns diverge, and by how much."""

    symbol: str
    left_name: str
    right_name: str
    overlap: int
    """Sessions both sources cover. One shared session yields no return to
    compare, so fewer than two is not agreement -- it is nothing measured."""
    disagreements: list[tuple[datetime, float, float]] = field(default_factory=list)
    """``(date, left return, right return)`` for each day over the tolerance."""
    worst: float = 0.0

    @property
    def comparable(self) -> bool:
        """Whether anything was actually measured.

        Two shared sessions produce one comparable return; one produces none.
        Counting an unmeasurable pair as a disagreement turns a broken
        comparison into a data problem and sends the reader after the wrong bug."""
        return self.overlap >= 2

    @property
    def disagrees(self) -> bool:
        return self.comparable and bool(self.disagreements)

    @property
    def agrees(self) -> bool:
        return self.comparable and not self.disagreements

    def suspected_splits(self) -> list[str]:
        """Disagreements whose shape is an unadjusted split, and on which side.

        Naming the ratio saves the next person a trip to a corporate-actions
        calendar to work out whether -49.7% was a split or a bad day.
        """
        out: list[str] = []
        for moment, left, right in self.disagreements:
            for value, ratio in _SPLIT_RETURNS.items():
                if abs(left - value) < _SPLIT_TOLERANCE and abs(right) < _SPLIT_TOLERANCE:
                    out.append(
                        f"{moment.date()}: {self.left_name} shows a {ratio} split "
                        f"that {self.right_name} has already adjusted"
                    )
                elif abs(right - value) < _SPLIT_TOLERANCE and abs(left) < _SPLIT_TOLERANCE:
                    out.append(
                        f"{moment.date()}: {self.right_name} shows a {ratio} split "
                        f"that {self.left_name} has already adjusted"
                    )
        return out

    def __str__(self) -> str:
        head = f"{self.symbol}: {self.left_name} vs {self.right_name}"
        if self.overlap == 0:
            return f"{head} -- no overlap, nothing compared"
        if self.agrees:
            return f"{head} -- agree on all {self.overlap - 1} daily returns"
        lines = [
            f"{head} -- {len(self.disagreements)} of {self.overlap - 1} daily returns differ "
            f"(worst {self.worst:.1%})"
        ]
        for moment, left, right in self.disagreements[:10]:
            lines.append(
                f"    {moment.date()}  {self.left_name} {left:+7.1%}   "
                f"{self.right_name} {right:+7.1%}"
            )
        if len(self.disagreements) > 10:
            lines.append(f"    ... and {len(self.disagreements) - 10} more")
        lines += [f"  ! {note}" for note in self.suspected_splits()]
        return "\n".join(lines)

    def as_dict(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "left": self.left_name,
            "right": self.right_name,
            "overlap": self.overlap,
            "disagreements": [
                (m.isoformat(), left, right) for m, left, right in self.disagreements
            ],
            "worst": self.worst,
            "suspected_splits": self.suspected_splits(),
        }


def compare(
    left: Bars,
    right: Bars,
    *,
    left_name: str = "left",
    right_name: str = "right",
    tolerance: float = TOLERANCE,
) -> ComparisonReport:
    """Compare two sources' daily returns over the days they both cover.

    Returns, not prices: a constant price factor between two correctly adjusted
    series is legitimate and must not read as a disagreement.
    """
    # By session date, not by instant: the same trading day carries a
    # different timestamp at every vendor.
    shared = np.intersect1d(left.session, right.session)
    if shared.size < 2:
        return ComparisonReport(
            symbol=left.symbol,
            left_name=left_name,
            right_name=right_name,
            overlap=int(shared.size),
        )

    left_close = _closes_at(left, shared)
    right_close = _closes_at(right, shared)

    with np.errstate(divide="ignore", invalid="ignore"):
        left_ret = np.diff(left_close) / left_close[:-1]
        right_ret = np.diff(right_close) / right_close[:-1]

    delta = np.abs(left_ret - right_ret)
    # A day either source could not price is not a disagreement about that day.
    delta = np.where(np.isfinite(delta), delta, 0.0)
    offenders = np.flatnonzero(delta > tolerance)

    disagreements = [
        (
            datetime.fromtimestamp(int(shared[i + 1]) * 86_400, tz=UTC),
            float(left_ret[i]),
            float(right_ret[i]),
        )
        for i in offenders
    ]
    return ComparisonReport(
        symbol=left.symbol,
        left_name=left_name,
        right_name=right_name,
        overlap=int(shared.size),
        disagreements=disagreements,
        worst=float(delta.max()) if delta.size else 0.0,
    )


def _closes_at(bars: Bars, sessions: np.ndarray) -> np.ndarray:
    """Closes for exactly these session dates, in order."""
    index = np.searchsorted(bars.session, sessions)
    return np.asarray(bars.close, dtype=np.float64)[index]
