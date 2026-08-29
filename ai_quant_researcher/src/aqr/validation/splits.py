"""Train / validation / test splits and walk-forward windows.

Architecture sections 12 and 13. The rule the whole module exists to serve:

    The test segment is never seen -- not by a parameter search, not by a human
    picking the best of ten runs, and above all not by the LLM.

Splits are by *position in the bar series*, and always contiguous in time.
Shuffled or random splits would let a strategy learn from its own future, which
is the same look-ahead bias the backtester works to avoid, laundered through the
sampling scheme.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np

from aqr.data.bars import Bars

__all__ = [
    "TEST_BARS",
    "TRAIN_BARS",
    "Fold",
    "Split",
    "context_slice",
    "purge_overlap",
    "three_way_split",
    "walk_forward_folds",
]


@dataclass(frozen=True, slots=True)
class Split:
    """Positional bounds of a contiguous segment, half-open ``[start, stop)``."""

    start: int
    stop: int
    label: str

    def __post_init__(self) -> None:
        if self.stop < self.start:
            raise ValueError(f"{self.label}: stop {self.stop} precedes start {self.start}")

    def __len__(self) -> int:
        return self.stop - self.start

    def apply(self, bars: Bars) -> Bars:
        return bars.slice(self.start, self.stop)


@dataclass(frozen=True, slots=True)
class Fold:
    """One walk-forward step: fit on ``train``, judge on ``test``."""

    index: int
    train: Split
    test: Split

    def describe(self, bars: Bars) -> str:
        def when(pos: int) -> str:
            pos = min(max(pos, 0), len(bars) - 1)
            return datetime.fromtimestamp(int(bars.event_time[pos]), tz=UTC).date().isoformat()

        return (
            f"fold {self.index}: train {when(self.train.start)}..{when(self.train.stop - 1)} "
            f"({len(self.train)} bars) -> test {when(self.test.start)}..{when(self.test.stop - 1)} "
            f"({len(self.test)} bars)"
        )


def three_way_split(
    n: int, train: float = 0.6, validation: float = 0.2, test: float = 0.2
) -> tuple[Split, Split, Split]:
    """Chronological 60/20/20 split by default (architecture section 12)."""
    total = train + validation + test
    if not np.isclose(total, 1.0):
        raise ValueError(f"split fractions must sum to 1.0, got {total}")
    if n < 3:
        raise ValueError(f"need at least 3 bars to split, got {n}")
    train_stop = int(n * train)
    validation_stop = train_stop + int(n * validation)
    return (
        Split(0, train_stop, "train"),
        Split(train_stop, validation_stop, "validation"),
        Split(validation_stop, n, "test"),
    )


# The default walk-forward geometry, in daily bars.
#
# Chosen for the 2180-session window the point-in-time S&P universe covers
# (2016-01-04 .. 2024-08-30), which is the window every result in this project
# is now measured on.
#
# The previous 756 / 252 was set for a ten-year window. On 2180 sessions it
# yields 6 folds, 5 once a 21-bar embargo is applied. ``positive_fold_rate`` is
# the most common rejection reason in the evaluator, and a rate estimated from 5
# folds moves 20 points when a single fold changes sign -- the gate would be
# measuring fold arithmetic rather than the strategy.
#
# 504 / 126 gives 13 folds, 11 with a 21-bar embargo. The price is a two-year
# fit window rather than three, and that is the trade, taken deliberately.
#
# ``MIN_POSITIVE_FOLD_RATE`` is deliberately NOT loosened to compensate. If the
# shorter window cannot support the gate, that is a finding about the window and
# it should be reported as one, not absorbed into a threshold.
TRAIN_BARS = 504
TEST_BARS = 126


def walk_forward_folds(
    n: int,
    train_bars: int,
    test_bars: int,
    *,
    anchored: bool = True,
    embargo_bars: int = 0,
) -> list[Fold]:
    """Rolling out-of-sample windows.

    ``anchored``      train windows all start at bar 0 and grow (the default,
                      matching section 13); otherwise they slide at fixed width.
    ``embargo_bars``  gap left between train and test. With overlapping labels
                      (a 20-bar holding period, say) the last trades of the
                      train window resolve inside the test window; the embargo
                      removes that leakage.
    """
    if train_bars < 1 or test_bars < 1:
        raise ValueError("train_bars and test_bars must both be >= 1")

    folds: list[Fold] = []
    train_start = 0
    train_stop = train_bars
    index = 0
    while True:
        test_start = train_stop + embargo_bars
        test_stop = min(test_start + test_bars, n)
        if test_start >= n or test_stop - test_start < max(1, test_bars // 2):
            # A stub final window would report a metric on a handful of bars and
            # then be averaged in with equal weight. Drop it instead.
            break
        folds.append(
            Fold(
                index=index,
                train=Split(train_start, train_stop, f"train[{index}]"),
                test=Split(test_start, test_stop, f"test[{index}]"),
            )
        )
        index += 1
        train_stop = test_stop
        if not anchored:
            train_start = max(0, train_stop - train_bars)
        if train_stop >= n:
            break
    return folds


def purge_overlap(folds: list[Fold], holding_bars: int) -> list[Fold]:
    """Push each test window forward so trades opened in train cannot bleed in.

    Equivalent to an embargo sized to the strategy's own holding period, applied
    after the fact. Folds left too small to judge are dropped.
    """
    if holding_bars <= 0:
        return folds
    purged: list[Fold] = []
    for fold in folds:
        start = fold.test.start + holding_bars
        if fold.test.stop - start < max(1, len(fold.test) // 2):
            continue
        purged.append(
            Fold(
                index=len(purged),
                train=fold.train,
                test=Split(start, fold.test.stop, fold.test.label),
            )
        )
    return purged


def context_slice(bars: Bars, split: Split, warmup_bars: int) -> tuple[Bars, int]:
    """Bars for a fold, widened backwards by ``warmup_bars``.

    Returns the widened series and the offset at which the fold's own window
    begins inside it. Callers report performance from that offset onward, so the
    warm-up history informs the indicators without contributing trades.
    """
    start = max(0, split.start - warmup_bars)
    return bars.slice(start, split.stop), split.start - start
