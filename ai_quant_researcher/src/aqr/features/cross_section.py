"""Comparing a symbol to the rest of its universe.

The largest gap in the vocabulary until now. Every rule saw one instrument at a
time, so "buy the strongest names in a strong market" -- the oldest documented
equity anomaly there is -- could not be written down, and neither could any form
of "this stock is moving and its peers are not".

Three properties carry it, and each fails silently if it is got wrong.

**Point in time.** The cross-section at session ``t`` holds exactly the symbols
that had a bar at ``t``. Ranking against a universe assembled with hindsight is
look-ahead of the most flattering kind: the anomaly it discovers is "the stocks
that were later added to the index outperformed". A symbol contributes from its
first bar and stops at its last.

**Causality.** A rank at ``t`` is built from returns ending at ``t``. Peers doing
something dramatic afterwards cannot reach back and change it, which the tests
pin the same way they pin every other feature.

**Alignment.** Symbols share neither a bar count nor a calendar, so the work is
done on a session grid and the answer is read back onto each symbol's own bars.

The whole grid is built once per universe and memoised, and so is every
per-session reduction taken from it -- ranks, medians, breadth. Both halves of
that matter. Caching the grid but ranking it again for each symbol is still
quadratic in the universe size, and a research loop that redoes a
five-hundred-name ranking five hundred times per feature is a research loop that
does not finish. Everything below the ``symbol`` argument is universe-wide;
the symbol only chooses which column and rows to read back.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np

from aqr.data.bars import Array, Bars

__all__ = ["CrossSection"]

# Below this many peers there is no cross-section worth the name: ranking a
# symbol against one other produces 0.0 or 1.0 and nothing in between, which
# reads as a signal and is not one.
MIN_PEERS = 2


@dataclass(slots=True)
class CrossSection:
    """The universe, aligned on sessions, for peer-relative features."""

    peers: Mapping[str, Bars]
    _sessions: np.ndarray = field(init=False)
    _rows: dict[str, np.ndarray] = field(init=False)
    _columns: dict[str, int] = field(init=False)
    _cache: dict[tuple[str, str, int], Array] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        if not self.peers:
            raise ValueError("a cross-section needs at least one symbol")
        self._sessions = np.unique(
            np.concatenate([b.session for b in self.peers.values()])
        )
        # Where each symbol's bars land on the shared session grid. Computed
        # once; every feature reads it.
        self._rows = {
            symbol: np.searchsorted(self._sessions, bars.session)
            for symbol, bars in self.peers.items()
        }
        # Column order is the sorted symbol order, fixed here so that no lookup
        # re-sorts the universe. Determinism as much as speed: every grid below
        # is indexed by this and by nothing else.
        self._columns = {symbol: i for i, symbol in enumerate(sorted(self.peers))}
        self._cache = {}

    # ------------------------------------------------------------------ #
    # Public features
    # ------------------------------------------------------------------ #

    def rank(self, symbol: str, period: int) -> Array:
        """Percentile rank of this symbol's ``period``-bar return, in [0, 1].

        1.0 is the strongest name in the universe on that session. NaN where the
        symbol has no return yet, or where fewer than :data:`MIN_PEERS` symbols
        do -- a rank against nobody is not a weak signal, it is no signal.
        """
        return self._compute("rank", symbol, period)

    def relative_return(self, symbol: str, period: int) -> Array:
        """This symbol's ``period``-bar return less the universe median.

        The median rather than the mean: one name up 400% should not redefine
        what "average" means for the other forty-nine.
        """
        return self._compute("relative", symbol, period)

    def breadth(self, symbol: str, period: int) -> Array:
        """Fraction of the universe with a positive ``period``-bar return.

        A market-state feature rather than a symbol one -- it is the same number
        for every symbol on a given session, and is aligned to this symbol's
        bars only so it can be combined with the rest.
        """
        return self._compute("breadth", symbol, period)

    # ------------------------------------------------------------------ #

    def _compute(self, kind: str, symbol: str, period: int) -> Array:
        if symbol not in self.peers:
            raise KeyError(f"{symbol} is not in this universe: {sorted(self.peers)}")
        key = (kind, symbol, int(period))
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        period = int(period)
        row_index = self._rows[symbol]
        column = self._columns[symbol]

        if kind == "rank":
            values = self._rank_grid(period)[row_index, column]
        elif kind == "relative":
            mine = self._return_grid(period)[row_index, column]
            values = mine - self._median_rows(period)[row_index]
        else:
            values = self._breadth_rows(period)[row_index]

        out = np.asarray(values, dtype=np.float64)
        out.flags.writeable = False
        self._cache[key] = out
        return out

    # ------------------------------------------------------------------ #
    # Universe-wide grids. Each is computed once per period and shared by
    # every symbol; nothing in here depends on which symbol asked.
    # ------------------------------------------------------------------ #

    def _grid(
        self, kind: str, period: int, build: Callable[[], np.ndarray]
    ) -> np.ndarray:
        """Memoise one universe-wide array under ``(kind, "", period)``."""
        cache_key = (kind, "", period)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return np.asarray(cached)
        value = np.asarray(build(), dtype=np.float64)
        value.flags.writeable = False
        self._cache[cache_key] = value
        return value

    def _return_grid(self, period: int) -> np.ndarray:
        """``sessions x symbols`` of trailing ``period``-bar returns.

        NaN where a symbol has no bar on that session, or has not yet
        accumulated ``period`` bars of its own history. Both absences mean the
        same thing to every consumer here: this name does not take part in this
        session's cross-section.
        """
        return self._grid("grid", period, lambda: self._build_return_grid(period))

    def _build_return_grid(self, period: int) -> np.ndarray:
        grid = np.full(
            (self._sessions.size, len(self._columns)), np.nan, dtype=np.float64
        )
        for symbol, column in self._columns.items():
            bars = self.peers[symbol]
            close = np.asarray(bars.close, dtype=np.float64)
            returns = np.full(close.size, np.nan)
            if close.size > period:
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = close[period:] / close[:-period]
                returns[period:] = np.where(close[:-period] > 0, ratio - 1.0, np.nan)
            grid[self._rows[symbol], column] = returns
        return grid

    def _rank_grid(self, period: int) -> np.ndarray:
        return self._grid(
            "rank_grid", period, lambda: self._rank_rows(self._return_grid(period))
        )

    def _median_rows(self, period: int) -> np.ndarray:
        return self._grid(
            "median_rows", period, lambda: self._row_median(self._return_grid(period))
        )

    def _breadth_rows(self, period: int) -> np.ndarray:
        return self._grid(
            "breadth_rows", period, lambda: self._row_breadth(self._return_grid(period))
        )

    @staticmethod
    def _rank_rows(grid: np.ndarray) -> np.ndarray:
        """Per-session percentile rank, ignoring absent symbols.

        The rank of a name is the count of present peers strictly below it over
        ``n - 1``, so the weakest name is 0.0 and the strongest is 1.0 whatever
        ``n`` is on that session, and ties share the lower value.

        Counted by sorting the row and binary-searching it rather than by
        comparing every pair: ``n log n`` per session instead of ``n**2``. On a
        five-hundred-name universe the pairwise form builds a quarter-million
        element boolean matrix per session, which is what made this unusable.
        """
        present = np.isfinite(grid)
        counts = present.sum(axis=1)
        ranks = np.full(grid.shape, np.nan, dtype=np.float64)
        for i in np.flatnonzero(counts >= MIN_PEERS):
            row = grid[i]
            finite = np.flatnonzero(present[i])
            values = row[finite]
            ordered = np.sort(values)
            below = np.searchsorted(ordered, values, side="left")
            ranks[i, finite] = below / (values.size - 1)
        return ranks

    @staticmethod
    def _row_median(grid: np.ndarray) -> np.ndarray:
        with np.errstate(invalid="ignore"):
            counts = np.isfinite(grid).sum(axis=1)
            median = np.full(grid.shape[0], np.nan)
            usable = counts >= MIN_PEERS
            if usable.any():
                median[usable] = np.nanmedian(grid[usable], axis=1)
        return median

    @staticmethod
    def _row_breadth(grid: np.ndarray) -> np.ndarray:
        present = np.isfinite(grid)
        counts = present.sum(axis=1)
        positive = (present & (grid > 0)).sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            breadth = np.where(
                counts >= MIN_PEERS, positive / np.maximum(counts, 1), np.nan
            )
        return breadth
