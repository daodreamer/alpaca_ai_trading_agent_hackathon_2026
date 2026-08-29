"""The market structure engine — specs/06-structure.md, ADR 0008.

Identifies swings, labels them against the previous swing of the same kind, and
reports breaks of the most recent unbroken structural level. It does not decide
anything about trend and it does not notify anyone (CLAUDE.md §5).

The engine is a `BarConsumer`, so it inherits the series binding, ordering and
partial-bar rules that indicators already follow. It *composes* indicators —
ATR for the noise filter, a volume average for volume confirmation — rather than
recomputing them.

Everything it emits carries the bar that produced it, and nothing it has emitted
is ever revised (ADR 0008 D6).
"""

from __future__ import annotations

import dataclasses
from collections import deque
from decimal import Decimal

from alphagate.core.bar import Bar
from alphagate.core.errors import InvariantViolation
from alphagate.core.indicators import Atr, VolumeSma
from alphagate.core.streaming import BarConsumer, ComponentSpec, require_period
from alphagate.core.structure.model import (
    BreakEvent,
    BreakKind,
    EqualPricePolicy,
    StructureLabel,
    StructureUpdate,
    SwingKind,
    SwingPoint,
    SwingStatus,
)

__all__ = ["BreakPolicy", "StructureEngine"]


@dataclasses.dataclass(frozen=True, slots=True)
class BreakPolicy:
    """How a break of structure is confirmed (specs/06-structure.md, ADR 0008 D7)."""

    closes_required: int = 1
    """Consecutive closes beyond the level. A close back inside resets the count."""

    use_wicks: bool = False
    """Whether an intrabar high/low beyond the level confirms. Off: a wick is not a break."""

    volume_multiple: float | None = None
    """When set, the breaking bar's volume must be at least this multiple of the
    average of the preceding `volume_period` bars. The breaking bar is excluded
    from that average, or a heavy bar would inflate its own benchmark."""

    volume_period: int = 20

    def __post_init__(self) -> None:
        require_period(self.closes_required, field="closes_required")
        require_period(self.volume_period, field="volume_period")
        if self.volume_multiple is not None and self.volume_multiple <= 0:
            raise InvariantViolation(
                f"volume_multiple: must be positive when set, got {self.volume_multiple}"
            )


@dataclasses.dataclass(frozen=True, slots=True)
class _Observed:
    """A bar as the engine needs it: extremes, close, and the ATR at that moment."""

    bar: Bar
    atr: float | None


class StructureEngine(BarConsumer[StructureUpdate]):
    """Swing detection, structure labels and break of structure."""

    def __init__(
        self,
        *,
        left_bars: int = 3,
        right_bars: int = 3,
        minimum_move_atr: float = 0.0,
        atr_period: int = 14,
        equal_prices: EqualPricePolicy = EqualPricePolicy.STRICT,
        breaks: BreakPolicy | None = None,
        allow_partial: bool = False,
    ) -> None:
        super().__init__(allow_partial=allow_partial)
        self.left_bars = require_period(left_bars, field="left_bars")
        self.right_bars = require_period(right_bars, field="right_bars")
        if minimum_move_atr < 0:
            raise InvariantViolation(
                f"minimum_move_atr: must not be negative, got {minimum_move_atr}"
            )
        self.minimum_move_atr = minimum_move_atr
        self.atr_period = require_period(atr_period, field="atr_period")
        self.equal_prices = equal_prices
        self.breaks_policy = breaks if breaks is not None else BreakPolicy()

        self._atr = Atr(period=self.atr_period)
        self._volume = VolumeSma(period=self.breaks_policy.volume_period)
        self._window: deque[_Observed] = deque(maxlen=self.left_bars + self.right_bars + 1)

        self._swings: list[SwingPoint] = []
        self._breaks: list[BreakEvent] = []
        self._last_high: SwingPoint | None = None
        self._last_low: SwingPoint | None = None
        self._structural_high: SwingPoint | None = None
        self._structural_low: SwingPoint | None = None
        self._closes_above = 0
        self._closes_below = 0

    # -- identity ----------------------------------------------------------

    @property
    def spec(self) -> ComponentSpec:
        return ComponentSpec(
            "structure",
            (
                ("left", str(self.left_bars)),
                ("right", str(self.right_bars)),
                ("min_move_atr", str(self.minimum_move_atr)),
                ("atr", str(self.atr_period)),
                ("equal", self.equal_prices.value),
                ("closes", str(self.breaks_policy.closes_required)),
                ("wicks", str(self.breaks_policy.use_wicks)),
            ),
        )

    @property
    def warmup(self) -> int:
        """Bars before structure can appear.

        The pivot window, or ATR's warmup when the noise filter needs it: a
        filter that switches on part-way through would make the first structure
        of a rebuild depend on where the data was cut (ADR 0008 D4).
        """
        pivot_window = self.left_bars + self.right_bars + 1
        if self.minimum_move_atr > 0:
            return max(pivot_window, self._atr.warmup)
        return pivot_window

    # -- reading -----------------------------------------------------------

    @property
    def swings(self) -> tuple[SwingPoint, ...]:
        """Confirmed swings, in pivot order. Append-only (ADR 0008 D6)."""
        return tuple(self._swings)

    @property
    def breaks(self) -> tuple[BreakEvent, ...]:
        return tuple(self._breaks)

    @property
    def structural_high(self) -> SwingPoint | None:
        """The most recent confirmed swing high that has not been broken."""
        return self._structural_high

    @property
    def structural_low(self) -> SwingPoint | None:
        return self._structural_low

    # -- writing -----------------------------------------------------------

    def _consume(self, bar: Bar) -> StructureUpdate:
        volume_benchmark = self._volume.value  # as of the previous bar, by design
        self._window.append(_Observed(bar=bar, atr=self._atr.update(bar)))

        confirmed = self._confirm_pivots(bar)
        breaks = self._detect_breaks(bar, volume_benchmark)
        self._volume.update(bar)

        return StructureUpdate(
            confirmed_swings=confirmed,
            breaks=breaks,
            candidate_high=self._candidate(bar, SwingKind.HIGH),
            candidate_low=self._candidate(bar, SwingKind.LOW),
        )

    # -- pivots ------------------------------------------------------------

    def _confirm_pivots(self, bar: Bar) -> tuple[SwingPoint, ...]:
        """Evaluate the bar that now has a full right-hand window."""
        if len(self._window) < self._window.maxlen:  # type: ignore[operator]
            return ()

        centre = self._window[self.left_bars]
        confirmed: list[SwingPoint] = []

        for kind in (SwingKind.HIGH, SwingKind.LOW):
            if not self._is_pivot(self.left_bars, kind):
                continue
            swing = SwingPoint(
                kind=kind,
                status=SwingStatus.CONFIRMED,
                bar_start_utc=centre.bar.start_time_utc,
                detected_at_utc=bar.start_time_utc,
                price=_extreme(centre.bar, kind),
            )
            recorded = self._record(swing, atr=centre.atr)
            if recorded is not None:
                confirmed.append(recorded)

        return tuple(confirmed)

    def _is_pivot(
        self, position: int, kind: SwingKind, *, right_available: int | None = None
    ) -> bool:
        """Left-inclusive, right-strict (ADR 0008 D2).

        `right_available` limits how many right-hand bars are considered, which
        is what makes a candidate evaluable before its window has filled.

        Both call sites pass `position >= left_bars`, so the left window is
        always in range.
        """
        window = self._window
        centre = _extreme(window[position].bar, kind)
        right_count = self.right_bars if right_available is None else right_available

        for offset in range(1, self.left_bars + 1):
            neighbour = _extreme(window[position - offset].bar, kind)
            if kind is SwingKind.HIGH:
                if centre < neighbour:
                    return False
            elif centre > neighbour:
                return False

        for offset in range(1, right_count + 1):
            neighbour = _extreme(window[position + offset].bar, kind)
            if kind is SwingKind.HIGH:
                if centre <= neighbour:
                    return False
            elif centre >= neighbour:
                return False

        return True

    def _candidate(self, bar: Bar, kind: SwingKind) -> SwingPoint | None:
        """The forming swing: full left window, right window still filling."""
        window = self._window
        for position in range(len(window) - 1, self.left_bars - 1, -1):
            available = len(window) - 1 - position
            if available >= self.right_bars:
                break  # already resolved by _confirm_pivots
            if self._is_pivot(position, kind, right_available=available):
                return SwingPoint(
                    kind=kind,
                    status=SwingStatus.CANDIDATE,
                    bar_start_utc=window[position].bar.start_time_utc,
                    detected_at_utc=bar.start_time_utc,
                    price=_extreme(window[position].bar, kind),
                )
        return None

    def _record(self, swing: SwingPoint, *, atr: float | None) -> SwingPoint | None:
        """Label a confirmed pivot and keep it, unless the noise filter drops it."""
        previous = self._last_high if swing.kind is SwingKind.HIGH else self._last_low

        if self.minimum_move_atr > 0:
            if atr is None:
                return None  # unmeasurable, so not structure (ADR 0008 D4)
            if previous is not None:
                move = abs(float(swing.price) - float(previous.price))
                if move < self.minimum_move_atr * atr:
                    return None

        labelled = dataclasses.replace(swing, label=self._label(swing, previous))
        self._swings.append(labelled)

        if labelled.kind is SwingKind.HIGH:
            self._last_high = labelled
            self._structural_high = labelled
            self._closes_above = 0
        else:
            self._last_low = labelled
            self._structural_low = labelled
            self._closes_below = 0

        return labelled

    def _label(self, swing: SwingPoint, previous: SwingPoint | None) -> StructureLabel | None:
        if previous is None:
            return None

        inclusive = self.equal_prices is EqualPricePolicy.INCLUSIVE
        higher = swing.price > previous.price or (inclusive and swing.price == previous.price)

        if swing.kind is SwingKind.HIGH:
            return StructureLabel.HH if higher else StructureLabel.LH
        return StructureLabel.HL if higher else StructureLabel.LL

    # -- breaks ------------------------------------------------------------

    def _detect_breaks(self, bar: Bar, volume_benchmark: float | None) -> tuple[BreakEvent, ...]:
        events: list[BreakEvent] = []

        if self._structural_high is not None:
            level = self._structural_high
            probe = bar.high if self.breaks_policy.use_wicks else bar.close
            if probe > level.price:
                self._closes_above += 1
                event = self._confirm_break(
                    BreakKind.BREAKOUT, level, bar, self._closes_above, volume_benchmark
                )
                if event is not None:
                    events.append(event)
                    self._structural_high = None
                    self._closes_above = 0
            else:
                self._closes_above = 0

        if self._structural_low is not None:
            level = self._structural_low
            probe = bar.low if self.breaks_policy.use_wicks else bar.close
            if probe < level.price:
                self._closes_below += 1
                event = self._confirm_break(
                    BreakKind.BREAKDOWN, level, bar, self._closes_below, volume_benchmark
                )
                if event is not None:
                    events.append(event)
                    self._structural_low = None
                    self._closes_below = 0
            else:
                self._closes_below = 0

        self._breaks.extend(events)
        return tuple(events)

    def _confirm_break(
        self,
        kind: BreakKind,
        level: SwingPoint,
        bar: Bar,
        closes: int,
        volume_benchmark: float | None,
    ) -> BreakEvent | None:
        if closes < self.breaks_policy.closes_required:
            return None

        volume_confirmed = False
        multiple = self.breaks_policy.volume_multiple
        if multiple is not None:
            if volume_benchmark is None:
                return None  # unverifiable is not satisfied (ADR 0008 D7)
            if float(bar.volume) < multiple * volume_benchmark:
                return None
            volume_confirmed = True

        return BreakEvent(
            kind=kind,
            level_price=level.price,
            level_bar_start_utc=level.bar_start_utc,
            bar_start_utc=bar.start_time_utc,
            closes=closes,
            volume_confirmed=volume_confirmed,
        )


def _extreme(bar: Bar, kind: SwingKind) -> Decimal:
    return bar.high if kind is SwingKind.HIGH else bar.low
