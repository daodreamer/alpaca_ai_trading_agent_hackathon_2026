"""Volume at price — specs/07-levels.md "Full version", ADR 0025 D1/D2.

A volume profile answers a question OHLC alone cannot: not where price *went*,
but where it did business. The point of control, the value area and the nodes
either side of it are the levels that come out of that, and `volume_context` —
the strength component `specs/07-levels.md` has carried at weight 0 since the
MVP — is finally computable from it.

Two things decide whether this is trustworthy:

* **the grid**, anchored to multiples of the bin width rather than to the
  window's lowest bar, so a bin boundary — and every node sitting on one — does
  not move because one more bar arrived;
* **the distribution**, uniform across each bar's range. Where inside a bar the
  volume actually traded is not knowable from the bar; spreading it evenly is
  the honest reading of that ignorance, and it is stated here rather than
  disguised. Tick or trade-level data would replace this function, not correct
  it.

Everything is exact-domain arithmetic and the allocation conserves total volume
to the cent: the residual left by rounding is assigned to the bin with the
largest overlap, so `sum(bin.volume) == sum(bar.volume)` exactly.
"""

from __future__ import annotations

import dataclasses
import itertools
from collections.abc import Callable, Iterable, Sequence
from decimal import Decimal

from alphagate.core.bar import Bar
from alphagate.core.errors import InvariantViolation
from alphagate.core.level_engine.model import ProfileConfig
from alphagate.core.levels import Zone
from alphagate.core.numeric import DOMAIN_CONTEXT, from_approximate
from alphagate.core.numeric import price as exact_price
from alphagate.core.numeric import quantity as exact_quantity

__all__ = [
    "ProfileBin",
    "ProfileNode",
    "VolumeProfile",
    "build_volume_profile",
]

_TWO = Decimal(2)


@dataclasses.dataclass(frozen=True, slots=True)
class ProfileBin:
    """One cell of the grid, `[low, high)`, and what traded in it."""

    index: int
    low: Decimal
    high: Decimal
    volume: Decimal

    @property
    def midpoint(self) -> Decimal:
        return exact_price(DOMAIN_CONTEXT.divide(self.low + self.high, _TWO), field="bin midpoint")


@dataclasses.dataclass(frozen=True, slots=True)
class ProfileNode:
    """A named place in the distribution — a POC, a high node, a low node.

    A node is a *run* of bins, not a single one: a high-volume shelf three bins
    wide is one node, and reporting three would be reporting the grid rather than
    the market (ADR 0025 D2).
    """

    price: Decimal
    low: Decimal
    high: Decimal
    volume: Decimal

    @property
    def zone(self) -> Zone:
        return Zone(low=self.low, high=self.high)


@dataclasses.dataclass(frozen=True, slots=True)
class VolumeProfile:
    """Volume by price over one window of bars.

    Contiguous bins in ascending price order, covering the whole traded range —
    including the empty cells inside it, which are exactly the low-volume nodes.
    """

    bins: tuple[ProfileBin, ...]
    bin_width: Decimal
    total_volume: Decimal
    poc: ProfileNode
    value_area: Zone
    high_nodes: tuple[ProfileNode, ...] = ()
    low_nodes: tuple[ProfileNode, ...] = ()

    def __post_init__(self) -> None:
        if not self.bins:
            raise InvariantViolation("a volume profile needs at least one bin")
        for previous, current in itertools.pairwise(self.bins):
            if current.index != previous.index + 1:
                raise InvariantViolation(
                    f"profile bins must be contiguous; {previous.index} is followed by "
                    f"{current.index}"
                )

    @property
    def low(self) -> Decimal:
        return self.bins[0].low

    @property
    def high(self) -> Decimal:
        return self.bins[-1].high

    def volume_in(self, low: Decimal, high: Decimal) -> Decimal:
        """Volume traded within `[low, high]`, bins counted in proportion.

        A window covering half a bin is credited half that bin's volume — the
        same uniform reading used to fill the bins in the first place.
        """
        if high <= low:
            return exact_quantity(0)
        total = Decimal(0)
        for cell in self.bins:
            overlap = min(high, cell.high) - max(low, cell.low)
            if overlap <= 0 or cell.volume <= 0:
                continue
            width = cell.high - cell.low
            if overlap >= width:
                total += cell.volume
            else:
                total += DOMAIN_CONTEXT.divide(DOMAIN_CONTEXT.multiply(cell.volume, overlap), width)
        return exact_quantity(total, field="window volume")

    def context(self, zone: Zone) -> float | None:
        """How busy this zone is, against the busiest band of the same width.

        In [0, 1], and `None` when the window traded nothing — an undefined
        component rather than a zero one, which is the distinction `confidence`
        exists to keep (ADR 0009 D7).

        Measured over a band at least one bin wide, so an exact level with no
        zone is still asked a meaningful question instead of being handed the
        zero that a zero-width window would return.
        """
        if self.total_volume <= 0:
            return None

        width = max(zone.width, self.bin_width)
        half = DOMAIN_CONTEXT.divide(width, _TWO)
        centre = zone.midpoint
        inside = self.volume_in(centre - half, centre + half)

        best = max(self.volume_in(cell.low, cell.low + width) for cell in self.bins)
        if best <= 0:  # pragma: no cover — total_volume > 0 puts volume in some bin
            return None
        return min(1.0, float(DOMAIN_CONTEXT.divide(inside, best)))


def build_volume_profile(
    bars: Sequence[Bar],
    *,
    config: ProfileConfig | None = None,
    atr: Decimal | None = None,
) -> VolumeProfile | None:
    """Bin a window of finalized bars by price.

    `None` — never a fabricated profile — when there is nothing to bin: no
    finalized bars, no volume in them, or a bin width that cannot be resolved
    because ATR is still warming. That is the same refusal `ZoneWidth.resolve`
    makes for zone widths (ADR 0009 D3).
    """
    settings = config if config is not None else ProfileConfig()
    final = [bar for bar in bars if bar.is_final and bar.volume > 0]
    if not final:
        return None

    reference = final[-1].close
    width = settings.bin_width.resolve(price=reference, atr=atr)
    if width is None or width <= 0:
        return None

    width = _fit_grid(final, width=width, max_bins=settings.max_bins)
    tally = _tally(final, width=width)
    if not tally:  # pragma: no cover — every bar above carries volume
        return None

    bins = tuple(
        ProfileBin(
            index=index,
            low=exact_price(DOMAIN_CONTEXT.multiply(width, index), field="bin low"),
            high=exact_price(DOMAIN_CONTEXT.multiply(width, index + 1), field="bin high"),
            volume=exact_quantity(tally.get(index, Decimal(0)), field="bin volume"),
        )
        for index in range(min(tally), max(tally) + 1)
    )

    total = exact_quantity(sum((cell.volume for cell in bins), Decimal(0)), field="total volume")
    poc_at = _poc_index(bins)
    low_edge, high_edge = _value_area(bins, poc_at=poc_at, total=total, share=settings.value_area)

    high_nodes, low_nodes = _nodes(bins, poc_at=poc_at, config=settings)
    return VolumeProfile(
        bins=bins,
        bin_width=width,
        total_volume=total,
        poc=ProfileNode(
            price=bins[poc_at].midpoint,
            low=bins[poc_at].low,
            high=bins[poc_at].high,
            volume=bins[poc_at].volume,
        ),
        value_area=Zone(low=bins[low_edge].low, high=bins[high_edge].high),
        high_nodes=high_nodes,
        low_nodes=low_nodes,
    )


# -- the grid ---------------------------------------------------------------


def _bin_index(value: Decimal, *, width: Decimal) -> int:
    """Which cell of the global grid `value` falls in.

    Anchored at zero, so the grid is a property of the width alone and not of
    whichever bars happen to be in the window (ADR 0025 D1).
    """
    return int(DOMAIN_CONTEXT.divide_int(value, width))


def _fit_grid(bars: Sequence[Bar], *, width: Decimal, max_bins: int) -> Decimal:
    """Widen the bins by a whole factor until the grid fits inside `max_bins`.

    A whole factor, because every multiple of `k * width` is also a multiple of
    `width`: the grid coarsens without re-anchoring, so the boundaries that
    survive are boundaries the finer grid also had.
    """
    lowest = min(bar.low for bar in bars)
    highest = max(bar.high for bar in bars)
    needed = _bin_index(highest, width=width) - _bin_index(lowest, width=width) + 1
    if needed <= max_bins:
        return width
    factor = -(-needed // max_bins)  # ceiling division
    return exact_price(DOMAIN_CONTEXT.multiply(width, factor), field="bin width")


def _tally(bars: Iterable[Bar], *, width: Decimal) -> dict[int, Decimal]:
    tally: dict[int, Decimal] = {}
    for bar in bars:
        for index, volume in _spread(bar, width=width):
            tally[index] = tally.get(index, Decimal(0)) + volume
    return tally


def _spread(bar: Bar, *, width: Decimal) -> tuple[tuple[int, Decimal], ...]:
    """One bar's volume, distributed uniformly across the bins it spans.

    The rounding residual goes to the bin the bar overlapped most — lowest index
    on a tie — so the profile's total is the bars' total exactly, rather than
    theirs minus a few thousandths of a share per bar.
    """
    if bar.high == bar.low:
        return ((_bin_index(bar.low, width=width), bar.volume),)

    span = bar.high - bar.low
    portions: list[tuple[int, Decimal, Decimal]] = []
    for index in range(_bin_index(bar.low, width=width), _bin_index(bar.high, width=width) + 1):
        low = DOMAIN_CONTEXT.multiply(width, index)
        overlap = min(bar.high, low + width) - max(bar.low, low)
        if overlap <= 0:
            continue
        share = exact_quantity(
            DOMAIN_CONTEXT.divide(DOMAIN_CONTEXT.multiply(bar.volume, overlap), span),
            field="binned volume",
        )
        portions.append((index, overlap, share))

    residual = bar.volume - sum((share for _, _, share in portions), Decimal(0))
    fullest = min(portions, key=lambda portion: (-portion[1], portion[0]))[0]
    return tuple(
        (index, share + residual if index == fullest else share) for index, _, share in portions
    )


# -- reading the distribution -----------------------------------------------


def _poc_index(bins: Sequence[ProfileBin]) -> int:
    """The heaviest bin; the lowest of them when several tie."""
    return min(range(len(bins)), key=lambda index: (-bins[index].volume, index))


def _value_area(
    bins: Sequence[ProfileBin], *, poc_at: int, total: Decimal, share: float
) -> tuple[int, int]:
    """Grow outward from the POC until `share` of the volume is covered.

    Each step takes the heavier of the two adjacent bins, and the lower one when
    they are equal — a tie-break rather than a preference, but it has to be
    written down or the answer depends on how the comparison happens to fall.
    """
    target = DOMAIN_CONTEXT.multiply(total, from_approximate(share, field="value_area"))
    low = high = poc_at
    covered = bins[poc_at].volume

    while covered < target and (low > 0 or high < len(bins) - 1):
        below = bins[low - 1].volume if low > 0 else None
        above = bins[high + 1].volume if high < len(bins) - 1 else None
        if above is None or (below is not None and below >= above):
            low -= 1
            covered += bins[low].volume
        else:
            high += 1
            covered += bins[high].volume
    return low, high


def _nodes(
    bins: Sequence[ProfileBin], *, poc_at: int, config: ProfileConfig
) -> tuple[tuple[ProfileNode, ...], tuple[ProfileNode, ...]]:
    """High- and low-volume nodes, as runs of bins (ADR 0025 D2)."""
    peak = bins[poc_at].volume
    high_floor = DOMAIN_CONTEXT.multiply(
        peak, from_approximate(config.high_node_ratio, field="high_node_ratio")
    )
    low_ceiling = DOMAIN_CONTEXT.multiply(
        peak, from_approximate(config.low_node_ratio, field="low_node_ratio")
    )

    high_runs = [
        run
        for run in _runs(bins, lambda cell: cell.volume >= high_floor)
        if not run[0] <= poc_at <= run[1]  # the POC has its own source
    ]
    # A low-volume run at either end of the profile is the tail of the
    # distribution, not a shelf the market crossed quickly. Only interior runs
    # are nodes.
    low_runs = [
        run
        for run in _runs(bins, lambda cell: cell.volume <= low_ceiling)
        if run[0] > 0 and run[1] < len(bins) - 1
    ]

    return (
        tuple(_peak_node(bins, run) for run in high_runs),
        tuple(_span_node(bins, run) for run in low_runs),
    )


def _runs(
    bins: Sequence[ProfileBin], predicate: Callable[[ProfileBin], bool]
) -> tuple[tuple[int, int], ...]:
    """Maximal runs of consecutive bins satisfying `predicate`, as index pairs."""
    matches = [index for index, cell in enumerate(bins) if predicate(cell)]
    runs: list[tuple[int, int]] = []
    for index in matches:
        if runs and index == runs[-1][1] + 1:
            runs[-1] = (runs[-1][0], index)
        else:
            runs.append((index, index))
    return tuple(runs)


def _peak_node(bins: Sequence[ProfileBin], run: tuple[int, int]) -> ProfileNode:
    """A high node sits at its heaviest bin — the price inside the shelf that
    actually drew the trade, not the shelf's geometric middle."""
    start, end = run
    peak = min(range(start, end + 1), key=lambda index: (-bins[index].volume, index))
    return ProfileNode(
        price=bins[peak].midpoint,
        low=bins[start].low,
        high=bins[end].high,
        volume=exact_quantity(
            sum((bins[index].volume for index in range(start, end + 1)), Decimal(0)),
            field="node volume",
        ),
    )


def _span_node(bins: Sequence[ProfileBin], run: tuple[int, int]) -> ProfileNode:
    """A low node has no peak to sit on, so it sits in the middle of its gap."""
    start, end = run
    low = bins[start].low
    high = bins[end].high
    return ProfileNode(
        price=exact_price(DOMAIN_CONTEXT.divide(low + high, _TWO), field="node price"),
        low=low,
        high=high,
        volume=exact_quantity(
            sum((bins[index].volume for index in range(start, end + 1)), Decimal(0)),
            field="node volume",
        ),
    )
