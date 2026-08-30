"""Resample a bar series to a coarser timeframe.

Alpaca has no native 4-hour bar. Aggregating from 1h keeps the same CSV schema
and point-in-time semantics as every other provider in this project.
"""

from __future__ import annotations

import numpy as np

from aqr.data.bars import Bars, bar_duration

__all__ = ["aggregate_bars"]


def aggregate_bars(bars: Bars, *, target_timeframe: str, factor: int) -> Bars:
    """Roll ``factor`` consecutive source bars into one target bar.

    A gap larger than 1.5× the source bar duration starts a new group, so
    overnight and weekend breaks do not leak across sessions.

    A session rarely divides evenly: US regular hours give six 1h bars, and
    four does not go into six. The remainder is emitted as a short final bar
    rather than discarded, because discarding it would drop the last two hours
    of every session -- the close included -- and leave the target bar's
    ``close`` holding a midday price under a session-close name.
    """
    if factor < 2:
        raise ValueError(f"factor must be >= 2, got {factor}")
    if len(bars) == 0:
        return bars.slice(0, 0)

    source_step = int(bar_duration(bars.timeframe).total_seconds())
    max_gap = int(source_step * 1.5)

    # Group boundaries only: every bar belongs to exactly one group, so the
    # starts alone describe the partition and the groups stay contiguous.
    starts: list[int] = [0]
    count = 1
    for i in range(1, len(bars)):
        gap = int(bars.event_time[i] - bars.event_time[i - 1])
        if gap > max_gap or count == factor:
            starts.append(i)
            count = 0
        count += 1

    begin = np.array(starts, dtype=np.int64)
    # The bar before the next group starts; the last group ends at the end.
    end = np.empty_like(begin)
    end[:-1] = begin[1:] - 1
    end[-1] = len(bars) - 1

    return Bars(
        symbol=bars.symbol,
        timeframe=target_timeframe,
        event_time=bars.event_time[begin],
        available_time=bars.available_time[end],
        open=bars.open[begin],
        # reduceat folds each [begin[i], begin[i+1]) slice, which is exactly
        # the group, and handles the ragged final group without padding.
        high=np.maximum.reduceat(bars.high, begin),
        low=np.minimum.reduceat(bars.low, begin),
        close=bars.close[end],
        volume=np.add.reduceat(bars.volume, begin),
    )
