"""Level and zone engine — specs/07-levels.md, ADR 0009 and ADR 0025.

Turns evidence — confirmed swings, session extremes, 52-week extremes, moving
averages, and since F2 volume-profile nodes, anchored VWAPs, Fibonacci
retracements and unfilled gaps — into clustered, scored `Level` objects. Pure
functions over a snapshot: the same candidates and configuration always produce
the same levels, identity included, which is what makes a rebuild comparable to
the original.

Strength ranks technical evidence. It is not a buy or sell recommendation
(CLAUDE.md §15), and every component of it is returned alongside the number.

Level *lifecycle* is the other half: `invalidate` is the MVP rule — a close
beyond the zone — and `advance` adds F2's decay, expiry and polarity flip on top
of it, all three off unless configured.

User levels are deliberately absent from all of this: a `UserLevel` is never
generated, clustered, re-priced or invalidated by the engine (ADR 0009 D9).
"""

from alphagate.core.level_engine.engine import (
    advance,
    bars_since_touch,
    build_levels,
    cluster_candidates,
    count_touch_episodes,
    decay,
    flip,
    invalidate,
    level_id,
    score_cluster,
)
from alphagate.core.level_engine.model import (
    DEFAULT_FIBONACCI_RATIOS,
    DEFAULT_SOURCE_QUALITY,
    AdvancedSources,
    LevelCandidate,
    LevelConfig,
    LifecycleConfig,
    ProfileConfig,
    StrengthWeights,
    ZoneWidth,
    ZoneWidthMode,
)
from alphagate.core.level_engine.sources import (
    anchored_vwap,
    anchored_vwap_candidates,
    collect_candidates,
    fibonacci_candidates,
    gap_candidates,
    moving_average_candidate,
    previous_day_candidates,
    previous_week_candidates,
    swing_candidates,
    volume_profile_candidates,
    year_extreme_candidates,
)
from alphagate.core.level_engine.volume_profile import (
    ProfileBin,
    ProfileNode,
    VolumeProfile,
    build_volume_profile,
)

__all__ = [
    "DEFAULT_FIBONACCI_RATIOS",
    "DEFAULT_SOURCE_QUALITY",
    "AdvancedSources",
    "LevelCandidate",
    "LevelConfig",
    "LifecycleConfig",
    "ProfileBin",
    "ProfileConfig",
    "ProfileNode",
    "StrengthWeights",
    "VolumeProfile",
    "ZoneWidth",
    "ZoneWidthMode",
    "advance",
    "anchored_vwap",
    "anchored_vwap_candidates",
    "bars_since_touch",
    "build_levels",
    "build_volume_profile",
    "cluster_candidates",
    "collect_candidates",
    "count_touch_episodes",
    "decay",
    "fibonacci_candidates",
    "flip",
    "gap_candidates",
    "invalidate",
    "level_id",
    "moving_average_candidate",
    "previous_day_candidates",
    "previous_week_candidates",
    "score_cluster",
    "swing_candidates",
    "volume_profile_candidates",
    "year_extreme_candidates",
]
