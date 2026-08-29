"""Level-engine inputs and configuration — specs/07-levels.md, ADR 0009.

Everything that decides where a level sits or how strong it is lives here, as
configuration. `specs/07-levels.md` is explicit that weights must be
configuration rather than hidden constants, and the same reasoning applies to
zone widths, merge distances and per-source quality: a number that changes what
the system says about a market should be visible in one place and recorded with
the output.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from datetime import datetime
from decimal import Decimal
from enum import Enum
from types import MappingProxyType
from typing import Self

from alphagate.core.errors import InvariantViolation
from alphagate.core.levels import LevelKind, LevelSource
from alphagate.core.numeric import DOMAIN_CONTEXT, ExactInput
from alphagate.core.numeric import price as exact_price
from alphagate.core.time_model import ensure_utc

__all__ = [
    "DEFAULT_FIBONACCI_RATIOS",
    "DEFAULT_SOURCE_QUALITY",
    "AdvancedSources",
    "LevelCandidate",
    "LevelConfig",
    "LifecycleConfig",
    "ProfileConfig",
    "StrengthWeights",
    "ZoneWidth",
    "ZoneWidthMode",
]


class ZoneWidthMode(Enum):
    ABSOLUTE = "ABSOLUTE"
    PERCENT = "PERCENT"
    ATR = "ATR"


@dataclasses.dataclass(frozen=True, slots=True)
class ZoneWidth:
    """A price distance, expressed in one of three ways (ADR 0009 D3).

    Used both for a zone's half-width and for clustering distances, so
    "cluster within 0.5 ATR" and "cluster within 0.25%" are equally sayable.
    """

    mode: ZoneWidthMode
    value: Decimal

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", exact_price(self.value, field="zone width"))
        if self.value < 0:
            raise InvariantViolation(f"zone width: must not be negative, got {self.value}")

    @classmethod
    def absolute(cls, value: ExactInput) -> Self:
        return cls(ZoneWidthMode.ABSOLUTE, exact_price(value, field="zone width"))

    @classmethod
    def percent(cls, value: ExactInput) -> Self:
        return cls(ZoneWidthMode.PERCENT, exact_price(value, field="zone width"))

    @classmethod
    def atr(cls, multiple: ExactInput) -> Self:
        return cls(ZoneWidthMode.ATR, exact_price(multiple, field="zone width"))

    def resolve(self, *, price: Decimal, atr: Decimal | None) -> Decimal | None:
        """The distance in price terms, or `None` when it cannot be known.

        `None` happens only in ATR mode before ATR is warm. The caller must not
        substitute a made-up width: a level with no zone is an exact level, and
        that is an honest answer (ADR 0009 D3).
        """
        match self.mode:
            case ZoneWidthMode.ABSOLUTE:
                return self.value
            case ZoneWidthMode.PERCENT:
                return exact_price(
                    DOMAIN_CONTEXT.divide(DOMAIN_CONTEXT.multiply(price, self.value), Decimal(100)),
                    field="zone width",
                )
            case ZoneWidthMode.ATR:
                if atr is None:
                    return None
                return exact_price(DOMAIN_CONTEXT.multiply(atr, self.value), field="zone width")

    @property
    def label(self) -> str:
        return f"{self.mode.value}:{self.value.normalize()}"


@dataclasses.dataclass(frozen=True, slots=True)
class LevelCandidate:
    """One piece of evidence at a price, before clustering.

    `observed_at_utc` is the bar the evidence is dated to — the pivot bar of a
    swing, the session whose high it was. It is what recency is measured from,
    and it is deliberately not "now".
    """

    price: Decimal
    kind: LevelKind
    source: LevelSource
    observed_at_utc: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "price", exact_price(self.price, field="price"))
        if self.price <= 0:
            raise InvariantViolation(f"price: must be positive, got {self.price}")
        object.__setattr__(
            self, "observed_at_utc", ensure_utc(self.observed_at_utc, field="observed_at_utc")
        )

    @property
    def sort_key(self) -> tuple[Decimal, str, datetime]:
        """Total order for clustering — no tie can fall back on input order."""
        return (self.price, self.source.value, self.observed_at_utc)


@dataclasses.dataclass(frozen=True, slots=True)
class StrengthWeights:
    """Weights for the strength components (ADR 0009 D6).

    `volume_context` and `multi_timeframe` are declared with weight 0: the data
    they need (volume-at-price, multi-timeframe confluence) is Full-version
    work, and keeping them in the table makes enabling them a configuration
    change rather than a schema change.
    """

    source_quality: float = 1.0
    touches: float = 1.0
    recency: float = 1.0
    confluence: float = 1.0
    volume_context: float = 0.0
    multi_timeframe: float = 0.0

    def __post_init__(self) -> None:
        for field in dataclasses.fields(self):
            weight = getattr(self, field.name)
            if weight < 0:
                raise InvariantViolation(f"{field.name}: weight must not be negative, got {weight}")
        if self.total <= 0:
            raise InvariantViolation("at least one strength weight must be positive")

    @property
    def total(self) -> float:
        return math.fsum(getattr(self, field.name) for field in dataclasses.fields(self))

    def as_mapping(self) -> Mapping[str, float]:
        return MappingProxyType(
            {field.name: getattr(self, field.name) for field in dataclasses.fields(self)}
        )


DEFAULT_SOURCE_QUALITY: Mapping[LevelSource, float] = MappingProxyType(
    {
        # Quality is "how much does this kind of evidence tell us", in [0, 1].
        # A 52-week extreme is watched by everyone; a moving average is a moving
        # target. These are defaults, not truths — they are configuration.
        LevelSource.YEAR_HIGH: 0.9,
        LevelSource.YEAR_LOW: 0.9,
        LevelSource.SWING_HIGH: 0.8,
        LevelSource.SWING_LOW: 0.8,
        LevelSource.PREV_WEEK_HIGH: 0.7,
        LevelSource.PREV_WEEK_LOW: 0.7,
        LevelSource.PREV_DAY_HIGH: 0.6,
        LevelSource.PREV_DAY_LOW: 0.6,
        LevelSource.MOVING_AVERAGE: 0.5,
        LevelSource.USER: 1.0,
        # F2 (ADR 0025 D7). The point of control is where the most business was
        # actually done, so it ranks with the swings; a Fibonacci retracement is
        # geometry with no trade behind it and ranks lowest of the lot.
        LevelSource.VOLUME_POC: 0.8,
        LevelSource.VOLUME_HVN: 0.7,
        LevelSource.VOLUME_LVN: 0.5,
        LevelSource.ANCHORED_VWAP: 0.6,
        LevelSource.GAP: 0.6,
        LevelSource.FIBONACCI: 0.4,
    }
)

DEFAULT_FIBONACCI_RATIOS: tuple[str, ...] = ("0.382", "0.5", "0.618", "0.786")
"""The retracements traders actually draw (ADR 0025 D3).

0.5 is not a Fibonacci ratio at all and is in the list anyway, because it is
watched; 0.236 is left out because it sits so close to the leg's end that it
rarely survives clustering as anything but noise."""


@dataclasses.dataclass(frozen=True, slots=True)
class ProfileConfig:
    """How a volume profile is binned and what counts as a node (ADR 0025 D1).

    `bin_width` is resolved once, against the last close of the window, so the
    whole profile shares one grid. The grid itself is anchored to multiples of
    that width rather than to the window's lowest bar, which is what keeps a bin
    boundary from moving — and every node with it — because one more bar arrived.
    """

    bin_width: ZoneWidth = dataclasses.field(default_factory=lambda: ZoneWidth.percent("0.25"))
    value_area: float = 0.70
    """Share of total volume the value area covers. 70% is the convention."""

    high_node_ratio: float = 0.70
    """A bin is part of a high-volume node at this share of the POC's volume."""

    low_node_ratio: float = 0.30
    """…and part of a low-volume node at or below this share."""

    max_bins: int = 400
    """Cap on the grid. Exceeding it widens the bins by an integer factor, which
    keeps the grid aligned rather than re-anchoring it."""

    def __post_init__(self) -> None:
        if not 0.0 < self.value_area <= 1.0:
            raise InvariantViolation(f"value_area: must be within (0, 1], got {self.value_area}")
        if not 0.0 < self.high_node_ratio <= 1.0:
            raise InvariantViolation(
                f"high_node_ratio: must be within (0, 1], got {self.high_node_ratio}"
            )
        if not 0.0 <= self.low_node_ratio < self.high_node_ratio:
            raise InvariantViolation(
                f"low_node_ratio ({self.low_node_ratio}) must be within "
                f"[0, high_node_ratio) — a bin cannot be both kinds of node"
            )
        if self.max_bins < 1:
            raise InvariantViolation(f"max_bins: must be at least 1, got {self.max_bins}")

    @property
    def label(self) -> str:
        return (
            f"bin={self.bin_width.label},va={self.value_area},"
            f"hvn={self.high_node_ratio},lvn={self.low_node_ratio},max={self.max_bins}"
        )


@dataclasses.dataclass(frozen=True, slots=True)
class AdvancedSources:
    """Which F2 sources contribute candidates (ADR 0025 D6).

    On by default, one flag each: the point of F2 is that these levels exist, and
    a feature that ships switched off has not shipped. Each flag is what turns one
    back off for a series where it says nothing.
    """

    volume_profile: bool = True
    anchored_vwap: bool = True
    fibonacci: bool = True
    gaps: bool = True

    fibonacci_ratios: tuple[Decimal, ...] = dataclasses.field(
        default_factory=lambda: tuple(Decimal(ratio) for ratio in DEFAULT_FIBONACCI_RATIOS)
    )
    minimum_gap: ZoneWidth | None = dataclasses.field(
        default_factory=lambda: ZoneWidth.percent("0.1")
    )
    """Gaps narrower than this are not levels. Without a floor every one-tick
    hole between two minute bars becomes a zone, which is how a chart stops
    being readable."""

    vwap_anchors: int = 1
    """How many recent confirmed swings of each direction to anchor a VWAP at."""

    def __post_init__(self) -> None:
        ratios = tuple(
            exact_price(ratio, field="fibonacci ratio") for ratio in self.fibonacci_ratios
        )
        for ratio in ratios:
            if not Decimal(0) < ratio < Decimal(1):
                raise InvariantViolation(
                    f"fibonacci ratio: must be within (0, 1), got {ratio} — a retracement "
                    "beyond its own leg is an extension, which this is not"
                )
        object.__setattr__(self, "fibonacci_ratios", tuple(sorted(set(ratios))))
        if self.vwap_anchors < 0:
            raise InvariantViolation(f"vwap_anchors: must not be negative, got {self.vwap_anchors}")

    @property
    def label(self) -> str:
        enabled = ",".join(
            name
            for name, flag in (
                ("profile", self.volume_profile),
                ("avwap", self.anchored_vwap),
                ("fib", self.fibonacci),
                ("gaps", self.gaps),
            )
            if flag
        )
        ratios = "/".join(format(ratio.normalize(), "f") for ratio in self.fibonacci_ratios)
        gap = self.minimum_gap.label if self.minimum_gap is not None else "none"
        return f"{enabled}|fib={ratios}|gap>={gap}|anchors={self.vwap_anchors}"


@dataclasses.dataclass(frozen=True, slots=True)
class LifecycleConfig:
    """What happens to a level that is standing but no longer being tested.

    All three are off by default, for the reason `specs/07-levels.md` gives about
    recency decay: a policy that is on by default is a hidden constant. MVP
    invalidation — a close beyond the zone — is not configurable and is not here;
    it always applies (ADR 0009 D8).
    """

    untouched_decay: float = 1.0
    """Per-bar decay of a standing level's strength, in (0, 1]. 1.0 is off."""

    expire_after_untouched_bars: int | None = None
    """Bars a level may go untested before it is invalidated. `None` never."""

    polarity_flip: bool = False
    """Whether a level broken by a close comes back as its opposite."""

    def __post_init__(self) -> None:
        if not 0.0 < self.untouched_decay <= 1.0:
            raise InvariantViolation(
                f"untouched_decay: must be within (0, 1], got {self.untouched_decay}"
            )
        if self.expire_after_untouched_bars is not None and self.expire_after_untouched_bars < 1:
            raise InvariantViolation(
                "expire_after_untouched_bars: must be at least 1, got "
                f"{self.expire_after_untouched_bars}"
            )

    @property
    def label(self) -> str:
        return (
            f"decay={self.untouched_decay},expire={self.expire_after_untouched_bars},"
            f"flip={self.polarity_flip}"
        )


@dataclasses.dataclass(frozen=True, slots=True)
class LevelConfig:
    """Everything the level engine is allowed to decide with."""

    zone_width: ZoneWidth = dataclasses.field(default_factory=lambda: ZoneWidth.atr("0.25"))
    merge_distance: ZoneWidth = dataclasses.field(default_factory=lambda: ZoneWidth.atr("0.5"))
    max_cluster_width: ZoneWidth = dataclasses.field(default_factory=lambda: ZoneWidth.atr("1.5"))
    recency_decay: float = 1.0
    """Per-bar decay of a candidate's weight, in (0, 1]. Default 1.0 — off."""

    weights: StrengthWeights = dataclasses.field(default_factory=StrengthWeights)
    source_quality: Mapping[LevelSource, float] = DEFAULT_SOURCE_QUALITY
    touch_saturation: int = 3
    confluence_saturation: int = 3

    profile: ProfileConfig = dataclasses.field(default_factory=ProfileConfig)
    advanced: AdvancedSources = dataclasses.field(default_factory=AdvancedSources)
    lifecycle: LifecycleConfig = dataclasses.field(default_factory=LifecycleConfig)

    def __post_init__(self) -> None:
        if not 0.0 < self.recency_decay <= 1.0:
            raise InvariantViolation(
                f"recency_decay: must be within (0, 1], got {self.recency_decay}"
            )
        if self.touch_saturation < 1:
            raise InvariantViolation(
                f"touch_saturation: must be at least 1, got {self.touch_saturation}"
            )
        if self.confluence_saturation < 1:
            raise InvariantViolation(
                f"confluence_saturation: must be at least 1, got {self.confluence_saturation}"
            )
        for source, quality in self.source_quality.items():
            if not 0.0 <= quality <= 1.0:
                raise InvariantViolation(
                    f"source_quality[{source.value}]: must be within [0, 1], got {quality}"
                )

    def quality_of(self, source: LevelSource) -> float:
        """Configured quality, or the neutral 0.5 for a source with no entry."""
        return self.source_quality.get(source, 0.5)
