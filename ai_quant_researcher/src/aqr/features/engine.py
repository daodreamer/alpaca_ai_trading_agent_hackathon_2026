"""Builds and caches the feature arrays a strategy asks for.

The engine is deliberately lazy and memoised: a walk-forward run evaluates the
same ``ema(200)`` across dozens of folds, and recomputing it each time is the
difference between a research loop that finishes and one that does not.

Determinism is the contract. ``FeatureFrame`` for a given ``(Bars, feature key)``
is a pure function -- there is no clock, no configuration and no randomness in
this layer.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np

from aqr.data.bars import Array, Bars
from aqr.features.cross_section import CrossSection
from aqr.features.registry import resolve

__all__ = ["FeatureFrame", "FeatureKey"]


@dataclass(frozen=True, slots=True)
class FeatureKey:
    """A feature call, e.g. ``ema(200)`` -> ``FeatureKey("ema", (200.0,))``."""

    name: str
    args: tuple[float, ...] = ()

    def __str__(self) -> str:
        if not self.args:
            return f"{self.name}()"
        rendered = ", ".join(
            str(int(a)) if float(a).is_integer() else str(a) for a in self.args
        )
        return f"{self.name}({rendered})"


@dataclass(slots=True)
class FeatureFrame:
    """Column store over one symbol's bars, keyed by :class:`FeatureKey`.

    ``cross_section`` carries the rest of the universe, for the handful of
    features that compare a symbol to its peers. It is shared across every
    symbol's frame: building it per symbol would be quadratic in the universe
    size, and a fifty-name research loop that rebuilds it fifty times per
    feature is one that does not finish.
    """

    bars: Bars
    cross_section: CrossSection | None = None
    _cache: dict[FeatureKey, Array] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.bars)

    def get(self, key: FeatureKey) -> Array:
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        spec = resolve(key.name)
        if len(key.args) > spec.arity:
            raise ValueError(
                f"{key}: {spec.name} takes at most {spec.arity} argument(s), got {len(key.args)}"
            )
        if spec.build_cross is not None:
            if self.cross_section is None:
                # Loudly, not as NaN. A silent NaN makes the strategy fire on
                # nothing and get rejected for "never fires" -- a true
                # statement about the wrong problem, and one that would send
                # the next person hunting for a bad threshold.
                raise ValueError(
                    f"{key}: {spec.name} is cross-sectional and needs the whole "
                    "universe, but this frame was built for one symbol alone"
                )
            values = np.asarray(
                spec.build_cross(self.cross_section, self.bars.symbol, key.args),
                dtype=np.float64,
            )
        else:
            assert spec.build is not None  # every non-cross feature has one
            values = np.asarray(spec.build(self.bars, key.args), dtype=np.float64)
        if values.size != len(self.bars):
            raise ValueError(
                f"{key}: feature produced {values.size} rows for {len(self.bars)} bars"
            )
        values.flags.writeable = False
        self._cache[key] = values
        return values

    def warmup(self, keys: Iterable[FeatureKey]) -> int:
        """Bars of history required before *every* requested feature is valid.

        Two sources are combined: the registry's declared warm-up, and the
        observed leading NaN run. The declared value alone is not enough --
        ``adx`` chains two Wilder passes and its true warm-up is longer than the
        obvious formula suggests. Trusting the data avoids a class of subtle
        "strategy trades on a half-warmed indicator" bugs.
        """
        needed = 0
        for key in keys:
            spec = resolve(key.name)
            declared = int(spec.warmup(key.args))
            values = self.get(key)
            finite = np.flatnonzero(~np.isnan(values))
            observed = int(finite[0]) + 1 if finite.size else len(self.bars)
            needed = max(needed, declared, observed)
        return min(needed, len(self.bars))
