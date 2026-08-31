"""Option features for the DSL — specs/10-options-research.md D6.

``options/spec.py``'s docstring already says an ``OptionSpec.entry`` is parsed
by the *existing* ``dsl/expr.py`` -- "same tokenizer, same whitelist, same
``feature_keys`` walk... only the feature table changes." That sentence is the
integration mechanism, made literal: ``dsl.expr.parse`` and ``dsl.expr.evaluate``
now take their feature table and their data source as swappable, structurally
typed parameters (``FeatureLookup`` and ``FeatureSource``) instead of hard-wired
references to ``aqr.features.registry`` and ``aqr.features.engine.FeatureFrame``.
Every existing caller passes neither, so equity parsing and evaluation are
byte-for-byte unchanged; this module is the one caller that does.

Two new pieces plug into those seams:

``resolve_entry_feature`` is the feature table. It tries :data:`OPTION_FEATURES`
first, falls back to the bar registry, and raises the same "closest match"
``KeyError`` :func:`aqr.features.registry.resolve` does, but drawn from the
union of both vocabularies. ``OptionSpec.__post_init__`` passes it to
``parse``; nothing else does, so an equity ``StrategySpec`` never sees
``iv_rank`` in its vocabulary.

:class:`OptionFeatureFrame` is the data source. It wraps a bar-only
``FeatureFrame`` and answers ``close`` and ``sma(200)`` by delegating to it
unchanged; it answers ``iv_rank()`` and friends from precomputed,
bar-day-aligned arrays built from ``volatility_history`` and the option chain.
``run_option_backtest`` builds one of these instead of a bare ``FeatureFrame``
and hands it to the unchanged ``evaluate()`` -- the engine does not know or
care which vocabulary a given ``Call`` node belongs to.

**Why a wrapper frame rather than registering option features into
``aqr.features.registry.REGISTRY`` directly:** that registry is also the
vocabulary an equity ``StrategySpec`` is validated against and the list an
equity proposer prompt enumerates (CLAUDE.md §2b: the two projects' vocabularies
stay deliberately unmixed). Registering into it would make ``iv_rank()`` parse
as a *valid but broken* equity feature -- syntactically legal, and an
``AssertionError`` three layers down the first time anything tried to evaluate
it, because a registry ``build`` callable only ever receives ``Bars``, and no
``Bars`` carries an implied-volatility surface. The wrapper keeps the two
tables (and their failure modes) apart, at the cost of the two small seams in
``dsl/expr.py`` this module leans on.

**Alignment and the forward-fill bound (D6).** Every option feature is
resolved onto ``run_option_backtest``'s own grid -- the underlying's daily bar
index -- because that is what ``FeatureFrame`` warms up against and what
``evaluate`` reads at the decision bar (``options/engine.py``). Neither
``volatility_history`` nor the option chain shares that grid: the vendor
samples weekly in 2019, thins to Mon/Wed/Fri after, and 15 rows carry blank
year extremes. So every feature here is built as a *sparse, session-dated*
series first (one entry per session that actually has an answer; a session
that doesn't -- a blank extreme, a delta the ladder didn't offer that day --
contributes nothing rather than a hole to paper over) and then carried forward
onto the bar grid, capped at :data:`MAX_FORWARD_FILL_DAYS` calendar days. Older
than that, the feature is ``NaN``, and ``NaN`` compares ``False`` in every
direction ``dsl.expr.evaluate`` uses (that propagation already exists; nothing
here has to reimplement it).
"""

from __future__ import annotations

import difflib
from bisect import bisect_right
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime

import numpy as np

from aqr.data.bars import Array, Bars
from aqr.dsl.expr import FeatureLookup
from aqr.features.engine import FeatureFrame, FeatureKey
from aqr.features.registry import feature_names as _bar_feature_names
from aqr.features.registry import resolve as _resolve_bar_feature
from aqr.options.chain import ChainIndex, NoSuchContract

__all__ = [
    "MAX_FORWARD_FILL_DAYS",
    "OPTION_FEATURES",
    "OptionFeatureFrame",
    "OptionFeatureSpec",
    "VolatilityHistory",
    "VolatilityRow",
    "resolve_entry_feature",
]

MAX_FORWARD_FILL_DAYS = 5
"""D6's bound. Two vendor gaps make it necessary rather than cosmetic: 15
``volatility_history`` rows carry blank year extremes, and the 2019 era is
weekly, so two adjacent snapshots can sit 13 days apart. An unbounded
as-of fill would answer a rule conditioned on ``iv_rank()`` with a number up
to a fortnight stale and never say so; bounded, the feature is honestly
``NaN`` instead, and ``NaN`` makes every comparison ``dsl.expr.evaluate``
supports come back ``False``."""

_ATM_DELTA_TARGET = 0.50
_ATM_DELTA_TOLERANCE = 0.10
"""Measured on the SPY research cache (753 sessions). ``chain.py``'s own
``DEFAULT_DELTA_TOLERANCE`` of 0.06 -- right for *selecting a trade*, where an
approximation must refuse -- misses the near-the-money call on 85 of 753
sessions at the ~49 DTE bucket alone, because that bucket's ladder is thinner
than the ~28 DTE one it is tuned for. 0.10 leaves 1 miss across all three DTE
targets combined; the rest become ``NaN`` through the same forward-fill path
as a missing vendor row, never a refusal, because this is a descriptive
feature read by a rule's condition, not a leg an order will be built from."""

_SKEW_DELTA_TARGET = 0.25
_SKEW_DELTA_TOLERANCE = 0.10
"""Same measurement, at the 25-delta strike ``skew_25d`` reads: 0.06 misses the
25-delta call on 164 of 753 sessions; 0.10 leaves 2."""

_NO_DTE_LIMIT = 3650
"""Passed as ``SessionChain.select``'s ``dte_tolerance`` so it always resolves
to whichever expiry is nearest the target, with no refusal band. D0 measures
the longest-dated bucket running 43-66 DTE across the cache -- a rule of "must
be within 10 days of 49" would refuse on 171 of 753 sessions before the delta
selection even runs. Trade selection refuses an approximation (``chain.py``);
``atm_iv(49)`` on a session whose longest listed expiry is 66 days out is still
the right number to report for that session, not a reason to produce nothing."""


def _float_or_nan(raw: str) -> float:
    """Vendor blanks (D0: 15 rows for the year extremes, 114 for month-ago on
    this cache) become ``NaN``, never 0.0 -- a blank low read as zero would
    report SPY sitting at the top of a 52-week range that was never measured."""
    raw = raw.strip()
    return float(raw) if raw else float("nan")


def _bar_days(bars: Bars) -> list[date]:
    return [datetime.fromtimestamp(int(t), tz=UTC).date() for t in bars.event_time]


def _valid_series(pairs: Iterable[tuple[date, float]]) -> tuple[list[date], list[float]]:
    """Drop ``NaN`` entries and sort by date -- the shape :func:`_asof_ffill`
    needs.

    A ``NaN`` here means the session's arithmetic refused (a blank vendor
    field, no contract near the named delta): that session said nothing, so
    the fill must look further back for the last session that did, rather than
    freezing on a hole with a value that looks like data.
    """
    kept = sorted((d, v) for d, v in pairs if not np.isnan(v))
    return [d for d, _ in kept], [v for _, v in kept]


def _asof_ffill(
    source_days: Sequence[date], source_values: Sequence[float], bar_days: Sequence[date]
) -> Array:
    """The value most recently observed at or before each bar day, carried
    forward at most :data:`MAX_FORWARD_FILL_DAYS` calendar days.

    ``source_days`` must already be ``NaN``-free and sorted ascending -- every
    caller here builds it through :func:`_valid_series`. Binary search per bar
    rather than a merge join: the bar grid can run 2x the length of the source
    grid (1,677 daily bars against ~750 chain sessions) and this keeps the cost
    at ``O(bars * log(sessions))`` without a hand-rolled two-pointer walk that
    would have to get the tie at "exactly on the bound" right on its own.

    This is also the entire causality argument for every feature in this
    module: bar ``i``'s value is read from ``source_days`` strictly at or
    before ``bar_days[i]``, so a row dated after bar ``i`` can never reach it
    -- asserted directly in ``tests/test_option_features.py``'s
    ``TestNoLookAhead`` by truncating the source and checking nothing at or
    before the cutoff moved.
    """
    out = np.full(len(bar_days), np.nan, dtype=np.float64)
    if not source_days:
        return out
    for i, day in enumerate(bar_days):
        idx = bisect_right(source_days, day) - 1
        if idx < 0:
            continue
        if (day - source_days[idx]).days <= MAX_FORWARD_FILL_DAYS:
            out[i] = source_values[idx]
    return out


# --------------------------------------------------------------------------- #
# volatility_history, indexed by session
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class VolatilityRow:
    """One session of ``volatility_history``, parsed. Blank vendor fields are
    ``NaN`` (see :func:`_float_or_nan`), never coerced to a number."""

    session: date
    iv_current: float
    iv_year_high: float
    iv_year_low: float
    iv_week_ago: float
    iv_month_ago: float
    hv_current: float


@dataclass(frozen=True, slots=True)
class VolatilityHistory:
    """``volatility_history``, indexed by session -- the same shape
    :class:`~aqr.options.chain.ChainIndex` gives the option chain, so the two
    tables are consumed identically by the rest of this module."""

    _by_session: dict[date, VolatilityRow]

    def __len__(self) -> int:
        return len(self._by_session)

    def __contains__(self, session: date) -> bool:
        return session in self._by_session

    def __getitem__(self, session: date) -> VolatilityRow:
        return self._by_session[session]

    @property
    def sessions(self) -> tuple[date, ...]:
        return tuple(self._by_session)

    @property
    def rows(self) -> tuple[VolatilityRow, ...]:
        return tuple(self._by_session.values())

    @classmethod
    def from_rows(
        cls, rows: Iterable[Mapping[str, str]], *, before: date | None = None
    ) -> VolatilityHistory:
        """Parse the vendor's columns into sessions.

        ``before`` drops sessions on or after a boundary, mirroring
        ``ChainIndex.from_rows``: the crude, disk-level half of D3's embargo,
        which still leaves ``run_option_backtest`` to refuse an entry whose
        *settlement* crosses it.
        """
        parsed: dict[date, VolatilityRow] = {}
        for row in rows:
            session = date.fromisoformat(row["date"])
            if before is not None and session >= before:
                continue
            parsed[session] = VolatilityRow(
                session=session,
                iv_current=_float_or_nan(row.get("iv_current", "")),
                iv_year_high=_float_or_nan(row.get("iv_year_high", "")),
                iv_year_low=_float_or_nan(row.get("iv_year_low", "")),
                iv_week_ago=_float_or_nan(row.get("iv_week_ago", "")),
                iv_month_ago=_float_or_nan(row.get("iv_month_ago", "")),
                hv_current=_float_or_nan(row.get("hv_current", "")),
            )
        return cls(_by_session=dict(sorted(parsed.items())))


# --------------------------------------------------------------------------- #
# Per-session series: volatility_history features
# --------------------------------------------------------------------------- #


def _iv_rank_series(vol: VolatilityHistory) -> tuple[list[date], list[float]]:
    """``(iv_current - iv_year_low) / (iv_year_high - iv_year_low) * 100``.

    ``high > low`` strictly: it is false whenever either extreme is ``NaN``
    (any comparison against ``NaN`` is ``False`` in IEEE 754, which is what
    lets this guard double as the blank-field check) and it also excludes the
    degenerate ``high == low`` row a division would turn into ``inf``, not a
    percentile. Matches ``tests/test_option_cache_claims.py``'s independent
    measurement of this exact formula on this exact cache.
    """
    pairs = []
    for row in vol.rows:
        if row.iv_year_high > row.iv_year_low:
            span = row.iv_year_high - row.iv_year_low
            value = (row.iv_current - row.iv_year_low) / span * 100.0
        else:
            value = float("nan")
        pairs.append((row.session, value))
    return _valid_series(pairs)


def _iv_hv_spread_series(vol: VolatilityHistory) -> tuple[list[date], list[float]]:
    return _valid_series((r.session, r.iv_current - r.hv_current) for r in vol.rows)


def _iv_current_series(vol: VolatilityHistory) -> tuple[list[date], list[float]]:
    return _valid_series((r.session, r.iv_current) for r in vol.rows)


def _hv_current_series(vol: VolatilityHistory) -> tuple[list[date], list[float]]:
    return _valid_series((r.session, r.hv_current) for r in vol.rows)


_IV_CHANGE_COLUMNS = {5: "iv_week_ago", 21: "iv_month_ago"}
"""The only two lookbacks the vendor table carries. Not a placeholder for a
general n-day change: there is no ``iv_10_days_ago`` column to read one from,
and interpolating between the two that exist would report a number the vendor
never measured as though it had."""


def _iv_change_series(vol: VolatilityHistory, n: int) -> tuple[list[date], list[float]]:
    column = _IV_CHANGE_COLUMNS.get(n)
    if column is None:
        raise ValueError(
            f"iv_change({n}): the vendor table carries only iv_week_ago (n=5) and "
            f"iv_month_ago (n=21) -- refusing rather than interpolating a number "
            f"the data does not have"
        )
    return _valid_series((r.session, r.iv_current - getattr(r, column)) for r in vol.rows)


# --------------------------------------------------------------------------- #
# Per-session series: option_chain features
# --------------------------------------------------------------------------- #


def _atm_iv_by_session(chain: ChainIndex, dte_target: int) -> dict[date, float]:
    """The IV of the call nearest 0.50 delta, in the expiry nearest
    ``dte_target``. Shared by :data:`OPTION_FEATURES`'s ``atm_iv`` and by
    ``term_slope`` so the two agree, bucket by bucket, on what "nearest"
    means.

    The call side only, not an average of call and put: near the money the two
    should agree by put-call parity, and averaging would hide a session where
    they don't (a data or alignment problem) behind a number that looks fine.
    """
    out: dict[date, float] = {}
    for session in chain.sessions:
        try:
            quote = chain[session].select(
                right="call",
                dte_target=dte_target,
                dte_tolerance=_NO_DTE_LIMIT,
                delta_target=_ATM_DELTA_TARGET,
                delta_tolerance=_ATM_DELTA_TOLERANCE,
            )
        except NoSuchContract:
            continue
        out[session] = quote.iv
    return out


def _term_slope_by_session(chain: ChainIndex) -> dict[date, float]:
    near = _atm_iv_by_session(chain, 14)
    far = _atm_iv_by_session(chain, 49)
    return {session: far[session] - near[session] for session in far if session in near}


def _skew_25d_by_session(chain: ChainIndex, dte_target: int = 28) -> dict[date, float]:
    out: dict[date, float] = {}
    for session in chain.sessions:
        try:
            put = chain[session].select(
                right="put",
                dte_target=dte_target,
                dte_tolerance=_NO_DTE_LIMIT,
                delta_target=_SKEW_DELTA_TARGET,
                delta_tolerance=_SKEW_DELTA_TOLERANCE,
            )
            call = chain[session].select(
                right="call",
                dte_target=dte_target,
                dte_tolerance=_NO_DTE_LIMIT,
                delta_target=_SKEW_DELTA_TARGET,
                delta_tolerance=_SKEW_DELTA_TOLERANCE,
            )
        except NoSuchContract:
            continue
        out[session] = put.iv - call.iv
    return out


# --------------------------------------------------------------------------- #
# Argument parsing -- mirrors aqr.features.registry's `_int`
# --------------------------------------------------------------------------- #


def _int_literal(args: tuple[float, ...], i: int, *, default: int | None, feature: str) -> int:
    if len(args) <= i:
        if default is None:
            raise ValueError(f"{feature}: an argument is required")
        return default
    value = args[i]
    if value != int(value):
        raise ValueError(f"{feature}: expected an integer argument, got {value}")
    return int(value)


# --------------------------------------------------------------------------- #
# The option feature table
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class _OptionData:
    """Everything an option feature's ``raw`` builder needs, beyond the bar
    grid it will eventually be aligned to. Not exported: reached only through
    :class:`OptionFeatureFrame`."""

    volatility: VolatilityHistory | None
    chain: ChainIndex | None


def _need_volatility(data: _OptionData, feature: str) -> VolatilityHistory:
    if data.volatility is None:
        raise ValueError(
            f"{feature}() needs volatility_history data, but this backtest run was "
            f"not given any -- pass volatility= to run_option_backtest (or "
            f"OptionFeatureFrame)"
        )
    return data.volatility


def _need_chain(data: _OptionData, feature: str) -> ChainIndex:
    if data.chain is None:
        raise ValueError(
            f"{feature}() needs the option chain, but this backtest run was not "
            f"given one"
        )
    return data.chain


@dataclass(frozen=True, slots=True)
class OptionFeatureSpec:
    """One entry in :data:`OPTION_FEATURES`. Structurally interchangeable with
    ``aqr.features.registry.FeatureSpec`` wherever only ``.arity`` is read
    (``dsl.expr.FeatureLookup``), without either module importing the other.

    ``raw`` returns the *sparse* session series -- dates and values with every
    ``NaN`` already dropped -- not a bar-aligned array. :meth:`OptionFeatureFrame.get`
    is the one place that calls :func:`_asof_ffill` on the result, so every
    feature here is bounded the same way and there is exactly one place that
    bound could be gotten wrong.
    """

    name: str
    arity: int
    raw: Callable[[_OptionData, tuple[float, ...]], tuple[list[date], list[float]]]
    doc: str


OPTION_FEATURES: dict[str, OptionFeatureSpec] = {}


def _register(
    name: str,
    arity: int,
    raw: Callable[[_OptionData, tuple[float, ...]], tuple[list[date], list[float]]],
    doc: str,
) -> None:
    OPTION_FEATURES[name] = OptionFeatureSpec(name=name, arity=arity, raw=raw, doc=doc)


_register(
    "iv_rank",
    0,
    lambda data, _a: _iv_rank_series(_need_volatility(data, "iv_rank")),
    "iv_rank(): (iv_current - iv_year_low) / (iv_year_high - iv_year_low) * 100, "
    "on a 0..100 scale; the vendor supplies the year extremes",
)
_register(
    "iv_hv_spread",
    0,
    lambda data, _a: _iv_hv_spread_series(_need_volatility(data, "iv_hv_spread")),
    "iv_hv_spread(): iv_current - hv_current, the variance risk premium",
)
_register(
    "iv_current",
    0,
    lambda data, _a: _iv_current_series(_need_volatility(data, "iv_current")),
    "iv_current(): the vendor's current implied volatility level",
)
_register(
    "hv_current",
    0,
    lambda data, _a: _hv_current_series(_need_volatility(data, "hv_current")),
    "hv_current(): the vendor's current realised (historical) volatility level",
)
_register(
    "iv_change",
    1,
    lambda data, a: _iv_change_series(
        _need_volatility(data, "iv_change"), _int_literal(a, 0, default=None, feature="iv_change")
    ),
    "iv_change(n): iv_current minus n days ago; n must be 5 (week) or 21 "
    "(month) -- the only lookbacks the vendor table carries",
)
_register(
    "atm_iv",
    1,
    lambda data, a: _valid_series(
        _atm_iv_by_session(
            _need_chain(data, "atm_iv"), _int_literal(a, 0, default=28, feature="atm_iv")
        ).items()
    ),
    "atm_iv(dte): IV of the call nearest 0.50 delta, in the expiry nearest "
    "dte days out (default 28, the cache's median)",
)
_register(
    "term_slope",
    0,
    lambda data, _a: _valid_series(_term_slope_by_session(_need_chain(data, "term_slope")).items()),
    "term_slope(): atm_iv(49) - atm_iv(14), the term-structure slope",
)
_register(
    "skew_25d",
    0,
    lambda data, _a: _valid_series(_skew_25d_by_session(_need_chain(data, "skew_25d")).items()),
    "skew_25d(): 25-delta put IV minus 25-delta call IV, in the ~28 DTE bucket",
)


def resolve_entry_feature(name: str) -> FeatureLookup:
    """The combined feature table: :data:`OPTION_FEATURES` first, then the bar
    registry. Passed to ``dsl.expr.parse`` as its ``resolve_feature`` (D5:
    "only the feature table changes") so an :class:`~aqr.options.spec.OptionSpec`
    entry can name either vocabulary; nothing else calls this, so nothing else
    gains option features by accident.
    """
    if name in OPTION_FEATURES:
        return OPTION_FEATURES[name]
    try:
        return _resolve_bar_feature(name)
    except KeyError:
        pass
    candidates = set(OPTION_FEATURES) | set(_bar_feature_names())
    close = difflib.get_close_matches(name, candidates, n=3, cutoff=0.6)
    hint = f" Did you mean: {', '.join(close)}?" if close else ""
    raise KeyError(f"unknown feature {name!r}.{hint}")


# --------------------------------------------------------------------------- #
# The evaluation-time frame
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class OptionFeatureFrame:
    """Reads ``close`` and ``iv_rank()`` off one object, for
    ``dsl.expr.evaluate`` (``FeatureSource``).

    A bar-only key is delegated to a wrapped, unmodified
    ``aqr.features.engine.FeatureFrame``. An option key is built from
    :data:`OPTION_FEATURES`, as a sparse session series, then aligned onto the
    same bar grid the wrapped frame uses with :func:`_asof_ffill` -- see the
    module docstring for why alignment and the forward-fill bound live here
    rather than in the registry entries themselves.
    """

    bars: Bars
    volatility: VolatilityHistory | None = None
    chain: ChainIndex | None = None
    _bar_frame: FeatureFrame = field(init=False, repr=False, default=None)  # type: ignore[assignment]
    _bar_days: list[date] = field(init=False, repr=False, default_factory=list)
    _cache: dict[FeatureKey, Array] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._bar_frame = FeatureFrame(self.bars)
        self._bar_days = _bar_days(self.bars)

    def __len__(self) -> int:
        return len(self.bars)

    def get(self, key: FeatureKey) -> Array:
        spec = OPTION_FEATURES.get(key.name)
        if spec is None:
            return self._bar_frame.get(key)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        if len(key.args) > spec.arity:
            raise ValueError(
                f"{key}: {spec.name} takes at most {spec.arity} argument(s), "
                f"got {len(key.args)}"
            )
        days, values = spec.raw(_OptionData(volatility=self.volatility, chain=self.chain), key.args)
        array = _asof_ffill(days, values, self._bar_days)
        array.flags.writeable = False
        self._cache[key] = array
        return array

    def warmup(self, keys: Iterable[FeatureKey]) -> int:
        """Bars of history required before every requested key is valid.

        Mirrors ``FeatureFrame.warmup``'s "observed leading NaN run" for the
        option half: there is no *declared* warm-up for an option feature the
        way ``ema(200)`` declares 200, because how long the leading NaN run is
        depends on when the cache's first usable session lands relative to the
        underlying's first bar, not on an argument in the call. Bar-only keys
        are routed to the wrapped frame's own ``warmup`` rather than counted
        here, because that frame's ``resolve`` cannot look up an option name
        and would raise ``KeyError`` if handed one.
        """
        keys = list(keys)
        bar_keys = [k for k in keys if k.name not in OPTION_FEATURES]
        option_keys = [k for k in keys if k.name in OPTION_FEATURES]
        needed = self._bar_frame.warmup(bar_keys) if bar_keys else 0
        for key in option_keys:
            values = self.get(key)
            finite = np.flatnonzero(~np.isnan(values))
            observed = int(finite[0]) + 1 if finite.size else len(self)
            needed = max(needed, observed)
        return min(needed, len(self))
