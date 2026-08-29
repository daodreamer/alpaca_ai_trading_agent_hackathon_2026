"""Domain layer — pure, deterministic, infrastructure-free.

Nothing in this package may import React-adjacent code, HTTP frameworks, ORM
models, provider SDKs, notification SDKs, or any other `pmm` layer. The rule is
enforced by `tests/test_domain_boundaries.py`, not by convention.

Phases 1-2 deliver the primitives, the ports and the pure data pipeline; Phase 3
adds the indicator engine, Phase 4 market structure and levels. Trend and alerts
arrive in Phases 5-6.

Modules:

    errors         the rules the Domain enforces, as exceptions
    identifiers    ticker and id types
    numeric        the exact (Decimal) numeric domain — ADR 0005
    clock          the Clock port
    time_model     timeframes, sessions, the TradingCalendar port — ADR 0004
    market_data    provider ports and capability reporting — ADR 0002
    symbol         the tradable instrument
    bar            the canonical OHLC bar
    aggregation    session grid and timeframe folding — ADR 0004 D4/D5
    normalization  BarSeries: dedupe, ordering, revisions, gaps
    streaming      BarConsumer: the shared bar-stream discipline
    indicators     the online indicator engine — ADR 0007
    structure      swings, HH/HL/LH/LL, break of structure — ADR 0008
    levels         Zone, Level, UserLevel
    level_engine   candidates, clustering, strength, invalidation — ADR 0009
    trend          TrendPhase ladder and TrendState
    alerts         rules, policies, events
"""

from alphagate.core.aggregation import GridCell, aggregate_bars, session_grid
from alphagate.core.alerts import (
    AlertAcknowledgement,
    AlertCondition,
    AlertEvent,
    AlertEventType,
    AlertPolicy,
    AlertRule,
    AlertScope,
    ConfirmationPolicy,
    Cooldown,
    EmissionStatus,
    NotificationChannel,
    ProximityKind,
    ProximityThreshold,
    RearmMode,
    RearmPolicy,
)
from alphagate.core.bar import AdjustmentMode, Bar, BarKey, Feed
from alphagate.core.clock import Clock
from alphagate.core.errors import (
    CalendarError,
    CalendarHorizonExceeded,
    DomainError,
    InvalidTransition,
    InvariantViolation,
    UnknownExchange,
)
from alphagate.core.identifiers import (
    AlertEventId,
    AlertRuleId,
    LevelId,
    Ticker,
    UserId,
    UserLevelId,
    ticker,
)
from alphagate.core.indicators import (
    Atr,
    Ema,
    IndicatorPoint,
    IndicatorSeries,
    IndicatorSpec,
    Macd,
    MacdValue,
    OnlineIndicator,
    PriceSource,
    Rsi,
    SessionVwap,
    Sma,
    VolumeSma,
    compute_series,
)
from alphagate.core.level_engine import (
    LevelCandidate,
    LevelConfig,
    StrengthWeights,
    ZoneWidth,
    build_levels,
    invalidate,
)
from alphagate.core.levels import (
    Level,
    LevelKind,
    LevelSource,
    LevelStatus,
    Priority,
    UserLevel,
    Zone,
)
from alphagate.core.market_data import (
    BarRange,
    FeedHealth,
    FeedStatus,
    HistoricalBarSource,
    ProviderCapabilities,
    StreamingBarSource,
)
from alphagate.core.normalization import BarSeries, IngestOutcome
from alphagate.core.numeric import (
    PRICE_PLACES,
    format_exact,
    from_approximate,
    price,
    quantity,
)
from alphagate.core.streaming import BarConsumer, ComponentSpec
from alphagate.core.structure import (
    BreakEvent,
    BreakKind,
    BreakPolicy,
    EqualPricePolicy,
    StructureEngine,
    StructureLabel,
    StructureUpdate,
    SwingKind,
    SwingPoint,
    SwingStatus,
)
from alphagate.core.symbol import AssetType, Symbol
from alphagate.core.time_model import (
    SessionKind,
    SessionWindow,
    Timeframe,
    TradingCalendar,
    ensure_utc,
)
from alphagate.core.trend import TrendDirection, TrendPhase, TrendState, is_valid_transition

__all__ = [
    "PRICE_PLACES",
    "AdjustmentMode",
    "AlertAcknowledgement",
    "AlertCondition",
    "AlertEvent",
    "AlertEventId",
    "AlertEventType",
    "AlertPolicy",
    "AlertRule",
    "AlertRuleId",
    "AlertScope",
    "AssetType",
    "Atr",
    "Bar",
    "BarConsumer",
    "BarKey",
    "BarRange",
    "BarSeries",
    "BreakEvent",
    "BreakKind",
    "BreakPolicy",
    "CalendarError",
    "CalendarHorizonExceeded",
    "Clock",
    "ComponentSpec",
    "ConfirmationPolicy",
    "Cooldown",
    "DomainError",
    "Ema",
    "EmissionStatus",
    "EqualPricePolicy",
    "Feed",
    "FeedHealth",
    "FeedStatus",
    "GridCell",
    "HistoricalBarSource",
    "IndicatorPoint",
    "IndicatorSeries",
    "IndicatorSpec",
    "IngestOutcome",
    "InvalidTransition",
    "InvariantViolation",
    "Level",
    "LevelCandidate",
    "LevelConfig",
    "LevelId",
    "LevelKind",
    "LevelSource",
    "LevelStatus",
    "Macd",
    "MacdValue",
    "NotificationChannel",
    "OnlineIndicator",
    "PriceSource",
    "Priority",
    "ProviderCapabilities",
    "ProximityKind",
    "ProximityThreshold",
    "RearmMode",
    "RearmPolicy",
    "Rsi",
    "SessionKind",
    "SessionVwap",
    "SessionWindow",
    "Sma",
    "StreamingBarSource",
    "StrengthWeights",
    "StructureEngine",
    "StructureLabel",
    "StructureUpdate",
    "SwingKind",
    "SwingPoint",
    "SwingStatus",
    "Symbol",
    "Ticker",
    "Timeframe",
    "TradingCalendar",
    "TrendDirection",
    "TrendPhase",
    "TrendState",
    "UnknownExchange",
    "UserId",
    "UserLevel",
    "UserLevelId",
    "VolumeSma",
    "Zone",
    "ZoneWidth",
    "aggregate_bars",
    "build_levels",
    "compute_series",
    "ensure_utc",
    "format_exact",
    "from_approximate",
    "invalidate",
    "is_valid_transition",
    "price",
    "quantity",
    "session_grid",
    "ticker",
]
