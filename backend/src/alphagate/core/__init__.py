"""Domain layer — pure, deterministic, infrastructure-free.

Stdlib only. No HTTP framework, no provider SDK, no model client, no clock read;
`tests/test_boundaries.py` guards 1 and 3 enforce that structurally rather than
by convention.

Extracted from Personal Market Monitor (adr/0001), and pruned on 2026-09-04 to
the modules AlphaGate actually perceives with — see adr/0001 D5. There is no
re-export barrel here on purpose: every consumer imports the module it needs by
name, so what a layer depends on is visible in its own import block rather than
hidden behind a flat namespace.

Modules:

    errors         the rules the domain enforces, as exceptions
    identifiers    ticker and id types
    numeric        the exact (Decimal) numeric domain — ADR 0005
    time_model     timeframes, sessions, the TradingCalendar port — ADR 0004
    bar            the canonical OHLC bar
    streaming      BarConsumer: the shared bar-stream discipline
    indicators     the online indicator engine — ADR 0007
    structure      swings, HH/HL/LH/LL, break of structure — ADR 0008
    levels         Zone, Level and their lifecycle
    level_engine   candidates, clustering, strength, invalidation — ADR 0009
    trend          TrendPhase ladder and TrendState
    trend_engine   the evidence ladder that moves a symbol between phases
    confluence     what agrees with what, across timeframes
"""
