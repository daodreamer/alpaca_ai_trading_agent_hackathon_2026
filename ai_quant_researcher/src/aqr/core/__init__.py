"""Pure numerics. No I/O, no LLM, no configuration — deterministic functions only.

Every indicator here is *causal*: the value at index ``i`` is a function of
``x[:i + 1]`` and nothing else. Warm-up periods are ``NaN`` rather than
back-filled, because back-filling is look-ahead bias wearing a disguise.
"""

from aqr.core.indicators import (  # noqa: F401
    adx,
    atr,
    bollinger,
    ema,
    macd,
    percentile_rank,
    realized_vol,
    roc,
    rolling_max,
    rolling_min,
    rsi,
    rvol,
    sma,
    stochastic_k,
    true_range,
    vwap_session,
)
