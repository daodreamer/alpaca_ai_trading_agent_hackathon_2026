"""AI Quant Researcher — an LLM-driven quantitative strategy research lab.

The layering is strict and enforced by ``tests/test_boundaries.py``:

    core/        pure numerics. numpy + stdlib only.
    data/        point-in-time OHLCV loading and caching.
    features/    deterministic feature construction over core indicators.
    dsl/         the strategy language an LLM is allowed to emit.
    backtest/    causal simulation with costs.
    validation/  splits, walk-forward, robustness, overfitting detection.
    evaluator/   strategy scoring.
    registry/    experiment + strategy persistence.
    agent/       the ONLY layer permitted to call an LLM.
"""

__version__ = "0.1.0"
