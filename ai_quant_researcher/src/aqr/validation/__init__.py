"""Splits, walk-forward, robustness and overfitting detection. Pure."""

from aqr.validation.overfitting import (
    OverfittingReport,
    Signal,
    deflated_sharpe,
    detect_overfitting,
)
from aqr.validation.params import Slot, apply_params, get_param, neighbours, set_param, slots
from aqr.validation.robustness import (
    AssetReport,
    MonteCarloReport,
    ParameterReport,
    RegimeReport,
    asset_robustness,
    monte_carlo,
    parameter_stability,
    regime_robustness,
)
from aqr.validation.splits import (
    Fold,
    Split,
    context_slice,
    purge_overlap,
    three_way_split,
    walk_forward_folds,
)
from aqr.validation.walkforward import FoldResult, WalkForwardReport, run_walk_forward, warmup_for

__all__ = [
    "AssetReport",
    "Fold",
    "FoldResult",
    "MonteCarloReport",
    "OverfittingReport",
    "ParameterReport",
    "RegimeReport",
    "Signal",
    "Slot",
    "Split",
    "WalkForwardReport",
    "apply_params",
    "asset_robustness",
    "context_slice",
    "deflated_sharpe",
    "detect_overfitting",
    "get_param",
    "monte_carlo",
    "neighbours",
    "parameter_stability",
    "purge_overlap",
    "regime_robustness",
    "run_walk_forward",
    "set_param",
    "slots",
    "three_way_split",
    "walk_forward_folds",
    "warmup_for",
]
