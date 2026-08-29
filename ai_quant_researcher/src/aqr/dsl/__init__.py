"""The strategy language. Pure: parse, validate, hash. It never executes code."""

from aqr.dsl.expr import Expr, ParseError, evaluate, feature_keys, parse
from aqr.dsl.loader import dumps, load_file, loads, save_file
from aqr.dsl.schema import (
    ExitRules,
    Sizing,
    StopLoss,
    StrategySpec,
    TakeProfit,
    Universe,
    spec_from_dict,
    spec_to_dict,
)
from aqr.dsl.validator import ValidationReport, validate, validate_against

__all__ = [
    "ExitRules",
    "Expr",
    "ParseError",
    "Sizing",
    "StopLoss",
    "StrategySpec",
    "TakeProfit",
    "Universe",
    "ValidationReport",
    "dumps",
    "evaluate",
    "feature_keys",
    "load_file",
    "loads",
    "parse",
    "save_file",
    "spec_from_dict",
    "spec_to_dict",
    "validate",
    "validate_against",
]
