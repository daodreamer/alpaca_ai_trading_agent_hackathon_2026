"""YAML in, validated :class:`StrategySpec` out.

``yaml.safe_load`` is used deliberately: the strategy file is untrusted input,
because in the intended workflow an LLM wrote it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from aqr.dsl.schema import StrategySpec, spec_from_dict, spec_to_dict

__all__ = ["dumps", "load_file", "loads", "save_file"]


def loads(text: str) -> StrategySpec:
    """Parse YAML text into a spec, with the source line kept in the error."""
    try:
        raw: Any = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(f"strategy YAML is malformed: {exc}") from exc
    if raw is None:
        raise ValueError("strategy YAML is empty")
    return spec_from_dict(raw)


def load_file(path: Path | str) -> StrategySpec:
    path = Path(path)
    try:
        return loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc


def dumps(spec: StrategySpec) -> str:
    body = {"strategy": spec_to_dict(spec)}
    return yaml.safe_dump(body, sort_keys=False, allow_unicode=True, width=100)


def save_file(spec: StrategySpec, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps(spec), encoding="utf-8")
    return path
