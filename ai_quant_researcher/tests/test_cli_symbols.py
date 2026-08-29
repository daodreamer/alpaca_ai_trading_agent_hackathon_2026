"""Regression tests for how the CLI resolves the symbol universe.

These exist because of a real bug: making ``--symbols`` optional for the
strategy-driven commands accidentally emptied the default for ``research`` too.
``"".split(",")`` is ``[""]``, so the loop cheerfully researched an instrument
named "" — the synthetic provider generates bars for any string, so nothing
crashed and the report looked entirely normal apart from a blank symbol column.

The lesson worth keeping: a data source that never refuses an input will turn a
configuration mistake into a plausible result.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from aqr.cli import _symbols_for, app
from aqr.dsl.loader import load_file

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
runner = CliRunner()


def test_a_strategy_universe_is_always_loaded_in_full() -> None:
    spec = load_file(EXAMPLES / "trend_pullback.yaml")
    assert set(spec.universe.symbols) <= set(_symbols_for(spec, ""))


def test_extra_symbols_are_added_not_substituted() -> None:
    spec = load_file(EXAMPLES / "trend_pullback.yaml")
    resolved = _symbols_for(spec, "TLT,GLD")
    assert set(spec.universe.symbols) <= set(resolved)
    assert {"TLT", "GLD"} <= set(resolved)


def test_symbols_are_deduplicated_and_upper_cased() -> None:
    spec = load_file(EXAMPLES / "trend_pullback.yaml")
    resolved = _symbols_for(spec, "spy,SPY, qqq ")
    assert len(resolved) == len(set(resolved))
    assert all(s == s.upper() for s in resolved)


def test_research_has_a_real_default_universe(tmp_path: Path) -> None:
    """The bug: with no --symbols, the loop must not research a blank ticker."""
    result = runner.invoke(
        app,
        [
            "research",
            "--iterations",
            "1",
            "--start",
            "2016-01-01",
            "--end",
            "2022-01-01",
            "--db",
            str(tmp_path / "r.sqlite"),
            "--save-to",
            str(tmp_path / "s"),
        ],
    )
    assert result.exit_code == 0, result.stdout

    from aqr.registry.db import Registry

    with Registry(tmp_path / "r.sqlite") as reg:
        rows = reg.experiments(limit=5)
    assert rows, "the run recorded no experiments"
    for row in rows:
        loaded = [s for s in row["symbols"].split(",") if s]
        assert loaded, f"an experiment ran against a blank symbol: {row['symbols']!r}"


@pytest.mark.parametrize("bad", ["", " ", ",", " , "])
def test_an_empty_symbol_list_is_refused(bad: str, tmp_path: Path) -> None:
    """Exit code 2 is Click's usage error: the run stops rather than proceeding."""
    result = runner.invoke(
        app,
        [
            "research",
            "--iterations",
            "1",
            "--symbols",
            bad,
            "--db",
            str(tmp_path / "r.sqlite"),
            "--save-to",
            str(tmp_path / "s"),
        ],
    )
    assert result.exit_code == 2
    assert not (tmp_path / "r.sqlite").exists(), "a rejected run still touched the database"


@pytest.mark.parametrize("stray", ["SPY,,QQQ", "SPY, ,QQQ"])
def test_a_stray_comma_is_tolerated(stray: str) -> None:
    """A doubled comma is a typo with an obvious reading, not an ambiguity.

    Dropping the empty entry is safe because nothing is substituted for it --
    unlike the original bug, where an empty string became a tradable symbol.
    """
    from aqr.cli import _universe

    assert _universe("", stray) == ["SPY", "QQQ"]


@pytest.mark.parametrize("bad", [[], [""], ["SPY", ""], ["SPY", "  "]])
def test_the_loader_names_the_problem(bad: list[str]) -> None:
    import typer

    from aqr.cli import _load

    with pytest.raises(typer.BadParameter, match="empty or contains a blank"):
        _load("synthetic", bad, "2020-01-01", "2021-01-01", "1D", "data")
