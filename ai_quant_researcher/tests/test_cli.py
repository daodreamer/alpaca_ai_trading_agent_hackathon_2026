"""The CLI, over the shipped example strategies.

These are smoke tests with teeth: they run the real commands against the real
example files, so a broken wiring between layers surfaces here even when every
unit test still passes. They also pin the exit codes, because the exit code is
what a scheduled research job actually reads.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from aqr.cli import app

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
# The strategy's own universe is always loaded, so these options only pin the
# window. Passing a shorter one keeps the CLI tests fast.
SHORT_WINDOW = ["--start", "2014-01-01", "--end", "2022-01-01"]

runner = CliRunner()


@pytest.fixture(scope="module")
def example() -> str:
    return str(EXAMPLES / "trend_pullback.yaml")


def test_features_lists_the_vocabulary() -> None:
    result = runner.invoke(app, ["features"])
    assert result.exit_code == 0
    assert "ema" in result.stdout and "rvol" in result.stdout


def test_validate_accepts_a_shipped_example(example: str) -> None:
    result = runner.invoke(app, ["validate", example, *SHORT_WINDOW])
    assert result.exit_code == 0, result.stdout
    assert "fingerprint" in result.stdout


def test_validate_rejects_an_impossible_strategy(tmp_path: Path) -> None:
    broken = tmp_path / "broken.yaml"
    broken.write_text(
        "strategy:\n"
        "  name: impossible\n"
        "  entry: rsi(14) > 101\n"
        "  universe: {symbols: [SPY]}\n",
        encoding="utf-8",
    )
    result = runner.invoke(app, ["validate", str(broken), *SHORT_WINDOW])
    assert result.exit_code == 1
    assert "never fires" in result.stdout


def test_validate_reports_a_parse_error_clearly(tmp_path: Path) -> None:
    broken = tmp_path / "syntax.yaml"
    broken.write_text(
        "strategy:\n"
        "  name: bad\n"
        "  entry: close > ema(\n"
        "  universe: {symbols: [SPY]}\n",
        encoding="utf-8",
    )
    result = runner.invoke(app, ["validate", str(broken), *SHORT_WINDOW])
    assert result.exit_code != 0


def test_backtest_runs_and_warns_about_in_sample_results(example: str) -> None:
    result = runner.invoke(app, ["backtest", example, *SHORT_WINDOW, "--show-trades", "3"])
    assert result.exit_code == 0, result.stdout
    assert "equity" in result.stdout
    assert "in-sample only" in result.stdout, "an in-sample number was shown without a caveat"


def test_walkforward_reports_folds(example: str) -> None:
    result = runner.invoke(
        app, ["walkforward", example, *SHORT_WINDOW, "--train-bars", "500", "--test-bars", "250"]
    )
    assert result.exit_code == 0, result.stdout
    assert "folds" in result.stdout and "OOS Sharpe" in result.stdout


def test_walkforward_fails_cleanly_without_enough_history(example: str) -> None:
    result = runner.invoke(
        app,
        [
            "walkforward",
            example,
            "--start",
            "2020-01-01",
            "--end",
            "2021-01-01",
        ],
    )
    assert result.exit_code == 1
    assert "not enough history" in result.stdout


def test_evaluate_writes_to_the_registry_and_sets_an_exit_code(
    example: str, tmp_path: Path
) -> None:
    db = tmp_path / "research.sqlite"
    result = runner.invoke(
        app, ["evaluate", example, *SHORT_WINDOW, "--db", str(db), "--train-bars", "500"]
    )
    # Exit code encodes the verdict: 0 means promotable, 1 means it is not.
    assert result.exit_code in (0, 1)
    assert db.exists()

    listed = runner.invoke(app, ["experiments", "--db", str(db)])
    assert listed.exit_code == 0
    assert "cumulative backtests" in listed.stdout


def test_evaluate_json_output_is_machine_readable(example: str, tmp_path: Path) -> None:
    import json

    result = runner.invoke(
        app,
        [
            "evaluate",
            example,
            *SHORT_WINDOW,
            "--db",
            str(tmp_path / "r.sqlite"),
            "--train-bars",
            "500",
            "--json",
        ],
    )
    payload = json.loads(result.stdout)
    assert payload["strategy"] == "trend_pullback_v1"
    assert payload["verdict"] in ("ACCEPT", "PAPER", "REVIEW", "REJECT")


def test_research_runs_the_offline_loop(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "research",
            "--iterations",
            "2",
            "--symbols",
            "SPY,QQQ",
            "--start",
            "2014-01-01",
            "--end",
            "2022-01-01",
            "--db",
            str(tmp_path / "r.sqlite"),
            "--save-to",
            str(tmp_path / "strategies"),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "cumulative backtests" in result.stdout


def test_registry_and_promote(example: str, tmp_path: Path) -> None:
    db = tmp_path / "r.sqlite"
    runner.invoke(app, ["evaluate", example, *SHORT_WINDOW, "--db", str(db), "--train-bars", "500"])

    listing = runner.invoke(app, ["registry", "--db", str(db)])
    assert listing.exit_code == 0

    from aqr.registry.db import Registry

    with Registry(db) as reg:
        fingerprint = reg.strategies()[0].fingerprint

    # The jump the lifecycle exists to prevent.
    jumped = runner.invoke(app, ["promote", fingerprint, "LIVE", "--db", str(db)])
    assert jumped.exit_code == 1
    assert "cannot go" in jumped.stdout
