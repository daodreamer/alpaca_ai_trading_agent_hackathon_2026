"""Architectural boundaries, enforced rather than trusted.

Three rules from the architecture, checked by reading the import graph:

1.  **Only ``agent/`` may call an LLM.** If a model call appears in the evaluator,
    the evaluator stops being an independent judge of the model's proposals and
    the entire validation apparatus becomes decorative.

2.  **The pure layers stay pure.** ``core``, ``features``, ``dsl``, ``backtest``,
    ``validation`` and ``evaluator`` may not reach the network, the filesystem,
    the clock, or the database. Determinism is what makes an experiment
    replayable, and a single ``datetime.now()`` in a feature is enough to end it.

3.  **Nothing here reaches a trading API.** This project finds strategies and
    hands off a target book; execution is a different system with different
    invariants (PLAN Phase 4). The market-data host is allowed and the trading
    host is not, which is a distinction a test can hold and a README cannot.

Checked with the ``ast`` module rather than by importing, so a violation is
caught even in a module that is never executed.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "aqr"

PURE_LAYERS = ("core", "features", "dsl", "backtest", "validation", "evaluator", "options")

LLM_MODULES = {"anthropic", "openai", "google", "cohere", "mistralai", "ollama", "litellm"}

# I/O and non-determinism the pure layers must not reach for.
FORBIDDEN_IN_PURE = {
    "requests",
    "httpx",
    "urllib",
    "socket",
    "sqlite3",
    "yfinance",
    "aiohttp",
    "ib_async",
}

# ``random`` is absent from this list on purpose: Monte Carlo needs a generator.
# It is required to be explicitly seeded, which the robustness tests assert.


def python_files(*parts: str) -> list[Path]:
    root = SRC.joinpath(*parts) if parts else SRC
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def imported_modules(path: Path) -> set[str]:
    """Top-level module names imported by a file, including inside functions."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module.split(".")[0])
    return found


@pytest.mark.parametrize("path", python_files(), ids=lambda p: str(p.relative_to(SRC)))
def test_only_the_agent_layer_may_call_an_llm(path: Path) -> None:
    relative = path.relative_to(SRC)
    if relative.parts[0] == "agent":
        return
    offenders = imported_modules(path) & LLM_MODULES
    assert not offenders, (
        f"{relative} imports {sorted(offenders)}. Only aqr/agent/ may call an LLM — "
        "a model call anywhere else means the scorer is judging its own proposals."
    )


@pytest.mark.parametrize(
    "path",
    [p for layer in PURE_LAYERS for p in python_files(layer)],
    ids=lambda p: str(p.relative_to(SRC)),
)
def test_pure_layers_do_no_io(path: Path) -> None:
    offenders = imported_modules(path) & FORBIDDEN_IN_PURE
    assert not offenders, (
        f"{path.relative_to(SRC)} imports {sorted(offenders)}. "
        f"The {path.relative_to(SRC).parts[0]} layer must stay deterministic and offline."
    )


@pytest.mark.parametrize(
    "path",
    [p for layer in ("core", "features", "dsl") for p in python_files(layer)],
    ids=lambda p: str(p.relative_to(SRC)),
)
def test_lowest_layers_do_not_read_the_clock(path: Path) -> None:
    """Time is a parameter, never something a computation reaches for itself.

    ``dsl/schema.py`` and the data layer legitimately format timestamps; what is
    banned here is *reading the current time*, which would make a feature's value
    depend on when it was computed.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in ("now", "utcnow", "today", "time"):
            continue
        owner = getattr(node.func.value, "id", None) or getattr(node.func.value, "attr", None)
        if owner in ("datetime", "date", "time"):
            raise AssertionError(
                f"{path.relative_to(SRC)}:{node.lineno} reads the clock. "
                "Pass the time in as a parameter instead."
            )


# --------------------------------------------------------------------------- #
# The handoff boundary
# --------------------------------------------------------------------------- #

# This project produces a target book and something else executes it. That is a
# decision (PLAN Phase 4), and a decision worth anything is one a test can break.
#
# ``data.alpaca.markets`` is the market-data host and is allowed: reading bars is
# what this project does. Everything below is the trading side -- the hosts that
# accept an order, the paths that place one, and the SDKs whose only purpose is
# to reach them.
TRADING_HOSTS = ("api.alpaca.markets", "paper-api.alpaca.markets", "broker-api.alpaca.markets")
ORDER_PATHS = ("/v2/orders", "/v2/positions", "/v2/account")
TRADING_MODULES = {"alpaca_trade_api", "alpaca", "alphagate", "ib_insync"}
# ``ALPACA_API_KEY_ID`` is absent on purpose: the data provider needs a key to
# read bars, and reading bars is the job. What may not appear is anything that
# names an *account* -- an account id is the thing an order is placed against.
CREDENTIAL_NAMES = ("ALPACA_ACCOUNT", "ACCOUNT_ID", "APCA_API_KEY_ID")


@pytest.mark.parametrize("path", python_files(), ids=lambda p: str(p.relative_to(SRC)))
def test_nothing_here_can_reach_a_trading_api(path: Path) -> None:
    """No trading host, no order path, no broker SDK, anywhere under ``src/aqr``.

    The README promises that order placement will never live here. A promise in
    prose survives exactly until someone adds ``import alpaca`` at 2am, so the
    promise is checked instead: a book is written to a file, and whatever
    executes it reads the file.
    """
    text = path.read_text(encoding="utf-8")
    offenders = [h for h in TRADING_HOSTS if h in text]
    offenders += [p for p in ORDER_PATHS if p in text]
    offenders += [n for n in CREDENTIAL_NAMES if n in text]
    offenders += sorted(imported_modules(path) & TRADING_MODULES)
    assert not offenders, (
        f"{path.relative_to(SRC)} names {offenders}. This project hands off a target "
        "book; execution is a different system with different invariants, and the "
        "file is the whole interface between them."
    )


def test_the_data_host_is_the_only_alpaca_host() -> None:
    """Stated as its own check so the allowance is explicit rather than implied.

    ``data.alpaca.markets`` serves bars. If a second Alpaca host ever appears,
    the test above is what refuses it -- this one records why the first is fine.
    """
    hosts = {
        line.strip()
        for path in python_files()
        for line in path.read_text(encoding="utf-8").splitlines()
        if "alpaca.markets" in line
    }
    assert all("data.alpaca.markets" in line for line in hosts), sorted(hosts)


def test_dependency_direction_is_downward() -> None:
    """A lower layer may never import a higher one."""
    order = [
        "core",
        "data",
        "features",
        "dsl",
        "backtest",
        "validation",
        "evaluator",
        "registry",
        "agent",
    ]
    rank = {name: i for i, name in enumerate(order)}
    violations: list[str] = []

    for layer in order:
        for path in python_files(layer):
            for module in imported_modules(path):
                if module != "aqr":
                    continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom) or not node.module:
                    continue
                parts = node.module.split(".")
                if len(parts) < 2 or parts[0] != "aqr":
                    continue
                target = parts[1]
                if target in rank and rank[target] > rank[layer]:
                    violations.append(
                        f"{path.relative_to(SRC)}:{node.lineno} imports upward: "
                        f"{layer} -> {target}"
                    )
    assert not violations, "\n".join(violations)
