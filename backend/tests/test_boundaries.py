"""Architecture guards for AlphaGate.

These encode specs/01-architecture.md as tests rather than prose. Adapted from
the guard suite that ships with `alphagate.core` upstream.

Guard 1 — the pure layers (core, options, risk, equity) import only the standard
          library and each other.
Guard 2 — no LLM SDK outside `alphagate.agent`. Rule 1 of specs/01.
Guard 3 — no network stack in the pure layers. Rule 2's precondition.
Guard 4 — `Decimal` is never constructed from a float literal. Money is exact.
Guard 5 — each gated order type is minted inside exactly one module, and
          `execution` accepts nothing else. Rule 2 of specs/01 and specs/09 D7,
          which together are the whole claim of the project.
Guard 9 — `alphagate` does not import `aqr`, and `aqr` does not import
          `alphagate`. specs/09 D0: the seam between the two projects is a file.
"""

from __future__ import annotations

import ast
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "alphagate"

PURE_PACKAGES = ("core", "options", "risk", "equity")
LLM_PACKAGES = frozenset({"anthropic", "openai", "google", "litellm", "langchain"})
NETWORK_PACKAGES = frozenset(
    {"httpx", "requests", "aiohttp", "websockets", "urllib3", "fastapi", "starlette"}
)


def _python_files(root: Path) -> Iterator[Path]:
    yield from sorted(root.rglob("*.py"))


def _top_level_imports(path: Path) -> Iterator[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.module.split(".")[0]


def _pure_files() -> Iterator[Path]:
    for package in PURE_PACKAGES:
        root = SRC / package
        if root.is_dir():
            yield from _python_files(root)


def test_pure_layers_import_only_stdlib_and_each_other() -> None:
    """Guard 1. A pure layer that grows a third-party import stops being testable
    offline and stops being deterministic. Both are load-bearing claims here."""
    allowed = sys.stdlib_module_names | {"alphagate"}
    offenders: list[str] = []
    for path in _pure_files():
        for name in _top_level_imports(path):
            if name not in allowed:
                offenders.append(f"{path.relative_to(SRC)} imports {name}")
    assert not offenders, "pure layers must import stdlib only:\n" + "\n".join(offenders)


def test_pure_layers_never_import_alphagate_infrastructure() -> None:
    """Guard 1, second half. `alphagate.core` may not reach sideways into
    `alphagate.infra` or upward into agent/execution/interface."""
    offenders: list[str] = []
    for path in _pure_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                head = node.module.split(".")
                if head[0] == "alphagate" and len(head) > 1 and head[1] not in PURE_PACKAGES:
                    offenders.append(f"{path.relative_to(SRC)} imports {node.module}")
    assert not offenders, "pure layers may only import pure layers:\n" + "\n".join(offenders)


def test_llm_sdks_live_only_in_the_agent_package() -> None:
    """Guard 2. If a model call appears in `risk`, the Gate is not a gate."""
    offenders: list[str] = []
    for path in _python_files(SRC):
        if path.relative_to(SRC).parts[0] == "agent":
            continue
        for name in _top_level_imports(path):
            if name in LLM_PACKAGES:
                offenders.append(f"{path.relative_to(SRC)} imports {name}")
    assert not offenders, "only alphagate.agent may talk to a model:\n" + "\n".join(offenders)


def test_network_stack_stays_out_of_the_pure_layers() -> None:
    """Guard 3."""
    offenders: list[str] = []
    for path in _pure_files():
        for name in _top_level_imports(path):
            if name in NETWORK_PACKAGES:
                offenders.append(f"{path.relative_to(SRC)} imports {name}")
    assert not offenders, "pure layers must not reach the network:\n" + "\n".join(offenders)


def test_decimal_is_never_built_from_a_float_literal() -> None:
    """Guard 4. `Decimal(0.1)` is not 0.1. Ruff's RUF032 is the fast half of this
    rule; this test is the half that survives someone editing the lint config."""
    offenders: list[str] = []
    for path in _python_files(SRC):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if name != "Decimal" or not node.args:
                continue
            first = node.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, float):
                offenders.append(f"{path.relative_to(SRC)}:{node.lineno}")
    assert not offenders, "Decimal from a float literal:\n" + "\n".join(offenders)


@pytest.mark.parametrize("package", PURE_PACKAGES)
def test_pure_packages_are_present_or_not_yet_written(package: str) -> None:
    """`options` and `risk` arrive on day one and day two. Until then this test
    documents the intended shape rather than failing the suite."""
    root = SRC / package
    if not root.is_dir():
        pytest.skip(f"alphagate.{package} not implemented yet — see specs/")
    assert (root / "__init__.py").is_file()


# ---------------------------------------------------------------------- #
# Guard 5 — the doors. specs/01 Rule 2, specs/03 D3, specs/04 D1, specs/09 D7.
# ---------------------------------------------------------------------- #

GATED_TYPES = {
    "GatedOrder": SRC / "risk" / "gate.py",
    "GatedEquityOrder": SRC / "risk" / "equity_gate.py",
}
"""Every type that means "this passed a Gate", and the one module allowed to
mint it.

A dict rather than two tests, because the failure this guard exists to catch is
a *third* order path arriving with no guard of its own. Adding a gated type
without adding a row here leaves it uncovered — so the entry-point test below
asserts that every `submit*` in `execution` names a key of this mapping, which
turns that omission into a failure rather than a silence."""

APPROVAL_TYPES = {
    "Approved": SRC / "risk" / "gate.py",
    "ApprovedEquity": SRC / "risk" / "equity_gate.py",
}


@pytest.mark.parametrize("name", sorted(GATED_TYPES | APPROVAL_TYPES))
def test_gated_orders_are_minted_in_exactly_one_module(name: str) -> None:
    """A `GatedOrder(...)` call anywhere else is a bypass, whatever it is named.

    The verdict modules refuse such a call at runtime by walking the stack; this
    is the static half, and it fails during a normal test run rather than at
    09:31 on a trading morning.
    """
    minting = (GATED_TYPES | APPROVAL_TYPES)[name]
    offenders: list[str] = []
    for path in _python_files(SRC):
        if path == minting:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == name:
                offenders.append(f"{path.relative_to(SRC)}:{node.lineno}")
    assert not offenders, (
        f"only {minting.relative_to(SRC)} may mint a {name}:\n" + "\n".join(offenders)
    )


def test_execution_accepts_nothing_but_a_gated_order() -> None:
    """specs/04 D1 and specs/09 D7. Every entry point takes a gated type.

    Scanned by *prefix* — any function in `execution` whose name starts with
    `submit` — rather than by exact name. A second order path called
    `submit_equity` would have slipped past a guard that only knew the word
    `submit`, and the whole point of this file is that a new door cannot be
    added without a key.
    """
    execution = SRC / "execution"
    if not execution.is_dir():
        pytest.skip("alphagate.execution not implemented yet — see specs/04")

    submits = []
    for path in _python_files(execution):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith("submit"):
                submits.append((path, node))
    assert submits, "alphagate.execution exists but exposes no `submit`"

    found: set[str] = set()
    for path, node in submits:
        first = node.args.args[0] if node.args.args else None
        annotation = ast.unparse(first.annotation) if first and first.annotation else None
        assert annotation in GATED_TYPES, (
            f"{path.relative_to(SRC)}:{node.lineno} — {node.name}'s first parameter "
            f"is {annotation!r}, which is not one of {sorted(GATED_TYPES)}. A door "
            "that accepts an ungated type is a bypass."
        )
        found.add(annotation)
    assert found == set(GATED_TYPES), (
        f"every gated type needs a door: {sorted(set(GATED_TYPES) - found)} has none, "
        "which means it is minted and never used, or used through a path this "
        "guard cannot see"
    )
    # An override is a *parameter* or a *name*, never a word in a sentence. The
    # first version of this guard scanned raw text and tripped on the docstring
    # explaining why no bypass exists, which is the wrong thing to fail on.
    banned = {"force", "bypass", "override", "skip_checks", "unsafe", "no_gate"}
    offenders: list[str] = []
    for path in _python_files(execution):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                args = node.args
                names = {
                    a.arg
                    for a in [*args.posonlyargs, *args.args, *args.kwonlyargs]
                }
                for hit in names & banned:
                    offenders.append(
                        f"{path.relative_to(SRC)}:{node.lineno} — {node.name}() takes {hit!r}"
                    )
            elif isinstance(node, ast.Call):
                for kw in node.keywords:
                    if kw.arg in banned:
                        offenders.append(
                            f"{path.relative_to(SRC)}:{node.lineno} — call passes {kw.arg!r}"
                        )
    assert not offenders, (
        "execution must expose no override switch:\n" + "\n".join(offenders)
    )


# ---------------------------------------------------------------------- #
# Guard 6 — market data is read-only. adr/0002 D2, specs/01 Rule 1b.
# ---------------------------------------------------------------------- #


def test_market_data_never_writes() -> None:
    """`alphagate.marketdata` may only issue GETs.

    Orders leave through exactly one door — `execution.submit`, holding a
    `GatedOrder` — and a data adapter that could POST would be a second door
    nobody was watching. The exemption that lets this package import `httpx`
    (pyproject, TID251) is what makes this guard necessary rather than academic.
    """
    package = SRC / "marketdata"
    if not package.is_dir():
        pytest.skip("alphagate.marketdata not implemented yet")

    write_verbs = {"post", "put", "patch", "delete", "stream", "send"}
    offenders: list[str] = []
    for path in _python_files(package):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in write_verbs
            ):
                offenders.append(
                    f"{path.relative_to(SRC)}:{node.lineno} calls .{node.func.attr}()"
                )
    assert not offenders, "market data must be read-only:\n" + "\n".join(offenders)


def test_only_the_agent_package_holds_a_model_key() -> None:
    """Rule 1 of specs/01, checked from the other direction.

    `test_llm_sdks_live_only_in_the_agent_package` catches an import. This
    catches the subtler version: a module outside `agent/` reading an LLM API key
    out of the environment, which is what building a second model path starts
    with.
    """
    key_names = {"ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY", "OPENAI_API_KEY"}
    offenders: list[str] = []
    for path in _python_files(SRC):
        if path.relative_to(SRC).parts[0] == "agent":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and node.value in key_names:
                offenders.append(f"{path.relative_to(SRC)}:{node.lineno} names {node.value}")
    assert not offenders, "only alphagate.agent may reach a model:\n" + "\n".join(offenders)


# ---------------------------------------------------------------------- #
# Guard 7 — self-reported confidence never touches sizing or gating.
# specs/05 D5, test plan item 5.
# ---------------------------------------------------------------------- #


def test_confidence_is_recorded_and_never_acted_on() -> None:
    """A model's self-report may be stored and rendered. It may not be *read*.

    specs/05 D5: "the system does not scale position size by it. Self-reported
    confidence is not calibrated, and treating it as a probability is the most
    common way an agent turns a good structure into a bad bet."

    A behavioural test can only show that today's code path ignores it. This is
    the structural version, and it is enforceable only because the field is
    called `self_reported_confidence`. Two other types here have a
    `confidence` — `TrendState` and `Level` — and theirs is a *measured*
    quantity: how much of the requested evidence an engine could actually read.
    The first draft of this guard could not tell the two apart and flagged the
    trend engine, which is how the naming got fixed rather than the guard
    loosened.

    The verbosity is also the defence. Nobody types
    `choice.self_reported_confidence` into a sizing formula without noticing
    what they are about to do — and that, not a deliberate multiplication, is
    the plausible mistake.
    """
    carriers = {"model.py", "deepseek.py", "replay.py"}
    """Modules that *construct* a `Choice`, in either direction.

    `replay.py` is here for the same reason `deepseek.py` is: it rebuilds a
    recorded `Choice` out of the journal (specs/06 D6), so it necessarily names
    every field the record carries. That is carrying, not acting on — the guard
    is about the value reaching sizing or gating, and neither of those is
    reachable from a reader of a JSONL file.

    Adding a module here is a deliberate act and should stay one. The set is
    small on purpose: three modules that build the value, and nothing else in
    the codebase permitted to say its name."""
    offenders: list[str] = []
    for path in _python_files(SRC):
        if path.name in carriers:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            named = (
                isinstance(node, ast.Attribute) and node.attr == "self_reported_confidence"
            ) or (isinstance(node, ast.keyword) and node.arg == "self_reported_confidence")
            if named:
                offenders.append(f"{path.relative_to(SRC)} names self_reported_confidence")
    assert not offenders, (
        "the model's self-report is recorded, never acted on (specs/05 D5):\n"
        + "\n".join(offenders)
    )


def test_sizing_depends_on_nothing_the_model_said() -> None:
    """specs/05 D4, structurally. Quantity is a function of risk and limits.

    The signature is the guarantee: there is no parameter through which a
    `Choice`, a `ModelCall` or a rationale could reach the arithmetic.
    """
    import inspect

    from alphagate.agent.sizing import size_for

    assert set(inspect.signature(size_for).parameters) == {"risk", "limits", "equity"}

    source = (SRC / "agent" / "sizing.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
    } | {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
    }
    assert not names & {"confidence", "rationale", "choice", "candidate_index"}


# ---------------------------------------------------------------------- #
# Guard 8 — the dashboard cannot trade, and the composition root is the
# only place that knows a live account exists. specs/01, and the property
# that makes it safe to leave a browser open during the demo.
# ---------------------------------------------------------------------- #


def test_the_dashboard_cannot_reach_a_broker() -> None:
    """`alphagate.interface` reads the journal and nothing else.

    There must be no code path from a browser to an order. Asserted on imports
    rather than behaviour: a dashboard that merely *could* import `submit` is
    one refactor away from being able to trade, and the refactor will not
    announce itself.

    `alphagate.live` is banned here too. It is the composition root — it holds
    the MCP session and the market data client — so a dashboard that imported it
    would inherit both.
    """
    root = SRC / "interface"
    if not root.is_dir():  # pragma: no cover - the package exists
        pytest.skip("no interface package")
    banned = ("alphagate.execution", "alphagate.marketdata", "alphagate.live")
    offenders: list[str] = []
    for path in _python_files(root):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            module = None
            if isinstance(node, ast.ImportFrom) and node.module:
                module = node.module
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith(banned):
                        offenders.append(f"{path.relative_to(SRC)} imports {alias.name}")
                continue
            if module and module.startswith(banned):
                offenders.append(f"{path.relative_to(SRC)} imports {module}")
    assert not offenders, (
        "the dashboard must not be able to place an order:\n" + "\n".join(offenders)
    )


def test_only_the_live_package_opens_a_real_session() -> None:
    """`StdioSession` is constructed in exactly one place.

    Everything else depends on the `McpSession` protocol, which is what keeps
    the test suite offline (adr/0002 D4). A second construction site is a second
    thing to audit before believing that no test can reach the broker.
    """
    offenders: list[str] = []
    for path in _python_files(SRC):
        if path.relative_to(SRC).parts[0] in {"live", "execution"}:
            continue
        text = path.read_text(encoding="utf-8")
        if "StdioSession(" in text:
            offenders.append(str(path.relative_to(SRC)))
    assert not offenders, (
        "only alphagate.live may open a live MCP session:\n" + "\n".join(offenders)
    )


def test_the_status_snapshot_is_the_only_bridge() -> None:
    """The dashboard learns the live book from a file, never from a session.

    `interface/status.py` deliberately re-declares what it needs rather than
    importing `live.status`'s dataclass, because importing it would pull in
    `alphagate.live` — which owns the MCP session and the market data client —
    and delete the guard above.

    That duplication is the cost of the guarantee, so it is asserted rather
    than left as a comment somebody helpfully "cleans up".
    """
    reader = SRC / "interface" / "status.py"
    if not reader.is_file():  # pragma: no cover - the module exists
        pytest.skip("no status reader")
    for name in _top_level_imports(reader):
        assert name in (sys.stdlib_module_names | {"alphagate"}), (
            f"the status reader must stay dependency-free, found {name}"
        )
    tree = ast.parse(reader.read_text(encoding="utf-8"), filename=str(reader))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert not node.module.startswith("alphagate."), (
                f"the status reader must not import {node.module}; it parses JSON"
            )


# ---------------------------------------------------------------------- #
# Guard 9 — the two projects in this repository stay apart. specs/09 D0.
# ---------------------------------------------------------------------- #

REPO = Path(__file__).resolve().parents[2]
RESEARCHER = REPO / "ai_quant_researcher" / "src" / "aqr"
SCRIPTS = REPO / "scripts"


def test_alphagate_never_imports_the_researcher() -> None:
    """The handoff is a JSON file, and it stays a file.

    `ai_quant_researcher` holds no money and places no orders, so it has no
    `Decimal` rule and no Risk Gate; AlphaGate has both. An import in either
    direction would make one project's invariants the other's problem, which
    CLAUDE.md §2b already says not to do.

    Checked as an import rather than as a dependency declaration because the
    dependency is what would be *noticed*. A stray `sys.path` append and a
    single `from aqr...` is what this actually looks like when it happens.
    """
    offenders: list[str] = []
    for path in _python_files(SRC):
        for name in _top_level_imports(path):
            if name == "aqr":
                offenders.append(f"{path.relative_to(SRC)} imports aqr")
    assert not offenders, (
        "alphagate must not import the researcher; the seam is the target book "
        "file (specs/09 D0):\n" + "\n".join(offenders)
    )


def test_the_pipeline_driver_imports_neither_project() -> None:
    """`scripts/pipeline.py` runs both CLIs and belongs to neither.

    It is the one file in the repository whose whole job is to know that both
    projects exist, which makes it the obvious place for the seam to leak — a
    single `from aqr...` to reuse a constant, and the two projects share a
    process. Running them as subprocesses is the only coupling specs/09 D0
    permits, and this is what holds that line.
    """
    if not SCRIPTS.is_dir():  # pragma: no cover - the directory exists
        pytest.skip("no scripts/ directory")
    offenders: list[str] = []
    for path in _python_files(SCRIPTS):
        for name in _top_level_imports(path):
            if name in {"aqr", "alphagate"}:
                offenders.append(f"{path.relative_to(SCRIPTS)} imports {name}")
    assert not offenders, (
        "the pipeline driver must run both CLIs, not import them:\n"
        + "\n".join(offenders)
    )


def test_the_researcher_never_imports_alphagate() -> None:
    """The other direction, checked from here because this suite is the one CI runs.

    `ai_quant_researcher/tests/test_boundaries.py` asserts its own half — that no
    trading host, order path or broker SDK appears under `src/aqr/`. This is the
    narrower claim that the two packages do not know each other exists, and it
    lives here so that adding the import to *either* side fails *this* build.
    """
    if not RESEARCHER.is_dir():  # pragma: no cover - the sibling project exists
        pytest.skip("ai_quant_researcher is not checked out beside backend/")
    offenders: list[str] = []
    for path in _python_files(RESEARCHER):
        for name in _top_level_imports(path):
            if name == "alphagate":
                offenders.append(f"{path.relative_to(RESEARCHER)} imports alphagate")
    assert not offenders, (
        "the researcher must not import alphagate:\n" + "\n".join(offenders)
    )
