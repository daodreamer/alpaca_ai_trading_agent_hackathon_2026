"""Architecture guards for AlphaGate.

These encode specs/01-architecture.md as tests rather than prose. Adapted from
the guard suite that ships with `alphagate.core` upstream.

Guard 1 — the pure layers (core, options, risk) import only the standard library
          and each other.
Guard 2 — no LLM SDK outside `alphagate.agent`. Rule 1 of specs/01.
Guard 3 — no network stack in the pure layers. Rule 2's precondition.
Guard 4 — `Decimal` is never constructed from a float literal. Money is exact.
Guard 5 — `GatedOrder` is minted only inside `alphagate.risk.gate`, and
          `execution` accepts nothing else. Rule 2 of specs/01, which is the
          whole claim of the project.
"""

from __future__ import annotations

import ast
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "alphagate"

PURE_PACKAGES = ("core", "options", "risk")
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
# Guard 5 — the one door. specs/01 Rule 2, specs/03 D3, specs/04 D1.
# ---------------------------------------------------------------------- #

GATE_MODULE = SRC / "risk" / "gate.py"


def test_gated_orders_are_minted_in_exactly_one_module() -> None:
    """A `GatedOrder(...)` call anywhere else is a bypass, whatever it is named.

    `verdict.py` refuses such a call at runtime; this is the static half, and it
    fails during a normal test run rather than at 09:31 on a trading morning.
    """
    offenders: list[str] = []
    for path in _python_files(SRC):
        if path == GATE_MODULE:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "GatedOrder":
                offenders.append(f"{path.relative_to(SRC)}:{node.lineno}")
    assert not offenders, (
        "only alphagate.risk.gate may mint a GatedOrder:\n" + "\n".join(offenders)
    )


def test_execution_accepts_nothing_but_a_gated_order() -> None:
    """specs/04 D1. `submit` takes a `GatedOrder`; there is no REST fallback,
    because a fallback is a bypass."""
    execution = SRC / "execution"
    if not execution.is_dir():
        pytest.skip("alphagate.execution not implemented yet — see specs/04")

    entry = execution / "submit.py"
    candidates = [entry] if entry.is_file() else list(_python_files(execution))
    submits = []
    for path in candidates:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "submit":
                submits.append((path, node))
    assert submits, "alphagate.execution exists but exposes no `submit`"

    for path, node in submits:
        first = node.args.args[0] if node.args.args else None
        annotation = ast.unparse(first.annotation) if first and first.annotation else None
        assert annotation == "GatedOrder", (
            f"{path.relative_to(SRC)}:{node.lineno} — submit's first parameter is "
            f"{annotation!r}, not GatedOrder"
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
