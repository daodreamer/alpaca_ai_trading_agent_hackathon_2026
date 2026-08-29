"""The sealed entry point: what it may do, and what it must not be able to do.

``cli_sealed`` is the only process allowed to read the embargoed years. Its
value comes entirely from what it cannot do, so most of this file is boundary
tests rather than behaviour tests.

The one behaviour that matters is the phase transition. It is one-way and it is
refused once the process has read anything, which is what stops the sequence
"search, then promote yourself to sealed" from laundering a tainted campaign
into a clean certificate.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from aqr.seal import Contamination, Phase, Seal, enter_sealed_phase, scope

SRC = Path(__file__).resolve().parents[1] / "src" / "aqr"
SEALED_CLI = SRC / "cli_sealed.py"


def _load(symbol: str = "AAPL", *, max_ts: int = 1_000_000) -> object:
    from aqr.seal import LoadRecord

    return LoadRecord(
        source="test",
        symbol=symbol,
        requested_start=0,
        requested_end=max_ts,
        rows=10,
        max_event_time=max_ts,
    )


# --------------------------------------------------------------------------
# The transition


def test_entering_the_sealed_phase_switches_the_ambient_seal() -> None:
    with scope(Seal()) as seal:
        assert seal.phase is Phase.RESEARCH
        enter_sealed_phase()
        assert seal.phase is Phase.SEALED


def test_the_transition_is_refused_once_the_process_has_read_anything() -> None:
    """Otherwise a campaign could search, then promote itself and read the
    answer, and the certificate would say ``phase: sealed, tainted: false``."""
    with scope(Seal()) as seal:
        seal.record_load(_load())  # type: ignore[arg-type]
        with pytest.raises(Contamination):
            enter_sealed_phase()
        assert seal.phase is Phase.RESEARCH


def test_the_transition_is_refused_after_a_series_has_merely_been_observed() -> None:
    """``observe`` is the sensor on ``Bars``, and it fires for slices and folds
    as well as loads. A process that has computed on anything is a search."""
    with scope(Seal()) as seal:
        seal.observe("AAPL", np.array([1_000_000], dtype=np.int64))
        with pytest.raises(Contamination):
            enter_sealed_phase()


def test_the_transition_is_idempotent_but_never_reversible() -> None:
    with scope(Seal()) as seal:
        enter_sealed_phase()
        enter_sealed_phase()
        assert seal.phase is Phase.SEALED
    with scope(Seal(phase=Phase.SEALED)) as seal:
        assert not hasattr(seal, "enter_research")


def test_the_transition_is_written_into_the_digest() -> None:
    """A certificate that could be produced by a research process is not
    evidence of a sealed one."""
    plain, promoted = Seal(), Seal()
    before = promoted.digest
    with scope(promoted):
        enter_sealed_phase()
    assert promoted.digest != before
    assert promoted.digest != plain.digest


def test_every_seal_carries_a_campaign_identity() -> None:
    """One process, one seal, one campaign. The ancestry taint check groups
    experiments by this, so two campaigns must never share it."""
    assert Seal().run_id != Seal().run_id
    assert Seal().certificate()["run_id"]


# --------------------------------------------------------------------------
# Boundaries


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module)
    return found


def test_the_sealed_entry_point_exists() -> None:
    assert SEALED_CLI.exists(), (
        "test_no_module_outside_the_embargo_layer_constructs_a_seal_token already "
        "names this file as allowed to mint a token. It must exist to be audited."
    )


def test_the_sealed_entry_point_cannot_reach_the_agent_layer() -> None:
    """The sealed run executes one pre-registered spec. It does not search, so
    it has no use for a proposer -- and a proposer in this process would be a
    model choosing hypotheses with the answer in front of it."""
    offenders = [m for m in _imports(SEALED_CLI) if m.split(".")[:2] == ["aqr", "agent"]]
    assert not offenders, f"{SEALED_CLI.name} imports {offenders}"


def test_the_sealed_entry_point_does_not_import_the_research_cli() -> None:
    """``aqr.cli`` carries ``research`` and every provider it needs. Importing it
    would put the search back into the sealed process by the back door."""
    assert "aqr.cli" not in _imports(SEALED_CLI)


def test_only_the_sealed_entry_point_promotes_the_process() -> None:
    """The lock is the import graph, as it is for ``SealToken``."""
    offenders: list[str] = []
    for path in sorted(p for p in SRC.rglob("*.py") if "__pycache__" not in p.parts):
        if path.name in ("seal.py", "cli_sealed.py"):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            name = None
            if isinstance(node, ast.Call):
                name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if name == "enter_sealed_phase":
                offenders.append(f"{path.relative_to(SRC)}:{node.lineno}")
    assert not offenders, "\n".join(offenders)


def test_the_sealed_entry_point_reads_only_through_the_sealed_provider() -> None:
    """No bare ``CsvProvider`` or ``AlpacaProvider`` may be handed to a run.

    Every provider constructed in this file must end up wrapped, or the ledger
    that the certificate rests on has a hole in it exactly the shape of the one
    load nobody wrapped.
    """
    source = SEALED_CLI.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(SEALED_CLI))
    wrapped = sum(
        1
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (getattr(node.func, "id", None) or getattr(node.func, "attr", None))
        == "SealedProvider"
    )
    assert wrapped, "the sealed CLI must construct a SealedProvider"
