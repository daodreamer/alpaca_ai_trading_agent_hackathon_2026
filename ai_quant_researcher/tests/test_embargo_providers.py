"""The two cache roots, and the wrappers that keep them apart.

The seal detects a peek. These wrappers make the common ones impossible, which
is better: an alarm that fires after the campaign is over has still cost the
campaign.

``ResearchProvider`` truncates at the embargo on the way out, so search code that
asks for "everything up to today" silently gets everything up to the embargo --
silently on purpose, because the alternative is an exception in the middle of a
40-hypothesis campaign over a request that was reasonable to make.

``SealedProvider`` is the only thing that can return the embargoed years, and it
cannot be constructed by accident: it requires an explicit token and the sealed
phase. Two locks rather than one, because the failure they guard is unrecoverable
-- once the embargoed years have informed a choice, no later run can un-inform it.
"""

from __future__ import annotations

import ast
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from aqr.data.bars import Bars
from aqr.data.embargo import (
    RESEARCH_ROOT,
    SEALED_ROOT,
    ResearchProvider,
    SealedProvider,
    SealToken,
    audit_cache_root,
)
from aqr.seal import CANARY_SYMBOL, EMBARGO_START, Contamination, Phase, Seal, scope

SRC = Path(__file__).resolve().parents[1] / "src" / "aqr"

START = datetime(2015, 1, 1, tzinfo=UTC)
PAST_EMBARGO = EMBARGO_START + timedelta(days=200)


class FakeProvider:
    """A provider that always returns bars spanning the whole requested window,
    including the embargoed years. Standing in for a live vendor call, which is
    exactly the thing that cannot be trusted to respect an end date."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, datetime, datetime]] = []

    def dataset_version(self, timeframe: str) -> str:
        return "fake-1"

    def load(
        self, symbol: str, start: datetime, end: datetime, timeframe: str = "1D"
    ) -> Bars:
        self.calls.append((symbol, start, end))
        a, b = int(start.timestamp()), int(end.timestamp())
        t = np.arange(a, b, 86_400, dtype=np.int64)
        price = np.linspace(100.0, 200.0, t.size)
        return Bars(
            symbol=symbol,
            timeframe=timeframe,
            event_time=t,
            open=price,
            high=price + 1.0,
            low=price - 1.0,
            close=price,
            volume=np.full(t.size, 1e6),
        )


# --------------------------------------------------------------------------
# ResearchProvider


def test_research_provider_truncates_at_the_embargo() -> None:
    with scope(Seal()) as seal:
        bars = ResearchProvider(FakeProvider()).load("AAPL", START, PAST_EMBARGO)
        assert len(bars) > 0
        assert bars.event_time.max() < int(EMBARGO_START.timestamp())
        assert seal.tainted is False


def test_research_provider_does_not_even_ask_for_embargoed_bars() -> None:
    """Truncating the response is not enough when the response crosses a network.

    A vendor request for the embargoed years is logged on the vendor's side, is
    paid for, and on a delayed feed can arrive after the truncation decision was
    made. The end date is clamped before the call, not after.
    """
    inner = FakeProvider()
    ResearchProvider(inner).load("AAPL", START, PAST_EMBARGO)
    _, _, asked_end = inner.calls[0]
    assert asked_end <= EMBARGO_START


def test_research_provider_leaves_an_early_window_untouched() -> None:
    inner = FakeProvider()
    end = datetime(2020, 1, 1, tzinfo=UTC)
    bars = ResearchProvider(inner).load("AAPL", START, end)
    assert inner.calls[0][2] == end
    assert len(bars) > 0


def test_research_provider_writes_one_ledger_row_per_load() -> None:
    with scope(Seal()) as seal:
        provider = ResearchProvider(FakeProvider())
        provider.load("AAPL", START, PAST_EMBARGO)
        provider.load("MSFT", START, PAST_EMBARGO)
        assert [r.symbol for r in seal.loads] == ["AAPL", "MSFT"]
        assert all(r.max_event_time < int(EMBARGO_START.timestamp()) for r in seal.loads)
        assert all(r.source.startswith("research:") for r in seal.loads)


def test_research_provider_refuses_the_sealed_phase() -> None:
    with scope(Seal(phase=Phase.SEALED)), pytest.raises(Contamination):
        ResearchProvider(FakeProvider()).load("AAPL", START, PAST_EMBARGO)


def test_a_window_entirely_inside_the_embargo_returns_nothing_rather_than_raising() -> None:
    """Asking for the embargoed years from research code is a mistake, but it is
    a recoverable one and the campaign should not die of it. It must produce no
    data and leave a ledger row saying so."""
    with scope(Seal()) as seal:
        bars = ResearchProvider(FakeProvider()).load(
            "AAPL", EMBARGO_START + timedelta(days=1), PAST_EMBARGO
        )
        assert len(bars) == 0
        assert seal.tainted is False
        assert seal.loads[-1].rows == 0


# --------------------------------------------------------------------------
# SealedProvider


def test_sealed_provider_needs_a_token() -> None:
    with pytest.raises(TypeError):
        SealedProvider(FakeProvider())  # type: ignore[call-arg]


def test_sealed_provider_needs_the_sealed_phase() -> None:
    with scope(Seal(phase=Phase.RESEARCH)), pytest.raises(Contamination):
        SealedProvider(FakeProvider(), token=SealToken()).load("AAPL", START, PAST_EMBARGO)


def test_sealed_provider_returns_the_embargoed_years() -> None:
    with scope(Seal(phase=Phase.SEALED)) as seal:
        bars = SealedProvider(FakeProvider(), token=SealToken()).load(
            "AAPL", START, PAST_EMBARGO
        )
        assert bars.event_time.max() >= int(EMBARGO_START.timestamp())
        assert seal.tainted is False
        assert seal.loads[-1].source.startswith("sealed:")


# --------------------------------------------------------------------------
# The roots


def test_the_two_roots_are_distinct_and_named_for_what_they_hold() -> None:
    assert RESEARCH_ROOT != SEALED_ROOT
    assert "research" in RESEARCH_ROOT.name
    assert "sealed" in SEALED_ROOT.name


def test_audit_reports_the_latest_bar_in_a_cache_root(tmp_path: Path) -> None:
    """The research root's guarantee is physical: the rows are not on the disk.

    ``audit_cache_root`` is what makes that checkable without loading anything
    into the process that is doing the checking.
    """
    root = tmp_path / "1D"
    root.mkdir(parents=True)
    (root / "AAPL.csv").write_text(
        "timestamp,open,high,low,close,volume,available_time\n"
        "2020-01-02T00:00:00+00:00,1,1,1,1,100,2020-01-02T00:00:00+00:00\n"
        "2021-01-04T00:00:00+00:00,1,1,1,1,100,2021-01-04T00:00:00+00:00\n",
        encoding="utf-8",
    )
    report = audit_cache_root(tmp_path)
    assert report.clean is True
    assert report.latest == datetime(2021, 1, 4, tzinfo=UTC)


def test_audit_flags_a_research_root_holding_embargoed_rows(tmp_path: Path) -> None:
    root = tmp_path / "1D"
    root.mkdir(parents=True)
    late = (EMBARGO_START + timedelta(days=5)).isoformat()
    (root / "AAPL.csv").write_text(
        f"timestamp,open,high,low,close,volume,available_time\n{late},1,1,1,1,100,{late}\n",
        encoding="utf-8",
    )
    report = audit_cache_root(tmp_path)
    assert report.clean is False
    assert "AAPL" in report.offenders


def test_audit_reports_the_canary_without_calling_it_contamination(tmp_path: Path) -> None:
    """The tripwire lives in the research root and holds embargoed rows on purpose.

    A canary placed somewhere a peek cannot reach catches nothing, so the audit
    has to tell the two apart: a real symbol past the embargo is an offence, the
    canary is the alarm working.
    """
    root = tmp_path / "1D"
    root.mkdir(parents=True)
    late = (EMBARGO_START + timedelta(days=5)).isoformat()
    (root / f"{CANARY_SYMBOL}.csv").write_text(
        f"timestamp,open,high,low,close,volume,available_time\n{late},1,1,1,1,100,{late}\n",
        encoding="utf-8",
    )
    report = audit_cache_root(tmp_path)
    assert report.canary_present is True
    assert report.offenders == ()
    assert report.clean is True


def test_audit_does_not_taint_the_ambient_seal(tmp_path: Path) -> None:
    """Auditing reads timestamps, not prices, and must not itself be the peek it
    is looking for -- otherwise nobody can ever check."""
    root = tmp_path / "1D"
    root.mkdir(parents=True)
    late = (EMBARGO_START + timedelta(days=5)).isoformat()
    (root / "AAPL.csv").write_text(
        f"timestamp,open,high,low,close,volume,available_time\n{late},1,1,1,1,100,{late}\n",
        encoding="utf-8",
    )
    with scope(Seal()) as seal:
        audit_cache_root(tmp_path)
        assert seal.tainted is False


# --------------------------------------------------------------------------
# Boundaries


def test_no_module_outside_the_embargo_layer_constructs_a_seal_token() -> None:
    """The token is the whole lock. If the research loop can mint one, there is
    no lock -- so the only places allowed to name it are the embargo module and
    the sealed CLI entry point."""
    allowed = {"embargo.py", "cli_sealed.py"}
    offenders: list[str] = []
    for path in sorted(p for p in SRC.rglob("*.py") if "__pycache__" not in p.parts):
        if path.name in allowed:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            name = None
            if isinstance(node, ast.Call):
                name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if name == "SealToken":
                offenders.append(f"{path.relative_to(SRC)}:{node.lineno}")
    assert not offenders, "\n".join(offenders)


def test_the_agent_layer_cannot_reach_the_sealed_provider() -> None:
    """The model must never be handed the embargoed years, by any route.

    This is the import-graph half; the prompt-content half lives in the research
    tests, which assert no post-embargo number reaches a prompt string.
    """
    offenders: list[str] = []
    for path in sorted((SRC / "agent").rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "aqr.data.embargo":
                for alias in node.names:
                    if alias.name in ("SealedProvider", "SealToken", "SEALED_ROOT"):
                        offenders.append(f"{path.relative_to(SRC)}:{node.lineno} {alias.name}")
    assert not offenders, "\n".join(offenders)
