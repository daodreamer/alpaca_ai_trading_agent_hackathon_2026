"""The provenance seal: proof that the search never read the embargoed years.

The last two years are reserved for a single, pre-registered validation run. A
convention would not survive contact with a debugging session, so the embargo is
enforced by three mechanisms with different failure modes, and this file pins all
three.

1.  **A sensor on the type, not on the I/O.** Every series in this system becomes
    a ``Bars`` before anything can compute on it -- CSV, Yahoo, Alpaca, IBKR, the
    simulator, a test fixture. Putting the sensor in ``Bars.__post_init__`` means
    a route nobody anticipated is still observed, which is the only kind of route
    that matters. A check placed on each provider would only cover the providers
    somebody remembered.

2.  **A monotone taint bit.** It goes clean -> tainted and never back. There is no
    public setter, because a flag that can be cleared records the intention of
    whoever cleared it and nothing else.

3.  **An append-only ledger with a hash chain.** The bit says *whether*; the
    ledger says *what was read*, so the audit is a query rather than an act of
    trust. The chain makes the record self-consistent: editing the stored verdict
    without replaying the loads that produced it does not reproduce the digest.

And a canary, for the routes the first three do not anticipate: a symbol that
exists only after the embargo. Nothing legitimate loads it, so its appearance is
physical evidence rather than an inference.

What none of this proves is stated in ``test_the_seal_does_not_certify_knowledge``.
"""

from __future__ import annotations

import ast
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from aqr.data.bars import Bars
from aqr.seal import (
    CANARY_SYMBOL,
    EMBARGO_START,
    Contamination,
    LoadRecord,
    Phase,
    Seal,
    current,
    scope,
)

SRC = Path(__file__).resolve().parents[1] / "src" / "aqr"

BEFORE = EMBARGO_START - timedelta(days=400)
AFTER = EMBARGO_START + timedelta(days=1)


def _epoch(when: datetime, n: int, step: int = 86_400) -> np.ndarray:
    start = int(when.timestamp())
    return np.arange(start, start + n * step, step, dtype=np.int64)


def _bars(symbol: str, when: datetime, n: int = 10) -> Bars:
    t = _epoch(when, n)
    price = np.linspace(100.0, 110.0, n)
    return Bars(
        symbol=symbol,
        timeframe="1D",
        event_time=t,
        open=price,
        high=price + 1.0,
        low=price - 1.0,
        close=price,
        volume=np.full(n, 1e6),
    )


def _load(
    symbol: str = "AAPL", *, source: str = "csv:data-research", max_ts: int = 0
) -> LoadRecord:
    return LoadRecord(
        source=source,
        symbol=symbol,
        requested_start=int(BEFORE.timestamp()),
        requested_end=int(BEFORE.timestamp()) + 86_400,
        rows=10,
        max_event_time=max_ts or int(BEFORE.timestamp()),
    )


# --------------------------------------------------------------------------
# The bit


def test_a_fresh_seal_is_clean() -> None:
    seal = Seal()
    assert seal.tainted is False
    assert seal.max_event_time is None
    assert seal.loads == ()


def test_observing_pre_embargo_data_leaves_the_seal_clean() -> None:
    seal = Seal()
    seal.observe("AAPL", _epoch(BEFORE, 100))
    assert seal.tainted is False
    assert seal.max_event_time is not None
    assert seal.max_event_time < int(EMBARGO_START.timestamp())


def test_observing_post_embargo_data_taints_the_seal() -> None:
    seal = Seal()
    seal.observe("AAPL", _epoch(AFTER, 10))
    assert seal.tainted is True


def test_a_bar_exactly_at_the_embargo_start_is_already_embargoed() -> None:
    """The boundary is inclusive on the embargoed side.

    Which side owns the boundary has to be decided once and written down, or two
    call sites will each assume the other convention.
    """
    seal = Seal()
    seal.observe("AAPL", np.array([int(EMBARGO_START.timestamp())], dtype=np.int64))
    assert seal.tainted is True


def test_taint_is_monotone() -> None:
    seal = Seal()
    seal.observe("AAPL", _epoch(AFTER, 5))
    seal.observe("MSFT", _epoch(BEFORE, 5))
    assert seal.tainted is True, "clean data must not launder an earlier peek"


def test_taint_has_no_public_setter() -> None:
    seal = Seal()
    seal.observe("AAPL", _epoch(AFTER, 5))
    with pytest.raises(AttributeError):
        seal.tainted = False  # type: ignore[misc]


def test_an_empty_series_observes_nothing() -> None:
    seal = Seal()
    seal.observe("AAPL", np.empty(0, dtype=np.int64))
    assert seal.tainted is False
    assert seal.max_event_time is None


# --------------------------------------------------------------------------
# The canary


def test_the_canary_symbol_taints_whatever_its_timestamps_say() -> None:
    """A tripwire for routes the timestamp check does not cover.

    The canary exists only after the embargo, so nothing legitimate can reach it.
    Its taint does not depend on the timestamps it happens to carry, because the
    failure it is there to catch is one where the timestamps are wrong too.
    """
    seal = Seal()
    seal.observe(CANARY_SYMBOL, _epoch(BEFORE, 5))
    assert seal.tainted is True


def test_the_canary_is_named_so_it_cannot_collide_with_a_ticker() -> None:
    assert not CANARY_SYMBOL.isalnum()


# --------------------------------------------------------------------------
# The ledger and the chain


def test_the_ledger_is_append_only() -> None:
    seal = Seal()
    seal.record_load(_load("AAPL"))
    seal.record_load(_load("MSFT"))
    assert [r.symbol for r in seal.loads] == ["AAPL", "MSFT"]
    with pytest.raises((AttributeError, TypeError)):
        seal.loads.append(_load("NVDA"))  # type: ignore[attr-defined]


def test_the_digest_is_deterministic_for_the_same_history() -> None:
    a, b = Seal(), Seal()
    for seal in (a, b):
        seal.record_load(_load("AAPL"))
        seal.record_load(_load("MSFT"))
    assert a.digest == b.digest


def test_the_digest_changes_when_a_load_differs() -> None:
    a, b = Seal(), Seal()
    a.record_load(_load("AAPL"))
    b.record_load(_load("AAPL", max_ts=int(BEFORE.timestamp()) + 86_400))
    assert a.digest != b.digest


def test_the_digest_is_order_sensitive() -> None:
    """Two campaigns that read the same symbols in a different order are not the
    same campaign, and a chain that says they are cannot detect a replayed load."""
    a, b = Seal(), Seal()
    a.record_load(_load("AAPL"))
    a.record_load(_load("MSFT"))
    b.record_load(_load("MSFT"))
    b.record_load(_load("AAPL"))
    assert a.digest != b.digest


def test_a_load_carries_its_own_verdict_into_the_ledger() -> None:
    seal = Seal()
    seal.record_load(_load("AAPL", max_ts=int(AFTER.timestamp())))
    assert seal.tainted is True, "recording a load must apply the same test as observing one"


# --------------------------------------------------------------------------
# The certificate


def test_the_certificate_reports_everything_an_auditor_needs() -> None:
    seal = Seal()
    seal.record_load(_load("AAPL"))
    cert = seal.certificate()
    assert cert["tainted"] is False
    assert cert["digest"] == seal.digest
    assert cert["loads"] == 1
    assert cert["max_event_time"] == int(BEFORE.timestamp())
    assert cert["embargo_start"] == int(EMBARGO_START.timestamp())
    assert cert["phase"] == Phase.RESEARCH.value


def test_the_certificate_is_json_safe() -> None:
    import json

    seal = Seal()
    seal.record_load(_load("AAPL"))
    json.dumps(seal.certificate())


# --------------------------------------------------------------------------
# Phases


def test_research_phase_refuses_a_sealed_read() -> None:
    seal = Seal(phase=Phase.RESEARCH)
    with pytest.raises(Contamination):
        seal.require(Phase.SEALED)


def test_sealed_phase_refuses_a_research_read() -> None:
    """The sealed run is a fresh process with one pre-registered spec.

    If it could also propose, the embargoed years would become a search space
    with nobody counting the trials.
    """
    seal = Seal(phase=Phase.SEALED)
    with pytest.raises(Contamination):
        seal.require(Phase.RESEARCH)


def test_the_sealed_phase_does_not_taint_on_embargoed_data() -> None:
    """Taint is a research-phase concept. The sealed run is *supposed* to read
    those years; what makes it honest is that it cannot search."""
    seal = Seal(phase=Phase.SEALED)
    seal.observe("AAPL", _epoch(AFTER, 10))
    assert seal.tainted is False


# --------------------------------------------------------------------------
# The sensor, in place


def test_constructing_bars_observes_them() -> None:
    with scope(Seal()) as seal:
        _bars("AAPL", BEFORE)
        assert seal.tainted is False
        _bars("AAPL", AFTER)
        assert seal.tainted is True


def test_slicing_bars_does_not_write_a_ledger_row() -> None:
    """``slice`` and ``as_of`` construct new ``Bars`` on every walk-forward fold.

    They must still be observed -- a subset of clean data is clean -- but a fold
    is not a load, and a ledger that counts them cannot be read."""
    with scope(Seal()) as seal:
        bars = _bars("AAPL", BEFORE, n=100)
        bars.slice(0, 50)
        bars.as_of(BEFORE + timedelta(days=10))
        assert seal.loads == ()


def test_the_ambient_seal_is_a_singleton() -> None:
    assert current() is current()


# --------------------------------------------------------------------------
# Boundaries


def _python_files() -> list[Path]:
    return sorted(p for p in SRC.rglob("*.py") if "__pycache__" not in p.parts)


def test_only_the_seal_module_may_swap_the_ambient_seal() -> None:
    """``scope`` exists so tests can isolate. If production code could call it,
    the singleton would be a suggestion."""
    offenders: list[str] = []
    for path in _python_files():
        if path.name == "seal.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
                if name in ("scope", "_install"):
                    offenders.append(f"{path.relative_to(SRC)}:{node.lineno} calls {name}()")
    assert not offenders, "\n".join(offenders)


def test_the_seal_reaches_for_no_io_and_no_clock() -> None:
    """The seal is evidence about a computation and must not be able to influence
    one, nor to depend on when it ran."""
    tree = ast.parse((SRC / "seal.py").read_text(encoding="utf-8"))
    banned = {"sqlite3", "httpx", "requests", "urllib", "socket", "yfinance", "ib_async"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] not in banned, f"seal.py imports {alias.name}"
        elif isinstance(node, ast.ImportFrom) and node.module:
            assert node.module.split(".")[0] not in banned, f"seal.py imports {node.module}"
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in ("now", "utcnow", "today")
        ):
            raise AssertionError(f"seal.py:{node.lineno} reads the clock")


def test_the_embargo_is_a_constant_not_a_parameter_of_the_search() -> None:
    """A configurable embargo is not an embargo. ``Seal`` takes one for testing;
    no other module may pass it."""
    offenders: list[str] = []
    for path in _python_files():
        if path.name == "seal.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                for kw in node.keywords:
                    if kw.arg == "embargo":
                        offenders.append(f"{path.relative_to(SRC)}:{node.lineno}")
    assert not offenders, "\n".join(offenders)


def test_the_seal_does_not_certify_knowledge() -> None:
    """The one thing this machinery cannot do, asserted so it stays documented.

    The seal proves the embargoed *data* was not read. It cannot prove the
    embargoed *period* did not inform a decision: the researcher lived through
    it, and every model in ``providers`` has a training cutoff after it. The
    certificate therefore records the exposure rather than denying it, and any
    claim built on it has to be worded accordingly.
    """
    cert = Seal().certificate()
    assert "knowledge_exposure" in cert
    assert cert["knowledge_exposure"] is None or isinstance(cert["knowledge_exposure"], dict)
