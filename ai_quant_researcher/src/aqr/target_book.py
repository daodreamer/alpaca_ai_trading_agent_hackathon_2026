"""The handoff: a target book, written to a file, for something else to execute.

**This project does not place orders, and this module is where that decision is
enforced rather than merely stated.** What comes out of here is a description of
a book -- weights per symbol, as of a session, under a named spec -- and nothing
in this package sends it anywhere. The consumer reads the file. Neither side
imports the other, and the file is the whole interface.

Four properties are what make the artefact worth anything.

**It comes off the validated code path.** :func:`build_target_book` calls
``run_strategy``, the same entry point the backtest, the walk-forward, the
robustness passes and the sealed run all go through. The handoff and the
backtest differ in where the data stops and in nothing else, so a book cannot
describe a strategy that was never measured.
``test_the_book_matches_what_the_backtest_held`` fails the build if they ever
diverge.

**Weights, not shares and not notionals.** This project does not know the
account's equity, and inventing one would be the first step towards it placing
the order. Sizing belongs to whatever executes, along with everything else in
:data:`CONSUMER_MUST_SUPPLY` -- recorded inside the artefact so the boundary
travels with the file instead of living in a document nobody reads.

**It is self-describing.** Fingerprint, spec name, session, dataset version,
universe, the seal state it was produced under, and the sealed-run verdict that
justified it. A reader that has never heard of ``aqr`` can audit where the book
came from without this repository.

**One session of lag, named rather than hidden.** The engine decides at a close
and fills at the next open, so the weights in force on the final session were
decided on the session before it. A book produced from data through session T
therefore carries the decision made at T-1, and an executor placing it at the
next open acts one session later than the backtest did. Reporting the decision
made *at* T instead would mean re-implementing the selection outside the engine,
which is the one thing this module exists not to do.

**On the seal.** Producing a book means reading the present, so the process that
produces one is tainted by construction and its certificate says so. That is
correct rather than a defect: the taint records "this process read past the
embargo", which is the entire job here. What the process must never also be is
the search -- and it is not, because a book is only written for a candidate
whose seal has already been spent, so there is no unspent answer left to leak.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from aqr.backtest.engine import BacktestConfig
from aqr.backtest.portfolio import PortfolioResult
from aqr.backtest.run import run_strategy
from aqr.data.bars import Bars
from aqr.data.universes import PointInTimeUniverse
from aqr.dsl.schema import StrategySpec

__all__ = [
    "BOOK_SCHEMA_VERSION",
    "CONSUMER_MUST_SUPPLY",
    "TargetBook",
    "book_digest",
    "build_target_book",
    "load_book",
    "validate_book",
    "write_book",
]

BOOK_SCHEMA_VERSION = 1
"""Bumped when a field changes meaning, never when one is added.

A consumer pins this. Reading a book with an unknown field is harmless; reading
one where ``weights`` means something new under the same number is how a silent
execution bug happens.
"""

CONSUMER_MUST_SUPPLY = (
    "account equity, and the share counts that follow from it",
    "reconciliation against the positions actually held",
    "turnover and notional caps",
    "an equity-shaped risk gate -- AlphaGate's is options-shaped and does not apply",
    "a kill switch",
    "a fill journal",
)
"""Everything between this file and a placed order.

Carried inside the artefact on purpose. A boundary recorded only in a design
document is one the first person to consume the file will not see.
"""


@dataclass(frozen=True, slots=True)
class TargetBook:
    """One book, as of one session, traceable to the hypothesis that produced it."""

    schema_version: int
    generated_at: str
    spec_fingerprint: str
    spec_name: str
    spec_version: int
    as_of: str
    """The session the weights were in force on, as an ISO date. Not "today"."""
    as_of_event_time: int
    dataset_version: str
    universe: str
    timeframe: str
    symbols_loaded: int
    symbols_declared: int
    weights: dict[str, float]
    core_weights: dict[str, float]
    sleeve_weights: dict[str, float]
    seal: dict[str, Any]
    provenance: dict[str, Any]
    fill_convention: str
    consumer_must_supply: tuple[str, ...] = CONSUMER_MUST_SUPPLY

    @property
    def gross(self) -> float:
        """Total target exposure. Roughly 1.0 for an always-invested book."""
        return float(sum(self.weights.values()))

    def as_dict(self) -> dict[str, Any]:
        """The artefact, exactly as it is written and hashed."""
        return {
            "schema_version": self.schema_version,
            "generated_at": self.generated_at,
            "spec_fingerprint": self.spec_fingerprint,
            "spec_name": self.spec_name,
            "spec_version": self.spec_version,
            "as_of": self.as_of,
            "as_of_event_time": self.as_of_event_time,
            "dataset_version": self.dataset_version,
            "universe": self.universe,
            "timeframe": self.timeframe,
            "symbols_loaded": self.symbols_loaded,
            "symbols_declared": self.symbols_declared,
            "gross": self.gross,
            "positions": len(self.weights),
            "weights": dict(sorted(self.weights.items())),
            "core_weights": dict(sorted(self.core_weights.items())),
            "sleeve_weights": dict(sorted(self.sleeve_weights.items())),
            "seal": self.seal,
            "provenance": self.provenance,
            "fill_convention": self.fill_convention,
            "consumer_must_supply": list(self.consumer_must_supply),
        }

    def summary(self) -> str:
        core = sorted(self.core_weights.items(), key=lambda kv: (-kv[1], kv[0]))
        lines = [
            f"{self.spec_name} [{self.spec_fingerprint}] as of {self.as_of}",
            f"  {len(self.weights)} positions, gross {self.gross:.4f} "
            f"(core {sum(self.core_weights.values()):.4f}, "
            f"sleeve {sum(self.sleeve_weights.values()):.4f})",
            f"  dataset {self.dataset_version}",
            "  core:",
        ]
        lines += [f"    {symbol:<8} {weight:.6f}" for symbol, weight in core]
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Building
# --------------------------------------------------------------------------- #


def build_target_book(
    spec: StrategySpec,
    data: dict[str, Bars],
    *,
    generated_at: datetime,
    dataset_version: str,
    universe: str,
    provenance: dict[str, Any],
    seal: dict[str, Any],
    config: BacktestConfig | None = None,
    membership: PointInTimeUniverse | None = None,
    peers: dict[str, Bars] | None = None,
) -> TargetBook:
    """Run ``spec`` over ``data`` and take the weights in force on the last session.

    ``generated_at`` is a parameter rather than a call to the clock, for the same
    reason it is everywhere else here: a value read from the environment makes
    the result of a function depend on when it ran.

    Refuses a signal-mode spec. A trigger strategy has no target book -- its
    positions are opened by events and closed by stops, so "the weights it holds"
    is not a quantity it defines, and inventing one would hand off a portfolio
    interpretation of a strategy nobody validated as a portfolio.
    """
    if spec.mode != "portfolio":
        raise ValueError(
            f"{spec.name}: only a portfolio spec has a target book. A signal spec "
            "opens positions on events, so its book is a consequence of fills "
            "rather than a set of weights."
        )

    result = run_strategy(spec, data, config, peers=peers, membership=membership)
    if not isinstance(result, PortfolioResult):  # pragma: no cover - run_strategy pins this
        raise TypeError(f"expected a PortfolioResult, got {type(result).__name__}")
    if result.timeline.size == 0:
        raise ValueError(f"{spec.name}: the run produced no sessions")
    if result.first_fill_step is None:
        raise ValueError(
            f"{spec.name}: the run never filled -- {result.timeline.size} sessions is "
            f"not enough to clear a warm-up of {result.warmup_bars}. There is no book "
            "to hand off, and an empty one would read as 'hold nothing'."
        )

    last = int(result.timeline.size) - 1
    session = datetime.fromtimestamp(int(result.timeline[last]), tz=UTC)
    return TargetBook(
        schema_version=BOOK_SCHEMA_VERSION,
        generated_at=generated_at.astimezone(UTC).isoformat(),
        spec_fingerprint=spec.fingerprint(),
        spec_name=spec.name,
        spec_version=spec.version,
        as_of=session.date().isoformat(),
        as_of_event_time=int(result.timeline[last]),
        dataset_version=dataset_version,
        universe=universe,
        timeframe=spec.universe.timeframe,
        symbols_loaded=len(result.symbols),
        symbols_declared=len(spec.universe.symbols),
        weights=dict(sorted(result.weights_at(last).items())),
        core_weights=dict(sorted(result.core_weights_at(last).items())),
        sleeve_weights=dict(sorted(result.sleeve_weights_at(last).items())),
        seal=dict(seal),
        provenance=dict(provenance),
        fill_convention=(
            "Weights in force on as_of: decided at the close of the session before "
            "it and filled at the open of as_of. An executor placing them at the "
            "next open is one session behind the backtest, which is the cost of "
            "not re-implementing the selection outside the engine."
        ),
    )


# --------------------------------------------------------------------------- #
# The file
# --------------------------------------------------------------------------- #


def _canonical(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, indent=2, default=str)


def book_digest(payload: dict[str, Any]) -> str:
    """SHA-256 over the canonical serialisation, so a file can be checked later.

    The registry stores this. A book on disk that no longer hashes to what was
    recorded has been edited since it was handed off, and being able to say so is
    the reason for recording it.
    """
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


def write_book(book: TargetBook, path: Path) -> str:
    """Write the artefact and return its digest.

    Overwriting is safe here in a way it is not for the sealed run: a book is a
    description of a session, regenerable from the same inputs, not a measurement
    that can only be taken once.
    """
    payload = book.as_dict()
    validate_book(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_canonical(payload) + "\n", encoding="utf-8")
    return book_digest(payload)


def load_book(path: Path) -> dict[str, Any]:
    """Read an artefact back, validated. Raises rather than returning a bad book."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_book(payload)
    return dict(payload)


_REQUIRED: dict[str, tuple[type, ...]] = {
    "schema_version": (int,),
    "generated_at": (str,),
    "spec_fingerprint": (str,),
    "spec_name": (str,),
    "spec_version": (int,),
    "as_of": (str,),
    "as_of_event_time": (int,),
    "dataset_version": (str,),
    "universe": (str,),
    "timeframe": (str,),
    "symbols_loaded": (int,),
    "symbols_declared": (int,),
    "gross": (int, float),
    "positions": (int,),
    "weights": (dict,),
    "core_weights": (dict,),
    "sleeve_weights": (dict,),
    "seal": (dict,),
    "provenance": (dict,),
    "fill_convention": (str,),
    "consumer_must_supply": (list,),
}

# Shares, notionals and quantities are the consumer's to compute. A field naming
# one would be this project sizing a position, which is the line the whole
# handoff exists to keep on the other side.
_FORBIDDEN_FIELDS = frozenset(
    {"shares", "quantity", "qty", "notional", "equity", "account", "orders"}
)


def validate_book(payload: dict[str, Any]) -> None:
    """Check an artefact against the schema, reporting every fault at once.

    Checked on write as well as on read. A malformed book that reaches disk is a
    book something downstream will try to execute, and the cheapest place to stop
    it is before it is written.
    """
    faults: list[str] = []
    for field, kinds in _REQUIRED.items():
        if field not in payload:
            faults.append(f"missing {field!r}")
            continue
        value = payload[field]
        if isinstance(value, bool) or not isinstance(value, kinds):
            faults.append(
                f"{field!r} is {type(value).__name__}, expected {kinds[0].__name__}"
            )
    if faults:
        raise ValueError("; ".join(faults))

    if payload["schema_version"] != BOOK_SCHEMA_VERSION:
        faults.append(
            f"schema_version {payload['schema_version']} is not "
            f"{BOOK_SCHEMA_VERSION}; this reader cannot know what its fields mean"
        )
    try:
        datetime.fromisoformat(payload["as_of"])
    except ValueError:
        faults.append(f"as_of {payload['as_of']!r} is not an ISO date")

    for field in ("weights", "core_weights", "sleeve_weights"):
        for symbol, weight in payload[field].items():
            if not isinstance(symbol, str) or not symbol.strip():
                faults.append(f"{field}: {symbol!r} is not a symbol")
            if isinstance(weight, bool) or not isinstance(weight, (int, float)):
                faults.append(f"{field}[{symbol}] is {type(weight).__name__}, not a number")

    if payload["positions"] != len(payload["weights"]):
        faults.append(
            f"positions {payload['positions']} disagrees with "
            f"{len(payload['weights'])} weights"
        )
    present = _FORBIDDEN_FIELDS & set(payload)
    if present:
        faults.append(
            f"{sorted(present)}: a target book carries weights only -- sizing "
            "belongs to whatever executes it"
        )
    if faults:
        raise ValueError("; ".join(faults))
