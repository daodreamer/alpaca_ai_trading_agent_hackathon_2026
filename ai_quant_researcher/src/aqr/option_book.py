"""The options handoff: a rule, written to a file, for something else to execute.

[`target_book.py`](target_book.py)'s counterpart and deliberately **not** an
extension of it. specs/10 D10 says why in one line: an option book is a list of
legs, not a vector of weights, so ``TargetBook`` does not describe one and must
not be stretched to. Stretching it would make every consumer branch on which
half of the type was populated, and the consumer here is a risk gate.

The four properties that make the equity artefact worth anything hold here, one
of them differently:

**It comes off the validated code path.** :func:`build_option_book` runs
``run_option_strategy`` — the same entry point the backtest, the walk-forward,
the robustness passes and the sealed run go through — so a book cannot describe
a rule that was never measured, and the counts it carries are that run's.

**The rule, not the positions. And deliberately not the strikes.** A book
written from a Tuesday close naming strike 5480 is wrong by Wednesday's open:
the vendor resamples about 24 rungs around the money every session and the
expiries roll, so a strike is a fact about one snapshot. What travels is what
specs/10 D5 says a rule *is* — a structure kind, a DTE target, an anchor delta,
a width delta, a cadence and a risk fraction — and the executor resolves that
against a live chain. Which is also why the executor needs its own
delta-selection code and cannot import ``aqr.options.chain``: the seam is a
file, in both directions.

**It is self-describing.** Fingerprint, name, session, dataset version (which
now names the price adjustment — specs/10 D0), the seal state it was produced
under, and the sealed-run verdict that justified it. A reader who has never
heard of ``aqr`` can audit where the rule came from without this repository.

**What it will not do is claim more than the evidence.** ``sealed_measurement``
travels with the book and so does the sentence that window is entitled to: about
25 independent cycles can refute a rule and cannot confirm one (D8). No field
here is allowed to word it otherwise, and :data:`CONSUMER_MUST_SUPPLY` carries
the rest of the boundary inside the artefact rather than in a document nobody
reads.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from aqr.options.engine import OptionBacktestConfig
from aqr.options.run import OptionMarket, run_option_strategy
from aqr.options.spec import OptionSpec
from aqr.validation.cycles import independent_cycles

__all__ = [
    "OPTION_BOOK_SCHEMA_VERSION",
    "OPTION_CONSUMER_MUST_SUPPLY",
    "OptionBook",
    "build_option_book",
    "load_option_book",
    "option_book_digest",
    "validate_option_book",
    "write_option_book",
]

OPTION_BOOK_SCHEMA_VERSION = 1
"""Bumped when a field changes meaning, never when one is added.

A consumer pins this. Reading a book with an unknown field is harmless; reading
one where ``anchor_delta`` means something new under the same number is how a
silent execution bug happens — and here it would be a bug that builds a
different structure from the one that was validated.
"""

OPTION_CONSUMER_MUST_SUPPLY = (
    "a live option chain, and the delta-selection that resolves this rule against it",
    "account equity, and the contract counts that follow from it",
    "reconciliation against the structures actually open",
    "an options-shaped risk gate -- defined risk in the type, no naked short legs",
    "position and concurrency limits at the account level, across every sleeve",
    "a kill switch",
    "a fill journal",
    "an exit policy for early assignment, which this research does not model",
)
"""Everything between this file and a placed order.

Carried inside the artefact on purpose. The last entry is the one a reader is
most likely to miss: specs/10 D1 settles European-style, SPY options are
American, and an in-the-money short leg can be assigned before expiry — most
likely around an ex-dividend date. The research is optimistic about exactly that
case, and the executor is the only place it can be handled.
"""


@dataclass(frozen=True, slots=True)
class OptionBook:
    """One rule, as of one session, traceable to the hypothesis that produced it."""

    schema_version: int
    generated_at: str
    spec_fingerprint: str
    spec_name: str
    spec_version: int
    as_of: str
    """The last chain session the rule was run over, as an ISO date."""
    dataset_version: str
    underlying: str
    entry: str
    structure_kind: str
    dte_target: int
    dte_tolerance: int
    anchor_delta: float
    anchor_tolerance: float
    width_delta: float | None
    width_points: float | None
    call_anchor_delta: float | None
    call_width_delta: float | None
    min_sessions_between_entries: int
    risk_per_trade: float
    max_concurrent: int
    evidence: dict[str, Any]
    """What the rule did over the window this book was built from.

    Not a promise and not a target: the trade count, the independent-cycle count
    and the skip census of the run that produced this artefact. An executor that
    opens ten structures a week from a rule whose research produced four cycles
    a year is running something else, and this is the only number in the file
    that lets anyone notice."""
    seal: dict[str, Any]
    provenance: dict[str, Any]
    fill_convention: str
    exit_convention: str
    consumer_must_supply: tuple[str, ...] = OPTION_CONSUMER_MUST_SUPPLY

    def as_dict(self) -> dict[str, Any]:
        """The artefact, exactly as it is written and hashed."""
        return {
            "schema_version": self.schema_version,
            "generated_at": self.generated_at,
            "spec_fingerprint": self.spec_fingerprint,
            "spec_name": self.spec_name,
            "spec_version": self.spec_version,
            "as_of": self.as_of,
            "dataset_version": self.dataset_version,
            "underlying": self.underlying,
            "rule": {
                "entry": self.entry,
                "structure": self.structure_kind,
                "dte": {"target": self.dte_target, "tolerance": self.dte_tolerance},
                "anchor": {
                    "delta": self.anchor_delta,
                    "tolerance": self.anchor_tolerance,
                },
                "width_delta": self.width_delta,
                "width_points": self.width_points,
                "call_anchor_delta": self.call_anchor_delta,
                "call_width_delta": self.call_width_delta,
                "cadence": {"min_sessions_between_entries": self.min_sessions_between_entries},
                "sizing": {
                    "risk_per_trade": self.risk_per_trade,
                    "max_concurrent": self.max_concurrent,
                },
            },
            "evidence": self.evidence,
            "seal": self.seal,
            "provenance": self.provenance,
            "fill_convention": self.fill_convention,
            "exit_convention": self.exit_convention,
            "consumer_must_supply": list(self.consumer_must_supply),
        }

    def summary(self) -> str:
        width = (
            f"width delta {self.width_delta}"
            if self.width_delta is not None
            else f"width {self.width_points} points"
            if self.width_points is not None
            else "one leg, no width"
        )
        return "\n".join(
            [
                f"{self.spec_name} [{self.spec_fingerprint}] as of {self.as_of}",
                f"  {self.structure_kind} on {self.underlying}, "
                f"{self.dte_target}±{self.dte_tolerance} DTE",
                f"  anchor delta {self.anchor_delta}±{self.anchor_tolerance}, {width}",
                f"  entry: {self.entry}",
                f"  every {self.min_sessions_between_entries} sessions at most, "
                f"{self.risk_per_trade:.2%} of equity per structure, "
                f"{self.max_concurrent} concurrent",
                f"  research evidence: {self.evidence.get('trades', 0)} trades, "
                f"{self.evidence.get('independent_cycles', 0)} independent cycles",
                f"  dataset {self.dataset_version}",
            ]
        )


# --------------------------------------------------------------------------- #
# Building
# --------------------------------------------------------------------------- #


def build_option_book(
    spec: OptionSpec,
    market: OptionMarket,
    *,
    generated_at: datetime,
    dataset_version: str,
    provenance: dict[str, Any],
    seal: dict[str, Any],
    config: OptionBacktestConfig | None = None,
) -> OptionBook:
    """Run ``spec`` over ``market`` and describe the rule that was run.

    ``generated_at`` is a parameter rather than a call to the clock, for the same
    reason it is everywhere else here: a value read from the environment makes
    the result of a function depend on when it ran.

    The run is not decoration. It is what makes ``evidence`` a measurement rather
    than a restatement of the spec, and it is what refuses a rule that cannot
    fire: handing off a rule whose entry condition is never true would give an
    executor something that looks like a strategy and does nothing, and the first
    person to notice would be whoever asked why the account had no positions a
    month later.
    """
    result = run_option_strategy(spec, market, config)
    if not market.chain.sessions:
        raise ValueError(f"{spec.name}: the market holds no chain sessions")
    if not result.option_trades:
        raise ValueError(
            f"{spec.name}: the rule opened no structures across "
            f"{len(market.chain.sessions)} sessions -- {result.skip_census}. There is "
            "nothing to hand off, and a book for a rule that cannot fire would read "
            "as a strategy that is merely waiting."
        )

    last_session = market.chain.sessions[-1]
    last_entry = max(t.entry_session for t in result.option_trades)
    structure = spec.structure
    return OptionBook(
        schema_version=OPTION_BOOK_SCHEMA_VERSION,
        generated_at=generated_at.astimezone(UTC).isoformat(),
        spec_fingerprint=spec.fingerprint(),
        spec_name=spec.name,
        spec_version=spec.version,
        as_of=last_session.isoformat(),
        dataset_version=dataset_version,
        underlying=spec.underlying,
        entry=spec.entry,
        structure_kind=structure.type,
        dte_target=structure.dte.target,
        dte_tolerance=structure.dte.tolerance,
        anchor_delta=structure.anchor.delta,
        anchor_tolerance=structure.anchor.tolerance,
        width_delta=structure.width_delta,
        width_points=structure.width_points,
        call_anchor_delta=(
            structure.call_anchor.delta if structure.call_anchor is not None else None
        ),
        call_width_delta=structure.call_width_delta,
        min_sessions_between_entries=spec.cadence.min_sessions_between_entries,
        risk_per_trade=spec.sizing.risk_per_trade,
        max_concurrent=spec.sizing.max_concurrent,
        evidence={
            "sessions": len(market.chain.sessions),
            "first_session": market.chain.sessions[0].isoformat(),
            "trades": len(result.option_trades),
            "independent_cycles": independent_cycles(result.trades),
            "last_entry_session": last_entry.isoformat(),
            "skip_census": result.skip_census.as_dict(),
            # D8a, carried rather than left for a reader to recompute: a rule
            # whose research was mostly affordability-skipped will trade far
            # more often in an account that can afford it than the cycle count
            # above suggests, and the executor is the one sizing it.
            "risk_per_trade_researched": spec.sizing.risk_per_trade,
        },
        seal=dict(seal),
        provenance=dict(provenance),
        fill_convention=(
            "Decide at the close of session t-1; select and fill from session t's "
            "chain, buying at the ask and selling at the bid, per leg, spread "
            "crossed in full. The research charges no better fill than that and an "
            "executor should not assume one."
        ),
        exit_convention=(
            "Held to expiry. There is no stop, no profit target, no roll and no "
            "management rule, because the research data cannot price one: a "
            "specific contract is re-quoted on 1-3% of later sessions (specs/10 "
            "D0, D1). Settlement is modelled European-style against the "
            "underlying's close on the expiration date; SPY options are American, "
            "so early assignment is real, is not modelled, and belongs to the "
            "executor."
        ),
    )


# --------------------------------------------------------------------------- #
# The file
# --------------------------------------------------------------------------- #


def _canonical(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, indent=2, default=str)


def option_book_digest(payload: dict[str, Any]) -> str:
    """SHA-256 over the canonical serialisation, so a file can be checked later.

    The registry stores this. A book on disk that no longer hashes to what was
    recorded has been edited since it was handed off, and being able to say so is
    the reason for recording it.
    """
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


def write_option_book(book: OptionBook, path: Path) -> str:
    """Write the artefact and return its digest."""
    payload = book.as_dict()
    validate_option_book(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_canonical(payload) + "\n", encoding="utf-8")
    return option_book_digest(payload)


def load_option_book(path: Path) -> dict[str, Any]:
    """Read an artefact back, validated. Raises rather than returning a bad book."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_option_book(payload)
    return dict(payload)


_REQUIRED: dict[str, tuple[type, ...]] = {
    "schema_version": (int,),
    "generated_at": (str,),
    "spec_fingerprint": (str,),
    "spec_name": (str,),
    "spec_version": (int,),
    "as_of": (str,),
    "dataset_version": (str,),
    "underlying": (str,),
    "rule": (dict,),
    "evidence": (dict,),
    "seal": (dict,),
    "provenance": (dict,),
    "fill_convention": (str,),
    "exit_convention": (str,),
    "consumer_must_supply": (list,),
}

_REQUIRED_RULE: tuple[str, ...] = (
    "entry",
    "structure",
    "dte",
    "anchor",
    "cadence",
    "sizing",
)

# Strikes, expiries, contract counts and prices are the executor's to resolve
# against a live chain. A field naming one would be this project deciding what
# to trade rather than what rule to trade, and it would be stale by the next
# open -- which is the specific failure specs/10 D5's delta selection exists to
# avoid.
_FORBIDDEN_FIELDS = frozenset(
    {
        "strike",
        "strikes",
        "expiration",
        "expirations",
        "contracts",
        "quantity",
        "qty",
        "notional",
        "premium",
        "limit_price",
        "equity",
        "account",
        "orders",
    }
)

_UNBOUNDED_KINDS = frozenset({"naked_put", "naked_call", "short_put", "short_call", "custom"})
"""Not structures this project can produce -- ``StructureKind`` has no such
member -- which is exactly why they are checked here. This validator also reads
books it did not write, and a hand-edited artefact naming one of these would ask
an executor to open a position with unbounded loss. Refusing at the file
boundary costs nothing and is the last place before the gate."""


def validate_option_book(payload: dict[str, Any]) -> None:
    """Check an artefact against the schema, reporting every fault at once.

    Checked on write as well as on read. A malformed book that reaches disk is a
    book something downstream will try to execute, and the cheapest place to stop
    it is before it is written.

    Every fault at once, not the first: a consumer debugging a rejected book
    should need one round trip, and the equity ``validate_book`` set that
    precedent for the same reason.
    """
    faults: list[str] = []
    for name, kinds in _REQUIRED.items():
        if name not in payload:
            faults.append(f"missing {name!r}")
            continue
        value = payload[name]
        if isinstance(value, bool) or not isinstance(value, kinds):
            faults.append(f"{name!r} is {type(value).__name__}, expected {kinds[0].__name__}")
    if faults:
        raise ValueError("; ".join(faults))

    if payload["schema_version"] != OPTION_BOOK_SCHEMA_VERSION:
        faults.append(
            f"schema_version {payload['schema_version']} is not "
            f"{OPTION_BOOK_SCHEMA_VERSION}; this reader cannot know what its fields mean"
        )
    try:
        datetime.fromisoformat(payload["as_of"])
    except ValueError:
        faults.append(f"as_of {payload['as_of']!r} is not an ISO date")

    rule = payload["rule"]
    for name in _REQUIRED_RULE:
        if name not in rule:
            faults.append(f"rule is missing {name!r}")
    kind = str(rule.get("structure", ""))
    if kind in _UNBOUNDED_KINDS:
        faults.append(
            f"rule.structure {kind!r} has unbounded loss and cannot be represented "
            "here; specs/10 D4 lists every structure this project may produce"
        )
    if not str(rule.get("entry", "")).strip():
        faults.append(
            "rule.entry is empty: a rule with no condition opens a position every "
            "session it can, which is a schedule rather than a hypothesis"
        )
    sizing = rule.get("sizing")
    if isinstance(sizing, dict):
        risk = sizing.get("risk_per_trade")
        if isinstance(risk, bool) or not isinstance(risk, (int, float)):
            faults.append("rule.sizing.risk_per_trade is not a number")
        elif not 0 < float(risk) <= 0.1:
            faults.append(
                f"rule.sizing.risk_per_trade {risk} is outside (0, 0.1]; the research "
                "never measured this rule at that size"
            )

    present = _FORBIDDEN_FIELDS & (set(payload) | set(rule))
    if present:
        faults.append(
            f"{sorted(present)}: an option book carries the rule only -- strikes and "
            "sizes are resolved against a live chain by whatever executes it, "
            "because a strike named from yesterday's close is wrong by today's open"
        )
    if faults:
        raise ValueError("; ".join(faults))
