"""The option book, as AlphaGate is allowed to believe it — specs/07 D1.

PURE. Stdlib plus `alphagate.core` and `alphagate.options`. No I/O:
`load_option_book` takes a parsed mapping, never a path, so the validation is
testable without a filesystem and identical whether the bytes came from disk, a
fixture or a socket.

The sibling of [`equity/book.py`](../equity/book.py) and deliberately not an
extension of it, for the reason `ai_quant_researcher/src/aqr/option_book.py`
gives from the other side: an option book carries a *rule*, not a vector of
weights, and stretching one type over both would make every consumer branch on
which half was populated. The consumer here is a risk gate.

`ai_quant_researcher` writes this artefact and stops. It does not import
`alphagate` and `alphagate` does not import it — the file is the whole interface
(CLAUDE.md §2b, specs/09 D0). Which means this module is where an unvalidated
rule would get in, and so it is the module with the least freedom.

**The fingerprint is pinned by the operator, not read from the book.** Same
argument as the equity side: a book names the rule it describes, and if that
name were believed, replacing the file would replace the strategy and nothing
would notice. `ALPHAGATE_OPTION_FINGERPRINT` has no default, exactly as
`ALPHAGATE_STRATEGY_FINGERPRINT` has none.

**There are no strikes in a book, on purpose.** A book written from a Tuesday
close naming strike 5480 is wrong by Wednesday's open. What travels is the rule
— structure kind, DTE target, anchor delta, width delta, cadence, sizing — and
the executor resolves it against a live chain. That is why this package has its
own delta selection and cannot import the researcher's.

**A rule naming a feature this account cannot measure is refused, not
approximated.** `EntryRule` parses the book's entry expression against the
features a `MarketRead` actually carries, and a book that names anything else is
unusable. The alternative is to substitute a near-enough number for the one the
research conditioned on, which would execute a different rule under a
fingerprint that certifies this one. See `agent/iv_store.py` for why `iv_rank`
in particular is the one that bites.

Money is `Decimal`, end to end: `risk_per_trade` arrives as a JSON float and
becomes `Decimal` here, at the boundary, via `str`. The greeks stay `float` —
they are estimates, not money (CLAUDE.md §3.4).
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Final

from alphagate.core.errors import InvariantViolation
from alphagate.core.identifiers import Ticker, ticker

__all__ = [
    "MEASURABLE_FEATURES",
    "OPTION_BOOK_SCHEMA_VERSION",
    "TRADEABLE_STATUS",
    "EntryRule",
    "OptionBook",
    "OptionRule",
    "SealedOptionRun",
    "UnusableOptionBook",
    "entry_refusal",
    "load_option_book",
    "measurable_read",
]

OPTION_BOOK_SCHEMA_VERSION: Final = 1
"""The only schema this reader understands.

`aqr` bumps it when a field changes meaning, never when one is added — so an
unknown *field* is harmless and an unknown *version* is fatal. Reading a book
whose `rule` means something new under the same number is the silent execution
bug this pin exists to prevent."""

TRADEABLE_STATUS: Final = frozenset({"PAPER", "LIVE"})
"""Registry lifecycle states that may be executed.

`CANDIDATE → PAPER → LIVE` and nothing skips a step. Reading the state here
means that rule is enforced on the execution side too, rather than trusted to
have been enforced upstream."""

DEFINED_RISK_STRUCTURES: Final = frozenset(
    {"put_credit_spread", "call_credit_spread", "iron_condor"}
)
"""The structures this executor builds, and every one of them is defined risk.

Narrower than the researcher's whitelist on purpose. `long_put`, `long_call`,
`put_debit_spread` and `call_debit_spread` are all defined risk too and are all
refused here, because `agent/candidates.py` builds vertical *credit* spreads and
a book asking for a structure the executor cannot construct must say so at load
time rather than produce an empty menu every cycle and look like a quiet market.

Nothing with an uncovered short leg appears, and nothing could: specs/02 D3
makes such a structure unrepresentable in the type system, so a book naming one
is refused twice."""

MEASURABLE_FEATURES: Final = frozenset(
    {"iv_rank", "iv_percentile", "iv_vs_hv", "hv_rank", "atr_pct"}
)
"""Features an entry expression may name.

Exactly the fields `MarketRead` carries and this account can compute. The
researcher's vocabulary is larger — it had a seven-year vendor volatility
history and a full chain per session — and the gap is the point: a rule
conditioned on something this executor cannot measure is refused rather than
executed against a substitute.

`iv_rank` is in the list because `MarketRead` has the field, NOT because the
field is always populated. It is `None` until the IV store holds
`options.volatility.MIN_HISTORY` sessions, and an entry that cannot be decided
declines — see `EntryRule.decide`."""

_UNREACHABLE_TIME: Final = datetime.min.replace(tzinfo=UTC)
"""The placeholder returned alongside a recorded fault.

Never reaches a caller — `load_option_book` raises whenever `faults` is
non-empty — but it is tz-aware anyway, because CLAUDE.md rule 5 is end to end
and a naive datetime that only escapes on an impossible path is exactly the one
nobody notices."""

_COMPARISONS: Final = {
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
}

_CLAUSE = re.compile(
    r"^\s*(?P<feature>[a-z_][a-z0-9_]*)\s*\(\s*\)\s*"
    r"(?P<op><=|>=|<|>)\s*"
    r"(?P<threshold>-?\d+(?:\.\d+)?)\s*$"
)
"""One clause: `feature() OP number`.

Deliberately not a general expression grammar. The researcher's DSL has one and
this is not a second implementation of it — it is the subset an option book has
ever contained, and anything outside it is refused by name. A parser that
accepted more than it could faithfully evaluate would be the failure this whole
module exists to prevent."""


class UnusableOptionBook(InvariantViolation):
    """An option book that must not be executed, and every reason why.

    Every fault is collected before raising. A book wrong in three ways should
    be reported three times and fixed once, not discovered three mornings
    running.
    """


@dataclass(frozen=True, slots=True)
class EntryRule:
    """The book's entry condition, parsed into clauses this executor can decide.

    Clauses are joined by `and` only. `or` is refused at parse time rather than
    supported: no option book has contained one, and a disjunction quietly
    mis-associated against an `and` is a different rule that still loads.
    """

    expression: str
    clauses: tuple[tuple[str, str, float], ...]

    def features(self) -> frozenset[str]:
        return frozenset(feature for feature, _, _ in self.clauses)

    def decide(self, read: Mapping[str, Decimal | None]) -> tuple[bool, str]:
        """`(fires, why)` for one reading.

        Returns `False` with a reason when any feature the rule names is
        unmeasured. That is specs/05 D6 applied to the entry itself: an
        undecidable condition is not a false one and is certainly not a true
        one, and the journal gets the distinction in words.
        """
        unmeasured = sorted(
            feature for feature, _, _ in self.clauses if read.get(feature) is None
        )
        if unmeasured:
            return False, (
                f"{', '.join(unmeasured)} unmeasured, so `{self.expression}` cannot be "
                "decided; standing aside rather than guessing at the number the rule "
                "was researched on"
            )

        failed: list[str] = []
        for feature, op, threshold in self.clauses:
            value = read[feature]
            if value is None:  # pragma: no cover - `unmeasured` returned above
                continue
            if not _COMPARISONS[op](float(value), threshold):
                failed.append(f"{feature}()={float(value):.4g} not {op} {threshold:g}")
        if failed:
            return False, "; ".join(failed)
        return True, f"`{self.expression}` holds"


@dataclass(frozen=True, slots=True)
class OptionRule:
    """The rule the research validated, as the executor needs it.

    No strikes and no expiration date — those are resolved against a live chain.
    Deltas are `float` because they are estimates; `risk_per_trade` is `Decimal`
    because it is about to be multiplied by equity.
    """

    underlying: Ticker
    structure: str
    entry: EntryRule
    dte_target: int
    dte_tolerance: int
    anchor_delta: float
    anchor_tolerance: float
    width_delta: float
    min_sessions_between_entries: int
    risk_per_trade: Decimal
    max_concurrent: int

    def dte_window(self) -> tuple[int, int]:
        """The expiry range a chain request should cover, floored at 1.

        0DTE stays excluded whatever the book says: specs/03 excludes it, and a
        tolerance that reached zero would let a book re-admit it silently.
        """
        low = max(1, self.dte_target - self.dte_tolerance)
        return low, self.dte_target + self.dte_tolerance


@dataclass(frozen=True, slots=True)
class SealedOptionRun:
    """What the pre-registered out-of-sample run measured.

    Carried for the record and for the dashboard — never acted on. Nothing here
    sizes a position or gates an order: the sealed window decided whether this
    rule may run at all, and that decision was made before this process started.

    `can_confirm` is `False` by construction upstream and is kept under that
    name rather than flattened into a verdict, because "was not refuted" and
    "was confirmed" are different claims and only one of them is available. The
    option window is worth about 25 independent cycles (specs/10 D8), so no
    artefact reading this may word it as validation.
    """

    strategy_return: float
    strategy_sharpe: float
    benchmark_sharpe: float
    max_drawdown: float
    trades: int
    observations: int
    alpha: float
    beta: float
    t_alpha: float
    significance_bar: float
    is_significant: bool
    refuted: bool
    can_confirm: bool
    first_session: str
    last_session: str
    looks: int
    note: str


@dataclass(frozen=True, slots=True)
class OptionBook:
    """A validated handoff. Existence means every refusal below was survived."""

    fingerprint: str
    name: str
    version: int
    as_of: date
    generated_at: datetime
    dataset_version: str
    rule: OptionRule
    status: str
    hypothesis: str
    selection_rule: str
    distinct_hypotheses: int
    campaign_hypotheses: int
    exit_convention: str
    sealed: SealedOptionRun
    digest: str

    @property
    def underlying(self) -> Ticker:
        return self.rule.underlying


def load_option_book(
    payload: Mapping[str, Any],
    *,
    pinned_fingerprint: str,
    digest: str,
) -> OptionBook:
    """Validate a parsed option book. Pure.

    `digest` is computed by the caller over the bytes it read, because hashing a
    re-serialised mapping would hash this process's JSON formatting rather than
    the file.

    Faults are accumulated and raised together as an `InvariantViolation`
    subclass: an unusable book is not a recoverable condition to be retried, it
    is a statement that the thing upstream produced must not be executed.
    """
    faults: list[str] = []

    version = payload.get("schema_version")
    if version != OPTION_BOOK_SCHEMA_VERSION:
        faults.append(
            f"schema_version is {version!r}, not {OPTION_BOOK_SCHEMA_VERSION}; this "
            "reader cannot know what its fields mean"
        )

    fingerprint = str(payload.get("spec_fingerprint", ""))
    if not fingerprint:
        faults.append("no spec_fingerprint")
    elif fingerprint != pinned_fingerprint:
        faults.append(
            f"book is for {fingerprint}, and the pinned rule is {pinned_fingerprint}. "
            "Only the pinned fingerprint may be executed — see specs/07 D1"
        )

    provenance = payload.get("provenance")
    provenance = provenance if isinstance(provenance, Mapping) else {}

    status = str(provenance.get("status", ""))
    if status not in TRADEABLE_STATUS:
        faults.append(
            f"registry status is {status!r}; only {sorted(TRADEABLE_STATUS)} have "
            "earned a paper position"
        )

    looks = _int(provenance.get("sealed_look"))
    if looks < 1:
        faults.append(
            "the seal is unspent (sealed_look < 1); this rule still owes an "
            "out-of-sample verdict and must not be executed before it has one"
        )

    measurement = provenance.get("sealed_measurement")
    measurement = measurement if isinstance(measurement, Mapping) else {}
    if not measurement:
        faults.append("no sealed_measurement; nothing recorded what the seal bought")
    elif bool(measurement.get("refuted")):
        faults.append("the sealed window refuted this rule")

    rule = _rule(payload, faults=faults)
    as_of = _date(payload.get("as_of"), faults=faults)
    generated_at = _timestamp(payload.get("generated_at"), faults=faults)

    if faults:
        raise UnusableOptionBook(
            "this option book must not be executed:\n  - " + "\n  - ".join(faults)
        )
    if rule is None:  # pragma: no cover - `_rule` always records a fault first
        raise UnusableOptionBook(
            "this option book must not be executed: the rule could not be read"
        )

    return OptionBook(
        fingerprint=fingerprint,
        name=str(payload.get("spec_name", "")),
        version=_int(payload.get("spec_version")),
        as_of=as_of,
        generated_at=generated_at,
        dataset_version=str(payload.get("dataset_version", "")),
        rule=rule,
        status=status,
        hypothesis=str(provenance.get("hypothesis", "")),
        selection_rule=_selection_rule(provenance),
        distinct_hypotheses=_int(provenance.get("distinct_option_hypotheses")),
        campaign_hypotheses=_int(provenance.get("campaign_hypotheses")),
        exit_convention=str(payload.get("exit_convention", "")),
        sealed=_sealed(measurement, looks=_int(provenance.get("sealed_looks_total")) or looks),
        digest=digest,
    )


# --------------------------------------------------------------------------- #
# The rule
# --------------------------------------------------------------------------- #


def _rule(payload: Mapping[str, Any], *, faults: list[str]) -> OptionRule | None:
    raw = payload.get("rule")
    if not isinstance(raw, Mapping):
        faults.append("no rule object; a book with no rule describes nothing")
        return None

    structure = str(raw.get("structure", ""))
    if structure not in DEFINED_RISK_STRUCTURES:
        faults.append(
            f"structure {structure!r} is not one this executor builds "
            f"({sorted(DEFINED_RISK_STRUCTURES)}); refusing at load rather than "
            "producing an empty menu every cycle"
        )

    if raw.get("width_points") is not None:
        faults.append(
            "width_points is set; this executor selects the protective leg by delta "
            "(specs/07 D5). A points width resolves on 23% of sessions and would "
            "silently change which structure is opened"
        )

    anchor = raw.get("anchor")
    anchor = anchor if isinstance(anchor, Mapping) else {}
    anchor_delta = _float(anchor.get("delta"), field="rule.anchor.delta", faults=faults)
    anchor_tolerance = _float(
        anchor.get("tolerance"), field="rule.anchor.tolerance", faults=faults
    )
    width_delta = _float(raw.get("width_delta"), field="rule.width_delta", faults=faults)

    if not 0.0 < anchor_delta < 1.0:
        faults.append(
            f"anchor delta {anchor_delta} is not a magnitude in (0, 1); a put's delta "
            "is negative in the data and the rule names the magnitude"
        )
    if not 0.0 < width_delta < 1.0:
        faults.append(f"width delta {width_delta} is not a magnitude in (0, 1)")
    elif width_delta >= anchor_delta:
        # The whole of "defined risk" is here. A protective leg at or inside the
        # anchor is not protection, and the structure it describes has an
        # uncovered short leg however the book labels it.
        faults.append(
            f"width delta {width_delta} is not strictly less than anchor delta "
            f"{anchor_delta}; the protective leg must be further out of the money or "
            "the structure is not defined risk (CLAUDE.md §3.6)"
        )

    dte = raw.get("dte")
    dte = dte if isinstance(dte, Mapping) else {}
    dte_target = _int(dte.get("target"))
    dte_tolerance = _int(dte.get("tolerance"))
    if dte_target <= 0:
        faults.append(f"dte target {dte_target} is not a positive number of days")
    if dte_tolerance < 0:
        faults.append(f"dte tolerance {dte_tolerance} is negative")

    cadence = raw.get("cadence")
    cadence = cadence if isinstance(cadence, Mapping) else {}
    spacing = _int(cadence.get("min_sessions_between_entries"))
    if spacing < 1:
        faults.append(
            f"cadence {spacing} is under one session; the research measured a rule "
            "that enters at most once per session"
        )

    sizing = raw.get("sizing")
    sizing = sizing if isinstance(sizing, Mapping) else {}
    risk = _decimal(sizing.get("risk_per_trade"), field="rule.sizing.risk_per_trade", faults=faults)
    if not Decimal(0) < risk <= Decimal("0.05"):
        # specs/10 D8a: the same rule produced 21 independent cycles at 1% of
        # equity and 57 at 2%. Sizing is part of the experiment, so a book
        # carrying a fraction the research never ran at is refused rather than
        # clamped.
        faults.append(
            f"risk_per_trade {risk} is outside (0, 0.05]; the sealed run measured this "
            "rule at one size and a different one is a different experiment "
            "(specs/10 D8a)"
        )
    concurrent = _int(sizing.get("max_concurrent"))
    if concurrent < 1:
        faults.append(f"max_concurrent {concurrent} admits no position at all")

    entry = _entry(raw.get("entry"), faults=faults)

    symbol = str(payload.get("underlying", ""))
    try:
        underlying = ticker(symbol)
    except (InvariantViolation, ValueError):
        faults.append(f"underlying {symbol!r} is not a usable ticker")
        return None

    if entry is None:
        return None
    return OptionRule(
        underlying=underlying,
        structure=structure,
        entry=entry,
        dte_target=dte_target,
        dte_tolerance=dte_tolerance,
        anchor_delta=anchor_delta,
        anchor_tolerance=anchor_tolerance,
        width_delta=width_delta,
        min_sessions_between_entries=spacing,
        risk_per_trade=risk,
        max_concurrent=concurrent,
    )


def _entry(raw: Any, *, faults: list[str]) -> EntryRule | None:
    expression = str(raw or "").strip()
    if not expression:
        faults.append(
            "no entry condition; a rule with none opens a position every session it "
            "can, which is a schedule rather than a hypothesis"
        )
        return None

    lowered = expression.lower()
    if " or " in lowered or lowered.startswith("not ") or " not " in lowered:
        faults.append(
            f"entry {expression!r} uses `or`/`not`; this reader joins clauses with "
            "`and` only and refuses rather than risk mis-associating a disjunction"
        )
        return None

    clauses: list[tuple[str, str, float]] = []
    unknown: list[str] = []
    for part in re.split(r"\band\b", expression, flags=re.IGNORECASE):
        match = _CLAUSE.match(part)
        if match is None:
            faults.append(
                f"entry clause {part.strip()!r} is not `feature() OP number`; this "
                "executor does not reimplement the researcher's expression language "
                "and refuses what it cannot faithfully evaluate"
            )
            continue
        feature = match.group("feature")
        if feature not in MEASURABLE_FEATURES:
            unknown.append(feature)
            continue
        clauses.append((feature, match.group("op"), float(match.group("threshold"))))

    if unknown:
        faults.append(
            f"entry names {sorted(set(unknown))}, which this account cannot measure "
            f"(it carries {sorted(MEASURABLE_FEATURES)}). A rule conditioned on a "
            "feature the executor does not have would be executed against a "
            "substitute, under a fingerprint certifying the original"
        )
        return None
    if not clauses:
        return None
    return EntryRule(expression=expression, clauses=tuple(clauses))


# --------------------------------------------------------------------------- #
# Coercion at the boundary
# --------------------------------------------------------------------------- #


def _sealed(measurement: Mapping[str, Any], *, looks: int) -> SealedOptionRun:
    residual = measurement.get("residual")
    residual = residual if isinstance(residual, Mapping) else {}
    return SealedOptionRun(
        strategy_return=_plain_float(measurement.get("strategy_return")),
        strategy_sharpe=_plain_float(measurement.get("strategy_sharpe")),
        benchmark_sharpe=_plain_float(measurement.get("benchmark_sharpe")),
        max_drawdown=_plain_float(measurement.get("max_drawdown")),
        trades=_int(measurement.get("trades")),
        observations=_int(measurement.get("observations")),
        alpha=_plain_float(residual.get("alpha")),
        beta=_plain_float(residual.get("beta")),
        t_alpha=_plain_float(residual.get("t_alpha")),
        significance_bar=_plain_float(measurement.get("significance_bar")),
        is_significant=bool(residual.get("is_significant")),
        refuted=bool(measurement.get("refuted")),
        can_confirm=bool(measurement.get("can_confirm")),
        first_session=str(measurement.get("first_session", "")),
        last_session=str(measurement.get("last_session", "")),
        looks=looks,
        note=str(measurement.get("note", "")),
    )


def _selection_rule(provenance: Mapping[str, Any]) -> str:
    declaration = provenance.get("preregistration")
    declaration = declaration if isinstance(declaration, Mapping) else {}
    return str(declaration.get("selection_rule", ""))


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _plain_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _float(value: Any, *, field: str, faults: list[str]) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        faults.append(f"{field} is {value!r}, not a number")
        return 0.0
    return float(value)


def _decimal(value: Any, *, field: str, faults: list[str]) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        faults.append(f"{field} is {value!r}, not a number")
        return Decimal(0)
    try:
        # Via `str`: `Decimal(0.02)` is 0.0200000000000000004163336342344337...
        return Decimal(str(value))
    except InvalidOperation:
        faults.append(f"{field} is {value!r}, which is not a decimal")
        return Decimal(0)


def _date(value: Any, *, faults: list[str]) -> date:
    try:
        return date.fromisoformat(str(value))
    except (TypeError, ValueError):
        faults.append(f"as_of {value!r} is not an ISO date")
        return date.min


def _timestamp(value: Any, *, faults: list[str]) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        faults.append(f"generated_at {value!r} is not an ISO timestamp")
        return _UNREACHABLE_TIME
    if parsed.tzinfo is None:
        faults.append(f"generated_at {value!r} has no timezone; all times are UTC here")
        return _UNREACHABLE_TIME
    return parsed


def entry_refusal(
    rule: OptionRule, *, open_structures: int, sessions_since_entry: int | None
) -> str:
    """Why the *rule* forbids an entry right now, or `""` if it does not. Pure.

    Two caps travel in every option book and until now neither was enforced:
    they were parsed here, printed on the status page and the dashboard, and
    read by nothing. On 2026-09-02 the agent opened two spreads in one session
    while the page beside it said "cadence: at most one entry per session".

    **These are not the Gate's limits and do not replace them.** The Gate's caps
    are about what this account can survive (specs/03 D5); these are about what
    was actually measured. A rule validated at three concurrent positions and
    one entry a session is a different rule at five and three — and five is what
    the Gate's book-heat budget happens to allow here, which is the number the
    executor would drift to on its own. specs/07 D1 pins the rule by
    fingerprint precisely so that "we execute what the research validated" is
    checkable; a cap that only ever gets printed makes it uncheckable again.

    Concurrency is tested first: both refusals are true when both apply, and the
    one about risk already on the book is the more useful sentence to journal.

    `sessions_since_entry=None` means no entry is on record at all, which is not
    zero sessions ago — a fresh journal must not read as "already traded today".
    """
    if open_structures >= rule.max_concurrent:
        return (
            f"the rule allows {rule.max_concurrent} concurrent position(s) and "
            f"{open_structures} are open"
        )
    spacing = rule.min_sessions_between_entries
    if sessions_since_entry is not None and sessions_since_entry < spacing:
        return (
            f"the rule wants {spacing} session(s) between entries and the last "
            f"entry was {sessions_since_entry} session(s) ago"
        )
    return ""


def measurable_read(read: Any) -> dict[str, Decimal | None]:
    """The subset of a `MarketRead` an `EntryRule` may condition on.

    Takes the read structurally rather than by import so this module stays free
    of `agent.model` and the dependency runs one way. Every name in
    `MEASURABLE_FEATURES` appears in the result, `None` when unmeasured — a
    missing key and a measured `None` must not be distinguishable here, because
    the rule treats both as undecidable.
    """
    return {feature: getattr(read, feature, None) for feature in sorted(MEASURABLE_FEATURES)}
