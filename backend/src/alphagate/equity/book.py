"""The target book, as AlphaGate is allowed to believe it — specs/09 D1.

PURE. Stdlib plus `alphagate.core`. No I/O: `load_target_book` takes a parsed
mapping, never a path, so the validation is testable without a filesystem and
identical whether the bytes came from disk, a fixture or a socket.

`ai_quant_researcher` writes this artefact and stops. It does not import
`alphagate` and `alphagate` does not import it — the file is the whole interface
(specs/09 D0). Which means this module is where an unvalidated strategy would
get in, and so it is the module with the least freedom.

**The fingerprint is pinned by the operator, not read from the book.** A book
names the strategy it describes. If that name were believed, replacing the file
would replace the strategy and nothing would notice; pinning it means a
different book is refused by name rather than executed by accident. That single
argument is what makes "only the strategy the researcher validated" a checkable
statement.

Weights arrive as JSON floats and become `Decimal` here, at the boundary, via
`str`. They are about to be multiplied by equity, and equity is money.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Final

from alphagate.core.errors import InvariantViolation
from alphagate.core.identifiers import Ticker, ticker

__all__ = [
    "BOOK_SCHEMA_VERSION",
    "GROSS_TOLERANCE",
    "TRADEABLE_STATUS",
    "SealedRun",
    "TargetBook",
    "UnusableBook",
    "load_target_book",
]

BOOK_SCHEMA_VERSION: Final = 1
"""The only schema this reader understands.

`aqr` bumps it when a field changes meaning, never when one is added — so an
unknown *field* is harmless and an unknown *version* is fatal. Reading a book
whose `weights` mean something new under the same number is the silent execution
bug this pin exists to prevent."""

TRADEABLE_STATUS: Final = frozenset({"PAPER", "LIVE"})
"""Registry lifecycle states that may be executed.

`CANDIDATE → PAPER → LIVE` and nothing skips a step — the researcher's registry
refuses `CANDIDATE → LIVE` outright. Reading the state here means that rule is
enforced on the execution side too, rather than trusted to have been enforced
upstream."""

GROSS_TOLERANCE: Final = Decimal("0.005")
"""How far above 1.0 a book's gross exposure may sit before it reads as leverage.

Not zero: the weights are floats summed from a hundred names, so an exactly-1.0
book routinely arrives as 1.0000000000000002. Half a percent is far below any
leverage anybody would take deliberately and far above float noise."""


class UnusableBook(InvariantViolation):
    """A target book that must not be executed, and every reason why.

    Every fault is collected before raising. A book wrong in three ways should
    be reported three times and fixed once, not discovered three mornings
    running.
    """


@dataclass(frozen=True, slots=True)
class SealedRun:
    """What the pre-registered out-of-sample run measured.

    Carried for the record and for the dashboard (specs/09 D10) — never acted
    on. Nothing here sizes a position or gates an order: the sealed window
    decided whether this strategy may run at all, and that decision was made
    before this process started.

    `can_confirm` is `False` by construction upstream, and it is kept with that
    name rather than flattened into a boolean verdict, because "was not refuted"
    and "was confirmed" are different claims and only one of them is available.
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
    information_ratio: float
    is_significant: bool
    refuted: bool
    can_confirm: bool
    first_session: str
    last_session: str
    looks: int
    """How many candidates this sealed window has now screened.

    The multiplicity denominator. A `t` of +2.22 clears the bar as the only
    candidate ever screened and does not clear it as the seventh, so the count
    travels with the measurement rather than being reconstructible only from the
    researcher's database."""


@dataclass(frozen=True, slots=True)
class TargetBook:
    """A validated book: weights per symbol, and the provenance behind them.

    Weights are `Decimal` because the next thing that happens to them is
    multiplication by equity. Everything the sealed run measured is `float`,
    because those are estimates and not money — the same split as
    [CLAUDE.md](../../../CLAUDE.md) §3 rule 4.
    """

    fingerprint: str
    name: str
    version: int
    as_of: date
    """The session the weights were in force on. Not "today"."""
    generated_at: datetime
    dataset_version: str
    universe: str
    timeframe: str
    weights: Mapping[Ticker, Decimal]
    core_weights: Mapping[Ticker, Decimal]
    symbols_loaded: int
    symbols_declared: int
    status: str
    hypothesis: str
    selection_rule: str
    distinct_hypotheses: int
    sealed: SealedRun
    digest: str
    """SHA-256 of the bytes this book was loaded from. Journalled with every
    plan, so a file that has since been regenerated can be told apart from the
    one that was actually executed (specs/09 D9)."""

    def __post_init__(self) -> None:
        if not self.weights:
            raise InvariantViolation(
                "a target book with no weights is not 'hold nothing', it is a book "
                "that failed to build; refusing to read it as an instruction"
            )

    @property
    def gross(self) -> Decimal:
        return sum(self.weights.values(), Decimal(0))

    @property
    def sleeve_weights(self) -> Mapping[Ticker, Decimal]:
        """Everything that is not core exposure, per symbol.

        Derived rather than read, because the artefact's `core_weights` and
        `sleeve_weights` are the two halves of `weights` and carrying all three
        would let them disagree.
        """
        return {
            symbol: weight - self.core_weights.get(symbol, Decimal(0))
            for symbol, weight in self.weights.items()
        }

    def age_days(self, today: date) -> int:
        """Calendar days since the session the weights were in force on."""
        return (today - self.as_of).days

    def label(self) -> str:
        return f"{self.name} [{self.fingerprint}] as of {self.as_of.isoformat()}"


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def load_target_book(
    payload: Mapping[str, Any],
    *,
    pinned_fingerprint: str,
    digest: str,
) -> TargetBook:
    """Validate a parsed book against the seven refusals of specs/09 D1.

    Pure. `digest` is computed by the caller over the bytes it read, because
    hashing a re-serialised mapping would hash this process's JSON formatting
    rather than the file.

    Faults are accumulated and raised together. The exception is deliberately an
    `InvariantViolation` subclass: an unusable book is not a recoverable
    condition to be retried, it is a statement that the thing upstream produced
    must not be executed.
    """
    faults: list[str] = []

    version = payload.get("schema_version")
    if version != BOOK_SCHEMA_VERSION:
        faults.append(
            f"schema_version is {version!r}, not {BOOK_SCHEMA_VERSION}; this reader "
            "cannot know what its fields mean"
        )

    fingerprint = str(payload.get("spec_fingerprint", ""))
    if not fingerprint:
        faults.append("no spec_fingerprint")
    elif fingerprint != pinned_fingerprint:
        faults.append(
            f"book is for {fingerprint}, and the pinned strategy is "
            f"{pinned_fingerprint}. Only the pinned fingerprint may be executed — "
            "see specs/09 D1"
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
        faults.append("the sealed window refuted this strategy")

    weights = _weights(payload.get("weights"), field="weights", faults=faults)
    core = _weights(payload.get("core_weights"), field="core_weights", faults=faults)

    negative = sorted(symbol for symbol, weight in weights.items() if weight < 0)
    if negative:
        faults.append(
            f"negative weights on {negative}: a short equity leg needs a locate, "
            "accrues borrow and has unbounded loss. specs/09 D6 — not expressible here"
        )

    gross = sum(weights.values(), Decimal(0))
    if gross > Decimal(1) + GROSS_TOLERANCE:
        faults.append(
            f"gross exposure {gross} exceeds 1.0; that is leverage, and no run "
            "upstream measured it"
        )

    as_of = _date(payload.get("as_of"), faults=faults)
    generated_at = _timestamp(payload.get("generated_at"), faults=faults)

    if faults:
        raise UnusableBook(
            "this target book must not be executed:\n  - " + "\n  - ".join(faults)
        )

    return TargetBook(
        fingerprint=fingerprint,
        name=str(payload.get("spec_name", "")),
        version=_int(payload.get("spec_version")),
        as_of=as_of,
        generated_at=generated_at,
        dataset_version=str(payload.get("dataset_version", "")),
        universe=str(payload.get("universe", "")),
        timeframe=str(payload.get("timeframe", "")),
        weights=weights,
        core_weights=core,
        symbols_loaded=_int(payload.get("symbols_loaded")),
        symbols_declared=_int(payload.get("symbols_declared")),
        status=status,
        hypothesis=str(provenance.get("hypothesis", "")),
        selection_rule=_selection_rule(provenance),
        distinct_hypotheses=_int(provenance.get("distinct_hypotheses")),
        sealed=_sealed(measurement, looks=_int(provenance.get("sealed_looks_total")) or looks),
        digest=digest,
    )


def _selection_rule(provenance: Mapping[str, Any]) -> str:
    registration = provenance.get("preregistration")
    if isinstance(registration, Mapping):
        return str(registration.get("selection_rule", ""))
    return ""


def _sealed(measurement: Mapping[str, Any], *, looks: int) -> SealedRun:
    residual = measurement.get("residual")
    residual = residual if isinstance(residual, Mapping) else {}
    return SealedRun(
        strategy_return=_float(measurement.get("strategy_return")),
        strategy_sharpe=_float(measurement.get("strategy_sharpe")),
        benchmark_sharpe=_float(measurement.get("benchmark_sharpe")),
        max_drawdown=_float(measurement.get("max_drawdown")),
        trades=_int(measurement.get("trades")),
        observations=_int(measurement.get("observations")),
        alpha=_float(residual.get("alpha")),
        beta=_float(residual.get("beta")),
        t_alpha=_float(residual.get("t_alpha")),
        information_ratio=_float(residual.get("information_ratio")),
        is_significant=bool(residual.get("is_significant")),
        refuted=bool(measurement.get("refuted")),
        can_confirm=bool(measurement.get("can_confirm")),
        first_session=str(measurement.get("first_session", "")),
        last_session=str(measurement.get("last_session", "")),
        looks=looks,
    )


def _weights(
    raw: Any, *, field: str, faults: list[str]
) -> dict[Ticker, Decimal]:
    """Parse one weight map, collecting faults rather than raising on the first.

    `Decimal(str(x))` rather than `Decimal(x)`: the JSON value is a float, and
    the exact binary expansion of 0.0019230769230769232 is not what anybody
    means. Going through `str` keeps the seventeen digits that were written and
    discards the ones that were an artefact of the format.
    """
    if not isinstance(raw, Mapping):
        faults.append(f"{field} is {type(raw).__name__}, expected an object")
        return {}
    parsed: dict[Ticker, Decimal] = {}
    for symbol, weight in raw.items():
        try:
            name = ticker(str(symbol))
        except InvariantViolation:
            faults.append(f"{field}: {symbol!r} is not a ticker")
            continue
        if isinstance(weight, bool) or not isinstance(weight, (int, float, str)):
            faults.append(f"{field}[{symbol}] is {type(weight).__name__}, not a number")
            continue
        try:
            value = Decimal(str(weight))
        except InvalidOperation:
            faults.append(f"{field}[{symbol}] is {weight!r}, which is not a number")
            continue
        if not value.is_finite():
            faults.append(f"{field}[{symbol}] is {value}, which is not finite")
            continue
        parsed[name] = value
    return parsed


def _date(raw: Any, *, faults: list[str]) -> date:
    try:
        return date.fromisoformat(str(raw))
    except (TypeError, ValueError):
        faults.append(f"as_of {raw!r} is not an ISO date")
        return date(1970, 1, 1)


def _timestamp(raw: Any, *, faults: list[str]) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(raw))
    except (TypeError, ValueError):
        faults.append(f"generated_at {raw!r} is not an ISO timestamp")
        return datetime.fromisoformat("1970-01-01T00:00:00+00:00")
    if parsed.tzinfo is None:
        faults.append(f"generated_at {raw!r} has no timezone")
        return datetime.fromisoformat("1970-01-01T00:00:00+00:00")
    return parsed


def _int(value: Any) -> int:
    if isinstance(value, bool) or value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _float(value: Any) -> float:
    if isinstance(value, bool) or value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
