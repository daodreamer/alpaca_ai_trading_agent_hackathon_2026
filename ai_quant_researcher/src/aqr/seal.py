"""The provenance seal: evidence that a campaign never read the embargoed years.

The last two years of data are reserved for one pre-registered validation run.
Everything before them is the search space. Keeping those apart by convention
does not survive a debugging session at 2am, so the separation is mechanical and
this module holds the mechanism.

**The sensor is on the type, not on the I/O.** Every series in this system
becomes a :class:`~aqr.data.bars.Bars` before anything can compute on it -- CSV,
Yahoo, Alpaca, IBKR, the simulator, a test fixture, a future provider nobody has
written yet. A check on each provider covers the providers somebody remembered;
a check in ``Bars.__post_init__`` covers the route nobody anticipated, which is
the only route that was ever going to be the problem.

**The bit is monotone.** Clean to tainted, never back, and there is no public
setter. A flag that can be cleared records the intention of whoever cleared it.

**The ledger says what was read.** The bit answers *whether*; the ledger answers
*what*, so an audit is a query rather than an act of trust. Its hash chain makes
the record self-consistent: editing a stored verdict without replaying the loads
that produced it does not reproduce the digest.

**The canary catches the rest.** ``__CANARY__`` is a symbol that exists only
after the embargo. Nothing legitimate loads it, so its appearance anywhere is
physical evidence rather than an inference -- the tripwire for a route none of
the above anticipated.

What this cannot do is in :meth:`Seal.certificate`, under ``knowledge_exposure``,
and it matters: the seal proves the embargoed *data* was not read. It cannot
prove the embargoed *period* did not inform a decision. The researcher lived
through it and every model in ``providers`` has a training cutoff after it. So
the certificate records that exposure instead of denying it, and a claim built on
this seal has to be worded as "the data was not read", never as "uncontaminated".
"""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "CANARY_SYMBOL",
    "EMBARGO_START",
    "Contamination",
    "LoadRecord",
    "Phase",
    "Seal",
    "current",
    "enter_sealed_phase",
    "scope",
]

# The embargo. A constant, not a parameter: a configurable embargo is a
# suggestion, and ``test_the_embargo_is_a_constant_not_a_parameter_of_the_search``
# forbids any other module from passing one.
EMBARGO_START = datetime(2024, 9, 1, tzinfo=UTC)

# Inclusive on the embargoed side: a bar stamped exactly at the boundary is
# already forbidden. Written down once here so two call sites cannot each assume
# the other convention.
_EMBARGO_EPOCH = int(EMBARGO_START.timestamp())

# Exists only after the embargo, in the research cache and nowhere else. Named so
# that it can never collide with a ticker.
CANARY_SYMBOL = "__CANARY__"


class Phase(Enum):
    """Which half of the protocol a process is executing.

    ``RESEARCH`` may search and must not see the embargoed years. ``SEALED`` may
    see them and must not search -- it is a fresh process running one
    pre-registered spec. Neither is allowed to do both, because a process that
    could would turn the embargoed years into a search space with nobody counting
    the trials.
    """

    RESEARCH = "research"
    SEALED = "sealed"


class Contamination(RuntimeError):
    """Raised when a phase boundary is crossed. Never caught internally."""


@dataclass(frozen=True, slots=True)
class LoadRecord:
    """One materialisation of data, as it will be audited later.

    ``max_event_time`` rather than the requested end: what was asked for is an
    intention, what came back is the evidence.
    """

    source: str
    symbol: str
    requested_start: int
    requested_end: int
    rows: int
    max_event_time: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "symbol": self.symbol,
            "requested_start": self.requested_start,
            "requested_end": self.requested_end,
            "rows": self.rows,
            "max_event_time": self.max_event_time,
        }


class Seal:
    """Append-only evidence about one process's data access.

    Constructed with an embargo only so the tests can move it. Production code
    uses the ambient singleton and the module constant.
    """

    __slots__ = (
        "_digest",
        "_embargo",
        "_exposure",
        "_loads",
        "_max_event_time",
        "_phase",
        "_run_id",
        "_tainted",
    )

    def __init__(
        self,
        *,
        embargo: datetime = EMBARGO_START,
        phase: Phase = Phase.RESEARCH,
    ) -> None:
        self._embargo = int(embargo.timestamp())
        self._phase = phase
        self._tainted = False
        self._max_event_time: int | None = None
        self._loads: list[LoadRecord] = []
        self._exposure: dict[str, Any] | None = None
        # One process, one seal, one campaign -- so this is the campaign's name.
        # Deliberately outside the digest: two campaigns that read the same bars
        # in the same order must still produce the same chain, or the chain
        # stops being a statement about what was read.
        self._run_id = uuid.uuid4().hex
        # Seeded with the embargo and the phase so that two campaigns run under
        # different rules cannot produce the same digest from the same loads.
        self._digest = hashlib.sha256(
            f"aqr-seal-v1|{self._embargo}|{phase.value}".encode()
        ).hexdigest()

    # -- read-only surface ------------------------------------------------

    @property
    def phase(self) -> Phase:
        return self._phase

    @property
    def run_id(self) -> str:
        """Which campaign this is. Stamped on every experiment the process writes.

        The ancestry taint check groups by it: contamination is a property of the
        process, not of the individual backtest, because a process that read the
        embargoed years while evaluating hypothesis 12 was still contaminated
        when it evaluated hypothesis 13.
        """
        return self._run_id

    @property
    def tainted(self) -> bool:
        """True once embargoed data has been materialised in this process.

        No setter, deliberately.
        """
        return self._tainted

    @property
    def max_event_time(self) -> int | None:
        """The latest bar this process has ever seen, or None if it has seen none."""
        return self._max_event_time

    @property
    def loads(self) -> tuple[LoadRecord, ...]:
        return tuple(self._loads)

    @property
    def digest(self) -> str:
        return self._digest

    # -- the sensor -------------------------------------------------------

    def observe(self, symbol: str, event_time: NDArray[np.int64]) -> None:
        """Register that a series has been materialised.

        Called from ``Bars.__post_init__``, so it runs for every slice and every
        walk-forward fold as well as every load. It must therefore stay cheap and
        must not write to the ledger: a fold is not a load, and a ledger that
        counts folds cannot be read by a human.
        """
        if symbol == CANARY_SYMBOL:
            # The canary's own timestamps are not consulted. The failure it
            # exists to catch is one where the timestamps are wrong too.
            self._taint()
            return
        if event_time.size == 0:
            return
        latest = int(np.max(event_time))
        if self._max_event_time is None or latest > self._max_event_time:
            self._max_event_time = latest
        if latest >= self._embargo:
            self._taint()

    def record_load(self, record: LoadRecord) -> None:
        """Append one load to the ledger and extend the hash chain."""
        self._loads.append(record)
        self._digest = hashlib.sha256(
            (self._digest + "|" + repr(record.as_dict())).encode()
        ).hexdigest()
        if record.symbol == CANARY_SYMBOL:
            self._taint()
            return
        if record.rows:
            if self._max_event_time is None or record.max_event_time > self._max_event_time:
                self._max_event_time = record.max_event_time
            if record.max_event_time >= self._embargo:
                self._taint()

    def _taint(self) -> None:
        # Taint is a research-phase concept. The sealed run is *supposed* to read
        # those years; what makes it honest is that it cannot search.
        if self._phase is Phase.RESEARCH:
            self._tainted = True

    # -- phases -----------------------------------------------------------

    def enter_sealed(self) -> None:
        """Promote this process to the sealed phase. One way, and only while clean.

        Refused once anything has been read. Without that condition the sequence
        "search, then promote, then read the answer" would produce a certificate
        saying ``phase: sealed, tainted: false`` -- which is exactly the claim
        the seal exists to make unfalsifiable.

        There is no way back, and no ``enter_research``. A phase that can be left
        is a phase that will be left at 2am.
        """
        if self._phase is Phase.SEALED:
            return
        if self._loads or self._max_event_time is not None or self._tainted:
            raise Contamination(
                "this process has already read data; only a fresh process may "
                "enter the sealed phase"
            )
        self._phase = Phase.SEALED
        # Folded into the chain so a research certificate can never be mistaken
        # for a sealed one, and so the moment of promotion is itself recorded.
        self._digest = hashlib.sha256(
            (self._digest + "|phase:sealed").encode()
        ).hexdigest()

    def require(self, phase: Phase) -> None:
        if self._phase is not phase:
            raise Contamination(
                f"this operation requires the {phase.value} phase, "
                f"but the process is in {self._phase.value}"
            )

    # -- the record -------------------------------------------------------

    def note_knowledge_exposure(self, **facts: Any) -> None:
        """Record what the researcher and the model could have known anyway.

        Model id, training cutoff, campaign timestamps. Not a mitigation -- a
        disclosure, so that the claim can be worded honestly.
        """
        self._exposure = dict(facts)

    def certificate(self) -> dict[str, Any]:
        """What goes into the experiment record, and what an auditor reads."""
        return {
            "digest": self._digest,
            "run_id": self._run_id,
            "tainted": self._tainted,
            "phase": self._phase.value,
            "embargo_start": self._embargo,
            "max_event_time": self._max_event_time,
            "loads": len(self._loads),
            "knowledge_exposure": self._exposure,
        }


# The ambient seal. One per process, by design: taint that can be scoped away is
# taint that will be scoped away.
_SEAL = Seal()


def current() -> Seal:
    return _SEAL


def enter_sealed_phase() -> None:
    """Promote the ambient seal. **The sealed entry point only.**

    ``test_only_the_sealed_entry_point_promotes_the_process`` fails the build if
    any other module under ``src/aqr`` calls this, which is what keeps the phase
    separation from being a convention.
    """
    _SEAL.enter_sealed()


@contextmanager
def scope(seal: Seal) -> Iterator[Seal]:
    """Swap the ambient seal. **Tests only.**

    ``test_only_the_seal_module_may_swap_the_ambient_seal`` fails the build if any
    module under ``src/aqr`` calls this, which is what keeps the singleton from
    being a suggestion.
    """
    global _SEAL
    previous = _SEAL
    _SEAL = seal
    try:
        yield seal
    finally:
        _SEAL = previous
