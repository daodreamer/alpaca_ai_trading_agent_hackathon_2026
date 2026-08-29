"""The seam around the one nondeterministic step — specs/05 D3 and D7.

A `Proposer` is handed a `MarketRead` and a menu of `Candidate`s, and returns a
`Choice`: an index, or `None`. That signature *is* the safety argument. There is
no parameter through which a proposer could size a trade, name a contract, set a
price, or reach the broker.

Four implementations, and the fact that they are interchangeable is the point:

* `DeepSeekProposer` (`deepseek.py`) — the live model.
* `DeterministicProposer` — a pure rule over the same `MarketRead`. The backtest
  runs this, which is why specs/05 D7 insists the submission say plainly that
  **what the backtest measures is the strategy, not the model.**
* `RecordedProposer` — replays a journalled choice by cycle id, so a day replays
  to an identical order set without a network or a model.
* `DecliningProposer` — the fail-closed default. Used wherever a proposer is
  required and none is configured, so that a missing model produces no trades
  rather than an exception at the wrong moment.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final, Protocol, runtime_checkable

from alphagate.agent.model import Candidate, Choice, MarketRead, ModelCall

__all__ = [
    "DecliningProposer",
    "DeterministicProposer",
    "Proposal",
    "Proposer",
    "RecordedProposer",
]


@dataclass(frozen=True, slots=True)
class Proposal:
    """A choice plus the evidence of how it was made.

    The `ModelCall` travels with the `Choice` rather than being logged
    separately, because specs/06 D2 wants them in one record: a rationale
    without the model id and prompt version behind it is a quote with no source.
    """

    choice: Choice
    call: ModelCall


@runtime_checkable
class Proposer(Protocol):
    """Pick one candidate, or none."""

    def propose(
        self, read: MarketRead, candidates: Sequence[Candidate], *, cycle_id: str
    ) -> Proposal: ...


@dataclass(frozen=True, slots=True)
class DecliningProposer:
    """Declines, always. The fail-closed default — specs/05 D6.

    Exists so that "no model configured" is a system that does not trade, rather
    than a system that raises `None has no attribute propose` halfway through a
    cycle and leaves the journal without a record.
    """

    reason: str = "no proposer configured"

    def propose(
        self, read: MarketRead, candidates: Sequence[Candidate], *, cycle_id: str
    ) -> Proposal:
        return Proposal(
            choice=Choice(candidate_index=None, rationale=self.reason),
            call=ModelCall(
                model="none",
                prompt_version="n/a",
                temperature=0.0,
                latency_ms=0,
                raw_response="",
                error=self.reason,
            ),
        )


@dataclass(frozen=True, slots=True)
class DeterministicProposer:
    """A pure rule over the same `MarketRead`. No model, no network.

    This is what the backtest runs (specs/05 D7). It takes the top-ranked
    candidate when the read says premium is expensive, and declines otherwise —
    deliberately simple, because its job is to let the *strategy* be measured
    over months without months of model calls, not to be a good trader.

    Interchangeable with the live proposer by construction: same inputs, same
    return type, no extra channel. If swapping it in changed what could reach the
    broker, the seam would not be a seam.
    """

    min_iv_rank: str = "50"
    name: str = "deterministic-v1"

    def propose(
        self, read: MarketRead, candidates: Sequence[Candidate], *, cycle_id: str
    ) -> Proposal:
        from decimal import Decimal

        threshold = Decimal(self.min_iv_rank)
        if not candidates:
            choice = Choice(None, "no candidates", 0.0)
        elif read.iv_rank is None:
            # Fail closed. An unranked implied volatility is not a low one, and
            # "sell premium" is not answerable without knowing whether premium
            # is rich — specs/05 D6.
            choice = Choice(
                None, "iv_rank is unmeasured: not enough history to rank premium", 0.0
            )
        elif read.iv_rank < threshold:
            choice = Choice(
                None,
                f"iv_rank {read.iv_rank} below {threshold}: premium is not rich enough to sell",
                0.0,
            )
        else:
            choice = Choice(
                0,
                f"iv_rank {read.iv_rank} at or above {threshold}; taking the best "
                "return on risk on the menu",
                0.0,
            )
        return Proposal(
            choice=choice,
            call=ModelCall(
                model=self.name,
                prompt_version="n/a",
                temperature=0.0,
                latency_ms=0,
                raw_response=choice.rationale,
            ),
        )


@dataclass
class RecordedProposer:
    """Replays journalled choices by cycle id — specs/05 D7, specs/06 D6.

    A cycle id with no recorded choice **declines**. It does not fall through to
    a live model and it does not raise: replaying a day that has grown an extra
    cycle should produce a shorter order set, visibly, not a different one
    quietly or a crash halfway through.
    """

    choices: Mapping[str, Choice] = field(default_factory=dict)
    seen: list[str] = field(default_factory=list)
    name: str = "recorded"

    def propose(
        self, read: MarketRead, candidates: Sequence[Candidate], *, cycle_id: str
    ) -> Proposal:
        self.seen.append(cycle_id)
        recorded = self.choices.get(cycle_id)
        choice = (
            recorded
            if recorded is not None
            else Choice(None, f"no recorded choice for cycle {cycle_id}", 0.0)
        )
        return Proposal(
            choice=choice,
            call=ModelCall(
                model=self.name,
                prompt_version="replay",
                temperature=0.0,
                latency_ms=0,
                raw_response=choice.rationale,
            ),
        )


DEFAULT_PROPOSER: Final = DecliningProposer()
