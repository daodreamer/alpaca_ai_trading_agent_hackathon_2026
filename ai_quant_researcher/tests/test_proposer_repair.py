"""``repair`` on the real proposers.

The research loop asks a proposer to fix a rule that compiled and then fired on
nothing. That path is worth testing without a network, because the interesting
behaviour is entirely in what gets *sent*: the loop's complaint has to reach the
model, or the second attempt is just a second roll of the same dice.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from aqr.agent.proposer import AnthropicProposer, OpenAICompatProposer, Proposal
from tests.test_agent_and_pipeline import GOOD


class _Message:
    def __init__(self, content: str) -> None:
        self.content = content


class _Choice:
    def __init__(self, content: str) -> None:
        self.message = _Message(content)


class _Completion:
    def __init__(self, content: str) -> None:
        self.choices = [_Choice(content)]


class _StubOpenAI:
    def __init__(self, *replies: str) -> None:
        self._replies = list(replies)
        self.sent: list[list[dict[str, str]]] = []
        self.chat = self

    @property
    def completions(self) -> _StubOpenAI:
        return self

    def create(self, **kwargs: Any) -> _Completion:
        self.sent.append([dict(m) for m in kwargs["messages"]])
        return _Completion(self._replies.pop(0))


DEAD = {**GOOD, "name": "dead_v1", "entry": "rsi(14) > 150"}


def _proposer(*replies: str) -> tuple[OpenAICompatProposer, _StubOpenAI]:
    client = _StubOpenAI(*replies)
    return (
        OpenAICompatProposer(client=client, api_key="unused", provider="deepseek"),
        client,
    )


class TestOpenAICompatRepair:
    def test_the_reason_the_rule_was_rejected_reaches_the_model(self) -> None:
        proposer, client = _proposer(json.dumps(GOOD))
        proposer.repair(
            proposal=Proposal(fields=dict(DEAD), source="deepseek", raw=json.dumps(DEAD)),
            problems=["dead_v1 never fires on AAPL: the entry condition is unsatisfiable here"],
        )

        sent = "\n".join(m["content"] for m in client.sent[0])
        assert "never fires" in sent
        assert "rsi(14) > 150" in sent, "the model must see what it wrote"

    def test_a_repaired_proposal_comes_back_checked(self) -> None:
        proposer, _ = _proposer(json.dumps(GOOD))
        result = proposer.repair(
            proposal=Proposal(fields=dict(DEAD), source="deepseek", raw=json.dumps(DEAD)),
            problems=["never fires"],
        )
        assert result.fields["entry"] == GOOD["entry"]
        assert result.source == "deepseek"

    def test_a_repair_that_returns_garbage_raises_rather_than_returning_it(self) -> None:
        # The loop treats a raised repair as "no repair" and evaluates the
        # original. Returning an unparseable proposal instead would push the
        # failure one layer down, where the reason is gone.
        proposer, _ = _proposer("not json at all")
        with pytest.raises(ValueError):
            proposer.repair(
                proposal=Proposal(fields=dict(DEAD), source="deepseek", raw="{}"),
                problems=["never fires"],
            )

    def test_repair_does_not_consult_research_memory(self) -> None:
        """A repair is a correction, not a new hypothesis. Re-sending the whole
        memory would invite the model to change the idea rather than fix it."""
        proposer, client = _proposer(json.dumps(GOOD))
        proposer.repair(
            proposal=Proposal(fields=dict(DEAD), source="deepseek", raw=json.dumps(DEAD)),
            problems=["never fires"],
        )
        sent = "\n".join(m["content"] for m in client.sent[0])
        assert "Experiments already run" not in sent


class TestAnthropicRepairExists:
    def test_the_anthropic_proposer_exposes_repair(self) -> None:
        # The loop discovers repair by duck-typing, so a missing method is a
        # silent loss of the feature rather than an error.
        assert callable(getattr(AnthropicProposer, "repair", None))
