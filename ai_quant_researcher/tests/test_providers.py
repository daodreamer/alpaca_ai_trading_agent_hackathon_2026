"""Credential loading and the OpenAI-compatible proposer.

No test here touches the network. The interesting behaviour of a proposer that
talks to a schema-less endpoint is what it does when the reply is *wrong*, and
that is entirely testable with a stub client — which is the whole reason the
client is injectable.

The credential tests exist for a narrower reason: this is the one module that
handles secrets, and the property worth pinning is that a real environment
variable always beats a file, so a CI secret can never be silently replaced by a
stale ``.env`` someone committed years ago.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from aqr.agent.proposer import (
    OpenAICompatProposer,
    build_spec,
    check_proposal,
)
from aqr.config import _parse, credential, describe, load_env_files
from tests.test_agent_and_pipeline import GOOD

# --------------------------------------------------------------------------- #
# Credentials
# --------------------------------------------------------------------------- #


class TestEnvParsing:
    def test_parses_plain_quoted_and_exported_forms(self) -> None:
        parsed = _parse(
            "\n".join(
                [
                    "# a comment",
                    "PLAIN=value",
                    'QUOTED="quoted value"',
                    "SINGLE='single'",
                    "export EXPORTED=exported",
                    "  SPACED  =  spaced  ",
                    "",
                    "NOT_AN_ASSIGNMENT",
                ]
            )
        )
        assert parsed == {
            "PLAIN": "value",
            "QUOTED": "quoted value",
            "SINGLE": "single",
            "EXPORTED": "exported",
            "SPACED": "spaced",
        }

    def test_a_value_containing_equals_is_kept_whole(self) -> None:
        """Base64 and JWT secrets routinely contain '='."""
        assert _parse("KEY=abc=def==")["KEY"] == "abc=def=="


class TestEnvLoading:
    def test_the_real_environment_wins(self, tmp_path: Path, monkeypatch) -> None:
        (tmp_path / ".env.local").write_text("AQR_TEST_KEY=from_file\n", encoding="utf-8")
        monkeypatch.setenv("AQR_TEST_KEY", "from_environment")
        load_env_files(tmp_path)
        assert os.environ["AQR_TEST_KEY"] == "from_environment", (
            "a dotenv file overrode an explicitly exported variable"
        )

    def test_env_local_beats_env(self, tmp_path: Path, monkeypatch) -> None:
        (tmp_path / ".env").write_text("AQR_TEST_KEY=shared\n", encoding="utf-8")
        (tmp_path / ".env.local").write_text("AQR_TEST_KEY=personal\n", encoding="utf-8")
        monkeypatch.delenv("AQR_TEST_KEY", raising=False)
        load_env_files(tmp_path)
        assert os.environ["AQR_TEST_KEY"] == "personal"

    def test_searches_upward_from_a_subdirectory(self, tmp_path: Path, monkeypatch) -> None:
        """The key lives at the repository root; the researcher runs one level down."""
        (tmp_path / ".env.local").write_text("AQR_TEST_KEY=found\n", encoding="utf-8")
        nested = tmp_path / "subproject" / "deeper"
        nested.mkdir(parents=True)
        monkeypatch.delenv("AQR_TEST_KEY", raising=False)
        loaded = load_env_files(nested)
        assert loaded and os.environ["AQR_TEST_KEY"] == "found"

    def test_a_missing_key_says_where_to_put_one(self, monkeypatch) -> None:
        monkeypatch.setenv("DEEPSEEK_API_KEY", "")
        with pytest.raises(RuntimeError, match=".env.local"):
            credential("deepseek", load=False)

    def test_an_unknown_provider_is_rejected(self) -> None:
        with pytest.raises(KeyError, match="unknown provider"):
            credential("nonesuch", load=False)

    def test_describe_never_reveals_a_value(self, monkeypatch) -> None:
        secret = "sk-do-not-print-this-anywhere"
        monkeypatch.setenv("DEEPSEEK_API_KEY", secret)
        rendered = "\n".join(str(s) for s in describe(load=False))
        assert secret not in rendered
        assert str(len(secret)) in rendered


# --------------------------------------------------------------------------- #
# Proposal checking
# --------------------------------------------------------------------------- #


class TestCheckProposal:
    def test_a_good_proposal_has_no_problems(self) -> None:
        assert check_proposal(dict(GOOD)) == []

    def test_a_misspelled_feature_gets_a_suggestion(self) -> None:
        problems = check_proposal(dict(GOOD, entry="emaa(20) > 1"))
        assert len(problems) == 1
        assert "Did you mean" in problems[0] and "ema" in problems[0]

    def test_an_expression_that_is_not_a_condition_is_caught(self) -> None:
        problems = check_proposal(dict(GOOD, entry="ema(20)"))
        assert "not a condition" in problems[0]

    def test_code_shaped_input_is_caught(self) -> None:
        problems = check_proposal(dict(GOOD, entry="__import__('os').system('rm -rf /')"))
        assert problems and "does not parse" in problems[0]

    @pytest.mark.parametrize("field", ["name", "entry", "hypothesis"])
    def test_a_missing_required_field_is_reported_by_name(self, field: str) -> None:
        problems = check_proposal({k: v for k, v in GOOD.items() if k != field})
        assert any(field in p for p in problems)

    def test_a_bad_direction_is_reported(self) -> None:
        problems = check_proposal(dict(GOOD, direction="sideways"))
        assert any("long" in p and "short" in p for p in problems)

    def test_a_string_where_a_number_belongs_is_reported(self) -> None:
        problems = check_proposal(dict(GOOD, stop_loss_atr_multiple="two"))
        assert any("must be a number" in p for p in problems)

    def test_the_regime_is_checked_too(self) -> None:
        problems = check_proposal(dict(GOOD, regime="close > nonexistent(3)"))
        assert any("regime does not parse" in p for p in problems)

    def test_a_non_object_is_rejected(self) -> None:
        assert check_proposal(["not", "an", "object"]) == [  # type: ignore[arg-type]
            "expected a JSON object, got list"
        ]


# --------------------------------------------------------------------------- #
# The OpenAI-compatible proposer, against a stub endpoint
# --------------------------------------------------------------------------- #


@dataclass
class _Message:
    content: str


@dataclass
class _Choice:
    message: _Message


@dataclass
class _Response:
    choices: list[_Choice]


class _StubClient:
    """Replays scripted replies and records the messages it was sent."""

    def __init__(self, *replies: str) -> None:
        self._replies = list(replies)
        self.requests: list[list[dict[str, str]]] = []
        self.chat = self  # the SDK path is client.chat.completions.create

    @property
    def completions(self) -> _StubClient:
        return self

    def create(self, **kwargs: Any) -> _Response:
        self.requests.append(list(kwargs["messages"]))
        index = min(len(self.requests) - 1, len(self._replies) - 1)
        return _Response(choices=[_Choice(message=_Message(self._replies[index]))])


def _proposer(*replies: str, repair_attempts: int = 1) -> OpenAICompatProposer:
    return OpenAICompatProposer(
        "stub-model",
        provider="stub",
        repair_attempts=repair_attempts,
        client=_StubClient(*replies),
    )


class TestOpenAICompatProposer:
    def test_a_valid_first_reply_is_accepted(self) -> None:
        proposer = _proposer(json.dumps(GOOD))
        proposal = proposer.propose(symbols=["SPY"], timeframe="1D", memory=[])
        assert proposal.fields["name"] == GOOD["name"]
        assert proposal.source == "stub"
        assert proposal.model == "stub-model"
        assert proposal.prompt_hash

    def test_the_schema_is_sent_in_the_system_prompt(self) -> None:
        """The endpoint cannot enforce a schema, so the prompt must carry it."""
        proposer = _proposer(json.dumps(GOOD))
        proposer.propose(symbols=["SPY"], timeframe="1D", memory=[])
        client: Any = proposer._client
        system = client.requests[0][0]["content"]
        assert "stop_loss_atr_multiple" in system
        assert "json" in system.lower(), "JSON mode needs the word 'json' in the prompt"

    def test_a_broken_reply_is_repaired_on_the_second_attempt(self) -> None:
        broken = json.dumps(dict(GOOD, entry="emaa(20) > 1"))
        proposer = _proposer(broken, json.dumps(GOOD))
        proposal = proposer.propose(symbols=["SPY"], timeframe="1D", memory=[])
        assert proposal.fields["entry"] == GOOD["entry"]

        client: Any = proposer._client
        assert len(client.requests) == 2
        repair = client.requests[1][-1]["content"]
        assert "rejected" in repair and "Did you mean" in repair, (
            "the repair turn did not tell the model what was actually wrong"
        )

    def test_invalid_json_is_also_repaired(self) -> None:
        proposer = _proposer("this is not json at all", json.dumps(GOOD))
        proposal = proposer.propose(symbols=["SPY"], timeframe="1D", memory=[])
        assert proposal.fields["name"] == GOOD["name"]

    def test_giving_up_names_the_last_problem(self) -> None:
        """Repair is bounded. Failing loudly beats looping on a confused model."""
        broken = json.dumps(dict(GOOD, entry="emaa(20) > 1"))
        proposer = _proposer(broken, broken)
        with pytest.raises(ValueError, match="unusable proposal"):
            proposer.propose(symbols=["SPY"], timeframe="1D", memory=[])
        client: Any = proposer._client
        assert len(client.requests) == 2, "repair attempts were not bounded"

    def test_repair_can_be_switched_off(self) -> None:
        broken = json.dumps(dict(GOOD, entry="emaa(20) > 1"))
        proposer = _proposer(broken, json.dumps(GOOD), repair_attempts=0)
        with pytest.raises(ValueError):
            proposer.propose(symbols=["SPY"], timeframe="1D", memory=[])
        client: Any = proposer._client
        assert len(client.requests) == 1

    def test_an_accepted_proposal_compiles_into_a_strategy(self) -> None:
        """End of the containment chain: the model's words become a validated spec."""
        proposer = _proposer(json.dumps(GOOD))
        proposal = proposer.propose(symbols=["SPY"], timeframe="1D", memory=[])
        spec = build_spec(proposal, ["SPY", "QQQ"])
        assert spec.universe.symbols == ("SPY", "QQQ"), "the model chose its own universe"
        assert spec.fingerprint()

    def test_memory_reaches_the_prompt(self) -> None:
        proposer = _proposer(json.dumps(GOOD))
        proposer.propose(
            symbols=["SPY"],
            timeframe="1D",
            memory=[{"name": "prior_idea", "verdict": "REJECT", "hypothesis": "h"}],
        )
        client: Any = proposer._client
        user = client.requests[0][1]["content"]
        assert "prior_idea" in user and "REJECT" in user
