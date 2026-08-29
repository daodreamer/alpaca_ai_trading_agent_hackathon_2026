"""The live proposer, offline — specs/05 D3, D5, D6, D7.

No network. `httpx.MockTransport` answers the request, which lets the suite test
the shapes a real model actually produces at 3am: a 502, a timeout, prose
instead of JSON, a fenced code block, an index of 2.5, a `true` where an integer
belongs.

Every one of them must produce a decline with the reason recorded, and none of
them may raise into the cycle. That is specs/05 D6 in one sentence, and this
file is the whole of its evidence.

The other property under test is what the module *cannot* do: retry. A retried
proposal is a second sample from a distribution, and taking the first answer
that parses is sampling until the model agrees with us.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict

import httpx
import pytest

from alphagate.agent.deepseek import DEEPSEEK_BASE_URL, DeepSeekProposer, parse_choice
from alphagate.agent.prompt import PROMPT_VERSION, SYSTEM_PROMPT, build_user_message
from tests.agent.conftest import menu, read

FAKE_KEY = "sk-not-a-real-key-0000000000000000"


Handler = Callable[[httpx.Request], httpx.Response]


def responder(
    payload: object, status: int = 200
) -> tuple[Handler, list[httpx.Request]]:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if isinstance(payload, Exception):
            raise payload
        return httpx.Response(status, json=payload)

    return handler, calls


def proposer_with(handler: Handler, **kwargs: object) -> DeepSeekProposer:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return DeepSeekProposer(api_key=FAKE_KEY, client=client, **kwargs)  # type: ignore[arg-type]


def completion(content: str) -> dict[str, object]:
    return {
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 900, "completion_tokens": 40, "total_tokens": 940},
    }


def answer(
    index: object, rationale: str = "reasoning", confidence: float = 0.6
) -> dict[str, object]:
    return completion(
        json.dumps(
            {"candidate_index": index, "rationale": rationale, "confidence": confidence}
        )
    )


class TestTheHappyPath:
    def test_a_valid_index_is_returned(self) -> None:
        handler, _ = responder(answer(1))
        result = proposer_with(handler).propose(read(), menu(), cycle_id="c-1")
        assert result.choice.candidate_index == 1
        assert result.choice.rationale == "reasoning"
        assert result.call.error is None

    def test_the_call_is_recorded_for_the_journal(self) -> None:
        handler, _ = responder(answer(0))
        result = proposer_with(handler).propose(read(), menu(), cycle_id="c-1")
        assert result.call.model == "deepseek-chat"
        assert result.call.prompt_version == PROMPT_VERSION
        assert result.call.temperature == 0.0
        assert result.call.usage["total_tokens"] == 940
        assert result.call.latency_ms >= 0

    def test_it_asks_for_json_at_temperature_zero(self) -> None:
        """specs/05 D7's fencing, as sent on the wire."""
        handler, calls = responder(answer(0))
        proposer_with(handler).propose(read(), menu(), cycle_id="c-1")
        body = json.loads(calls[0].content)
        assert body["temperature"] == 0.0
        assert body["response_format"] == {"type": "json_object"}
        assert body["model"] == "deepseek-chat"

    def test_the_prompt_carries_the_read_and_the_menu(self) -> None:
        handler, calls = responder(answer(0))
        proposer_with(handler).propose(read(), menu(), cycle_id="c-1")
        body = json.loads(calls[0].content)
        assert body["messages"][0]["content"] == SYSTEM_PROMPT
        user = json.loads(body["messages"][1]["content"])
        assert user["market_read"]["iv_rank_0_100"] == "62"
        assert len(user["menu"]) == len(menu())


class TestFailClosedD6:
    """Every one of these is a decline with a reason, and none of them raises."""

    def test_a_server_error(self) -> None:
        handler, _ = responder({"error": "internal"}, status=500)
        result = proposer_with(handler).propose(read(), menu(), cycle_id="c-1")
        assert result.choice.declined
        assert result.call.error is not None

    def test_a_timeout(self) -> None:
        handler, _ = responder(httpx.ReadTimeout("too slow"))
        result = proposer_with(handler).propose(read(), menu(), cycle_id="c-1")
        assert result.choice.declined
        assert "ReadTimeout" in str(result.call.error)

    def test_a_connection_failure(self) -> None:
        handler, _ = responder(httpx.ConnectError("no route to host"))
        result = proposer_with(handler).propose(read(), menu(), cycle_id="c-1")
        assert result.choice.declined

    def test_prose_instead_of_json(self) -> None:
        handler, _ = responder(completion("I would sell the 752/747 put spread."))
        result = proposer_with(handler).propose(read(), menu(), cycle_id="c-1")
        assert result.choice.declined
        assert "invalid JSON" in str(result.call.error)

    def test_an_empty_response(self) -> None:
        handler, _ = responder({"choices": []})
        result = proposer_with(handler).propose(read(), menu(), cycle_id="c-1")
        assert result.choice.declined

    def test_a_refusal(self) -> None:
        handler, _ = responder(completion("I cannot provide financial advice."))
        result = proposer_with(handler).propose(read(), menu(), cycle_id="c-1")
        assert result.choice.declined

    def test_an_empty_menu_is_not_even_asked(self) -> None:
        """Spending a call and a prompt on a question with one answer."""
        handler, calls = responder(answer(0))
        result = proposer_with(handler).propose(read(), (), cycle_id="c-1")
        assert result.choice.declined
        assert calls == []

    def test_nothing_raises_into_the_cycle(self) -> None:
        for payload in (
            httpx.ReadTimeout("slow"),
            httpx.ConnectError("down"),
            {"nonsense": True},
            completion("```not json```"),
        ):
            handler, _ = responder(payload)
            result = proposer_with(handler).propose(read(), menu(), cycle_id="c-1")
            assert result.choice.declined


class TestNoRetry:
    def test_one_call_on_success(self) -> None:
        handler, calls = responder(answer(0))
        proposer_with(handler).propose(read(), menu(), cycle_id="c-1")
        assert len(calls) == 1

    def test_one_call_on_failure(self) -> None:
        """"Taking the first answer that parses is sampling until the model
        agrees with us." Once, at temperature 0, and then the journal."""
        handler, calls = responder({"error": "boom"}, status=500)
        proposer_with(handler).propose(read(), menu(), cycle_id="c-1")
        assert len(calls) == 1

    def test_one_call_on_malformed_output(self) -> None:
        handler, calls = responder(completion("not json"))
        proposer_with(handler).propose(read(), menu(), cycle_id="c-1")
        assert len(calls) == 1


class TestParsing:
    """`parse_choice` never raises. specs/05 D3."""

    def test_a_null_index_is_a_decline_with_the_rationale_kept(self) -> None:
        choice, error = parse_choice(
            json.dumps({"candidate_index": None, "rationale": "iv too low", "confidence": 0.2}), 5
        )
        assert choice.declined
        assert choice.rationale == "iv too low"
        assert error is None, "declining is not an error"

    @pytest.mark.parametrize("index", [5, 6, 99, -1])
    def test_an_out_of_range_index_is_a_decline(self, index: int) -> None:
        choice, error = parse_choice(json.dumps({"candidate_index": index}), 5)
        assert choice.declined
        assert error is not None
        assert "outside the menu" in error

    def test_a_float_that_is_whole_is_accepted(self) -> None:
        """JSON has one number type; a serialiser may write 2 as 2.0."""
        choice, error = parse_choice(json.dumps({"candidate_index": 2.0}), 5)
        assert choice.candidate_index == 2
        assert error is None

    def test_a_fractional_index_is_refused_not_rounded(self) -> None:
        """"2.7 is not evidence the model meant 2 or 3; it is evidence it was
        not answering the question asked." """
        choice, error = parse_choice(json.dumps({"candidate_index": 2.7}), 5)
        assert choice.declined
        assert error is not None

    def test_a_boolean_is_not_an_index(self) -> None:
        """`bool` is an `int` in Python; `true` would otherwise mean index 1."""
        choice, _ = parse_choice(json.dumps({"candidate_index": True}), 5)
        assert choice.declined

    def test_a_string_index_is_refused(self) -> None:
        choice, _ = parse_choice(json.dumps({"candidate_index": "2"}), 5)
        assert choice.declined

    def test_a_fenced_block_is_unwrapped(self) -> None:
        fenced = '```json\n{"candidate_index": 3, "rationale": "r", "confidence": 0.5}\n```'
        choice, _ = parse_choice(fenced, 5)
        assert choice.candidate_index == 3

    def test_a_missing_rationale_gets_a_placeholder_not_a_crash(self) -> None:
        choice, _ = parse_choice(json.dumps({"candidate_index": 0}), 5)
        assert choice.rationale == "no rationale given"

    def test_a_bare_array_is_a_decline(self) -> None:
        choice, error = parse_choice("[0]", 5)
        assert choice.declined
        assert error is not None


class TestConfidenceD5:
    @pytest.mark.parametrize(
        ("given_value", "expected"),
        [(0.7, 0.7), (1.5, 1.0), (-2, 0.0), ("high", 0.0), (None, 0.0)],
    )
    def test_it_is_clamped_and_never_crashes(self, given_value: object, expected: float) -> None:
        choice, _ = parse_choice(
            json.dumps({"candidate_index": 0, "confidence": given_value}), 5
        )
        assert choice.self_reported_confidence == pytest.approx(expected)

    def test_it_is_recorded(self) -> None:
        handler, _ = responder(answer(0, confidence=0.42))
        result = proposer_with(handler).propose(read(), menu(), cycle_id="c-1")
        assert result.choice.self_reported_confidence == pytest.approx(0.42)


class TestTheKeyNeverLeaks:
    """specs/06 D4. The demo video shows this terminal."""

    def test_the_repr_is_redacted(self) -> None:
        proposer = DeepSeekProposer(api_key=FAKE_KEY)
        assert FAKE_KEY not in repr(proposer)
        assert "deepseek-chat" in repr(proposer)

    def test_it_is_not_in_the_model_call_record(self) -> None:
        handler, _ = responder(answer(0))
        result = proposer_with(handler).propose(read(), menu(), cycle_id="c-1")
        # `ModelCall` uses slots, so ask the dataclass rather than a __dict__.
        assert FAKE_KEY not in json.dumps(asdict(result.call), default=str)

    def test_it_is_not_in_an_error_path_either(self) -> None:
        handler, _ = responder(httpx.ConnectError("no route"))
        result = proposer_with(handler).propose(read(), menu(), cycle_id="c-1")
        assert FAKE_KEY not in str(result.call.error)
        assert FAKE_KEY not in result.choice.rationale

    def test_it_does_go_in_the_authorization_header(self) -> None:
        """The one place it belongs."""
        handler, calls = responder(answer(0))
        proposer_with(handler).propose(read(), menu(), cycle_id="c-1")
        assert calls[0].headers["authorization"] == f"Bearer {FAKE_KEY}"


class TestThePrompt:
    def test_an_unmeasured_trend_says_so_rather_than_neutral(self) -> None:
        """"A trend nobody measured is not a flat trend." """
        message = json.loads(build_user_message(read(trend=None), menu()))
        assert message["market_read"]["trend"] == "unmeasured"

    def test_an_unranked_volatility_says_so_rather_than_zero(self) -> None:
        """The regression from the first live run: a level rendered as a rank."""
        message = json.loads(build_user_message(read(iv_rank=None), menu()))
        assert message["market_read"]["iv_rank_0_100"] == "unmeasured"

    def test_the_volatility_keys_carry_their_units(self) -> None:
        """"A field whose name does not disambiguate its units will be misread
        again." 15 must not be readable as both a rank and a percentage."""
        message = json.loads(build_user_message(read(), menu()))
        assert "iv_rank_0_100" in message["market_read"]
        assert "iv_vs_hv_ratio" in message["market_read"]
        assert "iv_rank" not in message["market_read"]

    def test_the_system_prompt_explains_that_rank_is_not_level(self) -> None:
        assert "is a **rank**, not a level" in SYSTEM_PROMPT
        assert "Treat it as unknown, never" in SYSTEM_PROMPT
    def test_the_default_endpoint_is_deepseek(self) -> None:
        assert DEEPSEEK_BASE_URL == "https://api.deepseek.com"

    def test_the_system_prompt_says_declining_is_good(self) -> None:
        """Models are agreeable, and an agreeable model asked to pick from a
        list will pick from the list."""
        assert "null" in SYSTEM_PROMPT
        assert "good answer" in SYSTEM_PROMPT

    def test_the_system_prompt_forbids_sizing_and_symbols(self) -> None:
        assert "do not choose quantity" in SYSTEM_PROMPT.lower()
        assert "do not name contracts" in SYSTEM_PROMPT.lower()
