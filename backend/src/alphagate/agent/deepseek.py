"""The live proposer — specs/05 D3, D6, D7.

DeepSeek over its OpenAI-compatible chat endpoint, driven with `httpx` rather
than a vendor SDK. That is a deliberate choice and not laziness: the request is
one JSON object, the seam is `Proposer`, and adding an SDK to reach one endpoint
would buy retry logic we do not want (see below) and a dependency that would
then need pinning through the competition.

**Everything here fails closed.** A timeout, a 500, a refusal, malformed JSON, a
missing field, an index of the wrong type — every one of them produces a
`Choice` with `candidate_index=None` and an `error` on the `ModelCall`. Nothing
in this module raises into the cycle, and nothing retries.

Not retrying is the interesting decision. A retried proposal is a second sample
from a distribution, and taking the first answer that parses is sampling until
the model agrees with us. Once. Temperature 0. If it comes back unusable, the
cycle records a decline and we look at the journal afterwards.

Determinism fencing (D7): temperature 0, a fixed model id, and a prompt version
that goes into the record. None of that makes an LLM deterministic; it makes the
nondeterminism *bounded and attributable*, which is the honest claim.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

import httpx

from alphagate.agent.model import Candidate, Choice, MarketRead, ModelCall
from alphagate.agent.prompt import PROMPT_VERSION, SYSTEM_PROMPT, build_user_message
from alphagate.agent.proposer import Proposal

__all__ = ["API_KEY_VAR", "DEEPSEEK_BASE_URL", "DeepSeekProposer", "MissingModelKey"]

API_KEY_VAR: Final = "DEEPSEEK_API_KEY"
"""The one place in the codebase that names it. See `from_env`."""

DEEPSEEK_BASE_URL: Final = "https://api.deepseek.com"
DEFAULT_MODEL: Final = "deepseek-chat"
DEFAULT_TIMEOUT: Final = 45.0
MAX_RESPONSE_CHARS: Final = 20_000
"""Recorded responses are truncated before they reach the journal. A model that
returns a megabyte of text is a model that has already failed the schema."""


class MissingModelKey(RuntimeError):
    """No model key configured. Not a crash — specs/05 D6 has a path."""


@dataclass
class DeepSeekProposer:
    """Asks a model to pick an index. Cannot do anything else.

    The API key is held here and **never** rendered: not in an exception, not in
    the `ModelCall`, not in the journal (specs/06 D4). The only place it appears
    is an `Authorization` header on an outbound request.
    """

    api_key: str
    model: str = DEFAULT_MODEL
    base_url: str = DEEPSEEK_BASE_URL
    timeout: float = DEFAULT_TIMEOUT
    temperature: float = 0.0
    client: httpx.Client | None = field(default=None, repr=False)

    @classmethod
    def from_env(cls, env: Mapping[str, str], **overrides: object) -> DeepSeekProposer:
        """Build one from a credentials mapping. The key name lives here.

        A factory rather than the caller doing `env["DEEPSEEK_API_KEY"]`, because
        specs/01 Rule 1 says only this package may reach a model — and a
        composition root that knows the *name* of the model key is a composition
        root that is one line away from using it. `tests/test_boundaries.py`
        enforces that, and it caught this exact shortcut.
        """
        key = env.get(API_KEY_VAR, "")
        if not key:
            raise MissingModelKey(
                f"{API_KEY_VAR} is not set; run with --no-model to use the "
                "deterministic proposer instead (specs/05 D6)"
            )
        return cls(api_key=key, **overrides)  # type: ignore[arg-type]

    def __repr__(self) -> str:
        """Redacted by construction, because a repr ends up in a traceback."""
        return f"DeepSeekProposer(model={self.model!r}, base_url={self.base_url!r})"

    # ------------------------------------------------------------------ #

    def propose(
        self, read: MarketRead, candidates: Sequence[Candidate], *, cycle_id: str
    ) -> Proposal:
        if not candidates:
            # Nothing to choose between. Asking anyway would spend a call and a
            # prompt on a question with one answer.
            return self._declined("no candidates to choose from", raw="", latency_ms=0)

        body = {
            "model": self.model,
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_message(read, candidates)},
            ],
        }

        started = time.monotonic()
        try:
            payload = self._post(body)
        except (httpx.HTTPError, OSError) as exc:
            # D6: the model being unavailable is not a reason to trade anyway.
            return self._declined(
                f"{type(exc).__name__}: {exc}",
                raw="",
                latency_ms=_elapsed(started),
            )
        latency_ms = _elapsed(started)

        content, usage = _extract(payload)
        if content is None:
            return self._declined(
                "response contained no message content",
                raw=json.dumps(payload)[:MAX_RESPONSE_CHARS],
                latency_ms=latency_ms,
            )

        choice, error = parse_choice(content, len(candidates))
        return Proposal(
            choice=choice,
            call=ModelCall(
                model=self.model,
                prompt_version=PROMPT_VERSION,
                temperature=self.temperature,
                latency_ms=latency_ms,
                raw_response=content[:MAX_RESPONSE_CHARS],
                error=error,
                usage=usage,
            ),
        )

    # ------------------------------------------------------------------ #

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        if self.client is not None:
            response = self.client.post(url, json=body, headers=headers, timeout=self.timeout)
        else:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(url, json=body, headers=headers)
        response.raise_for_status()
        parsed = response.json()
        if not isinstance(parsed, dict):
            raise httpx.HTTPError(f"expected a JSON object, got {type(parsed).__name__}")
        return parsed

    def _declined(self, error: str, *, raw: str, latency_ms: int) -> Proposal:
        return Proposal(
            choice=Choice(None, f"declined: {error}", 0.0),
            call=ModelCall(
                model=self.model,
                prompt_version=PROMPT_VERSION,
                temperature=self.temperature,
                latency_ms=latency_ms,
                raw_response=raw,
                error=error,
            ),
        )


def parse_choice(content: str, menu_size: int) -> tuple[Choice, str | None]:
    """Read a model response into a `Choice`. Never raises.

    Every unusable shape becomes a decline with the reason recorded. specs/05 D3
    is explicit that an out-of-range index **is a decline, not an error to retry
    around** — a model naming a candidate that does not exist has not chosen, and
    asking it again is asking it to guess harder.

    A bare `float` index is refused rather than rounded: `2.7` is not evidence
    that the model meant 2 or 3, it is evidence that it was not answering the
    question asked. `2.0` is accepted, because JSON has one number type and a
    serialiser is allowed to write an integer that way.
    """
    try:
        parsed = json.loads(_strip_fences(content))
    except json.JSONDecodeError as exc:
        return Choice(None, "declined: response was not JSON", 0.0), f"invalid JSON: {exc}"

    if not isinstance(parsed, dict):
        return (
            Choice(None, "declined: response was not an object", 0.0),
            f"expected an object, got {type(parsed).__name__}",
        )

    rationale = str(parsed.get("rationale", "")).strip() or "no rationale given"
    confidence = _confidence(parsed.get("confidence"))
    raw_index = parsed.get("candidate_index", None)

    if raw_index is None:
        return Choice(None, rationale, confidence), None
    if isinstance(raw_index, bool):
        # bool is an int in Python; `true` is not an index.
        return Choice(None, rationale, confidence), f"candidate_index was {raw_index!r}"
    if isinstance(raw_index, float) and not raw_index.is_integer():
        return Choice(None, rationale, confidence), f"candidate_index was {raw_index!r}"
    if not isinstance(raw_index, (int, float)):
        return Choice(None, rationale, confidence), f"candidate_index was {raw_index!r}"

    index = int(raw_index)
    if not 0 <= index < menu_size:
        return (
            Choice(None, rationale, confidence),
            f"candidate_index {index} outside the menu of {menu_size}",
        )
    return Choice(index, rationale, confidence), None


def _strip_fences(content: str) -> str:
    """Unwrap ```json fences. Some models add them despite json mode."""
    text = content.strip()
    if not text.startswith("```"):
        return text
    body = text.split("\n", 1)[-1]
    return body.rsplit("```", 1)[0].strip()


def _confidence(value: object) -> float:
    """Clamped to [0, 1]. Recorded, never acted on — specs/05 D5."""
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    if number != number:  # NaN
        return 0.0
    return max(0.0, min(1.0, number))


def _extract(payload: dict[str, Any]) -> tuple[str | None, dict[str, int]]:
    usage_raw = payload.get("usage")
    usage = (
        {k: int(v) for k, v in usage_raw.items() if isinstance(v, (int, float))}
        if isinstance(usage_raw, dict)
        else {}
    )
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return None, usage
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    return (content if isinstance(content, str) and content.strip() else None), usage


def _elapsed(started: float) -> int:
    return int((time.monotonic() - started) * 1000)
