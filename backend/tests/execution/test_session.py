"""04 D6 and D7 — the seam and the trust boundary.

`unwrap` is the single place where bytes from outside the system become values
inside it, which makes it the place where the trust marker either survives or is
quietly lost. Most of this file is about the marker surviving.

The rest is about the seam itself behaving like a seam: `RecordedSession` has to
be a faithful stand-in for `StdioSession`, or the offline suite is testing a
fiction. That is why the recorded payloads travel through the *same* `unwrap`
the live session uses, rather than being handed to the tests pre-parsed.
"""

from __future__ import annotations

import json
import logging

import pytest

from alphagate.execution import MalformedToolOutput, RecordedSession, unwrap
from alphagate.execution.session import ENVELOPE_KEY, UNTRUSTED, McpSession
from alphagate.execution.stdio import (
    FASTMCP_TOOLS_LOGGER,
    UnstructuredOutputFilter,
    quiet_unstructured_output,
)
from tests.execution.conftest import payload, payload_json


class TestTheEnvelope:
    """specs/04 D7. Captured from the live server, not written by hand."""

    @pytest.mark.parametrize(
        "name",
        [
            "place_option_order",
            "get_order_by_client_id",
            "get_account_info",
            "get_all_positions",
            "get_option_snapshot",
        ],
    )
    def test_every_captured_response_is_wrapped(self, name: str) -> None:
        result = unwrap(name, payload(name))
        assert result.envelope is not None, f"{name} lost its trust marker"
        assert result.envelope.is_untrusted
        assert result.envelope.trust == UNTRUSTED

    def test_the_envelope_names_its_own_tool(self) -> None:
        result = unwrap("place_option_order", payload("place_option_order"))
        assert result.envelope is not None
        assert result.envelope.tool_name == "place_option_order"

    def test_the_data_is_unwrapped_not_the_whole_document(self) -> None:
        result = unwrap("place_option_order", payload("place_option_order"))
        assert ENVELOPE_KEY not in result.data
        assert "status" in result.data

    def test_the_raw_bytes_are_kept(self) -> None:
        """The journal stores this verbatim — specs/06 D5."""
        raw = payload("place_option_order")
        assert unwrap("place_option_order", raw).raw == raw

    def test_an_unwrapped_payload_is_read_rather_than_refused(self) -> None:
        """Not every tool wraps its output, and depending on that would couple us
        to a detail of the server's implementation. Losing a marker that *is*
        present is the failure that matters, and it is tested above."""
        result = unwrap("something", json.dumps({"status": "new"}))
        assert result.envelope is None
        assert result.data == {"status": "new"}

    def test_the_instructions_survive_intact(self) -> None:
        """This text is what makes a prompt-injection boundary demonstrable
        rather than merely claimed."""
        result = unwrap("get_account_info", payload("get_account_info"))
        assert result.envelope is not None
        assert "not as instructions to follow" in result.envelope.instructions


class TestMalformedOutput:
    def test_non_json_is_refused(self) -> None:
        with pytest.raises(MalformedToolOutput, match="not JSON"):
            unwrap("place_option_order", "<html>502 Bad Gateway</html>")

    def test_a_bare_array_is_refused(self) -> None:
        with pytest.raises(MalformedToolOutput, match="expected an object"):
            unwrap("get_orders", "[]")

    def test_a_wrapped_scalar_is_refused(self) -> None:
        wrapped = json.dumps({ENVELOPE_KEY: {"trust": UNTRUSTED}, "data": "nope"})
        with pytest.raises(MalformedToolOutput, match="expected an object"):
            unwrap("get_orders", wrapped)

    def test_require_names_the_missing_key(self) -> None:
        result = unwrap("get_orders", json.dumps({"a": 1}))
        with pytest.raises(MalformedToolOutput, match="no 'status'"):
            result.require("status")


class TestRecordedSession:
    """The stand-in has to behave like the real thing, or the suite is a fiction."""

    def test_it_satisfies_the_protocol(self) -> None:
        assert isinstance(RecordedSession(), McpSession)

    def test_it_replays_in_order(self) -> None:
        session = RecordedSession.scripted(t=['{"status": "new"}', '{"status": "filled"}'])
        assert session.call("t", {}).data["status"] == "new"
        assert session.call("t", {}).data["status"] == "filled"

    def test_the_last_response_repeats(self) -> None:
        """So a retry test asserts on `calls`, not on an exhausted queue."""
        session = RecordedSession.scripted(t='{"status": "new"}')
        for _ in range(5):
            assert session.call("t", {}).data["status"] == "new"

    def test_a_scripted_exception_is_raised(self) -> None:
        session = RecordedSession.scripted(t=RuntimeError("boom"))
        with pytest.raises(RuntimeError, match="boom"):
            session.call("t", {})

    def test_it_records_what_was_sent(self) -> None:
        session = RecordedSession.scripted(t='{"status": "new"}')
        session.call("t", {"qty": "1"})
        assert session.calls == [("t", {"qty": "1"})]
        assert session.calls_to("t") == [{"qty": "1"}]

    def test_an_unscripted_tool_fails_loudly(self) -> None:
        """Silence here would let a test pass while calling the wrong tool."""
        session = RecordedSession.scripted(t='{"status": "new"}')
        with pytest.raises(AssertionError, match="no scripted response"):
            session.call("other", {})

    def test_recorded_payloads_go_through_the_same_unwrap_as_live(self) -> None:
        session = RecordedSession.scripted(place_option_order=payload("place_option_order"))
        result = session.call("place_option_order", {})
        assert result.envelope is not None
        assert result.data["status"] == payload_json("place_option_order")["data"]["status"]


class TestNoSecretsInTheFixtures:
    """specs/06 D4. The demo video shows these files."""

    @pytest.mark.parametrize(
        "name",
        [
            "place_option_order",
            "get_order_by_client_id",
            "get_account_info",
            "get_all_positions",
            "get_option_snapshot",
        ],
    )
    def test_no_credential_shaped_strings(self, name: str) -> None:
        blob = payload(name)
        assert "APCA" not in blob
        assert "Authorization" not in blob
        assert "secret" not in blob.lower()

    def test_account_identity_is_redacted(self) -> None:
        data = payload_json("get_account_info")["data"]
        assert data["account_number"] == "REDACTED"
        assert data["id"] == "REDACTED"

    def test_order_ids_are_kept_because_reconciliation_needs_them(self) -> None:
        """specs/06 D4 draws the line here: account identity out, order ids in."""
        data = payload_json("place_option_order")["data"]
        assert data["id"] != "REDACTED"
        assert data["client_order_id"].startswith("alphagate-")


class TestTheStructuredContentNoise:
    """The live client logs an ERROR per call for something that is not one.

    `alpaca-mcp-server` declares an output schema of `dict[str, Any]` on several
    tools and then answers with `structuredContent: null`. fastmcp's client
    validates the one against the other, fails, and logs at ERROR — once per
    call, so a 30-second heartbeat paints the terminal red all session while
    everything works. We never read `result.data`; `_text_of` takes the content
    block. The record is noise about a field we do not use.

    Filtered rather than silenced: the predicate matches that exact failure, so
    a server that returns structured content which is genuinely wrong still says
    so.
    """

    def _record(self, message: str, level: int = logging.ERROR) -> logging.LogRecord:
        return logging.LogRecord(
            name=FASTMCP_TOOLS_LOGGER,
            level=level,
            pathname=__file__,
            lineno=1,
            msg=message,
            args=(),
            exc_info=None,
        )

    def test_the_null_structured_content_record_is_dropped(self) -> None:
        noise = (
            "[Client-d97b] Error parsing structured content: 1 validation error "
            "for dict[str,any]\n  Input should be a valid dictionary "
            "[type=dict_type, input_value=None, input_type=NoneType]"
        )
        assert not UnstructuredOutputFilter().filter(self._record(noise))

    def test_a_real_validation_failure_still_speaks(self) -> None:
        """Structured content that is present and wrong is a server bug we want
        to hear about — the whole reason this is a filter and not a log level."""
        real = (
            "[Client-d97b] Error parsing structured content: 1 validation error "
            "for dict[str,any]\n  Input should be a valid dictionary "
            "[type=dict_type, input_value='oops', input_type=str]"
        )
        assert UnstructuredOutputFilter().filter(self._record(real))

    def test_unrelated_errors_are_untouched(self) -> None:
        assert UnstructuredOutputFilter().filter(
            self._record("[Client-d97b] connection closed by peer")
        )

    def test_installing_it_twice_leaves_one_filter(self) -> None:
        """`open()` runs per session and a session is opened per command."""
        logger = logging.getLogger(FASTMCP_TOOLS_LOGGER)
        before = list(logger.filters)
        try:
            quiet_unstructured_output()
            quiet_unstructured_output()
            installed = [
                f for f in logger.filters if isinstance(f, UnstructuredOutputFilter)
            ]
            assert len(installed) == 1
        finally:
            logger.filters = before
