"""Choosing which gateway to talk to.

TWS listens on 7497 for a paper login and 7496 for a live one; IB Gateway uses
4002 and 4001. Which of the four is correct is a property of the running
process, not of this code, so it has to be an option rather than a constant.

The port also decides which account the connection lands on, and one of the four
is the real one. Nothing here places an order -- ``IbkrProvider`` calls
``reqHistoricalData`` and nothing else -- but "the code has no order path" is an
argument that has to be re-made every time the code changes, so the CLI says
plainly which endpoint it is about to use before it uses it.
"""

from __future__ import annotations

import pytest

from aqr.cli import _provider, describe_endpoint
from aqr.data.ibkr import GATEWAY_PAPER_PORT, TWS_PAPER_PORT, IbkrProvider


class TestProviderTakesAnEndpoint:
    def test_the_default_is_the_paper_port(self) -> None:
        assert IbkrProvider().port == TWS_PAPER_PORT == 7497

    def test_host_port_and_client_id_reach_the_provider(self) -> None:
        provider = _provider(
            "ibkr", "data", ibkr_host="10.0.0.5", ibkr_port=7496, ibkr_client_id=42
        )
        assert isinstance(provider, IbkrProvider)
        assert (provider.host, provider.port, provider.client_id) == ("10.0.0.5", 7496, 42)

    def test_other_sources_ignore_the_ibkr_options(self) -> None:
        # The options are on one shared command; passing them with --source
        # yahoo must not become an error a user has to think about.
        assert _provider("synthetic", "data", ibkr_port=7496) is not None


class TestNamingTheEndpoint:
    @pytest.mark.parametrize(
        ("port", "expected"),
        [
            (7497, "TWS paper"),
            (7496, "TWS LIVE"),
            (4002, "IB Gateway paper"),
            (4001, "IB Gateway LIVE"),
        ],
    )
    def test_each_known_port_is_named(self, port: int, expected: str) -> None:
        assert expected in describe_endpoint(port)

    def test_a_live_port_is_flagged_in_upper_case(self) -> None:
        """Whatever else a user skims past, they will not skim past LIVE."""
        for live in (7496, 4001):
            assert "LIVE" in describe_endpoint(live)
        for paper in (TWS_PAPER_PORT, GATEWAY_PAPER_PORT):
            assert "LIVE" not in describe_endpoint(paper)

    def test_an_unknown_port_is_described_rather_than_guessed(self) -> None:
        note = describe_endpoint(9999)
        assert "9999" in note
        assert "LIVE" not in note
