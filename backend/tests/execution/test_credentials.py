"""The paper-account guard, and the env-name translation.

CLAUDE.md §3: paper trading only, never real money. A rule that important should
not live in a comment, so it lives in a function that raises — and a function
that raises deserves a test that proves it raises on each way of getting it
wrong.

The guard checks two independent signals, the key prefix and the trading URL,
and demands both. That redundancy is the point: either one alone can be edited
by accident, and the cost of guessing wrong is not a failed test.

Nothing in this file contains a real credential. The values are the right
*shape* and nothing more.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from alphagate.execution import (
    ExecutionError,
    load_env_file,
    mcp_environment,
    require_paper_account,
)
from alphagate.execution.credentials import PAPER_HOST

PAPER = {
    "ALPACA_API_KEY_ID": "PK000000000000000000000000",
    "ALPACA_API_SECRET_KEY": "CT0000000000000000000000000000000000000000000",
    "ALPACA_TRADING_URL": f"https://{PAPER_HOST}/v2",
}


def without(key: str) -> dict[str, str]:
    return {k: v for k, v in PAPER.items() if k != key}


class TestThePaperGuard:
    def test_a_paper_configuration_passes(self) -> None:
        require_paper_account(PAPER)  # does not raise

    def test_a_live_key_is_refused(self) -> None:
        """Live keys begin `AK`. This is the cheap second opinion."""
        with pytest.raises(ExecutionError, match="paper key"):
            require_paper_account({**PAPER, "ALPACA_API_KEY_ID": "AK00000000000000000000"})

    def test_a_live_endpoint_is_refused(self) -> None:
        """Even with a paper-looking key. Both signals must agree."""
        with pytest.raises(ExecutionError, match="never touches a live account"):
            require_paper_account(
                {**PAPER, "ALPACA_TRADING_URL": "https://api.alpaca.markets/v2"}
            )

    @pytest.mark.parametrize("missing", ["ALPACA_API_KEY_ID", "ALPACA_API_SECRET_KEY"])
    def test_missing_credentials_are_refused(self, missing: str) -> None:
        with pytest.raises(ExecutionError, match="must both be set"):
            require_paper_account(without(missing))

    def test_a_missing_url_is_refused_rather_than_assumed(self) -> None:
        """Absence of evidence is not evidence of paper. The failure mode of
        guessing wrong here is real money."""
        with pytest.raises(ExecutionError, match="not paper-api"):
            require_paper_account(without("ALPACA_TRADING_URL"))

    def test_an_empty_configuration_is_refused(self) -> None:
        with pytest.raises(ExecutionError):
            require_paper_account({})

    def test_the_error_never_contains_the_secret(self) -> None:
        """An exception string ends up in a log, and logs end up in a demo."""
        with pytest.raises(ExecutionError) as caught:
            require_paper_account({**PAPER, "ALPACA_API_KEY_ID": "AK00000000000000000000"})
        assert PAPER["ALPACA_API_SECRET_KEY"] not in str(caught.value)


class TestTheEnvironmentItBuilds:
    def test_it_translates_our_names_to_the_servers(self) -> None:
        """We use Alpaca's REST header names; alpaca-mcp-server 2.3.0 reads
        `ALPACA_API_KEY` / `ALPACA_SECRET_KEY`. Verified against its source."""
        env = mcp_environment(PAPER, inherit=False)
        assert env["ALPACA_API_KEY"] == PAPER["ALPACA_API_KEY_ID"]
        assert env["ALPACA_SECRET_KEY"] == PAPER["ALPACA_API_SECRET_KEY"]

    def test_paper_mode_is_forced_on(self) -> None:
        assert mcp_environment(PAPER, inherit=False)["ALPACA_PAPER_TRADE"] == "true"

    def test_a_live_configuration_never_becomes_an_environment(self) -> None:
        """Refused at build time, not built and then hopefully unused."""
        with pytest.raises(ExecutionError):
            mcp_environment({**PAPER, "ALPACA_API_KEY_ID": "AK00000000000000000000"})

    def test_inheriting_the_process_environment_is_optional(self) -> None:
        assert set(mcp_environment(PAPER, inherit=False)) == {
            "ALPACA_API_KEY",
            "ALPACA_SECRET_KEY",
            "ALPACA_PAPER_TRADE",
        }


class TestEnvFileParsing:
    def test_it_reads_keys_and_values(self, tmp_path: Path) -> None:
        path = tmp_path / ".env.local"
        path.write_text(
            "# a comment\n"
            "\n"
            "ALPACA_API_KEY_ID=PK123\n"
            'ALPACA_API_SECRET_KEY="quoted"\n'
            "ALPACA_TRADING_URL=https://paper-api.alpaca.markets/v2\n"
            "MALFORMED_LINE\n",
            encoding="utf-8",
        )
        values = load_env_file(path)
        assert values["ALPACA_API_KEY_ID"] == "PK123"
        assert values["ALPACA_API_SECRET_KEY"] == "quoted"
        assert "MALFORMED_LINE" not in values

    def test_a_value_containing_an_equals_sign_survives(self, tmp_path: Path) -> None:
        """Base64 secrets end in `=` padding. Splitting on every `=` truncates
        the key and produces an authentication failure nobody can explain."""
        path = tmp_path / ".env"
        path.write_text("K=abc==\n", encoding="utf-8")
        assert load_env_file(path)["K"] == "abc=="

    def test_the_real_env_file_is_a_paper_configuration(self) -> None:
        """The one test here that touches the actual file.

        It reads it and asserts the guard passes; it does not print, log, or
        assert on any value. Skipped rather than failed when absent, because a
        fresh clone has no `.env.local` and this suite must stay runnable.
        """
        env_path = Path(__file__).resolve().parents[3] / ".env.local"
        if not env_path.is_file():
            pytest.skip("no .env.local in this checkout")
        require_paper_account(load_env_file(env_path))
