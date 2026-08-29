"""Credentials for the MCP subprocess — and the paper-account guard.

Two jobs, both small, both easy to get wrong in a way that is expensive.

**Name translation.** `.env.local` uses Alpaca's REST header names
(`ALPACA_API_KEY_ID` / `ALPACA_API_SECRET_KEY`); alpaca-mcp-server 2.3.0 reads
`ALPACA_API_KEY` / `ALPACA_SECRET_KEY`. Verified by reading the server's own
source, not guessed. The mapping lives here so that exactly one module knows
about the discrepancy, and so a rename upstream is a one-line change.

**The paper guard.** `require_paper_account` refuses to build an environment
that could reach a live account. CLAUDE.md is unambiguous: paper trading only,
never real money. That rule is worth more than a comment, so it is a function
that raises, checked against two independent signals — the key prefix and the
trading URL — because either one alone can be edited by accident.

Nothing here logs, formats, reprs or returns a secret in an error message. The
values pass from a file into a subprocess environment and are never rendered.
specs/06 D4: the demo video shows this terminal.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Final

from alphagate.execution.errors import ExecutionError

__all__ = [
    "PAPER_HOST",
    "load_env_file",
    "mcp_environment",
    "require_paper_account",
]

PAPER_HOST: Final = "paper-api.alpaca.markets"
_PAPER_KEY_PREFIX: Final = "PK"
"""Alpaca paper keys begin `PK`; live keys begin `AK`. A cheap second opinion."""

_KEY_ID: Final = "ALPACA_API_KEY_ID"
# S105 flags the name, not a value: these are env-var keys, and the whole
# point of the module is that no secret is ever written down here.
_SECRET: Final = "ALPACA_API_SECRET_KEY"  # noqa: S105
_TRADING_URL: Final = "ALPACA_TRADING_URL"

_SERVER_KEY: Final = "ALPACA_API_KEY"
_SERVER_SECRET: Final = "ALPACA_SECRET_KEY"  # noqa: S105
_SERVER_PAPER: Final = "ALPACA_PAPER_TRADE"


def load_env_file(path: str | Path) -> dict[str, str]:
    """Parse a `KEY=value` file. No interpolation, no export, no shell.

    Deliberately not `python-dotenv`: this is fifteen lines, and a dependency
    that can execute shell syntax is a dependency that can execute shell syntax.
    """
    values: dict[str, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def require_paper_account(env: Mapping[str, str]) -> None:
    """Refuse anything that is not the dedicated paper account.

    Raises on the *absence* of positive evidence, not on the presence of danger.
    A missing trading URL is not "probably paper"; it is a configuration nobody
    checked, and the failure mode of guessing wrong is real money.
    """
    key_id = env.get(_KEY_ID, "")
    url = env.get(_TRADING_URL, "")

    if not key_id or not env.get(_SECRET):
        raise ExecutionError(
            f"{_KEY_ID} and {_SECRET} must both be set before opening a session"
        )
    if not key_id.startswith(_PAPER_KEY_PREFIX):
        raise ExecutionError(
            f"{_KEY_ID} does not look like a paper key (expected a {_PAPER_KEY_PREFIX} "
            "prefix). AlphaGate never touches a live account — CLAUDE.md §3."
        )
    if PAPER_HOST not in url:
        raise ExecutionError(
            f"{_TRADING_URL} is {url!r}, which is not {PAPER_HOST}. "
            "AlphaGate never touches a live account — CLAUDE.md §3."
        )


def mcp_environment(env: Mapping[str, str], *, inherit: bool = True) -> dict[str, str]:
    """Build the subprocess environment for alpaca-mcp-server.

    Checks the paper guard first: an environment that could reach a live account
    is not built at all, rather than built and then hopefully not used.
    """
    require_paper_account(env)
    base = dict(os.environ) if inherit else {}
    base[_SERVER_KEY] = env[_KEY_ID]
    base[_SERVER_SECRET] = env[_SECRET]
    base[_SERVER_PAPER] = "true"
    return base
