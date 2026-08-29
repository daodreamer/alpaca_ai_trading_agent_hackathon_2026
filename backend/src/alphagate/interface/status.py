"""Reading the agent's status snapshot — the dashboard side of `live/status.py`.

Deliberately a separate, tiny module that parses a dict rather than importing
`alphagate.live.status` and its dataclass. The type lives with the writer; the
reader takes JSON off disk and nothing else.

That is not fastidiousness. `tests/test_boundaries.py` forbids
`alphagate.interface` from importing `alphagate.live`, because `live` holds the
MCP session and the market data client — and "there is no code path from a
browser to an order" stops being checkable the moment the dashboard can import
the module that owns the broker. A shared dataclass would be exactly that
import.

The cost is one duplicated constant and no shared schema; the benefit is a guard
that still means something on demo day.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

__all__ = [
    "EQUITY_STALE_AFTER",
    "STALE_AFTER",
    "read_equity_status",
    "read_status",
]

STATUS_FILENAME = "status.json"
EQUITY_STATUS_FILENAME = "equity-status.json"

STALE_AFTER = 1200.0
"""Seconds before a snapshot is treated as "not running".

Cycles are fifteen minutes apart, so a healthy agent rewrites this every 900
seconds. Twenty minutes allows one slow slot — a cold chain request, a retried
submission — without crying wolf, and still notices a stopped agent inside one
cycle. Tighter than the cadence would flag every slow slot; much looser and a
crashed agent would look alive through lunch.
"""


EQUITY_STALE_AFTER = 120.0
"""Seconds before the *equity* snapshot is treated as "not running".

Much tighter than the options figure, because the cadences differ by an order of
magnitude: the equity agent heartbeats every thirty seconds whether or not it is
deciding anything (specs/09 D8), so two minutes of silence is four missed beats
rather than one slow slot.

The two numbers are separate for exactly that reason. Sharing one would either
make the equity page claim a dead agent was alive for twenty minutes, or make
the options page cry wolf on every slow chain request.
"""


def read_equity_status(directory: Path) -> dict[str, Any] | None:
    """The equity snapshot, or `None`. Same contract as `read_status`.

    A second file rather than a second key in the first, because the two agents
    are two processes: one file each means neither can overwrite the other's
    view of the account, and a page can say "options running, equity stopped"
    rather than having to guess which process last wrote a merged document.
    """
    return _load(directory / EQUITY_STATUS_FILENAME)


def read_status(directory: Path) -> dict[str, Any] | None:
    """The snapshot, or `None` if there is not a readable one.

    A corrupt or half-written file reads as `None` rather than raising. The
    dashboard's job in that moment is to say it has lost contact, not to return
    a 500 — and the writer writes atomically, so a torn read means something
    stranger than a race and the page should say so plainly.
    """
    return _load(directory / STATUS_FILENAME)


def _load(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return None
    return loaded if isinstance(loaded, dict) else None


def _age_of(stamp: Any) -> float | None:
    """Seconds since the snapshot was written, by this machine's clock.

    Computed on read rather than stored, so a stopped agent produces an ageing
    number instead of a frozen one that looks fresh forever.
    """
    if not isinstance(stamp, str):
        return None
    try:
        written = datetime.fromisoformat(stamp)
    except ValueError:
        return None
    if written.tzinfo is None:
        written = written.replace(tzinfo=UTC)
    return max(0.0, (datetime.now(UTC) - written).total_seconds())
