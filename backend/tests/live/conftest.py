"""Fixtures for the live composition roots.

The equity suite reuses the target-book fixtures from `tests/equity/conftest.py`
rather than defining a second book. See `tests/risk/conftest.py` for why: two
fixtures for one artefact drift, and the drift lands between the thing the
planner is tested against and the thing the runner is tested against.
"""

from __future__ import annotations

from tests.equity.conftest import (  # noqa: F401
    book,
    book_payload,
    holdings,
    marks,
    policy,
    portfolio,
)
