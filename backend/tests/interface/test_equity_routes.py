"""The equity routes — specs/09 D10.

Shallow, like the rest of `test_app.py`: the decisions live in
`live/equity_status.py` and are tested there. What is asserted here is the
contract the page depends on, and the property the whole layout exists to
protect.

**The dashboard still cannot trade.** A second agent's status is a second
temptation to hand the page a broker session so it can show a live book, and
`test_boundaries.py` is what makes refusing that a rule rather than a habit.
This suite is the behavioural half: the routes are all `GET`, and the equity
one reads a file.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from alphagate.interface.app import build_app
from alphagate.interface.status import EQUITY_STALE_AFTER
from alphagate.journal import Journal
from tests.journal.conftest import journal  # noqa: F401  - re-exported fixture


def write_status(directory: Path, *, as_of: datetime, **overrides: object) -> None:
    document: dict[str, object] = {
        "as_of": as_of.isoformat(),
        "session_day": as_of.date().isoformat(),
        "next_pass": None,
        "heartbeat_sequence": 7,
        "strategy": {
            "fingerprint": "3f6e2c8a9309068b",
            "name": "rs_volatility_consistency_neutral_v1",
            "sealed": {"t_alpha": 2.22, "looks": 1, "refuted": False},
        },
        "equity": "100000",
        "lines": [],
        "unpriced": [],
        "stale": [],
        "off_book": [],
        **overrides,
    }
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "equity-status.json").write_text(
        json.dumps(document), encoding="utf-8"
    )


@pytest.fixture
def client(journal: Journal) -> TestClient:  # noqa: F811 - the re-exported fixture
    return TestClient(build_app(journal.directory))


class TestEquityStatus:
    def test_no_file_reads_as_not_running(self, client: TestClient) -> None:
        """The agent has never run. That is a state, not a 500 — and the page
        says so rather than rendering an empty book as a flat one."""
        body = client.get("/api/equity/status").json()
        assert body == {"running": False, "snapshot": None}

    def test_a_fresh_heartbeat_reads_as_running(
        self, client: TestClient, journal: Journal  # noqa: F811
    ) -> None:
        write_status(journal.directory, as_of=datetime.now(UTC))
        body = client.get("/api/equity/status").json()
        assert body["running"] is True
        assert body["age_seconds"] < 5
        assert body["snapshot"]["strategy"]["fingerprint"] == "3f6e2c8a9309068b"

    def test_a_stale_heartbeat_reads_as_stopped(
        self, client: TestClient, journal: Journal  # noqa: F811
    ) -> None:
        """The staleness is computed on read rather than trusted from the file,
        so a stopped agent produces an ageing number instead of a frozen one
        that looks fresh forever."""
        old = datetime.fromtimestamp(
            datetime.now(UTC).timestamp() - EQUITY_STALE_AFTER - 60, tz=UTC
        )
        write_status(journal.directory, as_of=old)
        body = client.get("/api/equity/status").json()
        assert body["running"] is False
        assert body["age_seconds"] > EQUITY_STALE_AFTER
        assert body["snapshot"] is not None

    def test_the_equity_threshold_is_tighter_than_the_options_one(self) -> None:
        """Four missed thirty-second beats, not one slow fifteen-minute slot.

        Sharing one number would either let the equity page claim a dead agent
        was alive for twenty minutes, or make the options page cry wolf on every
        slow chain request.
        """
        from alphagate.interface.status import STALE_AFTER

        assert EQUITY_STALE_AFTER < STALE_AFTER

    def test_a_corrupt_file_reads_as_never_run(
        self, client: TestClient, journal: Journal  # noqa: F811
    ) -> None:
        journal.directory.mkdir(parents=True, exist_ok=True)
        (journal.directory / "equity-status.json").write_text("{tor", encoding="utf-8")
        assert client.get("/api/equity/status").json()["snapshot"] is None

    def test_the_two_agents_have_separate_files(
        self, client: TestClient, journal: Journal  # noqa: F811
    ) -> None:
        """So the page can say "options running, equity stopped" rather than
        guessing which process last wrote a merged document."""
        write_status(journal.directory, as_of=datetime.now(UTC))
        assert client.get("/api/equity/status").json()["running"] is True
        assert client.get("/api/status").json()["snapshot"] is None


class TestEquityDay:
    def test_only_equity_cycles_come_back(
        self, client: TestClient, journal: Journal  # noqa: F811
    ) -> None:
        """One journal, two agents. The filtered view means the Equity tab does
        not have to know what an options cycle looks like in order to skip
        one."""
        day = "2026-08-28"
        journal.append(
            {
                "cycle_id": f"{day}-SPY-001",
                "as_of": f"{day}T13:45:00+00:00",
                "stage": "no_candidates",
            }
        )
        journal.append(
            {
                "cycle_id": f"{day}-EQ-000",
                "kind": "equity",
                "as_of": f"{day}T13:45:00+00:00",
                "stage": "no_trades",
            }
        )
        body = client.get(f"/api/equity/day/{day}").json()
        assert [record["cycle_id"] for record in body] == [f"{day}-EQ-000"]

    def test_the_unfiltered_day_still_carries_both(
        self, client: TestClient, journal: Journal  # noqa: F811
    ) -> None:
        day = "2026-08-28"
        journal.append(
            {"cycle_id": f"{day}-SPY-001", "as_of": f"{day}T13:45:00+00:00"}
        )
        journal.append(
            {"cycle_id": f"{day}-EQ-000", "kind": "equity", "as_of": f"{day}T13:45:00+00:00"}
        )
        assert len(client.get(f"/api/day/{day}").json()) == 2

    def test_only_the_options_cycle_is_stamped_with_a_category(
        self, client: TestClient, journal: Journal  # noqa: F811
    ) -> None:
        """`category` is options' own decline taxonomy (specs/07 D1). Running
        it over an equity cycle's `kind` would mislabel a sleeve it was never
        about, so the equity record passes through unstamped."""
        day = "2026-08-28"
        journal.append(
            {"cycle_id": f"{day}-SPY-001", "as_of": f"{day}T13:45:00+00:00", "stage": "vetoed"}
        )
        journal.append(
            {
                "cycle_id": f"{day}-EQ-000",
                "kind": "equity",
                "as_of": f"{day}T13:45:00+00:00",
                "stage": "no_trades",
            }
        )
        by_id = {r["cycle_id"]: r for r in client.get(f"/api/day/{day}").json()}
        assert by_id[f"{day}-SPY-001"]["category"] == "gate_veto"
        assert "category" not in by_id[f"{day}-EQ-000"]

    def test_a_malformed_date_is_a_400(self, client: TestClient) -> None:
        assert client.get("/api/equity/day/not-a-date").status_code == 400

    def test_an_empty_day_is_an_empty_list(self, client: TestClient) -> None:
        assert client.get("/api/equity/day/2020-01-01").json() == []


def test_no_equity_route_accepts_a_write(client: TestClient) -> None:
    """The read-only property, asserted rather than assumed.

    A dashboard with a POST route one refactor away from `submit_equity` is a
    demo you cannot safely leave open on a projector.
    """
    routes = client.app.routes  # type: ignore[attr-defined]
    for route in routes:
        methods = getattr(route, "methods", set()) or set()
        assert methods <= {"GET", "HEAD"}, f"{route} accepts {methods}"
