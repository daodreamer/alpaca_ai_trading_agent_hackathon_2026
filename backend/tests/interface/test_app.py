"""The dashboard — `interface/app.py`.

Routing and HTML, so the tests are correspondingly shallow: `read.py` holds
every decision worth arguing about and is tested properly next door.

Two things here are not shallow.

**The read-only property.** There is no route that can place an order, and the
absence is asserted rather than assumed — a dashboard with a POST route one
refactor away from `submit` is a demo you cannot safely leave open on a
projector.

**Escaping.** A model rationale is untrusted text (specs/06 D5) that goes
straight into a page. It must not be able to close a tag.
"""

from __future__ import annotations

from datetime import date

import pytest
from fastapi.testclient import TestClient

from alphagate.agent import Stage
from alphagate.interface.app import build_app
from alphagate.journal import Journal
from tests.interface.test_read import equity_pass
from tests.journal.conftest import DAY, at_stage, cycle


@pytest.fixture
def client(journal: Journal) -> TestClient:
    journal.append(cycle(with_setup=False))
    journal.append(at_stage(Stage.VETOED, sequence=1))
    journal.append(at_stage(Stage.FILLED, sequence=2))
    return TestClient(build_app(journal.directory))


class TestRoutes:
    def test_health(self, client: TestClient) -> None:
        assert client.get("/healthz").json()["ok"] is True

    def test_the_index_serves_a_page(self, client: TestClient) -> None:
        """`/` is the React dashboard once the frontend is built, and the
        server-rendered day when it is not. Both are real UIs, so this asserts
        only that something coherent comes back — the day view has its own
        tests below and the SPA has its own suite."""
        response = client.get("/")
        assert response.status_code == 200
        assert "<html" in response.text.lower()

    def test_the_day_page_lists_every_cycle(self, client: TestClient) -> None:
        body = client.get(f"/day/{DAY}").text
        for sequence in ("000", "001", "002"):
            assert f"2026-08-26-SPY-{sequence}" in body

    def test_the_quiet_cycles_are_on_the_page(self, client: TestClient) -> None:
        """specs/06 D2. They are the majority and they are the point.

        Asserted on the rendered label rather than the stored token: the page
        exists to be read, and `no_setup` on a badge tells a reader nothing
        about whether the system is healthy. The token stays in the journal,
        where it is matched on; the page says what it means.
        """
        body = client.get(f"/day/{DAY}").text
        assert "no opportunity" in body
        assert "no_setup" not in body

    def test_a_cycle_page_carries_the_menu_and_the_checks(self, client: TestClient) -> None:
        body = client.get(f"/cycle/{DAY}/2026-08-26-SPY-002").text
        assert "the menu" in body
        assert "the gate" in body
        assert "checks passed" in body

    def test_a_cycle_page_carries_the_trust_boundary(self, client: TestClient) -> None:
        body = client.get(f"/cycle/{DAY}/2026-08-26-SPY-002").text
        assert "trust boundary" in body

    def test_the_json_is_the_line_off_disk(self, client: TestClient) -> None:
        payload = client.get(f"/api/day/{DAY}").json()
        assert [record["cycle_id"] for record in payload] == [
            "2026-08-26-SPY-000",
            "2026-08-26-SPY-001",
            "2026-08-26-SPY-002",
        ]

    def test_the_json_carries_its_own_decline_classification(
        self, client: TestClient
    ) -> None:
        """The single-classifier fix: the React app reads `category` off this
        response rather than re-deriving it, so `/day/...` and `/api/day/...`
        cannot disagree about why a cycle declined."""
        payload = client.get(f"/api/day/{DAY}").json()
        by_id = {record["cycle_id"]: record for record in payload}
        assert by_id["2026-08-26-SPY-001"]["category"] == "gate_veto"
        assert by_id["2026-08-26-SPY-002"]["category"] == "traded"
        assert by_id["2026-08-26-SPY-000"]["category_label"]
        assert by_id["2026-08-26-SPY-000"]["category_detail"]

    def test_an_unknown_cycle_is_404(self, client: TestClient) -> None:
        assert client.get(f"/cycle/{DAY}/nope").status_code == 404

    def test_a_bad_date_is_400_not_500(self, client: TestClient) -> None:
        assert client.get("/day/not-a-date").status_code == 400

    def test_an_empty_day_renders(self, client: TestClient) -> None:
        response = client.get(f"/day/{date(2020, 1, 1)}")
        assert response.status_code == 200

    def test_an_empty_journal_directory_renders_an_explanation(
        self, tmp_path: object
    ) -> None:
        """The server-rendered fallback, which is what a checkout with no
        frontend build gets. Asserted on `/day/...` rather than `/` because the
        latter is the SPA whenever `static/` exists."""
        from datetime import date
        from pathlib import Path

        empty = TestClient(build_app(Path(str(tmp_path)) / "nothing"))
        response = empty.get(f"/day/{date(2020, 1, 1)}")
        assert response.status_code == 200


class TestBothSleevesOnOnePage:
    """The day page covers a day, and two agents write to it.

    The options table cannot render an equity pass — it would print a menu of
    zero and "never reached the Gate" about a pass carrying a full check tape
    per order — and dropping the pass instead would hide a record that exists.
    So it gets its own summary panel, and this asserts both halves.
    """

    @pytest.fixture
    def both(self, journal: Journal) -> TestClient:
        journal.append(at_stage(Stage.FILLED))
        journal.append(equity_pass())
        return TestClient(build_app(journal.directory))

    def test_the_pass_is_on_the_page(self, both: TestClient) -> None:
        page = both.get(f"/day/{DAY}").text
        assert "equity sleeve" in page
        assert "2026-08-26-EQ-000" in page
        assert "2 of 2 intents submitted" in page

    def test_it_is_not_in_the_options_table(self, both: TestClient) -> None:
        """It has no cycle page, and nothing offers one."""
        page = both.get(f"/day/{DAY}").text
        assert f"/cycle/{DAY}/2026-08-26-EQ-000" not in page
        assert both.get(f"/cycle/{DAY}/2026-08-26-EQ-000").status_code == 404

    def test_the_api_still_hands_back_both(self, both: TestClient) -> None:
        """`/api/day` is the whole day — the React tab splits it by `kind`."""
        kinds = [record.get("kind") for record in both.get(f"/api/day/{DAY}").json()]
        assert kinds == [None, "equity"]


class TestItCannotTrade:
    """The property that makes it safe to leave open during a demo."""

    def test_there_are_no_write_routes(self, client: TestClient) -> None:
        methods: set[str] = set()
        for route in client.app.routes:  # type: ignore[attr-defined]
            methods |= set(getattr(route, "methods", ()) or ())
        assert methods <= {"GET", "HEAD"}

    def test_posting_anywhere_is_refused(self, client: TestClient) -> None:
        assert client.post(f"/day/{DAY}").status_code == 405

    def test_it_imports_nothing_that_can_place_an_order(self) -> None:
        """Structural, not behavioural, and checked on the imports rather than
        the text — the first draft of this grepped the source and failed on its
        own docstring, which is a test that measures prose.

        The package-wide version of this guard lives in `test_boundaries.py`;
        this is the fast local copy."""
        import ast
        from pathlib import Path

        import alphagate.interface.app as module

        assert module.__file__ is not None
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
            elif isinstance(node, ast.Import):
                imported |= {alias.name for alias in node.names}

        forbidden = {"alphagate.execution", "alphagate.marketdata", "alphagate.live"}
        assert not {name for name in imported if any(
            name.startswith(bad) for bad in forbidden
        )}, f"the dashboard must not import a way to trade: {sorted(imported)}"


class TestUntrustedTextIsEscaped:
    def test_a_rationale_cannot_close_a_tag(self, journal: Journal) -> None:
        """A model rationale is untrusted text that goes straight into a page."""
        from alphagate.agent import Choice
        from tests.journal.conftest import FixedProposer

        nasty = "</div><script>alert(1)</script>"
        journal.append(cycle(proposer=FixedProposer(Choice(0, nasty, 0.5))))
        client = TestClient(build_app(journal.directory))
        body = client.get(f"/cycle/{DAY}/2026-08-26-SPY-000").text
        assert "<script>alert(1)</script>" not in body
        assert "&lt;script&gt;" in body
