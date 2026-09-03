"""The entrypoint — `live/cli.py`.

Argument parsing and the two helpers that decide *which cycle this is*. The
commands themselves need a broker and are exercised by running them; what is
tested here is the part that was wrong when it shipped.

`once` used to take `slots[0]` unconditionally, so two manual runs in one
morning produced two decisions sharing a `cycle_id` and `Journal.read` collapsed
them into one. It happened on the first live run, and `duplicate_cycles`
reported it — which is the report working, and also a thing that should not need
reporting.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from alphagate.agent import SessionResult, Slot, Stage, session_slots
from alphagate.agent.book import read_book
from alphagate.agent.iv_store import IvHistoryStore
from alphagate.core.identifiers import ticker
from alphagate.execution import AccountRead
from alphagate.journal import Journal
from alphagate.live import cli
from alphagate.live.cli import (
    _adhoc_slot,
    _free_sequence,
    _latch_if_breached,
    _slot_now,
    build_parser,
    main,
)
from alphagate.live.wiring import OPTIONS_SLEEVE_BASIS, LiveContext, SessionState
from tests.journal.conftest import at_stage

OPEN = datetime(2026, 8, 26, 13, 30, tzinfo=UTC)
CLOSE = datetime(2026, 8, 26, 20, 0, tzinfo=UTC)
SLOTS = session_slots(OPEN, CLOSE)


class TestTheParser:
    @pytest.mark.parametrize(
        "command", ["preflight", "once", "run", "show", "serve", "iv-seed"]
    )
    def test_every_command_parses(self, command: str) -> None:
        assert build_parser().parse_args([command]).command == command

    def test_once_is_dry_by_default(self) -> None:
        """A debugging command that places orders is a debugging command that
        places orders by accident."""
        assert build_parser().parse_args(["once"]).submit is False

    def test_run_trades_by_default(self) -> None:
        """`run` is what a trading day is. Asking it to trade is not a
        surprise."""
        assert build_parser().parse_args(["run"]).dry_run is False

    def test_no_command_is_an_error_not_a_default_action(self) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args([])

    def test_the_env_path_is_overridable(self) -> None:
        args = build_parser().parse_args(["--env", "other.env", "preflight"])
        assert args.env == "other.env"


class TestWhichSlotThisIs:
    def test_it_picks_the_slot_the_schedule_would_be_running(self) -> None:
        chosen = _slot_now(SLOTS, datetime(2026, 8, 26, 15, 2, tzinfo=UTC))
        assert chosen.at >= datetime(2026, 8, 26, 15, 2, tzinfo=UTC)
        assert chosen.at.minute in {5, 20, 35, 50}

    def test_before_the_open_it_takes_the_first(self) -> None:
        assert _slot_now(SLOTS, OPEN).sequence == 0

    def test_after_the_close_it_takes_the_last(self) -> None:
        """Rather than raising. A cycle run after hours is still a cycle worth
        journalling — it will simply find nothing fresh to trade."""
        assert _slot_now(SLOTS, CLOSE).sequence == SLOTS[-1].sequence


class TestTheAdHocSlotIsEvaluatedNow:
    """`once` fetches a chain *now* and must judge its freshness against
    *now* -- not against the next scheduled slot, which is up to fifteen
    minutes away and would fail every quote on `max_quote_age` regardless of
    the market. See `_adhoc_slot`'s own docstring for the live bug this
    guards against.
    """

    def test_its_timestamp_is_now_not_the_next_scheduled_slot(self) -> None:
        now = datetime(2026, 8, 26, 15, 3, 34, tzinfo=UTC)
        scheduled = _slot_now(SLOTS, now)
        assert scheduled.at > now, "the fixture must actually be in the future"

        adhoc = _adhoc_slot(SLOTS, now)
        assert adhoc.at == now

    def test_its_identity_still_comes_from_the_schedule(self) -> None:
        """Only `.at` is corrected -- `kind` and `sequence` keep the identity
        `_slot_now` chose, which is what keeps `cycle_id` collision-free
        across repeated manual runs in one morning."""
        now = datetime(2026, 8, 26, 15, 3, 34, tzinfo=UTC)
        scheduled = _slot_now(SLOTS, now)
        adhoc = _adhoc_slot(SLOTS, now)
        assert adhoc.kind == scheduled.kind
        assert adhoc.sequence == scheduled.sequence

    def test_run_directly_at_a_slot_boundary_is_unaffected(self) -> None:
        """When `now` lands exactly on a scheduled slot there is no gap to
        close, and the fix must be a no-op rather than shifting anything."""
        on_the_dot = SLOTS[5].at
        assert _adhoc_slot(SLOTS, on_the_dot).at == on_the_dot


class TestSequencesDoNotCollide:
    def test_a_free_day_uses_the_slots_own_sequence(self, tmp_path: Path) -> None:
        context = _context(tmp_path)
        assert _free_sequence(context, SLOTS[3], "SPY") == 3

    def test_a_used_sequence_is_stepped_over(self, tmp_path: Path) -> None:
        """Two manual runs must not write one line."""
        context = _context(tmp_path)
        context.journal.append(at_stage(Stage.DRY_RUN, sequence=0))
        assert _free_sequence(context, SLOTS[0], "SPY") == 1

    def test_it_keeps_stepping(self, tmp_path: Path) -> None:
        context = _context(tmp_path)
        for sequence in range(3):
            context.journal.append(at_stage(Stage.DRY_RUN, sequence=sequence))
        assert _free_sequence(context, SLOTS[0], "SPY") == 3

    def test_another_underlying_does_not_block_this_one(self, tmp_path: Path) -> None:
        """`cycle_id` carries the name, so SPY-000 and QQQ-000 are different
        decisions and neither shadows the other."""
        context = _context(tmp_path)
        context.journal.append(at_stage(Stage.DRY_RUN, sequence=0))
        assert _free_sequence(context, SLOTS[0], "QQQ") == 0


def test_an_unknown_command_exits_nonzero() -> None:
    with pytest.raises(SystemExit):
        main(["nonsense"])


class TestIvSeed:
    """`iv-seed` — the other way into `iv_rank` inside a four-day window."""

    def _csv(self, tmp_path: Path, rows: str) -> Path:
        path = tmp_path / "vendor.csv"
        path.write_text(
            "date,act_symbol,iv_current\n" + rows, encoding="utf-8"
        )
        return path

    def test_it_seeds_the_store_and_reports_the_count(self, tmp_path: Path) -> None:
        csv_path = self._csv(
            tmp_path,
            "2026-08-26,SPY,0.12\n2026-08-27,SPY,0.13\n2026-08-28,SPY,0.11\n",
        )
        args = build_parser().parse_args(
            ["--iv", str(tmp_path / "iv"), "iv-seed", "--from", str(csv_path)]
        )
        assert cli.cmd_iv_seed(args) == 0
        store = IvHistoryStore(directory=tmp_path / "iv")
        assert store.observations(ticker("SPY")) == [0.12, 0.13, 0.11]

    def test_the_window_defaults_to_a_trailing_year(self, tmp_path: Path) -> None:
        """`--days` is load-bearing, not a convenience — a row far outside the
        vendor's own one-year range must not be seeded by default."""
        rows = "2019-01-01,SPY,0.90\n2026-08-28,SPY,0.11\n"
        csv_path = self._csv(tmp_path, rows)
        args = build_parser().parse_args(
            ["--iv", str(tmp_path / "iv"), "iv-seed", "--from", str(csv_path)]
        )
        assert cli.cmd_iv_seed(args) == 0
        store = IvHistoryStore(directory=tmp_path / "iv")
        assert store.observations(ticker("SPY")) == [0.11]

    def test_a_wider_days_window_reaches_further_back(self, tmp_path: Path) -> None:
        rows = "2019-01-01,SPY,0.90\n2026-08-28,SPY,0.11\n"
        csv_path = self._csv(tmp_path, rows)
        args = build_parser().parse_args(
            [
                "--iv", str(tmp_path / "iv"),
                "iv-seed", "--from", str(csv_path), "--days", "3000",
            ]
        )
        assert cli.cmd_iv_seed(args) == 0
        store = IvHistoryStore(directory=tmp_path / "iv")
        assert store.observations(ticker("SPY")) == [0.90, 0.11]

    def test_a_missing_file_is_reported_not_a_traceback(self, tmp_path: Path) -> None:
        args = build_parser().parse_args(
            ["--iv", str(tmp_path / "iv"), "iv-seed", "--from", str(tmp_path / "absent.csv")]
        )
        assert cli.cmd_iv_seed(args) == 1

    def test_re_running_is_idempotent(self, tmp_path: Path) -> None:
        csv_path = self._csv(tmp_path, "2026-08-28,SPY,0.11\n2026-08-27,SPY,0.12\n")
        args = build_parser().parse_args(
            ["--iv", str(tmp_path / "iv"), "iv-seed", "--from", str(csv_path)]
        )
        assert cli.cmd_iv_seed(args) == 0
        assert cli.cmd_iv_seed(args) == 0
        store = IvHistoryStore(directory=tmp_path / "iv")
        assert len(store.observations(ticker("SPY"))) == 2

    def test_the_default_from_path_points_at_the_sibling_projects_data(self) -> None:
        """Overridable, but the default is a real file this repo ships, not a
        placeholder — see `DEFAULT_VOLATILITY_HISTORY`'s own docstring for why
        this is a path and not an import."""
        args = build_parser().parse_args(["iv-seed"])
        assert "ai_quant_researcher" in args.frm
        assert args.symbol == "SPY"
        assert args.days == 365


def _context(tmp_path: Path) -> LiveContext:
    from alphagate.agent import IvHistoryStore
    from alphagate.marketdata import RecordedMarketData

    return LiveContext(
        data=RecordedMarketData(directory=tmp_path),
        mcp=None,
        journal=Journal(directory=tmp_path / "journal"),
        iv=IvHistoryStore(directory=tmp_path / "iv"),
        state=SessionState(path=tmp_path / "state.json"),
    )


class TestTheSupervisor:
    """A dropped connection at 14:00 must not cost the afternoon.

    The agent is a foreground process on a laptop for a week, and four trading
    days is the whole P&L sample. But resuming is not unconditionally right —
    two stops mean a human has to look before any more orders go out, and a
    supervisor that restarted through them would be actively harmful.
    """

    def test_a_dead_session_is_resumed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        attempts: list[int] = []

        def flaky(args: object, slots: object) -> SessionResult:
            attempts.append(1)
            if len(attempts) == 1:
                raise RuntimeError("transport went away")
            return SessionResult()

        monkeypatch.setattr(cli, "_one_session", flaky)
        code = cli.supervised_run(
            args=build_parser().parse_args(["run"]),
            open_at=OPEN,
            close_at=CLOSE,
            now=lambda: OPEN,
            sleep=lambda _s: None,
        )
        assert code == 0
        assert len(attempts) == 2, "it tried again"

    def test_it_gives_up_rather_than_spinning_forever(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A process respawning against a broken credential burns the day
        writing the same traceback."""
        attempts: list[int] = []

        def always_dies(args: object, slots: object) -> SessionResult:
            attempts.append(1)
            raise RuntimeError("still broken")

        monkeypatch.setattr(cli, "_one_session", always_dies)
        code = cli.supervised_run(
            args=build_parser().parse_args(["run"]),
            open_at=OPEN,
            close_at=CLOSE,
            now=lambda: OPEN,
            sleep=lambda _s: None,
        )
        assert code == 1
        assert len(attempts) == cli.MAX_RESTARTS + 1

    def test_a_breach_is_not_resumed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """specs/04 D5. A partial fill is a naked leg and a latched kill switch;
        resuming would put the agent back to work on a book nobody has seen."""
        attempts: list[int] = []

        def breached(args: object, slots: object) -> SessionResult:
            attempts.append(1)
            return SessionResult(
                stopped_early="partial fill breach — opens blocked, reconcile by hand"
            )

        monkeypatch.setattr(cli, "_one_session", breached)
        code = cli.supervised_run(
            args=build_parser().parse_args(["run"]),
            open_at=OPEN,
            close_at=CLOSE,
            now=lambda: OPEN,
            sleep=lambda _s: None,
        )
        assert code == 1
        assert len(attempts) == 1, "it stopped, it did not retry"

    def test_a_latched_killswitch_is_not_resumed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        attempts: list[int] = []

        def latched(args: object, slots: object) -> SessionResult:
            attempts.append(1)
            return SessionResult(stopped_early="kill switch latched on the incoming snapshot")

        monkeypatch.setattr(cli, "_one_session", latched)
        assert (
            cli.supervised_run(
                args=build_parser().parse_args(["run"]),
                open_at=OPEN,
                close_at=CLOSE,
                now=lambda: OPEN,
                sleep=lambda _s: None,
            )
            == 1
        )
        assert len(attempts) == 1

    def test_a_clean_session_is_not_restarted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        attempts: list[int] = []

        def clean(args: object, slots: object) -> SessionResult:
            attempts.append(1)
            return SessionResult()

        monkeypatch.setattr(cli, "_one_session", clean)
        assert (
            cli.supervised_run(
                args=build_parser().parse_args(["run"]),
                open_at=OPEN,
                close_at=CLOSE,
                now=lambda: OPEN,
                sleep=lambda _s: None,
            )
            == 0
        )
        assert len(attempts) == 1

    def test_a_finished_day_is_not_restarted_either(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """After the close there are no pending slots, so there is nothing to
        resume — the loop must notice rather than respawn against an empty
        schedule."""
        attempts: list[int] = []

        def never(args: object, slots: object) -> SessionResult:
            attempts.append(1)
            return SessionResult()

        monkeypatch.setattr(cli, "_one_session", never)
        code = cli.supervised_run(
            args=build_parser().parse_args(["run"]),
            open_at=OPEN,
            close_at=CLOSE,
            now=lambda: CLOSE + timedelta(hours=1),
            sleep=lambda _s: None,
        )
        assert code == 0
        assert attempts == []

    def test_only_pending_slots_are_resumed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Resumption is slot-based, which is what makes it safe: a restart
        costs the slot it happened in and never replays one that already has a
        journal line (specs/06 D2)."""
        seen: list[int] = []

        def record_slots(args: object, slots: Sequence[Slot]) -> SessionResult:
            seen.append(len(slots))
            return SessionResult()

        monkeypatch.setattr(cli, "_one_session", record_slots)
        midday = datetime(2026, 8, 26, 17, 0, tzinfo=UTC)
        cli.supervised_run(
            args=build_parser().parse_args(["run"]),
            open_at=OPEN,
            close_at=CLOSE,
            now=lambda: midday,
            sleep=lambda _s: None,
        )
        remaining = len([s for s in session_slots(OPEN, CLOSE) if s.at > midday])
        assert seen == [remaining]
        assert remaining < len(session_slots(OPEN, CLOSE))

    @pytest.mark.parametrize(
        ("reason", "terminal"),
        [
            ("partial fill breach — opens blocked, reconcile by hand", True),
            ("kill switch latched on the incoming snapshot", True),
            ("gather failed at 14:30 ConnectionError: reset", False),
            ("", False),
        ],
    )
    def test_which_stops_a_restart_must_not_paper_over(
        self, reason: str, terminal: bool
    ) -> None:
        assert cli._is_terminal_stop(reason) is terminal


class TestTheKillSwitchOutlivesTheProcess:
    """specs/04 D5: a partial fill blocks new opens "until a human clears it".

    `SessionState.killswitch_tripped` has been loaded on every start and handed
    to `read_book` since sleeves existed, and nothing ever wrote it. So the
    latch lived in one process: `supervised_run` refused to resume, and the next
    `alphagate run` -- an hour later or the next morning -- started clean and
    went back to opening positions with a naked leg outstanding. The equity path
    has always persisted it; this is the options half.
    """

    def context(self, tmp_path: Path) -> LiveContext:
        return LiveContext(
            data=None,  # type: ignore[arg-type]
            mcp=None,
            journal=Journal(directory=tmp_path / "journal"),
            iv=IvHistoryStore(directory=tmp_path / "iv"),
            state=SessionState(path=tmp_path / "state.json", basis=OPTIONS_SLEEVE_BASIS),
        )

    def stopped(self, reason: str) -> SessionResult:
        result = SessionResult()
        result.stopped_early = reason
        return result

    def test_a_breach_is_written_to_disk(self, tmp_path: Path) -> None:
        context = self.context(tmp_path)
        _latch_if_breached(
            context, self.stopped("partial fill breach - opens blocked, reconcile by hand")
        )
        reloaded = SessionState.load(tmp_path / "state.json", basis=OPTIONS_SLEEVE_BASIS)
        assert reloaded.killswitch_tripped is True

    def test_a_latch_on_the_incoming_snapshot_is_written_too(self, tmp_path: Path) -> None:
        """The second terminal stop. One day's latch has to reach the next."""
        context = self.context(tmp_path)
        _latch_if_breached(context, self.stopped("kill switch latched on the incoming snapshot"))
        assert context.state.killswitch_tripped is True

    def test_an_ordinary_end_of_day_latches_nothing(self, tmp_path: Path) -> None:
        context = self.context(tmp_path)
        _latch_if_breached(context, SessionResult())
        assert not (tmp_path / "state.json").exists()

    def test_a_survivable_failure_latches_nothing(self, tmp_path: Path) -> None:
        """A gather that failed three times is a bad connection, not a naked
        leg. `supervised_run` resumes those, and a latch would make a network
        blip cost the rest of the week."""
        context = self.context(tmp_path)
        _latch_if_breached(
            context, self.stopped("gather failed at slot 3: TransportFailure: reset")
        )
        assert context.state.killswitch_tripped is False

    def test_it_does_not_rewrite_a_latch_it_already_holds(self, tmp_path: Path) -> None:
        context = self.context(tmp_path)
        context.state.killswitch_tripped = True
        _latch_if_breached(context, self.stopped("partial fill breach"))
        assert not (tmp_path / "state.json").exists(), "nothing changed, nothing written"

    def test_the_latch_reaches_the_gate_on_the_next_start(self, tmp_path: Path) -> None:
        """The whole point, end to end: what is written is what `read_book` is
        handed, and `PortfolioSnapshot.killswitch_tripped` is what the Gate
        refuses opens on."""
        context = self.context(tmp_path)
        _latch_if_breached(context, self.stopped("partial fill breach"))

        tomorrow = SessionState.load(tmp_path / "state.json", basis=OPTIONS_SLEEVE_BASIS)
        assert tomorrow.killswitch_tripped is True
        book = read_book(
            _account(), (), [], killswitch_tripped=tomorrow.killswitch_tripped
        )
        assert book.snapshot.killswitch_tripped is True


def _account() -> AccountRead:
    return AccountRead(
        equity=Decimal("100000"),
        last_equity=Decimal("100000"),
        buying_power=Decimal("100000"),
        options_buying_power=Decimal("100000"),
        options_level=3,
        cash=Decimal("100000"),
        multiplier=1,
        is_blocked=False,
        envelope=None,
        observed_at=datetime(2026, 8, 26, 14, 30, tzinfo=UTC),
    )
