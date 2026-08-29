"""`python -m alphagate <command>` — the entrypoint.

Two agents share this entry point, and the commands are grouped by which.

```
alphagate preflight          # are the four hard gates actually satisfied?
alphagate once               # one options cycle, right now, dry by default
alphagate run                # a whole options session, on the schedule
alphagate serve              # the dashboard over the journal

alphagate equity-preflight   # is there a target book, and may it be executed?
alphagate equity-plan        # one rebalance pass, gated, nothing sent
alphagate equity-rebalance   # one rebalance pass, orders placed
alphagate equity-run         # an equity session: heartbeat and one pass
alphagate equity-status      # the book, and how far it has drifted
```

The `equity-*` half executes the strategy `ai_quant_researcher` validated, read
from a target-book file (specs/09). It is built in `live/equity_cli.py`, because
the two agents share a process and nothing else: one is a fifteen-minute loop
over an option chain, the other a daily rebalance against a file.

**`preflight` exists because three of specs/00's four hard gates are silent
failures.** A live key, an account that is not the dedicated one, an options
level below 3 — none of those announce themselves until an order is rejected at
14:30, by which point the trading day is half gone. It checks them against the
broker, prints a line per gate, and exits non-zero if any of them fails. Run it
before the open, every day, and the answer is a fact rather than a memory.

**`once` is dry by default and `run` is not.** The asymmetry is deliberate:
`once` is what you type while debugging, and a debugging command that places
orders is a debugging command that places orders by accident. `run` is what a
trading day is, and asking it to trade is not a surprise.

Nothing here prints a key, an account number, or an account id — specs/06 D4,
and the demo video shows this terminal.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from alphagate.agent import (
    DefaultScreen,
    SessionResult,
    Slot,
    Stage,
    run_cycle,
    run_session,
    session_slots,
    tradeable_today,
)
from alphagate.agent.iv_store import IvHistoryStore
from alphagate.execution import ExecutionError, load_env_file, require_paper_account
from alphagate.interface.status import STALE_AFTER, _age_of, read_status
from alphagate.journal import Journal, trust_report
from alphagate.live.equity_cli import add_equity_commands
from alphagate.live.wiring import LiveContext, SessionState, build_market_data, gather_for
from alphagate.live.wiring import mcp_session as open_mcp

__all__ = ["main"]

OK = "  ok  "
FAIL = " FAIL "
WARN = " warn "

def _repo_root() -> Path:
    """The directory holding `specs/`, walking up from this file.

    The defaults below have to be anchored to something, and the working
    directory is the wrong thing: `uv run --directory backend` puts the process
    in `backend/` while `.env.local` and `journal/` live one level up, so a
    plain `python -m alphagate preflight` would look for credentials in the
    wrong place and report a missing env file.

    Anchoring to the repo means the commands work from anywhere, which matters
    more than it sounds: the ones you type under pressure at 09:25 are the ones
    that must not need flags.
    """
    for parent in Path(__file__).resolve().parents:
        if (parent / "specs").is_dir():
            return parent
    return Path.cwd()


ROOT = _repo_root()

DEFAULT_ENV = str(ROOT / ".env.local")
DEFAULT_JOURNAL = str(ROOT / "journal")
DEFAULT_STATE = str(ROOT / "journal" / "state.json")
DEFAULT_IV = str(ROOT / "journal" / "iv")
DEFAULT_EQUITY_STATE = str(ROOT / "journal" / "equity-state.json")
DEFAULT_TARGET_BOOKS = str(ROOT / "ai_quant_researcher" / "runs" / "target_books")
"""Where `aqr target-book` writes, and the only line in AlphaGate that knows the
sibling project has a directory.

A path, not an import. specs/09 D0 makes the artefact the whole interface
between the two projects, and `tests/test_boundaries.py` guard 9 fails the build
if either ever imports the other. Overridable with `--books` or
`ALPHAGATE_TARGET_BOOKS`, so the two can live anywhere relative to each other."""

_MODEL_KEY_NOTE = "a cycle with no model declines rather than trades (--no-model)"


# ------------------------------------------------------------------ #
# preflight — specs/00's hard gates, checked rather than remembered
# ------------------------------------------------------------------ #


def cmd_preflight(args: argparse.Namespace) -> int:
    """Verify the four hard gates and the things that silently break a day."""
    failures = 0
    env: dict[str, str] = {}

    def check(label: str, passed: bool, detail: str = "", fatal: bool = True) -> None:
        nonlocal failures
        mark = OK if passed else (FAIL if fatal else WARN)
        print(f"[{mark}] {label}{'  — ' + detail if detail else ''}")
        if not passed and fatal:
            failures += 1

    print(f"AlphaGate pre-flight — {datetime.now(UTC).isoformat(timespec='seconds')}\n")

    # -- 1. credentials present -------------------------------------- #
    path = Path(args.env)
    if not path.is_file():
        check(f"env file {args.env}", False, "not found")
        return 1
    env = load_env_file(path)
    required = ("ALPACA_API_KEY_ID", "ALPACA_API_SECRET_KEY")
    check("credentials present", all(env.get(k) for k in required))

    # -- 2. HARD GATE 4: paper account only --------------------------- #
    try:
        require_paper_account(env)
        check("paper account (hard gate 4)", True, "key prefix and trading URL agree")
    except ExecutionError as exc:
        check("paper account (hard gate 4)", False, str(exc))
        print("\nRefusing to go further. This is the one rule that is not negotiable.")
        return 1

    # -- 3. market data reachable ------------------------------------- #
    universe = tradeable_today()
    check(
        "watchlist has tradeable names",
        bool(universe),
        ", ".join(str(u.symbol) for u in universe) or "none — see agent/watchlist.py",
    )
    data = build_market_data(env, feed=args.feed)
    try:
        spot = data.latest_price(universe[0].symbol)
        check("market data (REST)", True, f"{universe[0].symbol} at {spot}")
    except Exception as exc:
        check("market data (REST)", False, f"{type(exc).__name__}: {exc}")

    # -- 4. HARD GATES 1 & 2: the Trading API, over MCP --------------- #
    try:
        with open_mcp(env, timeout=args.timeout) as mcp:
            tools = mcp.list_tools()
            check(
                "MCP server (hard gate 2)",
                bool(tools),
                f"{len(tools)} tools",
            )
            check(
                "options order tool (hard gate 1 & 3)",
                "place_option_order" in tools,
                "place_option_order",
            )
            account = LiveContext(
                data=data,
                mcp=mcp,
                journal=Journal(directory=Path(args.journal)),
                iv=IvHistoryStore(directory=Path(args.iv)),
                state=SessionState.load(Path(args.state)),
            ).account(as_of=datetime.now(UTC))
            check("account readable", True, f"equity {account.equity}")
            check(
                "options level 3 (spreads)",
                account.can_trade_spreads,
                f"level {account.options_level}"
                + (" — BLOCKED" if account.is_blocked else ""),
            )
    except Exception as exc:
        check("MCP server (hard gate 2)", False, f"{type(exc).__name__}: {exc}")

    # -- 5. HARD GATE 4, second half: is it a *dedicated new* account? - #
    check(
        "dedicated new paper account (hard gate 4)",
        args.confirm_dedicated,
        "pass --confirm-dedicated once you have verified this by hand"
        if not args.confirm_dedicated
        else "confirmed by the operator",
        fatal=True,
    )

    # -- 6. the model key --------------------------------------------- #
    check("model key present", _model_key_present(env), _MODEL_KEY_NOTE, fatal=False)

    print()
    if failures:
        print(f"{failures} gate(s) failed. Fix before the open.")
        return 1
    print("All gates pass. Clear to trade.")
    return 0


def _model_key_present(env: dict[str, str]) -> bool:
    """Ask `agent/` whether it can build a proposer. Never name the key here.

    specs/01 Rule 1: only `alphagate.agent` may reach a model, and a pre-flight
    that reads the key by name is reaching. So it tries to build the thing and
    reports whether it could."""
    from alphagate.agent.deepseek import DeepSeekProposer, MissingModelKey

    try:
        DeepSeekProposer.from_env(env)
    except MissingModelKey:
        return False
    return True


# ------------------------------------------------------------------ #
# once / run
# ------------------------------------------------------------------ #


def _context(args: argparse.Namespace, mcp: Any) -> LiveContext:
    env = load_env_file(Path(args.env))
    return LiveContext(
        data=build_market_data(env, feed=args.feed),
        mcp=mcp,
        journal=Journal(directory=Path(args.journal)),
        iv=IvHistoryStore(directory=Path(args.iv)),
        state=SessionState.load(Path(args.state)),
    )


def _proposer(args: argparse.Namespace) -> Any:
    """The live proposer, or a deterministic stand-in.

    `--no-model` is not a debugging convenience: it is the mode that keeps the
    agent trading when the model endpoint is down, and specs/05 D6 makes that a
    first-class path rather than an outage.
    """
    if args.no_model:
        from alphagate.agent.proposer import DeterministicProposer

        return DeterministicProposer()
    from alphagate.agent.deepseek import DeepSeekProposer

    return DeepSeekProposer.from_env(load_env_file(Path(args.env)))


def cmd_once(args: argparse.Namespace) -> int:
    """One cycle, now. Dry unless `--submit` is passed."""
    submit = bool(args.submit)
    env = load_env_file(Path(args.env))
    session = open_mcp(env, timeout=args.timeout)
    with session as mcp:
        context = _context(args, mcp)
        now = datetime.now(UTC)
        slots = session_slots(*_session_bounds(args, now))
        slot = _slot_now(slots, now)
        gather = gather_for(context, slots=slots)
        inputs = gather(slot)
        slot = replace(slot, sequence=_free_sequence(context, slot, inputs.read.underlying))

        book = context.last_book
        if book is not None and not book.is_clean:
            print(f"! {len(book.unexplained)} unexplained legs at the broker:")
            for leg in book.unexplained:
                print(f"    {leg.contract}  qty {leg.quantity}")

        record = run_cycle(
            read=inputs.read,
            setup=DefaultScreen().screen(inputs.read),
            candidates=inputs.candidates,
            portfolio=inputs.portfolio,
            limits=context.limits,
            as_of=now,
            mcp=mcp if submit else None,
            proposer=_proposer(args),
            sequence=slot.sequence,
        )
        context.journal.append(record)
        _print_cycle(record)
    return 0 if record.stage is not Stage.BREACHED else 1


def cmd_run(args: argparse.Namespace) -> int:
    """A whole session on the schedule. Submits unless `--dry-run`.

    Supervised by default: a dropped connection at 14:00 should not cost the
    afternoon. `--no-supervise` runs exactly one session and returns, which is
    what you want when you are watching it.
    """
    now = datetime.now(UTC)
    open_at, close_at = _session_bounds(args, now)
    slots = tuple(s for s in session_slots(open_at, close_at) if s.at > now)
    if not slots:
        print("No slots left today. The session is over.")
        return 0

    print(
        f"AlphaGate — {len(slots)} slots, "
        f"{open_at:%H:%M}-{close_at:%H:%M} UTC, "
        f"{'DRY RUN' if args.dry_run else 'LIVE (paper)'}, "
        f"{'supervised' if not args.no_supervise else 'single session'}"
    )

    if args.no_supervise:
        result = _one_session(args, slots)
        _report(result)
        return 0
    return supervised_run(args=args, open_at=open_at, close_at=close_at)


MAX_RESTARTS = 8
"""How many times a day may be resumed after a session dies.

Not unbounded: a process that respawns forever against a broken credential
burns a trading day writing the same traceback. Not one either — a dropped
websocket at 14:00 should not cost the afternoon, and eight restarts across a
six-and-a-half hour session is roughly one every fifty minutes, which is far
more headroom than a healthy day needs and a clear signal when it is used up.
"""

RESTART_PAUSE = 20.0
"""Seconds before resuming. Long enough that a rate limit or a server restart
has passed, short enough to be inside one fifteen-minute slot."""


def supervised_run(
    *,
    args: argparse.Namespace,
    open_at: datetime,
    close_at: datetime,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """Run a session, and resume it if it dies. Returns a process exit code.

    The agent is a foreground process on a laptop for a week. `run` on its own
    is one session: if the MCP subprocess dies, the network drops, or an
    unhandled error escapes, the day ends there and the remaining slots are
    never journalled. That is a bad way to lose an afternoon when four trading
    days is the whole sample.

    Resumption is *slot-based*, which is what makes it safe. Every restart
    recomputes the schedule and keeps only the slots still ahead, so nothing is
    replayed: the journal never grows a second decision for a slot that already
    has one, and `cycle_id` stays what it was (specs/06 D2). A restart costs the
    slot it happened in and nothing else.

    **Two conditions stop it for good rather than resuming.** A partial-fill
    breach means a naked leg and a latched kill switch (specs/04 D5) — resuming
    would put the agent back to work on a book a human has not looked at. A
    kill switch already latched on the incoming snapshot means the same thing
    one day later.
    """
    restarts = 0
    total = SessionResult()

    while True:
        pending = tuple(s for s in session_slots(open_at, close_at) if s.at > now())
        if not pending:
            break

        try:
            result = _one_session(args, pending)
        except Exception as exc:
            restarts += 1
            print(f"\n! session died: {type(exc).__name__}: {exc}")
            if restarts > MAX_RESTARTS:
                print(f"! {MAX_RESTARTS} restarts used up — stopping. Look at the journal.")
                return 1
            print(f"! resuming in {RESTART_PAUSE:.0f}s ({restarts}/{MAX_RESTARTS})")
            sleep(RESTART_PAUSE)
            continue

        total.records.extend(result.records)
        total.reconciled += result.reconciled
        total.unreadable.extend(result.unreadable)

        if result.stopped_early and _is_terminal_stop(result.stopped_early):
            total.stopped_early = result.stopped_early
            print(f"\n! {result.stopped_early}")
            print("! not resuming: this needs a human before any more orders go out.")
            _report(total)
            return 1

        if not result.stopped_early:
            break

        restarts += 1
        print(f"\n! session stopped early: {result.stopped_early}")
        if restarts > MAX_RESTARTS:
            print(f"! {MAX_RESTARTS} restarts used up — stopping.")
            break
        print(f"! resuming in {RESTART_PAUSE:.0f}s ({restarts}/{MAX_RESTARTS})")
        sleep(RESTART_PAUSE)

    _report(total)
    return 0


_TERMINAL_STOPS = ("partial fill breach", "kill switch")


def _is_terminal_stop(reason: str) -> bool:
    """Whether this is a stop a restart must not paper over."""
    return any(marker in reason for marker in _TERMINAL_STOPS)


def _one_session(args: argparse.Namespace, slots: Sequence[Slot]) -> SessionResult:
    """One MCP session over the given slots. Opened and closed here.

    The transport is rebuilt on every restart rather than reused, because the
    most likely reason a session died is that the subprocess behind it is gone.
    """
    env = load_env_file(Path(args.env))
    with open_mcp(env, timeout=args.timeout) as mcp:
        context = _context(args, mcp)
        return run_session(
            slots,
            gather_for(context, submit_exits=not args.dry_run, slots=slots),
            limits=context.limits,
            journal=context.journal,
            proposer=_proposer(args),
            mcp=None if args.dry_run else mcp,
            sleep=time.sleep,
            now=lambda: datetime.now(UTC),
        )


def _report(result: SessionResult) -> None:
    print("\n" + result.summary())
    for cycle_id, why in result.unreadable:
        print(f"  ! UNREADABLE {cycle_id}: {why}")
    if result.records:
        print("\nlast cycles:")
        for record in result.records[-5:]:
            _print_cycle(record, prefix="  ")


def _slot_now(slots: Sequence[Slot], now: datetime) -> Slot:
    """The slot the schedule would be running. Not simply the first one.

    `once` used to take `slots[0]`, which meant the underlying it picked and the
    id it wrote were the same every time — so running it twice in a morning
    produced two decisions sharing a `cycle_id`, and `Journal.read` collapsed
    them (specs/06 D2). Choosing by the clock makes an ad-hoc run land where a
    scheduled one would.
    """
    for slot in slots:
        if slot.at >= now:
            return slot
    return slots[-1]


def _free_sequence(context: LiveContext, slot: Slot, underlying: object) -> int:
    """A sequence this day has not used for this name.

    The schedule guarantees uniqueness within a session; `once` is outside a
    session and guarantees nothing, so it asks the journal. Without this, two
    manual runs write one line and the earlier decision is on disk and not in
    the day — the exact failure `Journal.duplicate_cycles` exists to report.
    """
    from alphagate.agent import cycle_id_for

    existing = {
        str(record.get("cycle_id", "")) for record in context.journal.read(slot.at.date())
    }
    sequence = slot.sequence
    while cycle_id_for(slot.at, str(underlying), sequence) in existing:
        sequence += 1
    return sequence


def _session_bounds(args: argparse.Namespace, now: datetime) -> tuple[datetime, datetime]:
    """Regular hours for today, from the exchange clock where available."""
    from alphagate.live.wiring import market_session

    if args.open and args.close:
        day = now.date()
        return (_at(day, args.open), _at(day, args.close))
    return market_session(now)


def _at(day: date, hhmm: str) -> datetime:
    """`HH:MM` on `day`, in UTC. Everything in this system is tz-aware."""
    hour, minute = (int(part) for part in hhmm.split(":", 1))
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=UTC)


def _print_cycle(record: Any, *, prefix: str = "") -> None:
    stage = record.stage.value.upper()
    print(f"{prefix}{record.cycle_id}  {stage:<14} {record.note}")
    if record.verdict is not None and record.veto_reasons:
        print(f"{prefix}    vetoed by: {', '.join(record.veto_reasons)}")


# ------------------------------------------------------------------ #
# serve / show
# ------------------------------------------------------------------ #


def cmd_status(args: argparse.Namespace) -> int:
    """Print what the agent is doing, from the snapshot it left behind.

    Reads the same file the dashboard reads, so the terminal and the browser
    can never disagree — and it needs no broker session, so it answers
    instantly and works when the agent is not running, which is exactly when
    the question gets asked.
    """
    directory = Path(args.journal)
    snapshot = read_status(directory)
    if snapshot is None:
        print(f"no status yet in {directory} — the agent has not run")
        return 1

    age = _age_of(snapshot.get("as_of"))
    running = age is not None and age < STALE_AFTER
    mark = OK if running else FAIL
    stamp = str(snapshot.get("as_of", "?"))
    print(f"[{mark}] {'running' if running else 'NOT RUNNING'} — last cycle {stamp}")
    if age is not None and not running:
        print(f"        last heartbeat {age / 60:.0f} minutes ago")
    if snapshot.get("next_slot"):
        print(f"        next slot {snapshot['next_slot']}")
    print(f"        watching {', '.join(snapshot.get('universe', []))}")

    print()
    print(f"  equity           {snapshot.get('equity')}   (today {snapshot.get('session_change')})")
    print(f"  cash             {snapshot.get('cash')}")
    print(f"  options level    {snapshot.get('options_level')}"
          f"{'' if snapshot.get('can_trade_spreads') else '  ** CANNOT TRADE SPREADS **'}")
    print(f"  drawdown         {snapshot.get('drawdown_pct')} of "
          f"{snapshot.get('max_drawdown_pct')}")
    print(f"  open risk        {snapshot.get('open_risk')} of {snapshot.get('max_portfolio_risk')}")
    print(f"  structures       {snapshot.get('open_structures')} of "
          f"{snapshot.get('max_open_structures')}")
    print(f"  fills today      {snapshot.get('fills_today')} of {snapshot.get('max_daily_trades')}")
    if snapshot.get("killswitch_tripped"):
        print("  ** KILL SWITCH LATCHED — opens blocked until a human clears it **")

    positions = snapshot.get("positions") or []
    print()
    if not positions:
        print("  no open positions")
    else:
        print(f"  {len(positions)} open:")
        for position in positions:
            print(f"    {position.get('underlying')} {position.get('structure')} "
                  f"x{position.get('quantity')}  entry {position.get('entry_premium')} "
                  f"mark {position.get('mark')}  dte {position.get('days_to_expiry')}")
            print(f"      {position.get('rule')}: {position.get('detail')}")

    unexplained = snapshot.get("unexplained") or []
    if unexplained:
        print()
        print(f"  ! {len(unexplained)} legs the journal cannot explain "
              "(not in the risk model):")
        for leg in unexplained:
            print(f"      {leg}")

    stages = snapshot.get("stage_counts") or {}
    if stages:
        print()
        print("  today: " + ", ".join(f"{k}={v}" for k, v in sorted(stages.items())))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from alphagate.interface.app import serve

    serve(journal_dir=Path(args.journal), host=args.host, port=args.port)
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    """Print a day from the journal. No server, no browser."""
    journal = Journal(directory=Path(args.journal))
    day = date.fromisoformat(args.day) if args.day else datetime.now(UTC).date()
    records = journal.read(day)
    if not records:
        print(f"nothing journalled for {day}")
        return 0

    print(f"{day} — {len(records)} cycles\n")
    for record in records:
        stage = str(record.get("stage", "?")).upper()
        print(f"{record['cycle_id']}  {stage:<14} {record.get('note', '')}")
        if args.verbose:
            print(f"    {trust_report(record)}")
            outcome = record.get("outcome")
            if isinstance(outcome, dict):
                print(f"    outcome: {outcome.get('status')} "
                      f"realised {outcome.get('realised_pl', '—')}")

    duplicates = journal.duplicate_cycles(day)
    if duplicates:
        print(f"\n! duplicate cycle ids (decisions collapsed on read): {duplicates}")
    orphans = journal.orphaned_amendments(day)
    if orphans:
        print(f"! amendments with no record: {orphans}")

    stages: dict[str, int] = {}
    for record in records:
        key = str(record.get("stage", "?"))
        stages[key] = stages.get(key, 0) + 1
    print("\n" + ", ".join(f"{k}={v}" for k, v in sorted(stages.items())))
    return 0


# ------------------------------------------------------------------ #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="alphagate",
        description="An options trading agent that can be overruled.",
    )
    parser.add_argument("--env", default=DEFAULT_ENV, help="credentials file")
    parser.add_argument("--journal", default=DEFAULT_JOURNAL, help="journal directory")
    parser.add_argument("--state", default=DEFAULT_STATE, help="session state file")
    parser.add_argument("--iv", default=DEFAULT_IV, help="IV history directory")
    parser.add_argument("--feed", default=None, help="market data feed (iex/sip)")
    parser.add_argument("--timeout", type=float, default=30.0, help="MCP call timeout")
    sub = parser.add_subparsers(dest="command", required=True)

    pre = sub.add_parser("preflight", help="check the hard gates before a trading day")
    pre.add_argument(
        "--confirm-dedicated",
        action="store_true",
        help="you have verified this is a new, dedicated paper account (hard gate 4)",
    )
    pre.set_defaults(func=cmd_preflight)

    once = sub.add_parser("once", help="run one cycle now (dry by default)")
    once.add_argument("--submit", action="store_true", help="actually place the order")
    once.add_argument("--no-model", action="store_true", help="deterministic proposer")
    once.add_argument("--open", default=None, help="session open, HH:MM UTC")
    once.add_argument("--close", default=None, help="session close, HH:MM UTC")
    once.set_defaults(func=cmd_once)

    run = sub.add_parser("run", help="run a whole session on the schedule")
    run.add_argument("--dry-run", action="store_true", help="gate everything, submit nothing")
    run.add_argument(
        "--no-supervise",
        action="store_true",
        help="run one session and stop, rather than resuming after a failure",
    )
    run.add_argument("--no-model", action="store_true", help="deterministic proposer")
    run.add_argument("--open", default=None, help="session open, HH:MM UTC")
    run.add_argument("--close", default=None, help="session close, HH:MM UTC")
    run.set_defaults(func=cmd_run)

    status = sub.add_parser("status", help="what the agent is doing right now")
    status.set_defaults(func=cmd_status)

    show = sub.add_parser("show", help="print a journalled day")
    show.add_argument("--day", default=None, help="YYYY-MM-DD, default today")
    show.add_argument("-v", "--verbose", action="store_true")
    show.set_defaults(func=cmd_show)

    serve = sub.add_parser("serve", help="the dashboard — live status and the journal")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(func=cmd_serve)

    add_equity_commands(
        sub,
        default_books=DEFAULT_TARGET_BOOKS,
        default_state=DEFAULT_EQUITY_STATE,
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("\ninterrupted — the journal is intact up to the last completed cycle")
        return 130
    except ExecutionError as exc:
        print(f"execution error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
