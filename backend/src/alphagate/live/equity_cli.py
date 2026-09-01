"""The equity commands — specs/09 D8.

```
alphagate equity-preflight   # is there a book, and may it be executed?
alphagate equity-plan        # one pass, gated, nothing sent
alphagate equity-rebalance   # one pass, orders placed
alphagate equity-run         # a session: heartbeat, one pass, reconcile
alphagate equity-status      # what it holds and how far it has drifted
```

Kept in its own module rather than folded into `cli.py`, because the two agents
share a process entry point and nothing else. The options runner is a
fifteen-minute loop over an option chain; this is a daily rebalance against a
file another project wrote. Threading both through one set of helpers would have
produced helpers that serve neither.

**`equity-plan` is dry and `equity-rebalance` is not**, the same asymmetry
`once` and `run` keep. The command you type while debugging must not place
orders by accident; the command whose name is the act is not a surprise.

**The strategy is pinned by configuration, not chosen at the command line.**
`--fingerprint` exists for a second account or a test, but the default comes
from `ALPHAGATE_STRATEGY_FINGERPRINT` in the environment file, and a book that
disagrees is refused by name (specs/09 D1). Making the pin a required flag would
have made "which strategy is running" a property of somebody's shell history.
"""

from __future__ import annotations

import argparse
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from alphagate.equity import DEFAULT_EQUITY_POLICY, EQUITY_SLEEVE_ALLOCATION, EquitySide
from alphagate.execution import (
    load_env_file,
    read_account,
    read_clock,
    read_share_positions,
)
from alphagate.journal import Journal
from alphagate.live.equity import (
    EquityContext,
    EquityCycleRecord,
    EquityStage,
    already_decided,
    changed_pin_advice,
    digest_of,
    find_latest_book,
    marks_from,
    read_book,
    run_equity_cycle,
    strategy_view,
    today_totals,
    unpinned_books,
)
from alphagate.live.equity_status import (
    build_equity_status,
    read_equity_status,
    write_equity_status,
)
from alphagate.live.wiring import (
    EQUITY_SLEEVE_BASIS,
    SessionState,
    build_market_data,
    market_session,
)
from alphagate.live.wiring import mcp_session as open_mcp

__all__ = ["add_equity_commands"]

HEARTBEAT_SECONDS = 30.0
"""How often the status file is rewritten while nothing is being decided.

Thirty seconds is chosen against the dashboard's own poll rather than against
the market: the page refreshes every fifteen, so a slower heartbeat would show a
visibly ageing snapshot on a healthy system and teach its reader to ignore the
staleness indicator."""

PASS_OFFSET_MINUTES = 15
"""Minutes after the open before the rebalance runs — specs/09 D8.

Late enough that the opening auction has settled and a snapshot mid is a price
rather than an artefact; early enough that the book being placed is still the
book the strategy decided on, rather than a lunchtime approximation of it."""


# ------------------------------------------------------------------ #
# Wiring
# ------------------------------------------------------------------ #


def _fingerprint(args: argparse.Namespace, env: dict[str, str]) -> str:
    pinned = args.fingerprint or env.get("ALPHAGATE_STRATEGY_FINGERPRINT", "")
    if not pinned:
        raise SystemExit(
            "no strategy pinned. Set ALPHAGATE_STRATEGY_FINGERPRINT in the env "
            "file, or pass --fingerprint. specs/09 D1: the pin is what makes "
            "'only the strategy the researcher validated' a checkable statement, "
            "so there is deliberately no default."
        )
    return pinned


def _books(args: argparse.Namespace, env: dict[str, str]) -> Path:
    return Path(args.books or env.get("ALPHAGATE_TARGET_BOOKS") or args.default_books)


def _context(args: argparse.Namespace, mcp: Any) -> tuple[EquityContext, SessionState]:
    env = load_env_file(Path(args.env))
    state = SessionState.load(Path(args.equity_state), basis=EQUITY_SLEEVE_BASIS)
    if state.discarded_peak is not None:
        # Loud on purpose. specs/03 D6 changed what the high-water mark is a
        # mark *on*, and a kill switch that silently forgot its history is the
        # one thing worse than one that resets.
        print(
            f"note: discarded a high-water mark of {state.discarded_peak} — it was "
            f"measured against account equity, and this sleeve marks "
            f"{EQUITY_SLEEVE_ALLOCATION}. Tracking restarts from today."
        )
    context = EquityContext(
        data=build_market_data(env, feed=args.feed),
        mcp=mcp,
        journal=Journal(directory=Path(args.journal)),
        books=_books(args, env),
        pinned_fingerprint=_fingerprint(args, env),
        policy=DEFAULT_EQUITY_POLICY,
        allocation=EQUITY_SLEEVE_ALLOCATION,
        peak_equity=state.peak_equity,
        killswitch_tripped=state.killswitch_tripped,
    )
    return context, state


def _persist(context: EquityContext, state: SessionState) -> None:
    """Carry the high-water mark back to disk.

    A mark that resets at midnight is a kill switch that cannot latch across the
    days it exists for — specs/03 D4, and the equity path inherits it unchanged.
    """
    if context.peak_equity is not None:
        state.observe(context.peak_equity)
    if context.killswitch_tripped != state.killswitch_tripped:
        state.killswitch_tripped = context.killswitch_tripped
        state.save()


# ------------------------------------------------------------------ #
# equity-preflight
# ------------------------------------------------------------------ #


def cmd_equity_preflight(args: argparse.Namespace) -> int:
    """Is there a book, may it be executed, and does the account agree?

    Every refusal in specs/09 D1 is checked here rather than discovered at the
    open. A book whose seal is unspent fails the same way it would fail during a
    rebalance, and finding that out at 09:20 costs nothing.
    """
    env = load_env_file(Path(args.env))
    pinned = _fingerprint(args, env)
    books = _books(args, env)
    failures = 0

    def check(label: str, passed: bool, detail: str = "") -> None:
        nonlocal failures
        mark = "  ok  " if passed else " FAIL "
        print(f"[{mark}] {label}" + (f"  — {detail}" if detail else ""))
        if not passed:
            failures += 1

    print(f"AlphaGate equity pre-flight — {datetime.now(UTC).isoformat(timespec='seconds')}\n")
    check("strategy pinned", True, pinned)

    path = find_latest_book(books, pinned)
    check("a target book exists", path is not None, str(path) if path else str(books))
    if path is None:
        # A stale pin and an empty directory are the same sentence from here
        # — the glob is on the fingerprint — and completely different
        # problems. Say which one it is.
        advice = changed_pin_advice(unpinned_books(books, pinned))
        fallback = "Run `aqr target-book <fingerprint>` in ai_quant_researcher first."
        print("\n" + (advice or fallback))
        return 1

    try:
        book, _ = read_book(path, pinned_fingerprint=pinned)
    except Exception as failure:  # reported, not swallowed
        check("the book may be executed", False, str(failure).replace("\n", " "))
        return 1

    check("the book may be executed", True, book.label())
    age = book.age_days(datetime.now(UTC).date())
    check(
        "the book is fresh",
        age <= DEFAULT_EQUITY_POLICY.max_book_age_days,
        f"{age}d old, limit {DEFAULT_EQUITY_POLICY.max_book_age_days}d",
    )
    sealed = book.sealed
    check(
        "the sealed window did not refute it",
        not sealed.refuted,
        f"alpha {sealed.alpha:+.2%}/yr  beta {sealed.beta:.2f}  "
        f"t {sealed.t_alpha:+.2f}  looks {sealed.looks}",
    )

    session = open_mcp(env, timeout=args.timeout)
    with session as mcp:
        now = datetime.now(UTC)
        account = read_account(mcp, observed_at=now)
        check("account readable", True, f"equity {account.equity}")
        check("account not blocked", not account.is_blocked)
        held = read_share_positions(mcp)
        print(f"[  ok  ] {len(held)} equity positions held, {len(book.weights)} wanted")

        # Reported, never failed. Pre-flight is a thing you run the evening
        # before as well as at 09:20, and a closed market is not a fault — it is
        # the reason every mark would be stale if you traded now.
        clock = read_clock(mcp, observed_at=now)
        print(
            f"[  ok  ] market {'open' if clock.is_open else 'closed'}"
            + ("" if clock.is_open else f"  — next open {clock.next_open:%Y-%m-%d %H:%M} UTC")
        )

    print()
    if failures:
        print(f"{failures} gate(s) failed. Fix before the open.")
        return 1
    print("Ready.")
    return 0


# ------------------------------------------------------------------ #
# equity-plan / equity-rebalance
# ------------------------------------------------------------------ #


def cmd_equity_plan(args: argparse.Namespace) -> int:
    """One pass, gated, nothing sent."""
    return _one_pass(args, submit=False)


def cmd_equity_rebalance(args: argparse.Namespace) -> int:
    """One pass, orders placed."""
    return _one_pass(args, submit=True)


def _one_pass(args: argparse.Namespace, *, submit: bool) -> int:
    env = load_env_file(Path(args.env))
    session = open_mcp(env, timeout=args.timeout)
    with session as mcp:
        context, state = _context(args, mcp)
        now = datetime.now(UTC)
        clock = read_clock(mcp, observed_at=now)
        if not clock.is_open:
            # Said before the plan rather than after, because the plan will
            # otherwise report a hundred stale marks and leave the reader to
            # work out why. The pass still runs: the Gate refusing a closed
            # market is the behaviour under test, and hiding it behind an early
            # return would mean nothing exercised it.
            print(
                f"! the market is closed; next open {clock.next_open:%Y-%m-%d %H:%M} UTC. "
                "Every mark will be stale and nothing will trade."
            )
        record = run_equity_cycle(
            context, as_of=now, submit=submit, sequence=_next_sequence(context, now)
        )
        context.journal.append(record)
        _persist(context, state)
        _publish(context, record, as_of=now, next_pass=None, sequence=0)
        _print_cycle(record)
    return 0 if record.stage is not EquityStage.HALTED else 1


def _next_sequence(context: EquityContext, now: datetime) -> int:
    """The first sequence not already used today.

    A restart mid-session must not mint a `cycle_id` that already exists: the
    journal keys on it, so a duplicate collapses two decisions into one line
    (specs/06 D2).
    """
    used = {
        str(record.get("cycle_id", ""))
        for record in context.journal.read(now.date())
    }
    for sequence in range(1000):
        if f"{now.date().isoformat()}-EQ-{sequence:03d}" not in used:
            return sequence
    return 999  # pragma: no cover - a thousand passes in one day is not a thing


# ------------------------------------------------------------------ #
# equity-run
# ------------------------------------------------------------------ #


def cmd_equity_run(args: argparse.Namespace) -> int:
    """A session: heartbeat continuously, rebalance once — specs/09 D8.

    The heartbeat is most of what this does, and it is not decoration. On this
    strategy the plan is empty four days in five, so a process that only woke to
    trade would be indistinguishable from a process that had died. Re-reading the
    account and re-marking the book every thirty seconds is what makes the
    dashboard say *running* honestly.

    The pass is guarded by the journal rather than by a flag, and keyed on the
    *book* rather than on the day: a restart at 14:00 finds today's record for
    the book on disk and does not replay it, while a book regenerated at 14:00
    is a new instruction and gets its own pass. specs/09 D8.
    """
    now = datetime.now(UTC)
    open_at, close_at = _bounds(args, now)
    pass_at = open_at + timedelta(minutes=args.offset)
    submit = not args.dry_run

    print(
        f"AlphaGate equity — session {open_at:%H:%M}-{close_at:%H:%M} UTC, "
        f"rebalance at {pass_at:%H:%M}, "
        f"{'DRY RUN' if args.dry_run else 'LIVE (paper)'}"
    )

    env = load_env_file(Path(args.env))
    session = open_mcp(env, timeout=args.timeout)
    sequence = 0
    with session as mcp:
        clock = read_clock(mcp, observed_at=now)
        if not clock.opens_today:
            print(
                f"The market does not open today. Next open "
                f"{clock.next_open:%Y-%m-%d %H:%M} UTC."
            )
            return 0
        context, state = _context(args, mcp)
        while True:
            now = datetime.now(UTC)
            if now >= close_at:
                print("Session over.")
                break

            pending = _book_awaiting_a_pass(context, now)
            if pending and now >= pass_at:
                record = run_equity_cycle(
                    context,
                    as_of=now,
                    submit=submit,
                    sequence=_next_sequence(context, now),
                )
                context.journal.append(record)
                _print_cycle(record)
                pending = not record.stage.decided
                if record.stage is EquityStage.HALTED:
                    print("Halted. Reconcile by hand before resuming.")
                    _persist(context, state)
                    return 1

            sequence += 1
            _heartbeat(
                context, as_of=now, next_pass=pass_at if pending else None, sequence=sequence
            )
            _persist(context, state)
            time.sleep(min(HEARTBEAT_SECONDS, max(1.0, (close_at - now).total_seconds())))
    return 0


def _book_awaiting_a_pass(context: EquityContext, now: datetime) -> bool:
    """Whether the newest book on disk still needs a pass today.

    Asked every beat rather than latched at the open, because the answer can
    change during a session: `aqr target-book` writing a better book is a new
    instruction, and the session should pick it up rather than sleep through it
    (specs/09 D8).

    Three cases, all of them "keep looking":

    * no book on disk — the pass journals why, and `aqr` may still be running;
    * the file cannot be read this instant — it is being rewritten;
    * the newest book has not reached a deciding stage today.

    The archive is what makes the middle case safe to retry: a pass that got
    part-way records what it saw against the digest it saw it under.
    """
    path = find_latest_book(context.books, context.pinned_fingerprint)
    if path is None:
        return True
    digest = digest_of(path)
    if digest is None:
        return True
    return not already_decided(context.journal, now.date(), digest)


# ------------------------------------------------------------------ #
# The heartbeat
# ------------------------------------------------------------------ #


def _heartbeat(
    context: EquityContext,
    *,
    as_of: datetime,
    next_pass: datetime | None,
    sequence: int,
) -> None:
    """Re-read the account, re-mark the book, rewrite the status file.

    Never raises. Status is a convenience; trading is not. A disk that is full
    or a snapshot request that times out must cost the operator a stale page,
    not a session.
    """
    try:
        if context.mcp is None:
            return
        account = read_account(context.mcp, observed_at=as_of)
        context.last_account = account
        context.observe(account.equity)
        holdings = read_share_positions(context.mcp)
        context.last_holdings = holdings

        book = context.last_book
        if book is None:
            path = find_latest_book(context.books, context.pinned_fingerprint)
            if path is not None:
                book, _ = read_book(path, pinned_fingerprint=context.pinned_fingerprint)
                context.last_book = book

        wanted = set(book.weights) if book else set()
        symbols = sorted(wanted | {h.symbol for h in holdings}, key=str)
        snapshots = context.data.stock_snapshots(symbols)
        marks = marks_from(snapshots, context.tradeability_for(symbols), as_of=as_of)
        context.last_marks = marks

        orders, turnover = today_totals(context.journal, as_of.date())
        snapshot = build_equity_status(
            account=account,
            book=book,
            strategy=strategy_view(book) if book else {},
            holdings=holdings,
            marks=marks,
            policy=context.policy,
            peak_equity=context.peak_equity,
            killswitch_tripped=context.killswitch_tripped,
            orders_today=orders,
            turnover_today=turnover,
            as_of=as_of,
            next_pass=next_pass,
            heartbeat_sequence=sequence,
            stage_counts=_stages_today(context, as_of),
            note=context.last_note,
        )
        write_equity_status(snapshot, directory=context.journal.directory)
    except Exception:  # see the docstring
        return


def _publish(
    context: EquityContext,
    record: EquityCycleRecord,
    *,
    as_of: datetime,
    next_pass: datetime | None,
    sequence: int,
) -> None:
    _heartbeat(context, as_of=as_of, next_pass=next_pass, sequence=sequence)


def _stages_today(context: EquityContext, as_of: datetime) -> dict[str, int]:
    counts: dict[str, int] = {}
    try:
        records = context.journal.read(as_of.date())
    except OSError:
        return counts
    for record in records:
        if record.get("kind") != "equity":
            continue
        stage = str(record.get("stage", ""))
        if stage:
            counts[stage] = counts.get(stage, 0) + 1
    return counts


# ------------------------------------------------------------------ #
# equity-status
# ------------------------------------------------------------------ #


def cmd_equity_status(args: argparse.Namespace) -> int:
    """What it holds and how far it has drifted. Reads the file, not the broker.

    The same file the dashboard reads, so the terminal and the page cannot
    disagree — and so this command inherits the property that a stopped agent
    reports as stopped rather than as flat.
    """
    snapshot = read_equity_status(Path(args.journal))
    if snapshot is None:
        print("No equity status on disk. The agent has not run.")
        return 1

    age = _age_of(snapshot.get("as_of"))
    running = age is not None and age < 120
    strategy = snapshot.get("strategy") or {}
    sealed = strategy.get("sealed") or {}

    print(f"{'RUNNING' if running else 'STALE'}  —  {snapshot.get('as_of')}"
          + (f"  ({age:.0f}s ago)" if age is not None else ""))
    if strategy:
        print(f"\n{strategy.get('name')} [{strategy.get('fingerprint')}] "
              f"as of {strategy.get('as_of')}")
        print(f"  sealed  alpha {sealed.get('alpha', 0):+.2%}/yr  "
              f"beta {sealed.get('beta', 0):.2f}  t {sealed.get('t_alpha', 0):+.2f}  "
              f"IR {sealed.get('information_ratio', 0):+.2f}  "
              f"looks {sealed.get('looks')}")
        print("  this window can refute and cannot confirm")

    print(f"\nequity {snapshot.get('equity')}  cash {snapshot.get('cash')}  "
          f"today {snapshot.get('session_change')}")
    print(f"gross {float(snapshot.get('gross_exposure', 0)):.4f}x  "
          f"held {snapshot.get('positions_held')} of {snapshot.get('positions_wanted')}  "
          f"drawdown {float(snapshot.get('drawdown_pct', 0)):.2%} "
          f"of {snapshot.get('max_drawdown_pct')}")
    print(f"orders today {snapshot.get('orders_today')}/{snapshot.get('max_daily_orders')}  "
          f"turnover {snapshot.get('turnover_today')}  "
          f"band {float(snapshot.get('drift_band_pct', 0)):.0%} of position, "
          f"floor {snapshot.get('min_order_notional')}")

    lines = snapshot.get("lines") or []
    outside = [line for line in lines if not line.get("inside_band")]
    if outside:
        print(f"\n{len(outside)} symbols outside the band:")
        for line in sorted(outside, key=lambda x: -abs(float(x.get("drift", 0))))[:20]:
            print(f"  {line['symbol']:<6} target {float(line['target_weight']):.4f}  "
                  f"held {float(line['held_weight']):.4f}  "
                  f"drift {float(line['drift']):+12.2f}  "
                  f"band {float(line['threshold']):.2f}")
    if snapshot.get("unpriced"):
        print(f"\n! unpriced: {', '.join(snapshot['unpriced'][:20])}")
    if snapshot.get("off_book"):
        print(f"! held but not wanted: {', '.join(snapshot['off_book'][:20])}")
    return 0


def _age_of(stamp: Any) -> float | None:
    if not isinstance(stamp, str):
        return None
    try:
        return (datetime.now(UTC) - datetime.fromisoformat(stamp)).total_seconds()
    except ValueError:
        return None


# ------------------------------------------------------------------ #
# Printing
# ------------------------------------------------------------------ #


def _print_cycle(record: EquityCycleRecord) -> None:
    strategy = record.strategy or {}
    print(f"\n{record.cycle_id}  {record.stage.value.upper()}  {record.note}")
    if strategy:
        print(f"  {strategy.get('name')} [{strategy.get('fingerprint')}] "
              f"as of {strategy.get('as_of')}")
    if not record.orders:
        return
    print(f"  equity {record.equity}  band {record.band_pct:.0%}  turnover {record.turnover}")
    for order in record.orders:
        arrow = "+" if order.side == EquitySide.BUY.value else "-"
        print(
            f"    {arrow} {order.symbol:<6} {order.shares:>12} @ ~{order.reference_price:<10} "
            f"= {order.notional:>12}   {order.outcome}"
        )
        if not order.approved:
            for reason in order.verdict.reasons:  # type: ignore[union-attr]
                print(f"        vetoed: {reason.check} — {reason.detail}")


# ------------------------------------------------------------------ #
# Registration
# ------------------------------------------------------------------ #


def add_equity_commands(
    sub: Any, *, default_books: str, default_state: str
) -> None:
    """Attach the equity subcommands to the top-level parser.

    `default_books` points at wherever `aqr target-book` writes. It is a *path*,
    not an import: specs/09 D0 makes the artefact the whole interface between
    the two projects, and this is the one line in AlphaGate that knows the other
    project has a directory.
    """

    def common(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument(
            "--fingerprint",
            default=None,
            help="strategy to execute; defaults to ALPHAGATE_STRATEGY_FINGERPRINT",
        )
        parser.add_argument(
            "--books", default=None, help="directory of target books from aqr"
        )
        parser.add_argument(
            "--equity-state", default=default_state, help="equity session state file"
        )
        parser.set_defaults(default_books=default_books)
        return parser

    common(
        sub.add_parser("equity-preflight", help="check the book and the account")
    ).set_defaults(func=cmd_equity_preflight)

    common(
        sub.add_parser("equity-plan", help="one rebalance pass, gated, nothing sent")
    ).set_defaults(func=cmd_equity_plan)

    common(
        sub.add_parser("equity-rebalance", help="one rebalance pass, orders placed")
    ).set_defaults(func=cmd_equity_rebalance)

    run = common(sub.add_parser("equity-run", help="a session: heartbeat and one pass"))
    run.add_argument("--dry-run", action="store_true", help="gate everything, submit nothing")
    run.add_argument("--open", default=None, help="session open, HH:MM UTC")
    run.add_argument("--close", default=None, help="session close, HH:MM UTC")
    run.add_argument(
        "--offset",
        type=int,
        default=PASS_OFFSET_MINUTES,
        help="minutes after the open to rebalance",
    )
    run.set_defaults(func=cmd_equity_run)

    status = sub.add_parser("equity-status", help="what the equity book looks like now")
    status.set_defaults(func=cmd_equity_status)


def _bounds(args: argparse.Namespace, now: datetime) -> tuple[datetime, datetime]:
    open_at, close_at = market_session(now)
    if args.open:
        open_at = _at(now, args.open)
    if args.close:
        close_at = _at(now, args.close)
    return open_at, close_at


def _at(now: datetime, hhmm: str) -> datetime:
    hour, minute = (int(part) for part in hhmm.split(":", 1))
    return datetime(now.year, now.month, now.day, hour, minute, tzinfo=UTC)

