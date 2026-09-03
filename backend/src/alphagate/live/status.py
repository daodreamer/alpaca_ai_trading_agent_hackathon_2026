"""What the agent is doing right now, written to a file the dashboard can read.

The journal answers "what did it decide, and why" — that is specs/06 and it is
the artefact the submission ships. It cannot answer "is it running, what does it
hold, how close is it to a limit", because a journal is written *after* a
decision and a decision only happens every fifteen minutes.

So the agent drops a snapshot each slot, and the dashboard reads the file.

**Via a file, deliberately, and not by giving the dashboard a broker session.**
`tests/test_boundaries.py` forbids `alphagate.interface` from importing
`execution`, `marketdata` or `live` — which is what makes "there is no code path
from a browser to an order" a checkable statement rather than a hope. Handing
the dashboard a live `McpSession` to render a positions table would delete that
guarantee for a feature that a 2 KB JSON file provides just as well.

It also fails honestly. If the agent stops, the file stops being rewritten, and
`age_seconds` grows — so a stale page is visibly stale instead of quietly
showing yesterday's book as though it were now. A dashboard that cannot tell you
it has lost contact is worse than no dashboard.

The fields follow Alpaca's own account and portfolio summaries: equity, cash,
buying power, options level, unrealised and day P&L, position count and
concentration. Two of those come with a warning worth repeating here, because
both are easy to get subtly wrong:

* `options_level` is the *effective* level, not the approved one — an account
  approved for 3 and capped at 2 reads as tradeable and rejects every order;
* `multiplier == 4` is the only pattern-day-trader signal the account payload
  carries. There is no `pattern_day_trader` field, and nothing may read one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from alphagate.agent.book import BookRead, HeldPosition
from alphagate.agent.exits import ExitPolicy, ExitRule, evaluate_exit
from alphagate.execution import AccountRead
from alphagate.journal import encode, redact
from alphagate.options import StructureRisk
from alphagate.risk import RiskLimits

__all__ = ["PositionStatus", "StatusSnapshot", "read_status", "write_status"]

STATUS_FILENAME = "status.json"


@dataclass(frozen=True, slots=True)
class PositionStatus:
    """One open position, with the numbers that say how close it is to closing.

    `to_target` and `to_stop` are the point of this type. "Up 34%" is a fact
    about the past; "16 points from the profit target and 166 from the stop" is
    what tells you what the agent is about to do, which is what someone watching
    it actually wants to know.
    """

    cycle_id: str
    underlying: str
    structure: str
    quantity: int
    expiry: str
    days_to_expiry: int
    entry_premium: Decimal
    mark: Decimal
    unrealised: Decimal
    fraction_of_credit: Decimal | None
    rule: str
    """What the exit policy says right now — usually `hold`."""
    detail: str
    to_target: Decimal | None
    """Percentage points of premium still to earn before the profit rule fires.
    `None` for a debit structure, where "fraction of the credit" names nothing."""
    to_stop: Decimal | None
    max_loss: Decimal

    @property
    def should_close(self) -> bool:
        """Whether the policy has actually decided to close this position.

        Asked as "is this one of the closing rules" rather than "is this not
        `hold`", because `unpriced` is neither. A position we could not mark is
        being *held* — that is the whole point of the state — and the first
        version of this property read it as closing, which would have shown a
        page announcing an exit that was never going to happen.
        """
        try:
            return ExitRule(self.rule).closes
        except ValueError:
            return False


@dataclass(frozen=True, slots=True)
class StatusSnapshot:
    """Everything about the running agent, as of one slot."""

    as_of: datetime
    session_day: str
    next_slot: str | None
    """When the next cycle is due. `None` once the session is over."""
    slot_sequence: int
    universe: tuple[str, ...]

    equity: Decimal
    cash: Decimal
    buying_power: Decimal
    options_buying_power: Decimal
    options_level: int
    can_trade_spreads: bool
    is_blocked: bool
    is_pattern_day_trader: bool
    session_change: Decimal
    """Today's P&L as the broker computes it — equity less last_equity."""

    peak_equity: Decimal | None
    drawdown_pct: Decimal
    killswitch_tripped: bool
    fills_today: int
    open_structures: int
    open_risk: Decimal

    max_open_structures: int
    max_daily_trades: int
    max_portfolio_risk: Decimal
    """The portfolio loss budget in account currency, so the page renders
    used-against-limit rather than a percentage the reader has to un-multiply
    by an equity figure sitting three tiles away."""
    max_drawdown_pct: Decimal

    positions: tuple[PositionStatus, ...]
    unexplained: tuple[str, ...]
    """Broker legs no journalled fill accounts for. Rendered loudly: they are
    risk the Gate is not budgeting against."""

    profit_target: Decimal
    stop_multiple: Decimal
    min_dte: int

    stage_counts: dict[str, int]
    note: str = ""
    closing: tuple[str, ...] = ()
    """Cycle ids whose close order is already working at the broker.

    Rendered so "the exit rule fired and nothing happened" and "the exit is out
    and waiting to fill" do not look identical on the page. The agent skips
    these positions until the close settles (`agent.book.working_closes`)."""

    @property
    def unrealised(self) -> Decimal:
        return sum((p.unrealised for p in self.positions), Decimal(0))

    @property
    def is_healthy(self) -> bool:
        return (
            self.can_trade_spreads
            and not self.killswitch_tripped
            and not self.unexplained
        )


def build_status(
    *,
    account: AccountRead,
    book: BookRead,
    limits: RiskLimits,
    policy: ExitPolicy,
    marks: dict[str, StructureRisk],
    as_of: datetime,
    next_slot: datetime | None,
    slot_sequence: int,
    universe: tuple[str, ...],
    peak_equity: Decimal | None,
    stage_counts: dict[str, int],
    note: str = "",
) -> StatusSnapshot:
    """Assemble a snapshot. Pure — every input is a value.

    `marks` is keyed by `cycle_id` so a position the caller could not re-price
    is simply absent rather than defaulted to zero. A position rendered at a
    mark of zero would show a 100% profit and an imminent close, which is the
    most alarming possible way to say "the chain request failed".
    """
    return StatusSnapshot(
        as_of=as_of,
        session_day=as_of.date().isoformat(),
        next_slot=next_slot.isoformat() if next_slot else None,
        slot_sequence=slot_sequence,
        universe=universe,
        equity=account.equity,
        cash=account.cash,
        buying_power=account.buying_power,
        options_buying_power=account.options_buying_power,
        options_level=account.options_level,
        can_trade_spreads=account.can_trade_spreads,
        is_blocked=account.is_blocked,
        is_pattern_day_trader=account.is_pattern_day_trader,
        session_change=account.session_change,
        peak_equity=peak_equity,
        drawdown_pct=book.snapshot.drawdown_pct,
        killswitch_tripped=book.snapshot.killswitch_tripped,
        fills_today=book.snapshot.fills_today,
        open_structures=book.snapshot.open_structures,
        open_risk=book.snapshot.open_risk,
        max_open_structures=limits.max_open_structures,
        max_daily_trades=limits.max_daily_trades,
        max_portfolio_risk=limits.max_portfolio_loss(account.equity),
        max_drawdown_pct=limits.max_drawdown_pct,
        positions=tuple(
            _position(item, marks.get(item.cycle_id), policy=policy, as_of=as_of)
            for item in book.held
        ),
        unexplained=tuple(str(leg.contract) for leg in book.unexplained),
        closing=book.closing,
        profit_target=policy.profit_target,
        stop_multiple=policy.stop_multiple,
        min_dte=policy.min_dte,
        stage_counts=stage_counts,
        note=note,
    )


def _position(
    item: HeldPosition,
    mark: StructureRisk | None,
    *,
    policy: ExitPolicy,
    as_of: datetime,
) -> PositionStatus:
    structure = item.position.structure
    common: dict[str, Any] = {
        "cycle_id": item.cycle_id,
        "underlying": str(item.position.underlying),
        "structure": _label(item),
        "quantity": item.position.quantity,
        "expiry": structure.expiry.isoformat(),
        "entry_premium": item.entry_premium,
        "max_loss": item.position.max_loss,
    }
    if mark is None:
        # Unpriced, and saying so. Not zero — see `build_status`.
        return PositionStatus(
            **common,
            days_to_expiry=structure.days_to_expiry(as_of.date()),
            mark=Decimal(0),
            unrealised=Decimal(0),
            fraction_of_credit=None,
            rule="unpriced",
            detail="no fresh quote for every leg; held and re-checked next slot",
            to_target=None,
            to_stop=None,
        )

    decision = evaluate_exit(
        item.position, mark, item.entry_premium, as_of=as_of, policy=policy
    )
    fraction = decision.fraction_of_credit
    return PositionStatus(
        **common,
        days_to_expiry=decision.days_to_expiry,
        mark=mark.net_premium,
        unrealised=decision.unrealised,
        fraction_of_credit=fraction,
        rule=decision.rule.value,
        detail=decision.detail,
        to_target=None if fraction is None else policy.profit_target - fraction,
        to_stop=None if fraction is None else fraction + policy.stop_multiple,
    )


def _label(item: HeldPosition) -> str:
    kind = item.position.structure.kind.value.replace("_", " ")
    strikes = "/".join(
        f"{leg.contract.strike.normalize():f}" for leg in item.position.structure.legs
    )
    return f"{kind} {strikes}"


def write_status(snapshot: StatusSnapshot, *, directory: Path) -> Path:
    """Write the snapshot, atomically, next to the journal.

    Atomic because the dashboard polls: a reader that caught a half-written file
    would show a parse error every few seconds on an otherwise healthy system,
    and the fix people reach for is to make the reader tolerant, which hides the
    real thing it was meant to notice.

    Redacted on the way out with the journal's own function — this file is on
    screen during the demo for the same reasons the journal is (specs/06 D4).
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / STATUS_FILENAME
    temporary = path.with_suffix(".json.tmp")
    document = redact(encode(snapshot))
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def read_status(directory: Path) -> dict[str, Any] | None:
    """Read the snapshot back. `None` when the agent has never run.

    A partially written or corrupt file also reads as `None` rather than
    raising: the dashboard's job in that moment is to say it has lost contact,
    not to return a 500.
    """
    path = directory / STATUS_FILENAME
    if not path.is_file():
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return loaded if isinstance(loaded, dict) else None
