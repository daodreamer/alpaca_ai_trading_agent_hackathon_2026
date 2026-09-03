"""What the equity agent is doing right now, written to a file — specs/09 D10.

The journal answers "what did it decide, and why". It cannot answer "is it
running, what does it hold, how far has the book drifted", because a journal is
written after a decision and on this strategy a decision happens once a session.

So the agent drops a snapshot every heartbeat, and the dashboard reads the file.

**Via a file, deliberately.** `tests/test_boundaries.py` forbids
`alphagate.interface` from importing `execution`, `marketdata` or `live`, which
is what makes "there is no code path from a browser to an order" a checkable
statement. Handing the dashboard a broker session to render a positions table
would delete that guarantee for a feature a JSON file provides just as well.

It also fails honestly. If the agent stops, the file stops being rewritten and
`age_seconds` grows — so a stale page is visibly stale instead of quietly
showing this morning's book as though it were now.

The block that makes this a demo rather than a screenshot is `strategy`: the
fingerprint, the sealed-run alpha, beta and `t`, the number of looks that `t`
had to clear a bar for, and the fact that the sealed window can refute and
cannot confirm. Every number there is recorded and none of it is acted on.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from alphagate.core.identifiers import Ticker
from alphagate.equity import EquityPolicy, Holding, Mark, TargetBook
from alphagate.execution import AccountRead
from alphagate.journal import encode, redact

__all__ = [
    "EQUITY_STATUS_FILENAME",
    "EquityStatusSnapshot",
    "PositionLine",
    "build_equity_status",
    "read_equity_status",
    "write_equity_status",
]

EQUITY_STATUS_FILENAME = "equity-status.json"


@dataclass(frozen=True, slots=True)
class PositionLine:
    """One symbol: what the book wants, what the account holds, and the gap.

    `drift` and `inside_band` are the point of this type. "8.2% target" is a
    fact about the strategy; "$140 away from target, inside the $250 band" is
    what tells you what the agent is about to do — which is what someone
    watching it actually wants to know.

    A symbol with no mark is rendered with `price=None` rather than zero. A
    position shown at a mark of zero would read as a total loss, which is an
    alarming way to say "the snapshot request missed one name".
    """

    symbol: str
    target_weight: Decimal
    held_weight: Decimal
    target_shares: Decimal | None
    held_shares: Decimal
    price: Decimal | None
    price_age_seconds: float | None
    market_value: Decimal
    drift: Decimal
    threshold: Decimal
    """How much drift this position must show before it is traded.

    Per position, because the band is proportional (specs/09 D3): a fifth of an
    $8,276 core name is $1,655, and a fifth of a $194 sleeve name is $39. A page
    showing one global number would be wrong about a hundred of the rows."""
    inside_band: bool
    core: bool
    """True when this name is part of the 10-name core rather than the sleeve.
    The core is the strategy's actual selection; the sleeve is the benchmark it
    holds so that an idle deviation budget does not sit in cash."""
    tradeable: bool


@dataclass(frozen=True, slots=True)
class EquityStatusSnapshot:
    """Everything about the running equity agent, as of one heartbeat."""

    as_of: datetime
    session_day: str
    next_pass: str | None
    heartbeat_sequence: int

    strategy: dict[str, Any]
    """Provenance, copied from the book. Rendered, never acted on."""

    equity: Decimal
    cash: Decimal
    buying_power: Decimal
    session_change: Decimal
    peak_equity: Decimal | None
    drawdown_pct: Decimal
    killswitch_tripped: bool
    is_blocked: bool

    gross_exposure: Decimal
    positions_held: int
    positions_wanted: int
    drift_band_pct: Decimal
    min_order_notional: Decimal
    unpriced: tuple[str, ...]
    """Symbols in the book the snapshot request could not price. Rendered
    loudly: they are positions the plan is deliberately not touching."""
    stale: tuple[str, ...]
    """Symbols whose mark is older than the policy allows.

    Separate from `unpriced` because the two mean different things and the
    remedies differ. An unpriced name is a data gap; a stale one is usually the
    whole book at once, and the reason is that the market is closed. A page that
    conflated them would report a hundred data gaps every evening."""
    off_book: tuple[str, ...]
    """Held names the book does not want. They are sold on the next pass — this
    is the line that says so before it happens."""

    orders_today: int
    turnover_today: Decimal
    max_daily_orders: int
    max_daily_turnover: Decimal
    max_position_pct: Decimal
    max_drawdown_pct: Decimal

    lines: tuple[PositionLine, ...]
    stage_counts: dict[str, int]
    note: str = ""

    @property
    def is_healthy(self) -> bool:
        return (
            not self.killswitch_tripped
            and not self.is_blocked
            and not self.unpriced
            and bool(self.strategy)
        )

    @property
    def marks_are_fresh(self) -> bool:
        """Whether the book could be traded on the prices it is showing.

        Reported beside `is_healthy` rather than folded into it: an agent
        holding a correct book on stale marks at 22:00 is entirely healthy and
        entirely unable to trade, and calling that unhealthy would cry wolf every
        evening."""
        return not self.stale


def build_equity_status(
    *,
    account: AccountRead,
    book: TargetBook | None,
    strategy: dict[str, Any],
    holdings: tuple[Holding, ...],
    marks: dict[Ticker, Mark],
    policy: EquityPolicy,
    sleeve_equity: Decimal,
    peak_equity: Decimal | None,
    killswitch_tripped: bool,
    orders_today: int,
    turnover_today: Decimal,
    as_of: datetime,
    next_pass: datetime | None,
    heartbeat_sequence: int,
    stage_counts: dict[str, int],
    note: str = "",
) -> EquityStatusSnapshot:
    """Assemble a snapshot. Pure — every input is a value, including the time.

    `book=None` is a real state and not an error: the agent is up, and there is
    no artefact to execute. The page then shows the account and says so, which
    is more useful than an empty page and much more useful than a page that
    looks like a flat book.

    **Two equity figures, and they are not interchangeable.** `account.equity` is
    the broker's whole account and is what the page reports, because that is the
    number a reader can check against Alpaca (see `interface/sleeves.py` for the
    rest of that argument). `sleeve_equity` is this strategy's own capital
    (specs/03 D6) and is what the drawdown is measured on — because the Gate's
    `drawdown_killswitch` measures it there, and a page whose drawdown is
    computed on a different quantity from the one that stops trading is a page
    that cannot warn you. On 2026-09-03 the two differed by a factor of a
    hundred: 0.08% here, 10.07% at the Gate, on the pass it refused.
    """
    held = {holding.symbol: holding for holding in holdings}
    wanted = dict(book.weights) if book else {}
    core = set(book.core_weights) if book else set()
    equity = account.equity

    lines: list[PositionLine] = []
    unpriced: list[str] = []
    stale: list[str] = []
    off_book: list[str] = []
    gross = Decimal(0)

    for symbol in sorted(set(wanted) | set(held), key=str):
        target_weight = wanted.get(symbol, Decimal(0))
        holding = held.get(symbol)
        held_shares = holding.shares if holding else Decimal(0)
        mark = marks.get(symbol)
        price = mark.price if mark else None

        if price is None and symbol in wanted:
            unpriced.append(str(symbol))
        elif mark is not None and mark.age_seconds > policy.max_quote_age:
            stale.append(str(symbol))
        if symbol not in wanted and held_shares != 0:
            off_book.append(str(symbol))

        # The broker's own market_value is the fallback, for the same reason the
        # Gate's snapshot uses it: a position valued at zero would understate the
        # book by exactly the names we know least about.
        if price is not None:
            value = held_shares * price
        else:
            value = holding.market_value if holding else Decimal(0)
        gross += value
        target_notional = target_weight * equity
        threshold = policy.threshold(target_notional, value)
        lines.append(
            PositionLine(
                symbol=str(symbol),
                target_weight=target_weight,
                held_weight=value / equity if equity else Decimal(0),
                target_shares=(
                    target_notional / price if price is not None and price > 0 else None
                ),
                held_shares=held_shares,
                price=price,
                price_age_seconds=mark.age_seconds if mark else None,
                market_value=value,
                drift=target_notional - value,
                threshold=threshold,
                inside_band=abs(target_notional - value) < threshold,
                core=symbol in core,
                tradeable=mark.tradeable if mark else False,
            )
        )

    return EquityStatusSnapshot(
        as_of=as_of,
        session_day=as_of.date().isoformat(),
        next_pass=next_pass.isoformat() if next_pass else None,
        heartbeat_sequence=heartbeat_sequence,
        strategy=strategy,
        equity=equity,
        cash=account.cash,
        buying_power=account.buying_power,
        session_change=account.session_change,
        peak_equity=peak_equity,
        drawdown_pct=_drawdown(peak_equity, sleeve_equity),
        killswitch_tripped=killswitch_tripped,
        is_blocked=account.is_blocked,
        gross_exposure=gross / equity if equity else Decimal(0),
        positions_held=sum(1 for h in holdings if h.shares != 0),
        positions_wanted=len(wanted),
        drift_band_pct=policy.drift_band_pct,
        min_order_notional=policy.min_order_notional,
        unpriced=tuple(unpriced),
        stale=tuple(stale),
        off_book=tuple(off_book),
        orders_today=orders_today,
        turnover_today=turnover_today,
        max_daily_orders=policy.max_daily_orders,
        max_daily_turnover=policy.max_daily_turnover(equity),
        max_position_pct=policy.max_position_pct,
        max_drawdown_pct=policy.max_drawdown_pct,
        lines=tuple(lines),
        stage_counts=stage_counts,
        note=note,
    )


def _drawdown(peak: Decimal | None, equity: Decimal) -> Decimal:
    if peak is None or peak <= 0:
        return Decimal(0)
    return max(Decimal(0), (peak - equity) / peak)


def write_equity_status(snapshot: EquityStatusSnapshot, *, directory: Path) -> Path:
    """Write the snapshot, atomically, next to the journal.

    Atomic because the dashboard polls: a reader that caught a half-written file
    would show a parse error every few seconds on an otherwise healthy system,
    and the fix people reach for is to make the reader tolerant — which hides the
    real thing it was meant to notice.

    Redacted on the way out with the journal's own function. This file is on
    screen during the demo for the same reasons the journal is (specs/06 D4).
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / EQUITY_STATUS_FILENAME
    temporary = path.with_suffix(".json.tmp")
    document = redact(encode(snapshot))
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def read_equity_status(directory: Path) -> dict[str, Any] | None:
    """Read the snapshot back. `None` when the agent has never run.

    A partially written or corrupt file also reads as `None` rather than
    raising: the dashboard's job in that moment is to say it has lost contact,
    not to return a 500.
    """
    path = directory / EQUITY_STATUS_FILENAME
    if not path.is_file():
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return loaded if isinstance(loaded, dict) else None
