"""Append-only decision journal — specs/06 D1, D3, D4, D5.

JSONL, one object per line, one file per trading day. Not a database: a
seven-day project does not need migrations, and an append-only text file is
trivially replayable, diffable, and shippable with the submission as evidence
(adr/0001 D4).

Three disciplines, all of them cheap here and expensive to add later:

**Append then flush.** A crash mid-cycle loses at most the current line, and a
truncated final line is skipped on read rather than failing the load. A journal
that refuses to open because the process died mid-write is a journal you stop
trusting on the morning you need it.

**Amendment, not mutation** (D3). A record is written once. Later facts — a fill
hours after submission, realised P&L on close — arrive as separate lines keyed
by `cycle_id`. Reading applies them in file order, so the original decision stays
exactly as it was made and no hindsight leaks backwards into it. Same rule as
specs/01 Rule 4, applied to the record of what we knew.

**Redaction** (D4). Nothing here writes an API key, a secret, an account number
or an account id. `redact` is applied to every line on the way out, and a test
runs it against a fixture that deliberately contains a fake key. The demo video
shows this file; a leaked key on screen is unrecoverable.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Final, Protocol, runtime_checkable

__all__ = ["REDACTED", "Amendable", "Journal", "encode", "redact"]

REDACTED: Final = "[REDACTED]"

_SECRET_KEYS: Final = frozenset(
    {
        "account_number",
        "account_id",
        "api_key",
        "apikey",
        "authorization",
        "secret",
        "secret_key",
        "password",
        "token",
        "alpaca_api_key",
        "alpaca_secret_key",
        "anthropic_api_key",
        "deepseek_api_key",
    }
)

_SECRET_PATTERNS: Final = (
    re.compile(r"\bPK[A-Z0-9]{18,}\b"),
    re.compile(r"\bAK[A-Z0-9]{18,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._-]{16,}", re.IGNORECASE),
    re.compile(r"\bPA[0-9][A-Z0-9]{6,}\b"),
)
"""Shapes, not values. A pattern list cannot be defeated by a key we forgot to
add to a deny-list of literals, and it costs one regex pass per line.

The last one is an Alpaca paper account number (`PA3ABCDEFG`). It is here rather
than only in `_SECRET_KEYS` because the dangerous appearance of an account
number is not the one in a field called `account_number` — it is the one pasted
into a rationale or an error message, where no key name protects it."""

_ACCOUNT_MARKERS: Final = frozenset(
    {"account_number", "buying_power", "options_buying_power", "portfolio_value"}
)
"""Fields that only an account object has.

This exists because of the one credential a shape cannot catch: **the account
id**. D4 forbids writing it, but Alpaca returns it as a bare `id`, and an order
also has a bare `id` that D4 explicitly requires us to keep — both are plain
UUIDs, so no regex can tell them apart. The only thing that distinguishes them
is the object they sit in, so that is what we look at: an `id` inside an object
carrying `buying_power` is an account, an `id` beside `client_order_id` is an
order.

Structure, not naming, and deliberately more than one marker: a redaction that
turns on a single field name is one Alpaca release away from being wrong."""

_ACCOUNT_PARENTS: Final = frozenset({"account", "account_info", "trading_account"})
"""Keys whose value is an account object even when it has been trimmed to
`{"id": ...}` and carries no marker field of its own."""

_IDENTITY_KEYS: Final = frozenset({"id", "uuid"})
"""Redacted **only** inside an account object. Everywhere else these are order
and asset identifiers, and reconciliation needs them (D4)."""


def redact(value: Any, *, parent_key: str = "") -> Any:
    """Strip credentials by key name, by shape, and by containing object.

    Order id and `client_order_id` are deliberately **not** redacted: specs/06
    D4 draws the line at account identity, because reconciliation needs the
    order ids and a journal you cannot reconcile against is decoration.

    The account id is the hard case, and the reason this function has to know
    where in the document it is standing. See `_ACCOUNT_MARKERS`.
    """
    if isinstance(value, Mapping):
        in_account = _is_account_object(value, parent_key)
        return {
            key: (
                REDACTED
                if _is_secret(str(key), in_account=in_account)
                else redact(item, parent_key=str(key))
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact(item, parent_key=parent_key) for item in value]
    if isinstance(value, str):
        redacted = value
        for pattern in _SECRET_PATTERNS:
            redacted = pattern.sub(REDACTED, redacted)
        return redacted
    return value


def _is_secret(key: str, *, in_account: bool) -> bool:
    lowered = key.lower()
    if lowered in _SECRET_KEYS:
        return True
    return in_account and lowered in _IDENTITY_KEYS


def _is_account_object(value: Mapping[Any, Any], parent_key: str) -> bool:
    """Whether this mapping is Alpaca's account — and so whether its `id` is one.

    Fails safe in the direction that matters. A false positive redacts an order
    id and breaks a reconciliation, which is loud and fixable in a minute. A
    false negative writes the account id into the file that goes on video, which
    is neither.
    """
    if parent_key.lower() in _ACCOUNT_PARENTS:
        return True
    return any(marker in value for marker in _ACCOUNT_MARKERS)

def encode(value: Any) -> Any:
    """Turn domain values into JSON-safe ones, losing no precision.

    `Decimal` becomes a **string**, never a float. Round-tripping money through
    a float is exactly the mistake specs/01 Rule 3 exists to prevent, and a
    journal is the last place to make it — the record would disagree with the
    order it records.
    """
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): encode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [encode(item) for item in value]
    if hasattr(value, "__dataclass_fields__"):
        return {
            name: encode(getattr(value, name))
            for name in value.__dataclass_fields__
            if not name.startswith("_")
        }
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


@runtime_checkable
class Amendable(Protocol):
    """Anything that can be written as a D3 amendment: it names its cycle.

    Deliberately the smallest possible surface. Storage does not need to know
    what an outcome contains — only which decision it belongs to — and a wider
    contract here would be storage reaching into the domain it stores.
    """

    @property
    def cycle_id(self) -> str: ...


@dataclass(frozen=True, slots=True)
class Journal:
    """One directory of daily JSONL files."""

    directory: Path

    def path_for(self, day: date) -> Path:
        return self.directory / f"{day.isoformat()}.jsonl"

    def days(self) -> tuple[date, ...]:
        """Every day this journal holds a file for, oldest first.

        Oldest first because the readers that want the whole history are
        accumulating something over it — realised P&L, an open-position match —
        and both want the records in the order they happened.

        A file whose name is not a date is skipped rather than fatal. The
        directory is also a place humans put things.
        """
        if not self.directory.is_dir():
            return ()
        found: list[date] = []
        for path in self.directory.glob("*.jsonl"):
            try:
                found.append(date.fromisoformat(path.stem))
            except ValueError:
                continue
        return tuple(sorted(found))

    def read_through(self, day: date) -> list[dict[str, Any]]:
        """Every record up to and including `day`, oldest first.

        The whole-history read, for the two callers that cannot work from one
        file. `agent.book.read_book` is both of them: realised P&L is cumulative
        by definition, and matching broker legs against journalled fills needs
        the day a position was *opened*, which for anything held overnight is
        not the day being read.

        Bounded by `day` rather than reading the directory whole, so that a
        replay of Tuesday cannot see Wednesday. Look-ahead is look-ahead even
        when the thing leaking backwards is our own P&L.
        """
        records: list[dict[str, Any]] = []
        for each in self.days():
            if each > day:
                break
            records.extend(self.read(each))
        return records

    def append(self, entry: Any, *, day: date | None = None) -> Path:
        """Write one record. Encode, redact, append, flush.

        Redaction runs on the *encoded* form, so it sees the same strings the
        file will contain rather than the objects behind them.
        """
        document = redact(encode(entry))
        when = day if day is not None else _day_of(document)
        path = self.path_for(when)
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(document, ensure_ascii=False, sort_keys=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
        return path

    def amend(self, cycle_id: str, **facts: Any) -> Path:
        """Add a later fact as its own line — specs/06 D3. Never an edit."""
        return self.append(
            {"amendment": True, "cycle_id": cycle_id, **{k: encode(v) for k, v in facts.items()}}
        )

    def record_outcome(self, outcome: Amendable, *, day: date | None = None) -> Path:
        """Amend a cycle with what became of its order — specs/06 D2 `outcome`.

        Takes the record rather than keyword arguments so the shape of an
        outcome is decided in one place, and the same shape is written by the
        live reconciler, the backtest and the test suite.

        The parameter is a `Protocol` rather than `OutcomeRecord` so that this
        module — which is D1, storage, and nothing else — does not acquire a
        dependency on `journal.outcome` and through it on `alphagate.execution`.
        The contract is the whole of what storage needs to know: an amendment
        must name the cycle it amends.
        """
        cycle_id = str(outcome.cycle_id or "")
        if not cycle_id:
            raise ValueError("an outcome must name the cycle it amends")
        return self.append(
            {"amendment": True, "cycle_id": cycle_id, "outcome": encode(outcome)},
            day=day,
        )

    def read(self, day: date) -> list[dict[str, Any]]:
        """Load a day, applying amendments in file order.

        A truncated final line is skipped, not fatal (D1). It is the only line
        that can be truncated — everything before it was flushed — so skipping
        exactly it loses exactly the cycle that was in flight when the process
        died.

        An amendment that arrives **before** the record it amends is held and
        applied when the original turns up (test plan item 4). File order is the
        order the facts were learned, not the order they were caused: a
        reconciler restarted mid-morning can flush a fill it read from the
        broker before the day it belongs to has been replayed into memory, and
        a reader that dropped it would lose the fill and say nothing.

        Amendments are still applied in file order among themselves, so a later
        fact beats an earlier one whatever order the originals appear in — which
        is what makes "the same final state" true rather than approximately
        true.
        """
        entries: dict[str, dict[str, Any]] = {}
        order: list[str] = []
        pending: dict[str, list[dict[str, Any]]] = {}

        for line in self._lines(day):
            cycle_id = str(line.get("cycle_id", ""))
            if not cycle_id:
                continue
            if line.get("amendment"):
                amendment = {k: v for k, v in line.items() if k != "amendment"}
                if cycle_id in entries:
                    entries[cycle_id] = {**entries[cycle_id], **amendment}
                else:
                    pending.setdefault(cycle_id, []).append(amendment)
                continue
            if cycle_id not in entries:
                order.append(cycle_id)
            entries[cycle_id] = line
            for held in pending.pop(cycle_id, ()):
                entries[cycle_id] = {**entries[cycle_id], **held}

        return [entries[cycle_id] for cycle_id in order]

    def orphaned_amendments(self, day: date) -> tuple[str, ...]:
        """Cycle ids amended but never recorded, in file order.

        Not an error and not silently swallowed either. It means a fill exists
        for a decision this day's file does not contain — the cycle was journal-
        led to another day, or the line that held it was the truncated one — and
        that is a reconciliation question, so it is answerable rather than lost.
        """
        seen: set[str] = set()
        orphans: list[str] = []
        for line in self._lines(day):
            cycle_id = str(line.get("cycle_id", ""))
            if not cycle_id:
                continue
            if line.get("amendment"):
                if cycle_id not in seen and cycle_id not in orphans:
                    orphans.append(cycle_id)
                continue
            seen.add(cycle_id)
            if cycle_id in orphans:
                orphans.remove(cycle_id)
        return tuple(orphans)

    def duplicate_cycles(self, day: date) -> dict[str, int]:
        """Cycle ids written more than once, and how many times. Usually empty.

        `cycle_id` is collision-free by construction (specs/06 D2): the day, the
        underlying and the session sequence. A duplicate therefore does not mean
        the id scheme failed — it means the same sequence was run twice, which
        is what happens when an operator restarts the agent mid-session or runs
        two processes over one watchlist.

        It matters because `read` keys on `cycle_id`, so duplicates collapse:
        three decisions become one line and the earlier two are on disk but not
        in the day. Silently is the wrong way for that to happen to a journal
        whose first claim is one record per cycle, so it is reportable.
        """
        counts: dict[str, int] = {}
        for line in self._lines(day):
            if line.get("amendment"):
                continue
            cycle_id = str(line.get("cycle_id", ""))
            if cycle_id:
                counts[cycle_id] = counts.get(cycle_id, 0) + 1
        return {cycle_id: n for cycle_id, n in counts.items() if n > 1}

    def raw_lines(self, day: date) -> list[dict[str, Any]]:
        """Every line as written, amendments included. For the redaction test."""
        return list(self._lines(day))

    def _lines(self, day: date) -> Iterator[dict[str, Any]]:
        path = self.path_for(day)
        if not path.is_file():
            return
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                # A truncated tail. Skipping is the documented behaviour; the
                # alternative is a day that will not load at all.
                continue
            if isinstance(parsed, dict):
                yield parsed


def _day_of(document: Any) -> date:
    if isinstance(document, Mapping):
        stamp = document.get("as_of")
        if isinstance(stamp, str):
            try:
                return datetime.fromisoformat(stamp).date()
            except ValueError:  # pragma: no cover - as_of is always an isoformat
                pass
        cycle_id = document.get("cycle_id")
        if isinstance(cycle_id, str) and len(cycle_id) >= 10:
            try:
                return date.fromisoformat(cycle_id[:10])
            except ValueError:  # pragma: no cover
                pass
    raise ValueError("cannot determine the trading day for this record; pass day=")
