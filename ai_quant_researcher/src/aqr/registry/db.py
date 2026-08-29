"""Experiment database and strategy registry (architecture sections 16.1, 17).

Section 16.1 makes the point plainly: without a record of every experiment, no
one -- human or model -- can answer "what has the LLM already tried". Worse, the
overfitting detector's most important input is *how many backtests bought this
result*, and that number is unknowable unless every attempt is written down,
including the failures. Especially the failures.

SQLite, one file, no server. The MVP does not need PostgreSQL, and a research
log that requires infrastructure to be running is a research log that quietly
stops being written.

Everything needed to replay an experiment is stored: the full spec, the data
window, the code and prompt hashes, the metrics, and the verdict.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from aqr.dsl.loader import dumps
from aqr.dsl.schema import StrategySpec

__all__ = [
    "ExperimentRecord",
    "Preregistration",
    "PreregistrationError",
    "Registry",
    "Status",
    "StrategyRecord",
    "TaintReport",
]

Status = Literal["CANDIDATE", "ACTIVE", "PAPER", "LIVE", "DEGRADED", "RETIRED", "REJECTED"]

_VALID_STATUS: set[str] = {
    "CANDIDATE",
    "ACTIVE",
    "PAPER",
    "LIVE",
    "DEGRADED",
    "RETIRED",
    "REJECTED",
}

# A strategy may only move along paths that make sense. Promoting straight from
# CANDIDATE to LIVE is exactly the shortcut this project exists to prevent.
_TRANSITIONS: dict[str, set[str]] = {
    "CANDIDATE": {"PAPER", "REJECTED", "ACTIVE"},
    "ACTIVE": {"PAPER", "REJECTED", "RETIRED"},
    "PAPER": {"LIVE", "DEGRADED", "RETIRED", "REJECTED"},
    "LIVE": {"DEGRADED", "RETIRED"},
    "DEGRADED": {"PAPER", "RETIRED", "LIVE"},
    "RETIRED": set(),
    "REJECTED": set(),
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS strategies (
    fingerprint   TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    version       INTEGER NOT NULL,
    parent        TEXT,
    status        TEXT NOT NULL,
    hypothesis    TEXT NOT NULL DEFAULT '',
    spec_yaml     TEXT NOT NULL,
    score         REAL,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS experiments (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    fingerprint       TEXT NOT NULL,
    strategy_name     TEXT NOT NULL,
    hypothesis        TEXT NOT NULL DEFAULT '',
    symbols           TEXT NOT NULL,
    timeframe         TEXT NOT NULL,
    data_start        TEXT NOT NULL,
    data_end          TEXT NOT NULL,
    dataset_version   TEXT NOT NULL,
    train_metrics     TEXT,
    oos_metrics       TEXT,
    robustness        TEXT,
    overfitting       TEXT,
    evaluation        TEXT,
    verdict           TEXT,
    score             REAL,
    backtests_run     INTEGER NOT NULL DEFAULT 1,
    llm_model         TEXT,
    prompt_hash       TEXT,
    code_hash         TEXT,
    error             TEXT,
    seal              TEXT,
    created_at        TEXT NOT NULL,
    FOREIGN KEY (fingerprint) REFERENCES strategies (fingerprint)
);

CREATE INDEX IF NOT EXISTS idx_experiments_fingerprint ON experiments (fingerprint);
CREATE INDEX IF NOT EXISTS idx_experiments_created ON experiments (created_at);
CREATE INDEX IF NOT EXISTS idx_strategies_status ON strategies (status);

-- One row per candidate that was ever declared. Written *before* the sealed
-- years are read, which is the only thing that distinguishes a prediction from
-- a description. ``fingerprint`` is the primary key rather than an id, so a
-- second declaration collides at the storage layer and not merely in a check.
CREATE TABLE IF NOT EXISTS preregistration (
    fingerprint     TEXT PRIMARY KEY,
    declared_at     TEXT NOT NULL,
    selection_rule  TEXT NOT NULL,
    seal_digest     TEXT NOT NULL,
    FOREIGN KEY (fingerprint) REFERENCES strategies (fingerprint)
);

-- One row per target book handed off. The artefact on disk is the interface;
-- this is the audit trail behind it, so a book found in a directory can be
-- traced back to the hypothesis, the campaign and the sealed run that justified
-- it. ``digest`` is what makes that traceability survive the file being edited.
CREATE TABLE IF NOT EXISTS target_books (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    fingerprint   TEXT NOT NULL,
    as_of         TEXT NOT NULL,
    path          TEXT NOT NULL,
    digest        TEXT NOT NULL,
    book          TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    FOREIGN KEY (fingerprint) REFERENCES strategies (fingerprint)
);

CREATE INDEX IF NOT EXISTS idx_target_books_fingerprint ON target_books (fingerprint);

CREATE TABLE IF NOT EXISTS status_log (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    fingerprint   TEXT NOT NULL,
    from_status   TEXT,
    to_status     TEXT NOT NULL,
    reason        TEXT NOT NULL DEFAULT '',
    created_at    TEXT NOT NULL
);
"""


# Columns added after the first campaigns were already recorded. A schema
# change that required starting over would delete ``distinct_hypotheses()``,
# which is the multiple-comparisons denominator and the one number in this
# project that cannot be reconstructed.
_MIGRATIONS: tuple[tuple[str, str, str], ...] = (
    ("experiments", "seal", "TEXT"),
    ("experiments", "costs", "TEXT"),
    ("strategies", "sealed_run_at", "TEXT"),
    ("strategies", "sealed_result", "TEXT"),
)


class PreregistrationError(RuntimeError):
    """Raised when the one-shot protocol would be violated.

    Its own type rather than ``ValueError`` because the CLI has to distinguish
    "you typed the wrong fingerprint" from "this seal has already been spent",
    and the second one is not a mistake the user can retry their way out of.
    """


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _json(value: Any) -> str | None:
    return None if value is None else json.dumps(value, default=str, sort_keys=True)


@dataclass(slots=True)
class StrategyRecord:
    fingerprint: str
    name: str
    version: int
    parent: str | None
    status: str
    hypothesis: str
    spec_yaml: str
    score: float | None
    created_at: str
    updated_at: str


@dataclass(slots=True)
class ExperimentRecord:
    """One evaluated hypothesis. Immutable once written."""

    fingerprint: str
    strategy_name: str
    symbols: tuple[str, ...]
    timeframe: str
    data_start: str
    data_end: str
    dataset_version: str
    hypothesis: str = ""
    train_metrics: dict[str, Any] | None = None
    oos_metrics: dict[str, Any] | None = None
    robustness: dict[str, Any] | None = None
    overfitting: dict[str, Any] | None = None
    evaluation: dict[str, Any] | None = None
    verdict: str | None = None
    score: float | None = None
    backtests_run: int = 1
    llm_model: str | None = None
    prompt_hash: str | None = None
    code_hash: str | None = None
    error: str | None = None
    costs: dict[str, Any] | None = None
    """The cost schedule this verdict was decided under.

    Cost retention is a fatal gate, so the schedule is part of the verdict rather
    than context for it. Two runs under different schedules are not comparable,
    and comparing them anyway is how a change of broker gets attributed to a
    change of strategy. ``None`` means *not recorded* — every experiment written
    before this column existed."""
    seal: dict[str, Any] | None = None
    """The seal certificate of the process that ran this experiment.

    Written with every experiment so the ancestry taint check is a query rather
    than an act of trust. ``None`` means *not recorded*, which the check reports
    separately from *clean*: silence is not evidence of innocence."""
    id: int | None = field(default=None)
    created_at: str = field(default_factory=_now)


@dataclass(frozen=True, slots=True)
class Preregistration:
    """One candidate, declared before the sealed years were read.

    ``selection_rule`` is the part that carries the weight. "The best of 305
    hypotheses by score" and "the only one that cleared t > 2" are different
    claims about the multiple-comparisons problem, and which of them was true is
    unrecoverable an hour after the fact unless somebody wrote it down first.

    ``seal_digest`` pins the state of the research ledger at the moment of
    declaration, so a declaration cannot later be claimed to predate loads that
    it did not.
    """

    fingerprint: str
    declared_at: str
    selection_rule: str
    seal_digest: str


@dataclass(frozen=True, slots=True)
class TaintReport:
    """Whether anything in the ancestry of a candidate read the embargoed years.

    Three counts rather than one boolean, because "no taint found" and "no taint
    recorded" are different findings and collapsing them would let this check
    pass on a database where it never ran.
    """

    fingerprint: str
    experiments: int
    """Every experiment in the ancestry: its own, and every sibling from the
    same campaigns."""
    tainted: tuple[int, ...] = ()
    """Experiment ids whose certificate says the process was tainted."""
    unrecorded: int = 0
    """Experiments written before the seal was recorded. Not clean -- unknown."""
    campaigns: tuple[str, ...] = ()
    """The ``run_id`` of every process that touched this fingerprint."""

    @property
    def clean(self) -> bool:
        return not self.tainted

    def __str__(self) -> str:
        state = "clean" if self.clean else f"TAINTED ({len(self.tainted)} experiments)"
        note = f", {self.unrecorded} with no seal recorded" if self.unrecorded else ""
        return (
            f"ancestry {state}: {self.experiments} experiments across "
            f"{len(self.campaigns)} campaign(s){note}"
        )


class Registry:
    """The research log. Open it, write to it, never edit history by hand."""

    def __init__(self, path: Path | str = "runs/research.sqlite") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._migrate()
        self._conn.commit()

    def _migrate(self) -> None:
        """Add columns an older research log does not have. Never drops anything."""
        for table, column, kind in _MIGRATIONS:
            existing = {
                row["name"]
                for row in self._conn.execute(f"PRAGMA table_info({table})").fetchall()
            }
            if column not in existing:
                self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {kind}")

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Registry:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        try:
            yield self._conn
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    # ----------------------------------------------------------------- #
    # Strategies
    # ----------------------------------------------------------------- #

    def upsert_strategy(self, spec: StrategySpec, status: Status = "CANDIDATE") -> str:
        """Record a strategy. Re-registering an existing one keeps its status.

        The fingerprint is content-derived, so proposing the same rule twice --
        which an LLM will do -- collapses onto one row rather than inflating the
        strategy count.
        """
        fingerprint = spec.fingerprint()
        with self._tx() as conn:
            existing = conn.execute(
                "SELECT status FROM strategies WHERE fingerprint = ?", (fingerprint,)
            ).fetchone()
            if existing:
                conn.execute(
                    "UPDATE strategies SET updated_at = ? WHERE fingerprint = ?",
                    (_now(), fingerprint),
                )
                return fingerprint
            now = _now()
            conn.execute(
                "INSERT INTO strategies (fingerprint, name, version, parent, status, "
                "hypothesis, spec_yaml, score, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?, ?)",
                (
                    fingerprint,
                    spec.name,
                    spec.version,
                    spec.parent,
                    status,
                    spec.hypothesis,
                    dumps(spec),
                    now,
                    now,
                ),
            )
            conn.execute(
                "INSERT INTO status_log (fingerprint, from_status, to_status, reason, created_at) "
                "VALUES (?, NULL, ?, 'registered', ?)",
                (fingerprint, status, now),
            )
        return fingerprint

    def set_status(self, fingerprint: str, status: Status, reason: str = "") -> None:
        """Move a strategy through its lifecycle, refusing illegal jumps."""
        if status not in _VALID_STATUS:
            raise ValueError(f"unknown status {status!r}; valid: {sorted(_VALID_STATUS)}")
        current = self.get_strategy(fingerprint)
        if current is None:
            raise KeyError(f"no strategy {fingerprint!r}")
        if current.status == status:
            return
        allowed = _TRANSITIONS[current.status]
        if status not in allowed:
            raise ValueError(
                f"{fingerprint}: cannot go {current.status} -> {status}; "
                f"allowed from {current.status}: {sorted(allowed) or 'nothing (terminal)'}"
            )
        with self._tx() as conn:
            now = _now()
            conn.execute(
                "UPDATE strategies SET status = ?, updated_at = ? WHERE fingerprint = ?",
                (status, now, fingerprint),
            )
            conn.execute(
                "INSERT INTO status_log (fingerprint, from_status, to_status, reason, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (fingerprint, current.status, status, reason, now),
            )

    def set_score(self, fingerprint: str, score: float) -> None:
        with self._tx() as conn:
            conn.execute(
                "UPDATE strategies SET score = ?, updated_at = ? WHERE fingerprint = ?",
                (float(score), _now(), fingerprint),
            )

    def get_strategy(self, fingerprint: str) -> StrategyRecord | None:
        row = self._conn.execute(
            "SELECT * FROM strategies WHERE fingerprint = ?", (fingerprint,)
        ).fetchone()
        return _strategy_from_row(row) if row else None

    def strategies(
        self, status: Status | None = None, limit: int = 100
    ) -> list[StrategyRecord]:
        if status:
            rows = self._conn.execute(
                "SELECT * FROM strategies WHERE status = ? "
                "ORDER BY COALESCE(score, -1) DESC, updated_at DESC LIMIT ?",
                (status, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM strategies ORDER BY COALESCE(score, -1) DESC, updated_at DESC "
                "LIMIT ?",
                (limit,),
            ).fetchall()
        return [_strategy_from_row(r) for r in rows]

    # ----------------------------------------------------------------- #
    # Pre-registration and the sealed run
    # ----------------------------------------------------------------- #

    def preregister(
        self, fingerprint: str, *, selection_rule: str, seal_digest: str
    ) -> Preregistration:
        """Declare a candidate for the sealed run, before any sealed bar is read.

        Refused if the strategy is unknown, if the rule is blank, if the
        candidate has already been declared, or if its seal has already been
        spent. Each of those is a way the declaration would stop being a
        prediction and become a description.
        """
        rule = selection_rule.strip()
        if not rule:
            raise PreregistrationError(
                "the selection rule may not be blank: an unstated rule is "
                "indistinguishable from one invented after the result"
            )
        row = self._conn.execute(
            "SELECT sealed_run_at FROM strategies WHERE fingerprint = ?", (fingerprint,)
        ).fetchone()
        if row is None:
            raise PreregistrationError(
                f"no strategy {fingerprint!r}: register the spec before declaring it"
            )
        if row["sealed_run_at"]:
            raise PreregistrationError(
                f"{fingerprint}: the seal was already spent on {row['sealed_run_at']}"
            )
        if self.preregistration(fingerprint) is not None:
            raise PreregistrationError(
                f"{fingerprint} is already pre-registered; a second declaration is a "
                "rewritten hypothesis wearing the timestamp of the first"
            )
        declared = Preregistration(
            fingerprint=fingerprint,
            declared_at=_now(),
            selection_rule=rule,
            seal_digest=seal_digest,
        )
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO preregistration (fingerprint, declared_at, selection_rule, "
                "seal_digest) VALUES (?, ?, ?, ?)",
                (
                    declared.fingerprint,
                    declared.declared_at,
                    declared.selection_rule,
                    declared.seal_digest,
                ),
            )
        return declared

    def preregistration(self, fingerprint: str) -> Preregistration | None:
        row = self._conn.execute(
            "SELECT * FROM preregistration WHERE fingerprint = ?", (fingerprint,)
        ).fetchone()
        if row is None:
            return None
        return Preregistration(
            fingerprint=row["fingerprint"],
            declared_at=row["declared_at"],
            selection_rule=row["selection_rule"],
            seal_digest=row["seal_digest"],
        )

    def preregistrations(self) -> list[Preregistration]:
        rows = self._conn.execute(
            "SELECT fingerprint FROM preregistration ORDER BY declared_at"
        ).fetchall()
        declared = [self.preregistration(r["fingerprint"]) for r in rows]
        return [d for d in declared if d is not None]

    def record_sealed_run(
        self, fingerprint: str, *, result: dict[str, Any], at: str | None = None
    ) -> str:
        """Spend the seal on one pre-registered candidate. Exactly once.

        A second call raises rather than overwriting. That is the entire point:
        once the first result can be discarded, "we re-ran it" and "we re-ran it
        until it worked" look the same from outside.
        """
        if self.preregistration(fingerprint) is None:
            raise PreregistrationError(
                f"{fingerprint} was never pre-registered; a sealed run on an "
                "undeclared candidate measures nothing"
            )
        row = self._conn.execute(
            "SELECT sealed_run_at FROM strategies WHERE fingerprint = ?", (fingerprint,)
        ).fetchone()
        if row is None:
            raise PreregistrationError(f"no strategy {fingerprint!r}")
        if row["sealed_run_at"]:
            raise PreregistrationError(
                f"{fingerprint}: the seal was spent on {row['sealed_run_at']} and "
                "there is no second one"
            )
        stamp = at or _now()
        with self._tx() as conn:
            conn.execute(
                "UPDATE strategies SET sealed_run_at = ?, sealed_result = ?, "
                "updated_at = ? WHERE fingerprint = ?",
                (stamp, _json(result), stamp, fingerprint),
            )
        return stamp

    def sealed_run(self, fingerprint: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT sealed_run_at, sealed_result FROM strategies WHERE fingerprint = ?",
            (fingerprint,),
        ).fetchone()
        if row is None or not row["sealed_run_at"]:
            return None
        return {
            "sealed_run_at": row["sealed_run_at"],
            "result": json.loads(row["sealed_result"]) if row["sealed_result"] else {},
        }

    # ----------------------------------------------------------------- #
    # The handoff
    # ----------------------------------------------------------------- #

    def record_target_book(
        self, fingerprint: str, *, as_of: str, path: str, digest: str, book: dict[str, Any]
    ) -> int:
        """Record a target book against the strategy that produced it.

        Appended, never replaced. Two books for the same session are two handoffs
        and the record should say so -- if the second differs from the first,
        something between the spec and the artefact changed, and overwriting
        would be the one thing that hides it.

        Refused for an unknown fingerprint. A book whose strategy is not in the
        registry cannot be traced back to a hypothesis, which is the only reason
        this table exists.
        """
        row = self._conn.execute(
            "SELECT 1 FROM strategies WHERE fingerprint = ?", (fingerprint,)
        ).fetchone()
        if row is None:
            raise ValueError(
                f"no strategy {fingerprint!r}: a target book that cannot be traced "
                "back to a registered hypothesis records nothing"
            )
        with self._tx() as conn:
            cursor = conn.execute(
                "INSERT INTO target_books (fingerprint, as_of, path, digest, book, "
                "created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (fingerprint, as_of, path, digest, _json(book), _now()),
            )
            return int(cursor.lastrowid or 0)

    def target_books(
        self, fingerprint: str | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        """Books handed off, newest first."""
        if fingerprint:
            rows = self._conn.execute(
                "SELECT * FROM target_books WHERE fingerprint = ? "
                "ORDER BY created_at DESC, id DESC LIMIT ?",
                (fingerprint, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM target_books ORDER BY created_at DESC, id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            record = dict(row)
            record["book"] = json.loads(row["book"]) if row["book"] else {}
            out.append(record)
        return out

    def sealed_looks(self) -> int:
        """How many candidates the embargoed window has screened so far.

        The one-shot rule is per candidate, not per window: a genuinely new
        hypothesis gets its own sealed run, which is what makes an ongoing
        research loop possible at all. What that costs is multiplicity — the
        survivor of a seven-way screen is a weaker claim than the survivor of a
        one-way screen — and this is the denominator that says which claim is
        being made. Nothing here refuses the seventh look; counting it is the
        defence.
        """
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM strategies WHERE sealed_run_at IS NOT NULL"
        ).fetchone()
        return int(row["n"]) if row else 0

    def sealed_look(self, fingerprint: str) -> int | None:
        """Which look this candidate was, 1-based. ``None`` if its seal is unspent.

        Derived from the order the seals were spent in rather than stored, so it
        cannot drift out of step with the runs it counts.
        """
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM strategies WHERE sealed_run_at IS NOT NULL "
            "AND sealed_run_at <= (SELECT sealed_run_at FROM strategies WHERE "
            "fingerprint = ?)",
            (fingerprint,),
        ).fetchone()
        ordinal = int(row["n"]) if row else 0
        return ordinal or None

    def ancestry_taint(self, fingerprint: str) -> TaintReport:
        """Did anything that produced this candidate read the embargoed years?

        The unit of contamination is the *campaign*, not the backtest. A process
        that read an embargoed bar while evaluating hypothesis 12 was still
        contaminated when it evaluated hypothesis 13, so the ancestry is every
        experiment on this fingerprint plus every sibling sharing a ``run_id``
        with one of them.
        """
        rows = self._conn.execute(
            "SELECT id, seal FROM experiments WHERE fingerprint = ?", (fingerprint,)
        ).fetchall()
        campaigns: list[str] = []
        for row in rows:
            run_id = _run_id(row["seal"])
            if run_id and run_id not in campaigns:
                campaigns.append(run_id)

        ancestry: dict[int, str | None] = {int(r["id"]): r["seal"] for r in rows}
        if campaigns:
            # No JSON index on the column, and the ledger is small enough that
            # one scan is cheaper than maintaining one.
            for row in self._conn.execute("SELECT id, seal FROM experiments").fetchall():
                if _run_id(row["seal"]) in campaigns:
                    ancestry[int(row["id"])] = row["seal"]

        tainted: list[int] = []
        unrecorded = 0
        for experiment_id, blob in sorted(ancestry.items()):
            certificate = _certificate(blob)
            if certificate is None:
                unrecorded += 1
                continue
            if certificate.get("tainted"):
                tainted.append(experiment_id)

        return TaintReport(
            fingerprint=fingerprint,
            experiments=len(ancestry),
            tainted=tuple(tainted),
            unrecorded=unrecorded,
            campaigns=tuple(campaigns),
        )

    # ----------------------------------------------------------------- #
    # Experiments
    # ----------------------------------------------------------------- #

    def record_experiment(self, record: ExperimentRecord) -> int:
        with self._tx() as conn:
            cursor = conn.execute(
                "INSERT INTO experiments (fingerprint, strategy_name, hypothesis, symbols, "
                "timeframe, data_start, data_end, dataset_version, train_metrics, oos_metrics, "
                "robustness, overfitting, evaluation, verdict, score, backtests_run, llm_model, "
                "prompt_hash, code_hash, error, seal, costs, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    record.fingerprint,
                    record.strategy_name,
                    record.hypothesis,
                    ",".join(record.symbols),
                    record.timeframe,
                    record.data_start,
                    record.data_end,
                    record.dataset_version,
                    _json(record.train_metrics),
                    _json(record.oos_metrics),
                    _json(record.robustness),
                    _json(record.overfitting),
                    _json(record.evaluation),
                    record.verdict,
                    record.score,
                    record.backtests_run,
                    record.llm_model,
                    record.prompt_hash,
                    record.code_hash,
                    record.error,
                    _json(record.seal),
                    _json(record.costs),
                    record.created_at,
                ),
            )
        return int(cursor.lastrowid or 0)

    def experiments(self, limit: int = 50, fingerprint: str | None = None) -> list[dict[str, Any]]:
        if fingerprint:
            rows = self._conn.execute(
                "SELECT * FROM experiments WHERE fingerprint = ? ORDER BY id DESC LIMIT ?",
                (fingerprint, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM experiments ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def total_backtests(self) -> int:
        """Every backtest ever run against this database.

        This is the *compute* spent, not the multiple-comparisons denominator --
        see :meth:`distinct_hypotheses` for that. It grows monotonically and is
        never reset.
        """
        row = self._conn.execute(
            "SELECT COALESCE(SUM(backtests_run), 0) AS total FROM experiments"
        ).fetchone()
        return int(row["total"])

    def distinct_hypotheses(self) -> int:
        """Distinct rules ever evaluated: the multiple-comparisons denominator.

        Not :meth:`total_backtests`. One hypothesis costs around 57 backtests
        here -- an in-sample run, a frictionless one, ten walk-forward folds,
        forty parameter perturbations, one per symbol for asset robustness --
        and none of those perturbations is a hypothesis anyone chose between.
        They are diagnostics on a rule already selected. Counting them inflated
        N by a factor of 57 and the deflation term by about a third.

        Failed attempts count, and that is the point: forgetting the ones that
        went nowhere is the mechanism by which a search looks luckier than it
        was. Re-evaluating one rule does not count twice -- rediscovering your
        own last idea is not a fresh place to have got lucky.
        """
        row = self._conn.execute(
            "SELECT COUNT(DISTINCT fingerprint) AS total FROM experiments"
        ).fetchone()
        return int(row["total"])

    def has_tried(self, fingerprint: str) -> bool:
        """Whether this exact rule has been evaluated before.

        The research loop checks this before spending a backtest, which is what
        stops an LLM from rediscovering its own last idea for twenty iterations.
        """
        row = self._conn.execute(
            "SELECT 1 FROM experiments WHERE fingerprint = ? LIMIT 1", (fingerprint,)
        ).fetchone()
        return row is not None

    def memory(self, limit: int = 20) -> list[dict[str, Any]]:
        """Compact history for the LLM prompt (architecture section 28).

        Deliberately small and deliberately includes rejects: the failures are
        the part that stops the next hypothesis repeating the last one.
        """
        rows = self._conn.execute(
            "SELECT strategy_name, hypothesis, verdict, score, oos_metrics, overfitting, error "
            "FROM experiments ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            oos = json.loads(row["oos_metrics"]) if row["oos_metrics"] else {}
            over = json.loads(row["overfitting"]) if row["overfitting"] else {}
            out.append(
                {
                    "name": row["strategy_name"],
                    "hypothesis": row["hypothesis"],
                    "verdict": row["verdict"] or ("ERROR" if row["error"] else "UNKNOWN"),
                    "score": row["score"],
                    "oos_sharpe": oos.get("sharpe"),
                    "oos_trades": oos.get("num_trades"),
                    "overfitting": over.get("verdict"),
                    "error": row["error"],
                }
            )
        return out


def _certificate(blob: str | None) -> dict[str, Any] | None:
    """A stored seal certificate, or None when there is nothing to read.

    Unparseable is treated the same as absent. A certificate nobody can read is
    not evidence, and guessing at one would be the only dishonest option here.
    """
    if not blob:
        return None
    try:
        loaded = json.loads(blob)
    except (TypeError, ValueError):
        return None
    return loaded if isinstance(loaded, dict) else None


def _run_id(blob: str | None) -> str | None:
    """The campaign a stored certificate belongs to, or None if it has none."""
    certificate = _certificate(blob)
    if certificate is None:
        return None
    run_id = certificate.get("run_id")
    return str(run_id) if run_id else None


def _strategy_from_row(row: sqlite3.Row) -> StrategyRecord:
    return StrategyRecord(
        fingerprint=row["fingerprint"],
        name=row["name"],
        version=row["version"],
        parent=row["parent"],
        status=row["status"],
        hypothesis=row["hypothesis"],
        spec_yaml=row["spec_yaml"],
        score=row["score"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )
