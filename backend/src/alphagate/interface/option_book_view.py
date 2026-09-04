"""The option book's provenance, for the dashboard — specs/07 D1, specs/07 D8.

`ai_quant_researcher` writes an option book; `agent/option_book.py` is the one
module that may read one and turn it into something the executor can act on
(CLAUDE.md §2). This module does neither of those things — it never sizes a
position and it is not on the path that decides what the agent does. It exists
so the dashboard can show *the same artefact the executor is pinned to*,
without the interface package importing `alphagate.live` to get at it.

**Duplicated resolution, on purpose.** `interface/status.py` re-declares what it
needs from a snapshot rather than importing `alphagate.live.status`'s dataclass,
because importing `alphagate.live` would pull in the module that owns the MCP
session and delete the guard in `tests/test_boundaries.py`. The same argument
holds for finding the book on disk: `alphagate.live.equity.find_latest_book`
and `execution.load_env_file` both do part of this job already, and both live
in packages this one may not import. So the fifteen lines are copied rather
than shared — the cost of the guarantee, not an oversight.

**A book that fails to load is shown, not hidden.** specs/07 D8 makes the
refusal itself the interesting fact: a rule whose sealed window was refuted, or
whose registry status has not earned a paper position, must not be executed —
and a dashboard that swallowed that into an empty panel would erase the one
thing a judge should see working. `resolve_pinned_option_book` never raises;
`UnusableOptionBook`'s reasons travel into `OptionBookView.reasons` instead.

**The honesty requirement.** `sealed_window_can_refute_not_confirm` is a raw
provenance string, not a field `SealedOptionRun` carries, because
`agent/option_book.py` is owned by another module in this codebase's division
of labour and this file must not extend it. It is read straight off the parsed
payload here, alongside the validated `OptionBook`, so the dashboard can render
the artefact's own sentence rather than paraphrase it (CLAUDE.md's non-negotiable
rule about specs/07 D8: never render this as validated, passed, or confirmed).

**`rule.risk_per_trade` is not what sizes a live trade, and the wire shape says
so.** It is the fraction the *research* ran at — specs/10 D8a's 2% of the
$100,000 the sealed run was sized against — carried for the record. Live
sizing is `agent/sizing.py` reading `SLEEVE_LIMITS.max_trade_loss(equity)`
against `OPTIONS_SLEEVE_ALLOCATION`, a different number from a different
module that never consults the book's fraction. `option_book_to_json` adds
`sleeve_allocation`, `live_trade_budget_pct` and `live_trade_budget` alongside
`risk_per_trade` precisely so a dashboard cannot collapse the two into one
sentence — they happen to agree in dollars (that is the point of
`OPTIONS_SLEEVE_ALLOCATION`'s own docstring), but agreeing is a fact about the
configuration, not a mechanism this module may imply exists.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
from typing import Any

from alphagate.agent.option_book import OptionBook, UnusableOptionBook, load_option_book
from alphagate.risk.limits import OPTIONS_SLEEVE_ALLOCATION, SLEEVE_LIMITS

__all__ = [
    "OPTION_FINGERPRINT_VAR",
    "OptionBookView",
    "default_option_books_dir",
    "find_latest_option_book",
    "option_book_to_json",
    "pinned_option_fingerprint",
    "resolve_pinned_option_book",
]

OPTION_FINGERPRINT_VAR = "ALPHAGATE_OPTION_FINGERPRINT"
"""specs/07 D1: the pin is the only checkable meaning of "this executes the
rule that was sealed". No default, matching `agent/option_book.py`'s own
docstring — a fallback here would let the dashboard show a book nobody pinned."""

OPTION_BOOKS_DIR_VAR = "ALPHAGATE_OPTION_BOOKS"


def _repo_root() -> Path:
    """The directory holding `specs/`, walking up from this file.

    Copied from `alphagate.live.cli._repo_root` rather than imported, for the
    same reason the rest of this module duplicates instead of importing: that
    function lives in a package this one may not depend on.
    """
    for parent in Path(__file__).resolve().parents:
        if (parent / "specs").is_dir():
            return parent
    return Path.cwd()


def default_option_books_dir() -> Path:
    """Where `aqr option-book` writes, absent an override.

    Mirrors `alphagate.live.cli.DEFAULT_TARGET_BOOKS`'s reasoning for the equity
    side: a path, not an import, so specs/09 D0's rule that neither project
    imports the other holds here too.
    """
    return _repo_root() / "ai_quant_researcher" / "runs" / "option_books"


def _env_file_value(name: str) -> str | None:
    """One value from `.env.local`, without importing `execution.load_env_file`.

    Matched line for line against that function's own parsing rules (no
    interpolation, no export, no shell) so the two never disagree about what a
    line means — but kept as a separate fifteen lines rather than a shared
    import, because importing `alphagate.execution` is exactly what
    `tests/test_boundaries.py` guard 8 forbids for this package.

    Reads one name and returns one value — never the whole file — so a secret
    sitting in the same file as this pin can never travel through this
    function even by accident (CLAUDE.md rule 10).
    """
    path = _repo_root() / ".env.local"
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        if key.strip() == name:
            cleaned = value.strip().strip('"').strip("'")
            return cleaned or None
    return None


def pinned_option_fingerprint() -> str:
    """The fingerprint `ALPHAGATE_OPTION_FINGERPRINT` pins, or `""`.

    Checked in the process environment first and in `.env.local` second. The
    dashboard is typically started as `python -m alphagate serve` with no env
    file sourced into the shell, so a lookup that only worked when the variable
    happened to be exported would work by accident rather than by design.
    """
    return os.environ.get(OPTION_FINGERPRINT_VAR) or _env_file_value(OPTION_FINGERPRINT_VAR) or ""


def default_option_books_directory_override() -> Path | None:
    raw = os.environ.get(OPTION_BOOKS_DIR_VAR) or _env_file_value(OPTION_BOOKS_DIR_VAR)
    return Path(raw) if raw else None


def find_latest_option_book(directory: Path, fingerprint: str) -> Path | None:
    """The newest artefact for one rule, by the session it names.

    Mirrors `alphagate.live.equity.find_latest_book` exactly: sorted by
    **filename**, not mtime. `aqr option-book` names its output
    `<name>-<fingerprint>-<as_of>.json`, so the lexicographic maximum is the
    latest session, whereas mtime would prefer whichever file was regenerated
    most recently — the wrong one on a re-run of an old date.
    """
    if not fingerprint or not directory.is_dir():
        return None
    candidates = sorted(directory.glob(f"*-{fingerprint}-*.json"))
    return candidates[-1] if candidates else None


@dataclass(frozen=True, slots=True)
class OptionBookView:
    """What the dashboard has to say about the pinned option book.

    Never a `TargetBook`-style success-only type: `available=False` is a real,
    common state (nothing pinned yet, no book written yet, a refuted rule) and
    the dashboard's job in that state is to say so plainly, not to 500.
    """

    available: bool
    path: str | None = None
    book: OptionBook | None = None
    can_refute_not_confirm: str = ""
    reasons: tuple[str, ...] = field(default_factory=tuple)


def resolve_pinned_option_book(
    *, books_dir: Path | None = None, fingerprint: str | None = None
) -> OptionBookView:
    """Find, read and validate the pinned option book. Never raises.

    Every failure mode becomes a value: no pin, no file, a corrupt file, or a
    book `load_option_book` refuses. specs/07 D8 is explicit that the refusal
    itself is worth showing, so `UnusableOptionBook`'s reasons are carried
    through in `OptionBookView.reasons` rather than swallowed into an empty
    panel that looks like a missing file.
    """
    pin = fingerprint if fingerprint is not None else pinned_option_fingerprint()
    if not pin:
        return OptionBookView(
            available=False,
            reasons=(f"no fingerprint pinned ({OPTION_FINGERPRINT_VAR} is not set)",),
        )

    directory = books_dir
    if directory is None:
        directory = default_option_books_directory_override() or default_option_books_dir()

    path = find_latest_option_book(directory, pin)
    if path is None:
        return OptionBookView(
            available=False,
            reasons=(f"no option book for {pin} in {directory}",),
        )

    try:
        raw_bytes = path.read_bytes()
        payload = json.loads(raw_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return OptionBookView(
            available=False,
            path=path.name,
            reasons=(f"{path.name} could not be read: {exc}",),
        )
    if not isinstance(payload, dict):
        return OptionBookView(
            available=False,
            path=path.name,
            reasons=(f"{path.name} is a {type(payload).__name__}, expected an object",),
        )

    note = _refute_confirm_note(payload)
    digest = sha256(raw_bytes).hexdigest()
    try:
        book = load_option_book(payload, pinned_fingerprint=pin, digest=digest)
    except UnusableOptionBook as exc:
        return OptionBookView(
            available=False,
            path=path.name,
            can_refute_not_confirm=note,
            reasons=tuple(str(exc).split("\n  - ")),
        )
    return OptionBookView(
        available=True, path=path.name, book=book, can_refute_not_confirm=note
    )


def _refute_confirm_note(payload: Mapping[str, Any]) -> str:
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        return ""
    return str(provenance.get("sealed_window_can_refute_not_confirm", ""))


def option_book_to_json(view: OptionBookView) -> dict[str, Any]:
    """The wire shape for `/api/option-book`.

    Money stays a string (`Decimal`, CLAUDE.md rule 4); alpha, beta and `t` stay
    JSON numbers because they are `float` estimates — the same split
    `equity_status.py` renders for the equity sleeve's own sealed run.
    """
    if not view.available or view.book is None:
        return {
            "available": False,
            "path": view.path,
            "can_refute_not_confirm": view.can_refute_not_confirm,
            "reasons": list(view.reasons),
        }
    book = view.book
    rule = book.rule
    sealed = book.sealed
    return {
        "available": True,
        "path": view.path,
        "fingerprint": book.fingerprint,
        "name": book.name,
        "version": book.version,
        "as_of": book.as_of.isoformat(),
        "generated_at": book.generated_at.isoformat(),
        "dataset_version": book.dataset_version,
        "status": book.status,
        "hypothesis": book.hypothesis,
        "selection_rule": book.selection_rule,
        "distinct_hypotheses": book.distinct_hypotheses,
        "campaign_hypotheses": book.campaign_hypotheses,
        "exit_convention": book.exit_convention,
        "underlying": str(book.underlying),
        "rule": {
            "structure": rule.structure,
            "entry_expression": rule.entry.expression,
            "dte_target": rule.dte_target,
            "dte_tolerance": rule.dte_tolerance,
            "anchor_delta": rule.anchor_delta,
            "anchor_tolerance": rule.anchor_tolerance,
            "width_delta": rule.width_delta,
            "min_sessions_between_entries": rule.min_sessions_between_entries,
            "risk_per_trade": _decimal_str(rule.risk_per_trade),
            "max_concurrent": rule.max_concurrent,
            # What actually binds live is a different number from a different
            # place: `agent/sizing.py` sizes every trade off
            # `SLEEVE_LIMITS.max_trade_loss(OPTIONS_SLEEVE_ALLOCATION)` and never
            # reads `risk_per_trade` above. The two agree in dollars on purpose —
            # `OPTIONS_SLEEVE_ALLOCATION`'s own docstring in `risk/limits.py`
            # explains that the sleeve was sized to $10,000 specifically so the
            # live budget lands on the fraction the sealed run measured — and
            # that agreement is worth showing rather than presenting the book's
            # own fraction as though it did the sizing itself.
            "sleeve_allocation": _decimal_str(OPTIONS_SLEEVE_ALLOCATION),
            "live_trade_budget_pct": _decimal_str(SLEEVE_LIMITS.max_trade_loss_pct),
            "live_trade_budget": _decimal_str(
                SLEEVE_LIMITS.max_trade_loss(OPTIONS_SLEEVE_ALLOCATION)
            ),
        },
        "sealed": {
            "strategy_return": sealed.strategy_return,
            "strategy_sharpe": sealed.strategy_sharpe,
            "benchmark_sharpe": sealed.benchmark_sharpe,
            "max_drawdown": sealed.max_drawdown,
            "trades": sealed.trades,
            "observations": sealed.observations,
            "alpha": sealed.alpha,
            "beta": sealed.beta,
            "t_alpha": sealed.t_alpha,
            "significance_bar": sealed.significance_bar,
            "is_significant": sealed.is_significant,
            "refuted": sealed.refuted,
            "can_confirm": sealed.can_confirm,
            "first_session": sealed.first_session,
            "last_session": sealed.last_session,
            "looks": sealed.looks,
            "note": sealed.note,
        },
        "can_refute_not_confirm": view.can_refute_not_confirm,
    }


def _decimal_str(value: Decimal) -> str:
    return format(value, "f")
