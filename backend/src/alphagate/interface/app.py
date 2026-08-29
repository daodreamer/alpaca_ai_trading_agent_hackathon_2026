"""The dashboard — specs/06's artefact, made openable.

> "A judge who opens any fill and reads the reasoning that produced it —
> including the checks that nearly stopped it — is looking at Technology
> Implementation and Presentation at the same time."

That is the whole brief for this file. Three views:

* **the day** — every cycle, quiet ones included, because `NO_SETUP` and
  `DECLINED` are the majority and answering "why didn't it trade at 14:30?" is
  the thing the journal exists for;
* **the cycle** — the market read, the *whole menu* the model chose from, the
  rationale, and all thirteen checks with their numbers;
* **the JSON** — the same line off disk, for anyone who would rather read the
  record than a rendering of it.

**It is read-only, structurally.** This module imports the journal and nothing
else — no `McpSession`, no `submit`, no market data client. There is no code
path from a browser to an order, and on demo day that is worth more than any
feature it might otherwise have.

HTML is built in Python rather than with a template engine. Not a stylistic
preference: it keeps the dependency list at what `pyproject.toml` already
declares, and the whole surface is three pages.
"""

from __future__ import annotations

import html
import json
from collections.abc import Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from alphagate.interface.read import CheckView, CycleView, DayView, available_days, day_view
from alphagate.interface.status import STALE_AFTER, _age_of, read_status
from alphagate.journal import Journal

__all__ = ["build_app", "serve"]

STAGE_COLOURS = {
    "filled": "#3fb950",
    "submitted": "#58a6ff",
    "dry_run": "#8b949e",
    "vetoed": "#d29922",
    "rejected": "#f85149",
    "breached": "#f85149",
    "declined": "#6e7681",
    "no_setup": "#484f58",
    "no_candidates": "#484f58",
}


def build_app(journal_dir: Path) -> FastAPI:
    app = FastAPI(title="AlphaGate", docs_url=None, redoc_url=None)
    journal = Journal(directory=journal_dir)

    @app.get("/healthz")
    def healthz() -> JSONResponse:
        return JSONResponse({"ok": True, "journal": str(journal_dir)})

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        """The React dashboard, or the server-rendered day if it is not built.

        Both are real UIs. The SPA answers "what is happening now" as well as
        "what was decided"; the server-rendered pages answer only the second,
        and exist so the journal is readable from a checkout with nothing but
        Python installed. They stay reachable at `/day/...` either way.
        """
        if spa_is_built():
            return HTMLResponse((STATIC / "index.html").read_text(encoding="utf-8"))
        days = available_days(journal)
        if not days:
            return HTMLResponse(_empty_page(journal_dir))
        return HTMLResponse(_day_page(day_view(journal, days[0]), days))

    @app.get("/day/{day}", response_class=HTMLResponse)
    def day_page(day: str) -> HTMLResponse:
        return HTMLResponse(_day_page(day_view(journal, _parse(day)), available_days(journal)))

    @app.get("/cycle/{day}/{cycle_id}", response_class=HTMLResponse)
    def cycle_page(day: str, cycle_id: str) -> HTMLResponse:
        view = day_view(journal, _parse(day))
        for cycle in view.cycles:
            if cycle.cycle_id == cycle_id:
                return HTMLResponse(_cycle_page(cycle, view))
        raise HTTPException(status_code=404, detail=f"no cycle {cycle_id} on {day}")

    @app.get("/api/day/{day}")
    def api_day(day: str) -> JSONResponse:
        return JSONResponse(journal.read(_parse(day)))

    @app.get("/api/status")
    def api_status() -> JSONResponse:
        """What the agent is doing right now, plus how stale that is.

        Read from `status.json`, which the agent rewrites every slot. The
        dashboard never talks to a broker — see `live/status.py` for why that
        is a structural decision and not a shortcut.

        `age_seconds` is computed here rather than trusted from the file, so a
        stopped agent shows an ageing page instead of a confident one.
        """
        snapshot = read_status(journal_dir)
        if snapshot is None:
            return JSONResponse({"running": False, "snapshot": None})
        age = _age_of(snapshot.get("as_of"))
        return JSONResponse(
            {
                "running": age is not None and age < STALE_AFTER,
                "age_seconds": age,
                "stale_after": STALE_AFTER,
                "snapshot": snapshot,
            }
        )

    @app.get("/api/days")
    def api_days() -> JSONResponse:
        return JSONResponse([day.isoformat() for day in available_days(journal)])

    _mount_spa(app)
    return app


STATIC = Path(__file__).parent / "static"


def spa_is_built() -> bool:
    return (STATIC / "index.html").is_file()


def _mount_spa(app: FastAPI) -> None:
    """Serve the built React dashboard's assets, if it has been built.

    Mounted after every API route so nothing shadows `/api/*`, and skipped
    entirely when `static/` is absent. That absence is a supported state, not a
    broken one: a checkout that has never run `npm run build` still gets a
    working server and falls through to the server-rendered pages, which need
    no toolchain at all. A dashboard that refused to start because a frontend
    was not compiled would be a bad thing to discover at 09:20.
    """
    if not spa_is_built():
        return
    app.mount("/assets", StaticFiles(directory=STATIC / "assets"), name="assets")

    @app.get("/favicon.svg")
    def favicon() -> FileResponse:
        return FileResponse(STATIC / "favicon.svg", media_type="image/svg+xml")


def serve(*, journal_dir: Path, host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn

    print(f"AlphaGate dashboard — http://{host}:{port}  (journal: {journal_dir})")
    uvicorn.run(build_app(journal_dir), host=host, port=port, log_level="warning")


def _parse(day: str) -> date:
    try:
        return date.fromisoformat(day)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"{day!r} is not a date") from exc


# ------------------------------------------------------------------ #
# Rendering
# ------------------------------------------------------------------ #

_CSS = """
:root {
  --bg: #0d1117; --panel: #161b22; --line: #30363d; --text: #e6edf3;
  --muted: #8b949e; --accent: #58a6ff; --good: #3fb950; --warn: #d29922;
  --bad: #f85149;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text);
  font: 14px/1.5 ui-monospace, SFMono-Regular, "SF Mono", Menlo, monospace; }
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
header { padding: 20px 28px; border-bottom: 1px solid var(--line);
  display: flex; align-items: baseline; gap: 20px; flex-wrap: wrap; }
h1 { margin: 0; font-size: 18px; letter-spacing: .04em; }
h1 span { color: var(--muted); font-weight: 400; }
main { padding: 24px 28px; max-width: 1200px; }
.tiles { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 24px; }
.tile { background: var(--panel); border: 1px solid var(--line); border-radius: 6px;
  padding: 12px 16px; min-width: 110px; }
.tile b { display: block; font-size: 22px; font-weight: 600; }
.tile span { color: var(--muted); font-size: 12px; text-transform: uppercase;
  letter-spacing: .06em; }
table { width: 100%; border-collapse: collapse; }
th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--line);
  vertical-align: top; }
th { color: var(--muted); font-weight: 500; font-size: 12px;
  text-transform: uppercase; letter-spacing: .06em; }
tr:hover td { background: #161b2266; }
.tag { display: inline-block; padding: 1px 8px; border-radius: 10px;
  font-size: 11px; letter-spacing: .05em; text-transform: uppercase;
  border: 1px solid currentColor; }
.note { color: var(--muted); }
.panel { background: var(--panel); border: 1px solid var(--line);
  border-radius: 6px; padding: 16px 20px; margin-bottom: 20px; }
.panel h2 { margin: 0 0 12px; font-size: 13px; color: var(--muted);
  text-transform: uppercase; letter-spacing: .08em; font-weight: 500; }
.kv { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
  gap: 10px 24px; }
.kv div span { display: block; color: var(--muted); font-size: 11px;
  text-transform: uppercase; letter-spacing: .05em; }
.bar { height: 4px; background: #21262d; border-radius: 2px; overflow: hidden;
  margin-top: 4px; min-width: 60px; }
.bar i { display: block; height: 100%; }
.rationale { background: #0d1117; border-left: 2px solid var(--accent);
  padding: 10px 14px; color: #c9d1d9; white-space: pre-wrap; }
.warn { border-color: var(--warn); color: var(--warn); }
.chosen td { background: #1f6feb22; }
pre { background: #0d1117; border: 1px solid var(--line); border-radius: 6px;
  padding: 14px; overflow-x: auto; font-size: 12px; color: #c9d1d9; }
.days { display: flex; gap: 8px; flex-wrap: wrap; }
.days a { padding: 2px 10px; border: 1px solid var(--line); border-radius: 4px; }
.days a.on { background: var(--accent); color: #0d1117; border-color: var(--accent); }
"""


def _page(title: str, body: str) -> str:
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{html.escape(title)}</title><style>{_CSS}</style></head>"
        f"<body>{body}</body></html>"
    )


def _header(subtitle: str, days: Sequence[date] = (), current: date | None = None) -> str:
    links = "".join(
        f"<a href='/day/{day}' class='{'on' if day == current else ''}'>{day}</a>"
        for day in days
    )
    return (
        "<header><h1>AlphaGate <span>— an options agent that can be overruled</span></h1>"
        f"<div class='note'>{html.escape(subtitle)}</div>"
        f"<div class='days' style='margin-left:auto'>{links}</div></header>"
    )


def _empty_page(directory: Path) -> str:
    return _page(
        "AlphaGate",
        _header("no journal yet")
        + "<main><div class='panel'><h2>nothing journalled</h2>"
        f"<p>No <code>*.jsonl</code> files in <code>{html.escape(str(directory))}</code>.</p>"
        "<p class='note'>Run <code>python -m alphagate run --dry-run</code> to produce a day."
        "</p></div></main>",
    )


def _stage_tag(stage: str) -> str:
    colour = STAGE_COLOURS.get(stage, "#8b949e")
    return f"<span class='tag' style='color:{colour}'>{html.escape(stage)}</span>"


def _day_page(view: DayView, days: Sequence[date]) -> str:
    tiles = [
        ("cycles", str(len(view.cycles)), None),
        ("fills", str(view.fills), "var(--good)"),
        ("vetoes", str(view.vetoes), "var(--warn)"),
        ("quiet", str(view.quiet), "var(--muted)"),
        ("realised", f"{view.realised}", "var(--good)" if view.realised >= 0 else "var(--bad)"),
    ]
    tile_html = "".join(
        f"<div class='tile'><b style='color:{colour or 'inherit'}'>{html.escape(value)}</b>"
        f"<span>{label}</span></div>"
        for label, value, colour in tiles
    )

    warnings = ""
    if view.has_warnings:
        parts = []
        if view.duplicates:
            parts.append(
                f"duplicate cycle ids (decisions collapsed on read): "
                f"{html.escape(str(view.duplicates))}"
            )
        if view.orphans:
            parts.append(f"amendments with no record: {html.escape(str(view.orphans))}")
        warnings = (
            "<div class='panel' style='border-color:var(--warn)'>"
            "<h2 style='color:var(--warn)'>reconciliation warnings</h2>"
            + "".join(f"<div>{part}</div>" for part in parts)
            + "</div>"
        )

    rows = "".join(_day_row(cycle, view.day) for cycle in view.cycles)
    return _page(
        f"AlphaGate — {view.day}",
        _header(f"{view.day}", days, view.day)
        + f"<main><div class='tiles'>{tile_html}</div>{warnings}"
        "<table><thead><tr><th>time</th><th>cycle</th><th>stage</th>"
        "<th>structure</th><th>menu</th><th>why</th></tr></thead>"
        f"<tbody>{rows}</tbody></table></main>",
    )


def _day_row(cycle: CycleView, day: date) -> str:
    near = cycle.near_misses
    flag = (
        f" <span class='tag warn' title='closest check'>near {near[0].name}</span>"
        if near
        else ""
    )
    return (
        "<tr>"
        f"<td>{html.escape(cycle.as_of)}</td>"
        f"<td><a href='/cycle/{day}/{html.escape(cycle.cycle_id)}'>"
        f"{html.escape(cycle.underlying)}</a></td>"
        f"<td>{_stage_tag(cycle.stage)}{flag}</td>"
        f"<td>{html.escape(cycle.structure)}"
        + (f" ×{cycle.quantity}" if cycle.quantity else "")
        + "</td>"
        f"<td>{cycle.candidate_count}</td>"
        f"<td class='note'>{html.escape(cycle.note)}</td>"
        "</tr>"
    )


def _cycle_page(cycle: CycleView, view: DayView) -> str:
    read_panel = _kv_panel(
        "what the engines saw",
        [
            ("underlying", cycle.underlying),
            ("spot", cycle.spot),
            ("iv rank", cycle.iv_rank),
            ("trend", cycle.trend),
            ("time", cycle.as_of),
            ("stage", cycle.stage),
        ],
    )

    model_panel = (
        "<div class='panel'><h2>what the model said</h2>"
        + _kv(
            [
                ("model", cycle.model),
                ("prompt", cycle.prompt_version),
                ("chose", "—" if cycle.chosen_index is None else f"#{cycle.chosen_index}"),
                ("confidence", cycle.confidence + " (recorded, never acted on)"),
            ]
        )
        + (
            f"<div class='rationale'>{html.escape(cycle.rationale)}</div>"
            if cycle.rationale
            else "<div class='note'>no rationale — the model was not reached</div>"
        )
        + "</div>"
    )

    menu_panel = _menu_panel(cycle)
    checks_panel = _checks_panel(cycle.checks)

    trust_panel = (
        "<div class='panel'><h2>trust boundary — specs/06 D5</h2>"
        f"<div>{html.escape(cycle.trust)}</div>"
        + (
            "<div class='note' style='margin-top:8px'>untrusted regions: "
            + ", ".join(f"<code>{html.escape(p)}</code>" for p in cycle.untrusted_paths)
            + "</div>"
            if cycle.untrusted_paths
            else ""
        )
        + "</div>"
    )

    outcome_panel = ""
    if cycle.outcome_status or cycle.realised:
        outcome_panel = _kv_panel(
            "what happened (amendment — specs/06 D3)",
            [
                ("status", cycle.outcome_status or "—"),
                ("realised", cycle.realised or "still open"),
            ],
        )

    raw = html.escape(json.dumps(cycle.raw, indent=2, sort_keys=True))
    return _page(
        f"AlphaGate — {cycle.cycle_id}",
        _header(f"{cycle.cycle_id}", (view.day,), view.day)
        + "<main>"
        f"<p><a href='/day/{view.day}'>← {view.day}</a></p>"
        f"<div class='panel'><h2>outcome</h2>{_stage_tag(cycle.stage)} "
        f"<span class='note'>{html.escape(cycle.note)}</span></div>"
        + read_panel
        + model_panel
        + menu_panel
        + checks_panel
        + outcome_panel
        + trust_panel
        + f"<div class='panel'><h2>the journal line, verbatim</h2><pre>{raw}</pre></div>"
        "</main>",
    )


def _menu_panel(cycle: CycleView) -> str:
    """The whole menu — specs/06 D2. Without it the rationale is unfalsifiable."""
    candidates = cycle.raw.get("candidates")
    if not isinstance(candidates, Sequence) or not candidates:
        return (
            "<div class='panel'><h2>the menu</h2>"
            "<div class='note'>no candidates were built — nothing survived pricing, "
            "freshness, spread, DTE and sizing</div></div>"
        )
    rows = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        index = item.get("index")
        raw_risk = item.get("risk")
        risk: dict[str, Any] = raw_risk if isinstance(raw_risk, dict) else {}
        chosen = index == cycle.chosen_index
        rows.append(
            f"<tr class='{'chosen' if chosen else ''}'>"
            f"<td>{'▶ ' if chosen else ''}{html.escape(str(index))}</td>"
            f"<td>{html.escape(_label(item))}</td>"
            f"<td>{html.escape(str(item.get('quantity', '')))}</td>"
            f"<td>{html.escape(str(risk.get('max_profit', '—')))}</td>"
            f"<td>{html.escape(str(risk.get('max_loss', '—')))}</td>"
            f"<td>{html.escape(str(risk.get('days_to_expiry', '—')))}</td>"
            f"<td>{html.escape(str(risk.get('breakevens', '—')))}</td>"
            "</tr>"
        )
    return (
        f"<div class='panel'><h2>the menu — {len(rows)} structures the model chose between</h2>"
        "<table><thead><tr><th>#</th><th>structure</th><th>qty</th><th>max profit</th>"
        "<th>max loss</th><th>dte</th><th>breakeven</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _label(candidate: dict[str, Any]) -> str:
    structure = candidate.get("structure")
    if not isinstance(structure, dict):
        return "—"
    kind = str(structure.get("kind", "")).replace("_", " ")
    legs = structure.get("legs")
    strikes = []
    if isinstance(legs, Sequence):
        for leg in legs:
            if isinstance(leg, dict) and isinstance(leg.get("contract"), dict):
                strikes.append(str(leg["contract"].get("strike", "")))
    return f"{kind} {'/'.join(s.rstrip('0').rstrip('.') for s in strikes)}"


def _checks_panel(checks: Sequence[CheckView]) -> str:
    """Every check, passed and failed — specs/06 D2.

    Sorted by headroom so the near-misses sit at the top. A risk system that
    only ever shows the checks it failed looks arbitrary; one that shows how
    close the passes came looks like it is doing something.
    """
    if not checks:
        return (
            "<div class='panel'><h2>the gate</h2>"
            "<div class='note'>the cycle never reached the Gate</div></div>"
        )
    ordered = sorted(
        checks, key=lambda c: (c.passed, c.headroom if c.headroom is not None else 1.0)
    )
    rows = []
    for check in ordered:
        colour = "var(--bad)" if not check.passed else (
            "var(--warn)" if check.is_near_miss else "var(--good)"
        )
        bar = ""
        if check.headroom is not None:
            width = max(2, int((1 - check.headroom) * 100))
            bar = f"<div class='bar'><i style='width:{width}%;background:{colour}'></i></div>"
        rows.append(
            "<tr>"
            f"<td style='color:{colour}'>{'PASS' if check.passed else 'FAIL'}</td>"
            f"<td>{html.escape(check.name)}</td>"
            f"<td>{html.escape(check.observed)}{bar}</td>"
            f"<td>{html.escape(check.limit)}</td>"
            f"<td class='note'>{html.escape(check.detail)}</td>"
            "</tr>"
        )
    passed = sum(1 for check in checks if check.passed)
    return (
        f"<div class='panel'><h2>the gate — {passed}/{len(checks)} checks passed, "
        "tightest first</h2>"
        "<table><thead><tr><th></th><th>check</th><th>observed</th><th>limit</th>"
        "<th>detail</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _kv(pairs: Sequence[tuple[str, str]]) -> str:
    return (
        "<div class='kv'>"
        + "".join(
            f"<div><span>{html.escape(k)}</span>{html.escape(v)}</div>" for k, v in pairs
        )
        + "</div>"
    )


def _kv_panel(title: str, pairs: Sequence[tuple[str, str]]) -> str:
    return f"<div class='panel'><h2>{html.escape(title)}</h2>{_kv(pairs)}</div>"


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
