# The dashboard

The Options and Equity tabs of AlphaGate's dashboard. React 19 + TypeScript,
Vite, Tailwind v4, shadcn/ui.

It **builds into the Python package** rather than into a sibling `dist/`, so one
FastAPI process serves both the page and the API. That is one fewer thing to
start on demo day, and it removes the CORS question entirely.

```bash
npm install
npm run build      # -> ../backend/src/alphagate/interface/static/
```

Then serve it from the repository root:

```bash
uv run --directory backend python -m alphagate serve   # http://127.0.0.1:8000
```

## Developing

```bash
npm run dev        # hot reload on :5173, proxying /api to :8000
```

`npm run dev` needs the backend running in another terminal (`alphagate serve`),
because every panel reads the API. The proxy is in
[vite.config.ts](vite.config.ts).

```bash
npx eslint .       # or: npm run lint
npm run typecheck  # tsc --noEmit
npm run format     # prettier
```

Both must be green before a commit that touches this directory — it is the last
line of the Definition of Done in [CLAUDE.md](../CLAUDE.md) §7.

## The build is optional

With no `static/` directory the server still runs and falls back to
server-rendered journal pages that need no Node toolchain at all. A dashboard
that refused to start because a frontend was not compiled would be a bad thing
to discover at 09:20.

`static/` is generated, so it is **not** committed — see the root
[.gitignore](../.gitignore). A fresh clone runs `npm run build` once.

## What is here

| File | What it renders |
| --- | --- |
| [src/App.tsx](src/App.tsx) | The three tabs, and the poll loop behind them |
| [src/components/live-status.tsx](src/components/live-status.tsx) | **Options** — health, money, each open structure travelling between its stop and its target, room left before the Gate refuses its own next proposal |
| [src/components/equity-status.tsx](src/components/equity-status.tsx) | **Equity** — the strategy's provenance first, then target weight against held weight with each position's no-trade band, then today's orders |
| [src/components/journal.tsx](src/components/journal.tsx) | **Journal** — every cycle, quiet ones included; expand one for the market read, the model's rationale and all thirteen Gate checks, sorted tightest-first |
| [src/components/option-book-card.tsx](src/components/option-book-card.tsx) | The pinned option rule: structure, entry condition, deltas, DTE window, and what the sealed window did and did not establish |
| [src/components/sleeves-overview.tsx](src/components/sleeves-overview.tsx) | The $90k/$10k split and each sleeve's own budget |
| [src/components/ui/](src/components/ui/) | shadcn/ui primitives. `npx shadcn@latest add <name>` puts new ones here |

## The API it reads

Served by [interface/app.py](../backend/src/alphagate/interface/app.py). All
read-only, all sourced from files the agent writes:

```
GET /api/status              the options sleeve, right now
GET /api/equity/status       the equity book, right now
GET /api/sleeves             the two allocations and their budgets
GET /api/option-book         the pinned option rule and its provenance
GET /api/days                which days have a journal
GET /api/day/{day}           one day of options cycles
GET /api/equity/day/{day}    one day of equity passes
```

**This page cannot trade, structurally.** `alphagate.interface` imports the
journal and nothing else — no MCP session, no market data client, no
`alphagate.live` — so there is no code path from a browser to an order. That is
guard 8 in
[tests/test_boundaries.py](../backend/tests/test_boundaries.py), enforced rather
than intended.

It also means the page learns the live book **from a file**, which is what lets
it fail honestly: if the agent stops, `journal/status.json` stops being
rewritten and the page says *not running* instead of showing a stale book with a
confident face.
