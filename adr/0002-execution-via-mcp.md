# ADR 0002 — Orders go through Alpaca's MCP server; market data does not

- Status: accepted
- Date: 2026-08-26 (pre-competition)

## Context

The competition requires Alpaca's Trading API **and** either its MCP server or
its CLI ([specs/00-brief.md](../specs/00-brief.md)). Three surfaces exist, and
they were each probed against the paper account before this decision:

| Surface | Version | Probe result |
| --- | --- | --- |
| Trading REST API | stable | `POST /v2/orders` with `order_class: mleg` → 200, both legs accepted |
| MCP server | **2.3.0** | 72 tools over stdio, driven programmatically from Python; `place_option_order` supports multi-leg |
| CLI | **0.0.13**, "Alpha Preview" | Ships a Windows binary. README: *"Commands, flags, and output formats may change or be removed without notice. Do not depend on current behavior in production workflows."* |

## Decision

**D1 — Orders go through the MCP server.** `alphagate.execution` submits and
cancels through `place_option_order` / `cancel_order_by_id`, not through raw
REST. The requirement is then satisfied by the part of the system that actually
matters, rather than by a decorative call somewhere off the critical path.

**D2 — Market data does not.** Bars, chains and snapshots are read through a
direct REST adapter. Two reasons, both load-bearing:

- The backtest and the live agent must run the same code path with only the
  clock differing (specs/01 Rule 4). A recorded-payload transport seam gives us
  that; an MCP session does not replay.
- MCP wraps every response in an `_alpaca_mcp_security` envelope. That envelope
  is genuinely useful on the write path — it is where model-facing output lives —
  and pure overhead on a bar fetch.

**D3 — Operations run beside the critical path, never on it.** Three jobs need
a surface that is not order submission: pre-open connectivity, end-of-day
reconciliation, and a manual kill switch. At 0.0.13 with an explicit
no-stability warning, a renamed CLI flag mid-competition must not be able to
break any of them, let alone break submission.

**As built, none of the three use the CLI.** They are Python, in
`alphagate.live`, over the same MCP seam as everything else:

| Job | Surface |
| --- | --- |
| Pre-open connectivity | `python -m alphagate preflight` — checks the four hard gates against the live account |
| End-of-day reconciliation | `journal.reconcile`, driven each slot by the runner (specs/06 D7) |
| Kill switch | Latched in `journal/state.json`, read by the Gate (specs/03 D4); cleared by hand |

The decision the CLI was chosen for still holds — operations must not be able
to break submission — but it is enforced by layering rather than by using a
separate tool. Keeping the CLI as a documented dependency we never invoke would
have been a dependency nobody tested.

**D4 — The MCP client sits behind a transport seam.** `alphagate.execution`
depends on a `Protocol`, not on `fastmcp`. A fake implementation replays recorded
tool responses, so execution has offline tests. This mirrors the seam
`alphagate.core` already inherits from upstream.

## Consequences

- The one hard gate we cannot verify by reading docs is verified: the MCP server
  runs, lists 72 tools, and answers `get_account_info`, `get_clock` and
  `get_orders` against the paper account from a plain Python process.
- `fastmcp>=3.4.7` and the `uvx alpaca-mcp-server` subprocess become runtime
  dependencies of the live agent, but not of the tests or the backtest.
- Latency is worse than raw REST by one stdio round trip. Irrelevant at our
  trade frequency (specs/00: ~30 fills over four days, 7–60 DTE structures).
