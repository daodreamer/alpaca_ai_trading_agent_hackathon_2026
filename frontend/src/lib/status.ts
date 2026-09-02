/**
 * The shapes the backend serves, and the small amount of arithmetic the page
 * is allowed to do on them.
 *
 * Two rules hold everywhere below.
 *
 * **Money arrives as a string and stays one until it is rendered.** The backend
 * is `Decimal` end to end (specs/01 Rule 3) and serialises to strings for that
 * reason; parsing every figure into a float on arrival would undo the whole
 * discipline at the last hop. `num()` exists for the two places that genuinely
 * need a number — a progress bar's width and a sort order — and nowhere else.
 *
 * **`null` means unmeasured, not zero.** The backend is careful about this: an
 * `iv_rank` nobody could compute, a position nobody could re-price, a fraction
 * of credit that names nothing for a debit structure. Rendering those as `0`
 * would turn "we don't know" into a confident and wrong number, so `fmt()`
 * renders an em dash instead.
 */

import type { EquityCycle } from "@/lib/equity"

export type PositionStatus = {
  cycle_id: string
  underlying: string
  structure: string
  quantity: number
  expiry: string
  days_to_expiry: number
  entry_premium: string
  mark: string
  unrealised: string
  fraction_of_credit: string | null
  rule: string
  detail: string
  to_target: string | null
  to_stop: string | null
  max_loss: string
}

export type Snapshot = {
  as_of: string
  session_day: string
  next_slot: string | null
  slot_sequence: number
  universe: string[]

  equity: string
  cash: string
  buying_power: string
  options_buying_power: string
  options_level: number
  can_trade_spreads: boolean
  is_blocked: boolean
  is_pattern_day_trader: boolean
  session_change: string

  peak_equity: string | null
  drawdown_pct: string
  killswitch_tripped: boolean
  fills_today: number
  open_structures: number
  open_risk: string

  max_open_structures: number
  max_daily_trades: number
  max_portfolio_risk: string
  max_drawdown_pct: string

  positions: PositionStatus[]
  unexplained: string[]

  profit_target: string
  stop_multiple: string
  min_dte: number

  stage_counts: Record<string, number>
  note: string
}

export type StatusResponse = {
  running: boolean
  age_seconds?: number | null
  stale_after?: number
  snapshot: Snapshot | null
}

export type JournalCycle = {
  cycle_id: string
  as_of?: string
  stage?: string
  note?: string
  read?: { underlying?: string; spot?: string; iv_rank?: string | null }
  choice?: { rationale?: string; candidate_index?: number | null }
  candidates?: unknown[]
  verdict?: { checks?: Check[] }
  proposal?: { quantity?: number; risk?: { max_loss?: string } }
  outcome?: { status?: string; realised_pl?: string | null }
  /**
   * Why nothing happened, computed once in Python (`CycleView.category` in
   * `interface/read.py`) and stamped onto this record by `/api/day/{day}`
   * (`day_records_with_category`). Absent on an equity record — `category` is
   * options' own decline taxonomy — so every reader here must treat a missing
   * value as "not classified" rather than guess.
   *
   * There is deliberately no TypeScript reimplementation of this
   * classification. Two independent copies of one judgement are two chances
   * for them to quietly disagree, which is exactly the failure this field
   * exists to rule out — see `interface/read.py`'s `day_records_with_category`
   * docstring.
   */
  category?: DeclineCategory
  category_label?: string
  category_detail?: string
}

export type Check = {
  name: string
  passed: boolean
  observed: string | null
  limit: string | null
  detail?: string
}

/**
 * One line of `/api/day/{day}` — which is a *day*, not a sleeve.
 *
 * Both agents journal into the same daily file (see `live/equity.py`'s
 * `CYCLE_KIND`: the question "what happened on 2026-09-02" is about the
 * account, not about which process asked), so this route hands back both and
 * the reader has to tell them apart.
 *
 * That is not a formatting nicety. Every field on `JournalCycle` is the
 * options sleeve's vocabulary — `iv_rank`, a menu of candidates, one structure,
 * one verdict — and an equity pass has none of them. Rendered as an options
 * cycle it does not come out mislabelled, it comes out invented: no spot, an
 * `iv_rank` of "unmeasured", a menu of zero, and no checks, which the card
 * words as "the cycle never reached the Gate" about a pass carrying twelve
 * passed checks per order.
 */
export type DayRecord = JournalCycle | EquityCycle

/** Which agent wrote this line. The `kind` the backend stamps, nothing else. */
export function isEquityPass(record: DayRecord): record is EquityCycle {
  return (record as EquityCycle).kind === "equity"
}

/** Parse for arithmetic only — never for display. See the module note. */
export function num(value: string | null | undefined): number {
  if (value === null || value === undefined || value === "") return 0
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : 0
}

/** Render a money string. Unmeasured stays unmeasured. */
export function fmt(value: string | null | undefined, digits = 2): string {
  if (value === null || value === undefined || value === "") return "—"
  const parsed = Number(value)
  if (!Number.isFinite(parsed)) return String(value)
  return parsed.toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  })
}

export function pct(value: string | null | undefined, digits = 1): string {
  if (value === null || value === undefined || value === "") return "—"
  const parsed = Number(value)
  if (!Number.isFinite(parsed)) return String(value)
  return `${(parsed * 100).toFixed(digits)}%`
}

export function signed(value: string | null | undefined): string {
  if (value === null || value === undefined || value === "") return "—"
  const parsed = Number(value)
  if (!Number.isFinite(parsed)) return String(value)
  return `${parsed >= 0 ? "+" : ""}${fmt(value)}`
}

export function clock(iso: string | null | undefined): string {
  if (!iso) return "—"
  const when = new Date(iso)
  if (Number.isNaN(when.getTime())) return iso
  return when.toLocaleTimeString(undefined, {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  })
}

/**
 * How long until the next slot, in words.
 *
 * "idle" and "idle, next at 15:05" are different amounts of reassurance, and
 * this is the difference.
 */
export function untilNext(iso: string | null): string {
  if (!iso) return "session over"
  const seconds = (new Date(iso).getTime() - Date.now()) / 1000
  if (Number.isNaN(seconds)) return "—"
  if (seconds <= 0) return "due now"
  const minutes = Math.floor(seconds / 60)
  return minutes >= 1 ? `in ${minutes}m` : `in ${Math.floor(seconds)}s`
}

/**
 * Why nothing happened, in a bucket a judge can tell apart from the rest.
 *
 * The classification itself lives in exactly one place: `CycleView.category`
 * in `backend/src/alphagate/interface/read.py`. This type exists only so the
 * value the backend already stamped onto `JournalCycle.category` (via
 * `day_records_with_category`) is a checked union rather than a bare string —
 * there is no `declineCategory()` here, on purpose. A second implementation of
 * one judgement, in a second language, is two chances for the two to quietly
 * disagree about why the agent declined, and that disagreement would not
 * announce itself: it would just make this page and the server-rendered
 * fallback tell a judge two different stories.
 */
export type DeclineCategory =
  | "traded"
  | "approved_not_sent"
  | "gate_veto"
  | "broker_rejected"
  | "model_declined"
  | "undecidable"
  | "no_setup"
  | "no_candidates"
  | "other"

/**
 * What the whole system is doing, in one word, for the header.
 *
 * The order is a priority order, not a checklist: a latched kill switch matters
 * more than an unexplained leg, and both matter more than whether the next slot
 * is due. Whatever is worst is what the header says.
 */
export type Health = {
  label: string
  tone: "ok" | "warn" | "bad" | "idle"
  detail: string
}

export function health(status: StatusResponse): Health {
  const snapshot = status.snapshot
  if (!snapshot) {
    return {
      label: "never run",
      tone: "idle",
      detail: "no status.json yet — run the agent once to produce one",
    }
  }
  if (!status.running) {
    const age = status.age_seconds ?? 0
    return {
      label: "not running",
      tone: "bad",
      detail: `last heartbeat ${Math.round(age / 60)} minutes ago`,
    }
  }
  if (snapshot.killswitch_tripped) {
    return {
      label: "kill switch latched",
      tone: "bad",
      detail: "opens are blocked until a human clears it",
    }
  }
  if (snapshot.is_blocked || !snapshot.can_trade_spreads) {
    return {
      label: "cannot trade",
      tone: "bad",
      detail: snapshot.is_blocked
        ? "the broker has blocked this account"
        : `options level ${snapshot.options_level}; spreads need 3`,
    }
  }
  if (snapshot.unexplained.length > 0) {
    return {
      label: "unexplained legs",
      tone: "warn",
      detail: `${snapshot.unexplained.length} broker legs the journal cannot account for`,
    }
  }
  return {
    label: "running",
    tone: "ok",
    detail: snapshot.note || `slot ${snapshot.slot_sequence}`,
  }
}
