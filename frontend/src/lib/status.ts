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
  /**
   * Cycle ids whose close order is already at the broker and still working.
   *
   * The agent skips these positions until the close settles, so without this
   * the page shows a position whose exit rule has fired and an agent doing
   * nothing about it — which is the picture of a bug, not of an order waiting
   * to fill. Optional because a snapshot written before this field existed is
   * still a snapshot.
   */
  closing?: string[]

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
  if (!iso) return "not until the market opens again"
  const seconds = (new Date(iso).getTime() - Date.now()) / 1000
  if (Number.isNaN(seconds)) return "—"
  if (seconds <= 0) return "any moment now"
  const minutes = Math.floor(seconds / 60)
  return minutes >= 1 ? `in ${minutes} min` : `in ${Math.floor(seconds)}s`
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
 * What each outcome name means, for someone who has not read the code.
 *
 * The backend writes these as short machine tokens (`no_setup`, `dry_run`), and
 * a badge reading `no_candidates` tells a reader nothing about whether the
 * system is healthy. `stageLabel` is what goes on the badge; `stageHint` is the
 * one-line explanation beside it, and every one of them answers the same
 * question: is this normal, or is something wrong?
 */
const STAGE_LABEL: Record<string, string> = {
  no_setup: "no opportunity",
  no_candidates: "nothing tradeable",
  declined: "model passed",
  dry_run: "rehearsal only",
  submitted: "order sent",
  filled: "order filled",
  rejected: "broker refused",
  vetoed: "blocked by risk checks",
  breached: "risk limit breached",
  closed: "position closed",
  error: "something went wrong",
  planned: "planned, not sent",
  no_trades: "nothing to do",
  no_marks: "no live prices",
  skipped: "skipped",
}

const STAGE_HINT: Record<string, string> = {
  no_setup: "The market did not meet the entry rule. Normal on most days.",
  no_candidates:
    "The entry rule fired but no actual contract was tradeable — stale prices, too wide a spread, or nothing in the right expiry.",
  declined: "Tradeable options existed and the model chose not to take them. Not an error.",
  dry_run: "A rehearsal — everything was worked out and deliberately not sent.",
  submitted: "An order went to the broker.",
  filled: "An order went to the broker and was executed.",
  rejected: "The broker turned an order down. Usually account permissions or buying power.",
  vetoed: "The risk checks stopped an order before it was sent. The safety layer working.",
  breached: "A safety limit was crossed. The agent stops rather than trading through it.",
  closed: "An existing position was closed.",
  error: "The cycle failed. Check the detail below and the terminal running the agent.",
  planned: "Orders were worked out and deliberately not sent — this was a rehearsal.",
  no_trades: "Everything already matches the target closely enough. Normal on most days.",
  no_marks:
    "The broker returned no usable prices, so nothing was traded. Almost always means the market is closed.",
  skipped: "Left alone on purpose.",
}

/** A badge label anyone can read. Unknown values pass through, tidied. */
export function stageLabel(stage: string): string {
  return STAGE_LABEL[stage] ?? stage.replace(/_/g, " ")
}

/** One line saying whether that outcome is a problem. `""` when there is nothing to add. */
export function stageHint(stage: string): string {
  return STAGE_HINT[stage] ?? ""
}

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
      label: "not started",
      tone: "idle",
      detail: "The options agent has never run, so there is nothing to show yet.",
    }
  }
  if (!status.running) {
    const minutes = Math.round((status.age_seconds ?? 0) / 60)
    const ago =
      minutes < 1 ? "under a minute ago" : minutes === 1 ? "a minute ago" : `${minutes} minutes ago`
    return {
      label: "stopped",
      tone: "bad",
      detail: `The agent last checked in ${ago} and has gone quiet. Nothing is being watched or traded until it is restarted.`,
    }
  }
  if (snapshot.killswitch_tripped) {
    return {
      label: "halted — kill switch",
      tone: "bad",
      detail:
        "Losses hit the safety limit, so the agent has stopped opening new positions. It will not resume on its own; a person has to clear it.",
    }
  }
  if (snapshot.is_blocked || !snapshot.can_trade_spreads) {
    return {
      label: "cannot trade",
      tone: "bad",
      detail: snapshot.is_blocked
        ? "The broker has frozen this account, so every order will be rejected."
        : `This account is approved for options level ${snapshot.options_level}, and the spreads this agent trades need level 3. Every order will be rejected until that is raised.`,
    }
  }
  if (snapshot.unexplained.length > 0) {
    return {
      label: "running — check positions",
      tone: "warn",
      detail: `Trading normally, but ${snapshot.unexplained.length} position${
        snapshot.unexplained.length === 1 ? " is" : "s are"
      } open at the broker that this agent did not place, so ${
        snapshot.unexplained.length === 1 ? "it is" : "they are"
      } not covered by its risk limits.`,
    }
  }
  return {
    label: "running",
    tone: "ok",
    detail:
      snapshot.note ||
      `Watching ${snapshot.universe.join(", ") || "the market"} and re-checking every 15 minutes.`,
  }
}
