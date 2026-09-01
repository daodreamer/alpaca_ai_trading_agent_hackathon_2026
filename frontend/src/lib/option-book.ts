/**
 * The pinned option book's provenance — specs/07 D1, specs/07 D8.
 *
 * Two rules from `status.ts` and `equity.ts` hold here too. Money
 * (`risk_per_trade`) arrives as a string and stays one until rendered. The
 * sealed-run numbers are `number` because they are `float` estimates — an
 * alpha is not an amount anybody holds — and they are rendered, never acted
 * on: nothing on this page sizes a position by a `t`.
 *
 * **The non-negotiable rule lives in `optionSealedVerdict`.** This sealed
 * window measured `t=+1.11` against a bar of `1.96`: not significant, and
 * explicitly *not refuted* either — `SealedRun.refuted` and `.can_confirm` are
 * two separate booleans upstream because "was not refuted" and "was
 * confirmed" are different claims and only the first is available
 * (specs/10 D8). This module must never produce the word "confirmed",
 * "validated" or "passed" for this artefact, and no caller may attach a green
 * "pass" tone to whatever this returns — see `option-book-card.tsx`.
 *
 * **`risk_per_trade` never sizes a live trade, and this file must not imply it
 * does.** It is the fraction the *research* ran at — specs/10 D8a's 2% of the
 * $100,000 the sealed run was sized against. What actually binds a live order
 * is `agent/sizing.py` reading `SLEEVE_LIMITS.max_trade_loss(equity)` against
 * `OPTIONS_SLEEVE_ALLOCATION`, a different module that never consults the
 * book's own fraction. `live_trade_budget` and `live_trade_budget_pct` are
 * that number, computed on the backend from `risk/limits.py` rather than from
 * anything this file could hardcode — see `researchedSizingLabel` and
 * `liveSizingLabel` below, which render the two facts separately on purpose.
 */

export type OptionSealedRun = {
  strategy_return: number
  strategy_sharpe: number
  benchmark_sharpe: number
  max_drawdown: number
  trades: number
  observations: number
  alpha: number
  beta: number
  t_alpha: number
  significance_bar: number
  is_significant: boolean
  refuted: boolean
  can_confirm: boolean
  first_session: string
  last_session: string
  looks: number
  note: string
}

export type OptionRule = {
  structure: string
  entry_expression: string
  dte_target: number
  dte_tolerance: number
  anchor_delta: number
  anchor_tolerance: number
  width_delta: number
  min_sessions_between_entries: number
  /** The fraction the *research* ran at — specs/10 D8a. Never consulted by a
   * live trade; see the module docstring. */
  risk_per_trade: string
  max_concurrent: number
  /** The live options sleeve's allocation, from `OPTIONS_SLEEVE_ALLOCATION`. */
  sleeve_allocation: string
  /** `SLEEVE_LIMITS.max_trade_loss_pct` — what `agent/sizing.py` actually uses. */
  live_trade_budget_pct: string
  /** `sleeve_allocation * live_trade_budget_pct`, in dollars — the number that
   * binds a live order. */
  live_trade_budget: string
}

export type OptionBookAvailable = {
  available: true
  path: string | null
  fingerprint: string
  name: string
  version: number
  as_of: string
  generated_at: string
  dataset_version: string
  status: string
  hypothesis: string
  selection_rule: string
  distinct_hypotheses: number
  campaign_hypotheses: number
  exit_convention: string
  underlying: string
  rule: OptionRule
  sealed: OptionSealedRun
  can_refute_not_confirm: string
}

export type OptionBookUnavailable = {
  available: false
  path: string | null
  can_refute_not_confirm?: string
  reasons: string[]
}

export type OptionBookResponse = OptionBookAvailable | OptionBookUnavailable

/**
 * How the sealed run reads, in words that keep the two claims apart.
 *
 * Mirrors `equity.ts`'s `sealedVerdict` exactly, because the underlying
 * artefact (`SealedOptionRun` / `SealedRun`) makes the same distinction on
 * both sleeves: a rule that has not been refuted has not thereby been
 * confirmed, and this string is not allowed to blur that.
 */
export function optionSealedVerdict(sealed: OptionSealedRun): string {
  if (sealed.refuted) return "refuted by the sealed window"
  const t = `${sealed.t_alpha >= 0 ? "+" : ""}${sealed.t_alpha.toFixed(2)}`
  return sealed.is_significant
    ? `not refuted · t=${t} clears the bar at ${sealed.looks} look${sealed.looks === 1 ? "" : "s"}`
    : `not refuted · t=${t} does not clear the bar`
}

export function dteRange(rule: OptionRule): string {
  return `${rule.dte_target} days ± ${rule.dte_tolerance}`
}

export function anchorRange(rule: OptionRule): string {
  return `${rule.anchor_delta.toFixed(2)}Δ ± ${rule.anchor_tolerance.toFixed(2)}`
}

export function cadenceLabel(rule: OptionRule): string {
  return rule.min_sessions_between_entries === 1
    ? "at most one entry per session"
    : `at most one entry per ${rule.min_sessions_between_entries} sessions`
}

/**
 * What the research ran at — a fact about the $100,000 the sealed run was
 * sized against (specs/10 D8a), not about this account's $10,000 sleeve.
 * Deliberately does not mention dollars: `risk_per_trade` is a fraction of an
 * account this executor does not hold, and stating a dollar figure from it
 * would imply a size it was never used to compute.
 */
export function researchedSizingLabel(rule: OptionRule): string {
  const pct = (Number(rule.risk_per_trade) * 100).toFixed(1)
  return `researched at ${pct}% of equity per trade (specs/10 D8a)`
}

/**
 * What actually binds a live order. Every number here comes from the backend's
 * `risk/limits.py` via `/api/option-book`, not from `rule.risk_per_trade` —
 * `agent/sizing.py` never reads the book's own fraction. See the module
 * docstring for why the two are kept apart rather than rendered as one
 * sentence.
 */
export function liveSizingLabel(rule: OptionRule): string {
  const pct = (Number(rule.live_trade_budget_pct) * 100).toFixed(0)
  const allocation = Number(rule.sleeve_allocation).toLocaleString(undefined, {
    maximumFractionDigits: 0,
  })
  const budget = Number(rule.live_trade_budget).toLocaleString(undefined, {
    maximumFractionDigits: 0,
  })
  return `$${budget} per trade live — ${pct}% of the $${allocation} sleeve, ${rule.max_concurrent} concurrent max`
}
