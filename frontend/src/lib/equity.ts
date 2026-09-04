/**
 * The shapes the equity agent serves — specs/09 D10.
 *
 * Two rules from `status.ts` hold here unchanged, and one is added.
 *
 * **Money arrives as a string and stays one until it is rendered.** The backend
 * is `Decimal` end to end and serialises to strings for that reason.
 *
 * **`null` means unmeasured, not zero.** A position with no mark has
 * `price: null`, and rendering that as `0` would show a total loss where the
 * truth is "the snapshot request missed one name".
 *
 * **The sealed-run numbers are `number`, and that is deliberate.** They are
 * `float` on the backend too, because they are estimates rather than money —
 * an alpha is not an amount anybody holds. They are also *rendered and never
 * acted on*: nothing on this page or behind it sizes a position by a `t`. The
 * sealed window decided whether the strategy may run at all, and that decision
 * was made before the process that wrote this file started.
 */

export type SealedRun = {
  return: number
  sharpe: number
  benchmark_sharpe: number
  max_drawdown: number
  trades: number
  observations: number
  alpha: number
  beta: number
  t_alpha: number
  information_ratio: number
  is_significant: boolean
  refuted: boolean
  can_confirm: boolean
  looks: number
  window: string
}

export type Strategy = {
  fingerprint: string
  name: string
  as_of: string
  digest: string
  status: string
  universe: string
  dataset_version: string
  hypothesis: string
  selection_rule: string
  distinct_hypotheses: number
  positions: number
  gross: string
  sealed: SealedRun
}

export type PositionLine = {
  symbol: string
  target_weight: string
  held_weight: string
  target_shares: string | null
  held_shares: string
  price: string | null
  price_age_seconds: number | null
  market_value: string
  drift: string
  threshold: string
  inside_band: boolean
  core: boolean
  tradeable: boolean
}

export type EquitySnapshot = {
  as_of: string
  session_day: string
  next_pass: string | null
  heartbeat_sequence: number

  strategy: Strategy | Record<string, never>

  equity: string
  cash: string
  buying_power: string
  session_change: string
  peak_equity: string | null
  drawdown_pct: string
  killswitch_tripped: boolean
  is_blocked: boolean

  gross_exposure: string
  positions_held: number
  positions_wanted: number
  drift_band_pct: string
  min_order_notional: string
  unpriced: string[]
  stale: string[]
  off_book: string[]

  orders_today: number
  turnover_today: string
  max_daily_orders: number
  max_daily_turnover: string
  max_position_pct: string
  max_drawdown_pct: string

  lines: PositionLine[]
  stage_counts: Record<string, number>
  note: string
}

export type EquityStatusResponse = {
  running: boolean
  age_seconds?: number | null
  stale_after?: number
  snapshot: EquitySnapshot | null
}

/** One order from a journalled rebalance pass. */
export type EquityOrder = {
  symbol: string
  side: string
  shares: string
  reference_price: string
  notional: string
  target_weight: string
  held_weight: string
  outcome: string
  verdict?: {
    checks?: EquityCheck[]
    reasons?: { check: string; detail: string }[]
    waived?: { check: string; detail: string }[]
  }
  submission?: { status?: string; client_order_id?: string } | null
}

export type EquityCheck = {
  name: string
  passed: boolean
  observed: string | null
  limit: string | null
  detail?: string
}

export type EquityCycle = {
  cycle_id: string
  kind?: string
  as_of?: string
  stage?: string
  note?: string
  equity?: string
  band_pct?: string
  turnover?: string
  strategy?: Strategy | Record<string, never>
  orders?: EquityOrder[]
  skipped?: { symbol: string; reason: string; detail: string; drift: string }[]
}

export function hasStrategy(
  strategy: Strategy | Record<string, never> | undefined,
): strategy is Strategy {
  return Boolean(strategy && "fingerprint" in strategy && strategy.fingerprint)
}

/**
 * What the equity agent is doing, in one word, for the header.
 *
 * A priority order rather than a checklist: whatever is worst is what the
 * header says. **Stale marks are not in it**, deliberately — an agent holding a
 * correct book on prices from last night is entirely healthy and entirely
 * unable to trade, and calling that unhealthy would cry wolf every evening. It
 * gets its own line instead.
 */
export type EquityHealth = {
  label: string
  tone: "ok" | "warn" | "bad" | "idle"
  detail: string
}

export function equityHealth(status: EquityStatusResponse): EquityHealth {
  const snapshot = status.snapshot
  if (!snapshot) {
    return {
      label: "not started",
      tone: "idle",
      detail: "The stock agent has never run, so there is nothing to show yet.",
    }
  }
  if (!status.running) {
    const age = status.age_seconds ?? 0
    const ago =
      age < 90 ? `${Math.round(age)} seconds ago` : `${Math.round(age / 60)} minutes ago`
    return {
      label: "stopped",
      tone: "bad",
      detail: `The agent last checked in ${ago} and has gone quiet. Your holdings are untouched, but nothing is being rebalanced until it is restarted.`,
    }
  }
  if (snapshot.killswitch_tripped) {
    return {
      label: "halted — kill switch",
      tone: "bad",
      detail:
        "Losses hit the safety limit, so the agent has stopped buying. Selling still works, so it can still reduce risk. It will not resume on its own.",
    }
  }
  if (snapshot.is_blocked) {
    return {
      label: "cannot trade",
      tone: "bad",
      detail: "The broker has frozen this account, so every order will be rejected.",
    }
  }
  if (!hasStrategy(snapshot.strategy)) {
    return {
      label: "no strategy loaded",
      tone: "warn",
      detail:
        "The agent has no list of what to hold, so it will not trade. Rebuild it with: python scripts/pipeline.py book",
    }
  }
  if (snapshot.unpriced.length > 0) {
    return {
      label: "running — some prices missing",
      tone: "warn",
      detail: `Trading normally, except for ${snapshot.unpriced.length} holding${
        snapshot.unpriced.length === 1 ? "" : "s"
      } the broker gave no price for. Those are left alone rather than traded on a guess.`,
    }
  }
  return {
    label: "running",
    tone: "ok",
    detail: "Holdings are being checked every 30 seconds against the target list.",
  }
}

/**
 * How the sealed run reads, in the words the researcher actually used.
 *
 * `can_confirm` is `False` by construction upstream, and the phrasing here
 * keeps that distinction rather than flattening it into "passed". Two years is
 * about 500 sessions, where the standard error of an annualised Sharpe is
 * around 0.7 — the window can refute a rule and cannot confirm one, and a
 * dashboard that said "confirmed" would be claiming something nobody measured.
 */
export function sealedVerdict(sealed: SealedRun): string {
  if (sealed.refuted) {
    return "Disproved on the held-back years — this strategy is not allowed to trade."
  }
  const t = sealed.t_alpha.toFixed(2)
  return sealed.is_significant
    ? `Survived a test on years of market data it had never seen, by a comfortable margin (score ${t}). That is the strongest verdict available: this test can disprove a strategy, never prove one.`
    : `Was not disproved on years of market data it had never seen, but the margin is too small to read much into (score ${t}). It runs because it survived, not because it was proven.`
}
