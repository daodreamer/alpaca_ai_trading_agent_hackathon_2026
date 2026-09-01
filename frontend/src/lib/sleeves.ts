/**
 * Both sleeves, measured apart — specs/03 D6.
 *
 * `interface/sleeves.py`'s module docstring explains why `equity` here can be
 * `null` even while a sleeve is `running`: it is a display-only figure this
 * route derives from the two agents' last published snapshots, not the number
 * either kill switch actually enforces, and it is honestly absent whenever it
 * cannot be derived (the options sleeve not reporting yet, for the equity
 * residual) rather than guessed at zero.
 *
 * The one property this whole file exists to make visible: `options.equity`
 * and `equity.equity` are never the same figure, and `options.max_drawdown_pct`
 * and `equity.max_drawdown_pct` are two different thresholds — 20% and 10% by
 * configuration, not by coincidence. A page that showed one blended number
 * would erase the reason the sleeve design exists.
 */

export type SleeveSummary = {
  name: string
  allocation: string
  running: boolean
  equity: string | null
  realised: string | null
  unrealised: string | null
  drawdown_pct: string | null
  max_drawdown_pct: string | null
  killswitch_tripped: boolean | null
  open_positions: number | null
  activity_today: number | null
  activity_label: string
  note: string
}

export type SleevesResponse = {
  options: SleeveSummary
  equity: SleeveSummary
}
