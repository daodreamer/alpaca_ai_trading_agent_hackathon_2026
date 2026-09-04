/**
 * The journal tab — what it decided today, and why.
 *
 * Every cycle, quiet ones included: `no_setup` and `declined` are the majority
 * and they are the reason the journal exists. A view that listed only fills
 * would answer the easy question and leave "why didn't it trade at 14:30?"
 * exactly as unanswered as no journal at all.
 *
 * The Gate's checks are sorted tightest-first, so a check that passed with 4%
 * of its budget left sits above one that passed with 90%. That ordering is the
 * whole argument that the risk layer is load-bearing rather than decorative.
 *
 * **Both sleeves, one chronological list, two cards.** The day covers the
 * account rather than one agent, so an equity rebalance pass belongs on this
 * page — but it is rendered by `EquityPassCard`, because every field
 * `CycleCard` reads is options vocabulary an equity pass does not have. Put
 * through the options card it read "unclassified · iv rank unmeasured · menu 0
 * · the cycle never reached the Gate" about a pass that had just passed twelve
 * checks per order, which is the most misleading sentence this dashboard could
 * print about a risk system whose whole claim is that the Gate is load-bearing.
 */

import { useState } from "react"
import { ChevronRight } from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Empty, EmptyDescription, EmptyHeader, EmptyTitle } from "@/components/ui/empty"
import { Progress } from "@/components/ui/progress"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { cn } from "@/lib/utils"
import type { EquityCycle } from "@/lib/equity"
import {
  type Check,
  type DayRecord,
  type DeclineCategory,
  type JournalCycle,
  clock,
  fmt,
  isEquityPass,
  num,
  pct,
  stageHint,
  stageLabel,
} from "@/lib/status"

const STAGE_TONE: Record<string, "default" | "secondary" | "destructive" | "outline"> = {
  filled: "default",
  submitted: "default",
  dry_run: "secondary",
  vetoed: "destructive",
  rejected: "destructive",
  breached: "destructive",
  declined: "outline",
  no_setup: "outline",
  no_candidates: "outline",
}

/**
 * A second badge, next to the stage — requirement 3.
 *
 * `stage` alone reads `no_setup` for both "the entry rule could not be
 * decided" and "the market did not qualify", and reads `vetoed` for a Risk
 * Gate stop without distinguishing it from either. This tone map is what
 * makes those three read as different facts rather than the same shrug.
 */
const CATEGORY_TONE: Record<DeclineCategory, "default" | "secondary" | "destructive" | "outline"> = {
  traded: "default",
  approved_not_sent: "secondary",
  gate_veto: "destructive",
  broker_rejected: "destructive",
  model_declined: "outline",
  undecidable: "secondary",
  no_setup: "outline",
  no_candidates: "outline",
  other: "outline",
}

/**
 * Categories where the journal's own note adds something.
 *
 * These are the ones a reader might have to act on, and the note is where the
 * specifics live — the broker's rejection text, the failing check's numbers.
 * Everywhere else the note restates the plain-English explanation in internal
 * vocabulary, and printing both says the same thing twice, the second time
 * worse.
 */
const NOTE_WORTH_SHOWING = new Set<DeclineCategory>([
  "gate_veto",
  "broker_rejected",
  "traded",
  "other",
])

export function Journal({ cycles, day }: { cycles: DayRecord[]; day: string }) {
  const [open, setOpen] = useState<string | null>(null)

  if (cycles.length === 0) {
    return (
      <Empty>
        <EmptyHeader>
          <EmptyTitle>No record for {day}</EmptyTitle>
          <EmptyDescription>
            The agent writes a line every time it looks at the market, even when
            it decides to do nothing. An empty day therefore means it was not
            running that day — not that it looked and found nothing.
          </EmptyDescription>
        </EmptyHeader>
      </Empty>
    )
  }

  return (
    <div className="flex flex-col gap-4">
      {cycles.map((record) => {
        const shared = {
          key: record.cycle_id,
          expanded: open === record.cycle_id,
          onToggle: () =>
            setOpen(open === record.cycle_id ? null : record.cycle_id),
        }
        // One chronological list, two cards. The day is what happened on the
        // account, in order, so both sleeves belong here — but an equity pass
        // put through `CycleCard` would be described in a vocabulary it has
        // none of. See `DayRecord` in `lib/status.ts`.
        return isEquityPass(record) ? (
          <EquityPassCard {...shared} pass={record} />
        ) : (
          <CycleCard {...shared} cycle={record} />
        )
      })}
    </div>
  )
}

/**
 * One equity rebalance pass — specs/09 D10.
 *
 * Shows what the options card shows, in the terms this sleeve has: no menu and
 * no structure, but a target weight per order and, per order, the check tape
 * the equity Gate produced. The Equity tab's Today card lists the same orders;
 * what this adds is the tape, which is the journal's whole job — "the checks
 * that nearly stopped it" is as much a question about a rebalance as about a
 * spread.
 */
function EquityPassCard({
  pass,
  expanded,
  onToggle,
}: {
  pass: EquityCycle
  expanded: boolean
  onToggle: () => void
}) {
  const stage = pass.stage ?? "unknown"
  const orders = pass.orders ?? []
  const skipped = pass.skipped ?? []
  const submitted = orders.filter((order) => order.submission).length
  const vetoed = orders.filter(
    (order) => (order.verdict?.reasons ?? []).length > 0,
  ).length
  // The same headline the options card carries, for the same reason: a check
  // that passed with 0.2% of its budget left is the evidence that this Gate is
  // load-bearing, and it is worth a badge rather than a scroll.
  const near = nearMisses(orders.flatMap((order) => order.verdict?.checks ?? []))

  return (
    <Card>
      <CardHeader>
        <div className="flex flex-wrap items-center gap-3">
          <span className="text-muted-foreground font-mono text-sm tabular-nums">
            {clock(pass.as_of)}
          </span>
          <CardTitle className="text-base">stock rebalance</CardTitle>
          <Badge variant={STAGE_TONE[stage] ?? "outline"}>
            {stageLabel(stage)}
          </Badge>
          <Badge variant="secondary">
            {orders.length} order{orders.length === 1 ? "" : "s"}
          </Badge>
          {vetoed > 0 ? (
            <Badge variant="destructive">
              {vetoed} stopped by a safety check
            </Badge>
          ) : null}
          {near.length > 0 ? (
            <Badge variant="secondary">
              close to the “{near[0].name.replace(/_/g, " ")}” limit
            </Badge>
          ) : null}
          {orders.length > 0 ? (
            <Button
              variant="ghost"
              size="sm"
              className="ml-auto"
              onClick={onToggle}
            >
              <ChevronRight
                data-icon="inline-start"
                className={cn("transition-transform", expanded && "rotate-90")}
              />
              {expanded ? "less" : "detail"}
            </Button>
          ) : null}
        </div>
        <CardDescription>
          {pass.note ||
            `${submitted} of ${orders.length} planned order${
              orders.length === 1 ? "" : "s"
            } reached the broker. ${stageHint(stage)}`}
        </CardDescription>
      </CardHeader>

      {expanded && orders.length > 0 ? (
        <CardContent className="flex flex-col gap-6">
          <section className="grid grid-cols-2 gap-4 sm:grid-cols-4">
            {(
              [
                ["sleeve equity", fmt(pass.equity, 0)],
                ["turnover", fmt(pass.turnover, 0)],
                ["no-trade band", pct(pass.band_pct, 0)],
                ["skipped", String(skipped.length)],
              ] as const
            ).map(([label, value]) => (
              <div key={label} className="flex flex-col gap-0.5">
                <span className="text-muted-foreground text-xs tracking-wide uppercase">
                  {label}
                </span>
                <span className="tabular-nums">{value}</span>
              </div>
            ))}
          </section>

          {orders.map((order, index) => (
            <section
              key={`${order.symbol}-${index}`}
              className="flex flex-col gap-2"
            >
              <div className="flex flex-wrap items-center gap-2">
                <span className="font-mono text-sm">{order.symbol}</span>
                <Badge variant={order.side === "buy" ? "secondary" : "outline"}>
                  {order.side}
                </Badge>
                <span className="text-muted-foreground text-sm tabular-nums">
                  {fmt(order.shares, 4)} @ {fmt(order.reference_price)} ={" "}
                  {fmt(order.notional, 0)}
                </span>
                <Badge
                  variant={
                    (order.verdict?.reasons ?? []).length > 0
                      ? "destructive"
                      : "outline"
                  }
                >
                  {order.outcome}
                </Badge>
              </div>
              {(order.verdict?.reasons ?? []).map((reason) => (
                <p key={reason.check} className="text-destructive text-xs">
                  Not sent — failed the “{reason.check.replace(/_/g, " ")}”
                  safety check: {reason.detail}
                </p>
              ))}
              {(order.verdict?.waived ?? []).map((reason) => (
                <p key={reason.check} className="text-muted-foreground text-xs">
                  Allowed through despite “{reason.check.replace(/_/g, " ")}”:{" "}
                  {reason.detail}. This order reduces risk, so the limit does not
                  apply to it.
                </p>
              ))}
              <OrderChecks checks={order.verdict?.checks ?? []} />
            </section>
          ))}

          {skipped.length > 0 ? (
            <section className="flex flex-col gap-1">
              <h3 className="text-muted-foreground text-xs font-medium">
                LEFT ALONE — already close enough to target, or not priceable
              </h3>
              {skipped.map((item) => (
                <p key={item.symbol} className="text-muted-foreground text-xs">
                  {item.symbol} · {item.reason.replace(/_/g, " ")} —{" "}
                  {item.detail}
                </p>
              ))}
            </section>
          ) : null}
        </CardContent>
      ) : null}
    </Card>
  )
}

function CycleCard({
  cycle,
  expanded,
  onToggle,
}: {
  cycle: JournalCycle
  expanded: boolean
  onToggle: () => void
}) {
  const stage = cycle.stage ?? "unknown"
  const checks = cycle.verdict?.checks ?? []
  const near = nearMisses(checks)
  // Read off the backend, never re-derived here — see the module-level
  // comment on `JournalCycle.category` in `lib/status.ts` for why a second,
  // client-side classifier is exactly the drift this field exists to rule out.
  // `undefined` on a record predating the field reads as "other" rather than
  // crashing the card; an equity pass never arrives here at all.
  const category: DeclineCategory = cycle.category ?? "other"
  const categoryLabel = cycle.category_label ?? "unclassified"

  return (
    <Card>
      <CardHeader>
        <div className="flex flex-wrap items-center gap-3">
          <span className="text-muted-foreground font-mono text-sm tabular-nums">
            {clock(cycle.as_of)}
          </span>
          <CardTitle className="text-base">
            {cycle.read?.underlying ?? "—"}
          </CardTitle>
          {/* One badge, not two. The stage and the category describe the same
              event at different resolutions, and where they differ the stage
              is the misleading one: a cycle that could not evaluate its entry
              rule is stored as `no_setup`, which renders "no opportunity" —
              the exact wrong conclusion — right beside the category saying the
              volatility history is missing. The category is always at least as
              specific, so it wins whenever the backend classified the cycle. */}
          {category === "other" ? (
            <Badge variant={STAGE_TONE[stage] ?? "outline"}>
              {stageLabel(stage)}
            </Badge>
          ) : (
            <Badge variant={CATEGORY_TONE[category]}>{categoryLabel}</Badge>
          )}
          {near.length > 0 ? (
            <Badge variant="secondary">
              close to the “{near[0].name.replace(/_/g, " ")}” limit
            </Badge>
          ) : null}
          <Button
            variant="ghost"
            size="sm"
            className="ml-auto"
            onClick={onToggle}
          >
            <ChevronRight
              data-icon="inline-start"
              className={cn("transition-transform", expanded && "rotate-90")}
            />
            {expanded ? "less" : "detail"}
          </Button>
        </div>
        <CardDescription className="flex flex-col gap-1">
          <span>{cycle.category_detail || stageHint(stage) || cycle.note}</span>
          {/* The note is shown only where it carries something the sentence
              above cannot: on a failure it is the only place the broker's or
              the process's own words appear, and those are what someone
              debugging actually needs. On a quiet cycle it merely restates the
              explanation in the vocabulary this page exists to avoid, so it is
              dropped rather than printed twice. */}
          {cycle.note && NOTE_WORTH_SHOWING.has(category) ? (
            <span className="text-muted-foreground font-mono text-xs wrap-break-word">
              {cycle.note}
            </span>
          ) : null}
        </CardDescription>
      </CardHeader>

      {expanded ? (
        <CardContent className="flex flex-col gap-6">
          <Read cycle={cycle} />
          {cycle.choice?.rationale ? (
            <section className="flex flex-col gap-2">
              <h3 className="text-muted-foreground text-xs font-medium">
                WHY THE MODEL PICKED THIS ONE
              </h3>
              <blockquote className="border-primary bg-muted/40 border-l-2 p-3 text-sm">
                {cycle.choice.rationale}
              </blockquote>
              <p className="text-muted-foreground text-xs">
                Picked option {(cycle.choice.candidate_index ?? 0) + 1} out of{" "}
                {cycle.candidates?.length ?? 0} it was shown. The model only
                chooses from a pre-approved shortlist, and how confident it says
                it is has no effect on the size of the trade.
              </p>
            </section>
          ) : null}
          <Checks checks={checks} />
        </CardContent>
      ) : null}
    </Card>
  )
}

/**
 * One equity order's check tape — in full when it is interesting.
 *
 * A rebalance pass can carry eighty-four orders, and eighty-four twelve-row
 * tables is a page nobody reads. So the tape is shown whenever it says
 * something — a check failed, or one passed inside 15% of its limit — and
 * summarised otherwise. The summary still names the tightest check and how
 * much of its budget it used, because that is the number the full table exists
 * to surface; what is dropped is eleven rows saying "plenty of room".
 */
function OrderChecks({ checks }: { checks: Check[] }) {
  if (checks.length === 0) {
    return (
      <p className="text-muted-foreground text-sm">
        No safety checks were recorded for this order.
      </p>
    )
  }
  const failed = checks.filter((check) => !check.passed)
  const near = nearMisses(checks)
  if (failed.length > 0 || near.length > 0) return <Checks checks={checks} />

  const measured = checks
    .map((check) => ({ check, room: headroom(check) }))
    .filter((item): item is { check: Check; room: number } => item.room !== null)
    .sort((a, b) => a.room - b.room)
  const tightest = measured[0]

  return (
    <p className="text-muted-foreground text-xs">
      All {checks.length} safety checks passed with room to spare
      {tightest
        ? ` — the closest was “${tightest.check.name.replace(/_/g, " ")}”, which used ${Math.round(
            (1 - tightest.room) * 100,
          )}% of what it is allowed`
        : ""}
      .
    </p>
  )
}

function Read({ cycle }: { cycle: JournalCycle }) {
  const pairs = [
    [
      "price at the time",
      cycle.read?.spot ? fmt(cycle.read.spot) : "—",
      "",
    ],
    [
      "how jumpy the market was",
      cycle.read?.iv_rank ?? "not available",
      cycle.read?.iv_rank
        ? "0 = calmest in a year, 100 = wildest"
        : "the past year of readings is missing",
    ],
    [
      "options it could have traded",
      String(cycle.candidates?.length ?? 0),
      "shortlist built for the model",
    ],
    [
      "most it could have lost",
      cycle.proposal?.risk?.max_loss ? fmt(cycle.proposal.risk.max_loss) : "—",
      "capped by design",
    ],
  ] as const

  return (
    <section className="grid grid-cols-2 gap-4 sm:grid-cols-4">
      {pairs.map(([label, value, hint]) => (
        <div key={label} className="flex flex-col gap-0.5">
          <span className="text-muted-foreground text-xs">{label}</span>
          <span className="tabular-nums">{value}</span>
          {hint ? (
            <span className="text-muted-foreground text-xs">{hint}</span>
          ) : null}
        </div>
      ))}
    </section>
  )
}

/**
 * Render a check's observed or limit value.
 *
 * The backend serialises `Decimal` faithfully, so a liquidity ratio arrives as
 * `0.0358744394618834080717488789` — every digit true and none of them useful
 * in a table. Numeric values are trimmed to something readable; anything that
 * is not a number (`present`, `[-29.73, 29.73]`) is passed through untouched,
 * because those are already the form the Gate reported them in.
 */
function measure(value: string | null): string {
  if (value === null || value === "") return "—"
  const parsed = Number(value)
  if (!Number.isFinite(parsed)) return value
  if (Number.isInteger(parsed)) return String(parsed)
  const digits = Math.abs(parsed) < 1 ? 4 : 2
  return parsed.toFixed(digits)
}

/**
 * Headroom, 0–1. `null` where the check is not a magnitude.
 *
 * Mirrors `_headroom` in `interface/read.py`, including its floor rule: a check
 * that *passed* with observed above its limit is measured against a floor, not
 * a budget, and 1 − observed/limit would rank the safest check on the page as
 * the tightest one. Unrankable is the honest answer.
 */
function headroom(check: Check): number | null {
  if (check.observed === null || check.limit === null) return null
  const limit = Math.abs(num(check.limit))
  if (limit === 0) return null
  if (!check.passed) return 0
  const used = Math.abs(num(check.observed)) / limit
  if (used > 1) return null
  return Math.max(0, Math.min(1, 1 - used))
}

function nearMisses(checks: Check[]): Check[] {
  return checks
    .filter((check) => {
      const room = headroom(check)
      return check.passed && room !== null && room < 0.15
    })
    .sort((a, b) => (headroom(a) ?? 1) - (headroom(b) ?? 1))
}

function Checks({ checks }: { checks: Check[] }) {
  if (checks.length === 0) {
    return (
      <p className="text-muted-foreground text-sm">
        No safety checks ran, because there was no order to check — the agent
        decided not to trade before it got that far.
      </p>
    )
  }
  const ordered = [...checks].sort((a, b) => {
    if (a.passed !== b.passed) return a.passed ? 1 : -1
    return (headroom(a) ?? 1) - (headroom(b) ?? 1)
  })
  const passed = checks.filter((check) => check.passed).length

  return (
    <section className="flex flex-col gap-2">
      <h3 className="text-muted-foreground text-xs font-medium">
        SAFETY CHECKS — {passed} OF {checks.length} PASSED. Closest call first;
        a full bar means that check nearly stopped the order.
      </h3>
      <div className="overflow-x-auto">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="w-16" />
              <TableHead>Check</TableHead>
              <TableHead className="text-right">Measured</TableHead>
              <TableHead className="text-right">Allowed</TableHead>
              <TableHead className="w-32">How close</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {ordered.map((check) => {
              const room = headroom(check)
              return (
                <TableRow key={check.name}>
                  <TableCell>
                    <Badge variant={check.passed ? "outline" : "destructive"}>
                      {check.passed ? "pass" : "fail"}
                    </Badge>
                  </TableCell>
                  <TableCell className="text-xs">
                    {check.name.replace(/_/g, " ")}
                  </TableCell>
                  <TableCell className="text-right tabular-nums">
                    {measure(check.observed)}
                  </TableCell>
                  <TableCell className="text-right tabular-nums">
                    {measure(check.limit)}
                  </TableCell>
                  <TableCell>
                    {room === null ? (
                      <span className="text-muted-foreground text-xs">n/a</span>
                    ) : (
                      <Progress value={(1 - room) * 100} />
                    )}
                  </TableCell>
                </TableRow>
              )
            })}
          </TableBody>
        </Table>
      </div>
    </section>
  )
}
