/**
 * The live page — what the agent is doing right now.
 *
 * Ordered by what someone glancing at it needs first: is it alive, is anything
 * wrong, what does it hold, how much room is left before a limit stops it.
 * The journal answers "why did it decide that" and lives on the other tab;
 * this answers "what is true now", which the journal structurally cannot.
 */

import {
  AlertTriangle,
  Ban,
  CircleCheck,
  CircleSlash,
  Clock,
  ShieldAlert,
} from "lucide-react"

import { OptionBookCard } from "@/components/option-book-card"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import {
  Empty,
  EmptyDescription,
  EmptyHeader,
  EmptyMedia,
  EmptyTitle,
} from "@/components/ui/empty"
import { Progress } from "@/components/ui/progress"
import { Separator } from "@/components/ui/separator"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import type { OptionBookResponse } from "@/lib/option-book"
import { cn } from "@/lib/utils"
import {
  type PositionStatus,
  type Snapshot,
  type StatusResponse,
  clock,
  fmt,
  health,
  num,
  pct,
  signed,
  untilNext,
} from "@/lib/status"

export function LiveStatus({
  status,
  optionBook,
}: {
  status: StatusResponse
  optionBook: OptionBookResponse | null
}) {
  const state = health(status)
  const snapshot = status.snapshot

  if (!snapshot) {
    return (
      <Empty>
        <EmptyHeader>
          <EmptyMedia variant="icon">
            <CircleSlash />
          </EmptyMedia>
          <EmptyTitle>The agent has not run yet</EmptyTitle>
          <EmptyDescription>
            {state.detail}. Start it with{" "}
            <code className="font-mono">python -m alphagate run --dry-run</code>.
          </EmptyDescription>
        </EmptyHeader>
      </Empty>
    )
  }

  return (
    <div className="flex flex-col gap-6">
      <HealthBanner status={status} />
      <OptionBookCard optionBook={optionBook} />
      <Money snapshot={snapshot} />
      <Warnings snapshot={snapshot} />
      <Positions snapshot={snapshot} />
      <Limits snapshot={snapshot} />
      <Today snapshot={snapshot} />
    </div>
  )
}

/* ------------------------------------------------------------------ */

function HealthBanner({ status }: { status: StatusResponse }) {
  const state = health(status)
  const snapshot = status.snapshot!
  const tone =
    state.tone === "ok"
      ? "default"
      : state.tone === "bad"
        ? "destructive"
        : "secondary"

  return (
    <div className="flex flex-wrap items-center gap-3">
      <Badge variant={tone} className="gap-1.5">
        {state.tone === "ok" ? <CircleCheck /> : <AlertTriangle />}
        {state.label}
      </Badge>
      <span className="text-muted-foreground text-sm">{state.detail}</span>
      <Separator orientation="vertical" className="h-4" />
      <span className="text-muted-foreground flex items-center gap-1.5 text-sm">
        <Clock className="size-3.5" />
        last cycle {clock(snapshot.as_of)} · next {untilNext(snapshot.next_slot)}
      </span>
      <div className="ml-auto flex items-center gap-2">
        {snapshot.universe.map((symbol) => (
          <Badge key={symbol} variant="outline">
            {symbol}
          </Badge>
        ))}
      </div>
    </div>
  )
}

function Money({ snapshot }: { snapshot: Snapshot }) {
  const day = num(snapshot.session_change)
  const unrealised = snapshot.positions.reduce(
    (total, position) => total + num(position.unrealised),
    0,
  )
  return (
    <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
      <Stat label="Equity" value={fmt(snapshot.equity)} hint="paper account" />
      <Stat
        label="Today"
        value={signed(snapshot.session_change)}
        hint="equity less last close"
        tone={day === 0 ? undefined : day > 0 ? "up" : "down"}
      />
      <Stat
        label="Unrealised"
        value={signed(String(unrealised))}
        hint={`${snapshot.positions.length} open`}
        tone={unrealised === 0 ? undefined : unrealised > 0 ? "up" : "down"}
      />
      <Stat
        label="Options BP"
        value={fmt(snapshot.options_buying_power)}
        hint={`level ${snapshot.options_level}${
          snapshot.is_pattern_day_trader ? " · PDT" : ""
        }`}
      />
    </div>
  )
}

function Stat({
  label,
  value,
  hint,
  tone,
}: {
  label: string
  value: string
  hint?: string
  tone?: "up" | "down"
}) {
  return (
    <Card>
      <CardHeader>
        <CardDescription>{label}</CardDescription>
        <CardTitle
          className={cn(
            "text-2xl tabular-nums",
            tone === "up" && "text-emerald-600 dark:text-emerald-500",
            tone === "down" && "text-destructive",
          )}
        >
          {value}
        </CardTitle>
      </CardHeader>
      {hint ? (
        <CardContent>
          <p className="text-muted-foreground text-xs">{hint}</p>
        </CardContent>
      ) : null}
    </Card>
  )
}

function Warnings({ snapshot }: { snapshot: Snapshot }) {
  const warnings = []

  if (snapshot.killswitch_tripped) {
    warnings.push(
      <Alert variant="destructive" key="kill">
        <ShieldAlert />
        <AlertTitle>Kill switch latched</AlertTitle>
        <AlertDescription>
          Opens are refused until a human clears it. Closes still go through —
          the Gate never blocks an exit.
        </AlertDescription>
      </Alert>,
    )
  }

  if (!snapshot.can_trade_spreads) {
    warnings.push(
      <Alert variant="destructive" key="level">
        <Ban />
        <AlertTitle>
          {snapshot.is_blocked
            ? "The broker has blocked this account"
            : `Options level ${snapshot.options_level} — spreads need 3`}
        </AlertTitle>
        <AlertDescription>
          Every vertical in the menu is unfillable. Fix this before the open;
          finding out from a rejection at 14:30 costs half a trading day.
        </AlertDescription>
      </Alert>,
    )
  }

  if (snapshot.unexplained.length > 0) {
    warnings.push(
      <Alert key="unexplained">
        <AlertTriangle />
        <AlertTitle>
          {snapshot.unexplained.length} legs the journal cannot explain
        </AlertTitle>
        <AlertDescription>
          <p>
            These are open at the broker with no journalled fill behind them — a
            manual trade, an assignment, or a session whose journal is missing.
            They are deliberately <strong>not</strong> in the risk model, so the
            Gate is budgeting as though they were not there.
          </p>
          <ul className="mt-2 flex flex-col gap-1 font-mono text-xs">
            {snapshot.unexplained.map((leg) => (
              <li key={leg}>{leg}</li>
            ))}
          </ul>
        </AlertDescription>
      </Alert>,
    )
  }

  if (warnings.length === 0) return null
  return <div className="flex flex-col gap-3">{warnings}</div>
}

function Positions({ snapshot }: { snapshot: Snapshot }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Open positions</CardTitle>
        <CardDescription>
          Re-priced every slot and put through the exit policy — take{" "}
          {pct(snapshot.profit_target, 0)} of the credit, stop at{" "}
          {fmt(snapshot.stop_multiple, 1)}×, close at {snapshot.min_dte} days.
        </CardDescription>
      </CardHeader>
      <CardContent>
        {snapshot.positions.length === 0 ? (
          <Empty className="border-0">
            <EmptyHeader>
              <EmptyTitle>Flat</EmptyTitle>
              <EmptyDescription>
                Nothing open. The agent is looking for a setup on{" "}
                {snapshot.universe.join(", ")}.
              </EmptyDescription>
            </EmptyHeader>
          </Empty>
        ) : (
          <div className="overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Structure</TableHead>
                  <TableHead className="text-right">Qty</TableHead>
                  <TableHead className="text-right">Entry</TableHead>
                  <TableHead className="text-right">Mark</TableHead>
                  <TableHead className="text-right">P&amp;L</TableHead>
                  <TableHead className="text-right">DTE</TableHead>
                  <TableHead>Toward exit</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {snapshot.positions.map((position) => (
                  <PositionRow
                    key={position.cycle_id}
                    position={position}
                    snapshot={snapshot}
                  />
                ))}
              </TableBody>
            </Table>
          </div>
        )}
      </CardContent>
    </Card>
  )
}

function PositionRow({
  position,
  snapshot,
}: {
  position: PositionStatus
  snapshot: Snapshot
}) {
  const pnl = num(position.unrealised)
  const unpriced = position.rule === "unpriced"

  return (
    <TableRow>
      <TableCell>
        <div className="font-medium">{position.underlying}</div>
        <div className="text-muted-foreground font-mono text-xs">
          {position.structure}
        </div>
      </TableCell>
      <TableCell className="text-right tabular-nums">
        {position.quantity}
      </TableCell>
      <TableCell className="text-right tabular-nums">
        {fmt(position.entry_premium)}
      </TableCell>
      <TableCell className="text-right tabular-nums">
        {unpriced ? "—" : fmt(position.mark)}
      </TableCell>
      <TableCell
        className={cn(
          "text-right tabular-nums",
          !unpriced && pnl > 0 && "text-emerald-600 dark:text-emerald-500",
          !unpriced && pnl < 0 && "text-destructive",
        )}
      >
        {unpriced ? "—" : signed(position.unrealised)}
      </TableCell>
      <TableCell
        className={cn(
          "text-right tabular-nums",
          position.days_to_expiry <= snapshot.min_dte && "text-destructive",
        )}
      >
        {position.days_to_expiry}
      </TableCell>
      <TableCell className="min-w-56">
        {unpriced ? (
          <Badge variant="secondary">unpriced this slot</Badge>
        ) : (
          <ExitProgress position={position} snapshot={snapshot} />
        )}
        <div className="text-muted-foreground mt-1 text-xs">
          {position.detail}
        </div>
      </TableCell>
    </TableRow>
  )
}

/**
 * How close this position is to being closed, as a bar.
 *
 * The scale runs from the stop on the left to the profit target on the right,
 * with the current fraction of credit sitting somewhere between. That is the
 * honest picture: both rules are live at once, and a position is always
 * travelling between them.
 */
function ExitProgress({
  position,
  snapshot,
}: {
  position: PositionStatus
  snapshot: Snapshot
}) {
  if (position.fraction_of_credit === null) {
    return <Badge variant="outline">debit — no credit fraction</Badge>
  }
  const fraction = num(position.fraction_of_credit)
  const target = num(snapshot.profit_target)
  const stop = -num(snapshot.stop_multiple)
  const span = target - stop
  const value = span === 0 ? 0 : ((fraction - stop) / span) * 100

  return (
    <div className="flex flex-col gap-1">
      <Progress value={Math.max(0, Math.min(100, value))} />
      <div className="text-muted-foreground flex justify-between text-xs tabular-nums">
        <span>stop {pct(String(stop), 0)}</span>
        <span
          className={cn(
            "font-medium",
            position.rule !== "hold" && "text-foreground",
          )}
        >
          {pct(position.fraction_of_credit, 0)}
        </span>
        <span>target {pct(snapshot.profit_target, 0)}</span>
      </div>
    </div>
  )
}

function Limits({ snapshot }: { snapshot: Snapshot }) {
  const rows = [
    {
      label: "Open structures",
      used: snapshot.open_structures,
      limit: snapshot.max_open_structures,
      render: (v: number) => String(v),
    },
    {
      label: "Fills today",
      used: snapshot.fills_today,
      limit: snapshot.max_daily_trades,
      render: (v: number) => String(v),
    },
    {
      label: "Portfolio risk",
      used: num(snapshot.open_risk),
      limit: num(snapshot.max_portfolio_risk),
      render: (v: number) => fmt(String(v)),
    },
    {
      label: "Drawdown",
      used: num(snapshot.drawdown_pct),
      limit: num(snapshot.max_drawdown_pct),
      render: (v: number) => pct(String(v)),
    },
  ]

  return (
    <Card>
      <CardHeader>
        <CardTitle>Room before the Gate stops it</CardTitle>
        <CardDescription>
          The four budgeted limits, against what is used. A bar near full is the
          agent about to refuse its own next proposal.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="flex flex-col gap-4">
          {rows.map((row) => {
            const ratio = row.limit === 0 ? 0 : (row.used / row.limit) * 100
            return (
              <div key={row.label} className="flex flex-col gap-1.5">
                <div className="flex items-baseline justify-between text-sm">
                  <span>{row.label}</span>
                  <span className="text-muted-foreground tabular-nums">
                    {row.render(row.used)} / {row.render(row.limit)}
                  </span>
                </div>
                <Progress value={Math.max(0, Math.min(100, ratio))} />
              </div>
            )
          })}
        </div>
        {snapshot.peak_equity ? (
          <p className="text-muted-foreground mt-4 text-xs">
            Drawdown is measured from the high-water mark of{" "}
            {fmt(snapshot.peak_equity)}, carried across days — a mark that reset
            overnight would be a kill switch that could never latch.
          </p>
        ) : null}
      </CardContent>
    </Card>
  )
}

function Today({ snapshot }: { snapshot: Snapshot }) {
  const entries = Object.entries(snapshot.stage_counts).sort(([a], [b]) =>
    a.localeCompare(b),
  )
  return (
    <Card>
      <CardHeader>
        <CardTitle>Cycles today</CardTitle>
        <CardDescription>
          Every cycle is journalled, including the ones that decided nothing —
          those are the majority and they are the point.
        </CardDescription>
      </CardHeader>
      <CardContent>
        {entries.length === 0 ? (
          <p className="text-muted-foreground text-sm">
            No cycles journalled yet for {snapshot.session_day}.
          </p>
        ) : (
          <div className="flex flex-wrap gap-2">
            {entries.map(([stage, count]) => (
              <Badge key={stage} variant="outline" className="gap-1.5">
                {stage}
                <span className="tabular-nums font-medium">{count}</span>
              </Badge>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
