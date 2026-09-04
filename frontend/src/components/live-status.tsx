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
  stageHint,
  stageLabel,
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
          <EmptyTitle>The options agent has not run yet</EmptyTitle>
          <EmptyDescription>
            {state.detail} Start it from the project folder — this one rehearses
            a whole day without placing anything:
            <br />
            <code className="font-mono">
              uv run --directory backend python -m alphagate run --dry-run
            </code>
            <br />
            This page fills in on its own once the agent starts.
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
        last checked {clock(snapshot.as_of)}
        {/* A "next check" countdown on a stopped agent is a promise nothing is
            going to keep — the schedule it comes from is only advanced by the
            process that has stopped running. */}
        {status.running ? ` · next check ${untilNext(snapshot.next_slot)}` : ""}
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
      <Stat
        label="Account value"
        value={fmt(snapshot.equity)}
        hint="practice money — none of this is real"
      />
      <Stat
        label="Today"
        value={signed(snapshot.session_change)}
        hint="change since yesterday's close"
        tone={day === 0 ? undefined : day > 0 ? "up" : "down"}
      />
      <Stat
        label="Open profit / loss"
        value={signed(String(unrealised))}
        hint={`across ${snapshot.positions.length} open position${
          snapshot.positions.length === 1 ? "" : "s"
        }, not yet banked`}
        tone={unrealised === 0 ? undefined : unrealised > 0 ? "up" : "down"}
      />
      <Stat
        label="Available to trade"
        value={fmt(snapshot.options_buying_power)}
        hint={`options level ${snapshot.options_level}${
          snapshot.is_pattern_day_trader ? " · flagged as a day trader" : ""
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
        <AlertTitle>Trading halted — losses hit the safety limit</AlertTitle>
        <AlertDescription>
          No new positions will be opened. Existing ones can still be closed, so
          the agent can still get out of what it holds. It will not restart
          itself: someone has to look at what happened and clear it by hand.
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
            ? "The broker has frozen this account"
            : "This account is not approved for the options it wants to trade"}
        </AlertTitle>
        <AlertDescription>
          {snapshot.is_blocked
            ? "Every order will be rejected until the broker lifts it. Nothing below will happen in the meantime."
            : `The strategy uses two-leg option spreads, which need options level 3. This account is at level ${snapshot.options_level}, so every order will be rejected. Request level 3 from Alpaca — the account settings page has the form.`}
        </AlertDescription>
      </Alert>,
    )
  }

  if (snapshot.unexplained.length > 0) {
    warnings.push(
      <Alert key="unexplained">
        <AlertTriangle />
        <AlertTitle>
          {snapshot.unexplained.length} position
          {snapshot.unexplained.length === 1 ? "" : "s"} this agent did not open
        </AlertTitle>
        <AlertDescription>
          <p>
            These are open at the broker but the agent has no record of buying
            them — someone traded by hand, an option was assigned, or a previous
            session's records were lost. They are{" "}
            <strong>not counted in its risk limits</strong>, so the real risk on
            this account is higher than the numbers below say. Close them, or
            accept that the limits are measuring less than you hold.
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
          Every 15 minutes each position is re-priced and checked against three
          exit rules: take the profit once {pct(snapshot.profit_target, 0)} of
          what was collected is banked, cut the loss at{" "}
          {fmt(snapshot.stop_multiple, 1)}× that amount, and close anything with{" "}
          {snapshot.min_dte} days left to run. No model is asked — exits are
          fixed rules.
        </CardDescription>
      </CardHeader>
      <CardContent>
        {snapshot.positions.length === 0 ? (
          <Empty className="border-0">
            <EmptyHeader>
              <EmptyTitle>Nothing open right now</EmptyTitle>
              <EmptyDescription>
                This is normal. The agent is watching{" "}
                {snapshot.universe.join(", ")} and will only open a position
                when its entry rule is met.
              </EmptyDescription>
            </EmptyHeader>
          </Empty>
        ) : (
          <div className="overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Position</TableHead>
                  <TableHead className="text-right">Contracts</TableHead>
                  <TableHead className="text-right">Collected</TableHead>
                  <TableHead className="text-right">Worth now</TableHead>
                  <TableHead className="text-right">Profit</TableHead>
                  <TableHead className="text-right">Days left</TableHead>
                  <TableHead>Progress toward closing</TableHead>
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
  // The close is out and waiting to fill. Until it settles the agent proposes
  // nothing further for this position -- one decision, one order.
  const closing = (snapshot.closing ?? []).includes(position.cycle_id)

  return (
    <TableRow>
      <TableCell>
        <div className="font-medium">{position.underlying}</div>
        <div className="text-muted-foreground font-mono text-xs">
          {position.structure}
        </div>
        {closing ? (
          <Badge variant="secondary" className="mt-1">
            closing — waiting for the broker
          </Badge>
        ) : null}
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
          <div className="flex flex-col gap-1">
            <Badge variant="secondary">no live price</Badge>
            <span className="text-muted-foreground text-xs">
              The broker gave no quote this round, so the exit rules could not
              be checked. Usually the market is closed.
            </span>
          </div>
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
    return (
      <div className="flex flex-col gap-1">
        <Badge variant="outline">paid to open</Badge>
        <span className="text-muted-foreground text-xs">
          This position cost money to open rather than collecting it, so the
          profit target below does not apply to it.
        </span>
      </div>
    )
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
        <span>close at a {pct(String(-stop), 0)} loss</span>
        <span
          className={cn(
            "font-medium",
            position.rule !== "hold" && "text-foreground",
          )}
        >
          now {pct(position.fraction_of_credit, 0)}
        </span>
        <span>close at a {pct(snapshot.profit_target, 0)} gain</span>
      </div>
    </div>
  )
}

function Limits({ snapshot }: { snapshot: Snapshot }) {
  const rows = [
    {
      label: "Positions open at once",
      used: snapshot.open_structures,
      limit: snapshot.max_open_structures,
      render: (v: number) => String(v),
    },
    {
      label: "Trades today",
      used: snapshot.fills_today,
      limit: snapshot.max_daily_trades,
      render: (v: number) => String(v),
    },
    {
      label: "Most it could lose, all positions",
      used: num(snapshot.open_risk),
      limit: num(snapshot.max_portfolio_risk),
      render: (v: number) => fmt(String(v)),
    },
    {
      label: "Down from its best ever",
      used: num(snapshot.drawdown_pct),
      limit: num(snapshot.max_drawdown_pct),
      render: (v: number) => pct(String(v)),
    },
  ]

  return (
    <Card>
      <CardHeader>
        <CardTitle>How close it is to its safety limits</CardTitle>
        <CardDescription>
          Four hard ceilings the agent will not trade past. A full bar means it
          will refuse its own next trade — that is the system working, not a
          fault.
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
            Drawdown is measured against the best this account has ever been
            worth, {fmt(snapshot.peak_equity)} — not against this morning. A
            limit that reset overnight could never actually stop a slow losing
            streak.
          </p>
        ) : null}
      </CardContent>
    </Card>
  )
}

function Today({ snapshot }: { snapshot: Snapshot }) {
  const entries = Object.entries(snapshot.stage_counts).sort(
    ([, a], [, b]) => b - a,
  )
  return (
    <Card>
      <CardHeader>
        <CardTitle>What it did today</CardTitle>
        <CardDescription>
          The agent looks at the market every 15 minutes and records the result
          every time, including when it decides to do nothing — which is most
          of the time, and is the expected behaviour.
        </CardDescription>
      </CardHeader>
      <CardContent>
        {entries.length === 0 ? (
          <p className="text-muted-foreground text-sm">
            Nothing recorded yet for {snapshot.session_day}. If the agent is
            running, the first entry appears at the next 15-minute check.
          </p>
        ) : (
          <ul className="flex flex-col gap-2">
            {entries.map(([stage, count]) => (
              <li key={stage} className="flex flex-wrap items-baseline gap-2">
                <Badge variant="outline" className="gap-1.5">
                  <span className="tabular-nums font-medium">{count}×</span>
                  {stageLabel(stage)}
                </Badge>
                <span className="text-muted-foreground text-xs">
                  {stageHint(stage)}
                </span>
              </li>
            ))}
          </ul>
        )}
      </CardContent>
    </Card>
  )
}
