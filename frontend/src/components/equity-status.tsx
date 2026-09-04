/**
 * The Equity page — the strategy `ai_quant_researcher` validated, being held.
 *
 * Ordered by what someone glancing at it needs first, and the first thing is
 * **which strategy this is and what earned it the account**. That block is what
 * makes the page a claim rather than a screenshot: the fingerprint, the sealed
 * out-of-sample alpha, beta and `t`, the number of candidates that window has
 * screened, and the sentence saying the window can refute and cannot confirm.
 *
 * Then the money, then the warnings, then the book itself — target weight
 * against held weight, with each position's own no-trade band, because the band
 * is proportional and one global number would be wrong about most of the rows.
 *
 * Nothing here can place an order. `alphagate.interface` imports neither
 * `execution` nor `live` nor `marketdata`, and a boundary test fails the build
 * if it ever does — so this page reads a JSON file and has no path to a broker.
 */

import { useState } from "react"
import {
  AlertTriangle,
  Ban,
  CircleCheck,
  CircleSlash,
  Clock,
  FlaskConical,
  ShieldAlert,
} from "lucide-react"

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
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
import { cn } from "@/lib/utils"
import {
  type EquityCycle,
  type EquityOrder,
  type EquitySnapshot,
  type EquityStatusResponse,
  type PositionLine,
  type SealedRun,
  type Strategy,
  equityHealth,
  hasStrategy,
  sealedVerdict,
} from "@/lib/equity"
import { clock, fmt, num, pct, signed, stageHint, stageLabel } from "@/lib/status"

export function EquityStatus({
  status,
  cycles,
}: {
  status: EquityStatusResponse
  cycles: EquityCycle[] | null
}) {
  const state = equityHealth(status)
  const snapshot = status.snapshot

  if (!snapshot) {
    return (
      <Empty>
        <EmptyHeader>
          <EmptyMedia variant="icon">
            <CircleSlash />
          </EmptyMedia>
          <EmptyTitle>The stock agent has not run yet</EmptyTitle>
          <EmptyDescription>
            {state.detail} Start it from the project folder — this one works out
            every order and places none of them:
            <br />
            <code className="font-mono">
              uv run --directory backend python -m alphagate equity-plan
            </code>
            <br />
            This page fills in on its own once it has run.
          </EmptyDescription>
        </EmptyHeader>
      </Empty>
    )
  }

  return (
    <div className="flex flex-col gap-6">
      <HealthBanner status={status} snapshot={snapshot} />
      {hasStrategy(snapshot.strategy) ? (
        <StrategyCard strategy={snapshot.strategy} />
      ) : null}
      <Money snapshot={snapshot} />
      <Warnings snapshot={snapshot} />
      <Book snapshot={snapshot} />
      <Limits snapshot={snapshot} />
      <Today snapshot={snapshot} cycles={cycles} />
    </div>
  )
}

/* ------------------------------------------------------------------ */

function HealthBanner({
  status,
  snapshot,
}: {
  status: EquityStatusResponse
  snapshot: EquitySnapshot
}) {
  const state = equityHealth(status)
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
        heartbeat {clock(snapshot.as_of)}
        {snapshot.next_pass ? ` · next rebalance ${clock(snapshot.next_pass)}` : ""}
      </span>
      {snapshot.stale.length > 0 ? (
        <Badge variant="outline" className="ml-auto">
          prices are out of date — nothing will trade
        </Badge>
      ) : null}
    </div>
  )
}

/**
 * The provenance. Rendered, never acted on.
 *
 * Every number below was measured before this process existed, on a window the
 * search could not read. Showing them beside the positions is the whole point
 * of the page: anyone can screenshot a book, and this says what bought the
 * right to hold it.
 */
function StrategyCard({ strategy }: { strategy: Strategy }) {
  const sealed = strategy.sealed
  return (
    <Card>
      <CardHeader>
        <CardDescription className="flex flex-wrap items-center gap-2">
          <FlaskConical className="size-3.5" />
          the one stock strategy this agent is allowed to trade
          <Badge variant="outline">picks from {strategy.universe}</Badge>
        </CardDescription>
        <CardTitle className="font-mono text-base">
          {strategy.name}{" "}
          <span className="text-muted-foreground">[{strategy.fingerprint}]</span>
        </CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        <p className="text-muted-foreground text-sm">{strategy.hypothesis}</p>

        <div className="grid gap-4 sm:grid-cols-3 lg:grid-cols-6">
          <Figure
            label="return above the market"
            value={`${(sealed.alpha * 100).toFixed(2)}%/yr`}
            hint="in the locked-away years"
          />
          <Figure
            label="market exposure"
            value={sealed.beta.toFixed(2)}
            hint="1.00 = moves with the market"
          />
          <Figure
            label="confidence score"
            value={sealed.t_alpha.toFixed(2)}
            hint="above ~2.6 is a strong result"
            tone={sealed.is_significant ? "up" : undefined}
          />
          <Figure
            label="worst drop"
            value={`${(sealed.max_drawdown * 100).toFixed(2)}%`}
            hint="peak to trough"
          />
          <Figure
            label="trades in the test"
            value={String(sealed.trades)}
            hint={`over ${sealed.observations} trading days`}
          />
          <Figure
            label="strategies tested this way"
            value={String(sealed.looks)}
            hint="each one raises the bar"
          />
        </div>

        <Separator />

        <dl className="grid gap-x-6 gap-y-2 text-sm sm:grid-cols-2">
          <Row term="years it was tested on" detail={sealed.window} />
          <Row
            term="how it tested"
            detail={sealedVerdict(sealed)}
            tone={sealed.refuted ? "bad" : "ok"}
          />
          <Row term="holdings decided on" detail={strategy.as_of} />
          <Row
            term="ideas tried"
            detail={`${strategy.distinct_hypotheses}. The more ideas you try, the more likely one looks good by luck — so this number is held against the result, not hidden from it.`}
          />
          <Row term="why this one" detail={strategy.selection_rule} />
          <Row term="data it was tested on" detail={strategy.dataset_version} />
        </dl>

        <p className="text-muted-foreground border-l-2 pl-3 text-xs">
          <strong>What this test can and cannot tell you.</strong> The strategy
          was designed without access to these {sealed.observations} trading
          days, then tried against them once. Coming through that is the
          strongest evidence available here — but it is still only two years,
          which is not enough to prove any strategy works. It is enough to prove
          one <em>doesn't</em>, and this one was not caught out. Nothing on this
          page is a prediction or a recommendation.
        </p>
      </CardContent>
    </Card>
  )
}

function Figure({
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
    <div>
      <p className="text-muted-foreground text-xs">{label}</p>
      <p
        className={cn(
          "text-lg tabular-nums",
          tone === "up" && "text-emerald-600 dark:text-emerald-500",
          tone === "down" && "text-destructive",
        )}
      >
        {value}
      </p>
      {hint ? <p className="text-muted-foreground text-xs">{hint}</p> : null}
    </div>
  )
}

function Row({
  term,
  detail,
  tone,
}: {
  term: string
  detail: string
  tone?: "ok" | "bad"
}) {
  return (
    <div className="flex gap-2">
      <dt className="text-muted-foreground w-40 shrink-0">{term}</dt>
      <dd
        className={cn(
          tone === "bad" && "text-destructive",
          tone === "ok" && "text-emerald-600 dark:text-emerald-500",
        )}
      >
        {detail}
      </dd>
    </div>
  )
}

/* ------------------------------------------------------------------ */

function Money({ snapshot }: { snapshot: EquitySnapshot }) {
  const day = num(snapshot.session_change)
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
        label="Invested"
        value={`${(num(snapshot.gross_exposure) * 100).toFixed(1)}%`}
        // Not "N of M": the two counts move independently, and right after the
        // strategy re-picks its list the agent holds more names than it wants,
        // which "86 of 75" renders as a bug rather than as a pending sell.
        hint={`${snapshot.positions_held} companies held, ${snapshot.positions_wanted} wanted`}
      />
      <Stat label="Cash" value={fmt(snapshot.cash)} hint="not yet invested" />
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

function Warnings({ snapshot }: { snapshot: EquitySnapshot }) {
  const warnings = []

  if (snapshot.killswitch_tripped) {
    warnings.push(
      <Alert variant="destructive" key="kill">
        <ShieldAlert />
        <AlertTitle>Trading halted — losses hit the safety limit</AlertTitle>
        <AlertDescription>
          Nothing more will be bought. Selling still works, so the agent can
          still reduce what it holds. It will not restart itself: someone has to
          look at what happened and clear it by hand.
        </AlertDescription>
      </Alert>,
    )
  }

  if (snapshot.is_blocked) {
    warnings.push(
      <Alert variant="destructive" key="blocked">
        <Ban />
        <AlertTitle>The broker has frozen this account</AlertTitle>
        <AlertDescription>
          Every order will be rejected until the broker lifts it. Your existing
          holdings are untouched, but nothing below will happen in the meantime.
        </AlertDescription>
      </Alert>,
    )
  }

  if (snapshot.unpriced.length > 0) {
    warnings.push(
      <Alert key="unpriced">
        <AlertTriangle />
        <AlertTitle>
          No price available for {snapshot.unpriced.length} holding
          {snapshot.unpriced.length === 1 ? "" : "s"}
        </AlertTitle>
        <AlertDescription>
          <span className="font-mono text-xs">
            {snapshot.unpriced.slice(0, 24).join(" ")}
            {snapshot.unpriced.length > 24 ? " …" : ""}
          </span>
          <br />
          These are left exactly as they are. The agent will not buy or sell
          something it cannot value, so they are skipped rather than traded on a
          guess. Everything else continues normally.
        </AlertDescription>
      </Alert>,
    )
  }

  if (snapshot.off_book.length > 0) {
    warnings.push(
      <Alert key="offbook">
        <AlertTriangle />
        <AlertTitle>
          {snapshot.off_book.length} holding
          {snapshot.off_book.length === 1 ? "" : "s"} the strategy no longer
          wants
        </AlertTitle>
        <AlertDescription>
          <span className="font-mono text-xs">
            {snapshot.off_book.join(" ")}
          </span>
          <br />
          These will be sold off completely at the next rebalance. Expected
          behaviour — the strategy re-picks what it holds every few days, and
          anything dropped from the list is sold rather than left behind.
        </AlertDescription>
      </Alert>,
    )
  }

  if (snapshot.stale.length > 0) {
    warnings.push(
      <Alert key="stale">
        <Clock />
        <AlertTitle>Prices are out of date — nothing will trade</AlertTitle>
        <AlertDescription>
          The broker's last prices for {snapshot.stale.length} holding
          {snapshot.stale.length === 1 ? "" : "s"} are too old to act on.{" "}
          <strong>
            Almost always this just means the US market is closed
          </strong>{" "}
          — it is normal outside 09:30–16:00 New York time. The agent refuses to
          trade on last night's prices, so it will wait. If you see this while
          the market is open, the price feed is the thing to check.
        </AlertDescription>
      </Alert>,
    )
  }

  return warnings.length > 0 ? (
    <div className="flex flex-col gap-3">{warnings}</div>
  ) : null
}

/* ------------------------------------------------------------------ */

type Filter = "outside" | "core" | "all"

function Book({ snapshot }: { snapshot: EquitySnapshot }) {
  const [filter, setFilter] = useState<Filter>("outside")

  const outside = snapshot.lines.filter((line) => !line.inside_band)
  const core = snapshot.lines.filter((line) => line.core)
  const shown =
    filter === "outside" ? outside : filter === "core" ? core : snapshot.lines

  const sorted = [...shown].sort(
    (a, b) => Math.abs(num(b.drift)) - Math.abs(num(a.drift)),
  )

  return (
    <Card>
      <CardHeader>
        <CardDescription className="flex flex-wrap items-center gap-2">
          what it wants to hold, against what it actually holds
          <span className="text-muted-foreground">
            a holding is only traded once it has drifted more than{" "}
            {pct(snapshot.drift_band_pct, 0)} away from its target, and never for
            less than {fmt(snapshot.min_order_notional, 0)} — small corrections
            cost more in fees than they are worth
          </span>
        </CardDescription>
        <CardTitle className="flex flex-wrap items-center gap-2 text-base">
          {outside.length} of {snapshot.lines.length} need adjusting
          <div className="ml-auto flex gap-1">
            {(
              [
                ["outside", "needs adjusting", outside.length],
                ["core", "biggest holdings", core.length],
                ["all", "everything", snapshot.lines.length],
              ] as [Filter, string, number][]
            ).map(([option, label, count]) => (
              <Button
                key={option}
                size="sm"
                variant={option === filter ? "default" : "outline"}
                onClick={() => setFilter(option)}
              >
                {label} ({count})
              </Button>
            ))}
          </div>
        </CardTitle>
      </CardHeader>
      <CardContent>
        {sorted.length === 0 ? (
          <p className="text-muted-foreground text-sm">
            Nothing needs adjusting — every holding is already close enough to
            its target. This is the normal answer on roughly four days out of
            five, because the strategy only re-picks what it holds every five
            trading days.
          </p>
        ) : (
          <div className="overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>company</TableHead>
                  <TableHead className="text-right">wanted</TableHead>
                  <TableHead className="text-right">held</TableHead>
                  <TableHead className="text-right">value</TableHead>
                  <TableHead className="text-right">off by</TableHead>
                  <TableHead className="text-right">allowed</TableHead>
                  <TableHead className="text-right">price</TableHead>
                  <TableHead>what happens</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {sorted.slice(0, 120).map((line) => (
                  <BookRow key={line.symbol} line={line} />
                ))}
              </TableBody>
            </Table>
            {sorted.length > 120 ? (
              <p className="text-muted-foreground pt-2 text-xs">
                showing 120 of {sorted.length}
              </p>
            ) : null}
          </div>
        )}
      </CardContent>
    </Card>
  )
}

function BookRow({ line }: { line: PositionLine }) {
  const drift = num(line.drift)
  const action = line.inside_band
    ? "leave alone"
    : drift > 0
      ? num(line.held_shares) === 0
        ? "buy in"
        : "buy more"
      : num(line.target_weight) === 0
        ? "sell all"
        : "trim"

  return (
    <TableRow>
      <TableCell className="font-mono">
        {line.symbol}
        {line.core ? (
          <Badge variant="outline" className="ml-2">
            core
          </Badge>
        ) : null}
      </TableCell>
      <TableCell className="text-right tabular-nums">
        {pct(line.target_weight, 2)}
      </TableCell>
      <TableCell className="text-right tabular-nums">
        {pct(line.held_weight, 2)}
      </TableCell>
      <TableCell className="text-right tabular-nums">
        {fmt(line.market_value, 0)}
      </TableCell>
      <TableCell
        className={cn(
          "text-right tabular-nums",
          !line.inside_band && drift > 0 && "text-emerald-600 dark:text-emerald-500",
          !line.inside_band && drift < 0 && "text-destructive",
        )}
      >
        {signed(line.drift)}
      </TableCell>
      <TableCell className="text-muted-foreground text-right tabular-nums">
        {fmt(line.threshold, 0)}
      </TableCell>
      <TableCell className="text-right tabular-nums">
        {line.price === null ? "—" : fmt(line.price)}
      </TableCell>
      <TableCell>
        <Badge variant={action === "leave alone" ? "outline" : "secondary"}>
          {action}
        </Badge>
      </TableCell>
    </TableRow>
  )
}

/* ------------------------------------------------------------------ */

function Limits({ snapshot }: { snapshot: EquitySnapshot }) {
  const bars = [
    {
      label: "orders placed today",
      used: snapshot.orders_today,
      limit: snapshot.max_daily_orders,
      shown: `${snapshot.orders_today} / ${snapshot.max_daily_orders}`,
    },
    {
      label: "money traded today",
      used: num(snapshot.turnover_today),
      limit: num(snapshot.max_daily_turnover),
      shown: `${fmt(snapshot.turnover_today, 0)} / ${fmt(snapshot.max_daily_turnover, 0)}`,
    },
    {
      label: "share of the account invested",
      used: num(snapshot.gross_exposure),
      limit: 1,
      shown: `${(num(snapshot.gross_exposure) * 100).toFixed(1)}% / 100%`,
    },
    {
      label: "down from its best ever",
      used: num(snapshot.drawdown_pct),
      limit: num(snapshot.max_drawdown_pct),
      shown: `${pct(snapshot.drawdown_pct, 2)} / ${pct(snapshot.max_drawdown_pct, 0)}`,
    },
  ]

  return (
    <Card>
      <CardHeader>
        <CardDescription>
          Hard ceilings the agent will not trade past. A full bar means it stops
          itself — that is the system working, not a fault.
        </CardDescription>
        <CardTitle className="text-base">
          How close it is to its safety limits
        </CardTitle>
      </CardHeader>
      <CardContent className="grid gap-4 sm:grid-cols-2">
        {bars.map((bar) => {
          const share = bar.limit > 0 ? (bar.used / bar.limit) * 100 : 0
          return (
            <div key={bar.label} className="flex flex-col gap-1.5">
              <div className="flex items-baseline justify-between text-sm">
                <span className="text-muted-foreground">{bar.label}</span>
                <span className="tabular-nums">{bar.shown}</span>
              </div>
              <Progress value={Math.min(100, Math.max(0, share))} />
            </div>
          )
        })}
      </CardContent>
    </Card>
  )
}

/* ------------------------------------------------------------------ */

function Today({
  snapshot,
  cycles,
}: {
  snapshot: EquitySnapshot
  cycles: EquityCycle[] | null
}) {
  const passes = cycles ?? []
  const orders = passes.flatMap((cycle) => cycle.orders ?? [])

  return (
    <Card>
      <CardHeader>
        <CardDescription>{snapshot.session_day}</CardDescription>
        <CardTitle className="text-base">
          Today — {passes.length} pass{passes.length === 1 ? "" : "es"},{" "}
          {orders.length} order{orders.length === 1 ? "" : "s"}
        </CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        <div className="flex flex-col gap-2">
          {Object.entries(snapshot.stage_counts).map(([stage, count]) => (
            <div key={stage} className="flex flex-wrap items-baseline gap-2">
              <Badge variant="outline">
                {count}× {stageLabel(stage)}
              </Badge>
              <span className="text-muted-foreground text-xs">
                {stageHint(stage)}
              </span>
            </div>
          ))}
          {Object.keys(snapshot.stage_counts).length === 0 ? (
            <span className="text-muted-foreground text-sm">
              The agent has not rebalanced yet today. It does this once, shortly
              after the market opens.
            </span>
          ) : null}
        </div>

        {snapshot.note ? (
          <p className="text-muted-foreground text-sm">{snapshot.note}</p>
        ) : null}

        {orders.length > 0 ? (
          <div className="overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>symbol</TableHead>
                  <TableHead>side</TableHead>
                  <TableHead className="text-right">shares</TableHead>
                  <TableHead className="text-right">at</TableHead>
                  <TableHead className="text-right">notional</TableHead>
                  <TableHead>outcome</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {orders.map((order, index) => (
                  <OrderRow key={`${order.symbol}-${index}`} order={order} />
                ))}
              </TableBody>
            </Table>
          </div>
        ) : null}
      </CardContent>
    </Card>
  )
}

function OrderRow({ order }: { order: EquityOrder }) {
  const reasons = order.verdict?.reasons ?? []
  const waived = order.verdict?.waived ?? []
  return (
    <>
      <TableRow>
        <TableCell className="font-mono">{order.symbol}</TableCell>
        <TableCell>
          <Badge variant={order.side === "buy" ? "secondary" : "outline"}>
            {order.side}
          </Badge>
        </TableCell>
        <TableCell className="text-right tabular-nums">
          {fmt(order.shares, 4)}
        </TableCell>
        <TableCell className="text-right tabular-nums">
          {fmt(order.reference_price)}
        </TableCell>
        <TableCell className="text-right tabular-nums">
          {fmt(order.notional, 0)}
        </TableCell>
        <TableCell>
          <Badge
            variant={
              order.outcome === "filled"
                ? "default"
                : reasons.length > 0
                  ? "destructive"
                  : "outline"
            }
          >
            {order.outcome}
          </Badge>
        </TableCell>
      </TableRow>
      {reasons.length > 0 || waived.length > 0 ? (
        <TableRow>
          <TableCell colSpan={6} className="text-xs">
            {reasons.map((reason) => (
              <p key={reason.check} className="text-destructive">
                Not sent — failed the “{reason.check.replace(/_/g, " ")}” safety
                check: {reason.detail}
              </p>
            ))}
            {waived.map((reason) => (
              <p key={reason.check} className="text-muted-foreground">
                Allowed through despite “{reason.check.replace(/_/g, " ")}”:{" "}
                {reason.detail}. This order reduces risk, so the limit does not
                apply to it.
              </p>
            ))}
          </TableCell>
        </TableRow>
      ) : null}
    </>
  )
}

export type { SealedRun }
