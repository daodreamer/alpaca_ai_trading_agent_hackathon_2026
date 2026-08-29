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
import { clock, fmt, num, pct, signed } from "@/lib/status"

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
          <EmptyTitle>The equity agent has not run yet</EmptyTitle>
          <EmptyDescription>
            {state.detail}. Start it with{" "}
            <code className="font-mono">
              python -m alphagate equity-plan
            </code>
            .
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
        {snapshot.next_pass ? ` · rebalance at ${clock(snapshot.next_pass)}` : ""}
      </span>
      {snapshot.stale.length > 0 ? (
        <Badge variant="outline" className="ml-auto">
          {snapshot.stale.length} marks stale — nothing will trade
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
          the strategy this account executes
          <Badge variant="outline">{strategy.status}</Badge>
          <Badge variant="outline">{strategy.universe}</Badge>
        </CardDescription>
        <CardTitle className="font-mono text-base">
          {strategy.name}{" "}
          <span className="text-muted-foreground">[{strategy.fingerprint}]</span>
        </CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        <p className="text-muted-foreground text-sm">{strategy.hypothesis}</p>

        <div className="grid gap-4 sm:grid-cols-3 lg:grid-cols-6">
          <Figure label="alpha" value={`${(sealed.alpha * 100).toFixed(2)}%/yr`} />
          <Figure label="beta" value={sealed.beta.toFixed(2)} />
          <Figure
            label="t(alpha)"
            value={`${sealed.t_alpha >= 0 ? "+" : ""}${sealed.t_alpha.toFixed(2)}`}
            tone={sealed.is_significant ? "up" : undefined}
          />
          <Figure
            label="info ratio"
            value={`${sealed.information_ratio >= 0 ? "+" : ""}${sealed.information_ratio.toFixed(2)}`}
          />
          <Figure label="sealed trades" value={String(sealed.trades)} />
          <Figure label="looks" value={String(sealed.looks)} />
        </div>

        <Separator />

        <dl className="grid gap-x-6 gap-y-2 text-sm sm:grid-cols-2">
          <Row term="sealed window" detail={sealed.window} />
          <Row
            term="verdict"
            detail={sealedVerdict(sealed)}
            tone={sealed.refuted ? "bad" : "ok"}
          />
          <Row term="book as of" detail={strategy.as_of} />
          <Row
            term="hypotheses searched"
            detail={`${strategy.distinct_hypotheses} — the multiplicity denominator`}
          />
          <Row term="selection rule" detail={strategy.selection_rule} />
          <Row term="dataset" detail={strategy.dataset_version} />
        </dl>

        <p className="text-muted-foreground border-l-2 pl-3 text-xs italic">
          The sealed window can refute and cannot confirm: the standard error on
          an annualised Sharpe over {sealed.observations} sessions is about
          ±0.71. It also proves only that the embargoed <em>data</em> was not
          read — not that the embargoed <em>period</em> did not inform a
          decision.
        </p>
      </CardContent>
    </Card>
  )
}

function Figure({
  label,
  value,
  tone,
}: {
  label: string
  value: string
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
      <Stat label="Equity" value={fmt(snapshot.equity)} hint="paper account" />
      <Stat
        label="Today"
        value={signed(snapshot.session_change)}
        hint="equity less last close"
        tone={day === 0 ? undefined : day > 0 ? "up" : "down"}
      />
      <Stat
        label="Invested"
        value={`${(num(snapshot.gross_exposure) * 100).toFixed(1)}%`}
        hint={`${snapshot.positions_held} held of ${snapshot.positions_wanted} wanted`}
      />
      <Stat
        label="Cash"
        value={fmt(snapshot.cash)}
        hint="an idle sleeve holds the benchmark, not cash"
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

function Warnings({ snapshot }: { snapshot: EquitySnapshot }) {
  const warnings = []

  if (snapshot.killswitch_tripped) {
    warnings.push(
      <Alert variant="destructive" key="kill">
        <ShieldAlert />
        <AlertTitle>Kill switch latched</AlertTitle>
        <AlertDescription>
          Buys are refused until a human clears it. Sells still go through — the
          Gate waives a budget for an order that reduces risk, and never for one
          that adds it.
        </AlertDescription>
      </Alert>,
    )
  }

  if (snapshot.is_blocked) {
    warnings.push(
      <Alert variant="destructive" key="blocked">
        <Ban />
        <AlertTitle>The broker has blocked this account</AlertTitle>
        <AlertDescription>
          Every order will be rejected. Nothing below will happen until it is
          cleared.
        </AlertDescription>
      </Alert>,
    )
  }

  if (snapshot.unpriced.length > 0) {
    warnings.push(
      <Alert key="unpriced">
        <AlertTriangle />
        <AlertTitle>
          {snapshot.unpriced.length} names in the book have no price
        </AlertTitle>
        <AlertDescription>
          <span className="font-mono text-xs">
            {snapshot.unpriced.slice(0, 24).join(" ")}
            {snapshot.unpriced.length > 24 ? " …" : ""}
          </span>
          <br />
          They are held rather than traded on a guess. A position we cannot
          value is one the plan deliberately does not touch.
        </AlertDescription>
      </Alert>,
    )
  }

  if (snapshot.off_book.length > 0) {
    warnings.push(
      <Alert key="offbook">
        <AlertTriangle />
        <AlertTitle>
          {snapshot.off_book.length} holdings the book does not want
        </AlertTitle>
        <AlertDescription>
          <span className="font-mono text-xs">
            {snapshot.off_book.join(" ")}
          </span>
          <br />
          They are sold to zero on the next pass. A symbol absent from the book
          has a target of zero, so its whole position is the drift — there is no
          separate exit rule, and none to forget to run.
        </AlertDescription>
      </Alert>,
    )
  }

  if (snapshot.stale.length > 0) {
    warnings.push(
      <Alert key="stale">
        <Clock />
        <AlertTitle>
          {snapshot.stale.length} marks are older than the freshness limit
        </AlertTitle>
        <AlertDescription>
          Usually this is the whole book at once, and the reason is that the
          market is closed. Nothing will trade on a stale price: the Gate refuses
          it, and a plan built on prices from last night is a plan about a market
          that no longer exists.
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
          the book — target against held
          <span className="text-muted-foreground">
            band {pct(snapshot.drift_band_pct, 0)} of each position, floor{" "}
            {fmt(snapshot.min_order_notional, 0)}
          </span>
        </CardDescription>
        <CardTitle className="flex flex-wrap items-center gap-2 text-base">
          {outside.length} of {snapshot.lines.length} outside the band
          <div className="ml-auto flex gap-1">
            {(["outside", "core", "all"] as Filter[]).map((option) => (
              <Button
                key={option}
                size="sm"
                variant={option === filter ? "default" : "outline"}
                onClick={() => setFilter(option)}
              >
                {option}
                {option === "outside" ? ` (${outside.length})` : ""}
                {option === "core" ? ` (${core.length})` : ""}
                {option === "all" ? ` (${snapshot.lines.length})` : ""}
              </Button>
            ))}
          </div>
        </CardTitle>
      </CardHeader>
      <CardContent>
        {sorted.length === 0 ? (
          <p className="text-muted-foreground text-sm">
            Nothing here. Every position is inside its own no-trade band, which
            is the honest answer four sessions in five — the strategy rebalances
            every five.
          </p>
        ) : (
          <div className="overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>symbol</TableHead>
                  <TableHead className="text-right">target</TableHead>
                  <TableHead className="text-right">held</TableHead>
                  <TableHead className="text-right">value</TableHead>
                  <TableHead className="text-right">drift</TableHead>
                  <TableHead className="text-right">band</TableHead>
                  <TableHead className="text-right">price</TableHead>
                  <TableHead>next</TableHead>
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
    ? "hold"
    : drift > 0
      ? num(line.held_shares) === 0
        ? "open"
        : "buy"
      : num(line.target_weight) === 0
        ? "close"
        : "sell"

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
        <Badge variant={action === "hold" ? "outline" : "secondary"}>
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
      label: "orders today",
      used: snapshot.orders_today,
      limit: snapshot.max_daily_orders,
      shown: `${snapshot.orders_today} / ${snapshot.max_daily_orders}`,
    },
    {
      label: "turnover today",
      used: num(snapshot.turnover_today),
      limit: num(snapshot.max_daily_turnover),
      shown: `${fmt(snapshot.turnover_today, 0)} / ${fmt(snapshot.max_daily_turnover, 0)}`,
    },
    {
      label: "invested",
      used: num(snapshot.gross_exposure),
      limit: 1,
      shown: `${(num(snapshot.gross_exposure) * 100).toFixed(1)}% / 100%`,
    },
    {
      label: "drawdown",
      used: num(snapshot.drawdown_pct),
      limit: num(snapshot.max_drawdown_pct),
      shown: `${pct(snapshot.drawdown_pct, 2)} / ${pct(snapshot.max_drawdown_pct, 0)}`,
    },
  ]

  return (
    <Card>
      <CardHeader>
        <CardDescription>
          how much room is left before the Gate refuses
        </CardDescription>
        <CardTitle className="text-base">Limits</CardTitle>
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
        <div className="flex flex-wrap gap-2">
          {Object.entries(snapshot.stage_counts).map(([stage, count]) => (
            <Badge key={stage} variant="outline">
              {stage.replace(/_/g, " ")} × {count}
            </Badge>
          ))}
          {Object.keys(snapshot.stage_counts).length === 0 ? (
            <span className="text-muted-foreground text-sm">
              no pass has run today
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
                vetoed · {reason.check} — {reason.detail}
              </p>
            ))}
            {waived.map((reason) => (
              <p key={reason.check} className="text-muted-foreground">
                waived · {reason.check} — {reason.detail} (this order reduces
                risk)
              </p>
            ))}
          </TableCell>
        </TableRow>
      ) : null}
    </>
  )
}

export type { SealedRun }
