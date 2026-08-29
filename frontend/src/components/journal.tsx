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
import { type Check, type JournalCycle, clock, fmt, num } from "@/lib/status"

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

export function Journal({ cycles, day }: { cycles: JournalCycle[]; day: string }) {
  const [open, setOpen] = useState<string | null>(null)

  if (cycles.length === 0) {
    return (
      <Empty>
        <EmptyHeader>
          <EmptyTitle>Nothing journalled for {day}</EmptyTitle>
          <EmptyDescription>
            Every cycle writes a line, so an empty day means the agent has not
            run — not that it found nothing.
          </EmptyDescription>
        </EmptyHeader>
      </Empty>
    )
  }

  return (
    <div className="flex flex-col gap-4">
      {cycles.map((cycle) => (
        <CycleCard
          key={cycle.cycle_id}
          cycle={cycle}
          expanded={open === cycle.cycle_id}
          onToggle={() =>
            setOpen(open === cycle.cycle_id ? null : cycle.cycle_id)
          }
        />
      ))}
    </div>
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
          <Badge variant={STAGE_TONE[stage] ?? "outline"}>{stage}</Badge>
          {near.length > 0 ? (
            <Badge variant="secondary">near {near[0].name}</Badge>
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
        <CardDescription>{cycle.note}</CardDescription>
      </CardHeader>

      {expanded ? (
        <CardContent className="flex flex-col gap-6">
          <Read cycle={cycle} />
          {cycle.choice?.rationale ? (
            <section className="flex flex-col gap-2">
              <h3 className="text-muted-foreground text-xs font-medium tracking-wide uppercase">
                What the model said
              </h3>
              <blockquote className="border-primary bg-muted/40 border-l-2 p-3 text-sm">
                {cycle.choice.rationale}
              </blockquote>
              <p className="text-muted-foreground text-xs">
                Chose #{cycle.choice.candidate_index ?? "—"} of{" "}
                {cycle.candidates?.length ?? 0} structures. Self-reported
                confidence is recorded and never acted on.
              </p>
            </section>
          ) : null}
          <Checks checks={checks} />
        </CardContent>
      ) : null}
    </Card>
  )
}

function Read({ cycle }: { cycle: JournalCycle }) {
  const pairs = [
    ["spot", cycle.read?.spot ? fmt(cycle.read.spot) : "—"],
    ["iv rank", cycle.read?.iv_rank ?? "unmeasured"],
    ["menu", String(cycle.candidates?.length ?? 0)],
    ["max loss", cycle.proposal?.risk?.max_loss ? fmt(cycle.proposal.risk.max_loss) : "—"],
  ] as const

  return (
    <section className="grid grid-cols-2 gap-4 sm:grid-cols-4">
      {pairs.map(([label, value]) => (
        <div key={label} className="flex flex-col gap-0.5">
          <span className="text-muted-foreground text-xs tracking-wide uppercase">
            {label}
          </span>
          <span className="tabular-nums">{value}</span>
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

/** Headroom, 0–1. `null` where the check is not a magnitude. */
function headroom(check: Check): number | null {
  if (check.observed === null || check.limit === null) return null
  const limit = Math.abs(num(check.limit))
  if (limit === 0) return null
  if (!check.passed) return 0
  return Math.max(0, Math.min(1, 1 - Math.abs(num(check.observed)) / limit))
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
        The cycle never reached the Gate.
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
      <h3 className="text-muted-foreground text-xs font-medium tracking-wide uppercase">
        The Gate — {passed}/{checks.length} passed, tightest first
      </h3>
      <div className="overflow-x-auto">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="w-16" />
              <TableHead>Check</TableHead>
              <TableHead className="text-right">Observed</TableHead>
              <TableHead className="text-right">Limit</TableHead>
              <TableHead className="w-32">Room</TableHead>
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
                  <TableCell className="font-mono text-xs">{check.name}</TableCell>
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
