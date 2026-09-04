/**
 * Both sleeves, side by side — specs/03 D6.
 *
 * Shown above the tabs rather than inside either one, because the property
 * this panel exists to demonstrate — that one sleeve's drawdown cannot trip
 * the other's kill switch — is only visible when both are on screen at once.
 * A judge who has to switch tabs to compare two numbers is being asked to
 * remember the first one; this panel exists so they never have to.
 *
 * Every figure here is deliberately paired with its *own* threshold. There is
 * no combined "portfolio drawdown" bar anywhere in this file, and that
 * omission is the point: `interface/sleeves.py`'s module docstring explains
 * why a single blended number would misrepresent the isolation the sleeve
 * design provides.
 */

import { ShieldAlert } from "lucide-react"

import { Badge } from "@/components/ui/badge"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import type { SleeveSummary, SleevesResponse } from "@/lib/sleeves"
import { fmt, num, pct } from "@/lib/status"

export function SleevesOverview({ sleeves }: { sleeves: SleevesResponse | null }) {
  if (sleeves === null) return null
  return (
    <div className="grid gap-4 sm:grid-cols-2">
      <SleeveCard sleeve={sleeves.options} accent="options" />
      <SleeveCard sleeve={sleeves.equity} accent="equity" />
    </div>
  )
}

function SleeveCard({
  sleeve,
  accent,
}: {
  sleeve: SleeveSummary
  accent: "options" | "equity"
}) {
  const drawdown = num(sleeve.drawdown_pct)
  const limit = num(sleeve.max_drawdown_pct)
  const share = limit > 0 ? (drawdown / limit) * 100 : 0

  return (
    <Card>
      <CardHeader>
        <CardDescription className="flex flex-wrap items-center gap-2">
          {accent === "options" ? "options half" : "stocks half"}
          <Badge variant="outline">
            ${fmt(sleeve.allocation, 0)} set aside for it
          </Badge>
          {!sleeve.running ? (
            <Badge variant="destructive">stopped</Badge>
          ) : null}
          {sleeve.killswitch_tripped ? (
            <Badge variant="destructive" className="gap-1">
              <ShieldAlert className="size-3" />
              halted on losses
            </Badge>
          ) : null}
        </CardDescription>
        <CardTitle className="text-2xl tabular-nums">
          {sleeve.equity === null ? "—" : fmt(sleeve.equity)}
        </CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        {sleeve.equity === null ? (
          <p className="text-muted-foreground text-xs">
            {sleeve.note ||
              "No value to show yet — this half has not reported since it was started."}
          </p>
        ) : (
          <div className="grid grid-cols-2 gap-3 text-xs">
            <Stat
              label="banked"
              value={sleeve.realised === null ? "—" : fmt(sleeve.realised)}
            />
            <Stat
              label="on open positions"
              value={sleeve.unrealised === null ? "—" : fmt(sleeve.unrealised)}
            />
          </div>
        )}

        <div className="flex flex-col gap-1.5">
          <div className="flex items-baseline justify-between text-sm">
            <span className="text-muted-foreground">
              down from its best, against its own stop
            </span>
            <span className="tabular-nums">
              {pct(sleeve.drawdown_pct, 2)} / {pct(sleeve.max_drawdown_pct, 0)}
            </span>
          </div>
          <Progress value={Math.min(100, Math.max(0, share))} />
        </div>

        <div className="flex items-center justify-between text-sm">
          <span className="text-muted-foreground">open positions</span>
          <span className="tabular-nums">{sleeve.open_positions ?? "—"}</span>
        </div>
        <div className="flex items-center justify-between text-sm">
          <span className="text-muted-foreground">{sleeve.activity_label || "today"}</span>
          <span className="tabular-nums">{sleeve.activity_today ?? "—"}</span>
        </div>
      </CardContent>
    </Card>
  )
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <p className="text-muted-foreground">{label}</p>
      <p className="tabular-nums">{value}</p>
    </div>
  )
}
