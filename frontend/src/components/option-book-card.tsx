/**
 * The strategy provenance panel for the options sleeve — specs/07 D1, D8.
 *
 * The equity tab's `StrategyCard` (`equity-status.tsx`) is the template: name,
 * fingerprint, hypothesis, then the rule and the sealed run side by side, so a
 * judge can hold "what it trades" and "what earned it the account" in one
 * screen. This is that same panel for the option rule, and it carries the same
 * non-negotiable constraint that card does not have to: the sealed run here
 * measured `t=+1.11` against a bar of `1.96` — not significant, and the
 * artefact is explicit that the window can refute this rule and cannot confirm
 * it (specs/10 D8). Nothing below may render that as "confirmed", "validated"
 * or "passed", and no figure here gets a green "pass" tone — see
 * `optionSealedVerdict` in `lib/option-book.ts`.
 *
 * **An unavailable book is shown, not hidden.** `load_option_book` refuses a
 * rule whose sealed window was refuted or whose registry status has not earned
 * a paper position, and that refusal is exactly the control working — so the
 * reasons travel here and render as plainly as the numbers do when the book
 * loads.
 */

import { AlertTriangle, FlaskConical } from "lucide-react"

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Separator } from "@/components/ui/separator"
import {
  type OptionBookResponse,
  type OptionRule,
  type OptionSealedRun,
  anchorRange,
  cadenceLabel,
  dteRange,
  liveSizingLabel,
  optionSealedVerdict,
  researchedSizingLabel,
} from "@/lib/option-book"
import { clock } from "@/lib/status"

export function OptionBookCard({ optionBook }: { optionBook: OptionBookResponse | null }) {
  if (optionBook === null) return null

  if (!optionBook.available) {
    return (
      <Alert variant="destructive">
        <AlertTriangle />
        <AlertTitle>No options will be traded</AlertTitle>
        <AlertDescription>
          <p>
            The agent has no rule it is allowed to follow, so it will keep
            watching and never place an order. This is a setup problem, not a
            market one:
          </p>
          <ul className="mt-2 flex list-disc flex-col gap-1 pl-4">
            {optionBook.reasons.map((reason) => (
              <li key={reason}>{reason}</li>
            ))}
          </ul>
        </AlertDescription>
      </Alert>
    )
  }

  const { rule, sealed } = optionBook

  return (
    <Card>
      <CardHeader>
        <CardDescription className="flex flex-wrap items-center gap-2">
          <FlaskConical className="size-3.5" />
          the one options rule this agent is allowed to trade
          <Badge variant="outline">trades {optionBook.underlying}</Badge>
        </CardDescription>
        <CardTitle className="font-mono text-base">
          {optionBook.name}{" "}
          <span className="text-muted-foreground">[{optionBook.fingerprint}]</span>
        </CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        <p className="text-muted-foreground text-sm">{optionBook.hypothesis}</p>

        <RuleGrid rule={rule} />

        <Separator />

        <SealedFigures sealed={sealed} />

        <Separator />

        <dl className="grid gap-x-6 gap-y-2 text-sm sm:grid-cols-2">
          <Row term="when it enters" detail={rule.entry_expression} mono />
          <Row
            term="how it tested"
            detail={optionSealedVerdict(sealed)}
            tone={sealed.refuted ? "bad" : "neutral"}
          />
          <Row term="size, in testing" detail={researchedSizingLabel(rule)} />
          <Row term="size, live" detail={liveSizingLabel(rule)} />
          <Row term="rule written on" detail={optionBook.as_of} />
          <Row term="last rebuilt" detail={clock(optionBook.generated_at)} />
          <Row
            term="ideas tried"
            detail={`${optionBook.campaign_hypotheses} in the search that found this one, ${optionBook.distinct_hypotheses} ever. The more ideas you try, the more likely one looks good by luck — so this number is held against the result, not hidden from it.`}
          />
          <Row term="why this one" detail={optionBook.selection_rule} />
          <Row term="how it exits" detail={optionBook.exit_convention} />
          <Row term="data it was tested on" detail={optionBook.dataset_version} />
        </dl>

        <p className="text-muted-foreground border-l-2 pl-3 text-xs">
          <strong>Read the test result above before the numbers.</strong> This
          rule was tried once against two years of market data that were locked
          away while it was being designed. It came through without being
          disproved — but only about 32 separate trades fit into those two
          years, which is far too few to call it a winner. A test this size can
          tell you a rule is broken; it can never tell you a rule works. The
          agent trades it on that basis, and nothing here should be read as a
          promise that it will make money.
        </p>
      </CardContent>
    </Card>
  )
}

function RuleGrid({ rule }: { rule: OptionRule }) {
  const cells: [string, string, string][] = [
    ["what it trades", rule.structure.replace(/_/g, " "), "the shape of the position"],
    ["time to expiry", dteRange(rule), "how far out the contracts are"],
    ["strike it sells", anchorRange(rule), "further from today's price = safer, less paid"],
    ["strike it buys", rule.width_delta.toFixed(2), "the far leg that caps the loss"],
    ["how often", cadenceLabel(rule), ""],
    ["open at once", String(rule.max_concurrent), "hard ceiling"],
  ]
  return (
    <div className="grid gap-4 sm:grid-cols-3 lg:grid-cols-6">
      {cells.map(([label, value, hint]) => (
        <div key={label}>
          <p className="text-muted-foreground text-xs">{label}</p>
          <p className="text-sm tabular-nums">{value}</p>
          {hint ? <p className="text-muted-foreground text-xs">{hint}</p> : null}
        </div>
      ))}
    </div>
  )
}

/**
 * The sealed-run figures, rendered and never acted on.
 *
 * `t(alpha)` deliberately carries no colour tone here, unlike the equity tab's
 * equivalent figure. This sleeve's own sealed run did not clear its bar, and
 * even where a future book's did, a green "up" tint on the number is exactly
 * the kind of pass-shaped signal the honesty requirement forbids — the word
 * "not refuted" has to do all the work, not the colour around it.
 */
function SealedFigures({ sealed }: { sealed: OptionSealedRun }) {
  const figures: [string, string, string][] = [
    [
      "return above the market",
      `${(sealed.alpha * 100).toFixed(2)}%/yr`,
      "in the locked-away years",
    ],
    ["market exposure", sealed.beta.toFixed(2), "1.00 = moves with the market"],
    [
      "confidence score",
      sealed.t_alpha.toFixed(2),
      `${sealed.significance_bar.toFixed(2)} needed to mean much`,
    ],
    ["worst drop", `${(sealed.max_drawdown * 100).toFixed(2)}%`, "peak to trough"],
    [
      "trades in the test",
      String(sealed.trades),
      "many overlapped, so they count for less",
    ],
    ["rules tested this way", String(sealed.looks), "each one raises the bar"],
  ]
  return (
    <div className="grid gap-4 sm:grid-cols-3 lg:grid-cols-6">
      {figures.map(([label, value, hint]) => (
        <div key={label}>
          <p className="text-muted-foreground text-xs">{label}</p>
          <p className="text-lg tabular-nums">{value}</p>
          <p className="text-muted-foreground text-xs">{hint}</p>
        </div>
      ))}
    </div>
  )
}

function Row({
  term,
  detail,
  tone,
  mono,
}: {
  term: string
  detail: string
  tone?: "bad" | "neutral"
  mono?: boolean
}) {
  return (
    <div className="flex gap-2">
      <dt className="text-muted-foreground w-40 shrink-0">{term}</dt>
      <dd className={tone === "bad" ? "text-destructive" : mono ? "font-mono" : undefined}>
        {detail}
      </dd>
    </div>
  )
}
