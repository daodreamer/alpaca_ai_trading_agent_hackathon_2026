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
      <Alert>
        <AlertTriangle />
        <AlertTitle>The pinned option rule is not being executed</AlertTitle>
        <AlertDescription>
          <ul className="flex flex-col gap-1">
            {optionBook.reasons.map((reason) => (
              <li key={reason}>{reason}</li>
            ))}
          </ul>
          {optionBook.can_refute_not_confirm ? (
            <p className="mt-2 italic">{optionBook.can_refute_not_confirm}</p>
          ) : null}
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
          the option rule this sleeve executes
          <Badge variant="outline">{optionBook.status}</Badge>
          <Badge variant="outline">{optionBook.underlying}</Badge>
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
          <Row term="entry rule" detail={rule.entry_expression} mono />
          <Row
            term="verdict"
            detail={optionSealedVerdict(sealed)}
            tone={sealed.refuted ? "bad" : "neutral"}
          />
          <Row term="sizing — researched" detail={researchedSizingLabel(rule)} />
          <Row term="sizing — live" detail={liveSizingLabel(rule)} />
          <Row term="book as of" detail={optionBook.as_of} />
          <Row term="generated" detail={clock(optionBook.generated_at)} />
          <Row
            term="hypotheses searched"
            detail={`${optionBook.distinct_hypotheses} distinct, ${optionBook.campaign_hypotheses} this campaign — the multiplicity denominator`}
          />
          <Row term="selection rule" detail={optionBook.selection_rule} />
          <Row term="exit convention" detail={optionBook.exit_convention} />
          <Row term="dataset" detail={optionBook.dataset_version} />
        </dl>

        <p className="text-muted-foreground border-l-2 pl-3 text-xs italic">
          The two sizing figures above are deliberately made to agree in
          dollars: {liveSizingLabel(rule)} was chosen so that this sleeve's
          live per-trade budget lands on exactly the fraction the sealed run
          measured. They come from different places and nothing on this side
          reads the other — <code>agent/sizing.py</code> never consults{" "}
          <code>risk_per_trade</code>; the agreement is a property of how the
          sleeve was sized, not a live calculation.
        </p>

        <p className="text-muted-foreground border-l-2 pl-3 text-xs italic">
          {optionBook.can_refute_not_confirm ||
            `The sealed window can refute and cannot confirm: t=${sealed.t_alpha.toFixed(2)} against a bar of ${sealed.significance_bar.toFixed(2)} over ${sealed.observations} sessions is not the same claim as a passed test.`}
        </p>
      </CardContent>
    </Card>
  )
}

function RuleGrid({ rule }: { rule: OptionRule }) {
  const cells: [string, string][] = [
    ["structure", rule.structure.replace(/_/g, " ")],
    ["dte", dteRange(rule)],
    ["anchor delta", anchorRange(rule)],
    ["width delta", rule.width_delta.toFixed(2)],
    ["cadence", cadenceLabel(rule)],
    ["max concurrent", String(rule.max_concurrent)],
  ]
  return (
    <div className="grid gap-4 sm:grid-cols-3 lg:grid-cols-6">
      {cells.map(([label, value]) => (
        <div key={label}>
          <p className="text-muted-foreground text-xs">{label}</p>
          <p className="text-sm tabular-nums">{value}</p>
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
  const figures: [string, string][] = [
    ["alpha", `${(sealed.alpha * 100).toFixed(2)}%/yr`],
    ["beta", sealed.beta.toFixed(2)],
    ["t(alpha)", `${sealed.t_alpha >= 0 ? "+" : ""}${sealed.t_alpha.toFixed(2)}`],
    ["bar", sealed.significance_bar.toFixed(2)],
    ["sealed trades", String(sealed.trades)],
    ["looks", String(sealed.looks)],
  ]
  return (
    <div className="grid gap-4 sm:grid-cols-3 lg:grid-cols-6">
      {figures.map(([label, value]) => (
        <div key={label}>
          <p className="text-muted-foreground text-xs">{label}</p>
          <p className="text-lg tabular-nums">{value}</p>
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
