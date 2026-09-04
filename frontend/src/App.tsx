/**
 * AlphaGate dashboard.
 *
 * Three tabs, answering three questions the others structurally cannot:
 *
 * * **Live** — what the options agent is doing now. Read from `status.json`,
 *   which it rewrites every slot.
 * * **Equity** — the strategy `ai_quant_researcher` validated, being held. Read
 *   from `equity-status.json`, rewritten every thirty-second heartbeat, with
 *   the sealed out-of-sample numbers that earned it the account.
 * * **Journal** — what either decided, and why. Read from the append-only JSONL
 *   the submission ships.
 *
 * The page polls rather than holding a socket. Neither cadence needs streaming,
 * a poll survives an agent restarting underneath it, and the staleness of the
 * last snapshot is then a fact the page can *see* rather than a connection
 * state it has to infer.
 *
 * Two status routes rather than one, because they are two processes. The page
 * can then say "options running, equity stopped" instead of guessing which one
 * last wrote a merged document.
 */

import { useCallback, useEffect, useState } from "react"
import { RefreshCw, TriangleAlert } from "lucide-react"

import { EquityStatus } from "@/components/equity-status"
import { Journal } from "@/components/journal"
import { LiveStatus } from "@/components/live-status"
import { SleevesOverview } from "@/components/sleeves-overview"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Skeleton } from "@/components/ui/skeleton"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import type { EquityCycle, EquityStatusResponse } from "@/lib/equity"
import type { OptionBookResponse } from "@/lib/option-book"
import type { SleevesResponse } from "@/lib/sleeves"
import type { DayRecord, StatusResponse } from "@/lib/status"

const POLL_MS = 15_000

export default function App() {
  const [status, setStatus] = useState<StatusResponse | null>(null)
  const [equity, setEquity] = useState<EquityStatusResponse | null>(null)
  const [equityCycles, setEquityCycles] = useState<EquityCycle[] | null>(null)
  const [cycles, setCycles] = useState<DayRecord[] | null>(null)
  const [days, setDays] = useState<string[]>([])
  const [error, setError] = useState<string | null>(null)
  const [optionBook, setOptionBook] = useState<OptionBookResponse | null>(null)
  const [sleeves, setSleeves] = useState<SleevesResponse | null>(null)

  // The day the user clicked, or `null` for "follow the agent". Storing the
  // *choice* rather than the resolved day means nothing has to write state
  // inside an effect to keep them in sync — the first version did, and it
  // both tripped the react-hooks lint and would have fought a user who
  // selected a day while a poll was in flight.
  const [chosenDay, setChosenDay] = useState<string | null>(null)

  const refresh = useCallback(async () => {
    try {
      const responses = await Promise.all([
        fetch("/api/status"),
        fetch("/api/equity/status"),
        fetch("/api/days"),
        fetch("/api/option-book"),
        fetch("/api/sleeves"),
      ])
      // A 500 still parses as JSON on some paths, so an unchecked `.json()`
      // would quietly install an error object as though it were a snapshot and
      // the page would render nonsense with a confident face. Refuse the whole
      // round instead: a stale page next to a visible warning is honest, and a
      // fresh-looking page built from an error is not.
      const failed = responses.find((response) => !response.ok)
      if (failed) {
        throw new Error(
          `The dashboard server answered ${failed.status} for ${new URL(failed.url).pathname}.`,
        )
      }
      const [status, equity, days, book, sleeves] = await Promise.all(
        responses.map((response) => response.json()),
      )
      setStatus(status)
      setEquity(equity)
      setDays(days)
      setOptionBook(book)
      setSleeves(sleeves)
      setError(null)
    } catch (cause) {
      // Browsers word a dead connection as "Failed to fetch" / "NetworkError",
      // which tells a reader nothing about what to do. Say what broke and what
      // to try; keep the original wording only when it is our own message.
      const raw = cause instanceof Error ? cause.message : String(cause)
      setError(
        /fetch|network|load failed/i.test(raw)
          ? "Cannot reach the dashboard server. It has probably stopped — restart it with `alphagate serve`. The figures below are the last ones received."
          : `${raw} The figures below are the last ones received.`,
      )
    }
  }, [])

  useEffect(() => {
    // `refresh` is async and awaits both fetches before touching state, so
    // nothing is set during the effect's synchronous phase — but the rule
    // cannot see through the await, and the directive has to sit on the call
    // itself rather than the line above it.
    void refresh() // eslint-disable-line react-hooks/set-state-in-effect
    const timer = setInterval(() => void refresh(), POLL_MS)
    return () => clearInterval(timer)
  }, [refresh])

  // The agent's session day, not the browser's date: a machine in another
  // timezone would otherwise open on an empty page just after midnight UTC.
  const day =
    chosenDay ??
    status?.snapshot?.session_day ??
    equity?.snapshot?.session_day ??
    days[0] ??
    todayInBrowser()

  // Refetch when the day changes, or when the agent has actually done
  // something — keyed on the snapshot's timestamp rather than the whole
  // response, so a poll that returns identical data does not refetch the day.
  const lastCycle = status?.snapshot?.as_of ?? null
  const lastBeat = equity?.snapshot?.as_of ?? null

  useEffect(() => {
    let cancelled = false
    fetch(`/api/day/${day}`)
      .then((response) => response.json())
      .then((loaded: DayRecord[]) => {
        if (!cancelled) setCycles(loaded)
      })
      .catch(() => {
        if (!cancelled) setCycles([])
      })
    return () => {
      cancelled = true
    }
  }, [day, lastCycle])

  // The equity day is fetched separately and keyed on the heartbeat, so a
  // thirty-second beat refreshes today's passes without also refetching the
  // options journal every time.
  useEffect(() => {
    let cancelled = false
    fetch(`/api/equity/day/${day}`)
      .then((response) => response.json())
      .then((loaded: EquityCycle[]) => {
        if (!cancelled) setEquityCycles(loaded)
      })
      .catch(() => {
        if (!cancelled) setEquityCycles([])
      })
    return () => {
      cancelled = true
    }
  }, [day, lastBeat])

  return (
    <div className="bg-background text-foreground min-h-screen">
      <header className="border-border flex flex-wrap items-baseline gap-x-4 gap-y-2 border-b px-6 py-4">
        <h1 className="text-lg font-semibold tracking-tight">AlphaGate</h1>
        <p className="text-muted-foreground text-sm">
          agents that can be overruled
        </p>
        <div className="ml-auto flex items-center gap-2">
          {error ? <Badge variant="destructive">disconnected</Badge> : null}
          <Button variant="outline" size="sm" onClick={() => void refresh()}>
            <RefreshCw data-icon="inline-start" />
            refresh
          </Button>
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-6 py-6">
        {/* A badge alone was too easy to miss, and "everything on this page is
            stale" is the one thing a reader must not miss. */}
        {error ? (
          <Alert variant="destructive" className="mb-6">
            <TriangleAlert />
            <AlertTitle>This page is not updating</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        ) : null}

        <div className="mb-6">
          <SleevesOverview sleeves={sleeves} />
        </div>

        <Tabs defaultValue="live">
          <TabsList>
            <TabsTrigger value="live">Options</TabsTrigger>
            <TabsTrigger value="equity">Equity</TabsTrigger>
            <TabsTrigger value="journal">Journal</TabsTrigger>
          </TabsList>

          <TabsContent value="live" className="pt-6">
            {status === null ? (
              <Loading />
            ) : (
              <LiveStatus status={status} optionBook={optionBook} />
            )}
          </TabsContent>

          <TabsContent value="equity" className="pt-6">
            {equity === null ? (
              <Loading />
            ) : (
              <EquityStatus status={equity} cycles={equityCycles} />
            )}
          </TabsContent>

          <TabsContent value="journal" className="pt-6">
            <div className="mb-4 flex flex-wrap items-center gap-2">
              {days.map((available) => (
                <Button
                  key={available}
                  size="sm"
                  variant={available === day ? "default" : "outline"}
                  onClick={() => setChosenDay(available)}
                >
                  {available}
                </Button>
              ))}
            </div>
            {cycles === null ? <Loading /> : <Journal cycles={cycles} day={day} />}
          </TabsContent>
        </Tabs>
      </main>
    </div>
  )
}

function todayInBrowser(): string {
  return new Date().toISOString().slice(0, 10)
}


function Loading() {
  return (
    <div className="flex flex-col gap-4">
      <Skeleton className="h-8 w-64" />
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {[0, 1, 2, 3].map((slot) => (
          <Skeleton key={slot} className="h-28" />
        ))}
      </div>
      <Skeleton className="h-64" />
    </div>
  )
}
