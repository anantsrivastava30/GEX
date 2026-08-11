"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import EarningsTable from "@/components/calendar/EarningsTable";
import EconomicReleaseList from "@/components/calendar/EconomicReleaseList";
import Panel from "@/components/ui/Panel";
import {
  api,
  CalendarResponse,
  EarningsEvent,
  EarningsScope,
  EconomicRelease,
} from "@/lib/api";

// Forward-looking by default: the next 7 days sit on top, the last 7 days sit
// under them. One request covers both halves and the page splits it on today.

const WINDOWS = [7, 14, 30];

// Nasdaq's market-wide feed is thousands of names per fortnight. Focus is the
// default: the companies this app already tracks, which is the sector-ETF
// holdings (semis, tech, and the rest) plus the snapshot and watchlist tickers.
const SCOPES: { id: EarningsScope; label: string; hint: string }[] = [
  { id: "focus", label: "Focus", hint: "Tracked sector leaders" },
  { id: "large", label: "Large cap", hint: "Market cap over $10B" },
  { id: "all", label: "All", hint: "Every Nasdaq filer" },
];

function isoDate(date: Date): string {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

function dayOffset(isoValue: string, todayIso: string): number {
  const target = new Date(`${isoValue.slice(0, 10)}T00:00:00`).getTime();
  const today = new Date(`${todayIso}T00:00:00`).getTime();
  if (Number.isNaN(target) || Number.isNaN(today)) return 0;
  return Math.round((target - today) / 86_400_000);
}

function relativeDay(isoValue: string, todayIso: string): string {
  const offset = dayOffset(isoValue, todayIso);
  if (offset === 0) return "today";
  if (offset === 1) return "tomorrow";
  if (offset === -1) return "yesterday";
  return offset > 0 ? `in ${offset}d` : `${Math.abs(offset)}d ago`;
}

export default function CalendarPage() {
  const controller = useRef<AbortController | null>(null);
  const [days, setDays] = useState(7);
  const [scope, setScope] = useState<EarningsScope>("focus");
  const [group, setGroup] = useState<string | null>(null);
  const [symbolInput, setSymbolInput] = useState("");
  const [symbols, setSymbols] = useState<string[]>([]);
  const [data, setData] = useState<CalendarResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  // Pinned once per load so the upcoming/past split cannot drift mid-render.
  const [todayIso, setTodayIso] = useState(() => isoDate(new Date()));

  const load = useCallback(
    (
      windowDays: number,
      selectedSymbols: string[],
      selectedScope: EarningsScope,
    ) => {
    controller.current?.abort();
    const nextController = new AbortController();
    controller.current = nextController;
    const today = new Date();
    setTodayIso(isoDate(today));
    const start = new Date(today);
    start.setDate(start.getDate() - windowDays);
    const end = new Date(today);
    end.setDate(end.getDate() + windowDays);
    api
      .calendar(
        {
          start: isoDate(start),
          end: isoDate(end),
          symbols: selectedSymbols.length ? selectedSymbols : undefined,
          scope: selectedScope,
        },
        nextController.signal,
      )
      .then((response) => {
        if (controller.current === nextController) setData(response);
      })
      .catch((cause) => {
        if (cause instanceof DOMException && cause.name === "AbortError") return;
        if (controller.current === nextController)
          setError(cause instanceof Error ? cause.message : String(cause));
      })
      .finally(() => {
        if (controller.current === nextController) {
          controller.current = null;
          setLoading(false);
        }
      });
    },
    [],
  );

  useEffect(() => {
    load(days, symbols, scope);
    return () => controller.current?.abort();
  }, [days, symbols, scope, load]);

  function applySymbols() {
    const next = [
      ...new Set(
        symbolInput
          .split(",")
          .map((symbol) => symbol.trim().toUpperCase())
          .filter(Boolean),
      ),
    ].slice(0, 25);
    setLoading(true);
    setError(null);
    setData(null);
    if (next.join(",") === symbols.join(",")) load(days, next, scope);
    else setSymbols(next);
  }

  function reload(apply: () => void) {
    setLoading(true);
    setError(null);
    setData(null);
    apply();
  }

  // Sector chips come from what actually reported in the window, busiest first,
  // so "who in semis reports this week" is one click.
  const groupCounts = useMemo(() => {
    const counts = new Map<string, number>();
    for (const event of data?.earnings ?? []) {
      if (!event.group) continue;
      counts.set(event.group, (counts.get(event.group) ?? 0) + 1);
    }
    return [...counts.entries()].sort(
      (a, b) => b[1] - a[1] || a[0].localeCompare(b[0]),
    );
  }, [data]);

  const split = useMemo(() => {
    const releases = data?.economic_releases ?? [];
    const earnings = (data?.earnings ?? []).filter(
      (event) => !group || event.group === group,
    );
    const isUpcoming = (value: string) => value.slice(0, 10) >= todayIso;
    return {
      upcomingReleases: releases
        .filter((item: EconomicRelease) => isUpcoming(item.release_date))
        .sort((a, b) => a.release_date.localeCompare(b.release_date)),
      pastReleases: releases
        .filter((item: EconomicRelease) => !isUpcoming(item.release_date))
        .sort((a, b) => b.release_date.localeCompare(a.release_date)),
      upcomingEarnings: earnings
        .filter((item: EarningsEvent) => isUpcoming(item.earnings_at))
        .sort((a, b) => a.earnings_at.localeCompare(b.earnings_at)),
      pastEarnings: earnings
        .filter((item: EarningsEvent) => !isUpcoming(item.earnings_at))
        .sort((a, b) => b.earnings_at.localeCompare(a.earnings_at)),
    };
  }, [data, todayIso, group]);

  const label = useCallback(
    (isoValue: string) => relativeDay(isoValue, todayIso),
    [todayIso],
  );

  function section(
    heading: string,
    caption: string,
    releases: EconomicRelease[],
    earnings: EarningsEvent[],
    releaseEmpty: string,
    earningsEmpty: string,
  ) {
    return (
      <section className="space-y-3">
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <h2 className="text-sm font-semibold uppercase tracking-wide text-foreground">
            {heading}
          </h2>
          <span className="text-xs text-muted">{caption}</span>
        </div>
        <div className="grid gap-4 xl:grid-cols-[minmax(0,1.15fr)_minmax(0,1fr)]">
          <Panel
            title="US economic releases"
            right={
              <span className="font-mono text-xs text-muted">
                {loading ? "..." : `${releases.length}`}
              </span>
            }
            bodyClassName="p-0"
          >
            <div className="max-h-[30rem] overflow-auto">
              <EconomicReleaseList
                releases={releases}
                loading={loading}
                emptyMessage={releaseEmpty}
                relativeLabel={label}
              />
            </div>
          </Panel>
          <Panel
            title="Earnings"
            right={
              <span className="font-mono text-xs text-muted">
                {loading ? "..." : `${earnings.length}`}
              </span>
            }
            bodyClassName="p-0"
          >
            <EarningsTable
              events={earnings}
              loading={loading}
              emptyMessage={earningsEmpty}
            />
          </Panel>
        </div>
      </section>
    );
  }

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-lg font-semibold">Market Calendar</h1>
        <p className="mt-1 max-w-3xl text-sm text-muted">
          Company earnings from Nasdaq and a curated US macro release schedule
          from FRED. Earnings default to the companies this app tracks - the
          holdings of every tracked sector ETF, semis and tech included, plus
          the snapshot and watchlist tickers - so the list stays readable;
          switch to Large cap or All to widen it. Each macro row carries its
          headline reading:
          what printed, the prior period, and the move. FRED publishes no
          consensus, so these are moves against the prior period, not beats
          or misses against expectations.
        </p>
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <span className="text-xs text-muted">Companies</span>
        {SCOPES.map((option) => (
          <button
            key={option.id}
            disabled={loading}
            title={option.hint}
            aria-pressed={scope === option.id}
            onClick={() => {
              if (option.id === scope) return;
              reload(() => {
                setGroup(null);
                setScope(option.id);
              });
            }}
            className={`rounded-full border px-3 py-1 text-xs disabled:opacity-50 ${
              scope === option.id
                ? "border-accent bg-accent/15 text-foreground"
                : "border-border bg-surface text-muted hover:text-foreground"
            }`}
          >
            {option.label}
          </button>
        ))}
        <span className="mx-1 h-4 w-px bg-border" />
        <span className="text-xs text-muted">Window</span>
        {WINDOWS.map((windowDays) => (
          <button
            key={windowDays}
            disabled={loading}
            aria-pressed={days === windowDays}
            onClick={() => {
              if (windowDays === days) return;
              reload(() => setDays(windowDays));
            }}
            className={`rounded-full border px-3 py-1 text-xs disabled:opacity-50 ${
              days === windowDays
                ? "border-accent bg-accent/15 text-foreground"
                : "border-border bg-surface text-muted hover:text-foreground"
            }`}
          >
            ±{windowDays}d
          </button>
        ))}
        <form
          onSubmit={(event) => {
            event.preventDefault();
            applySymbols();
          }}
          className="ml-2 flex gap-2"
        >
          <input
            value={symbolInput}
            onChange={(event) => setSymbolInput(event.target.value.toUpperCase())}
            placeholder="Optional: AAPL, NVDA"
            aria-label="Optional earnings ticker filter"
            className="w-52 rounded border border-border bg-surface px-2 py-1 text-sm outline-none focus:border-accent"
          />
          <button
            type="submit"
            disabled={loading}
            className="rounded border border-accent bg-accent/15 px-3 py-1 text-xs disabled:opacity-50"
          >
            {loading ? "Loading calendar..." : "Apply ticker filter"}
          </button>
        </form>
      </div>

      {groupCounts.length > 1 && (
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-xs text-muted">Sector</span>
          <button
            type="button"
            aria-pressed={group === null}
            onClick={() => setGroup(null)}
            className={`rounded-full border px-2.5 py-1 text-xs ${
              group === null
                ? "border-accent bg-accent/15 text-foreground"
                : "border-border bg-surface text-muted hover:text-foreground"
            }`}
          >
            All
          </button>
          {groupCounts.map(([name, count]) => (
            <button
              key={name}
              type="button"
              aria-pressed={group === name}
              onClick={() => setGroup(group === name ? null : name)}
              className={`rounded-full border px-2.5 py-1 text-xs ${
                group === name
                  ? "border-accent bg-accent/15 text-foreground"
                  : "border-border bg-surface text-muted hover:text-foreground"
              }`}
            >
              {name}
              <span className="ml-1 text-faint">{count}</span>
            </button>
          ))}
        </div>
      )}

      {data && (
        <div className="space-y-2 text-xs">
          <div className="flex flex-wrap gap-2">
            <span className="rounded border border-border px-2 py-1 text-muted">
              {data.earnings.length.toLocaleString()} of{" "}
              {data.earnings_total.toLocaleString()} reporting companies
              {data.earnings_scope === "focus"
                ? " · tracked sector leaders"
                : data.earnings_scope === "large"
                  ? " · market cap over $10B"
                  : ""}
            </span>
            {Object.entries(data.sources).map(([name, source]) => (
              <span
                key={name}
                title={source.message ?? undefined}
                className={`rounded border px-2 py-1 ${
                  source.available && !source.partial
                    ? "border-positive/50 text-positive"
                    : source.partial
                      ? "border-warning/50 text-warning"
                      : "border-negative/50 text-negative"
                }`}
              >
                {name}:{" "}
                {source.available
                  ? source.partial
                    ? "partial"
                    : "available"
                  : source.configured
                    ? "unavailable"
                    : "not configured"}
              </span>
            ))}
            <span className="rounded border border-border px-2 py-1 font-mono text-muted">
              {data.start_date} to {data.end_date}
            </span>
          </div>
          {Object.entries(data.sources)
            .filter(([, source]) => source.message)
            .map(([name, source]) => (
              <p key={name} className="text-muted">
                <span className="uppercase text-faint">{name}:</span> {source.message}
              </p>
            ))}
        </div>
      )}

      {error && (
        <p className="rounded border border-negative/50 bg-surface px-4 py-3 text-sm text-negative">
          Unable to load calendar: {error}
        </p>
      )}

      {section(
        "Upcoming",
        `Next ${days} days`,
        split.upcomingReleases,
        split.upcomingEarnings,
        "No curated macro releases scheduled in this window.",
        group
          ? `No ${group} names report in this window.`
          : "No tracked companies report in this window. Widen the window or switch to All.",
      )}

      {section(
        "Past",
        `Last ${days} days`,
        split.pastReleases,
        split.pastEarnings,
        "No curated macro releases printed in this window.",
        group
          ? `No ${group} names reported in this window.`
          : "No tracked companies reported in this window. Widen the window or switch to All.",
      )}

      <p className="max-w-4xl text-xs text-muted">
        Earnings use Nasdaq&apos;s market-wide daily calendar; the ticker input is
        an optional filter. The Focus scope keeps names held by the tracked
        sector ETFs (read from the provider daily, with a configured fallback)
        plus the snapshot and watchlist tickers, and tags each with the sector
        it was found in; holdings drift, so a name can enter or leave this list
        over time. Macro readings come from FRED vintages: a past release
        shows the value that release first published against the previous period,
        and an upcoming release shows the last published reading for context.
        FRED exposes no consensus, actual-vs-estimate, or impact rating, and the
        configured FMP tier does not include its economic calendar, so nothing
        here is a beat or miss. Green and red only mark the direction a move is
        conventionally read as good or bad. FOMC announcements are omitted
        because FRED lists that release on every calendar day rather than on
        meeting dates. The listed releases are project-selected US macro series.
      </p>
    </div>
  );
}
