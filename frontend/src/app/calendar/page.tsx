"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Panel from "@/components/ui/Panel";
import { api, CalendarResponse } from "@/lib/api";

const WINDOWS = [7, 14, 30];
type CalendarDirection = "past" | "upcoming";

function isoDate(date: Date): string {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

function formatNumber(value?: number | null): string {
  return value == null ? "-" : value.toFixed(2);
}

function formatEarningsDate(value: string, session?: string | null): string {
  if (value.length === 10) {
    const [year, month, day] = value.split("-");
    const date = new Date(Number(year), Number(month) - 1, Number(day));
    return `${date.toLocaleDateString([], { month: "short", day: "numeric" })} · ${session || "time TBD"}`;
  }
  return new Date(value).toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

export default function CalendarPage() {
  const controller = useRef<AbortController | null>(null);
  const [days, setDays] = useState(14);
  const [direction, setDirection] = useState<CalendarDirection>("past");
  const [symbolInput, setSymbolInput] = useState("");
  const [symbols, setSymbols] = useState<string[]>([]);
  const [data, setData] = useState<CalendarResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const load = useCallback((windowDays: number, selectedSymbols: string[], windowDirection: CalendarDirection) => {
    controller.current?.abort();
    const nextController = new AbortController();
    controller.current = nextController;
    const today = new Date();
    const start = new Date(today);
    const end = new Date(today);
    if (windowDirection === "past") {
      start.setDate(start.getDate() - windowDays + 1);
    } else {
      end.setDate(end.getDate() + windowDays - 1);
    }
    api.calendar(
      { start: isoDate(start), end: isoDate(end), symbols: selectedSymbols.length ? selectedSymbols : undefined },
      nextController.signal,
    ).then((response) => {
      if (controller.current === nextController) setData(response);
    }).catch((cause) => {
      if (cause instanceof DOMException && cause.name === "AbortError") return;
      if (controller.current === nextController) setError(cause instanceof Error ? cause.message : String(cause));
    }).finally(() => {
      if (controller.current === nextController) {
        controller.current = null;
        setLoading(false);
      }
    });
  }, []);

  useEffect(() => {
    load(days, symbols, direction);
    return () => controller.current?.abort();
  }, [days, symbols, direction, load]);

  function applySymbols() {
    const next = [...new Set(symbolInput.split(",").map((symbol) => symbol.trim().toUpperCase()).filter(Boolean))].slice(0, 25);
    setLoading(true);
    setError(null);
    setData(null);
    if (next.join(",") === symbols.join(",")) load(days, next, direction);
    else setSymbols(next);
  }

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-lg font-semibold">Market Calendar</h1>
        <p className="mt-1 max-w-3xl text-sm text-muted">
          Market-wide company earnings from Nasdaq and a curated US macro release schedule from FRED.
        </p>
      </div>

      <div className="flex flex-wrap items-center gap-2">
        {(["past", "upcoming"] as CalendarDirection[]).map((windowDirection) => (
          <button key={windowDirection} disabled={loading} aria-pressed={direction === windowDirection} onClick={() => { if (windowDirection === direction) return; setLoading(true); setError(null); setData(null); setDirection(windowDirection); }} className={`rounded-full border px-3 py-1 text-xs capitalize disabled:opacity-50 ${direction === windowDirection ? "border-warning bg-warning/10 text-warning" : "border-border bg-surface text-muted hover:text-foreground"}`}>
            {windowDirection}
          </button>
        ))}
        <span className="mx-1 h-4 w-px bg-border" />
        {WINDOWS.map((windowDays) => (
          <button key={windowDays} disabled={loading} aria-pressed={days === windowDays} onClick={() => { if (windowDays === days) return; setLoading(true); setError(null); setData(null); setDays(windowDays); }} className={`rounded-full border px-3 py-1 text-xs disabled:opacity-50 ${days === windowDays ? "border-accent bg-accent/15 text-foreground" : "border-border bg-surface text-muted hover:text-foreground"}`}>
            {windowDays}d
          </button>
        ))}
        <form onSubmit={(event) => { event.preventDefault(); applySymbols(); }} className="ml-2 flex gap-2">
          <input value={symbolInput} onChange={(event) => setSymbolInput(event.target.value.toUpperCase())} placeholder="Optional: AAPL, NVDA" aria-label="Optional earnings ticker filter" className="w-52 rounded border border-border bg-surface px-2 py-1 text-sm outline-none focus:border-accent" />
          <button type="submit" disabled={loading} className="rounded border border-accent bg-accent/15 px-3 py-1 text-xs disabled:opacity-50">{loading ? "Loading calendar..." : "Apply ticker filter"}</button>
        </form>
      </div>
      <p className="text-xs text-muted">Direction and date range load automatically. The ticker filter changes only after you apply it; leave the field blank for the market-wide calendar.</p>

      {data && (
        <div className="space-y-2 text-xs">
          <div className="flex flex-wrap gap-2">
          {Object.entries(data.sources).map(([name, source]) => (
            <span key={name} title={source.message ?? undefined} className={`rounded border px-2 py-1 ${source.available && !source.partial ? "border-emerald-500/50 text-emerald-300" : source.partial ? "border-amber-500/50 text-amber-300" : "border-rose-500/50 text-rose-300"}`}>
              {name}: {source.available ? source.partial ? "partial" : "available" : source.configured ? "unavailable" : "not configured"}
            </span>
          ))}
          <span className="rounded border border-border px-2 py-1 font-mono text-muted">{data.start_date} to {data.end_date}</span>
          </div>
          {Object.entries(data.sources).filter(([, source]) => source.message).map(([name, source]) => <p key={name} className="text-muted"><span className="uppercase text-faint">{name}:</span> {source.message}</p>)}
        </div>
      )}

      {error && <p className="rounded border border-rose-500/50 bg-surface px-4 py-3 text-sm text-rose-300">Unable to load calendar: {error}</p>}

      <div className="grid gap-4 xl:grid-cols-2">
        <Panel title="Earnings" right={<span className="font-mono text-xs text-muted">{loading ? "..." : `${data?.earnings.length ?? 0} events`}</span>} bodyClassName="p-0">
          <div className="max-h-[36rem] overflow-auto">
            <table className="w-full min-w-[560px] text-sm">
              <thead className="sticky top-0 bg-surface-2 text-left text-xs text-muted"><tr><th className="px-3 py-2">Date</th><th className="px-3 py-2">Symbol</th><th className="px-3 py-2 text-right">EPS est.</th><th className="px-3 py-2 text-right">Reported</th><th className="px-3 py-2 text-right">Surprise</th></tr></thead>
              <tbody>{loading && <tr><td colSpan={5} className="px-4 py-10 text-center text-muted">Loading market-wide earnings...</td></tr>}{(data?.earnings ?? []).map((event) => <tr key={`${event.symbol}-${event.earnings_at}`} className="border-t border-border hover:bg-surface-hover"><td className="whitespace-nowrap px-3 py-2 font-mono text-muted">{formatEarningsDate(event.earnings_at, event.session)}</td><td className="px-3 py-2"><a href={event.url} target="_blank" rel="noreferrer" className="font-mono text-accent">{event.symbol}</a>{event.company_name && <span className="block max-w-56 truncate text-xs text-faint" title={event.company_name}>{event.company_name}</span>}</td><td className="px-3 py-2 text-right font-mono text-muted">{formatNumber(event.eps_estimate)}</td><td className="px-3 py-2 text-right font-mono text-muted">{formatNumber(event.reported_eps)}</td><td className="px-3 py-2 text-right font-mono text-muted">{event.surprise_pct == null ? "-" : `${event.surprise_pct.toFixed(1)}%`}</td></tr>)}{!loading && !error && data && !data.earnings.length && <tr><td colSpan={5} className="px-4 py-8 text-center text-muted">No market-wide earnings match this window and optional ticker filter.</td></tr>}</tbody>
            </table>
          </div>
        </Panel>

        <Panel title="US economic releases" right={<span className="font-mono text-xs text-muted">{loading ? "..." : `${data?.economic_releases.length ?? 0} releases`}</span>} bodyClassName="p-0">
          <div className="max-h-[36rem] overflow-auto">
            <table className="w-full text-sm">
              <thead className="sticky top-0 bg-surface-2 text-left text-xs text-muted"><tr><th className="px-3 py-2">Date</th><th className="px-3 py-2">Release</th></tr></thead>
              <tbody>{loading && <tr><td colSpan={2} className="px-4 py-10 text-center text-muted">Loading economic releases...</td></tr>}{(data?.economic_releases ?? []).map((event) => <tr key={`${event.release_id}-${event.release_date}`} className="border-t border-border hover:bg-surface-hover"><td className="whitespace-nowrap px-3 py-2 font-mono text-muted">{event.release_date}</td><td className="px-3 py-2"><a href={event.url} target="_blank" rel="noreferrer" className="text-foreground hover:text-accent">{event.release_name}</a></td></tr>)}{!loading && !error && data && !data.economic_releases.length && <tr><td colSpan={2} className="px-4 py-8 text-center text-muted">{data.sources.fred?.message ?? "No curated releases in this window."}</td></tr>}</tbody>
            </table>
          </div>
        </Panel>
      </div>

      <p className="max-w-4xl text-xs text-muted">
        Earnings use Nasdaq&apos;s market-wide daily calendar; ticker input is an optional filter. FRED provides release dates, not consensus, actual, prior, impact, or exact publication-time fields. The listed releases are project-selected US macro series, not FRED impact ratings.
      </p>
    </div>
  );
}
