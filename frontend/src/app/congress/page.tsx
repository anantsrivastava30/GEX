"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  api,
  CongressChamber,
  CongressTrade,
  CongressTradesResponse,
} from "@/lib/api";
import Panel from "@/components/ui/Panel";

// Congressional stock-trade disclosures from free public mirrors. These are
// periodic transaction reports, disclosed on a lag, so the page leads with the
// as-of date and a staleness badge and never implies a live feed.

const CHAMBERS: { id: CongressChamber | "all"; label: string }[] = [
  { id: "all", label: "Both" },
  { id: "senate", label: "Senate" },
  { id: "house", label: "House" },
];

const WINDOWS: { id: number; label: string }[] = [
  { id: 30, label: "30d" },
  { id: 90, label: "90d" },
  { id: 180, label: "180d" },
  { id: 365, label: "1y" },
];

function tradeTone(type: string): string {
  const t = type.toLowerCase();
  if (t === "buy") return "text-positive";
  if (t === "sell") return "text-negative";
  return "text-muted";
}

function tradeLabel(type: string): string {
  const t = type.toLowerCase();
  if (t === "buy") return "BUY";
  if (t === "sell") return "SELL";
  return type.toUpperCase();
}

export default function CongressPage() {
  const controller = useRef<AbortController | null>(null);
  const activeTicker = useRef("");
  const [ticker, setTicker] = useState("");
  const [chamber, setChamber] = useState<CongressChamber | "all">("all");
  const [days, setDays] = useState(90);
  const [data, setData] = useState<CongressTradesResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const fetchTrades = useCallback(
    (nextTicker: string, nextChamber: CongressChamber | "all", nextDays: number) => {
      controller.current?.abort();
      const nextController = new AbortController();
      controller.current = nextController;
      api
        .congressTrades(
          {
            ticker: nextTicker.trim() || undefined,
            chamber: nextChamber === "all" ? undefined : nextChamber,
            days: nextDays,
            limit: 300,
          },
          nextController.signal,
        )
        .then((response) => {
          if (controller.current === nextController) setData(response);
        })
        .catch((cause) => {
          if (cause instanceof DOMException && cause.name === "AbortError") return;
          if (controller.current === nextController) {
            setError(cause instanceof Error ? cause.message : String(cause));
          }
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
    fetchTrades(activeTicker.current, chamber, days);
    // Ticker applies on submit; chamber/window apply immediately via handlers.
  }, [chamber, days, fetchTrades]);

  function changeChamber(next: CongressChamber | "all") {
    if (next === chamber) return;
    setLoading(true);
    setError(null);
    setData(null);
    setChamber(next);
  }

  function changeWindow(next: number) {
    if (next === days) return;
    setLoading(true);
    setError(null);
    setData(null);
    setDays(next);
  }

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-lg font-semibold">Congress Trades</h1>
        <p className="mt-1 max-w-3xl text-sm text-muted">
          Senate and House stock-trade disclosures from free public mirrors.
          These are periodic transaction reports filed on a lag of up to 45
          days, not real-time trades.
        </p>
      </div>

      <div className="flex flex-wrap items-center gap-2">
        {CHAMBERS.map((c) => (
          <button
            key={c.id}
            onClick={() => changeChamber(c.id)}
            className={`rounded-full border px-3 py-1 text-xs transition-colors ${
              chamber === c.id
                ? "border-accent bg-accent/15 text-foreground"
                : "border-border bg-surface text-muted hover:text-foreground"
            }`}
          >
            {c.label}
          </button>
        ))}
        <span className="mx-1 h-4 w-px bg-border" />
        {WINDOWS.map((w) => (
          <button
            key={w.id}
            onClick={() => changeWindow(w.id)}
            className={`rounded-full border px-3 py-1 text-xs transition-colors ${
              days === w.id
                ? "border-accent bg-accent/15 text-foreground"
                : "border-border bg-surface text-muted hover:text-foreground"
            }`}
          >
            {w.label}
          </button>
        ))}
      </div>
      <p className="text-xs text-muted">Chamber and date range load automatically. The ticker filter changes only after you choose Apply.</p>

      <form
        onSubmit={(event) => {
          event.preventDefault();
          setLoading(true);
          setError(null);
          setData(null);
          activeTicker.current = ticker.trim().toUpperCase();
          fetchTrades(ticker, chamber, days);
        }}
        className="flex flex-wrap items-center gap-2"
      >
        <input
          value={ticker}
          onChange={(event) => setTicker(event.target.value.toUpperCase())}
          placeholder="Filter by ticker (e.g. NVDA)"
          aria-label="Filter by ticker"
          className="w-52 rounded border border-border bg-surface px-2 py-1 text-sm outline-none focus:border-accent"
        />
        <button
          type="submit"
          disabled={loading}
          className="rounded-md border border-accent bg-accent/15 px-3 py-1 text-sm hover:bg-accent/25 disabled:opacity-50"
        >
          {loading ? "Loading…" : "Apply"}
        </button>
        {ticker && (
          <button
            type="button"
            onClick={() => {
              setTicker("");
              activeTicker.current = "";
              setLoading(true);
              setError(null);
              setData(null);
              fetchTrades("", chamber, days);
            }}
            className="rounded-md border border-border px-3 py-1 text-sm text-muted hover:text-foreground"
          >
            Clear
          </button>
        )}
      </form>

      {data && (
        <div className="flex flex-wrap gap-2 text-xs">
          <span className="rounded border border-border bg-surface px-2 py-1 font-mono text-muted">
            Latest: {data.as_of ?? "unavailable"}
          </span>
          <span
            className={`rounded border px-2 py-1 ${
              data.stale
                ? "border-amber-500/50 text-amber-300"
                : "border-emerald-500/50 text-emerald-300"
            }`}
          >
            {data.stale ? "Feed stale or incomplete" : "Feed current"}
          </span>
          {data.unavailable_sources.length > 0 && (
            <span className="rounded border border-rose-500/50 px-2 py-1 text-rose-300">
              Unreachable: {data.unavailable_sources.join(", ")}
            </span>
          )}
        </div>
      )}

      {error && (
        <p className="rounded-md border border-rose-500/50 bg-surface px-4 py-3 text-sm text-rose-300">
          Unable to load disclosures: {error}
        </p>
      )}

      <Panel
        title="Disclosures"
        right={
          <span className="font-mono text-xs text-muted">
            {loading ? "..." : `${data?.count ?? 0} filings`}
          </span>
        }
        bodyClassName="p-0"
      >
        <div className="max-h-[38rem] overflow-auto">
          <table className="w-full min-w-[820px] text-sm">
            <thead className="sticky top-0 bg-surface-2 text-left text-xs text-muted">
              <tr>
                <th className="px-3 py-2 font-medium">Traded</th>
                <th className="px-3 py-2 font-medium">Disclosed</th>
                <th className="px-3 py-2 font-medium">Member</th>
                <th className="px-3 py-2 font-medium">Chamber</th>
                <th className="px-3 py-2 font-medium">Ticker</th>
                <th className="px-3 py-2 font-medium">Side</th>
                <th className="px-3 py-2 text-right font-medium">Amount</th>
              </tr>
            </thead>
            <tbody>
              {loading && (
                <tr>
                  <td colSpan={7} className="px-4 py-10 text-center text-muted">
                    Loading congressional disclosures...
                  </td>
                </tr>
              )}
              {(data?.rows ?? []).map((row: CongressTrade, index) => (
                <tr
                  key={`${row.member ?? ""}-${row.ticker ?? ""}-${row.transaction_date ?? ""}-${index}`}
                  className="border-t border-border hover:bg-surface-hover"
                >
                  <td className="whitespace-nowrap px-3 py-1.5 font-mono text-muted">
                    {row.transaction_date ?? "—"}
                  </td>
                  <td className="whitespace-nowrap px-3 py-1.5 font-mono text-faint">
                    {row.disclosure_date ?? "—"}
                  </td>
                  <td className="px-3 py-1.5 text-foreground">
                    {row.member ?? "—"}
                    {row.owner && row.owner.toLowerCase() !== "self" && (
                      <span className="ml-1 text-xs text-faint">
                        ({row.owner})
                      </span>
                    )}
                  </td>
                  <td className="px-3 py-1.5 capitalize text-muted">
                    {row.chamber}
                  </td>
                  <td className="whitespace-nowrap px-3 py-1.5 font-mono font-medium text-foreground">
                    {row.ticker ?? "—"}
                    {row.asset_description && !row.ticker && (
                      <span
                        className="block max-w-[16rem] truncate text-xs font-normal text-faint"
                        title={row.asset_description}
                      >
                        {row.asset_description}
                      </span>
                    )}
                  </td>
                  <td
                    className={`whitespace-nowrap px-3 py-1.5 font-mono ${tradeTone(row.transaction_type)}`}
                  >
                    {tradeLabel(row.transaction_type)}
                  </td>
                  <td className="whitespace-nowrap px-3 py-1.5 text-right font-mono text-muted">
                    {row.amount ?? "—"}
                  </td>
                </tr>
              ))}
              {!loading && !error && data && data.rows.length === 0 && (
                <tr>
                  <td colSpan={7} className="px-4 py-8 text-center text-muted">
                    No disclosures match these filters in the selected window.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </Panel>

      <p className="max-w-3xl text-xs text-muted">
        Source: public Senate eFD / House Clerk PTR mirrors. Amounts are the
        disclosed dollar ranges. Absence of a filing does not imply absence of
        trading.
      </p>
    </div>
  );
}
