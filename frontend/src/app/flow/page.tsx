"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  api,
  CachedFlowResponse,
  CachedFlowRow,
  HottestChainRow,
  HottestChainsResponse,
} from "@/lib/api";
import Panel from "@/components/ui/Panel";

type SideFilter = "all" | "call" | "put";
type FeedSortKey = keyof CachedFlowRow;
type ChainSortKey = keyof HottestChainRow;

const FEED_COLUMNS: { key: FeedSortKey; label: string; digits?: number }[] = [
  { key: "ticker", label: "Symbol" },
  { key: "option_type", label: "Side" },
  { key: "expiration_date", label: "Expiry" },
  { key: "strike", label: "Strike", digits: 2 },
  { key: "volume", label: "Volume", digits: 0 },
  { key: "open_interest", label: "OI", digits: 0 },
  { key: "volume_oi", label: "Vol/OI", digits: 2 },
  { key: "oi_change", label: "OI chg", digits: 0 },
  { key: "mid_iv", label: "Mid IV", digits: 1 },
  { key: "iv_change", label: "IV chg", digits: 1 },
  { key: "score", label: "Score", digits: 1 },
];

const CHAIN_COLUMNS: { key: ChainSortKey; label: string; digits?: number }[] = [
  { key: "ticker", label: "Symbol" },
  { key: "expiration_date", label: "Expiry" },
  { key: "contracts", label: "Contracts", digits: 0 },
  { key: "total_volume", label: "Volume", digits: 0 },
  { key: "total_open_interest", label: "OI", digits: 0 },
  { key: "volume_oi", label: "Vol/OI", digits: 2 },
  { key: "oi_change", label: "OI chg", digits: 0 },
  { key: "iv_change", label: "IV chg", digits: 1 },
  { key: "score", label: "Score", digits: 1 },
];

function fmt(value: number | null | undefined, digits = 0, percent = false) {
  if (value == null) return "-";
  return `${value.toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  })}${percent ? "%" : ""}`;
}

function formatCell(key: string, value: string | number | boolean | null | undefined, digits?: number) {
  if (typeof value === "boolean") return value ? "Ready" : "Unavailable";
  if (typeof value === "string") return key === "option_type" ? value.toUpperCase() : value;
  return fmt(value, digits, key === "mid_iv" || key === "iv_change");
}

function Status({ data }: { data: CachedFlowResponse | null }) {
  if (!data) return null;
  const unavailable = data.unavailable_tickers.join(", ");
  const history = data.unavailable_history_tickers.join(", ");
  return (
    <div className="flex flex-wrap gap-2 text-xs">
      <span className="rounded border border-border bg-surface px-2 py-1 font-mono text-muted">
        Snapshot: {data.as_of ?? "unavailable"}
      </span>
      <span className={`rounded border px-2 py-1 ${data.stale ? "border-amber-500/50 text-amber-300" : "border-emerald-500/50 text-emerald-300"}`}>
        {data.stale ? "Stale or incomplete cache" : "Current cache"}
      </span>
      {data.unavailable_history && (
        <span className="rounded border border-amber-500/50 px-2 py-1 text-amber-300">
          History unavailable{history ? `: ${history}` : ""}
        </span>
      )}
      {unavailable && (
        <span className="rounded border border-rose-500/50 px-2 py-1 text-rose-300">
          Snapshots unavailable: {unavailable}
        </span>
      )}
    </div>
  );
}

export default function FlowPage() {
  const controller = useRef<AbortController | null>(null);
  const [feed, setFeed] = useState<CachedFlowResponse | null>(null);
  const [chains, setChains] = useState<HottestChainsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [symbol, setSymbol] = useState("");
  const [side, setSide] = useState<SideFilter>("all");
  const [minVolumeOi, setMinVolumeOi] = useState("0");
  const [minOiChange, setMinOiChange] = useState("0");
  const [minIvChange, setMinIvChange] = useState("0");
  const [feedSort, setFeedSort] = useState<FeedSortKey>("score");
  const [feedAsc, setFeedAsc] = useState(false);
  const [chainSort, setChainSort] = useState<ChainSortKey>("score");
  const [chainAsc, setChainAsc] = useState(false);

  async function load() {
    controller.current?.abort();
    const nextController = new AbortController();
    controller.current = nextController;
    setLoading(true);
    setError(null);
    try {
      const [nextFeed, nextChains] = await Promise.all([
        api.flowFeed(500, nextController.signal),
        api.hottestChains(200, nextController.signal),
      ]);
      if (controller.current === nextController) {
        setFeed(nextFeed);
        setChains(nextChains);
      }
    } catch (cause) {
      if (cause instanceof DOMException && cause.name === "AbortError") return;
      if (controller.current === nextController) {
        setError(cause instanceof Error ? cause.message : String(cause));
      }
    } finally {
      if (controller.current === nextController) {
        controller.current = null;
        setLoading(false);
      }
    }
  }

  useEffect(() => {
    const initialController = new AbortController();
    controller.current = initialController;
    Promise.all([
      api.flowFeed(500, initialController.signal),
      api.hottestChains(200, initialController.signal),
    ])
      .then(([nextFeed, nextChains]) => {
        if (controller.current === initialController) {
          setFeed(nextFeed);
          setChains(nextChains);
        }
      })
      .catch((cause) => {
        if (cause instanceof DOMException && cause.name === "AbortError") return;
        if (controller.current === initialController) {
          setError(cause instanceof Error ? cause.message : String(cause));
        }
      })
      .finally(() => {
        if (controller.current === initialController) {
          controller.current = null;
          setLoading(false);
        }
      });
    return () => initialController.abort();
  }, []);

  const filteredFeed = useMemo(() => {
    const query = symbol.trim().toUpperCase();
    const ratio = Number(minVolumeOi) || 0;
    const oiChange = Number(minOiChange) || 0;
    const ivChange = Number(minIvChange) || 0;
    const direction = feedAsc ? 1 : -1;
    return (feed?.rows ?? [])
      .filter((row) =>
        (!query || row.ticker.includes(query)) &&
        (side === "all" || row.option_type.toLowerCase() === side) &&
        (row.volume_oi ?? 0) >= ratio &&
        Math.abs(row.oi_change ?? 0) >= oiChange &&
        Math.abs(row.iv_change ?? 0) >= ivChange,
      )
      .sort((a, b) => {
        const av = a[feedSort];
        const bv = b[feedSort];
        if (av == null) return bv == null ? 0 : 1;
        if (bv == null) return -1;
        return (typeof av === "string" ? av.localeCompare(String(bv)) : Number(av) - Number(bv)) * direction;
      });
  }, [feed, symbol, side, minVolumeOi, minOiChange, minIvChange, feedSort, feedAsc]);

  const sortedChains = useMemo(() => {
    const direction = chainAsc ? 1 : -1;
    return [...(chains?.rows ?? [])].sort((a, b) => {
      const av = a[chainSort];
      const bv = b[chainSort];
      if (av == null) return bv == null ? 0 : 1;
      if (bv == null) return -1;
      return (typeof av === "string" ? av.localeCompare(String(bv)) : Number(av) - Number(bv)) * direction;
    });
  }, [chains, chainSort, chainAsc]);

  function toggleFeedSort(key: FeedSortKey) {
    if (key === feedSort) setFeedAsc((value) => !value);
    else { setFeedSort(key); setFeedAsc(false); }
  }

  function toggleChainSort(key: ChainSortKey) {
    if (key === chainSort) setChainAsc((value) => !value);
    else { setChainSort(key); setChainAsc(false); }
  }

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h1 className="text-lg font-semibold">Options Positioning Proxy</h1>
          <p className="mt-1 max-w-3xl text-sm text-muted">
            Cached multi-ticker contract anomalies ranked from volume, open interest, and IV changes. This is not trade tape, sweep, execution, or aggressor data.
          </p>
        </div>
        <button onClick={load} disabled={loading} className="rounded-md border border-accent bg-accent/15 px-3 py-1.5 text-sm hover:bg-accent/25 disabled:opacity-50">
          {loading ? "Refreshing..." : "Refresh cache"}
        </button>
      </div>

      <Status data={feed} />
      {error && <p className="rounded-md border border-rose-500/50 bg-surface px-4 py-3 text-sm text-rose-300">Unable to load cached positioning: {error}</p>}

      <Panel title="Hottest Chains" right={<span className="font-mono text-xs text-muted">{sortedChains.length} cached chains</span>} bodyClassName="p-0">
        <div className="max-h-80 overflow-auto">
          <table className="w-full min-w-[850px] text-sm">
            <thead className="sticky top-0 bg-surface-2 text-left text-xs text-muted"><tr>{CHAIN_COLUMNS.map((column) => <th key={column.key} onClick={() => toggleChainSort(column.key)} className="cursor-pointer select-none whitespace-nowrap px-3 py-2 text-right font-medium hover:text-foreground">{column.label} {chainSort === column.key ? (chainAsc ? "▲" : "▼") : ""}</th>)}</tr></thead>
            <tbody>{sortedChains.map((row) => <tr key={`${row.ticker}-${row.expiration_date}`} className="border-t border-border hover:bg-surface-hover">{CHAIN_COLUMNS.map((column) => <td key={column.key} className="whitespace-nowrap px-3 py-1.5 text-right font-mono text-muted">{formatCell(column.key, row[column.key], column.digits)}</td>)}</tr>)}{!loading && sortedChains.length === 0 && <tr><td colSpan={CHAIN_COLUMNS.length} className="px-4 py-8 text-center text-muted">No cached chain rankings are available.</td></tr>}</tbody>
          </table>
        </div>
      </Panel>

      <Panel title="Contract-Level Feed" right={<span className="font-mono text-xs text-muted">{filteredFeed.length} of {feed?.rows.length ?? 0} contracts</span>} bodyClassName="p-0">
        <div className="flex flex-wrap gap-2 border-b border-border p-3">
          <input value={symbol} onChange={(event) => setSymbol(event.target.value)} placeholder="Filter symbol" className="w-32 rounded border border-border bg-surface px-2 py-1 text-sm uppercase outline-none focus:border-accent" />
          <select value={side} onChange={(event) => setSide(event.target.value as SideFilter)} className="rounded border border-border bg-surface px-2 py-1 text-sm"><option value="all">All sides</option><option value="call">Calls</option><option value="put">Puts</option></select>
          <input type="number" min="0" value={minVolumeOi} onChange={(event) => setMinVolumeOi(event.target.value)} aria-label="Minimum volume to open interest" placeholder="Min vol/OI" className="w-28 rounded border border-border bg-surface px-2 py-1 text-sm" />
          <input type="number" min="0" value={minOiChange} onChange={(event) => setMinOiChange(event.target.value)} aria-label="Minimum absolute open interest change" placeholder="Min |OI chg|" className="w-28 rounded border border-border bg-surface px-2 py-1 text-sm" />
          <input type="number" min="0" step="0.01" value={minIvChange} onChange={(event) => setMinIvChange(event.target.value)} aria-label="Minimum absolute implied volatility change" placeholder="Min |IV chg|" className="w-28 rounded border border-border bg-surface px-2 py-1 text-sm" />
        </div>
        <div className="max-h-[34rem] overflow-auto">
          <table className="w-full min-w-[1050px] text-sm">
            <thead className="sticky top-0 bg-surface-2 text-left text-xs text-muted"><tr>{FEED_COLUMNS.map((column) => <th key={column.key} onClick={() => toggleFeedSort(column.key)} className="cursor-pointer select-none whitespace-nowrap px-3 py-2 text-right font-medium hover:text-foreground">{column.label} {feedSort === column.key ? (feedAsc ? "▲" : "▼") : ""}</th>)}</tr></thead>
            <tbody>{filteredFeed.map((row) => <tr key={`${row.ticker}-${row.expiration_date}-${row.strike}-${row.option_type}`} className="border-t border-border hover:bg-surface-hover">{FEED_COLUMNS.map((column) => <td key={column.key} className="whitespace-nowrap px-3 py-1.5 text-right font-mono text-muted">{formatCell(column.key, row[column.key], column.digits)}</td>)}</tr>)}{!loading && filteredFeed.length === 0 && <tr><td colSpan={FEED_COLUMNS.length} className="px-4 py-8 text-center text-muted">No cached contracts match the active filters.</td></tr>}</tbody>
          </table>
        </div>
      </Panel>
    </div>
  );
}
