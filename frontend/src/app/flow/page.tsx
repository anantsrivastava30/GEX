"use client";

import { useMemo, useState } from "react";
import { api, UnusualResponse, UnusualRow } from "@/lib/api";
import Panel from "@/components/ui/Panel";

// Free-data proxy for a flow feed: strikes ranked by volume/open-interest
// anomaly across the nearest expirations. A true trade-tape feed slots in
// once a paid OPRA provider is connected (Phase 4).

function fmt(value: number | null | undefined, digits = 2) {
  if (value == null) return "—";
  return value.toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

type SortKey = keyof UnusualRow;
const COLUMNS: { key: SortKey; label: string; digits: number; strong?: boolean }[] = [
  { key: "strike", label: "Strike", digits: 1 },
  { key: "vol_oi_call", label: "Call vol/OI", digits: 2 },
  { key: "vol_oi_put", label: "Put vol/OI", digits: 2 },
  { key: "open_interest_call", label: "Call OI", digits: 0 },
  { key: "open_interest_put", label: "Put OI", digits: 0 },
  { key: "total_vol_oi", label: "Total vol/OI", digits: 2, strong: true },
];

export default function FlowPage() {
  const [symbol, setSymbol] = useState("SPY");
  const [data, setData] = useState<UnusualResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [sortKey, setSortKey] = useState<SortKey>("total_vol_oi");
  const [asc, setAsc] = useState(false);

  async function scan(sym: string) {
    const upper = sym.trim().toUpperCase();
    if (!upper) return;
    setLoading(true);
    setError(null);
    setData(null);
    try {
      const exps = await api.expirations(upper);
      if (!exps.length) throw new Error("No expirations available for this symbol.");
      const rows = await api.unusual(upper, exps.slice(0, 4), 25);
      setData(rows);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  function toggleSort(key: SortKey) {
    if (key === sortKey) setAsc((v) => !v);
    else {
      setSortKey(key);
      setAsc(false);
    }
  }

  const rows = useMemo(() => {
    if (!data) return [];
    const factor = asc ? 1 : -1;
    return [...data.rows].sort((a, b) => {
      const av = a[sortKey] ?? -Infinity;
      const bv = b[sortKey] ?? -Infinity;
      return (Number(av) - Number(bv)) * factor;
    });
  }, [data, sortKey, asc]);

  return (
    <div className="space-y-4">
      <h1 className="text-lg font-semibold">Flow — Unusual Activity</h1>
      <p className="max-w-2xl text-sm text-muted">
        Strikes ranked by volume-to-open-interest anomaly across the four
        nearest expirations. High vol/OI flags fresh positioning rather than
        resting inventory.
      </p>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          scan(symbol);
        }}
        className="flex items-center gap-2"
      >
        <input
          value={symbol}
          onChange={(e) => setSymbol(e.target.value)}
          placeholder="Ticker"
          className="w-32 rounded-md border border-border bg-surface px-3 py-1.5 font-mono text-sm uppercase outline-none focus:border-accent"
        />
        <button
          type="submit"
          disabled={loading}
          className="rounded-md border border-accent bg-accent/15 px-4 py-1.5 text-sm text-foreground hover:bg-accent/25 disabled:opacity-50"
        >
          {loading ? "Scanning…" : "Scan"}
        </button>
      </form>

      {error && (
        <p className="rounded-md border border-border bg-surface px-4 py-3 text-sm text-muted">
          {error}
        </p>
      )}

      {data && (
        <Panel
          title={`${data.symbol} · unusual strikes`}
          right={
            <span className="font-mono text-xs text-muted">
              {data.expirations.length} exp · {rows.length} rows
            </span>
          }
          bodyClassName="p-0"
        >
          <div className="max-h-[32rem] overflow-auto">
            <table className="w-full text-sm">
              <thead className="sticky top-0 bg-surface-2 text-left text-xs text-muted">
                <tr>
                  {COLUMNS.map((col) => {
                    const active = col.key === sortKey;
                    return (
                      <th
                        key={col.key}
                        onClick={() => toggleSort(col.key)}
                        className={`cursor-pointer select-none px-4 py-2 font-medium hover:text-foreground ${
                          col.key === "strike" ? "text-left" : "text-right"
                        } ${active ? "text-foreground" : ""}`}
                      >
                        <span className="inline-flex items-center gap-1">
                          {col.key !== "strike" && <span className="flex-1" />}
                          {col.label}
                          <span className="w-2 text-accent">
                            {active ? (asc ? "▲" : "▼") : ""}
                          </span>
                        </span>
                      </th>
                    );
                  })}
                </tr>
              </thead>
              <tbody>
                {rows.map((row) => (
                  <tr
                    key={row.strike}
                    className="border-t border-border hover:bg-surface-hover"
                  >
                    {COLUMNS.map((col) => (
                      <td
                        key={col.key}
                        className={`px-4 py-1.5 font-mono ${
                          col.key === "strike" ? "text-left" : "text-right"
                        } ${col.strong ? "text-foreground" : "text-muted"}`}
                      >
                        {fmt(row[col.key] as number | null | undefined, col.digits)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>
      )}
    </div>
  );
}
