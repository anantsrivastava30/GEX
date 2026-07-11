"use client";

import { useMemo, useRef, useState } from "react";
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

// UW-style filter chips: single-select per group, applied client-side.
type SideFilter = "all" | "calls" | "puts";
const SIDE_CHIPS: { id: SideFilter; label: string }[] = [
  { id: "all", label: "All flow" },
  { id: "calls", label: "Higher call vol/OI" },
  { id: "puts", label: "Higher put vol/OI" },
];
const RATIO_CHIPS = [
  { id: 0, label: "Any vol/OI" },
  { id: 2, label: "vol/OI ≥ 2" },
  { id: 5, label: "vol/OI ≥ 5" },
];
const OI_CHIPS = [
  { id: 0, label: "Any OI" },
  { id: 1_000, label: "OI ≥ 1k" },
  { id: 10_000, label: "OI ≥ 10k" },
];

function Chip({
  active,
  label,
  onClick,
}: {
  active: boolean;
  label: string;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={`rounded-full border px-3 py-1 text-xs transition-colors ${
        active
          ? "border-accent bg-accent/15 text-foreground"
          : "border-border bg-surface text-muted hover:text-foreground"
      }`}
    >
      {label}
    </button>
  );
}

export default function FlowPage() {
  const [symbol, setSymbol] = useState("SPY");
  const [data, setData] = useState<UnusualResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [sortKey, setSortKey] = useState<SortKey>("total_vol_oi");
  const [asc, setAsc] = useState(false);
  const [side, setSide] = useState<SideFilter>("all");
  const [minRatio, setMinRatio] = useState(0);
  const [minOI, setMinOI] = useState(0);
  const scanController = useRef<AbortController | null>(null);

  async function scan(sym: string) {
    const upper = sym.trim().toUpperCase();
    if (!upper) return;
    scanController.current?.abort();
    const controller = new AbortController();
    scanController.current = controller;
    setLoading(true);
    setError(null);
    setData(null);
    try {
      const exps = await api.expirations(upper, controller.signal);
      if (!exps.length) throw new Error("No expirations available for this symbol.");
      const rows = await api.unusual(upper, exps.slice(0, 4), 25, controller.signal);
      setData(rows);
    } catch (e) {
      if (e instanceof DOMException && e.name === "AbortError") return;
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      if (scanController.current === controller) {
        scanController.current = null;
        setLoading(false);
      }
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
    return data.rows
      .filter((row) => {
        if (side === "calls" && !((row.vol_oi_call ?? 0) > (row.vol_oi_put ?? 0)))
          return false;
        if (side === "puts" && !((row.vol_oi_put ?? 0) > (row.vol_oi_call ?? 0)))
          return false;
        if (minRatio > 0 && (row.total_vol_oi ?? 0) < minRatio) return false;
        if (
          minOI > 0 &&
          (row.open_interest_call ?? 0) + (row.open_interest_put ?? 0) < minOI
        )
          return false;
        return true;
      })
      .sort((a, b) => {
        const av = a[sortKey];
        const bv = b[sortKey];
        if (av == null) return bv == null ? 0 : 1;
        if (bv == null) return -1;
        return (Number(av) - Number(bv)) * factor;
      });
  }, [data, sortKey, asc, side, minRatio, minOI]);

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
        <div className="flex flex-wrap items-center gap-2">
          {SIDE_CHIPS.map((chip) => (
            <Chip
              key={chip.id}
              active={side === chip.id}
              label={chip.label}
              onClick={() => setSide(chip.id)}
            />
          ))}
          <span className="h-4 w-px bg-border" aria-hidden />
          {RATIO_CHIPS.map((chip) => (
            <Chip
              key={chip.id}
              active={minRatio === chip.id}
              label={chip.label}
              onClick={() => setMinRatio(chip.id)}
            />
          ))}
          <span className="h-4 w-px bg-border" aria-hidden />
          {OI_CHIPS.map((chip) => (
            <Chip
              key={chip.id}
              active={minOI === chip.id}
              label={chip.label}
              onClick={() => setMinOI(chip.id)}
            />
          ))}
        </div>
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
                {rows.length === 0 && (
                  <tr className="border-t border-border">
                    <td
                      colSpan={COLUMNS.length}
                      className="px-4 py-8 text-center text-sm text-muted"
                    >
                      No strikes match the active filters.
                    </td>
                  </tr>
                )}
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
