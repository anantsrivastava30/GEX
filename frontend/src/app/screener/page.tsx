"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import FilterBuilder from "@/components/screener/FilterBuilder";
import Panel from "@/components/ui/Panel";
import {
  api,
  CustomScreenerRequest,
  CustomScreenerResponse,
  ScreenerFieldSpec,
  ScreenerPreset,
  ScreenerResponse,
  ScreenerRow,
} from "@/lib/api";

const PRESETS: { id: ScreenerPreset; label: string; blurb: string }[] = [
  { id: "high_vol_oi", label: "High Vol/OI", blurb: "Fresh positioning vs resting inventory" },
  { id: "unusually_bullish", label: "Unusually Bullish", blurb: "Call contracts with elevated vol/OI" },
  { id: "gamma_squeeze", label: "Gamma Squeeze Candidates", blurb: "Positive gamma zone and above-spot magnet" },
];

const CUSTOM_DEFAULT: CustomScreenerRequest = {
  scope: "contract",
  conditions: [{ field: "volume_oi", operator: "gte", value: 2 }],
  sort: { field: "volume_oi", direction: "desc" },
  limit: 50,
};

type Column = { key: keyof ScreenerRow; label: string; digits?: number };
const CONTRACT_COLUMNS: Column[] = [
  { key: "symbol", label: "Symbol" },
  { key: "option_type", label: "Side" },
  { key: "expiration_date", label: "Expiry" },
  { key: "strike", label: "Strike", digits: 2 },
  { key: "volume", label: "Volume", digits: 0 },
  { key: "open_interest", label: "OI", digits: 0 },
  { key: "volume_oi", label: "Vol/OI", digits: 2 },
];
const GAMMA_COLUMNS: Column[] = [
  { key: "symbol", label: "Symbol" },
  { key: "spot", label: "Spot", digits: 2 },
  { key: "gamma_magnet_strike", label: "Magnet", digits: 1 },
  { key: "gamma_gap_distance", label: "Distance", digits: 2 },
  { key: "gamma_gap_score", label: "Score", digits: 0 },
  { key: "volume_oi", label: "Vol/OI", digits: 2 },
];

function cell(row: ScreenerRow, column: Column): string {
  const value = row[column.key];
  if (value == null) return "-";
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (typeof value === "string") return column.key === "option_type" ? value.toUpperCase() : value;
  return value.toLocaleString(undefined, {
    minimumFractionDigits: column.digits ?? 0,
    maximumFractionDigits: column.digits ?? 0,
  });
}

function customValue(value: string | number | boolean | null): string {
  if (value == null) return "-";
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (typeof value === "number") return value.toLocaleString(undefined, { maximumFractionDigits: 4 });
  return value;
}

export default function ScreenerPage() {
  const controller = useRef<AbortController | null>(null);
  const [mode, setMode] = useState<ScreenerPreset | "custom">("high_vol_oi");
  const [minVolOi, setMinVolOi] = useState("");
  const [minOI, setMinOI] = useState("");
  const [data, setData] = useState<ScreenerResponse | null>(null);
  const [customData, setCustomData] = useState<CustomScreenerResponse | null>(null);
  const [fields, setFields] = useState<ScreenerFieldSpec[]>([]);
  const [query, setQuery] = useState<CustomScreenerRequest>(CUSTOM_DEFAULT);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const fetchPreset = useCallback((preset: ScreenerPreset, volOi: string, oi: string) => {
    controller.current?.abort();
    const nextController = new AbortController();
    controller.current = nextController;
    api.screener(
      preset,
      { minVolOi: volOi ? Number(volOi) : undefined, minOpenInterest: oi ? Number(oi) : undefined },
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
    const nextController = new AbortController();
    api.screenerFields(nextController.signal).then((response) => setFields(response.fields)).catch(() => setFields([]));
    return () => nextController.abort();
  }, []);

  useEffect(() => {
    if (mode !== "custom") fetchPreset(mode, minVolOi, minOI);
    return () => controller.current?.abort();
    // Preset filters apply on form submit.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode, fetchPreset]);

  function changeMode(next: ScreenerPreset | "custom") {
    if (next === mode) return;
    controller.current?.abort();
    setError(null);
    setLoading(next !== "custom");
    setMode(next);
  }

  function runCustom() {
    controller.current?.abort();
    const nextController = new AbortController();
    controller.current = nextController;
    setLoading(true);
    setError(null);
    api.customScreener(query, nextController.signal).then((response) => {
      if (controller.current === nextController) setCustomData(response);
    }).catch((cause) => {
      if (cause instanceof DOMException && cause.name === "AbortError") return;
      if (controller.current === nextController) setError(cause instanceof Error ? cause.message : String(cause));
    }).finally(() => {
      if (controller.current === nextController) {
        controller.current = null;
        setLoading(false);
      }
    });
  }

  const columns = mode === "gamma_squeeze" ? GAMMA_COLUMNS : CONTRACT_COLUMNS;
  const customColumns = customData?.rows.length ? Object.keys(customData.rows[0].values) : [];
  const activeData = mode === "custom" ? customData : data;
  const resultCount = mode === "custom" ? customData?.rows.length : data?.rows.length;

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-lg font-semibold">Options Screener</h1>
        <p className="mt-1 max-w-3xl text-sm text-muted">
          Presets and custom AND-filters over persisted snapshots. No screen triggers a live chain scan.
        </p>
      </div>

      <div className="flex flex-wrap gap-2">
        {PRESETS.map((preset) => (
          <button key={preset.id} onClick={() => changeMode(preset.id)} title={preset.blurb} className={`rounded-full border px-3 py-1 text-xs ${mode === preset.id ? "border-accent bg-accent/15 text-foreground" : "border-border bg-surface text-muted hover:text-foreground"}`}>
            {preset.label}
          </button>
        ))}
        <button onClick={() => changeMode("custom")} className={`rounded-full border px-3 py-1 text-xs ${mode === "custom" ? "border-accent bg-accent/15 text-foreground" : "border-border bg-surface text-muted hover:text-foreground"}`}>
          Custom Builder
        </button>
      </div>

      {mode === "custom" ? (
        <Panel title="Custom snapshot filter">
          <FilterBuilder fields={fields} value={query} onChange={setQuery} />
          <button onClick={runCustom} disabled={loading || !fields.length} className="mt-4 rounded-md border border-accent bg-accent/15 px-3 py-1.5 text-sm hover:bg-accent/25 disabled:opacity-50">
            {loading ? "Screening..." : "Run custom screen"}
          </button>
        </Panel>
      ) : (
        <form onSubmit={(event) => { event.preventDefault(); setLoading(true); setError(null); fetchPreset(mode, minVolOi, minOI); }} className="flex flex-wrap items-center gap-2">
          <input type="number" min="0" step="0.1" value={minVolOi} onChange={(event) => setMinVolOi(event.target.value)} placeholder="Min vol/OI" aria-label="Minimum volume to open interest" className="w-28 rounded border border-border bg-surface px-2 py-1 text-sm outline-none focus:border-accent" />
          <input type="number" min="0" value={minOI} onChange={(event) => setMinOI(event.target.value)} placeholder="Min OI" aria-label="Minimum open interest" className="w-28 rounded border border-border bg-surface px-2 py-1 text-sm outline-none focus:border-accent" />
          <button type="submit" disabled={loading} className="rounded-md border border-accent bg-accent/15 px-3 py-1 text-sm hover:bg-accent/25 disabled:opacity-50">{loading ? "Screening..." : "Apply"}</button>
        </form>
      )}

      {activeData && (
        <div className="flex flex-wrap gap-2 text-xs">
          <span className="rounded border border-border bg-surface px-2 py-1 font-mono text-muted">Snapshot: {activeData.as_of ?? "unavailable"}</span>
          <span className={`rounded border px-2 py-1 ${activeData.stale ? "border-amber-500/50 text-amber-300" : "border-emerald-500/50 text-emerald-300"}`}>{activeData.stale ? "Stale or incomplete snapshots" : "Current snapshots"}</span>
          {activeData.unavailable_symbols.length > 0 && <span className="rounded border border-rose-500/50 px-2 py-1 text-rose-300">No snapshots: {activeData.unavailable_symbols.join(", ")}</span>}
        </div>
      )}

      {error && <p className="rounded-md border border-rose-500/50 bg-surface px-4 py-3 text-sm text-rose-300">Unable to screen: {error}</p>}

      <Panel title={mode === "custom" ? "Custom results" : PRESETS.find((preset) => preset.id === mode)?.label} right={<span className="font-mono text-xs text-muted">{resultCount ?? 0} candidates</span>} bodyClassName="p-0">
        <div className="max-h-[38rem] overflow-auto">
          {mode === "custom" ? (
            <table className="w-full min-w-[760px] text-sm">
              <thead className="sticky top-0 bg-surface-2 text-left text-xs text-muted"><tr><th className="px-3 py-2">Symbol</th>{customColumns.filter((column) => column !== "symbol").map((column) => <th key={column} className="px-3 py-2 text-right font-medium">{column.replaceAll("_", " ")}</th>)}</tr></thead>
              <tbody>{(customData?.rows ?? []).map((row) => <tr key={row.row_key} className="border-t border-border hover:bg-surface-hover"><td className="px-3 py-1.5 font-mono text-foreground">{row.symbol}</td>{customColumns.filter((column) => column !== "symbol").map((column) => <td key={column} className="whitespace-nowrap px-3 py-1.5 text-right font-mono text-muted">{customValue(row.values[column])}</td>)}</tr>)}{!loading && !(customData?.rows.length) && <tr><td colSpan={Math.max(1, customColumns.length)} className="px-4 py-8 text-center text-muted">Run a custom screen or relax its conditions.</td></tr>}</tbody>
            </table>
          ) : (
            <table className="w-full min-w-[720px] text-sm">
              <thead className="sticky top-0 bg-surface-2 text-left text-xs text-muted"><tr>{columns.map((column) => <th key={column.key} className="whitespace-nowrap px-3 py-2 text-right font-medium first:text-left">{column.label}</th>)}</tr></thead>
              <tbody>{(data?.rows ?? []).map((row, index) => <tr key={`${row.symbol}-${row.expiration_date ?? ""}-${row.strike ?? ""}-${index}`} className="border-t border-border hover:bg-surface-hover">{columns.map((column) => <td key={column.key} className="whitespace-nowrap px-3 py-1.5 text-right font-mono text-muted first:text-left first:text-foreground">{cell(row, column)}</td>)}</tr>)}{!loading && !(data?.rows.length) && <tr><td colSpan={columns.length} className="px-4 py-8 text-center text-muted">No candidates match this screen yet.</td></tr>}</tbody>
            </table>
          )}
        </div>
      </Panel>

      {activeData && <p className="max-w-3xl text-xs text-muted">Methodology: {activeData.methodology}</p>}
    </div>
  );
}
