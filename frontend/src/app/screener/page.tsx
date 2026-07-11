"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { api, ScreenerPreset, ScreenerResponse, ScreenerRow } from "@/lib/api";
import Panel from "@/components/ui/Panel";

// Snapshot-backed screener over the configured universe. Presets come from
// the backend with an explicit methodology string; this page never scans
// live option chains.

const PRESETS: { id: ScreenerPreset; label: string; blurb: string }[] = [
  {
    id: "high_vol_oi",
    label: "High Vol/OI",
    blurb: "Fresh positioning vs resting inventory",
  },
  {
    id: "unusually_bullish",
    label: "Unusually Bullish",
    blurb: "Call contracts with elevated vol/OI",
  },
  {
    id: "gamma_squeeze",
    label: "Gamma Squeeze Candidates",
    blurb: "Positive gamma zone, above-spot magnet, score ≥ 50",
  },
];

type Column = {
  key: keyof ScreenerRow;
  label: string;
  digits?: number;
  format?: (row: ScreenerRow) => string;
};

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
  { key: "open_interest", label: "Total OI", digits: 0 },
];

function cell(row: ScreenerRow, column: Column): string {
  const value = row[column.key];
  if (value == null) return "—";
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (typeof value === "string")
    return column.key === "option_type" ? value.toUpperCase() : value;
  return value.toLocaleString(undefined, {
    minimumFractionDigits: column.digits ?? 0,
    maximumFractionDigits: column.digits ?? 0,
  });
}

export default function ScreenerPage() {
  const controller = useRef<AbortController | null>(null);
  const [preset, setPreset] = useState<ScreenerPreset>("high_vol_oi");
  const [minVolOi, setMinVolOi] = useState("");
  const [minOI, setMinOI] = useState("");
  const [data, setData] = useState<ScreenerResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const fetchScreen = useCallback(
    (nextPreset: ScreenerPreset, volOi: string, oi: string) => {
      controller.current?.abort();
      const nextController = new AbortController();
      controller.current = nextController;
      api
        .screener(
          nextPreset,
          {
            minVolOi: volOi ? Number(volOi) : undefined,
            minOpenInterest: oi ? Number(oi) : undefined,
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

  // Loading/error resets happen in the event handlers (chip click, form
  // submit); the effect only performs the fetch for the active preset.
  useEffect(() => {
    fetchScreen(preset, minVolOi, minOI);
    // Filters apply on submit, not on each keystroke.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [preset, fetchScreen]);

  function changePreset(next: ScreenerPreset) {
    if (next === preset) return;
    setLoading(true);
    setError(null);
    setPreset(next);
  }

  const columns = preset === "gamma_squeeze" ? GAMMA_COLUMNS : CONTRACT_COLUMNS;
  const activePreset = PRESETS.find((p) => p.id === preset);

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-lg font-semibold">Options Screener</h1>
        <p className="mt-1 max-w-3xl text-sm text-muted">
          Preset screens over the persisted snapshot universe. Candidates come
          from our own accumulated data, never a live chain scan.
        </p>
      </div>

      <div className="flex flex-wrap gap-2">
        {PRESETS.map((p) => (
          <button
            key={p.id}
            onClick={() => changePreset(p.id)}
            title={p.blurb}
            className={`rounded-full border px-3 py-1 text-xs transition-colors ${
              preset === p.id
                ? "border-accent bg-accent/15 text-foreground"
                : "border-border bg-surface text-muted hover:text-foreground"
            }`}
          >
            {p.label}
          </button>
        ))}
      </div>

      <form
        onSubmit={(event) => {
          event.preventDefault();
          setLoading(true);
          setError(null);
          fetchScreen(preset, minVolOi, minOI);
        }}
        className="flex flex-wrap items-center gap-2"
      >
        <input
          type="number"
          min="0"
          step="0.1"
          value={minVolOi}
          onChange={(event) => setMinVolOi(event.target.value)}
          placeholder="Min vol/OI"
          aria-label="Minimum volume to open interest"
          className="w-28 rounded border border-border bg-surface px-2 py-1 text-sm outline-none focus:border-accent"
        />
        <input
          type="number"
          min="0"
          value={minOI}
          onChange={(event) => setMinOI(event.target.value)}
          placeholder="Min OI"
          aria-label="Minimum open interest"
          className="w-28 rounded border border-border bg-surface px-2 py-1 text-sm outline-none focus:border-accent"
        />
        <button
          type="submit"
          disabled={loading}
          className="rounded-md border border-accent bg-accent/15 px-3 py-1 text-sm hover:bg-accent/25 disabled:opacity-50"
        >
          {loading ? "Screening…" : "Apply"}
        </button>
      </form>

      {data && (
        <div className="flex flex-wrap gap-2 text-xs">
          <span className="rounded border border-border bg-surface px-2 py-1 font-mono text-muted">
            Snapshot: {data.as_of ?? "unavailable"}
          </span>
          <span
            className={`rounded border px-2 py-1 ${
              data.stale
                ? "border-amber-500/50 text-amber-300"
                : "border-emerald-500/50 text-emerald-300"
            }`}
          >
            {data.stale ? "Stale or incomplete snapshots" : "Current snapshots"}
          </span>
          {data.unavailable_symbols.length > 0 && (
            <span className="rounded border border-rose-500/50 px-2 py-1 text-rose-300">
              No snapshots: {data.unavailable_symbols.join(", ")}
            </span>
          )}
        </div>
      )}

      {error && (
        <p className="rounded-md border border-rose-500/50 bg-surface px-4 py-3 text-sm text-rose-300">
          Unable to screen: {error}
        </p>
      )}

      <Panel
        title={activePreset?.label ?? "Results"}
        right={
          <span className="font-mono text-xs text-muted">
            {data?.rows.length ?? 0} candidates
          </span>
        }
        bodyClassName="p-0"
      >
        <div className="max-h-[34rem] overflow-auto">
          <table className="w-full min-w-[720px] text-sm">
            <thead className="sticky top-0 bg-surface-2 text-left text-xs text-muted">
              <tr>
                {columns.map((column) => (
                  <th
                    key={column.key}
                    className="whitespace-nowrap px-3 py-2 text-right font-medium first:text-left"
                  >
                    {column.label}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {(data?.rows ?? []).map((row, index) => (
                <tr
                  key={`${row.symbol}-${row.expiration_date ?? ""}-${row.strike ?? ""}-${row.option_type ?? ""}-${index}`}
                  className="border-t border-border hover:bg-surface-hover"
                >
                  {columns.map((column) => (
                    <td
                      key={column.key}
                      className="whitespace-nowrap px-3 py-1.5 text-right font-mono text-muted first:text-left first:text-foreground"
                    >
                      {cell(row, column)}
                    </td>
                  ))}
                </tr>
              ))}
              {!loading && (data?.rows.length ?? 0) === 0 && (
                <tr>
                  <td
                    colSpan={columns.length}
                    className="px-4 py-8 text-center text-muted"
                  >
                    No candidates match this screen yet. Candidates appear as
                    snapshot history accrues.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </Panel>

      {data && (
        <p className="max-w-3xl text-xs text-muted">
          Methodology: {data.methodology}
        </p>
      )}
    </div>
  );
}
