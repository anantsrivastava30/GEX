"use client";

import { useEffect, useRef, useState } from "react";
import { api, GammaGapHistoryRow } from "@/lib/api";
import Panel from "@/components/ui/Panel";
import StatTile from "@/components/ui/StatTile";

function formatNumber(value: number | null | undefined, digits = 1) {
  return value == null ? "-" : value.toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

function formatTimestamp(value: string) {
  return value.replace("T", " ").replace("Z", "").slice(0, 16);
}

export default function TrackRecordPage() {
  const [ticker, setTicker] = useState("");
  const [activeTicker, setActiveTicker] = useState("");
  const [rows, setRows] = useState<GammaGapHistoryRow[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const controller = useRef<AbortController | null>(null);

  function load(symbol?: string) {
    controller.current?.abort();
    const nextController = new AbortController();
    controller.current = nextController;
    setRows(null);
    setError(null);
    api
      .gammaGapHistory(symbol, 250, nextController.signal)
      .then((result) => {
        if (controller.current === nextController) setRows(result);
      })
      .catch((cause) => {
        if (cause instanceof DOMException && cause.name === "AbortError") return;
        if (controller.current === nextController) {
          setError(cause instanceof Error ? cause.message : String(cause));
          setRows([]);
        }
      });
  }

  useEffect(() => {
    const initialController = new AbortController();
    controller.current = initialController;
    api
      .gammaGapHistory(undefined, 250, initialController.signal)
      .then((result) => {
        if (controller.current === initialController) setRows(result);
      })
      .catch((cause) => {
        if (cause instanceof DOMException && cause.name === "AbortError") return;
        if (controller.current === initialController) {
          setError(cause instanceof Error ? cause.message : String(cause));
          setRows([]);
        }
      });
    return () => initialController.abort();
  }, []);

  const signals = rows ?? [];
  const scored = signals.filter((row) => row.score != null);
  const averageScore = scored.length
    ? scored.reduce((total, row) => total + (row.score ?? 0), 0) / scored.length
    : null;
  const positiveZones = signals.filter((row) => row.positive_zone === 1).length;
  const tickers = new Set(signals.map((row) => row.ticker)).size;

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-lg font-semibold">Gamma Gap Track Record</h1>
        <p className="mt-1 max-w-3xl text-sm text-muted">
          An auditable log of the gamma-gap signals captured by the scheduler. Outcome
          scoring will be added once enough price history has accrued; this page does
          not infer a hit rate before that validation exists.
        </p>
      </div>

      <form
        className="flex flex-wrap items-center gap-2"
        onSubmit={(event) => {
          event.preventDefault();
          const symbol = ticker.trim().toUpperCase();
          setActiveTicker(symbol);
          load(symbol || undefined);
        }}
      >
        <input
          value={ticker}
          onChange={(event) => setTicker(event.target.value)}
          placeholder="Filter ticker"
          className="w-36 rounded-md border border-border bg-surface px-3 py-1.5 font-mono text-sm uppercase outline-none focus:border-accent"
        />
        <button
          type="submit"
          className="rounded-md border border-accent bg-accent/15 px-4 py-1.5 text-sm text-foreground hover:bg-accent/25"
        >
          Apply
        </button>
        {activeTicker && (
          <button
            type="button"
            onClick={() => {
              setTicker("");
              setActiveTicker("");
              load();
            }}
            className="px-2 py-1.5 text-sm text-muted hover:text-foreground"
          >
            Clear
          </button>
        )}
      </form>

      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <StatTile label="Logged signals" value={rows == null ? "-" : String(signals.length)} accent />
        <StatTile label="Average score" value={averageScore == null ? "-" : averageScore.toFixed(0)} sub="0-120 scale" />
        <StatTile label="Long-gamma zone" value={rows == null ? "-" : String(positiveZones)} />
        <StatTile label="Tickers covered" value={rows == null ? "-" : String(tickers)} />
      </div>

      {error && (
        <p className="rounded-md border border-border bg-surface px-4 py-3 text-sm text-muted">
          {error}
        </p>
      )}

      <Panel
        title={activeTicker ? `${activeTicker} signal log` : "Signal log"}
        right={
          rows && (
            <span className="font-mono text-xs text-muted">
              Most recent {signals.length} scans
            </span>
          )
        }
        bodyClassName="p-0"
      >
        {rows == null ? (
          <div className="h-72 animate-pulse bg-surface-2" />
        ) : signals.length === 0 ? (
          <p className="px-4 py-12 text-center text-sm text-muted">
            No logged gamma-gap signals yet. Start the scheduler during market hours to
            build this record.
          </p>
        ) : (
          <div className="max-h-[36rem] overflow-auto">
            <table className="w-full min-w-[720px] text-sm">
              <thead className="sticky top-0 bg-surface-2 text-left text-xs text-muted">
                <tr>
                  <th className="px-4 py-2 font-medium">Time</th>
                  <th className="px-4 py-2 font-medium">Ticker</th>
                  <th className="px-4 py-2 text-right font-medium">Spot</th>
                  <th className="px-4 py-2 text-right font-medium">Magnet</th>
                  <th className="px-4 py-2 text-right font-medium">Distance</th>
                  <th className="px-4 py-2 text-right font-medium">Score</th>
                  <th className="px-4 py-2 text-right font-medium">Dealer zone</th>
                </tr>
              </thead>
              <tbody>
                {signals.map((row, index) => (
                  <tr key={`${row.ts}-${row.ticker}-${index}`} className="border-t border-border hover:bg-surface-hover">
                    <td className="whitespace-nowrap px-4 py-2 font-mono text-xs text-muted">{formatTimestamp(row.ts)}</td>
                    <td className="px-4 py-2 font-mono text-foreground">{row.ticker}</td>
                    <td className="px-4 py-2 text-right font-mono">{formatNumber(row.spot, 2)}</td>
                    <td className="px-4 py-2 text-right font-mono">{formatNumber(row.magnet_strike, 1)}</td>
                    <td className="px-4 py-2 text-right font-mono">{formatNumber(row.distance, 2)}</td>
                    <td className="px-4 py-2 text-right font-mono">{formatNumber(row.score, 0)}</td>
                    <td className="px-4 py-2 text-right text-xs">
                      <span className={row.positive_zone === 1 ? "text-positive" : "text-negative"}>
                        {row.positive_zone === 1 ? "Long gamma" : "Short gamma"}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Panel>
    </div>
  );
}
