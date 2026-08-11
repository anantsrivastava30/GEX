"use client";

import { useEffect, useRef, useState } from "react";
import { api, GammaGapOutcomeRow, GammaGapOutcomesResponse } from "@/lib/api";
import Panel from "@/components/ui/Panel";
import StatTile from "@/components/ui/StatTile";
import DataFreshness from "@/components/ui/DataFreshness";
import OutcomesPanel from "@/components/direction/OutcomesPanel";

// The public differentiator: every gamma-gap scan is logged, then scored
// against realized daily ranges. A hit means the magnet strike traded
// within the horizon of sessions AFTER the signal date (the signal day
// never counts). Pending signals are shown but excluded from the rate.

function formatNumber(value: number | null | undefined, digits = 1) {
  return value == null
    ? "—"
    : value.toLocaleString(undefined, {
        minimumFractionDigits: digits,
        maximumFractionDigits: digits,
      });
}

function formatTimestamp(value: string) {
  return value.replace("T", " ").replace("Z", "").slice(0, 16);
}

function OutcomeBadge({ row }: { row: GammaGapOutcomeRow }) {
  if (row.outcome === "hit") {
    return (
      <span className="rounded bg-positive/12 px-2 py-0.5 text-xs text-positive">
        Hit · {row.sessions_to_hit}s
      </span>
    );
  }
  if (row.outcome === "miss") {
    return (
      <span className="rounded bg-negative/12 px-2 py-0.5 text-xs text-negative">
        Miss
      </span>
    );
  }
  return (
    <span className="rounded bg-surface-2 px-2 py-0.5 text-xs text-muted">
      Pending {row.evaluated_sessions > 0 ? `(${row.evaluated_sessions}s in)` : ""}
    </span>
  );
}

export default function TrackRecordPage() {
  const [ticker, setTicker] = useState("");
  const [activeTicker, setActiveTicker] = useState("");
  const [data, setData] = useState<GammaGapOutcomesResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const controller = useRef<AbortController | null>(null);

  function fetchOutcomes(symbol?: string) {
    controller.current?.abort();
    const nextController = new AbortController();
    controller.current = nextController;
    api
      .gammaGapOutcomes(symbol, 5, 250, nextController.signal)
      .then((result) => {
        if (controller.current === nextController) setData(result);
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
  }

  function load(symbol?: string) {
    // Event-handler path: reset visible state, then fetch.
    setData(null);
    setError(null);
    setLoading(true);
    fetchOutcomes(symbol);
  }

  useEffect(() => {
    // Initial state is already empty; fetch without synchronous resets.
    fetchOutcomes();
    return () => controller.current?.abort();
  }, []);

  const summary = data?.summary;
  const rows = data?.rows ?? [];
  const hitRate =
    summary?.hit_rate != null ? `${(summary.hit_rate * 100).toFixed(0)}%` : "—";

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h1 className="text-lg font-semibold">Track Record</h1>
          <p className="mt-1 max-w-3xl text-sm text-muted">
            Every call this app makes is written down when it fires, then graded
            against what the market actually did - no cherry-picking. Gamma-gap
            price magnets are below; O&apos;Neil market-turn and breakout calls
            are at the bottom.
          </p>
        </div>
        <DataFreshness />
      </div>

      <div>
        <h2 className="text-base font-semibold">Gamma Gap</h2>
        <p className="mt-1 max-w-3xl text-sm text-muted">
          A gamma-gap call predicts that a stock gets pulled toward a nearby
          options &quot;magnet&quot; price. It counts as a{" "}
          <span className="text-positive">hit</span> if the stock trades at that
          magnet within {data?.horizon_sessions ?? 5} trading days of the call.
          The day the call fires never counts, and a call still inside its
          window stays <span className="text-muted">pending</span> - shown here
          but left out of the hit rate until it resolves.
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
          disabled={loading}
          className="rounded-md border border-accent bg-accent/15 px-4 py-1.5 text-sm text-foreground hover:bg-accent/25 disabled:opacity-50"
        >
          {loading ? "Loading..." : "Apply ticker filter"}
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
        <StatTile
          label="Hit rate"
          value={data == null ? "—" : hitRate}
          sub={
            summary
              ? `${summary.hits} of ${summary.decided} decided signals`
              : loading
                ? "Loading..."
                : error
                  ? "Unavailable"
                  : "No decided signals"
          }
          accent
        />
        <StatTile
          label="Logged signals"
          value={data == null ? "—" : String(summary?.signals ?? 0)}
          sub={summary ? `${summary.pending} pending` : undefined}
        />
        <StatTile
          label="Avg sessions to hit"
          value={
            summary?.avg_sessions_to_hit != null
              ? summary.avg_sessions_to_hit.toFixed(1)
              : "—"
          }
        />
        <StatTile
          label="High-conviction hit rate"
          value={
            summary?.by_score?.[2]?.hit_rate != null
              ? `${(summary.by_score[2].hit_rate * 100).toFixed(0)}%`
              : "—"
          }
          sub={
            summary?.by_score?.[2]
              ? `Score ≥ ${summary.by_score[2].score_min.toFixed(0)} · ${summary.by_score[2].decided} decided`
              : "Top score bucket"
          }
        />
      </div>

      {summary && summary.decided === 0 && summary.pending > 0 && (
        <p className="rounded-md border border-border bg-surface px-4 py-3 text-sm text-muted">
          Tracking {summary.pending} call{summary.pending === 1 ? "" : "s"}. None
          have finished their {data?.horizon_sessions ?? 5}-session window yet, so
          there is no hit rate to show. Calls resolve as the sessions complete -
          check back over the next week.
        </p>
      )}

      {summary && summary.decided > 0 && (
        <div className="flex flex-wrap gap-2 text-xs">
          {summary.by_score.map((bucket, index) => (
            <span
              key={bucket.score_min}
              className="rounded border border-border bg-surface px-2 py-1 font-mono text-muted"
            >
              Score{" "}
              {index === summary.by_score.length - 1
                ? `≥ ${bucket.score_min.toFixed(0)}`
                : `${bucket.score_min.toFixed(0)}–${bucket.score_max.toFixed(0)}`}
              :{" "}
              {bucket.hit_rate != null
                ? `${(bucket.hit_rate * 100).toFixed(0)}% (${bucket.hits}/${bucket.decided})`
                : "no decided signals"}
            </span>
          ))}
        </div>
      )}

      {error && (
        <p className="rounded-md border border-border bg-surface px-4 py-3 text-sm text-muted">
          {error}
        </p>
      )}

      <Panel
        title={activeTicker ? `${activeTicker} signal log` : "Signal log"}
        right={
          !loading && data && (
            <span className="font-mono text-xs text-muted">
              Most recent {rows.length} scans
            </span>
          )
        }
        bodyClassName="p-0"
      >
        {loading ? (
          <div role="status" aria-label="Loading signal history" className="h-72 animate-pulse bg-surface-2" />
        ) : error ? (
          <p className="px-4 py-12 text-center text-sm text-muted">Signal history could not be loaded.</p>
        ) : rows.length === 0 ? (
          <p className="px-4 py-12 text-center text-sm text-muted">
            {activeTicker
              ? `No logged gamma-gap signals match ${activeTicker}.`
              : "No logged gamma-gap signals yet. Start the scheduler during market hours to build this record."}
          </p>
        ) : (
          <div className="max-h-[36rem] overflow-auto">
            <table className="w-full min-w-[820px] text-sm">
              <thead className="sticky top-0 bg-surface-2 text-left text-xs text-muted">
                <tr>
                  <th className="px-4 py-2 font-medium">Date</th>
                  <th className="px-4 py-2 font-medium">Ticker</th>
                  <th className="px-4 py-2 font-medium">The call</th>
                  <th className="px-4 py-2 text-right font-medium">Price then</th>
                  <th className="px-4 py-2 text-right font-medium">Pin strength</th>
                  <th className="px-4 py-2 font-medium">Behavior</th>
                  <th className="px-4 py-2 text-right font-medium">Outcome</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row, index) => (
                  <tr
                    key={`${row.ts}-${row.ticker}-${index}`}
                    className="border-t border-border hover:bg-surface-hover"
                  >
                    <td className="whitespace-nowrap px-4 py-2 font-mono text-xs text-muted">
                      {formatTimestamp(row.ts)}
                    </td>
                    <td className="px-4 py-2 font-mono text-foreground">{row.ticker}</td>
                    <td className="px-4 py-2">
                      <div className="text-foreground">
                        {row.direction === "down" ? "Pull down to" : "Pull up to"}{" "}
                        <span className="font-mono">{formatNumber(row.magnet_strike, 1)}</span>
                      </div>
                      <div className="text-xs text-muted">
                        {row.gap_pct != null
                          ? `${Math.abs(row.gap_pct).toFixed(1)}% away`
                          : "—"}
                        {row.scan_count && row.scan_count > 1
                          ? ` · seen in ${row.scan_count} scans`
                          : ""}
                      </div>
                    </td>
                    <td className="px-4 py-2 text-right font-mono">
                      {formatNumber(row.spot, 2)}
                    </td>
                    <td className="px-4 py-2 text-right font-mono">
                      {formatNumber(row.score, 0)}
                    </td>
                    <td className="px-4 py-2 text-xs">
                      <span
                        className={
                          row.positive_zone === 1 ? "text-positive" : "text-negative"
                        }
                        title={
                          row.positive_zone === 1
                            ? "Long gamma: dealers dampen moves, so price tends to get pinned toward the magnet."
                            : "Short gamma: dealers amplify moves, so the pull is weaker and price can run."
                        }
                      >
                        {row.positive_zone === 1 ? "Pins (long gamma)" : "Drifts (short gamma)"}
                      </span>
                    </td>
                    <td className="whitespace-nowrap px-4 py-2 text-right">
                      <OutcomeBadge row={row} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Panel>

      <OutcomesPanel />
    </div>
  );
}
