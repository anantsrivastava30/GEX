"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { api, GammaGapRadarResponse, GammaGapRadarRow } from "@/lib/api";
import Panel from "@/components/ui/Panel";
import StatTile from "@/components/ui/StatTile";
import DataFreshness from "@/components/ui/DataFreshness";

// Gamma Gap Radar: the hero list. One ranked row per ticker - which names are
// most likely to get pinned to their options magnet next, highest gap-fill
// score first, each with the plain-English read.

function formatNumber(value: number | null | undefined, digits = 1) {
  return value == null
    ? "—"
    : value.toLocaleString(undefined, {
        minimumFractionDigits: digits,
        maximumFractionDigits: digits,
      });
}

function pinStrength(score: number | null | undefined): { label: string; cls: string } {
  if (score == null) return { label: "—", cls: "text-muted" };
  if (score >= 35) return { label: "High", cls: "text-positive" };
  if (score >= 20) return { label: "Moderate", cls: "text-foreground" };
  return { label: "Low", cls: "text-muted" };
}

function CallCell({ row }: { row: GammaGapRadarRow }) {
  return (
    <div>
      <div className="text-foreground">
        {row.direction === "down" ? "Pull down to" : "Pull up to"}{" "}
        <span className="font-mono">{formatNumber(row.magnet_strike, 1)}</span>
      </div>
      <div className="text-xs text-muted">
        {row.gap_pct != null ? `${Math.abs(row.gap_pct).toFixed(1)}% away` : "—"}
        {row.positive_zone === 1 ? " · pins (long gamma)" : " · drifts (short gamma)"}
      </div>
    </div>
  );
}

export default function RadarPage() {
  const [data, setData] = useState<GammaGapRadarResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const controller = useRef<AbortController | null>(null);

  useEffect(() => {
    const next = new AbortController();
    controller.current = next;
    api
      .gammaGapRadar(0, next.signal)
      .then((result) => {
        if (controller.current === next) setData(result);
      })
      .catch((cause) => {
        if (cause instanceof DOMException && cause.name === "AbortError") return;
        if (controller.current === next)
          setError(cause instanceof Error ? cause.message : String(cause));
      })
      .finally(() => {
        if (controller.current === next) setLoading(false);
      });
    return () => next.abort();
  }, []);

  const rows = data?.rows ?? [];
  const strong = rows.filter((r) => (r.score ?? 0) >= 35).length;
  const top = rows[0];

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h1 className="text-lg font-semibold">Gamma Gap Radar</h1>
          <p className="mt-1 max-w-3xl text-sm text-muted">
            Today&apos;s pin candidates, ranked. Each row is the latest read on
            one ticker: the options &quot;magnet&quot; price its dealers&apos;
            hedging tends to pull it toward, strongest gap-fill score first.
            Every call here is logged and later graded on the{" "}
            <Link href="/track-record" className="text-accent hover:underline">
              track record
            </Link>{" "}
            page.
          </p>
        </div>
        <DataFreshness />
      </div>

      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <StatTile
          label="Tickers scanned"
          value={data == null ? "—" : String(data.tickers)}
          sub={data?.as_of ? `As of ${data.as_of}` : undefined}
          accent
        />
        <StatTile
          label="High-conviction pins"
          value={data == null ? "—" : String(strong)}
          sub="Score ≥ 35"
        />
        <StatTile
          label="Top candidate"
          value={top ? top.ticker : "—"}
          sub={top ? `Score ${formatNumber(top.score, 0)}` : undefined}
        />
        <StatTile
          label="Sessions logged"
          value={data == null ? "—" : String(data.sessions)}
          sub="own history"
        />
      </div>

      {error && (
        <p className="rounded-md border border-border bg-surface px-4 py-3 text-sm text-muted">
          {error}
        </p>
      )}

      <Panel
        title="Ranked pin candidates"
        right={
          !loading && data ? (
            <span className="font-mono text-xs text-muted">
              {rows.length} tickers · highest score first
            </span>
          ) : undefined
        }
        bodyClassName="p-0"
      >
        {loading ? (
          <div
            role="status"
            aria-label="Loading radar"
            className="h-72 animate-pulse bg-surface-2"
          />
        ) : rows.length === 0 ? (
          <p className="px-4 py-12 text-center text-sm text-muted">
            No gamma-gap scans logged yet. Start the scheduler during market
            hours to build the radar.
          </p>
        ) : (
          <div className="max-h-[42rem] overflow-auto">
            <table className="w-full min-w-[760px] text-sm">
              <thead className="sticky top-0 bg-surface-2 text-left text-xs text-muted">
                <tr>
                  <th className="px-4 py-2 font-medium">#</th>
                  <th className="px-4 py-2 font-medium">Ticker</th>
                  <th className="px-4 py-2 text-right font-medium">Pin strength</th>
                  <th className="px-4 py-2 font-medium">The call</th>
                  <th className="px-4 py-2 text-right font-medium">Price now</th>
                  <th className="px-4 py-2 font-medium">Read</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row, index) => {
                  const strength = pinStrength(row.score);
                  return (
                    <tr
                      key={`${row.ticker}-${index}`}
                      className="border-t border-border hover:bg-surface-hover"
                    >
                      <td className="px-4 py-2 font-mono text-xs text-muted">
                        {index + 1}
                      </td>
                      <td className="px-4 py-2">
                        <Link
                          href={`/stock/${row.ticker}`}
                          className="font-mono text-foreground hover:text-accent hover:underline"
                        >
                          {row.ticker}
                        </Link>
                        {row.stale && (
                          <span
                            className="ml-1.5 rounded bg-surface-2 px-1 py-0.5 text-[10px] text-muted"
                            title="Latest scan for this ticker predates the newest session."
                          >
                            stale
                          </span>
                        )}
                      </td>
                      <td className="px-4 py-2 text-right">
                        <span className={`font-mono ${strength.cls}`}>
                          {formatNumber(row.score, 0)}
                        </span>{" "}
                        <span className="text-xs text-muted">{strength.label}</span>
                      </td>
                      <td className="px-4 py-2">
                        <CallCell row={row} />
                      </td>
                      <td className="px-4 py-2 text-right font-mono">
                        {formatNumber(row.spot, 2)}
                      </td>
                      <td className="max-w-[22rem] px-4 py-2 text-xs text-muted">
                        {row.read}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </Panel>

      <p className="text-xs text-muted">
        Pin strength is the model&apos;s gap-fill score, not a probability.
        &quot;Pins (long gamma)&quot; means dealers dampen moves so the magnet
        tends to hold; &quot;drifts (short gamma)&quot; means the pull is weaker.
        This is research tooling, not investment advice.
      </p>
    </div>
  );
}
