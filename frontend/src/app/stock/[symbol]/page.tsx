"use client";

import Link from "next/link";
import { use, useEffect, useState } from "react";
import { api, GexProfile, TickerSnapshot } from "@/lib/api";
import GexStrikeChart from "@/components/charts/GexStrikeChart";

export default function StockPage({
  params,
}: {
  params: Promise<{ symbol: string }>;
}) {
  const { symbol } = use(params);
  const upper = symbol.toUpperCase();

  const [snapshot, setSnapshot] = useState<TickerSnapshot | null>(null);
  const [expirations, setExpirations] = useState<string[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [gex, setGex] = useState<GexProfile | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.tickerSnapshot(upper).then(setSnapshot).catch((e) => setError(e.message));
    api
      .expirations(upper)
      .then((exps) => {
        setExpirations(exps);
        if (exps.length) setSelected(exps[0]);
      })
      .catch((e) => setError(e.message));
  }, [upper]);

  useEffect(() => {
    if (!selected) return;
    setGex(null);
    api.gexProfile(upper, selected).then(setGex).catch((e) => setError(e.message));
  }, [upper, selected]);

  return (
    <div className="space-y-6">
      <div className="flex items-baseline gap-4">
        <h1 className="text-xl font-semibold">{upper}</h1>
        {snapshot?.last != null && (
          <span className="font-mono text-lg">{snapshot.last.toFixed(2)}</span>
        )}
        {snapshot?.change_percentage != null && (
          <span
            className={`font-mono text-sm ${
              snapshot.change_percentage >= 0 ? "text-positive" : "text-negative"
            }`}
          >
            {snapshot.change_percentage >= 0 ? "+" : ""}
            {snapshot.change_percentage.toFixed(2)}%
          </span>
        )}
      </div>

      {error && (
        <p className="rounded-md border border-border bg-surface px-4 py-3 text-sm text-muted">
          {error}
        </p>
      )}

      {expirations.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {expirations.slice(0, 8).map((exp) => (
            <button
              key={exp}
              onClick={() => setSelected(exp)}
              className={`rounded-full border px-3 py-1 font-mono text-xs transition-colors ${
                exp === selected
                  ? "border-accent bg-accent/15 text-foreground"
                  : "border-border bg-surface text-muted hover:text-foreground"
              }`}
            >
              {exp}
            </button>
          ))}
        </div>
      )}

      {gex && (
        <div className="grid gap-4 lg:grid-cols-2">
          <section className="rounded-lg border border-border bg-surface p-4">
            <h2 className="mb-3 text-sm font-medium text-muted">
              Net GEX by strike — {gex.expiration} (spot {gex.spot.toFixed(2)})
            </h2>
            <GexStrikeChart profile={gex.profile} spot={gex.spot} />
            <details className="mt-3 text-xs text-muted">
              <summary className="cursor-pointer select-none hover:text-foreground">
                Table view
              </summary>
              <div className="mt-2 max-h-72 overflow-y-auto">
                <table className="w-full text-sm">
                  <thead className="text-left text-xs text-muted">
                    <tr>
                      <th className="py-1">Strike</th>
                      <th className="py-1 text-right">Net GEX</th>
                    </tr>
                  </thead>
                  <tbody>
                    {gex.profile.map((point) => (
                      <tr key={point.strike} className="border-t border-border">
                        <td className="py-1 font-mono">{point.strike.toFixed(1)}</td>
                        <td
                          className={`py-1 text-right font-mono ${
                            point.net_gex >= 0 ? "text-positive" : "text-negative"
                          }`}
                        >
                          {Math.round(point.net_gex).toLocaleString()}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </details>
          </section>

          <section className="space-y-4">
            {gex.gamma_gap && (
              <div className="rounded-lg border border-border bg-surface p-4">
                <h2 className="mb-2 text-sm font-medium text-muted">Gamma Gap signal</h2>
                <p className="font-mono text-2xl">
                  {gex.gamma_gap.score.toFixed(0)}
                  <span className="text-sm text-muted"> / 120</span>
                </p>
                <p className="mt-1 text-sm">
                  Magnet{" "}
                  <span className="font-mono">{gex.gamma_gap.magnet_strike.toFixed(1)}</span>{" "}
                  · distance{" "}
                  <span className="font-mono">{gex.gamma_gap.distance.toFixed(2)}</span>
                </p>
                {gex.gamma_gap.commentary && (
                  <p className="mt-2 text-sm text-muted">{gex.gamma_gap.commentary}</p>
                )}
              </div>
            )}
            <div className="rounded-lg border border-border bg-surface p-4">
              <h2 className="mb-2 text-sm font-medium text-muted">Interpretation</h2>
              <ul className="space-y-1.5 text-sm">
                {gex.interpretation.map((line) => (
                  <li key={line}>{line.replaceAll("**", "")}</li>
                ))}
              </ul>
            </div>
          </section>
        </div>
      )}

      <p className="text-xs text-muted">
        More views coming to this page: IV skew, vanna/charm exposure, OI heatmap,{" "}
        <Link href="/market" className="text-accent">
          market context
        </Link>
        .
      </p>
    </div>
  );
}
