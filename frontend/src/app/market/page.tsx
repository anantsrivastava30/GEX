"use client";

import { useEffect, useState } from "react";
import { api, MarketOverview } from "@/lib/api";

function pct(value: number | null | undefined) {
  if (value == null) return "—";
  const sign = value > 0 ? "+" : "";
  return `${sign}${value.toFixed(2)}%`;
}

function changeClass(value: number | null | undefined) {
  if (value == null) return "text-muted";
  return value >= 0 ? "text-positive" : "text-negative";
}

export default function MarketPage() {
  const [overview, setOverview] = useState<MarketOverview | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.marketOverview().then(setOverview).catch((e) => setError(String(e.message ?? e)));
  }, []);

  return (
    <div className="space-y-6">
      <h1 className="text-lg font-semibold">Market Overview</h1>

      {error && (
        <p className="rounded-md border border-border bg-surface px-4 py-3 text-sm text-muted">
          Backend unavailable: {error}. Start it with{" "}
          <code className="font-mono">uvicorn backend.app.main:app --port 8000</code>.
        </p>
      )}

      {!overview && !error && <p className="text-sm text-muted">Loading…</p>}

      {overview && (
        <div className="grid gap-4 lg:grid-cols-2">
          <section className="rounded-lg border border-border bg-surface p-4">
            <h2 className="mb-3 text-sm font-medium text-muted">Volatility & Rates</h2>
            <table className="w-full text-sm">
              <tbody>
                <tr className="border-b border-border">
                  <td className="py-2">VIX</td>
                  <td className="py-2 text-right font-mono">
                    {overview.vix?.spot?.toFixed(2) ?? "—"}
                  </td>
                  <td className={`py-2 text-right font-mono ${changeClass(overview.vix?.["1d_return"])}`}>
                    {pct(overview.vix?.["1d_return"])}
                  </td>
                </tr>
                <tr>
                  <td className="py-2">10Y Treasury</td>
                  <td className="py-2 text-right font-mono">
                    {typeof overview.yields?.spot === "number"
                      ? overview.yields.spot.toFixed(2)
                      : "—"}
                  </td>
                  <td
                    className={`py-2 text-right font-mono ${changeClass(
                      typeof overview.yields?.["1d_return"] === "number"
                        ? overview.yields["1d_return"]
                        : null,
                    )}`}
                  >
                    {pct(
                      typeof overview.yields?.["1d_return"] === "number"
                        ? overview.yields["1d_return"]
                        : null,
                    )}
                  </td>
                </tr>
              </tbody>
            </table>
          </section>

          <section className="rounded-lg border border-border bg-surface p-4">
            <h2 className="mb-3 text-sm font-medium text-muted">Futures</h2>
            <table className="w-full text-sm">
              <tbody>
                {Object.entries(overview.futures).map(([symbol, quote]) => (
                  <tr key={symbol} className="border-b border-border last:border-0">
                    <td className="py-2 font-mono">{symbol}</td>
                    <td className="py-2 text-right font-mono">
                      {quote.last?.toFixed(2) ?? "—"}
                    </td>
                    <td className={`py-2 text-right font-mono ${changeClass(quote.change_pct)}`}>
                      {pct(quote.change_pct)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>
        </div>
      )}
    </div>
  );
}
