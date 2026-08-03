"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { api, MarketOverview } from "@/lib/api";
import StatTile from "@/components/ui/StatTile";

function num(value: number | null | undefined) {
  return typeof value === "number" ? value : null;
}

export default function MarketPage() {
  const [overview, setOverview] = useState<MarketOverview | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.marketOverview().then(setOverview).catch((e) => setError(String(e.message ?? e)));
  }, []);

  const vixSpot = num(overview?.vix?.spot);
  const vixRet = num(overview?.vix?.["1d_return"]);
  const yldSpot = num(overview?.yields?.spot);
  const yldRet = num(overview?.yields?.["1d_return"]);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-lg font-semibold">Market Overview</h1>
        <p className="text-sm text-muted">Volatility, rates, and futures at a glance.</p>
      </div>

      <Link
        href="/direction"
        className="flex items-center justify-between gap-3 rounded-lg border border-border bg-surface px-4 py-3 transition-colors hover:bg-surface-hover"
      >
        <span>
          <span className="block text-sm font-medium">Market Direction</span>
          <span className="block text-xs text-muted">
            Follow-through days, distribution counts, EMA levels, and sector
            breadth - the O&apos;Neil uptrend/bottom read.
          </span>
        </span>
        <span aria-hidden className="text-muted">→</span>
      </Link>

      {error && (
        <p className="rounded-md border border-border bg-surface px-4 py-3 text-sm text-muted">
          Backend unavailable: {error}. Start it with{" "}
          <code className="font-mono">uvicorn backend.app.main:app --port 8000</code>.
        </p>
      )}

      {!overview && !error && (
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
          {Array.from({ length: 6 }).map((_, i) => (
            <div key={i} className="h-20 animate-pulse rounded-lg border border-border bg-surface" />
          ))}
        </div>
      )}

      {overview && (
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
          <StatTile
            label="VIX"
            value={vixSpot != null ? vixSpot.toFixed(2) : "—"}
            delta={vixRet}
            accent
            sub="Implied volatility"
          />
          <StatTile
            label="10Y Treasury"
            value={yldSpot != null ? `${yldSpot.toFixed(2)}%` : "—"}
            delta={yldRet}
            sub="Benchmark yield"
          />
          {Object.entries(overview.futures).map(([symbol, quote]) => (
            <StatTile
              key={symbol}
              label={symbol}
              value={num(quote.last) != null ? (quote.last as number).toFixed(2) : "—"}
              delta={num(quote.change_pct)}
              sub="Futures"
            />
          ))}
        </div>
      )}
    </div>
  );
}
