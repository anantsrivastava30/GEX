"use client";

import { useState } from "react";
import { api, UnusualResponse } from "@/lib/api";

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

export default function FlowPage() {
  const [symbol, setSymbol] = useState("SPY");
  const [data, setData] = useState<UnusualResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

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
        <section className="rounded-lg border border-border bg-surface p-4">
          <h2 className="mb-3 text-sm font-medium text-muted">
            {data.symbol} · {data.expirations.length} expiration
            {data.expirations.length === 1 ? "" : "s"} · {data.rows.length} strikes
          </h2>
          <div className="max-h-[32rem] overflow-auto">
            <table className="w-full text-sm">
              <thead className="sticky top-0 bg-surface text-left text-xs text-muted">
                <tr>
                  <th className="py-1.5">Strike</th>
                  <th className="py-1.5 text-right">Call vol/OI</th>
                  <th className="py-1.5 text-right">Put vol/OI</th>
                  <th className="py-1.5 text-right">Call OI</th>
                  <th className="py-1.5 text-right">Put OI</th>
                  <th className="py-1.5 text-right">Total vol/OI</th>
                </tr>
              </thead>
              <tbody>
                {data.rows.map((row) => (
                  <tr key={row.strike} className="border-t border-border">
                    <td className="py-1.5 font-mono">{row.strike.toFixed(1)}</td>
                    <td className="py-1.5 text-right font-mono">{fmt(row.vol_oi_call)}</td>
                    <td className="py-1.5 text-right font-mono">{fmt(row.vol_oi_put)}</td>
                    <td className="py-1.5 text-right font-mono">
                      {fmt(row.open_interest_call, 0)}
                    </td>
                    <td className="py-1.5 text-right font-mono">
                      {fmt(row.open_interest_put, 0)}
                    </td>
                    <td className="py-1.5 text-right font-mono text-foreground">
                      {fmt(row.total_vol_oi)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </div>
  );
}
