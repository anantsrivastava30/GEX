"use client";

import { use, useEffect, useState } from "react";
import { api, GexProfile, TickerSnapshot } from "@/lib/api";
import GexStrikeChart from "@/components/charts/GexStrikeChart";
import Panel from "@/components/ui/Panel";
import StatTile from "@/components/ui/StatTile";
import Tabs, { Tab } from "@/components/ui/Tabs";

const TABS: Tab[] = [
  { id: "overview", label: "Overview" },
  { id: "gex", label: "GEX" },
  { id: "vol", label: "Volatility", soon: true },
  { id: "flow", label: "Flow", soon: true },
];

function fmt(v: number | null | undefined, digits = 2) {
  return v == null ? "—" : v.toFixed(digits);
}

export default function StockPage({
  params,
}: {
  params: Promise<{ symbol: string }>;
}) {
  const { symbol } = use(params);
  const upper = symbol.toUpperCase();

  const [tab, setTab] = useState("overview");
  const [snapshot, setSnapshot] = useState<TickerSnapshot | null>(null);
  const [expirations, setExpirations] = useState<string[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [gex, setGex] = useState<GexProfile | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setError(null);
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

  const chg = snapshot?.change_percentage;
  const up = (chg ?? 0) >= 0;

  return (
    <div className="space-y-5">
      {/* Ticker header */}
      <div className="flex flex-wrap items-end justify-between gap-4 border-b border-border pb-4">
        <div className="flex items-baseline gap-3">
          <h1 className="text-2xl font-semibold tracking-tight">{upper}</h1>
          <span className="font-mono text-2xl">
            {snapshot?.last != null ? snapshot.last.toFixed(2) : "—"}
          </span>
          {chg != null && (
            <span
              className={`inline-flex items-center gap-1 rounded-md px-2 py-0.5 font-mono text-sm ${
                up ? "bg-positive/12 text-positive" : "bg-negative/12 text-negative"
              }`}
            >
              <span aria-hidden>{up ? "▲" : "▼"}</span>
              {up ? "+" : ""}
              {chg.toFixed(2)}%
            </span>
          )}
        </div>
        <div className="flex flex-wrap gap-x-6 gap-y-1 text-xs text-muted">
          <span>Bid <span className="font-mono text-foreground">{fmt(snapshot?.bid)}</span></span>
          <span>Ask <span className="font-mono text-foreground">{fmt(snapshot?.ask)}</span></span>
          <span>
            Vol{" "}
            <span className="font-mono text-foreground">
              {snapshot?.volume != null ? snapshot.volume.toLocaleString() : "—"}
            </span>
          </span>
          <span>
            52w{" "}
            <span className="font-mono text-foreground">
              {fmt(snapshot?.week_52_low)}–{fmt(snapshot?.week_52_high)}
            </span>
          </span>
        </div>
      </div>

      <Tabs tabs={TABS} active={tab} onChange={setTab} />

      {error && (
        <p className="rounded-md border border-border bg-surface px-4 py-3 text-sm text-muted">
          {error}
        </p>
      )}

      {/* Expiration selector (GEX-driven views) */}
      {expirations.length > 0 && tab !== "overview" && (
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

      {tab === "overview" && (
        <div className="grid gap-4 lg:grid-cols-3">
          <div className="grid grid-cols-2 gap-3 lg:col-span-1">
            <StatTile label="Last" value={fmt(snapshot?.last)} delta={chg} accent />
            <StatTile
              label="Day change"
              value={snapshot?.change != null ? snapshot.change.toFixed(2) : "—"}
            />
            <StatTile label="Bid" value={fmt(snapshot?.bid)} />
            <StatTile label="Ask" value={fmt(snapshot?.ask)} />
          </div>
          <div className="lg:col-span-2">
            {gex?.gamma_gap ? (
              <Panel title="Gamma Gap signal" className="h-full">
                <div className="flex items-baseline gap-3">
                  <span className="font-mono text-3xl">
                    {gex.gamma_gap.score.toFixed(0)}
                  </span>
                  <span className="text-sm text-muted">/ 120</span>
                  <span
                    className={`ml-auto rounded px-2 py-0.5 text-xs ${
                      gex.gamma_gap.positive_zone
                        ? "bg-positive/12 text-positive"
                        : "bg-negative/12 text-negative"
                    }`}
                  >
                    {gex.gamma_gap.positive_zone ? "Long-gamma zone" : "Short-gamma zone"}
                  </span>
                </div>
                <p className="mt-2 text-sm">
                  Magnet{" "}
                  <span className="font-mono">{gex.gamma_gap.magnet_strike.toFixed(1)}</span>{" "}
                  · distance{" "}
                  <span className="font-mono">{gex.gamma_gap.distance.toFixed(2)}</span>{" "}
                  ({gex.gamma_gap.distance_pct.toFixed(2)}%)
                </p>
                {gex.gamma_gap.commentary && (
                  <p className="mt-2 text-sm text-muted">{gex.gamma_gap.commentary}</p>
                )}
              </Panel>
            ) : (
              <Panel title="Gamma Gap signal" className="h-full">
                <p className="text-sm text-muted">
                  {error ? "Signal unavailable." : "Loading dealer positioning…"}
                </p>
              </Panel>
            )}
          </div>

          {gex && (
            <div className="lg:col-span-3">
              <Panel title="Interpretation">
                <ul className="grid gap-1.5 text-sm sm:grid-cols-2">
                  {gex.interpretation.map((line) => (
                    <li key={line} className="flex gap-2">
                      <span className="mt-1.5 h-1 w-1 shrink-0 rounded-full bg-accent" />
                      {line.replaceAll("**", "")}
                    </li>
                  ))}
                </ul>
              </Panel>
            </div>
          )}
        </div>
      )}

      {tab === "gex" && (
        <>
          {!gex && !error && <p className="text-sm text-muted">Loading GEX profile…</p>}
          {gex && (
            <Panel
              title={`Net GEX by strike — ${gex.expiration}`}
              right={<span className="font-mono text-xs text-muted">spot {gex.spot.toFixed(2)}</span>}
            >
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
            </Panel>
          )}
        </>
      )}
    </div>
  );
}
