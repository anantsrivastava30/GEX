"use client";

import { use, useEffect, useState } from "react";
import {
  api,
  APIError,
  Candle,
  DailyMetricsPoint,
  ExposureResponse,
  GexProfile,
  IVRankResponse,
  MaxPainResponse,
  SkewResponse,
  TermStructureResponse,
  TickerSnapshot,
} from "@/lib/api";
import GexStrikeChart from "@/components/charts/GexStrikeChart";
import StrikeBarChart from "@/components/charts/StrikeBarChart";
import CandlestickChart from "@/components/charts/CandlestickChart";
import HistoricalGexChart from "@/components/charts/HistoricalGexChart";
import SkewChart from "@/components/charts/SkewChart";
import TermStructureChart from "@/components/charts/TermStructureChart";
import Panel from "@/components/ui/Panel";
import StatTile from "@/components/ui/StatTile";
import Tabs, { Tab } from "@/components/ui/Tabs";

const TABS: Tab[] = [
  { id: "overview", label: "Overview" },
  { id: "gex", label: "GEX" },
  { id: "vol", label: "Volatility" },
  { id: "flow", label: "Flow", soon: true },
];

function fmt(v: number | null | undefined, digits = 2) {
  return v == null ? "—" : v.toFixed(digits);
}

function fmtCompact(v: number) {
  const abs = Math.abs(v);
  if (abs >= 1e9) return `${(v / 1e9).toFixed(1)}B`;
  if (abs >= 1e6) return `${(v / 1e6).toFixed(1)}M`;
  if (abs >= 1e3) return `${(v / 1e3).toFixed(1)}K`;
  return v.toFixed(1);
}

export default function StockPage({
  params,
}: {
  params: Promise<{ symbol: string }>;
}) {
  const { symbol } = use(params);
  const upper = symbol.toUpperCase();

  return <StockDataPage key={upper} upper={upper} />;
}

function StockDataPage({ upper }: { upper: string }) {

  const [tab, setTab] = useState("overview");
  const [snapshot, setSnapshot] = useState<TickerSnapshot | null>(null);
  const [expirations, setExpirations] = useState<string[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [gex, setGex] = useState<GexProfile | null>(null);
  const [candles, setCandles] = useState<Candle[] | null>(null);
  const [exposure, setExposure] = useState<ExposureResponse | null>(null);
  const [maxPain, setMaxPain] = useState<MaxPainResponse | null>(null);
  const [skew, setSkew] = useState<SkewResponse | null | "error">(null);
  const [term, setTerm] = useState<TermStructureResponse | null | "error">(null);
  const [ivRank, setIvRank] = useState<
    IVRankResponse | null | "none" | "error"
  >(null);
  const [metrics, setMetrics] = useState<DailyMetricsPoint[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    const { signal } = controller;

    api
      .tickerSnapshot(upper, signal)
      .then(setSnapshot)
      .catch((e) => {
        if (e.name !== "AbortError") setError(e.message);
      });
    api
      .candles(upper, 120, signal)
      .then(setCandles)
      .catch((e) => {
        if (e.name !== "AbortError") setCandles([]);
      });
    api
      .termStructure(upper, 6, signal)
      .then(setTerm)
      .catch((e) => {
        if (e.name !== "AbortError") setTerm("error");
      });
    api
      .ivRank(upper, signal)
      .then(setIvRank)
      .catch((e) => {
        if (e.name === "AbortError") return;
        setIvRank(e instanceof APIError && e.status === 404 ? "none" : "error");
      });
    api
      .dailyMetrics(upper, signal)
      .then(setMetrics)
      .catch((e) => {
        if (e.name !== "AbortError") setMetrics([]);
      });
    api
      .expirations(upper, signal)
      .then((exps) => {
        setExpirations(exps);
        if (exps.length) setSelected(exps[0]);
      })
      .catch((e) => {
        if (e.name !== "AbortError") setError(e.message);
      });

    return () => controller.abort();
  }, [upper]);

  useEffect(() => {
    if (!selected) return;
    const controller = new AbortController();
    const { signal } = controller;

    api
      .gexProfile(upper, selected, 35, signal)
      .then(setGex)
      .catch((e) => {
        if (e.name !== "AbortError") setError(e.message);
      });
    api
      .exposure(upper, [selected], 35, signal)
      .then(setExposure)
      .catch((e) => {
        if (e.name !== "AbortError") setExposure(null);
      });
    api
      .maxPain(upper, selected, signal)
      .then(setMaxPain)
      .catch((e) => {
        if (e.name !== "AbortError") setMaxPain(null);
      });
    api
      .skew(upper, selected, signal)
      .then(setSkew)
      .catch((e) => {
        if (e.name !== "AbortError") setSkew("error");
      });

    return () => controller.abort();
  }, [upper, selected]);

  function selectExpiration(expiration: string) {
    if (expiration === selected) return;
    setError(null);
    setGex(null);
    setExposure(null);
    setMaxPain(null);
    setSkew(null);
    setSelected(expiration);
  }

  const chg = snapshot?.change_percentage;
  const up = (chg ?? 0) >= 0;
  const spotValue = snapshot?.last ?? gex?.spot ?? null;

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

      {/* Expiration selector (chain-driven views) */}
      {expirations.length > 0 && tab !== "overview" && (
        <div className="flex flex-wrap gap-2">
          {expirations.slice(0, 8).map((exp) => (
            <button
              key={exp}
              onClick={() => selectExpiration(exp)}
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
          <div className="lg:col-span-3">
            <Panel
              title="Price — daily"
              right={
                candles && candles.length > 0 ? (
                  <span className="font-mono text-xs text-muted">
                    {candles.length} sessions
                  </span>
                ) : null
              }
            >
              {candles == null ? (
                <div className="h-56 animate-pulse rounded bg-surface-2" />
              ) : candles.length === 0 ? (
                <p className="py-8 text-center text-sm text-muted">
                  No price history available.
                </p>
              ) : (
                <CandlestickChart candles={candles} />
              )}
            </Panel>
          </div>
          <div className="grid grid-cols-2 gap-3 lg:col-span-1">
            <StatTile label="Last" value={fmt(snapshot?.last)} delta={chg} accent />
            <StatTile
              label="Day change"
              value={snapshot?.change != null ? snapshot.change.toFixed(2) : "—"}
            />
            <StatTile label="Bid" value={fmt(snapshot?.bid)} />
            <StatTile label="Ask" value={fmt(snapshot?.ask)} />
            <StatTile
              label="IV rank"
              value={
                ivRank && ivRank !== "none" && ivRank !== "error"
                  ? ivRank.iv_rank.toFixed(0)
                  : "—"
              }
              sub={
                ivRank === "none"
                  ? "Accruing snapshot history"
                  : ivRank === "error"
                    ? "History unavailable"
                    : ivRank
                    ? `${ivRank.days_of_history}d of history`
                    : "Loading…"
              }
            />
            <StatTile
              label="ATM IV"
              value={
                ivRank && ivRank !== "none" && ivRank !== "error"
                  ? `${(ivRank.iv * 100).toFixed(1)}%`
                  : term && term !== "error" && term.points.length
                    ? `${(term.points[0].atm_iv * 100).toFixed(1)}%`
                    : "—"
              }
            />
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
                  ({(gex.gamma_gap.distance_pct * 100).toFixed(2)}%)
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
        <div className="space-y-4">
          {!gex && !error && <p className="text-sm text-muted">Loading GEX profile…</p>}
          {gex && (
            <div className="grid gap-4 lg:grid-cols-3">
              <div className="lg:col-span-2">
                <Panel
                  title={`Net GEX by strike — ${gex.expiration}`}
                  right={
                    <span className="font-mono text-xs text-muted">
                      spot {gex.spot.toFixed(2)}
                    </span>
                  }
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
              </div>
              <div className="space-y-3 lg:col-span-1">
                <StatTile
                  label="Max pain"
                  value={maxPain ? maxPain.max_pain.toFixed(1) : "—"}
                  sub={
                    maxPain
                      ? `Spot ${Math.abs(
                          ((maxPain.spot - maxPain.max_pain) / maxPain.spot) * 100,
                        ).toFixed(2)}% ${
                          maxPain.spot >= maxPain.max_pain ? "above" : "below"
                        } max pain`
                      : "Needs open interest"
                  }
                  accent
                />
                {gex.gamma_gap && (
                  <StatTile
                    label="Gamma gap score"
                    value={gex.gamma_gap.score.toFixed(0)}
                    sub={`Magnet ${gex.gamma_gap.magnet_strike.toFixed(1)}`}
                  />
                )}
                {exposure && (
                  <StatTile
                    label="Risk-free rate"
                    value={`${(exposure.risk_free_rate * 100).toFixed(2)}%`}
                    sub="13-week T-bill (^IRX)"
                  />
                )}
              </div>
            </div>
          )}

          {exposure && (
            <div className="grid gap-4 lg:grid-cols-2">
              <Panel
                title={`Vanna exposure — ${exposure.expirations[0]}`}
                right={
                  <span className="font-mono text-xs text-muted">
                    dDelta / dVol · local Black-Scholes
                  </span>
                }
              >
                <StrikeBarChart
                  points={exposure.points.map((p) => ({ strike: p.strike, value: p.vanna }))}
                  spot={exposure.spot}
                  positiveLabel="Net vanna (+)"
                  negativeLabel="Net vanna (−)"
                  formatValue={fmtCompact}
                  maxHeightClass="max-h-[22rem]"
                />
              </Panel>
              <Panel
                title={`Charm exposure — ${exposure.expirations[0]}`}
                right={
                  <span className="font-mono text-xs text-muted">
                    dDelta / dTime (per year)
                  </span>
                }
              >
                <StrikeBarChart
                  points={exposure.points.map((p) => ({ strike: p.strike, value: p.charm }))}
                  spot={exposure.spot}
                  positiveLabel="Net charm (+)"
                  negativeLabel="Net charm (−)"
                  formatValue={fmtCompact}
                  maxHeightClass="max-h-[22rem]"
                />
              </Panel>
            </div>
          )}

          <Panel
            title="Net GEX history"
            right={
              metrics && metrics.length > 0 ? (
                <span className="font-mono text-xs text-muted">
                  {metrics.length} daily snapshots
                </span>
              ) : null
            }
          >
            {metrics == null ? (
              <div className="h-44 animate-pulse rounded bg-surface-2" />
            ) : metrics.filter((point) => point.net_gex_total != null).length >= 2 ? (
              <HistoricalGexChart points={metrics} />
            ) : (
              <p className="py-8 text-center text-sm text-muted">
                GEX history will appear after at least two daily snapshots are captured.
              </p>
            )}
          </Panel>
        </div>
      )}

      {tab === "vol" && (
        <div className="space-y-4">
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <StatTile
              label="IV rank"
              value={
                ivRank && ivRank !== "none" && ivRank !== "error"
                  ? ivRank.iv_rank.toFixed(0)
                  : "—"
              }
              sub={
                ivRank === "none"
                  ? "Accruing snapshot history"
                  : ivRank === "error"
                    ? "History unavailable"
                    : ivRank
                    ? `Range ${(ivRank.iv_low * 100).toFixed(1)}–${(ivRank.iv_high * 100).toFixed(1)}%`
                    : "Loading…"
              }
              accent
            />
            <StatTile
              label="IV percentile"
              value={
                ivRank && ivRank !== "none" && ivRank !== "error"
                  ? `${ivRank.iv_percentile.toFixed(0)}%`
                  : "—"
              }
              sub={
                ivRank && ivRank !== "none" && ivRank !== "error"
                  ? `${ivRank.days_of_history}d of history`
                  : undefined
              }
            />
            <StatTile
              label="ATM IV"
              value={
                ivRank && ivRank !== "none" && ivRank !== "error"
                  ? `${(ivRank.iv * 100).toFixed(1)}%`
                  : term && term !== "error" && term.points.length
                    ? `${(term.points[0].atm_iv * 100).toFixed(1)}%`
                    : "—"
              }
            />
            <StatTile
              label="Term slope"
              value={
                term && term !== "error" && term.points.length >= 2
                  ? `${(
                      (term.points[term.points.length - 1].atm_iv -
                        term.points[0].atm_iv) *
                      100
                    ).toFixed(1)}pt`
                  : "—"
              }
              sub={
                term && term !== "error" && term.points.length >= 2
                  ? term.points[term.points.length - 1].atm_iv >= term.points[0].atm_iv
                    ? "Contango (far > near)"
                    : "Inverted (near > far)"
                  : undefined
              }
            />
          </div>

          <Panel
            title={`IV skew by strike — ${selected ?? ""}`}
            right={
              skew && skew !== "error" ? (
                <span className="font-mono text-xs text-muted">
                  {skew.points.length} strikes
                </span>
              ) : null
            }
          >
            {skew && skew !== "error" && skew.points.length >= 2 && spotValue != null ? (
              <SkewChart points={skew.points} spot={spotValue} />
            ) : (
              <p className="py-8 text-center text-sm text-muted">
                {skew === null ? "Loading IV skew…" : "IV skew unavailable."}
              </p>
            )}
          </Panel>

          <Panel title="IV term structure — ATM by expiration">
            {term && term !== "error" && term.points.length >= 2 ? (
              <TermStructureChart points={term.points} />
            ) : (
              <p className="py-8 text-center text-sm text-muted">
                {term === null
                  ? "Loading term structure…"
                  : term === "error"
                    ? "Term structure unavailable."
                    : "Not enough expirations."}
              </p>
            )}
          </Panel>
        </div>
      )}
    </div>
  );
}
