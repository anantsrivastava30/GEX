"use client";

import { use, useEffect, useState } from "react";
import {
  api,
  APIError,
  Candle,
  DailyMetricsPoint,
  DeltaProjectionResponse,
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
import DeltaProjectionChart from "@/components/charts/DeltaProjectionChart";
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

type ExpirationStatus = "loading" | "ready" | "empty" | "error";
type ResourceStatus = "loading" | "ready" | "error";

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
  const [expirationStatus, setExpirationStatus] = useState<ExpirationStatus>("loading");
  const [expirationError, setExpirationError] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [selectedExpirations, setSelectedExpirations] = useState<string[]>([]);
  const [compareMode, setCompareMode] = useState(false);
  const [gex, setGex] = useState<GexProfile | null>(null);
  const [gexProfiles, setGexProfiles] = useState<GexProfile[] | null>(null);
  const [gexError, setGexError] = useState<string | null>(null);
  const [candles, setCandles] = useState<Candle[] | null>(null);
  const [exposure, setExposure] = useState<ExposureResponse | null>(null);
  const [maxPain, setMaxPain] = useState<MaxPainResponse | null>(null);
  const [maxPainStatus, setMaxPainStatus] = useState<ResourceStatus>("loading");
  const [deltaProj, setDeltaProj] = useState<DeltaProjectionResponse | null>(null);
  const [skew, setSkew] = useState<SkewResponse | null | "error">(null);
  const [term, setTerm] = useState<TermStructureResponse | null | "error">(null);
  const [ivRank, setIvRank] = useState<
    IVRankResponse | null | "none" | "error"
  >(null);
  const [metrics, setMetrics] = useState<DailyMetricsPoint[] | null>(null);
  const [metricsStatus, setMetricsStatus] = useState<ResourceStatus>("loading");
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
      .then((result) => {
        setMetrics(result);
        setMetricsStatus("ready");
      })
      .catch((e) => {
        if (e.name !== "AbortError") setMetricsStatus("error");
      });
    api
      .expirations(upper, signal)
      .then((exps) => {
        setExpirations(exps);
        setExpirationStatus(exps.length ? "ready" : "empty");
        if (exps.length) {
          setSelected(exps[0]);
          setSelectedExpirations([exps[0]]);
        }
      })
      .catch((e) => {
        if (e.name !== "AbortError") {
          setExpirationStatus("error");
          setExpirationError(e.message);
        }
      });

    return () => controller.abort();
  }, [upper]);

  useEffect(() => {
    if (!selected || !selectedExpirations.length) return;
    const controller = new AbortController();
    const { signal } = controller;

    Promise.allSettled(
      selectedExpirations.map((expiration) =>
        api.gexProfile(upper, expiration, 35, signal),
      ),
    ).then((results) => {
      if (signal.aborted) return;
      const profiles = results.flatMap((result) =>
        result.status === "fulfilled" ? [result.value] : [],
      );
      setGexProfiles(profiles);
      setGex(
        profiles.find((profile) => profile.expiration === selected) ??
          profiles[0] ??
          null,
      );
      const failed = results.length - profiles.length;
      setGexError(
        failed
          ? `${failed} selected expiration${failed === 1 ? "" : "s"} could not be loaded.`
          : null,
      );
    });

    return () => controller.abort();
  }, [upper, selected, selectedExpirations]);

  useEffect(() => {
    if (!selected) return;
    const controller = new AbortController();
    const { signal } = controller;

    api
      .exposure(upper, [selected], 35, signal)
      .then(setExposure)
      .catch((e) => {
        if (e.name !== "AbortError") setExposure(null);
      });
    api
      .maxPain(upper, selected, signal)
      .then((result) => {
        setMaxPain(result);
        setMaxPainStatus("ready");
      })
      .catch((e) => {
        if (e.name !== "AbortError") setMaxPainStatus("error");
      });
    api
      .skew(upper, selected, signal)
      .then(setSkew)
      .catch((e) => {
        if (e.name !== "AbortError") setSkew("error");
      });
    api
      .deltaProjection(upper, selected, 35, signal)
      .then(setDeltaProj)
      .catch((e) => {
        if (e.name !== "AbortError") setDeltaProj(null);
      });

    return () => controller.abort();
  }, [upper, selected]);

  function clearPrimaryExpirationData() {
    setError(null);
    setExposure(null);
    setMaxPain(null);
    setMaxPainStatus("loading");
    setSkew(null);
    setDeltaProj(null);
  }

  function selectExpiration(
    expiration: string,
    event: React.MouseEvent<HTMLButtonElement>,
  ) {
    const compare =
      tab === "gex" && (compareMode || event.ctrlKey || event.metaKey);
    setGexError(null);

    if (compare) {
      if (selectedExpirations.includes(expiration)) {
        if (selectedExpirations.length === 1) return;
        const next = selectedExpirations.filter((item) => item !== expiration);
        setGexProfiles(null);
        setGex(null);
        setSelectedExpirations(next);
        if (expiration === selected) {
          clearPrimaryExpirationData();
          setSelected(next[0]);
        }
      } else {
        setGexProfiles(null);
        setGex(null);
        setSelectedExpirations([...selectedExpirations, expiration]);
      }
      return;
    }

    if (expiration === selected && selectedExpirations.length === 1) return;
    if (expiration !== selected) clearPrimaryExpirationData();
    setGexProfiles(null);
    setGex(null);
    setSelectedExpirations([expiration]);
    setSelected(expiration);
  }

  const chg = snapshot?.change_percentage;
  const up = (chg ?? 0) >= 0;
  const spotValue = snapshot?.last ?? gex?.spot ?? null;
  const gexSeries = (gexProfiles ?? []).map((profile) => ({
    expiration: profile.expiration,
    profile: profile.profile,
  }));
  const gexValues = new Map(
    gexSeries.map((item) => [
      item.expiration,
      new Map(item.profile.map((point) => [point.strike, point.net_gex])),
    ]),
  );
  const comparisonStrikes = [
    ...new Set(gexSeries.flatMap((item) => item.profile.map((point) => point.strike))),
  ].sort((a, b) => b - a);

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
      <p className="text-xs text-faint">Quote, price history, expirations, and the nearest-expiration analytics load automatically.</p>

      <Tabs tabs={TABS} active={tab} onChange={setTab} />

      {error && (
        <p className="rounded-md border border-border bg-surface px-4 py-3 text-sm text-muted">
          {error}
        </p>
      )}

      {(tab !== "overview" || expirationStatus === "error") && expirationStatus !== "ready" && (
        <p role={expirationStatus === "error" ? "alert" : "status"} className={`rounded border px-4 py-3 text-sm ${expirationStatus === "error" ? "border-rose-500/50 text-rose-300" : "border-border bg-surface text-muted"}`}>
          {expirationStatus === "loading"
            ? "Loading option expirations..."
            : expirationStatus === "empty"
              ? `No option expirations are available for ${upper}.`
              : `Unable to load option expirations${expirationError ? `: ${expirationError}` : "."}`}
        </p>
      )}

      {/* Expiration selector (chain-driven views) */}
      {expirations.length > 0 && tab !== "overview" && (
        <div className="space-y-2">
          <div className="flex flex-wrap gap-2">
            {expirations.slice(0, 8).map((exp) => {
              const included =
                tab === "gex" ? selectedExpirations.includes(exp) : exp === selected;
              const primary = exp === selected;
              return (
                <button
                  key={exp}
                  type="button"
                  aria-pressed={included}
                  title={
                    tab === "gex"
                      ? "Click for one expiration; Ctrl/Cmd-click to add or remove a comparison"
                      : undefined
                  }
                  onClick={(event) => selectExpiration(exp, event)}
                  className={`rounded-full border px-3 py-1 font-mono text-xs transition-colors ${
                    primary
                      ? "border-accent bg-accent/15 text-foreground"
                      : included
                        ? "border-accent/60 bg-accent/5 text-foreground"
                        : "border-border bg-surface text-muted hover:text-foreground"
                  }`}
                >
                  {exp}
                </button>
              );
            })}
            {tab === "gex" && (
              <button
                type="button"
                aria-pressed={compareMode}
                onClick={() => setCompareMode((active) => !active)}
                className={`rounded-full border px-3 py-1 text-xs transition-colors ${
                  compareMode
                    ? "border-warning/70 bg-warning/10 text-warning"
                    : "border-border bg-surface text-muted hover:text-foreground"
                }`}
              >
                Compare {compareMode ? "on" : "off"}
              </button>
            )}
          </div>
          {tab === "gex" && (
            <p className="text-xs text-faint">
              Click selects one expiration. Ctrl-click or Cmd-click adds/removes
              comparisons; touch users can enable Compare. The primary expiration (
              {selected ?? "none"}) drives max pain, greeks, and delta projection.
            </p>
          )}
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
                  {expirationStatus === "loading" || (expirationStatus === "ready" && gexProfiles === null)
                    ? "Loading dealer positioning..."
                    : expirationStatus === "empty"
                      ? "No option expirations are available for this ticker."
                      : expirationStatus === "error" || gexError
                        ? "Dealer-positioning signal unavailable."
                        : gex
                          ? "No gamma-gap signal was produced for this expiration."
                          : "Dealer-positioning signal unavailable."}
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
          {gexError && (
            <p className="rounded-md border border-warning/40 bg-warning/5 px-3 py-2 text-sm text-warning">
              {gexError}
            </p>
          )}
          {expirationStatus === "ready" && gexProfiles === null && (
            <p className="text-sm text-muted">
              Loading {selectedExpirations.length > 1 ? "GEX profiles" : "GEX profile"}…
            </p>
          )}
          {expirationStatus === "ready" && gexProfiles !== null && !gex && (
            <p className="rounded border border-rose-500/40 px-4 py-3 text-sm text-rose-300">No selected GEX profile could be loaded.</p>
          )}
          {gex && (
            <div className="grid gap-4 lg:grid-cols-3">
              <div className="lg:col-span-2">
                <Panel
                  title={
                    gexSeries.length > 1
                      ? `Net GEX by strike — ${gexSeries.length} expirations`
                      : `Net GEX by strike — ${gex.expiration}`
                  }
                  right={
                    <span className="font-mono text-xs text-muted">
                      spot {gex.spot.toFixed(2)}
                    </span>
                  }
                >
                  {gexSeries.some((item) => item.profile.length) ? <GexStrikeChart series={gexSeries} spot={gex.spot} /> : <p className="py-10 text-center text-sm text-muted">No strike-level GEX points are available for this selection.</p>}
                  {gexSeries.some((item) => item.profile.length) && <details className="mt-3 text-xs text-muted">
                    <summary className="cursor-pointer select-none hover:text-foreground">
                      Table view
                    </summary>
                    <div className="mt-2 max-h-72 overflow-auto">
                      <table className="w-full text-sm">
                        <thead className="text-left text-xs text-muted">
                          <tr>
                            <th className="py-1">Strike</th>
                            {gexSeries.map((item) => (
                              <th
                                key={item.expiration}
                                className="whitespace-nowrap py-1 text-right"
                              >
                                {item.expiration}
                              </th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {comparisonStrikes.map((strike) => (
                            <tr key={strike} className="border-t border-border">
                              <td className="py-1 font-mono">{strike.toFixed(1)}</td>
                              {gexSeries.map((item) => {
                                const value = gexValues.get(item.expiration)?.get(strike);
                                return (
                                  <td
                                    key={item.expiration}
                                    className={`whitespace-nowrap py-1 text-right font-mono ${
                                      value == null
                                        ? "text-faint"
                                        : value >= 0
                                          ? "text-positive"
                                          : "text-negative"
                                    }`}
                                  >
                                    {value == null
                                      ? "—"
                                      : Math.round(value).toLocaleString()}
                                  </td>
                                );
                              })}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </details>}
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
                      : maxPainStatus === "loading"
                        ? "Loading max pain..."
                        : maxPainStatus === "error"
                          ? "Max pain unavailable"
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

          {deltaProj && deltaProj.bars.length > 0 && (
            <Panel
              title={`Intraday delta projection — ${deltaProj.expiration}`}
              right={
                <span className="font-mono text-xs text-muted">
                  {deltaProj.bars.length} bars · ±{deltaProj.offset} strikes
                </span>
              }
            >
              <DeltaProjectionChart data={deltaProj} />
              <p className="mt-2 text-xs text-muted">
                Net dealer delta exposure of the current chain within ±
                {deltaProj.offset} of each bar&apos;s spot; the dashed extension is a
                linear trend projected to the 16:00 close.
              </p>
            </Panel>
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
            {metricsStatus === "loading" ? (
              <div className="h-44 animate-pulse rounded bg-surface-2" />
            ) : metricsStatus === "error" ? (
              <p className="py-8 text-center text-sm text-muted">GEX history is currently unavailable.</p>
            ) : metrics && metrics.filter((point) => point.net_gex_total != null).length >= 2 ? (
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
            title={selected ? `IV skew by strike — ${selected}` : "IV skew by strike"}
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
                 {expirationStatus === "loading" || (expirationStatus === "ready" && skew === null)
                   ? "Loading IV skew..."
                   : expirationStatus === "empty"
                     ? "No option expirations are available."
                     : "IV skew unavailable."}
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
