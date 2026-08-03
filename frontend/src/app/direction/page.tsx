"use client";

import { useEffect, useRef, useState } from "react";
import {
  api,
  DirectionDetail,
  DirectionOverview,
  DirectionSignal,
  IndexDirection,
  ScorecardStatus,
} from "@/lib/api";
import Panel from "@/components/ui/Panel";
import DirectionChart from "@/components/direction/DirectionChart";
import StatePill from "@/components/direction/StatePill";

// O'Neil market direction: the follow-through-day state machine, EMA touch
// levels, RSI timing, and the index-adapted CAN SLIM scorecard across the
// tracked market and sector index universe.

const SCORE_DOT: Record<ScorecardStatus, string> = {
  met: "bg-positive",
  borderline: "bg-warning",
  not_met: "bg-negative",
  unavailable: "bg-faint",
};

const SCORE_BADGE: Record<ScorecardStatus, string> = {
  met: "border-positive/40 bg-positive/10 text-positive",
  borderline: "border-warning/40 bg-warning/10 text-warning",
  not_met: "border-negative/40 bg-negative/10 text-negative",
  unavailable: "border-border bg-surface-2 text-muted",
};

const SCORE_LABEL: Record<ScorecardStatus, string> = {
  met: "Met",
  borderline: "Borderline",
  not_met: "Not met",
  unavailable: "No data",
};

const SIGNAL_BADGE: Record<string, string> = {
  follow_through_day: "border-accent/40 bg-accent/10 text-accent-strong",
  rally_day1: "border-border-strong bg-surface-2 text-foreground",
  rally_failed: "border-negative/40 bg-negative/10 text-negative",
  correction_entered: "border-negative/40 bg-negative/10 text-negative",
  under_pressure: "border-warning/40 bg-warning/10 text-warning",
};

function signalBadge(type: string): string {
  if (type.startsWith("ema_touch_")) {
    return "border-accent/40 bg-accent/10 text-accent-strong";
  }
  return SIGNAL_BADGE[type] ?? "border-border bg-surface-2 text-muted";
}

function signalTypeLabel(type: string): string {
  if (type.startsWith("ema_touch_")) return `${type.replace("ema_touch_", "")}d EMA touch`;
  const labels: Record<string, string> = {
    follow_through_day: "Follow-through",
    rally_day1: "Rally day 1",
    rally_failed: "Rally failed",
    correction_entered: "Correction",
    under_pressure: "Distribution",
    distribution_day: "Distribution day",
  };
  return labels[type] ?? type;
}

function rsiChipClass(zone: string): string {
  if (zone === "oversold") return "border-positive/40 bg-positive/10 text-positive";
  if (zone === "overbought") return "border-warning/40 bg-warning/10 text-warning";
  if (zone === "neutral") return "border-border-strong bg-surface-2 text-muted";
  return "border-border bg-surface-2 text-faint";
}

function IndexCard({
  index,
  selected,
  onSelect,
}: {
  index: IndexDirection;
  selected: boolean;
  onSelect: (symbol: string) => void;
}) {
  return (
    <button
      type="button"
      onClick={() => onSelect(index.symbol)}
      aria-pressed={selected}
      className={`flex flex-col gap-2 rounded-lg border p-3 text-left transition-colors ${
        selected
          ? "border-accent bg-accent/5"
          : "border-border bg-surface hover:bg-surface-hover"
      }`}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <div className="font-mono text-sm font-semibold">{index.symbol}</div>
          <div className="text-[11px] text-muted">{index.label}</div>
        </div>
        <StatePill state={index.state} label={index.state_label} />
      </div>

      {index.domain && (
        <div className="self-start rounded border border-border bg-surface-2 px-1.5 py-0.5 text-[10px] leading-snug text-faint">
          Rules: {index.domain}
        </div>
      )}

      <div className="flex flex-wrap items-center gap-1.5 text-[11px]">
        <span
          className={`inline-flex items-center rounded-full border px-1.5 py-0.5 ${rsiChipClass(index.rsi_zone)}`}
          title={index.timing_label}
        >
          RSI {index.rsi != null ? index.rsi.toFixed(0) : "–"}
        </span>
        {index.emas.map((ema) => (
          <span
            key={ema.period}
            className={`inline-flex items-center gap-0.5 rounded-full border border-border px-1.5 py-0.5 font-mono ${
              ema.above ? "text-positive" : "text-negative"
            }`}
            title={`${ema.distance_pct != null ? `${ema.distance_pct > 0 ? "+" : ""}${ema.distance_pct}%` : ""} vs ${ema.period}-day EMA${ema.touched ? " (touching)" : ""}`}
          >
            {ema.above ? "▴" : "▾"}
            {ema.period}
          </span>
        ))}
      </div>

      <div className="flex items-center justify-between text-[11px] text-muted">
        <span className="inline-flex items-center gap-1" title="Index-adapted CAN SLIM criteria">
          {index.scorecard.map((row) => (
            <span key={row.letter} className="inline-flex flex-col items-center">
              <span className={`h-1.5 w-1.5 rounded-full ${SCORE_DOT[row.status]}`} />
              <span className="mt-0.5 text-[9px] leading-none">{row.letter}</span>
            </span>
          ))}
        </span>
        <span className="font-mono">
          {index.score ? `${index.score.met}/${index.score.scored} met` : ""}
        </span>
      </div>
    </button>
  );
}

export default function DirectionPage() {
  const [overview, setOverview] = useState<DirectionOverview | null>(null);
  const [overviewError, setOverviewError] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [detail, setDetail] = useState<DirectionDetail | null>(null);
  const [detailError, setDetailError] = useState<string | null>(null);
  const [signals, setSignals] = useState<DirectionSignal[] | null>(null);
  const [signalsError, setSignalsError] = useState<string | null>(null);
  const detailRequest = useRef(0);

  useEffect(() => {
    const controller = new AbortController();
    api
      .directionOverview(controller.signal)
      .then((data) => {
        setOverview(data);
        setSelected((current) => current ?? data.benchmark);
      })
      .catch((e) => {
        if (controller.signal.aborted) return;
        setOverviewError(String(e.message ?? e));
      });
    api
      .directionSignals(30, undefined, controller.signal)
      .then((data) => setSignals(data.items))
      .catch((e) => {
        if (controller.signal.aborted) return;
        setSignalsError(String(e.message ?? e));
      });
    return () => controller.abort();
  }, []);

  useEffect(() => {
    if (!selected) return;
    const requestId = ++detailRequest.current;
    const controller = new AbortController();
    api
      .directionDetail(selected, controller.signal)
      .then((data) => {
        if (detailRequest.current !== requestId) return;
        setDetail(data);
      })
      .catch((e) => {
        if (controller.signal.aborted || detailRequest.current !== requestId) return;
        setDetailError(String(e.message ?? e));
      });
    return () => controller.abort();
  }, [selected]);

  // Loading is derived (selected but no data yet); switching symbols clears
  // stale detail in the click handler, keeping setState out of the effect.
  const detailLoading = Boolean(selected) && !detail && !detailError;
  const selectSymbol = (symbol: string) => {
    if (symbol === selected) return;
    setDetail(null);
    setDetailError(null);
    setSelected(symbol);
  };

  const benchmark = overview?.indices.find((i) => i.symbol === overview.benchmark);
  const groups = overview
    ? Array.from(new Set(overview.indices.map((i) => i.group)))
    : [];
  const thresholds = overview?.thresholds;

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <h1 className="text-lg font-semibold">Market Direction</h1>
          <p className="text-sm text-muted">
            O&apos;Neil follow-through-day analysis, EMA levels, and the
            index-adapted CAN SLIM read across market and sector indices.
          </p>
        </div>
        {overview?.provisional && (
          <span className="rounded-full border border-warning/40 bg-warning/10 px-2 py-0.5 text-[11px] text-warning">
            Intraday - today&apos;s bar is provisional until the close
          </span>
        )}
      </div>

      {overviewError && (
        <p className="rounded-md border border-border bg-surface px-4 py-3 text-sm text-muted">
          Backend unavailable: {overviewError}
        </p>
      )}

      {!overview && !overviewError && (
        <div className="space-y-4">
          <div className="h-40 animate-pulse rounded-lg border border-border bg-surface" />
          <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-4">
            {Array.from({ length: 8 }).map((_, i) => (
              <div key={i} className="h-28 animate-pulse rounded-lg border border-border bg-surface" />
            ))}
          </div>
        </div>
      )}

      {overview && benchmark && (
        <Panel
          title={
            <>
              Broad market - {benchmark.label}
              {benchmark.domain && (
                <span className="ml-2 normal-case tracking-normal text-faint">
                  · {benchmark.domain}
                </span>
              )}
            </>
          }
          right={
            <span className="text-[11px] text-muted">
              as of {overview.as_of ?? "n/a"}
            </span>
          }
        >
          <div className="flex flex-col gap-4 lg:flex-row lg:items-start">
            <div className="min-w-0 flex-1 space-y-3">
              <div className="flex flex-wrap items-center gap-3">
                <StatePill state={benchmark.state} label={benchmark.state_label} size="lg" />
                {benchmark.rally_day != null && (
                  <span className="text-sm text-muted">
                    Day {benchmark.rally_day} of the attempt
                  </span>
                )}
                {benchmark.distribution_count != null && (
                  <span className="text-sm text-muted">
                    {benchmark.distribution_count} distribution day(s)
                  </span>
                )}
                <span
                  className={`inline-flex items-center rounded-full border px-2 py-0.5 text-[11px] ${rsiChipClass(benchmark.rsi_zone)}`}
                >
                  RSI {benchmark.rsi != null ? benchmark.rsi.toFixed(0) : "–"} · {benchmark.timing_label}
                </span>
              </div>
              <div className="space-y-1 text-sm text-foreground/90">
                {benchmark.narrative.map((line, i) => (
                  <p key={i}>{line}</p>
                ))}
              </div>
              <p className="text-xs text-muted">{overview.breadth.reading}{" "}
                {overview.breadth.total > 0 &&
                  `(${overview.breadth.uptrend_count} of ${overview.breadth.total} in uptrends, ${overview.breadth.above_ema50} above their 50-day EMA.)`}
              </p>
            </div>

            {overview.durability && (
              <div className="w-full shrink-0 rounded-lg border border-border bg-surface-2 p-3 lg:w-80">
                <div className="mb-2 flex items-center justify-between">
                  <h3 className="text-xs font-medium uppercase tracking-wide text-muted">
                    Bottom durability
                  </h3>
                  <span className="text-[11px] text-muted">
                    FTD {overview.durability.ftd_date} · {overview.durability.sessions_since} sessions ago
                  </span>
                </div>
                <ul className="space-y-2">
                  {overview.durability.checks.map((check) => (
                    <li key={check.name} className="flex items-start gap-2 text-xs">
                      <span
                        aria-hidden
                        className={`mt-0.5 inline-flex h-4 w-4 shrink-0 items-center justify-center rounded-full text-[10px] font-bold ${
                          check.passed
                            ? "bg-positive/15 text-positive"
                            : "bg-negative/15 text-negative"
                        }`}
                      >
                        {check.passed ? "✓" : "✗"}
                      </span>
                      <span>
                        <span className="text-foreground">{check.name}</span>
                        {check.passed ? " (passing)" : " (failing)"}
                        <span className="block text-muted">{check.detail}</span>
                      </span>
                    </li>
                  ))}
                </ul>
                <p className="mt-2 text-[11px] text-muted">
                  Not every follow-through works; a failed one is the accepted
                  cost of catching real bottoms early.
                </p>
              </div>
            )}
          </div>
        </Panel>
      )}

      {overview && groups.length > 0 && (
        <Panel
          title="Index universe"
          right={
            overview.unavailable.length > 0 ? (
              <span className="text-[11px] text-warning">
                No data: {overview.unavailable.join(", ")}
              </span>
            ) : undefined
          }
        >
          <div className="space-y-4">
            {groups.map((group) => (
              <div key={group}>
                <h3 className="mb-2 text-xs font-medium uppercase tracking-wide text-faint">
                  {group}
                </h3>
                <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
                  {overview.indices
                    .filter((i) => i.group === group)
                    .map((index) => (
                      <IndexCard
                        key={index.symbol}
                        index={index}
                        selected={selected === index.symbol}
                        onSelect={selectSymbol}
                      />
                    ))}
                </div>
              </div>
            ))}
          </div>
        </Panel>
      )}

      {selected && (
        <Panel title={`${selected} - state, chart, and scorecard`}>
          {detailLoading && (
            <div className="h-72 animate-pulse rounded-lg border border-border bg-surface-2" />
          )}
          {!detailLoading && detailError && (
            <p className="text-sm text-muted">
              Could not load {selected}: {detailError}
            </p>
          )}
          {!detailLoading && !detailError && detail && (
            <div className="space-y-4">
              <div className="space-y-1 text-sm text-foreground/90">
                {detail.index.narrative.map((line, i) => (
                  <p key={i}>{line}</p>
                ))}
              </div>
              <DirectionChart
                candles={detail.candles}
                emaSeries={detail.ema_series}
                markers={detail.markers}
              />
              <div className="overflow-x-auto">
                <table className="w-full min-w-[560px] text-left text-xs">
                  <thead>
                    <tr className="border-b border-border text-[11px] uppercase tracking-wide text-muted">
                      <th className="py-2 pr-2">Criterion</th>
                      <th className="py-2 pr-2">Status</th>
                      <th className="py-2 pr-2">Value</th>
                      <th className="py-2">Read</th>
                    </tr>
                  </thead>
                  <tbody>
                    {detail.index.scorecard.map((row) => (
                      <tr key={row.letter} className="border-b border-border/60">
                        <td className="py-2 pr-2">
                          <span className="font-mono font-semibold">{row.letter}</span>{" "}
                          <span className="text-muted">{row.name}</span>
                        </td>
                        <td className="py-2 pr-2">
                          <span
                            className={`inline-flex items-center rounded-full border px-1.5 py-0.5 text-[10px] ${SCORE_BADGE[row.status]}`}
                          >
                            {SCORE_LABEL[row.status]}
                          </span>
                        </td>
                        <td className="whitespace-nowrap py-2 pr-2 font-mono">{row.value}</td>
                        <td className="py-2 text-muted">{row.detail}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="text-[11px] text-muted">
                CAN SLIM adapted to indices: earnings letters map to trend and
                volume proxies since an index has no earnings report. See the
                page help for the full mapping.
              </p>
            </div>
          )}
        </Panel>
      )}

      <Panel title="Signal feed" right={
        <span className="text-[11px] text-muted">
          Recorded once per completed session
        </span>
      }>
        {signalsError && (
          <p className="text-sm text-muted">Signals unavailable: {signalsError}</p>
        )}
        {!signals && !signalsError && (
          <div className="space-y-2">
            {Array.from({ length: 3 }).map((_, i) => (
              <div key={i} className="h-12 animate-pulse rounded-md border border-border bg-surface-2" />
            ))}
          </div>
        )}
        {signals && signals.length === 0 && (
          <p className="text-sm text-muted">
            No signals recorded yet. Follow-through days, rally attempts,
            failed rallies, corrections, and 20/50/200-day EMA touches are
            written here after each completed session.
          </p>
        )}
        {signals && signals.length > 0 && (
          <ul className="space-y-2">
            {signals.map((signal) => (
              <li
                key={signal.id}
                className="rounded-md border border-border bg-surface-2 px-3 py-2"
              >
                <div className="flex flex-wrap items-center gap-2">
                  <span
                    className={`inline-flex items-center rounded-full border px-1.5 py-0.5 text-[10px] ${signalBadge(signal.signal_type)}`}
                  >
                    {signalTypeLabel(signal.signal_type)}
                  </span>
                  <span className="font-mono text-xs font-semibold">{signal.symbol}</span>
                  <span className="text-[11px] text-muted">{signal.signal_date}</span>
                </div>
                <p className="mt-1 text-xs text-foreground/90">{signal.message}</p>
              </li>
            ))}
          </ul>
        )}
      </Panel>

      {thresholds && (
        <p className="text-[11px] text-faint">
          Rules in use: follow-through needs a {thresholds.ftd_gain_pct}%+ gain
          on rising volume from day {thresholds.ftd_min_day} of a rally attempt
          (ideal through day {thresholds.ftd_ideal_max_day}) · distribution day
          = {thresholds.distribution_decline_pct}%+ decline on rising volume,
          counted over {thresholds.distribution_lookback} sessions, pressure at{" "}
          {thresholds.distribution_pressure_count} · correction at{" "}
          {thresholds.correction_drawdown_pct}% drawdown · EMA touch band{" "}
          {thresholds.ema_touch_band_pct}% · RSI({thresholds.rsi_period}).
          Research tooling, not investment advice.
        </p>
      )}
    </div>
  );
}
