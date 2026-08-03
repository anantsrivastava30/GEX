"use client";

import { useEffect, useState } from "react";
import { api, LeadersResponse, StockLeader, StockReadiness } from "@/lib/api";
import Panel from "@/components/ui/Panel";
import StatePill from "@/components/direction/StatePill";
import { SCORE_BADGE, SCORE_DOT, SCORE_LABEL } from "@/components/direction/scorecard";

// CAN SLIM stock leaders: which stocks are flagged to buy, and why. Every
// flag is gated by the market-direction state - O'Neil's first rule.

const READINESS_BADGE: Record<StockReadiness, string> = {
  buy_candidate: "border-positive/40 bg-positive/10 text-positive",
  extended: "border-warning/40 bg-warning/10 text-warning",
  near_pivot: "border-accent/40 bg-accent/10 text-accent-strong",
  wait_market: "border-warning/40 bg-warning/10 text-warning",
  not_ready: "border-border bg-surface-2 text-muted",
  insufficient_data: "border-border bg-surface-2 text-faint",
};

function fmtPct(value: number | null | undefined, signed = true): string {
  if (value == null) return "–";
  const sign = signed && value > 0 ? "+" : "";
  return `${sign}${value.toFixed(0)}%`;
}

function GateBanner({ data }: { data: LeadersResponse }) {
  const tone = data.gate_open
    ? "border-positive/40 bg-positive/5"
    : data.market_state === "correction"
      ? "border-negative/40 bg-negative/5"
      : "border-warning/40 bg-warning/5";
  return (
    <div className={`flex flex-wrap items-center gap-3 rounded-lg border px-4 py-3 ${tone}`}>
      <StatePill state={data.market_state} label={data.market_state_label} />
      <p className="text-sm text-foreground/90">{data.gate_message}</p>
    </div>
  );
}

function LeaderRow({
  item,
  selected,
  onSelect,
}: {
  item: StockLeader;
  selected: boolean;
  onSelect: (symbol: string) => void;
}) {
  return (
    <tr
      onClick={() => onSelect(item.symbol)}
      aria-selected={selected}
      className={`cursor-pointer border-b border-border/60 transition-colors ${
        selected ? "bg-accent/5" : "hover:bg-surface-hover"
      }`}
    >
      <td className="py-2 pr-2">
        <span className="font-mono text-xs font-semibold">{item.symbol}</span>
        {item.name && (
          <span className="ml-1.5 hidden text-[11px] text-muted lg:inline">
            {item.name}
          </span>
        )}
      </td>
      <td className="py-2 pr-2">
        <span
          className={`inline-flex items-center whitespace-nowrap rounded-full border px-1.5 py-0.5 text-[10px] ${READINESS_BADGE[item.readiness]}`}
        >
          {item.readiness_label}
        </span>
      </td>
      <td className="py-2 pr-2">
        <span className="inline-flex items-center gap-1" title="C A N S L I M statuses">
          {item.scorecard.map((row) => (
            <span
              key={row.letter}
              title={`${row.letter}: ${SCORE_LABEL[row.status]}`}
              className={`h-1.5 w-1.5 rounded-full ${SCORE_DOT[row.status]}`}
            />
          ))}
        </span>
      </td>
      <td className="whitespace-nowrap py-2 pr-2 font-mono">
        {item.score.met}/{item.score.scored}
      </td>
      <td className="py-2 pr-2 font-mono">
        {item.rs_percentile != null ? item.rs_percentile.toFixed(0) : "–"}
      </td>
      <td className="py-2 pr-2 font-mono">{fmtPct(item.quarterly_eps_growth_pct)}</td>
      <td className="py-2 pr-2 font-mono">{fmtPct(item.annual_eps_growth_pct)}</td>
      <td className="py-2 pr-2 font-mono">
        {item.roe_pct != null ? `${item.roe_pct.toFixed(0)}%` : "–"}
      </td>
      <td className="py-2 pr-2 font-mono">{fmtPct(item.off_high_pct)}</td>
      <td className="py-2 pr-2 font-mono">
        {item.institutional_pct != null ? `${item.institutional_pct.toFixed(0)}%` : "–"}
      </td>
      <td className="py-2 pr-2">
        {item.breakout ? (
          <span className="inline-flex items-center whitespace-nowrap rounded-full border border-positive/40 bg-positive/10 px-1.5 py-0.5 text-[10px] text-positive">
            {item.breakout.volume_ratio}x vol · {item.breakout.date}
          </span>
        ) : (
          <span className="text-[11px] text-faint">–</span>
        )}
      </td>
      <td className="py-2" title={item.entry?.detail}>
        {item.entry ? (
          <span
            className={`inline-flex items-center whitespace-nowrap rounded-full border px-1.5 py-0.5 text-[10px] ${
              item.entry.status === "buyable"
                ? "border-positive/40 bg-positive/10 text-positive"
                : "border-warning/40 bg-warning/10 text-warning"
            }`}
          >
            {item.entry.extension_pct > 0 ? "+" : ""}
            {item.entry.extension_pct}% · stop {item.entry.stop_price}
          </span>
        ) : (
          <span className="text-[11px] text-faint">–</span>
        )}
      </td>
    </tr>
  );
}

export default function LeadersPage() {
  const [data, setData] = useState<LeadersResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    api
      .canslimLeaders(controller.signal)
      .then((response) => {
        setData(response);
        setSelected(
          (current) => current ?? response.items[0]?.symbol ?? null,
        );
      })
      .catch((e) => {
        if (controller.signal.aborted) return;
        setError(String(e.message ?? e));
      });
    return () => controller.abort();
  }, []);

  const selectedItem = data?.items.find((i) => i.symbol === selected) ?? null;
  const buyCount = data?.items.filter((i) => i.readiness === "buy_candidate").length ?? 0;
  const watchCount = data?.items.filter((i) => i.readiness === "near_pivot").length ?? 0;
  const extendedCount = data?.items.filter((i) => i.readiness === "extended").length ?? 0;

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <h1 className="text-lg font-semibold">Stock Leaders</h1>
          <p className="text-sm text-muted">
            CAN SLIM scan of the snapshot universe: earnings growth, breakouts,
            relative strength, and sponsorship - buy flags only when the market
            allows them.
          </p>
        </div>
        {data?.provisional && (
          <span className="rounded-full border border-warning/40 bg-warning/10 px-2 py-0.5 text-[11px] text-warning">
            Intraday - today&apos;s bar is provisional until the close
          </span>
        )}
      </div>

      {error && (
        <p className="rounded-md border border-border bg-surface px-4 py-3 text-sm text-muted">
          Backend unavailable: {error}
        </p>
      )}

      {!data && !error && (
        <div className="space-y-4">
          <div className="h-14 animate-pulse rounded-lg border border-border bg-surface" />
          <div className="h-72 animate-pulse rounded-lg border border-border bg-surface" />
        </div>
      )}

      {data && (
        <>
          <GateBanner data={data} />

          <Panel
            title="Scan results"
            right={
              <span className="text-[11px] text-muted">
                {buyCount} buy candidate(s) · {watchCount} near pivot ·{" "}
                {extendedCount} extended · as of {data.as_of ?? "n/a"}
              </span>
            }
          >
            {data.items.length === 0 ? (
              <p className="text-sm text-muted">
                No stocks could be scanned. The universe comes from the snapshot
                configuration and shared watchlists
                {data.unavailable.length > 0 &&
                  `; unavailable: ${data.unavailable.join(", ")}`}
                .
              </p>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full min-w-[980px] text-left text-xs">
                  <thead>
                    <tr className="border-b border-border text-[11px] uppercase tracking-wide text-muted">
                      <th className="py-2 pr-2">Stock</th>
                      <th className="py-2 pr-2">Readiness</th>
                      <th className="py-2 pr-2">Letters</th>
                      <th className="py-2 pr-2">Met</th>
                      <th className="py-2 pr-2" title="Relative strength percentile">RS</th>
                      <th className="py-2 pr-2" title="Quarterly EPS growth YoY">Qtr EPS</th>
                      <th className="py-2 pr-2" title="Annual EPS growth per year">Ann EPS</th>
                      <th className="py-2 pr-2">ROE</th>
                      <th className="py-2 pr-2" title="Distance from 52-week high">Off high</th>
                      <th className="py-2 pr-2" title="Institutional ownership">Inst</th>
                      <th className="py-2 pr-2">Breakout</th>
                      <th className="py-2" title="Distance from the pivot and the stop level">
                        Entry vs pivot
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.items.map((item) => (
                      <LeaderRow
                        key={item.symbol}
                        item={item}
                        selected={selected === item.symbol}
                        onSelect={setSelected}
                      />
                    ))}
                  </tbody>
                </table>
              </div>
            )}
            {data.unavailable.length > 0 && data.items.length > 0 && (
              <p className="mt-2 text-[11px] text-warning">
                No price history for: {data.unavailable.join(", ")}
              </p>
            )}
          </Panel>

          {selectedItem && (
            <Panel title={`${selectedItem.symbol} - full CAN SLIM read`}>
              <div className="space-y-4">
                <div className="space-y-1 text-sm text-foreground/90">
                  {selectedItem.narrative.map((line, i) => (
                    <p key={i}>{line}</p>
                  ))}
                </div>
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
                      {selectedItem.scorecard.map((row) => (
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
                {selectedItem.missing.length > 0 && (
                  <p className="text-[11px] text-muted">
                    Unavailable fundamental fields for this stock:{" "}
                    {selectedItem.missing.join(", ")}. Yahoo fundamental data is
                    best-effort; unavailable letters are never guessed.
                  </p>
                )}
              </div>
            </Panel>
          )}

          <p className="text-[11px] text-faint">
            {data.methodology} Excluded (no earnings): {data.excluded.join(", ")}.
            Risk discipline: O&apos;Neil cut every loss at {data.stop_loss_pct}%
            below purchase price, no exceptions. Research tooling, not
            investment advice.
          </p>
        </>
      )}
    </div>
  );
}
