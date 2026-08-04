"use client";

import { useEffect, useRef, useState } from "react";
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

// When a growth rate is undefined because the company lost money, the
// criterion still has a verdict - show that word rather than a blank dash.
function letterValue(
  item: StockLeader,
  letter: string,
  numeric: number | null | undefined,
): string {
  if (numeric != null) return fmtPct(numeric);
  const row = item.scorecard.find((r) => r.letter === letter);
  if (!row || row.status === "unavailable") return "–";
  return row.value.replace(" · ROE", "").split(" · ")[0] || "–";
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
          className="text-[11px] text-muted"
          title={
            item.sector_rank
              ? `${item.sector_label} ranks #${item.sector_rank} by relative strength`
              : undefined
          }
        >
          {item.sector_label ?? "—"}
          {item.sector_rank ? (
            <span className="ml-1 font-mono text-faint">#{item.sector_rank}</span>
          ) : null}
        </span>
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
      <td
        className="whitespace-nowrap py-2 pr-2 font-mono"
        title={
          item.fundamentals_fetched
            ? [
                item.eps_acceleration && item.eps_acceleration !== "unknown"
                  ? `EPS growth ${item.eps_acceleration}`
                  : null,
                item.sales_acceleration && item.sales_acceleration !== "unknown"
                  ? `sales growth ${item.sales_acceleration}`
                  : null,
              ]
                .filter(Boolean)
                .join(" · ") || undefined
            : "Technical-only row"
        }
      >
        {item.fundamentals_fetched
          ? letterValue(item, "C", item.quarterly_eps_growth_pct)
          : "·"}
        {item.fundamentals_fetched && item.eps_acceleration === "accelerating" && (
          <span className="ml-0.5 text-positive">▲</span>
        )}
        {item.fundamentals_fetched && item.eps_acceleration === "decelerating" && (
          <span className="ml-0.5 text-negative">▼</span>
        )}
      </td>
      <td className="whitespace-nowrap py-2 pr-2 font-mono">
        {item.fundamentals_fetched
          ? letterValue(item, "A", item.annual_eps_growth_pct)
          : "·"}
      </td>
      <td className="py-2 pr-2 font-mono">
        {!item.fundamentals_fetched
          ? "·"
          : item.roe_pct != null
            ? `${item.roe_pct.toFixed(0)}%`
            : "–"}
      </td>
      <td className="py-2 pr-2 font-mono">{fmtPct(item.off_high_pct)}</td>
      <td
        className="whitespace-nowrap py-2 pr-2 font-mono"
        title={
          item.sponsorship_trend === "accruing"
            ? "Ownership trend still accruing from our own daily observations"
            : item.sponsorship_trend
              ? `Ownership ${item.sponsorship_trend}`
              : undefined
        }
      >
        {!item.fundamentals_fetched
          ? "·"
          : item.institutional_pct != null
            ? `${item.institutional_pct.toFixed(0)}%`
            : "–"}
        {item.sponsorship_trend === "rising" && (
          <span className="ml-0.5 text-positive">↑</span>
        )}
        {item.sponsorship_trend === "falling" && (
          <span className="ml-0.5 text-negative">↓</span>
        )}
      </td>
      <td className="py-2 pr-2">
        {item.breakout ? (
          <span className="flex flex-col gap-0.5">
            <span className="inline-flex w-fit items-center whitespace-nowrap rounded-full border border-positive/40 bg-positive/10 px-1.5 py-0.5 text-[10px] text-positive">
              {item.breakout.volume_ratio}x vol · {item.breakout.date}
            </span>
            {item.base && (
              <span
                className={`text-[10px] ${
                  item.base.quality === "proper"
                    ? "text-muted"
                    : item.base.quality === "acceptable"
                      ? "text-muted"
                      : "text-warning"
                }`}
                title={`Base: ${item.base.sessions} sessions, ${item.base.depth_pct}% deep (${item.base.quality})`}
              >
                base {item.base.weeks}w / {item.base.depth_pct}%
                {item.base.quality === "short" || item.base.quality === "deep"
                  ? ` (${item.base.quality})`
                  : ""}
              </span>
            )}
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
  const [detail, setDetail] = useState<StockLeader | null>(null);
  const [detailError, setDetailError] = useState<string | null>(null);
  const [sector, setSector] = useState<string>("all");
  const [buyOnly, setBuyOnly] = useState(false);
  const detailRequest = useRef(0);

  const selectSymbol = (symbol: string) => {
    if (symbol === selected) return;
    setDetail(null);
    setDetailError(null);
    setSelected(symbol);
  };

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

  // Rows outside the fundamentals shortlist arrive technical-only. Opening
  // one fetches its financials on demand (and warms the shared 24h cache),
  // which is what the table's "open one to pull its financials" promises.
  const rowItem = data?.items.find((i) => i.symbol === selected) ?? null;
  const selectedItem = detail?.symbol === selected ? detail : rowItem;
  const detailPending = Boolean(
    selected && rowItem && !rowItem.fundamentals_fetched && !detail && !detailError,
  );

  useEffect(() => {
    if (!selected) return;
    const row = data?.items.find((i) => i.symbol === selected);
    if (!row || row.fundamentals_fetched) return;
    const requestId = ++detailRequest.current;
    const controller = new AbortController();
    api
      .canslimDetail(selected, controller.signal)
      .then((response) => {
        if (detailRequest.current !== requestId) return;
        setDetail(response);
      })
      .catch((e) => {
        if (controller.signal.aborted || detailRequest.current !== requestId) return;
        setDetailError(String(e.message ?? e));
      });
    return () => controller.abort();
  }, [selected, data]);
  const buyCount = data?.items.filter((i) => i.readiness === "buy_candidate").length ?? 0;
  const watchCount = data?.items.filter((i) => i.readiness === "near_pivot").length ?? 0;
  const extendedCount = data?.items.filter((i) => i.readiness === "extended").length ?? 0;

  const actionable = new Set(["buy_candidate", "extended", "near_pivot"]);
  const visibleItems = (data?.items ?? []).filter((item) => {
    if (sector !== "all" && (item.sector_symbol ?? "watchlist") !== sector) return false;
    if (buyOnly && !actionable.has(item.readiness)) return false;
    return true;
  });

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <h1 className="text-lg font-semibold">Stock Leaders</h1>
          <p className="text-sm text-muted">
            CAN SLIM scan across the holdings of every tracked market and
            sector ETF: earnings growth, breakouts, relative strength, and
            sponsorship - buy flags only when the market allows them.
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

          <div className="flex flex-wrap items-center gap-1.5">
            <span className="mr-1 text-[11px] uppercase tracking-wide text-muted">
              Group
            </span>
            <button
              type="button"
              onClick={() => setSector("all")}
              aria-pressed={sector === "all"}
              className={`rounded-full border px-2 py-0.5 text-[11px] transition-colors ${
                sector === "all"
                  ? "border-accent bg-accent/10 text-accent-strong"
                  : "border-border bg-surface text-muted hover:bg-surface-hover"
              }`}
            >
              All ({data.items.length})
            </button>
            {data.sectors.map((s) => {
              const key = s.symbol ?? "watchlist";
              const count = data.items.filter(
                (i) => (i.sector_symbol ?? "watchlist") === key,
              ).length;
              if (!count) return null;
              return (
                <button
                  key={key}
                  type="button"
                  onClick={() => setSector(key)}
                  aria-pressed={sector === key}
                  className={`rounded-full border px-2 py-0.5 text-[11px] transition-colors ${
                    sector === key
                      ? "border-accent bg-accent/10 text-accent-strong"
                      : "border-border bg-surface text-muted hover:bg-surface-hover"
                  }`}
                >
                  {s.label} ({count})
                </button>
              );
            })}
            <button
              type="button"
              onClick={() => setBuyOnly((v) => !v)}
              aria-pressed={buyOnly}
              className={`ml-2 rounded-full border px-2 py-0.5 text-[11px] transition-colors ${
                buyOnly
                  ? "border-positive/50 bg-positive/10 text-positive"
                  : "border-border bg-surface text-muted hover:bg-surface-hover"
              }`}
            >
              Actionable only
            </button>
          </div>

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
                No stocks could be scanned. The universe is built from the
                holdings of the tracked market and sector ETFs
                {data.unavailable.length > 0 &&
                  `; unavailable: ${data.unavailable.join(", ")}`}
                .
              </p>
            ) : visibleItems.length === 0 ? (
              <p className="text-sm text-muted">
                No stocks match these filters. Clear the group or actionable
                filter to see the full scan.
              </p>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full min-w-[980px] text-left text-xs">
                  <thead>
                    <tr className="border-b border-border text-[11px] uppercase tracking-wide text-muted">
                      <th className="py-2 pr-2">Stock</th>
                      <th className="py-2 pr-2" title="Strongest ETF holding this stock">
                        Group
                      </th>
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
                    {visibleItems.map((item) => (
                      <LeaderRow
                        key={item.symbol}
                        item={item}
                        selected={selected === item.symbol}
                        onSelect={selectSymbol}
                      />
                    ))}
                  </tbody>
                </table>
              </div>
            )}
            <p className="mt-3 text-[11px] text-muted">
              Scanned {data.scanned} of {data.universe_size} candidates drawn
              from the tracked ETFs
              {data.holdings_source === "configured" &&
                " (holdings provider unreachable - using the configured constituent lists, which can drift)"}
              {data.holdings_source === "mixed" &&
                " (some ETFs used configured constituent lists because live holdings were unreachable)"}
              . Fundamentals are loaded for {data.fundamentals_scanned} of them;
              rows marked <span className="font-mono">·</span> are
              technical-only until the background warm-up finishes - opening one
              fetches its financials immediately.
              {data.dropped_for_capacity > 0 &&
                ` ${data.dropped_for_capacity} further candidate(s) exceeded the scan cap.`}
            </p>
            {data.unavailable.length > 0 && (
              <p className="mt-1 text-[11px] text-warning">
                No price history for: {data.unavailable.slice(0, 12).join(", ")}
                {data.unavailable.length > 12 &&
                  ` +${data.unavailable.length - 12} more`}
              </p>
            )}
          </Panel>

          {selectedItem && (
            <Panel
              title={`${selectedItem.symbol} - full CAN SLIM read`}
              right={
                detailPending ? (
                  <span className="text-[11px] text-muted">
                    Fetching financials…
                  </span>
                ) : detailError ? (
                  <span className="text-[11px] text-warning">
                    Financials unavailable: {detailError}
                  </span>
                ) : undefined
              }
            >
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
