"use client";

import { useEffect, useState } from "react";
import { api, DirectionOutcomeRow, DirectionOutcomesResponse } from "@/lib/api";
import Panel from "@/components/ui/Panel";

// Realized outcomes for O'Neil signals. The honest answer to "how often
// does this whipsaw me": hold rate over decided signals, the typical
// drawdown you had to sit through, and how often a mechanical stop fired.

function pct(value: number | null | undefined, digits = 0): string {
  return value == null ? "—" : `${(value * 100).toFixed(digits)}%`;
}

function signed(value: number | null | undefined): string {
  if (value == null) return "—";
  return `${value > 0 ? "+" : ""}${value.toFixed(1)}%`;
}

type OutcomeSortKey =
  | "signal"
  | "date"
  | "entry"
  | "invalidation"
  | "outcome"
  | "gain"
  | "drawdown"
  | "stop";

const OUTCOME_ORDER: Record<DirectionOutcomeRow["outcome"], number> = {
  held: 0,
  failed: 1,
  pending: 2,
};

function outcomeSortValue(
  row: DirectionOutcomeRow,
  key: OutcomeSortKey,
): string | number | null {
  if (key === "signal") return `${row.symbol} ${row.signal_type}`;
  if (key === "date") return row.signal_date;
  if (key === "entry") return row.entry_price;
  if (key === "invalidation") return row.invalidation_level;
  if (key === "outcome") return OUTCOME_ORDER[row.outcome];
  if (key === "gain") return row.max_gain_pct ?? null;
  if (key === "drawdown") return row.max_drawdown_pct ?? null;
  return row.stop_hit == null ? null : Number(row.stop_hit);
}

function compareValues(
  left: string | number | null,
  right: string | number | null,
  ascending: boolean,
) {
  if (left == null) return right == null ? 0 : 1;
  if (right == null) return -1;
  const compared =
    typeof left === "number" && typeof right === "number"
      ? left - right
      : String(left).localeCompare(String(right), undefined, {
          numeric: true,
          sensitivity: "base",
        });
  return compared * (ascending ? 1 : -1);
}

function SortHeader({
  sortKey,
  active,
  ascending,
  label,
  onSort,
}: {
  sortKey: OutcomeSortKey;
  active: OutcomeSortKey;
  ascending: boolean;
  label: string;
  onSort: (key: OutcomeSortKey) => void;
}) {
  const selected = sortKey === active;
  return (
    <th
      className="py-1 pr-2"
      aria-sort={selected ? (ascending ? "ascending" : "descending") : "none"}
    >
      <button
        type="button"
        onClick={() => onSort(sortKey)}
        className="min-h-8 whitespace-nowrap text-left font-medium hover:text-foreground"
      >
        {label} {selected ? (ascending ? "▲" : "▼") : ""}
      </button>
    </th>
  );
}

export default function OutcomesPanel() {
  const [data, setData] = useState<DirectionOutcomesResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [sortKey, setSortKey] = useState<OutcomeSortKey>("date");
  const [sortAscending, setSortAscending] = useState(false);

  useEffect(() => {
    const controller = new AbortController();
    api
      .directionOutcomes({ limit: 200 }, controller.signal)
      .then(setData)
      .catch((e) => {
        if (controller.signal.aborted) return;
        setError(String(e.message ?? e));
      });
    return () => controller.abort();
  }, []);

  const summary = data?.summary;
  const sortedRows = (data?.rows ?? [])
    .map((row, originalIndex) => ({ row, originalIndex }))
    .sort((left, right) => {
      const compared = compareValues(
        outcomeSortValue(left.row, sortKey),
        outcomeSortValue(right.row, sortKey),
        sortAscending,
      );
      return (
        compared ||
        left.row.symbol.localeCompare(right.row.symbol) ||
        left.originalIndex - right.originalIndex
      );
    })
    .map(({ row }) => row);

  function toggleSort(key: OutcomeSortKey) {
    if (key === sortKey) setSortAscending((value) => !value);
    else {
      setSortKey(key);
      setSortAscending(true);
    }
  }

  return (
    <Panel
      title="O'Neil signal outcomes"
      right={
        data ? (
          <span className="text-[11px] text-muted">
            {data.horizon_sessions}-session horizon · {data.stop_pct}% stop test
          </span>
        ) : undefined
      }
    >
      {error && <p className="text-sm text-muted">Outcomes unavailable: {error}</p>}

      {!data && !error && (
        <div className="h-28 animate-pulse rounded-md border border-border bg-surface-2" />
      )}

      {data && summary && summary.signals === 0 && (
        <p className="text-sm text-muted">
          No follow-through or breakout signals have been logged yet. Once they
          accrue, this panel reports how many held, the drawdown you had to sit
          through, and how often a {data.stop_pct}% stop would have fired -
          measured, not assumed.
        </p>
      )}

      {data && summary && summary.signals > 0 && (
        <div className="space-y-4">
          <p className="max-w-3xl text-sm text-muted">
            These are the app&apos;s market-direction calls: a{" "}
            <span className="text-foreground">market-turn</span> call says a
            downtrend has flipped up, and a{" "}
            <span className="text-foreground">breakout</span> call says a stock
            cleared its base. Each is graded against the price that would prove
            it wrong. A call needs {data.horizon_sessions} sessions (about{" "}
            {Math.round(data.horizon_sessions / 5)} weeks) to be{" "}
            <span className="text-positive">confirmed</span>; until then it shows
            live as working or at risk.
          </p>

          {summary.decided === 0 ? (
            <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
              <div className="rounded-md border border-border bg-surface-2 p-3">
                <div className="text-[11px] uppercase tracking-wide text-muted">Working</div>
                <div className="font-mono text-lg text-positive">{summary.working ?? 0}</div>
                <div className="text-[11px] text-muted">still above their line</div>
              </div>
              <div className="rounded-md border border-border bg-surface-2 p-3">
                <div className="text-[11px] uppercase tracking-wide text-muted">At risk</div>
                <div className="font-mono text-lg">{summary.at_risk ?? 0}</div>
                <div className="text-[11px] text-muted">cushion under 2%</div>
              </div>
              <div className="rounded-md border border-border bg-surface-2 p-3">
                <div className="text-[11px] uppercase tracking-wide text-muted">
                  Avg move so far
                </div>
                <div className="font-mono text-lg">{signed(summary.avg_pending_gain_pct)}</div>
                <div className="text-[11px] text-muted">vs entry, latest close</div>
              </div>
              <div className="rounded-md border border-border bg-surface-2 p-3">
                <div className="text-[11px] uppercase tracking-wide text-muted">
                  First confirmation
                </div>
                <div className="font-mono text-lg">
                  {summary.next_confirmation_sessions != null
                    ? `${summary.next_confirmation_sessions}s`
                    : "—"}
                </div>
                <div className="text-[11px] text-muted">sessions away</div>
              </div>
            </div>
          ) : (
            <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
              <div className="rounded-md border border-border bg-surface-2 p-3">
                <div className="text-[11px] uppercase tracking-wide text-muted">Hold rate</div>
                <div className="font-mono text-lg">{pct(summary.hold_rate)}</div>
                <div className="text-[11px] text-muted">
                  {summary.held} of {summary.decided} confirmed
                </div>
              </div>
              <div className="rounded-md border border-border bg-surface-2 p-3">
                <div className="text-[11px] uppercase tracking-wide text-muted">
                  Avg max drawdown
                </div>
                <div className="font-mono text-lg">{signed(summary.avg_max_drawdown_pct)}</div>
                <div className="text-[11px] text-muted">worst dip after entry</div>
              </div>
              <div className="rounded-md border border-border bg-surface-2 p-3">
                <div className="text-[11px] uppercase tracking-wide text-muted">Avg max gain</div>
                <div className="font-mono text-lg">{signed(summary.avg_max_gain_pct)}</div>
                <div className="text-[11px] text-muted">best move after entry</div>
              </div>
              <div className="rounded-md border border-border bg-surface-2 p-3">
                <div className="text-[11px] uppercase tracking-wide text-muted">
                  {data.stop_pct}% stop fired
                </div>
                <div className="font-mono text-lg">{pct(summary.stop_hit_rate)}</div>
                <div className="text-[11px] text-muted">
                  {summary.working ?? summary.pending} still working
                </div>
              </div>
            </div>
          )}

          <div className="overflow-x-auto">
            <table className="w-full min-w-[720px] text-left text-xs">
              <thead>
                <tr className="border-b border-border text-[11px] uppercase tracking-wide text-muted">
                  <SortHeader
                    sortKey="signal"
                    active={sortKey}
                    ascending={sortAscending}
                    label="Signal"
                    onSort={toggleSort}
                  />
                  <SortHeader
                    sortKey="date"
                    active={sortKey}
                    ascending={sortAscending}
                    label="Date"
                    onSort={toggleSort}
                  />
                  <SortHeader
                    sortKey="entry"
                    active={sortKey}
                    ascending={sortAscending}
                    label="Entry"
                    onSort={toggleSort}
                  />
                  <SortHeader
                    sortKey="invalidation"
                    active={sortKey}
                    ascending={sortAscending}
                    label="Fails below"
                    onSort={toggleSort}
                  />
                  <SortHeader
                    sortKey="outcome"
                    active={sortKey}
                    ascending={sortAscending}
                    label="Outcome"
                    onSort={toggleSort}
                  />
                  <SortHeader
                    sortKey="gain"
                    active={sortKey}
                    ascending={sortAscending}
                    label="Max gain"
                    onSort={toggleSort}
                  />
                  <SortHeader
                    sortKey="drawdown"
                    active={sortKey}
                    ascending={sortAscending}
                    label="Max drawdown"
                    onSort={toggleSort}
                  />
                  <SortHeader
                    sortKey="stop"
                    active={sortKey}
                    ascending={sortAscending}
                    label="Stop"
                    onSort={toggleSort}
                  />
                </tr>
              </thead>
              <tbody>
                {sortedRows.map((row) => (
                  <tr key={row.id} className="border-b border-border/60">
                    <td className="py-2 pr-2">
                      <span className="font-mono font-semibold">{row.symbol}</span>{" "}
                      <span className="text-muted">
                        {row.signal_type === "follow_through_day"
                          ? "market turned up"
                          : "breakout"}
                      </span>
                    </td>
                    <td className="whitespace-nowrap py-2 pr-2 font-mono">{row.signal_date}</td>
                    <td className="py-2 pr-2 font-mono">{row.entry_price}</td>
                    <td className="py-2 pr-2 font-mono">{row.invalidation_level}</td>
                    <td className="py-2 pr-2">
                      <span
                        className={`inline-flex items-center rounded-full border px-1.5 py-0.5 text-[10px] ${
                          row.outcome === "held"
                            ? "border-positive/40 bg-positive/10 text-positive"
                            : row.outcome === "failed"
                              ? "border-negative/40 bg-negative/10 text-negative"
                              : row.status === "at_risk"
                                ? "border-warning/40 bg-warning/10 text-warning"
                                : "border-border bg-surface-2 text-muted"
                        }`}
                      >
                        {row.outcome === "held"
                          ? "Confirmed"
                          : row.outcome === "failed"
                            ? `Failed (${row.sessions_to_fail}s)`
                            : row.status === "at_risk"
                              ? "At risk"
                              : "Working"}
                      </span>
                      {row.outcome === "pending" && (
                        <div className="mt-0.5 text-[10px] text-muted">
                          {signed(row.latest_gain_pct)} · {row.evaluated_sessions}/
                          {data.horizon_sessions}s
                        </div>
                      )}
                    </td>
                    <td className="py-2 pr-2 font-mono text-positive">
                      {signed(row.max_gain_pct)}
                    </td>
                    <td className="py-2 pr-2 font-mono text-negative">
                      {signed(row.max_drawdown_pct)}
                    </td>
                    <td className="py-2 text-[11px]">
                      {row.stop_hit == null ? "—" : row.stop_hit ? "fired" : "held"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {!sortedRows.some((r) => r.signal_type === "stock_breakout") && (
            <p className="rounded-md border border-border bg-surface-2 px-3 py-2 text-[11px] text-muted">
              No stock breakouts here yet. O&apos;Neil made no new buys during
              corrections, so breakout calls are only logged when the market is
              in a confirmed uptrend. The calls above are index market-turn
              signals; breakouts will appear as uptrends produce them.
            </p>
          )}

          <p className="text-[11px] text-muted">
            <span className="text-foreground">Working</span> means the call is
            still above its &quot;fails below&quot; price;{" "}
            <span className="text-warning">at risk</span> means its cushion is
            under 2%; <span className="text-positive">confirmed</span> means it
            survived the full {data.horizon_sessions}-session window;{" "}
            <span className="text-negative">failed</span> means it closed below
            that price first. The day a call fires never counts, and unresolved
            calls stay out of the hold rate. Some market-turn calls are expected
            to fail - that is the cost of being early enough to catch the real
            move.
          </p>
        </div>
      )}
    </Panel>
  );
}
