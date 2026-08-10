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
          <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
            <div className="rounded-md border border-border bg-surface-2 p-3">
              <div className="text-[11px] uppercase tracking-wide text-muted">Hold rate</div>
              <div className="font-mono text-lg">{pct(summary.hold_rate)}</div>
              <div className="text-[11px] text-muted">
                {summary.held} of {summary.decided} decided
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
              <div className="text-[11px] text-muted">{summary.pending} pending</div>
            </div>
          </div>

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
                    label="Invalidation"
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
                          ? "follow-through"
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
                              : "border-border bg-surface-2 text-muted"
                        }`}
                      >
                        {row.outcome === "held"
                          ? "Held"
                          : row.outcome === "failed"
                            ? `Failed (${row.sessions_to_fail}s)`
                            : `Pending (${row.evaluated_sessions}s in)`}
                      </span>
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

          <p className="text-[11px] text-muted">
            A follow-through fails when the index closes back below its
            rally-attempt low; a breakout fails when the stock closes below its
            stop. The signal day never counts and signals without a full
            horizon stay pending, excluded from the rates. O&apos;Neil expected
            a share of follow-throughs to fail - that cost is the price of
            being early enough to catch the real move.
          </p>
        </div>
      )}
    </Panel>
  );
}
