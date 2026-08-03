"use client";

import { CachedFlowResponse, CachedFlowRow } from "@/lib/api";

// Single-ticker rendering of the cached proxy feed (/api/flow/feed?ticker=X),
// used by the ticker page's Flow tab. Columns and formatting mirror the
// global /flow page minus the redundant Symbol column.
const COLUMNS: { key: keyof CachedFlowRow; label: string; digits?: number }[] = [
  { key: "option_type", label: "Side" },
  { key: "expiration_date", label: "Expiry" },
  { key: "strike", label: "Strike", digits: 2 },
  { key: "volume", label: "Volume", digits: 0 },
  { key: "open_interest", label: "OI", digits: 0 },
  { key: "volume_oi", label: "Vol/OI", digits: 2 },
  { key: "oi_change", label: "OI chg", digits: 0 },
  { key: "mid_iv", label: "Mid IV", digits: 1 },
  { key: "iv_change", label: "IV chg", digits: 1 },
  { key: "score", label: "Score", digits: 1 },
];

function fmt(value: number | null | undefined, digits = 0, percent = false) {
  if (value == null) return "-";
  return `${value.toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  })}${percent ? "%" : ""}`;
}

function formatCell(
  key: string,
  value: string | number | boolean | null | undefined,
  digits?: number,
) {
  if (typeof value === "boolean") return value ? "Ready" : "Unavailable";
  if (typeof value === "string")
    return key === "option_type" ? value.toUpperCase() : value;
  return fmt(value, digits, key === "mid_iv" || key === "iv_change");
}

export default function ProxyFeedTable({ data }: { data: CachedFlowResponse }) {
  const history = data.unavailable_history_tickers.join(", ");
  return (
    <>
      <div className="flex flex-wrap gap-2 border-b border-border px-4 py-3 text-xs">
        <span className="rounded border border-border bg-surface px-2 py-1 font-mono text-muted">
          Snapshot: {data.as_of ?? "unavailable"}
        </span>
        <span
          className={`rounded border px-2 py-1 ${
            data.stale
              ? "border-amber-500/50 text-amber-300"
              : "border-emerald-500/50 text-emerald-300"
          }`}
        >
          {data.stale ? "Stale or incomplete cache" : "Current cache"}
        </span>
        {data.unavailable_history && (
          <span className="rounded border border-amber-500/50 px-2 py-1 text-amber-300">
            History unavailable{history ? `: ${history}` : ""}
          </span>
        )}
      </div>
      <div className="max-h-[34rem] overflow-auto">
        <table className="w-full min-w-[760px] text-sm">
          <thead className="sticky top-0 bg-surface-2 text-left text-xs text-muted">
            <tr>
              {COLUMNS.map((column) => (
                <th
                  key={column.key}
                  className="whitespace-nowrap px-3 py-2 text-right font-medium first:text-left"
                >
                  {column.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.rows.map((row) => (
              <tr
                key={`${row.expiration_date}-${row.strike}-${row.option_type}`}
                className="border-t border-border hover:bg-surface-hover"
              >
                {COLUMNS.map((column) => (
                  <td
                    key={column.key}
                    className="whitespace-nowrap px-3 py-1.5 text-right font-mono text-muted first:text-left"
                  >
                    {formatCell(column.key, row[column.key], column.digits)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}
