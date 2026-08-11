"use client";

import { useState } from "react";
import { EarningsEvent } from "@/lib/api";

// The market-wide Nasdaq calendar runs to thousands of rows per fortnight, and
// the page now renders an upcoming and a past table at once. Cap what reaches
// the DOM and let the reader ask for the rest.
const ROW_CAP = 150;

function formatNumber(value?: number | null): string {
  return value == null ? "—" : value.toFixed(2);
}

function formatEarningsDate(value: string, session?: string | null): string {
  if (value.length === 10) {
    const [year, month, day] = value.split("-");
    const date = new Date(Number(year), Number(month) - 1, Number(day));
    return `${date.toLocaleDateString([], { month: "short", day: "numeric" })} · ${
      session || "time TBD"
    }`;
  }
  return new Date(value).toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

export default function EarningsTable({
  events,
  loading,
  emptyMessage,
}: {
  events: EarningsEvent[];
  loading: boolean;
  emptyMessage: string;
}) {
  const [showAll, setShowAll] = useState(false);
  const visible = showAll ? events : events.slice(0, ROW_CAP);
  const hidden = events.length - visible.length;

  return (
    <>
    <div className="max-h-[30rem] overflow-auto">
      <table className="w-full min-w-[560px] text-sm">
        <thead className="sticky top-0 bg-surface-2 text-left text-xs text-muted">
          <tr>
            <th className="px-3 py-2">Date</th>
            <th className="px-3 py-2">Symbol</th>
            <th className="px-3 py-2 text-right">EPS est.</th>
            <th className="px-3 py-2 text-right">Reported</th>
            <th className="px-3 py-2 text-right">Surprise</th>
          </tr>
        </thead>
        <tbody>
          {loading && (
            <tr>
              <td colSpan={5} className="px-4 py-10 text-center text-muted">
                Loading market-wide earnings...
              </td>
            </tr>
          )}
          {!loading &&
            visible.map((event) => (
              <tr
                key={`${event.symbol}-${event.earnings_at}`}
                className="border-t border-border hover:bg-surface-hover"
              >
                <td className="whitespace-nowrap px-3 py-2 font-mono text-muted">
                  {formatEarningsDate(event.earnings_at, event.session)}
                </td>
                <td className="px-3 py-2">
                  <a
                    href={event.url}
                    target="_blank"
                    rel="noreferrer"
                    className="font-mono text-accent"
                  >
                    {event.symbol}
                  </a>
                  {event.company_name && (
                    <span
                      className="block max-w-56 truncate text-xs text-faint"
                      title={event.company_name}
                    >
                      {event.company_name}
                    </span>
                  )}
                </td>
                <td className="px-3 py-2 text-right font-mono text-muted">
                  {formatNumber(event.eps_estimate)}
                </td>
                <td className="px-3 py-2 text-right font-mono text-muted">
                  {formatNumber(event.reported_eps)}
                </td>
                <td
                  className={`px-3 py-2 text-right font-mono ${
                    event.surprise_pct == null
                      ? "text-muted"
                      : event.surprise_pct >= 0
                        ? "text-positive"
                        : "text-negative"
                  }`}
                >
                  {event.surprise_pct == null
                    ? "—"
                    : `${event.surprise_pct > 0 ? "+" : ""}${event.surprise_pct.toFixed(1)}%`}
                </td>
              </tr>
            ))}
          {!loading && !events.length && (
            <tr>
              <td colSpan={5} className="px-4 py-10 text-center text-muted">
                {emptyMessage}
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
    {!loading && hidden > 0 && (
      <button
        type="button"
        onClick={() => setShowAll(true)}
        className="w-full border-t border-border px-3 py-2 text-xs text-muted hover:text-foreground"
      >
        Showing {visible.length} of {events.length} · show all
      </button>
    )}
    </>
  );
}
