"use client";

import { useEffect, useState } from "react";
import { api, DataCoverage } from "@/lib/api";

// Honest freshness read sourced from logged scans: the latest session we hold
// data for, how many sessions are logged, and how many tickers the newest
// session covered. Makes snapshot gaps visible instead of silent.

function formatDate(value: string): string {
  const parsed = new Date(`${value}T00:00:00`);
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}

export default function DataFreshness({ className = "" }: { className?: string }) {
  const [data, setData] = useState<DataCoverage | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    api
      .dataCoverage(controller.signal)
      .then(setData)
      .catch(() => {
        /* freshness is non-critical; stay silent on failure */
      });
    return () => controller.abort();
  }, []);

  if (!data || !data.as_of) return null;

  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded border border-border bg-surface-2 px-2 py-1 text-[11px] text-muted ${className}`}
      title="The most recent session we have logged data for. Gaps mean the snapshot job missed sessions."
    >
      <span className="h-1.5 w-1.5 rounded-full bg-positive" aria-hidden />
      Data through {formatDate(data.as_of)} · {data.sessions} session
      {data.sessions === 1 ? "" : "s"}
      {data.latest_ticker_count ? ` · ${data.latest_ticker_count} tickers` : ""}
    </span>
  );
}
