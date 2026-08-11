"use client";

import { EconomicRelease, EconomicReleaseSeries } from "@/lib/api";

// Economic releases as readable rows rather than a date-and-name list. Each row
// carries the headline FRED series behind that release: what printed, what it
// printed against, and how it moved.
//
// FRED publishes no consensus figure and the configured FMP tier does not
// expose its economic calendar, so there is no beat/miss against expectations
// anywhere in here. Every comparison is against the prior period.

function formatValue(value: number | null | undefined, units: string): string {
  if (value == null) return "—";
  if (units === "%" || units === "% annualized") return `${value.toFixed(2)}%`;
  if (units === "index") return value.toFixed(3);
  if (Math.abs(value) >= 1000) return Math.round(value).toLocaleString();
  return value.toLocaleString(undefined, { maximumFractionDigits: 2 });
}

function formatChange(value: number | null | undefined, units: string): string {
  if (value == null) return "—";
  const sign = value > 0 ? "+" : value < 0 ? "-" : "";
  const magnitude = formatValue(Math.abs(value), units);
  if (units === "%" || units === "% annualized") {
    return `${sign}${Math.abs(value).toFixed(2)}pp`;
  }
  return `${sign}${magnitude}`;
}

function formatPeriod(period: string | null | undefined): string {
  if (!period) return "";
  const parsed = new Date(`${period}T00:00:00`);
  if (Number.isNaN(parsed.getTime())) return period;
  return parsed.toLocaleDateString(undefined, { month: "short", year: "numeric" });
}

// Colour only says which way the move usually reads, never whether it beat an
// expectation. Series with no conventional good direction stay neutral.
function moveTone(series: EconomicReleaseSeries): string {
  const change = series.change;
  if (change == null || change === 0 || !series.favorable) return "text-muted";
  const good = series.favorable === "up" ? change > 0 : change < 0;
  return good ? "text-positive" : "text-negative";
}

function SeriesRow({ series }: { series: EconomicReleaseSeries }) {
  const change = series.change;
  const arrow = change == null || change === 0 ? "" : change > 0 ? "▲" : "▼";
  return (
    <div className="grid grid-cols-[minmax(0,1fr)_auto] items-baseline gap-x-3 gap-y-0.5 py-1 sm:grid-cols-[minmax(0,11rem)_auto_minmax(0,1fr)]">
      <span className="truncate text-xs text-muted" title={series.series_id}>
        {series.label}
      </span>
      <span className="whitespace-nowrap font-mono text-sm text-foreground">
        {formatValue(series.actual, series.units)}
        <span className="ml-1 text-[11px] text-faint">{series.units}</span>
      </span>
      <span className="col-span-2 flex flex-wrap items-baseline gap-x-2 text-xs sm:col-span-1">
        <span className={`font-mono ${moveTone(series)}`}>
          {arrow} {formatChange(series.change, series.units)}
          {series.change_pct != null && (
            <span className="ml-1 text-faint">
              ({series.change_pct > 0 ? "+" : ""}
              {series.change_pct.toFixed(2)}%)
            </span>
          )}
        </span>
        <span className="text-faint">
          vs prior {formatValue(series.prior, series.units)}
        </span>
        {series.change_yoy_pct != null && (
          <span className="text-faint">
            · {series.change_yoy_pct > 0 ? "+" : ""}
            {series.change_yoy_pct.toFixed(2)}% y/y
          </span>
        )}
        {series.period && (
          <span className="text-faint">· {formatPeriod(series.period)}</span>
        )}
      </span>
    </div>
  );
}

export default function EconomicReleaseList({
  releases,
  loading,
  emptyMessage,
  relativeLabel,
}: {
  releases: EconomicRelease[];
  loading: boolean;
  emptyMessage: string;
  relativeLabel: (isoDate: string) => string;
}) {
  if (loading) {
    return (
      <p role="status" className="px-4 py-10 text-center text-sm text-muted">
        Loading economic releases...
      </p>
    );
  }
  if (!releases.length) {
    return <p className="px-4 py-10 text-center text-sm text-muted">{emptyMessage}</p>;
  }

  return (
    <div className="divide-y divide-border">
      {releases.map((release) => {
        // A release we could not pin a published value to still shows the most
        // recent reading, but it must not be read as this release's print.
        const fallbackOnly =
          release.series.length > 0 && release.series.every((item) => !item.matched);
        return (
          <div key={`${release.release_id}-${release.release_date}`} className="px-4 py-3">
            <div className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
              <a
                href={release.url}
                target="_blank"
                rel="noreferrer"
                className="text-sm text-foreground hover:text-accent"
              >
                {release.release_name}
              </a>
              <span className="whitespace-nowrap font-mono text-xs text-faint">
                {release.release_date} · {relativeLabel(release.release_date)}
              </span>
            </div>

            {release.series.length > 0 ? (
              <div className="mt-1.5 border-t border-border pt-1">
                {release.series.map((series) => (
                  <SeriesRow key={series.series_id} series={series} />
                ))}
                {fallbackOnly && (
                  <p className="mt-1 text-[11px] text-faint">
                    {release.status === "scheduled"
                      ? "Last published reading, shown for context. This release has not printed yet."
                      : "FRED has no vintage for this date (revision or delayed publication); showing the latest reading."}
                  </p>
                )}
              </div>
            ) : (
              <p className="mt-1 text-xs text-faint">
                No headline series mapped for this release.
              </p>
            )}
          </div>
        );
      })}
    </div>
  );
}
