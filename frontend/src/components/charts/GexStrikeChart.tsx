"use client";

import { GexPoint } from "@/lib/api";
import { useMemo, useState } from "react";
import StrikeBarChart from "./StrikeBarChart";

// Single expirations use the established polarity chart. Comparisons use
// grouped diverging bars on one shared scale so expiration profiles remain
// directly comparable without hiding sign or magnitude.

interface Props {
  series: Array<{ expiration: string; profile: GexPoint[] }>;
  spot: number;
}

const SERIES_COLORS = [
  "#818cf8",
  "#22c55e",
  "#f59e0b",
  "#ec4899",
  "#06b6d4",
  "#a78bfa",
  "#f97316",
  "#94a3b8",
];

function shortDate(expiration: string) {
  const [, month, day] = expiration.split("-");
  return month && day ? `${month}/${day}` : expiration;
}

function MultiExpirationGexChart({ series, spot }: Props) {
  const [hover, setHover] = useState<string | null>(null);
  const { rows, maxAbs, spotStrike } = useMemo(() => {
    const byStrike = new Map<number, Map<string, number>>();
    for (const item of series) {
      for (const point of item.profile) {
        const values = byStrike.get(point.strike) ?? new Map<string, number>();
        values.set(item.expiration, point.net_gex);
        byStrike.set(point.strike, values);
      }
    }
    const rows = [...byStrike.entries()]
      .map(([strike, values]) => ({ strike, values }))
      .sort((a, b) => b.strike - a.strike);
    const maxAbs = Math.max(
      1e-12,
      ...rows.flatMap((row) => [...row.values.values()].map(Math.abs)),
    );
    let spotStrike = rows[0]?.strike ?? spot;
    let distance = Infinity;
    for (const row of rows) {
      const nextDistance = Math.abs(row.strike - spot);
      if (nextDistance < distance) {
        distance = nextDistance;
        spotStrike = row.strike;
      }
    }
    return { rows, maxAbs, spotStrike };
  }, [series, spot]);

  if (!rows.length) return null;

  return (
    <div>
      <div className="mb-3 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-muted">
        {series.map((item, index) => (
          <span key={item.expiration} className="flex items-center gap-1.5">
            <span
              className="inline-block h-2 w-3 rounded-sm"
              style={{ backgroundColor: SERIES_COLORS[index % SERIES_COLORS.length] }}
            />
            {item.expiration}
          </span>
        ))}
        <span className="ml-auto flex items-center gap-1.5">
          <span className="inline-block h-3 w-0.5 bg-accent" />
          spot {spot.toFixed(2)}
        </span>
      </div>

      <div className="max-h-[32rem] overflow-y-auto pr-1">
        {rows.map((row) => {
          const isSpot = row.strike === spotStrike;
          const entries = series
            .map((item, index) => ({
              item,
              index,
              value: row.values.get(item.expiration),
            }))
            .filter(
              (entry): entry is typeof entry & { value: number } =>
                entry.value != null,
            )
            .sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
          const activeEntry = entries.find(
            (entry) => hover === `${row.strike}-${entry.item.expiration}`,
          );
          return (
            <div
              key={row.strike}
              className={`grid grid-cols-[3.5rem_minmax(12rem,1fr)] gap-2 rounded-sm py-1 sm:grid-cols-[3.5rem_minmax(12rem,1fr)_6rem] ${
                isSpot ? "bg-accent/5" : ""
              }`}
            >
              <span
                className={`self-center text-right font-mono text-[11px] ${
                  isSpot ? "font-semibold text-accent" : "text-muted"
                }`}
              >
                {row.strike.toFixed(1)}
              </span>
              <div className="relative h-5">
                <div className="absolute inset-y-0 left-1/2 w-px -translate-x-1/2 bg-border" />
                {entries.map(({ item, index, value }) => {
                  const key = `${row.strike}-${item.expiration}`;
                  const active = hover === key;
                  const width = (Math.abs(value) / maxAbs) * 100;
                  const positive = value >= 0;
                  return (
                    <div
                      key={item.expiration}
                      className={`pointer-events-none absolute inset-y-0 flex items-center ${
                        positive
                          ? "left-1/2 right-0 justify-start pl-px"
                          : "left-0 right-1/2 justify-end pr-px"
                      }`}
                    >
                      <div
                        title={`${item.expiration} · ${row.strike.toFixed(1)} · ${Math.round(value).toLocaleString()}`}
                        aria-label={`${item.expiration}, strike ${row.strike.toFixed(1)}, net GEX ${Math.round(value).toLocaleString()}`}
                        tabIndex={0}
                        onMouseEnter={() => setHover(key)}
                        onMouseLeave={() => setHover(null)}
                        onFocus={() => setHover(key)}
                        onBlur={() => setHover(null)}
                        className={`pointer-events-auto h-3.5 border-y transition-opacity ${
                          positive ? "rounded-r-sm" : "rounded-l-sm"
                        }`}
                        style={{
                          backgroundColor: SERIES_COLORS[index % SERIES_COLORS.length],
                          borderColor: SERIES_COLORS[index % SERIES_COLORS.length],
                          opacity: active ? 1 : 0.55,
                          width: `${width}%`,
                          zIndex: active ? 50 : undefined,
                        }}
                      />
                    </div>
                  );
                })}
                {activeEntry && (
                  <span className="absolute right-1 top-0 rounded bg-surface/90 px-1 font-mono text-[9px] leading-5 text-foreground sm:hidden">
                    {shortDate(activeEntry.item.expiration)} {Math.round(activeEntry.value).toLocaleString()}
                  </span>
                )}
              </div>
              <div className="hidden self-center text-right font-mono text-[10px] text-muted sm:block">
                {activeEntry
                  ? `${shortDate(activeEntry.item.expiration)} ${Math.round(activeEntry.value).toLocaleString()}`
                  : `${entries.length} dates`}
              </div>
            </div>
          );
        })}
      </div>
      <p className="mt-2 text-xs text-faint">
        Expirations overlap independently on one shared scale; values are not summed.
        Bars left of center are negative GEX and bars right are positive.
      </p>
    </div>
  );
}

export default function GexStrikeChart({ series, spot }: Props) {
  if (series.length > 1) {
    return <MultiExpirationGexChart series={series} spot={spot} />;
  }
  const profile = series[0]?.profile ?? [];
  return (
    <StrikeBarChart
      points={profile.map((p) => ({ strike: p.strike, value: p.net_gex }))}
      spot={spot}
      positiveLabel="Long gamma (+)"
      negativeLabel="Short gamma (−)"
    />
  );
}
