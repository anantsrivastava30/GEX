"use client";

import { useMemo, useState } from "react";

// Generic horizontal diverging bar chart over a strike ladder (net GEX,
// vanna, charm, ...). Polarity is encoded twice: bar direction from the
// centered zero baseline AND color (positive/negative tokens), so the read
// survives color-vision deficiency. Dependency-free SVG/CSS.

export interface StrikePoint {
  strike: number;
  value: number;
}

interface Props {
  points: StrikePoint[];
  spot: number;
  positiveLabel: string;
  negativeLabel: string;
  formatValue?: (value: number) => string;
  maxHeightClass?: string;
}

const defaultFormat = (value: number) => Math.round(value).toLocaleString();

export default function StrikeBarChart({
  points,
  spot,
  positiveLabel,
  negativeLabel,
  formatValue = defaultFormat,
  maxHeightClass = "max-h-[28rem]",
}: Props) {
  const [hover, setHover] = useState<number | null>(null);

  const { rows, maxAbs, spotStrike } = useMemo(() => {
    // Strikes descend so higher prices sit at the top, like an option chain.
    const sorted = [...points].sort((a, b) => b.strike - a.strike);
    const maxAbs = Math.max(1e-12, ...sorted.map((p) => Math.abs(p.value)));
    let spotStrike = sorted[0]?.strike ?? spot;
    let best = Infinity;
    for (const p of sorted) {
      const d = Math.abs(p.strike - spot);
      if (d < best) {
        best = d;
        spotStrike = p.strike;
      }
    }
    return { rows: sorted, maxAbs, spotStrike };
  }, [points, spot]);

  if (!rows.length) return null;

  return (
    <div className="relative">
      <div className="mb-2 flex items-center gap-4 text-xs text-muted">
        <span className="flex items-center gap-1.5">
          <span className="inline-block h-2 w-3 rounded-sm bg-positive" />
          {positiveLabel}
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block h-2 w-3 rounded-sm bg-negative" />
          {negativeLabel}
        </span>
        <span className="ml-auto flex items-center gap-1.5">
          <span className="inline-block h-3 w-0.5 bg-accent" />
          spot {spot.toFixed(2)}
        </span>
      </div>

      <div className={`${maxHeightClass} overflow-y-auto pr-1`}>
        {rows.map((p) => {
          const pctWidth = (Math.abs(p.value) / maxAbs) * 100;
          const positive = p.value >= 0;
          const isSpot = p.strike === spotStrike;
          const active = hover === p.strike;
          return (
            <div
              key={p.strike}
              onMouseEnter={() => setHover(p.strike)}
              onMouseLeave={() => setHover(null)}
              className={`flex items-center gap-2 rounded-sm py-[1px] ${
                active ? "bg-surface-hover" : ""
              }`}
            >
              <span
                className={`w-14 shrink-0 text-right font-mono text-[11px] ${
                  isSpot ? "font-semibold text-accent" : "text-muted"
                }`}
              >
                {p.strike.toFixed(1)}
              </span>
              <div className="relative flex h-4 flex-1 items-center">
                {/* zero baseline */}
                <div
                  className={`absolute inset-y-0 left-1/2 w-px -translate-x-1/2 ${
                    isSpot ? "bg-accent" : "bg-border"
                  }`}
                />
                {/* negative half (grows left from center) */}
                <div className="flex h-full flex-1 justify-end pr-px">
                  {!positive && (
                    <div
                      className="h-2.5 rounded-l-[3px] bg-negative"
                      style={{ width: `${pctWidth}%` }}
                    />
                  )}
                </div>
                {/* positive half (grows right from center) */}
                <div className="flex h-full flex-1 justify-start pl-px">
                  {positive && (
                    <div
                      className="h-2.5 rounded-r-[3px] bg-positive"
                      style={{ width: `${pctWidth}%` }}
                    />
                  )}
                </div>
              </div>
              <span
                className={`w-24 shrink-0 text-right font-mono text-[11px] transition-opacity ${
                  active ? "opacity-100" : "opacity-0"
                } ${positive ? "text-positive" : "text-negative"}`}
              >
                {formatValue(p.value)}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
