"use client";

import { useMemo, useState } from "react";
import { DailyMetricsPoint } from "@/lib/api";

const WIDTH = 760;
const HEIGHT = 230;
const PAD = { t: 16, r: 12, b: 28, l: 60 };

function formatValue(value: number) {
  const absolute = Math.abs(value);
  if (absolute >= 1e9) return `${(value / 1e9).toFixed(1)}B`;
  if (absolute >= 1e6) return `${(value / 1e6).toFixed(1)}M`;
  if (absolute >= 1e3) return `${(value / 1e3).toFixed(1)}K`;
  return value.toFixed(0);
}

interface Props {
  points: DailyMetricsPoint[];
}

// Daily net-GEX history comes from the project's own end-of-day snapshots.
// The zero baseline keeps long- and short-gamma regimes directly comparable.
export default function HistoricalGexChart({ points }: Props) {
  const [hover, setHover] = useState<number | null>(null);
  const model = useMemo(() => {
    const rows = points
      .filter((point) => Number.isFinite(point.net_gex_total))
      .map((point) => ({ date: point.date, value: Number(point.net_gex_total) }))
      .sort((a, b) => a.date.localeCompare(b.date));
    if (rows.length < 2) return null;

    const plotWidth = WIDTH - PAD.l - PAD.r;
    const plotHeight = HEIGHT - PAD.t - PAD.b;
    const minValue = Math.min(0, ...rows.map((point) => point.value));
    const maxValue = Math.max(0, ...rows.map((point) => point.value));
    const padding = (maxValue - minValue) * 0.12 || 1;
    const min = minValue - padding;
    const max = maxValue + padding;
    const x = (index: number) => PAD.l + (index / (rows.length - 1)) * plotWidth;
    const y = (value: number) => PAD.t + ((max - value) / (max - min)) * plotHeight;
    const path = rows
      .map((point, index) => `${index === 0 ? "M" : "L"}${x(index).toFixed(1)},${y(point.value).toFixed(1)}`)
      .join("");
    const ticks = [min, (min + max) / 2, max];

    return { rows, x, y, path, ticks, zeroY: y(0) };
  }, [points]);

  if (!model) return null;

  const active = hover == null ? null : model.rows[hover];

  return (
    <div className="relative">
      <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} className="h-auto w-full select-none">
        {model.ticks.map((value, index) => (
          <g key={index}>
            <line
              x1={PAD.l}
              x2={WIDTH - PAD.r}
              y1={model.y(value)}
              y2={model.y(value)}
              stroke="var(--border)"
              strokeWidth={1}
            />
            <text
              x={PAD.l - 7}
              y={model.y(value) + 3}
              fill="var(--faint)"
              fontFamily="var(--font-mono)"
              fontSize={10}
              textAnchor="end"
            >
              {formatValue(value)}
            </text>
          </g>
        ))}
        <line
          x1={PAD.l}
          x2={WIDTH - PAD.r}
          y1={model.zeroY}
          y2={model.zeroY}
          stroke="var(--faint)"
          strokeDasharray="4 4"
          strokeWidth={1}
        />
        <path
          d={model.path}
          fill="none"
          stroke="var(--accent)"
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
        />
        {model.rows.map((point, index) => (
          <g key={point.date}>
            <circle
              cx={model.x(index)}
              cy={model.y(point.value)}
              fill={point.value >= 0 ? "var(--positive)" : "var(--negative)"}
              r={3.5}
              stroke="var(--surface)"
              strokeWidth={2}
            />
            <circle
              cx={model.x(index)}
              cy={model.y(point.value)}
              fill="transparent"
              r={12}
              onMouseEnter={() => setHover(index)}
              onMouseLeave={() => setHover((current) => (current === index ? null : current))}
            />
            {(index === 0 || index === model.rows.length - 1) && (
              <text
                x={model.x(index)}
                y={HEIGHT - 8}
                fill="var(--faint)"
                fontFamily="var(--font-mono)"
                fontSize={10}
                textAnchor={index === 0 ? "start" : "end"}
              >
                {point.date.slice(5)}
              </text>
            )}
          </g>
        ))}
      </svg>
      {active && (
        <div
          className="pointer-events-none absolute top-1 z-10 rounded-md border border-border-strong bg-surface-2 px-2 py-1.5 text-[11px] shadow-lg"
          style={{
            left: `${(model.x(hover as number) / WIDTH) * 100}%`,
            transform: model.x(hover as number) > WIDTH / 2 ? "translateX(-105%)" : "translateX(5%)",
          }}
        >
          <div className="font-mono text-muted">{active.date}</div>
          <div className="font-mono">
            Net GEX <span className={active.value >= 0 ? "text-positive" : "text-negative"}>{formatValue(active.value)}</span>
          </div>
        </div>
      )}
    </div>
  );
}
