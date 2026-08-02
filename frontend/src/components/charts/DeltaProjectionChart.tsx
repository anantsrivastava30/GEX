"use client";

import { useMemo, useState } from "react";
import { DeltaProjectionResponse } from "@/lib/api";

// Intraday spot path (top) and net dealer delta exposure per bar (bottom),
// as two stacked charts sharing one time axis - never a dual-axis chart.
// The dashed extension in the bottom panel is a linear projection of the
// exposure trend to the 16:00 close, marked with a hollow point.

const VBW = 760;
const PRICE_H = 130;
const GAP = 26;
const EXPO_H = 130;
const VBH = PRICE_H + GAP + EXPO_H + 22;
const PAD = { r: 64, l: 8 };

function clock(ts: string) {
  const m = ts.match(/T(\d{2}:\d{2})/);
  return m ? m[1] : ts.slice(11, 16);
}

function compact(v: number) {
  const abs = Math.abs(v);
  if (abs >= 1e9) return `${(v / 1e9).toFixed(1)}B`;
  if (abs >= 1e6) return `${(v / 1e6).toFixed(1)}M`;
  if (abs >= 1e3) return `${(v / 1e3).toFixed(0)}K`;
  return v.toFixed(0);
}

interface Props {
  data: DeltaProjectionResponse;
}

export default function DeltaProjectionChart({ data }: Props) {
  const [hover, setHover] = useState<number | null>(null);

  const model = useMemo(() => {
    const bars = data.bars;
    if (!bars.length) return null;

    // Slot-index x positions: prev close sits one slot left of the session
    // and the projection one slot right. A time-linear axis would waste half
    // the plot on the overnight gap between yesterday's close and the open.
    const hasProjection = data.projection_ts != null;
    const slots = bars.length + 1 + (hasProjection ? 1 : 0);
    const slotX = (i: number) =>
      PAD.l + ((i + 0.5) / slots) * (VBW - PAD.l - PAD.r);
    const slotByTs = new Map<string, number>();
    slotByTs.set(data.prev_close.ts, 0);
    bars.forEach((b, i) => slotByTs.set(b.ts, i + 1));
    if (data.projection_ts) slotByTs.set(data.projection_ts, bars.length + 1);
    const x = (ts: string) => slotX(slotByTs.get(ts) ?? bars.length);

    const prices = [data.prev_close.price, ...bars.map((b) => b.price)];
    let pMin = Math.min(...prices);
    let pMax = Math.max(...prices);
    const pPad = (pMax - pMin) * 0.12 || 0.5;
    pMin -= pPad;
    pMax += pPad;
    const yPrice = (p: number) =>
      8 + ((pMax - p) / (pMax - pMin)) * (PRICE_H - 16);

    const expoValues = bars.map((b) => b.delta_exposure);
    if (data.projection_exposure != null) expoValues.push(data.projection_exposure);
    let eMax = Math.max(0, ...expoValues);
    let eMin = Math.min(0, ...expoValues);
    if (eMax === eMin) eMax = eMin + 1;
    const ePad = (eMax - eMin) * 0.1;
    eMax += ePad;
    eMin -= ePad;
    const yExpo = (v: number) =>
      PRICE_H + GAP + ((eMax - v) / (eMax - eMin)) * (EXPO_H - 8);

    const pricePath = [data.prev_close, ...bars]
      .map(
        (p, i) =>
          `${i === 0 ? "M" : "L"}${x(p.ts).toFixed(1)},${yPrice(p.price).toFixed(1)}`,
      )
      .join("");

    const slot = bars.length
      ? (x(bars[bars.length - 1].ts) - x(bars[0].ts)) / Math.max(1, bars.length - 1)
      : 24;
    const barW = Math.max(3, Math.min(18, slot * 0.55));

    return { bars, x, yPrice, yExpo, pricePath, barW, pMin, pMax, eMin, eMax };
  }, [data]);

  if (!model) return null;

  const active = hover != null ? model.bars[hover] : null;
  const zeroY = model.yExpo(0);
  const lastBar = model.bars[model.bars.length - 1];

  return (
    <div className="relative">
      <div className="mb-2 flex items-center gap-4 text-xs text-muted">
        <span className="flex items-center gap-1.5">
          <span className="inline-block h-0.5 w-4 rounded bg-accent" />
          Spot (30m)
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block h-2 w-3 rounded-sm bg-positive" />
          Net delta exposure
        </span>
        {data.projection_exposure != null && (
          <span className="flex items-center gap-1.5">
            <span className="inline-block h-2 w-2 rounded-full border-2 border-warning" />
            Projected close {compact(data.projection_exposure)}
          </span>
        )}
      </div>

      <svg viewBox={`0 0 ${VBW} ${VBH}`} className="h-auto w-full select-none">
        {/* price panel gridline + labels */}
        {[model.pMax, model.pMin].map((p, i) => (
          <g key={`p-${i}`}>
            <line
              x1={PAD.l}
              x2={VBW - PAD.r}
              y1={model.yPrice(p)}
              y2={model.yPrice(p)}
              stroke="var(--border)"
              strokeWidth={1}
            />
            <text
              x={VBW - PAD.r + 6}
              y={model.yPrice(p) + 3}
              fontSize={10}
              fill="var(--faint)"
              fontFamily="var(--font-mono)"
            >
              {p.toFixed(2)}
            </text>
          </g>
        ))}

        {/* prev-close reference */}
        <line
          x1={PAD.l}
          x2={VBW - PAD.r}
          y1={model.yPrice(data.prev_close.price)}
          y2={model.yPrice(data.prev_close.price)}
          stroke="var(--faint)"
          strokeWidth={1}
          strokeDasharray="3 3"
          opacity={0.6}
        />

        <path
          d={model.pricePath}
          fill="none"
          stroke="var(--accent)"
          strokeWidth={2}
          strokeLinejoin="round"
          strokeLinecap="round"
        />

        {/* exposure panel: zero baseline + extremes */}
        <line
          x1={PAD.l}
          x2={VBW - PAD.r}
          y1={zeroY}
          y2={zeroY}
          stroke="var(--border-strong)"
          strokeWidth={1}
        />
        <text
          x={VBW - PAD.r + 6}
          y={zeroY + 3}
          fontSize={10}
          fill="var(--faint)"
          fontFamily="var(--font-mono)"
        >
          0
        </text>

        {model.bars.map((bar, i) => {
          const positive = bar.delta_exposure >= 0;
          const top = positive ? model.yExpo(bar.delta_exposure) : zeroY;
          const h = Math.max(1, Math.abs(model.yExpo(bar.delta_exposure) - zeroY));
          return (
            <rect
              key={bar.ts}
              x={model.x(bar.ts) - model.barW / 2}
              y={top}
              width={model.barW}
              height={h}
              rx={2}
              fill={positive ? "var(--positive)" : "var(--negative)"}
              opacity={hover === null || hover === i ? 1 : 0.45}
            />
          );
        })}

        {/* projection: dashed trend extension + hollow marker */}
        {data.projection_ts && data.projection_exposure != null && (
          <g>
            <line
              x1={model.x(lastBar.ts)}
              y1={model.yExpo(lastBar.delta_exposure)}
              x2={model.x(data.projection_ts)}
              y2={model.yExpo(data.projection_exposure)}
              stroke="var(--warning)"
              strokeWidth={1.5}
              strokeDasharray="4 3"
            />
            <circle
              cx={model.x(data.projection_ts)}
              cy={model.yExpo(data.projection_exposure)}
              r={5}
              fill="var(--surface)"
              stroke="var(--warning)"
              strokeWidth={2}
            />
          </g>
        )}

        {/* hover capture across both panels */}
        {model.bars.map((bar, i) => (
          <rect
            key={`h-${bar.ts}`}
            x={model.x(bar.ts) - model.barW / 2 - 4}
            y={0}
            width={model.barW + 8}
            height={VBH - 18}
            fill="transparent"
            onMouseEnter={() => setHover(i)}
            onMouseLeave={() => setHover((h) => (h === i ? null : h))}
          />
        ))}
        {active && (
          <g>
            <line
              x1={model.x(active.ts)}
              x2={model.x(active.ts)}
              y1={4}
              y2={VBH - 18}
              stroke="var(--border-strong)"
              strokeWidth={1}
            />
            <circle
              cx={model.x(active.ts)}
              cy={model.yPrice(active.price)}
              r={4}
              fill="var(--accent)"
              stroke="var(--surface)"
              strokeWidth={2}
            />
          </g>
        )}

        {/* time ticks: prev close, midday, last bar */}
        {[
          data.prev_close.ts,
          model.bars[Math.floor(model.bars.length / 2)]?.ts,
          lastBar.ts,
        ]
          .filter((ts, i, all) => Boolean(ts) && all.indexOf(ts) === i)
          .map((ts, i) => (
            <text
              key={`t-${ts}-${i}`}
              x={model.x(ts as string)}
              y={VBH - 5}
              fontSize={10}
              fill="var(--faint)"
              fontFamily="var(--font-mono)"
              textAnchor={i === 0 ? "start" : "middle"}
            >
              {clock(ts as string)}
            </text>
          ))}
      </svg>

      {active && (
        <div
          className="pointer-events-none absolute top-6 z-10 rounded-md border border-border-strong bg-surface-2 px-2 py-1.5 text-[11px] shadow-lg"
          style={{
            left: `${(model.x(active.ts) / VBW) * 100}%`,
            transform:
              model.x(active.ts) > VBW / 2 ? "translateX(-105%)" : "translateX(5%)",
          }}
        >
          <div className="mb-0.5 font-mono text-muted">{clock(active.ts)}</div>
          <div className="grid grid-cols-2 gap-x-3 gap-y-0.5 font-mono">
            <span className="text-muted">Spot</span>
            <span className="text-right">{active.price.toFixed(2)}</span>
            <span className="text-muted">Δ exp</span>
            <span
              className={`text-right ${
                active.delta_exposure >= 0 ? "text-positive" : "text-negative"
              }`}
            >
              {compact(active.delta_exposure)}
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
