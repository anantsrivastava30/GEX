"use client";

import { useMemo, useState } from "react";
import { Candle } from "@/lib/api";

// Dependency-free SVG candlestick chart. Fixed viewBox scaled to container
// width; up/down encoded by the app's positive/negative tokens. Per-candle
// hover tooltip with the full OHLC read.

const VBW = 760;
const VBH = 300;
const PAD = { t: 10, r: 48, b: 20, l: 8 };

interface Props {
  candles: Candle[];
}

export default function CandlestickChart({ candles }: Props) {
  const [hover, setHover] = useState<number | null>(null);

  const model = useMemo(() => {
    const n = candles.length;
    if (!n) return null;
    const plotW = VBW - PAD.l - PAD.r;
    const plotH = VBH - PAD.t - PAD.b;
    const lows = candles.map((c) => c.low);
    const highs = candles.map((c) => c.high);
    let min = Math.min(...lows);
    let max = Math.max(...highs);
    const pad = (max - min) * 0.04 || 1;
    min -= pad;
    max += pad;
    const slot = plotW / n;
    const bodyW = Math.max(1, Math.min(slot * 0.62, 12));
    const x = (i: number) => PAD.l + (i + 0.5) * slot;
    const y = (p: number) => PAD.t + ((max - p) / (max - min)) * plotH;
    const ticks = Array.from({ length: 5 }, (_, i) => min + ((max - min) * i) / 4);
    return { n, slot, bodyW, x, y, ticks, min, max, plotW };
  }, [candles]);

  if (!model) return null;

  const last = candles[candles.length - 1];
  const first = candles[0];
  const mid = candles[Math.floor(candles.length / 2)];
  const active = hover != null ? candles[hover] : null;

  return (
    <div className="relative">
      <svg viewBox={`0 0 ${VBW} ${VBH}`} className="h-auto w-full select-none">
        {/* price gridlines + labels */}
        {model.ticks.map((p, i) => (
          <g key={i}>
            <line
              x1={PAD.l}
              x2={VBW - PAD.r}
              y1={model.y(p)}
              y2={model.y(p)}
              stroke="var(--border)"
              strokeWidth={1}
            />
            <text
              x={VBW - PAD.r + 6}
              y={model.y(p) + 3}
              fontSize={10}
              fill="var(--faint)"
              fontFamily="var(--font-mono)"
            >
              {p.toFixed(2)}
            </text>
          </g>
        ))}

        {/* last-close reference line */}
        <line
          x1={PAD.l}
          x2={VBW - PAD.r}
          y1={model.y(last.close)}
          y2={model.y(last.close)}
          stroke="var(--accent)"
          strokeWidth={1}
          strokeDasharray="3 3"
          opacity={0.7}
        />

        {/* candles */}
        {candles.map((c, i) => {
          const up = c.close >= c.open;
          const color = up ? "var(--positive)" : "var(--negative)";
          const cx = model.x(i);
          const yOpen = model.y(c.open);
          const yClose = model.y(c.close);
          const top = Math.min(yOpen, yClose);
          const h = Math.max(1, Math.abs(yClose - yOpen));
          return (
            <g key={c.date}>
              <line
                x1={cx}
                x2={cx}
                y1={model.y(c.high)}
                y2={model.y(c.low)}
                stroke={color}
                strokeWidth={1}
              />
              <rect
                x={cx - model.bodyW / 2}
                y={top}
                width={model.bodyW}
                height={h}
                fill={color}
                rx={1}
              />
            </g>
          );
        })}

        {/* hover capture + crosshair */}
        {candles.map((c, i) => (
          <rect
            key={`h-${c.date}`}
            x={model.x(i) - model.slot / 2}
            y={PAD.t}
            width={model.slot}
            height={VBH - PAD.t - PAD.b}
            fill="transparent"
            onMouseEnter={() => setHover(i)}
            onMouseLeave={() => setHover((h) => (h === i ? null : h))}
          />
        ))}
        {active && (
          <line
            x1={model.x(hover as number)}
            x2={model.x(hover as number)}
            y1={PAD.t}
            y2={VBH - PAD.b}
            stroke="var(--border-strong)"
            strokeWidth={1}
          />
        )}

        {/* date ticks */}
        {[first, mid, last].map((c, i) => (
          <text
            key={`d-${c.date}-${i}`}
            x={i === 0 ? PAD.l : i === 1 ? VBW / 2 : VBW - PAD.r}
            y={VBH - 6}
            fontSize={10}
            fill="var(--faint)"
            fontFamily="var(--font-mono)"
            textAnchor={i === 0 ? "start" : i === 1 ? "middle" : "end"}
          >
            {c.date}
          </text>
        ))}
      </svg>

      {active && (
        <div
          className="pointer-events-none absolute top-1 z-10 rounded-md border border-border-strong bg-surface-2 px-2 py-1.5 text-[11px] shadow-lg"
          style={{
            left: `${((model.x(hover as number)) / VBW) * 100}%`,
            transform:
              (hover as number) > model.n / 2
                ? "translateX(-105%)"
                : "translateX(5%)",
          }}
        >
          <div className="mb-0.5 font-mono text-muted">{active.date}</div>
          <div className="grid grid-cols-2 gap-x-3 gap-y-0.5 font-mono">
            <span className="text-muted">O</span>
            <span className="text-right">{active.open.toFixed(2)}</span>
            <span className="text-muted">H</span>
            <span className="text-right">{active.high.toFixed(2)}</span>
            <span className="text-muted">L</span>
            <span className="text-right">{active.low.toFixed(2)}</span>
            <span className="text-muted">C</span>
            <span
              className={`text-right ${
                active.close >= active.open ? "text-positive" : "text-negative"
              }`}
            >
              {active.close.toFixed(2)}
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
