"use client";

import { useMemo, useState } from "react";
import { TermStructurePoint } from "@/lib/api";

// IV term structure: ATM implied vol per expiration. Single accent series
// (no legend needed - the panel title names it), >=8px markers with a
// surface ring, per-point hover tooltip. Upward slope = contango,
// downward = event premium in the front.

const VBW = 760;
const VBH = 200;
const PAD = { t: 14, r: 56, b: 24, l: 8 };

interface Props {
  points: TermStructurePoint[];
}

export default function TermStructureChart({ points }: Props) {
  const [hover, setHover] = useState<number | null>(null);

  const model = useMemo(() => {
    const rows = [...points].sort((a, b) => a.expiration.localeCompare(b.expiration));
    if (rows.length < 2) return null;

    const plotW = VBW - PAD.l - PAD.r;
    const plotH = VBH - PAD.t - PAD.b;
    let min = Math.min(...rows.map((p) => p.atm_iv));
    let max = Math.max(...rows.map((p) => p.atm_iv));
    const padV = (max - min) * 0.15 || 0.01;
    min -= padV;
    max += padV;

    const x = (i: number) => PAD.l + (rows.length === 1 ? plotW / 2 : (i / (rows.length - 1)) * plotW);
    const y = (iv: number) => PAD.t + ((max - iv) / (max - min)) * plotH;
    const ticks = Array.from({ length: 3 }, (_, i) => min + ((max - min) * i) / 2);
    const path = rows
      .map((p, i) => `${i === 0 ? "M" : "L"}${x(i).toFixed(1)},${y(p.atm_iv).toFixed(1)}`)
      .join("");
    return { rows, x, y, ticks, path };
  }, [points]);

  if (!model) return null;

  const active = hover != null ? model.rows[hover] : null;

  return (
    <div className="relative">
      <svg viewBox={`0 0 ${VBW} ${VBH}`} className="h-auto w-full select-none">
        {model.ticks.map((iv, i) => (
          <g key={i}>
            <line
              x1={PAD.l}
              x2={VBW - PAD.r}
              y1={model.y(iv)}
              y2={model.y(iv)}
              stroke="var(--border)"
              strokeWidth={1}
            />
            <text
              x={VBW - PAD.r + 6}
              y={model.y(iv) + 3}
              fontSize={10}
              fill="var(--faint)"
              fontFamily="var(--font-mono)"
            >
              {(iv * 100).toFixed(1)}%
            </text>
          </g>
        ))}

        <path
          d={model.path}
          fill="none"
          stroke="var(--accent)"
          strokeWidth={2}
          strokeLinejoin="round"
          strokeLinecap="round"
        />

        {model.rows.map((p, i) => (
          <g key={p.expiration}>
            <circle
              cx={model.x(i)}
              cy={model.y(p.atm_iv)}
              r={4}
              fill="var(--accent)"
              stroke="var(--surface)"
              strokeWidth={2}
            />
            {/* generous hit target around the small marker */}
            <circle
              cx={model.x(i)}
              cy={model.y(p.atm_iv)}
              r={14}
              fill="transparent"
              onMouseEnter={() => setHover(i)}
              onMouseLeave={() => setHover((h) => (h === i ? null : h))}
            />
            <text
              x={model.x(i)}
              y={VBH - 8}
              fontSize={10}
              fill="var(--faint)"
              fontFamily="var(--font-mono)"
              textAnchor={i === 0 ? "start" : i === model.rows.length - 1 ? "end" : "middle"}
            >
              {p.dte != null ? `${p.dte}d` : p.expiration.slice(5)}
            </text>
          </g>
        ))}
      </svg>

      {active && (
        <div
          className="pointer-events-none absolute top-1 z-10 rounded-md border border-border-strong bg-surface-2 px-2 py-1.5 text-[11px] shadow-lg"
          style={{
            left: `${(model.x(hover as number) / VBW) * 100}%`,
            transform:
              model.x(hover as number) > VBW / 2
                ? "translateX(-105%)"
                : "translateX(5%)",
          }}
        >
          <div className="mb-0.5 font-mono text-muted">{active.expiration}</div>
          <div className="font-mono">
            ATM IV{" "}
            <span className="text-foreground">{(active.atm_iv * 100).toFixed(1)}%</span>
            {active.dte != null && <span className="text-muted"> · {active.dte} DTE</span>}
          </div>
        </div>
      )}
    </div>
  );
}
