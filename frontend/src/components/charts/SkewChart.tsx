"use client";

import { useMemo, useState } from "react";
import { SkewPoint } from "@/lib/api";

// IV smile/skew: call vs put implied vol by strike. Two 2px lines in the
// calls/puts tokens; identity is never color-alone (legend + direct
// end-labels) since green/red sits in the CVD floor band. Crosshair +
// tooltip on hover; spot marked with a vertical reference line.

const VBW = 760;
const VBH = 260;
const PAD = { t: 14, r: 76, b: 22, l: 8 };

interface Props {
  points: SkewPoint[];
  spot: number;
}

export default function SkewChart({ points, spot }: Props) {
  const [hover, setHover] = useState<number | null>(null);

  const model = useMemo(() => {
    const rows = points
      .filter((p) => p.iv_call != null || p.iv_put != null)
      .sort((a, b) => a.strike - b.strike);
    if (rows.length < 2) return null;

    const plotW = VBW - PAD.l - PAD.r;
    const plotH = VBH - PAD.t - PAD.b;
    const ivs = rows.flatMap((p) =>
      [p.iv_call, p.iv_put].filter((v): v is number => v != null),
    );
    let min = Math.min(...ivs);
    let max = Math.max(...ivs);
    const padV = (max - min) * 0.08 || 0.01;
    min -= padV;
    max += padV;

    const k0 = rows[0].strike;
    const k1 = rows[rows.length - 1].strike;
    const x = (k: number) => PAD.l + ((k - k0) / (k1 - k0 || 1)) * plotW;
    const y = (iv: number) => PAD.t + ((max - iv) / (max - min)) * plotH;
    const ticks = Array.from({ length: 4 }, (_, i) => min + ((max - min) * i) / 3);

    const path = (key: "iv_call" | "iv_put") =>
      rows
        .map((p, i) =>
          p[key] == null
            ? ""
            : `${i === 0 || rows[i - 1][key] == null ? "M" : "L"}${x(p.strike).toFixed(1)},${y(p[key] as number).toFixed(1)}`,
        )
        .join("");

    return { rows, x, y, ticks, callPath: path("iv_call"), putPath: path("iv_put") };
  }, [points]);

  if (!model) return null;

  const active = hover != null ? model.rows[hover] : null;
  const lastRow = model.rows[model.rows.length - 1];
  const pct = (v: number | null | undefined) =>
    v == null ? "—" : `${(v * 100).toFixed(1)}%`;

  return (
    <div className="relative">
      <div className="mb-2 flex items-center gap-4 text-xs text-muted">
        <span className="flex items-center gap-1.5">
          <span className="inline-block h-0.5 w-4 rounded bg-positive" />
          Call IV
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block h-0.5 w-4 rounded bg-negative" />
          Put IV
        </span>
        <span className="ml-auto flex items-center gap-1.5">
          <span className="inline-block h-3 w-0.5 bg-accent" />
          spot {spot.toFixed(2)}
        </span>
      </div>

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

        {/* spot reference */}
        {spot >= model.rows[0].strike && spot <= lastRow.strike && (
          <line
            x1={model.x(spot)}
            x2={model.x(spot)}
            y1={PAD.t}
            y2={VBH - PAD.b}
            stroke="var(--accent)"
            strokeWidth={1}
            opacity={0.6}
          />
        )}

        <path
          d={model.callPath}
          fill="none"
          stroke="var(--positive)"
          strokeWidth={2}
          strokeLinejoin="round"
          strokeLinecap="round"
        />
        <path
          d={model.putPath}
          fill="none"
          stroke="var(--negative)"
          strokeWidth={2}
          strokeLinejoin="round"
          strokeLinecap="round"
        />

        {/* Direct labels (secondary identity channel for the CVD floor-band
            pair). Anchored at the LEFT edge where put and call IV separate
            most - the lines converge at the right, so end-labels collide. */}
        {model.rows[0].iv_put != null && (
          <text
            x={model.x(model.rows[0].strike) + 4}
            y={model.y(model.rows[0].iv_put) - 6}
            fontSize={10}
            fill="var(--muted)"
          >
            Put IV
          </text>
        )}
        {model.rows[0].iv_call != null && (
          <text
            x={model.x(model.rows[0].strike) + 4}
            y={model.y(model.rows[0].iv_call) + 14}
            fontSize={10}
            fill="var(--muted)"
          >
            Call IV
          </text>
        )}

        {/* hover capture + crosshair */}
        {model.rows.map((p, i) => {
          const left =
            i === 0 ? PAD.l : (model.x(p.strike) + model.x(model.rows[i - 1].strike)) / 2;
          const right =
            i === model.rows.length - 1
              ? VBW - PAD.r
              : (model.x(p.strike) + model.x(model.rows[i + 1].strike)) / 2;
          return (
            <rect
              key={p.strike}
              x={left}
              y={PAD.t}
              width={Math.max(0, right - left)}
              height={VBH - PAD.t - PAD.b}
              fill="transparent"
              onMouseEnter={() => setHover(i)}
              onMouseLeave={() => setHover((h) => (h === i ? null : h))}
            />
          );
        })}
        {active && (
          <g>
            <line
              x1={model.x(active.strike)}
              x2={model.x(active.strike)}
              y1={PAD.t}
              y2={VBH - PAD.b}
              stroke="var(--border-strong)"
              strokeWidth={1}
            />
            {active.iv_call != null && (
              <circle
                cx={model.x(active.strike)}
                cy={model.y(active.iv_call)}
                r={4}
                fill="var(--positive)"
                stroke="var(--surface)"
                strokeWidth={2}
              />
            )}
            {active.iv_put != null && (
              <circle
                cx={model.x(active.strike)}
                cy={model.y(active.iv_put)}
                r={4}
                fill="var(--negative)"
                stroke="var(--surface)"
                strokeWidth={2}
              />
            )}
          </g>
        )}

        {/* strike ticks */}
        {[model.rows[0], model.rows[Math.floor(model.rows.length / 2)], lastRow].map(
          (p, i) => (
            <text
              key={`k-${p.strike}-${i}`}
              x={i === 0 ? PAD.l : i === 1 ? (PAD.l + VBW - PAD.r) / 2 : VBW - PAD.r}
              y={VBH - 6}
              fontSize={10}
              fill="var(--faint)"
              fontFamily="var(--font-mono)"
              textAnchor={i === 0 ? "start" : i === 1 ? "middle" : "end"}
            >
              {p.strike.toFixed(1)}
            </text>
          ),
        )}
      </svg>

      {active && (
        <div
          className="pointer-events-none absolute top-6 z-10 rounded-md border border-border-strong bg-surface-2 px-2 py-1.5 text-[11px] shadow-lg"
          style={{
            left: `${(model.x(active.strike) / VBW) * 100}%`,
            transform:
              model.x(active.strike) > VBW / 2
                ? "translateX(-105%)"
                : "translateX(5%)",
          }}
        >
          <div className="mb-0.5 font-mono text-muted">
            strike {active.strike.toFixed(1)}
          </div>
          <div className="grid grid-cols-2 gap-x-3 gap-y-0.5 font-mono">
            <span className="text-muted">Call IV</span>
            <span className="text-right">{pct(active.iv_call)}</span>
            <span className="text-muted">Put IV</span>
            <span className="text-right">{pct(active.iv_put)}</span>
            <span className="text-muted">Skew</span>
            <span className="text-right">
              {active.iv_skew == null ? "—" : (active.iv_skew * 100).toFixed(2)}
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
