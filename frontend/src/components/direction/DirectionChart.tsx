"use client";

import { useMemo, useState } from "react";
import { Candle, DirectionMarker, EmaSeries } from "@/lib/api";

// Annotated price + volume chart for the market-direction page: candles,
// EMA overlays, and O'Neil event markers (rally day 1, follow-through day,
// distribution days). Two stacked panes share one time axis - never a
// dual-axis chart. Dependency-free SVG in the app's chart idiom.
//
// EMA palette validated (dataviz six checks, dark surface #0f1118):
// 20d #5b8def · 50d #d97706 · 200d #b06ef7. Identity is never color-alone:
// line-end labels and the legend name each series, markers differ by glyph.

const VBW = 760;
const PRICE_H = 240;
const VOL_H = 64;
const GAP = 14;
const PAD = { t: 22, r: 56, b: 20, l: 8 };
const VBH = PAD.t + PRICE_H + GAP + VOL_H + PAD.b;

const EMA_COLORS: Record<number, string> = {
  20: "#5b8def",
  50: "#d97706",
  200: "#b06ef7",
};

const MARKER_META: Record<
  string,
  { label: string; color: string; glyph: "up" | "circle" | "x" | "down" | "square" | "dot" }
> = {
  follow_through_day: { label: "Follow-through day", color: "var(--accent-strong)", glyph: "up" },
  rally_day1: { label: "Rally attempt day 1", color: "var(--muted)", glyph: "circle" },
  rally_failed: { label: "Rally attempt failed", color: "var(--negative)", glyph: "x" },
  correction_entered: { label: "Correction entered", color: "var(--negative)", glyph: "down" },
  under_pressure: { label: "Distribution cluster", color: "var(--warning)", glyph: "square" },
  distribution_day: { label: "Distribution day", color: "var(--negative)", glyph: "dot" },
};

interface Props {
  candles: Candle[];
  emaSeries: EmaSeries[];
  markers: DirectionMarker[];
}

function MarkerGlyph({
  kind,
  cx,
  cy,
  size = 5,
}: {
  kind: string;
  cx: number;
  cy: number;
  size?: number;
}) {
  const meta = MARKER_META[kind];
  if (!meta) return null;
  const c = meta.color;
  switch (meta.glyph) {
    case "up":
      return (
        <path
          d={`M ${cx} ${cy - size} L ${cx + size} ${cy + size} L ${cx - size} ${cy + size} Z`}
          fill={c}
        />
      );
    case "down":
      return (
        <path
          d={`M ${cx} ${cy + size} L ${cx + size} ${cy - size} L ${cx - size} ${cy - size} Z`}
          fill={c}
        />
      );
    case "circle":
      return <circle cx={cx} cy={cy} r={size - 0.5} fill="none" stroke={c} strokeWidth={1.8} />;
    case "x":
      return (
        <g stroke={c} strokeWidth={1.8} strokeLinecap="round">
          <line x1={cx - size + 1} y1={cy - size + 1} x2={cx + size - 1} y2={cy + size - 1} />
          <line x1={cx - size + 1} y1={cy + size - 1} x2={cx + size - 1} y2={cy - size + 1} />
        </g>
      );
    case "square":
      return (
        <rect x={cx - size + 1} y={cy - size + 1} width={(size - 1) * 2} height={(size - 1) * 2} fill={c} rx={1} />
      );
    default:
      return <circle cx={cx} cy={cy} r={2.2} fill={c} />;
  }
}

export default function DirectionChart({ candles, emaSeries, markers }: Props) {
  const [hover, setHover] = useState<number | null>(null);

  const model = useMemo(() => {
    const n = candles.length;
    if (!n) return null;
    const plotW = VBW - PAD.l - PAD.r;
    let min = Math.min(...candles.map((c) => c.low));
    let max = Math.max(...candles.map((c) => c.high));
    for (const s of emaSeries) {
      for (const v of s.values) {
        if (v == null) continue;
        if (v < min) min = v;
        if (v > max) max = v;
      }
    }
    const pad = (max - min) * 0.05 || 1;
    min -= pad;
    max += pad;
    const maxVol = Math.max(1, ...candles.map((c) => c.volume ?? 0));
    const slot = plotW / n;
    const bodyW = Math.max(1, Math.min(slot * 0.62, 9));
    const x = (i: number) => PAD.l + (i + 0.5) * slot;
    const y = (p: number) => PAD.t + ((max - p) / (max - min)) * PRICE_H;
    const volTop = PAD.t + PRICE_H + GAP;
    const vy = (v: number) => volTop + (1 - v / maxVol) * VOL_H;
    const ticks = Array.from({ length: 4 }, (_, i) => min + ((max - min) * i) / 3);
    const markersByIndex = new Map<number, DirectionMarker[]>();
    const dateIndex = new Map(candles.map((c, i) => [c.date, i]));
    for (const m of markers) {
      const i = dateIndex.get(m.date);
      if (i == null) continue;
      const list = markersByIndex.get(i) ?? [];
      list.push(m);
      markersByIndex.set(i, list);
    }
    return { n, slot, bodyW, x, y, vy, volTop, ticks, maxVol, markersByIndex };
  }, [candles, emaSeries, markers]);

  if (!model) return null;

  const first = candles[0];
  const mid = candles[Math.floor(candles.length / 2)];
  const last = candles[candles.length - 1];
  const active = hover != null ? candles[hover] : null;
  const activeMarkers = hover != null ? model.markersByIndex.get(hover) ?? [] : [];
  const distributionDates = new Set(
    markers.filter((m) => m.kind === "distribution_day").map((m) => m.date),
  );
  const ftdDates = new Set(
    markers.filter((m) => m.kind === "follow_through_day").map((m) => m.date),
  );
  const usedKinds = Array.from(new Set(markers.map((m) => m.kind))).filter(
    (k) => MARKER_META[k],
  );

  const emaPath = (s: EmaSeries) => {
    let d = "";
    s.values.forEach((v, i) => {
      if (v == null) return;
      const cmd = d === "" ? "M" : "L";
      d += `${cmd} ${model.x(i).toFixed(1)} ${model.y(v).toFixed(1)} `;
    });
    return d;
  };

  // Series with no in-window values (young history) are dropped entirely so
  // the legend never names an invisible line. End labels are collision-
  // resolved top-down with a minimum gap.
  const visibleSeries = emaSeries.filter((s) => s.values.some((v) => v != null));
  const endLabels = visibleSeries
    .map((s) => {
      const lastValue = [...s.values].reverse().find((v) => v != null);
      return lastValue != null
        ? { period: s.period, y: model.y(lastValue) }
        : null;
    })
    .filter((l): l is { period: number; y: number } => l !== null)
    .sort((a, b) => a.y - b.y);
  for (let i = 1; i < endLabels.length; i++) {
    if (endLabels[i].y - endLabels[i - 1].y < 12) {
      endLabels[i].y = endLabels[i - 1].y + 12;
    }
  }

  return (
    <div className="relative">
      <svg viewBox={`0 0 ${VBW} ${VBH}`} className="h-auto w-full select-none">
        {/* price gridlines */}
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
              {p.toFixed(p >= 1000 ? 0 : 2)}
            </text>
          </g>
        ))}

        {/* volume pane baseline */}
        <line
          x1={PAD.l}
          x2={VBW - PAD.r}
          y1={model.volTop + VOL_H}
          y2={model.volTop + VOL_H}
          stroke="var(--border)"
          strokeWidth={1}
        />
        <text
          x={VBW - PAD.r + 6}
          y={model.volTop + 8}
          fontSize={9}
          fill="var(--faint)"
        >
          Vol
        </text>

        {/* volume bars: neutral, distribution days tinted, FTD accented */}
        {candles.map((c, i) => {
          const vol = c.volume ?? 0;
          if (!vol) return null;
          const isDist = distributionDates.has(c.date);
          const isFtd = ftdDates.has(c.date);
          const fill = isFtd
            ? "var(--accent-strong)"
            : isDist
              ? "var(--negative)"
              : "var(--faint)";
          return (
            <rect
              key={`v-${c.date}`}
              x={model.x(i) - model.bodyW / 2}
              y={model.vy(vol)}
              width={model.bodyW}
              height={model.volTop + VOL_H - model.vy(vol)}
              fill={fill}
              opacity={isFtd || isDist ? 0.9 : 0.45}
              rx={1}
            />
          );
        })}

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

        {/* EMA overlays */}
        {visibleSeries.map((s) => (
          <path
            key={s.period}
            d={emaPath(s)}
            fill="none"
            stroke={EMA_COLORS[s.period] ?? "var(--muted)"}
            strokeWidth={2}
            opacity={0.9}
          />
        ))}

        {/* collision-resolved line-end labels, inside the plot edge */}
        {endLabels.map((l) => (
          <g key={`el-${l.period}`}>
            <rect
              x={VBW - PAD.r - 37}
              y={l.y - 8}
              width={33}
              height={11}
              fill="var(--surface)"
              opacity={0.85}
              rx={2}
            />
            <circle
              cx={VBW - PAD.r - 31}
              cy={l.y - 2.5}
              r={2.5}
              fill={EMA_COLORS[l.period] ?? "var(--muted)"}
            />
            <text
              x={VBW - PAD.r - 26}
              y={l.y + 1}
              fontSize={9}
              fill="var(--muted)"
            >
              {l.period}d
            </text>
          </g>
        ))}

        {/* event markers above the price pane (FTD also gets a guide line) */}
        {Array.from(model.markersByIndex.entries()).map(([i, list]) => (
          <g key={`m-${i}`}>
            {list.some((m) => m.kind === "follow_through_day") && (
              <line
                x1={model.x(i)}
                x2={model.x(i)}
                y1={PAD.t}
                y2={model.volTop + VOL_H}
                stroke="var(--accent-strong)"
                strokeWidth={1}
                strokeDasharray="3 4"
                opacity={0.55}
              />
            )}
            {list.slice(0, 2).map((m, j) => (
              <MarkerGlyph
                key={m.kind}
                kind={m.kind}
                cx={model.x(i)}
                cy={PAD.t - 12 + j * 11 + 5}
              />
            ))}
          </g>
        ))}

        {/* hover capture + crosshair */}
        {candles.map((c, i) => (
          <rect
            key={`h-${c.date}`}
            x={model.x(i) - model.slot / 2}
            y={0}
            width={model.slot}
            height={VBH}
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
            y2={model.volTop + VOL_H}
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
          className="pointer-events-none absolute top-1 z-10 max-w-64 rounded-md border border-border-strong bg-surface-2 px-2 py-1.5 text-[11px] shadow-lg"
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
            {active.volume != null && (
              <>
                <span className="text-muted">Vol</span>
                <span className="text-right">
                  {Intl.NumberFormat("en-US", { notation: "compact" }).format(active.volume)}
                </span>
              </>
            )}
          </div>
          {activeMarkers.length > 0 && (
            <div className="mt-1 border-t border-border pt-1">
              {activeMarkers.map((m) => (
                <div key={m.kind} className="text-foreground">
                  {m.label}
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* legend: EMA series + marker glyph key */}
      <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-muted">
        {visibleSeries.map((s) => (
          <span key={s.period} className="inline-flex items-center gap-1.5">
            <span
              className="inline-block h-0.5 w-4 rounded"
              style={{ background: EMA_COLORS[s.period] ?? "var(--muted)" }}
            />
            {s.period}-day EMA
          </span>
        ))}
        {usedKinds.map((kind) => (
          <span key={kind} className="inline-flex items-center gap-1.5">
            <svg viewBox="0 0 12 12" className="h-3 w-3">
              <MarkerGlyph kind={kind} cx={6} cy={6} size={4.5} />
            </svg>
            {MARKER_META[kind].label}
          </span>
        ))}
      </div>
    </div>
  );
}
