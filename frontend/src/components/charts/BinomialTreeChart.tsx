"use client";

import { useMemo, useState } from "react";
import { BinomialNode } from "@/lib/api";

interface Props {
  nodes: BinomialNode[];
  steps: number;
}

function money(value: number) {
  return value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

export default function BinomialTreeChart({ nodes, steps }: Props) {
  const [selected, setSelected] = useState<BinomialNode | null>(null);
  const { ordered, priceMin, priceMax, optionMin, optionMax } = useMemo(() => {
    const ordered = [...nodes].sort((a, b) => a.step - b.step || a.node - b.node);
    return {
      ordered,
      priceMin: Math.min(...ordered.map((node) => node.price)),
      priceMax: Math.max(...ordered.map((node) => node.price)),
      optionMin: Math.min(...ordered.map((node) => node.option)),
      optionMax: Math.max(...ordered.map((node) => node.option)),
    };
  }, [nodes]);

  if (!ordered.length) return null;

  const width = Math.max(680, steps * 30 + 80);
  const height = 400;
  const padding = { left: 42, right: 22, top: 22, bottom: 30 };
  const x = (step: number) => padding.left + (step / Math.max(steps, 1)) * (width - padding.left - padding.right);
  const y = (price: number) => {
    const range = priceMax - priceMin || 1;
    return padding.top + ((priceMax - price) / range) * (height - padding.top - padding.bottom);
  };
  const optionRange = optionMax - optionMin || 1;
  const optionColor = (option: number) => {
    const ratio = (option - optionMin) / optionRange;
    return `hsl(${218 - ratio * 78} 78% ${38 + ratio * 18}%)`;
  };

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-muted">
        <span><span className="mr-1 inline-block h-2 w-2 rounded-full bg-accent" />Underlying price level</span>
        <span>Node color: lower to higher option value</span>
        <span className="ml-auto font-mono">{ordered.length.toLocaleString()} nodes</span>
      </div>
      <div className="overflow-x-auto rounded border border-border bg-background p-2">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          className="min-w-[680px]"
          role="img"
          aria-label={`CRR binomial tree with ${steps} steps. Select a node to inspect its underlying and option value.`}
        >
          {Array.from({ length: steps + 1 }, (_, step) => (
            <g key={step}>
              <line x1={x(step)} x2={x(step)} y1={padding.top} y2={height - padding.bottom} stroke="var(--border)" strokeWidth="1" />
              <text x={x(step)} y={height - 9} fill="var(--muted)" fontSize="10" textAnchor="middle">{step}</text>
            </g>
          ))}
          {ordered.filter((node) => node.step > 0).map((node) => {
            const parents = ordered.filter((parent) => parent.step === node.step - 1 && (parent.node === node.node || parent.node === node.node - 1));
            return parents.map((parent) => (
              <line key={`${parent.step}-${parent.node}-${node.step}-${node.node}`} x1={x(parent.step)} y1={y(parent.price)} x2={x(node.step)} y2={y(node.price)} stroke="var(--border-strong)" strokeWidth="1" />
            ));
          })}
          {ordered.map((node) => {
            const active = selected?.step === node.step && selected.node === node.node;
            return (
              <circle
                key={`${node.step}-${node.node}`}
                cx={x(node.step)}
                cy={y(node.price)}
                r={active ? 6 : 4}
                fill={optionColor(node.option)}
                stroke={active ? "var(--foreground)" : "var(--background)"}
                strokeWidth={active ? 2 : 1}
                tabIndex={0}
                role="button"
                aria-label={`Step ${node.step}, node ${node.node}: underlying ${money(node.price)}, option ${money(node.option)}`}
                onFocus={() => setSelected(node)}
                onMouseEnter={() => setSelected(node)}
                onClick={() => setSelected(node)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" || event.key === " ") setSelected(node);
                }}
              />
            );
          })}
          <text x={10} y={padding.top + 5} fill="var(--muted)" fontSize="10">underlying</text>
          <text x={width - padding.right} y={height - 9} fill="var(--muted)" fontSize="10" textAnchor="end">step</text>
        </svg>
      </div>
      <p className="min-h-5 rounded bg-surface-2 px-3 py-2 font-mono text-xs text-muted" aria-live="polite">
        {selected
          ? `Step ${selected.step}, node ${selected.node}: underlying $${money(selected.price)} | option $${money(selected.option)}`
          : "Hover, focus, or select a node for exact values."}
      </p>
      <details className="rounded border border-border bg-surface">
        <summary className="cursor-pointer px-3 py-2 text-sm text-muted hover:text-foreground">
          Accessible node table ({ordered.length.toLocaleString()} rows)
        </summary>
        <div className="max-h-80 overflow-auto border-t border-border">
          <table className="w-full min-w-[480px] text-sm">
            <thead className="sticky top-0 bg-surface-2 text-left text-xs text-muted">
              <tr><th className="px-3 py-2">Step</th><th className="px-3 py-2">Node</th><th className="px-3 py-2 text-right">Underlying</th><th className="px-3 py-2 text-right">Option</th></tr>
            </thead>
            <tbody>
              {ordered.map((node) => (
                <tr key={`${node.step}-${node.node}`} className="border-t border-border hover:bg-surface-hover">
                  <td className="px-3 py-1.5 font-mono text-muted">{node.step}</td><td className="px-3 py-1.5 font-mono text-muted">{node.node}</td><td className="px-3 py-1.5 text-right font-mono">${money(node.price)}</td><td className="px-3 py-1.5 text-right font-mono">${money(node.option)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </div>
  );
}
