import { ReactNode } from "react";

// KPI tile: label, big value, optional signed delta. Deltas carry an arrow
// and sign so direction never rests on color alone.

function DeltaBadge({ value, suffix = "%" }: { value: number; suffix?: string }) {
  const up = value >= 0;
  return (
    <span
      className={`inline-flex items-center gap-0.5 font-mono text-xs ${
        up ? "text-positive" : "text-negative"
      }`}
    >
      <span aria-hidden>{up ? "▲" : "▼"}</span>
      {up ? "+" : ""}
      {value.toFixed(2)}
      {suffix}
    </span>
  );
}

export default function StatTile({
  label,
  value,
  delta,
  deltaSuffix,
  sub,
  accent = false,
}: {
  label: string;
  value: ReactNode;
  delta?: number | null;
  deltaSuffix?: string;
  sub?: ReactNode;
  accent?: boolean;
}) {
  return (
    <div
      className={`rounded-lg border bg-surface px-4 py-3 ${
        accent ? "border-accent/40" : "border-border"
      }`}
    >
      <p className="text-[11px] uppercase tracking-wide text-muted">{label}</p>
      <div className="mt-1.5 flex items-baseline gap-2">
        <span className="font-mono text-xl leading-none text-foreground">
          {value}
        </span>
        {delta != null && <DeltaBadge value={delta} suffix={deltaSuffix} />}
      </div>
      {sub && <p className="mt-1 text-xs text-muted">{sub}</p>}
    </div>
  );
}
