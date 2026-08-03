import { EntryStatus } from "@/lib/api";

// Separates "the trend is up" from "this is a buyable price". The extended
// state is the one that matters most: a confirmed uptrend plus a chased
// entry is how a correct signal turns into a stopped-out trade.

const STYLES: Record<EntryStatus, string> = {
  buyable: "border-positive/40 bg-positive/10 text-positive",
  pullback_entry: "border-accent/40 bg-accent/10 text-accent-strong",
  extended: "border-warning/40 bg-warning/10 text-warning",
  wait: "border-border-strong bg-surface-2 text-muted",
  no_entry: "border-negative/40 bg-negative/10 text-negative",
};

export default function EntryChip({
  status,
  label,
  detail,
  className = "",
}: {
  status: EntryStatus;
  label: string;
  detail?: string;
  className?: string;
}) {
  return (
    <span
      title={detail}
      className={`inline-flex items-center whitespace-nowrap rounded-full border px-2 py-0.5 text-[11px] ${STYLES[status]} ${className}`}
    >
      {label}
    </span>
  );
}
