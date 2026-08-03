// Colored status pill for the four O'Neil market states. Status colors are
// used for status here (their reserved job); text stays on ink tokens where
// the tint alone would be ambiguous.

const STYLES: Record<string, string> = {
  confirmed_uptrend: "border-positive/40 bg-positive/10 text-positive",
  uptrend_under_pressure: "border-warning/40 bg-warning/10 text-warning",
  rally_attempt: "border-accent/40 bg-accent/10 text-accent-strong",
  correction: "border-negative/40 bg-negative/10 text-negative",
  unavailable: "border-border bg-surface-2 text-muted",
};

export default function StatePill({
  state,
  label,
  size = "sm",
}: {
  state: string;
  label: string;
  size?: "sm" | "lg";
}) {
  const style = STYLES[state] ?? STYLES.unavailable;
  const sizing =
    size === "lg"
      ? "px-3 py-1 text-sm font-semibold"
      : "px-2 py-0.5 text-[11px] font-medium";
  return (
    <span
      className={`inline-flex items-center whitespace-nowrap rounded-full border ${sizing} ${style}`}
    >
      {label}
    </span>
  );
}
