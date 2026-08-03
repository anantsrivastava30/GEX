import { ScorecardStatus } from "@/lib/api";

// Shared styling for CAN SLIM criterion statuses, used by the Direction
// index cards and the Leaders stock table.

export const SCORE_DOT: Record<ScorecardStatus, string> = {
  met: "bg-positive",
  borderline: "bg-warning",
  not_met: "bg-negative",
  unavailable: "bg-faint",
};

export const SCORE_BADGE: Record<ScorecardStatus, string> = {
  met: "border-positive/40 bg-positive/10 text-positive",
  borderline: "border-warning/40 bg-warning/10 text-warning",
  not_met: "border-negative/40 bg-negative/10 text-negative",
  unavailable: "border-border bg-surface-2 text-muted",
};

export const SCORE_LABEL: Record<ScorecardStatus, string> = {
  met: "Met",
  borderline: "Borderline",
  not_met: "Not met",
  unavailable: "No data",
};
