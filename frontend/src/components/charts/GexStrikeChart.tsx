"use client";

import { GexPoint } from "@/lib/api";
import StrikeBarChart from "./StrikeBarChart";

// Net GEX by strike: thin wrapper over the generic diverging strike chart.

interface Props {
  profile: GexPoint[];
  spot: number;
}

export default function GexStrikeChart({ profile, spot }: Props) {
  return (
    <StrikeBarChart
      points={profile.map((p) => ({ strike: p.strike, value: p.net_gex }))}
      spot={spot}
      positiveLabel="Long gamma (+)"
      negativeLabel="Short gamma (−)"
    />
  );
}
