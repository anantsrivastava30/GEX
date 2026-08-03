"use client";

import { usePathname } from "next/navigation";
import PageHelp, { HelpSection } from "./PageHelp";

type HelpContent = { label: string; sections: readonly HelpSection[] };

const HELP: Record<string, HelpContent> = {
  market: {
    label: "Market Overview",
    sections: [{ bullets: [
      "Use VIX for the volatility backdrop, the 10-year yield for rate context, and futures for broad direction.",
      "Read several tiles together; no single market tile is a standalone trade signal.",
      "A missing value means its upstream source is unavailable, not that the value is zero.",
    ] }],
  },
  flow: {
    label: "Flow",
    sections: [{ bullets: [
      "Each Hottest Expiration Chain row aggregates all call and put strikes for one symbol and expiry; it intentionally has no single strike.",
      "Listed contracts is the number of option rows, while Total Volume and Total OI are sums across the expiration. Click a row to inspect its strikes below.",
      "Score combines relative Vol/OI, absolute OI change, and absolute IV change. It is not a probability or customer-direction signal.",
      "This is a cached positioning proxy, not a live trade tape. Confirm the as-of and history badges before interpreting it.",
      "Use symbol, side, Vol/OI, OI-change, and IV-change filters to reduce noise.",
    ] }],
  },
  screener: {
    label: "Options Screener",
    sections: [{ bullets: [
      "High Vol/OI finds contracts trading heavily relative to resting open interest; call activity alone does not prove bullish intent.",
      "Gamma Squeeze Candidates are model candidates, not squeeze forecasts.",
      "Custom Builder applies every condition together. Use contract scope for individual options and ticker scope for aggregate positioning.",
      "All results come from persisted snapshots, so check the as-of and stale badges first.",
    ] }],
  },
  watchlists: {
    label: "Watchlists",
    sections: [{ bullets: [
      "Create focused symbol groups for alert evaluation and repeatable research.",
      "Only symbols in the scheduled snapshot universe are available.",
      "The workspace is shared; changes require the workspace PIN.",
      "Move or remove dependent alert rules before deleting a watchlist.",
    ] }],
  },
  alerts: {
    label: "Alerts",
    sections: [{ bullets: [
      "Create a watchlist, then attach contract- or ticker-level conditions to it.",
      "Rules evaluate scheduled snapshots, not live ticks. Stale or incomplete snapshots are skipped.",
      "In-app events are retained; Discord and email require server configuration.",
      "Use the workspace PIN to change rules or mark inbox events as read.",
    ] }],
  },
  news: {
    label: "Market News",
    sections: [{ bullets: [
      "Open a headline to read and verify the original publisher's article.",
      "Use source and timestamp to judge recency; ordering is not a sentiment score.",
      "Confirm market-moving claims with primary reporting before using them in a decision.",
    ] }],
  },
  "track-record": {
    label: "Gamma-Gap Track Record",
    sections: [{ bullets: [
      "A Hit means the magnet traded within the next five sessions; the signal day never counts.",
      "Pending signals are excluded from hit rate until their full evaluation window closes.",
      "Compare score buckets and use the ticker filter to test whether performance is concentrated.",
      "Repeated scans of one setup are observations, not independent trades.",
    ] }],
  },
  ai: {
    label: "AI Analysis",
    sections: [{ bullets: [
      "Choose a ticker, expirations, and strike range before building the positioning packet.",
      "Build payload only makes no OpenAI call; copy its prompt or JSON into another workflow.",
      "Review the exact request before using the PIN-protected analysis action.",
      "Treat generated narratives as research assistance and verify freshness, inputs, and risk.",
    ] }],
  },
  calendar: {
    label: "Market Calendar",
    sections: [{ bullets: [
      "Choose Past or Upcoming and a 7, 14, or 30-day window. With no ticker filter, earnings cover the full Nasdaq calendar.",
      "Add comma-separated symbols only when you want to narrow the market-wide earnings list.",
      "Date-only earnings entries show the supplied session or time TBD. Confirm timing through the Nasdaq source link.",
      "FRED supplies curated release dates, not consensus, actual, prior, impact, or exact publication time.",
      "Yahoo and FRED fail independently, so check both source-status badges.",
    ] }],
  },
  congress: {
    label: "Congress Trades",
    sections: [{ bullets: [
      "Filter by chamber, lookback window, and ticker to narrow periodic transaction reports.",
      "Disclosures can arrive up to 45 days after a transaction and are not a live feed.",
      "Amounts are reported ranges; Buy or Sell is a filing classification, not a recommendation.",
      "Check latest-date and source badges before interpreting missing or stale records.",
    ] }],
  },
  binomial: {
    label: "Binomial Tree",
    sections: [{ bullets: [
      "Choose ticker, expiration, strike, call or put, tree steps, and pricing-horizon DTE.",
      "The output is a European CRR theoretical value, not a market quote, and does not model early exercise.",
      "Market rate and contract IV are used unless you provide labeled overrides.",
      "More steps create a finer tree; the pricing horizon can intentionally differ from calendar DTE.",
    ] }],
  },
  stock: {
    label: "Ticker Research",
    sections: [
      { heading: "Overview", bullets: [
        "Start with price, quote quality, daily candles, IV rank, and the primary expiration's gamma-gap signal.",
        "IV rank and percentile use this project's recorded history and may be rank-since-inception while history accrues.",
      ] },
      { heading: "GEX", bullets: [
        "Click one expiration for a single profile. Ctrl/Cmd-click or Compare mode overlays additional expirations without summing them.",
        "The highlighted primary expiration drives max pain, vanna/charm, gamma-gap, and delta projection.",
        "Positive local GEX is a mean-reversion heuristic; negative GEX can amplify movement. Gamma-gap score is not a probability.",
      ] },
      { heading: "Volatility", bullets: [
        "Use strike skew for the primary expiration and term structure to compare ATM IV across expirations.",
        "An inverted curve means near-term IV exceeds farther IV; missing values usually mean insufficient provider data or history.",
      ] },
      { heading: "Flow", bullets: [
        "Ticker Flow is not implemented yet. Use the top-level Flow page for the current snapshot-backed proxy.",
      ] },
    ],
  },
};

function contentForPath(pathname: string): HelpContent {
  if (pathname.startsWith("/stock/")) return HELP.stock;
  if (pathname.startsWith("/tools/binomial")) return HELP.binomial;
  const key = pathname.split("/").filter(Boolean)[0] ?? "market";
  return HELP[key] ?? HELP.market;
}

export default function ContextHelp() {
  const pathname = usePathname();
  const content = contentForPath(pathname);
  return <PageHelp label={content.label} sections={content.sections} />;
}
