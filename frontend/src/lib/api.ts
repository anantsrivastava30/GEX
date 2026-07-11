// Typed client for the GEX FastAPI backend.
//
// All requests go to /api/* which Next.js rewrites to BACKEND_URL
// (see next.config.ts), so the browser only ever talks to this origin.

export interface TickerSnapshot {
  symbol: string;
  last?: number | null;
  change?: number | null;
  change_percentage?: number | null;
  bid?: number | null;
  ask?: number | null;
  volume?: number | null;
  average_volume?: number | null;
  week_52_high?: number | null;
  week_52_low?: number | null;
}

export interface GexPoint {
  strike: number;
  net_gex: number;
}

export interface GammaGap {
  magnet_strike: number;
  magnet_gex: number;
  distance: number;
  distance_pct: number;
  score: number;
  positive_zone: boolean;
  band_low: number;
  band_high: number;
  commentary?: string | null;
}

export interface GexProfile {
  symbol: string;
  expiration: string;
  spot: number;
  offset: number;
  profile: GexPoint[];
  interpretation: string[];
  gamma_gap?: GammaGap | null;
}

export interface Candle {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number | null;
}

export interface SkewPoint {
  strike: number;
  iv_call?: number | null;
  iv_put?: number | null;
  iv_skew?: number | null;
}

export interface SkewResponse {
  symbol: string;
  expiration: string;
  points: SkewPoint[];
}

export interface RatiosResponse {
  symbol: string;
  expirations: string[];
  pc_volume_ratio: number;
  pc_oi_ratio: number;
}

export interface UnusualRow {
  strike: number;
  vol_oi_call?: number | null;
  vol_oi_put?: number | null;
  open_interest_call?: number | null;
  open_interest_put?: number | null;
  total_vol_oi?: number | null;
}

export interface UnusualResponse {
  symbol: string;
  expirations: string[];
  rows: UnusualRow[];
}

export interface NewsItem {
  title: string;
  link: string;
  source: string;
  date: string;
}

export interface IVRankResponse {
  symbol: string;
  iv: number;
  iv_rank: number;
  iv_percentile: number;
  iv_low: number;
  iv_high: number;
  days_of_history: number;
}

export interface MarketOverview {
  vix?: Record<string, number | null> | null;
  yields?: Record<string, number | null> | null;
  futures: Record<string, Record<string, number | null>>;
}

export interface ExposurePoint {
  strike: number;
  vanna: number;
  charm: number;
}

export interface ExposureResponse {
  symbol: string;
  expirations: string[];
  spot: number;
  offset: number;
  risk_free_rate: number;
  points: ExposurePoint[];
}

export interface MaxPainPoint {
  strike: number;
  call_pain: number;
  put_pain: number;
  total_pain: number;
}

export interface MaxPainResponse {
  symbol: string;
  expiration: string;
  spot: number;
  max_pain: number;
  curve: MaxPainPoint[];
}

export interface TermStructurePoint {
  expiration: string;
  dte?: number | null;
  atm_iv: number;
}

export interface TermStructureResponse {
  symbol: string;
  spot: number;
  points: TermStructurePoint[];
}

export class APIError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "APIError";
  }
}

async function getJSON<T>(path: string, signal?: AbortSignal): Promise<T> {
  const resp = await fetch(path, {
    headers: { accept: "application/json" },
    signal,
  });
  if (!resp.ok) {
    let detail = `Backend returned ${resp.status}`;
    try {
      const body = await resp.json();
      if (body?.detail) detail = String(body.detail);
    } catch {
      // non-JSON error body; keep the status message
    }
    throw new APIError(detail, resp.status);
  }
  return resp.json() as Promise<T>;
}

function qs(params: Record<string, string | number | string[]>): string {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (Array.isArray(value)) value.forEach((v) => search.append(key, v));
    else search.append(key, String(value));
  }
  const s = search.toString();
  return s ? `?${s}` : "";
}

export const api = {
  tickerSnapshot: (symbol: string, signal?: AbortSignal) =>
    getJSON<TickerSnapshot>(`/api/ticker/${symbol}/snapshot`, signal),

  expirations: (symbol: string, signal?: AbortSignal) =>
    getJSON<string[]>(`/api/ticker/${symbol}/expirations`, signal),

  gexProfile: (
    symbol: string,
    expiration: string,
    offset = 35,
    signal?: AbortSignal,
  ) =>
    getJSON<GexProfile>(
      `/api/ticker/${symbol}/gex${qs({ expiration, offset })}`,
      signal,
    ),

  candles: (symbol: string, days = 120, signal?: AbortSignal) =>
    getJSON<Candle[]>(`/api/ticker/${symbol}/candles${qs({ days })}`, signal),

  skew: (symbol: string, expiration: string, signal?: AbortSignal) =>
    getJSON<SkewResponse>(
      `/api/ticker/${symbol}/skew${qs({ expiration })}`,
      signal,
    ),

  ratios: (symbol: string, expirations: string[]) =>
    getJSON<RatiosResponse>(`/api/ticker/${symbol}/ratios${qs({ expirations })}`),

  unusual: (
    symbol: string,
    expirations: string[],
    topN = 15,
    signal?: AbortSignal,
  ) =>
    getJSON<UnusualResponse>(
      `/api/flow/unusual${qs({ symbol, expirations, top_n: topN })}`,
      signal,
    ),

  marketOverview: () => getJSON<MarketOverview>(`/api/market/overview`),

  news: () => getJSON<NewsItem[]>(`/api/news`),

  ivRank: (symbol: string, signal?: AbortSignal) =>
    getJSON<IVRankResponse>(`/api/history/${symbol}/iv-rank`, signal),

  exposure: (
    symbol: string,
    expirations: string[],
    offset = 35,
    signal?: AbortSignal,
  ) =>
    getJSON<ExposureResponse>(
      `/api/ticker/${symbol}/exposure${qs({ expirations, offset })}`,
      signal,
    ),

  maxPain: (symbol: string, expiration: string, signal?: AbortSignal) =>
    getJSON<MaxPainResponse>(
      `/api/ticker/${symbol}/max-pain${qs({ expiration })}`,
      signal,
    ),

  termStructure: (symbol: string, expirations = 6, signal?: AbortSignal) =>
    getJSON<TermStructureResponse>(
      `/api/ticker/${symbol}/term-structure${qs({ expirations })}`,
      signal,
    ),
};
