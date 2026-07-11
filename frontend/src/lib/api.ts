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

export interface CachedFlowRow {
  ticker: string;
  expiration_date: string;
  strike: number;
  option_type: string;
  snapshot_date: string;
  volume?: number | null;
  open_interest?: number | null;
  volume_oi?: number | null;
  oi_change?: number | null;
  mid_iv?: number | null;
  iv_change?: number | null;
  history_available: boolean;
  score: number;
}

export interface CachedFlowResponse {
  as_of?: string | null;
  stale: boolean;
  unavailable_history: boolean;
  unavailable_tickers: string[];
  unavailable_history_tickers: string[];
  rows: CachedFlowRow[];
}

export interface HottestChainRow {
  ticker: string;
  expiration_date: string;
  snapshot_date: string;
  contracts: number;
  total_volume: number;
  total_open_interest: number;
  volume_oi?: number | null;
  oi_change?: number | null;
  iv_change?: number | null;
  history_available: boolean;
  score: number;
}

export interface HottestChainsResponse {
  as_of?: string | null;
  stale: boolean;
  unavailable_history: boolean;
  unavailable_tickers: string[];
  unavailable_history_tickers: string[];
  rows: HottestChainRow[];
}

export interface DeltaProjectionBar {
  ts: string;
  price: number;
  delta_exposure: number;
}

export interface DeltaProjectionResponse {
  symbol: string;
  expiration: string;
  offset: number;
  prev_close: { ts: string; price: number };
  bars: DeltaProjectionBar[];
  projection_ts?: string | null;
  projection_exposure?: number | null;
}

export type GammaGapOutcome = "hit" | "miss" | "pending";

export interface GammaGapOutcomeRow extends GammaGapHistoryRow {
  outcome: GammaGapOutcome;
  sessions_to_hit?: number | null;
  evaluated_sessions: number;
}

export interface GammaGapScoreBucket {
  score_min: number;
  score_max: number;
  signals: number;
  decided: number;
  hits: number;
  hit_rate?: number | null;
}

export interface GammaGapOutcomeSummary {
  signals: number;
  decided: number;
  hits: number;
  misses: number;
  pending: number;
  hit_rate?: number | null;
  avg_sessions_to_hit?: number | null;
  by_score: GammaGapScoreBucket[];
}

export interface GammaGapOutcomesResponse {
  horizon_sessions: number;
  summary: GammaGapOutcomeSummary;
  rows: GammaGapOutcomeRow[];
}

export type ScreenerPreset = "high_vol_oi" | "unusually_bullish" | "gamma_squeeze";

export interface ScreenerRow {
  symbol: string;
  snapshot_date: string;
  expiration_date?: string | null;
  strike?: number | null;
  option_type?: "call" | "put" | null;
  volume?: number | null;
  open_interest?: number | null;
  volume_oi?: number | null;
  spot?: number | null;
  net_gex_total?: number | null;
  gamma_magnet_strike?: number | null;
  gamma_gap_distance?: number | null;
  gamma_gap_score?: number | null;
  gamma_positive_zone?: boolean | null;
}

export interface ScreenerResponse {
  preset: ScreenerPreset;
  as_of?: string | null;
  stale: boolean;
  unavailable_symbols: string[];
  methodology: string;
  rows: ScreenerRow[];
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

export interface DailyMetricsPoint {
  date: string;
  ticker: string;
  spot?: number | null;
  net_gex_total?: number | null;
  gamma_gap_score?: number | null;
  gamma_magnet_strike?: number | null;
  gamma_gap_distance?: number | null;
}

export interface GammaGapHistoryRow {
  ts: string;
  ticker: string;
  expiration?: string | null;
  dte?: number | null;
  spot?: number | null;
  magnet_strike?: number | null;
  magnet_gex?: number | null;
  distance?: number | null;
  score?: number | null;
  positive_zone?: number | null;
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

export type BinomialOptionType = "call" | "put";
export type BinomialCalibrationSource = "market" | "override";

export interface BinomialCalibration {
  risk_free_rate: number;
  risk_free_rate_source: BinomialCalibrationSource;
  implied_volatility: number;
  implied_volatility_source: BinomialCalibrationSource;
}

export interface BinomialNode {
  step: number;
  node: number;
  price: number;
  option: number;
}

export interface BinomialTreeResponse {
  symbol: string;
  expiration: string;
  spot: number;
  strike: number;
  option_type: BinomialOptionType;
  steps: number;
  days_to_exp: number;
  calibration: BinomialCalibration;
  nodes: BinomialNode[];
}

export interface BinomialTreeRequest {
  expiration: string;
  strike: number;
  optionType: BinomialOptionType;
  steps: number;
  daysToExp: number;
  rateOverride?: number;
  ivOverride?: number;
}

export interface AIStatus {
  openai_configured: boolean;
  pin_required: boolean;
  default_model: string;
}

export interface AIAnalyzeRequest {
  symbol: string;
  expirations?: string[];
  model?: string;
  offset: number;
  pin?: string;
}

export interface AIAnalyzeResponse {
  symbol: string;
  model: string;
  response: string;
  prompt_tokens?: number | null;
  payload: Record<string, unknown>;
}

export interface AIHistoryItem {
  ts?: string | null;
  ticker?: string | null;
  expirations?: unknown;
  response?: string | null;
  token_count?: unknown;
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

async function getJSON<T>(
  path: string,
  signal?: AbortSignal,
  headers?: HeadersInit,
): Promise<T> {
  const resp = await fetch(path, {
    headers: { accept: "application/json", ...headers },
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

async function postJSON<T>(path: string, body: unknown): Promise<T> {
  const resp = await fetch(path, {
    method: "POST",
    headers: {
      accept: "application/json",
      "content-type": "application/json",
    },
    body: JSON.stringify(body),
  });
  if (!resp.ok) {
    let detail = `Backend returned ${resp.status}`;
    try {
      const errorBody = await resp.json();
      if (errorBody?.detail) detail = String(errorBody.detail);
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

  flowFeed: (limit = 500, signal?: AbortSignal) =>
    getJSON<CachedFlowResponse>(`/api/flow/feed${qs({ limit })}`, signal),

  hottestChains: (limit = 200, signal?: AbortSignal) =>
    getJSON<HottestChainsResponse>(
      `/api/flow/hottest-chains${qs({ limit })}`,
      signal,
    ),

  screener: (
    preset: ScreenerPreset,
    filters: { minVolOi?: number; minOpenInterest?: number; limit?: number } = {},
    signal?: AbortSignal,
  ) => {
    const params: Record<string, string | number> = { preset };
    if (filters.minVolOi != null) params.min_vol_oi = filters.minVolOi;
    if (filters.minOpenInterest != null)
      params.min_open_interest = filters.minOpenInterest;
    if (filters.limit != null) params.limit = filters.limit;
    return getJSON<ScreenerResponse>(`/api/screener${qs(params)}`, signal);
  },

  marketOverview: () => getJSON<MarketOverview>(`/api/market/overview`),

  news: () => getJSON<NewsItem[]>(`/api/news`),

  ivRank: (symbol: string, signal?: AbortSignal) =>
    getJSON<IVRankResponse>(`/api/history/${symbol}/iv-rank`, signal),

  dailyMetrics: (symbol: string, signal?: AbortSignal) =>
    getJSON<DailyMetricsPoint[]>(`/api/history/metrics${qs({ symbol })}`, signal),

  gammaGapHistory: (ticker?: string, limit = 250, signal?: AbortSignal) =>
    getJSON<GammaGapHistoryRow[]>(
      `/api/history/gamma-gap${qs(ticker ? { ticker, limit } : { limit })}`,
      signal,
    ),

  gammaGapOutcomes: (
    ticker?: string,
    horizon = 5,
    limit = 250,
    signal?: AbortSignal,
  ) =>
    getJSON<GammaGapOutcomesResponse>(
      `/api/history/gamma-gap/outcomes${qs(
        ticker ? { ticker, horizon, limit } : { horizon, limit },
      )}`,
      signal,
    ),

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

  deltaProjection: (
    symbol: string,
    expiration: string,
    offset = 35,
    signal?: AbortSignal,
  ) =>
    getJSON<DeltaProjectionResponse>(
      `/api/ticker/${symbol}/delta-projection${qs({ expiration, offset })}`,
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

  binomialTree: (
    symbol: string,
    request: BinomialTreeRequest,
    signal?: AbortSignal,
  ) =>
    getJSON<BinomialTreeResponse>(
      `/api/ticker/${symbol}/binomial-tree${qs({
        expiration: request.expiration,
        strike: request.strike,
        option_type: request.optionType,
        steps: request.steps,
        days_to_exp: request.daysToExp,
        ...(request.rateOverride == null ? {} : { rate_override: request.rateOverride }),
        ...(request.ivOverride == null ? {} : { iv_override: request.ivOverride }),
      })}`,
      signal,
    ),

  aiStatus: (signal?: AbortSignal) => getJSON<AIStatus>("/api/ai/status", signal),

  aiAnalyze: (request: AIAnalyzeRequest) =>
    postJSON<AIAnalyzeResponse>("/api/ai/analyze", request),

  aiHistory: (pin: string, limit = 15, signal?: AbortSignal) =>
    getJSON<AIHistoryItem[]>(`/api/ai/analyses${qs({ limit })}`, signal, {
      "X-AI-PIN": pin,
    }),
};
