"""Pydantic response models for the public API."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class TickerSnapshot(BaseModel):
    symbol: str
    last: Optional[float] = None
    change: Optional[float] = None
    change_percentage: Optional[float] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    volume: Optional[int] = None
    average_volume: Optional[int] = None
    week_52_high: Optional[float] = None
    week_52_low: Optional[float] = None


class GexPoint(BaseModel):
    strike: float
    net_gex: float


class GammaGap(BaseModel):
    magnet_strike: float
    magnet_gex: float
    distance: float
    distance_pct: float
    score: float
    positive_zone: bool
    band_low: float
    band_high: float
    commentary: Optional[str] = None


class GexProfile(BaseModel):
    symbol: str
    expiration: str
    spot: float
    offset: int
    profile: List[GexPoint]
    interpretation: List[str]
    gamma_gap: Optional[GammaGap] = None


class SkewPoint(BaseModel):
    strike: float
    iv_call: Optional[float] = None
    iv_put: Optional[float] = None
    iv_skew: Optional[float] = None


class SkewResponse(BaseModel):
    symbol: str
    expiration: str
    points: List[SkewPoint]


class Candle(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: Optional[int] = None


class RatiosResponse(BaseModel):
    symbol: str
    expirations: List[str]
    pc_volume_ratio: float
    pc_oi_ratio: float


class UnusualRow(BaseModel):
    strike: float
    vol_oi_call: Optional[float] = None
    vol_oi_put: Optional[float] = None
    open_interest_call: Optional[float] = None
    open_interest_put: Optional[float] = None
    total_vol_oi: Optional[float] = None


class UnusualResponse(BaseModel):
    symbol: str
    expirations: List[str]
    rows: List[UnusualRow]


class CachedFlowRow(BaseModel):
    ticker: str
    expiration_date: str
    strike: float
    option_type: str
    snapshot_date: str
    volume: Optional[float] = None
    open_interest: Optional[float] = None
    volume_oi: Optional[float] = None
    oi_change: Optional[float] = None
    mid_iv: Optional[float] = None
    iv_change: Optional[float] = None
    history_available: bool
    score: float


class CachedFlowResponse(BaseModel):
    """Cached contract-anomaly proxy, not a trade tape or directional feed."""

    as_of: Optional[str] = None
    stale: bool
    unavailable_history: bool
    unavailable_tickers: List[str]
    unavailable_history_tickers: List[str]
    rows: List[CachedFlowRow]


class HottestChainRow(BaseModel):
    ticker: str
    expiration_date: str
    snapshot_date: str
    contracts: int
    total_volume: float
    total_open_interest: float
    volume_oi: Optional[float] = None
    oi_change: Optional[float] = None
    iv_change: Optional[float] = None
    history_available: bool
    score: float


class HottestChainsResponse(BaseModel):
    """Cached chain rankings, not a representation of option trade activity."""

    as_of: Optional[str] = None
    stale: bool
    unavailable_history: bool
    unavailable_tickers: List[str]
    unavailable_history_tickers: List[str]
    rows: List[HottestChainRow]


class ScreenerRow(BaseModel):
    """A candidate derived from the latest persisted option snapshots."""

    symbol: str
    snapshot_date: str
    expiration_date: Optional[str] = None
    strike: Optional[float] = None
    option_type: Optional[Literal["call", "put"]] = None
    volume: Optional[float] = None
    open_interest: Optional[float] = None
    volume_oi: Optional[float] = None
    spot: Optional[float] = None
    net_gex_total: Optional[float] = None
    gamma_magnet_strike: Optional[float] = None
    gamma_gap_distance: Optional[float] = None
    gamma_gap_score: Optional[float] = None
    gamma_positive_zone: Optional[bool] = None


class ScreenerResponse(BaseModel):
    """Snapshot-only screener output, not a live options-flow feed."""

    preset: Literal["high_vol_oi", "unusually_bullish", "gamma_squeeze"]
    as_of: Optional[str] = None
    stale: bool
    unavailable_symbols: List[str]
    methodology: str
    rows: List[ScreenerRow]


class NewsItem(BaseModel):
    title: str
    link: str
    source: str
    date: str


class IVRankResponse(BaseModel):
    symbol: str
    iv: float
    iv_rank: float
    iv_percentile: float
    iv_low: float
    iv_high: float
    days_of_history: int


class OIChangeRow(BaseModel):
    expiration_date: str
    strike: float
    option_type: str
    open_interest: float
    open_interest_prev: float
    oi_change: float
    from_date: str
    to_date: str


class MarketOverview(BaseModel):
    vix: Optional[Dict[str, Any]] = None
    yields: Optional[Dict[str, Any]] = None
    futures: Dict[str, Dict[str, Any]] = {}


class ExposurePoint(BaseModel):
    strike: float
    vanna: float
    charm: float


class ExposureResponse(BaseModel):
    symbol: str
    expirations: List[str]
    spot: float
    offset: int
    risk_free_rate: float
    points: List[ExposurePoint]


class MaxPainPoint(BaseModel):
    strike: float
    call_pain: float
    put_pain: float
    total_pain: float


class MaxPainResponse(BaseModel):
    symbol: str
    expiration: str
    spot: float
    max_pain: float
    curve: List[MaxPainPoint]


class TermStructurePoint(BaseModel):
    expiration: str
    dte: Optional[int] = None
    atm_iv: float


class TermStructureResponse(BaseModel):
    symbol: str
    spot: float
    points: List[TermStructurePoint]


class BinomialCalibration(BaseModel):
    risk_free_rate: float
    risk_free_rate_source: Literal["market", "override"]
    implied_volatility: float
    implied_volatility_source: Literal["market", "override"]


class BinomialNode(BaseModel):
    step: int
    node: int
    price: float
    option: float


class BinomialTreeResponse(BaseModel):
    symbol: str
    expiration: str
    spot: float
    strike: float
    option_type: Literal["call", "put"]
    steps: int
    days_to_exp: int
    calibration: BinomialCalibration
    nodes: List[BinomialNode]


class GammaGapHistoryRow(BaseModel):
    ts: str
    ticker: str
    expiration: Optional[str] = None
    dte: Optional[int] = None
    spot: Optional[float] = None
    magnet_strike: Optional[float] = None
    magnet_gex: Optional[float] = None
    distance: Optional[float] = None
    score: Optional[float] = None
    positive_zone: Optional[int] = None


class AIAnalyzeRequest(BaseModel):
    symbol: str = Field(min_length=1, max_length=12, pattern=r"^[A-Za-z0-9.^-]+$")
    expirations: Optional[List[str]] = Field(default=None, max_length=8)
    model: Optional[str] = Field(default=None, max_length=100)
    offset: int = Field(default=35, ge=1, le=100)
    pin: Optional[str] = Field(default=None, max_length=128)


class AIAnalyzeResponse(BaseModel):
    symbol: str
    model: str
    response: str
    prompt_tokens: Optional[int] = None
    payload: Dict[str, Any]


class AIStatus(BaseModel):
    openai_configured: bool
    pin_required: bool
    default_model: str


class AIHistoryItem(BaseModel):
    ts: Optional[str] = None
    ticker: Optional[str] = None
    expirations: Optional[Any] = None
    response: Optional[str] = None
    token_count: Optional[Any] = None
