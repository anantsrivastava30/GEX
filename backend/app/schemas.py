"""Pydantic response models for the public API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

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
