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


class ScreenerFieldSpec(BaseModel):
    name: str
    label: str
    type: Literal["number", "string", "boolean", "date"]
    scopes: List[Literal["contract", "ticker"]]
    operators: List[Literal["eq", "ne", "gt", "gte", "lt", "lte"]]


class ScreenerFieldsResponse(BaseModel):
    fields: List[ScreenerFieldSpec]


class ScreenerCondition(BaseModel):
    field: str = Field(min_length=1, max_length=64)
    operator: Literal["eq", "ne", "gt", "gte", "lt", "lte"]
    value: Any


class ScreenerSort(BaseModel):
    field: str = Field(min_length=1, max_length=64)
    direction: Literal["asc", "desc"] = "desc"


class CustomScreenerRequest(BaseModel):
    scope: Literal["contract", "ticker"]
    symbols: Optional[List[str]] = Field(default=None, max_length=25)
    conditions: List[ScreenerCondition] = Field(default_factory=list, max_length=12)
    sort: Optional[ScreenerSort] = None
    limit: int = Field(default=50, ge=1, le=200)


class CustomScreenerRow(BaseModel):
    row_key: str
    symbol: str
    snapshot_date: str
    values: Dict[str, Any]


class CustomScreenerResponse(BaseModel):
    scope: Literal["contract", "ticker"]
    as_of: Optional[str] = None
    stale: bool
    unavailable_symbols: List[str]
    methodology: str
    rows: List[CustomScreenerRow]


class CalendarSourceStatus(BaseModel):
    configured: bool
    available: bool
    partial: bool = False
    unavailable: List[str] = Field(default_factory=list)
    message: Optional[str] = None


class EarningsEvent(BaseModel):
    symbol: str
    company_name: Optional[str] = None
    earnings_at: str
    session: Optional[str] = None
    eps_estimate: Optional[float] = None
    reported_eps: Optional[float] = None
    surprise_pct: Optional[float] = None
    market_cap: Optional[float] = None
    fiscal_quarter: Optional[str] = None
    estimate_count: Optional[int] = None
    url: str


class EconomicRelease(BaseModel):
    release_id: int
    release_name: str
    release_date: str
    url: str


class CalendarResponse(BaseModel):
    start_date: str
    end_date: str
    generated_at: str
    earnings: List[EarningsEvent]
    economic_releases: List[EconomicRelease]
    sources: Dict[str, CalendarSourceStatus]


class WatchlistMutation(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    symbols: List[str] = Field(min_length=1, max_length=25)


class Watchlist(BaseModel):
    id: int
    name: str
    symbols: List[str]
    created_at: str
    updated_at: str


class WatchlistListResponse(BaseModel):
    workspace: Literal["shared"] = "shared"
    available_symbols: List[str]
    snapshot_symbol_limit: int
    scheduler_enabled: bool
    tradier_configured: bool
    items: List[Watchlist]


class AlertRuleMutation(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    enabled: bool = True
    watchlist_id: int = Field(ge=1)
    query: CustomScreenerRequest
    notify_discord: bool = False
    notify_email: bool = False


class AlertRule(BaseModel):
    id: int
    name: str
    enabled: bool
    watchlist_id: int
    watchlist_name: str
    query: CustomScreenerRequest
    notify_discord: bool
    notify_email: bool
    version: int
    last_evaluated_at: Optional[str] = None
    last_error: Optional[str] = None
    created_at: str
    updated_at: str


class AlertStatus(BaseModel):
    workspace: Literal["shared"] = "shared"
    scheduler_enabled: bool
    pin_required: bool
    discord_configured: bool
    email_configured: bool
    evaluation_interval_minutes: int = 30


class AlertEvent(BaseModel):
    id: int
    alert_rule_id: int
    rule_name: str
    snapshot_date: str
    symbol: str
    scope: Literal["contract", "ticker"]
    title: str
    message: str
    payload: Dict[str, Any]
    created_at: str
    read_at: Optional[str] = None
    discord_status: str
    email_status: str


class AlertEventsResponse(BaseModel):
    unread: int
    items: List[AlertEvent]


class NewsItem(BaseModel):
    title: str
    link: str
    source: str
    date: str


class CongressTrade(BaseModel):
    chamber: Literal["house", "senate"]
    member: Optional[str] = None
    party: Optional[str] = None
    state: Optional[str] = None
    ticker: Optional[str] = None
    asset_description: Optional[str] = None
    transaction_type: str
    amount: Optional[str] = None
    owner: Optional[str] = None
    transaction_date: Optional[str] = None
    disclosure_date: Optional[str] = None


class CongressTradesResponse(BaseModel):
    """Congressional disclosures from free public mirrors.

    These are periodic transaction reports (PTRs), disclosed on a lag of up to
    45 days; ``as_of`` is the newest transaction date seen, and ``stale`` flags
    when the newest disclosure is unusually old or a source could not be read.
    """

    as_of: Optional[str] = None
    stale: bool
    unavailable_sources: List[str]
    count: int
    rows: List[CongressTrade]


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


class PricePoint(BaseModel):
    ts: str
    price: float


class DeltaProjectionBar(BaseModel):
    ts: str
    price: float
    delta_exposure: float


class DeltaProjectionResponse(BaseModel):
    """Intraday spot path with net dealer delta exposure per bar.

    The projection extrapolates the exposure trend to the 16:00 ET close
    with a linear fit; null when fewer than two bars exist.
    """

    symbol: str
    expiration: str
    offset: int
    prev_close: PricePoint
    bars: List[DeltaProjectionBar]
    projection_ts: Optional[str] = None
    projection_exposure: Optional[float] = None


class GammaGapOutcomeRow(GammaGapHistoryRow):
    outcome: Literal["hit", "miss", "pending"]
    sessions_to_hit: Optional[int] = None
    evaluated_sessions: int


class GammaGapScoreBucket(BaseModel):
    score_min: float
    score_max: float
    signals: int
    decided: int
    hits: int
    hit_rate: Optional[float] = None


class GammaGapOutcomeSummary(BaseModel):
    signals: int
    decided: int
    hits: int
    misses: int
    pending: int
    hit_rate: Optional[float] = None
    avg_sessions_to_hit: Optional[float] = None
    by_score: List[GammaGapScoreBucket]


class GammaGapOutcomesResponse(BaseModel):
    """Realized outcomes for logged gamma-gap signals.

    Hit = the magnet strike traded within the horizon of sessions AFTER the
    signal date (the signal day never counts). The hit rate covers decided
    signals only; pending signals are shown but excluded.
    """

    horizon_sessions: int
    summary: GammaGapOutcomeSummary
    rows: List[GammaGapOutcomeRow]


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


class AIPayloadRequest(BaseModel):
    symbol: str = Field(min_length=1, max_length=12, pattern=r"^[A-Za-z0-9.^-]+$")
    expirations: Optional[List[str]] = Field(default=None, max_length=8)
    offset: int = Field(default=35, ge=1, le=100)


class AIPayloadResponse(BaseModel):
    symbol: str
    expirations: List[str]
    prompt_tokens: Optional[int] = None
    payload: Dict[str, Any]
    messages: List[Dict[str, Any]]


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


class CaptureResponse(BaseModel):
    captured: int
    requested: int
    tickers: List[str]
    gamma_gap_rows: int
    intraday_archived: int = 0
    run_id: Optional[str] = None
    manifest_written: bool = False
    as_of: str


class EmaStatus(BaseModel):
    period: int
    value: Optional[float] = None
    distance_pct: Optional[float] = None
    above: Optional[bool] = None
    touched: bool = False
    new_touch: bool = False
    crossed_up: bool = False
    crossed_down: bool = False


class ScorecardRow(BaseModel):
    letter: str
    name: str
    status: Literal["met", "not_met", "borderline", "unavailable"]
    value: str
    detail: str


class ScorecardSummary(BaseModel):
    met: int
    scored: int
    total: int


class FollowThroughDay(BaseModel):
    date: str
    gain_pct: float
    volume_ratio: Optional[float] = None
    day_number: int
    quality: str
    anchor_low: Optional[float] = None
    close: Optional[float] = None


class DurabilityCheck(BaseModel):
    name: str
    passed: bool
    detail: str


class BottomDurability(BaseModel):
    """Post-follow-through confirmation checklist for the benchmark index."""

    ftd_date: str
    sessions_since: int
    distribution_since: int
    gains_held: bool
    checks: List[DurabilityCheck] = []
    assessment: Optional[str] = None


class EntryAssessment(BaseModel):
    """Whether the index is buyable here or extended past its entry.

    A confirmed uptrend is permission to buy, not an entry price; this
    separates the regime call from the chase risk.
    """

    status: Literal["buyable", "extended", "pullback_entry", "wait", "no_entry"]
    label: str
    detail: str


class IndexDirection(BaseModel):
    symbol: str
    label: str
    group: str
    domain: Optional[str] = None
    entry: Optional[EntryAssessment] = None
    state: str
    state_label: str
    state_since: Optional[str] = None
    as_of: Optional[str] = None
    last_close: Optional[float] = None
    change_pct: Optional[float] = None
    return_3m: Optional[float] = None
    drawdown_pct: Optional[float] = None
    rally_day: Optional[int] = None
    rally_day1_low: Optional[float] = None
    rally_day1_date: Optional[str] = None
    distribution_count: Optional[int] = None
    last_ftd: Optional[FollowThroughDay] = None
    durability: Optional[BottomDurability] = None
    rsi: Optional[float] = None
    rsi_zone: str = "unavailable"
    timing_label: str = ""
    emas: List[EmaStatus] = []
    scorecard: List[ScorecardRow] = []
    score: Optional[ScorecardSummary] = None
    narrative: List[str] = []


class DirectionBreadth(BaseModel):
    uptrend_count: int
    rally_count: int
    total: int
    above_ema50: int
    reading: str


class DirectionThresholds(BaseModel):
    ftd_gain_pct: float
    ftd_min_day: int
    ftd_ideal_max_day: int
    distribution_decline_pct: float
    distribution_lookback: int
    distribution_pressure_count: int
    correction_drawdown_pct: float
    ema_periods: List[int]
    ema_touch_band_pct: float
    rsi_period: int


class DirectionOverview(BaseModel):
    """O'Neil market-direction read for the tracked index universe.

    States follow the published vocabulary: confirmed uptrend, uptrend under
    pressure, rally attempt, correction. Signals persist per completed
    session; intraday reads are flagged provisional.
    """

    as_of: Optional[str] = None
    provisional: bool = False
    benchmark: str
    breadth: DirectionBreadth
    durability: Optional[BottomDurability] = None
    indices: List[IndexDirection]
    unavailable: List[str] = []
    thresholds: DirectionThresholds


class EmaSeriesModel(BaseModel):
    period: int
    values: List[Optional[float]]


class DirectionMarker(BaseModel):
    date: str
    kind: str
    label: str


class DirectionDetail(BaseModel):
    index: IndexDirection
    candles: List[Candle]
    ema_series: List[EmaSeriesModel]
    markers: List[DirectionMarker]
    provisional: bool = False


class DirectionSignal(BaseModel):
    id: int
    symbol: str
    label: str
    signal_type: str
    signal_date: str
    state: str
    title: str
    message: str
    payload: Dict[str, Any] = {}
    created_at: str


class DirectionSignalsResponse(BaseModel):
    items: List[DirectionSignal]
    count: int


class StockBreakout(BaseModel):
    date: str
    price: float
    prior_high: float
    volume_ratio: float
    sessions_ago: int
    stop_price: float


class StockEntry(BaseModel):
    """Pivot-relative entry discipline for a breakout candidate."""

    status: Literal["buyable", "extended"]
    pivot: float
    buy_limit: float
    stop_price: float
    extension_pct: float
    detail: str


class StockLeader(BaseModel):
    """One stock's CAN SLIM evaluation and buy-readiness verdict."""

    symbol: str
    name: Optional[str] = None
    readiness: Literal[
        "buy_candidate",
        "extended",
        "near_pivot",
        "wait_market",
        "not_ready",
        "insufficient_data",
    ]
    readiness_label: str
    score: ScorecardSummary
    scorecard: List[ScorecardRow]
    rs_percentile: Optional[float] = None
    off_high_pct: Optional[float] = None
    quarterly_eps_growth_pct: Optional[float] = None
    quarterly_revenue_growth_pct: Optional[float] = None
    annual_eps_growth_pct: Optional[float] = None
    roe_pct: Optional[float] = None
    institutional_pct: Optional[float] = None
    sector_symbol: Optional[str] = None
    sector_label: Optional[str] = None
    sector_rank: Optional[int] = None
    fundamentals_fetched: bool = False
    breakout: Optional[StockBreakout] = None
    entry: Optional[StockEntry] = None
    last_close: Optional[float] = None
    change_pct: Optional[float] = None
    narrative: List[str] = []
    missing: List[str] = []


class SectorOption(BaseModel):
    symbol: Optional[str] = None
    label: str


class LeadersResponse(BaseModel):
    """CAN SLIM scan over the holdings of every tracked market/sector ETF.

    Price and volume are screened for the whole universe; company
    fundamentals are fetched only for the strongest technical candidates,
    and the rest are reported as technical-only rather than guessed.
    """

    as_of: Optional[str] = None
    provisional: bool = False
    market_state: str
    market_state_label: str
    gate_open: bool
    gate_message: str
    items: List[StockLeader]
    universe: List[str] = []
    universe_size: int = 0
    scanned: int = 0
    fundamentals_scanned: int = 0
    holdings_source: Literal["provider", "configured", "mixed", "unavailable"] = (
        "unavailable"
    )
    etfs_without_holdings: List[str] = []
    dropped_for_capacity: int = 0
    sectors: List[SectorOption] = []
    excluded: List[str] = []
    unavailable: List[str] = []
    stop_loss_pct: float
    methodology: str


class DirectionOutcomeRow(BaseModel):
    id: int
    symbol: str
    label: str
    signal_type: str
    signal_date: str
    entry_price: float
    invalidation_level: float
    outcome: Literal["held", "failed", "pending"]
    sessions_to_fail: Optional[int] = None
    evaluated_sessions: int
    max_gain_pct: Optional[float] = None
    max_drawdown_pct: Optional[float] = None
    latest_gain_pct: Optional[float] = None
    stop_hit: Optional[bool] = None


class DirectionOutcomeSummary(BaseModel):
    signals: int
    decided: int
    held: int
    failed: int
    pending: int
    hold_rate: Optional[float] = None
    avg_max_drawdown_pct: Optional[float] = None
    avg_max_gain_pct: Optional[float] = None
    stop_hit_rate: Optional[float] = None


class DirectionOutcomesResponse(BaseModel):
    """Realized outcomes for logged follow-through and breakout signals.

    Failure is measured against each signal's own invalidation level; a
    signal without a full horizon of completed sessions stays pending and
    is excluded from the rates.
    """

    horizon_sessions: int
    stop_pct: float
    summary: DirectionOutcomeSummary
    rows: List[DirectionOutcomeRow]
