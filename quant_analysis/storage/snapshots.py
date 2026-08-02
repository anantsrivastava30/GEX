"""Daily market-history snapshots.

Captures option-chain state (OI, volume, IV, greeks) and derived positioning
metrics into git-versioned, append-only files so history-dependent features
(IV rank, OI change, historical GEX, gamma-gap track record) run off our own
accumulated dataset instead of paid history APIs.

Storage layout under ``data/snapshots/`` (configurable via ``config.yaml``):

    data/snapshots/
    ├── YYYY-MM-DD/
    │   └── {TICKER}.csv.gz     # per-contract chain snapshot for that day
    └── daily_metrics.csv       # one row per (date, ticker): derived metrics

Re-running on the same day overwrites that day's files (upsert semantics),
so an intraday re-run simply refreshes the day's snapshot.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import pandas as pd

from quant_analysis.config import CONFIG, PROJECT_ROOT

logger = logging.getLogger(__name__)

SNAPSHOT_CONFIG = CONFIG.get("snapshots", {})
DEFAULT_BASE_DIR = PROJECT_ROOT / SNAPSHOT_CONFIG.get("dir", "data/snapshots")
DEFAULT_TICKERS = SNAPSHOT_CONFIG.get("tickers", ["SPY", "QQQ"])
DEFAULT_NUM_EXPIRATIONS = int(SNAPSHOT_CONFIG.get("num_expirations", 4))
DEFAULT_STRIKE_OFFSET = int(SNAPSHOT_CONFIG.get("strike_offset", 35))

MARKET_TZ = ZoneInfo("America/New_York")

METRICS_FILE = "daily_metrics.csv"

# Per-contract columns persisted when present in the chain DataFrame.
CONTRACT_COLUMNS = [
    "expiration_date",
    "strike",
    "option_type",
    "open_interest",
    "volume",
    "last",
    "bid",
    "ask",
    "contract_size",
    "delta",
    "gamma",
    "theta",
    "vega",
    "mid_iv",
    "smv_vol",
    "DTE",
    "GammaExposure",
    "DeltaExposure",
]

# Keys shared by contract files and the join logic in compute_oi_change.
CONTRACT_KEY = ["expiration_date", "strike", "option_type"]


def _observed(day: date) -> date:
    if day.weekday() == 5:
        return day - timedelta(days=1)
    if day.weekday() == 6:
        return day + timedelta(days=1)
    return day


def _nth_weekday(year: int, month: int, weekday: int, occurrence: int) -> date:
    day = date(year, month, 1)
    day += timedelta(days=(weekday - day.weekday()) % 7)
    return day + timedelta(weeks=occurrence - 1)


def _last_weekday(year: int, month: int, weekday: int) -> date:
    if month == 12:
        day = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        day = date(year, month + 1, 1) - timedelta(days=1)
    return day - timedelta(days=(day.weekday() - weekday) % 7)


def _easter(year: int) -> date:
    """Gregorian Easter date (Anonymous Gregorian algorithm)."""

    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    month_seed = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * month_seed) // 451
    month = (h + month_seed - 7 * m + 114) // 31
    day = (h + month_seed - 7 * m + 114) % 31 + 1
    return date(year, month, day)


def _market_holidays(year: int) -> set[date]:
    holidays = {
        _observed(date(year, 1, 1)),
        _nth_weekday(year, 1, 0, 3),  # Martin Luther King Jr. Day
        _nth_weekday(year, 2, 0, 3),  # Presidents Day
        _easter(year) - timedelta(days=2),  # Good Friday
        _last_weekday(year, 5, 0),  # Memorial Day
        _observed(date(year, 7, 4)),
        _nth_weekday(year, 9, 0, 1),  # Labor Day
        _nth_weekday(year, 11, 3, 4),  # Thanksgiving
        _observed(date(year, 12, 25)),
        # A Saturday New Year can be observed on Dec 31 of this year.
        _observed(date(year + 1, 1, 1)),
    }
    if year >= 2022:
        holidays.add(_observed(date(year, 6, 19)))
    return holidays


def is_market_session(day: date) -> bool:
    return day.weekday() < 5 and day not in _market_holidays(day.year)


def previous_market_session(day: date) -> date:
    candidate = day - timedelta(days=1)
    while not is_market_session(candidate):
        candidate -= timedelta(days=1)
    return candidate


def are_consecutive_market_sessions(from_date: str, to_date: str) -> bool:
    try:
        previous = date.fromisoformat(from_date)
        current = date.fromisoformat(to_date)
    except ValueError:
        return False
    return previous_market_session(current) == previous


def capture_session_date(now: Optional[datetime] = None) -> Optional[str]:
    """Current session key, or None when an off-session capture is unsafe."""

    now = now or datetime.now(tz=MARKET_TZ)
    if now.tzinfo is None:
        now = now.replace(tzinfo=MARKET_TZ)
    else:
        now = now.astimezone(MARKET_TZ)
    if not is_market_session(now.date()) or now.time() < time(9, 30):
        return None
    return now.date().isoformat()


def market_date_today() -> str:
    """Current date in US market time (snapshots are keyed by this)."""

    return datetime.now(tz=MARKET_TZ).date().isoformat()


def last_trading_date() -> str:
    """Most recent started weekday session in US market time.

    Premarket and weekend captures key to the previous weekday so they cannot
    overwrite or falsely advertise a session that has not started. Staleness
    checks use the same rule.
    Standard full-day US equity-market holidays are skipped. Exceptional
    closures are not modelled.
    """

    now = datetime.now(tz=MARKET_TZ)
    day = now.date()
    if day.weekday() < 5 and now.time() < time(9, 30):
        day -= timedelta(days=1)
    while not is_market_session(day):
        day -= timedelta(days=1)
    return day.isoformat()


def build_snapshot(
    ticker: str,
    snapshot_date: str,
    spot: Optional[float],
    contracts: pd.DataFrame,
    offset: int = DEFAULT_STRIKE_OFFSET,
) -> Optional[Dict[str, Any]]:
    """Derive the persisted snapshot from an option-chain DataFrame.

    ``contracts`` is the output of ``market_data.load_options_data`` (greeks
    flattened, ``GammaExposure``/``DeltaExposure`` computed, puts negated).
    Pure function — no network or file IO — so it is directly testable.
    """

    if contracts is None or contracts.empty:
        return None

    from quant_analysis.analytics.visualization import compute_gamma_gap_metrics
    from quant_analysis.services.market_data import compute_put_call_ratios

    df = contracts.copy()
    # Normalise to plain YYYY-MM-DD so contract keys join across days
    # regardless of the timestamp precision the loader produced.
    df["expiration_date"] = pd.to_datetime(df["expiration_date"]).dt.date.astype(str)
    keep = [c for c in CONTRACT_COLUMNS if c in df.columns]
    slim = df[keep].copy()

    calls = df[df["option_type"] == "call"]
    puts = df[df["option_type"] == "put"]
    vol_ratio, oi_ratio = compute_put_call_ratios(df)

    metrics: Dict[str, Any] = {
        "date": snapshot_date,
        "ticker": ticker,
        "spot": spot,
        "num_contracts": int(len(df)),
        "expirations": "|".join(
            sorted(pd.to_datetime(df["expiration_date"]).dt.date.astype(str).unique())
        ),
        "call_volume": int(calls["volume"].fillna(0).sum()),
        "put_volume": int(puts["volume"].fillna(0).sum()),
        "call_oi": int(calls["open_interest"].fillna(0).sum()),
        "put_oi": int(puts["open_interest"].fillna(0).sum()),
        "pc_volume_ratio": float(vol_ratio),
        "pc_oi_ratio": float(oi_ratio),
        "atm_iv": _atm_iv(df, spot),
    }

    if spot is not None and "GammaExposure" in df.columns:
        near = df[(df["strike"] >= spot - offset) & (df["strike"] <= spot + offset)]
        metrics["net_gex_total"] = float(near["GammaExposure"].sum())
        df_net = (
            near.groupby("strike")["GammaExposure"]
            .sum()
            .reset_index()
            .rename(columns={"strike": "Strike", "GammaExposure": "GEX"})
        )
        gap = compute_gamma_gap_metrics(df_net, spot, offset=offset)
        if gap:
            metrics.update(
                {
                    "gamma_magnet_strike": gap["magnet_strike"],
                    "gamma_magnet_gex": gap["magnet_gex"],
                    "gamma_gap_distance": gap["distance"],
                    "gamma_gap_score": gap["score"],
                    "gamma_positive_zone": int(gap["positive_zone"]),
                }
            )

    return {
        "ticker": ticker,
        "date": snapshot_date,
        "spot": spot,
        "contracts": slim,
        "metrics": metrics,
    }


def _atm_iv(df: pd.DataFrame, spot: Optional[float]) -> Optional[float]:
    """Mean implied vol at the strike nearest spot on the nearest expiration."""

    if spot is None or df.empty:
        return None

    iv = df.get("mid_iv")
    if iv is None or iv.fillna(0).eq(0).all():
        iv = df.get("smv_vol")
    if iv is None:
        return None

    work = df.assign(_iv=iv).dropna(subset=["_iv"])
    work = work[work["_iv"] > 0]
    if work.empty:
        return None

    nearest_exp = pd.to_datetime(work["expiration_date"]).min()
    work = work[pd.to_datetime(work["expiration_date"]) == nearest_exp]
    nearest_strike = work.loc[(work["strike"] - spot).abs().idxmin(), "strike"]
    return float(work[work["strike"] == nearest_strike]["_iv"].mean())


def capture_ticker_snapshot(
    ticker: str,
    token: str,
    num_expirations: int = DEFAULT_NUM_EXPIRATIONS,
    offset: int = DEFAULT_STRIKE_OFFSET,
) -> Optional[Dict[str, Any]]:
    """Fetch live chain data from Tradier and derive a snapshot."""

    from quant_analysis.services.market_data import (
        get_expirations,
        get_stock_quote,
        load_options_data,
    )

    snapshot_date = capture_session_date()
    if snapshot_date is None:
        logger.warning("Off-session snapshot capture skipped for %s", ticker)
        return None

    spot = get_stock_quote(ticker, token)
    expirations = get_expirations(ticker, token)[:num_expirations]
    if not expirations:
        logger.warning("No expirations returned for %s; skipping", ticker)
        return None

    df = load_options_data(ticker, expirations, token, require_all=True)
    return build_snapshot(ticker, snapshot_date, spot, df, offset=offset)


def write_snapshot(snapshot: Dict[str, Any], base_dir: Path = DEFAULT_BASE_DIR) -> Path:
    """Persist one ticker snapshot: contract file + daily-metrics upsert."""

    base_dir = Path(base_dir)
    day_dir = base_dir / snapshot["date"]
    day_dir.mkdir(parents=True, exist_ok=True)

    path = day_dir / f"{snapshot['ticker'].upper()}.csv.gz"
    snapshot["contracts"].to_csv(path, index=False, compression="gzip")

    upsert_daily_metrics([snapshot["metrics"]], base_dir)
    return path


def upsert_daily_metrics(rows: List[Dict[str, Any]], base_dir: Path = DEFAULT_BASE_DIR) -> None:
    """Append metric rows, replacing existing rows for the same (date, ticker)."""

    if not rows:
        return

    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    path = base_dir / METRICS_FILE

    new = pd.DataFrame(rows)
    if path.exists():
        existing = pd.read_csv(path)
        keys = set(zip(new["date"], new["ticker"]))
        mask = [
            (d, t) not in keys for d, t in zip(existing["date"], existing["ticker"])
        ]
        combined = pd.concat([existing[mask], new], ignore_index=True)
    else:
        combined = new

    combined = combined.sort_values(["date", "ticker"]).reset_index(drop=True)
    combined.to_csv(path, index=False)


def load_daily_metrics(
    base_dir: Path = DEFAULT_BASE_DIR, ticker: Optional[str] = None
) -> pd.DataFrame:
    path = Path(base_dir) / METRICS_FILE
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if ticker:
        df = df[df["ticker"] == ticker.upper()].reset_index(drop=True)
    return df


def load_contract_history(
    ticker: str,
    base_dir: Path = DEFAULT_BASE_DIR,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Concatenate all daily contract snapshots for a ticker.

    Adds a ``snapshot_date`` column taken from each day directory name.
    """

    base_dir = Path(base_dir)
    frames = []
    if base_dir.exists():
        for day_dir in sorted(base_dir.iterdir()):
            if not day_dir.is_dir():
                continue
            date = day_dir.name
            if (start and date < start) or (end and date > end):
                continue
            path = day_dir / f"{ticker.upper()}.csv.gz"
            if path.exists():
                frame = pd.read_csv(path)
                frame.insert(0, "snapshot_date", date)
                frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def compute_oi_change(
    ticker: str, base_dir: Path = DEFAULT_BASE_DIR
) -> pd.DataFrame:
    """Per-contract open-interest change between the two most recent snapshots.

    This is the free-data building block for OI-change tracking and the proxy
    flow feed (Phase 2): large overnight OI jumps flag where new positioning
    appeared.
    """

    history = load_contract_history(ticker, base_dir)
    if history.empty:
        return pd.DataFrame()

    dates = sorted(history["snapshot_date"].unique())
    if len(dates) < 2:
        return pd.DataFrame()

    prev_date, last_date = dates[-2], dates[-1]
    if not are_consecutive_market_sessions(prev_date, last_date):
        return pd.DataFrame()
    prev = history[history["snapshot_date"] == prev_date]
    last = history[history["snapshot_date"] == last_date]
    merged = compute_contract_snapshot_diff(prev, last, prev_date, last_date)
    # Keep the existing public OI-change frame stable. Consumers needing IV
    # changes use the lower-level contract diff helper directly.
    legacy_columns = CONTRACT_KEY + ["open_interest"]
    legacy_columns += [
        column for column in ("volume", "mid_iv") if column in merged.columns
    ]
    legacy_columns += ["open_interest_prev", "oi_change", "from_date", "to_date"]
    merged = merged[legacy_columns]
    return merged.sort_values(
        "oi_change", key=lambda s: s.abs(), ascending=False
    ).reset_index(drop=True)


def compute_contract_snapshot_diff(
    previous: pd.DataFrame,
    current: pd.DataFrame,
    from_date: str,
    to_date: str,
) -> pd.DataFrame:
    """Compare two daily contract snapshots by contract key.

    Missing prior OI is treated as zero, matching ``compute_oi_change``'s
    established behavior. IV is not zero-filled: a missing prior quote means
    the daily IV change is unavailable rather than a measured move.
    """

    current_columns = CONTRACT_KEY + ["open_interest"]
    current_columns += [
        column for column in ("volume", "mid_iv") if column in current.columns
    ]
    previous_columns = CONTRACT_KEY + ["open_interest"]
    if "mid_iv" in previous.columns:
        previous_columns.append("mid_iv")

    prev = previous[previous_columns].copy()
    last = current[current_columns].copy()
    merged = last.merge(prev, on=CONTRACT_KEY, how="left", suffixes=("", "_prev"))
    merged["open_interest_prev"] = merged["open_interest_prev"].fillna(0)
    merged["oi_change"] = merged["open_interest"] - merged["open_interest_prev"]

    if "mid_iv" in merged.columns and "mid_iv_prev" in merged.columns:
        merged["iv_change"] = merged["mid_iv"] - merged["mid_iv_prev"]
    elif "mid_iv" in merged.columns:
        merged["mid_iv_prev"] = pd.NA
        merged["iv_change"] = pd.NA

    merged["from_date"] = from_date
    merged["to_date"] = to_date
    return merged


def compute_iv_rank(
    ticker: str, base_dir: Path = DEFAULT_BASE_DIR
) -> Optional[Dict[str, float]]:
    """IV rank/percentile of the latest ATM IV within our recorded history.

    Honest labelling: until ~1 year of snapshots accrues this is a
    "rank since inception" — ``days_of_history`` tells the UI how much
    history backs the number.
    """

    metrics = load_daily_metrics(base_dir, ticker)
    if metrics.empty or "atm_iv" not in metrics.columns:
        return None

    series = metrics.dropna(subset=["atm_iv"]).sort_values("date")["atm_iv"]
    if len(series) < 2:
        return None

    latest = float(series.iloc[-1])
    low, high = float(series.min()), float(series.max())
    rank = 0.0 if high == low else (latest - low) / (high - low) * 100
    percentile = float((series < latest).mean() * 100)

    return {
        "iv": latest,
        "iv_rank": rank,
        "iv_percentile": percentile,
        "iv_low": low,
        "iv_high": high,
        "days_of_history": int(len(series)),
    }
