"""Background jobs (P1.2): intraday snapshots + gamma-gap scoring.

Reuses the Phase 0 snapshot engine every 30 minutes during US market hours.
Snapshot writes are upserts, so intraday runs simply refresh the day's files;
each run also appends a row per ticker to ``gamma_gap_analysis`` so the
gamma-gap signal accrues a verifiable intraday track record (the future
public track-record page).

Off by default; enable with ``SCHEDULER_ENABLED=true``. Jobs call Tradier
outside the API token bucket - roughly six requests per ticker per run,
negligible against the ~120 req/min budget.
"""

from __future__ import annotations

import logging
import time as time_module
from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from backend.app.config import get_settings

logger = logging.getLogger(__name__)

MARKET_TZ = ZoneInfo("America/New_York")
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)


def market_is_open(now: datetime | None = None) -> bool:
    """Weekday regular-hours check (holidays excluded; a holiday run just
    re-writes unchanged data, which the upsert makes harmless)."""

    now = now or datetime.now(tz=MARKET_TZ)
    if now.weekday() >= 5:
        return False
    return MARKET_OPEN <= now.time() <= MARKET_CLOSE


def run_intraday_snapshots() -> None:
    """Capture chains, write snapshots, and log gamma-gap scan rows."""

    settings = get_settings()
    if not settings.tradier_token:
        logger.info("Intraday snapshot skipped: TRADIER_TOKEN not configured.")
        return
    if not market_is_open():
        logger.debug("Intraday snapshot skipped: market closed.")
        return

    capture_universe()


def capture_universe() -> dict:
    """Capture the configured snapshot universe once and persist results.

    Shared by the scheduler job and the admin capture endpoint. Off-hours
    runs are safe: chains fetched while the market is closed carry the last
    session's OI/volume and key to ``last_trading_date()``, so a weekend
    capture seeds Friday's snapshot instead of a bogus weekend row.
    """

    from quant_analysis.storage import snapshots as snapshot_store
    from quant_analysis.storage.db import save_gamma_gap_results

    settings = get_settings()
    base_dir = Path(settings.snapshot_dir)
    gap_rows = []
    captured = 0
    tickers = settings.snapshot_tickers
    for index, ticker in enumerate(tickers):
        snap = None
        try:
            snap = snapshot_store.capture_ticker_snapshot(
                ticker, settings.tradier_token
            )
        except Exception:
            logger.exception("Snapshot capture failed for %s", ticker)
        if snap:
            try:
                snapshot_store.write_snapshot(snap, base_dir)
                captured += 1

                metrics = snap["metrics"]
                if metrics.get("gamma_gap_score") is not None:
                    expirations = str(metrics.get("expirations", ""))
                    gap_rows.append(
                        {
                            "ticker": ticker,
                            "expiration": expirations,
                            "dte": None,
                            "spot": snap.get("spot"),
                            "magnet_strike": metrics.get("gamma_magnet_strike"),
                            "magnet_gex": metrics.get("gamma_magnet_gex"),
                            "distance": metrics.get("gamma_gap_distance"),
                            "score": metrics.get("gamma_gap_score"),
                            "positive_zone": metrics.get("gamma_positive_zone"),
                        }
                    )
            except Exception:
                logger.exception("Snapshot persistence failed for %s", ticker)

        # Captures make several upstream calls. Pace the bounded configured
        # universe instead of allowing an intraday run to burst the free plan.
        if index < len(tickers) - 1 and settings.snapshot_refresh_pause_seconds:
            time_module.sleep(settings.snapshot_refresh_pause_seconds)

    if gap_rows:
        try:
            save_gamma_gap_results(gap_rows)
        except Exception:
            logger.exception("Failed to persist gamma-gap scan rows")

    logger.info(
        "Snapshot capture done: %d/%d tickers, %d gamma-gap rows",
        captured,
        len(tickers),
        len(gap_rows),
    )
    return {
        "captured": captured,
        "requested": len(tickers),
        "tickers": list(tickers),
        "gamma_gap_rows": len(gap_rows),
        "as_of": snapshot_store.last_trading_date(),
    }


def create_scheduler() -> BackgroundScheduler:
    """Scheduler with the intraday job registered (caller starts it)."""

    scheduler = BackgroundScheduler(timezone=str(MARKET_TZ))
    scheduler.add_job(
        run_intraday_snapshots,
        CronTrigger(
            day_of_week="mon-fri",
            hour="9-16",
            minute="0,30",
            timezone=str(MARKET_TZ),
        ),
        id="intraday_snapshots",
        coalesce=True,
        max_instances=1,
        misfire_grace_time=300,
    )
    return scheduler
