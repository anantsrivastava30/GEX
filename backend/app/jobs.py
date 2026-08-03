"""Background jobs (P1.2): intraday snapshots + gamma-gap scoring.

Reuses the Phase 0 snapshot engine every 30 minutes during US market hours.
Each successful capture refreshes the canonical daily file and appends an
immutable Parquet archive. Gamma-gap rows also remain append-only.

Off by default; enable with ``SCHEDULER_ENABLED=true``. Jobs call Tradier
outside the API token bucket and pace the configured universe between tickers.
"""

from __future__ import annotations

import logging
import os
import time as time_module
from datetime import datetime, time, timezone
from pathlib import Path
from threading import Lock
from uuid import uuid4
from zoneinfo import ZoneInfo

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from backend.app.config import get_settings

logger = logging.getLogger(__name__)

MARKET_TZ = ZoneInfo("America/New_York")
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)
ALERT_EVALUATION_CLOSE = time(16, 15)
_CAPTURE_LOCK = Lock()


def market_is_open(now: datetime | None = None) -> bool:
    """US equity regular-hours check, excluding standard full-day holidays."""

    now = now or datetime.now(tz=MARKET_TZ)
    from quant_analysis.storage.snapshots import is_market_session

    if not is_market_session(now.date()):
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

    try:
        capture_universe()
    except RuntimeError:
        logger.info("Intraday snapshot skipped: another capture is already running.")


def alert_evaluation_window(now: datetime | None = None) -> bool:
    """Allow evaluations after each snapshot, including the 16:10 close run."""

    now = now or datetime.now(tz=MARKET_TZ)
    from quant_analysis.storage.snapshots import is_market_session

    if not is_market_session(now.date()):
        return False
    return MARKET_OPEN <= now.time() <= ALERT_EVALUATION_CLOSE


def run_alert_evaluation() -> None:
    if not alert_evaluation_window():
        return
    from backend.app.services_alerts import evaluate_enabled_alerts

    result = evaluate_enabled_alerts()
    logger.info(
        "Alert evaluation done: %d rules, %d events",
        result["evaluated"],
        result["emitted"],
    )


def run_direction_evaluation() -> None:
    """Persist O'Neil direction signals shortly after the close.

    Runs on completed daily bars only; the lazy GET-path evaluation shares
    the same idempotent insert, so double-runs never double-alert.
    """

    now = datetime.now(tz=MARKET_TZ)
    from quant_analysis.storage.snapshots import is_market_session

    if not is_market_session(now.date()):
        return
    try:
        from backend.app.services_direction import evaluate_direction_signals

        result = evaluate_direction_signals()
        logger.info(
            "Direction evaluation done: %d indices, %d new signals (as of %s)",
            result["evaluated"],
            result["inserted"],
            result["as_of"],
        )
    except Exception:
        logger.exception("Direction evaluation failed")
    try:
        from backend.app.services_canslim import evaluate_stock_signals

        result = evaluate_stock_signals()
        logger.info(
            "CAN SLIM stock evaluation done: %d stocks, %d breakout signals (as of %s)",
            result["evaluated"],
            result["inserted"],
            result["as_of"],
        )
    except Exception:
        logger.exception("CAN SLIM stock evaluation failed")


def warm_congress_cache() -> None:
    """Best-effort daily warm-up for the six-hour disclosure cache."""

    try:
        from backend.app.services_congress import get_congress_trades

        result = get_congress_trades(limit=1)
        logger.info(
            "Congress cache warm-up done: as_of=%s stale=%s",
            result.get("as_of"),
            result.get("stale"),
        )
    except Exception:
        logger.exception("Congress cache warm-up failed")


def capture_universe(producer: str = "scheduler") -> dict:
    """Serialize scheduler and admin captures within this backend process."""

    if not _CAPTURE_LOCK.acquire(blocking=False):
        raise RuntimeError("Snapshot capture is already running")
    try:
        return _capture_universe(producer)
    finally:
        _CAPTURE_LOCK.release()


def _capture_universe(producer: str) -> dict:
    """Capture the configured and shared-watchlist universe once.

    Shared by the scheduler job and the admin capture endpoint. Off-session
    runs are skipped so a rolled provider expiration universe cannot overwrite
    the previous session with non-comparable data.
    """

    from quant_analysis.storage import snapshots as snapshot_store
    from quant_analysis.storage.db import save_gamma_gap_results
    from backend.app.services_watchlists import snapshot_symbols

    settings = get_settings()
    base_dir = Path(settings.snapshot_dir)
    intraday_base_dir = Path(settings.intraday_snapshot_dir)
    tickers = snapshot_symbols()
    run_started_at = datetime.now(timezone.utc)
    session_date = snapshot_store.capture_session_date(run_started_at)
    if session_date is None:
        logger.info("Snapshot capture skipped: no active market session today")
        return {
            "captured": 0,
            "requested": len(tickers),
            "tickers": list(tickers),
            "gamma_gap_rows": 0,
            "intraday_archived": 0,
            "as_of": snapshot_store.last_trading_date(),
        }
    gap_rows = []
    run_id = uuid4().hex
    captured = 0
    intraday_archived = 0
    captured_tickers = []
    archived_tickers = []
    capture_failed_tickers = []
    archive_failed_tickers = []
    daily_failed_tickers = []
    for index, ticker in enumerate(tickers):
        snap = None
        try:
            snap = snapshot_store.capture_ticker_snapshot(
                ticker, settings.tradier_token
            )
        except Exception:
            logger.exception("Snapshot capture failed for %s", ticker)
            capture_failed_tickers.append(ticker)
        if snap:
            archive_succeeded = False
            try:
                snapshot_store.write_intraday_snapshot(
                    snap,
                    intraday_base_dir,
                    producer=producer,
                    run_id=run_id,
                )
                intraday_archived += 1
                archived_tickers.append(ticker)
                archive_succeeded = True
            except Exception:
                archive_failed_tickers.append(ticker)
                logger.exception("Intraday archive failed for %s", ticker)

            if archive_succeeded:
                try:
                    snapshot_store.write_snapshot(snap, base_dir)
                    captured += 1
                    captured_tickers.append(ticker)

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
                    daily_failed_tickers.append(ticker)
                    logger.exception("Snapshot persistence failed for %s", ticker)
        elif ticker not in capture_failed_tickers:
            capture_failed_tickers.append(ticker)

        # Captures make several upstream calls. Pace the bounded scheduled
        # universe instead of allowing an intraday run to burst the free plan.
        if index < len(tickers) - 1 and settings.snapshot_refresh_pause_seconds:
            time_module.sleep(settings.snapshot_refresh_pause_seconds)

    if gap_rows:
        try:
            save_gamma_gap_results(gap_rows)
        except Exception:
            logger.exception("Failed to persist gamma-gap scan rows")

    run_completed_at = datetime.now(timezone.utc)
    run_complete = captured == len(tickers) and intraday_archived == len(tickers)
    manifest_written = False
    try:
        snapshot_store.write_intraday_run_manifest(
            {
                "run_id": run_id,
                "schema_version": snapshot_store.INTRADAY_SCHEMA_VERSION,
                "status": "complete" if run_complete else "partial",
                "producer": producer,
                "provider": "tradier",
                "source_revision": os.environ.get("SOURCE_REVISION", ""),
                "session_date": session_date,
                "started_at_utc": run_started_at.isoformat(),
                "completed_at_utc": run_completed_at.isoformat(),
                "requested_tickers": tickers,
                "archived_tickers": archived_tickers,
                "daily_tickers": captured_tickers,
                "capture_failed_tickers": capture_failed_tickers,
                "archive_failed_tickers": archive_failed_tickers,
                "daily_failed_tickers": daily_failed_tickers,
            },
            intraday_base_dir,
        )
        manifest_written = True
    except Exception:
        logger.exception("Failed to persist intraday run manifest %s", run_id)

    logger.info(
        "Snapshot capture done: %d/%d daily, %d intraday, %d gamma-gap rows",
        captured,
        len(tickers),
        intraday_archived,
        len(gap_rows),
    )
    return {
        "captured": captured,
        "requested": len(tickers),
        "tickers": list(tickers),
        "gamma_gap_rows": len(gap_rows),
        "intraday_archived": intraday_archived,
        "run_id": run_id,
        "manifest_written": manifest_written,
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
    scheduler.add_job(
        run_alert_evaluation,
        CronTrigger(
            day_of_week="mon-fri",
            hour="9-16",
            minute="10,40",
            timezone=str(MARKET_TZ),
        ),
        id="alert_evaluation",
        coalesce=True,
        max_instances=1,
        misfire_grace_time=300,
    )
    scheduler.add_job(
        run_direction_evaluation,
        CronTrigger(
            day_of_week="mon-fri",
            hour=16,
            minute=20,
            timezone=str(MARKET_TZ),
        ),
        id="direction_evaluation",
        coalesce=True,
        max_instances=1,
        misfire_grace_time=1800,
    )
    scheduler.add_job(
        warm_congress_cache,
        CronTrigger(
            day_of_week="mon-fri",
            hour=7,
            minute=15,
            timezone=str(MARKET_TZ),
        ),
        id="congress_cache_warmup",
        coalesce=True,
        max_instances=1,
        misfire_grace_time=1800,
    )
    return scheduler
