"""CLI to capture the daily market snapshot for the configured watchlist.

Usage:
    python -m quant_analysis.scripts.snapshot                    # config watchlist
    python -m quant_analysis.scripts.snapshot --tickers SPY,QQQ  # explicit list

The Tradier token is read from the ``TRADIER_TOKEN`` environment variable
(as in CI) or from Streamlit secrets when running locally.

Run daily after the close by ``.github/workflows/snapshot.yml``, which
commits the resulting files under ``data/snapshots/``.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

from quant_analysis.storage.snapshots import (
    DEFAULT_BASE_DIR,
    DEFAULT_NUM_EXPIRATIONS,
    DEFAULT_TICKERS,
    capture_ticker_snapshot,
    capture_session_date,
    write_snapshot,
)

logger = logging.getLogger(__name__)

# Pause between tickers to stay well inside Tradier's rate limit
# (~120 requests/min; each ticker costs 2 + num_expirations requests).
TICKER_PAUSE_SECONDS = 4.0


def resolve_token() -> Optional[str]:
    token = os.environ.get("TRADIER_TOKEN")
    if token:
        return token
    try:
        import streamlit as st

        return st.secrets["TRADIER_TOKEN"]
    except Exception:
        return None


def run(tickers: List[str], num_expirations: int, base_dir: Path) -> int:
    token = resolve_token()
    if not token:
        logger.error(
            "No Tradier token found: set TRADIER_TOKEN or add it to Streamlit secrets"
        )
        return 2
    if capture_session_date() is None:
        logger.info("Snapshot run skipped: no active market session today")
        return 0

    captured, failed = [], []
    for i, ticker in enumerate(tickers):
        ticker = ticker.strip().upper()
        if not ticker:
            continue
        if i:
            time.sleep(TICKER_PAUSE_SECONDS)
        try:
            snapshot = capture_ticker_snapshot(ticker, token, num_expirations)
            if snapshot is None:
                failed.append(ticker)
                continue
            path = write_snapshot(snapshot, base_dir)
            captured.append(ticker)
            logger.info("Captured %s -> %s", ticker, path)
        except Exception:
            logger.exception("Snapshot failed for %s", ticker)
            failed.append(ticker)

    logger.info(
        "Snapshot run complete: %d captured %s, %d failed %s",
        len(captured), captured, len(failed), failed,
    )
    # Partial failures (illiquid ticker, transient API error) shouldn't lose
    # the day's data for everything else — only a full wipeout is an error.
    return 1 if captured == [] else 0


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tickers",
        default=",".join(DEFAULT_TICKERS),
        help="Comma-separated tickers (default: snapshots.tickers in config.yaml)",
    )
    parser.add_argument(
        "--expirations",
        type=int,
        default=DEFAULT_NUM_EXPIRATIONS,
        help="Number of near expirations to capture per ticker",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_BASE_DIR,
        help="Snapshot output directory",
    )
    args = parser.parse_args(argv)
    return run(args.tickers.split(","), args.expirations, args.base_dir)


if __name__ == "__main__":
    sys.exit(main())
