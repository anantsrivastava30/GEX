"""Pure price-technical calculations used by snapshot-backed alerts."""

from __future__ import annotations

import math
from typing import Iterable, Optional


def price_sma_cross(
    closes: Iterable[float], current_price: float, window: int
) -> Optional[dict]:
    """Return the rolling SMA and today's price-cross direction.

    ``closes`` contains completed daily closes through the prior market session.
    The current rolling average replaces the oldest close with ``current_price``.
    """

    values = [float(value) for value in closes]
    price = float(current_price)
    if window < 2 or len(values) < window:
        return None
    values = values[-window:]
    if not math.isfinite(price) or price <= 0:
        return None
    if any(not math.isfinite(value) or value <= 0 for value in values):
        return None

    previous_close = values[-1]
    previous_sma = sum(values) / window
    current_sma = (sum(values[1:]) + price) / window
    cross = None
    if previous_close <= previous_sma and price > current_sma:
        cross = "above"
    elif previous_close >= previous_sma and price < current_sma:
        cross = "below"

    return {
        "sma": current_sma,
        "previous_sma": previous_sma,
        "previous_close": previous_close,
        "distance_pct": (price / current_sma - 1.0) * 100.0,
        "cross": cross,
    }
