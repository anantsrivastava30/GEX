"""Congressional stock-trade disclosures from free/optional providers.

Unusual Whales surfaces Senate and House trading disclosures. Both chambers
file periodic transaction reports (PTRs) that several providers expose as JSON.
Those community mirrors are notoriously unstable (hosts move, buckets lock), so
this module is deliberately provider-pluggable and tolerant:

* Providers are tried in order and normalised into one row shape.
* An optional Financial Modeling Prep key (``FMP_API_KEY``) is used first when
  present - the most reliable free-tier source.
* Config-driven JSON mirrors (``congress.sources`` in ``config.yaml``) follow.
* A single provider failing never fails the endpoint; it is reported in
  ``unavailable_sources`` and the response is flagged ``stale``.

No provider is required for the endpoint to respond; it simply returns an empty,
stale result the UI labels honestly until a reachable source is configured.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Callable, Dict, List, Optional

import requests

from quant_analysis.config import CONFIG

from backend.app.cache import cache
from backend.app.config import get_settings

# 6h TTL: disclosures update at most daily and source files are large.
TTL_CONGRESS = 6 * 3600
_HTTP_TIMEOUT = 20
_MAX_NORMALISED = 4000
_STALE_AFTER_DAYS = 14
_DATE_FORMATS = ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%Y/%m/%d", "%m/%d/%Y %H:%M:%S")
_UA = "Mozilla/5.0 (compatible; GEX-Terminal/0.2; +https://github.com)"

_CONGRESS_CONFIG = CONFIG.get("congress", {}) if isinstance(CONFIG, dict) else {}

# Ordered JSON mirror providers. Each entry: name, chamber, url. The default
# stock-watcher mirrors are kept as a best-effort fallback (they have gone dark
# before); override or extend via config.yaml when a reachable mirror exists.
_DEFAULT_SOURCES: List[Dict[str, str]] = [
    {
        "name": "senate-stock-watcher",
        "chamber": "senate",
        "url": (
            "https://senate-stock-watcher-data.s3-us-west-2.amazonaws.com/"
            "aggregate/all_transactions.json"
        ),
    },
    {
        "name": "house-stock-watcher",
        "chamber": "house",
        "url": (
            "https://house-stock-watcher-data.s3-us-west-2.amazonaws.com/"
            "data/all_transactions.json"
        ),
    },
]

_SOURCES: List[Dict[str, str]] = _CONGRESS_CONFIG.get("sources") or _DEFAULT_SOURCES

_FMP_BASE = "https://financialmodelingprep.com/stable"


# --------------------------------------------------------------------------- #
# Tolerant field access + normalisation
# --------------------------------------------------------------------------- #

def _first(record: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in record and record[key] not in (None, "", "--"):
            return record[key]
    return None


def _parse_date(raw: Any) -> Optional[str]:
    if not raw or not isinstance(raw, str):
        return None
    text = raw.strip()
    if not text or text.lower() in {"--", "n/a", "none"}:
        return None
    # ISO datetimes ("2023-09-27T00:00:00") -> keep the date part.
    if "T" in text and len(text) >= 10:
        try:
            return datetime.fromisoformat(text[:19]).date().isoformat()
        except ValueError:
            pass
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def _normalise_type(raw: Any) -> str:
    text = str(raw or "").strip().lower()
    if not text:
        return "unknown"
    if "purchase" in text or text == "buy":
        return "buy"
    if "sale" in text or "sell" in text:
        return "sell"
    if "exchange" in text:
        return "exchange"
    if "receive" in text:
        return "receive"
    return text


def _clean_ticker(raw: Any) -> Optional[str]:
    text = str(raw or "").strip().upper()
    if not text or text in {"--", "N/A", "NONE", "<BLANK>"}:
        return None
    return text


def _member_name(record: Dict[str, Any]) -> Optional[str]:
    direct = _first(record, "representative", "senator", "office", "member")
    if direct:
        return str(direct).strip()
    first = _first(record, "firstName", "first_name")
    last = _first(record, "lastName", "last_name")
    if first or last:
        return f"{first or ''} {last or ''}".strip()
    return None


def _normalise(record: Dict[str, Any], chamber_hint: str) -> Optional[Dict[str, Any]]:
    """Collapse any provider's record into one row; probes many key aliases."""
    txn_date = _parse_date(
        _first(record, "transaction_date", "transactionDate", "traded")
    )
    disc_date = _parse_date(
        _first(record, "disclosure_date", "disclosureDate", "dateRecieved", "published")
    )
    if not (txn_date or disc_date):
        return None
    return {
        "chamber": chamber_hint,
        "member": _member_name(record),
        "party": _first(record, "party"),
        "state": _first(record, "district", "state"),
        "ticker": _clean_ticker(_first(record, "ticker", "symbol")),
        "asset_description": (
            str(
                _first(record, "asset_description", "assetDescription", "asset") or ""
            ).strip()
            or None
        ),
        "transaction_type": _normalise_type(_first(record, "type", "transaction_type")),
        "amount": str(_first(record, "amount", "range") or "").strip() or None,
        "owner": str(_first(record, "owner") or "").strip() or None,
        "transaction_date": txn_date,
        "disclosure_date": disc_date,
    }


# --------------------------------------------------------------------------- #
# Providers
# --------------------------------------------------------------------------- #

def _get_json(url: str, params: Optional[Dict[str, Any]] = None) -> Any:
    resp = requests.get(
        url, params=params, timeout=_HTTP_TIMEOUT, headers={"user-agent": _UA}
    )
    resp.raise_for_status()
    return resp.json()


def _unwrap_list(data: Any) -> List[Any]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("transactions", "data", "results"):
            if isinstance(data.get(key), list):
                return data[key]
    return []


def _mirror_provider(source: Dict[str, str]) -> List[Dict[str, Any]]:
    records = _unwrap_list(_get_json(source["url"]))
    chamber = source.get("chamber", "senate")
    out: List[Dict[str, Any]] = []
    for record in records:
        if isinstance(record, dict):
            row = _normalise(record, chamber)
            if row:
                out.append(row)
    return out


def _fmp_provider(api_key: str) -> List[Dict[str, Any]]:
    # FMP's current "stable" API. The free tier returns the latest ~100 rows per
    # chamber; pagination/limit and symbol lookups are premium, so we pull the
    # latest feeds and filter client-side.
    out: List[Dict[str, Any]] = []
    for endpoint, chamber in (
        ("senate-latest", "senate"),
        ("house-latest", "house"),
    ):
        records = _unwrap_list(
            _get_json(f"{_FMP_BASE}/{endpoint}", {"apikey": api_key})
        )
        for record in records:
            if isinstance(record, dict):
                row = _normalise(record, chamber)
                if row:
                    out.append(row)
    return out


def _providers() -> List[tuple[str, Callable[[], List[Dict[str, Any]]]]]:
    """Fallback JSON mirror providers (tried only when FMP yields nothing)."""
    return [
        (source["name"], lambda s=source: _mirror_provider(s)) for source in _SOURCES
    ]


# --------------------------------------------------------------------------- #
# Aggregate + filter
# --------------------------------------------------------------------------- #

def _sort_key(row: Dict[str, Any]) -> str:
    return row.get("transaction_date") or row.get("disclosure_date") or ""


def _is_stale(latest: Optional[str]) -> bool:
    if not latest:
        return True
    try:
        latest_date = date.fromisoformat(latest)
    except ValueError:
        return True
    return (date.today() - latest_date).days > _STALE_AFTER_DAYS


def _load_all() -> Dict[str, Any]:
    """Load disclosures, preferring FMP; mirrors are a fallback only.

    When an ``FMP_API_KEY`` is configured and returns rows, the config mirrors
    are skipped entirely (they are an unreliable fallback), so a healthy FMP
    feed reports no unavailable sources. Only when FMP is absent or empty do we
    try the mirrors and report their per-source status.
    """
    rows: List[Dict[str, Any]] = []
    unavailable: List[str] = []
    ok = 0

    api_key = getattr(get_settings(), "fmp_api_key", None)
    if api_key:
        try:
            fmp_rows = _fmp_provider(api_key)
        except Exception:
            unavailable.append("financial-modeling-prep")
        else:
            if fmp_rows:
                ok += 1
                rows.extend(fmp_rows)
            else:
                unavailable.append("financial-modeling-prep")

    # Fall back to the config JSON mirrors only if FMP gave us nothing.
    if not rows:
        for name, provider in _providers():
            try:
                provider_rows = provider()
            except Exception:
                unavailable.append(name)
                continue
            if provider_rows:
                ok += 1
                rows.extend(provider_rows)
            else:
                unavailable.append(name)

    # De-dupe across providers on the identifying tuple.
    seen: set = set()
    deduped: List[Dict[str, Any]] = []
    for row in rows:
        key = (
            row.get("chamber"),
            row.get("member"),
            row.get("ticker"),
            row.get("transaction_date"),
            row.get("transaction_type"),
            row.get("amount"),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)

    deduped.sort(key=_sort_key, reverse=True)
    deduped = deduped[:_MAX_NORMALISED]

    latest = deduped[0]["transaction_date"] if deduped else None
    stale = ok == 0 or _is_stale(latest)
    return {
        "rows": deduped,
        "unavailable_sources": unavailable,
        "as_of": latest,
        "stale": stale,
    }


def get_congress_trades(
    ticker: Optional[str] = None,
    chamber: Optional[str] = None,
    days: Optional[int] = None,
    limit: int = 200,
) -> Dict[str, Any]:
    """Filtered congressional disclosures with source availability + staleness."""
    bundle = cache.get_or_compute("congress:all", TTL_CONGRESS, _load_all)
    rows: List[Dict[str, Any]] = bundle["rows"]

    ticker_filter = _clean_ticker(ticker) if ticker else None
    chamber_filter = chamber.strip().lower() if chamber else None
    cutoff: Optional[date] = None
    if days and days > 0:
        cutoff = date.fromordinal(date.today().toordinal() - days)

    filtered: List[Dict[str, Any]] = []
    for row in rows:
        if ticker_filter and row.get("ticker") != ticker_filter:
            continue
        if chamber_filter and row.get("chamber") != chamber_filter:
            continue
        if cutoff is not None:
            txn = row.get("transaction_date")
            if not txn:
                continue
            try:
                if date.fromisoformat(txn) < cutoff:
                    continue
            except ValueError:
                continue
        filtered.append(row)
        if len(filtered) >= limit:
            break

    return {
        "as_of": bundle["as_of"],
        "stale": bundle["stale"],
        "unavailable_sources": bundle["unavailable_sources"],
        "count": len(filtered),
        "rows": filtered,
    }
