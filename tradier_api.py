import requests
from typing import Any, Dict, List, Optional


class TradierAPI:
    """Simple wrapper around the Tradier REST API."""

    def __init__(self, token: str, base_url: str = "https://api.tradier.com/v1"):
        self.base_url = base_url.rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        }

    def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        resp = requests.get(url, params=params, headers=self.headers)
        resp.raise_for_status()
        return resp.json()

    # Convenience wrappers --------------------------------------------------
    def option_chain(
        self,
        symbol: str,
        expiration: str,
        greeks: str = "true",
        include_all_roots: bool = True,
    ) -> List[Dict[str, Any]]:
        params = {"symbol": symbol, "expiration": expiration, "greeks": greeks}
        if include_all_roots:
            params["includeAllRoots"] = "true"
        data = self.get("markets/options/chains", params)
        return data.get("options", {}).get("option", [])

    def expirations(self, symbol: str, include_all_roots: bool = False) -> List[str]:
        params = {"symbol": symbol}
        if include_all_roots:
            params["includeAllRoots"] = "true"
        data = self.get("markets/options/expirations", params)
        return data.get("expirations", {}).get("date", [])

    def quote(self, symbol: str) -> Dict[str, Any]:
        data = self.get("markets/quotes", {"symbols": symbol})
        q = data.get("quotes", {}).get("quote")
        if isinstance(q, list):
            q = q[0]
        return q or {}

    def history(
        self,
        symbol: str,
        interval: str = "daily",
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        params = {"symbol": symbol, "interval": interval}
        if start:
            params["start"] = start
        if end:
            params["end"] = end
        data = self.get("markets/history", params)
        return data.get("history", {}).get("day", [])

