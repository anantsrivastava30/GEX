import requests
from typing import Any, Dict, List, Optional


class TradierAPIError(RuntimeError):
    """Raised when Tradier returns an error or unexpected payload."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


class TradierAPI:
    """Simple wrapper around the Tradier REST API."""

    def __init__(self, token: str, base_url: str = "https://api.tradier.com/v1"):
        self.base_url = base_url.rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        }

    @staticmethod
    def _coerce_list(value: Any) -> List[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        return [value]

    @staticmethod
    def _extract_error_message(resp: requests.Response) -> str:
        try:
            payload = resp.json()
        except ValueError:
            text = resp.text.strip()
            return text or resp.reason

        if isinstance(payload, dict):
            fault = payload.get("fault")
            if isinstance(fault, dict):
                faultstring = fault.get("faultstring")
                detail = fault.get("detail")
                if isinstance(detail, dict):
                    errorcode = detail.get("errorcode")
                    if faultstring and errorcode:
                        return f"{faultstring} ({errorcode})"
                if faultstring:
                    return str(faultstring)

            errors = payload.get("errors")
            if isinstance(errors, dict):
                return str(errors)

        return str(payload)

    def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        try:
            resp = requests.get(url, params=params, headers=self.headers, timeout=20)
        except requests.RequestException as exc:
            raise TradierAPIError(f"Tradier request failed: {exc}") from exc

        if not resp.ok:
            message = self._extract_error_message(resp)
            raise TradierAPIError(
                f"Tradier request failed ({resp.status_code}) for {endpoint}: {message}",
                status_code=resp.status_code,
            )

        try:
            payload = resp.json()
        except ValueError as exc:
            raise TradierAPIError(
                f"Tradier returned non-JSON response for {endpoint}: {resp.text[:200]}",
                status_code=resp.status_code,
            ) from exc

        if not isinstance(payload, dict):
            raise TradierAPIError(
                f"Tradier returned unexpected payload type for {endpoint}: {type(payload).__name__}",
                status_code=resp.status_code,
            )

        return payload

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
        options = data.get("options", {})
        if not isinstance(options, dict):
            return []
        return self._coerce_list(options.get("option"))

    def expirations(self, symbol: str, include_all_roots: bool = False) -> List[str]:
        params = {"symbol": symbol}
        if include_all_roots:
            params["includeAllRoots"] = "true"
        data = self.get("markets/options/expirations", params)
        expirations = data.get("expirations", {})
        if not isinstance(expirations, dict):
            return []
        return self._coerce_list(expirations.get("date"))

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

    def orderbook(self, symbol: str) -> Dict[str, Any]:
        """Return the current order book for a symbol if available."""
        data = self.get("markets/orderbook", {"symbol": symbol})
        return data.get("book", {})
