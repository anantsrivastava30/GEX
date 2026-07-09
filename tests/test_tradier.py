import pytest

from quant_analysis.integrations.tradier import TradierAPI, TradierAPIError


class FakeResponse:
    def __init__(self, payload=None, text="", reason="Bad Request"):
        self._payload = payload
        self.text = text
        self.reason = reason

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class TestCoerceList:
    def test_none_becomes_empty_list(self):
        assert TradierAPI._coerce_list(None) == []

    def test_scalar_wrapped_in_list(self):
        assert TradierAPI._coerce_list("2025-01-17") == ["2025-01-17"]

    def test_list_passthrough(self):
        assert TradierAPI._coerce_list([1, 2]) == [1, 2]


class TestExtractErrorMessage:
    def test_fault_with_errorcode(self):
        resp = FakeResponse(
            {"fault": {"faultstring": "Rate limit", "detail": {"errorcode": "policies.ratelimit"}}}
        )
        assert TradierAPI._extract_error_message(resp) == "Rate limit (policies.ratelimit)"

    def test_faultstring_only(self):
        resp = FakeResponse({"fault": {"faultstring": "Invalid token"}})
        assert TradierAPI._extract_error_message(resp) == "Invalid token"

    def test_errors_dict(self):
        resp = FakeResponse({"errors": {"error": "Unknown symbol"}})
        assert "Unknown symbol" in TradierAPI._extract_error_message(resp)

    def test_non_json_uses_text(self):
        resp = FakeResponse(payload=None, text="Service Unavailable")
        assert TradierAPI._extract_error_message(resp) == "Service Unavailable"

    def test_non_json_empty_text_uses_reason(self):
        resp = FakeResponse(payload=None, text="  ", reason="Bad Gateway")
        assert TradierAPI._extract_error_message(resp) == "Bad Gateway"


class TestEndpointParsing:
    def _api(self, monkeypatch, payload):
        api = TradierAPI("token")
        monkeypatch.setattr(TradierAPI, "get", lambda self, endpoint, params=None: payload)
        return api

    def test_option_chain_single_option_coerced(self, monkeypatch):
        api = self._api(monkeypatch, {"options": {"option": {"strike": 100}}})
        assert api.option_chain("SPY", "2025-01-17") == [{"strike": 100}]

    def test_option_chain_null_options(self, monkeypatch):
        api = self._api(monkeypatch, {"options": "null"})
        assert api.option_chain("SPY", "2025-01-17") == []

    def test_expirations_list(self, monkeypatch):
        api = self._api(monkeypatch, {"expirations": {"date": ["2025-01-17", "2025-01-24"]}})
        assert api.expirations("SPY") == ["2025-01-17", "2025-01-24"]

    def test_quote_list_takes_first(self, monkeypatch):
        api = self._api(monkeypatch, {"quotes": {"quote": [{"last": 1.0}, {"last": 2.0}]}})
        assert api.quote("SPY") == {"last": 1.0}

    def test_quote_missing_returns_empty_dict(self, monkeypatch):
        api = self._api(monkeypatch, {"quotes": {}})
        assert api.quote("SPY") == {}


def test_error_has_status_code():
    err = TradierAPIError("boom", status_code=429)
    assert err.status_code == 429
    with pytest.raises(TradierAPIError):
        raise err
