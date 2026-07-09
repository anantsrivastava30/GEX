import pytest
from fastapi.testclient import TestClient

from backend.app import services
from backend.app.cache import cache
from backend.app.main import app
from backend.app.ratelimit import TokenBucket
from quant_analysis.storage.snapshots import upsert_daily_metrics


def make_chain(spot=100.0):
    chain = []
    for strike in (95.0, 100.0, 105.0):
        for option_type, oi_mult in (("call", 10), ("put", 5)):
            chain.append(
                {
                    "strike": strike,
                    "option_type": option_type,
                    "open_interest": int(strike * oi_mult),
                    "volume": 500 if option_type == "call" else 250,
                    "contract_size": 100,
                    "greeks": {
                        "gamma": 0.05 if strike == spot else 0.02,
                        "delta": 0.5 if option_type == "call" else -0.5,
                        "vega": 0.1,
                        "mid_iv": 0.22,
                    },
                }
            )
    return chain


class FakeTradier:
    def quote(self, symbol):
        return {"last": 100.0, "bid": 99.9, "ask": 100.1, "volume": 1_000_000}

    def option_chain(self, symbol, expiration, greeks="true", include_all_roots=True):
        return make_chain()

    def expirations(self, symbol, include_all_roots=False):
        return ["2026-08-21", "2026-09-18"]


@pytest.fixture(autouse=True)
def isolated_cache():
    cache.clear()
    yield
    cache.clear()


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(services, "get_tradier", lambda: FakeTradier())
    monkeypatch.setattr(services, "tradier_call", lambda fn, *a, **kw: fn(*a, **kw))
    return TestClient(app)


def test_health(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_ticker_snapshot(client):
    resp = client.get("/api/ticker/spy/snapshot")
    assert resp.status_code == 200
    body = resp.json()
    assert body["symbol"] == "SPY"
    assert body["last"] == 100.0


def test_expirations(client):
    resp = client.get("/api/ticker/SPY/expirations")
    assert resp.json() == ["2026-08-21", "2026-09-18"]


def test_gex_profile(client):
    resp = client.get(
        "/api/ticker/SPY/gex", params={"expiration": "2026-08-21", "offset": 35}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["spot"] == 100.0
    assert len(body["profile"]) == 3
    assert body["gamma_gap"]["magnet_strike"] == 100.0
    assert body["gamma_gap"]["commentary"]
    assert body["interpretation"]


def test_skew(client):
    resp = client.get("/api/ticker/SPY/skew", params={"expiration": "2026-08-21"})
    assert resp.status_code == 200
    points = resp.json()["points"]
    assert len(points) == 3
    assert points[0]["iv_skew"] == pytest.approx(0.0)


def test_ratios(client):
    resp = client.get(
        "/api/ticker/SPY/ratios", params={"expirations": ["2026-08-21"]}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["pc_volume_ratio"] == pytest.approx(0.5)
    assert body["pc_oi_ratio"] == pytest.approx(0.5)


def test_unusual_flow(client):
    resp = client.get(
        "/api/flow/unusual",
        params={"symbol": "SPY", "expirations": ["2026-08-21"], "top_n": 2},
    )
    assert resp.status_code == 200
    rows = resp.json()["rows"]
    assert len(rows) == 2
    assert "total_vol_oi" in rows[0]


def test_iv_rank_from_snapshots(client, monkeypatch, tmp_path):
    monkeypatch.setattr(services, "_snapshot_dir", lambda: tmp_path)
    upsert_daily_metrics(
        [
            {"date": "2026-07-07", "ticker": "SPY", "atm_iv": 0.10},
            {"date": "2026-07-08", "ticker": "SPY", "atm_iv": 0.30},
            {"date": "2026-07-09", "ticker": "SPY", "atm_iv": 0.20},
        ],
        tmp_path,
    )

    resp = client.get("/api/history/SPY/iv-rank")
    assert resp.status_code == 200
    body = resp.json()
    assert body["iv_rank"] == pytest.approx(50.0)
    assert body["days_of_history"] == 3


def test_iv_rank_without_history_404(client, monkeypatch, tmp_path):
    monkeypatch.setattr(services, "_snapshot_dir", lambda: tmp_path)
    assert client.get("/api/history/SPY/iv-rank").status_code == 404


def test_missing_token_returns_503(monkeypatch):
    from backend.app import config, deps

    monkeypatch.delenv("TRADIER_TOKEN", raising=False)
    config.get_settings.cache_clear()
    deps.get_tradier.cache_clear()
    with TestClient(app) as bare_client:
        resp = bare_client.get("/api/ticker/SPY/snapshot")
    assert resp.status_code == 503
    config.get_settings.cache_clear()


def test_token_bucket_exhausts_and_refills():
    bucket = TokenBucket(requests_per_minute=2)
    assert bucket.try_acquire()
    assert bucket.try_acquire()
    assert not bucket.try_acquire()
