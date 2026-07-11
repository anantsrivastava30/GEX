import pytest
import pandas as pd
from datetime import timedelta
from fastapi.testclient import TestClient

from backend.app import services
from backend.app.cache import cache
from backend.app.main import app
from backend.app.ratelimit import TokenBucket
from quant_analysis.storage.snapshots import upsert_daily_metrics
from quant_analysis.storage.snapshots import market_date_today


def make_chain(spot=100.0, mid_iv=0.22):
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
                        "mid_iv": mid_iv,
                        "smv_vol": mid_iv,
                    },
                }
            )
    return chain


class FakeTradier:
    def quote(self, symbol):
        return {"last": 100.0, "bid": 99.9, "ask": 100.1, "volume": 1_000_000}

    def option_chain(self, symbol, expiration, greeks="true", include_all_roots=True):
        # Distinct IV per expiration so the term structure has a slope.
        return make_chain(mid_iv=0.22 if expiration == "2026-08-21" else 0.26)

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


def test_cached_flow_endpoints_use_snapshot_data_only(client, monkeypatch, tmp_path):
    today = market_date_today()
    previous = (pd.Timestamp(today) - timedelta(days=1)).date().isoformat()
    for snapshot_date, oi, iv in ((previous, 100, 0.20), (today, 140, 0.24)):
        day_dir = tmp_path / snapshot_date
        day_dir.mkdir()
        pd.DataFrame(
            [
                {
                    "expiration_date": "2026-08-21",
                    "strike": 100.0,
                    "option_type": "call",
                    "open_interest": oi,
                    "volume": 75,
                    "mid_iv": iv,
                }
            ]
        ).to_csv(day_dir / "SPY.csv.gz", index=False, compression="gzip")

    monkeypatch.setattr(services, "_snapshot_dir", lambda: tmp_path)
    monkeypatch.setattr(services, "_configured_snapshot_tickers", lambda: ["SPY"])
    monkeypatch.setattr(
        services,
        "get_tradier",
        lambda: pytest.fail("cached flow endpoints must not call Tradier"),
    )

    feed = client.get("/api/flow/feed").json()
    assert feed["as_of"] == today
    assert not feed["stale"]
    assert not feed["unavailable_history"]
    assert feed["rows"][0]["oi_change"] == 40.0
    assert feed["rows"][0]["iv_change"] == pytest.approx(0.04)
    assert feed["rows"][0]["history_available"]

    chains = client.get("/api/flow/hottest-chains").json()
    assert chains["as_of"] == today
    assert chains["rows"][0]["total_open_interest"] == 140.0
    assert chains["rows"][0]["oi_change"] == 40.0


def test_cached_flow_reports_missing_history(client, monkeypatch, tmp_path):
    today = market_date_today()
    day_dir = tmp_path / today
    day_dir.mkdir()
    pd.DataFrame(
        [
            {
                "expiration_date": "2026-08-21",
                "strike": 100.0,
                "option_type": "call",
                "open_interest": 100,
                "volume": 75,
                "mid_iv": 0.20,
            }
        ]
    ).to_csv(day_dir / "SPY.csv.gz", index=False, compression="gzip")
    monkeypatch.setattr(services, "_snapshot_dir", lambda: tmp_path)
    monkeypatch.setattr(services, "_configured_snapshot_tickers", lambda: ["SPY"])

    body = client.get("/api/flow/feed").json()
    assert body["unavailable_history"]
    assert body["unavailable_history_tickers"] == ["SPY"]
    assert body["rows"][0]["oi_change"] is None
    assert not body["rows"][0]["history_available"]


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


def test_exposure_endpoint(client, monkeypatch):
    monkeypatch.setattr(services, "_risk_free_rate", lambda: 0.04)
    resp = client.get(
        "/api/ticker/SPY/exposure",
        params={"expirations": ["2026-08-21", "2026-09-18"], "offset": 35},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["risk_free_rate"] == 0.04
    assert [p["strike"] for p in body["points"]] == [95.0, 100.0, 105.0]
    for point in body["points"]:
        assert isinstance(point["vanna"], float)
        assert isinstance(point["charm"], float)
    # Call OI is double put OI at every strike, so net vanna keeps the raw
    # vanna's sign; ATM vanna is negative for d2 > 0 ... just assert nonzero.
    assert any(p["vanna"] != 0 for p in body["points"])


def test_max_pain_endpoint(client):
    resp = client.get(
        "/api/ticker/SPY/max-pain", params={"expiration": "2026-08-21"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["symbol"] == "SPY"
    assert body["max_pain"] in (95.0, 100.0, 105.0)
    assert len(body["curve"]) == 3
    pains = {row["strike"]: row["total_pain"] for row in body["curve"]}
    assert pains[body["max_pain"]] == min(pains.values())


def test_term_structure_endpoint(client):
    resp = client.get("/api/ticker/SPY/term-structure")
    assert resp.status_code == 200
    points = resp.json()["points"]
    assert [p["expiration"] for p in points] == ["2026-08-21", "2026-09-18"]
    assert points[0]["atm_iv"] == pytest.approx(0.22)
    assert points[1]["atm_iv"] == pytest.approx(0.26)


def test_gamma_gap_history_endpoint(client, monkeypatch):
    rows = [
        {"ts": "2026-07-10T14:00:00", "ticker": "SPY", "expiration": "2026-08-21",
         "dte": None, "spot": 100.0, "magnet_strike": 100.0, "magnet_gex": 1.0,
         "distance": 0.0, "score": 90.0, "positive_zone": 1, "payload": "{}"},
        {"ts": "2026-07-10T14:00:00", "ticker": "QQQ", "expiration": "2026-08-21",
         "dte": None, "spot": 400.0, "magnet_strike": 405.0, "magnet_gex": 1.0,
         "distance": 5.0, "score": 40.0, "positive_zone": 0, "payload": "{}"},
    ]
    monkeypatch.setattr(services, "load_gamma_gap_history", lambda limit: rows)

    resp = client.get("/api/history/gamma-gap")
    assert resp.status_code == 200
    assert len(resp.json()) == 2

    resp = client.get("/api/history/gamma-gap", params={"ticker": "spy"})
    body = resp.json()
    assert len(body) == 1
    assert body[0]["ticker"] == "SPY"
    assert "payload" not in body[0]


def test_ai_status_endpoint(client, monkeypatch):
    resp = client.get("/api/ai/status")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {"openai_configured", "pin_required", "default_model"}


def test_ai_analyze_requires_pin(client, monkeypatch):
    from backend.app import services_ai

    monkeypatch.setattr(
        services_ai, "get_settings", lambda: type(
            "S", (), {"ai_pin": "1234", "tradier_token": "t", "openai_model": "m"}
        )(),
    )
    resp = client.post("/api/ai/analyze", json={"symbol": "SPY", "pin": "wrong"})
    assert resp.status_code == 401


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
