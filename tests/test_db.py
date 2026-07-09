import quant_analysis.storage.db as db


class TestParseTokenValue:
    def test_none_and_empty(self):
        assert db._parse_token_value(None) == 0
        assert db._parse_token_value("") == 0
        assert db._parse_token_value("None") == 0

    def test_numeric_strings(self):
        assert db._parse_token_value("1234") == 1234
        assert db._parse_token_value("12.7") == 12

    def test_garbage_returns_zero(self):
        assert db._parse_token_value("not-a-number") == 0

    def test_int_passthrough(self):
        assert db._parse_token_value(42) == 42


class TestSqliteRoundTrip:
    def _use_tmp_db(self, monkeypatch, tmp_path):
        monkeypatch.setattr(db, "DB_FILE", tmp_path / "test.db")
        monkeypatch.setattr(db, "_supabase_client", None)

    def test_save_and_load_analysis(self, monkeypatch, tmp_path):
        self._use_tmp_db(monkeypatch, tmp_path)
        db.save_analysis(
            ticker="SPY",
            expirations=["2025-01-17"],
            payload={"spot": 600.0},
            response="Buy calls",
            token_count=123,
        )
        rows = db.load_analyses(limit=5)
        assert len(rows) == 1
        assert rows[0]["ticker"] == "SPY"
        assert rows[0]["response"] == "Buy calls"

    def test_total_token_usage_sums_rows(self, monkeypatch, tmp_path):
        self._use_tmp_db(monkeypatch, tmp_path)
        db.save_analysis("SPY", [], {}, "a", 100)
        db.save_analysis("QQQ", [], {}, "b", 250)
        assert db.get_total_token_usage() == 350

    def test_gamma_gap_round_trip(self, monkeypatch, tmp_path):
        self._use_tmp_db(monkeypatch, tmp_path)
        db.save_gamma_gap_results(
            [
                {
                    "ticker": "NVDA",
                    "expiration": "2025-02-21",
                    "dte": 7,
                    "spot": 130.0,
                    "magnet_strike": 135.0,
                    "magnet_gex": 1_000_000.0,
                    "distance": 5.0,
                    "score": 88.5,
                    "positive_zone": True,
                }
            ]
        )
        rows = db.load_gamma_gap_history(limit=5)
        assert len(rows) == 1
        assert rows[0]["ticker"] == "NVDA"
        assert rows[0]["positive_zone"] == 1

    def test_save_gamma_gap_ignores_empty(self, monkeypatch, tmp_path):
        self._use_tmp_db(monkeypatch, tmp_path)
        db.save_gamma_gap_results([])
        assert db.load_gamma_gap_history(limit=5) == []
