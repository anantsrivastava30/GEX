import sys
import os
import types
import pandas as pd
import pytest

# provide lightweight stand-ins for heavy optional deps imported by helpers
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

sys.modules.setdefault("yfinance", types.ModuleType("yfinance"))
sys.modules.setdefault("feedparser", types.ModuleType("feedparser"))
sys.modules.setdefault("streamlit", types.ModuleType("streamlit"))
ta = types.ModuleType("tradier_api")
ta.TradierAPI = object
sys.modules.setdefault("tradier_api", ta)
yaml = types.ModuleType("yaml")
yaml.safe_load = lambda s: {}
sys.modules.setdefault("yaml", yaml)

from helpers import compute_put_call_ratios, compute_unusual_spikes


def test_compute_put_call_ratios_basic():
    data = [
        {"option_type": "call", "volume": 100, "open_interest": 200},
        {"option_type": "call", "volume": 100, "open_interest": 200},
        {"option_type": "put",  "volume": 30,  "open_interest": 80},
        {"option_type": "put",  "volume": 70,  "open_interest": 120},
        {"option_type": "put",  "volume": 50,  "open_interest": 100},
    ]
    df = pd.DataFrame(data)
    vol_r, oi_r = compute_put_call_ratios(df)
    assert vol_r == pytest.approx(0.75)
    assert oi_r == pytest.approx(0.75)


def test_compute_unusual_spikes_simple():
    data = [
        {"strike": 100, "option_type": "put",  "volume": 10,  "open_interest": 5},
        {"strike": 100, "option_type": "call", "volume": 5,   "open_interest": 10},
        {"strike": 101, "option_type": "put",  "volume": 20,  "open_interest": 10},
        {"strike": 101, "option_type": "call", "volume": 5,   "open_interest": 0},
        {"strike": 102, "option_type": "put",  "volume": 0,   "open_interest": 50},
        {"strike": 102, "option_type": "call", "volume": 100, "open_interest": 25},
    ]
    df = pd.DataFrame(data)
    res = compute_unusual_spikes(df, top_n=2)
    assert list(res["strike"]) == [102, 100]
    assert list(res["vol_oi_call"]) == pytest.approx([4.0, 0.5])
    assert list(res["vol_oi_put"]) == pytest.approx([0.0, 2.0])
    assert list(res["total_vol_oi"]) == pytest.approx([4.0, 2.5])
