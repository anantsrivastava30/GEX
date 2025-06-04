import sys
import os
import types
import pandas as pd
import pytest

# dummy modules required by utils import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

sys.modules.setdefault("yfinance", types.ModuleType("yfinance"))
sys.modules.setdefault("seaborn", types.ModuleType("seaborn"))

mpl = types.ModuleType("matplotlib")
pyplot = types.ModuleType("matplotlib.pyplot")
mpl.pyplot = pyplot
sys.modules.setdefault("matplotlib", mpl)
sys.modules.setdefault("matplotlib.pyplot", pyplot)
sys.modules.setdefault("plotly", types.ModuleType("plotly"))
sys.modules.setdefault("plotly.graph_objects", types.ModuleType("plotly.graph_objects"))
sys.modules.setdefault("plotly.express", types.ModuleType("plotly.express"))
plotly_subplots = types.ModuleType("plotly.subplots")
plotly_subplots.make_subplots = lambda *a, **k: None
sys.modules.setdefault("plotly.subplots", plotly_subplots)
sys.modules["plotly.graph_objects"].Figure = object
sys.modules.setdefault("matplotlib.dates", types.ModuleType("matplotlib.dates"))
ta = types.ModuleType("tradier_api")
ta.TradierAPI = object
sys.modules.setdefault("tradier_api", ta)

from utils import compute_net_gamma_exposure


def test_compute_net_gamma_exposure_basic():
    chain = [
        {"strike": 100, "option_type": "call", "open_interest": 10, "contract_size": 100, "greeks": {"gamma": 0.5}},
        {"strike": 100, "option_type": "put",  "open_interest": 5,  "contract_size": 100, "greeks": {"gamma": 0.2}},
        {"strike": 101, "option_type": "call", "open_interest": 8,  "contract_size": 100, "greeks": {"gamma": 0.4}},
        {"strike": 101, "option_type": "put",  "open_interest": 0,  "contract_size": 100, "greeks": {"gamma": 0.3}},
        {"strike": 102, "option_type": "call", "open_interest": 5,  "contract_size": 100, "greeks": {}},
        {"strike": 105, "option_type": "call", "open_interest": 10, "contract_size": 100, "greeks": {"gamma": 0.1}},
    ]
    df = compute_net_gamma_exposure(chain, S=100, offset=2)
    assert list(df["Strike"]) == [100, 101]
    assert list(df["Net GEX"]) == [400.0, 320.0]
