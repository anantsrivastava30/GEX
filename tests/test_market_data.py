import pandas as pd
import pytest

from quant_analysis.services.market_data import (
    compute_net_gex,
    compute_put_call_ratios,
    compute_unusual_spikes,
)


def _make_chain():
    return [
        {"strike": 100, "option_type": "call", "open_interest": 10, "contract_size": 100,
         "greeks": {"gamma": 0.02}},
        {"strike": 100, "option_type": "put", "open_interest": 5, "contract_size": 100,
         "greeks": {"gamma": 0.03}},
        {"strike": 105, "option_type": "call", "open_interest": 20, "contract_size": 100,
         "greeks": {"gamma": 0.01}},
        # No greeks -> should be skipped
        {"strike": 105, "option_type": "put", "open_interest": 50, "contract_size": 100,
         "greeks": {}},
        # Outside the offset window -> should be filtered out
        {"strike": 200, "option_type": "call", "open_interest": 99, "contract_size": 100,
         "greeks": {"gamma": 0.5}},
    ]


class TestComputeNetGex:
    def test_puts_subtract_from_calls(self):
        df = compute_net_gex(_make_chain(), spot=100, offset=10)
        row_100 = df[df["Strike"] == 100].iloc[0]
        # call: 0.02 * 10 * 100 = 20, put: -(0.03 * 5 * 100) = -15
        assert row_100["Net GEX"] == pytest.approx(5.0)

    def test_strikes_outside_offset_excluded(self):
        df = compute_net_gex(_make_chain(), spot=100, offset=10)
        assert 200 not in df["Strike"].values

    def test_missing_gamma_skipped(self):
        df = compute_net_gex(_make_chain(), spot=100, offset=10)
        row_105 = df[df["Strike"] == 105].iloc[0]
        # Only the call contributes: 0.01 * 20 * 100 = 20
        assert row_105["Net GEX"] == pytest.approx(20.0)

    def test_sorted_by_strike(self):
        df = compute_net_gex(_make_chain(), spot=100, offset=10)
        assert list(df["Strike"]) == sorted(df["Strike"])


class TestPutCallRatios:
    def test_basic_ratios(self):
        df = pd.DataFrame(
            {
                "option_type": ["call", "call", "put"],
                "volume": [100, 100, 100],
                "open_interest": [50, 50, 200],
            }
        )
        vol_r, oi_r = compute_put_call_ratios(df)
        assert vol_r == pytest.approx(0.5)
        assert oi_r == pytest.approx(2.0)

    def test_zero_call_volume_does_not_divide_by_zero(self):
        df = pd.DataFrame(
            {
                "option_type": ["put"],
                "volume": [100],
                "open_interest": [10],
            }
        )
        vol_r, oi_r = compute_put_call_ratios(df)
        assert vol_r == pytest.approx(100.0)
        assert oi_r == pytest.approx(10.0)


class TestUnusualSpikes:
    def _df(self):
        return pd.DataFrame(
            {
                "strike": [100, 100, 105, 105, 110],
                "option_type": ["call", "put", "call", "put", "call"],
                "volume": [500, 100, 50, 60, 10],
                "open_interest": [100, 100, 100, 100, 100],
            }
        )

    def test_columns_renamed_per_side(self):
        spikes = compute_unusual_spikes(self._df())
        for col in ("vol_oi_call", "vol_oi_put", "open_interest_call", "open_interest_put"):
            assert col in spikes.columns

    def test_ordered_by_total_vol_oi(self):
        spikes = compute_unusual_spikes(self._df())
        totals = spikes["total_vol_oi"].tolist()
        assert totals == sorted(totals, reverse=True)
        assert spikes.iloc[0]["strike"] == 100

    def test_top_n_limits_rows(self):
        spikes = compute_unusual_spikes(self._df(), top_n=2)
        assert len(spikes) == 2
