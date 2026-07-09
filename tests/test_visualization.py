import math

import numpy as np
import pandas as pd
import pytest

from quant_analysis.analytics.visualization import (
    compute_gamma_gap_metrics,
    describe_gamma_gap,
    generate_binomial_tree,
    interpret_net_gex,
)


def _df_net(strikes, gex):
    return pd.DataFrame({"Strike": strikes, "GEX": gex})


class TestGammaGapMetrics:
    def test_magnet_is_max_positive_gex(self):
        df = _df_net([95, 100, 105, 110], [-50, 200, 500, 100])
        metrics = compute_gamma_gap_metrics(df, spot=100, offset=20)
        assert metrics["magnet_strike"] == 105
        assert metrics["magnet_gex"] == 500
        assert metrics["distance"] == pytest.approx(5.0)

    def test_score_bounded_between_0_and_120(self):
        df = _df_net([99, 100, 101], [10, 5000, 10])
        metrics = compute_gamma_gap_metrics(df, spot=100, offset=20)
        assert 0 <= metrics["score"] <= 120

    def test_positive_zone_flag(self):
        df = _df_net([95, 100, 105], [10, 300, 20])
        metrics = compute_gamma_gap_metrics(df, spot=100, offset=20)
        assert metrics["positive_zone"] is True

    def test_returns_none_without_positive_gamma(self):
        df = _df_net([95, 100, 105], [-10, -300, -20])
        assert compute_gamma_gap_metrics(df, spot=100, offset=20) is None

    def test_returns_none_for_empty_or_missing_input(self):
        assert compute_gamma_gap_metrics(pd.DataFrame(columns=["Strike", "GEX"]), 100) is None
        assert compute_gamma_gap_metrics(None, 100) is None

    def test_describe_produces_bullets(self):
        df = _df_net([95, 100, 105], [10, 300, 20])
        metrics = compute_gamma_gap_metrics(df, spot=100, offset=20)
        text = describe_gamma_gap(metrics)
        assert "100.0" in text
        assert text.startswith("- ")


class TestInterpretNetGex:
    def test_reports_magnet_and_flip(self):
        df = _df_net([95, 100, 105], [-100, 50, 200])
        lines = interpret_net_gex(df, S=100)
        joined = "\n".join(lines)
        assert "105.0" in joined  # magnet strike
        assert "Gamma-flip" in joined

    def test_reports_no_flip_when_all_positive(self):
        df = _df_net([95, 100, 105], [100, 50, 200])
        joined = "\n".join(interpret_net_gex(df, S=100))
        assert "No gamma-flip" in joined


class TestBinomialTree:
    def test_node_count(self):
        steps = 5
        tree = generate_binomial_tree(100, 100, 0.5, 0.05, 0.2, steps, "call")
        assert len(tree) == (steps + 1) * (steps + 2) // 2

    def test_terminal_call_payoffs(self):
        steps = 4
        strike = 100.0
        tree = generate_binomial_tree(100, strike, 0.25, 0.05, 0.2, steps, "call")
        terminal = tree[tree["step"] == steps]
        expected = np.maximum(terminal["price"] - strike, 0)
        assert np.allclose(terminal["option"], expected)

    def test_terminal_put_payoffs(self):
        steps = 4
        strike = 100.0
        tree = generate_binomial_tree(100, strike, 0.25, 0.05, 0.2, steps, "put")
        terminal = tree[tree["step"] == steps]
        expected = np.maximum(strike - terminal["price"], 0)
        assert np.allclose(terminal["option"], expected)

    def test_converges_near_black_scholes(self):
        # Black-Scholes reference price for S=100, K=100, T=1, r=5%, sigma=20% -> ~10.45
        tree = generate_binomial_tree(100, 100, 1.0, 0.05, 0.2, 200, "call")
        root_price = tree[(tree["step"] == 0) & (tree["node"] == 0)]["option"].iloc[0]
        assert root_price == pytest.approx(10.45, abs=0.1)

    def test_deep_itm_call_close_to_intrinsic_forward(self):
        S0, K, T, r = 200, 100, 0.5, 0.05
        tree = generate_binomial_tree(S0, K, T, r, 0.2, 100, "call")
        root_price = tree[(tree["step"] == 0) & (tree["node"] == 0)]["option"].iloc[0]
        assert root_price == pytest.approx(S0 - K * math.exp(-r * T), rel=0.02)
