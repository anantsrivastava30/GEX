"""Golden-value tests for the local Black-Scholes vanna/charm derivation.

Vanna and charm are verified against central finite differences of
``bs_delta``, so any drift in the closed forms shows up immediately.
"""

import numpy as np
import pandas as pd
import pytest

from quant_analysis.analytics.greeks import (
    bs_charm,
    bs_delta,
    bs_vanna,
    compute_max_pain,
    compute_vanna_charm_exposures,
)

CASES = [
    # spot, strike, t, r, sigma, q
    (100.0, 100.0, 0.25, 0.05, 0.20, 0.0),
    (100.0, 110.0, 0.10, 0.05, 0.35, 0.0),
    (450.0, 430.0, 0.50, 0.04, 0.18, 0.013),
    (50.0, 55.0, 1.00, 0.03, 0.60, 0.0),
]


@pytest.mark.parametrize("spot,strike,t,r,sigma,q", CASES)
@pytest.mark.parametrize("option_type", ["call", "put"])
def test_vanna_matches_finite_difference(spot, strike, t, r, sigma, q, option_type):
    h = 1e-5
    fd = (
        bs_delta(spot, strike, t, r, sigma + h, option_type, q)
        - bs_delta(spot, strike, t, r, sigma - h, option_type, q)
    ) / (2 * h)
    assert bs_vanna(spot, strike, t, r, sigma, q) == pytest.approx(fd, rel=1e-5)


@pytest.mark.parametrize("spot,strike,t,r,sigma,q", CASES)
@pytest.mark.parametrize("option_type", ["call", "put"])
def test_charm_matches_finite_difference(spot, strike, t, r, sigma, q, option_type):
    # charm = dDelta/dt with tau = T - t, so dDelta/dt = -dDelta/dtau.
    h = 1e-6
    fd = (
        bs_delta(spot, strike, t - h, r, sigma, option_type, q)
        - bs_delta(spot, strike, t + h, r, sigma, option_type, q)
    ) / (2 * h)
    assert bs_charm(spot, strike, t, r, sigma, option_type, q) == pytest.approx(
        fd, rel=1e-4
    )


def test_vanna_golden_value():
    # Hand-checked ATM case: d1=0.175, d2=0.075, phi(d1)=0.392882,
    # vanna = -phi(d1) * d2 / sigma = -0.392882 * 0.375 = -0.147331.
    assert bs_vanna(100, 100, 0.25, 0.05, 0.20) == pytest.approx(-0.147331, abs=1e-5)


def test_vanna_same_for_calls_and_puts():
    # Put-call parity: dDelta_call/dVol == dDelta_put/dVol.
    for spot, strike, t, r, sigma, q in CASES:
        call_fd = bs_delta(spot, strike, t, r, sigma + 1e-5, "call", q) - bs_delta(
            spot, strike, t, r, sigma - 1e-5, "call", q
        )
        put_fd = bs_delta(spot, strike, t, r, sigma + 1e-5, "put", q) - bs_delta(
            spot, strike, t, r, sigma - 1e-5, "put", q
        )
        assert call_fd == pytest.approx(put_fd, rel=1e-6)


def _chain_frame():
    rows = []
    for strike in (95.0, 100.0, 105.0):
        for option_type in ("call", "put"):
            rows.append(
                {
                    "strike": strike,
                    "option_type": option_type,
                    "open_interest": 100 if option_type == "call" else 40,
                    "contract_size": 100,
                    "mid_iv": 0.22,
                    "DTE": 30,
                }
            )
    return pd.DataFrame(rows)


def test_exposure_aggregation_nets_calls_minus_puts():
    df = _chain_frame()
    out = compute_vanna_charm_exposures(df, spot=100.0, r=0.05)
    assert list(out["strike"]) == [95.0, 100.0, 105.0]

    t = 30 / 365.0
    expected = bs_vanna(100.0, 100.0, t, 0.05, 0.22) * (100 - 40) * 100
    row = out[out["strike"] == 100.0].iloc[0]
    assert row["vanna_exposure"] == pytest.approx(expected, rel=1e-9)

    expected_charm = bs_charm(100.0, 100.0, t, 0.05, 0.22, "call") * 100 * 100
    expected_charm -= bs_charm(100.0, 100.0, t, 0.05, 0.22, "put") * 40 * 100
    assert row["charm_exposure"] == pytest.approx(expected_charm, rel=1e-9)


def test_exposure_offset_filter_and_smv_fallback():
    df = _chain_frame()
    df.loc[df["strike"] == 95.0, "mid_iv"] = None
    df.loc[df["strike"] == 95.0, "smv_vol"] = 0.30
    out = compute_vanna_charm_exposures(df, spot=100.0, r=0.05, offset=2.0)
    assert list(out["strike"]) == [100.0]

    full = compute_vanna_charm_exposures(df, spot=100.0, r=0.05)
    assert 95.0 in set(full["strike"])  # smv_vol filled the gap


def test_exposure_handles_zero_dte_and_empty():
    df = _chain_frame()
    df["DTE"] = 0
    out = compute_vanna_charm_exposures(df, spot=100.0, r=0.05)
    assert not out.empty
    assert np.isfinite(out["vanna_exposure"]).all()

    empty = compute_vanna_charm_exposures(pd.DataFrame(), spot=100.0, r=0.05)
    assert empty.empty


def test_max_pain_textbook_case():
    rows = []
    call_oi = {95.0: 10, 100.0: 50, 105.0: 100}
    put_oi = {95.0: 100, 100.0: 50, 105.0: 10}
    for strike in (95.0, 100.0, 105.0):
        rows.append(
            {"strike": strike, "option_type": "call",
             "open_interest": call_oi[strike], "contract_size": 100}
        )
        rows.append(
            {"strike": strike, "option_type": "put",
             "open_interest": put_oi[strike], "contract_size": 100}
        )
    result = compute_max_pain(pd.DataFrame(rows))
    assert result["max_pain"] == 100.0

    by_strike = {row["strike"]: row for row in result["curve"]}
    assert by_strike[100.0]["total_pain"] == pytest.approx(10_000.0)
    assert by_strike[95.0]["total_pain"] == pytest.approx(35_000.0)
    assert by_strike[105.0]["total_pain"] == pytest.approx(35_000.0)


def test_max_pain_requires_open_interest():
    df = pd.DataFrame(
        [{"strike": 100.0, "option_type": "call", "open_interest": 0}]
    )
    assert compute_max_pain(df) is None
    assert compute_max_pain(pd.DataFrame()) is None
