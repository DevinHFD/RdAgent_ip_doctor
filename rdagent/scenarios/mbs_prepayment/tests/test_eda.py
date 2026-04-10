"""Tests for MBS domain-specific EDA (Priority 5)."""
import numpy as np
import pandas as pd
import pytest

from rdagent.scenarios.mbs_prepayment.eda import profile_mbs_panel, write_eda_report


@pytest.fixture
def synthetic_panel():
    rng = np.random.default_rng(42)
    cusips = [f"CU{i:04d}" for i in range(30)]
    dates = pd.date_range("2019-01-01", "2023-12-01", freq="MS")
    rows = []
    for c in cusips:
        # Persistent SMM_DECIMAL with seasonal + rate effects
        base = rng.uniform(0.005, 0.015)
        for t, d in enumerate(dates):
            # Fake rate incentive that changes over time
            rate_incentive = 0.5 * np.sin(t / 12 * np.pi) + rng.normal(0, 0.3)
            smm = np.clip(base + 0.01 * max(0, rate_incentive) + rng.normal(0, 0.002), 0.0, 0.5)
            rows.append({
                "cusip": c,
                "fh_effdt": d,
                "coupon": rng.choice([2.5, 3.5, 4.5, 5.5]),
                "wala": t + 12,
                "rate_incentive": rate_incentive,
                "smm_decimal": smm,
            })
    return pd.DataFrame(rows)


@pytest.mark.offline
def test_profile_basic_fields(synthetic_panel):
    profile = profile_mbs_panel(synthetic_panel)
    assert profile.n_cusips == 30
    assert profile.n_dates == 60
    assert profile.smm_mean > 0
    assert profile.smm_p99 >= profile.smm_p95


@pytest.mark.offline
def test_coupon_distribution_sums_to_one(synthetic_panel):
    profile = profile_mbs_panel(synthetic_panel)
    total = sum(profile.coupon_distribution.values())
    assert abs(total - 1.0) < 0.01


@pytest.mark.offline
def test_autocorrelation_reasonable(synthetic_panel):
    """Panel is designed with a persistent base, so autocorr should be positive."""
    profile = profile_mbs_panel(synthetic_panel)
    assert profile.smm_autocorr_lag1 > 0.0


@pytest.mark.offline
def test_itm_fraction_by_year(synthetic_panel):
    profile = profile_mbs_panel(synthetic_panel)
    assert 2019 in profile.itm_fraction_by_year
    assert 2023 in profile.itm_fraction_by_year
    for v in profile.itm_fraction_by_year.values():
        assert 0.0 <= v <= 1.0


@pytest.mark.offline
def test_structural_breaks_within_range(synthetic_panel):
    profile = profile_mbs_panel(synthetic_panel)
    # 2020-03 and 2022-03 are within 2019-2023 range
    dates = [b["date"] for b in profile.structural_breaks]
    assert "2020-03-01" in dates
    assert "2022-03-01" in dates


@pytest.mark.offline
def test_feature_correlations_include_rate_incentive(synthetic_panel):
    profile = profile_mbs_panel(synthetic_panel)
    # rate_incentive is built into smm, should show positive correlation
    assert "rate_incentive" in profile.feature_correlations
    assert profile.feature_correlations["rate_incentive"] > 0


@pytest.mark.offline
def test_markdown_output_readable(synthetic_panel):
    profile = profile_mbs_panel(synthetic_panel)
    md = profile.to_markdown()
    assert "## MBS Data Profile" in md
    assert "SMM_DECIMAL" in md
    assert "Panel dimensions" in md
    assert "Coupon distribution" in md


@pytest.mark.offline
def test_write_eda_report(synthetic_panel, tmp_path):
    profile = profile_mbs_panel(synthetic_panel)
    out = tmp_path / "eda.md"
    write_eda_report(profile, out)
    assert out.exists()
    content = out.read_text()
    assert "MBS Data Profile" in content
