"""Smoke tests for MBSEvaluationHarness (Priority 1).

These tests verify the scorecard structure and that obvious signals are captured:
- monotonic predictions vs rate_incentive should produce high Spearman
- per-coupon RMSE should be reported for all configured buckets
- scorecard should be JSON-serializable
"""
import json

import numpy as np
import pandas as pd
import pytest

from rdagent.scenarios.mbs_prepayment.evaluation import (
    MBSEvaluationHarness,
    _compute_s_curve_rmse,
    write_scorecard,
)


@pytest.fixture
def synthetic_mbs_data():
    rng = np.random.default_rng(42)
    n = 500
    cusips = [f"CU{i:04d}" for i in rng.integers(0, 50, n)]
    fh_effdt = pd.date_range("2020-01-01", periods=n, freq="D")
    coupons = rng.choice([2.5, 3.25, 3.75, 4.25, 4.75, 5.25], size=n)
    rate_incentive = rng.normal(0, 1.0, n)  # percent incentive
    wala = rng.integers(1, 120, n)
    # Ground truth: sigmoid in rate_incentive
    true_smm = 0.005 + 0.03 / (1 + np.exp(-2.0 * rate_incentive))
    noise = rng.normal(0, 0.002, n)
    y_true = np.clip(true_smm + noise, 0.0, 1.0)

    # Column names must match gnma_feature.md and MBSEvaluationHarness defaults
    # fh_effdt as integer YYYYMMDD to match the real panel format
    features = pd.DataFrame({
        "cusip": cusips,
        "fh_effdt": fh_effdt.strftime("%Y%m%d").astype(int),
        "WAC": coupons,
        "Avg_Prop_Refi_Incentive_WAC_30yr_2mos": rate_incentive,
        "WALA": wala,
    })
    return features, y_true


@pytest.mark.offline
def test_scorecard_structure(synthetic_mbs_data):
    features, y_true = synthetic_mbs_data
    # Use a good predictor so monotonicity is positive
    y_pred = y_true + np.random.default_rng(1).normal(0, 0.001, len(y_true))

    harness = MBSEvaluationHarness()
    scorecard = harness.evaluate(y_true, y_pred, features)

    assert "accuracy" in scorecard
    assert "rate_sensitivity" in scorecard
    assert "temporal_robustness" in scorecard
    assert "structural_properties" in scorecard
    assert "primary_metric" in scorecard
    assert scorecard["primary_metric"]["name"] == "overall_rmse"
    assert scorecard["primary_metric"]["lower_is_better"] is True


@pytest.mark.offline
def test_per_coupon_rmse_all_buckets_reported(synthetic_mbs_data):
    features, y_true = synthetic_mbs_data
    y_pred = y_true + np.random.default_rng(2).normal(0, 0.001, len(y_true))

    harness = MBSEvaluationHarness()
    scorecard = harness.evaluate(y_true, y_pred, features)

    per_coupon = scorecard["accuracy"]["rmse_by_coupon_bucket"]
    # All 6 default buckets should be reported (some may be NaN if empty)
    assert len(per_coupon) == 6
    # At least 4 of the 6 should have finite values on 500 random rows
    finite_count = sum(1 for v in per_coupon.values() if not np.isnan(v))
    assert finite_count >= 4


@pytest.mark.offline
def test_monotonicity_detected_for_good_model(synthetic_mbs_data):
    features, y_true = synthetic_mbs_data
    # Prediction perfectly monotonic in Avg_Prop_Refi_Incentive_WAC_30yr_2mos
    y_pred = features["Avg_Prop_Refi_Incentive_WAC_30yr_2mos"].to_numpy() * 0.01 + 0.01

    harness = MBSEvaluationHarness()
    scorecard = harness.evaluate(y_true, y_pred, features)
    assert scorecard["rate_sensitivity"]["monotonicity_spearman"] > 0.95


@pytest.mark.offline
def test_s_curve_rmse_zero_for_perfect_predictions():
    rng = np.random.default_rng(0)
    n = 6_000
    incentive = rng.uniform(0.4, 1.9, n)
    y_true = 0.005 + 0.05 / (1 + np.exp(-8.0 * (incentive - 1.05)))
    y_pred = y_true.copy()  # perfect
    upb = rng.uniform(1e6, 5e7, n)

    out = _compute_s_curve_rmse(
        y_true=y_true,
        y_pred=y_pred,
        incentive=incentive,
        upb=upb,
        bin_edges=[0.0, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, np.inf],
        upb_cap=150e6,
        min_rows_per_bin=30,
        left_tail_max=0.9,
        right_tail_min=1.4,
    )

    assert out["s_curve_bins_used"] >= 8
    assert out["s_curve_rmse_overall"] < 1e-12
    assert out["s_curve_rmse_mid_belly"] < 1e-12


@pytest.mark.offline
def test_s_curve_rmse_segments_isolate_error_location():
    rng = np.random.default_rng(1)
    n = 8_000
    incentive = rng.uniform(0.4, 1.9, n)
    y_true = 0.005 + 0.05 / (1 + np.exp(-8.0 * (incentive - 1.05)))
    # Bias predictions only in the mid-belly bins (0.9 ≤ x < 1.4)
    bias = np.where((incentive >= 0.9) & (incentive < 1.4), 0.020, 0.0)
    y_pred = y_true + bias

    out = _compute_s_curve_rmse(
        y_true=y_true,
        y_pred=y_pred,
        incentive=incentive,
        upb=None,  # exercise the unweighted fallback
        bin_edges=[0.0, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, np.inf],
        upb_cap=150e6,
        min_rows_per_bin=30,
        left_tail_max=0.9,
        right_tail_min=1.4,
    )

    # Mid-belly bias of 0.02 should dominate; tails should be near zero.
    assert out["s_curve_rmse_mid_belly"] > 0.015
    assert out["s_curve_rmse_left_tail"] < 1e-9
    assert out["s_curve_rmse_right_tail"] < 1e-9
    assert out["s_curve_rmse_overall"] > out["s_curve_rmse_left_tail"]


@pytest.mark.offline
def test_s_curve_rmse_drops_sparse_bins():
    rng = np.random.default_rng(2)
    # 200 rows in mid bins, only 5 rows in the [0, 0.6) left-tail bin
    incentive = np.concatenate([
        rng.uniform(0.9, 1.4, 200),
        rng.uniform(0.0, 0.55, 5),
    ])
    y_true = np.full_like(incentive, 0.01)
    y_pred = np.full_like(incentive, 0.012)

    out = _compute_s_curve_rmse(
        y_true=y_true,
        y_pred=y_pred,
        incentive=incentive,
        upb=None,
        bin_edges=[0.0, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, np.inf],
        upb_cap=150e6,
        min_rows_per_bin=30,
        left_tail_max=0.9,
        right_tail_min=1.4,
    )

    # Sparse left-tail bin (5 rows) is dropped -> left_tail RMSE is NaN.
    assert np.isnan(out["s_curve_rmse_left_tail"])
    # Mid belly is densely populated -> finite, equals the |0.002| bias.
    assert abs(out["s_curve_rmse_mid_belly"] - 0.002) < 1e-9


@pytest.mark.offline
def test_scorecard_json_serializable(synthetic_mbs_data, tmp_path):
    features, y_true = synthetic_mbs_data
    y_pred = y_true + np.random.default_rng(3).normal(0, 0.001, len(y_true))
    harness = MBSEvaluationHarness()
    scorecard = harness.evaluate(y_true, y_pred, features)

    out = tmp_path / "scorecard.json"
    write_scorecard(scorecard, str(out))
    reloaded = json.loads(out.read_text())
    assert reloaded["primary_metric"]["name"] == "overall_rmse"
