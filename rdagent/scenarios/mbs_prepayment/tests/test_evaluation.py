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

from rdagent.scenarios.mbs_prepayment.evaluation import MBSEvaluationHarness, write_scorecard


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

    features = pd.DataFrame({
        "cusip": cusips,
        "fh_effdt": fh_effdt,
        "coupon": coupons,
        "rate_incentive": rate_incentive,
        "wala": wala,
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
    # Prediction perfectly monotonic in rate_incentive
    y_pred = features["rate_incentive"].to_numpy() * 0.01 + 0.01

    harness = MBSEvaluationHarness()
    scorecard = harness.evaluate(y_true, y_pred, features)
    assert scorecard["rate_sensitivity"]["monotonicity_spearman"] > 0.95


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
