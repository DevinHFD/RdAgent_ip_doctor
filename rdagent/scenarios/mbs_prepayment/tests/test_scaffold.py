"""Smoke tests for the MBS scaffold (single-pkl + scaler layout)."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from rdagent.scenarios.mbs_prepayment.scaffold import (
    GNMA_HARNESS_FEATURES,
    MBSContractViolation,
    MBSCUSIPSplit,
    MBSDataContract,
    MBSTrainTestSplit,
    MBSWorkflow,
    _append_test_predictions,
    clip_predictions,
    inverse_transform_features,
)


@pytest.fixture
def synthetic_panel() -> pd.DataFrame:
    """Panel with GNMA feature names already normalized to mean 0 / std 1.

    Uses 100 distinct CUSIPs spread across 2019–2024 so the CUSIP split
    always produces non-empty train/val/test partitions.
    """
    rng = np.random.default_rng(7)
    n = 3000
    dates = pd.date_range("2019-01-01", periods=n, freq="D")
    df = pd.DataFrame(
        {
            "cusip": [f"CU{i:04d}" for i in rng.integers(0, 100, n)],
            "fh_effdt": dates.strftime("%Y%m%d").astype(int),
            "fh_upb": rng.uniform(1e6, 200e6, n),
            "SMM_DECIMAL": np.clip(rng.uniform(0, 0.05, n), 0, 1),
        }
    )
    for f in GNMA_HARNESS_FEATURES:
        df[f] = rng.normal(0, 1, n)
    return df


@pytest.fixture
def synthetic_scaler() -> SimpleNamespace:
    """Mimic a fitted sklearn StandardScaler surface."""
    feats = list(GNMA_HARNESS_FEATURES)
    return SimpleNamespace(
        mean_=np.arange(1, len(feats) + 1, dtype=float),
        scale_=np.full(len(feats), 2.0),
        feature_names_in_=np.array(feats),
    )


@pytest.mark.offline
def test_contract_rejects_missing_gnma_feature(synthetic_panel):
    df = synthetic_panel.drop(columns=["WAC"])
    with pytest.raises(MBSContractViolation, match="required GNMA feature columns"):
        MBSDataContract().validate(df)


@pytest.mark.offline
def test_contract_rejects_forbidden_columns(synthetic_panel):
    df = synthetic_panel.copy()
    df["future_smm"] = 0.01
    with pytest.raises(MBSContractViolation, match="forbidden"):
        MBSDataContract().validate(df)


@pytest.mark.offline
def test_contract_rejects_out_of_range_target(synthetic_panel):
    df = synthetic_panel.copy()
    df.loc[0, "SMM_DECIMAL"] = 1.5
    with pytest.raises(MBSContractViolation, match="out of range"):
        MBSDataContract().validate(df)


# ---------------------------------------------------------------------------
# MBSCUSIPSplit
# ---------------------------------------------------------------------------


@pytest.mark.offline
def test_cusip_split_produces_nonempty_partitions(synthetic_panel):
    splitter = MBSCUSIPSplit(train_end_date="2023-01-01", random_seed=42)
    train, val, test = splitter.split(synthetic_panel)
    assert len(train) > 0
    assert len(val) > 0
    assert len(test) > 0


@pytest.mark.offline
def test_cusip_split_cusips_are_disjoint(synthetic_panel):
    splitter = MBSCUSIPSplit(train_end_date="2023-01-01", random_seed=42)
    train, val, test = splitter.split(synthetic_panel)
    t_cusips = set(train["cusip"].unique())
    v_cusips = set(val["cusip"].unique())
    s_cusips = set(test["cusip"].unique())
    assert t_cusips.isdisjoint(v_cusips)
    assert t_cusips.isdisjoint(s_cusips)
    assert v_cusips.isdisjoint(s_cusips)


@pytest.mark.offline
def test_cusip_split_test_has_all_time_rows(synthetic_panel):
    splitter = MBSCUSIPSplit(train_end_date="2023-01-01", random_seed=42)
    _train, _val, test = splitter.split(synthetic_panel)
    # Test CUSIPs should include rows from BOTH before and after the cutoff.
    test_dates = pd.to_datetime(test["fh_effdt"], format="%Y%m%d")
    assert test_dates.min() < pd.Timestamp("2023-01-01")
    assert test_dates.max() >= pd.Timestamp("2023-01-01")


@pytest.mark.offline
def test_cusip_split_train_val_temporal_cutoff(synthetic_panel):
    splitter = MBSCUSIPSplit(train_end_date="2022-06-01", random_seed=42)
    train, val, _test = splitter.split(synthetic_panel)
    assert pd.to_datetime(train["fh_effdt"], format="%Y%m%d").max() <= pd.Timestamp("2022-06-01")
    assert pd.to_datetime(val["fh_effdt"], format="%Y%m%d").max() <= pd.Timestamp("2022-06-01")


@pytest.mark.offline
def test_cusip_split_is_deterministic(synthetic_panel):
    s1 = MBSCUSIPSplit(train_end_date="2023-01-01", random_seed=99)
    s2 = MBSCUSIPSplit(train_end_date="2023-01-01", random_seed=99)
    _, _, test1 = s1.split(synthetic_panel)
    _, _, test2 = s2.split(synthetic_panel)
    pd.testing.assert_frame_equal(test1.reset_index(drop=True), test2.reset_index(drop=True))


@pytest.mark.offline
def test_cusip_split_different_seeds_give_different_test_sets(synthetic_panel):
    _, _, test_a = MBSCUSIPSplit(train_end_date="2023-01-01", random_seed=1).split(synthetic_panel)
    _, _, test_b = MBSCUSIPSplit(train_end_date="2023-01-01", random_seed=2).split(synthetic_panel)
    cusips_a = set(test_a["cusip"].unique())
    cusips_b = set(test_b["cusip"].unique())
    assert cusips_a != cusips_b


@pytest.mark.offline
def test_cusip_split_test_fraction(synthetic_panel):
    splitter = MBSCUSIPSplit(train_end_date="2023-01-01", test_fraction=1 / 7, random_seed=42)
    _train, _val, test = splitter.split(synthetic_panel)
    n_total = synthetic_panel["cusip"].nunique()
    n_test = test["cusip"].nunique()
    expected = round(n_total / 7)
    assert abs(n_test - expected) <= 1


# ---------------------------------------------------------------------------
# Legacy MBSTrainTestSplit (kept for backward compat)
# ---------------------------------------------------------------------------


@pytest.mark.offline
def test_temporal_split_is_strictly_temporal(synthetic_panel):
    splitter = MBSTrainTestSplit(train_end_date="2023-01-01")
    train, test = splitter.split(synthetic_panel)
    assert len(train) > 0
    assert len(test) > 0
    assert pd.to_datetime(train["fh_effdt"], format="%Y%m%d").max() <= pd.Timestamp("2023-01-01")
    assert pd.to_datetime(test["fh_effdt"], format="%Y%m%d").min() > pd.Timestamp("2023-01-01")


# ---------------------------------------------------------------------------
# Clip / inverse-transform
# ---------------------------------------------------------------------------


@pytest.mark.offline
def test_clip_predictions_enforces_range():
    contract = MBSDataContract()
    raw = np.array([-0.5, 0.0, 0.3, 1.5, np.nan])
    clipped = clip_predictions(raw, contract)
    assert clipped[0] == 0.0
    assert clipped[3] == 1.0
    assert clipped[2] == 0.3


@pytest.mark.offline
def test_inverse_transform_reverses_normalization(synthetic_panel, synthetic_scaler):
    raw = inverse_transform_features(
        synthetic_panel, synthetic_scaler, ["WAC", "WALA"]
    )
    wac_idx = list(synthetic_scaler.feature_names_in_).index("WAC")
    expected = (
        synthetic_panel["WAC"].to_numpy() * synthetic_scaler.scale_[wac_idx]
        + synthetic_scaler.mean_[wac_idx]
    )
    np.testing.assert_allclose(raw["WAC"].to_numpy(), expected)


# ---------------------------------------------------------------------------
# MBSWorkflow end-to-end
# ---------------------------------------------------------------------------


@pytest.mark.offline
def test_workflow_end_to_end(synthetic_panel):
    class ToyModel:
        def fit(self, X, y, **kwargs):
            self.mean_ = float(np.mean(y))
            return self

        def predict(self, X):
            return np.full(len(X), self.mean_)

    workflow = MBSWorkflow(splitter=MBSCUSIPSplit(train_end_date="2023-01-01"))
    result = workflow.run(synthetic_panel, ToyModel)
    assert result["y_pred"].min() >= 0.0
    assert result["y_pred"].max() <= 1.0
    assert len(result["y_true"]) == len(result["y_pred"])
    assert "WAC" in result["feature_columns"]
    sub = result["submission"]
    assert list(sub.columns) == ["cusip", "fh_effdt", "smm_decimal_pred"]  # output always lowercase
    assert len(sub) == len(result["y_pred"])
    # Workflow returns all three splits.
    assert "train_df" in result
    assert "val_df" in result
    assert "test_df" in result


@pytest.mark.offline
def test_workflow_passes_val_data_to_model(synthetic_panel):
    """Model that records whether X_val was received."""
    received: dict = {}

    class ValCapture:
        def fit(self, X, y, X_val=None, y_val=None, sample_weight=None):
            received["X_val"] = X_val
            received["y_val"] = y_val
            received["sample_weight"] = sample_weight
            self.mean_ = float(np.mean(y))

        def predict(self, X):
            return np.full(len(X), self.mean_)

    workflow = MBSWorkflow(splitter=MBSCUSIPSplit(train_end_date="2023-01-01"))
    workflow.run(synthetic_panel, ValCapture)
    assert received.get("X_val") is not None
    assert len(received["X_val"]) > 0
    # fh_upb present → sample_weight should be set.
    assert received.get("sample_weight") is not None


@pytest.mark.offline
def test_workflow_falls_back_for_sklearn_model(synthetic_panel):
    """Pure sklearn models that reject X_val should not crash."""
    from sklearn.linear_model import Ridge

    workflow = MBSWorkflow(splitter=MBSCUSIPSplit(train_end_date="2023-01-01"))
    result = workflow.run(synthetic_panel, Ridge)
    assert len(result["y_pred"]) > 0


# ---------------------------------------------------------------------------
# _append_test_predictions
# ---------------------------------------------------------------------------


@pytest.mark.offline
def test_append_test_predictions_creates_file(synthetic_panel, tmp_path):
    contract = MBSDataContract()
    test_df = synthetic_panel.head(50).copy()
    y_pred = np.random.default_rng(0).uniform(0, 0.05, len(test_df))
    out = _append_test_predictions(tmp_path, test_df, y_pred, contract)
    assert out.exists()
    df = pd.read_csv(out)
    assert "loop_number" in df.columns
    assert "smm_decimal_pred" in df.columns
    assert df["loop_number"].iloc[0] == 1
    assert len(df) == len(test_df)


@pytest.mark.offline
def test_append_test_predictions_increments_loop_number(synthetic_panel, tmp_path):
    contract = MBSDataContract()
    test_df = synthetic_panel.head(30).copy()
    y1 = np.zeros(len(test_df))
    y2 = np.ones(len(test_df)) * 0.01

    _append_test_predictions(tmp_path, test_df, y1, contract)
    _append_test_predictions(tmp_path, test_df, y2, contract)

    hist = pd.read_csv(tmp_path / "test_predictions_history.csv")
    assert set(hist["loop_number"].unique()) == {1, 2}
    assert len(hist) == 2 * len(test_df)
