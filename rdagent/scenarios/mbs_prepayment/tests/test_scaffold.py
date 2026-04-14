"""Smoke tests for the MBS scaffold (single-pkl + scaler layout)."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from rdagent.scenarios.mbs_prepayment.scaffold import (
    GNMA_HARNESS_FEATURES,
    MBSContractViolation,
    MBSDataContract,
    MBSTrainTestSplit,
    MBSWorkflow,
    clip_predictions,
    inverse_transform_features,
)


@pytest.fixture
def synthetic_panel() -> pd.DataFrame:
    """Panel with GNMA feature names already normalized to mean 0 / std 1."""
    rng = np.random.default_rng(7)
    n = 2000
    df = pd.DataFrame(
        {
            "cusip": [f"CU{i:04d}" for i in rng.integers(0, 80, n)],
            "fh_effdt": pd.date_range("2019-01-01", periods=n, freq="D"),
            "smm_decimal": np.clip(rng.uniform(0, 0.05, n), 0, 1),
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
    df.loc[0, "smm_decimal"] = 1.5
    with pytest.raises(MBSContractViolation, match="out of range"):
        MBSDataContract().validate(df)


@pytest.mark.offline
def test_temporal_split_is_strictly_temporal(synthetic_panel):
    splitter = MBSTrainTestSplit(train_end_date="2023-01-01")
    train, test = splitter.split(synthetic_panel)
    assert len(train) > 0
    assert len(test) > 0
    assert pd.to_datetime(train["fh_effdt"]).max() <= pd.Timestamp("2023-01-01")
    assert pd.to_datetime(test["fh_effdt"]).min() > pd.Timestamp("2023-01-01")


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


@pytest.mark.offline
def test_workflow_end_to_end(synthetic_panel):
    class ToyModel:
        def fit(self, X, y):
            self.mean_ = float(np.mean(y))
            return self

        def predict(self, X):
            return np.full(len(X), self.mean_)

    workflow = MBSWorkflow(splitter=MBSTrainTestSplit(train_end_date="2023-01-01"))
    result = workflow.run(synthetic_panel, ToyModel)
    assert result["y_pred"].min() >= 0.0
    assert result["y_pred"].max() <= 1.0
    assert len(result["y_true"]) == len(result["y_pred"])
    assert "WAC" in result["feature_columns"]
    sub = result["submission"]
    assert list(sub.columns) == ["cusip", "fh_effdt", "smm_decimal_pred"]
    assert len(sub) == len(result["y_pred"])
