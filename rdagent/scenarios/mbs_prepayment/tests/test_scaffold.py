"""Smoke tests for MBS scaffold (Priority 2)."""
import numpy as np
import pandas as pd
import pytest

from rdagent.scenarios.mbs_prepayment.scaffold import (
    MBSContractViolation,
    MBSDataContract,
    MBSTrainTestSplit,
    MBSWorkflow,
    clip_predictions,
)


@pytest.fixture
def synthetic_raw_df():
    rng = np.random.default_rng(7)
    n = 2000
    df = pd.DataFrame({
        "cusip": [f"CU{i:04d}" for i in rng.integers(0, 80, n)],
        "fh_effdt": pd.date_range("2019-01-01", periods=n, freq="D"),
        "coupon": rng.choice([2.5, 3.5, 4.5, 5.5], n),
        "rate_incentive": rng.normal(0, 1.2, n),
        "wala": rng.integers(1, 120, n),
        "smm_decimal": np.clip(rng.uniform(0, 0.05, n), 0, 1),
    })
    return df


@pytest.mark.offline
def test_contract_rejects_missing_index(synthetic_raw_df):
    df = synthetic_raw_df.drop(columns=["cusip"])
    with pytest.raises(MBSContractViolation, match="Required index columns missing"):
        MBSDataContract().validate(df)


@pytest.mark.offline
def test_contract_rejects_forbidden_columns(synthetic_raw_df):
    df = synthetic_raw_df.copy()
    df["future_smm"] = 0.01
    with pytest.raises(MBSContractViolation, match="Forbidden"):
        MBSDataContract().validate(df)


@pytest.mark.offline
def test_contract_rejects_out_of_range_target(synthetic_raw_df):
    df = synthetic_raw_df.copy()
    df.loc[0, "smm_decimal"] = 1.5
    with pytest.raises(MBSContractViolation, match="out of range"):
        MBSDataContract().validate(df)


@pytest.mark.offline
def test_temporal_split_is_strictly_temporal(synthetic_raw_df):
    splitter = MBSTrainTestSplit(train_end_date="2023-01-01")
    train, test = splitter.split(synthetic_raw_df)
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
    assert clipped[1] == 0.0
    assert clipped[2] == 0.3


@pytest.mark.offline
def test_workflow_end_to_end(synthetic_raw_df):
    """LLM-style feature_fn and model_builder plugged in — scaffold controls everything else."""
    def feature_fn(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["incentive_sq"] = out["rate_incentive"] ** 2
        return out

    class ToyModel:
        def fit(self, X, y):
            self.mean_ = float(np.mean(y))
            return self
        def predict(self, X):
            return np.full(len(X), self.mean_)

    workflow = MBSWorkflow(splitter=MBSTrainTestSplit(train_end_date="2023-01-01"))
    result = workflow.run(synthetic_raw_df, feature_fn, ToyModel)
    assert result["y_pred"].min() >= 0.0
    assert result["y_pred"].max() <= 1.0
    assert len(result["y_true"]) == len(result["y_pred"])
    assert "incentive_sq" in result["feature_columns"]


@pytest.mark.offline
def test_workflow_rejects_leakage_in_llm_features(synthetic_raw_df):
    """LLM feature_fn that injects a forbidden column is rejected by the scaffold."""
    def leaky_feature_fn(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["future_smm"] = out["smm_decimal"].shift(-1).fillna(0)
        return out

    class ToyModel:
        def fit(self, X, y): return self
        def predict(self, X): return np.zeros(len(X))

    workflow = MBSWorkflow(splitter=MBSTrainTestSplit(train_end_date="2023-01-01"))
    with pytest.raises(MBSContractViolation, match="Forbidden"):
        workflow.run(synthetic_raw_df, leaky_feature_fn, ToyModel)
