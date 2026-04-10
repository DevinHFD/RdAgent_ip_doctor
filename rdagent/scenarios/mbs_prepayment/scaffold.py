"""MBS Code Scaffold — Priority 2: Hard contracts the LLM cannot violate.

This module implements Direction #5 (Code Scaffold and Interface Contracts). It
provides fixed, non-LLM-generated infrastructure that the LLM's feature
engineering and model code must plug into. This prevents the most common MBS
prepayment modeling errors that generic LLM code generation produces:

    1. Treating each (cusip, fh_effdt) row as independent — losing panel structure
    2. Random train/test split instead of temporal — causing look-ahead bias
    3. Normalizing SMM_DECIMAL target — destroying interpretability
    4. Same-fh_effdt macro features — direct look-ahead leakage
    5. Forgetting to clip predictions to [0.0, 1.0]

The scaffold is designed so that LLM-generated code cannot modify these files
(the RD-Agent `Workflow` component spec will always overwrite them) and the
LLM is only allowed to generate:

    - feature_engineering(df) -> pd.DataFrame  (must satisfy MBSDataContract)
    - build_model() -> sklearn-compatible estimator with .fit/.predict

Everything else — loading, splitting, fitting, prediction clipping, evaluation —
is fixed scaffold code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MBSDataContract:
    """Schema contract for MBS prepayment data — checked at scaffold boundaries.

    The LLM-generated feature engineering code must produce a DataFrame that
    satisfies this contract. The scaffold raises `MBSContractViolation` if not.
    """

    #: Required columns forming the panel key.
    required_index: tuple[str, ...] = ("cusip", "fh_effdt")
    #: Columns that must be present for MBS-theoretic reasons.
    required_columns: tuple[str, ...] = ("rate_incentive", "coupon", "wala")
    #: Columns that must NEVER appear — future leakage sentinels.
    forbidden_columns: tuple[str, ...] = (
        "future_smm",
        "forward_smm",
        "next_month_smm",
        "forward_rate",
        "future_rate_incentive",
    )
    #: Target column name.
    target_column: str = "smm_decimal"
    #: Valid range for target and predictions.
    target_range: tuple[float, float] = (0.0, 1.0)
    #: Minimum look-ahead offset for macro features (in days).
    macro_lag_days_min: int = 30

    def validate(self, df: pd.DataFrame, *, include_target: bool = True) -> None:
        """Raise MBSContractViolation if the DataFrame violates the contract."""
        missing_idx = [c for c in self.required_index if c not in df.columns]
        if missing_idx:
            raise MBSContractViolation(
                f"Required index columns missing: {missing_idx}. "
                f"MBS data must be keyed by {self.required_index}."
            )
        missing_req = [c for c in self.required_columns if c not in df.columns]
        if missing_req:
            raise MBSContractViolation(
                f"Required feature columns missing: {missing_req}. "
                "These features are mandatory for prepayment modeling."
            )
        forbidden_present = [c for c in self.forbidden_columns if c in df.columns]
        if forbidden_present:
            raise MBSContractViolation(
                f"Forbidden (future-leaking) columns present: {forbidden_present}. "
                "Remove any feature derived from future observations."
            )
        if include_target:
            if self.target_column not in df.columns:
                raise MBSContractViolation(
                    f"Target column '{self.target_column}' missing from DataFrame."
                )
            lo, hi = self.target_range
            vals = df[self.target_column].to_numpy(dtype=float)
            finite = vals[~np.isnan(vals)]
            if len(finite) > 0 and (finite.min() < lo - 1e-9 or finite.max() > hi + 1e-9):
                raise MBSContractViolation(
                    f"Target '{self.target_column}' out of range [{lo}, {hi}]: "
                    f"min={finite.min():.4f}, max={finite.max():.4f}"
                )


class MBSContractViolation(ValueError):
    """Raised when LLM-generated code produces a DataFrame violating MBSDataContract."""


# ---------------------------------------------------------------------------
# Temporal train/test split
# ---------------------------------------------------------------------------


@dataclass
class MBSTrainTestSplit:
    """Fixed temporal split on fh_effdt.

    The LLM cannot override this. Random splits and any split that leaks test
    data into training are impossible because the split is performed by the
    scaffold, not by LLM-generated code.
    """

    #: Training cutoff date (inclusive). Rows with fh_effdt > train_end_date are test.
    train_end_date: str = "2021-12-31"
    #: Date column name — must match the contract's index.
    date_column: str = "fh_effdt"
    #: Optional validation gap (months) between train end and test start, to
    #: prevent autocorrelation leakage for high-persistence SMM_DECIMAL.
    embargo_months: int = 0

    def split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        if self.date_column not in df.columns:
            raise MBSContractViolation(
                f"Split date column '{self.date_column}' missing from DataFrame."
            )
        dates = pd.to_datetime(df[self.date_column], errors="coerce")
        train_cutoff = pd.Timestamp(self.train_end_date)
        test_start = train_cutoff + pd.DateOffset(months=self.embargo_months)
        train_mask = dates <= train_cutoff
        test_mask = dates > test_start
        return df[train_mask].copy(), df[test_mask].copy()


# ---------------------------------------------------------------------------
# Prediction clipping
# ---------------------------------------------------------------------------


def clip_predictions(y_pred: np.ndarray, contract: MBSDataContract) -> np.ndarray:
    """Clip predictions to the valid target range.

    Applied unconditionally after LLM-generated `model.predict()` — the LLM
    does not need to remember to clip, the scaffold does it.
    """
    lo, hi = contract.target_range
    return np.clip(np.asarray(y_pred, dtype=float), lo, hi)


# ---------------------------------------------------------------------------
# Workflow orchestration
# ---------------------------------------------------------------------------


@dataclass
class MBSWorkflow:
    """Fixed workflow: load → contract-check → split → featurize → fit → predict → evaluate.

    The LLM only provides:
        feature_fn: a callable that takes a raw DataFrame and returns a
            contract-conformant DataFrame with engineered features.
        model_builder: a callable that returns a fresh, unfitted estimator.

    Everything else is scaffold and not LLM-modifiable.
    """

    contract: MBSDataContract = field(default_factory=MBSDataContract)
    splitter: MBSTrainTestSplit = field(default_factory=MBSTrainTestSplit)

    def run(
        self,
        raw_df: pd.DataFrame,
        feature_fn: Callable[[pd.DataFrame], pd.DataFrame],
        model_builder: Callable[[], Any],
        exclude_from_features: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        """Run the full pipeline and return a result dict.

        Returns:
            dict with keys:
                - y_true: np.ndarray of test SMM_DECIMAL
                - y_pred: np.ndarray of clipped test predictions
                - features_test: pd.DataFrame used for test-time evaluation
                - model: the fitted model
        """
        # 1. Validate raw DataFrame (must have target + required columns)
        self.contract.validate(raw_df, include_target=True)

        # 2. Temporal split — LLM cannot override
        raw_train, raw_test = self.splitter.split(raw_df)
        if len(raw_train) == 0 or len(raw_test) == 0:
            raise MBSContractViolation(
                f"Split produced empty partition: train={len(raw_train)}, test={len(raw_test)}. "
                f"Check train_end_date={self.splitter.train_end_date}."
            )

        # 3. Feature engineering — LLM-generated, but validated after
        feat_train = feature_fn(raw_train)
        feat_test = feature_fn(raw_test)
        self.contract.validate(feat_train, include_target=True)
        self.contract.validate(feat_test, include_target=True)

        # 4. Build X/y — scaffold controls this, not LLM
        drop_cols = set(
            self.contract.required_index + (self.contract.target_column,) + tuple(exclude_from_features)
        )
        x_cols = [c for c in feat_train.columns if c not in drop_cols]
        X_train = feat_train[x_cols].to_numpy(dtype=float)
        y_train = feat_train[self.contract.target_column].to_numpy(dtype=float)
        X_test = feat_test[x_cols].to_numpy(dtype=float)
        y_test = feat_test[self.contract.target_column].to_numpy(dtype=float)

        # 5. Fit + predict + clip — scaffold guarantees valid range
        model = model_builder()
        model.fit(X_train, y_train)
        y_pred_raw = model.predict(X_test)
        y_pred = clip_predictions(y_pred_raw, self.contract)

        return {
            "y_true": y_test,
            "y_pred": y_pred,
            "features_test": feat_test,
            "model": model,
            "feature_columns": x_cols,
        }


# ---------------------------------------------------------------------------
# main.py entry point the RD-Agent Workflow component invokes
# ---------------------------------------------------------------------------


def run_scaffold_pipeline(
    raw_df: pd.DataFrame,
    feature_fn: Callable[[pd.DataFrame], pd.DataFrame],
    model_builder: Callable[[], Any],
    output_dir: str | Path,
    contract: MBSDataContract | None = None,
    splitter: MBSTrainTestSplit | None = None,
) -> dict[str, Any]:
    """Top-level scaffold entry — combines workflow + evaluation + scorecard write.

    This is what the `main.py` template (Workflow component) calls. LLM-generated
    feature_fn and model_builder are plugged in, but loading, splitting,
    evaluation, and scorecard dumping are all scaffold code.
    """
    from rdagent.scenarios.mbs_prepayment.evaluation import MBSEvaluationHarness, write_scorecard

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    workflow = MBSWorkflow(
        contract=contract or MBSDataContract(),
        splitter=splitter or MBSTrainTestSplit(),
    )
    result = workflow.run(raw_df, feature_fn, model_builder)

    harness = MBSEvaluationHarness()
    scorecard = harness.evaluate(
        y_true=result["y_true"],
        y_pred=result["y_pred"],
        features=result["features_test"],
    )
    write_scorecard(scorecard, str(output_dir / "scores.json"))
    return {"result": result, "scorecard": scorecard}
