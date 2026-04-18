"""MBS Code Scaffold — Priority 2: Hard contracts the LLM cannot violate.

This scaffold assumes the single-panel layout:

    <data_dir>/<competition>/
        tfminput.pkl   # Full CUSIP-level monthly panel, ALL feature columns
                       # stored in normalized form (mean 0, std 1). Also
                       # carries cusip, fh_effdt, and the target
                       # smm_decimal (which is NOT normalized, it lives in
                       # [0, 1] already).
        scaler.sav     # joblib-saved fitted sklearn-style scaler. Inverse-
                       # transforms the GNMA features listed in
                       # example/gnma_feature.md (WAC, WALA,
                       # Avg_Prop_Refi_Incentive_WAC_30yr_2mos, burnout
                       # features, etc.) back to their raw (percent / month)
                       # scale — the MBS evaluation harness needs raw WAC
                       # for coupon-bucket RMSE and raw refi-incentive for
                       # S-curve R² and inflection-point diagnostics.
        description.md
        sample_submission.csv

The coder plugs in one LLM-controlled callable::

    build_model() -> sklearn-compatible estimator with .fit/.predict

Loading, temporal splitting, scaler inverse-transform, prediction clipping,
submission writing, and MBS scorecard evaluation are all scaffold code —
the LLM cannot modify them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data contract
# ---------------------------------------------------------------------------


#: Default GNMA feature columns (from example/gnma_feature.md) that the scaler
#: inverse-transforms back to raw units for the evaluation harness.
GNMA_HARNESS_FEATURES: tuple[str, ...] = (
    "WAC",
    "WALA",
    "Avg_Prop_Refi_Incentive_WAC_30yr_2mos",
    "Avg_Prop_Switch_To_15yr_Incentive_2mos",
    "Burnout_Prop_WAC_30yr_log_sum60",
    "Burnout_Prop_30yr_Switch_to_15_Lag1",
    "CLTV",
    "SATO",
    "Pool_HPA_2yr",
)


@dataclass(frozen=True)
class MBSDataContract:
    """Schema contract for the MBS panel pickle.

    The panel ships normalized (mean 0, std 1) for every feature column, but
    the scorecard needs raw-scale values for a subset of GNMA features. That
    subset is listed in ``harness_raw_features`` and is inverse-transformed
    by the scaffold using ``scaler.sav`` before evaluation.
    """

    cusip_col: str = "cusip"
    date_col: str = "fh_effdt"
    target_col: str = "smm_decimal"
    target_range: tuple[float, float] = (0.0, 1.0)

    #: Columns that must be present in the panel (normalized scale is fine —
    #: these names are looked up by the scaffold and by the harness after
    #: inverse-transform).
    required_columns: tuple[str, ...] = GNMA_HARNESS_FEATURES

    #: GNMA feature columns the scaler can invert — these drive the scorecard.
    harness_raw_features: tuple[str, ...] = GNMA_HARNESS_FEATURES

    #: Columns that must never appear — future-leakage sentinels.
    #: CPR_DECIMAL is the annualised form of the target SMM_DECIMAL
    #: (CPR = 1 - (1 - SMM)^12), so using it as a feature is direct
    #: target leakage and is forbidden unconditionally.
    forbidden_columns: tuple[str, ...] = (
        "future_smm",
        "forward_smm",
        "next_month_smm",
        "forward_rate",
        "future_rate_incentive",
        "CPR_DECIMAL",
    )

    def validate(self, df: pd.DataFrame, *, include_target: bool = True) -> None:
        for c in (self.cusip_col, self.date_col):
            if c not in df.columns:
                raise MBSContractViolation(
                    f"Panel missing required index column '{c}'."
                )
        missing = [c for c in self.required_columns if c not in df.columns]
        if missing:
            raise MBSContractViolation(
                f"Panel missing required GNMA feature columns: {missing}. "
                "These must appear in tfminput.pkl (normalized scale is fine)."
            )
        forbidden = [c for c in self.forbidden_columns if c in df.columns]
        if forbidden:
            raise MBSContractViolation(
                f"Panel contains forbidden (future-leaking) columns: {forbidden}."
            )
        if include_target:
            if self.target_col not in df.columns:
                raise MBSContractViolation(
                    f"Panel missing target column '{self.target_col}'."
                )
            vals = df[self.target_col].to_numpy(dtype=float)
            finite = vals[~np.isnan(vals)]
            lo, hi = self.target_range
            if len(finite) > 0 and (finite.min() < lo - 1e-9 or finite.max() > hi + 1e-9):
                raise MBSContractViolation(
                    f"Target '{self.target_col}' out of range [{lo}, {hi}]: "
                    f"min={finite.min():.4f}, max={finite.max():.4f} — the "
                    "target must ship unnormalized in decimal form."
                )


class MBSContractViolation(ValueError):
    """Raised when the MBS panel or scaler violates MBSDataContract."""


# ---------------------------------------------------------------------------
# Temporal train/test split
# ---------------------------------------------------------------------------


@dataclass
class MBSTrainTestSplit:
    """Fixed temporal split on fh_effdt. LLM cannot override."""

    train_end_date: str = "2024-10-31"
    date_column: str = "fh_effdt"
    embargo_months: int = 0

    def split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        if self.date_column not in df.columns:
            raise MBSContractViolation(
                f"Split date column '{self.date_column}' missing from panel."
            )
        # fh_effdt is stored as integer YYYYMMDD (e.g., 20241031) in the
        # panel pickle.  Plain pd.to_datetime(..., errors="coerce") would
        # interpret these integers as nanoseconds-since-epoch (a date in
        # 1970), producing NaT or wrong dates and causing empty holdout sets.
        dates = pd.to_datetime(df[self.date_column], format="%Y%m%d", errors="coerce")
        n_nat = dates.isna().sum()
        if n_nat > 0:
            raise MBSContractViolation(
                f"{n_nat} rows have unparseable {self.date_column} values "
                f"(expected integer YYYYMMDD). Check the panel data."
            )
        train_cutoff = pd.Timestamp(self.train_end_date)
        test_start = train_cutoff + pd.DateOffset(months=self.embargo_months)
        return df[dates <= train_cutoff].copy(), df[dates > test_start].copy()


# ---------------------------------------------------------------------------
# Scaler inverse-transform
# ---------------------------------------------------------------------------


def _load_scaler(scaler_path: Path) -> Any:
    try:
        import joblib
    except ImportError as e:
        raise MBSContractViolation(
            "joblib is required to load scaler.sav — install it in the runner env."
        ) from e
    if not scaler_path.exists():
        raise MBSContractViolation(f"Missing scaler file: {scaler_path}")
    return joblib.load(scaler_path)


def inverse_transform_features(
    df: pd.DataFrame, scaler: Any, feature_names: Sequence[str]
) -> pd.DataFrame:
    """Return a DataFrame of raw-scale values for the named GNMA features.

    The scaler is expected to be a sklearn-style object with either
    ``inverse_transform`` (operating on the full normalized feature matrix)
    or the ``mean_`` / ``scale_`` attribute pair. We prefer per-column
    reconstruction so callers can pass a subset of columns.
    """
    out_cols: dict[str, np.ndarray] = {}
    mean_ = getattr(scaler, "mean_", None)
    scale_ = getattr(scaler, "scale_", None)
    raw_names = getattr(scaler, "feature_names_in_", None)
    feature_names_in = list(raw_names) if raw_names is not None else []
    missing = [f for f in feature_names if f not in df.columns]
    if missing:
        raise MBSContractViolation(
            f"Features requested for inverse-transform not in panel: {missing}"
        )
    if mean_ is not None and scale_ is not None and feature_names_in:
        idx_map = {name: i for i, name in enumerate(feature_names_in)}
        for f in feature_names:
            if f not in idx_map:
                raise MBSContractViolation(
                    f"Scaler does not know feature '{f}'. "
                    f"Known: {feature_names_in[:10]}..."
                )
            i = idx_map[f]
            out_cols[f] = df[f].to_numpy(dtype=float) * scale_[i] + mean_[i]
        return pd.DataFrame(out_cols, index=df.index)
    if hasattr(scaler, "inverse_transform") and feature_names_in:
        # Fall back to full-matrix inverse_transform using the scaler's known columns.
        known = [c for c in feature_names_in if c in df.columns]
        if not known:
            raise MBSContractViolation(
                "Scaler exposes inverse_transform but no known features overlap the panel."
            )
        block = df[known].to_numpy(dtype=float)
        inv = scaler.inverse_transform(block)
        inv_df = pd.DataFrame(inv, columns=known, index=df.index)
        for f in feature_names:
            if f not in inv_df.columns:
                raise MBSContractViolation(
                    f"Scaler cannot inverse '{f}' — not in scaler.feature_names_in_."
                )
            out_cols[f] = inv_df[f].to_numpy(dtype=float)
        return pd.DataFrame(out_cols, index=df.index)
    raise MBSContractViolation(
        "scaler.sav must expose either (mean_, scale_, feature_names_in_) or "
        "(inverse_transform, feature_names_in_)."
    )


# ---------------------------------------------------------------------------
# Prediction clipping
# ---------------------------------------------------------------------------


def clip_predictions(y_pred: np.ndarray, contract: MBSDataContract) -> np.ndarray:
    lo, hi = contract.target_range
    return np.clip(np.asarray(y_pred, dtype=float), lo, hi)


# ---------------------------------------------------------------------------
# Workflow orchestration
# ---------------------------------------------------------------------------


@dataclass
class MBSWorkflow:
    """Fixed workflow: load → validate → split → fit → predict → clip.

    The LLM only provides ``model_builder()`` returning a fresh, unfitted
    estimator with ``.fit(X, y)`` and ``.predict(X)``. Features are
    pre-built and pre-normalized — no feature-engineering hook.
    """

    contract: MBSDataContract = field(default_factory=MBSDataContract)
    splitter: MBSTrainTestSplit = field(default_factory=MBSTrainTestSplit)

    def _feature_columns(self, df: pd.DataFrame) -> list[str]:
        drop = {self.contract.cusip_col, self.contract.date_col, self.contract.target_col}
        return [c for c in df.columns if c not in drop]

    def run(
        self,
        panel: pd.DataFrame,
        model_builder: Callable[[], Any],
    ) -> dict[str, Any]:
        self.contract.validate(panel, include_target=True)
        train_df, test_df = self.splitter.split(panel)
        if len(train_df) == 0 or len(test_df) == 0:
            raise MBSContractViolation(
                f"Split produced empty partition: train={len(train_df)}, "
                f"test={len(test_df)}. Check train_end_date={self.splitter.train_end_date}."
            )
        feat_cols = self._feature_columns(panel)
        X_train = train_df[feat_cols].to_numpy(dtype=float)
        y_train = train_df[self.contract.target_col].to_numpy(dtype=float)
        X_test = test_df[feat_cols].to_numpy(dtype=float)
        y_test = test_df[self.contract.target_col].to_numpy(dtype=float)

        model = model_builder()
        model.fit(X_train, y_train)
        y_pred = clip_predictions(model.predict(X_test), self.contract)

        submission = pd.DataFrame(
            {
                self.contract.cusip_col: test_df[self.contract.cusip_col].values,
                self.contract.date_col: test_df[self.contract.date_col].values,
                f"{self.contract.target_col}_pred": y_pred,
            }
        )
        return {
            "model": model,
            "y_true": y_test,
            "y_pred": y_pred,
            "test_df": test_df,
            "feature_columns": feat_cols,
            "submission": submission,
        }


# ---------------------------------------------------------------------------
# main.py entry point the RD-Agent Workflow component invokes
# ---------------------------------------------------------------------------


def run_scaffold_pipeline(
    panel_path: str | Path,
    scaler_path: str | Path,
    model_builder: Callable[[], Any],
    output_dir: str | Path,
    contract: MBSDataContract | None = None,
    splitter: MBSTrainTestSplit | None = None,
) -> dict[str, Any]:
    """End-to-end scaffold entry for the Workflow component's ``main.py``.

    - ``panel_path``: ``tfminput.pkl`` path (normalized panel).
    - ``scaler_path``: ``scaler.sav`` path (joblib).
    - ``model_builder``: LLM-provided callable returning an unfitted model.
    - ``output_dir``: workspace output root. Writes ``submission.csv``
      (normalized test rows + predictions), ``scores.json`` (full MBS
      scorecard computed on raw-scale GNMA features via the scaler), and
      ``scores.csv`` (primary metric bridged for the DS runner).
    """
    from rdagent.scenarios.mbs_prepayment.evaluation import (
        MBSEvaluationHarness,
        write_scorecard,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    panel = pd.read_pickle(panel_path)
    if not isinstance(panel, pd.DataFrame):
        raise MBSContractViolation(
            f"{panel_path} must unpickle to a DataFrame; got {type(panel).__name__}."
        )
    scaler = _load_scaler(Path(scaler_path))

    workflow = MBSWorkflow(
        contract=contract or MBSDataContract(),
        splitter=splitter or MBSTrainTestSplit(),
    )
    result = workflow.run(panel, model_builder)

    # Persist submission in DS-compatible format.
    submission_path = output_dir / "submission.csv"
    result["submission"].to_csv(submission_path, index=False)

    # Harness needs raw-scale GNMA features — inverse-transform and assemble
    # a feature frame carrying cusip/fh_effdt + raw-scale metadata.
    test_df: pd.DataFrame = result["test_df"]
    raw_feats = inverse_transform_features(
        test_df, scaler, workflow.contract.harness_raw_features
    )
    harness_features = pd.concat(
        [
            test_df[[workflow.contract.cusip_col, workflow.contract.date_col]].reset_index(drop=True),
            raw_feats.reset_index(drop=True),
        ],
        axis=1,
    )
    harness = MBSEvaluationHarness()
    scorecard = harness.evaluate(
        y_true=result["y_true"], y_pred=result["y_pred"], features=harness_features
    )
    write_scorecard(scorecard, str(output_dir / "scores.json"))

    primary_value = scorecard.get("primary_metric", {}).get("value", float("nan"))
    pd.DataFrame(
        {"rmse_smm_decimal": [primary_value]},
        index=pd.Index(["ensemble"], name=""),
    ).to_csv(output_dir / "scores.csv")

    return {"result": result, "scorecard": scorecard, "submission_path": str(submission_path)}
