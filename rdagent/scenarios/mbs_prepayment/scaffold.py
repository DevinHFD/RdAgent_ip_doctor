"""MBS Code Scaffold — Priority 2: Hard contracts the LLM cannot violate.

This scaffold assumes the single-panel layout:

    <data_dir>/<competition>/
        tfminput.pkl   # Full CUSIP-level monthly panel, ALL feature columns
                       # stored in normalized form (mean 0, std 1). Also
                       # carries cusip, fh_effdt, and the target
                       # SMM_DECIMAL (which is NOT normalized, it lives in
                       # [0, 1] already). NOTE: the column is uppercase
                       # "SMM_DECIMAL" in the file, not "smm_decimal".
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

Loading, CUSIP-stratified splitting, scaler inverse-transform, prediction
clipping, submission writing, and MBS scorecard evaluation are all scaffold
code — the LLM cannot modify them.

Split design (fixed by seed, enforced by scaffold):
    - Test CUSIPs  : 1/7 of all unique CUSIPs, sampled with seed=42.
                     ALL time rows for these CUSIPs are held out.
    - Train CUSIPs : 80 % of remaining CUSIPs, rows with
                     fh_effdt <= train_end_date.
    - Val CUSIPs   : 20 % of remaining CUSIPs, rows with
                     fh_effdt <= train_end_date.
    Scoring is on the fixed test-CUSIP set every iteration, so all loops
    are directly comparable.
"""
from __future__ import annotations

import csv
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
    #: Target column name — uppercase "SMM_DECIMAL" as stored in tfminput.pkl.
    target_col: str = "SMM_DECIMAL"
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
# CUSIP-stratified train / val / test split
# ---------------------------------------------------------------------------


@dataclass
class MBSCUSIPSplit:
    """Fixed CUSIP-stratified three-way split enforced by the scaffold.

    Partition design (all random choices use ``random_seed`` for
    reproducibility across loops):

    * **Test CUSIPs** (``test_fraction`` ≈ 1/7 of all CUSIPs):
      ALL time rows — holds out entire pools so every loop scores on
      exactly the same observations.
    * **Train CUSIPs** (80 % of the remaining CUSIPs):
      Only rows with ``fh_effdt <= train_end_date``.
    * **Val CUSIPs** (20 % of the remaining CUSIPs):
      Only rows with ``fh_effdt <= train_end_date`` — used by models for
      early stopping / hyperparameter selection.

    The scaffold passes ``X_val`` / ``y_val`` to ``model.fit()`` via
    keyword arguments so GBM early-stopping and PyTorch validation loops
    can consume them.  Sklearn models that do not accept these kwargs
    fall back to ``fit(X_train, y_train)`` automatically.
    """

    train_end_date: str = "2024-10-31"
    date_column: str = "fh_effdt"
    cusip_column: str = "cusip"
    test_fraction: float = 1.0 / 7.0
    val_fraction: float = 0.20
    random_seed: int = 42

    def split(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Return (train_df, val_df, test_df).

        * train_df : train_cusips  ×  fh_effdt <= cutoff
        * val_df   : val_cusips    ×  fh_effdt <= cutoff
        * test_df  : test_cusips   ×  ALL rows
        """
        if self.date_column not in df.columns:
            raise MBSContractViolation(
                f"Split date column '{self.date_column}' missing from panel."
            )
        if self.cusip_column not in df.columns:
            raise MBSContractViolation(
                f"Split CUSIP column '{self.cusip_column}' missing from panel."
            )
        # fh_effdt is stored as integer YYYYMMDD in the panel pickle.
        dates = pd.to_datetime(df[self.date_column], format="%Y%m%d", errors="coerce")
        n_nat = dates.isna().sum()
        if n_nat > 0:
            raise MBSContractViolation(
                f"{n_nat} rows have unparseable {self.date_column} values "
                f"(expected integer YYYYMMDD). Check the panel data."
            )

        # Sorted unique CUSIPs → deterministic base ordering before shuffle.
        all_cusips = np.array(sorted(df[self.cusip_column].unique()))
        rng = np.random.default_rng(self.random_seed)
        shuffled = all_cusips.copy()
        rng.shuffle(shuffled)

        n_total = len(shuffled)
        n_test = max(1, round(n_total * self.test_fraction))
        test_cusips = set(shuffled[:n_test])
        remaining = shuffled[n_test:]

        n_val = max(1, round(len(remaining) * self.val_fraction))
        val_cusips = set(remaining[:n_val])
        train_cusips = set(remaining[n_val:])

        cutoff = pd.Timestamp(self.train_end_date)
        temporal_mask = (dates <= cutoff).to_numpy()
        cusip_arr = df[self.cusip_column].to_numpy()

        train_mask = np.isin(cusip_arr, list(train_cusips)) & temporal_mask
        val_mask = np.isin(cusip_arr, list(val_cusips)) & temporal_mask
        test_mask = np.isin(cusip_arr, list(test_cusips))

        train_df = df[train_mask].copy()
        val_df = df[val_mask].copy()
        test_df = df[test_mask].copy()

        if len(train_df) == 0:
            raise MBSContractViolation(
                "CUSIP split produced an empty train set — check train_end_date "
                f"({self.train_end_date}) and random_seed ({self.random_seed})."
            )
        if len(test_df) == 0:
            raise MBSContractViolation(
                "CUSIP split produced an empty test set — check test_fraction "
                f"({self.test_fraction}) and the number of unique CUSIPs."
            )
        return train_df, val_df, test_df

    # Keep a thin shim so old callers that expect a 2-tuple still work.
    def split_train_test(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        train_df, _val_df, test_df = self.split(df)
        return train_df, test_df


# ---------------------------------------------------------------------------
# Legacy temporal splitter (kept for backward compatibility / tests)
# ---------------------------------------------------------------------------


@dataclass
class MBSTrainTestSplit:
    """Simple temporal split — superseded by MBSCUSIPSplit.

    Kept for backward compatibility; the scaffold pipeline now uses
    ``MBSCUSIPSplit`` by default.
    """

    train_end_date: str = "2024-10-31"
    date_column: str = "fh_effdt"
    embargo_months: int = 0

    def split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        if self.date_column not in df.columns:
            raise MBSContractViolation(
                f"Split date column '{self.date_column}' missing from panel."
            )
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
# Test-prediction history (appended each loop)
# ---------------------------------------------------------------------------

_HISTORY_COLUMNS = ["loop_number", "cusip", "fh_effdt", "fh_upb", "smm_decimal", "smm_decimal_pred"]


def _append_test_predictions(
    output_dir: Path,
    test_df: pd.DataFrame,
    y_pred: np.ndarray,
    contract: MBSDataContract,
    filename: str = "test_predictions_history.csv",
) -> Path:
    """Append this loop's test-set predictions to the shared history file.

    The file accumulates one block of rows per successful loop so downstream
    analysis can compare model evolution over multiple iterations.

    Columns written:
        loop_number, cusip, fh_effdt, fh_upb, smm_decimal, smm_decimal_pred
    """
    out_path = output_dir / filename

    # Determine loop number from existing file (auto-increment).
    loop_num = 1
    if out_path.exists() and out_path.stat().st_size > 0:
        try:
            existing = pd.read_csv(out_path, usecols=["loop_number"])
            if len(existing) > 0:
                loop_num = int(existing["loop_number"].max()) + 1
        except Exception:
            pass

    # Collect available columns from test_df.
    record = test_df.reset_index(drop=True).copy()
    cols_to_keep = [c for c in [contract.cusip_col, contract.date_col, "fh_upb", contract.target_col] if c in record.columns]
    record = record[cols_to_keep].copy()
    record["smm_decimal_pred"] = y_pred
    record.insert(0, "loop_number", loop_num)

    # Rename to canonical column names for the history file.
    # Normalise to lowercase column names in the history CSV so readers don't
    # need to know whether the source panel used "SMM_DECIMAL" or "smm_decimal".
    rename = {
        contract.cusip_col: "cusip",
        contract.date_col: "fh_effdt",
        contract.target_col: "smm_decimal",
    }
    record = record.rename(columns=rename)

    write_header = not (out_path.exists() and out_path.stat().st_size > 0)
    record.to_csv(out_path, mode="a", header=write_header, index=False)
    return out_path


# ---------------------------------------------------------------------------
# Workflow orchestration
# ---------------------------------------------------------------------------


@dataclass
class MBSWorkflow:
    """Fixed workflow: load → validate → split → fit → predict → clip.

    The LLM only provides ``model_builder()`` returning a fresh, unfitted
    estimator with ``.fit(X, y, **kwargs)`` and ``.predict(X)``.

    The scaffold calls::

        model.fit(X_train, y_train,
                  X_val=X_val, y_val=y_val,
                  sample_weight=train_weights)

    Keyword arguments are passed through gracefully — models that accept
    ``X_val`` / ``y_val`` (PyTorch loops, custom early-stopping wrappers)
    use them; sklearn models that do not accept them fall back automatically.

    ``sample_weight`` is set to ``min(fh_upb, 150e6)`` when the ``fh_upb``
    column is present in the panel, matching the SOTA UPB-weighted loss.
    """

    contract: MBSDataContract = field(default_factory=MBSDataContract)
    splitter: MBSCUSIPSplit = field(default_factory=MBSCUSIPSplit)
    #: UPB weight cap (matching SOTA model: 150 M).
    upb_weight_cap: float = 150e6
    #: Column name for pool unpaid principal balance (weight source).
    upb_col: str = "fh_upb"

    def _feature_columns(self, df: pd.DataFrame) -> list[str]:
        drop = {self.contract.cusip_col, self.contract.date_col, self.contract.target_col}
        return [c for c in df.columns if c not in drop]

    def _fit_model(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        sample_weight: np.ndarray | None,
    ) -> None:
        """Call model.fit with progressive kwarg fallback."""
        fit_kwargs: dict[str, Any] = {}
        if len(X_val) > 0:
            fit_kwargs["X_val"] = X_val
            fit_kwargs["y_val"] = y_val
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight

        if not fit_kwargs:
            model.fit(X_train, y_train)
            return

        # Try full kwargs first; fall back step-by-step on TypeError.
        for attempt_kwargs in [
            fit_kwargs,
            {k: v for k, v in fit_kwargs.items() if k == "sample_weight"},
            {},
        ]:
            try:
                model.fit(X_train, y_train, **attempt_kwargs)
                return
            except TypeError:
                if not attempt_kwargs:
                    raise

    def run(
        self,
        panel: pd.DataFrame,
        model_builder: Callable[[], Any],
    ) -> dict[str, Any]:
        self.contract.validate(panel, include_target=True)
        train_df, val_df, test_df = self.splitter.split(panel)

        feat_cols = self._feature_columns(panel)
        X_train = train_df[feat_cols].to_numpy(dtype=float)
        y_train = train_df[self.contract.target_col].to_numpy(dtype=float)
        X_val = val_df[feat_cols].to_numpy(dtype=float) if len(val_df) > 0 else np.empty((0, len(feat_cols)))
        y_val = val_df[self.contract.target_col].to_numpy(dtype=float) if len(val_df) > 0 else np.empty(0)
        X_test = test_df[feat_cols].to_numpy(dtype=float)
        y_test = test_df[self.contract.target_col].to_numpy(dtype=float)

        # UPB-weighted sample_weight for training (SOTA convention).
        sample_weight: np.ndarray | None = None
        if self.upb_col in train_df.columns:
            sample_weight = np.minimum(
                train_df[self.upb_col].to_numpy(dtype=float), self.upb_weight_cap
            )

        model = model_builder()
        self._fit_model(model, X_train, y_train, X_val, y_val, sample_weight)
        y_pred = clip_predictions(model.predict(X_test), self.contract)

        submission = pd.DataFrame(
            {
                self.contract.cusip_col: test_df[self.contract.cusip_col].values,
                self.contract.date_col: test_df[self.contract.date_col].values,
                # Output column is lowercase for readability; the source column
                # in the panel is uppercase SMM_DECIMAL.
                "smm_decimal_pred": y_pred,
            }
        )
        return {
            "model": model,
            "y_true": y_test,
            "y_pred": y_pred,
            "train_df": train_df,
            "val_df": val_df,
            "test_df": test_df,
            "feature_columns": feat_cols,
            "submission": submission,
        }


# ---------------------------------------------------------------------------
# Per-loop SMM actual-vs-predicted plot
# ---------------------------------------------------------------------------

_TRAIN_END_DATE = "2024-10-31"
_UPB_WEIGHT_CAP = 150_000_000.0


def _write_smm_plot(
    output_dir: Path,
    test_df: pd.DataFrame,
    y_pred: "np.ndarray",
    contract: "MBSDataContract",
    scorecard: dict,
) -> None:
    """Save smm_actual_vs_pred.html to output_dir (Plotly, self-contained).

    x-axis: fh_effdt (monthly dates)
    y-axis: fh_upb-weighted average SMM_DECIMAL and smm_decimal_pred
    Vertical line marks the train/val cutoff (2024-10-31).
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        return  # plotly not installed — skip silently

    df = test_df[[contract.date_col, contract.target_col]].copy().reset_index(drop=True)
    df["smm_pred"] = np.asarray(y_pred, dtype=float)
    upb_cap = _UPB_WEIGHT_CAP
    if "fh_upb" in test_df.columns:
        df["_w"] = np.minimum(test_df["fh_upb"].to_numpy(dtype=float), upb_cap)
    else:
        df["_w"] = 1.0
    df["_date"] = pd.to_datetime(df[contract.date_col], format="%Y%m%d", errors="coerce")

    def wavg(group):
        w = group["_w"]
        s = w.sum()
        if s == 0:
            return pd.Series({"smm_actual": float("nan"), "smm_pred": float("nan")})
        return pd.Series(
            {
                "smm_actual": (group[contract.target_col] * w).sum() / s,
                "smm_pred": (group["smm_pred"] * w).sum() / s,
            }
        )

    agg = df.groupby("_date").apply(wavg).sort_index().reset_index()

    overall_rmse = scorecard.get("accuracy", {}).get("overall_rmse", float("nan"))
    oot_rmse = scorecard.get("accuracy", {}).get("oot_rmse", float("nan"))

    cutoff = pd.Timestamp(_TRAIN_END_DATE)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=agg["_date"], y=agg["smm_actual"],
            mode="lines", name="Actual SMM_DECIMAL",
            line=dict(color="#1f77b4", width=1.8),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=agg["_date"], y=agg["smm_pred"],
            mode="lines", name="Predicted smm_decimal",
            line=dict(color="#ff7f0e", width=1.8, dash="dot"),
        )
    )
    fig.add_vline(
        x=cutoff.timestamp() * 1000,
        line_width=1.5, line_dash="dash", line_color="grey",
        annotation_text="Train cutoff", annotation_position="top left",
    )
    fig.update_layout(
        title=(
            f"UPB-Weighted Avg SMM — Actual vs Predicted<br>"
            f"<sup>Overall RMSE: {overall_rmse:.5f} | OOT RMSE: {oot_rmse:.5f}</sup>"
        ),
        xaxis_title="Date (fh_effdt)",
        yaxis_title="UPB-Weighted Avg SMM",
        legend=dict(x=0.01, y=0.99),
        template="plotly_white",
        height=420,
    )
    fig.write_html(str(output_dir / "smm_actual_vs_pred.html"), include_plotlyjs="cdn")


# ---------------------------------------------------------------------------
# main.py entry point the RD-Agent Workflow component invokes
# ---------------------------------------------------------------------------


def run_scaffold_pipeline(
    panel_path: str | Path,
    scaler_path: str | Path,
    model_builder: Callable[[], Any],
    output_dir: str | Path,
    contract: MBSDataContract | None = None,
    splitter: MBSCUSIPSplit | None = None,
    history_output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """End-to-end scaffold entry for the Workflow component's ``main.py``.

    - ``panel_path``: ``tfminput.pkl`` path (normalized panel).
    - ``scaler_path``: ``scaler.sav`` path (joblib).
    - ``model_builder``: LLM-provided callable returning an unfitted model.
    - ``output_dir``: workspace output root. Writes ``submission.csv``
      (test-CUSIP rows + predictions), ``scores.json`` (full MBS scorecard),
      and ``scores.csv`` (primary metric for the DS runner).
    - ``history_output_dir``: where to write the cross-loop test predictions
      history file.  Defaults to ``./mbs_output`` next to the workspace.
    """
    from rdagent.scenarios.mbs_prepayment.evaluation import (
        MBSEvaluationHarness,
        write_scorecard,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve history output directory (default: ./mbs_output relative to cwd).
    if history_output_dir is None:
        history_output_dir = Path("./mbs_output")
    history_output_dir = Path(history_output_dir)
    history_output_dir.mkdir(parents=True, exist_ok=True)

    panel = pd.read_pickle(panel_path)
    if not isinstance(panel, pd.DataFrame):
        raise MBSContractViolation(
            f"{panel_path} must unpickle to a DataFrame; got {type(panel).__name__}."
        )
    scaler = _load_scaler(Path(scaler_path))

    used_contract = contract or MBSDataContract()
    workflow = MBSWorkflow(
        contract=used_contract,
        splitter=splitter or MBSCUSIPSplit(),
    )
    result = workflow.run(panel, model_builder)

    # Persist submission in DS-compatible format.
    submission_path = output_dir / "submission.csv"
    result["submission"].to_csv(submission_path, index=False)

    # Harness needs raw-scale GNMA features — inverse-transform and assemble.
    test_df: pd.DataFrame = result["test_df"]
    raw_feats = inverse_transform_features(
        test_df, scaler, workflow.contract.harness_raw_features
    )
    extra_cols = [workflow.contract.cusip_col, workflow.contract.date_col]
    if "fh_upb" in test_df.columns:
        extra_cols.append("fh_upb")
    harness_features = pd.concat(
        [
            test_df[extra_cols].reset_index(drop=True),
            raw_feats.reset_index(drop=True),
        ],
        axis=1,
    )
    harness = MBSEvaluationHarness()
    scorecard = harness.evaluate(
        y_true=result["y_true"], y_pred=result["y_pred"], features=harness_features
    )
    write_scorecard(scorecard, str(output_dir / "scores.json"))

    # Generate per-loop SMM actual-vs-predicted time-series plot.
    _write_smm_plot(output_dir, test_df, result["y_pred"], used_contract, scorecard)

    primary_value = scorecard.get("primary_metric", {}).get("value", float("nan"))
    pd.DataFrame(
        {"rmse_smm_decimal": [primary_value]},
        index=pd.Index(["ensemble"], name=""),
    ).to_csv(output_dir / "scores.csv")

    # Append test predictions to the cross-loop history file.
    history_path = _append_test_predictions(
        history_output_dir,
        test_df,
        result["y_pred"],
        used_contract,
    )
    print(f"[scaffold] Test predictions appended to {history_path}")

    return {
        "result": result,
        "scorecard": scorecard,
        "submission_path": str(submission_path),
        "history_path": str(history_path),
    }
