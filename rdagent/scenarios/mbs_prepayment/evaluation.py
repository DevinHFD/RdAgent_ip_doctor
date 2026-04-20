"""MBS Evaluation Harness — Priority 1: Multi-dimensional scorecard.

This module implements the fixed evaluation function for MBS CUSIP-level prepayment
forecasting. It is deliberately NOT something the LLM can modify — the `main.py`
template (the `Workflow` component) calls this harness to produce a structured JSON
scorecard that the feedback LLM then reasons over.

Why this design:
    The default RD-Agent loop compares experiments using a single scalar metric.
    For MBS prepayment, a model can be "better" on one axis and "worse" on another,
    and the feedback LLM must reason about these tradeoffs explicitly. Consolidating
    evaluation into one fixed harness also prevents the LLM from gaming metrics by
    subtly changing how they are computed in LLM-generated `main.py`.

Scorecard dimensions:
    Accuracy:
        - overall_rmse: Overall RMSE of SMM_DECIMAL across (cusip, fh_effdt)
        - rmse_by_coupon_bucket: Per-coupon-bucket RMSE (CRITICAL — prepayment
            behavior is heavily coupon-dependent; high-coupon CUSIPs drive most
            of the refi risk and must be modeled separately)
        - rmse_tail_high / rmse_tail_low: RMSE on CUSIPs in top/bottom 10% SMM
        - rmse_by_vintage: Per-vintage-year RMSE
    Rate Sensitivity Fidelity:
        - monotonicity_spearman: Spearman rank corr (predicted, rate_incentive)
        - s_curve_r2: R^2 of logistic fit to (incentive, predicted)
        - inflection_point_ratio: Refi-incentive RATIO
          (Avg_Prop_Refi_Incentive_WAC_30yr_2mos = WAC / avg(mortgage_rate_lag1,
          mortgage_rate_lag2); >1 means the pool coupon exceeds recent market
          rate — i.e. refi incentive) at which the fitted logistic S-curve
          crosses the mid-point of the normalized prediction range.
    Temporal Robustness:
        - regime_transition_rmse: RMSE in the first 3 months after each
            major regime transition in the holdout period
        - rolling_12m_rmse_ratio: max/min ratio of rolling 12m RMSE
        - extrapolation_sanity: Predicted SMM at +/-300bps incentive (NaN-safe)
    Structural Properties:
        - burnout_effect_pvalue: t-test p-value on (old ITM) vs (fresh) groups
        - seasonality_pvalue: F-test p-value on fh_effdt month-of-year dummies
        - cusip_differentiation_std: std of predicted SMM_DECIMAL across CUSIPs
            (>0 means the model actually uses CUSIP-level features)

Usage:
    from rdagent.scenarios.mbs_prepayment.evaluation import MBSEvaluationHarness

    harness = MBSEvaluationHarness(
        coupon_buckets=[(0, 3.0), (3.0, 3.5), (3.5, 4.0), (4.0, 4.5), (4.5, 5.0), (5.0, 99)],
        regime_transition_dates=["2020-03-01", "2022-03-01"],
    )
    scorecard = harness.evaluate(y_true=y_test, y_pred=preds, features=X_test)
    # scorecard is a dict; dump it to scores.json for the feedback LLM to read
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class MBSEvaluationHarness:
    """Fixed evaluation for MBS prepayment models. Not LLM-modifiable."""

    #: Coupon buckets as list of (lo, hi) tuples — closed-open intervals [lo, hi).
    coupon_buckets: list[tuple[float, float]] = field(
        default_factory=lambda: [
            (0.0, 3.0),
            (3.0, 3.5),
            (3.5, 4.0),
            (4.0, 4.5),
            (4.5, 5.0),
            (5.0, 99.0),
        ]
    )
    #: Known regime transition dates (YYYY-MM-DD) — used for regime_transition_rmse.
    regime_transition_dates: list[str] = field(
        default_factory=lambda: ["2013-05-01", "2020-03-01", "2022-03-01"]
    )
    #: Column names in the features DataFrame — must match gnma_feature.md
    #: and GNMA_HARNESS_FEATURES in scaffold.py exactly (case-sensitive).
    coupon_col: str = "WAC"
    rate_incentive_col: str = "Avg_Prop_Refi_Incentive_WAC_30yr_2mos"
    fh_effdt_col: str = "fh_effdt"
    cusip_col: str = "cusip"
    wala_col: str = "WALA"
    upb_col: str = "fh_upb"
    upb_weight_cap: float = 150_000_000.0
    #: Train/val cutoff — rows strictly after this date form the out-of-time (OOT)
    #: evaluation window. Format: "YYYY-MM-DD". None disables OOT RMSE.
    train_end_date: str | None = "2024-10-31"

    def evaluate(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray | pd.Series,
        features: pd.DataFrame,
    ) -> dict[str, Any]:
        """Compute the full multi-dimensional scorecard.

        Args:
            y_true: Actual SMM_DECIMAL values. Must be indexed by (cusip, fh_effdt)
                or be a DataFrame with those columns.
            y_pred: Predicted SMM_DECIMAL values. Same length/order as y_true.
            features: Feature DataFrame used for the predictions. Must contain
                `coupon`, `rate_incentive`, `fh_effdt`, `cusip`, and `wala`
                columns (names configurable via dataclass fields).

        Returns:
            A dict with four top-level keys: `accuracy`, `rate_sensitivity`,
            `temporal_robustness`, `structural_properties`. Each contains
            the numeric sub-metrics documented above.
        """
        y_true_arr = np.asarray(y_true, dtype=float).flatten()
        y_pred_arr = np.asarray(y_pred, dtype=float).flatten()
        if len(y_true_arr) != len(y_pred_arr):
            raise ValueError(
                f"y_true ({len(y_true_arr)}) and y_pred ({len(y_pred_arr)}) length mismatch"
            )
        if len(y_true_arr) != len(features):
            raise ValueError(
                f"features ({len(features)}) length does not match y_true ({len(y_true_arr)})"
            )

        scorecard: dict[str, Any] = {
            "accuracy": self._accuracy_dimensions(y_true_arr, y_pred_arr, features),
            "rate_sensitivity": self._rate_sensitivity_dimensions(y_pred_arr, features),
            "temporal_robustness": self._temporal_robustness_dimensions(
                y_true_arr, y_pred_arr, features
            ),
            "structural_properties": self._structural_properties_dimensions(
                y_true_arr, y_pred_arr, features
            ),
        }
        scorecard["primary_metric"] = {
            "name": "overall_rmse",
            "value": scorecard["accuracy"]["overall_rmse"],
            "lower_is_better": True,
        }
        return scorecard

    # --- Accuracy ---------------------------------------------------------

    def _accuracy_dimensions(
        self, y_true: np.ndarray, y_pred: np.ndarray, features: pd.DataFrame
    ) -> dict[str, Any]:
        # Extract UPB weights once; all RMSE metrics use the same weighting
        # convention (min(fh_upb, cap)) to stay consistent with the SOTA MLP
        # training loss. Falls back to uniform weights when fh_upb is absent.
        upb: np.ndarray | None = None
        if self.upb_col in features.columns:
            upb = np.asarray(features[self.upb_col], dtype=float)

        out: dict[str, Any] = {}
        out["overall_rmse"] = float(_rmse_w(y_true, y_pred, upb, self.upb_weight_cap))
        # Out-of-time RMSE: only rows strictly after the train/val cutoff.
        if self.train_end_date is not None and self.fh_effdt_col in features.columns:
            cutoff = pd.Timestamp(self.train_end_date)
            fh_dates = pd.to_datetime(features[self.fh_effdt_col], format="%Y%m%d", errors="coerce")
            oot_mask = (fh_dates > cutoff).to_numpy()
            out["oot_rmse"] = float(_rmse_w_on_mask(y_true, y_pred, upb, self.upb_weight_cap, oot_mask))
        out["rmse_by_coupon_bucket"] = self._rmse_by_coupon_bucket(y_true, y_pred, features, upb)
        out["rmse_tail_high"] = float(
            _rmse_w_on_mask(y_true, y_pred, upb, self.upb_weight_cap, y_true >= np.quantile(y_true, 0.90))
        )
        out["rmse_tail_low"] = float(
            _rmse_w_on_mask(y_true, y_pred, upb, self.upb_weight_cap, y_true <= np.quantile(y_true, 0.10))
        )
        out["rmse_by_vintage"] = self._rmse_by_vintage(y_true, y_pred, features, upb)
        return out

    def _rmse_by_coupon_bucket(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        features: pd.DataFrame,
        upb: np.ndarray | None,
    ) -> dict[str, float]:
        """Per-coupon-bucket UPB-weighted RMSE.

        High-coupon (deeply in-the-money) buckets drive most of the refi risk.
        A model with good overall RMSE but terrible high-coupon RMSE is
        unusable for hedging or valuation.
        """
        result: dict[str, float] = {}
        if self.coupon_col not in features.columns:
            return {"_error": f"feature '{self.coupon_col}' not found"}
        coupons = features[self.coupon_col].to_numpy(dtype=float)
        for lo, hi in self.coupon_buckets:
            mask = (coupons >= lo) & (coupons < hi)
            label = f"{lo:.1f}-{hi:.1f}" if hi < 99 else f"{lo:.1f}+"
            result[label] = float(_rmse_w_on_mask(y_true, y_pred, upb, self.upb_weight_cap, mask))
        return result

    def _rmse_by_vintage(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        features: pd.DataFrame,
        upb: np.ndarray | None,
    ) -> dict[str, float]:
        """Per-vintage-year UPB-weighted RMSE, inferred from WALA and fh_effdt."""
        if self.wala_col not in features.columns or self.fh_effdt_col not in features.columns:
            return {}
        fh = pd.to_datetime(features[self.fh_effdt_col], format="%Y%m%d", errors="coerce")
        wala = features[self.wala_col].to_numpy(dtype=float)
        vintage_year = (fh.dt.year - (wala / 12.0).astype(int)).to_numpy()
        result: dict[str, float] = {}
        for v in np.unique(vintage_year[~np.isnan(vintage_year)]):
            mask = vintage_year == v
            if mask.sum() >= 30:
                result[str(int(v))] = float(_rmse_w_on_mask(y_true, y_pred, upb, self.upb_weight_cap, mask))
        return result

    # --- Rate sensitivity fidelity ---------------------------------------

    def _rate_sensitivity_dimensions(
        self, y_pred: np.ndarray, features: pd.DataFrame
    ) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.rate_incentive_col not in features.columns:
            return {"_error": f"feature '{self.rate_incentive_col}' not found"}
        incentive = features[self.rate_incentive_col].to_numpy(dtype=float)
        valid = ~(np.isnan(incentive) | np.isnan(y_pred))
        if valid.sum() < 10:
            return {"_error": "not enough valid rows for rate sensitivity check"}
        out["monotonicity_spearman"] = float(
            _spearman(incentive[valid], y_pred[valid])
        )
        s_curve = _fit_logistic_s_curve(incentive[valid], y_pred[valid])
        out["s_curve_r2"] = float(s_curve["r2"])
        out["inflection_point_ratio"] = float(s_curve["inflection_ratio"])
        return out

    # --- Temporal robustness ---------------------------------------------

    def _temporal_robustness_dimensions(
        self, y_true: np.ndarray, y_pred: np.ndarray, features: pd.DataFrame
    ) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.fh_effdt_col not in features.columns:
            return {"_error": f"feature '{self.fh_effdt_col}' not found"}
        fh = pd.to_datetime(features[self.fh_effdt_col], format="%Y%m%d", errors="coerce")
        upb: np.ndarray | None = None
        if self.upb_col in features.columns:
            upb = np.asarray(features[self.upb_col], dtype=float)

        # Regime transition RMSE: first 3 months after each known transition
        regime_rmses: dict[str, float] = {}
        for t_str in self.regime_transition_dates:
            t = pd.Timestamp(t_str)
            mask = (fh >= t) & (fh < t + pd.DateOffset(months=3))
            mask_arr = mask.to_numpy()
            if mask_arr.sum() >= 10:
                regime_rmses[t_str] = float(_rmse_w_on_mask(y_true, y_pred, upb, self.upb_weight_cap, mask_arr))
        out["regime_transition_rmse"] = regime_rmses

        # Rolling 12-month RMSE stability (within-month UPB-weighted, across-month simple mean)
        df = pd.DataFrame({"fh_effdt": fh, "y_true": y_true, "y_pred": y_pred}).dropna()
        if upb is not None:
            df["w"] = np.minimum(np.where(np.isnan(upb), 0.0, upb), self.upb_weight_cap)
        else:
            df["w"] = 1.0
        df = df[df["w"] > 0]
        if len(df) >= 12:
            df["month"] = df["fh_effdt"].dt.to_period("M")
            df["wsq"] = df["w"] * (df["y_true"] - df["y_pred"]) ** 2
            g = df.groupby("month")
            monthly_wmse = g["wsq"].sum() / g["w"].sum()
            rolling = monthly_wmse.rolling(12).mean().dropna() ** 0.5
            if len(rolling) > 0:
                out["rolling_12m_rmse_ratio"] = float(rolling.max() / max(rolling.min(), 1e-12))
                out["rolling_12m_rmse_max"] = float(rolling.max())
                out["rolling_12m_rmse_min"] = float(rolling.min())
        return out

    # --- Structural properties -------------------------------------------

    def _structural_properties_dimensions(
        self, y_true: np.ndarray, y_pred: np.ndarray, features: pd.DataFrame
    ) -> dict[str, Any]:
        out: dict[str, Any] = {}

        # Burnout effect: do CUSIPs with sustained ITM predict lower than fresh?
        if self.rate_incentive_col in features.columns:
            incentive = features[self.rate_incentive_col].to_numpy(dtype=float)
            # Avg_Prop_Refi_Incentive_WAC_30yr_2mos is a dimensionless ratio
            # (WAC / avg(mortgage_rate_lag1, mortgage_rate_lag2)); ratio > 1 means
            # the pool coupon exceeds the recent market rate, i.e. refi incentive.
            itm_mask = incentive > 1.0
            if itm_mask.sum() >= 30 and (~itm_mask).sum() >= 30:
                out["burnout_effect_mean_itm"] = float(np.nanmean(y_pred[itm_mask]))
                out["burnout_effect_mean_non_itm"] = float(np.nanmean(y_pred[~itm_mask]))

        # Seasonality: residual variance by month-of-year
        if self.fh_effdt_col in features.columns:
            fh = pd.to_datetime(features[self.fh_effdt_col], format="%Y%m%d", errors="coerce")
            moy = fh.dt.month
            resid = y_true - y_pred
            by_month = pd.DataFrame({"moy": moy, "resid": resid}).dropna().groupby("moy")["resid"].mean()
            if len(by_month) >= 6:
                out["seasonality_residual_range"] = float(by_month.max() - by_month.min())

        # CUSIP differentiation: std of predictions across cusips
        if self.cusip_col in features.columns:
            by_cusip = pd.DataFrame({"cusip": features[self.cusip_col].values, "y_pred": y_pred})
            mean_by_cusip = by_cusip.groupby("cusip")["y_pred"].mean()
            out["cusip_differentiation_std"] = float(mean_by_cusip.std())
        return out


def _rmse_w(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    upb: np.ndarray | None,
    cap: float,
) -> float:
    """UPB-weighted RMSE. Falls back to unweighted when upb is None."""
    if upb is None:
        valid = ~(np.isnan(y_true) | np.isnan(y_pred))
        if valid.sum() == 0:
            return float("nan")
        return float(np.sqrt(np.mean((y_true[valid] - y_pred[valid]) ** 2)))
    w = np.minimum(np.where(np.isnan(upb), 0.0, upb), cap)
    valid = ~(np.isnan(y_true) | np.isnan(y_pred)) & (w > 0)
    if valid.sum() == 0:
        return float("nan")
    w_v = w[valid]
    sq_err = (y_true[valid] - y_pred[valid]) ** 2
    return float(np.sqrt(np.sum(w_v * sq_err) / np.sum(w_v)))


def _rmse_w_on_mask(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    upb: np.ndarray | None,
    cap: float,
    mask: np.ndarray,
) -> float:
    if mask.sum() == 0:
        return float("nan")
    upb_m = upb[mask] if upb is not None else None
    return _rmse_w(y_true[mask], y_pred[mask], upb_m, cap)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    cov = np.mean((rx - rx.mean()) * (ry - ry.mean()))
    denom = rx.std() * ry.std()
    if denom < 1e-12:
        return 0.0
    return float(cov / denom)


def _fit_logistic_s_curve(incentive: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Lightweight logistic fit. Avoids scipy dependency."""
    x = incentive
    y = y_pred
    y_min, y_max = y.min(), y.max()
    if y_max - y_min < 1e-8:
        return {"r2": 0.0, "inflection_ratio": 0.0}
    yn = (y - y_min) / (y_max - y_min + 1e-12)
    yn = np.clip(yn, 1e-4, 1 - 1e-4)
    z = np.log(yn / (1 - yn))
    xm = x - x.mean()
    denom = float(np.sum(xm * xm))
    if denom < 1e-12:
        return {"r2": 0.0, "inflection_ratio": 0.0}
    slope = float(np.sum(xm * (z - z.mean())) / denom)
    intercept = float(z.mean() - slope * x.mean())
    z_hat = slope * x + intercept
    ss_res = float(np.sum((z - z_hat) ** 2))
    ss_tot = float(np.sum((z - z.mean()) ** 2))
    r2 = 1 - ss_res / max(ss_tot, 1e-12)
    # Inflection is expressed in the same units as the input feature
    # (Avg_Prop_Refi_Incentive_WAC_30yr_2mos), i.e. a dimensionless ratio.
    inflection = -intercept / slope if abs(slope) > 1e-12 else 0.0
    return {"r2": max(0.0, min(1.0, r2)), "inflection_ratio": float(inflection)}


def write_scorecard(scorecard: dict[str, Any], path: str) -> None:
    """Serialize a scorecard to JSON at the given path."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(scorecard, f, indent=2, default=str)
