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
        - s_curve_rmse_*: bin RMSE between UPB-weighted ACTUAL and PREDICTED
            SMM bin curves over Avg_Prop_Refi_Incentive_WAC_30yr_2mos.
        - s_curve_rmse_{overall,left_tail,mid_belly,right_tail}: RMSE between
          the UPB-weighted ACTUAL SMM_DECIMAL bin-curve and the UPB-weighted
          PREDICTED SMM_DECIMAL bin-curve, where bins are over
          Avg_Prop_Refi_Incentive_WAC_30yr_2mos (a dimensionless refi-incentive
          ratio). Each bin contributes one point to each curve; bin RMSE is
          equal-weighted across bins. Segments: left_tail = bins with right
          edge ≤ left_tail_max_ratio; right_tail = bins with left edge ≥
          right_tail_min_ratio; mid_belly = the rest. Tail decomposition
          surfaces which part of the S-curve needs improvement.
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

    # --- S-curve bin-RMSE configuration -----------------------------------
    #: Right-exclusive bin edges over Avg_Prop_Refi_Incentive_WAC_30yr_2mos.
    #: Default: 13 bins spanning [0, ∞). The curve is built by computing the
    #: UPB-weighted mean of y_true and y_pred per bin.
    s_curve_bin_edges: list[float] = field(
        default_factory=lambda: [
            0.0, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7,
            float("inf"),
        ]
    )
    #: A bin belongs to the LEFT TAIL if its right edge ≤ this ratio.
    s_curve_left_tail_max_ratio: float = 0.9
    #: A bin belongs to the RIGHT TAIL if its left edge ≥ this ratio.
    s_curve_right_tail_min_ratio: float = 1.4
    #: Bins with fewer than this many valid rows are dropped from the RMSE
    #: aggregation. Matches the per-vintage population threshold.
    s_curve_min_rows_per_bin: int = 30

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
            "rate_sensitivity": self._rate_sensitivity_dimensions(
                y_true_arr, y_pred_arr, features
            ),
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
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        features: pd.DataFrame,
    ) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.rate_incentive_col not in features.columns:
            return {"_error": f"feature '{self.rate_incentive_col}' not found"}
        incentive = features[self.rate_incentive_col].to_numpy(dtype=float)
        upb: np.ndarray | None = None
        if self.upb_col in features.columns:
            upb = np.asarray(features[self.upb_col], dtype=float)

        valid = ~(np.isnan(incentive) | np.isnan(y_pred))
        if valid.sum() < 10:
            return {"_error": "not enough valid rows for rate sensitivity check"}
        out["monotonicity_spearman"] = float(
            _spearman(incentive[valid], y_pred[valid])
        )
        out.update(
            _compute_s_curve_rmse(
                y_true=y_true,
                y_pred=y_pred,
                incentive=incentive,
                upb=upb,
                bin_edges=self.s_curve_bin_edges,
                upb_cap=self.upb_weight_cap,
                min_rows_per_bin=self.s_curve_min_rows_per_bin,
                left_tail_max=self.s_curve_left_tail_max_ratio,
                right_tail_min=self.s_curve_right_tail_min_ratio,
            )
        )
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


def _compute_s_curve_rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    incentive: np.ndarray,
    upb: np.ndarray | None,
    bin_edges: list[float],
    upb_cap: float,
    min_rows_per_bin: int,
    left_tail_max: float,
    right_tail_min: float,
) -> dict[str, Any]:
    """S-curve bin-RMSE between actual and predicted UPB-weighted bin means.

    Bins ``incentive`` per ``bin_edges`` (right-exclusive). For each bin with
    at least ``min_rows_per_bin`` valid rows, computes UPB-weighted means of
    y_true and y_pred. RMSE between the two bin sequences is the s_curve
    RMSE; per-segment RMSE (left tail / mid belly / right tail) and the
    overall RMSE share the same kernel. Each bin contributes one point to
    each curve and contributes equally to the RMSE — bin equal-weighting
    matches the "do the two curves agree" interpretation independent of bin
    population.
    """
    edges = np.asarray(bin_edges, dtype=float)
    n_bins = len(edges) - 1

    valid = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isnan(incentive))
    if upb is not None:
        w = np.minimum(np.where(np.isnan(upb), 0.0, np.maximum(upb, 0.0)), upb_cap)
        valid &= w > 0
    else:
        w = np.ones_like(y_true)

    yt = y_true[valid]
    yp = y_pred[valid]
    xv = incentive[valid]
    wv = w[valid]

    # Right-exclusive bin index in [0, n_bins-1]. The first / last bins are
    # open-bounded by construction (edges[0] = 0, edges[-1] = inf typically).
    bin_idx = np.digitize(xv, edges[1:-1], right=False)

    actual = np.full(n_bins, np.nan, dtype=float)
    predicted = np.full(n_bins, np.nan, dtype=float)
    bin_counts = np.zeros(n_bins, dtype=int)

    for k in range(n_bins):
        mask = bin_idx == k
        n = int(mask.sum())
        bin_counts[k] = n
        if n < min_rows_per_bin:
            continue
        sum_w = float(wv[mask].sum())
        if sum_w <= 0:
            continue
        actual[k] = float((wv[mask] * yt[mask]).sum() / sum_w)
        predicted[k] = float((wv[mask] * yp[mask]).sum() / sum_w)

    # Bin centers: midpoint of finite bins; for a right-unbounded last bin
    # we shift the left edge by the prior bin's width so plot scales stay sane.
    centers = np.zeros(n_bins, dtype=float)
    for k in range(n_bins):
        l, r = edges[k], edges[k + 1]
        if np.isfinite(r):
            centers[k] = 0.5 * (l + r)
        else:
            prev_width = (
                (edges[k] - edges[k - 1])
                if k >= 1 and np.isfinite(edges[k - 1])
                else 0.1
            )
            centers[k] = l + 0.5 * prev_width

    bin_left = edges[:-1]
    bin_right = edges[1:]
    is_left_tail = bin_right <= left_tail_max
    is_right_tail = bin_left >= right_tail_min
    is_mid_belly = ~is_left_tail & ~is_right_tail

    populated = ~np.isnan(actual) & ~np.isnan(predicted)

    def _seg_rmse(seg_mask: np.ndarray) -> float:
        m = populated & seg_mask
        if not m.any():
            return float("nan")
        diffs = actual[m] - predicted[m]
        return float(np.sqrt(np.mean(diffs ** 2)))

    return {
        "s_curve_rmse_overall": _seg_rmse(np.ones(n_bins, dtype=bool)),
        "s_curve_rmse_left_tail": _seg_rmse(is_left_tail),
        "s_curve_rmse_mid_belly": _seg_rmse(is_mid_belly),
        "s_curve_rmse_right_tail": _seg_rmse(is_right_tail),
        "s_curve_bins_used": int(populated.sum()),
        "s_curve_bin_edges": edges.tolist(),
        "s_curve_bin_centers": centers.tolist(),
        "s_curve_actual": [None if np.isnan(v) else float(v) for v in actual],
        "s_curve_predicted": [None if np.isnan(v) else float(v) for v in predicted],
        "s_curve_bin_counts": bin_counts.tolist(),
    }


def write_scorecard(scorecard: dict[str, Any], path: str) -> None:
    """Serialize a scorecard to JSON at the given path."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(scorecard, f, indent=2, default=str)
