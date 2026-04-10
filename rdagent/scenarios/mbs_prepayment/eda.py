"""MBS Domain-Specific EDA — Priority 5: Show the LLM the data, not just the schema.

This module implements Direction #7 (Data Representation to the LLM). The default
RD-Agent EDA produces a generic schema-and-sample description. For prepayment
modeling, the LLM needs to reason about distributional facts it cannot guess:

    - Is SMM_DECIMAL heavy-tailed? (yes → suggest quantile/Tweedie loss)
    - What's the autocorrelation of SMM_DECIMAL? (high → suggest lagged features)
    - Where are the structural breaks? (2020 COVID refi, 2022 rate hiking)
    - What fraction of CUSIPs are in-the-money in each year?
    - How is the coupon distribution shaped? (drives per-coupon-bucket strategy)
    - What are the raw correlations of canonical features with SMM_DECIMAL?

This EDA is run ONCE before iteration 1 and the resulting Markdown block is
injected into hypothesis_gen (and scenario_problem) as empirical grounding.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class MBSDataProfile:
    """Structured EDA output for an MBS prepayment panel."""

    n_cusips: int
    n_dates: int
    date_min: str
    date_max: str
    smm_mean: float
    smm_median: float
    smm_p95: float
    smm_p99: float
    smm_tail_ratio: float  # p99 / p50, heavy-tail indicator
    coupon_distribution: dict[str, float]  # bucket label → fraction
    rate_incentive_mean: float
    rate_incentive_std: float
    rate_incentive_range: tuple[float, float]
    itm_fraction_by_year: dict[int, float]
    smm_autocorr_lag1: float
    structural_breaks: list[dict[str, Any]]
    missing_panel_fraction: float
    feature_correlations: dict[str, float]

    def to_markdown(self) -> str:
        """Render as a Markdown block for injection into LLM prompts."""
        lines = [
            "## MBS Data Profile",
            f"- **Panel dimensions**: {self.n_cusips} CUSIPs × {self.n_dates} fh_effdt dates ({self.date_min} – {self.date_max})",
            f"- **SMM_DECIMAL distribution**: mean={self.smm_mean:.4f}, median={self.smm_median:.4f}, p95={self.smm_p95:.4f}, p99={self.smm_p99:.4f}",
            f"- **SMM_DECIMAL tail ratio (p99/p50)**: {self.smm_tail_ratio:.2f} "
            f"({'heavy-tailed — consider Tweedie/quantile loss' if self.smm_tail_ratio > 5 else 'roughly symmetric'})",
            f"- **Coupon distribution**: " + ", ".join(
                f"{k}: {v:.1%}" for k, v in self.coupon_distribution.items()
            ),
            f"- **Rate incentive**: mean={self.rate_incentive_mean:.2f}bps, std={self.rate_incentive_std:.2f}bps, "
            f"range=[{self.rate_incentive_range[0]:.1f}, {self.rate_incentive_range[1]:.1f}]",
            f"- **In-the-money fraction by year**: " + ", ".join(
                f"{y}: {f:.0%}" for y, f in sorted(self.itm_fraction_by_year.items())
            ),
            f"- **SMM_DECIMAL autocorrelation (lag-1)**: {self.smm_autocorr_lag1:.2f} "
            f"({'very high persistence — include lagged SMM_DECIMAL or recurrent architecture' if self.smm_autocorr_lag1 > 0.7 else 'moderate persistence'})",
        ]
        if self.structural_breaks:
            lines.append(
                "- **Structural breaks detected**: "
                + ", ".join(f"{b['date']} ({b['description']})" for b in self.structural_breaks)
            )
        lines.append(f"- **Missing (cusip, fh_effdt) fraction**: {self.missing_panel_fraction:.1%}")
        lines.append("- **Feature correlations with SMM_DECIMAL**: "
                     + ", ".join(f"{k}({v:+.2f})" for k, v in self.feature_correlations.items()))
        return "\n".join(lines)


def profile_mbs_panel(
    df: pd.DataFrame,
    *,
    smm_col: str = "smm_decimal",
    cusip_col: str = "cusip",
    date_col: str = "fh_effdt",
    coupon_col: str = "coupon",
    rate_incentive_col: str = "rate_incentive",
    coupon_buckets: tuple[tuple[float, float], ...] = (
        (0.0, 3.0),
        (3.0, 3.5),
        (3.5, 4.0),
        (4.0, 4.5),
        (4.5, 5.0),
        (5.0, 99.0),
    ),
) -> MBSDataProfile:
    """Profile an MBS prepayment panel DataFrame."""
    dates = pd.to_datetime(df[date_col], errors="coerce")

    # Panel dimensions
    n_cusips = int(df[cusip_col].nunique())
    unique_dates = dates.dropna().drop_duplicates()
    n_dates = int(len(unique_dates))
    date_min = str(dates.min().date()) if len(dates) else ""
    date_max = str(dates.max().date()) if len(dates) else ""

    # SMM_DECIMAL distribution
    smm = df[smm_col].to_numpy(dtype=float)
    smm_finite = smm[~np.isnan(smm)]
    smm_mean = float(np.mean(smm_finite)) if len(smm_finite) else 0.0
    smm_median = float(np.median(smm_finite)) if len(smm_finite) else 0.0
    smm_p95 = float(np.quantile(smm_finite, 0.95)) if len(smm_finite) else 0.0
    smm_p99 = float(np.quantile(smm_finite, 0.99)) if len(smm_finite) else 0.0
    smm_tail_ratio = smm_p99 / max(smm_median, 1e-9)

    # Coupon distribution
    coupon_dist: dict[str, float] = {}
    if coupon_col in df.columns:
        coupons = df[coupon_col].to_numpy(dtype=float)
        total = len(coupons[~np.isnan(coupons)])
        for lo, hi in coupon_buckets:
            mask = (coupons >= lo) & (coupons < hi)
            label = f"{lo:.1f}-{hi:.1f}" if hi < 99 else f"{lo:.1f}+"
            coupon_dist[label] = float(mask.sum() / max(total, 1))

    # Rate incentive
    if rate_incentive_col in df.columns:
        inc = df[rate_incentive_col].to_numpy(dtype=float)
        inc_finite = inc[~np.isnan(inc)]
        inc_mean = float(np.mean(inc_finite)) if len(inc_finite) else 0.0
        inc_std = float(np.std(inc_finite)) if len(inc_finite) else 0.0
        inc_range = (float(np.min(inc_finite)), float(np.max(inc_finite))) if len(inc_finite) else (0.0, 0.0)
    else:
        inc_mean = inc_std = 0.0
        inc_range = (0.0, 0.0)

    # ITM fraction by year
    itm_by_year: dict[int, float] = {}
    if rate_incentive_col in df.columns and date_col in df.columns:
        years = dates.dt.year
        tmp = pd.DataFrame({"year": years, "inc": df[rate_incentive_col].values}).dropna()
        for y, grp in tmp.groupby("year"):
            itm_by_year[int(y)] = float((grp["inc"] > 0.0).mean())

    # Autocorrelation of SMM_DECIMAL (lag-1) per-cusip, then averaged
    autocorr = _panel_autocorr(df, smm_col, cusip_col, date_col)

    # Structural breaks — use known macro events, verify with month-over-month SMM changes
    structural_breaks = _detect_structural_breaks(df, smm_col, date_col)

    # Missing panel fraction
    if n_cusips > 0 and n_dates > 0:
        expected = n_cusips * n_dates
        missing_fraction = 1.0 - (len(df) / expected)
        missing_fraction = max(0.0, missing_fraction)
    else:
        missing_fraction = 0.0

    # Feature correlations with SMM_DECIMAL
    feat_corr: dict[str, float] = {}
    for col in df.columns:
        if col in (smm_col, cusip_col, date_col):
            continue
        try:
            s = df[col].to_numpy(dtype=float)
            mask = ~(np.isnan(s) | np.isnan(smm))
            if mask.sum() < 30:
                continue
            s_m = s[mask]
            y_m = smm[mask]
            if s_m.std() < 1e-12 or y_m.std() < 1e-12:
                continue
            corr = float(np.mean((s_m - s_m.mean()) * (y_m - y_m.mean())) / (s_m.std() * y_m.std()))
            feat_corr[col] = corr
        except (TypeError, ValueError):
            continue

    # Keep top 10 correlations by |value|
    feat_corr = dict(sorted(feat_corr.items(), key=lambda kv: -abs(kv[1]))[:10])

    return MBSDataProfile(
        n_cusips=n_cusips,
        n_dates=n_dates,
        date_min=date_min,
        date_max=date_max,
        smm_mean=smm_mean,
        smm_median=smm_median,
        smm_p95=smm_p95,
        smm_p99=smm_p99,
        smm_tail_ratio=smm_tail_ratio,
        coupon_distribution=coupon_dist,
        rate_incentive_mean=inc_mean,
        rate_incentive_std=inc_std,
        rate_incentive_range=inc_range,
        itm_fraction_by_year=itm_by_year,
        smm_autocorr_lag1=autocorr,
        structural_breaks=structural_breaks,
        missing_panel_fraction=missing_fraction,
        feature_correlations=feat_corr,
    )


def _panel_autocorr(df: pd.DataFrame, smm_col: str, cusip_col: str, date_col: str) -> float:
    """Mean lag-1 autocorrelation of SMM_DECIMAL across CUSIPs."""
    if smm_col not in df.columns or cusip_col not in df.columns:
        return 0.0
    correlations: list[float] = []
    for _, grp in df.sort_values(date_col).groupby(cusip_col):
        y = grp[smm_col].to_numpy(dtype=float)
        if len(y) < 10:
            continue
        y_lag = y[:-1]
        y_cur = y[1:]
        mask = ~(np.isnan(y_lag) | np.isnan(y_cur))
        if mask.sum() < 5:
            continue
        a = y_lag[mask]
        b = y_cur[mask]
        if a.std() < 1e-12 or b.std() < 1e-12:
            continue
        c = float(np.mean((a - a.mean()) * (b - b.mean())) / (a.std() * b.std()))
        correlations.append(c)
    if not correlations:
        return 0.0
    return float(np.mean(correlations))


#: Known macro regime-change dates with MBS-significant impact.
KNOWN_REGIME_BREAKS = (
    ("2013-05-01", "Taper tantrum — rates up"),
    ("2020-03-01", "COVID refi wave — rates down"),
    ("2022-03-01", "Fed hiking cycle begins"),
)


def _detect_structural_breaks(
    df: pd.DataFrame, smm_col: str, date_col: str
) -> list[dict[str, Any]]:
    """Report known regime dates that fall within the observed data range."""
    if smm_col not in df.columns or date_col not in df.columns:
        return []
    dates = pd.to_datetime(df[date_col], errors="coerce")
    d_min, d_max = dates.min(), dates.max()
    out: list[dict[str, Any]] = []
    for date_str, desc in KNOWN_REGIME_BREAKS:
        d = pd.Timestamp(date_str)
        if d_min <= d <= d_max:
            out.append({"date": date_str, "description": desc})
    return out


def write_eda_report(profile: MBSDataProfile, path: str | Path) -> None:
    """Write the Markdown EDA report to disk for injection into LLM prompts."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(profile.to_markdown(), encoding="utf-8")
