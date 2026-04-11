# MBS CUSIP-level Prepayment Forecasting

## Overview

### Description

Forecast **SMM_DECIMAL** (Single Monthly Mortality, expressed in decimal form) for
agency MBS pools at the CUSIP level. The target measures the fraction of the
remaining pool balance that prepays in a given month. Accurate SMM_DECIMAL
forecasts drive MBS pricing, hedging, and OAS analytics.

### Objective

Build a regression model that takes monthly CUSIP-level observations as input
and predicts `SMM_DECIMAL ∈ [0, 1]` for the holdout period. Predictions must
respect MBS prepayment theory:

- **Rate sensitivity** must be monotonic in `rate_incentive = WAC - current_mortgage_rate`
- **S-curve** refi response inflects in the **50–150 bps** incentive range
- **Burnout** halflife should be in the **8–18 month** range
- **Seasoning ramp**, **seasonality**, and **turnover** components should be identifiable

---

## Data Description

- **Panel key**: `(cusip, fh_effdt)` — one row per CUSIP per month
- **Target**: `smm_decimal ∈ [0, 1]`
- **Required features**: `rate_incentive`, `coupon`, `wala`
- **Forbidden features** (future leakage): `future_smm`, `forward_smm`, `next_month_smm`, `forward_rate`, `future_rate_incentive`
- **Macro features** (unemployment, HPI, treasury curves) must be lagged by ≥ 30 days

### Train/Test Split

**Temporal split only** — no random shuffle. Training: `fh_effdt <= 2021-12-31`, Test: `fh_effdt > 2021-12-31`.

---

## Evaluation

Primary metric: **RMSE of SMM_DECIMAL** on the holdout.

The scorecard also reports:

- **Per-coupon-bucket RMSE** across buckets `[0,3), [3,3.5), [3.5,4), [4,4.5), [4.5,5), [5,∞)`
- **Rate-sensitivity monotonicity** (Spearman rank correlation with `rate_incentive`)
- **S-curve R²** and inflection point in bps
- **Regime-transition RMSE** at 2013-05, 2020-03, 2022-03
- **Structural properties**: burnout t-test, seasonality F-test, CUSIP-differentiation std

A model that improves overall RMSE but **degrades per-coupon RMSE uniformity**,
monotonicity with respect to rate_incentive, or regime-transition RMSE is a **REJECT**.

---

## Submission Format

Produce a parquet file with columns `(cusip, fh_effdt, smm_decimal_pred)` covering
every holdout row. Predictions are automatically clipped to `[0, 1]` by the scaffold.
