# MBS CUSIP-level Prepayment Forecasting

## Overview

### Description

Forecast **SMM_DECIMAL** (Single Monthly Mortality, expressed in decimal form) for
agency GNMA MBS pools at the CUSIP level. The target measures the fraction of the
remaining pool balance that prepays in a given month. Accurate SMM_DECIMAL
forecasts drive MBS pricing, hedging, and OAS analytics.

### Objective

Build a regression model that takes monthly CUSIP-level observations as input
and predicts `SMM_DECIMAL ∈ [0, 1]` for the holdout period. Predictions must
respect MBS prepayment theory:

- **Rate sensitivity** must be monotonic in `Avg_Prop_Refi_Incentive_WAC_30yr_2mos`
  (the 30yr refinance incentive)
- **S-curve** refi response inflects in the **50–150 bps** incentive range
- **Burnout** — experiments should meaningfully use
  `Burnout_Prop_WAC_30yr_log_sum60` and/or
  `Burnout_Prop_30yr_Switch_to_15_Lag1`
- **Seasoning** on `WALA` (indicators `WALA_less_eq_6`, `WALA_bet_7_and_12`
  are also available), **seasonality** via `is_Jan`…`is_Nov` and
  **turnover** components should be identifiable

---

## Data Description

- **Panel key**: `(cusip, fh_effdt)` — one row per CUSIP per month
- **Target**: `smm_decimal ∈ [0, 1]` (stored unnormalized in the panel)
- **Features**: every GNMA feature in `gnma_feature.md` is pre-built and
  **pre-normalized to mean 0 / std 1** in the shipped panel. The coder must
  not re-normalize or re-engineer them.
- **Headline GNMA features** (see `gnma_feature.md` for the full list):
  - Gross coupon: `WAC`
  - Refinance incentives: `Avg_Prop_Refi_Incentive_WAC_30yr_2mos`,
    `Avg_Prop_Switch_To_15yr_Incentive_2mos`
  - Burnout: `Burnout_Prop_WAC_30yr_log_sum60`,
    `Burnout_Prop_30yr_Switch_to_15_Lag1`
  - Loan age: `WALA` (+ indicators `WALA_less_eq_6`, `WALA_bet_7_and_12`)
  - Collateral / credit: `CLTV`, `Orig_FICO`, `Orig_LTV`, `SATO`,
    `Coll_HPA_2yr`, `Orig_Avg_Loan_Size`
  - Servicer mix, channel mix, geography, program (FHA/VA), seasonality
- **Forbidden features** (future leakage): `future_smm`, `forward_smm`,
  `next_month_smm`, `forward_rate`, `future_rate_incentive`

### Files shipped to the coder

- `tfminput.pkl` — **single pickled DataFrame** containing the full
  CUSIP-level monthly panel (all cusips, all `fh_effdt` months, all
  normalized features, and the unnormalized `smm_decimal` target). Load with:

  ```python
  import pandas as pd
  df = pd.read_pickle("tfminput.pkl")
  ```

- `scaler.sav` — joblib-saved sklearn-style scaler fit on the GNMA feature
  columns. The scaffold uses it to inverse-transform features back to raw
  units (percent, months, etc.) for the evaluation harness. The coder
  normally does **not** need to touch it; if you do:

  ```python
  import joblib
  scaler = joblib.load("scaler.sav")   # .mean_, .scale_, .feature_names_in_
  ```

There are **no separate train/test files** — the scaffold splits on
`fh_effdt` in-memory.

### Train/Test Split

**Temporal split only** — no random shuffle. The scaffold applies:

```python
train = df[df["fh_effdt"] <= "2021-12-31"]
test  = df[df["fh_effdt"] >  "2021-12-31"]
```

---

## Evaluation

Primary metric: **RMSE of SMM_DECIMAL** on the holdout.

Before scoring, the scaffold inverse-transforms the GNMA features via
`scaler.sav`, so every diagnostic below is computed on **raw-unit** values:

- **Per-coupon-bucket RMSE** across `WAC` buckets
  `[0,3), [3,3.5), [3.5,4), [4,4.5), [4.5,5), [5,∞)`
- **Rate-sensitivity monotonicity** (Spearman rank correlation between
  predictions and raw `Avg_Prop_Refi_Incentive_WAC_30yr_2mos`)
- **S-curve R²** and inflection point in bps
- **Regime-transition RMSE** at 2013-05, 2020-03, 2022-03
- **Structural properties**: burnout ITM/non-ITM gap, seasonality residual
  range, CUSIP-differentiation std

A model that improves overall RMSE but **degrades per-coupon RMSE
uniformity**, refi-incentive monotonicity, or regime-transition RMSE is a
**REJECT**.

---

## Submission Format

The scaffold writes `submission.csv` with columns
`(cusip, fh_effdt, smm_decimal_pred)` covering every holdout row
(i.e. every row with `fh_effdt > 2021-12-31` in `tfminput.pkl`). Predictions
are automatically clipped to `[0, 1]` by the scaffold. The coder only needs
to provide a `build_model()` callable returning an unfitted
sklearn-compatible estimator.
