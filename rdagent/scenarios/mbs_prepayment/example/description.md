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

- **Rate sensitivity** must be monotonic in `Avg_Prop_Refi_Incentive_WAC_30yr_2mos`,
  a **dimensionless ratio** equal to the pool's gross coupon divided by the
  2-month average 30yr market rate
  (`WAC / avg(mortgage_rate_lag1, mortgage_rate_lag2)`). A ratio **> 1** means
  the pool coupon exceeds the recent market rate — i.e. there is a refinance
  incentive. This feature is NOT expressed in bps.
- **S-curve** refi response: the model's UPB-weighted predicted SMM bin
  curve over `Avg_Prop_Refi_Incentive_WAC_30yr_2mos` should track the
  actual UPB-weighted SMM bin curve. Quality is measured by the
  bin-level RMSE between the two curves, reported overall plus per
  segment (left tail / mid belly / right tail) so the loop can see
  which part of the S-curve needs improvement
- **Burnout** — experiments should meaningfully use
  `Burnout_Prop_WAC_30yr_log_sum60` and/or
  `Burnout_Prop_30yr_Switch_to_15_Lag1`
- **Seasoning** on `WALA` (indicators `WALA_less_eq_6`, `WALA_bet_7_and_12`
  are also available), **seasonality** via `is_Jan`…`is_Nov` and
  **turnover** components should be identifiable

---

## Data Description

- **Panel key**: `(cusip, fh_effdt)` — one row per CUSIP per month
- **Target**: `SMM_DECIMAL ∈ [0, 1]` (stored unnormalized in the panel —
  column name is **uppercase** `"SMM_DECIMAL"`, not `"smm_decimal"`)
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
    `Pool_HPA_2yr`, `Orig_Avg_Loan_Size`
  - Servicer mix, channel mix, geography, program (FHA/VA), seasonality
- **Forbidden features** (target leakage / future leakage):
  `CPR_DECIMAL` (annualised form of the target — direct leakage),
  `future_smm`, `forward_smm`, `next_month_smm`, `forward_rate`,
  `future_rate_incentive`

### Files shipped to the coder

- `tfminput.pkl` — **single pickled DataFrame** containing the full
  CUSIP-level monthly panel (all cusips, all `fh_effdt` months, all
  normalized features, and the unnormalized `SMM_DECIMAL` target column).
  Load with:

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
train = df[df["fh_effdt"] <= "2024-10-31"]
test  = df[df["fh_effdt"] >  "2024-10-31"]
```

---

## Evaluation

Primary metric: **fh_upb-weighted RMSE of SMM_DECIMAL** on the holdout,
where the per-row weight is `min(fh_upb, 150_000_000)` (larger pools
contribute more; any single pool is capped at 150M to prevent dominance).
Plain / unweighted RMSE is NOT the scoring metric and should not be used as
a training or validation loss.

Before scoring, the scaffold inverse-transforms the GNMA features via
`scaler.sav`, so every diagnostic below is computed on **raw-unit** values.
**Every RMSE figure in the scorecard is UPB-weighted** with the same
`min(fh_upb, 150M)` weights:

- **Per-coupon-bucket UPB-weighted RMSE** across `WAC` buckets
  `[0,3), [3,3.5), [3.5,4), [4,4.5), [4.5,5), [5,∞)`
- **Rate-sensitivity monotonicity** (Spearman rank correlation between
  predictions and raw `Avg_Prop_Refi_Incentive_WAC_30yr_2mos`)
- **S-curve bin RMSE** between UPB-weighted actual and predicted SMM
  bin curves over `Avg_Prop_Refi_Incentive_WAC_30yr_2mos` —
  `s_curve_rmse_overall` plus per-segment
  `s_curve_rmse_{left_tail, mid_belly, right_tail}`
- **UPB-weighted regime-transition RMSE** at 2013-05, 2020-03, 2022-03
- **Structural properties**: burnout ITM/non-ITM gap, seasonality residual
  range, CUSIP-differentiation std

A model that improves overall UPB-weighted RMSE but **degrades per-coupon
UPB-weighted RMSE uniformity**, refi-incentive monotonicity, or
UPB-weighted regime-transition RMSE is a **REJECT**.

---

## Current SOTA (beat this)

The reference model to beat is an **MLP trained with fh_upb-weighted RMSE
as the training loss** (NOT plain MSE / RMSE). Any iteration that claims to
reproduce, match, or beat the SOTA MUST use the UPB-weighted loss; a
reproduction that uses plain RMSE is not a reproduction.

- **Architecture**: PyTorch MLP, 3 hidden layers of sizes `[10, 20, 10]`,
  Leaky ReLU activations on hidden layers, Sigmoid on the final output.
- **Features**: all GNMA feature columns listed in `gnma_feature.md`,
  consumed as-is from the already-normalized `tfminput.pkl`. Do not
  re-normalize.
- **Loss (critical)**: UPB-weighted MSE / RMSE, where the per-sample weight
  is `w = min(fh_upb, 150e6)`. Equivalent forms (all acceptable):
  - PyTorch: `loss = (w * (y_pred - y_true) ** 2).sum() / w.sum()` then
    `sqrt(.)` if you want RMSE units (MSE vs RMSE is equivalent for
    gradient direction; the scorecard reports RMSE).
  - sklearn-wrapper style: `model.fit(X, y, sample_weight=w)` with
    `w = np.minimum(df_train["fh_upb"].to_numpy(), 150e6)`.
- **Optimizer / training regime**: Adam, mini-batch `batch_size ≥ 4096`,
  early-stopping on UPB-weighted *validation* RMSE using
  `w_val = np.minimum(X_val["fh_upb"].to_numpy(), 150e6)`.
- **Target**: `SMM_DECIMAL ∈ [0, 1]` directly (not CPR, not logit-
  transformed); the Sigmoid output keeps predictions in range.

Non-MLP reproductions (GBM, linear, etc.) are welcome, but the same loss
contract applies: training weight = `min(fh_upb, 150e6)`, validation
metric = UPB-weighted RMSE with the same weights.

---

## Submission Format

The scaffold writes `submission.csv` with columns
`(cusip, fh_effdt, smm_decimal_pred)` for the fixed test-CUSIP set (~1/7 of
all CUSIPs, all time rows, seed=42). The target column in the panel is
**`SMM_DECIMAL`** (uppercase). Predictions are automatically clipped to
`[0, 1]` by the scaffold. The coder only needs to provide a `build_model()`
callable returning an unfitted
sklearn-compatible estimator.
