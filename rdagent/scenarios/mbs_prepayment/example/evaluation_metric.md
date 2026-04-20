# MBS Evaluation Harness — Metric Formulas

Exact mathematical definitions of every field written to `scores.json` by
[`MBSEvaluationHarness`](../evaluation.py). All RMSE metrics are
UPB-weighted with a per-pool cap.

---

## Notation

- $N$ samples, indexed $i = 1, \dots, N$.
- $y_i$ = true `SMM_DECIMAL`; $\hat{y}_i$ = prediction.
- UPB weight per row:
  $$w_i \;=\; \min\!\bigl(\max(0,\, u_i),\; c\bigr),
  \quad u_i = \mathrm{fh\_upb}_i \text{ (NaN} \to 0\text{)},
  \quad c = 150{,}000{,}000.$$
  If the `fh_upb` column is absent, $w_i \equiv 1$ (unweighted fallback).
- Valid mask:
  $$V \;=\; \{\, i \;:\; \lnot\mathrm{NaN}(y_i) \;\land\; \lnot\mathrm{NaN}(\hat{y}_i) \;\land\; w_i > 0 \,\}.$$
- For any subset mask $M$, every formula below restricts to $i \in M \cap V$.

### Weighted-RMSE kernel

Used by every RMSE field in the scorecard
([`_rmse_w`](../evaluation.py#L306)):

$$
\mathrm{RMSE}_w(M) \;=\;
\sqrt{\dfrac{\displaystyle\sum_{i \,\in\, M \cap V} w_i\, (y_i - \hat{y}_i)^2}
{\displaystyle\sum_{i \,\in\, M \cap V} w_i}}
\qquad
\text{(NaN if } M \cap V = \varnothing\text{).}
$$

Unweighted fallback (no `fh_upb` column):

$$
\mathrm{RMSE}(M) \;=\; \sqrt{\dfrac{1}{|M \cap V|} \sum_{i \in M \cap V} (y_i - \hat{y}_i)^2}.
$$

---

## `accuracy` dimension

### `overall_rmse` &nbsp;·&nbsp; [L152](../evaluation.py#L152)

$$
\mathrm{overall\_rmse} \;=\; \mathrm{RMSE}_w(\text{all rows}).
$$

### `oot_rmse` &nbsp;·&nbsp; [L158](../evaluation.py#L158)

With cutoff $T_\mathrm{cut} = \text{2024-10-31}$:

$$
M_\mathrm{OOT} \;=\; \{\, i \;:\; \mathrm{fh\_effdt}_i > T_\mathrm{cut} \,\},
\qquad
\mathrm{oot\_rmse} \;=\; \mathrm{RMSE}_w(M_\mathrm{OOT}).
$$

### `rmse_by_coupon_bucket[b]` &nbsp;·&nbsp; [L169](../evaluation.py#L169)

For each bucket $b = [\ell_b, h_b)$:

$$
M_b \;=\; \{\, i \;:\; \ell_b \le \mathrm{WAC}_i < h_b \,\},
\qquad
\mathrm{RMSE}_w(M_b).
$$

Default buckets: $[0, 3),\; [3, 3.5),\; [3.5, 4),\; [4, 4.5),\; [4.5, 5),\; [5, \infty)$.

### `rmse_tail_high`, `rmse_tail_low` &nbsp;·&nbsp; [L160–L165](../evaluation.py#L160-L165)

$$
q_{90} \;=\; \mathrm{quantile}(y,\, 0.90),
\qquad
q_{10} \;=\; \mathrm{quantile}(y,\, 0.10).
$$

$$
\mathrm{tail\_high} \;=\; \mathrm{RMSE}_w\bigl(\{\, i \,:\, y_i \ge q_{90} \,\}\bigr),
\qquad
\mathrm{tail\_low} \;=\; \mathrm{RMSE}_w\bigl(\{\, i \,:\, y_i \le q_{10} \,\}\bigr).
$$

### `rmse_by_vintage[v]` &nbsp;·&nbsp; [L192](../evaluation.py#L192)

Origination vintage year inferred from WALA:

$$
v_i \;=\; \mathrm{year}(\mathrm{fh\_effdt}_i) \;-\; \lfloor \mathrm{WALA}_i / 12 \rfloor.
$$

For each vintage $v$ with $|\{\, i \,:\, v_i = v \,\}| \ge 30$:

$$
\mathrm{rmse\_by\_vintage}[v] \;=\; \mathrm{RMSE}_w\bigl(\{\, i \,:\, v_i = v \,\}\bigr).
$$

---

## `rate_sensitivity` dimension

Let $x_i = \mathrm{Avg\_Prop\_Refi\_Incentive\_WAC\_30yr\_2mos}_i$
(restricted to rows where both $x_i$ and $\hat{y}_i$ are non-NaN; requires
$\ge 10$ valid rows).

### `monotonicity_spearman` &nbsp;·&nbsp; [`_spearman`](../evaluation.py#L340)

Let $R(\cdot)$ be the average-rank transform, with
$r^x_i = R(x)_i$ and $r^y_i = R(\hat{y})_i$. Then

$$
\rho_S \;=\;
\dfrac{\dfrac{1}{n} \displaystyle\sum_i \bigl(r^x_i - \bar{r}^x\bigr)\bigl(r^y_i - \bar{r}^y\bigr)}
{\sigma_{r^x}\,\sigma_{r^y}},
$$

with **population** standard deviation ($\mathrm{ddof} = 0$).
Returns $0$ if either $\sigma < 10^{-12}$.

### `s_curve_r2` and `inflection_point_ratio` &nbsp;·&nbsp; [`_fit_logistic_s_curve`](../evaluation.py#L350)

> **Units of $x$:** `Avg_Prop_Refi_Incentive_WAC_30yr_2mos` is a
> **dimensionless ratio** equal to
> $\mathrm{WAC} / \mathrm{avg}(\mathrm{mortgage\_rate}_{t-1}, \mathrm{mortgage\_rate}_{t-2})$.
> A value $>1$ means the pool coupon exceeds the recent market rate — i.e.
> refi incentive. It is **not** in bps. Every quantity derived from $x$
> (including the inflection below) inherits these ratio units.

**Step 1 — normalize predictions to $[0, 1]$:**

$$
\tilde{y}_i \;=\; \dfrac{\hat{y}_i - \min \hat{y}}{\max \hat{y} - \min \hat{y} + 10^{-12}},
\qquad
\tilde{y}_i \;\leftarrow\; \mathrm{clip}\!\bigl(\tilde{y}_i,\, 10^{-4},\, 1 - 10^{-4}\bigr).
$$

If $\max \hat{y} - \min \hat{y} < 10^{-8}$, return $R^2 = 0$ and
$\mathrm{inflection\_ratio} = 0$.

**Step 2 — logit linearization:**

$$
z_i \;=\; \ln\!\dfrac{\tilde{y}_i}{1 - \tilde{y}_i}.
$$

**Step 3 — closed-form OLS of $z$ on $x$:**

$$
\beta \;=\; \dfrac{\sum_i (x_i - \bar{x})(z_i - \bar{z})}{\sum_i (x_i - \bar{x})^2},
\qquad
\alpha \;=\; \bar{z} - \beta\,\bar{x},
\qquad
\hat{z}_i \;=\; \alpha + \beta\, x_i.
$$

**Step 4 — $R^2$:**

$$
R^2 \;=\; \mathrm{clip}\!\left(
1 - \dfrac{\sum_i (z_i - \hat{z}_i)^2}{\max\bigl(\sum_i (z_i - \bar{z})^2,\; 10^{-12}\bigr)},\;
0,\; 1
\right).
$$

**Step 5 — inflection point (same ratio units as $x$; not bps):**

$$
\mathrm{inflection\_ratio} \;=\;
\begin{cases}
-\,\alpha / \beta & \text{if } |\beta| > 10^{-12},\\
0 & \text{otherwise.}
\end{cases}
$$

> **Note:** the $z$ on $x$ regression is in logit-of-prediction space; the
> inflection is the $x$ (in refi-incentive ratio units) at which the fitted
> logistic crosses the mid-point of the normalized prediction range. A
> well-calibrated S-curve sits slightly above the refi boundary —
> $\mathrm{inflection\_ratio} \in [1.00,\, 1.20]$.

---

## `temporal_robustness` dimension

### `regime_transition_rmse[T_k]` &nbsp;·&nbsp; [L246–L253](../evaluation.py#L246-L253)

For each transition date $T_k \in \{\text{2013-05-01},\; \text{2020-03-01},\; \text{2022-03-01}\}$:

$$
M_{T_k} \;=\; \{\, i \;:\; T_k \le \mathrm{fh\_effdt}_i < T_k + 3\ \text{months} \,\}.
$$

If $|M_{T_k}| \ge 10$:

$$
\mathrm{regime\_transition\_rmse}[T_k] \;=\; \mathrm{RMSE}_w(M_{T_k}).
$$

### Rolling 12-month RMSE stability &nbsp;·&nbsp; [L255–L271](../evaluation.py#L255-L271)

**Step 1 — per-month UPB-weighted MSE.** For each calendar month $m$:

$$
\mathrm{WMSE}_m \;=\; \dfrac{\sum_{i \in m \cap V} w_i\,(y_i - \hat{y}_i)^2}{\sum_{i \in m \cap V} w_i}.
$$

**Step 2 — 12-month rolling mean of MSE, then $\sqrt{\cdot}$** (ordered on
consecutive months $m_1 < m_2 < \dots$):

$$
\mathrm{RollRMSE}_m \;=\; \sqrt{\;\dfrac{1}{12}\sum_{j = m-11}^{m} \mathrm{WMSE}_j\;}.
$$

**Step 3 — outputs** (discard leading months with < 12 observations):

$$
\mathrm{rolling\_12m\_rmse\_max} \;=\; \max_m \mathrm{RollRMSE}_m,
$$
$$
\mathrm{rolling\_12m\_rmse\_min} \;=\; \min_m \mathrm{RollRMSE}_m,
$$
$$
\mathrm{rolling\_12m\_rmse\_ratio} \;=\;
\dfrac{\max_m \mathrm{RollRMSE}_m}{\max\!\bigl(\min_m \mathrm{RollRMSE}_m,\; 10^{-12}\bigr)}.
$$

> **Note:** this is *mean of MSE, then square root* — **not** mean of RMSE.
> The two are not the same when monthly MSEs differ in scale.

---

## `structural_properties` dimension

### `burnout_effect_mean_itm`, `burnout_effect_mean_non_itm` &nbsp;·&nbsp; [L281–L287](../evaluation.py#L281-L287)

In-the-money set (pool coupon exceeds recent avg 30yr market rate — i.e.
refi-incentive ratio $> 1$):

$$
I \;=\; \{\, i \;:\; x_i > 1 \,\},
\qquad x_i = \mathrm{Avg\_Prop\_Refi\_Incentive\_WAC\_30yr\_2mos}_i.
$$

Require $|I| \ge 30$ **and** $|I^c| \ge 30$, then:

$$
\overline{\hat{y}}_\mathrm{ITM} \;=\; \mathrm{nanmean}\bigl(\{\, \hat{y}_i \,:\, i \in I \,\}\bigr),
$$
$$
\overline{\hat{y}}_{\lnot \mathrm{ITM}} \;=\; \mathrm{nanmean}\bigl(\{\, \hat{y}_i \,:\, i \in I^c \,\}\bigr).
$$

A well-behaved model has $\overline{\hat{y}}_\mathrm{ITM} > \overline{\hat{y}}_{\lnot \mathrm{ITM}}$.

### `seasonality_residual_range` &nbsp;·&nbsp; [L290–L296](../evaluation.py#L290-L296)

Residuals and month-of-year:

$$
e_i \;=\; y_i - \hat{y}_i,
\qquad
m_i \;=\; \mathrm{month}(\mathrm{fh\_effdt}_i) \in \{1, 2, \dots, 12\}.
$$

Mean residual per month-of-year:

$$
\mu_m \;=\; \mathrm{mean}\bigl(\{\, e_i \,:\, m_i = m,\; \text{both non-NaN} \,\}\bigr).
$$

If at least $6$ distinct months are present:

$$
\mathrm{seasonality\_residual\_range} \;=\; \max_m \mu_m \;-\; \min_m \mu_m.
$$

### `cusip_differentiation_std` &nbsp;·&nbsp; [L299–L302](../evaluation.py#L299-L302)

Per-CUSIP mean prediction:

$$
\overline{\hat{y}}_c \;=\; \mathrm{mean}\bigl(\{\, \hat{y}_i \,:\, \mathrm{cusip}_i = c \,\}\bigr).
$$

Sample standard deviation across CUSIPs ($\mathrm{ddof} = 1$):

$$
\mathrm{cusip\_differentiation\_std} \;=\; \mathrm{std}\bigl(\{\, \overline{\hat{y}}_c \,:\, c \in \mathrm{CUSIPs} \,\}\bigr).
$$

A near-zero value means the model has collapsed to a CUSIP-agnostic predictor.

---

## `primary_metric`

$$
\text{value} \;=\; \mathrm{overall\_rmse},
\qquad
\text{lower\_is\_better} \;=\; \text{True}.
$$
