# MBS CUSIP-Level Prepayment Forecast: RD-Agent Data Science Loop Customization

## Context

This document captures a comprehensive analysis of how to customize the RD-Agent data science loop to build a model for mortgage-backed securities (MBS) CUSIP-level prepayment forecasting. The target variable is **SMM_DECIMAL** (single monthly mortality in decimal form). Observations are keyed by `(cusip, fh_effdt)` where `cusip` identifies the MBS pool and `fh_effdt` is the factor-history effective date. The primary evaluation metric is **RMSE of SMM_DECIMAL**. The customization directions cover: (1) system prompts, (2) context window and memory management, (3) orchestration design, and extensions into evaluation function, code scaffold, search strategy, data representation, interpretability, execution environment, and multi-agent specialization.

---

## Part 1: The Three Core Directions

### Direction 1: Customize System Prompts

#### 1A. Replace `scen.role` with MBS-specific persona

The current shared role in `rdagent/scenarios/data_science/share.yaml` says "Kaggle Grandmaster." For MBS, every prompt that includes `{% include "scenarios.data_science.share:scen.role" %}` should instead inject:

> You are an expert quantitative prepayment modeler with deep knowledge of agency MBS (FNMA, FHLMC, GNMA), the PSA prepayment convention, and the Richard-Roll decomposition framework. You understand that prepayment speed (SMM_DECIMAL — single monthly mortality in decimal form) is driven by four distinct components: (1) **turnover** — housing activity, seasonal, and demographic-driven, (2) **refinancing** — rate-incentive-driven with an S-curve response and burnout decay, (3) **curtailment** — partial prepayments increasing with loan age, (4) **defaults/involuntary** — credit-driven, correlated with unemployment and HPI. You know that CUSIP-level features (WAC, WAM, WALA, loan count, avg FICO, avg LTV, geographic concentration, loan purpose mix, original balance, coupon) interact non-linearly with macro drivers (current mortgage rate, treasury curve shape, HPI, unemployment rate, seasonal indicators).

This is not cosmetic — it prevents the LLM from proposing generic gradient-boosted trees on flattened features and steers it toward structured prepayment decomposition from iteration 1.

#### 1B. Domain-inject `hypothesis_specification` in `prompts_v2.yaml`

The `hypothesis_specification` variable (injected into `hypothesis_gen` and `direct_exp_gen`) currently contains 3 generic guidelines. For MBS, replace with:

```
1. Hypotheses must target a specific prepayment component (turnover, refinancing,
   curtailment, or default) or a specific interaction (e.g., "burnout moderates
   the refinancing S-curve slope after 6 months of elevated incentive").
2. Rate-related hypotheses must specify: which rate (primary mortgage rate,
   10yr treasury, spread), the functional form being tested (linear, piecewise
   linear, logistic S-curve, spline), and the expected direction of effect.
3. Feature engineering hypotheses must distinguish between static CUSIP features
   (origination WAC, original LTV, coupon) and dynamic features (current WAM,
   current rate incentive, burnout index, seasoning ramp) — the model must
   handle both.
4. Never propose "try XGBoost/LightGBM on all features" as a hypothesis. Every
   iteration must have a prepayment-theoretic justification grounded in the
   Richard-Roll framework or empirical prepayment literature.
```

#### 1C. Restructure `scenario_problem` and `feedback_problem` prompts

The `scenario_problem` prompt (lines 1-53 of `prompts_v2.yaml`) asks the LLM to identify "challenges" from the scenario description. For MBS, the scenario description itself needs to encode domain-specific evaluation criteria:

**Scenario description should include:**
- Target variable definition: monthly CUSIP-level SMM_DECIMAL (single monthly mortality expressed as a decimal in `[0.0, 1.0]`)
- Observation keys: `(cusip, fh_effdt)` — the factor-history effective date is the observation timestamp
- Evaluation metrics hierarchy: (1) overall RMSE of SMM_DECIMAL across CUSIPs and effective dates, (2) per-coupon-bucket RMSE (critical — prepayment behavior is heavily coupon-dependent), (3) S-curve shape fidelity — correlation between predicted and actual prepayment response to rate incentive, (4) regime transition error — RMSE in first 3 months after Fed rate changes, (5) cohort stability — max variance of RMSE across vintage buckets
- Known modeling pitfalls: look-ahead bias from using same-`fh_effdt` macro data, survivorship bias from dropped CUSIPs, non-stationarity of refinancing efficiency over decades

The `feedback_problem` prompt should be extended to check for MBS-specific failure modes:
- "Is the model's rate sensitivity monotonic and reasonable? (higher incentive → higher prepayment)"
- "Does the model show burnout decay — does refi response decrease for CUSIPs that have been in-the-money for many months?"
- "Is the seasonal pattern captured — are summer months showing higher turnover?"
- "Does the model extrapolate dangerously outside the training rate regime?"
- "Does per-coupon RMSE show that high-coupon (deeply in-the-money) buckets are well modeled?"

#### 1D. Component specs (`task_specification`)

The current `component_spec` in `share.yaml` defines generic DataLoader/Feature/Model specs. For MBS:

**DataLoadSpec** should mandate:
- Panel data structure: `(cusip, fh_effdt)` as the observation unit
- Feature groups: static CUSIP characteristics (including coupon), dynamic CUSIP state, macro environment, derived interaction terms
- Temporal alignment: all macro features must be lagged by at least 1 month relative to `fh_effdt` to prevent look-ahead

**FeatureEng** spec should list canonical prepayment features:
- `rate_incentive = WAC - current_mortgage_rate` (the single most important predictor)
- `coupon_bucket` — categorical binning of CUSIP coupon (e.g., [<3.0, 3.0-3.5, 3.5-4.0, 4.0-4.5, 4.5-5.0, 5.0+])
- `burnout_index = months_since_first_in_the_money / total_months_in_the_money` (or exponential decay variant)
- `seasoning_ramp = min(WALA, 30) / 30` (prepayment ramps up in first 30 months)
- `yield_curve_slope = 10yr_treasury - 2yr_treasury`
- `refi_efficiency = moving_average(market_refi_share, 3)` (media effect proxy)
- Seasonal dummies or cyclical encoding of `fh_effdt` month-of-year

**Model** spec should enforce:
- Output in `[0.0, 1.0]` range (SMM_DECIMAL is a decimal probability)
- Must handle panel data — no flattening CUSIPs into independent observations without CUSIP fixed effects or pooling strategy
- Train/test split must be temporal on `fh_effdt` (e.g., train on 2010-2020, test on 2021-2023) — **never random split**

#### 1E. Feedback evaluation prompt (`dev/prompts.yaml`)

The current `exp_feedback` output schema has `{New Result, Best Result, Replace Best Result, ...}`. Add MBS-specific evaluation fields:

```json
{
  "rate_sensitivity_check": "Does the model show increasing SMM_DECIMAL with increasing rate incentive? [Pass/Fail with explanation]",
  "burnout_check": "For CUSIPs with >12 months of positive rate incentive, does predicted SMM_DECIMAL decrease relative to fresh CUSIPs? [Pass/Fail]",
  "temporal_stability": "Is the RMSE on the holdout period within 1.5x of the training RMSE? [Pass/Fail with numbers]",
  "extreme_regime_check": "When rate incentive exceeds +200bps, does predicted SMM_DECIMAL plateau or explode? [Reasonable/Dangerous]",
  "coupon_bucket_check": "Is RMSE reasonably uniform across coupon buckets, or concentrated in specific coupon ranges? [report per-bucket RMSE]"
}
```

---

### Direction 2: Context Window and Memory Management

#### 2A. Domain knowledge pre-loading (RAG corpus)

The CoSTEER coder uses `RAGStrategy` to retrieve "similar successful implementations" and "former failed implementations." For MBS, seed the knowledge base with:

1. **Canonical prepayment model implementations**: A reference implementation of a 2-component (turnover + refi) model with proper S-curve and burnout. This ensures the coder always has a high-quality anchor, even in iteration 1.

2. **Known failure patterns**: Catalog of common MBS modeling mistakes with their symptoms:
   - "Model predicts negative SMM_DECIMAL or values > 1.0" → missing output clipping
   - "Validation RMSE is 3x training" → likely look-ahead bias or random split
   - "Model misses 2020 refi wave" → burnout feature not decaying properly
   - "All CUSIPs predict same SMM_DECIMAL" → CUSIP-level features not being used, only macro
   - "High-coupon bucket RMSE dominates overall error" → S-curve saturation not modeled

3. **Feature engineering recipes**: Rate incentive calculation, burnout index variants (exponential, linear, threshold-based), seasoning ramp functions, media effect proxies, coupon bucket construction.

This corpus should be **loaded before any iteration starts**, not discovered organically. The RD-Agent `KnowledgeManagement` component supports this — inject it via `similar_successful_knowledge` in the coder prompts.

#### 2B. Trace compression strategy

The current trace (in `describe.trace` template) shows the last 10 experiments with: problem, component, hypothesis, code_change_summary, decision, score, feedback. For MBS, 10 generic entries will quickly fill context. Customize the trace to:

1. **Group by prepayment component**: Instead of chronological last-10, show: best experiment per component (turnover, refi, curtailment, ensemble), plus the 3 most recent failures. This gives the LLM structural memory of what's been tried per component.

2. **Compress code diffs**: MBS model code can be large (data loading, feature eng, model definition, training loop, evaluation). The diff shown in the trace should be **summarized to only the mathematical changes**: "Changed refi response from `linear(rate_incentive)` to `sigmoid(rate_incentive, steepness=2.5)`" — not the full diff.

3. **Persist key model properties as structured state**: After each iteration, extract and store:
   ```json
   {
     "rate_sensitivity_slope": 0.0083,
     "burnout_halflife_months": 8,
     "seasonal_amplitude": 0.0124,
     "holdout_rmse": 0.0183,
     "holdout_rmse_by_coupon": {"<3.0": 0.009, "3.0-3.5": 0.012, "3.5-4.0": 0.018, "4.0-4.5": 0.024, "4.5+": 0.031},
     "n_features_used": 14,
     "model_type": "LightGBM + S-curve overlay"
   }
   ```
   Pass this structured summary (not raw code) into hypothesis generation. The LLM can reason much better over "burnout halflife is 8 months, literature suggests 12-18" than over reading the full model code.

#### 2C. Feature attribution memory (ties to IG/Captum plan)

After each successful experiment, run Integrated Gradients (IG) and store the top-10 feature attributions. Feed this into the next iteration's `hypothesis_gen`:

```
Previous model's top feature attributions (by |IG|):
1. rate_incentive: +0.00042 (higher incentive → higher SMM_DECIMAL, as expected)
2. WALA: +0.00018 (older loans prepay more — seasoning effect)
3. burnout_index: -0.00015 (burnout reduces SMM_DECIMAL — correct direction)
4. unemployment_rate: -0.00008 (unexpected — usually negative relationship is stronger)
...
```

This gives the hypothesis generator **empirical grounding** — it can propose "The unemployment effect is underfit; adding state-level unemployment instead of national may improve geographic heterogeneity" rather than generic guesses.

#### 2D. Sliding context window by iteration phase

Not all context is needed at all times. Implement phase-dependent context loading:

| Iteration Phase | Context Loaded | Context Excluded |
|---|---|---|
| Hypothesis gen | Structured model properties, feature attributions, last 3 feedbacks | Full code, full diffs |
| Coding | Full SOTA code, component spec, 2 RAG examples | Old experiment history |
| Feedback eval | SOTA results, current results, code diff | RAG knowledge, hypothesis history |

This prevents the 128K window from filling with stale experiment traces during the coding phase when the LLM needs maximum space for code reasoning.

---

### Direction 3: Orchestration Design

#### 3A. Replace generic component decomposition

The current 5-component split (`DataLoadSpec`, `FeatureEng`, `Model`, `Ensemble`, `Workflow`) maps poorly to MBS. Replace with:

```
MBS Components:
├── DataLoader          # Load CUSIP-level panel data + macro time series
├── RateCurveFeatures   # Mortgage rate, treasury curve, spread, incentive
├── PoolDynamics        # Burnout, seasoning, age-based ramps, vintage, coupon bucket
├── MacroFeatures       # Unemployment, HPI, refi index, seasonal encoding
├── PrepaymentModel     # Core model (can be single unified or decomposed)
├── ScenarioValidator   # Run model under rate shock scenarios, sanity check
└── Workflow            # Orchestrate train/predict/evaluate pipeline
```

The key insight: **feature engineering is not one component for MBS — it's three** (rate curve, CUSIP dynamics, macro), and they evolve independently. A hypothesis about burnout modeling shouldn't trigger re-generation of rate curve features.

#### 3B. Multi-phase iteration strategy with hard gates

Instead of the current stage-awareness (Draft → Improvement → Ensemble, determined by `use_ratio`), implement MBS-specific phases with **hard gates** (must pass validation before advancing):

**Phase 1: Baseline (iterations 1-3)**
- Goal: Establish a working pipeline with `rate_incentive` as primary feature, simple model (Ridge or small GBM)
- Gate: Model produces valid SMM_DECIMAL in `[0.0, 1.0]`, temporal train/test split works, holdout RMSE < 0.040
- Constraint: `hypothesis_select` only allows `DataLoadSpec` and `FeatureEng` components

**Phase 2: Rate Response (iterations 4-8)**
- Goal: Get the S-curve right — nonlinear refinancing response to rate incentive
- Gate: Rate sensitivity is monotonically positive, S-curve inflection point is in the `[50bps, 150bps]` incentive range
- Constraint: Hypotheses must target refinancing component; model must demonstrate S-curve shape
- Validation: Plot predicted SMM_DECIMAL vs rate incentive, verify sigmoidal shape

**Phase 3: Dynamics (iterations 9-14)**
- Goal: Add burnout, seasoning, vintage effects, seasonal patterns
- Gate: Burnout effect is statistically significant (CUSIPs in-the-money >12 months predict lower than fresh), seasonal pattern present
- Constraint: Each iteration adds exactly one dynamic feature; ablation test confirms it helps

**Phase 4: Macro & Regime (iterations 15-18)**
- Goal: Add macro environment sensitivity, test regime transition performance
- Gate: Model RMSE during 2020 refi wave is within 2x of overall RMSE
- Validation: Out-of-time test specifically on regime transitions (2013 taper tantrum, 2020 COVID refi, 2022 rate hiking)

**Phase 5: Ensemble & Robustness (iterations 19-22)**
- Goal: Ensemble multiple model architectures, stress test
- Gate: Ensemble improves over best single model; no individual CUSIP has RMSE > 3x average

Implementation: Override `hypothesis_select` in `prompts_v2.yaml` to inject the current phase's constraints. The `plan` object already has `draft`/`suggest_model_architecture`/`suggest_ensemble` flags — extend this to carry `current_phase` and `gate_criteria`.

#### 3C. Add validation node after execution

The current loop is: `Proposal → Code → Execute → Feedback`. For MBS, insert a **domain validation node** between Execute and Feedback:

```
Execute → DomainValidator → Feedback
              |
              ↓ (fail)
         Auto-reject with specific reason
```

The `DomainValidator` is **not an LLM call** — it's deterministic Python that checks:
1. All SMM_DECIMAL predictions are in `[0.0, 1.0]`
2. Rate sensitivity is monotonic (compute correlation of `predicted_SMM_DECIMAL` with `rate_incentive` — must be > 0.3)
3. No NaN/Inf in predictions
4. Holdout period predictions exist for all CUSIPs (no silent data drops)
5. Training time < timeout threshold

If any check fails, the experiment is auto-rejected with a specific diagnostic message that goes directly back to the coder (bypassing the LLM feedback evaluation — saving an LLM call and giving more precise error information).

#### 3D. Parallel component evolution with dependency graph

The current orchestration evolves one component per iteration (chosen by `component_gen` or `hypothesis_select`). For MBS, some components are independent and can evolve in parallel:

```
RateCurveFeatures ──┐
PoolDynamics ───────┼──→ PrepaymentModel ──→ ScenarioValidator
MacroFeatures ──────┘
```

When the orchestrator decides to improve feature engineering, it can spawn **parallel traces** (the MCTS infrastructure already supports this) — one improving rate features, one improving CUSIP dynamics — and merge the best of each into the next model iteration.

Implementation: Override the `component_gen` prompt to return **a list of components** instead of a single one. Modify the loop to fan-out parallel experiments when components are on independent branches of the dependency graph.

#### 3E. Human-in-the-loop at phase gates

Integrate the `interrupt()` pattern (from LangGraph) at phase gate transitions:

```python
# At end of each phase, before advancing:
review_payload = {
    "current_phase": "Rate Response",
    "next_phase": "Dynamics",
    "gate_results": gate_check_results,
    "s_curve_plot": plot_path,           # Visual for human
    "model_properties": structured_state,
    "recommendation": "Phase 2 gate passed. S-curve inflection at 87bps incentive, "
                      "monotonicity confirmed. Recommend advancing to Phase 3."
}
interrupt(review_payload)
```

This gives the quant modeler control at the strategic level (phase transitions) without requiring review of every iteration. The modeler can:
- **Approve**: Advance to next phase
- **Reject with guidance**: "S-curve is too steep above 150bps incentive — add a plateau/cap before advancing"
- **Override phase**: "Skip macro features, go directly to ensemble — we have a deadline"

---

### Priority Summary for Directions 1–3

| Priority | Direction | Customization | Impact |
|---|---|---|---|
| 1 | Prompts | MBS-specific `scen.role` + `hypothesis_specification` | Prevents generic ML approaches from day 1 |
| 2 | Orchestration | Domain validation node (deterministic, no LLM) | Catches 80% of failures without wasting LLM calls |
| 3 | Prompts | MBS-specific component specs with canonical features | Ensures correct feature engineering |
| 4 | Memory | Feature attribution feedback loop (IG → hypothesis) | Grounds hypotheses in empirical evidence |
| 5 | Orchestration | Multi-phase with hard gates | Enforces structured model development progression |
| 6 | Memory | Structured model properties instead of raw traces | Better LLM reasoning about model behavior |
| 7 | Orchestration | MBS-specific component decomposition | Finer-grained evolution control |
| 8 | Memory | Domain knowledge RAG pre-loading | Improves code quality from iteration 1 |

---

## Part 2: Additional Customization Directions

### Direction 4: Evaluation Function Design

This is **not** the feedback prompt — it's the actual computation that produces the numbers the feedback prompt sees. The current loop runs `main.py`, captures a results table, and the LLM compares "new score vs old score." For MBS, the evaluation function itself is a major design surface.

**What's missing from the current design:**

The RD-Agent loop has a single scalar metric (or a small results table). MBS prepayment demands **multi-dimensional evaluation** where a model can be "better" on one axis and "worse" on another:

```
Evaluation Dimensions:
├── Accuracy
│   ├── Overall weighted RMSE of SMM_DECIMAL (across CUSIPs × fh_effdt)
│   ├── Per-coupon-bucket RMSE: RMSE reported separately for each coupon bucket
│   │       (e.g., <3.0, 3.0-3.5, 3.5-4.0, 4.0-4.5, 4.5-5.0, 5.0+) — critical
│   │       because prepayment behavior is coupon-dependent and high-coupon
│   │       CUSIPs drive most of the refi risk
│   ├── Tail accuracy: RMSE on CUSIPs in top/bottom 10% SMM_DECIMAL
│   └── Per-vintage RMSE (2015, 2016, ..., 2023 originations)
├── Rate Sensitivity Fidelity
│   ├── Monotonicity score: Spearman(predicted_SMM_DECIMAL, rate_incentive)
│   ├── S-curve R²: fit logistic to (incentive, predicted_SMM_DECIMAL), report R²
│   └── Inflection point: estimated bps where refi response steepens
├── Temporal Robustness
│   ├── Regime transition RMSE (first 3 months after major rate move)
│   ├── Rolling 12-month RMSE stability (max/min ratio)
│   └── Extrapolation behavior: predicted SMM_DECIMAL at ±300bps incentive (sanity)
├── Structural Properties
│   ├── Burnout effect present: t-test on (old in-the-money) vs (fresh) CUSIPs
│   ├── Seasonality present: F-test on fh_effdt month-of-year dummies in residuals
│   └── CUSIP differentiation: std(predicted_SMM_DECIMAL) across CUSIPs (>0 means model uses CUSIP features)
```

**The customization:** Override the `main.py` template (the `Workflow` component) to compute all these metrics and output them as a structured JSON, not a single number. Then the feedback LLM evaluates a **multi-dimensional scorecard** and must reason about tradeoffs ("overall RMSE improved but coupon 4.5+ bucket RMSE degraded — reject").

This also enables **Pareto-based selection**: the system tracks a Pareto frontier across iterations rather than a single "best" experiment. An experiment that sacrifices 0.0005 overall RMSE but fixes the S-curve shape or improves per-coupon RMSE uniformity is worth keeping.

---

### Direction 5: Code Scaffold and Interface Contracts

The current loop gives the LLM a `component_spec` (from `share.yaml`) that describes the function signature. But for MBS, you need much more rigid scaffolding:

**What's missing:** The LLM can generate code that's syntactically correct and runs, but structurally wrong for prepayment modeling. Examples:
- Treats each `(cusip, fh_effdt)` row as an independent observation (ignores panel structure)
- Uses `sklearn.model_selection.train_test_split` with `shuffle=True` (time series leakage)
- Normalizes SMM_DECIMAL target variable (destroys interpretability)
- Computes features using same-`fh_effdt` macro data (look-ahead bias)

**The customization:** Provide a **hard scaffold** — non-LLM-generated code that the LLM's code plugs into:

```python
# scaffold.py — NOT generated by LLM, provided as fixed infrastructure
class MBSDataContract:
    """The LLM-generated feature code must produce a DataFrame with this schema."""
    required_index = ["cusip", "fh_effdt"]  # MultiIndex
    required_columns = ["rate_incentive", "coupon"]   # At minimum
    forbidden_columns = ["future_smm", "forward_rate"]  # Leakage prevention
    target_column = "smm_decimal"
    target_range = (0.0, 1.0)

class MBSTrainTestSplit:
    """Fixed temporal split on fh_effdt — LLM cannot override this."""
    def split(self, df):
        cutoff = self.train_end_date  # e.g., "2021-12-31"
        return df[df.fh_effdt <= cutoff], df[df.fh_effdt > cutoff]

class MBSEvaluationHarness:
    """Fixed evaluation — LLM generates the model, this code evaluates it."""
    def evaluate(self, model, test_data) -> dict:
        # Computes ALL metrics from Direction 4, including per-coupon RMSE
        # LLM cannot game metrics by changing how they're computed
```

The LLM only generates: (1) feature engineering functions that produce conformant DataFrames, (2) model classes that accept features and produce SMM_DECIMAL predictions. Everything else — data loading, splitting, evaluation, metric computation — is **fixed scaffold** that the LLM cannot modify.

This is implemented by customizing the `Workflow` component spec and the `inject_code_from_file_dict` method to always overwrite scaffold files, regardless of what the LLM proposes.

---

### Direction 6: Search Strategy / Exploration-Exploitation

The current system's exploration strategy is governed by:
- `hypothesis_select` choosing between Draft/Improvement/Ensemble stages
- MCTS trace selection when running parallel traces
- `scen_prob_multiplier` balancing scenario-problems vs feedback-problems
- The "avoid same component 5 times in a row" heuristic in `component_gen`

These are all generic. For MBS, the search landscape has known structure:

**The customization:** Implement a **curriculum-aware search strategy** that exploits the known dependency structure of prepayment model components:

```
Knowledge: rate_incentive is necessary before burnout is meaningful
           (burnout = decay of RESPONSE to rate incentive)

Knowledge: individual model quality matters more than ensemble
           when you have <5 models

Knowledge: feature engineering has diminishing returns after
           ~15 well-chosen features; model architecture matters more
```

Concretely:
- **Dependency-constrained exploration**: Don't allow hypotheses about burnout until rate incentive feature exists and model shows monotonic rate sensitivity. Don't allow ensemble hypotheses until at least 3 distinct model architectures exist. Implement as a filter in `hypothesis_select` that checks preconditions.

- **Adaptive exploration radius**: Early iterations should explore broadly (different model families: GBM, neural net, GAM, GLM). Once a promising architecture is found, narrow to hyperparameter/feature refinement. The current `use_ratio`-based stage selection is a rough version of this — replace with a metric-driven switch: "if last 3 iterations improved RMSE by <5% each, switch from exploitation to exploration."

- **Backtracking policy**: If 3 consecutive iterations fail to improve, don't just keep pushing — **revert to the best checkpoint and try a different component branch**. The current system's `last_successful_exp` does this implicitly, but a deliberate backtracking strategy with a "cooldown" on the recently-failed component is stronger.

---

### Direction 7: Data Representation to the LLM

How the LLM "sees" the data is a distinct customization surface from prompts. The current system generates `EDA.md` and injects data folder descriptions. For MBS:

**What's missing:** The LLM never sees actual data distributions, just schema descriptions. For prepayment modeling, the LLM needs to understand:

- The distribution of rate incentive across the dataset (is it centered near zero? heavy-tailed?)
- How many CUSIPs are in-the-money vs out-of-the-money at different time points
- The autocorrelation structure of SMM_DECIMAL (high autocorrelation means the model needs to handle persistence)
- Whether there are structural breaks in the data (the 2020 COVID refi wave looks very different from 2015-2019)
- The distribution of CUSIPs across coupon buckets

**The customization:** Build a **domain-specific EDA** that runs automatically before iteration 1 and produces structured diagnostic outputs:

```markdown
## MBS Data Profile
- **Panel dimensions**: 2,847 CUSIPs × 156 fh_effdt months (Jan 2010 – Dec 2022)
- **SMM_DECIMAL distribution**: mean=0.0082, median=0.0051, p95=0.0284, p99=0.0521 (heavy right tail)
- **Coupon distribution**: <3.0: 12%, 3.0-3.5: 24%, 3.5-4.0: 31%, 4.0-4.5: 18%, 4.5-5.0: 11%, 5.0+: 4%
- **Rate incentive distribution**: mean=-12bps, std=89bps, range=[-245bps, +310bps]
- **In-the-money fraction by year**: 2010:34%, 2012:62%, 2015:28%, 2019:41%, 2020:78%, 2022:8%
- **SMM_DECIMAL autocorrelation (lag-1)**: 0.91 (very high persistence)
- **Structural breaks detected**: Mar 2020 (COVID refi), Mar 2022 (rate hiking)
- **Missing data**: 3.2% of (cusip, fh_effdt) pairs missing; concentrated in 2010-2011 vintages
- **Feature correlations with SMM_DECIMAL**: rate_incentive(0.61), WALA(0.23), loan_count(-0.08)
```

This gives the hypothesis generator **empirical grounding** it cannot get from a generic schema description. It would naturally propose "SMM_DECIMAL has a heavy right tail — consider a model with heteroscedastic error or a quantile regression approach" or "autocorrelation is 0.91 — the model should include lagged SMM_DECIMAL or use a recurrent architecture."

---

### Direction 8: Interpretability as a First-Class Loop Signal

This connects directly to the IG/Captum plan but goes further — make attribution **a driver of the R&D loop**, not just a post-hoc analysis.

**The customization:** After every successful iteration, automatically run feature attribution (IG or SHAP) and feed the results into the **next iteration's hypothesis generation** as structured input. But beyond that:

- **Attribution-driven hypothesis generation**: "Feature X has high attribution but the partial dependence plot shows a linear relationship — hypothesis: adding a nonlinear transform of feature X will capture the true relationship better"
- **Attribution sanity checking**: If `unemployment_rate` gets a **positive** attribution on SMM_DECIMAL (higher unemployment → higher prepayment), that's economically wrong — auto-flag it as a modeling defect that needs fixing
- **Attribution stability tracking**: If feature rankings change wildly between iterations, the model is unstable and the loop should focus on regularization/stability before adding complexity
- **Counterfactual testing**: Use IG baselines to answer "what would SMM_DECIMAL be if rate incentive were zero?" — if the answer is unreasonable, the model has a structural problem

This makes the loop **self-auditing**: each iteration produces not just a metric but an explanation that is checked against domain knowledge before the next iteration begins.

---

### Direction 9: Execution Environment and Compute Strategy

The current system runs code in Docker with a timeout. For MBS:

- **Data mounting strategy**: CUSIP-level panel data can be large (millions of rows). The execution environment needs pre-mounted, pre-processed data in efficient formats (Parquet with proper partitioning by vintage or `fh_effdt`). The LLM-generated code should read from a standard path, not process raw CSVs each time.

- **Compute budget allocation**: Different phases need different compute. Feature engineering: CPU-only, fast. Model training: potentially GPU (for neural nets). Evaluation: CPU, moderate. Give more timeout to model training iterations, less to feature engineering.

- **Incremental execution**: If only the feature engineering changed, don't retrain the model from scratch — use cached model weights and just re-evaluate. If only the model changed, don't re-run feature engineering. The current system re-runs everything each iteration. Caching intermediate artifacts (feature matrices, trained models) across iterations dramatically speeds up the loop.

- **Reproducibility infrastructure**: Set random seeds, log package versions, save model checkpoints. For MBS (which has regulatory scrutiny), every experiment must be reproducible — this needs to be baked into the execution environment, not left to LLM-generated code.

---

### Direction 10: Multi-Agent Specialization

The current loop uses a single LLM persona throughout. For MBS, different phases benefit from different "expertise":

- **Hypothesis phase**: Needs a "quant researcher" persona that thinks in terms of prepayment theory, economic intuition, and structural models
- **Coding phase**: Needs a "ML engineer" persona that thinks in terms of pandas, PyTorch, efficient data handling, and numerical stability
- **Evaluation phase**: Needs a "model validator" persona that thinks in terms of backtesting methodology, out-of-sample robustness, and regulatory requirements
- **Debug phase**: Needs a "data engineer" persona that thinks in terms of data quality, missing values, type mismatches, and performance optimization

**The customization:** Use different system prompts (or even different models — cheaper/faster model for debugging, more capable model for hypothesis generation) at different nodes. The current architecture already supports this since each LLM call goes through `APIBackend()` independently — you can override the model per call.

---

## Final Priority Map: All 10 Directions

| Priority | Direction | Key Customization | Why it matters for MBS |
|---|---|---|---|
| 1 | **Evaluation function** (#4) | Multi-dimensional scorecard with structural checks + per-coupon RMSE | Prevents "good overall RMSE but broken economics" |
| 2 | **Code scaffold** (#5) | Fixed temporal split on fh_effdt, leakage prevention, panel structure | Prevents the most common MBS modeling errors |
| 3 | Prompts (#1) | MBS-specific role, hypothesis constraints, feature specs | Steers all LLM reasoning toward prepayment domain |
| 4 | **Interpretability loop** (#8) | IG attribution → hypothesis feedback cycle | Makes the loop self-auditing against domain knowledge |
| 5 | **Data representation** (#7) | Domain-specific EDA with distributional analysis | Grounds hypotheses in actual data characteristics |
| 6 | **Search strategy** (#6) | Curriculum-aware, dependency-constrained exploration | Prevents wasting iterations on premature complexity |
| 7 | Memory (#2) | Structured model properties + attribution memory | Better LLM reasoning across iterations |
| 8 | Orchestration (#3) | Phase gates with human review | Strategic control without per-iteration overhead |
| 9 | **Execution environment** (#9) | Incremental caching, compute allocation | 3-5x speedup per iteration |
| 10 | **Multi-agent** (#10) | Specialized personas per phase | Marginal quality improvement, easy to implement |

The top four (evaluation function, code scaffold, prompts, interpretability loop) are where 80% of the value is. They prevent the system from producing "technically running but economically meaningless" models — which is the primary risk when applying generic ML automation to structured finance.

---

## Implementation Status

All 10 priorities have been implemented, smoke-tested, committed, and pushed to `myfork/main`. Each module lives under `rdagent/scenarios/mbs_prepayment/` and has a corresponding `@pytest.mark.offline` test file under `rdagent/scenarios/mbs_prepayment/tests/`.

| Priority | Commit | Module |
|---|---|---|
| 1 | Evaluation | [evaluation.py](rdagent/scenarios/mbs_prepayment/evaluation.py) — multi-dim scorecard w/ per-coupon RMSE |
| 2 | Scaffold | [scaffold.py](rdagent/scenarios/mbs_prepayment/scaffold.py) — data contract + temporal split |
| 3 | Prompts | [prompts.yaml](rdagent/scenarios/mbs_prepayment/prompts.yaml) + [prompt_loader.py](rdagent/scenarios/mbs_prepayment/prompt_loader.py) |
| 4 | Interpretability | [interpretability.py](rdagent/scenarios/mbs_prepayment/interpretability.py) — IG + economic priors |
| 5 | EDA | [eda.py](rdagent/scenarios/mbs_prepayment/eda.py) — panel profile + structural breaks |
| 6 | Search strategy | [search_strategy.py](rdagent/scenarios/mbs_prepayment/search_strategy.py) — curriculum gating |
| 7 | Memory | [memory.py](rdagent/scenarios/mbs_prepayment/memory.py) — phase-aware context |
| **8** | **Orchestration** | [orchestration.py](rdagent/scenarios/mbs_prepayment/orchestration.py) — phase gates, DomainValidator, HumanReviewPayload — commit `90c987bc` |
| **9** | **Execution env** | [execution_env.py](rdagent/scenarios/mbs_prepayment/execution_env.py) — ArtifactCache, IncrementalRunner, ComputeBudget — commit `323eed84` |
| **10** | **Personas** | [personas.py](rdagent/scenarios/mbs_prepayment/personas.py) — 4 personas + PersonaRouter — commit `0f7c051b` |

Each module has a corresponding `@pytest.mark.offline` test file in [tests/](rdagent/scenarios/mbs_prepayment/tests/) and was smoke-tested before commit. The MBS customization stack is now fully in place end-to-end.
