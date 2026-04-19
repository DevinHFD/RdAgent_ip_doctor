# MBS Prepayment Scenario

A customization of the RD-Agent **data science loop** for CUSIP-level MBS
prepayment (SMM_DECIMAL) forecasting. This scenario plugs ten Priority
modules — evaluation harness, data contract, temporal splitter, search
strategy, memory, domain validator, phase gates, execution environment,
personas, interpretability — into the existing
`rdagent.scenarios.data_science.loop.DataScienceRDLoop` without
forking the core loop code.

> **Not to be confused with `ip_doctor`.** `ip_doctor` is a standalone
> LangGraph workflow with its own `MBS_`-prefixed settings. This
> scenario uses the `MBSP_` prefix and is driven by `DS_SCEN` / the DS
> loop CLI.

---

## Architecture at a glance

```
.env                                     rdagent data_science --competition mbs_prepayment
  │                                                    │
  ├── DS_SCEN=...MBSPrepaymentScen                     ▼
  ├── DS_COMPETITION=mbs_prepayment          DataScienceRDLoop (unchanged core)
  ├── KG_LOCAL_DATA_PATH=./mbs_data                    │
  ├── DS_LOCAL_DATA_PATH=./mbs_data                    ▼
  └── MBSP_* (scenario knobs)               MBSPrepaymentScen(DataScienceScen)
                                                       │
                             ┌─────────────────────────┼─────────────────────────┐
                             ▼                         ▼                         ▼
                      mbs_contract              mbs_harness              mbs_orchestrator
                      mbs_splitter              mbs_memory               (memory + search
                      mbs_validator             mbs_search_state          + validator + gate)
                      mbs_gate                  mbs_persona_router
```

All ten building blocks live on the scenario object and are reachable
from downstream coder / feedback templates via `scen.mbs_*`.

---

## Setup guide

### 1. Install dependencies

```bash
cd /home/devin/RD-Agent
make dev                # editable install + all extras + pre-commit
```

Verify:

```bash
rdagent health_check
```

### 2. Create the data / output folder layout

```bash
mkdir -p ./mbs_data/mbs_prepayment
mkdir -p ./mbs_models
mkdir -p ./mbs_output/cache
```

Folder contract:

| Path | Purpose | Env var |
|---|---|---|
| `./mbs_data/` | Root for all MBS competitions | `MBSP_DATA_DIR`, `KG_LOCAL_DATA_PATH`, `DS_LOCAL_DATA_PATH` |
| `./mbs_data/mbs_prepayment/` | This competition's data folder | (derived from `DS_COMPETITION`) |
| `./mbs_data/mbs_prepayment/description.md` | Task description the DS loop reads | — |
| `./mbs_data/mbs_prepayment/tfminput.pkl` | **Single pickled DataFrame** — all cusips, all months, all GNMA features pre-normalized to mean 0 / std 1, plus the unnormalized `smm_decimal` target | `MBSP_PANEL_FILENAME` |
| `./mbs_data/mbs_prepayment/scaler.sav` | joblib-saved sklearn-style scaler used by the scaffold to inverse-transform GNMA features back to raw units (WAC %, refi incentive, etc.) for the scorecard | `MBSP_SCALER_FILENAME` |
| `./mbs_models/` | Model checkpoints, scalers | `MBSP_MODEL_CHECKPOINT_DIR` |
| `./mbs_output/` | Scorecards, plots, reports | `MBSP_OUTPUT_DIR` |
| `./mbs_output/memory.json` | Phase-aware memory store | `MBSP_MEMORY_PATH` |
| `./mbs_output/search_state.json` | Curriculum / cooldown state | `MBSP_SEARCH_STATE_PATH` |
| `./mbs_output/cache/` | Content-addressed artifact cache | `MBSP_CACHE_DIR` |

### 3. Place the task description

Copy the shipped example to the competition folder:

```bash
cp rdagent/scenarios/mbs_prepayment/example/description.md \
   ./mbs_data/mbs_prepayment/description.md
```

Edit as needed — the DS loop reads this verbatim as the task spec.

### 4. Drop in the panel data and scaler

This scenario expects two files in `./mbs_data/mbs_prepayment/`:

1. **`tfminput.pkl`** — a pickled `pandas.DataFrame` holding the full
   CUSIP-level monthly panel. Every GNMA feature (see
   [example/gnma_feature.md](example/gnma_feature.md)) is pre-built and
   **pre-normalized to mean 0 / std 1**. The target `smm_decimal` is
   carried **unnormalized** (decimal form in `[0, 1]`). Panel key is
   `(cusip, fh_effdt)`.
2. **`scaler.sav`** — a joblib-saved sklearn-style scaler
   (`StandardScaler`-compatible: exposes `mean_`, `scale_`,
   `feature_names_in_`). The scaffold uses it to inverse-transform the
   GNMA feature columns back to raw units (WAC %, refi incentive, burnout
   log-sum, WALA months, …) before scoring, so every harness diagnostic
   is on the natural scale.

Contract (see [scaffold.py](scaffold.py)):

- Required GNMA features present in the panel (names must match
  `gnma_feature.md`): `WAC`, `WALA`,
  `Avg_Prop_Refi_Incentive_WAC_30yr_2mos`,
  `Avg_Prop_Switch_To_15yr_Incentive_2mos`,
  `Burnout_Prop_WAC_30yr_log_sum60`,
  `Burnout_Prop_30yr_Switch_to_15_Lag1`, `CLTV`, `SATO`, `Pool_HPA_2yr`
- No forbidden leakage columns: `future_smm`, `forward_smm`,
  `next_month_smm`, `forward_rate`, `future_rate_incentive`
- `smm_decimal ∈ [0, 1]`

The train/test split is performed in-memory on `fh_effdt`
(`<= 2024-10-31` for train, `> 2024-10-31` for test). The coder loads
the panel as:

```python
import pandas as pd
df = pd.read_pickle("tfminput.pkl")
```

Features are already built and normalized — the coder must not
re-normalize or re-engineer them. The coder only provides a
`build_model()` callable returning an unfitted sklearn-compatible
estimator; the scaffold handles load → split → fit → predict → clip →
scaler inverse-transform → scorecard → `submission.csv`.

### 5. Configure `.env`

Copy the template to the repo root:

```bash
cp .env.mbs_prepayment.example .env
```

Fill in your LLM credentials (`OPENAI_API_KEY`, `OPENAI_API_BASE`,
`LITELLM_PROXY_*` if using a separate embedding provider). The three
lines that wire the DS loop at this scenario are:

```ini
DS_SCEN=rdagent.scenarios.mbs_prepayment.scenario.MBSPrepaymentScen
DS_COMPETITION=mbs_prepayment
KG_LOCAL_DATA_PATH=./mbs_data
DS_LOCAL_DATA_PATH=./mbs_data
```

Everything else (`MBSP_*`) tunes the scenario-specific modules. See
the template for the full list with defaults.

### 6. Run the scenario

```bash
rdagent mbs_prepayment --competition mbs_prepayment
```

This invokes `MBSPrepaymentRDLoop` (a subclass of `DataScienceRDLoop`)
with `MBSPrepaymentScen` as the scenario. The loop will:

1. Read `description.md` from the competition folder
2. Instantiate `MBSPrepaymentScen`, attaching all ten Priority modules
3. Run the proposal → codegen → execution → feedback → record loop, with
   MBS memory/search/phase context injected into every LLM call,
   `DomainValidator` auto-rejecting degenerate experiments, the
   `MODEL_VALIDATOR` persona driving feedback, and phase gates
   auto-advancing when criteria are met

> The old `rdagent data_science --competition mbs_prepayment` invocation
> still works but runs the generic DS loop — only scenario metadata is
> MBS-aware, none of the Priority 6–10 modules are invoked at runtime.
> Use `rdagent mbs_prepayment` for the fully wired loop.

### 7. View logs

```bash
rdagent ui --port 19899 --log-dir ./log
```

---

## How `MBSPrepaymentSettings` / `env_prefix="MBSP_"` works

The settings class in [conf.py](conf.py):

```python
class MBSPrepaymentSettings(ExtendedBaseSettings):
    model_config = SettingsConfigDict(env_prefix="MBSP_", protected_namespaces=())

    train_end_date: str = "2024-10-31"
    gate_baseline_max_rmse: float = 0.040
    # ...
```

means:

1. **You do not wire fields manually.** Pydantic-settings auto-loads
   any env var whose name is `MBSP_<FIELD_NAME_UPPER>`. Example:
   `MBSP_TRAIN_END_DATE=2022-06-30` in `.env` overrides the default.
2. **Precedence**: shell env > `.env` file > class default.
3. **Namespace isolation**: because the prefix is `MBSP_`, the
   scenario cannot collide with `ip_doctor`'s `MBS_`-prefixed settings
   even if both scenarios share one `.env`.
4. **How it reaches the scenario**: [conf.py](conf.py) instantiates a
   module-level `MBSP_SETTINGS = MBSPrepaymentSettings()`, and
   [scenario.py](scenario.py) reads from it when constructing every
   Priority module:

   ```python
   from .conf import MBSP_SETTINGS

   self.mbs_contract = MBSDataContract(
       target_column=MBSP_SETTINGS.target_column,
       target_range=(MBSP_SETTINGS.target_min, MBSP_SETTINGS.target_max),
       ...
   )
   ```

So tuning the scenario is purely a matter of editing `.env` — no code
changes required.

---

## Namespace cheat-sheet

| Prefix | Owner | Purpose |
|---|---|---|
| `MBSP_` | this scenario (`mbs_prepayment`) | MBS prepayment knobs (Priority 1–10) |
| `MBS_` | `ip_doctor` (independent LangGraph app) | **Do not reuse here** |
| `DS_` | `rdagent/app/data_science/conf.py` | Data science loop wiring (`DS_SCEN`, `DS_COMPETITION`, `DS_CODER_MAX_LOOP`, …) |
| `KG_` | `rdagent/app/kaggle/conf.py` (parent of DS) | Kaggle base settings (`KG_LOCAL_DATA_PATH`) |

---

## Runtime wiring (Priority 1–10 integration fix)

Earlier versions of this scenario attached all ten Priority modules to
`MBSPrepaymentScen` but relied on the stock `DataScienceRDLoop` to drive
the loop. That left the modules **defined but never invoked** at
runtime: the DS runner expected scalar `scores.csv`, not the MBS
multi-dimensional scorecard; MBS personas, memory, search state, phase
gates, and domain validator were never called during the proposal →
codegen → execution → feedback → record chain.

The following five changes wire the modules directly into the live loop:

| # | File | Role |
|---|---|---|
| 1 | [scaffold.py](scaffold.py) | `run_scaffold_pipeline()` now writes **both** `scores.json` (MBS scorecard) and `scores.csv` (primary-metric bridge for the DS runner). |
| 2 | [scenario.py](scenario.py) | `get_scenario_all_desc()` override injects current phase + gate criteria, search-strategy constraints, MBS memory, and the data contract into **every** downstream LLM prompt. |
| 3 | [feedback.py](feedback.py) | New `MBSExperiment2Feedback(DSExperiment2Feedback)` — prepends the `MODEL_VALIDATOR` persona to the feedback system prompt and appends the `scores.json` scorecard + feedback-phase memory to the user prompt. |
| 4 | [loop.py](loop.py) | New `MBSPrepaymentRDLoop(DataScienceRDLoop)` — overrides `feedback()` to run `DomainValidator` as an auto-reject gate (no LLM call on obvious rejects) and overrides `record()` to update `MBSMemory`, `MBSSearchState`, and evaluate/advance `PhaseGate`. |
| 5 | [../../app/mbs_prepayment/loop.py](../../app/mbs_prepayment/loop.py) + [../../app/cli.py](../../app/cli.py) | New `rdagent mbs_prepayment` CLI command that instantiates `MBSPrepaymentRDLoop` instead of the generic `DataScienceRDLoop`. |

### Modules: before vs. after

| Module | Before | After |
|---|---|---|
| `scenario.py` (static descriptions) | Active | Active |
| `evaluation.py` (`MBSEvaluationHarness`) | Defined, not consumed | **Active** — scorecard read by feedback + record |
| `memory.py` (`MBSMemory`) | Defined, not consumed | **Active** — updated on record, injected into all prompts |
| `search_strategy.py` (curriculum/cooldown) | Defined, not consumed | **Active** — constraints injected into all prompts |
| `orchestration.py` (`DomainValidator`, `PhaseGate`, `MBSOrchestrator`) | Defined, not consumed | **Active** — validator auto-rejects; gates auto-advance |
| `personas.py` (`MODEL_VALIDATOR`) | Defined, not consumed | **Active** — system prompt for feedback LLM |
| `scaffold.py` (`MBSDataContract`, `MBSTrainTestSplit`) | Defined, partially used | Data contract injected into prompts; `scores.csv` bridge added |
| `prompt_loader.py` / `prompts.yaml` | Defined, not consumed | Still not directly in the proposal path |
| `execution_env.py` (`IncrementalRunner`, `ArtifactCache`) | Defined, not consumed | Still not in the runner path |

### End-to-End Function Call Chain

The complete chain for one iteration of `rdagent mbs_prepayment`:

#### 0. Entry & Bootstrap

```
app/mbs_prepayment/loop.py::main()
├── RD_AGENT_SETTINGS.app_tpl = "app/mbs_prepayment/tpl"     ← activates MBS prompt overrides
├── DS_RD_SETTING.scen = "...mbs_prepayment.scenario.MBSPrepaymentScen"
│
├── MBSPrepaymentRDLoop(DS_RD_SETTING)                        [scenarios/mbs_prepayment/loop.py]
│   └── super().__init__() → DataScienceRDLoop.__init__()     [scenarios/data_science/loop.py:96]
│       ├── import_class(scen)() → MBSPrepaymentScen(competition)  [scenarios/mbs_prepayment/scenario.py]
│       │   ├── super().__init__() → DataScienceScen()
│       │   │   ├── reads description.md from data folder
│       │   │   ├── competition_description_template → LLM call (APIBackend)
│       │   │   └── sets metric_direction, metric_name, etc.
│       │   ├── MBSDataContract(), MBSTrainTestSplit()        [scaffold.py]
│       │   ├── MBSOrchestrator()                              [orchestration.py]
│       │   ├── DomainValidator()                              [orchestration.py]
│       │   ├── MBSStructuredMemory()                          [memory.py]
│       │   └── MBSSearchState()                               [search_strategy.py]
│       │
│       ├── DSProposalV2ExpGen(scen)                  ← exp_gen (proposer)
│       ├── DataLoaderCoSTEER(scen)                   ← component coders
│       ├── FeatureCoSTEER(scen)
│       ├── ModelCoSTEER(scen)
│       ├── EnsembleCoSTEER(scen)
│       ├── WorkflowCoSTEER(scen)
│       ├── PipelineCoSTEER(scen)
│       ├── DSCoSTEERRunner(scen)                     ← runner
│       ├── DSTrace(scen)                             ← trace history
│       └── MBSExperiment2Feedback(scen)              ← summarizer (overridden by MBS __init__)
│
└── asyncio.run(mbs_loop.run())
```

#### 1. `direct_exp_gen` — Proposal (7+ LLM calls)

```
MBSPrepaymentRDLoop.direct_exp_gen()                          [inherited from DataScienceRDLoop]
├── ckp_selector.get_selection(trace)                          ← MCTS checkpoint selection
├── exp_gen.async_gen(trace) → DSProposalV2ExpGen.gen()        [proposal/exp_gen/proposal.py:1300]
│   │
│   ├── scen.get_scenario_all_desc()                           [mbs_prepayment/scenario.py:197]
│   │   ├── super().get_scenario_all_desc()                    ← DS base description
│   │   ├── mbs_orchestrator.phase_spec()                      ← current phase, gate criteria
│   │   ├── mbs_orchestrator.iteration_constraints()           ← curriculum filter
│   │   ├── mbs_memory.render_context(HYPOTHESIS_GEN)          ← structured memory
│   │   └── MBS Data Contract (features, split, scoring)
│   │
│   ├── [Step 1] identify_problem()                            [proposal.py:569]
│   │   ├── identify_scenario_problem()                        ← LLM call #1
│   │   │   ├── T(".prompts_v2:scenario_problem.system")
│   │   │   │   └── {% include "scenarios.data_science.share:scen.role" %}
│   │   │   │       └── app_tpl override → MBS scen.role (prepayment modeler persona)
│   │   │   └── APIBackend().build_messages_and_create_chat_completion()
│   │   │
│   │   └── identify_feedback_problem()                        ← LLM call #2
│   │       ├── T(".prompts_v2:feedback_problem.system")
│   │       │   └── {% include scen.role %} → MBS persona
│   │       └── APIBackend()...
│   │
│   ├── [Step 2] hypothesis_gen()                              ← LLM call #3
│   │   ├── T(".prompts_v2:hypothesis_gen.system")
│   │   │   └── {% include scen.role %} → MBS persona
│   │   └── APIBackend()...
│   │
│   ├── [Step 2.1] hypothesis_critique()                       ← LLM call #4
│   │   ├── T(".prompts_v2:hypothesis_critique.system")
│   │   │   └── {% include scen.role %} → MBS persona
│   │   └── APIBackend()...
│   │
│   ├── [Step 2.2] hypothesis_rewrite()                        ← LLM call #5
│   │   ├── T(".prompts_v2:hypothesis_rewrite.system")
│   │   │   └── {% include scen.role %} → MBS persona
│   │   └── APIBackend()...
│   │
│   ├── [Step 3] hypothesis_select_with_llm()                  ← LLM call #6
│   │   ├── T(".prompts_v2:hypothesis_select.system")
│   │   └── APIBackend()...
│   │
│   └── [Step 4] task_gen()                                    ← LLM call #7
│       ├── get_component(hypothesis.component)
│       │   └── T("scenarios.data_science.share:component_spec.{component}")
│       │       └── app_tpl override → MBS component specs
│       │           DataLoadSpec → MBS DataLoader spec
│       │           FeatureEng  → MBS FeatureEng spec (forbids cusip_target_enc)
│       │           Model       → MBS PrepaymentModel spec
│       │           Workflow    → MBS ScenarioValidator spec
│       │           Ensemble    → falls through to DS default
│       ├── T(".prompts_v2:task_gen.system")
│       ├── T(".prompts:hypothesis_specification")
│       │   └── app_tpl override → MBS hypothesis_specification
│       │       (prepayment components, rate-curve, per-coupon-bucket RMSE)
│       └── APIBackend()...
│           → returns DSExperiment(pending_tasks_list, hypothesis)
│
└── interactor.interact(exp, trace)                            ← optional human interaction
```

#### 2. `coding` — Code Generation (2+ LLM calls per component)

```
MBSPrepaymentRDLoop.coding()                                   [inherited from DataScienceRDLoop]
├── for each task in exp.pending_tasks_list:
│   ├── isinstance check → dispatch to correct coder:
│   │
│   │   [If FeatureTask] → FeatureCoSTEER.develop(exp)
│   │   [If ModelTask]   → ModelCoSTEER.develop(exp)
│   │   [If PipelineTask]→ PipelineCoSTEER.develop(exp)        ← most common with coder_on_whole_pipeline
│   │   ...etc
│   │
│   └── CoSTEER.develop(exp)                                   [components/coder/CoSTEER]
│       ├── MultiProcessEvolvingStrategy.implement_one_task()
│       │   ├── system_prompt includes:
│       │   │   └── T("scenarios.data_science.share:component_spec.{component}")
│       │   │       → MBS override via app_tpl (Model → PrepaymentModel spec, etc.)
│       │   ├── APIBackend()... → generates code                ← LLM call (code gen)
│       │   └── writes .py files to experiment workspace
│       │
│       └── CoSTEEREvaluator.evaluate()
│           ├── runs generated code in subprocess/Docker
│           ├── parses output, checks format
│           └── APIBackend()... → eval feedback                 ← LLM call (code eval)
│               └── T("scenarios.data_science.dev.runner.prompts:DSCoSTEER_eval")
│                   └── {% include scen.role %} → MBS persona
│
│   (loop: evolve code up to max_evolve iterations)
```

#### 3. `running` — Execution

```
MBSPrepaymentRDLoop.running()                                  [inherited from DataScienceRDLoop]
├── DSCoSTEERRunner.develop(exp)                               [scenarios/data_science/dev/runner]
│   ├── MultiProcessEvolvingStrategy → runs code in Docker
│   │   ├── subprocess: python main.py                         ← user's code
│   │   │   └── (in MBS) calls scaffold:
│   │   │       run_scaffold_pipeline(panel_path, scaler_path, model_builder, output_dir)
│   │   │       ├── pd.read_pickle("tfminput.pkl")             ← load panel
│   │   │       ├── MBSDataContract.validate(panel)            ← check required/forbidden cols
│   │   │       ├── MBSWorkflow.run(panel, model_builder)
│   │   │       │   ├── MBSTrainTestSplit.split(df)            ← temporal on fh_effdt ≤ 2024-10-31
│   │   │       │   ├── model_builder().fit(X_train, y_train)  ← LLM-generated model
│   │   │       │   ├── model.predict(X_test)
│   │   │       │   └── clip_predictions(y_pred, contract)     ← clamp to [0, 1]
│   │   │       ├── inverse_transform_features(test_df, scaler, GNMA_HARNESS_FEATURES)
│   │   │       │   └── raw_value = normalized × scale + mean   ← Pool_HPA_2yr, WAC, WALA, etc.
│   │   │       ├── MBSEvaluationHarness.evaluate()            [evaluation.py]
│   │   │       │   ├── overall RMSE
│   │   │       │   ├── per-coupon-bucket RMSE (WAC buckets)
│   │   │       │   ├── monotonicity_spearman (refi incentive)
│   │   │       │   ├── regime-transition RMSE
│   │   │       │   └── structural properties (burnout, seasonality, CUSIP differentiation)
│   │   │       ├── write_scorecard() → scores.json
│   │   │       └── write scores.csv (primary_metric: rmse_smm_decimal)
│   │   │
│   │   └── parses scores.csv → exp.result
│   │
│   └── DSRunnerEvaluator.evaluate()
│       └── T("...DSCoSTEER_eval.system")
│           └── {% include scen.role %} → MBS persona          ← LLM call (runner eval)
```

#### 4. `feedback` — MBS-Specific Feedback (1-2 LLM calls)

```
MBSPrepaymentRDLoop.feedback()                                 [mbs_prepayment/loop.py]
│
├── _read_scorecard(exp)                                       ← reads scores.json from workspace
│
├── _domain_validate(scorecard)                                ← deterministic auto-reject gate
│   └── DomainValidator.validate_from_scorecard(scorecard)     [orchestration.py]
│       ├── check monotonicity_spearman > threshold
│       ├── check overall_rmse < max
│       └── check per-coupon uniformity
│       → if FAIL: return ExperimentFeedback(decision=False)   ← NO LLM call, saves a round trip
│
├── [if validation passes] → super().feedback()
│   └── DataScienceRDLoop.feedback()
│       └── MBSExperiment2Feedback.generate_feedback()         [mbs_prepayment/feedback.py]
│           │
│           ├── system_prompt:
│           │   ├── MODEL_VALIDATOR.system_prompt              ← persona preamble [personas.py]
│           │   └── T("scenarios.data_science.dev.prompts:exp_feedback.system")
│           │       └── rendered with scen.get_scenario_all_desc()
│           │           → includes MBS phase, memory, data contract
│           │
│           ├── user_prompt:
│           │   ├── T("scenarios.data_science.dev.prompts:exp_feedback.user")
│           │   │   └── SOTA desc, current exp code/results, diff
│           │   ├── + mbs_scorecard_text                       ← scores.json (per-coupon, monotonicity)
│           │   └── + mbs_memory_text                          ← MBS memory FEEDBACK phase context
│           │
│           ├── APIBackend().build_messages_and_create_chat_completion()  ← LLM call (feedback)
│           │   → JSON with: Observations, Feedback for Hypothesis,
│           │     New Hypothesis, Reasoning, Code Change Summary,
│           │     Replace Best Result, EDA Improvement, Acceptable
│           │     (MBS-specific checks — burnout, per-coupon uniformity,
│           │     regime robustness — are surfaced deterministically via
│           │     the scorecard and reasoned about inline in Observations
│           │     / Feedback for Hypothesis)
│           │
│           └── returns HypothesisFeedback(decision, reason, ...)
```

#### 5. `record` — MBS State Update (no LLM)

```
MBSPrepaymentRDLoop.record()                                   [mbs_prepayment/loop.py]
├── super().record()                                           ← DS base: trace sync, SOTA update, archiving
│   └── DataScienceRDLoop.record()
│       ├── trace.sync_dag_parent_and_hist()
│       ├── sota_exp_selector.get_sota_exp_to_submit()
│       └── log_object(trace, sota_experiment)
│
├── _update_memory(loop_id, exp, feedback, scorecard, success) ← MBS structured memory
│   ├── ModelProperties.from_scorecard()                       ← extract model props from scorecard
│   ├── TraceEntry(iteration, component, hypothesis, ...)
│   └── mbs_memory.append_entry(entry)
│
├── _update_search_state(loop_id, exp, success, scorecard)     ← curriculum / cooldown
│   └── mbs_search_state.append(IterationRecord)
│
└── mbs_orchestrator.evaluate_gate()                           ← phase gate evaluation
    ├── checks gate criteria (rmse < threshold, monotonicity > threshold, etc.)
    └── if gate_result.passed:
        └── mbs_orchestrator.advance_phase()                   ← moves to next phase
            → e.g. BASELINE → RATE_SENSITIVITY → ROBUSTNESS → REFINEMENT
```

#### Prompt Override Flow (via `app_tpl`)

```
Any T("scenarios.data_science.share:scen.role").r()
│
├── load_content() tries (in order):
│   1. rdagent/app/mbs_prepayment/tpl/scenarios/data_science/share.yaml  ← FOUND
│   │   scen.role: {% include "scenarios.mbs_prepayment.prompts:scen.role" %}
│   │   └── loads from rdagent/scenarios/mbs_prepayment/prompts.yaml     ← source of truth
│   │       → "You are an expert quantitative prepayment modeler..."
│   2. (skipped) rdagent/scenarios/data_science/share.yaml
│
Same pattern for component_spec.{DataLoadSpec,FeatureEng,Model,Workflow}
and hypothesis_specification.
```

---

## Summary of Changes

### Problem

The generic `DataScienceRDLoop` drove everything. MBS modules
(`MBSEvaluationHarness`, `MBSMemory`, `MBSSearchState`,
`MBSOrchestrator`, `DomainValidator`, `PhaseGate`, `PersonaRouter`) were
instantiated on the scenario object but **never called** during the
live runtime chain:

- DS runner expected scalar `scores.csv`; MBS harness only wrote rich
  `scores.json` → runner couldn't consume it.
- Feedback used the generic DS summarizer; MBS personas never loaded.
- No phase gating, no domain validation, no memory updates — each loop
  iteration was stateless from MBS's perspective.
- The `get_scenario_all_desc()` inherited from `DataScienceScen`
  omitted MBS phase/search/memory context from downstream LLM prompts.

### Change 1 — `scores.json` → `scores.csv` bridge

File: [scaffold.py](scaffold.py) — `run_scaffold_pipeline()`

```python
# Bridge: write scores.csv so the DS runner can consume the primary metric.
primary_value = scorecard.get("primary_metric", {}).get("value", float("nan"))
metric_name   = scorecard.get("primary_metric", {}).get("name", "rmse_smm_decimal")
pd.DataFrame(
    {metric_name: [primary_value]},
    index=pd.Index(["ensemble"], name=""),
).to_csv(output_dir / "scores.csv")
```

Both files are now written: `scores.json` (full MBS scorecard, consumed
by feedback + record) and `scores.csv` (scalar metric, consumed by the
DS runner's `DSRunnerEvaluator`).

### Change 2 — `get_scenario_all_desc()` override

File: [scenario.py](scenario.py) — `MBSPrepaymentScen`

Every downstream LLM call (proposal, coding, feedback) reads
`scen.get_scenario_all_desc()`. The override appends four MBS sections
to the base DS description on every call:

1. **Current phase + gate criteria** — from `mbs_orchestrator.phase_spec()`
2. **Search-strategy constraints** — from `mbs_orchestrator.iteration_constraints()` via `format_filter_for_prompt()`
3. **MBS memory context** — `mbs_memory.render_context(IterationPhase.HYPOTHESIS_GEN)`
4. **Data contract reminder** — target, forbidden leakage columns,
   required features

Single injection point → all LLM stages become phase-aware without
modifying DS templates.

### Change 3 — `MBSExperiment2Feedback`

File: [feedback.py](feedback.py) (new) — subclass of
`DSExperiment2Feedback`

- Prepends `MODEL_VALIDATOR.system_prompt` to the feedback system
  prompt.
- Reads `scores.json` from `exp.experiment_workspace.file_dict` and
  appends it (JSON-fenced) to the user prompt, with a directive that
  an experiment improving overall RMSE but degrading per-coupon
  uniformity / regime robustness / monotonicity is a REJECT.
- Appends `mbs_memory.render_context(IterationPhase.FEEDBACK)` to the
  user prompt.

### Change 4 — `MBSPrepaymentRDLoop`

File: [loop.py](loop.py) (new) — subclass of `DataScienceRDLoop`

- `__init__`: validates the scenario is MBS-compatible (has
  `mbs_orchestrator`); replaces `self.summarizer` with
  `MBSExperiment2Feedback`.
- `feedback()`: reads scorecard via `_read_scorecard()`, runs
  `DomainValidator` via `_domain_validate()`. If validation fails the
  experiment is **auto-rejected with no LLM call** (saves a round
  trip); otherwise falls through to the persona-led LLM feedback.
- `record()`: after the base record logic:
  - `_update_memory()` — appends a `TraceEntry` (with
    `ModelProperties.from_scorecard()` on success) to `MBSMemory`.
  - `_update_search_state()` — appends an `IterationRecord` to
    `MBSSearchState` (curriculum / cooldown / exploration mode).
  - `mbs_orchestrator.evaluate_gate()` — if passed, calls
    `advance_phase()` and logs the transition.

### Change 5 — CLI command + app entry point

Files:
- [../../app/mbs_prepayment/__init__.py](../../app/mbs_prepayment/__init__.py) (new, empty)
- [../../app/mbs_prepayment/loop.py](../../app/mbs_prepayment/loop.py) (new) — `main()` instantiates `MBSPrepaymentRDLoop(DS_RD_SETTING)` (not `DataScienceRDLoop`)
- [../../app/cli.py](../../app/cli.py) — new `@app.command(name="mbs_prepayment")` registering the CLI

Invocation:

```bash
rdagent mbs_prepayment --competition mbs_prepayment
```

### Verification

- 91 tests pass (one pre-existing unrelated failure in
  `test_cache_stats_and_clear` excluded).
- `rdagent mbs_prepayment --help` registers and parses the expected
  flags (`--path`, `--checkout/--no-checkout`, `--step-n`, `--loop-n`,
  `--timeout`, `--competition`).

---

## Files in this folder

| File | Role |
|---|---|
| [conf.py](conf.py) | `MBSPrepaymentSettings` + module-level `MBSP_SETTINGS` |
| [scenario.py](scenario.py) | `MBSPrepaymentScen(DataScienceScen)` — DS loop entry |
| [loop.py](loop.py) | `MBSPrepaymentRDLoop(DataScienceRDLoop)` — wires MBS modules into the live loop |
| [feedback.py](feedback.py) | `MBSExperiment2Feedback` — persona + scorecard + memory-aware feedback |
| [scaffold.py](scaffold.py) | `MBSDataContract`, `MBSTrainTestSplit` (Priority 2) |
| [evaluation.py](evaluation.py) | `MBSEvaluationHarness` scorecard (Priority 1) |
| [interpretability.py](interpretability.py) | Integrated Gradients attribution (Priority 4) |
| [eda.py](eda.py) | MBS-specific EDA helpers (Priority 5) |
| [search_strategy.py](search_strategy.py) | `MBSSearchState` curriculum/cooldown (Priority 6) |
| [memory.py](memory.py) | `MBSMemory` phase-aware memory (Priority 7) |
| [orchestration.py](orchestration.py) | `DomainValidator`, `PhaseGate`, `MBSOrchestrator` (Priority 8) |
| [execution_env.py](execution_env.py) | Per-stage budgets (Priority 9) |
| [personas.py](personas.py) | `PersonaRouter` (Priority 10) |
| [prompts.yaml](prompts.yaml) | MBS-specific prompt templates (Priority 3) |
| [example/description.md](example/description.md) | Copy to `./mbs_data/mbs_prepayment/description.md` |
