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
| `./mbs_data/mbs_prepayment/tfminput.parquet` | **Single panel file** — all cusips, all months, all features, the `smm_decimal` target | — |
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

### 4. Drop in the panel data

This scenario expects a **single parquet file** at
`./mbs_data/mbs_prepayment/tfminput.parquet` containing the full
CUSIP-level monthly panel (all cusips, all `fh_effdt` months, all
feature columns, and the `smm_decimal` target in one file). The file
must satisfy the data contract in [scaffold.py](scaffold.py):

- Panel key `(cusip, fh_effdt)` — one row per CUSIP per month
- Target column `smm_decimal` ∈ [0, 1]
- Required feature columns: `rate_incentive, coupon, wala`
- **No** forbidden leakage columns: `future_smm, forward_smm,
  next_month_smm, forward_rate, future_rate_incentive`
- Macro features lagged ≥ 30 days

The train/test split is performed in-memory on `fh_effdt`
(`<= 2021-12-31` for train, `> 2021-12-31` for test) — there are no
separate train/test files. The shipped `description.md` already tells
the coder how to load and split:

```python
import pandas as pd
df = pd.read_parquet("tfminput.parquet")
train = df[df["fh_effdt"] <= "2021-12-31"]
test  = df[df["fh_effdt"] >  "2021-12-31"]
```

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

    train_end_date: str = "2021-12-31"
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

### New calling chain

```
rdagent mbs_prepayment
  └── rdagent/app/mbs_prepayment/loop.py :: main()
        └── MBSPrepaymentRDLoop(DS_RD_SETTING)          # loop.py (this folder)
              ├── __init__:   swap summarizer → MBSExperiment2Feedback (feedback.py)
              ├── direct_exp_gen / coding / running     # inherited from DataScienceRDLoop
              │     └── every LLM call reads scen.get_scenario_all_desc()
              │           → injects phase + search constraints + MBS memory + data contract
              ├── feedback:   DomainValidator auto-reject  →  MODEL_VALIDATOR-led LLM feedback
              └── record:     MBSMemory.append_entry()
                              MBSSearchState.append()
                              MBSOrchestrator.evaluate_gate() → advance_phase()
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
