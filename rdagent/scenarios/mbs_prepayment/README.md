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
| `./mbs_data/mbs_prepayment/*.parquet` | CUSIP-level panel data | — |
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

Place your CUSIP-level parquet files under
`./mbs_data/mbs_prepayment/`. The files must satisfy the data
contract in [scaffold.py](scaffold.py):

- Panel key `(cusip, fh_effdt)` — one row per CUSIP per month
- Target column `smm_decimal` ∈ [0, 1]
- Required feature columns: `rate_incentive, coupon, wala`
- **No** forbidden leakage columns: `future_smm, forward_smm,
  next_month_smm, forward_rate, future_rate_incentive`
- Macro features lagged ≥ 30 days

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
rdagent data_science --competition mbs_prepayment
```

This invokes `DataScienceRDLoop` with `MBSPrepaymentScen` as the
scenario. The loop will:

1. Read `description.md` from the competition folder
2. Instantiate `MBSPrepaymentScen`, attaching all ten Priority modules
3. Run the proposal → codegen → execution → feedback loop, with the
   MBS evaluation harness scoring each iteration and `PhaseGate` /
   `DomainValidator` rejecting models that degrade per-coupon
   uniformity or rate-sensitivity monotonicity

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

## Files in this folder

| File | Role |
|---|---|
| [conf.py](conf.py) | `MBSPrepaymentSettings` + module-level `MBSP_SETTINGS` |
| [scenario.py](scenario.py) | `MBSPrepaymentScen(DataScienceScen)` — DS loop entry |
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
