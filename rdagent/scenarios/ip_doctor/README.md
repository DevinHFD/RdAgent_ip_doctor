# MBS Prepayment Attribution Analysis

An agentic workflow that answers natural language questions about an MBS prepayment model's behavior using **Integrated Gradients** (Captum). The agent generates analysis code, executes it, pauses for human review, and produces a markdown report with matplotlib/seaborn plots.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Analysis Types](#analysis-types)
4. [Attribution Semantics](#attribution-semantics)
5. [File Structure](#file-structure)
6. [Prerequisites](#prerequisites)
7. [Setup](#setup)
8. [Data & Model Requirements](#data--model-requirements)
9. [Configuration Reference](#configuration-reference)
10. [Running the Workflow](#running-the-workflow)
11. [Human Review Interface](#human-review-interface)
12. [Output Files](#output-files)
13. [Workflow Internals](#workflow-internals)
14. [Troubleshooting](#troubleshooting)

---

## Overview

Given a natural language question such as:

> *"For CUSIP 3140GXPJ8, what features drove the CPR change from January to February 2024?"*

the workflow will:

1. **Parse** the question to identify analysis type and CUSIPs
2. **Plan** a structured `AnalysisPlan` (model checkpoint, features, IG parameters)
3. **Generate** a Python attribution script from a Jinja template (deterministic, no hallucination)
4. **Validate** the script for syntax and security
5. **Execute** it in a subprocess — runs Integrated Gradients, writes `output.json`
6. **Pause** for human review — shows top driving features in plain language
7. **Report** — generates PNG plots and an LLM-written markdown narrative

The human reviewer can **approve** (proceed to report) or **reject with feedback** (loop back to the planner for a revised plan).

---

## Architecture

The workflow is implemented as a **LangGraph `StateGraph`** with a `MemorySaver` checkpointer, enabling session persistence and the human-in-the-loop `interrupt()` pattern.

```
question_parser
      │
      ▼
   planner  ◄──────────────────────────────────────┐
      │                                             │  (reject)
      ▼                                             │
code_generator ◄──────────────────┐           human_reviewer
      │                           │  (invalid)      │
      ▼                           │                 │  (approve)
code_validator ──────────────────►│                 │
      │  (valid)                                    ▼
      ▼                                          reporter
   executor                                         │
   /     \                                          ▼
(error) (success)                                  END
   │         └──────────────────► human_reviewer
   ▼
debugger ──► (retry < 3) ──► code_validator
         └── (give up)  ──► human_reviewer
```

### Node Responsibilities

| Node | Type | Description |
|------|------|-------------|
| `question_parser` | LLM | Classifies question type (`cusip_attribution` / `scenario_comparison`), extracts CUSIPs and scenario parameters |
| `planner` | LLM | Generates a structured `AnalysisPlan` JSON — model checkpoint, feature names (read from actual parquet), IG parameters |
| `code_generator` | Template + LLM (fix only) | Renders a Jinja template to Python on first pass; uses LLM only to fix validation errors |
| `code_validator` | Pure Python | Syntax check, security scan (forbids `os.system`, bare `eval()`, `subprocess`, etc.), output contract check |
| `executor` | Subprocess | Runs the generated script in isolation; parses `output.json` into `ExecutionResult` |
| `debugger` | LLM | Repairs runtime errors (up to `max_debug_attempts`); if exhausted, forwards to human_reviewer |
| `human_reviewer` | CLI interrupt | Calls `interrupt()` to pause the graph; presents plan summary + top attribution features; resumes on human input |
| `reporter` | LLM + matplotlib | Generates PNG plots and an LLM markdown narrative |

---

## Analysis Types

### `cusip_attribution` — Month-over-Month

Explains what features drove the prepayment (CPR/SMM) change between consecutive months for specific CUSIPs.

**Example questions:**
```
"For CUSIP 3140GXPJ8, what drove the CPR change from January to February 2024?"
"Show me month-over-month attribution for CUSIPs ABC123 and DEF456 from Q1 2024."
```

**How it works:** Integrated Gradients runs with month T as input and month T−1 (or zero) as baseline. Attribution per feature = contribution to the predicted CPR/SMM change.

---

### `scenario_comparison` — Stress Scenario vs Base

Compares a stress scenario (e.g., +100 bps rate shock) against a base scenario to explain model output differences across CUSIPs.

**Example questions:**
```
"Compare the rate shock +100bps scenario against base for CUSIPs ABC123 and DEF456."
"What features explain the difference between base and rate_shock_200bps across all CUSIPs?"
```

**How it works:** IG baseline = zero (or base scenario rows). The LLM planner maps scenario names to date windows in the single parquet file:
```json
{
  "base":           ["2023-01", "2023-06"],
  "rate_shock_100": ["2023-07", "2023-12"]
}
```

---

## Attribution Semantics

| Value | Meaning | Scale | Used for |
|-------|---------|-------|----------|
| `attributions_normalized` | IG output — contribution of this feature to the CPR/SMM change | Model output units (e.g., `+0.0012 SMM`) | Bar chart x-axis, report narrative |
| `feature_values_original` | Actual feature level inverse-transformed to original scale | Feature units (e.g., WALA: `120.3 months`) | Y-axis labels, report context |

**Key principle:** Attribution values are *not* rescaled to feature units. A value of `+0.0012` means "this feature contributed +0.0012 SMM to the total output change." The original-scale feature value (e.g., WALA = 120.3 months) is shown as context so readers know *at what level* the feature was when it produced that attribution.

---

## File Structure

```
rdagent/scenarios/mbs_prepayment/
├── README.md                            # this file
├── __init__.py                          # exports build_graph
├── app.py                               # CLI entry point (typer)
├── conf.py                              # MBSPrepaymentSettings (pydantic-settings)
├── graph.py                             # LangGraph StateGraph assembly
├── state.py                             # MBSAnalysisState TypedDict + Pydantic models
├── nodes/
│   ├── __init__.py
│   ├── prompts.yaml                     # All LLM prompts (question_parser, planner,
│   │                                    #   code_generator, debugger, reporter)
│   ├── question_parser.py
│   ├── planner.py
│   ├── code_generator.py
│   ├── code_validator.py
│   ├── executor.py
│   ├── debugger.py
│   ├── human_reviewer.py
│   └── reporter.py
└── templates/
    ├── ig_cusip_attribution.py.jinja    # IG script template for month-over-month
    └── ig_scenario_comparison.py.jinja  # IG script template for scenario comparison
```

---

## Prerequisites

- Python 3.10+
- A PyTorch prepayment model saved with `torch.save()` (`.pth` or `.pt`)
- A `sklearn.preprocessing.StandardScaler` saved with `joblib.dump()` (`.sav`)
- Loan-level data in a single Parquet file with `cusip` and `fh_effdt` columns
- An OpenAI-compatible LLM API key (GPT-4o recommended)

---

## Setup

### 1. Install dependencies

```bash
cd /path/to/RD-Agent

# Install RD-Agent (editable)
make install

# Install MBS-specific dependencies
pip install "langgraph>=0.2" captum joblib
```

> `matplotlib`, `seaborn`, and `scikit-learn` are already in `requirements.txt`.

---

### 2. Configure LLM credentials

```bash
cp .env.example .env
```

Edit `.env` — minimum required:

```dotenv
CHAT_MODEL=gpt-4o
OPENAI_API_KEY=sk-...

# Azure alternative:
# AZURE_API_KEY=...
# AZURE_API_BASE=...
# AZURE_API_VERSION=...
# CHAT_MODEL=azure/gpt-4o
```

---

### 3. Prepare model files

```bash
mkdir -p mbs_models
```

Save your PyTorch model and scaler from Python:

```python
import torch
import joblib

# Save the model
torch.save(model, "mbs_models/model.pth")

# Save the fitted StandardScaler
joblib.dump(scaler, "mbs_models/scaler.sav")
```

> The scaler must be fitted on the **same feature set and order** as the model's input layer.

---

## Data & Model Requirements

### Parquet file

Place your data file at `mbs_data/data.parquet` (configurable via `MBS_DATA_FILE`).

Required columns:

| Column | Type | Example | Description |
|--------|------|---------|-------------|
| `cusip` | string | `"3140GXPJ8"` | CUSIP identifier (name configurable via `MBS_CUSIP_COL`) |
| `fh_effdt` | int | `20240301` | Effective date in `YYYYMMDD` integer format (configurable via `MBS_DATE_COL` / `MBS_DATE_FORMAT`) |
| `<feature_1>` ... `<feature_N>` | float | `-0.312` | **Normalized** feature values (mean=0, std=1) — exactly as fed to the model during training |

> **Important:** All feature columns must already be z-scored (normalized). The workflow uses the scaler only to inverse-transform values back to original scale for human-readable display — it does not re-normalize.

### Model checkpoint

- File format: anything `torch.load()` accepts (`.pth`, `.pt`)
- Must be a callable `nn.Module`; the script calls `model.eval()` before attribution
- Output: the model must return a tensor where index 0 = CPR prediction, index 1 = SMM prediction (or set `MBS_IG_TARGET_OUTPUT` to match your output structure)

### Scaler

- Format: joblib-serialized `sklearn.preprocessing.StandardScaler` (`.sav`)
- Must match the feature order in the parquet file exactly
- Used via `scaler.inverse_transform()` to convert normalized feature values back to original scale for plots and reports

---

## Configuration Reference

All settings use the `MBS_` environment variable prefix. Set them in `.env`, in your shell, or inline.

```bash
# Filesystem paths
MBS_MODEL_CHECKPOINT_DIR=./mbs_models        # directory scanned for .pth / .pt files
MBS_DATA_FILE=./mbs_data/data.parquet        # single parquet with all loan data
MBS_OUTPUT_DIR=./mbs_output                  # where plots and output.json are saved
MBS_SCALER_PATH=./mbs_models/scaler.sav      # joblib-saved sklearn StandardScaler

# Column names in the parquet
MBS_CUSIP_COL=cusip                          # CUSIP identifier column name
MBS_DATE_COL=fh_effdt                        # effective date column name
MBS_DATE_FORMAT=%Y%m%d                       # strptime format for the date column
                                             # "%Y%m%d" handles integer 20240301
                                             # "%Y-%m-%d" handles string "2024-03-01"

# Integrated Gradients
MBS_IG_BASELINE_STRATEGY=zero               # "zero" = zero-input baseline
                                             # "mean" = previous month as baseline
MBS_IG_N_STEPS=50                           # IG approximation steps (more = more accurate)
MBS_IG_TARGET_OUTPUT=cpr                    # "cpr" (model output index 0)
                                             # "smm" (model output index 1)
MBS_IG_BATCH_SIZE=32                        # rows per IG batch (reduce if OOM)

# Execution safety
MBS_EXECUTION_TIMEOUT_SECONDS=300           # max seconds for the IG subprocess
MBS_MAX_DEBUG_ATTEMPTS=3                    # LLM debug retries on runtime error

# Workflow control
MBS_MAX_ITERATIONS=10                       # hard cap on planner→review loops
MBS_SKIP_HUMAN_REVIEW=false                 # true = auto-approve (CI / batch runs)
```

---

## Running the Workflow

### Month-over-month attribution (single CUSIP)

```bash
python -m rdagent.scenarios.mbs_prepayment.app \
    "For CUSIP 3140GXPJ8, what features drove the CPR change from January to February 2024?"
```

### Month-over-month attribution (multiple CUSIPs)

```bash
python -m rdagent.scenarios.mbs_prepayment.app \
    "For CUSIPs 3140GXPJ8 and 31418DSB5, show month-over-month CPR attribution for Q1 2024."
```

### Scenario comparison

```bash
python -m rdagent.scenarios.mbs_prepayment.app \
    "Compare base vs rate_shock_100bps for CUSIPs 3140GXPJ8 and 31418DSB5. \
     Base period: Jan-Jun 2024. Shock period: Jul-Dec 2024."
```

### SMM instead of CPR

```bash
MBS_IG_TARGET_OUTPUT=smm python -m rdagent.scenarios.mbs_prepayment.app \
    "For CUSIP 3140GXPJ8, what drove the SMM change from March to April 2024?"
```

### Skip human review (automated / CI)

```bash
python -m rdagent.scenarios.mbs_prepayment.app \
    --skip-review \
    "For CUSIP 3140GXPJ8, what drove the January-February CPR change?"
```

### Resume an interrupted session

The session ID is printed at the start of every run:

```
Session ID: a1b2c3d4-e5f6-7890-abcd-ef1234567890
```

Resume with:

```bash
python -m rdagent.scenarios.mbs_prepayment.app \
    --session-id a1b2c3d4-e5f6-7890-abcd-ef1234567890 \
    "For CUSIP 3140GXPJ8, ..."
```

---

## Human Review Interface

When IG execution succeeds the workflow pauses and shows:

```
======================================================================
  HUMAN REVIEW REQUIRED
======================================================================
Question  : For CUSIP 3140GXPJ8, what drove the Jan→Feb CPR change?
Iteration : 1

--- Analysis Plan ---
Analysis type : cusip_attribution
CUSIPs        : ['3140GXPJ8']
Baseline      : zero  n_steps=50  target=cpr
Rationale     : Month-over-month IG using zero baseline for Jan→Feb 2024.

--- Execution Summary ---
Top features by mean |attribution| (SMM/CPR contribution):
  WALA                            mean=+0.0023  std=0.0005
  WAC                             mean=-0.0018  std=0.0003
  loan_age                        mean=+0.0011  std=0.0002
  ...

Respond with:  approve  or  reject
======================================================================

Your decision >
```

**Approve** — proceeds to `reporter` (plots + LLM narrative).

**Reject** — guided prompts collect structured feedback:

```
Your decision > reject
What is wrong? > wrong months selected
Suggested change? > use Feb→Mar instead of Jan→Feb
Focus CUSIPs (comma-separated, or blank)? >
```

The planner re-runs with your feedback incorporated. The loop continues until approval or `MBS_MAX_ITERATIONS` is reached.

You may also provide raw JSON:

```
Your decision > {"decision": "reject", "what_is_wrong": "wrong months", "suggested_change": "use Feb→Mar"}
```

---

## Output Files

After a successful run, `MBS_OUTPUT_DIR` (default `./mbs_output`) contains:

```
mbs_output/
├── output.json                          # raw IG results (from executor subprocess)
├── attribution_<CUSIP>.png              # bar chart per CUSIP (cusip_attribution mode)
└── scenario_comparison_heatmap.png      # feature × scenario heatmap (scenario_comparison mode)
```

The markdown report is printed to stdout and stored in the LangGraph session state (accessible on resume via `--session-id`).

### `output.json` schema

```json
{
  "analysis_type": "cusip_attribution",
  "attributions_normalized": {
    "<CUSIP>": {
      "2024-01->2024-02": {
        "<feature>": 0.00123
      }
    }
  },
  "feature_values_original": {
    "<CUSIP>": {
      "2024-01->2024-02": {
        "<feature>": 120.3
      }
    }
  },
  "metadata": {
    "n_steps": 50,
    "baseline": "zero",
    "target_output": "cpr",
    "cusip_count": 1,
    "period_count": 1,
    "feature_count": 42,
    "data_file": "./mbs_data/data.parquet"
  },
  "summary_stats": {
    "<feature>": { "mean_attr": 0.00089, "std_attr": 0.00021 }
  }
}
```

### Plots

**`cusip_attribution` — horizontal bar chart per CUSIP**

- X-axis: IG attribution (CPR/SMM contribution per feature)
- Y-axis: feature name + average original-scale value, e.g. `"WALA  (avg: 120.3)"`
- Red = positive contribution to prepayment speed, blue = negative
- Shows top 15 features by `|mean attribution|`

**`scenario_comparison` — seaborn heatmap**

- Rows: top 20 features by maximum absolute attribution across scenarios
- Columns: stress scenario names
- Cell values averaged across all CUSIPs
- Y-axis labels include original-scale feature context

---

## Workflow Internals

### State flow

```
app.py sets initial_state (question, session_id)
    │
    ▼ question_parser → sets question_type, cusip_list, scenario_params
    │
    ▼ planner → reads feature names from parquet header (zero rows, no data load)
    │           reads checkpoint filenames from model_checkpoint_dir
    │           LLM fills in comparison_months / scenario_params
    │           → sets analysis_plan
    │
    ▼ code_generator → renders Jinja template (no LLM on first pass)
    │                  LLM only called if code_validator returns errors
    │                  → sets generated_code
    │
    ▼ code_validator → ast.parse + forbidden pattern regex + output.json contract
    │                  → sets code_valid, validation_errors
    │
    ▼ executor → subprocess(generated_code, timeout=300s)
    │            parses output.json → ExecutionResult
    │            → sets execution_result or execution_error
    │
    ▼ human_reviewer → interrupt() surfaces review payload to app.py
    │                  app.py collects CLI input, resumes with Command(resume=feedback)
    │                  → sets human_feedback
    │
    ▼ reporter → _avg_across_periods() flattens {period: {feat: val}}
                 matplotlib bar charts / seaborn heatmap saved to output_dir
                 LLM writes markdown narrative
                 → sets report_markdown, plot_paths
```

### Prompt resolution

All LLM prompts live in `nodes/prompts.yaml`. The `T(".prompts:key.subkey")` call resolves the path relative to the **calling file's directory** — since all node files are in `nodes/`, they correctly find `nodes/prompts.yaml`.

### Code generation strategy

On first pass, `code_generator` renders a **deterministic Jinja template** — fast, reproducible, no LLM cost. The LLM is invoked *only* if `code_validator` reports errors, to fix them. This makes the happy path entirely template-driven.

### Feature name and checkpoint discovery

`planner_node` reads the parquet schema (zero rows — no full file load) and scans `model_checkpoint_dir` for `.pth`/`.pt` files at runtime before calling the LLM. The prompt instructs the LLM to use these exact values — preventing hallucinated feature names and checkpoint paths.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `Cannot find .prompts:question_parser.system` | Old separate `prompts/*.yaml` files | Prompts must be in `nodes/prompts.yaml` — separate files in a subdirectory are not resolved |
| `Forbidden pattern 'eval()' found` | `model.eval()` caught by validator | Already fixed — validator uses negative lookbehind `(?<!\.)\beval\s*\(` |
| `No data for CUSIP=X, month=2024-01` | Integer date column not parsed | Confirm `MBS_DATE_FORMAT=%Y%m%d`; column should contain integers like `20240101` |
| Feature names wrong or too short | LLM guessed instead of reading parquet | Planner now auto-reads parquet columns; ensure `MBS_DATA_FILE` is accessible at run time |
| Checkpoint set to `latest_checkpoint.pt` | Old default or LLM guess | Place model in `MBS_MODEL_CHECKPOINT_DIR`; planner scans for `.pth`/`.pt` and passes exact paths to LLM |
| Human review not pausing | `interrupt_before` conflict (old version) | Fixed — `interrupt()` inside `human_reviewer_node` is the sole pause mechanism |
| No plots generated | Reporter iterating wrong dict level | Fixed — `_avg_across_periods()` correctly flattens `{period: {feature: val}}` before sorting |
| OOM during IG | Batch size too large | Reduce `MBS_IG_BATCH_SIZE=8` or lower |
| Slow / inaccurate IG | Too few approximation steps | Increase `MBS_IG_N_STEPS=100` or `200` |
| `FileNotFoundError: scaler.sav` | Wrong scaler path | Set `MBS_SCALER_PATH=/absolute/path/to/scaler.sav` |
