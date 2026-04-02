# IP Doctor — MBS Prepayment Attribution Analysis

An agentic workflow that answers natural language questions about an MBS prepayment model's behavior using **Integrated Gradients** (Captum). The agent generates analysis code, executes it, pauses for human review, and produces a self-contained **PDF report** with embedded matplotlib plots.

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
    - [Option A — Streamlit Chat UI](#option-a--streamlit-chat-ui)
    - [Option B — Gradio Chat UI](#option-b--gradio-chat-ui)
    - [Option C — CLI](#option-c--cli)
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
3. **Generate** a Python attribution script from a Jinja template (deterministic, no hallucination on first pass)
4. **Validate** the script for syntax and security
5. **Execute** it in a subprocess — runs Integrated Gradients, writes `output.json` with T0/T1 feature values, model predictions, and attributions
6. **Pause** for human review — shows top features with Before/After/Δ original-scale values and per-CUSIP SMM forecast changes
7. **Report** — generates PNG plots (per-CUSIP + aggregated mean across all CUSIPs) and an LLM-written markdown narrative, then renders everything into a self-contained PDF

The human reviewer can **approve** (proceed to report) or **reject with feedback** (loop back to the planner, optionally adding new CUSIPs to the analysis).

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
debugger ──► (retry < N) ──► code_validator
         └── (give up)  ──► human_reviewer
```

### Node Responsibilities

| Node | Type | Description |
|------|------|-------------|
| `question_parser` | LLM | Classifies question type (`cusip_attribution` / `scenario_comparison`), extracts CUSIPs and scenario parameters |
| `planner` | LLM | Generates a structured `AnalysisPlan` JSON — model checkpoint, feature names (read from actual parquet header), IG parameters. On reject loops, merges `focus_cusips` from human feedback into the authoritative CUSIP list |
| `code_generator` | Template + LLM (fix only) | Renders a Jinja template to Python on first pass; uses LLM only to fix validation errors |
| `code_validator` | Pure Python | Syntax check, security scan (forbids `os.system`, bare `eval()`, `subprocess`, etc.), output contract check |
| `executor` | Subprocess | Runs the generated script in isolation; parses `output.json` into `ExecutionResult` |
| `debugger` | LLM | Repairs runtime errors (up to `MBS_MAX_DEBUG_ATTEMPTS`); if exhausted, forwards to `human_reviewer` |
| `human_reviewer` | CLI interrupt | Calls `interrupt()` to pause the graph; presents plan summary, top features with Before/After/Δ values, and per-CUSIP SMM forecast; resumes on human input |
| `reporter` | LLM + matplotlib + WeasyPrint | Generates PNG plots (one per CUSIP + one aggregated), an LLM markdown narrative, and a self-contained PDF report |

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

Both T0 (previous month) and T1 (current month) feature values are inverse-transformed to original scale and stored alongside the attributions. The delta (T1 − T0) is also stored.

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
| `attributions_normalized` | IG output — contribution of this feature to CPR/SMM change | Model output units (e.g., `+0.0012 SMM`) | Bar chart x-axis, report narrative |
| `feature_values_original` | Actual feature values, both endpoints and delta, in original scale | Feature units (e.g., WALA: `120.3 months`) | Y-axis labels, report context |
| `model_predictions` | Raw model output (SMM and CPR) for T0 and T1 (or base and scenario) | SMM/CPR decimal (e.g., `0.0145`) | Human review prediction table, report narrative |

**Key principle:** Attribution values are *not* rescaled to feature units. `+0.0012` means "this feature contributed +0.0012 SMM to the total output change." The original-scale feature values (e.g., WALA moved from `120.3 → 121.3 months, Δ+1.0`) are shown as context so readers understand *what changed* in the loan pool that produced that attribution.

**Feature importance ranking** in all outputs (plots, report, human review) is sorted by `|mean attribution|` — not by feature value delta size.

---

## File Structure

```
rdagent/scenarios/ip_doctor/
├── README.md                            # this file
├── __init__.py                          # exports build_graph
├── app.py                               # CLI entry point (typer)
├── conf.py                              # MBSPrepaymentSettings (pydantic-settings, MBS_ prefix)
├── graph.py                             # LangGraph StateGraph assembly
├── state.py                             # MBSAnalysisState TypedDict + Pydantic models
├── nodes/
│   ├── __init__.py
│   ├── prompts.yaml                     # All LLM prompts (question_parser, planner,
│   │                                    #   code_generator, debugger, reporter)
│   ├── question_parser.py
│   ├── planner.py                       # also merges focus_cusips from human feedback
│   ├── code_generator.py
│   ├── code_validator.py
│   ├── executor.py
│   ├── debugger.py
│   ├── human_reviewer.py                # shows Before/After/Δ feature table + SMM predictions
│   └── reporter.py                      # plots + LLM narrative + PDF generation
├── templates/
│   ├── ig_cusip_attribution.py.jinja    # IG script template for month-over-month
│   └── ig_scenario_comparison.py.jinja  # IG script template for scenario comparison
└── ui/
    ├── __init__.py
    ├── streamlit_app.py                 # Streamlit chatbot UI  ← recommended
    └── gradio_app.py                    # Gradio chatbot UI
```

---

## Prerequisites

- Python 3.10+
- A PyTorch prepayment model saved with `torch.save()` (`.pth` or `.pt`)
- A `sklearn.preprocessing.StandardScaler` saved with `joblib.dump()` (`.sav`)
- Loan-level data in a single Parquet file with `cusip` and `fh_effdt` columns
- An OpenAI-compatible LLM API key (GPT-4o recommended)
- System libraries for WeasyPrint PDF rendering (see Setup)
- `streamlit` for the Streamlit UI, or `gradio` for the Gradio UI (see Running the Workflow)

---

## Setup

### 1. Install dependencies

```bash
cd /path/to/RD-Agent

# Install RD-Agent (editable)
make install

# Install ip_doctor-specific dependencies
pip install "langgraph>=0.2" captum joblib markdown weasyprint
```

WeasyPrint requires system libraries for PDF rendering:

```bash
# Debian / Ubuntu
sudo apt-get install libpango-1.0-0 libpangocairo-1.0-0 libcairo2

# macOS (Homebrew)
brew install pango cairo
```

> If WeasyPrint cannot be installed, the workflow still runs fully — PDF generation is skipped with a warning and `report.md` is still written.

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
| `cusip` | string | `"3140GXPJ8"` | CUSIP identifier (configurable via `MBS_CUSIP_COL`) |
| `fh_effdt` | int | `20240301` | Effective date in `YYYYMMDD` integer format (configurable via `MBS_DATE_COL` / `MBS_DATE_FORMAT`) |
| `<feature_1>` ... `<feature_N>` | float | `-0.312` | **Normalized** feature values (mean=0, std=1) — exactly as fed to the model during training |

> **Important:** All feature columns must already be z-scored (normalized). The scaler is used only to inverse-transform values back to original scale for display — the workflow does not re-normalize inputs.

### Model checkpoint

- File format: anything `torch.load()` accepts (`.pth`, `.pt`)
- Must be a callable `nn.Module`; the script calls `model.eval()` before attribution
- Output shape: tensor where index 0 = CPR, index 1 = SMM (or configure `MBS_IG_TARGET_OUTPUT`)

### Scaler

- Format: joblib-serialized `sklearn.preprocessing.StandardScaler` (`.sav`)
- Must match the feature column order in the parquet file exactly
- Used via `scaler.inverse_transform()` to show original-scale feature values in plots and reports

---

## Configuration Reference

All settings use the `MBS_` environment variable prefix. Set them in `.env`, in your shell, or inline.

```bash
# Filesystem paths
MBS_MODEL_CHECKPOINT_DIR=./mbs_models        # directory scanned for .pth / .pt files
MBS_DATA_FILE=./mbs_data/data.parquet        # single parquet with all loan data
MBS_OUTPUT_DIR=./mbs_output                  # where plots, output.json, and report.pdf are saved
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

### Option A — Streamlit Chat UI

The Streamlit UI provides a Claude Code–inspired chat experience with live step tracking and a rich review card.

#### Install

```bash
pip install streamlit
```

#### Launch

```bash
# From the repo root
streamlit run rdagent/scenarios/ip_doctor/ui/streamlit_app.py
```

Streamlit opens `http://localhost:8501` automatically. To customise the port or run headless on a remote server:

```bash
streamlit run rdagent/scenarios/ip_doctor/ui/streamlit_app.py \
    --server.port 8080 --server.headless true
```

#### UI walkthrough

1. **Type a question** in the chat box at the bottom (or click an example in the sidebar).
2. **Watch the steps** appear live as each agent node fires — each step is a compact colour-coded pill:
   - 🔵 Blue ⟳ = currently running
   - 🟢 Green ✓ = completed
   - 🔴 Red ✗ = error
3. **Review card** appears when the executor finishes:
   - Structured analysis plan (type, CUSIPs, baseline, rationale)
   - Horizontal bar chart of top feature IG attributions (green = positive, red = negative)
   - Before / After / Δ table with original-scale feature values
   - Model SMM prediction table (T0, T1, Δ per CUSIP)
   - **Approve ✓** button to proceed to the final report
   - **Reject ✗** option expands a form: *What is wrong?*, *Suggested change*, *Additional CUSIPs to add*
4. **Final report** renders as formatted markdown with inline attribution plots and a **Download PDF** button.

---

### Option B — Gradio Chat UI

The Gradio UI offers collapsible step panels and inline gallery output.

```bash
pip install "gradio>=4.0"
python rdagent/scenarios/ip_doctor/ui/gradio_app.py
# custom port:
python rdagent/scenarios/ip_doctor/ui/gradio_app.py --server_port 7861
# public share link (for remote access):
python rdagent/scenarios/ip_doctor/ui/gradio_app.py --share
```

Open `http://localhost:7860` in your browser.

---

### Option C — CLI

```bash
# Month-over-month attribution (single CUSIP)
python -m rdagent.scenarios.ip_doctor.app \
    "For CUSIP 3140GXPJ8, what features drove the CPR change from January to February 2024?"

# Month-over-month attribution (multiple CUSIPs)
python -m rdagent.scenarios.ip_doctor.app \
    "For CUSIPs 3140GXPJ8 and 31418DSB5, show month-over-month CPR attribution for Q1 2024."

# Scenario comparison
python -m rdagent.scenarios.ip_doctor.app \
    "Compare base vs rate_shock_100bps for CUSIPs 3140GXPJ8 and 31418DSB5. \
     Base period: Jan-Jun 2024. Shock period: Jul-Dec 2024."

# Target SMM instead of CPR
MBS_IG_TARGET_OUTPUT=smm python -m rdagent.scenarios.ip_doctor.app \
    "For CUSIP 3140GXPJ8, what drove the SMM change from March to April 2024?"

# Skip human review (automated / CI)
python -m rdagent.scenarios.ip_doctor.app \
    --skip-review \
    "For CUSIP 3140GXPJ8, what drove the January-February CPR change?"

# Resume an interrupted session
python -m rdagent.scenarios.ip_doctor.app \
    --session-id a1b2c3d4-e5f6-7890-abcd-ef1234567890 \
    "For CUSIP 3140GXPJ8, ..."
```

The session ID is printed at the start of every run:
```
Session ID: a1b2c3d4-e5f6-7890-abcd-ef1234567890
```

---

## Human Review Interface

The workflow pauses after every successful IG execution for human review. The experience differs by entry point:

| Entry point | Review experience |
|-------------|-------------------|
| **Streamlit UI** | Review card with attribution bar chart, Before/After/Δ table, model-prediction table, and Approve / Reject controls |
| **Gradio UI** | Dark-panel review block with plan summary and execution summary; Approve / Reject buttons with guided reject form |
| **CLI** | Plain-text summary printed to stdout; type `approve` or `reject` at the prompt |

### CLI review prompt

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
  Feature                         Attribution      Before       After           Δ
  ────────────────────────────────────────────────────────────────────────────────
  WALA                             +0.0023      120.300     121.300      +1.000
  WAC                              -0.0018        5.500       5.520      +0.020
  loan_age                         +0.0011       36.200      37.200      +1.000
  ...

Model SMM Predictions:
  CUSIP          Period                   T0 SMM    T1 SMM     Δ SMM
  ────────────────────────────────────────────────────────────────────
  ALL CUSIPs (mean)                    0.012300  0.014500  +0.002200  ← 1 CUSIPs
  3140GXPJ8      2024-01->2024-02      0.012300  0.014500  +0.002200

Your decision >
```

**Approve** — proceeds to `reporter` (plots + LLM narrative + PDF).

**Reject** — guided prompts collect structured feedback:

```
Your decision > reject
What is wrong? > wrong months selected
Suggested change? > use Feb→Mar instead of Jan→Feb
Focus CUSIPs (comma-separated, or blank)? > 31418DSB5, 3140ABCD1
```

> **Adding CUSIPs on reject:** any CUSIPs listed in "Focus CUSIPs" are **merged** into the existing CUSIP list for the next iteration — they do not replace the original list. Leave blank to keep the current list unchanged.

The planner re-runs with your feedback and the updated CUSIP list incorporated. The loop continues until approval or `MBS_MAX_ITERATIONS` is reached.

Raw JSON is also accepted:

```
Your decision > {"decision": "reject", "what_is_wrong": "wrong months", "suggested_change": "use Feb→Mar", "focus_cusips": ["31418DSB5"]}
```

---

## Output Files

After a successful run, `MBS_OUTPUT_DIR` (default `./mbs_output`) contains:

```
mbs_output/
├── output.json                          # raw IG results (from executor subprocess)
├── attribution_aggregated.png           # mean attribution across ALL CUSIPs (first in PDF)
├── attribution_<CUSIP>.png              # bar chart per CUSIP (cusip_attribution mode)
├── scenario_comparison_heatmap.png      # feature × scenario heatmap (scenario_comparison mode)
├── report.md                            # LLM-written markdown narrative
└── report.pdf                           # self-contained PDF: narrative + all plots embedded
```

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
        "t0":    { "<feature>": 120.3 },
        "t1":    { "<feature>": 121.3 },
        "delta": { "<feature>": 1.0   }
      }
    }
  },
  "model_predictions": {
    "<CUSIP>": {
      "2024-01->2024-02": {
        "t0_smm": 0.0123,  "t1_smm": 0.0145,  "delta_smm": 0.0022,
        "t0_cpr": 0.1380,  "t1_cpr": 0.1620,  "delta_cpr": 0.0240
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

For `scenario_comparison`, `feature_values_original` uses `"base"`, `"scenario"`, `"delta"` sub-keys instead of `"t0"`, `"t1"`, `"delta"`, and `model_predictions` uses `"base_smm"`, `"scenario_smm"`, `"delta_smm"`.

### Plots

**`attribution_aggregated.png`** — mean across all CUSIPs (always generated first)

- X-axis: mean IG attribution (CPR/SMM contribution per feature)
- Y-axis: `"WALA  (120.3 → 121.3, Δ+1.0)"` — mean original-scale feature change
- Features sorted by `|mean attribution|` descending
- Provides a portfolio-level view regardless of how many CUSIPs were analyzed

**`attribution_<CUSIP>.png`** — one chart per CUSIP

- Same axes as the aggregated chart but for a single CUSIP
- Generated for every CUSIP in the analysis (no cap)

**`scenario_comparison_heatmap.png`**

- Rows: top features by maximum absolute attribution across scenarios, sorted by `|mean attribution|`
- Columns: stress scenario names
- Y-axis labels: `"WALA  (base: 120.3 | scen: 125.1, Δ+4.8)"`

### PDF report (`report.pdf`)

The PDF is fully self-contained — all PNG plots are base64-encoded and embedded, so no external file references are needed. It contains:

1. The LLM-written markdown narrative (converted to styled HTML)
2. An "Attribution Plots" section with all charts appended (aggregated chart first)

---

## Workflow Internals

### State flow

```
app.py sets initial_state (question, session_id, cusip_list=[])
    │
    ▼ question_parser → sets question_type, cusip_list, scenario_params
    │
    ▼ planner → reads feature names from parquet header (zero rows, no data load)
    │           reads checkpoint filenames from model_checkpoint_dir
    │           merges focus_cusips from human_feedback into cusip_list (on reject loops)
    │           persists merged cusip_list back to state
    │           LLM fills in comparison_months / scenario_params
    │           → sets analysis_plan, cusip_list
    │
    ▼ code_generator → renders Jinja template (no LLM on first pass)
    │                  LLM only called if code_validator returns errors
    │                  passes absolute output_dir path so subprocess writes correctly
    │                  → sets generated_code
    │
    ▼ code_validator → ast.parse + forbidden pattern regex + output.json contract
    │                  (negative lookbehind allows model.eval() while blocking bare eval())
    │                  → sets code_valid, validation_errors
    │
    ▼ executor → subprocess(generated_code, timeout=N seconds)
    │            script writes t0/t1/delta feature values and SMM/CPR model predictions
    │            reads output.json from absolute MBS_OUTPUT_DIR path
    │            → sets execution_result or execution_error
    │
    ▼ human_reviewer → interrupt() surfaces review payload to app.py
    │                  shows: top features ranked by |attribution| with Before/After/Δ columns
    │                          per-CUSIP SMM predictions with ALL CUSIPs aggregate row
    │                  app.py collects CLI input, resumes with Command(resume=feedback)
    │                  → sets human_feedback
    │
    ▼ reporter → _build_feature_change_table(): sorted by |mean_attr|, includes Rank column
    │            _build_prediction_summary(): aggregate row + per-CUSIP rows
    │            _plot_cusip_attribution(): one chart per CUSIP + aggregated mean chart
    │            LLM writes markdown narrative (compact prompt — no raw attributions JSON)
    │            _generate_pdf(): markdown → HTML → PDF with base64-embedded plots
    │            → sets report_markdown, plot_paths, pdf_path
```

### Prompt design

All LLM prompts live in `nodes/prompts.yaml`. The `T(".prompts:key.subkey")` call resolves relative to the **calling file's directory** — since all node files are in `nodes/`, they correctly find `nodes/prompts.yaml`.

The reporter LLM receives pre-processed, compact inputs rather than raw JSON blobs:
- **`feature_change_table`** — pre-formatted table sorted by `|mean attribution|`, with Rank, Before, After, and Δ columns
- **`prediction_summary`** — pre-formatted table with an "ALL CUSIPs (mean)" aggregate row at top
- **`summary_stats`** — per-feature mean ± std attribution

The raw `attributions_normalized` dict is intentionally excluded from the LLM prompt to avoid token limit failures when many CUSIPs are analyzed.

### Code generation strategy

On first pass, `code_generator` renders a **deterministic Jinja template** — fast, reproducible, no LLM cost. The LLM is invoked *only* if `code_validator` reports errors. This makes the happy path entirely template-driven.

### Feature name and checkpoint discovery

`planner_node` reads the parquet schema (zero rows — no full file load) and scans `model_checkpoint_dir` for `.pth`/`.pt` files at runtime before calling the LLM. The prompt instructs the LLM to use these exact values, preventing hallucinated feature names and checkpoint paths.

### CUSIP accumulation across iterations

On each reject loop, `planner_node` merges `human_feedback.focus_cusips` into `state["cusip_list"]` (preserving order, no duplicates) and writes the merged list back to state. This means each iteration accumulates CUSIPs rather than resetting to the original question's list.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `Cannot find .prompts:question_parser.system` | Old separate `prompts/*.yaml` files | Prompts must be in `nodes/prompts.yaml` |
| `Forbidden pattern 'eval()' found` | `model.eval()` caught by validator | Already fixed — validator uses negative lookbehind `(?<!\.)\beval\s*\(` |
| `No data for CUSIP=X, month=2024-01` | Integer date column not parsed | Confirm `MBS_DATE_FORMAT=%Y%m%d`; column should contain integers like `20240101` |
| Feature names wrong or too short | LLM guessed instead of reading parquet | Planner auto-reads parquet columns; ensure `MBS_DATA_FILE` is accessible at run time |
| Checkpoint set to `latest_checkpoint.pt` | Old default or LLM guess | Place model in `MBS_MODEL_CHECKPOINT_DIR`; planner scans for `.pth`/`.pt` and injects exact paths |
| Human review not pausing | `interrupt_before` conflict (old version) | Fixed — `interrupt()` inside `human_reviewer_node` is the sole pause mechanism |
| Feature values show "—" in human review | Old `output.json` from before t0/t1/delta changes | Re-run the workflow to regenerate `output.json` with the new template |
| Second iteration ignores new CUSIPs | Old version without CUSIP merge | Fixed — `focus_cusips` from reject feedback is merged into `state["cusip_list"]` |
| PDF generation fails / skipped | WeasyPrint system libraries missing | `apt-get install libpango-1.0-0 libpangocairo-1.0-0`; markdown + report.md still written |
| LLM call fails after 10 retries | Prompt too large (many CUSIPs) | Fixed — raw `attributions_normalized` JSON removed from prompt; only compact summary tables sent |
| Feature importance ranking wrong | Old version sorted by \|Δ feature\| | Fixed — now sorted by \|mean attribution\| from `summary_stats` |
| Aggregated report mentions only one CUSIP | Old version had no aggregate prediction row | Fixed — "ALL CUSIPs (mean)" row added to prediction summary table |
| OOM during IG | Batch size too large | Reduce `MBS_IG_BATCH_SIZE=8` or lower |
| Slow / inaccurate IG | Too few approximation steps | Increase `MBS_IG_N_STEPS=100` or `200` |
| `FileNotFoundError: scaler.sav` | Wrong scaler path | Set `MBS_SCALER_PATH=/absolute/path/to/scaler.sav` |
