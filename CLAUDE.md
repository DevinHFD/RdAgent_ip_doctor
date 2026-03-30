




# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RD-Agent is a Microsoft research framework that automates data-driven R&D using LLM agents. It evolves machine learning models, financial factors, and research hypotheses through iterative proposal → implementation → evaluation loops. Key scenarios: quantitative finance (via Qlib), Kaggle competitions, general model research from papers, and LLM fine-tuning.

## Commands

### Install / Setup
```bash
make dev                  # Full dev setup (editable install + all extras + pre-commit)
make install              # Editable install only
make dev-<extra>          # Install specific extras: docs, lint, package, test, torch
```

### Testing
```bash
make test                 # Full test suite with coverage (20% threshold in CI)
make test-offline         # Only tests marked @pytest.mark.offline (no external APIs)
pytest test/oai/test_completion.py                        # Single file
pytest test/oai/test_completion.py::test_function_name   # Single test
pytest -m offline                                         # Offline marker filter
```

### Linting
```bash
make lint                 # Run all linters (mypy, ruff, isort, black, toml-sort)
make auto-lint            # Auto-fix all linters
make pre-commit           # Run all pre-commit hooks
```

Individual linters: `make black`, `make isort`, `make mypy`, `make ruff`
Auto-fix variants: `make auto-black`, `make auto-isort`

**Code style:** Line length 120, conventional commits for PRs.

### Running Scenarios
```bash
rdagent health_check                          # Validate environment and config
rdagent ui --port 19899 --log-dir <path>     # Streamlit log visualization
rdagent server_ui                             # Flask backend for web UI
```

### Frontend (web/)
```bash
npm run build:flask   # Build Vue app into Flask static folder
```

### Docs
```bash
make docs             # Build full Sphinx documentation
make docs-autobuild   # Auto-rebuild on changes
```

## Architecture

### Core Framework (`rdagent/core/`)

The central abstraction is an **evolving R&D loop**:

1. **Proposal** (`proposal.py`) — LLM generates hypotheses/ideas
2. **Development** (`developer.py`) — Implements the hypothesis as code (Task → Workspace)
3. **Evaluation** (`evaluation.py`) — Runs experiments, collects feedback
4. **Knowledge Update** (`knowledge_base.py`) — Stores results for future RAG

Key abstractions:
- `Scenario` (`scenario.py`) — Base class all scenarios implement
- `EvolvingStrategy` / `EvolvableSubjects` (`evolving_framework.py`) — Strategy pattern for iterative evolution
- `Task` / `Workspace` / `RunningInfo` (`experiment.py`) — Task lifecycle management
- `RAGStrategy` — Retrieval-Augmented Generation for knowledge reuse

### Applications (`rdagent/app/`)

CLI entry points (via `rdagent/app/cli.py`, typer-based). Each subdirectory is a runnable scenario:
- `qlib_rd_loop/` — Financial factor/model evolution using Qlib
- `data_science/` — Kaggle and general data science
- `general_model/` — Extract and implement models from papers
- `finetune/` — LLM fine-tuning

### Components (`rdagent/components/`)

Reusable building blocks used across scenarios:
- `coder/` — Code generation (CoSTEER framework) and execution
- `proposal/` — Hypothesis generation mechanisms
- `runner/` — Code execution environments (including Docker)
- `knowledge_management/` — Storage and retrieval

### Scenarios (`rdagent/scenarios/`)

Concrete implementations binding core + components together:
- `qlib/` — Financial quantitative trading (requires Qlib + Python 3.8 env via `make init-qlib-env`)
- `data_science/` — Kaggle / general ML competitions
- `general_model/` — General model research
- `finetune/` — Fine-tuning workflows

### LLM Backend (`rdagent/oai/`)

Wraps LiteLLM (primary), OpenAI, and Azure. Configured via `.env` (see `.env.example`):
- `CHAT_MODEL`, `EMBEDDING_MODEL` — model identifiers
- `USE_CHAT_CACHE` — enable response caching
- `MAX_RETRY`, `RETRY_WAIT_SECONDS` — retry behavior

### Logging & UI (`rdagent/log/`)

- `ui/app.py`, `ui/dsapp.py` — Streamlit-based trace viewers
- `server/` — Flask backend for the Vue web UI

## Configuration

Copy `.env.example` to `.env` before running. Required fields: `CHAT_MODEL`, `EMBEDDING_MODEL`, and the relevant API key (`OPENAI_API_KEY` or Azure equivalents).

## Type Checking Scope

Strict mypy is currently enforced only on `rdagent/core/`. Ruff is also scoped to `rdagent/core/`. Expanding these to other modules requires resolving existing violations first.

## Test Markers

- `@pytest.mark.offline` — test does not call external APIs; safe for CI
- Tests in `workspace/` are excluded from pytest collection
