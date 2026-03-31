from __future__ import annotations

from typing import Literal, TypedDict

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Pydantic support models
# These are stored as plain dicts in LangGraph state (via .model_dump()) so
# that the MemorySaver checkpointer can JSON-serialize them.
# ---------------------------------------------------------------------------


class DataLoadSpec(BaseModel):
    cusip_list: list[str]
    date_range: tuple[str, str] | None = None  # (start_month, end_month) e.g. ("2024-01", "2024-03")
    scenario_params: dict = Field(default_factory=dict)


class ModelLoadSpec(BaseModel):
    checkpoint_path: str
    feature_names: list[str]


class IGParameters(BaseModel):
    baseline_strategy: Literal["zero", "mean"] = "zero"
    n_steps: int = 50
    target_output: Literal["cpr", "smm"] = "cpr"
    batch_size: int = 32


class AnalysisPlan(BaseModel):
    analysis_type: Literal["cusip_attribution", "scenario_comparison"]
    data_spec: DataLoadSpec
    model_spec: ModelLoadSpec
    ig_params: IGParameters
    comparison_months: list[str] | None = None  # for cusip_attribution
    scenario_names: list[str] | None = None  # for scenario_comparison
    rationale: str


class HumanFeedback(BaseModel):
    decision: Literal["approve", "reject"]
    what_is_wrong: str | None = None
    suggested_change: str | None = None
    focus_cusips: list[str] | None = None


class ExecutionResult(BaseModel):
    """
    Standardized output written as output.json by every generated analysis script.

    - attributions_normalized: IG output as-is (represents SMM/CPR contribution per feature).
        Structure: CUSIP -> period_key -> {feature_name -> float}
        period_key examples:
          cusip_attribution  : "2024-01->2024-02"
          scenario_comparison: "rate_shock" (scenario name)
    - feature_values_original: feature values inverse-transformed to original scale for context
        (e.g. WALA: 120.3 months). Same nested structure as attributions_normalized.
        Used in plots and reports so readers understand what feature level caused the attribution.
    """

    analysis_type: Literal["cusip_attribution", "scenario_comparison"]
    attributions_normalized: dict
    feature_values_original: dict
    metadata: dict
    summary_stats: dict  # feature_name -> {"mean_attr": float, "std_attr": float}


# ---------------------------------------------------------------------------
# LangGraph TypedDict state
# ---------------------------------------------------------------------------


class MBSAnalysisState(TypedDict):
    # Input
    question: str
    # Parsed from question
    question_type: Literal["cusip_attribution", "scenario_comparison"]
    cusip_list: list[str]
    scenario_params: dict
    # Planning
    analysis_plan: dict | None  # AnalysisPlan.model_dump()
    # Code lifecycle
    generated_code: str
    code_valid: bool
    validation_errors: list[str]
    # Execution
    execution_result: dict | None  # ExecutionResult.model_dump()
    execution_error: str | None
    debug_attempts: int
    # Human review
    human_feedback: dict | None  # HumanFeedback.model_dump()
    # Report
    report_markdown: str | None
    plot_paths: list[str]
    # Metadata
    iteration_count: int
    session_id: str
