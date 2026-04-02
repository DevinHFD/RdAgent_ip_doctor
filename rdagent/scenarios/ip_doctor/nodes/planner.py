"""
planner node — generate a structured AnalysisPlan via LLM.

On reject loops, incorporates human feedback into the prompt so the revised
plan directly addresses the reviewer's concerns.
"""

from pathlib import Path

import pandas as pd

from rdagent.log import rdagent_logger as logger
from rdagent.utils.agent.tpl import T
from rdagent.utils.agent.workflow import build_cls_from_json_with_retry

from ..conf import MBS_SETTINGS
from ..state import AnalysisPlan, HumanFeedback, MBSAnalysisState


def _discover_checkpoints() -> list[str]:
    """Return all .pth/.pt filenames found in model_checkpoint_dir."""
    ckpt_dir = MBS_SETTINGS.model_checkpoint_dir
    if not ckpt_dir.exists():
        return []
    return sorted(
        str(p) for p in ckpt_dir.iterdir() if p.suffix in (".pth", ".pt")
    )


def _read_feature_names() -> list[str]:
    """
    Read the parquet header and return feature columns — everything except
    cusip_col and date_col.  Returns an empty list if the file is missing.
    """
    data_file = MBS_SETTINGS.data_file
    if not data_file.exists():
        logger.warning(f"data_file not found for feature discovery: {data_file}")
        return []
    # Read only the schema (zero rows) to avoid loading the full dataset
    df = pd.read_parquet(data_file).iloc[:0]
    exclude = {MBS_SETTINGS.cusip_col, MBS_SETTINGS.date_col}
    return [c for c in df.columns if c not in exclude]


def planner_node(state: MBSAnalysisState) -> dict:
    """
    LLM call producing a validated AnalysisPlan.
    Returns partial state update, resetting all downstream fields.
    """
    iteration = state.get("iteration_count", 0)
    human_fb_dict = state.get("human_feedback")
    human_fb = HumanFeedback.model_validate(human_fb_dict) if human_fb_dict else None

    if human_fb:
        logger.info(f"Re-planning after reject (iteration {iteration}). Feedback: {human_fb}")
    else:
        logger.info(f"Generating analysis plan (iteration {iteration}).")

    # Discover ground-truth values from disk so the LLM doesn't have to guess
    available_checkpoints = _discover_checkpoints()
    feature_names = _read_feature_names()
    logger.info(f"Discovered checkpoints: {available_checkpoints}")
    logger.info(f"Discovered {len(feature_names)} feature columns from parquet.")

    # Merge focus_cusips from human feedback into the authoritative CUSIP list.
    # Preserve order: original CUSIPs first, then any newly requested ones.
    current_cusips: list[str] = list(state.get("cusip_list") or [])
    if human_fb and human_fb.focus_cusips:
        existing = set(current_cusips)
        for c in human_fb.focus_cusips:
            if c not in existing:
                current_cusips.append(c)
                existing.add(c)
        logger.info(f"Merged focus_cusips into CUSIP list: {current_cusips}")

    sys_prompt = T(".prompts:planner.system").r(
        question_type=state["question_type"],
        ig_baseline_strategy=MBS_SETTINGS.ig_baseline_strategy,
        ig_n_steps=MBS_SETTINGS.ig_n_steps,
        ig_target_output=MBS_SETTINGS.ig_target_output,
        model_checkpoint_dir=str(MBS_SETTINGS.model_checkpoint_dir),
        data_file=str(MBS_SETTINGS.data_file),
        available_checkpoints=available_checkpoints,
        feature_names=feature_names,
    )
    user_prompt = T(".prompts:planner.user").r(
        question=state["question"],
        cusip_list=current_cusips,
        scenario_params=state["scenario_params"],
        human_feedback=human_fb,
        iteration=iteration,
        model_checkpoint_dir=str(MBS_SETTINGS.model_checkpoint_dir),
        data_file=str(MBS_SETTINGS.data_file),
        available_checkpoints=available_checkpoints,
        feature_names=feature_names,
    )

    plan: AnalysisPlan = build_cls_from_json_with_retry(
        cls=AnalysisPlan,
        system_prompt=sys_prompt,
        user_prompt=user_prompt,
        retry_n=5,
    )

    logger.info(
        f"Plan generated: type={plan.analysis_type}, "
        f"CUSIPs={plan.data_spec.cusip_list}, "
        f"rationale={plan.rationale[:100]}"
    )

    return {
        "analysis_plan": plan.model_dump(),
        "cusip_list": current_cusips,  # persist merged list for future iterations
        "iteration_count": iteration + 1,
        # Reset all downstream state for the new plan
        "generated_code": "",
        "code_valid": False,
        "validation_errors": [],
        "execution_result": None,
        "execution_error": None,
        "debug_attempts": 0,
        "human_feedback": None,  # consumed; cleared so next planner call starts fresh
    }
