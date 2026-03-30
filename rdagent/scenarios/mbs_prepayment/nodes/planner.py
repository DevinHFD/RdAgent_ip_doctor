"""
planner node — generate a structured AnalysisPlan via LLM.

On reject loops, incorporates human feedback into the prompt so the revised
plan directly addresses the reviewer's concerns.
"""

from rdagent.log import rdagent_logger as logger
from rdagent.utils.agent.tpl import T
from rdagent.utils.agent.workflow import build_cls_from_json_with_retry

from ..conf import MBS_SETTINGS
from ..state import AnalysisPlan, HumanFeedback, MBSAnalysisState


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

    sys_prompt = T(".prompts:planner.system").r(
        question_type=state["question_type"],
        ig_baseline_strategy=MBS_SETTINGS.ig_baseline_strategy,
        ig_n_steps=MBS_SETTINGS.ig_n_steps,
        ig_target_output=MBS_SETTINGS.ig_target_output,
        model_checkpoint_dir=str(MBS_SETTINGS.model_checkpoint_dir),
        data_file=str(MBS_SETTINGS.data_file),
    )
    user_prompt = T(".prompts:planner.user").r(
        question=state["question"],
        cusip_list=state["cusip_list"],
        scenario_params=state["scenario_params"],
        human_feedback=human_fb,
        iteration=iteration,
        model_checkpoint_dir=str(MBS_SETTINGS.model_checkpoint_dir),
        data_file=str(MBS_SETTINGS.data_file),
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
