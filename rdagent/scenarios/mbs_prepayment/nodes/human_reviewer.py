"""
human_reviewer node — CLI human-in-the-loop review via LangGraph interrupt().

When this node executes, LangGraph suspends the graph and surfaces a review
payload to the caller (app.py). The caller shows it to the human, collects
structured feedback, then resumes with:
    graph.invoke(Command(resume=feedback_dict), config=config)
"""

import json

from langgraph.types import interrupt

from rdagent.log import rdagent_logger as logger

from ..state import ExecutionResult, HumanFeedback, MBSAnalysisState


def _format_plan_summary(plan: dict) -> str:
    data_spec = plan.get("data_spec", {})
    ig_params = plan.get("ig_params", {})
    return (
        f"Analysis type : {plan.get('analysis_type')}\n"
        f"CUSIPs        : {data_spec.get('cusip_list', [])}\n"
        f"Baseline      : {ig_params.get('baseline_strategy')}  "
        f"n_steps={ig_params.get('n_steps')}  "
        f"target={ig_params.get('target_output')}\n"
        f"Rationale     : {plan.get('rationale', '')}"
    )


def _format_execution_summary(result: ExecutionResult) -> str:
    """Show top-5 features by mean |attribution| in original scale."""
    stats = result.summary_stats
    ranked = sorted(stats.items(), key=lambda kv: abs(kv[1].get("mean_attr_original", 0)), reverse=True)
    lines = ["Top features by mean |attribution| (original scale):"]
    for feat, s in ranked[:5]:
        mean_val = s.get("mean_attr_original", 0)
        std_val = s.get("std_attr_original", 0)
        lines.append(f"  {feat:30s}  mean={mean_val:+.4f}  std={std_val:.4f}")
    return "\n".join(lines)


def human_reviewer_node(state: MBSAnalysisState) -> dict:
    """
    Suspend graph execution for human review.

    The interrupt payload is shown to the human via app.py.
    The graph resumes when the caller calls:
        graph.invoke(Command(resume={"decision": "approve"|"reject", ...}), config=config)
    """
    plan = state.get("analysis_plan") or {}
    exec_result_dict = state.get("execution_result")
    error_note = state.get("execution_error")

    plan_summary = _format_plan_summary(plan)

    if exec_result_dict:
        exec_result = ExecutionResult.model_validate(exec_result_dict)
        execution_summary = _format_execution_summary(exec_result)
    else:
        execution_summary = "No successful execution result available."

    review_payload = {
        "question": state["question"],
        "iteration": state.get("iteration_count", 1),
        "plan_summary": plan_summary,
        "execution_summary": execution_summary,
        "error_note": error_note,
        "instructions": (
            "Respond with a JSON object:\n"
            '  {"decision": "approve"}            — accept results, proceed to report\n'
            '  {"decision": "reject",\n'
            '   "what_is_wrong": "...",\n'
            '   "suggested_change": "...",\n'
            '   "focus_cusips": [...]}             — reject, loop back to planner\n'
            "\nShorthand accepted: just type  approve  or  reject"
        ),
    }

    logger.info("Suspending graph for human review.")
    raw_feedback = interrupt(review_payload)

    # Parse resumed input
    try:
        if isinstance(raw_feedback, str):
            raw_feedback = raw_feedback.strip()
            if raw_feedback.lower() in ("approve", "reject"):
                feedback_dict = {"decision": raw_feedback.lower()}
            else:
                feedback_dict = json.loads(raw_feedback)
        else:
            feedback_dict = raw_feedback

        feedback = HumanFeedback.model_validate(feedback_dict)
    except Exception as e:
        logger.warning(f"Could not parse human feedback ({e}); defaulting to approve.")
        feedback = HumanFeedback(decision="approve")

    logger.info(f"Human decision: {feedback.decision}")
    return {"human_feedback": feedback.model_dump()}
