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


def _avg_feat_group(nested_orig: dict, group: str) -> dict[str, float]:
    """
    Average one sub-group (e.g. "t0", "t1", "delta", "base", "scenario")
    across all CUSIPs and all valid period_keys.
    nested_orig = {cusip: {period_key: {group: {feat: val}}}}
    """
    from collections import defaultdict
    totals: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for cusip_data in nested_orig.values():
        for period_data in cusip_data.values():
            if not isinstance(period_data, dict) or "error" in period_data:
                continue
            for feat, val in period_data.get(group, {}).items():
                if isinstance(val, (int, float)):
                    totals[feat] += val
                    counts[feat] += 1
    return {feat: totals[feat] / counts[feat] for feat in totals if counts[feat] > 0}


def _format_execution_summary(result: ExecutionResult) -> str:
    """
    Show top-5 features by mean |attribution| (SMM/CPR contribution).
    For cusip_attribution: also show t0 → t1 → Δ original-scale feature change.
    For scenario_comparison: show base | scenario | Δ.
    """
    stats = result.summary_stats
    ranked = sorted(stats.items(), key=lambda kv: abs(kv[1].get("mean_attr", 0)), reverse=True)

    analysis_type = result.analysis_type
    feat_orig = result.feature_values_original

    if analysis_type == "cusip_attribution":
        t0 = _avg_feat_group(feat_orig, "t0")
        t1 = _avg_feat_group(feat_orig, "t1")
        delta = _avg_feat_group(feat_orig, "delta")
    else:
        t0 = _avg_feat_group(feat_orig, "base")
        t1 = _avg_feat_group(feat_orig, "scenario")
        delta = _avg_feat_group(feat_orig, "delta")

    lines = ["Top features by mean |attribution| (SMM/CPR contribution):"]
    lines.append(
        f"  {'Feature':<28}  {'Attribution':>14}  {'Before':>10}  {'After':>10}  {'Δ':>10}"
    )
    lines.append("  " + "-" * 80)
    for feat, s in ranked[:5]:
        mean_val = s.get("mean_attr", 0)
        t0_val = t0.get(feat)
        t1_val = t1.get(feat)
        d_val = delta.get(feat)
        t0_str = f"{t0_val:.3f}" if t0_val is not None else "—"
        t1_str = f"{t1_val:.3f}" if t1_val is not None else "—"
        d_str = f"{d_val:+.3f}" if d_val is not None else "—"
        lines.append(
            f"  {feat:<28}  {mean_val:>+14.4f}  {t0_str:>10}  {t1_str:>10}  {d_str:>10}"
        )
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
