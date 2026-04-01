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


_KNOWN_SUBKEYS = {"t0", "t1", "delta", "base", "scenario", "error"}


def _avg_feat_group(nested_orig: dict, group: str) -> dict[str, float]:
    """
    Average one sub-group (e.g. "t0", "t1", "delta", "base", "scenario")
    across all CUSIPs and all valid period_keys.
    nested_orig = {cusip: {period_key: {group: {feat: val}}}}

    Falls back gracefully when period_data uses the old flat format
    {feat: val} (no sub-group keys). In that case only "t1" / "scenario"
    groups return data; "t0" / "base" / "delta" return {}.
    """
    from collections import defaultdict
    totals: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for cusip_data in nested_orig.values():
        for period_data in cusip_data.values():
            if not isinstance(period_data, dict) or "error" in period_data:
                continue
            # Detect old flat format: no recognised sub-group key present
            if not any(k in period_data for k in _KNOWN_SUBKEYS):
                if group in ("t1", "scenario"):
                    group_data = period_data
                else:
                    group_data = {}
            else:
                group_data = period_data.get(group, {})
            for feat, val in group_data.items():
                if isinstance(val, (int, float)):
                    totals[feat] += val
                    counts[feat] += 1
    return {feat: totals[feat] / counts[feat] for feat in totals if counts[feat] > 0}


def _format_prediction_summary(result: ExecutionResult) -> str:
    """Show model SMM/CPR predictions per CUSIP and period."""
    preds = result.model_predictions
    if not preds:
        return ""

    analysis_type = result.analysis_type
    if analysis_type == "cusip_attribution":
        col_b, col_a = "T0 SMM", "T1 SMM"
        k_b, k_a, k_d = "t0_smm", "t1_smm", "delta_smm"
    else:
        col_b, col_a = "Base SMM", "Scen SMM"
        k_b, k_a, k_d = "base_smm", "scenario_smm", "delta_smm"

    lines = [
        "\nModel SMM Predictions:",
        f"  {'CUSIP':<14} {'Period/Scenario':<24} {col_b:>10} {col_a:>10} {'Δ SMM':>10}",
        "  " + "-" * 72,
    ]
    for cusip, cusip_data in preds.items():
        for period, vals in cusip_data.items():
            if not isinstance(vals, dict):
                continue
            b = vals.get(k_b)
            a = vals.get(k_a)
            d = vals.get(k_d)
            b_s = f"{b:>10.6f}" if b is not None else f"{'—':>10}"
            a_s = f"{a:>10.6f}" if a is not None else f"{'—':>10}"
            d_s = f"{d:>+10.6f}" if d is not None else f"{'—':>10}"
            lines.append(f"  {cusip:<14} {period:<24} {b_s} {a_s} {d_s}")
    return "\n".join(lines)


def _format_execution_summary(result: ExecutionResult) -> str:
    """
    Show top-5 features by mean |attribution| (SMM/CPR contribution),
    model SMM predictions per CUSIP/period, and original-scale feature changes.
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

    # Append model predictions
    pred_text = _format_prediction_summary(result)
    if pred_text:
        lines.append(pred_text)

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
