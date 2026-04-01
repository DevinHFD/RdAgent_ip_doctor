"""
reporter node — generate matplotlib/seaborn plots and an LLM narrative markdown report.

Attribution values (bar x-axis): IG output = SMM/CPR contribution per feature.
Y-axis labels show the original-scale feature change between the two periods:
  cusip_attribution  : "WALA  (120.3 → 121.3, Δ+1.0)"
  scenario_comparison: "WALA  (base: 120.3 | shock: 125.1, Δ+4.8)"
"""

from collections import defaultdict
from pathlib import Path

from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import APIBackend
from rdagent.utils.agent.tpl import T

from ..conf import MBS_SETTINGS
from ..state import ExecutionResult, MBSAnalysisState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _avg_attributions(nested: dict) -> dict[str, float]:
    """
    Average {period_key: {feature: value}} across all valid periods.
    Skips periods with an "error" key.
    """
    totals: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for period_data in nested.values():
        if not isinstance(period_data, dict) or "error" in period_data:
            continue
        for feat, val in period_data.items():
            if isinstance(val, (int, float)):
                totals[feat] += val
                counts[feat] += 1
    return {feat: totals[feat] / counts[feat] for feat in totals if counts[feat] > 0}


def _avg_feat_group(nested_orig: dict, group: str) -> dict[str, float]:
    """
    Average one sub-group (e.g. "t0", "t1", "delta", "base", "scenario")
    across all valid periods.
    nested_orig = {period_key: {"t0": {feat: val}, "t1": ..., "delta": ...}}
    """
    totals: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for period_data in nested_orig.values():
        if not isinstance(period_data, dict) or "error" in period_data:
            continue
        group_data = period_data.get(group, {})
        for feat, val in group_data.items():
            if isinstance(val, (int, float)):
                totals[feat] += val
                counts[feat] += 1
    return {feat: totals[feat] / counts[feat] for feat in totals if counts[feat] > 0}


def _make_labels_cusip(features: list[str],
                       t0_means: dict[str, float],
                       t1_means: dict[str, float],
                       delta_means: dict[str, float]) -> list[str]:
    """
    Build y-axis labels showing the feature change between months:
      "WALA  (120.3 → 121.3, Δ+1.0)"
    """
    labels = []
    for feat in features:
        t0 = t0_means.get(feat)
        t1 = t1_means.get(feat)
        d  = delta_means.get(feat)
        if t0 is not None and t1 is not None and d is not None:
            labels.append(f"{feat}  ({t0:.2f} → {t1:.2f}, Δ{d:+.2f})")
        elif t1 is not None:
            labels.append(f"{feat}  (avg: {t1:.2f})")
        else:
            labels.append(feat)
    return labels


def _make_labels_scenario(features: list[str],
                          base_means: dict[str, float],
                          scenario_means: dict[str, float],
                          delta_means: dict[str, float]) -> list[str]:
    """
    Build y-axis labels showing base vs scenario feature levels:
      "WALA  (base: 120.3 | shock: 125.1, Δ+4.8)"
    """
    labels = []
    for feat in features:
        base = base_means.get(feat)
        scen = scenario_means.get(feat)
        d    = delta_means.get(feat)
        if base is not None and scen is not None and d is not None:
            labels.append(f"{feat}  (base: {base:.2f} | scen: {scen:.2f}, Δ{d:+.2f})")
        elif scen is not None:
            labels.append(f"{feat}  (avg: {scen:.2f})")
        else:
            labels.append(feat)
    return labels


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def _plot_cusip_attribution(result: ExecutionResult, output_dir: Path) -> list[str]:
    """
    One horizontal bar chart per CUSIP (up to 10).
    X-axis : IG attribution (SMM/CPR contribution).
    Y-axis : "WALA  (120.3 → 121.3, Δ+1.0)"  — original-scale change between months.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    target_output = result.metadata.get("target_output", "output")
    paths = []
    cusips = list(result.attributions_normalized.keys())[:10]

    for cusip in cusips:
        attr_mean  = _avg_attributions(result.attributions_normalized.get(cusip, {}))
        orig_data  = result.feature_values_original.get(cusip, {})
        t0_means   = _avg_feat_group(orig_data, "t0")
        t1_means   = _avg_feat_group(orig_data, "t1")
        delta_means = _avg_feat_group(orig_data, "delta")

        if not attr_mean:
            logger.warning(f"No valid attribution data for CUSIP {cusip}; skipping plot.")
            continue

        sorted_items = sorted(attr_mean.items(), key=lambda kv: abs(kv[1]), reverse=True)[:15]
        features = [item[0] for item in sorted_items]
        values   = [item[1] for item in sorted_items]
        colors   = ["#d62728" if v > 0 else "#1f77b4" for v in values]
        labels   = _make_labels_cusip(features, t0_means, t1_means, delta_means)

        fig, ax = plt.subplots(figsize=(13, max(4, len(features) * 0.55)))
        ax.barh(labels[::-1], values[::-1], color=colors[::-1])
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel(f"IG Attribution  ({target_output.upper()} contribution)")
        ax.set_title(f"CUSIP {cusip} — Feature Attributions\n"
                     f"(y-axis shows original-scale feature change between months)")
        ax.tick_params(axis="y", labelsize=8)
        fig.tight_layout()

        path = output_dir / f"attribution_{cusip}.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        paths.append(str(path))
        logger.info(f"Saved attribution chart: {path}")

    return paths


def _plot_scenario_comparison(result: ExecutionResult, output_dir: Path) -> list[str]:
    """
    Seaborn heatmap: features (rows) × scenarios (columns), averaged across CUSIPs.
    Y-axis labels show "WALA  (base: 120.3 | scen: 125.1, Δ+4.8)".
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    target_output = result.metadata.get("target_output", "output")

    # Aggregate IG attributions: scenario -> feature -> [values across CUSIPs]
    scenario_feat_vals: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    # Aggregate original-scale values by sub-group
    base_feat_vals:     dict[str, list] = defaultdict(list)
    scen_feat_vals:     dict[str, list] = defaultdict(list)
    delta_feat_vals:    dict[str, list] = defaultdict(list)

    for cusip, cusip_data in result.attributions_normalized.items():
        for scenario, feat_dict in cusip_data.items():
            if not isinstance(feat_dict, dict) or "error" in feat_dict:
                continue
            for feat, val in feat_dict.items():
                scenario_feat_vals[scenario][feat].append(val)

        for scenario, orig_dict in result.feature_values_original.get(cusip, {}).items():
            if not isinstance(orig_dict, dict) or "error" in orig_dict:
                continue
            for feat, val in orig_dict.get("base", {}).items():
                if isinstance(val, (int, float)):
                    base_feat_vals[feat].append(val)
            for feat, val in orig_dict.get("scenario", {}).items():
                if isinstance(val, (int, float)):
                    scen_feat_vals[feat].append(val)
            for feat, val in orig_dict.get("delta", {}).items():
                if isinstance(val, (int, float)):
                    delta_feat_vals[feat].append(val)

    scenarios = list(scenario_feat_vals.keys())
    all_features = list(dict.fromkeys(
        feat for s_data in scenario_feat_vals.values() for feat in s_data
    ))

    if not scenarios or not all_features:
        logger.warning("No valid scenario attribution data for heatmap.")
        return []

    data = {
        scenario: {
            feat: (sum(vals) / len(vals) if vals else 0.0)
            for feat, vals in scenario_feat_vals[scenario].items()
        }
        for scenario in scenarios
    }
    df = pd.DataFrame(data, index=all_features).fillna(0.0)
    df = df.loc[df.abs().max(axis=1).sort_values(ascending=False).index]
    df = df.iloc[:20]

    base_means  = {feat: sum(v)/len(v) for feat, v in base_feat_vals.items()  if v}
    scen_means  = {feat: sum(v)/len(v) for feat, v in scen_feat_vals.items()  if v}
    delta_means = {feat: sum(v)/len(v) for feat, v in delta_feat_vals.items() if v}
    y_labels = _make_labels_scenario(list(df.index), base_means, scen_means, delta_means)
    df.index = y_labels

    fig, ax = plt.subplots(figsize=(max(8, len(scenarios) * 2), max(6, len(df) * 0.55)))
    sns.heatmap(
        df, ax=ax, center=0, cmap="RdBu_r", linewidths=0.3,
        annot=len(scenarios) <= 6, fmt=".4f",
        cbar_kws={"label": f"IG Attribution ({target_output.upper()} contribution)"},
    )
    ax.set_title("Scenario Comparison — Feature Attributions (avg across CUSIPs)\n"
                 "(y-axis: base | scenario | Δ in original scale)")
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Feature  (base | scenario | Δ original scale)")
    ax.tick_params(axis="x", rotation=30, labelsize=9)
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()

    path = output_dir / "scenario_comparison_heatmap.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved scenario comparison heatmap: {path}")
    return [str(path)]


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


def reporter_node(state: MBSAnalysisState) -> dict:
    output_dir = MBS_SETTINGS.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    exec_result_dict = state.get("execution_result")
    if not exec_result_dict:
        logger.warning("reporter_node called with no execution_result; skipping plots.")
        return {"report_markdown": "No execution result available.", "plot_paths": []}

    exec_result = ExecutionResult.model_validate(exec_result_dict)
    analysis_type = exec_result.analysis_type

    logger.info(f"Generating plots for analysis_type={analysis_type}")
    if analysis_type == "cusip_attribution":
        plot_paths = _plot_cusip_attribution(exec_result, output_dir)
    else:
        plot_paths = _plot_scenario_comparison(exec_result, output_dir)

    logger.info("Generating LLM narrative report.")
    sys_prompt = T(".prompts:reporter.system").r()
    user_prompt = T(".prompts:reporter.user").r(
        question=state["question"],
        analysis_type=analysis_type,
        attributions_normalized=exec_result.attributions_normalized,
        feature_values_original=exec_result.feature_values_original,
        summary_stats=exec_result.summary_stats,
        metadata=exec_result.metadata,
        plot_paths=plot_paths,
    )

    report_markdown = APIBackend().build_messages_and_create_chat_completion(
        user_prompt=user_prompt,
        system_prompt=sys_prompt,
    )

    logger.info(f"Report generated ({len(report_markdown)} chars). Plots: {plot_paths}")
    return {"report_markdown": report_markdown, "plot_paths": plot_paths}
