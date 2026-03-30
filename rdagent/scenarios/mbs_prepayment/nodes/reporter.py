"""
reporter node — generate matplotlib/seaborn plots and an LLM narrative markdown report.

All values shown to the user (axes, annotations, narrative) are in original feature scale.
Attribution and feature data are read directly from execution_result.attributions_original
and execution_result.feature_values_original — no gradient re-run required.
"""

from pathlib import Path

from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import APIBackend
from rdagent.utils.agent.tpl import T

from ..conf import MBS_SETTINGS
from ..state import ExecutionResult, MBSAnalysisState


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def _plot_cusip_attribution(result: ExecutionResult, output_dir: Path) -> list[str]:
    """
    Horizontal bar chart per CUSIP (up to 10 CUSIPs).
    X-axis: mean original-scale attribution; bars colored by sign.
    Returns list of saved PNG paths.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    cusips = list(result.attributions_original.keys())[:10]

    for cusip in cusips:
        attr_dict = result.attributions_original[cusip]
        # Sort by absolute attribution descending, show top 15 features
        sorted_items = sorted(attr_dict.items(), key=lambda kv: abs(kv[1]), reverse=True)[:15]
        features = [item[0] for item in sorted_items]
        values = [item[1] for item in sorted_items]
        colors = ["#d62728" if v > 0 else "#1f77b4" for v in values]

        fig, ax = plt.subplots(figsize=(10, max(4, len(features) * 0.45)))
        ax.barh(features[::-1], values[::-1], color=colors[::-1])
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Attribution (original scale)")
        ax.set_title(f"CUSIP {cusip} — Feature Attributions")
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
    Seaborn heatmap: features (rows) × CUSIPs (columns).
    Values in original-scale attributions.
    Returns list of saved PNG paths.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    cusips = list(result.attributions_original.keys())
    # Collect all features across all CUSIPs
    all_features = list(
        dict.fromkeys(
            feat
            for attr_dict in result.attributions_original.values()
            for feat in attr_dict
        )
    )

    data = {
        cusip: {feat: result.attributions_original[cusip].get(feat, 0.0) for feat in all_features}
        for cusip in cusips
    }
    df = pd.DataFrame(data, index=all_features)

    # Sort rows by max absolute value across CUSIPs
    df = df.loc[df.abs().max(axis=1).sort_values(ascending=False).index]
    df = df.iloc[:20]  # cap at top 20 features

    fig, ax = plt.subplots(figsize=(max(8, len(cusips) * 1.2), max(6, len(df) * 0.5)))
    sns.heatmap(
        df,
        ax=ax,
        center=0,
        cmap="RdBu_r",
        linewidths=0.3,
        annot=len(cusips) <= 8,
        fmt=".3f",
        cbar_kws={"label": "Attribution (original scale)"},
    )
    ax.set_title("Scenario Comparison — Feature Attributions (original scale)")
    ax.set_xlabel("CUSIP")
    ax.set_ylabel("Feature")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
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
    """
    1. Generate matplotlib/seaborn plots saved to MBS_SETTINGS.output_dir.
    2. Call LLM to produce a markdown narrative using original-scale attributions and
       feature context values.
    Returns partial state update: report_markdown, plot_paths.
    """
    output_dir = MBS_SETTINGS.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    exec_result_dict = state.get("execution_result")
    if not exec_result_dict:
        logger.warning("reporter_node called with no execution_result; skipping plots.")
        return {"report_markdown": "No execution result available.", "plot_paths": []}

    exec_result = ExecutionResult.model_validate(exec_result_dict)
    analysis_type = exec_result.analysis_type

    # --- Generate plots ---
    logger.info(f"Generating plots for analysis_type={analysis_type}")
    if analysis_type == "cusip_attribution":
        plot_paths = _plot_cusip_attribution(exec_result, output_dir)
    else:
        plot_paths = _plot_scenario_comparison(exec_result, output_dir)

    # --- LLM narrative ---
    logger.info("Generating LLM narrative report.")
    sys_prompt = T(".prompts:reporter.system").r()
    user_prompt = T(".prompts:reporter.user").r(
        question=state["question"],
        analysis_type=analysis_type,
        attributions_original=exec_result.attributions_original,
        feature_values_original=exec_result.feature_values_original,
        summary_stats=exec_result.summary_stats,
        metadata=exec_result.metadata,
        plot_paths=plot_paths,
        iteration_count=state.get("iteration_count", 1),
    )

    report_markdown = APIBackend().build_messages_and_create_chat_completion(
        user_prompt=user_prompt,
        system_prompt=sys_prompt,
    )

    logger.info(f"Report generated ({len(report_markdown)} chars). Plots: {plot_paths}")
    return {
        "report_markdown": report_markdown,
        "plot_paths": plot_paths,
    }
