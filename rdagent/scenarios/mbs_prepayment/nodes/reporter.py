"""
reporter node — generate matplotlib/seaborn plots and an LLM narrative markdown report.

Attribution values (bar x-axis): IG output = SMM/CPR contribution per feature.
Y-axis labels show the original-scale feature change between the two periods:
  cusip_attribution  : "WALA  (120.3 → 121.3, Δ+1.0)"
  scenario_comparison: "WALA  (base: 120.3 | shock: 125.1, Δ+4.8)"

A self-contained PDF report (report.pdf) is also written to output_dir, with all
plots embedded as base64 images so no external file references are needed.
"""

import base64
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


_KNOWN_SUBKEYS = {"t0", "t1", "delta", "base", "scenario", "error"}


def _avg_feat_group(nested_orig: dict, group: str) -> dict[str, float]:
    """
    Average one sub-group (e.g. "t0", "t1", "delta", "base", "scenario")
    across all valid periods.
    nested_orig = {period_key: {"t0": {feat: val}, "t1": ..., "delta": ...}}

    Falls back gracefully when period_data uses the old flat format
    {feat: val} (no sub-group keys). In that case only "t1" / "scenario"
    groups return data; "t0" / "base" / "delta" return {}.
    """
    totals: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for period_data in nested_orig.values():
        if not isinstance(period_data, dict) or "error" in period_data:
            continue
        # Detect old flat format: no recognised sub-group key present
        if not any(k in period_data for k in _KNOWN_SUBKEYS):
            # Old format — period_data IS the feature dict (single time point)
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
# Feature change table for LLM prompt
# ---------------------------------------------------------------------------


def _build_feature_change_table(exec_result: ExecutionResult) -> str:
    """
    Build a compact, human-readable table of feature changes for the LLM prompt.

    For cusip_attribution:
      Feature              T0 (before)   T1 (after)      Δ Change
      WALA                     120.30       121.30          +1.00
      ...

    For scenario_comparison:
      Feature              Base           Scenario         Δ Change
      WALA                     120.30       125.10          +4.80
      ...

    All values are averages across CUSIPs and periods.
    Sorted by |Δ| descending so the biggest movers appear first.
    """
    analysis_type = exec_result.analysis_type
    feat_orig = exec_result.feature_values_original

    if analysis_type == "cusip_attribution":
        before_key, after_key = "t0", "t1"
        col_before, col_after = "T0 (before)", "T1 (after)"
    else:
        before_key, after_key = "base", "scenario"
        col_before, col_after = "Base", "Scenario"

    # Aggregate per-CUSIP data (cusip level here)
    before_means = _avg_feat_group(feat_orig, before_key)
    after_means  = _avg_feat_group(feat_orig, after_key)
    delta_means  = _avg_feat_group(feat_orig, "delta")

    # Fall back: compute delta from before/after if delta group is missing
    if not delta_means and before_means and after_means:
        delta_means = {f: after_means[f] - before_means[f]
                       for f in before_means if f in after_means}

    all_feats = sorted(
        set(before_means) | set(after_means) | set(delta_means),
        key=lambda f: abs(delta_means.get(f, 0)),
        reverse=True,
    )

    if not all_feats:
        return "(No feature change data available.)"

    header = f"  {'Feature':<28}  {col_before:>14}  {col_after:>14}  {'Δ Change':>10}"
    sep    = "  " + "-" * 72
    rows = [header, sep]
    for feat in all_feats[:20]:
        b = before_means.get(feat)
        a = after_means.get(feat)
        d = delta_means.get(feat)
        b_s = f"{b:>14.4f}" if b is not None else f"{'—':>14}"
        a_s = f"{a:>14.4f}" if a is not None else f"{'—':>14}"
        d_s = f"{d:>+10.4f}" if d is not None else f"{'—':>10}"
        rows.append(f"  {feat:<28}  {b_s}  {a_s}  {d_s}")

    note = (
        "(values are averages across CUSIPs and periods; sorted by |Δ| descending)"
    )
    return "\n".join(rows) + "\n" + note


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
# PDF generation
# ---------------------------------------------------------------------------

_PDF_CSS = """
@page { margin: 2cm; size: A4; }
body { font-family: "DejaVu Sans", Arial, sans-serif; font-size: 11pt; color: #222; }
h1 { font-size: 18pt; border-bottom: 2px solid #444; padding-bottom: 4px; }
h2 { font-size: 14pt; margin-top: 1.4em; color: #2c4770; border-bottom: 1px solid #ccc; }
h3 { font-size: 12pt; color: #2c4770; }
pre, code { background: #f4f4f4; border-radius: 3px; padding: 2px 5px; font-size: 9.5pt; }
pre { padding: 8px; overflow-x: auto; }
table { border-collapse: collapse; width: 100%; margin: 1em 0; }
th, td { border: 1px solid #ccc; padding: 5px 8px; text-align: left; }
th { background: #e8ecf0; }
img.plot { max-width: 100%; height: auto; margin: 1em 0; display: block; page-break-inside: avoid; }
hr { border: none; border-top: 1px solid #ccc; margin: 1.5em 0; }
"""


def _generate_pdf(report_markdown: str, plot_paths: list[str], output_dir: Path) -> str:
    """
    Render report_markdown to a self-contained PDF with all plots embedded.

    Plots are inserted between the markdown narrative and any existing image
    references are ignored — instead each PNG in plot_paths is appended as a
    separate full-width figure at the end of the document.

    Returns the absolute path to the written PDF.
    """
    try:
        import markdown as md_lib
        from weasyprint import HTML, CSS
    except ImportError as exc:
        logger.warning(f"PDF generation skipped — missing dependency: {exc}. "
                       "Install with: pip install markdown weasyprint")
        return ""

    # 1. Convert markdown narrative to HTML body
    html_body = md_lib.markdown(
        report_markdown,
        extensions=["tables", "fenced_code", "nl2br"],
    )

    # 2. Build <img> tags for each plot, embedding PNGs as base64 data URIs
    img_tags = []
    for path_str in plot_paths:
        p = Path(path_str)
        if not p.exists():
            logger.warning(f"Plot file not found, skipping in PDF: {p}")
            continue
        b64 = base64.b64encode(p.read_bytes()).decode("ascii")
        img_tags.append(
            f'<figure>'
            f'<img class="plot" src="data:image/png;base64,{b64}" alt="{p.name}" />'
            f'<figcaption style="font-size:9pt;color:#555;">{p.name}</figcaption>'
            f'</figure>'
        )

    plots_html = "\n".join(img_tags)

    # 3. Assemble full HTML document
    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>MBS Attribution Report</title></head>
<body>
{html_body}
{('<hr/><h2>Attribution Plots</h2>' + plots_html) if img_tags else ''}
</body>
</html>"""

    # 4. Write PDF
    pdf_path = output_dir / "report.pdf"
    HTML(string=full_html, base_url=str(output_dir)).write_pdf(
        str(pdf_path),
        stylesheets=[CSS(string=_PDF_CSS)],
    )
    logger.info(f"PDF report written to {pdf_path}")
    return str(pdf_path)


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
    feature_change_table = _build_feature_change_table(exec_result)
    sys_prompt = T(".prompts:reporter.system").r()
    user_prompt = T(".prompts:reporter.user").r(
        question=state["question"],
        analysis_type=analysis_type,
        attributions_normalized=exec_result.attributions_normalized,
        feature_change_table=feature_change_table,
        summary_stats=exec_result.summary_stats,
        metadata=exec_result.metadata,
        plot_paths=plot_paths,
    )

    report_markdown = APIBackend().build_messages_and_create_chat_completion(
        user_prompt=user_prompt,
        system_prompt=sys_prompt,
    )

    logger.info(f"Report generated ({len(report_markdown)} chars). Plots: {plot_paths}")

    # Write markdown to disk
    md_path = output_dir / "report.md"
    md_path.write_text(report_markdown, encoding="utf-8")
    logger.info(f"Markdown report written to {md_path}")

    # Generate PDF with embedded plots
    pdf_path = _generate_pdf(report_markdown, plot_paths, output_dir)

    return {"report_markdown": report_markdown, "plot_paths": plot_paths, "pdf_path": pdf_path}
