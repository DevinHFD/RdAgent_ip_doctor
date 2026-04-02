"""
IP Doctor — Streamlit chatbot UI

Claude Code–inspired interface for the LangGraph ip_doctor workflow:
  • Compact step pills (icon + label) stream live as each node fires
  • Human-review card: plan summary, Before/After/Δ feature table,
    attribution bar chart, model-prediction table, Approve / Reject controls
  • Final report displayed as markdown with embedded attribution plots
    and a PDF download button

Install:
    pip install streamlit

Run (from repo root):
    streamlit run rdagent/scenarios/ip_doctor/ui/streamlit_app.py

Options:
    streamlit run rdagent/scenarios/ip_doctor/ui/streamlit_app.py \\
        --server.port 8080 --server.headless true
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Any

import streamlit as st
from langgraph.types import Command

# ---------------------------------------------------------------------------
# Page config — must be the first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="IP Doctor",
    page_icon="🏥",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Lazy backend import (keeps startup fast; surfaces import errors gracefully)
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading analysis engine…")
def _load_graph():
    from rdagent.scenarios.ip_doctor.graph import build_graph

    return build_graph()


# ---------------------------------------------------------------------------
# Node metadata
# ---------------------------------------------------------------------------

NODE_META: dict[str, tuple[str, str]] = {
    "question_parser": ("🔍", "Parsing question"),
    "planner":         ("🗺️",  "Planning analysis"),
    "code_generator":  ("⚙️",  "Generating attribution code"),
    "code_validator":  ("✔️",  "Validating code"),
    "executor":        ("🚀",  "Running IG analysis"),
    "debugger":        ("🔧",  "Fixing errors"),
    "human_reviewer":  ("👤",  "Preparing review"),
    "reporter":        ("📊",  "Generating report"),
}

TOP_N = 12  # features to show in the review chart

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

st.markdown(
    """
<style>
/* ── page background ── */
[data-testid="stAppViewContainer"] { background: #f8f8f7; }
[data-testid="stSidebar"]          { background: #f0efe9;
                                      border-right: 1px solid #e3e2dc; }

/* ── step pills (Claude Code style) ── */
.step-list { display:flex; flex-direction:column; gap:3px; margin:6px 0 2px; }
.step-pill {
    display: inline-flex; align-items: center; gap: 6px;
    font-size: 0.80rem; color: #5a5a58;
    background: #efefed; border-radius: 20px;
    padding: 2px 11px; width: fit-content;
    font-family: 'SF Mono','Fira Code',monospace;
}
.step-pill.done  { color: #1d6e31; background: #e6f4ea; }
.step-pill.spin  { color: #1558a0; background: #e3f0fd; }
.step-pill.error { color: #b71c1c; background: #fdecea; }

/* ── review card ── */
.rc {
    background: #fff;
    border: 1.5px solid #d4d2cc;
    border-left: 4px solid #f59e0b;
    border-radius: 10px;
    padding: 20px 22px 14px;
    margin: 8px 0 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,.06);
}
.rc-header {
    font-size: 1.05rem; font-weight: 700;
    color: #92400e; margin-bottom: 14px;
}
.plan-kv { font-size: 0.88rem; color: #333; margin: 3px 0; }
.plan-key { font-weight: 600; color: #111;
            min-width: 110px; display: inline-block; }
.error-note {
    background: #fffbeb; border-left: 4px solid #f59e0b;
    border-radius: 5px; padding: 8px 13px;
    font-size: 0.87rem; color: #78350f; margin: 8px 0;
}

/* ── report container ── */
.report-wrap {
    background: #fff;
    border: 1px solid #dddcd7;
    border-radius: 10px;
    padding: 20px 24px;
    margin: 8px 0;
}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Session-state initialisation
# ---------------------------------------------------------------------------


def _init() -> None:
    if "session_id" not in st.session_state:
        st.session_state.update(
            session_id=str(uuid.uuid4()),
            graph=_load_graph(),
            messages=[],           # chat history list[dict]
            workflow_status="idle",  # idle | streaming | review | done | error
            interrupt_payload=None,
            pending_question=None,
            pending_resume=None,
            last_error=None,
        )


def _reset() -> None:
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.rerun()


# ---------------------------------------------------------------------------
# LangGraph helpers
# ---------------------------------------------------------------------------


def _config() -> dict:
    return {"configurable": {"thread_id": st.session_state.session_id}}


def _initial_state(question: str) -> dict:
    return {
        "question": question,
        "question_type": "",
        "cusip_list": [],
        "scenario_params": {},
        "analysis_plan": None,
        "generated_code": "",
        "code_valid": False,
        "validation_errors": [],
        "execution_result": None,
        "execution_error": None,
        "debug_attempts": 0,
        "human_feedback": None,
        "report_markdown": None,
        "plot_paths": [],
        "pdf_path": None,
        "iteration_count": 0,
        "session_id": st.session_state.session_id,
    }


# ---------------------------------------------------------------------------
# Execution-summary parsers
# ---------------------------------------------------------------------------


def _parse_features(exec_summary: str) -> list[dict]:
    """
    Parse the feature attribution table from execution_summary text.

    Expected format (from human_reviewer.py):
      {feat:<28}  {attribution:>+14.4f}  {t0:>10}  {t1:>10}  {delta:>10}

    Returns list of dicts with keys: name, attribution, t0, t1, delta.
    """
    results: list[dict] = []
    in_table = False
    for line in exec_summary.splitlines():
        stripped = line.strip()
        if stripped.startswith("Top features"):
            in_table = True
            continue
        if stripped.startswith("Model SMM Predictions"):
            break
        if not in_table or not stripped or stripped.startswith("Feature") or set(stripped) <= {"─", "-", " "}:
            continue
        # Split on 2+ whitespace — first token is feature name, rest are numbers
        parts = re.split(r"\s{2,}", stripped)
        if len(parts) < 2:
            continue
        name = parts[0].strip()
        nums: list[float | None] = []
        for p in parts[1:]:
            p = p.strip()
            if p in ("—", "-"):
                nums.append(None)
            else:
                try:
                    nums.append(float(p))
                except ValueError:
                    nums.append(None)
        if not nums:
            continue
        attr = nums[0] if nums else None
        t0   = nums[1] if len(nums) > 1 else None
        t1   = nums[2] if len(nums) > 2 else None
        delta = nums[3] if len(nums) > 3 else None
        if attr is not None:
            results.append({"name": name, "attribution": attr, "t0": t0, "t1": t1, "delta": delta})
    return results


def _parse_predictions(exec_summary: str) -> list[dict]:
    """
    Parse the model SMM predictions table from execution_summary text.

    Expected format:
      {cusip:<14} {period:<24} {b:>10} {a:>10} {d:>10}

    Returns list of dicts: cusip, period, before, after, delta.
    """
    results: list[dict] = []
    in_pred = False
    for line in exec_summary.splitlines():
        stripped = line.strip()
        if stripped.startswith("Model SMM Predictions"):
            in_pred = True
            continue
        if not in_pred:
            continue
        if not stripped or stripped.startswith("CUSIP") or set(stripped) <= {"─", "-", " "}:
            continue
        parts = re.split(r"\s{2,}", stripped)
        if len(parts) < 3:
            continue
        cusip  = parts[0].strip()
        period = parts[1].strip() if len(parts) > 1 else ""
        nums: list[float | None] = []
        for p in parts[2:]:
            p = p.strip()
            if p in ("—", "-"):
                nums.append(None)
            else:
                try:
                    nums.append(float(p))
                except ValueError:
                    nums.append(None)
        results.append({
            "cusip":  cusip,
            "period": period,
            "before": nums[0] if nums else None,
            "after":  nums[1] if len(nums) > 1 else None,
            "delta":  nums[2] if len(nums) > 2 else None,
        })
    return results


def _parse_plan(plan_summary: str) -> list[tuple[str, str]]:
    """Parse 'Key   : value' lines into (key, value) tuples."""
    rows: list[tuple[str, str]] = []
    for line in plan_summary.splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            rows.append((k.strip(), v.strip()))
    return rows


# ---------------------------------------------------------------------------
# Step-pills renderer
# ---------------------------------------------------------------------------


def _steps_html(steps: list[dict]) -> str:
    parts = ['<div class="step-list">']
    for s in steps:
        cls = {"done": "done", "error": "error", "running": "spin"}.get(s["status"], "done")
        suffix = " ⟳" if s["status"] == "running" else (" ✓" if s["status"] == "done" else " ✗")
        parts.append(
            f'<span class="step-pill {cls}">{s["icon"]} {s["label"]}{suffix}</span>'
        )
    parts.append("</div>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Attribution bar chart (matplotlib, rendered via st.pyplot)
# ---------------------------------------------------------------------------


def _attribution_chart(features: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    names  = [f["name"] for f in features]
    values = [f["attribution"] for f in features]
    colors = ["#16a34a" if v >= 0 else "#dc2626" for v in values]

    fig, ax = plt.subplots(figsize=(7, max(2.5, len(names) * 0.38)))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#fafaf8")

    bars = ax.barh(names[::-1], values[::-1], color=colors[::-1], height=0.62, edgecolor="none")
    ax.axvline(0, color="#555", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Mean IG Attribution (CPR/SMM contribution)", fontsize=8, color="#555")
    ax.tick_params(axis="y", labelsize=8)
    ax.tick_params(axis="x", labelsize=7)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:+.4f}"))
    ax.grid(axis="x", color="#e0dfda", linewidth=0.5)

    for bar, val in zip(bars, values[::-1]):
        ax.text(
            val + (0.00005 if val >= 0 else -0.00005),
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.4f}",
            va="center",
            ha="left" if val >= 0 else "right",
            fontsize=7, color="#333",
        )

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Review card
# ---------------------------------------------------------------------------


def _review_card(payload: dict) -> None:
    iteration    = payload.get("iteration", 1)
    plan_summary = payload.get("plan_summary", "")
    exec_summary = payload.get("execution_summary", "")
    error_note   = payload.get("error_note") or ""

    plan_rows = _parse_plan(plan_summary)
    features  = _parse_features(exec_summary)
    preds     = _parse_predictions(exec_summary)

    # ── Card wrapper ──────────────────────────────────────────────────────
    st.markdown('<div class="rc">', unsafe_allow_html=True)

    st.markdown(
        f'<div class="rc-header">⚠️ Human Review Required'
        f'<span style="font-weight:400;font-size:.9rem"> &nbsp;·&nbsp; '
        f'Iteration {iteration}</span></div>',
        unsafe_allow_html=True,
    )

    # ── Analysis Plan ─────────────────────────────────────────────────────
    st.markdown("**Analysis Plan**")
    kv_html = "".join(
        f'<div class="plan-kv"><span class="plan-key">{k}</span>{v}</div>'
        for k, v in plan_rows
    )
    st.markdown(kv_html, unsafe_allow_html=True)

    # ── Error note ────────────────────────────────────────────────────────
    if error_note:
        st.markdown(
            f'<div class="error-note">⚠️ <b>Error note:</b> {error_note}</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Feature attribution chart ─────────────────────────────────────────
    if features:
        st.markdown("**Top Feature Attributions** *(mean IG contribution)*")
        _attribution_chart(features[:TOP_N])

        # Before / After / Δ table
        import pandas as pd
        rows_data = []
        for f in features[:TOP_N]:
            rows_data.append(
                {
                    "Feature":     f["name"],
                    "Attribution": f"{f['attribution']:+.4f}" if f["attribution"] is not None else "—",
                    "Before":      f"{f['t0']:.3f}"   if f["t0"]    is not None else "—",
                    "After":       f"{f['t1']:.3f}"   if f["t1"]    is not None else "—",
                    "Δ":           f"{f['delta']:+.3f}" if f["delta"] is not None else "—",
                }
            )
        if rows_data:
            st.dataframe(
                pd.DataFrame(rows_data).set_index("Feature"),
                use_container_width=True,
                height=min(38 * len(rows_data) + 40, 360),
            )
    elif exec_summary and exec_summary != "No successful execution result available.":
        st.markdown("**Execution Summary**")
        st.code(exec_summary, language=None)

    # ── Model predictions ─────────────────────────────────────────────────
    if preds:
        st.markdown("**Model SMM Predictions**")
        import pandas as pd
        pred_rows = []
        for p in preds:
            pred_rows.append(
                {
                    "CUSIP":   p["cusip"],
                    "Period":  p["period"],
                    "Before":  f"{p['before']:.6f}" if p["before"] is not None else "—",
                    "After":   f"{p['after']:.6f}"  if p["after"]  is not None else "—",
                    "Δ SMM":   f"{p['delta']:+.6f}" if p["delta"]  is not None else "—",
                }
            )
        st.dataframe(
            pd.DataFrame(pred_rows),
            use_container_width=True,
            hide_index=True,
            height=min(38 * len(pred_rows) + 40, 260),
        )

    st.divider()

    # ── Approve / Reject controls ─────────────────────────────────────────
    decision = st.radio(
        "Your decision",
        ["Approve ✓", "Reject ✗"],
        horizontal=True,
        key=f"decision_{iteration}",
        label_visibility="collapsed",
    )

    feedback: dict[str, Any] = {}

    if decision == "Reject ✗":
        c1, c2 = st.columns(2)
        with c1:
            what_wrong = st.text_area(
                "What is wrong?",
                placeholder="e.g. Wrong months, incorrect baseline…",
                key=f"what_{iteration}",
                height=80,
            )
        with c2:
            suggestion = st.text_area(
                "Suggested change",
                placeholder="e.g. Use Feb→Mar instead of Jan→Feb",
                key=f"sug_{iteration}",
                height=80,
            )
        focus_raw = st.text_input(
            "Additional CUSIPs to add (comma-separated, or blank)",
            placeholder="e.g. 31418DSB5, 3140ABCD1",
            key=f"cusips_{iteration}",
        )
        focus = [c.strip() for c in focus_raw.split(",") if c.strip()] or None
        feedback = {
            "decision": "reject",
            "what_is_wrong":    what_wrong or None,
            "suggested_change": suggestion or None,
            "focus_cusips":     focus,
        }
        btn_label = "✗ Submit Rejection"
        btn_type  = "secondary"
    else:
        feedback  = {"decision": "approve"}
        btn_label = "✓ Approve — Proceed to Report"
        btn_type  = "primary"

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button(btn_label, type=btn_type, use_container_width=True, key=f"submit_{iteration}"):
        # Record the decision in chat history
        label = "Approved" if feedback["decision"] == "approve" else "Rejected"
        st.session_state.messages.append(
            {"role": "assistant", "type": "review_decision",
             "content": label, "feedback": feedback}
        )
        st.session_state.pending_resume   = Command(resume=feedback)
        st.session_state.workflow_status  = "streaming"
        st.session_state.interrupt_payload = None
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Final report renderer
# ---------------------------------------------------------------------------


def _render_report(report_md: str, plot_paths: list[str], pdf_path: str | None) -> None:
    st.markdown('<div class="report-wrap">', unsafe_allow_html=True)
    st.markdown("#### 📊 Analysis Report")
    st.markdown(report_md)

    if plot_paths:
        st.markdown("---")
        st.markdown("**Attribution Plots**")
        # Show aggregated plot first (if present), then per-CUSIP
        ordered = sorted(plot_paths, key=lambda p: (0 if "aggregated" in p else 1, p))
        cols = st.columns(min(2, len(ordered)))
        for i, path in enumerate(ordered):
            p = Path(path)
            if p.exists():
                cols[i % 2].image(str(p), caption=p.stem, use_container_width=True)
            else:
                cols[i % 2].caption(f"Plot saved: `{path}`")

    if pdf_path and Path(pdf_path).exists():
        st.markdown("---")
        with open(pdf_path, "rb") as f:
            st.download_button(
                label="📄 Download PDF Report",
                data=f.read(),
                file_name=Path(pdf_path).name,
                mime="application/pdf",
            )

    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Message renderer (history replay)
# ---------------------------------------------------------------------------


def _render_message(msg: dict) -> None:
    role  = msg["role"]
    mtype = msg.get("type", "text")

    if role == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
        return

    with st.chat_message("assistant", avatar="🏥"):
        if mtype == "steps":
            st.markdown(_steps_html(msg["steps"]), unsafe_allow_html=True)
        elif mtype == "report":
            _render_report(msg["content"], msg.get("plot_paths", []), msg.get("pdf_path"))
        elif mtype == "review_decision":
            icon = "✅" if msg["content"] == "Approved" else "🔄"
            st.markdown(f"{icon} **{msg['content']}** — resuming analysis…")
        elif mtype == "error":
            st.error(msg["content"])
        else:
            st.markdown(msg["content"])


# ---------------------------------------------------------------------------
# Streaming engine
# ---------------------------------------------------------------------------


def _run_streaming() -> None:
    """
    Execute or resume the LangGraph graph, streaming node events live.
    Uses st.status() for the live step display, then stores steps in history.
    Called on every rerun while workflow_status == "streaming".
    """
    graph   = st.session_state.graph
    cfg     = _config()
    resume  = st.session_state.pending_resume
    question = st.session_state.pending_question

    graph_input = resume if resume is not None else _initial_state(question)
    st.session_state.pending_resume   = None
    st.session_state.pending_question = None

    new_steps: list[dict] = []

    with st.chat_message("assistant", avatar="🏥"):
        with st.status("Thinking…", expanded=True) as status:
            try:
                for event in graph.stream(graph_input, config=cfg, stream_mode="updates"):

                    # ── Interrupt ───────────────────────────────────────────
                    if "__interrupt__" in event:
                        interrupts = event["__interrupt__"]
                        payload = interrupts[0].value if interrupts else {}
                        if new_steps:
                            new_steps[-1]["status"] = "done"
                        st.markdown(_steps_html(new_steps), unsafe_allow_html=True)
                        status.update(label="⚠️ Review required", state="running", expanded=False)
                        _save_steps(new_steps)
                        st.session_state.interrupt_payload = payload
                        st.session_state.workflow_status   = "review"
                        st.rerun()
                        return

                    # ── Normal node event ───────────────────────────────────
                    node = next(iter(event), None)
                    if node and node != "__end__":
                        icon, label = NODE_META.get(node, ("•", node))
                        new_steps.append({"icon": icon, "label": label, "status": "done"})
                        st.markdown(_steps_html(new_steps), unsafe_allow_html=True)
                        status.update(label=f"{icon} {label}…")

            except Exception as exc:  # noqa: BLE001
                _save_steps(new_steps, error=True)
                st.session_state.last_error      = str(exc)
                st.session_state.workflow_status = "error"
                status.update(label="❌ Error", state="error", expanded=True)
                st.session_state.messages.append(
                    {"role": "assistant", "type": "error",
                     "content": f"**Analysis error:** {exc}"}
                )
                st.rerun()
                return

            status.update(label="✅ Complete", state="complete", expanded=False)

    _save_steps(new_steps)

    # Collect final outputs
    final  = graph.get_state(cfg).values
    report = final.get("report_markdown") or ""
    plots  = final.get("plot_paths", [])
    pdf    = final.get("pdf_path") or None

    st.session_state.messages.append(
        {"role": "assistant", "type": "report",
         "content": report, "plot_paths": plots, "pdf_path": pdf}
    )
    st.session_state.workflow_status = "done"
    st.rerun()


def _save_steps(steps: list[dict], *, error: bool = False) -> None:
    if not steps:
        return
    if error and steps:
        steps[-1]["status"] = "error"
    st.session_state.messages.append(
        {"role": "assistant", "type": "steps", "steps": list(steps)}
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def _sidebar() -> None:
    with st.sidebar:
        st.markdown("## 🏥 IP Doctor")
        st.markdown("*MBS Prepayment Attribution*")
        st.divider()

        st.markdown(f"**Session**  \n`{st.session_state.session_id[:20]}…`")
        icons = {"idle": "💤", "streaming": "⚡", "review": "👤",
                 "done": "✅", "error": "❌"}
        labels = {"idle": "Idle", "streaming": "Running", "review": "Awaiting Review",
                  "done": "Complete", "error": "Error"}
        s = st.session_state.workflow_status
        st.markdown(f"**Status**  \n{icons.get(s,'❓')} {labels.get(s,'Unknown')}")

        st.divider()
        if st.button("🔄 New Session", use_container_width=True):
            _reset()

        st.divider()
        st.markdown("**Try an example**")
        examples = [
            "For CUSIP 3140GXPJ8, what drove the CPR change from Jan to Feb 2024?",
            "For CUSIPs 3140GXPJ8 and 31418DSB5, show month-over-month SMM attribution for Q1 2024.",
            "Compare base vs rate_shock_100bps for CUSIPs 3140GXPJ8 and 31418DSB5.",
        ]
        for ex in examples:
            if st.button(ex[:55] + "…" if len(ex) > 55 else ex,
                         use_container_width=True, key=f"ex_{ex[:15]}"):
                if st.session_state.workflow_status in ("idle", "done"):
                    st.session_state.messages.append({"role": "user", "content": ex})
                    st.session_state.pending_question = ex
                    st.session_state.workflow_status  = "streaming"
                    st.rerun()

        st.divider()
        st.caption(
            "Flow: question parser → planner → code generator → "
            "validator → executor → [debugger] → **human review** → reporter"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    _init()
    _sidebar()

    # Header
    st.markdown("## 🏥 IP Doctor — MBS Prepayment Attribution")
    st.caption(
        "Ask a natural-language question about your prepayment model. "
        "The agent plans, generates, and executes an Integrated Gradients analysis — "
        "then pauses for your review before writing the final report."
    )
    st.divider()

    # Replay chat history
    for msg in st.session_state.messages:
        _render_message(msg)

    status = st.session_state.workflow_status

    # Active states
    if status == "streaming":
        _run_streaming()

    elif status == "review":
        payload = st.session_state.interrupt_payload or {}
        with st.chat_message("assistant", avatar="🏥"):
            _review_card(payload)

    elif status == "error":
        st.error(f"**Analysis error:** {st.session_state.last_error or 'Unknown'}")
        if st.button("🔄 Start over"):
            _reset()

    # Chat input — disabled while busy
    busy = status in ("streaming", "review")
    prompt = st.chat_input(
        placeholder="Ask about your MBS model…" if not busy else "Analysis in progress…",
        disabled=busy,
    )
    if prompt and status in ("idle", "done"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.pending_question = prompt
        st.session_state.workflow_status  = "streaming"
        st.rerun()


if __name__ == "__main__":
    main()
