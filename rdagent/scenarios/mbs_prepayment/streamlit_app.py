"""
streamlit_app.py — Chatbot UI for MBS prepayment attribution analysis.

Inspired by the Claude Code terminal UI: step-by-step progress is shown inline
as compact labelled items, and the human-review interrupt surfaces a rich card
with an attribution chart and Approve / Reject controls.

Run with:
    streamlit run rdagent/scenarios/mbs_prepayment/streamlit_app.py

Environment variables (same as CLI):
    MBS_MODEL_CHECKPOINT_DIR, MBS_DATA_FILE, MBS_OUTPUT_DIR, ...
    See rdagent/scenarios/mbs_prepayment/conf.py for the full list.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import streamlit as st
from langgraph.types import Command

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="MBS Prepayment Analysis",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Lazy backend import (avoids hard failure if heavy deps not installed yet)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading analysis engine…")
def _load_graph():
    from rdagent.scenarios.mbs_prepayment.graph import build_graph
    return build_graph()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NODE_META: dict[str, tuple[str, str]] = {
    "question_parser": ("🔍", "Parsing question"),
    "planner":         ("🗺️",  "Planning analysis"),
    "code_generator":  ("💻",  "Generating code"),
    "code_validator":  ("✔️",  "Validating code"),
    "executor":        ("⚡",  "Running attribution"),
    "debugger":        ("🔧",  "Fixing errors"),
    "human_reviewer":  ("👤",  "Preparing review"),
    "reporter":        ("📊",  "Writing report"),
}

# How many top features to show in the review bar chart
TOP_N_FEATURES = 12

# ---------------------------------------------------------------------------
# CSS — Claude Code–inspired minimal dark-on-light palette
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
<style>
/* ── global ── */
[data-testid="stAppViewContainer"] { background: #f9f9f8; }
[data-testid="stSidebar"] { background: #f0efe9; border-right: 1px solid #e5e4de; }

/* ── chat messages ── */
[data-testid="stChatMessage"] { border-radius: 12px; margin-bottom: 4px; }

/* ── step pills ── */
.step-list { display:flex; flex-direction:column; gap:3px; margin: 6px 0; }
.step-pill {
    display: inline-flex; align-items: center; gap: 6px;
    font-size: 0.82rem; color: #5a5a5a;
    background: #efefed; border-radius: 20px;
    padding: 2px 10px; width: fit-content;
    font-family: 'SF Mono', 'Fira Code', monospace;
}
.step-pill.done  { color: #2e7d32; background: #e8f5e9; }
.step-pill.error { color: #c62828; background: #ffebee; }
.step-pill.spin  { color: #1565c0; background: #e3f2fd; }

/* ── review card ── */
.review-card {
    background: #ffffff;
    border: 1.5px solid #d0cfc9;
    border-left: 4px solid #f59e0b;
    border-radius: 10px;
    padding: 20px 24px;
    margin: 10px 0;
    box-shadow: 0 2px 8px rgba(0,0,0,.07);
}
.review-header {
    font-size: 1.05rem; font-weight: 700;
    color: #92400e; letter-spacing: .3px;
    margin-bottom: 14px;
    display: flex; align-items: center; gap: 8px;
}
.plan-row { font-size: 0.88rem; color: #444; margin: 3px 0; }
.plan-key { font-weight: 600; color: #222; min-width: 100px; display: inline-block; }

/* ── approve / reject buttons ── */
div[data-testid="column"] button[kind="secondary"]:first-child {
    background: #16a34a !important; color: white !important;
    border: none !important; border-radius: 8px !important;
}

/* ── report block ── */
.report-container {
    background: #ffffff;
    border: 1px solid #e0dfda;
    border-radius: 10px;
    padding: 20px 24px;
    margin: 10px 0;
}

/* ── error box ── */
.error-note {
    background: #fff3cd; border-left: 4px solid #ffc107;
    border-radius: 6px; padding: 10px 14px;
    font-size: 0.87rem; color: #6d4c00; margin: 8px 0;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session-state initialisation
# ---------------------------------------------------------------------------

def _init_session() -> None:
    if "session_id" not in st.session_state:
        st.session_state.session_id      = str(uuid.uuid4())
        st.session_state.graph           = _load_graph()
        st.session_state.messages        = []   # chat history
        st.session_state.steps           = []   # [{icon, label, status}]
        st.session_state.workflow_status = "idle"   # idle | streaming | review | done | error
        st.session_state.interrupt_payload = None
        st.session_state.pending_question  = None   # queued user input
        st.session_state.pending_resume    = None   # Command(resume=...) after review
        st.session_state.last_error        = None


def _new_session() -> None:
    """Wipe session and start fresh (sidebar button)."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    # _load_graph() is cached — graph object reused, but a new thread_id is used
    st.rerun()


# ---------------------------------------------------------------------------
# LangGraph helpers
# ---------------------------------------------------------------------------

def _make_config() -> dict:
    return {"configurable": {"thread_id": st.session_state.session_id}}


def _get_interrupt_payload() -> dict | None:
    graph  = st.session_state.graph
    config = _make_config()
    state  = graph.get_state(config)
    if not state.next:
        return None
    tasks = state.tasks
    if tasks and tasks[0].interrupts:
        return tasks[0].interrupts[0].value
    return None


def _build_initial_state(question: str) -> dict:
    return {
        "question":          question,
        "question_type":     "",
        "cusip_list":        [],
        "scenario_params":   {},
        "analysis_plan":     None,
        "generated_code":    "",
        "code_valid":        False,
        "validation_errors": [],
        "execution_result":  None,
        "execution_error":   None,
        "debug_attempts":    0,
        "human_feedback":    None,
        "report_markdown":   None,
        "plot_paths":        [],
        "iteration_count":   0,
        "session_id":        st.session_state.session_id,
    }


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _render_steps(steps: list[dict], *, container=None) -> None:
    """Render a compact step list (icon + label + status indicator)."""
    target = container or st
    html_parts = ['<div class="step-list">']
    for s in steps:
        cls = {"done": "done", "error": "error", "running": "spin"}.get(s["status"], "done")
        suffix = " ⟳" if s["status"] == "running" else (" ✓" if s["status"] == "done" else " ✗")
        html_parts.append(
            f'<span class="step-pill {cls}">{s["icon"]} {s["label"]}{suffix}</span>'
        )
    html_parts.append("</div>")
    target.markdown("".join(html_parts), unsafe_allow_html=True)


def _render_review_card(payload: dict) -> None:
    """Rich review card with plan info, attribution chart, and Approve/Reject controls."""
    iteration   = payload.get("iteration", 1)
    plan_summary = payload.get("plan_summary", "")
    exec_summary = payload.get("execution_summary", "")
    error_note   = payload.get("error_note")

    # ── Parse plan_summary lines into key-value pairs ──
    plan_rows: list[tuple[str, str]] = []
    for line in plan_summary.splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            plan_rows.append((k.strip(), v.strip()))

    # ── Parse execution_summary into feature rows ──
    feat_names:  list[str]   = []
    feat_values: list[float] = []
    for line in exec_summary.splitlines():
        line = line.strip()
        if line.startswith("Top features") or not line:
            continue
        parts = line.split()
        if len(parts) >= 2:
            name = parts[0]
            for p in parts[1:]:
                if p.startswith("mean="):
                    try:
                        val = float(p.replace("mean=", "").replace("+", ""))
                        feat_names.append(name)
                        feat_values.append(val)
                    except ValueError:
                        pass

    st.markdown('<div class="review-card">', unsafe_allow_html=True)

    # Header
    st.markdown(
        f'<div class="review-header">⚠️ Human Review Required &nbsp;·&nbsp; '
        f'<span style="font-weight:400;font-size:0.9rem">Iteration {iteration}</span></div>',
        unsafe_allow_html=True,
    )

    # Plan summary
    st.markdown("**Analysis Plan**")
    rows_html = "".join(
        f'<div class="plan-row"><span class="plan-key">{k}</span>{v}</div>'
        for k, v in plan_rows
    )
    st.markdown(rows_html, unsafe_allow_html=True)

    st.divider()

    # Feature attribution chart
    if feat_names and feat_values:
        st.markdown("**Top Feature Attributions** *(mean IG contribution)*")
        _render_attribution_chart(feat_names[:TOP_N_FEATURES], feat_values[:TOP_N_FEATURES])
    elif exec_summary and exec_summary != "No successful execution result available.":
        st.markdown("**Execution Summary**")
        st.code(exec_summary, language=None)

    # Error note
    if error_note:
        st.markdown(
            f'<div class="error-note">⚠️ <strong>Error note:</strong> {error_note}</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Reject form (shown when radio="Reject") ──
    decision_key = f"review_decision_{iteration}"
    decision = st.radio(
        "Your decision",
        ["Approve ✓", "Reject ✗"],
        horizontal=True,
        key=decision_key,
        label_visibility="collapsed",
    )

    feedback: dict[str, Any] = {}

    if decision == "Reject ✗":
        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                what_wrong = st.text_area(
                    "What is wrong?",
                    placeholder="e.g. Wrong months selected, unexpected CUSIPs…",
                    key=f"what_wrong_{iteration}",
                    height=80,
                )
            with col2:
                suggestion = st.text_area(
                    "Suggested change",
                    placeholder="e.g. Use Feb→Mar instead of Jan→Feb",
                    key=f"suggestion_{iteration}",
                    height=80,
                )
            focus_raw = st.text_input(
                "Focus CUSIPs (comma-separated, optional)",
                placeholder="e.g. 3140GXPJ8, 31418DSB5",
                key=f"focus_cusips_{iteration}",
            )
            focus_cusips = [c.strip() for c in focus_raw.split(",") if c.strip()] or None
            feedback = {
                "decision": "reject",
                "what_is_wrong": what_wrong or None,
                "suggested_change": suggestion or None,
                "focus_cusips": focus_cusips,
            }
    else:
        feedback = {"decision": "approve"}

    # Submit button
    st.markdown("<br>", unsafe_allow_html=True)
    btn_label = "✓ Approve — Proceed to Report" if decision == "Approve ✓" else "✗ Submit Rejection"
    btn_type  = "primary" if decision == "Approve ✓" else "secondary"
    if st.button(btn_label, type=btn_type, use_container_width=True, key=f"submit_review_{iteration}"):
        st.session_state.pending_resume    = Command(resume=feedback)
        st.session_state.workflow_status   = "streaming"
        st.session_state.interrupt_payload = None
        # Append the review decision as an assistant message for history
        decision_label = "Approved" if feedback["decision"] == "approve" else "Rejected"
        st.session_state.messages.append({
            "role": "assistant",
            "type": "review_decision",
            "content": decision_label,
            "feedback": feedback,
        })
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


def _render_attribution_chart(names: list[str], values: list[float]) -> None:
    """Horizontal bar chart using matplotlib rendered via st.pyplot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    colors = ["#16a34a" if v > 0 else "#dc2626" for v in values]

    fig, ax = plt.subplots(figsize=(7, max(2.5, len(names) * 0.38)))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#fafaf9")

    bars = ax.barh(names[::-1], values[::-1], color=colors[::-1], height=0.6, edgecolor="none")
    ax.axvline(0, color="#444", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Mean IG Attribution (CPR/SMM contribution)", fontsize=8, color="#555")
    ax.tick_params(axis="y", labelsize=8)
    ax.tick_params(axis="x", labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:+.4f}"))
    ax.grid(axis="x", color="#e0dfda", linewidth=0.5)

    # Value labels on bars
    for bar, val in zip(bars, values[::-1]):
        ax.text(
            val + (0.0001 if val >= 0 else -0.0001),
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.4f}",
            va="center",
            ha="left" if val >= 0 else "right",
            fontsize=7,
            color="#333",
        )

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def _render_report(report_markdown: str, plot_paths: list[str]) -> None:
    """Render the final report markdown and embedded plots."""
    st.markdown('<div class="report-container">', unsafe_allow_html=True)
    st.markdown("#### 📊 Analysis Report")
    st.markdown(report_markdown)
    if plot_paths:
        st.markdown("---")
        st.markdown("**Generated Plots**")
        for path in plot_paths:
            p = Path(path)
            if p.exists():
                st.image(str(p), caption=p.name, use_container_width=True)
            else:
                st.caption(f"Plot saved: `{path}`")
    st.markdown("</div>", unsafe_allow_html=True)


def _render_message(msg: dict) -> None:
    """Dispatch a stored message to the appropriate renderer."""
    role = msg["role"]
    mtype = msg.get("type", "text")

    if role == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])

    elif role == "assistant":
        with st.chat_message("assistant", avatar="📈"):
            if mtype == "steps":
                _render_steps(msg["steps"])
            elif mtype == "report":
                _render_report(msg["content"], msg.get("plot_paths", []))
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
    Execute (or resume) the LangGraph graph, streaming events in real time.
    Steps are displayed live inside an st.status() block, then saved to
    session state for persistent replay.

    Called every rerun when workflow_status == "streaming".
    """
    graph   = st.session_state.graph
    config  = _make_config()
    resume  = st.session_state.pending_resume
    question = st.session_state.pending_question

    # Decide input: fresh run vs. resume after human review
    if resume is not None:
        graph_input = resume
        st.session_state.pending_resume = None
    else:
        graph_input = _build_initial_state(question)
        st.session_state.pending_question = None

    # Accumulate steps for this streaming segment
    new_steps: list[dict] = []

    with st.chat_message("assistant", avatar="📈"):
        with st.status("Thinking…", expanded=True) as status_widget:
            try:
                for event in graph.stream(graph_input, config=config, stream_mode="updates"):

                    # ── Interrupt event ──────────────────────────────────────
                    if "__interrupt__" in event:
                        interrupts = event["__interrupt__"]
                        payload = interrupts[0].value if interrupts else {}
                        # Mark last step as done
                        if new_steps:
                            new_steps[-1]["status"] = "done"
                        _render_steps(new_steps)
                        status_widget.update(label="⚠️ Review required", state="running", expanded=False)

                        st.session_state.interrupt_payload = payload
                        st.session_state.workflow_status   = "review"
                        _save_steps_to_history(new_steps)
                        st.rerun()
                        return  # unreachable but defensive

                    # ── Normal node event ────────────────────────────────────
                    node_name = next(iter(event), None)
                    if node_name and node_name != "__end__":
                        icon, label = NODE_META.get(node_name, ("•", node_name))
                        step = {"icon": icon, "label": label, "status": "done"}
                        new_steps.append(step)
                        # Mark previous as done, current as just appended
                        _render_steps(new_steps)
                        status_widget.update(label=f"{icon} {label}…")

            except Exception as exc:  # noqa: BLE001
                st.session_state.workflow_status = "error"
                st.session_state.last_error = str(exc)
                status_widget.update(label="❌ Error", state="error", expanded=True)
                _save_steps_to_history(new_steps, error=True)
                st.session_state.messages.append({
                    "role": "assistant",
                    "type": "error",
                    "content": f"**Error during analysis:** {exc}",
                })
                st.rerun()
                return

            # ── Graph finished (reached END) ─────────────────────────────
            status_widget.update(label="✅ Done", state="complete", expanded=False)

    _save_steps_to_history(new_steps)

    # Retrieve final state
    final_state  = graph.get_state(config).values
    report_md    = final_state.get("report_markdown") or ""
    plot_paths   = final_state.get("plot_paths", [])

    st.session_state.messages.append({
        "role":       "assistant",
        "type":       "report",
        "content":    report_md,
        "plot_paths": plot_paths,
    })
    st.session_state.workflow_status = "done"
    st.rerun()


def _save_steps_to_history(steps: list[dict], *, error: bool = False) -> None:
    """Persist the current segment's steps as an assistant message in history."""
    if not steps:
        return
    if error and steps:
        steps[-1]["status"] = "error"
    st.session_state.messages.append({
        "role":  "assistant",
        "type":  "steps",
        "steps": list(steps),  # copy
    })


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## MBS Prepayment")
        st.markdown("*Integrated Gradients Attribution*")
        st.divider()

        st.markdown(f"**Session**  \n`{st.session_state.session_id[:18]}…`")

        status_map = {
            "idle":      ("💤", "Idle"),
            "streaming": ("⚡", "Running"),
            "review":    ("👤", "Awaiting Review"),
            "done":      ("✅", "Complete"),
            "error":     ("❌", "Error"),
        }
        icon, label = status_map.get(st.session_state.workflow_status, ("❓", "Unknown"))
        st.markdown(f"**Status**  \n{icon} {label}")

        st.divider()
        if st.button("🔄 New Session", use_container_width=True):
            _new_session()

        st.divider()
        st.markdown("**Example questions**")
        examples = [
            "For CUSIP 3140GXPJ8, what drove the CPR change from Jan to Feb 2024?",
            "Compare base vs rate_shock_100bps for CUSIPs 3140GXPJ8 and 31418DSB5.",
        ]
        for ex in examples:
            if st.button(ex, use_container_width=True, key=f"ex_{ex[:20]}"):
                if st.session_state.workflow_status in ("idle", "done"):
                    st.session_state.pending_question  = ex
                    st.session_state.workflow_status   = "streaming"
                    st.session_state.messages.append({"role": "user", "content": ex})
                    st.rerun()

        st.divider()
        st.caption(
            "Steps: question parser → planner → code generator → "
            "validator → executor → [debugger] → **human review** → reporter"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _init_session()
    _render_sidebar()

    # ── Header ──────────────────────────────────────────────────────────────
    st.markdown("## 📈 MBS Prepayment Analysis")
    st.caption(
        "Ask a natural-language question about your MBS model. "
        "The agent plans, generates, and executes an Integrated Gradients "
        "attribution analysis — then pauses for your review before producing the final report."
    )
    st.divider()

    # ── Replay chat history ─────────────────────────────────────────────────
    for msg in st.session_state.messages:
        _render_message(msg)

    # ── Active workflow states ──────────────────────────────────────────────
    status = st.session_state.workflow_status

    if status == "streaming":
        _run_streaming()

    elif status == "review":
        payload = st.session_state.interrupt_payload or {}
        with st.chat_message("assistant", avatar="📈"):
            _render_review_card(payload)

    elif status == "error":
        err = st.session_state.last_error or "Unknown error"
        st.error(f"**Analysis error:** {err}")
        if st.button("🔄 Start over"):
            _new_session()

    # ── Chat input (disabled while busy) ────────────────────────────────────
    is_busy = status in ("streaming", "review")
    prompt = st.chat_input(
        placeholder="Ask about your MBS model…" if not is_busy else "Analysis in progress…",
        disabled=is_busy,
    )

    if prompt and status in ("idle", "done"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.pending_question = prompt
        st.session_state.workflow_status  = "streaming"
        st.rerun()


if __name__ == "__main__":
    main()
