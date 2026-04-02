"""
IP Doctor — Gradio chatbot UI

Presents the LangGraph ip_doctor workflow as an interactive chat with:
  • Collapsible intermediate steps (HTML <details> tags, Claude Code style)
  • Human-review panel with ✅ Approve / ❌ Reject buttons
  • Guided reject form (what's wrong / suggested change / additional CUSIPs)
  • Inline attribution plots + downloadable PDF report

Install:
    pip install "gradio>=4.0"

Run:
    python rdagent/scenarios/ip_doctor/ui/gradio_app.py
    # or with a custom port:
    python rdagent/scenarios/ip_doctor/ui/gradio_app.py --server_port 7861
"""

from __future__ import annotations

import queue
import threading
import uuid
from pathlib import Path

import gradio as gr
from langgraph.types import Command

from rdagent.scenarios.ip_doctor.graph import build_graph
from rdagent.scenarios.ip_doctor.state import MBSAnalysisState

# ---------------------------------------------------------------------------
# Node metadata + HTML step formatter
# ---------------------------------------------------------------------------

NODE_META: dict[str, tuple[str, str]] = {
    "question_parser": ("🔍", "Parsing question"),
    "planner":         ("🗺️", "Planning analysis"),
    "code_generator":  ("⚙️",  "Generating attribution code"),
    "code_validator":  ("✔️",  "Validating code"),
    "executor":        ("🚀", "Running IG analysis"),
    "debugger":        ("🔧", "Debugging code"),
    "human_reviewer":  ("👤", "Human review"),
    "reporter":        ("📊", "Generating report"),
}


def _step_body(node: str, out: dict) -> str:
    """Return the HTML body shown inside the <details> block."""
    if node == "question_parser":
        return (
            f"<b>Type:</b> <code>{out.get('question_type', '?')}</code><br>"
            f"<b>CUSIPs:</b> <code>{out.get('cusip_list', [])}</code>"
        )
    if node == "planner":
        plan = out.get("analysis_plan") or {}
        ds, ig = plan.get("data_spec", {}), plan.get("ig_params", {})
        months = plan.get("comparison_months") or []
        rat = (plan.get("rationale") or "")[:300]
        return (
            f"<b>CUSIPs:</b> <code>{ds.get('cusip_list', [])}</code><br>"
            f"<b>Periods:</b> <code>{months}</code><br>"
            f"<b>Baseline:</b> <code>{ig.get('baseline_strategy')}</code> · "
            f"n_steps=<code>{ig.get('n_steps')}</code> · "
            f"target=<code>{ig.get('target_output')}</code><br>"
            f"<b>Rationale:</b> {rat}"
        )
    if node == "code_generator":
        lines = len((out.get("generated_code") or "").splitlines())
        return f"Generated <b>{lines}</b> lines of Python from Jinja template."
    if node == "code_validator":
        errs = out.get("validation_errors") or []
        if errs:
            items = "".join(f"<li><code>{e}</code></li>" for e in errs)
            return f"❌ Validation failed:<ul>{items}</ul>"
        return "✅ All checks passed (syntax · security · output contract)."
    if node == "executor":
        if out.get("execution_result"):
            n = len((out["execution_result"].get("attributions_normalized")) or {})
            return f"✅ IG analysis complete — <b>{n}</b> CUSIP(s) processed."
        err = (out.get("execution_error") or "unknown error")[:500]
        return f"❌ Execution failed:<br><pre style='font-size:.85em;white-space:pre-wrap'>{err}</pre>"
    if node == "debugger":
        return f"Debug attempt <b>{out.get('debug_attempts', '?')}</b> — revised script submitted."
    if node == "reporter":
        plots = out.get("plot_paths") or []
        pdf = out.get("pdf_path") or ""
        return (
            f"Generated <b>{len(plots)}</b> plot(s).<br>"
            + (f"PDF: <code>{pdf}</code>" if pdf else
               "<i>(PDF skipped — WeasyPrint not available)</i>")
        )
    return "(no summary)"


def _step_html(node: str, out: dict) -> str:
    """Wrap a node's body in a collapsible <details> block."""
    icon, label = NODE_META.get(node, ("•", node))
    body = _step_body(node, out)
    return (
        f"<details><summary><b>{icon}&nbsp;{label}</b></summary>"
        f"<div style='padding:6px 14px;margin-top:4px;"
        f"border-left:3px solid #555;font-size:.9em;line-height:1.5'>"
        f"{body}</div></details>"
    )


def _review_html(payload: dict) -> str:
    plan  = payload.get("plan_summary", "")
    summ  = payload.get("execution_summary", "")
    itr   = payload.get("iteration", 1)
    err   = payload.get("error_note", "")
    parts = [
        f"<h3>🔎 Human Review — Iteration {itr}</h3>",
        "<b>Analysis Plan</b>",
        f"<pre style='background:#1a1a2e;padding:10px;border-radius:6px;"
        f"font-size:.85em;white-space:pre-wrap'>{plan}</pre>",
        "<b>Execution Summary</b>",
        f"<pre style='background:#1a1a2e;padding:10px;border-radius:6px;"
        f"font-size:.85em;white-space:pre-wrap'>{summ}</pre>",
    ]
    if err:
        parts.append(f"<p>⚠️ <b>Error note:</b> {err}</p>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Graph streaming helper — runs graph.stream() in a daemon thread
# ---------------------------------------------------------------------------

def _stream_graph(graph, input_or_cmd, config: dict):
    """
    Synchronous generator that yields ("chunk"|"error", payload) pairs.
    graph.stream() runs in a daemon thread; results are passed via a queue.
    """
    q: queue.Queue = queue.Queue()

    def _run() -> None:
        try:
            for chunk in graph.stream(
                input_or_cmd, config=config, stream_mode="updates"
            ):
                q.put(("chunk", chunk))
        except Exception as exc:
            q.put(("error", str(exc)))
        finally:
            q.put(("done", None))

    threading.Thread(target=_run, daemon=True).start()

    while True:
        kind, payload = q.get()
        if kind == "done":
            return
        yield kind, payload


# ---------------------------------------------------------------------------
# Core pass runner — yields (history, interrupt_payload | None)
# ---------------------------------------------------------------------------

def _run_pass(graph, input_or_cmd, config: dict, history: list):
    """
    Drive one graph.stream() call.  Yields (updated_history, interrupt_or_None).
    interrupt_or_None is the interrupt payload dict when the graph is paused,
    or None while still running.  Returns after END or first interrupt.
    """
    for kind, payload in _stream_graph(graph, input_or_cmd, config):
        if kind == "error":
            history = history + [
                {"role": "assistant", "content": f"❌ **Workflow error:**\n```\n{payload}\n```"}
            ]
            yield history, None
            return

        chunk: dict = payload
        for node_name, node_out in chunk.items():
            if node_name == "__interrupt__":
                interrupt_payload = node_out[0].value if node_out else {}
                yield history, interrupt_payload
                return

            history = history + [
                {"role": "assistant", "content": _step_html(node_name, node_out)}
            ]
            yield history, None


# ---------------------------------------------------------------------------
# Gradio event handlers (all are generators for live streaming)
# ---------------------------------------------------------------------------

def _initial_state(question: str, session_id: str) -> MBSAnalysisState:
    return {
        "question": question, "question_type": "", "cusip_list": [],
        "scenario_params": {}, "analysis_plan": None, "generated_code": "",
        "code_valid": False, "validation_errors": [], "execution_result": None,
        "execution_error": None, "debug_attempts": 0, "human_feedback": None,
        "report_markdown": None, "plot_paths": [], "pdf_path": None,
        "iteration_count": 0, "session_id": session_id,
    }


def _finish(graph, config, history):
    """Collect final report output from ended graph."""
    state = graph.get_state(config).values
    plots = [p for p in (state.get("plot_paths") or []) if Path(p).exists()]
    pdf   = state.get("pdf_path") or ""
    pdf   = pdf if pdf and Path(pdf).exists() else None
    report_md = state.get("report_markdown") or ""
    if report_md:
        history = history + [{"role": "assistant", "content": report_md}]
    return history, plots, pdf


def submit_fn(message, history, state):
    """
    Handle a new user question.
    Yields: (history, state, review_html, approve_vis, reject_vis,
             reject_form_vis, gallery, pdf, input_interactive)
    """
    message = (message or "").strip()
    if not message:
        yield history, state, "", False, False, False, None, None, True
        return

    # Build a fresh session for each question
    session_id = str(uuid.uuid4())
    graph = build_graph()
    config = {"configurable": {"thread_id": session_id}}
    state = {"graph": graph, "config": config}

    history = (history or []) + [{"role": "user", "content": message}]
    yield history, state, "", False, False, False, None, None, False

    initial = _initial_state(message, session_id)
    interrupt_payload = None

    for hist, interrupt in _run_pass(graph, initial, config, history):
        history = hist
        if interrupt is not None:
            interrupt_payload = interrupt
            state["interrupt_payload"] = interrupt
            review = _review_html(interrupt)
            yield history, state, review, True, True, False, None, None, False
            return
        yield history, state, "", False, False, False, None, None, False

    # Reached END on first pass (no review needed — skip_review mode or no interrupt)
    history, plots, pdf = _finish(graph, config, history)
    yield history, state, "", False, False, False, plots or None, pdf, True


def approve_fn(history, state):
    """
    Handle Approve button click. Resumes graph and streams remaining steps.
    Yields same tuple as submit_fn.
    """
    graph  = state.get("graph")
    config = state.get("config")
    if not graph or not config:
        yield history, state, "", False, False, False, None, None, True
        return

    history = history + [{"role": "assistant", "content": "✅ **Approved.** Generating report…"}]
    yield history, state, "", False, False, False, None, None, False

    # May loop through multiple review cycles
    input_or_cmd = Command(resume={"decision": "approve"})
    while True:
        interrupt_payload = None
        for hist, interrupt in _run_pass(graph, input_or_cmd, config, history):
            history = hist
            if interrupt is not None:
                interrupt_payload = interrupt
                state["interrupt_payload"] = interrupt
                review = _review_html(interrupt)
                yield history, state, review, True, True, False, None, None, False
                break
            yield history, state, "", False, False, False, None, None, False

        if interrupt_payload is not None:
            return  # wait for next approve/reject

        # No interrupt → graph reached END
        break

    history, plots, pdf = _finish(graph, config, history)
    yield history, state, "", False, False, False, plots or None, pdf, True


def open_reject_fn(history, state):
    """Show the reject form; hide approve/reject buttons."""
    return history, state, "", False, False, True, None, None, False


def submit_reject_fn(what_wrong, suggestion, focus_cusips_raw, history, state):
    """
    Handle reject form submission. Resumes graph with reject feedback.
    Yields same tuple as submit_fn.
    """
    graph  = state.get("graph")
    config = state.get("config")
    if not graph or not config:
        yield history, state, "", False, False, False, None, None, True
        return

    focus = [c.strip() for c in (focus_cusips_raw or "").split(",") if c.strip()]
    feedback = {
        "decision":         "reject",
        "what_is_wrong":    what_wrong or "",
        "suggested_change": suggestion or "",
        "focus_cusips":     focus or None,
    }

    fb_summary = (
        f"❌ **Rejected** — _{what_wrong}_"
        + (f"<br>Suggestion: _{suggestion}_" if suggestion else "")
        + (f"<br>Adding CUSIPs: `{focus}`" if focus else "")
    )
    history = history + [{"role": "assistant", "content": fb_summary}]
    history = history + [{"role": "assistant", "content": "🔄 Revising plan…"}]
    yield history, state, "", False, False, False, None, None, False

    input_or_cmd = Command(resume=feedback)
    while True:
        interrupt_payload = None
        for hist, interrupt in _run_pass(graph, input_or_cmd, config, history):
            history = hist
            if interrupt is not None:
                interrupt_payload = interrupt
                state["interrupt_payload"] = interrupt
                review = _review_html(interrupt)
                yield history, state, review, True, True, False, None, None, False
                break
            yield history, state, "", False, False, False, None, None, False

        if interrupt_payload is not None:
            return

        break

    history, plots, pdf = _finish(graph, config, history)
    yield history, state, "", False, False, False, plots or None, pdf, True


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

TITLE = "🏥 IP Doctor — MBS Prepayment Attribution"

WELCOME = """
Ask me to explain what drove your prepayment model's forecast change using **Integrated Gradients**.

**Example questions:**
- *For CUSIP 3140GXPJ8, what drove the CPR change from January to February 2024?*
- *For CUSIPs ABC123 and DEF456, show month-over-month SMM attribution for Q1 2024.*
- *Compare base vs rate_shock_100bps for CUSIPs ABC123 and DEF456.*
""".strip()

OUTPUTS = [
    "chatbot",       # gr.Chatbot
    "state",         # gr.State
    "review_html",   # gr.HTML  — review panel
    "approve_vis",   # bool     — approve button visible
    "reject_vis",    # bool     — reject button visible
    "form_vis",      # bool     — reject form visible
    "gallery",       # list     — plots
    "pdf_file",      # str|None — PDF path
    "input_inter",   # bool     — input interactive
]


def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title=TITLE,
        theme=gr.themes.Soft(primary_hue="slate", neutral_hue="slate"),
        css="""
        .chatbot-wrap { border-radius: 10px; }
        details { margin: 4px 0; }
        details summary { cursor: pointer; padding: 4px 2px; }
        details summary:hover { opacity: .8; }
        .review-panel { border: 1px solid #444; border-radius: 8px;
                        padding: 12px; margin: 8px 0; background: #1a1a2e; }
        """,
    ) as demo:

        gr.Markdown(f"# {TITLE}\n{WELCOME}")

        chatbot = gr.Chatbot(
            label="Conversation",
            type="messages",
            height=560,
            show_copy_button=True,
            elem_classes=["chatbot-wrap"],
            render_markdown=True,
            sanitize_html=False,   # allow our <details> HTML
        )

        # ── Human review panel ──────────────────────────────────────────────
        with gr.Group(visible=True) as review_group:
            review_html = gr.HTML(value="", visible=False)

            with gr.Row(visible=False) as action_row:
                approve_btn = gr.Button("✅  Approve — generate report",
                                        variant="primary", scale=1)
                reject_btn  = gr.Button("❌  Reject — revise plan",
                                        variant="stop",    scale=1)

        # ── Reject form ─────────────────────────────────────────────────────
        with gr.Group(visible=False) as reject_form:
            gr.Markdown("### Reject Feedback")
            what_wrong_box = gr.Textbox(
                label="What is wrong with the current results?",
                placeholder="e.g. wrong months selected, incorrect baseline",
                lines=2,
            )
            suggestion_box = gr.Textbox(
                label="What change would you like?",
                placeholder="e.g. use Feb→Mar instead of Jan→Feb",
                lines=2,
            )
            cusips_box = gr.Textbox(
                label="Additional CUSIPs to add (comma-separated, or leave blank)",
                placeholder="e.g. 31418DSB5, 3140ABCD1",
            )
            submit_reject_btn = gr.Button("Submit Rejection", variant="secondary")

        # ── Chat input ──────────────────────────────────────────────────────
        with gr.Row():
            msg_box  = gr.Textbox(
                placeholder="Ask a question about your MBS model...",
                show_label=False,
                scale=9,
                container=False,
            )
            send_btn = gr.Button("Send", variant="primary", scale=1)

        # ── Outputs ─────────────────────────────────────────────────────────
        with gr.Row(visible=True):
            gallery  = gr.Gallery(
                label="Attribution Plots",
                columns=2,
                height=400,
                visible=False,
            )
        with gr.Row():
            pdf_file = gr.File(label="📄 Download PDF Report", visible=False)

        # ── State ───────────────────────────────────────────────────────────
        state = gr.State({})

        # ── Shared output list ───────────────────────────────────────────────
        # Every generator yields a 9-tuple matching this list
        shared_outputs = [
            chatbot,
            state,
            review_html,
            action_row,     # visible bool maps to gr.update(visible=...)
            reject_btn,     # individual reject button visibility (same row)
            reject_form,
            gallery,
            pdf_file,
            msg_box,
        ]

        # We use a helper to build gr.update dicts from the yielded tuple
        def _build_updates(hist, st, rev_val, appr_vis, rej_vis,
                           form_vis, plots, pdf, inp_inter):
            return (
                hist,
                st,
                gr.update(value=rev_val, visible=bool(rev_val)),
                gr.update(visible=appr_vis),
                gr.update(visible=rej_vis),
                gr.update(visible=form_vis),
                gr.update(value=plots,   visible=bool(plots)),
                gr.update(value=pdf,     visible=bool(pdf)),
                gr.update(interactive=inp_inter),
            )

        def _wrap(gen_fn, *fn_args):
            """Wrap a generator function so each yield is passed through _build_updates."""
            for tup in gen_fn(*fn_args):
                yield _build_updates(*tup)

        # ── Wire events ──────────────────────────────────────────────────────
        send_event_kw = dict(
            outputs=shared_outputs,
            show_progress="hidden",
        )

        send_btn.click(
            fn=lambda msg, hist, st: _wrap(submit_fn, msg, hist, st),
            inputs=[msg_box, chatbot, state],
            **send_event_kw,
        ).then(fn=lambda: "", outputs=msg_box)

        msg_box.submit(
            fn=lambda msg, hist, st: _wrap(submit_fn, msg, hist, st),
            inputs=[msg_box, chatbot, state],
            **send_event_kw,
        ).then(fn=lambda: "", outputs=msg_box)

        approve_btn.click(
            fn=lambda hist, st: _wrap(approve_fn, hist, st),
            inputs=[chatbot, state],
            **send_event_kw,
        )

        reject_btn.click(
            fn=lambda hist, st: _wrap(open_reject_fn, hist, st),
            inputs=[chatbot, state],
            **send_event_kw,
        )

        submit_reject_btn.click(
            fn=lambda w, s, c, hist, st: _wrap(submit_reject_fn, w, s, c, hist, st),
            inputs=[what_wrong_box, suggestion_box, cusips_box, chatbot, state],
            **send_event_kw,
        ).then(
            fn=lambda: ("", "", ""),
            outputs=[what_wrong_box, suggestion_box, cusips_box],
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="IP Doctor Gradio UI")
    parser.add_argument("--server_port", type=int, default=7860)
    parser.add_argument("--server_name", type=str, default="0.0.0.0")
    parser.add_argument("--share", action="store_true",
                        help="Create a public Gradio share link")
    args = parser.parse_args()

    demo = build_ui()
    demo.queue()   # enable streaming / generator support
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
    )
