"""
IP Doctor — Chainlit chatbot UI

Presents the LangGraph ip_doctor workflow as an interactive chat with:
  • Collapsible intermediate steps (one per graph node, live-streamed)
  • Human-review panel with Approve / Reject action buttons
  • Inline attribution plots (aggregated first, then per-CUSIP)
  • Downloadable self-contained PDF report

Run:
    chainlit run rdagent/scenarios/ip_doctor/ui/app.py
    chainlit run rdagent/scenarios/ip_doctor/ui/app.py --port 8000
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import AsyncGenerator

import chainlit as cl
from langgraph.types import Command

from rdagent.scenarios.ip_doctor.graph import build_graph
from rdagent.scenarios.ip_doctor.state import MBSAnalysisState

# ---------------------------------------------------------------------------
# Node display metadata
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


def _fmt_node(node: str, out: dict) -> str:
    """Return a short markdown summary of what a node produced."""
    if node == "question_parser":
        return (
            f"**Type:** `{out.get('question_type', '?')}`  \n"
            f"**CUSIPs:** `{out.get('cusip_list', [])}`"
        )
    if node == "planner":
        plan = out.get("analysis_plan") or {}
        ds   = plan.get("data_spec", {})
        ig   = plan.get("ig_params", {})
        months = plan.get("comparison_months") or []
        return (
            f"**CUSIPs:** `{ds.get('cusip_list', [])}`  \n"
            f"**Periods:** `{months}`  \n"
            f"**Baseline:** `{ig.get('baseline_strategy')}` · "
            f"n_steps=`{ig.get('n_steps')}` · target=`{ig.get('target_output')}`  \n"
            f"**Rationale:** {plan.get('rationale', '')[:300]}"
        )
    if node == "code_generator":
        lines = len((out.get("generated_code") or "").splitlines())
        return f"Generated **{lines}** lines of Python from Jinja template."
    if node == "code_validator":
        errs = out.get("validation_errors") or []
        if errs:
            return "❌ Validation failed:\n" + "\n".join(f"- `{e}`" for e in errs)
        return "✅ All checks passed (syntax + security + output contract)."
    if node == "executor":
        if out.get("execution_result"):
            n = len((out["execution_result"].get("attributions_normalized")) or {})
            return f"✅ IG analysis complete — **{n}** CUSIP(s) processed."
        err = out.get("execution_error") or "unknown error"
        short = err[:600] if len(err) > 600 else err
        return f"❌ Execution failed:\n```\n{short}\n```"
    if node == "debugger":
        attempt = out.get("debug_attempts", "?")
        return f"Debug attempt **{attempt}** — revised script submitted for re-validation."
    if node == "reporter":
        plots = out.get("plot_paths") or []
        pdf   = out.get("pdf_path") or ""
        return (
            f"Generated **{len(plots)}** plot(s).  \n"
            + (f"PDF: `{pdf}`" if pdf else "_(PDF generation skipped — WeasyPrint not available)_")
        )
    return "(node output)"


async def _show_step(node: str, out: dict) -> None:
    """Render a single node's result as a collapsed Chainlit step."""
    icon, label = NODE_META.get(node, ("•", node))
    body = _fmt_node(node, out)
    async with cl.Step(name=f"{icon} {label}", type="tool") as step:
        step.output = body


# ---------------------------------------------------------------------------
# Async graph streamer — runs graph.stream() in a thread, yields chunks live
# ---------------------------------------------------------------------------

async def _graph_chunks(
    graph, input_or_cmd, config: dict
) -> AsyncGenerator[dict, None]:
    """Async generator that streams graph update chunks from a background thread."""
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def _produce() -> None:
        try:
            for chunk in graph.stream(
                input_or_cmd, config=config, stream_mode="updates"
            ):
                loop.call_soon_threadsafe(queue.put_nowait, chunk)
        except Exception as exc:
            loop.call_soon_threadsafe(queue.put_nowait, {"__error__": str(exc)})
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel

    fut = loop.run_in_executor(None, _produce)
    while True:
        chunk = await queue.get()
        if chunk is None:
            break
        yield chunk
    await fut


# ---------------------------------------------------------------------------
# Human review panel — Approve / Reject with guided reject form
# ---------------------------------------------------------------------------

async def _human_review(payload: dict) -> dict:
    """
    Render the review panel with action buttons and collect structured feedback.
    Called when LangGraph fires interrupt() inside human_reviewer_node.
    """
    plan_summary = payload.get("plan_summary", "")
    exec_summary = payload.get("execution_summary", "")
    iteration    = payload.get("iteration", 1)
    error_note   = payload.get("error_note", "")

    lines = [
        f"## 🔎 Human Review — Iteration {iteration}",
        "",
        "### Analysis Plan",
        f"```\n{plan_summary}\n```",
        "",
        "### Execution Summary",
        f"```\n{exec_summary}\n```",
    ]
    if error_note:
        lines += ["", f"> ⚠️ **Error:** {error_note}"]

    await cl.Message(content="\n".join(lines)).send()

    res = await cl.AskActionMessage(
        content="**Approve these results or reject to revise the plan?**",
        actions=[
            cl.Action(name="approve", value="approve", label="✅  Approve — generate report"),
            cl.Action(name="reject",  value="reject",  label="❌  Reject — revise plan"),
        ],
        timeout=3600,
    ).send()

    if res is None or res.get("value") == "approve":
        await cl.Message(content="✅ Approved. Generating report…").send()
        return {"decision": "approve"}

    # Guided reject form
    r1 = await cl.AskUserMessage(
        content="**What is wrong** with the current results?", timeout=600
    ).send()
    r2 = await cl.AskUserMessage(
        content="**What change** would you like? (e.g. different months, baseline strategy)",
        timeout=600,
    ).send()
    r3 = await cl.AskUserMessage(
        content="**Additional CUSIPs** to add (comma-separated), or press Enter to skip:",
        timeout=600,
    ).send()

    focus: list[str] = []
    if r3 and r3.get("output", "").strip():
        focus = [c.strip() for c in r3["output"].split(",") if c.strip()]

    await cl.Message(content="🔄 Revising plan based on your feedback…").send()
    return {
        "decision":         "reject",
        "what_is_wrong":    (r1 or {}).get("output", ""),
        "suggested_change": (r2 or {}).get("output", ""),
        "focus_cusips":     focus or None,
    }


# ---------------------------------------------------------------------------
# Report display — inline plots + PDF download
# ---------------------------------------------------------------------------

async def _show_report(state: dict) -> None:
    report_md  = state.get("report_markdown", "")
    plot_paths = state.get("plot_paths", [])
    pdf_path   = state.get("pdf_path", "")

    elements: list = []

    # Plots — aggregated is always first in the list
    for p in plot_paths:
        path = Path(p)
        if path.exists():
            elements.append(cl.Image(name=path.stem, path=str(path), display="inline"))

    # Downloadable PDF
    if pdf_path and Path(pdf_path).exists():
        elements.append(cl.File(name="report.pdf", path=str(pdf_path), display="inline"))

    await cl.Message(
        content=report_md or "*(No report generated.)*",
        elements=elements,
    ).send()


# ---------------------------------------------------------------------------
# Core workflow driver — handles the full graph including human review loops
# ---------------------------------------------------------------------------

async def _run_workflow(graph, initial_state: MBSAnalysisState, config: dict) -> None:
    """
    Drive the LangGraph workflow from start to END.
    Handles multiple human-review → re-plan loops transparently.
    """
    input_or_cmd: object = initial_state

    while True:
        interrupt_found = False

        async for chunk in _graph_chunks(graph, input_or_cmd, config):
            # Each chunk is {node_name: node_output_dict}
            for node_name, node_out in chunk.items():

                if node_name == "__error__":
                    await cl.Message(
                        content=f"❌ **Workflow error:**\n```\n{node_out}\n```"
                    ).send()
                    return

                if node_name == "__interrupt__":
                    # node_out is a tuple/list of Interrupt objects
                    payload = node_out[0].value if node_out else {}
                    feedback = await _human_review(payload)
                    input_or_cmd = Command(resume=feedback)
                    interrupt_found = True
                    break  # break inner for-loop over chunk items

                await _show_step(node_name, node_out)

            if interrupt_found:
                break  # break async-for over chunks; re-enter while loop

        if not interrupt_found:
            break  # graph reached END

    # Display final report
    final = graph.get_state(config).values
    await _show_report(final)


# ---------------------------------------------------------------------------
# Chainlit lifecycle hooks
# ---------------------------------------------------------------------------

@cl.on_chat_start
async def on_chat_start() -> None:
    session_id = str(uuid.uuid4())
    graph = build_graph()
    cl.user_session.set("session_id", session_id)
    cl.user_session.set("graph", graph)

    await cl.Message(
        content=(
            "# 🏥 IP Doctor — MBS Prepayment Attribution\n\n"
            "Ask me to explain what drove your prepayment model's forecast change "
            "using **Integrated Gradients**.\n\n"
            "**Example questions:**\n"
            "> *For CUSIP 3140GXPJ8, what drove the CPR change from January to February 2024?*\n\n"
            "> *For CUSIPs ABC123 and DEF456, show month-over-month SMM attribution for Q1 2024.*\n\n"
            "> *Compare base vs rate_shock_100bps for CUSIPs ABC123 and DEF456. "
            "Base period: Jan–Jun 2024. Shock period: Jul–Dec 2024.*\n\n"
            "---\n"
            f"🔑 **Session ID:** `{session_id}`  \n"
            "_Send a new question at any time to start a fresh analysis._"
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    session_id = cl.user_session.get("session_id")
    graph      = cl.user_session.get("graph")

    if not graph or not session_id:
        await cl.Message(
            content="⚠️ Session not initialised. Please refresh the page."
        ).send()
        return

    config: dict = {"configurable": {"thread_id": session_id}}

    # Each new user question starts a fresh LangGraph thread
    # (new session_id so we don't replay old checkpointed state)
    new_session_id = str(uuid.uuid4())
    cl.user_session.set("session_id", new_session_id)
    config = {"configurable": {"thread_id": new_session_id}}

    initial_state: MBSAnalysisState = {
        "question":          message.content,
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
        "pdf_path":          None,
        "iteration_count":   0,
        "session_id":        new_session_id,
    }

    await _run_workflow(graph, initial_state, config)
