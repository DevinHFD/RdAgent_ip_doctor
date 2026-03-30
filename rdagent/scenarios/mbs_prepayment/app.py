"""
app.py — CLI entry point for the MBS prepayment attribution workflow.

Usage (new session):
    python -m rdagent.scenarios.mbs_prepayment.app \\
        "For CUSIP XYZ123, show what drove the CPR change from January to February."

Resume an interrupted session:
    python -m rdagent.scenarios.mbs_prepayment.app \\
        --session-id <uuid> \\
        "For CUSIP XYZ123, show what drove the CPR change from January to February."

Skip human review (automated / testing):
    python -m rdagent.scenarios.mbs_prepayment.app \\
        --skip-review \\
        "For CUSIP XYZ123, ..."

Environment variables (override defaults from conf.py):
    MBS_MODEL_CHECKPOINT_DIR, MBS_DATA_DIR, MBS_OUTPUT_DIR, MBS_SCALER_PATH,
    MBS_IG_BASELINE_STRATEGY, MBS_IG_N_STEPS, MBS_IG_TARGET_OUTPUT,
    MBS_MAX_DEBUG_ATTEMPTS, MBS_MAX_ITERATIONS, MBS_SKIP_HUMAN_REVIEW
"""

import json
import sys
import uuid
from typing import Optional

import typer
from langgraph.types import Command

from rdagent.log import rdagent_logger as logger

from .graph import build_graph
from .state import MBSAnalysisState

app = typer.Typer(
    name="mbs-prepayment",
    help="MBS prepayment attribution analysis via Integrated Gradients.",
    add_completion=False,
)


# ---------------------------------------------------------------------------
# Human-in-the-loop CLI interaction
# ---------------------------------------------------------------------------


def _print_review_payload(payload: dict) -> None:
    """Pretty-print the review payload surfaced by human_reviewer_node."""
    sep = "=" * 70
    print(f"\n{sep}")
    print("  HUMAN REVIEW REQUIRED")
    print(sep)
    print(f"Question  : {payload.get('question', '')}")
    print(f"Iteration : {payload.get('iteration', '?')}")
    print()
    print("--- Analysis Plan ---")
    print(payload.get("plan_summary", "(none)"))
    print()
    print("--- Execution Summary ---")
    print(payload.get("execution_summary", "(none)"))
    if payload.get("error_note"):
        print()
        print("--- Error Note ---")
        print(payload["error_note"])
    print()
    print(payload.get("instructions", ""))
    print(sep)


def _collect_cli_feedback() -> dict:
    """Read JSON feedback (or shorthand) from stdin."""
    while True:
        try:
            raw = input("\nYour decision > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nNo input received; defaulting to approve.")
            return {"decision": "approve"}

        if not raw:
            continue

        if raw.lower() in ("approve", "reject"):
            if raw.lower() == "reject":
                what = input("What is wrong? > ").strip()
                change = input("Suggested change? > ").strip()
                cusips_raw = input("Focus CUSIPs (comma-separated, or blank)? > ").strip()
                focus = [c.strip() for c in cusips_raw.split(",") if c.strip()]
                return {
                    "decision": "reject",
                    "what_is_wrong": what,
                    "suggested_change": change,
                    "focus_cusips": focus or None,
                }
            return {"decision": "approve"}

        # Try JSON parse
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            print("Could not parse input as JSON. Type 'approve', 'reject', or valid JSON.")


# ---------------------------------------------------------------------------
# Graph invocation helpers
# ---------------------------------------------------------------------------


def _get_interrupt_payload(graph, config: dict) -> dict | None:
    """
    After an invoke, check whether the graph is suspended at human_reviewer.
    Returns the interrupt payload dict if suspended, else None.
    """
    state = graph.get_state(config)
    if not state.next:
        return None
    # The interrupt payload is stored in state.tasks[0].interrupts[0].value
    tasks = state.tasks
    if tasks and tasks[0].interrupts:
        return tasks[0].interrupts[0].value
    return None


def run_workflow(question: str, session_id: str, skip_review: bool = False) -> None:
    """Execute the full workflow, handling human review loops via CLI."""
    graph = build_graph()
    config = {"configurable": {"thread_id": session_id}}

    initial_state: MBSAnalysisState = {
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
        "iteration_count": 0,
        "session_id": session_id,
    }

    logger.info(f"Starting MBS attribution workflow. session_id={session_id}")
    print(f"\nSession ID: {session_id}")
    print(f"Question  : {question}\n")

    # First invocation
    graph.invoke(initial_state, config=config)

    # Human review loop
    while True:
        payload = _get_interrupt_payload(graph, config)
        if payload is None:
            break  # graph reached END

        if skip_review:
            # Automated mode: auto-approve
            logger.info("skip_review=True; auto-approving.")
            feedback = {"decision": "approve"}
        else:
            _print_review_payload(payload)
            feedback = _collect_cli_feedback()

        graph.invoke(Command(resume=feedback), config=config)

    # Retrieve final state
    final_state = graph.get_state(config).values
    report = final_state.get("report_markdown")
    plot_paths = final_state.get("plot_paths", [])

    print("\n" + "=" * 70)
    print("  ANALYSIS REPORT")
    print("=" * 70)
    if report:
        print(report)
    else:
        print("(No report generated.)")

    if plot_paths:
        print("\nPlots saved:")
        for p in plot_paths:
            print(f"  {p}")

    print(f"\nOutput directory: {final_state.get('analysis_plan', {})}")
    logger.info("Workflow complete.")


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------


@app.command()
def main(
    question: str = typer.Argument(..., help="Natural language question about the MBS model."),
    session_id: Optional[str] = typer.Option(
        None,
        "--session-id",
        "-s",
        help="Resume an existing session by providing its UUID.",
    ),
    skip_review: bool = typer.Option(
        False,
        "--skip-review",
        help="Auto-approve human review (useful for automated runs and testing).",
    ),
) -> None:
    """
    Run the MBS prepayment attribution analysis workflow.

    Accepts a natural language question such as:
      "For CUSIPs ABC123 and DEF456, what features drove the January-February CPR change?"
      "Compare rate shock +100bps vs base scenario across CUSIPs XYZ789 and QRS012."
    """
    sid = session_id or str(uuid.uuid4())
    try:
        run_workflow(question=question, session_id=sid, skip_review=skip_review)
    except KeyboardInterrupt:
        print(f"\nInterrupted. Resume with: --session-id {sid}")
        sys.exit(1)


if __name__ == "__main__":
    app()
