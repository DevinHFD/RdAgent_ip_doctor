"""
graph.py — LangGraph StateGraph assembly for the MBS prepayment attribution workflow.

Topology:
    question_parser → planner → code_generator → code_validator
                                        ↑               |
                                        └── (invalid) ──┘
                                                        |
                                                   (valid)
                                                        ↓
                                                   executor
                                                  /        \\
                                           (error)        (success)
                                               ↓               ↓
                                           debugger     human_reviewer [INTERRUPT]
                                          /       \\         /           \\
                                  (retry<3)   (give up) (reject)     (approve)
                                      ↓           ↓        ↓               ↓
                                code_validator  human   planner         reporter → END
                                              reviewer
"""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from .conf import MBS_SETTINGS
from .nodes.code_generator import code_generator_node
from .nodes.code_validator import code_validator_node
from .nodes.debugger import debugger_node
from .nodes.executor import executor_node
from .nodes.human_reviewer import human_reviewer_node
from .nodes.planner import planner_node
from .nodes.question_parser import question_parser_node
from .nodes.reporter import reporter_node
from .state import MBSAnalysisState


# ---------------------------------------------------------------------------
# Conditional edge functions (pure, no LLM)
# ---------------------------------------------------------------------------


def _after_validator(state: MBSAnalysisState) -> str:
    """Route to executor if code is valid, else back to code_generator for LLM fix."""
    if state.get("code_valid"):
        return "executor"
    return "code_generator"


def _after_executor(state: MBSAnalysisState) -> str:
    """Route to human_reviewer on success, debugger on failure."""
    if state.get("execution_error") is None and state.get("execution_result") is not None:
        return "human_reviewer"
    return "debugger"


def _after_debugger(state: MBSAnalysisState) -> str:
    """
    After a debug attempt:
    - If under the attempt limit, route back through code_validator before re-execution.
    - If at/over the limit, give up and route to human_reviewer so the human sees the error.
    """
    if state.get("debug_attempts", 0) < MBS_SETTINGS.max_debug_attempts:
        return "code_validator"
    return "human_reviewer"


def _after_human_reviewer(state: MBSAnalysisState) -> str:
    """
    Route to reporter on approve, back to planner on reject.
    Max iteration guard prevents infinite reject loops.
    """
    iteration = state.get("iteration_count", 1)
    if iteration > MBS_SETTINGS.max_iterations:
        # Hard cap: force report regardless of feedback
        return "reporter"

    fb = state.get("human_feedback") or {}
    if fb.get("decision") == "approve":
        return "reporter"
    return "planner"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_graph() -> "CompiledStateGraph":  # noqa: F821
    """
    Assemble and compile the MBS attribution LangGraph.

    Returns a compiled graph with:
    - MemorySaver checkpointer (enables interrupt/resume and multi-turn sessions)
    - interrupt_before=["human_reviewer"] so callers can surface the review payload

    Usage:
        graph = build_graph()
        config = {"configurable": {"thread_id": session_id}}
        result = graph.invoke(initial_state, config=config)
        # If graph.get_state(config).next is non-empty → human review needed
        result = graph.invoke(Command(resume=feedback), config=config)
    """
    builder = StateGraph(MBSAnalysisState)

    # Register all nodes
    builder.add_node("question_parser", question_parser_node)
    builder.add_node("planner", planner_node)
    builder.add_node("code_generator", code_generator_node)
    builder.add_node("code_validator", code_validator_node)
    builder.add_node("executor", executor_node)
    builder.add_node("debugger", debugger_node)
    builder.add_node("human_reviewer", human_reviewer_node)
    builder.add_node("reporter", reporter_node)

    # Entry point
    builder.set_entry_point("question_parser")

    # Linear edges
    builder.add_edge("question_parser", "planner")
    builder.add_edge("planner", "code_generator")
    builder.add_edge("code_generator", "code_validator")

    # Conditional: after validation
    builder.add_conditional_edges(
        "code_validator",
        _after_validator,
        {"executor": "executor", "code_generator": "code_generator"},
    )

    # Conditional: after execution
    builder.add_conditional_edges(
        "executor",
        _after_executor,
        {"human_reviewer": "human_reviewer", "debugger": "debugger"},
    )

    # Conditional: after debug attempt
    builder.add_conditional_edges(
        "debugger",
        _after_debugger,
        {"code_validator": "code_validator", "human_reviewer": "human_reviewer"},
    )

    # Conditional: after human review
    builder.add_conditional_edges(
        "human_reviewer",
        _after_human_reviewer,
        {"reporter": "reporter", "planner": "planner"},
    )

    # Terminal edge
    builder.add_edge("reporter", END)

    # Compile with persistent checkpointer and human interrupt
    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)
