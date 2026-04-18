"""MBS Multi-Agent Personas — Priority 10.

Implements Direction #10 (Multi-Agent Specialization). The stock RD-Agent
loop uses a single LLM persona throughout; for MBS, different phases of
an iteration benefit from different expertise:

    - Hypothesis generation → *quant researcher*: thinks in prepayment
      theory (Richard-Roll decomposition, S-curve refi response, burnout
      halflife, seasoning ramps), economic intuition, and structural
      modelling.
    - Coding → *ML engineer*: thinks in pandas/PyTorch, numerical
      stability, panel-data correctness, efficient data handling.
    - Feedback / evaluation → *model validator*: thinks in backtesting,
      out-of-sample robustness, regime sensitivity, and regulatory
      review. Skeptical by default.
    - Debugging → *data engineer*: thinks in data quality, types,
      missing values, performance bottlenecks.

This module defines the `Persona` objects (system prompt + suggested
model + temperature), a `PersonaRouter` that maps a loop-phase +
iteration node to the right persona, and a helper `invoke_persona()`
that wraps the RD-Agent `APIBackend` call so every LLM call in the MBS
loop can be a one-liner:

    response = invoke_persona(PersonaKind.QUANT_RESEARCHER,
                              user_prompt=...,
                              json_mode=True)

If `APIBackend` is unavailable at import time (e.g., in a CI environment
without the OpenAI SDK), `invoke_persona` falls back to raising a clear
error only when actually called — so unit tests can still import this
module and assert routing behavior without needing network access.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Persona definitions
# ---------------------------------------------------------------------------


class PersonaKind(str, Enum):
    QUANT_RESEARCHER = "quant_researcher"
    ML_ENGINEER = "ml_engineer"
    MODEL_VALIDATOR = "model_validator"
    DATA_ENGINEER = "data_engineer"


@dataclass(frozen=True)
class Persona:
    """A named LLM persona with its system prompt and call-time defaults.

    `model_hint` is a *preference*, not a hard binding — the caller can
    override it per call. The goal is to route hypothesis generation to
    a more capable model (where quality matters most) and debugging to
    a faster/cheaper model (where iteration count matters more).
    """

    kind: PersonaKind
    display_name: str
    system_prompt: str
    model_hint: str
    temperature: float
    response_format: str = "json"   # "json" or "text"

    def as_system_message(self) -> dict[str, str]:
        return {"role": "system", "content": self.system_prompt}


# System prompts intentionally mirror the MBS-specific role work from
# Priority 3 (prompts.yaml) but specialize per call site.

QUANT_RESEARCHER = Persona(
    kind=PersonaKind.QUANT_RESEARCHER,
    display_name="Quant Researcher (MBS Prepayment)",
    system_prompt=(
        "You are a senior quantitative researcher specializing in MBS prepayment "
        "modelling. You think in the Richard-Roll decomposition: turnover, "
        "refinancing (S-curve + burnout), curtailment, defaults. You reason "
        "about rate_incentive = WAC - current_mortgage_rate, the S-curve "
        "inflection in the 50-150 bps range, burnout halflife in the 8-18 "
        "month range, seasoning ramps, and seasonal refi patterns. You reject "
        "generic ML proposals (XGBoost with default hyperparameters, arbitrary "
        "feature stacks) and require every hypothesis to be grounded in an "
        "economic mechanism. You require every hypothesis to predict per-coupon "
        "bucket RMSE impact before acceptance. Your target metric is RMSE of "
        "SMM_DECIMAL (target ∈ [0, 1]) on a temporal holdout keyed on fh_effdt. "
        "You never propose hypotheses that would break the temporal split, "
        "or introduce look-ahead features."
    ),
    model_hint="gpt-4-turbo",
    temperature=0.7,
    response_format="json",
)

ML_ENGINEER = Persona(
    kind=PersonaKind.ML_ENGINEER,
    display_name="ML Engineer (Panel Data / PyTorch)",
    system_prompt=(
        "You are an ML engineer implementing MBS prepayment models. You "
        "write clean pandas and PyTorch code. You ALWAYS preserve the "
        "(cusip, fh_effdt) panel structure; you NEVER shuffle time series "
        "or use sklearn.model_selection.train_test_split with shuffle=True; "
        "you NEVER normalize the SMM_DECIMAL target variable; you NEVER use "
        "same-fh_effdt macro data for features (look-ahead bias); you "
        "ALWAYS clip predictions to [0, 1]. You implement the MBSDataContract "
        "from rdagent.scenarios.mbs_prepayment.scaffold and conform to the "
        "MBSEvaluationHarness scorecard shape. You prefer vectorized numpy/pandas "
        "over Python loops. Your code is numerically stable (watch for log(0), "
        "divide-by-zero, overflow in sigmoid). You write concise code; you do "
        "not add speculative abstractions, decorative comments, or unused "
        "helper functions."
    ),
    model_hint="gpt-4-turbo",
    temperature=0.2,
    response_format="text",
)

MODEL_VALIDATOR = Persona(
    kind=PersonaKind.MODEL_VALIDATOR,
    display_name="Model Validator (Skeptical Reviewer)",
    system_prompt=(
        "You are a model validator reviewing an MBS prepayment experiment. "
        "Your job is skeptical scrutiny, not cheerleading. You compare the "
        "current scorecard to the SOTA scorecard across ALL dimensions: "
        "overall RMSE, per-coupon-bucket RMSE (especially high-coupon buckets "
        "where refi risk concentrates), S-curve R², inflection point, "
        "regime transition RMSE (2013/2020/2022), and structural properties. "
        "An experiment that improves overall RMSE while degrading per-coupon "
        "RMSE uniformity or regime robustness is a REJECT. You explicitly call out economic-prior violations (e.g., "
        "unemployment with positive attribution on SMM_DECIMAL, burnout_index "
        "with positive sign). You provide specific, falsifiable criticism."
    ),
    model_hint="gpt-4-turbo",
    temperature=0.3,
    response_format="json",
)

DATA_ENGINEER = Persona(
    kind=PersonaKind.DATA_ENGINEER,
    display_name="Data Engineer (Debugger)",
    system_prompt=(
        "You are a data engineer debugging failing MBS prepayment code. "
        "You focus on the minimal fix that restores execution: missing "
        "columns, type mismatches (datetime vs string for fh_effdt, float vs "
        "categorical for coupon), NaN/Inf handling, memory pressure on large "
        "CUSIP panels, index alignment bugs, off-by-one in temporal splits. "
        "You do NOT refactor for style, NOT add new features, NOT change the "
        "model architecture. You return the smallest diff that makes the code "
        "run AND preserves the original experimental intent. You are fast and "
        "pragmatic, not thorough — debugging iterations are cheap and a minimal "
        "fix is better than a speculative rewrite."
    ),
    model_hint="gpt-4o-mini",
    temperature=0.1,
    response_format="text",
)


PERSONAS: dict[PersonaKind, Persona] = {
    PersonaKind.QUANT_RESEARCHER: QUANT_RESEARCHER,
    PersonaKind.ML_ENGINEER: ML_ENGINEER,
    PersonaKind.MODEL_VALIDATOR: MODEL_VALIDATOR,
    PersonaKind.DATA_ENGINEER: DATA_ENGINEER,
}


def get_persona(kind: PersonaKind) -> Persona:
    return PERSONAS[kind]


# ---------------------------------------------------------------------------
# Persona routing: which persona handles which loop-phase + node
# ---------------------------------------------------------------------------


class LoopNode(str, Enum):
    """Node labels for each LLM-calling point in the MBS iteration.

    These are the places in the RD-Agent loop where a system prompt
    gets injected. The router maps each to a persona.
    """

    HYPOTHESIS_GEN = "hypothesis_gen"
    HYPOTHESIS_SELECT = "hypothesis_select"
    HYPOTHESIS_CRITIQUE = "hypothesis_critique"
    HYPOTHESIS_REWRITE = "hypothesis_rewrite"
    COMPONENT_GEN = "component_gen"
    CODE_GEN = "code_gen"
    CODE_FIX = "code_fix"
    FEEDBACK_EVAL = "feedback_eval"
    DOMAIN_VALIDATION_EXPLAIN = "domain_validation_explain"
    DEBUG = "debug"


#: Canonical mapping of loop nodes to personas. A downstream config can
#: override individual entries (e.g., route hypothesis_rewrite to
#: MODEL_VALIDATOR instead) by passing an override dict into PersonaRouter.
DEFAULT_ROUTING: dict[LoopNode, PersonaKind] = {
    LoopNode.HYPOTHESIS_GEN: PersonaKind.QUANT_RESEARCHER,
    LoopNode.HYPOTHESIS_SELECT: PersonaKind.QUANT_RESEARCHER,
    LoopNode.HYPOTHESIS_CRITIQUE: PersonaKind.MODEL_VALIDATOR,
    LoopNode.HYPOTHESIS_REWRITE: PersonaKind.QUANT_RESEARCHER,
    LoopNode.COMPONENT_GEN: PersonaKind.QUANT_RESEARCHER,
    LoopNode.CODE_GEN: PersonaKind.ML_ENGINEER,
    LoopNode.CODE_FIX: PersonaKind.DATA_ENGINEER,
    LoopNode.FEEDBACK_EVAL: PersonaKind.MODEL_VALIDATOR,
    LoopNode.DOMAIN_VALIDATION_EXPLAIN: PersonaKind.MODEL_VALIDATOR,
    LoopNode.DEBUG: PersonaKind.DATA_ENGINEER,
}


@dataclass
class PersonaRouter:
    """Resolves loop-node labels to `Persona` instances.

    The router exists so that a single config object can drive all LLM
    calls in the loop. Swap in a different router at loop init and
    every call site picks up the new persona assignments without
    individual code changes.
    """

    overrides: dict[LoopNode, PersonaKind] = field(default_factory=dict)

    def route(self, node: LoopNode) -> Persona:
        kind = self.overrides.get(node, DEFAULT_ROUTING[node])
        return PERSONAS[kind]

    def set(self, node: LoopNode, kind: PersonaKind) -> None:
        self.overrides[node] = kind

    def routing_table(self) -> dict[str, str]:
        """Return a flat `{node → persona display_name}` map for logging."""
        return {
            n.value: PERSONAS[self.overrides.get(n, DEFAULT_ROUTING[n])].display_name
            for n in LoopNode
        }


# ---------------------------------------------------------------------------
# Invocation helper: one-liner LLM call with the right persona attached
# ---------------------------------------------------------------------------


def build_messages(persona: Persona, user_prompt: str) -> list[dict[str, str]]:
    """Construct the OpenAI-style messages list for a persona invocation."""
    return [
        persona.as_system_message(),
        {"role": "user", "content": user_prompt},
    ]


def invoke_persona(
    node_or_kind: LoopNode | PersonaKind,
    user_prompt: str,
    *,
    router: PersonaRouter | None = None,
    model_override: str | None = None,
    temperature_override: float | None = None,
    json_mode: bool | None = None,
    backend: Any | None = None,
) -> str:
    """Call an LLM with the persona resolved from `node_or_kind`.

    Parameters
    ----------
    node_or_kind:
        Either a `LoopNode` (routed through the router) or a direct
        `PersonaKind`. Direct kinds bypass routing — useful for tests
        and for one-off calls from outside the main loop.
    user_prompt:
        The user-role message content.
    router:
        Optional `PersonaRouter`. Required only when `node_or_kind` is
        a `LoopNode`; ignored when it's a `PersonaKind`.
    model_override, temperature_override:
        Runtime overrides for the persona's model_hint and temperature.
    json_mode:
        When None, uses `persona.response_format == "json"`.
    backend:
        Optional `APIBackend` instance. If omitted, the function tries
        to import and instantiate one. This is lazy so unit tests can
        import this module without an LLM backend available.
    """
    if isinstance(node_or_kind, LoopNode):
        persona = (router or PersonaRouter()).route(node_or_kind)
    else:
        persona = PERSONAS[node_or_kind]

    model = model_override or persona.model_hint
    temperature = persona.temperature if temperature_override is None else temperature_override
    use_json = persona.response_format == "json" if json_mode is None else json_mode

    if backend is None:
        try:
            from rdagent.oai.llm_utils import APIBackend  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "invoke_persona() requires rdagent.oai.llm_utils.APIBackend — "
                "install the `oai` extra or pass a custom `backend` argument."
            ) from exc
        backend = APIBackend()

    return backend.build_messages_and_create_chat_completion(
        user_prompt=user_prompt,
        system_prompt=persona.system_prompt,
        json_mode=use_json,
        chat_model=model,
        temperature=temperature,
    )
