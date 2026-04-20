"""Tests for MBS multi-agent personas (Priority 10)."""
import pytest

from rdagent.scenarios.mbs_prepayment.personas import (
    DEFAULT_ROUTING,
    LoopNode,
    Persona,
    PersonaKind,
    PersonaRouter,
    build_messages,
    get_persona,
    invoke_persona,
)


# ---------------------------------------------------------------------------
# Persona definitions
# ---------------------------------------------------------------------------


@pytest.mark.offline
def test_all_persona_kinds_have_definitions():
    for kind in PersonaKind:
        p = get_persona(kind)
        assert isinstance(p, Persona)
        assert p.kind == kind
        assert p.system_prompt  # non-empty
        assert p.model_hint
        assert 0.0 <= p.temperature <= 2.0


@pytest.mark.offline
def test_quant_researcher_prompt_mentions_mbs_concepts():
    p = get_persona(PersonaKind.QUANT_RESEARCHER)
    prompt = p.system_prompt.lower()
    assert "richard-roll" in prompt or "s-curve" in prompt
    assert (
        "refi incentive" in prompt
        or "refinance incentive" in prompt
        or "avg_prop_refi_incentive" in prompt
    )
    assert "smm_decimal" in prompt or "smm" in prompt


@pytest.mark.offline
def test_ml_engineer_prompt_contains_leakage_guardrails():
    p = get_persona(PersonaKind.ML_ENGINEER)
    prompt = p.system_prompt.lower()
    assert "cusip" in prompt
    assert "fh_effdt" in prompt
    assert "shuffle" in prompt or "look-ahead" in prompt or "temporal" in prompt


@pytest.mark.offline
def test_model_validator_prompt_is_skeptical():
    p = get_persona(PersonaKind.MODEL_VALIDATOR)
    prompt = p.system_prompt.lower()
    assert "skeptic" in prompt or "reject" in prompt or "review" in prompt
    assert "per-coupon" in prompt or "coupon" in prompt


@pytest.mark.offline
def test_data_engineer_prompt_is_minimal_fix_focused():
    p = get_persona(PersonaKind.DATA_ENGINEER)
    prompt = p.system_prompt.lower()
    assert "minimal" in prompt or "smallest" in prompt
    assert "debug" in prompt or "fix" in prompt


@pytest.mark.offline
def test_data_engineer_uses_a_cheaper_model_hint():
    """Debugging is the high-volume call site → cheap/fast model preferred."""
    debug = get_persona(PersonaKind.DATA_ENGINEER)
    research = get_persona(PersonaKind.QUANT_RESEARCHER)
    # A clear hint of cost/latency separation — they should NOT be identical
    assert debug.model_hint != research.model_hint


@pytest.mark.offline
def test_debug_temperature_lower_than_research():
    debug = get_persona(PersonaKind.DATA_ENGINEER)
    research = get_persona(PersonaKind.QUANT_RESEARCHER)
    assert debug.temperature < research.temperature


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


@pytest.mark.offline
def test_default_routing_has_every_loop_node():
    for node in LoopNode:
        assert node in DEFAULT_ROUTING


@pytest.mark.offline
def test_default_routing_matches_customization_spec():
    """Spec: hypothesis→quant, coding→ML engineer, feedback→validator, debug→data engineer."""
    assert DEFAULT_ROUTING[LoopNode.HYPOTHESIS_GEN] == PersonaKind.QUANT_RESEARCHER
    assert DEFAULT_ROUTING[LoopNode.CODE_GEN] == PersonaKind.ML_ENGINEER
    assert DEFAULT_ROUTING[LoopNode.FEEDBACK_EVAL] == PersonaKind.MODEL_VALIDATOR
    assert DEFAULT_ROUTING[LoopNode.DEBUG] == PersonaKind.DATA_ENGINEER
    assert DEFAULT_ROUTING[LoopNode.CODE_FIX] == PersonaKind.DATA_ENGINEER


@pytest.mark.offline
def test_router_returns_default_persona():
    router = PersonaRouter()
    persona = router.route(LoopNode.HYPOTHESIS_GEN)
    assert persona.kind == PersonaKind.QUANT_RESEARCHER


@pytest.mark.offline
def test_router_applies_override():
    router = PersonaRouter()
    router.set(LoopNode.HYPOTHESIS_GEN, PersonaKind.MODEL_VALIDATOR)
    persona = router.route(LoopNode.HYPOTHESIS_GEN)
    assert persona.kind == PersonaKind.MODEL_VALIDATOR
    # Unchanged nodes still use default
    assert router.route(LoopNode.CODE_GEN).kind == PersonaKind.ML_ENGINEER


@pytest.mark.offline
def test_router_override_via_constructor():
    router = PersonaRouter(overrides={LoopNode.CODE_GEN: PersonaKind.DATA_ENGINEER})
    assert router.route(LoopNode.CODE_GEN).kind == PersonaKind.DATA_ENGINEER


@pytest.mark.offline
def test_routing_table_is_human_readable():
    router = PersonaRouter()
    table = router.routing_table()
    assert len(table) == len(LoopNode)
    # Display names should NOT be raw enum keys
    for name in table.values():
        assert " " in name or "(" in name  # "Quant Researcher (MBS ...)" etc.


# ---------------------------------------------------------------------------
# Message building + invocation (with mock backend)
# ---------------------------------------------------------------------------


@pytest.mark.offline
def test_build_messages_has_system_and_user():
    persona = get_persona(PersonaKind.QUANT_RESEARCHER)
    messages = build_messages(persona, "What feature should I add?")
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[0]["content"] == persona.system_prompt
    assert messages[1]["content"] == "What feature should I add?"


class _MockBackend:
    """Stand-in APIBackend — captures every call for inspection."""

    def __init__(self):
        self.calls = []

    def build_messages_and_create_chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        return '{"result": "ok"}'


@pytest.mark.offline
def test_invoke_persona_with_loop_node_uses_routed_persona():
    backend = _MockBackend()
    router = PersonaRouter()
    result = invoke_persona(
        LoopNode.HYPOTHESIS_GEN,
        user_prompt="Propose a baseline model.",
        router=router,
        backend=backend,
    )
    assert result == '{"result": "ok"}'
    assert len(backend.calls) == 1
    call = backend.calls[0]
    assert call["system_prompt"] == get_persona(PersonaKind.QUANT_RESEARCHER).system_prompt
    assert call["user_prompt"] == "Propose a baseline model."
    assert call["json_mode"] is True  # quant_researcher is json
    assert call["temperature"] == get_persona(PersonaKind.QUANT_RESEARCHER).temperature


@pytest.mark.offline
def test_invoke_persona_with_kind_bypasses_routing():
    backend = _MockBackend()
    invoke_persona(
        PersonaKind.DATA_ENGINEER,
        user_prompt="Fix this NaN error.",
        backend=backend,
    )
    call = backend.calls[0]
    assert call["system_prompt"] == get_persona(PersonaKind.DATA_ENGINEER).system_prompt
    assert call["json_mode"] is False  # data_engineer is text


@pytest.mark.offline
def test_invoke_persona_override_model_and_temperature():
    backend = _MockBackend()
    invoke_persona(
        PersonaKind.QUANT_RESEARCHER,
        user_prompt="...",
        model_override="gpt-5-imaginary",
        temperature_override=0.05,
        backend=backend,
    )
    call = backend.calls[0]
    assert call["chat_model"] == "gpt-5-imaginary"
    assert call["temperature"] == 0.05


@pytest.mark.offline
def test_invoke_persona_json_mode_override():
    backend = _MockBackend()
    invoke_persona(
        PersonaKind.ML_ENGINEER,   # normally text
        user_prompt="...",
        json_mode=True,
        backend=backend,
    )
    assert backend.calls[0]["json_mode"] is True
