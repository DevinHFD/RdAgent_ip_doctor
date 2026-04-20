"""Tests for MBS orchestration / phase gates (Priority 8)."""
import math

import pytest

from rdagent.scenarios.mbs_prepayment.memory import (
    MBSMemory,
    ModelProperties,
    TraceEntry,
)
from rdagent.scenarios.mbs_prepayment.orchestration import (
    DomainValidator,
    MBSOrchestrator,
    PHASE_SPECS,
    Phase,
    PhaseGate,
    next_phase,
)
from rdagent.scenarios.mbs_prepayment.search_strategy import (
    IterationRecord,
    MBSSearchState,
)


BASELINE_SCORECARD = {
    "accuracy": {
        "overall_rmse": 0.028,
        "rmse_by_coupon_bucket": {"0.0-3.0": 0.020, "3.0-3.5": 0.022, "5.0+": 0.030},
    },
    "rate_sensitivity": {"monotonicity_spearman": 0.8, "s_curve_r2": 0.75, "inflection_point_ratio": 1.10},
    "temporal_robustness": {"regime_transition_rmse": {"2020-03-01": 0.045}},
    "structural_properties": {
        "cusip_differentiation_std": 0.004,
        "seasonality_residual_range": 0.001,
    },
}


# ---------------------------------------------------------------------------
# Phase ordering
# ---------------------------------------------------------------------------


@pytest.mark.offline
def test_phase_ordering_is_linear():
    assert next_phase(Phase.BASELINE) == Phase.RATE_RESPONSE
    assert next_phase(Phase.RATE_RESPONSE) == Phase.DYNAMICS
    assert next_phase(Phase.DYNAMICS) == Phase.MACRO_REGIME
    assert next_phase(Phase.MACRO_REGIME) == Phase.ENSEMBLE
    assert next_phase(Phase.ENSEMBLE) is None


@pytest.mark.offline
def test_phase_specs_have_allowed_components():
    for phase, spec in PHASE_SPECS.items():
        assert spec.phase == phase
        assert len(spec.allowed_components) > 0
        assert spec.iteration_budget[0] <= spec.iteration_budget[1]


# ---------------------------------------------------------------------------
# DomainValidator
# ---------------------------------------------------------------------------


@pytest.mark.offline
def test_validator_passes_on_clean_predictions():
    v = DomainValidator()
    report = v.validate(
        y_pred=[0.01, 0.02, 0.05, 0.10, 0.15],
        rate_incentive=[-50, -10, 20, 80, 150],
        training_seconds=42.0,
    )
    assert report.ok
    assert report.auto_reject_reason == ""


@pytest.mark.offline
def test_validator_rejects_out_of_range():
    v = DomainValidator()
    report = v.validate(y_pred=[0.1, 1.5, 0.2])  # 1.5 out of [0, 1]
    assert not report.ok
    assert "range" in report.auto_reject_reason.lower() or "0, 1" in report.auto_reject_reason


@pytest.mark.offline
def test_validator_rejects_nan():
    v = DomainValidator()
    report = v.validate(y_pred=[0.1, float("nan"), 0.2])
    assert not report.ok
    assert any(not c.passed for c in report.checks if c.name == "no_nan_inf")



@pytest.mark.offline
def test_validator_rejects_missing_cusips():
    v = DomainValidator()
    report = v.validate(
        y_pred=[0.01, 0.02],
        cusip_ids_train=["A", "B", "C"],
        cusip_ids_pred=["A", "B"],  # missing C
    )
    assert not report.ok
    assert any(c.name == "all_cusips_present" and not c.passed for c in report.checks)


@pytest.mark.offline
def test_validator_rejects_training_time_over_budget():
    v = DomainValidator(max_training_seconds=100)
    report = v.validate(y_pred=[0.01, 0.02], training_seconds=500)
    assert not report.ok
    assert any(c.name == "training_time_budget" and not c.passed for c in report.checks)


# ---------------------------------------------------------------------------
# PhaseGate
# ---------------------------------------------------------------------------


@pytest.mark.offline
def test_baseline_gate_passes_with_good_props():
    gate = PhaseGate()
    props = ModelProperties.from_scorecard(1, "Ridge", "RateCurveFeatures", BASELINE_SCORECARD)
    result = gate.evaluate(Phase.BASELINE, props)
    assert result.passed, result.summary


@pytest.mark.offline
def test_baseline_gate_fails_with_high_rmse():
    gate = PhaseGate(baseline_max_rmse=0.020)
    props = ModelProperties.from_scorecard(1, "Ridge", "RateCurveFeatures", BASELINE_SCORECARD)
    result = gate.evaluate(Phase.BASELINE, props)
    assert not result.passed


@pytest.mark.offline
def test_rate_response_gate_checks_s_curve_and_inflection():
    gate = PhaseGate()
    props = ModelProperties.from_scorecard(5, "LightGBM", "RateCurveFeatures", BASELINE_SCORECARD)
    result = gate.evaluate(Phase.RATE_RESPONSE, props)
    # s_curve_r2 0.75, inflection ratio 1.10 (within [1.00, 1.20]) — all good
    assert result.passed


@pytest.mark.offline
def test_rate_response_gate_fails_on_inflection_out_of_range():
    gate = PhaseGate()
    scorecard = {
        **BASELINE_SCORECARD,
        "rate_sensitivity": {
            "monotonicity_spearman": 0.8,
            "s_curve_r2": 0.75,
            "inflection_point_ratio": 3.00,  # out of [1.00, 1.20]
        },
    }
    props = ModelProperties.from_scorecard(5, "LightGBM", "RateCurveFeatures", scorecard)
    result = gate.evaluate(Phase.RATE_RESPONSE, props)
    assert not result.passed
    assert any("inflection" in c.name for c in result.criteria_results if not c.passed)


@pytest.mark.offline
def test_dynamics_gate_checks_per_coupon_rmse():
    gate = PhaseGate()
    props = ModelProperties.from_scorecard(10, "LightGBM", "PoolDynamics", BASELINE_SCORECARD)
    result = gate.evaluate(Phase.DYNAMICS, props)
    # worst coupon RMSE 0.030 < 0.035 cap → pass
    assert result.passed


@pytest.mark.offline
def test_dynamics_gate_fails_on_high_worst_coupon():
    gate = PhaseGate()
    scorecard = {
        **BASELINE_SCORECARD,
        "accuracy": {
            "overall_rmse": 0.028,
            "rmse_by_coupon_bucket": {"0.0-3.0": 0.020, "5.0+": 0.080},  # worst too high
        },
    }
    props = ModelProperties.from_scorecard(10, "LightGBM", "PoolDynamics", scorecard)
    result = gate.evaluate(Phase.DYNAMICS, props)
    assert not result.passed


@pytest.mark.offline
def test_macro_regime_gate_ratio_check():
    gate = PhaseGate(macro_regime_transition_ratio=2.0)
    # overall=0.028, regime=0.045 → ratio 1.6 < 2.0 → pass
    props = ModelProperties.from_scorecard(16, "LightGBM", "MacroFeatures", BASELINE_SCORECARD)
    result = gate.evaluate(Phase.MACRO_REGIME, props)
    assert result.passed


@pytest.mark.offline
def test_macro_regime_gate_fails_on_bad_ratio():
    gate = PhaseGate(macro_regime_transition_ratio=1.0)
    props = ModelProperties.from_scorecard(16, "LightGBM", "MacroFeatures", BASELINE_SCORECARD)
    result = gate.evaluate(Phase.MACRO_REGIME, props)
    assert not result.passed


@pytest.mark.offline
def test_gate_without_sota_returns_fail():
    gate = PhaseGate()
    result = gate.evaluate(Phase.BASELINE, None)
    assert not result.passed
    assert "SOTA" in result.summary or "gate" in result.summary.lower()


# ---------------------------------------------------------------------------
# MBSOrchestrator
# ---------------------------------------------------------------------------


def _seed_memory_with_baseline(memory: MBSMemory) -> None:
    memory.append_entry(TraceEntry(
        iteration=1,
        component="RateCurveFeatures",
        hypothesis="Ridge baseline with rate_incentive",
        code_change_summary="ridge.fit(X, y)",
        decision="accept",
        success=True,
        properties=ModelProperties.from_scorecard(1, "Ridge", "RateCurveFeatures", BASELINE_SCORECARD),
    ))


@pytest.mark.offline
def test_orchestrator_iteration_constraints_intersect_phase_allowlist(tmp_path):
    memory = MBSMemory(memory_path=tmp_path / "mem.json")
    state = MBSSearchState()
    orch = MBSOrchestrator(memory=memory, search_state=state, current_phase=Phase.BASELINE)
    filt = orch.iteration_constraints()

    phase_allowed = set(PHASE_SPECS[Phase.BASELINE].allowed_components)
    # Anything allowed must be in the phase allowlist
    for c in filt.allowed_components:
        assert c in phase_allowed
    # PoolDynamics isn't in the baseline phase allowlist
    assert "PoolDynamics" not in filt.allowed_components
    assert "baseline" in filt.guidance.lower()


@pytest.mark.offline
def test_orchestrator_advance_phase_blocked_without_passing_gate(tmp_path):
    memory = MBSMemory(memory_path=tmp_path / "mem.json")
    state = MBSSearchState()
    orch = MBSOrchestrator(memory=memory, search_state=state, current_phase=Phase.BASELINE)
    # No SOTA yet → gate fails → no advance
    assert orch.advance_phase() is None
    assert orch.current_phase == Phase.BASELINE


@pytest.mark.offline
def test_orchestrator_advances_after_gate_passes(tmp_path):
    memory = MBSMemory(memory_path=tmp_path / "mem.json")
    _seed_memory_with_baseline(memory)
    state = MBSSearchState()
    state.append(IterationRecord(
        iteration=1,
        component_touched="RateCurveFeatures",
        overall_rmse=0.028,
        has_rate_incentive=True,
        architecture_family="Ridge",
        success=True,
    ))
    orch = MBSOrchestrator(memory=memory, search_state=state, current_phase=Phase.BASELINE)
    new = orch.advance_phase()
    assert new == Phase.RATE_RESPONSE
    assert orch.current_phase == Phase.RATE_RESPONSE


@pytest.mark.offline
def test_orchestrator_build_review_payload_contains_recommendation(tmp_path):
    memory = MBSMemory(memory_path=tmp_path / "mem.json")
    _seed_memory_with_baseline(memory)
    state = MBSSearchState()
    orch = MBSOrchestrator(memory=memory, search_state=state, current_phase=Phase.BASELINE)
    payload = orch.build_phase_review_payload()
    d = payload.to_dict()
    assert d["current_phase"] == "baseline"
    assert d["proposed_next_phase"] == "rate_response"
    assert "recommendation" in d
    assert d["gate_result"]["passed"] is True
    assert d["model_properties"] is not None


@pytest.mark.offline
def test_orchestrator_review_payload_when_gate_fails(tmp_path):
    memory = MBSMemory(memory_path=tmp_path / "mem.json")
    # Seed with a BAD model (RMSE too high)
    bad_scorecard = {
        **BASELINE_SCORECARD,
        "accuracy": {"overall_rmse": 0.200, "rmse_by_coupon_bucket": {}},
    }
    memory.append_entry(TraceEntry(
        iteration=1, component="RateCurveFeatures",
        hypothesis="bad baseline", code_change_summary="",
        decision="accept", success=True,
        properties=ModelProperties.from_scorecard(1, "Ridge", "RateCurveFeatures", bad_scorecard),
    ))
    state = MBSSearchState()
    orch = MBSOrchestrator(memory=memory, search_state=state, current_phase=Phase.BASELINE)
    payload = orch.build_phase_review_payload()
    assert payload.proposed_next_phase is None
    assert not payload.gate_result.passed
    assert "not passed" in payload.recommendation.lower() or "stay" in payload.recommendation.lower()


@pytest.mark.offline
def test_orchestrator_override_phase(tmp_path):
    memory = MBSMemory(memory_path=tmp_path / "mem.json")
    state = MBSSearchState()
    orch = MBSOrchestrator(memory=memory, search_state=state, current_phase=Phase.BASELINE)
    orch.override_phase(Phase.ENSEMBLE)
    assert orch.current_phase == Phase.ENSEMBLE


@pytest.mark.offline
def test_orchestrator_final_phase_stays_final(tmp_path):
    memory = MBSMemory(memory_path=tmp_path / "mem.json")
    # Seed with SOTA satisfying ensemble gate
    good_scorecard = {
        **BASELINE_SCORECARD,
        "structural_properties": {"cusip_differentiation_std": 0.010, "seasonality_residual_range": 0.002},
    }
    memory.append_entry(TraceEntry(
        iteration=20, component="Ensemble",
        hypothesis="ensemble 3 models", code_change_summary="",
        decision="accept", success=True,
        properties=ModelProperties.from_scorecard(20, "Ensemble", "Ensemble", good_scorecard),
    ))
    state = MBSSearchState()
    orch = MBSOrchestrator(memory=memory, search_state=state, current_phase=Phase.ENSEMBLE)
    payload = orch.build_phase_review_payload()
    # Final phase: no next phase even when gate passes
    assert payload.proposed_next_phase is None
    assert "final" in payload.recommendation.lower() or "freez" in payload.recommendation.lower()
