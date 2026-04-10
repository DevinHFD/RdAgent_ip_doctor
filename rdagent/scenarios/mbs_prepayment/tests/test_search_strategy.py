"""Tests for MBS curriculum-aware search strategy (Priority 6)."""
import pytest

from rdagent.scenarios.mbs_prepayment.search_strategy import (
    ExplorationMode,
    HypothesisFilter,
    IterationRecord,
    MBSSearchState,
    decide_next_iteration,
    format_filter_for_prompt,
)


@pytest.mark.offline
def test_initial_state_blocks_dependent_components():
    state = MBSSearchState()
    filt = decide_next_iteration(state)
    assert "PoolDynamics" in filt.blocked_components
    assert "Ensemble" in filt.blocked_components
    # Initial-state components with no prereqs should be allowed
    assert "DataLoader" in filt.allowed_components
    assert "RateCurveFeatures" in filt.allowed_components


@pytest.mark.offline
def test_initial_mode_is_exploration():
    state = MBSSearchState()
    filt = decide_next_iteration(state)
    assert filt.mode == ExplorationMode.EXPLORATION
    assert "baseline" in filt.guidance.lower()


@pytest.mark.offline
def test_after_rate_incentive_added_pool_dynamics_still_blocked_without_monotonicity():
    state = MBSSearchState()
    state.append(IterationRecord(
        iteration=1,
        component_touched="RateCurveFeatures",
        overall_rmse=0.03,
        rate_sensitivity_monotonic=False,
        has_rate_incentive=True,
        architecture_family="Ridge",
        success=True,
    ))
    filt = decide_next_iteration(state)
    # PoolDynamics needs BOTH rate_incentive AND monotonicity
    assert "PoolDynamics" in filt.blocked_components


@pytest.mark.offline
def test_after_monotonic_rate_sensitivity_pool_dynamics_allowed():
    state = MBSSearchState()
    state.append(IterationRecord(
        iteration=1,
        component_touched="RateCurveFeatures",
        overall_rmse=0.03,
        rate_sensitivity_monotonic=True,
        has_rate_incentive=True,
        architecture_family="Ridge",
        success=True,
    ))
    filt = decide_next_iteration(state)
    assert "PoolDynamics" in filt.allowed_components


@pytest.mark.offline
def test_ensemble_blocked_until_three_distinct_architectures():
    state = MBSSearchState()
    for i, arch in enumerate(["Ridge", "LightGBM"], start=1):
        state.append(IterationRecord(
            iteration=i,
            component_touched="PrepaymentModel",
            overall_rmse=0.03 - i*0.001,
            rate_sensitivity_monotonic=True,
            has_rate_incentive=True,
            architecture_family=arch,
            success=True,
        ))
    filt = decide_next_iteration(state)
    assert "Ensemble" in filt.blocked_components

    # Add a 3rd architecture
    state.append(IterationRecord(
        iteration=3,
        component_touched="PrepaymentModel",
        overall_rmse=0.027,
        rate_sensitivity_monotonic=True,
        has_rate_incentive=True,
        architecture_family="MLP",
        success=True,
    ))
    filt2 = decide_next_iteration(state)
    assert "Ensemble" in filt2.allowed_components


@pytest.mark.offline
def test_exploitation_mode_when_improving():
    state = MBSSearchState(improvement_threshold=0.05, stall_window=3)
    rmses = [0.03, 0.025, 0.02, 0.015]  # each iteration improves >16%
    for i, r in enumerate(rmses, start=1):
        state.append(IterationRecord(
            iteration=i,
            component_touched="PrepaymentModel",
            overall_rmse=r,
            rate_sensitivity_monotonic=True,
            has_rate_incentive=True,
            architecture_family="LightGBM",
            success=True,
        ))
    filt = decide_next_iteration(state)
    assert filt.mode == ExplorationMode.EXPLOITATION


@pytest.mark.offline
def test_exploration_mode_when_stalled():
    state = MBSSearchState(improvement_threshold=0.05, stall_window=3)
    rmses = [0.030, 0.0295, 0.0294, 0.0293]  # each iteration improves <0.5%
    for i, r in enumerate(rmses, start=1):
        state.append(IterationRecord(
            iteration=i,
            component_touched="FeatureEng",
            overall_rmse=r,
            rate_sensitivity_monotonic=True,
            has_rate_incentive=True,
            architecture_family="LightGBM",
            success=True,
        ))
    filt = decide_next_iteration(state)
    assert filt.mode == ExplorationMode.EXPLORATION
    assert "stall" in filt.guidance.lower() or "different" in filt.guidance.lower()


@pytest.mark.offline
def test_backtrack_after_k_consecutive_failures():
    state = MBSSearchState(backtrack_trigger=3)
    # One success to establish a baseline RMSE
    state.append(IterationRecord(
        iteration=1, component_touched="PrepaymentModel",
        overall_rmse=0.02, rate_sensitivity_monotonic=True,
        has_rate_incentive=True, architecture_family="Ridge", success=True,
    ))
    # Three failed iterations touching the same component
    for i in range(2, 5):
        state.append(IterationRecord(
            iteration=i, component_touched="PrepaymentModel",
            overall_rmse=0.025, rate_sensitivity_monotonic=True,
            has_rate_incentive=True, architecture_family="Ridge", success=False,
        ))
    filt = decide_next_iteration(state)
    assert filt.should_backtrack
    assert filt.mode == ExplorationMode.BACKTRACK
    assert "PrepaymentModel" in filt.blocked_components or "PrepaymentModel" not in filt.allowed_components


@pytest.mark.offline
def test_format_filter_for_prompt_readable():
    state = MBSSearchState()
    filt = decide_next_iteration(state)
    md = format_filter_for_prompt(filt)
    assert "Search Strategy Constraints" in md
    assert "Mode" in md
    assert "Guidance" in md


@pytest.mark.offline
def test_save_and_load_round_trip(tmp_path):
    state = MBSSearchState()
    state.append(IterationRecord(
        iteration=1, component_touched="RateCurveFeatures",
        overall_rmse=0.025, rate_sensitivity_monotonic=True,
        has_rate_incentive=True, architecture_family="Ridge", success=True,
    ))
    path = tmp_path / "state.json"
    state.save(path)
    loaded = MBSSearchState.load(path)
    assert len(loaded.history) == 1
    assert loaded.history[0].overall_rmse == 0.025
    assert loaded.has_rate_incentive()
