"""Tests for MBS interpretability loop (Priority 4)."""
import numpy as np
import pytest

from rdagent.scenarios.mbs_prepayment.interpretability import (
    AttributionMemory,
    compute_attributions,
    sanity_check_attributions,
    run_interpretability_pass,
)


class LinearToyModel:
    """Toy model with known feature signs for prior-check testing."""

    def __init__(self, coefs: np.ndarray):
        self.coefs = np.asarray(coefs, dtype=float)

    def predict(self, X):
        return X @ self.coefs


@pytest.mark.offline
def test_permutation_importance_runs(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (200, 4))
    model = LinearToyModel(np.array([1.0, -0.5, 0.2, 0.0]))
    feature_names = ["rate_incentive", "burnout_index", "wala", "noise"]
    result = compute_attributions(model, X, feature_names)
    assert result.method == "permutation_importance"
    assert len(result.mean_attribution) == 4
    # rate_incentive (coef=1.0) should dominate
    top = result.top_n(1)
    assert top[0][0] == "rate_incentive"


@pytest.mark.offline
def test_sanity_check_detects_wrong_sign():
    """Model with rate_incentive coef = -1 (wrong sign) should trigger a violation."""
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (200, 2))
    # Wrong signs: rate_incentive NEGATIVE (should be +), burnout POSITIVE (should be -)
    model = LinearToyModel(np.array([-1.0, +1.0]))
    feature_names = ["rate_incentive", "burnout_index"]
    result = compute_attributions(model, X, feature_names)
    report = sanity_check_attributions(result)
    assert report.has_violations
    violated_features = {v["feature"] for v in report.violations}
    assert "rate_incentive" in violated_features
    assert "burnout_index" in violated_features


@pytest.mark.offline
def test_sanity_check_passes_correct_signs():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (200, 3))
    # Correct signs: rate_incentive positive, burnout negative, wala positive
    model = LinearToyModel(np.array([+1.0, -0.5, +0.3]))
    feature_names = ["rate_incentive", "burnout_index", "wala"]
    result = compute_attributions(model, X, feature_names)
    report = sanity_check_attributions(result)
    assert not report.has_violations
    passed_features = {p["feature"] for p in report.passed}
    assert {"rate_incentive", "burnout_index", "wala"}.issubset(passed_features)


@pytest.mark.offline
def test_attribution_memory_persists_across_calls(tmp_path):
    memory_path = tmp_path / "attr_memory.json"
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (100, 2))
    model = LinearToyModel(np.array([1.0, -0.5]))
    feature_names = ["rate_incentive", "burnout_index"]

    # Iteration 1
    run_interpretability_pass(model, X, feature_names, iteration=1, memory_path=memory_path)
    # Iteration 2 with different model
    model2 = LinearToyModel(np.array([0.8, -0.4]))
    run_interpretability_pass(model2, X, feature_names, iteration=2, memory_path=memory_path)

    memory = AttributionMemory(memory_path)
    history = memory.load()
    assert len(history) == 2
    assert history[0]["iteration"] == 1
    assert history[1]["iteration"] == 2


@pytest.mark.offline
def test_attribution_memory_format_for_prompt(tmp_path):
    memory_path = tmp_path / "attr_memory.json"
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (100, 2))

    # Iteration 1: correct signs (no violations)
    run_interpretability_pass(
        LinearToyModel(np.array([1.0, -0.5])),
        X, ["rate_incentive", "burnout_index"], iteration=1, memory_path=memory_path,
    )
    # Iteration 2: wrong signs (violations)
    run_interpretability_pass(
        LinearToyModel(np.array([-1.0, 0.5])),
        X, ["rate_incentive", "burnout_index"], iteration=2, memory_path=memory_path,
    )

    memory = AttributionMemory(memory_path)
    formatted = memory.format_for_prompt(last_n=2)
    assert "Iteration 2" in formatted
    assert "Iteration 1" in formatted
    assert "violation" in formatted.lower() or "⚠" in formatted


@pytest.mark.offline
def test_run_interpretability_pass_returns_summary(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (100, 3))
    model = LinearToyModel(np.array([1.0, -0.5, 0.2]))
    feature_names = ["rate_incentive", "burnout_index", "wala"]
    summary = run_interpretability_pass(
        model, X, feature_names, iteration=1, memory_path=tmp_path / "m.json"
    )
    assert "top_attributions" in summary
    assert "has_violations" in summary
    assert summary["has_violations"] is False
    assert summary["method"] in ("permutation_importance", "integrated_gradients")
