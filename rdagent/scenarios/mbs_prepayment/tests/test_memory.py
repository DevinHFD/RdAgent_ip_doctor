"""Tests for MBS memory / context management (Priority 7)."""
import pytest

from rdagent.scenarios.mbs_prepayment.memory import (
    IterationPhase,
    MBSMemory,
    ModelProperties,
    TraceEntry,
    summarize_code_change,
)


SCORECARD_BASELINE = {
    "accuracy": {
        "overall_rmse": 0.030,
        "rmse_by_coupon_bucket": {"0.0-3.0": 0.020, "3.0-3.5": 0.025, "5.0+": 0.045},
    },
    "rate_sensitivity": {"monotonicity_spearman": 0.6, "s_curve_rmse_overall": 0.0040, "s_curve_rmse_mid_belly": 0.0046},
    "temporal_robustness": {"regime_transition_rmse": {"2020-03-01": 0.040}},
    "structural_properties": {
        "cusip_differentiation_std": 0.003,
        "seasonality_residual_range": 0.001,
    },
}

SCORECARD_IMPROVED = {
    "accuracy": {
        "overall_rmse": 0.022,
        "rmse_by_coupon_bucket": {"0.0-3.0": 0.015, "3.0-3.5": 0.018, "5.0+": 0.030},
    },
    "rate_sensitivity": {"monotonicity_spearman": 0.88, "s_curve_rmse_overall": 0.0028, "s_curve_rmse_mid_belly": 0.0033},
    "temporal_robustness": {"regime_transition_rmse": {"2020-03-01": 0.028}},
    "structural_properties": {
        "cusip_differentiation_std": 0.006,
        "seasonality_residual_range": 0.002,
    },
}


@pytest.mark.offline
def test_model_properties_from_scorecard():
    props = ModelProperties.from_scorecard(
        iteration=1,
        model_type="Ridge",
        component_touched="RateCurveFeatures",
        scorecard=SCORECARD_BASELINE,
        n_features_used=5,
    )
    assert props.overall_rmse == 0.030
    assert props.rmse_by_coupon_bucket["5.0+"] == 0.045
    assert props.regime_transition_rmse_mean == 0.040
    assert "rmse=0.03000" in props.summary_line()


@pytest.mark.offline
def test_memory_persists_and_tracks_best_per_component(tmp_path):
    memory = MBSMemory(memory_path=tmp_path / "mem.json")
    memory.append_entry(TraceEntry(
        iteration=1, component="RateCurveFeatures",
        hypothesis="Add rate_incentive", code_change_summary="rate_incentive = WAC - mortgage_rate",
        decision="accept", success=True,
        properties=ModelProperties.from_scorecard(1, "Ridge", "RateCurveFeatures", SCORECARD_BASELINE),
    ))
    memory.append_entry(TraceEntry(
        iteration=2, component="RateCurveFeatures",
        hypothesis="Try log incentive", code_change_summary="log incentive",
        decision="accept", success=True,
        properties=ModelProperties.from_scorecard(2, "LightGBM", "RateCurveFeatures", SCORECARD_IMPROVED),
    ))

    best = memory.best_per_component()
    assert "RateCurveFeatures" in best
    # Best should be the improved one (lowest overall RMSE)
    assert best["RateCurveFeatures"]["iteration"] == 2


@pytest.mark.offline
def test_memory_tracks_failures(tmp_path):
    memory = MBSMemory(memory_path=tmp_path / "mem.json")
    for i in range(1, 5):
        memory.append_entry(TraceEntry(
            iteration=i, component="PoolDynamics",
            hypothesis=f"Attempt {i}", code_change_summary="",
            decision="reject", success=False,
            feedback_reason=f"Iteration {i} failed: RMSE regression",
        ))
    failures = memory.recent_failures(k=3)
    assert len(failures) == 3
    assert failures[-1]["iteration"] == 4
    assert "iteration 4" in failures[-1]["feedback_reason"].lower() or "Iteration 4" in failures[-1]["feedback_reason"]


@pytest.mark.offline
def test_hypothesis_phase_context_has_no_code(tmp_path):
    memory = MBSMemory(memory_path=tmp_path / "mem.json")
    memory.append_entry(TraceEntry(
        iteration=1, component="RateCurveFeatures",
        hypothesis="Add rate_incentive feature",
        code_change_summary="def compute_rate_incentive(wac, mrate): return wac - mrate",
        decision="accept", success=True,
        properties=ModelProperties.from_scorecard(1, "Ridge", "RateCurveFeatures", SCORECARD_BASELINE),
    ))
    ctx = memory.render_context(IterationPhase.HYPOTHESIS_GEN)
    assert "hypothesis phase" in ctx.lower()
    assert "RateCurveFeatures" in ctx
    assert "rmse" in ctx.lower()
    # Full code should NOT be there (we only show compressed summaries)
    assert "def compute_rate_incentive" not in ctx


@pytest.mark.offline
def test_feedback_phase_context_shows_sota(tmp_path):
    memory = MBSMemory(memory_path=tmp_path / "mem.json")
    memory.append_entry(TraceEntry(
        iteration=1, component="RateCurveFeatures",
        hypothesis="baseline", code_change_summary="",
        decision="accept", success=True,
        properties=ModelProperties.from_scorecard(1, "Ridge", "RateCurveFeatures", SCORECARD_BASELINE),
    ))
    memory.append_entry(TraceEntry(
        iteration=2, component="PrepaymentModel",
        hypothesis="use lightgbm", code_change_summary="",
        decision="accept", success=True,
        properties=ModelProperties.from_scorecard(2, "LightGBM", "PrepaymentModel", SCORECARD_IMPROVED),
    ))
    ctx = memory.render_context(IterationPhase.FEEDBACK)
    assert "SOTA" in ctx
    assert "0.02200" in ctx  # the improved overall_rmse
    assert "per-coupon" in ctx.lower()


@pytest.mark.offline
def test_summarize_code_change_extracts_math():
    diff = """\
--- a/model.py
+++ b/model.py
@@ -1,3 +1,5 @@
 def build():
-    return LinearRegression()
+    return GradientBoostingRegressor(loss='huber', n_estimators=200)
+# Add S-curve overlay
+    refi_response = 1.0 / (1.0 + np.exp(-2.5 * rate_incentive))
"""
    summary = summarize_code_change("Refi S-curve", diff)
    assert "GradientBoosting" in summary or "refi_response" in summary
    assert len(summary) < 500


@pytest.mark.offline
def test_memory_round_trips(tmp_path):
    path = tmp_path / "mem.json"
    memory = MBSMemory(memory_path=path)
    memory.append_entry(TraceEntry(
        iteration=1, component="RateCurveFeatures",
        hypothesis="baseline", code_change_summary="",
        decision="accept", success=True,
        properties=ModelProperties.from_scorecard(1, "Ridge", "RateCurveFeatures", SCORECARD_BASELINE),
    ))
    # Load a fresh memory instance from the same path
    memory2 = MBSMemory(memory_path=path)
    assert len(memory2._data["entries"]) == 1
    assert memory2.best_per_component()["RateCurveFeatures"]["iteration"] == 1
