"""Tests for MBS prompt loader (Priority 3)."""
import pytest

from rdagent.scenarios.mbs_prepayment.prompt_loader import MBSPromptLoader


@pytest.mark.offline
def test_scen_role_mentions_mbs_concepts():
    loader = MBSPromptLoader()
    role = loader.get("scen.role")
    assert "SMM_DECIMAL" in role
    assert "Richard-Roll" in role
    assert "burnout" in role.lower()
    assert "agency MBS" in role


@pytest.mark.offline
def test_hypothesis_specification_blocks_generic_ml():
    loader = MBSPromptLoader()
    spec = loader.get("hypothesis_specification")
    assert "XGBoost" in spec or "LightGBM" in spec
    assert "Richard-Roll" in spec or "prepayment-theoretic" in spec
    assert "per-coupon-bucket" in spec


@pytest.mark.offline
def test_component_specs_present_for_all_mbs_components():
    loader = MBSPromptLoader()
    for comp in ["DataLoader", "FeatureEng", "PrepaymentModel", "ScenarioValidator"]:
        spec = loader.get(f"component_spec.{comp}")
        assert len(spec) > 100, f"{comp} spec too short"


@pytest.mark.offline
def test_dataloader_spec_enforces_temporal_lag():
    loader = MBSPromptLoader()
    spec = loader.get("component_spec.DataLoader")
    assert "lag" in spec.lower()
    assert "fh_effdt" in spec
    assert "cusip" in spec


@pytest.mark.offline
def test_feature_eng_spec_lists_canonical_features():
    loader = MBSPromptLoader()
    spec = loader.get("component_spec.FeatureEng")
    # Use actual column names from gnma_feature.md (case-sensitive)
    for feat in ["Avg_Prop_Refi_Incentive_WAC_30yr_2mos", "Burnout_Prop_WAC_30yr_log_sum60", "seasoning_ramp", "coupon_bucket"]:
        assert feat in spec, f"Canonical feature '{feat}' missing from FeatureEng spec"


@pytest.mark.offline
def test_feedback_schema_includes_coupon_bucket_check():
    loader = MBSPromptLoader()
    schema = loader.get("feedback_schema_extra")
    assert "coupon_bucket_check" in schema
    assert "burnout_check" in schema


@pytest.mark.offline
def test_all_keys_discoverable():
    loader = MBSPromptLoader()
    keys = loader.all_keys()
    assert "scen.role" in keys
    assert "hypothesis_specification" in keys
    assert "component_spec.DataLoader" in keys
    assert "feedback_schema_extra" in keys
