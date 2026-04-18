"""MBS Prepayment scenario settings.

This file exposes configuration for the MBS CUSIP-level prepayment forecasting
customization of the RD-Agent data science loop.

Env prefix is ``MBSP_`` (NOT ``MBS_``, which is owned by the ip_doctor LangGraph
scenario). The two scenarios are independent — this settings object does not
import anything from ip_doctor and vice versa.

Usage
-----
The data science loop reads ``DS_*`` env vars via the global
``DS_RD_SETTING`` object. To point the loop at this scenario, set in ``.env``::

    DS_SCEN=rdagent.scenarios.mbs_prepayment.scenario.MBSPrepaymentScen
    DS_LOCAL_DATA_PATH=./mbs_data
    KG_LOCAL_DATA_PATH=./mbs_data    # kaggle base setting (same path)

then run::

    rdagent data_science --competition mbs_prepayment

The MBS-specific knobs below (rate-sensitivity thresholds, phase-gate RMSE
ceilings, coupon-bucket definitions, etc.) are read by the Priority 1–10
modules via ``MBSP_SETTINGS``, without requiring any change to the core
data science loop.
"""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import SettingsConfigDict

from rdagent.core.conf import ExtendedBaseSettings


class MBSPrepaymentSettings(ExtendedBaseSettings):
    """Scenario-specific settings for MBS prepayment modelling.

    All fields have sensible defaults; override any of them via a ``.env`` file
    or shell env vars using the ``MBSP_`` prefix, e.g. ``MBSP_TRAIN_END_DATE``.
    """

    model_config = SettingsConfigDict(env_prefix="MBSP_", protected_namespaces=())

    # ------------------------------------------------------------------
    # 1. Filesystem layout
    # ------------------------------------------------------------------
    data_dir: Path = Path("./mbs_data")
    """Directory containing per-competition MBS folders.

    Each competition sub-folder ships a single-panel layout:
    ``<data_dir>/<competition>/tfminput.pkl`` (normalized features, mean 0 /
    std 1), ``<data_dir>/<competition>/scaler.sav`` (joblib-saved fitted
    scaler inverse-transforming the GNMA features listed in
    ``example/gnma_feature.md``), and ``<data_dir>/<competition>/description.md``.
    """

    panel_filename: str = "tfminput.pkl"
    scaler_filename: str = "scaler.sav"
    sample_submission_filename: str = "sample_submission.csv"
    submission_filename: str = "submission.csv"

    model_checkpoint_dir: Path = Path("./mbs_models")
    """Directory for saving model weights / scalers between iterations."""

    output_dir: Path = Path("./mbs_output")
    """Root for plots, scorecards, reports, and cached artifacts."""

    memory_path: Path = Path("./mbs_output/memory.json")
    """Persistent phase-aware memory file (Priority 7)."""

    search_state_path: Path = Path("./mbs_output/search_state.json")
    """Persistent curriculum/cooldown state (Priority 6)."""

    cache_dir: Path = Path("./mbs_output/cache")
    """Content-addressed artifact cache root (Priority 9)."""

    # ------------------------------------------------------------------
    # 2. Data contract (Priority 2 - scaffold.py)
    # ------------------------------------------------------------------
    target_column: str = "smm_decimal"
    target_min: float = 0.0
    target_max: float = 1.0
    cusip_col: str = "cusip"
    date_col: str = "fh_effdt"
    required_columns: str = "WAC,WALA,Avg_Prop_Refi_Incentive_WAC_30yr_2mos"
    """Comma-separated GNMA feature columns that must exist in the panel and
    be invertible via ``scaler.sav`` — the harness inverse-transforms these
    to raw scale for per-coupon bucketing, refi-incentive Spearman, and
    seasoning diagnostics. The panel itself is stored normalized (mean 0,
    std 1); these names drive the scorecard, not any LLM DataFrame.
    """
    forbidden_columns: str = (
        "future_smm,forward_smm,next_month_smm,forward_rate,future_rate_incentive"
    )
    """Comma-separated list of future-leaking columns that must not appear."""
    macro_lag_days_min: int = 30

    # ------------------------------------------------------------------
    # 3. Temporal train/test split
    # ------------------------------------------------------------------
    train_end_date: str = "2024-10-31"
    embargo_months: int = 0

    # ------------------------------------------------------------------
    # 4. Evaluation harness (Priority 1 - evaluation.py)
    # ------------------------------------------------------------------
    #: Coupon-bucket intervals reference the raw ``WAC`` column (GNMA gross
    #: coupon, expressed in percent) carried unnormalized in ``info.csv``.
    coupon_buckets: str = "0.0:3.0;3.0:3.5;3.5:4.0;4.0:4.5;4.5:5.0;5.0:99.0"
    """Semicolon-separated `lo:hi` half-open intervals [lo, hi)."""
    regime_transition_dates: str = "2013-05-01,2020-03-01,2022-03-01"
    coupon_col: str = "WAC"
    rate_incentive_col: str = "Avg_Prop_Refi_Incentive_WAC_30yr_2mos"
    wala_col: str = "WALA"

    # ------------------------------------------------------------------
    # 5. Search strategy (Priority 6)
    # ------------------------------------------------------------------
    improvement_threshold: float = 0.05
    stall_window: int = 3
    cooldown_duration: int = 2
    backtrack_trigger: int = 3

    # ------------------------------------------------------------------
    # 6. Memory packing (Priority 7)
    # ------------------------------------------------------------------
    memory_max_failures_shown: int = 3
    memory_max_properties_shown: int = 5

    # ------------------------------------------------------------------
    # 7. Domain validator (Priority 8)
    # ------------------------------------------------------------------
    validator_max_training_seconds: float = 3600.0

    # ------------------------------------------------------------------
    # 8. Phase gates (Priority 8)
    # ------------------------------------------------------------------
    gate_baseline_max_rmse: float = 0.040
    gate_rate_response_min_s_curve_r2: float = 0.6
    gate_rate_response_inflection_min_bps: float = 50.0
    gate_rate_response_inflection_max_bps: float = 150.0
    gate_dynamics_max_worst_coupon_rmse: float = 0.035
    gate_macro_regime_transition_ratio: float = 2.0

    phase_budget_baseline: int = 5
    phase_budget_rate_response: int = 8
    phase_budget_dynamics: int = 8
    phase_budget_macro_regime: int = 6
    phase_budget_ensemble: int = 4

    # ------------------------------------------------------------------
    # 9. Execution environment per-stage budgets (Priority 9)
    # ------------------------------------------------------------------
    stage_data_load_timeout: int = 300
    stage_data_load_memory_gb: float = 16.0
    stage_feature_eng_timeout: int = 600
    stage_feature_eng_memory_gb: float = 24.0
    stage_training_timeout: int = 3600
    stage_training_memory_gb: float = 32.0
    stage_training_allow_gpu: bool = True
    stage_evaluation_timeout: int = 300
    stage_evaluation_memory_gb: float = 8.0
    stage_attribution_timeout: int = 900
    stage_attribution_memory_gb: float = 16.0
    stage_attribution_allow_gpu: bool = True

    # ------------------------------------------------------------------
    # 10. Personas (Priority 10)
    # ------------------------------------------------------------------
    persona_quant_researcher_model: str = "gpt-4-turbo"
    persona_quant_researcher_temperature: float = 0.7
    persona_ml_engineer_model: str = "gpt-4-turbo"
    persona_ml_engineer_temperature: float = 0.2
    persona_model_validator_model: str = "gpt-4-turbo"
    persona_model_validator_temperature: float = 0.3
    persona_data_engineer_model: str = "gpt-4o-mini"
    persona_data_engineer_temperature: float = 0.1

    # ------------------------------------------------------------------
    # 11. Interpretability (Priority 4)
    # ------------------------------------------------------------------
    interpretability_n_samples: int = 500
    interpretability_baseline_strategy: str = "zero"
    ig_n_steps: int = 50
    ig_target_output: str = "cpr"
    ig_batch_size: int = 32

    # ------------------------------------------------------------------
    # 12. Reproducibility
    # ------------------------------------------------------------------
    random_seed: int = 42
    log_level: str = "INFO"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def required_columns_list(self) -> list[str]:
        return [c.strip() for c in self.required_columns.split(",") if c.strip()]

    def forbidden_columns_list(self) -> list[str]:
        return [c.strip() for c in self.forbidden_columns.split(",") if c.strip()]

    def regime_transition_dates_list(self) -> list[str]:
        return [d.strip() for d in self.regime_transition_dates.split(",") if d.strip()]

    def coupon_buckets_list(self) -> list[tuple[float, float]]:
        out: list[tuple[float, float]] = []
        for chunk in self.coupon_buckets.split(";"):
            chunk = chunk.strip()
            if not chunk:
                continue
            lo_s, hi_s = chunk.split(":")
            out.append((float(lo_s), float(hi_s)))
        return out


MBSP_SETTINGS = MBSPrepaymentSettings()
