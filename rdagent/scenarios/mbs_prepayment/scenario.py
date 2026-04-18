"""MBS Prepayment scenario class — plugs the Priority 1–10 customizations
into the RD-Agent data science loop.

This subclasses ``DataScienceScen`` so the existing
``rdagent.scenarios.data_science.loop.DataScienceRDLoop`` can drive it
unchanged. The differences from the default ``DataScienceScen``:

    1. No LLM-based competition-description analysis — MBS has a known
       fixed schema and a known metric (RMSE of SMM_DECIMAL), so we hard-
       wire these instead of asking GPT to guess them.

    2. Attaches the MBS-specific building blocks (evaluation harness,
       data contract, temporal splitter, orchestrator, memory, search
       state, personas router) as attributes of the scenario object so
       downstream coder / feedback templates can reach them via ``scen``.

    3. Uses ``MBSP_SETTINGS`` (env prefix ``MBSP_``) for MBS-specific
       knobs, independent of the ``MBS_``-prefixed ip_doctor settings.

To wire this into the DS loop, set in your ``.env``::

    DS_SCEN=rdagent.scenarios.mbs_prepayment.scenario.MBSPrepaymentScen
    DS_LOCAL_DATA_PATH=./mbs_data
    KG_LOCAL_DATA_PATH=./mbs_data

Then::

    rdagent data_science --competition mbs_prepayment
"""
from __future__ import annotations

from pathlib import Path

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.data_science.scen import DataScienceScen

from .conf import MBSP_SETTINGS
from .evaluation import MBSEvaluationHarness
from .memory import IterationPhase, MBSMemory
from .orchestration import DomainValidator, MBSOrchestrator, PhaseGate
from .personas import PersonaRouter
from .scaffold import GNMA_HARNESS_FEATURES, MBSDataContract, MBSTrainTestSplit
from .search_strategy import MBSSearchState, format_filter_for_prompt


class MBSPrepaymentScen(DataScienceScen):
    """Data science scenario specialized for MBS CUSIP-level prepayment.

    Parameters
    ----------
    competition:
        Directory name under ``DS_LOCAL_DATA_PATH`` containing
        ``description.md`` and the parquet panel data. Conventionally
        ``"mbs_prepayment"``.
    """

    def __init__(self, competition: str = "mbs_prepayment") -> None:
        self.competition = competition
        self.metric_name = "rmse_smm_decimal"
        self.metric_direction = False  # lower RMSE is better
        self.metric_direction_guess = False
        self.timeout_increase_count = 0

        # Hard-code the task profile — no LLM call needed. MBS schema is known.
        self.task_type = "Regression (panel)"
        self.data_type = "Monthly CUSIP-level MBS pool panel (parquet)"
        self.brief_description = (
            "Forecast SMM_DECIMAL (Single Monthly Mortality, decimal form in "
            "[0, 1]) for each (cusip, fh_effdt) observation in the holdout "
            "period, using the Richard-Roll prepayment decomposition "
            "(turnover + refi S-curve + burnout + curtailment)."
        )
        self.dataset_description = (
            "Single-panel layout: "
            f"`{MBSP_SETTINGS.panel_filename}` (pickled DataFrame; all GNMA "
            "features pre-normalized to mean 0 / std 1; keyed on "
            f"{MBSP_SETTINGS.cusip_col}, {MBSP_SETTINGS.date_col}; carries "
            f"the unnormalized target {MBSP_SETTINGS.target_column} in "
            f"[{MBSP_SETTINGS.target_min}, {MBSP_SETTINGS.target_max}]) "
            f"plus `{MBSP_SETTINGS.scaler_filename}` (joblib-saved sklearn-"
            "style scaler that inverse-transforms the GNMA features back to "
            "raw units — WAC as gross coupon %, "
            "Avg_Prop_Refi_Incentive_WAC_30yr_2mos as refinance incentive, "
            "Burnout_Prop_WAC_30yr_log_sum60 and "
            "Burnout_Prop_30yr_Switch_to_15_Lag1 as burnout features, "
            "WALA as weighted-average loan age). Features are already "
            "engineered — the coder only plugs a model into the scaffold. "
            "See example/gnma_feature.md for the full feature dictionary."
        )
        self.model_output_channel = 1
        self.metric_description = (
            "RMSE of SMM_DECIMAL on the temporal holdout (fh_effdt > "
            f"{MBSP_SETTINGS.train_end_date}). The scaffold inverse-transforms "
            "the GNMA features with scaler.sav before scoring, so reported "
            "diagnostics are in raw units: per-coupon-bucket RMSE (on raw "
            "WAC), S-curve R² and inflection point (vs raw "
            "Avg_Prop_Refi_Incentive_WAC_30yr_2mos), and regime-transition "
            "RMSE. A model that improves overall RMSE but degrades "
            "per-coupon uniformity or regime-transition RMSE is a REJECT."
        )
        self.submission_specifications = (
            f"Produce {MBSP_SETTINGS.submission_filename} with columns "
            "(cusip, fh_effdt, smm_decimal_pred) covering every holdout row "
            f"(fh_effdt > {MBSP_SETTINGS.train_end_date}). Predictions are "
            "automatically clipped to [0, 1] by the scaffold."
        )
        self.coder_longer_time_limit_required = False
        self.runner_longer_time_limit_required = False

        # Load description.md if present (used by the prompt templates)
        self.raw_description = self._load_description_md()
        self.processed_data_folder_description = ""
        self.debug_path = str(MBSP_SETTINGS.data_dir / competition)

        # Attach MBS building blocks so downstream coder/feedback can reach them.
        # The panel is all-normalized; harness_raw_features names the GNMA
        # columns the scaffold inverse-transforms via scaler.sav for scoring.
        harness_raw = tuple(
            dict.fromkeys(
                list(MBSP_SETTINGS.required_columns_list()) + list(GNMA_HARNESS_FEATURES)
            )
        )
        self.mbs_contract = MBSDataContract(
            cusip_col=MBSP_SETTINGS.cusip_col,
            date_col=MBSP_SETTINGS.date_col,
            target_col=MBSP_SETTINGS.target_column,
            target_range=(MBSP_SETTINGS.target_min, MBSP_SETTINGS.target_max),
            required_columns=harness_raw,
            harness_raw_features=harness_raw,
            forbidden_columns=tuple(MBSP_SETTINGS.forbidden_columns_list()),
        )
        self.mbs_splitter = MBSTrainTestSplit(
            train_end_date=MBSP_SETTINGS.train_end_date,
            date_column=MBSP_SETTINGS.date_col,
            embargo_months=MBSP_SETTINGS.embargo_months,
        )
        self.mbs_harness = MBSEvaluationHarness(
            coupon_buckets=MBSP_SETTINGS.coupon_buckets_list(),
            regime_transition_dates=MBSP_SETTINGS.regime_transition_dates_list(),
            coupon_col=MBSP_SETTINGS.coupon_col,
            rate_incentive_col=MBSP_SETTINGS.rate_incentive_col,
            fh_effdt_col=MBSP_SETTINGS.date_col,
            cusip_col=MBSP_SETTINGS.cusip_col,
            wala_col=MBSP_SETTINGS.wala_col,
        )
        self.mbs_memory = MBSMemory(
            memory_path=MBSP_SETTINGS.memory_path,
            max_failures_shown=MBSP_SETTINGS.memory_max_failures_shown,
            max_properties_shown=MBSP_SETTINGS.memory_max_properties_shown,
        )
        self.mbs_search_state = MBSSearchState(
            improvement_threshold=MBSP_SETTINGS.improvement_threshold,
            stall_window=MBSP_SETTINGS.stall_window,
            cooldown_duration=MBSP_SETTINGS.cooldown_duration,
            backtrack_trigger=MBSP_SETTINGS.backtrack_trigger,
        )
        self.mbs_validator = DomainValidator(
            target_min=MBSP_SETTINGS.target_min,
            target_max=MBSP_SETTINGS.target_max,
            max_training_seconds=MBSP_SETTINGS.validator_max_training_seconds,
        )
        self.mbs_gate = PhaseGate(
            baseline_max_rmse=MBSP_SETTINGS.gate_baseline_max_rmse,
            rate_response_min_s_curve_r2=MBSP_SETTINGS.gate_rate_response_min_s_curve_r2,
            rate_response_inflection_range_bps=(
                MBSP_SETTINGS.gate_rate_response_inflection_min_bps,
                MBSP_SETTINGS.gate_rate_response_inflection_max_bps,
            ),
            dynamics_max_worst_coupon_rmse=MBSP_SETTINGS.gate_dynamics_max_worst_coupon_rmse,
            macro_regime_transition_ratio=MBSP_SETTINGS.gate_macro_regime_transition_ratio,
        )
        self.mbs_orchestrator = MBSOrchestrator(
            memory=self.mbs_memory,
            search_state=self.mbs_search_state,
            validator=self.mbs_validator,
            gate=self.mbs_gate,
        )
        self.mbs_persona_router = PersonaRouter()

        logger.info(
            f"MBSPrepaymentScen initialized: competition={competition}, "
            f"metric={self.metric_name}, train_end={MBSP_SETTINGS.train_end_date}"
        )

    # ------------------------------------------------------------------
    # Override LLM-driven helpers with MBS static descriptions
    # ------------------------------------------------------------------

    def reanalyze_competition_description(self) -> None:
        """MBS schema is fixed — nothing to reanalyze."""
        logger.info("MBSPrepaymentScen: reanalyze_competition_description is a no-op.")

    def get_scenario_all_desc(self, eda_output=None) -> str:
        """Extend the base DS scenario description with MBS-specific context.

        Every LLM call in the loop (proposal, coding, runner eval, feedback)
        reads this description.  By appending MBS memory, search strategy
        constraints, and phase info here, all calls become MBS-aware without
        modifying any shared DS code.
        """
        base = super().get_scenario_all_desc(eda_output=eda_output)

        sections: list[str] = []

        # 1) Current phase and gate criteria
        spec = self.mbs_orchestrator.phase_spec()
        sections.append(
            f"## MBS Phase: {spec.phase.value}\n"
            f"**Goal**: {spec.goal}\n"
            f"**Gate criteria**: {spec.gate_criteria_description}\n"
            f"**Allowed components**: {', '.join(spec.allowed_components)}"
        )

        # 2) Search strategy constraints (curriculum filter)
        constraints = self.mbs_orchestrator.iteration_constraints()
        sections.append(format_filter_for_prompt(constraints))

        # 3) MBS memory context (best experiments, recent failures, trends)
        memory_ctx = self.mbs_memory.render_context(IterationPhase.HYPOTHESIS_GEN)
        if memory_ctx:
            sections.append(memory_ctx)

        # 4) Data contract reminder
        sections.append(
            "## MBS Data Contract\n"
            f"- Inputs: `{MBSP_SETTINGS.panel_filename}` (pickled DataFrame) "
            f"+ `{MBSP_SETTINGS.scaler_filename}` (joblib-saved scaler).\n"
            "- The panel is pre-normalized (mean 0, std 1) on all GNMA "
            "feature columns; do NOT re-engineer or re-normalize features.\n"
            f"- Panel index: ({self.mbs_contract.cusip_col}, "
            f"{self.mbs_contract.date_col}). Target: "
            f"{self.mbs_contract.target_col} in "
            f"[{self.mbs_contract.target_range[0]}, "
            f"{self.mbs_contract.target_range[1]}] (unnormalized).\n"
            f"- Required GNMA features: {self.mbs_contract.required_columns}\n"
            f"- Forbidden (future-leaking) columns: "
            f"{self.mbs_contract.forbidden_columns}\n"
            f"- Temporal split at {self.mbs_splitter.train_end_date} on "
            f"{self.mbs_splitter.date_column} (embargo "
            f"{self.mbs_splitter.embargo_months} months).\n"
            "- Scoring: scaffold uses scaler.sav to inverse-transform GNMA "
            "features back to raw units (WAC in %, "
            "Avg_Prop_Refi_Incentive_WAC_30yr_2mos in raw incentive units, "
            "etc.) so the harness measures per-coupon RMSE, S-curve R², "
            "inflection point, and seasoning effects on the real scale.\n"
            f"- Submission: write `{MBSP_SETTINGS.submission_filename}` "
            "with (cusip, fh_effdt, smm_decimal_pred). Scaffold clips "
            "predictions to [0, 1]."
        )

        return base + "\n\n" + "\n\n".join(sections)

    def _load_description_md(self) -> str:
        """Load description.md from the MBS data folder if it exists."""
        desc_path = MBSP_SETTINGS.data_dir / self.competition / "description.md"
        if desc_path.exists():
            return desc_path.read_text()
        return (
            "# MBS CUSIP-level Prepayment Forecasting\n\n"
            "Target: SMM_DECIMAL in [0, 1] on a (cusip, fh_effdt) panel. "
            "Temporal split at "
            f"{MBSP_SETTINGS.train_end_date}. Metric: RMSE of SMM_DECIMAL, "
            "with per-coupon-bucket breakdown and rate-sensitivity "
            "S-curve diagnostics. See scorecard from MBSEvaluationHarness."
        )
