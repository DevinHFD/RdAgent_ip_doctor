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
from .scaffold import GNMA_HARNESS_FEATURES, MBSCUSIPSplit, MBSDataContract, MBSTrainTestSplit
from .search_strategy import MBS_TO_DS_COMPONENT, MBSSearchState, format_filter_for_prompt


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
            f"fh_upb-weighted RMSE of SMM_DECIMAL (fh_upb capped at 150M per "
            f"pool) on the fixed test-CUSIP set (≈1/7 of all CUSIPs, all time "
            f"rows, fixed seed={MBSP_SETTINGS.cusip_split_seed}). All RMSE "
            "metrics in the scorecard are UPB-weighted. The scaffold "
            "inverse-transforms the GNMA features with scaler.sav before "
            "scoring, so reported diagnostics are in raw units: UPB-weighted "
            "per-coupon-bucket RMSE (on raw WAC), S-curve bin RMSE between "
            "the UPB-weighted ACTUAL and PREDICTED SMM bin curves over raw "
            "Avg_Prop_Refi_Incentive_WAC_30yr_2mos "
            "(= WAC / avg(mortgage_rate_lag1, mortgage_rate_lag2); >1 means "
            "refi incentive) — reported overall plus per-segment "
            "(left tail / mid belly / right tail) so the loop can see which "
            "part of the S-curve needs improvement, and "
            "UPB-weighted regime-transition RMSE. A model that improves "
            "overall UPB-weighted RMSE but degrades per-coupon uniformity or "
            "regime-transition RMSE is a REJECT. "
            "Current SOTA: MLP [10, 20, 10] hidden units, Leaky ReLU hidden "
            "activations, Sigmoid output, trained on all GNMA features with "
            "fh_upb-weighted RMSE (fh_upb capped at 150M). "
            f"Runtime hard limit: {MBSP_SETTINGS.validator_max_training_seconds/3600:.0f} hours."
        )
        self.submission_specifications = (
            f"Produce {MBSP_SETTINGS.submission_filename} with columns "
            "(cusip, fh_effdt, smm_decimal_pred) covering the test-CUSIP set "
            f"(≈1/7 of all CUSIPs, fixed by seed={MBSP_SETTINGS.cusip_split_seed}). Predictions are "
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
        self.mbs_splitter = MBSCUSIPSplit(
            train_end_date=MBSP_SETTINGS.train_end_date,
            date_column=MBSP_SETTINGS.date_col,
            cusip_column=MBSP_SETTINGS.cusip_col,
            test_fraction=MBSP_SETTINGS.test_cusip_fraction,
            val_fraction=MBSP_SETTINGS.val_cusip_fraction,
            random_seed=MBSP_SETTINGS.cusip_split_seed,
        )
        self.mbs_harness = MBSEvaluationHarness(
            coupon_buckets=MBSP_SETTINGS.coupon_buckets_list(),
            regime_transition_dates=MBSP_SETTINGS.regime_transition_dates_list(),
            coupon_col=MBSP_SETTINGS.coupon_col,
            rate_incentive_col=MBSP_SETTINGS.rate_incentive_col,
            fh_effdt_col=MBSP_SETTINGS.date_col,
            cusip_col=MBSP_SETTINGS.cusip_col,
            wala_col=MBSP_SETTINGS.wala_col,
            s_curve_bin_edges=list(MBSP_SETTINGS.s_curve_bin_edges),
            s_curve_left_tail_max_ratio=MBSP_SETTINGS.s_curve_left_tail_max_ratio,
            s_curve_right_tail_min_ratio=MBSP_SETTINGS.s_curve_right_tail_min_ratio,
            s_curve_min_rows_per_bin=MBSP_SETTINGS.s_curve_min_rows_per_bin,
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
            rate_response_max_s_curve_rmse_overall=MBSP_SETTINGS.gate_rate_response_max_s_curve_rmse_overall,
            rate_response_max_s_curve_rmse_mid_belly=MBSP_SETTINGS.gate_rate_response_max_s_curve_rmse_mid_belly,
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
        # The phase allowlist is in MBS-native names (richer, 7-name taxonomy);
        # the DS proposer can only emit one of {DataLoadSpec, FeatureEng, Model,
        # Ensemble, Workflow} so we show the translated DS set alongside.
        spec = self.mbs_orchestrator.phase_spec()
        ds_allowed = sorted({MBS_TO_DS_COMPONENT.get(c, c) for c in spec.allowed_components})
        sections.append(
            f"## MBS Phase: {spec.phase.value}\n"
            f"**Goal**: {spec.goal}\n"
            f"**Gate criteria**: {spec.gate_criteria_description}\n"
            f"**MBS focus areas** (for hypothesis content): "
            f"{', '.join(spec.allowed_components)}\n"
            f"**Allowed DS components** (for the `component` field in your "
            f"structured output): {', '.join(ds_allowed)}"
        )

        # 2) Search strategy constraints (curriculum filter)
        constraints = self.mbs_orchestrator.iteration_constraints()
        sections.append(format_filter_for_prompt(constraints))

        # 3) MBS memory context (best experiments, recent failures, trends)
        memory_ctx = self.mbs_memory.render_context(IterationPhase.HYPOTHESIS_GEN)
        if memory_ctx:
            sections.append(memory_ctx)

        # 4) MBS main.py mandate — overrides the generic share.yaml Workflow spec
        # The DS loop injects share.yaml's Workflow spec which tells the LLM to
        # write a standard Kaggle pipeline (load_data, feature, model_*, ensemble).
        # That spec is wrong for MBS: main.py MUST call run_scaffold_pipeline()
        # AND build_model must be defined inline in the same main.py because the
        # pipeline coder only emits a single file per iteration.
        sections.append(
            "## MANDATORY: MBS main.py is a SINGLE self-contained file\n\n"
            "**IGNORE** the generic Workflow spec about `load_data.py`, `feature.py`, "
            "`feat_eng()`, `model_workflow()`, `ensemble.py`, KFold, or writing "
            "`submission.csv` / `scores.csv` yourself. Those instructions are for "
            "standard Kaggle competitions and DO NOT apply here.\n\n"
            "**Critical constraint**: DS pipeline mode emits exactly ONE file per "
            "iteration — `main.py`. No other file (no `model_gbm.py`, no "
            "`ensemble.py`, no `model_*.py`) will be written to the workspace. "
            "Therefore `main.py` MUST define `build_model()` inline — do NOT write "
            "`from model_xx import build_model` or any other import of a non-stdlib "
            "local module; that import will fail at runtime because the file does "
            "not exist.\n\n"
            "For MBS Prepayment, `main.py` MUST follow this exact pattern. "
            "**The default model family is PyTorch neural networks** (MLP, "
            "Mixture-of-Experts, residual MLP, etc.) to align with the "
            "current SOTA — MLP[10,20,10] with Leaky ReLU + Sigmoid trained "
            "with fh_upb-weighted RMSE. GBM/linear estimators are permitted "
            "only as diagnostic baselines or when a hypothesis explicitly "
            "justifies a non-NN architecture:\n\n"
            "```python\n"
            "import os\n"
            "from pathlib import Path\n"
            "import numpy as np\n"
            "import torch\n"
            "import torch.nn as nn\n"
            "from sklearn.base import BaseEstimator, RegressorMixin\n"
            "from rdagent.scenarios.mbs_prepayment.scaffold import run_scaffold_pipeline\n\n"
            "class MLPRegressor(BaseEstimator, RegressorMixin):\n"
            '    """Sklearn-style wrapper around a PyTorch MLP.\n\n'
            "    fit signature accepts X_val, y_val, sample_weight to match the\n"
            "    scaffold's call convention (UPB-weighted training + early-stop\n"
            "    on UPB-weighted validation RMSE).\n"
            '    """\n'
            "    def __init__(self, hidden=(16, 32, 16), lr=1e-3, batch_size=8192,\n"
            "                 max_epochs=50, patience=5):\n"
            "        self.hidden = tuple(hidden); self.lr = lr\n"
            "        self.batch_size = batch_size\n"
            "        self.max_epochs = max_epochs; self.patience = patience\n\n"
            "    def _build(self, n_features):\n"
            "        layers, prev = [], n_features\n"
            "        for h in self.hidden:\n"
            "            layers += [nn.Linear(prev, h), nn.LeakyReLU(0.1)]\n"
            "            prev = h\n"
            "        layers += [nn.Linear(prev, 1), nn.Sigmoid()]\n"
            "        return nn.Sequential(*layers)\n\n"
            "    def fit(self, X, y, X_val=None, y_val=None, sample_weight=None):\n"
            "        # Hypotheses edit this body — change `self._build`, the optimizer,\n"
            "        # the loss kernel, or the architecture (e.g. swap MLP for an MoE).\n"
            "        # SHAPE CONTRACT (do not break): keep y_true and sample_weight\n"
            "        # 1-D, and reshape every PyTorch network output to 1-D BEFORE\n"
            "        # using it in a custom loss. nn.Linear(..., 1) returns shape\n"
            "        # (batch, 1); subtracting a 1-D y_true with shape (batch,)\n"
            "        # broadcasts to (batch, batch) and the loss is computed on an\n"
            "        # outer product — only the first prediction effectively\n"
            "        # contributes and gradients are silently wrong. Always do\n"
            "        # `pred = pred.reshape(-1)` (or `.view(-1)`) before the\n"
            "        # weighted-MSE / RMSE / Huber / quantile / etc. kernel.\n"
            "        self.net_ = self._build(X.shape[1])\n"
            "        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr)\n"
            "        Xt = torch.as_tensor(np.asarray(X), dtype=torch.float32)\n"
            "        yt = torch.as_tensor(np.asarray(y), dtype=torch.float32).reshape(-1)\n"
            "        wt = torch.as_tensor(\n"
            "            sample_weight if sample_weight is not None else np.ones(len(y)),\n"
            "            dtype=torch.float32,\n"
            "        ).reshape(-1)\n"
            "        best_val, stale = float('inf'), 0\n"
            "        for epoch in range(self.max_epochs):\n"
            "            perm = torch.randperm(len(Xt))\n"
            "            for i in range(0, len(Xt), self.batch_size):\n"
            "                idx = perm[i:i + self.batch_size]\n"
            "                pred = self.net_(Xt[idx]).reshape(-1)   # (B,1) -> (B,)\n"
            "                loss = (wt[idx] * (pred - yt[idx]) ** 2).sum() / wt[idx].sum()\n"
            "                opt.zero_grad(); loss.backward(); opt.step()\n"
            "            if X_val is not None:\n"
            "                with torch.no_grad():\n"
            "                    vp = self.net_(torch.as_tensor(np.asarray(X_val),\n"
            "                                                   dtype=torch.float32)).reshape(-1)\n"
            "                    vy = torch.as_tensor(np.asarray(y_val),\n"
            "                                         dtype=torch.float32).reshape(-1)\n"
            "                    val_rmse = float(torch.sqrt(((vp - vy) ** 2).mean()))\n"
            "                if val_rmse + 1e-6 < best_val:\n"
            "                    best_val, stale = val_rmse, 0\n"
            "                else:\n"
            "                    stale += 1\n"
            "                    if stale >= self.patience:\n"
            "                        break\n"
            "        return self\n\n"
            "    def predict(self, X):\n"
            "        with torch.no_grad():\n"
            "            return self.net_(torch.as_tensor(np.asarray(X),\n"
            "                                             dtype=torch.float32)).numpy().ravel()\n\n"
            "def build_model():\n"
            '    """Return a fresh, unfitted estimator.\n\n'
            "    Model hypotheses edit THIS function's body inline (e.g. change\n"
            "    architecture from MLP to MoE, add a residual head, swap loss).\n"
            "    Ensemble hypotheses rewrite this function to return a wrapper\n"
            "    estimator whose sub-models are defined as helper classes/functions\n"
            "    ABOVE this function in the same main.py. No separate files.\n"
            '    """\n'
            "    return MLPRegressor()\n\n"
            'if __name__ == "__main__":\n'
            '    data_dir = os.environ.get("DATA_DIR", ".")\n'
            '    output_dir = "."\n'
            "    result = run_scaffold_pipeline(\n"
            "        panel_path=os.path.join(data_dir, \"tfminput.pkl\"),\n"
            "        scaler_path=os.path.join(data_dir, \"scaler.sav\"),\n"
            "        model_builder=build_model,\n"
            "        output_dir=output_dir,\n"
            "    )\n"
            "    # Verify that run_scaffold_pipeline produced scores.json\n"
            "    assert Path(output_dir, \"scores.json\").exists(), (\n"
            '        "scores.json missing — run_scaffold_pipeline did not complete"\n'
            "    )\n"
            "```\n\n"
            "The scaffold handles EVERYTHING: data loading, CUSIP split, model.fit() "
            "with X_val/y_val/sample_weight, clipping, scoring, and writing ALL output "
            "files (`submission.csv`, `scores.csv`, `scores.json`). "
            "**Do NOT write any of those files yourself.** "
            "If `scores.json` is missing after running, the experiment is FAILED.\n\n"
            "`build_model()` must return an unfitted sklearn-compatible estimator "
            "whose `.fit(X, y, X_val=..., y_val=..., sample_weight=...)` and "
            "`.predict(X) -> np.ndarray` signatures match the scaffold's call "
            "convention. All model / ensemble logic lives inline in main.py."
        )

        # 5) Data contract reminder
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
            f"- Split (seed={self.mbs_splitter.random_seed}, fixed every loop):\n"
            f"    * Test CUSIPs  : {self.mbs_splitter.test_fraction:.4f} of all CUSIPs, "
            "ALL time rows — identical across loops for fair comparison.\n"
            f"    * Train CUSIPs : {1.0 - self.mbs_splitter.val_fraction:.0%} of remaining, "
            f"fh_effdt ≤ {self.mbs_splitter.train_end_date}.\n"
            f"    * Val CUSIPs   : {self.mbs_splitter.val_fraction:.0%} of remaining, "
            f"fh_effdt ≤ {self.mbs_splitter.train_end_date} — for early stopping.\n"
            "- The scaffold passes X_val / y_val to model.fit() so GBM early-stopping\n"
            "  and PyTorch validation loops can consume them. Sklearn models that\n"
            "  do not accept these kwargs fall back automatically.\n"
            "- Scoring: scaffold uses scaler.sav to inverse-transform GNMA "
            "features back to raw units (WAC in %, "
            "Avg_Prop_Refi_Incentive_WAC_30yr_2mos in raw incentive units, "
            "etc.) so the harness measures per-coupon RMSE, S-curve R², "
            "inflection point, and seasoning effects on the real scale.\n"
            "- After each successful run the scaffold appends test-set predictions "
            f"(loop_number, cusip, fh_effdt, fh_upb, smm_decimal, smm_decimal_pred) "
            f"to `mbs_output/{MBSP_SETTINGS.test_predictions_filename}`.\n"
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
            f"{MBSP_SETTINGS.train_end_date}. Metric: fh_upb-weighted RMSE "
            "of SMM_DECIMAL, where the per-row weight is "
            "min(fh_upb, 150_000_000). All scorecard RMSE diagnostics "
            "(per-coupon-bucket, regime-transition, rolling 12-month) use "
            "the same UPB weights. Current SOTA is a PyTorch MLP "
            "[10, 20, 10] Leaky ReLU + Sigmoid trained with fh_upb-weighted "
            "RMSE (NOT plain RMSE); any reproduction must use the "
            "UPB-weighted loss. See scorecard from MBSEvaluationHarness."
        )
