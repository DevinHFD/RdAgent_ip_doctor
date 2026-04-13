"""MBS Prepayment RD Loop — subclasses DataScienceRDLoop to wire in
the Priority 1-10 MBS modules that the generic DS loop cannot reach.

What this loop changes vs the stock ``DataScienceRDLoop``:

1. **Feedback**: Uses ``MBSExperiment2Feedback`` which injects the MBS
   scorecard (from ``scores.json``) and the model-validator persona into
   the feedback LLM call.

2. **Domain validation**: After the runner executes code, reads the MBS
   scorecard and runs ``DomainValidator`` checks. If any check fails the
   experiment is auto-rejected with a diagnostic message — no feedback
   LLM call needed, saving a round trip.

3. **Record**: After the base record step, updates MBS memory
   (structured model properties), MBS search state (curriculum /
   cooldown), and evaluates phase gates.  When a gate passes, the
   orchestrator advances to the next phase automatically.

4. **Scenario description**: The scenario's ``get_scenario_all_desc()``
   override (in ``scenario.py``) already injects MBS memory, search
   strategy constraints, phase info, and data contract into every LLM
   call.  This loop ensures the orchestrator state stays up-to-date so
   those injections reflect reality.
"""
from __future__ import annotations

import json
from typing import Any, Optional

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.components.workflow.conf import BasePropSetting
from rdagent.core.exception import CoderError, RunnerError
from rdagent.core.proposal import ExperimentFeedback
from rdagent.core.utils import import_class
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.data_science.experiment.experiment import DSExperiment
from rdagent.scenarios.data_science.loop import DataScienceRDLoop
from rdagent.scenarios.data_science.proposal.exp_gen.base import DataScienceScen

from .feedback import MBSExperiment2Feedback
from .memory import ModelProperties, TraceEntry
from .orchestration import ValidationReport
from .search_strategy import IterationRecord


class MBSPrepaymentRDLoop(DataScienceRDLoop):
    """DataScienceRDLoop with MBS-specific orchestration, memory, domain
    validation, and phase-gated progression wired in.
    """

    def __init__(self, PROP_SETTING: BasePropSetting) -> None:
        super().__init__(PROP_SETTING)

        # The scenario must be an MBSPrepaymentScen (or compatible)
        scen = self.trace.scen
        if not hasattr(scen, "mbs_orchestrator"):
            raise TypeError(
                f"MBSPrepaymentRDLoop requires an MBSPrepaymentScen scenario, "
                f"got {type(scen).__name__}. Set DS_SCEN=rdagent.scenarios."
                f"mbs_prepayment.scenario.MBSPrepaymentScen in .env."
            )
        self.mbs_scen = scen

        # Replace the generic DS feedback summarizer with the MBS one
        self.summarizer = MBSExperiment2Feedback(scen)
        logger.info(
            f"MBSPrepaymentRDLoop initialized: phase={scen.mbs_orchestrator.current_phase.value}"
        )

    # ------------------------------------------------------------------
    # Override: feedback — add domain validation as auto-reject gate
    # ------------------------------------------------------------------

    def feedback(self, prev_out: dict[str, Any]) -> ExperimentFeedback:
        exp: DSExperiment = prev_out["running"]

        # Try to read the MBS scorecard written by the scaffold
        scorecard = self._read_scorecard(exp)

        # Run deterministic domain validation if we have scorecard data
        if scorecard:
            validation = self._domain_validate(scorecard)
            if not validation.ok:
                logger.warning(
                    f"MBS domain validation failed: {validation.auto_reject_reason}"
                )
                return ExperimentFeedback(
                    reason=(
                        "Auto-rejected by MBS DomainValidator (no LLM call):\n"
                        + validation.auto_reject_reason
                    ),
                    decision=False,
                )

        # Fall through to the LLM-based MBS feedback
        return super().feedback(prev_out)

    # ------------------------------------------------------------------
    # Override: record — update MBS memory, search state, phase gates
    # ------------------------------------------------------------------

    def record(self, prev_out: dict[str, Any]) -> None:
        # Run the base DS record logic (trace, SOTA, archiving)
        super().record(prev_out)

        # Now update MBS-specific state
        e = prev_out.get(self.EXCEPTION_KEY, None)
        loop_id = prev_out[self.LOOP_IDX_KEY]

        if e is not None:
            # Exception path — still update search state with failure
            exp = (
                prev_out.get("direct_exp_gen")
                if isinstance(e, CoderError)
                else prev_out.get("coding")
            )
            if exp is not None:
                self._update_search_state(
                    loop_id=loop_id,
                    exp=exp,
                    success=False,
                    scorecard=None,
                )
            return

        exp: DSExperiment = prev_out["running"]
        feedback: ExperimentFeedback = prev_out["feedback"]
        scorecard = self._read_scorecard(exp)
        success = bool(getattr(feedback, "decision", False))

        # Update structured memory
        self._update_memory(
            loop_id=loop_id,
            exp=exp,
            feedback=feedback,
            scorecard=scorecard,
            success=success,
        )

        # Update search state (curriculum, cooldown, exploration mode)
        self._update_search_state(
            loop_id=loop_id,
            exp=exp,
            success=success,
            scorecard=scorecard,
        )

        # Evaluate phase gates — advance if criteria are met
        gate_result = self.mbs_scen.mbs_orchestrator.evaluate_gate()
        logger.log_object(gate_result.to_dict(), tag="mbs_gate_result")
        if gate_result.passed:
            new_phase = self.mbs_scen.mbs_orchestrator.advance_phase()
            if new_phase:
                logger.info(f"MBS phase advanced to: {new_phase.value}")
                logger.log_object(
                    {"new_phase": new_phase.value},
                    tag="mbs_phase_advance",
                )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_scorecard(self, exp: DSExperiment) -> dict[str, Any] | None:
        """Read ``scores.json`` from the experiment workspace.

        The DS runner only loads ``scores.csv`` back into the experiment
        object; ``scores.json`` is written by the MBS scaffold to disk but
        is not re-injected into ``file_dict``.  We therefore read it
        directly from ``workspace_path`` (with a ``file_dict`` fallback in
        case a future code path does inject it).
        """
        if exp is None or exp.experiment_workspace is None:
            return None
        # Primary: read from disk (where the scaffold wrote it).
        ws_path = getattr(exp.experiment_workspace, "workspace_path", None)
        if ws_path is not None:
            scores_fp = ws_path / "scores.json"
            if scores_fp.exists():
                try:
                    return json.loads(scores_fp.read_text())
                except (json.JSONDecodeError, OSError):
                    logger.warning("MBSPrepaymentRDLoop: could not parse scores.json on disk")
        # Fallback: file_dict (older code paths / cached runs).
        raw = exp.experiment_workspace.file_dict.get("scores.json")
        if not raw:
            return None
        try:
            return json.loads(raw) if isinstance(raw, str) else raw
        except (json.JSONDecodeError, TypeError):
            logger.warning("MBSPrepaymentRDLoop: could not parse scores.json")
            return None

    def _domain_validate(self, scorecard: dict[str, Any]) -> ValidationReport:
        """Run the deterministic DomainValidator using scorecard fields.

        The raw ``y_pred`` / ``rate_incentive`` arrays are not available on
        the DS experiment object, but the MBS scaffold has already written
        aggregate diagnostics (overall_rmse, spearman monotonicity, per-
        dimension errors) to ``scores.json`` via ``MBSEvaluationHarness``.
        ``validate_from_scorecard`` inspects those directly, so the auto-
        reject gate fires on real failure modes.
        """
        return self.mbs_scen.mbs_validator.validate_from_scorecard(scorecard)

    def _update_memory(
        self,
        loop_id: int,
        exp: DSExperiment,
        feedback: ExperimentFeedback,
        scorecard: dict[str, Any] | None,
        success: bool,
    ) -> None:
        """Append a structured entry to MBS memory."""
        component = self._get_component(exp)
        hypothesis_text = str(getattr(exp, "hypothesis", ""))

        props = None
        if success and scorecard:
            props = ModelProperties.from_scorecard(
                iteration=loop_id,
                model_type=component,
                component_touched=component,
                scorecard=scorecard,
            )

        entry = TraceEntry(
            iteration=loop_id,
            component=component,
            hypothesis=hypothesis_text[:500],
            code_change_summary=getattr(feedback, "code_change_summary", ""),
            decision="accept" if success else "reject",
            success=success,
            properties=props,
            feedback_reason=getattr(feedback, "reason", "")[:500],
        )
        self.mbs_scen.mbs_memory.append_entry(entry)

    def _update_search_state(
        self,
        loop_id: int,
        exp: DSExperiment,
        success: bool,
        scorecard: dict[str, Any] | None,
    ) -> None:
        """Update the curriculum / cooldown search state."""
        component = self._get_component(exp)
        acc = (scorecard or {}).get("accuracy", {})
        rs = (scorecard or {}).get("rate_sensitivity", {})

        record = IterationRecord(
            iteration=loop_id,
            component_touched=component,
            overall_rmse=float(acc.get("overall_rmse", float("inf"))),
            rate_sensitivity_monotonic=float(rs.get("monotonicity_spearman", 0.0)) > 0.3,
            has_rate_incentive=(
                "rate_sensitivity" in (scorecard or {})
                and "_error" not in (scorecard or {}).get("rate_sensitivity", {})
            ),
            architecture_family=component,
            success=success,
        )
        self.mbs_scen.mbs_search_state.append(record)

    #: Default mapping from the DS proposal schema enum to the MBS component
    #: namespace used by MBSSearchState / MBSOrchestrator. The DS proposer
    #: cannot emit MBS-native names (its schema is a fixed Literal), so we
    #: translate at record-time. ``FeatureEng`` is ambiguous — it could be
    #: rate-curve, pool-dynamics, or macro — so ``_get_component`` inspects
    #: the hypothesis text to pick the most specific MBS feature family
    #: before falling back to this default.
    DS_TO_MBS_COMPONENT: dict[str, str] = {
        "DataLoadSpec": "DataLoader",
        "FeatureEng": "RateCurveFeatures",
        "Model": "PrepaymentModel",
        "Ensemble": "Ensemble",
        "Workflow": "ScenarioValidator",
    }

    #: MBS-native component names (from ``search_strategy.MBSComponent``) —
    #: if the hypothesis/component text mentions one of these directly we
    #: prefer it over the DS-enum translation.
    _MBS_COMPONENT_NAMES: tuple[str, ...] = (
        "DataLoader",
        "RateCurveFeatures",
        "PoolDynamics",
        "MacroFeatures",
        "PrepaymentModel",
        "ScenarioValidator",
        "Ensemble",
    )

    #: Keyword → MBS component heuristics for disambiguating ``FeatureEng``.
    #: Ordered by specificity; the first keyword found in the hypothesis text
    #: wins.
    _FEATURE_KEYWORDS: tuple[tuple[str, str], ...] = (
        ("burnout", "PoolDynamics"),
        ("seasoning", "PoolDynamics"),
        ("vintage", "PoolDynamics"),
        ("seasonality", "PoolDynamics"),
        ("unemployment", "MacroFeatures"),
        ("hpi", "MacroFeatures"),
        ("refi index", "MacroFeatures"),
        ("macro", "MacroFeatures"),
        ("s-curve", "RateCurveFeatures"),
        ("rate_incentive", "RateCurveFeatures"),
        ("rate incentive", "RateCurveFeatures"),
    )

    @classmethod
    def _get_component(cls, exp: DSExperiment) -> str:
        """Extract the MBS-namespace component touched by this experiment.

        Priority:
          1. If the hypothesis text directly mentions an MBS-native component
             name, use it (proposers sometimes echo the injected phase text).
          2. If the DS component is ``FeatureEng``, disambiguate via the
             hypothesis keyword map.
          3. Otherwise fall back to ``DS_TO_MBS_COMPONENT``.
        """
        if exp is None:
            return "unknown"
        hypo = getattr(exp, "hypothesis", None)
        if hypo is None:
            return "unknown"
        ds_component = getattr(hypo, "component", "unknown")
        hypo_text = (
            f"{getattr(hypo, 'hypothesis', '')} "
            f"{getattr(hypo, 'problem_desc', '')} "
            f"{getattr(hypo, 'reason', '')}"
        ).lower()

        # 1) direct MBS-native mention wins
        for mbs_name in cls._MBS_COMPONENT_NAMES:
            if mbs_name.lower() in hypo_text:
                return mbs_name

        # 2) FeatureEng: disambiguate via keywords
        if ds_component == "FeatureEng":
            for keyword, mbs_name in cls._FEATURE_KEYWORDS:
                if keyword in hypo_text:
                    return mbs_name

        # 3) default translation table
        return cls.DS_TO_MBS_COMPONENT.get(ds_component, ds_component)
