"""MBS-specific experiment feedback — wires the model validator persona,
MBS scorecard, and structured memory into the feedback LLM call.

Subclasses ``DSExperiment2Feedback`` so the generic DS loop can use it as
a drop-in replacement via ``MBSPrepaymentRDLoop``.
"""
from __future__ import annotations

import json
from typing import Dict

import pandas as pd

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.core.proposal import ExperimentFeedback, HypothesisFeedback
from rdagent.core.scenario import Scenario
from rdagent.log import rdagent_logger as logger
from rdagent.log.utils import dict_get_with_warning
from rdagent.oai.llm_utils import APIBackend
from rdagent.scenarios.data_science.dev.feedback import DSExperiment2Feedback
from rdagent.scenarios.data_science.experiment.experiment import DSExperiment
from rdagent.scenarios.data_science.proposal.exp_gen import DSTrace
from rdagent.scenarios.data_science.proposal.exp_gen.idea_pool import DSIdea
from rdagent.utils import convert2bool
from rdagent.utils.agent.tpl import T
from rdagent.utils.repo.diff import generate_diff_from_dict

from .memory import IterationPhase
from .personas import MODEL_VALIDATOR, PersonaKind, PersonaRouter


class MBSExperiment2Feedback(DSExperiment2Feedback):
    """Feedback generator that uses the MBS model validator persona and
    injects the MBS scorecard into the LLM prompt.
    """

    def __init__(self, scen: Scenario, version: str = "exp_feedback") -> None:
        super().__init__(scen, version=version)

    def generate_feedback(self, exp: DSExperiment, trace: DSTrace) -> ExperimentFeedback:
        sota_exp = trace.sota_experiment()
        sota_desc = T("scenarios.data_science.share:describe.exp").r(
            exp=sota_exp, heading="SOTA of previous exploration of the scenario"
        )

        feedback_desc = T("scenarios.data_science.share:describe.feedback").r(
            exp_and_feedback=trace.last_exp_fb(), heading="Previous Trial Feedback"
        )

        if sota_exp and sota_exp.experiment_workspace and exp.experiment_workspace:
            sota_exp_files = sota_exp.experiment_workspace.file_dict
            current_exp_files = exp.experiment_workspace.file_dict
            diff_edition = generate_diff_from_dict(sota_exp_files, current_exp_files)
        else:
            diff_edition = []

        cur_vs_sota_score = None
        if sota_exp:
            cur_score = pd.DataFrame(exp.result).loc["ensemble"].iloc[0]
            sota_score = pd.DataFrame(sota_exp.result).loc["ensemble"].iloc[0]
            cur_vs_sota_score = (
                f"The current score is {cur_score}, while the SOTA score is {sota_score}. "
                f"{'In this competition, higher is better.' if self.scen.metric_direction else 'In this competition, lower is better.'}"
            )

        eda_output = exp.experiment_workspace.file_dict.get("EDA.md", None)

        # --- MBS-specific additions ---

        # 1) Read MBS scorecard from workspace (written by scaffold).
        # The DS runner does not re-inject scores.json into file_dict, so we
        # read it directly from the workspace path (with a file_dict fallback).
        mbs_scorecard_text = ""
        scorecard = None
        ws_path = getattr(exp.experiment_workspace, "workspace_path", None)
        if ws_path is not None:
            scores_fp = ws_path / "scores.json"
            if scores_fp.exists():
                try:
                    scorecard = json.loads(scores_fp.read_text())
                except (json.JSONDecodeError, OSError):
                    logger.warning("MBSExperiment2Feedback: could not parse scores.json on disk")
        if scorecard is None:
            scorecard_raw = exp.experiment_workspace.file_dict.get("scores.json", None)
            if scorecard_raw:
                try:
                    scorecard = json.loads(scorecard_raw) if isinstance(scorecard_raw, str) else scorecard_raw
                except (json.JSONDecodeError, TypeError):
                    logger.warning("MBSExperiment2Feedback: could not parse scores.json")
        if scorecard:
            mbs_scorecard_text = (
                "\n\n## MBS Evaluation Scorecard (from MBSEvaluationHarness)\n"
                "```json\n"
                f"{json.dumps(scorecard, indent=2, default=str)}\n"
                "```\n"
                "Use this multi-dimensional scorecard to assess the experiment. "
                "An experiment that improves overall RMSE but degrades per-coupon "
                "RMSE uniformity or regime robustness is a REJECT."
            )

        # 2) MBS memory: feedback-phase context
        mbs_memory_text = ""
        if hasattr(self.scen, "mbs_memory"):
            mbs_memory_text = self.scen.mbs_memory.render_context(IterationPhase.FEEDBACK)

        # 3) Use model validator persona as system prompt prefix
        validator_preamble = MODEL_VALIDATOR.system_prompt

        # Build prompts — reuse DS templates but prepend MBS context
        base_scenario_desc = self.scen.get_scenario_all_desc(eda_output=eda_output)
        # Use absolute template path — `.prompts:` would resolve relative to
        # this module (rdagent/scenarios/mbs_prepayment/prompts.yaml), which
        # does not carry the DS exp_feedback schema. The DS feedback
        # templates live in scenarios/data_science/dev/prompts.yaml.
        system_prompt = T(f"scenarios.data_science.dev.prompts:{self.version}.system").r(
            scenario=base_scenario_desc
        )
        # Prepend the model validator persona
        system_prompt = validator_preamble + "\n\n" + system_prompt

        user_prompt = T(f"scenarios.data_science.dev.prompts:{self.version}.user").r(
            sota_desc=sota_desc,
            cur_exp=exp,
            diff_edition=diff_edition,
            feedback_desc=feedback_desc,
            cur_vs_sota_score=cur_vs_sota_score,
        )

        # Append MBS scorecard and memory context to the user prompt.
        # Prepayment-specific checks (burnout, per-coupon uniformity, regime
        # robustness) are already surfaced deterministically via the scorecard
        # block above, so the LLM reasons about them inline within the standard
        # "Observations" and "Feedback for Hypothesis" fields.
        user_prompt = (
            user_prompt
            + mbs_scorecard_text
            + "\n"
            + mbs_memory_text
        )

        resp_dict = json.loads(
            APIBackend().build_messages_and_create_chat_completion(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                json_mode=True,
                json_target_type=Dict[str, str | bool | int],
            )
        )

        if evaluation_not_aligned := dict_get_with_warning(resp_dict, "Evaluation Aligned With Task", "no") == "no":
            exp.result = None

        hypothesis_feedback = HypothesisFeedback(
            observations=dict_get_with_warning(resp_dict, "Observations", "No observations provided"),
            hypothesis_evaluation=dict_get_with_warning(resp_dict, "Feedback for Hypothesis", "No feedback provided"),
            new_hypothesis=dict_get_with_warning(resp_dict, "New Hypothesis", "No new hypothesis provided"),
            reason=dict_get_with_warning(resp_dict, "Reasoning", "No reasoning provided")
            + ("\nRejected because evaluation code not aligned with task." if evaluation_not_aligned else ""),
            code_change_summary=dict_get_with_warning(
                resp_dict, "Code Change Summary", "No code change summary provided"
            ),
            decision=(
                False
                if evaluation_not_aligned
                else convert2bool(dict_get_with_warning(resp_dict, "Replace Best Result", "no"))
            ),
            eda_improvement=dict_get_with_warning(resp_dict, "EDA Improvement", "no"),
            acceptable=convert2bool(dict_get_with_warning(resp_dict, "Acceptable", "no")),
        )

        if hypothesis_feedback and DS_RD_SETTING.enable_knowledge_base:
            ds_idea = DSIdea(
                {
                    "competition": self.scen.get_competition_full_desc(),
                    "idea": exp.hypothesis.hypothesis,
                    "method": exp.pending_tasks_list[0][0].get_task_information(),
                    "hypothesis": {exp.hypothesis.problem_label: exp.hypothesis.problem_desc},
                }
            )
            trace.knowledge_base.add_idea(idea=ds_idea)

        return hypothesis_feedback
