"""
question_parser node — classify question type and extract structured parameters.
"""

from rdagent.log import rdagent_logger as logger
from rdagent.utils.agent.tpl import T
from rdagent.utils.agent.workflow import build_cls_from_json_with_retry

from ..state import MBSAnalysisState


class _QuestionParseResult:
    """Thin wrapper so build_cls_from_json_with_retry can instantiate via **kwargs."""

    def __init__(self, question_type: str, cusip_list: list, scenario_params: dict):
        self.question_type = question_type
        self.cusip_list = cusip_list
        self.scenario_params = scenario_params


def question_parser_node(state: MBSAnalysisState) -> dict:
    """
    LLM call to classify the question and extract CUSIPs / scenario params.
    Returns partial state update.
    """
    question = state["question"]
    logger.info(f"Parsing question: {question[:120]}")

    sys_prompt = T(".prompts:question_parser.system").r()
    user_prompt = T(".prompts:question_parser.user").r(question=question)

    result: _QuestionParseResult = build_cls_from_json_with_retry(
        cls=_QuestionParseResult,
        system_prompt=sys_prompt,
        user_prompt=user_prompt,
        retry_n=5,
    )

    logger.info(
        f"Question classified as '{result.question_type}'. "
        f"CUSIPs: {result.cusip_list}. "
        f"Scenario params: {result.scenario_params}"
    )

    return {
        "question_type": result.question_type,
        "cusip_list": result.cusip_list,
        "scenario_params": result.scenario_params,
        "iteration_count": state.get("iteration_count", 0),
    }
