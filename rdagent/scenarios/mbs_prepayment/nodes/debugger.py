"""
debugger node — LLM-based runtime error analysis and code repair.

Called when the executor fails. Attempts to fix the generated code and routes
back through the validator before re-execution. Gives up after max_debug_attempts
and routes to human_reviewer so the human can see the error.
"""

import re

from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import APIBackend
from rdagent.utils.agent.tpl import T

from ..conf import MBS_SETTINGS
from ..state import MBSAnalysisState


def _extract_code_block(text: str) -> str:
    match = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def debugger_node(state: MBSAnalysisState) -> dict:
    """
    LLM diagnoses execution error and returns fixed code.
    Increments debug_attempts and clears execution_error so the executor re-runs clean.
    Returns partial state update.
    """
    attempts = state.get("debug_attempts", 0) + 1
    error = state.get("execution_error", "")
    code = state.get("generated_code", "")
    plan = state.get("analysis_plan", {})

    logger.info(
        f"Debug attempt {attempts}/{MBS_SETTINGS.max_debug_attempts}. "
        f"Error summary: {error[:200]}"
    )

    sys_prompt = T(".prompts:debugger.system").r()
    user_prompt = T(".prompts:debugger.user").r(
        code=code,
        error=error,
        plan=plan,
        attempt_number=attempts,
        max_attempts=MBS_SETTINGS.max_debug_attempts,
    )

    resp = APIBackend().build_messages_and_create_chat_completion(
        user_prompt=user_prompt,
        system_prompt=sys_prompt,
    )
    fixed_code = _extract_code_block(resp)

    logger.info(f"Debugger produced fixed code ({len(fixed_code)} chars).")

    return {
        "generated_code": fixed_code,
        "debug_attempts": attempts,
        "execution_error": None,   # cleared so executor re-runs fresh
        "code_valid": False,       # force re-validation before re-execution
        "validation_errors": [],
    }
