"""
code_generator node — render Jinja2 template or LLM-fix existing code.

First pass  : renders the appropriate .jinja template deterministically (fast, no LLM).
Fix pass    : if validation_errors exist on existing code, uses LLM to fix it instead
              of re-rendering the template.
"""

import re
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import APIBackend
from rdagent.utils.agent.tpl import T

from ..conf import MBS_SETTINGS
from ..state import AnalysisPlan, MBSAnalysisState

TEMPLATES_DIR = Path(__file__).parent.parent / "templates"

_TEMPLATE_MAP = {
    "cusip_attribution": "ig_cusip_attribution.py.jinja",
    "scenario_comparison": "ig_scenario_comparison.py.jinja",
}


def _extract_code_block(text: str) -> str:
    """Extract content from a ```python ... ``` code block, or return text as-is."""
    match = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def code_generator_node(state: MBSAnalysisState) -> dict:
    """
    Generate or fix analysis code.
    Returns partial state update: generated_code, validation_errors (cleared).
    """
    plan = AnalysisPlan.model_validate(state["analysis_plan"])
    existing_code = state.get("generated_code", "")
    validation_errors = state.get("validation_errors", [])

    # Fix pass: existing code has validation errors → ask LLM to fix
    if existing_code and validation_errors:
        logger.info(f"LLM fix pass: addressing {len(validation_errors)} validation error(s).")
        sys_prompt = T(".prompts:code_generator.fix_system").r()
        user_prompt = T(".prompts:code_generator.fix_user").r(
            code=existing_code,
            errors=validation_errors,
            plan=plan.model_dump(),
        )
        resp = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=sys_prompt,
        )
        fixed_code = _extract_code_block(resp)
        logger.info("LLM fix applied.")
        return {"generated_code": fixed_code, "validation_errors": []}

    # First pass: render Jinja2 template deterministically
    template_name = _TEMPLATE_MAP.get(plan.analysis_type)
    if template_name is None:
        raise ValueError(f"Unknown analysis_type: {plan.analysis_type!r}")

    logger.info(f"Rendering template: {template_name}")
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        undefined=StrictUndefined,
    )
    template = env.get_template(template_name)
    code = template.render(
        plan=plan.model_dump(),
        mbs_settings=MBS_SETTINGS.model_dump(mode="json"),
        output_dir=str(MBS_SETTINGS.output_dir.resolve()),  # absolute so subprocess cwd doesn't matter
    )
    logger.info("Template rendered successfully.")
    return {"generated_code": code, "validation_errors": []}
