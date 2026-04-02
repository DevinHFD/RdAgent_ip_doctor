"""
code_validator node — pure Python, no LLM required.

Checks generated code for:
1. Python syntax validity (ast.parse)
2. Forbidden patterns (security / sandbox enforcement)
3. Output contract (code must write output.json)
"""

import ast
import re

from rdagent.log import rdagent_logger as logger

from ..state import MBSAnalysisState

# Patterns that must not appear in generated code.
# Generated scripts run in a subprocess but we still enforce this to prevent
# accidental or injected dangerous calls.
_FORBIDDEN_PATTERNS: list[tuple[str, str]] = [
    (r"\bos\.system\b", "os.system"),
    (r"\bos\.popen\b", "os.popen"),
    (r"\bsubprocess\b", "subprocess"),
    (r"(?<!\.)(\beval\s*\()", "eval()"),  # bare eval() only; .eval() method calls are allowed
    (r"\bexec\s*\(", "exec()"),
    (r"\b__import__\s*\(", "__import__()"),
    (r"\bsocket\b", "socket"),
    (r"\bshutil\.rmtree\b", "shutil.rmtree"),
    (r"\bpickle\.loads\b", "pickle.loads"),
]


def code_validator_node(state: MBSAnalysisState) -> dict:
    """
    Validate generated code. Returns partial state update:
      code_valid: bool
      validation_errors: list[str]
    """
    code = state.get("generated_code", "")
    errors: list[str] = []

    if not code.strip():
        return {"code_valid": False, "validation_errors": ["Generated code is empty."]}

    # 1. Syntax check
    try:
        ast.parse(code)
    except SyntaxError as e:
        errors.append(f"SyntaxError at line {e.lineno}: {e.msg}")

    # 2. Forbidden pattern scan
    for pattern, label in _FORBIDDEN_PATTERNS:
        if re.search(pattern, code):
            errors.append(f"Forbidden pattern '{label}' found in generated code.")

    # 3. Output contract: generated code must write output.json
    if "output.json" not in code:
        errors.append("Generated code must write results to 'output.json' (output contract).")

    if errors:
        logger.warning(f"Code validation failed with {len(errors)} error(s): {errors}")
    else:
        logger.info("Code validation passed.")

    return {
        "code_valid": len(errors) == 0,
        "validation_errors": errors,
    }
