"""
executor node — runs generated attribution code in a subprocess.

The generated script must write output.json conforming to ExecutionResult schema.
Attribution values are IG output in SMM/CPR contribution units (no rescaling).
Feature values are inverse-transformed to original scale for context only.
"""

import json
import subprocess
import tempfile
from pathlib import Path

from rdagent.log import rdagent_logger as logger

from ..conf import MBS_SETTINGS
from ..state import ExecutionResult, MBSAnalysisState


def executor_node(state: MBSAnalysisState) -> dict:
    """
    Write generated code to a temp directory, execute as a subprocess,
    parse output.json into ExecutionResult.

    Returns partial state update:
      execution_result: dict | None
      execution_error: str | None
    """
    code = state["generated_code"]
    work_dir = Path(tempfile.mkdtemp(prefix="mbs_exec_"))
    script_path = work_dir / "run_attribution.py"
    script_path.write_text(code, encoding="utf-8")

    logger.info(f"Executing attribution analysis script in {work_dir}")

    try:
        proc = subprocess.run(
            ["python", str(script_path)],
            cwd=str(work_dir),
            capture_output=True,
            text=True,
            timeout=MBS_SETTINGS.execution_timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        msg = f"Execution timed out after {MBS_SETTINGS.execution_timeout_seconds}s."
        logger.warning(msg)
        return {"execution_result": None, "execution_error": msg}
    except Exception as e:
        msg = f"Unexpected subprocess error: {e}"
        logger.error(msg)
        return {"execution_result": None, "execution_error": msg}

    if proc.returncode != 0:
        error_msg = (
            f"Script exited with code {proc.returncode}.\n"
            f"STDOUT:\n{proc.stdout[-3000:]}\n"
            f"STDERR:\n{proc.stderr[-3000:]}"
        )
        logger.warning(f"Execution failed (exit {proc.returncode}).")
        return {"execution_result": None, "execution_error": error_msg}

    output_path = work_dir / "output.json"
    if not output_path.exists():
        msg = "output.json not found after successful script exit."
        logger.warning(msg)
        return {"execution_result": None, "execution_error": msg}

    try:
        raw = json.loads(output_path.read_text(encoding="utf-8"))
        result = ExecutionResult.model_validate(raw)
        logger.info(
            f"Execution succeeded. "
            f"CUSIPs in result: {len(result.attributions_normalized)}. "
            f"Features: {list(result.summary_stats.keys())[:5]}"
        )
        return {"execution_result": result.model_dump(), "execution_error": None}
    except Exception as e:
        msg = f"Failed to parse output.json: {e}"
        logger.error(msg)
        return {"execution_result": None, "execution_error": msg}
