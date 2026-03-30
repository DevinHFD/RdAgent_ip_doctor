from pathlib import Path

from pydantic_settings import SettingsConfigDict

from rdagent.core.conf import ExtendedBaseSettings


class MBSPrepaymentSettings(ExtendedBaseSettings):
    model_config = SettingsConfigDict(env_prefix="MBS_", protected_namespaces=())

    # Filesystem paths
    model_checkpoint_dir: Path = Path("./mbs_models")
    # Single parquet file containing all loans: must have cusip_col, date_col, and all feature columns
    data_file: Path = Path("./mbs_data/data.parquet")
    output_dir: Path = Path("./mbs_output")
    scaler_path: Path = Path("./mbs_models/scaler.sav")  # joblib-saved sklearn scaler (StandardScaler)

    # Column names in the parquet file
    cusip_col: str = "cusip"      # column holding the CUSIP identifier
    date_col: str = "fh_effdt"   # column holding the effective date (date or string)

    # Integrated Gradients defaults
    ig_baseline_strategy: str = "zero"  # "zero" | "mean" (mean = prev month/base scenario as baseline)
    ig_n_steps: int = 50
    ig_target_output: str = "cpr"  # "cpr" | "smm"  (model output index 0=CPR, 1=SMM)
    ig_batch_size: int = 32  # CUSIPs per batch to avoid OOM

    # Execution safety
    execution_timeout_seconds: int = 300
    max_debug_attempts: int = 3

    # Workflow control
    max_iterations: int = 10  # hard cap on planner loops before forcing reporter
    skip_human_review: bool = False  # set True for automated testing


MBS_SETTINGS = MBSPrepaymentSettings()
