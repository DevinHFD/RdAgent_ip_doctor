"""MBS Execution Environment — Priority 9.

Implements Direction #9 (Execution Environment and Compute Strategy). The
stock RD-Agent loop runs every experiment from scratch in Docker with a
fixed timeout; for MBS that's wasteful in several ways:

    1. **Incremental caching**: If only the feature code changed, the
       feature matrix must be recomputed but the model weights from a
       prior run of the same model on the same features are still valid.
       If only the model code changed, the cached feature matrix can be
       reused and only training re-runs. This module implements a
       content-addressed artifact cache keyed on the canonical
       fingerprint of the upstream code + data + config, so an iteration
       that touches only one stage pays only that stage's cost.

    2. **Compute budget allocation**: Feature engineering, training,
       and evaluation have very different compute profiles. A tight
       budget on feature eng catches infinite loops early; a generous
       budget on training accommodates neural nets. This module exposes
       per-stage `ComputeBudget` objects that the runner reads.

    3. **Reproducibility infrastructure**: Every cached artifact records
       its fingerprint, the package versions that produced it, the
       random seed, and the wall-clock time. For MBS this is not just
       ergonomics — regulatory review requires that any model decision
       can be reproduced from the recorded state.

The artifact cache is intentionally plain-filesystem + JSON manifest. No
database, no daemon. Each artifact lives at
`<cache_dir>/<stage>/<fingerprint>/` with a `manifest.json` next to the
actual payload file(s).
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable


class Stage(str, Enum):
    """Coarse stages of an MBS iteration with distinct compute profiles."""

    DATA_LOAD = "data_load"
    FEATURE_ENG = "feature_eng"
    TRAINING = "training"
    EVALUATION = "evaluation"
    ATTRIBUTION = "attribution"


# ---------------------------------------------------------------------------
# Per-stage compute budgets
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ComputeBudget:
    """Resource envelope for a single stage run."""

    stage: Stage
    timeout_seconds: int
    memory_limit_gb: float
    allow_gpu: bool
    max_retries: int = 0
    cpu_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage.value,
            "timeout_seconds": self.timeout_seconds,
            "memory_limit_gb": self.memory_limit_gb,
            "allow_gpu": self.allow_gpu,
            "max_retries": self.max_retries,
            "cpu_count": self.cpu_count,
        }


#: Default budgets tuned for MBS workloads. Feature engineering is tight
#: (most runs should be <60s — if a feature function takes 10 minutes it's
#: a bug). Training has the most generous budget and is the only stage
#: allowed to touch the GPU. Evaluation is CPU-bound metric computation.
DEFAULT_BUDGETS: dict[Stage, ComputeBudget] = {
    Stage.DATA_LOAD: ComputeBudget(
        stage=Stage.DATA_LOAD, timeout_seconds=300, memory_limit_gb=16.0,
        allow_gpu=False, max_retries=1, cpu_count=4,
    ),
    Stage.FEATURE_ENG: ComputeBudget(
        stage=Stage.FEATURE_ENG, timeout_seconds=600, memory_limit_gb=24.0,
        allow_gpu=False, max_retries=0, cpu_count=8,
    ),
    Stage.TRAINING: ComputeBudget(
        stage=Stage.TRAINING, timeout_seconds=3600, memory_limit_gb=32.0,
        allow_gpu=True, max_retries=1, cpu_count=8,
    ),
    Stage.EVALUATION: ComputeBudget(
        stage=Stage.EVALUATION, timeout_seconds=300, memory_limit_gb=8.0,
        allow_gpu=False, max_retries=0, cpu_count=4,
    ),
    Stage.ATTRIBUTION: ComputeBudget(
        stage=Stage.ATTRIBUTION, timeout_seconds=900, memory_limit_gb=16.0,
        allow_gpu=True, max_retries=0, cpu_count=4,
    ),
}


def get_budget(stage: Stage, overrides: dict[Stage, ComputeBudget] | None = None) -> ComputeBudget:
    if overrides and stage in overrides:
        return overrides[stage]
    return DEFAULT_BUDGETS[stage]


# ---------------------------------------------------------------------------
# Fingerprinting: canonical hash of upstream inputs for cache keying
# ---------------------------------------------------------------------------


def _stable_json(obj: Any) -> str:
    """JSON dump with sorted keys so dicts hash deterministically."""
    return json.dumps(obj, sort_keys=True, default=str, separators=(",", ":"))


def fingerprint(inputs: dict[str, Any]) -> str:
    """Return a stable SHA-256 hex digest of the given inputs dict.

    `inputs` should contain every upstream thing that could change the
    result of the stage: source-code text, config values, data-file
    content hashes (NOT mtimes), random seed. Dicts are sorted; lists
    are *not* sorted (order matters). Bytes are hashed directly.
    """
    h = hashlib.sha256()
    h.update(_stable_json(inputs).encode("utf-8"))
    return h.hexdigest()


def hash_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 hex digest of a file's contents. Used for data fingerprints."""
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def hash_source(source: str) -> str:
    """SHA-256 hex digest of a source-code string. Used for code fingerprints."""
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Artifact cache
# ---------------------------------------------------------------------------


@dataclass
class ArtifactManifest:
    """Metadata next to a cached artifact.

    Records everything needed to (a) verify a cache hit is actually valid
    and (b) reproduce the artifact from scratch if the cache is gone.
    """

    stage: str
    fingerprint: str
    created_ts: float
    wall_seconds: float
    inputs: dict[str, Any]           # the dict that was hashed to get `fingerprint`
    artifact_files: list[str]        # basenames relative to the artifact directory
    env: dict[str, str]              # package / python / platform versions
    random_seed: int | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True, default=str)

    @classmethod
    def from_file(cls, path: Path) -> ArtifactManifest:
        data = json.loads(Path(path).read_text())
        return cls(**data)


def current_environment() -> dict[str, str]:
    """Collect package versions and platform info for reproducibility."""
    env: dict[str, str] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    # Probe key packages without failing if any is missing
    for pkg in ("numpy", "pandas", "sklearn", "torch", "lightgbm", "xgboost", "captum"):
        try:
            mod = __import__(pkg)
        except ImportError:
            continue
        version = getattr(mod, "__version__", None)
        if version:
            env[pkg] = str(version)
    return env


@dataclass
class ArtifactCache:
    """Content-addressed filesystem cache for stage artifacts.

    Layout:

        <cache_dir>/<stage>/<fingerprint>/manifest.json
        <cache_dir>/<stage>/<fingerprint>/<payload files>

    Atomicity: writes go to a sibling `.staging/` directory and are moved
    into place only after the manifest is written — so a partial write
    can never be mistaken for a cache hit.
    """

    cache_dir: Path

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _dir(self, stage: Stage, fp: str) -> Path:
        return self.cache_dir / stage.value / fp

    def _staging_dir(self, stage: Stage, fp: str) -> Path:
        return self.cache_dir / stage.value / ".staging" / fp

    def has(self, stage: Stage, fp: str) -> bool:
        manifest = self._dir(stage, fp) / "manifest.json"
        return manifest.exists()

    def lookup(self, stage: Stage, fp: str) -> ArtifactManifest | None:
        manifest_path = self._dir(stage, fp) / "manifest.json"
        if not manifest_path.exists():
            return None
        try:
            return ArtifactManifest.from_file(manifest_path)
        except (json.JSONDecodeError, TypeError, KeyError):
            return None

    def artifact_path(self, stage: Stage, fp: str, filename: str) -> Path:
        return self._dir(stage, fp) / filename

    def store(
        self,
        *,
        stage: Stage,
        fp: str,
        inputs: dict[str, Any],
        files: Iterable[tuple[str, bytes]],
        wall_seconds: float,
        random_seed: int | None = None,
        extras: dict[str, Any] | None = None,
    ) -> ArtifactManifest:
        """Atomically store a set of files as the artifact for (stage, fp)."""
        staging = self._staging_dir(stage, fp)
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True, exist_ok=True)

        file_names: list[str] = []
        for name, payload in files:
            # Guard against path traversal in file names
            if "/" in name or "\\" in name or name.startswith("."):
                raise ValueError(f"Invalid artifact file name: {name!r}")
            (staging / name).write_bytes(payload)
            file_names.append(name)

        manifest = ArtifactManifest(
            stage=stage.value,
            fingerprint=fp,
            created_ts=time.time(),
            wall_seconds=wall_seconds,
            inputs=inputs,
            artifact_files=sorted(file_names),
            env=current_environment(),
            random_seed=random_seed,
            extras=extras or {},
        )
        (staging / "manifest.json").write_text(manifest.to_json())

        final_dir = self._dir(stage, fp)
        if final_dir.exists():
            shutil.rmtree(final_dir)
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        staging.rename(final_dir)
        return manifest

    def clear(self, stage: Stage | None = None) -> int:
        """Remove all artifacts (optionally only for one stage). Returns count removed."""
        count = 0
        if stage is None:
            for child in list(self.cache_dir.iterdir()):
                if child.is_dir():
                    count += sum(1 for _ in child.glob("*") if _.is_dir())
                    shutil.rmtree(child)
        else:
            stage_dir = self.cache_dir / stage.value
            if stage_dir.exists():
                count = sum(1 for p in stage_dir.glob("*") if p.is_dir())
                shutil.rmtree(stage_dir)
        return count

    def cache_stats(self) -> dict[str, int]:
        """Number of stored artifacts per stage."""
        stats: dict[str, int] = {}
        for stage in Stage:
            stage_dir = self.cache_dir / stage.value
            if not stage_dir.exists():
                stats[stage.value] = 0
                continue
            stats[stage.value] = sum(
                1
                for p in stage_dir.iterdir()
                if p.is_dir() and p.name != ".staging" and (p / "manifest.json").exists()
            )
        return stats


# ---------------------------------------------------------------------------
# Incremental runner: "only re-run the stages whose fingerprint changed"
# ---------------------------------------------------------------------------


@dataclass
class StageResult:
    stage: Stage
    fingerprint: str
    cache_hit: bool
    wall_seconds: float
    manifest: ArtifactManifest | None
    payload: Any | None = None


@dataclass
class IncrementalRunner:
    """Chain stages so each one reuses the cached output of prior stages.

    Usage sketch:

        runner = IncrementalRunner(cache=ArtifactCache(cache_dir))

        def run_features(prior_fps, inputs):
            # ... compute feature matrix ...
            return feature_df, {"features.parquet": feature_bytes}

        res_feat = runner.run_stage(
            stage=Stage.FEATURE_ENG,
            inputs={"code": feat_code_hash, "data": data_hash},
            compute=run_features,
        )

        res_train = runner.run_stage(
            stage=Stage.TRAINING,
            inputs={
                "code": model_code_hash,
                "upstream": res_feat.fingerprint,   # links to feature stage
                "seed": 42,
            },
            compute=run_training,
        )

    If only the model code changes between iterations, the feature stage
    hits the cache (same fingerprint) and training re-runs. If only the
    feature code changes, training's `upstream` component of its inputs
    changes, so training also re-runs. This is correct dependency
    propagation without any hand-written if/else.
    """

    cache: ArtifactCache
    budget_overrides: dict[Stage, ComputeBudget] = field(default_factory=dict)

    def budget(self, stage: Stage) -> ComputeBudget:
        return get_budget(stage, self.budget_overrides)

    def run_stage(
        self,
        *,
        stage: Stage,
        inputs: dict[str, Any],
        compute,
        random_seed: int | None = None,
    ) -> StageResult:
        """Compute or retrieve the artifact for one stage.

        `compute` is a zero-arg callable that returns either:
          - `(payload, {filename: bytes, ...})` — payload for in-memory
            downstream use, files for disk cache
          - `{filename: bytes, ...}` — files only
        """
        fp = fingerprint(inputs)
        manifest = self.cache.lookup(stage, fp)
        if manifest is not None:
            return StageResult(
                stage=stage,
                fingerprint=fp,
                cache_hit=True,
                wall_seconds=0.0,
                manifest=manifest,
                payload=None,
            )

        # Cache miss — compute
        if random_seed is not None:
            random.seed(random_seed)
            os.environ["PYTHONHASHSEED"] = str(random_seed)

        t0 = time.time()
        result = compute()
        elapsed = time.time() - t0

        if isinstance(result, tuple) and len(result) == 2:
            payload, files = result
        else:
            payload, files = None, result

        manifest = self.cache.store(
            stage=stage,
            fp=fp,
            inputs=inputs,
            files=files.items(),
            wall_seconds=elapsed,
            random_seed=random_seed,
        )
        return StageResult(
            stage=stage,
            fingerprint=fp,
            cache_hit=False,
            wall_seconds=elapsed,
            manifest=manifest,
            payload=payload,
        )


# ---------------------------------------------------------------------------
# Data mounting helpers
# ---------------------------------------------------------------------------


@dataclass
class MountedDataset:
    """Descriptor for pre-mounted MBS panel data.

    The execution environment mounts the CUSIP panel Parquet partitions
    once at container startup; each iteration reads from the standard
    path, not from raw CSVs. This dataclass encodes the contract and
    exposes a `content_fingerprint()` for use in stage inputs dicts.
    """

    root: Path
    partitions: tuple[str, ...] = ("fh_effdt",)   # partition columns
    file_format: str = "parquet"
    expected_columns: tuple[str, ...] = ("cusip", "fh_effdt", "smm_decimal", "coupon", "wala")

    def __post_init__(self) -> None:
        self.root = Path(self.root)

    def exists(self) -> bool:
        return self.root.exists()

    def files(self) -> list[Path]:
        if not self.exists():
            return []
        return sorted(self.root.rglob(f"*.{self.file_format}"))

    def content_fingerprint(self) -> str:
        """Hash every parquet file in the root — used in stage inputs dicts.

        For large datasets this can be cached itself (e.g., stored next
        to the dataset); for now compute fresh every call and rely on
        the upstream incremental cache to avoid redundant work.
        """
        files = self.files()
        if not files:
            return "empty"
        h = hashlib.sha256()
        for p in files:
            h.update(str(p.relative_to(self.root)).encode("utf-8"))
            h.update(hash_file(p).encode("utf-8"))
        return h.hexdigest()
