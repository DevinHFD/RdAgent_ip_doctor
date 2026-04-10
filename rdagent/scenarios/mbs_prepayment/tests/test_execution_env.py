"""Tests for MBS execution environment (Priority 9)."""
import time

import pytest

from rdagent.scenarios.mbs_prepayment.execution_env import (
    ArtifactCache,
    ComputeBudget,
    DEFAULT_BUDGETS,
    IncrementalRunner,
    MountedDataset,
    Stage,
    fingerprint,
    get_budget,
    hash_file,
    hash_source,
)


# ---------------------------------------------------------------------------
# Budgets
# ---------------------------------------------------------------------------


@pytest.mark.offline
def test_default_budgets_exist_for_every_stage():
    for stage in Stage:
        budget = get_budget(stage)
        assert isinstance(budget, ComputeBudget)
        assert budget.stage == stage
        assert budget.timeout_seconds > 0


@pytest.mark.offline
def test_only_training_and_attribution_allow_gpu():
    assert DEFAULT_BUDGETS[Stage.TRAINING].allow_gpu
    assert DEFAULT_BUDGETS[Stage.ATTRIBUTION].allow_gpu
    assert not DEFAULT_BUDGETS[Stage.FEATURE_ENG].allow_gpu
    assert not DEFAULT_BUDGETS[Stage.EVALUATION].allow_gpu


@pytest.mark.offline
def test_budget_override():
    override = ComputeBudget(stage=Stage.FEATURE_ENG, timeout_seconds=10, memory_limit_gb=1, allow_gpu=False)
    b = get_budget(Stage.FEATURE_ENG, {Stage.FEATURE_ENG: override})
    assert b.timeout_seconds == 10


# ---------------------------------------------------------------------------
# Fingerprinting
# ---------------------------------------------------------------------------


@pytest.mark.offline
def test_fingerprint_is_stable_across_dict_order():
    a = fingerprint({"code": "x", "seed": 42, "data": "abc"})
    b = fingerprint({"seed": 42, "data": "abc", "code": "x"})
    assert a == b


@pytest.mark.offline
def test_fingerprint_changes_when_any_input_changes():
    base = {"code": "x", "seed": 42}
    assert fingerprint(base) != fingerprint({"code": "y", "seed": 42})
    assert fingerprint(base) != fingerprint({"code": "x", "seed": 43})


@pytest.mark.offline
def test_hash_source_is_stable_and_sensitive():
    assert hash_source("def f(): return 1") == hash_source("def f(): return 1")
    assert hash_source("def f(): return 1") != hash_source("def f(): return 2")


@pytest.mark.offline
def test_hash_file_matches_content(tmp_path):
    p = tmp_path / "data.bin"
    p.write_bytes(b"hello world")
    h1 = hash_file(p)
    h2 = hash_file(p)
    assert h1 == h2
    p.write_bytes(b"hello worlds")
    assert hash_file(p) != h1


# ---------------------------------------------------------------------------
# Artifact cache
# ---------------------------------------------------------------------------


@pytest.mark.offline
def test_cache_store_and_lookup(tmp_path):
    cache = ArtifactCache(cache_dir=tmp_path / "cache")
    fp = "abc123"
    assert not cache.has(Stage.FEATURE_ENG, fp)

    manifest = cache.store(
        stage=Stage.FEATURE_ENG,
        fp=fp,
        inputs={"code": "x"},
        files=[("features.parquet", b"dummy parquet bytes")],
        wall_seconds=1.5,
        random_seed=42,
    )
    assert cache.has(Stage.FEATURE_ENG, fp)
    loaded = cache.lookup(Stage.FEATURE_ENG, fp)
    assert loaded is not None
    assert loaded.fingerprint == fp
    assert loaded.wall_seconds == 1.5
    assert loaded.random_seed == 42
    assert "features.parquet" in loaded.artifact_files


@pytest.mark.offline
def test_cache_store_atomically_replaces_existing(tmp_path):
    cache = ArtifactCache(cache_dir=tmp_path / "cache")
    fp = "zzz"
    cache.store(
        stage=Stage.TRAINING, fp=fp, inputs={"v": 1},
        files=[("model.pkl", b"v1")], wall_seconds=0.5,
    )
    cache.store(
        stage=Stage.TRAINING, fp=fp, inputs={"v": 2},
        files=[("model.pkl", b"v2")], wall_seconds=0.5,
    )
    # Most-recent write wins
    manifest = cache.lookup(Stage.TRAINING, fp)
    assert manifest.inputs["v"] == 2
    assert (cache.artifact_path(Stage.TRAINING, fp, "model.pkl")).read_bytes() == b"v2"


@pytest.mark.offline
def test_cache_rejects_bad_file_names(tmp_path):
    cache = ArtifactCache(cache_dir=tmp_path / "cache")
    with pytest.raises(ValueError):
        cache.store(
            stage=Stage.FEATURE_ENG, fp="fp",
            inputs={}, files=[("../escape.bin", b"x")], wall_seconds=0.1,
        )
    with pytest.raises(ValueError):
        cache.store(
            stage=Stage.FEATURE_ENG, fp="fp2",
            inputs={}, files=[(".hidden", b"x")], wall_seconds=0.1,
        )


@pytest.mark.offline
def test_cache_stats_and_clear(tmp_path):
    cache = ArtifactCache(cache_dir=tmp_path / "cache")
    for stage, fp in [(Stage.FEATURE_ENG, "a"), (Stage.FEATURE_ENG, "b"), (Stage.TRAINING, "c")]:
        cache.store(stage=stage, fp=fp, inputs={}, files=[("x.bin", b"x")], wall_seconds=0.1)
    stats = cache.cache_stats()
    assert stats["feature_eng"] == 2
    assert stats["training"] == 1
    removed = cache.clear(Stage.FEATURE_ENG)
    assert removed == 2
    assert cache.cache_stats()["feature_eng"] == 0
    assert cache.cache_stats()["training"] == 1


# ---------------------------------------------------------------------------
# Incremental runner
# ---------------------------------------------------------------------------


@pytest.mark.offline
def test_incremental_runner_cache_hit_avoids_recompute(tmp_path):
    runner = IncrementalRunner(cache=ArtifactCache(cache_dir=tmp_path / "cache"))
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return "payload", {"out.bin": b"ok"}

    # First call — miss
    r1 = runner.run_stage(stage=Stage.FEATURE_ENG, inputs={"code": "v1"}, compute=compute)
    assert not r1.cache_hit
    assert calls["n"] == 1

    # Second call with same inputs — hit, compute NOT invoked
    r2 = runner.run_stage(stage=Stage.FEATURE_ENG, inputs={"code": "v1"}, compute=compute)
    assert r2.cache_hit
    assert calls["n"] == 1
    assert r2.fingerprint == r1.fingerprint


@pytest.mark.offline
def test_incremental_runner_propagates_upstream_changes(tmp_path):
    """Training stage rebuilds when feature fingerprint changes."""
    runner = IncrementalRunner(cache=ArtifactCache(cache_dir=tmp_path / "cache"))
    train_calls = {"n": 0}

    def feat_compute_v1():
        return None, {"features.bin": b"f_v1"}

    def feat_compute_v2():
        return None, {"features.bin": b"f_v2"}

    def train_compute():
        train_calls["n"] += 1
        return None, {"model.bin": b"m"}

    # v1 features → train once
    f1 = runner.run_stage(stage=Stage.FEATURE_ENG, inputs={"code": "fv1"}, compute=feat_compute_v1)
    runner.run_stage(
        stage=Stage.TRAINING,
        inputs={"code": "mv1", "upstream_fp": f1.fingerprint},
        compute=train_compute,
    )
    assert train_calls["n"] == 1

    # Same feature code again → both cache-hit, train NOT re-run
    f1b = runner.run_stage(stage=Stage.FEATURE_ENG, inputs={"code": "fv1"}, compute=feat_compute_v1)
    assert f1b.cache_hit
    t_same = runner.run_stage(
        stage=Stage.TRAINING,
        inputs={"code": "mv1", "upstream_fp": f1b.fingerprint},
        compute=train_compute,
    )
    assert t_same.cache_hit
    assert train_calls["n"] == 1

    # Feature code changes → feature re-runs AND training re-runs (upstream changed)
    f2 = runner.run_stage(stage=Stage.FEATURE_ENG, inputs={"code": "fv2"}, compute=feat_compute_v2)
    assert not f2.cache_hit
    t2 = runner.run_stage(
        stage=Stage.TRAINING,
        inputs={"code": "mv1", "upstream_fp": f2.fingerprint},
        compute=train_compute,
    )
    assert not t2.cache_hit
    assert train_calls["n"] == 2


@pytest.mark.offline
def test_incremental_runner_reuses_features_when_only_model_changes(tmp_path):
    """The whole point: changing only the model code should NOT re-run features."""
    runner = IncrementalRunner(cache=ArtifactCache(cache_dir=tmp_path / "cache"))
    feat_calls = {"n": 0}

    def feat_compute():
        feat_calls["n"] += 1
        return None, {"features.bin": b"f"}

    def train_compute():
        return None, {"model.bin": b"m"}

    f1 = runner.run_stage(stage=Stage.FEATURE_ENG, inputs={"code": "fv1"}, compute=feat_compute)
    runner.run_stage(
        stage=Stage.TRAINING, inputs={"code": "mv1", "upstream_fp": f1.fingerprint},
        compute=train_compute,
    )
    assert feat_calls["n"] == 1

    # Model changes; features unchanged → feat hits cache, train misses
    f2 = runner.run_stage(stage=Stage.FEATURE_ENG, inputs={"code": "fv1"}, compute=feat_compute)
    assert f2.cache_hit
    t2 = runner.run_stage(
        stage=Stage.TRAINING, inputs={"code": "mv2", "upstream_fp": f2.fingerprint},
        compute=train_compute,
    )
    assert not t2.cache_hit
    assert feat_calls["n"] == 1  # feature computation still only once


# ---------------------------------------------------------------------------
# Mounted dataset
# ---------------------------------------------------------------------------


@pytest.mark.offline
def test_mounted_dataset_fingerprint_changes_with_content(tmp_path):
    ds_root = tmp_path / "mbs_panel"
    ds_root.mkdir()
    (ds_root / "part_a.parquet").write_bytes(b"aaa")
    (ds_root / "part_b.parquet").write_bytes(b"bbb")

    ds = MountedDataset(root=ds_root)
    assert ds.exists()
    fp1 = ds.content_fingerprint()

    # Same content → same fingerprint
    fp_same = MountedDataset(root=ds_root).content_fingerprint()
    assert fp_same == fp1

    # Modify a file → fingerprint changes
    (ds_root / "part_b.parquet").write_bytes(b"bbbx")
    fp2 = MountedDataset(root=ds_root).content_fingerprint()
    assert fp2 != fp1


@pytest.mark.offline
def test_mounted_dataset_empty_returns_sentinel(tmp_path):
    ds_root = tmp_path / "empty"
    ds_root.mkdir()
    ds = MountedDataset(root=ds_root)
    assert ds.content_fingerprint() == "empty"
