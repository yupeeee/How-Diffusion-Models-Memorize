"""Unexecuted numerical-stage routing, cache identity and preservation regressions.

Scientific workers are stubs here. Mathematical examples live in the separate
numerical engine tests; none of these orchestration fixtures uses a model.
"""
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from utils.common.io import atomic_write_frame_parquet, atomic_write_json, file_sha256, read_json
from utils.experiments.theory import numerical_reduce as stage
from utils.experiments.theory.contracts import NUMERICAL_KEYS, TheoryError, numerical_config
from utils.experiments.theory.paper_contracts import PaperPaths, saved_plot_configuration


def config(**changes):
    return numerical_config(model_name="sdv1", scheduler_name="ddim", **changes)


def row(index="001", seed=0):
    return dict(run_id="run", original_index=index, record_id="record-" + index,
                target_id="target-" + index, seed=seed, step_index=0, direct_prop5_log_probability_gain=-.125)


def forbidden(*args, **kwargs):
    raise AssertionError("Numerical-only retry invoked an upstream scientific stage")


@pytest.mark.parametrize("name,value", [
    ("numerical_decimal_precision", 31), ("numerical_decimal_precision", True),
    ("numerical_max_decimal_products", -1), ("numerical_max_decimal_products", .5),
    ("numerical_max_variation_nodes", 0), ("numerical_max_variation_nodes", False),
    ("numerical_variation_absolute_width", 0), ("numerical_variation_absolute_width", float("nan")),
])
def test_invalid_numerical_budgets_fail_before_science(name, value):
    with pytest.raises(TheoryError):
        config(**{name: value})


def test_budget_is_separate_from_input_and_observation_identities():
    first = config()
    changed = config(numerical_max_decimal_products=0, numerical_max_variation_nodes=2)
    inputs = stage.input_recipe()
    assert stage._science_without_policy(first) == stage._science_without_policy(changed)
    assert stage.policy_recipe(first) != stage.policy_recipe(changed)
    assert stage.input_recipe() == inputs
    assert not set(NUMERICAL_KEYS).intersection(inputs)


def test_old_saved_config_inherits_new_policy_defaults(tmp_path):
    saved = {name: value for name, value in config(num_loss_seeds=23).items() if name not in NUMERICAL_KEYS}
    atomic_write_json(tmp_path / "run_config.json", {"scientific_config": saved})
    inherited = saved_plot_configuration(tmp_path, requested=config(), explicit_keys=set())
    assert inherited["num_loss_seeds"] == 23
    assert all(inherited[key] == config()[key] for key in NUMERICAL_KEYS)


def test_compact_saved_evidence_join_keeps_legacy_values_and_every_identity(tmp_path):
    base = pd.DataFrame([row(seed=0), row(seed=1)])
    extra = base[stage.KEYS].iloc[::-1].assign(extra_measurement=[20., 10.])
    specs = {}
    for name, frame in (("matched_updates", base), ("extra", extra)):
        path = tmp_path / (name + ".parquet")
        atomic_write_frame_parquet(frame, path)
        specs[name] = {"path": str(path), "sha256": file_sha256(path), "rows": len(frame)}
    manifest = {"files": specs, "logical_joins": {"matched_updates": {
        "base": "matched_updates", "supplement": "extra", "keys": stage.KEYS,
        "how": "exact_one_to_one_additive"}}}
    result = stage._read_tables(manifest)["matched_updates"]
    assert result.extra_measurement.tolist() == [10., 20.]
    assert result.direct_prop5_log_probability_gain.tolist() == [-.125, -.125]
    (tmp_path / "extra.parquet").write_bytes(b"changed receipt")
    with pytest.raises(TheoryError, match="receipt differs"):
        stage._read_tables(manifest)


def test_numerical_retry_missing_evidence_never_falls_through_to_inference(tmp_path, monkeypatch):
    from utils.experiments.theory import cache_reader, evidence_reduce
    monkeypatch.setattr(cache_reader, "discover_sources", lambda *a, **k: SimpleNamespace())
    monkeypatch.setattr(evidence_reduce, "run_evidence_analysis", forbidden)
    with pytest.raises(TheoryError, match="No upstream work was invoked.*--recompute-experiments"):
        stage.run_precision_analysis(tmp_path, config=config(), refine_only=True)


def test_published_scientific_change_cannot_hide_behind_numerical_retry(tmp_path):
    output = PaperPaths.build(tmp_path, **config()).output_directory
    atomic_write_json(output / "run_config.json", {"scientific_config": config(num_loss_seeds=7)})
    with pytest.raises(TheoryError, match="beyond numerical policy"):
        stage.load_saved_evidence(tmp_path, config=config(), sources=SimpleNamespace())


def test_multiworker_policy_resume_preserves_observations_and_input_identity(tmp_path, monkeypatch):
    from utils.experiments.theory import cache_reader, direct_reduce, evidence_reduce, reduce
    core = tmp_path / "theory_measurements" / "direct_analysis" / "core"
    integrated = core / "integrations" / "integral"
    integrated.mkdir(parents=True)
    for name in ("reference_law.pt", "schedule.pt"):
        (core / name).write_bytes(b"protected tensor sentinel")
    records = [SimpleNamespace(original_index=index, metadata={"original_index": index, "seeds": [0]})
               for index in ("001", "002")]
    individual = {}
    for record in records:
        folder = integrated / "records" / record.original_index
        folder.mkdir(parents=True)
        atomic_write_json(folder / "complete.json", {"complete": True, "id": record.original_index})
        path = folder / "matched_updates.parquet"
        atomic_write_frame_parquet(pd.DataFrame([row(record.original_index)]), path)
        individual[record.original_index] = (folder, {"matched_updates": path})
    base = pd.DataFrame([row(record.original_index) for record in records])
    original = integrated / "matched_updates.parquet"
    atomic_write_frame_parquet(base, original)
    evidence_dir = tmp_path / "theory_measurements" / "evidence_supplements" / "fixed" / "collections" / "fixed"
    atomic_write_json(evidence_dir / "manifest.json", {"fixture": "immutable evidence"})
    evidence = {"tables": {"matched_updates": base}, "directory": evidence_dir,
        "files": {"matched_updates": {"path": str(original), "sha256": file_sha256(original), "rows": len(base)}},
        "logical_joins": {}, "provenance": {"analysis_hash": "old-evidence", "reference_law_hash": "law",
                                             "analysis_manifest": str(integrated / "analysis_manifest.json")}}
    protected = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in tmp_path.rglob("*") if path.is_file()}
    sources = SimpleNamespace(selected=records)
    monkeypatch.setattr(cache_reader, "discover_sources", lambda *a, **k: sources)
    monkeypatch.setattr(stage, "load_saved_evidence", lambda *a, **k: evidence)
    monkeypatch.setattr(stage, "_validate_backing", lambda law, schedule, provenance: (file_sha256(law), file_sha256(schedule)))
    monkeypatch.setattr(evidence_reduce, "run_evidence_analysis", forbidden)
    monkeypatch.setattr(direct_reduce, "_paths", lambda _base, record: individual[record.original_index])
    monkeypatch.setattr(reduce, "_worker_sources", lambda _sources, _records: _sources)
    monkeypatch.setattr(reduce, "_resolve_theory_devices", lambda _device: ("cuda:0", "cuda:1"))
    scheduled, bars = [], []

    class Progress:
        def __init__(self, **kwargs):
            bars.append(kwargs)
            self.queue = None
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def report(self, *args):
            pass

    class Executor:
        def __init__(self, **kwargs):
            assert kwargs["max_workers"] == 2
            assert kwargs["mp_context"].get_start_method() == "spawn"
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def submit(self, function, **kwargs):
            future = Future()
            try:
                future.set_result(function(**kwargs))
            except Exception as error:
                future.set_exception(error)
            return future

    def worker(**kwargs):
        assert len(kwargs["records"]) == 1
        output = []
        for task in kwargs["tasks"]:
            scheduled.append(task)
            frame = pd.DataFrame([row(task["record_index"])])[stage.KEYS].assign(
                fixed_cache_gain_sign="negative", condition_sign_status="negative",
                condition_value_status="unavailable_variation_not_computed",
                numerical_variation_l2=float("nan"), numerical_publication_blocker=False)
            stage._save(task["path"], task["identity"], {"matched_numerical": frame})
            output.append({"task_hash": task["task_hash"], "status": "reduced", "device": kwargs["device"]})
        return output

    monkeypatch.setattr(stage, "RecordProgress", Progress)
    monkeypatch.setattr(stage, "ProcessPoolExecutor", Executor)
    monkeypatch.setattr(stage, "_worker", worker)
    first = stage.run_precision_analysis(tmp_path, config=config(), refine_only=True)
    assert len(scheduled) == 2 and len(bars) == 1
    assert bars[0]["devices"] == 2
    assert first["tables"]["matched_updates"].numerical_variation_l2.isna().all()
    assert first["tables"]["matched_updates"].condition_sign_status.eq("negative").all()
    assert first["tables"]["matched_updates"].direct_prop5_log_probability_gain.eq(-.125).all()
    assert first["files"]["matched_updates"] == evidence["files"]["matched_updates"]
    repeated = stage.run_precision_analysis(tmp_path, config=config(), refine_only=True)
    assert len(scheduled) == 2 and repeated["directory"] == first["directory"]
    changed = stage.run_precision_analysis(tmp_path, config=config(numerical_max_decimal_products=0), refine_only=True)
    assert len(scheduled) == 4 and changed["directory"] != first["directory"]
    assert [task["input_path"] for task in scheduled[:2]] == [task["input_path"] for task in scheduled[2:]]
    assert [task["path"] for task in scheduled[:2]] != [task["path"] for task in scheduled[2:]]
    assert {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in protected} == protected
    assert not list((tmp_path / "theory_measurements" / "numerical_refinement").rglob("*.pt"))


def test_root_refinement_calls_only_theory_and_forwards_policy(tmp_path):
    from tests.test_theory_plot_cli import _run_all_stub, _option
    result, calls = _run_all_stub(tmp_path, ["--model", "sdv1", "--scheduler", "ddim", "--refine-numerics",
        "--numerical-decimal-precision", "96", "--numerical-max-decimal-products", "0",
        "--numerical-max-variation-nodes", "2", "--numerical-variation-absolute-width", ".001"])
    assert result.returncode == 0, result.stderr
    assert len(calls) == 1 and calls[0][0] == "theory_validation.sh"
    assert "--refine-numerics" in calls[0]
    assert _option(calls[0], "--numerical-max-decimal-products") == "0"
    assert _option(calls[0], "--numerical-max-variation-nodes") == "2"


@pytest.mark.parametrize("conflict", ["--plot", "--recompute-experiments", "--download", "--overwrite"])
def test_root_refinement_rejects_upstream_or_plot_mode_conflicts(tmp_path, conflict):
    from tests.test_theory_plot_cli import _run_all_stub
    result, calls = _run_all_stub(tmp_path, ["--refine-numerics", conflict])
    assert result.returncode != 0 and not calls


def test_cli_refinement_inherits_observations_and_accepts_new_policy(tmp_path, monkeypatch):
    from scripts import theory_validation
    from utils.experiments.theory import paper_reduce
    previous = config(num_loss_seeds=23, numerical_decimal_precision=64)
    output = PaperPaths.build(tmp_path, **previous).output_directory
    atomic_write_json(output / "run_config.json", {"scientific_config": previous})
    calls = []
    monkeypatch.setattr(theory_validation, "PROJECT_ROOT", tmp_path)
    def run(_root, **options):
        calls.append(options)
        return output
    monkeypatch.setattr(paper_reduce, "run_paper", run)
    assert theory_validation.main(["--refine-numerics", "--numerical-decimal-precision", "96"]) == 0
    assert len(calls) == 1
    assert calls[0]["refine_numerics"] and calls[0]["recompute"]
    assert calls[0]["num_loss_seeds"] == 23 and calls[0]["numerical_decimal_precision"] == 96


def test_backing_schedule_and_law_are_pinned_to_original_evidence(tmp_path):
    import torch
    from utils.common.io import atomic_torch_save, canonical_hash
    from utils.experiments.theory.support import _tensor_hash
    schedule = tmp_path / "schedule.pt"
    schedule.write_bytes(b"fixed original schedule")
    atoms = torch.tensor([[0., 1.], [2., 3.]], dtype=torch.float64)
    weights = torch.tensor([.5, .5], dtype=torch.float64)
    metadata = {"tensor_sha256": _tensor_hash(atoms), "weights": [.5, .5],
                "atom_ids": ["a", "b"], "aliases": {"a": 0, "b": 1}}
    payload = {"atoms": atoms, "weights": weights, "metadata": metadata, "law_hash": canonical_hash(metadata)}
    law = tmp_path / "law.pt"
    atomic_torch_save(payload, law)
    provenance = {"reference_law_hash": payload["law_hash"], "reference_law": metadata,
                  "evidence_recipe": {"supplement_recipe": {"schedule_sha256": file_sha256(schedule)}}}
    stage._validate_backing(law, schedule, provenance)
    schedule.write_bytes(b"changed backing schedule")
    with pytest.raises(TheoryError, match="pinned evidence"):
        stage._validate_backing(law, schedule, provenance)
    schedule.write_bytes(b"fixed original schedule")
    atomic_torch_save(payload | {"atoms": atoms + 1}, law)
    with pytest.raises(TheoryError, match="pinned atom/mass"):
        stage._validate_backing(law, schedule, provenance)


@pytest.mark.parametrize("options", [["--reference-law", "manifest"], ["--reference-manifest", "/saved/reference.json"]])
def test_root_refinement_allows_saved_reference_option_inheritance(tmp_path, options):
    from tests.test_theory_plot_cli import _run_all_stub
    result, calls = _run_all_stub(tmp_path, ["--model", "sdv1", "--scheduler", "ddim", "--refine-numerics", *options])
    assert result.returncode == 0, result.stderr
    assert len(calls) == 1 and calls[0][0] == "theory_validation.sh"


def test_worker_reassembles_completed_batches_without_raw_or_payload_deserialization(tmp_path, monkeypatch):
    from utils.experiments.theory import cache_reader, direct_reduce, scheduler_adapter
    source = tmp_path / "source"
    source.mkdir()
    record = SimpleNamespace(original_index="001", metadata={"seeds": [0]})
    base = pd.DataFrame([row(), row() | {"step_index": 1}]).assign(candidate_target_atom_id="atom")
    matched = source / "matched_updates.parquet"
    atomic_write_frame_parquet(base, matched)
    atomic_write_json(source / "complete.json", {"complete": True, "unchanged": True})
    schedule = tmp_path / "schedule.pt"
    schedule.write_bytes(b"checked saved schedule fixture")
    law_path = tmp_path / "law.pt"
    law_path.write_bytes(b"checked law fixture")
    identity = {"recipe": {"source_code": {}},
                "observation_marker_sha256": file_sha256(source / "complete.json"),
                "matched_sha256": file_sha256(matched)}
    input_identity = {"recipe": {"support_sources": {}}, "record": record.metadata}
    task = {"path": str(tmp_path / "numerical-record"), "record_index": "001", "task_hash": "task",
            "identity": identity, "input_identity": input_identity, "input_path": str(tmp_path / "inputs")}
    for step in range(2):
        input_path = Path(task["input_path"]) / f"step-{step:04d}-seeds-000000-000001"
        input_path.mkdir(parents=True)
        (input_path / "payload.pt").write_bytes(b"completed sufficient inputs need no deserialization")
        atomic_write_json(input_path / "complete.json", {"complete": True,
            "identity": {"record_inputs": input_identity, "step_index": step, "seeds": ["0"]},
            "sha256": file_sha256(input_path / "payload.pt")})
        batch_identity = {"record_policy": identity, "input_marker_sha256": file_sha256(input_path / "complete.json"),
                          "step_index": step, "seeds": ["0"]}
        extra = base.loc[base.step_index.eq(step), stage.KEYS].assign(numerical_saved_result=step + .25)
        stage._save(Path(task["path"]) / "batches" / input_path.name, batch_identity, {"matched_numerical": extra})
    preserved = {path: (path.read_bytes(), path.stat().st_mtime_ns)
                 for path in Path(task["input_path"]).rglob("*") if path.is_file()}
    law = SimpleNamespace(law_hash="law", support=SimpleNamespace(aliases={"atom": 0}))
    law.target_id_for = forbidden
    coeff = SimpleNamespace(affine=True, kappa=.5, alpha=.5, sigma=.5, destination_alpha=.5, destination_sigma=.5)
    monkeypatch.setattr(cache_reader, "load_record", forbidden)
    monkeypatch.setattr(direct_reduce, "_read_law", lambda *a, **k: law)
    monkeypatch.setattr(direct_reduce, "_paths", lambda *a, **k: (source, {"matched_updates": matched}))
    monkeypatch.setattr(scheduler_adapter, "SchedulerAdapter", lambda *a, **k: SimpleNamespace(coefficients=lambda step: coeff))
    monkeypatch.setattr(stage, "configure_worker_cpu_threads", lambda *a: None)
    def only_schedule(path):
        assert Path(path) == schedule, "Completed numerical payload was deserialized"
        return {}
    monkeypatch.setattr(stage, "safe_torch_load", only_schedule)
    result = stage._worker(sources=SimpleNamespace(runs={"experiment": {"scientific_config": {}}}, experiment=object()),
        tasks=[task], records=[record], direct_directory=tmp_path, law_path=law_path, law_digest=file_sha256(law_path),
        schedule_path=schedule, schedule_digest=file_sha256(schedule), config=config(num_inference_steps=2),
        device="cpu", worker_count=1, candidate_chunk_size=2, query_chunk_size=1)
    assert result[0]["status"] == "reduced"
    rebuilt = pd.read_parquet(Path(task["path"]) / "matched_numerical.parquet")
    assert rebuilt.numerical_saved_result.tolist() == [.25, 1.25]
    assert {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in preserved} == preserved


def test_screening_is_a_policy_dependency_not_a_sufficient_input_dependency():
    assert "numerical_screening.py" in stage.policy_recipe(config())["source_code"]
    assert "numerical_screening.py" not in stage.input_recipe()["support_sources"]
    assert stage.policy_recipe(config())["policy"]["version"] == "fixed-cache-numerics-2"
