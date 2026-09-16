"""Unexecuted evidence cache, source isolation, and CLI contracts."""
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from utils.common.io import file_sha256
from utils.experiments.theory.contracts import TheoryError, numerical_config
from utils.experiments.theory import evidence_reduce as stage
from utils.experiments.theory.paper_contracts import recompute_command, saved_plot_configuration


def rows():
    return pd.DataFrame([dict(run_id="run", original_index="007", record_id="r", target_id="t", seed=seed,
                              step_index=0, value=float(seed)) for seed in (4, 9)])


def test_additive_join_aligns_seeds_and_rejects_missing_or_relabelled_observations():
    base = rows()
    extra = base[stage.KEYS].iloc[::-1].copy()
    extra["new_scalar"] = [90., 40.]
    result = stage._join(base, extra)
    assert result.new_scalar.tolist() == [40., 90.]
    assert result.value.tolist() == [4., 9.]
    with pytest.raises(TheoryError, match="every retained"):
        stage._join(base, extra.iloc[:1])
    with pytest.raises(TheoryError, match="overwrite"):
        stage._join(base, extra.assign(value=0.))
    with pytest.raises(TheoryError, match="duplicate"):
        stage._join(base, pd.concat([extra, extra.iloc[:1]]))


def test_supplement_cache_requires_complete_identity_and_preserves_old_sources(tmp_path):
    protected = tmp_path / "old-core.parquet"
    protected.write_bytes(b"unchanged completed vector core")
    before = (file_sha256(protected), protected.stat().st_mtime_ns)
    root = tmp_path / "supplement"
    identity = {"law_hash": "law", "formula": "fixed-evidence", "terminal_alpha": .05}
    stage._save(root, identity, {"initial": rows()}, {"complete": True})
    assert stage._complete(root, identity)
    assert not stage._complete(root, identity | {"terminal_alpha": .1})
    (root / "initial.parquet").write_bytes(b"interrupted payload")
    assert not stage._complete(root, identity)
    assert before == (file_sha256(protected), protected.stat().st_mtime_ns)


def test_gaussian_control_never_substitutes_an_unmatched_generated_state():
    from utils.models.sampling import make_initial_noise
    science = {"seeds": [4, 9], "latent_shape": [1, 2, 2]}
    bank = make_initial_noise(science["seeds"], science["latent_shape"])
    states = torch.stack([bank, bank], dim=1)
    noise = torch.zeros(2, 1, 1, 2, 2)
    target = torch.ones(1, 2, 2)
    record = SimpleNamespace(metadata={"seeds": [4, 9]})
    schedule = {"init_noise_sigma": 1.0, "alphas_cumprod_t": torch.tensor([.25])}
    matched = stage._gaussian_control((states, noise, noise, target), record, science, schedule, rows())
    expected = (2 * bank.double() - target).flatten(1).norm(dim=1) / 2
    assert matched.unconditional_target_error_rmse.tolist() == expected.tolist()
    altered = states.clone()
    altered[:, 0] += 0.01
    missing = stage._gaussian_control((altered, noise, noise, target), record, science, schedule, rows())
    assert missing.unconditional_target_error_rmse.isna().all()
    assert missing.gaussian_control_status.str.startswith("unavailable").all()


@pytest.mark.parametrize("option,value", [("reference_snr_decades", 0), ("reference_snr_decades", 13),
                                         ("terminal_noise_run_alpha", 0), ("terminal_noise_run_alpha", 1),
                                         ("terminal_noise_run_alpha", True)])
def test_evidence_science_flags_are_validated(option, value):
    with pytest.raises(TheoryError):
        numerical_config(model_name="sdv1", scheduler_name="ddim", **{option: value})


def test_evidence_science_saved_plot_inheritance_and_recompute_command(tmp_path):
    from utils.common.io import atomic_write_json
    saved = numerical_config(model_name="sdv1", scheduler_name="ddim", reference_snr_decades=5., terminal_noise_run_alpha=.02)
    atomic_write_json(tmp_path / "run_config.json", {"scientific_config": saved})
    requested = numerical_config(model_name="sdv1", scheduler_name="ddim")
    assert saved_plot_configuration(tmp_path, requested=requested, explicit_keys=set()) == saved
    with pytest.raises(TheoryError, match="conflict"):
        saved_plot_configuration(tmp_path, requested=requested, explicit_keys={"terminal_noise_run_alpha"})
    command = recompute_command(saved)
    assert "--reference-snr-decades 5.0" in command and "--terminal-noise-run-alpha 0.02" in command


def test_root_forwards_new_options_to_theory_only(tmp_path):
    from tests.test_theory_plot_cli import _run_all_stub, _option
    result, calls = _run_all_stub(tmp_path, ["--recompute-experiments", "--model", "sdv1", "--scheduler", "ddim",
                                           "--reference-snr-decades", "5", "--terminal-noise-run-alpha", "0.02"])
    assert result.returncode == 0, result.stderr
    assert len(calls) == 1 and calls[0][0] == "theory_validation.sh"
    assert _option(calls[0], "--reference-snr-decades") == "5"
    assert _option(calls[0], "--terminal-noise-run-alpha") == "0.02"


def test_wrapper_reuses_direct_histories_and_only_schedules_missing_supplements(tmp_path, monkeypatch):
    """Exercise real cache receipts/joins; every scientific worker is a stub."""
    from concurrent.futures import Future

    import numpy as np

    from utils.common.io import atomic_write_frame_parquet, atomic_write_json, read_json
    from utils.experiments.theory import cache_reader, direct_reduce, evidence_reference, reduce

    core = tmp_path / "measurement-cache" / "direct_analysis" / "fixed-core"
    integrated = core / "integrations" / "fixed-integral"
    integrated.mkdir(parents=True)
    for name in ("reference_law.pt", "schedule.pt"):
        (core / name).write_bytes(b"synthetic immutable input")
    source_record = integrated / "record"
    source_record.mkdir()
    atomic_write_json(source_record / "complete.json", {"core": "unchanged"})
    initial = rows().rename(columns={"value": "direct_scalar"})
    trajectory = pd.concat([initial, initial.assign(step_index=1)], ignore_index=True)
    terminal = initial.assign(step_index=1)
    gaussian = trajectory.assign(terminal_sscd_matched_initialization=[True, False, False, False])
    base = {"initial": initial, "trajectory": trajectory, "terminal": terminal,
            "gaussian_conditional": gaussian}
    files = {}
    for name, frame in base.items():
        path = integrated / (name + ".parquet")
        atomic_write_frame_parquet(frame, path)
        files[name] = {"path": str(path), "sha256": file_sha256(path), "rows": len(frame)}
    protected = {path: (path.read_bytes(), path.stat().st_mtime_ns)
                 for path in core.rglob("*") if path.is_file()}
    direct = {"tables": base, "files": files, "directory": integrated,
              "provenance": {"reference_law_hash": "fixed-law", "analysis_hash": "fixed-integral"}}
    record = SimpleNamespace(original_index="007", metadata={"seeds": [4, 9]})
    science = {"seeds": [4, 9], "latent_shape": [1, 2, 2],
               "scientific_tensor_storage": {"dtype": "float16"}}
    sources = SimpleNamespace(selected=[record], runs={"experiment": {
        "scientific_config": science, "scientific_config_hash": "run"}})
    direct_requests, scheduled = [], []

    def saved_direct(_root, *, config, **_execution):
        direct_requests.append(dict(config))
        return direct

    def scientific_worker_stub(**kwargs):
        assert kwargs["terminal_update_count"] == 2
        assert kwargs["run_scope"] == "run"
        receipts = []
        for task in kwargs["tasks"]:
            scheduled.append(task["kind"])
            if task["kind"] == "reference":
                frames = {"reference_analytical": pd.DataFrame({
                    "seed": [4, 9], "grid_index": task["grid_index"], "snr": task["snr"],
                    "reference_scale_rmse": task["identity"]["reference"]["reference_scale_rmse"],
                    "reference_mean_error_rmse": [0.25, 0.5]})}
            else:
                frames = {name: frame[stage.KEYS].iloc[::-1].assign(evidence_added=[90., 40.] if name != "trajectory" else [90., 40., 90., 40.])
                          for name, frame in base.items() if name != "gaussian_conditional"}
                frames["gaussian_control"] = initial[stage.KEYS].iloc[::-1].assign(
                    unconditional_target_error_rmse=[9., 4.],
                    gaussian_control_status="same_saved_gaussian_input_canonical_epsilon")
            stage._save(task["path"], task["identity"], frames, {"complete": True})
            receipts.append({"task_hash": task["task_hash"], "status": "computed", "device": "cuda:0"})
        return receipts

    class InlineExecutor:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def submit(self, worker, **kwargs):
            assert worker is scientific_worker_stub
            future = Future()
            future.set_result(worker(**kwargs))
            return future

    class SilentProgress:
        queue = None

        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def report(self, *_args):
            pass

    def forbidden_executor(**_kwargs):
        raise AssertionError("A completed supplement must not launch measurement workers")

    monkeypatch.setattr(direct_reduce, "run_direct_analysis", saved_direct)
    monkeypatch.setattr(direct_reduce, "_paths", lambda *_args: (source_record, None))
    monkeypatch.setattr(direct_reduce, "_read_law", lambda *_args: SimpleNamespace(law_hash="fixed-law"))
    monkeypatch.setattr(cache_reader, "discover_sources", lambda *_args, **_kwargs: sources)
    monkeypatch.setattr(stage, "safe_torch_load", lambda _path: {
        "alphas_cumprod_t": [.25], "init_noise_sigma": 1.0})
    monkeypatch.setattr(stage, "package_version", lambda _package: "fixture-version")
    monkeypatch.setattr(stage, "_sources", lambda *names: {name: "fixture-source-sha" for name in names})
    monkeypatch.setattr(evidence_reference, "reference_grid", lambda snr, decades: np.asarray([snr / 10 ** decades, snr]))
    monkeypatch.setattr(evidence_reference, "reference_atoms", lambda _law: pd.DataFrame({
        "atom_id": ["t"], "weight": [1.0], "reference_scale_rmse": [.75]}))
    monkeypatch.setattr(reduce, "_resolve_theory_devices", lambda _device: ("cuda:0",))
    monkeypatch.setattr(reduce, "_worker_sources", lambda source, _chosen: source)
    monkeypatch.setattr(stage, "ProcessPoolExecutor", InlineExecutor)
    monkeypatch.setattr(stage, "RecordProgress", SilentProgress)
    monkeypatch.setattr(stage, "_worker", scientific_worker_stub)
    config = {"reference_snr_decades": 6., "terminal_noise_run_alpha": .05, "guidance_scale": 7.5, "mean_source": "cached-targets"}
    first = stage.run_evidence_analysis(tmp_path, config=config, device="cuda:0")
    assert scheduled == ["reference", "reference", "record"]
    assert first["provenance"]["direct_analysis_hash"] == "fixed-integral"
    manifest = read_json(first["directory"] / "manifest.json")
    assert manifest["logical_joins"] == first["logical_joins"]
    for name, item in files.items():
        assert first["files"][name] == item
        assert not (first["directory"] / (name + ".parquet")).exists()
        pd.testing.assert_frame_equal(base[name], pd.read_parquet(item["path"]))
    for name in ("initial", "trajectory", "terminal"):
        assert first["logical_joins"][name] == {
            "base": name, "supplement": name + "_evidence", "keys": stage.KEYS,
            "how": "exact_one_to_one_additive", "preserves_every_base_row": True}
        saved_extra = pd.read_parquet(first["files"][name + "_evidence"]["path"])
        assert set(saved_extra) == {*stage.KEYS, "evidence_added"}
        pd.testing.assert_frame_equal(first["tables"][name], stage._join(base[name], saved_extra))
        assert first["tables"][name].direct_scalar.tolist() == base[name].direct_scalar.tolist()
    assert first["tables"]["initial"].evidence_added.tolist() == [40., 90.]
    control = first["tables"]["gaussian_conditional"]
    assert control.unconditional_target_error_rmse.iloc[0] == 4.
    assert control.unconditional_target_error_rmse.iloc[1:].isna().all()
    assert first["logical_joins"]["gaussian_conditional"]["how"] == "left_one_to_one"

    scheduled.clear()
    monkeypatch.setattr(stage, "ProcessPoolExecutor", forbidden_executor)
    resumed = stage.run_evidence_analysis(tmp_path, config=config, device="cuda:0")
    assert not scheduled
    assert resumed["directory"] == first["directory"]
    assert resumed["files"] == first["files"]
    monkeypatch.setattr(stage, "ProcessPoolExecutor", InlineExecutor)
    changed_grid = stage.run_evidence_analysis(tmp_path, config=config | {"reference_snr_decades": 5.}, device="cuda:0")
    assert scheduled == ["reference", "reference"]
    assert changed_grid["directory"] != first["directory"]
    scheduled.clear()
    changed_noise = stage.run_evidence_analysis(tmp_path, config=config | {"terminal_noise_run_alpha": .02}, device="cuda:0")
    assert scheduled == ["record"]
    assert changed_noise["directory"] != first["directory"]
    for result in (resumed, changed_grid, changed_noise):
        assert result["provenance"]["direct_analysis_hash"] == "fixed-integral"
        assert all(result["files"][name] == item for name, item in files.items())
    assert len(direct_requests) == 4
    assert all(request["guidance_scale"] == 7.5 for request in direct_requests)
    for path, original in protected.items():
        assert (path.read_bytes(), path.stat().st_mtime_ns) == original


@pytest.mark.parametrize("files", [[], ["payload.parquet"], {"payload.parquet": None},
                                    {"payload.parquet": {}}, {"../outside.parquet": {"sha256": "x"}}])
def test_malformed_or_escaping_completion_marker_is_incomplete(tmp_path, files):
    from utils.common.io import atomic_write_json

    root = tmp_path / "supplement"
    root.mkdir()
    (root / "payload.parquet").write_bytes(b"placeholder")
    atomic_write_json(root / "complete.json", {
        "schema_version": stage.SUPPLEMENT_SCHEMA_VERSION, "complete": True,
        "identity": {"recipe": "fixture"}, "files": files})
    assert not stage._complete(root, {"recipe": "fixture"})


def test_supplement_rejects_symlink_roots_and_does_not_follow_payload_links(tmp_path):
    from utils.common.io import atomic_write_json

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "payload.parquet").write_bytes(b"immutable external scalar")
    linked_root = tmp_path / "linked"
    linked_root.symlink_to(outside, target_is_directory=True)
    with pytest.raises(TheoryError, match="symbolic-link"):
        stage._complete(linked_root, {})
    with pytest.raises(TheoryError, match="symbolic-link"):
        stage._save(linked_root, {}, {"initial": rows()})
    root = tmp_path / "supplement"
    root.mkdir()
    (root / "payload.parquet").symlink_to(outside / "payload.parquet")
    atomic_write_json(root / "complete.json", {
        "schema_version": stage.SUPPLEMENT_SCHEMA_VERSION, "complete": True,
        "identity": {}, "files": {"payload.parquet": {
            "sha256": file_sha256(outside / "payload.parquet")}}})
    assert not stage._complete(root, {})
    assert (outside / "payload.parquet").read_bytes() == b"immutable external scalar"
