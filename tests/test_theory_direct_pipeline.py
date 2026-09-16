"""Direct orchestration/configuration regressions; authored, not executed."""
from types import SimpleNamespace

import pandas as pd
import pytest

from utils.common.io import atomic_write_json
from utils.experiments.theory.contracts import TheoryError, numerical_config
from utils.experiments.theory.paper_contracts import saved_plot_configuration, recompute_command
from tests.test_theory_plot_cli import _run_all_stub, _option

BASE = dict(model_name="sdv1", scheduler_name="ddim")


def test_saved_plot_inherits_measurement_settings_without_reading_manifest(tmp_path):
    config = numerical_config(**BASE, num_loss_seeds=128, loss_seed=42, loss_timesteps="saved",
                              reference_law="manifest", reference_manifest=tmp_path / "not-mounted.json")
    atomic_write_json(tmp_path / "run_config.json", {"scientific_config": config})
    inherited = saved_plot_configuration(tmp_path, requested=numerical_config(**BASE), explicit_keys=set())
    assert inherited == config
    assert inherited["num_loss_seeds"] == 128 and inherited["measure_unconditional_loss"]


@pytest.mark.parametrize("changed", [dict(num_loss_seeds=64), dict(loss_seed=7), dict(loss_timesteps="initial")])
def test_explicit_plot_measurement_conflict_reports_recompute(tmp_path, changed):
    saved = numerical_config(**BASE, num_loss_seeds=128, loss_seed=42, loss_timesteps="saved")
    atomic_write_json(tmp_path / "run_config.json", {"scientific_config": saved})
    with pytest.raises(TheoryError, match="Explicit plot settings conflict.*Recompute"):
        saved_plot_configuration(tmp_path, requested={**BASE, **changed}, explicit_keys=set(changed))


def test_portable_config_does_not_require_default_repository_identity(tmp_path):
    saved = numerical_config(model_name="sdv2", scheduler_name="ddim", num_seeds=7)
    atomic_write_json(tmp_path / "run_config.json", {"scientific_config": saved})
    assert saved_plot_configuration(tmp_path, requested=numerical_config(**BASE), portable=True) == saved
    with pytest.raises(TheoryError, match="model_name"):
        saved_plot_configuration(tmp_path, requested=numerical_config(**BASE), portable=True,
                                 explicit_keys={"model_name"})


@pytest.mark.parametrize("kwargs", [dict(num_loss_seeds=0), dict(loss_seed=-1), dict(loss_seed=2**64),
                                    dict(loss_timesteps="all"), dict(probe_batch_size=8),
                                    dict(reference_law="manifest"), dict(reference_manifest="x.json"),
                                    dict(num_unconditional_loss_seeds=True)])
def test_invalid_science_is_not_silently_ignored(kwargs):
    with pytest.raises(TheoryError):
        numerical_config(**BASE, **kwargs)


def test_recompute_command_preserves_all_measurement_settings():
    config = numerical_config(**BASE, num_loss_seeds=128, loss_seed=91, loss_timesteps="saved",
                              num_unconditional_loss_seeds=384, measure_unconditional_loss=False)
    command = recompute_command(config)
    assert "--num-loss-seeds 128" in command
    assert "--num-unconditional-loss-seeds 384" in command
    assert "--loss-timesteps saved" in command and "--loss-seed 91" in command
    assert "--no-unconditional-loss" in command


@pytest.mark.parametrize("mode", ["--plot", "--recompute-experiments", None])
def test_root_forwards_science_only_to_theory(tmp_path, mode):
    args = ([mode] if mode else []) + ["--model", "sdv1", "--scheduler", "ddim",
             "--num-loss-seeds", "128", "--loss-seed", "19", "--loss-timesteps", "saved",
             "--num-unconditional-loss-seeds", "256", "--probe-batch-size", "4",
             "--no-unconditional-loss"]
    result, calls = _run_all_stub(tmp_path, args)
    assert result.returncode == 0, result.stderr
    for call in calls:
        if call[0] == "theory_validation.sh":
            assert _option(call, "--num-loss-seeds") == "128"
            assert _option(call, "--loss-seed") == "19"
            assert _option(call, "--loss-timesteps") == "saved"
            assert "--no-unconditional-loss" in call
        else:
            assert "--num-loss-seeds" not in call and "--probe-batch-size" not in call
    if mode == "--recompute-experiments":
        assert [call[0] for call in calls] == ["theory_validation.sh"]


def test_root_plot_omits_unrequested_science_to_enable_inheritance(tmp_path):
    result, calls = _run_all_stub(tmp_path, ["--plot", "--model", "sdv1", "--scheduler", "ddim"])
    assert result.returncode == 0, result.stderr
    assert all("--num-loss-seeds" not in call and "--loss-timesteps" not in call for call in calls)


def _worker_fixture(tmp_path, monkeypatch, *, complete=True):
    from utils.experiments.theory import direct_reduce, direct_measurements, scheduler_adapter
    from utils.experiments import cache
    config = numerical_config(**BASE, num_seeds=2, num_inference_steps=2)
    record = SimpleNamespace(original_index="007", metadata={"record_id": "r", "seeds": [0, 1], "scientific_config_hash": "run", "target_image_sha256": "target"})
    sources = SimpleNamespace(experiment=object(), runs={"experiment": {"scientific_config": {}}})
    seen = []
    log = object()
    monkeypatch.setattr(direct_reduce, "_read_law", lambda *args: object())
    monkeypatch.setattr(direct_reduce, "safe_torch_load", lambda path: {})
    monkeypatch.setattr(scheduler_adapter, "SchedulerAdapter", lambda *a, **k: object())
    monkeypatch.setattr(cache, "validate_generation_record", lambda *a, **k: SimpleNamespace(valid=True))
    monkeypatch.setattr(direct_reduce, "load_record", lambda *a: seen.append("full_record") or log)
    monkeypatch.setattr(direct_reduce, "load_scores", lambda *a: [0.2, 0.9])

    def measure(loaded, measured_record, law, science, **options):
        assert loaded is log and measured_record is record
        assert options["integration"] is False
        assert callable(options["payload_callback"])
        assert options["terminal_sscd"] == [0.2, 0.9]
        seen.append("all_vector_statements_and_integral_payloads")
        table = pd.DataFrame([dict(run_id="run", original_index="007", record_id="r", target_id="target", seed=seed, step_index=k)
                              for k in range(2) for seed in range(2)])
        return {"initial": table[table.step_index == 0], "trajectory": table,
                "matched_updates": table.copy(), "terminal": table[table.step_index == 1],
                "audit": {"complete": complete, "failed_updates": [] if complete else [{"step_index": 0}]}}
    monkeypatch.setattr(direct_measurements, "measure_direct_record", measure)
    path = tmp_path / "schedule.pt"
    path.write_bytes(b"synthetic descriptor")
    from utils.common.io import file_sha256
    kwargs = dict(sources=sources, records=[record], bundle=tmp_path / "bundle",
                  analysis_hash="science", law_path=path, law_digest="mocked",
                  schedule_path=path, schedule_digest=file_sha256(path), config=config,
                  device="cpu", worker_count=1, candidate_chunk_size=2, query_chunk_size=2)
    return direct_reduce, kwargs, seen


def test_whole_record_is_shared_by_all_statements_and_resumed(tmp_path, monkeypatch):
    stage, kwargs, seen = _worker_fixture(tmp_path, monkeypatch)
    first = stage._run_analytical_worker(**kwargs)
    assert first[0]["status"] == "reduced"
    assert seen == ["full_record", "all_vector_statements_and_integral_payloads"]
    second = stage._run_analytical_worker(**kwargs)
    assert second[0]["status"] == "resumed"
    assert len(seen) == 2


def test_failed_vector_core_keeps_rows_and_cannot_be_resumed_as_complete(tmp_path, monkeypatch):
    stage, kwargs, seen = _worker_fixture(tmp_path, monkeypatch, complete=False)
    result = stage._run_analytical_worker(**kwargs)
    assert result[0]["status"] == "failed"
    record = kwargs["records"][0]
    _, paths = stage._paths(kwargs["bundle"], record)
    assert all(path.is_file() for path in paths.values())
    assert not stage._valid_record(kwargs["bundle"], record, "science")
    stage._run_analytical_worker(**kwargs)
    assert seen.count("full_record") == 2


def test_missing_integration_reuses_vector_core_without_raw_record_read(tmp_path, monkeypatch):
    stage, kwargs, seen = _worker_fixture(tmp_path, monkeypatch)
    first = stage._run_analytical_worker(**kwargs)
    assert first[0]["status"] == "reduced"
    # Independent integration recipe change invalidates only that stage.
    second = stage._run_analytical_worker(**{**kwargs, "integration_hash": "changed-quadrature"})
    assert second[0]["status"] == "reduced"
    assert second[0]["core_resumed"] and not second[0]["integration_resumed"]
    assert seen.count("full_record") == 1


def test_core_completion_requires_immutable_audit(tmp_path, monkeypatch):
    stage, kwargs, _ = _worker_fixture(tmp_path, monkeypatch)
    stage._run_analytical_worker(**kwargs)
    root, _ = stage._paths(kwargs["bundle"], kwargs["records"][0])
    (root / "audit.json").write_text('{"complete": true, "altered": true}')
    assert not stage._valid_record(kwargs["bundle"], kwargs["records"][0], "science")


def test_saved_payload_scalar_join_and_missing_coverage(tmp_path, monkeypatch):
    from utils.experiments.theory import direct_integration
    from utils.common.io import atomic_torch_save, safe_torch_load
    stage, kwargs, _ = _worker_fixture(tmp_path, monkeypatch)
    stage._run_analytical_worker(**kwargs)
    record = kwargs["records"][0]
    core_root, paths = stage._paths(kwargs["bundle"], record)
    tables = {name: pd.read_parquet(path) for name, path in paths.items()}
    for name in ("initial", "matched_updates", "terminal"):
        frame = tables[name]
        frame["direct_prop5_status"] = "not_applicable"
        frame.loc[frame.step_index == 0, "direct_prop5_status"] = "integration_required_not_computed"
    tables["audit"] = {"complete": True, "core_complete": True}
    relative = "payloads/initial.pt"
    atomic_torch_save({"schema_version": stage.INTEGRATION_PAYLOAD_SCHEMA_VERSION,
                       "step_index": 0, "seeds": [1, 0], "payload": {"bank_hash": "law"}}, core_root / relative)
    stage._save_record(core_root, paths, tables, record, "science", payloads=[relative])
    monkeypatch.setattr(stage, "safe_torch_load", safe_torch_load)
    monkeypatch.setattr(direct_integration, "integrate_proposition5_payload", lambda *a, **k: {
        "direct_prop5_status": "matched_finite_law_comparison",
        "direct_prop5_margin_rmse": [20., 10.],
    })
    law = SimpleNamespace(law_hash="law", support=object())
    destination = tmp_path / "joined"
    stage._integrate_record(kwargs["bundle"], destination, record, law, kwargs["config"], "integral")
    _, integrated = stage._paths(destination, record)
    rows = pd.read_parquet(integrated["matched_updates"])
    assert rows.loc[rows.step_index == 0].sort_values("seed").direct_prop5_margin_rmse.tolist() == [10., 20.]
    stage._save_record(core_root, paths, tables, record, "science", payloads=[])
    with pytest.raises(TheoryError, match="integration incomplete"):
        stage._integrate_record(kwargs["bundle"], tmp_path / "missing", record, law, kwargs["config"], "missing")
    assert not stage._valid_record(tmp_path / "missing", record, "missing")


@pytest.mark.parametrize("query_chunk,clean", [(1, True), (2, False)])
def test_real_vector_worker_persists_and_integrates_pending_diagnostics(tmp_path, monkeypatch, query_chunk, clean):
    """Exercise the real measurement -> disk -> integration boundary on CPU."""
    import torch
    from utils.common.io import atomic_torch_save, file_sha256, read_json, safe_torch_load
    from utils.experiments import cache
    from utils.experiments.theory import direct_reduce as stage, direct_measurements, scheduler_adapter
    from tests.test_theory_direct_math import adapter, law

    reference = law(((2.0, 1.0), (3.0, 0.0)))
    reference.support.query_chunk = query_chunk
    scheduler = adapter(clean=clean)
    target = reference.support.atoms[0]
    z = torch.zeros(3, 3, 2, dtype=torch.float64)
    z[:, 0] = torch.tensor([[0.2, 0.3], [0.5, 0.1], [0.7, -0.2]], dtype=torch.float64)
    u = torch.zeros(3, 2, 2, dtype=torch.float64)
    c = torch.full_like(u, 0.1)
    for step in range(2):
        coeff = scheduler.coefficients(step)
        _, _, _, mg = direct_measurements.clean_estimates(
            z[:, step], u[:, step], c[:, step], coeff.alpha, coeff.sigma, 2.0, latent_ndim=1,
        )
        z[:, step + 1] = coeff.A * z[:, step] + coeff.kappa * mg
    metadata = {"seeds": [4, 9, 13], "scientific_config_hash": "run", "original_index": "007",
                "record_id": "record", "target_image_sha256": "image-sha",
                "tensor_file_sha256": {"target_latent": "latent-file-sha"},
                "stored_prediction_type": "epsilon"}
    record = SimpleNamespace(original_index="007", metadata=metadata)
    sources = SimpleNamespace(experiment=object(), runs={"experiment": {"scientific_config": {}}})
    monkeypatch.setattr(stage, "_read_law", lambda *a: reference)
    monkeypatch.setattr(scheduler_adapter, "SchedulerAdapter", lambda *a, **k: scheduler)
    monkeypatch.setattr(cache, "validate_generation_record", lambda *a, **k: SimpleNamespace(valid=True))
    reads = []
    monkeypatch.setattr(stage, "load_record", lambda *a: reads.append("raw") or (z, u, c, target))
    monkeypatch.setattr(stage, "load_scores", lambda *a: [0.2, 0.9, 0.75])
    schedule_path = tmp_path / "schedule.pt"
    atomic_torch_save({}, schedule_path)
    config = numerical_config(**BASE, num_seeds=3, num_inference_steps=2, guidance_scale=2.0)
    kwargs = dict(sources=sources, records=[record], bundle=tmp_path / "core",
                  analysis_hash="core-science", law_path=schedule_path, law_digest="mocked",
                  schedule_path=schedule_path, schedule_digest=file_sha256(schedule_path), config=config,
                  device="cpu", worker_count=1, candidate_chunk_size=2, query_chunk_size=query_chunk,
                  integration_directory=tmp_path / "integrated", integration_hash="quadrature")
    receipt = stage._run_analytical_worker(**kwargs)[0]
    assert receipt["status"] == "reduced", receipt.get("traceback", receipt)
    core_root, core_paths = stage._paths(kwargs["bundle"], record)
    core_rows = pd.read_parquet(core_paths["matched_updates"])
    pending = core_rows.direct_prop5_status.eq("integration_required_not_computed")
    assert pending.any() and core_rows.loc[pending, "direct_prop5_variation_l2"].isna().all()
    marker = read_json(core_root / "complete.json")
    assert marker["payloads"]
    for name in marker["payloads"]:
        saved = safe_torch_load(core_root / name)
        assert "base_metrics" not in saved
        assert saved["schema_version"] == 2
        for value in saved["payload"].values():
            if isinstance(value, torch.Tensor):
                assert value.device.type == "cpu" and value.is_contiguous()
                assert torch.isfinite(value).all()
    _, integrated_paths = stage._paths(kwargs["integration_directory"], record)
    joined = pd.read_parquet(integrated_paths["matched_updates"])
    assert len(joined) == 6
    assert not joined.direct_prop5_status.isin(["integration_required_not_computed", "failed_integration"]).any()
    assert joined.loc[joined.step_index == 0, "direct_prop5_variation_l2"].notna().all()
    assert pd.read_parquet(integrated_paths["terminal"]).terminal_sscd.tolist() == [0.2, 0.9, 0.75]
    # A new integration recipe must reuse the saved core and keep raw reads at one.
    retry = stage._run_analytical_worker(**(kwargs | {"integration_hash": "changed-quadrature"}))[0]
    assert retry["status"] == "reduced", retry
    assert retry["core_resumed"] and not retry["integration_resumed"]
    assert reads == ["raw"]
