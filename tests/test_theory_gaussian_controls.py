"""Unexecuted specifications for shared genuine Gaussian control backfill."""
from types import SimpleNamespace
import math

import pandas as pd
import pytest
import torch

from utils.common.io import file_sha256
from utils.experiments.theory import gaussian_controls as controls


def base_rows():
    pair = {"run_id": "run", "original_index": "1", "record_id": "pair", "target_id": "target"}
    return pd.DataFrame([{**pair, "seed": 0, "step_index": 0, "unconditional_target_error_rmse": 7.,
                          "gaussian_control_status": "same_saved_gaussian_input_canonical_epsilon",
                          "terminal_sscd_matched_initialization": True},
                         {**pair, "seed": 1, "step_index": 0, "unconditional_target_error_rmse": math.nan,
                          "gaussian_control_status": "unavailable_saved_initialization_not_same_gaussian_input",
                          "terminal_sscd_matched_initialization": False},
                         {**pair, "seed": 1, "step_index": 1, "unconditional_target_error_rmse": math.nan,
                          "gaussian_control_status": "unavailable", "terminal_sscd_matched_initialization": False}])


def supplement(base):
    result = base.loc[base.step_index.eq(0), controls.KEYS].copy()
    result["unconditional_target_error_rmse"] = [123., 2.]
    result["unconditional_target_error_l2"] = [246., 4.]
    result["gaussian_control_status"] = "measured_independent_gaussian_control"
    result["gaussian_control_input_source"] = "true_gaussian_initialization_probe"
    result["gaussian_control_input_sha256"] = ["input0", "input1"]
    result["gaussian_control_task_hash"] = ["task0", "task1"]
    result["gaussian_control_prediction_source"] = "fresh_empty_prompt_canonical_epsilon_once"
    result["gaussian_control_inference_dtype"] = "float32"
    result["gaussian_control_conversion_dtype"] = "float64"
    return result


def test_missing_only_overlay_preserves_verified_controls_and_actual_trajectory_identity():
    original = base_rows()
    result = controls.overlay_gaussian_controls(original, supplement(original))
    assert controls.missing_initial_controls(original).tolist() == [False, True, False]
    assert result.loc[0, "unconditional_target_error_rmse"] == 7.
    assert result.loc[0, "gaussian_control_status"] == "same_saved_gaussian_input_canonical_epsilon"
    assert result.loc[1, "unconditional_target_error_rmse"] == 2.
    assert result.loc[1, "gaussian_control_status"] == "measured_independent_gaussian_control"
    assert math.isnan(result.loc[2, "unconditional_target_error_rmse"])
    assert result.terminal_sscd_matched_initialization.tolist() == [True, False, False]
    assert not controls.missing_initial_controls(result).any()
    assert math.isnan(original.loc[1, "unconditional_target_error_rmse"])


def test_overlay_rejects_missing_or_duplicate_seed_pair_observations():
    base = base_rows()
    extra = supplement(base)
    with pytest.raises(Exception, match="absent or incomplete"):
        controls.overlay_gaussian_controls(base, extra.iloc[:1])
    with pytest.raises(Exception, match="unique"):
        controls.overlay_gaussian_controls(base, pd.concat([extra, extra.iloc[:1]], ignore_index=True))


def test_existing_controls_do_not_touch_sources_or_models(monkeypatch):
    from utils.experiments.theory import cache_reader
    frame = controls.overlay_gaussian_controls(base_rows(), supplement(base_rows()))
    result = {"tables": {"gaussian_conditional": frame}}
    def forbidden(*args, **kwargs):
        raise AssertionError("Verified controls triggered new work")
    monkeypatch.setattr(cache_reader, "discover_sources", forbidden)
    monkeypatch.setattr(controls, "_load_replica", forbidden)
    assert controls.run_gaussian_control_analysis("/unused", result=result, config={}) is result


def probe_fixture(tmp_path, monkeypatch):
    science = {"model_id": "model", "model_revision": "fixed", "native_prediction_type": "epsilon",
               "inference_dtype": "float32", "stored_prediction_type": "epsilon", "seeds": [0, 1, 2, 3],
               "latent_shape": [1, 2, 2], "scientific_tensor_storage": {"dtype": "float16"}}
    schedule = {"timesteps": torch.tensor([9, 5]), "alphas_cumprod_t": torch.tensor([.36, .64], dtype=torch.float64),
                "init_noise_sigma": 1.}
    schedule_path = tmp_path / "schedule.pt"
    schedule_path.write_bytes(b"saved schedule")
    records = []
    for index in range(3):
        path = tmp_path / ("target-" + str(index))
        path.write_bytes(("target " + str(index)).encode())
        records.append(SimpleNamespace(original_index=str(index), metadata={
            "record_id": "pair" + str(index), "target_image_sha256": "image" + str(index),
            "tensor_file_sha256": {"target_latent": file_sha256(path)}, "seeds": science["seeds"]}))
    sources = SimpleNamespace(experiment=SimpleNamespace(schedule=schedule_path,
        target_latent_path=lambda index: tmp_path / ("target-" + str(index))),
        runs={"experiment": {"scientific_config": science, "scientific_config_hash": "run"}})
    monkeypatch.setattr(controls, "_source_code", lambda: {"source": "hash"})
    monkeypatch.setattr(controls, "package_version_metadata", lambda: dict(torch="v", diffusers="v", transformers="v"))
    return sources, records, schedule


def test_gaussian_control_tasks_are_unique_seed_tasks_not_one_inference_per_pair(tmp_path, monkeypatch):
    sources, records, schedule = probe_fixture(tmp_path, monkeypatch)
    planned = controls.plan_control_tasks(sources, records, schedule)
    reversed_records = controls.plan_control_tasks(sources, records[::-1], schedule)
    assert len(planned) == 4
    assert [task["task_hash"] for task in planned] == [task["task_hash"] for task in reversed_records]
    assert all(task["count"] == 3 for task in planned)
    assert [task["seed"] for task in planned] == [0, 1, 2, 3]
    assert all("device" not in task["identity"] and "batch_size" not in task["identity"] for task in planned)


@pytest.mark.parametrize("device", [pytest.param("cuda:0", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for GPU Gaussian-control reductions"))])
def test_one_fresh_prediction_per_seed_is_shared_across_every_target(tmp_path, monkeypatch, device):
    from utils.experiments.theory import evidence_reference
    sources, records, schedule = probe_fixture(tmp_path, monkeypatch)
    tasks = controls.plan_control_tasks(sources, records, schedule)
    bank = torch.stack([torch.full((1, 2, 2), seed + 1., dtype=torch.float64) for seed in range(4)])
    monkeypatch.setattr(evidence_reference, "gaussian_bank", lambda *args: bank)
    monkeypatch.setattr(controls, "configure_worker_cpu_threads", lambda count: None)
    monkeypatch.setattr(controls, "_completed_task", lambda *args: None)
    monkeypatch.setattr(controls, "report_worker_record", lambda *args: None)
    def load(path):
        return schedule if path == sources.experiment.schedule else torch.full((1, 2, 2), float(path.name.split("-")[1]))
    monkeypatch.setattr(controls, "safe_torch_load", load)
    calls = {"replicas": 0, "prompts": [], "inputs": []}
    def replica(*args):
        calls["replicas"] += 1
        return SimpleNamespace(tokenizer=object(), text_encoder=object(), inference_dtype=torch.float32,
                               device_metadata={}, package_versions={}), object()
    monkeypatch.setattr(controls, "_load_replica", replica)
    def prompt(value, *args):
        calls["prompts"].append(value)
        return object()
    monkeypatch.setattr(controls, "encode_prompt_condition", prompt)
    def predictions(samples, *, batch_size, **kwargs):
        assert str(samples.device) == device
        for start in range(0, len(samples), batch_size):
            stop = min(start + batch_size, len(samples))
            calls["inputs"].extend(samples[start:stop, 0, 0, 0].tolist())
            yield start, stop, torch.zeros_like(samples[start:stop]), {
                "prediction_conversion_dtype": "float64", "input_quantization_l2": torch.zeros(stop-start),
                "conversion_input_quantization_l2": torch.zeros(stop-start), "oom_retries": 0}
    monkeypatch.setattr(controls, "prediction_batches", predictions)
    output = tmp_path / "controls"
    result = controls._worker(sources=sources, records=records, tasks=tasks, output_directory=output,
                              device=device, batch_size=2, worker_count=1)
    assert result["failures"] == []
    assert len(result["completed"]) == 4
    assert result["network_observations"] == 4
    assert calls == {"replicas": 1, "prompts": [""], "inputs": [1., 2., 3., 4.]}
    for task in tasks:
        data, marker = controls._task_paths(output, task)
        rows = pd.read_parquet(data)
        assert len(rows) == 3 and set(rows.seed) == {task["seed"]}
        assert rows.gaussian_control_status.eq("measured_independent_gaussian_control").all()
        for row in rows.itertuples():
            expected = abs((task["seed"] + 1.) / .6 - int(row.original_index))
            assert row.unconditional_target_error_rmse == pytest.approx(expected)
            assert row.unconditional_target_error_l2 == pytest.approx(2 * expected)


def test_cache_only_missing_controls_never_load_a_model(tmp_path, monkeypatch):
    from utils.experiments.theory import cache_reader
    sources = SimpleNamespace(selected=[SimpleNamespace(original_index="1")])
    monkeypatch.setattr(cache_reader, "discover_sources", lambda *args, **kwargs: sources)
    monkeypatch.setattr(cache_reader, "load_schedule", lambda *args: {})
    monkeypatch.setattr(controls, "plan_control_tasks", lambda *args: [{"task_hash": "missing", "seed": 1}])
    monkeypatch.setattr(controls, "_completed_task", lambda *args: None)
    monkeypatch.setattr(controls.PaperPaths, "build", lambda *args, **kwargs: SimpleNamespace(output_directory=tmp_path / "theory" / "paper"))
    def forbidden(*args, **kwargs):
        raise AssertionError("Cache-only control request tried to infer")
    monkeypatch.setattr(controls, "_load_replica", forbidden)
    monkeypatch.setattr(controls, "_worker", forbidden)
    config = {"model_name": "sdv1", "scheduler_name": "ddim", "guidance_scale": 7.5,
              "num_inference_steps": 50, "num_seeds": 20}
    with pytest.raises(Exception, match="cache-only.*recompute-experiments"):
        controls.run_gaussian_control_analysis(tmp_path, result={"tables": {"gaussian_conditional": base_rows()}},
                                               config=config, allow_compute=False)


def test_missing_only_overlay_requires_the_same_gaussian_conditional_input():
    base = base_rows()
    base["input_sha256"] = ["input0", "different-input", "unused-next-step"]
    with pytest.raises(Exception, match="input differs"):
        controls.overlay_gaussian_controls(base, supplement(base))
    base.loc[1, "input_sha256"] = "input1"
    assert controls.overlay_gaussian_controls(base, supplement(base)).loc[1, "unconditional_target_error_rmse"] == 2.


def test_partial_missing_backfill_does_not_schedule_verified_seeds(tmp_path, monkeypatch):
    from utils.experiments.theory import cache_reader
    sources = SimpleNamespace(selected=[SimpleNamespace(original_index="1")])
    monkeypatch.setattr(cache_reader, "discover_sources", lambda *args, **kwargs: sources)
    monkeypatch.setattr(cache_reader, "load_schedule", lambda *args: {})
    monkeypatch.setattr(controls, "plan_control_tasks", lambda *args: [
        {"task_hash": "already-verified", "seed": 0}, {"task_hash": "missing", "seed": 1}])
    checked = []
    def completed(directory, task):
        checked.append(task["seed"])
        return None
    monkeypatch.setattr(controls, "_completed_task", completed)
    monkeypatch.setattr(controls.PaperPaths, "build", lambda *args, **kwargs: SimpleNamespace(output_directory=tmp_path / "theory" / "paper"))
    def forbidden(*args, **kwargs):
        raise AssertionError("Cache-only request attempted network inference")
    monkeypatch.setattr(controls, "_worker", forbidden)
    config = {"model_name": "sdv1", "scheduler_name": "ddim", "guidance_scale": 7.5,
              "num_inference_steps": 50, "num_seeds": 20}
    with pytest.raises(Exception, match="cache-only.*recompute-experiments"):
        controls.run_gaussian_control_analysis(tmp_path, result={"tables": {"gaussian_conditional": base_rows()}},
                                               config=config, allow_compute=False)
    assert checked == [1]
