"""Optional learned-counterfactual specifications; authored without execution."""
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from utils.common.io import atomic_write_frame_parquet, atomic_write_json, canonical_hash, file_sha256
from utils.experiments.theory import counterfactual_probes as probes


def science():
    return {"model_id": "preserved/model", "model_revision": "immutable", "native_prediction_type": "epsilon",
            "inference_dtype": "float32", "stored_prediction_type": "epsilon", "seeds": [0, 1],
            "latent_shape": [1, 2, 2], "scientific_tensor_storage": {"dtype": "float16"}}


class Adapter:
    def __init__(self, *args, **kwargs):
        pass

    def coefficients(self, step):
        return SimpleNamespace(alpha=.6 if step == 0 else .8, sigma=.8 if step == 0 else .6,
            destination_alpha=.8, destination_sigma=.6, A=.5, kappa=.25, noise_std=0.,
            deterministic=True, affine=True, status="valid_affine", destination_timestep=(5, 1, -1)[step])


def task(*, step=0):
    level = {"step_index": step, "destination_step_index": step + 1, "timestep": 9,
             "destination_timestep": 5, "destination_alpha": 1., "destination_sigma": 1.,
             "deterministic_update": True, "applicable": True, "reason": "valid"}
    pair = {"run_id": "run", "original_index": "1", "record_id": "pair", "target_id": "target"}
    identity = {"table": probes.TABLE, "count": 2, "level": level, "pair": pair,
                "latent_shape": [1, 2, 2], "guidance_scale": 7.5, "source_record": {"receipt": "saved"},
                "source_code": {"source": "sha"}, "schedule_sha256": "sha", "same_seed_terminal_sscd": {"0": .9, "1": .2}}
    return {"table": probes.TABLE, "count": 2, "identity": identity, "task_hash": canonical_hash(identity),
            "level": level, "pair": pair, "record_index": "1", "seeds": [0, 1]}


def test_disabled_supplement_returns_original_without_source_reads_or_inference(monkeypatch):
    from utils.experiments.theory import cache_reader
    def forbidden(*args, **kwargs):
        raise AssertionError("Disabled supplement touched scientific inputs or model")
    monkeypatch.setattr(cache_reader, "discover_sources", forbidden)
    monkeypatch.setattr(probes, "_load_replica", forbidden)
    original = {"tables": {}, "provenance": {"unchanged": True}}
    assert probes.run_counterfactual_analysis("/unused", result=original, config={}) is original


@pytest.mark.parametrize("steps", [[], [True], [-1], [0, 0], "0,1", [0., 1]])
def test_backend_requires_a_predeclared_valid_step_list(steps):
    with pytest.raises(Exception, match="counterfactual_steps"):
        probes._fixed_steps({"counterfactual_steps": steps})


@pytest.mark.parametrize("step", [2, 3, 100])
def test_terminal_or_missing_snapshot_never_requests_a_prediction(step):
    class Forbidden:
        def coefficients(self, step):
            raise AssertionError("No terminal predictor exists")
    contract = probes._snapshot_contract(Forbidden(), {"timesteps": torch.tensor([9, 5, 1])}, step)
    assert not contract["applicable"]
    assert contract["destination_timestep"] is None
    assert "no_saved_next_branch_prediction" in contract["reason"]


def test_native_destination_must_be_the_next_actual_saved_prediction_label():
    contract = probes._snapshot_contract(Adapter(), {"timesteps": torch.tensor([9, 4, 1])}, 0)
    assert not contract["applicable"]
    assert "destination_is_not" in contract["reason"]


def test_metrics_preserve_signed_learned_improvement_and_normalize_once():
    target = torch.zeros(1, 2, 2, dtype=torch.float64)
    actual = torch.ones(2, 1, 2, 2, dtype=torch.float64)
    cf = torch.stack((actual[0] * 3, actual[1] * .5))
    epsilon = torch.zeros_like(actual)
    saved = torch.full_like(actual, 4, dtype=torch.float16)
    result = probes.counterfactual_metrics(cf, actual, epsilon, epsilon, saved, target, alpha=1., sigma=1.)
    torch.testing.assert_close(result["counterfactual_I_net"], torch.tensor([2., -.5], dtype=torch.float64))
    torch.testing.assert_close(result["counterfactual_target_error_l2"], torch.tensor([6., 1.], dtype=torch.float64))
    assert result["actual_next_target_error_rmse"].eq(1).all()
    assert result["saved_actual_next_target_error_rmse"].eq(3).all()
    assert result["saved_actual_epsilon_parity_l2"].eq(8).all()
    assert not result["saved_actual_epsilon_exact_equal"].any()
    assert not result["saved_actual_epsilon_storage_quantized_equal"].any()


def test_equal_endpoint_predictions_have_zero_improvement_without_a_posterior():
    state = torch.ones(2, 1, 2, 2, dtype=torch.float64)
    epsilon = torch.full_like(state, .25)
    result = probes.counterfactual_metrics(state, state, epsilon, epsilon, epsilon, state[0], alpha=.6, sigma=.8)
    assert result["counterfactual_I_net"].eq(0).all()
    assert result["saved_actual_epsilon_exact_equal"].all()
    with pytest.raises(Exception, match="positive-noise"):
        probes.counterfactual_metrics(state, state, epsilon, epsilon, epsilon, state[0], alpha=1., sigma=0.)


def test_both_endpoints_are_freshly_predicted_with_saved_actual_only_for_parity(monkeypatch):
    observed = []
    class Matched:
        def direct_matched_update(self, state, actual, u, c, guidance, step, **kwargs):
            return {"direct_lemma4_independent": True}, torch.full_like(state, 3, dtype=torch.float64), torch.zeros_like(state)
    def prediction(samples, *, condition, timestep, components, scheduler, batch_size):
        assert condition is condition_token and components is component_token and scheduler is scheduler_token
        assert timestep == 5
        for start in range(0, len(samples), batch_size):
            stop = min(start + batch_size, len(samples))
            observed.extend(samples[start:stop, 0, 0, 0].tolist())
            yield start, stop, torch.zeros_like(samples[start:stop]), {
                "input_quantization_l2": torch.zeros(stop - start),
                "conversion_input_quantization_l2": torch.zeros(stop - start),
                "prediction_conversion_dtype": "float64", "oom_retries": 0}
    monkeypatch.setattr(probes, "prediction_batches", prediction)
    condition_token, component_token, scheduler_token = object(), object(), object()
    z = torch.zeros(2, 3, 1, 2, 2, dtype=torch.float16)
    z[:, 1] = 1
    u, c = torch.zeros(2, 2, 1, 2, 2, dtype=torch.float16), torch.zeros(2, 2, 1, 2, 2, dtype=torch.float16)
    u[:, 1] = 4
    original_saved = u.clone()
    frame = probes._measure_task(task(), (z, u, c, torch.zeros(1, 2, 2)), Matched(), science=science(),
        components=component_token, scheduler=scheduler_token, condition=condition_token, batch_size=1)
    assert observed == [3., 1., 3., 1.]
    assert frame.seed.tolist() == [0, 1]
    assert frame.counterfactual_I_net.tolist() == [2., 2.]
    assert not frame.saved_actual_prediction_reused.any()
    assert frame.saved_actual_epsilon_parity_l2.eq(8).all()
    assert frame.prediction_source.eq(probes.POLICY["actual_prediction"]).all()
    torch.testing.assert_close(u, original_saved)


def test_task_identity_does_not_depend_on_step_order_batching_workers_or_extra_steps(tmp_path, monkeypatch):
    from utils.experiments.theory import scheduler_adapter
    monkeypatch.setattr(scheduler_adapter, "SchedulerAdapter", Adapter)
    monkeypatch.setattr(probes, "_source_code", lambda: {"counterfactual": "source"})
    monkeypatch.setattr(probes, "package_version_metadata", lambda: dict(torch="v", diffusers="v", transformers="v"))
    monkeypatch.setattr(probes, "_record_stamp", lambda sources, record: {"record": str(record.original_index)})
    schedule_path = tmp_path / "schedule.pt"
    schedule_path.write_bytes(b"immutable schedule receipt")
    records = [SimpleNamespace(original_index=index, metadata={"record_id": "pair" + index,
        "target_image_sha256": "image" + index, "tensor_file_sha256": {"target_latent": "latent" + index}, "seeds": [0, 1]})
        for index in ("2", "1")]
    sources = SimpleNamespace(experiment=SimpleNamespace(schedule=schedule_path),
        runs={"experiment": {"scientific_config": science(), "scientific_config_hash": "run"}},
        proximity=pd.DataFrame([{"original_index": record.original_index, "seed": seed, "sscd": score}
                                for record in records for seed, score in [(0, .9), (1, .2)]]))
    schedule = {"timesteps": torch.tensor([9, 5, 1])}
    common = {"guidance_scale": 7.5, "counterfactual_unconditional": True}
    first = probes.plan_counterfactual_tasks(sources, records, schedule, common | {"counterfactual_steps": [0]})
    more = probes.plan_counterfactual_tasks(sources, records[::-1], schedule,
        common | {"counterfactual_steps": [1, 0], "probe_batch_size": 1, "device": "cuda:3"})
    assert [item["task_hash"] for item in first] == [item["task_hash"] for item in more if item["level"]["step_index"] == 0]
    assert all(item["seeds"] == [0, 1] for item in more)
    assert all(item["count"] == 2 for item in more)


def test_shard_resume_requires_exact_task_and_payload_receipts(tmp_path):
    item = task()
    data, marker = probes._task_paths(tmp_path, item)
    atomic_write_frame_parquet(pd.DataFrame({"seed": [0, 1]}), data)
    atomic_write_json(marker, {"schema_version": 1, "task_hash": item["task_hash"], "identity": item["identity"],
                              "rows": 2, "complete": True, "sha256": file_sha256(data)})
    assert probes._completed_task(tmp_path, item) is not None
    data.write_bytes(data.read_bytes() + b"changed")
    assert probes._completed_task(tmp_path, item) is None


def test_cache_only_mode_fails_before_raw_loading_or_checkpoint_access(tmp_path, monkeypatch):
    from utils.experiments.theory import cache_reader
    sources = SimpleNamespace(selected=[SimpleNamespace(original_index="1")])
    monkeypatch.setattr(cache_reader, "discover_sources", lambda *args, **kwargs: sources)
    monkeypatch.setattr(cache_reader, "load_schedule", lambda *args: {})
    monkeypatch.setattr(probes, "plan_counterfactual_tasks", lambda *args: [task()])
    monkeypatch.setattr(probes, "_completed_task", lambda *args: None)
    monkeypatch.setattr(probes.PaperPaths, "build", lambda *args, **kwargs: SimpleNamespace(output_directory=tmp_path / "theory" / "paper"))
    def forbidden(*args, **kwargs):
        raise AssertionError("A cache-only request entered scientific execution")
    monkeypatch.setattr(cache_reader, "load_record", forbidden)
    monkeypatch.setattr(probes, "_load_replica", forbidden)
    monkeypatch.setattr(probes, "_worker", forbidden)
    config = {"counterfactual_unconditional": True, "counterfactual_steps": [0], "model_name": "sdv1",
              "scheduler_name": "ddim", "guidance_scale": 7.5, "num_inference_steps": 50, "num_seeds": 20}
    with pytest.raises(Exception, match="cache-only.*counterfactual-unconditional.*counterfactual-steps 0"):
        probes.run_counterfactual_analysis(tmp_path, result={}, config=config, allow_compute=False)


def test_worker_uses_one_replica_and_one_record_load_for_multiple_fixed_steps(tmp_path, monkeypatch):
    from utils.common import io
    from utils.experiments.theory import cache_reader, scheduler_adapter
    calls = {"load": 0, "replica": 0, "prompt": []}
    monkeypatch.setattr(probes, "configure_worker_cpu_threads", lambda count: None)
    monkeypatch.setattr(probes, "_source_code", lambda: {"source": "sha"})
    monkeypatch.setattr(probes, "file_sha256", lambda path: "sha")
    monkeypatch.setattr(probes, "_record_stamp", lambda *args: {"receipt": "saved"})
    monkeypatch.setattr(probes, "_completed_task", lambda *args: None)
    monkeypatch.setattr(io, "safe_torch_load", lambda *args: {"timesteps": torch.tensor([9, 5, 1])})
    monkeypatch.setattr(scheduler_adapter, "SchedulerAdapter", Adapter)
    monkeypatch.setattr(probes, "report_worker_record", lambda *args: None)
    monkeypatch.setattr(probes, "atomic_write_frame_parquet", lambda *args: None)
    monkeypatch.setattr(probes, "atomic_write_json", lambda *args: None)
    def load(*args, **kwargs):
        calls["load"] += 1
        return object()
    def replica(*args):
        calls["replica"] += 1
        return SimpleNamespace(tokenizer=object(), text_encoder=object(), inference_dtype=torch.float32,
                               device_metadata={}, package_versions={}), object()
    def prompt(value, *args):
        calls["prompt"].append(value)
        return object()
    monkeypatch.setattr(cache_reader, "load_record", load)
    monkeypatch.setattr(probes, "_load_replica", replica)
    monkeypatch.setattr(probes, "encode_prompt_condition", prompt)
    monkeypatch.setattr(probes, "_measure_task", lambda item, *args, **kwargs: pd.DataFrame({"seed": item["seeds"]}))
    sources = SimpleNamespace(runs={"experiment": {"scientific_config": science()}}, experiment=object())
    record = SimpleNamespace(original_index="1")
    result = probes._worker(sources=sources, records=[record], tasks=[task(step=0), task(step=1)],
        schedule_path=tmp_path / "schedule.pt", output_directory=tmp_path, device="cpu", batch_size=2, worker_count=1)
    assert not result["failures"]
    assert len(result["completed"]) == 2
    assert calls == {"load": 1, "replica": 1, "prompt": [""]}
