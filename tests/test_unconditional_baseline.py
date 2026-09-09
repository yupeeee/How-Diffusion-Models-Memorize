from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch

from utils.common.cli import MAX_SEED
from utils.common.io import (
    atomic_torch_save,
    atomic_write_json,
    canonical_hash,
    read_json,
    safe_torch_load,
)
from utils.experiments import unconditional_baseline as baseline
from utils.experiments.cache import (
    GENERATION_SELECTION_POLICY,
    GenerationPaths,
    generation_paths,
)
from utils.models.registry import get_model_spec


LATENT_SHAPE = (1, 2, 2)
NUM_SEEDS = 2
NUM_BASELINE_SEEDS = 3
NUM_STEPS = 3
REFERENCE_SEEDS = (2, 3)
BASELINE_SEEDS = (2, 3, 4)
SCHEDULER_CLASSES = {
    "ddim": "DDIMScheduler",
    "ddpm": "DDPMScheduler",
}


class _Progress:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.total = kwargs["total"]
        self.updates: list[int] = []
        self.closed = False

    def update(self, amount: int) -> None:
        self.updates.append(amount)

    def close(self) -> None:
        self.closed = True


class _ImmediateFuture:
    def __init__(self, result: object) -> None:
        self._result = result

    def result(self) -> object:
        return self._result


class _ImmediateExecutor:
    created: list[tuple[str, int]] = []

    def __init__(
        self,
        *,
        max_workers: int,
        mp_context: object,
        initializer: Any,
        initargs: tuple[object, ...],
    ) -> None:
        assert max_workers == 1
        assert mp_context is not None
        initializer(*initargs)
        self.created.append((str(initargs[1]), int(initargs[2])))

    def __enter__(self) -> _ImmediateExecutor:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def submit(self, function: Any, *args: object) -> _ImmediateFuture:
        return _ImmediateFuture(function(*args))


def _science(scheduler_name: str = "ddim") -> dict[str, object]:
    spec = get_model_spec("sdv1")
    scheduler_config = {"num_train_timesteps": 10, "prediction_type": "epsilon"}
    return {
        "model_cli_name": "sdv1",
        "dataset_model": spec.dataset_model,
        "model_id": spec.model_id,
        "model_revision": "model-revision",
        "vae_id": spec.vae_id or spec.model_id,
        "vae_revision": "vae-revision",
        "scheduler": {
            "name": scheduler_name,
            "class": SCHEDULER_CLASSES[scheduler_name],
            "config": scheduler_config,
        },
        "native_prediction_type": "epsilon",
        "stored_prediction_type": "epsilon",
        "guidance_scale": 7.5,
        "num_inference_steps": NUM_STEPS,
        "num_seeds": NUM_SEEDS,
        "seeds": list(REFERENCE_SEEDS),
        "selection_policy": GENERATION_SELECTION_POLICY,
        "latent_shape": list(LATENT_SHAPE),
        "inference_dtype": "float32",
        "package_versions": {
            "torch": "test-torch",
            "diffusers": "test-diffusers",
            "transformers": "test-transformers",
            "cuda": None,
        },
        "trajectory_order": "noise_to_image",
        "scientific_tensor_storage": {
            "dtype": "float32",
            "device": "cpu",
            "layout": "contiguous",
        },
    }


def _schedule(scheduler_name: str = "ddim") -> dict[str, object]:
    alpha = torch.tensor([0.1, 0.25, 0.5], dtype=torch.float32)
    sigma = torch.sqrt(1.0 - alpha.square()).contiguous()
    return {
        "timesteps": torch.tensor([9, 6, 3], dtype=torch.int64),
        "alpha_t": alpha,
        "sigma_t": sigma,
        "alphas_cumprod_t": alpha.square().contiguous(),
        "init_noise_sigma": 1.0,
        "scheduler_name": scheduler_name,
        "scheduler_class": SCHEDULER_CLASSES[scheduler_name],
        "native_prediction_type": "epsilon",
        "stored_prediction_type": "epsilon",
        "trajectory_order": "noise_to_image",
        "scheduler_config": {
            "num_train_timesteps": 10,
            "prediction_type": "epsilon",
        },
    }


def _create_reference_run(root: Path, scheduler_name: str = "ddim") -> GenerationPaths:
    paths = generation_paths(
        root,
        model_name="sdv1",
        scheduler_name=scheduler_name,
        guidance_scale=7.5,
        num_inference_steps=NUM_STEPS,
        num_seeds=NUM_SEEDS,
        seed_start=NUM_SEEDS,
    )
    paths.run_directory.mkdir(parents=True)
    science = _science(scheduler_name)
    atomic_write_json(
        paths.run_config,
        {
            "scientific_config": science,
            "scientific_config_hash": canonical_hash(science),
        },
    )
    atomic_torch_save(_schedule(scheduler_name), paths.schedule)
    return paths


def _fake_estimate_value(position: int) -> float:
    return 1.0 if position < 2 else float(position + 2)


def _install_fake_cpu_inference(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        baseline, "resolve_devices", lambda _device: (torch.device("cpu"),)
    )

    def build_runtime(
        contract: baseline._SourceContract,
        *,
        device: str | torch.device,
        worker_count: int,
    ) -> SimpleNamespace:
        assert worker_count == 1
        return SimpleNamespace(
            contract=contract,
            components=SimpleNamespace(device=torch.device(device)),
        )

    def estimate(
        runtime: SimpleNamespace, entries: tuple[tuple[int, int], ...]
    ) -> baseline._SeedBatchEstimate:
        assert runtime.contract.baseline_seeds
        values = torch.stack(
            [
                torch.full(
                    LATENT_SHAPE,
                    _fake_estimate_value(position),
                    dtype=torch.float64,
                )
                for position, _seed in entries
            ]
        ).contiguous()
        return baseline._SeedBatchEstimate(tuple(entries), values)

    monkeypatch.setattr(baseline, "_build_inference_runtime", build_runtime)
    monkeypatch.setattr(baseline, "_estimate_seed_batch_adaptive", estimate)


def _compute(
    root: Path,
    *,
    scheduler_name: str = "ddim",
    num_baseline_seeds: int = NUM_BASELINE_SEEDS,
    **kwargs: object,
) -> baseline.UnconditionalBaselineArtifact:
    return baseline.compute_unconditional_baseline(
        root,
        model_name="sdv1",
        scheduler_name=scheduler_name,
        guidance_scale=7.5,
        num_inference_steps=NUM_STEPS,
        num_seeds=NUM_SEEDS,
        num_baseline_seeds=num_baseline_seeds,
        **kwargs,
    )


def _load(
    root: Path,
    *,
    scheduler_name: str = "ddim",
    num_baseline_seeds: int = NUM_BASELINE_SEEDS,
    load_tensor: bool = True,
) -> baseline.UnconditionalBaselineArtifact:
    return baseline.load_unconditional_baseline(
        root,
        model_name="sdv1",
        scheduler_name=scheduler_name,
        guidance_scale=7.5,
        num_inference_steps=NUM_STEPS,
        num_seeds=NUM_SEEDS,
        num_baseline_seeds=num_baseline_seeds,
        load_tensor=load_tensor,
    )


def _load_cli_module() -> ModuleType:
    path = Path(__file__).parents[1] / "scripts" / "unconditional_baseline.py"
    spec = importlib.util.spec_from_file_location("baseline_cli_for_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_defaults_to_one_thousand_baseline_seeds_and_auto_device() -> None:
    script = _load_cli_module()
    arguments = script.build_parser().parse_args([])
    assert arguments.scheduler == "ddim"
    assert arguments.N == 20
    assert arguments.num_baseline_seeds == 1000
    assert arguments.device == "auto"
    assert script.build_parser().parse_args(["--scheduler", "ddpm"]).scheduler == "ddpm"


def test_exact_empty_condition_formula(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _create_reference_run(tmp_path)
    contract = baseline._load_source_contract(
        tmp_path,
        model_name="sdv1",
        scheduler_name="ddim",
        guidance_scale=7.5,
        num_inference_steps=NUM_STEPS,
        num_seeds=NUM_SEEDS,
        num_baseline_seeds=NUM_BASELINE_SEEDS,
    )
    raw = (
        torch.arange(2 * int(torch.tensor(LATENT_SHAPE).prod()))
        .reshape(2, *LATENT_SHAPE)
        .float()
        .div(10.0)
        .contiguous()
    )
    empty = torch.tensor([[17.0]])
    captures: list[dict[str, object]] = []

    def fake_noise(seeds: tuple[int, ...], shape: tuple[int, int, int]) -> torch.Tensor:
        assert seeds == BASELINE_SEEDS[:2]
        assert shape == LATENT_SHAPE
        return raw.clone()

    def fake_predict(**kwargs: object) -> torch.Tensor:
        captures.append(kwargs)
        samples = kwargs["samples"]
        assert isinstance(samples, torch.Tensor)
        assert kwargs["condition"] is empty
        assert kwargs["timestep"] == 9
        assert kwargs["conversion_sample"] is samples
        return (samples * 0.25 + 1.0).contiguous()

    monkeypatch.setattr(baseline, "make_initial_noise", fake_noise)
    monkeypatch.setattr(baseline, "predict_conditional_epsilon", fake_predict)
    runtime = baseline._InferenceRuntime(
        contract=contract,
        components=SimpleNamespace(
            device=torch.device("cpu"),
            inference_dtype=torch.float32,
            unet=object(),
        ),
        scheduler=object(),
        empty_condition=empty,
    )
    result = baseline._estimate_seed_batch(runtime, ((0, 2), (1, 3)))
    epsilon = raw * 0.25 + 1.0
    expected = (raw.double() - contract.sigma_t * epsilon.double()) / contract.alpha_t
    torch.testing.assert_close(result.estimates, expected, rtol=0.0, atol=0.0)
    assert len(captures) == 1


def test_computation_weights_each_dedicated_seed_once_and_has_one_progress_bar(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = _create_reference_run(tmp_path)
    assert set(paths.run_directory.iterdir()) == {paths.run_config, paths.schedule}
    _install_fake_cpu_inference(monkeypatch)
    progresses: list[_Progress] = []

    def progress_factory(**kwargs: object) -> _Progress:
        result = _Progress(**kwargs)
        progresses.append(result)
        return result

    artifact = _compute(
        tmp_path,
        device="auto",
        progress_factory=progress_factory,
    )
    expected = torch.full(LATENT_SHAPE, 2.0, dtype=torch.float64)
    assert artifact.mu_hat is not None
    torch.testing.assert_close(artifact.mu_hat, expected, rtol=0.0, atol=0.0)
    assert artifact.tensor_path == (
        paths.run_directory / "unconditional_baseline" / "S2_N3" / "mu_hat.pt"
    )
    assert artifact.metadata_path == artifact.tensor_path.with_name("metadata.json")
    assert artifact.num_baseline_seeds == NUM_BASELINE_SEEDS
    assert artifact.baseline_seed_start == NUM_SEEDS
    assert artifact.baseline_seeds == BASELINE_SEEDS
    assert artifact.metadata["baseline_seed_start"] == NUM_SEEDS
    assert artifact.metadata["baseline_seeds"] == list(BASELINE_SEEDS)
    assert artifact.metadata["num_baseline_seeds"] == NUM_BASELINE_SEEDS
    assert artifact.metadata["latent_dimension"] == 4
    assert artifact.metadata["condition"] == "empty string only"
    assert artifact.metadata["weighting"] == (
        "one unit per baseline seed in ascending seed order"
    )
    assert artifact.metadata["package_versions"] == _science()["package_versions"]
    assert len(progresses) == 1
    assert progresses[0].total == NUM_BASELINE_SEEDS
    assert progresses[0].kwargs["desc"] == baseline.PROGRESS_DESCRIPTION
    assert progresses[0].kwargs["unit"] == "seed"
    assert progresses[0].updates == [NUM_BASELINE_SEEDS]
    assert progresses[0].closed


def test_full_and_metadata_only_loaders_validate_without_tensor_work(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _create_reference_run(tmp_path)
    _install_fake_cpu_inference(monkeypatch)
    computed = _compute(tmp_path)
    loaded = _load(tmp_path)
    assert loaded.mu_hat is not None
    assert torch.equal(loaded.mu_hat, computed.mu_hat)
    assert loaded.mu_hat_sha256 == computed.mu_hat_sha256

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("metadata-only loading performed tensor/model work")

    monkeypatch.setattr(baseline, "safe_torch_load", forbidden)
    monkeypatch.setattr(baseline, "make_initial_noise", forbidden)
    monkeypatch.setattr(baseline, "load_model_components", forbidden)
    metadata_only = _load(tmp_path, load_tensor=False)
    assert metadata_only.mu_hat is None
    assert metadata_only.mu_hat_sha256 == computed.mu_hat_sha256
    assert metadata_only.baseline_seeds == BASELINE_SEEDS


def test_baseline_count_has_an_isolated_artifact_namespace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _create_reference_run(tmp_path)
    _install_fake_cpu_inference(monkeypatch)
    three = _compute(tmp_path, num_baseline_seeds=3)
    four = _compute(tmp_path, num_baseline_seeds=4)
    assert three.tensor_path.parent.name == "S2_N3"
    assert four.tensor_path.parent.name == "S2_N4"
    assert three.tensor_path != four.tensor_path
    assert three.tensor_path.is_file()
    assert four.tensor_path.is_file()
    assert _load(tmp_path, num_baseline_seeds=3).baseline_seeds == (2, 3, 4)
    assert _load(tmp_path, num_baseline_seeds=4).baseline_seeds == (2, 3, 4, 5)


def test_multi_cuda_auto_path_round_robin_shards_and_reports_B(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _create_reference_run(tmp_path)
    resolved: list[object] = []
    monkeypatch.setattr(
        baseline,
        "resolve_devices",
        lambda value: (
            resolved.append(value) or (torch.device("cuda:0"), torch.device("cuda:1"))
        ),
    )
    built: list[tuple[str, int]] = []
    batches: list[tuple[str, tuple[tuple[int, int], ...]]] = []

    def build_runtime(
        contract: baseline._SourceContract,
        *,
        device: str | torch.device,
        worker_count: int,
    ) -> SimpleNamespace:
        built.append((str(device), worker_count))
        return SimpleNamespace(
            contract=contract,
            components=SimpleNamespace(device=torch.device(device)),
        )

    def estimate(
        runtime: SimpleNamespace, entries: tuple[tuple[int, int], ...]
    ) -> baseline._SeedBatchEstimate:
        batches.append((str(runtime.components.device), tuple(entries)))
        values = torch.stack(
            [
                torch.full(LATENT_SHAPE, float(seed), dtype=torch.float64)
                for _position, seed in entries
            ]
        ).contiguous()
        return baseline._SeedBatchEstimate(tuple(entries), values)

    _ImmediateExecutor.created.clear()
    monkeypatch.setattr(baseline, "_build_inference_runtime", build_runtime)
    monkeypatch.setattr(baseline, "_estimate_seed_batch_adaptive", estimate)
    monkeypatch.setattr(baseline, "ProcessPoolExecutor", _ImmediateExecutor)
    monkeypatch.setattr(baseline, "as_completed", lambda futures: tuple(futures))
    progresses: list[_Progress] = []
    artifact = _compute(
        tmp_path,
        num_baseline_seeds=5,
        device="auto",
        progress_factory=lambda **kwargs: (
            progresses.append(_Progress(**kwargs)) or progresses[-1]
        ),
    )

    assert resolved == ["auto"]
    assert _ImmediateExecutor.created == [("cuda:0", 2), ("cuda:1", 2)]
    assert built == [("cuda:0", 2), ("cuda:1", 2)]
    assert batches == [
        ("cuda:0", ((0, 2), (2, 4), (4, 6))),
        ("cuda:1", ((1, 3), (3, 5))),
    ]
    assert artifact.mu_hat is not None
    torch.testing.assert_close(
        artifact.mu_hat,
        torch.full(LATENT_SHAPE, 4.0, dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )
    assert len(progresses) == 1
    assert progresses[0].total == 5
    assert sum(progresses[0].updates) == 5
    assert progresses[0].closed


def test_cli_reuses_valid_cache_and_overwrite_recomputes(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    script = _load_cli_module()
    load_calls: list[dict[str, object]] = []
    compute_calls: list[dict[str, object]] = []
    cached = SimpleNamespace(metadata_path=Path("cached/metadata.json"))

    def fake_load(_root: Path, **kwargs: object) -> SimpleNamespace:
        load_calls.append(dict(kwargs))
        return cached

    def fake_compute(_root: Path, **kwargs: object) -> SimpleNamespace:
        compute_calls.append(dict(kwargs))
        return SimpleNamespace(
            tensor_path=Path("fresh/mu_hat.pt"),
            metadata_path=Path("fresh/metadata.json"),
        )

    monkeypatch.setattr(script, "load_unconditional_baseline", fake_load)
    monkeypatch.setattr(script, "compute_unconditional_baseline", fake_compute)
    assert script.main(["--N", "7", "--num-baseline-seeds", "11"]) == 0
    assert len(load_calls) == 1
    assert load_calls[0]["scheduler_name"] == "ddim"
    assert load_calls[0]["num_seeds"] == 7
    assert load_calls[0]["num_baseline_seeds"] == 11
    assert not compute_calls
    assert "skipped" in capsys.readouterr().out

    load_calls.clear()
    assert (
        script.main(
            [
                "--N",
                "7",
                "--num-baseline-seeds",
                "11",
                "--scheduler",
                "ddpm",
                "--device",
                "cpu",
                "--overwrite",
            ]
        )
        == 0
    )
    assert not load_calls
    assert len(compute_calls) == 1
    assert compute_calls[0]["scheduler_name"] == "ddpm"
    assert compute_calls[0]["num_seeds"] == 7
    assert compute_calls[0]["num_baseline_seeds"] == 11
    assert compute_calls[0]["device"] == "cpu"


def test_mu_hat_file_tampering_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _create_reference_run(tmp_path)
    _install_fake_cpu_inference(monkeypatch)
    artifact = _compute(tmp_path)
    assert artifact.mu_hat is not None
    atomic_torch_save(torch.zeros_like(artifact.mu_hat), artifact.tensor_path)
    with pytest.raises(baseline.UnconditionalBaselineError, match="SHA-256"):
        _load(tmp_path, load_tensor=False)


@pytest.mark.parametrize("load_tensor", (False, True))
@pytest.mark.parametrize("mutation", ("unexpected", "missing"))
def test_metadata_requires_exact_current_schema(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    load_tensor: bool,
    mutation: str,
) -> None:
    _create_reference_run(tmp_path)
    _install_fake_cpu_inference(monkeypatch)
    artifact = _compute(tmp_path)
    metadata = read_json(artifact.metadata_path)
    if mutation == "unexpected":
        metadata["unexpected_field"] = "not part of the current schema"
    else:
        metadata.pop("created_at_utc")
    atomic_write_json(artifact.metadata_path, metadata)
    with pytest.raises(baseline.UnconditionalBaselineError, match="invalid schema"):
        _load(tmp_path, load_tensor=load_tensor)


def test_metadata_seed_tampering_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _create_reference_run(tmp_path)
    _install_fake_cpu_inference(monkeypatch)
    artifact = _compute(tmp_path)
    metadata = read_json(artifact.metadata_path)
    metadata["baseline_seeds"] = [2, 3, 99]
    atomic_write_json(artifact.metadata_path, metadata)
    with pytest.raises(baseline.UnconditionalBaselineError, match="metadata differs"):
        _load(tmp_path, load_tensor=False)


def test_stale_model_revision_and_schedule_are_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = _create_reference_run(tmp_path)
    _install_fake_cpu_inference(monkeypatch)
    _compute(tmp_path)

    run_config = read_json(paths.run_config)
    science = run_config["scientific_config"]
    assert isinstance(science, dict)
    science["model_revision"] = "different-revision"
    run_config["scientific_config_hash"] = canonical_hash(science)
    atomic_write_json(paths.run_config, run_config)
    with pytest.raises(baseline.UnconditionalBaselineError, match="metadata differs"):
        _load(tmp_path, load_tensor=False)

    paths = _create_reference_run(tmp_path / "schedule-case")
    _install_fake_cpu_inference(monkeypatch)
    _compute(tmp_path / "schedule-case")
    changed = _schedule()
    changed["diagnostic_only"] = True
    atomic_torch_save(changed, paths.schedule)
    with pytest.raises(baseline.UnconditionalBaselineError, match="metadata differs"):
        _load(tmp_path / "schedule-case", load_tensor=False)


@pytest.mark.parametrize("scheduler_name", ("ddim", "ddpm"))
def test_active_scheduler_coefficients_are_cross_checked(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, scheduler_name: str
) -> None:
    _create_reference_run(tmp_path, scheduler_name)
    contract = baseline._load_source_contract(
        tmp_path,
        model_name="sdv1",
        scheduler_name=scheduler_name,
        guidance_scale=7.5,
        num_inference_steps=NUM_STEPS,
        num_seeds=NUM_SEEDS,
        num_baseline_seeds=NUM_BASELINE_SEEDS,
    )
    assert contract.schedule_payload is not None
    active = SimpleNamespace(init_noise_sigma=1.0, timesteps=None)

    def set_timesteps(steps: int, *, device: torch.device) -> None:
        assert steps == NUM_STEPS
        active.timesteps = contract.schedule_payload["timesteps"].to(device)

    active.set_timesteps = set_timesteps

    requested_schedulers: list[str] = []

    def build_scheduler(_original: object, name: str) -> SimpleNamespace:
        requested_schedulers.append(name)
        return SimpleNamespace(
            scheduler=active,
            name=name,
            class_name=SCHEDULER_CLASSES[name],
            config=contract.scheduler_config,
        )

    monkeypatch.setattr(
        baseline,
        "build_scheduler",
        build_scheduler,
    )
    saved = tuple(
        contract.schedule_payload[name].clone()
        for name in ("alpha_t", "sigma_t", "alphas_cumprod_t")
    )
    monkeypatch.setattr(
        baseline,
        "schedule_alpha_sigma",
        lambda _scheduler, _timesteps, *, dtype: tuple(
            value.to(dtype=dtype) for value in saved
        ),
    )
    components = SimpleNamespace(
        original_scheduler=object(), device=torch.device("cpu")
    )
    assert baseline._validate_active_scheduler(components, contract) is active
    assert requested_schedulers == [scheduler_name]

    changed = (saved[0] + 0.01, saved[1], saved[2])
    monkeypatch.setattr(
        baseline,
        "schedule_alpha_sigma",
        lambda _scheduler, _timesteps, *, dtype: tuple(
            value.to(dtype=dtype) for value in changed
        ),
    )
    with pytest.raises(
        baseline.UnconditionalBaselineError, match="coefficients differ"
    ):
        baseline._validate_active_scheduler(components, contract)
    assert requested_schedulers == [scheduler_name, scheduler_name]


def test_active_ddpm_scheduler_identity_is_cross_checked(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _create_reference_run(tmp_path, "ddpm")
    contract = baseline._load_source_contract(
        tmp_path,
        model_name="sdv1",
        scheduler_name="ddpm",
        guidance_scale=7.5,
        num_inference_steps=NUM_STEPS,
        num_seeds=NUM_SEEDS,
        num_baseline_seeds=NUM_BASELINE_SEEDS,
    )
    monkeypatch.setattr(
        baseline,
        "build_scheduler",
        lambda _original, requested: SimpleNamespace(
            scheduler=object(),
            name="ddim",
            class_name="DDIMScheduler",
            config={"requested": requested},
        ),
    )
    components = SimpleNamespace(original_scheduler=object())
    with pytest.raises(
        baseline.UnconditionalBaselineError, match="scheduler identity differs"
    ):
        baseline._validate_active_scheduler(components, contract)


def test_loaded_runtime_package_versions_must_match_reference(
    tmp_path: Path,
) -> None:
    _create_reference_run(tmp_path)
    contract = baseline._load_source_contract(
        tmp_path,
        model_name="sdv1",
        scheduler_name="ddim",
        guidance_scale=7.5,
        num_inference_steps=NUM_STEPS,
        num_seeds=NUM_SEEDS,
        num_baseline_seeds=NUM_BASELINE_SEEDS,
    )
    components = SimpleNamespace(
        model_id=contract.science["model_id"],
        model_revision=contract.science["model_revision"],
        vae_id=contract.science["vae_id"],
        vae_revision=contract.science["vae_revision"],
        inference_dtype=contract.stored_dtype,
        package_versions={"torch": "different"},
    )
    with pytest.raises(baseline.UnconditionalBaselineError, match="package_versions"):
        baseline._validate_loaded_components(components, contract)


def test_reference_seed_namespace_is_mandatory(tmp_path: Path) -> None:
    paths = _create_reference_run(tmp_path)
    experiment = paths.run_directory.parent / "experiment_S0_N2"
    paths.run_directory.rename(experiment)
    with pytest.raises(
        baseline.UnconditionalBaselineError, match="generation run is missing"
    ):
        _load(tmp_path, load_tensor=False)


def test_ddim_and_ddpm_artifacts_are_scheduler_isolated(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = {
        scheduler_name: _create_reference_run(tmp_path, scheduler_name)
        for scheduler_name in ("ddim", "ddpm")
    }
    _install_fake_cpu_inference(monkeypatch)
    artifacts = {
        scheduler_name: _compute(tmp_path, scheduler_name=scheduler_name)
        for scheduler_name in ("ddim", "ddpm")
    }

    for scheduler_name, artifact in artifacts.items():
        assert artifact.metadata["scheduler_name"] == scheduler_name
        assert (
            artifact.tensor_path.parent.parent.parent
            == paths[scheduler_name].run_directory
        )
        loaded = _load(tmp_path, scheduler_name=scheduler_name)
        metadata_only = _load(
            tmp_path, scheduler_name=scheduler_name, load_tensor=False
        )
        assert loaded.mu_hat is not None
        assert metadata_only.mu_hat is None
        torch.testing.assert_close(
            loaded.mu_hat,
            artifacts[scheduler_name].mu_hat,
            rtol=0.0,
            atol=0.0,
        )
        assert metadata_only.mu_hat_sha256 == artifact.mu_hat_sha256

    assert artifacts["ddim"].tensor_path != artifacts["ddpm"].tensor_path
    assert (
        artifacts["ddim"].source_scientific_config_hash
        != artifacts["ddpm"].source_scientific_config_hash
    )
    assert (
        artifacts["ddim"].source_schedule_sha256
        != artifacts["ddpm"].source_schedule_sha256
    )
    assert (
        artifacts["ddim"].metadata["input_identity_sha256"]
        != artifacts["ddpm"].metadata["input_identity_sha256"]
    )


@pytest.mark.parametrize("damage", ("science", "schedule", "metadata"))
def test_ddpm_scheduler_mismatches_are_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, damage: str
) -> None:
    paths = _create_reference_run(tmp_path, "ddpm")
    if damage == "science":
        configuration = read_json(paths.run_config)
        science = configuration["scientific_config"]
        assert isinstance(science, dict)
        science["scheduler"] = {
            "name": "ddim",
            "class": "DDIMScheduler",
            "config": {"num_train_timesteps": 10, "prediction_type": "epsilon"},
        }
        configuration["scientific_config_hash"] = canonical_hash(science)
        atomic_write_json(paths.run_config, configuration)
        with pytest.raises(
            baseline.UnconditionalBaselineError, match="reference.*scheduler"
        ):
            _load(tmp_path, scheduler_name="ddpm")
        return
    if damage == "schedule":
        schedule = _schedule("ddpm")
        schedule["scheduler_name"] = "ddim"
        atomic_torch_save(schedule, paths.schedule)
        with pytest.raises(
            baseline.UnconditionalBaselineError, match="saved generation schedule"
        ):
            _load(tmp_path, scheduler_name="ddpm")
        return

    _install_fake_cpu_inference(monkeypatch)
    artifact = _compute(tmp_path, scheduler_name="ddpm")
    metadata = read_json(artifact.metadata_path)
    metadata["scheduler_name"] = "ddim"
    atomic_write_json(artifact.metadata_path, metadata)
    with pytest.raises(
        baseline.UnconditionalBaselineError, match="metadata differs.*scheduler_name"
    ):
        _load(tmp_path, scheduler_name="ddpm", load_tensor=False)


def test_ddpm_metadata_only_validation_names_the_actual_scheduler(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _create_reference_run(tmp_path, "ddpm")
    _install_fake_cpu_inference(monkeypatch)
    artifact = _compute(tmp_path, scheduler_name="ddpm")
    metadata = read_json(artifact.metadata_path)
    metadata["init_noise_sigma"] = 2.0
    atomic_write_json(artifact.metadata_path, metadata)
    with pytest.raises(
        baseline.UnconditionalBaselineError,
        match="DDPM init_noise_sigma must equal 1",
    ):
        _load(tmp_path, scheduler_name="ddpm", load_tensor=False)


def test_seed_block_bounds_and_public_namespace() -> None:
    assert baseline.unconditional_baseline_namespace(20, 1000) == "S20_N1000"
    with pytest.raises(baseline.UnconditionalBaselineError, match="seed block"):
        baseline.unconditional_baseline_namespace(MAX_SEED, 2)
    with pytest.raises(baseline.UnconditionalBaselineError, match="positive integer"):
        baseline.unconditional_baseline_namespace(0, 0)


def test_saved_mu_hat_is_float64_contiguous_cpu(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _create_reference_run(tmp_path)
    _install_fake_cpu_inference(monkeypatch)
    artifact = _compute(tmp_path)
    payload = safe_torch_load(artifact.tensor_path)
    assert isinstance(payload, torch.Tensor)
    assert payload.dtype == torch.float64
    assert payload.device.type == "cpu"
    assert payload.is_contiguous()
    assert torch.isfinite(payload).all()


def test_source_has_no_prompt_record_selection_sscd_or_target_access() -> None:
    source = inspect.getsource(baseline)
    assert "list_completed_records" not in source
    assert "validate_generation_record" not in source
    assert "utils.data.selection" not in source
    assert "experiments.sscd" not in source
    assert "target_latent" not in source
    assert "noise_prediction" not in source
