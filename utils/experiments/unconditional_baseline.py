"""Compute and load the shared model-implied unconditional baseline."""

from __future__ import annotations

import hashlib
import math
import multiprocessing
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from utils.common.cli import validate_seed_block
from utils.common.io import (
    CacheIOError,
    atomic_torch_save,
    atomic_write_json,
    canonical_hash,
    file_sha256,
    read_json,
    safe_torch_load,
    utc_now,
)
from utils.experiments.cache import (
    GENERATION_SELECTION_POLICY,
    GenerationCacheError,
    GenerationPaths,
    generation_paths,
    require_generation_run,
)
from utils.models.devices import (
    DeviceSelectionError,
    configure_worker_cpu_threads,
    resolve_devices,
    round_robin_shard,
    worker_count_for_tasks,
)
from utils.models.loading import (
    compile_loaded_unet,
    load_model_components,
    preflight_model_components,
    select_runtime,
)
from utils.models.prediction_conversion import schedule_alpha_sigma
from utils.models.registry import get_model_spec
from utils.models.sampling import (
    encode_prompt_condition,
    make_initial_noise,
    predict_conditional_epsilon,
    validate_latent_shape,
)
from utils.models.schedulers import build_scheduler


BASELINE_SCHEMA_VERSION = 2
BASELINE_ALGORITHM_VERSION = 1
BASELINE_ARTIFACT = "model_implied_unconditional_baseline"
BASELINE_DIRECTORY_NAME = "unconditional_baseline"
BASELINE_TENSOR_NAME = "mu_hat.pt"
BASELINE_METADATA_NAME = "metadata.json"
DEFAULT_NUM_BASELINE_SEEDS = 1000
DEFAULT_SAMPLE_BATCH_SIZE = 8
PROGRESS_DESCRIPTION = "[Unconditional baseline] Gaussian seeds"


class UnconditionalBaselineError(RuntimeError):
    """A source generation run or saved unconditional baseline is invalid."""


@dataclass(frozen=True, slots=True)
class UnconditionalBaselineArtifact:
    """A validated baseline tensor and the provenance needed by consumers."""

    mu_hat: torch.Tensor | None
    mu_hat_sha256: str
    metadata: Mapping[str, Any]
    metadata_path: Path
    tensor_path: Path
    source_scientific_config_hash: str
    source_schedule_sha256: str
    source_seed_start: int
    source_seeds: tuple[int, ...]
    num_baseline_seeds: int
    timestep: int
    alpha_t: float
    sigma_t: float
    latent_shape: tuple[int, int, int]

    @property
    def baseline_seed_start(self) -> int:
        """Return the first dedicated baseline seed."""

        return self.source_seed_start

    @property
    def baseline_seeds(self) -> tuple[int, ...]:
        """Return the dedicated baseline seed pool."""

        return self.source_seeds


@dataclass(frozen=True, slots=True)
class _ReferenceIdentity:
    root: Path
    paths: GenerationPaths
    science: Mapping[str, Any]
    scientific_hash: str
    schedule_sha256: str
    reference_seeds: tuple[int, ...]
    latent_shape: tuple[int, int, int]
    stored_dtype: torch.dtype


@dataclass(frozen=True, slots=True)
class _SourceContract:
    identity: _ReferenceIdentity
    schedule_payload: Mapping[str, Any] | None
    scheduler_config: Mapping[str, Any]
    init_noise_sigma: float
    timestep: int
    alpha_t: float
    sigma_t: float
    baseline_seeds: tuple[int, ...]

    @property
    def root(self) -> Path:
        return self.identity.root

    @property
    def paths(self) -> GenerationPaths:
        return self.identity.paths

    @property
    def science(self) -> Mapping[str, Any]:
        return self.identity.science

    @property
    def scientific_hash(self) -> str:
        return self.identity.scientific_hash

    @property
    def schedule_sha256(self) -> str:
        return self.identity.schedule_sha256

    @property
    def latent_shape(self) -> tuple[int, int, int]:
        return self.identity.latent_shape

    @property
    def stored_dtype(self) -> torch.dtype:
        return self.identity.stored_dtype


@dataclass(frozen=True, slots=True)
class _InferenceRuntime:
    contract: _SourceContract
    components: Any
    scheduler: Any
    empty_condition: torch.Tensor


@dataclass(frozen=True, slots=True)
class _SeedBatchEstimate:
    entries: tuple[tuple[int, int], ...]
    estimates: torch.Tensor


_WORKER_RUNTIME: _InferenceRuntime | None = None


def unconditional_baseline_namespace(seed_start: int, num_baseline_seeds: int) -> str:
    """Return the stable namespace for one dedicated baseline seed block."""

    try:
        start, count = validate_seed_block(seed_start, num_baseline_seeds)
    except ValueError as error:
        raise UnconditionalBaselineError(str(error)) from error
    return f"S{start}_N{count}"


def unconditional_baseline_artifact_paths(
    reference_run_directory: str | Path,
    *,
    seed_start: int,
    num_baseline_seeds: int,
) -> tuple[Path, Path]:
    """Return mu_hat and metadata paths for one baseline seed namespace."""

    directory = (
        Path(reference_run_directory)
        / BASELINE_DIRECTORY_NAME
        / unconditional_baseline_namespace(seed_start, num_baseline_seeds)
    )
    return directory / BASELINE_TENSOR_NAME, directory / BASELINE_METADATA_NAME


def compute_unconditional_baseline(
    project_root: str | Path,
    *,
    model_name: str,
    scheduler_name: str,
    guidance_scale: float,
    num_inference_steps: int,
    num_seeds: int,
    num_baseline_seeds: int = DEFAULT_NUM_BASELINE_SEEDS,
    device: str | torch.device = "auto",
    progress_factory: Callable[..., Any] = tqdm,
) -> UnconditionalBaselineArtifact:
    """Estimate mu_hat from a dedicated prompt-independent Gaussian seed pool."""

    contract = _load_source_contract(
        project_root,
        model_name=model_name,
        scheduler_name=scheduler_name,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        num_seeds=num_seeds,
        num_baseline_seeds=num_baseline_seeds,
    )
    try:
        devices = resolve_devices(device)
    except DeviceSelectionError as error:
        raise UnconditionalBaselineError(str(error)) from error
    try:
        estimates = _compute_seedwise_estimates(
            contract,
            devices=devices,
            progress_factory=progress_factory,
        )
    except UnconditionalBaselineError:
        raise
    except Exception as error:
        raise UnconditionalBaselineError(
            f"unconditional baseline inference failed: {type(error).__name__}: {error}"
        ) from error

    accumulator = torch.zeros(contract.latent_shape, dtype=torch.float64)
    for position in range(len(contract.baseline_seeds)):
        accumulator.add_(estimates[position])
    mu_hat = accumulator.div(float(len(contract.baseline_seeds))).contiguous()
    if not bool(torch.isfinite(mu_hat).all()):
        raise UnconditionalBaselineError("computed mu_hat is non-finite")
    mu_hat_norm_rmse = float(mu_hat.square().mean().sqrt())
    if not math.isfinite(mu_hat_norm_rmse) or mu_hat_norm_rmse < 0.0:
        raise UnconditionalBaselineError("computed mu_hat norm is invalid")

    tensor_path, metadata_path = unconditional_baseline_artifact_paths(
        contract.paths.run_directory,
        seed_start=contract.baseline_seeds[0],
        num_baseline_seeds=len(contract.baseline_seeds),
    )
    _prepare_artifact_directory(tensor_path.parent)
    mu_hat_sha256 = atomic_torch_save(mu_hat, tensor_path)
    mu_hat_value_sha256 = _tensor_value_sha256(mu_hat)
    metadata: dict[str, object] = {
        "schema_version": BASELINE_SCHEMA_VERSION,
        "algorithm_version": BASELINE_ALGORITHM_VERSION,
        "artifact": BASELINE_ARTIFACT,
        "definition": (
            "mean over dedicated Gaussian seeds of "
            "xhat_empty=(x_T-sigma_T*epsilon_empty(x_T,T))/alpha_T"
        ),
        "condition": "empty string only",
        "model_name": model_name,
        "model_id": contract.science.get("model_id"),
        "model_revision": contract.science.get("model_revision"),
        "vae_id": contract.science.get("vae_id"),
        "vae_revision": contract.science.get("vae_revision"),
        "scheduler_name": scheduler_name,
        "guidance_scale": float(guidance_scale),
        "num_inference_steps": num_inference_steps,
        "source_run_directory": _relative(contract.paths.run_directory, contract.root),
        "source_scientific_config_hash": contract.scientific_hash,
        "source_schedule_sha256": contract.schedule_sha256,
        "reference_run_seed_start": num_seeds,
        "reference_run_seeds": list(contract.identity.reference_seeds),
        "num_reference_run_seeds": num_seeds,
        "baseline_seed_start": contract.baseline_seeds[0],
        "baseline_seeds": list(contract.baseline_seeds),
        "num_baseline_seeds": len(contract.baseline_seeds),
        "baseline_seed_fingerprint_sha256": _seed_fingerprint(contract.baseline_seeds),
        "input_identity_sha256": _input_identity_sha256(contract),
        "baseline_estimate_value_sha256": _tensor_value_sha256(estimates),
        "timestep": contract.timestep,
        "alpha_T": contract.alpha_t,
        "sigma_T": contract.sigma_t,
        "snr_T": contract.alpha_t**2 / contract.sigma_t**2,
        "init_noise_sigma": contract.init_noise_sigma,
        "latent_shape": list(contract.latent_shape),
        "latent_dimension": math.prod(contract.latent_shape),
        "inference_dtype": _dtype_name(contract.stored_dtype),
        "package_versions": dict(contract.science["package_versions"]),
        "accumulation_dtype": "float64",
        "weighting": "one unit per baseline seed in ascending seed order",
        "runtime": {
            "requested_device": str(device),
            "resolved_devices": [str(value) for value in devices],
            "sample_batch_size": DEFAULT_SAMPLE_BATCH_SIZE,
        },
        "mu_hat_norm_rmse": mu_hat_norm_rmse,
        "coordinate_artifacts": {
            "mu_hat": {
                "path": _relative(tensor_path, contract.root),
                "dtype": "float64",
                "shape": list(contract.latent_shape),
                "sha256": mu_hat_sha256,
                "value_sha256": mu_hat_value_sha256,
            }
        },
        "created_at_utc": utc_now(),
    }
    atomic_write_json(metadata_path, metadata)
    return _artifact_from_validated(
        contract,
        metadata,
        metadata_path,
        tensor_path,
        mu_hat_sha256,
        mu_hat,
    )


def load_unconditional_baseline(
    project_root: str | Path,
    *,
    model_name: str,
    scheduler_name: str,
    guidance_scale: float,
    num_inference_steps: int,
    num_seeds: int,
    num_baseline_seeds: int = DEFAULT_NUM_BASELINE_SEEDS,
    load_tensor: bool = True,
) -> UnconditionalBaselineArtifact:
    """Load and validate one B-specific baseline under the reference run."""

    if not isinstance(load_tensor, bool):
        raise UnconditionalBaselineError("load_tensor must be a boolean")
    if not load_tensor:
        from utils.experiments._unconditional_baseline_metadata import (
            load_unconditional_baseline_metadata_only,
        )

        return load_unconditional_baseline_metadata_only(
            project_root,
            model_name=model_name,
            scheduler_name=scheduler_name,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            num_seeds=num_seeds,
            num_baseline_seeds=num_baseline_seeds,
        )

    contract = _load_source_contract(
        project_root,
        model_name=model_name,
        scheduler_name=scheduler_name,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        num_seeds=num_seeds,
        num_baseline_seeds=num_baseline_seeds,
    )
    tensor_path, metadata_path = unconditional_baseline_artifact_paths(
        contract.paths.run_directory,
        seed_start=contract.baseline_seeds[0],
        num_baseline_seeds=len(contract.baseline_seeds),
    )
    metadata = _read_artifact_metadata(metadata_path, tensor_path)
    _validate_metadata(metadata, contract, metadata_path, tensor_path)
    description = _mu_hat_description(metadata)
    expected_file_hash = description.get("sha256")
    if (
        not _is_sha256(expected_file_hash)
        or file_sha256(tensor_path) != expected_file_hash
    ):
        raise UnconditionalBaselineError(
            "unconditional baseline mu_hat SHA-256 differs"
        )
    try:
        payload = safe_torch_load(tensor_path)
    except CacheIOError as error:
        raise UnconditionalBaselineError(str(error)) from error
    mu_hat = _validate_mu_hat(payload, contract.latent_shape)
    expected_value_hash = description.get("value_sha256")
    if (
        not _is_sha256(expected_value_hash)
        or _tensor_value_sha256(mu_hat) != expected_value_hash
    ):
        raise UnconditionalBaselineError(
            "unconditional baseline mu_hat value SHA-256 differs"
        )
    observed_rmse = float(mu_hat.square().mean().sqrt())
    expected_rmse = _nonnegative_finite_float(
        metadata.get("mu_hat_norm_rmse"), "mu_hat norm"
    )
    if not math.isclose(observed_rmse, expected_rmse, rel_tol=1e-12, abs_tol=1e-12):
        raise UnconditionalBaselineError("unconditional baseline mu_hat norm differs")
    return _artifact_from_validated(
        contract,
        metadata,
        metadata_path,
        tensor_path,
        str(expected_file_hash),
        mu_hat,
    )


def _load_reference_identity(
    project_root: str | Path,
    *,
    model_name: str,
    scheduler_name: str,
    guidance_scale: float,
    num_inference_steps: int,
    num_seeds: int,
) -> _ReferenceIdentity:
    root = Path(project_root).expanduser().resolve()
    if scheduler_name != "ddim":
        raise UnconditionalBaselineError(
            "unconditional baseline requires --scheduler ddim"
        )
    try:
        reference_seed_start, reference_count = validate_seed_block(
            num_seeds, num_seeds
        )
        paths = generation_paths(
            root,
            model_name=model_name,
            scheduler_name=scheduler_name,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            num_seeds=reference_count,
            seed_start=reference_seed_start,
        )
        configuration = require_generation_run(paths)
    except (CacheIOError, GenerationCacheError, ValueError) as error:
        raise UnconditionalBaselineError(str(error)) from error
    science = configuration.get("scientific_config")
    science_hash = configuration.get("scientific_config_hash")
    if (
        not isinstance(science, Mapping)
        or not _is_sha256(science_hash)
        or canonical_hash(science) != science_hash
    ):
        raise UnconditionalBaselineError("generation scientific provenance is invalid")

    spec = get_model_spec(model_name)
    reference_seeds = tuple(
        range(reference_seed_start, reference_seed_start + reference_count)
    )
    expected = {
        "model_cli_name": model_name,
        "dataset_model": spec.dataset_model,
        "model_id": spec.model_id,
        "guidance_scale": float(guidance_scale),
        "num_inference_steps": num_inference_steps,
        "num_seeds": reference_count,
        "seeds": list(reference_seeds),
        "selection_policy": GENERATION_SELECTION_POLICY,
        "trajectory_order": "noise_to_image",
        "stored_prediction_type": "epsilon",
    }
    wrong = [key for key, value in expected.items() if science.get(key) != value]
    scheduler = science.get("scheduler")
    if not isinstance(scheduler, Mapping) or scheduler.get("name") != "ddim":
        wrong.append("scheduler")
    for key in ("model_revision", "vae_id", "vae_revision", "inference_dtype"):
        if not isinstance(science.get(key), str) or not str(science.get(key)):
            wrong.append(key)
    package_versions = science.get("package_versions")
    if (
        not isinstance(package_versions, Mapping)
        or set(package_versions) != {"torch", "diffusers", "transformers", "cuda"}
        or any(
            value is not None and (not isinstance(value, str) or not value)
            for value in package_versions.values()
        )
    ):
        wrong.append("package_versions")
    if wrong:
        raise UnconditionalBaselineError(
            "reference generation cache differs at: " + ", ".join(sorted(set(wrong)))
        )
    latent_shape = _latent_shape(science.get("latent_shape"))
    storage = science.get("scientific_tensor_storage")
    if not isinstance(storage, Mapping):
        raise UnconditionalBaselineError(
            "generation tensor storage metadata is invalid"
        )
    stored_dtype = _dtype_from_name(storage.get("dtype"))
    if (
        storage.get("device") != "cpu"
        or storage.get("layout") != "contiguous"
        or science.get("inference_dtype") != _dtype_name(stored_dtype)
    ):
        raise UnconditionalBaselineError(
            "generation tensor storage metadata is invalid"
        )
    if paths.schedule.is_symlink() or not paths.schedule.is_file():
        raise UnconditionalBaselineError(
            "saved generation schedule is missing or unsafe"
        )
    return _ReferenceIdentity(
        root=root,
        paths=paths,
        science=science,
        scientific_hash=str(science_hash),
        schedule_sha256=file_sha256(paths.schedule),
        reference_seeds=reference_seeds,
        latent_shape=latent_shape,
        stored_dtype=stored_dtype,
    )


def _load_source_contract(
    project_root: str | Path,
    *,
    model_name: str,
    scheduler_name: str,
    guidance_scale: float,
    num_inference_steps: int,
    num_seeds: int,
    num_baseline_seeds: int,
) -> _SourceContract:
    identity = _load_reference_identity(
        project_root,
        model_name=model_name,
        scheduler_name=scheduler_name,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        num_seeds=num_seeds,
    )
    baseline_seeds = _baseline_seed_values(num_seeds, num_baseline_seeds)
    try:
        schedule = safe_torch_load(identity.paths.schedule)
    except CacheIOError as error:
        raise UnconditionalBaselineError(str(error)) from error
    if not isinstance(schedule, Mapping):
        raise UnconditionalBaselineError("saved generation schedule must be a mapping")
    timesteps = _schedule_tensor(
        schedule, "timesteps", num_inference_steps, torch.int64
    )
    alphas = _schedule_tensor(schedule, "alpha_t", num_inference_steps, torch.float32)
    sigmas = _schedule_tensor(schedule, "sigma_t", num_inference_steps, torch.float32)
    cumulative = _schedule_tensor(
        schedule, "alphas_cumprod_t", num_inference_steps, torch.float32
    )
    if (
        bool((timesteps < 0).any())
        or len(set(timesteps.tolist())) != num_inference_steps
        or bool((alphas <= 0).any())
        or bool((sigmas <= 0).any())
        or not torch.allclose(alphas.square(), cumulative, rtol=1e-5, atol=1e-6)
        or not torch.allclose(sigmas.square(), 1.0 - cumulative, rtol=1e-5, atol=1e-6)
    ):
        raise UnconditionalBaselineError("saved schedule coefficients are invalid")
    scheduler_metadata = identity.science.get("scheduler")
    if not isinstance(scheduler_metadata, Mapping):
        raise UnconditionalBaselineError("generation scheduler metadata is invalid")
    schedule_expected = {
        "scheduler_name": "ddim",
        "scheduler_class": scheduler_metadata.get("class"),
        "native_prediction_type": identity.science.get("native_prediction_type"),
        "stored_prediction_type": "epsilon",
        "trajectory_order": "noise_to_image",
    }
    schedule_wrong = [
        key for key, value in schedule_expected.items() if schedule.get(key) != value
    ]
    if schedule_wrong:
        raise UnconditionalBaselineError(
            "saved generation schedule differs at: " + ", ".join(schedule_wrong)
        )
    scheduler_config = _normalized_scheduler_config(schedule.get("scheduler_config"))
    if canonical_hash(scheduler_config) != canonical_hash(
        _normalized_scheduler_config(scheduler_metadata.get("config"))
    ):
        raise UnconditionalBaselineError("saved scheduler configurations differ")
    init_noise_sigma = _positive_finite_float(
        schedule.get("init_noise_sigma"), "scheduler initial-noise scale"
    )
    if not math.isclose(init_noise_sigma, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise UnconditionalBaselineError("DDIM init_noise_sigma must equal 1")
    return _SourceContract(
        identity=identity,
        schedule_payload=schedule,
        scheduler_config=scheduler_config,
        init_noise_sigma=init_noise_sigma,
        timestep=int(timesteps[0]),
        alpha_t=float(alphas[0].double()),
        sigma_t=float(sigmas[0].double()),
        baseline_seeds=baseline_seeds,
    )


def _baseline_seed_values(seed_start: int, num_baseline_seeds: int) -> tuple[int, ...]:
    try:
        start, count = validate_seed_block(seed_start, num_baseline_seeds)
    except ValueError as error:
        raise UnconditionalBaselineError(str(error)) from error
    return tuple(range(start, start + count))


def _revision_resolver(contract: _SourceContract) -> Callable[[str], str]:
    revisions = {
        str(contract.science["model_id"]): contract.science["model_revision"],
        str(contract.science["vae_id"]): contract.science["vae_revision"],
    }

    def resolve(repository_id: str) -> str:
        revision = revisions.get(repository_id)
        if not isinstance(revision, str) or not revision:
            raise UnconditionalBaselineError(
                f"reference cache has no pinned revision for {repository_id}"
            )
        return revision

    return resolve


def _validate_active_scheduler(components: Any, contract: _SourceContract) -> Any:
    result = build_scheduler(components.original_scheduler, "ddim")
    scheduler = result.scheduler
    setter = getattr(scheduler, "set_timesteps", None)
    if not callable(setter):
        raise UnconditionalBaselineError(
            "active DDIM scheduler has no timestep interface"
        )
    setter(
        int(contract.science["num_inference_steps"]),
        device=torch.device(components.device),
    )
    active_timesteps = torch.as_tensor(scheduler.timesteps).detach().cpu().long()
    saved = contract.schedule_payload
    if saved is None:
        raise UnconditionalBaselineError("computation requires the saved schedule")
    saved_timesteps = _schedule_tensor(
        saved,
        "timesteps",
        int(contract.science["num_inference_steps"]),
        torch.int64,
    )
    if not torch.equal(active_timesteps, saved_timesteps):
        raise UnconditionalBaselineError(
            "active DDIM timesteps differ from the reference cache"
        )
    active_coefficients = schedule_alpha_sigma(
        scheduler, active_timesteps, dtype=torch.float32
    )
    saved_coefficients = tuple(
        _schedule_tensor(
            saved,
            name,
            int(contract.science["num_inference_steps"]),
            torch.float32,
        )
        for name in ("alpha_t", "sigma_t", "alphas_cumprod_t")
    )
    if any(
        not torch.equal(active, cached)
        for active, cached in zip(active_coefficients, saved_coefficients, strict=True)
    ):
        raise UnconditionalBaselineError(
            "active DDIM coefficients differ from the reference cache"
        )
    if canonical_hash(_normalized_scheduler_config(result.config)) != canonical_hash(
        contract.scheduler_config
    ):
        raise UnconditionalBaselineError(
            "active DDIM configuration differs from the reference cache"
        )
    active_sigma = _positive_finite_float(
        getattr(scheduler, "init_noise_sigma", None),
        "active DDIM initial-noise scale",
    )
    if not math.isclose(
        active_sigma, contract.init_noise_sigma, rel_tol=0.0, abs_tol=1e-12
    ):
        raise UnconditionalBaselineError(
            "active DDIM initial-noise scale differs from the reference cache"
        )
    return scheduler


def _validate_loaded_components(components: Any, contract: _SourceContract) -> None:
    expected = {
        "model_id": contract.science.get("model_id"),
        "model_revision": contract.science.get("model_revision"),
        "vae_id": contract.science.get("vae_id"),
        "vae_revision": contract.science.get("vae_revision"),
    }
    observed = {
        "model_id": getattr(components, "model_id", None),
        "model_revision": getattr(components, "model_revision", None),
        "vae_id": getattr(components, "vae_id", None),
        "vae_revision": getattr(components, "vae_revision", None),
    }
    wrong = [key for key, value in expected.items() if observed.get(key) != value]
    if getattr(components, "inference_dtype", None) != contract.stored_dtype:
        wrong.append("inference_dtype")
    if getattr(components, "package_versions", None) != contract.science.get(
        "package_versions"
    ):
        wrong.append("package_versions")
    if wrong:
        raise UnconditionalBaselineError(
            "loaded components differ from the reference cache at: " + ", ".join(wrong)
        )
    if (
        validate_latent_shape(components.unet, expected_shape=contract.latent_shape)
        != contract.latent_shape
    ):
        raise UnconditionalBaselineError(
            "loaded UNet latent shape differs from the reference cache"
        )


def _offload_unused_vae(components: Any) -> None:
    selected = torch.device(components.device)
    if selected.type == "cpu":
        return
    move = getattr(components.vae, "to", None)
    if not callable(move):
        raise UnconditionalBaselineError("loaded VAE has no device-transfer interface")
    move(device=torch.device("cpu"), dtype=torch.float32)
    if selected.type == "cuda":
        torch.cuda.empty_cache()


def _build_inference_runtime(
    contract: _SourceContract,
    *,
    device: str | torch.device,
    worker_count: int,
) -> _InferenceRuntime:
    configure_worker_cpu_threads(worker_count)
    concrete_device = torch.device(device)
    if concrete_device.type == "cuda":
        if concrete_device.index is None:
            raise UnconditionalBaselineError(
                "baseline CUDA workers require concrete device indices"
            )
        torch.cuda.set_device(concrete_device)
    selected = select_runtime(concrete_device, warn_without_cuda=False)
    components = load_model_components(
        get_model_spec(str(contract.science["model_cli_name"])),
        runtime=selected,
        revision_resolver=_revision_resolver(contract),
    )
    scheduler = _validate_active_scheduler(components, contract)
    preflight_model_components(
        components,
        scheduler=scheduler,
        num_inference_steps=int(contract.science["num_inference_steps"]),
        active_cuda_index=(
            concrete_device.index if concrete_device.type == "cuda" else None
        ),
    )
    _validate_loaded_components(components, contract)
    _offload_unused_vae(components)
    components = compile_loaded_unet(components)
    empty_condition = encode_prompt_condition(
        "",
        components.tokenizer,
        components.text_encoder,
        components.device,
        components.inference_dtype,
    )
    return _InferenceRuntime(
        contract=contract,
        components=components,
        scheduler=scheduler,
        empty_condition=empty_condition,
    )


def _estimate_seed_batch(
    runtime: _InferenceRuntime,
    entries: Sequence[tuple[int, int]],
) -> _SeedBatchEstimate:
    canonical_entries = tuple((int(position), int(seed)) for position, seed in entries)
    if not canonical_entries:
        raise UnconditionalBaselineError("baseline seed batch must not be empty")
    seeds = tuple(seed for _position, seed in canonical_entries)
    raw = make_initial_noise(seeds, runtime.contract.latent_shape)
    expected_shape = (len(seeds), *runtime.contract.latent_shape)
    if (
        not isinstance(raw, torch.Tensor)
        or tuple(raw.shape) != expected_shape
        or raw.dtype != torch.float32
        or raw.device.type != "cpu"
        or not raw.is_contiguous()
        or raw.requires_grad
        or not bool(torch.isfinite(raw).all())
    ):
        raise UnconditionalBaselineError("generated baseline Gaussian noise is invalid")
    device = torch.device(runtime.components.device)
    with torch.inference_mode():
        x_t = raw.to(device=device, dtype=runtime.components.inference_dtype).mul(
            runtime.contract.init_noise_sigma
        )
        epsilon = predict_conditional_epsilon(
            samples=x_t,
            timestep=runtime.contract.timestep,
            condition=runtime.empty_condition,
            unet=runtime.components.unet,
            scheduler=runtime.scheduler,
            conversion_sample=x_t,
        )
        if (
            not isinstance(epsilon, torch.Tensor)
            or tuple(epsilon.shape) != expected_shape
            or epsilon.device != x_t.device
            or epsilon.dtype != x_t.dtype
            or not bool(torch.isfinite(epsilon).all())
        ):
            raise UnconditionalBaselineError(
                "unconditional epsilon prediction is invalid"
            )
        estimates = (
            x_t.detach().cpu().double()
            - runtime.contract.sigma_t * epsilon.detach().cpu().double()
        ).div(runtime.contract.alpha_t)
    estimates = estimates.contiguous()
    if (
        tuple(estimates.shape) != expected_shape
        or estimates.dtype != torch.float64
        or not bool(torch.isfinite(estimates).all())
    ):
        raise UnconditionalBaselineError("seedwise unconditional estimates are invalid")
    return _SeedBatchEstimate(canonical_entries, estimates)


def _is_cuda_out_of_memory(error: RuntimeError, device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    out_of_memory_type = getattr(torch.cuda, "OutOfMemoryError", ())
    return (
        isinstance(error, out_of_memory_type)
        or "out of memory" in str(error).casefold()
    )


def _estimate_seed_batch_adaptive(
    runtime: _InferenceRuntime,
    entries: Sequence[tuple[int, int]],
) -> _SeedBatchEstimate:
    canonical = tuple(entries)
    try:
        return _estimate_seed_batch(runtime, canonical)
    except RuntimeError as error:
        device = torch.device(runtime.components.device)
        if len(canonical) == 1 or not _is_cuda_out_of_memory(error, device):
            raise
        torch.cuda.empty_cache()
        middle = len(canonical) // 2
        left = _estimate_seed_batch_adaptive(runtime, canonical[:middle])
        right = _estimate_seed_batch_adaptive(runtime, canonical[middle:])
        return _SeedBatchEstimate(
            entries=left.entries + right.entries,
            estimates=torch.cat((left.estimates, right.estimates), dim=0).contiguous(),
        )


def _initialize_inference_worker(
    contract: _SourceContract,
    device: str,
    worker_count: int,
) -> None:
    global _WORKER_RUNTIME
    _WORKER_RUNTIME = _build_inference_runtime(
        contract,
        device=device,
        worker_count=worker_count,
    )


def _inference_worker_task(
    entries: tuple[tuple[int, int], ...],
) -> _SeedBatchEstimate:
    if _WORKER_RUNTIME is None:
        raise UnconditionalBaselineError("baseline worker was not initialized")
    return _estimate_seed_batch_adaptive(_WORKER_RUNTIME, entries)


def _batches(
    entries: Sequence[tuple[int, int]], batch_size: int = DEFAULT_SAMPLE_BATCH_SIZE
) -> tuple[tuple[tuple[int, int], ...], ...]:
    if batch_size <= 0:
        raise UnconditionalBaselineError("baseline sample batch size must be positive")
    return tuple(
        tuple(entries[start : start + batch_size])
        for start in range(0, len(entries), batch_size)
    )


def _place_batch(
    destination: torch.Tensor,
    observed: set[int],
    result: _SeedBatchEstimate,
    expected_entries: Sequence[tuple[int, int]],
    contract: _SourceContract,
) -> None:
    expected = tuple(expected_entries)
    if result.entries != expected:
        raise UnconditionalBaselineError(
            "baseline worker changed seed order or identity"
        )
    if (
        result.estimates.dtype != torch.float64
        or result.estimates.device.type != "cpu"
        or not result.estimates.is_contiguous()
        or tuple(result.estimates.shape) != (len(expected), *contract.latent_shape)
        or not bool(torch.isfinite(result.estimates).all())
    ):
        raise UnconditionalBaselineError("baseline worker returned invalid estimates")
    for local_position, (position, seed) in enumerate(expected):
        if (
            position in observed
            or not 0 <= position < len(contract.baseline_seeds)
            or contract.baseline_seeds[position] != seed
        ):
            raise UnconditionalBaselineError(
                "baseline workers returned duplicate or invalid seed positions"
            )
        destination[position].copy_(result.estimates[local_position])
        observed.add(position)


def _compute_seedwise_estimates(
    contract: _SourceContract,
    *,
    devices: Sequence[torch.device],
    progress_factory: Callable[..., Any],
) -> torch.Tensor:
    if not devices:
        raise UnconditionalBaselineError("at least one execution device is required")
    entries = tuple(enumerate(contract.baseline_seeds))
    worker_count = worker_count_for_tasks(devices, len(entries))
    active_devices = tuple(devices[:worker_count])
    use_parallel = worker_count > 1
    if use_parallel and not all(
        device.type == "cuda" and device.index is not None for device in active_devices
    ):
        raise UnconditionalBaselineError(
            "multiple baseline workers require concrete CUDA devices"
        )

    destination = torch.empty(
        (len(entries), *contract.latent_shape), dtype=torch.float64
    )
    observed: set[int] = set()
    progress = progress_factory(
        total=len(entries),
        desc=PROGRESS_DESCRIPTION,
        unit="seed",
        dynamic_ncols=True,
    )
    try:
        if not use_parallel:
            runtime = _build_inference_runtime(
                contract,
                device=active_devices[0],
                worker_count=1,
            )
            for batch in _batches(entries):
                result = _estimate_seed_batch_adaptive(runtime, batch)
                _place_batch(destination, observed, result, batch, contract)
                progress.update(len(batch))
        else:
            context = multiprocessing.get_context("spawn")
            futures: dict[Future[_SeedBatchEstimate], tuple[tuple[int, int], ...]] = {}
            with ExitStack() as stack:
                for worker_index, selected_device in enumerate(active_devices):
                    executor = stack.enter_context(
                        ProcessPoolExecutor(
                            max_workers=1,
                            mp_context=context,
                            initializer=_initialize_inference_worker,
                            initargs=(
                                contract,
                                str(selected_device),
                                worker_count,
                            ),
                        )
                    )
                    shard = round_robin_shard(
                        entries,
                        worker_index=worker_index,
                        worker_count=worker_count,
                    )
                    for batch in _batches(shard):
                        futures[executor.submit(_inference_worker_task, batch)] = batch
                for future in as_completed(futures):
                    batch = futures[future]
                    result = future.result()
                    _place_batch(destination, observed, result, batch, contract)
                    progress.update(len(batch))
    finally:
        close = getattr(progress, "close", None)
        if callable(close):
            close()
    if observed != set(range(len(entries))):
        raise UnconditionalBaselineError(
            "baseline inference returned an incomplete seed population"
        )
    return destination.contiguous()


def _input_identity_sha256(contract: _SourceContract) -> str:
    return canonical_hash(
        {
            "algorithm_version": BASELINE_ALGORITHM_VERSION,
            "source_scientific_config_hash": contract.scientific_hash,
            "source_schedule_sha256": contract.schedule_sha256,
            "baseline_seed_start": contract.baseline_seeds[0],
            "baseline_seeds": list(contract.baseline_seeds),
            "num_baseline_seeds": len(contract.baseline_seeds),
            "timestep": contract.timestep,
            "alpha_T": contract.alpha_t,
            "sigma_T": contract.sigma_t,
            "init_noise_sigma": contract.init_noise_sigma,
            "condition": "",
            "weighting": "one unit per seed in ascending seed order",
        }
    )


def _seed_fingerprint(seeds: Sequence[int]) -> str:
    return canonical_hash({"seeds": [int(seed) for seed in seeds]})


def _validate_metadata(
    metadata: Mapping[str, object],
    contract: _SourceContract,
    metadata_path: Path,
    tensor_path: Path,
) -> None:
    expected = {
        "schema_version": BASELINE_SCHEMA_VERSION,
        "algorithm_version": BASELINE_ALGORITHM_VERSION,
        "artifact": BASELINE_ARTIFACT,
        "definition": (
            "mean over dedicated Gaussian seeds of "
            "xhat_empty=(x_T-sigma_T*epsilon_empty(x_T,T))/alpha_T"
        ),
        "condition": "empty string only",
        "model_name": contract.science.get("model_cli_name"),
        "model_id": contract.science.get("model_id"),
        "model_revision": contract.science.get("model_revision"),
        "vae_id": contract.science.get("vae_id"),
        "vae_revision": contract.science.get("vae_revision"),
        "scheduler_name": "ddim",
        "guidance_scale": contract.science.get("guidance_scale"),
        "num_inference_steps": contract.science.get("num_inference_steps"),
        "source_run_directory": _relative(contract.paths.run_directory, contract.root),
        "source_scientific_config_hash": contract.scientific_hash,
        "source_schedule_sha256": contract.schedule_sha256,
        "reference_run_seed_start": contract.identity.reference_seeds[0],
        "reference_run_seeds": list(contract.identity.reference_seeds),
        "num_reference_run_seeds": len(contract.identity.reference_seeds),
        "baseline_seed_start": contract.baseline_seeds[0],
        "baseline_seeds": list(contract.baseline_seeds),
        "num_baseline_seeds": len(contract.baseline_seeds),
        "baseline_seed_fingerprint_sha256": _seed_fingerprint(contract.baseline_seeds),
        "input_identity_sha256": _input_identity_sha256(contract),
        "timestep": contract.timestep,
        "alpha_T": contract.alpha_t,
        "sigma_T": contract.sigma_t,
        "snr_T": contract.alpha_t**2 / contract.sigma_t**2,
        "init_noise_sigma": contract.init_noise_sigma,
        "latent_shape": list(contract.latent_shape),
        "latent_dimension": math.prod(contract.latent_shape),
        "inference_dtype": _dtype_name(contract.stored_dtype),
        "package_versions": dict(contract.science["package_versions"]),
        "accumulation_dtype": "float64",
        "weighting": "one unit per baseline seed in ascending seed order",
    }
    wrong = [key for key, value in expected.items() if metadata.get(key) != value]
    if wrong:
        raise UnconditionalBaselineError(
            "unconditional baseline metadata differs at: " + ", ".join(wrong)
        )
    legacy = {
        "number_of_prompts",
        "number_of_prompt_seed_entries",
        "number_of_unique_initial_latents",
        "duplicate_initial_latent_entries",
        "unique_initial_latent_value_sha256",
    }
    present_legacy = sorted(legacy.intersection(metadata))
    if present_legacy:
        raise UnconditionalBaselineError(
            "unconditional baseline contains obsolete prompt metadata: "
            + ", ".join(present_legacy)
        )
    if not _is_sha256(metadata.get("baseline_estimate_value_sha256")):
        raise UnconditionalBaselineError(
            "baseline estimate value fingerprint is invalid"
        )
    _nonnegative_finite_float(metadata.get("mu_hat_norm_rmse"), "mu_hat norm")
    created = metadata.get("created_at_utc")
    if not isinstance(created, str) or not created:
        raise UnconditionalBaselineError(
            "unconditional baseline metadata is incomplete"
        )
    runtime = metadata.get("runtime")
    if not isinstance(runtime, Mapping):
        raise UnconditionalBaselineError("baseline runtime metadata is invalid")
    resolved = runtime.get("resolved_devices")
    if (
        not isinstance(resolved, Sequence)
        or isinstance(resolved, (str, bytes))
        or not resolved
        or any(not isinstance(value, str) or not value for value in resolved)
    ):
        raise UnconditionalBaselineError("baseline runtime devices are invalid")
    if _positive_integer(runtime.get("sample_batch_size"), "sample batch size") <= 0:
        raise AssertionError("unreachable")

    description = _mu_hat_description(metadata)
    expected_description = {
        "path": _relative(tensor_path, contract.root),
        "dtype": "float64",
        "shape": list(contract.latent_shape),
    }
    description_wrong = [
        key
        for key, value in expected_description.items()
        if description.get(key) != value
    ]
    if (
        description_wrong
        or not _is_sha256(description.get("sha256"))
        or not _is_sha256(description.get("value_sha256"))
    ):
        raise UnconditionalBaselineError("mu_hat artifact description differs")
    if metadata_path.parent != tensor_path.parent:
        raise UnconditionalBaselineError("baseline artifact paths are inconsistent")


def _read_artifact_metadata(
    metadata_path: Path, tensor_path: Path
) -> Mapping[str, Any]:
    for path, label in ((metadata_path, "metadata"), (tensor_path, "mu_hat")):
        if path.is_symlink() or not path.is_file():
            raise UnconditionalBaselineError(
                f"unconditional baseline {label} is missing or unsafe: {path}"
            )
    try:
        return read_json(metadata_path)
    except CacheIOError as error:
        raise UnconditionalBaselineError(str(error)) from error


def _mu_hat_description(metadata: Mapping[str, object]) -> Mapping[str, object]:
    artifacts = metadata.get("coordinate_artifacts")
    description = artifacts.get("mu_hat") if isinstance(artifacts, Mapping) else None
    if not isinstance(description, Mapping):
        raise UnconditionalBaselineError("mu_hat artifact description is missing")
    return description


def _artifact_from_validated(
    contract: _SourceContract,
    metadata: Mapping[str, Any],
    metadata_path: Path,
    tensor_path: Path,
    mu_hat_sha256: str,
    mu_hat: torch.Tensor | None,
) -> UnconditionalBaselineArtifact:
    return UnconditionalBaselineArtifact(
        mu_hat=mu_hat,
        mu_hat_sha256=mu_hat_sha256,
        metadata=metadata,
        metadata_path=metadata_path,
        tensor_path=tensor_path,
        source_scientific_config_hash=contract.scientific_hash,
        source_schedule_sha256=contract.schedule_sha256,
        source_seed_start=contract.baseline_seeds[0],
        source_seeds=contract.baseline_seeds,
        num_baseline_seeds=len(contract.baseline_seeds),
        timestep=contract.timestep,
        alpha_t=contract.alpha_t,
        sigma_t=contract.sigma_t,
        latent_shape=contract.latent_shape,
    )


def _prepare_artifact_directory(path: Path) -> None:
    if path.is_symlink() or (path.exists() and not path.is_dir()):
        raise UnconditionalBaselineError(f"unsafe baseline directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    allowed = {BASELINE_TENSOR_NAME, BASELINE_METADATA_NAME}
    unexpected = sorted(
        item.name
        for item in path.iterdir()
        if item.name not in allowed or item.is_symlink() or not item.is_file()
    )
    if unexpected:
        raise UnconditionalBaselineError(
            "baseline directory contains unexpected artifacts: " + ", ".join(unexpected)
        )


def _schedule_tensor(
    schedule: Mapping[str, object], name: str, length: int, dtype: torch.dtype
) -> torch.Tensor:
    value = schedule.get(name)
    if (
        not isinstance(value, torch.Tensor)
        or tuple(value.shape) != (length,)
        or value.dtype != dtype
        or value.device.type != "cpu"
        or not value.is_contiguous()
        or value.requires_grad
        or (value.is_floating_point() and not bool(torch.isfinite(value).all()))
    ):
        raise UnconditionalBaselineError(
            f"saved schedule {name} violates its tensor contract"
        )
    return value


def _validate_tensor(
    value: object,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    label: str,
) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or tuple(value.shape) != shape
        or value.dtype != dtype
        or value.device.type != "cpu"
        or not value.is_contiguous()
        or value.requires_grad
        or not bool(torch.isfinite(value).all())
    ):
        raise UnconditionalBaselineError(f"{label} violates its tensor contract")
    return value


def _validate_mu_hat(value: object, shape: tuple[int, int, int]) -> torch.Tensor:
    return _validate_tensor(value, shape=shape, dtype=torch.float64, label="mu_hat")


def _tensor_value_sha256(value: torch.Tensor) -> str:
    cpu = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(cpu.dtype).encode("ascii"))
    digest.update(repr(tuple(cpu.shape)).encode("ascii"))
    digest.update(cpu.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _latent_shape(value: object) -> tuple[int, int, int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise UnconditionalBaselineError("latent shape must be [C, H, W]")
    result = tuple(value)
    if len(result) != 3 or any(
        isinstance(item, bool) or not isinstance(item, int) or item <= 0
        for item in result
    ):
        raise UnconditionalBaselineError("latent shape must be [C, H, W]")
    return result[0], result[1], result[2]


def _dtype_from_name(value: object) -> torch.dtype:
    dtypes = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    try:
        return dtypes[str(value)]
    except KeyError as error:
        raise UnconditionalBaselineError(
            f"unsupported generation tensor dtype: {value!r}"
        ) from error


def _dtype_name(value: torch.dtype) -> str:
    return str(value).removeprefix("torch.")


def _normalized_scheduler_config(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise UnconditionalBaselineError("scheduler configuration must be a mapping")
    result = {str(key): item for key, item in value.items()}
    defaults = result.get("_use_default_values")
    if isinstance(defaults, Sequence) and not isinstance(defaults, (str, bytes)):
        result["_use_default_values"] = sorted(str(item) for item in defaults)
    return result


def _positive_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise UnconditionalBaselineError(f"{label} must be a positive integer")
    return value


def _positive_finite_float(value: object, label: str) -> float:
    result = _finite_float(value, label)
    if result <= 0.0:
        raise UnconditionalBaselineError(f"{label} must be positive and finite")
    return result


def _nonnegative_finite_float(value: object, label: str) -> float:
    result = _finite_float(value, label)
    if result < 0.0:
        raise UnconditionalBaselineError(f"{label} must be nonnegative and finite")
    return result


def _finite_float(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise UnconditionalBaselineError(f"{label} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise UnconditionalBaselineError(f"{label} must be finite") from error
    if not math.isfinite(result):
        raise UnconditionalBaselineError(f"{label} must be finite")
    return result


def _nonnegative_integer(value: object, label: str) -> int:
    if isinstance(value, bool):
        raise UnconditionalBaselineError(f"{label} must be a nonnegative integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise UnconditionalBaselineError(
            f"{label} must be a nonnegative integer"
        ) from error
    if result != value or result < 0:
        raise UnconditionalBaselineError(f"{label} must be a nonnegative integer")
    return result


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError as error:
        raise UnconditionalBaselineError(f"path leaves project root: {path}") from error
