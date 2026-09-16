"""Resumable, shared learned probes for the seven direct manuscript comparisons.

Only the author-invoked analysis entry loads a checkpoint. Workers own disjoint
logical tasks and one denoiser each; the parent owns the sole progress bar and
aggregate publication. No scheduler update, image decoder, or outcome selection
is performed. All persisted observations are scalars, never posterior histories.
"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import hashlib
import math
from multiprocessing import get_context
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch

from utils.common.cli import MAX_SEED
from utils.common.io import (
    CacheIOError,
    atomic_torch_save,
    atomic_write_frame_parquet,
    atomic_write_json,
    canonical_hash,
    file_sha256,
    read_json,
    safe_torch_load,
)
from utils.models.devices import (
    configure_worker_cpu_threads,
    round_robin_shard,
    worker_count_for_tasks,
)
from utils.models.loading import package_version_metadata, select_runtime
from utils.models.prediction_conversion import alpha_sigma_for_timestep, scheduler_prediction_type
from utils.models.probe_loading import load_denoiser_components
from utils.models.sampling import (
    encode_prompt_condition,
    make_initial_noise,
    predict_conditional_epsilon,
    validate_latent_shape,
)
from utils.models.schedulers import build_scheduler_from_config
from .cache_reader import discover_sources, load_record, load_schedule
from .contracts import TheoryError
from .direct_probe_cache import resolve_completed_task
from .direct_probe_math import (
    FORMULA_VERSION,
    conditional_forward_metrics,
    conditional_gaussian_metrics,
    gaussian_reference_metrics,
    marginal_forward_metrics,
    pinsker_quantities,
    summarize_forward_losses,
    summarize_gaussian_conditional,
)
from .progress import RecordProgress, install_progress_queue, report_worker_record

SCHEMA_VERSION = 1
STREAM_DOMAIN = "direct-theory-probes-v1"
TASK_TABLES = ("forward_loss_draws", "gaussian_conditional", "gaussian_reference", "forward_unconditional_loss")
PROBE_POLICY = {
    "schema_version": SCHEMA_VERSION,
    "formula_version": FORMULA_VERSION,
    "stream_domain": STREAM_DOMAIN,
    "forward_noise": "independent_per_draw_CPU_float64_standard_normal",
    "gaussian_bank": "unique_evaluation_seed_CPU_float32_standard_normal_then_one_fixed_saved_precision_bank_at_all_native_steps; quantization_recorded",
    "loss_reduction": "float64_coordinate_sum_then_draw_mean",
    "conversion": "native_to_epsilon_once_against_unscaled_conceptual_input",
    "conditional_prediction": "exact_prompt_no_CFG",
    "batching": "bounded_halving_on_CUDA_OOM; logical_draws_never_advance_on_failure",
    "draw_count_change": "new_task_identity; deterministic_prefix_preserved_but_prefix_shard_reuse_not_implemented",
    "assumption": "single_target_conditional_training_law_is_assumed_not_empirically_established",
}


def derive_draw_seed(root_seed, *, checkpoint, pair, timestep, purpose, draw_index, forbidden=()):
    """Stable independent logical streams, unaffected by workers or microbatches."""
    if isinstance(root_seed, bool) or not isinstance(root_seed, int) or not 0 <= root_seed <= MAX_SEED:
        raise ValueError("loss_seed must be an integer in the supported seed domain")
    if isinstance(draw_index, bool) or not isinstance(draw_index, int) or draw_index < 0:
        raise ValueError("draw_index must be a nonnegative integer")
    if purpose not in {"forward_target", "forward_marginal_noise", "forward_marginal_atom"}:
        raise ValueError("Unknown probe random-stream domain")
    key = {
        "domain": STREAM_DOMAIN, "root": root_seed, "checkpoint": checkpoint,
        "pair": pair, "timestep": int(timestep), "purpose": purpose, "draw": draw_index,
    }
    excluded = set(forbidden)
    for counter in range(128):
        digest = canonical_hash(key | {"collision_counter": counter})
        # A disjoint high range avoids ordinary evaluation/selection seed blocks.
        seed = (int(digest[:16], 16) & ((1 << 62) - 1)) | (1 << 62)
        if seed not in excluded:
            return seed
    raise TheoryError("Cannot derive a disjoint probe seed")


def draw_noise(seeds, latent_shape):
    """Generate each draw on CPU independently, preserving order under batching."""
    return torch.stack([
        torch.randn(tuple(latent_shape), dtype=torch.float64, generator=torch.Generator(device="cpu").manual_seed(int(seed)))
        for seed in seeds
    ])


def _tensor_digest(tensor):
    value = torch.as_tensor(tensor).detach().to(device="cpu").contiguous()
    return canonical_hash({
        "dtype": str(value.dtype), "shape": list(value.shape),
        "bytes": hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest(),
    })


def _source_code():
    directory = Path(__file__).parent
    names = [directory / name for name in ("direct_probes.py", "direct_probe_math.py")]
    models = directory.parent.parent / "models"
    names += [models / name for name in ("probe_loading.py", "loading.py", "sampling.py", "prediction_conversion.py", "schedulers.py")]
    return {str(path.relative_to(directory.parent.parent)): file_sha256(path) for path in names}


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise TheoryError(f"{name} must be a positive integer")
    return value


def _pair_keys(sources, record):
    return {
        "run_id": sources.runs["experiment"]["scientific_config_hash"],
        "original_index": str(record.original_index),
        "record_id": str(record.metadata["record_id"]),
        "target_id": str(record.metadata["target_image_sha256"]),
        "target_latent_sha256": str(record.metadata["tensor_file_sha256"]["target_latent"]),
    }


def _record_stamp(sources, record):
    paths = sources.experiment
    files = [paths.target_latent_path(record.original_index), paths.latent_path(record.original_index), paths.noise_prediction_path(record.original_index)]
    return {
        "completion_identity": canonical_hash(record.metadata),
        "completion_marker_sha256": file_sha256(record.marker_path),
        "files": {str(p.relative_to(sources.root)): {"size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in files},
    }


def _safe_probe_path(path):
    path = Path(path).absolute()
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise TheoryError(f"Probe outputs must not traverse a symbolic link: {path}")
    return path


def _task_paths(directory, task):
    base = _safe_probe_path(Path(directory) / "tasks" / task["table"] / task["task_hash"])
    return base.with_suffix(".parquet"), base.with_suffix(".json")


def _completed_task(directory, task):
    return resolve_completed_task(directory, task, path_for_task=_task_paths)


def _valid_task(directory, task):
    return _completed_task(directory, task) is not None


def plan_probe_tasks(sources, records, law, schedule, config):
    """Make independent identities before device scheduling; no model is loaded."""
    records = sorted(list(records), key=lambda record: str(record.original_index))
    count = _positive_integer(config.get("num_loss_seeds", 64), "num_loss_seeds")
    marginal_count = _positive_integer(config.get("num_unconditional_loss_seeds", 256), "num_unconditional_loss_seeds")
    mode = config.get("loss_timesteps", "initial")
    if mode not in {"initial", "saved"}:
        raise TheoryError("loss_timesteps must be initial or saved")
    marginal = config.get("measure_unconditional_loss", mode == "saved")
    if not isinstance(marginal, bool):
        raise TheoryError("measure_unconditional_loss must be a boolean")
    science = sources.runs["experiment"]["scientific_config"]
    inference_dtype = science.get("inference_dtype")
    if inference_dtype not in {"float16", "bfloat16", "float32"}:
        raise TheoryError("Preserved checkpoint inference dtype is missing or unsupported")
    if science.get("stored_prediction_type") != "epsilon":
        raise TheoryError("Probe cache reuse requires already canonical epsilon")
    checkpoint = {k: science[k] for k in ("model_id", "model_revision", "native_prediction_type", "inference_dtype")}
    runtime_packages = package_version_metadata()
    shared = {
        "runtime_packages": {key: runtime_packages[key] for key in ("torch", "diffusers", "transformers")},
        "policy": PROBE_POLICY, "source_code": _source_code(), "checkpoint": checkpoint,
        "schedule_sha256": file_sha256(sources.experiment.schedule),
        "latent_shape": science["latent_shape"],
        "target_preprocessing": science.get("target_preprocessing"),
        "vae_id": science.get("vae_id"), "vae_revision": science.get("vae_revision"),
    }
    levels = []
    for k, native in enumerate(schedule["timesteps"].tolist()):
        cumulative = float(schedule["alphas_cumprod_t"][k])
        if not 0 < cumulative < 1:
            continue
        levels.append({"step_index": k, "timestep": int(native), "alpha": math.sqrt(cumulative), "sigma": math.sqrt(1 - cumulative), "snr": cumulative / (1 - cumulative)})
    if not levels or levels[0]["step_index"] != 0:
        raise TheoryError("Actual initial native timestep is unavailable for learned probes")
    measured = levels if mode == "saved" else levels[:1]
    tasks, seen_streams = [], {}
    forbidden = tuple(science["seeds"]) + tuple(sources.runs["reference"]["scientific_config"]["seeds"])
    root_seed = config.get("loss_seed", 0)

    def append(table, level, record, draws):
        pair = _pair_keys(sources, record) if record is not None else {"run_id": sources.runs["experiment"]["scientific_config_hash"]}
        identity = shared | {"table": table, "level": level, "pair": pair, "count": draws}
        task = {"table": table, "level": level, "record_index": str(record.original_index) if record is not None else None, "pair": pair, "count": draws}
        if record is not None:
            identity["prompt_sha256"] = hashlib.sha256(record.metadata["prompt_raw"].encode("utf-8")).hexdigest()
            identity["source_record"] = _record_stamp(sources, record)
        if table in {"gaussian_reference", "forward_unconditional_loss"}:
            identity["reference_law_hash"] = law.law_hash
            identity["reference_sources"] = {name: file_sha256(Path(__file__).parent / name) for name in ("reference_law.py", "support.py")}
        if table.startswith("gaussian"):
            identity["gaussian_seeds"] = list(science["seeds"])
            identity["initial_cache_source"] = _record_stamp(sources, record or records[0]) if level["step_index"] == 0 else None
            if table == "gaussian_conditional" and level["step_index"] == 0:
                paired = sources.proximity.loc[sources.proximity.original_index.eq(record.original_index), ["seed", "sscd"]].sort_values("seed")
                identity["paired_initial_outcomes"] = paired.to_dict("records")
        else:
            identity["loss_seed"] = root_seed
            purposes = ["forward_target"] if table == "forward_loss_draws" else ["forward_marginal_noise", "forward_marginal_atom"]
            for purpose in purposes:
                seeds = [derive_draw_seed(root_seed, checkpoint=checkpoint, pair=pair, timestep=level["timestep"], purpose=purpose, draw_index=i, forbidden=forbidden) for i in range(draws)]
                for i, seed in enumerate(seeds):
                    owner = (tuple(sorted(pair.items())), level["timestep"], purpose, i)
                    if seed in seen_streams and seen_streams[seed] != owner:
                        raise TheoryError("Probe RNG seed collision across independent domains")
                    seen_streams[seed] = owner
                task["atom_seeds" if purpose.endswith("atom") else "noise_seeds"] = seeds
            identity["stream_fingerprint"] = canonical_hash({key: value for key, value in task.items() if key.endswith("seeds")})
        task["identity"] = identity
        task["task_hash"] = canonical_hash(identity)
        tasks.append(task)

    for record in records:
        for level in measured:
            append("forward_loss_draws", level, record, count)
            append("gaussian_conditional", level, record, len(science["seeds"]))
    for level in levels:
        append("gaussian_reference", level, None, len(science["seeds"]))
    if marginal:
        for level in measured:
            append("forward_unconditional_loss", level, None, marginal_count)
    return tasks


def _load_replica(science, schedule, device):
    runtime = select_runtime(device, warn_without_cuda=False)
    runtime = replace(runtime, inference_dtype=getattr(torch, science["inference_dtype"]))
    components = load_denoiser_components(model_id=science["model_id"], model_revision=science["model_revision"], runtime=runtime)
    validate_latent_shape(components.unet, science["latent_shape"])
    scheduler = build_scheduler_from_config(schedule["scheduler_config"], schedule["scheduler_name"]).scheduler
    scheduler.set_timesteps(len(schedule["timesteps"]), device=device)
    if type(scheduler).__name__ != schedule["scheduler_class"]:
        raise TheoryError("Active probe scheduler class differs from the saved schedule")
    if not torch.equal(torch.as_tensor(scheduler.timesteps).cpu().long(), schedule["timesteps"].cpu().long()):
        raise TheoryError("Active probe native grid differs from the saved schedule")
    if scheduler_prediction_type(scheduler) != science["native_prediction_type"]:
        raise TheoryError("Active native prediction type differs from the preserved checkpoint")
    if float(scheduler.init_noise_sigma) != float(schedule["init_noise_sigma"]):
        raise TheoryError("Active initialization scale differs from the preserved checkpoint")
    for k, timestep in enumerate(schedule["timesteps"]):
        a, s = alpha_sigma_for_timestep(scheduler, timestep, dtype=torch.float64)
        if float(a.float()) != float(schedule["alpha_t"][k]) or float(s.float()) != float(schedule["sigma_t"][k]):
            raise TheoryError(f"Active probe noise scales differ at saved step {k}")
    return components, scheduler


def _prediction_microbatch(samples, *, condition, timestep, components, scheduler):
    device = torch.device(components.device)
    conversion_dtype = torch.float32 if device.type == "mps" else torch.float64
    conceptual = samples.to(device=device, dtype=conversion_dtype)
    model_input = conceptual.to(dtype=components.inference_dtype)
    with torch.inference_mode():
        prediction = predict_conditional_epsilon(model_input, timestep, condition, components.unet, scheduler, conversion_sample=conceptual)
    prediction = prediction.detach().to(dtype=torch.float64)
    if prediction.shape != samples.shape or not bool(torch.isfinite(prediction).all()):
        raise TheoryError("Native probe prediction has invalid shape or values")
    return prediction, {
        "input_quantization_l2": (model_input.detach().double() - samples.to(device=device, dtype=torch.float64)).flatten(1).norm(dim=1),
        "conversion_input_quantization_l2": (conceptual.detach().double() - samples.to(device=device, dtype=torch.float64)).flatten(1).norm(dim=1),
        "prediction_conversion_dtype": str(conversion_dtype).removeprefix("torch."),
        "inference_dtype": str(components.inference_dtype).removeprefix("torch."),
    }


def prediction_batches(samples, *, condition, timestep, components, scheduler, batch_size):
    """Yield successful slices; retry only failed CUDA microbatches at most 32 times."""
    size = _positive_integer(batch_size, "probe_batch_size")
    position, retries = 0, 0
    device = torch.device(components.device)
    while position < len(samples):
        stop = min(len(samples), position + size)
        retry = False
        try:
            prediction, precision = _prediction_microbatch(
                samples[position:stop], condition=condition, timestep=timestep,
                components=components, scheduler=scheduler,
            )
        except RuntimeError as error:
            is_oom = device.type == "cuda" and (isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower())
            if not is_oom or stop - position <= 1 or retries >= 32:
                raise
            size = max(1, (stop - position) // 2)
            retries += 1
            retry = True
        if retry:
            # The exception traceback and failed frame have been released here.
            torch.cuda.empty_cache()
            continue
        yield position, stop, prediction, precision | {"oom_retries": retries}
        position = stop



def cached_prediction_batches(samples, cached_epsilon, gaussian, *, batch_size):
    """Slice one immutable saved bank independently of the consumer's variables.

    In particular, a consumer may bind each yielded tensor to ``epsilon``;
    that must never replace the full bank used to construct the next slice.
    """
    size = _positive_integer(batch_size, "probe_batch_size")
    if samples.ndim < 2 or samples.shape != cached_epsilon.shape or samples.shape != gaussian.shape:
        raise TheoryError("Cached Gaussian states, predictions and seed bank must have identical batch/latent shapes")
    for start in range(0, len(samples), size):
        stop = min(len(samples), start + size)
        yield start, stop, cached_epsilon[start:stop].double(), {
            "input_quantization_l2": (samples[start:stop] - gaussian[start:stop].double()).flatten(1).norm(dim=1),
            "conversion_input_quantization_l2": torch.zeros(stop - start, dtype=torch.float64, device=samples.device),
            "prediction_conversion_dtype": str(cached_epsilon.dtype).removeprefix("torch."),
            "oom_retries": 0,
        }


def _rows(values, count, metadata):
    rows = [dict(metadata) for _ in range(count)]
    for key, value in values.items():
        if isinstance(value, torch.Tensor):
            items = value.detach().cpu().reshape(-1).tolist()
            if len(items) != count:
                raise TheoryError(f"Nonscalar probe field: {key}")
            for row, item in zip(rows, items, strict=True):
                row[key] = item
        else:
            for row in rows:
                row[key] = value
    return rows


def _cached_initial(sources, record, gaussian, schedule):
    z, u, c, target = load_record(sources.experiment, record, initial_only=True, verify_hashes=False)
    z, u, c, target = (value.to(device=gaussian.device) for value in (z, u, c, target))
    expected = gaussian.to(dtype=z.dtype) * float(schedule["init_noise_sigma"])
    exact = z.shape == expected.shape and torch.equal(z, expected)
    compatible = exact and float(schedule["init_noise_sigma"]) == 1.0
    status = "matched_saved_standard_gaussian_with_recorded_quantization" if compatible else "saved_initialization_not_theorem_gaussian; fresh_probe_required"
    return {"state": z, "unconditional": u, "conditional": c, "target": target, "compatible": compatible, "status": status}


def _persist_worker_inputs(directory, sources, records, law):
    """Share large immutable inputs through validated files, never tensor IPC."""
    support = law.support
    law_identity = {
        "law_hash": law.law_hash,
        "atoms": _tensor_digest(support.atoms),
        "weights": _tensor_digest(support.weights),
        "aliases": support.aliases,
        "atom_ids": support.atom_ids,
    }
    location = _safe_probe_path(Path(directory) / "shared_inputs" / canonical_hash(law_identity))
    law_path, law_marker = location / "law.pt", location / "law.json"
    valid = False
    if law_path.is_file() and law_marker.is_file() and not law_path.is_symlink() and not law_marker.is_symlink():
        try:
            prior = read_json(law_marker)
            valid = prior.get("identity") == law_identity and prior.get("sha256") == file_sha256(law_path)
        except (CacheIOError, OSError, ValueError):
            pass
    if not valid:
        digest = atomic_torch_save(law.to_payload(), law_path)
        atomic_write_json(law_marker, {
            "identity": law_identity, "sha256": digest, "metadata": law.metadata,
            "candidate_chunk": support.candidate_chunk, "query_chunk": support.query_chunk,
        })
    source = {
        "root": str(sources.root), "config": sources.config, "runs": sources.runs,
        "experiment_directory": str(sources.experiment.run_directory),
        "records": [{"original_index": str(r.original_index), "source_row_number": r.source_row_number,
                     "marker_path": str(r.marker_path), "metadata": dict(r.metadata)} for r in records],
        "proximity": sources.proximity[["original_index", "seed", "sscd"]].to_dict("records"),
        "schedule_path": str(sources.experiment.schedule),
        "schedule_sha256": file_sha256(sources.experiment.schedule),
    }
    source_path = _safe_probe_path(Path(directory) / "shared_inputs" / (canonical_hash(source) + ".json"))
    try:
        source_valid = source_path.is_file() and read_json(source_path) == source
    except (CacheIOError, OSError, ValueError):
        source_valid = False
    if not source_valid:
        atomic_write_json(source_path, source)
    return {"law_marker": str(law_marker), "law_marker_sha256": file_sha256(law_marker),
            "source_path": str(source_path), "source_sha256": file_sha256(source_path)}


def _read_worker_inputs(specification, *, device="cpu"):
    from utils.experiments.cache import CompletedGenerationRecord, GenerationPaths
    from .reference_law import ReferenceLaw
    from .support import FiniteSupport

    for name in ("law_marker", "source_path"):
        path = Path(specification[name])
        digest_key = "law_marker_sha256" if name == "law_marker" else "source_sha256"
        if path.is_symlink() or file_sha256(path) != specification[digest_key]:
            raise TheoryError("Changed or unsafe learned-probe worker input")
    saved = read_json(specification["law_marker"])
    tensor_path = Path(specification["law_marker"]).parent / "law.pt"
    if tensor_path.is_symlink() or file_sha256(tensor_path) != saved["sha256"]:
        raise TheoryError("Changed learned-probe reference atoms")
    tensors = safe_torch_load(tensor_path)
    identity = saved["identity"]
    if _tensor_digest(tensors["atoms"]) != identity["atoms"] or _tensor_digest(tensors["weights"]) != identity["weights"]:
        raise TheoryError("Learned-probe reference-law fingerprint differs")
    support = FiniteSupport(tensors["atoms"].to(device), identity["atom_ids"], identity["aliases"], weights=tensors["weights"].to(device),
                            candidate_chunk=saved["candidate_chunk"], query_chunk=saved["query_chunk"])
    # Preserve exact declared masses, without a second floating normalization.
    support.weights = tensors["weights"].to(device).clone()
    support.log_weights = support.weights.log()
    law = ReferenceLaw(support, saved["metadata"], identity["law_hash"], tensors.get("estimated_mean_vector"))
    source = read_json(specification["source_path"])
    records = [CompletedGenerationRecord(r["original_index"], r["source_row_number"], Path(r["marker_path"]), r["metadata"]) for r in source["records"]]
    sources = SimpleNamespace(root=Path(source["root"]), config=source["config"], runs=source["runs"],
                              experiment=GenerationPaths(Path(source["experiment_directory"])),
                              selected=records, proximity=pd.DataFrame(source["proximity"]))
    path = Path(source["schedule_path"])
    if path.is_symlink() or file_sha256(path) != source["schedule_sha256"]:
        raise TheoryError("Preserved probe scheduler changed")
    return sources, law, safe_torch_load(path)


def _run_probe_worker(*, inputs, tasks, output_directory, device, batch_size, worker_count):
    """Spawn entry: lazy single replica, exact-prompt embedding cache, scalar tasks."""
    configure_worker_cpu_threads(worker_count)
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    sources, law, schedule = _read_worker_inputs(inputs, device=device)
    host_atom_weights = law.support.weights.detach().cpu()
    science = sources.runs["experiment"]["scientific_config"]
    if any(task["identity"]["source_code"] != _source_code() for task in tasks[:1]):
        raise TheoryError("Learned-probe numerical source changed after planning")
    for task in tasks:
        if "reference_sources" in task["identity"]:
            if any(file_sha256(Path(__file__).parent / name) != digest for name, digest in task["identity"]["reference_sources"].items()):
                raise TheoryError("Reference posterior implementation changed after planning")
            break
    records = {str(r.original_index): r for r in sources.selected}
    first = records[sorted(records)[0]]
    # Preserve the generation RNG stream exactly; only seeded input construction
    # runs on the host. All latent-vector arithmetic follows on the worker GPU.
    gaussian = make_initial_noise(science["seeds"], science["latent_shape"]).to(device)
    storage_dtype = getattr(torch, science["scientific_tensor_storage"]["dtype"])
    gaussian_bank = gaussian.to(dtype=storage_dtype).double() if float(schedule["init_noise_sigma"]) == 1.0 else gaussian.double()
    components = scheduler = None
    embeddings, initial, targets = {}, {}, {}
    failures, completed = [], []
    replica_metadata = {}

    def cached(record):
        key = str(record.original_index)
        if key not in initial:
            initial[key] = _cached_initial(sources, record, gaussian, schedule)
        return initial[key]

    def prediction(samples, prompt, timestep):
        nonlocal components, scheduler, replica_metadata
        if components is None:
            components, scheduler = _load_replica(science, schedule, device)
            replica_metadata = components.device_metadata | {"package_versions": components.package_versions}
        if prompt not in embeddings:
            embeddings[prompt] = encode_prompt_condition(prompt, components.tokenizer, components.text_encoder, device, components.inference_dtype)
        yield from prediction_batches(samples, condition=embeddings[prompt], timestep=timestep, components=components, scheduler=scheduler, batch_size=batch_size)

    try:
        for task in tasks:
            try:
                if _valid_task(output_directory, task):
                    report_worker_record(task["task_hash"], "resumed", device)
                    continue
                table, level = task["table"], task["level"]
                record = records.get(task["record_index"])
                target = None
                if record is not None:
                    key = str(record.original_index)
                    if key not in targets:
                        path = sources.experiment.target_latent_path(record.original_index)
                        if file_sha256(path) != record.metadata["tensor_file_sha256"]["target_latent"]:
                            raise TheoryError("Preserved forward-loss target hash differs")
                        targets[key] = safe_torch_load(path).to(device=device, dtype=torch.float64)
                    target = targets[key]
                metadata = task["pair"] | level | {
                    "task_hash": task["task_hash"], "status": "measured",
                    "input_source": "forward_target" if table == "forward_loss_draws" else "forward_marginal" if table == "forward_unconditional_loss" else "gaussian_probe",
                    "reference_law_hash": law.law_hash if table in {"gaussian_reference", "forward_unconditional_loss"} else "fixed_pair_target",
                    "reference_law_scope": str(law.metadata.get("reference_law_scope", "declared_finite_law")) if table in {"gaussian_reference", "forward_unconditional_loss"} else "fixed_pair_target",
                    "model_revision": science["model_revision"],
                    "native_prediction_type": science["native_prediction_type"],
                    "inference_dtype": science["inference_dtype"],
                }
                noise = atom_indices = None
                cache = None
                if table.startswith("gaussian"):
                    samples = gaussian_bank
                    if level["step_index"] == 0:
                        cache = cached(record or first)
                        if cache["compatible"]:
                            samples = cache["state"].double()
                    metadata["initialization_status"] = cache["status"] if cache else "fresh_standard_gaussian_at_saved_native_noise"
                    metadata["gaussian_bank_identity"] = _tensor_digest(gaussian)
                else:
                    noise = draw_noise(task["noise_seeds"], science["latent_shape"]).to(device)
                    if table == "forward_loss_draws":
                        samples = level["alpha"] * target.unsqueeze(0) + level["sigma"] * noise
                        metadata.update(pinsker_quantities(target, level["alpha"], level["sigma"]))
                    else:
                        # CPU categorical draws retain the declared seeded atom
                        # stream; sampled atoms and corruption stay on the GPU.
                        atom_indices = torch.tensor([int(torch.multinomial(host_atom_weights, 1, generator=torch.Generator().manual_seed(seed))) for seed in task["atom_seeds"]], device=device)
                        samples = level["alpha"] * law.support.atoms[atom_indices] + level["sigma"] * noise
                prompt = record.metadata["prompt_raw"] if record is not None else ""
                if cache and cache["compatible"]:
                    cached_epsilon = cache["conditional"] if record is not None else cache["unconditional"]
                    prediction_source = "saved_initial_canonical_epsilon; not_converted_again"
                    batches = cached_prediction_batches(samples, cached_epsilon, gaussian, batch_size=batch_size)
                    metadata["cached_prediction_sha256"] = _tensor_digest(cached_epsilon)
                else:
                    prediction_source = "fresh_checkpoint_one_prediction_no_scheduler_update"
                    batches = prediction(samples, prompt, level["timestep"])
                metadata["prediction_source"] = prediction_source
                output_rows = []
                for start, stop, epsilon, precision in batches:
                    z = samples[start:stop]
                    if not 0 <= start < stop <= len(samples) or epsilon.shape != z.shape:
                        raise TheoryError(f"Probe batch does not match its logical seed slice [{start}:{stop}]")
                    a, s = level["alpha"], level["sigma"]
                    if table == "forward_loss_draws":
                        values = conditional_forward_metrics(target, noise[start:stop], z, epsilon, a, s)
                    elif table == "gaussian_conditional":
                        values = conditional_gaussian_metrics(target, z, epsilon, a, s)
                    else:
                        posterior = law.posterior_mean(z, a, s)
                        values = gaussian_reference_metrics(z, epsilon, posterior, law.theory_mean_vector, a, s, law.max_atom_norm, bank_mean=law.mean_vector) if table == "gaussian_reference" else marginal_forward_metrics(noise[start:stop], z, epsilon, posterior, a, s)
                        if table == "gaussian_reference":
                            mean_receipt = law.theory_mean_metadata
                            values["theory_mean_source"] = str(mean_receipt.get("source", "declared_finite_bank_mean"))
                            values["theory_mean_sha256"] = str(mean_receipt.get("vector_sha256", law.metadata.get("mean_sha256", "")))
                    rows = _rows(values | precision, stop - start, metadata)
                    for i, row in enumerate(rows, start):
                        row["input_sha256"] = _tensor_digest(samples[i])
                        if table.startswith("gaussian"):
                            seed = int(science["seeds"][i])
                            row["seed"] = seed
                            row["terminal_sscd"] = math.nan
                            row["terminal_sscd_matched_initialization"] = False
                            row["gaussian_bank_quantization_l2"] = float((samples[i] - gaussian[i].double()).norm())
                            row["same_seed_sscd_status"] = "not_a_saved_generated_trajectory"
                            if record is not None and cache and cache["compatible"]:
                                score = sources.proximity.loc[sources.proximity.original_index.eq(record.original_index) & sources.proximity.seed.eq(seed), "sscd"]
                                if len(score) != 1:
                                    raise TheoryError("Missing exact initial-seed SSCD identity")
                                row["terminal_sscd"] = float(score.iloc[0])
                                row["same_seed_sscd_status"] = "exact_saved_initial_seed"
                                row["terminal_sscd_matched_initialization"] = True
                        else:
                            row["draw_index"] = i
                            row["noise_seed"] = task["noise_seeds"][i]
                            row["noise_sha256"] = _tensor_digest(noise[i])
                            if atom_indices is not None:
                                row["atom_seed"] = task["atom_seeds"][i]
                                row["sampled_atom_id"] = str(law.support.atom_ids[int(atom_indices[i])])
                        output_rows.append(row)
                frame = pd.DataFrame(output_rows)
                if len(frame) != task["count"]:
                    raise TheoryError("Probe task did not preserve every logical observation")
                if record is not None and _record_stamp(sources, record) != task["identity"]["source_record"]:
                    raise TheoryError("Protected source record changed during probe inference")
                initial_source = task["identity"].get("initial_cache_source")
                if initial_source is not None and _record_stamp(sources, record or first) != initial_source:
                    raise TheoryError("Protected Gaussian initialization source changed during probes")
                data, marker = _task_paths(output_directory, task)
                atomic_write_frame_parquet(frame, data)
                atomic_write_json(marker, {"schema_version": SCHEMA_VERSION, "task_hash": task["task_hash"], "identity": task["identity"], "complete": True, "rows": len(frame), "sha256": file_sha256(data), "execution": {"device": str(device), "numeric_backend": "worker_device_float64", "random_stream_backend": "preserved_CPU_seeded_input_draws", "batch_size_requested": batch_size, "packages": components.package_versions if components else science.get("package_versions", {}), "cached_only": components is None, "replica": replica_metadata}})
                initial.clear()
                targets.clear()
                completed.append(task["task_hash"])
                report_worker_record(task["task_hash"], "reduced", device)
            except Exception as error:
                failures.append(task["pair"] | task["level"] | {"task_hash": task["task_hash"], "table": task["table"], "error": f"{type(error).__name__}: {error}", "status": "failed"})
                report_worker_record(task["task_hash"], "failed", device)
    finally:
        embeddings.clear()
        initial.clear()
        targets.clear()
        components = scheduler = None
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return {"completed": completed, "failures": failures, "device": str(device), "replica": replica_metadata}


def run_direct_probes(project_root, *, records, support, config, output_directory, device="auto", probe_batch_size=8, sources=None):
    """Collect missing learned observations, returning scalar tables and provenance.

    ``support`` is the ReferenceLaw adapter. ``output_directory`` is a stable
    backing directory below theory_measurements, independent of paper style and
    integration identities. A caller must await this stage before analytical GPU
    workers start. Full protected payload verification belongs to that shared pass.
    """
    from .paper_contracts import publication_lock
    from .reduce import _resolve_theory_devices

    sources = sources or discover_sources(project_root, **config)
    records = sorted(list(records), key=lambda record: str(record.original_index))
    if not records or {str(r.original_index) for r in records} != {str(r.original_index) for r in sources.selected}:
        raise TheoryError("Direct probes require all and only frozen retained records")
    if len(records) != len({str(r.original_index) for r in records}):
        raise TheoryError("Duplicate retained probe record")
    batch_size = _positive_integer(probe_batch_size, "probe_batch_size")
    schedule = load_schedule(sources)
    tasks = plan_probe_tasks(sources, records, support, schedule, config)
    directory = Path(output_directory).absolute()
    root_outputs = Path(project_root).absolute() / "outputs"
    if not directory.is_relative_to(root_outputs) or "theory_measurements" not in directory.relative_to(root_outputs).parts:
        raise TheoryError("Learned probe shards must stay in the theory_measurements backing hierarchy")
    with publication_lock(directory):
        pending = [task for task in tasks if not _valid_task(directory, task)]
        inputs = _persist_worker_inputs(directory, sources, records, support) if pending else None
        devices = tuple(_resolve_theory_devices(device)) if pending else ()
        workers = worker_count_for_tasks(devices, len(pending)) if pending else 0
        context = get_context("spawn")
        failures = []
        execution = []
        with RecordProgress(total=len(tasks), devices=workers, context=context) as progress:
            pending_ids = {task["task_hash"] for task in pending}
            for task in tasks:
                if task["task_hash"] not in pending_ids:
                    progress.report(task["task_hash"], "resumed", "saved")
            if workers:
                assigned = [[] for _ in range(workers)]
                owners = {str(record.original_index): i % workers for i, record in enumerate(records)}
                shared = [task for task in pending if task["record_index"] is None]
                for task in pending:
                    if task["record_index"] is not None:
                        assigned[owners[task["record_index"]]].append(task)
                for i in range(workers):
                    assigned[i].extend(round_robin_shard(shared, worker_index=i, worker_count=workers))
                with ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=install_progress_queue, initargs=(progress.queue,)) as executor:
                    futures = [(i, executor.submit(_run_probe_worker, inputs=inputs, tasks=shard, output_directory=directory, device=str(devices[i]), batch_size=batch_size, worker_count=workers)) for i, shard in enumerate(assigned) if shard]
                    for i, future in futures:
                        try:
                            result = future.result()
                            failures.extend(result["failures"])
                            execution.append(result)
                        except Exception as error:
                            for task in assigned[i]:
                                if not _valid_task(directory, task):
                                    failures.append(task["pair"] | task["level"] | {"task_hash": task["task_hash"], "table": task["table"], "status": "failed", "error": f"worker exited: {type(error).__name__}: {error}"})
                                    progress.report(task["task_hash"], "failed", devices[i])
        completed_tasks = {task["task_hash"]: _completed_task(directory, task) for task in tasks}
        incomplete = [task for task in tasks if completed_tasks[task["task_hash"]] is None]
        if tasks[0]["identity"]["source_code"] != _source_code():
            raise TheoryError("Learned-probe numerical sources changed before publication")
        for task in tasks:
            if "reference_sources" in task["identity"]:
                if any(file_sha256(Path(__file__).parent / name) != digest for name, digest in task["identity"]["reference_sources"].items()):
                    raise TheoryError("Reference posterior sources changed before probe publication")
                break
        task_sources = {
            planned: {"source_task_hash": saved[0], "path": str(saved[1]),
                      "marker_path": str(saved[2]), "marker_sha256": file_sha256(saved[2]),
                      "compatible_predecessor": saved[0] != planned}
            for planned, saved in completed_tasks.items() if saved is not None
        }
        aggregate_identity = {"schema_version": SCHEMA_VERSION,
                              "task_hashes": sorted(task["task_hash"] for task in tasks),
                              "source_task_hashes": {key: value["source_task_hash"] for key, value in sorted(task_sources.items())}}

        fingerprint = canonical_hash(aggregate_identity)
        aggregate = _safe_probe_path(directory / "collections" / fingerprint)
        atomic_write_frame_parquet(pd.DataFrame(failures, columns=None if failures else ["task_hash", "table", "status", "error"]), aggregate / "failed.parquet")
        provenance = {
            "schema_version": SCHEMA_VERSION, "collection_hash": fingerprint,
            "identity": aggregate_identity, "policy": PROBE_POLICY,
            "complete": not incomplete, "task_count": len(tasks), "failed_task_count": len(incomplete),
            "reference_law_hash": support.law_hash, "reference_law": support.metadata,
            "requested_device": str(device), "resolved_devices": [str(d) for d in devices],
            "workers": execution,
            "task_sources": task_sources,
            "task_source_policy": "Rows retain their original task_hash; audited pre-fix completed shards are read unchanged and explicitly mapped to current planned tasks.",
            "native_grid": [{"step_index": k, "timestep": int(timestep),
                             "status": "valid_positive_native_noise" if 0 < float(schedule["alphas_cumprod_t"][k]) < 1 else "not_applicable_nonpositive_native_signal_or_noise"}
                            for k, timestep in enumerate(schedule["timesteps"].tolist())],
            "initial_cache_validation": "completion_identity_and_file_stamps_plus_exact_initial_noise; whole_payload_hashes_owned_by_shared_trajectory_pass",
            "input_precision": "Gaussian draws use generation float32 RNG and one fixed saved-precision bank across native steps; forward draws/corruption float64; quantization and conversion precision recorded per row",
        }
        if incomplete:
            atomic_write_json(aggregate / "manifest.json", provenance)
            causes = Counter(failure["error"] for failure in failures)
            details = "; ".join(f"{count} tasks: {error}" for error, count in causes.most_common(3))
            raise TheoryError(
                f"{len(incomplete)}/{len(tasks)} direct learned-probe tasks failed. "
                f"{details or 'Required task completion is missing'}. "
                f"Failure details: {aggregate / 'failed.parquet'}. Completed shards are retained at {directory}"
            )
        tables = {}
        files = {}
        for table in TASK_TABLES:
            parts = [pd.read_parquet(completed_tasks[task["task_hash"]][1]) for task in tasks if task["table"] == table]
            tables[table] = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        tables["forward_loss_summary"] = summarize_forward_losses(tables["forward_loss_draws"])
        tables["gaussian_conditional_summary"] = summarize_gaussian_conditional(tables["gaussian_conditional"])
        for name, frame in tables.items():
            path = aggregate / (name + ".parquet")
            atomic_write_frame_parquet(frame, path)
            files[name] = {"path": str(path), "sha256": file_sha256(path), "rows": len(frame)}
        provenance["files"] = files
        atomic_write_json(aggregate / "manifest.json", provenance)
        return {"tables": tables, "provenance": provenance, "files": files}
