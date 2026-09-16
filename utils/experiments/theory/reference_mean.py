"""Independent posterior-reference mean with resumable CUDA moments.

One Gaussian bank is evaluated under the unchanged finite cached reference law
at the analytical sweep's minimum SNR by default. The original initial native
level remains an explicit historical mode. Only the posterior is evaluated;
no learned network, image decoder or diffusion trajectory is invoked.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
import inspect
import math

import torch

from utils.common.io import (CacheIOError, atomic_torch_save, atomic_write_json,
                             canonical_hash, file_sha256, read_json, safe_torch_load)
from utils.models.devices import configure_worker_cpu_threads, round_robin_shard, worker_count_for_tasks
from .contracts import TheoryError
from .evidence_reference import REFERENCE_GRID_POINTS, REFERENCE_SWEEP_VERSION, reference_grid
from .paper_contracts import PaperPaths, contained_path, publication_lock, recompute_command, reject_symlinks
from .progress import RecordProgress, install_progress_queue, report_worker_record
from .support import FiniteSupport, _tensor_hash

SCHEMA_VERSION = 2
LOGICAL_SHARD_SIZE = 128
SUM_FIELDS = ("sum", "sum_squares", "even_sum", "odd_sum")
MOMENT_FIELDS = (*SUM_FIELDS, "mean", "centered_sum_squares")
POLICY = {
    "version": "independent-selected-snr-reference-mean-cuda-2",
    "target": "E_Z[bar_x(Z,empty; selected_SNR)] under unchanged cached atom law; Z~N(0,I)",
    "interpretation": "finite_selected_noise_posterior_mean_average; not_identical_to_prior_atom_mean_at_finite_SNR",
    "noise": "independent_per_draw_CUDA_float64_standard_normal; no_saved_bank_quantization",
    "seed_domain": "reference_mean_2^61_to_2^62_v1; disjoint_from_direct_forward_seeds_and_declared_evaluation_seeds",
    "native_level": "first_saved_positive_signal_and_noise_label_at_step_index_zero",
    "default_estimation_level": "reference_grid(initial_native_SNR,reference_snr_decades)[0]",
    "conditional_law": "unconditional_declared_finite_reference_distribution",
    "posterior": "unchanged_atom_weights_times_Gaussian_likelihood; raw_squared_L2_exponent",
    "moments": "fixed_logical_shard_float64_CUDA_sum_sum_squares_and_centered_M2",
    "aggregation": "ascending_logical_shard_order_on_one_CUDA_device",
    "uncertainty": "fixed_order_Chan_merge_of_centered_M2; unbiased_coordinate_sample_variance_and_Monte_Carlo_standard_error_only",
    "split": "even_versus_odd_logical_draw_indices",
    "logical_shard_size": LOGICAL_SHARD_SIZE,
}

MEAN_SOURCES = {
    "reference-min-snr": "minimum_snr_unconditional_reference_monte_carlo",
    "reference-initial": "initial_unconditional_reference_monte_carlo",
}
LEVEL_RECEIPT_FIELDS = ("mean_source", "source", "level", "initial_level", "initial_snr",
                        "estimation_snr", "reference_snr_decades", "reference_grid_definition")


def _integer(value, name, *, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TheoryError(f"{name} must be an integer >= {minimum}")
    return value


def _cuda_device(device):
    device = torch.device(device)
    if device.type != "cuda" or getattr(torch.version, "hip", None):
        raise TheoryError("The independent reference-mean estimator requires CUDA; CPU/MPS fallback is disabled")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return device


def _source_code():
    base = Path(__file__).parent
    paths = [base / name for name in ("reference_mean.py", "support.py")]
    paths += [base.parent.parent / "models" / "devices.py"]
    result = {str(path.relative_to(base.parent.parent)): file_sha256(path) for path in paths}
    # Pin the shared grid definition without invalidating this independent bank
    # for unrelated changes to reference-curve scalar fields in the same file.
    result["experiments/theory/evidence_reference.py:reference_grid"] = canonical_hash({
        "source": inspect.getsource(reference_grid), "points": REFERENCE_GRID_POINTS,
        "revision": REFERENCE_SWEEP_VERSION})
    return result


def _runtime():
    return {"torch": torch.__version__, "torch_cuda": torch.version.cuda}


def draw_seeds(context, count, *, forbidden=()):
    """Deterministic prefix-stable high-range seeds, separate from evaluation."""
    count = _integer(count, "num_mean_samples", minimum=2)
    blocked = {int(seed) for seed in forbidden}
    used, result = set(), []
    for draw in range(count):
        nonce = 0
        while True:
            digest = canonical_hash({"domain": POLICY["seed_domain"], "context": context,
                                     "draw_index": draw, "collision_nonce": nonce})
            seed = (1 << 61) | (int(digest[:16], 16) & ((1 << 61) - 1))
            if seed not in blocked and seed not in used:
                break
            nonce += 1
        result.append(seed)
        used.add(seed)
    return result


def plan_mean(sources, config, schedule, reference_law):
    """Return a prompt-independent bank receipt and fixed resumable draw shards."""
    count = _integer(config.get("num_mean_samples", 10_000), "num_mean_samples", minimum=2)
    root_seed = _integer(config.get("mean_seed", 0), "mean_seed")
    science = sources.runs["experiment"]["scientific_config"]
    shape = list(reference_law.support.latent_shape)
    if shape != list(science["latent_shape"]):
        raise TheoryError("Reference-mean support and saved Gaussian latent shapes differ")
    if not shape or any(isinstance(size, bool) or not isinstance(size, int) or size <= 0 for size in shape):
        raise TheoryError("Reference-mean latent shape must have positive integer dimensions")
    atom_identity = {"atoms_sha256": _tensor_hash(reference_law.support.atoms),
                     "weights_sha256": _tensor_hash(reference_law.support.weights),
                     "atom_ids": list(reference_law.support.atom_ids), "latent_shape": shape}
    atom_law_hash = canonical_hash(atom_identity)
    cumulative = float(torch.as_tensor(schedule["alphas_cumprod_t"])[0])
    if not math.isfinite(cumulative) or not 0 < cumulative < 1:
        raise TheoryError("Reference-mean estimation requires positive initial native signal and noise")
    initial_level = {"source_range": "native", "grid_index": None, "step_index": 0,
                     "timestep": int(torch.as_tensor(schedule["timesteps"])[0]),
                     "alpha": math.sqrt(cumulative), "sigma": math.sqrt(1 - cumulative),
                     "snr": cumulative / (1 - cumulative)}
    mode = config.get("mean_source", "reference-min-snr")
    if mode not in MEAN_SOURCES:
        raise TheoryError("Reference-mean estimation requires mean_source reference-min-snr or reference-initial")
    decades = float(config.get("reference_snr_decades", 6.0))
    if not math.isfinite(decades) or not 0 < decades <= 12:
        raise TheoryError("Analytical reference decades must be in (0, 12]")
    minimum_requested = initial_level["snr"] * 10.0 ** -decades
    if not math.isfinite(minimum_requested) or minimum_requested <= 0:
        raise TheoryError("Reference-mean analytical SNR grid must remain positive and finite")
    grid = reference_grid(initial_level["snr"], decades)
    if len(grid) != REFERENCE_GRID_POINTS or any(not math.isfinite(float(value)) or value <= 0 for value in grid):
        raise TheoryError("Reference-mean analytical SNR grid must remain positive and finite")
    minimum_snr = float(grid[0])
    level = (dict(initial_level) if mode == "reference-initial" else {
        "source_range": "analytical", "grid_index": 0, "step_index": None, "timestep": None,
        "alpha": math.sqrt(minimum_snr / (1.0 + minimum_snr)),
        "sigma": 1.0 / math.sqrt(1.0 + minimum_snr), "snr": minimum_snr})
    if any(not math.isfinite(level[name]) or level[name] <= 0 for name in ("alpha", "sigma", "snr")):
        raise TheoryError("Reference-mean estimation level must have positive finite signal, noise and SNR")
    level_receipt = {"mean_source": mode, "source": MEAN_SOURCES[mode], "level": level,
        "initial_level": initial_level, "initial_snr": initial_level["snr"],
        "estimation_snr": level["snr"], "reference_snr_decades": decades,
        "reference_grid_definition": {"function": "evidence_reference.reference_grid",
            "revision": REFERENCE_SWEEP_VERSION, "points": REFERENCE_GRID_POINTS,
            "minimum_snr": minimum_snr, "maximum_snr": float(grid[-1]),
            "grid_sha256": canonical_hash([float(value) for value in grid])}}
    schedule_hash = file_sha256(sources.experiment.schedule)
    forbidden = sorted({int(seed) for run in sources.runs.values()
                        for seed in run.get("scientific_config", {}).get("seeds", [])})
    seed_context = {"mean_seed": root_seed, "reference_atom_law_hash": atom_law_hash,
                    "schedule_sha256": schedule_hash, "latent_shape": shape, **level_receipt}
    seeds = draw_seeds(seed_context, count, forbidden=forbidden)
    bank = {"policy": POLICY, "source_code": _source_code(), "runtime_packages": _runtime(),
            "reference_atom_law": atom_identity, "reference_atom_law_hash": atom_law_hash,
            "latent_shape": shape, **level_receipt,
            "schedule_sha256": schedule_hash, "mean_seed": root_seed,
            "excluded_evaluation_seeds": forbidden, "seed_context": seed_context}
    tasks = []
    for start in range(0, count, LOGICAL_SHARD_SIZE):
        stop = min(start + LOGICAL_SHARD_SIZE, count)
        identity = {"bank": bank, "draw_start": start, "draw_stop": stop,
                    "draw_seeds": seeds[start:stop]}
        tasks.append({"identity": identity, "task_hash": canonical_hash(identity),
                      "draw_start": start, "draw_stop": stop, "count": stop - start})
    identity = {"schema_version": SCHEMA_VERSION, "bank": bank, "num_mean_samples": count,
                "draw_seed_sha256": canonical_hash(seeds), "shard_hashes": [task["task_hash"] for task in tasks]}
    return {"identity": identity, "estimator_hash": canonical_hash(identity), "tasks": tasks}


def _shard_paths(directory, task):
    root = contained_path(directory, "shards/" + task["task_hash"])
    return contained_path(root, "moments.pt"), contained_path(root, "complete.json")


def _completed_shard(directory, task):
    data, marker = _shard_paths(directory, task)
    try:
        saved = read_json(marker)
        if (saved.get("schema_version") != SCHEMA_VERSION or saved.get("complete") is not True
                or saved.get("identity") != task["identity"] or saved.get("task_hash") != task["task_hash"]
                or saved.get("count") != task["count"] or saved.get("sha256") != file_sha256(data)):
            return None
        return saved
    except (CacheIOError, OSError, ValueError, KeyError, TypeError):
        return None


def gaussian_draws(seeds, shape, device):
    """Draw on CUDA per logical seed; worker placement and batches cannot reseed."""
    device = _cuda_device(device)
    samples = torch.empty((len(seeds), *shape), dtype=torch.float64, device=device)
    for index, seed in enumerate(seeds):
        generator = torch.Generator(device=device).manual_seed(int(seed))
        torch.randn(tuple(shape), generator=generator, dtype=torch.float64,
                    device=device, out=samples[index])
    return samples


def shard_moments(clean, draw_start):
    """Reduce one complete fixed shard, independently of inference microbatches."""
    _cuda_device(clean.device)
    if clean.dtype != torch.float64 or clean.ndim < 2 or not bool(torch.isfinite(clean).all()):
        raise TheoryError("Reference-mean clean estimates must be finite CUDA float64 vectors")
    indices = torch.arange(draw_start, draw_start + len(clean), device=clean.device)
    even = indices.remainder(2).eq(0)
    values = {"sum": clean.sum(dim=0), "sum_squares": clean.square().sum(dim=0),
              "even_sum": clean[even].sum(dim=0), "odd_sum": clean[~even].sum(dim=0)}
    values["mean"] = values["sum"] / len(clean)
    values["centered_sum_squares"] = (clean - values["mean"]).square().sum(dim=0)
    if not all(bool(torch.isfinite(value).all()) for value in values.values()):
        raise TheoryError("Independent reference-mean moments overflowed float64")
    return values


def _verify_bank(bank, schedule_path):
    if bank["source_code"] != _source_code() or bank["runtime_packages"] != _runtime():
        raise TheoryError("Independent reference-mean sources or runtime changed after planning")
    if file_sha256(reject_symlinks(schedule_path)) != bank["schedule_sha256"]:
        raise TheoryError("Independent reference-mean saved schedule changed after planning")


def _persist_reference(directory, reference_law, atom_identity):
    """Persist the exact prior atoms once; theory-mean attachments are excluded."""
    root = contained_path(directory, "inputs/" + canonical_hash(atom_identity))
    data, marker = contained_path(root, "atoms.pt"), contained_path(root, "complete.json")
    try:
        receipt = read_json(marker)
        valid = (receipt.get("identity") == atom_identity and receipt.get("complete") is True
                 and receipt.get("sha256") == file_sha256(data))
    except (CacheIOError, OSError, ValueError, TypeError):
        valid = False
    if not valid:
        digest = atomic_torch_save({"atoms": reference_law.support.atoms.detach().cpu(),
                                    "weights": reference_law.support.weights.detach().cpu()}, data)
        atomic_write_json(marker, {"identity": atom_identity, "complete": True, "sha256": digest})
    return {"data_path": str(data), "data_sha256": file_sha256(data),
            "marker_path": str(marker), "marker_sha256": file_sha256(marker)}


def _load_reference(inputs, bank, device, *, candidate_chunk, query_chunk):
    device = _cuda_device(device)
    for name in ("data", "marker"):
        path = reject_symlinks(inputs[name + "_path"])
        if file_sha256(path) != inputs[name + "_sha256"]:
            raise TheoryError("Reference-mean immutable atom input changed")
    receipt = read_json(inputs["marker_path"])
    identity = bank["reference_atom_law"]
    if (receipt.get("identity") != identity or receipt.get("complete") is not True
            or receipt.get("sha256") != inputs["data_sha256"]):
        raise TheoryError("Reference-mean atom receipt differs from the planned law")
    payload = safe_torch_load(inputs["data_path"])
    if (not isinstance(payload, dict) or set(payload) != {"atoms", "weights"}
            or _tensor_hash(payload["atoms"]) != identity["atoms_sha256"]
            or _tensor_hash(payload["weights"]) != identity["weights_sha256"]):
        raise TheoryError("Reference-mean atom vectors or masses differ")
    atoms, weights = (payload[name].to(device=device, dtype=torch.float64) for name in ("atoms", "weights"))
    support = FiniteSupport(atoms, identity["atom_ids"],
                            {key: index for index, key in enumerate(identity["atom_ids"])},
                            weights=weights, candidate_chunk=candidate_chunk, query_chunk=query_chunk)
    # Preserve the exact declared floating masses; do not renormalize twice.
    support.weights = weights.clone()
    support.log_weights = support.weights.log()
    if list(support.latent_shape) != bank["latent_shape"]:
        raise TheoryError("Reference-mean atom input has the wrong latent shape")
    return support


def posterior_batches(samples, *, support, level, batch_size):
    """Bounded CUDA OOM retries for unchanged logical analytical query slices."""
    _cuda_device(samples.device)
    if support.flat.device != samples.device:
        raise TheoryError("Reference-mean posterior and observations must share the same CUDA device")
    size = _integer(batch_size, "mean_batch_size", minimum=1)
    position, retries = 0, 0
    while position < len(samples):
        stop = min(position + size, len(samples))
        retry = False
        try:
            with torch.inference_mode():
                result = support.posterior_mean(samples[position:stop], level["alpha"], level["sigma"])
        except RuntimeError as error:
            is_oom = isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower()
            if not is_oom or stop - position <= 1 or retries >= 32:
                raise
            size = max(1, (stop - position) // 2)
            retries += 1
            retry = True
        if retry:
            torch.cuda.empty_cache()
            continue
        yield position, stop, result, retries
        position = stop


def _worker(*, inputs, schedule_path, tasks, directory, device, batch_size, worker_count,
            candidate_chunk=256, query_chunk=16):
    configure_worker_cpu_threads(worker_count)
    device = _cuda_device(device)
    torch.cuda.set_device(device)
    bank = tasks[0]["identity"]["bank"]
    _verify_bank(bank, schedule_path)
    support = None
    samples = clean = None
    completed, failures = [], []
    try:
        for task in tasks:
            try:
                if _completed_shard(directory, task) is not None:
                    report_worker_record(task["task_hash"], "resumed", device)
                    continue
                if support is None:
                    support = _load_reference(inputs, bank, device,
                                              candidate_chunk=candidate_chunk, query_chunk=query_chunk)
                samples = gaussian_draws(task["identity"]["draw_seeds"], bank["latent_shape"], device)
                clean = torch.empty_like(samples)
                covered, oom_retries = 0, 0
                for start, stop, posterior, retries in posterior_batches(
                        samples, support=support, level=bank["level"], batch_size=batch_size):
                    if start != covered or not start < stop <= len(samples) or posterior.shape != samples[start:stop].shape:
                        raise TheoryError("Reference-mean posterior omitted or repeated logical draws")
                    if posterior.device != samples.device or posterior.dtype != torch.float64:
                        raise TheoryError("Reference-mean posterior vectors must remain CUDA float64")
                    clean[start:stop] = posterior
                    covered, oom_retries = stop, max(oom_retries, retries)
                if covered != task["count"]:
                    raise TheoryError("Reference-mean shard has an incomplete Gaussian bank")
                moments = shard_moments(clean, task["draw_start"])
                _verify_bank(bank, schedule_path)
                data, marker = _shard_paths(directory, task)
                digest = atomic_torch_save({name: value.detach().cpu() for name, value in moments.items()}, data)
                atomic_write_json(marker, {"schema_version": SCHEMA_VERSION, "complete": True,
                    "identity": task["identity"], "task_hash": task["task_hash"], "count": task["count"], "sha256": digest,
                    "execution": {"device": str(device), "device_name": torch.cuda.get_device_name(device),
                        "numeric_backend": "CUDA_float64", "random_stream_backend": "CUDA_per_draw_generator_float64",
                        "batch_size_requested": batch_size, "oom_retries": oom_retries,
                        "posterior_observations": covered, "network_observations": 0,
                        "candidate_chunk": candidate_chunk, "query_chunk": query_chunk,
                        "reference_atom_law_hash": bank["reference_atom_law_hash"]}})
                completed.append(task["task_hash"])
                report_worker_record(task["task_hash"], "reduced", device)
            except Exception as error:
                failures.append({"task_hash": task["task_hash"], "draw_start": task["draw_start"],
                                 "draw_stop": task["draw_stop"], "error": f"{type(error).__name__}: {error}"})
                report_worker_record(task["task_hash"], "failed", device)
            finally:
                samples = clean = None
    finally:
        support = None
        torch.cuda.empty_cache()
    return {"completed": completed, "failures": failures, "device": str(device)}


def combine_moments(parts, counts):
    """Combine fixed-order CUDA moments and compute coordinate Monte Carlo error."""
    if not parts or len(parts) != len(counts):
        raise TheoryError("Reference-mean aggregation requires every fixed logical shard")
    device = _cuda_device(parts[0]["sum"].device)
    totals = {name: torch.zeros_like(parts[0][name], device=device) for name in SUM_FIELDS}
    combined_mean = torch.zeros_like(parts[0]["mean"], device=device)
    centered_sum_squares = torch.zeros_like(combined_mean)
    count = 0
    for part, size in zip(parts, counts, strict=True):
        _integer(size, "logical shard sample count", minimum=1)
        for name in MOMENT_FIELDS:
            value = part[name]
            if (value.device != device or value.dtype != torch.float64 or value.shape != combined_mean.shape
                    or not bool(torch.isfinite(value).all())):
                raise TheoryError("Reference-mean moment shape, device, dtype or finite-value contract differs")
        if bool((part["centered_sum_squares"] < 0).any()):
            raise TheoryError("Reference-mean centered shard variance cannot be negative")
        for name in SUM_FIELDS:
            totals[name].add_(part[name])
        delta = part["mean"] - combined_mean
        combined_count = count + size
        centered_sum_squares.add_(part["centered_sum_squares"] + delta.square() * (count * size / combined_count))
        combined_mean.add_(delta * (size / combined_count))
        count = combined_count
    if count < 2:
        raise TheoryError("Reference-mean variance requires at least two independent draws")
    mean = totals["sum"] / count
    # Centered shard M2 and Chan merging avoid subtracting two huge raw moments
    # when a small Monte Carlo variance lies around a much larger vector mean.
    variance = centered_sum_squares / (count - 1)
    standard_error = (variance / count).sqrt()
    even_count, odd_count = (count + 1) // 2, count // 2
    split_difference = totals["even_sum"] / even_count - totals["odd_sum"] / odd_count
    if not all(bool(torch.isfinite(value).all()) for value in (mean, variance, standard_error, split_difference)):
        raise TheoryError("Reference-mean aggregate is not finite")
    root_d = math.sqrt(mean.numel())
    metadata = {"sample_count": count, "latent_dimension": mean.numel(),
                "mean_norm_l2": float(mean.norm()), "mean_norm_rmse": float(mean.norm()) / root_d,
                "mean_mc_standard_error_l2": float(standard_error.norm()),
                "mean_mc_standard_error_rmse": float(standard_error.norm()) / root_d,
                "mean_mc_standard_error_max_coordinate": float(standard_error.max()),
                "split_half_difference_rmse": float(split_difference.norm()) / root_d,
                "split_half_counts": [even_count, odd_count],
                "variance_algorithm": "centered_per_shard_M2_with_fixed_order_Chan_merge",
                "uncertainty_scope": "Monte_Carlo_sampling_variance_at_fixed_selected_noise; excludes_finite_noise_bias_and_reference_law_misspecification"}
    return {"mean": mean, "coordinate_variance": variance, "coordinate_standard_error": standard_error}, metadata


def _load_completed(directory, identity):
    """Validate cached vector/uncertainty receipts; this read performs no estimation."""
    try:
        metadata = read_json(contained_path(directory, "complete.json"))
        bank, count = identity["bank"], identity["num_mean_samples"]
        shape = bank["latent_shape"]
        if (metadata.get("schema_version") != SCHEMA_VERSION or metadata.get("complete") is not True
                or metadata.get("identity") != identity or metadata.get("estimator_hash") != canonical_hash(identity)
                or type(metadata.get("sample_count")) is not int or metadata["sample_count"] != count
                or type(metadata.get("mean_seed")) is not int or metadata["mean_seed"] != bank["mean_seed"]
                or any(metadata.get(name) != bank[name] for name in LEVEL_RECEIPT_FIELDS)
                or metadata.get("reference_atom_law_hash") != bank["reference_atom_law_hash"]
                or metadata.get("reference_atom_law") != bank["reference_atom_law"]
                or metadata.get("latent_dimension") != math.prod(shape)
                or metadata.get("split_half_counts") != [(count + 1) // 2, count // 2]):
            return None
        scalar_names = ("mean_norm_l2", "mean_norm_rmse", "mean_mc_standard_error_l2",
                        "mean_mc_standard_error_rmse", "mean_mc_standard_error_max_coordinate",
                        "split_half_difference_rmse")
        for name in scalar_names:
            value = metadata.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                return None
        root_d = math.sqrt(math.prod(shape))
        for prefix in ("mean_norm", "mean_mc_standard_error"):
            if not math.isclose(metadata[prefix + "_rmse"], metadata[prefix + "_l2"] / root_d,
                                rel_tol=1e-12, abs_tol=1e-14):
                return None
        files = metadata["files"]
        if set(files) != {"mean.pt", "uncertainty.pt"}:
            return None
        for name, digest in files.items():
            if file_sha256(contained_path(directory, name)) != digest:
                return None
        mean = safe_torch_load(contained_path(directory, "mean.pt"))
        if (not isinstance(mean, torch.Tensor) or mean.dtype != torch.float64 or list(mean.shape) != shape
                or not bool(torch.isfinite(mean).all()) or metadata.get("vector_sha256") != _tensor_hash(mean)):
            return None
        uncertainty = safe_torch_load(contained_path(directory, "uncertainty.pt"))
        if not isinstance(uncertainty, dict) or set(uncertainty) != {"coordinate_variance", "coordinate_standard_error"}:
            return None
        for value in uncertainty.values():
            if (not isinstance(value, torch.Tensor) or value.dtype != torch.float64 or list(value.shape) != shape
                    or not bool(torch.isfinite(value).all()) or bool((value < 0).any())):
                return None
        return {"vector": mean, "metadata": metadata, "directory": Path(directory)}
    except (CacheIOError, OSError, ValueError, KeyError, TypeError):
        return None


def estimate_reference_mean(sources, config, *, reference_law, device="auto", batch_size=8, allow_compute=True):
    """Average reference posteriors at one declared SNR, or load the exact vector.

    Batching, worker count and prompts are absent from sample identities. Missing
    shards require ``allow_compute=True``; cached full-vector reads need no CUDA.
    """
    from .reduce import _resolve_theory_devices

    batch_size = _integer(batch_size, "mean_batch_size", minimum=1)
    schedule_path = reject_symlinks(Path(sources.experiment.schedule).absolute())
    schedule = safe_torch_load(schedule_path)
    plan = plan_mean(sources, config, schedule, reference_law)
    paper = PaperPaths.build(sources.root, **config).output_directory
    root = contained_path(paper.parent.parent, "theory_measurements/reference_mean")
    directory = contained_path(root, "collections/" + plan["estimator_hash"])
    saved = _load_completed(directory, plan["identity"])
    if saved is not None:
        return saved
    if not allow_compute:
        raise TheoryError("Independent selected-SNR reference-mean cache is missing or incompatible. Run " + recompute_command(config))
    devices = tuple(_resolve_theory_devices(device))
    if not devices:
        raise TheoryError("Reference-mean estimation requires at least one CUDA device")
    for selected in devices:
        _cuda_device(selected)
    tasks, bank = plan["tasks"], plan["identity"]["bank"]
    with publication_lock(root):
        saved = _load_completed(directory, plan["identity"])
        if saved is not None:
            return saved
        pending = [task for task in tasks if _completed_shard(root, task) is None]
        workers = worker_count_for_tasks(devices, len(pending)) if pending else 0
        inputs = _persist_reference(root, reference_law, bank["reference_atom_law"]) if pending else None
        context = get_context("spawn")
        failures, execution = [], []
        with RecordProgress(total=len(tasks), devices=workers, context=context) as progress:
            level_label = "min SNR" if bank["mean_source"] == "reference-min-snr" else "initial SNR"
            progress.bar.set_description(
                f"[Theory] Reference mean ({level_label}={bank['estimation_snr']:.6g}, "
                f"{plan['identity']['num_mean_samples']:,} Gaussian draws)")
            progress.bar.unit = "shard"
            pending_ids = {task["task_hash"] for task in pending}
            for task in tasks:
                if task["task_hash"] not in pending_ids:
                    progress.report(task["task_hash"], "resumed", "saved")
            if workers:
                with ProcessPoolExecutor(max_workers=workers, mp_context=context,
                        initializer=install_progress_queue, initargs=(progress.queue,)) as executor:
                    futures = {}
                    for index in range(workers):
                        assigned = round_robin_shard(pending, worker_index=index, worker_count=workers)
                        future = executor.submit(_worker, inputs=inputs,
                            schedule_path=str(schedule_path), tasks=assigned, directory=str(root),
                            device=str(devices[index]), batch_size=batch_size, worker_count=workers,
                            candidate_chunk=reference_law.support.candidate_chunk,
                            query_chunk=reference_law.support.query_chunk)
                        futures[future] = index, assigned
                    for future in as_completed(futures):
                        index, assigned = futures[future]
                        try:
                            result = future.result()
                        except Exception as error:
                            result = {"completed": [], "device": str(devices[index]), "failures": [
                                {"task_hash": task["task_hash"], "error": f"worker exited: {type(error).__name__}: {error}"}
                                for task in assigned if _completed_shard(root, task) is None]}
                        failures.extend(result["failures"])
                        execution.append(result)
                        for task in assigned:
                            status = "reduced" if _completed_shard(root, task) is not None else "failed"
                            progress.report(task["task_hash"], status, devices[index])
        incomplete = [task for task in tasks if _completed_shard(root, task) is None]
        if incomplete:
            atomic_write_json(contained_path(directory, "failed.json"), {"complete": False,
                "estimator_hash": plan["estimator_hash"], "incomplete_shards": [task["task_hash"] for task in incomplete],
                "failures": failures})
            raise TheoryError(f"Independent reference mean has {len(incomplete)}/{len(tasks)} incomplete shards; completed GPU moments retained at {root}")
        _verify_bank(bank, schedule_path)
        parts, receipts = [], []
        for task in tasks:
            data, marker = _shard_paths(root, task)
            saved_shard = _completed_shard(root, task)
            if saved_shard is None:
                raise TheoryError("A reference-mean shard changed before aggregation")
            payload = safe_torch_load(data)
            if not isinstance(payload, dict) or set(payload) != set(MOMENT_FIELDS):
                raise TheoryError("Invalid independent reference-mean moment payload")
            for value in payload.values():
                if not isinstance(value, torch.Tensor) or list(value.shape) != bank["latent_shape"]:
                    raise TheoryError("Reference-mean moment payload has an invalid latent shape")
            parts.append({name: value.to(devices[0]) for name, value in payload.items()})
            receipts.append({"task_hash": task["task_hash"], "sha256": saved_shard["sha256"],
                             "marker_sha256": file_sha256(marker), "count": task["count"]})
        values, metadata = combine_moments(parts, [task["count"] for task in tasks])
        _verify_bank(bank, schedule_path)
        mean = values["mean"].detach().cpu()
        files = {"mean.pt": atomic_torch_save(mean, contained_path(directory, "mean.pt")),
                 "uncertainty.pt": atomic_torch_save({name: values[name].detach().cpu() for name in (
                     "coordinate_variance", "coordinate_standard_error")}, contained_path(directory, "uncertainty.pt"))}
        metadata.update(schema_version=SCHEMA_VERSION, complete=True, identity=plan["identity"],
            estimator_hash=plan["estimator_hash"], files=files, shard_receipts=receipts,
            vector_sha256=_tensor_hash(mean),
            vector_path=str(contained_path(directory, "mean.pt")), mean_seed=bank["mean_seed"],
            **{name: bank[name] for name in LEVEL_RECEIPT_FIELDS},
            reference_atom_law_hash=bank["reference_atom_law_hash"], reference_atom_law=bank["reference_atom_law"],
            definition=POLICY["target"], bias_scope=POLICY["interpretation"],
            interpretation=POLICY["interpretation"], estimator_target=POLICY["target"],
            execution={"devices": list(map(str, devices)), "worker_count": workers,
                       "submitted_shards": len(pending), "workers": execution, "batch_size_requested": batch_size,
                       "aggregation_device": str(devices[0]), "new_posterior_observations": sum(task["count"] for task in pending), "network_observations": 0,
                       "parent_progress_bars": 1, "no_scheduler_updates_or_decoding": True})
        atomic_write_json(contained_path(directory, "complete.json"), metadata)
        result = _load_completed(directory, plan["identity"])
        if result is None:
            raise TheoryError("New reference-mean vector did not pass its completion receipt")
        return result
