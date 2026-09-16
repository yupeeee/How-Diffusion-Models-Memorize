"""Opt-in learned empty-prompt comparisons at fixed matched next-state inputs.

Both endpoints are always inferred in the same new checkpoint/dtype context.
The saved actual prediction is a parity diagnostic, never silently reused.
Tasks contain one retained record and one predeclared chronological update;
all its evaluation seeds survive as measured, inapplicable, or failed rows.
No scheduler step, new innovation, trajectory rollout, or decoder is invoked.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
import math

import pandas as pd
import torch

from utils.common.io import atomic_write_frame_parquet, atomic_write_json, canonical_hash, file_sha256
from utils.models.devices import configure_worker_cpu_threads, worker_count_for_tasks
from utils.models.loading import package_version_metadata
from utils.models.sampling import encode_prompt_condition
from .contracts import TheoryError
from .direct_probes import (_completed_task, _load_replica, _pair_keys, _positive_integer,
                            _record_stamp, _rows, _task_paths, _tensor_digest, prediction_batches)
from .paper_contracts import PaperPaths, contained_path, publication_lock, recompute_command, reject_symlinks
from .progress import RecordProgress, install_progress_queue, report_worker_record
from .supplemental_cache import existing_collection, scientific_base_receipt

SCHEMA_VERSION = 1
TABLE = "counterfactual_unconditional"
POLICY = {
    "version": "learned-matched-unconditional-1",
    "input_law": "optional_learned_network_counterfactual_probe",
    "prompt": "exact_empty_prompt_no_CFG",
    "actual_prediction": "always_remeasure_both_endpoints_same_new_inference_context",
    "saved_prediction": "parity_only_already_canonical_epsilon_no_reconversion",
    "new_prediction": "native_to_epsilon_once_against_unscaled_conceptual_input",
    "matched_update": "preserved_SchedulerAdapter_direct_matched_update",
    "stochastic_noise": "recovered_shared_realized_innovation_not_independent_replay",
    "reduction": "float64_raw_L2_then_divide_by_sqrt_dimension_once",
    "parity": "raw_L2_max_abs_exact_and_saved_storage_quantized_agreement_no_certification",
    "selection": "fixed_chronological_steps_all_retained_records_and_evaluation_seeds",
    "terminal": "requires_existing_next_prediction_and_matching_positive_noise_native_label",
}


def _source_code():
    theory = Path(__file__).parent
    paths = [theory / name for name in ("counterfactual_probes.py", "supplemental_cache.py", "direct_probes.py", "direct_probe_cache.py",
                                       "scheduler_adapter.py", "metrics.py", "cache_reader.py")]
    paths += [theory.parent.parent / "models" / name for name in (
        "probe_loading.py", "loading.py", "sampling.py", "prediction_conversion.py", "schedulers.py")]
    return {str(path.relative_to(theory.parent.parent)): file_sha256(path) for path in paths}


def _base_config(config):
    return {key: value for key, value in config.items()
            if key not in {"counterfactual_unconditional", "counterfactual_steps"}}


def _fixed_steps(config):
    steps = config.get("counterfactual_steps", [0])
    if not isinstance(steps, (list, tuple)) or not steps:
        raise TheoryError("counterfactual_steps must be a nonempty fixed list of chronological indices")
    if any(isinstance(step, bool) or not isinstance(step, int) or step < 0 for step in steps):
        raise TheoryError("counterfactual_steps must contain nonnegative integers")
    if len(set(steps)) != len(steps):
        raise TheoryError("counterfactual_steps must not repeat a snapshot")
    return tuple(sorted(steps))


def _snapshot_contract(adapter, schedule, step):
    """Never invent a noise label or a branch prediction at the final output."""
    native = torch.as_tensor(schedule["timesteps"]).tolist()
    result = {"step_index": step, "destination_step_index": step + 1,
              "timestep": int(native[step]) if step < len(native) else None,
              "destination_timestep": int(native[step + 1]) if step + 1 < len(native) else None}
    if step >= len(native) - 1:
        return result | {"applicable": False, "reason": "no_saved_next_branch_prediction_at_terminal_or_out_of_range_snapshot"}
    if adapter is None:
        return result | {"applicable": False, "reason": "scheduler_adapter_unavailable"}
    coeff = adapter.coefficients(step)
    result.update({name: float(getattr(coeff, name)) for name in (
        "alpha", "sigma", "destination_alpha", "destination_sigma", "A", "kappa", "noise_std")})
    # Nonfinite unsupported coefficients belong to explicit missingness, not an
    # invalid JSON task identity. Their scheduler status supplies the reason.
    result = {key: None if isinstance(value, float) and not math.isfinite(value) else value
              for key, value in result.items()}
    result["scheduler_status"] = coeff.status
    result["deterministic_update"] = bool(coeff.deterministic)
    if not coeff.affine or not math.isfinite(coeff.kappa) or coeff.kappa <= 0:
        return result | {"applicable": False, "reason": "inapplicable_matched_affine_update:" + coeff.status}
    if coeff.destination_timestep != result["destination_timestep"]:
        return result | {"applicable": False, "reason": "scheduler_destination_is_not_the_next_saved_prediction_label"}
    if not all(math.isfinite(value) and value > 0 for value in (
            coeff.alpha, coeff.sigma, coeff.destination_alpha, coeff.destination_sigma)):
        return result | {"applicable": False, "reason": "nonpositive_or_nonfinite_current_or_destination_signal_noise"}
    next_coeff = adapter.coefficients(step + 1)
    if not (math.isclose(coeff.destination_alpha, next_coeff.alpha, rel_tol=2e-6, abs_tol=1e-8)
            and math.isclose(coeff.destination_sigma, next_coeff.sigma, rel_tol=2e-6, abs_tol=1e-8)):
        return result | {"applicable": False, "reason": "destination_coefficients_differ_from_next_saved_prediction_contract"}
    return result | {"applicable": True, "reason": "applicable_saved_nonterminal_affine_transition"}


def plan_counterfactual_tasks(sources, records, schedule, config):
    """Immutable record/snapshot tasks, independent of workers and microbatches."""
    from .scheduler_adapter import SchedulerAdapter

    steps = _fixed_steps(config)
    science = sources.runs["experiment"]["scientific_config"]
    if science.get("stored_prediction_type") != "epsilon":
        raise TheoryError("Counterfactual probes require saved canonical epsilon")
    if science.get("inference_dtype") not in {"float16", "bfloat16", "float32"}:
        raise TheoryError("Counterfactual probes require the preserved inference dtype")
    try:
        adapter = SchedulerAdapter(schedule, recorded_diffusers_version=science.get("package_versions", {}).get("diffusers"))
        adapter_error = None
    except (ValueError, TypeError, KeyError) as error:
        adapter, adapter_error = None, f"{type(error).__name__}: {error}"
    runtime = package_version_metadata()
    shared = {"schema_version": SCHEMA_VERSION, "policy": POLICY, "source_code": _source_code(),
              "schedule_sha256": file_sha256(sources.experiment.schedule),
              "checkpoint": {name: science[name] for name in (
                  "model_id", "model_revision", "native_prediction_type", "inference_dtype")},
              "storage_contract": science.get("scientific_tensor_storage"),
              "latent_shape": science["latent_shape"], "target_preprocessing": science.get("target_preprocessing"),
              "vae_id": science.get("vae_id"), "vae_revision": science.get("vae_revision"),
              "guidance_scale": float(config["guidance_scale"]),
              "runtime_packages": {name: runtime[name] for name in ("torch", "diffusers", "transformers")}}
    tasks = []
    for record in sorted(records, key=lambda item: str(item.original_index)):
        seeds = list(map(int, record.metadata["seeds"]))
        if seeds != list(map(int, science["seeds"])) or len(seeds) != len(set(seeds)):
            raise TheoryError("Counterfactual probes must retain the complete saved evaluation seed bank")
        scores = sources.proximity.loc[sources.proximity.original_index.astype(str).eq(str(record.original_index))]
        by_seed = {}
        for seed in seeds:
            values = scores.loc[scores.seed.astype(int).eq(seed), "sscd"]
            if len(values) != 1 or not math.isfinite(float(values.iloc[0])):
                raise TheoryError("Counterfactual probe lacks its exact same-seed target SSCD")
            by_seed[str(seed)] = float(values.iloc[0])
        pair = _pair_keys(sources, record)
        stamp = _record_stamp(sources, record)
        for step in steps:
            level = _snapshot_contract(adapter, schedule, step)
            if adapter_error and step < len(schedule["timesteps"]) - 1:
                level["reason"] += ":" + adapter_error
            identity = {**shared, "table": TABLE, "count": len(seeds), "pair": pair,
                        "level": level, "seeds": seeds, "source_record": stamp,
                        "same_seed_terminal_sscd": by_seed}
            tasks.append({"identity": identity, "task_hash": canonical_hash(identity), "table": TABLE,
                          "count": len(seeds), "record_index": str(record.original_index),
                          "pair": pair, "level": level, "seeds": seeds})
    return tasks


def _status_rows(task, *, status, reason):
    level = task["level"]
    return [{**task["pair"], **level, "seed": seed, "task_hash": task["task_hash"],
             "terminal_sscd": task["identity"]["same_seed_terminal_sscd"][str(seed)],
             "status": status, "counterfactual_status": status, "reason": reason,
             "counterfactual_applicable": bool(level["applicable"]),
             "input_source": POLICY["input_law"], "latent_dimension": math.prod(task["identity"]["latent_shape"]),
             "counterfactual_I_net": math.nan, "counterfactual_target_error_l2": math.nan,
             "counterfactual_target_error_rmse": math.nan, "actual_next_target_error_l2": math.nan,
             "actual_next_target_error_rmse": math.nan, "saved_actual_prediction_reused": False}
            for seed in task["seeds"]]


def counterfactual_metrics(counterfactual, actual, epsilon_counterfactual, epsilon_actual,
                           saved_epsilon_actual, target, *, alpha, sigma):
    """Learned clean errors and saved-canonical parity, with one normalization."""
    cf, actual, cf_eps, actual_eps, saved = [torch.as_tensor(value).double() for value in (
        counterfactual, actual, epsilon_counterfactual, epsilon_actual, saved_epsilon_actual)]
    if not (cf.shape == actual.shape == cf_eps.shape == actual_eps.shape == saved.shape) or cf.ndim < 2:
        raise TheoryError("Counterfactual endpoint and prediction shapes differ")
    if not math.isfinite(alpha) or not math.isfinite(sigma) or alpha <= 0 or sigma <= 0:
        raise TheoryError("A learned next-state comparison needs an actual positive-noise label")
    target = torch.as_tensor(target, dtype=torch.float64, device=cf.device)
    if target.shape != cf.shape[1:]:
        raise TheoryError("Counterfactual target shape differs")
    if not all(bool(torch.isfinite(value).all()) for value in (cf, actual, cf_eps, actual_eps, saved, target)):
        raise TheoryError("Counterfactual measurement received nonfinite values")
    dimension = target.numel()
    norm = lambda value: value.flatten(1).norm(dim=1)
    cf_error = norm((cf - sigma * cf_eps) / alpha - target)
    actual_error = norm((actual - sigma * actual_eps) / alpha - target)
    saved_error = norm((actual - sigma * saved) / alpha - target)
    difference = actual_eps - saved
    stored_dtype = torch.as_tensor(saved_epsilon_actual).dtype
    return {"counterfactual_I_net": (cf_error - actual_error) / math.sqrt(dimension),
            "counterfactual_target_error_l2": cf_error,
            "counterfactual_target_error_rmse": cf_error / math.sqrt(dimension),
            "actual_next_target_error_l2": actual_error,
            "actual_next_target_error_rmse": actual_error / math.sqrt(dimension),
            "saved_actual_next_target_error_l2": saved_error,
            "saved_actual_next_target_error_rmse": saved_error / math.sqrt(dimension),
            "saved_actual_epsilon_parity_l2": norm(difference),
            "saved_actual_epsilon_parity_rmse": norm(difference) / math.sqrt(dimension),
            "saved_actual_epsilon_parity_max_abs": difference.flatten(1).abs().amax(1),
            "saved_actual_epsilon_exact_equal": (difference == 0).flatten(1).all(1),
            "saved_actual_epsilon_storage_quantized_equal": (actual_eps.to(dtype=stored_dtype).double() == saved).flatten(1).all(1)}


def _measure_task(task, log, adapter, *, science, components, scheduler, condition, batch_size):
    z, u, c, target = log
    step, level = task["level"]["step_index"], task["level"]
    diagnostics, matched, _shift = adapter.direct_matched_update(
        z[:, step], z[:, step + 1], u[:, step], c[:, step],
        task["identity"]["guidance_scale"], step, latent_ndim=target.ndim)
    actual = z[:, step + 1].double()
    # Interleave paired endpoints; no cached epsilon is ever fed through native
    # conversion. Adaptive OOM retries cannot drop or reorder a logical sample.
    samples = torch.stack((matched.cpu(), actual.cpu()), dim=1).flatten(0, 1)
    fresh = torch.empty_like(samples, dtype=torch.float64)
    precision_rows = [None] * len(samples)
    for start, stop, prediction, precision in prediction_batches(
            samples, condition=condition, timestep=level["destination_timestep"], components=components,
            scheduler=scheduler, batch_size=2 * batch_size):
        if prediction.shape != samples[start:stop].shape:
            raise TheoryError("Counterfactual prediction batch does not match its logical endpoint slice")
        fresh[start:stop] = prediction
        precision_rows[start:stop] = _rows(precision, stop - start, {})
    if any(row is None for row in precision_rows):
        raise TheoryError("Counterfactual inference omitted an endpoint")
    paired = fresh.reshape(len(actual), 2, *actual.shape[1:])
    values = counterfactual_metrics(matched, actual, paired[:, 0], paired[:, 1], u[:, step + 1], target,
                                   alpha=level["destination_alpha"], sigma=level["destination_sigma"])
    rows = _status_rows(task, status="measured", reason="both_learned_endpoints_measured_in_same_new_context")
    measurements = _rows(values | diagnostics, len(rows), {})
    for index, row in enumerate(rows):
        row.update(measurements[index])
        row.update({"prediction_source": POLICY["actual_prediction"], "prompt": "",
                    "model_id": science["model_id"], "model_revision": science["model_revision"],
                    "native_prediction_type": science["native_prediction_type"],
                    "stored_prediction_type": "epsilon", "inference_dtype": science["inference_dtype"],
                    "saved_prediction_storage_dtype": str(u.dtype).removeprefix("torch."),
                    "counterfactual_input_sha256": _tensor_digest(matched[index]),
                    "actual_input_sha256": _tensor_digest(z[index, step + 1]),
                    "saved_actual_epsilon_sha256": _tensor_digest(u[index, step + 1]),
                    "matched_endpoint_contract": "independent_affine_counterfactual_vs_saved_actual" if level["deterministic_update"]
                         else "recovered_shared_innovation_counterfactual_vs_saved_actual_not_independent_replay",
                    "parity_status": "exact_epsilon_values" if row["saved_actual_epsilon_exact_equal"]
                         else "new_inference_differs_from_saved_epsilon; saved_prediction_not_reused",
                    "parity_scope": "descriptive_prediction_difference_not_a_model_precision_certificate"})
        for prefix, precision in (("counterfactual_", precision_rows[2 * index]), ("actual_", precision_rows[2 * index + 1])):
            row.update({prefix + name: value for name, value in precision.items()})
    return pd.DataFrame(rows)


def _worker(*, sources, records, tasks, schedule_path, output_directory, device, batch_size, worker_count):
    from .cache_reader import load_record
    from .scheduler_adapter import SchedulerAdapter

    configure_worker_cpu_threads(worker_count)
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    from utils.common.io import safe_torch_load
    if tasks[0]["identity"]["source_code"] != _source_code():
        raise TheoryError("Counterfactual measurement code changed after task planning")
    if file_sha256(schedule_path) != tasks[0]["identity"]["schedule_sha256"]:
        raise TheoryError("Preserved counterfactual scheduler changed")
    schedule = safe_torch_load(schedule_path)
    science = sources.runs["experiment"]["scientific_config"]
    records = {str(record.original_index): record for record in records}
    components = scheduler = condition = adapter = None
    loaded_record, log = None, None
    completed, failures = [], []
    replica_metadata = {}
    try:
        for task in tasks:
            status = "failed"
            try:
                if _completed_task(output_directory, task) is not None:
                    status = "resumed"
                    continue
                record = records[task["record_index"]]
                if _record_stamp(sources, record) != task["identity"]["source_record"]:
                    raise TheoryError("Preserved counterfactual source record changed")
                if not task["level"]["applicable"]:
                    frame = pd.DataFrame(_status_rows(task, status="not_applicable", reason=task["level"]["reason"]))
                else:
                    if loaded_record != task["record_index"]:
                        log = load_record(sources.experiment, record, verify_hashes=True)
                        loaded_record = task["record_index"]
                    if adapter is None:
                        adapter = SchedulerAdapter(schedule, recorded_diffusers_version=science.get("package_versions", {}).get("diffusers"))
                    if components is None:
                        components, scheduler = _load_replica(science, schedule, device)
                        condition = encode_prompt_condition("", components.tokenizer, components.text_encoder, device, components.inference_dtype)
                        replica_metadata = components.device_metadata | {"package_versions": components.package_versions}
                    frame = _measure_task(task, log, adapter, science=science, components=components,
                                          scheduler=scheduler, condition=condition, batch_size=batch_size)
                    if _record_stamp(sources, record) != task["identity"]["source_record"]:
                        raise TheoryError("Preserved counterfactual inputs changed during inference")
                if len(frame) != task["count"] or frame.seed.astype(int).tolist() != task["seeds"]:
                    raise TheoryError("Counterfactual task changed its complete evaluation seed population")
                data, marker = _task_paths(output_directory, task)
                atomic_write_frame_parquet(frame, data)
                atomic_write_json(marker, {"schema_version": SCHEMA_VERSION, "task_hash": task["task_hash"],
                    "identity": task["identity"], "complete": True, "rows": len(frame), "sha256": file_sha256(data),
                    "execution": {"device": str(device), "batch_size_requested": batch_size,
                                  "replica": replica_metadata, "new_inference": task["level"]["applicable"]}})
                completed.append(task["task_hash"])
                status = "reduced"
            except Exception as error:
                failures.append({"task_hash": task["task_hash"], "record_index": task["record_index"],
                                 "step_index": task["level"]["step_index"], "status": "failed",
                                 "error": f"{type(error).__name__}: {error}"})
            finally:
                report_worker_record(task["task_hash"], status, device)
    finally:
        log = components = scheduler = condition = adapter = None
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return {"completed": completed, "failures": failures, "device": str(device), "replica": replica_metadata}


def _missing_command(config):
    command = recompute_command(_base_config(config))
    return command + " --counterfactual-unconditional --counterfactual-steps " + ",".join(map(str, _fixed_steps(config)))


def run_counterfactual_analysis(project_root, *, result, config, device="auto", probe_batch_size=8,
                                allow_compute=True, **_unused_options):
    """Enrich a completed analysis only when this separate supplement is enabled.

    allow_compute=False resolves exact saved optional shards or gives an explicit
    author command; it never falls through into raw reads or checkpoint loading.
    Optional failures remain missing observations and cannot block default figures.
    """
    enabled = config.get("counterfactual_unconditional", False)
    if not isinstance(enabled, bool):
        raise TheoryError("counterfactual_unconditional must be boolean")
    if not enabled:
        return result
    from .cache_reader import discover_sources, load_schedule
    from .reduce import _resolve_theory_devices, _worker_sources

    batch_size = _positive_integer(probe_batch_size, "probe_batch_size")
    sources = discover_sources(project_root, **_base_config(config))
    records = sorted(sources.selected, key=lambda record: str(record.original_index))
    if not records:
        raise TheoryError("Counterfactual supplement requires frozen retained records")
    if len({str(record.original_index) for record in records}) != len(records):
        raise TheoryError("Counterfactual supplement found duplicate retained records")
    schedule = load_schedule(sources)
    tasks = plan_counterfactual_tasks(sources, records, schedule, config)
    paper = PaperPaths.build(project_root, **_base_config(config)).output_directory
    directory = contained_path(paper.parent.parent, "theory_measurements/counterfactual_probes")
    pending = [task for task in tasks if _completed_task(directory, task) is None]
    if pending and not allow_compute:
        raise TheoryError(f"{len(pending)} optional learned-counterfactual tasks are missing; cache-only refinement cannot run inference. Run {_missing_command(config)}")
    receipts, failures = [], []
    with publication_lock(directory):
        pending = [task for task in tasks if _completed_task(directory, task) is None]
        if pending and not allow_compute:
            raise TheoryError("Counterfactual receipts changed during cache-only resume. Run " + _missing_command(config))
        devices = tuple(_resolve_theory_devices(device)) if pending else ()
        pending_records = sorted({task["record_index"] for task in pending})
        workers = worker_count_for_tasks(devices, len(pending_records)) if pending else 0
        context = get_context("spawn")
        with RecordProgress(total=len(tasks), devices=workers, context=context) as progress:
            pending_ids = {task["task_hash"] for task in pending}
            for task in tasks:
                if task["task_hash"] not in pending_ids:
                    progress.report(task["task_hash"], "resumed", "saved")
            if workers:
                owners = {record: index % workers for index, record in enumerate(pending_records)}
                shards = [[task for task in pending if owners[task["record_index"]] == index] for index in range(workers)]
                with ProcessPoolExecutor(max_workers=workers, mp_context=context,
                        initializer=install_progress_queue, initargs=(progress.queue,)) as executor:
                    futures = {}
                    for index, shard in enumerate(shards):
                        assigned = [record for record in records if owners.get(str(record.original_index)) == index]
                        future = executor.submit(_worker, sources=_worker_sources(sources, assigned), records=assigned,
                            tasks=shard, schedule_path=sources.experiment.schedule, output_directory=directory,
                            device=str(devices[index]), batch_size=batch_size, worker_count=workers)
                        futures[future] = (index, shard)
                    for future in as_completed(futures):
                        index, shard = futures[future]
                        try:
                            receipt = future.result()
                        except Exception as error:
                            receipt = {"completed": [], "device": str(devices[index]), "failures": [
                                {"task_hash": task["task_hash"], "status": "failed", "error": f"worker exited: {type(error).__name__}: {error}"}
                                for task in shard]}
                        receipts.append(receipt)
                        failures.extend(receipt["failures"])
                        failed_ids = {item["task_hash"] for item in receipt["failures"]}
                        for task in shard:
                            progress.report(task["task_hash"], "failed" if task["task_hash"] in failed_ids else "reduced", devices[index])
        completed = {task["task_hash"]: _completed_task(directory, task) for task in tasks}
        errors = {item["task_hash"]: item["error"] for item in failures}
        frames = []
        task_receipts = {}
        for task in tasks:
            resolved = completed[task["task_hash"]]
            if resolved is None:
                frames.append(pd.DataFrame(_status_rows(task, status="failed", reason=errors.get(task["task_hash"], "missing_task_completion"))))
                continue
            source_hash, data, marker = resolved
            frame = pd.read_parquet(data)
            if len(frame) != task["count"] or frame.seed.astype(int).tolist() != task["seeds"]:
                raise TheoryError("Completed counterfactual seed population differs")
            frames.append(frame)
            task_receipts[task["task_hash"]] = {"source_task_hash": source_hash, "path": str(data),
                                              "sha256": file_sha256(data), "rows": len(frame), "marker_sha256": file_sha256(marker)}
        if tasks[0]["identity"]["source_code"] != _source_code():
            raise TheoryError("Counterfactual measurement code changed before publication")
        frame = pd.concat(frames, ignore_index=True)
        keys = ["run_id", "original_index", "record_id", "target_id", "seed", "step_index"]
        if frame.duplicated(keys).any() or len(frame) != sum(task["count"] for task in tasks):
            raise TheoryError("Counterfactual collection changed its declared observation population")
        base_manifest = reject_symlinks(Path(result["directory"]) / "manifest.json")
        base_receipt = {"manifest": str(base_manifest), "sha256": file_sha256(base_manifest)}
        identity = {"policy": POLICY, "task_hashes": [task["task_hash"] for task in tasks],
                    "completed_tasks": {key: {name: spec[name] for name in ("source_task_hash", "sha256", "rows")}
                                        for key, spec in task_receipts.items()},
                    "base_analysis": scientific_base_receipt(result)}
        digest = canonical_hash(identity)
        collection = contained_path(directory, "collections/" + digest)
        saved = existing_collection(collection, identity=identity, kind=TABLE)
        if saved is not None:
            return {"tables": {**result["tables"], TABLE: frame}, "files": saved["files"],
                    "logical_joins": saved.get("logical_joins", {}), "provenance": saved["provenance"],
                    "directory": collection}
        data = collection / (TABLE + ".parquet")
        atomic_write_frame_parquet(frame, data)
        atomic_write_frame_parquet(pd.DataFrame(failures, columns=None if failures else ["task_hash", "status", "error"]), collection / "failed.parquet")
        files = {**result["files"], TABLE: {"path": str(data), "sha256": file_sha256(data), "rows": len(frame)}}
        incomplete = sum(value is None for value in completed.values())
        supplement = {"enabled": True, "complete": incomplete == 0, "measurement_hash": digest,
                      "fixed_steps": list(_fixed_steps(config)), "task_count": len(tasks),
                      "incomplete_tasks": incomplete, "rows": len(frame), "status_counts": frame.status.value_counts().to_dict(),
                      "policy": POLICY, "task_receipts": task_receipts,
                      "execution": {"devices": list(map(str, devices[:workers])), "workers": workers,
                                    "spawn": True, "parent_progress_bars": 1, "receipts": receipts,
                                    "execution_scope": "artifact_creation_run; immutable_on_resume"},
                      "missing_command": _missing_command(config) if incomplete else None}
        provenance = {**result["provenance"], "analysis_hash": digest,
                      "counterfactual_unconditional": supplement}
        joins = dict(result.get("logical_joins", {}))
        atomic_write_json(collection / "manifest.json", {"schema_version": SCHEMA_VERSION,
            "kind": TABLE, "complete": incomplete == 0, "identity": identity, "base_analysis": base_receipt,
            "files": files, "logical_joins": joins, "provenance": provenance})
        return {"tables": {**result["tables"], TABLE: frame}, "files": files, "logical_joins": joins,
                "provenance": provenance, "directory": collection}
