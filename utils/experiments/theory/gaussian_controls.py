"""Separately cached genuine Gaussian unconditional controls when absent.

One empty-prompt inference per unique initial seed is shared across all retained
pair targets. Only scalar target errors and input/precision receipts are saved;
there is no new trajectory, image decoding, or stored prediction-vector bank.
Verified existing controls are retained verbatim in the augmented logical view.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
import math

import pandas as pd
import torch

from utils.common.io import (atomic_write_frame_parquet, atomic_write_json, canonical_hash,
                             file_sha256, safe_torch_load)
from utils.models.devices import configure_worker_cpu_threads, round_robin_shard, worker_count_for_tasks
from utils.models.loading import package_version_metadata
from utils.models.sampling import encode_prompt_condition
from .contracts import FOUR_STAGE_OPTION_KEYS, TheoryError
from .direct_probes import (_completed_task, _load_replica, _pair_keys, _positive_integer,
                            _rows, _task_paths, _tensor_digest, prediction_batches)
from .evidence_reduce import KEYS
from .paper_contracts import PaperPaths, contained_path, publication_lock, recompute_command, reject_symlinks
from .progress import RecordProgress, install_progress_queue, report_worker_record
from .supplemental_cache import existing_collection, scientific_base_receipt

TABLE = "gaussian_control_backfill"
SCHEMA_VERSION = 1
VALID_CONTROL_STATUSES = {"same_saved_gaussian_input_canonical_epsilon", "measured_independent_gaussian_control"}
TARGET_BATCH_SIZE = 16
POLICY = {
    "version": "genuine-gaussian-unconditional-controls-1",
    "input_law": "true_gaussian_initialization_probe",
    "bank": "same_generation_CPU_float32_seed_bank_and_saved_quantization_as_direct_gaussian_probes",
    "prediction": "one_fresh_empty_prompt_inference_per_unique_seed_shared_across_all_targets",
    "conversion": "native_to_canonical_epsilon_once_against_unscaled_conceptual_input",
    "reduction": "float64_target_L2_divided_by_sqrt_dimension_once",
    "retained_observations": "all_fixed_pair_targets_and_complete_evaluation_seed_bank_no_outcome_filter",
    "overlay": "fill_only_missing_or_unverified_initial_controls; existing_verified_scalars_unchanged",
    "terminal_scores": "actual_generated_pair_outcome_remains_separate_from_Gaussian_probe_identity",
}


def _source_code():
    base = Path(__file__).parent
    paths = [base / name for name in ("gaussian_controls.py", "supplemental_cache.py", "direct_probes.py", "direct_probe_cache.py", "evidence_reference.py")]
    paths += [base.parent.parent / "models" / name for name in (
        "probe_loading.py", "loading.py", "sampling.py", "prediction_conversion.py", "schedulers.py")]
    return {str(path.relative_to(base.parent.parent)): file_sha256(path) for path in paths}


def missing_initial_controls(frame):
    if not set(KEYS) <= set(frame):
        raise TheoryError("Genuine Gaussian conditional rows lack complete logical identities")
    if frame.duplicated(KEYS).any():
        raise TheoryError("Duplicate genuine Gaussian conditional identities")
    initial = frame.step_index.eq(0)
    value = pd.to_numeric(frame.get("unconditional_target_error_rmse", pd.Series(math.nan, index=frame.index)), errors="coerce")
    status = frame.get("gaussian_control_status", pd.Series("", index=frame.index)).astype(str)
    finite = value.notna() & value.abs().ne(math.inf) & value.ge(0)
    return initial & ~(finite & status.isin(VALID_CONTROL_STATUSES))


def overlay_gaussian_controls(base, supplement):
    """Exact scalar-key overlay; never alter an existing verified observation."""
    result = base.copy()
    missing = missing_initial_controls(result)
    if not missing.any():
        return result
    if not set(KEYS) <= set(supplement) or supplement.duplicated(KEYS).any():
        raise TheoryError("Genuine Gaussian controls lack unique scalar identities")
    lookup = {tuple(str(row[key]) for key in KEYS): row for row in supplement.to_dict("records")}
    new_fields = ("unconditional_target_error_rmse", "unconditional_target_error_l2", "gaussian_control_status",
                  "gaussian_control_input_source", "gaussian_control_input_sha256", "gaussian_control_task_hash",
                  "gaussian_control_prediction_source", "gaussian_control_inference_dtype", "gaussian_control_conversion_dtype")
    for name in new_fields:
        if name not in result:
            result[name] = math.nan if name in {"unconditional_target_error_rmse", "unconditional_target_error_l2"} else None
    for index in result.index[missing]:
        key = tuple(str(result.at[index, name]) for name in KEYS)
        row = lookup.get(key)
        if row is None or row.get("gaussian_control_status") != "measured_independent_gaussian_control":
            raise TheoryError("A required genuine Gaussian control is absent or incomplete")
        expected_input = result.at[index, "input_sha256"] if "input_sha256" in result else None
        if pd.notna(expected_input) and str(expected_input) != str(row["gaussian_control_input_sha256"]):
            raise TheoryError("Genuine Gaussian control input differs from its conditional probe")
        for name in new_fields:
            result.at[index, name] = row[name]
    # terminal_sscd_matched_initialization is deliberately untouched: a fresh
    # Gaussian predictor does not become the recorded generation trajectory.
    return result


def plan_control_tasks(sources, records, schedule):
    science = sources.runs["experiment"]["scientific_config"]
    if science.get("stored_prediction_type") != "epsilon":
        raise TheoryError("Gaussian controls require already canonical saved epsilon contracts")
    native = int(torch.as_tensor(schedule["timesteps"])[0])
    cumulative = float(torch.as_tensor(schedule["alphas_cumprod_t"])[0])
    if not 0 < cumulative < 1:
        raise TheoryError("Genuine Gaussian controls need a positive initial signal and noise label")
    seeds = list(map(int, science["seeds"]))
    if not seeds or len(seeds) != len(set(seeds)):
        raise TheoryError("Gaussian controls require unique evaluation seeds")
    pairs = []
    for record in sorted(records, key=lambda record: str(record.original_index)):
        if list(map(int, record.metadata["seeds"])) != seeds:
            raise TheoryError("Gaussian controls must preserve the full evaluation seed bank")
        pairs.append(_pair_keys(sources, record))
    if not pairs:
        raise TheoryError("Gaussian controls require every frozen retained pair")
    versions = package_version_metadata()
    shared = {"table": TABLE, "count": len(pairs), "policy": POLICY, "source_code": _source_code(),
              "checkpoint": {name: science[name] for name in ("model_id", "model_revision", "native_prediction_type", "inference_dtype")},
              "pairs": pairs, "latent_shape": science["latent_shape"],
              "target_preprocessing": science.get("target_preprocessing"), "vae_id": science.get("vae_id"),
              "vae_revision": science.get("vae_revision"), "storage_contract": science.get("scientific_tensor_storage"),
              "schedule_sha256": file_sha256(sources.experiment.schedule),
              "initialization_noise_sigma": float(schedule["init_noise_sigma"]),
              "level": {"step_index": 0, "timestep": native, "alpha": math.sqrt(cumulative), "sigma": math.sqrt(1 - cumulative)},
              "runtime_packages": {name: versions[name] for name in ("torch", "diffusers", "transformers")}}
    return [{"table": TABLE, "count": len(pairs), "seed": seed, "identity": shared | {"seed": seed},
             "task_hash": canonical_hash(shared | {"seed": seed})} for seed in seeds]


def _worker(*, sources, records, tasks, output_directory, device, batch_size, worker_count):
    from .evidence_reference import gaussian_bank

    configure_worker_cpu_threads(worker_count)
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if tasks[0]["identity"]["source_code"] != _source_code():
        raise TheoryError("Gaussian control measurement sources changed")
    path = sources.experiment.schedule
    if file_sha256(path) != tasks[0]["identity"]["schedule_sha256"]:
        raise TheoryError("Preserved Gaussian-control scheduler changed")
    schedule = safe_torch_load(path)
    science = sources.runs["experiment"]["scientific_config"]
    # Input draws retain the generation RNG convention; reductions use CUDA.
    bank = gaussian_bank(science, schedule).to(device)
    seed_positions = {int(seed): index for index, seed in enumerate(science["seeds"])}
    tasks = [task for task in tasks if _completed_task(output_directory, task) is None]
    if not tasks:
        return {"completed": [], "failures": [], "device": str(device), "replica": {}, "network_observations": 0}
    samples = bank[[seed_positions[task["seed"]] for task in tasks]]
    records = {str(record.original_index): record for record in records}
    pairs, level = tasks[0]["identity"]["pairs"], tasks[0]["identity"]["level"]
    components = scheduler = condition = None
    completed, failures, replica = [], [], {}
    network_observations = 0
    try:
        components, scheduler = _load_replica(science, schedule, device)
        condition = encode_prompt_condition("", components.tokenizer, components.text_encoder, device, components.inference_dtype)
        replica = components.device_metadata | {"package_versions": components.package_versions}
        for start, stop, epsilon, precision in prediction_batches(
                samples, condition=condition, timestep=level["timestep"], components=components,
                scheduler=scheduler, batch_size=batch_size):
            current_tasks = tasks[start:stop]
            network_observations += stop - start
            clean = (samples[start:stop].double() - level["sigma"] * epsilon.double()) / level["alpha"]
            precision_rows = _rows(precision, stop - start, {})
            rows = [[] for _ in current_tasks]
            input_hashes = [_tensor_digest(value) for value in samples[start:stop]]
            for target_start in range(0, len(pairs), TARGET_BATCH_SIZE):
                selected = pairs[target_start:target_start + TARGET_BATCH_SIZE]
                targets = []
                for pair in selected:
                    record = records[str(pair["original_index"])]
                    target_path = sources.experiment.target_latent_path(record.original_index)
                    if file_sha256(target_path) != pair["target_latent_sha256"]:
                        raise TheoryError("Preserved Gaussian-control target hash differs")
                    target = safe_torch_load(target_path).to(device)
                    if list(target.shape) != science["latent_shape"] or not bool(torch.isfinite(target).all()):
                        raise TheoryError("Preserved Gaussian-control target latent is invalid")
                    targets.append(target.double())
                target_bank = torch.stack(targets)
                errors = (clean[:, None] - target_bank[None]).flatten(2).norm(dim=2)
                if not bool(torch.isfinite(errors).all()):
                    raise TheoryError("Nonfinite genuine Gaussian target-error reduction")
                dimension = target_bank[0].numel()
                for local, item in enumerate(current_tasks):
                    for column, pair in enumerate(selected):
                        error = float(errors[local, column])
                        rows[local].append({**pair, **level, "seed": item["seed"], "task_hash": item["task_hash"],
                            "unconditional_target_error_l2": error, "unconditional_target_error_rmse": error / math.sqrt(dimension),
                            "gaussian_control_status": "measured_independent_gaussian_control",
                            "gaussian_control_input_source": POLICY["input_law"], "input_source": POLICY["input_law"],
                            "gaussian_control_input_sha256": input_hashes[local], "gaussian_control_task_hash": item["task_hash"],
                            "gaussian_control_prediction_source": "fresh_empty_prompt_canonical_epsilon_once",
                            "gaussian_control_inference_dtype": science["inference_dtype"],
                            "gaussian_control_conversion_dtype": precision_rows[local]["prediction_conversion_dtype"],
                            "latent_dimension": dimension, "model_revision": science["model_revision"],
                            "native_prediction_type": science["native_prediction_type"],
                            "terminal_score_scope": "not_the_generated_input; actual_pair_terminal_outcome_is_separate",
                            **precision_rows[local]})
            for item, scalar_rows in zip(current_tasks, rows, strict=True):
                if len(scalar_rows) != item["count"]:
                    raise TheoryError("Gaussian control task omitted a retained pair")
                data, marker = _task_paths(output_directory, item)
                atomic_write_frame_parquet(pd.DataFrame(scalar_rows), data)
                atomic_write_json(marker, {"schema_version": SCHEMA_VERSION, "task_hash": item["task_hash"],
                    "identity": item["identity"], "rows": len(scalar_rows), "complete": True, "sha256": file_sha256(data),
                    "execution": {"device": str(device), "numeric_backend": "worker_device_float64", "random_stream_backend": "preserved_CPU_seeded_input_draws", "batch_size_requested": batch_size, "replica": replica,
                                  "network_observations": 1, "shared_target_count": len(pairs)}})
                completed.append(item["task_hash"])
                report_worker_record(item["task_hash"], "reduced", device)
    except Exception as error:
        for item in tasks:
            if _completed_task(output_directory, item) is None:
                failures.append({"task_hash": item["task_hash"], "seed": item["seed"], "status": "failed",
                                 "error": f"{type(error).__name__}: {error}"})
                report_worker_record(item["task_hash"], "failed", device)
    finally:
        samples = components = scheduler = condition = None
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return {"completed": completed, "failures": failures, "device": str(device), "replica": replica,
            "network_observations": network_observations}


def run_gaussian_control_analysis(project_root, *, result, config, device="auto", probe_batch_size=8,
                                  allow_compute=True, **_unused_options):
    """Fill only absent genuine initial controls, sharing inference across targets."""
    base = result["tables"]["gaussian_conditional"]
    missing = missing_initial_controls(base)
    if not missing.any():
        return result
    from .cache_reader import discover_sources, load_schedule
    from .reduce import _resolve_theory_devices, _worker_sources

    scientific = {key: value for key, value in config.items() if key not in FOUR_STAGE_OPTION_KEYS}
    batch_size = _positive_integer(probe_batch_size, "probe_batch_size")
    sources = discover_sources(project_root, **scientific)
    records = sorted(sources.selected, key=lambda record: str(record.original_index))
    schedule = load_schedule(sources)
    required_seeds = set(map(int, base.loc[missing, "seed"]))
    tasks = [task for task in plan_control_tasks(sources, records, schedule)
             if task["seed"] in required_seeds]
    if {task["seed"] for task in tasks} != required_seeds:
        raise TheoryError("Missing Gaussian controls refer to seeds outside the frozen evaluation bank")
    paper = PaperPaths.build(project_root, **scientific).output_directory
    directory = contained_path(paper.parent.parent, "theory_measurements/gaussian_controls")
    pending = [task for task in tasks if _completed_task(directory, task) is None]
    if pending and not allow_compute:
        raise TheoryError("Genuine initial Gaussian control shards are missing; cache-only refinement cannot run inference. Run " + recompute_command(scientific))
    failures, receipts = [], []
    with publication_lock(directory):
        pending = [task for task in tasks if _completed_task(directory, task) is None]
        if pending and not allow_compute:
            raise TheoryError("Gaussian control receipts changed during cache-only resume. Run " + recompute_command(scientific))
        devices = tuple(_resolve_theory_devices(device)) if pending else ()
        workers = worker_count_for_tasks(devices, len(pending)) if pending else 0
        context = get_context("spawn")
        with RecordProgress(total=len(tasks), devices=workers, context=context) as progress:
            pending_ids = {task["task_hash"] for task in pending}
            for item in tasks:
                if item["task_hash"] not in pending_ids:
                    progress.report(item["task_hash"], "resumed", "saved")
            if workers:
                with ProcessPoolExecutor(max_workers=workers, mp_context=context,
                        initializer=install_progress_queue, initargs=(progress.queue,)) as executor:
                    futures = {}
                    for index in range(workers):
                        shard = round_robin_shard(pending, worker_index=index, worker_count=workers)
                        future = executor.submit(_worker, sources=_worker_sources(sources, records), records=records,
                            tasks=shard, output_directory=directory, device=str(devices[index]),
                            batch_size=batch_size, worker_count=workers)
                        futures[future] = (index, shard)
                    for future in as_completed(futures):
                        index, shard = futures[future]
                        try:
                            receipt = future.result()
                        except Exception as error:
                            receipt = {"completed": [], "device": str(devices[index]), "failures": [
                                {"task_hash": item["task_hash"], "seed": item["seed"], "status": "failed", "error": f"worker exited: {type(error).__name__}: {error}"}
                                for item in shard]}
                        receipts.append(receipt)
                        failures.extend(receipt["failures"])
                        failed_ids = {item["task_hash"] for item in receipt["failures"]}
                        for item in shard:
                            progress.report(item["task_hash"], "failed" if item["task_hash"] in failed_ids else "reduced", devices[index])
        completed = {item["task_hash"]: _completed_task(directory, item) for item in tasks}
        if any(value is None for value in completed.values()):
            failure_path = directory / "failed.json"
            atomic_write_json(failure_path, {"complete": False, "failures": failures,
                "missing_tasks": [key for key, value in completed.items() if value is None]})
            raise TheoryError(f"Genuine Gaussian-control inference incomplete; compatible seed shards retained. See {failure_path}. Run {recompute_command(scientific)}")
        if tasks[0]["identity"]["source_code"] != _source_code():
            raise TheoryError("Gaussian-control measurement source changed before publication")
        parts, source_receipts = [], {}
        for item in tasks:
            source_hash, data, marker = completed[item["task_hash"]]
            frame = pd.read_parquet(data)
            if len(frame) != item["count"] or set(frame.seed.astype(int)) != {item["seed"]}:
                raise TheoryError("Gaussian-control seed shard population differs")
            parts.append(frame)
            source_receipts[item["task_hash"]] = {"source_task_hash": source_hash, "path": str(data),
                "sha256": file_sha256(data), "rows": len(frame), "marker_sha256": file_sha256(marker)}
        supplement = pd.concat(parts, ignore_index=True)
        updated_conditional = overlay_gaussian_controls(base, supplement)
        base_manifest = reject_symlinks(Path(result["directory"]) / "manifest.json")
        base_receipt = {"manifest": str(base_manifest), "sha256": file_sha256(base_manifest)}
        identity = {"policy": POLICY, "base_analysis": scientific_base_receipt(result),
                    "task_receipts": {key: {name: spec[name] for name in ("source_task_hash", "sha256", "rows")}
                                      for key, spec in source_receipts.items()}}
        digest = canonical_hash(identity)
        collection = contained_path(directory, "collections/" + digest)
        saved = existing_collection(collection, identity=identity, kind=TABLE)
        if saved is not None:
            return {"tables": {**result["tables"], "gaussian_conditional": updated_conditional, TABLE: supplement},
                    "files": saved["files"], "logical_joins": saved.get("logical_joins", {}),
                    "provenance": saved["provenance"], "directory": collection}
        path = collection / (TABLE + ".parquet")
        atomic_write_frame_parquet(supplement, path)
        files = {**result["files"], TABLE: {"path": str(path), "sha256": file_sha256(path), "rows": len(supplement)}}
        joins = dict(result.get("logical_joins", {}))
        previous = joins.get("gaussian_conditional")
        joins["gaussian_conditional"] = {"base": "gaussian_conditional", "supplement": TABLE, "keys": KEYS,
            "how": "fill_missing_initial_gaussian_controls", "prior_join": previous,
            "existing_verified_values_unchanged": True, "generated_terminal_identity_unchanged": True}
        provenance = {**result["provenance"], "analysis_hash": digest, "gaussian_control_backfill": {
            "complete": True, "policy": POLICY, "base_analysis": base_receipt,
            "new_control_rows": int(missing.sum()), "unique_seed_tasks": len(tasks),
            "saved_scalar_rows": len(supplement), "measurement_hash": digest, "task_receipts": source_receipts,
            "execution": {"devices": list(map(str, devices[:workers])), "workers": workers,
                          "spawn": True, "parent_progress_bars": 1, "receipts": receipts,
                          "submitted_seed_tasks": len(pending), "resumed_seed_tasks": len(tasks) - len(pending),
                          "network_observations_this_run": (sum(item["network_observations"] for item in receipts)
                              if all("network_observations" in item for item in receipts) else None),
                          "execution_scope": "artifact_creation_run; immutable_on_resume"}}}
        atomic_write_json(collection / "manifest.json", {"schema_version": SCHEMA_VERSION, "kind": TABLE,
            "complete": True, "identity": identity, "base_analysis": base_receipt,
            "files": files, "logical_joins": joins, "provenance": provenance})
        return {"tables": {**result["tables"], "gaussian_conditional": updated_conditional, TABLE: supplement},
                "files": files, "logical_joins": joins, "provenance": provenance, "directory": collection}
