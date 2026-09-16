"""Separately resumable, additive four-stage scalar supplements.

Only missing record shards read protected vectors. Workers never load a model;
one parent progress bar covers all visible-GPU, spawn-based analytical workers.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
import traceback

import pandas as pd
import torch

from utils.common.io import atomic_torch_save, atomic_write_frame_parquet, atomic_write_json, canonical_hash, file_sha256, safe_torch_load
from utils.models.devices import configure_worker_cpu_threads, round_robin_shard, worker_count_for_tasks
from .contracts import TheoryError
from .evidence_reduce import KEYS, _complete, _join, _save, _sources, _verify_sources
from .paper_contracts import contained_path, publication_lock, recompute_command
from .progress import RecordProgress, install_progress_queue, report_worker_record
from .supplemental_cache import existing_collection, scientific_base_receipt

TABLES = ("trajectory_four_stage", "initial_four_stage", "branch_motion", "trajectory_shapes", "feedback_response")


def measurement_recipe(config=None):
    from .four_stage_measurements import FOUR_STAGE_VERSION, SHAPE_ATOL, SHAPE_RTOL
    config = {} if config is None else config
    return {"version": FOUR_STAGE_VERSION,
            "response_numerical_policy": {"backend": "cuda_outward_binary64", "mantissa_bits": 53,
                "max_products_per_seed_whole_dose_grid": int(config.get("numerical_max_decimal_products", 2_000_000)),
                "scope": "reduced_stored_logits_only",
                "selection": "unresolved_or_nonfinite_stable_gain_or_H_G_sign_disagreement"},
            "source_code": _sources("four_stage_measurements.py", "four_stage_reduce.py", "numerical_refinement.py", "gpu_intervals.py", "gpu_refinement.py", "metrics.py", "evidence_measurements.py", "supplemental_cache.py"),
            "dose_grid": "sorted(unique(j/40 for j=0..40 union 1/g))",
            "shape_atol_rmse": SHAPE_ATOL, "shape_rtol": SHAPE_RTOL,
            "initial_baseline": "genuine_unique_seed_gaussian_probes_with_repeated_prompt_vector_disagreement_audit"}



def collection_identity(result, recipe, task_hashes):
    """Scientific identity excludes publication paths and execution receipts."""
    return {"recipe": recipe, "base_scientific": scientific_base_receipt(result),
            "task_hashes": list(task_hashes)}


def record_task(core, record, law_digest, schedule_digest, recipe=None):
    """Public construction also permits endpoint-first orchestration reuse."""
    from .numerical_reduce import input_recipe
    core = Path(core)
    recipe = measurement_recipe() if recipe is None else recipe
    identity = {"recipe": recipe, "record": record.metadata,
                "law_sha256": law_digest, "schedule_sha256": schedule_digest,
                "core_identity": core.name}
    digest = canonical_hash(identity)
    input_identity = {"recipe": input_recipe(), "record": record.metadata,
                      "law_sha256": law_digest, "schedule_sha256": schedule_digest}
    return {"identity": identity, "task_hash": digest, "record_index": str(record.original_index),
            "path": str(core.parent.parent / "four_stage_measurements" / "records" / digest),
            "input_identity": input_identity,
            "input_path": str(core.parent.parent / "numerical_refinement" / "inputs" / canonical_hash(input_identity))}


def _worker(*, sources, records, tasks, core, law_path, law_digest, schedule_path, schedule_digest,
            config, device, worker_count, candidate_chunk_size, query_chunk_size, prepare_core=False):
    from .cache_reader import load_record, load_scores
    from .direct_reduce import (_paths, _read_law, _valid_record, _save_record, _validate_rows,
                                _cpu_payload, INTEGRATION_PAYLOAD_SCHEMA_VERSION)
    from .direct_measurements import measure_direct_record
    from .evidence_reference import gaussian_bank
    from .four_stage_measurements import measure_record
    from .numerical_reduce import INPUT_BATCH_SIZE, _input_complete
    from .scheduler_adapter import SchedulerAdapter
    configure_worker_cpu_threads(worker_count)
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    law = _read_law(Path(law_path), law_digest, device, candidate_chunk_size, query_chunk_size)
    if file_sha256(schedule_path) != schedule_digest:
        raise TheoryError("Saved schedule changed before four-stage measurements")
    schedule = safe_torch_load(schedule_path)
    science = sources.runs["experiment"]["scientific_config"]
    adapter = SchedulerAdapter(schedule, recorded_diffusers_version=science.get("package_versions", {}).get("diffusers"))
    bank = gaussian_bank(science, schedule)
    lookup = {str(record.original_index): record for record in records}
    receipts = []
    for task in tasks:
        status, log = "failed", None
        try:
            path = Path(task["path"])
            _verify_sources(task["identity"]["recipe"]["source_code"])
            record = lookup[task["record_index"]]
            if _complete(path, task["identity"]) and (not prepare_core or _valid_record(core, record, Path(core).name)):
                status = "resumed"
            else:
                source, files = _paths(core, record)
                if not _valid_record(core, record, Path(core).name):
                    if not prepare_core:
                        raise TheoryError("Compatible core observations are unavailable")
                    payloads = []
                    def capture(step, start, stop, payload, base_metrics):
                        name = f"payloads/step-{step:04d}-seeds-{start:06d}-{stop:06d}.pt"
                        atomic_torch_save(_cpu_payload({"schema_version": INTEGRATION_PAYLOAD_SCHEMA_VERSION,
                            "step_index": step, "seeds": record.metadata["seeds"][start:stop], "payload": payload}),
                            contained_path(source, name))
                        payloads.append(name)
                    log = load_record(sources.experiment, record)
                    with torch.inference_mode():
                        core_tables = measure_direct_record(log, record, law, config, adapter=adapter,
                            terminal_sscd=load_scores(sources, record), integration=False, payload_callback=capture)
                    _validate_rows(core_tables, record, config)
                    core_tables["audit"]["complete"] = core_tables["audit"].get("core_complete", False)
                    _save_record(source, files, core_tables, record, Path(core).name, payloads=payloads)
                    if not core_tables["audit"]["complete"]:
                        raise TheoryError("Fast core endpoint measurements incomplete; see retained core audit")
                tables = {name: pd.read_parquet(files[name]) for name in ("initial", "trajectory", "matched_updates")}
                seeds = list(map(str, record.metadata["seeds"]))
                def payload_loader(start, stop):
                    identity = {"record_inputs": task["input_identity"], "step_index": 0, "seeds": seeds[start:stop]}
                    saved = Path(task["input_path"]) / f"step-0000-seeds-{start:06d}-{stop:06d}"
                    return safe_torch_load(saved / "payload.pt") if _input_complete(saved, identity) else None
                def payload_saver(start, stop, payload):
                    identity = {"record_inputs": task["input_identity"], "step_index": 0, "seeds": seeds[start:stop]}
                    saved = Path(task["input_path"]) / f"step-0000-seeds-{start:06d}-{stop:06d}"
                    digest = atomic_torch_save(_cpu_payload(payload), saved / "payload.pt")
                    atomic_write_json(saved / "complete.json", {"complete": True, "identity": identity, "sha256": digest})
                if log is None:
                    log = load_record(sources.experiment, record)
                try:
                    with torch.inference_mode():
                        frames = measure_record(log, record, law, adapter, config, tables, gaussian_bank=bank,
                                                gaussian_initialization_compatible=float(schedule["init_noise_sigma"]) == 1.,
                                                payload_loader=payload_loader, payload_saver=payload_saver, batch_size=INPUT_BATCH_SIZE)
                finally:
                    log = None
                for name, base in (("trajectory_four_stage", "trajectory"), ("initial_four_stage", "initial")):
                    _join(tables[base], frames[name])
                _verify_sources(task["identity"]["recipe"]["source_code"])
                _save(path, task["identity"], frames,
                      {"complete": True, "raw_record_loads": 1, "new_model_evaluations": 0,
                       "endpoint_response_independent_of_variation_integration": True,
                       "precision": "saved_inputs_promoted_before_sensitive_float64_arithmetic"})
                status = "reduced"
            receipts.append({"task_hash": task["task_hash"], "status": status, "device": str(device)})
        except Exception as error:
            receipt = {"task_hash": task["task_hash"], "status": "failed", "device": str(device),
                       "reason": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()}
            atomic_write_json(Path(task["path"]) / "failure.json", receipt)
            receipts.append(receipt)
        finally:
            log = None
            report_worker_record(task["task_hash"], status, device)
    return receipts


def run_four_stage_analysis(project_root, *, result, config, device="auto", probe_batch_size=8,
                            candidate_chunk_size=256, query_chunk_size=16, allow_compute=True):
    from .cache_reader import discover_sources
    from .contracts import FOUR_STAGE_OPTION_KEYS
    from .four_stage_measurements import baseline_summary, prompt_shape_summaries
    from .numerical_reduce import _validate_backing
    from .reduce import _resolve_theory_devices, _worker_sources
    scientific = {name: value for name, value in config.items() if name not in FOUR_STAGE_OPTION_KEYS}
    sources = discover_sources(project_root, **scientific)
    provenance = result["provenance"]
    core = Path(provenance["analysis_manifest"]).parent.parent.parent
    law_path, schedule_path = core / "reference_law.pt", core / "schedule.pt"
    law_digest, schedule_digest = _validate_backing(law_path, schedule_path, provenance)
    recipe = measurement_recipe(scientific)
    tasks = [record_task(core, record, law_digest, schedule_digest, recipe) for record in sources.selected]
    if not tasks:
        raise TheoryError("No retained records for four-stage measurements")
    backing = core.parent.parent / "four_stage_measurements"
    with publication_lock(backing):
        pending = [task for task in tasks if not _complete(task["path"], task["identity"])]
        if pending and not allow_compute:
            raise TheoryError("Required four-stage endpoint/motion shards are missing; numerical-only mode does not backfill them. "
                              "Run " + recompute_command(scientific))
        devices = tuple(_resolve_theory_devices(device)) if pending else ()
        count = worker_count_for_tasks(devices, len(pending)) if pending else 0
        context, receipts = get_context("spawn"), []
        with RecordProgress(total=len(tasks), devices=count, context=context) as progress:
            pending_hashes = {task["task_hash"] for task in pending}
            for task in tasks:
                if task["task_hash"] not in pending_hashes:
                    progress.report(task["task_hash"], "resumed", "saved")
            if count:
                with ProcessPoolExecutor(max_workers=count, mp_context=context,
                                         initializer=install_progress_queue, initargs=(progress.queue,)) as executor:
                    futures = {}
                    for index in range(count):
                        shard = round_robin_shard(pending, worker_index=index, worker_count=count)
                        selected = {task["record_index"] for task in shard}
                        records = [record for record in sources.selected if str(record.original_index) in selected]
                        future = executor.submit(_worker, sources=_worker_sources(sources, records), records=records,
                            tasks=shard, core=core, law_path=law_path, law_digest=law_digest, schedule_path=schedule_path,
                            schedule_digest=schedule_digest, config=scientific, device=str(devices[index]), worker_count=count,
                            candidate_chunk_size=candidate_chunk_size, query_chunk_size=query_chunk_size)
                        futures[future] = index, shard
                    for future in as_completed(futures):
                        index, shard = futures[future]
                        try:
                            completed = future.result()
                        except Exception as error:
                            completed = [{"task_hash": task["task_hash"], "status": "failed", "device": str(devices[index]),
                                          "reason": f"{type(error).__name__}: {error}"} for task in shard]
                        receipts.extend(completed)
                        for item in completed:
                            progress.report(item["task_hash"], item["status"], item["device"])
        incomplete = [task for task in tasks if not _complete(task["path"], task["identity"])]
        if incomplete:
            failure = backing / "failed.json"
            atomic_write_json(failure, {"complete": False, "incomplete_records": len(incomplete), "receipts": receipts})
            raise TheoryError(f"{len(incomplete)} four-stage records incomplete; compatible shards preserved. See {failure}")
        frames = {name: pd.concat([pd.read_parquet(Path(task["path"]) / (name + ".parquet")) for task in tasks], ignore_index=True) for name in TABLES}
        tables = dict(result["tables"])
        tables["trajectory_metrics"] = _join(tables["trajectory"], frames["trajectory_four_stage"])
        tables["initial_generated_samples"] = _join(tables["initial"], frames["initial_four_stage"])
        frames["initial_baseline_summary"], frames["initial_baseline_disagreements"] = baseline_summary(
            tables["gaussian_reference"], tables["initial_generated_samples"], tables["reference_atoms"])
        gaussian_samples = tables["gaussian_conditional"].loc[tables["gaussian_conditional"].step_index.eq(0)].copy()
        native_reference = tables["gaussian_reference"].loc[tables["gaussian_reference"].step_index.eq(0)].copy()
        reference_names = {"learned_mean_error_l2": "gaussian_unconditional_mean_error_l2",
            "learned_mean_error_rmse": "gaussian_unconditional_mean_error_rmse",
            "unconditional_reference_error_l2": "gaussian_unconditional_reference_error_l2",
            "unconditional_reference_error_rmse": "gaussian_unconditional_reference_error_rmse",
            "reference_mean_error_l2": "gaussian_reference_mean_error_l2",
            "reference_mean_error_rmse": "gaussian_reference_mean_error_rmse",
            "input_source": "gaussian_reference_input_source"}
        join_keys = ["run_id", "seed", "step_index"]
        reference_columns = [name for name in reference_names if name in native_reference]
        native_reference = native_reference[join_keys + reference_columns].rename(columns=reference_names)
        for key in ("run_id", "seed"):
            gaussian_samples[key] = gaussian_samples[key].astype(str)
            native_reference[key] = native_reference[key].astype(str)
        frames["initial_samples"] = gaussian_samples.merge(native_reference, how="left", on=join_keys,
                                                            validate="many_to_one", indicator=True, sort=False)
        frames["initial_samples"]["initial_reference_alignment_status"] = frames["initial_samples"]._merge.map({
            "both": "matched_unique_genuine_Gaussian_seed_reference", "left_only": "unavailable_Gaussian_reference_seed",
            "right_only": "unexpected_reference_only"}).astype(str)
        frames["initial_samples"] = frames["initial_samples"].drop(columns="_merge")
        frames["trajectory_shape_prompts"], frames["trajectory_shape_mixed_prompts"] = prompt_shape_summaries(frames["trajectory_shapes"])
        frames["reference_law"] = tables["reference_atoms"].copy()
        for name in ("mean_norm_l2", "mean_norm_rmse", "bank_mean_norm_l2", "bank_mean_norm_rmse", "mean_offset_l2", "mean_offset_rmse", "theory_mean_source", "theory_mean_sha256"):
            if name in frames["initial_baseline_summary"]:
                frames["reference_law"][name] = frames["initial_baseline_summary"][name].iloc[0]
        radius_columns = [name for name in ("candidate_target_atom_id", "target_id", "direct_target_radius_l2", "reference_law_hash", "latent_dimension") if name in tables["trajectory"]]
        frames["reference_target_radii"] = tables["trajectory"][radius_columns].drop_duplicates().reset_index(drop=True)
        base_receipt = {"manifest": str(Path(result["directory"]) / "manifest.json"),
                        "sha256": file_sha256(Path(result["directory"]) / "manifest.json")}
        identity = collection_identity(result, recipe, [task["task_hash"] for task in tasks])
        digest = canonical_hash(identity)
        collection = backing / "collections" / digest
        _verify_sources(recipe["source_code"])
        saved_collection = existing_collection(collection, identity=identity, kind="four_stage_measurements")
        if saved_collection is not None:
            # Preserve immutable scalar bytes and the original execution receipt;
            # a CPU resume cannot rewrite a previous multi-GPU collection.
            tables.update(frames)
            tables["feedback_endpoints"], tables["terminal_metrics"] = tables["matched_updates"], tables["terminal"]
            return {"tables": tables, "files": saved_collection["files"],
                    "logical_joins": saved_collection.get("logical_joins", {}),
                    "provenance": saved_collection["provenance"], "directory": collection,
                    "base_numerical_directory": result["directory"]}
        files = dict(result["files"])
        for name, frame in frames.items():
            path = contained_path(collection, name + ".parquet")
            atomic_write_frame_parquet(frame, path)
            files[name] = {"path": str(path), "sha256": file_sha256(path), "rows": len(frame)}
            tables[name] = frame
        joins = dict(result.get("logical_joins", {}))
        joins["trajectory_metrics"] = {"base": "trajectory", "supplement": "trajectory_four_stage", "keys": KEYS, "how": "exact_one_to_one_additive"}
        joins["initial_generated_samples"] = {"base": "initial", "supplement": "initial_four_stage", "keys": KEYS, "how": "exact_one_to_one_additive"}
        tables["feedback_endpoints"], tables["terminal_metrics"] = tables["matched_updates"], tables["terminal"]
        updated = {**provenance, "analysis_hash": digest, "four_stage_measurements": {"recipe": recipe, "base_numerical": base_receipt,
                   "reference_law_manifest": provenance["reference_law"],
                   "reference_law_backing_receipt": {"path": str(law_path), "sha256": law_digest,
                       "law_hash": provenance["reference_law_hash"],
                       "mean_vector_sha256": provenance["reference_law"].get("mean_sha256"),
                       "mean_vector_definition": "exact finite-bank sum_j saved_positive_weight_j * full_saved_atom_vector_j; distinct from selected theory mu",
                       "mean_storage": "reuse_immutable_atoms_weights_receipt_no_duplicate_bank_mean_vector",
                       "theory_mean": provenance["reference_law"].get("theory_mean", {}),
                       "theory_mean_storage": "separately identified selected centre retained in the reference-law backing payload",
                       "aliases_and_membership": "complete_reference_law_manifest",
                       "radii_table": "reference_target_radii_full_support_maxima"},
                   "record_shards": [task["path"] for task in tasks],
                   "devices": list(map(str, devices[:count])), "worker_count": count, "spawn": True, "parent_progress_bars": 1,
                   "complete": True, "optional_counterfactual_required": False}}
        _verify_sources(recipe["source_code"])
        atomic_write_json(collection / "manifest.json", {"schema_version": 1, "kind": "four_stage_measurements", "complete": True,
                          "identity": identity, "files": files, "logical_joins": joins, "provenance": updated})
        return {"tables": tables, "files": files, "logical_joins": joins, "provenance": updated, "directory": collection,
                "base_numerical_directory": result["directory"]}


def prepare_four_stage_primary(project_root, *, config, device="auto", probe_batch_size=8,
                               candidate_chunk_size=256, query_chunk_size=16):
    """Persist fast core/response/motion shards before any costly integration.

    Uses unchanged legacy core construction and recipe, preserving its identity.
    Workers finish before the normal learned-probe/integration stages begin.
    This function does not declare the full mandatory suite complete.
    """
    from .cache_reader import discover_sources, load_schedule
    from .contracts import FOUR_STAGE_OPTION_KEYS
    from .direct_reduce import core_recipe, _record_key, _valid_record
    from .paper_contracts import PaperPaths
    from .reference_law import build_reference_law
    from .reduce import _resolve_theory_devices, _worker_sources
    from .support import _tensor_hash
    scientific = {name: value for name, value in config.items() if name not in FOUR_STAGE_OPTION_KEYS}
    sources = discover_sources(project_root, **scientific)
    records = sources.selected
    if not records:
        raise TheoryError("No retained complete records for endpoint-first four-stage preparation")
    primary_device = _resolve_theory_devices(device)[0]
    schedule = load_schedule(sources)
    law = build_reference_law(sources, {**scientific, "candidate_chunk_size": candidate_chunk_size,
                                      "query_chunk_size": query_chunk_size}, device=primary_device,
                              allow_mean_compute=True, mean_device=device, mean_batch_size=probe_batch_size)
    identity = {"recipe": core_recipe(), "reference_law_hash": law.law_hash,
                "source_metadata_hash": sources.metadata_hash(), "record_hashes": [_record_key(record) for record in records],
                "guidance_scale": scientific["guidance_scale"], "target_error_tolerance": scientific.get("target_error_tolerance")}
    digest = canonical_hash(identity)
    paper = PaperPaths.build(project_root, **scientific).output_directory
    backing = contained_path(paper.parent.parent, "theory_measurements")
    core = contained_path(backing, "direct_analysis/" + digest)
    law_path, schedule_path = core / "reference_law.pt", core / "schedule.pt"
    with publication_lock(backing / "four_stage_measurements"):
        if law_path.is_file():
            saved = safe_torch_load(law_path)
            if (saved.get("law_hash") != law.law_hash or _tensor_hash(saved["atoms"]) != _tensor_hash(law.support.atoms)
                    or not torch.equal(saved["weights"].double(), law.support.weights.cpu().double())):
                raise TheoryError("Existing fixed-law core backing differs from endpoint-first identity")
            law_digest = file_sha256(law_path)
        else:
            law_digest = atomic_torch_save(law.to_payload(), law_path)
        if schedule_path.is_file():
            # The schedule's complete mapping contains tensors; compare the
            # canonical existing sampler fingerprint rather than Python ==.
            from .cache_reader import load_schedule as _load_schedule
            existing = safe_torch_load(schedule_path)
            fresh = _load_schedule(sources)
            if set(existing) != set(fresh):
                raise TheoryError("Saved scheduler backing keys differ")
            for name, value in fresh.items():
                if isinstance(value, torch.Tensor):
                    if not isinstance(existing[name], torch.Tensor) or not torch.equal(existing[name], value):
                        raise TheoryError("Saved scheduler backing tensor differs: " + name)
                elif existing[name] != value:
                    raise TheoryError("Saved scheduler backing coefficient differs: " + name)
            schedule_digest = file_sha256(schedule_path)
        else:
            schedule_digest = atomic_torch_save(schedule, schedule_path)
        recipe = measurement_recipe(scientific)
        tasks = [record_task(core, record, law_digest, schedule_digest, recipe) for record in records]
        pending = [task for task, record in zip(tasks, records, strict=True)
                   if not _complete(task["path"], task["identity"]) or not _valid_record(core, record, digest)]
        devices = tuple(_resolve_theory_devices(device)) if pending else ()
        count = worker_count_for_tasks(devices, len(pending)) if pending else 0
        context, receipts = get_context("spawn"), []
        with RecordProgress(total=len(tasks), devices=count, context=context) as progress:
            pending_hashes = {task["task_hash"] for task in pending}
            for task in tasks:
                if task["task_hash"] not in pending_hashes:
                    progress.report(task["task_hash"], "resumed", "saved")
            if count:
                with ProcessPoolExecutor(max_workers=count, mp_context=context,
                                         initializer=install_progress_queue, initargs=(progress.queue,)) as executor:
                    futures = {}
                    for index in range(count):
                        shard = round_robin_shard(pending, worker_index=index, worker_count=count)
                        ids = {task["record_index"] for task in shard}
                        selected = [record for record in records if str(record.original_index) in ids]
                        future = executor.submit(_worker, sources=_worker_sources(sources, selected), records=selected,
                            tasks=shard, core=core, law_path=law_path, law_digest=law_digest, schedule_path=schedule_path,
                            schedule_digest=schedule_digest, config=scientific, device=str(devices[index]), worker_count=count,
                            candidate_chunk_size=candidate_chunk_size, query_chunk_size=query_chunk_size, prepare_core=True)
                        futures[future] = index, shard
                    for future in as_completed(futures):
                        index, shard = futures[future]
                        try:
                            completed = future.result()
                        except Exception as error:
                            completed = [{"task_hash": task["task_hash"], "status": "failed", "device": str(devices[index]),
                                          "reason": f"{type(error).__name__}: {error}"} for task in shard]
                        receipts.extend(completed)
                        for item in completed:
                            progress.report(item["task_hash"], item["status"], item["device"])
        missing = [task for task in tasks if not _complete(task["path"], task["identity"])]
        complete = not missing and all(_valid_record(core, record, digest) for record in records)
        status_path = backing / "four_stage_measurements" / "primary_preparation.json"
        atomic_write_json(status_path, {"fast_primary_complete": complete, "mandatory_analysis_complete": False,
                          "core_hash": digest, "tasks": [task["task_hash"] for task in tasks], "receipts": receipts,
                          "remaining_stages": ["genuine_missing_probes", "original_variation_integration", "precision_refinement", "terminal_supplements", "publication"]})
        if not complete:
            raise TheoryError(f"Fast four-stage primary preparation incomplete; compatible shards retained. See {status_path}")
        return {"core_directory": core, "task_hashes": [task["task_hash"] for task in tasks], "status": "fast_primary_complete_full_suite_pending"}
