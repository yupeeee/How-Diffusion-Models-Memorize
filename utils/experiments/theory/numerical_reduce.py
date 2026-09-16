"""Cache-only, separately resumable precision stage for the fixed evidence suite.

Observation caches remain authoritative. This stage writes additive scalar
columns and compact sufficient tensors, never another copy of raw histories.
A changed precision budget changes policy receipts, not model/probe identities.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
import hashlib
import inspect
import math
import traceback

import pandas as pd
import torch

from utils.common.io import (atomic_torch_save, atomic_write_frame_parquet, atomic_write_json,
                             canonical_hash, file_sha256, safe_torch_load)
from utils.models.devices import configure_worker_cpu_threads, round_robin_shard, worker_count_for_tasks
from .contracts import NUMERICAL_KEYS, TheoryError, numerical_config, read_object
from .evidence_reduce import KEYS, _complete, _join, _save, _sources, _verify_sources
from .paper_contracts import PaperPaths, contained_path, publication_lock, recompute_command, reject_symlinks
from .progress import RecordProgress, install_progress_queue, report_worker_record

NUMERICAL_COLLECTION_VERSION = 1
INPUT_BATCH_SIZE = 16


def _science_without_policy(config):
    return {key: value for key, value in numerical_config(**config).items() if key not in NUMERICAL_KEYS}


def _missing(config, reason):
    return TheoryError(f"Numerical refinement requires compatible saved scientific inputs: {reason}. "
                       f"No upstream work was invoked. Run {recompute_command(config)}")


def _read_tables(manifest):
    frames = {}
    for name, spec in manifest["files"].items():
        path = reject_symlinks(Path(spec["path"]).absolute())
        if not path.is_file() or file_sha256(path) != spec["sha256"]:
            raise TheoryError(f"Saved evidence scalar receipt differs: {path}")
        frames[name] = pd.read_parquet(path)
        if len(frames[name]) != spec["rows"]:
            raise TheoryError(f"Saved evidence scalar row count differs: {path}")
    for name, rule in manifest.get("logical_joins", {}).items():
        if rule.get("keys") != KEYS:
            raise TheoryError("Unknown saved evidence logical key contract")
        base, extra = frames[rule["base"]], frames[rule["supplement"]]
        if rule["how"] == "exact_one_to_one_additive":
            frames[name] = _join(base, extra)
        elif rule["how"] == "left_one_to_one" and name == "gaussian_conditional":
            base, extra = base.copy(), extra.copy()
            for key in KEYS:
                if key != "step_index":
                    base[key], extra[key] = base[key].astype(str), extra[key].astype(str)
            frames[name] = base.merge(extra, how="left", on=KEYS, validate="one_to_one", sort=False)
            eligible = frames[name].terminal_sscd_matched_initialization.fillna(False).astype(bool)
            frames[name].loc[~eligible, "unconditional_target_error_rmse"] = math.nan
            frames[name].loc[~eligible, "gaussian_control_status"] = "unavailable_not_same_saved_gaussian_input"
        else:
            raise TheoryError("Unknown saved evidence join contract")
    return frames


def load_saved_evidence(project_root, *, config, sources):
    """Follow the active publication's exact base receipt; never select by result.

    Only small scalar and metadata files are read here. The worker later verifies
    raw tensor receipts if a numerical task actually needs its original inputs.
    """
    paper = PaperPaths.build(project_root, **config).output_directory
    try:
        saved = read_object(paper / "run_config.json")
        if _science_without_policy(saved["scientific_config"]) != _science_without_policy(config):
            raise TheoryError("scientific settings changed beyond numerical policy")
        directory = reject_symlinks(Path(saved["source_analysis"]["path"]).absolute())
        manifest = read_object(directory / "manifest.json")
        if manifest.get("kind") == "numerical_refinement":
            base = manifest["base_evidence"]
            path = reject_symlinks(Path(base["manifest"]).absolute())
            if file_sha256(path) != base["sha256"]:
                raise TheoryError("base evidence manifest changed")
            directory, manifest = path.parent, read_object(path)
        if manifest.get("complete") is not True or "evidence_recipe" not in manifest["provenance"]:
            raise TheoryError("a completed fixed evidence collection is required")
        provenance = manifest["provenance"]
        direct_path = reject_symlinks(Path(provenance["analysis_manifest"]).absolute())
        if file_sha256(direct_path) != provenance["analysis_manifest_sha256"]:
            raise TheoryError("original direct-analysis receipt changed")
        direct = read_object(direct_path)
        core = direct["core_identity"]
        from .direct_reduce import core_recipe, _record_key
        if (core["recipe"] != core_recipe() or core["source_metadata_hash"] != sources.metadata_hash()
                or core["record_hashes"] != [_record_key(record) for record in sources.selected]
                or core["reference_law_hash"] != provenance["reference_law_hash"]):
            raise TheoryError("saved observations differ from current frozen sources or measurement recipe")
        recipe = provenance["evidence_recipe"]
        _verify_sources(recipe["supplement_recipe"]["source_code"])
        _verify_sources(recipe["reference_recipe"]["source_code"])
        return {"tables": _read_tables(manifest), "files": manifest["files"],
                "logical_joins": manifest.get("logical_joins", {}), "provenance": provenance,
                "directory": directory}
    except (TheoryError, OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        raise _missing(config, str(error)) from error


def _validate_backing(law_path, schedule_path, provenance):
    """Validate backing contents against the original evidence, not a fresh hash."""
    from .support import _tensor_hash
    law_path, schedule_path = reject_symlinks(law_path), reject_symlinks(schedule_path)
    schedule_digest = file_sha256(schedule_path)
    expected_schedule = provenance["evidence_recipe"]["supplement_recipe"]["schedule_sha256"]
    if schedule_digest != expected_schedule:
        raise TheoryError("Saved scheduler backing differs from the pinned evidence receipt")
    law_digest = file_sha256(law_path)
    payload = safe_torch_load(law_path)
    metadata = payload["metadata"]
    identity = {name: value for name, value in metadata.items()
                if name not in {"candidate_chunk", "query_chunk", "complexity"}}
    expected_metadata = {name: value for name, value in provenance["reference_law"].items()
                         if name not in {"candidate_chunk", "query_chunk", "complexity"}}
    if (payload.get("law_hash") != provenance["reference_law_hash"]
            or canonical_hash(identity) != provenance["reference_law_hash"]
            or identity != expected_metadata
            or _tensor_hash(payload["atoms"]) != metadata["tensor_sha256"]
            or payload["weights"].double().tolist() != metadata["weights"]):
        raise TheoryError("Saved reference-law backing differs from the pinned atom/mass identity")
    return law_digest, schedule_digest


def _verified_inputs(log, coeff, *, indices, step, steps, guidance):
    z, u, c, target = log
    return {"schema_version": 1, "state": z[indices, step], "epsilon_u": u[indices, step],
            "epsilon_c": c[indices, step], "saved_endpoint": z[indices, step + 1], "target": target,
            "alpha": coeff.alpha, "sigma": coeff.sigma, "A": coeff.A, "kappa": coeff.kappa,
            "guidance": guidance, "destination_alpha": coeff.destination_alpha,
            "destination_sigma": coeff.destination_sigma, "noise_std": coeff.noise_std,
            "manuscript_domain": step < steps - 1}


def input_recipe():
    """Hash only input construction, separately from retry algorithms/budgets."""
    from . import numerical_refinement as engine
    functions = (engine._flat, engine._norm, engine._target_logits,
                 engine.build_refinement_payload, _verified_inputs)
    return {"payload_version": engine.PAYLOAD_VERSION, "batch_layout": INPUT_BATCH_SIZE,
            "construction_sources": {function.__name__: hashlib.sha256(inspect.getsource(function).encode()).hexdigest()
                                     for function in functions},
            "support_sources": _sources("support.py", "scheduler_adapter.py", "reference_law.py", "cache_reader.py")}


def policy_recipe(config):
    from .numerical_refinement import NumericalPolicy
    policy = NumericalPolicy(**{name.removeprefix("numerical_"): config[name] for name in NUMERICAL_KEYS})
    return {"policy": policy.identity(), "source_code": _sources("numerical_reduce.py", "numerical_refinement.py", "numerical_intervals.py", "numerical_screening.py")}


def _input_complete(path, identity):
    marker = Path(path) / "complete.json"
    try:
        receipt = read_object(marker)
        payload = contained_path(path, "payload.pt")
        return (receipt.get("complete") is True and receipt.get("identity") == identity
                and file_sha256(payload) == receipt["sha256"])
    except (TheoryError, OSError, ValueError, KeyError):
        return False


def _unavailable(base, law_hash, reason):
    rows = []
    for identity in base[KEYS].to_dict("records"):
        rows.append({**identity, "refinement_applicable": False, "input_contract_status": reason,
            "endpoint_construction_method": "not_applicable", "arithmetic_status": "not_applicable",
            "arithmetic_error_method": "not_applicable", "quadrature_status": "not_applicable",
            "quadrature_error_scope": "not_applicable", "source_sensitivity_status": "not_applicable",
            "source_sensitivity_model": "not_applicable", "reference_law_id": law_hash,
            "fixed_cache_gain_sign": "not_applicable", "source_robust_gain_sign": "not_applicable",
            "condition_sign_status": "not_applicable", "condition_value_status": "not_applicable",
            "numerical_log_probability_gain": math.nan, "numerical_log_odds_gain": math.nan,
            "numerical_original_margin_l2": math.nan, "numerical_original_margin_rmse": math.nan,
            "numerical_gain_before": "retained_in_direct_prop5_columns",
            "numerical_condition_before": "retained_in_direct_prop5_columns",
            "numerical_stopping_reason": reason, "numerical_publication_blocker": False,
            "implication_eligible": False, "implication_audit_status": "structurally_inapplicable"})
    return rows


def _worker(*, sources, tasks, records, direct_directory, law_path, law_digest, schedule_path,
            schedule_digest, config, device, worker_count, candidate_chunk_size, query_chunk_size):
    from .cache_reader import load_record
    from .direct_reduce import _paths, _read_law
    from .numerical_refinement import NumericalPolicy, build_refinement_payload, refine_payload
    from .scheduler_adapter import SchedulerAdapter
    configure_worker_cpu_threads(worker_count)
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    law = _read_law(Path(law_path), law_digest, device, candidate_chunk_size, query_chunk_size)
    if file_sha256(schedule_path) != schedule_digest:
        raise TheoryError("Saved scheduler changed before numerical refinement")
    schedule = safe_torch_load(schedule_path)
    science = sources.runs["experiment"]["scientific_config"]
    adapter = SchedulerAdapter(schedule, recorded_diffusers_version=science.get("package_versions", {}).get("diffusers"))
    policy = NumericalPolicy(**{name.removeprefix("numerical_"): config[name] for name in NUMERICAL_KEYS})
    records = {str(record.original_index): record for record in records}
    receipts = []
    try:
        for task in tasks:
            status, log = "failed", None
            try:
                _verify_sources(task["identity"]["recipe"]["source_code"])
                _verify_sources(task["input_identity"]["recipe"]["support_sources"])
                path = Path(task["path"])
                record = records[task["record_index"]]
                root, original_paths = _paths(direct_directory, record)
                if file_sha256(root / "complete.json") != task["identity"]["observation_marker_sha256"]:
                    raise TheoryError("Saved direct observation receipt changed")
                base = pd.read_parquet(original_paths["matched_updates"])
                if file_sha256(original_paths["matched_updates"]) != task["identity"]["matched_sha256"]:
                    raise TheoryError("Saved matched scalar observations changed")
                if _complete(path, task["identity"]):
                    status = "resumed"
                else:
                    # The checked scalar receipt already identifies the exact
                    # law atom. Completed numerical batches need no raw witness.
                    target_atom, target_known = None, "candidate_target_atom_id" in base
                    if target_known:
                        atom_ids = base.candidate_target_atom_id.dropna().astype(str).unique()
                        if len(atom_ids) > 1:
                            raise TheoryError("Saved matched rows disagree on their reference target atom")
                        if len(atom_ids):
                            if atom_ids[0] not in law.support.aliases:
                                raise TheoryError("Saved target atom no longer belongs to the declared law")
                            target_atom = law.support.aliases[atom_ids[0]]

                    def ensure_raw_record():
                        nonlocal log, target_atom, target_known
                        if log is None:
                            log = load_record(sources.experiment, record)
                            target_id = law.target_id_for(log[3].to(device=device, dtype=torch.float64), required=False)
                            observed_atom = None if target_id is None else law.support.aliases[target_id]
                            if target_known and observed_atom != target_atom:
                                raise TheoryError("Raw target differs from the pinned scalar target atom")
                            target_atom, target_known = observed_atom, True
                        return log

                    if not target_known:
                        ensure_raw_record()  # Compatibility with older scalar-only receipts.
                    all_rows, input_receipts = [], []
                    seeds = list(map(str, record.metadata["seeds"]))
                    steps = int(config["num_inference_steps"])
                    for step in range(steps):
                        coeff = adapter.coefficients(step)
                        selected = base.loc[base.step_index.eq(step)]
                        by_seed = selected.set_index(selected.seed.astype(str))
                        if not by_seed.index.is_unique or set(by_seed.index) != set(seeds):
                            raise TheoryError("Numerical refinement must preserve every step/seed identity")
                        selected = by_seed.loc[seeds].reset_index(drop=True)
                        structural = (target_atom is not None and coeff.affine and coeff.kappa > 0
                                      and coeff.alpha > 0 and coeff.sigma > 0 and coeff.destination_alpha > 0
                                      and coeff.destination_sigma > 0 and config["guidance_scale"] > 1)
                        if not structural:
                            all_rows.extend(_unavailable(selected, law.law_hash, "outside_affine_positive_noise_reference_domain"))
                            continue
                        for start in range(0, len(seeds), INPUT_BATCH_SIZE):
                            stop = min(start + INPUT_BATCH_SIZE, len(seeds))
                            old = selected.iloc[start:stop]
                            identity = {"record_inputs": task["input_identity"], "step_index": step, "seeds": seeds[start:stop]}
                            input_path = Path(task["input_path"]) / f"step-{step:04d}-seeds-{start:06d}-{stop:06d}"
                            input_complete = _input_complete(input_path, identity)
                            batch_path = path / "batches" / input_path.name
                            if input_complete:
                                marker_digest = file_sha256(input_path / "complete.json")
                                batch_identity = {"record_policy": task["identity"], "input_marker_sha256": marker_digest,
                                                  "step_index": step, "seeds": seeds[start:stop]}
                                if _complete(batch_path, batch_identity):
                                    all_rows.extend(pd.read_parquet(batch_path / "matched_numerical.parquet").to_dict("records"))
                                    input_receipts.append({"path": str(input_path / "complete.json"), "sha256": marker_digest})
                                    continue
                            # Only a genuinely pending batch reads the protected
                            # vectors or deserializes sufficient tensor inputs.
                            inputs = _verified_inputs(ensure_raw_record(), coeff, indices=list(range(start, stop)), step=step,
                                steps=steps, guidance=float(config["guidance_scale"]))
                            if input_complete:
                                payload = safe_torch_load(input_path / "payload.pt")
                            else:
                                with torch.inference_mode():
                                    payload = build_refinement_payload(law.support, inputs, bank_hash=law.law_hash, target_atom=target_atom)
                                stored = {name: value.detach().cpu() if isinstance(value, torch.Tensor) else value
                                          for name, value in payload.items()}
                                digest = atomic_torch_save(stored, input_path / "payload.pt")
                                atomic_write_json(input_path / "complete.json", {"complete": True, "identity": identity, "sha256": digest})
                                marker_digest = file_sha256(input_path / "complete.json")
                                batch_identity = {"record_policy": task["identity"], "input_marker_sha256": marker_digest,
                                                  "step_index": step, "seeds": seeds[start:stop]}
                            if payload.get("bank_hash") != law.law_hash or payload.get("target_atom") != target_atom:
                                raise TheoryError("Numerical sufficient inputs changed law or target")
                            with torch.inference_mode():
                                refined = refine_payload(law.support, payload, old, policy=policy, verified_inputs=inputs)
                            if len(refined) != len(old):
                                raise TheoryError("Numerical refinement returned incomplete seed coverage")
                            rows = [{**key, **value} for key, value in zip(old[KEYS].to_dict("records"), refined, strict=True)]
                            _save(batch_path, batch_identity, {"matched_numerical": pd.DataFrame(rows)})
                            all_rows.extend(rows)
                            input_receipts.append({"path": str(input_path / "complete.json"), "sha256": marker_digest})
                    extra = pd.DataFrame(all_rows)
                    _join(base, extra)
                    _verify_sources(task["identity"]["recipe"]["source_code"])
                    _verify_sources(task["input_identity"]["recipe"]["support_sources"])
                    _save(path, task["identity"], {"matched_numerical": extra},
                          {"input_receipts": input_receipts, "complete": True,
                           "migration": "additive_new_classifications; original_columns_and_payloads_retained"})
                    status = "reduced"
                receipts.append({"task_hash": task["task_hash"], "status": status, "device": str(device)})
            except Exception as error:
                receipt = {"task_hash": task["task_hash"], "status": "failed", "device": str(device),
                           "reason": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()}
                atomic_write_json(Path(task["path"]) / "failure.json", receipt)
                receipts.append(receipt)
            finally:
                del log
                report_worker_record(task["task_hash"], status, device)
    finally:
        del law
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return receipts


def run_precision_analysis(project_root, *, config, refine_only=False, device="auto", probe_batch_size=8,
                           candidate_chunk_size=256, query_chunk_size=16):
    from .cache_reader import discover_sources
    from .direct_reduce import _paths
    from .reduce import _resolve_theory_devices, _worker_sources
    root = Path(project_root).absolute()
    config = numerical_config(**config)
    try:
        sources = discover_sources(root, **config)
    except (TheoryError, OSError) as error:
        if refine_only:
            raise _missing(config, str(error)) from error
        raise
    try:
        evidence = load_saved_evidence(root, config=config, sources=sources)
    except TheoryError:
        if refine_only:
            raise
        from .evidence_reduce import run_evidence_analysis
        evidence = run_evidence_analysis(root, config=config, device=device, probe_batch_size=probe_batch_size,
                                        candidate_chunk_size=candidate_chunk_size, query_chunk_size=query_chunk_size)
    provenance = evidence["provenance"]
    direct_directory = Path(provenance["analysis_manifest"]).parent
    core = direct_directory.parent.parent
    law_path, schedule_path = core / "reference_law.pt", core / "schedule.pt"
    try:
        law_digest, schedule_digest = _validate_backing(law_path, schedule_path, provenance)
    except (TheoryError, OSError, ValueError, KeyError, TypeError) as error:
        raise _missing(config, str(error)) from error
    recipe, construction = policy_recipe(config), input_recipe()
    numerical_root = contained_path(core.parent.parent, "numerical_refinement")
    tasks = []
    for record in sources.selected:
        source_root, paths = _paths(direct_directory, record)
        input_identity = {"recipe": construction, "record": record.metadata,
                          "law_sha256": law_digest, "schedule_sha256": schedule_digest}
        identity = {"recipe": recipe, "input_identity_hash": canonical_hash(input_identity),
                    "observation_marker_sha256": file_sha256(source_root / "complete.json"),
                    "matched_sha256": file_sha256(paths["matched_updates"])}
        digest = canonical_hash(identity)
        tasks.append({"identity": identity, "task_hash": digest, "record_index": str(record.original_index),
                      "path": str(numerical_root / "policies" / canonical_hash(recipe) / "records" / digest),
                      "input_identity": input_identity,
                      "input_path": str(numerical_root / "inputs" / canonical_hash(input_identity))})
    if not tasks:
        raise _missing(config, "no retained records")
    with publication_lock(numerical_root):
        pending = [task for task in tasks if not _complete(task["path"], task["identity"])]
        devices = tuple(_resolve_theory_devices(device)) if pending else ()
        count = worker_count_for_tasks(devices, len(pending)) if pending else 0
        context, receipts = get_context("spawn"), []
        with RecordProgress(total=len(tasks), devices=count, context=context) as progress:
            pending_ids = {task["task_hash"] for task in pending}
            for task in tasks:
                if task["task_hash"] not in pending_ids:
                    progress.report(task["task_hash"], "resumed", "saved")
            if count:
                with ProcessPoolExecutor(max_workers=count, mp_context=context,
                                         initializer=install_progress_queue, initargs=(progress.queue,)) as executor:
                    futures = {}
                    for index in range(count):
                        shard = round_robin_shard(pending, worker_index=index, worker_count=count)
                        ids = {task["record_index"] for task in shard}
                        records = [record for record in sources.selected if str(record.original_index) in ids]
                        future = executor.submit(_worker, sources=_worker_sources(sources, records), tasks=shard, records=records,
                            direct_directory=direct_directory, law_path=law_path, law_digest=law_digest,
                            schedule_path=schedule_path, schedule_digest=schedule_digest, config=config,
                            device=str(devices[index]), worker_count=count,
                            candidate_chunk_size=candidate_chunk_size, query_chunk_size=query_chunk_size)
                        futures[future] = (index, shard)
                    for future in as_completed(futures):
                        index, shard = futures[future]
                        try:
                            result = future.result()
                        except Exception as error:
                            result = [{"task_hash": task["task_hash"], "status": "failed", "device": str(devices[index]),
                                       "reason": f"{type(error).__name__}: {error}"} for task in shard]
                        receipts.extend(result)
                        for item in result:
                            progress.report(item["task_hash"], item["status"], item["device"])
        incomplete = [task for task in tasks if not _complete(task["path"], task["identity"])]
        if incomplete:
            failure = numerical_root / "policies" / canonical_hash(recipe) / "failed.json"
            atomic_write_json(failure, {"incomplete_records": len(incomplete), "receipts": receipts})
            raise TheoryError(f"{len(incomplete)} numerical records incomplete; compatible observations retained. See {failure}. "
                              f"If required source inputs are missing, run {recompute_command(config)}")
        _verify_sources(recipe["source_code"])
        _verify_sources(construction["support_sources"])
        extra = pd.concat([pd.read_parquet(Path(task["path"]) / "matched_numerical.parquet") for task in tasks], ignore_index=True)
        tables = dict(evidence["tables"])
        tables["matched_updates"] = _join(tables["matched_updates"], extra)
        base_manifest = Path(evidence["directory"]) / "manifest.json"
        base_receipt = {"manifest": str(base_manifest), "sha256": file_sha256(base_manifest)}
        identity = {"recipe": recipe, "base_evidence": base_receipt, "task_hashes": [task["task_hash"] for task in tasks]}
        digest = canonical_hash(identity)
        collection = numerical_root / "collections" / digest
        path = collection / "matched_numerical.parquet"
        atomic_write_frame_parquet(extra, path)
        files = {**evidence["files"], "matched_numerical": {"path": str(path), "sha256": file_sha256(path), "rows": len(extra)}}
        joins = dict(evidence.get("logical_joins", {}))
        joins["matched_updates"] = {"base": "matched_updates", "supplement": "matched_numerical", "keys": KEYS,
                                    "how": "exact_one_to_one_additive", "preserves_every_base_row": True}
        updated = {**provenance, "analysis_hash": digest, "base_evidence_analysis_hash": provenance["analysis_hash"],
            "numerical_policy": recipe, "numerical_input_recipe": construction,
            "numerical_migration": {"old_columns_preserved": True, "old_classifications_preserved": True,
                "old_interpretation": "legacy_dtype_sensitivity_combined_with_numerical_assessment",
                "new_interpretation": "fixed_cached_input_arithmetic; source_sensitivity_and_reference_identification_separate",
                "base_evidence": base_receipt},
            "numerical_execution": {"devices": list(map(str, devices[:count])), "worker_count": count,
                                    "spawn": True, "parent_progress_bars": 1}}
        atomic_write_json(collection / "manifest.json", {"schema_version": NUMERICAL_COLLECTION_VERSION,
            "kind": "numerical_refinement", "complete": True, "identity": identity,
            "base_evidence": base_receipt, "files": files, "logical_joins": joins, "provenance": updated})
        return {"tables": tables, "files": files, "logical_joins": joins, "provenance": updated, "directory": collection}
