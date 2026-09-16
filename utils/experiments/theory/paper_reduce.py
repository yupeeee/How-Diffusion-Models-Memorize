"""Publish the four mechanism experiments from shared immutable measurements."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from utils.common.io import atomic_write_json, canonical_hash, file_sha256
from .contracts import FOUR_STAGE_OPTION_KEYS, TheoryError, numerical_config, read_object
from .paper_contracts import (
    BUNDLE_SCHEMA_VERSION,
    METRIC_SCHEMA_VERSION,
    PaperPaths,
    contained_path,
    load_paper_inputs,
    measurement_sources,
    publication_lock,
    reject_symlinks,
    retire_obsolete_figures,
    retire_legacy_figures,
    staged_publication,
    saved_scientific_configuration,
    write_plot_table,
)
from .paper_registry import REGISTRY_VERSION, paper_registry
from .progress import StageProgress


def _clean(value):
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_clean(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if value is pd.NA:
        return None
    return value


ENDPOINT_SOURCES = frozenset(
    {
        "candidate_feedback.py",
        "candidate_metrics.py",
        "candidate_reduce.py",
        "cache_reader.py",
        "metrics.py",
        "scheduler_adapter.py",
        "support.py",
        "feedback.py",
    }
)


def _source_recipe_current(manifest, *, diagnostics=False):
    sources = manifest.get("source_code")
    required = ENDPOINT_SOURCES | (
        {"candidate_integration.py"} if diagnostics else set()
    )
    return (
        isinstance(sources, dict)
        and required <= sources.keys()
        and all(
            file_sha256(Path(__file__).parent / name) == sources[name]
            for name in required
        )
    )


def _source_bundle(root, config, *, source_analysis, diagnostics, device, **options):
    from .candidate_contracts import (
        find_candidate_bundle,
        validate_candidate_bundle,
        validate_candidate_sources,
    )

    if source_analysis is not None:
        bundle = reject_symlinks(Path(source_analysis).absolute())
        manifest = validate_candidate_bundle(bundle, expected_config=config)
        return bundle, manifest, "explicit_saved_analysis"
    try:
        bundle = find_candidate_bundle(root, **config)
        manifest = validate_candidate_bundle(bundle, expected_config=config)
        validate_candidate_sources(bundle, root)
        if not _source_recipe_current(manifest, diagnostics=diagnostics):
            raise TheoryError(
                "Candidate endpoint/integration definitions changed or lack source provenance"
            )
        if diagnostics and not manifest.get("complete"):
            raise TheoryError("Requested integration diagnostics are incomplete")
        return bundle, manifest, "validated_current_analysis"
    except (TheoryError, OSError):
        from .reduce import run_theory
        from .candidate_reduce import run_candidates

        base = run_theory(
            root,
            **config,
            **options,
            device=device,
            defer_path_integration=True,
            prepare_candidate_endpoints=True,
        )
        bundle = run_candidates(
            root,
            **config,
            **options,
            device=device,
            base_bundle=base,
            render=False,
            endpoint_only=not diagnostics,
        )
        return (
            bundle,
            validate_candidate_bundle(bundle, expected_config=config),
            "protected_cache_analysis",
        )


def _auxiliary(frames, metadata):
    auxiliary = metadata.pop("auxiliary_tables", {})
    if isinstance(auxiliary, list):
        auxiliary = {name: frames.pop(name) for name in auxiliary if name in frames}
    return auxiliary


def _save_table(stage, relative, frame, numerical_files):
    if len(frame.columns) == 0:
        frame = pd.DataFrame(columns=["status"])
    specification = write_plot_table(frame, contained_path(stage, relative))
    numerical_files[relative] = specification["sha256"]
    return {"path": relative, **specification}


def run_paper(
    project_root, *, source_analysis=None, source_logs=None, diagnostics=False,
    recompute=False, refine_numerics=False, device="auto", candidate_chunk_size=256, query_chunk_size=16,
    probe_batch_size=8, **configuration,
):
    """Publish direct measurements transactionally, preserving all upstream caches.

    Recompute authorizes a changed scientific configuration. Valid independent
    task shards are still reused: it does not repeat denoiser calls unnecessarily.
    """
    from .numerical_reduce import run_precision_analysis
    from .four_stage_reduce import prepare_four_stage_primary, run_four_stage_analysis
    from .four_stage_figures import build_four_stage_plot_inputs
    from .counterfactual_probes import run_counterfactual_analysis
    from .gaussian_controls import run_gaussian_control_analysis
    from .paper_plotting import render_paper

    if source_analysis is not None or source_logs is not None:
        raise TheoryError(
            "Legacy candidate/archived-log migration cannot supply the new measured losses. "
            "Run direct analysis against the declared protected generation caches without "
            "--source-analysis/--source-logs; historical candidate bundles remain preserved."
        )
    root = Path(project_root).absolute()
    config = numerical_config(**configuration)
    output = PaperPaths.build(root, **config).output_directory
    with publication_lock(output):
        previous_config = (read_object(output / "run_config.json")
                           if (output / "run_config.json").exists() else None)
        if (previous_config and previous_config.get("schema_version") == BUNDLE_SCHEMA_VERSION
                and saved_scientific_configuration(previous_config) != config and not recompute):
            raise TheoryError(
                "Active paper scientific configuration differs; use --recompute-experiments "
                "to archive the previous publication and measure the requested configuration"
            )
        base_config = {key: value for key, value in config.items() if key not in FOUR_STAGE_OPTION_KEYS}
        if not refine_numerics:
            with StageProgress("Preparing fast endpoints and trajectory measurements"):
                prepare_four_stage_primary(
                    root, config=base_config, device=device,
                    candidate_chunk_size=candidate_chunk_size, query_chunk_size=query_chunk_size,
                )
        with StageProgress("Numerical refinement and collecting saved measurement tables"):
            result = run_precision_analysis(
                root, config=base_config, refine_only=refine_numerics, device=device, probe_batch_size=probe_batch_size,
                candidate_chunk_size=candidate_chunk_size, query_chunk_size=query_chunk_size,
            )
        base_analysis = {"path": str(result["directory"]), "analysis_hash": result["provenance"]["analysis_hash"],
                         "mode": "preserved_numerical_and_evidence_measurements"}
        with StageProgress("Checking and collecting initial Gaussian controls"):
            result = run_gaussian_control_analysis(
                root, result=result, config=base_config, device=device, probe_batch_size=probe_batch_size,
                candidate_chunk_size=candidate_chunk_size, query_chunk_size=query_chunk_size,
                allow_compute=not refine_numerics,
            )
        with StageProgress("Collecting four-stage tables and baseline/trajectory summaries"):
            result = run_four_stage_analysis(
                root, result=result, config=config, device=device, probe_batch_size=probe_batch_size,
                candidate_chunk_size=candidate_chunk_size, query_chunk_size=query_chunk_size,
                allow_compute=not refine_numerics,
            )
        if config.get("counterfactual_unconditional", False):
            with StageProgress("Checking and collecting optional learned counterfactuals"):
                result = run_counterfactual_analysis(
                    root, result=result, config=config, device=device, probe_batch_size=probe_batch_size,
                    candidate_chunk_size=candidate_chunk_size, query_chunk_size=query_chunk_size,
                    allow_compute=not refine_numerics,
                )
        tables, provenance = result["tables"], result["provenance"]
        with StageProgress("Reducing plot inputs, bootstrap intervals, and figure audits"):
            frames, metadata = build_four_stage_plot_inputs(tables, config=config, provenance=provenance)
        registry_options = {"counterfactual": config.get("counterfactual_unconditional", False)}
        auxiliary = _auxiliary(frames, metadata)
        figures = metadata["figures"]
        # Compact science depends on measured scalar contents and formula recipes,
        # never on worker placement, wall time, batch size or figure styling.
        identity = _clean({
            "config": config, "metric_schema_version": METRIC_SCHEMA_VERSION,
            "measurement_sources": measurement_sources(),
            "analysis_hash": provenance["analysis_hash"],
            "reference_law_hash": provenance["reference_law_hash"],
            "scalar_sources": {name: {key: value for key, value in spec.items() if key in {"sha256", "rows"}}
                               for name, spec in result["files"].items()},
        })
        scientific_hash = canonical_hash(identity)
        audit = _clean({"schema_version": 2, "audit_version": "four-stage-evidence-audit-1",
                        "scientific_hash": scientific_hash, **metadata.get("audit", {})})
        failed_figures = {name: item for name, item in figures.items()
                          if item["status"] in {"error", "blocked", "missing"}}
        audit["blocking"] = bool(audit.get("blocking", False) or failed_figures)
        unavailable_required = {
            entry["stem"]: figures[entry["stem"]]
            for entry in paper_registry(diagnostics=diagnostics, **registry_options)
            if figures[entry["stem"]]["status"] == "unavailable"
            and not entry.get("allow_unavailable", entry["category"] == "diagnostics")
        }
        if audit["blocking"] or unavailable_required:
            failed = output.parent / ".failed-attempts" / output.name / scientific_hash
            failed.mkdir(parents=True, exist_ok=True)
            atomic_write_json(failed / "audit.json", {**audit, "unavailable_required": list(unavailable_required)})
            atomic_write_json(failed / "figure_status.json", _clean(figures))
            for name, frame in auxiliary.items():
                _save_table(failed, name + ".csv", frame, {})
            if unavailable_required and not audit["blocking"]:
                from .paper_contracts import recompute_command
                raise TheoryError(f"Required evidence inputs unavailable; statuses retained: {failed}. "
                                  f"Run {recompute_command(config)}")
            raise TheoryError(f"Paper correctness audit blocked publication; retained identities and reasons: {failed}")
        previous = (read_object(output / "figure_manifest.json")
                    if (output / "figure_manifest.json").exists() else {})
        run_config = _clean({
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "metric_schema_version": METRIC_SCHEMA_VERSION,
            "scientific_config": base_config,
            "supplemental_config": {key: value for key, value in config.items() if key in FOUR_STAGE_OPTION_KEYS},
            "scientific_identity": identity,
            "scientific_hash": scientific_hash, "registry_version": REGISTRY_VERSION,
            "source_analysis": base_analysis,
            "supplemental_analysis": {"path": str(result["directory"]), "analysis_hash": provenance["analysis_hash"],
                                      "mode": "four_stage_additive_measurements"},
            "provenance": provenance,
        })
        archive_previous = previous_config is not None and previous_config.get("scientific_hash") != scientific_hash
        with StageProgress("Publishing paper bundle"), staged_publication(output, archive_previous=archive_previous) as stage:
            with StageProgress("Saving scalar tables and publication metadata") as progress:
                numerical_files, plot_data = {}, {}
                for entry in paper_registry(diagnostics=True, **registry_options):
                    name = entry["stem"]
                    if name not in figures:
                        raise TheoryError(f"Missing paper measurement contract: {name}")
                    if name in frames and len(frames[name].columns):
                        progress.set_detail(f"plot_data/{name}.csv: {len(frames[name]):,} rows")
                        plot_data[name] = _save_table(stage, f"plot_data/{name}.csv", frames[name], numerical_files)
                for name, frame in auxiliary.items():
                    progress.set_detail(f"audit_data/{name}.csv: {len(frame):,} rows")
                    _save_table(stage, "audit_data/" + name + ".csv", frame, numerical_files)
                baseline = tables.get("initial_baseline_summary", auxiliary.get("initial_baseline_summary"))
                if baseline is not None:
                    progress.set_detail("initial_baseline_summary.csv")
                    _save_table(stage, "initial_baseline_summary.csv", baseline, numerical_files)
                    atomic_write_json(stage / "initial_baseline_summary.json", _clean({
                        "schema_version": 1, "rows": baseline.to_dict("records"),
                        "scope": "unique_Gaussian_seed_mean_baseline_about_declared_empirical_law",
                    }))
                    numerical_files["initial_baseline_summary.json"] = file_sha256(stage / "initial_baseline_summary.json")
                for name in ("initial", "terminal"):
                    progress.set_detail(f"{name}.csv: {len(tables[name]):,} rows")
                    _save_table(stage, name + ".csv", tables[name], numerical_files)
                _save_table(stage, "failed.csv", pd.DataFrame(columns=["record_id", "reason"]), numerical_files)
                scalar_aliases = {}
                for logical_name, source_name in (("initial_pairs", "initial_loss_recovery"),):
                    if source_name in plot_data:
                        scalar_aliases[logical_name] = plot_data[source_name]
                for logical_name, source_name in (("feedback_endpoints", "matched_updates"), ("terminal_metrics", "terminal")):
                    if logical_name not in result["files"] and source_name in result["files"]:
                        scalar_aliases[logical_name] = {"base_table": source_name,
                            "additive_join": result.get("logical_joins", {}).get(source_name)}
                logical = {
                    "schema_version": 2, "source_analysis": str(result["directory"]),
                    "tables": result["files"], "scalar_aliases": scalar_aliases,
                    "additive_joins": result.get("logical_joins", {}),
                    "reading_policy": "Analysis-only scalar tables; plot mode reads compact plot_data CSVs only.",
                }
                for name, value in (("run_config.json", run_config), ("audit.json", audit), ("logical_tables.json", logical)):
                    progress.set_detail(name)
                    atomic_write_json(stage / name, _clean(value))
                    numerical_files[name] = file_sha256(stage / name)
                atomic_write_json(stage / "registry.json", {"version": REGISTRY_VERSION, "figures": paper_registry(diagnostics=True, **registry_options)})
                summary = _clean({
                    "schema_version": 2, "complete": True, "scientific_hash": scientific_hash,
                    "figures": figures, "plot_data": plot_data, "numerical_files": numerical_files,
                    "counts": {"initial_rows": len(tables["initial"]), "terminal_rows": len(tables["terminal"]),
                               "trajectory_rows": len(tables["trajectory"])},
                })
                progress.set_detail("summary.json")
                atomic_write_json(stage / "summary.json", summary)
            with StageProgress("Validating saved scalar publication"):
                load_paper_inputs(stage, expected_config=config, diagnostics=diagnostics)
            render_paper(stage, diagnostics=diagnostics)
            with StageProgress("Retiring verified obsolete figures"):
                retire_obsolete_figures(stage, previous)
        print(f"Paper outputs: {output}", flush=True)
    return output
