"""Author-run regressions for the fixed curation; no scientific execution needed."""
from pathlib import Path
import shutil

import pytest

from tests.test_theory_paper_plotting import compact_fixture
from utils.common.io import atomic_write_json, canonical_hash, file_sha256, read_json
from utils.experiments.theory import paper_contracts as contracts, paper_plotting
from utils.experiments.theory.contracts import TheoryError
from utils.experiments.theory.paper_registry import (
    FIGURE_RETIREMENTS, REGISTRY_VERSION, RETIRED_RENDER_STEMS,
    paper_registry, measurement_registry, previous_paper_registry, _ZERO_BASELINE_FIGURES,
)


def _snapshot(bundle):
    return {path.relative_to(bundle).as_posix(): (
        ("symlink", path.readlink().as_posix()) if path.is_symlink()
        else ("file", file_sha256(path)))
        for path in bundle.rglob("*") if path.is_symlink() or path.is_file()}


def _old_bundle(bundle, *, diagnostics=False):
    compact_fixture(bundle, terminal=True, diagnostics=diagnostics, previous_registry=True,
                    saved_prompt_curve=True, saved_grouped_coverage=True, saved_zero_baselines=True,
                    saved_prompt_reference_curves=True, saved_reference_gap_curve=True, saved_guidance_fit=True, saved_reference_variation=True)
    files = {}
    historical = [*previous_paper_registry(diagnostics=diagnostics),
                  *measurement_registry(diagnostics=diagnostics), *_ZERO_BASELINE_FIGURES]
    for entry in historical:
        for name in entry["outputs"].values():
            path = bundle / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(("old reviewed export: " + name).encode())
            files[name] = file_sha256(path)
    companions = []
    for step in (0, 1):
        outputs = {suffix: f"appendix/posterior_feedback_condition_margin/step_{step:03d}.{suffix}"
                   for suffix in ("png", "pdf")}
        for name in outputs.values():
            path = bundle / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(("old reviewed timestep: " + name).encode())
            files[name] = file_sha256(path)
        companions.append({"stem": f"posterior_feedback_condition_margin/step_{step:03d}", "outputs": outputs})
    (bundle / "figure_captions.md").write_text("Previous reviewed captions\n")
    files["figure_captions.md"] = file_sha256(bundle / "figure_captions.md")
    previous = {"complete": True, "registry_version": "four-stage-paper-curation-19", "files": files,
                "figures": historical, "timestep_figures": companions}
    atomic_write_json(bundle / "figure_manifest.json", previous)
    atomic_write_json(bundle / "registry.json", {"version": "four-stage-paper-curation-19", "figures": historical})
    return previous


def _small_exports(monkeypatch, *, fail_pdf=False):
    """Exercise the real paired publisher and role swap without font/layout work."""
    class SavedFigure:
        def __init__(self, stem):
            self.stem = stem

        def savefig(self, path, *, format, **options):
            Path(path).write_bytes((self.stem + ":" + format).encode())
            if fail_pdf and format == "pdf":
                raise RuntimeError("injected second-format export failure")
    monkeypatch.setattr(paper_plotting, "_draw", lambda entry, *args: (SavedFigure(entry["stem"]), {}))
    monkeypatch.setattr(paper_plotting.plt, "close", lambda figure: None)
    monkeypatch.setattr(paper_plotting, "_finalize_presentations", lambda *args: {})


def test_saved_twenty_figure_bundle_migrates_to_six_without_touching_science_and_is_idempotent(tmp_path, monkeypatch):
    bundle = tmp_path / "paper"
    previous = _old_bundle(bundle)
    _small_exports(monkeypatch)
    for name in ("plot_data/branch_gap_motion_high_sscd.csv", "audit_data/branch_motion.csv",
                 "cached_images/keep.png", "proximity.csv", "user_notes.txt"):
        path = bundle / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(b"protected scientific or user artifact")
    protected = {name: value for name, value in _snapshot(bundle).items()
                 if name not in {"registry.json", "figure_manifest.json", "figure_captions.md"}
                 and not name.startswith(("main/", "appendix/", "diagnostics/"))}
    manifest = contracts.render_saved_paper(bundle, formats=("png", "pdf"))
    images = {name for name in manifest["files"] if Path(name).suffix in {".png", ".pdf"}}
    expected = {name for entry in paper_registry() for name in entry["outputs"].values()}
    assert images == expected and len(images) == 12
    assert all(name.startswith("figures/") for name in images)
    assert len(manifest["figures"]) == 6 and not manifest.get("timestep_figures", [])
    assert "figures/guidance_scale_vs_loss.png" in images
    assert "figures/corollary3_guidance_scale_vs_loss.png" not in images
    old_images = {name for name in previous["files"] if Path(name).suffix in {".png", ".pdf"}}
    assert all(not (bundle / name).exists() for name in old_images)
    assert {name: _snapshot(bundle)[name] for name in protected} == protected
    ledger = read_json(bundle / "figure_retirement.json")
    removed = [row for row in ledger["records"] if row["status"] == "removed"]
    assert {row["old_path"] for row in removed} == old_images
    assert all(row["sha256"] for row in removed)
    alias_move = next(row for row in removed if row["old_path"] == "appendix/corollary3_guidance_scale_vs_loss.png")
    assert alias_move["destination"] == "figures/guidance_scale_vs_loss.png"
    registry = read_json(bundle / "registry.json")
    assert registry["version"] == REGISTRY_VERSION
    assert len(registry["figures"]) == 6
    assert registry["measurement_inventory"] == measurement_registry(diagnostics=True)
    assert {"branch_gap_per_prompt", "reference_variation_per_prompt"}.issubset(
        entry["stem"] for entry in registry["measurement_inventory"])
    first = _snapshot(bundle)
    contracts.render_saved_paper(bundle, formats=("png", "pdf"))
    assert _snapshot(bundle) == first
    assert not (bundle / "archive").exists()
    assert not (bundle.parent / ".archives").exists()
    assert not list(bundle.parent.glob(".paper.backup-*"))
    assert not list(bundle.parent.glob(".paper.stage-*"))


def test_retirement_preserves_and_reports_unowned_modified_links_outside_and_unrelated(tmp_path, monkeypatch, capsys):
    bundle = tmp_path / "paper"
    previous = _old_bundle(bundle)
    _small_exports(monkeypatch)
    modified = "appendix/branch_gap_motion_high_sscd.png"
    unowned = "appendix/branch_gap_motion_high_sscd.pdf"
    link = "appendix/branch_gap_motion_lower_sscd.png"
    unrelated = "diagnostics/user_comparison.png"
    (bundle / modified).write_bytes(b"manual edit")
    del previous["files"][unowned]
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"outside the active bundle")
    (bundle / link).unlink()
    (bundle / link).symlink_to(outside)
    (bundle / unrelated).parent.mkdir(exist_ok=True)
    (bundle / unrelated).write_bytes(b"not authorized for retirement")
    previous["files"][unrelated] = file_sha256(bundle / unrelated)
    previous["files"]["../outside.png"] = file_sha256(outside)
    (bundle / "user_link").symlink_to(tmp_path / "nonexistent")
    atomic_write_json(bundle / "figure_manifest.json", previous)
    before = {name: _snapshot(bundle)[name] for name in (modified, unowned, link, unrelated, "user_link")}
    contracts.render_saved_paper(bundle, formats=("png", "pdf"))
    assert {name: _snapshot(bundle)[name] for name in before} == before
    assert outside.read_bytes() == b"outside the active bundle"
    rows = {row["old_path"]: row for row in read_json(bundle / "figure_retirement.json")["records"]}
    assert rows[modified]["decision"] == "preserved_modified"
    assert rows[unowned]["decision"] == "preserved_unowned"
    assert rows[link]["decision"] == rows["../outside.png"]["decision"] == "preserved_unsafe_path"
    assert unrelated in read_json(bundle / "figure_manifest.json")["preserved_files"]
    assert "Figure migration conflict" in capsys.readouterr().out
    ledger_before = (bundle / "figure_retirement.json").read_bytes()
    contracts.render_saved_paper(bundle, formats=("png", "pdf"))
    assert (bundle / "figure_retirement.json").read_bytes() == ledger_before


@pytest.mark.parametrize("failure", ["pdf", "publication"])
def test_migration_rollback_preserves_previous_images_and_publication_metadata(tmp_path, monkeypatch, failure):
    bundle = tmp_path / "paper"
    _old_bundle(bundle)
    before = _snapshot(bundle)
    _small_exports(monkeypatch, fail_pdf=failure == "pdf")
    if failure == "publication":
        replace = contracts.os.replace
        def fail_install(source, destination):
            if Path(source).name.startswith(".paper.stage-") and Path(destination) == bundle:
                raise RuntimeError("injected role publication failure")
            return replace(source, destination)
        monkeypatch.setattr(contracts.os, "replace", fail_install)
    with pytest.raises(RuntimeError, match="injected"):
        contracts.render_saved_paper(bundle, formats=("png", "pdf"))
    assert _snapshot(bundle) == before
    assert not list(bundle.parent.glob(".paper.backup-*"))
    assert not list(bundle.parent.glob(".paper.stage-*"))


def test_moved_files_require_both_owned_hash_valid_replacement_formats(tmp_path):
    old = "appendix/terminal_bound_coverage"
    new = "figures/terminal_bound_coverage"
    previous = {"files": {}}
    current = {"complete": True, "files": {}}
    for stem, manifest in ((old, previous), (new, current)):
        for suffix in ("png", "pdf"):
            name = f"{stem}.{suffix}"
            path = tmp_path / name
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(name.encode())
            manifest["files"][name] = file_sha256(path)
    current["files"][new + ".pdf"] = "stale receipt"
    atomic_write_json(tmp_path / "figure_manifest.json", current)
    ledger = contracts.retire_obsolete_figures(tmp_path, previous)
    assert all((tmp_path / f"{old}.{suffix}").exists() for suffix in ("png", "pdf"))
    decisions = {row["decision"] for row in ledger["records"] if row["old_path"].startswith(old)}
    assert decisions == {"preserved_missing_replacement_pair"}


@pytest.mark.parametrize("name", ["figure_retirement.json", "registry.json"])
def test_unrecognized_presentation_metadata_is_preserved_and_rolls_back(tmp_path, monkeypatch, name):
    bundle = tmp_path / "paper"
    _old_bundle(bundle)
    _small_exports(monkeypatch)
    atomic_write_json(bundle / name, {"user_notes": "preserve this file"})
    before = _snapshot(bundle)
    with pytest.raises(TheoryError, match="unrecognized " + name):
        contracts.render_saved_paper(bundle, formats=("png", "pdf"))
    assert _snapshot(bundle) == before


def test_unselected_owned_diagnostics_retire_even_when_diagnostics_requested(tmp_path, monkeypatch):
    bundle = tmp_path / "paper"
    _old_bundle(bundle, diagnostics=True)
    _small_exports(monkeypatch)
    entries = [entry for entry in measurement_registry(diagnostics=True) if entry["category"] == "diagnostics"]
    assert entries
    previous_images = {name for entry in entries for name in entry["outputs"].values()}
    scientific_before = {name: digest for name, digest in _snapshot(bundle).items()
                         if name.startswith(("plot_data/", "audit_data/"))}
    first = contracts.render_saved_paper(bundle, diagnostics=True, formats=("png", "pdf"))
    assert len(first["figures"]) == 6
    assert all(not (bundle / name).exists() for name in previous_images)
    assert previous_images.isdisjoint(first["preserved_files"])
    assert {name: _snapshot(bundle)[name] for name in scientific_before} == scientific_before
    second = contracts.render_saved_paper(bundle, diagnostics=True, formats=("png", "pdf"))
    assert set(second["files"]) == set(first["files"])
    assert not any(name.startswith("diagnostics/") for name in second["files"])


def test_copied_nondefault_bundle_migrates_without_devices_models_raw_reads_or_reduction(tmp_path, monkeypatch):
    from utils.models import probe_loading
    from utils.experiments.theory import (
        cache_reader, direct_integration, feedback, numerical_reduce,
        four_stage_reduce, four_stage_figures, gaussian_controls,
    )
    import numpy as np
    import pandas as pd
    import torch

    original = tmp_path / "original"
    _old_bundle(original)
    config = read_json(original / "run_config.json")
    config["scientific_config"]["num_loss_seeds"] = 128
    config["scientific_identity"]["config"] = config["scientific_config"]
    config["scientific_hash"] = canonical_hash(config["scientific_identity"])
    atomic_write_json(original / "run_config.json", config)
    for name in ("summary.json", "audit.json"):
        data = read_json(original / name)
        data["scientific_hash"] = config["scientific_hash"]
        atomic_write_json(original / name, data)
    summary = read_json(original / "summary.json")
    summary["numerical_files"].update({name: file_sha256(original / name)
                                       for name in ("run_config.json", "audit.json")})
    atomic_write_json(original / "summary.json", summary)
    bundle = tmp_path / "portable" / "renamed"
    shutil.copytree(original, bundle)
    _small_exports(monkeypatch)
    def forbidden(*args, **kwargs):
        raise AssertionError("plot-only curation attempted scientific work")
    for module, names in (
        (torch, ("load", "randn")), (torch.cuda, ("init",)),
        (np, ("load",)), (pd, ("read_parquet",)),
        (probe_loading, ("load_denoiser_components",)),
        (cache_reader, ("load_record", "discover_sources")),
        (feedback, ("proposition5_feedback",)),
        (direct_integration, ("integrate_proposition5_payload",)),
        (numerical_reduce, ("run_precision_analysis", "ProcessPoolExecutor", "get_context")),
        (four_stage_reduce, ("prepare_four_stage_primary", "run_four_stage_analysis")),
        (four_stage_figures, ("build_four_stage_plot_inputs",)),
        (gaussian_controls, ("run_gaussian_control_analysis",)),
    ):
        for name in names:
            monkeypatch.setattr(module, name, forbidden)
    original_hash = contracts.file_sha256
    def compact_hash(path):
        if Path(path).suffix in {".pt", ".npy", ".safetensors", ".parquet"}:
            forbidden()
        return original_hash(path)
    monkeypatch.setattr(contracts, "file_sha256", compact_hash)
    scientific_before = {name: digest for name, digest in _snapshot(bundle).items()
                         if name.startswith("plot_data/") or name in {"run_config.json", "summary.json", "audit.json"}}
    inherited = contracts.saved_plot_configuration(bundle, requested={}, portable=True)
    assert inherited["num_loss_seeds"] == 128
    contracts.render_saved_paper(bundle, expected_config=inherited, formats=("png", "pdf"))
    assert {name: _snapshot(bundle)[name] for name in scientific_before} == scientific_before
    with pytest.raises(TheoryError, match="Explicit plot settings conflict"):
        contracts.saved_plot_configuration(bundle, requested={"num_loss_seeds": 64},
                                             explicit_keys={"num_loss_seeds"}, portable=True)


@pytest.mark.parametrize("status", ["unavailable", "not_applicable"])
def test_required_missing_scalar_is_distinct_from_mathematical_inapplicability(tmp_path, monkeypatch, status):
    bundle = tmp_path / "paper"
    _old_bundle(bundle)
    _small_exports(monkeypatch)
    summary = read_json(bundle / "summary.json")
    summary["figures"]["terminal_bound_coverage"].update(status=status, reason="explicit saved scope reason")
    atomic_write_json(bundle / "summary.json", summary)
    if status == "unavailable":
        with pytest.raises(TheoryError, match=r"terminal_bound_coverage.*explicit saved scope reason.*--recompute-experiments"):
            contracts.render_saved_paper(bundle, formats=("png", "pdf"))
    else:
        manifest = contracts.render_saved_paper(bundle, formats=("png", "pdf"))
        entry = next(row for row in manifest["figures"] if row["stem"] == "terminal_bound_coverage")
        assert entry["status"] == "not_applicable" and entry["outputs"] == {}
        assert (bundle / "main/terminal_bound_coverage.png").exists()
        ledger = read_json(bundle / "figure_retirement.json")
        assert any(row["old_path"] == "main/terminal_bound_coverage.png"
                   and row["decision"] == "preserved_missing_replacement_pair" for row in ledger["records"])


@pytest.mark.parametrize("category", ["main", "diagnostics"])
def test_only_explicitly_owned_retired_locations_are_retired(tmp_path, category):
    owned = {}
    for stem in RETIRED_RENDER_STEMS:
        for suffix in ("png", "pdf"):
            path = tmp_path / category / f"{stem}.{suffix}"
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(b"previous renderer image")
            owned[path.relative_to(tmp_path).as_posix()] = file_sha256(path)
    unrelated = tmp_path / category / "branch_gap_other.png"
    unrelated.write_bytes(b"unrelated")
    owned[unrelated.relative_to(tmp_path).as_posix()] = file_sha256(unrelated)
    atomic_write_json(tmp_path / "figure_manifest.json", {"complete": True, "files": {}})
    ledger = contracts.retire_obsolete_figures(tmp_path, {"files": owned})
    removed = [row for row in ledger["records"] if row["status"] == "removed"]
    assert len(removed) == 8
    assert all(row["old_path"].startswith(category + "/") for row in removed)
    assert unrelated.read_bytes() == b"unrelated"


def test_peak_descriptive_reduction_keeps_unresolved_mass_and_original_histogram():
    import pandas as pd
    from utils.experiments.theory.paper_reduce import _save_peak_descriptive_summary

    frame = pd.DataFrame({"group": ["SSCD > 0.75"] * 4, "step_index": [0, 1, 2, 3],
                          "fraction": [.1, .1, .4, .2]})
    before = frame.copy(deep=True)
    figures = {"branch_gap_peak_step": {"status": "available", "shape_group_counts": {
        "SSCD > 0.75": {"samples": 10, "resolved_peak_count": 8, "unassigned_weight_fraction": .2,
                        "status_counts": {"resolved_peak": 8, "flat": 1, "incomplete": 1},
                        "multiple_exact_peak_count": 2, "multiple_near_peak_count": 3}}}}
    auxiliary = {}
    _save_peak_descriptive_summary({"branch_gap_peak_step": frame}, figures, auxiliary)
    saved = auxiliary["trajectory_peak_summary"].iloc[0]
    assert saved.resolved_weight_fraction == pytest.approx(.8)
    assert saved.unassigned_weight_fraction == .2 and saved.mass_at_initialization == .1
    assert saved.median_resolved_peak_step == 2
    assert saved.q25_resolved_peak_step == 1 and saved.q75_resolved_peak_step == 2
    assert saved.multiple_exact_peak_count == 2
    assert "Resolved individual peaks only" in saved.quantile_population
    assert frame.equals(before)


def test_unselected_missing_prompt_curve_does_not_block_selected_publication(tmp_path, monkeypatch):
    bundle = compact_fixture(tmp_path / "old", previous_registry=True, saved_grouped_coverage=True,
                             saved_zero_baselines=True, saved_prompt_reference_curves=True, saved_reference_gap_curve=True,
                             saved_guidance_fit=True, saved_reference_variation=True)
    assert "branch_gap_per_prompt" not in read_json(bundle / "summary.json")["plot_data"]
    _assert_unselected_inputs_remain_unneeded(bundle, monkeypatch)


def test_previous_curated_registry_migrates_without_drawing_unselected_prompt_curves(tmp_path, monkeypatch):
    bundle = compact_fixture(tmp_path / "curated")
    atomic_write_json(bundle / "registry.json", {"version": "four-stage-paper-curation-19", "figures": measurement_registry()})
    _small_exports(monkeypatch)
    manifest = contracts.render_saved_paper(bundle, formats=("png", "pdf"))
    assert manifest["requested_counts"] == {"main": 6, "appendix": 0, "diagnostics": 0, "timestep_figures": 0}
    assert "appendix/branch_gap_per_prompt.png" not in manifest["files"]
    assert "branch_gap_per_prompt" in read_json(bundle / "summary.json")["plot_data"]
    assert read_json(bundle / "registry.json")["version"] == REGISTRY_VERSION


def test_legacy_pooled_terminal_coverage_cannot_masquerade_as_grouped_saved_input(tmp_path, monkeypatch):
    """Old pooled ECDFs cannot reconstruct per-seed SSCD membership in plot mode."""
    import pandas as pd
    from utils.experiments.theory import four_stage_figures

    bundle = compact_fixture(tmp_path / "pooled_terminal", terminal=True)
    stem = "terminal_bound_coverage"
    relative = f"plot_data/{stem}.csv"
    # Keep a valid table hash/schema and current scientific receipt, isolating
    # the missing group/population contract from unrelated version failures.
    pooled = pd.DataFrame([
        {"distribution": distribution, "value": value, "cdf": fraction}
        for distribution in ("actual", "observable", "reference")
        for value, fraction in ((.2, .25), (1., .75), (4., 1.))
    ])
    specification = contracts.write_plot_table(pooled, bundle / relative)
    summary = read_json(bundle / "summary.json")
    summary["plot_data"][stem] = {"path": relative, **specification}
    summary["numerical_files"][relative] = specification["sha256"]
    summary["figures"][stem] = {
        "status": "available", "reason": None,
        "terminal_scope": "original_clean_terminal_theorem",
        "counts": {"rows": 4, "prompts": 2, "seeds": 2},
        "zero_mass": {name: 0. for name in ("actual", "observable", "reference")},
    }
    atomic_write_json(bundle / "summary.json", summary)
    old_exports = {}
    for suffix in ("png", "pdf"):
        name = f"main/{stem}.{suffix}"
        path = bundle / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"previous reviewed pooled coverage export")
        old_exports[name] = file_sha256(path)
    (bundle / "figure_captions.md").write_text("Previously reviewed pooled coverage caption\n")
    old_exports["figure_captions.md"] = file_sha256(bundle / "figure_captions.md")
    atomic_write_json(bundle / "figure_manifest.json", {"complete": True, "files": old_exports})
    before = _snapshot(bundle)
    expected_command = contracts.recompute_command(read_json(bundle / "run_config.json"))

    def forbidden(*args, **kwargs):
        raise AssertionError("missing grouped terminal scalars triggered drawing, reduction, or publication")

    monkeypatch.setattr(paper_plotting, "_draw", forbidden)
    monkeypatch.setattr(four_stage_figures, "build_four_stage_plot_inputs", forbidden)
    monkeypatch.setattr(contracts, "staged_publication", forbidden)
    with pytest.raises(TheoryError, match="Missing required paper columns for terminal_bound_coverage") as captured:
        contracts.render_saved_paper(bundle, formats=("png", "pdf"))
    message = str(captured.value)
    for field in ("group", "denominator_weight", "eligible_count", "prompt_count"):
        assert field in message
    assert message.endswith("Run " + expected_command)
    assert "--recompute-experiments" in expected_command
    assert "--model sdv1 --scheduler ddim" in expected_command
    assert _snapshot(bundle) == before
    assert not list(bundle.parent.glob(".pooled_terminal.stage-*"))
    assert not list(bundle.parent.glob(".pooled_terminal.backup-*"))


def test_missing_retired_zero_measurements_do_not_block_saved_plotting(tmp_path, monkeypatch):
    from utils.experiments.theory import four_stage_figures

    bundle = compact_fixture(tmp_path / "old_zero", previous_registry=True,
                             saved_prompt_curve=True, saved_grouped_coverage=True,
                             saved_prompt_reference_curves=True, saved_reference_gap_curve=True, saved_guidance_fit=True, saved_reference_variation=True)
    zero_stems = {entry["stem"] for entry in _ZERO_BASELINE_FIGURES}
    assert zero_stems.isdisjoint(read_json(bundle / "summary.json")["plot_data"])
    before = _snapshot(bundle)
    _small_exports(monkeypatch)
    def forbidden(*args, **kwargs):
        raise AssertionError("Missing retired zero norms triggered scientific recomputation")
    monkeypatch.setattr(four_stage_figures, "build_four_stage_plot_inputs", forbidden)
    manifest = contracts.render_saved_paper(bundle, formats=("png", "pdf"))
    assert manifest["requested_counts"] == {"main": 6, "appendix": 0, "diagnostics": 0, "timestep_figures": 0}
    assert zero_stems.isdisjoint(entry["stem"] for entry in manifest["figures"])
    assert {name: _snapshot(bundle)[name] for name in before} == before
    assert not list(bundle.parent.glob(".old_zero.stage-*"))


def test_unselected_missing_prompt_reference_curves_do_not_trigger_analysis(tmp_path, monkeypatch):
    bundle = compact_fixture(tmp_path / "old_prompt_references", previous_registry=True,
                             saved_prompt_curve=True, saved_grouped_coverage=True,
                             saved_zero_baselines=True, saved_prompt_reference_curves=False,
                             saved_reference_gap_curve=True, saved_guidance_fit=True, saved_reference_variation=True)
    _assert_unselected_inputs_remain_unneeded(bundle, monkeypatch)


@pytest.mark.parametrize("stem", ["conditional_reference_error_per_prompt",
                                  "unconditional_reference_error_per_prompt",
                                  "target_probability_per_prompt",
                                  "reference_branch_gap_per_prompt"])
def test_unselected_prompt_reference_columns_are_not_required_for_publication(tmp_path, monkeypatch, stem):
    import pandas as pd

    bundle = compact_fixture(tmp_path / "missing_prompt_values")
    summary = read_json(bundle / "summary.json")
    entry = next(entry for entry in measurement_registry() if entry["stem"] == stem)
    path = bundle / entry["source_table"]
    wrong = pd.read_csv(path).rename(columns={entry["value_column"]: "mean_gap_rmse"})
    specification = contracts.write_plot_table(wrong, path)
    summary["plot_data"][stem] = {"path": entry["source_table"], **specification}
    summary["numerical_files"][entry["source_table"]] = specification["sha256"]
    atomic_write_json(bundle / "summary.json", summary)
    _assert_unselected_inputs_remain_unneeded(bundle, monkeypatch)



def test_unselected_missing_reference_gap_does_not_trigger_analysis(tmp_path, monkeypatch):
    bundle = compact_fixture(tmp_path / "old_reference_gap", previous_registry=True,
                             saved_prompt_curve=True, saved_grouped_coverage=True,
                             saved_zero_baselines=True, saved_prompt_reference_curves=True,
                             saved_reference_gap_curve=False, saved_guidance_fit=True, saved_reference_variation=True)
    assert "reference_branch_gap_per_prompt" not in read_json(bundle / "summary.json")["plot_data"]
    _assert_unselected_inputs_remain_unneeded(bundle, monkeypatch)


@pytest.mark.parametrize("stem", ["initial_unconditional_mean_concentration_zero",
                                  "unconditional_reference_convergence_zero"])
@pytest.mark.parametrize("disposition", ["unowned", "modified"])
def test_retired_zero_exports_preserve_unowned_or_modified_files_and_all_saved_scalars(
    tmp_path, monkeypatch, stem, disposition
):
    bundle = tmp_path / "zero_retirement"
    previous = _old_bundle(bundle)
    relative = f"appendix/{stem}.png"
    if disposition == "unowned":
        del previous["files"][relative]
    else:
        (bundle / relative).write_bytes(b"user-modified zero-baseline figure")
    atomic_write_json(bundle / "figure_manifest.json", previous)
    image_before = (bundle / relative).read_bytes()
    protected = {name: value for name, value in _snapshot(bundle).items()
                 if name.startswith(("plot_data/", "audit_data/"))
                 or name in {"run_config.json", "summary.json", "audit.json"}}
    _small_exports(monkeypatch)
    manifest = contracts.render_saved_paper(bundle, formats=("png", "pdf"))
    assert (bundle / relative).read_bytes() == image_before
    assert not (bundle / f"appendix/{stem}.pdf").exists()
    assert {name: _snapshot(bundle)[name] for name in protected} == protected
    assert not any(row["stem"] == stem for row in manifest["figures"])
    records = read_json(bundle / "figure_retirement.json")["records"]
    assert any(row["old_path"] == relative and row["decision"] == "preserved_" + disposition
               for row in records)


def test_missing_guidance_fit_requires_analysis_and_never_reuses_injection_plot(tmp_path, monkeypatch):
    from utils.experiments.theory import four_stage_figures

    bundle = compact_fixture(tmp_path / "missing_guidance_fit", previous_registry=True,
                             saved_prompt_curve=True, saved_grouped_coverage=True,
                             saved_prompt_reference_curves=True, saved_reference_gap_curve=True,
                             saved_guidance_fit=False, saved_reference_variation=True)
    before = _snapshot(bundle)
    expected_command = contracts.recompute_command(read_json(bundle / "run_config.json"))

    def forbidden(*args, **kwargs):
        raise AssertionError("Missing guidance-fit scalars triggered rendering, fitting, or publication")

    monkeypatch.setattr(paper_plotting, "_draw", forbidden)
    monkeypatch.setattr(four_stage_figures, "build_four_stage_plot_inputs", forbidden)
    monkeypatch.setattr(contracts, "staged_publication", forbidden)
    with pytest.raises(TheoryError, match="corollary3_guidance_scale_vs_loss") as captured:
        contracts.render_saved_paper(bundle, formats=("png", "pdf"))
    assert str(captured.value).endswith("Run " + expected_command)
    assert "--recompute-experiments" in expected_command
    assert _snapshot(bundle) == before
    assert not list(bundle.parent.glob(".missing_guidance_fit.stage-*"))
    assert not list(bundle.parent.glob(".missing_guidance_fit.backup-*"))


def test_unselected_missing_reference_variation_does_not_trigger_reduction(tmp_path, monkeypatch):
    bundle = compact_fixture(tmp_path / "missing_reference_variation", previous_registry=True,
                             saved_prompt_curve=True, saved_grouped_coverage=True,
                             saved_prompt_reference_curves=True, saved_reference_gap_curve=True,
                             saved_guidance_fit=True, saved_reference_variation=False)
    assert "reference_variation_per_prompt" not in read_json(bundle / "summary.json")["plot_data"]
    _assert_unselected_inputs_remain_unneeded(bundle, monkeypatch)


def _assert_unselected_inputs_remain_unneeded(bundle, monkeypatch):
    from utils.experiments.theory import four_stage_figures

    protected = {name: digest for name, digest in _snapshot(bundle).items()
                 if name.startswith(("plot_data/", "audit_data/"))
                 or name in {"run_config.json", "summary.json", "audit.json"}}
    _small_exports(monkeypatch)
    def forbidden(*args, **kwargs):
        raise AssertionError("Unselected compact inputs triggered scientific reduction")
    monkeypatch.setattr(four_stage_figures, "build_four_stage_plot_inputs", forbidden)
    manifest = contracts.render_saved_paper(bundle, formats=("png", "pdf"))
    assert len(manifest["figures"]) == 6
    assert not manifest.get("timestep_figures", [])
    assert {name: _snapshot(bundle)[name] for name in protected} == protected


@pytest.mark.parametrize("disposition", ["unowned", "modified"])
def test_old_timestep_exports_retire_only_with_matching_ownership(tmp_path, monkeypatch, disposition):
    bundle = tmp_path / "old_steps"
    previous = _old_bundle(bundle)
    relative = "appendix/posterior_feedback_condition_margin/step_000.png"
    if disposition == "unowned":
        del previous["files"][relative]
    else:
        (bundle / relative).write_bytes(b"user-edited step image")
    atomic_write_json(bundle / "figure_manifest.json", previous)
    protected = (bundle / relative).read_bytes()
    unknown = bundle / "appendix/posterior_feedback_condition_margin/notes.png"
    unknown.write_bytes(b"user image in a historical timestep folder")
    _small_exports(monkeypatch)
    manifest = contracts.render_saved_paper(bundle, formats=("png", "pdf"))
    assert (bundle / relative).read_bytes() == protected
    assert unknown.read_bytes() == b"user image in a historical timestep folder"
    assert not (bundle / "appendix/posterior_feedback_condition_margin/step_000.pdf").exists()
    assert not manifest.get("timestep_figures", [])
    assert len(manifest["figures"]) == 6
    rows = {row["old_path"]: row for row in read_json(bundle / "figure_retirement.json")["records"]}
    assert rows[relative]["decision"] == "preserved_" + disposition
    assert unknown.relative_to(bundle).as_posix() not in rows
    first = _snapshot(bundle)
    contracts.render_saved_paper(bundle, formats=("png", "pdf"))
    assert _snapshot(bundle) == first


def test_pdf_only_publication_preserves_all_pngs_and_scientific_inputs(tmp_path, monkeypatch):
    bundle = tmp_path / "paper"
    previous = _old_bundle(bundle)
    # Include both an owned current PNG and an unowned current PNG. Neither is
    # an output or retirement candidate during the PDF-only publication.
    for index, entry in enumerate(paper_registry()[:2]):
        name = entry["outputs"]["png"]
        path = bundle / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"preserved current PNG")
        if index == 0:
            previous["files"][name] = file_sha256(path)
    atomic_write_json(bundle / "figure_manifest.json", previous)
    pngs = {path.relative_to(bundle).as_posix(): (file_sha256(path), path.stat().st_mtime_ns)
            for path in bundle.rglob("*.png")}
    immutable = {name: value for name, value in _snapshot(bundle).items()
                 if name not in {"registry.json", "figure_manifest.json", "figure_captions.md"}
                 and not name.endswith((".png", ".pdf"))}
    _small_exports(monkeypatch)
    manifest = contracts.render_saved_paper(bundle, formats=("pdf",))
    assert manifest["export_formats"] == ["pdf"]
    assert len(manifest["figures"]) == 6 and len(manifest["files"]) == 7
    expected = {entry["outputs"]["pdf"] for entry in paper_registry()}
    assert set(manifest["files"]) == expected | {"figure_captions.md"}
    for entry in manifest["figures"]:
        assert set(entry["outputs"]) == set(entry["requested_outputs"]) == {"pdf"}
        assert entry["output_hashes"] == {"pdf": file_sha256(bundle / entry["outputs"]["pdf"])}
        assert paper_plotting.paper_style.is_selected(entry)
    assert {path.relative_to(bundle).as_posix(): (file_sha256(path), path.stat().st_mtime_ns)
            for path in bundle.rglob("*.png")} == pngs
    assert {name: _snapshot(bundle)[name] for name in immutable} == immutable
    assert {name for name in previous["files"] if name.endswith(".png")} <= set(manifest["preserved_files"])
    ledger = read_json(bundle / "figure_retirement.json")
    assert all(not row["old_path"].endswith(".png") for row in ledger["records"])
    old_pdfs = {name for name in previous["files"] if name.endswith(".pdf")}
    assert all(not (bundle / name).exists() for name in old_pdfs)
    before = _snapshot(bundle)
    contracts.render_saved_paper(bundle, formats=("pdf",))
    assert _snapshot(bundle) == before


def test_pdf_only_export_failure_preserves_previous_bundle(tmp_path, monkeypatch):
    bundle = tmp_path / "paper"
    _old_bundle(bundle)
    before = _snapshot(bundle)
    _small_exports(monkeypatch, fail_pdf=True)
    with pytest.raises(RuntimeError, match="injected second-format"):
        contracts.render_saved_paper(bundle, formats=("pdf",))
    assert _snapshot(bundle) == before
    assert not list(bundle.parent.glob(".paper.stage-*"))
    assert not list(bundle.parent.glob(".paper.backup-*"))
