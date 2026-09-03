"""Offline tests for prompt-labeled held-out TV and N preview exports."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from PIL import Image

from scripts import compute_proximity as proximity_cli
from utils.common.io import (
    atomic_write_frame_csv,
    atomic_write_json,
    canonical_hash,
    file_sha256,
)
from utils.data import selection as selection_module
from utils.experiments import held_out as held_out_module
from utils.experiments.cache import GenerationPaths
from utils.experiments.held_out import (
    HeldOutNExportError,
    HeldOutNExportResult,
    HeldOutTVExportError,
    export_held_out_n_results,
    export_held_out_tv_results,
)


def _selection_row(
    index: str,
    source_row: int,
    *,
    label: str,
    included: bool,
    prompt: str,
) -> dict[str, object]:
    if label == "TV" and not included:
        status = "excluded_tv_target_unsupported"
        reason = "tv_selection_mean_sscd_lt_0_25"
        semantics = "unsupported_template_target"
    elif label == "TV":
        status = "included_tv_target_supported"
        reason = "tv_selection_mean_sscd_ge_0_25"
        semantics = "empirically_singleton_compatible_tv"
    elif label == "N" and not included:
        status = "excluded_n_target_unsupported"
        reason = "n_selection_mean_sscd_lt_0_25"
        semantics = "n_target_unsupported_under_frozen_reference_criterion"
    elif label == "N":
        status = "included_n_target_supported"
        reason = "n_selection_mean_sscd_ge_0_25"
        semantics = "n_target_supported_under_frozen_reference_criterion"
    else:
        status = "invalid_record"
        reason = "invalid_prompt_or_target"
        semantics = None
    selection_mean = 0.40 if included else 0.10
    validation_mean = 0.42 if included else 0.12
    return {
        "original_index": index,
        "record_id": f"sdv1-{index}",
        "source_row_number": source_row,
        "prompt_raw": prompt,
        "target_image_sha256": str(source_row + 1) * 64,
        "webster_overfit_type_normalized": label,
        "include_target_pair": included,
        "selection_status": status,
        "selection_reason": reason,
        "target_semantics": semantics,
        "threshold": 0.25,
        "selection_mean_sscd": selection_mean,
        "reference_validation_mean_sscd": validation_mean,
    }


def _project(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    seeds: tuple[int, ...] = tuple(range(20, 40)),
    rows: list[dict[str, object]] | None = None,
    analysis_indices: tuple[str, ...] | None = None,
) -> SimpleNamespace:
    root = tmp_path.resolve()
    run_name = f"sdv1_test_N{len(seeds)}"
    role = "experiment" if seeds[0] == 0 else "reference"
    namespace = f"{role}_S{seeds[0]}_N{len(seeds)}"
    paths = GenerationPaths(root / "logs" / run_name / namespace)
    paths.create()
    science = {
        "model_cli_name": "sdv1",
        "seeds": list(seeds),
        "num_seeds": len(seeds),
    }
    generation_hash = canonical_hash(science)
    atomic_write_json(
        paths.run_config,
        {
            "scientific_config": science,
            "scientific_config_hash": generation_hash,
        },
    )
    if rows is None:
        rows = [
            _selection_row(
                "100",
                0,
                label="TV",
                included=False,
                prompt="held-out <TV> prompt\nwith unicode café",
            ),
            _selection_row(
                "101",
                1,
                label="TV",
                included=True,
                prompt="included TV prompt",
            ),
            _selection_row(
                "102",
                2,
                label="N",
                included=False,
                prompt="held-out <N> target-unsupported prompt",
            ),
        ]
    for row in rows:
        index = str(row["original_index"])
        preview = paths.image_path(index)
        preview.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8), (int(index) % 255, 20, 30)).save(preview)
        atomic_write_json(
            paths.record_path(index),
            {
                "original_index": index,
                "record_id": row["record_id"],
                "source_row_number": row["source_row_number"],
                "prompt_raw": row["prompt_raw"],
                "target_image_sha256": row["target_image_sha256"],
                "scientific_config_hash": generation_hash,
                "seeds": list(seeds),
                "preview_image_path": f"image/{index}.png",
                "preview_image_sha256": file_sha256(preview),
            },
        )

    selection_hash = "a" * 64
    selection = SimpleNamespace(
        root=root,
        model_name="sdv1",
        sha256=selection_hash,
        frame=pd.DataFrame(rows),
    )
    monkeypatch.setattr(
        selection_module,
        "load_target_pair_selection",
        lambda *_args, **_kwargs: selection,
    )

    output = (
        root
        / "outputs"
        / run_name
        / "proximity"
        / namespace
    )
    output.mkdir(parents=True)
    atomic_write_json(
        output / "run_config.json",
        {
            "generation_scientific_config_hash": generation_hash,
            "selection_hash": selection_hash,
        },
    )
    atomic_write_json(
        output / "summary.json",
        {
            "complete": True,
            "generation_scientific_config_hash": generation_hash,
            "selection_hash": selection_hash,
        },
    )
    atomic_write_frame_csv(
        pd.DataFrame(
            {
                "original_index": (
                    [str(row["original_index"]) for row in rows]
                    if analysis_indices is None
                    else list(analysis_indices)
                ),
                "seed": [seeds[0]]
                * (len(rows) if analysis_indices is None else len(analysis_indices)),
            }
        ),
        output / "paired_all.csv",
    )
    selected_indices = [
        str(row["original_index"])
        for row in rows
        if bool(row["include_target_pair"])
    ]
    atomic_write_frame_csv(
        pd.DataFrame(
            {
                "original_index": selected_indices,
                "seed": [seeds[0]] * len(selected_indices),
            }
        ),
        output / "paired_selected.csv",
    )
    return SimpleNamespace(
        root=root,
        paths=paths,
        output=output,
        rows=rows,
        seeds=seeds,
    )


def _export(project: SimpleNamespace):
    return export_held_out_tv_results(
        project.root,
        model_name="sdv1",
        generation_run=project.paths.run_directory,
        output_directory=project.output,
    )


def _export_n(project: SimpleNamespace):
    return export_held_out_n_results(
        project.root,
        model_name="sdv1",
        generation_run=project.paths.run_directory,
        output_directory=project.output,
    )


def test_export_is_prompt_level_and_keeps_exact_prompt_traceability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = _project(tmp_path, monkeypatch)
    source = project.paths.image_path("100")

    result = _export(project)

    assert result.prompt_count == 1
    assert result.total_prompt_count == 1
    assert result.missing_prompt_count == 0
    assert result.reused is False
    copied = result.directory / "prompts/100/generated.png"
    prompt = result.directory / "prompts/100/prompt.txt"
    assert copied.read_bytes() == source.read_bytes()
    assert prompt.read_text(encoding="utf-8") == project.rows[0]["prompt_raw"]
    manifest = pd.read_csv(result.manifest_path, dtype={"original_index": str})
    assert manifest["original_index"].tolist() == ["100"]
    assert json.loads(manifest.at[0, "seeds"]) == list(range(20, 40))
    assert manifest.at[0, "prompt_raw"] == project.rows[0]["prompt_raw"]
    assert manifest.at[0, "selection_status"] == "excluded_tv_target_unsupported"
    assert manifest.at[0, "source_preview_path"] == (
        source.relative_to(project.root).as_posix()
    )
    config = json.loads(
        (result.directory / "config.json").read_text(encoding="utf-8")
    )
    assert config["generation_run_path"] == (
        project.paths.run_directory.relative_to(project.root).as_posix()
    )
    gallery = result.gallery_path.read_text(encoding="utf-8")
    assert "held-out &lt;TV&gt; prompt" in gallery
    assert "validation seed half" in gallery
    assert "included TV prompt" not in gallery
    assert "held-out &lt;N&gt; target-unsupported prompt" not in gallery


@pytest.mark.parametrize(
    "seeds",
    (tuple(range(20, 40)), tuple(range(3))),
)
def test_export_preserves_role_scoped_seed_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    seeds: tuple[int, ...],
) -> None:
    project = _project(tmp_path, monkeypatch, seeds=seeds)

    result = _export(project)

    config = json.loads((result.directory / "config.json").read_text(encoding="utf-8"))
    assert config["seeds"] == list(seeds)
    assert config["seed_start"] == seeds[0]
    assert config["num_seeds"] == len(seeds)
    assert result.directory.parent == project.output


def test_export_n_contains_only_excluded_n_and_is_diagnostic_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rows = [
        _selection_row(
            "100",
            0,
            label="TV",
            included=False,
            prompt="excluded TV prompt",
        ),
        _selection_row(
            "102",
            2,
            label="N",
            included=False,
            prompt="target-unsupported <N> prompt",
        ),
        _selection_row(
            "103",
            3,
            label="N",
            included=True,
            prompt="included N prompt",
        ),
    ]
    project = _project(tmp_path, monkeypatch, rows=rows)
    source = project.paths.image_path("102")
    scientific_paths = (
        project.paths.run_config,
        project.output / "run_config.json",
        project.output / "summary.json",
        project.output / "paired_all.csv",
        project.output / "paired_selected.csv",
    )
    before = {path: path.read_bytes() for path in scientific_paths}

    result = _export_n(project)

    assert isinstance(result, HeldOutNExportResult)
    assert result.directory == project.output / "held_out_n"
    assert result.prompt_count == 1
    assert result.total_prompt_count == 1
    assert result.missing_prompt_count == 0
    assert result.reused is False
    copied = result.directory / "prompts/102/generated.png"
    assert copied.read_bytes() == source.read_bytes()
    assert (
        result.directory / "prompts/102/prompt.txt"
    ).read_text(encoding="utf-8") == rows[1]["prompt_raw"]

    manifest = pd.read_csv(
        result.manifest_path,
        dtype={
            "original_index": str,
            "target_image_sha256": str,
        },
    )
    assert list(manifest.columns) == list(held_out_module.MANIFEST_COLUMNS)
    assert manifest["original_index"].tolist() == ["102"]
    assert manifest["webster_overfit_type_normalized"].tolist() == ["N"]
    assert manifest.at[0, "record_id"] == rows[1]["record_id"]
    assert manifest.at[0, "source_row_number"] == rows[1]["source_row_number"]
    assert manifest.at[0, "prompt_raw"] == rows[1]["prompt_raw"]
    assert manifest.at[0, "target_image_sha256"] == rows[1]["target_image_sha256"]
    assert manifest["selection_reason"].tolist() == [
        "n_selection_mean_sscd_lt_0_25"
    ]
    assert manifest.at[0, "selection_mean_sscd"] == pytest.approx(0.10)
    assert manifest.at[0, "reference_validation_mean_sscd"] == pytest.approx(
        0.12
    )
    assert manifest.at[0, "selection_hash"] == "a" * 64
    assert json.loads(manifest.at[0, "seeds"]) == list(range(20, 40))
    assert manifest.at[0, "source_preview_path"] == (
        source.relative_to(project.root).as_posix()
    )
    assert manifest.at[0, "preview_image_sha256"] == file_sha256(source)

    config = json.loads(
        (result.directory / "config.json").read_text(encoding="utf-8")
    )
    assert config["artifact"] == "held_out_n_generation_previews"
    assert config["frozen_held_out_n_prompt_count"] == 1
    assert config["exported_prompt_count"] == 1
    assert config["scientific_metric_input"] is False
    assert config["affects_selection_or_analysis"] is False
    summary = json.loads(
        (result.directory / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["complete"] is True
    assert summary["held_out_n_prompt_count"] == 1
    assert summary["copied_preview_count"] == 1

    gallery = result.gallery_path.read_text(encoding="utf-8")
    assert "Held-out N prompt generations" in gallery
    assert "target-unsupported &lt;N&gt; prompt" in gallery
    assert "excluded TV prompt" not in gallery
    assert "included N prompt" not in gallery
    assert "[Held-out N] Preview gallery" in capsys.readouterr().err
    assert {path: path.read_bytes() for path in scientific_paths} == before


def test_empty_held_out_n_set_still_publishes_full_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        _selection_row(
            "100",
            0,
            label="TV",
            included=False,
            prompt="excluded TV prompt",
        ),
        _selection_row(
            "103",
            3,
            label="N",
            included=True,
            prompt="included N prompt",
        ),
    ]
    project = _project(tmp_path, monkeypatch, rows=rows)

    result = _export_n(project)

    assert result.prompt_count == 0
    assert result.total_prompt_count == 0
    manifest = pd.read_csv(result.manifest_path)
    assert list(manifest.columns) == list(held_out_module.MANIFEST_COLUMNS)
    assert manifest.empty
    summary = json.loads(
        (result.directory / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["held_out_n_prompt_count"] == 0
    assert "0 prompt(s)" in result.gallery_path.read_text(encoding="utf-8")


def test_n_preview_error_uses_n_error_and_leaves_no_partial_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = _project(tmp_path, monkeypatch)
    marker = project.paths.record_path("102")
    values = json.loads(marker.read_text(encoding="utf-8"))
    values["preview_image_sha256"] = "f" * 64
    atomic_write_json(marker, values)

    with pytest.raises(HeldOutNExportError, match="preview hash differs"):
        _export_n(project)

    assert not (project.output / "held_out_n").exists()
    assert not list(project.output.glob(".held_out_n.*.tmp"))


def test_export_rerun_is_idempotent_and_detects_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = _project(tmp_path, monkeypatch)
    first = _export(project)
    first_bytes = (first.directory / "prompts/100/generated.png").read_bytes()

    second = _export(project)

    assert second.reused is True
    assert (second.directory / "prompts/100/generated.png").read_bytes() == first_bytes
    (second.directory / "prompts/100/prompt.txt").write_text(
        "tampered", encoding="utf-8"
    )
    with pytest.raises(HeldOutTVExportError, match="prompt text differs"):
        _export(project)


def test_hash_mismatch_leaves_no_partial_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = _project(tmp_path, monkeypatch)
    marker = project.paths.record_path("100")
    values = json.loads(marker.read_text(encoding="utf-8"))
    values["preview_image_sha256"] = "f" * 64
    atomic_write_json(marker, values)

    with pytest.raises(HeldOutTVExportError, match="preview hash differs"):
        _export(project)

    assert not (project.output / "held_out_tv").exists()
    assert not list(project.output.glob(".held_out_tv.*.tmp"))


def test_copy_race_is_detected_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = _project(tmp_path, monkeypatch)
    real_copy = held_out_module.atomic_copy

    def corrupting_copy(source: Path, destination: Path) -> None:
        real_copy(source, destination)
        destination.write_bytes(b"changed after the source hash check")

    monkeypatch.setattr(held_out_module, "atomic_copy", corrupting_copy)

    with pytest.raises(HeldOutTVExportError, match="copied.*preview differs"):
        _export(project)

    assert not (project.output / "held_out_tv").exists()
    assert not list(project.output.glob(".held_out_tv.*.tmp"))


def test_missing_current_run_prompts_are_counted_and_identified(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        _selection_row("100", 0, label="TV", included=False, prompt="present"),
        _selection_row("103", 3, label="TV", included=False, prompt="absent"),
    ]
    project = _project(
        tmp_path,
        monkeypatch,
        rows=rows,
        analysis_indices=("100",),
    )

    result = _export(project)

    assert result.prompt_count == 1
    assert result.total_prompt_count == 2
    assert result.missing_prompt_count == 1
    config = json.loads((result.directory / "config.json").read_text(encoding="utf-8"))
    assert config["not_present_in_current_run_indices"] == ["103"]
    summary = json.loads(
        (result.directory / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["not_present_in_current_run_count"] == 1


def test_empty_held_out_set_still_publishes_browsable_headers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        _selection_row("101", 1, label="TV", included=True, prompt="included TV prompt")
    ]
    project = _project(tmp_path, monkeypatch, rows=rows)

    result = _export(project)

    assert result.prompt_count == 0
    manifest = pd.read_csv(result.manifest_path)
    assert list(manifest.columns) == list(held_out_module.MANIFEST_COLUMNS)
    assert manifest.empty
    summary = json.loads(
        (result.directory / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["held_out_tv_prompt_count"] == 0
    assert "0 prompt(s)" in result.gallery_path.read_text(encoding="utf-8")


def test_proximity_cli_automatically_runs_both_exports_after_complete_analysis(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = Path("/tmp/output")
    generation = Path("/tmp/generation")
    summary = SimpleNamespace(
        exit_code=0,
        paths=SimpleNamespace(
            generation_run=generation,
            output_directory=output,
            summary_json=output / "summary.json",
        ),
    )
    calls: list[tuple[str, object, object]] = []
    monkeypatch.setattr(
        "utils.experiments.proximity.run_proximity",
        lambda *_args, **_kwargs: summary,
    )

    def fake_export(category: str):
        def export(root: object, **kwargs: object) -> SimpleNamespace:
            calls.append((category, root, kwargs))
            return SimpleNamespace(
                reused=False,
                prompt_count=7 if category == "TV" else 3,
                total_prompt_count=7 if category == "TV" else 3,
                missing_prompt_count=0,
                gallery_path=output
                / f"held_out_{category.casefold()}/gallery.html",
            )

        return export

    monkeypatch.setattr(
        held_out_module,
        "export_held_out_tv_results",
        fake_export("TV"),
    )
    monkeypatch.setattr(
        held_out_module,
        "export_held_out_n_results",
        fake_export("N"),
    )

    status = proximity_cli.main([])

    captured = capsys.readouterr()
    expected_kwargs = {
        "model_name": "sdv1",
        "generation_run": generation,
        "output_directory": output,
    }
    assert status == 0
    assert calls == [
        ("TV", proximity_cli.PROJECT_ROOT, expected_kwargs),
        ("N", proximity_cli.PROJECT_ROOT, expected_kwargs),
    ]
    assert "Saved 7/7 held-out TV prompt previews" in captured.out
    assert "Saved 3/3 held-out N prompt previews" in captured.out


@pytest.mark.parametrize(
    ("failing_category", "error_type"),
    (
        ("TV", HeldOutTVExportError),
        ("N", HeldOutNExportError),
    ),
)
def test_proximity_cli_attempts_both_exports_and_reports_category_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failing_category: str,
    error_type: type[RuntimeError],
) -> None:
    output = Path("/tmp/output")
    summary = SimpleNamespace(
        exit_code=0,
        paths=SimpleNamespace(
            generation_run=Path("/tmp/generation"),
            output_directory=output,
            summary_json=output / "summary.json",
        ),
    )
    calls: list[str] = []
    monkeypatch.setattr(
        "utils.experiments.proximity.run_proximity",
        lambda *_args, **_kwargs: summary,
    )

    def fake_export(category: str):
        def export(*_args: object, **_kwargs: object) -> SimpleNamespace:
            calls.append(category)
            if category == failing_category:
                raise error_type("synthetic gallery failure")
            return SimpleNamespace(
                reused=True,
                prompt_count=1,
                total_prompt_count=2,
                missing_prompt_count=1,
                gallery_path=output
                / f"held_out_{category.casefold()}/gallery.html",
            )

        return export

    monkeypatch.setattr(
        held_out_module,
        "export_held_out_tv_results",
        fake_export("TV"),
    )
    monkeypatch.setattr(
        held_out_module,
        "export_held_out_n_results",
        fake_export("N"),
    )

    status = proximity_cli.main([])

    captured = capsys.readouterr()
    assert status == 1
    assert calls == ["TV", "N"]
    assert f"Summary: {output / 'summary.json'}" in captured.out
    assert (
        f"Held-out {failing_category} gallery failed after scientific "
        "proximity completed: synthetic gallery failure"
    ) in captured.err
    successful_category = "N" if failing_category == "TV" else "TV"
    assert f"held-out {successful_category} prompt previews" in captured.out
    assert (
        f"see held_out_{successful_category.casefold()}/config.json"
        in captured.err
    )
