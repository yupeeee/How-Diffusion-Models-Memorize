"""External PDF publication regressions; authored without executing the suite."""

from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from tests.test_theory_paper_plotting import compact_fixture
from utils.common.io import atomic_write_json, file_sha256, read_json
from utils.experiments.theory import paper_contracts as contracts
from utils.experiments.theory import paper_plotting as plotting
from utils.experiments.theory.contracts import TheoryError
from utils.experiments.theory.paper_registry import paper_registry


RUN = "sdv1_ddim_g7.5_T50_N2"
ROLE = "experiment_S0_N2"
PUBLICATION_METADATA = {"figure_manifest.json", "figure_captions.md", "registry.json"}


def _snapshot(directory):
    return {
        path.relative_to(directory).as_posix(): (file_sha256(path), path.stat().st_mtime_ns)
        for path in Path(directory).rglob("*")
        if path.is_file()
    }


def _immutable_snapshot(directory):
    return {name: receipt for name, receipt in _snapshot(directory).items()
            if name not in PUBLICATION_METADATA}


@pytest.fixture
def publication_case(tmp_path, monkeypatch):
    """Exercise real scalar validation/transactions without drawing or encoding images."""
    project = tmp_path / "project"
    bundle = compact_fixture(project / "outputs" / RUN / "theory" / ROLE)
    destination = project / "figures" / RUN / "theory" / ROLE
    state = SimpleNamespace(project=project, bundle=bundle, destination=destination,
                            payload=b"first render", drawn=[], published=[], closed=[])

    def draw(entry, _frame, _metadata, _config):
        # Logical routes remain intact until after the selected-style dispatch.
        assert plotting.paper_style.is_selected(entry)
        assert entry["outputs"] == {"pdf": "figures/" + Path(entry["outputs"]["pdf"]).name}
        state.drawn.append(entry["stem"])
        return object(), {"synthetic_publication_fixture": True}

    def publish(output, figures, *, progress=None, export_options=None, formats=("png", "pdf")):
        assert tuple(formats) == ("pdf",)
        for _figure, names in figures:
            assert set(names) == {"pdf"}
            relative = names["pdf"]
            assert Path(relative).name == relative
            target = Path(output) / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if progress is not None:
                progress(relative, "saving")
            target.write_bytes(state.payload + b":" + relative.encode())
            state.published.append(relative)
            if progress is not None:
                progress(relative, "saved")

    monkeypatch.setattr(plotting, "_draw", draw)
    monkeypatch.setattr(plotting, "_finalize_presentations", lambda *_args: {})
    monkeypatch.setattr(plotting, "publish_figures", publish)
    monkeypatch.setattr(plotting.plt, "close", state.closed.append)
    return state


def test_default_publication_writes_six_external_pdfs_and_preserves_scalar_bundle(publication_case):
    case = publication_case
    # Historical cached images remain untouched, including an unowned PNG.
    old_png = case.bundle / "figures/initial_loss_recovery.png"
    old_pdf = case.bundle / "figures/initial_loss_recovery.pdf"
    old_png.parent.mkdir()
    old_png.write_bytes(b"preserved cached PNG")
    old_pdf.write_bytes(b"preserved cached PDF")
    before = _immutable_snapshot(case.bundle)

    manifest = contracts.render_saved_paper(case.bundle)

    expected = {Path(entry["outputs"]["pdf"]).name for entry in paper_registry()}
    assert len(expected) == 6
    assert set(_snapshot(case.destination)) == expected
    assert not list(case.destination.rglob("*.png"))
    assert _immutable_snapshot(case.bundle) == before
    assert manifest["export_formats"] == ["pdf"]
    assert manifest["publication"] == {
        "directory": str(case.destination),
        "files": {name: file_sha256(case.destination / name) for name in expected},
        "preserved_files": {},
        "paths_relative_to": "publication.directory",
        "metadata_location": "scientific bundle; publication contains figures only",
    }
    assert set(manifest["files"]) == {"figure_captions.md"}
    assert len(manifest["figures"]) == len(case.drawn) == len(case.closed) == 6
    assert manifest["timestep_figures"] == []
    for entry in manifest["figures"]:
        assert set(entry["outputs"]) == set(entry["requested_outputs"]) == {"pdf"}
        assert plotting.paper_style.is_selected(entry)
        name = Path(entry["outputs"]["pdf"]).name
        assert entry["publication_outputs"] == {"pdf": name}
        assert entry["output_hashes"] == {"pdf": manifest["publication"]["files"][name]}
    assert read_json(case.bundle / "figure_manifest.json")["publication"] == manifest["publication"]
    assert not (case.bundle / "figure_retirement.json").exists()
    assert not (case.bundle / "archive").exists()


@pytest.mark.parametrize("conflict", ["unowned", "modified_owned"])
def test_external_conflict_preserves_both_trees(publication_case, conflict):
    case = publication_case
    if conflict == "modified_owned":
        contracts.render_saved_paper(case.bundle)
    else:
        case.destination.mkdir(parents=True)
    target = case.destination / "initial_loss_recovery.pdf"
    target.write_bytes(b"user-owned or user-edited external PDF")
    before_bundle = _snapshot(case.bundle)
    before_external = _snapshot(case.destination)
    published_before = len(case.published)

    with pytest.raises(TheoryError, match="unowned|modified|ownership"):
        contracts.render_saved_paper(case.bundle)

    assert len(case.published) == published_before
    assert _snapshot(case.bundle) == before_bundle
    assert _snapshot(case.destination) == before_external


def test_registry_metadata_failure_rolls_back_external_pdfs_and_bundle(publication_case, monkeypatch):
    case = publication_case
    contracts.render_saved_paper(case.bundle)
    before_bundle = _snapshot(case.bundle)
    before_external = _snapshot(case.destination)
    case.payload = b"replacement render that must never become public"
    published_before = len(case.published)
    original = contracts.atomic_write_json

    def fail_registry(path, value, *args, **kwargs):
        if Path(path).name == "registry.json":
            raise OSError("injected publication metadata failure")
        return original(path, value, *args, **kwargs)

    monkeypatch.setattr(contracts, "atomic_write_json", fail_registry)
    with pytest.raises(OSError, match="publication metadata failure"):
        contracts.render_saved_paper(case.bundle)

    assert len(case.published) == published_before + 6
    assert _snapshot(case.bundle) == before_bundle
    assert _snapshot(case.destination) == before_external
    for parent in (case.bundle.parent, case.destination.parent):
        assert not list(parent.glob(".*.stage-*"))
        assert not list(parent.glob(".*.backup-*"))
        assert not list(parent.glob(".*.transaction.json"))


@pytest.mark.parametrize("failed_tree", ["figures", "bundle"])
def test_directory_install_failure_restores_both_previous_publications(publication_case, monkeypatch, failed_tree):
    case = publication_case
    contracts.render_saved_paper(case.bundle)
    before_bundle = _snapshot(case.bundle)
    before_external = _snapshot(case.destination)
    case.payload = b"replacement render that must roll back"
    replace = contracts.os.replace
    failed = []
    failed_destination = case.destination if failed_tree == "figures" else case.bundle

    def fail_external_install(source, destination):
        if Path(destination) == failed_destination and ".stage-" in Path(source).name:
            failed.append((Path(source), Path(destination)))
            raise OSError("injected external directory install failure")
        return replace(source, destination)

    monkeypatch.setattr(contracts.os, "replace", fail_external_install)
    with pytest.raises(OSError, match="external directory install failure"):
        contracts.render_saved_paper(case.bundle)

    assert len(failed) == 1
    assert _snapshot(case.bundle) == before_bundle
    assert _snapshot(case.destination) == before_external
    for parent in (case.bundle.parent, case.destination.parent):
        assert not list(parent.glob(".*.stage-*"))
        assert not list(parent.glob(".*.backup-*"))
        assert not list(parent.glob(".*.transaction.json"))


def test_portable_bundle_cli_uses_project_root_and_saved_run_identity(tmp_path, monkeypatch):
    from scripts import theory_validation

    project = tmp_path / "checkout"
    bundle = compact_fixture(tmp_path / "portable_copy" / "renamed_scalars")
    before = _snapshot(bundle)
    exports = []
    monkeypatch.setattr(theory_validation, "PROJECT_ROOT", project)
    monkeypatch.setattr(contracts, "render_saved_paper",
                        lambda source, **options: exports.append((source, options)))

    assert theory_validation.main(["--bundle", str(bundle), "--plot"]) == 0

    assert len(exports) == 1
    source, options = exports[0]
    assert source == bundle
    assert Path(options["figure_directory"]) == project / "figures" / RUN / "theory" / ROLE
    assert tuple(options.get("formats", ("pdf",))) == ("pdf",)
    assert options["expected_config"] == read_json(bundle / "run_config.json")["scientific_config"]
    assert _snapshot(bundle) == before
    assert not (tmp_path / "portable_copy" / "figures").exists()
    assert not project.exists()


@pytest.mark.parametrize("crash_point", ["after_external_backup", "after_external_install"])
def test_pending_destination_owner_blocks_another_bundle_until_owner_recovers(
    publication_case, tmp_path, crash_point,
):
    """A portable source cannot overwrite an abandoned canonical-source transaction."""
    case = publication_case
    contracts.render_saved_paper(case.bundle)
    other_bundle = compact_fixture(tmp_path / "portable" / "different_source")
    before_bundle = _snapshot(case.bundle)
    before_external = _snapshot(case.destination)
    before_other = _snapshot(other_bundle)
    published_before = len(case.published)

    # Construct the on-disk state left by a hard process exit, which ordinary
    # raised exceptions cannot model because the context manager rolls back.
    bundle_stage = case.bundle.parent / ("." + ROLE + ".stage-crash")
    bundle_backup = case.bundle.parent / ("." + ROLE + ".backup-crash")
    figure_stage = case.destination.parent / ("." + ROLE + ".stage-crash")
    figure_backup = case.destination.parent / ("." + ROLE + ".backup-crash")
    shutil.copytree(case.bundle, bundle_stage)
    shutil.copytree(case.destination, figure_stage)
    (figure_stage / "initial_loss_recovery.pdf").write_bytes(b"interrupted new PDF")
    case.destination.rename(figure_backup)
    if crash_point == "after_external_install":
        figure_stage.rename(case.destination)
    marker = case.bundle.parent / ("." + ROLE + ".transaction.json")
    atomic_write_json(marker, {
        "schema_version": 2, "phase": "installing", "archive_previous": False,
        "stage": bundle_stage.name, "backup": bundle_backup.name, "had_output": True,
        "publication": {"directory": str(case.destination), "stage": figure_stage.name,
                        "backup": figure_backup.name, "had_output": True},
    })
    _lock, pending = contracts._external_lock_paths(case.destination)
    atomic_write_json(pending, {
        "schema_version": 1, "directory": str(case.destination),
        "bundle": str(case.bundle), "transaction_marker": str(marker),
    })
    crashed_external = _snapshot(case.destination)
    pending_bytes = pending.read_bytes()

    with pytest.raises(TheoryError, match="unfinished transaction owned by"):
        contracts.render_saved_paper(other_bundle, figure_directory=case.destination)

    assert len(case.published) == published_before
    assert _snapshot(other_bundle) == before_other
    assert _snapshot(case.bundle) == before_bundle
    assert _snapshot(case.destination) == crashed_external
    assert pending.read_bytes() == pending_bytes
    assert marker.exists() and figure_backup.exists()

    with contracts.publication_lock(case.bundle, figure_directory=case.destination):
        assert _snapshot(case.bundle) == before_bundle
        assert _snapshot(case.destination) == before_external
        assert not marker.exists()

    assert not pending.exists()
    assert not bundle_stage.exists() and not bundle_backup.exists()
    assert not figure_stage.exists() and not figure_backup.exists()
    assert _snapshot(other_bundle) == before_other
